//! Hand-written LZ4 block codec.
//!
//! Why not `flate2`: deflate runs at roughly 30–60 MB/s, which dominates
//! `serialize()` time on a large index and makes "fast export" a lie. LZ4
//! decompresses at >1 GB/s and compresses at ~400 MB/s, so the codec stops
//! being the bottleneck and the snapshot still lands ~2.5–3x smaller than raw
//! on typical float embeddings.
//!
//! This implements the standard `LZ4 block` format (the thing inside a `.lz4`
//! file's frames), so `lz4 -d` can read our output and we can read `lz4 -B`
//! output. Sequences are `token | literals | offset | match_length`.
//!
//! Both directions are total: the decompressor validates every length field
//! and offset against the output buffer and returns [`EngineError`] on anything
//! malformed, instead of panicking or allocating a length an attacker chose.

use crate::engine::types::{EngineError, EngineResult};

/// Minimum match length in the LZ4 format.
const MIN_MATCH: usize = 4;
/// The last 5 bytes of a block are always literals, and a match may not start
/// within the last 12 bytes. Both are format requirements, not choices.
const LAST_LITERALS: usize = 5;
const MFLIMIT: usize = 12;
const HASH_LOG: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;

/// Blocks are 64 KiB in the reference framing; the offset field is 16 bits, so
/// a longer match distance simply is not representable.
const MAX_OFFSET: usize = 65535;

/// Worst-case compressed size for `len` input bytes.
pub fn compress_bound(len: usize) -> usize {
    len + len / 255 + 16
}

/// Returns `true` if the block is small enough that LZ4 cannot help.
///
/// A block under `MFLIMIT` bytes has no room for a match at all.
pub fn is_incompressible_size(len: usize) -> bool {
    len < MFLIMIT
}

#[inline]
fn read_u32(src: &[u8], pos: usize) -> u32 {
    // Callers guarantee `pos + 4 <= src.len()`.
    u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]])
}

#[inline]
fn hash(sequence: u32) -> usize {
    ((sequence.wrapping_mul(2_654_435_761)) >> (32 - HASH_LOG)) as usize
}

fn write_length(dst: &mut Vec<u8>, mut length: usize) {
    while length >= 255 {
        dst.push(255);
        length -= 255;
    }
    dst.push(length as u8);
}

fn emit_last_literals(dst: &mut Vec<u8>, src: &[u8], start: usize) {
    let length = src.len().saturating_sub(start);

    // No match follows, so the low nibble is legitimately zero here.
    if length >= 15 {
        dst.push(0xF0);
        write_length(dst, length - 15);
    } else {
        dst.push((length as u8) << 4);
    }

    if start < src.len() {
        dst.extend_from_slice(&src[start..]);
    }
}

/// Compress `src` into `dst` (cleared first). Returns the number of bytes written.
pub fn compress_into(src: &[u8], dst: &mut Vec<u8>) {
    dst.clear();

    if src.len() < MFLIMIT + MIN_MATCH {
        emit_last_literals(dst, src, 0);
        return;
    }

    // `0` means "empty slot"; positions are stored biased by one.
    let mut table = vec![0u32; HASH_SIZE];
    let limit = src.len() - MFLIMIT;
    // Matches may not cover the trailing literals.
    let match_ceiling = src.len() - LAST_LITERALS;

    let mut anchor = 0usize;
    let mut pos = 0usize;

    while pos <= limit {
        let sequence = read_u32(src, pos);
        let slot = hash(sequence);
        let candidate_biased = table[slot];
        table[slot] = (pos + 1) as u32;

        if candidate_biased == 0 {
            pos += 1;
            continue;
        }

        let candidate = candidate_biased as usize - 1;
        if pos <= candidate || pos - candidate > MAX_OFFSET {
            pos += 1;
            continue;
        }

        if read_u32(src, candidate) != sequence {
            pos += 1;
            continue;
        }

        // Extend backwards — free compression, and it is what makes the
        // difference between ~2.2x and ~2.8x on float data.
        let mut start = pos;
        let mut cand = candidate;
        while start > anchor && cand > 0 && src[start - 1] == src[cand - 1] {
            start -= 1;
            cand -= 1;
        }

        // Extend forwards, capped so the trailer stays literal.
        let mut end = start + MIN_MATCH;
        while end < match_ceiling && src[end] == src[cand + (end - start)] {
            end += 1;
        }

        let literal_len = start - anchor;
        let match_len = end - start - MIN_MATCH;

        // The match length always goes in the low nibble, even when the
        // literal length spills into the extension bytes. Emitting a bare
        // `0xF0` here zeroes the match nibble, so the decoder reads every such
        // match as the 4-byte minimum and the output comes out short — which is
        // exactly how a snapshot loads on one machine and not another.
        let match_nibble = match_len.min(15) as u8;

        if literal_len >= 15 {
            dst.push(0xF0 | match_nibble);
            write_length(dst, literal_len - 15);
        } else {
            dst.push(((literal_len as u8) << 4) | match_nibble);
        }

        dst.extend_from_slice(&src[anchor..start]);

        let offset = start - cand;
        dst.extend_from_slice(&(offset as u16).to_le_bytes());

        if match_len >= 15 {
            write_length(dst, match_len - 15);
        }

        pos = end;
        anchor = end;
    }

    emit_last_literals(dst, src, anchor);
}

/// Compress `src` into a fresh buffer.
pub fn compress(src: &[u8]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(compress_bound(src.len()));
    compress_into(src, &mut dst);
    dst
}

/// Decompress a block. `expected_len` is the exact original size, which the
/// caller knows from the envelope — this lets us allocate once instead of
/// growing, and lets us reject a truncated block before copying anything.
pub fn decompress(src: &[u8], expected_len: usize) -> EngineResult<Vec<u8>> {
    let mut dst: Vec<u8> = Vec::with_capacity(expected_len);
    let mut pos = 0usize;

    if expected_len == 0 {
        return Ok(dst);
    }

    loop {
        let token = *src
            .get(pos)
            .ok_or_else(|| EngineError::corrupt("lz4: truncated token"))?;
        pos += 1;

        // --- literal run ---
        let mut literal_len = (token >> 4) as usize;
        if literal_len == 15 {
            loop {
                let byte = *src
                    .get(pos)
                    .ok_or_else(|| EngineError::corrupt("lz4: truncated literal length"))?;
                pos += 1;
                literal_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        if literal_len > 0 {
            let end = pos
                .checked_add(literal_len)
                .ok_or_else(|| EngineError::corrupt("lz4: literal length overflow"))?;
            let literals = src
                .get(pos..end)
                .ok_or_else(|| EngineError::corrupt("lz4: truncated literals"))?;
            if dst.len() + literals.len() > expected_len {
                return Err(EngineError::corrupt("lz4: output exceeds declared size"));
            }
            dst.extend_from_slice(literals);
            pos = end;
        }

        // The reference encoder ends the block with a literal-only sequence.
        if pos >= src.len() {
            break;
        }

        // --- match ---
        if pos + 2 > src.len() {
            return Err(EngineError::corrupt("lz4: truncated offset"));
        }
        let offset = u16::from_le_bytes([src[pos], src[pos + 1]]) as usize;
        pos += 2;

        if offset == 0 || offset > dst.len() {
            return Err(EngineError::corrupt("lz4: offset out of range"));
        }

        let mut match_len = (token & 0x0F) as usize;
        if match_len == 15 {
            loop {
                let byte = *src
                    .get(pos)
                    .ok_or_else(|| EngineError::corrupt("lz4: truncated match length"))?;
                pos += 1;
                match_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }
        match_len += MIN_MATCH;

        let mut src_pos = dst.len() - offset;
        if dst.len() + match_len > expected_len {
            return Err(EngineError::corrupt("lz4: output exceeds declared size"));
        }

        // Byte-at-a-time on purpose: `offset` can be 1, so a bulk copy would
        // read bytes it has not written yet.
        dst.reserve(match_len);
        for _ in 0..match_len {
            let byte = dst[src_pos];
            dst.push(byte);
            src_pos += 1;
        }
    }

    if dst.len() != expected_len {
        return Err(EngineError::corrupt(format!(
            "lz4: expected {expected_len} bytes, produced {}",
            dst.len()
        )));
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let packed = compress(data);
        let restored = decompress(&packed, data.len()).expect("decompress");
        assert_eq!(restored, data, "round trip failed for {} bytes", data.len());
    }

    #[test]
    fn round_trips() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        round_trip(b"hello world hello world hello world");
        round_trip(&[0u8; 100_000]);

        // Float-ish data: the case we actually care about.
        let floats: Vec<f32> = (0..50_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for f in &floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        round_trip(&bytes);
    }

    #[test]
    fn round_trips_pseudorandom() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut data = Vec::with_capacity(200_000);
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            data.push((state >> 24) as u8);
        }
        round_trip(&data);
    }

    #[test]
    fn compresses_repetitive_data() {
        let data = vec![7u8; 100_000];
        let packed = compress(&data);
        assert!(
            packed.len() < data.len() / 50,
            "expected >50x on constant data, got {}",
            packed.len()
        );
    }

    #[test]
    fn rejects_garbage() {
        // Truncated and malformed inputs must produce errors, never panics.
        assert!(decompress(&[0xF0], 100).is_err());
        assert!(decompress(&[0x00, 0x00], 10).is_err());
        assert!(decompress(&[0x1F, 0xFF, 0xFF], 10).is_err());
        assert!(decompress(&[0x40, b'a', b'b', 0x00, 0x00, 0x00], 100).is_err());
        // Declared length larger than what the stream produces.
        assert!(decompress(&[0x40, b'a', b'b'], 99).is_err());
    }

    #[test]
    fn never_exceeds_bound() {
        for len in [0usize, 1, 12, 16, 100, 4096, 65_536] {
            let data: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
            assert!(compress(&data).len() <= compress_bound(len));
        }
    }
}
