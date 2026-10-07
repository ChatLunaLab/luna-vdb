//! Snapshot format (`LVD2`).
//!
//! # Why a new format
//!
//! The previous format was `gzip(bincode(Index))`. Three problems, all of which
//! show up as user-visible symptoms:
//!
//! 1. **Speed.** deflate compresses at 30–60 MB/s and runs on *every* byte of
//!    a snapshot that is mostly near-random f32 mantissas it cannot compress.
//!    Serialising a 100 MB index took seconds. We now write a flat binary
//!    layout and compress with [`crate::engine::lz4`], which runs at several
//!    hundred MB/s.
//! 2. **No validation.** bincode reads lengths straight off the wire. A
//!    truncated or version-mismatched snapshot either produced a wild
//!    allocation or aborted the module — the "神秘的空指针" that only showed up
//!    after calling into JS. Every length is now bounds-checked against the
//!    remaining buffer, and the whole payload carries a CRC-32.
//! 3. **Layout.** `Vec<Vec<f32>>` round-trips through N allocations. The flat
//!    arena here is two contiguous memcpys.
//!
//! # Compatibility
//!
//! [`read`] sniffs the gzip magic (`1f 8b`) and routes to [`read_legacy`],
//! which understands the old `bincode` layout. Old snapshots therefore keep
//! loading; new ones are written in `LVD2`.
//!
//! # Layout
//!
//! ```text
//! offset  size  field
//!      0     4  magic "LVD2"
//!      4     2  format version (u16 LE)
//!      6     2  flags (bit 0: payload is LZ4-compressed)
//!      8     4  raw payload length (u32 LE)
//!     12     4  CRC-32 over bytes 4..12 followed by the stored payload
//!     16     …  stored payload
//! ```
//!
//! The checksum covers the version, flags and length as well as the payload,
//! so a flipped bit in the header is caught before the length is used for
//! anything.

use crate::engine::crc32::{crc32, crc32_update};
use crate::engine::lz4;
use crate::engine::types::{Distance, EngineError, EngineResult};

/// `LVD2`, little-endian on the wire.
pub const MAGIC: [u8; 4] = *b"LVD2";
/// Current format version. Bump on any layout change.
pub const FORMAT_VERSION: u16 = 2;
/// Fixed header size: magic, version, flags, raw length, payload CRC.
pub const HEADER_LEN: usize = 16;

const FLAG_COMPRESSED: u16 = 1 << 0;

/// Largest payload the format can describe: the length field is a `u32`.
///
/// This used to be 8 GiB, which let a 64-bit host write a 4–8 GiB snapshot
/// whose length silently truncated to 32 bits and could never be read back.
/// (And written as `8 << 30` in `usize`, it overflowed to zero on wasm32, so
/// every snapshot there was rejected as "too large".) Writes above this limit
/// now fail up front.
const MAX_RAW_LEN: u64 = u32::MAX as u64;

/// The ceiling as a `usize`, saturated to what this target can index.
fn max_raw_len() -> usize {
    MAX_RAW_LEN.min(usize::MAX as u64) as usize
}

/// CRC of everything the header vouches for: version, flags, length, payload.
fn envelope_crc(header: &[u8], stored: &[u8]) -> u32 {
    crc32_update(crc32(header.get(4..12).unwrap_or(&[])), stored)
}

/// `true` if `bytes` starts like a snapshot this build can read: `LVD2`, or a
/// gzip stream from luna-vdb ≤ 0.0.12. A header sniff only — it does not
/// validate the payload.
pub fn is_snapshot(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC) || looks_like_gzip(bytes)
}

/// Format version of `bytes`: the `LVD2` header's version field, `1` for a
/// legacy gzip snapshot, `0` if unrecognised.
pub fn snapshot_version(bytes: &[u8]) -> u16 {
    if bytes.len() >= 6 && bytes.starts_with(&MAGIC) {
        u16::from_le_bytes([bytes[4], bytes[5]])
    } else if looks_like_gzip(bytes) {
        1
    } else {
        0
    }
}

/// Everything needed to rebuild an [`crate::engine::Engine`], in a plain
/// struct so the codec has no dependency on the engine's runtime state.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub distance: Distance,
    pub dim: u32,
    pub count: u32,
    /// `count × dim`, row-major.
    pub data: Vec<f32>,
    /// `count` cached squared norms.
    pub cache: Vec<f32>,
    /// `count` ids, in row order.
    pub ids: Vec<String>,
    pub nlist: u32,
    pub nprobe: u32,
    /// `nlist × dim`, row-major.
    pub centroids: Vec<f32>,
    /// `nlist` lists of row indices.
    pub lists: Vec<Vec<u32>>,
    pub pq: Option<PqSnapshot>,
    /// `count × pq.m` codes.
    pub codes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct PqSnapshot {
    pub m: u32,
    pub ksub: u32,
    pub subvector_dim: u32,
    /// `m × ksub × subvector_dim`, row-major.
    pub codebooks: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Serialise `snapshot`. When `compressed` is set the payload is LZ4-packed;
/// callers should pick compression for storage and skip it for an in-process
/// transfer where the copy costs more than it saves.
pub fn write(snapshot: &Snapshot, compressed: bool) -> EngineResult<Vec<u8>> {
    let mut payload = Vec::with_capacity(estimate_payload_len(snapshot));
    write_payload(snapshot, &mut payload);

    let raw_len = payload.len();
    if raw_len > max_raw_len() {
        return Err(EngineError::new(format!(
            "snapshot too large to serialise: {raw_len} bytes (limit {})",
            max_raw_len()
        )));
    }

    let (flags, stored): (u16, Vec<u8>) = if compressed {
        let packed = lz4::compress(&payload);
        // Compressing already-compressed or tiny payloads can grow them. Keep
        // whichever is smaller so the flag never costs the caller bytes.
        if packed.len() < raw_len {
            (FLAG_COMPRESSED, packed)
        } else {
            (0, payload)
        }
    } else {
        (0, payload)
    };

    let mut out = Vec::with_capacity(HEADER_LEN + stored.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(raw_len as u32).to_le_bytes());
    let crc = envelope_crc(&out, &stored);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&stored);

    Ok(out)
}

fn write_payload(snapshot: &Snapshot, out: &mut Vec<u8>) {
    out.push(snapshot.distance.to_u8());
    out.push(0); // padding, keeps the u32s that follow naturally aligned
    out.extend_from_slice(&snapshot.dim.to_le_bytes());
    out.extend_from_slice(&snapshot.count.to_le_bytes());

    write_floats(&snapshot.data, out);
    write_floats(&snapshot.cache, out);

    for id in &snapshot.ids {
        out.extend_from_slice(&(id.len() as u32).to_le_bytes());
        out.extend_from_slice(id.as_bytes());
    }

    out.extend_from_slice(&snapshot.nlist.to_le_bytes());
    out.extend_from_slice(&snapshot.nprobe.to_le_bytes());
    write_floats(&snapshot.centroids, out);

    for list in &snapshot.lists {
        out.extend_from_slice(&(list.len() as u32).to_le_bytes());
        for &row in list {
            out.extend_from_slice(&row.to_le_bytes());
        }
    }

    match &snapshot.pq {
        Some(pq) => {
            out.push(1);
            out.extend_from_slice(&pq.m.to_le_bytes());
            out.extend_from_slice(&pq.ksub.to_le_bytes());
            out.extend_from_slice(&pq.subvector_dim.to_le_bytes());
            write_floats(&pq.codebooks, out);
            out.extend_from_slice(&(snapshot.codes.len() as u32).to_le_bytes());
            out.extend_from_slice(&snapshot.codes);
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
}

/// Bulk little-endian f32 write.
///
/// Reserves once and fills, rather than pushing per element — on a 100 MB index
/// that is the difference between ~30 ms and ~400 ms.
#[inline]
fn write_floats(values: &[f32], out: &mut Vec<u8>) {
    out.reserve(values.len() * 4);

    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn estimate_payload_len(snapshot: &Snapshot) -> usize {
    let floats = snapshot.data.len()
        + snapshot.cache.len()
        + snapshot.centroids.len()
        + snapshot.pq.as_ref().map_or(0, |pq| pq.codebooks.len());

    let id_bytes: usize = snapshot.ids.iter().map(|id| id.len() + 4).sum();
    let list_bytes: usize = snapshot.lists.iter().map(|l| l.len() * 4 + 4).sum();

    floats * 4 + id_bytes + list_bytes + snapshot.codes.len() + 64
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Bounds-checked cursor. Every read verifies length first and returns
/// [`EngineError`] rather than panicking, which is what keeps a malformed
/// snapshot from poisoning the wasm instance.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, len: usize) -> EngineResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| EngineError::corrupt("offset overflow"))?;

        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| EngineError::corrupt("truncated payload"))?;

        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> EngineResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> EngineResult<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> EngineResult<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read `count` f32s. Rejects the read before allocating when the buffer
    /// cannot possibly hold them.
    fn floats(&mut self, count: usize) -> EngineResult<Vec<f32>> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| EngineError::corrupt("float count overflow"))?;

        let slice = self.take(bytes)?;

        // Read through a chunked loop rather than a cast. The payload is packed
        // with ids and length prefixes in between, so the float runs are not
        // guaranteed 4-byte aligned; a `slice as &[f32]` cast would be UB on an
        // unaligned buffer. `chunks_exact(4)` plus `from_le_bytes` is alignment
        // agnostic, compiles to a plain load, and is endian-explicit so a
        // snapshot written on a big-endian host still reads correctly.
        let mut out = Vec::with_capacity(count);
        let mut chunks = slice.chunks_exact(4);

        for chunk in &mut chunks {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(out)
    }

    fn string(&mut self) -> EngineResult<String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        // Ids are arbitrary user strings but must be valid UTF-8; a bad one is
        // a corrupt file, not a panic.
        String::from_utf8(bytes.to_vec()).map_err(|_| EngineError::corrupt("id is not valid UTF-8"))
    }
}

/// Parse a snapshot in the current (`LVD2`) format.
pub fn read(bytes: &[u8]) -> EngineResult<Snapshot> {
    let mut reader = Reader::new(bytes);

    let magic = reader.take(4)?;
    if magic != MAGIC {
        if looks_like_gzip(bytes) {
            return read_legacy(bytes);
        }
        return Err(EngineError::corrupt(format!(
            "bad magic: expected {MAGIC:?}, found {magic:?}"
        )));
    }

    let version = reader.u16()?;
    if version != FORMAT_VERSION {
        return Err(EngineError::new(format!(
            "snapshot format version {version} is not supported by this build (expected {FORMAT_VERSION})"
        )));
    }

    let flags = reader.u16()?;
    let raw_len = reader.u32()? as u64;
    let expected_crc = reader.u32()?;

    // Compared as `u64` so the check means the same thing on every target.
    if raw_len > max_raw_len() as u64 {
        return Err(EngineError::corrupt(format!(
            "declared payload of {raw_len} bytes exceeds the {} byte limit",
            max_raw_len()
        )));
    }
    let raw_len = raw_len as usize;

    let stored = reader.take(reader.remaining())?;

    // Verify the checksum *before* using anything the header says: it turns
    // "random garbage that happens to have plausible lengths" into one clear
    // error, and it covers the length field, so a corrupted length never
    // reaches an allocation.
    let actual_crc = envelope_crc(bytes, stored);
    if actual_crc != expected_crc {
        return Err(EngineError::corrupt(format!(
            "checksum mismatch: header says {expected_crc:#010x}, contents give {actual_crc:#010x}"
        )));
    }

    let payload = if flags & FLAG_COMPRESSED != 0 {
        lz4::decompress(stored, raw_len)?
    } else {
        if stored.len() != raw_len {
            return Err(EngineError::corrupt(format!(
                "uncompressed payload is {} bytes but the header declares {raw_len}",
                stored.len()
            )));
        }
        stored.to_vec()
    };

    read_payload(&payload)
}

fn read_payload(payload: &[u8]) -> EngineResult<Snapshot> {
    let mut reader = Reader::new(payload);

    let distance = Distance::from_u8(reader.u8()?)
        .ok_or_else(|| EngineError::corrupt("unknown distance metric code"))?;
    let _padding = reader.u8()?;
    let dim = reader.u32()?;
    let count = reader.u32()?;

    // Guard every `count × dim` product before it reaches an allocation.
    //
    // Multiplied in `u64` rather than `usize`: two `u32` fields can produce a
    // product that overflows a 32-bit `usize`, and the *wrapped* value would be
    // small enough to slip past the check below and then index out of bounds.
    let elements = (count as u64)
        .checked_mul(dim as u64)
        .ok_or_else(|| EngineError::corrupt("count × dim overflow"))?;

    if elements > MAX_RAW_LEN / 4 || elements > max_raw_len() as u64 / 4 {
        return Err(EngineError::corrupt(
            "vector count × dimension is implausible",
        ));
    }
    let elements = elements as usize;

    let data = reader.floats(elements)?;
    let cache = reader.floats(count as usize)?;

    // Every id costs at least its 4-byte length prefix, so the remaining
    // bytes bound how many there can be — reserve no more than that, whatever
    // `count` claims.
    let mut ids = Vec::with_capacity((count as usize).min(reader.remaining() / 4));
    for _ in 0..count {
        ids.push(reader.string()?);
    }

    let nlist = reader.u32()?;
    let nprobe = reader.u32()?;

    // No `nlist <= count` check: deletions shrink the corpus without shrinking
    // the cell count, so the engine legitimately writes more cells than rows,
    // and rejecting that made such snapshots unloadable. Every list entry is
    // checked against `count` below instead.

    let centroids = reader.floats(
        (nlist as usize)
            .checked_mul(dim as usize)
            .ok_or_else(|| EngineError::corrupt("centroid count overflow"))?,
    )?;

    // Each list costs at least a 4-byte length, so bound the reservation by
    // the remaining bytes — `nlist` alone is attacker-controlled.
    let mut lists = Vec::with_capacity((nlist as usize).min(reader.remaining() / 4));
    for _ in 0..nlist {
        let len = reader.u32()? as usize;
        if len > count as usize {
            return Err(EngineError::corrupt(format!(
                "list length {len} exceeds the vector count {count}"
            )));
        }

        let mut list = Vec::with_capacity(len);
        for _ in 0..len {
            let row = reader.u32()?;
            if row >= count {
                return Err(EngineError::corrupt(format!(
                    "list references row {row}, but only {count} exist"
                )));
            }
            list.push(row);
        }
        lists.push(list);
    }

    let has_pq = reader.u8()? != 0;
    let pq = if has_pq {
        let m = reader.u32()?;
        let ksub = reader.u32()?;
        let subvector_dim = reader.u32()?;

        if m as usize > dim.max(1) as usize {
            return Err(EngineError::corrupt(
                "PQ subquantizer count exceeds the dimension",
            ));
        }
        // A code is one byte, so a codebook has at most 256 entries; and a
        // zero-width subvector cannot encode anything.
        if !(1..=256).contains(&ksub) || subvector_dim == 0 {
            return Err(EngineError::corrupt(format!(
                "PQ shape ksub={ksub} subvector_dim={subvector_dim} is invalid"
            )));
        }

        let codebook_len = (m as usize)
            .checked_mul(ksub as usize)
            .and_then(|v| v.checked_mul(subvector_dim as usize))
            .ok_or_else(|| EngineError::corrupt("codebook size overflow"))?;

        let codebooks = reader.floats(codebook_len)?;
        let code_len = reader.u32()? as usize;

        let expected_codes = (count as u64) * (m as u64);
        if code_len as u64 != expected_codes {
            return Err(EngineError::corrupt(format!(
                "code buffer is {code_len} bytes, expected {expected_codes}"
            )));
        }

        let codes = reader.take(code_len)?.to_vec();

        Some(PqSnapshot {
            m,
            ksub,
            subvector_dim,
            codebooks,
        })
        .map(|pq| (pq, codes))
    } else {
        let _reserved = reader.u32()?;
        None
    };

    let (pq, codes) = match pq {
        Some((pq, codes)) => (Some(pq), codes),
        None => (None, Vec::new()),
    };

    Ok(Snapshot {
        distance,
        dim,
        count,
        data,
        cache,
        ids,
        nlist,
        nprobe,
        centroids,
        lists,
        pq,
        codes,
    })
}

/// The gzip magic. Used to detect snapshots written by luna-vdb ≤ 0.0.12.
fn looks_like_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

// ---------------------------------------------------------------------------
// Legacy (≤ 0.0.12) import
// ---------------------------------------------------------------------------

/// Mirrors the pre-`LVD2` `Index` layout so `bincode` can consume it.
///
/// `bincode` is not self-describing: it serialises structs as a bare field
/// sequence and enums as their variant index. So the field order, field types
/// and variant order below must match the old definitions *exactly*, and the
/// `#[serde(rename)]` attributes the old code carried are irrelevant here
/// (they only affect human-readable formats). This is why the structs are
/// duplicated rather than derived from the current types.
///
/// `dead_code` is allowed deliberately. Several of these fields exist only so
/// the decoder consumes the right number of bytes for their type — the
/// PQ codebooks, `hash`, `coarse_assignments` and friends are translated into
/// the new layout only in part, or re-derived from `embeddings` instead. They
/// look unused to the compiler and are load-bearing for the byte offsets; a
/// well-meaning cleanup that drops one silently corrupts every import from an
/// older snapshot.
#[allow(dead_code)]
mod legacy {
    use serde::Deserialize;
    use std::collections::HashMap;

    /// Variant order must match the old `Distance` enum: Euclidean, Cosine,
    /// DotProduct — the same as the current one, which is why the discriminants
    /// line up and no translation table is needed.
    #[derive(Debug, Deserialize)]
    pub enum LegacyDistance {
        Euclidean,
        Cosine,
        DotProduct,
    }

    #[derive(Debug, Deserialize)]
    pub struct LegacyVectorData {
        pub vector: Vec<f32>,
        pub cache_attr: f32,
    }

    #[derive(Debug, Deserialize)]
    pub struct LegacyProductQuantizer {
        pub m: usize,
        pub ksub: usize,
        pub subvector_dim: usize,
        pub codebooks: Vec<Vec<Vec<f32>>>,
    }

    /// Current-shape legacy index (v0.0.10 … 0.0.12, i.e. after IVF-PQ landed).
    #[derive(Debug, Deserialize)]
    pub struct LegacyIndex {
        pub embeddings: Vec<LegacyVectorData>,
        pub hash: HashMap<u64, usize>,
        pub ids: Vec<String>,
        pub distance: LegacyDistance,
        pub dimension: usize,
        pub nlist: usize,
        pub nprobe: usize,
        pub coarse_centroids: Vec<Vec<f32>>,
        pub coarse_assignments: Vec<usize>,
        pub lists: Vec<Vec<usize>>,
        pub pq: LegacyProductQuantizer,
        pub codes: Vec<Vec<u8>>,
    }

    /// Pre-IVF shape (v0.0.9 and earlier, KD-tree era).
    #[derive(Debug, Deserialize)]
    pub struct LegacyIndexPreIvf {
        pub embeddings: Vec<LegacyVectorData>,
        pub hash: HashMap<u64, usize>,
        pub ids: Vec<String>,
        pub distance: LegacyDistance,
        pub dimension: usize,
    }
}

/// Read a snapshot written by luna-vdb ≤ 0.0.12 (gzip + bincode).
///
/// The IVF structures are dropped on purpose: they were trained with the old
/// modulo seeding, and carrying them over would preserve exactly the bad
/// clusters [`crate::engine::kmeans`] exists to fix. The vectors and ids are
/// what matter; the index is retrained on load.
pub fn read_legacy(bytes: &[u8]) -> EngineResult<Snapshot> {
    let raw = inflate_capped(bytes, LEGACY_LIMIT)?;

    if let Ok(index) = bincode::deserialize::<legacy::LegacyIndex>(&raw) {
        return convert_legacy(
            index.embeddings,
            index.ids,
            &index.distance,
            index.dimension,
        );
    }

    if let Ok(index) = bincode::deserialize::<legacy::LegacyIndexPreIvf>(&raw) {
        return convert_legacy(
            index.embeddings,
            index.ids,
            &index.distance,
            index.dimension,
        );
    }

    Err(EngineError::corrupt(
        "legacy snapshot did not match any known pre-0.1 layout",
    ))
}

/// Inflate at most this much from a legacy snapshot.
const LEGACY_LIMIT: u64 = 1 << 30;

/// Gunzip `bytes`, refusing output beyond `limit`.
///
/// A gzip stream can expand by ~1000x, so an unbounded `read_to_end` is a
/// memory bomb. The cap also keeps the buffer's doubling growth inside a
/// 32-bit address space, where `read_to_end` reports an allocation failure as
/// an error rather than panicking.
fn inflate_capped(bytes: &[u8], limit: u64) -> EngineResult<Vec<u8>> {
    use std::io::Read;

    let mut decoder = flate2::read::GzDecoder::new(bytes).take(limit.saturating_add(1));
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|error| EngineError::corrupt(format!("legacy gzip: {error}")))?;

    if raw.len() as u64 > limit {
        return Err(EngineError::corrupt(format!(
            "legacy snapshot inflates past {limit} bytes; refusing to import"
        )));
    }
    Ok(raw)
}

fn legacy_distance(distance: &legacy::LegacyDistance) -> Distance {
    match distance {
        legacy::LegacyDistance::Euclidean => Distance::Euclidean,
        legacy::LegacyDistance::Cosine => Distance::Cosine,
        legacy::LegacyDistance::DotProduct => Distance::DotProduct,
    }
}

/// Flatten the legacy per-vector layout.
///
/// The IVF structures are dropped on purpose: they were trained with the old
/// modulo seeding, and the engine retrains on load. The `dimension` field is
/// checked against the vectors themselves rather than trusted — it used to go
/// straight into `Vec::with_capacity(count * dimension)`, so a single corrupt
/// field could request terabytes, or wrap on wasm32 and slip past the length
/// check that follows.
fn convert_legacy(
    embeddings: Vec<legacy::LegacyVectorData>,
    ids: Vec<String>,
    distance: &legacy::LegacyDistance,
    dimension: usize,
) -> EngineResult<Snapshot> {
    let count = embeddings.len();

    if ids.len() != count {
        return Err(EngineError::corrupt(format!(
            "legacy snapshot has {} ids for {count} vectors",
            ids.len()
        )));
    }
    if let Some(bad) = embeddings.iter().find(|e| e.vector.len() != dimension) {
        return Err(EngineError::corrupt(format!(
            "legacy snapshot declares dimension {dimension} but holds a vector of length {}",
            bad.vector.len()
        )));
    }
    let dim = u32::try_from(dimension)
        .map_err(|_| EngineError::corrupt("legacy dimension does not fit in 32 bits"))?;
    let count_u32 = u32::try_from(count)
        .map_err(|_| EngineError::corrupt("legacy vector count does not fit in 32 bits"))?;

    // Every vector has been checked to hold `dimension` floats, so this is the
    // size of data actually present, not a number read off the wire.
    let total: usize = embeddings.iter().map(|e| e.vector.len()).sum();
    let mut data = Vec::with_capacity(total);
    let mut cache = Vec::with_capacity(count);
    for embedding in &embeddings {
        data.extend_from_slice(&embedding.vector);
        cache.push(embedding.cache_attr);
    }

    Ok(Snapshot {
        distance: legacy_distance(distance),
        dim,
        count: count_u32,
        data,
        cache,
        ids,
        nlist: 0,
        nprobe: 0,
        centroids: Vec::new(),
        lists: Vec::new(),
        pq: None,
        codes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Snapshot {
        Snapshot {
            distance: Distance::Cosine,
            dim: 3,
            count: 2,
            data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            cache: vec![0.14, 0.77],
            ids: vec!["cat".to_string(), "dog".to_string()],
            nlist: 1,
            nprobe: 1,
            centroids: vec![0.25, 0.35, 0.45],
            lists: vec![vec![0, 1]],
            pq: None,
            codes: Vec::new(),
        }
    }

    fn sample_with_pq() -> Snapshot {
        Snapshot {
            pq: Some(PqSnapshot {
                m: 3,
                ksub: 2,
                subvector_dim: 1,
                codebooks: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            }),
            codes: vec![0, 1, 0, 1, 1, 0],
            ..sample()
        }
    }

    #[test]
    fn round_trips_both_modes() {
        for compressed in [false, true] {
            for snapshot in [sample(), sample_with_pq()] {
                let bytes = write(&snapshot, compressed).expect("write");
                let restored = read(&bytes).expect("read");

                assert_eq!(restored.distance, snapshot.distance);
                assert_eq!(restored.dim, snapshot.dim);
                assert_eq!(restored.count, snapshot.count);
                assert_eq!(restored.data, snapshot.data);
                assert_eq!(restored.cache, snapshot.cache);
                assert_eq!(restored.ids, snapshot.ids);
                assert_eq!(restored.lists, snapshot.lists);
                assert_eq!(restored.codes, snapshot.codes);
                assert_eq!(
                    restored.pq.as_ref().map(|p| p.codebooks.clone()),
                    snapshot.pq.as_ref().map(|p| p.codebooks.clone())
                );
            }
        }
    }

    #[test]
    fn empty_snapshot_round_trips() {
        let snapshot = Snapshot::default();
        let bytes = write(&snapshot, true).expect("write");
        let restored = read(&bytes).expect("read");
        assert_eq!(restored.count, 0);
        assert!(restored.ids.is_empty());
        assert!(restored.data.is_empty());
    }

    #[test]
    fn detects_corruption() {
        let bytes = write(&sample(), true).expect("write");

        // Flip a byte in the payload — the CRC must catch it.
        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
        assert!(
            read(&corrupted).is_err(),
            "CRC did not catch a flipped byte"
        );

        // Truncate — must be an error, never a panic or a huge allocation.
        for cut in [0usize, 1, 7, 15, 17, bytes.len() / 2] {
            let result = read(&bytes[..cut]);
            assert!(result.is_err(), "truncation to {cut} should fail");
        }

        // Garbage that is not a snapshot at all.
        assert!(read(b"not a snapshot at all, definitely").is_err());
        assert!(read(&[]).is_err());
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = write(&sample(), false).expect("write");
        bytes[4] = 99;
        bytes[5] = 0;
        let error = read(&bytes).expect_err("version check");
        assert!(error.message.contains("not supported"), "{}", error.message);
    }

    /// Assemble a header around `stored` with a *valid* checksum, so a test
    /// exercises the length handling rather than being stopped by the CRC.
    fn envelope(flags: u16, raw_len: u32, stored: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&raw_len.to_le_bytes());
        let crc = envelope_crc(&bytes, stored);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(stored);
        bytes
    }

    #[test]
    fn rejects_implausible_lengths() {
        // REGRESSION: this test used to build its header with `4u32 << 30`,
        // which is 0, and an all-zero CRC, so it passed on the checksum and
        // never reached the length handling it was named for.
        for stored in [&[][..], &[0x10, b'x'][..], &[0u8; 16][..]] {
            for raw_len in [u32::MAX, 1 << 31, 1 << 30] {
                let bytes = envelope(FLAG_COMPRESSED, raw_len, stored);
                assert!(read(&bytes).is_err(), "raw_len {raw_len}");
                let bytes = envelope(0, raw_len, stored);
                assert!(read(&bytes).is_err(), "raw_len {raw_len}");
            }
        }
    }

    #[test]
    fn header_fields_are_checksummed() {
        let good = write(&sample(), true).expect("write");
        for offset in 4..16 {
            let mut bad = good.clone();
            bad[offset] ^= 0x01;
            assert!(read(&bad).is_err(), "flip at header byte {offset}");
        }
    }

    #[test]
    fn hostile_counts_do_not_allocate() {
        // count = 0 with nlist = u32::MAX: the list reservation used to be
        // `nlist` long, which is a capacity-overflow trap on wasm32.
        let mut payload = vec![0u8, 0];
        payload.extend_from_slice(&0u32.to_le_bytes()); // dim
        payload.extend_from_slice(&0u32.to_le_bytes()); // count
        payload.extend_from_slice(&u32::MAX.to_le_bytes()); // nlist
        payload.extend_from_slice(&0u32.to_le_bytes()); // nprobe
        let bytes = envelope(0, payload.len() as u32, &payload);
        assert!(read(&bytes).is_err());

        // A PQ block claiming a 2^32-entry codebook of zero-width vectors.
        let mut snapshot = sample_with_pq();
        snapshot.pq = Some(PqSnapshot {
            m: 1,
            ksub: u32::MAX,
            subvector_dim: 0,
            codebooks: Vec::new(),
        });
        snapshot.codes = vec![0, 0];
        let bytes = write(&snapshot, false).expect("write");
        assert!(read(&bytes).is_err());
    }

    #[test]
    fn more_cells_than_rows_round_trips() {
        // REGRESSION: deletions leave `nlist` above the row count, and the
        // reader used to reject exactly the snapshots the engine wrote then.
        let snapshot = Snapshot {
            nlist: 4,
            centroids: vec![0.0; 12],
            lists: vec![vec![0, 1], Vec::new(), Vec::new(), Vec::new()],
            ..sample()
        };
        let restored = read(&write(&snapshot, true).expect("write")).expect("read");
        assert_eq!(restored.nlist, 4);
        assert_eq!(restored.lists, snapshot.lists);
    }

    #[test]
    fn sniffing_helpers() {
        let bytes = write(&sample(), true).expect("write");
        assert!(is_snapshot(&bytes));
        assert_eq!(snapshot_version(&bytes), FORMAT_VERSION);
        assert!(is_snapshot(&[0x1f, 0x8b, 0x08]));
        assert_eq!(snapshot_version(&[0x1f, 0x8b, 0x08]), 1);
        assert!(!is_snapshot(b"nope"));
        assert_eq!(snapshot_version(&[]), 0);
    }

    // -- legacy import ----------------------------------------------------

    /// The pre-0.1 types, as they were serialised. Field order and types are
    /// copied from commit `ca10590` (`src/engine/types.rs`); bincode encodes
    /// structs as bare field sequences, so this is the wire format.
    mod old {
        use serde::Serialize;
        use std::collections::HashMap;

        #[derive(Serialize)]
        #[allow(dead_code)]
        pub enum Distance {
            Euclidean,
            Cosine,
            DotProduct,
        }

        #[derive(Serialize)]
        pub struct VectorData {
            pub vector: Vec<f32>,
            pub cache_attr: f32,
        }

        #[derive(Serialize, Default)]
        pub struct ProductQuantizer {
            pub m: usize,
            pub ksub: usize,
            pub subvector_dim: usize,
            pub codebooks: Vec<Vec<Vec<f32>>>,
        }

        #[derive(Serialize)]
        pub struct Index {
            pub embeddings: Vec<VectorData>,
            pub hash: HashMap<u64, usize>,
            pub ids: Vec<String>,
            pub distance: Distance,
            pub dimension: usize,
            pub nlist: usize,
            pub nprobe: usize,
            pub coarse_centroids: Vec<Vec<f32>>,
            pub coarse_assignments: Vec<usize>,
            pub lists: Vec<Vec<usize>>,
            pub pq: ProductQuantizer,
            pub codes: Vec<Vec<u8>>,
        }
    }

    fn gzip(raw: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(raw).expect("gzip write");
        encoder.finish().expect("gzip finish")
    }

    fn old_index(vectors: Vec<Vec<f32>>, ids: Vec<&str>, dimension: usize) -> old::Index {
        // Sized from the vectors, not `dimension`: the hostile-dimension test
        // passes 2^40, and the fixture must not be the thing that allocates it.
        let centroid = vec![0.0; vectors.first().map_or(0, Vec::len)];
        old::Index {
            embeddings: vectors
                .into_iter()
                .map(|vector| old::VectorData {
                    cache_attr: vector.iter().map(|x| x * x).sum(),
                    vector,
                })
                .collect(),
            hash: std::collections::HashMap::new(),
            ids: ids.into_iter().map(String::from).collect(),
            distance: old::Distance::Cosine,
            dimension,
            nlist: 1,
            nprobe: 1,
            coarse_centroids: vec![centroid],
            coarse_assignments: vec![0, 0],
            lists: vec![vec![0, 1]],
            pq: old::ProductQuantizer::default(),
            codes: Vec::new(),
        }
    }

    #[test]
    fn imports_a_legacy_snapshot() {
        let index = old_index(
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec!["a", "b"],
            3,
        );
        let bytes = gzip(&bincode::serialize(&index).expect("bincode"));

        assert!(is_snapshot(&bytes));
        let snapshot = read(&bytes).expect("legacy import");
        assert_eq!(snapshot.distance, Distance::Cosine);
        assert_eq!(snapshot.dim, 3);
        assert_eq!(snapshot.count, 2);
        assert_eq!(snapshot.ids, vec!["a", "b"]);
        assert_eq!(snapshot.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            snapshot.nlist, 0,
            "the old index is discarded and retrained"
        );
    }

    #[test]
    fn legacy_dimension_is_checked_not_trusted() {
        // REGRESSION: `dimension` went straight into
        // `Vec::with_capacity(count * dimension)`. `usize::MAX >> 2` is 2^62
        // natively and 2^30 on wasm32, so the test compiles on both.
        for dimension in [usize::MAX, usize::MAX >> 2, 1 << 30, 2, 0] {
            let index = old_index(
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                vec!["a", "b"],
                dimension,
            );
            let bytes = gzip(&bincode::serialize(&index).expect("bincode"));
            assert!(read(&bytes).is_err(), "dimension {dimension}");
        }
    }

    #[test]
    fn legacy_inflate_is_capped() {
        let bytes = gzip(&vec![0u8; 1 << 20]);
        assert!(inflate_capped(&bytes, 1 << 10).is_err());
        assert_eq!(
            inflate_capped(&bytes, 1 << 20).expect("within cap").len(),
            1 << 20
        );
    }

    #[test]
    fn imports_legacy_duplicate_ids_without_shifting() {
        // The pre-0.1 engine never checked ids, so its snapshots can repeat
        // one. The parser passes them through; the engine keeps the first.
        let index = old_index(
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]],
            vec!["a", "a", "b"],
            2,
        );
        let bytes = gzip(&bincode::serialize(&index).expect("bincode"));
        let snapshot = read(&bytes).expect("legacy import");
        assert_eq!(snapshot.ids, vec!["a", "a", "b"]);

        let engine =
            crate::engine::Engine::from_snapshot(snapshot, Default::default()).expect("load");
        assert_eq!(engine.len(), 2);
        // `b` must still own `[-1, 0]`. Before the fix it was shifted onto the
        // duplicate's `[0, 1]`, one unit of cosine distance away.
        let hit = &engine.search(&[-1.0, 0.0], 1).neighbors[0];
        assert_eq!(hit.id, "b");
        assert!(
            hit.distance.abs() < 1e-6,
            "b is at distance {}",
            hit.distance
        );
    }
}
