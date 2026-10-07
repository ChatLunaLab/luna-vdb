//! CRC-32 (IEEE 802.3, the same polynomial as zlib/PKZIP), slicing-by-4.
//!
//! Hand-rolled rather than pulled from a crate for two reasons: it has to be
//! cheap enough to run over a 100 MB snapshot on every `serialize()`, and the
//! only place a checksum is needed is the snapshot envelope — a single-table
//! byte-at-a-time loop would cost ~30 ms/100 MB, slicing-by-4 costs ~7 ms.
//!
//! The table is built in a `const fn`, so it costs nothing at runtime and
//! nothing at load time beyond the 4 KB of `.rodata`.

const POLY: u32 = 0xEDB8_8320;

const fn build_table<const SLICES: usize>() -> [[u32; 256]; SLICES] {
    let mut table = [[0u32; 256]; SLICES];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[0][i] = crc;
        i += 1;
    }

    // Derive the remaining slices: table[n][i] == crc of byte i with n zero
    // bytes appended, which is exactly the shift-by-one-byte operation.
    let mut slice = 1usize;
    while slice < SLICES {
        let mut i = 0usize;
        while i < 256 {
            let prev = table[slice - 1][i];
            table[slice][i] = (prev >> 8) ^ table[0][(prev & 0xFF) as usize];
            i += 1;
        }
        slice += 1;
    }

    table
}

static TABLE: [[u32; 256]; 4] = build_table::<4>();

/// CRC-32 of `data`, as an unsigned 32-bit value (not the flipped/negated
/// "CRC32" that some APIs report).
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

/// Continue a CRC across chunks. `seed` is the value returned by the previous
/// call; `0` starts a fresh checksum.
pub fn crc32_update(seed: u32, data: &[u8]) -> u32 {
    let mut crc = !seed;

    let mut chunks = data.chunks_exact(4);
    for chunk in &mut chunks {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        crc ^= word;
        crc = TABLE[3][(crc & 0xFF) as usize]
            ^ TABLE[2][((crc >> 8) & 0xFF) as usize]
            ^ TABLE[1][((crc >> 16) & 0xFF) as usize]
            ^ TABLE[0][((crc >> 24) & 0xFF) as usize];
    }

    for &byte in chunks.remainder() {
        crc = (crc >> 8) ^ TABLE[0][((crc ^ byte as u32) & 0xFF) as usize];
    }

    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // Standard check values for CRC-32/ISO-HDLC.
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn chunking_is_stable() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let whole = crc32(&data);

        for split in [1usize, 3, 4, 5, 7, 64, 999] {
            let a = crc32_update(0, &data[..split]);
            let b = crc32_update(a, &data[split..]);
            assert_eq!(b, whole, "split at {split}");
        }
    }
}
