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
//! loading; new ones are written in `LVD2` and are roughly 3x faster to read.

use crate::engine::crc32::crc32;
use crate::engine::lz4;
use crate::engine::types::{Distance, EngineError, EngineResult};

/// `LVD2`, little-endian on the wire.
pub const MAGIC: [u8; 4] = *b"LVD2";
/// Current format version. Bump on any layout change.
pub const FORMAT_VERSION: u16 = 2;
/// Fixed header size: magic, version, flags, raw length, payload CRC.
pub const HEADER_LEN: usize = 16;

const FLAG_COMPRESSED: u16 = 1 << 0;

/// Refuse to allocate more than this from a header. A corrupt length field is
/// the single most likely way to turn a bad file into an OOM kill, so the
/// ceiling is enforced before any allocation happens.
///
/// This is a `u64`, not a `usize`, and that is not cosmetic. Written as
/// `8 << 30` in `usize`, the constant **overflows to zero on wasm32**, where
/// `usize` is 32 bits — so every snapshot of any size was rejected as "too
/// large to serialise", and every restore of any real file failed. Native
/// builds (64-bit) never saw it. Anything compared against this must widen
/// through [`max_raw_len`] rather than casting back down.
const MAX_RAW_LEN: u64 = 8 << 30; // 8 GiB

/// The ceiling as a `usize`, saturated to what this target can actually index.
///
/// On a 32-bit target the real limit is `usize::MAX`, not 8 GiB: a snapshot
/// that a wasm module could hold is bounded by its address space, and
/// saturating here means the check never rejects a file for a reason the
/// platform already guarantees.
fn max_raw_len() -> usize {
    MAX_RAW_LEN.min(usize::MAX as u64) as usize
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
    out.extend_from_slice(&crc32(&stored).to_le_bytes());
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
    if raw_len > MAX_RAW_LEN || raw_len > max_raw_len() as u64 {
        return Err(EngineError::corrupt(format!(
            "declared payload of {raw_len} bytes exceeds the {} byte limit",
            max_raw_len()
        )));
    }
    let raw_len = raw_len as usize;

    let stored = reader.take(reader.remaining())?;

    // Verify the checksum *before* parsing: it turns "random garbage that
    // happens to have plausible lengths" into one clear error.
    let actual_crc = crc32(stored);
    if actual_crc != expected_crc {
        return Err(EngineError::corrupt(format!(
            "checksum mismatch: header says {expected_crc:#010x}, payload is {actual_crc:#010x}"
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

    let mut ids = Vec::with_capacity(count as usize);
    for _ in 0..count {
        ids.push(reader.string()?);
    }

    let nlist = reader.u32()?;
    let nprobe = reader.u32()?;

    if nlist as usize > count as usize && count > 0 {
        return Err(EngineError::corrupt(format!(
            "nlist {nlist} exceeds the vector count {count}"
        )));
    }

    let centroids = reader.floats(
        (nlist as usize)
            .checked_mul(dim as usize)
            .ok_or_else(|| EngineError::corrupt("centroid count overflow"))?,
    )?;

    let mut lists = Vec::with_capacity(nlist as usize);
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

        let codebook_len = (m as usize)
            .checked_mul(ksub as usize)
            .and_then(|v| v.checked_mul(subvector_dim as usize))
            .ok_or_else(|| EngineError::corrupt("codebook size overflow"))?;

        let codebooks = reader.floats(codebook_len)?;
        let code_len = reader.u32()? as usize;

        if code_len != count as usize * m as usize {
            return Err(EngineError::corrupt(format!(
                "code buffer is {code_len} bytes, expected {}",
                count as usize * m as usize
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
    use std::io::Read;

    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut raw = Vec::new();

    decoder
        .read_to_end(&mut raw)
        .map_err(|error| EngineError::corrupt(format!("legacy gzip: {error}")))?;

    if let Ok(index) = bincode::deserialize::<legacy::LegacyIndex>(&raw) {
        return Ok(convert_legacy(index));
    }

    if let Ok(index) = bincode::deserialize::<legacy::LegacyIndexPreIvf>(&raw) {
        return Ok(convert_legacy_pre_ivf(index));
    }

    Err(EngineError::corrupt(
        "legacy snapshot did not match any known pre-0.1 layout",
    ))
}

fn legacy_distance(distance: &legacy::LegacyDistance) -> Distance {
    match distance {
        legacy::LegacyDistance::Euclidean => Distance::Euclidean,
        legacy::LegacyDistance::Cosine => Distance::Cosine,
        legacy::LegacyDistance::DotProduct => Distance::DotProduct,
    }
}

fn convert_legacy(index: legacy::LegacyIndex) -> Snapshot {
    let dim = index.dimension;
    let count = index.embeddings.len();

    let mut data = Vec::with_capacity(count * dim);
    let mut cache = Vec::with_capacity(count);
    for embedding in &index.embeddings {
        data.extend_from_slice(&embedding.vector);
        cache.push(embedding.cache_attr);
    }

    Snapshot {
        distance: legacy_distance(&index.distance),
        dim: dim as u32,
        count: count as u32,
        data,
        cache,
        ids: index.ids,
        // Everything IVF-related is intentionally discarded; the engine
        // retrains on load.
        nlist: 0,
        nprobe: 0,
        centroids: Vec::new(),
        lists: Vec::new(),
        pq: None,
        codes: Vec::new(),
    }
}

fn convert_legacy_pre_ivf(index: legacy::LegacyIndexPreIvf) -> Snapshot {
    let dim = index.dimension;
    let count = index.embeddings.len();

    let mut data = Vec::with_capacity(count * dim);
    let mut cache = Vec::with_capacity(count);
    for embedding in &index.embeddings {
        data.extend_from_slice(&embedding.vector);
        cache.push(embedding.cache_attr);
    }

    Snapshot {
        distance: legacy_distance(&index.distance),
        dim: dim as u32,
        count: count as u32,
        data,
        cache,
        ids: index.ids,
        nlist: 0,
        nprobe: 0,
        centroids: Vec::new(),
        lists: Vec::new(),
        pq: None,
        codes: Vec::new(),
    }
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

    #[test]
    fn rejects_implausible_lengths() {
        // Hand-build a header claiming 4 GiB of vectors, with no payload
        // behind it. The reader must reject on the length check rather than
        // trying to allocate 4 GiB.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&(4u32 << 30).to_le_bytes()); // raw_len
        bytes.extend_from_slice(&0u32.to_le_bytes()); // crc
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(read(&bytes).is_err());
    }
}
