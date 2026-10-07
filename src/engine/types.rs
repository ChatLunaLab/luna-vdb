//! Core value types for the engine.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// A dense f32 vector.
pub type Embedding = Vec<f32>;

/// Result alias used throughout the engine.
pub type EngineResult<T> = Result<T, EngineError>;

/// Distance metric. The numeric discriminants are part of the on-disk format —
/// never renumber them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum Distance {
    #[default]
    Euclidean = 0,
    Cosine = 1,
    DotProduct = 2,
}

impl Distance {
    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Inverse of [`Distance::to_u8`]. Returns `None` for unknown codes so a
    /// corrupt snapshot becomes an error rather than a silent fallback.
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Distance::Euclidean),
            1 => Some(Distance::Cosine),
            2 => Some(Distance::DotProduct),
            _ => None,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "euclidean" | "l2" => Some(Distance::Euclidean),
            "cosine" => Some(Distance::Cosine),
            "dot" | "dotproduct" | "ip" | "inner_product" => Some(Distance::DotProduct),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Distance::Euclidean => "euclidean",
            Distance::Cosine => "cosine",
            Distance::DotProduct => "dot",
        }
    }
}

/// Every failure the engine can report. There is deliberately no `panic!` on
/// the user-input path: JS callers get one of these converted into a catchable
/// exception, never a poisoned wasm instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    pub message: String,
}

impl EngineError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn dimension_mismatch(expected: usize, got: usize) -> Self {
        Self::new(format!(
            "dimension mismatch: index expects {expected}, got {got}"
        ))
    }

    pub fn duplicate_id(id: &str) -> Self {
        Self::new(format!("id already exists: {id}"))
    }

    pub fn missing_id(id: &str) -> Self {
        Self::new(format!("id not found: {id}"))
    }

    pub fn corrupt(detail: impl Into<String>) -> Self {
        Self::new(format!("corrupt snapshot: {}", detail.into()))
    }
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EngineError {}

/// Row-major flat vector arena.
///
/// Replaces the old `Vec<VectorData>` (one heap allocation per vector). Two
/// reasons this matters more than it looks:
///
/// 1. **Cache.** A scan touches every row in order; with one allocation per
///    vector the CPU stalls on a pointer chase per vector, and the prefetcher
///    cannot help. One contiguous buffer keeps the hardware prefetcher running
///    at full width.
/// 2. **SIMD.** `dot`/`l2_sq` need a contiguous `&[f32]`. With `Vec<Vec<f32>>`
///    every call is `len < 8` and falls into the scalar tail, so the vector
///    units never engage.
#[derive(Debug, Clone, Default)]
pub struct VectorStore {
    data: Vec<f32>,
    /// `sum(v_i^2)` per row, precomputed once. Euclidean reuses it to avoid a
    /// second pass over the row; Cosine keeps it as the magnitude.
    cache: Vec<f32>,
    dim: usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            cache: Vec::new(),
            dim,
        }
    }

    pub fn with_capacity(dim: usize, rows: usize) -> Self {
        Self {
            data: Vec::with_capacity(dim.saturating_mul(rows)),
            cache: Vec::with_capacity(rows),
            dim,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of *rows*, including rows whose vector has been logically deleted.
    #[inline]
    pub fn rows(&self) -> usize {
        self.cache.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// # Panics
    /// Never in practice — but callers must only pass indices `< rows()`.
    #[inline]
    pub fn row(&self, index: usize) -> &[f32] {
        let start = index * self.dim;
        // `get` instead of slicing: a bad index yields an empty slice, which
        // every distance function tolerates, rather than a slice-index panic.
        match self.data.get(start..start + self.dim) {
            Some(row) => row,
            None => &[],
        }
    }

    #[inline]
    pub fn cache_attr(&self, index: usize) -> f32 {
        self.cache.get(index).copied().unwrap_or(0.0)
    }

    /// Append one row. Returns its row index.
    ///
    /// If `vector` is not exactly `dim` long it is zero-padded or truncated —
    /// this mirrors the old `prepare_vector` behaviour so a client that sends a
    /// slightly-off dimension keeps working instead of throwing.
    pub fn push(&mut self, vector: &[f32]) -> usize {
        let index = self.rows();
        let start = self.data.len();
        self.data.resize(start + self.dim, 0.0);

        if self.dim > 0 {
            let copy = vector.len().min(self.dim);
            self.data[start..start + copy].copy_from_slice(&vector[..copy]);
        }

        let row = &self.data[start..start + self.dim];
        self.cache.push(crate::engine::simd::norm_sq(row));
        index
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.cache.clear();
    }

    /// Drop every row listed in `keep` (a sorted, deduplicated list of row
    /// indices), preserving order. Used by compaction.
    pub fn retain_rows(&mut self, keep: &[u32]) {
        if keep.len() == self.rows() {
            return;
        }

        let dim = self.dim;
        let mut data = Vec::with_capacity(keep.len().saturating_mul(dim));
        let mut cache = Vec::with_capacity(keep.len());

        for &row in keep {
            let row = row as usize;
            let start = row * dim;
            if let Some(slice) = self.data.get(start..start + dim) {
                data.extend_from_slice(slice);
            } else {
                data.resize(data.len() + dim, 0.0);
            }
            cache.push(self.cache.get(row).copied().unwrap_or(0.0));
        }

        self.data = data;
        self.cache = cache;
    }

    /// Byte size of the backing buffers, for `stats()`.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>() + self.cache.len() * std::mem::size_of::<f32>()
    }
}
