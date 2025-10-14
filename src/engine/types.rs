use serde::{Deserialize, Serialize};
use std::error::Error;
use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
};

pub type Embedding = Vec<f32>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Distance {
    #[serde(rename = "euclidean")]
    Euclidean,
    #[serde(rename = "cosine")]
    Cosine,
    #[serde(rename = "dot")]
    DotProduct,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorData {
    pub vector: Vec<f32>,
    pub cache_attr: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProductQuantizer {
    pub m: usize,
    pub ksub: usize,
    pub subvector_dim: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl Default for ProductQuantizer {
    fn default() -> Self {
        Self {
            m: 0,
            ksub: 0,
            subvector_dim: 0,
            codebooks: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Index {
    pub embeddings: Vec<VectorData>,
    pub hash: HashMap<u64, usize>,
    pub ids: Vec<String>,
    pub distance: Distance,
    pub dimension: usize,
    #[serde(default)]
    pub nlist: usize,
    #[serde(default)]
    pub nprobe: usize,
    #[serde(default)]
    pub coarse_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    pub coarse_assignments: Vec<usize>,
    #[serde(default)]
    pub lists: Vec<Vec<usize>>,
    #[serde(default)]
    pub pq: ProductQuantizer,
    #[serde(default)]
    pub codes: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub struct EngineError {
    pub message: String,
}

impl EngineError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.message)
    }
}

impl Error for EngineError {}
