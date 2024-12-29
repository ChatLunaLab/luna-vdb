use kiddo::float::kdtree::KdTree;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
};

// Wasm has a 4GB memory limit. Should make sure the bucket size and capacity
// doesn't exceed it and cause stack overflow.
// More detail: https://v8.dev/blog/4gb-wasm-memory
const BUCKET_SIZE: usize = 32;

pub type Embedding = Vec<f32>;

pub type Tree = KdTree<f32, u64, 1024, BUCKET_SIZE, u16>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Index {
    pub tree: Tree,
    pub hash: HashMap<u64, String>,
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
