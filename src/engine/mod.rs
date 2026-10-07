//! Pure-Rust vector search engine.
//!
//! No `wasm-bindgen` types appear below this module boundary, which means the
//! whole engine is unit-testable with plain `cargo test`.

pub mod codec;
pub mod crc32;
pub mod ids;
pub mod index;
pub mod kmeans;
pub mod lz4;
pub mod simd;
pub mod types;

pub use ids::{IdMap, hash_bytes, hash_str};
pub use index::{
    Engine, IndexOptions, Neighbor, SearchOutcome, add, clear, dump, dump_compressed, index, load,
    load_with_options, remove, search, search_exact, size,
};
pub use types::{Distance, Embedding, EngineError, VectorStore};
