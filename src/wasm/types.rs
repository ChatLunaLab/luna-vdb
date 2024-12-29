use serde::{Deserialize, Serialize};
use tsify::Tsify;

pub type TopK = usize;
pub type SearchResult = Vec<String>;
pub type SerializedIndex = Vec<u8>;

#[derive(Serialize, Deserialize, Debug, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct EmbeddedResource {
    pub id: String,
    pub embeddings: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Resource {
    pub embeddings: Vec<EmbeddedResource>,
}