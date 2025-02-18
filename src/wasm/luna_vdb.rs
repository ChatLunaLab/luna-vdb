use crate::engine::Embedding;
use crate::utils::set_panic_hook;
use crate::{engine, wasm::*};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LunaVDB {
    index: engine::Index,
}

#[wasm_bindgen]
impl LunaVDB {
    #[wasm_bindgen(constructor)]
    pub fn new(resource: Option<Resource>) -> LunaVDB {
        set_panic_hook();

        let resource: Resource = match resource {
            Some(res) => res,
            _ => Resource { embeddings: vec![] },
        };

        let (data, ids) = resource
            .embeddings
            .into_iter()
            .map(|res| (res.embeddings, res.id))
            .unzip();

        let index = engine::index(&data, &ids);
        LunaVDB { index }
    }

    pub fn index(&mut self, resource: Resource) {
        let (data, ids) = resource
            .embeddings
            .into_iter()
            .map(|res| (res.embeddings, res.id))
            .unzip();

        let index = engine::index(&data, &ids);
        self.index = index
    }

    pub fn search(&self, query: Embedding, k: TopK) -> SearchResult {
        engine::search(&self.index, &query, k)
    }

    pub fn add(&mut self, resource: Resource) {
        for res in resource.embeddings {
            engine::add(&mut self.index, res.id, &res.embeddings).unwrap()
        }
    }

    pub fn remove(&mut self, ids: Vec<String>) {
        engine::remove(&mut self.index, &ids).unwrap();
    }

    pub fn clear(&mut self) {
        engine::clear(&mut self.index);
    }

    pub fn size(&self) -> usize {
        engine::size(&self.index)
    }

    pub fn serialize(&mut self) -> SerializedIndex {
        engine::dump(&mut self.index).unwrap()
    }

    pub fn deserialize(index: SerializedIndex) -> LunaVDB {
        let index = engine::load(&index);

        LunaVDB { index }
    }
}
