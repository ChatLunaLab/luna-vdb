use crate::engine::types::*;
use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use std::{collections::HashMap, convert::TryInto};

const BUFFER_LEN: usize = 1024 * 1024 * 1024;
const SCRATCH_LEN: usize = 1024 * 1024;

pub fn index<'a>(data: &'a Vec<Embedding>, ids: &'a Vec<String>) -> Index {
    let data_vec: Vec<([f32; 1024], u64, String)> = data
        .iter()
        .zip(ids.iter())
        .map(|(embedding, id)| {
            let mut embedding: Vec<f32> = embedding.to_owned();

            embedding.resize(1024, 0.0);

            let hash = super::hash(id);

            let embedding: [f32; 1024] = embedding.try_into().unwrap();

            (embedding, hash, id.to_owned())
        })
        .collect();

    let mut tree = KdTree::new();
    let mut doc = HashMap::new();

    for (embedding, hash, id) in data_vec {
        tree.add(&embedding, hash);
        doc.insert(hash, id);
    }
    Index { tree, hash: doc }
}

pub fn search<'a>(index: &'a Index, query: &'a Embedding, k: usize) -> Vec<String> {
    let mut query: Vec<f32> = query.clone();

    query.resize(1024, 0.0);

    let query: &[f32; 1024] = &query.try_into().unwrap();

    let neighbors = index.tree.nearest_n::<SquaredEuclidean>(query, k);

    let mut result: Vec<String> = vec![];

    for neighbor in &neighbors {
        let id = index.hash.get(&neighbor.item);

        if id.is_some() {
            result.push(id.unwrap().to_owned());
        }
    }

    result
}

pub fn add<'a>(index: &'a mut Index, id: &'a String, query: &'a Embedding) {
    let mut query: Vec<f32> = query.clone();

    query.resize(1024, 0.0);

    let query: &[f32; 1024] = &query.try_into().unwrap();

    let hash = super::hash(id);

    index.hash.insert(hash, id.to_owned());
    index.tree.add(query, hash);
}

pub fn remove<'a>(index: &'a mut Index, id: &'a String) -> Result<(), EngineError> {
    let hash = super::hash(id);

    let hit = index.hash.remove(&hash);

    if hit.is_none() {
        return Err(EngineError::new(format!("Id {} not found", id)));
    }

    let mut embedding: Option<[f32; 1024]> = None;

    for (vector_hash, vector) in index.tree.iter() {
        if vector_hash == hash {
            embedding = Some(vector);
            break;
        }
    }

    if let Some(embedding) = embedding {
        index.tree.remove(&embedding, hash);
        Ok(())
    } else {
        Err(EngineError::new(format!("Id {} not found", id)))
    }
}

pub fn size<'a>(index: &'a Index) -> usize {
    index.hash.len()
}

pub fn clear<'a>(index: &'a mut Index) {
    index.tree = Tree::new();
    index.hash = HashMap::new();
}

pub fn dump<'a>(index: &'a mut Index) -> String {
    // Wasm64 is experimental, so we can't use it yet.
   /*  let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);

    unsafe { serialize_scratch.set_len(SCRATCH_LEN) };
    serialize_buffer.clear();

    let mut serializer = CompositeSerializer::new(
        AlignedSerializer::new(&mut serialize_buffer),
        BufferScratch::new(&mut serialize_scratch),
        Infallible,
    );

    serializer
        .serialize_value(index)
        .expect("Could not serialize with rkyv");

    let buf = serializer.into_serializer().into_inner();

    buf.to_vec() */

    return serde_json::to_string(index).unwrap()
}

pub fn load<'a>(data: &'a String) -> Index {
    // Wasm64 is experimental, so we can't use it yet.
   /*  let archived = unsafe { rkyv::archived_root::<Index>(&data[..]) };

    // And you can always deserialize back to the original type
    let deserialized = archived.deserialize(&mut rkyv::Infallible).unwrap();
    return deserialized; */

    serde_json::from_str::<Index>(data).unwrap()
}
