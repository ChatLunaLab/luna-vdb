use crate::engine::types::*;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use std::{collections::HashMap, convert::TryInto};

pub fn index<'a>(data: &'a Vec<Embedding>, ids: &'a Vec<String>) -> Index {
    let mut tree = KdTree::with_capacity(100);
    let mut doc = HashMap::new();

    for i in 0..data.len() {
        let mut embedding: Vec<f32> = data[i].clone();
        let id = ids[i].clone();

        embedding.resize(1024, 0.0);

        let hash = super::hash(&id);

        let embedding: [f32; 1024] = embedding.try_into().unwrap();

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

pub fn add<'a>(
    index: &'a mut Index,
    id: &'a String,
    query: &'a Embedding,
) -> Result<(), EngineError> {
    let hash = super::hash(id);

    if index.hash.contains_key(&hash) {
        return Err(EngineError::new(format!("Id {} already exists", id)));
    }

    let mut query: Vec<f32> = query.clone();

    query.resize(1024, 0.0);

    let query: &[f32; 1024] = &query.try_into().unwrap();

    index.hash.insert(hash, id.to_owned());
    index.tree.add(query, hash);

    Ok(())
}

pub fn remove<'a>(index: &'a mut Index, ids: &'a Vec<String>) -> Result<(), EngineError> {
    let mut embeddings: Vec<(u64, [f32; 1024])> = vec![];

    let hash_ids = ids.iter().map(|id| super::hash(id)).collect::<Vec<u64>>();

    for (vector_hash, vector) in index.tree.iter() {
        if hash_ids.contains(&vector_hash) {
            embeddings.push((vector_hash, vector.clone()));
            continue;
        }
        return Err(EngineError::new(format!("Id {} not found", index.hash.get(&vector_hash).unwrap())));
    }

    for (vector_hash, vector) in embeddings {
        index.hash.remove(&vector_hash);
        index.tree.remove(&vector, vector_hash);
    }

    Ok(())
}

pub fn size<'a>(index: &'a Index) -> u64 {
    assert_eq!(index.tree.size(), index.hash.len() as u64);
    index.hash.len() as u64
}

pub fn clear<'a>(index: &'a mut Index) {
    index.tree = Tree::new();
    index.hash = HashMap::new();
}

pub fn dump<'a>(index: &'a mut Index) -> Result<Vec<u8>, std::io::Error> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &index).unwrap();

    encoder.finish()
}

pub fn load<'a>(data: &'a Vec<u8>) -> Index {
    let mut decoder = GzDecoder::new(std::io::Cursor::new(data));

    let index: Index = bincode::deserialize_from(&mut decoder).unwrap();

    return index;
}
