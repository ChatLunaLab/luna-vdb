use crate::{engine::types::*, Neighbor, SearchResult};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use std::{collections::HashMap, convert::TryInto};

pub fn index(data: &Vec<Embedding>, ids: &Vec<String>) -> Index {
    let mut tree = KdTree::with_capacity(100);
    let mut doc = HashMap::new();

    for i in 0..data.len() {
        let mut embedding: Vec<f32> = data[i].to_owned();
        let id = ids[i].to_owned();

        if embedding.len() != 1024 {
            embedding.resize(1024, 0.0);
        }

        let hash = super::hash(&id);

        let embedding: [f32; 1024] = embedding.try_into().unwrap();

        tree.add(&embedding, hash);
        doc.insert(hash, id);
    }

    Index { tree, hash: doc }
}

pub fn search(index: &Index, query: &Embedding, k: usize) -> SearchResult {
    let mut query: Vec<f32> = query.to_owned();

    if query.len() != 1024 {
        query.resize(1024, 0.0);
    }

    let query: &[f32; 1024] = &query.try_into().unwrap();


    let neighbors = index.tree.nearest_n::<SquaredEuclidean>(&query, k);

    let mut result: Vec<Neighbor> = vec![];

    for neighbor in &neighbors {
        let id = index.hash.get(&neighbor.item);

        if id.is_some() {
            result.push(Neighbor {
                id: id.unwrap().to_owned(),
                distance: neighbor.distance,
            });
        }
    }

    SearchResult { neighbors: result }
}

pub fn add(index: &mut Index, id: String, query: &Embedding) -> Result<(), EngineError> {
    let hash = super::hash(&id);

    if index.hash.contains_key(&hash) {
        return Err(EngineError::new(format!("Id {} already exists", id)));
    }

    let mut query: Vec<f32> = query.to_owned();

    if query.len() != 1024 {
        query.resize(1024, 0.0);
    }

    let query: &[f32; 1024] = &query.try_into().unwrap();


    index.hash.insert(hash, id);
    index.tree.add(query, hash);

    Ok(())
}

pub fn remove(index: &mut Index, ids: &Vec<String>) -> Result<(), EngineError> {
    let mut embeddings: Vec<(u64, [f32; 1024])> = vec![];

    let hash_ids = ids.iter().map(|id| super::hash(id)).collect::<Vec<u64>>();

    for (vector_hash, vector) in index.tree.iter() {
        if hash_ids.contains(&vector_hash) {
            embeddings.push((vector_hash, vector));
            continue;
        }
    }

    if hash_ids.len() != embeddings.len() {
        let not_found_ids = hash_ids
            .iter()
            .filter(|id| {
                !embeddings
                    .iter()
                    .any(|(vector_hash, _)| *vector_hash == **id)
            })
            .map(|id| index.hash.get(id).unwrap().to_owned())
            .collect::<Vec<String>>();

        return Err(EngineError::new(format!(
            "The ids {} not found",
            not_found_ids.join(",")
        )));
    }

    for (vector_hash, vector) in embeddings {
        index.hash.remove(&vector_hash);
        index.tree.remove(&vector, vector_hash);
    }

    Ok(())
}

pub fn size(index: &Index) -> usize {
    assert_eq!(index.tree.size(), index.hash.len() as u64);
    index.hash.len()
}

pub fn clear(index: &mut Index) {
    index.tree = Tree::new();
    index.hash = HashMap::new();
}

pub fn dump(index: &mut Index) -> Result<Vec<u8>, std::io::Error> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &index).unwrap();

    encoder.finish()
}

pub fn load(data: &Vec<u8>) -> Index {
    let mut decoder = GzDecoder::new(std::io::Cursor::new(data));

    bincode::deserialize_from::<_, Index>(&mut decoder).unwrap()
}
