use crate::{
    Neighbor, SearchResult,
    engine::{
        similarity::{ScoreIndex, get_cache_attr, get_distance_fn, normalize},
        types::*,
    },
};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

const DEFAULT_DISTANCE: Distance = Distance::Euclidean;
const DEFAULT_NLIST: usize = 64;
const DEFAULT_NPROBE: usize = 5;
const DEFAULT_M: usize = 8;
const DEFAULT_KSUB: usize = 256;
const KMEANS_MAX_ITER: usize = 8;
const CANDIDATE_MULTIPLIER: usize = 4;
const MAX_COARSE_TRAINING_SAMPLES: usize = 256;
const MAX_PQ_TRAINING_SAMPLES: usize = 256;
const IVF_BUILD_THRESHOLD: usize = 20_000;

pub fn index(data: &Vec<Embedding>, ids: &Vec<String>) -> Index {
    assert_eq!(data.len(), ids.len(), "embeddings and ids length mismatch");

    let distance = DEFAULT_DISTANCE;
    let dimension = data.iter().map(|vector| vector.len()).max().unwrap_or(0);

    if ids.is_empty() {
        return empty_index(distance);
    }

    let prepared_vectors: Vec<Vec<f32>> = data
        .iter()
        .map(|vector| {
            let mut prepared = prepare_vector(vector, dimension);
            if distance == Distance::Cosine {
                prepared = normalize(&prepared);
            }
            prepared
        })
        .collect();

    let embeddings = prepared_vectors
        .iter()
        .map(|vector| build_vector_data(vector.clone(), distance))
        .collect();

    let mut hash = HashMap::with_capacity(ids.len());
    let mut stored_ids = Vec::with_capacity(ids.len());

    for (idx, id) in ids.iter().enumerate() {
        hash.insert(super::hash(id), idx);
        stored_ids.push(id.clone());
    }

    let (nlist, nprobe, coarse_centroids, coarse_assignments, lists, pq, codes) =
        train_ivf_pq(&prepared_vectors, dimension);

    Index {
        embeddings,
        hash,
        ids: stored_ids,
        distance,
        dimension,
        nlist,
        nprobe,
        coarse_centroids,
        coarse_assignments,
        lists,
        pq,
        codes,
    }
}

pub fn search(index: &Index, query: &Embedding, k: usize) -> SearchResult {
    if k == 0 || index.embeddings.is_empty() || index.dimension == 0 {
        return SearchResult {
            neighbors: Vec::new(),
        };
    }

    let mut prepared_query = prepare_vector(query, index.dimension);

    if index.distance == Distance::Cosine {
        prepared_query = normalize(&prepared_query);
    }

    let query_cache = get_cache_attr(index.distance, &prepared_query);
    let distance_fn = get_distance_fn(index.distance);
    let target_k = k.min(index.embeddings.len());

    let candidate_indices = gather_candidates(index, &prepared_query, target_k);

    let mut heap = BinaryHeap::with_capacity(target_k);

    for idx in candidate_indices {
        let embedding = &index.embeddings[idx];
        let score = distance_fn(
            &embedding.vector,
            &prepared_query,
            embedding.cache_attr,
            query_cache,
        );

        if score.is_nan() {
            continue;
        }

        let entry = ScoreIndex { score, index: idx };

        if heap.len() < target_k {
            heap.push(entry);
        } else if let Some(top) = heap.peek() {
            if entry < *top {
                heap.pop();
                heap.push(entry);
            }
        }
    }

    let mut results: Vec<ScoreIndex> = heap.into_vec();
    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

    let neighbors = results
        .into_iter()
        .map(
            |ScoreIndex {
                 score,
                 index: item_index,
             }| {
                let distance = match index.distance {
                    Distance::DotProduct => -score,
                    _ => score,
                };

                Neighbor {
                    id: index.ids[item_index].clone(),
                    distance,
                }
            },
        )
        .collect();

    SearchResult { neighbors }
}

pub fn add(index: &mut Index, id: String, vector: &Embedding) -> Result<(), EngineError> {
    let hash = super::hash(&id);

    if index.hash.contains_key(&hash) {
        return Err(EngineError::new(format!("Id {} already exists", id)));
    }

    if index.dimension == 0 {
        index.dimension = vector.len();
    } else if vector.len() != index.dimension {
        return Err(EngineError::new(format!(
            "Dimension mismatch: expected {}, got {}",
            index.dimension,
            vector.len()
        )));
    }

    let mut prepared = prepare_vector(vector, index.dimension);

    if index.distance == Distance::Cosine {
        prepared = normalize(&prepared);
    }

    let vector_data = build_vector_data(prepared, index.distance);
    let position = index.embeddings.len();

    index.embeddings.push(vector_data);
    index.ids.push(id);
    index.hash.insert(hash, position);

    rebuild_ivf_pq(index);

    Ok(())
}

pub fn remove(index: &mut Index, ids: &Vec<String>) -> Result<(), EngineError> {
    for id in ids {
        let hash = super::hash(id);
        let position = index
            .hash
            .remove(&hash)
            .ok_or_else(|| EngineError::new(format!("Id {} not found", id)))?;

        index.embeddings.swap_remove(position);
        index.ids.swap_remove(position);

        if position < index.embeddings.len() {
            let moved_hash = super::hash(&index.ids[position]);
            index.hash.insert(moved_hash, position);
        }
    }

    if index.embeddings.is_empty() {
        index.dimension = 0;
        index.nlist = 0;
        index.nprobe = 0;
        index.coarse_centroids.clear();
        index.coarse_assignments.clear();
        index.lists.clear();
        index.pq = ProductQuantizer::default();
        index.codes.clear();
    } else {
        rebuild_ivf_pq(index);
    }

    Ok(())
}

pub fn size(index: &Index) -> usize {
    index.embeddings.len()
}

pub fn clear(index: &mut Index) {
    index.embeddings.clear();
    index.ids.clear();
    index.hash.clear();
    index.codes.clear();
    index.coarse_assignments.clear();
    index.coarse_centroids.clear();
    index.lists.clear();
    index.dimension = 0;
    index.nlist = 0;
    index.nprobe = 0;
    index.distance = DEFAULT_DISTANCE;
    index.pq = ProductQuantizer::default();
}

pub fn dump(index: &Index) -> Result<Vec<u8>, std::io::Error> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, index).unwrap();

    encoder.finish()
}

pub fn load(data: &Vec<u8>) -> Index {
    let mut decoder = GzDecoder::new(std::io::Cursor::new(data));

    bincode::deserialize_from::<_, Index>(&mut decoder).unwrap()
}

fn prepare_vector(vector: &[f32], dimension: usize) -> Vec<f32> {
    if dimension == 0 {
        return vector.to_vec();
    }

    if vector.len() == dimension {
        vector.to_vec()
    } else {
        let mut prepared = vec![0.0; dimension];
        let len = vector.len().min(dimension);
        prepared[..len].copy_from_slice(&vector[..len]);
        prepared
    }
}

fn build_vector_data(vector: Vec<f32>, distance: Distance) -> VectorData {
    let cache_attr = get_cache_attr(distance, &vector);

    VectorData { vector, cache_attr }
}

fn empty_index(distance: Distance) -> Index {
    Index {
        embeddings: Vec::new(),
        hash: HashMap::new(),
        ids: Vec::new(),
        distance,
        dimension: 0,
        nlist: 0,
        nprobe: 0,
        coarse_centroids: Vec::new(),
        coarse_assignments: Vec::new(),
        lists: Vec::new(),
        pq: ProductQuantizer::default(),
        codes: Vec::new(),
    }
}

fn train_ivf_pq(
    vectors: &[Vec<f32>],
    dimension: usize,
) -> (
    usize,
    usize,
    Vec<Vec<f32>>,
    Vec<usize>,
    Vec<Vec<usize>>,
    ProductQuantizer,
    Vec<Vec<u8>>,
) {
    if vectors.is_empty() || dimension == 0 || vectors.len() < IVF_BUILD_THRESHOLD {
        return (
            0,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            ProductQuantizer::default(),
            Vec::new(),
        );
    }

    let nlist = choose_nlist(vectors.len());
    let nprobe = DEFAULT_NPROBE.min(nlist).max(1);
    let coarse_training = sample_vectors(vectors, MAX_COARSE_TRAINING_SAMPLES);
    let coarse_centroids = train_kmeans(&coarse_training, nlist, dimension, KMEANS_MAX_ITER);
    let coarse_assignments = assign_to_centroids(vectors, &coarse_centroids);
    let mut lists = vec![Vec::new(); coarse_centroids.len()];

    for (idx, &cluster) in coarse_assignments.iter().enumerate() {
        lists[cluster].push(idx);
    }

    let residuals = build_residuals(vectors, &coarse_centroids, &coarse_assignments);

    let m = choose_m(dimension);
    let ksub = choose_ksub(vectors.len());
    let pq_training = sample_vectors(&residuals, MAX_PQ_TRAINING_SAMPLES);
    let pq = ProductQuantizer::train(&pq_training, dimension, m, ksub, KMEANS_MAX_ITER);
    let codes = encode_residuals(&pq, &residuals);

    (
        nlist,
        nprobe,
        coarse_centroids,
        coarse_assignments,
        lists,
        pq,
        codes,
    )
}

fn choose_nlist(count: usize) -> usize {
    if count == 0 {
        return 0;
    }

    DEFAULT_NLIST.min(count).max(1)
}

fn choose_m(dimension: usize) -> usize {
    if dimension == 0 {
        return 0;
    }

    let mut m = DEFAULT_M.min(dimension);

    while m > 1 && dimension % m != 0 {
        m -= 1;
    }

    m.max(1)
}

fn choose_ksub(count: usize) -> usize {
    if count == 0 {
        return 0;
    }

    let max_based_on_count = (count as f64).sqrt().ceil() as usize;
    let limited = max_based_on_count.max(2).min(count.max(2));
    DEFAULT_KSUB.min(limited)
}

fn train_kmeans(
    vectors: &[Vec<f32>],
    k: usize,
    dimension: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    if vectors.is_empty() || dimension == 0 {
        return vec![vec![0.0; dimension]; k.max(1)];
    }

    let effective_k = k.max(1).min(vectors.len());
    let mut centroids = Vec::with_capacity(effective_k);

    for i in 0..effective_k {
        centroids.push(vectors[i % vectors.len()].clone());
    }

    let mut assignments = vec![0usize; vectors.len()];

    for _ in 0..max_iter {
        let mut changed = false;

        for (idx, vector) in vectors.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f32::MAX;

            for (centroid_idx, centroid) in centroids.iter().enumerate() {
                let dist = l2_distance(vector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = centroid_idx;
                }
            }

            if assignments[idx] != best {
                assignments[idx] = best;
                changed = true;
            }
        }

        let mut counts = vec![0usize; effective_k];
        let mut new_centroids = vec![vec![0.0; dimension]; effective_k];

        for (vector, &cluster) in vectors.iter().zip(assignments.iter()) {
            counts[cluster] += 1;

            for (dim, value) in vector.iter().enumerate() {
                new_centroids[cluster][dim] += value;
            }
        }

        for (cluster_idx, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[cluster_idx] > 0 {
                let scale = 1.0 / counts[cluster_idx] as f32;

                for value in centroid.iter_mut() {
                    *value *= scale;
                }
            } else {
                *centroid = centroids[cluster_idx].clone();
            }
        }

        centroids = new_centroids;

        if !changed {
            break;
        }
    }

    if effective_k < k {
        let mut extended = centroids.clone();

        while extended.len() < k {
            extended.push(centroids[extended.len() % centroids.len()].clone());
        }

        return extended;
    }

    centroids
}

fn assign_to_centroids(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    let mut assignments = Vec::with_capacity(vectors.len());

    for vector in vectors {
        let mut best = 0usize;
        let mut best_dist = f32::MAX;

        for (idx, centroid) in centroids.iter().enumerate() {
            let dist = l2_distance(vector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best = idx;
            }
        }

        assignments.push(best);
    }

    assignments
}

fn build_residuals(
    vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
    assignments: &[usize],
) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .zip(assignments.iter())
        .map(|(vector, &cluster)| vector_difference(vector, &centroids[cluster]))
        .collect()
}

fn encode_residuals(pq: &ProductQuantizer, residuals: &[Vec<f32>]) -> Vec<Vec<u8>> {
    residuals
        .iter()
        .map(|residual| pq.encode(residual))
        .collect()
}

fn vector_difference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| {
        let diff = x - y;
        acc + diff * diff
    })
}

fn sample_vectors(vectors: &[Vec<f32>], max_samples: usize) -> Vec<Vec<f32>> {
    if max_samples == 0 || vectors.len() <= max_samples {
        return vectors.to_vec();
    }

    let step = (vectors.len() + max_samples - 1) / max_samples;

    vectors
        .iter()
        .step_by(step)
        .take(max_samples)
        .cloned()
        .collect()
}

fn rebuild_ivf_pq(index: &mut Index) {
    if index.embeddings.is_empty() || index.dimension == 0 {
        index.nlist = 0;
        index.nprobe = 0;
        index.coarse_centroids.clear();
        index.coarse_assignments.clear();
        index.lists.clear();
        index.pq = ProductQuantizer::default();
        index.codes.clear();
        return;
    }

    let vectors: Vec<Vec<f32>> = index
        .embeddings
        .iter()
        .map(|data| data.vector.clone())
        .collect();

    let (nlist, nprobe, coarse_centroids, coarse_assignments, lists, pq, codes) =
        train_ivf_pq(&vectors, index.dimension);

    index.nlist = nlist;
    index.nprobe = nprobe;
    index.coarse_centroids = coarse_centroids;
    index.coarse_assignments = coarse_assignments;
    index.lists = lists;
    index.pq = pq;
    index.codes = codes;
}

fn gather_candidates(index: &Index, query: &[f32], target_k: usize) -> Vec<usize> {
    if index.coarse_centroids.is_empty() || index.lists.is_empty() {
        return (0..index.embeddings.len()).collect();
    }

    let mut coarse_scores = Vec::with_capacity(index.coarse_centroids.len());

    for (idx, centroid) in index.coarse_centroids.iter().enumerate() {
        let dist = l2_distance(query, centroid);

        if dist.is_nan() {
            continue;
        }

        coarse_scores.push((idx, dist));
    }

    if coarse_scores.is_empty() {
        return (0..index.embeddings.len()).collect();
    }

    coarse_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let nprobe = index.nprobe.max(1).min(coarse_scores.len());
    let candidate_budget = (target_k * CANDIDATE_MULTIPLIER)
        .max(target_k)
        .min(index.embeddings.len());
    let mut approx_candidates = Vec::new();

    for &(cluster_idx, _) in coarse_scores.iter().take(nprobe) {
        let centroid = &index.coarse_centroids[cluster_idx];
        let residual = vector_difference(query, centroid);
        let tables = index.pq.distance_tables(&residual);

        for &embedding_idx in &index.lists[cluster_idx] {
            let approx = if tables.is_empty() {
                l2_distance(query, &index.embeddings[embedding_idx].vector)
            } else {
                index
                    .pq
                    .approximate_distance(&index.codes[embedding_idx], &tables)
            };

            if approx.is_nan() {
                continue;
            }

            approx_candidates.push((approx, embedding_idx));
        }
    }

    if approx_candidates.is_empty() {
        return (0..index.embeddings.len()).collect();
    }

    approx_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    approx_candidates
        .into_iter()
        .take(candidate_budget)
        .map(|(_, idx)| idx)
        .collect()
}

impl ProductQuantizer {
    pub fn train(
        residuals: &[Vec<f32>],
        dimension: usize,
        m: usize,
        ksub: usize,
        max_iter: usize,
    ) -> Self {
        if residuals.is_empty() || dimension == 0 || m == 0 || ksub == 0 {
            return Self::default();
        }

        let subvector_dim = dimension / m;
        let mut codebooks = Vec::with_capacity(m);

        for sub in 0..m {
            let start = sub * subvector_dim;
            let end = start + subvector_dim;
            let mut sub_vectors = Vec::with_capacity(residuals.len());

            for residual in residuals {
                sub_vectors.push(residual[start..end].to_vec());
            }

            let centroids = train_kmeans(&sub_vectors, ksub, subvector_dim, max_iter);
            codebooks.push(centroids);
        }

        let actual_ksub = codebooks.first().map(|cb| cb.len()).unwrap_or(0);

        Self {
            m,
            ksub: actual_ksub,
            subvector_dim,
            codebooks,
        }
    }

    pub fn encode(&self, residual: &[f32]) -> Vec<u8> {
        if self.m == 0 || residual.is_empty() {
            return Vec::new();
        }

        let mut code = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let start = sub * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query = &residual[start..end];

            let mut best = 0usize;
            let mut best_dist = f32::MAX;

            for (idx, centroid) in self.codebooks[sub].iter().enumerate() {
                let dist = l2_distance(query, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = idx;
                }
            }

            code.push(best as u8);
        }

        code
    }

    pub fn distance_tables(&self, residual: &[f32]) -> Vec<Vec<f32>> {
        if self.m == 0 || residual.is_empty() {
            return Vec::new();
        }

        let mut tables = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let start = sub * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query = &residual[start..end];
            let mut table = Vec::with_capacity(self.codebooks[sub].len());

            for centroid in &self.codebooks[sub] {
                table.push(l2_distance(query, centroid));
            }

            tables.push(table);
        }

        tables
    }

    pub fn approximate_distance(&self, code: &[u8], tables: &[Vec<f32>]) -> f32 {
        if self.m == 0 || tables.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        for (sub, &centroid_idx) in code.iter().enumerate() {
            if sub >= tables.len() {
                break;
            }

            let idx = centroid_idx as usize;

            if idx < tables[sub].len() {
                score += tables[sub][idx];
            }
        }

        score
    }
}
