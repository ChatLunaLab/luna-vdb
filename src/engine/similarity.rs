use crate::engine::types::Distance;
use std::cmp::Ordering;

pub fn get_cache_attr(metric: Distance, vec: &[f32]) -> f32 {
    match metric {
        Distance::DotProduct => 0.0,
        Distance::Euclidean => vec.iter().map(|&x| x.mul_add(x, 0.0)).sum(),
        Distance::Cosine => vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt(),
    }
}

pub fn get_distance_fn(
    metric: Distance,
) -> impl Fn(&[f32], &[f32], f32, f32) -> f32 {
    match metric {
        Distance::Euclidean => euclidean_distance,
        Distance::Cosine => cosine_distance,
        Distance::DotProduct => dot_product_distance,
    }
}

fn euclidean_distance(a: &[f32], b: &[f32], a_sum_squares: f32, b_sum_squares: f32) -> f32 {
    let dot = a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y);
    (a_sum_squares + b_sum_squares - 2.0 * dot).max(0.0).sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32], a_magnitude: f32, b_magnitude: f32) -> f32 {
    let denom = a_magnitude * b_magnitude;

    if denom <= std::f32::EPSILON {
        1.0
    } else {
        1.0 - a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y) / denom
    }
}

fn dot_product_distance(a: &[f32], b: &[f32], _: f32, _: f32) -> f32 {
    -a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y)
}

pub fn normalize(vec: &[f32]) -> Vec<f32> {
    let magnitude = vec.iter().fold(0.0, |acc, &val| val.mul_add(val, acc)).sqrt();

    if magnitude > std::f32::EPSILON {
        vec.iter().map(|&val| val / magnitude).collect()
    } else {
        vec.to_vec()
    }
}

pub struct ScoreIndex {
    pub score: f32,
    pub index: usize,
}

impl PartialEq for ScoreIndex {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for ScoreIndex {}

impl PartialOrd for ScoreIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoreIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
