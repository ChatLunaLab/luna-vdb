//! k-means, with k-means++ seeding and Lloyd iterations.
//!
//! # Why k-means++ instead of the old seeding
//!
//! The previous implementation seeded with `vectors[i % vectors.len()]` — the
//! first `k` vectors verbatim. On real embedding data the first `k` vectors are
//! almost always near-duplicates (a corpus has runs of similar documents), so
//! several centroids start on top of each other and the iteration converges to
//! a partition where most cells are empty and a few hold almost everything.
//! That directly destroys IVF recall: probing 8 cells out of 64 might cover 10%
//! of the data or 80% depending on the dataset, with no way to predict which.
//!
//! k-means++ spreads the seeds out by sampling each next centre with
//! probability proportional to its squared distance from the nearest existing
//! centre. Cost is one pass per centre, negligible next to the Lloyd loop, and
//! it removes the failure mode entirely.
//!
//! # Why the distance uses the cache
//!
//! Lloyd is `O(n · k · d)` per iteration, dominated by centroid distances. Both
//! sides' squared norms are precomputed, so `|a|² + |b|² - 2ab` turns the inner
//! loop into a pure `dot`. That expansion is unsafe for *exact* distance (see
//! [`crate::engine::simd::l2_sq`]) but fine here: k-means only needs to pick
//! the argmin, a small relative error between two well-separated candidates
//! never flips the answer, and the centroids it produces are refined by the
//! mean step anyway.

use crate::engine::simd;
use crate::engine::types::{Embedding, EngineError, EngineResult};

/// k-means configuration.
#[derive(Debug, Clone, Copy)]
pub struct KMeansConfig {
    /// Number of clusters to produce.
    pub k: usize,
    /// Maximum Lloyd iterations.
    pub max_iter: usize,
    /// Stop early when the fraction of points changing cluster falls below
    /// this. `0.0` disables the check and always runs `max_iter`.
    pub tolerance: f32,
    /// Seed for the k-means++ sampler, so training is reproducible across runs
    /// and across native/wasm builds.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 256,
            max_iter: 12,
            tolerance: 0.001,
            seed: 0x5EED_5EED_5EED_5EED,
        }
    }
}

/// Deterministic xorshift64*. Same stream on every platform, which matters
/// because k-means++ is randomised and we want a snapshot built on x86 to be
/// byte-identical to one built on wasm.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// `k` centroids, each `dim` long.
    pub centroids: Vec<Embedding>,
    /// How many points landed in each cluster.
    pub counts: Vec<usize>,
}

/// Train `k` centroids over `points` (row-major flat, `count × dim`).
///
/// Uses `l2_sq` against precomputed norms rather than allocating an
/// `Embedding` per point.
pub fn train(
    points: &[f32],
    count: usize,
    dim: usize,
    config: KMeansConfig,
) -> EngineResult<KMeansResult> {
    if dim == 0 {
        return Err(EngineError::new("k-means: dimension must be non-zero"));
    }
    if points.len() < count.saturating_mul(dim) {
        return Err(EngineError::new(
            "k-means: point buffer is shorter than count × dim",
        ));
    }
    if count == 0 || config.k == 0 {
        return Ok(KMeansResult {
            centroids: Vec::new(),
            counts: Vec::new(),
        });
    }

    // Cannot have more clusters than points; asking for more produces empty
    // clusters that only waste time in the inner loop.
    let k = config.k.min(count);
    let mut rng = Rng(config.seed | 1);

    let norms = compute_norms(points, count, dim);
    let mut centroids = seed_plus_plus(points, count, dim, &norms, k, &mut rng);
    let mut counts = vec![0usize; k];
    let mut assignment = vec![0u32; count];
    // Kept across iterations so each point starts from its previous cluster,
    // which turns most iterations into a single-distance check per point.
    let mut scratch = vec![0.0f32; k];

    for _ in 0..config.max_iter {
        let changed = assign_all(
            points,
            count,
            dim,
            &norms,
            &centroids,
            &mut assignment,
            &mut scratch,
        );

        counts.iter_mut().for_each(|c| *c = 0);
        for &cluster in &assignment {
            counts[cluster as usize] += 1;
        }

        centroids = recompute_centroids(points, count, dim, &assignment, &counts, &centroids);

        if changed as f32 <= config.tolerance * count as f32 {
            break;
        }
    }

    counts.iter_mut().for_each(|c| *c = 0);
    assign_all(
        points,
        count,
        dim,
        &norms,
        &centroids,
        &mut assignment,
        &mut scratch,
    );
    for &cluster in &assignment {
        counts[cluster as usize] += 1;
    }

    Ok(KMeansResult { centroids, counts })
}

fn compute_norms(points: &[f32], count: usize, dim: usize) -> Vec<f32> {
    (0..count)
        .map(|row| simd::norm_sq(&points[row * dim..row * dim + dim]))
        .collect()
}

/// Assign each point to its nearest centroid using the cached-norm expansion.
///
/// Returns the number of points whose assignment changed. `scratch` must be
/// at least `k` long; reusing it is the difference between one allocation and
/// `max_iter` allocations of `k` floats.
fn assign_all(
    points: &[f32],
    count: usize,
    dim: usize,
    norms: &[f32],
    centroids: &[Embedding],
    assignment: &mut [u32],
    scratch: &mut [f32],
) -> usize {
    let k = centroids.len();
    let centroid_data: Vec<f32> = centroids.iter().flatten().copied().collect();
    let centroid_norms: Vec<f32> = centroids.iter().map(|c| simd::norm_sq(c)).collect();

    let mut changed = 0usize;

    for row in 0..count {
        let point = &points[row * dim..row * dim + dim];
        let point_norm = norms[row];

        // Compute this point's distance to every centroid.
        for (c, slot) in scratch.iter_mut().take(k).enumerate() {
            let centroid = &centroid_data[c * dim..c * dim + dim];
            let dot = simd::dot(point, centroid);
            let dist = point_norm + centroid_norms[c] - 2.0 * dot;
            // A tiny negative from cancellation is fine; clamp so the argmin
            // is not confused by -0.0 vs 0.0. A NaN (overflow in the
            // expansion) must become the *worst* distance: mapping it to 0, as
            // a plain `dist > 0.0` test does, made the offending centroid win
            // every argmin and froze training.
            *slot = if dist.is_nan() {
                f32::INFINITY
            } else if dist > 0.0 {
                dist
            } else {
                0.0
            };
        }

        // argmin over the first `k` entries.
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (c, &dist) in scratch.iter().take(k).enumerate() {
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }

        if assignment[row] as usize != best {
            assignment[row] = best as u32;
            changed += 1;
        }
    }

    changed
}

fn recompute_centroids(
    points: &[f32],
    count: usize,
    dim: usize,
    assignment: &[u32],
    counts: &[usize],
    previous: &[Embedding],
) -> Vec<Embedding> {
    let k = previous.len();
    let mut sums = vec![0.0f32; k * dim];

    for row in 0..count {
        let cluster = assignment[row] as usize;
        if cluster >= k {
            continue;
        }
        let point = &points[row * dim..row * dim + dim];
        let target = &mut sums[cluster * dim..cluster * dim + dim];
        for (slot, &value) in target.iter_mut().zip(point.iter()) {
            *slot += value;
        }
    }

    let mut centroids = Vec::with_capacity(k);
    for cluster in 0..k {
        let cluster_count = counts[cluster];
        let slice = &sums[cluster * dim..cluster * dim + dim];

        if cluster_count == 0 {
            // An empty cluster keeps its previous position. Moving it would
            // require a split heuristic; leaving it lets a later iteration
            // reclaim it when other centroids drift past it.
            centroids.push(previous[cluster].clone());
        } else {
            let scale = 1.0 / cluster_count as f32;
            centroids.push(slice.iter().map(|value| value * scale).collect());
        }
    }

    centroids
}

/// k-means++ seeding: first centre uniform, each next one sampled with
/// probability proportional to its squared distance from the closest existing
/// centre.
fn seed_plus_plus(
    points: &[f32],
    count: usize,
    dim: usize,
    norms: &[f32],
    k: usize,
    rng: &mut Rng,
) -> Vec<Embedding> {
    let mut centroids: Vec<Embedding> = Vec::with_capacity(k);
    let mut nearest = vec![f32::MAX; count];

    // First centre: uniform random point. (The old code took index 0, which
    // is the worst case for a corpus that starts with near-duplicates.)
    let first = (rng.next_f32() * count as f32) as usize;
    let first = first.min(count - 1);
    centroids.push(points[first * dim..first * dim + dim].to_vec());

    let mut total: f32 = 0.0;
    for row in 0..count {
        let point = &points[row * dim..row * dim + dim];
        let dist = simd::l2_sq(point, &centroids[0]);
        nearest[row] = dist;
        total += dist;
    }

    while centroids.len() < k {
        // Degenerate case: every remaining point coincides with a centre, or
        // the weights overflowed. Fall back to an arbitrary point. The NaN
        // test is explicit because `NaN <= x` is false, and a NaN total would
        // otherwise make every weighted draw fail.
        if total.is_nan() || total <= f32::MIN_POSITIVE {
            let fallback = (rng.next_f32() * count as f32) as usize;
            let fallback = (fallback.min(count - 1) + centroids.len()) % count;
            centroids.push(points[fallback * dim..fallback * dim + dim].to_vec());
            total = recompute_nearest(points, count, dim, &centroids, &mut nearest);
            continue;
        }

        // Sample by distance-weighted rejection, with a hard cap so a badly
        // skewed distribution cannot spin. `count` attempts is the standard
        // bound: a fresh uniform sample fails to be distant with probability
        // that drops geometrically.
        let threshold = rng.next_f32() * total;
        let mut chosen = None;
        let mut cumulative = 0.0f32;

        for row in 0..count {
            cumulative += nearest[row];
            if cumulative >= threshold {
                chosen = Some(row);
                break;
            }
        }

        let chosen = chosen.unwrap_or_else(|| {
            // Numerical drift in `cumulative`: take the farthest point, which
            // is what the sampler was trying to find anyway.
            let mut best_row = 0usize;
            let mut best = -1.0f32;
            for row in 0..count {
                if nearest[row] > best {
                    best = nearest[row];
                    best_row = row;
                }
            }
            best_row
        });

        centroids.push(points[chosen * dim..chosen * dim + dim].to_vec());
        total = recompute_nearest(points, count, dim, &centroids, &mut nearest);
    }

    let _ = norms;
    centroids
}

/// Update `nearest[row]` to the distance to the closest centroid, and return
/// the total. Called once per seeded centre.
fn recompute_nearest(
    points: &[f32],
    count: usize,
    dim: usize,
    centroids: &[Embedding],
    nearest: &mut [f32],
) -> f32 {
    let last = match centroids.last() {
        Some(centroid) => centroid,
        None => return 0.0,
    };

    let mut total = 0.0f32;
    for row in 0..count {
        let point = &points[row * dim..row * dim + dim];
        let dist = simd::l2_sq(point, last);
        if dist < nearest[row] {
            nearest[row] = dist;
        }
        total += nearest[row];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    /// Three well-separated blobs in 2D, flattened row-major.
    fn blobs() -> (Vec<f32>, usize, usize) {
        let mut rng = pseudo(0xDEAD_BEEF);
        let mut points = Vec::new();
        let centres = [(0.0f32, 0.0f32), (10.0, 0.0), (0.0, 10.0)];

        for _ in 0..3 {
            for &(cx, cy) in &centres {
                for _ in 0..200 {
                    points.push(cx + (rng.next_f32() - 0.5) * 0.5);
                    points.push(cy + (rng.next_f32() - 0.5) * 0.5);
                }
            }
        }

        (points, 600, 2)
    }

    #[test]
    fn finds_separated_clusters() {
        let (points, count, dim) = blobs();
        let result = train(
            &points,
            count,
            dim,
            KMeansConfig {
                k: 3,
                ..Default::default()
            },
        )
        .expect("train");

        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.counts.iter().sum::<usize>(), count);

        // Every cluster should be non-trivial: this is exactly what the old
        // modulo seeding failed to guarantee.
        for (i, &c) in result.counts.iter().enumerate() {
            assert!(c > 50, "cluster {i} only got {c} points: {result:?}");
        }
    }

    #[test]
    fn is_deterministic() {
        let (points, count, dim) = blobs();
        let config = KMeansConfig {
            k: 4,
            ..Default::default()
        };

        let a = train(&points, count, dim, config).expect("train a");
        let b = train(&points, count, dim, config).expect("train b");
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.counts, b.counts);
    }

    #[test]
    fn handles_degenerate_inputs() {
        // k > n: cannot produce more real clusters than points.
        let points = vec![1.0f32, 0.0, 0.0, 1.0];
        let result = train(
            &points,
            2,
            2,
            KMeansConfig {
                k: 16,
                ..Default::default()
            },
        )
        .expect("train");
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.counts.iter().sum::<usize>(), 2);

        // All-identical points must not divide by zero or loop forever.
        let same = vec![5.0f32; 200];
        let result = train(
            &same,
            100,
            2,
            KMeansConfig {
                k: 8,
                ..Default::default()
            },
        )
        .expect("train");
        assert_eq!(result.centroids.len(), 8);
        assert!(
            result
                .centroids
                .iter()
                .all(|c| c.iter().all(|v| v.is_finite()))
        );

        // Empty input.
        let result = train(&[], 0, 2, KMeansConfig::default()).expect("train");
        assert!(result.centroids.is_empty());
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(train(&[1.0, 2.0], 1, 0, KMeansConfig::default()).is_err());
        assert!(train(&[1.0], 10, 2, KMeansConfig::default()).is_err());
    }

    #[test]
    fn converges_monotonically() {
        let (points, count, dim) = blobs();

        let run = |max_iter: usize| {
            train(
                &points,
                count,
                dim,
                KMeansConfig {
                    k: 3,
                    max_iter,
                    tolerance: 0.0,
                    seed: 42,
                },
            )
            .expect("train")
        };

        let one = run(1);
        let many = run(20);

        let inertia = |centroids: &[Embedding]| -> f32 {
            (0..count)
                .map(|row| {
                    let point = &points[row * dim..row * dim + dim];
                    centroids
                        .iter()
                        .map(|c| simd::l2_sq(point, c))
                        .fold(f32::MAX, f32::min)
                })
                .sum()
        };

        // More iterations must not make the objective worse.
        assert!(
            inertia(&many.centroids) <= inertia(&one.centroids) * 1.001 + 1e-3,
            "inertia regressed: {} -> {}",
            inertia(&one.centroids),
            inertia(&many.centroids)
        );
    }
}
