//! Search benchmark. Native `cargo bench`, no wasm.
//!
//! Run with:
//! ```text
//! cargo bench --bench search
//! ```
//!
//! It reports, for each corpus size:
//!
//! * `flat` — brute force over the flat arena with SIMD kernels. This is the
//!   reference: its recall is 1.0 by definition, because the ground truth is
//!   computed with it.
//! * `default` — `Engine::search` with default options: approximate, with the
//!   `nprobe` the index calibrated for recall@10 of 0.95. Its recall should
//!   print at 0.9 or above.
//! * `exact` — the same index with `approximate: false`: the IVF-pruned
//!   exact search. Its recall must print as 1.0; anything else is a bug.
//!
//! The corpus is uniform random, which has no cluster structure — the worst
//! case for any partitioning index, so both gain least here. See
//! `compare/src/sweep.rs` for clustered data, and `compare/src/main.rs` for
//! the comparison against the pre-rewrite engine.
//!
//! `harness = false` (see `Cargo.toml`) because these are throughput
//! measurements with a fixed iteration count, not statistically-tested
//! microbenchmarks — `std::hint::black_box` plus a warm-up is sufficient and
//! keeps the harness dependency-free.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use luna_vdb::engine::{Engine, IndexOptions};

/// Dimensions to report over. 1536 is a typical embedding size (OpenAI
/// `text-embedding-3-small`-class), 384 and 768 cover the small local models.
const DIMS: [usize; 2] = [384, 1536];

/// Corpus sizes. 1k is the "no index at all" case, 100k is where an index
/// genuinely pays.
const SIZES: [usize; 4] = [1_000, 10_000, 50_000, 200_000];

const QUERIES: usize = 50;
const K: usize = 10;

/// SplitMix64 seed mixing — see the note in the engine tests for why `| 1`
/// alone is not enough.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = splitmix(seed) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn corpus(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut next = rng(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect()
}

fn ids_for(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("v{i}")).collect()
}

struct Measurement {
    mean: Duration,
    p50: Duration,
    recall: f32,
}

/// Time `QUERIES` searches, warm up first so allocator growth and branch
/// predictor training do not land in the measurement.
fn measure<F>(mut run: F, truth: &[HashSet<String>]) -> Measurement
where
    F: FnMut(usize, usize) -> Vec<String>,
{
    // Warm-up: 5 queries, discarded.
    for query in 0..5.min(QUERIES) {
        black_box(run(query, K));
    }

    let mut samples = Vec::with_capacity(QUERIES);
    let mut hits = 0usize;

    for (query, expected) in truth.iter().enumerate() {
        let started = Instant::now();
        let results = black_box(run(query, K));
        samples.push(started.elapsed());

        hits += results
            .iter()
            .filter(|id| expected.contains(id.as_str()))
            .count();
    }

    samples.sort_unstable();
    let total: Duration = samples.iter().sum();

    Measurement {
        mean: total / QUERIES as u32,
        p50: samples[QUERIES / 2],
        recall: hits as f32 / (QUERIES * K) as f32,
    }
}

fn main() {
    println!("luna-vdb search benchmark");
    println!(
        "kernel: {}, queries: {QUERIES}, k: {K}\n",
        luna_vdb::simd_backend()
    );

    for &dim in &DIMS {
        if dim % 8 != 0 {
            continue;
        }

        for &size in &SIZES {
            if size < 1_000 {
                continue;
            }

            let data = corpus(size, dim, 0xC0FFEE ^ size as u64);
            let ids = ids_for(size);
            let queries: Vec<Vec<f32>> = corpus(QUERIES, dim, 0xBEEF ^ size as u64);

            let engine = match Engine::build(&data, &ids, IndexOptions::default()) {
                Ok(engine) => engine,
                Err(error) => {
                    println!("dim {dim} size {size}: build failed: {error}");
                    continue;
                }
            };

            // Ground truth from the flat scan.
            let truth: Vec<HashSet<String>> = queries
                .iter()
                .map(|query| {
                    engine
                        .search_exact(query, K)
                        .neighbors
                        .into_iter()
                        .map(|neighbor| neighbor.id)
                        .collect()
                })
                .collect();

            let flat = measure(
                |query, k| {
                    engine
                        .search_exact(&queries[query], k)
                        .neighbors
                        .into_iter()
                        .map(|neighbor| neighbor.id)
                        .collect()
                },
                &truth,
            );

            let default = measure(
                |query, k| {
                    engine
                        .search(&queries[query], k)
                        .neighbors
                        .into_iter()
                        .map(|neighbor| neighbor.id)
                        .collect()
                },
                &truth,
            );

            let mut exact_engine = engine.clone();
            exact_engine.set_approximate(false);
            let exact = measure(
                |query, k| {
                    exact_engine
                        .search(&queries[query], k)
                        .neighbors
                        .into_iter()
                        .map(|neighbor| neighbor.id)
                        .collect()
                },
                &truth,
            );

            let speedup = |m: &Measurement| {
                flat.mean.as_secs_f64() / m.mean.as_secs_f64().max(f64::MIN_POSITIVE)
            };

            println!(
                "dim {dim:>5}  n {size:>7}  nlist {:>5}  nprobe {:>4}",
                engine.nlist(),
                engine.nprobe(),
            );
            println!(
                "    flat      {:>10.3} ms  p50 {:>10.3} ms  recall {:.3}",
                ms(flat.mean),
                ms(flat.p50),
                flat.recall,
            );
            println!(
                "    default   {:>10.3} ms  p50 {:>10.3} ms  recall {:.3}   {:>6.1}x vs flat",
                ms(default.mean),
                ms(default.p50),
                default.recall,
                speedup(&default),
            );
            println!(
                "    exact     {:>10.3} ms  p50 {:>10.3} ms  recall {:.3}   {:>6.1}x vs flat",
                ms(exact.mean),
                ms(exact.p50),
                exact.recall,
                speedup(&exact),
            );

            let outcome = engine.search(&queries[0], K);
            println!(
                "    default scanned {} of {} rows ({:.1}%) in {} cells",
                outcome.candidates_scored,
                size,
                100.0 * outcome.candidates_scored as f64 / size as f64,
                outcome.cells_probed,
            );
            println!();
        }
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
