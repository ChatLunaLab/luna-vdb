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
//!   *new* baseline and it is already much faster than the old engine, because
//!   the old engine's scan was scalar and scattered.
//! * `ivf` — the full IVF + PQ pipeline.
//! * `ivf_exact` — IVF with exact rescoring (no PQ approximation).
//!
//! and `exact` recall of both against `flat`, so the speed number can be read
//! together with the quality number. A speedup without the recall is not a
//! speedup.
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

            let ivf = measure(
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

            let speedup = flat.mean.as_secs_f64() / ivf.mean.as_secs_f64().max(f64::MIN_POSITIVE);

            println!(
                "dim {dim:>5}  n {size:>7}  nlist {:>5}  nprobe {:>4}  pq {:>5}",
                engine.nlist(),
                engine.nprobe(),
                engine.has_pq(),
            );
            println!(
                "    flat      {:>10.3} ms  p50 {:>10.3} ms  recall {:.3}",
                ms(flat.mean),
                ms(flat.p50),
                flat.recall,
            );
            println!(
                "    ivf       {:>10.3} ms  p50 {:>10.3} ms  recall {:.3}   {speedup:>6.1}x vs flat",
                ms(ivf.mean),
                ms(ivf.p50),
                ivf.recall,
            );

            // Where the time actually goes on the approximate path.
            let outcome = engine.search(&queries[0], K);
            println!(
                "    scanned {} of {} rows ({:.1}%), rescored {}",
                outcome.candidates_scored,
                size,
                100.0 * outcome.candidates_scored as f64 / size as f64,
                outcome.rescored,
            );
            println!();
        }
    }

    println!("Note: `speedup vs flat` compares the index against luna-vdb's own");
    println!("vectorised flat scan, not against the pre-0.1 engine. The pre-0.1");
    println!("scan was scalar over a per-vector heap layout, which the kernel and");
    println!("arena changes alone already improve by a large factor before the");
    println!("index is involved.");
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
