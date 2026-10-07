//! Recall/latency sweep for the new engine, on two kinds of data.
//!
//! The comparison binary showed that on *uniform random* vectors the default
//! IVF path returns 2–10% recall: with no cluster structure the true
//! neighbours are spread evenly over the cells, so probing 5% of the cells
//! finds about 5% of them. That is a property of the data as much as of the
//! index — uniform random vectors in hundreds of dimensions have no
//! neighbourhood structure for any partitioning index to exploit — but real
//! embeddings are strongly clustered, and a fair verdict needs both.
//!
//! So this sweeps, for each dataset and size:
//!
//! * the exact SIMD scan (the reference, recall 1.0 by definition),
//! * IVF with exact rescoring of every probed row ("IVF-Flat"),
//! * IVF with the PQ prefilter (the current default),
//!
//! across `nprobe`, and prints recall@10 and latency side by side. `nprobe` is
//! changed with `Engine::set_nprobe` on one built index, so each curve costs a
//! single build.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use luna_vdb::engine::{Engine, IndexOptions};

const K: usize = 10;
const QUERIES: usize = 100;
const NPROBES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const CASES: [(usize, usize); 5] = [
    (384, 10_000),
    (1536, 10_000),
    (384, 50_000),
    (1536, 50_000),
    (384, 200_000),
];

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(splitmix(seed) | 1)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal, Box–Muller.
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(f32::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy)]
enum Data {
    /// Uniform in `[-0.5, 0.5)^dim`: no structure at all. The worst case for
    /// any partitioning index.
    Uniform,
    /// Gaussian mixture: `n / 64` centres, uniform in the cube, each point a
    /// centre plus isotropic noise. The noise is set so clusters overlap
    /// noticeably — a point's 10 nearest neighbours are usually, but not
    /// always, in its own cluster — which is closer to real text embeddings
    /// than either well-separated blobs or uniform noise.
    Clustered,
}

impl Data {
    fn name(self) -> &'static str {
        match self {
            Data::Uniform => "uniform",
            Data::Clustered => "clustered",
        }
    }
}

/// Corpus and queries drawn from the same distribution, so a query is a new
/// point "about" one of the corpus's topics, not a copy of a stored row.
fn generate(data: Data, count: usize, queries: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng::new(seed);
    match data {
        Data::Uniform => {
            let mut draw = |n: usize| -> Vec<Vec<f32>> {
                (0..n)
                    .map(|_| (0..dim).map(|_| rng.unit() - 0.5).collect())
                    .collect()
            };
            let corpus = draw(count);
            let queries = draw(queries);
            (corpus, queries)
        }
        Data::Clustered => {
            let centres_n = (count / 64).max(1);
            let centres: Vec<Vec<f32>> = (0..centres_n)
                .map(|_| (0..dim).map(|_| rng.unit() - 0.5).collect())
                .collect();
            // Per-coordinate spread of the centres is 1/sqrt(12) ≈ 0.289; noise
            // at 0.6 of that makes within-cluster distances comparable to, but
            // smaller than, the gap to neighbouring clusters.
            let sigma = 0.6 * 0.2887;
            let mut draw = |n: usize| -> Vec<Vec<f32>> {
                (0..n)
                    .map(|_| {
                        let pick = ((rng.unit() * centres_n as f32) as usize).min(centres_n - 1);
                        centres[pick].iter().map(|c| c + sigma * rng.normal()).collect()
                    })
                    .collect()
            };
            let corpus = draw(count);
            let queries = draw(queries);
            (corpus, queries)
        }
    }
}

fn measure(queries: &[Vec<f32>], truth: &[HashSet<String>], mut search: impl FnMut(&[f32]) -> Vec<String>) -> (Duration, f64) {
    for query in queries.iter().take(5) {
        black_box(search(query));
    }
    let mut total = Duration::ZERO;
    let mut hits = 0usize;
    for (query, expected) in queries.iter().zip(truth) {
        let started = Instant::now();
        let found = black_box(search(query));
        total += started.elapsed();
        hits += found.iter().filter(|id| expected.contains(id.as_str())).count();
    }
    (
        total / queries.len() as u32,
        hits as f64 / (queries.len() * K) as f64,
    )
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn ids_of(engine: &Engine, query: &[f32], exact: bool) -> Vec<String> {
    let outcome = if exact {
        engine.search_exact(query, K)
    } else {
        engine.search(query, K)
    };
    outcome.neighbors.into_iter().map(|n| n.id).collect()
}

fn main() {
    println!("luna-vdb recall/latency sweep");
    println!("kernel: {}, queries: {QUERIES}, k: {K}", luna_vdb::simd_backend());

    for data in [Data::Clustered, Data::Uniform] {
        for &(dim, size) in &CASES {
            let seed = splitmix((dim as u64) << 40 ^ size as u64 ^ data as u64);
            let (corpus, queries) = generate(data, size, QUERIES, dim, seed);
            let ids: Vec<String> = (0..size).map(|i| format!("v{i}")).collect();

            println!("\n== {} dim {dim} n {size}", data.name());

            let flat_options = IndexOptions {
                ivf_threshold: usize::MAX,
                ..IndexOptions::default()
            };
            let flat = match Engine::build(&corpus, &ids, flat_options) {
                Ok(engine) => engine,
                Err(error) => {
                    println!("   flat build failed: {error}");
                    continue;
                }
            };
            let truth: Vec<HashSet<String>> = queries
                .iter()
                .map(|q| ids_of(&flat, q, true).into_iter().collect())
                .collect();
            let (flat_latency, _) = measure(&queries, &truth, |q| ids_of(&flat, q, true));
            println!("   exact scan                 {:>9.4} ms   recall 1.000", ms(flat_latency));
            drop(flat);

            for (label, exact_rescore_only) in [("ivf-flat", true), ("ivf-pq  ", false)] {
                let options = IndexOptions {
                    exact_rescore_only,
                    ..IndexOptions::default()
                };
                let started = Instant::now();
                let mut engine = match Engine::build(&corpus, &ids, options) {
                    Ok(engine) => engine,
                    Err(error) => {
                        println!("   {label} build failed: {error}");
                        continue;
                    }
                };
                let build = started.elapsed();
                println!(
                    "   {label} build {:>9.1} ms  nlist {}  default nprobe {}  pq {}",
                    ms(build),
                    engine.nlist(),
                    engine.nprobe(),
                    engine.has_pq(),
                );

                let mut last = 0;
                for &nprobe in &NPROBES {
                    if nprobe > engine.nlist() || engine.nlist() == 0 {
                        break;
                    }
                    engine.set_nprobe(nprobe);
                    if engine.nprobe() == last {
                        continue;
                    }
                    last = engine.nprobe();
                    let (latency, recall) = measure(&queries, &truth, |q| ids_of(&engine, q, false));
                    println!(
                        "   {label} nprobe {:>4}   {:>9.4} ms   recall {recall:.3}   {:>6.1}x vs exact",
                        nprobe,
                        ms(latency),
                        flat_latency.as_secs_f64() / latency.as_secs_f64().max(1e-12),
                    );
                }
            }
        }
    }
}
