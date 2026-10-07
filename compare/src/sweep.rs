//! Recall/latency sweep for the new engine, on two kinds of data.
//!
//! For each dataset and size it prints:
//!
//! * the brute-force scan (the reference, recall 1.0 by definition),
//! * the default search — approximate, with the `nprobe` the index calibrated
//!   for itself — and the exact IVF-pruned search (`approximate: false`),
//! * approximate mode across `nprobe`, with exact scoring of every probed row
//!   ("ivf-flat") and with the PQ prefilter ("ivf-pq").
//!
//! Two datasets, because the answer depends on them: uniform random vectors
//! have no neighbourhood structure for any partitioning index to exploit, so
//! the pruned search scans most rows and a fixed small `nprobe` has low
//! recall; an overlapping Gaussian mixture is much closer to real text
//! embeddings.
//!
//! The `cells` column is the number of cells actually probed, averaged over
//! the queries. The default row is measured on fresh queries, while the
//! calibration used stored rows, so its recall checks the calibration too.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use luna_vdb::engine::{Engine, IndexOptions, SearchOutcome};

mod data;
use data::{Data, generate, splitmix};

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

struct Measured {
    latency: Duration,
    recall: f64,
    scanned: f64,
    cells: f64,
}

fn measure(
    queries: &[Vec<f32>],
    truth: &[HashSet<String>],
    mut search: impl FnMut(&[f32]) -> SearchOutcome,
) -> Measured {
    for query in queries.iter().take(5) {
        black_box(search(query));
    }
    let mut total = Duration::ZERO;
    let mut hits = 0usize;
    let mut scanned = 0usize;
    let mut cells = 0usize;
    for (query, expected) in queries.iter().zip(truth) {
        let started = Instant::now();
        let outcome = black_box(search(query));
        total += started.elapsed();
        hits += outcome
            .neighbors
            .iter()
            .filter(|n| expected.contains(n.id.as_str()))
            .count();
        scanned += outcome.candidates_scored;
        cells += outcome.cells_probed;
    }
    let n = queries.len() as f64;
    Measured {
        latency: total / queries.len() as u32,
        recall: hits as f64 / (queries.len() * K) as f64,
        scanned: scanned as f64 / n,
        cells: cells as f64 / n,
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn main() {
    println!("luna-vdb recall/latency sweep");
    println!(
        "kernel: {}, queries: {QUERIES}, k: {K}",
        luna_vdb::simd_backend()
    );

    for data in [Data::Clustered, Data::Uniform] {
        for &(dim, size) in &CASES {
            let seed = splitmix(((dim as u64) << 40) ^ size as u64 ^ data as u64);
            let (corpus, queries) = generate(data, size, QUERIES, dim, seed);
            let ids: Vec<String> = (0..size).map(|i| format!("v{i}")).collect();

            println!("\n== {} dim {dim} n {size}", data.name());

            let started = Instant::now();
            let mut engine = match Engine::build(&corpus, &ids, IndexOptions::default()) {
                Ok(engine) => engine,
                Err(error) => {
                    println!("   build failed: {error}");
                    continue;
                }
            };
            let build = started.elapsed();

            let truth: Vec<HashSet<String>> = queries
                .iter()
                .map(|q| {
                    engine
                        .search_exact(q, K)
                        .neighbors
                        .into_iter()
                        .map(|n| n.id)
                        .collect()
                })
                .collect();

            let flat = measure(&queries, &truth, |q| engine.search_exact(q, K));
            println!(
                "   exact scan               {:>9.4} ms   recall {:.3}",
                ms(flat.latency),
                flat.recall
            );

            let default = measure(&queries, &truth, |q| engine.search(q, K));
            print_row(
                &format!("default (nprobe {:>3})", engine.nprobe()),
                &default,
                flat.latency,
            );
            println!("   build {:.0} ms, nlist {}", ms(build), engine.nlist());

            engine.set_approximate(false);
            let pruned = measure(&queries, &truth, |q| engine.search(q, K));
            print_row("exact (pruned)      ", &pruned, flat.latency);

            engine.set_approximate(true);
            sweep("ivf-flat", &mut engine, &queries, &truth, flat.latency);

            let started = Instant::now();
            let mut pq = match Engine::build(
                &corpus,
                &ids,
                IndexOptions {
                    exact_rescore_only: false,
                    approximate: true,
                    ..IndexOptions::default()
                },
            ) {
                Ok(engine) => engine,
                Err(error) => {
                    println!("   pq build failed: {error}");
                    continue;
                }
            };
            println!(
                "   ivf-pq build {:.0} ms, pq {} (calibration drops it when it cannot reach the recall target)",
                ms(started.elapsed()),
                pq.has_pq()
            );
            let pq_default = measure(&queries, &truth, |q| pq.search(q, K));
            print_row(
                &format!("pq default (nprobe {:>3})", pq.nprobe()),
                &pq_default,
                flat.latency,
            );
            sweep("ivf-pq  ", &mut pq, &queries, &truth, flat.latency);
        }
    }
}

fn print_row(label: &str, m: &Measured, reference: Duration) {
    println!(
        "   {label} {:>9.4} ms   recall {:.3}   {:>6.1}x vs scan   scanned {:>8.0} rows in {:>5.1} cells",
        ms(m.latency),
        m.recall,
        reference.as_secs_f64() / m.latency.as_secs_f64().max(1e-12),
        m.scanned,
        m.cells,
    );
}

fn sweep(
    label: &str,
    engine: &mut Engine,
    queries: &[Vec<f32>],
    truth: &[HashSet<String>],
    reference: Duration,
) {
    let mut last = 0usize;
    for &nprobe in &NPROBES {
        if nprobe > engine.nlist() {
            break;
        }
        engine.set_nprobe(nprobe);
        let m = measure(queries, truth, |q| engine.search(q, K));
        // Print each effective configuration once.
        let effective = m.cells.round() as usize;
        if effective == last {
            continue;
        }
        last = effective;
        println!(
            "   {label} cells {:>6.1}   {:>9.4} ms   recall {:.3}   {:>6.1}x vs scan",
            m.cells,
            ms(m.latency),
            m.recall,
            reference.as_secs_f64() / m.latency.as_secs_f64().max(1e-12),
        );
    }
}
