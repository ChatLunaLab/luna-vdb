//! Head-to-head benchmark: the rewritten engine against the last pre-rewrite
//! commit (`ca10590`), in one process, on identical data.
//!
//! What it reports, per (dimension, corpus size):
//!
//! * **build** — `new LunaVDB(resource)` (old) vs `Engine::build` (new).
//! * **search** — mean latency of the default `search` path on each engine,
//!   with recall@10 measured against the exact answer. A speedup without the
//!   recall next to it is not a speedup, so the two are always printed together.
//! * **exact scan** — the new engine's brute-force path, for reference: it is
//!   what the index has to beat, and on small corpora it is what runs.
//! * **snapshot** — serialise and restore time plus size, old gzip+bincode
//!   against the new LZ4+CRC format.
//!
//! and separately the incremental-ingest cost, where the old engine retrained
//! its whole index on every `add`.
//!
//! Both engines receive the query as an owned `Vec<f32>`, because that is what
//! the JS bindings hand them; the copy is inside the timed region for both.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use legacy::{EmbeddedResource as OldItem, LunaVDB as OldDb, Resource as OldResource};
use luna_vdb::engine::{self as new_engine, Engine, IndexOptions};

const K: usize = 10;
const QUERIES: usize = 100;

/// (dimension, corpus size). 384 and 1536 bracket the common embedding models;
/// 1k is "no index", 50k is past the old engine's 20k IVF threshold so both
/// engines are running their approximate path.
const CASES: [(usize, usize); 6] = [
    (384, 1_000),
    (384, 10_000),
    (384, 50_000),
    (1536, 1_000),
    (1536, 10_000),
    (1536, 50_000),
];

/// Ingest: adds onto an existing corpus past the old engine's IVF threshold.
const INGEST_DIM: usize = 384;
const INGEST_BASE: usize = 25_000;
const INGEST_ADDS: usize = 50;
/// The old engine's per-add cost is a full index rebuild, so its run is capped
/// by wall-clock and the per-add figure extrapolated from what completed.
const INGEST_OLD_BUDGET: Duration = Duration::from_secs(90);

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

/// One independent stream per row, so adjacent rows are not correlated.
fn corpus(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|row| {
            let mut next = rng(splitmix(seed).wrapping_add(row as u64));
            (0..dim).map(|_| next()).collect()
        })
        .collect()
}

fn ids_of(count: usize, prefix: &str) -> Vec<String> {
    (0..count).map(|i| format!("{prefix}{i}")).collect()
}

fn old_resource(data: &[Vec<f32>], ids: &[String]) -> OldResource {
    OldResource {
        embeddings: data
            .iter()
            .zip(ids)
            .map(|(vector, id)| OldItem {
                id: id.clone(),
                embeddings: vector.clone(),
            })
            .collect(),
    }
}

fn timed<T>(run: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let value = run();
    (value, started.elapsed())
}

struct QueryStats {
    mean: Duration,
    recall: f64,
}

fn run_queries(
    queries: &[Vec<f32>],
    truth: &[HashSet<String>],
    mut search: impl FnMut(Vec<f32>) -> Vec<String>,
) -> QueryStats {
    // Warm-up, discarded: allocator growth and branch training stay out of the
    // measurement.
    for query in queries.iter().take(5) {
        black_box(search(query.clone()));
    }

    let mut total = Duration::ZERO;
    let mut hits = 0usize;

    for (query, expected) in queries.iter().zip(truth) {
        let owned = query.clone();
        let started = Instant::now();
        let found = black_box(search(owned));
        total += started.elapsed();

        hits += found
            .iter()
            .filter(|id| expected.contains(id.as_str()))
            .count();
    }

    QueryStats {
        mean: total / queries.len() as u32,
        recall: hits as f64 / (queries.len() * K) as f64,
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn ratio(old: Duration, new: Duration) -> f64 {
    old.as_secs_f64() / new.as_secs_f64().max(1e-12)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

struct Row {
    dim: usize,
    size: usize,
    search_speedup: f64,
    old_recall: f64,
    new_recall: f64,
    build_speedup: f64,
    save_speedup: f64,
    load_speedup: f64,
}

fn main() {
    println!("luna-vdb: rewritten engine vs pre-rewrite (ca10590)");
    println!(
        "kernel: {}, queries: {QUERIES}, k: {K}\n",
        luna_vdb::simd_backend()
    );

    let mut rows = Vec::new();

    for &(dim, size) in &CASES {
        let seed = 0xC0FF_EE00 ^ ((dim as u64) << 32) ^ size as u64;
        let data = corpus(size, dim, seed);
        let ids = ids_of(size, "v");
        let queries = corpus(QUERIES, dim, seed ^ 0xBEEF);

        // -- build ---------------------------------------------------------
        let (new_db, new_build) = timed(|| Engine::build(&data, &ids, IndexOptions::default()));
        let new_db = match new_db {
            Ok(engine) => engine,
            Err(error) => {
                println!("dim {dim} n {size}: new engine build failed: {error}");
                continue;
            }
        };

        let resource = old_resource(&data, &ids);
        let (old_db, old_build) = timed(|| OldDb::new(Some(resource)));

        // Ground truth from the new engine's exact scan — both engines are
        // Euclidean by default, and recall is computed on ids alone.
        let truth: Vec<HashSet<String>> = queries
            .iter()
            .map(|query| {
                new_db
                    .search_exact(query, K)
                    .neighbors
                    .into_iter()
                    .map(|neighbor| neighbor.id)
                    .collect()
            })
            .collect();

        // -- search --------------------------------------------------------
        let old_search = run_queries(&queries, &truth, |query| {
            old_db
                .search(query, K)
                .neighbors
                .into_iter()
                .map(|neighbor| neighbor.id)
                .collect()
        });

        let new_search = run_queries(&queries, &truth, |query| {
            new_db
                .search(&query, K)
                .neighbors
                .into_iter()
                .map(|neighbor| neighbor.id)
                .collect()
        });

        let new_exact = run_queries(&queries, &truth, |query| {
            new_db
                .search_exact(&query, K)
                .neighbors
                .into_iter()
                .map(|neighbor| neighbor.id)
                .collect()
        });

        // -- snapshot ------------------------------------------------------
        let (old_bytes, old_save) = timed(|| old_db.serialize());
        let old_len = old_bytes.len();
        let (old_restored, old_load) = timed(|| OldDb::deserialize(old_bytes));
        black_box(old_restored.size());

        let (new_bytes, new_save) = timed(|| new_db.serialize(true));
        let new_bytes = match new_bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                println!("dim {dim} n {size}: new engine serialise failed: {error}");
                continue;
            }
        };
        let (new_restored, new_load) = timed(|| new_engine::load(&new_bytes));
        match new_restored {
            Ok(engine) => {
                black_box(engine.len());
            }
            Err(error) => {
                println!("dim {dim} n {size}: new engine restore failed: {error}");
                continue;
            }
        }

        // -- report --------------------------------------------------------
        println!(
            "dim {dim:>5}  n {size:>6}   (new: nlist {}, nprobe {}, pq {})",
            new_db.nlist(),
            new_db.nprobe(),
            new_db.has_pq(),
        );
        println!(
            "  build        old {:>10.2} ms   new {:>10.2} ms   {:>7.1}x",
            ms(old_build),
            ms(new_build),
            ratio(old_build, new_build),
        );
        println!(
            "  search       old {:>10.4} ms   new {:>10.4} ms   {:>7.1}x   recall old {:.3} new {:.3}",
            ms(old_search.mean),
            ms(new_search.mean),
            ratio(old_search.mean, new_search.mean),
            old_search.recall,
            new_search.recall,
        );
        println!(
            "  exact scan                         new {:>10.4} ms   {:>7.1}x vs old search   recall {:.3}",
            ms(new_exact.mean),
            ratio(old_search.mean, new_exact.mean),
            new_exact.recall,
        );
        println!(
            "  serialise    old {:>10.2} ms   new {:>10.2} ms   {:>7.1}x   size old {:.2} MiB new {:.2} MiB",
            ms(old_save),
            ms(new_save),
            ratio(old_save, new_save),
            mib(old_len),
            mib(new_bytes.len()),
        );
        println!(
            "  restore      old {:>10.2} ms   new {:>10.2} ms   {:>7.1}x",
            ms(old_load),
            ms(new_load),
            ratio(old_load, new_load),
        );
        println!();

        rows.push(Row {
            dim,
            size,
            search_speedup: ratio(old_search.mean, new_search.mean),
            old_recall: old_search.recall,
            new_recall: new_search.recall,
            build_speedup: ratio(old_build, new_build),
            save_speedup: ratio(old_save, new_save),
            load_speedup: ratio(old_load, new_load),
        });
    }

    // -- ingest --------------------------------------------------------------
    println!("ingest: {INGEST_ADDS} single-vector adds onto n={INGEST_BASE}, dim {INGEST_DIM}");

    let base = corpus(INGEST_BASE, INGEST_DIM, 0x1A6E57);
    let base_ids = ids_of(INGEST_BASE, "b");
    let extra = corpus(INGEST_ADDS, INGEST_DIM, 0xADD5);
    let extra_ids = ids_of(INGEST_ADDS, "x");

    let mut old_db = OldDb::new(Some(old_resource(&base, &base_ids)));
    let mut old_done = 0usize;
    let old_started = Instant::now();
    for (vector, id) in extra.iter().zip(&extra_ids) {
        old_db.add(OldResource {
            embeddings: vec![OldItem {
                id: id.clone(),
                embeddings: vector.clone(),
            }],
        });
        old_done += 1;
        if old_started.elapsed() > INGEST_OLD_BUDGET {
            break;
        }
    }
    let old_per_add = old_started.elapsed() / old_done.max(1) as u32;

    let new_per_add = match Engine::build(&base, &base_ids, IndexOptions::default()) {
        Ok(mut engine) => {
            let started = Instant::now();
            let mut failed = false;
            for (vector, id) in extra.iter().zip(&extra_ids) {
                if let Err(error) = engine.add(id.clone(), vector) {
                    println!("  new engine add failed: {error}");
                    failed = true;
                    break;
                }
            }
            if failed {
                None
            } else {
                Some(started.elapsed() / INGEST_ADDS as u32)
            }
        }
        Err(error) => {
            println!("  new engine build failed: {error}");
            None
        }
    };

    println!(
        "  old {:>10.3} ms/add  ({old_done} of {INGEST_ADDS} completed within {}s)",
        ms(old_per_add),
        INGEST_OLD_BUDGET.as_secs(),
    );
    if let Some(new_per_add) = new_per_add {
        println!(
            "  new {:>10.3} ms/add  {:>9.1}x",
            ms(new_per_add),
            ratio(old_per_add, new_per_add),
        );
    }
    println!();

    // -- summary -------------------------------------------------------------
    println!("summary (old ÷ new; higher is better for the new engine)");
    println!("  dim      n    search   recall old→new    build   serialise   restore");
    for row in &rows {
        println!(
            "  {:>4} {:>6}  {:>7.1}x     {:.3}→{:.3}   {:>6.1}x   {:>8.1}x  {:>7.1}x",
            row.dim,
            row.size,
            row.search_speedup,
            row.old_recall,
            row.new_recall,
            row.build_speedup,
            row.save_speedup,
            row.load_speedup,
        );
    }
}
