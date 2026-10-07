//! Native integration suite — `cargo test`, no wasm, no JS.
//!
//! # Why this drives [`Engine`] and not `LunaVDB`
//!
//! `LunaVDB` is the `#[wasm_bindgen]` surface. Its methods return
//! `Result<_, JsValue>`, and on a non-wasm target `wasm-bindgen` cannot build a
//! `JsValue` at all: `__wbindgen_throw` is an unresolved wasm import, so
//! touching an error path aborts the process with "thread caused non-unwinding
//! panic. aborting." rather than unwinding. (If that message shows up in a
//! native run, this is why.)
//!
//! So the division of labour is:
//!
//! * **here** — the algorithm. Every distance kernel, the IVF/PQ index, the
//!   snapshot codec, k-means, LZ4, CRC. Plain Rust, plain `Result`.
//! * **`tests/common/wasm_suite.rs`** — the binding layer: type conversions and
//!   the JS-visible error behaviour, run under `wasm-pack test` on wasm32 where
//!   `JsValue` is real.
//!
//! The two suites deliberately cover the same behaviour from either side.
//!
//! Tests marked `REGRESSION` pin down a specific defect in the pre-0.1 engine.

use std::collections::HashSet;

use luna_vdb::engine::{
    Distance, Engine, EngineError, IndexOptions, add, clear, dump, dump_compressed, index, load,
    remove, search, search_exact, size,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// SplitMix64, so adjacent seeds give genuinely different streams. (`seed | 1`
/// alone maps 2n and 2n+1 onto the same state, which made two different test
/// corpora come out byte-identical.)
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

/// Build a corpus where every row is a genuinely distinct vector.
///
/// Each row mixes its *own* index into the seed rather than continuing one
/// stream. That matters: with a single shared stream the generator's state can
/// land on the same value for two rows — which it did, producing byte-identical
/// vectors. Every query for such a pair then legitimately returns distance
/// `0.0` for both, and any test asserting "the exact vector is the top hit" is
/// comparing two indistinguishable candidates. Deriving per-row state removes
/// the ambiguity from the fixture rather than from the assertions.
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

/// Options that force IVF on for small test corpora.
fn fast() -> IndexOptions {
    IndexOptions {
        ivf_threshold: 256,
        ..Default::default()
    }
}

/// The engine must never panic, so every `Result` here is unwrapped with a
/// message that names the call rather than a bare `unwrap()`.
fn build(data: &[Vec<f32>], ids: &[String], options: IndexOptions) -> Engine {
    match Engine::build(data, ids, options) {
        Ok(engine) => engine,
        Err(error) => panic!("build failed: {error}"),
    }
}

// ---------------------------------------------------------------------------
// Basic API
// ---------------------------------------------------------------------------

#[test]
fn basic_index_and_size() {
    let engine = build(
        &[vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        &["1".to_string(), "2".to_string()],
        IndexOptions::default(),
    );

    assert_eq!(engine.len(), 2);
    assert_eq!(engine.dim(), 3);
    assert_eq!(engine.distance(), Distance::Euclidean);
    assert!(engine.contains("1"));
    assert!(!engine.contains("3"));
}

#[test]
fn search_returns_ranked_neighbours() {
    let items: Vec<(&str, Vec<f32>)> = vec![
        ("cat", vec![0.8, 0.7, 0.6, 0.2, 0.1]),
        ("dog", vec![0.7, 0.8, 0.6, 0.3, 0.1]),
        ("bird", vec![0.6, 0.5, 0.8, 0.4, 0.2]),
        ("fish", vec![0.2, 0.3, 0.4, 0.8, 0.7]),
        ("car", vec![-0.1, -0.2, -0.3, -0.8, -0.9]),
    ];

    let data: Vec<Vec<f32>> = items.iter().map(|(_, vector)| vector.clone()).collect();
    let ids: Vec<String> = items.iter().map(|(id, _)| id.to_string()).collect();
    let engine = build(&data, &ids, IndexOptions::default());

    let result = engine.search(&[0.8, 0.7, 0.6, 0.2, 0.1], 3);
    assert_eq!(result.neighbors.len(), 3);
    assert_eq!(result.neighbors[0].id, "cat");
    assert_eq!(result.neighbors[1].id, "dog");
    assert_eq!(result.neighbors[2].id, "bird");

    // Ascending distance.
    assert!(result.neighbors[0].distance < result.neighbors[1].distance);
    assert!(result.neighbors[1].distance < result.neighbors[2].distance);

    // An exact query returns its own vector at distance ~0.
    let exact = engine.search(&[0.8, 0.7, 0.6, 0.2, 0.1], 1);
    assert_eq!(exact.neighbors[0].id, "cat");
    assert!(exact.neighbors[0].distance < 1e-6);

    let negative = engine.search(&[-0.1, -0.2, -0.3, -0.8, -0.9], 1);
    assert_eq!(negative.neighbors[0].id, "car");
    assert!(negative.neighbors[0].distance < 1e-6);

    // k beyond the corpus is clamped, not padded.
    assert_eq!(engine.search(&[0.0; 5], 10).neighbors.len(), 5);
}

#[test]
fn boundary_and_zero_queries() {
    let engine = build(&corpus(50, 8, 0x1234), &ids_of(50, "v"), fast());

    for query in [
        vec![0.0f32; 8],
        vec![1.0f32; 8],
        vec![-1.0f32; 8],
        vec![f32::MAX; 8],
        vec![f32::MIN; 8],
    ] {
        let result = engine.search(&query, 5);
        assert_eq!(result.neighbors.len(), 5);

        for neighbor in &result.neighbors {
            // REGRESSION: the old `|a|² + |b|² - 2ab` distance produced NaN for
            // large-magnitude components, which then sorted arbitrarily.
            assert!(!neighbor.distance.is_nan(), "NaN for {}", neighbor.id);
        }
    }
}

#[test]
fn empty_engine_behaviour() {
    let engine = Engine::new(IndexOptions::default());

    let result = engine.search(&[0.1, 0.2, 0.3], 5);
    assert_eq!(result.neighbors.len(), 0);
    assert!(result.exact);
    assert_eq!(result.candidates_scored, 0);

    // k = 0 is a no-op, not an error.
    assert_eq!(engine.search(&[0.1, 0.2, 0.3], 0).neighbors.len(), 0);
}

#[test]
fn single_vector_engine() {
    let engine = build(
        &[vec![1.0, 2.0, 3.0]],
        &["single".to_string()],
        IndexOptions::default(),
    );

    let result = engine.search(&[1.0, 2.0, 3.0], 1);
    assert_eq!(result.neighbors.len(), 1);
    assert_eq!(result.neighbors[0].id, "single");
    assert!(result.neighbors[0].distance < 1e-6);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

#[test]
fn add_and_remove() {
    let mut engine = Engine::new(IndexOptions::default());

    engine.add("3".to_string(), &[0.7, 0.8, 0.9]).expect("add");
    assert_eq!(engine.len(), 1);
    assert!(engine.contains("3"));

    engine.remove(&["3".to_string()]).expect("remove");
    assert_eq!(engine.len(), 0);
    assert!(!engine.contains("3"));
}

#[test]
fn large_corpus_add() {
    let mut engine = build(&corpus(1000, 128, 0xBEEF), &ids_of(1000, "vec-"), fast());
    assert_eq!(engine.len(), 1000);
    assert_eq!(engine.dim(), 128);

    assert_eq!(engine.search(&vec![0.5; 128], 10).neighbors.len(), 10);

    // REGRESSION: the old engine retrained k-means over every vector on every
    // insert, which made this ingest quadratic.
    let extra = corpus(100, 128, 0xF00D);
    for (i, vector) in extra.iter().enumerate() {
        engine.add(format!("extra-{i}"), vector).expect("add");
    }
    assert_eq!(engine.len(), 1100);

    // Every added vector is retrievable.
    for (i, vector) in extra.iter().enumerate().take(10) {
        let found = engine.search(vector, 1);
        assert_eq!(found.neighbors[0].id, format!("extra-{i}"));
    }
}

#[test]
fn dynamic_operations() {
    let ids = ids_of(400, "v");
    let mut engine = build(&corpus(400, 32, 0xCAFE), &ids, fast());

    // Remove every fourth id.
    let to_remove: Vec<String> = (0..50).map(|i| ids[i * 4].clone()).collect();
    engine.remove(&to_remove).expect("remove");
    assert_eq!(engine.len(), 350);

    for id in &to_remove {
        assert!(!engine.contains(id), "{id} should be gone");
    }

    for (i, vector) in corpus(100, 32, 0xFEED).iter().enumerate() {
        engine.add(format!("n{i}"), vector).expect("add");
    }
    assert_eq!(engine.len(), 450);

    let query: Vec<f32> = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        .iter()
        .copied()
        .cycle()
        .take(32)
        .collect();

    assert_eq!(engine.search(&query, 20).neighbors.len(), 20);
}

#[test]
fn clear_resets_and_stays_usable() {
    let mut engine = build(&corpus(200, 16, 0xAAA), &ids_of(200, "v"), fast());
    assert_eq!(engine.len(), 200);

    engine.clear();
    assert_eq!(engine.len(), 0);
    assert_eq!(engine.dim(), 0);
    assert!(!engine.is_indexed());
    assert_eq!(engine.search(&[0.0; 16], 5).neighbors.len(), 0);

    // REGRESSION: the old `clear` left the dimension at 0 but kept the IVF
    // structures, so the next `add` trained against stale centroids.
    engine
        .add("after-clear".to_string(), &[1.0; 16])
        .expect("add after clear");

    assert_eq!(engine.len(), 1);
    assert_eq!(engine.dim(), 16);
    assert_eq!(engine.search(&[1.0; 16], 1).neighbors[0].id, "after-clear");
}

#[test]
fn duplicate_ids_are_reported_not_swallowed() {
    let mut engine = Engine::new(IndexOptions::default());
    engine
        .add("dup".to_string(), &[0.1, 0.2, 0.3])
        .expect("first add");

    let error = engine
        .add("dup".to_string(), &[0.4, 0.5, 0.6])
        .expect_err("duplicate should error");

    assert!(error.message.contains("already exists"), "{error}");
    assert_eq!(engine.len(), 1);
}

#[test]
fn duplicate_ids_within_one_batch() {
    let error = Engine::build(
        &[vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        &["duplicate".to_string(), "duplicate".to_string()],
        IndexOptions::default(),
    )
    .expect_err("duplicate batch should error");

    assert!(error.message.contains("already exists"), "{error}");
}

#[test]
fn dimension_mismatch_is_reported() {
    let mut engine = Engine::new(IndexOptions::default());
    engine.add("a".to_string(), &[0.1; 8]).expect("add");

    let error = engine
        .add("b".to_string(), &[0.1; 16])
        .expect_err("dimension mismatch should error");

    assert!(error.message.contains("dimension mismatch"), "{error}");
    assert_eq!(engine.len(), 1);
}

#[test]
fn remove_unknown_id_is_reported() {
    let mut engine = build(&corpus(10, 8, 1), &ids_of(10, "v"), fast());

    let error = engine
        .remove(&["nope".to_string()])
        .expect_err("unknown id should error");
    assert!(error.message.contains("not found"), "{error}");

    assert_eq!(engine.len(), 10);
}

#[test]
fn build_rejects_mismatched_lengths() {
    let error = Engine::build(&corpus(3, 4, 1), &ids_of(2, "v"), IndexOptions::default())
        .expect_err("length mismatch should error");

    assert!(error.message.contains("mismatch"), "{error}");
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

#[test]
fn serialization_round_trip() {
    let data = corpus(500, 64, 0x5EED);
    let engine = build(&data, &ids_of(500, "vec-"), fast());

    for compressed in [true, false] {
        let bytes = engine.serialize(compressed).expect("serialize");
        assert!(!bytes.is_empty());
        assert!(luna_vdb::is_snapshot(&bytes));
        assert_eq!(luna_vdb::snapshot_version(&bytes), 2);

        let restored = load(&bytes).expect("load");
        assert_eq!(restored.len(), 500);
        assert_eq!(restored.dim(), 64);
        assert_eq!(restored.distance(), engine.distance());

        // Identical results, not merely similar.
        for query_index in [0usize, 123, 499] {
            assert_eq!(
                engine.search(&data[query_index], 10).neighbors,
                restored.search(&data[query_index], 10).neighbors,
                "query {query_index} (compressed={compressed})"
            );
        }
    }
}

#[test]
fn serialization_after_mutation() {
    let mut engine = build(&corpus(400, 32, 0x1111), &ids_of(400, "vec-"), fast());

    for (i, vector) in corpus(60, 32, 0x2222).iter().enumerate() {
        engine.add(format!("late-{i}"), vector).expect("add");
    }

    let remove: Vec<String> = (0..40).map(|i| format!("vec-{i}")).collect();
    engine.remove(&remove).expect("remove");

    let bytes = engine.serialize(true).expect("serialize");
    let restored = load(&bytes).expect("load");

    assert_eq!(restored.len(), engine.len());
    assert_eq!(restored.len(), 420);

    let query = vec![0.1f32; 32];
    assert_eq!(
        engine.search(&query, 10).neighbors,
        restored.search(&query, 10).neighbors
    );
}

#[test]
fn empty_engine_round_trips() {
    let engine = Engine::new(IndexOptions::default());

    for compressed in [true, false] {
        let bytes = engine.serialize(compressed).expect("serialize");
        let restored = load(&bytes).expect("load");
        assert_eq!(restored.len(), 0);
        assert_eq!(restored.dim(), 0);
    }
}

#[test]
fn snapshot_of_engine_without_index_round_trips() {
    // Below the IVF threshold there are no centroids and no codes; the loader
    // must rebuild the index rather than restoring an inconsistent one.
    let engine = build(
        &corpus(50, 16, 0x33),
        &ids_of(50, "v"),
        IndexOptions {
            ivf_threshold: 100_000,
            ..Default::default()
        },
    );
    assert!(!engine.is_indexed());

    let restored = load(&engine.serialize(true).expect("serialize")).expect("load");
    assert_eq!(restored.len(), 50);
    assert_eq!(
        engine.search(&[0.1; 16], 10).neighbors,
        restored.search(&[0.1; 16], 10).neighbors
    );
}

/// REGRESSION: the "神秘的空指针" reproducer.
///
/// The old `load` ended in `bincode::deserialize_from(..).unwrap()`. A truncated
/// or corrupt payload panicked out of the wasm export, poisoning the instance —
/// so the *next* call returned garbage and JS blamed an innocent line, long
/// after the real panic had scrolled off. These must all be ordinary errors, and
/// the engine must stay usable throughout.
#[test]
fn corrupt_snapshots_error_without_poisoning() {
    let engine = build(&corpus(200, 24, 0x777), &ids_of(200, "v"), fast());
    let good = engine.serialize(true).expect("serialize");

    // Truncation at every structurally interesting offset.
    for cut in [
        0usize,
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        15,
        16,
        17,
        good.len() / 2,
        good.len() - 1,
    ] {
        assert!(
            load(&good[..cut]).is_err(),
            "truncation to {cut} should error"
        );
    }

    // Bit flips anywhere in the payload.
    for offset in [16usize, 20, 32, good.len() / 2, good.len() - 1] {
        if offset >= good.len() {
            continue;
        }
        let mut corrupted = good.clone();
        corrupted[offset] ^= 0xFF;
        assert!(
            load(&corrupted).is_err(),
            "bit flip at {offset} should error"
        );
    }

    assert!(load(b"hello world, definitely not a snapshot").is_err());
    assert!(load(&[]).is_err());

    // The engine is untouched, and a valid snapshot still loads.
    assert_eq!(engine.len(), 200);
    assert_eq!(engine.search(&[0.0; 24], 5).neighbors.len(), 5);
    assert!(load(&good).is_ok());
}

/// Bitwise CRC-32 (IEEE), independent of the crate's table-driven one, so the
/// test builds a *valid* envelope without trusting the code under test.
fn crc32_ieee(seed: u32, data: &[u8]) -> u32 {
    let mut crc = !seed;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[test]
fn oversized_declared_length_is_rejected() {
    // REGRESSION: this test used to declare `4u32 << 30` — which is 0 — next
    // to an all-zero CRC, so it passed on the checksum and never exercised a
    // length. It now builds headers whose checksum is *valid*, so only the
    // length handling stands between them and a 4 GiB reservation.
    for flags in [0u16, 1] {
        for raw_len in [u32::MAX, 1 << 31, 1 << 30] {
            for stored in [&[][..], &[0x10, b'x'][..], &[0u8; 16][..]] {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(b"LVD2");
                bytes.extend_from_slice(&2u16.to_le_bytes());
                bytes.extend_from_slice(&flags.to_le_bytes());
                bytes.extend_from_slice(&raw_len.to_le_bytes());
                let crc = crc32_ieee(crc32_ieee(0, &bytes[4..12]), stored);
                bytes.extend_from_slice(&crc.to_le_bytes());
                bytes.extend_from_slice(stored);

                assert!(load(&bytes).is_err(), "flags {flags} raw_len {raw_len}");
            }
        }
    }
}

#[test]
fn header_bit_flips_are_caught() {
    let engine = build(&corpus(50, 8, 0x5), &ids_of(50, "v"), fast());
    let good = engine.serialize(true).expect("serialize");
    for offset in 4..16 {
        let mut bad = good.clone();
        bad[offset] ^= 0x01;
        assert!(load(&bad).is_err(), "flip at header byte {offset}");
    }
}

#[test]
fn future_format_version_is_rejected_clearly() {
    let engine = build(&corpus(20, 8, 1), &ids_of(20, "v"), fast());
    let mut bytes = engine.serialize(true).expect("serialize");

    bytes[4] = 99;
    bytes[5] = 0;

    let error = load(&bytes).expect_err("version check");
    assert!(
        error.message.contains("not supported"),
        "unhelpful message: {error}"
    );
}

// ---------------------------------------------------------------------------
// Options and metrics
// ---------------------------------------------------------------------------

#[test]
fn cosine_distance_is_scale_invariant() {
    let engine = build(
        &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        &["x".to_string(), "y".to_string()],
        IndexOptions {
            distance: Distance::Cosine,
            ivf_threshold: 256,
            ..Default::default()
        },
    );

    // Same direction, 100x the magnitude.
    let result = engine.search(&[100.0, 0.0, 0.0], 1);
    assert_eq!(result.neighbors[0].id, "x");
    assert!(result.neighbors[0].distance.abs() < 1e-5);

    // A zero query has no direction and must not produce NaN.
    let zero = engine.search(&[0.0, 0.0, 0.0], 2);
    assert_eq!(zero.neighbors.len(), 2);
    assert!(zero.neighbors.iter().all(|n| !n.distance.is_nan()));
}

#[test]
fn dot_product_ranks_by_magnitude() {
    let engine = build(
        &[vec![1.0, 0.0], vec![5.0, 0.0]],
        &["small".to_string(), "big".to_string()],
        IndexOptions {
            distance: Distance::DotProduct,
            ..Default::default()
        },
    );

    let result = engine.search(&[1.0, 0.0], 2);
    assert_eq!(result.neighbors[0].id, "big");
    // The reported distance is the dot product, not the negated internal score.
    assert!((result.neighbors[0].distance - 5.0).abs() < 1e-5);
}

#[test]
fn distance_parsing() {
    assert_eq!(Distance::parse("euclidean"), Some(Distance::Euclidean));
    assert_eq!(Distance::parse("L2"), Some(Distance::Euclidean));
    assert_eq!(Distance::parse("Cosine"), Some(Distance::Cosine));
    assert_eq!(Distance::parse("dot"), Some(Distance::DotProduct));
    assert_eq!(Distance::parse("inner_product"), Some(Distance::DotProduct));
    assert_eq!(Distance::parse("hamming"), None);

    // The byte encoding is part of the on-disk format and must not drift.
    assert_eq!(Distance::Euclidean.to_u8(), 0);
    assert_eq!(Distance::Cosine.to_u8(), 1);
    assert_eq!(Distance::DotProduct.to_u8(), 2);
    assert_eq!(Distance::from_u8(1), Some(Distance::Cosine));
    assert_eq!(Distance::from_u8(200), None);
}

#[test]
fn approximate_search_does_less_work_than_exact() {
    let engine = build(
        &corpus(3000, 32, 0xABC),
        &ids_of(3000, "v"),
        IndexOptions {
            approximate: true,
            nprobe: Some(4),
            ..fast()
        },
    );
    assert!(engine.is_indexed());

    let approx = engine.search(&[0.1; 32], 10);
    assert!(!approx.exact);
    assert!(
        approx.candidates_scored < 3000,
        "the index scanned everything ({})",
        approx.candidates_scored
    );
    assert!(approx.cells_probed > 0);

    let exact = engine.search_exact(&[0.1; 32], 10);
    assert!(exact.exact);
    assert_eq!(exact.candidates_scored, 3000);
}

/// 40 well-separated blobs of 32-d points.
fn blobs(count: usize) -> Vec<Vec<f32>> {
    let centres = corpus(40, 32, 0xCE17);
    let noise = corpus(count, 32, 0x2015E);
    noise
        .iter()
        .enumerate()
        .map(|(i, n)| {
            centres[i % 40]
                .iter()
                .zip(n)
                .map(|(c, e)| c * 4.0 + 0.1 * e)
                .collect()
        })
        .collect()
}

#[test]
fn default_search_is_approximate_fast_and_accurate_on_clustered_data() {
    let data = blobs(6064);
    let (data, queries) = data.split_at(6000);
    let engine = build(data, &ids_of(6000, "v"), fast());

    let outcome = engine.search(&queries[0], 10);
    assert!(!outcome.exact, "the default search is approximate");
    assert!(
        outcome.candidates_scored < 6000 / 10,
        "scanned {} of 6000 rows",
        outcome.candidates_scored
    );
    let measured = recall(&engine, queries, 10);
    assert!(measured >= 0.9, "recall was {measured}");
}

#[test]
fn exact_mode_is_exact_and_prunes_on_clustered_data() {
    // The exact search must return exactly the brute-force answer while
    // touching only a fraction of the rows.
    let data = blobs(6000);
    let engine = build(
        &data,
        &ids_of(6000, "v"),
        IndexOptions {
            approximate: false,
            ..fast()
        },
    );

    let mut scanned = 0usize;
    for row in (0..6000).step_by(250) {
        let fast_result = engine.search(&data[row], 10);
        let slow_result = engine.search_exact(&data[row], 10);
        assert!(fast_result.exact);
        let fast_ids: Vec<&str> = fast_result
            .neighbors
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        let slow_ids: Vec<&str> = slow_result
            .neighbors
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        assert_eq!(fast_ids, slow_ids, "query row {row}");
        scanned += fast_result.candidates_scored;
    }

    let average = scanned / 24;
    assert!(
        average < 6000 / 5,
        "the exact search scanned {average} of 6000 rows on average; it should prune"
    );
}

#[test]
fn below_threshold_stays_exact() {
    let engine = build(
        &corpus(100, 16, 0x55),
        &ids_of(100, "v"),
        IndexOptions {
            ivf_threshold: 4096,
            ..Default::default()
        },
    );

    assert!(!engine.is_indexed());

    // With no index, the flat scan runs and the result is exact.
    let outcome = engine.search(&[0.0; 16], 5);
    assert!(outcome.exact);
}

// ---------------------------------------------------------------------------
// Quality
// ---------------------------------------------------------------------------

/// Fraction of the true top-`k` that the approximate path returns.
fn recall(engine: &Engine, queries: &[Vec<f32>], k: usize) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;

    for query in queries {
        let truth: HashSet<String> = engine
            .search_exact(query, k)
            .neighbors
            .into_iter()
            .map(|neighbor| neighbor.id)
            .collect();

        hits += engine
            .search(query, k)
            .neighbors
            .iter()
            .filter(|neighbor| truth.contains(&neighbor.id))
            .count();
        total += k;
    }

    hits as f32 / total as f32
}

#[test]
fn approximate_recall_is_high_without_pq() {
    let data = corpus(4000, 64, 0x7ECA110);
    let engine = build(
        &data,
        &ids_of(4000, "v"),
        IndexOptions {
            ivf_threshold: 256,
            approximate: true,
            nlist: Some(63),
            nprobe: Some(32),
            ..Default::default()
        },
    );

    let queries: Vec<Vec<f32>> = data.iter().step_by(400).cloned().collect();
    let measured = recall(&engine, &queries, 10);

    assert!(measured > 0.85, "recall was {measured}");
}

#[test]
fn pq_index_recall_is_acceptable() {
    let data = corpus(6000, 32, 0x99);
    let engine = build(
        &data,
        &ids_of(6000, "v"),
        IndexOptions {
            ivf_threshold: 256,
            nlist: Some(46),
            nprobe: Some(24),
            exact_rescore_only: false,
            approximate: true,
            ..Default::default()
        },
    );

    assert!(engine.has_pq(), "PQ should be active at 6000 vectors");

    let queries: Vec<Vec<f32>> = data.iter().step_by(600).cloned().collect();
    let measured = recall(&engine, &queries, 10);

    assert!(measured > 0.5, "PQ recall was {measured}");
}

#[test]
fn default_recall_holds_on_unstructured_data() {
    // Uniform noise is the worst case for a partitioning index: the
    // neighbours of a query are spread over many cells. Calibration has to
    // notice and probe enough of them, where a fixed fraction used to find a
    // fraction of the neighbours.
    let data = corpus(5000, 48, 0xF1A7);
    let engine = build(&data, &ids_of(5000, "v"), fast());
    let queries = corpus(30, 48, 0x0E);
    let measured = recall(&engine, &queries, 10);
    assert!(measured >= 0.9, "recall was {measured}");
}

#[test]
fn exact_mode_recall_is_perfect() {
    let data = corpus(5000, 48, 0xF1A7);
    let engine = build(
        &data,
        &ids_of(5000, "v"),
        IndexOptions {
            approximate: false,
            ..fast()
        },
    );
    let queries = corpus(30, 48, 0x0E);
    assert_eq!(recall(&engine, &queries, 10), 1.0);
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// Compaction must not lose, duplicate or reorder rows.
///
/// The invariant is checked *per surviving vector*: every row that is still
/// live must remain its own nearest neighbour, before and after the sweep. That
/// is a property of the data structure.
///
/// The obvious-looking version of this test — remember one query's result
/// before compaction and compare it after — is wrong, and wrong in a way that
/// took a while to see. Compaction renumbers rows: with 80 of the first 500
/// deleted, old row 300 becomes something else, so `data[300]` is no longer the
/// vector that query meant. The comparison then fails while the engine is
/// behaving perfectly, and the failure looks like a recall regression.
#[test]
fn compaction_preserves_answers() {
    let data = corpus(500, 32, 0x5150);
    let ids = ids_of(500, "vec-");
    let options = IndexOptions {
        ivf_threshold: 256,
        exact_rescore_only: true,
        nprobe: Some(32),
        ..Default::default()
    };

    let mut engine = build(&data, &ids, options);

    // Stay under the compaction ratio so the sweep is still pending.
    let remove: Vec<String> = (0..80).map(|i| format!("vec-{i}")).collect();
    engine.remove(&remove).expect("remove");
    assert_eq!(engine.len(), 420);
    assert!(engine.dead_rows() > 0, "80/500 is under the 30% ratio");

    // Survivors, and where each one currently lives.
    let survivors: Vec<usize> = (80..500).collect();

    let is_self_hit = |engine: &Engine| -> Vec<usize> {
        survivors
            .iter()
            .copied()
            .filter(|&row| engine.search(&data[row], 1).neighbors[0].id == ids[row])
            .collect()
    };

    let before = is_self_hit(&engine);
    assert_eq!(
        before.len(),
        survivors.len(),
        "before compaction, {} of {} survivors were not their own top hit",
        survivors.len() - before.len(),
        survivors.len()
    );

    engine.compact();
    assert_eq!(engine.dead_rows(), 0, "explicit compaction must sweep all");
    assert_eq!(engine.len(), 420);

    let after = is_self_hit(&engine);
    assert_eq!(
        after, before,
        "compaction changed which vectors are their own nearest neighbour"
    );
    assert_eq!(after.len(), survivors.len());
}

#[test]
fn compaction_keeps_only_live_rows() {
    let ids = ids_of(200, "v");
    let mut engine = build(&corpus(200, 16, 0x42), &ids, fast());

    engine
        .remove(&(0..100).map(|i| ids[i].clone()).collect::<Vec<_>>())
        .expect("remove");

    assert_eq!(engine.len(), 100);
    engine.compact();

    let result = engine.search(&[0.05; 16], 100);
    for neighbor in &result.neighbors {
        assert!(engine.contains(&neighbor.id), "{} is not live", neighbor.id);
    }
}

// ---------------------------------------------------------------------------
// Free-function API — still available for callers on the old shape
// ---------------------------------------------------------------------------

#[test]
fn free_functions_match_the_methods() {
    let data = corpus(300, 16, 0x88);
    let ids = ids_of(300, "v");

    let mut engine = index(&data, &ids).expect("index()");
    assert_eq!(size(&engine), 300);

    assert_eq!(
        search(&engine, &data[7], 5).neighbors,
        engine.search(&data[7], 5).neighbors
    );
    assert_eq!(
        search_exact(&engine, &data[7], 5).neighbors,
        engine.search_exact(&data[7], 5).neighbors
    );

    add(&mut engine, "extra".to_string(), &data[0]).expect("add()");
    assert_eq!(size(&engine), 301);

    remove(&mut engine, &["extra".to_string()]).expect("remove()");
    assert_eq!(size(&engine), 300);

    let bytes = dump(&engine).expect("dump");
    assert_eq!(load(&bytes).expect("load").len(), 300);

    let raw = dump_compressed(&engine, false).expect("dump_compressed");
    assert_eq!(load(&raw).expect("load").len(), 300);

    clear(&mut engine);
    assert_eq!(size(&engine), 0);
}

#[test]
fn engine_default_is_empty() {
    let engine = Engine::default();
    assert_eq!(engine.len(), 0);
    assert!(!engine.is_indexed());
    assert!(!engine.has_pq());
}

#[test]
fn engine_error_displays_its_message() {
    let error = EngineError::new("something went wrong");
    assert_eq!(error.to_string(), "something went wrong");

    // The constructors produce the messages the tests above match on.
    assert!(
        EngineError::duplicate_id("x")
            .message
            .contains("already exists")
    );
    assert!(EngineError::missing_id("x").message.contains("not found"));
    assert!(EngineError::corrupt("bad").message.contains("bad"));
    assert!(EngineError::dimension_mismatch(8, 4).message.contains('8'));
}

// ---------------------------------------------------------------------------
// Diagnostics surface
// ---------------------------------------------------------------------------

#[test]
fn simd_backend_is_reported() {
    let backend = luna_vdb::simd_backend();
    assert!(
        ["wasm-simd128", "avx2", "neon", "scalar"].contains(&backend),
        "unexpected backend {backend}"
    );
}

#[test]
fn snapshot_detection_helpers() {
    assert!(!luna_vdb::is_snapshot(b"nope"));
    assert!(!luna_vdb::is_snapshot(&[]));
    assert_eq!(luna_vdb::snapshot_version(b"nope"), 0);

    let engine = Engine::new(IndexOptions::default());
    let bytes = engine.serialize(true).expect("serialize");
    assert!(luna_vdb::is_snapshot(&bytes));
    assert_eq!(luna_vdb::snapshot_version(&bytes), 2);

    // The gzip magic marks a legacy (pre-0.1) snapshot.
    assert!(luna_vdb::is_snapshot(&[0x1f, 0x8b, 0x08, 0x00]));
    assert_eq!(luna_vdb::snapshot_version(&[0x1f, 0x8b, 0x08, 0x00]), 1);
}

#[test]
fn stats_and_memory_reporting() {
    let engine = build(&corpus(2000, 32, 0x424242), &ids_of(2000, "v"), fast());

    assert!(engine.is_indexed());
    assert!(engine.nlist() > 0);
    assert!(engine.nprobe() > 0);
    assert_eq!(engine.dead_rows(), 0);

    let bytes = engine.memory_bytes();
    // 2000 × 32 f32 = 256 KB of vectors alone, before the index.
    assert!(
        bytes > 2000 * 32 * 4,
        "memory_bytes {bytes} looks too small"
    );
}

#[test]
fn set_nprobe_retunes_without_rebuild() {
    // An engine with no index ignores the call rather than inventing a value.
    let mut empty = Engine::new(IndexOptions::default());
    empty.set_nprobe(8);
    assert_eq!(empty.nprobe(), 0);

    let data = corpus(2_000, 16, 0x5E7);
    let ids = ids_of(2_000, "v");
    let mut engine = build(&data, &ids, fast());
    let nlist = engine.nlist();
    assert!(
        nlist > 1,
        "test corpus should be indexed, got nlist {nlist}"
    );

    // Clamped to `1..=nlist` at both ends.
    engine.set_nprobe(0);
    assert_eq!(engine.nprobe(), 1);
    engine.set_nprobe(usize::MAX);
    assert_eq!(engine.nprobe(), nlist);

    // Probing every cell must find a stored vector as its own nearest
    // neighbour — the cells partition the corpus, so nothing is out of reach.
    for row in [0usize, 7, 999, 1_999] {
        assert_eq!(engine.search(&data[row], 1).neighbors[0].id, ids[row]);
    }

    // The value sticks as the configured one.
    assert_eq!(engine.options().nprobe, Some(nlist));
}
