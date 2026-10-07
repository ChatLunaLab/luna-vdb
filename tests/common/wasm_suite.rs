//! Shared wasm test suite, compiled into both `tests/web.rs` (browser) and
//! `tests/node.rs` (node).
//!
//! The two hosts run identical assertions — only the `wasm_bindgen_test_configure!`
//! line differs — so the body lives here once. The previous layout duplicated
//! ~400 lines across the two files, and they had already drifted apart.
//!
//! Everything here goes through the **public wasm-bindgen API**, so these tests
//! exercise the generated glue (type conversions, `into_wasm_abi` handling)
//! rather than the engine directly. That is the layer where the old null-pointer
//! reports originated, and it is the layer `cargo test` cannot reach.

use luna_vdb::{
    EmbeddedResource, LunaOptions, LunaVDB, Resource, is_snapshot, simd_backend, snapshot_version,
    version,
};

/// SplitMix64 seed mixing — `seed | 1` alone maps adjacent seeds to the same
/// stream.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic PRNG — a wasm test must not depend on `getrandom`, which needs
/// host entropy and a JS shim in some runtimes.
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = splitmix(seed) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// Deterministic corpus with `prefix`-numbered ids.
///
/// The prefix matters: ids are `{prefix}{index}`, so two calls with different
/// prefixes are two disjoint id spaces. The mutation tests rely on this —
/// seeding only the vectors and reusing a shared id space meant the second
/// `add` re-inserted ids the first batch had already claimed, and the engine
/// correctly rejected the batch as duplicates.
fn generate_prefixed(count: usize, dim: usize, seed: u64, prefix: &str) -> Vec<EmbeddedResource> {
    let mut next = rng(seed);
    (0..count)
        .map(|i| EmbeddedResource {
            id: format!("{prefix}{i}"),
            embeddings: (0..dim).map(|_| next()).collect(),
        })
        .collect()
}

fn generate_test_data(count: usize, dim: usize, seed: u64) -> Vec<EmbeddedResource> {
    generate_prefixed(count, dim, seed, "vec-")
}

fn resource(items: Vec<EmbeddedResource>) -> Resource {
    Resource { embeddings: items }
}

fn fast_options() -> LunaOptions {
    LunaOptions {
        ivf_threshold: Some(256),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Construction and indexing
// ---------------------------------------------------------------------------

pub fn test_new_empty() {
    let engine = LunaVDB::new(None).expect("construct");
    assert_eq!(engine.size(), 0);
    assert_eq!(engine.dimension(), 0);
    assert_eq!(engine.distance(), "euclidean");
    assert!(!engine.has("anything"));
}

pub fn test_new_with_options() {
    let engine = LunaVDB::new(Some(LunaOptions {
        distance: Some("cosine".to_string()),
        nprobe: Some(4),
        ..Default::default()
    }))
    .expect("construct");

    assert_eq!(engine.distance(), "cosine");
    assert_eq!(engine.size(), 0);
}

/// An invalid option must throw *and* leave the module usable. Before the
/// rewrite a constructor panic poisoned the instance for the rest of its life.
pub fn test_invalid_option_does_not_poison() {
    assert!(
        LunaVDB::new(Some(LunaOptions {
            distance: Some("hamming".to_string()),
            ..Default::default()
        }))
        .is_err()
    );

    // The module still works.
    let engine = LunaVDB::new(None).expect("construct after failure");
    assert_eq!(engine.size(), 0);
}

pub fn test_index_and_size() {
    let mut engine = LunaVDB::new(None).expect("construct");

    engine
        .index(resource(vec![
            EmbeddedResource {
                id: "1".to_string(),
                embeddings: vec![0.1, 0.2, 0.3],
            },
            EmbeddedResource {
                id: "2".to_string(),
                embeddings: vec![0.4, 0.5, 0.6],
            },
        ]))
        .expect("index");

    assert_eq!(engine.size(), 2);
    assert_eq!(engine.dimension(), 3);
    assert!(engine.has("1"));
    assert!(!engine.has("3"));
}

pub fn test_index_replaces_contents() {
    let mut engine = LunaVDB::new(None).expect("construct");

    engine
        .index(resource(generate_test_data(50, 8, 1)))
        .expect("first index");
    assert_eq!(engine.size(), 50);

    engine
        .index(resource(generate_test_data(10, 8, 2)))
        .expect("second index");
    assert_eq!(engine.size(), 10);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

pub fn test_search_ranks_neighbours() {
    let mut engine = LunaVDB::new(None).expect("construct");

    let items: Vec<(&str, Vec<f32>)> = vec![
        ("cat", vec![0.8, 0.7, 0.6, 0.2, 0.1]),
        ("dog", vec![0.7, 0.8, 0.6, 0.3, 0.1]),
        ("bird", vec![0.6, 0.5, 0.8, 0.4, 0.2]),
        ("fish", vec![0.2, 0.3, 0.4, 0.8, 0.7]),
        ("car", vec![-0.1, -0.2, -0.3, -0.8, -0.9]),
    ];

    engine
        .index(resource(
            items
                .iter()
                .map(|(id, embeddings)| EmbeddedResource {
                    id: id.to_string(),
                    embeddings: embeddings.clone(),
                })
                .collect(),
        ))
        .expect("index");

    let result = engine.search(vec![0.8, 0.7, 0.6, 0.2, 0.1], 3);
    assert_eq!(result.neighbors.len(), 3);
    assert_eq!(result.neighbors[0].id, "cat");
    assert_eq!(result.neighbors[1].id, "dog");
    assert_eq!(result.neighbors[2].id, "bird");

    // Ascending distance.
    assert!(result.neighbors[0].distance < result.neighbors[1].distance);
    assert!(result.neighbors[1].distance < result.neighbors[2].distance);
}

pub fn test_search_boundary_vectors() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(40, 8, 0x1234)))
        .expect("index");

    // Zero, unit, negative and extreme queries all need to come back sane.
    for query in [
        vec![0.0f32; 8],
        vec![1.0f32; 8],
        vec![-1.0f32; 8],
        vec![f32::MAX; 8],
        vec![f32::MIN; 8],
    ] {
        let result = engine.search(query, 4);
        assert_eq!(result.neighbors.len(), 4);

        for neighbor in &result.neighbors {
            // Regression guard: large-magnitude components used to produce NaN
            // distances through the `|a|² + |b|² - 2ab` expansion.
            assert!(!neighbor.distance.is_nan(), "NaN for {}", neighbor.id);
        }
    }
}

pub fn test_search_k_greater_than_size() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(5, 4, 7)))
        .expect("index");

    // Must clamp to the corpus size, not pad or panic.
    let result = engine.search(vec![0.0; 4], 100);
    assert_eq!(result.neighbors.len(), 5);
}

pub fn test_search_empty_engine() {
    let engine = LunaVDB::new(None).expect("construct");
    let result = engine.search(vec![0.1, 0.2, 0.3], 5);
    assert_eq!(result.neighbors.len(), 0);
}

pub fn test_search_reports_work_done() {
    let mut engine = LunaVDB::new(Some(fast_options())).expect("construct");
    engine
        .index(resource(generate_test_data(2000, 32, 0xABC)))
        .expect("index");

    let result = engine.search(vec![0.1; 32], 10);
    assert!(!result.exact, "the index should be approximate here");
    assert!(result.scanned <= 2000);
    assert!(result.cells_probed > 0);

    let exact = engine.search_exact(vec![0.1; 32], 10);
    assert!(exact.exact);
    assert_eq!(exact.scanned, 2000);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

pub fn test_add_and_remove() {
    let mut engine = LunaVDB::new(None).expect("construct");

    engine
        .add(resource(vec![EmbeddedResource {
            id: "3".to_string(),
            embeddings: vec![0.7, 0.8, 0.9],
        }]))
        .expect("add");
    assert_eq!(engine.size(), 1);

    engine.remove(vec!["3".to_string()]).expect("remove");
    assert_eq!(engine.size(), 0);
    assert!(!engine.has("3"));
}

pub fn test_clear_stays_usable() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(100, 16, 3)))
        .expect("index");

    engine.clear();
    assert_eq!(engine.size(), 0);
    assert_eq!(engine.dimension(), 0);

    // Regression guard: the old `clear` reset the dimension but kept the index
    // structures, so the next `add` trained against stale centroids.
    engine
        .add(resource(vec![EmbeddedResource {
            id: "after".to_string(),
            embeddings: vec![1.0; 16],
        }]))
        .expect("add after clear");

    assert_eq!(engine.size(), 1);
    assert_eq!(engine.dimension(), 16);
    assert_eq!(engine.search(vec![1.0; 16], 1).neighbors[0].id, "after");
}

pub fn test_large_corpus_ingest() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(1000, 128, 9)))
        .expect("index");
    assert_eq!(engine.size(), 1000);

    engine
        .add(resource(generate_prefixed(200, 128, 10, "extra-")))
        .expect("add");
    assert_eq!(engine.size(), 1200);
}

// ---------------------------------------------------------------------------
// Errors — the null-pointer regression set
// ---------------------------------------------------------------------------

pub fn test_duplicate_id_throws() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .add(resource(vec![EmbeddedResource {
            id: "dup".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        }]))
        .expect("first add");

    assert!(
        engine
            .add(resource(vec![EmbeddedResource {
                id: "dup".to_string(),
                embeddings: vec![0.4, 0.5, 0.6],
            }]))
            .is_err(),
        "duplicate id must be reported"
    );

    // And the instance survives it.
    assert_eq!(engine.size(), 1);
    assert_eq!(engine.search(vec![0.0; 3], 1).neighbors.len(), 1);
}

pub fn test_dimension_mismatch_throws() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .add(resource(vec![EmbeddedResource {
            id: "a".to_string(),
            embeddings: vec![0.1; 8],
        }]))
        .expect("add");

    assert!(
        engine
            .add(resource(vec![EmbeddedResource {
                id: "b".to_string(),
                embeddings: vec![0.1; 4],
            }]))
            .is_err(),
        "dimension mismatch must be reported"
    );

    assert_eq!(engine.size(), 1);
    assert_eq!(engine.search(vec![0.0; 8], 1).neighbors.len(), 1);
}

pub fn test_remove_unknown_id_throws() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .add(resource(generate_test_data(10, 8, 11)))
        .expect("add");

    assert!(
        engine.remove(vec!["not-here".to_string()]).is_err(),
        "unknown id must be reported"
    );

    assert_eq!(engine.size(), 10);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

pub fn test_serialize_round_trip() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(300, 64, 0x5EED)))
        .expect("index");

    let bytes = engine.serialize(Some(true)).expect("serialize");
    assert!(!bytes.is_empty());
    assert!(is_snapshot(&bytes));
    assert_eq!(snapshot_version(&bytes), 2);

    let restored = LunaVDB::deserialize(&bytes, None).expect("deserialize");
    assert_eq!(restored.size(), 300);
    assert_eq!(restored.dimension(), 64);

    // Identical answers, not merely similar ones. The corpus is regenerated
    // rather than reused so the query vectors are rebuilt from the same seed
    // the engine was indexed with, which is what makes them exact hits.
    let data = generate_test_data(300, 64, 0x5EED);
    for seed in [1usize, 50, 150, 299] {
        let query = data[seed].embeddings.clone();
        assert_eq!(
            engine.search(query.clone(), 10),
            restored.search(query, 10),
            "mismatch at {seed}"
        );
    }
}

pub fn test_serialize_uncompressed() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(150, 32, 0x77)))
        .expect("index");

    let raw = engine.serialize(Some(false)).expect("serialize");
    let packed = engine.serialize(Some(true)).expect("serialize");

    let from_raw = LunaVDB::deserialize(&raw, None).expect("deserialize raw");
    let from_packed = LunaVDB::deserialize(&packed, None).expect("deserialize packed");
    assert_eq!(from_raw.size(), 150);
    assert_eq!(from_packed.size(), 150);

    let query = vec![0.25f32; 32];
    assert_eq!(
        from_raw.search(query.clone(), 5),
        from_packed.search(query, 5)
    );
}

pub fn test_serialize_after_mutation() {
    let mut engine = LunaVDB::new(Some(fast_options())).expect("construct");
    engine
        .index(resource(generate_test_data(400, 32, 0x31)))
        .expect("index");
    engine
        .add(resource(generate_prefixed(60, 32, 0x32, "added-")))
        .expect("add");

    let remove: Vec<String> = (0..40).map(|i| format!("vec-{i}")).collect();
    engine.remove(remove).expect("remove");

    let bytes = engine.serialize(Some(true)).expect("serialize");
    let restored = LunaVDB::deserialize(&bytes, None).expect("deserialize");

    assert_eq!(restored.size(), engine.size());

    let query = vec![0.1f32; 32];
    assert_eq!(engine.search(query.clone(), 10), restored.search(query, 10));
}

pub fn test_empty_round_trip() {
    let engine = LunaVDB::new(None).expect("construct");

    for compressed in [true, false] {
        let bytes = engine.serialize(Some(compressed)).expect("serialize");
        let restored = LunaVDB::deserialize(&bytes, None).expect("deserialize");
        assert_eq!(restored.size(), 0);
    }
}

/// The core null-pointer regression: every one of these used to unwind out of a
/// `.unwrap()` in the binding, or abort, or silently return garbage on the
/// *next* call. Each must now be an ordinary `Err` while the module stays alive.
pub fn test_corrupt_snapshot_is_recoverable() {
    let mut engine = LunaVDB::new(None).expect("construct");
    engine
        .index(resource(generate_test_data(200, 24, 0x99)))
        .expect("index");

    let good = engine.serialize(Some(true)).expect("serialize");

    for cut in [0usize, 1, 4, 8, 15, 16, 17, good.len() / 2, good.len() - 1] {
        assert!(
            LunaVDB::deserialize(&good[..cut], None).is_err(),
            "truncation to {cut} should error"
        );
    }

    for offset in [16usize, 32, good.len() / 2, good.len() - 1] {
        if offset >= good.len() {
            continue;
        }
        let mut corrupted = good.clone();
        corrupted[offset] ^= 0xFF;
        assert!(
            LunaVDB::deserialize(&corrupted, None).is_err(),
            "bit flip at {offset} should error"
        );
    }

    assert!(LunaVDB::deserialize(b"totally unrelated bytes here", None).is_err());
    assert!(LunaVDB::deserialize(&[], None).is_err());

    // The module is still fully functional.
    assert_eq!(engine.size(), 200);
    assert_eq!(engine.search(vec![0.0; 24], 5).neighbors.len(), 5);
    assert!(LunaVDB::deserialize(&good, None).is_ok());
}

pub fn test_restore_into_preserves_on_failure() {
    let mut source = LunaVDB::new(None).expect("construct");
    source
        .index(resource(generate_test_data(80, 16, 0x41)))
        .expect("index");
    let bytes = source.serialize(Some(true)).expect("serialize");

    let mut target = LunaVDB::new(None).expect("construct");
    target.restore_into(&bytes).expect("restore");
    assert_eq!(target.size(), 80);

    // A failed restore must not wipe what was already there.
    assert!(target.restore_into(b"garbage").is_err());
    assert_eq!(target.size(), 80);
}

// ---------------------------------------------------------------------------
// Metrics and diagnostics
// ---------------------------------------------------------------------------

pub fn test_cosine_is_scale_invariant() {
    let mut engine = LunaVDB::new(Some(LunaOptions {
        distance: Some("cosine".to_string()),
        ..Default::default()
    }))
    .expect("construct");

    engine
        .index(resource(vec![
            EmbeddedResource {
                id: "x".to_string(),
                embeddings: vec![1.0, 0.0, 0.0],
            },
            EmbeddedResource {
                id: "y".to_string(),
                embeddings: vec![0.0, 1.0, 0.0],
            },
        ]))
        .expect("index");

    // Same direction, 100x the magnitude.
    let result = engine.search(vec![100.0, 0.0, 0.0], 1);
    assert_eq!(result.neighbors[0].id, "x");
    assert!(result.neighbors[0].distance.abs() < 1e-5);

    // Zero query: no direction, so it must not produce NaN.
    let zero = engine.search(vec![0.0, 0.0, 0.0], 2);
    assert_eq!(zero.neighbors.len(), 2);
    assert!(zero.neighbors.iter().all(|n| !n.distance.is_nan()));
}

pub fn test_dot_product() {
    let mut engine = LunaVDB::new(Some(LunaOptions {
        distance: Some("dot".to_string()),
        ..Default::default()
    }))
    .expect("construct");

    engine
        .index(resource(vec![
            EmbeddedResource {
                id: "small".to_string(),
                embeddings: vec![1.0, 0.0],
            },
            EmbeddedResource {
                id: "big".to_string(),
                embeddings: vec![5.0, 0.0],
            },
        ]))
        .expect("index");

    let result = engine.search(vec![1.0, 0.0], 2);
    assert_eq!(result.neighbors[0].id, "big");
    // The reported value is the dot product, not the negated internal score.
    assert!((result.neighbors[0].distance - 5.0).abs() < 1e-5);
}

pub fn test_stats() {
    let mut engine = LunaVDB::new(Some(fast_options())).expect("construct");
    engine
        .index(resource(generate_test_data(1000, 16, 0x51)))
        .expect("index");

    let stats = engine.stats();
    assert_eq!(stats.size, 1000);
    assert_eq!(stats.dimension, 16);
    assert_eq!(stats.distance, "euclidean");
    assert!(stats.indexed);
    assert!(stats.nlist > 0);
    assert!(stats.nprobe > 0);
    assert!(stats.memory_bytes > 0);
    assert_eq!(stats.pending_deletes, 0);
    assert!(!stats.simd.is_empty());
}

pub fn test_simd_backend_is_known() {
    let backend = simd_backend();
    let known = ["wasm-simd128", "avx2", "neon", "scalar"];
    assert!(
        known.iter().any(|name| *name == backend),
        "unexpected backend {backend}"
    );

    let engine = LunaVDB::new(None).expect("construct");
    // `hasSimd` must agree with the reported backend.
    assert_eq!(engine.has_simd(), backend == "wasm-simd128");
}

pub fn test_version_reported() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

pub fn test_compaction_preserves_results() {
    let mut engine = LunaVDB::new(Some(fast_options())).expect("construct");
    let items = generate_test_data(500, 32, 0x61);
    engine.index(resource(items.clone())).expect("index");

    let query = items[300].embeddings.clone();
    let before = engine.search(query.clone(), 5);
    assert_eq!(before.neighbors[0].id, "vec-300");

    // Delete enough to cross the compaction ratio.
    let remove: Vec<String> = (0..250).map(|i| format!("vec-{i}")).collect();
    engine.remove(remove).expect("remove");
    assert_eq!(engine.size(), 250);

    // vec-300 survived the range, so it must still win.
    let after = engine.search(query, 5);
    assert_eq!(after.neighbors[0].id, "vec-300");

    // An explicit compact is idempotent for the caller.
    engine.compact();
    let post = engine.search(items[300].embeddings.clone(), 5);
    assert_eq!(post.neighbors[0].id, "vec-300");
}
