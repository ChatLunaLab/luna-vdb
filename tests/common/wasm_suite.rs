//! Shared wasm test suite, compiled into both `tests/web.rs` (browser) and
//! `tests/node.rs` (node).
//!
//! The two hosts run identical assertions — only the
//! `wasm_bindgen_test_configure!` line in the including file differs — so the
//! tests live here once, each carrying its own `#[wasm_bindgen_test]`. (The
//! previous layout listed every test again as a wrapper in both files, and the
//! lists had already drifted apart.)
//!
//! Everything here goes through the **public wasm-bindgen API**, so these tests
//! exercise the generated glue — argument decoding, `Result` → exception,
//! handle borrowing — rather than the engine directly. That is the layer where
//! the old null-pointer reports originated, and it is the layer `cargo test`
//! cannot reach.

use luna_vdb::{
    EmbeddedResource, LunaOptions, LunaVDB, Resource, SearchResult, is_snapshot, simd_backend,
    snapshot_version, version,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

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
/// prefixes are two disjoint id spaces. The mutation tests rely on this.
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

// -- JS-side argument builders -------------------------------------------
//
// The bindings take their structured arguments as `JsValue` (see the module
// docs in `src/wasm/luna_vdb.rs` for why), so tests build real JS values,
// exactly as a JS caller would.

fn resource(items: Vec<EmbeddedResource>) -> JsValue {
    serde_wasm_bindgen::to_value(&Resource { embeddings: items }).expect("serialise resource")
}

fn ids(list: Vec<String>) -> JsValue {
    serde_wasm_bindgen::to_value(&list).expect("serialise ids")
}

fn options(value: LunaOptions) -> JsValue {
    serde_wasm_bindgen::to_value(&value).expect("serialise options")
}

fn no_options() -> JsValue {
    JsValue::UNDEFINED
}

fn bytes(data: &[u8]) -> JsValue {
    js_sys::Uint8Array::from(data).into()
}

fn fast_options() -> JsValue {
    options(LunaOptions {
        ivf_threshold: Some(256),
        ..Default::default()
    })
}

fn new_db() -> LunaVDB {
    LunaVDB::new(no_options()).expect("construct")
}

fn ids_of(result: &SearchResult) -> Vec<String> {
    result.neighbors.iter().map(|n| n.id.clone()).collect()
}

/// Bitwise CRC-32 (IEEE), independent of the crate's table-driven one, so a
/// test does not trust the code under test to build its own input.
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = !0u32;
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

// ---------------------------------------------------------------------------
// Construction and indexing
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_new_empty() {
    let engine = new_db();
    assert_eq!(engine.size(), 0);
    assert_eq!(engine.dimension(), 0);
    assert_eq!(engine.distance(), "euclidean");
    assert!(!engine.has("anything"));

    // `null` options behave like omitted ones.
    assert!(LunaVDB::new(JsValue::NULL).is_ok());
}

#[wasm_bindgen_test]
pub fn test_new_with_options() {
    let engine = LunaVDB::new(options(LunaOptions {
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
#[wasm_bindgen_test]
pub fn test_invalid_option_does_not_poison() {
    assert!(
        LunaVDB::new(options(LunaOptions {
            distance: Some("hamming".to_string()),
            ..Default::default()
        }))
        .is_err()
    );

    // Wrong *shape*, not just a wrong value.
    assert!(LunaVDB::new(JsValue::from_str("cosine")).is_err());

    // The module still works.
    let engine = new_db();
    assert_eq!(engine.size(), 0);
}

/// 0.0.x took the initial vectors in the constructor. That call must still
/// build them, not decode as empty options and drop them.
#[wasm_bindgen_test]
pub fn test_new_with_a_resource_indexes_it() {
    let engine = LunaVDB::new(resource(vec![
        EmbeddedResource {
            id: "a".to_string(),
            embeddings: vec![1.0, 0.0],
        },
        EmbeddedResource {
            id: "b".to_string(),
            embeddings: vec![0.0, 1.0],
        },
    ]))
    .expect("construct");

    assert_eq!(engine.size(), 2);
    assert_eq!(engine.distance(), "euclidean");
    let hits = engine.search(vec![0.0, 1.0], 1).expect("search");
    assert_eq!(ids_of(&hits), vec!["b"]);

    // A bad batch throws, like `index()`.
    assert!(
        LunaVDB::new(resource(vec![
            EmbeddedResource {
                id: "a".to_string(),
                embeddings: vec![1.0],
            },
            EmbeddedResource {
                id: "a".to_string(),
                embeddings: vec![2.0],
            },
        ]))
        .is_err()
    );
}

#[wasm_bindgen_test]
pub fn test_index_and_size() {
    let mut engine = new_db();

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

#[wasm_bindgen_test]
pub fn test_index_replaces_contents() {
    let mut engine = new_db();

    engine
        .index(resource(generate_test_data(50, 8, 1)))
        .expect("first index");
    assert_eq!(engine.size(), 50);

    engine
        .index(resource(generate_test_data(10, 8, 2)))
        .expect("second index");
    assert_eq!(engine.size(), 10);

    // A failed rebuild leaves the previous contents alone. NaN does not
    // survive a JS round trip as a number literal, so use an empty vector.
    let mut bad = generate_test_data(5, 8, 3);
    bad[2].embeddings.clear();
    assert!(engine.index(resource(bad)).is_err());
    assert_eq!(engine.size(), 10);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_search_ranks_neighbours() {
    let mut engine = new_db();

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

    let result = engine
        .search(vec![0.8, 0.7, 0.6, 0.2, 0.1], 3)
        .expect("search");
    assert_eq!(ids_of(&result), vec!["cat", "dog", "bird"]);

    // Ascending distance.
    assert!(result.neighbors[0].distance < result.neighbors[1].distance);
    assert!(result.neighbors[1].distance < result.neighbors[2].distance);
}

#[wasm_bindgen_test]
pub fn test_search_boundary_vectors() {
    let mut engine = new_db();
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
        let result = engine.search(query, 4).expect("search");
        assert_eq!(result.neighbors.len(), 4);

        for neighbor in &result.neighbors {
            assert!(!neighbor.distance.is_nan(), "NaN for {}", neighbor.id);
        }
    }

    // A query with no meaningful distances throws instead of returning noise.
    let mut nan = vec![0.0f32; 8];
    nan[3] = f32::NAN;
    assert!(engine.search(nan, 4).is_err());
    assert!(engine.search_exact(vec![f32::INFINITY; 8], 4).is_err());
}

#[wasm_bindgen_test]
pub fn test_search_k_greater_than_size() {
    let mut engine = new_db();
    engine
        .index(resource(generate_test_data(5, 4, 7)))
        .expect("index");

    // Must clamp to the corpus size, not pad, allocate for `k`, or panic. JS
    // `-1` arrives as `u32::MAX`.
    for k in [100usize, u32::MAX as usize] {
        let result = engine.search(vec![0.0; 4], k).expect("search");
        assert_eq!(result.neighbors.len(), 5);
    }
}

#[wasm_bindgen_test]
pub fn test_search_empty_engine() {
    let engine = new_db();
    let result = engine.search(vec![0.1, 0.2, 0.3], 5).expect("search");
    assert_eq!(result.neighbors.len(), 0);
}

#[wasm_bindgen_test]
pub fn test_default_search_is_exact() {
    let mut engine = LunaVDB::new(fast_options()).expect("construct");
    engine
        .index(resource(generate_test_data(2000, 32, 0xABC)))
        .expect("index");
    assert!(engine.stats().indexed);

    let mut next = rng(0xD1CE);
    for _ in 0..10 {
        let query: Vec<f32> = (0..32).map(|_| next()).collect();
        let fast = engine.search(query.clone(), 10).expect("search");
        let slow = engine.search_exact(query, 10).expect("search");
        assert!(fast.exact);
        assert_eq!(ids_of(&fast), ids_of(&slow));
    }
}

#[wasm_bindgen_test]
pub fn test_approximate_mode_reports_itself() {
    let mut engine = LunaVDB::new(options(LunaOptions {
        ivf_threshold: Some(256),
        approximate: Some(true),
        nprobe: Some(2),
        ..Default::default()
    }))
    .expect("construct");
    engine
        .index(resource(generate_test_data(2000, 32, 0xABC)))
        .expect("index");

    let result = engine.search(vec![0.1; 32], 10).expect("search");
    assert!(!result.exact, "approximate mode should say so");
    assert!(result.scanned < 2000);
    assert!(result.cells_probed > 0);
    assert!(engine.stats().approximate);

    let exact = engine.search_exact(vec![0.1; 32], 10).expect("search");
    assert!(exact.exact);
    assert_eq!(exact.scanned, 2000);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_add_and_remove() {
    let mut engine = new_db();

    engine
        .add(resource(vec![EmbeddedResource {
            id: "3".to_string(),
            embeddings: vec![0.7, 0.8, 0.9],
        }]))
        .expect("add");
    assert_eq!(engine.size(), 1);

    engine.remove(ids(vec!["3".to_string()])).expect("remove");
    assert_eq!(engine.size(), 0);
    assert!(!engine.has("3"));
}

#[wasm_bindgen_test]
pub fn test_clear_stays_usable() {
    let mut engine = new_db();
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
    let hit = engine.search(vec![1.0; 16], 1).expect("search");
    assert_eq!(hit.neighbors[0].id, "after");
}

#[wasm_bindgen_test]
pub fn test_large_corpus_ingest() {
    let mut engine = new_db();
    engine
        .index(resource(generate_prefixed(1000, 128, 9, "base-")))
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

#[wasm_bindgen_test]
pub fn test_duplicate_id_throws() {
    let mut engine = new_db();
    engine
        .add(resource(vec![EmbeddedResource {
            id: "dup".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        }]))
        .expect("first add");

    let error = engine
        .add(resource(vec![EmbeddedResource {
            id: "dup".to_string(),
            embeddings: vec![0.4, 0.5, 0.6],
        }]))
        .expect_err("duplicate id must be reported");
    assert!(
        error.is_instance_of::<js_sys::Error>(),
        "must be a real Error"
    );

    // And the instance survives it.
    assert_eq!(engine.size(), 1);
    let result = engine.search(vec![0.0; 3], 1).expect("search");
    assert_eq!(result.neighbors.len(), 1);
}

#[wasm_bindgen_test]
pub fn test_dimension_mismatch_throws() {
    let mut engine = new_db();
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
    let result = engine.search(vec![0.0; 8], 1).expect("search");
    assert_eq!(result.neighbors.len(), 1);
}

#[wasm_bindgen_test]
pub fn test_add_batch_is_all_or_nothing() {
    let mut engine = new_db();
    let mut batch = generate_test_data(10, 4, 0x42);
    batch[7].embeddings = vec![0.0; 3];

    assert!(engine.add(resource(batch)).is_err());
    assert_eq!(engine.size(), 0, "nothing from a failed batch is kept");

    let mut batch = generate_test_data(10, 4, 0x42);
    batch[9].id = batch[1].id.clone();
    assert!(engine.add(resource(batch)).is_err());
    assert_eq!(engine.size(), 0);
}

#[wasm_bindgen_test]
pub fn test_remove_unknown_id_throws() {
    let mut engine = new_db();
    engine
        .add(resource(generate_test_data(10, 8, 11)))
        .expect("add");

    assert!(
        engine
            .remove(ids(vec!["vec-1".to_string(), "not-here".to_string()]))
            .is_err(),
        "unknown id must be reported"
    );

    // All-or-nothing: `vec-1` was not removed either.
    assert_eq!(engine.size(), 10);
    assert!(engine.has("vec-1"));
}

/// REGRESSION: a malformed argument used to throw while wasm-bindgen still
/// held the handle's borrow, so every later call failed with "recursive use of
/// an object detected" and the handle could not even be freed.
#[wasm_bindgen_test]
pub fn test_malformed_arguments_do_not_lock_the_handle() {
    let mut engine = new_db();
    engine
        .add(resource(generate_test_data(5, 4, 0x77)))
        .expect("add");

    let numeric_id =
        js_sys::JSON::parse(r#"{"embeddings":[{"id":42,"embeddings":[1,2,3,4]}]}"#).expect("json");
    assert!(engine.add(numeric_id).is_err());
    assert!(engine.add(JsValue::from_str("not a resource")).is_err());
    assert!(engine.index(JsValue::NULL).is_err());
    assert!(engine.remove(JsValue::from_f64(42.0)).is_err());
    let numeric_ids = js_sys::JSON::parse("[1, 2]").expect("json");
    assert!(engine.remove(numeric_ids).is_err());
    assert!(engine.restore_into(JsValue::from_str("bytes")).is_err());

    // Every kind of call still works on the same handle.
    assert_eq!(engine.size(), 5);
    assert!(engine.search(vec![0.0; 4], 2).is_ok());
    engine
        .add(resource(generate_prefixed(1, 4, 0x78, "late-")))
        .expect("add after bad calls");
    engine
        .remove(ids(vec!["late-0".to_string()]))
        .expect("remove");
    engine.clear();
    assert_eq!(engine.size(), 0);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_serialize_round_trip() {
    let mut engine = new_db();
    let data = generate_test_data(300, 64, 0x5EED);
    engine.index(resource(data.clone())).expect("index");

    let serialized = engine.serialize(Some(true)).expect("serialize");
    assert!(!serialized.is_empty());
    assert!(is_snapshot(&serialized));
    assert_eq!(snapshot_version(&serialized), 2);

    let restored = LunaVDB::deserialize(bytes(&serialized), no_options()).expect("deserialize");
    assert_eq!(restored.size(), 300);
    assert_eq!(restored.dimension(), 64);

    // Identical answers, not merely similar ones.
    for row in [1usize, 50, 150, 299] {
        let query = data[row].embeddings.clone();
        assert_eq!(
            engine.search(query.clone(), 10).expect("search"),
            restored.search(query, 10).expect("search"),
            "mismatch at {row}"
        );
    }
}

#[wasm_bindgen_test]
pub fn test_deserialize_accepts_an_array_buffer() {
    // REGRESSION: the `&[u8]` parameter read an `ArrayBuffer` as zero bytes
    // and reported a valid snapshot as truncated.
    let mut engine = new_db();
    engine
        .index(resource(generate_test_data(20, 4, 0xAB)))
        .expect("index");
    let serialized = engine.serialize(None).expect("serialize");

    let array = js_sys::Uint8Array::from(serialized.as_slice());
    let buffer: JsValue = array.buffer().into();
    assert!(buffer.is_instance_of::<js_sys::ArrayBuffer>());

    let restored = LunaVDB::deserialize(buffer.clone(), no_options()).expect("from ArrayBuffer");
    assert_eq!(restored.size(), 20);

    let mut target = new_db();
    target
        .restore_into(buffer)
        .expect("restore from ArrayBuffer");
    assert_eq!(target.size(), 20);

    // Something that is neither: a clear TypeError.
    let error = LunaVDB::deserialize(JsValue::from_f64(1.0), no_options()).expect_err("not bytes");
    assert!(error.is_instance_of::<js_sys::TypeError>());
    assert!(error.dyn_ref::<js_sys::Error>().is_some());
}

#[wasm_bindgen_test]
pub fn test_serialize_uncompressed() {
    let mut engine = new_db();
    engine
        .index(resource(generate_test_data(150, 32, 0x77)))
        .expect("index");

    let raw = engine.serialize(Some(false)).expect("serialize");
    let packed = engine.serialize(Some(true)).expect("serialize");

    let from_raw = LunaVDB::deserialize(bytes(&raw), no_options()).expect("deserialize raw");
    let from_packed =
        LunaVDB::deserialize(bytes(&packed), no_options()).expect("deserialize packed");
    assert_eq!(from_raw.size(), 150);
    assert_eq!(from_packed.size(), 150);

    let query = vec![0.25f32; 32];
    assert_eq!(
        from_raw.search(query.clone(), 5).expect("search"),
        from_packed.search(query, 5).expect("search")
    );
}

#[wasm_bindgen_test]
pub fn test_serialize_after_mutation() {
    let mut engine = LunaVDB::new(fast_options()).expect("construct");
    engine
        .index(resource(generate_test_data(400, 32, 0x31)))
        .expect("index");
    engine
        .add(resource(generate_prefixed(60, 32, 0x32, "added-")))
        .expect("add");

    let remove: Vec<String> = (0..40).map(|i| format!("vec-{i}")).collect();
    engine.remove(ids(remove)).expect("remove");

    let serialized = engine.serialize(Some(true)).expect("serialize");
    let restored = LunaVDB::deserialize(bytes(&serialized), no_options()).expect("deserialize");

    assert_eq!(restored.size(), engine.size());

    let query = vec![0.1f32; 32];
    assert_eq!(
        engine.search(query.clone(), 10).expect("search"),
        restored.search(query, 10).expect("search")
    );
}

#[wasm_bindgen_test]
pub fn test_empty_round_trip() {
    let engine = new_db();

    for compressed in [true, false] {
        let serialized = engine.serialize(Some(compressed)).expect("serialize");
        let restored = LunaVDB::deserialize(bytes(&serialized), no_options()).expect("deserialize");
        assert_eq!(restored.size(), 0);
    }
}

/// The core null-pointer regression: every one of these used to unwind out of a
/// `.unwrap()` in the binding, or abort, or silently return garbage on the
/// *next* call. Each must now be an ordinary `Err` while the module stays alive.
#[wasm_bindgen_test]
pub fn test_corrupt_snapshot_is_recoverable() {
    let mut engine = new_db();
    engine
        .index(resource(generate_test_data(200, 24, 0x99)))
        .expect("index");

    let good = engine.serialize(Some(true)).expect("serialize");

    for cut in [0usize, 1, 4, 8, 15, 16, 17, good.len() / 2, good.len() - 1] {
        assert!(
            LunaVDB::deserialize(bytes(&good[..cut]), no_options()).is_err(),
            "truncation to {cut} should error"
        );
    }

    // Every header byte as well as the payload: the checksum covers the
    // version, flags and length fields, so a flipped length can never reach
    // an allocation.
    let mut offsets: Vec<usize> = (4..16).collect();
    offsets.extend([16usize, 32, good.len() / 2, good.len() - 1]);
    for offset in offsets {
        let mut corrupted = good.clone();
        corrupted[offset] ^= 0xFF;
        assert!(
            LunaVDB::deserialize(bytes(&corrupted), no_options()).is_err(),
            "bit flip at {offset} should error"
        );
    }

    // A hand-built header with a *valid* checksum that claims a 4 GiB
    // compressed payload: this used to reserve the 4 GiB before reading
    // anything, a capacity-overflow trap on wasm32.
    let mut header = Vec::new();
    header.extend_from_slice(b"LVD2");
    header.extend_from_slice(&2u16.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&u32::MAX.to_le_bytes());
    let crc = crc32_ieee(&header[4..12]);
    header.extend_from_slice(&crc.to_le_bytes());
    assert!(LunaVDB::deserialize(bytes(&header), no_options()).is_err());

    assert!(LunaVDB::deserialize(bytes(b"totally unrelated bytes here"), no_options()).is_err());
    assert!(LunaVDB::deserialize(bytes(&[]), no_options()).is_err());

    // The module is still fully functional.
    assert_eq!(engine.size(), 200);
    let result = engine.search(vec![0.0; 24], 5).expect("search");
    assert_eq!(result.neighbors.len(), 5);
    assert!(LunaVDB::deserialize(bytes(&good), no_options()).is_ok());
}

#[wasm_bindgen_test]
pub fn test_restore_into_preserves_on_failure() {
    let mut source = new_db();
    source
        .index(resource(generate_test_data(80, 16, 0x41)))
        .expect("index");
    let serialized = source.serialize(Some(true)).expect("serialize");

    let mut target = new_db();
    target.restore_into(bytes(&serialized)).expect("restore");
    assert_eq!(target.size(), 80);

    // A failed restore must not wipe what was already there.
    assert!(target.restore_into(bytes(b"garbage")).is_err());
    assert_eq!(target.size(), 80);
}

#[wasm_bindgen_test]
pub fn test_restore_keeps_the_snapshot_metric_for_rebuilds() {
    // REGRESSION: `index()` rebuilt with the handle's own options, so after
    // restoring a cosine snapshot into a default handle the next `index()`
    // silently switched the metric to euclidean.
    let mut cosine = LunaVDB::new(options(LunaOptions {
        distance: Some("cosine".to_string()),
        ..Default::default()
    }))
    .expect("construct");
    cosine
        .index(resource(generate_test_data(30, 4, 0x55)))
        .expect("index");
    let serialized = cosine.serialize(None).expect("serialize");

    let mut restored = LunaVDB::deserialize(bytes(&serialized), no_options()).expect("deserialize");
    assert_eq!(restored.distance(), "cosine");
    restored
        .index(resource(generate_test_data(10, 4, 0x56)))
        .expect("rebuild");
    assert_eq!(restored.distance(), "cosine");

    let mut target = new_db();
    target.restore_into(bytes(&serialized)).expect("restore");
    target
        .index(resource(generate_test_data(10, 4, 0x57)))
        .expect("rebuild");
    assert_eq!(target.distance(), "cosine");
}

#[wasm_bindgen_test]
pub fn test_deserialize_honours_nprobe() {
    let mut engine = LunaVDB::new(fast_options()).expect("construct");
    engine
        .index(resource(generate_test_data(2000, 8, 0x66)))
        .expect("index");
    let serialized = engine.serialize(None).expect("serialize");

    let tuned = LunaVDB::deserialize(
        bytes(&serialized),
        options(LunaOptions {
            nprobe: Some(3),
            ..Default::default()
        }),
    )
    .expect("deserialize");
    assert_eq!(tuned.stats().nprobe, 3);
}

// ---------------------------------------------------------------------------
// Metrics and diagnostics
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_cosine_is_scale_invariant() {
    let mut engine = LunaVDB::new(options(LunaOptions {
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
    let result = engine.search(vec![100.0, 0.0, 0.0], 1).expect("search");
    assert_eq!(result.neighbors[0].id, "x");
    assert!(result.neighbors[0].distance.abs() < 1e-5);

    // Zero query: no direction, so it must not produce NaN.
    let zero = engine.search(vec![0.0, 0.0, 0.0], 2).expect("search");
    assert_eq!(zero.neighbors.len(), 2);
    assert!(zero.neighbors.iter().all(|n| !n.distance.is_nan()));
}

#[wasm_bindgen_test]
pub fn test_dot_product() {
    let mut engine = LunaVDB::new(options(LunaOptions {
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

    let result = engine.search(vec![1.0, 0.0], 2).expect("search");
    assert_eq!(result.neighbors[0].id, "big");
    // The reported value is the dot product, not the negated internal score.
    assert!((result.neighbors[0].distance - 5.0).abs() < 1e-5);
}

#[wasm_bindgen_test]
pub fn test_stats() {
    let mut engine = LunaVDB::new(fast_options()).expect("construct");
    engine
        .index(resource(generate_test_data(1000, 16, 0x51)))
        .expect("index");

    let stats = engine.stats();
    assert_eq!(stats.size, 1000);
    assert_eq!(stats.dimension, 16);
    assert_eq!(stats.distance, "euclidean");
    assert!(stats.indexed);
    assert!(!stats.approximate);
    assert!(stats.nlist > 0);
    assert!(stats.nprobe > 0);
    assert!(stats.memory_bytes > 0);
    assert_eq!(stats.pending_deletes, 0);
    assert_eq!(stats.simd, simd_backend());
}

#[wasm_bindgen_test]
pub fn test_simd_backend_is_known() {
    let backend = simd_backend();

    // The name comes from the kernel module that is actually compiled in, so
    // this fails if a SIMD build is ever routed to the scalar kernels — the
    // failure CI's opcode grep cannot see, because autovectorised loops
    // elsewhere still emit vector instructions.
    let expected = if cfg!(target_feature = "simd128") {
        "wasm-simd128"
    } else {
        "scalar"
    };
    assert_eq!(backend, expected);

    let engine = new_db();
    assert_eq!(engine.has_simd(), backend == "wasm-simd128");
}

#[wasm_bindgen_test]
pub fn test_version_reported() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
pub fn test_compaction_preserves_results() {
    let mut engine = LunaVDB::new(fast_options()).expect("construct");
    let items = generate_test_data(500, 32, 0x61);
    engine.index(resource(items.clone())).expect("index");

    let query = items[300].embeddings.clone();
    let before = engine.search(query.clone(), 5).expect("search");
    assert_eq!(before.neighbors[0].id, "vec-300");

    // Delete enough to cross the compaction ratio.
    let remove: Vec<String> = (0..250).map(|i| format!("vec-{i}")).collect();
    engine.remove(ids(remove)).expect("remove");
    assert_eq!(engine.size(), 250);

    // vec-300 survived the range, so it must still win.
    let after = engine.search(query, 5).expect("search");
    assert_eq!(after.neighbors[0].id, "vec-300");

    // An explicit compact is idempotent for the caller.
    engine.compact();
    let post = engine
        .search(items[300].embeddings.clone(), 5)
        .expect("search");
    assert_eq!(post.neighbors[0].id, "vec-300");
}
