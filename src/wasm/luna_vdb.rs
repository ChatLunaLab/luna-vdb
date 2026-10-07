//! wasm-bindgen surface.
//!
//! # The contract
//!
//! **Nothing in this module panics on user input.** Every fallible operation
//! returns `Result<_, JsValue>`, which `wasm-bindgen` turns into a normal
//! throwable JS `Error`. This is a change from the previous version, where
//! `add`, `remove`, `serialize` and `deserialize` all ended in `.unwrap()`.
//!
//! That `.unwrap()` is the entire explanation for the "神秘的空指针" reports.
//! When a Rust panic unwinds out of a wasm export, the wasm instance's state is
//! poisoned: the shadow stack and the allocator are left inconsistent, so the
//! *next* call into the module returns garbage instead of a value — which JS
//! then reports as a null pointer, or as `undefined`, or as a silently wrong
//! result. The panic itself is long gone from the console by then, so the crash
//! appears to come from an unrelated line.
//!
//! Concrete triggers that hit it in the old code:
//!
//! * `deserialize()` on a truncated, corrupt, or older-format snapshot →
//!   `bincode::deserialize_from(..).unwrap()`.
//! * `add()` with a duplicate id or a mismatched dimension →
//!   `engine::add(..).unwrap()`.
//! * `remove()` of an id that is not present → `engine::remove(..).unwrap()`.
//! * `serialize()` on an engine whose dimension no longer matches its rows.
//!
//! All four are now ordinary `catch (e)` cases.

use wasm_bindgen::prelude::*;

use crate::engine::{
    self, Engine, IndexOptions,
    types::{Distance, EngineError},
};
use crate::utils::set_panic_hook;

// ---------------------------------------------------------------------------
// Transfer types
// ---------------------------------------------------------------------------

/// A single `(id, vector)` pair, as consumed by `index`/`add`.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, Debug, Clone, PartialEq)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct EmbeddedResource {
    pub id: String,
    pub embeddings: Vec<f32>,
}

/// A batch of [`EmbeddedResource`].
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, Debug, Clone, PartialEq)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct Resource {
    pub embeddings: Vec<EmbeddedResource>,
}

/// One search hit.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, PartialEq, Debug)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct Neighbor {
    pub id: String,
    pub distance: f32,
}

/// Search results plus diagnostics.
///
/// `scanned`/`rescored`/`cellsProbed` let a caller see how much work the index
/// did — useful for choosing `nprobe` without guessing.
///
/// `rename_all` is load-bearing. Without it `cells_probed` goes over the wire
/// as `cells_probed` while `LunaOptions` accepts `ivfThreshold`, so the JS
/// surface contradicts itself and every field beyond the first word silently
/// reads `undefined` — which is exactly what the package smoke test caught.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, PartialEq, Debug, Default)]
#[tsify(from_wasm_abi, into_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub neighbors: Vec<Neighbor>,
    /// Rows examined.
    pub scanned: usize,
    /// Candidates that received an exact rescore.
    pub rescored: usize,
    /// Coarse cells probed.
    pub cells_probed: usize,
    /// `true` when the answer is exact.
    pub exact: bool,
}

impl From<engine::SearchOutcome> for SearchResult {
    fn from(outcome: engine::SearchOutcome) -> Self {
        Self {
            neighbors: outcome
                .neighbors
                .into_iter()
                .map(|neighbor| Neighbor {
                    id: neighbor.id,
                    distance: neighbor.distance,
                })
                .collect(),
            scanned: outcome.candidates_scored,
            rescored: outcome.rescored,
            cells_probed: outcome.cells_probed,
            exact: outcome.exact,
        }
    }
}

/// Index statistics.
///
/// camelCase for the same reason as [`SearchResult`]: `pendingDeletes` and
/// `memoryBytes` must match what `LunaOptions` and the rest of the JS-facing
/// API use.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, PartialEq, Debug)]
#[tsify(into_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub size: usize,
    pub dimension: usize,
    /// `"euclidean" | "cosine" | "dot"`.
    pub distance: String,
    /// Cells in the coarse quantiser; `0` when there is no index.
    pub nlist: usize,
    pub nprobe: usize,
    /// `true` once IVF is in use.
    pub indexed: bool,
    /// `true` once PQ codes are in use.
    pub pq: bool,
    /// Rows awaiting compaction.
    pub pending_deletes: usize,
    /// Approximate heap footprint in bytes.
    pub memory_bytes: usize,
    /// Active SIMD backend: `"wasm-simd128" | "avx2" | "neon" | "scalar"`.
    pub simd: String,
}

/// Construction options, all optional on the JS side.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, Debug, Clone, Default)]
#[tsify(from_wasm_abi)]
#[serde(default, rename_all = "camelCase")]
pub struct LunaOptions {
    /// `"euclidean" | "cosine" | "dot"`.
    pub distance: Option<String>,
    /// Coarse cells. `null` picks a default from the corpus size.
    pub nlist: Option<usize>,
    /// Cells probed per query.
    pub nprobe: Option<usize>,
    /// PQ subquantisers.
    pub pq_m: Option<usize>,
    /// PQ centroids per subquantiser, max 256.
    pub pq_ksub: Option<usize>,
    /// Corpus size at which IVF is built.
    pub ivf_threshold: Option<usize>,
    /// Disable the PQ prefilter for exact rescoring.
    pub exact_rescore_only: Option<bool>,
}

impl LunaOptions {
    fn into_engine(self) -> EngineResult<IndexOptions> {
        let mut options = IndexOptions::default();

        if let Some(distance) = &self.distance {
            options.distance = Distance::parse(distance).ok_or_else(|| {
                EngineError::new(format!(
                    "unknown distance {distance:?}; expected \"euclidean\", \"cosine\" or \"dot\""
                ))
            })?;
        }

        if let Some(nlist) = self.nlist {
            options.nlist = Some(nlist);
        }
        if let Some(nprobe) = self.nprobe {
            options.nprobe = Some(nprobe);
        }
        if let Some(pq_m) = self.pq_m {
            options.pq_m = Some(pq_m);
        }
        if let Some(pq_ksub) = self.pq_ksub {
            options.pq_ksub = Some(pq_ksub);
        }
        if let Some(ivf_threshold) = self.ivf_threshold {
            options.ivf_threshold = ivf_threshold;
        }
        if let Some(exact) = self.exact_rescore_only {
            options.exact_rescore_only = exact;
        }

        Ok(options)
    }
}

type EngineResult<T> = Result<T, EngineError>;

/// Convert an engine error into a JS exception.
///
/// On wasm the error class is a real `Error`, so `instanceof Error` works in a
/// `catch` and the stack is preserved. We deliberately do *not* throw a bare
/// string: that loses the stack and breaks `instanceof`, which is part of why a
/// failed `deserialize` used to look like a mystery rather than an error.
///
/// The native arm exists so the whole wasm-facing API — error paths included —
/// is covered by plain `cargo test` instead of only by the browser suite.
#[cfg(target_arch = "wasm32")]
fn to_js(error: EngineError) -> JsValue {
    js_sys::Error::new(&error.message).into()
}

#[cfg(not(target_arch = "wasm32"))]
fn to_js(error: EngineError) -> JsValue {
    JsValue::from_str(&error.message)
}

// ---------------------------------------------------------------------------
// LunaVDB
// ---------------------------------------------------------------------------

/// The database handle.
#[wasm_bindgen]
#[derive(Debug)]
pub struct LunaVDB {
    engine: Engine,
    /// Retained so `deserialize` into an existing handle keeps the original
    /// search-time tuning instead of silently reverting to defaults.
    options: IndexOptions,
}

#[wasm_bindgen]
impl LunaVDB {
    /// Create an empty database.
    ///
    /// ```js
    /// const db = new LunaVDB();
    /// const db = new LunaVDB({ distance: "cosine" });
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<LunaOptions>) -> Result<LunaVDB, JsValue> {
        set_panic_hook();

        let options = options.unwrap_or_default().into_engine().map_err(to_js)?;

        Ok(LunaVDB {
            engine: Engine::new(options),
            options,
        })
    }

    /// Build from a batch, replacing any existing contents.
    pub fn index(&mut self, resource: Resource) -> Result<(), JsValue> {
        let mut data = Vec::with_capacity(resource.embeddings.len());
        let mut ids = Vec::with_capacity(resource.embeddings.len());

        for item in resource.embeddings {
            data.push(item.embeddings);
            ids.push(item.id);
        }

        self.engine = Engine::build(&data, &ids, self.options).map_err(to_js)?;
        Ok(())
    }

    /// Add a batch. Duplicate ids and dimension mismatches throw.
    pub fn add(&mut self, resource: Resource) -> Result<(), JsValue> {
        for item in resource.embeddings {
            self.engine.add(item.id, &item.embeddings).map_err(to_js)?;
        }
        Ok(())
    }

    /// Remove ids. An unknown id throws and the batch stops there.
    pub fn remove(&mut self, ids: Vec<String>) -> Result<(), JsValue> {
        self.engine.remove(&ids).map_err(to_js)
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.engine.clear();
    }

    /// `k` nearest neighbours of `query`.
    pub fn search(&self, query: Vec<f32>, k: usize) -> SearchResult {
        self.engine.search(&query, k).into()
    }

    /// Exact brute-force search, ignoring the index. For measuring recall.
    #[wasm_bindgen(js_name = searchExact)]
    pub fn search_exact(&self, query: Vec<f32>, k: usize) -> SearchResult {
        self.engine.search_exact(&query, k).into()
    }

    /// Number of stored vectors.
    pub fn size(&self) -> usize {
        self.engine.len()
    }

    /// Vector dimension, or `0` before anything is stored.
    pub fn dimension(&self) -> usize {
        self.engine.dim()
    }

    /// Effective distance metric.
    pub fn distance(&self) -> String {
        self.engine.distance().as_str().to_string()
    }

    /// `true` if `id` is present.
    pub fn has(&self, id: &str) -> bool {
        self.engine.contains(id)
    }

    /// Index and runtime statistics. See [`Stats`].
    pub fn stats(&self) -> Stats {
        Stats {
            size: self.engine.len(),
            dimension: self.engine.dim(),
            distance: self.engine.distance().as_str().to_string(),
            nlist: self.engine.nlist(),
            nprobe: self.engine.nprobe(),
            indexed: self.engine.is_indexed(),
            pq: self.engine.has_pq(),
            pending_deletes: self.engine.dead_rows(),
            memory_bytes: self.engine.memory_bytes(),
            simd: engine::simd::KERNEL_NAME.to_string(),
        }
    }

    /// Force compaction of tombstones. Normally automatic.
    pub fn compact(&mut self) {
        self.engine.compact();
    }

    /// Serialise to `Uint8Array`.
    ///
    /// `compressed` defaults to `true`; pass `false` when the bytes are going
    /// straight back into `LunaVDB.deserialize` in the same process, where the
    /// compress/decompress round trip costs more than the smaller transfer.
    #[wasm_bindgen(js_name = serialize)]
    pub fn serialize(&self, compressed: Option<bool>) -> Result<Vec<u8>, JsValue> {
        self.engine
            .serialize(compressed.unwrap_or(true))
            .map_err(to_js)
    }

    /// Restore from `serialize()` output.
    ///
    /// Accepts snapshots written by luna-vdb ≤ 0.0.12 (the gzip/bincode format)
    /// as well as the current format. A truncated, corrupt or oversized payload
    /// throws an `Error`; it never aborts the instance.
    ///
    /// `options` overrides search-time tuning — `nprobe`, mainly. Without it the
    /// restored handle uses defaults, which matters when the snapshot was
    /// written with a tuned `nprobe`.
    pub fn deserialize(bytes: &[u8], options: Option<LunaOptions>) -> Result<LunaVDB, JsValue> {
        let parsed = options.unwrap_or_default().into_engine().map_err(to_js)?;

        let engine = engine::load_with_options(bytes, parsed).map_err(to_js)?;

        Ok(LunaVDB {
            engine,
            options: parsed,
        })
    }

    /// Restore *in place*, reusing this handle.
    ///
    /// Cheaper than `deserialize` when the caller already holds a handle: no
    /// second wasm-side allocation survives the call.
    #[wasm_bindgen(js_name = restoreInto)]
    pub fn restore_into(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.engine = engine::load_with_options(bytes, self.options).map_err(to_js)?;
        Ok(())
    }

    /// `true` when this build was compiled with wasm SIMD128.
    #[wasm_bindgen(js_name = hasSimd)]
    pub fn has_simd(&self) -> bool {
        crate::has_simd()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Which SIMD backend this build uses.
#[wasm_bindgen(js_name = simdBackend)]
pub fn simd_backend() -> String {
    engine::simd::KERNEL_NAME.to_string()
}

/// Current crate version, for diagnostics.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// `true` if `bytes` looks like a snapshot this build can read.
///
/// Cheap header sniff — useful before committing to a large `deserialize` call,
/// or for telling the user which file is the problem.
#[wasm_bindgen(js_name = isSnapshot)]
pub fn is_snapshot(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && (&bytes[..4] == b"LVD2" || (bytes[0] == 0x1f && bytes[1] == 0x8b))
}

/// Snapshot format version of `bytes`, or `0` if unrecognised.
#[wasm_bindgen(js_name = snapshotVersion)]
pub fn snapshot_version(bytes: &[u8]) -> u16 {
    if bytes.len() >= 6 && &bytes[..4] == b"LVD2" {
        u16::from_le_bytes([bytes[4], bytes[5]])
    } else if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        1
    } else {
        0
    }
}
