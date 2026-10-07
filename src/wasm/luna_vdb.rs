//! wasm-bindgen surface.
//!
//! # The contract
//!
//! **Nothing in this module panics on user input, and no bad argument can
//! leave a handle unusable.** Every fallible operation returns
//! `Result<_, JsValue>`, which `wasm-bindgen` turns into a normal throwable JS
//! `Error`.
//!
//! That is the entire explanation for the "神秘的空指针" reports. When a Rust
//! panic unwinds out of a wasm export, the wasm instance's state is poisoned:
//! the shadow stack and the allocator are left inconsistent, so the *next*
//! call into the module returns garbage instead of a value — which JS then
//! reports as a null pointer, or as `undefined`, or as a silently wrong
//! result. The panic itself is long gone from the console by then, so the crash
//! appears to come from an unrelated line.
//!
//! Concrete triggers in the pre-1.0 bindings:
//!
//! * `deserialize()` on a truncated, corrupt, or older-format snapshot →
//!   `bincode::deserialize_from(..).unwrap()`.
//! * `add()` with a duplicate id or a mismatched dimension →
//!   `engine::add(..).unwrap()`.
//! * `remove()` of an id that is not present → `engine::remove(..).unwrap()`.
//! * `serialize()` on an engine whose dimension no longer matches its rows.
//!
//! # Why the structured arguments are `JsValue`
//!
//! `index`, `add` and `remove` take their argument as a raw `JsValue` and
//! decode it inside the method body. Declaring them as `Resource` /
//! `Vec<String>` looks equivalent and is not: wasm-bindgen takes the handle's
//! `&mut self` borrow *before* converting the arguments, and a failed
//! conversion throws straight out of the export without running destructors.
//! The borrow is never released, so one call with a malformed argument —
//! `add({ embeddings: [{ id: 42, embeddings: [1] }] })` — left the handle
//! permanently locked: every later call threw "recursive use of an object
//! detected", and even `free()` failed. Decoding in the body turns the same
//! mistake into an ordinary `Error` from that one call. The TypeScript
//! signatures are unchanged, via `unchecked_param_type`.

use wasm_bindgen::JsCast;
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
/// did. `exact` is `true` unless the handle was built with
/// `approximate: true`.
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
    /// Candidates that received an exact rescore after a PQ prefilter.
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
    /// Cells probed per query in approximate mode.
    pub nprobe: usize,
    /// `true` once IVF is in use.
    pub indexed: bool,
    /// `true` when searches are approximate (`approximate: true`).
    pub approximate: bool,
    /// `true` once PQ codes are in use.
    pub pq: bool,
    /// Rows awaiting compaction.
    pub pending_deletes: usize,
    /// Approximate heap footprint in bytes.
    pub memory_bytes: usize,
    /// Active SIMD backend: `"wasm-simd128" | "scalar"`.
    pub simd: String,
}

/// Construction options, all optional on the JS side.
#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify, Debug, Clone, Default)]
#[tsify(from_wasm_abi)]
#[serde(default, rename_all = "camelCase")]
pub struct LunaOptions {
    /// `"euclidean" | "cosine" | "dot"`.
    pub distance: Option<String>,
    /// Coarse cells. `null` picks `sqrt(size)`.
    pub nlist: Option<usize>,
    /// Cells probed per query in approximate mode.
    pub nprobe: Option<usize>,
    /// PQ subquantisers.
    pub pq_m: Option<usize>,
    /// PQ centroids per subquantiser, max 256.
    pub pq_ksub: Option<usize>,
    /// Corpus size at which the IVF index is built. Default 4096.
    pub ivf_threshold: Option<usize>,
    /// `true` (default) skips product quantisation. Set `false` to train a PQ
    /// prefilter; it is used only when `approximate` is also `true`.
    pub exact_rescore_only: Option<bool>,
    /// `false` (default): every search returns the exact nearest neighbours.
    /// `true`: probe a fixed `nprobe` cells instead — faster on data with
    /// little cluster structure, at the cost of recall.
    pub approximate: Option<bool>,
}

impl LunaOptions {
    fn into_engine(self) -> Result<IndexOptions, EngineError> {
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
        if let Some(approximate) = self.approximate {
            options.approximate = approximate;
        }

        Ok(options)
    }
}

// ---------------------------------------------------------------------------
// Argument decoding
// ---------------------------------------------------------------------------

/// Convert an engine error into a JS exception.
///
/// The error class is a real `Error`, so `instanceof Error` works in a `catch`
/// and the stack is preserved. We deliberately do *not* throw a bare string:
/// that loses the stack and breaks `instanceof`, which is part of why a failed
/// `deserialize` used to look like a mystery rather than an error.
fn to_js(error: EngineError) -> JsValue {
    js_sys::Error::new(&error.message).into()
}

fn type_error(message: &str) -> JsValue {
    js_sys::TypeError::new(message).into()
}

/// Decode a structured argument, reporting a shape mismatch as a `TypeError`
/// that names the argument.
fn decode<T: serde::de::DeserializeOwned>(value: JsValue, what: &str) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|error| type_error(&format!("invalid {what}: {error}")))
}

fn decode_options(value: JsValue) -> Result<IndexOptions, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(IndexOptions::default());
    }
    decode::<LunaOptions>(value, "options")?
        .into_engine()
        .map_err(to_js)
}

/// Copy snapshot bytes out of JS.
///
/// Accepts a `Uint8Array` (including a Node `Buffer`) or an `ArrayBuffer`. The
/// previous `&[u8]` parameter took anything with a `length`: an `ArrayBuffer`,
/// which has none, arrived as zero bytes and was reported as a truncated
/// snapshot — an error message that sent people looking for corruption that
/// was not there.
fn bytes_from_js(value: &JsValue) -> Result<Vec<u8>, JsValue> {
    if let Some(array) = value.dyn_ref::<js_sys::Uint8Array>() {
        return Ok(array.to_vec());
    }
    if let Some(buffer) = value.dyn_ref::<js_sys::ArrayBuffer>() {
        return Ok(js_sys::Uint8Array::new(buffer).to_vec());
    }
    Err(type_error(
        "snapshot must be a Uint8Array or an ArrayBuffer",
    ))
}

/// The first bytes of a snapshot, without copying the rest — enough for the
/// header sniffs, which used to copy the whole snapshot into wasm memory.
fn header_from_js(value: &JsValue) -> Vec<u8> {
    let view = if let Some(array) = value.dyn_ref::<js_sys::Uint8Array>() {
        array.clone()
    } else if let Some(buffer) = value.dyn_ref::<js_sys::ArrayBuffer>() {
        js_sys::Uint8Array::new(buffer)
    } else {
        return Vec::new();
    };
    view.subarray(0, engine::codec::HEADER_LEN as u32).to_vec()
}

fn check_query(query: &[f32]) -> Result<(), JsValue> {
    match query.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(type_error(&format!(
            "query has a non-finite value (NaN or infinity) at index {index}"
        ))),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// LunaVDB
// ---------------------------------------------------------------------------

/// The database handle.
#[wasm_bindgen]
#[derive(Debug)]
pub struct LunaVDB {
    engine: Engine,
    /// Retained so `index()` rebuilds with the same tuning, and so a restore
    /// keeps the caller's search-time options.
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
    pub fn new(
        #[wasm_bindgen(unchecked_optional_param_type = "LunaOptions")] options: JsValue,
    ) -> Result<LunaVDB, JsValue> {
        set_panic_hook();

        let options = decode_options(options)?;
        Ok(LunaVDB {
            engine: Engine::new(options),
            options,
        })
    }

    /// Build from a batch, replacing any existing contents. A failed build
    /// leaves the existing contents in place.
    pub fn index(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Resource")] resource: JsValue,
    ) -> Result<(), JsValue> {
        let resource: Resource = decode(resource, "resource")?;

        let mut data = Vec::with_capacity(resource.embeddings.len());
        let mut ids = Vec::with_capacity(resource.embeddings.len());
        for item in resource.embeddings {
            data.push(item.embeddings);
            ids.push(item.id);
        }

        self.engine = Engine::build(&data, &ids, self.options).map_err(to_js)?;
        Ok(())
    }

    /// Add a batch. All-or-nothing: a duplicate id, a dimension mismatch or a
    /// non-finite value anywhere in the batch throws and adds nothing.
    pub fn add(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Resource")] resource: JsValue,
    ) -> Result<(), JsValue> {
        let resource: Resource = decode(resource, "resource")?;
        let items: Vec<(String, Vec<f32>)> = resource
            .embeddings
            .into_iter()
            .map(|item| (item.id, item.embeddings))
            .collect();

        self.engine.add_many(&items).map_err(to_js)
    }

    /// Remove ids. All-or-nothing: an unknown id throws and removes nothing.
    pub fn remove(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "string[]")] ids: JsValue,
    ) -> Result<(), JsValue> {
        let ids: Vec<String> = decode(ids, "ids")?;
        self.engine.remove(&ids).map_err(to_js)
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.engine.clear();
    }

    /// `k` nearest neighbours of `query`. `k` larger than `size()` returns
    /// every vector. A query containing NaN or infinity throws.
    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<SearchResult, JsValue> {
        check_query(&query)?;
        Ok(self.engine.search(&query, k).into())
    }

    /// Brute-force search, ignoring the index. For measuring recall of the
    /// approximate mode; the default `search` already returns this answer.
    #[wasm_bindgen(js_name = searchExact)]
    pub fn search_exact(&self, query: Vec<f32>, k: usize) -> Result<SearchResult, JsValue> {
        check_query(&query)?;
        Ok(self.engine.search_exact(&query, k).into())
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
            approximate: self.engine.options().approximate,
            pq: self.engine.has_pq(),
            pending_deletes: self.engine.dead_rows(),
            memory_bytes: self.engine.memory_bytes(),
            simd: crate::simd_backend().to_string(),
        }
    }

    /// Force compaction of tombstones. Normally automatic.
    pub fn compact(&mut self) {
        self.engine.compact();
    }

    /// Serialise to a `Uint8Array`.
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

    /// Restore from `serialize()` output (a `Uint8Array` or `ArrayBuffer`).
    ///
    /// Accepts snapshots written by luna-vdb ≤ 0.0.12 (the gzip/bincode format)
    /// as well as the current format. A truncated, corrupt or oversized payload
    /// throws an `Error`; it never aborts the instance.
    ///
    /// `options` overrides search-time tuning — `nprobe` and `approximate`.
    /// The distance metric always comes from the snapshot, because the stored
    /// vectors were prepared for it.
    pub fn deserialize(
        #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "LunaOptions")] options: JsValue,
    ) -> Result<LunaVDB, JsValue> {
        set_panic_hook();

        let options = decode_options(options)?;
        let bytes = bytes_from_js(&bytes)?;
        let engine = engine::load_with_options(&bytes, options).map_err(to_js)?;

        Ok(LunaVDB {
            options: engine.options(),
            engine,
        })
    }

    /// Restore *in place*, reusing this handle and its options. A failed
    /// restore leaves the current contents untouched.
    #[wasm_bindgen(js_name = restoreInto)]
    pub fn restore_into(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: JsValue,
    ) -> Result<(), JsValue> {
        let bytes = bytes_from_js(&bytes)?;
        let engine = engine::load_with_options(&bytes, self.options).map_err(to_js)?;
        // Keep the snapshot's metric for later `index()` calls, so a rebuild
        // does not silently switch e.g. a cosine index to euclidean.
        self.options = engine.options();
        self.engine = engine;
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
//
// The Rust names collide with the crate-root helpers of the same name, which
// shadow these glob re-exports for Rust callers; JS sees them by `js_name`.
// ---------------------------------------------------------------------------

/// Which SIMD backend this build uses.
#[wasm_bindgen(js_name = simdBackend)]
pub fn simd_backend() -> String {
    crate::simd_backend().to_string()
}

/// Current crate version, for diagnostics.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    crate::version().to_string()
}

/// `true` if `bytes` looks like a snapshot this build can read.
///
/// Cheap header sniff — useful before committing to a large `deserialize` call,
/// or for telling the user which file is the problem.
#[wasm_bindgen(js_name = isSnapshot)]
pub fn is_snapshot(
    #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: JsValue,
) -> bool {
    engine::codec::is_snapshot(&header_from_js(&bytes))
}

/// Snapshot format version of `bytes`, or `0` if unrecognised.
#[wasm_bindgen(js_name = snapshotVersion)]
pub fn snapshot_version(
    #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: JsValue,
) -> u16 {
    engine::codec::snapshot_version(&header_from_js(&bytes))
}
