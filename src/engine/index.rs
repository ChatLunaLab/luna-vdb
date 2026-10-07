//! The vector index: an IVF partition over a flat arena. Searched
//! approximately by default, at a recall target the index calibrates for
//! itself, or exactly on request.
//!
//! # Approximate by default, at a measured recall
//!
//! The default search probes the `nprobe` cells whose centroids rank closest
//! to the query and scores every live row in them exactly. Fewer cells is
//! faster and finds fewer of the true neighbours, so `nprobe` is not a fixed
//! fraction: after every training run the index measures it. Stored rows serve
//! as leave-one-out queries, their exact neighbours are found, and `nprobe` is
//! set to the fewest cells that recover 95% of those neighbours
//! (`TARGET_RECALL`, `Engine::calibrate_nprobe`). On clustered data — real embeddings —
//! that is a handful of cells. On data with no structure at all it would be
//! most of them, at no real saving over a scan, so calibration then sets
//! `nprobe = nlist` and the search is exact instead of quietly returning the
//! wrong rows. (The pre-1.0 engine probed a fixed fraction and measured
//! 0.000–0.005 recall@10 on such data.) An explicit `nprobe` turns the
//! calibration off.
//!
//! # Exact on request
//!
//! With `approximate: false` the search returns the same neighbours as a
//! brute-force scan. The IVF cells are then used for *pruning*, not for
//! guessing: every cell records the largest distance from its centroid to any
//! row filed in it (its radius), and by the triangle inequality no row in a
//! cell can be closer to the query than `d(query, centroid) - radius`. Cells
//! are visited in order of that lower bound, and the scan stops at the first
//! cell whose bound is worse than the current k-th best — every cell after it
//! is provably worse too. Dot product uses the Cauchy–Schwarz form of the same
//! bound, `q·x <= q·c + |q|·radius`. The bounds are loose in high dimensions,
//! so this is typically 1–2x faster than a flat scan rather than 10x.
//!
//! The approximate search falls back to the exact one whenever the probe would
//! cover every cell anyway, or the probed cells hold fewer than `k` rows.
//! `SearchOutcome::exact` says which kind of answer came back.
//!
//! # What changed from the pre-1.0 engine
//!
//! 1. **The scan was scalar and scattered.** See [`crate::engine::simd`] and
//!    [`crate::engine::types::VectorStore`].
//! 2. **`add()` retrained the whole index.** `rebuild_ivf_pq` re-ran k-means
//!    over every vector on every insert, so ingesting N vectors was `O(N²)`.
//!    Insertion now files the vector in its nearest existing cell, which is
//!    `O(nlist · d)` and independent of `N`.
//! 3. **Deletion rebuilt the index too.** Rows are now tombstoned and squeezed
//!    out in one compaction pass once the dead fraction crosses a threshold.
//! 4. **Above 20 000 vectors the index returned almost nothing useful.** The
//!    old IVF-PQ path measured recall@10 of 0.000–0.005 on the comparison
//!    benchmark (`compare/`); below 20 000 it was an unindexed scalar scan.
//!
//! `compare/` runs this engine against the last pre-rewrite commit on identical
//! data, so the speedup can be checked rather than taken on faith.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::engine::codec::{self, PqSnapshot, Snapshot};
use crate::engine::ids::IdMap;
use crate::engine::kmeans::{self, KMeansConfig};
use crate::engine::simd;
use crate::engine::types::{Distance, Embedding, EngineError, EngineResult, VectorStore};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Build an IVF index once the corpus reaches this many vectors.
///
/// Below it a flat SIMD scan is exact, fast — 4 096 × 1536-d is about a
/// millisecond — and free to build. Above it the pruned search is still exact
/// and is never meaningfully slower than the flat scan, so the only cost of
/// indexing early is the k-means run at build time.
pub const DEFAULT_IVF_THRESHOLD: usize = 4_096;

/// Recall@[`CALIBRATION_K`] the automatic `nprobe` is chosen to reach on the
/// calibration queries. Above the 0.9 callers asked for, because the estimate
/// comes from queries drawn from the corpus itself; real queries can sit
/// further from the stored vectors than that.
const TARGET_RECALL: f32 = 0.95;

/// Neighbours per calibration query — the `k` the calibrated `nprobe` is for.
/// See [`Engine::effective_nprobe`] for larger `k`.
const CALIBRATION_K: usize = 10;

/// Stored rows used as calibration queries. Each costs one exact search per
/// training run; 32 queries of 10 neighbours put the standard error of the
/// recall estimate around 0.01–0.02.
const CALIBRATION_QUERIES: usize = 32;

/// Calibration never picks fewer cells than this. The calibration queries are
/// stored rows, which sit inside the clusters; a real query can land between
/// two, where a single cell holds only part of its neighbourhood. On clustered
/// data one cell often reaches the target alone, and the second costs a few
/// hundredths of a millisecond.
const MIN_CALIBRATED_NPROBE: usize = 2;

/// If the calibrated probe would still scan this fraction of the rows, probe
/// every cell instead — which makes the search exact. On data with no cluster
/// structure the target needs most cells, and those are the large ones near
/// the middle of the data: CI measured 86–95% of rows scanned for 1.0–1.1x
/// over the flat scan at recall 0.92–0.97. Full recall at the same speed is
/// the better deal.
const EXACT_FALLBACK_FRACTION: f32 = 0.5;

/// Fallback `nprobe` when nothing was calibrated, as a fraction of `nlist`.
const DEFAULT_NPROBE_RATIO: f32 = 0.05;

/// Fallback `nprobe` floor.
const MIN_DEFAULT_NPROBE: usize = 8;

/// Approximate mode with PQ: candidates to rescore per requested neighbour.
/// The PQ prefilter is approximate, so we rescore a multiple of `k` and take
/// the best `k` exactly.
const RESCORE_FACTOR: usize = 8;

/// Minimum candidates to rescore, so small `k` still has slack.
const MIN_RESCORE_CANDIDATES: usize = 64;

/// Retrain once this fraction of the corpus has been added since the last
/// training run. Drift never affects correctness — the radii keep the bounds
/// valid — but it loosens them, which costs speed.
const RETRAIN_GROWTH_RATIO: f32 = 0.25;

/// Never retrain for fewer additions than this, however small the corpus.
const MIN_RETRAIN_GROWTH: usize = 256;

/// Compact once this fraction of rows are tombstones.
const DEFAULT_COMPACT_RATIO: f32 = 0.3;

/// Only build PQ codes when there are enough vectors to give k-means something
/// to work with per subquantiser.
const PQ_MIN_VECTORS: usize = 2_048;

/// Coarse k-means trains on this many points per centroid, not on the whole
/// corpus. k-means cost is `points × centroids × d × iterations`; training on
/// every row made `index()` of 50 000 × 1536-d vectors take 20–30 s, while the
/// quality of the partition stops improving long before that.
const SAMPLES_PER_CENTROID: usize = 48;

/// Same idea for PQ codebooks, per codeword.
const PQ_SAMPLES_PER_CODEWORD: usize = 64;

/// Hard cap on any training sample.
const MAX_TRAINING_SAMPLES: usize = 65_536;

/// Lloyd iterations for both quantisers.
const KMEANS_ITERATIONS: usize = 10;

/// Relative slack subtracted from every pruning bound.
///
/// The bound is `d(q, c) - r`, a difference of two rounded f32 sums, so it
/// carries an absolute error proportional to `d(q, c) + r`. The worst-case
/// relative error of an f32 sum of `d` squares, accumulated in four or more
/// independent lanes, is about `(d / 4) · 2⁻²⁴` — under `1e-4` even at
/// `d = 4096`. Shrinking the bound by `1e-3` of that sum means rounding can
/// only make the search scan *more* cells than necessary, never prune one
/// that holds a true neighbour, at a cost in pruning too small to measure.
const BOUND_SLACK: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Build-time and runtime tuning. All fields have sensible defaults; the whole
/// point of the defaults is that callers never have to touch this.
#[derive(Debug, Clone, Copy)]
pub struct IndexOptions {
    pub distance: Distance,
    /// Coarse cells. `None` picks `round(sqrt(n))`.
    pub nlist: Option<usize>,
    /// Approximate mode only: cells probed per query, for `k` up to 10. `None`
    /// calibrates it after each training run to reach recall@10 of 0.95;
    /// see the module docs.
    pub nprobe: Option<usize>,
    /// PQ subquantisers. `None` derives one byte per 8 dimensions, rounded so
    /// `dim % m == 0`.
    pub pq_m: Option<usize>,
    /// PQ centroids per subquantiser. Capped at 256 so a code fits one byte.
    pub pq_ksub: Option<usize>,
    /// Corpus size at which IVF is built.
    pub ivf_threshold: usize,
    /// Additions since training, as a fraction of the corpus, before retraining.
    pub retrain_growth_ratio: f32,
    /// Tombstone fraction before compaction.
    pub compact_ratio: f32,
    /// `false` trains PQ codes and uses them as a prefilter in approximate
    /// mode. The default, `true`, skips PQ entirely: no codebook training at
    /// build time, and every probed row is scored exactly. PQ is dropped again
    /// when calibration finds it cannot reach the recall target.
    pub exact_rescore_only: bool,
    /// `true` (default): probe the calibrated `nprobe` cells. `false`: return
    /// the exact nearest neighbours, using the cells only to prune. See the
    /// module docs.
    pub approximate: bool,
    /// Seed for k-means, so a snapshot is reproducible.
    pub seed: u64,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            distance: Distance::Euclidean,
            nlist: None,
            nprobe: None,
            pq_m: None,
            pq_ksub: None,
            ivf_threshold: DEFAULT_IVF_THRESHOLD,
            retrain_growth_ratio: RETRAIN_GROWTH_RATIO,
            compact_ratio: DEFAULT_COMPACT_RATIO,
            exact_rescore_only: true,
            approximate: true,
            seed: 0x5EED_5EED_5EED_5EED,
        }
    }
}

impl IndexOptions {
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }
}

// ---------------------------------------------------------------------------
// IVF state
// ---------------------------------------------------------------------------

/// Coarse quantiser plus inverted lists.
#[derive(Debug, Clone, Default)]
struct Ivf {
    /// `nlist × dim`, row-major.
    centroids: Vec<f32>,
    /// `nlist` lists of row indices. Stale rows stay until compaction.
    lists: Vec<Vec<u32>>,
    /// Per cell, an upper bound on the L2 distance from the centroid to any
    /// row ever filed in the cell. Only ever grows between rebuilds, so it
    /// stays a valid bound after tombstoning; it is recomputed from scratch
    /// whenever the lists are rebuilt. Not stored in snapshots — it is derived
    /// data, recomputed on load in one pass.
    radii: Vec<f32>,
}

impl Ivf {
    fn nlist(&self) -> usize {
        self.lists.len()
    }

    fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    fn centroid(&self, cell: usize, dim: usize) -> &[f32] {
        self.centroids
            .get(cell * dim..cell * dim + dim)
            .unwrap_or(&[])
    }

    /// Nearest centroid to `vector` by exact L2, and the squared distance to
    /// it. Ties and NaNs resolve to the lowest cell index.
    fn nearest_centroid(&self, vector: &[f32], dim: usize) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;

        for cell in 0..self.nlist() {
            let dist = simd::l2_sq(vector, self.centroid(cell, dim));
            if dist < best_dist {
                best_dist = dist;
                best = cell;
            }
        }

        (best, best_dist)
    }

    /// Record that a row at squared distance `dist_sq` was filed in `cell`.
    fn widen(&mut self, cell: usize, dist_sq: f32) {
        if let Some(radius) = self.radii.get_mut(cell) {
            let dist = dist_sq.sqrt();
            // A non-finite distance makes the bound useless; `INFINITY` turns
            // pruning off for this cell instead of letting a NaN slip through
            // the `max`.
            *radius = if dist.is_finite() {
                radius.max(dist)
            } else {
                f32::INFINITY
            };
        }
    }
}

/// Product quantiser: `m` subquantisers, each with its own codebook.
#[derive(Debug, Clone, Default)]
struct Pq {
    m: usize,
    ksub: usize,
    subvector_dim: usize,
    /// `m × ksub × subvector_dim`, row-major.
    codebooks: Vec<f32>,
}

impl Pq {
    fn is_empty(&self) -> bool {
        self.m == 0 || self.ksub == 0
    }

    /// Encode one residual. Returns `m` bytes, one per subquantiser.
    fn encode(&self, residual: &[f32]) -> Vec<u8> {
        let mut code = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let start = sub * self.subvector_dim;
            let end = start + self.subvector_dim;
            let target = match residual.get(start..end) {
                Some(slice) => slice,
                None => {
                    code.push(0);
                    continue;
                }
            };

            let mut best = 0usize;
            let mut best_dist = f32::MAX;
            let base = sub * self.ksub * self.subvector_dim;

            for centroid in 0..self.ksub {
                let offset = base + centroid * self.subvector_dim;
                let candidate = &self.codebooks[offset..offset + self.subvector_dim];
                let dist = simd::l2_sq(target, candidate);
                if dist < best_dist {
                    best_dist = dist;
                    best = centroid;
                }
            }

            code.push(best as u8);
        }

        code
    }

    /// Squared L2 from `residual` to every centroid of every subquantiser.
    ///
    /// `m × ksub` floats, laid out per subquantiser. Codes encode the residual
    /// against a row's own cell, so the tables are built once per probed cell,
    /// from the query's residual against that cell's centroid.
    fn distance_tables(&self, residual: &[f32]) -> Vec<f32> {
        let mut tables = vec![0.0f32; self.m * self.ksub];

        for sub in 0..self.m {
            let start = sub * self.subvector_dim;
            let end = start + self.subvector_dim;
            let target = match residual.get(start..end) {
                Some(slice) => slice,
                None => continue,
            };

            let codebook_base = sub * self.ksub * self.subvector_dim;
            let table_base = sub * self.ksub;

            for centroid in 0..self.ksub {
                let offset = codebook_base + centroid * self.subvector_dim;
                let candidate = &self.codebooks[offset..offset + self.subvector_dim];
                tables[table_base + centroid] = simd::l2_sq(target, candidate);
            }
        }

        tables
    }

    /// Score one code against precomputed tables.
    #[inline]
    fn score_code(&self, code: &[u8], tables: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for sub in 0..self.m.min(code.len()) {
            let centroid = code[sub] as usize;
            let index = sub * self.ksub + centroid;
            if let Some(&value) = tables.get(index) {
                total += value;
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The index. Owns all state; every operation is a method so the borrow checker
/// enforces that a search cannot observe a half-applied insert.
#[derive(Debug, Clone)]
pub struct Engine {
    options: IndexOptions,
    store: VectorStore,
    ids: IdMap,
    dim: usize,
    distance: Distance,
    ivf: Ivf,
    pq: Pq,
    /// `slots × pq.m` codes, row-major. Flat for the same reason the vectors
    /// are: `Vec<Vec<u8>>` costs one allocation and one pointer chase per row.
    codes: Vec<u8>,
    nprobe: usize,
    /// Rows present when the last training run finished.
    rows_at_last_train: usize,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(IndexOptions::default())
    }
}

impl Engine {
    pub fn new(options: IndexOptions) -> Self {
        Self {
            distance: options.distance,
            options,
            store: VectorStore::new(0),
            ids: IdMap::new(),
            dim: 0,
            ivf: Ivf::default(),
            pq: Pq::default(),
            codes: Vec::new(),
            nprobe: 0,
            rows_at_last_train: 0,
        }
    }

    /// Build an engine from a column of vectors and ids.
    ///
    /// Every vector is validated first — non-empty, finite, unique id — so a
    /// bad item fails the whole build instead of producing a partial index.
    /// Vectors shorter than the longest one are zero-padded, which matches the
    /// pre-1.0 behaviour.
    pub fn build(data: &[Embedding], ids: &[String], options: IndexOptions) -> EngineResult<Self> {
        if data.len() != ids.len() {
            return Err(EngineError::new(format!(
                "embeddings and ids length mismatch: {} vs {}",
                data.len(),
                ids.len()
            )));
        }

        for (vector, id) in data.iter().zip(ids) {
            validate_vector(id, vector)?;
        }

        let mut engine = Engine::new(options);
        if data.is_empty() {
            return Ok(engine);
        }

        let dim = data.iter().map(|vector| vector.len()).max().unwrap_or(0);
        engine.dim = dim;
        engine.store = VectorStore::with_capacity(dim, data.len());
        engine.ids = IdMap::with_capacity(data.len());

        for (vector, id) in data.iter().zip(ids) {
            engine.ids.insert(id.clone())?;
            engine.push_prepared(vector);
        }

        engine.train();
        Ok(engine)
    }

    /// Rebuild from a parsed snapshot.
    ///
    /// `Snapshot` is a public type with public fields, so this cannot assume it
    /// came from [`codec::read`]: every size and index is checked here too, and
    /// an inconsistent snapshot is an error rather than a later panic.
    pub fn from_snapshot(snapshot: Snapshot, options: IndexOptions) -> EngineResult<Self> {
        let dim = snapshot.dim as usize;
        let count = snapshot.count as usize;

        // Multiplied in `u64`: on wasm32 a `usize` product of two large `u32`
        // fields can wrap to a small number that matches a short buffer, and
        // the slicing below would then panic.
        let expected = (count as u64)
            .checked_mul(dim as u64)
            .ok_or_else(|| EngineError::corrupt("count × dim overflow"))?;
        if snapshot.data.len() as u64 != expected {
            return Err(EngineError::corrupt(format!(
                "vector buffer holds {} floats, expected {expected}",
                snapshot.data.len()
            )));
        }
        if snapshot.ids.len() != count {
            return Err(EngineError::corrupt(format!(
                "{} ids for {count} vectors",
                snapshot.ids.len()
            )));
        }
        if count > 0 && dim == 0 {
            return Err(EngineError::corrupt("vectors with zero dimension"));
        }
        if snapshot.data.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::corrupt("vector data contains NaN or infinity"));
        }

        let mut store = VectorStore::with_capacity(dim, count);
        for row in 0..count {
            store.push(&snapshot.data[row * dim..row * dim + dim]);
        }

        let mut engine = Engine::new(options);
        // The metric belongs to the data, not to the caller's options: the
        // vectors were normalised (or not) for it when they were stored.
        engine.distance = snapshot.distance;
        engine.options.distance = snapshot.distance;
        engine.dim = dim;
        engine.store = store;
        engine.ids = IdMap::from_ids(snapshot.ids);
        engine.rows_at_last_train = count;

        let nlist = snapshot.nlist as usize;
        let has_index = nlist > 0 && count > 0 && !snapshot.centroids.is_empty();

        if has_index {
            let centroid_floats = (nlist as u64)
                .checked_mul(dim as u64)
                .ok_or_else(|| EngineError::corrupt("centroid count overflow"))?;
            if snapshot.centroids.len() as u64 != centroid_floats {
                return Err(EngineError::corrupt(format!(
                    "centroid buffer holds {} floats, expected {centroid_floats}",
                    snapshot.centroids.len()
                )));
            }
            if snapshot.centroids.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::corrupt("centroids contain NaN or infinity"));
            }
            if snapshot.lists.len() != nlist {
                return Err(EngineError::corrupt(format!(
                    "{} inverted lists for {nlist} cells",
                    snapshot.lists.len()
                )));
            }

            let lists_valid = lists_cover_rows_once(&snapshot.lists, count);
            engine.ivf = Ivf {
                centroids: snapshot.centroids,
                lists: snapshot.lists,
                radii: Vec::new(),
            };

            if lists_valid {
                engine.recompute_radii();
            } else {
                // A row missing from every list would be unreachable, and a
                // row in two lists would be returned twice. Our writer never
                // produces either, but re-filing is cheap insurance next to
                // returning a wrong answer.
                engine.assign_all_rows();
            }

            // The caller's `nprobe` wins over the stored one — it is a
            // search-time knob, and `load_with_options` promises exactly this.
            let stored = snapshot.nprobe as usize;
            engine.nprobe = options.nprobe.unwrap_or(stored).clamp(1, nlist);

            match snapshot.pq {
                Some(pq) => {
                    engine.install_pq(pq, snapshot.codes, count)?;
                }
                None => {
                    engine.train_pq();
                    // The stored `nprobe` was calibrated without a prefilter.
                    if engine.has_pq() && options.nprobe.is_none() {
                        engine.nprobe = engine.calibrate_nprobe();
                    }
                }
            }
        } else {
            // No index in the snapshot (a legacy import, or a corpus that was
            // below the threshold when written). Build one if it is due.
            engine.train();
        }

        // Duplicate ids in the snapshot became tombstones in `IdMap::from_ids`,
        // keeping every later id on its own vector. Sweep them now so the
        // engine is dense, as one built from scratch would be.
        if engine.ids.dead() > 0 {
            engine.compact();
        }

        Ok(engine)
    }

    /// Validate and adopt stored PQ codebooks and codes.
    fn install_pq(&mut self, pq: PqSnapshot, codes: Vec<u8>, count: usize) -> EngineResult<()> {
        let m = pq.m as usize;
        let ksub = pq.ksub as usize;
        let subvector_dim = pq.subvector_dim as usize;

        if m == 0 || subvector_dim == 0 || !(1..=256).contains(&ksub) {
            return Err(EngineError::corrupt(format!(
                "PQ shape m={m} ksub={ksub} subvector_dim={subvector_dim} is invalid"
            )));
        }
        if m.checked_mul(subvector_dim) != Some(self.dim) {
            return Err(EngineError::corrupt(format!(
                "PQ covers {m} × {subvector_dim} dimensions, the index has {}",
                self.dim
            )));
        }
        let codebook_len = (m as u64) * (ksub as u64) * (subvector_dim as u64);
        if pq.codebooks.len() as u64 != codebook_len {
            return Err(EngineError::corrupt(format!(
                "PQ codebooks hold {} floats, expected {codebook_len}",
                pq.codebooks.len()
            )));
        }
        if (codes.len() as u64) != (count as u64) * (m as u64) {
            return Err(EngineError::corrupt(format!(
                "PQ code buffer is {} bytes, expected {}",
                codes.len(),
                (count as u64) * (m as u64)
            )));
        }
        if let Some(&code) = codes.iter().find(|&&code| code as usize >= ksub) {
            return Err(EngineError::corrupt(format!(
                "PQ code {code} is outside the {ksub}-entry codebook"
            )));
        }

        self.pq = Pq {
            m,
            ksub,
            subvector_dim,
            codebooks: pq.codebooks,
        };
        self.codes = codes;
        Ok(())
    }

    // -- simple accessors -------------------------------------------------

    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn distance(&self) -> Distance {
        self.distance
    }

    #[inline]
    pub fn options(&self) -> IndexOptions {
        self.options
    }

    /// Tombstoned rows awaiting compaction.
    #[inline]
    pub fn dead_rows(&self) -> usize {
        self.ids.dead()
    }

    #[inline]
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Change how many cells an approximate query probes, without rebuilding.
    ///
    /// `nprobe` is purely a search-time knob — the cells and their contents do
    /// not depend on it — so retuning recall against latency should not cost a
    /// rebuild. The value also becomes the configured one, so a later retrain
    /// keeps it instead of recalibrating. Clamped to `1..=nlist`; ignored
    /// while there is no index. Has no effect on the exact search, which
    /// decides for itself which cells to visit.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        let nlist = self.ivf.nlist();
        if nlist == 0 {
            return;
        }
        let nprobe = nprobe.clamp(1, nlist);
        self.nprobe = nprobe;
        self.options.nprobe = Some(nprobe);
    }

    /// Switch between approximate (default) and exact search.
    pub fn set_approximate(&mut self, approximate: bool) {
        self.options.approximate = approximate;
    }

    /// Coarse cell count; `0` when there is no IVF index.
    #[inline]
    pub fn nlist(&self) -> usize {
        self.ivf.nlist()
    }

    /// `true` once an IVF index has been built.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        !self.ivf.is_empty()
    }

    /// `true` once PQ codes are in use.
    #[inline]
    pub fn has_pq(&self) -> bool {
        !self.pq.is_empty() && !self.codes.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    /// Rough heap footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.store.memory_bytes()
            + self.ids.memory_bytes()
            + self.ivf.centroids.len() * 4
            + self.ivf.radii.len() * 4
            + self
                .ivf
                .lists
                .iter()
                .map(|l| l.capacity() * 4)
                .sum::<usize>()
            + self.pq.codebooks.len() * 4
            + self.codes.capacity()
    }

    // -- mutation ---------------------------------------------------------

    /// Append one vector. `O(nlist · d)`, independent of the corpus size.
    ///
    /// Takes a slice rather than `&Embedding` so callers can pass an array
    /// literal or a borrowed row without allocating a `Vec` first.
    pub fn add(&mut self, id: String, vector: &[f32]) -> EngineResult<()> {
        validate_vector(&id, vector)?;

        if self.ids.contains(&id) {
            return Err(EngineError::duplicate_id(&id));
        }

        if self.dim == 0 && self.store.rows() == 0 {
            // First vector into an empty engine fixes the dimension. Checking
            // the store as well as `dim` matters: replacing a store that still
            // holds rows would detach every existing id from its vector.
            self.dim = vector.len();
            self.store = VectorStore::new(self.dim);
        } else if vector.len() != self.dim {
            return Err(EngineError::dimension_mismatch(self.dim, vector.len()));
        }

        // Id first, then row: both append, so the row index `insert` returns
        // is exactly the one `push_prepared` is about to use.
        self.ids.insert(id)?;
        let row = self.push_prepared(vector);

        // Incremental: place the row in its nearest existing cell and encode
        // it against the existing codebooks. No retraining.
        if self.ivf.is_empty() {
            self.maybe_train();
        } else {
            self.insert_into_ivf(row);
            self.maybe_retrain();
        }

        Ok(())
    }

    /// Add many vectors, atomically.
    ///
    /// The whole batch is validated first — ids unique against the index and
    /// within the batch, one dimension, every value finite — so a bad item
    /// leaves the engine exactly as it was rather than holding half a batch.
    pub fn add_many(&mut self, items: &[(String, Embedding)]) -> EngineResult<()> {
        let mut dim = if self.dim == 0 && self.store.rows() == 0 {
            None
        } else {
            Some(self.dim)
        };
        let mut seen: HashSet<&str> = HashSet::with_capacity(items.len());

        for (id, vector) in items {
            validate_vector(id, vector)?;
            if self.ids.contains(id) || !seen.insert(id.as_str()) {
                return Err(EngineError::duplicate_id(id));
            }
            match dim {
                None => dim = Some(vector.len()),
                Some(expected) if expected != vector.len() => {
                    return Err(EngineError::dimension_mismatch(expected, vector.len()));
                }
                Some(_) => {}
            }
        }

        for (id, vector) in items {
            self.add(id.clone(), vector)?;
        }
        Ok(())
    }

    /// Remove ids, atomically: every id is checked before any is removed, so
    /// an unknown id leaves the engine untouched. An id repeated within the
    /// call is removed once. Rows are tombstoned; compaction happens when the
    /// dead fraction crosses `options.compact_ratio`.
    pub fn remove(&mut self, ids: &[String]) -> EngineResult<()> {
        if let Some(missing) = ids.iter().find(|id| !self.ids.contains(id)) {
            return Err(EngineError::missing_id(missing));
        }

        let mut seen: HashSet<&str> = HashSet::with_capacity(ids.len());
        for id in ids {
            if seen.insert(id.as_str()) {
                self.ids.remove(id)?;
            }
        }

        if self.ids.is_empty() {
            self.reset();
            return Ok(());
        }

        self.maybe_compact();
        Ok(())
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.reset();
    }

    fn reset(&mut self) {
        self.store.clear();
        self.ids.clear();
        self.ivf = Ivf::default();
        self.pq = Pq::default();
        self.codes.clear();
        self.dim = 0;
        self.nprobe = 0;
        self.rows_at_last_train = 0;
    }

    /// Force compaction, regardless of the tombstone ratio.
    pub fn compact(&mut self) {
        if self.ids.dead() == 0 {
            return;
        }

        let slots = self.store.rows();

        // Build old-row → new-row mapping.
        let mut mapping: Vec<Option<u32>> = vec![None; slots];
        let mut next = 0u32;
        for (row, slot) in mapping.iter_mut().enumerate() {
            if self.ids.id_of(row).is_some() {
                *slot = Some(next);
                next += 1;
            }
        }

        self.ids.compact(&mapping);

        // Which *old* rows survive, in new-row order.
        //
        // `mapping` is old-row → new-row, so `mapping.iter().flatten()` yields
        // the new row numbers — using that as a keep list silently keeps the
        // wrong rows. With 80 dead rows at the front it keeps old rows
        // 0..420 (80 of them tombstones) and drops the live tail instead, so
        // every surviving id ends up attached to some other vector's data.
        // The list we need is the old rows that still have a mapping, and
        // because the mapping is monotonic that is already new-row order.
        let keep: Vec<u32> = (0..slots as u32)
            .filter(|row| mapping[*row as usize].is_some())
            .collect();
        self.store.retain_rows(&keep);

        // Rewrite codes. Rows are few and `m` is small, so a full rebuild is
        // simpler and faster than an in-place filter.
        if !self.pq.is_empty() {
            let m = self.pq.m;
            let mut codes = Vec::with_capacity(keep.len() * m);

            for &old_row in &keep {
                let start = old_row as usize * m;
                if let Some(slice) = self.codes.get(start..start + m) {
                    codes.extend_from_slice(slice);
                } else {
                    codes.resize(codes.len() + m, 0);
                }
            }

            self.codes = codes;
        }

        // Renumber the lists in place. Every surviving row stays in the cell it
        // was filed in, so the radii are still valid bounds — dropping rows can
        // only make a cell's true radius smaller. `O(n)`, where re-filing every
        // row against every centroid would be `O(n · nlist · d)`.
        for list in &mut self.ivf.lists {
            list.retain_mut(|row| match mapping.get(*row as usize).copied().flatten() {
                Some(new_row) => {
                    *row = new_row;
                    true
                }
                None => false,
            });
        }

        // Retraining is due after a fraction of the corpus has been *added*
        // since the last training run. Growth is measured as `rows -
        // rows_at_last_train`, so removing rows must lower the baseline by the
        // same amount, or the next retrain is pushed back by every deletion.
        let removed = slots - keep.len();
        self.rows_at_last_train = self.rows_at_last_train.saturating_sub(removed);
    }

    // -- search -----------------------------------------------------------

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// Approximate unless `options.approximate` is off; `SearchOutcome::exact`
    /// says which came back. `k` is clamped to the number of stored vectors, so an absurd `k`
    /// costs nothing extra. A query containing NaN or infinity has no
    /// meaningful neighbours and returns none — the wasm binding reports it as
    /// an error before getting here.
    pub fn search(&self, query: &[f32], k: usize) -> SearchOutcome {
        let k = k.min(self.len());
        if k == 0 || self.dim == 0 || !is_finite(query) {
            // Nothing to approximate: an empty answer is exact, and saying
            // otherwise would make a caller waste a `search_exact` call
            // verifying a trivially-correct result.
            return SearchOutcome {
                exact: true,
                ..SearchOutcome::default()
            };
        }

        let prepared = self.prepare_query(query);

        if self.ivf.is_empty() {
            return self.search_flat(&prepared, k);
        }
        // A probe of every cell would scan every row; the pruned search does
        // no more work than that and is exact.
        if self.options.approximate && self.effective_nprobe(k) < self.ivf.nlist() {
            return self.search_approximate(&prepared, k);
        }
        self.search_pruned(&prepared, k)
    }

    /// Brute-force search, ignoring the IVF index. Exists so callers can
    /// measure recall against ground truth — and so the benchmark has a
    /// baseline that does not depend on index quality.
    pub fn search_exact(&self, query: &[f32], k: usize) -> SearchOutcome {
        let k = k.min(self.len());
        if k == 0 || self.dim == 0 || !is_finite(query) {
            return SearchOutcome {
                exact: true,
                ..SearchOutcome::default()
            };
        }

        let prepared = self.prepare_query(query);
        self.search_flat(&prepared, k)
    }

    /// Coerce a query to the index dimension and, for Cosine, to unit length.
    ///
    /// A too-short query is zero-padded and an over-long one truncated, which
    /// matches the pre-1.0 `prepare_vector`.
    fn prepare_query(&self, query: &[f32]) -> Vec<f32> {
        let mut prepared = self.fit_dimension(query);
        if self.distance == Distance::Cosine {
            simd::normalize_in_place(&mut prepared);
        }
        prepared
    }

    /// Score every live row.
    ///
    /// A plain sorted `Vec` beats `BinaryHeap` as the top-`k` collector here:
    /// `k` is small (tens), the vector stays in cache, and there is no
    /// per-element allocation.
    fn search_flat(&self, query: &[f32], k: usize) -> SearchOutcome {
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k.saturating_add(1));
        let mut scored = 0usize;

        for row in 0..self.store.rows() {
            if self.ids.id_of(row).is_none() {
                continue;
            }
            scored += 1;
            // `score_row` normalises NaN to `+inf`, so no extra check here.
            insert_top(&mut top, self.score_row(row, query), row as u32, k);
        }

        SearchOutcome {
            neighbors: self.collect(&top),
            candidates_scored: scored,
            exact: true,
            ..SearchOutcome::default()
        }
    }

    /// Exact search over the IVF cells, pruned by the per-cell radius bound.
    /// See the module docs for why this is exact.
    fn search_pruned(&self, query: &[f32], k: usize) -> SearchOutcome {
        let (top, scored, probed) = self.pruned_top(query, k);
        SearchOutcome {
            neighbors: self.collect(&top),
            candidates_scored: scored,
            rescored: 0,
            cells_probed: probed,
            exact: true,
        }
    }

    /// The exact top `k` as `(score, row)`, with the rows scored and cells
    /// probed to find it.
    fn pruned_top(&self, query: &[f32], k: usize) -> (Vec<(f32, u32)>, usize, usize) {
        let query_norm = simd::norm_sq(query).sqrt();

        // (lower bound, tie-break, cell). Visiting in bound order lets the scan
        // stop at the first cell that cannot beat the current k-th best: every
        // later cell has a bound at least as large. Among cells whose bound is
        // equal — typically several at zero, when the query sits inside their
        // radius — the closest centroid goes first, so the k-th best tightens
        // as early as possible.
        let mut order: Vec<(f32, f32, u32)> = (0..self.ivf.nlist())
            .map(|cell| {
                let (bound, closeness) = self.cell_bound(query, query_norm, cell);
                (bound, closeness, cell as u32)
            })
            .collect();
        order.sort_unstable_by(|a, b| cmp_score(a.0, b.0).then(cmp_score(a.1, b.1)));

        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k.saturating_add(1));
        let mut scored = 0usize;
        let mut probed = 0usize;

        for &(bound, _, cell) in &order {
            if top.len() >= k && top.last().is_some_and(|&(worst, _)| bound > worst) {
                break;
            }

            probed += 1;
            for &row in &self.ivf.lists[cell as usize] {
                let row = row as usize;
                if self.ids.id_of(row).is_none() {
                    continue;
                }
                scored += 1;
                insert_top(&mut top, self.score_row(row, query), row as u32, k);
            }
        }

        (top, scored, probed)
    }

    /// A lower bound on the score of any row filed in `cell`, plus the
    /// centroid's own score as a tie-break.
    ///
    /// * Euclidean: `|q - x| >= |q - c| - r` (triangle inequality).
    /// * Cosine: rows and query are unit length (or zero), and for those
    ///   `1 - q·x >= |q - x|² / 2`, so the Euclidean bound squared and halved.
    /// * Dot product: `q·x = q·c + q·(x - c) <= q·c + |q|·r` (Cauchy–Schwarz),
    ///   and the score is `-q·x`.
    ///
    /// Each bound is loosened so rounding can only cost speed, never a
    /// neighbour: by [`BOUND_SLACK`] relative to the magnitudes involved, and
    /// for Cosine and dot product also by an absolute term for the rounding in
    /// a `d`-term dot product, whose error scales with `|q|·|x|` rather than
    /// with the (possibly near-zero) result. A NaN bound becomes `-inf` —
    /// "cannot prune" — rather than poisoning the ordering.
    fn cell_bound(&self, query: &[f32], query_norm: f32, cell: usize) -> (f32, f32) {
        let centroid = self.ivf.centroid(cell, self.dim);
        let radius = self.ivf.radii.get(cell).copied().unwrap_or(f32::INFINITY);
        let dot_rounding = 4.0 * self.dim as f32 * f32::EPSILON;

        let (bound, closeness) = match self.distance {
            Distance::Euclidean | Distance::Cosine => {
                let to_centroid = simd::l2_sq(query, centroid).sqrt();
                let slack = BOUND_SLACK * (to_centroid + radius);
                // `f32::max` drops a NaN operand, so an undefined gap is 0,
                // which never prunes.
                let gap = (to_centroid - radius - slack).max(0.0);
                let bound = if self.distance == Distance::Euclidean {
                    gap
                } else {
                    // Unit vectors are only unit to within rounding, and the
                    // score is `1 - dot`, which can land a few ulps below the
                    // exact identity's value. Subtracting the dot product's
                    // rounding allowance keeps a near-duplicate's cell from
                    // being pruned by a hair.
                    0.5 * gap * gap - dot_rounding
                };
                (bound, to_centroid)
            }
            Distance::DotProduct => {
                let along = simd::dot(query, centroid);
                let centroid_norm = simd::norm_sq(centroid).sqrt();
                let spread = query_norm * radius;
                // `|x| <= |c| + r` for every row in the cell, so this covers
                // the rounding of both `q·c` and the row's own `q·x`.
                let scale = query_norm * (centroid_norm + radius);
                let slack = (BOUND_SLACK + dot_rounding) * scale;
                (-(along + spread + slack), -along)
            }
        };

        let bound = if bound.is_nan() {
            f32::NEG_INFINITY
        } else {
            bound
        };
        let closeness = if closeness.is_nan() {
            f32::INFINITY
        } else {
            closeness
        };
        (bound, closeness)
    }

    /// Approximate search: probe `effective_nprobe(k)` cells, optionally
    /// prefilter with PQ, rescore exactly.
    fn search_approximate(&self, query: &[f32], k: usize) -> SearchOutcome {
        let probe = self.probe_top(query, k, self.effective_nprobe(k));

        // Recall guard: the probed cells did not hold `k` live rows, so the
        // answer would come back short. Fall back to the exact search.
        if probe.top.len() < k {
            let mut outcome = self.search_pruned(query, k);
            outcome.candidates_scored += probe.scored;
            return outcome;
        }

        SearchOutcome {
            neighbors: self.collect(&probe.top),
            candidates_scored: probe.scored,
            rescored: probe.rescored,
            cells_probed: probe.cells,
            exact: false,
        }
    }

    /// Every cell, scored the way the approximate search ranks them — lower
    /// is more promising; sort with [`cmp_cell`].
    ///
    /// Euclidean and Cosine rank by exact `l2_sq` to the centroid — the same
    /// function rows were filed with. Ranking by the cached-norm expansion
    /// `|q|² + |c|² - 2q·c` instead disagreed with the assignment after
    /// rounding, so a query could miss the very cell its twin was filed in.
    /// Dot product ranks by the upper bound `q·c + |q|·r`: ranking by centroid
    /// L2 skipped large-norm rows, which are exactly the ones an inner-product
    /// search is looking for.
    fn cell_scores(&self, query: &[f32]) -> Vec<(f32, u32)> {
        let query_norm = simd::norm_sq(query).sqrt();
        (0..self.ivf.nlist())
            .map(|cell| {
                let score = match self.distance {
                    Distance::Euclidean | Distance::Cosine => {
                        simd::l2_sq(query, self.ivf.centroid(cell, self.dim))
                    }
                    Distance::DotProduct => self.cell_bound(query, query_norm, cell).0,
                };
                let score = if score.is_nan() { f32::INFINITY } else { score };
                (score, cell as u32)
            })
            .collect()
    }

    /// `true` when approximate searches prefilter with PQ. Its tables
    /// approximate squared L2, so it is never used for dot product.
    fn uses_pq(&self) -> bool {
        !self.pq.is_empty()
            && !self.options.exact_rescore_only
            && self.distance != Distance::DotProduct
    }

    /// The top `k` from the `nprobe` best-ranked cells.
    fn probe_top(&self, query: &[f32], k: usize, nprobe: usize) -> Probe {
        let dim = self.dim;
        let nlist = self.ivf.nlist();

        // 1. Coarse: pick the `nprobe` most promising cells.
        let mut cell_scores = self.cell_scores(query);
        let nprobe = nprobe.clamp(1, nlist.max(1));
        if nprobe < cell_scores.len() {
            cell_scores.select_nth_unstable_by(nprobe - 1, cmp_cell);
            cell_scores.truncate(nprobe);
        }
        cell_scores.sort_unstable_by(cmp_cell);

        // 2. PQ prefilter, where it applies.
        let use_pq = self.uses_pq();
        let budget = k.saturating_mul(RESCORE_FACTOR).max(MIN_RESCORE_CANDIDATES);

        let mut candidates: Vec<(f32, u32)> = Vec::new();

        for &(_, cell) in &cell_scores {
            let cell = cell as usize;
            let list = &self.ivf.lists[cell];

            if !use_pq {
                for &row in list {
                    let row = row as usize;
                    if self.ids.id_of(row).is_some() {
                        candidates.push((self.score_row(row, query), row as u32));
                    }
                }
                continue;
            }

            let centroid = self.ivf.centroid(cell, dim);
            let residual: Vec<f32> = query
                .iter()
                .zip(centroid.iter())
                .map(|(q, c)| q - c)
                .collect();
            let tables = self.pq.distance_tables(&residual);

            for &row in list {
                let row = row as usize;
                if self.ids.id_of(row).is_none() {
                    continue;
                }

                let start = row * self.pq.m;
                match self.codes.get(start..start + self.pq.m) {
                    Some(code) => {
                        let approx = self.pq.score_code(code, &tables);
                        let approx = if approx.is_nan() {
                            f32::INFINITY
                        } else {
                            approx
                        };
                        candidates.push((approx, row as u32));
                    }
                    // A row without a code falls back to an exact score.
                    None => candidates.push((self.score_row(row, query), row as u32)),
                }
            }
        }

        let scored = candidates.len();

        // 3. Trim to the budget (PQ scores only — exact scores need no trim),
        //    then rescore exactly into the top-k.
        if use_pq && candidates.len() > budget {
            candidates.select_nth_unstable_by(budget - 1, |a, b| cmp_score(a.0, b.0));
            candidates.truncate(budget);
        }

        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k.saturating_add(1));
        let mut rescored = 0usize;
        for &(score, row) in &candidates {
            let score = if use_pq {
                rescored += 1;
                self.score_row(row as usize, query)
            } else {
                score
            };
            insert_top(&mut top, score, row, k);
        }

        Probe {
            top,
            scored,
            rescored,
            cells: cell_scores.len(),
        }
    }

    /// The smallest `nprobe` whose recall@[`CALIBRATION_K`] reaches
    /// [`TARGET_RECALL`] on the corpus itself.
    ///
    /// Leave-one-out: up to [`CALIBRATION_QUERIES`] stored rows, spread over
    /// the arena, each serve as a query with the row itself removed from both
    /// answers, so each stands in for a fresh vector from the same source.
    /// Their true neighbours come from the exact pruned search.
    ///
    /// Without PQ every row in a probed cell is scored exactly, so a true
    /// neighbour is found exactly when its cell ranks inside the probe. One
    /// ranking per query therefore gives the recall of *every* `nprobe` at
    /// once. With PQ the prefilter can lose neighbours too, so values from that
    /// starting point upwards are measured by running the search; if probing
    /// every cell still misses the target, PQ is dropped — a prefilter that
    /// cannot find the neighbours is not worth keeping.
    ///
    /// The result is at least [`MIN_CALIBRATED_NPROBE`], and is `nlist` —
    /// exact search — when an exactly-scored probe would cover
    /// [`EXACT_FALLBACK_FRACTION`] of the rows anyway. Costs about
    /// [`CALIBRATION_QUERIES`] exact searches per training run.
    fn calibrate_nprobe(&mut self) -> usize {
        let nlist = self.ivf.nlist();
        let live = self.ids.len();
        if nlist <= 1 || live < 2 {
            return nlist.max(1);
        }

        let k = CALIBRATION_K.min(live - 1);
        let step = live.div_ceil(CALIBRATION_QUERIES).max(1);
        let queries: Vec<u32> = self
            .ids
            .entries()
            .map(|(row, _)| row)
            .step_by(step)
            .collect();

        let mut cell_of = vec![u32::MAX; self.store.rows()];
        for (cell, list) in self.ivf.lists.iter().enumerate() {
            for &row in list {
                if let Some(slot) = cell_of.get_mut(row as usize) {
                    *slot = cell as u32;
                }
            }
        }

        // `found_at[rank]`: true neighbours filed in the cell a query ranks
        // `rank`-th. A neighbour in no list is never found and adds nothing.
        let mut found_at = vec![0usize; nlist];
        let mut rank_of = vec![0usize; nlist];
        let mut cases: Vec<(Vec<f32>, u32, Vec<u32>)> = Vec::with_capacity(queries.len());
        // Per query, its cells best first — to count the rows a probe scans.
        let mut orders: Vec<Vec<u32>> = Vec::with_capacity(queries.len());

        for &row in &queries {
            let query = self.store.row(row as usize).to_vec();
            let (top, _, _) = self.pruned_top(&query, k + 1);
            let truth: Vec<u32> = top
                .iter()
                .map(|&(_, r)| r)
                .filter(|&r| r != row)
                .take(k)
                .collect();

            let mut ranking = self.cell_scores(&query);
            ranking.sort_unstable_by(cmp_cell);
            for (rank, &(_, cell)) in ranking.iter().enumerate() {
                rank_of[cell as usize] = rank;
            }
            for &neighbour in &truth {
                if let Some(&rank) = cell_of
                    .get(neighbour as usize)
                    .and_then(|&cell| rank_of.get(cell as usize))
                {
                    found_at[rank] += 1;
                }
            }

            cases.push((query, row, truth));
            orders.push(ranking.into_iter().map(|(_, cell)| cell).collect());
        }

        // Rows an exactly-scored probe of `nprobe` cells scans, as a fraction
        // of all filed rows, averaged over the calibration queries.
        let filed: usize = self.ivf.lists.iter().map(Vec::len).sum();
        let scanned_fraction = |nprobe: usize| -> f32 {
            let scanned: usize = orders
                .iter()
                .map(|order| {
                    order
                        .iter()
                        .take(nprobe)
                        .map(|&cell| self.ivf.lists[cell as usize].len())
                        .sum::<usize>()
                })
                .sum();
            scanned as f32 / (orders.len().max(1) * filed.max(1)) as f32
        };

        let total: usize = cases.iter().map(|(_, _, truth)| truth.len()).sum();
        if total == 0 {
            return 1;
        }
        let needed = (TARGET_RECALL * total as f32).ceil() as usize;

        let mut nprobe = nlist;
        let mut covered = 0usize;
        for (rank, &count) in found_at.iter().enumerate() {
            covered += count;
            if covered >= needed {
                nprobe = rank + 1;
                break;
            }
        }
        let nprobe = nprobe.max(MIN_CALIBRATED_NPROBE).min(nlist);
        let flat = if scanned_fraction(nprobe) >= EXACT_FALLBACK_FRACTION {
            nlist
        } else {
            nprobe
        };

        if !self.uses_pq() {
            return flat;
        }

        let recall_at = |engine: &Engine, nprobe: usize| -> usize {
            cases
                .iter()
                .map(|(query, row, truth)| {
                    engine
                        .probe_top(query, k + 1, nprobe)
                        .top
                        .iter()
                        .map(|&(_, r)| r)
                        .filter(|r| r != row)
                        .take(k)
                        .filter(|r| truth.contains(r))
                        .count()
                })
                .sum()
        };

        let mut candidate = nprobe;
        loop {
            if recall_at(self, candidate) >= needed {
                return candidate;
            }
            if candidate >= nlist {
                break;
            }
            candidate = (candidate * 2).min(nlist);
        }

        self.pq = Pq::default();
        self.codes.clear();
        flat
    }

    /// Exact score of one row against a prepared query.
    ///
    /// Cosine is pre-normalised on both sides at insert/query time, so its
    /// score is `1 - dot` and does not need the norm lookup at all.
    ///
    /// # NaN handling
    ///
    /// Every arm funnels non-finite results to `+inf` rather than dropping the
    /// row. Inputs are validated finite, but finite inputs can still overflow:
    /// a query of `f32::MAX` against a stored `f32::MAX` produces
    /// `inf - inf = NaN` in a dot product. Mapping to `+inf` keeps the row in
    /// the running and sorts it last, which is the right answer: an
    /// unrepresentably large distance really is the worst case.
    #[inline]
    fn score_row(&self, row: usize, query: &[f32]) -> f32 {
        let vector = self.store.row(row);
        let score = match self.distance {
            Distance::Euclidean => simd::l2(vector, query),
            Distance::DotProduct => -simd::dot(vector, query),
            Distance::Cosine => 1.0 - simd::dot(vector, query),
        };

        if score.is_nan() { f32::INFINITY } else { score }
    }

    fn collect(&self, top: &[(f32, u32)]) -> Vec<Neighbor> {
        top.iter()
            .filter_map(|&(score, row)| {
                let id = self.ids.id_of(row as usize)?;
                let distance = match self.distance {
                    // DotProduct was negated internally; report the real value.
                    Distance::DotProduct => -score,
                    _ => score,
                };
                Some(Neighbor {
                    id: id.to_string(),
                    distance,
                })
            })
            .collect()
    }

    /// Cells to probe for a `k`-neighbour query.
    ///
    /// `nprobe` holds for `k` up to [`CALIBRATION_K`]. A larger `k` reaches
    /// further from the query and into more cells, so the probe grows with
    /// `sqrt(k / CALIBRATION_K)`: the extra neighbours sit on a slightly wider
    /// shell, not in proportionally more cells. The engine tests check the
    /// recall target still holds at `k = 50`.
    fn effective_nprobe(&self, k: usize) -> usize {
        let base = if self.nprobe > 0 {
            self.nprobe
        } else {
            self.default_nprobe()
        };
        let nlist = self.ivf.nlist().max(1);
        if k <= CALIBRATION_K {
            return base.clamp(1, nlist);
        }
        let scale = (k as f64 / CALIBRATION_K as f64).sqrt();
        ((base as f64 * scale).ceil() as usize).clamp(1, nlist)
    }

    fn default_nprobe(&self) -> usize {
        let nlist = self.ivf.nlist().max(1);
        ((nlist as f32 * DEFAULT_NPROBE_RATIO).round() as usize)
            .max(MIN_DEFAULT_NPROBE)
            .clamp(1, nlist)
    }

    // -- index construction ----------------------------------------------

    /// Fit to the index dimension, normalise for Cosine, store, and return
    /// the new row index. Does **not** touch the ids map — callers do that so
    /// a failed insert leaves no partial state.
    fn push_prepared(&mut self, vector: &[f32]) -> usize {
        let mut prepared = self.fit_dimension(vector);
        if self.distance == Distance::Cosine {
            simd::normalize_in_place(&mut prepared);
        }
        self.store.push(&prepared)
    }

    fn fit_dimension(&self, vector: &[f32]) -> Vec<f32> {
        if vector.len() == self.dim {
            return vector.to_vec();
        }

        let mut prepared = vec![0.0f32; self.dim];
        let copy = vector.len().min(self.dim);
        prepared[..copy].copy_from_slice(&vector[..copy]);
        prepared
    }

    fn maybe_train(&mut self) {
        if self.ids.len() >= self.options.ivf_threshold {
            self.train();
        }
    }

    fn maybe_retrain(&mut self) {
        let rows = self.store.rows();
        let growth = rows.saturating_sub(self.rows_at_last_train);
        let threshold = ((rows as f32) * self.options.retrain_growth_ratio) as usize;

        if growth >= threshold.max(MIN_RETRAIN_GROWTH) {
            self.train();
        }
    }

    fn maybe_compact(&mut self) {
        let slots = self.store.rows();
        if slots == 0 {
            return;
        }

        let dead = self.ids.dead();
        let ratio = dead as f32 / slots as f32;
        if ratio >= self.options.compact_ratio {
            self.compact();
        }
    }

    /// Train the coarse quantiser and rebuild all lists.
    ///
    /// Trains on a sample of `SAMPLES_PER_CENTROID` points per cell rather
    /// than on every row, so the cost is `O(nlist² · d)` per iteration rather
    /// than `O(N · nlist · d)`; filing every row afterwards is one more
    /// `O(N · nlist · d)` pass. Runs only on build and on drift-triggered
    /// retrains — never on a single insert.
    fn train(&mut self) {
        let rows = self.store.rows();
        let live = self.ids.len();
        let dim = self.dim;

        let drop_index = |engine: &mut Engine| {
            engine.ivf = Ivf::default();
            engine.pq = Pq::default();
            engine.codes.clear();
            engine.nprobe = 0;
            engine.rows_at_last_train = rows;
        };

        if live == 0 || dim == 0 || live < self.options.ivf_threshold {
            // Below the threshold a flat SIMD scan is the better index, so
            // drop any existing structure and let `search` take the flat path.
            drop_index(self);
            return;
        }

        let wanted = self
            .options
            .nlist
            .unwrap_or_else(|| default_nlist(live))
            .clamp(1, live);
        let sample_cap = wanted
            .saturating_mul(SAMPLES_PER_CENTROID)
            .clamp(wanted, MAX_TRAINING_SAMPLES.max(wanted));
        let sample = self.training_sample(sample_cap);
        let sample_rows = sample.len() / dim;

        if sample_rows == 0 {
            drop_index(self);
            return;
        }

        let config = KMeansConfig {
            k: wanted.min(sample_rows),
            max_iter: KMEANS_ITERATIONS,
            tolerance: 0.001,
            seed: self.options.seed,
        };

        let result = match kmeans::train(&sample, sample_rows, dim, config) {
            Ok(result) if !result.centroids.is_empty() => result,
            _ => {
                // A degenerate corpus (e.g. every vector identical) should
                // degrade to a flat scan, not fail the build.
                drop_index(self);
                return;
            }
        };

        let nlist = result.centroids.len();
        let mut centroids = Vec::with_capacity(nlist * dim);
        for centroid in &result.centroids {
            centroids.extend_from_slice(centroid);
        }

        self.ivf = Ivf {
            centroids,
            lists: vec![Vec::new(); nlist],
            radii: vec![0.0; nlist],
        };

        self.assign_all_rows();
        self.reorder_by_cell();
        self.rows_at_last_train = self.store.rows();

        self.train_pq();
        // After PQ: with a prefilter the same recall can need more cells.
        self.nprobe = match self.options.nprobe {
            Some(nprobe) => nprobe.clamp(1, nlist),
            None => self.calibrate_nprobe(),
        };
    }

    /// Lay the arena out cell by cell, so every inverted list is a contiguous
    /// run of rows.
    ///
    /// Without this a cell's rows are scattered through the arena in insertion
    /// order, and scanning one costs a cache miss per row — on unstructured
    /// data, where the exact search cannot prune much, that made the indexed
    /// scan slower than the flat one it is meant to beat. Tombstoned rows are
    /// dropped on the way, since they are in no list. Rows added after this
    /// are appended at the end until the next training run.
    fn reorder_by_cell(&mut self) {
        let slots = self.store.rows();
        let mut mapping: Vec<Option<u32>> = vec![None; slots];
        let mut keep: Vec<u32> = Vec::with_capacity(self.ids.len());

        for list in &mut self.ivf.lists {
            for row in list.iter_mut() {
                let new_row = keep.len() as u32;
                if let Some(slot) = mapping.get_mut(*row as usize) {
                    *slot = Some(new_row);
                    keep.push(*row);
                    *row = new_row;
                }
            }
        }

        self.ids.compact(&mapping);
        self.store.retain_rows(&keep);
        // Codes are keyed by row; `train_pq` re-encodes after this.
        self.codes.clear();
    }

    /// File every live row in its nearest cell, and recompute the radii.
    fn assign_all_rows(&mut self) {
        let dim = self.dim;
        let nlist = self.ivf.centroids.len() / dim.max(1);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nlist];
        let mut radii = vec![0.0f32; nlist];

        if nlist > 0 {
            // `nearest_centroid` iterates `lists.len()` cells.
            self.ivf.lists = vec![Vec::new(); nlist];

            for row in 0..self.store.rows() {
                if self.ids.id_of(row).is_none() {
                    continue;
                }
                let (cell, dist_sq) = self.ivf.nearest_centroid(self.store.row(row), dim);
                lists[cell].push(row as u32);
                let dist = dist_sq.sqrt();
                radii[cell] = if dist.is_finite() {
                    radii[cell].max(dist)
                } else {
                    f32::INFINITY
                };
            }
        }

        self.ivf.lists = lists;
        self.ivf.radii = radii;
    }

    /// Recompute every cell's radius from its current members. `O(N · d)`.
    fn recompute_radii(&mut self) {
        let dim = self.dim;
        let mut radii = vec![0.0f32; self.ivf.nlist()];

        for (cell, list) in self.ivf.lists.iter().enumerate() {
            let centroid = self.ivf.centroid(cell, dim);
            let mut radius = 0.0f32;
            for &row in list {
                let dist = simd::l2_sq(self.store.row(row as usize), centroid).sqrt();
                radius = if dist.is_finite() {
                    radius.max(dist)
                } else {
                    f32::INFINITY
                };
            }
            radii[cell] = radius;
        }

        self.ivf.radii = radii;
    }

    /// Incremental placement: nearest cell + one code. This is the whole point
    /// of the rewrite — the old code retrained here.
    fn insert_into_ivf(&mut self, row: usize) {
        let dim = self.dim;
        let (cell, dist_sq) = self.ivf.nearest_centroid(self.store.row(row), dim);

        if let Some(list) = self.ivf.lists.get_mut(cell) {
            list.push(row as u32);
        }
        self.ivf.widen(cell, dist_sq);

        self.encode_row(row, cell);
    }

    fn encode_row(&mut self, row: usize, cell: usize) {
        let m = self.pq.m;
        if m == 0 {
            return;
        }

        let dim = self.dim;
        let centroid = self.ivf.centroid(cell, dim);
        if centroid.is_empty() {
            return;
        }

        let vector = self.store.row(row);
        let residual: Vec<f32> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| v - c)
            .collect();
        let code = self.pq.encode(&residual);

        // Grow the code buffer if this row is beyond its current end.
        let needed = (row + 1) * m;
        if self.codes.len() < needed {
            self.codes.resize(needed, 0);
        }
        self.codes[row * m..row * m + m].copy_from_slice(&code);
    }

    fn train_pq(&mut self) {
        let rows = self.store.rows();
        let dim = self.dim;

        // PQ is only ever a prefilter for approximate L2-family searches; for
        // dot product it would never be consulted, so do not pay to train it.
        if rows < PQ_MIN_VECTORS
            || dim == 0
            || self.options.exact_rescore_only
            || self.distance == Distance::DotProduct
        {
            self.pq = Pq::default();
            self.codes.clear();
            return;
        }

        let m = self.options.pq_m.unwrap_or_else(|| default_pq_m(dim));
        if m == 0 || !dim.is_multiple_of(m) {
            self.pq = Pq::default();
            self.codes.clear();
            return;
        }

        let subvector_dim = dim / m;
        let ksub = self.options.pq_ksub.unwrap_or(64).clamp(2, 256).min(rows);

        // Train on residuals, per subquantiser.
        let cap = ksub
            .saturating_mul(PQ_SAMPLES_PER_CODEWORD)
            .min(MAX_TRAINING_SAMPLES);
        let sample = self.residual_sample(cap);
        let sample_rows = sample.len() / dim;

        if sample_rows < ksub {
            self.pq = Pq::default();
            self.codes.clear();
            return;
        }

        let mut codebooks = Vec::with_capacity(m * ksub * subvector_dim);

        for sub in 0..m {
            let start = sub * subvector_dim;

            // Gather this subquantiser's slice of every sampled residual into
            // one contiguous buffer for k-means.
            let mut subvectors = Vec::with_capacity(sample_rows * subvector_dim);
            for index in 0..sample_rows {
                let base = index * dim + start;
                subvectors.extend_from_slice(&sample[base..base + subvector_dim]);
            }

            let config = KMeansConfig {
                k: ksub,
                max_iter: KMEANS_ITERATIONS,
                tolerance: 0.001,
                seed: self.options.seed ^ (sub as u64).wrapping_mul(0x9E37_79B9),
            };

            match kmeans::train(&subvectors, sample_rows, subvector_dim, config) {
                Ok(result) if result.centroids.len() == ksub => {
                    for centroid in &result.centroids {
                        codebooks.extend_from_slice(centroid);
                    }
                }
                _ => {
                    self.pq = Pq::default();
                    self.codes.clear();
                    return;
                }
            }
        }

        self.pq = Pq {
            m,
            ksub,
            subvector_dim,
            codebooks,
        };

        self.encode_all();
    }

    fn encode_all(&mut self) {
        let rows = self.store.rows();
        let m = self.pq.m;

        if m == 0 {
            self.codes.clear();
            return;
        }

        let dim = self.dim;
        let mut codes = vec![0u8; rows * m];

        // Encode against the cell each row is actually filed in, not its
        // current nearest centroid: search computes the residual against the
        // cell being probed, so the two must agree.
        for (cell, list) in self.ivf.lists.iter().enumerate() {
            let centroid = self.ivf.centroid(cell, dim);
            for &row in list {
                let row = row as usize;
                if self.ids.id_of(row).is_none() || row >= rows {
                    continue;
                }
                let residual: Vec<f32> = self
                    .store
                    .row(row)
                    .iter()
                    .zip(centroid.iter())
                    .map(|(v, c)| v - c)
                    .collect();
                let code = self.pq.encode(&residual);
                codes[row * m..row * m + m].copy_from_slice(&code);
            }
        }

        self.codes = codes;
    }

    /// Flatten a sample of live rows for k-means training.
    fn training_sample(&self, max_samples: usize) -> Vec<f32> {
        let dim = self.dim;
        let rows = self.store.rows();
        let live = self.ids.len();

        if live == 0 || dim == 0 || max_samples == 0 {
            return Vec::new();
        }

        // Stride sampling keeps the sample spread over insertion order, which
        // for embedding corpora correlates with content.
        let step = live.div_ceil(max_samples).max(1);

        let mut out = Vec::with_capacity(live.min(max_samples) * dim);
        let mut taken = 0usize;

        for row in 0..rows {
            if self.ids.id_of(row).is_none() {
                continue;
            }
            let pick = taken.is_multiple_of(step);
            taken += 1;
            if !pick {
                continue;
            }

            out.extend_from_slice(self.store.row(row));
            if out.len() / dim >= max_samples {
                break;
            }
        }

        out
    }

    /// Residuals against each row's cell, for PQ training.
    fn residual_sample(&self, max_samples: usize) -> Vec<f32> {
        let dim = self.dim;
        let live = self.ids.len();

        if live == 0 || dim == 0 || self.ivf.is_empty() || max_samples == 0 {
            return Vec::new();
        }

        let step = live.div_ceil(max_samples).max(1);
        let mut out = Vec::with_capacity(live.min(max_samples) * dim);
        let mut taken = 0usize;

        'cells: for (cell, list) in self.ivf.lists.iter().enumerate() {
            let centroid = self.ivf.centroid(cell, dim);
            for &row in list {
                let row = row as usize;
                if self.ids.id_of(row).is_none() {
                    continue;
                }
                let pick = taken.is_multiple_of(step);
                taken += 1;
                if !pick {
                    continue;
                }

                for (v, c) in self.store.row(row).iter().zip(centroid.iter()) {
                    out.push(v - c);
                }
                if out.len() / dim >= max_samples {
                    break 'cells;
                }
            }
        }

        out
    }

    // -- snapshot ---------------------------------------------------------

    /// Build a snapshot. Only live rows are written, renumbered densely, so a
    /// snapshot never carries tombstones.
    pub fn to_snapshot(&self) -> Snapshot {
        let dim = self.dim;
        let count = self.ids.len();

        let mut data = Vec::with_capacity(count * dim);
        let mut cache = Vec::with_capacity(count);
        let mut ids = Vec::with_capacity(count);

        for (row, id) in self.ids.entries() {
            let row = row as usize;
            data.extend_from_slice(self.store.row(row));
            cache.push(self.store.cache_attr(row));
            ids.push(id.to_string());
        }

        // Remap lists onto the dense rows written above.
        let mut remap: Vec<Option<u32>> = vec![None; self.store.rows()];
        for (dense, (row, _)) in self.ids.entries().enumerate() {
            remap[row as usize] = Some(dense as u32);
        }

        let lists: Vec<Vec<u32>> = self
            .ivf
            .lists
            .iter()
            .map(|list| {
                list.iter()
                    .filter_map(|&row| remap.get(row as usize).copied().flatten())
                    .collect()
            })
            .collect();

        let mut codes = Vec::new();
        if !self.pq.is_empty() {
            codes.reserve(count * self.pq.m);
            for (row, _) in self.ids.entries() {
                let start = row as usize * self.pq.m;
                if let Some(slice) = self.codes.get(start..start + self.pq.m) {
                    codes.extend_from_slice(slice);
                } else {
                    codes.resize(codes.len() + self.pq.m, 0);
                }
            }
        }

        let pq = if self.pq.is_empty() {
            None
        } else {
            Some(PqSnapshot {
                m: self.pq.m as u32,
                ksub: self.pq.ksub as u32,
                subvector_dim: self.pq.subvector_dim as u32,
                codebooks: self.pq.codebooks.clone(),
            })
        };

        Snapshot {
            distance: self.distance,
            dim: dim as u32,
            count: count as u32,
            data,
            cache,
            ids,
            nlist: self.ivf.nlist() as u32,
            nprobe: self.nprobe as u32,
            centroids: self.ivf.centroids.clone(),
            lists,
            pq,
            codes,
        }
    }

    /// Serialise. `compressed` trades CPU for size — see
    /// [`crate::engine::codec::write`].
    pub fn serialize(&self, compressed: bool) -> EngineResult<Vec<u8>> {
        codec::write(&self.to_snapshot(), compressed)
    }
}

/// Reject vectors no search could make sense of. An empty vector used to fix
/// the index dimension at 0 and detach every later id from its row; a NaN
/// used to score as a perfect match against everything.
fn validate_vector(id: &str, vector: &[f32]) -> EngineResult<()> {
    if vector.is_empty() {
        return Err(EngineError::new(format!("vector for id {id:?} is empty")));
    }
    if let Some(position) = vector.iter().position(|value| !value.is_finite()) {
        return Err(EngineError::new(format!(
            "vector for id {id:?} has a non-finite value (NaN or infinity) at index {position}"
        )));
    }
    Ok(())
}

#[inline]
fn is_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// `true` when every row in `0..count` appears in exactly one list.
fn lists_cover_rows_once(lists: &[Vec<u32>], count: usize) -> bool {
    let mut seen = vec![false; count];
    for list in lists {
        for &row in list {
            match seen.get_mut(row as usize) {
                Some(slot) if !*slot => *slot = true,
                _ => return false,
            }
        }
    }
    seen.iter().all(|&covered| covered)
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// One search hit. Mirrors the public `Neighbor` without pulling wasm types
/// into the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    pub id: String,
    pub distance: f32,
}

/// Result of a search, with enough diagnostics to tell whether the index did
/// what the caller expects.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchOutcome {
    pub neighbors: Vec<Neighbor>,
    /// Rows the approximate pass looked at.
    pub candidates_scored: usize,
    /// Candidates that got an exact rescore.
    pub rescored: usize,
    /// Cells probed.
    pub cells_probed: usize,
    /// `true` when the result is exact (brute force or no PQ codes).
    pub exact: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// What [`Engine::probe_top`] found.
struct Probe {
    top: Vec<(f32, u32)>,
    /// Rows looked at, by PQ or exactly.
    scored: usize,
    /// Candidates rescored exactly after the PQ prefilter.
    rescored: usize,
    /// Cells probed.
    cells: usize,
}

/// Order `(score, cell)` pairs best first. The cell index breaks ties, so the
/// search and the calibration rank cells identically.
#[inline]
fn cmp_cell(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    cmp_score(a.0, b.0).then(a.1.cmp(&b.1))
}

/// NaN-safe ascending comparison. `total_cmp` gives a total order over all f32
/// bit patterns, so a NaN sorts to the end instead of scrambling the partition
/// — the old `partial_cmp(..).unwrap_or(Ordering::Equal)` made NaNs compare
/// "equal" to everything, which silently corrupts a sort.
#[inline]
fn cmp_score(a: f32, b: f32) -> Ordering {
    a.total_cmp(&b)
}

/// Insert `(score, row)` into an ascending top-`k` buffer.
#[inline]
fn insert_top(top: &mut Vec<(f32, u32)>, score: f32, row: u32, k: usize) {
    if top.len() < k {
        // Sorted insert; `k` is small so the memmove is cheaper than a heap.
        let position =
            top.partition_point(|&(existing, _)| cmp_score(existing, score) == Ordering::Less);
        top.insert(position, (score, row));
        return;
    }

    // Buffer is full: only a score better than the worst can enter.
    let last = match top.last() {
        Some(&(value, _)) => value,
        None => return,
    };

    if cmp_score(score, last) == Ordering::Less {
        let position =
            top.partition_point(|&(existing, _)| cmp_score(existing, score) == Ordering::Less);
        top.insert(position, (score, row));
        top.pop();
    }
}

/// Default cell count: `round(sqrt(n))`.
///
/// The exact search pays `nlist · d` per query to bound every cell, then scans
/// the cells it cannot prune; `sqrt(n)` cells of `sqrt(n)` rows balances the
/// two, and keeps the k-means build at `O(n · d)` per iteration.
fn default_nlist(rows: usize) -> usize {
    if rows == 0 {
        return 0;
    }

    ((rows as f64).sqrt().round() as usize).clamp(1, rows)
}

/// Roughly one PQ byte per 8 dimensions, adjusted so `dim % m == 0` and
/// `m <= 16` (beyond that the code grows faster than the precision improves).
fn default_pq_m(dim: usize) -> usize {
    if dim == 0 {
        return 0;
    }

    let mut m = (dim / 8).clamp(1, 16);
    while m > 1 && !dim.is_multiple_of(m) {
        m -= 1;
    }
    m
}

// ---------------------------------------------------------------------------
// Free-function API
// ---------------------------------------------------------------------------

/// Build an index. Errors on a data/ids length mismatch, a duplicate id, or an
/// empty or non-finite vector.
pub fn index(data: &[Embedding], ids: &[String]) -> EngineResult<Engine> {
    Engine::build(data, ids, IndexOptions::default())
}

/// See [`Engine::search`].
pub fn search(engine: &Engine, query: &[f32], k: usize) -> SearchOutcome {
    engine.search(query, k)
}

/// See [`Engine::search_exact`].
pub fn search_exact(engine: &Engine, query: &[f32], k: usize) -> SearchOutcome {
    engine.search_exact(query, k)
}

/// See [`Engine::add`].
pub fn add(engine: &mut Engine, id: String, vector: &[f32]) -> EngineResult<()> {
    engine.add(id, vector)
}

/// See [`Engine::remove`].
pub fn remove(engine: &mut Engine, ids: &[String]) -> EngineResult<()> {
    engine.remove(ids)
}

/// See [`Engine::len`].
pub fn size(engine: &Engine) -> usize {
    engine.len()
}

/// See [`Engine::clear`].
pub fn clear(engine: &mut Engine) {
    engine.clear()
}

/// Serialise, compressed.
pub fn dump(engine: &Engine) -> EngineResult<Vec<u8>> {
    engine.serialize(true)
}

/// Serialise, uncompressed.
pub fn dump_compressed(engine: &Engine, compressed: bool) -> EngineResult<Vec<u8>> {
    engine.serialize(compressed)
}

/// Restore an engine from a snapshot, in either format.
pub fn load(bytes: &[u8]) -> EngineResult<Engine> {
    load_with_options(bytes, IndexOptions::default())
}

/// Restore, overriding the runtime options. Build-time options stored in the
/// snapshot (nlist, centroids, codebooks) are preserved; only the search-time
/// knobs from `options` take effect.
pub fn load_with_options(bytes: &[u8], options: IndexOptions) -> EngineResult<Engine> {
    let snapshot = codec::read(bytes)?;
    Engine::from_snapshot(snapshot, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, used to derive a distinct state per row.
    ///
    /// The obvious `seed | 1` idiom is a trap here: it maps any adjacent pair
    /// `2n`/`2n+1` onto the *same* state, so a corpus built as
    /// `pseudo(0x1000 + i)` contains byte-identical vector pairs — and then a
    /// test that asks "is the exact vector its own nearest neighbour?" is
    /// comparing two indistinguishable candidates, where either answer is
    /// defensible. Mixing first gives every row a distinct state.
    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pseudo(seed: u64, len: usize) -> Vec<f32> {
        let mut state = splitmix(seed) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn corpus(count: usize, dim: usize) -> Vec<Embedding> {
        (0..count).map(|i| pseudo(0x1000 + i as u64, dim)).collect()
    }

    fn ids_of(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("id-{i}")).collect()
    }

    fn options() -> IndexOptions {
        IndexOptions {
            ivf_threshold: 256,
            ..Default::default()
        }
    }

    fn exact_options() -> IndexOptions {
        IndexOptions {
            approximate: false,
            ..options()
        }
    }

    #[test]
    fn build_and_search() {
        let data = corpus(500, 16);
        let ids = ids_of(500);
        let engine = Engine::build(&data, &ids, options()).expect("build");

        assert_eq!(engine.len(), 500);
        assert!(engine.is_indexed());

        // Query with an exact member: it must come back first, at distance 0.
        let outcome = engine.search(&data[42], 5);
        assert_eq!(outcome.neighbors.len(), 5);
        assert_eq!(outcome.neighbors[0].id, "id-42");
        assert!(outcome.neighbors[0].distance < 1e-5);

        // Distances ascend.
        for pair in outcome.neighbors.windows(2) {
            assert!(pair[0].distance <= pair[1].distance + 1e-6);
        }
    }

    #[test]
    fn incremental_add_matches_exact() {
        // The whole point of the rewrite: adding must not retrain, but it must
        // still produce the right answer.
        let data = corpus(300, 32);
        let ids = ids_of(300);
        let mut engine = Engine::build(&data, &ids, options()).expect("build");

        let extra = corpus(50, 32);
        for (i, vector) in extra.iter().enumerate() {
            engine.add(format!("extra-{i}"), vector).expect("add");
        }
        assert_eq!(engine.len(), 350);

        let query = &extra[7];
        let outcome = engine.search(query, 3);
        assert_eq!(outcome.neighbors[0].id, "extra-7");

        // And the freshly added vector is findable via a full exact scan too.
        let exact = engine.search_exact(query, 3);
        assert_eq!(exact.neighbors[0].id, "extra-7");
    }

    #[test]
    fn add_is_cheap() {
        // Regression guard for the O(N²) ingest bug: 200 inserts into a
        // 2000-vector index must not take time proportional to a full retrain.
        // We cannot assert wall-clock in a unit test, so assert that no
        // retraining happened — `rows_at_last_train` only moves on a train.
        let data = corpus(2000, 32);
        let engine = Engine::build(&data, &ids_of(2000), options()).expect("build");
        let before = engine.rows_at_last_train;

        let mut engine = engine;
        for i in 0..200 {
            engine
                .add(format!("new-{i}"), &pseudo(0x9999 + i, 32))
                .expect("add");
        }

        // 200 / 2000 = 10% growth, under the 25% retrain ratio.
        assert_eq!(
            engine.rows_at_last_train, before,
            "inserting 200 of 2000 vectors should not trigger a retrain"
        );
    }

    #[test]
    fn duplicate_and_dimension_errors() {
        let mut engine =
            Engine::build(&corpus(10, 8), &ids_of(10), IndexOptions::default()).expect("build");

        assert!(engine.add("id-0".to_string(), &[0.0; 8]).is_err());
        assert!(engine.add("fresh".to_string(), &[0.0; 5]).is_err());
        assert!(engine.add_many(&[("a".to_string(), vec![0.0; 8])]).is_ok());
        assert!(engine.add_many(&[("a".to_string(), vec![0.0; 8])]).is_err());
    }

    #[test]
    fn remove_tombstones_then_compacts() {
        let mut engine = Engine::build(&corpus(100, 16), &ids_of(100), options()).expect("build");

        // Below the compaction ratio the tombstone is retained, not swept.
        engine
            .remove(&(0..20).map(|i| format!("id-{i}")).collect::<Vec<_>>())
            .expect("remove");
        assert_eq!(engine.len(), 80);
        assert_eq!(engine.dead_rows(), 20, "20% dead is under the 30% ratio");

        // Crossing the ratio triggers compaction, which zeroes the tombstones.
        engine
            .remove(&(20..45).map(|i| format!("id-{i}")).collect::<Vec<_>>())
            .expect("remove");
        assert_eq!(engine.len(), 55);
        assert_eq!(engine.dead_rows(), 0, "45% dead should have compacted");

        // Removed ids are gone; survivors still answer.
        assert!(!engine.contains("id-0"));
        assert!(engine.contains("id-99"));

        // Post-compaction search must return live rows only. Asserting a
        // *specific* neighbour would be asserting a property of the test
        // fixture's random data rather than of compaction, so check the
        // invariant that actually matters: every hit is still live.
        let outcome = engine.search(&pseudo(0x1000 + 99, 16), 5);
        assert!(!outcome.neighbors.is_empty());
        for neighbor in &outcome.neighbors {
            assert!(
                engine.contains(&neighbor.id),
                "{} was returned but is not live",
                neighbor.id
            );
        }
    }

    #[test]
    fn remove_unknown_id_errors() {
        let mut engine =
            Engine::build(&corpus(10, 8), &ids_of(10), IndexOptions::default()).expect("build");
        assert!(engine.remove(&["nope".to_string()]).is_err());
    }

    #[test]
    fn removing_everything_resets() {
        let mut engine = Engine::build(&corpus(10, 8), &ids_of(10), options()).expect("build");
        let all: Vec<String> = ids_of(10);
        engine.remove(&all).expect("remove all");

        assert!(engine.is_empty());
        assert_eq!(engine.dim(), 0);
        assert_eq!(engine.search(&[0.0; 8], 5).neighbors.len(), 0);
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        // No vectors at all.
        let engine = Engine::build(&[], &[], options()).expect("build");
        assert!(engine.is_empty());
        assert_eq!(engine.search(&[1.0, 2.0], 3).neighbors.len(), 0);

        // Zero-dimensional vectors are rejected, not stored: an empty vector
        // used to fix the dimension at 0 and detach later ids from their rows.
        assert!(Engine::build(&[vec![]], &["empty".to_string()], options()).is_err());

        // Mismatched lengths.
        assert!(Engine::build(&corpus(3, 4), &ids_of(2), options()).is_err());

        // k = 0.
        let engine = Engine::build(&corpus(10, 4), &ids_of(10), options()).expect("build");
        assert_eq!(engine.search(&[0.0; 4], 0).neighbors.len(), 0);

        // Digging deeper than the corpus holds.
        let outcome = engine.search(&[0.0; 4], 100);
        assert_eq!(outcome.neighbors.len(), 10);
    }

    #[test]
    fn zero_and_extreme_queries_do_not_produce_nan() {
        let mut data = corpus(50, 8);
        data.push(vec![f32::MAX; 8]);
        data.push(vec![f32::MIN; 8]);
        data.push(vec![0.0; 8]);

        let mut ids = ids_of(50);
        ids.push("max".into());
        ids.push("min".into());
        ids.push("zero".into());

        let engine = Engine::build(&data, &ids, options()).expect("build");

        for query in [
            vec![0.0f32; 8],
            vec![1.0f32; 8],
            vec![-1.0f32; 8],
            vec![f32::MAX; 8],
            vec![f32::MIN; 8],
        ] {
            let outcome = engine.search(&query, 5);
            assert!(!outcome.neighbors.is_empty());
            for neighbor in &outcome.neighbors {
                assert!(
                    !neighbor.distance.is_nan(),
                    "NaN distance for query {query:?} on {}",
                    neighbor.id
                );
            }
        }
    }

    #[test]
    fn cosine_is_scale_invariant() {
        let data = vec![
            vec![1.0f32, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let ids = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let engine = Engine::build(
            &data,
            &ids,
            IndexOptions {
                distance: Distance::Cosine,
                ..options()
            },
        )
        .expect("build");

        // Scaled and unscaled parallel vectors are the same direction.
        let outcome = engine.search(&[100.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(outcome.neighbors[0].id, "x");
        assert!(outcome.neighbors[0].distance.abs() < 1e-5);

        // A zero query has no direction: it must return *something*, not NaN.
        let outcome = engine.search(&[0.0; 4], 2);
        assert_eq!(outcome.neighbors.len(), 2);
        assert!(outcome.neighbors.iter().all(|n| !n.distance.is_nan()));
    }

    #[test]
    fn dot_product_ranks_by_magnitude() {
        let data = vec![vec![1.0f32, 0.0], vec![5.0, 0.0], vec![0.0, 9.0]];
        let ids = vec!["small".to_string(), "big".to_string(), "orth".to_string()];
        let engine = Engine::build(
            &data,
            &ids,
            IndexOptions {
                distance: Distance::DotProduct,
                ..options()
            },
        )
        .expect("build");

        let outcome = engine.search(&[1.0, 0.0], 3);
        assert_eq!(outcome.neighbors[0].id, "big");
        // The reported distance is the true dot product, not the negated score.
        assert!((outcome.neighbors[0].distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn snapshot_round_trip_preserves_answers() {
        let data = corpus(800, 24);
        let ids = ids_of(800);
        let engine = Engine::build(&data, &ids, options()).expect("build");

        for compressed in [false, true] {
            let bytes = engine.serialize(compressed).expect("serialize");
            let restored = load(&bytes).expect("load");

            assert_eq!(restored.len(), engine.len());
            assert_eq!(restored.dim(), engine.dim());
            assert_eq!(restored.distance(), engine.distance());
            assert_eq!(restored.is_indexed(), engine.is_indexed());

            // Every query must produce byte-identical results.
            for query_index in [0usize, 7, 400, 799] {
                let original = engine.search(&data[query_index], 10);
                let loaded = restored.search(&data[query_index], 10);
                assert_eq!(original.neighbors, loaded.neighbors, "query {query_index}");
            }
        }
    }

    #[test]
    fn snapshot_after_mutation_round_trips() {
        let data = corpus(600, 16);
        let mut engine = Engine::build(&data, &ids_of(600), options()).expect("build");

        for i in 0..50 {
            engine
                .add(format!("late-{i}"), &pseudo(0x2000 + i, 16))
                .expect("add");
        }
        engine
            .remove(&(0..30).map(|i| format!("id-{i}")).collect::<Vec<_>>())
            .expect("remove");

        let bytes = engine.serialize(true).expect("serialize");
        let restored = load(&bytes).expect("load");

        assert_eq!(restored.len(), engine.len());
        for query_index in [0usize, 100, 550] {
            let original = engine.search(&data[query_index], 5);
            let loaded = restored.search(&data[query_index], 5);
            assert_eq!(original.neighbors, loaded.neighbors);
        }
    }

    #[test]
    fn rejects_corrupt_snapshot_without_panicking() {
        let engine = Engine::build(&corpus(100, 8), &ids_of(100), options()).expect("build");
        let bytes = engine.serialize(true).expect("serialize");

        for cut in [0usize, 1, 8, 15, bytes.len() / 3, bytes.len() - 1] {
            assert!(load(&bytes[..cut]).is_err());
        }

        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0x5A;
        assert!(load(&flipped).is_err());
    }

    #[test]
    fn recall_is_high_against_ground_truth() {
        // The index is approximate, but with PQ off the only approximation is
        // which cells get probed. On clustered data that should be near-exact.
        let data = corpus(4000, 32);
        let ids = ids_of(4000);
        let engine = Engine::build(
            &data,
            &ids,
            IndexOptions {
                ivf_threshold: 256,
                exact_rescore_only: true,
                nprobe: Some(16),
                ..Default::default()
            },
        )
        .expect("build");

        let mut hits = 0usize;
        let queries = 50usize;

        for query_index in (0..4000).step_by(4000 / queries) {
            let truth = engine.search_exact(&data[query_index], 10);
            let approx = engine.search(&data[query_index], 10);

            let truth_ids: std::collections::HashSet<&str> =
                truth.neighbors.iter().map(|n| n.id.as_str()).collect();
            hits += approx
                .neighbors
                .iter()
                .filter(|n| truth_ids.contains(n.id.as_str()))
                .count();
        }

        let recall = hits as f32 / (queries * 10) as f32;
        assert!(recall > 0.8, "recall@{queries} was {recall}");
    }

    #[test]
    fn pq_index_recall_is_acceptable() {
        // With PQ on, recall drops — that is the trade. It must stay usable.
        let data = corpus(6000, 32);
        let ids = ids_of(6000);
        let engine = Engine::build(
            &data,
            &ids,
            IndexOptions {
                ivf_threshold: 256,
                nlist: Some(46),
                nprobe: Some(24),
                exact_rescore_only: false,
                approximate: true,
                ..Default::default()
            },
        )
        .expect("build");

        assert!(engine.has_pq(), "PQ should be active at 6000 vectors");

        let mut hits = 0usize;
        let queries = 50usize;

        for query_index in (0..6000).step_by(6000 / queries) {
            let truth = engine.search_exact(&data[query_index], 10);
            let approx = engine.search(&data[query_index], 10);

            let truth_ids: std::collections::HashSet<&str> =
                truth.neighbors.iter().map(|n| n.id.as_str()).collect();
            hits += approx
                .neighbors
                .iter()
                .filter(|n| truth_ids.contains(n.id.as_str()))
                .count();
        }

        let recall = hits as f32 / (queries * 10) as f32;
        assert!(recall > 0.5, "PQ recall@{queries} was {recall}");
    }

    #[test]
    fn clear_resets_everything() {
        let mut engine = Engine::build(&corpus(500, 16), &ids_of(500), options()).expect("build");
        assert!(engine.is_indexed());

        engine.clear();
        assert!(engine.is_empty());
        assert!(!engine.is_indexed());
        assert_eq!(engine.dim(), 0);
        assert_eq!(engine.search(&[0.0; 16], 5).neighbors.len(), 0);

        // Still usable after clearing.
        engine
            .add("fresh".to_string(), &[1.0; 16])
            .expect("add after clear");
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn query_dimension_is_coerced_not_rejected() {
        // Matches the old `prepare_vector` behaviour so existing callers that
        // send a slightly-wrong dimension keep working.
        let engine = Engine::build(&corpus(50, 8), &ids_of(50), options()).expect("build");

        assert_eq!(engine.search(&[1.0; 4], 3).neighbors.len(), 3);
        assert_eq!(engine.search(&[1.0; 64], 3).neighbors.len(), 3);
        assert_eq!(engine.search(&[], 3).neighbors.len(), 3);
    }

    #[test]
    fn below_threshold_stays_exact() {
        let data = corpus(100, 16);
        let engine = Engine::build(
            &data,
            &ids_of(100),
            IndexOptions {
                ivf_threshold: 4096,
                ..Default::default()
            },
        )
        .expect("build");

        assert!(!engine.is_indexed());

        // With no index every result is exact.
        let outcome = engine.search(&data[3], 5);
        assert!(outcome.exact);
        assert_eq!(outcome.neighbors[0].id, "id-3");
    }

    #[test]
    fn default_nlist_is_sane() {
        assert_eq!(default_nlist(0), 0);
        assert!(default_nlist(100) >= 1);
        assert!(default_nlist(10_000) <= 10_000);
        // Large corpora should not produce absurd cell counts.
        assert!(default_nlist(1_000_000) <= 4000);
    }

    #[test]
    fn default_pq_m_divides_the_dimension() {
        for dim in [8usize, 12, 16, 32, 100, 384, 768, 1024, 1536] {
            let m = default_pq_m(dim);
            assert!(m > 0, "dim {dim} produced m = 0");
            assert_eq!(dim % m, 0, "dim {dim} not divisible by m {m}");
            assert!(m <= 16);
        }
        assert_eq!(default_pq_m(0), 0);
    }

    #[test]
    fn cmp_score_orders_nan_last() {
        assert_eq!(cmp_score(1.0, 2.0), Ordering::Less);
        assert_eq!(cmp_score(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(cmp_score(1.0, f32::NAN), Ordering::Less);
    }

    /// Clustered fixture: `centres` blobs with noise of `spread` around each.
    fn clustered(
        count: usize,
        dim: usize,
        centres: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<Embedding> {
        let middles: Vec<Embedding> = (0..centres)
            .map(|c| pseudo(seed ^ ((c as u64) << 20), dim))
            .collect();
        (0..count)
            .map(|i| {
                let noise = pseudo(seed.wrapping_add(0x51_0000 + i as u64), dim);
                middles[i % centres]
                    .iter()
                    .zip(noise)
                    .map(|(m, n)| m + spread * n)
                    .collect()
            })
            .collect()
    }

    /// The exact-mode search must return exactly what the brute-force scan
    /// returns — same ids, same order up to ties — for every metric, on data
    /// with and without cluster structure.
    fn assert_matches_exact(engine: &Engine, queries: &[Embedding], k: usize, label: &str) {
        for (q, query) in queries.iter().enumerate() {
            let fast = engine.search(query, k);
            let slow = engine.search_exact(query, k);
            assert!(fast.exact, "{label}: exact mode must report exact");
            assert_eq!(fast.neighbors.len(), slow.neighbors.len(), "{label} q{q}");
            for (a, b) in fast.neighbors.iter().zip(&slow.neighbors) {
                // Equal scores may come back in either order; equal *ids* at
                // unequal scores would be a real miss.
                assert!(
                    (a.distance - b.distance).abs() <= 1e-5 * (1.0 + b.distance.abs()),
                    "{label} q{q}: pruned search returned {} at {}, exact had {} at {}",
                    a.id,
                    a.distance,
                    b.id,
                    b.distance
                );
            }
        }
    }

    #[test]
    fn pruned_search_is_exact_for_every_metric() {
        for distance in [Distance::Euclidean, Distance::Cosine, Distance::DotProduct] {
            for (label, data) in [
                ("clustered", clustered(3000, 24, 20, 0.15, 0xC1)),
                ("uniform", corpus(3000, 24)),
            ] {
                let ids = ids_of(data.len());
                let engine = Engine::build(
                    &data,
                    &ids,
                    IndexOptions {
                        distance,
                        ..exact_options()
                    },
                )
                .expect("build");
                assert!(engine.is_indexed());

                let mut queries: Vec<Embedding> = (0..20).map(|i| pseudo(0xAB00 + i, 24)).collect();
                queries.extend(data.iter().step_by(311).cloned());
                assert_matches_exact(&engine, &queries, 10, &format!("{distance:?}/{label}"));
            }
        }
    }

    #[test]
    fn pruned_search_stays_exact_through_mutation_and_reload() {
        let data = clustered(2500, 16, 12, 0.2, 0x77);
        let mut engine = Engine::build(&data, &ids_of(2500), exact_options()).expect("build");

        // Rows added after training land outside the contiguous cell runs and
        // widen their cell's radius; deletions leave stale radii behind. Both
        // must keep the bound valid.
        for i in 0..400 {
            engine
                .add(format!("late-{i}"), &pseudo(0xD00D + i, 16))
                .expect("add");
        }
        engine
            .remove(&(0..500).map(|i| format!("id-{i}")).collect::<Vec<_>>())
            .expect("remove");

        let queries: Vec<Embedding> = (0..25).map(|i| pseudo(0xE00 + i, 16)).collect();
        assert_matches_exact(&engine, &queries, 7, "mutated");

        let restored =
            load_with_options(&engine.serialize(true).expect("serialize"), exact_options())
                .expect("load");
        assert_matches_exact(&restored, &queries, 7, "restored");
        for query in &queries {
            assert_eq!(
                engine.search(query, 7).neighbors,
                restored.search(query, 7).neighbors
            );
        }
    }

    #[test]
    fn pruned_search_skips_cells_on_clustered_data() {
        let data = clustered(8000, 32, 64, 0.05, 0x99);
        let engine = Engine::build(&data, &ids_of(8000), exact_options()).expect("build");

        let outcome = engine.search(&data[123], 10);
        assert!(outcome.exact);
        assert!(
            outcome.candidates_scored < 8000 / 4,
            "scored {} of 8000 rows; the bound should prune most cells",
            outcome.candidates_scored
        );
    }

    #[test]
    fn training_lays_cells_out_contiguously() {
        let engine = Engine::build(&corpus(1000, 8), &ids_of(1000), options()).expect("build");
        let mut next = 0u32;
        for list in &engine.ivf.lists {
            for &row in list {
                assert_eq!(row, next, "cells should be consecutive runs of rows");
                next += 1;
            }
        }
        assert_eq!(next as usize, engine.len());
        // And the ids moved with their vectors.
        let data = corpus(1000, 8);
        for i in [0usize, 17, 999] {
            assert_eq!(
                engine.search(&data[i], 1).neighbors[0].id,
                format!("id-{i}")
            );
        }
    }

    #[test]
    fn rejects_empty_and_non_finite_vectors() {
        let mut engine = Engine::new(IndexOptions::default());

        // REGRESSION: `add("a", [])` then `add("b", v)` used to replace the
        // store under "a", so "b" was filed on row 1 while its vector sat on
        // row 0, and the next snapshot could not be read back.
        assert!(engine.add("a".into(), &[]).is_err());
        assert!(engine.add("nan".into(), &[0.0, f32::NAN]).is_err());
        assert!(engine.add("inf".into(), &[f32::INFINITY, 0.0]).is_err());
        assert_eq!(engine.len(), 0);
        assert_eq!(engine.dim(), 0);

        engine.add("b".into(), &[1.0, 2.0]).expect("add");
        assert_eq!(engine.search(&[1.0, 2.0], 1).neighbors[0].id, "b");
        let restored = load(&engine.serialize(true).expect("serialize")).expect("load");
        assert_eq!(restored.len(), 1);

        // A non-finite query has no neighbours rather than garbage ones.
        assert!(engine.search(&[f32::NAN, 0.0], 1).neighbors.is_empty());
        assert!(
            engine
                .search_exact(&[0.0, f32::INFINITY], 1)
                .neighbors
                .is_empty()
        );
    }

    #[test]
    fn remove_and_add_many_are_atomic() {
        let mut engine = Engine::build(&corpus(10, 4), &ids_of(10), options()).expect("build");

        // One unknown id: nothing is removed.
        assert!(
            engine
                .remove(&["id-1".to_string(), "missing".to_string()])
                .is_err()
        );
        assert_eq!(engine.len(), 10);
        assert!(engine.contains("id-1"));

        // Repeating an id inside one call removes it once, without an error.
        engine
            .remove(&["id-2".to_string(), "id-2".to_string()])
            .expect("remove");
        assert_eq!(engine.len(), 9);

        // A bad item anywhere in a batch: nothing is added.
        let batch = vec![
            ("new-1".to_string(), vec![0.0; 4]),
            ("new-2".to_string(), vec![0.0; 3]),
        ];
        assert!(engine.add_many(&batch).is_err());
        assert!(!engine.contains("new-1"));
        let batch = vec![
            ("dup".to_string(), vec![0.0; 4]),
            ("dup".to_string(), vec![1.0; 4]),
        ];
        assert!(engine.add_many(&batch).is_err());
        assert!(!engine.contains("dup"));
        assert_eq!(engine.len(), 9);
    }

    #[test]
    fn huge_k_is_clamped() {
        for threshold in [256usize, usize::MAX] {
            let engine = Engine::build(
                &corpus(300, 8),
                &ids_of(300),
                IndexOptions {
                    ivf_threshold: threshold,
                    ..Default::default()
                },
            )
            .expect("build");
            for k in [301usize, usize::MAX >> 2, usize::MAX] {
                assert_eq!(engine.search(&[0.0; 8], k).neighbors.len(), 300);
                assert_eq!(engine.search_exact(&[0.0; 8], k).neighbors.len(), 300);
            }
        }
    }

    #[test]
    fn duplicate_ids_in_a_snapshot_keep_ids_on_their_vectors() {
        // REGRESSION: `IdMap::from_ids` skipped a repeated id without leaving
        // a slot, so every later id moved onto the previous row's vector.
        let snapshot = Snapshot {
            distance: Distance::Euclidean,
            dim: 2,
            count: 3,
            data: vec![0.0, 0.0, 5.0, 5.0, 9.0, 9.0],
            cache: vec![0.0; 3],
            ids: vec!["a".into(), "a".into(), "b".into()],
            ..Snapshot::default()
        };
        let engine = Engine::from_snapshot(snapshot, IndexOptions::default()).expect("load");

        assert_eq!(engine.len(), 2);
        assert_eq!(engine.dead_rows(), 0, "the duplicate is swept on load");
        assert_eq!(engine.search(&[9.0, 9.0], 1).neighbors[0].id, "b");
        assert_eq!(engine.search(&[0.0, 0.0], 1).neighbors[0].id, "a");
    }

    #[test]
    fn inconsistent_snapshots_are_errors_not_panics() {
        let base = || {
            let engine = Engine::build(&corpus(400, 8), &ids_of(400), options()).expect("build");
            engine.to_snapshot()
        };

        let mut wrong_lists = base();
        wrong_lists.lists.push(Vec::new());
        assert!(Engine::from_snapshot(wrong_lists, IndexOptions::default()).is_err());

        let mut zero_dim = base();
        zero_dim.dim = 0;
        zero_dim.data.clear();
        assert!(Engine::from_snapshot(zero_dim, IndexOptions::default()).is_err());

        let mut bad_pq = base();
        bad_pq.pq = Some(PqSnapshot {
            m: 1,
            ksub: u32::MAX,
            subvector_dim: 0,
            codebooks: Vec::new(),
        });
        bad_pq.codes = vec![0; 400];
        assert!(Engine::from_snapshot(bad_pq, IndexOptions::default()).is_err());

        let mut nan = base();
        nan.data[3] = f32::NAN;
        assert!(Engine::from_snapshot(nan, IndexOptions::default()).is_err());

        // A list that misses a row is repaired by re-filing, not trusted.
        let mut missing_row = base();
        let victim = missing_row
            .lists
            .iter()
            .position(|l| !l.is_empty())
            .expect("a non-empty list");
        let dropped = missing_row.lists[victim].remove(0);
        let engine = Engine::from_snapshot(missing_row, IndexOptions::default()).expect("load");
        let data = corpus(400, 8);
        let id = engine
            .ids
            .id_of(dropped as usize)
            .expect("live")
            .to_string();
        let row = id
            .trim_start_matches("id-")
            .parse::<usize>()
            .expect("index");
        assert_eq!(engine.search(&data[row], 1).neighbors[0].id, id);
    }

    #[test]
    fn load_honours_the_callers_nprobe() {
        let engine = Engine::build(&corpus(2000, 8), &ids_of(2000), options()).expect("build");
        let bytes = engine.serialize(false).expect("serialize");

        let tuned = load_with_options(
            &bytes,
            IndexOptions {
                nprobe: Some(3),
                ..Default::default()
            },
        )
        .expect("load");
        assert_eq!(tuned.nprobe(), 3);

        let default = load(&bytes).expect("load");
        assert_eq!(default.nprobe(), engine.nprobe());
    }

    #[test]
    fn approximate_dot_product_finds_large_norm_winners() {
        // Ranking cells by distance to the centroid, as the approximate path
        // used to for every metric, never probes the cell holding a far-away,
        // large-norm vector — which is exactly the dot-product winner.
        let mut data = corpus(3000, 8);
        let mut ids = ids_of(3000);
        data.push(vec![50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        ids.push("giant".into());

        let engine = Engine::build(
            &data,
            &ids,
            IndexOptions {
                distance: Distance::DotProduct,
                ivf_threshold: 256,
                approximate: true,
                nprobe: Some(1),
                ..Default::default()
            },
        )
        .expect("build");

        let outcome = engine.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(outcome.neighbors[0].id, "giant");
    }

    #[test]
    fn compaction_rebases_the_retrain_baseline() {
        let mut engine = Engine::build(&corpus(1000, 8), &ids_of(1000), options()).expect("build");
        assert_eq!(engine.rows_at_last_train, 1000);

        engine
            .remove(&(0..400).map(|i| format!("id-{i}")).collect::<Vec<_>>())
            .expect("remove");
        assert_eq!(engine.dead_rows(), 0, "40% dead compacts");
        assert_eq!(engine.rows_at_last_train, 600);
    }

    /// Recall@k of the default search against the brute-force answer.
    fn recall(engine: &Engine, queries: &[Embedding], k: usize) -> f32 {
        let mut hits = 0usize;
        let mut total = 0usize;
        for query in queries {
            let truth: HashSet<String> = engine
                .search_exact(query, k)
                .neighbors
                .into_iter()
                .map(|n| n.id)
                .collect();
            total += truth.len();
            hits += engine
                .search(query, k)
                .neighbors
                .iter()
                .filter(|n| truth.contains(&n.id))
                .count();
        }
        hits as f32 / total.max(1) as f32
    }

    /// `count` stored vectors and 64 fresh queries from the same source.
    fn split(mut all: Vec<Embedding>, count: usize) -> (Vec<Embedding>, Vec<Embedding>) {
        let queries = all.split_off(count);
        (all, queries)
    }

    #[test]
    fn default_search_meets_the_recall_target() {
        // Fresh queries, not stored rows: calibration estimates recall from
        // stored rows, and this checks the estimate carries over.
        let dim = 32;
        let n = 4_000;
        let cases = [
            ("clustered", split(clustered(n + 64, dim, 40, 0.15, 7), n)),
            ("uniform", split(corpus(n + 64, dim), n)),
        ];

        for (label, (data, queries)) in &cases {
            for distance in [Distance::Euclidean, Distance::Cosine, Distance::DotProduct] {
                let engine = Engine::build(
                    data,
                    &ids_of(n),
                    IndexOptions {
                        distance,
                        ..options()
                    },
                )
                .expect("build");
                assert!(engine.is_indexed());

                for k in [1, 10, 50] {
                    let measured = recall(&engine, queries, k);
                    assert!(
                        measured >= 0.9,
                        "{label} {distance:?} k={k}: recall {measured} with nprobe {} of {}",
                        engine.nprobe(),
                        engine.nlist()
                    );
                }

                if *label == "clustered" && distance != Distance::DotProduct {
                    // The point of approximate search: few cells on data with
                    // structure.
                    assert!(
                        engine.nprobe() * 4 <= engine.nlist(),
                        "{label} {distance:?}: nprobe {} of {}",
                        engine.nprobe(),
                        engine.nlist()
                    );
                    let outcome = engine.search(&queries[0], 10);
                    assert!(!outcome.exact);
                    assert!(outcome.candidates_scored < n / 2);
                }
            }
        }
    }

    #[test]
    fn unstructured_data_falls_back_to_exact() {
        // Uniform noise in 256-d: reaching the recall target would mean
        // probing most of the corpus, so calibration picks every cell.
        let data = corpus(4_000, 256);
        let engine = Engine::build(&data, &ids_of(4_000), options()).expect("build");
        assert_eq!(engine.nprobe(), engine.nlist());
        let query = pseudo(0xFEED, 256);
        let outcome = engine.search(&query, 10);
        assert!(outcome.exact);
        assert_eq!(outcome.neighbors, engine.search_exact(&query, 10).neighbors);
    }

    #[test]
    fn explicit_nprobe_is_kept_and_exact_mode_is_exact() {
        let data = clustered(2_000, 16, 20, 0.2, 3);
        let engine = Engine::build(
            &data,
            &ids_of(2_000),
            IndexOptions {
                nprobe: Some(3),
                ..options()
            },
        )
        .expect("build");
        assert_eq!(engine.nprobe(), 3);

        let mut exact = engine.clone();
        exact.set_approximate(false);
        assert_matches_exact(&exact, &data[..20], 10, "exact mode");
    }

    #[test]
    fn pq_that_misses_the_target_is_dropped() {
        // Uniform noise in 64-d is close to incompressible: 8 bytes of PQ
        // cannot rank it, and calibration must notice rather than ship a
        // prefilter that loses most neighbours.
        let (data, queries) = split(corpus(4_096 + 64, 64), 4_096);
        let engine = Engine::build(
            &data,
            &ids_of(4_096),
            IndexOptions {
                exact_rescore_only: false,
                ..options()
            },
        )
        .expect("build");

        let measured = recall(&engine, &queries, 10);
        assert!(
            measured >= 0.9,
            "recall {measured}, pq {}, nprobe {} of {}",
            engine.has_pq(),
            engine.nprobe(),
            engine.nlist()
        );
    }

    #[test]
    fn calibration_survives_a_round_trip() {
        let data = clustered(3_000, 16, 30, 0.2, 11);
        let engine = Engine::build(&data, &ids_of(3_000), options()).expect("build");
        let restored = load(&engine.serialize(true).expect("serialize")).expect("load");
        assert_eq!(restored.nprobe(), engine.nprobe());
        assert_eq!(
            restored.search(&data[5], 10).neighbors,
            engine.search(&data[5], 10).neighbors
        );
    }

    #[test]
    fn insert_top_keeps_the_best_k() {
        let mut top = Vec::new();
        for value in [5.0f32, 1.0, 9.0, 3.0, 7.0] {
            insert_top(&mut top, value, value as u32, 3);
        }
        let scores: Vec<f32> = top.iter().map(|(s, _)| *s).collect();
        assert_eq!(scores, vec![1.0, 3.0, 5.0]);
    }
}
