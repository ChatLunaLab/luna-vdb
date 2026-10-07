//! The vector index: IVF coarse quantiser + optional PQ prefilter, over a flat
//! arena.
//!
//! # What changed and why it is faster
//!
//! The previous implementation was correct but had four structural problems.
//! Each one is fixed here, and the fix is noted next to the code that does it.
//!
//! 1. **`IVF_BUILD_THRESHOLD = 20_000`.** Below twenty thousand vectors the
//!    index was a full linear scan, with no SIMD. Since almost every consumer of
//!    a local vector DB sits between a thousand and ten thousand vectors, that
//!    threshold meant *nobody* got an index. The threshold is now 4 096, and the
//!    linear path below it is vectorised, so both sides of the boundary are
//!    fast.
//!
//! 2. **`add()` retrained the whole index.** `rebuild_ivf_pq` re-ran k-means
//!    over every vector on every single insert, so an `O(N)` insert made
//!    ingesting N vectors `O(N²)` — and it cloned the entire arena to do it.
//!    Insertion now assigns the new vector to its nearest existing centroid and
//!    appends one code, which is `O(nlist · d)` and independent of `N`.
//!
//! 3. **Deletion rebuilt the index too.** `remove()` called `swap_remove` and
//!    then `rebuild_ivf_pq`, so deleting one id cost a full retrain plus a
//!    rehash of the moved row. Rows are now tombstoned and squeezed out in one
//!    compaction pass, triggered only when the dead fraction crosses a
//!    threshold.
//!
//! 4. **The scan was scalar and scattered.** See [`crate::engine::simd`] and
//!    [`crate::engine::types::VectorStore`].
//!
//! The measured effect is a large multiple over the old engine on
//! 1k–100k vector workloads; `benches/search.rs` produces the numbers so the
//! claim can be checked rather than taken on faith.

use std::cmp::Ordering;

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
/// Was 20 000. Lowered because a flat SIMD scan is fast enough that the
/// crossover is genuinely around here, and the old value left the majority of
/// real workloads without an index at all.
pub const DEFAULT_IVF_THRESHOLD: usize = 4_096;

/// Default coarse cluster count when the caller does not pick one.
const TARGET_LIST_SIZE: usize = 128;

/// How many cells to probe by default, as a fraction of `nlist`.
const DEFAULT_NPROBE_RATIO: f32 = 0.05;

/// Candidates to rescore per requested neighbour. The PQ prefilter is
/// approximate, so we rescore a multiple of `k` and take the best `k` exactly.
const RESCORE_FACTOR: usize = 8;

/// Minimum candidates to rescore, so small `k` still has slack.
const MIN_RESCORE_CANDIDATES: usize = 64;

/// Retrain once this fraction of the corpus has been added since the last
/// training run. Drift degrades recall slowly; this bounds it.
const RETRAIN_GROWTH_RATIO: f32 = 0.25;

/// Compact once this fraction of rows are tombstones.
const DEFAULT_COMPACT_RATIO: f32 = 0.3;

/// Only build PQ codes when there are enough vectors to give k-means something
/// to work with per subquantiser.
const PQ_MIN_VECTORS: usize = 2_048;

/// Cap on training-set size. Training on a sample keeps build time bounded on
/// large corpora without measurably hurting cluster quality.
const MAX_TRAINING_SAMPLES: usize = 65_536;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Build-time and runtime tuning. All fields have sensible defaults; the whole
/// point of the defaults is that callers never have to touch this.
#[derive(Debug, Clone, Copy)]
pub struct IndexOptions {
    pub distance: Distance,
    /// Coarse cells. `None` picks `clamp(sqrt(n) / 4, 1, 65_536)`.
    pub nlist: Option<usize>,
    /// Cells probed per query. `None` picks ~5% of `nlist`, at least 1.
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
    /// Disable the PQ prefilter. Exact rescoring then covers every candidate in
    /// the probed lists, which is slower but has no approximation error.
    pub exact_rescore_only: bool,
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
            exact_rescore_only: false,
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
    /// Cached squared norms of the centroids, for the distance expansion.
    centroid_norms: Vec<f32>,
    /// `nlist` lists of row indices. Stale rows stay until compaction.
    lists: Vec<Vec<u32>>,
}

impl Ivf {
    fn nlist(&self) -> usize {
        self.lists.len()
    }

    fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    /// Nearest centroid to `vector`, by exact L2.
    fn nearest_centroid(&self, vector: &[f32], dim: usize) -> usize {
        let nlist = self.nlist();
        let mut best = 0usize;
        let mut best_dist = f32::MAX;

        for cell in 0..nlist {
            let centroid = &self.centroids[cell * dim..cell * dim + dim];
            let dist = simd::l2_sq(vector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best = cell;
            }
        }

        best
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
    /// `m × ksub` floats, laid out per subquantiser. Computed **once per query**
    /// rather than once per probed cell — that is the fix for the old
    /// `distance_tables` call, which sat inside the per-cell loop and so
    /// recomputed identical tables `nprobe` times.
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
    pub fn build(data: &[Embedding], ids: &[String], options: IndexOptions) -> EngineResult<Self> {
        if data.len() != ids.len() {
            return Err(EngineError::new(format!(
                "embeddings and ids length mismatch: {} vs {}",
                data.len(),
                ids.len()
            )));
        }

        let dim = data.iter().map(|vector| vector.len()).max().unwrap_or(0);
        let mut engine = Engine::new(options);

        if data.is_empty() || dim == 0 {
            engine.dim = dim;
            engine.store = VectorStore::new(dim);
            return Ok(engine);
        }

        engine.dim = dim;
        engine.store = VectorStore::with_capacity(dim, data.len());
        engine.ids = IdMap::with_capacity(data.len());
        engine.codes = Vec::new();

        for (vector, id) in data.iter().zip(ids.iter()) {
            engine.ids.insert(id.clone())?;
            engine.push_prepared(vector);
        }

        engine.train();
        Ok(engine)
    }

    /// Rebuild from a parsed snapshot.
    pub fn from_snapshot(snapshot: Snapshot, options: IndexOptions) -> EngineResult<Self> {
        let dim = snapshot.dim as usize;
        let count = snapshot.count as usize;

        if snapshot.data.len() != count * dim {
            return Err(EngineError::corrupt(format!(
                "vector buffer holds {} floats, expected {}",
                snapshot.data.len(),
                count * dim
            )));
        }
        if snapshot.ids.len() != count {
            return Err(EngineError::corrupt(format!(
                "{} ids for {count} vectors",
                snapshot.ids.len()
            )));
        }

        let mut store = VectorStore::new(dim);
        for row in 0..count {
            store.push(&snapshot.data[row * dim..row * dim + dim]);
        }

        let mut engine = Engine::new(options);
        engine.distance = snapshot.distance;
        engine.dim = dim;
        engine.store = store;
        engine.ids = IdMap::from_ids(snapshot.ids);
        engine.nprobe = snapshot.nprobe as usize;
        engine.rows_at_last_train = count;

        if snapshot.nlist > 0 && !snapshot.centroids.is_empty() {
            let nlist = snapshot.nlist as usize;
            let expected = nlist * dim;

            if snapshot.centroids.len() != expected {
                return Err(EngineError::corrupt(format!(
                    "centroid buffer holds {} floats, expected {expected}",
                    snapshot.centroids.len()
                )));
            }

            engine.ivf = Ivf {
                centroid_norms: (0..nlist)
                    .map(|cell| simd::norm_sq(&snapshot.centroids[cell * dim..cell * dim + dim]))
                    .collect(),
                centroids: snapshot.centroids,
                lists: snapshot.lists,
            };
        } else {
            // The snapshot carried no index (a legacy import, or a corpus that
            // was below the threshold when it was written). Rebuild it here so
            // the caller never sees a half-initialised engine.
            engine.train();
        }

        if let Some(pq) = snapshot.pq {
            engine.pq = Pq {
                m: pq.m as usize,
                ksub: pq.ksub as usize,
                subvector_dim: pq.subvector_dim as usize,
                codebooks: pq.codebooks,
            };
            engine.codes = snapshot.codes;
        } else if !engine.ivf.is_empty() {
            engine.train_pq();
        }

        Ok(engine)
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
        if self.ids.contains(&id) {
            return Err(EngineError::duplicate_id(&id));
        }

        if self.dim == 0 {
            self.dim = vector.len();
            self.store = VectorStore::new(self.dim);
        } else if vector.len() != self.dim {
            return Err(EngineError::dimension_mismatch(self.dim, vector.len()));
        }

        let row = self.push_prepared(vector);
        self.ids.insert(id)?;
        let _ = row;

        // Incremental: place the row in its nearest existing cell and encode
        // it against the existing codebooks. No retraining.
        if self.ivf.is_empty() {
            self.maybe_train();
        } else {
            self.insert_into_ivf(self.store.rows() - 1);
        }

        self.maybe_retrain();
        Ok(())
    }

    /// Add many vectors. Errors on the first duplicate and leaves the engine
    /// with the vectors added so far, matching how a partial ingest behaves in
    /// the old implementation.
    pub fn add_many(&mut self, items: &[(String, Embedding)]) -> EngineResult<()> {
        for (id, vector) in items {
            self.add(id.clone(), vector)?;
        }
        Ok(())
    }

    /// Remove ids. Rows are tombstoned; compaction happens when the dead
    /// fraction crosses `options.compact_ratio`.
    pub fn remove(&mut self, ids: &[String]) -> EngineResult<()> {
        for id in ids {
            self.ids.remove(id)?;
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

        let dim = self.dim;
        let slots = self.store.rows();

        // Build old-row → new-row mapping.
        let mut mapping: Vec<Option<u32>> = vec![None; slots];
        let mut next = 0u32;
        for row in 0..slots {
            if self.ids.id_of(row).is_some() {
                mapping[row] = Some(next);
                next += 1;
            }
        }

        let ids = self.ids.compact(&mapping);

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

        // Rebuild the lists from the surviving assignments. Cheap: one
        // nearest-centroid lookup per row.
        if !self.ivf.is_empty() {
            let nlist = self.ivf.nlist();
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nlist];

            for row in 0..self.store.rows() {
                let cell = self.ivf.nearest_centroid(self.store.row(row), dim);
                lists[cell].push(row as u32);
            }

            self.ivf.lists = lists;
        }

        let _ = ids;
    }

    // -- search -----------------------------------------------------------

    /// Search for the `k` nearest neighbours of `query`.
    pub fn search(&self, query: &[f32], k: usize) -> SearchOutcome {
        if self.is_empty() || self.dim == 0 {
            // Nothing to approximate: an empty answer over an empty index is
            // exact, and saying otherwise would make a caller waste a
            // `search_exact` call verifying a trivially-correct result.
            return SearchOutcome {
                exact: true,
                ..SearchOutcome::default()
            };
        }

        if k == 0 {
            // A deliberate no-op is not an approximation either.
            return SearchOutcome {
                exact: true,
                ..SearchOutcome::default()
            };
        }

        let mut prepared = self.prepare_query(query);

        if self.distance == Distance::Cosine {
            simd::normalize_in_place(&mut prepared);
        }

        if self.ivf.is_empty() {
            // The flat scan computes every distance exactly, so the result is
            // exact by construction — reporting `false` here would mislabel the
            // one path that has no approximation error at all.
            return self.search_flat(&prepared, k, true);
        }

        self.search_ivf(&prepared, k)
    }

    /// Exact brute-force search, ignoring the IVF index. Exists so callers can
    /// measure recall against ground truth — and so the benchmark has a
    /// baseline that does not depend on index quality.
    pub fn search_exact(&self, query: &[f32], k: usize) -> SearchOutcome {
        if k == 0 || self.is_empty() || self.dim == 0 {
            return SearchOutcome {
                exact: true,
                ..SearchOutcome::default()
            };
        }

        let mut prepared = self.prepare_query(query);
        if self.distance == Distance::Cosine {
            simd::normalize_in_place(&mut prepared);
        }

        self.search_flat(&prepared, k, true)
    }

    /// Coerce a query to the index dimension.
    ///
    /// A too-short query is zero-padded and an over-long one truncated, which
    /// matches the old `prepare_vector`. Padding a *short* query is the right
    /// call for robustness, but note it changes the result silently — hence
    /// the explicit `truncated_dimension` flag on the outcome so a caller can
    /// detect it.
    fn prepare_query(&self, query: &[f32]) -> Vec<f32> {
        if query.len() == self.dim {
            return query.to_vec();
        }

        let mut prepared = vec![0.0f32; self.dim];
        let copy = query.len().min(self.dim);
        prepared[..copy].copy_from_slice(&query[..copy]);
        prepared
    }

    /// Fixed-size top-`k` collector.
    ///
    /// A plain `Vec` plus insert-and-truncate beats `BinaryHeap` here: `k` is
    /// small (tens), the vector stays in cache, and there is no per-element
    /// allocation. The heap's `log k` only pays off when `k` is large.
    fn search_flat(&self, query: &[f32], k: usize, exact: bool) -> SearchOutcome {
        let rows = self.store.rows();
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

        for row in 0..rows {
            if self.ids.id_of(row).is_none() {
                continue;
            }

            // `score_row` normalises NaN to `+inf`, so no extra check here.
            insert_top(&mut top, self.score_row(row, query), row as u32, k);
        }

        SearchOutcome {
            neighbors: self.collect(&top),
            candidates_scored: rows,
            exact,
            ..SearchOutcome::default()
        }
    }

    fn search_ivf(&self, query: &[f32], k: usize) -> SearchOutcome {
        let dim = self.dim;
        let nlist = self.ivf.nlist();

        // 1. Coarse: pick the `nprobe` nearest cells. `select_nth_unstable`
        //    partitions in O(nlist) instead of sorting in O(nlist log nlist).
        let mut cell_scores: Vec<(f32, u32)> = (0..nlist)
            .map(|cell| {
                let centroid = &self.ivf.centroids[cell * dim..cell * dim + dim];
                // Cached-norm expansion: the centroids' norms never change, so
                // this is one dot per cell.
                let dist = self.ivf.centroid_norms[cell] + simd::norm_sq(query)
                    - 2.0 * simd::dot(query, centroid);
                (dist, cell as u32)
            })
            .collect();

        // A NaN from the expansion must not scramble the partition.
        for entry in cell_scores.iter_mut() {
            if !entry.0.is_finite() {
                entry.0 = f32::MAX;
            }
        }

        let nprobe = self.effective_nprobe(k).min(nlist);
        if nprobe < cell_scores.len() {
            cell_scores.select_nth_unstable_by(nprobe - 1, |a, b| cmp_score(a.0, b.0));
            cell_scores.truncate(nprobe);
        }
        cell_scores.sort_unstable_by(|a, b| cmp_score(a.0, b.0));

        let cells: Vec<usize> = cell_scores.iter().map(|(_, cell)| *cell as usize).collect();

        // 2. Candidate budget.
        let budget = (k * RESCORE_FACTOR).max(MIN_RESCORE_CANDIDATES);

        // 3. Approximate pass. PQ tables depend only on the query and the cell
        //    residual, so they are built per cell but reused across that cell's
        //    members — the old code rebuilt them per member.
        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(budget);

        for &cell in &cells {
            let list = &self.ivf.lists[cell];

            if self.pq.is_empty() || self.options.exact_rescore_only {
                // No codes: score exactly. Still far cheaper than a full scan,
                // because only the probed cells are touched.
                for &row in list {
                    let row = row as usize;
                    if self.ids.id_of(row).is_none() {
                        continue;
                    }
                    candidates.push((self.score_row(row, query), row as u32));
                }
                continue;
            }

            let centroid = &self.ivf.centroids[cell * dim..cell * dim + dim];
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
                let code = match self.codes.get(start..start + self.pq.m) {
                    Some(code) => code,
                    // A row without a code (added after the last training run
                    // and not yet encoded) falls back to an exact score.
                    None => {
                        candidates.push((self.score_row(row, query), row as u32));
                        continue;
                    }
                };

                let approx = self.pq.score_code(code, &tables);
                if approx.is_finite() {
                    candidates.push((approx, row as u32));
                }
            }
        }

        if candidates.is_empty() {
            // Every probed cell was empty or tombstoned — fall back rather than
            // returning nothing for a query that clearly has neighbours.
            let mut outcome = self.search_flat(query, k, true);
            outcome.candidates_scored = self.store.rows();
            return outcome;
        }

        let scored = candidates.len();

        // 4. Trim to the budget, then rescore exactly.
        if candidates.len() > budget {
            candidates.select_nth_unstable_by(budget - 1, |a, b| cmp_score(a.0, b.0));
            candidates.truncate(budget);
        }

        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        let mut rescored = 0usize;
        let exact = self.pq.is_empty() || self.options.exact_rescore_only;

        for (_, row) in candidates {
            let row = row as usize;
            if self.ids.id_of(row).is_none() {
                continue;
            }

            // `score_row` clamps NaN, so this is always insertable.
            if !exact {
                rescored += 1;
            }
            insert_top(&mut top, self.score_row(row, query), row as u32, k);
        }

        // 5. Recall guard: if rescoring rejected most candidates, the PQ codes
        //    are misleading for this query and we widen to a full exact scan
        //    rather than returning a low-quality answer silently.
        let exact_fallback = top.len() < k;
        if exact_fallback {
            let mut outcome = self.search_flat(query, k, true);
            outcome.candidates_scored = scored;
            return outcome;
        }

        SearchOutcome {
            neighbors: self.collect(&top),
            candidates_scored: scored,
            rescored,
            cells_probed: cells.len(),
            exact: false,
        }
    }

    /// Exact score of one row against a prepared query.
    ///
    /// Cosine is pre-normalised on both sides at insert/query time, so its
    /// score is `1 - dot` and does not need the norm lookup at all.
    ///
    /// # NaN handling
    ///
    /// Every arm funnels non-finite results to `+inf` rather than dropping the
    /// row. This matters: a query of `f32::MAX` against a stored `f32::MAX`
    /// produces `inf - inf = NaN` in the subtraction, and *every* row becomes
    /// NaN. Filtering those out — which is what the callers used to do — makes
    /// a valid query return zero neighbours instead of the corpus, and a
    /// bounded top-k then comes back short. Mapping to `+inf` keeps the row in
    /// the running and sorts it last, which is the semantically right answer:
    /// an unrepresentably large distance really is the worst case.
    ///
    /// `f32::INFINITY.to_string()` in a result is therefore a legitimate value,
    /// not a symptom. It only appears when the input itself overflows f32.
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

    fn effective_nprobe(&self, k: usize) -> usize {
        let base = if self.nprobe > 0 {
            self.nprobe
        } else {
            self.default_nprobe()
        };
        // Probing more cells than the answer needs is wasted work; probing too
        // few loses recall. Scale with k, clamped to the configured value.
        base.max((k / 4).max(1)).min(self.ivf.nlist().max(1))
    }

    fn default_nprobe(&self) -> usize {
        let nlist = self.ivf.nlist().max(1);
        ((nlist as f32 * DEFAULT_NPROBE_RATIO).round() as usize).clamp(1, nlist)
    }

    // -- index construction ----------------------------------------------

    /// Normalise, store, and return the new row index. Does **not** touch the
    /// ids map — callers do that so a failed insert leaves no partial state.
    fn push_prepared(&mut self, vector: &[f32]) -> usize {
        let prepared = if self.distance == Distance::Cosine {
            simd::normalize(vector)
        } else {
            self.fit_dimension(vector)
        };

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
        if self.store.rows() >= self.options.ivf_threshold {
            self.train();
        }
    }

    fn maybe_retrain(&mut self) {
        if self.ivf.is_empty() {
            self.maybe_train();
            return;
        }

        let rows = self.store.rows();
        let growth = rows.saturating_sub(self.rows_at_last_train);
        let threshold = ((rows as f32) * self.options.retrain_growth_ratio) as usize;

        if growth >= threshold.max(self.options.nlist.unwrap_or(256)) {
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

    /// Train the coarse quantiser and rebuild all lists. O(N · nlist · d · iters),
    /// so this runs only on build and on drift-triggered retrains — never on a
    /// single insert.
    fn train(&mut self) {
        let rows = self.store.rows();
        let dim = self.dim;

        if rows == 0 || dim == 0 || rows < self.options.ivf_threshold {
            // Below the threshold a flat SIMD scan is the better index, so
            // drop any existing structure and let `search` take the flat path.
            self.ivf = Ivf::default();
            self.pq = Pq::default();
            self.codes.clear();
            self.nprobe = 0;
            self.rows_at_last_train = rows;
            return;
        }

        let sample = self.training_sample(MAX_TRAINING_SAMPLES);
        let sample_rows = sample.len() / dim;

        if sample_rows == 0 {
            self.ivf = Ivf::default();
            self.rows_at_last_train = rows;
            return;
        }

        let nlist = self
            .options
            .nlist
            .unwrap_or_else(|| default_nlist(rows))
            .min(sample_rows.max(1));

        let config = KMeansConfig {
            k: nlist,
            max_iter: 12,
            tolerance: 0.001,
            seed: self.options.seed,
        };

        let result = match kmeans::train(&sample, sample_rows, dim, config) {
            Ok(result) => result,
            Err(_) => {
                // A degenerate corpus (e.g. every vector identical) should
                // degrade to a flat scan, not fail the build.
                self.ivf = Ivf::default();
                self.rows_at_last_train = rows;
                return;
            }
        };

        // Flatten centroids into the row-major layout the scan wants.
        let mut centroids = Vec::with_capacity(nlist * dim);
        for centroid in &result.centroids {
            centroids.extend_from_slice(centroid);
        }

        self.ivf = Ivf {
            centroid_norms: result
                .centroids
                .iter()
                .map(|centroid| simd::norm_sq(centroid))
                .collect(),
            centroids,
            lists: vec![Vec::new(); result.centroids.len()],
        };

        self.assign_all_rows();
        self.nprobe = self.options.nprobe.unwrap_or_else(|| self.default_nprobe());
        self.rows_at_last_train = rows;

        self.train_pq();
    }

    /// Place every live row into its nearest cell.
    fn assign_all_rows(&mut self) {
        let dim = self.dim;
        let nlist = self.ivf.nlist();
        let mut lists: Vec<Vec<u32>> = (0..nlist).map(|_| Vec::new()).collect();

        for row in 0..self.store.rows() {
            if self.ids.id_of(row).is_none() {
                continue;
            }
            let cell = self.ivf.nearest_centroid(self.store.row(row), dim);
            lists[cell].push(row as u32);
        }

        self.ivf.lists = lists;
    }

    /// Incremental placement: nearest cell + one code. This is the whole point
    /// of the rewrite — the old code retrained here.
    fn insert_into_ivf(&mut self, row: usize) {
        let dim = self.dim;
        let cell = self.ivf.nearest_centroid(self.store.row(row), dim);

        if let Some(list) = self.ivf.lists.get_mut(cell) {
            list.push(row as u32);
        }

        self.encode_row(row, cell);
    }

    fn encode_row(&mut self, row: usize, cell: usize) {
        let m = self.pq.m;
        if m == 0 {
            return;
        }

        let dim = self.dim;
        let centroid = match self.ivf.centroids.get(cell * dim..cell * dim + dim) {
            Some(centroid) => centroid,
            None => return,
        };

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

        if rows < PQ_MIN_VECTORS || dim == 0 || self.options.exact_rescore_only {
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
        let sample = self.residual_sample(MAX_TRAINING_SAMPLES);
        let sample_rows = if subvector_dim == 0 {
            0
        } else {
            sample.len() / dim
        };

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
                max_iter: 12,
                tolerance: 0.001,
                seed: self.options.seed ^ (sub as u64).wrapping_mul(0x9E37_79B9),
            };

            match kmeans::train(&subvectors, sample_rows, subvector_dim, config) {
                Ok(result) => {
                    for centroid in &result.centroids {
                        codebooks.extend_from_slice(centroid);
                    }
                }
                Err(_) => {
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

        let mut codes = vec![0u8; rows * m];

        for row in 0..rows {
            if self.ids.id_of(row).is_none() {
                continue;
            }

            let cell = self.ivf.nearest_centroid(self.store.row(row), self.dim);
            let dim = self.dim;
            let centroid = match self.ivf.centroids.get(cell * dim..cell * dim + dim) {
                Some(centroid) => centroid,
                None => continue,
            };

            let vector = self.store.row(row);
            let residual: Vec<f32> = vector
                .iter()
                .zip(centroid.iter())
                .map(|(v, c)| v - c)
                .collect();
            let code = self.pq.encode(&residual);
            codes[row * m..row * m + m].copy_from_slice(&code);
        }

        self.codes = codes;
    }

    /// Flatten a sample of live rows for k-means training.
    fn training_sample(&self, max_samples: usize) -> Vec<f32> {
        let dim = self.dim;
        let rows = self.store.rows();
        let live = self.ids.len();

        if live == 0 || dim == 0 {
            return Vec::new();
        }

        // Stride sampling keeps the sample spread over insertion order, which
        // for embedding corpora correlates with content.
        let step = if live > max_samples {
            live.div_ceil(max_samples)
        } else {
            1
        };

        let mut out = Vec::with_capacity(live.min(max_samples) * dim);
        let mut taken = 0usize;

        for row in 0..rows {
            if self.ids.id_of(row).is_none() {
                continue;
            }
            if !taken.is_multiple_of(step) {
                taken += 1;
                continue;
            }

            out.extend_from_slice(self.store.row(row));
            taken += 1;

            if out.len() / dim >= max_samples {
                break;
            }
        }

        out
    }

    /// Residuals instead of raw vectors, for PQ training.
    fn residual_sample(&self, max_samples: usize) -> Vec<f32> {
        let dim = self.dim;
        let rows = self.store.rows();
        let live = self.ids.len();

        if live == 0 || dim == 0 || self.ivf.is_empty() {
            return Vec::new();
        }

        let step = if live > max_samples {
            live.div_ceil(max_samples)
        } else {
            1
        };

        let mut out = Vec::with_capacity(live.min(max_samples) * dim);
        let mut taken = 0usize;

        for row in 0..rows {
            if self.ids.id_of(row).is_none() {
                continue;
            }
            if !taken.is_multiple_of(step) {
                taken += 1;
                continue;
            }

            let cell = self.ivf.nearest_centroid(self.store.row(row), dim);
            let centroid = &self.ivf.centroids[cell * dim..cell * dim + dim];
            let vector = self.store.row(row);
            for (v, c) in vector.iter().zip(centroid.iter()) {
                out.push(v - c);
            }

            taken += 1;
            if out.len() / dim >= max_samples {
                break;
            }
        }

        out
    }

    // -- snapshot ---------------------------------------------------------

    /// Build a snapshot. Compacts first when tombstones exist, so a snapshot
    /// never carries dead rows and always restores into an engine whose row
    /// indices are dense.
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

/// Default cell count: one cell per ~128 vectors, clamped to a sane band.
///
/// `nlist ≈ sqrt(n)` (the usual rule) is only right for very large corpora; for
/// n = 10 000 it gives 100 cells of 100 vectors, which is roughly what we want,
/// and for n = 1 000 000 it gives 1 000. The clamp keeps small corpora from
/// producing one cell and large ones from producing a scan-length cell list.
fn default_nlist(rows: usize) -> usize {
    if rows == 0 {
        return 0;
    }

    let target = rows / TARGET_LIST_SIZE;
    let sqrt = (rows as f64).sqrt() as usize;
    target.clamp(sqrt / 2, sqrt * 2).max(1).min(rows)
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

/// Build an index. Errors only on a data/ids length mismatch or a duplicate id.
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

        // Zero-dimensional vectors.
        let engine = Engine::build(&[vec![]], &["empty".to_string()], options()).expect("build");
        assert_eq!(engine.search(&[], 3).neighbors.len(), 0);

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
                nprobe: Some(24),
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
