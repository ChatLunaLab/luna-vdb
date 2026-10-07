//! String → row-index map.
//!
//! The old code used `std::collections::HashMap<u64, usize>` keyed by
//! `DefaultHasher`. Two problems:
//!
//! 1. **Correctness.** `DefaultHasher` is not collision-resistant in a way that
//!    matters here — it is `SipHash-1-3`, which is fine cryptographically, but
//!    the map trusted the hash as the identity. Two different ids that hashed
//!    the same would silently overwrite each other, and `add()` would reject a
//!    brand-new id with "id already exists". Rare, but unreproducible and
//!    impossible to debug from a bug report.
//! 2. **Speed.** `SipHash` is slow (~1 byte/cycle) and the old code hashed the
//!    id *string* on every `add`/`remove`. We hash once at insert time and
//!    store both the hash and the string, so lookups compare the hash first and
//!    only then the string.
//!
//! We keep the hash for bucketing but always confirm with an actual string
//! comparison, which makes collisions a non-event rather than a data-loss bug.

use std::collections::HashMap;

use crate::engine::types::{EngineError, EngineResult};

/// FNV-1a, 64-bit. Chosen over `SipHash` purely for throughput on short
/// strings — these are user-supplied document ids, not attacker-controlled
/// secrets where HashDoS applies, and a collision is now harmless anyway.
#[inline]
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// FNV-1a over a string's bytes.
#[inline]
pub fn hash_str(value: &str) -> u64 {
    hash_bytes(value.as_bytes())
}

/// One bucket's worth of entries.
type Bucket = Vec<(u64, u32)>;

/// Maps ids to row indices.
///
/// Deletion is a tombstone: [`IdMap::remove`] records the dead row and
/// [`IdMap::compact`] squeezes them out in one pass. Rebuilding the map on
/// every delete (what the old `remove` did, via `swap_remove` plus a rehash of
/// the moved element) is why deleting was as expensive as inserting.
#[derive(Debug, Clone, Default)]
pub struct IdMap {
    buckets: HashMap<u64, Bucket>,
    /// Row index → id, indexed by row. `None` marks a tombstone.
    rows: Vec<Option<String>>,
    live: usize,
}

impl IdMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buckets: HashMap::with_capacity(capacity),
            rows: Vec::with_capacity(capacity),
            live: 0,
        }
    }

    /// Number of live (non-tombstoned) entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Total row slots, including tombstones.
    #[inline]
    pub fn slots(&self) -> usize {
        self.rows.len()
    }

    /// Number of tombstoned rows.
    #[inline]
    pub fn dead(&self) -> usize {
        self.rows.len() - self.live
    }

    /// Id for a row, or `None` if the row is a tombstone / out of range.
    #[inline]
    pub fn id_of(&self, row: usize) -> Option<&str> {
        self.rows.get(row).and_then(|slot| slot.as_deref())
    }

    /// Look up a row by id.
    pub fn get(&self, id: &str) -> Option<usize> {
        let hash = hash_str(id);
        let bucket = self.buckets.get(&hash)?;

        bucket
            .iter()
            .find(|(_, row)| self.id_of(*row as usize) == Some(id))
            .map(|(_, row)| *row as usize)
    }

    /// `true` if the id is present and live.
    #[inline]
    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Insert `id` at the next free row. Returns the row index.
    ///
    /// Errors on a duplicate id — matching the previous behaviour, but now the
    /// check is a real string comparison rather than a hash equality.
    pub fn insert(&mut self, id: String) -> EngineResult<u32> {
        if self.contains(&id) {
            return Err(EngineError::duplicate_id(&id));
        }

        let hash = hash_str(&id);
        let row = self.rows.len() as u32;
        self.rows.push(Some(id));
        self.buckets.entry(hash).or_default().push((hash, row));
        self.live += 1;
        Ok(row)
    }

    /// Insert without the duplicate check, overwriting any existing entry for
    /// the same id. Used when applying an externally-ordered update batch where
    /// last-write-wins is the intended semantics.
    pub fn upsert(&mut self, id: String) -> u32 {
        match self.get(&id) {
            Some(row) => row as u32,
            None => match self.insert(id) {
                Ok(row) => row,
                // `insert` only fails on a duplicate, which `get` just ruled
                // out, so this arm is unreachable; degrade to a fresh slot
                // rather than unwrapping.
                Err(_) => {
                    let row = self.rows.len() as u32;
                    self.rows.push(None);
                    row
                }
            },
        }
    }

    /// Mark the row for `id` as dead. Returns the freed row index.
    pub fn remove(&mut self, id: &str) -> EngineResult<u32> {
        let row = self.get(id).ok_or_else(|| EngineError::missing_id(id))? as u32;

        let hash = hash_str(id);
        if let Some(bucket) = self.buckets.get_mut(&hash) {
            bucket.retain(|(_, entry_row)| *entry_row != row);
            if bucket.is_empty() {
                self.buckets.remove(&hash);
            }
        }

        if let Some(slot) = self.rows.get_mut(row as usize) {
            *slot = None;
        }
        self.live -= 1;
        Ok(row)
    }

    /// All live ids in row order, paired with their row index.
    pub fn entries(&self) -> impl Iterator<Item = (u32, &str)> {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| slot.as_deref().map(|id| (row as u32, id)))
    }

    /// Rebuild the map after tombstones are removed. `mapping[old_row]` is the
    /// new row, or `None` if the old row was dead.
    ///
    /// The returned id list is in new-row order and is what the snapshot writes.
    pub fn compact(&mut self, mapping: &[Option<u32>]) -> Vec<String> {
        let mut buckets: HashMap<u64, Bucket> = HashMap::with_capacity(self.buckets.len());
        let mut rows: Vec<Option<String>> = Vec::with_capacity(self.live);
        let mut all_ids: Vec<String> = Vec::with_capacity(self.live);

        for (old_row, slot) in self.rows.iter().enumerate() {
            let (Some(id), Some(Some(new_row))) = (slot.as_deref(), mapping.get(old_row)) else {
                continue;
            };

            let hash = hash_str(id);
            buckets.entry(hash).or_default().push((hash, *new_row));
            all_ids.push(id.to_string());
            rows.push(Some(id.to_string()));
        }

        self.buckets = buckets;
        self.rows = rows;
        self.live = all_ids.len();
        all_ids
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.rows.clear();
        self.live = 0;
    }

    /// Replace the entire contents. Used by the snapshot loader.
    pub fn from_ids(ids: Vec<String>) -> Self {
        let mut map = Self::with_capacity(ids.len());
        for id in ids {
            // Duplicates in a snapshot would be a writer bug; keep the first
            // and drop the rest rather than failing the whole load.
            if map.contains(&id) {
                continue;
            }
            let hash = hash_str(&id);
            let row = map.rows.len() as u32;
            map.buckets.entry(hash).or_default().push((hash, row));
            map.rows.push(Some(id));
            map.live += 1;
        }
        map
    }

    /// Approximate heap usage, for `stats()`.
    pub fn memory_bytes(&self) -> usize {
        let bucket_bytes: usize = self
            .buckets
            .values()
            .map(|bucket| bucket.capacity() * std::mem::size_of::<(u64, u32)>())
            .sum();
        let id_bytes: usize = self
            .rows
            .iter()
            .flatten()
            .map(|id| id.capacity() + std::mem::size_of::<String>())
            .sum();

        bucket_bytes + id_bytes + self.buckets.capacity() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut map = IdMap::new();
        let row = map.insert("a".to_string()).expect("insert a");
        assert_eq!(row, 0);
        assert_eq!(map.get("a"), Some(0));
        assert_eq!(map.len(), 1);

        assert!(map.insert("a".to_string()).is_err());
        assert_eq!(map.len(), 1);

        assert_eq!(map.remove("a").expect("remove a"), 0);
        assert_eq!(map.get("a"), None);
        assert_eq!(map.len(), 0);
        assert_eq!(map.dead(), 1);
        assert!(map.remove("a").is_err());
    }

    #[test]
    fn compaction_renumbers() {
        let mut map = IdMap::new();
        for id in ["a", "b", "c", "d"] {
            map.insert(id.to_string()).expect("insert");
        }
        map.remove("b").expect("remove b");

        // Old rows: a=0, b=1(dead), c=2, d=3 → new: a=0, c=1, d=2
        let mapping = vec![Some(0u32), None, Some(1), Some(2)];
        let ids = map.compact(&mapping);

        assert_eq!(ids, vec!["a", "c", "d"]);
        assert_eq!(map.len(), 3);
        assert_eq!(map.dead(), 0);
        assert_eq!(map.get("a"), Some(0));
        assert_eq!(map.get("c"), Some(1));
        assert_eq!(map.get("d"), Some(2));
        assert_eq!(map.get("b"), None);
    }

    #[test]
    fn colliding_hashes_are_disambiguated_by_string() {
        // The old map was `HashMap<u64, usize>` — keyed on the hash alone, so
        // two ids that hashed alike made one silently overwrite the other, and
        // `add` would then reject a genuinely new id as a duplicate.
        //
        // We cannot easily synthesise a real FNV-1a collision, but we can
        // reproduce the *situation* the fix guards against: a bucket whose
        // entries do not all belong to the id being looked up. `get` must
        // filter by string rather than trusting the bucket.
        let mut map = IdMap::new();
        assert_eq!(map.insert("first".to_string()).expect("insert"), 0);
        assert_eq!(map.insert("second".to_string()).expect("insert"), 1);

        let second_hash = hash_str("second");
        let bucket = map.buckets.get_mut(&second_hash).expect("second bucket");

        // Poison the bucket: a stale entry for a *different* id points at row
        // 0, and it comes first. A hash-only lookup would return row 0 here.
        bucket.insert(0, (second_hash, 0));

        assert_eq!(map.get("second"), Some(1), "must skip the foreign entry");
        assert_eq!(map.get("first"), Some(0));

        // Removing `second` must drop only its entries, leaving `first` alone.
        assert_eq!(map.remove("second").expect("remove"), 1);
        assert_eq!(map.get("second"), None);
        assert_eq!(map.get("first"), Some(0));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn distinct_ids_hash_differently_in_practice() {
        // Not a guarantee the data structure relies on — just a sanity check
        // that FNV-1a is spreading short ids well enough to keep buckets small.
        let mut seen = std::collections::HashSet::new();
        for i in 0..1000 {
            assert!(
                seen.insert(hash_str(&format!("document-{i}"))),
                "collision at {i}"
            );
        }
    }

    #[test]
    fn from_ids_drops_duplicates() {
        let map = IdMap::from_ids(vec!["x".into(), "x".into(), "y".into()]);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("y"), Some(1));
    }
}
