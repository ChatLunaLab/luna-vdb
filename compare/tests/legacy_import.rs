//! Snapshots written by the real pre-rewrite engine must load in this one.
//!
//! The codec's own unit tests encode a legacy snapshot from a hand-copied
//! mirror of the old structs, which proves the decoder agrees with the mirror
//! — not that the mirror agrees with what the old engine actually wrote. This
//! test closes that gap: it links the old engine (commit `ca10590`), has it
//! serialise, and imports the bytes.

use legacy::{EmbeddedResource as OldItem, LunaVDB as OldDb, Resource as OldResource};
use luna_vdb::engine::{self, Distance};

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn corpus(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|row| {
            let mut state = splitmix(seed.wrapping_add(row as u64)) | 1;
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn old_snapshot(data: &[Vec<f32>]) -> Vec<u8> {
    let resource = OldResource {
        embeddings: data
            .iter()
            .enumerate()
            .map(|(i, vector)| OldItem {
                id: format!("doc-{i}"),
                embeddings: vector.clone(),
            })
            .collect(),
    };
    OldDb::new(Some(resource)).serialize()
}

#[test]
fn imports_snapshots_from_the_pre_rewrite_engine() {
    // Below and above the old engine's 20 000-vector IVF threshold, so both of
    // its on-disk shapes — with and without IVF/PQ fields filled — are covered.
    for (count, dim) in [(0usize, 0usize), (1, 3), (500, 32), (20_500, 16)] {
        let data = corpus(count, dim, 0x01D ^ count as u64);
        let bytes = old_snapshot(&data);

        assert!(luna_vdb::is_snapshot(&bytes), "n={count}");
        assert_eq!(luna_vdb::snapshot_version(&bytes), 1, "n={count}");

        let imported = engine::load(&bytes).unwrap_or_else(|e| panic!("n={count}: {e}"));
        assert_eq!(imported.len(), count);
        assert_eq!(imported.distance(), Distance::Euclidean);
        if count == 0 {
            continue;
        }
        assert_eq!(imported.dim(), dim);

        // Every vector came across intact and kept its id: each one is its
        // own nearest neighbour at distance 0.
        for row in (0..count).step_by((count / 50).max(1)) {
            let hit = &imported.search(&data[row], 1).neighbors[0];
            assert_eq!(hit.id, format!("doc-{row}"), "n={count}");
            assert!(
                hit.distance.abs() < 1e-6,
                "n={count} row={row}: {}",
                hit.distance
            );
        }

        // And it re-serialises in the new format.
        let again = engine::load(&imported.serialize(true).expect("serialize")).expect("reload");
        assert_eq!(again.len(), count);
    }
}
