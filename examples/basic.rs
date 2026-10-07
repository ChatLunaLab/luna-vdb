//! The Rust example from the README, compiled by CI so it cannot drift.
//!
//! ```text
//! cargo run --example basic
//! ```

use luna_vdb::engine::{self, Engine, IndexOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![vec![0.8, 0.7, 0.6], vec![0.4, 0.3, 0.2]];
    let ids = vec!["cat".to_string(), "dog".to_string()];
    let mut db = Engine::build(&data, &ids, IndexOptions::default())?;

    db.add("bird".to_string(), &[0.2, 0.1, 0.9])?;
    for hit in db.search(&[0.75, 0.65, 0.55], 2).neighbors {
        println!("{}: {}", hit.id, hit.distance);
    }

    // Same format as `serialize()` on the JS side; each can read the other's.
    let bytes = db.serialize(true)?;
    let restored = engine::load(&bytes)?;
    assert_eq!(restored.len(), 3);
    Ok(())
}
