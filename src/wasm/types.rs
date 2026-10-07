//! Re-exports for the wasm surface.
//!
//! The transfer types live next to the bindings that use them
//! ([`crate::wasm::luna_vdb`]); this module only re-exports what the crate root
//! and the JS typings need, so `tsify` sees one definition per type.

/// A `k`-nearest-neighbour count.
pub type TopK = usize;

/// Serialised index bytes, as produced by `serialize()`.
pub type SerializedIndex = Vec<u8>;
