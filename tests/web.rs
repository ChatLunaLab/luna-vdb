//! Test suite for the browser, run headless via `wasm-pack test --headless --chrome`.
//!
//! The tests themselves live in `tests/common/wasm_suite.rs` and are shared
//! with the node variant; only the host configuration below differs.

#![cfg(target_arch = "wasm32")]

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "common/wasm_suite.rs"]
mod suite;
