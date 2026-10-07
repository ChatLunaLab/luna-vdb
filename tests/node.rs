//! Test suite for Node.js, run via `wasm-pack test --node`.
//!
//! The tests themselves live in `tests/common/wasm_suite.rs` and are shared
//! with the browser variant; the only difference is the absence of
//! `wasm_bindgen_test_configure!(run_in_browser)`.

#![cfg(target_arch = "wasm32")]

#[path = "common/wasm_suite.rs"]
mod suite;
