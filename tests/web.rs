//! Test suite for the browser, run headless via `wasm-pack test --headless --chrome`.
//!
//! The assertions live in `tests/common/wasm_suite.rs` and are shared with the
//! node variant; only the host configuration below differs.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;

use wasm_bindgen_test::*;

#[path = "common/wasm_suite.rs"]
mod suite;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn new_empty() {
    suite::test_new_empty();
}

#[wasm_bindgen_test]
fn new_with_options() {
    suite::test_new_with_options();
}

#[wasm_bindgen_test]
fn invalid_option_does_not_poison() {
    suite::test_invalid_option_does_not_poison();
}

#[wasm_bindgen_test]
fn index_and_size() {
    suite::test_index_and_size();
}

#[wasm_bindgen_test]
fn index_replaces_contents() {
    suite::test_index_replaces_contents();
}

#[wasm_bindgen_test]
fn search_ranks_neighbours() {
    suite::test_search_ranks_neighbours();
}

#[wasm_bindgen_test]
fn search_boundary_vectors() {
    suite::test_search_boundary_vectors();
}

#[wasm_bindgen_test]
fn search_k_greater_than_size() {
    suite::test_search_k_greater_than_size();
}

#[wasm_bindgen_test]
fn search_empty_engine() {
    suite::test_search_empty_engine();
}

#[wasm_bindgen_test]
fn search_reports_work_done() {
    suite::test_search_reports_work_done();
}

#[wasm_bindgen_test]
fn add_and_remove() {
    suite::test_add_and_remove();
}

#[wasm_bindgen_test]
fn clear_stays_usable() {
    suite::test_clear_stays_usable();
}

#[wasm_bindgen_test]
fn large_corpus_ingest() {
    suite::test_large_corpus_ingest();
}

#[wasm_bindgen_test]
fn duplicate_id_throws() {
    suite::test_duplicate_id_throws();
}

#[wasm_bindgen_test]
fn dimension_mismatch_throws() {
    suite::test_dimension_mismatch_throws();
}

#[wasm_bindgen_test]
fn remove_unknown_id_throws() {
    suite::test_remove_unknown_id_throws();
}

#[wasm_bindgen_test]
fn serialize_round_trip() {
    suite::test_serialize_round_trip();
}

#[wasm_bindgen_test]
fn serialize_uncompressed() {
    suite::test_serialize_uncompressed();
}

#[wasm_bindgen_test]
fn serialize_after_mutation() {
    suite::test_serialize_after_mutation();
}

#[wasm_bindgen_test]
fn empty_round_trip() {
    suite::test_empty_round_trip();
}

#[wasm_bindgen_test]
fn corrupt_snapshot_is_recoverable() {
    suite::test_corrupt_snapshot_is_recoverable();
}

#[wasm_bindgen_test]
fn restore_into_preserves_on_failure() {
    suite::test_restore_into_preserves_on_failure();
}

#[wasm_bindgen_test]
fn cosine_is_scale_invariant() {
    suite::test_cosine_is_scale_invariant();
}

#[wasm_bindgen_test]
fn dot_product() {
    suite::test_dot_product();
}

#[wasm_bindgen_test]
fn stats() {
    suite::test_stats();
}

#[wasm_bindgen_test]
fn simd_backend_is_known() {
    suite::test_simd_backend_is_known();
}

#[wasm_bindgen_test]
fn version_reported() {
    suite::test_version_reported();
}

#[wasm_bindgen_test]
fn compaction_preserves_results() {
    suite::test_compaction_preserves_results();
}
