//! luna-vdb — a small, fast, WASM-first vector database.
//!
//! Layout:
//! * [`engine`] — pure-Rust core. No wasm-bindgen, no JS. Usable from native
//!   Rust, and the only place where the algorithms live.
//! * [`wasm`] — the JS-facing surface. Every fallible entry point returns
//!   `Result<_, JsValue>`; nothing here unwraps, so a bad call surfaces as a
//!   catchable JS exception instead of a poisoned wasm instance.
//!
//! # Panic policy
//!
//! The engine must never panic on user input. Anything that could fail is
//! returned as [`engine::EngineError`]. The clippy restriction lints below
//! are the enforcement mechanism — they are denied in CI.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
    )
)]
#![allow(clippy::needless_range_loop)]

pub mod engine;
mod utils;
mod wasm;

pub use engine::{Distance, EngineError};
pub use wasm::*;

/// Which SIMD backend was compiled in. Useful when filing perf issues.
///
/// Possible values: `"wasm-simd128"`, `"avx2"`, `"neon"`, `"scalar"`.
pub fn simd_backend() -> &'static str {
    engine::simd::KERNEL_NAME
}

/// `true` when the wasm module was compiled with SIMD128 enabled.
///
/// A module built with SIMD128 will fail to *instantiate* on a runtime that
/// does not support it, so JS callers that need to support ancient runtimes
/// should check this before importing — or just import
/// `@chatluna/luna-vdb/scalar` instead.
pub fn has_simd() -> bool {
    cfg!(target_feature = "simd128")
}
