//! luna-vdb — a small, fast, WASM-first vector database.
//!
//! Layout:
//! * [`engine`] — pure-Rust core. No wasm-bindgen, no JS. Usable from native
//!   Rust, and the only place where the algorithms live.
//! * `wasm` — the JS-facing surface, compiled only for `wasm32`. Every
//!   fallible entry point returns `Result<_, JsValue>`; nothing there unwraps,
//!   so a bad call surfaces as a catchable JS exception instead of a poisoned
//!   wasm instance. It is not built for native targets at all: `JsValue`
//!   needs a JS host, and on a native target constructing one aborts the
//!   process, so a native `LunaVDB` could only ever fail by crashing. Native
//!   Rust callers use [`engine::Engine`] directly.
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
#[cfg(target_arch = "wasm32")]
mod utils;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use engine::{Distance, EngineError};
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

/// Which SIMD backend is running. Useful when filing perf issues.
///
/// Possible values: `"wasm-simd128"`, `"avx2"`, `"neon"`, `"scalar"`. On
/// x86_64 this reflects the runtime CPU check, so an AVX2 build running on a
/// CPU without AVX2 reports `"scalar"`.
pub fn simd_backend() -> &'static str {
    engine::simd::backend_name()
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

/// This crate's version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// `true` if `bytes` starts like a snapshot this build can read. See
/// [`engine::codec::is_snapshot`].
pub fn is_snapshot(bytes: &[u8]) -> bool {
    engine::codec::is_snapshot(bytes)
}

/// Format version of a snapshot, `0` if unrecognised. See
/// [`engine::codec::snapshot_version`].
pub fn snapshot_version(bytes: &[u8]) -> u16 {
    engine::codec::snapshot_version(bytes)
}
