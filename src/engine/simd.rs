//! Distance kernels, hand-written per architecture.
//!
//! # Why this exists
//!
//! The previous implementation used `a.iter().zip(b).fold(0.0, |acc, (x, y)|
//! acc + x * y)`. That is a *serial dependent* chain: every multiply-add waits
//! for the previous one's result, so throughput is bounded by FMA latency
//! (~4 cycles on most cores), not by width. At 4 cycles per element a 1024-d
//! vector costs ~4096 cycles ≈ 1.7 µs, and it cannot be fixed by the compiler
//! because floating-point addition is not associative — auto-vectorisation of
//! a plain sum is illegal without `-ffast-math`.
//!
//! Each kernel here uses **multiple independent accumulators** so the FMA
//! units stay saturated, and reads a contiguous `&[f32]`, which is only
//! possible because [`crate::engine::types::VectorStore`] keeps rows in one
//! flat buffer.
//!
//! # Backends
//!
//! | target                              | backend      |
//! |-------------------------------------|--------------|
//! | `wasm32` + `-C target-feature=+simd128` | `wasm-simd128` |
//! | `x86_64` with AVX2+FMA (runtime-detected) | `avx2`   |
//! | `aarch64`                           | `neon`       |
//! | anything else                       | `scalar`     |
//!
//! `simd_backend()` in the crate root reports which one is live.
//!
//! # Accuracy
//!
//! All backends compute the same mathematical quantity; only the summation
//! order differs. Results are therefore *not* bit-identical across backends,
//! which is why the test suite compares scores with a tolerance rather than
//! for equality across targets. Within one build the order is deterministic,
//! so a snapshot round-trip is exact.

/// Which kernel implementation is compiled in. See the module docs.
pub const KERNEL_NAME: &str = {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        "wasm-simd128"
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    {
        "avx2"
    }
    #[cfg(all(target_arch = "aarch64", not(target_arch = "wasm32")))]
    {
        "neon"
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(target_arch = "x86_64", not(target_arch = "wasm32")),
        all(target_arch = "aarch64", not(target_arch = "wasm32")),
    )))]
    {
        "scalar"
    }
};

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

/// Four independent accumulators.
///
/// With a single accumulator the loop is latency-bound at ~4 cycles per
/// element. Four accumulators let the core issue one FMA per cycle, so the
/// same code runs roughly 3-4x faster with no SIMD at all — this is the floor
/// every other backend is measured against.
#[inline]
pub fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let mut chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
        acc0 += x[0] * y[0];
        acc1 += x[1] * y[1];
        acc2 += x[2] * y[2];
        acc3 += x[3] * y[3];
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3);
    for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
        total += x * y;
    }
    total
}

#[inline]
pub fn scalar_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let mut chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
        let d0 = x[0] - y[0];
        let d1 = x[1] - y[1];
        let d2 = x[2] - y[2];
        let d3 = x[3] - y[3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3);
    for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
        let d = x - y;
        total += d * d;
    }
    total
}

#[inline]
pub fn scalar_norm_sq(a: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let mut chunks = a.chunks_exact(4);
    for x in chunks.by_ref() {
        acc0 += x[0] * x[0];
        acc1 += x[1] * x[1];
        acc2 += x[2] * x[2];
        acc3 += x[3] * x[3];
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3);
    for &x in chunks.remainder() {
        total += x * x;
    }
    total
}

// ---------------------------------------------------------------------------
// wasm32 + simd128
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    // Imported under distinct names on purpose. Naming them `add`/`sub`/`mul`
    // and then wrapping them is how this module first shipped, and the wrappers
    // resolved to *themselves* rather than the intrinsics: an infinitely
    // recursive `mul` that the scalar build never compiles, so nothing caught
    // it until the SIMD target was actually exercised.
    //
    // In `core::arch::wasm32` these are ordinary safe functions — the compiler
    // emits the SIMD instruction directly — so the kernel bodies need no
    // `unsafe` at all.
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, f32x4_sub, v128, v128_load,
    };

    /// Horizontal sum of four lanes.
    #[inline]
    fn hadd(v: v128) -> f32 {
        let hi = f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v);
        let lo = f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v);
        lo + hi
    }

    /// Load four f32 as one `v128`.
    #[inline]
    fn load16(ptr: *const f32) -> v128 {
        // SAFETY: caller guarantees 16 readable bytes at `ptr`. Every call site
        // passes a pointer into a `chunks_exact(4 or 8 or 16)` slice, so the
        // four lanes are always in bounds.
        unsafe { v128_load(ptr as *const v128) }
    }

    #[inline]
    fn mul(a: v128, b: v128) -> v128 {
        f32x4_mul(a, b)
    }

    #[inline]
    fn add(a: v128, b: v128) -> v128 {
        f32x4_add(a, b)
    }

    #[inline]
    fn sub(a: v128, b: v128) -> v128 {
        f32x4_sub(a, b)
    }

    #[inline]
    fn zero() -> v128 {
        f32x4_splat(0.0)
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc0 = zero();
        let mut acc1 = zero();
        let mut acc2 = zero();
        let mut acc3 = zero();

        let mut chunks = a[..len].chunks_exact(16);
        let mut b_chunks = b[..len].chunks_exact(16);

        for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
            // SAFETY: each chunk is exactly 16 f32 and offsets 0/4/8/12 stay
            // inside it, so all four 16-byte loads are in bounds. The pointer
            // offsets themselves are the only unsafe arithmetic here; the
            // SIMD intrinsics are safe functions on wasm.
            unsafe {
                acc0 = add(acc0, mul(load16(x.as_ptr()), load16(y.as_ptr())));
                acc1 = add(
                    acc1,
                    mul(load16(x.as_ptr().add(4)), load16(y.as_ptr().add(4))),
                );
                acc2 = add(
                    acc2,
                    mul(load16(x.as_ptr().add(8)), load16(y.as_ptr().add(8))),
                );
                acc3 = add(
                    acc3,
                    mul(load16(x.as_ptr().add(12)), load16(y.as_ptr().add(12))),
                );
            }
        }

        let mut total = hadd(add(add(acc0, acc1), add(acc2, acc3)));
        for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
            total += x * y;
        }
        total
    }

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc0 = zero();
        let mut acc1 = zero();

        let mut chunks = a[..len].chunks_exact(8);
        let mut b_chunks = b[..len].chunks_exact(8);

        for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
            // SAFETY: each chunk is exactly 8 f32; offsets 0 and 4 are both
            // inside it.
            unsafe {
                let x0 = load16(x.as_ptr());
                let y0 = load16(y.as_ptr());
                let d0 = sub(x0, y0);
                acc0 = add(acc0, mul(d0, d0));

                let x1 = load16(x.as_ptr().add(4));
                let y1 = load16(y.as_ptr().add(4));
                let d1 = sub(x1, y1);
                acc1 = add(acc1, mul(d1, d1));
            }
        }

        let mut total = hadd(add(acc0, acc1));
        for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
            let d = x - y;
            total += d * d;
        }
        total
    }

    #[inline]
    pub fn norm_sq(a: &[f32]) -> f32 {
        let mut acc0 = zero();
        let mut acc1 = zero();

        let mut chunks = a.chunks_exact(8);
        for x in chunks.by_ref() {
            // SAFETY: each chunk is exactly 8 f32; offsets 0 and 4 are inside.
            unsafe {
                let v0 = load16(x.as_ptr());
                acc0 = add(acc0, mul(v0, v0));

                let v1 = load16(x.as_ptr().add(4));
                acc1 = add(acc1, mul(v1, v1));
            }
        }

        let mut total = hadd(add(acc0, acc1));
        for &x in chunks.remainder() {
            total += x * x;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// x86_64: AVX2 + FMA, selected at runtime
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
mod imp {
    use std::sync::atomic::{AtomicU8, Ordering};

    const UNKNOWN: u8 = 0;
    const SCALAR: u8 = 1;
    const AVX2: u8 = 2;

    /// Cached CPU capability. `is_x86_feature_detected!` reads an atomic and a
    /// cached bitset after the first call, but this avoids even that on the
    /// per-row hot path.
    static LEVEL: AtomicU8 = AtomicU8::new(UNKNOWN);

    #[inline]
    fn level() -> u8 {
        let cached = LEVEL.load(Ordering::Relaxed);
        if cached != UNKNOWN {
            return cached;
        }

        // AVX2 gives 8-wide f32; FMA halves the instruction count again. Both
        // are required — AVX2 alone still helps, but the FMA path is the one
        // that gets us near peak FLOPs.
        let detected = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            AVX2
        } else {
            SCALAR
        };
        LEVEL.store(detected, Ordering::Relaxed);
        detected
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        if level() == AVX2 {
            // SAFETY: guarded by the runtime AVX2+FMA check above.
            unsafe { avx2_dot(a, b) }
        } else {
            super::scalar_dot(a, b)
        }
    }

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        if level() == AVX2 {
            // SAFETY: guarded by the runtime AVX2+FMA check above.
            unsafe { avx2_l2_sq(a, b) }
        } else {
            super::scalar_l2_sq(a, b)
        }
    }

    #[inline]
    pub fn norm_sq(a: &[f32]) -> f32 {
        if level() == AVX2 {
            // SAFETY: guarded by the runtime AVX2+FMA check above.
            unsafe { avx2_norm_sq(a) }
        } else {
            super::scalar_norm_sq(a)
        }
    }

    // -----------------------------------------------------------------
    // Safe wrappers around the intrinsics.
    //
    // Rust 2024 turned `unsafe_op_in_unsafe_fn` into a lint, and this crate
    // denies warnings in CI. Rather than sprinkling `unsafe {}` through the
    // kernels — which would bury the actual vector logic in noise and make it
    // easy to lose track of which pointer arithmetic is justified — each
    // intrinsic gets a `#[inline]` safe wrapper holding exactly one `unsafe`
    // block. The kernels below are then ordinary safe Rust, and the safety
    // argument lives in one place per operation.
    //
    // The wrappers are `#[target_feature(enable = "avx2")]`, so they are only
    // callable from a context that already enabled AVX2. Callers reach them
    // exclusively through `level() == AVX2`, and a `#[target_feature]` fn may
    // only call another of the same feature set, so the guarantee propagates
    // through the type system rather than by convention. Calling one when the
    // CPU lacks AVX2 would be UB, which is why `level()` is the only gate.
    // -----------------------------------------------------------------

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn loadu(ptr: *const f32) -> std::arch::x86_64::__m256 {
        // SAFETY: caller guarantees 8 readable f32 at `ptr`. This is the single
        // place that assumption is expressed.
        unsafe { std::arch::x86_64::_mm256_loadu_ps(ptr) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd(
        a: std::arch::x86_64::__m256,
        b: std::arch::x86_64::__m256,
        c: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        // SAFETY: pure register arithmetic; `fma` is enabled by the attribute.
        std::arch::x86_64::_mm256_fmadd_ps(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn sub(
        a: std::arch::x86_64::__m256,
        b: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        // SAFETY: pure register arithmetic.
        std::arch::x86_64::_mm256_sub_ps(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn add(
        a: std::arch::x86_64::__m256,
        b: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        // SAFETY: pure register arithmetic.
        std::arch::x86_64::_mm256_add_ps(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn zero() -> std::arch::x86_64::__m256 {
        // SAFETY: `setzero_ps` has no preconditions.
        std::arch::x86_64::_mm256_setzero_ps()
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: every call below is to a `#[target_feature(enable = "avx2")]`
        // helper reached only from a context that enabled AVX2, and every load
        // reads within a chunk proven to hold 32 f32.
        unsafe {
            let len = a.len().min(b.len());
            let mut acc0 = zero();
            let mut acc1 = zero();
            let mut acc2 = zero();
            let mut acc3 = zero();

            let mut chunks = a[..len].chunks_exact(32);
            let mut b_chunks = b[..len].chunks_exact(32);

            for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
                let xp = x.as_ptr();
                let yp = y.as_ptr();

                acc0 = fmadd(loadu(xp), loadu(yp), acc0);
                acc1 = fmadd(loadu(xp.add(8)), loadu(yp.add(8)), acc1);
                acc2 = fmadd(loadu(xp.add(16)), loadu(yp.add(16)), acc2);
                acc3 = fmadd(loadu(xp.add(24)), loadu(yp.add(24)), acc3);
            }

            let combined = add(add(acc0, acc1), add(acc2, acc3));
            let mut total = hsum_ps(combined);
            for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
                total += x * y;
            }
            total
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_l2_sq(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: as in `avx2_dot` — AVX2-enabled context, reads within a
        // proven 32-element chunk.
        unsafe {
            let len = a.len().min(b.len());
            let mut acc0 = zero();
            let mut acc1 = zero();
            let mut acc2 = zero();
            let mut acc3 = zero();

            let mut chunks = a[..len].chunks_exact(32);
            let mut b_chunks = b[..len].chunks_exact(32);

            for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
                let xp = x.as_ptr();
                let yp = y.as_ptr();

                // Written out rather than looped over a `(index, offset)` table:
                // the table version needed a `match` to pick the accumulator on
                // each iteration, which defeats the whole point of having four
                // independent accumulators — the compiler could not keep them in
                // registers.
                let d0 = sub(loadu(xp), loadu(yp));
                acc0 = fmadd(d0, d0, acc0);

                let d1 = sub(loadu(xp.add(8)), loadu(yp.add(8)));
                acc1 = fmadd(d1, d1, acc1);

                let d2 = sub(loadu(xp.add(16)), loadu(yp.add(16)));
                acc2 = fmadd(d2, d2, acc2);

                let d3 = sub(loadu(xp.add(24)), loadu(yp.add(24)));
                acc3 = fmadd(d3, d3, acc3);
            }

            let combined = add(add(acc0, acc1), add(acc2, acc3));
            let mut total = hsum_ps(combined);
            for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
                let d = x - y;
                total += d * d;
            }
            total
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_norm_sq(a: &[f32]) -> f32 {
        // SAFETY: AVX2-enabled context, reads within a proven 32-element chunk.
        unsafe {
            let mut acc0 = zero();
            let mut acc1 = zero();
            let mut acc2 = zero();
            let mut acc3 = zero();

            let mut chunks = a.chunks_exact(32);
            for x in chunks.by_ref() {
                let xp = x.as_ptr();

                let v0 = loadu(xp);
                acc0 = fmadd(v0, v0, acc0);
                let v1 = loadu(xp.add(8));
                acc1 = fmadd(v1, v1, acc1);
                let v2 = loadu(xp.add(16));
                acc2 = fmadd(v2, v2, acc2);
                let v3 = loadu(xp.add(24));
                acc3 = fmadd(v3, v3, acc3);
            }

            let combined = add(add(acc0, acc1), add(acc2, acc3));
            let mut total = hsum_ps(combined);
            for &x in chunks.remainder() {
                total += x * x;
            }
            total
        }
    }

    /// Horizontal sum of eight lanes: fold 256→128, then 128→scalar.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn hsum_ps(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;

        // SAFETY: pure register shuffles and adds.
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let total = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(total)
    }
}

// ---------------------------------------------------------------------------
// aarch64: NEON (mandatory on the architecture, so no runtime check needed)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", not(target_arch = "wasm32")))]
mod imp {
    use core::arch::aarch64::*;

    #[inline]
    fn hadd(v: float32x4_t) -> f32 {
        // SAFETY: pure register arithmetic, no memory access.
        unsafe {
            let pair = vpaddq_f32(v, v);
            let quad = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(quad)
        }
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let mut acc1 = unsafe { vdupq_n_f32(0.0) };
        let mut acc2 = unsafe { vdupq_n_f32(0.0) };
        let mut acc3 = unsafe { vdupq_n_f32(0.0) };

        let mut chunks = a[..len].chunks_exact(16);
        let mut b_chunks = b[..len].chunks_exact(16);

        for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
            // SAFETY: 16 f32 = 64 bytes available in each chunk.
            unsafe {
                acc0 = vfmaq_f32(acc0, vld1q_f32(x.as_ptr()), vld1q_f32(y.as_ptr()));
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(x.as_ptr().add(4)),
                    vld1q_f32(y.as_ptr().add(4)),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(x.as_ptr().add(8)),
                    vld1q_f32(y.as_ptr().add(8)),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(x.as_ptr().add(12)),
                    vld1q_f32(y.as_ptr().add(12)),
                );
            }
        }

        // SAFETY: register arithmetic only.
        let mut total = unsafe { hadd(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3))) };
        for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
            total += x * y;
        }
        total
    }

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let mut acc1 = unsafe { vdupq_n_f32(0.0) };

        let mut chunks = a[..len].chunks_exact(8);
        let mut b_chunks = b[..len].chunks_exact(8);

        for (x, y) in chunks.by_ref().zip(b_chunks.by_ref()) {
            // SAFETY: 8 f32 = 32 bytes available in each chunk.
            unsafe {
                let d0 = vsubq_f32(vld1q_f32(x.as_ptr()), vld1q_f32(y.as_ptr()));
                acc0 = vfmaq_f32(acc0, d0, d0);
                let d1 = vsubq_f32(vld1q_f32(x.as_ptr().add(4)), vld1q_f32(y.as_ptr().add(4)));
                acc1 = vfmaq_f32(acc1, d1, d1);
            }
        }

        // SAFETY: register arithmetic only.
        let mut total = unsafe { hadd(vaddq_f32(acc0, acc1)) };
        for (x, y) in chunks.remainder().iter().zip(b_chunks.remainder()) {
            let d = x - y;
            total += d * d;
        }
        total
    }

    #[inline]
    pub fn norm_sq(a: &[f32]) -> f32 {
        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let mut acc1 = unsafe { vdupq_n_f32(0.0) };

        let mut chunks = a.chunks_exact(8);
        for x in chunks.by_ref() {
            // SAFETY: 8 f32 = 32 bytes available in this chunk.
            unsafe {
                let v0 = vld1q_f32(x.as_ptr());
                acc0 = vfmaq_f32(acc0, v0, v0);
                let v1 = vld1q_f32(x.as_ptr().add(4));
                acc1 = vfmaq_f32(acc1, v1, v1);
            }
        }

        // SAFETY: register arithmetic only.
        let mut total = unsafe { hadd(vaddq_f32(acc0, acc1)) };
        for &x in chunks.remainder() {
            total += x * x;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Generic scalar fallback (covers wasm32 without simd128, and everything else)
// ---------------------------------------------------------------------------

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    all(target_arch = "x86_64", not(target_arch = "wasm32")),
    all(target_arch = "aarch64", not(target_arch = "wasm32")),
)))]
mod imp {
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        super::scalar_dot(a, b)
    }

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        super::scalar_l2_sq(a, b)
    }

    #[inline]
    pub fn norm_sq(a: &[f32]) -> f32 {
        super::scalar_norm_sq(a)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Dot product of `a` and `b`, stopping at the shorter of the two.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    imp::dot(a, b)
}

/// Squared Euclidean distance, `sum((a_i - b_i)^2)`.
///
/// Used instead of the old `|a|^2 + |b|^2 - 2·a·b` expansion. That form has
/// two problems: it loses precision through cancellation when `a ≈ b` (exactly
/// the case that matters for a nearest-neighbour search), and it overflows to
/// `inf`/`NaN` for large-magnitude components such as `f32::MAX`. The direct
/// form costs one extra subtract per element and is immune to both.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    imp::l2_sq(a, b)
}

/// `sum(v_i^2)`, i.e. the squared L2 norm.
#[inline]
pub fn norm_sq(v: &[f32]) -> f32 {
    imp::norm_sq(v)
}

/// Euclidean distance, `sqrt(l2_sq(a, b))`.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_sq(a, b).max(0.0).sqrt()
}

/// All three metrics in one call. `cache_a`/`cache_b` are the precomputed
/// `norm_sq` values from [`crate::engine::types::VectorStore::cache_attr`].
///
/// Returns the **score** to be minimised: smaller is closer, for every metric.
/// `DotProduct` is negated so a single ordering comparison works everywhere.
#[inline]
pub fn score(
    distance: crate::engine::types::Distance,
    a: &[f32],
    b: &[f32],
    cache_a: f32,
    cache_b: f32,
) -> f32 {
    match distance {
        crate::engine::types::Distance::Euclidean => l2(a, b),
        crate::engine::types::Distance::DotProduct => -dot(a, b),
        crate::engine::types::Distance::Cosine => {
            // Cache holds the squared norm, so the magnitude is its sqrt.
            let denom = cache_a.max(0.0).sqrt() * cache_b.max(0.0).sqrt();
            if denom <= f32::MIN_POSITIVE {
                // A zero vector has no direction; treat everything as maximally
                // distant rather than producing a NaN that would poison the
                // ordering.
                1.0
            } else {
                (1.0 - dot(a, b) / denom).clamp(-1.0, 2.0)
            }
        }
    }
}

/// Scale `v` to unit length in place. A zero (or subnormal) vector is left
/// untouched rather than being turned into NaNs.
pub fn normalize_in_place(v: &mut [f32]) {
    let magnitude = norm_sq(v).max(0.0).sqrt();
    if magnitude > f32::MIN_POSITIVE {
        let inv = 1.0 / magnitude;
        for value in v.iter_mut() {
            *value *= inv;
        }
    }
}

/// Unit-length copy of `v`. Prefer [`normalize_in_place`] when you own the
/// buffer — this allocates.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize_in_place(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so the tests do not need `getrandom` on wasm.
    fn pseudo(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Every backend must agree with the scalar reference to within f32
    /// rounding. The tolerance scales with `len` because summation order
    /// differs; relative error grows like `sqrt(len) * eps`.
    fn assert_close(got: f32, want: f32, len: usize) {
        let tolerance = (len as f32).sqrt() * f32::EPSILON * want.abs().max(1.0) * 8.0;
        assert!(
            (got - want).abs() <= tolerance,
            "got {got}, want {want} (tol {tolerance})"
        );
    }

    #[test]
    fn matches_scalar_across_lengths() {
        // Lengths chosen to exercise every remainder path: 0..3 (no vector),
        // 4..7, 8..15, 16..31, and the aligned cases.
        for len in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 1024,
        ] {
            let a = pseudo(0x1234_5678, len);
            let b = pseudo(0x9ABC_DEF0, len);

            assert_close(dot(&a, &b), scalar_dot(&a, &b), len.max(1));
            assert_close(l2_sq(&a, &b), scalar_l2_sq(&a, &b), len.max(1));
            assert_close(norm_sq(&a), scalar_norm_sq(&a), len.max(1));
        }
    }

    #[test]
    fn self_consistency() {
        let a = pseudo(7, 128);
        // l2_sq(v, v) == 0 exactly: the same subtraction in the same lanes
        // yields exact zero, which is the property the old expansion lost.
        assert_eq!(l2_sq(&a, &a), 0.0);
        assert_close(dot(&a, &a), norm_sq(&a), 128);
    }

    #[test]
    fn handles_extremes_without_nan() {
        let max = vec![f32::MAX; 16];
        let min = vec![f32::MIN; 16];
        let zero = vec![0.0f32; 16];

        // These all overflow to +inf (which is the mathematically right answer
        // for f32) but must never be NaN, which is what the old
        // `|a|^2 + |b|^2 - 2ab` form produced.
        assert!(!l2_sq(&max, &min).is_nan());
        assert_eq!(l2_sq(&max, &max), 0.0);
        assert_eq!(l2_sq(&zero, &zero), 0.0);
        assert!(!dot(&max, &max).is_nan());
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(dot(&[], &[]), 0.0);
        assert_eq!(l2_sq(&[], &[]), 0.0);
        assert_eq!(norm_sq(&[]), 0.0);
        // Mismatched lengths truncate to the shorter one instead of panicking.
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[1.0]), 1.0);
        assert_eq!(l2_sq(&[1.0], &[]), 0.0);
    }

    #[test]
    fn normalize_handles_zero_vector() {
        let mut zero = vec![0.0f32; 8];
        normalize_in_place(&mut zero);
        assert!(zero.iter().all(|v| *v == 0.0));

        let mut v = vec![3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize_in_place(&mut v);
        assert_close(norm_sq(&v), 1.0, 8);
    }

    #[test]
    fn cosine_never_returns_nan() {
        use crate::engine::types::Distance;

        let zero = vec![0.0f32; 8];
        let other = vec![1.0f32; 8];
        let s = score(
            Distance::Cosine,
            &zero,
            &other,
            norm_sq(&zero),
            norm_sq(&other),
        );
        assert_eq!(s, 1.0);

        let s = score(
            Distance::Cosine,
            &other,
            &other,
            norm_sq(&other),
            norm_sq(&other),
        );
        assert_close(s, 0.0, 8);
    }

    #[test]
    fn dot_is_negated_for_ranking() {
        use crate::engine::types::Distance;
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![2.0f32, 0.0, 0.0, 0.0];
        let s = score(Distance::DotProduct, &a, &b, 0.0, 0.0);
        assert_eq!(s, -2.0);
    }
}
