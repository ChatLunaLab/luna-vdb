#!/usr/bin/env bash
#
# Build the published wasm packages.
#
# Produces two targets, both written under `pkg/`:
#
#   pkg/web      — `--target web`,      ESM,         SIMD128
#   pkg/nodejs   — `--target nodejs`,   CJS,         SIMD128
#   pkg/web-scalar    — `--target web`,    ESM,      no SIMD
#   pkg/nodejs-scalar — `--target nodejs`, CJS,      no SIMD
#
# Two builds, not one, because SIMD128 is not a runtime-checked feature: a
# module compiled with it *fails to instantiate* on a runtime that lacks it.
# Bun before 1.1, older Node on some platforms, and a handful of embedded
# engines still lack it. Shipping only the SIMD build would turn those into an
# instantiation crash with no fallback, so the scalar build is published too and
# `package.json` routes to whichever the environment supports.
#
# This script is meant to run in CI (or on the build host), not on a laptop —
# see `.github/workflows/ci.yml` and `scripts/remote-build.sh`.

set -euo pipefail

cd "$(dirname "$0")/.."

TARGET=wasm32-unknown-unknown
ARTIFACT="target/${TARGET}/release/luna_vdb.wasm"

command -v wasm-bindgen >/dev/null 2>&1 || {
  echo "error: wasm-bindgen CLI not found. Install with:" >&2
  echo "  cargo install wasm-bindgen-cli --version \$(grep -m1 '^wasm-bindgen = ' Cargo.toml | cut -d'\"' -f2)" >&2
  exit 1
}

rustup target list --installed | grep -q "^${TARGET}$" || rustup target add "$TARGET"

# The features every build needs, regardless of SIMD. These are *validation*
# flags, not optimisation ones — they widen the set of features wasm-opt will
# accept in its input. The list mirrors what rustc's `wasm32-unknown-unknown`
# target enables by default (LLVM's `generic` CPU since Rust 1.87), because
# that is what the compiler actually emits; binaryen's own default set is
# narrower and rejects ordinary Rust output:
#
#   * `--enable-bulk-memory` — `memory.copy`/`memory.fill`, which LLVM emits
#     for slice copies.
#   * `--enable-nontrapping-float-to-int` — `i32.trunc_sat_f32_u` and friends.
#     Rust's `as` casts from float to integer saturate, so these show up in
#     plain code; without the flag wasm-opt fails validation with
#     "unexpected false: all used features should be allowed".
#   * `--enable-reference-types` — wasm-bindgen's `externref` shims.
#   * `--enable-multivalue`, `--enable-sign-ext`, `--enable-mutable-globals`
#     — the remainder of rustc's default set.
#
# `--enable-simd` is deliberately NOT in this list. On the *scalar* build it
# would invite wasm-opt to accept exactly the instructions that must not appear
# there, and would undermine the artifact check that guards older runtimes.
COMMON_FEATURES=(
  --enable-bulk-memory
  --enable-nontrapping-float-to-int
  --enable-reference-types
  --enable-multivalue
  --enable-sign-ext
  --enable-mutable-globals
)

# `wasm-opt` shrinks the artifact substantially (typically 25-35%) and is worth
# running whenever it is available; it is optional so a contributor without it
# can still produce a working package.
optimise() {
  local file="$1"
  shift

  if ! command -v wasm-opt >/dev/null 2>&1; then
    echo "    (wasm-opt not found, skipping optimisation)"
    return
  fi

  wasm-opt "$file" -O4 "${COMMON_FEATURES[@]}" "$@" -o "$file.opt"
  mv "$file.opt" "$file"
}


rm -rf pkg
mkdir -p pkg

# ---------------------------------------------------------------------------
# SIMD128 build
# ---------------------------------------------------------------------------
echo "==> building wasm32 + simd128"
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
  cargo build --release --target "$TARGET" --features console_error_panic_hook

cp "$ARTIFACT" pkg/luna_vdb_simd.wasm

for out in web nodejs; do
  # `--target web` gets `--omit-default-module-path`: it makes the generated
  # loader avoid a hard-coded path so Vite/webpack can resolve the .wasm through
  # their own asset pipeline. `--target nodejs` must NOT get it — that target
  # emits a `require('./luna_vdb_bg.wasm')` and relies on the default path.
  extra=""
  [ "$out" = web ] && extra="--omit-default-module-path"

  echo "==> wasm-bindgen --target $out (simd)"
  wasm-bindgen "$ARTIFACT" --out-dir "pkg/$out" --target "$out" $extra
  optimise "pkg/$out/luna_vdb_bg.wasm" --enable-simd
done

# ---------------------------------------------------------------------------
# Scalar build
#
# RUSTFLAGS is set explicitly rather than appended so that a stale `+simd128`
# in the environment cannot leak into the scalar artifact — which would make the
# "no SIMD instructions present" check in CI fail intermittently.
# ---------------------------------------------------------------------------
echo "==> building wasm32 (scalar)"
RUSTFLAGS="-C target-feature=-simd128" \
  cargo build --release --target "$TARGET" --features console_error_panic_hook

cp "$ARTIFACT" pkg/luna_vdb_scalar.wasm

for out in web-scalar nodejs-scalar; do
  base="${out%-scalar}"
  extra=""
  [ "$base" = web ] && extra="--omit-default-module-path"

  echo "==> wasm-bindgen --target $base (scalar) -> pkg/$out"
  wasm-bindgen "$ARTIFACT" --out-dir "pkg/$out" --target "$base" $extra
  optimise "pkg/$out/luna_vdb_bg.wasm"
done

rm -f pkg/luna_vdb_simd.wasm pkg/luna_vdb_scalar.wasm


echo
echo "==> done"
find pkg -type f -name '*.wasm' -exec ls -lh {} \; | awk '{ print "    " $5 "\t" $9 }'
