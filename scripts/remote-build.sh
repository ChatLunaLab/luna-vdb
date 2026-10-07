#!/usr/bin/env bash
#
# Run the full check suite on the build host instead of locally.
#
#   scripts/remote-build.sh            # cargo test + clippy, native
#   scripts/remote-build.sh wasm       # everything, including wasm32 + simd128
#   scripts/remote-build.sh bench      # run the benchmark and print the table
#   scripts/remote-build.sh shell      # drop into a shell in the container
#
# Why this exists: the toolchain that produces a correct wasm artifact
# (rustup targets, wasm-bindgen CLI, wasm-opt, and the exact RUSTFLAGS) is
# fiddly enough that a laptop and CI drift apart. Rather than debug that drift
# per machine, the source is copied to a host that already has Docker and the
# suite runs in a pinned container there.
#
# The container is `rust:<pinned>` so a toolchain upgrade is an explicit,
# reviewable change to PINNED_IMAGE below rather than whatever `rustup update`
# last did to somebody's laptop.
#
# Nothing is written back to the working tree except `target/`, so this is safe
# to run with uncommitted changes.

set -euo pipefail

cd "$(dirname "$0")/.."

REMOTE="${LUNA_BUILD_HOST:-dingyi@10.1.1.14}"
PINNED_IMAGE="${LUNA_RUST_IMAGE:-rust:1.90-slim}"
REMOTE_DIR="${LUNA_BUILD_DIR:-/tmp/luna-vdb-build}"
MODE="${1:-native}"

# `--exclude` rather than a `.dockerignore`: rsync's exclude syntax is cheaper to
# reason about here, and the copy is the slow part we want to keep minimal.
RSYNC_EXCLUDES=(
  --exclude 'target/' \
  --exclude 'pkg/' \
  --exclude 'node_modules/' \
  --exclude '.git/' \
  --exclude '.yarn/' \
  --exclude '*.wasm'
)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

log "checking $REMOTE is reachable"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" true 2>/dev/null; then
  echo "error: cannot reach $REMOTE over ssh." >&2
  echo "       set LUNA_BUILD_HOST to override, or use GitHub Actions instead." >&2
  exit 1
fi

log "syncing source to $REMOTE:$REMOTE_DIR"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" ./ "$REMOTE:$REMOTE_DIR/"

# Run inside the container. `CARGO_HOME` and `CARGO_TARGET_DIR` point at a
# persistent volume so dependency compilation is not repeated on every
# invocation — that is the difference between ~20 s and ~3 min per run.
#
# ssh joins its arguments with spaces and the remote shell re-parses the
# result, so every argument is quoted with `printf %q` first. Passing them
# bare — as this script used to — split `RUSTFLAGS=-D warnings` into two
# words (docker then tried to pull an image called "warnings"), and every `;`
# in the container command ended the `sh -c` early, running the rest on the
# bare host instead of in the container.
run_in_container() {
  local rflags="$1"
  shift

  local remote
  remote="$(printf '%q ' docker run --rm \
    -v "$REMOTE_DIR:/work" \
    -v "luna-cargo-registry:/usr/local/cargo/registry" \
    -v "luna-cargo-git:/usr/local/cargo/git" \
    -v "luna-target:/work/target" \
    -w /work \
    -e "RUSTFLAGS=$rflags" \
    -e CARGO_TERM_COLOR=always \
    "$PINNED_IMAGE" \
    sh -c "$*")"

  ssh "$REMOTE" "$remote"
}

case "$MODE" in
  native)
    log "cargo test (native) on $PINNED_IMAGE"
    run_in_container "-D warnings" \
      'set -e; rustc -V; cargo -V; cargo test --all-features --verbose'
    ;;

  clippy)
    log "cargo clippy on $PINNED_IMAGE"
    run_in_container "-D warnings" \
      'set -e; rustup component add clippy rustfmt >/dev/null 2>&1; cargo fmt --all -- --check; cargo clippy --all-targets --all-features'
    ;;

  wasm)
    log "wasm32 + simd128 (native tests, then build, then clippy)"
    run_in_container "-D warnings" \
      'set -e; cargo test --all-features'
    run_in_container "-D warnings" \
      'set -e; rustup target add wasm32-unknown-unknown >/dev/null 2>&1; cargo build --release --target wasm32-unknown-unknown --features console_error_panic_hook'
    run_in_container "-D warnings -C target-feature=+simd128" \
      'set -e; rustup target add wasm32-unknown-unknown >/dev/null 2>&1; cargo build --release --target wasm32-unknown-unknown --features console_error_panic_hook'
    echo
    log "note: browser and node wasm tests need wasm-pack; run them in CI"
    ;;

  bench)
    log "cargo bench on $PINNED_IMAGE"
    # Output goes to a file first so a failed build exits non-zero instead of
    # being masked by `tail` at the end of a pipe.
    run_in_container "-D warnings" \
      'set -e; cargo bench --bench search > /tmp/bench.txt 2>&1 || { tail -60 /tmp/bench.txt; exit 1; }; tail -60 /tmp/bench.txt'
    ;;

  shell)
    log "opening a shell in $PINNED_IMAGE"
    ssh -t "$REMOTE" "$(printf '%q ' docker run --rm -it \
      -v "$REMOTE_DIR:/work" \
      -v "luna-cargo-registry:/usr/local/cargo/registry" \
      -v "luna-target:/work/target" \
      -w /work \
      "$PINNED_IMAGE" \
      sh)"
    ;;

  *)
    echo "usage: $0 [native|clippy|wasm|bench|shell]" >&2
    exit 2
    ;;
esac

log "done"
