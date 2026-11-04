#!/bin/bash
# Bytes RowConverter adds to a stripped binary, on this tree. Needs a wasm32-unknown-unknown target.
# ARM: run on an aarch64 host; `size -A` section sums there, because strip rounds to 64 KB pages.
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/tmp/converter-size-target}"
export CARGO_PROFILE_RELEASE_STRIP=true
added() { # label, target or "", opt-level, cargo args...
  local label=$1 target=$2 opt=$3; shift 3
  local dir="$CARGO_TARGET_DIR/release" ext=""
  local args=(--release -q --bins)
  if [ -n "$target" ]; then args+=(--target "$target"); dir="$CARGO_TARGET_DIR/$target/release"; fi
  [[ "$target" == wasm32* ]] && ext=.wasm
  CARGO_PROFILE_RELEASE_OPT_LEVEL=$opt cargo build "${args[@]}" "$@"
  local base conv; base=$(stat -c %s "$dir/base$ext"); conv=$(stat -c %s "$dir/conv$ext")
  printf '%s: base %s conv %s added %s\n' "$label" "$base" "$conv" "$((conv - base))"
}
added "$(uname -m) opt-level 3" "" 3
added "$(uname -m) opt-level s" "" s
added "$(uname -m) opt-level 3, avx512" "" 3 --features zenpixels-convert/avx512
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 opt-level 3, simd128" wasm32-unknown-unknown 3
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 opt-level z, simd128" wasm32-unknown-unknown z
