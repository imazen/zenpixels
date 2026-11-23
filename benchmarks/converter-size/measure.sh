#!/bin/bash
# Bytes RowPlan adds to a stripped binary across Accelerator bundles on this tree.
# Needs a wasm32-unknown-unknown target.
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
  local base conv lay web fast
  base=$(stat -c %s "$dir/base$ext")
  conv=$(stat -c %s "$dir/conv$ext")
  lay=$(stat -c %s "$dir/conv_layout$ext")
  web=$(stat -c %s "$dir/conv_web_sdr$ext")
  fast=$(stat -c %s "$dir/conv_fast$ext")
  printf '%-28s base=%7s | SCALAR=+%7s | LAYOUT=+%7s | WEB_SDR=+%7s | ALL=+%7s\n' \
    "$label" "$base" "$((conv - base))" "$((lay - base))" "$((web - base))" "$((fast - base))"
}

echo "=== Lean (no default features: core + rgb) ==="
added "$(uname -m) -O3" "" 3
added "$(uname -m) -Os" "" s
added "$(uname -m) -O3 +avx512" "" 3 --features zenpixels-convert/avx512
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 -O3 +simd128" wasm32-unknown-unknown 3
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 -Oz +simd128" wasm32-unknown-unknown z

echo "=== Default features (+std,+fast-transpose,+oklab,+hdr-transfers) ==="
added "$(uname -m) -O3 (def)" "" 3 --features zenpixels-convert/default
added "$(uname -m) -Os (def)" "" s --features zenpixels-convert/default
added "$(uname -m) -O3 +avx512 (def)" "" 3 --features zenpixels-convert/default,zenpixels-convert/avx512
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 -O3 +simd128 (def)" wasm32-unknown-unknown 3 --features zenpixels-convert/oklab,zenpixels-convert/hdr-transfers,zenpixels-convert/fast-transpose
RUSTFLAGS='-C target-feature=+simd128' added "wasm32 -Oz +simd128 (def)" wasm32-unknown-unknown z --features zenpixels-convert/oklab,zenpixels-convert/hdr-transfers,zenpixels-convert/fast-transpose
