#!/usr/bin/env bash
# Structural layout probe: size_of/align_of of zenpixels pixel views at a base
# revision and in this working tree. Not a heap or RSS measurement.
# Usage (from a git clone): benchmarks/layout-probe/run.sh [BASE_REV]
# Results of the 2026-10-28 run: benchmarks/layout_2026-10-28.md.
set -euo pipefail
base_rev="${1:-86d0e2d30406a006bf8332f142029b7f58353086}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
work="${TMPDIR:-$HOME/tmp}/zp-layout-probe"
case "$work" in */zp-layout-probe) rm -rf -- "$work" ;; *) exit 2 ;; esac
mkdir -p "$work/base" "$work/probe-base/src" "$work/probe-head/src"
git -C "$repo" archive "$base_rev" | tar -x -C "$work/base"
manifest() { # probe dir, zenpixels crate path
  cat > "$work/$1/Cargo.toml" <<TOML
[package]
name = "zp-layout-$1"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
zenpixels = { path = "$2" }

[workspace]
TOML
}
manifest probe-base "$work/base/zenpixels"
manifest probe-head "$repo/zenpixels"
cp "$repo/benchmarks/layout-probe/base.rs" "$work/probe-base/src/main.rs"
cp "$repo/benchmarks/layout-probe/head.rs" "$work/probe-head/src/main.rs"
cp "$work/base/Cargo.lock" "$work/probe-base/Cargo.lock"
cp "$repo/Cargo.lock" "$work/probe-head/Cargo.lock"
export CARGO_TARGET_DIR="$work/target"
echo "== base $base_rev"
cargo run -q --manifest-path "$work/probe-base/Cargo.toml"
echo "== head (this working tree)"
cargo run -q --manifest-path "$work/probe-head/Cargo.toml"
