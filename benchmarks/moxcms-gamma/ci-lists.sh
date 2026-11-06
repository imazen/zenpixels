#!/usr/bin/env bash
# Frozen x86 feature list from the experiment's upstream build_push.yml.
set -euo pipefail
cd "${1:?usage: ci-lists.sh MOXCMS_CHECKOUT LOG_DIRECTORY}"
logs=${2:?supply an output directory}
mkdir -p "$logs"
features=(none avx_shaper_paths avx_shaper_optimized_paths avx_shaper_fixed_point_paths avx_luts avx_shaper_fixed_point_paths,in_place sse_shaper_paths sse_shaper_optimized_paths sse_shaper_fixed_point_paths sse_luts sse_shaper_fixed_point_paths,in_place extended_range)
for feature in "${features[@]}"; do
  args=(--features "$feature")
  if [[ "$feature" == none ]]; then args=(--no-default-features); fi
  cargo test "${args[@]}" 2>&1 | tee "$logs/test-${feature//,/-}.log"
  cargo clippy "${args[@]}" -- -D warnings 2>&1 | tee "$logs/clippy-${feature//,/-}.log"
done
cargo clippy --features any_to_any -- -D warnings 2>&1 | tee "$logs/clippy-any_to_any.log"
cargo fmt --check -p moxcms
