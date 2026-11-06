#!/usr/bin/env bash
# Run the frozen AArch64 feature list in a supplied dependency checkout.
set -euo pipefail
cd "${1:?usage: arm-run.sh MOXCMS_CHECKOUT LOG_DIRECTORY}"
logs=${2:?supply an output directory}
mkdir -p "$logs"
for features in none neon_shaper_paths neon_shaper_optimized_paths neon_shaper_fixed_point_paths neon_luts neon_shaper_fixed_point_paths,in_place extended_range; do
  args=(--features "$features")
  if [[ "$features" == none ]]; then args=(--no-default-features); fi
  cargo test "${args[@]}" 2>&1 | tee "$logs/test-${features//,/-}.log"
done
