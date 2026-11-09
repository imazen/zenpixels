#!/usr/bin/env bash
set -euo pipefail
# Run through run-heavy. Source and output are explicit; no network or overwrites.
source_dir=${1:?libultrahdr source directory}
output_dir=${2:?fresh output directory}
mkdir "$output_dir"
cmake -S "$source_dir" -B "$output_dir/build" -DUHDR_BUILD_EXAMPLES=OFF -DUHDR_ENABLE_HEIF=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build "$output_dir/build" -j "${CARGO_BUILD_JOBS:-3}"
c++ -std=c++17 -O2 -I"$source_dir/lib/include" -I"$source_dir" tools/p3-export-oracle/check.cc "$output_dir/build/libuhdr.a" -ljpeg -llcms2 -lpthread -o "$output_dir/check"
cargo run -p zenpixels-convert --example export_p3 -- "$output_dir/default"
cargo run --manifest-path tools/icc-collapse-no-db/Cargo.toml --example export_p3 -- "$output_dir/isolated"
for mode in default isolated; do
    "$output_dir/check" "$output_dir/$mode/p3-13.icc" "$output_dir/$mode/p3-1.icc" "$output_dir/$mode/compat.icc"
    sha256sum "$output_dir/$mode/"*.icc
done
cmp "$output_dir/default/p3-13.icc" "$output_dir/isolated/p3-13.icc"
cmp "$output_dir/default/p3-1.icc" "$output_dir/isolated/p3-1.icc"
