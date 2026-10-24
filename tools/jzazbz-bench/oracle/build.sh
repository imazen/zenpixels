#!/usr/bin/env bash
# Build the libjxl oracle against a libjxl source checkout (headers only) and
# system Highway (libhwy-dev). Static AVX2 target, matching libjxl's AVX2 arm
# (Highway needs AES+PCLMUL for its AVX2 static target; without them it falls
# back to SSSE3, where MulAdd is not fused). -ffp-contract=off keeps the
# compiler from fusing anything libjxl does not fuse.
# Usage: LIBJXL=~/work/jxl-efforts/libjxl tools/jzazbz-bench/oracle/build.sh [OUT]
set -euo pipefail
: "${LIBJXL:?set LIBJXL to a libjxl source checkout}"
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../target/libjxl_oracle}"
mkdir -p "$(dirname "$out")"
g++ -std=c++17 -O2 -march=x86-64-v3 -maes -mpclmul -ffp-contract=off \
  -I"$LIBJXL" -I"$LIBJXL/lib/include" \
  "$here/libjxl_oracle.cc" -lhwy -o "$out"
echo "built $out (libjxl $(git -C "$LIBJXL" rev-parse --short HEAD))"
