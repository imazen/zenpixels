#!/usr/bin/env bash
# Fetch the C++ Simd library (ermig1979/Simd) for the cpp-simd feature.
# Kept out of git (third_party/ is gitignored); the exact commit is recorded
# in third_party/SIMD_COMMIT.txt and echoed so runs can cite it.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/Simd/.git ]; then
  git clone --depth 1 https://github.com/ermig1979/Simd third_party/Simd
fi
git -C third_party/Simd rev-parse HEAD | tee third_party/SIMD_COMMIT.txt
