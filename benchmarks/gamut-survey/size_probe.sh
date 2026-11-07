#!/usr/bin/env bash
# Standalone .text and ELF bytes for one color through each mapper, against matrix and clip.
set -euo pipefail
cd "$(dirname "$0")"
for mode in clip css ott soft; do
  if [ "$mode" = clip ]; then
    cargo build --release --bin size_probe
  else
    cargo build --release --bin size_probe --features "size-$mode"
  fi
  size -A target/release/size_probe > "target/size-$mode.sections.txt"
  stat -c "$mode %s bytes" target/release/size_probe
done
