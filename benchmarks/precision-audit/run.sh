#!/bin/bash
# Builds outside the repo, once per version pair, since two 0.6.x from one source can't share a
# lockfile. Run through run-heavy. Output goes to stdout.
set -e
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d -p "$HOME/tmp" precision-audit.XXXXXX)
mkdir -p "$work/src"; cp "$here/src/main.rs" "$work/src/"; cd "$work"
gen() { cat > Cargo.toml <<T
[package]
name = "precision-audit"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[dependencies]
linear-srgb = $1
garb = $2
T
rm -f Cargo.lock; }
LS=52896f4b24664315586bd60e5f048253b1f69052
GB=605276d717ea0b31a1cd2a36ae05301383c7b8b5
for pair in "0.6.8 0.2.5" "0.6.12 0.2.8" "git git"; do
  set -- $pair
  if [ "$1" = git ]; then
    # This revision doesn't compile without `transfer`.
    gen "{ git = \"https://github.com/imazen/linear-srgb\", rev = \"$LS\", features = [\"transfer\"] }" \
        "{ git = \"https://github.com/imazen/garb\", rev = \"$GB\", features = [\"experimental\"] }"
  else
    gen "{ version = \"=$1\" }" "{ version = \"=$2\", features = [\"experimental\"] }"
  fi
  echo "=== linear-srgb $1, garb $2"
  cargo run --release --quiet
done
