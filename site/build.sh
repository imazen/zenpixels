#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/build --remap-path-prefix=$PWD=/source"
cargo build --manifest-path site/converter-wasm/Cargo.toml --locked --release --target wasm32-unknown-unknown
wasm-bindgen --version | grep -Fx 'wasm-bindgen 0.2.130'
wasm-bindgen "${CARGO_TARGET_DIR:-site/converter-wasm/target}/wasm32-unknown-unknown/release/pixel_explorer_converter.wasm" --target web --out-dir site/wasm --out-name converter
mkdir -p site/dist/color-explorer
cp site/index.html site/converter.js site/dist/
cp -R site/wasm site/dist/
cp docs/color-explorer/{index.html,model.js,view.js} site/dist/color-explorer/
cp docs/format_browser.html site/dist/
# The teaching page's source-only Markdown link points at the repository.
python3 - <<'PY'
from pathlib import Path
p = Path('site/dist/color-explorer/index.html')
p.write_text(p.read_text().replace('../finalization-release-contract.md', 'https://github.com/imazen/zenpixels/blob/main/docs/finalization-release-contract.md'))
PY
