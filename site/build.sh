#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/build --remap-path-prefix=$PWD=/source"
cargo build --manifest-path site/converter-wasm/Cargo.toml --locked --release --target wasm32-unknown-unknown
wasm-bindgen --version | grep -Fx 'wasm-bindgen 0.2.130'
wasm-bindgen "${CARGO_TARGET_DIR:-site/converter-wasm/target}/wasm32-unknown-unknown/release/pixel_explorer_converter.wasm" --target web --out-dir site/wasm --out-name converter
mkdir -p site/dist/color-explorer
cp site/index.html site/review.html site/converter.js site/dist/
for doc in docs/guide.md docs/speed-and-size.md docs/eli5.md; do
  rustdoc --edition=2024 "$doc" -o site/dist --markdown-no-toc --markdown-css guide.css --html-in-header site/guide-head.html --html-before-content site/guide-nav.html
done
cp site/guide.css site/dist/
cp -R site/wasm site/dist/
cp docs/color-explorer/{index.html,model.js,view.js} site/dist/color-explorer/
cp docs/format_browser.html site/dist/
# Rewrite sibling .md links between rendered teaching pages to .html, set
# per-page meta descriptions, and point source-only links at GitHub.
python3 - <<'PY'
from pathlib import Path
import json, re, subprocess
from datetime import datetime, timezone

Path('site/dist/build-info.json').write_text(json.dumps({
    'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
    'builtAt': datetime.now(timezone.utc).isoformat(),
}))

pages = {
    'guide.html': 'A practical guide to owning images, borrowing cropped views, describing pixel samples, and converting pixels with zenpixels.',
    'speed-and-size.html': 'Balance SIMD speed and light compilation in zenpixels-convert using the universal scalar core, accel presets, const Accelerators helpers, and Cargo features.',
    'eli5.html': 'Plain-English mental model for zenpixels, zenpixels-convert, linear-srgb, and garb, covering core types, opt-in SIMD, FloatFacts, Oklab, and HDR gotchas.',
}
md_link = re.compile(r'href="(guide|speed-and-size|eli5)\.md(#[^"]*)?"')
desc_meta = re.compile(r'<meta name="description" content="[^"]*">')
for name, desc in pages.items():
    p = Path('site/dist') / name
    html = p.read_text()
    html = md_link.sub(lambda m: f'href="{m.group(1)}.html{m.group(2) or ""}"', html)
    html = desc_meta.sub(f'<meta name="description" content="{desc}">', html, count=1)
    p.write_text(html)

p = Path('site/dist/color-explorer/index.html')
p.write_text(p.read_text().replace('../finalization-release-contract.md', 'https://github.com/imazen/zenpixels/blob/main/docs/finalization-release-contract.md'))
PY
