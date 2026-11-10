# Pixel Format Explorer and guide

The [published explorer](https://imazen.github.io/zenpixels/) lets you inspect
samples, channels, byte order, row stride and actual conversion routes.
The [pixel types guide](https://imazen.github.io/zenpixels/guide.html) starts with
an owned image, a borrowed image/subimage view and a descriptor. Its canonical
source is [docs/guide.md](../docs/guide.md); rustdoc renders the HTML, so examples
and the published guide share one source.

## Build

Use Rust **1.99.1** with `wasm32-unknown-unknown` and `wasm-bindgen-cli`
**0.2.130**, then run `just site-build`. Serve `site/dist` over HTTP.
The independent, unpublished WASM boundary uses this checkout's core/converter
and the root's numerical dependency revisions. Generated bindings and WASM are
ignored. No core public API is added.

The build stamp identifies source revision, local changes and build time.
The browser compares its format facts with a catalog constructed through the
actual Rust format APIs before enabling the converter. A mismatched model or
stale binding fails the check rather than silently mixing versions.

## What is live, and what is illustrative

The converter panel runs `RowConverter` on the current stored samples. The
storage panel executes CodePlane readers for CbCr/CrCb, RGB10 + alpha1 and
RGB565. RGB565 uses three bitfield views of shared U16 words and converts
straight into canonical RGB8. Sample padding and final-row extents are visible.

JavaScript color calculations are teaching illustrations. Exact byte agreement
or differences are displayed without treating that model as a numerical oracle.
CMYK has no supplied ICC; native signed/wide scaling is illustrative. General
signed/wide conversion still refuses. HLG display rendering requires explicit
conditions, and is not compared with the bare scene-transfer model as though
they were the same operation. X padding is excluded from byte comparisons.

The [color decision model](../docs/color-explorer/index.html) separates unresolved
source claims from selected current color. The
[historical recommendation browser](../docs/format_browser.html) is labeled as an
illustration, not a current converter support table. Both are copied into the
Pages artifact. The dated review page links current issue tracking.

## Checks

With Playwright **1.63.0** and Chromium installed:

```sh
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright just site-check URL
just guide-check
```

Set `EXPLORER_RUST_EXAMPLES` to an output Markdown path when running the browser
check, then pass that path to `just guide-check PATH`. The latter compiles and
runs the actual copied Rust snippets against the local crate artifacts, as
well as the guide and both READMEs. `just readme-sync` regenerates the crates.io
copies using the shared zenutils generator.

The browser gate checks every displayed source/destination pair for a result
or explicit refusal, plus concrete BGR swizzle and strided U16 controls.
It checks source/catalog agreement, native extraction/padding, RGB565,
explicit HDR conditions, missing-WASM fallback, Unknown declarations and generic
alpha in copied code. Route availability is not proof of every route's pixels.

Open panels and wide native words must not cause document overflow at
320/360/390/768/1024/1440 pixels. Guide, color model, review and recommendation
pages also have viewport checks. Wide tables/diagrams retain local scrolling.
`EXPLORER_SCREENSHOTS` optionally names a fresh capture directory; five bounded
screenshots are saved for visual review, not committed.

`.github/workflows/pages.yml` builds and checks the WASM boundary, renders the
guide, runs browser checks and compiles all exported Rust examples. Main builds
deploy through GitHub Pages; PR builds validate without deployment. This is
independent of crate publication or downstream codec qualification.
