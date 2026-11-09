# Pixel Format Explorer

The [published explorer](https://imazen.github.io/zenpixels/) explains sample storage,
color metadata, row padding and conversions. Its optional WASM panel runs this
checkout's `zenpixels-convert::RowConverter` on the current stored pixels. All 54
formats appear at both ends; unsupported routes report the crate's refusal.
CMYK has no supplied ICC profile. The model's CMYK/native conventions remain
illustrations. Exact byte disagreements are shown, including floating-point
approximations and the model's selectable PQ anchor versus the converter's separately supplied
SDR-white anchor. X padding is excluded from comparison.

The existing [color decision model](../docs/color-explorer/index.html) and
[format recommendation browser](../docs/format_browser.html) are copied into the
Pages artifact and linked from the explorer. Their canonical source stays in docs.

Install Rust **1.99.1** with `wasm32-unknown-unknown`, plus
`wasm-bindgen-cli` **0.2.130**, then run `just site-build`. The unpublished boundary
crate is an independent workspace with its own lockfile and the same pinned
`garb`/`linear-srgb` patches as the root. `wasm-bindgen` exports compile under
`#![forbid(unsafe_code)]`. No core public API was added. Source paths are remapped
for the browser build; generated WASM and bindings are ignored and built by CI.

Serve `site/dist` over HTTP. With Playwright **1.63.0** and Chromium installed,
run `PLAYWRIGHT_MODULE=/path/to/node_modules/playwright just site-check <URL>`.
The browser check covers 108 format/alignment combinations, all 2,916 format
pairs, a byte-checked BGR swizzle, strided 16-bit input, the PQ discrepancy,
CMYK refusal, mobile controls and missing-WASM fallback. On the initial 0.3.1
build, 512 pairs converted and 2,404 returned errors; this measures route
availability, not the correctness of every successful route. The existing
JavaScript model remains usable without the converter assets.

`.github/workflows/pages.yml` checks the boundary, builds the release WASM,
runs the browser checks, and deploys `site/dist` through GitHub Pages. Pull
requests build and check the site without deploying it. The footer identifies the
built revision, local-change status and build time so a stale deployment is visible.

On the HDR intent candidate, the row simulator exposes optional SDR white, HDR
source peak and SDR target peak in nits. Blank inputs preserve the converter's
typed missing-input errors. HLG display rendering now accepts explicit peak, black and effective gamma,
plus SDR white. Missing/invalid policy still refuses. The JavaScript bare-OETF
model is not used as a display-rendering byte oracle.

The live storage panel exercises `PlaneRole::CbCr`/`CrCb`, interleaved
`CodePlane::component`, RGB10 + alpha1 through both current views, ignored sample
padding, final-row extents, odd byte offsets, signed I64 and exact U64 codes.
It exposes overlap rather than treating NativeCodeSlice as necessary for every
interleaved native representation. See the [design review and outstanding
issues](../docs/explorer-design-review.md). These are code extraction examples,
not range expansion, chroma reconstruction or color conversion.
