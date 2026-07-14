# API ergonomics findings — 2026-07-14

How to make `zenpixels` / `zenpixels-convert` call sites **shorter and cleaner**,
grounded in two inputs:

1. A downstream-usage audit across the `~/work/zen/` tree (~20 sibling crates:
   zenjpeg, zenpng, zenwebp, zengif, zenavif, zencodec, zencodecs, zenpipe,
   imageflow, zenmetrics, zenanalyze, zentone, …). Consumer counts below are
   from that audit.
2. Two new tested example galleries that exercise the common usages and expose
   the friction inline:
   - `zenpixels/examples/common_usage.rs` (9 scenarios)
   - `zenpixels-convert/examples/convert_pipeline.rs` (14 scenarios)

**Nothing here changes the public API.** Every public item in these crates is a
forever commitment (see `zenpixels/CLAUDE.md`), so the additive helpers below are
*proposals with real consumers*, not applied changes — they need explicit
sign-off before landing. Findings 6 and this doc are the only things landed now,
alongside the galleries.

## Summary

| # | Finding | Real consumers | Change | Semver |
|---|---------|---------------:|--------|--------|
| 1 | `PixelSlice::new_tight` / `PixelSliceMut::new_tight` (packed-stride ctor) | ~361 | add method | additive (0.2.x) |
| 2 | `Adapted::as_pixel_slice()` | 5 | add method | additive (0.2.x) |
| 3 | `RowConverter::convert_slice(PixelSlice) -> PixelBuffer` | 4–5 | add method | additive (0.2.x) |
| 4 | `PixelDescriptor::with_color_from_cicp(Cicp)` | 3+ | add method | additive (0.2.x) |
| 5 | Steer bare-`Vec` callers to `convert_to` | — | docs | none |
| 6 | Crate-doc examples are `rust,ignore` (untested) | — | docs | none |
| 7 | `finalize_for_output*` / `EncodeReady` / `OutputProfile` unadopted | **0** | demote | **0.3.0 breaking** |
| 8 | `PixelBufferConvertTypedExt` (`.to_rgba8`) is top idiom but `rgb`-gated | ~93 | docs | none |
| 9 | `Cicp::from_bytes([u8; 4])` | ~161 `Cicp::new` | add method | additive (0.2.x) |

---

## 1. Packed-stride slice constructor — the single biggest win

**~361 `PixelSlice::new(...)` call sites**, and nearly every one hand-computes a
tight stride first. It is the most-repeated single line in the whole audit.

```rust
// today — the stride restates width * bpp every time
let stride = width as usize * descriptor.bytes_per_pixel();
let ps = PixelSlice::new(data, width, height, stride, descriptor)?;

// proposed
let ps = PixelSlice::new_tight(data, width, height, descriptor)?;
```

Additive inherent methods on the erased slices (mirrors the existing `new`):

```rust
impl<'a> PixelSlice<'a> {
    /// `new` with a tightly-packed stride (`width * bytes_per_pixel`).
    pub fn new_tight(data: &'a [u8], width: u32, rows: u32, d: PixelDescriptor)
        -> Result<Self, At<BufferError>>
    {
        Self::new(data, width, rows, width as usize * d.bytes_per_pixel(), d)
    }
}
// ...and the same for PixelSliceMut.
```

Representative victims: `imageflow_core/.../zen_encoder.rs:618`,
`zengif/src/codec.rs:1166`, and the 5× `adapt_for_encode` blocks below.
Verdict: **highest value, lowest risk.** 361 concrete consumers clear the YAGNI
bar by a wide margin.

## 2. `Adapted::as_pixel_slice()` — kill the re-slice boilerplate

The `adapt_for_encode` → recompute-stride → `PixelSlice::new` → `encode` block is
**duplicated verbatim 5×** in zenpipe/zencodecs (`avif_enc.rs:159`,
`jxl_enc.rs:130`, `encode.rs:824`, `dispatch.rs:171`, `transcode.rs:706`).
`Adapted` already owns `data` / `descriptor` / `width` / `rows`.

```rust
// today
let adapted = adapt_for_encode(pixel_data, descriptor, w, h, stride, caps)?;
let adapted_stride = adapted.width as usize * adapted.descriptor.bytes_per_pixel();
let slice = PixelSlice::new(&adapted.data, adapted.width, adapted.rows, adapted_stride, adapted.descriptor)?;
encoder.encode(slice)?;

// proposed
let adapted = adapt_for_encode(pixel_data, descriptor, w, h, stride, caps)?;
encoder.encode(adapted.as_pixel_slice())?;   // infallible: Adapted invariants already hold
```

Additive method on `Adapted<'a>` returning `PixelSlice<'a>` over its own bytes.
Shown in `convert_pipeline.rs::adapt_pixels_before_encoding` (with the verbose
form flagged).

## 3. `RowConverter::convert_slice` — bundle the 6-arg row loop

The "build plan, `alloc w*h*bpp`, loop `convert_row`" block is hand-rolled in
~5 crates (`zenmetrics-api/src/metric.rs:2164`, `zenmetrics-gpu-core/src/lib.rs:79,288`,
`cvvdp/src/pipeline.rs:2367`, `zenanalyze/src/row_stream.rs`). `RowConverter::convert_rows`
already does whole-buffer work but takes six positional args
`(src, src_stride, dst, dst_stride, width, rows)` — the very unbundling
`PixelSlice` exists to prevent.

```rust
// today (free-function plan variant)
let plan = ConvertPlan::new(src.descriptor(), target)?;
let mut out = vec![0u8; row_bytes * h as usize];
for y in 0..h { convert_row(&plan, src.row(y), &mut out[y*row_bytes..][..row_bytes], w); }

// proposed
let out: PixelBuffer = RowConverter::new(src.descriptor(), target)?.convert_slice(src.as_slice())?;
```

Additive method taking a `PixelSlice` and returning an owned `PixelBuffer`
(pairs with finding 1). Aligns with the CLAUDE.md pixel-buffer-API rule
("anywhere a function takes the whole image it must accept stride / a `PixelSlice`").
`PixelBufferConvertExt::convert_to` already covers the "I hold a `PixelBuffer`"
case; this covers "I hold a `PixelSlice` and want scratch/streaming control."

## 4. `PixelDescriptor::with_color_from_cicp(Cicp)` — retag transfer+primaries

The "keep the format, adopt transfer+primaries from a decoded CICP" chain repeats
(`zenpng/src/codec.rs:2063`, `zenavif/src/codec.rs:3276`, `zencodec/src/helpers/icc.rs:135`):

```rust
// today
if let Some(tf) = TransferFunction::from_cicp(c.transfer_characteristics) { desc = desc.with_transfer(tf); }
if let Some(p)  = ColorPrimaries::from_cicp(c.color_primaries)          { desc = desc.with_primaries(p); }

// proposed (const fn, keeps format/type/alpha, updates only color axes)
let desc = desc.with_color_from_cicp(cicp);
```

Additive `const fn` on `PixelDescriptor`. Distinct from `Cicp::to_descriptor`,
which *builds* a fresh descriptor from a `PixelFormat` rather than retagging one.

## 5. `convert_buffer` hands back a bare `Vec<u8>`

`adapt::convert_buffer(bytes, w, h, from, to) -> Vec<u8>` drops dims and
descriptor — the caller has to carry them separately. When a `PixelBuffer` is in
hand, `PixelBufferConvertExt::convert_to` keeps everything. This is a
documentation/steering fix (make `convert_to` the headline in the crate docs and
in `convert_buffer`'s rustdoc), not an API change. Demonstrated side-by-side in
`convert_pipeline.rs`.

## 6. Crate-doc examples are `rust,ignore` (untested) — LANDED

Both crate-level docs narrate the pipeline with `rust,ignore` blocks
(`zenpixels-convert/src/lib.rs:81,150,198,207`, …). `ignore` blocks never
compile, so they rot silently. The new tested galleries replace them as the
source of truth; the crate docs now point at the galleries (and the illustrative
`ignore` blocks that use pseudo-symbols like `my_codec_decode` are labeled as
such). Follow-up: convert the self-contained ones to real doctests.

## 7. Unadopted encode-finalization surface → 0.3.0 demotion candidate

`finalize_for_output`, `finalize_for_output_with`, `EncodeReady`,
`OutputProfile`, `OutputMetadata` are public and documented as the "encode" step,
but the audit found **zero** non-def/non-doc downstream call sites. The encode
color role is filled by `adapt_for_encode` + zencodec's `resolve_color_emit`.
Per YAGNI these should be `pub(crate)`, but they shipped in 0.2.14, so removal is
breaking. Recommendation: add to `CHANGELOG.md` **QUEUED BREAKING CHANGES** and
demote at 0.3.0 (do not ship piecemeal). Same status as `cms_lite::ZenCmsLite`.

## 8. Top convert idiom is `rgb`-gated and under-documented

`PixelBufferConvertTypedExt::to_rgba8()` / `.to_rgb8()` is the most common
decode-output one-liner downstream (~93 fully-qualified imports), but it lives
behind the `rgb` feature and is absent from the default-features docs. The
default-available equivalent is `PixelBufferConvertExt::convert_to(desc)`.
Recommendation: document both prominently and cross-link the feature gate. (No
API change — the galleries use `convert_to` since they build with default
features.)

## 9. (minor) `Cicp::from_bytes([u8; 4])`

`Cicp::new(c[0], c[1], c[2], c[3] != 0)` from a parsed 4-byte tuple recurs
(~161 `Cicp::new` sites, many of this shape, e.g. `zenpng/src/codec.rs:2152`).
An additive `Cicp::from_bytes([u8; 4])` removes the `!= 0` papercut. Low priority.

---

## What landed vs. what needs approval

- **Landed** (safe, additive/docs): the two tested example galleries, the CI
  `--examples` step, and this document.
- **Needs sign-off** (public API is forever): findings 1–4, 8–9 are additive and
  each has concrete current consumers — ready to implement on approval.
- **0.3.0 queue**: finding 7 (demotion) belongs in the batched breaking release.
