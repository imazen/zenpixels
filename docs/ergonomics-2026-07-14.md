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

**Status.** Every public item in these crates is a forever commitment (see
`zenpixels/CLAUDE.md`), so the additive helpers below were surfaced as *proposals
with real consumers* first. Findings **1–4 and 9 were approved and have now
landed** (all additive, 0.2.x-compatible; each demonstrated in the galleries and
covered by a doctest). Findings 5, 6, 8 are docs-only. Finding 7 is a 0.3.0
demotion queued for the next breaking release.

## Summary

| # | Finding | Real consumers | Change | Semver | Status |
|---|---------|---------------:|--------|--------|--------|
| 1 | `PixelSlice::new_contiguous` / `PixelSliceMut::new_contiguous` — packed-stride ctor | ~361 | add method | additive (0.2.x) | **landed** |
| 2 | Owned-or-borrowed adapted pixels | 5 | return `PixelCow`; retain `Adapted` compatibility | additive (0.2.x) | **superseded and landed** |
| 3 | `RowConverter::convert_slice(PixelSlice) -> PixelBuffer` | 4–5 | add method | additive (0.2.x) | **landed** |
| 4 | `PixelDescriptor::with_color_from_cicp(Cicp)` | 3+ | add method | additive (0.2.x) | **landed** |
| 5 | Steer bare-`Vec` callers to `convert_to` | — | docs | none | doc |
| 6 | Crate-doc examples are `rust,ignore` (untested) | — | docs | none | galleries |
| 7 | `finalize_for_output_with` / `EncodeReady` — atomic encode contract, unadopted by codecs (weaker two-track path) | **0 ext.** | keep / adopt | none | see §7 |
| 8 | `PixelBufferConvertTypedExt` (`.to_rgba8`) is top idiom but `rgb`-gated | ~93 | docs | none | doc |
| 9 | Raw CICP tuple convenience | ~161 `Cicp::new` | leave validation in codec/container parsers | — | **abated** |

---

## 1. Packed-stride slice constructor — the single biggest win

**~361 `PixelSlice::new(...)` call sites**, and nearly every one hand-computes a
tight stride first. It is the most-repeated single line in the whole audit.

```rust
// today — the stride restates width * bpp every time
let stride = width as usize * descriptor.bytes_per_pixel();
let ps = PixelSlice::new(data, width, height, stride, descriptor)?;

// landed
let ps = PixelSlice::new_contiguous(data, width, height, descriptor)?;
```

Additive inherent methods on the erased slices (mirrors the existing `new`):

```rust
impl<'a> PixelSlice<'a> {
    /// `new` with a tightly-packed stride (`width * bytes_per_pixel`).
    pub fn new_contiguous(data: &'a [u8], width: u32, rows: u32, d: PixelDescriptor)
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

## 2. `PixelCow` — remove re-slicing and make ownership explicit

The `adapt_for_encode` → recompute-stride → `PixelSlice::new` → `encode` block is
**duplicated verbatim 5×** in zenpipe/zencodecs (`avif_enc.rs:159`,
`jxl_enc.rs:130`, `encode.rs:824`, `dispatch.rs:171`, `transcode.rs:706`).
The original proposal added an accessor to `Adapted`; subsequent review found
that `Adapted` duplicated a general owned-or-borrowed pixel concept.

```rust
// today
let adapted = adapt_for_encode(pixel_data, descriptor, w, h, stride, caps)?;
let adapted_stride = adapted.width as usize * adapted.descriptor.bytes_per_pixel();
let slice = PixelSlice::new(&adapted.data, adapted.width, adapted.rows, adapted_stride, adapted.descriptor)?;
encoder.encode(slice)?;

// canonical API
let adapted = adapt_for_encode_cow(pixel_data, descriptor, w, h, stride, caps)?;
encoder.encode(adapted.as_pixel_slice())?;
```

`PixelCow` carries bytes and layout together and exposes a `PixelSlice`
directly. The already-published `Adapted` type and free adaptation functions
remain as deprecated compatibility wrappers for 0.2.x; its accessor stays
fallible because an old `Adapted` can contain bytes that are not suitably
aligned. Shown in `convert_pipeline.rs::adapt_pixels_before_encoding`.

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

// landed
let out: PixelBuffer = RowConverter::new(src.descriptor(), target)?.convert_slice(src.as_slice())?;
```

Additive method taking a `PixelSlice` and returning an owned `PixelBuffer`
(pairs with finding 1). Aligns with the CLAUDE.md pixel-buffer-API rule
("anywhere a function takes the whole image it must accept stride / a `PixelSlice`").
`PixelBufferConvertExt::convert_to` already covers the "I hold a `PixelBuffer`"
case; this covers "I hold a `PixelSlice` and want scratch/streaming control."

## 4. `PixelDescriptor::with_color_from_cicp(Cicp)` — retag transfer+primaries

The "keep the format, adopt transfer+primaries from a decoded CICP" chain repeats
(`zenpng/src/codec.rs:2063` `enrich_descriptor_from_cicp`, `zenavif/src/codec.rs:2255`
**and** `:3274`, plus the same shape in `zenjxl/src/codec.rs` / `heic/src/codec.rs`):

```rust
// today
if let Some(tf) = TransferFunction::from_cicp(c.transfer_characteristics) { desc = desc.with_transfer(tf); }
if let Some(p)  = ColorPrimaries::from_cicp(c.color_primaries)          { desc = desc.with_primaries(p); }

// landed (const fn, keeps format/type/alpha, updates only color axes)
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

## 7. Encode-finalization surface (`finalize_for_output_with` / `EncodeReady`) — keep, do NOT demote

**Correction to an earlier draft of this report, which wrongly flagged this
surface for 0.3.0 demotion.** A full consumer audit confirms
`finalize_for_output_with`, `EncodeReady`, `OutputProfile`, `OutputMetadata`
have **zero external consumers** — but that is *under-adoption of the stronger
contract*, not dead weight to delete. (Only the `finalize_for_output`
`<C: ColorManagement>` overload and the `OutputMetadata::hdr` field are
deprecated.)

`finalize_for_output_with` is the crate's **type-enforced atomic encode
contract**: it converts pixels to the target profile and produces the matching
ICC/CICP *together*, so they cannot diverge — and its `EncodeReady::pixels()`
come off a `PixelBuffer`, hence always SIMD-aligned (unlike the deprecated
compatibility path through `Adapted`).

**What the codecs do instead (two-track, by-convention):**

- Track A — pixels: `adapt_for_encode_cow` (format adaptation only; it *refuses*
  to relabel primaries/range without a real conversion).
- Track B — metadata: `zencodec::resolve_color_emit` → `ColorEmitPlan{cicp,icc}`
  (pure/`no_std`, never sees pixels), lowered per-codec via
  `synthesize_icc_for_cicp`.

This mostly dodges the "P3 pixels / sRGB tag" bug **because the codecs never
color-convert at encode** — pixels arrive already in their working space. But
the guarantee is weaker than `EncodeReady`'s: the emitted metadata comes from a
caller-supplied `Metadata{cicp,icc}` that is a *separate input from the pixels*
and is never cross-checked. AVIF (`zenavif/src/codec.rs:999-1053`) treats the
caller CICP as authoritative and ignores the descriptor, so an upstream mislabel
is emitted faithfully wrong.

**Recommendation (reversed):** do NOT demote. Either (a) adopt
`finalize_for_output_with` in the codec lowering path for the type-enforced
guarantee, (b) keep the two-track design but add a pixel↔metadata cross-check,
or (c) at minimum keep the API as the reference contract. Two stale docs to fix
regardless: the crate's own module docs (`lib.rs:256,259,286,357,388`)
recommend the **deprecated** `finalize_for_output` instead of `_with`, and
`zencodec/src/color.rs:43` claims the encode path already lowers "through
zenpixels_convert's atomic finalize_for_output_with" — it does not.

## 8. Top convert idiom is `rgb`-gated and under-documented

`PixelBufferConvertTypedExt::to_rgba8()` / `.to_rgb8()` is the most common
decode-output one-liner downstream (~93 fully-qualified imports), but it lives
behind the `rgb` feature and is absent from the default-features docs. The
default-available equivalent is `PixelBufferConvertExt::convert_to(desc)`.
Recommendation: document both prominently and cross-link the feature gate. (No
API change — the galleries use `convert_to` since they build with default
features.)

## 9. Raw CICP tuple convenience

Abated after review. A four-byte container tuple has validation semantics
(especially for `video_full_range_flag`) that belong in the codec/container
parser, while `Cicp` deliberately preserves raw H.273 code points. No local
consumer needed a second constructor beside `Cicp::new`.

---

## What landed

- **Galleries + CI + docs**: the two tested example galleries, the CI
  `--examples` step, and this document.
- **Additive API (approved 2026-07-14)**: findings 1, 2, 3, 4 — `new_contiguous`
  on both slice types, `PixelCow` plus the canonical `_cow` adaptation APIs,
  `RowConverter::convert_slice`, and `PixelDescriptor::with_color_from_cicp`.
  Each is exercised in a gallery scenario and a doctest, and
  is used inside the crates where it removes boilerplate. `cargo semver-checks`
  classifies all of them as non-breaking additions.
- **Docs-only**: findings 5 (steer to `convert_to`), 6 (galleries replace the
  `rust,ignore` blocks as the source of truth), 8 (document the `rgb`-gated
  `to_rgba8`).
- **Finding 7 (corrected — NOT a demotion)**: `finalize_for_output_with` /
  `EncodeReady` is the type-enforced atomic encode contract and stays. The
  codecs' weaker two-track path (`adapt_for_encode_cow` + `resolve_color_emit`)
  carries the residual pixel↔metadata divergence gap. See §7.

Sibling repos (imageflow, zengif, zenpipe, zencodecs, zenpng, zenavif, …) still
hold the ~361 + N hand-rolled call sites the helpers target; migrating them is a
per-repo follow-up, done in each repo on its own.
