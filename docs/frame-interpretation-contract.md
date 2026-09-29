# Final color boundaries for the 0.2 and 0.3 releases

Updated 2026-09-29. This is the release-focused scope of PR #81. The prior
`Video::open()` sketch belongs to a future zencodecs façade, not zencodec and
not these zenpixels release gates. The [media reference](frame-interpretation-details.md)
is future design context; its unfinished MP4 implementation is not required to
ship the checked packed-pixel contract below.

## The concrete change

Keep raw signaling permissive. Make conversion into the smaller packed-pixel
description checked, and have the converter reuse that check.

```rust
use zenpixels::{Cicp, PixelFormat};
use zenpixels::cicp::CicpDescriptorError;

// Already-RGB declaration: no allocation or inspection of pixels.
let rgb = Cicp::SRGB.try_to_descriptor(PixelFormat::Rgb8)?;

// Original native YUV declaration cannot simply become an RGB descriptor.
let native = Cicp::new(9, 16, 9, false);
assert_eq!(
    native.try_to_descriptor(PixelFormat::Rgb16),
    Err(CicpDescriptorError::NonIdentityMatrix(9)),
);
# Ok::<(), CicpDescriptorError>(())
```

This implementation is on the PR branches, not main or a published release.

| Crate / API | 0.2 bridge | 0.3 cleanup |
|---|---|---|
| `zenpixels::Cicp::new` | Keep raw code points, including unknown/reserved values | Same |
| `Cicp::try_to_descriptor` | Add checked RGB/gray projection | Same signature and behavior |
| `Cicp::to_descriptor` | Deprecate; retain old behavior for compatibility | Remove |
| `cicp::CicpDescriptorError` | Add a small non-exhaustive error, without a root re-export | Same |
| Converter finalization | Reuse checked projection; retain cause as `ConvertError::CicpDescriptor` | Same |
| `sample::SampleEncoding` | Keep checked native-depth/placement vocabulary already in #76 | Same; legacy planar API stays removed |
| Generic matrix-hint resolver from #55 | Do not add | Do not add |

The new check requires identity matrix, RGB/gray layout and primaries/transfer
that the descriptor enums can represent. It preserves range and default alpha
semantics. It rejects CMYK/Oklab rather than attaching RGB signaling to them.
`UnmappedTransfer(7)` means the descriptor cannot represent that curve; it does
not mean that the raw standard code is invalid.

Do not change the matrix byte to make the check pass on unconverted samples.
After actual native-to-RGB reconstruction, the codec declares the output matrix,
range, layout and color that its reconstruction produced. Original signaling
remains in provenance. Normalized U16 remains a 16-bit numerical domain; this
method cannot turn native ten-bit words into normalized values.

## Two levels using existing types

The convenient checked path is `try_to_descriptor`. The lower-level path is an
explicit `PixelDescriptor` plus `ColorContext`; it is necessary for profiles and
raw codes outside the compact enums. No new frame or profile wrapper is needed.

```rust
use std::sync::Arc;
use zenpixels::{Cicp, ColorContext, ColorPrimaries, PixelDescriptor, TransferFunction};

// Example: already-packed RGB with an unimplemented transfer declaration.
// Transport it truthfully; a later conversion may require another backend.
let raw = Cicp::new(1, 200, 0, true);
let descriptor = PixelDescriptor::RGB8_SRGB
    .with_primaries(ColorPrimaries::Bt709)
    .with_transfer(TransferFunction::Unknown);
let context = Arc::new(ColorContext::from_cicp(raw));
```

The context must stay attached when a buffer is borrowed, copied or moved.
Descriptor `Unknown` does not retain the original numeric code by itself.
Raw storage acceptance is not a promise that a converter can process it.
`Cicp::try_to_descriptor` deliberately refuses this projection instead of silently
erasing that distinction. Core remains `alloc` + optional std, without a CMS.

## zenpixels-convert: existing planner and finalizer

Use `RowConverter`/`ConvertPlan` for explicit conversion, and
`finalize_for_output_with` when an owned image plus matching output metadata is
needed. No `Video` type and no file/codec selection enter this crate.

The finalizer already checked matrix/primaries/transfer before building its
output. This change removes duplicated checks and replaces the generic `NoPath`
failure with its precise cause:

```rust,ignore
match error.error() {
    ConvertError::CicpDescriptor(CicpDescriptorError::NonIdentityMatrix(code)) => {
        // Native component reconstruction is missing, or current context is stale.
    }
    ConvertError::CicpDescriptor(CicpDescriptorError::UnmappedTransfer(code)) => {
        // Preserve raw signaling; select a capable path rather than guessing.
    }
    _ => { /* existing conversion errors */ }
}
```

Invalid current/target CICP declarations fail before output allocation/conversion. A successful
projection still does not promise every format/transfer/backend combination is
implemented: the existing planner checks those capabilities next. Neither step
scans samples. Backend/sample-value errors during execution retain the existing
documented partial-write behavior for caller-provided buffers.

The already-agreed cleanup remains in #76/#77: hash-based ICC normalization on
`OutputProfile`, HDR computation in convert, unconditional estimation deprecation
then removal, explicit scan costs, and composition that avoids intermediate
quantization. This proposal does not reopen those decisions.

## zencodec: codec boundaries, not a reader façade

The inspected zencodec branches still declare **0.1.27**. The following is the
proposed zencodec **0.2/0.3 migration**, not a claim that those versions are already
staged or that their numbering must match zenpixels.

Keep `DecoderConfig → DecodeJob → Decode/StreamingDecode/AnimationFrameDecoder`
and the existing encode traits. Keep `DecodeOutput`, `OutputInfo`,
`AnimationFrame`, `SourceColor` and the buffers they already expose. Codecs report
their actual output; the application chooses codecs and conversion goals.

The concrete problems are in existing helpers, not a missing `Video::open`:

| Existing site | Problem | 0.2 migration | 0.3 result |
|---|---|---|---|
| `helpers::descriptor_for_decoded_pixels_v2` | Projects source primaries/transfer, drops source matrix/range, and falls back to sRGB with no metadata | Migrate codecs to declare their actual decoded output; deprecate the ambiguous helper after consumers migrate | Remove helper; retain the common explicit declarations |
| `helpers::resolve_color` | Its public tuple discards matrix/range and can supply an implicit sRGB assumption | Deprecate its use as a decoded-output resolver; codec-specific missing-color rules produce an explicit declaration | Remove the ambiguous helper after the 0.2 migration |
| `SourceColor::to_color_context` | Copies selected source signaling; it does not know whether reconstruction/CMS changed the pixels | Retain as a low-level source projection; document this limit and fix decoder call sites that attach it unchanged to transformed pixels | Same narrow contract; do not rename it speculatively |
| `DecodeOutput::new` / strip / animation output | Descriptor and attached context must describe delivered samples, not the original bitstream | Validate applicable known declarations in adapters before emitting output; expand shared testkit checks | Same buffer-based output contract; no required trait signature change for this work |
| zencodec dependency on zenpixels | Must admit the common checked migration without exposing two pixel type graphs | Raise the actual-use floor to 0.2.17 when adopting the method; retain the `<0.4` ceiling for these core lines | Keep one selected core version per connected pipeline |

Those zencodec changes are **pending**, including the codec migrations. There
are existing unrelated edits in its checkout; this PR has not modified them.
Do not deprecate helpers in a published release before real callers have a
documented, tested path using both admitted zenpixels versions.

Actual callers found include `zenjpeg/src/codec/decode.rs` and `codec/info.rs`.
For example, replace implicit fallback with the format adapter's explicit choice:

```rust,ignore
// OLD: a generic helper decides that no metadata means sRGB.
let desc = descriptor_for_decoded_pixels_v2(format, &source_color, None);

// NEW: this codec explicitly knows its output is full-range sRGB RGB.
let output_cicp = Cicp::SRGB;
let desc = output_cicp.try_to_descriptor(format)?; // map into codec error type
let context = Arc::new(ColorContext::from_cicp(output_cicp));
let pixels = PixelBuffer::from_vec(data, width, height, desc)?
    .with_color_context(context);
let output = DecodeOutput::new(pixels, info); // source claims remain in info
```

For ICC-described output, keep the actual current profile attached and use
unknown descriptor enums when a safe compact representation is unavailable.
Profile recognition alone is not permission to replace arbitrary TRCs/LUTs.
After RGB reconstruction, do not attach the original YUV matrix/range. After a
CMS conversion, do not attach the original ICC. These obligations apply equally
to one-shot pixels, streaming strips and animation frames.

Update `docs/correctness-model.md` with this distinction. Its current suggestion
to attach `SourceColor::to_color_context()` needs qualification, and its claim
that correctness metadata is always decided once per stream must not be extended
to changing video frame metadata. Add testkit fixtures for source YUV/current
RGB, source ICC/current converted color, unknown code retention, and strip/frame
context parity. Preserve each codec's own error type and classify failures using
the existing codec error facilities; no conversion-engine dependency in zencodec.

## What video requires from these releases

Native media views must bind `SampleEncoding` to their actual storage and carry
matrix/range/siting separately from a packed RGB descriptor. Keep those views,
backend ownership, timestamps and per-frame metadata in the experimental media
work until its real adapters prove the API. Do not add a replacement planar
container to zenpixels merely to finish 0.3.

MP4/AV1 precedence and missing-color rules belong in those adapters. zencodecs
may later provide convenient `CodecSet` operations. Neither requires core to
guess a matrix or to treat unknown raw codes as invalid. The older façade sketch
is not a release requirement or a proposed zencodec type.

## Verification and completion boundaries

Implemented now: checked core projection, 0.2 warning/0.3 removal, precise
converter error propagation, migrated local tests and same-source fixture.
Validation results are recorded in the PR description after running the checks.

Release gates for this change:

- Native matrix projection fails; unsupported raw codes remain preservable.
- Range, RGBX padding, alpha and already-RGB declarations are retained.
- CMYK/Oklab cannot receive an RGB CICP descriptor through the checked helper.
- Invalid current/target CICP produces a typed finalization error and unchanged input.
- Same migrated source builds with the admitted packaged core/converter pairs.
- Minimum Rust, no-std, API snapshots, deprecation/removal probes and compile cost.

The zencodec/codec/testkit migrations above remain a separate required release
checklist. This PR does not claim those consumers have already migrated, that
AV1-in-MP4 works, or that new artifacts have been published.

Checked 2026-09-29 at bridge `a4091ae` and cleanup `4d3cb8a`: full all-feature
tests passed (1,397 / 1,313), strict Clippy, API snapshots, core Rust 1.85,
converter Rust 1.89 and wasm32 no-default builds. Downstream probes passed
172 deprecation / 188 removal cases. The four packaged core/converter pairings
passed 12 feature configurations plus an existing-lockfile upgrade; this is not
a new all-sibling compatibility claim. A follow-up fixed redundant rustdoc links.

The [compile observation](../benchmarks/cicp-projection-compile-2026-09-29.json)
uses fresh git archives and the same locked dependencies. Core cold check was
1.301 to 1.294 seconds; converter check after core was 3.235 to 3.236 seconds.
These are single local paired observations, not statistical performance claims.
No manifest, feature, or production dependency changed.
