# Code review: adoption, planar retirement and remaining contracts

2026-09-27. **Read this instead of prose-only proposals when deciding the next
chunk.** Adoption, planar warnings and the small fixes listed below are implemented.
Historical before/after examples remain below; use the runnable case files for
current behavior. The broader prepared-worker, output/CMS and validation contracts
are still outstanding.

The complete current-behavior examples are checked in under `contract-cases/`.
Run `python3 scripts/check-contract-cases.py`. Most assertions deliberately
recognize bugs: a passing review suite is evidence of reproduction, **not a clean
bill of health**. It is separate from ordinary CI/release correctness tests.
Default and minimal feature configurations exercise different CMS/Clone cases.
Both runs passed 26 assertions/tests in this audit, covering 27 distinct test cases
(15 reproduce remaining defects; 11 verify fixes; one verifies the selected
final-output composition policy).

Implemented in the latest chunk: empty crop row reads, primary containment, CICP
padding semantics, orientation context, known-transfer adapter conversion, strict
in-place refusal of color retagging, scalar Adobe gamma, F16 underflow rounding,
CMS composition refusal, swap metadata and strided ImgVec adoption. No additional
image prepass was introduced. See the [performance review](performance-review-0.2-and-0.3.md)
for the owner-selected cost model and [U16 review](u16-signaling-and-narrowing-review.md)
for native-code signaling.

## Implemented: adoption and small retained errors

```rust
use zenpixels::{BufferError, PixelBuffer, PixelBufferParts};

fn adopt(parts: PixelBufferParts) -> Result<PixelBuffer, whereat::At<BufferError>> {
    // Release rejected image storage before propagating, boxing or retaining error.
    PixelBuffer::try_from_parts(parts).map_err(|e| e.without_buffer())
}

fn adopt_boxed(parts: PixelBufferParts) -> Result<PixelBuffer, Box<dyn std::error::Error>> {
    let buffer = PixelBuffer::try_from_parts(parts).map_err(|e| e.without_buffer())?;
    Ok(buffer)
}

fn recover(parts: PixelBufferParts) {
    match PixelBuffer::try_from_parts(parts) {
        Ok(buffer) => { drop(buffer); /* continue with the adopted allocation */ }
        Err(mut error) => {
            let rejected = error.take_parts().unwrap();
            assert!(error.take_parts().is_none());
            // Return rejected.data to a pool, repair/retry, or pass complete parts on.
            // Log/store the now-small error independently from the image allocation.
            eprintln!("{error}");
            drop(rejected);
        }
    }
}
```

`FromPartsError` stores the traced cause and optional rejected parts inline;
`error()` borrows the cause, `take_parts(&mut self)` takes it once, and
`without_buffer(self)` returns `At<BufferError>`. Debug output omits pixel data.
Successful adoption preserves pointer/length/capacity/offset/stride/context without allocating,
copying or cloning context. It validates storage as `PixelSlice::new` does, not
all proposed descriptor/color semantics. Failure captures the ordinary small
`whereat` trace allocation; the rejected parts themselves are stored inline. The new tests cover failure recovery,
misalignment, offset/extent/stride/overflow, empty packed images and context release.

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};
let mut parts = PixelBuffer::new(4, 1, PixelDescriptor::RGB8).into_parts();
parts.width = 1;
parts.height = 2;
parts.stride_bytes = 9; // 9 bytes before row 2, then only 3 visible bytes
let buffer = PixelBuffer::try_from_parts(parts).unwrap();
assert_eq!(buffer.as_slice().row(1).len(), 3); // no trailing padding required
```

## Implemented: deprecate the entire old planar module

```rust,ignore
// With planar enabled, all these now warn, through either crate's re-exports:
use zenpixels::planar::MultiPlaneImage;
use zenpixels::PlaneMask;
use zenpixels_convert::PlaneMask;
let mask = PlaneMask::LUMA;
let count = mask.count(); // inferred method access warns too
```

Keep the feature and code available in 0.2. Defer the old owned-container repair
API and borrowed YUV publication until video design review; do not claim a complete
replacement already exists. Decide removal in 0.3 only with a usable migration.
This explicitly supersedes the prior proposal to extend the existing planar module.

A refreshed local search found zenfilters uses **only `PlaneMask`** in
`zenpipe/zenfilters/src/{access.rs,masked.rs,filters/alpha.rs}`. Its `OklabPlanes`
are local types. The use is small in scope but `ChannelAccess` exposes the mask
publicly, so replacing it is a companion API migration, not just deleting an import.
The search also found a copied zenfilters under `imagers-research/extracted`;
this is not evidence of another independently designed consumer.

```rust,ignore
// Existing zenfilters public API:
pub struct ChannelAccess {
    pub reads: zenpixels::PlaneMask,
    pub writes: zenpixels::PlaneMask,
}
// Proposed companion migration: own this filter-channel selection in zenfilters.
// Define/migrate a filter-specific mask there before removing the old core mask;
// it does not require a universal video-plane container.
```

## Reproduced code cases

The files supply imports and, for CMS tests, the spy implementation shown below.
All test bodies in this section are copied from those runnable files. “Wanted”
blocks are proposed assertions, evaluated in the same setup unless explicitly
marked as a future interface. They are not claimed to pass today. Changing an
assertion alone is not a fix: output tests need independent reference checks too.

### Minimal final-row backing: fixed in this chunk

This used to accept the transform and then panic while borrowing the buffer. Adoption required fixing the same path. The displayed assertion now passes.

Current executable case ([storage.rs](contract-cases/storage.rs)):

```rust
fn adopted_minimal_extent_now_supports_owned_views() {
    let mut b = PixelBuffer::new(4, 1, PixelDescriptor::RGB8);
    b.transform_in_place(|p| {
        PixelSliceMut::new(p.bytes, 1, 2, 9, PixelDescriptor::RGB8).unwrap()
    });
    assert_eq!(b.stride(), 9);
    assert_eq!(b.as_slice().row(1).len(), 3);
}
```

Wanted:

```rust,ignore
assert_eq!(b.as_slice().row(1).len(), 3);
```

### Zero-width rows

The current assertion catches the panic. A valid empty crop must give empty visible rows. This broader empty-view fix is still proposed.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([storage.rs](contract-cases/storage.rs)):

```rust
fn empty_crop_positive_rows_panics() {
    let b = PixelBuffer::new(4, 4, PixelDescriptor::RGB8);
    let crop = b.crop_view(0, 0, 0, 2);
    assert!(std::panic::catch_unwind(|| crop.row(1)).is_err());
}
```

Wanted:

```rust,ignore
assert_eq!(crop.row(1), &[]);
```

### Gamut containment must not authorize clipping

The declared containment is false: Adobe green maps outside P3. Use an accurate predicate, and never treat approximate containment as exact preservation.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([storage.rs](contract-cases/storage.rs)):

```rust
fn p3_does_not_contain_adobe_green_despite_predicate() {
    assert!(ColorPrimaries::DisplayP3.contains(ColorPrimaries::AdobeRgb));
    let m = ColorPrimaries::AdobeRgb
        .gamut_matrix_to(ColorPrimaries::DisplayP3)
        .unwrap();
    eprintln!("Adobe green in P3: {:?}", [m[0][1], m[1][1], m[2][1]]);
    assert!(
        [m[0][1], m[1][1], m[2][1]]
            .iter()
            .any(|v| *v < -0.01 || *v > 1.01)
    );
}
```

Wanted:

```rust,ignore
assert!(!ColorPrimaries::DisplayP3.contains(ColorPrimaries::AdobeRgb));
```

### CICP must not turn an undefined lane into alpha

Signaling transfer/primaries must preserve the physical RGBX alpha semantics.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([storage.rs](contract-cases/storage.rs)):

```rust
fn cicp_to_descriptor_promotes_padding_to_alpha() {
    let d = Cicp::SRGB.to_descriptor(PixelFormat::Rgbx8);
    assert_eq!(d.alpha, Some(AlphaMode::Straight));
    assert_ne!(d.alpha, PixelFormat::Rgbx8.default_alpha());
}
```

Wanted:

```rust,ignore
assert_eq!(d.alpha, PixelFormat::Rgbx8.default_alpha());
```

### Equivalent named and CICP interpretation

Resolution must consistently support both representations or consistently refuse them; it must not lose the HDR meaning when translating the same signaling.

Current executable case ([storage.rs](contract-cases/storage.rs)):

```rust
fn named_pq_cicp_roundtrip_changes_resolution() {
    let named = ColorProfileSource::Named(NamedProfile::Bt2020Pq);
    let cicp = ColorProfileSource::Cicp(NamedProfile::Bt2020Pq.to_cicp().unwrap());
    assert!(named.resolve().is_some());
    assert!(cicp.resolve().is_none());
}
```

Wanted:

```rust,ignore
assert_eq!(named.resolve().is_some(), cicp.resolve().is_some());
```

### Contradictory alpha declarations

Keep descriptors as declarations, but reject contradictions at storage/plan acceptance. Do not make the type universally private to solve this.

Current executable case ([storage.rs](contract-cases/storage.rs)):

```rust
fn pixel_format_alpha_and_descriptor_alpha_can_disagree() {
    let d = PixelDescriptor::RGBX8.with_alpha(Some(AlphaMode::Straight));
    assert_eq!(d.pixel_format(), PixelFormat::Rgbx8);
    assert!(d.has_alpha());
    let d = PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Premultiplied));
    assert!(d.has_alpha());
    assert!(!d.pixel_format().has_alpha_bytes());
}
```

Wanted:

```rust,ignore
let invalid = PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Premultiplied));
assert!(PixelBuffer::try_new(1, 1, invalid).is_err());
assert!(PixelSlice::new(&[1, 2, 3], 1, 1, 3, invalid).is_err());
```

### Orientation must retain current color

The allocating route drops context while the in-place route keeps it. Both must retain the same current interpretation.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([storage.rs](contract-cases/storage.rs)):

```rust
fn orientation_allocating_drops_color_but_in_place_keeps_it() {
use std::sync::Arc;
use zenpixels::{Cicp, ColorContext, Orientation, PixelBuffer, PixelDescriptor};
use zenpixels_convert::orient::{apply_orientation, apply_orientation_in_place};
let ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
let mut src = PixelBuffer::new(2, 3, PixelDescriptor::RGB8).with_color_context(ctx);
let out = apply_orientation(src.as_slice(), Orientation::Rotate90);
assert!(out.color_context().is_none());
apply_orientation_in_place(&mut src, Orientation::Rotate90).unwrap();
assert!(src.color_context().is_some());
}
```

Wanted:

```rust,ignore
assert_eq!(out.color_context().unwrap().cicp, Some(Cicp::DISPLAY_P3));
```

### Known transfer changes require conversion

Current fast path relabels linear 128 as sRGB 128; real conversion is approximately 188. Fix both intent and explicit-policy adapters, and inspect in-place paths.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn known_transfer_is_silently_retagged() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let pixels = [128u8; 3];
    let out = adapt::adapt_for_encode_cow(&pixels, src, 1, 1, 3, &[PixelDescriptor::RGB8_SRGB])
        .unwrap();
    assert_eq!(
        out.as_slice().descriptor().transfer(),
        TransferFunction::Srgb
    );
    assert_eq!(out.as_slice().row(0), &[128; 3]);
    let mut real = RowConverter::new(src, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut converted = [0; 3];
    real.convert_row(&pixels, &mut converted, 1);
    assert!(converted[0] >= 187, "actual linear→sRGB must encode ~188");
}
```

Wanted:

```rust,ignore
assert!(!out.is_borrowed());
assert_eq!(out.as_slice().row(0), converted);
```

### Composition optimizes final output; preserve stages explicitly

The owner selected final-output optimization: ordinary composition may eliminate
the intermediate quantization. Execute the converters separately through a reusable
row to preserve an intentional integer stage. A future stage-preserving composition
option must retain descriptor boundaries and per-stage parameters, not merely
concatenate step lists.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn compose_optimizes_final_output_by_default() {
    let f = PixelDescriptor::RGBF32_LINEAR;
    let u = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let mut a = RowConverter::new(f, u).unwrap();
    let mut b = RowConverter::new(u, f).unwrap();
    let mut composed = a.compose(&b).unwrap();
    let src = [0.1234567f32; 3];
    let mut quantized = [0u8; 3];
    let mut separate = [0f32; 3];
    let mut together = [0f32; 3];
    a.convert_row(bytemuck::cast_slice(&src), &mut quantized, 1);
    b.convert_row(&quantized, bytemuck::cast_slice_mut(&mut separate), 1);
    composed.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut together),
        1,
    );
    assert!(composed.is_identity());
    assert_eq!(together, src);
    assert_ne!(together, separate);
}
```

Wanted:

```rust,ignore
// Owner-selected default: avoid accidental intermediate quantization.
assert!(composed.is_identity());
assert_eq!(together, src);
// Explicitly execute a then b through a reusable row to preserve the U8 stage.
assert_ne!(separate, src);
```

### Different layout is not identity

The first two RGBA bytes are not gray+alpha. Either support luminance conversion with retained alpha, or reject this route at construction.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn rgba_to_grayalpha_is_accepted_as_identity() {
    let src = PixelDescriptor::RGBA8_SRGB;
    let dst = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let mut converter = RowConverter::new(src, dst).unwrap();
    let mut out = [0u8; 2];
    converter.convert_row(&[200, 30, 10, 99], &mut out, 1);
    assert_eq!(out, [200, 30]);
    assert!(converter.is_identity());
}
```

Wanted:

```rust,ignore
// If the route is supported:
assert!(!converter.is_identity());
assert_eq!(out[1], 99); // alpha, not the source green channel
// Verify gray against the specified luminance/transfer reference too.
```

### Gamma22 must implement its advertised transfer

Returning the input is not a supported Gamma22 conversion. Correct the math or expose refusal at a supported boundary.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn gamma22_scalar_is_identity() {
    assert_eq!(TransferFunction::Gamma22.linearize(0.5), 0.5);
}
```

Wanted:

```rust,ignore
assert!((TransferFunction::Gamma22.linearize(0.5) - 0.5f32.powf(563.0 / 256.0)).abs() < 1e-6);
```

### F16 rounding at a subnormal boundary

The minimum subnormal multiplied by .375 is below the midpoint and must round to zero under nearest rounding. This is a numerical fix, not a source migration.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn f16_premultiply_rounds_below_midpoint_up() {
    let src = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let dst = src.with_alpha(Some(AlphaMode::Premultiplied));
    let mut converter = RowConverter::new(src, dst).unwrap();
    let samples = [1u16, 0, 0, 0x3600]; // minimum subnormal color; alpha .375
    let mut out = [0u16; 4];
    converter.convert_row(
        bytemuck::cast_slice(&samples),
        bytemuck::cast_slice_mut(&mut out),
        1,
    );
    assert_eq!(
        out[0], 1,
        "minimum-subnormal times .375 is below midpoint and should round to zero"
    );
}
```

Wanted:

```rust,ignore
assert_eq!(out[0], 0);
```

### DiscardIfOpaque must check the actual pixels

The policy is accepted but a transparent alpha channel is discarded. Fallible execution must report AlphaNotOpaque; strict in-place must detect it before mutation.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn opaque_only_policy_not_enforced_by_rowconverter() {
    let options = policy::ConvertOptions::permissive()
        .with_alpha_policy(policy::AlphaPolicy::DiscardIfOpaque);
    let mut converter = RowConverter::new_explicit(
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        &options,
    )
    .unwrap();
    let mut out = [0u8; 3];
    converter.convert_row(&[200, 30, 10, 0], &mut out, 1);
    assert_eq!(out, [200, 30, 10]);
}
```

Wanted:

```rust,ignore
// PROPOSED prepared worker, same policy and source:
let result = worker.try_convert_row(&[200, 30, 10, 0], &mut out, 1);
assert!(matches!(result.unwrap_err().error(), ConvertError::AlphaNotOpaque));
```

### Composite before dropping alpha or converting to gray

Transparent black over white must become white. The selected background is part of the requested operation sequence.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn composite_to_gray_ignores_background() {
    let options = policy::ConvertOptions::permissive().with_alpha_policy(
        policy::AlphaPolicy::CompositeOnto {
            r: 255,
            g: 255,
            b: 255,
        },
    );
    let dst = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Gray,
        None,
        TransferFunction::Srgb,
    );
    let mut converter =
        RowConverter::new_explicit(PixelDescriptor::RGBA8_SRGB, dst, &options).unwrap();
    let mut out = [123u8; 1];
    converter.convert_row(&[0, 0, 0, 0], &mut out, 1);
    assert_eq!(
        out,
        [0],
        "transparent black over white should have produced white"
    );
}
```

Wanted:

```rust,ignore
assert_eq!(out, [255]);
```

### Unassociate before nonlinear transfer conversion

Premultiplied encoded .25 with alpha .5 means straight encoded .5. The requested straight-linear result is about .214041, not .101752.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn premul_transfer_changes_values_in_wrong_domain() {
    let src = PixelDescriptor::RGBAF32_LINEAR
        .with_transfer(TransferFunction::Srgb)
        .with_alpha(Some(AlphaMode::Premultiplied));
    let dst = PixelDescriptor::RGBAF32_LINEAR;
    let mut converter = RowConverter::new(src, dst).unwrap();
    let values = [0.25f32, 0.25, 0.25, 0.5];
    let mut out = [0f32; 4];
    converter.convert_row(
        bytemuck::cast_slice(&values),
        bytemuck::cast_slice_mut(&mut out),
        1,
    );
    assert!((out[0] - 0.101752).abs() < 1e-5, "got {}", out[0]);
    assert!((TransferFunction::Srgb.linearize(0.5) - 0.214041).abs() < 1e-5);
}
```

Wanted:

```rust,ignore
assert!((out[0] - TransferFunction::Srgb.linearize(0.5)).abs() < 1e-5);
assert_eq!(out[3], 0.5);
```

### Unsupported gamut/layout paths fail before execution

Construction currently succeeds, then execution panics. Either implement the route or refuse during planning/preparation.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn adobe_oklab_plan_panics_during_execution() {
    let src = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb);
    let dst = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Oklab,
        None,
        TransferFunction::Linear,
    )
    .with_primaries(ColorPrimaries::AdobeRgb);
    let mut converter = RowConverter::new(src, dst).unwrap();
    let mut dst = [0f32; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        converter.convert_row(
            bytemuck::cast_slice(&[0.5f32; 3]),
            bytemuck::cast_slice_mut(&mut dst),
            1,
        );
    }));
    assert!(result.is_err());
}
```

Wanted:

```rust,ignore
// If this route remains unsupported:
let target = PixelDescriptor::new(
    ChannelType::F32, ChannelLayout::Oklab, None, TransferFunction::Linear,
).with_primaries(ColorPrimaries::AdobeRgb);
assert!(RowConverter::new(src, target).is_err());
// A supported implementation instead needs reference-checked output and no panic.
```

### CMS depth combinations must be supported as a pair

A backend supporting U8 and F32 separately does not imply U8-to-F32 support. Refuse while building/preparing, or implement the cross-depth route and execute fallibly.

Current executable case ([conversion.rs](contract-cases/conversion.rs)):

```rust
fn moxcms_crossdepth_panics_after_successful_build() {
    let from = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let mut converter = RowConverter::new_explicit_with_cms(
        from,
        PixelDescriptor::RGBF32_LINEAR,
        &policy::ConvertOptions::permissive(),
        Some(&MoxCms),
    )
    .unwrap();
    let mut dst = [0f32; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        converter.convert_row(&[128u8; 3], bytemuck::cast_slice_mut(&mut dst), 1);
    }));
    assert!(
        result.is_err(),
        "cross-depth must not silently write byte output into f32 storage"
    );
}
```

Wanted:

```rust,ignore
// If the backend cannot execute this pair:
assert!(RowConverter::new_explicit_with_cms(
    from, PixelDescriptor::RGBF32_LINEAR,
    &policy::ConvertOptions::permissive(), Some(&MoxCms),
).is_err());
```

### CMS fixture used by the following cases

```rust
    struct Fill;
    impl RowTransformMut for Fill {
        fn transform_row(&mut self, _s: &[u8], d: &mut [u8], _: u32) {
            d.fill(42)
        }
    }
    #[derive(Default)]
    struct Spy(Mutex<Vec<(bool, bool)>>);
    impl PluggableCms for Spy {
        fn build_source_transform(
            &self,
            s: ColorProfileSource<'_>,
            d: ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
            self.0.lock().unwrap().push((
                matches!(s, ColorProfileSource::Icc(_)),
                matches!(d, ColorProfileSource::Icc(_)),
            ));
            Some(Ok(Box::new(Fill)))
        }
    }
    fn custom_converter() -> RowConverter {
        RowConverter::new_explicit_with_cms(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
            &ConvertOptions::permissive(),
            Some(&Spy::default()),
        )
        .unwrap()
    }
```

### SameAsOrigin must transform back before restoring tags

The finalizer emits Display P3 metadata for unchanged BT.709 pixels. A successful result must describe P3 samples and contain the corresponding conversion.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn same_as_origin_can_mistag_current_pixels() {
    let b = PixelBuffer::from_vec(vec![200, 50, 10], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    let r = finalize_for_output_with(
        &b,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        None,
    )
    .unwrap();
    assert_eq!(r.metadata().cicp, Some(Cicp::DISPLAY_P3));
    assert_eq!(r.pixels().descriptor().primaries, ColorPrimaries::Bt709);
    assert_eq!(r.pixels().row(0), [200, 50, 10]);
}
```

Wanted:

```rust,ignore
assert_eq!(r.metadata().cicp, Some(Cicp::DISPLAY_P3));
assert_eq!(r.pixels().descriptor().primaries, ColorPrimaries::DisplayP3);
assert_ne!(r.pixels().row(0), [200, 50, 10]);
// Also compare pixels against an independent P3 transform; retagging alone cannot pass.
```

### Output identity includes alpha association

With alpha 128, RGB 64/32/16 are premultiplied. If requesting straight output, unassociate or refuse; keeping bytes and changing the tag is wrong.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn output_identity_relabels_premultiplied_as_straight() {
    let d = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
    let b = PixelBuffer::from_vec(vec![64, 32, 16, 128], 1, 1, d).unwrap();
    let r = finalize_for_output_with(
        &b,
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgba8,
        None,
    )
    .unwrap();
    assert_eq!(r.pixels().descriptor().alpha(), Some(AlphaMode::Straight));
    assert_eq!(r.pixels().row(0), [64, 32, 16, 128]);
}
```

Wanted:

```rust,ignore
assert_eq!(r.pixels().descriptor().alpha(), Some(AlphaMode::Straight));
assert_eq!(r.pixels().row(0), [128, 64, 32, 128]); // rounded U8 unassociation
```

### Output identity includes signal range

Limited-range tags must not accompany unchanged full-range white. Use correct range conversion or explicitly refuse unsupported signaling.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn output_ignores_cicp_signal_range() {
    let b =
        PixelBuffer::from_vec(vec![255, 255, 255], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let narrow = Cicp::new(1, 13, 0, false);
    let r = finalize_for_output_with(
        &b,
        &ColorOrigin::assumed(),
        OutputProfile::Named(narrow),
        PixelFormat::Rgb8,
        None,
    )
    .unwrap();
    assert_eq!(r.metadata().cicp, Some(narrow));
    assert_eq!(r.pixels().descriptor().signal_range, SignalRange::Full);
    assert_eq!(r.pixels().row(0), [255, 255, 255]);
}
```

Wanted:

```rust,ignore
// If supported, descriptor and metadata agree:
assert_eq!(r.pixels().descriptor().signal_range, SignalRange::Narrow);
assert_ne!(r.pixels().row(0), [255, 255, 255]);
// Check endpoints against the selected range specification.
```

### Pass actual source and target ICC profiles to CMS

The spy receives two non-ICC sources even though both actual profiles are ICC. The profile-aware request must carry the selected bytes, not descriptor guesses.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn output_cms_does_not_receive_icc() {
    let cms = Spy::default();
    let icc: std::sync::Arc<[u8]> = zenpixels_convert::icc_profiles::DISPLAY_P3_V4.into();
    let b = PixelBuffer::from_vec(vec![200, 50, 10], 1, 1, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .with_icc(icc.clone());
    let r = finalize_for_output_with(
        &b,
        &ColorOrigin::from_icc(icc.clone()),
        OutputProfile::Icc(icc),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(*cms.0.lock().unwrap(), vec![(false, false)]);
    assert_eq!(r.pixels().row(0), [42, 42, 42]);
}
```

Wanted:

```rust,ignore
// If CMS is invoked (rather than proving profile identity):
assert_eq!(*cms.0.lock().unwrap(), vec![(true, true)]);
// A production fixture must additionally check the profile bytes themselves.
```

### Composition must preserve CMS work

The custom transform fills 42. A successful composed converter must do that same work, not return input samples.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn compose_discards_external_transform() {
    let mut a = custom_converter();
    let b = RowConverter::new(a.to_descriptor(), a.to_descriptor()).unwrap();
    let mut c = a.compose(&b).unwrap();
    let mut direct = [0; 3];
    let mut composed = [0; 3];
    a.convert_row(&[1, 2, 3], &mut direct, 1);
    c.convert_row(&[1, 2, 3], &mut composed, 1);
    assert_eq!(direct, [42, 42, 42]);
    assert_eq!(composed, [1, 2, 3]);
}
```

Wanted:

```rust,ignore
assert_eq!(composed, direct); // both [42, 42, 42]
// Or legacy compose returns None before execution.
```

### Cloning must not erase an external transform

This reproduces with convert std disabled. Replace legacy worker cloning with independently prepared workers; do not silently lose backend state.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn no_std_clone_discards_external_transform() {
    let mut a = custom_converter();
    let mut c = a.clone();
    let mut direct = [0; 3];
    let mut cloned = [0; 3];
    a.convert_row(&[1, 2, 3], &mut direct, 1);
    c.convert_row(&[1, 2, 3], &mut cloned, 1);
    assert_eq!(direct, [42, 42, 42]);
    assert_eq!(cloned, [1, 2, 3]);
}
```

Wanted:

```rust,ignore
// On the future prepared API:
let mut a = plan.prepare(1)?;
let mut b = plan.prepare(1)?;
a.try_convert_row(&[1, 2, 3], &mut direct, 1)?;
b.try_convert_row(&[1, 2, 3], &mut cloned, 1)?;
assert_eq!(direct, cloned);
```

### Reinterpretation must validate the new sample alignment

The bytes are deliberately not F32-aligned. Equal bytes per pixel alone is insufficient.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn reinterpret_accepts_invalid_alignment() {
    let data = [0u8; 8];
    let offset = (0..4)
        .find(|&i| (data.as_ptr() as usize + i) % 4 != 0)
        .unwrap();
    let bytes = &data[offset..offset + 4];
    let s = PixelSlice::new(bytes, 1, 1, 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let f = PixelFormat::GrayF32.descriptor();
    assert!(PixelSlice::new(bytes, 1, 1, 4, f).is_err());
    assert!(s.reinterpret(f).is_ok());
}
```

Wanted:

```rust,ignore
assert!(s.reinterpret(f).is_err());
```

### Typed reinterpretation cannot keep a contradictory type

The same value remains statically RGBA while its descriptor says BGRA. Deprecate typed-preserving reinterpretation; make callers erase then retype explicitly.

Current executable case ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn typed_reinterpret_retains_wrong_type() {
    let mut data = [1u8, 2, 3, 4];
    let s = PixelSliceMut::<rgb::RGBA<u8>>::new_typed(&mut data, 1, 1, 1).unwrap();
    let s: PixelSliceMut<'_, rgb::RGBA<u8>> =
        s.reinterpret(PixelDescriptor::BGRA8_SRGB).unwrap();
    assert_eq!(s.descriptor().pixel_format(), PixelFormat::Bgra8);
}
```

Wanted:

```rust,ignore
// Existing operations demonstrate the intended explicit type change:
let erased = s.erase();
let bgra = erased.reinterpret(PixelDescriptor::BGRA8_SRGB)?;
let _: PixelSliceMut<'_ , rgb::BGRA<u8>> = bgra.try_typed()?;
```

### Channel reorder preserves color and alpha association

Swapping R/B is not a transfer, gamut or alpha conversion. The helper currently resets all three declarations.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn layout_helper_resets_color_and_alpha() {
    let mut data = [1u8, 2, 3, 128];
    let s = PixelSliceMut::<rgb::RGBA<u8>>::new_typed(&mut data, 1, 1, 1)
        .unwrap()
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_transfer(TransferFunction::Linear)
        .with_alpha_mode(Some(AlphaMode::Premultiplied));
    let s = s.swap_to_bgra();
    assert_eq!(s.descriptor().primaries, ColorPrimaries::Bt709);
    assert_eq!(s.descriptor().transfer(), TransferFunction::Srgb);
    assert_eq!(s.descriptor().alpha(), Some(AlphaMode::Straight));
}
```

Wanted:

```rust,ignore
assert_eq!(s.descriptor().primaries, ColorPrimaries::DisplayP3);
assert_eq!(s.descriptor().transfer(), TransferFunction::Linear);
assert_eq!(s.descriptor().alpha(), Some(AlphaMode::Premultiplied));
```

### Typed image adoption must retain the stride it actually stores

The copied data was compacted while the old stride survived. Either retain strided storage or update the stride to match the compacted allocation.

Before this fix (historical repro; runnable case now asserts the corrected behavior) ([output_cms.rs](contract-cases/output_cms.rs)):

```rust
fn from_imgvec_keeps_old_stride_after_compacting() {
    let px = rgb::RGB8::new(1, 2, 3);
    let img = imgref::Img::new_stride(vec![px; 6], 2, 2, 3);
    let b = PixelBuffer::<rgb::RGB8>::from_imgvec(img);
    assert_eq!(b.stride(), 9);
    assert!(std::panic::catch_unwind(|| b.as_slice().row(1).to_vec()).is_err());
}
```

Wanted:

```rust,ignore
assert_eq!(b.as_slice().row(1), &[1, 2, 3, 1, 2, 3]);
// If compacted, also assert_eq!(b.stride(), 6).
```

## Remaining interface proposals, with code to review

These are design examples, not implemented APIs. Their final spelling remains
open. They explain what changes at a caller boundary beyond the reproduced bugs.

### Resolve color authority before planning

```rust,ignore
// CURRENT: useful builders can also attach contradictory declarations.
let context = ColorContext::from_icc(custom_icc).with_cicp(Cicp::SRGB);
let selected_profile = context.as_profile_source(); // ICC
let advertised_srgb = context.is_srgb();            // may follow CICP

// PROPOSED contract: one selected authority, not two interpretations.
// A conflict must fail unless the caller selects an authority explicitly.
let encoding = PixelEncoding::resolve(descriptor, &context, Authority::Icc)?;
assert!(matches!(encoding.profile(), ColorProfileSource::Icc(_)));
// All identity/transfer/CMS decisions now use `encoding`.
// Unselected tags remain in source history, not current interpretation.
```

`PixelEncoding::resolve`, `Authority` and `profile()` above are provisional names.
This is a request for API review, not a second context getter: existing
`color_context()` already borrows. The same selected input must reach the finalizer.

### Preparation, width limits and row errors

```rust,ignore
// CURRENT repeated work: the free function can allocate scratch for every call.
let plan = ConvertPlan::new(source.descriptor(), target)?;
for y in 0..source.rows() {
    convert_row(&plan, source.row(y), output.row_mut(y), width);
}

// PROPOSED: allocate/initialize state once; execution cannot silently grow it.
let mut worker = plan.prepare(width)?;
for y in 0..source.rows() {
    worker.try_convert_row(source.row(y), output.row_mut(y), width)?;
}
// With sufficiently sized byte slices, this still refuses exceeding capacity:
assert!(worker.try_convert_row(wider_src, wider_dst, width + 1).is_err());
// A new capacity is explicit and fallible:
let mut larger_worker = plan.prepare(width + 1)?;
```

Slice execution additionally validates **both** source and destination against
the plan. PR #63 only checked the destination. This is the negative case we need:

```rust,ignore
// PROPOSED acceptance test; even identical byte extents cannot bypass semantics.
let mut worker = srgb_to_linear_plan.prepare(1)?;
let bytes = [128u8; 3];
let wrong_source = PixelSlice::new(
    &bytes, 1, 1, 3, PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear),
)?;
let mut dst = PixelBuffer::new(1, 1, PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear));
assert!(worker.convert_slice_into(wrong_source, dst.as_slice_mut()).is_err());
```

### Fallible CMS interface, without renaming the bridge again

```rust,ignore
// CURRENT implementor cannot report a row failure through this signature:
impl RowTransformMut for MyCmsTransform {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        self.backend.convert(src, dst, width).expect("no error channel");
    }
}

// PROPOSED new trait, final name subject to ownership prototype:
trait CmsRowExecutor: Send {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32)
        -> Result<(), whereat::At<ConvertError>>;
}
impl CmsRowExecutor for MyCmsTransform {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32)
        -> Result<(), whereat::At<ConvertError>> {
        self.backend.convert(src, dst, width).map_err(map_backend_error)
    }
}
```

The factory takes complete resolved encodings/policies, returns a distinct refusal
for unsupported requests, and prepares independent mutable executors. The test
must force an accepted transform to fail and assert the original error reaches
the caller, no fallback backend runs, and no encode-ready object is returned.
Destination bytes may be partially written; the API must say so. Keep old trait
implementations valid in 0.2 until their whole-trait migration is ready.

### Exact preservation cannot permit clipping or trust origin history

```rust,ignore
// CURRENT preset name does not mean no clipping:
let options = ConvertOptions::forbid_lossy();
assert!(options.clip_out_of_gamut); // current behavior; incompatible with exactness
// CURRENT provenance does not prove current representability:
let resized_sample = (0.0f32 + 1.0 / 255.0) / 2.0;
assert_ne!(resized_sample, (resized_sample * 255.0).round() / 255.0);

// PROPOSED request method; reject when exactness cannot be established.
let plan = request.require_sample_preservation().plan()?;
// On observed non-grid samples, refuse rather than silently round:
assert!(plan.prepare(1)?.try_convert_row(resized_f32, u8_output, 1).is_err());
```

A planner may reject the whole route rather than scan. A scan is explicit work,
not free proof. Reversible channel reorder or exact U8→U16 widening can succeed;
clipping, alpha removal without opacity proof, and arbitrary transfer/gamut changes
cannot be called sample-preserving merely because an estimator returns zero.

### HDR anchors and feature-independent refusal

```rust
use zenpixels::DiffuseWhite;
assert!(std::panic::catch_unwind(|| DiffuseWhite::new(f32::NAN)).is_err());
assert!(std::panic::catch_unwind(|| DiffuseWhite::new(0.0)).is_err());
let white = DiffuseWhite::new(203.0); // implemented, no spelling migration
```

```rust,ignore
// PROPOSED: same refusal with hdr-experimental enabled or disabled.
let hdr = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Pq);
let result = RowConverter::new(hdr, PixelDescriptor::RGB8_SRGB);
assert!(result.is_err()); // absent required source/target mapping policy

// Proposed explicit meaning for relative-linear values:
let relative_linear = [1.0f32, 1.0, 1.0];
let current_anchor = DiffuseWhite::new(203.0);
// Independent reference: those channels represent 203 nits, not source_peak
// or target_peak merely because all three parameters use the same units.
```

Keep measured maximum, percentile measurements, mastering peak and current anchor
separate. Algorithm changes need independently computed reference values, not two
paths through the same kernel. Unsupported mapping must refuse before pixels mutate.

### Ownership choices and refusal-before-mutation

```rust,ignore
// PROPOSED consuming identity: caller keeps the allocation through the result.
let parts = buffer.into_parts();
let pointer = parts.data.as_ptr();
let buffer = PixelBuffer::try_from_parts(parts).map_err(|e| e.without_buffer())?;
let result = identity_output_plan.into_output(buffer)?;
let (pixels, metadata) = result.into_parts();
assert_eq!(pixels.into_parts().data.as_ptr(), pointer);

// PROPOSED strict in-place: no widening fallback allocation, no partial mutation.
let before = buffer.as_slice().row(0).to_vec();
assert!(buffer.convert_in_place(&unsupported_in_place_plan).is_err());
assert_eq!(buffer.as_slice().row(0), before);

// Separate consuming operation may allocate; its name/docs must permit that:
let wider = buffer.into_converted(&widening_plan)?;
```

Actual method signatures remain a review decision. The first result must keep
matching metadata too. One-shot helper preparation costs must not be described
as allocation-free merely because the output image was supplied by the caller.

### Streaming: preserve source errors and don't invent an EOF row

```rust,ignore
// CURRENT problematic callback pattern:
encoder.encode_from(&mut |y, dst| -> usize {
    provider.fill(y, dst).unwrap_or(0) // an I/O/CMS error becomes EOF
})?;

// PROPOSED zencodec companion API (final spelling undecided):
encoder.try_encode_from(&mut |y, dst| provider.fill(y, dst))?;
// Test: a source error on row 2 is returned with its cause, not successful truncation.

// Existing zenpipe callback logic should follow this ordering:
let produced = callback(y, &mut reusable_row)?;
if !produced { return Ok(None); }
append_produced_row(&reusable_row);
```

The last block sketches the desired fallible protocol; today's boolean callback
has no Result. Reuse zencodec/zenpipe lending interfaces, preserving ICC/anchor per
frame. Row iterators over resident storage remain optional:

```rust,ignore
// CURRENT, already efficient:
for y in 0..view.rows() { consume(view.row(y)); }
// OPTIONAL proposed convenience; visible bytes, no final padding requirement:
for row in view.row_iter() { consume(row); }
```

### Video planes: requirements to test before designing the replacement

```rust,ignore
// CURRENT codec-owned data should remain owned by its codec:
let decoded = aom_decoder.decode(packet)?;
// Future borrowed view must retain significant bits (e.g. 10) separately from U16.
let view = decoded.as_video_planes()?; // PROPOSED adapter, not an existing method
assert_eq!(view.bit_depth(), 10);
assert_eq!(view.y().as_ptr(), decoded.y().as_ptr());
// AOM -> matching VMAF input should borrow; no RGB roundtrip or normalization.

// WRONG CVVDP input: unshifted 10-bit values in its normalized-U16 RGB entry point.
let maximum_10_bit = 1023u16;
assert!((maximum_10_bit as f32 / 65535.0) < 0.016);
// WANTED: range expansion + YCbCr matrix + chroma siting/filter + display RGB
// interpretation, with independently verified values; not merely left-shift U16.
```

No new video type or method is added now. Required negative cases include odd
chroma dimensions, nonzero crop phase, independent strides unsupported by a
backend, wrong significant bits, incorrect matrix/range, and unsynchronized
luma/chroma strips. Each must be checked/refused or explicitly transformed.

### No surprise cosmetic migration

```rust,ignore
// KEEP working across the designated bridge and 0.3:
use zenpixels_convert::orient::apply_orientation;
let rotated = apply_orientation(view, Orientation::Rotate90);

// KEEP the ability to implement existing open extension traits on local wrappers.
// Do not add a private supertrait or new required method between the two lines.

// OPTIONAL accurate naming, only if we adopt this cleanup with both names in 0.2:
// ByteOrder / byte_order -> ChannelOrder / channel_order.
// This concerns channel ordering, not changing sample endianness.
```

For docs.rs, retain import paths while selecting canonical documented re-exports:

```rust,ignore
#[doc(inline)]
pub use buffer::{PixelBuffer, PixelSlice, PixelSliceMut, PixelCow};
// Only hide implementation-module navigation after re-export pages and links work.
```

Compile-time work requires measurements, not API churn. No additional generic
color/policy taxonomy, unsafe POD implementation or public package split is approved.
Pair consumer builds with `-D deprecated` across both supported release lines and
inspect actual dependency unification; keep error matches and feature spellings.

The [consolidated proposal](zenpixels-0.2-and-0.3-review.md) remains the full scope
index; this document supplies the code and expected outcomes for choosing each
next implementation chunk. It does not authorize implementing all remaining cases
before review.

## Verification of this implementation chunk

- Core all-feature and minimal tests, including six new adoption/recovery tests and
  the new documentation examples, passed.
- Full workspace tests passed; the known-defect review suite is run separately.
- Strict core/all-feature and workspace/all-target Clippy passed.
- Rust 1.85 minimal core compilation and pinned public API snapshot generation passed.
- All 124 downstream deprecation checks passed (core and convert re-exports,
  inferred methods, estimation opt-in/off, default/minimal builds).
- The runnable code-review suite passed 26 cases in each of default and minimal
  configurations; this deliberately confirms current bugs, not proposed fixes.

The new adoption path validates storage using the existing view contract. General
color/descriptor consistency, empty-view repairs and the remaining conversion
changes above are not claimed complete by these results.
