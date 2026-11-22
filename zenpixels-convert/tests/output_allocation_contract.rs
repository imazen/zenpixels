use zenpixels::*;
use zenpixels_convert::{ConvertOptions, OutputProfile};

#[test]
fn borrowed_encoder_rows_avoid_pixel_allocation_and_match_owned_finalization() {
    for stride in [6, 10] {
        let source = vec![19u8; stride * 2];
        let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
            source.into(),
            2,
            2,
            stride,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap();
        let origin = ColorOrigin::assumed();
        let target = OutputProfile::Named(Cicp::SRGB);
        let plan = crate::test_conversion::output_plan(
            buffer.encoding(),
            &origin,
            target.clone(),
            PixelFormat::Rgb8,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        let mut borrowed = None;
        let info = allocation_counter::measure(|| {
            borrowed = plan.borrow_rows(buffer.as_slice()).unwrap();
        });
        assert_eq!(info.bytes_total, 0);
        let rows = borrowed.unwrap();
        assert_eq!(
            rows.as_strided_bytes().as_ptr(),
            buffer.as_slice().as_strided_bytes().as_ptr()
        );
        assert_eq!(rows.stride(), stride);
        let mut owned = None;
        let info = allocation_counter::measure(|| {
            owned = Some(
                crate::test_conversion::copy_output(
                    &buffer,
                    &origin,
                    target,
                    PixelFormat::Rgb8,
                    None,
                )
                .unwrap(),
            );
        });
        assert!(info.bytes_total >= 12);
        let owned = owned.unwrap();
        assert_ne!(
            owned.pixels().as_strided_bytes().as_ptr(),
            rows.as_strided_bytes().as_ptr()
        );
        assert_eq!(owned.pixels().descriptor(), rows.descriptor());
        assert_eq!(owned.metadata().icc, plan.metadata().icc);
        assert_eq!(owned.metadata().cicp, plan.metadata().cicp);
        for y in 0..2 {
            assert_eq!(rows.row(y), owned.pixels().row(y));
        }
    }
}

/// Completing interpretation from the same authoritative ICC via `EncodeReady::from_owned`
/// must preserve non-zero byte offsets, inter-row padding, and every sample bit (including
/// quiet-NaN payload bits `0x7fc0_1234` in `RgbaF32`) with zero pixel allocation (#69).
#[test]
fn from_owned_preserves_offset_and_wide_samples_when_completing_icc() {
    use std::sync::Arc;
    use zenpixels_convert::{ConversionSpec, EncodeReady, OutputPlan, OutputTarget};

    for format in [
        PixelFormat::Rgba8,
        PixelFormat::Rgba16,
        PixelFormat::RgbaF32,
    ] {
        let descriptor = format
            .descriptor()
            .with_primaries(ColorPrimaries::Unknown)
            .with_transfer(TransferFunction::Unknown);
        let bpp = descriptor.bytes_per_pixel();
        let offset = bpp;
        let stride = 3 * bpp;
        // Minimal final-row extent, with a prefix and inter-row padding.
        let data = vec![0x7fc0_1234u32; (offset + stride + 2 * bpp) / 4];
        let profile: Arc<[u8]> = Arc::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        let mut parts = PixelBufferParts::new(data.into(), 2, 2, stride, descriptor);
        parts.offset = offset;
        parts.color_context = Some(Arc::new(ColorContext::from_icc(profile.clone())));
        let buffer = PixelBuffer::try_from_parts(parts).unwrap();
        let ptr = buffer.as_slice().as_strided_bytes().as_ptr();
        let bytes = buffer.as_slice().as_strided_bytes().to_vec();
        let origin = ColorOrigin::from_icc(profile);

        let reference = crate::test_conversion::copy_output(
            &buffer,
            &origin,
            OutputProfile::SameAsOrigin,
            format,
            None,
        )
        .unwrap();

        let plan = OutputPlan::build(
            buffer.encoding(),
            OutputTarget::new(OutputProfile::SameAsOrigin, format).with_origin(&origin),
            &ConversionSpec::new(ConvertOptions::retain_channels()),
        )
        .unwrap();
        let mut owned = None;
        let allocs = allocation_counter::measure(|| {
            owned = Some(EncodeReady::from_owned(buffer, plan).unwrap());
        });
        assert_eq!(allocs.bytes_total, 0);
        let owned = owned.unwrap();

        let view = owned.pixels();
        assert_eq!(view.as_strided_bytes().as_ptr(), ptr);
        assert_eq!(view.as_strided_bytes(), bytes);
        assert_eq!(view.stride(), stride);
        assert_eq!(view.descriptor(), reference.pixels().descriptor());
        assert_eq!(view.descriptor().primaries, ColorPrimaries::DisplayP3);
        assert_eq!(view.descriptor().transfer(), TransferFunction::Srgb);
        assert_eq!(view.color_context(), reference.pixels().color_context());
        assert_eq!(owned.metadata().icc, reference.metadata().icc);
        assert_eq!(owned.metadata().cicp, reference.metadata().cicp);
        for y in 0..2 {
            assert_eq!(view.row(y), reference.pixels().row(y));
        }
        assert_eq!(owned.into_parts().0.into_parts().offset, offset);
    }
}

#[path = "support/conversion.rs"]
mod test_conversion;
