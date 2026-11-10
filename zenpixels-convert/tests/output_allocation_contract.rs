use zenpixels::*;
use zenpixels_convert::{ConvertOptions, EncodeReady, OutputPlan, OutputProfile};

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
        let plan = OutputPlan::new(
            buffer.descriptor(),
            None,
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
                EncodeReady::finalize(&buffer, &origin, target, PixelFormat::Rgb8, None).unwrap(),
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

/// `EncodeReady::finalize_owned` moves the consumed buffer's allocation on the
/// no-transform path — pointer identity, preserved stride, and byte/metadata
/// parity with `finalize`.
#[test]
fn finalize_owned_moves_source_allocation_on_identity() {
    for stride in [6, 10] {
        let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
            vec![19u8; stride * 2].into(),
            2,
            2,
            stride,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap();
        let src_ptr = buffer.as_slice().as_strided_bytes().as_ptr();
        let origin = ColorOrigin::assumed();
        let target = OutputProfile::Named(Cicp::SRGB);

        let mut owned = None;
        let info = allocation_counter::measure(|| {
            owned = Some(
                EncodeReady::finalize_owned(
                    buffer,
                    &origin,
                    target.clone(),
                    PixelFormat::Rgb8,
                    None,
                )
                .unwrap(),
            );
        });
        let owned = owned.unwrap();
        let out = owned.pixels();
        // Same allocation moved through: pointer and padded stride preserved.
        assert_eq!(out.as_strided_bytes().as_ptr(), src_ptr);
        assert_eq!(out.stride(), stride);

        // Parity with the allocating finalizer: descriptor, metadata, rows.
        let buffer2 = PixelBuffer::try_from_parts(PixelBufferParts::new(
            vec![19u8; stride * 2].into(),
            2,
            2,
            stride,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap();
        let mut reference = None;
        let ref_info = allocation_counter::measure(|| {
            reference = Some(
                EncodeReady::finalize(&buffer2, &origin, target.clone(), PixelFormat::Rgb8, None)
                    .unwrap(),
            );
        });
        let reference = reference.unwrap();
        // The identity path saves the whole pixel-buffer allocation; what
        // remains is OutputPlan construction shared with `finalize`.
        assert!(
            ref_info.bytes_total - info.bytes_total >= 12,
            "identity saved only {} bytes",
            ref_info.bytes_total - info.bytes_total
        );
        assert_eq!(out.descriptor(), reference.pixels().descriptor());
        assert_eq!(out.color_context(), reference.pixels().color_context());
        assert_eq!(owned.metadata().icc, reference.metadata().icc);
        assert_eq!(owned.metadata().cicp, reference.metadata().cicp);
        for y in 0..2 {
            assert_eq!(out.row(y), reference.pixels().row(y));
        }
    }
}

/// When a transform is required, `finalize_owned` allocates exactly like
/// `finalize` and produces identical pixels/metadata.
#[test]
fn finalize_owned_allocates_and_matches_when_transform_required() {
    let buffer = PixelBuffer::try_new(2, 2, PixelDescriptor::RGB8_SRGB).unwrap();
    let origin = ColorOrigin::assumed();
    let target = OutputProfile::Named(Cicp::SRGB);

    let owned =
        EncodeReady::finalize_owned(buffer, &origin, target.clone(), PixelFormat::Rgba8, None)
            .unwrap();
    let buffer2 = PixelBuffer::try_new(2, 2, PixelDescriptor::RGB8_SRGB).unwrap();
    let reference =
        EncodeReady::finalize(&buffer2, &origin, target, PixelFormat::Rgba8, None).unwrap();
    assert_eq!(owned.pixels().descriptor(), reference.pixels().descriptor());
    assert_eq!(owned.metadata().icc, reference.metadata().icc);
    assert_eq!(owned.metadata().cicp, reference.metadata().cicp);
    for y in 0..2 {
        assert_eq!(owned.pixels().row(y), reference.pixels().row(y));
    }
}

/// Completing interpretation from the same authoritative ICC must preserve
/// offsets, padding and every sample bit, including float alpha payloads.
#[test]
fn finalize_owned_preserves_offset_and_wide_samples_when_completing_icc() {
    use std::sync::Arc;
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
        let reference =
            EncodeReady::finalize(&buffer, &origin, OutputProfile::SameAsOrigin, format, None)
                .unwrap();
        let owned =
            EncodeReady::finalize_owned(buffer, &origin, OutputProfile::SameAsOrigin, format, None)
                .unwrap();
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
