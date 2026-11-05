//! Exact output is a proof obligation, independent of color conversion presets.
use std::sync::Arc;
use zenpixels_convert::{
    AlphaMode, Cicp, ColorContext, ColorOrigin, ConvertError, OutputPlan, OutputProfile,
    PixelDescriptor, PixelSlice, PixelSliceMut, TransferFunction,
};

#[test]
fn exact_output_widens_strided_samples_and_keeps_padding() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::RGB16_SRGB;
    let mut plan = OutputPlan::new_preserving_samples(
        from,
        None,
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::SRGB),
        to,
    )
    .unwrap();
    plan.prepare(2).unwrap();
    let source = [0, 1, 127, 128, 254, 255, 99, 99, 99, 13, 24, 35, 46, 57, 68];
    let mut target = [0xa55au16; 18];
    let rows = plan
        .convert_rows(
            PixelSlice::new(&source, 2, 2, 9, from).unwrap(),
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut target), 2, 2, 18, to).unwrap(),
        )
        .unwrap();
    assert_eq!(rows.color_context().unwrap().cicp, Some(Cicp::SRGB));
    for y in 0..2 {
        for x in 0..6 {
            assert_eq!(target[y * 9 + x], u16::from(source[y * 9 + x]) * 257);
        }
        assert_eq!(&target[y * 9 + 6..y * 9 + 9], &[0xa55a; 3]);
    }
}

#[test]
fn opaque_icc_identity_and_alpha_expansion_preserve_float_payloads() {
    let from = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let icc: Arc<[u8]> = Arc::from(&b"opaque current profile"[..]);
    let context = Arc::new(ColorContext::from_icc(icc.clone()));
    let samples = [f32::from_bits(0x7fc1_2345), -0.0, -1.5];
    let source = PixelSlice::new(bytemuck::cast_slice(&samples), 1, 1, 12, from)
        .unwrap()
        .with_color_context(context.clone());
    let mut identity = OutputPlan::new_preserving_samples(
        from,
        Some(context.clone()),
        &ColorOrigin::assumed(),
        OutputProfile::Icc(icc.clone()),
        from,
    )
    .unwrap();
    identity.prepare(1).unwrap();
    let borrowed = identity.borrow_rows(source.clone()).unwrap().unwrap();
    assert_eq!(borrowed.row(0).as_ptr(), source.row(0).as_ptr());
    assert_eq!(borrowed.row(0), source.row(0));
    let to = PixelDescriptor::RGBAF32_LINEAR
        .with_primaries(from.primaries)
        .with_transfer(from.transfer());
    let mut plan = OutputPlan::new_preserving_samples(
        from,
        Some(context),
        &ColorOrigin::assumed(),
        OutputProfile::Icc(icc),
        to,
    )
    .unwrap();
    plan.prepare(1).unwrap();
    let mut target = [0f32; 4];
    plan.convert_rows(
        source,
        PixelSliceMut::new(bytemuck::cast_slice_mut(&mut target), 1, 1, 16, to).unwrap(),
    )
    .unwrap();
    assert_eq!(
        target[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        samples.map(f32::to_bits)
    );
    assert_eq!(target[3].to_bits(), 1f32.to_bits());
}

#[test]
fn exact_output_refuses_color_math_depth_loss_and_reassociation() {
    let linear = Cicp::new(1, 8, 0, true);
    let cases = [
        (
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb),
            Cicp::SRGB,
        ),
        (
            PixelDescriptor::RGB16_SRGB,
            PixelDescriptor::RGB8_SRGB,
            Cicp::SRGB,
        ),
        (
            PixelDescriptor::RGBAF32_LINEAR,
            PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied)),
            linear,
        ),
        (
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            Cicp::SRGB,
        ),
    ];
    for (from, to, color) in cases {
        assert!(
            OutputPlan::new_preserving_samples(
                from,
                None,
                &ColorOrigin::assumed(),
                OutputProfile::Named(color),
                to
            )
            .is_err(),
            "accepted {from:?} -> {to:?}"
        );
    }
    let descriptor = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let error = OutputPlan::new_preserving_samples(
        descriptor,
        Some(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
            &b"first"[..],
        )))),
        &ColorOrigin::assumed(),
        OutputProfile::Icc(Arc::from(&b"second"[..])),
        descriptor,
    )
    .unwrap_err();
    assert!(matches!(error.error(), ConvertError::NoPath { .. }));
}

/// imazen/zenpixels#94: a decoded premultiplied JPEG XL image, with only an ICC profile to describe
/// its color, kept as it is. No CMS exists here, and none is needed.
#[test]
fn premultiplied_icc_samples_keep_every_bit_without_a_cms() {
    let from = PixelDescriptor::RGBAF32_LINEAR
        .with_alpha(Some(AlphaMode::Premultiplied))
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let icc: Arc<[u8]> = Arc::from(&b"profile no parser has to read"[..]);
    let context = Arc::new(ColorContext::from_icc(icc.clone()));
    let origin = ColorOrigin::from_icc(icc.clone());
    // Two rows, stride 32: a padding pixel, and a NaN payload and -0.0 among the samples.
    let samples = [
        f32::from_bits(0x7fc1_2345),
        -0.0,
        0.25,
        0.5,
        9.0,
        9.0,
        9.0,
        9.0,
        f32::from_bits(0x7fa5_5a5a),
        1.5,
        -0.125,
        0.75,
    ];
    let source = PixelSlice::new(bytemuck::cast_slice(&samples), 1, 2, 32, from)
        .unwrap()
        .with_color_context(context.clone());
    for profile in [OutputProfile::SameAsOrigin, OutputProfile::Icc(icc.clone())] {
        let mut plan =
            OutputPlan::new_preserving_samples(from, Some(context.clone()), &origin, profile, from)
                .unwrap();
        plan.prepare(1).unwrap();
        assert_eq!(plan.output_descriptor(), from);

        // Borrowed: the same memory, with the current profile.
        let view = plan.borrow_rows(source.clone()).unwrap().unwrap();
        assert_eq!(view.row(0).as_ptr(), source.row(0).as_ptr());
        assert_eq!(view.stride(), 32);
        assert_eq!(view.descriptor(), from);
        assert_eq!(view.color_context().unwrap().icc.as_deref(), Some(&*icc));

        // Copied into a strided destination: exact bits, and its padding untouched.
        let mut target = [f32::from_bits(0x1234_5678); 16];
        let rows = plan
            .convert_rows(
                source.clone(),
                PixelSliceMut::new(bytemuck::cast_slice_mut(&mut target), 1, 2, 48, from).unwrap(),
            )
            .unwrap();
        assert_eq!(rows.descriptor(), from);
        assert_eq!(rows.color_context().unwrap().icc.as_deref(), Some(&*icc));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&target[..4]), bits(&samples[..4]));
        assert_eq!(bits(&target[4..12]), [0x1234_5678; 8]);
        assert_eq!(bits(&target[12..]), bits(&samples[8..]));
    }

    // The format-only constructor still can't express this, and refuses rather than relabeling.
    let error = OutputPlan::new(
        from,
        Some(context),
        &origin,
        OutputProfile::SameAsOrigin,
        from.pixel_format(),
        &zenpixels_convert::ConvertOptions::retain_channels(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(error.error(), ConvertError::NeedsCms { .. }),
        "{error}"
    );
}
