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
