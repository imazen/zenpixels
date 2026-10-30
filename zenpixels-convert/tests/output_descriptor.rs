//! Required output interpretation is checked before constructing a transform.
use std::sync::Arc;
use zenpixels::{
    AlphaMode, Cicp, ColorContext, ColorOrigin, ColorPrimaries, PixelDescriptor, PixelFormat,
    PixelSlice, PixelSliceMut, SignalRange, TransferFunction,
};
use zenpixels_convert::{
    ConvertError, ConvertOptions, OutputPlan, OutputProfile,
    cms::{CmsPluginError, PluggableCms, RowTransformMut},
};

struct NoCms;
impl PluggableCms for NoCms {
    fn build_source_transform(
        &self,
        _: zenpixels::ColorProfileSource<'_>,
        _: zenpixels::ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
        panic!("identity and contradictory requirements must not invoke a CMS")
    }
}

#[test]
fn current_icc_resolves_unknown_descriptor_without_changing_same_profile_samples() {
    let profile: Arc<[u8]> = Arc::from(zenpixels_convert::icc_profiles::display_p3_icc(false));
    let context = Arc::new(ColorContext::from_icc(profile.clone()));
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba16,
        PixelFormat::RgbF32,
        PixelFormat::RgbaF32,
    ] {
        for unknown in [1, 2, 3] {
            let target = format
                .descriptor()
                .with_primaries(ColorPrimaries::DisplayP3)
                .with_transfer(TransferFunction::Srgb);
            let source = target
                .with_primaries(if unknown & 1 != 0 {
                    ColorPrimaries::Unknown
                } else {
                    target.primaries
                })
                .with_transfer(if unknown & 2 != 0 {
                    TransferFunction::Unknown
                } else {
                    target.transfer()
                });
            let mut plan = OutputPlan::new_with_descriptor(
                source,
                Some(context.clone()),
                &ColorOrigin::assumed(),
                OutputProfile::Icc(profile.clone()),
                target,
                &ConvertOptions::retain_channels(),
                Some(&NoCms),
            )
            .unwrap();
            let stride = 4 * source.bytes_per_pixel();
            let words = [
                0x7fc01234u32,
                0x80000000,
                0x00000001,
                0x40200000,
                0xbf800000,
                0x3e800000,
            ]
            .repeat(16);
            let input = PixelSlice::new(bytemuck::cast_slice(&words), 3, 2, stride, source)
                .unwrap()
                .with_color_context(context.clone());
            plan.prepare(3).unwrap();
            let borrowed = plan.borrow_rows(input.clone()).unwrap().unwrap();
            assert_eq!(borrowed.descriptor(), target);
            assert_eq!(borrowed.color_context(), Some(&context));
            assert_eq!(
                borrowed.as_strided_bytes().as_ptr(),
                input.as_strided_bytes().as_ptr()
            );
            assert_eq!(borrowed.as_strided_bytes(), input.as_strided_bytes());
            let mut destination = vec![0xa5a5a5a5u32; stride * 2 / 4];
            let output = PixelSliceMut::new(
                bytemuck::cast_slice_mut(&mut destination),
                3,
                2,
                stride,
                target,
            )
            .unwrap();
            let converted = plan.convert_rows(input.clone(), output).unwrap();
            for y in 0..2 {
                assert_eq!(converted.row(y), input.row(y));
            }
            for row in bytemuck::cast_slice::<u32, u8>(&destination).chunks_exact(stride) {
                assert!(
                    row[3 * source.bytes_per_pixel()..]
                        .iter()
                        .all(|b| *b == 0xa5)
                );
            }
            // Source facts remain bound to the original descriptor/context.
            let changed = input.with_color_context(Arc::new(ColorContext::default()));
            assert_eq!(
                plan.borrow_rows(changed).unwrap_err().error(),
                &ConvertError::SourceInterpretationChanged
            );
        }
    }
}

#[test]
fn required_association_borrows_and_copies_same_opaque_profile_exactly() {
    let profile: Arc<[u8]> = Arc::from(&b"opaque current profile, not interpreted for copying"[..]);
    let context = Arc::new(ColorContext::from_icc(profile.clone()));
    for format in [
        PixelFormat::Rgba8,
        PixelFormat::Rgba16,
        PixelFormat::RgbaF32,
    ] {
        for alpha in [
            AlphaMode::Straight,
            AlphaMode::Premultiplied,
            AlphaMode::Undefined,
        ] {
            let descriptor = format
                .descriptor()
                .with_primaries(ColorPrimaries::Unknown)
                .with_transfer(TransferFunction::Unknown)
                .with_alpha(Some(alpha));
            let mut plan = OutputPlan::new_with_descriptor(
                descriptor,
                Some(context.clone()),
                &ColorOrigin::from_icc(profile.clone()),
                OutputProfile::SameAsOrigin,
                descriptor,
                &ConvertOptions::retain_channels(),
                Some(&NoCms),
            )
            .unwrap();
            assert_eq!(plan.output_descriptor(), descriptor);
            let stride = 3 * descriptor.bytes_per_pixel();
            let source = [
                0u32,
                0x8000_0000,
                1,
                0x7fc0_1234,
                0x3e80_0000,
                0x7fc0_4321,
                0x3f80_0000,
                0x3e00_0000,
            ]
            .repeat(3);
            let input = PixelSlice::new(bytemuck::cast_slice(&source), 2, 2, stride, descriptor)
                .unwrap()
                .with_color_context(context.clone());
            plan.prepare(2).unwrap();
            let borrowed = plan.borrow_rows(input.clone()).unwrap().unwrap();
            assert_eq!(
                borrowed.as_strided_bytes().as_ptr(),
                input.as_strided_bytes().as_ptr()
            );
            assert_eq!(borrowed.descriptor(), descriptor);
            assert_eq!(borrowed.color_context(), Some(&context));
            let mut output = vec![0xa5a5_a5a5u32; stride * 2 / 4];
            let destination = PixelSliceMut::new(
                bytemuck::cast_slice_mut(&mut output),
                2,
                2,
                stride,
                descriptor,
            )
            .unwrap();
            let converted = plan.convert_rows(input.clone(), destination).unwrap();
            assert_eq!(converted.descriptor(), descriptor);
            assert_eq!(converted.color_context(), Some(&context));
            for y in 0..2 {
                assert_eq!(converted.row(y), input.row(y));
            }
            let output = bytemuck::cast_slice::<u32, u8>(&output);
            for y in 0..2 {
                assert!(
                    output[y * stride + 2 * descriptor.bytes_per_pixel()..(y + 1) * stride]
                        .iter()
                        .all(|b| *b == 0xa5)
                );
            }
            let changed = input
                .clone()
                .with_color_context(Arc::new(ColorContext::default()));
            assert_eq!(
                plan.borrow_rows(changed).unwrap_err().error(),
                &ConvertError::SourceInterpretationChanged
            );
            plan.prepare(1).unwrap();
            assert_eq!(
                plan.borrow_rows(input).unwrap_err().error(),
                &ConvertError::InvalidWidth(2)
            );
        }
    }
}

#[test]
fn required_profile_descriptor_conflicts_refuse_before_cms_construction() {
    let source = PixelDescriptor::RGB8_SRGB;
    for requested in [
        source.with_primaries(ColorPrimaries::Bt2020),
        source.with_transfer(TransferFunction::Linear),
        source.with_signal_range(SignalRange::Narrow),
    ] {
        let error = OutputPlan::new_with_descriptor(
            source,
            None,
            &ColorOrigin::assumed(),
            OutputProfile::Named(Cicp::SRGB),
            requested,
            &ConvertOptions::retain_channels(),
            Some(&NoCms),
        )
        .unwrap_err();
        assert_eq!(
            error.error(),
            &ConvertError::OutputInterpretationMismatch {
                requested,
                profile: source
            }
        );
    }
    let private: Arc<[u8]> = Arc::from(&b"private opaque target profile"[..]);
    let error = OutputPlan::new_with_descriptor(
        source,
        None,
        &ColorOrigin::assumed(),
        OutputProfile::Icc(private),
        source,
        &ConvertOptions::retain_channels(),
        Some(&NoCms),
    )
    .unwrap_err();
    assert!(matches!(
        error.error(),
        ConvertError::OutputInterpretationMismatch { .. }
    ));
    assert!(!format!("{error:?} {error}").contains("private opaque target profile"));
}

#[test]
fn required_association_does_not_change_format_convenience_or_bypass_sample_policy() {
    let source = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
    let profile = OutputProfile::Named(Cicp::SRGB);
    let options = ConvertOptions::retain_channels();
    let mut convenience = OutputPlan::new(
        source,
        None,
        &ColorOrigin::assumed(),
        profile.clone(),
        PixelFormat::Rgba8,
        &options,
        None,
    )
    .unwrap();
    assert_eq!(convenience.output_descriptor(), PixelDescriptor::RGBA8_SRGB);
    let mut explicit = OutputPlan::new_with_descriptor(
        source,
        None,
        &ColorOrigin::assumed(),
        profile.clone(),
        PixelDescriptor::RGBA8_SRGB,
        &options,
        None,
    )
    .unwrap();
    let input = [64, 32, 16, 128, 0xa5, 0xa5, 0xa5, 0xa5, 24, 12, 6, 64];
    let pixels = PixelSlice::new(&input, 1, 2, 8, source).unwrap();
    let mut a = [0x51; 16];
    let mut b = [0x51; 16];
    convenience
        .convert_rows(
            pixels.clone(),
            PixelSliceMut::new(&mut a, 1, 2, 8, PixelDescriptor::RGBA8_SRGB).unwrap(),
        )
        .unwrap();
    explicit
        .convert_rows(
            pixels,
            PixelSliceMut::new(&mut b, 1, 2, 8, PixelDescriptor::RGBA8_SRGB).unwrap(),
        )
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(&a[..4], &[128, 64, 32, 128]);
    assert_eq!(&a[4..8], &[0x51; 4]);
    assert_eq!(&a[12..], &[0x51; 4]);

    let error = OutputPlan::new_with_descriptor(
        PixelDescriptor::RGBA8_SRGB,
        None,
        &ColorOrigin::assumed(),
        profile.clone(),
        PixelDescriptor::RGB8_SRGB,
        &options,
        Some(&NoCms),
    )
    .unwrap_err();
    assert_eq!(error.error(), &ConvertError::AlphaRemovalForbidden);
    let error = OutputPlan::new_with_descriptor(
        PixelDescriptor::RGBA8_SRGB,
        None,
        &ColorOrigin::assumed(),
        profile,
        PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Opaque)),
        &options,
        Some(&NoCms),
    )
    .unwrap_err();
    assert_eq!(error.error(), &ConvertError::AlphaCheckRequired);
}

#[test]
fn required_descriptor_retains_range_identity_but_does_not_authorize_rescaling() {
    let cicp = Cicp::new(1, 13, 0, false);
    let descriptor = cicp.try_to_descriptor(PixelFormat::Rgb8).unwrap();
    let context = Arc::new(ColorContext::from_cicp(cicp));
    let mut plan = OutputPlan::new_with_descriptor(
        descriptor,
        Some(context.clone()),
        &ColorOrigin::from_cicp(cicp),
        OutputProfile::Named(cicp),
        descriptor,
        &ConvertOptions::retain_channels(),
        Some(&NoCms),
    )
    .unwrap();
    let input = [0, 16, 235, 255, 17, 234];
    let pixels = PixelSlice::new(&input, 2, 1, 6, descriptor)
        .unwrap()
        .with_color_context(context);
    let mut output = [0; 6];
    plan.convert_rows(
        pixels,
        PixelSliceMut::new(&mut output, 2, 1, 6, descriptor).unwrap(),
    )
    .unwrap();
    assert_eq!(output, input);
    assert_eq!(plan.metadata().cicp, Some(cicp));
    let error = OutputPlan::new_with_descriptor(
        PixelDescriptor::RGB8_SRGB,
        None,
        &ColorOrigin::assumed(),
        OutputProfile::Named(cicp),
        descriptor,
        &ConvertOptions::retain_channels(),
        Some(&NoCms),
    )
    .unwrap_err();
    assert!(matches!(error.error(), ConvertError::NoPath { .. }));
}

#[test]
fn matching_required_descriptors_do_not_authorize_relabeling_different_opaque_profiles() {
    let descriptor = PixelFormat::RgbaF32
        .descriptor()
        .with_primaries(ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown)
        .with_alpha(Some(AlphaMode::Premultiplied));
    let source: Arc<[u8]> = Arc::from(&b"original profile"[..]);
    let target: Arc<[u8]> = Arc::from(&b"different profile"[..]);
    let error = OutputPlan::new_with_descriptor(
        descriptor,
        Some(Arc::new(ColorContext::from_icc(source.clone()))),
        &ColorOrigin::from_icc(source),
        OutputProfile::Icc(target),
        descriptor,
        &ConvertOptions::retain_channels(),
        None,
    )
    .unwrap_err();
    assert!(matches!(error.error(), ConvertError::NeedsCms { .. }));
}

#[test]
fn required_premultiplication_executes_exact_linear_math_and_rejects_wrong_destination() {
    let cicp = Cicp::new(1, 8, 0, true);
    let source = cicp.try_to_descriptor(PixelFormat::RgbaF32).unwrap();
    let required = source.with_alpha(Some(AlphaMode::Premultiplied));
    let mut plan = OutputPlan::new_with_descriptor(
        source,
        None,
        &ColorOrigin::from_cicp(cicp),
        OutputProfile::Named(cicp),
        required,
        &ConvertOptions::retain_channels().with_clip_out_of_gamut(false),
        Some(&NoCms),
    )
    .unwrap();
    let input = [
        0.5f32, 0.25, -0.5, 0.5, 77.0, 78.0, 79.0, 80.0, 1.5, 0.0, 2.0, 0.25,
    ];
    let pixels = PixelSlice::new(bytemuck::cast_slice(&input), 1, 2, 32, source).unwrap();
    assert!(plan.borrow_rows(pixels.clone()).unwrap().is_none());
    let mut output = [0xa5a5_a5a5u32; 16];
    let wrong =
        PixelSliceMut::new(bytemuck::cast_slice_mut(&mut output), 1, 2, 32, source).unwrap();
    assert!(plan.convert_rows(pixels.clone(), wrong).is_err());
    assert_eq!(output, [0xa5a5_a5a5; 16]);
    let target =
        PixelSliceMut::new(bytemuck::cast_slice_mut(&mut output), 1, 2, 32, required).unwrap();
    let converted = plan.convert_rows(pixels, target).unwrap();
    assert_eq!(converted.descriptor(), required);
    assert_eq!(converted.color_context().unwrap().cicp, Some(cicp));
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(converted.row(0)),
        &[0.25, 0.125, -0.25, 0.5]
    );
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(converted.row(1)),
        &[0.375, 0.0, 0.5, 0.25]
    );
    assert_eq!(&output[4..8], &[0xa5a5_a5a5; 4]);
    assert_eq!(&output[12..16], &[0xa5a5_a5a5; 4]);
}
