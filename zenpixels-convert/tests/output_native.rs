//! Qualify the combined native-sample and profile-aware output review APIs.
use std::sync::Arc;
use zenpixels::{ColorContext, ColorOrigin, PixelFormat, PixelSlice, PixelSliceMut};
use zenpixels_convert::{ConvertOptions, OutputPlan, OutputProfile};

#[test]
fn native_cmyk_identity_plan_preserves_float_bits_and_strides() {
    for format in [PixelFormat::CmykF32, PixelFormat::CmykF32Inverted] {
        let descriptor = format
            .descriptor()
            .with_primaries(zenpixels::ColorPrimaries::Unknown);
        // Opaque identity does not interpret the ICC or the sample bit patterns.
        // The real native ICC consumer is qualified separately in zencodec.
        let profile: Arc<[u8]> = Arc::from(&b"unchanged current CMYK profile"[..]);
        let context = Arc::new(ColorContext::from_icc(profile.clone()));
        let mut plan = OutputPlan::new(
            descriptor,
            Some(context.clone()),
            &ColorOrigin::from_icc(profile.clone()),
            OutputProfile::SameAsOrigin,
            format,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        plan.prepare(2).unwrap();
        let source = [
            0u32,
            0x8000_0000,
            1,
            0x7fc0_1234,
            0x3f80_0000,
            0x3e80_0000,
            0x3f00_0000,
            0x3f40_0000,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0x3f80_0000,
            1,
            0x8000_0000,
            0,
            0x7fc0_4321,
            0x3e80_0000,
            0x3f00_0000,
            0x3f40_0000,
        ];
        let mut destination = [0x5a5a_5a5au32; 24];
        let input = PixelSlice::new(bytemuck::cast_slice(&source), 2, 2, 48, descriptor)
            .unwrap()
            .with_color_context(context);
        let borrowed = plan.borrow_rows(input.clone()).unwrap().unwrap();
        assert_eq!(
            borrowed.as_strided_bytes().as_ptr(),
            input.as_strided_bytes().as_ptr()
        );
        assert_eq!(borrowed.as_strided_bytes(), input.as_strided_bytes());
        assert_eq!(borrowed.stride(), input.stride());
        assert_eq!(borrowed.color_context().unwrap().icc, Some(profile.clone()));
        let output = PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut destination),
            2,
            2,
            48,
            plan.output_descriptor(),
        )
        .unwrap();
        let converted = plan.convert_rows(input.clone(), output).unwrap();
        assert_eq!(
            converted.color_context().unwrap().icc,
            Some(profile.clone())
        );
        assert_eq!(plan.metadata().icc, Some(profile));
        assert_eq!(converted.row(0), input.row(0));
        assert_eq!(converted.row(1), input.row(1));
        assert_eq!(&destination[8..12], &[0x5a5a_5a5a; 4]);
        assert_eq!(&destination[20..24], &[0x5a5a_5a5a; 4]);
    }
}

#[test]
fn borrow_rows_refuses_changed_interpretation_and_keeps_real_conversion_available() {
    use zenpixels::{Cicp, PixelDescriptor};
    use zenpixels_convert::ConvertError;
    let descriptor = PixelDescriptor::RGB8_SRGB;
    let input = [7, 19, 121, 23, 51, 93];
    let source = PixelSlice::new(&input, 2, 1, 6, descriptor).unwrap();
    for format in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let mut plan = OutputPlan::new(
            descriptor,
            None,
            &ColorOrigin::assumed(),
            OutputProfile::Named(Cicp::SRGB),
            format,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        let changed = source
            .clone()
            .with_color_context(Arc::new(ColorContext::default()));
        assert_eq!(
            plan.borrow_rows(changed).unwrap_err().error(),
            &ConvertError::SourceInterpretationChanged
        );
        plan.prepare(1).unwrap();
        assert_eq!(
            plan.borrow_rows(source.clone()).unwrap_err().error(),
            &ConvertError::InvalidWidth(2)
        );
        plan.prepare(2).unwrap();
        let borrowed = plan.borrow_rows(source.clone()).unwrap();
        assert_eq!(borrowed.is_some(), format == PixelFormat::Rgb8);
        if let Some(view) = borrowed {
            assert_eq!(view.as_strided_bytes().as_ptr(), input.as_ptr());
            assert_eq!(view.color_context().unwrap().cicp, Some(Cicp::SRGB));
        } else {
            let mut output = [0u8; 8];
            let destination =
                PixelSliceMut::new(&mut output, 2, 1, 8, plan.output_descriptor()).unwrap();
            let converted = plan.convert_rows(source.clone(), destination).unwrap();
            assert_eq!(converted.row(0), &[7, 19, 121, 255, 23, 51, 93, 255]);
            assert_eq!(converted.color_context().unwrap().cicp, Some(Cicp::SRGB));
        }
    }
}

#[test]
fn unchanged_opaque_profile_alpha_addition_preserves_strided_color_without_a_cms() {
    use zenpixels::{ChannelType, ColorPrimaries, TransferFunction};
    use zenpixels_convert::cms::{CmsPluginError, PluggableCms, RowTransformMut};
    struct NoColorWork;
    impl PluggableCms for NoColorWork {
        fn build_source_transform(
            &self,
            _: zenpixels::ColorProfileSource<'_>,
            _: zenpixels::ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
            panic!("adding opaque alpha requested a color transform")
        }
    }
    for (from, to) in [
        (PixelFormat::Rgb8, PixelFormat::Rgba8),
        (PixelFormat::Rgb16, PixelFormat::Rgba16),
        (PixelFormat::RgbF32, PixelFormat::RgbaF32),
        (PixelFormat::Gray8, PixelFormat::GrayA8),
        (PixelFormat::Gray16, PixelFormat::GrayA16),
        (PixelFormat::GrayF32, PixelFormat::GrayAF32),
    ] {
        let descriptor = from
            .descriptor()
            .with_transfer(TransferFunction::Unknown)
            .with_primaries(ColorPrimaries::Unknown);
        let profile: Arc<[u8]> = Arc::from(&b"unchanged opaque current profile"[..]);
        let context = Arc::new(ColorContext::from_icc(profile.clone()));
        let bpp = descriptor.bytes_per_pixel();
        // Four pixel slots per row, two active. Keep native f32 bit patterns,
        // including negative zero, NaN payloads, subnormals and infinities.
        let words = [
            0x8000_0000u32,
            0x7fc0_1234,
            1,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0x7fc0_5678,
            0xff80_0000,
            0,
            0x3f00_0000,
            0x3e80_0000,
            0x8000_0000,
        ];
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let source = PixelSlice::new(bytes, 2, 2, bpp * 4, descriptor)
            .unwrap()
            .with_color_context(context.clone());
        for cms in [None, Some(&NoColorWork as &dyn PluggableCms)] {
            let mut plan = OutputPlan::new(
                descriptor,
                Some(context.clone()),
                &ColorOrigin::from_icc(profile.clone()),
                OutputProfile::SameAsOrigin,
                to,
                &ConvertOptions::retain_channels(),
                cms,
            )
            .unwrap();
            plan.prepare(2).unwrap();
            assert!(plan.borrow_rows(source.clone()).unwrap().is_none());
            let target_bpp = plan.output_descriptor().bytes_per_pixel();
            let stride = target_bpp * 3;
            let mut target = vec![0xa5a5_a5a5u32; (stride * 2).div_ceil(4)];
            let destination = PixelSliceMut::new(
                bytemuck::cast_slice_mut(&mut target),
                2,
                2,
                stride,
                plan.output_descriptor(),
            )
            .unwrap();
            let view = plan.convert_rows(source.clone(), destination).unwrap();
            assert_eq!(view.color_context(), Some(&context));
            assert_eq!(plan.metadata().icc, Some(profile.clone()));
            let alpha = match descriptor.channel_type() {
                ChannelType::U8 => vec![255],
                ChannelType::U16 => u16::MAX.to_ne_bytes().to_vec(),
                ChannelType::F32 => 1.0f32.to_ne_bytes().to_vec(),
                _ => unreachable!(),
            };
            for y in 0..2 {
                for (input, output) in source
                    .row(y)
                    .chunks_exact(bpp)
                    .zip(view.row(y).chunks_exact(target_bpp))
                {
                    assert_eq!(&output[..bpp], input, "{from:?} to {to:?}");
                    assert_eq!(&output[bpp..], alpha.as_slice());
                }
            }
            let all = bytemuck::cast_slice::<u32, u8>(&target);
            for y in 0..2 {
                assert!(
                    all[y * stride + 2 * target_bpp..(y + 1) * stride]
                        .iter()
                        .all(|b| *b == 0xa5)
                );
            }
        }
    }
}

#[test]
fn alpha_addition_does_not_bypass_changed_profile_or_required_depth_policy() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zenpixels::{ColorPrimaries, TransferFunction};
    use zenpixels_convert::{
        ConvertError,
        cms::{CmsPluginError, PluggableCms, RowTransformMut},
    };
    struct Decline(AtomicUsize);
    impl PluggableCms for Decline {
        fn build_source_transform(
            &self,
            _: zenpixels::ColorProfileSource<'_>,
            _: zenpixels::ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            None
        }
    }
    let descriptor = PixelFormat::RgbF32
        .descriptor()
        .with_primaries(ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let current: Arc<[u8]> = Arc::from(&b"opaque current ICC"[..]);
    let context = Arc::new(ColorContext::from_icc(current.clone()));
    let origin = ColorOrigin::from_icc(current);
    let cms = Decline(AtomicUsize::new(0));
    let error = OutputPlan::new(
        descriptor,
        Some(context.clone()),
        &origin,
        OutputProfile::Icc(Arc::from(&b"different target ICC"[..])),
        PixelFormat::RgbaF32,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    )
    .unwrap_err();
    assert!(matches!(error.error(), ConvertError::NeedsCms { .. }));
    assert_eq!(cms.0.load(Ordering::SeqCst), 1);
    let error = OutputPlan::new(
        descriptor,
        Some(context),
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgba8,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    )
    .unwrap_err();
    assert_eq!(error.error(), &ConvertError::DepthReductionForbidden);
    assert_eq!(cms.0.load(Ordering::SeqCst), 1);
}
