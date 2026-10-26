use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, PixelFormat, TransferFunction,
};
use zenpixels_convert::{ConvertOptions, ConvertPlan, RowConverter};

#[test]
fn native_identity_is_bit_exact_but_unimplemented_transforms_refuse() {
    for ty in [
        ChannelType::I8,
        ChannelType::I16,
        ChannelType::U32,
        ChannelType::I32,
        ChannelType::U64,
        ChannelType::I64,
        ChannelType::F64,
    ] {
        for layout in [
            ChannelLayout::Gray,
            ChannelLayout::GrayAlpha,
            ChannelLayout::Rgb,
            ChannelLayout::Rgba,
        ] {
            let alpha = layout.has_alpha().then_some(AlphaMode::Straight);
            let from = PixelFormat::from_parts(ty, layout, alpha)
                .unwrap()
                .descriptor();
            let mut converter = RowConverter::new(from, from).unwrap();
            let n = from.bytes_per_pixel() * 3;
            let source: Vec<u8> = (0..n + 17).map(|i| (i * 73 + 19) as u8).collect();
            let mut out = vec![0xa5; n + 17];
            converter.try_convert_row(&source, &mut out, 3).unwrap();
            assert_eq!(&out[..n], &source[..n]);
            assert_eq!(&out[n..], &[0xa5; 17]);
            let target =
                PixelDescriptor::new(ChannelType::F32, layout, alpha, TransferFunction::Unknown);
            assert!(
                ConvertPlan::new(from, target).is_err(),
                "unqualified narrowing must refuse: {from}"
            );
            assert!(
                ConvertPlan::new(target, from).is_err(),
                "unqualified widening must refuse: {from}"
            );
            assert!(
                ConvertPlan::new(from, from.with_transfer(TransferFunction::Srgb)).is_err(),
                "no relabel-only transfer plan: {from}"
            );
            let changed = from.with_transfer(TransferFunction::Srgb);
            assert!(ConvertPlan::new_preserving_samples(from, changed).is_err());
            assert!(
                RowConverter::new_explicit_with_cms(
                    from,
                    changed,
                    &ConvertOptions::permissive(),
                    None
                )
                .is_err()
            );
            let changed = from.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
            assert!(
                RowConverter::new_explicit_with_cms(
                    from,
                    changed,
                    &ConvertOptions::permissive(),
                    None
                )
                .is_err()
            );
        }
    }
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn hdr_constructor_cannot_treat_native_identity_as_supported_math() {
    for format in [
        PixelFormat::RgbI16,
        PixelFormat::RgbU32,
        PixelFormat::RgbF64,
    ] {
        let desc = format.descriptor().with_transfer(TransferFunction::Linear);
        assert!(
            ConvertPlan::new_with_hdr_config(
                desc,
                desc,
                zenpixels_convert::HdrConfig::for_source_peak(1000.0)
            )
            .is_err()
        );
    }
}

#[test]
fn opaque_profiles_do_not_turn_native_samples_into_a_relabel() {
    use std::sync::Arc;
    use zenpixels::{ColorContext, ColorOrigin, PixelBuffer};
    use zenpixels_convert::{finalize_for_output_with, output::OutputProfile};

    for format in [
        PixelFormat::RgbI16,
        PixelFormat::RgbU32,
        PixelFormat::RgbF64,
    ] {
        let desc = format.descriptor();
        let source = vec![0x96; desc.bytes_per_pixel() * 2];
        let buffer = PixelBuffer::from_vec(source.clone(), 2, 1, desc)
            .unwrap()
            .with_color_context(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
                &b"unresolved-source-profile"[..],
            ))));
        assert!(
            finalize_for_output_with(
                &buffer,
                &ColorOrigin::assumed(),
                OutputProfile::Icc(Arc::from(&b"different-target-profile"[..])),
                format,
                None
            )
            .is_err()
        );
        assert_eq!(buffer.as_slice().row(0), source);
    }
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn moxcms_does_not_dispatch_native_samples_as_f32() {
    use zenpixels_convert::{cms::PluggableCms, cms_moxcms::MoxCms};
    for format in [
        PixelFormat::RgbI16,
        PixelFormat::RgbU32,
        PixelFormat::RgbF64,
    ] {
        let from = format
            .descriptor()
            .with_primaries(zenpixels::ColorPrimaries::DisplayP3)
            .color_profile_source();
        let to = format.descriptor().color_profile_source();
        assert!(
            MoxCms
                .build_source_transform(from, to, format, format, &ConvertOptions::permissive())
                .expect("unsupported sample types are diagnosed")
                .is_err()
        );
    }
}
