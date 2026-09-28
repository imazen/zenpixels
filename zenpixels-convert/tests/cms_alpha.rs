//! A CMS consumes straight samples; alpha association belongs around it.
use zenpixels_convert::{
    AlphaMode, ColorPrimaries, PixelDescriptor, RowConverter, policy::ConvertOptions,
};

#[test]
fn nonlinear_transfer_operates_on_unassociated_samples() {
    use zenpixels_convert::{ConvertPlan, TransferFunction};
    let from = PixelDescriptor::RGBAF32_LINEAR
        .with_transfer(TransferFunction::Srgb)
        .with_alpha(Some(AlphaMode::Premultiplied));
    let input = [0.25_f32, 0.125, 0.0, 0.5];
    // Independent sRGB EOTF at unassociated values 0.5 and 0.25.
    let linear = [0.21404114_f32, 0.05087609, 0.0, 0.5];
    for target_alpha in [AlphaMode::Straight, AlphaMode::Premultiplied] {
        let to = PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(target_alpha));
        assert!(ConvertPlan::new(from, to).is_err());
        let mut converter = RowConverter::new(from, to).unwrap();
        let mut output = [0.0_f32; 4];
        converter.convert_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut output),
            1,
        );
        for channel in 0..3 {
            let expected = linear[channel]
                * if target_alpha == AlphaMode::Premultiplied {
                    0.5
                } else {
                    1.0
                };
            // Match the existing transfer review's 1e-5 bound: the production
            // sRGB polynomial is approximate. The former alpha-domain error
            // produces 0.101752 here instead of approximately 0.214041.
            assert!(
                (output[channel] - expected).abs() < 1e-5,
                "{target_alpha:?} channel {channel}: {} != {expected}",
                output[channel]
            );
        }
        assert_eq!(output[3], 0.5);
    }
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn tone_mapping_rejects_associated_source_samples() {
    use zenpixels_convert::ConvertPlan;
    let from = PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied));
    assert!(ConvertPlan::new_with_hdr_peak(from, PixelDescriptor::RGBA8_SRGB, 1000.0).is_err());
}

#[test]
fn cross_gamut_preserves_partial_alpha_and_cloned_transforms() {
    let from = PixelDescriptor::RGBAF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
    let to = PixelDescriptor::RGBAF32_LINEAR;
    let mut options = ConvertOptions::permissive();
    options.clip_out_of_gamut = false;
    let straight = [
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.8, 0.3, 0.1, 0.25],
        [0.1, 0.7, 0.2, 0.5],
        [1.0, 0.0, 0.0, 1.0],
    ];
    let mut reference = [[0.0_f32; 4]; 4];
    RowConverter::new_explicit(from, to, &options)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&straight),
            bytemuck::cast_slice_mut(&mut reference),
            4,
        );
    assert!((reference[3][0] - straight[3][0]).abs() > 0.01);
    for source_alpha in [AlphaMode::Straight, AlphaMode::Premultiplied] {
        for target_alpha in [AlphaMode::Straight, AlphaMode::Premultiplied] {
            let mut source = straight;
            let mut expected = reference;
            for pixel in &mut source {
                if source_alpha == AlphaMode::Premultiplied {
                    let alpha = pixel[3];
                    for channel in &mut pixel[..3] {
                        *channel *= alpha;
                    }
                }
            }
            for pixel in &mut expected {
                if target_alpha == AlphaMode::Premultiplied {
                    let alpha = pixel[3];
                    for channel in &mut pixel[..3] {
                        *channel *= alpha;
                    }
                }
            }
            let converter = RowConverter::new_explicit(
                from.with_alpha(Some(source_alpha)),
                to.with_alpha(Some(target_alpha)),
                &options,
            )
            .unwrap();
            for mut converter in [converter.clone(), converter] {
                let mut output = [[0.0_f32; 4]; 4];
                // Repeated and changing-width rows exercise scratch reuse.
                for width in [4, 1, 4] {
                    converter.convert_row(
                        bytemuck::cast_slice(&source),
                        bytemuck::cast_slice_mut(&mut output),
                        width,
                    );
                    for (actual, expected) in output[..width as usize]
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                    {
                        assert!(
                            (actual - expected).abs() < 2e-6,
                            "{source_alpha:?}->{target_alpha:?}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cross_gamut_does_not_treat_padding_as_coverage() {
    let from = PixelDescriptor::RGBAF32_LINEAR
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_alpha(Some(AlphaMode::Undefined));
    assert!(RowConverter::new(from, PixelDescriptor::RGBAF32_LINEAR).is_err());
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn premultiplied_icc_finalization_uses_current_profile() {
    use std::sync::Arc;
    use zenpixels_convert::{
        Cicp, ColorContext, ColorOrigin, ColorProfileSource, PixelBuffer, PixelFormat,
        cms::PluggableCms, cms_moxcms::MoxCms, finalize_for_output_with, icc_profiles::ADOBE_RGB,
        output::OutputProfile,
    };
    let source = PixelBuffer::from_vec(
        vec![64, 32, 16, 128],
        1,
        1,
        PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
    )
    .unwrap()
    .with_color_context(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
        ADOBE_RGB,
    ))));
    let actual = finalize_for_output_with(
        &source,
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgba8,
        Some(&MoxCms),
    )
    .unwrap();
    let mut expected = [0_u8; 4];
    MoxCms
        .build_source_transform(
            ColorProfileSource::Icc(ADOBE_RGB),
            ColorProfileSource::Cicp(Cicp::SRGB),
            PixelFormat::Rgba8,
            PixelFormat::Rgba8,
            &ConvertOptions::permissive(),
        )
        .unwrap()
        .unwrap()
        .transform_row(&[128, 64, 32, 128], &mut expected, 1);
    assert_eq!(actual.pixels().row(0)[3], 128);
    for (actual, expected) in actual.pixels().row(0).iter().zip(expected) {
        assert!(actual.abs_diff(expected) <= 1, "{actual} != {expected}");
    }
}
