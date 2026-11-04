//! A CMS consumes straight samples; alpha association belongs around it.
//!
//! Carried from the retired `fix/finalize-current-color` branch (3ece79f3).
//! That branch refused nonlinear transfer and tone mapping on associated
//! samples, and padding-as-coverage; main now implements the first two by
//! unassociating around the transform, so these tests pin the results.
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
        assert!(ConvertPlan::new(from, to).is_ok());
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
fn padding_becomes_opaque_alpha_without_touching_color() {
    let padded = PixelDescriptor::RGBAF32_LINEAR
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_alpha(Some(AlphaMode::Undefined));
    let opaque = padded.with_alpha(Some(AlphaMode::Straight));
    let mut reference = RowConverter::new(opaque, PixelDescriptor::RGBAF32_LINEAR).unwrap();
    let mut expected = [0.0_f32; 4];
    reference.convert_row(
        bytemuck::cast_slice(&[0.5_f32, 0.25, 0.125, 1.0]),
        bytemuck::cast_slice_mut(&mut expected),
        1,
    );
    let mut converter = RowConverter::new(padded, PixelDescriptor::RGBAF32_LINEAR).unwrap();
    for padding in [0.0_f32, 0.5, 7.0, f32::NAN] {
        let mut output = [0.0_f32; 4];
        converter.convert_row(
            bytemuck::cast_slice(&[0.5_f32, 0.25, 0.125, padding]),
            bytemuck::cast_slice_mut(&mut output),
            1,
        );
        assert_eq!(output, expected, "padding {padding}");
    }
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn tone_mapping_associated_samples_matches_straight_samples() {
    use zenpixels_convert::ConvertPlan;
    let straight = PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Straight));
    let associated = PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied));
    for target in [
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
    ] {
        for (color, alpha) in [
            ([2.0_f32, 1.0, 0.25], 0.5_f32),
            ([4.0, 0.5, 0.1], 0.25),
            ([0.5, 0.5, 0.5], 1.0),
        ] {
            let run = |from: PixelDescriptor, pixel: [f32; 4]| {
                let plan = ConvertPlan::new_with_hdr_peak(from, target, 1000.0).unwrap();
                let mut converter = RowConverter::from_plan(plan);
                let mut output = [0_u8; 4];
                converter.convert_row(bytemuck::cast_slice(&pixel), &mut output, 1);
                output
            };
            let [r, g, b] = color;
            assert_eq!(
                run(associated, [r * alpha, g * alpha, b * alpha, alpha]),
                run(straight, [r, g, b, alpha]),
                "{target} {color:?} alpha {alpha}"
            );
        }
    }
}
