#![forbid(unsafe_code)]
macro_rules! compare {
    ($m:ident,$name:literal) => {{
        let mut values = Vec::new();
        for i in 0..=65535u32 {
            values.push(i as f32 / 65535.0);
        }
        for bits in [
            0u32,
            1,
            0x0080_0000,
            0x2980_0000,
            0x3f80_0000,
            0x5580_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0x7fc0_0000,
        ] {
            for offset in 0..=32 {
                values.push(f32::from_bits(bits.wrapping_add(offset)));
                values.push(f32::from_bits(bits.wrapping_sub(offset)));
            }
        }
        let mut state = 0x9e3779b9u32;
        for _ in 0..262144 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values.push(f32::from_bits(state));
        }
        let mut changed = 0u64;
        let mut pixel_difference = 0u64;
        for gamma in [1f32, 1.8, 563.0 / 256.0, 2.6, 2.8, 3., 4.] {
            for encode in [false, true] {
                let c = $m::curve_from_gamma(gamma);
                let r = review_range::curve_from_gamma(gamma);
                let (candidate, reference) = if encode {
                    (
                        c.make_gamma_evaluator().unwrap(),
                        r.make_gamma_evaluator().unwrap(),
                    )
                } else {
                    (
                        c.make_linear_evaluator().unwrap(),
                        r.make_linear_evaluator().unwrap(),
                    )
                };
                for &x in &values {
                    let actual = candidate.evaluate_value(x);
                    let expected = reference.evaluate_value(x);
                    changed += u64::from(
                        !(actual.is_nan() && expected.is_nan())
                            && actual.to_bits() != expected.to_bits(),
                    );
                    let pixel = candidate.evaluate_tristimulus($m::Rgb::new(x, 0.5, -0.5));
                    if !(pixel.r.is_nan() && actual.is_nan())
                        && pixel.r.to_bits() != actual.to_bits()
                    {
                        pixel_difference += 1;
                    }
                }
            }
        }
        println!(
            "{} samples={} reference_bit_changes={} pixel_scalar_disagreement={}",
            $name,
            values.len() * 14,
            changed,
            pixel_difference
        );
        assert_eq!(pixel_difference, 0, $name);
        if $name != "opt_wide" {
            assert_eq!(changed, 0, $name);
        }
    }};
}
macro_rules! transform_compare {
    ($m:ident,$name:literal) => {{
        let mut input = Vec::new();
        for i in 0..=65535u32 {
            let x = i as f32 / 65535.0;
            input.extend_from_slice(&[x, if i % 10 == 0 { 0.0 } else { -x }, 0.5]);
        }
        let mut checked = 0;
        for kind in 0..3 {
            let mut source = if kind == 0 {
                $m::ColorProfile::new_pro_photo_rgb()
            } else {
                $m::ColorProfile::new_srgb()
            };
            let mut reference_source = if kind == 0 {
                review_range::ColorProfile::new_pro_photo_rgb()
            } else {
                review_range::ColorProfile::new_srgb()
            };
            if kind != 0 {
                source.cicp = Some($m::CicpProfile {
                    color_primaries: $m::CicpColorPrimaries::Bt709,
                    transfer_characteristics: if kind == 1 {
                        $m::TransferCharacteristics::Bt470M
                    } else {
                        $m::TransferCharacteristics::Bt470Bg
                    },
                    matrix_coefficients: $m::MatrixCoefficients::Bt709,
                    full_range: true,
                });
                reference_source.cicp = Some(review_range::CicpProfile {
                    color_primaries: review_range::CicpColorPrimaries::Bt709,
                    transfer_characteristics: if kind == 1 {
                        review_range::TransferCharacteristics::Bt470M
                    } else {
                        review_range::TransferCharacteristics::Bt470Bg
                    },
                    matrix_coefficients: review_range::MatrixCoefficients::Bt709,
                    full_range: true,
                });
            }
            let target = if kind == 0 {
                $m::ColorProfile::new_srgb()
            } else {
                $m::ColorProfile::new_bt2020()
            };
            let reference_target = if kind == 0 {
                review_range::ColorProfile::new_srgb()
            } else {
                review_range::ColorProfile::new_bt2020()
            };
            let actual_transform = source
                .create_transform_f32(
                    $m::Layout::Rgb,
                    &target,
                    $m::Layout::Rgb,
                    $m::TransformOptions {
                        allow_extended_range_rgb_xyz: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let reference_transform = reference_source
                .create_transform_f32(
                    review_range::Layout::Rgb,
                    &reference_target,
                    review_range::Layout::Rgb,
                    review_range::TransformOptions {
                        allow_extended_range_rgb_xyz: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut actual = vec![0f32; input.len()];
            let mut expected = vec![0f32; input.len()];
            actual_transform.transform(&input, &mut actual).unwrap();
            reference_transform
                .transform(&input, &mut expected)
                .unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.to_bits(), b.to_bits(), "{} transform {}", $name, kind);
                checked += 1;
            }
        }
        println!("{} transform_channel_bit_matches={}", $name, checked);
    }};
}
fn main() {
    #[cfg(feature = "outlined")]
    {
        compare!(opt_cold, "opt_cold");
        transform_compare!(opt_cold, "opt_cold");
        let curve = opt_cold::curve_from_gamma(0.0);
        let reference = review_range::curve_from_gamma(0.0);
        for encode in [false, true] {
            let (actual, expected) = if encode {
                (
                    curve.make_gamma_evaluator().unwrap(),
                    reference.make_gamma_evaluator().unwrap(),
                )
            } else {
                (
                    curve.make_linear_evaluator().unwrap(),
                    reference.make_linear_evaluator().unwrap(),
                )
            };
            for x in [0f32, -0.0, 0.5, -0.5] {
                assert_eq!(
                    actual.evaluate_value(x).to_bits(),
                    expected.evaluate_value(x).to_bits()
                );
            }
        }
    }
    compare!(opt_or, "opt_or");
    compare!(opt_pixel, "opt_pixel");
    compare!(opt_wide, "opt_wide");
    transform_compare!(opt_or, "opt_or");
    transform_compare!(opt_pixel, "opt_pixel");
}
