#![cfg(feature = "cms-lcms")]
use zenpixels_convert::{
    Cicp, ColorProfileSource as Source, ConvertOptions, Lcms, PixelFormat as F, PluggableCms,
};

#[path = "common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;

fn build(
    src: Source<'_>,
    dst: Source<'_>,
    from: F,
    to: F,
) -> Box<dyn zenpixels_convert::RowTransformMut> {
    Lcms.build_source_transform(src, dst, from, to, &ConvertOptions::permissive())
        .expect("accepted format")
        .expect("valid profile")
}
const SRGB: Source<'static> = Source::Cicp(Cicp::new(1, 13, 0, true));

#[test]
fn preparation_bounds_and_short_rows_fail_before_writes() {
    let mut transform = build(SRGB, SRGB, F::Rgb8, F::Rgb8);
    transform.prepare(2).unwrap();
    for (input, width) in [(&[0u8; 2][..], 1), (&[0; 9][..], 3)] {
        let mut output = [0xa5; 12];
        assert!(
            transform
                .try_transform_row(input, &mut output, width)
                .is_err()
        );
        assert_eq!(output, [0xa5; 12]);
    }
    let mut output = [0xa5; 2];
    assert!(
        transform
            .try_transform_row(&[0; 3], &mut output, 1)
            .is_err()
    );
    assert_eq!(output, [0xa5; 2]);
}

#[test]
fn prepared_float_ink_rows_allocate_nothing_and_preserve_padding() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let mut transform = build(Source::Icc(&icc), SRGB, F::CmykF32Inverted, F::RgbF32);
    transform.prepare(17).unwrap();
    let input = [1.0f32; 17 * 4];
    let mut output = [f32::from_bits(0x7fc01234); 17 * 3 + 3];
    let measured = allocation_counter::measure(|| {
        transform
            .try_transform_row(
                bytemuck::cast_slice(&input),
                bytemuck::cast_slice_mut(&mut output),
                17,
            )
            .unwrap();
    });
    assert_eq!(measured.count_total, 0);
    assert!(output[..51].iter().all(|v| (*v - 1.0).abs() < 0.001));
    assert!(output[51..].iter().all(|v| v.to_bits() == 0x7fc01234));
}

#[test]
fn bgr_and_bgra_match_rgb_channel_order_and_keep_alpha() {
    for (from, to, input, expected) in [
        (F::Bgr8, F::Rgb8, vec![17, 83, 201], vec![201, 83, 17]),
        (
            F::Bgra8,
            F::Rgba8,
            vec![17, 83, 201, 73],
            vec![201, 83, 17, 73],
        ),
        (F::Rgb8, F::Bgr8, vec![201, 83, 17], vec![17, 83, 201]),
        (
            F::Rgba8,
            F::Bgra8,
            vec![201, 83, 17, 73],
            vec![17, 83, 201, 73],
        ),
    ] {
        let mut transform = build(SRGB, SRGB, from, to);
        let mut output = vec![0; expected.len()];
        transform.try_transform_row(&input, &mut output, 1).unwrap();
        assert_eq!(output, expected, "{from:?}>{to:?}");
    }
}

#[test]
fn adding_alpha_makes_it_opaque_without_touching_later_pixels() {
    for (from, to, input, expected) in [
        (F::Rgb8, F::Rgba8, vec![255; 3], vec![255; 4]),
        (F::Gray8, F::GrayA8, vec![255], vec![255; 2]),
    ] {
        let mut transform = build(SRGB, SRGB, from, to);
        let mut output = vec![0xa5; expected.len() + 2];
        transform.try_transform_row(&input, &mut output, 1).unwrap();
        assert_eq!(&output[..expected.len()], expected);
        assert_eq!(&output[expected.len()..], &[0xa5; 2]);
    }
}

#[test]
fn cmyk_u16_is_native_and_paper_white_keeps_alpha_opaque() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let mut transform = build(Source::Icc(&icc), SRGB, F::Cmyk16, F::Rgba16);
    let mut output = [0u16; 4];
    transform
        .try_transform_row(
            bytemuck::cast_slice(&[0u16; 4]),
            bytemuck::cast_slice_mut(&mut output),
            1,
        )
        .unwrap();
    assert!(output[..3].iter().all(|v| *v >= 65534), "{output:?}");
    assert_eq!(output[3], 65535);
}

#[test]
fn unsupported_and_device_dependent_descriptions_decline() {
    for format in [F::RgbF16, F::Rgbx8, F::OklabF32] {
        assert!(
            Lcms.build_source_transform(SRGB, SRGB, format, F::Rgb8, &ConvertOptions::permissive())
                .is_none()
        );
    }
    assert!(
        Lcms.build_source_transform(SRGB, SRGB, F::Cmyk8, F::Rgb8, &ConvertOptions::permissive())
            .is_none()
    );
    assert!(
        Lcms.build_source_transform(SRGB, SRGB, F::Rgb8, F::Rgb16, &ConvertOptions::permissive())
            .is_none()
    );
}

#[test]
fn unsupported_rgb_matrix_or_range_is_an_explicit_failure() {
    for cicp in [Cicp::new(1, 13, 9, true), Cicp::new(1, 13, 0, false)] {
        assert!(matches!(
            Lcms.build_source_transform(
                Source::Cicp(cicp),
                SRGB,
                F::Rgb8,
                F::Rgb8,
                &ConvertOptions::permissive()
            ),
            Some(Err(_))
        ));
    }
}

#[test]
fn unaligned_float_rows_preserve_every_alpha_bit_including_nan() {
    for format in [F::RgbaF32, F::GrayAF32] {
        let colors = format.channels() - 1;
        let mut input = vec![0x35];
        let alpha_bits = [
            0x8000_0000u32,
            0x7fc0_1234,
            1.5f32.to_bits(),
            (-0.5f32).to_bits(),
        ];
        for alpha in alpha_bits {
            for _ in 0..colors {
                input.extend_from_slice(&0.25f32.to_ne_bytes());
            }
            input.extend_from_slice(&alpha.to_ne_bytes());
        }
        let mut output = vec![0xa5; input.len() + 2];
        let mut transform = build(SRGB, SRGB, format, format);
        transform.prepare(4).unwrap();
        transform
            .try_transform_row(&input[1..], &mut output[1..], 4)
            .unwrap();
        for (pixel, alpha) in output[1..1 + 4 * format.bytes_per_pixel()]
            .chunks_exact(format.bytes_per_pixel())
            .zip(alpha_bits)
        {
            assert_eq!(&pixel[colors * 4..], &alpha.to_ne_bytes(), "{format:?}");
        }
        assert_eq!(output[0], 0xa5);
        assert_eq!(&output[output.len() - 2..], &[0xa5; 2]);
    }
}

/// Actual integer entry/exit formats, rather than float-normalized fixtures.
#[cfg(feature = "icc-db")]
#[test]
fn native_integer_pq_layouts_match_analytic_quantization_and_keep_alpha() {
    use zenpixels_convert::icc_profiles::SynthesizedIcc;
    for (format, gray, colors, alpha, depth) in [
        (F::Rgb8, false, 3, false, 1),
        (F::Rgba8, false, 3, true, 1),
        (F::Gray8, true, 1, false, 1),
        (F::GrayA8, true, 1, true, 1),
        (F::Rgb16, false, 3, false, 2),
        (F::Rgba16, false, 3, true, 2),
        (F::Gray16, true, 1, false, 2),
        (F::GrayA16, true, 1, true, 2),
    ] {
        let synth: fn(Cicp) -> SynthesizedIcc = if gray {
            SynthesizedIcc::for_gray_cicp
        } else {
            SynthesizedIcc::for_cicp
        };
        let SynthesizedIcc::Profile(pq) = synth(Cicp::new(9, 16, 0, true)) else {
            panic!("PQ profile unavailable");
        };
        let SynthesizedIcc::Profile(linear) = synth(Cicp::new(9, 8, 0, true)) else {
            panic!("matching linear profile unavailable");
        };
        let maximum = if depth == 1 { 255u32 } else { 65535 };
        let mut input = Vec::new();
        for code in 0..=maximum {
            for channel in 0..colors + usize::from(alpha) {
                let sample = if channel == colors {
                    code ^ (maximum / 3)
                } else {
                    code
                };
                if depth == 1 {
                    input.push(sample as u8);
                } else {
                    input.extend_from_slice(&(sample as u16).to_ne_bytes());
                }
            }
        }
        let decode = |encoded: f64| {
            let p = encoded.powf(32.0 / 2523.0);
            ((p - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p))
                .powf(8192.0 / 1305.0)
        };
        let encode = |linear: f64| {
            if linear == 0.0 {
                return 0.0;
            }
            let p = linear.powf(1305.0 / 8192.0);
            ((107.0 / 128.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p)).powf(2523.0 / 32.0)
        };
        for inverse in [false, true] {
            let mut output = vec![0xa5; input.len()];
            let (source, target) = if inverse {
                (&linear, &pq)
            } else {
                (&pq, &linear)
            };
            #[cfg(feature = "cms-moxcms")]
            assert!(
                zenpixels_convert::MoxCms
                    .build_source_transform(
                        Source::Icc(source),
                        Source::Icc(target),
                        format,
                        format,
                        &ConvertOptions::permissive(),
                    )
                    .is_none(),
                "MoxCMS must decline the unsupported PQ float pipeline"
            );
            let mut transform = build(Source::Icc(source), Source::Icc(target), format, format);
            transform
                .try_transform_row(&input, &mut output, maximum + 1)
                .unwrap();
            let pixel_bytes = (colors + usize::from(alpha)) * depth;
            let mut largest = 0u32;
            for (code, (before, after)) in (0..=maximum).zip(
                input
                    .chunks_exact(pixel_bytes)
                    .zip(output.chunks_exact(pixel_bytes)),
            ) {
                let value = f64::from(code) / f64::from(maximum);
                let reference = if inverse {
                    encode(value)
                } else {
                    decode(value)
                };
                let expected = (reference * f64::from(maximum)).round() as u32;
                for channel in 0..colors {
                    let actual = if depth == 1 {
                        u32::from(after[channel])
                    } else {
                        u32::from(u16::from_ne_bytes(
                            after[channel * 2..channel * 2 + 2].try_into().unwrap(),
                        ))
                    };
                    largest = largest.max(actual.abs_diff(expected));
                    let a = f64::from(actual) / f64::from(maximum);
                    let e = f64::from(expected) / f64::from(maximum);
                    let error = if inverse {
                        (decode(a) - decode(e)).abs()
                    } else {
                        (a - e).abs()
                    };
                    // Compare with correctly rounded native integer reference
                    // values under the unchanged 0.0005 linear-light budget.
                    assert!(
                        error <= 0.0005,
                        "{format:?} inverse={inverse} code {code}: {actual} versus {expected} (linear error {error})"
                    );
                    if depth == 1 {
                        assert_eq!(actual, expected, "{format:?} inverse={inverse} code {code}");
                    }
                }
                if alpha {
                    assert_eq!(
                        &after[colors * depth..],
                        &before[colors * depth..],
                        "{format:?} alpha code {code}"
                    );
                }
            }
            println!(
                "{format:?} inverse={inverse}: {} native inputs; max integer code error {largest}; alpha exact",
                maximum + 1
            );
        }
    }
}
