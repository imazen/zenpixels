//! What `MoxCms` must get right at each depth: white, rounding, row extent and table profiles.

#![cfg(feature = "cms-moxcms")]

#[path = "common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;

use zenpixels::{Cicp, ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4};
use zenpixels_convert::policy::ConvertOptions;

const SRGB: Source<'static> = Source::Cicp(Cicp::new(1, 13, 0, true));
const SIXTEEN_MISSED: f64 = 0.20;
const SIXTEEN_BIAS: f64 = 0.2;
const SIXTEEN_WORST: f64 = 8.0;

fn build(src: Source<'_>, dst: Source<'_>, from: F, to: F) -> Box<dyn RowTransformMut> {
    MoxCms
        .build_source_transform(src, dst, from, to, &ConvertOptions::permissive())
        .expect("declined")
        .expect("refused")
}

fn ramp8(pixels: usize) -> Vec<u8> {
    (0..pixels)
        .flat_map(|i| {
            [
                (i * 7 % 256) as u8,
                (i * 13 % 256) as u8,
                (i * 29 % 256) as u8,
            ]
        })
        .collect()
}

/// Generated profiles and ICC profiles must share a white: 8-bit white into generated sRGB was
/// (255, 255, 254).
#[test]
fn white_stays_white_between_icc_and_generated_profiles() {
    for icc in [ADOBE_RGB, DISPLAY_P3_V4] {
        let mut into_srgb = build(Source::Icc(icc), SRGB, F::Rgb8, F::Rgb8);
        let mut out = [0u8; 3];
        into_srgb.try_transform_row(&[255; 3], &mut out, 1).unwrap();
        assert_eq!(out, [255; 3]);

        let mut wide = build(Source::Icc(icc), SRGB, F::Rgb16, F::Rgb16);
        let mut out = [0u16; 3];
        wide.try_transform_row(
            bytemuck::cast_slice(&[65535u16; 3]),
            bytemuck::cast_slice_mut(&mut out),
            1,
        )
        .unwrap();
        for v in out {
            assert!(v >= 65534, "{v}");
        }
    }
}

/// The backend's own 8-bit path comes out low, missing nearest on 43% of samples.
#[test]
fn eight_bit_rounds_to_nearest() {
    for (src, dst) in [
        (Source::Icc(DISPLAY_P3_V4), SRGB),
        (SRGB, Source::Icc(DISPLAY_P3_V4)),
        (Source::Icc(ADOBE_RGB), SRGB),
    ] {
        let input = ramp8(4096);
        let mut out = vec![0u8; input.len()];
        build(src.clone(), dst.clone(), F::Rgb8, F::Rgb8)
            .try_transform_row(&input, &mut out, 4096)
            .unwrap();

        let input_f: Vec<f32> = input.iter().map(|&v| f32::from(v) / 255.0).collect();
        let mut reference = vec![0f32; input_f.len()];
        build(src, dst, F::RgbF32, F::RgbF32)
            .try_transform_row(
                bytemuck::cast_slice(&input_f),
                bytemuck::cast_slice_mut(&mut reference),
                4096,
            )
            .unwrap();

        let mut signed = 0.0f64;
        let mut wrong = 0usize;
        for (v, r) in out.iter().zip(&reference) {
            let r = f64::from(r.clamp(0.0, 1.0)) * 255.0;
            signed += f64::from(*v) - r;
            wrong += usize::from(f64::from(*v) != r.round());
        }
        let share = wrong as f64 / out.len() as f64;
        let bias = signed / out.len() as f64;
        assert!(
            share < 0.02,
            "{:.2}% of samples missed nearest",
            share * 100.0
        );
        assert!(bias.abs() < 0.02, "mean signed error {bias}");
    }
}

/// The backend's own 16-bit path truncates its sRGB table too: 57% of Display P3 to sRGB samples
/// missed nearest of the exact result, for a mean of -0.64 of a 16-bit code.
#[test]
fn sixteen_bit_rounds_to_nearest_of_the_exact_result() {
    // Exact math on the profile's curve and colorants, into sRGB scaled to the PCS white.
    let p3 = moxcms::ColorProfile::new_from_slice(DISPLAY_P3_V4).unwrap();
    let srgb = moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::Bt709,
        transfer_characteristics: moxcms::TransferCharacteristics::Srgb,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    });
    let (r, g, b) = (srgb.red_colorant, srgb.green_colorant, srgb.blue_colorant);
    let to_xyz = [
        [r.x, g.x, b.x].map(|v| v * 0.9642 / (r.x + g.x + b.x)),
        [r.y, g.y, b.y].map(|v| v / (r.y + g.y + b.y)),
        [r.z, g.z, b.z].map(|v| v * 0.8249 / (r.z + g.z + b.z)),
    ];
    let matrix = moxcms::Matrix3d { v: to_xyz }
        .inverse()
        .mat_mul(p3.colorant_matrix())
        .v;
    let moxcms::ToneReprCurve::Parametric(curve) = p3.red_trc.as_ref().unwrap() else {
        panic!("Display P3 stores a parametric curve")
    };
    let c: Vec<f64> = curve.iter().map(|&v| f64::from(v)).collect();
    let decode = |v: f64| {
        if v >= c[4] {
            (c[1] * v + c[2]).powf(c[0])
        } else {
            c[3] * v
        }
    };
    let encode = |v: f64| {
        if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    let mut state = 0x9E37_79B9u32;
    let input: Vec<u16> = (0..3 * 4096)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 16) as u16
        })
        .collect();
    let mut out = vec![0u16; input.len()];
    build(Source::Icc(DISPLAY_P3_V4), SRGB, F::Rgb16, F::Rgb16)
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut out),
            4096,
        )
        .unwrap();

    let (mut signed, mut wrong, mut worst) = (0.0f64, 0usize, 0.0f64);
    for (from, to) in input.as_chunks::<3>().0.iter().zip(out.as_chunks::<3>().0) {
        let linear = from.map(|v| decode(f64::from(v) / 65535.0));
        for (row, got) in matrix.iter().zip(to) {
            let value = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
            let exact = encode(value.clamp(0.0, 1.0)) * 65535.0;
            let error = f64::from(*got) - exact;
            signed += error;
            worst = worst.max(error.abs());
            wrong += usize::from(f64::from(*got) != exact.round());
        }
    }
    let share = wrong as f64 / out.len() as f64;
    let bias = signed / out.len() as f64;
    eprintln!("U16 against exact: missed nearest {share}, mean {bias}, worst {worst}");
    assert!(share < SIXTEEN_MISSED, "{share} of samples missed nearest");
    assert!(bias.abs() < SIXTEEN_BIAS, "mean signed error {bias}");
    assert!(worst < SIXTEEN_WORST, "worst error {worst}");
}

/// Blue of this color is -0.051 in linear Adobe RGB. The backend's pure-gamma evaluator drops the sign,
/// which gave 66 at 8 bits and 0.259 at F32.
#[test]
fn out_of_gamut_samples_clip_on_a_pure_gamma_target() {
    let dst = Source::Icc(ADOBE_RGB);
    let src = Source::Icc(DISPLAY_P3_V4);

    let mut eight = [0u8; 3];
    build(src.clone(), dst.clone(), F::Rgb8, F::Rgb8)
        .try_transform_row(&[247, 255, 0], &mut eight, 1)
        .unwrap();
    assert_eq!(eight[2], 0);

    let mut sixteen = [0u16; 3];
    build(src.clone(), dst.clone(), F::Rgb16, F::Rgb16)
        .try_transform_row(
            bytemuck::cast_slice(&[63479u16, 65535, 0]),
            bytemuck::cast_slice_mut(&mut sixteen),
            1,
        )
        .unwrap();
    assert_eq!(sixteen[2], 0);

    let input = [247.0f32 / 255.0, 1.0, 0.0];
    let mut clipped = [0f32; 3];
    build(src.clone(), dst.clone(), F::RgbF32, F::RgbF32)
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut clipped),
            1,
        )
        .unwrap();
    assert_eq!(clipped[2], 0.0);

    let mut unclipped = [0f32; 3];
    MoxCms
        .build_source_transform(
            src,
            dst,
            F::RgbF32,
            F::RgbF32,
            &ConvertOptions::permissive().with_clip_out_of_gamut(false),
        )
        .unwrap()
        .unwrap()
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut unclipped),
            1,
        )
        .unwrap();
    assert!(unclipped[2] < -0.2, "blue {}", unclipped[2]);
}

#[test]
fn alpha_is_exact_through_the_sixteen_bit_route() {
    let input: Vec<u16> = (0..=65535u16)
        .step_by(257)
        .flat_map(|v| [v, 65535 - v, v / 2, v])
        .collect();
    let mut out = vec![0u16; input.len()];
    build(Source::Icc(DISPLAY_P3_V4), SRGB, F::Rgba16, F::Rgba16)
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut out),
            256,
        )
        .unwrap();
    for (from, to) in input.as_chunks::<4>().0.iter().zip(out.as_chunks::<4>().0) {
        assert_eq!(from[3], to[3]);
    }
}

/// `High` barycentric weights turned this into a constant at 16 bits.
#[test]
fn table_profile_is_right_at_every_depth() {
    let icc = synthetic_icc::synthetic_rgb_lut_icc();
    let input = ramp8(4096);

    let mut eight = vec![0u8; input.len()];
    build(Source::Icc(&icc), SRGB, F::Rgb8, F::Rgb8)
        .try_transform_row(&input, &mut eight, 4096)
        .unwrap();

    let wide: Vec<u16> = input.iter().map(|&v| u16::from(v) * 257).collect();
    let mut sixteen = vec![0u16; wide.len()];
    build(Source::Icc(&icc), SRGB, F::Rgb16, F::Rgb16)
        .try_transform_row(
            bytemuck::cast_slice(&wide),
            bytemuck::cast_slice_mut(&mut sixteen),
            4096,
        )
        .unwrap();

    let float: Vec<f32> = input.iter().map(|&v| f32::from(v) / 255.0).collect();
    let mut thirty_two = vec![0f32; float.len()];
    build(Source::Icc(&icc), SRGB, F::RgbF32, F::RgbF32)
        .try_transform_row(
            bytemuck::cast_slice(&float),
            bytemuck::cast_slice_mut(&mut thirty_two),
            4096,
        )
        .unwrap();

    for (i, &v) in input.iter().enumerate() {
        let want = f32::from(v);
        for (name, got) in [
            ("U8", f32::from(eight[i])),
            ("U16", f32::from(sixteen[i]) / 257.0),
            ("F32", thirty_two[i] * 255.0),
        ] {
            assert!(
                (got - want).abs() < 4.0,
                "{name} sample {i}: {got} for {want}"
            );
        }
    }
}

/// A row's slices may be longer than its width, and may be too short.
#[test]
fn rows_honor_their_width() {
    for (from, sample) in [(F::Rgb8, 1usize), (F::Rgb16, 2), (F::RgbF32, 4)] {
        let mut t = build(Source::Icc(DISPLAY_P3_V4), SRGB, from, from);
        let stride = 3 * sample;
        let input: Vec<u16> = vec![0x3F00; 40 * stride / 2];
        let input: &[u8] = bytemuck::cast_slice(&input);
        let mut exact = vec![0u8; 32 * stride];
        t.try_transform_row(&input[..32 * stride], &mut exact, 32)
            .unwrap();

        let mut padded = vec![0xA5u8; 40 * stride + (40 * stride) % 4];
        let aligned = padded.len() - padded.len() % 4;
        t.try_transform_row(input, &mut padded[..aligned], 32)
            .unwrap();
        assert_eq!(&padded[..32 * stride], &exact[..], "{from:?}");
        assert!(
            padded[32 * stride..].iter().all(|&b| b == 0xA5),
            "{from:?} wrote past the row"
        );

        let mut short = vec![0u8; 8 * stride];
        assert!(
            t.try_transform_row(input, &mut short, 32).is_err(),
            "{from:?}"
        );
        assert!(
            t.try_transform_row(&input[..8 * stride], &mut padded[..aligned], 32)
                .is_err()
        );
    }
}

/// Table profiles take the F32 path, so 8-bit is nearest of F32 exactly.
#[test]
fn eight_bit_table_profiles_are_nearest_of_their_float_result() {
    let icc = synthetic_icc::synthetic_rgb_lut_icc();
    let input = ramp8(4096);
    let mut out = vec![0u8; input.len()];
    build(Source::Icc(&icc), SRGB, F::Rgb8, F::Rgb8)
        .try_transform_row(&input, &mut out, 4096)
        .unwrap();
    let float: Vec<f32> = input.iter().map(|&v| f32::from(v) / 255.0).collect();
    let mut reference = vec![0f32; float.len()];
    build(Source::Icc(&icc), SRGB, F::RgbF32, F::RgbF32)
        .try_transform_row(
            bytemuck::cast_slice(&float),
            bytemuck::cast_slice_mut(&mut reference),
            4096,
        )
        .unwrap();
    for (v, r) in out.iter().zip(&reference) {
        assert_eq!(*v, (f64::from(r.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8);
    }
}

#[test]
fn alpha_is_exact_through_the_eight_bit_route() {
    let input: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v, v / 2, v]).collect();
    let mut out = vec![0u8; input.len()];
    build(Source::Icc(DISPLAY_P3_V4), SRGB, F::Rgba8, F::Rgba8)
        .try_transform_row(&input, &mut out, 256)
        .unwrap();
    for i in 0..256 {
        assert_eq!(usize::from(out[i * 4 + 3]), i);
    }
}
