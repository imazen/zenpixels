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
