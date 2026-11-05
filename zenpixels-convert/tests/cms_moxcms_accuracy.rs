//! What `MoxCms` must get right at each depth: white, rounding, row extent and table profiles.

#![cfg(feature = "cms-moxcms")]

#[path = "common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;

use zenpixels::{Cicp, ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4, SynthesizedIcc};
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

/// Blue of this color is -0.051 in linear Adobe RGB. The backend's pure-gamma evaluator drops the sign,
/// which gave 66 at 8 bits and 0.259 at F32. 16-bit takes the backend's table and isn't affected.
#[test]
fn out_of_gamut_samples_clip_on_a_pure_gamma_target() {
    let dst = Source::Icc(ADOBE_RGB);
    let src = Source::Icc(DISPLAY_P3_V4);

    let mut eight = [0u8; 3];
    build(src.clone(), dst.clone(), F::Rgb8, F::Rgb8)
        .try_transform_row(&[247, 255, 0], &mut eight, 1)
        .unwrap();
    assert_eq!(eight[2], 0);

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

/// One ink has one color: 8-bit CMYK is F32 CMYK rounded, in every RGB layout, on every target.
/// The 8-bit path used to have no black-point compensation and its own intent.
#[test]
fn eight_bit_cmyk_is_f32_cmyk_rounded() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let ink: Vec<u8> = (0..4096u32 * 4)
        .map(|i| ((i * 37 + i / 4 * 11) % 256) as u8)
        .collect();
    let ink_f32: Vec<f32> = ink.iter().map(|&v| f32::from(v) / 255.0).collect();
    let p3 = Source::Cicp(Cicp::new(12, 13, 0, true));
    // A profile with a `cicp` tag, which F32 parsing drops.
    let SynthesizedIcc::Profile(tagged) = SynthesizedIcc::for_cicp(Cicp::new(12, 13, 0, true))
    else {
        panic!("no profile for Display P3");
    };
    for target in [SRGB, Source::Icc(ADOBE_RGB), Source::Icc(&tagged), p3] {
        for (to8, tof, channels) in [(F::Rgb8, F::RgbF32, 3), (F::Rgba8, F::RgbaF32, 4)] {
            let mut bytes = vec![0u8; 4096 * channels];
            build(Source::Icc(&icc), target.clone(), F::Cmyk8, to8)
                .try_transform_row(&ink, &mut bytes, 4096)
                .unwrap();
            let mut floats = vec![0f32; 4096 * channels];
            build(Source::Icc(&icc), target.clone(), F::CmykF32, tof)
                .try_transform_row(
                    bytemuck::cast_slice(&ink_f32),
                    bytemuck::cast_slice_mut(&mut floats),
                    4096,
                )
                .unwrap();
            for (byte, float) in bytes.iter().zip(&floats) {
                let rounded = (f64::from(float.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                assert_eq!(*byte, rounded, "{target:?} {to8:?}: {float}");
            }
        }
    }
}

/// A row splits anywhere, across the 8-bit path's stack chunk too, and alpha stays opaque.
#[test]
fn eight_bit_cmyk_rows_split_and_alpha_is_opaque() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let ink: Vec<u8> = (0..600u32 * 4).map(|i| (i * 53 % 256) as u8).collect();
    let mut whole = vec![0u8; 600 * 4];
    let mut t = build(Source::Icc(&icc), SRGB, F::Cmyk8, F::Rgba8);
    t.try_transform_row(&ink, &mut whole, 600).unwrap();
    let mut parts = Vec::new();
    for row in ink.chunks(7 * 4) {
        let mut out = vec![0u8; row.len()];
        t.try_transform_row(row, &mut out, (row.len() / 4) as u32)
            .unwrap();
        parts.extend(out);
    }
    assert_eq!(whole, parts);
    assert!(whole.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
}

/// RGB and RGBA targets carry the same color from CMYK, at both depths.
#[test]
fn cmyk_targets_agree_with_and_without_alpha() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let ink: Vec<u8> = (0..4096u32 * 4).map(|i| (i * 61 % 256) as u8).collect();
    let mut rgb = vec![0u8; 4096 * 3];
    let mut rgba = vec![0u8; 4096 * 4];
    build(Source::Icc(&icc), SRGB, F::Cmyk8, F::Rgb8)
        .try_transform_row(&ink, &mut rgb, 4096)
        .unwrap();
    build(Source::Icc(&icc), SRGB, F::Cmyk8, F::Rgba8)
        .try_transform_row(&ink, &mut rgba, 4096)
        .unwrap();
    for (a, b) in rgb.as_chunks::<3>().0.iter().zip(rgba.as_chunks::<4>().0) {
        assert_eq!(a, &b[..3]);
    }
    let ink_f32: Vec<f32> = ink.iter().map(|&v| f32::from(v) / 255.0).collect();
    let mut rgb = vec![0f32; 4096 * 3];
    let mut rgba = vec![0f32; 4096 * 4];
    build(Source::Icc(&icc), SRGB, F::CmykF32, F::RgbF32)
        .try_transform_row(
            bytemuck::cast_slice(&ink_f32),
            bytemuck::cast_slice_mut(&mut rgb),
            4096,
        )
        .unwrap();
    build(Source::Icc(&icc), SRGB, F::CmykF32, F::RgbaF32)
        .try_transform_row(
            bytemuck::cast_slice(&ink_f32),
            bytemuck::cast_slice_mut(&mut rgba),
            4096,
        )
        .unwrap();
    for (a, b) in rgb.as_chunks::<3>().0.iter().zip(rgba.as_chunks::<4>().0) {
        assert_eq!(a, &b[..3]);
        assert_eq!(b[3], 1.0);
    }
}
