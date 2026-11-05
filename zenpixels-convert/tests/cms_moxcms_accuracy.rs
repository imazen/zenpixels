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

// ---------------------------------------------------------------------------
// Pure-gamma curves at F32, which the backend evaluates with a fast power. It returns unrelated values
// for results it can't represent, and loses the sign of negative samples: gamma 1.8 of zero is -1.65e8.
// ---------------------------------------------------------------------------

use moxcms::{ColorProfile, ToneReprCurve};

fn float_row(src: &[u8], dst: &[u8], from: F, to: F, clip: bool, input: &[f32]) -> Vec<f32> {
    let channels = |f: F| f.bytes_per_pixel() / 4;
    let width = input.len() / channels(from);
    let mut out = vec![f32::NAN; width * channels(to)];
    MoxCms
        .build_source_transform(
            Source::Icc(src),
            Source::Icc(dst),
            from,
            to,
            &ConvertOptions::retain_channels().with_clip_out_of_gamut(clip),
        )
        .expect("declined")
        .expect("refused")
        .try_transform_row(
            bytemuck::cast_slice(input),
            bytemuck::cast_slice_mut(&mut out),
            width as u32,
        )
        .unwrap();
    out
}

fn no_cicp_icc(mut profile: ColorProfile) -> Vec<u8> {
    profile.cicp = None;
    profile.encode().unwrap()
}

fn prophoto_icc() -> Vec<u8> {
    no_cicp_icc(ColorProfile::new_pro_photo_rgb())
}

fn adobe_icc() -> Vec<u8> {
    no_cicp_icc(ColorProfile::new_adobe_rgb())
}

fn srgb_icc() -> Vec<u8> {
    no_cicp_icc(ColorProfile::new_srgb())
}

fn gray_icc(gamma: f32) -> Vec<u8> {
    ColorProfile::new_gray_with_gamma(gamma).encode().unwrap()
}

/// The exponent of an ICC profile's single-value curve, if that's what it has.
fn stored_gamma(icc: &[u8]) -> Option<f64> {
    let profile = ColorProfile::new_from_slice(icc).unwrap();
    match profile.gray_trc.or(profile.red_trc)? {
        ToneReprCurve::Lut(curve) if curve.len() == 1 => Some(f64::from(curve[0]) / 256.0),
        _ => None,
    }
}

fn odd_pow(x: f64, exponent: f64) -> f64 {
    x.abs().powf(exponent).copysign(x)
}

/// sRGB's curve, continued through its linear segment below zero as the backend does.
fn srgb_encode(v: f64) -> f64 {
    if v < 0.003041282560127521 {
        v * 12.92
    } else {
        1.055_010_718_947_586_6 * v.powf(1.0 / 2.4) - 0.055_010_718_947_586_6
    }
}

/// Matrix-to-matrix conversion in f64, with the target's stored curve or else sRGB's.
fn exact_rgb(src: &[u8], dst: &[u8], clip: bool, pixel: [f32; 3]) -> [f64; 3] {
    let (s, t) = (
        ColorProfile::new_from_slice(src).unwrap(),
        ColorProfile::new_from_slice(dst).unwrap(),
    );
    let m = t
        .rgb_to_xyz_matrix()
        .inverse()
        .mat_mul(s.rgb_to_xyz_matrix())
        .v;
    let decode = stored_gamma(src).unwrap();
    let linear = pixel.map(|x| odd_pow(f64::from(x), decode));
    [0, 1, 2].map(|c| {
        let v = m[c][0] * linear[0] + m[c][1] * linear[1] + m[c][2] * linear[2];
        let v = if clip { v.clamp(0.0, 1.0) } else { v };
        match stored_gamma(dst) {
            Some(gamma) => odd_pow(v, 1.0 / gamma),
            None => srgb_encode(v),
        }
    })
}

#[track_caller]
fn assert_close(got: &[f32], exact: &[f64], what: &str) {
    for (g, e) in got.iter().zip(exact) {
        assert!(
            (f64::from(*g) - e).abs() <= 2e-4 * e.abs().max(1.0),
            "{what}: got {got:?}, exact {exact:?}"
        );
    }
}

/// A zero channel made ordinary colors clip to full scale even with clipping on: ProPhoto
/// (0.5, 0.5, 0) came out (1, 1, 0).
#[test]
fn pure_gamma_source_matches_exact_math() {
    let (prophoto, adobe, srgb) = (prophoto_icc(), adobe_icc(), srgb_icc());
    let pixels = [
        [0.5f32, 0.5, 0.0],
        [0.6, 0.0, 0.3],
        [0.0, 0.4, 0.4],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ];
    for clip in [true, false] {
        for (src, dst, name) in [
            (&prophoto, &srgb, "ProPhoto to sRGB"),
            (&prophoto, &adobe, "ProPhoto to Adobe RGB"),
            (&adobe, &srgb, "Adobe RGB to sRGB"),
            (&adobe, &prophoto, "Adobe RGB to ProPhoto"),
        ] {
            for pixel in pixels {
                let got = float_row(src, dst, F::RgbF32, F::RgbF32, clip, &pixel);
                let what = format!("{name} clip={clip} {pixel:?}");
                assert_close(&got, &exact_rgb(src, dst, clip, pixel), &what);
            }
        }
    }
}

/// Unclipped black was about -2.1e9 from ProPhoto.
#[test]
fn unclipped_black_stays_black_from_a_pure_gamma_source() {
    let srgb = srgb_icc();
    for src in [prophoto_icc(), adobe_icc()] {
        let got = float_row(&src, &srgb, F::RgbF32, F::RgbF32, false, &[0.0; 3]);
        assert_eq!(got, [0.0; 3]);
    }
}

/// Adobe RGB of 1e-20 came out about -1.55e34 in linear light.
#[test]
fn tiny_samples_from_a_pure_gamma_source_are_exact() {
    let (adobe, srgb) = (adobe_icc(), srgb_icc());
    for clip in [true, false] {
        for pixel in [
            [1e-30f32, 1e-30, 1e-30],
            [1e-20, 1e-20, 1e-20],
            [1e-20, 0.5, 0.5],
            [f32::from_bits(1), 0.5, 0.5],
        ] {
            let got = float_row(&adobe, &srgb, F::RgbF32, F::RgbF32, clip, &pixel);
            let what = format!("clip={clip} {pixel:?}");
            assert_close(&got, &exact_rgb(&adobe, &srgb, clip, pixel), &what);
        }
    }
}

/// An unclipped negative sample is the curve's odd extension, from any gamma.
#[test]
fn negative_samples_from_a_pure_gamma_source_keep_their_sign() {
    let srgb = srgb_icc();
    for src in [prophoto_icc(), adobe_icc()] {
        let pixel = [-0.5f32, 0.5, 0.5];
        let got = float_row(&src, &srgb, F::RgbF32, F::RgbF32, false, &pixel);
        assert_close(&got, &exact_rgb(&src, &srgb, false, pixel), "negative red");
    }
}

/// 8-bit into sRGB takes its own matrix kernels, so these are independent routes to one answer.
#[test]
fn pure_gamma_source_agrees_with_the_eight_bit_route() {
    let (prophoto, srgb) = (prophoto_icc(), srgb_icc());
    let levels = 0..=255u8;
    let mut eight_in = Vec::new();
    for r in levels.clone().step_by(5) {
        for g in levels.clone().step_by(5) {
            for b in levels.clone().step_by(5) {
                eight_in.extend_from_slice(&[r, g, b]);
            }
        }
    }
    let float_in: Vec<f32> = eight_in.iter().map(|&v| f32::from(v) / 255.0).collect();
    let mut eight_out = vec![0u8; eight_in.len()];
    build(Source::Icc(&prophoto), Source::Icc(&srgb), F::Rgb8, F::Rgb8)
        .try_transform_row(&eight_in, &mut eight_out, (eight_in.len() / 3) as u32)
        .unwrap();
    let float_out = float_row(&prophoto, &srgb, F::RgbF32, F::RgbF32, true, &float_in);
    for (i, (e, f)) in eight_out.iter().zip(&float_out).enumerate() {
        let f = (f * 255.0).round() as i32;
        assert!(
            (f - i32::from(*e)).abs() <= 1,
            "sample {i} of {:?}: 8-bit {e}, F32 {f}",
            &float_in[i / 3 * 3..i / 3 * 3 + 3]
        );
    }
}

/// The row is staged in pieces: a long row, alpha, and every pixel's result must not depend on them.
#[test]
fn pure_gamma_source_is_the_same_per_pixel_and_keeps_alpha() {
    let (prophoto, srgb) = (prophoto_icc(), srgb_icc());
    let row: Vec<f32> = (0..1000)
        .flat_map(|i| {
            let v = |k: u32| ((i * k) % 256) as f32 / 255.0;
            [v(7), v(13), v(29), i as f32 * 0.25 - 40.0]
        })
        .collect();
    let whole = float_row(&prophoto, &srgb, F::RgbaF32, F::RgbaF32, false, &row);
    for (pixel, out) in row.as_chunks::<4>().0.iter().zip(whole.as_chunks::<4>().0) {
        let alone = float_row(&prophoto, &srgb, F::RgbaF32, F::RgbaF32, false, pixel);
        assert_eq!(alone.len(), 4);
        for (a, b) in alone.iter().zip(out) {
            assert_eq!(a.to_bits(), b.to_bits(), "{pixel:?}");
        }
        assert_eq!(out[3].to_bits(), pixel[3].to_bits(), "alpha is carried");
    }
}

/// Samples the backend's fast power can represent keep the backend's exact bits, so this path changes
/// only what was wrong.
#[test]
fn representable_samples_keep_the_backends_bits() {
    let options = moxcms::TransformOptions {
        rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
        interpolation_method: moxcms::InterpolationMethod::Tetrahedral,
        prefer_fixed_point: false,
        allow_extended_range_rgb_xyz: true,
    };
    let parsed = |icc: &[u8]| {
        let mut profile = ColorProfile::new_from_slice(icc).unwrap();
        profile.cicp = None;
        profile
    };
    let (prophoto, srgb) = (prophoto_icc(), srgb_icc());
    let row: Vec<f32> = (1..=1000).map(|i| i as f32 * 0.0037 + 1e-3).collect();
    let mut direct = vec![0f32; row.len()];
    parsed(&prophoto)
        .create_transform_f32(
            moxcms::Layout::Rgb,
            &parsed(&srgb),
            moxcms::Layout::Rgb,
            options,
        )
        .unwrap()
        .transform(&row[..999], &mut direct[..999])
        .unwrap();
    let adapter = float_row(&prophoto, &srgb, F::RgbF32, F::RgbF32, false, &row[..999]);
    for (a, d) in adapter.iter().zip(&direct) {
        assert_eq!(a.to_bits(), d.to_bits());
    }
    let (g18, g22) = (gray_icc(1.8), gray_icc(2.2));
    let mut direct = vec![0f32; row.len()];
    parsed(&g18)
        .create_transform_f32(
            moxcms::Layout::Gray,
            &parsed(&g22),
            moxcms::Layout::Gray,
            options,
        )
        .unwrap()
        .transform(&row, &mut direct)
        .unwrap();
    let adapter = float_row(&g18, &g22, F::GrayF32, F::GrayF32, false, &row);
    for (a, d) in adapter.iter().zip(&direct) {
        assert_eq!(a.to_bits(), d.to_bits());
    }
}

/// Gray of zero through gamma 1.8 into gray 2.2 came out full scale, clipped or not.
#[test]
fn pure_gamma_gray_source_and_target_are_exact() {
    let (g18, g22, g26, srgb, adobe) = (
        gray_icc(1.8),
        gray_icc(2.2),
        gray_icc(2.6),
        srgb_icc(),
        adobe_icc(),
    );
    for clip in [true, false] {
        for x in [0.0f32, 1e-30, 1e-20, 1e-6, 0.25, 0.5, 1.0, -0.5] {
            let clipped = if clip { x.clamp(0.0, 1.0) } else { x };
            for (src, dst) in [(&g18, &g22), (&g22, &g26), (&g26, &g18)] {
                let linear = odd_pow(f64::from(clipped), stored_gamma(src).unwrap());
                let exact = odd_pow(linear, 1.0 / stored_gamma(dst).unwrap());
                let got = float_row(src, dst, F::GrayF32, F::GrayF32, clip, &[x]);
                assert_close(&got, &[exact], &format!("gray to gray clip={clip} {x}"));
            }
            // Into RGB, gray's neutral axis through the target's own curve.
            let linear = odd_pow(f64::from(clipped), stored_gamma(&g18).unwrap());
            let got = float_row(&g18, &adobe, F::GrayF32, F::RgbF32, clip, &[x]);
            let exact = odd_pow(linear, 1.0 / stored_gamma(&adobe).unwrap());
            assert_close(&got, &[exact; 3], &format!("gray to Adobe clip={clip} {x}"));
            let got = float_row(&g18, &srgb, F::GrayF32, F::RgbF32, clip, &[x]);
            let exact = srgb_encode(linear);
            assert_close(&got, &[exact; 3], &format!("gray to sRGB clip={clip} {x}"));
        }
    }
}

/// A gray target weighs the source's linear channels by their luminance and keeps it in range.
#[test]
fn rgb_into_a_pure_gamma_gray_target_is_exact() {
    let g22 = gray_icc(2.2);
    let gamma = stored_gamma(&g22).unwrap();
    for (src, decode) in [(prophoto_icc(), true), (srgb_icc(), false)] {
        let profile = ColorProfile::new_from_slice(&src).unwrap();
        let y = profile.colorant_matrix().v[1];
        for pixel in [
            [0.5f32, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.3],
            [0.5, 0.5, 0.5],
        ] {
            let linear = pixel.map(|x| {
                let x = f64::from(x);
                if decode {
                    x.powf(stored_gamma(&src).unwrap())
                } else if x < 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            });
            let v = y[0] * linear[0] + y[1] * linear[1] + y[2] * linear[2];
            let exact = v.clamp(0.0, 1.0).powf(1.0 / gamma);
            let got = float_row(&src, &g22, F::RgbF32, F::GrayF32, true, &pixel);
            assert_close(&got, &[exact], &format!("RGB to gray {pixel:?}"));
        }
    }
}

/// A target with sRGB's white and primaries pulled toward it, so sRGB's corners are out of its gamut.
fn narrow_target_icc(curve: ToneReprCurve) -> Vec<u8> {
    let mut profile = ColorProfile::new_srgb();
    profile.cicp = None;
    profile.red_trc = Some(curve.clone());
    profile.green_trc = Some(curve.clone());
    profile.blue_trc = Some(curve);
    let white = [0.9642, 1.0, 0.8249];
    for colorant in [
        &mut profile.red_colorant,
        &mut profile.green_colorant,
        &mut profile.blue_colorant,
    ] {
        colorant.x = colorant.x * 0.7 + white[0] * 0.1;
        colorant.y = colorant.y * 0.7 + white[1] * 0.1;
        colorant.z = colorant.z * 0.7 + white[2] * 0.1;
    }
    profile.encode().unwrap()
}

/// At a table's grid nodes the result is the node's linear value through the curve. Red of sRGB came
/// out (1, 0.39, 0.39) on a narrower pure-gamma target, clipped or not.
#[test]
fn table_source_keeps_the_sign_at_grid_nodes_on_a_pure_gamma_target() {
    let table = synthetic_icc::synthetic_rgb_lut_icc();
    let encoded = narrow_target_icc(moxcms::curve_from_gamma(563.0 / 256.0));
    let linear = narrow_target_icc(ToneReprCurve::Lut(Vec::new()));
    for pixel in [
        [1.0f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ] {
        let light = float_row(&table, &linear, F::RgbF32, F::RgbF32, false, &pixel);
        assert!(
            light.iter().any(|&v| v < -0.01),
            "needs out-of-gamut: {light:?}"
        );
        for clip in [true, false] {
            let exact: Vec<f64> = light
                .iter()
                .map(|&v| {
                    let v = f64::from(v);
                    odd_pow(if clip { v.clamp(0.0, 1.0) } else { v }, 256.0 / 563.0)
                })
                .collect();
            let got = float_row(&table, &encoded, F::RgbF32, F::RgbF32, clip, &pixel);
            assert_close(&got, &exact, &format!("table node clip={clip} {pixel:?}"));
        }
    }
}

fn byte_row(
    src: &[u8],
    dst: &[u8],
    from: F,
    to: F,
    input: &[u8],
    width: usize,
    out: usize,
) -> Vec<u8> {
    let mut output = vec![0u8; out];
    build(Source::Icc(src), Source::Icc(dst), from, to)
        .try_transform_row(input, &mut output, width as u32)
        .unwrap();
    output
}

/// 8-bit samples that don't take the matrix kernels go through the F32 path: gray 1.8 of 0 into gray 2.2
/// was 255, and ProPhoto (128, 128, 0) into Adobe RGB was (255, 255, 0).
#[test]
fn eight_bit_pure_gamma_routes_are_nearest_of_exact_math() {
    let (g18, g22, prophoto, adobe) = (gray_icc(1.8), gray_icc(2.2), prophoto_icc(), adobe_icc());
    let (decode, encode) = (stored_gamma(&g18).unwrap(), stored_gamma(&g22).unwrap());
    let levels: Vec<u8> = (0..=255).collect();
    let out = byte_row(&g18, &g22, F::Gray8, F::Gray8, &levels, 256, 256);
    for (&level, &got) in levels.iter().zip(&out) {
        let exact = (f64::from(level) / 255.0).powf(decode).powf(1.0 / encode) * 255.0;
        assert!(
            (f64::from(got) - exact).abs() <= 0.51,
            "gray {level}: {got}, exact {exact}"
        );
    }
    let mut pixels = Vec::new();
    for r in (0..=255u8).step_by(15) {
        for g in (0..=255u8).step_by(15) {
            for b in (0..=255u8).step_by(15) {
                pixels.extend_from_slice(&[r, g, b]);
            }
        }
    }
    let width = pixels.len() / 3;
    let out = byte_row(
        &prophoto,
        &adobe,
        F::Rgb8,
        F::Rgb8,
        &pixels,
        width,
        pixels.len(),
    );
    for (input, got) in pixels.as_chunks::<3>().0.iter().zip(out.as_chunks::<3>().0) {
        let exact = exact_rgb(&prophoto, &adobe, true, input.map(|v| f32::from(v) / 255.0));
        for (g, e) in got.iter().zip(exact) {
            assert!(
                (f64::from(*g) - e * 255.0).abs() <= 0.51,
                "{input:?}: {got:?}, exact {exact:?}"
            );
        }
    }
}
