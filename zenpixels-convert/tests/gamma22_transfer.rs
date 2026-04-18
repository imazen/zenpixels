//! First-class Gamma 2.2 (Adobe RGB 1998) transfer function paths.
//!
//! Exercises Gamma22 ↔ Linear as a dedicated step and composed conversions
//! that route through linear (Gamma22 ↔ sRGB/BT.709/PQ/HLG, AdobeRGB ↔
//! BT.709/BT.2020/DisplayP3 gamut crossings).

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::RowConverter;

#[allow(clippy::excessive_precision)]
const ADOBE_GAMMA: f32 = 2.19921875; // 563/256, Adobe RGB 1998 spec (exact in f32).

/// Ground truth: Adobe RGB 1998 is a pure power curve at this exponent.
fn gamma22_to_linear(v: f32) -> f32 {
    v.max(0.0).powf(ADOBE_GAMMA)
}
fn linear_to_gamma22(v: f32) -> f32 {
    v.max(0.0).powf(1.0 / ADOBE_GAMMA)
}

fn f32_rgb_linear() -> PixelDescriptor {
    PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb)
}

fn f32_rgb_gamma22() -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
    )
    .with_primaries(ColorPrimaries::AdobeRgb)
}

fn convert_f32_rgb(src_desc: PixelDescriptor, dst_desc: PixelDescriptor, src: &[f32]) -> Vec<f32> {
    let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();
    let width = (src.len() / 3) as u32;
    let mut dst = vec![0.0f32; src.len()];
    conv.convert_row(
        bytemuck::cast_slice(src),
        bytemuck::cast_slice_mut(&mut dst),
        width,
    );
    dst
}

#[test]
fn gamma22_to_linear_f32_matches_powf() {
    let src_desc = f32_rgb_gamma22();
    let dst_desc = f32_rgb_linear();
    let src: Vec<f32> = (0..=20).flat_map(|i| [i as f32 / 20.0; 3]).collect();
    let dst = convert_f32_rgb(src_desc, dst_desc, &src);
    for (s, d) in src.iter().zip(dst.iter()) {
        let expected = gamma22_to_linear(*s);
        assert!(
            (d - expected).abs() < 1e-3,
            "EOTF mismatch: gamma22({s}) -> {d}, expected {expected}"
        );
    }
}

#[test]
fn linear_f32_to_gamma22_matches_powf() {
    let src_desc = f32_rgb_linear();
    let dst_desc = f32_rgb_gamma22();
    let src: Vec<f32> = (0..=20).flat_map(|i| [i as f32 / 20.0; 3]).collect();
    let dst = convert_f32_rgb(src_desc, dst_desc, &src);
    for (s, d) in src.iter().zip(dst.iter()) {
        let expected = linear_to_gamma22(*s);
        assert!(
            (d - expected).abs() < 1e-3,
            "OETF mismatch: linear({s}) -> {d}, expected {expected}"
        );
    }
}

#[test]
fn gamma22_linear_roundtrip_f32() {
    let src: Vec<f32> = (0..=100).flat_map(|i| [i as f32 / 100.0; 3]).collect();
    let lin = convert_f32_rgb(f32_rgb_gamma22(), f32_rgb_linear(), &src);
    let back = convert_f32_rgb(f32_rgb_linear(), f32_rgb_gamma22(), &lin);
    for (s, b) in src.iter().zip(back.iter()) {
        assert!(
            (s - b).abs() < 5e-4,
            "roundtrip drift: {s} -> {b} (delta {})",
            (s - b).abs()
        );
    }
}

/// Gamma22 → sRGB routes through linear (should be valid with matching primaries).
#[test]
fn gamma22_to_srgb_f32_via_linear() {
    // Same primaries, different TF: a round-trip of encoded values through linear.
    let src_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
    );
    let dst_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let src: Vec<f32> = (0..=10).flat_map(|i| [i as f32 / 10.0; 3]).collect();
    let dst = convert_f32_rgb(src_desc, dst_desc, &src);
    for (s, d) in src.iter().zip(dst.iter()) {
        // Path: gamma22-encoded -> linear (gamma22 EOTF) -> sRGB-encoded (sRGB OETF).
        let lin = gamma22_to_linear(*s);
        let expected = linear_srgb::tf::linear_to_srgb(lin);
        assert!(
            (d - expected).abs() < 2e-3,
            "gamma22->srgb via linear: {s} -> {d}, expected {expected}"
        );
    }
}

/// AdobeRGB F32 (gamma 2.2) → BT.2020 PQ F32: full HDR export path.
/// Verifies the planner wires linearize → gamut → PQ-encode.
#[test]
fn adobe_rgb_to_bt2020_pq_f32_preserves_neutral_gray() {
    let src_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
    )
    .with_primaries(ColorPrimaries::AdobeRgb);
    let dst_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    )
    .with_primaries(ColorPrimaries::Bt2020);

    // Neutral gray: AdobeRGB encodes as (v, v, v) with v^2.2 linear.
    let src: Vec<f32> = [0.25f32, 0.5, 0.75].iter().flat_map(|&v| [v; 3]).collect();
    let dst = convert_f32_rgb(src_desc, dst_desc, &src);

    // Gamut matrix preserves neutral gray (D65 → D65 white point match).
    // Expected: gamma22 EOTF(v), then identity on gray axis, then PQ OETF.
    for (chunk, &sv) in dst.chunks_exact(3).zip([0.25f32, 0.5, 0.75].iter()) {
        let lin = gamma22_to_linear(sv);
        let pq_expected = linear_srgb::tf::linear_to_pq(lin);
        for &d in chunk {
            assert!(
                (d - pq_expected).abs() < 5e-3,
                "gray axis not preserved: got {d}, expected {pq_expected} (src {sv})"
            );
        }
    }
}

/// AdobeRGB U8 → sRGB U8: round-trip through the plan that goes
/// NaiveU8→F32 → Gamma22 EOTF → gamut matrix → sRGB OETF → Naive F32→U8.
/// Verifies the planner picks the Gamma22 linearize/encode arms we added.
#[test]
fn adobe_rgb_u8_to_srgb_u8_neutral_gray() {
    let src_desc = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
    )
    .with_primaries(ColorPrimaries::AdobeRgb);
    let dst_desc = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt709);

    let src = [64u8, 64, 64, 128, 128, 128, 200, 200, 200];
    let mut dst = [0u8; 9];
    let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();
    conv.convert_row(&src, &mut dst, 3);

    // Neutral gray should stay neutral (same luminance on gamut axis).
    for px in dst.chunks_exact(3) {
        assert!(
            (px[0] as i32 - px[1] as i32).abs() <= 1 && (px[1] as i32 - px[2] as i32).abs() <= 1,
            "neutral gray should stay neutral: {px:?}"
        );
    }
    // Output channels should be > source channels since gamma 2.2 is a slightly
    // steeper EOTF than sRGB — re-encoding in sRGB yields a brighter signal
    // for the same linear light (at these mid-tones).
    assert!(
        dst[3] >= src[3],
        "mid gray should not darken: {} -> {}",
        src[3],
        dst[3]
    );
}

/// RGBA AdobeRGB → RGBA sRGB: with fully opaque alpha (1.0), alpha survives
/// the TF roundtrip exactly (pow(1, x) == 1). Verifies the plan composes
/// end-to-end for alpha-bearing layouts.
#[test]
fn adobe_rgb_rgba_f32_to_bt709_rgba_f32_opaque_alpha_preserved() {
    let src_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Gamma22,
    )
    .with_primaries(ColorPrimaries::AdobeRgb);
    let dst_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    )
    .with_primaries(ColorPrimaries::Bt709);

    let src = [0.5f32, 0.3, 0.7, 1.0, 0.9, 0.8, 0.6, 1.0];
    let mut dst = [0.0f32; 8];
    let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();
    conv.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        2,
    );

    assert!((dst[3] - 1.0).abs() < 1e-4, "alpha ch0: {}", dst[3]);
    assert!((dst[7] - 1.0).abs() < 1e-4, "alpha ch1: {}", dst[7]);
    // RGB channels should actually change (different primaries + TF).
    let rgb_changed = dst[0] != src[0] || dst[1] != src[1] || dst[2] != src[2];
    assert!(rgb_changed, "RGB channels should be transformed");
}
