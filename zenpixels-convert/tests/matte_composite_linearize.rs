//! Regression: `MatteComposite` used to blend sRGB-encoded f32/f16 pixel
//! data against a linearized matte — producing mathematically wrong colors
//! — when the plan was same-TF same-depth-float (no prior linearize step).
//! See issue #25.
//!
//! Fix: the planner now wraps `MatteComposite` with a linearize/delinearize
//! pair (plus F16↔F32 widen/pack for F16) when the pixel data entering the
//! step has a non-Linear, non-Unknown transfer function and a float channel
//! type. Integer paths unchanged — the existing U8/U16 kernel arms
//! hardcode sRGB EOTF/OETF inline, correct for the common sRGB case.

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
    policy::{AlphaPolicy, ConvertOptions},
};
use zenpixels_convert::RowConverter;

fn rgba_f32(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        tf,
    )
}

fn rgb_f32(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, tf)
}

fn rgba_f16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        tf,
    )
}

fn rgb_f16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::F16, ChannelLayout::Rgb, None, tf)
}

fn opts_with_bg(r: u8, g: u8, b: u8) -> ConvertOptions {
    ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto { r, g, b })
}

/// Reference sRGB EOTF for a single channel (piecewise polynomial per IEC 61966-2-1).
fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Reference sRGB OETF.
fn srgb_oetf(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Reference composite: take sRGB-encoded pixel + matte, do the over
/// operator in linear light, encode back to sRGB.
fn reference_srgb_composite(pixel: [f32; 4], matte_u8: (u8, u8, u8)) -> [f32; 3] {
    let (mr, mg, mb) = matte_u8;
    let mr_lin = srgb_eotf(mr as f32 / 255.0);
    let mg_lin = srgb_eotf(mg as f32 / 255.0);
    let mb_lin = srgb_eotf(mb as f32 / 255.0);

    let a = pixel[3].clamp(0.0, 1.0);
    let inv_a = 1.0 - a;

    let pr_lin = srgb_eotf(pixel[0]);
    let pg_lin = srgb_eotf(pixel[1]);
    let pb_lin = srgb_eotf(pixel[2]);

    let out_r_lin = pr_lin * a + mr_lin * inv_a;
    let out_g_lin = pg_lin * a + mg_lin * inv_a;
    let out_b_lin = pb_lin * a + mb_lin * inv_a;

    [
        srgb_oetf(out_r_lin),
        srgb_oetf(out_g_lin),
        srgb_oetf(out_b_lin),
    ]
}

/// F32 sRGB RGBA → F32 sRGB RGB with a grey matte. The sRGB-encoded pixel
/// and sRGB-encoded matte must both be linearized before the over operator,
/// then the result re-encoded. Tolerance 0.005 = 0.5% in normalized units;
/// pre-fix the error on this case was ~5-8%.
#[test]
fn f32_srgb_composite_matches_linear_light_reference() {
    // Mid-grey pixel at 50% alpha, blended onto black matte.
    // Under the buggy path (sRGB-encoded pixel blended as if linear),
    // output was ~0.25; the correct linear-light blend gives ~0.3639.
    let pixel: [f32; 4] = [0.5, 0.5, 0.5, 0.5];
    let matte = (0u8, 0u8, 0u8);

    let src_desc = rgba_f32(TransferFunction::Srgb);
    let dst_desc = rgb_f32(TransferFunction::Srgb);
    let opts = opts_with_bg(matte.0, matte.1, matte.2);

    let mut conv = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let mut dst = [0.0f32; 3];
    let src_bytes: [u8; 16] = bytemuck::cast(pixel);
    let mut dst_bytes = [0u8; 12];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    dst.copy_from_slice(bytemuck::cast_slice(&dst_bytes));

    let expected = reference_srgb_composite(pixel, matte);
    for (i, (got, exp)) in dst.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 0.005,
            "channel {i}: got {got}, expected {exp}, diff {diff}"
        );
    }
}

/// F16 sRGB RGBA → F16 sRGB RGB: same fix applies; F16 is widened to F32
/// for the blend and packed back afterwards.
#[test]
fn f16_srgb_composite_matches_linear_light_reference() {
    let pixel: [f32; 4] = [0.5, 0.5, 0.5, 0.5];
    let matte = (0u8, 0u8, 0u8);

    let src_desc = rgba_f16(TransferFunction::Srgb);
    let dst_desc = rgb_f16(TransferFunction::Srgb);
    let opts = opts_with_bg(matte.0, matte.1, matte.2);

    let mut conv = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();

    let src_bits: Vec<u16> = pixel
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let src_bytes: Vec<u8> = src_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut dst_bytes = vec![0u8; 6];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);

    let dst: Vec<f32> = dst_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    let expected = reference_srgb_composite(pixel, matte);
    for (i, (got, exp)) in dst.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        // F16 quantization adds ~1e-3 on top of the blend error budget.
        assert!(
            diff < 0.01,
            "channel {i}: got {got}, expected {exp}, diff {diff}"
        );
    }
}

/// Non-trivial matte + non-trivial pixel, F32 sRGB.
#[test]
fn f32_srgb_composite_red_over_blue_matte() {
    // Red-ish pixel at 40% alpha over a deep-blue matte.
    let pixel: [f32; 4] = [0.9, 0.1, 0.1, 0.4];
    let matte = (20u8, 30u8, 180u8);

    let src_desc = rgba_f32(TransferFunction::Srgb);
    let dst_desc = rgb_f32(TransferFunction::Srgb);
    let opts = opts_with_bg(matte.0, matte.1, matte.2);

    let mut conv = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let src_bytes: [u8; 16] = bytemuck::cast(pixel);
    let mut dst_bytes = [0u8; 12];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let dst: [f32; 3] = bytemuck::cast(dst_bytes);

    let expected = reference_srgb_composite(pixel, matte);
    for (i, (got, exp)) in dst.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 0.005,
            "channel {i}: got {got}, expected {exp}, diff {diff}"
        );
    }
}

/// Linear F32 path must NOT be wrapped (the wrap would be a no-op but
/// adds plan steps). Verify plan stays at a single MatteComposite step.
#[test]
fn f32_linear_composite_is_single_step() {
    use zenpixels_convert::ConvertPlan;

    let src_desc = rgba_f32(TransferFunction::Linear);
    let dst_desc = rgb_f32(TransferFunction::Linear);
    let opts = opts_with_bg(0, 0, 0);

    let plan = ConvertPlan::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let debug = format!("{:?}", plan);
    // One MatteComposite, no linearize/delinearize wrap.
    let composite_count = debug.matches("MatteComposite").count();
    let linearize_count = debug.matches("SrgbF32ToLinearF32").count();
    assert_eq!(composite_count, 1, "plan: {debug}");
    assert_eq!(linearize_count, 0, "plan should not linearize: {debug}");
}

/// Unknown TF must NOT be wrapped — we preserve bytes as-is when we don't
/// know the correct TF math (same convention as depth_steps).
#[test]
fn f32_unknown_tf_composite_is_single_step() {
    use zenpixels_convert::ConvertPlan;

    let src_desc = rgba_f32(TransferFunction::Unknown);
    let dst_desc = rgb_f32(TransferFunction::Unknown);
    let opts = opts_with_bg(0, 0, 0);

    let plan = ConvertPlan::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let debug = format!("{:?}", plan);
    let linearize_count = debug.matches("SrgbF32ToLinearF32").count();
    assert_eq!(linearize_count, 0, "Unknown TF: no wrap. plan: {debug}");
}
