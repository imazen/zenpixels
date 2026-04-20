//! Regression: `AlphaPolicy::CompositeOnto` used to produce wrong pixels
//! when the source descriptor declared `AlphaMode::Premultiplied`. The
//! `matte_composite` kernel uses the straight-alpha over operator
//! (`fg*a + bg*(1-a)`) after decoding to linear, so feeding it premul
//! source bytes made it multiply by `a` twice — `straight*a² + bg*(1-a)`.
//!
//! Fix: the planner now inserts `PremulToStraight` before `MatteComposite`
//! when the source alpha mode is `Premultiplied`, recovering straight
//! sRGB bytes (in our library's encoded-space premul convention) that
//! the kernel handles correctly.

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
    policy::{AlphaPolicy, ConvertOptions},
};
use zenpixels_convert::RowConverter;

fn rgba_u8(alpha: AlphaMode) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(alpha),
        TransferFunction::Srgb,
    )
}

fn rgb_u8() -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    )
}

fn opts_with_bg(r: u8, g: u8, b: u8) -> ConvertOptions {
    ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto { r, g, b })
}

fn convert_one_pixel(src_desc: PixelDescriptor, src_pixel: [u8; 4], bg: (u8, u8, u8)) -> [u8; 3] {
    let dst_desc = rgb_u8();
    let opts = opts_with_bg(bg.0, bg.1, bg.2);
    let mut c = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let mut dst = [0u8; 3];
    c.convert_row(&src_pixel, &mut dst, 1);
    dst
}

// ── Reference: what premul composite should produce ─────────────────────

/// Ground truth for premul-over-bg in linear light, matching what the
/// planner+kernel pair should produce after the fix.
///
/// Steps:
/// 1. Un-premultiply the source in encoded sRGB space: `straight = src/a` clipped.
/// 2. Decode straight sRGB to linear light.
/// 3. Over operator in linear: `out_lin = straight_lin * a + bg_lin * (1-a)`.
/// 4. Re-encode to sRGB u8.
fn reference_premul_composite(src_premul: [u8; 4], bg: (u8, u8, u8)) -> [u8; 3] {
    let a_byte = src_premul[3];
    let a = a_byte as f32 / 255.0;
    // Byte-space un-premul.
    let straight = if a_byte == 0 {
        [0u8, 0, 0]
    } else {
        let a32 = a_byte as u32;
        [
            ((src_premul[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
            ((src_premul[1] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
            ((src_premul[2] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
        ]
    };
    // Decode to linear, blend, re-encode.
    let mut out = [0u8; 3];
    for i in 0..3 {
        let fg_lin = srgb_eotf(straight[i] as f32 / 255.0);
        let bg_lin = srgb_eotf([bg.0, bg.1, bg.2][i] as f32 / 255.0);
        let out_lin = fg_lin * a + bg_lin * (1.0 - a);
        out[i] = (srgb_oetf(out_lin) * 255.0).round() as u8;
    }
    out
}

fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_oetf(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn premul_full_red_at_25pct_over_white() {
    // (64, 0, 0, 64) premul = "full-red at α=0.251 composited-over-black
    // stored as premul". Compositing onto solid white should produce near-white
    // (~255 in R, ~191 in G/B since the 75% background bleeds through).
    let src = [64u8, 0, 0, 64];
    let bg = (255u8, 255, 255);
    let actual = convert_one_pixel(rgba_u8(AlphaMode::Premultiplied), src, bg);
    let expected = reference_premul_composite(src, bg);
    for ch in 0..3 {
        let diff = (actual[ch] as i32 - expected[ch] as i32).abs();
        assert!(
            diff <= 1,
            "premul (64,0,0,64) over white: got {actual:?}, expected {expected:?}"
        );
    }
    // Full-red at 25% alpha over full-white should be nearly white in R, dim
    // in G/B. Critically, R must NOT be ~231 (the buggy pre-fix value).
    assert!(actual[0] >= 250, "R should be near 255: got {}", actual[0]);
}

#[test]
fn premul_half_red_over_black() {
    // (128, 0, 0, 128) premul. Over black should equal the premul RGB
    // (composite-over-black is the identity for premul).
    let src = [128u8, 0, 0, 128];
    let bg = (0u8, 0, 0);
    let actual = convert_one_pixel(rgba_u8(AlphaMode::Premultiplied), src, bg);
    let expected = reference_premul_composite(src, bg);
    for ch in 0..3 {
        let diff = (actual[ch] as i32 - expected[ch] as i32).abs();
        assert!(
            diff <= 1,
            "premul (128,0,0,128) over black: got {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn premul_zero_alpha_over_green_is_green() {
    // Fully transparent premul pixel has a=0. Result must be pure background.
    let src = [0u8, 0, 0, 0];
    let bg = (0u8, 255, 0);
    let actual = convert_one_pixel(rgba_u8(AlphaMode::Premultiplied), src, bg);
    assert_eq!(actual, [0u8, 255, 0], "a=0 premul must yield bg exactly");
}

#[test]
fn premul_full_alpha_equals_foreground() {
    // Opaque premul: bg must not be visible at all.
    // Premul (200, 100, 50, 255) = straight (200, 100, 50) at α=1.
    let src = [200u8, 100, 50, 255];
    let bg = (255u8, 0, 255); // magenta bg; should be invisible.
    let actual = convert_one_pixel(rgba_u8(AlphaMode::Premultiplied), src, bg);
    for ch in 0..3 {
        let diff = (actual[ch] as i32 - src[ch] as i32).abs();
        assert!(
            diff <= 1,
            "opaque premul over magenta: got {actual:?}, expected {:?}",
            &src[..3]
        );
    }
}

// ── Regression: straight-alpha path must still work ─────────────────────

#[test]
fn straight_full_red_at_25pct_over_white_unchanged() {
    // Straight alpha: (255, 0, 0, 64). Matches the "full-red at 25% over
    // white" case above but via the straight path (no un-premul needed).
    let src = [255u8, 0, 0, 64];
    let bg = (255u8, 255, 255);
    let actual = convert_one_pixel(rgba_u8(AlphaMode::Straight), src, bg);
    // Expected: straight kernel `fg*a + bg*(1-a)` in linear.
    let a = 64.0 / 255.0;
    let fg_r_lin = srgb_eotf(1.0);
    let bg_lin = srgb_eotf(1.0);
    let out_r_lin = fg_r_lin * a + bg_lin * (1.0 - a);
    let expected_r = (srgb_oetf(out_r_lin) * 255.0).round() as u8;
    let diff = (actual[0] as i32 - expected_r as i32).abs();
    assert!(
        diff <= 1,
        "straight (255,0,0,64) over white: got {actual:?}, expected R≈{expected_r}"
    );
    // Matches the premul-fixed case: the two representations should
    // produce the same visible pixel within ±1 LSB.
    let premul_equiv = convert_one_pixel(rgba_u8(AlphaMode::Premultiplied), [64u8, 0, 0, 64], bg);
    for ch in 0..3 {
        let diff = (actual[ch] as i32 - premul_equiv[ch] as i32).abs();
        assert!(
            diff <= 2,
            "straight vs premul representations of the same pixel disagree: \
             straight={actual:?}, premul={premul_equiv:?}"
        );
    }
}

// ── Plan-shape sanity ───────────────────────────────────────────────────

#[test]
fn plan_inserts_premul_to_straight_only_for_premul_source() {
    // Straight source: plan has MatteComposite but no PremulToStraight.
    let plan_straight = format!(
        "{:?}",
        zenpixels_convert::ConvertPlan::new_explicit(
            rgba_u8(AlphaMode::Straight),
            rgb_u8(),
            &opts_with_bg(255, 255, 255),
        )
        .unwrap()
    );
    assert!(plan_straight.contains("MatteComposite"));
    assert!(
        !plan_straight.contains("PremulToStraight"),
        "straight source must not get an un-premul step: {plan_straight}"
    );

    // Premul source: plan has both, in that order.
    let plan_premul = format!(
        "{:?}",
        zenpixels_convert::ConvertPlan::new_explicit(
            rgba_u8(AlphaMode::Premultiplied),
            rgb_u8(),
            &opts_with_bg(255, 255, 255),
        )
        .unwrap()
    );
    assert!(plan_premul.contains("PremulToStraight"));
    assert!(plan_premul.contains("MatteComposite"));
    let pre = plan_premul.find("PremulToStraight").unwrap();
    let mat = plan_premul.find("MatteComposite").unwrap();
    assert!(
        pre < mat,
        "PremulToStraight must precede MatteComposite: {plan_premul}"
    );
}
