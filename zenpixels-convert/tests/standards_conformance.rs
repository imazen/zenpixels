//! Regression suite for the cross-crate normative standards audit
//! (`ITU-T H.273`, `ITU-R BT.709/BT.2020/BT.2100-3/BT.2408/BT.2446-1`,
//! `ICC.1:2022-05`, `Adobe RGB (1998)`, `Safdar et al. (2017) Jzazbz`,
//! `W3C CSS Color 4/5`, `W3C Compositing-1`).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use zenpixels::hdr::{DiffuseWhite, HlgRendering, Nits};
use zenpixels::icc::IccIdentification;
use zenpixels::{
    AlphaMode, AlphaPolicy, ChannelLayout, ChannelType, Cicp, Color, ColorModel, ColorPrimaries,
    ColorProfileSource, ConvertOptions, DepthPolicy, IccParametricCurve, LumaCoefficients,
    NamedProfile, ParametricColor, PixelDescriptor, PixelFormat, RenderingIntent, SignalRange,
    TransferFunction,
};
use zenpixels_convert::css::CssColorError;
use zenpixels_convert::ext::ColorConvertExt;
use zenpixels_convert::{ConvertError, ConvertPlan, RowConverter};

const OKLCHF32: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Oklch,
    None,
    TransferFunction::Unknown,
);
const LCHF32: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Lch,
    None,
    TransferFunction::Unknown,
);
const JZAZBZF32: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Jzazbz,
    None,
    TransferFunction::Unknown,
);
const JZCZHZF32: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Jzczhz,
    None,
    TransferFunction::Unknown,
);
const HSLF32_SRGB: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Hsl,
    None,
    TransferFunction::Srgb,
);
const HSVF32_SRGB: PixelDescriptor = PixelDescriptor::new(
    ChannelType::F32,
    ChannelLayout::Hsv,
    None,
    TransferFunction::Srgb,
);

#[test]
fn cicp_from_descriptor_rejects_non_rgb_models() {
    assert_eq!(
        Cicp::from_descriptor(&PixelDescriptor::RGB8_SRGB),
        Some(Cicp::SRGB)
    );
    assert_eq!(
        Cicp::from_descriptor(&PixelDescriptor::GRAY8_SRGB),
        Some(Cicp::SRGB)
    );

    let lab_f32 = PixelDescriptor::from_pixel_format(
        PixelFormat::from_parts(ChannelType::F32, ChannelLayout::Lab, None).unwrap(),
    );
    let xyz_f32 = PixelDescriptor::from_pixel_format(
        PixelFormat::from_parts(ChannelType::F32, ChannelLayout::Xyz, None).unwrap(),
    );

    for non_rgb in [
        PixelDescriptor::CMYK8,
        PixelDescriptor::OKLABF32,
        OKLCHF32,
        lab_f32,
        LCHF32,
        xyz_f32,
        JZAZBZF32,
        JZCZHZF32,
        HSLF32_SRGB,
        HSVF32_SRGB,
    ] {
        assert_eq!(
            Cicp::from_descriptor(&non_rgb),
            None,
            "non-RGB/Gray descriptor {non_rgb:?} must not emit H.273 MC=0 CICP"
        );
    }
}

#[test]
fn named_profile_and_color_profile_source_cicp_guards() {
    // Narrow-range CICP must not resolve to full-range NamedProfile or (primaries, transfer).
    let srgb_narrow = Cicp::new(1, 13, 0, false);
    assert_eq!(NamedProfile::from_cicp(srgb_narrow), None);
    assert_eq!(
        ColorProfileSource::Cicp(srgb_narrow).primaries_transfer(),
        None
    );

    // YCbCr matrix coefficients must not resolve in ColorProfileSource::primaries_transfer.
    let bt709_ycbcr = Cicp::new(1, 1, 1, true);
    assert_eq!(
        ColorProfileSource::Cicp(bt709_ycbcr).primaries_transfer(),
        None
    );

    // BT.2100 PQ/HLG reject MC=2 (Unspecified) and MC=3 / MC=18..=255 (Reserved)
    // while keeping assigned video matrices (0, 1, 6, 9) working.
    for bad_mc in [2u8, 3, 18, 255] {
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 16, bad_mc, true)),
            None
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 18, bad_mc, true)),
            None
        );
    }
    for ok_mc in [0u8, 1, 6, 9] {
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 16, ok_mc, true)),
            Some(NamedProfile::Bt2020Pq)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 18, ok_mc, true)),
            Some(NamedProfile::Bt2020Hlg)
        );
    }
}

#[test]
fn icc_parametric_curve_requires_positive_a_and_extract_cicp_validates_header() {
    // ICC.1:2022-05 §10.18: domain threshold x >= -b / a requires a > 0.
    assert!(IccParametricCurve::try_type1_cie122(2.4, -1.0, 0.0).is_none());
    assert!(IccParametricCurve::try_type1_cie122(2.4, 0.0, 0.0).is_none());
    assert!(IccParametricCurve::try_type2_iec61966_3(2.4, -1.0, 0.0, 0.0).is_none());
    assert!(IccParametricCurve::try_type1_cie122(2.4, 1.0, 0.0).is_some());

    // Minimal v4.4 mntr profile with valid cicp tag succeeds, while v2 / link / bad full_range fail.
    let mut icc = vec![0u8; 256];
    icc[0..4].copy_from_slice(&256u32.to_be_bytes());
    icc[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
    icc[12..16].copy_from_slice(b"mntr");
    icc[16..20].copy_from_slice(b"RGB ");
    icc[20..24].copy_from_slice(b"XYZ ");
    icc[36..40].copy_from_slice(b"acsp");
    icc[128..132].copy_from_slice(&1u32.to_be_bytes());
    icc[132..136].copy_from_slice(b"cicp");
    icc[136..140].copy_from_slice(&144u32.to_be_bytes());
    icc[140..144].copy_from_slice(&12u32.to_be_bytes());
    icc[144..148].copy_from_slice(b"cicp");
    icc[152..156].copy_from_slice(&[1, 13, 0, 1]);
    assert_eq!(Cicp::from_icc(&icc), Some(Cicp::SRGB));
    let _ = IccIdentification::from_icc(&icc);

    let mut v2 = icc.clone();
    v2[8..12].copy_from_slice(&[2, 0x40, 0, 0]);
    assert_eq!(Cicp::from_icc(&v2), None);
}

#[test]
fn hlg_rendering_negative_zero_canonical_and_pluge_sub_black_lift() {
    let neg_zero = HlgRendering::new(Nits::new(1000.0), -0.0, 1.2).unwrap();
    assert_eq!(neg_zero, HlgRendering::REFERENCE_1000_NIT);

    let hash_of = |h: HlgRendering| {
        let mut s = DefaultHasher::new();
        h.hash(&mut s);
        s.finish()
    };
    assert_eq!(hash_of(neg_zero), hash_of(HlgRendering::REFERENCE_1000_NIT));

    // ITU-R BT.2100-3 Table 5 Note 5h: with L_W = 1000 cd/m² and L_B = 0.2 cd/m²,
    // gamma = 1.2, beta = sqrt(3 * (0.2 / 1000)^(1 / 1.2)) ≈ 0.020776.
    // Signal E' = 0 maps to L_B; PLUGE sub-black in (-beta / (1 - beta), 0) maps to (0, L_B)
    // even when clip_out_of_gamut == true!
    let hlg_cond = HlgRendering::new(Nits::new(1000.0), 0.2, 1.2).unwrap();
    let beta = (3.0_f64 * (0.2_f64 / 1000.0).powf(1.0 / 1.2)).sqrt() as f32;
    let pluge_mid = -0.5 * beta / (1.0 - beta);
    let pluge_floor = -beta / (1.0 - beta) - 0.01;

    let hlg_desc = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(ColorPrimaries::Bt2020)
        .with_transfer(TransferFunction::Hlg);
    let lin_desc = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
    let opts = ConvertOptions::permissive()
        .with_hlg_rendering(hlg_cond)
        .with_sdr_white(DiffuseWhite::new(1000.0))
        .with_clip_out_of_gamut(true);
    let mut conv = RowConverter::new_explicit(hlg_desc, lin_desc, &opts).unwrap();

    let src = [
        [0.0_f32, 0.0, 0.0],
        [pluge_mid, pluge_mid, pluge_mid],
        [pluge_floor, pluge_floor, pluge_floor],
    ];
    let mut dst = [[0.0_f32; 3]; 3];
    conv.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        3,
    );
    let y_zero = dst[0][1];
    let y_pluge = dst[1][1];
    let y_below = dst[2][1];
    assert!(
        (y_zero - 0.0002).abs() < 1e-6,
        "E'=0 must render at L_B / L_W = 0.0002, got {y_zero}"
    );
    assert!(
        y_pluge > 0.0 && y_pluge < y_zero,
        "BT.2100-3 Note 5h PLUGE sub-black must lie strictly in (0, L_B): got {y_pluge} vs L_B={y_zero}"
    );
    assert_eq!(y_below, 0.0, "below -beta/(1-beta) must clamp at 0.0");
}

#[test]
fn adobe_rgb_jzazbz_and_lch_roundtrip_matches_exact_rationals() {
    // Linear AdobeRGB <-> JZAZBZF32 and LCHF32 roundtrip using exact Adobe RGB (1998) §4.3.4.1 XYZ matrices.
    let adobe_lin = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb);
    for target in [JZAZBZF32, LCHF32] {
        let mut fwd = RowConverter::new(adobe_lin, target).unwrap();
        let mut inv = RowConverter::new(target, adobe_lin).unwrap();
        let px = [0.2_f32, 0.85, 0.3];
        let mut mid = [0.0_f32; 3];
        let mut rt = [0.0_f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&px),
            bytemuck::cast_slice_mut(&mut mid),
            1,
        );
        inv.convert_row(
            bytemuck::cast_slice(&mid),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for c in 0..3 {
            assert!(
                (rt[c] - px[c]).abs() < 1e-4,
                "AdobeRGB {target:?} rt: {rt:?} vs {px:?}"
            );
        }
    }
}

#[test]
fn cross_gamut_bt2020_pq_jzazbz_and_jzczhz_preserve_wide_gamut() {
    // Saturated Rec.2020 green (outside sRGB / BT.709 gamut) in PQ F32.
    // Must convert directly to/from JZAZBZF32 and JZCZHZF32 without routing
    // through BT.709 PQ (which would clip negative BT.709 R/B!).
    let bt2020_pq = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(ColorPrimaries::Bt2020)
        .with_transfer(TransferFunction::Pq);
    let px = [0.05_f32, 0.72, 0.08];

    for perceptual_desc in [JZAZBZF32, JZCZHZF32] {
        let mut fwd = RowConverter::new(bt2020_pq, perceptual_desc).unwrap();
        let mut inv = RowConverter::new(perceptual_desc, bt2020_pq).unwrap();
        let mut mid = [0.0_f32; 3];
        let mut rt = [0.0_f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&px),
            bytemuck::cast_slice_mut(&mut mid),
            1,
        );
        inv.convert_row(
            bytemuck::cast_slice(&mid),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for c in 0..3 {
            assert!(
                (rt[c] - px[c]).abs() < 2e-4,
                "Rec.2020 PQ <-> {perceptual_desc:?} clipped wide-gamut ch{c}: got {rt:?}, want {px:?}"
            );
        }
    }
}

#[test]
fn jzazbz_diffuse_white_10000_nits_anchor_reaches_peak_jz_one() {
    let lin = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
    let jab = JZAZBZF32;

    // Default unanchored: linear 1.0 = 203 cd/m² -> Jz ≈ 0.2220.
    let mut def_conv = RowConverter::new(lin, jab).unwrap();
    let white = [1.0_f32, 1.0, 1.0];
    let mut out_203 = [0.0_f32; 3];
    def_conv.convert_row(
        bytemuck::cast_slice(&white),
        bytemuck::cast_slice_mut(&mut out_203),
        1,
    );
    assert!(
        (out_203[0] - 0.2220).abs() < 5e-3,
        "203-nit Jz: {out_203:?}"
    );

    // Explicit 10,000 cd/m² anchor (`pq_scale == 1.0`): linear 1.0 = 10,000 cd/m² D65
    // -> Safdar (2017) Jz ≈ 0.98861 (vs 0.2220 at 203 cd/m²)!
    let opts_10k = ConvertOptions::permissive().with_sdr_white(DiffuseWhite::new(10_000.0));
    let mut conv_10k = RowConverter::new_explicit(lin, jab, &opts_10k).unwrap();
    let mut inv_10k = RowConverter::new_explicit(jab, lin, &opts_10k).unwrap();
    let mut out_10k = [0.0_f32; 3];
    let mut rt_10k = [0.0_f32; 3];
    conv_10k.convert_row(
        bytemuck::cast_slice(&white),
        bytemuck::cast_slice_mut(&mut out_10k),
        1,
    );
    assert!(
        (out_10k[0] - 0.98861).abs() < 1e-3,
        "10,000-nit anchored D65 linear 1.0 must reach Safdar Jz ≈ 0.98861, got {out_10k:?}"
    );
    inv_10k.convert_row(
        bytemuck::cast_slice(&out_10k),
        bytemuck::cast_slice_mut(&mut rt_10k),
        1,
    );
    for r in rt_10k {
        assert!((r - 1.0).abs() < 1e-4, "10,000-nit rt: {rt_10k:?}");
    }
}

#[test]
fn jzazbz_and_hsl_extended_range_roundtrip_without_poles() {
    let unclipped = ConvertOptions::permissive().with_clip_out_of_gamut(false);

    // Sub-black / imaginary linear RGB values (including tiny negatives near -1e-6)
    // must remain finite and invert cleanly in Jzazbz.
    let mut to_jab =
        RowConverter::new_explicit(PixelDescriptor::RGBF32_LINEAR, JZAZBZF32, &unclipped).unwrap();
    let mut from_jab =
        RowConverter::new_explicit(JZAZBZF32, PixelDescriptor::RGBF32_LINEAR, &unclipped).unwrap();
    for px in [
        [-0.25_f32, 0.5, 1.4],
        [-5e-7, -5e-7, -5e-7],
        [-0.5, -0.1, 0.2],
    ] {
        let mut jab = [0.0_f32; 3];
        let mut rt = [0.0_f32; 3];
        to_jab.convert_row(
            bytemuck::cast_slice(&px),
            bytemuck::cast_slice_mut(&mut jab),
            1,
        );
        assert!(
            jab.iter().all(|v| v.is_finite()),
            "non-finite jab for {px:?}: {jab:?}"
        );
        from_jab.convert_row(
            bytemuck::cast_slice(&jab),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for c in 0..3 {
            assert!(
                (rt[c] - px[c]).abs() < 2e-4,
                "extended Jzazbz roundtrip failed for {px:?}: got {rt:?}"
            );
        }
    }

    // Extended-range HSL with L > 1 and L < 0 (W3C CSS Color 4 §7.2 / §19).
    let srgb_f32 = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let hsl_f32 = HSLF32_SRGB;
    let mut to_hsl = RowConverter::new_explicit(srgb_f32, hsl_f32, &unclipped).unwrap();
    let mut from_hsl = RowConverter::new_explicit(hsl_f32, srgb_f32, &unclipped).unwrap();
    for px in [[0.8_f32, 1.4, 1.6], [-0.8, -0.3, -0.1], [-0.2, 0.5, 1.3]] {
        let mut hsl = [0.0_f32; 3];
        let mut rt = [0.0_f32; 3];
        to_hsl.convert_row(
            bytemuck::cast_slice(&px),
            bytemuck::cast_slice_mut(&mut hsl),
            1,
        );
        assert!(
            hsl[1] >= 0.0,
            "CSS HSL saturation must be non-negative: {hsl:?}"
        );
        from_hsl.convert_row(
            bytemuck::cast_slice(&hsl),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for c in 0..3 {
            assert!(
                (rt[c] - px[c]).abs() < 1e-5,
                "extended HSL roundtrip failed for {px:?} (hsl={hsl:?}): got {rt:?}"
            );
        }
    }
}

#[test]
fn straight_to_premul_with_gamut_change_premultiplies_after_gamut_matrix() {
    // Display P3 straight RGBA -> BT.709 premultiplied RGBA (same transfer = Srgb, same depth).
    // Gamut conversion MUST run on straight samples before premultiplying in destination encoding!
    let p3_straight_u8 = PixelDescriptor::RGBA8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let bt709_straight_u8 = PixelDescriptor::RGBA8_SRGB;
    let bt709_premul_u8 = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));

    let src = [255_u8, 40, 20, 128];
    let mut straight_mid = [0_u8; 4];
    let mut ref_premul = [0_u8; 4];
    let mut direct_premul = [0_u8; 4];

    RowConverter::new(p3_straight_u8, bt709_straight_u8)
        .unwrap()
        .convert_row(&src, &mut straight_mid, 1);
    RowConverter::new(bt709_straight_u8, bt709_premul_u8)
        .unwrap()
        .convert_row(&straight_mid, &mut ref_premul, 1);
    RowConverter::new(p3_straight_u8, bt709_premul_u8)
        .unwrap()
        .convert_row(&src, &mut direct_premul, 1);

    for c in 0..4 {
        assert!(
            direct_premul[c].abs_diff(ref_premul[c]) <= 1,
            "P3 straight -> BT.709 premul U8 ch{c}: direct={direct_premul:?} vs two-step={ref_premul:?}"
        );
    }

    // Same check in F32.
    let p3_straight_f32 = PixelDescriptor::RGBAF32_LINEAR
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_transfer(TransferFunction::Srgb);
    let bt709_straight_f32 = PixelDescriptor::RGBAF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let bt709_premul_f32 = bt709_straight_f32.with_alpha(Some(AlphaMode::Premultiplied));

    let src_f = [1.0_f32, 0.2, 0.1, 0.5];
    let mut mid_f = [0.0_f32; 4];
    let mut want_f = [0.0_f32; 4];
    let mut got_f = [0.0_f32; 4];
    RowConverter::new(p3_straight_f32, bt709_straight_f32)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&src_f),
            bytemuck::cast_slice_mut(&mut mid_f),
            1,
        );
    RowConverter::new(bt709_straight_f32, bt709_premul_f32)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&mid_f),
            bytemuck::cast_slice_mut(&mut want_f),
            1,
        );
    RowConverter::new(p3_straight_f32, bt709_premul_f32)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&src_f),
            bytemuck::cast_slice_mut(&mut got_f),
            1,
        );
    for c in 0..4 {
        assert!(
            (got_f[c] - want_f[c]).abs() < 1e-5,
            "P3 straight -> BT.709 premul F32: got={got_f:?} vs want={want_f:?}"
        );
    }
}

#[test]
fn narrow_range_rejects_premul_matte_and_non_rgb_models() {
    let rgba_straight = PixelDescriptor::RGBA8_SRGB.with_signal_range(SignalRange::Narrow);
    let rgba_premul = rgba_straight.with_alpha(Some(AlphaMode::Premultiplied));
    let rgb_narrow = PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow);

    assert!(matches!(
        ConvertPlan::new(rgba_straight, rgba_premul)
            .unwrap_err()
            .error(),
        ConvertError::NoPath { .. }
    ));
    assert!(matches!(
        ConvertPlan::new(rgba_premul, rgba_straight)
            .unwrap_err()
            .error(),
        ConvertError::NoPath { .. }
    ));

    let matte_opts = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(0, 0, 0),
        reference_white: None,
    });
    assert!(matches!(
        ConvertPlan::new_explicit(rgba_straight, rgb_narrow, &matte_opts)
            .unwrap_err()
            .error(),
        ConvertError::NoPath { .. }
    ));

    let narrow_lin_f32 = PixelDescriptor::RGBF32_LINEAR.with_signal_range(SignalRange::Narrow);
    let narrow_oklab = PixelDescriptor::OKLABF32.with_signal_range(SignalRange::Narrow);
    assert!(matches!(
        ConvertPlan::new_explicit(narrow_lin_f32, narrow_oklab, &ConvertOptions::permissive())
            .unwrap_err()
            .error(),
        ConvertError::NoPath { .. }
    ));
}

#[test]
fn premul_odd_divisor_exact_rounding_u8_ga_and_u16() {
    // Exhaustively verify all 65,536 (gray, alpha) U8 pairs:
    // GrayAlpha8 Straight -> Premultiplied must equal exact round(g * a / 255.0)
    // and match Rgba8 Straight -> Premultiplied bit-for-bit.
    let ga8_s = PixelDescriptor::GRAYA8_SRGB;
    let ga8_p = ga8_s.with_alpha(Some(AlphaMode::Premultiplied));
    let rgba8_s = PixelDescriptor::RGBA8_SRGB;
    let rgba8_p = rgba8_s.with_alpha(Some(AlphaMode::Premultiplied));

    let mut conv_ga = RowConverter::new(ga8_s, ga8_p).unwrap();
    let mut conv_rgba = RowConverter::new(rgba8_s, rgba8_p).unwrap();

    let mut ga_in = vec![0_u8; 256 * 2];
    let mut ga_out = vec![0_u8; 256 * 2];
    let mut rgba_in = vec![0_u8; 256 * 4];
    let mut rgba_out = vec![0_u8; 256 * 4];

    for g in 0..=255_u8 {
        for a in 0..=255_u8 {
            let idx = usize::from(a);
            ga_in[idx * 2] = g;
            ga_in[idx * 2 + 1] = a;
            rgba_in[idx * 4] = g;
            rgba_in[idx * 4 + 1] = g;
            rgba_in[idx * 4 + 2] = g;
            rgba_in[idx * 4 + 3] = a;
        }
        conv_ga.convert_row(&ga_in, &mut ga_out, 256);
        conv_rgba.convert_row(&rgba_in, &mut rgba_out, 256);
        for a in 0..=255_u8 {
            let idx = usize::from(a);
            let exact = ((u32::from(g) * u32::from(a) + 127) / 255) as u8;
            assert_eq!(ga_out[idx * 2], exact, "GA8 premul ({g}, {a})");
            assert_eq!(
                ga_out[idx * 2],
                rgba_out[idx * 4],
                "GA8 vs RGBA8 ({g}, {a})"
            );
        }
    }

    // U16 midpoint checks (`+ 32767` vs `+ 32768`):
    // c = 1, a = 32767 -> 32767 / 65535 = 0.49999237... -> must round to 0, NOT 1!
    // c = 1, a = 32768 -> 32768 / 65535 = 0.50000762... -> must round to 1!
    let ga16_s = PixelDescriptor::GRAYA16_SRGB;
    let ga16_p = ga16_s.with_alpha(Some(AlphaMode::Premultiplied));
    let rgba16_s = PixelDescriptor::RGBA16_SRGB;
    let rgba16_p = rgba16_s.with_alpha(Some(AlphaMode::Premultiplied));

    let mut ga16_out = [0_u16; 4];
    RowConverter::new(ga16_s, ga16_p).unwrap().convert_row(
        bytemuck::cast_slice(&[1_u16, 32767, 1, 32768]),
        bytemuck::cast_slice_mut(&mut ga16_out),
        2,
    );
    assert_eq!(ga16_out, [0, 32767, 1, 32768]);

    let mut rgba16_out = [0_u16; 8];
    RowConverter::new(rgba16_s, rgba16_p).unwrap().convert_row(
        bytemuck::cast_slice(&[1_u16, 65533, 0, 32767, 1, 65533, 65535, 32768]),
        bytemuck::cast_slice_mut(&mut rgba16_out),
        2,
    );
    assert_eq!(rgba16_out, [0, 32766, 0, 32767, 1, 32767, 32768, 32768]);
}

#[test]
fn rgba_to_grayalpha_u8_bt709_matches_exact_bt709_table_4_weights() {
    let opts = ConvertOptions::permissive().with_luma(Some(LumaCoefficients::Bt709));
    let mut rgba_ga = RowConverter::new_explicit(
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::GRAYA8_SRGB,
        &opts,
    )
    .unwrap();

    for r in (0..=255_u8).step_by(15) {
        for g in (0..=255_u8).step_by(15) {
            for b in (0..=255_u8).step_by(15) {
                let want =
                    (0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b) + 0.5)
                        .clamp(0.0, 255.0) as u8;
                let mut ga = [0_u8; 2];
                rgba_ga.convert_row(&[r, g, b, 200], &mut ga, 1);
                assert_eq!(ga, [want, 200], "({r}, {g}, {b}) mismatch");
            }
        }
    }
}

#[test]
fn depth_policy_truncate_floors_instead_of_rounding() {
    let trunc_opts = ConvertOptions::permissive().with_depth_policy(DepthPolicy::Truncate);
    let round_opts = ConvertOptions::permissive().with_depth_policy(DepthPolicy::Round);

    // U16 -> U8: 0x00FF (255) truncates to 0 (`255 >> 8`), rounds to 1 (`round(255 / 257)`);
    // 0x01FF (511) truncates to 1 (`511 >> 8`), rounds to 2 (`round(511 / 257)`).
    let mut u16_trunc = RowConverter::new_explicit(
        PixelDescriptor::GRAY16_SRGB,
        PixelDescriptor::GRAY8_SRGB,
        &trunc_opts,
    )
    .unwrap();
    let mut u16_round = RowConverter::new_explicit(
        PixelDescriptor::GRAY16_SRGB,
        PixelDescriptor::GRAY8_SRGB,
        &round_opts,
    )
    .unwrap();
    let src16 = [0x00FF_u16, 0x01FF, 0xFFFF];
    let mut out_t = [0_u8; 3];
    let mut out_r = [0_u8; 3];
    u16_trunc.convert_row(bytemuck::cast_slice(&src16), &mut out_t, 3);
    u16_round.convert_row(bytemuck::cast_slice(&src16), &mut out_r, 3);
    assert_eq!(out_t, [0, 1, 0xFF]);
    assert_eq!(out_r, [1, 2, 0xFF]);

    // Linear F32 -> Linear U8: 1.9 / 255.0 truncates to 1, rounds to 2.
    let lin_u8 = PixelDescriptor::GRAY8_SRGB.with_transfer(TransferFunction::Linear);
    let lin_f32 = PixelDescriptor::GRAYF32_LINEAR;
    let srcf = [1.9_f32 / 255.0, 127.8 / 255.0, 1.0];
    let mut ft = [0_u8; 3];
    let mut fr = [0_u8; 3];
    RowConverter::new_explicit(lin_f32, lin_u8, &trunc_opts)
        .unwrap()
        .convert_row(bytemuck::cast_slice(&srcf), &mut ft, 3);
    RowConverter::new_explicit(lin_f32, lin_u8, &round_opts)
        .unwrap()
        .convert_row(bytemuck::cast_slice(&srcf), &mut fr, 3);
    assert_eq!(ft, [1, 127, 255]);
    assert_eq!(fr, [2, 128, 255]);
}

#[test]
fn linear_premul_matte_composite_preserves_emissive_zero_alpha_and_scrubs_nan() {
    let premul_lin = PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied));
    let rgb_lin = PixelDescriptor::RGBF32_LINEAR;
    let bg = Color::try_new(rgb_lin, bytemuck::cast_slice(&[0.1_f32, 0.2, 0.3])).unwrap();
    let opts = ConvertOptions::permissive()
        .with_clip_out_of_gamut(false)
        .with_alpha_policy(AlphaPolicy::CompositeOnto {
            background: bg,
            reference_white: None,
        });

    let mut conv = RowConverter::new_explicit(premul_lin, rgb_lin, &opts).unwrap();
    // Pixel 0: emissive additive glow `(rgb > 0, alpha == 0)` -> `cs + bg * (1 - 0) = cs + bg`.
    // Pixel 1: NaN alpha -> scrubbed to 0.0 -> `cs + bg`.
    let src = [[0.4_f32, 0.2, 0.1, 0.0], [0.0, 0.0, 0.0, f32::NAN]];
    let mut dst = [[0.0_f32; 3]; 2];
    conv.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        2,
    );
    assert!(
        (dst[0][0] - 0.5).abs() < 1e-6
            && (dst[0][1] - 0.4).abs() < 1e-6
            && (dst[0][2] - 0.4).abs() < 1e-6
    );
    assert!(
        (dst[1][0] - 0.1).abs() < 1e-6
            && (dst[1][1] - 0.2).abs() < 1e-6
            && (dst[1][2] - 0.3).abs() < 1e-6
    );
}

#[test]
fn oklab_subnormal_cbrt_stays_accurate() {
    // Subnormal linear input `1e-40` has cbrt `≈ 4.6415888e-14`.
    let mut conv =
        RowConverter::new(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::OKLABF32).unwrap();
    let src = [1e-40_f32, 1e-40, 1e-40];
    let mut oklab = [0.0_f32; 3];
    conv.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut oklab),
        1,
    );
    let want_l = 1e-40_f64.cbrt() as f32;
    assert!(
        (oklab[0] - want_l).abs() < 1e-16,
        "subnormal Oklab L = {}, want {want_l}",
        oklab[0]
    );
}

#[cfg(feature = "bt2446a")]
#[test]
fn bt2446a_clamps_super_peak_to_sdr_peak() {
    let tm = zenpixels_convert::hdr::Bt2446A::new(1000.0, 100.0);
    let at_peak = tm.map_rgb([1.0, 1.0, 1.0]);
    let super_peak = tm.map_rgb([2.5, 2.5, 2.5]);
    assert_eq!(super_peak, at_peak);
    assert_eq!(tm.map_rgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
}

#[test]
fn css_color_4_grammar_and_system_colors() {
    // Valid modern space-separated mixed & bare-number forms, plus CR/FF whitespace.
    assert!(Color::parse_css("rgb(100% 0 128)").is_ok());
    assert!(Color::parse_css("hsl(120 100 50)").is_ok());
    assert!(Color::parse_css("\r\x0Crgb(10\r20\x0C30)\r").is_ok());

    // Invalid legacy comma mixes, trailing/empty commas, trailing '.', space before '(', NBSP.
    for invalid in [
        "rgb(100%, 0, 0)",
        "hsl(120, 100, 50)",
        "rgb(1, 2, 3,)",
        "rgb(, 1, 2, 3)",
        "rgb(1,, 2, 3)",
        "rgb(1. 2 3)",
        "rgb (1, 2, 3)",
        "\u{00A0}red",
    ] {
        assert_eq!(
            Color::parse_css(invalid),
            Err(CssColorError::Invalid),
            "{invalid:?} must be rejected as Invalid"
        );
    }

    // CSS Color 4 §6.2 <system-color> keywords and CSS Color 5 relative colors -> Unsupported.
    for unsupported in [
        "accentcolor",
        "accentcolortext",
        "activetext",
        "buttonborder",
        "buttonface",
        "buttontext",
        "field",
        "fieldtext",
        "graytext",
        "highlight",
        "highlighttext",
        "linktext",
        "mark",
        "marktext",
        "selecteditem",
        "selecteditemtext",
        "visitedtext",
        "rgb(from red r g b)",
        "oklch(from blue l c h)",
        "color(from red srgb r g b)",
    ] {
        assert_eq!(
            Color::parse_css(unsupported),
            Err(CssColorError::Unsupported),
            "{unsupported:?} must return CssColorError::Unsupported"
        );
    }
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn moxcms_declines_absolute_colorimetric_and_preserves_model_consistency() {
    use zenpixels_convert::cms::PluggableCms;
    let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
    let dst = ColorProfileSource::Named(NamedProfile::Srgb);
    let abs_opts =
        ConvertOptions::permissive().with_rendering_intent(RenderingIntent::AbsoluteColorimetric);
    assert!(
        zenpixels_convert::MoxCms
            .build_source_transform(
                src,
                dst.clone(),
                PixelFormat::Rgb8,
                PixelFormat::Rgb8,
                &abs_opts
            )
            .is_none()
    );

    let mut param = ParametricColor::from_cicp(Cicp::DISPLAY_P3).unwrap();
    param.rendering_intent = RenderingIntent::AbsoluteColorimetric;
    assert!(
        zenpixels_convert::MoxCms
            .build_source_transform(
                ColorProfileSource::Parametric(param),
                dst,
                PixelFormat::Rgb8,
                PixelFormat::Rgb8,
                &ConvertOptions::permissive(),
            )
            .is_none()
    );

    let pf = PixelFormat::from_parts(ChannelType::F32, ChannelLayout::Jzazbz, None).unwrap();
    assert_eq!(pf.color_model(), ColorModel::Jzazbz);
}

#[test]
fn decisions_a1_b1_c1_d1_conformance() {
    use zenpixels::GamutMapping;
    use zenpixels_convert::icc_profiles::SynthesizedIcc;

    // A1: Synthesized RGB ICC profiles (including PQ TC=16 and Display P3 exports)
    // carry ICC.1:2022-05 §8.2 required 'desc' and 'cprt' mluc tags and remain identifiable.
    for cicp in [
        Cicp::DISPLAY_P3,
        Cicp::new(12, 1, 0, true),
        Cicp::new(9, 16, 0, true),
        Cicp::new(9, 14, 0, true),
    ] {
        let SynthesizedIcc::Profile(icc) = SynthesizedIcc::for_cicp(cicp) else {
            panic!("expected synthesized profile for {cicp:?}");
        };
        let tag_count = u32::from_be_bytes(icc[128..132].try_into().unwrap()) as usize;
        let mut has_desc = false;
        let mut has_cprt = false;
        for i in 0..tag_count {
            let entry = &icc[132 + 12 * i..144 + 12 * i];
            let off = u32::from_be_bytes(entry[4..8].try_into().unwrap()) as usize;
            let len = u32::from_be_bytes(entry[8..12].try_into().unwrap()) as usize;
            if &entry[..4] == b"desc" && &icc[off..off + 4] == b"mluc" && len >= 16 {
                has_desc = true;
            }
            if &entry[..4] == b"cprt" && &icc[off..off + 4] == b"mluc" && len >= 16 {
                has_cprt = true;
            }
        }
        assert!(has_desc && has_cprt, "{cicp:?} missing 'desc' or 'cprt'");
        assert_eq!(Cicp::from_icc(&icc), Some(cicp));
    }

    // B1: W3C CSS Color HDR 1 §4.1 `color(rec2100-linear 1 1 1)` == sRGB white `#ffffff`.
    let rec2100_white = Color::parse_css("color(rec2100-linear 1 1 1)").unwrap();
    assert!(rec2100_white.same_color(&Color::srgb8(255, 255, 255)));
    let mut srgb8_out = [0_u8; 3];
    RowConverter::new(rec2100_white.descriptor(), PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .convert_row(rec2100_white.as_bytes(), &mut srgb8_out, 1);
    assert_eq!(srgb8_out, [255, 255, 255]);

    // C1: ITU-T H.273 Table 3 TC=14 and TC=15 (BT.2020 10/12-bit SDR) resolve to TransferFunction::Bt709.
    for tc in [14_u8, 15] {
        let bt2020_sdr = Cicp::new(9, tc, 0, true);
        assert_eq!(
            TransferFunction::from_cicp(tc),
            Some(TransferFunction::Bt709)
        );
        assert_eq!(
            NamedProfile::from_cicp(bt2020_sdr),
            Some(NamedProfile::Bt2020)
        );
        assert_eq!(
            ColorProfileSource::Cicp(bt2020_sdr).primaries_transfer(),
            Some((ColorPrimaries::Bt2020, TransferFunction::Bt709))
        );
        let desc = bt2020_sdr.try_to_descriptor(PixelFormat::Rgb16).unwrap();
        assert_eq!(desc.primaries, ColorPrimaries::Bt2020);
        assert_eq!(desc.transfer(), TransferFunction::Bt709);
    }

    // D1: Adobe RGB (1998) supports native Oklab, Oklch, map_to_gamut, same_color, and SoftCompress.
    let adobe_lin = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb);
    for perceptual in [PixelDescriptor::OKLABF32, OKLCHF32] {
        let mut fwd = RowConverter::new(adobe_lin, perceptual).unwrap();
        let mut inv = RowConverter::new(perceptual, adobe_lin).unwrap();
        let px = [0.25_f32, 0.80, 0.35];
        let mut mid = [0.0_f32; 3];
        let mut rt = [0.0_f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&px),
            bytemuck::cast_slice_mut(&mut mid),
            1,
        );
        inv.convert_row(
            bytemuck::cast_slice(&mid),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for c in 0..3 {
            assert!(
                (rt[c] - px[c]).abs() < 1e-4,
                "AdobeRGB {perceptual:?}: {rt:?} vs {px:?}"
            );
        }
    }
    let adobe_white =
        Color::try_new(adobe_lin, bytemuck::cast_slice(&[1.0_f32, 1.0, 1.0])).unwrap();
    assert!(adobe_white.same_color(&Color::srgb8(255, 255, 255)));

    let bt2020_green = Color::try_new(
        PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt2020),
        &[0, 255, 0],
    )
    .unwrap();
    let adobe_srgb = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::AdobeRgb);
    assert!(bt2020_green.map_to_gamut(adobe_srgb).is_ok());

    let soft_opts = ConvertOptions::permissive().with_gamut_mapping(GamutMapping::SoftCompress);
    let mut soft_conv = RowConverter::new_explicit(
        PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020),
        adobe_lin,
        &soft_opts,
    )
    .unwrap();
    let oog = [0.0_f32, 1.0, 0.0];
    let mut compressed = [0.0_f32; 3];
    soft_conv.convert_row(
        bytemuck::cast_slice(&oog),
        bytemuck::cast_slice_mut(&mut compressed),
        1,
    );
    assert!(
        compressed.iter().all(|&c| (0.0..=1.0).contains(&c)),
        "SoftCompress into AdobeRGB stayed in gamut: {compressed:?}"
    );

    // Deep BGR / BGRA generic formats (U16, F16, F32) round-trip through RGB / RGBA.
    let bgr16 = PixelFormat::from_parts(ChannelType::U16, ChannelLayout::Bgr, None).unwrap();
    assert_eq!(bgr16.name(), "Bgr16");
    let bgra16 = PixelFormat::from_parts(
        ChannelType::U16,
        ChannelLayout::Bgra,
        Some(AlphaMode::Straight),
    )
    .unwrap();
    assert_eq!(bgra16.name(), "Bgra16");
    let bgr16_srgb = bgr16.descriptor().with_transfer(TransferFunction::Srgb);
    let bgra16_srgb = bgra16
        .descriptor()
        .with_alpha(Some(AlphaMode::Straight))
        .with_transfer(TransferFunction::Srgb);
    let rgb16_px = [1000_u16, 2000, 3000];
    let mut bgr16_out = [0_u16; 3];
    let mut bgra16_out = [0_u16; 4];
    RowConverter::new(PixelDescriptor::RGB16_SRGB, bgr16_srgb)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&rgb16_px),
            bytemuck::cast_slice_mut(&mut bgr16_out),
            1,
        );
    assert_eq!(bgr16_out, [3000, 2000, 1000]);
    RowConverter::new(PixelDescriptor::RGB16_SRGB, bgra16_srgb)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&rgb16_px),
            bytemuck::cast_slice_mut(&mut bgra16_out),
            1,
        );
    assert_eq!(bgra16_out, [3000, 2000, 1000, 65535]);

    let bgrf32 = PixelFormat::from_parts(ChannelType::F32, ChannelLayout::Bgr, None).unwrap();
    assert_eq!(bgrf32.name(), "BgrF32");
    let bgrf32_lin = bgrf32.descriptor().with_transfer(TransferFunction::Linear);
    let rgbf32_px = [0.1_f32, 0.2, 0.3];
    let mut bgrf32_out = [0.0_f32; 3];
    RowConverter::new(PixelDescriptor::RGBF32_LINEAR, bgrf32_lin)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&rgbf32_px),
            bytemuck::cast_slice_mut(&mut bgrf32_out),
            1,
        );
    assert_eq!(bgrf32_out, [0.3, 0.2, 0.1]);
}
