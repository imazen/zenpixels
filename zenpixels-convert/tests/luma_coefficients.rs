//! Regression: `RgbToGray` / `RgbaToGray` used to hardcode BT.709 coefficients
//! and only handled U8. `ConvertOptions::luma` was silently ignored, and
//! U16/F32/F16 RGB→Gray plans produced garbage by running the U8 kernel on
//! cast bytes. Both fixed now.

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
    policy::{ConvertOptions, LumaCoefficients},
};
use zenpixels_convert::RowConverter;

fn rgb_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgb, None, tf)
}
fn gray_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Gray, None, tf)
}
fn rgb_u16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U16, ChannelLayout::Rgb, None, tf)
}
fn gray_u16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U16, ChannelLayout::Gray, None, tf)
}
fn rgb_f32(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, tf)
}
fn gray_f32(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::F32, ChannelLayout::Gray, None, tf)
}
fn rgba_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        tf,
    )
}

fn opts_with_luma(coefficients: LumaCoefficients) -> ConvertOptions {
    ConvertOptions::permissive().with_luma(Some(coefficients))
}

fn convert_u8(
    src_desc: PixelDescriptor,
    dst_desc: PixelDescriptor,
    opts: ConvertOptions,
    src: &[u8],
    dst_len: usize,
) -> Vec<u8> {
    let mut conv = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let mut dst = vec![0u8; dst_len];
    let width = (src.len() / src_desc.layout().channels()) as u32;
    conv.convert_row(src, &mut dst, width);
    dst
}

/// User-specified BT.601 must produce different output than the default
/// BT.709. Pure-red pixel exposes the difference clearly: BT.709 R=0.2126,
/// BT.601 R=0.299 → ~0.087 normalized difference (~22 LSB at u8).
#[test]
fn u8_bt601_gives_different_output_than_bt709() {
    let pixel = [255u8, 0, 0]; // pure red
    let dst_709 = convert_u8(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt709),
        &pixel,
        1,
    );
    let dst_601 = convert_u8(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt601),
        &pixel,
        1,
    );
    // BT.709 red weight = 0.2126 → 54.2 → 54
    // BT.601 red weight = 0.299  → 76.2 → 76
    assert_eq!(dst_709[0], 54, "BT.709 red expected 54, got {}", dst_709[0]);
    assert_eq!(dst_601[0], 76, "BT.601 red expected 76, got {}", dst_601[0]);
    assert_ne!(
        dst_709[0], dst_601[0],
        "BT.601 must differ from BT.709 — pre-fix the kernel ignored options.luma"
    );
}

/// BT.2020 weight for green is 0.6780 vs BT.709's 0.7152 — pure green
/// exposes the change.
#[test]
fn u8_bt2020_distinct_from_bt709() {
    let pixel = [0u8, 255, 0]; // pure green
    let dst_709 = convert_u8(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt709),
        &pixel,
        1,
    );
    let dst_2020 = convert_u8(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt2020),
        &pixel,
        1,
    );
    // BT.709  G = 0.7152 * 255 → 182.4 → 182
    // BT.2020 G = 0.6780 * 255 → 172.9 → 173
    assert_eq!(dst_709[0], 182);
    assert_eq!(dst_2020[0], 173);
}

/// DisplayP3 weight for red is 0.2289746, between BT.709 (0.2126) and BT.601
/// (0.299). Verify it actually fires.
#[test]
fn u8_displayp3_distinct_from_others() {
    let pixel = [255u8, 0, 0];
    let dst_p3 = convert_u8(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::DisplayP3),
        &pixel,
        1,
    );
    // 0.2289746 * 255 = 58.388 → 58
    assert_eq!(
        dst_p3[0], 58,
        "DisplayP3 red expected 58, got {}",
        dst_p3[0]
    );
}

/// RGBA→Gray honors coefficients too.
#[test]
fn u8_rgba_to_gray_honors_coefficients() {
    let pixel = [255u8, 0, 0, 128];
    let dst_709 = convert_u8(
        rgba_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt709),
        &pixel,
        1,
    );
    let dst_601 = convert_u8(
        rgba_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        opts_with_luma(LumaCoefficients::Bt601),
        &pixel,
        1,
    );
    // Alpha is dropped (RGBA→Gray drop alpha after weighting).
    assert_eq!(dst_709[0], 54);
    assert_eq!(dst_601[0], 76);
}

/// U16 RGB→Gray was producing garbage pre-fix (U8 kernel ran on cast
/// bytes — read 1/2 the data per pixel). This test pins correct behavior.
#[test]
fn u16_rgb_to_gray_works_and_honors_coefficients() {
    let pixel: [u16; 3] = [65535, 0, 0]; // pure red, full intensity
    let src_bytes: [u8; 6] = bytemuck::cast(pixel);
    let mut conv = RowConverter::new_explicit(
        rgb_u16(TransferFunction::Srgb),
        gray_u16(TransferFunction::Srgb),
        &opts_with_luma(LumaCoefficients::Bt709),
    )
    .unwrap();
    let mut dst_bytes = [0u8; 2];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let dst: [u16; 1] = bytemuck::cast(dst_bytes);
    // BT.709 red on full-scale u16: 0.2126 * 65535 = 13932.7 → 13933
    assert!(
        dst[0].abs_diff(13933) <= 2,
        "u16 BT.709 red expected ~13933, got {}",
        dst[0]
    );

    let mut conv601 = RowConverter::new_explicit(
        rgb_u16(TransferFunction::Srgb),
        gray_u16(TransferFunction::Srgb),
        &opts_with_luma(LumaCoefficients::Bt601),
    )
    .unwrap();
    let mut dst_bytes = [0u8; 2];
    conv601.convert_row(&src_bytes, &mut dst_bytes, 1);
    let dst601: [u16; 1] = bytemuck::cast(dst_bytes);
    // BT.601 red: 0.299 * 65535 = 19595.0 → 19595
    assert!(
        dst601[0].abs_diff(19595) <= 2,
        "u16 BT.601 red expected ~19595, got {}",
        dst601[0]
    );
    assert_ne!(dst[0], dst601[0]);
}

/// F32 RGB→Gray works on f32 data, applies coefficients to encoded values.
#[test]
fn f32_rgb_to_gray_honors_coefficients() {
    let pixel: [f32; 3] = [1.0, 0.0, 0.0];
    let src_bytes: [u8; 12] = bytemuck::cast(pixel);
    let mut conv = RowConverter::new_explicit(
        rgb_f32(TransferFunction::Srgb),
        gray_f32(TransferFunction::Srgb),
        &opts_with_luma(LumaCoefficients::Bt709),
    )
    .unwrap();
    let mut dst_bytes = [0u8; 4];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let dst: [f32; 1] = bytemuck::cast(dst_bytes);
    assert!(
        (dst[0] - 0.2126).abs() < 1e-5,
        "f32 BT.709 red expected 0.2126, got {}",
        dst[0]
    );

    let mut conv601 = RowConverter::new_explicit(
        rgb_f32(TransferFunction::Srgb),
        gray_f32(TransferFunction::Srgb),
        &opts_with_luma(LumaCoefficients::Bt601),
    )
    .unwrap();
    let mut dst_bytes = [0u8; 4];
    conv601.convert_row(&src_bytes, &mut dst_bytes, 1);
    let dst601: [f32; 1] = bytemuck::cast(dst_bytes);
    assert!((dst601[0] - 0.299).abs() < 1e-5);
}

/// Plan-shape regression: the chosen luma propagates into the variant.
#[test]
fn plan_records_resolved_coefficients() {
    use zenpixels_convert::ConvertPlan;

    let plan = ConvertPlan::new_explicit(
        rgb_u8(TransferFunction::Srgb),
        gray_u8(TransferFunction::Srgb),
        &opts_with_luma(LumaCoefficients::DisplayP3),
    )
    .unwrap();
    let debug = format!("{:?}", plan);
    assert!(
        debug.contains("coefficients: DisplayP3"),
        "plan should record resolved DisplayP3 coefficients: {debug}"
    );
    assert!(
        !debug.contains("coefficients: Bt709"),
        "plan should NOT carry the Bt709 placeholder: {debug}"
    );
}

/// Round-trip identity is preserved (gray → RGB → gray = identity), the
/// invariant tested by ulp_exhaustive.rs:561-564. For BT.709 this only
/// works because the weights sum to 1.0 in fixed-point math — verify
/// with multiple coefficient choices that the round-trip still works
/// for gray inputs since R=G=B and weights sum to ~1.0.
#[test]
fn gray_roundtrip_under_all_coefficient_choices() {
    for &coeffs in &[
        LumaCoefficients::Bt709,
        LumaCoefficients::Bt601,
        LumaCoefficients::Bt2020,
        LumaCoefficients::DisplayP3,
    ] {
        for v in [0u8, 64, 128, 200, 255] {
            // Gray → RGB
            let mut conv_expand = RowConverter::new_explicit(
                gray_u8(TransferFunction::Srgb),
                rgb_u8(TransferFunction::Srgb),
                &ConvertOptions::permissive(),
            )
            .unwrap();
            let mut rgb = [0u8; 3];
            conv_expand.convert_row(&[v], &mut rgb, 1);

            // RGB → Gray with the chosen coefficients
            let mut conv_collapse = RowConverter::new_explicit(
                rgb_u8(TransferFunction::Srgb),
                gray_u8(TransferFunction::Srgb),
                &opts_with_luma(coeffs),
            )
            .unwrap();
            let mut gray = [0u8; 1];
            conv_collapse.convert_row(&rgb, &mut gray, 1);

            // Sum-of-weights ≈ 1.0 in f32 → ±1 LSB tolerance.
            assert!(
                gray[0].abs_diff(v) <= 1,
                "{coeffs:?} round-trip {v} → {} (diff {}) — weights may not sum to 1",
                gray[0],
                gray[0].abs_diff(v)
            );
        }
    }
}
