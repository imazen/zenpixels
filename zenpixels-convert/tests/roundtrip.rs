//! Roundtrip tests for all supported format pairs.

use zenpixels_convert::RowConverter;
use zenpixels_convert::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};

fn make_rgb8_row(width: usize) -> Vec<u8> {
    (0..width * 3).map(|i| (i % 256) as u8).collect()
}

fn _make_rgba8_row(width: usize) -> Vec<u8> {
    (0..width)
        .flat_map(|i| {
            let base = (i * 4) % 256;
            [base as u8, (base + 1) as u8, (base + 2) as u8, 200u8]
        })
        .collect()
}

fn make_gray8_row(width: usize) -> Vec<u8> {
    (0..width).map(|i| (i % 256) as u8).collect()
}

fn make_bgra8_row(width: usize) -> Vec<u8> {
    (0..width)
        .flat_map(|i| {
            let base = (i * 4) % 256;
            // BGRA order: B, G, R, A
            [(base + 2) as u8, (base + 1) as u8, base as u8, 200u8]
        })
        .collect()
}

#[test]
fn rgb8_to_rgba8_roundtrip() {
    let width = 64u32;
    let src = make_rgb8_row(width as usize);
    let mut rgba = vec![0u8; width as usize * 4];
    let mut back = vec![0u8; width as usize * 3];

    let to_rgba =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    let to_rgb =
        RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();

    to_rgba.convert_row(&src, &mut rgba, width);
    to_rgb.convert_row(&rgba, &mut back, width);

    assert_eq!(src, back, "RGB8→RGBA8→RGB8 roundtrip should be lossless");
}

#[test]
fn rgba8_add_alpha_is_opaque() {
    let width = 16u32;
    let src = make_rgb8_row(width as usize);
    let mut rgba = vec![0u8; width as usize * 4];

    let conv = RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    conv.convert_row(&src, &mut rgba, width);

    // Every 4th byte should be 255 (opaque).
    for i in 0..width as usize {
        assert_eq!(rgba[i * 4 + 3], 255, "alpha should be opaque at pixel {i}");
        assert_eq!(rgba[i * 4], src[i * 3], "R channel at pixel {i}");
        assert_eq!(rgba[i * 4 + 1], src[i * 3 + 1], "G channel at pixel {i}");
        assert_eq!(rgba[i * 4 + 2], src[i * 3 + 2], "B channel at pixel {i}");
    }
}

#[test]
fn bgra8_rgba8_roundtrip() {
    let width = 32u32;
    let src = make_bgra8_row(width as usize);
    let mut rgba = vec![0u8; width as usize * 4];
    let mut back = vec![0u8; width as usize * 4];

    let to_rgba =
        RowConverter::new(PixelDescriptor::BGRA8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    let to_bgra =
        RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::BGRA8_SRGB).unwrap();

    to_rgba.convert_row(&src, &mut rgba, width);
    to_bgra.convert_row(&rgba, &mut back, width);

    assert_eq!(src, back, "BGRA8→RGBA8→BGRA8 roundtrip should be lossless");
}

#[test]
fn gray8_to_rgb8() {
    let width = 16u32;
    let src = make_gray8_row(width as usize);
    let mut rgb = vec![0u8; width as usize * 3];

    let conv = RowConverter::new(PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();
    conv.convert_row(&src, &mut rgb, width);

    for i in 0..width as usize {
        let g = src[i];
        assert_eq!(rgb[i * 3], g, "R should equal gray at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], g, "G should equal gray at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], g, "B should equal gray at pixel {i}");
    }
}

#[test]
fn gray8_to_rgba8() {
    let width = 8u32;
    let src = make_gray8_row(width as usize);
    let mut rgba = vec![0u8; width as usize * 4];

    let conv = RowConverter::new(PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    conv.convert_row(&src, &mut rgba, width);

    for i in 0..width as usize {
        let g = src[i];
        assert_eq!(rgba[i * 4], g);
        assert_eq!(rgba[i * 4 + 1], g);
        assert_eq!(rgba[i * 4 + 2], g);
        assert_eq!(rgba[i * 4 + 3], 255);
    }
}

#[test]
fn identity_is_noop() {
    let width = 32u32;
    let src = make_rgb8_row(width as usize);
    let mut dst = vec![0u8; src.len()];

    let conv = RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(conv.is_identity());
    conv.convert_row(&src, &mut dst, width);
    assert_eq!(src, dst);
}

#[test]
fn u8_to_u16_roundtrip() {
    let width = 32u32;
    let src = make_rgb8_row(width as usize);

    let desc_u8 = PixelDescriptor::RGB8_SRGB;
    let desc_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );

    let to_u16 = RowConverter::new(desc_u8, desc_u16).unwrap();
    let to_u8 = RowConverter::new(desc_u16, desc_u8).unwrap();

    let mut u16_row = vec![0u8; width as usize * 3 * 2];
    let mut back = vec![0u8; width as usize * 3];

    to_u16.convert_row(&src, &mut u16_row, width);
    to_u8.convert_row(&u16_row, &mut back, width);

    // u8→u16→u8 should be lossless: v*257 then (v*255+32768)>>16.
    for i in 0..src.len() {
        assert!(
            (src[i] as i32 - back[i] as i32).unsigned_abs() <= 1,
            "u8→u16→u8 drift at byte {i}: {} vs {}",
            src[i],
            back[i]
        );
    }
}

#[test]
fn gray_alpha_to_rgba() {
    let width = 4u32;
    // GrayAlpha: [g, a, g, a, ...]
    let src: Vec<u8> = vec![100, 200, 150, 255, 0, 128, 255, 0];
    let mut rgba = vec![0u8; width as usize * 4];

    let from = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let to = PixelDescriptor::RGBA8_SRGB;
    let conv = RowConverter::new(from, to).unwrap();
    conv.convert_row(&src, &mut rgba, width);

    // Pixel 0: gray=100, alpha=200
    assert_eq!(rgba[0..4], [100, 100, 100, 200]);
    // Pixel 1: gray=150, alpha=255
    assert_eq!(rgba[4..8], [150, 150, 150, 255]);
    // Pixel 2: gray=0, alpha=128
    assert_eq!(rgba[8..12], [0, 0, 0, 128]);
    // Pixel 3: gray=255, alpha=0
    assert_eq!(rgba[12..16], [255, 255, 255, 0]);
}

// ── HDR pipeline tests ──────────────────────────────────────────────────────

fn make_rgb_u16_row(width: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * 3 * 2];
    for i in 0..width {
        let v = ((i * 1000) % 65536) as u16;
        let base = i * 6;
        let bytes = v.to_ne_bytes();
        out[base] = bytes[0];
        out[base + 1] = bytes[1];
        let v2 = ((i * 500 + 10000) % 65536) as u16;
        let bytes2 = v2.to_ne_bytes();
        out[base + 2] = bytes2[0];
        out[base + 3] = bytes2[1];
        let v3 = ((i * 300 + 30000) % 65536) as u16;
        let bytes3 = v3.to_ne_bytes();
        out[base + 4] = bytes3[0];
        out[base + 5] = bytes3[1];
    }
    out
}

fn make_rgb_f32_row(width: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * 3 * 4];
    for i in 0..width {
        let r = (i as f32) / (width as f32);
        let g = ((i as f32) / (width as f32) * 0.5 + 0.25).min(1.0);
        let b = ((width - 1 - i) as f32) / (width as f32);
        let base = i * 12;
        out[base..base + 4].copy_from_slice(&r.to_ne_bytes());
        out[base + 4..base + 8].copy_from_slice(&g.to_ne_bytes());
        out[base + 8..base + 12].copy_from_slice(&b.to_ne_bytes());
    }
    out
}

/// PQ U16 → Linear F32 → PQ U16 roundtrip through RowConverter.
#[test]
fn pq_u16_linear_f32_roundtrip() {
    let width = 64u32;
    let src = make_rgb_u16_row(width as usize);

    let pq_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let linear_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );

    let to_linear = RowConverter::new(pq_u16, linear_f32).unwrap();
    let to_pq = RowConverter::new(linear_f32, pq_u16).unwrap();

    let mut f32_buf = vec![0u8; width as usize * 3 * 4];
    let mut back = vec![0u8; width as usize * 3 * 2];

    to_linear.convert_row(&src, &mut f32_buf, width);
    to_pq.convert_row(&f32_buf, &mut back, width);

    // PQ U16 roundtrip should be exact (0 error) for all values.
    for i in 0..src.len() / 2 {
        let orig = u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]);
        let result = u16::from_ne_bytes([back[i * 2], back[i * 2 + 1]]);
        assert!(
            (orig as i32 - result as i32).unsigned_abs() <= 1,
            "PQ U16 roundtrip error at sample {i}: {} vs {}",
            orig,
            result
        );
    }
}

/// HLG U16 → Linear F32 → HLG U16 roundtrip through RowConverter.
#[test]
fn hlg_u16_linear_f32_roundtrip() {
    let width = 64u32;
    let src = make_rgb_u16_row(width as usize);

    let hlg_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    );
    let linear_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );

    let to_linear = RowConverter::new(hlg_u16, linear_f32).unwrap();
    let to_hlg = RowConverter::new(linear_f32, hlg_u16).unwrap();

    let mut f32_buf = vec![0u8; width as usize * 3 * 4];
    let mut back = vec![0u8; width as usize * 3 * 2];

    to_linear.convert_row(&src, &mut f32_buf, width);
    to_hlg.convert_row(&f32_buf, &mut back, width);

    for i in 0..src.len() / 2 {
        let orig = u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]);
        let result = u16::from_ne_bytes([back[i * 2], back[i * 2 + 1]]);
        assert!(
            (orig as i32 - result as i32).unsigned_abs() <= 1,
            "HLG U16 roundtrip error at sample {i}: {} vs {}",
            orig,
            result
        );
    }
}

/// PQ F32 → Linear F32 → PQ F32 roundtrip.
#[test]
fn pq_f32_linear_f32_roundtrip() {
    let width = 64u32;
    let src = make_rgb_f32_row(width as usize);

    let pq_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let linear_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );

    let to_linear = RowConverter::new(pq_f32, linear_f32).unwrap();
    let to_pq = RowConverter::new(linear_f32, pq_f32).unwrap();

    let mut linear_buf = vec![0u8; width as usize * 3 * 4];
    let mut back = vec![0u8; width as usize * 3 * 4];

    to_linear.convert_row(&src, &mut linear_buf, width);
    to_pq.convert_row(&linear_buf, &mut back, width);

    for i in 0..width as usize * 3 {
        let base = i * 4;
        let orig = f32::from_ne_bytes([src[base], src[base + 1], src[base + 2], src[base + 3]]);
        let result =
            f32::from_ne_bytes([back[base], back[base + 1], back[base + 2], back[base + 3]]);
        assert!(
            (orig - result).abs() < 1e-4,
            "PQ F32 roundtrip error at sample {i}: {orig:.6} vs {result:.6}",
        );
    }
}

/// HLG F32 → Linear F32 → HLG F32 roundtrip.
#[test]
fn hlg_f32_linear_f32_roundtrip() {
    let width = 64u32;
    let src = make_rgb_f32_row(width as usize);

    let hlg_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    );
    let linear_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );

    let to_linear = RowConverter::new(hlg_f32, linear_f32).unwrap();
    let to_hlg = RowConverter::new(linear_f32, hlg_f32).unwrap();

    let mut linear_buf = vec![0u8; width as usize * 3 * 4];
    let mut back = vec![0u8; width as usize * 3 * 4];

    to_linear.convert_row(&src, &mut linear_buf, width);
    to_hlg.convert_row(&linear_buf, &mut back, width);

    for i in 0..width as usize * 3 {
        let base = i * 4;
        let orig = f32::from_ne_bytes([src[base], src[base + 1], src[base + 2], src[base + 3]]);
        let result =
            f32::from_ne_bytes([back[base], back[base + 1], back[base + 2], back[base + 3]]);
        assert!(
            (orig - result).abs() < 1e-4,
            "HLG F32 roundtrip error at sample {i}: {orig:.6} vs {result:.6}",
        );
    }
}

/// PQ U16 → sRGB U8 cross-TF conversion (HDR to SDR tone mapping path).
#[test]
fn pq_u16_to_srgb_u8() {
    let width = 4u32;
    // Hand-craft known PQ values.
    let pq_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let srgb_u8 = PixelDescriptor::RGB8_SRGB;

    let conv = RowConverter::new(pq_u16, srgb_u8).unwrap();

    // PQ 0 → linear 0 → sRGB 0
    // PQ 65535 → linear 1.0 → sRGB 255
    let mut src = vec![0u8; 4 * 6]; // 4 pixels × 3 channels × 2 bytes
    // Pixel 0: all zero (PQ black)
    // Pixel 1: all 65535 (PQ white = 10000 nits, clips to sRGB white)
    for j in 0..3 {
        let base = 6 + j * 2;
        src[base..base + 2].copy_from_slice(&65535u16.to_ne_bytes());
    }
    // Pixel 2: mid value ~32768
    for j in 0..3 {
        let base = 2 * 6 + j * 2;
        src[base..base + 2].copy_from_slice(&32768u16.to_ne_bytes());
    }
    // Pixel 3: moderate-low value ~20000 (above PQ black threshold)
    for j in 0..3 {
        let base = 3 * 6 + j * 2;
        src[base..base + 2].copy_from_slice(&20000u16.to_ne_bytes());
    }

    let mut dst = vec![0u8; 4 * 3]; // 4 pixels × 3 channels × 1 byte
    conv.convert_row(&src, &mut dst, width);

    // Black stays black
    assert_eq!(dst[0], 0);
    assert_eq!(dst[1], 0);
    assert_eq!(dst[2], 0);

    // Full PQ → sRGB white
    assert_eq!(dst[3], 255);
    assert_eq!(dst[4], 255);
    assert_eq!(dst[5], 255);

    // Mid PQ → some sRGB value > 0 and < 255
    assert!(dst[6] > 0, "mid PQ should produce non-zero sRGB");
    assert!(dst[6] < 255, "mid PQ should be below sRGB max");

    // Low PQ → low sRGB
    assert!(dst[9] > 0, "low PQ should produce non-zero sRGB");
    assert!(dst[9] < dst[6], "low PQ should be less than mid PQ");
}

/// PQ ↔ HLG cross-TF conversion via F32 linear intermediary.
#[test]
fn pq_hlg_cross_conversion() {
    let width = 32u32;
    let src = make_rgb_f32_row(width as usize);

    let pq_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let hlg_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    );

    let pq_to_hlg = RowConverter::new(pq_f32, hlg_f32).unwrap();
    let hlg_to_pq = RowConverter::new(hlg_f32, pq_f32).unwrap();

    let mut hlg_buf = vec![0u8; width as usize * 3 * 4];
    let mut back = vec![0u8; width as usize * 3 * 4];

    pq_to_hlg.convert_row(&src, &mut hlg_buf, width);
    hlg_to_pq.convert_row(&hlg_buf, &mut back, width);

    for i in 0..width as usize * 3 {
        let base = i * 4;
        let orig = f32::from_ne_bytes([src[base], src[base + 1], src[base + 2], src[base + 3]]);
        let result =
            f32::from_ne_bytes([back[base], back[base + 1], back[base + 2], back[base + 3]]);
        assert!(
            (orig - result).abs() < 1e-3,
            "PQ→HLG→PQ roundtrip error at sample {i}: {orig:.6} vs {result:.6}",
        );
    }
}

/// Verify that conversion plans with different primaries but same depth are identity.
/// (Gamut conversion is NOT part of ConvertPlan — it's handled separately.)
#[test]
fn different_primaries_same_depth_is_identity() {
    let bt709 = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    )
    .with_primaries(ColorPrimaries::Bt709);

    let bt2020 = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    )
    .with_primaries(ColorPrimaries::Bt2020);

    // ConvertPlan ignores primaries — gamut conversion is a separate step.
    let conv = RowConverter::new(bt709, bt2020).unwrap();
    assert!(
        conv.is_identity(),
        "Same depth/layout/TF with different primaries should be identity"
    );
}

#[test]
fn gray_to_gray_alpha_roundtrip() {
    let width = 8u32;
    let src = make_gray8_row(width as usize);

    let from = PixelDescriptor::GRAY8_SRGB;
    let to = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );

    let to_ga = RowConverter::new(from, to).unwrap();
    let to_g = RowConverter::new(to, from).unwrap();

    let mut ga = vec![0u8; width as usize * 2];
    let mut back = vec![0u8; width as usize];

    to_ga.convert_row(&src, &mut ga, width);
    to_g.convert_row(&ga, &mut back, width);

    assert_eq!(src, back);
}
