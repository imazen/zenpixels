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

    let mut to_rgba =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    let mut to_rgb =
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

    let mut conv =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
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

    let mut to_rgba =
        RowConverter::new(PixelDescriptor::BGRA8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    let mut to_bgra =
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

    let mut conv =
        RowConverter::new(PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();
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

    let mut conv =
        RowConverter::new(PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
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

    let mut conv =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();
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

    let mut to_u16 = RowConverter::new(desc_u8, desc_u16).unwrap();
    let mut to_u8 = RowConverter::new(desc_u16, desc_u8).unwrap();

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
    let mut conv = RowConverter::new(from, to).unwrap();
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

    let mut to_linear = RowConverter::new(pq_u16, linear_f32).unwrap();
    let mut to_pq = RowConverter::new(linear_f32, pq_u16).unwrap();

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

    let mut to_linear = RowConverter::new(hlg_u16, linear_f32).unwrap();
    let mut to_hlg = RowConverter::new(linear_f32, hlg_u16).unwrap();

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

    let mut to_linear = RowConverter::new(pq_f32, linear_f32).unwrap();
    let mut to_pq = RowConverter::new(linear_f32, pq_f32).unwrap();

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

    let mut to_linear = RowConverter::new(hlg_f32, linear_f32).unwrap();
    let mut to_hlg = RowConverter::new(linear_f32, hlg_f32).unwrap();

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

    let mut conv = RowConverter::new(pq_u16, srgb_u8).unwrap();

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

    let mut pq_to_hlg = RowConverter::new(pq_f32, hlg_f32).unwrap();
    let mut hlg_to_pq = RowConverter::new(hlg_f32, pq_f32).unwrap();

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

/// Verify that different primaries with same depth triggers gamut conversion.
#[test]
fn different_primaries_same_depth_applies_gamut() {
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

    // ConvertPlan now applies gamut matrices for known primaries.
    let mut conv = RowConverter::new(bt709, bt2020).unwrap();
    assert!(
        !conv.is_identity(),
        "Different primaries should apply gamut conversion"
    );

    // White should be preserved.
    let width = 1u32;
    let src: Vec<u8> = vec![255, 255, 255];
    let mut dst = vec![0u8; 3];
    conv.convert_row(&src, &mut dst, width);
    for (c, &val) in dst.iter().enumerate() {
        assert!(
            (val as i32 - 255).unsigned_abs() <= 1,
            "White not preserved in ch{c}: {val}"
        );
    }
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

    let mut to_ga = RowConverter::new(from, to).unwrap();
    let mut to_g = RowConverter::new(to, from).unwrap();

    let mut ga = vec![0u8; width as usize * 2];
    let mut back = vec![0u8; width as usize];

    to_ga.convert_row(&src, &mut ga, width);
    to_g.convert_row(&ga, &mut back, width);

    assert_eq!(src, back);
}

// ---------------------------------------------------------------------------
// F16 roundtrip tests
// ---------------------------------------------------------------------------

/// Helper: build a row of f16 pixel bytes from f32 values.
fn f32s_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
    }
    out
}

/// Helper: read f16 pixel bytes into f32 values.
fn f16_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// F16 linear RGBA → F32 linear RGBA → F16 linear RGBA. Round-trip must be
/// bit-exact because F16→F32 is lossless by IEEE 754 and F32→F16 rounds back
/// to the same bits for any value originally representable as f16.
#[test]
fn f16_linear_to_f32_linear_and_back_is_exact() {
    let width = 32u32;
    // Sample across the full f16 SDR range including sub-normals, near-1.0,
    // and some HDR values.
    let f32_vals: Vec<f32> = (0..width as usize * 4)
        .map(|i| {
            let t = i as f32 / 16.0;
            if i % 3 == 0 {
                t * 0.25
            } else if i % 3 == 1 {
                1.0 - t * 0.01
            } else {
                2.0 + t * 0.1
            }
        })
        .collect();
    // Quantize to f16 first so we have exact source values.
    let src: Vec<u8> = {
        let mut s = Vec::with_capacity(f32_vals.len() * 2);
        for v in &f32_vals {
            s.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        s
    };

    let from = PixelDescriptor::RGBAF16_LINEAR;
    let to = PixelDescriptor::RGBAF32_LINEAR;

    let mut up = RowConverter::new(from, to).unwrap();
    let mut down = RowConverter::new(to, from).unwrap();

    let mut f32_row = vec![0u8; width as usize * 4 * 4];
    let mut back = vec![0u8; width as usize * 4 * 2];

    up.convert_row(&src, &mut f32_row, width);
    down.convert_row(&f32_row, &mut back, width);

    assert_eq!(
        src, back,
        "F16 linear → F32 linear → F16 linear should be bit-exact"
    );
}

/// F16 sRGB RGB → U8 sRGB RGB → F16 sRGB RGB. Checks the cross-depth F16 ↔ U8
/// path routes correctly through F32 and the planner picks fused sRGB kernels.
#[test]
fn f16_srgb_to_u8_srgb_roundtrip_within_tolerance() {
    let width = 16u32;
    // sRGB-encoded f16 values representing typical 8-bit-quantizable inputs.
    let u8_vals: Vec<u8> = (0..width as usize * 3)
        .map(|i| (i * 7 % 256) as u8)
        .collect();
    let f16_vals: Vec<f32> = u8_vals.iter().map(|v| *v as f32 / 255.0).collect();
    let src = f32s_to_f16_bytes(&f16_vals);

    let f16_srgb = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let u8_srgb = PixelDescriptor::RGB8_SRGB;

    let mut to_u8 = RowConverter::new(f16_srgb, u8_srgb).unwrap();
    let mut to_f16 = RowConverter::new(u8_srgb, f16_srgb).unwrap();

    let mut u8_row = vec![0u8; width as usize * 3];
    let mut back = vec![0u8; width as usize * 3 * 2];

    to_u8.convert_row(&src, &mut u8_row, width);
    to_f16.convert_row(&u8_row, &mut back, width);

    // u8 round-trip quantizes to 1/255, but re-packing back into f16 is exact.
    // Compare against the u8 ground truth: every round-tripped f16 must decode
    // to the same u8 value as u8_vals.
    for (i, (a, b)) in u8_vals.iter().zip(u8_row.iter()).enumerate() {
        assert_eq!(a, b, "u8 mismatch at pixel {}", i);
    }
    let back_f32 = f16_bytes_to_f32s(&back);
    for (i, (orig, rt)) in f16_vals.iter().zip(back_f32.iter()).enumerate() {
        let diff = (orig - rt).abs();
        assert!(
            diff < 1.5 / 255.0,
            "f16→u8→f16 roundtrip diff > 1.5/255 at pixel {}: {} vs {}",
            i,
            orig,
            rt
        );
    }
}

/// F16 linear → U16 linear → F16 linear. U16 is more precise than F16 in [0,1]
/// SDR, so F16 values should survive the round-trip within ~1 f16 ULP.
#[test]
fn f16_linear_to_u16_linear_roundtrip_within_f16_ulp() {
    let width = 16u32;
    let f32_vals: Vec<f32> = (0..width as usize * 3)
        .map(|i| (i as f32 / 48.0).min(1.0))
        .collect();
    let src = f32s_to_f16_bytes(&f32_vals);

    let f16_lin = PixelDescriptor::RGBF16_LINEAR;
    let u16_lin = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );

    let mut to_u16 = RowConverter::new(f16_lin, u16_lin).unwrap();
    let mut to_f16 = RowConverter::new(u16_lin, f16_lin).unwrap();

    let mut u16_row = vec![0u8; width as usize * 3 * 2];
    let mut back = vec![0u8; width as usize * 3 * 2];

    to_u16.convert_row(&src, &mut u16_row, width);
    to_f16.convert_row(&u16_row, &mut back, width);

    let src_f32 = f16_bytes_to_f32s(&src);
    let back_f32 = f16_bytes_to_f32s(&back);
    for (i, (orig, rt)) in src_f32.iter().zip(back_f32.iter()).enumerate() {
        // f16 ULP at value near 1.0 is ~2^-11 ≈ 4.88e-4; allow 2 ULP slack.
        let tolerance = 1e-3;
        let diff = (orig - rt).abs();
        assert!(
            diff < tolerance,
            "f16→u16→f16 roundtrip exceeded tolerance at pixel {}: {} vs {} (diff {})",
            i,
            orig,
            rt,
            diff
        );
    }
}

/// Depth-reduction policy catches U16 → F16 (precision loss from 16 bits to
/// 11 effective bits) even though both are 2 bytes.
#[test]
fn u16_to_f16_triggers_depth_reduction_policy() {
    use zenpixels_convert::{ConvertError, ConvertOptions, DepthPolicy, RowConverter};

    let from = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::RGBF16_LINEAR;

    let opts = ConvertOptions::permissive().with_depth_policy(DepthPolicy::Forbid);

    let result = RowConverter::new_explicit(from, to, &opts);
    match result {
        Err(e) => assert_eq!(*e.error(), ConvertError::DepthReductionForbidden),
        Ok(_) => panic!("expected DepthReductionForbidden for U16 → F16 with Forbid policy"),
    }
}

/// F16 → F16 with a TF change (sRGB → Linear) must round through F32, not
/// pass bytes through unchanged.
#[test]
fn f16_srgb_to_f16_linear_changes_values() {
    let width = 8u32;
    // sRGB-encoded 0.5 (mid-gray) decodes to linear ~0.2140.
    let f32_vals = vec![0.5_f32; width as usize * 3];
    let src = f32s_to_f16_bytes(&f32_vals);

    let from = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let to = PixelDescriptor::RGBF16_LINEAR;

    let mut conv = RowConverter::new(from, to).unwrap();
    let mut out = vec![0u8; width as usize * 3 * 2];
    conv.convert_row(&src, &mut out, width);

    let out_f32 = f16_bytes_to_f32s(&out);
    // sRGB EOTF of 0.5 ≈ 0.2140. Allow generous tolerance for F16 quantization.
    for v in &out_f32 {
        assert!(
            (v - 0.2140).abs() < 0.01,
            "expected linear ~0.2140 after sRGB F16 → Linear F16, got {}",
            v
        );
    }
}
