//! Roundtrip tests for all supported format pairs.

use zenpixels::RowConverter;
use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

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
