//! Tests for the adapt module (codec helper functions).

use zencodec_types::{PixelDescriptor, PixelSlice};
use zenpixels::adapt::{adapt_for_encode, convert_buffer};

/// When input matches a supported format, should return borrowed data.
#[test]
fn adapt_exact_match_is_borrowed() {
    let width = 4u32;
    let rows = 2u32;
    let bpp = 3; // RGB8
    let stride = width as usize * bpp;
    let data = vec![128u8; stride * rows as usize];
    let desc = PixelDescriptor::RGB8_SRGB;

    let pixels = PixelSlice::new(&data, width, rows, stride, desc).unwrap();
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];

    let result = adapt_for_encode(&pixels, supported).unwrap();
    assert_eq!(result.descriptor, PixelDescriptor::RGB8_SRGB);
    assert!(
        matches!(result.data, std::borrow::Cow::Borrowed(_)),
        "exact match should be zero-copy"
    );
}

/// When input doesn't match, should convert and return owned data.
#[test]
fn adapt_converts_when_needed() {
    let width = 4u32;
    let rows = 1u32;
    let bpp = 4; // BGRA8
    let stride = width as usize * bpp;
    // BGRA pixels: B=10, G=20, R=30, A=255
    let mut data = Vec::new();
    for _ in 0..width {
        data.extend_from_slice(&[10, 20, 30, 255]);
    }
    let desc = PixelDescriptor::BGRA8_SRGB;

    let pixels = PixelSlice::new(&data, width, rows, stride, desc).unwrap();
    // Only supports RGBA8.
    let supported = &[PixelDescriptor::RGBA8_SRGB];

    let result = adapt_for_encode(&pixels, supported).unwrap();
    assert_eq!(result.descriptor, PixelDescriptor::RGBA8_SRGB);
    assert!(
        matches!(result.data, std::borrow::Cow::Owned(_)),
        "conversion should produce owned data"
    );

    // Verify swizzle: BGRA(10,20,30,255) → RGBA(30,20,10,255).
    assert_eq!(result.data[0], 30, "R");
    assert_eq!(result.data[1], 20, "G");
    assert_eq!(result.data[2], 10, "B");
    assert_eq!(result.data[3], 255, "A");
}

/// Empty format list should return error.
#[test]
fn adapt_empty_list_errors() {
    let width = 1u32;
    let rows = 1u32;
    let data = vec![0u8; 3];
    let desc = PixelDescriptor::RGB8_SRGB;
    let pixels = PixelSlice::new(&data, width, rows, 3, desc).unwrap();

    let result = adapt_for_encode(&pixels, &[]);
    assert!(result.is_err());
}

/// Transfer-agnostic match: source has Unknown transfer, target has Srgb,
/// but otherwise identical — should be zero-copy.
#[test]
fn adapt_transfer_agnostic_match() {
    let width = 2u32;
    let rows = 1u32;
    let data = vec![100u8; 6]; // 2 RGB pixels
    let desc = PixelDescriptor::RGB8; // Unknown transfer

    let pixels = PixelSlice::new(&data, width, rows, 6, desc).unwrap();
    let supported = &[PixelDescriptor::RGB8_SRGB];

    let result = adapt_for_encode(&pixels, supported).unwrap();
    assert_eq!(result.descriptor, PixelDescriptor::RGB8_SRGB);
    assert!(
        matches!(result.data, std::borrow::Cow::Borrowed(_)),
        "transfer-only diff should be zero-copy"
    );
}

/// convert_buffer with identity should just copy.
#[test]
fn convert_buffer_identity() {
    let src = vec![1u8, 2, 3, 4, 5, 6];
    let result = convert_buffer(
        &src,
        2,
        1,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    assert_eq!(result, src);
}

/// convert_buffer RGB8→RGBA8.
#[test]
fn convert_buffer_rgb_to_rgba() {
    let src = vec![10, 20, 30, 40, 50, 60];
    let result = convert_buffer(
        &src,
        2,
        1,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
    )
    .unwrap();

    assert_eq!(result.len(), 8);
    assert_eq!(result[0..4], [10, 20, 30, 255]);
    assert_eq!(result[4..8], [40, 50, 60, 255]);
}
