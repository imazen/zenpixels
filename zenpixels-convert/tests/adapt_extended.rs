#![allow(deprecated)]
//! Extended tests for adapt module — strided buffers, policy enforcement,
//! and edge cases not covered by the basic adapt tests.

/// The packed rows of `data` converted from `from` to `to` through the
/// owned-buffer path (formerly checked through the raw `convert_buffer`).
fn convert_packed(
    data: &[u8],
    width: u32,
    rows: u32,
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<Vec<u8>, whereat::At<zenpixels_convert::ConvertError>> {
    use zenpixels_convert::PixelBufferConvertExt;
    let source = zenpixels::PixelBuffer::from_vec(data.to_vec(), width, rows, from).unwrap();
    let converted = source.convert_to(to)?;
    Ok(converted.as_slice().contiguous_bytes().into_owned())
}
#[path = "common/adapt_rows.rs"]
mod adapt_rows;
use adapt_rows::adapt_rows;
use zenpixels_convert::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
use zenpixels_convert::{ConvertError, PixelDescriptor};

extern crate alloc;

// ---------------------------------------------------------------------------
// Strided buffer handling
// ---------------------------------------------------------------------------

#[test]
fn strided_buffer_packed_is_zero_copy() {
    // Packed buffer (stride = row bytes).
    let data = vec![100, 150, 200, 50, 100, 150];
    let desc = PixelDescriptor::RGB8_SRGB;

    let result = adapt_rows(
        &data,
        desc,
        2,
        1,
        6,
        &[desc],
        &zenpixels_convert::ConvertOptions::permissive()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardUnchecked),
    )
    .unwrap();

    assert!(
        matches!(result, zenpixels::PixelCow::Borrowed(_)),
        "packed exact match should be zero-copy"
    );
}

// ---------------------------------------------------------------------------
// Policy enforcement
// ---------------------------------------------------------------------------

#[test]
fn explicit_depth_forbid_returns_error() {
    // Source is RGBA16, target is RGBA8 — depth reduction.
    let values: [u16; 4] = [25700, 38550, 51400, 65535];
    let mut data = vec![0u8; 8];
    for (i, &v) in values.iter().enumerate() {
        let bytes = v.to_ne_bytes();
        data[i * 2] = bytes[0];
        data[i * 2 + 1] = bytes[1];
    }

    let options =
        ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardUnchecked);

    let result = adapt_rows(
        &data,
        PixelDescriptor::RGBA16_SRGB,
        1,
        1,
        8,
        &[PixelDescriptor::RGBA8_SRGB],
        &options,
    );

    assert!(result.is_err());
    let err = result.unwrap_err().decompose().0;
    assert_eq!(err, ConvertError::DepthReductionForbidden);
}

#[test]
fn explicit_alpha_forbid_returns_error() {
    // Source is RGBA8, target is RGB8 — alpha removal.
    let data = vec![100, 150, 200, 255];

    let options = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);

    let result = adapt_rows(
        &data,
        PixelDescriptor::RGBA8_SRGB,
        1,
        1,
        4,
        &[PixelDescriptor::RGB8_SRGB],
        &options,
    );

    assert!(result.is_err());
    let err = result.unwrap_err().decompose().0;
    assert_eq!(err, ConvertError::AlphaRemovalForbidden);
}

#[test]
fn explicit_discard_if_opaque_succeeds_when_opaque() {
    // Fully opaque RGBA8 → RGB8 with DiscardIfOpaque should succeed.
    let data = vec![100, 150, 200, 255, 50, 100, 150, 255];

    let options = ConvertOptions::permissive().with_luma(None);

    let result = adapt_rows(
        &data,
        PixelDescriptor::RGBA8_SRGB,
        2,
        1,
        8,
        &[PixelDescriptor::RGB8_SRGB],
        &options,
    );

    assert!(result.is_ok());
    let adapted = result.unwrap();
    assert_eq!(adapted.as_slice().descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!(
        &adapted.as_slice().contiguous_bytes()[..3],
        &[100, 150, 200]
    );
}

#[test]
fn explicit_discard_if_opaque_fails_when_semitransparent() {
    // Semi-transparent RGBA8 → RGB8 with DiscardIfOpaque should fail.
    let data = vec![100, 150, 200, 128]; // alpha = 128

    let options = ConvertOptions::permissive().with_luma(None);

    let result = adapt_rows(
        &data,
        PixelDescriptor::RGBA8_SRGB,
        1,
        1,
        4,
        &[PixelDescriptor::RGB8_SRGB],
        &options,
    );

    assert!(result.is_err());
    let err = result.unwrap_err().decompose().0;
    assert_eq!(err, ConvertError::AlphaNotOpaque);
}

// ---------------------------------------------------------------------------
// Packed conversion edge cases
// ---------------------------------------------------------------------------

#[test]
fn convert_buffer_identity_returns_copy() {
    let data = vec![100, 150, 200, 50, 100, 150];
    let result = convert_packed(
        &data,
        2,
        1,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    assert_eq!(result, data);
}

#[test]
fn convert_buffer_multi_row() {
    // 2×3 RGB8 → RGBA8
    let data = vec![
        10, 20, 30, 40, 50, 60, // row 0
        70, 80, 90, 100, 110, 120, // row 1
        130, 140, 150, 160, 170, 180, // row 2
    ];

    let result = convert_packed(
        &data,
        2,
        3,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
    )
    .unwrap();

    assert_eq!(result.len(), 2 * 3 * 4);
    // First pixel: R=10, G=20, B=30, A=255
    assert_eq!(&result[..4], &[10, 20, 30, 255]);
    // Last pixel: R=160, G=170, B=180, A=255
    assert_eq!(&result[20..24], &[160, 170, 180, 255]);
}

#[test]
fn convert_buffer_bgra_to_rgba() {
    // BGRA → RGBA swizzle
    let data = vec![200, 150, 100, 255]; // B=200, G=150, R=100, A=255
    let result = convert_packed(
        &data,
        1,
        1,
        PixelDescriptor::BGRA8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
    )
    .unwrap();
    assert_eq!(result, vec![100, 150, 200, 255]); // R=100, G=150, B=200, A=255
}

// ---------------------------------------------------------------------------
// Empty format list
// ---------------------------------------------------------------------------

#[test]
fn adapt_empty_supported_returns_error() {
    let data = vec![100, 150, 200];

    let result = adapt_rows(
        &data,
        PixelDescriptor::RGB8_SRGB,
        1,
        1,
        3,
        &[],
        &zenpixels_convert::ConvertOptions::permissive()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardUnchecked),
    );

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().decompose().0,
        ConvertError::EmptyFormatList
    );
}

#[test]
fn adapt_explicit_empty_supported_returns_error() {
    let data = vec![100, 150, 200];
    let options = ConvertOptions::permissive();

    let result = adapt_rows(&data, PixelDescriptor::RGB8_SRGB, 1, 1, 3, &[], &options);

    assert!(result.is_err());
}
