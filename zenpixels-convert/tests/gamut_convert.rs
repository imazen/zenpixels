//! Tests for gamut (color primaries) conversion and color context preservation.
//!
//! Key architectural note: RowConverter does NOT apply gamut matrices.
//! Primaries conversion is handled separately via `gamut::conversion_matrix()`
//! + `apply_matrix_row_f32()`. This tests both paths.

use zenpixels_convert::{
    ColorPrimaries, PixelDescriptor, RowConverter,
    ext::ColorPrimariesExt,
    gamut::{apply_matrix_f32, apply_matrix_row_f32, apply_matrix_row_rgba_f32, conversion_matrix},
};

/// Make an f32 RGB descriptor with specific primaries.
fn f32_linear(primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::RGBF32_LINEAR.with_primaries(primaries)
}

// ---------------------------------------------------------------------------
// RowConverter correctly treats different primaries as identity
// ---------------------------------------------------------------------------

#[test]
fn row_converter_ignores_primaries_difference() {
    // RowConverter intentionally does NOT apply gamut matrices.
    // Gamut conversion is a separate step (conversion_matrix + apply_matrix_row).
    let src = f32_linear(ColorPrimaries::Bt709);
    let dst = f32_linear(ColorPrimaries::Bt2020);

    let conv = RowConverter::new(src, dst).unwrap();
    assert!(conv.is_identity(), "RowConverter should be identity when only primaries differ");
}

// ---------------------------------------------------------------------------
// Manual gamut conversion pipeline (the correct way)
// ---------------------------------------------------------------------------

#[test]
fn manual_gamut_pipeline_bt709_to_bt2020() {
    // This is how a pipeline actually does gamut conversion:
    // 1. Convert to linear f32 via RowConverter
    // 2. Apply gamut matrix
    // 3. Convert to target format via RowConverter

    // Step 1: sRGB u8 → linear f32
    let src_desc = PixelDescriptor::RGB8_SRGB;
    let linear_desc = PixelDescriptor::RGBF32_LINEAR;
    let to_linear = RowConverter::new(src_desc, linear_desc).unwrap();

    let src_bytes = [128u8, 200, 64]; // one pixel
    let mut linear_bytes = [0u8; 12];
    to_linear.convert_row(&src_bytes, &mut linear_bytes, 1);

    // Step 2: Apply BT.709 → BT.2020 gamut matrix
    let m = conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
    let mut linear_f32 = [
        f32::from_ne_bytes([linear_bytes[0], linear_bytes[1], linear_bytes[2], linear_bytes[3]]),
        f32::from_ne_bytes([linear_bytes[4], linear_bytes[5], linear_bytes[6], linear_bytes[7]]),
        f32::from_ne_bytes([linear_bytes[8], linear_bytes[9], linear_bytes[10], linear_bytes[11]]),
    ];
    let before = linear_f32;
    apply_matrix_f32(&mut linear_f32, m);

    // The gamut matrix should change the values (BT.709 → BT.2020 is not identity).
    let changed = linear_f32.iter().zip(before.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "gamut matrix should change at least one channel for a non-white color");

    // Step 3: Roundtrip back to BT.709 and verify
    let m_back = conversion_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Bt709).unwrap();
    apply_matrix_f32(&mut linear_f32, m_back);

    for c in 0..3 {
        assert!(
            (linear_f32[c] - before[c]).abs() < 1e-3,
            "gamut roundtrip ch{c}: {:.6} vs {:.6}",
            linear_f32[c],
            before[c]
        );
    }
}

#[test]
fn conversion_matrix_returns_none_for_same_primaries() {
    assert!(conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt709).is_none());
    assert!(conversion_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Bt2020).is_none());
}

#[test]
fn conversion_matrix_returns_none_for_unknown() {
    assert!(conversion_matrix(ColorPrimaries::Unknown, ColorPrimaries::Bt709).is_none());
    assert!(conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::Unknown).is_none());
}

#[test]
fn all_named_primaries_pairs_have_matrices() {
    let known = [ColorPrimaries::Bt709, ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020];
    for &from in &known {
        for &to in &known {
            if from != to {
                assert!(
                    conversion_matrix(from, to).is_some(),
                    "missing matrix for {from:?} → {to:?}"
                );
            }
        }
    }
}

#[test]
fn gamut_row_conversion_multi_pixel() {
    let m = conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3).unwrap();

    // 3 pixels, RGB f32
    let mut data = [
        0.5f32, 0.3, 0.8,  // pixel 0
        1.0, 1.0, 1.0,     // pixel 1 (white)
        0.0, 0.0, 0.0,     // pixel 2 (black)
    ];

    let original = data;
    apply_matrix_row_f32(&mut data, 3, m);

    // White should be ~preserved
    assert!((data[3] - 1.0).abs() < 1e-4, "white R");
    assert!((data[4] - 1.0).abs() < 1e-4, "white G");
    assert!((data[5] - 1.0).abs() < 1e-4, "white B");

    // Black should be exactly preserved
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);

    // Non-white pixel should change
    let changed = (0..3).any(|c| (data[c] - original[c]).abs() > 1e-4);
    assert!(changed, "non-white pixel should change with gamut conversion");
}

#[test]
fn gamut_rgba_row_preserves_alpha() {
    let m = conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();

    let mut data = [0.5f32, 0.3, 0.8, 0.42, 0.1, 0.9, 0.2, 0.99];
    apply_matrix_row_rgba_f32(&mut data, 2, m);

    assert_eq!(data[3], 0.42, "alpha pixel 0 must be preserved");
    assert_eq!(data[7], 0.99, "alpha pixel 1 must be preserved");
}

// ---------------------------------------------------------------------------
// Extension trait: XYZ matrix consistency
// ---------------------------------------------------------------------------

#[test]
fn xyz_matrices_invert_correctly() {
    for primaries in [ColorPrimaries::Bt709, ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020] {
        let to_xyz = primaries.to_xyz_matrix().unwrap();
        let from_xyz = primaries.from_xyz_matrix().unwrap();

        let original = [0.6f32, 0.3, 0.7];
        let mut v = original;
        apply_matrix_f32(&mut v, to_xyz);
        apply_matrix_f32(&mut v, from_xyz);

        for c in 0..3 {
            assert!(
                (v[c] - original[c]).abs() < 1e-4,
                "{primaries:?} XYZ roundtrip ch{c}: {:.6} vs {:.6}",
                v[c],
                original[c]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// PixelBufferConvertExt: color context preservation
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "imgref")]
fn convert_to_preserves_color_context() {
    use alloc::sync::Arc;
    use zenpixels_convert::{ColorContext, PixelBuffer};
    use zenpixels_convert::ext::PixelBufferConvertExt;

    extern crate alloc;

    let data = vec![100u8, 150, 200, 50, 100, 150];
    let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();

    // Attach a color context with fake ICC data.
    let ctx = Arc::new(ColorContext::from_icc(vec![0xAA; 32]));
    let buf = buf.with_color_context(ctx);

    // Convert to RGBA8 — should preserve color context.
    let out = buf.convert_to(PixelDescriptor::RGBA8_SRGB).unwrap();

    assert!(out.color_context().is_some(), "color context should be preserved after conversion");
    let out_ctx = out.color_context().unwrap();
    assert!(out_ctx.icc.is_some());
    assert_eq!(out_ctx.icc.as_ref().unwrap().len(), 32);
}
