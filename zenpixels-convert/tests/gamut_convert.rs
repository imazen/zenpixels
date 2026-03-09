//! Tests for gamut (color primaries) conversion through RowConverter.
//!
//! Validates that converting between BT.709, Display P3, and BT.2020
//! primaries produces correct results through the full conversion pipeline.

use zenpixels_convert::{
    ColorPrimaries, PixelDescriptor, RowConverter,
    ext::ColorPrimariesExt,
    gamut::apply_matrix_f32,
};

/// Make an f32 RGB descriptor with specific primaries.
fn f32_linear(primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::RGBF32_LINEAR.with_primaries(primaries)
}

/// Convert a single pixel through RowConverter.
fn convert_pixel_f32(src_desc: PixelDescriptor, dst_desc: PixelDescriptor, pixel: [f32; 3]) -> [f32; 3] {
    let conv = RowConverter::new(src_desc, dst_desc).unwrap();
    let src_bytes: Vec<u8> = pixel.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut dst_bytes = vec![0u8; 12];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let r = f32::from_ne_bytes([dst_bytes[0], dst_bytes[1], dst_bytes[2], dst_bytes[3]]);
    let g = f32::from_ne_bytes([dst_bytes[4], dst_bytes[5], dst_bytes[6], dst_bytes[7]]);
    let b = f32::from_ne_bytes([dst_bytes[8], dst_bytes[9], dst_bytes[10], dst_bytes[11]]);
    [r, g, b]
}

// ---------------------------------------------------------------------------
// Gamut conversion through RowConverter
// ---------------------------------------------------------------------------

#[test]
fn bt709_to_bt2020_white_preserved() {
    let src = f32_linear(ColorPrimaries::Bt709);
    let dst = f32_linear(ColorPrimaries::Bt2020);

    let result = convert_pixel_f32(src, dst, [1.0, 1.0, 1.0]);

    for c in 0..3 {
        assert!(
            (result[c] - 1.0).abs() < 1e-3,
            "white point ch{c}: expected ~1.0, got {:.6}",
            result[c]
        );
    }
}

#[test]
fn bt709_to_bt2020_roundtrip() {
    let bt709 = f32_linear(ColorPrimaries::Bt709);
    let bt2020 = f32_linear(ColorPrimaries::Bt2020);

    let original = [0.5, 0.3, 0.8];
    let intermediate = convert_pixel_f32(bt709, bt2020, original);
    let recovered = convert_pixel_f32(bt2020, bt709, intermediate);

    for c in 0..3 {
        assert!(
            (recovered[c] - original[c]).abs() < 1e-3,
            "BT.709 → BT.2020 → BT.709 roundtrip ch{c}: {:.6} vs {:.6}",
            recovered[c],
            original[c]
        );
    }
}

#[test]
fn bt709_to_display_p3_roundtrip() {
    let bt709 = f32_linear(ColorPrimaries::Bt709);
    let p3 = f32_linear(ColorPrimaries::DisplayP3);

    let original = [0.7, 0.2, 0.4];
    let intermediate = convert_pixel_f32(bt709, p3, original);
    let recovered = convert_pixel_f32(p3, bt709, intermediate);

    for c in 0..3 {
        assert!(
            (recovered[c] - original[c]).abs() < 1e-3,
            "BT.709 → P3 → BT.709 roundtrip ch{c}: {:.6} vs {:.6}",
            recovered[c],
            original[c]
        );
    }
}

#[test]
fn same_primaries_is_identity() {
    let desc = f32_linear(ColorPrimaries::Bt709);
    let pixel = [0.5, 0.3, 0.8];
    let result = convert_pixel_f32(desc, desc, pixel);

    for c in 0..3 {
        assert!(
            (result[c] - pixel[c]).abs() < 1e-6,
            "identity ch{c}: {:.6} vs {:.6}",
            result[c],
            pixel[c]
        );
    }
}

#[test]
fn unknown_primaries_to_known_is_relabel() {
    let unknown = f32_linear(ColorPrimaries::Unknown);
    let bt709 = f32_linear(ColorPrimaries::Bt709);

    // Unknown → BT.709 should be a relabel (no matrix applied).
    let pixel = [0.5, 0.3, 0.8];
    let result = convert_pixel_f32(unknown, bt709, pixel);

    for c in 0..3 {
        assert!(
            (result[c] - pixel[c]).abs() < 1e-6,
            "relabel ch{c}: {:.6} vs {:.6}",
            result[c],
            pixel[c]
        );
    }
}

// ---------------------------------------------------------------------------
// Gamut conversion combined with depth change
// ---------------------------------------------------------------------------

#[test]
fn bt709_u8_srgb_to_bt2020_f32_linear() {
    // This tests the compound path: sRGB u8 → linear f32 + gamut matrix.
    let src = PixelDescriptor::RGB8_SRGB; // BT.709 primaries by default
    let dst = f32_linear(ColorPrimaries::Bt2020);

    let conv = RowConverter::new(src, dst).unwrap();

    // White pixel
    let src_bytes = [255u8, 255, 255];
    let mut dst_bytes = [0u8; 12];
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);

    let r = f32::from_ne_bytes([dst_bytes[0], dst_bytes[1], dst_bytes[2], dst_bytes[3]]);
    let g = f32::from_ne_bytes([dst_bytes[4], dst_bytes[5], dst_bytes[6], dst_bytes[7]]);
    let b = f32::from_ne_bytes([dst_bytes[8], dst_bytes[9], dst_bytes[10], dst_bytes[11]]);

    // White should be ~1.0 in any primaries.
    assert!(
        (r - 1.0).abs() < 0.02,
        "white R in BT.2020: {r:.4}"
    );
    assert!(
        (g - 1.0).abs() < 0.02,
        "white G in BT.2020: {g:.4}"
    );
    assert!(
        (b - 1.0).abs() < 0.02,
        "white B in BT.2020: {b:.4}"
    );
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
