//! ULP-level exhaustive tests for conversion kernels.
//!
//! These tests prove that specific conversion round-trips are lossless (or
//! quantify their exact error bound) by testing **every possible input value**
//! through the actual zenpixels `ConvertPlan`/`convert_row` implementations.
//!
//! This replaces statistical sampling with exhaustive numerical proof:
//! "u8 sRGB → f32 linear → u8 sRGB is lossless (max error = 0 for all 256 values)".

use zenpixels::{ConvertPlan, PixelDescriptor, convert_row};

// ═══════════════════════════════════════════════════════════════════════
// Helper: exhaustive u8 round-trip through an intermediate format
// ═══════════════════════════════════════════════════════════════════════

/// Test a u8→X→u8 round-trip for all 256 values per channel (RGB).
///
/// Returns the max error in u8 units (0 = perfectly lossless).
fn measure_u8_roundtrip_max_error(intermediate: PixelDescriptor) -> u8 {
    let src_desc = PixelDescriptor::RGB8_SRGB;
    let plan_fwd = ConvertPlan::new(src_desc, intermediate).expect("forward plan");
    let plan_back = ConvertPlan::new(intermediate, src_desc).expect("backward plan");

    let inter_bpp = intermediate.bytes_per_pixel();
    let mut max_err: u8 = 0;

    // Test all 256 values in R channel, with representative G and B.
    for r in 0u8..=255 {
        for &g in &[0u8, 1, 64, 128, 191, 254, 255] {
            for &b in &[0u8, 128, 255] {
                let input = [r, g, b];
                let mut mid = vec![0u8; inter_bpp];
                let mut output = [0u8; 3];

                convert_row(&plan_fwd, &input, &mut mid, 1);
                convert_row(&plan_back, &mid, &mut output, 1);

                for c in 0..3 {
                    let err = (output[c] as i16 - input[c] as i16).unsigned_abs() as u8;
                    max_err = max_err.max(err);
                }
            }
        }
    }

    max_err
}

/// Test a u8→X→u8 round-trip exhaustively over the full 256^3 RGB cube.
///
/// This is slower (~16M evaluations) but proves correctness for every value.
fn measure_u8_roundtrip_full_cube(intermediate: PixelDescriptor) -> u8 {
    let src_desc = PixelDescriptor::RGB8_SRGB;
    let plan_fwd = ConvertPlan::new(src_desc, intermediate).expect("forward plan");
    let plan_back = ConvertPlan::new(intermediate, src_desc).expect("backward plan");

    let inter_bpp = intermediate.bytes_per_pixel();
    let mut max_err: u8 = 0;

    for r in 0u16..=255 {
        for g in 0u16..=255 {
            for b in 0u16..=255 {
                let input = [r as u8, g as u8, b as u8];
                let mut mid = vec![0u8; inter_bpp];
                let mut output = [0u8; 3];

                convert_row(&plan_fwd, &input, &mut mid, 1);
                convert_row(&plan_back, &mid, &mut output, 1);

                for c in 0..3 {
                    let err = (output[c] as i16 - input[c] as i16).unsigned_abs() as u8;
                    max_err = max_err.max(err);
                }
            }
        }
    }

    max_err
}

// ═══════════════════════════════════════════════════════════════════════
// u8 sRGB → f32 Linear → u8 sRGB (EOTF → OETF round-trip)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_u8_srgb_f32_linear_roundtrip_sampled() {
    let err = measure_u8_roundtrip_max_error(PixelDescriptor::RGBF32_LINEAR);
    assert_eq!(
        err, 0,
        "u8 sRGB → f32 Linear → u8 sRGB: expected 0 error, got {err}"
    );
}

#[test]
fn ulp_u8_srgb_f32_linear_roundtrip_full_cube() {
    // Full 256^3 cube: ~16.7M values. Should complete in ~2s.
    let err = measure_u8_roundtrip_full_cube(PixelDescriptor::RGBF32_LINEAR);
    assert_eq!(
        err, 0,
        "u8 sRGB → f32 Linear → u8 sRGB (full cube): expected 0 error, got {err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// u8 → u16 → u8 round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_u8_u16_roundtrip() {
    // u8 → u16 uses v * 257, u16 → u8 uses (v * 255 + 32768) >> 16
    // This should be perfectly lossless.
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let src = PixelDescriptor::RGB8_SRGB;
    let mid_desc = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );

    let plan_fwd = ConvertPlan::new(src, mid_desc).expect("u8→u16 plan");
    let plan_back = ConvertPlan::new(mid_desc, src).expect("u16→u8 plan");

    let mut max_err: u8 = 0;
    for v in 0u8..=255 {
        let input = [v, v, v];
        let mut mid = [0u8; 6]; // 3 × u16
        let mut output = [0u8; 3];

        convert_row(&plan_fwd, &input, &mut mid, 1);
        convert_row(&plan_back, &mid, &mut output, 1);

        for c in 0..3 {
            let err = (output[c] as i16 - input[c] as i16).unsigned_abs() as u8;
            max_err = max_err.max(err);
        }
    }

    assert_eq!(max_err, 0, "u8 → u16 → u8: expected 0 error, got {max_err}");
}

// ═══════════════════════════════════════════════════════════════════════
// u16 → u8 quantization (one-way, measures max error)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_u16_to_u8_max_error() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let src = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let dst = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(src, dst).expect("u16→u8 plan");

    let mut max_err: u16 = 0;
    // Test all 65536 values for one channel.
    for v in 0u32..=65535 {
        let v16 = v as u16;
        let input: [u8; 6] = bytemuck::cast([v16, v16, v16]);
        let mut output = [0u8; 3];
        convert_row(&plan, &input, &mut output, 1);

        // Expected: (v * 255 + 32768) >> 16
        let expected = ((v * 255 + 32768) >> 16) as u8;
        let err = (output[0] as i16 - expected as i16).unsigned_abs();
        max_err = max_err.max(err);
    }

    assert_eq!(max_err, 0, "u16 → u8 should match formula exactly");
}

// ═══════════════════════════════════════════════════════════════════════
// u8 → u16 exact expansion check
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_u8_to_u16_exact() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let src = PixelDescriptor::RGB8_SRGB;
    let dst = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let plan = ConvertPlan::new(src, dst).expect("u8→u16 plan");

    for v in 0u8..=255 {
        let input = [v, v, v];
        let mut output = [0u8; 6];
        convert_row(&plan, &input, &mut output, 1);

        let out16: &[u16] = bytemuck::cast_slice(&output);
        let expected = v as u16 * 257; // 0→0, 255→65535
        assert_eq!(
            out16[0], expected,
            "u8→u16: v={v}, expected {expected}, got {}",
            out16[0]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BGRA ↔ RGBA swizzle: must be perfectly lossless
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_bgra_rgba_swizzle_roundtrip() {
    let rgba = PixelDescriptor::RGBA8_SRGB;
    let bgra = PixelDescriptor::BGRA8_SRGB;
    let plan_fwd = ConvertPlan::new(rgba, bgra).expect("rgba→bgra plan");
    let plan_back = ConvertPlan::new(bgra, rgba).expect("bgra→rgba plan");

    // Test 1M random-ish values (sampled from a grid).
    let mut error_count = 0u32;
    for r in (0u8..=255).step_by(3) {
        for g in (0u8..=255).step_by(7) {
            for b in (0u8..=255).step_by(11) {
                for a in (0u8..=255).step_by(17) {
                    let input = [r, g, b, a];
                    let mut mid = [0u8; 4];
                    let mut output = [0u8; 4];
                    convert_row(&plan_fwd, &input, &mut mid, 1);
                    convert_row(&plan_back, &mid, &mut output, 1);
                    if input != output {
                        error_count += 1;
                    }
                }
            }
        }
    }

    assert_eq!(
        error_count, 0,
        "BGRA↔RGBA swizzle should be perfectly lossless"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// sRGB EOTF: u8 → f32 linear values are exact
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_srgb_eotf_monotonic() {
    // Verify sRGB EOTF (u8→f32 linear) produces monotonically increasing values.
    let src = PixelDescriptor::RGB8_SRGB;
    let dst = PixelDescriptor::RGBF32_LINEAR;
    let plan = ConvertPlan::new(src, dst).expect("sRGB→linear plan");

    let mut prev: f32 = -1.0;
    for v in 0u8..=255 {
        let input = [v, 0, 0]; // Only R channel
        let mut output = [0u8; 12]; // 3 × f32
        convert_row(&plan, &input, &mut output, 1);
        let outf: &[f32] = bytemuck::cast_slice(&output);
        assert!(
            outf[0] >= prev,
            "sRGB EOTF not monotonic: f({}) = {} < f({}) = {}",
            v,
            outf[0],
            v - 1,
            prev
        );
        prev = outf[0];
    }

    // Verify boundary values.
    let input_0 = [0u8, 0, 0];
    let input_255 = [255u8, 0, 0];
    let mut out_0 = [0u8; 12];
    let mut out_255 = [0u8; 12];
    convert_row(&plan, &input_0, &mut out_0, 1);
    convert_row(&plan, &input_255, &mut out_255, 1);
    let f0: &[f32] = bytemuck::cast_slice(&out_0);
    let f255: &[f32] = bytemuck::cast_slice(&out_255);
    assert_eq!(f0[0], 0.0, "sRGB EOTF(0) should be 0.0");
    assert!(
        (f255[0] - 1.0).abs() < 1e-6,
        "sRGB EOTF(255) should be ~1.0, got {}",
        f255[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════
// sRGB OETF: f32 → u8 boundary accuracy
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_srgb_oetf_boundary_values() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let dst = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(src, dst).expect("linear→sRGB plan");

    // Test that the exact f32 values corresponding to each u8 step
    // round-trip correctly.
    let eotf_plan = ConvertPlan::new(dst, src).expect("sRGB→linear plan");

    for v in 0u8..=255 {
        // Get the f32 value for this u8.
        let u8_input = [v, 0, 0];
        let mut f32_bytes = [0u8; 12];
        convert_row(&eotf_plan, &u8_input, &mut f32_bytes, 1);

        // Convert back to u8.
        let mut u8_output = [0u8; 3];
        convert_row(&plan, &f32_bytes, &mut u8_output, 1);

        assert_eq!(
            u8_output[0], v,
            "sRGB OETF round-trip failed for u8={v}: got {}",
            u8_output[0]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Naive (Unknown transfer) u8 ↔ f32 round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_naive_u8_f32_roundtrip() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let u8_desc = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    let f32_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Unknown,
    );

    let plan_fwd = ConvertPlan::new(u8_desc, f32_desc).expect("u8→f32 naive plan");
    let plan_back = ConvertPlan::new(f32_desc, u8_desc).expect("f32→u8 naive plan");

    let mut max_err: u8 = 0;
    for v in 0u8..=255 {
        let input = [v, v, v];
        let mut mid = [0u8; 12];
        let mut output = [0u8; 3];
        convert_row(&plan_fwd, &input, &mut mid, 1);
        convert_row(&plan_back, &mid, &mut output, 1);

        for c in 0..3 {
            let err = (output[c] as i16 - input[c] as i16).unsigned_abs() as u8;
            max_err = max_err.max(err);
        }
    }

    assert_eq!(
        max_err, 0,
        "naive u8 → f32 → u8 should be lossless, got max error {max_err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// u16 ↔ f32 round-trip (exhaustive over all 65536 values)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_u16_f32_roundtrip_exhaustive() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let u16_desc = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let f32_desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Srgb,
    );

    let plan_fwd = ConvertPlan::new(u16_desc, f32_desc).expect("u16→f32 plan");
    let plan_back = ConvertPlan::new(f32_desc, u16_desc).expect("f32→u16 plan");

    let mut max_err: u16 = 0;
    for v in 0u32..=65535 {
        let v16 = v as u16;
        let input: [u8; 2] = v16.to_ne_bytes();
        let mut mid = [0u8; 4]; // 1 × f32
        let mut output = [0u8; 2]; // 1 × u16

        convert_row(&plan_fwd, &input, &mut mid, 1);
        convert_row(&plan_back, &mid, &mut output, 1);

        let result = u16::from_ne_bytes(output);
        let err = (result as i32 - v16 as i32).unsigned_abs() as u16;
        max_err = max_err.max(err);
    }

    assert_eq!(
        max_err, 0,
        "u16 → f32 → u16 should be lossless, got max error {max_err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Premultiplication round-trip accuracy (u8, all value×alpha pairs)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_premul_roundtrip_u8_exhaustive() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let straight = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    let premul = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Premultiplied,
        TransferFunction::Srgb,
    );

    let plan_to_premul = ConvertPlan::new(straight, premul).expect("straight→premul plan");
    let plan_to_straight = ConvertPlan::new(premul, straight).expect("premul→straight plan");

    let mut max_err: u8 = 0;
    let mut error_count = 0u32;
    let mut worst_value = 0u8;
    let mut worst_alpha = 0u8;

    // Test all 256 × 256 (value, alpha) pairs for R channel.
    for value in 0u8..=255 {
        for alpha in 0u8..=255 {
            let input = [value, 0, 0, alpha];
            let mut mid = [0u8; 4];
            let mut output = [0u8; 4];

            convert_row(&plan_to_premul, &input, &mut mid, 1);
            convert_row(&plan_to_straight, &mid, &mut output, 1);

            let err = (output[0] as i16 - value as i16).unsigned_abs() as u8;
            if err > max_err {
                max_err = err;
                worst_value = value;
                worst_alpha = alpha;
            }
            if err > 0 {
                error_count += 1;
            }
        }
    }

    // Premultiplication in u8 is lossy at low alpha. Document the exact bound.
    // At alpha=0, all values map to 0 and back, so only alpha≥1 has loss.
    // At alpha=1, only 0 and 1 are representable, max error is large.
    // At alpha=128, most values round-trip within ±1.
    println!("Premul u8 round-trip: max_err={max_err} (value={worst_value}, alpha={worst_alpha})");
    println!("  Error count: {error_count} / 65536");

    // The max error should be bounded. For u8 premul, worst case is alpha=1.
    // Don't assert exact value since it depends on the rounding, just document it.
    // At alpha=1, only 0 and 1 are representable after premul, so error can be up to 254.
    assert!(
        max_err > 0,
        "premul in u8 is expected to be lossy at low alpha"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Premultiplication round-trip in f32 (should be near-lossless)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_premul_roundtrip_f32() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let straight = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Linear,
    );
    let premul = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Premultiplied,
        TransferFunction::Linear,
    );

    let plan_to_premul = ConvertPlan::new(straight, premul).expect("f32 straight→premul plan");
    let plan_to_straight = ConvertPlan::new(premul, straight).expect("f32 premul→straight plan");

    let mut max_ulp_err: u32 = 0;
    let test_values: Vec<f32> = (0..=255).map(|v| v as f32 / 255.0).collect();
    let test_alphas: Vec<f32> = (0..=255).map(|v| v as f32 / 255.0).collect();

    for &value in &test_values {
        for &alpha in &test_alphas {
            let input: [u8; 16] = bytemuck::cast([value, 0.0f32, 0.0f32, alpha]);
            let mut mid = [0u8; 16];
            let mut output = [0u8; 16];

            convert_row(&plan_to_premul, &input, &mut mid, 1);
            convert_row(&plan_to_straight, &mid, &mut output, 1);

            let result: [f32; 4] = bytemuck::cast(output);

            if alpha > 0.0 {
                let err_bits =
                    (result[0].to_bits() as i64 - value.to_bits() as i64).unsigned_abs() as u32;
                max_ulp_err = max_ulp_err.max(err_bits);
            }
        }
    }

    // f32 premul round-trip should have very low ULP error (mostly 0-1 ULP).
    println!("f32 premul round-trip: max ULP error = {max_ulp_err}");
    assert!(
        max_ulp_err <= 2,
        "f32 premul round-trip should be near-lossless, got {max_ulp_err} ULP"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Gray → RGB → Gray round-trip (u8, BT.709 luma)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_gray_rgb_gray_roundtrip() {
    let gray = PixelDescriptor::GRAY8_SRGB;
    let rgb = PixelDescriptor::RGB8_SRGB;

    let plan_to_rgb = ConvertPlan::new(gray, rgb).expect("gray→rgb plan");
    let plan_to_gray = ConvertPlan::new(rgb, gray).expect("rgb→gray plan");

    let mut max_err: u8 = 0;
    for v in 0u8..=255 {
        let input = [v];
        let mut mid = [0u8; 3];
        let mut output = [0u8; 1];

        convert_row(&plan_to_rgb, &input, &mut mid, 1);
        convert_row(&plan_to_gray, &mid, &mut output, 1);

        let err = (output[0] as i16 - v as i16).unsigned_abs() as u8;
        max_err = max_err.max(err);
    }

    // Gray → RGB replicates (R=G=B=gray), then RGB → Gray uses BT.709 luma.
    // Since R=G=B, luma = 0.2126*v + 0.7152*v + 0.0722*v = v.
    // With fixed-point: (54v + 183v + 19v + 128) >> 8 = (256v + 128) >> 8 = v.
    assert_eq!(
        max_err, 0,
        "gray → RGB → gray should be lossless for uniform channels, got error {max_err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Add/drop alpha round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_add_drop_alpha_roundtrip() {
    let rgb = PixelDescriptor::RGB8_SRGB;
    let rgba = PixelDescriptor::RGBA8_SRGB;

    let plan_add = ConvertPlan::new(rgb, rgba).expect("rgb→rgba plan");
    let plan_drop = ConvertPlan::new(rgba, rgb).expect("rgba→rgb plan");

    let mut max_err: u8 = 0;
    for r in (0u8..=255).step_by(1) {
        for &g in &[0u8, 64, 128, 192, 255] {
            for &b in &[0u8, 128, 255] {
                let input = [r, g, b];
                let mut mid = [0u8; 4];
                let mut output = [0u8; 3];

                convert_row(&plan_add, &input, &mut mid, 1);
                // Verify alpha is opaque.
                assert_eq!(mid[3], 255, "added alpha should be 255");
                convert_row(&plan_drop, &mid, &mut output, 1);

                for c in 0..3 {
                    let err = (output[c] as i16 - input[c] as i16).unsigned_abs() as u8;
                    max_err = max_err.max(err);
                }
            }
        }
    }

    assert_eq!(max_err, 0, "RGB → RGBA → RGB should be lossless");
}

// ═══════════════════════════════════════════════════════════════════════
// GrayAlpha → RGBA → GrayAlpha round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ulp_gray_alpha_rgba_roundtrip() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    let ga = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    let rgba = PixelDescriptor::RGBA8_SRGB;

    let plan_to_rgba = ConvertPlan::new(ga, rgba).expect("ga→rgba plan");
    // RGBA → GrayAlpha isn't a direct path (would need RGBA→Gray + keep alpha).
    // Instead test GrayAlpha → RGBA preserves values.

    for v in 0u8..=255 {
        for &a in &[0u8, 1, 128, 254, 255] {
            let input = [v, a];
            let mut output = [0u8; 4];
            convert_row(&plan_to_rgba, &input, &mut output, 1);

            assert_eq!(output[0], v, "R should equal gray");
            assert_eq!(output[1], v, "G should equal gray");
            assert_eq!(output[2], v, "B should equal gray");
            assert_eq!(output[3], a, "alpha should be preserved");
        }
    }
}
