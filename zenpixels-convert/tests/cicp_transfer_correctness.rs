//! Comprehensive CICP and transfer function correctness tests.
//!
//! Tests every CICP code point mapping, every transfer function conversion
//! path, and every cross-TF conversion for correctness, monotonicity,
//! and roundtrip accuracy.

use zenpixels::buffer::PixelBuffer;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Cicp, ColorPrimaries, NamedProfile, PixelDescriptor,
    PixelFormat, SignalRange, TransferFunction,
};
use zenpixels_convert::RowConverter;
use zenpixels_convert::ext::{PixelBufferConvertExt, TransferFunctionExt};

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: CICP Code Point Mapping Exhaustiveness
// ═══════════════════════════════════════════════════════════════════════════

/// Every TransferFunction variant has a CICP code (except Unknown).
#[test]
fn all_transfer_functions_have_cicp_code() {
    let known = [
        (TransferFunction::Linear, 8),
        (TransferFunction::Srgb, 13),
        (TransferFunction::Bt709, 1),
        (TransferFunction::Pq, 16),
        (TransferFunction::Hlg, 18),
    ];
    for (tf, expected_code) in known {
        let code = tf.to_cicp();
        assert_eq!(
            code,
            Some(expected_code),
            "{tf:?} should map to CICP code {expected_code}"
        );
    }
    assert_eq!(TransferFunction::Unknown.to_cicp(), None);
}

/// Every ColorPrimaries variant has a CICP code (except Unknown).
#[test]
fn all_color_primaries_have_cicp_code() {
    let known = [
        (ColorPrimaries::Bt709, 1),
        (ColorPrimaries::Bt2020, 9),
        (ColorPrimaries::DisplayP3, 12),
    ];
    for (cp, expected_code) in known {
        let code = cp.to_cicp();
        assert_eq!(
            code,
            Some(expected_code),
            "{cp:?} should map to CICP code {expected_code}"
        );
    }
    assert_eq!(ColorPrimaries::Unknown.to_cicp(), None);
}

/// from_cicp and to_cicp round-trip for primary codes (aliases like 6↔7 may collapse).
#[test]
fn transfer_function_cicp_bijection() {
    // Primary codes that must round-trip exactly
    for code in [1u8, 8, 13, 16, 18] {
        let tf = TransferFunction::from_cicp(code).unwrap();
        assert_eq!(tf.to_cicp(), Some(code));
    }
    // Aliases that map to another primary code (SMPTE 170M/240M → BT.709 curve)
    assert_eq!(
        TransferFunction::from_cicp(6),
        Some(TransferFunction::Bt709)
    );
    assert_eq!(
        TransferFunction::from_cicp(7),
        Some(TransferFunction::Bt709)
    );
    // Reverse: every non-Unknown enum maps to its primary code
    for tf in [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ] {
        let code = tf.to_cicp().unwrap();
        let back = TransferFunction::from_cicp(code).unwrap();
        assert_eq!(back, tf);
    }
}

/// from_cicp and to_cicp round-trip for primary codes.
#[test]
fn color_primaries_cicp_bijection() {
    // Primary codes that must round-trip exactly
    for code in [1u8, 9, 12] {
        let cp = ColorPrimaries::from_cicp(code).unwrap();
        assert_eq!(cp.to_cicp(), Some(code));
    }
    for cp in [
        ColorPrimaries::Bt709,
        ColorPrimaries::Bt2020,
        ColorPrimaries::DisplayP3,
    ] {
        let code = cp.to_cicp().unwrap();
        let back = ColorPrimaries::from_cicp(code).unwrap();
        assert_eq!(back, cp);
    }
}

/// Unrecognized CICP codes return None.
#[test]
fn unknown_cicp_codes_return_none() {
    // TransferFunction: recognized = 1, 6, 7, 8, 13, 16, 18
    for code in [0, 2, 3, 4, 5, 9, 10, 11, 12, 14, 15, 17, 19, 99, 255] {
        assert!(
            TransferFunction::from_cicp(code).is_none(),
            "TC code {code} should not be recognized"
        );
    }
    // ColorPrimaries: recognized = 1, 9, 12
    for code in [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 22, 99, 255] {
        assert!(
            ColorPrimaries::from_cicp(code).is_none(),
            "CP code {code} should not be recognized"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: CICP ↔ PixelDescriptor Roundtrip
// ═══════════════════════════════════════════════════════════════════════════

/// All four CICP constants roundtrip through from_descriptor/to_descriptor.
#[test]
fn cicp_constants_descriptor_roundtrip() {
    for (cicp, name) in [
        (Cicp::SRGB, "SRGB"),
        (Cicp::BT2100_PQ, "BT2100_PQ"),
        (Cicp::BT2100_HLG, "BT2100_HLG"),
        (Cicp::DISPLAY_P3, "DISPLAY_P3"),
    ] {
        for format in [PixelFormat::Rgb8, PixelFormat::RgbF32, PixelFormat::Rgba8] {
            let desc = cicp.to_descriptor(format);
            assert_eq!(desc.format, format, "{name} format");
            assert_eq!(
                desc.signal_range,
                if cicp.full_range {
                    SignalRange::Full
                } else {
                    SignalRange::Narrow
                },
                "{name} signal range"
            );

            // Roundtrip back to CICP
            let back = Cicp::from_descriptor(&desc);
            let back = back.unwrap_or_else(|| {
                panic!("{name} ({format:?}): from_descriptor returned None for {desc:?}")
            });
            assert_eq!(back.color_primaries, cicp.color_primaries, "{name} CP");
            assert_eq!(
                back.transfer_characteristics, cicp.transfer_characteristics,
                "{name} TC"
            );
            assert_eq!(back.full_range, cicp.full_range, "{name} full_range");
            // matrix_coefficients is always 0 when round-tripping through descriptor
            assert_eq!(back.matrix_coefficients, 0, "{name} MC");
        }
    }
}

/// Narrow-range CICP roundtrips correctly.
#[test]
fn cicp_narrow_range_roundtrip() {
    let cicp = Cicp::new(1, 13, 0, false);
    let desc = cicp.to_descriptor(PixelFormat::Rgb8);
    assert_eq!(desc.signal_range, SignalRange::Narrow);
    let back = Cicp::from_descriptor(&desc).unwrap();
    assert!(!back.full_range);
}

/// Descriptors with Unknown transfer or primaries cannot create a CICP.
#[test]
fn from_descriptor_rejects_unknown() {
    let unknown_tf = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Unknown,
    );
    assert!(
        Cicp::from_descriptor(&unknown_tf).is_none(),
        "Unknown TF → None"
    );

    let unknown_cp = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Unknown);
    assert!(
        Cicp::from_descriptor(&unknown_cp).is_none(),
        "Unknown primaries → None"
    );
}

/// to_descriptor correctly derives alpha from pixel format.
#[test]
fn cicp_to_descriptor_alpha_handling() {
    // Format without alpha → None
    let desc = Cicp::SRGB.to_descriptor(PixelFormat::Rgb8);
    assert!(desc.alpha().is_none());

    // Format with alpha → Some(Straight)
    let desc = Cicp::SRGB.to_descriptor(PixelFormat::Rgba8);
    assert_eq!(desc.alpha(), Some(AlphaMode::Straight));

    // Gray format → None
    let desc = Cicp::SRGB.to_descriptor(PixelFormat::Gray8);
    assert!(desc.alpha().is_none());

    // GrayAlpha format → Some(Straight)
    let desc = Cicp::SRGB.to_descriptor(PixelFormat::GrayA8);
    assert_eq!(desc.alpha(), Some(AlphaMode::Straight));
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: NamedProfile ↔ CICP Roundtrip
// ═══════════════════════════════════════════════════════════════════════════

/// Every NamedProfile (except AdobeRgb) roundtrips through CICP.
#[test]
fn named_profile_cicp_roundtrip_all() {
    let profiles = [
        (NamedProfile::Srgb, "Srgb"),
        (NamedProfile::DisplayP3, "DisplayP3"),
        (NamedProfile::Bt2020, "Bt2020"),
        (NamedProfile::Bt2020Pq, "Bt2020Pq"),
        (NamedProfile::Bt2020Hlg, "Bt2020Hlg"),
        (NamedProfile::LinearSrgb, "LinearSrgb"),
    ];
    for (profile, name) in profiles {
        let cicp = profile
            .to_cicp()
            .unwrap_or_else(|| panic!("{name} should have CICP"));
        let back = NamedProfile::from_cicp(cicp)
            .unwrap_or_else(|| panic!("{name}: from_cicp({cicp:?}) returned None"));
        assert_eq!(back, profile, "{name} roundtrip");
    }
}

/// AdobeRgb has no CICP mapping.
#[test]
fn adobe_rgb_has_no_cicp() {
    assert!(NamedProfile::AdobeRgb.to_cicp().is_none());
}

/// BT.2100 PQ/HLG accept any matrix coefficient (YCbCr or RGB).
#[test]
fn bt2100_accepts_any_matrix() {
    for mc in [0, 1, 6, 9] {
        let pq = Cicp::new(9, 16, mc, true);
        assert_eq!(
            NamedProfile::from_cicp(pq),
            Some(NamedProfile::Bt2020Pq),
            "BT.2100 PQ with mc={mc}"
        );
        let hlg = Cicp::new(9, 18, mc, true);
        assert_eq!(
            NamedProfile::from_cicp(hlg),
            Some(NamedProfile::Bt2020Hlg),
            "BT.2100 HLG with mc={mc}"
        );
    }
}

/// CICP name functions cover all ITU-T H.273 codes we list.
#[test]
fn cicp_name_functions_cover_all_listed_codes() {
    // Color primaries: all codes in the match
    let cp_codes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];
    for code in cp_codes {
        let name = Cicp::color_primaries_name(code);
        assert_ne!(name, "Unknown", "CP code {code} should have a name");
    }
    assert_eq!(Cicp::color_primaries_name(3), "Unknown"); // gap
    assert_eq!(Cicp::color_primaries_name(13), "Unknown"); // gap
    assert_eq!(Cicp::color_primaries_name(200), "Unknown");

    // Transfer characteristics: all codes in the match
    let tc_codes = [
        0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    ];
    for code in tc_codes {
        let name = Cicp::transfer_characteristics_name(code);
        assert_ne!(name, "Unknown", "TC code {code} should have a name");
    }
    assert_eq!(Cicp::transfer_characteristics_name(3), "Unknown");
    assert_eq!(Cicp::transfer_characteristics_name(19), "Unknown");

    // Matrix coefficients: all codes in the match
    let mc_codes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    for code in mc_codes {
        let name = Cicp::matrix_coefficients_name(code);
        assert_ne!(name, "Unknown", "MC code {code} should have a name");
    }
    assert_eq!(Cicp::matrix_coefficients_name(3), "Unknown");
    assert_eq!(Cicp::matrix_coefficients_name(15), "Unknown");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Scalar Transfer Function Correctness
// ═══════════════════════════════════════════════════════════════════════════

/// All transfer functions are monotonically increasing on [0, 1].
#[test]
fn all_transfer_functions_monotonic() {
    let tfs = [
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Linear,
    ];
    let n = 1000;
    for tf in tfs {
        let mut prev_lin = -1.0f32;
        let mut prev_delin = -1.0f32;
        for i in 0..=n {
            let v = i as f32 / n as f32;
            let lin = tf.linearize(v);
            let delin = tf.delinearize(v);
            assert!(
                lin >= prev_lin,
                "{tf:?} linearize not monotonic at {v}: {lin} < {prev_lin}"
            );
            assert!(
                delin >= prev_delin,
                "{tf:?} delinearize not monotonic at {v}: {delin} < {prev_delin}"
            );
            prev_lin = lin;
            prev_delin = delin;
        }
    }
}

/// All transfer functions have correct boundary values.
#[test]
fn all_transfer_functions_boundaries() {
    let tfs = [
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Linear,
    ];
    for tf in tfs {
        // linearize(0) = 0
        let lin0 = tf.linearize(0.0);
        assert!(
            lin0.abs() < 1e-6,
            "{tf:?} linearize(0) = {lin0}, expected 0"
        );

        // delinearize(0) = 0
        let delin0 = tf.delinearize(0.0);
        assert!(
            delin0.abs() < 1e-6,
            "{tf:?} delinearize(0) = {delin0}, expected 0"
        );

        // linearize(1) ≈ 1 (for SDR; PQ/HLG map 1→1)
        let lin1 = tf.linearize(1.0);
        assert!(
            (lin1 - 1.0).abs() < 1e-4,
            "{tf:?} linearize(1) = {lin1}, expected ~1"
        );

        // delinearize(1) ≈ 1
        let delin1 = tf.delinearize(1.0);
        assert!(
            (delin1 - 1.0).abs() < 1e-4,
            "{tf:?} delinearize(1) = {delin1}, expected ~1"
        );
    }
}

/// Scalar roundtrip: linearize(delinearize(v)) ≈ v for all TFs.
#[test]
fn scalar_linearize_delinearize_roundtrip() {
    let tfs = [
        (TransferFunction::Srgb, 1e-5),
        (TransferFunction::Bt709, 1e-5),
        (TransferFunction::Pq, 5e-4), // rational poly has lower precision
        (TransferFunction::Hlg, 1e-4),
        (TransferFunction::Linear, 0.0),
    ];
    let test_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];

    for (tf, tol) in tfs {
        for &v in &test_values {
            let rt = tf.linearize(tf.delinearize(v));
            assert!(
                (rt - v).abs() <= tol,
                "{tf:?} roundtrip(delinearize→linearize) at {v}: got {rt}, delta={}",
                (rt - v).abs()
            );
            let rt2 = tf.delinearize(tf.linearize(v));
            assert!(
                (rt2 - v).abs() <= tol,
                "{tf:?} roundtrip(linearize→delinearize) at {v}: got {rt2}, delta={}",
                (rt2 - v).abs()
            );
        }
    }
}

/// sRGB mid-gray known value: sRGB(0.5) → linear ≈ 0.214.
#[test]
fn srgb_known_midgray() {
    let lin = TransferFunction::Srgb.linearize(0.5);
    assert!(
        (lin - 0.214).abs() < 0.003,
        "sRGB(0.5) → linear = {lin}, expected ~0.214"
    );
    let back = TransferFunction::Srgb.delinearize(lin);
    assert!(
        (back - 0.5).abs() < 1e-5,
        "sRGB delinearize({lin}) = {back}, expected 0.5"
    );
}

/// BT.709 mid-gray known value.
/// BT.709 EOTF differs from sRGB: 0.5 → ~0.26 (gamma ~2.2 with linear toe).
#[test]
fn bt709_known_midgray() {
    let lin = TransferFunction::Bt709.linearize(0.5);
    assert!(
        lin > 0.22 && lin < 0.30,
        "BT.709(0.5) → linear = {lin}, expected ~0.26"
    );
}

/// Linear is identity.
#[test]
fn linear_is_exact_identity() {
    for v in [0.0f32, 0.001, 0.1, 0.5, 0.99, 1.0, 2.5] {
        assert_eq!(TransferFunction::Linear.linearize(v), v);
        assert_eq!(TransferFunction::Linear.delinearize(v), v);
    }
}

/// Unknown is identity (pass-through).
#[test]
fn unknown_is_identity() {
    for v in [0.0f32, 0.5, 1.0] {
        assert_eq!(TransferFunction::Unknown.linearize(v), v);
        assert_eq!(TransferFunction::Unknown.delinearize(v), v);
    }
}

/// sRGB linearize is nonlinear: midpoint of input is NOT midpoint of output.
#[test]
fn srgb_is_nonlinear() {
    let lin = TransferFunction::Srgb.linearize(0.5);
    // If linear, would be 0.5. sRGB mid-gray is ~0.214.
    assert!(
        (lin - 0.5).abs() > 0.1,
        "sRGB(0.5) = {lin} — should not be 0.5"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 5: F32↔F32 Transfer Function Conversion Paths
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: create a RowConverter for F32 RGB with given TFs.
fn f32_converter(from_tf: TransferFunction, to_tf: TransferFunction) -> RowConverter {
    let from = PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, from_tf);
    let to = PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, to_tf);
    RowConverter::new(from, to)
        .unwrap_or_else(|e| panic!("Failed to create converter {from_tf:?}→{to_tf:?}: {e:?}"))
}

/// Helper: convert a single F32 RGB pixel through a RowConverter.
fn convert_f32_pixel(conv: &mut RowConverter, r: f32, g: f32, b: f32) -> [f32; 3] {
    let src: Vec<u8> = [r, g, b].iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut dst = vec![0u8; 12];
    conv.convert_row(&src, &mut dst, 1);
    let f: &[f32] = bytemuck::cast_slice(&dst);
    [f[0], f[1], f[2]]
}

/// All 20 non-identity F32↔F32 TF conversions can be created.
#[test]
fn all_f32_tf_pairs_create_converters() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ];
    for &from_tf in &tfs {
        for &to_tf in &tfs {
            if from_tf == to_tf {
                continue;
            }
            let from = PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, from_tf);
            let to = PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, to_tf);
            assert!(
                RowConverter::new(from, to).is_ok(),
                "Should create converter {from_tf:?} → {to_tf:?}"
            );
        }
    }
}

/// All F32↔F32 TF conversion pairs roundtrip correctly.
#[test]
fn all_f32_tf_pairs_roundtrip() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ];

    let test_pixels: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.8, 0.6, 0.4],
        [1.0, 1.0, 1.0],
    ];

    for &from_tf in &tfs {
        for &to_tf in &tfs {
            if from_tf == to_tf {
                continue;
            }
            let mut fwd = f32_converter(from_tf, to_tf);
            let mut rev = f32_converter(to_tf, from_tf);

            for &[r, g, b] in test_pixels {
                let mid = convert_f32_pixel(&mut fwd, r, g, b);
                let back = convert_f32_pixel(&mut rev, mid[0], mid[1], mid[2]);
                // PQ has lower precision due to rational polynomial approximation
                let tol = if matches!(from_tf, TransferFunction::Pq)
                    || matches!(to_tf, TransferFunction::Pq)
                {
                    5e-3
                } else {
                    1e-4
                };
                for (ch, (orig, rt)) in [r, g, b].iter().zip(back.iter()).enumerate() {
                    assert!(
                        (orig - rt).abs() < tol,
                        "{from_tf:?}→{to_tf:?}→{from_tf:?} ch{ch} [{r},{g},{b}]: \
                         {orig} → {rt}, delta={:.6}",
                        (orig - rt).abs()
                    );
                }
            }
        }
    }
}

/// Converting through linear gives the same result as a direct cross-TF path.
#[test]
fn cross_tf_via_linear_matches_direct() {
    let sdrs = [TransferFunction::Srgb, TransferFunction::Bt709];
    let hdrs = [TransferFunction::Pq, TransferFunction::Hlg];

    let test_pixels: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [0.2, 0.4, 0.6],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ];

    for &src_tf in sdrs.iter().chain(hdrs.iter()) {
        for &dst_tf in sdrs.iter().chain(hdrs.iter()) {
            if src_tf == dst_tf {
                continue;
            }
            let mut direct = f32_converter(src_tf, dst_tf);
            let mut to_lin = f32_converter(src_tf, TransferFunction::Linear);
            let mut from_lin = f32_converter(TransferFunction::Linear, dst_tf);

            for &[r, g, b] in test_pixels {
                let direct_result = convert_f32_pixel(&mut direct, r, g, b);
                let via_lin = {
                    let lin = convert_f32_pixel(&mut to_lin, r, g, b);
                    convert_f32_pixel(&mut from_lin, lin[0], lin[1], lin[2])
                };

                let tol = if matches!(src_tf, TransferFunction::Pq)
                    || matches!(dst_tf, TransferFunction::Pq)
                {
                    5e-3
                } else {
                    1e-4
                };
                for ch in 0..3 {
                    assert!(
                        (direct_result[ch] - via_lin[ch]).abs() < tol,
                        "{src_tf:?}→{dst_tf:?} ch{ch} [{r},{g},{b}]: \
                         direct={:.6} vs via_linear={:.6}",
                        direct_result[ch],
                        via_lin[ch]
                    );
                }
            }
        }
    }
}

/// Black is preserved through every cross-TF conversion.
#[test]
fn black_preserved_all_tf_pairs() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ];
    for &from_tf in &tfs {
        for &to_tf in &tfs {
            if from_tf == to_tf {
                continue;
            }
            let mut conv = f32_converter(from_tf, to_tf);
            let result = convert_f32_pixel(&mut conv, 0.0, 0.0, 0.0);
            for (ch, &v) in result.iter().enumerate() {
                assert!(v.abs() < 1e-6, "{from_tf:?}→{to_tf:?} black ch{ch}: {v}");
            }
        }
    }
}

/// White is preserved through every cross-TF conversion.
#[test]
fn white_preserved_all_tf_pairs() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ];
    for &from_tf in &tfs {
        for &to_tf in &tfs {
            if from_tf == to_tf {
                continue;
            }
            let mut conv = f32_converter(from_tf, to_tf);
            let result = convert_f32_pixel(&mut conv, 1.0, 1.0, 1.0);
            for (ch, &v) in result.iter().enumerate() {
                assert!(
                    (v - 1.0).abs() < 1e-3,
                    "{from_tf:?}→{to_tf:?} white ch{ch}: {v}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 6: sRGB/BT.709 F32↔Linear Kernel Accuracy
// ═══════════════════════════════════════════════════════════════════════════

/// sRGB F32 → Linear F32 known values.
#[test]
fn srgb_f32_to_linear_f32_known_values() {
    let mut conv = f32_converter(TransferFunction::Srgb, TransferFunction::Linear);

    // sRGB 0.5 → linear ~0.214
    let mid = convert_f32_pixel(&mut conv, 0.5, 0.5, 0.5);
    assert!(
        (mid[0] - 0.214).abs() < 0.003,
        "sRGB(0.5)→linear: got {}, expected ~0.214",
        mid[0]
    );

    // sRGB 0.0 → linear 0.0
    let black = convert_f32_pixel(&mut conv, 0.0, 0.0, 0.0);
    assert!(black[0].abs() < 1e-6);

    // sRGB 1.0 → linear 1.0
    let white = convert_f32_pixel(&mut conv, 1.0, 1.0, 1.0);
    assert!((white[0] - 1.0).abs() < 1e-5);
}

/// Linear F32 → sRGB F32 known values.
#[test]
fn linear_f32_to_srgb_f32_known_values() {
    let mut conv = f32_converter(TransferFunction::Linear, TransferFunction::Srgb);

    // linear 0.214 → sRGB ~0.5
    let mid = convert_f32_pixel(&mut conv, 0.214, 0.214, 0.214);
    assert!(
        (mid[0] - 0.5).abs() < 0.01,
        "linear(0.214)→sRGB: got {}, expected ~0.5",
        mid[0]
    );
}

/// BT.709 F32 → Linear F32 known values.
#[test]
fn bt709_f32_to_linear_f32_known_values() {
    let mut conv = f32_converter(TransferFunction::Bt709, TransferFunction::Linear);
    let mid = convert_f32_pixel(&mut conv, 0.5, 0.5, 0.5);
    // BT.709 EOTF at 0.5 → ~0.26
    assert!(
        mid[0] > 0.22 && mid[0] < 0.30,
        "BT.709(0.5)→linear: got {}, expected ~0.26",
        mid[0]
    );
}

/// Linear F32 → BT.709 F32 known values.
#[test]
fn linear_f32_to_bt709_f32_known_values() {
    let mut conv = f32_converter(TransferFunction::Linear, TransferFunction::Bt709);
    // BT.709 OETF(0.26) ≈ 0.5
    let mid = convert_f32_pixel(&mut conv, 0.26, 0.26, 0.26);
    assert!(
        mid[0] > 0.45 && mid[0] < 0.55,
        "linear(0.26)→BT.709: got {}, expected ~0.5",
        mid[0]
    );
}

/// sRGB F32 → Linear F32 roundtrip for all 256 "u8-equivalent" values.
#[test]
fn srgb_f32_linear_exhaustive_roundtrip() {
    let mut to_lin = f32_converter(TransferFunction::Srgb, TransferFunction::Linear);
    let mut to_srgb = f32_converter(TransferFunction::Linear, TransferFunction::Srgb);

    for i in 0..=255u8 {
        let v = i as f32 / 255.0;
        let lin = convert_f32_pixel(&mut to_lin, v, v, v);
        let back = convert_f32_pixel(&mut to_srgb, lin[0], lin[1], lin[2]);
        assert!(
            (back[0] - v).abs() < 1e-5,
            "sRGB F32 roundtrip at {i}/255 (v={v}): got {}, delta={}",
            back[0],
            (back[0] - v).abs()
        );
    }
}

/// BT.709 F32 → Linear F32 roundtrip for all 256 "u8-equivalent" values.
#[test]
fn bt709_f32_linear_exhaustive_roundtrip() {
    let mut to_lin = f32_converter(TransferFunction::Bt709, TransferFunction::Linear);
    let mut to_bt709 = f32_converter(TransferFunction::Linear, TransferFunction::Bt709);

    for i in 0..=255u8 {
        let v = i as f32 / 255.0;
        let lin = convert_f32_pixel(&mut to_lin, v, v, v);
        let back = convert_f32_pixel(&mut to_bt709, lin[0], lin[1], lin[2]);
        assert!(
            (back[0] - v).abs() < 1e-5,
            "BT.709 F32 roundtrip at {i}/255 (v={v}): got {}, delta={}",
            back[0],
            (back[0] - v).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 7: Buffer-Level Linearize/Delinearize
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: create a 4-pixel RGB8 buffer with known values.
fn make_test_rgb8_buffer(tf: TransferFunction) -> PixelBuffer {
    let data = vec![
        0, 0, 0, // black
        128, 128, 128, // mid-gray
        64, 128, 192, // mixed
        255, 255, 255, // white
    ];
    let desc = PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgb, None, tf);
    PixelBuffer::from_vec(data, 4, 1, desc).unwrap()
}

/// Helper: create a 4-pixel RGB F32 buffer with known linear values.
fn make_test_rgbf32_buffer(tf: TransferFunction, primaries: ColorPrimaries) -> PixelBuffer {
    let values: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.214, 0.214, 0.214, 0.05, 0.214, 0.5, 1.0, 1.0, 1.0,
    ];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let desc = PixelDescriptor::new_full(ChannelType::F32, ChannelLayout::Rgb, None, tf, primaries);
    PixelBuffer::from_vec(data, 4, 1, desc).unwrap()
}

/// Helper: read F32 value from a PixelBuffer row at byte offset.
fn read_f32(buf: &PixelBuffer, byte_offset: usize) -> f32 {
    let slice = buf.as_slice();
    let row = slice.row(0);
    f32::from_le_bytes([
        row[byte_offset],
        row[byte_offset + 1],
        row[byte_offset + 2],
        row[byte_offset + 3],
    ])
}

/// linearize() works for sRGB u8 input.
#[test]
fn buffer_linearize_srgb_u8() {
    let buf = make_test_rgb8_buffer(TransferFunction::Srgb);
    let lin = buf.linearize().unwrap();

    assert_eq!(lin.descriptor().transfer(), TransferFunction::Linear);
    assert_eq!(lin.descriptor().channel_type(), ChannelType::F32);

    // Black → 0
    assert!(read_f32(&lin, 0).abs() < 1e-6);
    // Mid-gray (128/255 sRGB) → ~0.216 linear
    assert!((read_f32(&lin, 12) - 0.216).abs() < 0.01);
    // White → 1
    assert!((read_f32(&lin, 36) - 1.0).abs() < 1e-5);
}

/// linearize() works for BT.709 u8 input.
#[test]
fn buffer_linearize_bt709_u8() {
    let buf = make_test_rgb8_buffer(TransferFunction::Bt709);
    let lin = buf.linearize().unwrap();

    assert_eq!(lin.descriptor().transfer(), TransferFunction::Linear);
    // Black → 0
    assert!(read_f32(&lin, 0).abs() < 1e-6);
    // Mid-gray → ~0.21
    let mid = read_f32(&lin, 12);
    assert!(
        mid > 0.18 && mid < 0.25,
        "BT.709 mid-gray linearized: {mid}"
    );
    // White → 1
    assert!((read_f32(&lin, 36) - 1.0).abs() < 1e-5);
}

/// linearize() works for PQ F32 input.
#[test]
fn buffer_linearize_pq_f32() {
    let buf = make_test_rgbf32_buffer(TransferFunction::Pq, ColorPrimaries::Bt2020);
    let lin = buf.linearize().unwrap();

    assert_eq!(lin.descriptor().transfer(), TransferFunction::Linear);
    assert_eq!(lin.descriptor().primaries, ColorPrimaries::Bt2020);
    // PQ black → 0
    assert!(read_f32(&lin, 0).abs() < 1e-6);
    // PQ white → 1
    assert!((read_f32(&lin, 36) - 1.0).abs() < 1e-3);
}

/// linearize() works for HLG F32 input.
#[test]
fn buffer_linearize_hlg_f32() {
    let buf = make_test_rgbf32_buffer(TransferFunction::Hlg, ColorPrimaries::Bt2020);
    let lin = buf.linearize().unwrap();

    assert_eq!(lin.descriptor().transfer(), TransferFunction::Linear);
    assert_eq!(lin.descriptor().primaries, ColorPrimaries::Bt2020);
    // HLG black → 0
    assert!(read_f32(&lin, 0).abs() < 1e-6);
}

/// delinearize() applies sRGB OETF to linear F32 buffer.
#[test]
fn buffer_delinearize_srgb() {
    let buf = make_test_rgbf32_buffer(TransferFunction::Linear, ColorPrimaries::Bt709);
    let srgb = buf.delinearize(TransferFunction::Srgb).unwrap();

    assert_eq!(srgb.descriptor().transfer(), TransferFunction::Srgb);
    // 0 → 0
    assert!(read_f32(&srgb, 0).abs() < 1e-6);
    // 0.214 → ~0.5
    let mid = read_f32(&srgb, 12);
    assert!((mid - 0.5).abs() < 0.01, "linear(0.214)→sRGB: {mid}");
    // 1 → 1
    assert!((read_f32(&srgb, 36) - 1.0).abs() < 1e-5);
}

/// delinearize() applies PQ OETF to linear F32 buffer.
#[test]
fn buffer_delinearize_pq() {
    let buf = make_test_rgbf32_buffer(TransferFunction::Linear, ColorPrimaries::Bt2020);
    let pq = buf.delinearize(TransferFunction::Pq).unwrap();

    assert_eq!(pq.descriptor().transfer(), TransferFunction::Pq);
    assert_eq!(pq.descriptor().primaries, ColorPrimaries::Bt2020);
    // 0 → 0
    assert!(read_f32(&pq, 0).abs() < 1e-6);
    // 1 → ~1 (PQ(1.0) = 10000 cd/m² signal = 1.0)
    assert!((read_f32(&pq, 36) - 1.0).abs() < 1e-3);
}

/// delinearize() applies HLG OETF to linear F32 buffer.
#[test]
fn buffer_delinearize_hlg() {
    let buf = make_test_rgbf32_buffer(TransferFunction::Linear, ColorPrimaries::Bt2020);
    let hlg = buf.delinearize(TransferFunction::Hlg).unwrap();

    assert_eq!(hlg.descriptor().transfer(), TransferFunction::Hlg);
    assert_eq!(hlg.descriptor().primaries, ColorPrimaries::Bt2020);
    // 0 → 0
    assert!(read_f32(&hlg, 0).abs() < 1e-6);
}

/// linearize → delinearize roundtrip preserves values for all TFs.
#[test]
fn buffer_linearize_delinearize_roundtrip_all_tfs() {
    let tfs = [
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ];

    for tf in tfs {
        let buf = make_test_rgbf32_buffer(tf, ColorPrimaries::Bt709);
        let lin = buf.linearize().unwrap();
        let back = lin.delinearize(tf).unwrap();

        assert_eq!(
            back.descriptor().transfer(),
            tf,
            "{tf:?}: descriptor transfer after roundtrip"
        );

        // Compare pixel 1 (mid-gray: 0.214)
        let orig = read_f32(&buf, 12);
        let rt = read_f32(&back, 12);
        let tol = if matches!(tf, TransferFunction::Pq) {
            5e-3
        } else {
            1e-4
        };
        assert!(
            (orig - rt).abs() < tol,
            "{tf:?} buffer roundtrip pixel 1: {orig} → {rt}, delta={}",
            (orig - rt).abs()
        );
    }
}

/// linearize preserves color context.
#[test]
fn buffer_linearize_preserves_color_context() {
    use zenpixels::ColorContext;
    let data = vec![128u8; 12];
    let mut buf = PixelBuffer::from_vec(data, 4, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let ctx = ColorContext::from_cicp(Cicp::SRGB);
    buf = buf.with_color_context(std::sync::Arc::new(ctx));

    let lin = buf.linearize().unwrap();
    assert!(
        lin.color_context().is_some(),
        "linearize should preserve color context"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 8: Cross-Depth TF Conversion Paths
// ═══════════════════════════════════════════════════════════════════════════

/// sRGB U8 → Linear F32 → sRGB U8 roundtrip: max drift ≤ 1.
#[test]
fn srgb_u8_f32_u8_roundtrip_all_values() {
    let srgb_u8 = PixelDescriptor::RGB8_SRGB;
    let linear_f32 = PixelDescriptor::RGBF32_LINEAR;

    let mut to_linear = RowConverter::new(srgb_u8, linear_f32).unwrap();
    let mut to_srgb = RowConverter::new(linear_f32, srgb_u8).unwrap();

    let width = 256u32;
    let mut src = Vec::with_capacity(256 * 3);
    for i in 0..256u16 {
        src.push(i as u8);
        src.push(i as u8);
        src.push(i as u8);
    }

    let mut f32_buf = vec![0u8; 256 * 3 * 4];
    let mut back = vec![0u8; 256 * 3];

    to_linear.convert_row(&src, &mut f32_buf, width);
    to_srgb.convert_row(&f32_buf, &mut back, width);

    let mut max_drift = 0;
    let mut drift_count = 0;
    for i in 0..256 {
        let drift = (src[i * 3] as i32 - back[i * 3] as i32).abs();
        max_drift = max_drift.max(drift);
        if drift > 0 {
            drift_count += 1;
        }
    }
    assert!(
        max_drift <= 1,
        "sRGB U8 roundtrip max drift: {max_drift} ({drift_count} values drifted)"
    );
}

/// PQ U16 → Linear F32 → PQ U16 roundtrip: max drift ≤ 1.
#[test]
fn pq_u16_f32_u16_roundtrip_wide_range() {
    let pq_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let linear_f32 = PixelDescriptor::RGBF32_LINEAR;

    let mut to_linear = RowConverter::new(pq_u16, linear_f32).unwrap();
    let mut to_pq = RowConverter::new(linear_f32, pq_u16).unwrap();

    // Test 256 evenly-spaced U16 values.
    let width = 256u32;
    let mut src = vec![0u8; 256 * 3 * 2];
    for i in 0..256usize {
        let v = (i * 256) as u16; // 0, 256, 512, ..., 65280
        for ch in 0..3 {
            let base = (i * 3 + ch) * 2;
            src[base..base + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }

    let mut f32_buf = vec![0u8; 256 * 3 * 4];
    let mut back = vec![0u8; 256 * 3 * 2];

    to_linear.convert_row(&src, &mut f32_buf, width);
    to_pq.convert_row(&f32_buf, &mut back, width);

    let mut max_drift = 0u32;
    for i in 0..256 * 3 {
        let orig = u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]);
        let result = u16::from_ne_bytes([back[i * 2], back[i * 2 + 1]]);
        let drift = (orig as i32 - result as i32).unsigned_abs();
        max_drift = max_drift.max(drift);
    }
    assert!(max_drift <= 1, "PQ U16 roundtrip max drift: {max_drift}");
}

/// HLG U16 → Linear F32 → HLG U16 roundtrip: max drift ≤ 1.
#[test]
fn hlg_u16_f32_u16_roundtrip_wide_range() {
    let hlg_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    );
    let linear_f32 = PixelDescriptor::RGBF32_LINEAR;

    let mut to_linear = RowConverter::new(hlg_u16, linear_f32).unwrap();
    let mut to_hlg = RowConverter::new(linear_f32, hlg_u16).unwrap();

    let width = 256u32;
    let mut src = vec![0u8; 256 * 3 * 2];
    for i in 0..256usize {
        let v = (i * 256) as u16;
        for ch in 0..3 {
            let base = (i * 3 + ch) * 2;
            src[base..base + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }

    let mut f32_buf = vec![0u8; 256 * 3 * 4];
    let mut back = vec![0u8; 256 * 3 * 2];

    to_linear.convert_row(&src, &mut f32_buf, width);
    to_hlg.convert_row(&f32_buf, &mut back, width);

    let mut max_drift = 0u32;
    for i in 0..256 * 3 {
        let orig = u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]);
        let result = u16::from_ne_bytes([back[i * 2], back[i * 2 + 1]]);
        let drift = (orig as i32 - result as i32).unsigned_abs();
        max_drift = max_drift.max(drift);
    }
    assert!(max_drift <= 1, "HLG U16 roundtrip max drift: {max_drift}");
}

/// PQ U16 → sRGB U8: HDR to SDR path produces reasonable output.
#[test]
fn pq_u16_to_srgb_u8_correctness() {
    let pq_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let srgb_u8 = PixelDescriptor::RGB8_SRGB;
    let mut conv = RowConverter::new(pq_u16, srgb_u8).unwrap();

    // Build test data: 5 pixels at different PQ levels.
    let width = 5u32;
    let pq_values: [u16; 5] = [0, 16384, 32768, 49152, 65535];
    let mut src = vec![0u8; 5 * 3 * 2];
    for (i, &v) in pq_values.iter().enumerate() {
        for ch in 0..3 {
            let base = (i * 3 + ch) * 2;
            src[base..base + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }

    let mut dst = vec![0u8; 5 * 3];
    conv.convert_row(&src, &mut dst, width);

    // Black stays black
    assert_eq!(dst[0], 0);

    // Values should be monotonically increasing
    for i in 1..5 {
        assert!(
            dst[i * 3] >= dst[(i - 1) * 3],
            "PQ→sRGB not monotonic: pixel {i} ({}) < pixel {} ({})",
            dst[i * 3],
            i - 1,
            dst[(i - 1) * 3]
        );
    }

    // Full PQ → sRGB white
    assert_eq!(dst[4 * 3], 255);
}

/// HLG U16 → sRGB U8 path produces reasonable output.
#[test]
fn hlg_u16_to_srgb_u8_correctness() {
    let hlg_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    );
    let srgb_u8 = PixelDescriptor::RGB8_SRGB;
    let mut conv = RowConverter::new(hlg_u16, srgb_u8).unwrap();

    let width = 3u32;
    let hlg_values: [u16; 3] = [0, 32768, 65535];
    let mut src = vec![0u8; 3 * 3 * 2];
    for (i, &v) in hlg_values.iter().enumerate() {
        for ch in 0..3 {
            let base = (i * 3 + ch) * 2;
            src[base..base + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }

    let mut dst = vec![0u8; 3 * 3];
    conv.convert_row(&src, &mut dst, width);

    // Black stays black
    assert_eq!(dst[0], 0);
    // Monotonically increasing
    assert!(dst[3] > dst[0]);
    assert!(dst[6] > dst[3]);
    // Full HLG → sRGB white
    assert_eq!(dst[6], 255);
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 9: Conversion Path Consistency
// ═══════════════════════════════════════════════════════════════════════════

/// sRGB U8 → Linear F32 via RowConverter matches scalar TransferFunctionExt.
#[test]
fn row_converter_matches_scalar_eotf() {
    let srgb_u8 = PixelDescriptor::RGB8_SRGB;
    let linear_f32 = PixelDescriptor::RGBF32_LINEAR;
    let mut conv = RowConverter::new(srgb_u8, linear_f32).unwrap();

    for i in 0..=255u8 {
        let v = i as f32 / 255.0;
        let expected = TransferFunction::Srgb.linearize(v);

        let src = [i, i, i];
        let mut dst = vec![0u8; 12];
        conv.convert_row(&src, &mut dst, 1);
        let result: &[f32] = bytemuck::cast_slice(&dst);

        // Allow tolerance: the U8→F32 kernel may use LUT or SIMD approximation.
        assert!(
            (result[0] - expected).abs() < 0.002,
            "sRGB U8({i}) → F32: converter={:.6}, scalar={:.6}",
            result[0],
            expected
        );
    }
}

/// sRGB F32 → Linear F32 via RowConverter matches scalar TransferFunctionExt.
#[test]
fn f32_converter_matches_scalar_srgb() {
    let mut conv = f32_converter(TransferFunction::Srgb, TransferFunction::Linear);

    for i in 0..=100u32 {
        let v = i as f32 / 100.0;
        let expected = TransferFunction::Srgb.linearize(v);
        let result = convert_f32_pixel(&mut conv, v, v, v);

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "sRGB F32({v}): converter={:.8}, scalar={:.8}",
            result[0],
            expected
        );
    }
}

/// BT.709 F32 → Linear F32 via RowConverter matches scalar TransferFunctionExt.
#[test]
fn f32_converter_matches_scalar_bt709() {
    let mut conv = f32_converter(TransferFunction::Bt709, TransferFunction::Linear);

    for i in 0..=100u32 {
        let v = i as f32 / 100.0;
        let expected = TransferFunction::Bt709.linearize(v);
        let result = convert_f32_pixel(&mut conv, v, v, v);

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "BT.709 F32({v}): converter={:.8}, scalar={:.8}",
            result[0],
            expected
        );
    }
}

/// PQ F32 → Linear F32 via RowConverter matches scalar TransferFunctionExt.
#[test]
fn f32_converter_matches_scalar_pq() {
    let mut conv = f32_converter(TransferFunction::Pq, TransferFunction::Linear);

    for i in 0..=100u32 {
        let v = i as f32 / 100.0;
        let expected = TransferFunction::Pq.linearize(v);
        let result = convert_f32_pixel(&mut conv, v, v, v);

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "PQ F32({v}): converter={:.8}, scalar={:.8}",
            result[0],
            expected
        );
    }
}

/// HLG F32 → Linear F32 via RowConverter matches scalar TransferFunctionExt.
#[test]
fn f32_converter_matches_scalar_hlg() {
    let mut conv = f32_converter(TransferFunction::Hlg, TransferFunction::Linear);

    for i in 0..=100u32 {
        let v = i as f32 / 100.0;
        let expected = TransferFunction::Hlg.linearize(v);
        let result = convert_f32_pixel(&mut conv, v, v, v);

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "HLG F32({v}): converter={:.8}, scalar={:.8}",
            result[0],
            expected
        );
    }
}
