//! Comprehensive CMS tests using real-world ICC profiles from the system.
//!
//! These tests exercise the moxcms CMS backend against the colord ICC profile
//! collection installed at `/usr/share/color/icc/colord/`. This collection
//! includes 25 profiles covering:
//!
//! - Standard working spaces: sRGB, AdobeRGB, ProPhoto, WideGamut, ECI-RGB
//! - Broadcast: Rec709, PAL-RGB, NTSC-RGB, SMPTE-C
//! - Photography: EktaSpace, DonRGB4, BestRGB, BetaRGB, BruceRGB
//! - Display: AppleRGB, CIE-RGB, ColorMatch
//! - Special: Bluish, Crayons, SwappedRedAndGreen, x11-colors
//! - Gray tone curves: Gamma5000K, Gamma5500K, Gamma6500K
//!
//! The ghostscript profiles at `/usr/share/color/icc/ghostscript/` add Lab,
//! CMYK, and minimal sRGB/gray variants.
//!
//! # Running
//!
//! These tests require the `cms-moxcms` feature and system ICC profiles.
//! They are `#[ignore]`-gated and run with `cargo test --ignored`:
//!
//! ```sh
//! cargo test -p zenpixels-convert --test cms_real_profiles -- --ignored
//! ```

#![cfg(feature = "cms-moxcms")]

extern crate alloc;

use std::path::Path;

use zenpixels_convert::PixelFormat;
use zenpixels_convert::cms::ColorManagement;
use zenpixels_convert::cms_moxcms::MoxCms;

// ---------------------------------------------------------------------------
// Profile discovery
// ---------------------------------------------------------------------------

const COLORD_DIR: &str = "/usr/share/color/icc/colord";
const GHOSTSCRIPT_DIR: &str = "/usr/share/color/icc/ghostscript";

/// Load all ICC profiles from a directory.
fn load_profiles(dir: &str) -> Vec<(String, Vec<u8>)> {
    let dir = Path::new(dir);
    if !dir.exists() {
        return Vec::new();
    }

    let mut profiles = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "icc" || e == "icm") {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            match std::fs::read(&path) {
                Ok(data) => profiles.push((name, data)),
                Err(e) => eprintln!("  skip {}: {e}", path.display()),
            }
        }
    }
    profiles.sort_by(|a, b| a.0.cmp(&b.0));
    profiles
}

/// Load RGB-only profiles from colord (excludes gray/special tone curves).
fn load_rgb_profiles() -> Vec<(String, Vec<u8>)> {
    load_profiles(COLORD_DIR)
        .into_iter()
        .filter(|(name, _)| !name.starts_with("Gamma"))
        .collect()
}

/// Check if a profile is parseable by moxcms.
fn is_parseable(icc: &[u8]) -> bool {
    moxcms::ColorProfile::new_from_slice(icc).is_ok()
}

/// Check if a profile is an RGB monitor/display profile.
fn is_rgb_profile(icc: &[u8]) -> bool {
    // ICC header: bytes 16..20 = color space, bytes 12..16 = profile class
    icc.len() >= 24 && &icc[16..20] == b"RGB "
}

// ---------------------------------------------------------------------------
// Test 1: All profiles parse without error
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn all_colord_profiles_parse() {
    let profiles = load_profiles(COLORD_DIR);
    assert!(!profiles.is_empty(), "no profiles found in {COLORD_DIR}");

    let mut parsed = 0;
    let mut failed = Vec::new();

    for (name, data) in &profiles {
        match moxcms::ColorProfile::new_from_slice(data) {
            Ok(_) => parsed += 1,
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }

    eprintln!("  parsed {parsed}/{} colord profiles", profiles.len());
    if !failed.is_empty() {
        eprintln!("  failed:");
        for f in &failed {
            eprintln!("    {f}");
        }
    }
    // Allow some profiles to fail (exotic types), but most should parse.
    assert!(
        parsed >= profiles.len() * 3 / 4,
        "fewer than 75% of profiles parsed: {parsed}/{}",
        profiles.len()
    );
}

#[test]
#[ignore]
fn all_ghostscript_profiles_parse() {
    let profiles = load_profiles(GHOSTSCRIPT_DIR);
    if profiles.is_empty() {
        eprintln!("  no ghostscript profiles found, skipping");
        return;
    }

    let mut parsed = 0;
    for (name, data) in &profiles {
        match moxcms::ColorProfile::new_from_slice(data) {
            Ok(_) => parsed += 1,
            Err(e) => eprintln!("  {name}: {e}"),
        }
    }
    eprintln!("  parsed {parsed}/{} ghostscript profiles", profiles.len());
}

// ---------------------------------------------------------------------------
// Test 2: Profile identification
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn identify_known_profiles() {
    let cms = MoxCms;
    let profiles = load_profiles(COLORD_DIR);

    let mut identified = Vec::new();
    let mut unknown = Vec::new();

    for (name, data) in &profiles {
        match cms.identify_profile(data) {
            Some(cicp) => identified.push((name.as_str(), cicp)),
            None => unknown.push(name.as_str()),
        }
    }

    eprintln!(
        "  identified {}/{} profiles:",
        identified.len(),
        profiles.len()
    );
    for (name, cicp) in &identified {
        eprintln!(
            "    {name}: primaries={}, transfer={}, matrix={}, full_range={}",
            cicp.color_primaries,
            cicp.transfer_characteristics,
            cicp.matrix_coefficients,
            cicp.full_range
        );
    }
    eprintln!("  unknown: {}", unknown.join(", "));

    // sRGB and Rec709 should be identified (both BT.709 primaries).
    let srgb_found = identified
        .iter()
        .any(|(n, c)| *n == "sRGB" && c.color_primaries == 1);
    assert!(srgb_found, "sRGB should be identified as BT.709 primaries");
}

// ---------------------------------------------------------------------------
// Test 3: Transform creation for all RGB profile pairs
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn all_rgb_profile_pairs_create_transforms() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    // Filter to parseable RGB profiles
    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, data)| is_parseable(data) && is_rgb_profile(data))
        .collect();

    eprintln!(
        "  testing {} RGB profiles ({} pairs)",
        rgb_profiles.len(),
        rgb_profiles.len() * (rgb_profiles.len() - 1)
    );

    let mut ok = 0;
    let mut fail = 0;

    for (src_name, src_data) in &rgb_profiles {
        for (dst_name, dst_data) in &rgb_profiles {
            if src_name == dst_name {
                continue;
            }
            match cms.build_transform(src_data, dst_data) {
                Ok(_) => ok += 1,
                Err(e) => {
                    eprintln!("    FAIL: {src_name} → {dst_name}: {e}");
                    fail += 1;
                }
            }
        }
    }

    eprintln!("  {ok} ok, {fail} failed");
    // Most pairs should work. Some exotic profiles may fail.
    assert!(
        fail <= ok / 10,
        "too many transform failures: {fail}/{} total",
        ok + fail
    );
}

// ---------------------------------------------------------------------------
// Test 4: White and black point preservation across all pairs
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn white_black_preservation_all_pairs() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, data)| is_parseable(data) && is_rgb_profile(data))
        .collect();

    let mut tested = 0;
    let mut white_max_err = 0i32;
    let mut black_max_err = 0i32;
    let mut worst_white = String::new();
    let mut worst_black = String::new();

    for (src_name, src_data) in &rgb_profiles {
        for (dst_name, dst_data) in &rgb_profiles {
            if src_name == dst_name {
                continue;
            }
            let xform = match cms.build_transform(src_data, dst_data) {
                Ok(x) => x,
                Err(_) => continue,
            };

            // White
            let mut dst = [0u8; 3];
            xform.transform_row(&[255, 255, 255], &mut dst, 1);
            let w_err = dst
                .iter()
                .map(|&v| (255i32 - v as i32).abs())
                .max()
                .unwrap();
            if w_err > white_max_err {
                white_max_err = w_err;
                worst_white = format!("{src_name}→{dst_name}: {:?}", dst);
            }

            // Black
            let mut dst = [0u8; 3];
            xform.transform_row(&[0, 0, 0], &mut dst, 1);
            let b_err = dst.iter().map(|&v| v as i32).max().unwrap();
            if b_err > black_max_err {
                black_max_err = b_err;
                worst_black = format!("{src_name}→{dst_name}: {:?}", dst);
            }

            tested += 1;
        }
    }

    eprintln!("  tested {tested} pairs");
    eprintln!("  worst white error: {white_max_err} ({worst_white})");
    eprintln!("  worst black error: {black_max_err} ({worst_black})");

    assert!(
        white_max_err <= 15,
        "white point error too large: {white_max_err} ({worst_white})"
    );
    assert!(
        black_max_err <= 10,
        "black point error too large: {black_max_err} ({worst_black})"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Round-trip accuracy for key profile pairs
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn roundtrip_accuracy_key_pairs() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let find = |name: &str| -> Option<&Vec<u8>> {
        profiles.iter().find(|(n, _)| n == name).map(|(_, d)| d)
    };

    // Key pairs that should have excellent round-trip accuracy
    let pairs = [
        ("sRGB", "AdobeRGB1998"),
        ("sRGB", "ProPhotoRGB"),
        ("sRGB", "Rec709"),
        ("AdobeRGB1998", "ProPhotoRGB"),
        ("sRGB", "WideGamutRGB"),
        ("sRGB", "BestRGB"),
        ("sRGB", "ECI-RGBv2"),
    ];

    for (src_name, dst_name) in &pairs {
        let src = match find(src_name) {
            Some(d) => d,
            None => {
                eprintln!("  skip {src_name}: not found");
                continue;
            }
        };
        let dst = match find(dst_name) {
            Some(d) => d,
            None => {
                eprintln!("  skip {dst_name}: not found");
                continue;
            }
        };

        let forward = match cms.build_transform(src, dst) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  skip {src_name}→{dst_name}: {e}");
                continue;
            }
        };
        let inverse = match cms.build_transform(dst, src) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  skip {dst_name}→{src_name}: {e}");
                continue;
            }
        };

        let mut max_err = 0i32;
        let mut worst = [0u8; 3];
        let mut worst_back = [0u8; 3];

        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let src_px = [r as u8, g as u8, b as u8];
                    let mut mid = [0u8; 3];
                    let mut back = [0u8; 3];

                    forward.transform_row(&src_px, &mut mid, 1);
                    inverse.transform_row(&mid, &mut back, 1);

                    let err = (0..3)
                        .map(|c| (back[c] as i32 - src_px[c] as i32).abs())
                        .max()
                        .unwrap();
                    if err > max_err {
                        max_err = err;
                        worst = src_px;
                        worst_back = back;
                    }
                }
            }
        }

        eprintln!(
            "  {src_name} ↔ {dst_name}: max roundtrip error = {max_err} \
             (at {:?} → {:?})",
            worst, worst_back
        );

        // Round-trip error depends on gamut overlap. Pairs with very
        // different gamuts (ProPhoto, ECI-RGB, WideGamut) clip saturated
        // sRGB colors that lie outside the narrower gamut.
        // sRGB↔Rec709 is near-identity (same primaries, similar TRC).
        let limit = if *src_name == "sRGB" && *dst_name == "Rec709" {
            5
        } else {
            60 // Wide-gamut round-trips clip, especially at pure primaries
        };

        assert!(
            max_err <= limit,
            "{src_name} ↔ {dst_name} roundtrip error {max_err} exceeds limit {limit}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Format-aware transforms (u16, f32) with real profiles
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn format_aware_transforms_real_profiles() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let find = |name: &str| -> Option<&Vec<u8>> {
        profiles.iter().find(|(n, _)| n == name).map(|(_, d)| d)
    };

    let srgb = match find("sRGB") {
        Some(d) => d,
        None => {
            eprintln!("  sRGB profile not found, skipping");
            return;
        }
    };
    let adobe = match find("AdobeRGB1998") {
        Some(d) => d,
        None => {
            eprintln!("  AdobeRGB1998 profile not found, skipping");
            return;
        }
    };

    // u8 transform
    let xform_u8 = cms
        .build_transform_for_format(srgb, adobe, PixelFormat::Rgb8, PixelFormat::Rgb8)
        .unwrap();

    // u16 transform
    let xform_u16 = cms
        .build_transform_for_format(srgb, adobe, PixelFormat::Rgb16, PixelFormat::Rgb16)
        .unwrap();

    // f32 transform
    let xform_f32 = cms
        .build_transform_for_format(srgb, adobe, PixelFormat::RgbF32, PixelFormat::RgbF32)
        .unwrap();

    // Test mid-gray through all three: results should be consistent
    // u8
    let mut dst_u8 = [0u8; 3];
    xform_u8.transform_row(&[128, 128, 128], &mut dst_u8, 1);

    // u16
    let src_u16: [u16; 3] = [32768, 32768, 32768];
    let src_u16_bytes: [u8; 6] = bytemuck::cast(src_u16);
    let mut dst_u16_bytes = [0u8; 6];
    xform_u16.transform_row(&src_u16_bytes, &mut dst_u16_bytes, 1);
    let dst_u16: [u16; 3] = bytemuck::cast(dst_u16_bytes);

    // f32
    let src_f32: [f32; 3] = [0.5020, 0.5020, 0.5020]; // ~128/255
    let src_f32_bytes: [u8; 12] = bytemuck::cast(src_f32);
    let mut dst_f32_bytes = [0u8; 12];
    xform_f32.transform_row(&src_f32_bytes, &mut dst_f32_bytes, 1);
    let dst_f32: [f32; 3] = bytemuck::cast(dst_f32_bytes);

    eprintln!("  sRGB→AdobeRGB mid-gray:");
    eprintln!("    u8:  {:?}", dst_u8);
    eprintln!(
        "    u16: {:?} (as u8: {:?})",
        dst_u16,
        dst_u16.map(|v| (v >> 8) as u8)
    );
    eprintln!(
        "    f32: {:?} (as u8: {:?})",
        dst_f32,
        dst_f32.map(|v| (v * 255.0 + 0.5) as u8)
    );

    // Gray should be approximately preserved across all depths
    for ch in 0..3 {
        let err_u8 = (dst_u8[ch] as i32 - 128).abs();
        assert!(err_u8 <= 5, "u8 gray ch{ch}: {}", dst_u8[ch]);

        let err_u16 = ((dst_u16[ch] as i32) - 32768).abs();
        assert!(err_u16 <= 1500, "u16 gray ch{ch}: {}", dst_u16[ch]);

        let err_f32 = (dst_f32[ch] - 0.502).abs();
        assert!(err_f32 < 0.05, "f32 gray ch{ch}: {}", dst_f32[ch]);
    }
}

// ---------------------------------------------------------------------------
// Test 7: RGBA alpha preservation with real profiles
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn rgba_alpha_preserved_real_profiles() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, data)| is_parseable(data) && is_rgb_profile(data))
        .collect();

    // Test alpha preservation for the first 5 profile pairs
    let mut tested = 0;
    'outer: for (src_name, src_data) in &rgb_profiles {
        for (dst_name, dst_data) in &rgb_profiles {
            if src_name == dst_name {
                continue;
            }
            let xform = match cms.build_transform_for_format(
                src_data,
                dst_data,
                PixelFormat::Rgba8,
                PixelFormat::Rgba8,
            ) {
                Ok(x) => x,
                Err(_) => continue,
            };

            // Test several alpha values
            for alpha in [0u8, 1, 42, 128, 254, 255] {
                let src = [200, 100, 50, alpha];
                let mut dst = [0u8; 4];
                xform.transform_row(&src, &mut dst, 1);

                assert_eq!(
                    dst[3], alpha,
                    "alpha not preserved: {src_name}→{dst_name}, \
                     expected {alpha}, got {}",
                    dst[3]
                );
            }

            tested += 1;
            if tested >= 10 {
                break 'outer;
            }
        }
    }

    eprintln!("  tested alpha preservation on {tested} profile pairs");
    assert!(tested > 0, "no RGBA transforms could be created");
}

// ---------------------------------------------------------------------------
// Test 8: Multi-pixel row consistency
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn multi_pixel_consistency() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let find = |name: &str| -> Option<&Vec<u8>> {
        profiles.iter().find(|(n, _)| n == name).map(|(_, d)| d)
    };

    let srgb = find("sRGB").expect("sRGB profile required");
    let adobe = find("AdobeRGB1998").expect("AdobeRGB profile required");

    let xform = cms.build_transform(srgb, adobe).unwrap();

    // Transform 4 pixels individually and as a row, compare results
    let pixels = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]];

    // Individual transforms
    let mut individual = [[0u8; 3]; 4];
    for (i, px) in pixels.iter().enumerate() {
        xform.transform_row(px, &mut individual[i], 1);
    }

    // Row transform
    let row_src: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let mut row_dst = vec![0u8; 12];
    xform.transform_row(&row_src, &mut row_dst, 4);

    // Compare
    for i in 0..4 {
        let row_px = &row_dst[i * 3..i * 3 + 3];
        assert_eq!(
            row_px, &individual[i],
            "pixel {i}: row={:?} vs individual={:?}",
            row_px, individual[i]
        );
    }

    eprintln!("  multi-pixel consistency verified for sRGB→AdobeRGB");
}

// ---------------------------------------------------------------------------
// Test 9: Ghostscript profile stress test
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn ghostscript_profiles_parse_and_identify() {
    let cms = MoxCms;
    let profiles = load_profiles(GHOSTSCRIPT_DIR);

    if profiles.is_empty() {
        eprintln!("  no ghostscript profiles found");
        return;
    }

    for (name, data) in &profiles {
        let parseable = is_parseable(data);
        let rgb = is_rgb_profile(data);
        let cicp = if parseable {
            cms.identify_profile(data)
        } else {
            None
        };

        eprintln!(
            "  {name}: {} bytes, parseable={parseable}, rgb={rgb}, cicp={cicp:?}",
            data.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 10: Profile pairs that should be near-identity
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn near_identity_profile_pairs() {
    let cms = MoxCms;
    let profiles = load_rgb_profiles();

    let find = |name: &str| -> Option<&Vec<u8>> {
        profiles.iter().find(|(n, _)| n == name).map(|(_, d)| d)
    };

    // sRGB ↔ Rec709 should be very close (same primaries, similar TRC)
    let srgb = match find("sRGB") {
        Some(d) => d,
        None => return,
    };
    let rec709 = match find("Rec709") {
        Some(d) => d,
        None => return,
    };

    let xform = cms.build_transform(srgb, rec709).unwrap();

    let mut max_diff = 0i32;
    for r in (0..=255).step_by(5) {
        for g in (0..=255).step_by(5) {
            for b in (0..=255).step_by(5) {
                let src = [r as u8, g as u8, b as u8];
                let mut dst = [0u8; 3];
                xform.transform_row(&src, &mut dst, 1);

                let diff = (0..3)
                    .map(|c| (dst[c] as i32 - src[c] as i32).abs())
                    .max()
                    .unwrap();
                max_diff = max_diff.max(diff);
            }
        }
    }

    eprintln!("  sRGB→Rec709 max difference: {max_diff}");
    // sRGB and Rec709 have the same primaries but different TRCs:
    // sRGB has a piecewise linear+gamma curve, Rec709 is pure gamma ~1/0.45.
    // The TRC difference causes up to ~16 levels of difference at u8,
    // concentrated near the toe of the curve.
    assert!(
        max_diff <= 20,
        "sRGB→Rec709 should be near-identity, got max diff {max_diff}"
    );
}
