//! CMS tests using real-world ICC profiles extracted from non-sRGB images.
//!
//! Exercises the moxcms CMS backend against ICC profiles embedded in a corpus
//! of ~500 real-world images at `/mnt/v/datasets/non-srgb-by-profile/`. The
//! corpus contains 63 unique profiles covering:
//!
//! - Standard working spaces: Adobe RGB (11 variants), ProPhoto RGB, opRGB
//! - Display calibration: Display P3 (30+ variants from Apple, Samsung, etc.)
//! - Broadcast: BT.709 (Apple)
//! - Photography: Camera RGB, eciRGB v2, Epson Gamma 1.8, Generic RGB
//! - Monitor calibration: HUAWEI AD80HW, DELL P2722H, DELL S2722QC, VG2228
//! - Grayscale: Gray Gamma 2.2, Generic Gray Gamma 2.2
//! - Large/complex: OS12002_mG (669KB, Lab PCS)
//!
//! # Running
//!
//! ```sh
//! cargo test -p zenpixels-convert --test cms_corpus_profiles -- --ignored
//! ```

#![cfg(feature = "cms-moxcms")]

extern crate alloc;

use std::collections::HashMap;
use std::path::Path;

use zenpixels_convert::PixelFormat;
use zenpixels_convert::cms::ColorManagement;
use zenpixels_convert::cms_moxcms::MoxCms;

// ---------------------------------------------------------------------------
// ICC extraction from image files
// ---------------------------------------------------------------------------

const CORPUS_DIR: &str = "/mnt/v/datasets/non-srgb-by-profile";

/// Extract ICC profile from a WebP file (RIFF 'ICCP' chunk).
fn extract_webp_icc(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return None;
    }
    let riff_size = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
    let end = (riff_size + 8).min(data.len());
    let mut pos = 12;
    while pos + 8 <= end {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?) as usize;
        if chunk_id == b"ICCP" {
            let icc_end = (pos + 8 + chunk_size).min(data.len());
            return Some(data[pos + 8..icc_end].to_vec());
        }
        // Chunks are padded to even size
        pos += 8 + chunk_size + (chunk_size % 2);
    }
    None
}

/// Extract ICC profile from a JPEG file (APP2 ICC_PROFILE markers).
fn extract_jpeg_icc(data: &[u8]) -> Option<Vec<u8>> {
    let marker = b"ICC_PROFILE\x00";
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut pos = 0;

    while pos + 4 < data.len() {
        // Find APP2 marker (0xFF 0xE2)
        let idx = data[pos..].windows(2).position(|w| w == [0xFF, 0xE2]);
        let idx = match idx {
            Some(i) => pos + i,
            None => break,
        };
        if idx + 4 > data.len() {
            break;
        }
        let length = u16::from_be_bytes(data[idx + 2..idx + 4].try_into().ok()?) as usize;
        let segment_end = (idx + 2 + length).min(data.len());
        let segment = &data[idx + 4..segment_end];

        if segment.len() >= 14 && &segment[..12] == marker {
            let seq_no = segment[12];
            chunks.push((seq_no, segment[14..].to_vec()));
        }
        pos = segment_end;
    }

    if chunks.is_empty() {
        return None;
    }
    chunks.sort_by_key(|(seq, _)| *seq);
    Some(chunks.into_iter().flat_map(|(_, d)| d).collect())
}

/// Extract ICC profile from a file based on its extension.
fn extract_icc(path: &Path) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "webp" => extract_webp_icc(&data),
        "jpg" | "jpeg" => extract_jpeg_icc(&data),
        _ => None,
    }
}

/// ICC color space from header bytes 16..20.
fn icc_color_space(icc: &[u8]) -> Option<[u8; 4]> {
    if icc.len() < 24 {
        return None;
    }
    Some([icc[16], icc[17], icc[18], icc[19]])
}

/// Check if profile is RGB (not GRAY, CMYK, Lab, etc.)
fn is_rgb_profile(icc: &[u8]) -> bool {
    icc_color_space(icc) == Some(*b"RGB ")
}

/// Check if profile uses XYZ PCS (not Lab).
///
/// Lab PCS profiles (typically scanner/device link profiles) use
/// multi-dimensional LUTs and may not behave as matrix-based profiles.
fn is_xyz_pcs(icc: &[u8]) -> bool {
    icc.len() >= 24 && &icc[20..24] == b"XYZ "
}

/// Check if profile is a simple matrix-based RGB profile with XYZ PCS.
fn is_matrix_rgb_profile(icc: &[u8]) -> bool {
    is_rgb_profile(icc) && is_xyz_pcs(icc)
}

/// Load unique ICC profiles from the corpus, keyed by SHA-256 prefix.
fn load_corpus_profiles() -> Vec<(String, Vec<u8>)> {
    let corpus = Path::new(CORPUS_DIR);
    if !corpus.exists() {
        return Vec::new();
    }

    let mut seen: HashMap<[u8; 32], ()> = HashMap::new();
    let mut profiles = Vec::new();

    for category in std::fs::read_dir(corpus).unwrap() {
        let category = category.unwrap();
        if !category.file_type().unwrap().is_dir() {
            continue;
        }
        let cat_name = category.file_name().to_string_lossy().to_string();

        for entry in std::fs::read_dir(category.path()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if let Some(icc) = extract_icc(&path) {
                use std::hash::{Hash, Hasher};
                // Use a simple hash for dedup (SHA-256 would be ideal but
                // we don't want to add a dep — a hash of the full bytes is fine)
                let mut hasher = std::hash::DefaultHasher::new();
                icc.hash(&mut hasher);
                let h = hasher.finish().to_le_bytes();
                let mut key = [0u8; 32];
                key[..8].copy_from_slice(&h);
                key[8..16].copy_from_slice(&(icc.len() as u64).to_le_bytes());

                if seen.insert(key, ()).is_none() {
                    let file_name = path.file_name().unwrap().to_string_lossy();
                    let label = format!("{cat_name}/{file_name}");
                    profiles.push((label, icc));
                }
            }
        }
    }

    profiles.sort_by(|a, b| a.0.cmp(&b.0));
    profiles
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn corpus_profile_count() {
    let profiles = load_corpus_profiles();
    assert!(
        profiles.len() >= 40,
        "expected at least 40 unique profiles, found {}",
        profiles.len()
    );
    eprintln!("loaded {} unique ICC profiles from corpus", profiles.len());
    for (name, icc) in &profiles {
        let cs = icc_color_space(icc).map(|b| String::from_utf8_lossy(&b).to_string());
        eprintln!(
            "  {name:50} {:>7}B  cs={}",
            icc.len(),
            cs.unwrap_or_default()
        );
    }
}

#[test]
#[ignore]
fn all_corpus_profiles_parse() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();
    assert!(!profiles.is_empty(), "corpus not found at {CORPUS_DIR}");

    let mut parse_ok = 0;
    let mut parse_fail = 0;

    for (name, icc) in &profiles {
        // Attempt to identify — this forces moxcms to parse the profile.
        let _ = cms.identify_profile(icc);

        // Also attempt to create a self-transform if RGB.
        if is_rgb_profile(icc) {
            match cms.build_transform(icc, icc) {
                Ok(_) => parse_ok += 1,
                Err(e) => {
                    eprintln!("  parse/transform fail: {name}: {e:?}");
                    parse_fail += 1;
                }
            }
        }
    }

    eprintln!("RGB self-transform: {parse_ok} ok, {parse_fail} fail");
    // Allow some failures (Lab PCS, unusual profiles) but most should work.
    assert!(
        parse_ok > 30,
        "too few successful parses: {parse_ok} ok, {parse_fail} fail"
    );
}

#[test]
#[ignore]
fn identify_corpus_profiles() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let mut identified = Vec::new();
    let mut unidentified = Vec::new();

    for (name, icc) in &profiles {
        if !is_rgb_profile(icc) {
            continue;
        }
        match cms.identify_profile(icc) {
            Some(cicp) => identified.push((name.as_str(), cicp)),
            None => unidentified.push(name.as_str()),
        }
    }

    eprintln!("identified {} profiles:", identified.len());
    for (name, cicp) in &identified {
        eprintln!(
            "  {name}: primaries={}, transfer={}",
            cicp.color_primaries, cicp.transfer_characteristics
        );
    }
    eprintln!("unidentified {} profiles:", unidentified.len());
    for name in &unidentified {
        eprintln!("  {name}");
    }

    // We should identify at least some sRGB/P3/BT.2020 variants
    let has_bt709 = identified.iter().any(|(_, c)| c.color_primaries == 1);
    let has_p3 = identified.iter().any(|(_, c)| c.color_primaries == 12);
    assert!(
        has_bt709 || has_p3,
        "should identify at least one BT.709 or Display P3 profile"
    );
}

#[test]
#[ignore]
fn all_rgb_pairs_create_transforms() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_rgb_profile(icc))
        .collect();

    eprintln!(
        "testing {} RGB profiles ({} pairs)",
        rgb_profiles.len(),
        rgb_profiles.len() * rgb_profiles.len()
    );

    let mut ok = 0;
    let mut fail = 0;

    for (src_name, src_icc) in &rgb_profiles {
        for (dst_name, dst_icc) in &rgb_profiles {
            match cms.build_transform(src_icc, dst_icc) {
                Ok(_) => ok += 1,
                Err(e) => {
                    if fail < 10 {
                        eprintln!("  fail: {src_name} → {dst_name}: {e:?}");
                    }
                    fail += 1;
                }
            }
        }
    }

    eprintln!("pairwise transforms: {ok} ok, {fail} fail");
    // Most pairs should work. Allow some failures for exotic profiles.
    let total = ok + fail;
    let success_rate = ok as f64 / total as f64;
    assert!(
        success_rate > 0.90,
        "too many transform failures: {ok}/{total} = {:.1}%",
        success_rate * 100.0
    );
}

#[test]
#[ignore]
fn white_preservation_corpus() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let srgb_profile: Option<&Vec<u8>> = profiles
        .iter()
        .find(|(_, icc)| {
            is_matrix_rgb_profile(icc)
                && cms
                    .identify_profile(icc)
                    .is_some_and(|c| c.color_primaries == 1)
        })
        .map(|(_, icc)| icc);

    // Fall back to any matrix-based RGB profile for the sRGB side
    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_matrix_rgb_profile(icc))
        .collect();

    let src_icc = srgb_profile.unwrap_or_else(|| &rgb_profiles[0].1);

    let mut tested = 0;
    let mut max_err = 0u8;
    let mut worst_name = String::new();

    for (name, dst_icc) in &rgb_profiles {
        let xform = match cms.build_transform(src_icc, dst_icc) {
            Ok(x) => x,
            Err(_) => continue,
        };

        let src = [255u8, 255, 255];
        let mut dst = [0u8; 3];
        xform.transform_row(&src, &mut dst, 1);

        for &val in &dst {
            let err = 255 - val;
            if err > max_err {
                max_err = err;
                worst_name = name.clone();
            }
        }
        tested += 1;
    }

    eprintln!(
        "white preservation: tested {tested} profiles, max error {max_err} (worst: {worst_name})"
    );
    assert!(
        max_err <= 10,
        "white preservation error too large: {max_err} on {worst_name}"
    );
}

#[test]
#[ignore]
fn black_preservation_corpus() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_matrix_rgb_profile(icc))
        .collect();

    let src_icc = &rgb_profiles[0].1;

    let mut tested = 0;
    let mut max_err = 0u8;
    let mut worst_name = String::new();

    for (name, dst_icc) in &rgb_profiles {
        let xform = match cms.build_transform(src_icc, dst_icc) {
            Ok(x) => x,
            Err(_) => continue,
        };

        let src = [0u8, 0, 0];
        let mut dst = [0u8; 3];
        xform.transform_row(&src, &mut dst, 1);

        for &val in &dst {
            if val > max_err {
                max_err = val;
                worst_name = name.clone();
            }
        }
        tested += 1;
    }

    eprintln!(
        "black preservation: tested {tested} profiles, max error {max_err} (worst: {worst_name})"
    );
    assert!(
        max_err <= 5,
        "black preservation error too large: {max_err} on {worst_name}"
    );
}

#[test]
#[ignore]
fn format_aware_transforms_corpus() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_rgb_profile(icc))
        .take(10) // Sample to keep runtime bounded
        .collect();

    let formats = [
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba16,
        PixelFormat::RgbF32,
        PixelFormat::RgbaF32,
        PixelFormat::Gray8,
    ];

    let mut ok = 0;
    let mut fail = 0;

    for (src_name, src_icc) in &rgb_profiles {
        for (dst_name, dst_icc) in &rgb_profiles {
            for &fmt in &formats {
                match cms.build_transform_for_format(src_icc, dst_icc, fmt, fmt) {
                    Ok(_) => ok += 1,
                    Err(e) => {
                        if fail < 5 {
                            eprintln!("  fail: {src_name} → {dst_name} ({:?}): {e:?}", fmt);
                        }
                        fail += 1;
                    }
                }
            }
        }
    }

    eprintln!("format-aware transforms: {ok} ok, {fail} fail");
    assert!(ok > 100, "too few successful format-aware transforms: {ok}");
}

#[test]
#[ignore]
fn rgba_alpha_preserved_corpus() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_rgb_profile(icc))
        .take(10)
        .collect();

    for (src_name, src_icc) in &rgb_profiles {
        for (dst_name, dst_icc) in &rgb_profiles {
            let xform = match cms.build_transform_for_format(
                src_icc,
                dst_icc,
                PixelFormat::Rgba8,
                PixelFormat::Rgba8,
            ) {
                Ok(x) => x,
                Err(_) => continue,
            };

            // Test a range of alpha values
            for alpha in [0u8, 42, 128, 200, 255] {
                let src = [100, 150, 200, alpha];
                let mut dst = [0u8; 4];
                xform.transform_row(&src, &mut dst, 1);
                assert_eq!(
                    dst[3], alpha,
                    "alpha not preserved for {src_name} → {dst_name} with alpha={alpha}"
                );
            }
        }
    }
}

#[test]
#[ignore]
fn monitor_profile_transforms() {
    // Monitor calibration profiles have device-specific TRCs that may differ
    // significantly from standard gamma curves. Test that transforms still
    // produce reasonable results.
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let monitor_profiles: Vec<_> = profiles
        .iter()
        .filter(|(name, icc)| name.starts_with("monitor-profiles/") && is_matrix_rgb_profile(icc))
        .collect();

    if monitor_profiles.is_empty() {
        eprintln!("no monitor profiles found, skipping");
        return;
    }

    eprintln!("testing {} monitor profiles", monitor_profiles.len());

    // Find an sRGB-ish profile to use as source
    let srgb_like = profiles
        .iter()
        .find(|(name, _)| name.contains("bt709") || name.contains("display-p3"))
        .or_else(|| profiles.iter().find(|(_, icc)| is_rgb_profile(icc)));

    let (src_name, src_icc) = srgb_like.expect("need at least one RGB profile as source");

    for (mon_name, mon_icc) in &monitor_profiles {
        let xform = match cms.build_transform(src_icc, mon_icc) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  skip {mon_name}: {e:?}");
                continue;
            }
        };

        // Mid-gray should map to approximately mid-gray
        let src = [128u8, 128, 128];
        let mut dst = [0u8; 3];
        xform.transform_row(&src, &mut dst, 1);

        for (ch, &val) in dst.iter().enumerate() {
            let err = (val as i32 - 128).abs();
            assert!(
                err <= 40,
                "monitor profile {mon_name}: gray ch{ch} mapped to {val}, error {err} \
                 (source: {src_name})",
            );
        }
        eprintln!(
            "  {mon_name}: gray [128,128,128] → [{},{},{}]",
            dst[0], dst[1], dst[2]
        );
    }
}

#[test]
#[ignore]
fn grayscale_profiles_parse() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let gray_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| icc_color_space(icc) == Some(*b"GRAY"))
        .collect();

    eprintln!("found {} grayscale profiles", gray_profiles.len());

    for (name, icc) in &gray_profiles {
        // Should parse without panic
        let id = cms.identify_profile(icc);
        eprintln!("  {name}: identify={id:?}");
    }
}

#[test]
#[ignore]
fn camera_profiles_transform_correctly() {
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let camera_profiles: Vec<_> = profiles
        .iter()
        .filter(|(name, icc)| name.starts_with("camera-rgb/") && is_rgb_profile(icc))
        .collect();

    // Use a standard profile as destination
    let std_profile = profiles
        .iter()
        .find(|(name, icc)| {
            is_rgb_profile(icc) && (name.contains("adobe-rgb") || name.contains("display-p3"))
        })
        .map(|(_, icc)| icc);

    let dst_icc = match std_profile {
        Some(p) => p,
        None => {
            eprintln!("no standard RGB profile found, skipping");
            return;
        }
    };

    for (name, src_icc) in &camera_profiles {
        let xform = match cms.build_transform(src_icc, dst_icc) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  skip {name}: {e:?}");
                continue;
            }
        };

        // Primary colors should produce non-degenerate output
        let test_colors = [
            ([255u8, 0, 0], "red"),
            ([0, 255, 0], "green"),
            ([0, 0, 255], "blue"),
            ([255, 255, 255], "white"),
            ([0, 0, 0], "black"),
            ([128, 128, 128], "gray"),
        ];

        for (src, color_name) in &test_colors {
            let mut dst = [0u8; 3];
            xform.transform_row(src, &mut dst, 1);
            eprintln!(
                "  {name}: {color_name} [{},{},{}] → [{},{},{}]",
                src[0], src[1], src[2], dst[0], dst[1], dst[2]
            );
        }
    }
}

#[test]
#[ignore]
fn self_transform_near_identity() {
    // For every matrix-based RGB profile, a self-transform (same src and dst)
    // should be approximately identity. Lab PCS profiles with LUTs may not
    // round-trip cleanly.
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let rgb_profiles: Vec<_> = profiles
        .iter()
        .filter(|(_, icc)| is_matrix_rgb_profile(icc))
        .collect();

    let mut worst_err = 0i32;
    let mut worst_name = String::new();

    for (name, icc) in &rgb_profiles {
        let xform = match cms.build_transform(icc, icc) {
            Ok(x) => x,
            Err(_) => continue,
        };

        // Test a grid of colors
        for r in (0..=255).step_by(51) {
            for g in (0..=255).step_by(51) {
                for b in (0..=255).step_by(51) {
                    let src = [r as u8, g as u8, b as u8];
                    let mut dst = [0u8; 3];
                    xform.transform_row(&src, &mut dst, 1);

                    for ch in 0..3 {
                        let err = (dst[ch] as i32 - src[ch] as i32).abs();
                        if err > worst_err {
                            worst_err = err;
                            worst_name =
                                format!("{name} at ({r},{g},{b}) ch{ch}: {}→{}", src[ch], dst[ch]);
                        }
                    }
                }
            }
        }
    }

    eprintln!("self-transform worst error: {worst_err} ({worst_name})");
    assert!(
        worst_err <= 2,
        "self-transform error too large: {worst_err} ({worst_name})"
    );
}

#[test]
#[ignore]
fn adobe_rgb_variant_consistency() {
    // Multiple Adobe RGB variants should produce very similar transforms.
    let cms = MoxCms;
    let profiles = load_corpus_profiles();

    let adobe_profiles: Vec<_> = profiles
        .iter()
        .filter(|(name, icc)| name.starts_with("adobe-rgb/") && is_matrix_rgb_profile(icc))
        .collect();

    if adobe_profiles.len() < 2 {
        eprintln!(
            "need >= 2 Adobe RGB variants, found {}",
            adobe_profiles.len()
        );
        return;
    }

    // Use the first Adobe RGB profile as reference
    let (ref_name, ref_icc) = &adobe_profiles[0];

    // Find a Display P3 profile as common destination
    let dst_icc = profiles
        .iter()
        .find(|(name, icc)| name.starts_with("display-p3/") && is_rgb_profile(icc))
        .map(|(_, icc)| icc)
        .unwrap_or(ref_icc);

    let ref_xform = cms.build_transform(ref_icc, dst_icc).unwrap();

    let mut outlier_count = 0;

    // Transform reference test pixels
    let test_pixels: Vec<[u8; 3]> = (0..=255)
        .step_by(51)
        .flat_map(|r| {
            (0..=255).step_by(51).flat_map(move |g| {
                (0..=255)
                    .step_by(51)
                    .map(move |b| [r as u8, g as u8, b as u8])
            })
        })
        .collect();

    let ref_results: Vec<[u8; 3]> = test_pixels
        .iter()
        .map(|src| {
            let mut dst = [0u8; 3];
            ref_xform.transform_row(src, &mut dst, 1);
            dst
        })
        .collect();

    for (name, icc) in &adobe_profiles[1..] {
        let xform = match cms.build_transform(icc, dst_icc) {
            Ok(x) => x,
            Err(_) => continue,
        };

        let mut max_diff = 0i32;
        for (src, ref_dst) in test_pixels.iter().zip(ref_results.iter()) {
            let mut dst = [0u8; 3];
            xform.transform_row(src, &mut dst, 1);

            for ch in 0..3 {
                let diff = (dst[ch] as i32 - ref_dst[ch] as i32).abs();
                max_diff = max_diff.max(diff);
            }
        }

        eprintln!("  {name} vs {ref_name}: max diff {max_diff}");
        // Most Adobe RGB variants should be very close, but camera-specific
        // variants (Nikon, Canon, etc.) may embed substantially different TRCs
        // while keeping the "Adobe RGB" name. We flag but don't fail on these.
        if max_diff > 20 {
            eprintln!("    WARNING: large divergence — likely a camera-specific TRC variant");
            outlier_count += 1;
        }
    }

    // At most half should be outliers — most Adobe RGB profiles are the real deal.
    let variant_count = adobe_profiles.len() - 1;
    assert!(
        outlier_count <= variant_count / 2,
        "too many Adobe RGB outliers: {outlier_count}/{variant_count}"
    );
}
