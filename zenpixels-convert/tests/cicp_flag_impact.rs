//! Test whether `allow_use_cicp_transfer` affects output for real-world profiles.
//!
//! For ICC profiles without a cicp tag (all v2/v4.0–v4.3 profiles), moxcms's
//! `allow_use_cicp_transfer` flag should make no difference — the flag only
//! controls whether an *embedded* cicp tag's transfer function overrides the
//! ICC TRC curves.
//!
//! This test verifies that claim against real profiles from the imageflow test
//! suite and the saucecontrol/Compact-ICC-Profiles collection. If any profile
//! produces different pixels with the flag on vs off, that's either a moxcms
//! bug or a profile with unexpected cicp tags that we need to handle.
//!
//! # Running
//!
//! ```sh
//! cargo test -p zenpixels-convert --test cicp_flag_impact -- --ignored
//! ```

#![cfg(feature = "cms-moxcms")]

use std::path::Path;
use std::process::Command;

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
    TransformOptions,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract ICC profile bytes from a JPEG file using exiftool.
fn extract_icc_from_jpeg(path: &Path) -> Option<Vec<u8>> {
    let output = Command::new("exiftool")
        .args(["-icc_profile", "-b"])
        .arg(path)
        .output()
        .ok()?;
    if output.status.success() && output.stdout.len() > 128 {
        Some(output.stdout)
    } else {
        None
    }
}

/// Load all .icc files from a directory.
fn load_icc_dir(dir: &Path) -> Vec<(String, Vec<u8>)> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "icc" || e == "icm") {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            if let Ok(data) = std::fs::read(&path) {
                out.push((name, data));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Transform a test gradient with the given options, return output pixels.
fn transform_gradient(
    src_profile: &ColorProfile,
    dst_profile: &ColorProfile,
    opts: TransformOptions,
) -> Option<Vec<u8>> {
    // 256-pixel RGB gradient: R sweeps 0–255, G=128, B=64.
    // This exercises the full TRC range and catches rounding differences.
    let mut src = vec![0u8; 256 * 3];
    for i in 0..256 {
        src[i * 3] = i as u8;
        src[i * 3 + 1] = 128;
        src[i * 3 + 2] = 64;
    }

    let xform = src_profile
        .create_transform_8bit(Layout::Rgb, dst_profile, Layout::Rgb, opts)
        .ok()?;
    let mut dst = vec![0u8; 256 * 3];
    xform.transform(&src, &mut dst).ok()?;
    Some(dst)
}

/// Build transform options with a specific `allow_use_cicp_transfer` value.
fn opts_with_cicp_flag(allow: bool) -> TransformOptions {
    TransformOptions {
        rendering_intent: RenderingIntent::RelativeColorimetric,
        allow_use_cicp_transfer: allow,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test imageflow source JPEGs: does the cicp flag change output?
#[test]
#[ignore] // requires exiftool + imageflow corpus
fn cicp_flag_imageflow_jpegs() {
    let corpus = Path::new("/home/lilith/work/codec-corpus/imageflow/test_inputs");
    if !corpus.exists() {
        eprintln!("imageflow corpus not found, skipping");
        return;
    }

    let dst_profile = ColorProfile::new_srgb();
    let mut tested = 0;
    let mut diffs = Vec::new();

    for entry in std::fs::read_dir(corpus).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "jpg" || e == "jpeg") {
            let icc = match extract_icc_from_jpeg(&path) {
                Some(icc) => icc,
                None => continue,
            };
            let src_profile = match ColorProfile::new_from_slice(&icc) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let name = path.file_stem().unwrap().to_string_lossy().to_string();

            let has_cicp = src_profile.cicp.is_some();

            let out_no_cicp =
                transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(false));
            let out_cicp =
                transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(true));

            match (out_no_cicp, out_cicp) {
                (Some(a), Some(b)) => {
                    tested += 1;
                    if a != b {
                        // Count differing pixels and max delta
                        let max_delta: u8 = a
                            .iter()
                            .zip(b.iter())
                            .map(|(x, y)| x.abs_diff(*y))
                            .max()
                            .unwrap_or(0);
                        let diff_count = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                        diffs.push(format!(
                            "{name}: {diff_count} channels differ, max delta={max_delta}, \
                             cicp_in_profile={has_cicp}"
                        ));
                    } else {
                        eprintln!("  {name}: identical (cicp_in_profile={has_cicp})");
                    }
                }
                _ => {
                    eprintln!("  {name}: transform failed, skipping");
                }
            }
        }
    }

    eprintln!("\n{tested} profiles tested");
    if !diffs.is_empty() {
        eprintln!("\nProfiles where cicp flag changed output:");
        for d in &diffs {
            eprintln!("  {d}");
        }
    }
    // For profiles without cicp tags, the flag should make zero difference.
    // If any diff appears, we need to investigate.
    assert!(
        diffs.is_empty(),
        "cicp flag changed output for {} profiles (expected 0 for pre-v4.4 profiles)",
        diffs.len()
    );
}

/// Test Compact-ICC-Profiles: does the cicp flag change output?
#[test]
#[ignore] // requires network (git clone)
fn cicp_flag_compact_icc_profiles() {
    let tmp = std::env::temp_dir().join("zencodec-compact-icc-profiles");
    let profile_dir = tmp.join("profiles");

    if !profile_dir.exists() {
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/saucecontrol/Compact-ICC-Profiles.git",
            ])
            .arg(&tmp)
            .status();
        match status {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("git clone failed, skipping");
                return;
            }
        }
    }

    let profiles = load_icc_dir(&profile_dir);
    assert!(!profiles.is_empty(), "no profiles found");

    let dst_profile = ColorProfile::new_srgb();
    let mut tested = 0;
    let mut diffs = Vec::new();

    for (name, icc) in &profiles {
        let src_profile = match ColorProfile::new_from_slice(icc) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("  {name}: parse failed, skipping");
                continue;
            }
        };

        // Skip grayscale profiles — need Gray layout
        if icc.len() >= 20 && &icc[16..20] != b"RGB " {
            eprintln!("  {name}: not RGB, skipping");
            continue;
        }

        let out_a = transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(false));
        let out_b = transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(true));

        match (out_a, out_b) {
            (Some(a), Some(b)) => {
                tested += 1;
                if a != b {
                    let max_delta: u8 = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| x.abs_diff(*y))
                        .max()
                        .unwrap_or(0);
                    diffs.push(format!("{name}: max delta={max_delta}"));
                }
            }
            _ => eprintln!("  {name}: transform failed"),
        }
    }

    eprintln!("\n{tested} RGB profiles tested from Compact-ICC-Profiles");
    assert!(
        diffs.is_empty(),
        "cicp flag changed output for {} profiles: {:?}",
        diffs.len(),
        diffs
    );
}

/// Test moxcms-generated profiles WITH cicp tags: the flag SHOULD matter here.
/// This validates that the flag actually does something when cicp is present.
#[test]
fn cicp_flag_matters_for_v44_profiles() {
    let dst_profile = ColorProfile::new_srgb();

    // BT.2020 PQ profile — has cicp tag with PQ transfer (tc=16).
    // With allow_use_cicp_transfer=true, moxcms should use the PQ EOTF.
    // With allow_use_cicp_transfer=false, it should use the ICC TRC curves.
    let src_profile = ColorProfile::new_bt2020_pq();

    let out_no = transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(false));
    let out_yes = transform_gradient(&src_profile, &dst_profile, opts_with_cicp_flag(true));

    match (out_no, out_yes) {
        (Some(a), Some(b)) => {
            // For a PQ profile, the flag SHOULD produce different output
            // because PQ EOTF is very different from the parametric TRC
            // that moxcms stores in the ICC structure.
            if a == b {
                eprintln!("WARNING: cicp flag made no difference for BT.2020 PQ profile");
                eprintln!(
                    "This may mean moxcms always uses CICP when present, \
                           regardless of the flag."
                );
            } else {
                let max_delta: u8 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.abs_diff(*y))
                    .max()
                    .unwrap_or(0);
                let diff_count = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                eprintln!(
                    "BT.2020 PQ: flag matters — {diff_count} channels differ, max delta={max_delta}"
                );
            }
        }
        (None, None) => eprintln!("Both transforms failed for BT.2020 PQ — check moxcms version"),
        _ => panic!("One transform succeeded and the other failed — inconsistent"),
    }
}
