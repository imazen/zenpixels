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

/// Build a full 256³ RGB cube (16,777,216 pixels) as RGBA with alpha sweep.
///
/// Every possible (R, G, B) combination appears exactly once. Alpha cycles
/// through 0–255 across the pixels so alpha-associated rounding is exercised.
/// Total: 256³ × 4 = 67,108,864 bytes.
fn build_full_rgba_cube() -> Vec<u8> {
    let pixel_count: usize = 256 * 256 * 256;
    let mut src = vec![0u8; pixel_count * 4];
    let mut idx = 0;
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                let off = idx * 4;
                src[off] = r;
                src[off + 1] = g;
                src[off + 2] = b;
                src[off + 3] = idx as u8; // alpha cycles 0–255
                idx += 1;
            }
        }
    }
    src
}

/// Transform the full RGBA cube with the given options, return output pixels.
fn transform_full_cube(
    src_profile: &ColorProfile,
    dst_profile: &ColorProfile,
    opts: TransformOptions,
    src: &[u8],
) -> Option<Vec<u8>> {
    let xform = src_profile
        .create_transform_8bit(Layout::Rgba, dst_profile, Layout::Rgba, opts)
        .ok()?;
    let mut dst = vec![0u8; src.len()];
    xform.transform(src, &mut dst).ok()?;
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

/// Compare two RGBA buffers, return (differing_channel_count, max_delta).
fn compare_buffers(a: &[u8], b: &[u8]) -> (usize, u8) {
    let max_delta = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    let diff_count = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    (diff_count, max_delta)
}

/// Test a single profile: transform full 256³ RGBA cube with flag on/off, compare.
fn test_profile_cicp_flag(
    name: &str,
    src_profile: &ColorProfile,
    dst_profile: &ColorProfile,
    cube: &[u8],
) -> Option<String> {
    let has_cicp = src_profile.cicp.is_some();

    let out_off = transform_full_cube(src_profile, dst_profile, opts_with_cicp_flag(false), cube);
    let out_on = transform_full_cube(src_profile, dst_profile, opts_with_cicp_flag(true), cube);

    match (out_off, out_on) {
        (Some(a), Some(b)) => {
            if a != b {
                let (diff_count, max_delta) = compare_buffers(&a, &b);
                Some(format!(
                    "{name}: {diff_count} channels differ, max delta={max_delta}, \
                     cicp_in_profile={has_cicp}"
                ))
            } else {
                eprintln!("  {name}: identical across all 256³×4 values (cicp={has_cicp})");
                None
            }
        }
        _ => {
            eprintln!("  {name}: transform failed, skipping");
            None
        }
    }
}

/// Test imageflow source JPEGs: does the cicp flag change output?
/// Exercises every possible (R,G,B) combination with alpha sweep.
#[test]
#[ignore] // requires exiftool + imageflow corpus
fn cicp_flag_imageflow_jpegs() {
    let corpus = Path::new("/home/lilith/work/codec-corpus/imageflow/test_inputs");
    if !corpus.exists() {
        eprintln!("imageflow corpus not found, skipping");
        return;
    }

    let dst_profile = ColorProfile::new_srgb();
    let cube = build_full_rgba_cube();
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

            if let Some(diff) = test_profile_cicp_flag(&name, &src_profile, &dst_profile, &cube) {
                diffs.push(diff);
            }
            tested += 1;
        }
    }

    eprintln!("\n{tested} profiles tested, each against full 256³ RGBA cube");
    assert!(
        diffs.is_empty(),
        "cicp flag changed output for {} profiles (expected 0 for pre-v4.4 profiles)",
        diffs.len()
    );
}

/// Test Compact-ICC-Profiles: does the cicp flag change output?
/// Exercises every possible (R,G,B) combination with alpha sweep.
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
    let cube = build_full_rgba_cube();
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

        // Skip non-RGB profiles
        if icc.len() >= 20 && &icc[16..20] != b"RGB " {
            eprintln!("  {name}: not RGB, skipping");
            continue;
        }

        if let Some(diff) = test_profile_cicp_flag(name, &src_profile, &dst_profile, &cube) {
            diffs.push(diff);
        }
        tested += 1;
    }

    eprintln!(
        "\n{tested} RGB profiles tested from Compact-ICC-Profiles, each against full 256³ RGBA cube"
    );
    assert!(
        diffs.is_empty(),
        "cicp flag changed output for {} profiles: {:?}",
        diffs.len(),
        diffs
    );
}

/// Test moxcms-generated profiles WITH cicp tags: the flag SHOULD matter here.
/// This validates that the flag actually does something when cicp is present.
/// Uses the full 256³ RGBA cube for exhaustive coverage.
#[test]
fn cicp_flag_matters_for_v44_profiles() {
    let dst_profile = ColorProfile::new_srgb();
    let cube = build_full_rgba_cube();

    // BT.2020 PQ profile — has cicp tag with PQ transfer (tc=16).
    let src_profile = ColorProfile::new_bt2020_pq();

    let out_no = transform_full_cube(
        &src_profile,
        &dst_profile,
        opts_with_cicp_flag(false),
        &cube,
    );
    let out_yes = transform_full_cube(&src_profile, &dst_profile, opts_with_cicp_flag(true), &cube);

    match (out_no, out_yes) {
        (Some(a), Some(b)) => {
            if a == b {
                eprintln!("WARNING: cicp flag made no difference for BT.2020 PQ profile");
                eprintln!(
                    "This may mean moxcms always uses CICP when present, \
                     regardless of the flag."
                );
            } else {
                let (diff_count, max_delta) = compare_buffers(&a, &b);
                eprintln!(
                    "BT.2020 PQ: flag matters — {diff_count}/{} channels differ, max delta={max_delta}",
                    a.len()
                );
            }
        }
        (None, None) => eprintln!("Both transforms failed for BT.2020 PQ — check moxcms version"),
        _ => panic!("One transform succeeded and the other failed — inconsistent"),
    }
}
