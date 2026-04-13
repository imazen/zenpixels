//! Compare fast_gamut vs moxcms relative colorimetric output pixel-by-pixel.
//!
//! For matrix+TRC profiles (P3, BT.2020), moxcms should do pure matrix+clamp
//! with relative colorimetric intent — same as fast_gamut. This test verifies
//! that claim exhaustively and reports any divergence.

#![cfg(feature = "cms-moxcms")]

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
    TransformOptions,
};

fn moxcms_opts() -> TransformOptions {
    TransformOptions {
        rendering_intent: RenderingIntent::RelativeColorimetric,
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

/// Build a ZenCmsLite u8 RGB transform as a closure.
fn build_lite_u8_fn(
    src_profile: zenpixels_convert::ColorProfileSource<'_>,
    dst_profile: zenpixels_convert::ColorProfileSource<'_>,
) -> Box<dyn Fn(&[u8], &mut [u8])> {
    use zenpixels_convert::cms::ColorManagement;
    let cms = zenpixels_convert::ZenCmsLite::default();
    let xf = cms
        .build_source_transform(
            src_profile,
            dst_profile,
            zenpixels_convert::PixelFormat::Rgb8,
            zenpixels_convert::PixelFormat::Rgb8,
        )
        .unwrap()
        .unwrap();
    Box::new(move |src: &[u8], dst: &mut [u8]| {
        let width = (src.len() / 3) as u32;
        xf.transform_row(src, dst, width);
    })
}

/// Build a ZenCmsLite f32 RGB in-place transform as a closure.
fn build_lite_f32_fn(
    src_profile: zenpixels_convert::ColorProfileSource<'_>,
    dst_profile: zenpixels_convert::ColorProfileSource<'_>,
) -> Box<dyn Fn(&mut [f32])> {
    use zenpixels_convert::cms::ColorManagement;
    let cms = zenpixels_convert::ZenCmsLite::default();
    let xf = cms
        .build_source_transform(
            src_profile,
            dst_profile,
            zenpixels_convert::PixelFormat::RgbF32,
            zenpixels_convert::PixelFormat::RgbF32,
        )
        .unwrap()
        .unwrap();
    Box::new(move |data: &mut [f32]| {
        let width = (data.len() / 3) as u32;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        // transform_row needs separate src/dst; copy src first.
        let src_copy = bytes.to_vec();
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(data);
        xf.transform_row(&src_copy, dst_bytes, width);
    })
}

/// Compare fast_gamut output vs moxcms for a full 256³ sweep.
/// Returns (max_delta, differing_pixel_count, total_pixels, example_worst).
fn compare_exhaustive_u8(
    fast_fn: &dyn Fn(&[u8], &mut [u8]),
    moxcms_src: &ColorProfile,
    moxcms_dst: &ColorProfile,
) -> (u8, usize, usize, Option<([u8; 3], [u8; 3], [u8; 3])>) {
    let opts = moxcms_opts();
    let xform = moxcms_src
        .create_transform_8bit(Layout::Rgb, moxcms_dst, Layout::Rgb, opts)
        .unwrap();

    let total = 256 * 256 * 256;
    let mut src = vec![0u8; total * 3];
    for i in 0..total {
        src[i * 3] = (i & 0xFF) as u8;
        src[i * 3 + 1] = ((i >> 8) & 0xFF) as u8;
        src[i * 3 + 2] = ((i >> 16) & 0xFF) as u8;
    }

    let mut fast_dst = vec![0u8; src.len()];
    let mut mox_dst = vec![0u8; src.len()];

    fast_fn(&src, &mut fast_dst);
    xform.transform(&src, &mut mox_dst).unwrap();

    let mut max_delta: u8 = 0;
    let mut diff_count: usize = 0;
    let mut worst: Option<([u8; 3], [u8; 3], [u8; 3])> = None;

    for i in 0..total {
        let off = i * 3;
        let f = [fast_dst[off], fast_dst[off + 1], fast_dst[off + 2]];
        let m = [mox_dst[off], mox_dst[off + 1], mox_dst[off + 2]];
        let s = [src[off], src[off + 1], src[off + 2]];

        if f != m {
            diff_count += 1;
            for ch in 0..3 {
                let d = f[ch].abs_diff(m[ch]);
                if d > max_delta {
                    max_delta = d;
                    worst = Some((s, f, m));
                }
            }
        }
    }

    (max_delta, diff_count, total, worst)
}

/// Same comparison but for f32, sampling a grid (not full 256³).
fn compare_f32_grid(
    fast_fn: &dyn Fn(&mut [f32]),
    moxcms_src: &ColorProfile,
    moxcms_dst: &ColorProfile,
    step: usize,
) -> (f32, usize, usize) {
    let opts = moxcms_opts();
    let xform = moxcms_src
        .create_transform_f32(Layout::Rgb, moxcms_dst, Layout::Rgb, opts)
        .unwrap();

    let steps = (256 + step - 1) / step;
    let total = steps * steps * steps;
    let mut src = vec![0.0f32; total * 3];
    let mut idx = 0;
    for r in (0..=255).step_by(step) {
        for g in (0..=255).step_by(step) {
            for b in (0..=255).step_by(step) {
                src[idx * 3] = r as f32 / 255.0;
                src[idx * 3 + 1] = g as f32 / 255.0;
                src[idx * 3 + 2] = b as f32 / 255.0;
                idx += 1;
            }
        }
    }

    let mut fast_buf = src.clone();
    let mut mox_dst = vec![0.0f32; src.len()];

    fast_fn(&mut fast_buf);
    xform.transform(&src, &mut mox_dst).unwrap();

    let mut max_delta: f32 = 0.0;
    let mut diff_count: usize = 0;

    for i in 0..idx {
        let off = i * 3;
        for ch in 0..3 {
            let d = (fast_buf[off + ch] - mox_dst[off + ch]).abs();
            if d > 1e-6 {
                diff_count += 1;
            }
            if d > max_delta {
                max_delta = d;
            }
        }
    }

    (max_delta, diff_count, idx)
}

// =========================================================================
// Tests
// =========================================================================

#[test]
fn p3_to_srgb_u8_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_u8_fn(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
        &*fast_fn,
        &ColorProfile::new_display_p3(),
        &ColorProfile::new_srgb(),
    );
    eprintln!("P3→sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
    if let Some((src, fast, mox)) = worst {
        eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
    }
    // Matrix+TRC profiles with relative colorimetric should be very close.
    // Allow ±2 due to different rounding/TRC precision.
    assert!(
        max_delta <= 2,
        "P3→sRGB u8: max delta {max_delta} > 2 — moxcms may use a different algorithm"
    );
}

#[test]
fn srgb_to_p3_u8_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_u8_fn(
        ColorProfileSource::Named(NamedProfile::Srgb),
        ColorProfileSource::Named(NamedProfile::DisplayP3),
    );
    let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
        &*fast_fn,
        &ColorProfile::new_srgb(),
        &ColorProfile::new_display_p3(),
    );
    eprintln!("sRGB→P3 u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
    if let Some((src, fast, mox)) = worst {
        eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
    }
    assert!(max_delta <= 2, "sRGB→P3 u8: max delta {max_delta} > 2");
}

#[test]
fn bt2020_sdr_to_srgb_u8_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_u8_fn(
        ColorProfileSource::Named(NamedProfile::Bt2020),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
        &*fast_fn,
        &ColorProfile::new_bt2020(),
        &ColorProfile::new_srgb(),
    );
    eprintln!("BT.2020 SDR→sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
    if let Some((src, fast, mox)) = worst {
        eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
    }
    // BT.2020→sRGB has large out-of-gamut regions. If moxcms does the same
    // clamp-after-matrix, deltas should be small (rounding only).
    // If deltas are large, moxcms is doing something different for OOG colors.
    assert!(
        max_delta <= 3,
        "BT.2020→sRGB u8: max delta {max_delta} > 3 — check if moxcms does gamut mapping"
    );
}

#[test]
fn adobergb_to_srgb_u8_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_u8_fn(
        ColorProfileSource::Named(NamedProfile::AdobeRgb),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
        &*fast_fn,
        &ColorProfile::new_adobe_rgb(),
        &ColorProfile::new_srgb(),
    );
    eprintln!("AdobeRGB→sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
    if let Some((src, fast, mox)) = worst {
        eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
    }
    assert!(
        max_delta <= 2,
        "AdobeRGB→sRGB u8: max delta {max_delta} > 2"
    );
}

#[test]
fn p3_to_srgb_f32_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_f32_fn(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let (max_delta, diff_count, total) = compare_f32_grid(
        &*fast_fn,
        &ColorProfile::new_display_p3(),
        &ColorProfile::new_srgb(),
        4,
    );
    eprintln!(
        "P3→sRGB f32: {diff_count}/{total}×3 channels differ >1e-6, max delta={max_delta:.6e}"
    );
    // f32 max delta should be tiny — polynomial vs LUT difference.
    assert!(
        max_delta < 0.01,
        "P3→sRGB f32: max delta {max_delta} — unexpected divergence"
    );
}

#[test]
fn bt2020_to_srgb_f32_vs_moxcms() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let fast_fn = build_lite_f32_fn(
        ColorProfileSource::Named(NamedProfile::Bt2020),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let (max_delta, diff_count, total) = compare_f32_grid(
        &*fast_fn,
        &ColorProfile::new_bt2020(),
        &ColorProfile::new_srgb(),
        4,
    );
    eprintln!(
        "BT.2020→sRGB f32: {diff_count}/{total}×3 channels differ >1e-6, max delta={max_delta:.6e}"
    );
    assert!(max_delta < 0.01, "BT.2020→sRGB f32: max delta {max_delta}");
}
