//! Determine who is more accurate: fast_gamut or moxcms.
//!
//! Uses f64 math as ground truth — compute the exact conversion with
//! f64 precision, round to u8, and compare both implementations against it.

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

// =========================================================================
// f64 ground truth implementation
// =========================================================================

/// sRGB EOTF (encoded → linear) in f64, C0-continuous constants.
fn srgb_to_linear_f64(v: f64) -> f64 {
    // C0-continuous threshold from moxcms / linear-srgb
    const THRESH: f64 = 0.0392857142857142850819238;
    if v <= THRESH {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB inverse EOTF (linear → encoded) in f64, C0-continuous constants.
fn linear_to_srgb_f64(v: f64) -> f64 {
    const THRESH: f64 = 0.00303993464041981300277518;
    if v <= THRESH {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// BT.709 OETF (linear → encoded) in f64.
fn linear_to_bt709_f64(v: f64) -> f64 {
    if v < 0.018053968510807807 {
        v * 4.5
    } else {
        1.09929682680944 * v.powf(0.45) - 0.09929682680944
    }
}

/// BT.709 inverse OETF (encoded → linear) in f64.
fn bt709_to_linear_f64(v: f64) -> f64 {
    if v < 0.08124285829863519 {
        v / 4.5
    } else {
        ((v + 0.09929682680944) / 1.09929682680944).powf(1.0 / 0.45)
    }
}

/// f64 3x3 matrix multiply.
fn mat3x3_f64(m: &[[f64; 3]; 3], r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

/// P3 → sRGB matrix in f64 (same derivation as fast_gamut, higher precision).
const P3_TO_SRGB_F64: [[f64; 3]; 3] = [
    [1.2249401763_f64, -0.2249401763_f64, 0.0_f64],
    [-0.0420569547_f64, 1.0420569547_f64, 0.0_f64],
    [-0.0196375546_f64, -0.0786360456_f64, 1.0982736001_f64],
];

const BT2020_TO_SRGB_F64: [[f64; 3]; 3] = [
    [1.6604910021_f64, -0.5876411388_f64, -0.0728498633_f64],
    [-0.1245504745_f64, 1.1328998971_f64, -0.0083494226_f64],
    [-0.0181507634_f64, -0.1005788980_f64, 1.1187296614_f64],
];

const ADOBERGB_TO_SRGB_F64: [[f64; 3]; 3] = [
    [1.3983557440_f64, -0.3983557440_f64, 0.0_f64],
    [0.0_f64, 1.0_f64, 0.0_f64],
    [0.0_f64, -0.0429289893_f64, 1.0429289893_f64],
];

/// Adobe RGB gamma
const ADOBE_GAMMA: f64 = 563.0 / 256.0;

/// Compute ground truth u8 output via f64 math.
fn ground_truth_u8(
    src: &[u8],
    matrix: &[[f64; 3]; 3],
    linearize: fn(f64) -> f64,
    encode: fn(f64) -> f64,
) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = linearize(s[0] as f64 / 255.0);
        let g = linearize(s[1] as f64 / 255.0);
        let b = linearize(s[2] as f64 / 255.0);
        let (nr, ng, nb) = mat3x3_f64(matrix, r, g, b);
        d[0] = (encode(nr.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        d[1] = (encode(ng.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        d[2] = (encode(nb.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    dst
}

/// Build full 256³ source buffer.
fn full_rgb_cube() -> Vec<u8> {
    let total = 256 * 256 * 256;
    let mut src = vec![0u8; total * 3];
    for i in 0..total {
        src[i * 3] = (i & 0xFF) as u8;
        src[i * 3 + 1] = ((i >> 8) & 0xFF) as u8;
        src[i * 3 + 2] = ((i >> 16) & 0xFF) as u8;
    }
    src
}

struct AccuracyResult {
    name: &'static str,
    fast_exact: usize,
    fast_off1: usize,
    fast_off2plus: usize,
    fast_max_delta: u8,
    mox_exact: usize,
    mox_off1: usize,
    mox_off2plus: usize,
    mox_max_delta: u8,
    fast_better: usize,
    mox_better: usize,
    tied: usize,
    total: usize,
}

impl AccuracyResult {
    fn print(&self) {
        let pct = |n: usize| n as f64 / self.total as f64 * 100.0;
        eprintln!("\n=== {} ({} pixels) ===", self.name, self.total);
        eprintln!(
            "  fast_gamut: exact={:.1}% ±1={:.1}% ±2+={:.1}% max_delta={}",
            pct(self.fast_exact),
            pct(self.fast_off1),
            pct(self.fast_off2plus),
            self.fast_max_delta
        );
        eprintln!(
            "  moxcms:     exact={:.1}% ±1={:.1}% ±2+={:.1}% max_delta={}",
            pct(self.mox_exact),
            pct(self.mox_off1),
            pct(self.mox_off2plus),
            self.mox_max_delta
        );
        eprintln!(
            "  fast_gamut closer to truth: {:.1}%",
            pct(self.fast_better)
        );
        eprintln!("  moxcms closer to truth:     {:.1}%", pct(self.mox_better));
        eprintln!("  tied (equal distance):      {:.1}%", pct(self.tied));
    }
}

fn compare_accuracy(
    name: &'static str,
    src: &[u8],
    truth: &[u8],
    fast_fn: &dyn Fn(&[u8], &mut [u8]),
    moxcms_src: &ColorProfile,
    moxcms_dst: &ColorProfile,
) -> AccuracyResult {
    let total = src.len() / 3;

    let mut fast_dst = vec![0u8; src.len()];
    let mut mox_dst = vec![0u8; src.len()];

    fast_fn(src, &mut fast_dst);
    let xform = moxcms_src
        .create_transform_8bit(Layout::Rgb, moxcms_dst, Layout::Rgb, moxcms_opts())
        .unwrap();
    xform.transform(src, &mut mox_dst).unwrap();

    let mut r = AccuracyResult {
        name,
        fast_exact: 0,
        fast_off1: 0,
        fast_off2plus: 0,
        fast_max_delta: 0,
        mox_exact: 0,
        mox_off1: 0,
        mox_off2plus: 0,
        mox_max_delta: 0,
        fast_better: 0,
        mox_better: 0,
        tied: 0,
        total,
    };

    for i in 0..total {
        let off = i * 3;
        let mut fast_dist: u16 = 0;
        let mut mox_dist: u16 = 0;

        for ch in 0..3 {
            let t = truth[off + ch];
            let f = fast_dst[off + ch];
            let m = mox_dst[off + ch];

            let fd = f.abs_diff(t);
            let md = m.abs_diff(t);
            fast_dist += fd as u16;
            mox_dist += md as u16;

            if fd > r.fast_max_delta {
                r.fast_max_delta = fd;
            }
            if md > r.mox_max_delta {
                r.mox_max_delta = md;
            }
        }

        // Per-pixel: count exact/off1/off2+
        let f_max_ch = (0..3)
            .map(|ch| fast_dst[off + ch].abs_diff(truth[off + ch]))
            .max()
            .unwrap();
        let m_max_ch = (0..3)
            .map(|ch| mox_dst[off + ch].abs_diff(truth[off + ch]))
            .max()
            .unwrap();

        match f_max_ch {
            0 => r.fast_exact += 1,
            1 => r.fast_off1 += 1,
            _ => r.fast_off2plus += 1,
        }
        match m_max_ch {
            0 => r.mox_exact += 1,
            1 => r.mox_off1 += 1,
            _ => r.mox_off2plus += 1,
        }

        // Who is closer (sum of absolute channel differences)?
        if fast_dist < mox_dist {
            r.fast_better += 1;
        } else if mox_dist < fast_dist {
            r.mox_better += 1;
        } else {
            r.tied += 1;
        }
    }

    r
}

/// Build a ZenCmsLite u8 RGB transform as a closure.
fn build_lite_u8_transform(
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

// =========================================================================
// Tests
// =========================================================================

#[test]
fn p3_to_srgb_accuracy() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let src = full_rgb_cube();
    let truth = ground_truth_u8(
        &src,
        &P3_TO_SRGB_F64,
        srgb_to_linear_f64,
        linear_to_srgb_f64,
    );
    let fast_fn = build_lite_u8_transform(
        ColorProfileSource::Named(NamedProfile::DisplayP3),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let r = compare_accuracy(
        "P3→sRGB",
        &src,
        &truth,
        &*fast_fn,
        &ColorProfile::new_display_p3(),
        &ColorProfile::new_srgb(),
    );
    r.print();
}

#[test]
fn bt2020_to_srgb_accuracy() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let src = full_rgb_cube();
    let truth = ground_truth_u8(
        &src,
        &BT2020_TO_SRGB_F64,
        bt709_to_linear_f64,
        linear_to_srgb_f64,
    );
    let fast_fn = build_lite_u8_transform(
        ColorProfileSource::Named(NamedProfile::Bt2020),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let r = compare_accuracy(
        "BT.2020 SDR→sRGB",
        &src,
        &truth,
        &*fast_fn,
        &ColorProfile::new_bt2020(),
        &ColorProfile::new_srgb(),
    );
    r.print();
}

#[test]
fn adobergb_to_srgb_accuracy() {
    use zenpixels_convert::{ColorProfileSource, NamedProfile};
    let src = full_rgb_cube();
    let truth = ground_truth_u8(
        &src,
        &ADOBERGB_TO_SRGB_F64,
        |v| v.powf(ADOBE_GAMMA),
        linear_to_srgb_f64,
    );
    let fast_fn = build_lite_u8_transform(
        ColorProfileSource::Named(NamedProfile::AdobeRgb),
        ColorProfileSource::Named(NamedProfile::Srgb),
    );
    let r = compare_accuracy(
        "AdobeRGB→sRGB",
        &src,
        &truth,
        &*fast_fn,
        &ColorProfile::new_adobe_rgb(),
        &ColorProfile::new_srgb(),
    );
    r.print();
}
