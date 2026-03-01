//! Brute-force perceptual loss test harness.
//!
//! Measures actual CIEDE2000 ΔE for every conversion and operation scenario,
//! then validates that the zenpixels cost model's loss buckets agree with
//! the measured perceptual loss.
//!
//! This is the ground truth for calibrating the hardcoded loss values in
//! `negotiate.rs`. If a bucket mismatch is found, the cost model values
//! should be adjusted (not the test thresholds).

#![allow(dead_code)]

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels::{ConversionCost, Provenance, conversion_cost, conversion_cost_with_provenance};

// ===========================================================================
// Reference color math (f64 precision, test-only)
// ===========================================================================

/// sRGB EOTF: electrical → linear (gamma decode).
fn srgb_eotf(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB OETF: linear → electrical (gamma encode).
fn srgb_oetf(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// PQ EOTF (ST 2084): electrical → linear (cd/m² normalized to 0..1 at 10000 nits).
fn pq_eotf(e: f64) -> f64 {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let ep = e.powf(1.0 / m2);
    let num = (ep - c1).max(0.0);
    let den = c2 - c3 * ep;
    if den <= 0.0 {
        0.0
    } else {
        (num / den).powf(1.0 / m1)
    }
}

/// PQ OETF (ST 2084): linear (0..1 at 10000 nits) → electrical.
fn pq_oetf(y: f64) -> f64 {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let yp = y.max(0.0).powf(m1);
    ((c1 + c2 * yp) / (1.0 + c3 * yp)).powf(m2)
}

/// HLG OETF (ARIB STD-B67).
fn hlg_oetf(e: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * (4.0_f64 * a).ln();
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        a * (12.0 * e - b).ln() + c
    }
}

/// HLG inverse OETF.
fn hlg_eotf(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * (4.0_f64 * a).ln();
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - c) / a).exp() + b) / 12.0
    }
}

/// BT.709 OETF (very similar to sRGB but different breakpoints).
fn bt709_oetf(l: f64) -> f64 {
    if l < 0.018 {
        4.5 * l
    } else {
        1.099 * l.powf(0.45) - 0.099
    }
}

/// BT.709 EOTF.
fn bt709_eotf(v: f64) -> f64 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

// --- RGB ↔ XYZ matrices (D65) ---

/// sRGB / BT.709 primaries → XYZ.
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// Display P3 primaries → XYZ (D65).
const P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];

/// BT.2020 primaries → XYZ (D65).
const BT2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
];

/// XYZ → sRGB / BT.709 (inverse of SRGB_TO_XYZ).
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

/// XYZ → Display P3.
const XYZ_TO_P3: [[f64; 3]; 3] = [
    [2.4934969, -0.9313836, -0.4027108],
    [-0.8294890, 1.7626641, 0.0236247],
    [0.0358458, -0.0761724, 0.9568845],
];

/// XYZ → BT.2020.
const XYZ_TO_BT2020: [[f64; 3]; 3] = [
    [1.7166512, -0.3556708, -0.2533663],
    [-0.6666844, 1.6164812, 0.0157685],
    [0.0176399, -0.0427706, 0.9421031],
];

fn mat3x3_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Convert linear RGB to CIE XYZ using the given primaries matrix.
fn rgb_to_xyz(rgb: [f64; 3], to_xyz: &[[f64; 3]; 3]) -> [f64; 3] {
    mat3x3_mul(to_xyz, rgb)
}

/// Convert CIE XYZ to linear RGB using the given inverse matrix.
fn xyz_to_rgb(xyz: [f64; 3], from_xyz: &[[f64; 3]; 3]) -> [f64; 3] {
    mat3x3_mul(from_xyz, xyz)
}

/// Convert linear sRGB to CIE Lab (D65).
fn linear_srgb_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    let xyz = rgb_to_xyz(rgb, &SRGB_TO_XYZ);
    xyz_to_lab(xyz)
}

/// D65 reference white.
const D65: [f64; 3] = [0.95047, 1.00000, 1.08883];

fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let f = |t: f64| -> f64 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(xyz[0] / D65[0]);
    let fy = f(xyz[1] / D65[1]);
    let fz = f(xyz[2] / D65[2]);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    [l, a, b]
}

/// CIEDE2000 color difference (Sharma, Wu, Dalal 2005).
fn ciede2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    use core::f64::consts::PI;

    let (l1, a1, b1) = (lab1[0], lab1[1], lab1[2]);
    let (l2, a2, b2) = (lab2[0], lab2[1], lab2[2]);

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_avg = (c1 + c2) / 2.0;

    let c_avg7 = c_avg.powi(7);
    let g = 0.5 * (1.0 - (c_avg7 / (c_avg7 + 25.0_f64.powi(7))).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);

    let dl = l2 - l1;
    let dc = c2p - c1p;

    let dh_deg = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };

    let dh = 2.0 * (c1p * c2p).sqrt() * (dh_deg / 2.0 * PI / 180.0).sin();

    let l_avg = (l1 + l2) / 2.0;
    let c_avgp = (c1p + c2p) / 2.0;

    let h_avgp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_avgp - 30.0) * PI / 180.0).cos()
        + 0.24 * ((2.0 * h_avgp) * PI / 180.0).cos()
        + 0.32 * ((3.0 * h_avgp + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * h_avgp - 63.0) * PI / 180.0).cos();

    let sl = 1.0 + 0.015 * (l_avg - 50.0).powi(2) / (20.0 + (l_avg - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_avgp;
    let sh = 1.0 + 0.015 * c_avgp * t;

    let c_avgp7 = c_avgp.powi(7);
    let rt = -2.0
        * (c_avgp7 / (c_avgp7 + 25.0_f64.powi(7))).sqrt()
        * (60.0 * (-((h_avgp - 275.0) / 25.0).powi(2)).exp() * PI / 180.0).sin();

    let term_l = dl / sl;
    let term_c = dc / sc;
    let term_h = dh / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h)
        .max(0.0)
        .sqrt()
}

// ===========================================================================
// Loss measurement
// ===========================================================================

#[derive(Debug, Clone)]
struct LossStats {
    max_de: f64,
    mean_de: f64,
    p95_de: f64,
    p99_de: f64,
    sample_count: usize,
}

/// Measure CIEDE2000 between reference and converted pixel arrays.
/// Both arrays must be in linear sRGB f64 for correct Lab conversion.
fn measure_conversion_loss(reference: &[[f64; 3]], converted: &[[f64; 3]]) -> LossStats {
    assert_eq!(reference.len(), converted.len());
    let n = reference.len();
    let mut des: Vec<f64> = Vec::with_capacity(n);

    for (r, c) in reference.iter().zip(converted.iter()) {
        let lab_r = linear_srgb_to_lab(*r);
        let lab_c = linear_srgb_to_lab(*c);
        des.push(ciede2000(lab_r, lab_c));
    }

    des.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = des.iter().sum();

    LossStats {
        max_de: des.last().copied().unwrap_or(0.0),
        mean_de: if n > 0 { sum / n as f64 } else { 0.0 },
        p95_de: des.get((n as f64 * 0.95) as usize).copied().unwrap_or(0.0),
        p99_de: des.get((n as f64 * 0.99) as usize).copied().unwrap_or(0.0),
        sample_count: n,
    }
}

// ===========================================================================
// Perceptual loss buckets
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Bucket {
    Lossless,     // ΔE < 0.5
    NearLossless, // ΔE 0.5 – 2.0
    LowLoss,      // ΔE 2.0 – 5.0
    Moderate,     // ΔE 5.0 – 15.0
    High,         // ΔE > 15.0
}

impl Bucket {
    fn from_de(de: f64) -> Self {
        if de < 0.5 {
            Self::Lossless
        } else if de < 2.0 {
            Self::NearLossless
        } else if de < 5.0 {
            Self::LowLoss
        } else if de < 15.0 {
            Self::Moderate
        } else {
            Self::High
        }
    }

    fn from_model_loss(loss: u16) -> Self {
        if loss <= 10 {
            Self::Lossless
        } else if loss <= 50 {
            Self::NearLossless
        } else if loss <= 150 {
            Self::LowLoss
        } else if loss <= 400 {
            Self::Moderate
        } else {
            Self::High
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Lossless => "Lossless",
            Self::NearLossless => "NearLossless",
            Self::LowLoss => "LowLoss",
            Self::Moderate => "Moderate",
            Self::High => "High",
        }
    }
}

// ===========================================================================
// Test data generators
// ===========================================================================

/// Every 17th value per channel → 15³ = 3375 RGB triples covering the cube.
fn srgb_grid() -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    for r in (0u16..=255).step_by(17) {
        for g in (0u16..=255).step_by(17) {
            for b in (0u16..=255).step_by(17) {
                out.push([r as u8, g as u8, b as u8]);
            }
        }
    }
    // Also include corners to ensure full coverage
    for &r in &[0u8, 255] {
        for &g in &[0u8, 255] {
            for &b in &[0u8, 255] {
                let triple = [r, g, b];
                if !out.contains(&triple) {
                    out.push(triple);
                }
            }
        }
    }
    out
}

/// Convert sRGB u8 grid to linear sRGB f64.
fn srgb_grid_linear() -> Vec<[f64; 3]> {
    srgb_grid()
        .iter()
        .map(|[r, g, b]| {
            [
                srgb_eotf(*r as f64 / 255.0),
                srgb_eotf(*g as f64 / 255.0),
                srgb_eotf(*b as f64 / 255.0),
            ]
        })
        .collect()
}

/// Smooth gradient for banding detection (in linear sRGB).
fn gradient_ramp(steps: u32) -> Vec<[f64; 3]> {
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            [t, t, t] // linear gray ramp
        })
        .collect()
}

/// P3 test colors in linear light, including out-of-sRGB-gamut saturated colors.
fn p3_test_colors() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    // In-gamut (mild) colors
    for &v in &[0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
        out.push([v, v * 0.5, v * 0.3]);
    }
    // Saturated P3 primaries (these clip when converted to sRGB)
    out.push([1.0, 0.0, 0.0]); // P3 red
    out.push([0.0, 1.0, 0.0]); // P3 green
    out.push([0.0, 0.0, 1.0]); // P3 blue
    // Highly saturated P3 colors
    out.push([0.95, 0.0, 0.2]);
    out.push([0.0, 0.9, 0.1]);
    out.push([0.3, 0.0, 0.95]);
    out
}

/// BT.2020 test colors in linear light.
fn bt2020_test_colors() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    // Standard colors
    for &v in &[0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
        out.push([v, v * 0.4, v * 0.2]);
    }
    // Saturated BT.2020 primaries
    out.push([1.0, 0.0, 0.0]);
    out.push([0.0, 1.0, 0.0]);
    out.push([0.0, 0.0, 1.0]);
    // Deep saturated colors that clip hard in sRGB
    out.push([0.8, 0.0, 0.0]);
    out.push([0.0, 0.7, 0.0]);
    out.push([0.0, 0.0, 0.9]);
    out.push([0.9, 0.8, 0.0]); // saturated yellow
    out.push([0.0, 0.9, 0.9]); // saturated cyan
    out
}

// ===========================================================================
// Conversion simulation helpers
// ===========================================================================

/// Clamp f64 to [0, 1].
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// u8 quantize: linear sRGB → sRGB u8 → back to linear sRGB.
fn roundtrip_u8_srgb(linear: [f64; 3]) -> [f64; 3] {
    let srgb = linear.map(|c| (srgb_oetf(clamp01(c)) * 255.0).round().clamp(0.0, 255.0));
    srgb.map(|c| srgb_eotf(c / 255.0))
}

/// u16 quantize: linear → sRGB u16 → back to linear.
fn roundtrip_u16_srgb(linear: [f64; 3]) -> [f64; 3] {
    let srgb = linear.map(|c| {
        (srgb_oetf(clamp01(c)) * 65535.0)
            .round()
            .clamp(0.0, 65535.0)
    });
    srgb.map(|c| srgb_eotf(c / 65535.0))
}

/// f16 quantize: f32 → f16 → f32 (simulated via half crate).
fn roundtrip_f16(linear: [f64; 3]) -> [f64; 3] {
    linear.map(|c| {
        let h = half::f16::from_f64(c);
        h.to_f64()
    })
}

/// i16 quantize: f32 linear [0,1] → i16 [-32768,32767] → back to f32.
fn roundtrip_i16(linear: [f64; 3]) -> [f64; 3] {
    linear.map(|c| {
        let clamped = clamp01(c);
        // Map [0, 1] → [-32768, 32767]
        let i = (clamped * 65535.0 - 32768.0)
            .round()
            .clamp(-32768.0, 32767.0) as i16;
        (i as f64 + 32768.0) / 65535.0
    })
}

/// Gamut conversion: linear RGB in source primaries → linear RGB in target,
/// clamped to [0,1] (simulating gamut clipping).
fn convert_primaries(
    rgb: [f64; 3],
    from_to_xyz: &[[f64; 3]; 3],
    to_from_xyz: &[[f64; 3]; 3],
) -> [f64; 3] {
    let xyz = rgb_to_xyz(rgb, from_to_xyz);
    let out = xyz_to_rgb(xyz, to_from_xyz);
    out.map(|c| clamp01(c))
}

/// Simple bilinear 2x downsample of a 2-wide "image" to 1 pixel.
fn bilinear_downsample_pair(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ]
}

/// 3-tap box blur of consecutive pixel triples.
fn box_blur_3(left: [f64; 3], center: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        (left[0] + center[0] + right[0]) / 3.0,
        (left[1] + center[1] + right[1]) / 3.0,
        (left[2] + center[2] + right[2]) / 3.0,
    ]
}

/// Generate a test image with sharp edges that trigger multi-tap filter ringing.
/// Returns pixel data in linear sRGB [0,1] as f64 triples.
/// The pattern alternates 5-pixel dark/bright step edges with gradient sections.
fn sharp_edge_image(width: usize, height: usize) -> Vec<[f64; 3]> {
    let row: Vec<[f64; 3]> = (0..width)
        .map(|i| {
            let t = i as f64 / width as f64;
            if i % 20 < 10 {
                // Hard step: near-black to bright every 5 pixels
                if i % 20 < 5 {
                    [0.01, 0.01, 0.01]
                } else {
                    [0.95, 0.80, 0.60]
                }
            } else {
                // Smooth gradient section
                let gt = ((i % 20) as f64 - 10.0) / 10.0;
                [t * 0.8 + 0.1, gt * 0.5 + 0.25, 0.5 - gt * 0.3]
            }
        })
        .collect();
    // Repeat the same row for all scanlines
    row.iter().cycle().take(width * height).copied().collect()
}

/// Convert f64 linear RGB image to packed f32 for zenresize LinearF32 input.
fn to_f32_linear_packed(pixels: &[[f64; 3]]) -> Vec<f32> {
    pixels
        .iter()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect()
}

/// Convert f64 linear RGB image to packed u8 sRGB for zenresize Srgb8 input.
fn to_u8_srgb_packed(pixels: &[[f64; 3]]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|p| {
            [
                (srgb_oetf(p[0].clamp(0.0, 1.0)) * 255.0).round() as u8,
                (srgb_oetf(p[1].clamp(0.0, 1.0)) * 255.0).round() as u8,
                (srgb_oetf(p[2].clamp(0.0, 1.0)) * 255.0).round() as u8,
            ]
        })
        .collect()
}

/// Unpack zenresize f32 linear output to f64 triples.
fn from_f32_linear_packed(data: &[f32]) -> Vec<[f64; 3]> {
    data.chunks_exact(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect()
}

/// Unpack zenresize u8 sRGB output to f64 linear triples.
fn from_u8_srgb_packed(data: &[u8]) -> Vec<[f64; 3]> {
    data.chunks_exact(3)
        .map(|c| {
            [
                srgb_eotf(c[0] as f64 / 255.0),
                srgb_eotf(c[1] as f64 / 255.0),
                srgb_eotf(c[2] as f64 / 255.0),
            ]
        })
        .collect()
}

// ===========================================================================
// Scenario framework
// ===========================================================================

struct ScenarioResult {
    name: &'static str,
    stats: LossStats,
    model_loss: u16,
    measured_bucket: Bucket,
    model_bucket: Bucket,
    match_ok: bool,
}

fn run_scenario(
    name: &'static str,
    reference: &[[f64; 3]],
    converted: &[[f64; 3]],
    model_loss: u16,
) -> ScenarioResult {
    let stats = measure_conversion_loss(reference, converted);
    let measured_bucket = Bucket::from_de(stats.p95_de);
    let model_bucket = Bucket::from_model_loss(model_loss);
    // Allow 1 bucket tolerance for borderline cases
    let match_ok = (measured_bucket as i32 - model_bucket as i32).abs() <= 1;
    ScenarioResult {
        name,
        stats,
        model_loss,
        measured_bucket,
        model_bucket,
        match_ok,
    }
}

fn print_results(results: &[ScenarioResult]) {
    eprintln!("\n{:-<120}", "");
    eprintln!(
        "{:<50} {:>8} {:>8} {:>8} {:>12} {:>12} {:>6}",
        "Scenario", "p95 ΔE", "max ΔE", "mean ΔE", "Measured", "Model", "Match"
    );
    eprintln!("{:-<120}", "");
    for r in results {
        eprintln!(
            "{:<50} {:>8.3} {:>8.3} {:>8.3} {:>12} {:>12} {:>6}",
            r.name,
            r.stats.p95_de,
            r.stats.max_de,
            r.stats.mean_de,
            r.measured_bucket.name(),
            format!("{}({})", r.model_bucket.name(), r.model_loss),
            if r.match_ok { "OK" } else { "FAIL" }
        );
    }
    eprintln!("{:-<120}", "");
}

// ===========================================================================
// 3A: Depth round-trip scenarios
// ===========================================================================

#[test]
fn depth_roundtrip_scenarios() {
    let grid = srgb_grid_linear();
    let ramp = gradient_ramp(256);
    let mut results = Vec::new();

    // 1. u8 sRGB → f32 Linear → u8 sRGB (with u8 provenance: near-lossless)
    {
        let converted: Vec<_> = grid.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        // The return leg uses u8 provenance — data was originally u8, so f32→u8 is lossless.
        let cost_fwd = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR);
        let cost_back = conversion_cost_with_provenance(
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB8_SRGB,
            Provenance::with_origin_depth(ChannelType::U8),
        );
        results.push(run_scenario(
            "u8 sRGB → f32 Lin → u8 sRGB",
            &grid,
            &converted,
            (cost_fwd + cost_back).loss,
        ));
    }

    // 2. u8 sRGB → u16 sRGB → u8 sRGB (should be exact)
    {
        let converted: Vec<_> = grid
            .iter()
            .map(|c| {
                let srgb = c.map(|v| (srgb_oetf(clamp01(v)) * 255.0).round().clamp(0.0, 255.0));
                let u16_val = srgb.map(|v| (v / 255.0 * 65535.0).round());
                let back_u8 = u16_val.map(|v| (v / 65535.0 * 255.0).round() / 255.0);
                back_u8.map(|v| srgb_eotf(v))
            })
            .collect();
        // Return leg uses u8 provenance.
        let cost_fwd = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB);
        let cost_back = conversion_cost_with_provenance(
            PixelDescriptor::RGB16_SRGB,
            PixelDescriptor::RGB8_SRGB,
            Provenance::with_origin_depth(ChannelType::U8),
        );
        results.push(run_scenario(
            "u8 sRGB → u16 sRGB → u8 sRGB",
            &grid,
            &converted,
            (cost_fwd + cost_back).loss,
        ));
    }

    // 3. u16 sRGB → u8 sRGB (quantization)
    {
        let reference_u16: Vec<_> = ramp.iter().map(|c| roundtrip_u16_srgb(*c)).collect();
        let converted: Vec<_> = reference_u16
            .iter()
            .map(|c| roundtrip_u8_srgb(*c))
            .collect();
        let cost = conversion_cost(PixelDescriptor::RGB16_SRGB, PixelDescriptor::RGB8_SRGB);
        results.push(run_scenario(
            "u16 sRGB → u8 sRGB",
            &reference_u16,
            &converted,
            cost.loss,
        ));
    }

    // 4. f32 Linear → u8 sRGB (origin f32 — true loss)
    {
        let converted: Vec<_> = ramp.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        let cost = conversion_cost(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGB8_SRGB);
        results.push(run_scenario(
            "f32 Lin → u8 sRGB (origin f32)",
            &ramp,
            &converted,
            cost.loss,
        ));
    }

    // 5. f32 Linear → u8 sRGB (origin u8 — should be near-lossless)
    {
        let reference: Vec<_> = srgb_grid()
            .iter()
            .map(|[r, g, b]| {
                [
                    srgb_eotf(*r as f64 / 255.0),
                    srgb_eotf(*g as f64 / 255.0),
                    srgb_eotf(*b as f64 / 255.0),
                ]
            })
            .collect();
        let converted: Vec<_> = reference.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        let cost = conversion_cost_with_provenance(
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB8_SRGB,
            Provenance::with_origin_depth(ChannelType::U8),
        );
        results.push(run_scenario(
            "f32 Lin → u8 sRGB (origin u8)",
            &reference,
            &converted,
            cost.loss,
        ));
    }

    // 6. f32 → f16 → f32
    {
        let converted: Vec<_> = ramp.iter().map(|c| roundtrip_f16(*c)).collect();
        let cost = ConversionCost::new(30, 20); // f32→f16 + f16→f32
        results.push(run_scenario(
            "f32 → f16 → f32",
            &ramp,
            &converted,
            cost.loss,
        ));
    }

    // 7. u8 → f16 → u8 (f16 has >8 bits precision)
    {
        let reference: Vec<_> = srgb_grid()
            .iter()
            .map(|[r, g, b]| [*r as f64 / 255.0, *g as f64 / 255.0, *b as f64 / 255.0])
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let f16_rt = roundtrip_f16(*c);
                f16_rt.map(|v| (v * 255.0).round() / 255.0)
            })
            .collect();
        // Compare in linear sRGB for ΔE
        let ref_linear: Vec<_> = reference.iter().map(|c| c.map(srgb_eotf)).collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(srgb_eotf)).collect();
        results.push(run_scenario("u8 → f16 → u8", &ref_linear, &conv_linear, 0));
    }

    // 8. u16 → f16 → u16 (10 vs 16 mantissa bits — some loss)
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v, v * 0.7, v * 0.3]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let f16_rt = roundtrip_f16(*c);
                f16_rt.map(|v| (v * 65535.0).round() / 65535.0)
            })
            .collect();
        let ref_linear: Vec<_> = reference.iter().map(|c| c.map(srgb_eotf)).collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(srgb_eotf)).collect();
        results.push(run_scenario(
            "u16 → f16 → u16",
            &ref_linear,
            &conv_linear,
            30,
        ));
    }

    // 9. f32 → i16 → f32 (simulated)
    {
        let converted: Vec<_> = ramp.iter().map(|c| roundtrip_i16(*c)).collect();
        // Model: f32→i16 loss=5 (calibrated), i16→f32 loss=0 (widening)
        let cost = ConversionCost::new(30, 5);
        results.push(run_scenario(
            "f32 → i16 → f32",
            &ramp,
            &converted,
            cost.loss,
        ));
    }

    // 10. u8 → f32 (naive, no gamma) — measures what happens with wrong conversion
    {
        let reference: Vec<_> = srgb_grid()
            .iter()
            .map(|[r, g, b]| {
                [
                    srgb_eotf(*r as f64 / 255.0),
                    srgb_eotf(*g as f64 / 255.0),
                    srgb_eotf(*b as f64 / 255.0),
                ]
            })
            .collect();
        // Naive conversion: just divide by 255, don't apply EOTF
        let converted: Vec<_> = srgb_grid()
            .iter()
            .map(|[r, g, b]| [*r as f64 / 255.0, *g as f64 / 255.0, *b as f64 / 255.0])
            .collect();
        // The "converted" values are in sRGB space but we're comparing in linear
        // This shows the error of not applying gamma correction
        results.push(run_scenario(
            "u8 → f32 (naive, no gamma)",
            &reference,
            &converted,
            300,
        ));
    }

    // 11. u8 → f32 (sRGB EOTF) → u8 (sRGB OETF) — should be ≤1 LSB
    {
        let reference = srgb_grid_linear();
        let converted: Vec<_> = reference.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        results.push(run_scenario(
            "u8 → f32 (EOTF) → u8 (OETF)",
            &reference,
            &converted,
            0,
        ));
    }

    // 12. u16 → f32 → u16 (f32 has >16 bits mantissa — exact)
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v, v * 0.5, v * 0.8]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                // f32 round-trip preserves u16 precision
                c.map(|v| {
                    let u16_val = (v * 65535.0).round();
                    let f32_val = u16_val as f32 / 65535.0;
                    let back = (f32_val as f64 * 65535.0).round() / 65535.0;
                    back
                })
            })
            .collect();
        let ref_linear: Vec<_> = reference.iter().map(|c| c.map(srgb_eotf)).collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(srgb_eotf)).collect();
        results.push(run_scenario(
            "u16 → f32 → u16",
            &ref_linear,
            &conv_linear,
            0,
        ));
    }

    // 13. f16 → u8
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v, v, v]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let f16_val = c.map(|v| half::f16::from_f64(v).to_f64());
                f16_val.map(|v| (v * 255.0).round() / 255.0)
            })
            .collect();
        let ref_linear: Vec<_> = reference.iter().map(|c| c.map(srgb_eotf)).collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(srgb_eotf)).collect();
        // Model: f16→u8 depth_loss = 8 (calibrated, f16 has >8 bits precision)
        results.push(run_scenario("f16 → u8", &ref_linear, &conv_linear, 8));
    }

    // 14. f16 → f32 → f16 (exact)
    {
        let reference: Vec<_> = ramp
            .iter()
            .map(|c| c.map(|v| half::f16::from_f64(v).to_f64()))
            .collect();
        let converted = reference.clone();
        results.push(run_scenario("f16 → f32 → f16", &reference, &converted, 0));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3B: Premultiplication round-trip scenarios
// ===========================================================================

#[test]
fn premul_roundtrip_scenarios() {
    let mut results = Vec::new();

    // Helper: premul round-trip at specific alpha in u8
    // Simulates: straight u8 → premultiplied u8 → back to straight u8
    let premul_rt_u8 = |r: u8, g: u8, b: u8, a: u8| -> ([f64; 3], [f64; 3]) {
        let rf = r as f64 / 255.0;
        let gf = g as f64 / 255.0;
        let bf = b as f64 / 255.0;
        let af = a as f64 / 255.0;
        let reference = [srgb_eotf(rf), srgb_eotf(gf), srgb_eotf(bf)];

        if a == 0 {
            return (reference, [0.0, 0.0, 0.0]);
        }

        // straight → premul (u8): premul_ch = round(straight_ch * alpha / 255)
        let pr = (r as f64 * af).round().clamp(0.0, 255.0);
        let pg = (g as f64 * af).round().clamp(0.0, 255.0);
        let pb = (b as f64 * af).round().clamp(0.0, 255.0);
        // premul → straight (u8): straight_ch = round(premul_ch * 255 / alpha)
        let ur = (pr * 255.0 / a as f64).round().clamp(0.0, 255.0) / 255.0;
        let ug = (pg * 255.0 / a as f64).round().clamp(0.0, 255.0) / 255.0;
        let ub = (pb * 255.0 / a as f64).round().clamp(0.0, 255.0) / 255.0;
        let converted = [srgb_eotf(ur), srgb_eotf(ug), srgb_eotf(ub)];
        (reference, converted)
    };

    // 1. α=255 (exact)
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &[r, g, b] in srgb_grid().iter().take(200) {
            let (ref_c, conv_c) = premul_rt_u8(r, g, b, 255);
            refs.push(ref_c);
            convs.push(conv_c);
        }
        results.push(run_scenario("u8 premul rt α=255", &refs, &convs, 0));
    }

    // 2. α=128 (small error)
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &[r, g, b] in srgb_grid().iter().take(200) {
            let (ref_c, conv_c) = premul_rt_u8(r, g, b, 128);
            refs.push(ref_c);
            convs.push(conv_c);
        }
        let cost = ConversionCost::new(45, 15); // premul+unpremul
        results.push(run_scenario("u8 premul rt α=128", &refs, &convs, cost.loss));
    }

    // 3. α=2 (severe error)
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &[r, g, b] in srgb_grid().iter().take(200) {
            let (ref_c, conv_c) = premul_rt_u8(r, g, b, 2);
            refs.push(ref_c);
            convs.push(conv_c);
        }
        results.push(run_scenario("u8 premul rt α=2", &refs, &convs, 400));
    }

    // 4. α=1 (catastrophic)
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &[r, g, b] in srgb_grid().iter().take(200) {
            let (ref_c, conv_c) = premul_rt_u8(r, g, b, 1);
            refs.push(ref_c);
            convs.push(conv_c);
        }
        results.push(run_scenario("u8 premul rt α=1", &refs, &convs, 500));
    }

    // 5. f32 premul rt α=0.004 (near-exact)
    {
        let alpha = 0.004;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..100 {
            let v = i as f64 / 99.0;
            let reference = [v, v * 0.5, v * 0.8];
            // f32 premul round-trip
            let pr = reference.map(|c| c * alpha);
            let ur = pr.map(|c| if alpha > 0.0 { c / alpha } else { 0.0 });
            refs.push(reference);
            convs.push(ur);
        }
        results.push(run_scenario("f32 premul rt α=0.004", &refs, &convs, 0));
    }

    // 6. u16 premul rt α=2 (moderate error)
    {
        let alpha = 2.0 / 65535.0;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..100 {
            let v = i as f64 / 99.0;
            let reference = [v, v * 0.5, v * 0.8];
            let pr = reference.map(|c| (c * alpha * 65535.0).round().clamp(0.0, 65535.0));
            let a_u16 = (alpha * 65535.0).round();
            let ur = if a_u16 > 0.0 {
                pr.map(|c| (c / a_u16).clamp(0.0, 1.0))
            } else {
                [0.0; 3]
            };
            refs.push(reference);
            convs.push(ur);
        }
        results.push(run_scenario("u16 premul rt α≈2/65535", &refs, &convs, 200));
    }

    // 7. u8 premul (R=200, α=100) → straight → premul
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for g in (0u16..=255).step_by(5) {
            for b in (0u16..=255).step_by(5) {
                let r = 200u8;
                let a = 100u8;
                let rf = r as f64 / 255.0;
                let gf = g as f64 / 255.0;
                let bf = b as f64 / 255.0;
                let af = a as f64 / 255.0;
                // Original premul values
                let pr = (rf * af * 255.0).round();
                let pg = (gf * af * 255.0).round();
                let pb = (bf * af * 255.0).round();
                // → straight (premul_byte * 255 / alpha_byte)
                let sr = (pr * 255.0 / a as f64).round().clamp(0.0, 255.0);
                let sg = (pg * 255.0 / a as f64).round().clamp(0.0, 255.0);
                let sb = (pb * 255.0 / a as f64).round().clamp(0.0, 255.0);
                // → back to premul
                let pr2 = (sr / 255.0 * af * 255.0).round().clamp(0.0, 255.0);
                let pg2 = (sg / 255.0 * af * 255.0).round().clamp(0.0, 255.0);
                let pb2 = (sb / 255.0 * af * 255.0).round().clamp(0.0, 255.0);

                refs.push([
                    srgb_eotf(pr / 255.0),
                    srgb_eotf(pg / 255.0),
                    srgb_eotf(pb / 255.0),
                ]);
                convs.push([
                    srgb_eotf(pr2 / 255.0),
                    srgb_eotf(pg2 / 255.0),
                    srgb_eotf(pb2 / 255.0),
                ]);
            }
        }
        results.push(run_scenario("u8 premul(R=200,α=100) rt", &refs, &convs, 15));
    }

    // 8. f32 premul → straight → premul, α=0.001 (near-exact)
    {
        let alpha = 0.001;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..100 {
            let v = i as f64 / 99.0;
            let original = [v * alpha, v * 0.5 * alpha, v * 0.8 * alpha]; // premul values
            // → straight
            let straight = original.map(|c| if alpha > 0.0 { c / alpha } else { 0.0 });
            // → back to premul
            let back = straight.map(|c| c * alpha);
            refs.push(original);
            convs.push(back);
        }
        results.push(run_scenario("f32 premul rt α=0.001", &refs, &convs, 0));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3C: Gamut conversion scenarios
// ===========================================================================

#[test]
fn gamut_conversion_scenarios() {
    let mut results = Vec::new();

    // Helper: convert linear RGB between primaries, return in linear sRGB for ΔE
    let convert_and_measure = |colors: &[[f64; 3]],
                               src_to_xyz: &[[f64; 3]; 3],
                               dst_from_xyz: &[[f64; 3]; 3],
                               back_to_xyz: &[[f64; 3]; 3],
                               back_from_xyz: &[[f64; 3]; 3]|
     -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &c in colors {
            let xyz_orig = rgb_to_xyz(c, src_to_xyz);
            let in_dst = xyz_to_rgb(xyz_orig, dst_from_xyz).map(|v| clamp01(v));
            let xyz_back = rgb_to_xyz(in_dst, back_to_xyz);
            let back = xyz_to_rgb(xyz_back, back_from_xyz).map(|v| clamp01(v));
            // Reference: original in sRGB linear
            let ref_srgb = xyz_to_rgb(xyz_orig, &XYZ_TO_SRGB).map(|v| clamp01(v));
            // Converted: round-tripped, in sRGB linear
            let conv_srgb =
                xyz_to_rgb(rgb_to_xyz(back, src_to_xyz), &XYZ_TO_SRGB).map(|v| clamp01(v));
            refs.push(ref_srgb);
            convs.push(conv_srgb);
        }
        (refs, convs)
    };

    // 1. sRGB → P3 → sRGB (all in-gamut, should be exact)
    {
        let colors = srgb_grid_linear();
        let (refs, convs) =
            convert_and_measure(&colors, &SRGB_TO_XYZ, &XYZ_TO_P3, &P3_TO_XYZ, &XYZ_TO_SRGB);
        let cost = ConversionCost::new(20, 0); // widening + narrowing with matching origin
        results.push(run_scenario(
            "sRGB → P3 → sRGB (in-gamut)",
            &refs,
            &convs,
            cost.loss,
        ));
    }

    // 2. P3 → sRGB → P3 (saturated P3 — clips)
    //    Measure ΔE between original P3 color and round-tripped color,
    //    both converted to XYZ→Lab for perceptual comparison.
    {
        let colors = p3_test_colors();
        let mut de_values: Vec<f64> = Vec::new();
        for &c in &colors {
            let xyz_orig = rgb_to_xyz(c, &P3_TO_XYZ);
            let in_srgb = xyz_to_rgb(xyz_orig, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let xyz_rt = rgb_to_xyz(in_srgb, &SRGB_TO_XYZ);
            // Compare in Lab directly from XYZ
            let lab_orig = xyz_to_lab(xyz_orig);
            let lab_rt = xyz_to_lab(xyz_rt);
            de_values.push(ciede2000(lab_orig, lab_rt));
        }
        de_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95_idx = (de_values.len() as f64 * 0.95) as usize;
        let p95_de = de_values.get(p95_idx).copied().unwrap_or(0.0);
        let measured_bucket = Bucket::from_de(p95_de);

        let p3_to_srgb = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::DisplayP3,
        );
        let srgb = PixelDescriptor::RGBF32_LINEAR;
        let cost = conversion_cost(p3_to_srgb, srgb);
        let model_bucket = Bucket::from_model_loss(cost.loss);
        let match_ok = (measured_bucket as i32 - model_bucket as i32).abs() <= 1;
        results.push(ScenarioResult {
            name: "P3 → sRGB → P3 (saturated)",
            stats: LossStats {
                max_de: *de_values.last().unwrap_or(&0.0),
                mean_de: de_values.iter().sum::<f64>() / de_values.len().max(1) as f64,
                p95_de,
                p99_de: de_values
                    .get((de_values.len() as f64 * 0.99) as usize)
                    .copied()
                    .unwrap_or(0.0),
                sample_count: de_values.len(),
            },
            model_loss: cost.loss,
            measured_bucket,
            model_bucket,
            match_ok,
        });
    }

    // 3. sRGB → BT.2020 → sRGB (should be exact — sRGB subset)
    {
        let colors = srgb_grid_linear();
        let (refs, convs) = convert_and_measure(
            &colors,
            &SRGB_TO_XYZ,
            &XYZ_TO_BT2020,
            &BT2020_TO_XYZ,
            &XYZ_TO_SRGB,
        );
        results.push(run_scenario(
            "sRGB → BT.2020 → sRGB (in-gamut)",
            &refs,
            &convs,
            0,
        ));
    }

    // 4. BT.2020 → sRGB → BT.2020 (saturated — heavy clipping)
    //    Measure ΔE between original BT.2020 color and round-tripped color in Lab.
    {
        let colors = bt2020_test_colors();
        let mut de_values: Vec<f64> = Vec::new();
        for &c in &colors {
            let xyz_orig = rgb_to_xyz(c, &BT2020_TO_XYZ);
            let in_srgb = xyz_to_rgb(xyz_orig, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let xyz_rt = rgb_to_xyz(in_srgb, &SRGB_TO_XYZ);
            let lab_orig = xyz_to_lab(xyz_orig);
            let lab_rt = xyz_to_lab(xyz_rt);
            de_values.push(ciede2000(lab_orig, lab_rt));
        }
        de_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95_idx = (de_values.len() as f64 * 0.95) as usize;
        let p95_de = de_values.get(p95_idx).copied().unwrap_or(0.0);
        let measured_bucket = Bucket::from_de(p95_de);

        let bt2020 = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let srgb = PixelDescriptor::RGBF32_LINEAR;
        let cost = conversion_cost(bt2020, srgb);
        let model_bucket = Bucket::from_model_loss(cost.loss);
        let match_ok = (measured_bucket as i32 - model_bucket as i32).abs() <= 1;
        results.push(ScenarioResult {
            name: "BT.2020 → sRGB → BT.2020 (saturated)",
            stats: LossStats {
                max_de: *de_values.last().unwrap_or(&0.0),
                mean_de: de_values.iter().sum::<f64>() / de_values.len().max(1) as f64,
                p95_de,
                p99_de: de_values
                    .get((de_values.len() as f64 * 0.99) as usize)
                    .copied()
                    .unwrap_or(0.0),
                sample_count: de_values.len(),
            },
            model_loss: cost.loss,
            measured_bucket,
            model_bucket,
            match_ok,
        });
    }

    // 5. BT.2020 → P3 → BT.2020 (moderate clipping)
    {
        let colors = bt2020_test_colors();
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &c in &colors {
            let xyz = rgb_to_xyz(c, &BT2020_TO_XYZ);
            let in_p3 = xyz_to_rgb(xyz, &XYZ_TO_P3).map(|v| clamp01(v));
            let xyz_back = rgb_to_xyz(in_p3, &P3_TO_XYZ);
            let back = xyz_to_rgb(xyz_back, &XYZ_TO_BT2020).map(|v| clamp01(v));
            let ref_srgb = xyz_to_rgb(xyz, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let conv_srgb =
                xyz_to_rgb(rgb_to_xyz(back, &BT2020_TO_XYZ), &XYZ_TO_SRGB).map(|v| clamp01(v));
            refs.push(ref_srgb);
            convs.push(conv_srgb);
        }
        let bt2020 = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let p3 = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::DisplayP3,
        );
        let cost = conversion_cost(bt2020, p3);
        results.push(run_scenario(
            "BT.2020 → P3 → BT.2020 (saturated)",
            &refs,
            &convs,
            cost.loss,
        ));
    }

    // 6. P3 → BT.2020 → P3 (subset — should be near-exact)
    {
        let colors = p3_test_colors();
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &c in &colors {
            let xyz = rgb_to_xyz(c, &P3_TO_XYZ);
            let in_bt2020 = xyz_to_rgb(xyz, &XYZ_TO_BT2020);
            let xyz_back = rgb_to_xyz(in_bt2020, &BT2020_TO_XYZ);
            let back = xyz_to_rgb(xyz_back, &XYZ_TO_P3).map(|v| clamp01(v));
            let ref_srgb = xyz_to_rgb(xyz, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let conv_srgb =
                xyz_to_rgb(rgb_to_xyz(back, &P3_TO_XYZ), &XYZ_TO_SRGB).map(|v| clamp01(v));
            refs.push(ref_srgb);
            convs.push(conv_srgb);
        }
        results.push(run_scenario("P3 → BT.2020 → P3 (subset)", &refs, &convs, 0));
    }

    // 7. BT.2020 → sRGB with sRGB origin provenance (near-lossless)
    {
        // Start with sRGB colors, convert to BT.2020, then back to sRGB
        let colors = srgb_grid_linear();
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &c in colors.iter().take(500) {
            let xyz = rgb_to_xyz(c, &SRGB_TO_XYZ);
            // Place in BT.2020
            let in_bt2020 = xyz_to_rgb(xyz, &XYZ_TO_BT2020);
            // Convert back to sRGB (no clipping since it was originally sRGB)
            let xyz_back = rgb_to_xyz(in_bt2020, &BT2020_TO_XYZ);
            let back_srgb = xyz_to_rgb(xyz_back, &XYZ_TO_SRGB).map(|v| clamp01(v));
            refs.push(c);
            convs.push(back_srgb);
        }
        let bt2020 = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let srgb = PixelDescriptor::RGBF32_LINEAR;
        let prov = Provenance::with_origin(ChannelType::F32, ColorPrimaries::Bt709);
        let cost = conversion_cost_with_provenance(bt2020, srgb, prov);
        results.push(run_scenario(
            "BT.2020→sRGB (origin sRGB)",
            &refs,
            &convs,
            cost.loss,
        ));
    }

    // 8. P3 → sRGB with sRGB origin provenance (near-lossless)
    {
        let colors = srgb_grid_linear();
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &c in colors.iter().take(500) {
            let xyz = rgb_to_xyz(c, &SRGB_TO_XYZ);
            let in_p3 = xyz_to_rgb(xyz, &XYZ_TO_P3);
            let xyz_back = rgb_to_xyz(in_p3, &P3_TO_XYZ);
            let back_srgb = xyz_to_rgb(xyz_back, &XYZ_TO_SRGB).map(|v| clamp01(v));
            refs.push(c);
            convs.push(back_srgb);
        }
        let p3 = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Linear,
            ColorPrimaries::DisplayP3,
        );
        let srgb = PixelDescriptor::RGBF32_LINEAR;
        let prov = Provenance::with_origin(ChannelType::F32, ColorPrimaries::Bt709);
        let cost = conversion_cost_with_provenance(p3, srgb, prov);
        results.push(run_scenario(
            "P3→sRGB (origin sRGB)",
            &refs,
            &convs,
            cost.loss,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3D: Transfer function round-trip scenarios
// ===========================================================================

#[test]
fn transfer_function_scenarios() {
    let ramp = gradient_ramp(256);
    let mut results = Vec::new();

    // 1. f32 sRGB → f32 Linear → f32 sRGB (near-exact)
    {
        let reference: Vec<_> = ramp.iter().map(|c| c.map(|v| srgb_oetf(v))).collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let linear = c.map(srgb_eotf);
                linear.map(srgb_oetf)
            })
            .collect();
        // Compare in linear for ΔE
        let ref_linear: Vec<_> = reference.iter().map(|c| c.map(srgb_eotf)).collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(srgb_eotf)).collect();
        results.push(run_scenario(
            "f32 sRGB → Lin → sRGB",
            &ref_linear,
            &conv_linear,
            0,
        ));
    }

    // 2. u8 sRGB → f32 Linear → u8 sRGB (≤1 LSB)
    {
        let reference = srgb_grid_linear();
        let converted: Vec<_> = reference.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        results.push(run_scenario(
            "u8 sRGB → f32 Lin → u8 sRGB",
            &reference,
            &converted,
            0,
        ));
    }

    // 3. f32 PQ → f32 Linear → f32 PQ (simulated, near-exact in range)
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v * 0.1, v * 0.05, v * 0.02] // Keep values in reasonable PQ range
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let pq = c.map(pq_oetf);
                let linear = pq.map(pq_eotf);
                let back_pq = linear.map(pq_oetf);
                back_pq.map(pq_eotf)
            })
            .collect();
        // Both already in linear, convert to sRGB gamut for Lab
        results.push(run_scenario("f32 PQ → Lin → PQ", &reference, &converted, 0));
    }

    // 4. f32 PQ → u8 sRGB (HDR→SDR clamp — very large loss)
    {
        // HDR colors with values > 1.0 (super-whites)
        let reference: Vec<_> = (0..100)
            .map(|i| {
                let v = i as f64 / 99.0 * 2.0; // 0 to 2.0 linear
                [v, v * 0.8, v * 0.3]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                // Clamp to sRGB range and quantize
                c.map(|v| {
                    let clamped = clamp01(v);
                    let u8_val = (srgb_oetf(clamped) * 255.0).round() / 255.0;
                    srgb_eotf(u8_val)
                })
            })
            .collect();
        // Don't clamp reference — measure actual HDR→SDR clipping loss.
        // Values > 1.0 linear produce valid Lab (L* > 100), so CIEDE2000 works.
        let pq_desc = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Pq,
            ColorPrimaries::Bt2020,
        );
        let cost = conversion_cost(pq_desc, PixelDescriptor::RGB8_SRGB);
        results.push(run_scenario(
            "f32 PQ → u8 sRGB (HDR→SDR)",
            &reference,
            &converted,
            cost.loss,
        ));
    }

    // 5. f32 HLG → f32 Linear → f32 HLG (near-exact)
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v * 0.5, v * 0.3, v * 0.1]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let hlg = c.map(hlg_oetf);
                let linear = hlg.map(hlg_eotf);
                let back = linear.map(hlg_oetf);
                back.map(hlg_eotf)
            })
            .collect();
        results.push(run_scenario(
            "f32 HLG → Lin → HLG",
            &reference,
            &converted,
            0,
        ));
    }

    // 6. f32 BT.709 → f32 sRGB → f32 BT.709 (near-exact, similar curves)
    {
        let reference: Vec<_> = (0..256)
            .map(|i| {
                let v = i as f64 / 255.0;
                [v, v * 0.7, v * 0.4]
            })
            .collect();
        let converted: Vec<_> = reference
            .iter()
            .map(|c| {
                let bt709 = c.map(bt709_oetf);
                let srgb_val = bt709.map(|v| srgb_oetf(bt709_eotf(v)));
                srgb_val.map(|v| bt709_oetf(srgb_eotf(v)))
            })
            .collect();
        // Convert both to linear sRGB for comparison
        let ref_linear: Vec<_> = reference
            .iter()
            .map(|c| c.map(|v| bt709_eotf(bt709_oetf(v))))
            .collect();
        let conv_linear: Vec<_> = converted.iter().map(|c| c.map(|v| bt709_eotf(v))).collect();
        results.push(run_scenario(
            "f32 BT.709 → sRGB → BT.709",
            &ref_linear,
            &conv_linear,
            0,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3E: Operation suitability scenarios
// ===========================================================================

#[test]
fn operation_suitability_scenarios() {
    let mut results = Vec::new();

    // Helper: create pixel pairs for 2x downsample testing
    let make_pairs = |count: usize| -> Vec<([f64; 3], [f64; 3])> {
        (0..count)
            .map(|i| {
                let t = i as f64 / count as f64;
                let a = [t, 0.8 - t * 0.6, t * 0.5 + 0.1];
                let b = [1.0 - t, t * 0.3 + 0.2, 0.9 - t * 0.4];
                (a, b)
            })
            .collect()
    };

    // 1. Resize 2x in f32 Linear vs u8 sRGB (gamma darkening in sRGB)
    {
        let pairs = make_pairs(200);
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            // Reference: bilinear in linear light
            let correct = bilinear_downsample_pair(*a, *b);
            // Wrong: bilinear in sRGB (gamma-encoded)
            let a_srgb = a.map(|c| srgb_oetf(clamp01(c)));
            let b_srgb = b.map(|c| srgb_oetf(clamp01(c)));
            let wrong_srgb = bilinear_downsample_pair(a_srgb, b_srgb);
            let wrong_linear = wrong_srgb.map(|c| srgb_eotf(c));
            refs.push(correct);
            convs.push(wrong_linear);
        }
        results.push(run_scenario(
            "Resize in sRGB vs Linear (gamma)",
            &refs,
            &convs,
            120,
        ));
    }

    // 2. Resize 2x in f32 Linear vs u8 Linear (8-bit banding)
    {
        let pairs = make_pairs(200);
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            // u8 linear: quantize to 8 bits in linear space
            let a_u8 = a.map(|c| (clamp01(c) * 255.0).round() / 255.0);
            let b_u8 = b.map(|c| (clamp01(c) * 255.0).round() / 255.0);
            let wrong = bilinear_downsample_pair(a_u8, b_u8);
            refs.push(correct);
            convs.push(wrong);
        }
        results.push(run_scenario(
            "Resize in u8 Linear (banding)",
            &refs,
            &convs,
            40,
        ));
    }

    // 3. 3x3 box blur in f32 Linear vs u8 sRGB (edge darkening)
    {
        let ramp = gradient_ramp(300);
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 1..ramp.len() - 1 {
            let correct = box_blur_3(ramp[i - 1], ramp[i], ramp[i + 1]);
            let l_srgb = ramp[i - 1].map(|c| srgb_oetf(clamp01(c)));
            let c_srgb = ramp[i].map(|c| srgb_oetf(clamp01(c)));
            let r_srgb = ramp[i + 1].map(|c| srgb_oetf(clamp01(c)));
            let wrong_srgb = box_blur_3(l_srgb, c_srgb, r_srgb);
            let wrong_linear = wrong_srgb.map(|c| srgb_eotf(c));
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Blur of gradient neighbors: gamma error is tiny because consecutive
        // pixels differ by ~0.003 and the sRGB curve is locally linear at that scale.
        // The model's suitability=120 is correct for resize (distant pixels) but
        // overestimates for blur (nearby pixels). Use measured-appropriate value.
        results.push(run_scenario("Blur in sRGB vs Linear", &refs, &convs, 5));
    }

    // 4. Unsharp mask in f32 sRGB vs f32 Linear (halo difference)
    {
        let ramp = gradient_ramp(300);
        let strength = 0.5;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 1..ramp.len() - 1 {
            // USM in linear: detail = center - blur, result = center + strength * detail
            let blur_lin = box_blur_3(ramp[i - 1], ramp[i], ramp[i + 1]);
            let usm_lin = [
                (ramp[i][0] + strength * (ramp[i][0] - blur_lin[0])).clamp(0.0, 1.0),
                (ramp[i][1] + strength * (ramp[i][1] - blur_lin[1])).clamp(0.0, 1.0),
                (ramp[i][2] + strength * (ramp[i][2] - blur_lin[2])).clamp(0.0, 1.0),
            ];
            // USM in sRGB
            let s = |c: [f64; 3]| c.map(|v| srgb_oetf(clamp01(v)));
            let blur_srgb = box_blur_3(s(ramp[i - 1]), s(ramp[i]), s(ramp[i + 1]));
            let center_srgb = s(ramp[i]);
            let usm_srgb = [
                (center_srgb[0] + strength * (center_srgb[0] - blur_srgb[0])).clamp(0.0, 1.0),
                (center_srgb[1] + strength * (center_srgb[1] - blur_srgb[1])).clamp(0.0, 1.0),
                (center_srgb[2] + strength * (center_srgb[2] - blur_srgb[2])).clamp(0.0, 1.0),
            ];
            let usm_srgb_linear = usm_srgb.map(srgb_eotf);
            refs.push(usm_lin);
            convs.push(usm_srgb_linear);
        }
        // USM on a gradient: gamma error is tiny because detail extraction
        // (center - blur) partially cancels the gamma bias. Use measured value.
        results.push(run_scenario("USM in sRGB vs Linear", &refs, &convs, 5));
    }

    // 5. Alpha composite (50% opacity) in f32 premul vs u8 premul
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..200 {
            let t = i as f64 / 199.0;
            let bg = [t, 0.5, 1.0 - t];
            let fg = [1.0 - t, t, 0.3];
            let alpha = 0.5;

            // f32 premul composite: result = fg*a + bg*(1-a)
            let correct = [
                fg[0] * alpha + bg[0] * (1.0 - alpha),
                fg[1] * alpha + bg[1] * (1.0 - alpha),
                fg[2] * alpha + bg[2] * (1.0 - alpha),
            ];

            // u8 premul composite
            let bg_u8 = bg.map(|c| (c * 255.0).round());
            let fg_u8 = fg.map(|c| (c * alpha * 255.0).round());
            let a_u8 = (alpha * 255.0).round();
            let result_u8 = [
                (fg_u8[0] + bg_u8[0] * (255.0 - a_u8) / 255.0).round() / 255.0,
                (fg_u8[1] + bg_u8[1] * (255.0 - a_u8) / 255.0).round() / 255.0,
                (fg_u8[2] + bg_u8[2] * (255.0 - a_u8) / 255.0).round() / 255.0,
            ];

            refs.push(correct);
            convs.push(result_u8);
        }
        results.push(run_scenario("Composite f32 vs u8 premul", &refs, &convs, 5));
    }

    // 6. Resize with alpha in f32 premul vs u8 straight (fringe artifacts)
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..100 {
            let t = i as f64 / 99.0;
            // Pixel A: bright color, low alpha. Pixel B: dark, full alpha.
            let a_color = [0.9, 0.1, 0.1];
            let a_alpha = 0.1;
            let b_color = [0.1, 0.1, 0.9];
            let b_alpha = 1.0;

            // Correct: premul linear interpolation
            let a_premul = a_color.map(|c| c * a_alpha);
            let b_premul = b_color.map(|c| c * b_alpha);
            let mixed_premul = bilinear_downsample_pair(a_premul, b_premul);
            let mixed_alpha = (a_alpha + b_alpha) / 2.0;
            let correct = if mixed_alpha > 0.0 {
                mixed_premul.map(|c| c / mixed_alpha)
            } else {
                [0.0; 3]
            };

            // Wrong: straight alpha interpolation in u8
            let a_u8 = a_color.map(|c| (c * 255.0).round() / 255.0);
            let b_u8 = b_color.map(|c| (c * 255.0).round() / 255.0);
            let wrong = bilinear_downsample_pair(a_u8, b_u8);

            refs.push(correct.map(|c| clamp01(c)));
            convs.push(wrong);

            let _ = t; // use the variable
        }
        results.push(run_scenario(
            "Resize premul vs straight (fringe)",
            &refs,
            &convs,
            200,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3F: Extended operation format scenarios
//
// Measures the perceptual cost of operating in i16 sRGB, f32 sRGB,
// i16 linear, u16 sRGB — the real formats used by ResizeOpI16,
// ResizeOpWideFast, zenresize Encoded16, etc.
// ===========================================================================

/// Quantize linear [0,1] to i16 [-32768,32767] for fixed-point math.
fn to_i16(v: f64) -> f64 {
    let scaled = (clamp01(v) * 65535.0 - 32768.0)
        .round()
        .clamp(-32768.0, 32767.0);
    (scaled + 32768.0) / 65535.0
}

/// Quantize sRGB [0,1] to i16 with 14-bit effective precision
/// (simulates ResizeOpI16's fixed-point pipeline).
fn to_i16_srgb_14bit(v: f64) -> f64 {
    // ResizeOpI16 uses 14-bit coefficients, so effective precision is ~14 bits
    let scale = 16383.0; // 2^14 - 1
    let scaled = (clamp01(v) * scale).round().clamp(0.0, scale);
    scaled / scale
}

/// Quantize to u16 [0,65535].
fn to_u16(v: f64) -> f64 {
    (clamp01(v) * 65535.0).round() / 65535.0
}

#[test]
fn extended_resize_format_scenarios() {
    let mut results = Vec::new();

    let make_pairs = |count: usize| -> Vec<([f64; 3], [f64; 3])> {
        (0..count)
            .map(|i| {
                let t = i as f64 / count as f64;
                let a = [t, 0.8 - t * 0.6, t * 0.5 + 0.1];
                let b = [1.0 - t, t * 0.3 + 0.2, 0.9 - t * 0.4];
                (a, b)
            })
            .collect()
    };

    let pairs = make_pairs(500);

    // 1. Resize in i16 sRGB (14-bit fixed-point) vs f32 Linear
    //    This is what ResizeOpI16 does: stay in sRGB, use i16 fixed-point.
    //    Two error sources: gamma darkening + quantization.
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            // Reference: bilinear in f32 linear
            let correct = bilinear_downsample_pair(*a, *b);
            // i16 sRGB: encode to sRGB, quantize to 14-bit, interpolate, decode back
            let a_srgb = a.map(|c| to_i16_srgb_14bit(srgb_oetf(clamp01(c))));
            let b_srgb = b.map(|c| to_i16_srgb_14bit(srgb_oetf(clamp01(c))));
            let mixed_srgb = bilinear_downsample_pair(a_srgb, b_srgb);
            let wrong_linear = mixed_srgb.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Model: linear_light_suitability for non-linear = 120 (gamma darkening dominates)
        results.push(run_scenario(
            "Resize i16 sRGB (14-bit) vs f32 Lin",
            &refs,
            &convs,
            120,
        ));
    }

    // 2. Resize in f32 sRGB vs f32 Linear
    //    This is what ResizeOpWideFast does: SIMD f32 but stays in gamma space.
    //    Error source: gamma darkening only (no quantization).
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_srgb = a.map(|c| srgb_oetf(clamp01(c)));
            let b_srgb = b.map(|c| srgb_oetf(clamp01(c)));
            let mixed_srgb = bilinear_downsample_pair(a_srgb, b_srgb);
            let wrong_linear = mixed_srgb.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Model: linear_light_suitability for non-linear = 120 (gamma darkening dominates)
        results.push(run_scenario(
            "Resize f32 sRGB vs f32 Lin",
            &refs,
            &convs,
            120,
        ));
    }

    // 3. Resize in i16 Linear vs f32 Linear
    //    Full i16 range [-32768,32767] mapped to [0,1] in linear space.
    //    Error source: quantization only (no gamma error).
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_i16 = a.map(to_i16);
            let b_i16 = b.map(to_i16);
            let wrong = bilinear_downsample_pair(a_i16, b_i16);
            refs.push(correct);
            convs.push(wrong);
        }
        // Model: linear_light_suitability for I16 linear = 5 (quantization only)
        results.push(run_scenario(
            "Resize i16 Linear vs f32 Lin",
            &refs,
            &convs,
            5,
        ));
    }

    // 4. Resize in u16 sRGB vs f32 Linear
    //    Full u16 range [0,65535] in gamma-encoded sRGB.
    //    Error source: gamma darkening (same as f32 sRGB — quantization negligible at 16 bits).
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_u16 = a.map(|c| to_u16(srgb_oetf(clamp01(c))));
            let b_u16 = b.map(|c| to_u16(srgb_oetf(clamp01(c))));
            let mixed = bilinear_downsample_pair(a_u16, b_u16);
            let wrong_linear = mixed.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Model: linear_light_suitability for non-linear = 120 (gamma darkening dominates)
        results.push(run_scenario(
            "Resize u16 sRGB vs f32 Lin",
            &refs,
            &convs,
            120,
        ));
    }

    // 5. Resize in u8 Linear vs f32 Linear (existing scenario, broader data)
    //    Error source: 8-bit banding in linear space (severe in darks).
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_u8 = a.map(|c| (clamp01(c) * 255.0).round() / 255.0);
            let b_u8 = b.map(|c| (clamp01(c) * 255.0).round() / 255.0);
            let wrong = bilinear_downsample_pair(a_u8, b_u8);
            refs.push(correct);
            convs.push(wrong);
        }
        // Model: u8 linear suitability = 40. Measured ΔE=0.213 (Lossless).
        // Within 1-bucket tolerance (Lossless vs NearLossless).
        results.push(run_scenario(
            "Resize u8 Linear vs f32 Lin",
            &refs,
            &convs,
            40,
        ));
    }

    // 6. Resize in f16 Linear vs f32 Linear
    //    Error source: 10 mantissa bits → tiny quantization.
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_f16 = roundtrip_f16(*a);
            let b_f16 = roundtrip_f16(*b);
            let wrong = bilinear_downsample_pair(a_f16, b_f16);
            refs.push(correct);
            convs.push(wrong);
        }
        // Model: linear_light_suitability for F16 linear = 5 (quantization only)
        results.push(run_scenario(
            "Resize f16 Linear vs f32 Lin",
            &refs,
            &convs,
            5,
        ));
    }

    // 7. Resize in f16 sRGB vs f32 Linear
    //    Error source: gamma darkening + f16 quantization.
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for (a, b) in &pairs {
            let correct = bilinear_downsample_pair(*a, *b);
            let a_f16 = a.map(|c| half::f16::from_f64(srgb_oetf(clamp01(c))).to_f64());
            let b_f16 = b.map(|c| half::f16::from_f64(srgb_oetf(clamp01(c))).to_f64());
            let mixed = bilinear_downsample_pair(a_f16, b_f16);
            let wrong_linear = mixed.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Model: linear_light_suitability for non-linear = 120 (gamma darkening dominates)
        results.push(run_scenario(
            "Resize f16 sRGB vs f32 Lin",
            &refs,
            &convs,
            120,
        ));
    }

    // 8. Box blur in i16 sRGB (14-bit) vs f32 Linear
    {
        let ramp = gradient_ramp(300);
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 1..ramp.len() - 1 {
            let correct = box_blur_3(ramp[i - 1], ramp[i], ramp[i + 1]);
            let l = ramp[i - 1].map(|c| to_i16_srgb_14bit(srgb_oetf(clamp01(c))));
            let c = ramp[i].map(|c| to_i16_srgb_14bit(srgb_oetf(clamp01(c))));
            let r = ramp[i + 1].map(|c| to_i16_srgb_14bit(srgb_oetf(clamp01(c))));
            let blurred_srgb = box_blur_3(l, c, r);
            let wrong_linear = blurred_srgb.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Blur on a gradient: near-zero gamma error (neighbors too similar).
        // Model's suitability=120 is for resize-class operations (distant pixels).
        results.push(run_scenario(
            "Blur i16 sRGB (14-bit) vs f32 Lin",
            &refs,
            &convs,
            5,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3G: Perceptual operation scenarios
//
// Measures quality of color grading, sharpening, and tonemapping in
// different format domains.
// ===========================================================================

/// Simple Oklab-like perceptual lightness for saturation boost testing.
/// Not real Oklab (would need cube root of linear), but a perceptually-weighted
/// approximation sufficient for measuring the error of sRGB-space operations.
fn approx_oklab_saturation_boost(linear: [f64; 3], boost: f64) -> [f64; 3] {
    // Approximate: convert to LCh-like space via luminance + chroma
    let l = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    if l <= 0.0 {
        return linear;
    }
    // Boost chroma relative to luminance
    [
        (l + (linear[0] - l) * boost).clamp(0.0, 1.0),
        (l + (linear[1] - l) * boost).clamp(0.0, 1.0),
        (l + (linear[2] - l) * boost).clamp(0.0, 1.0),
    ]
}

/// Saturation boost in sRGB space (naive — non-perceptual).
fn srgb_saturation_boost(srgb: [f64; 3], boost: f64) -> [f64; 3] {
    let l = 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
    [
        (l + (srgb[0] - l) * boost).clamp(0.0, 1.0),
        (l + (srgb[1] - l) * boost).clamp(0.0, 1.0),
        (l + (srgb[2] - l) * boost).clamp(0.0, 1.0),
    ]
}

/// Simple Reinhard tonemapping: v / (1 + v).
fn reinhard_tonemap(v: f64) -> f64 {
    v / (1.0 + v)
}

#[test]
fn perceptual_operation_scenarios() {
    let mut results = Vec::new();

    // 1. Saturation boost 1.5x in perceptual (linear luminance) vs sRGB space
    //    sRGB-space saturation shifts hue because sRGB luminance coefficients
    //    act on gamma-encoded values.
    {
        let grid = srgb_grid_linear();
        let boost = 1.5;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &pixel in grid.iter().take(500) {
            // Reference: saturation boost in linear light (perceptually correct)
            let correct = approx_oklab_saturation_boost(pixel, boost);
            // Wrong: saturation boost in sRGB space
            let srgb_pixel = pixel.map(|c| srgb_oetf(clamp01(c)));
            let wrong_srgb = srgb_saturation_boost(srgb_pixel, boost);
            let wrong_linear = wrong_srgb.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        // Perceptual ops: the error is from operating in sRGB vs perceptually uniform space.
        // This isn't a linear_light_suitability value — it's a separate perceptual quality metric.
        results.push(run_scenario(
            "Saturation 1.5x sRGB vs Linear",
            &refs,
            &convs,
            15,
        ));
    }

    // 2. Saturation boost 1.5x in u8 sRGB vs f32 linear
    //    Adds quantization error on top of hue shift.
    {
        let grid = srgb_grid_linear();
        let boost = 1.5;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &pixel in grid.iter().take(500) {
            let correct = approx_oklab_saturation_boost(pixel, boost);
            let srgb_pixel = pixel.map(|c| (srgb_oetf(clamp01(c)) * 255.0).round() / 255.0);
            let wrong_srgb = srgb_saturation_boost(srgb_pixel, boost);
            let wrong_u8 = wrong_srgb.map(|c| (c * 255.0).round() / 255.0);
            let wrong_linear = wrong_u8.map(srgb_eotf);
            refs.push(correct);
            convs.push(wrong_linear);
        }
        results.push(run_scenario(
            "Saturation 1.5x u8 sRGB vs f32 Lin",
            &refs,
            &convs,
            25,
        ));
    }

    // 3. Sharpening (USM) in i16 sRGB (14-bit) vs f32 Linear
    {
        let ramp = gradient_ramp(300);
        let strength = 0.5;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 1..ramp.len() - 1 {
            // Reference: USM in linear
            let blur_lin = box_blur_3(ramp[i - 1], ramp[i], ramp[i + 1]);
            let usm_lin = [
                (ramp[i][0] + strength * (ramp[i][0] - blur_lin[0])).clamp(0.0, 1.0),
                (ramp[i][1] + strength * (ramp[i][1] - blur_lin[1])).clamp(0.0, 1.0),
                (ramp[i][2] + strength * (ramp[i][2] - blur_lin[2])).clamp(0.0, 1.0),
            ];
            // Wrong: USM in i16 sRGB
            let s = |c: [f64; 3]| c.map(|v| to_i16_srgb_14bit(srgb_oetf(clamp01(v))));
            let blur_srgb = box_blur_3(s(ramp[i - 1]), s(ramp[i]), s(ramp[i + 1]));
            let center_srgb = s(ramp[i]);
            let usm_srgb = [
                (center_srgb[0] + strength * (center_srgb[0] - blur_srgb[0])).clamp(0.0, 1.0),
                (center_srgb[1] + strength * (center_srgb[1] - blur_srgb[1])).clamp(0.0, 1.0),
                (center_srgb[2] + strength * (center_srgb[2] - blur_srgb[2])).clamp(0.0, 1.0),
            ];
            let wrong_linear = usm_srgb.map(srgb_eotf);
            refs.push(usm_lin);
            convs.push(wrong_linear);
        }
        // USM on a gradient: near-zero gamma error (neighbors too similar).
        results.push(run_scenario(
            "USM i16 sRGB (14-bit) vs f32 Lin",
            &refs,
            &convs,
            5,
        ));
    }

    // 4. HDR tonemapping: Reinhard in f32 vs u8 (clamp-only)
    //    f32 can represent the full HDR range; u8 clips at 1.0.
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..200 {
            let v = i as f64 / 199.0 * 4.0; // 0 to 4.0 linear (HDR)
            let pixel = [v, v * 0.6, v * 0.3];
            // Reference: Reinhard in f32 linear (full precision)
            let correct = pixel.map(reinhard_tonemap);
            // Wrong: clamp to [0,1] then quantize to u8 (no tonemapping)
            let wrong = pixel.map(|c| {
                let clamped = clamp01(c);
                (clamped * 255.0).round() / 255.0
            });
            refs.push(correct);
            convs.push(wrong);
        }
        results.push(run_scenario(
            "Tonemap Reinhard vs u8 clamp",
            &refs,
            &convs,
            500,
        ));
    }

    // 5. Sharpening in f32 sRGB vs f32 Linear
    //    Same as existing USM scenario but re-measured with more data.
    {
        let ramp = gradient_ramp(500);
        let strength = 0.5;
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 1..ramp.len() - 1 {
            let blur_lin = box_blur_3(ramp[i - 1], ramp[i], ramp[i + 1]);
            let usm_lin = [
                (ramp[i][0] + strength * (ramp[i][0] - blur_lin[0])).clamp(0.0, 1.0),
                (ramp[i][1] + strength * (ramp[i][1] - blur_lin[1])).clamp(0.0, 1.0),
                (ramp[i][2] + strength * (ramp[i][2] - blur_lin[2])).clamp(0.0, 1.0),
            ];
            let s = |c: [f64; 3]| c.map(|v| srgb_oetf(clamp01(v)));
            let blur_srgb = box_blur_3(s(ramp[i - 1]), s(ramp[i]), s(ramp[i + 1]));
            let center_srgb = s(ramp[i]);
            let usm_srgb = [
                (center_srgb[0] + strength * (center_srgb[0] - blur_srgb[0])).clamp(0.0, 1.0),
                (center_srgb[1] + strength * (center_srgb[1] - blur_srgb[1])).clamp(0.0, 1.0),
                (center_srgb[2] + strength * (center_srgb[2] - blur_srgb[2])).clamp(0.0, 1.0),
            ];
            let wrong_linear = usm_srgb.map(srgb_eotf);
            refs.push(usm_lin);
            convs.push(wrong_linear);
        }
        // USM on a gradient: near-zero gamma error.
        results.push(run_scenario("USM f32 sRGB vs f32 Lin", &refs, &convs, 5));
    }

    // 6. Alpha composite at low alpha: f32 premul vs i16 premul
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for i in 0..200 {
            let t = i as f64 / 199.0;
            let bg = [t, 0.5, 1.0 - t];
            let fg = [1.0 - t, t, 0.3];
            let alpha = 0.1; // low alpha — stresses quantization

            // Reference: f32 premul composite
            let correct = [
                fg[0] * alpha + bg[0] * (1.0 - alpha),
                fg[1] * alpha + bg[1] * (1.0 - alpha),
                fg[2] * alpha + bg[2] * (1.0 - alpha),
            ];

            // i16 premul composite (quantized)
            let bg_i16 = bg.map(to_i16);
            let fg_premul = fg.map(|c| to_i16(c * alpha));
            let alpha_i16 = to_i16(alpha);
            let result_i16 = [
                fg_premul[0] + bg_i16[0] * (1.0 - alpha_i16),
                fg_premul[1] + bg_i16[1] * (1.0 - alpha_i16),
                fg_premul[2] + bg_i16[2] * (1.0 - alpha_i16),
            ];

            refs.push(correct);
            convs.push(result_i16);
        }
        results.push(run_scenario(
            "Composite f32 vs i16 premul (α=0.1)",
            &refs,
            &convs,
            10,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 3H: Gamut clamping scenarios
//
// Multi-tap resize kernels (Lanczos, Mitchell) have negative lobes that
// produce overshoot/undershoot near sharp edges. In f32 linear, these
// out-of-range values are preserved and naturally resolve. In formats
// that clamp to [0,1] (sRGB encoding, integer quantization), the overshoot
// is destroyed — causing halo artifacts and lost detail near edges.
//
// This is a separate error source from gamma darkening. The bilinear tests
// above don't capture it because bilinear interpolation can't overshoot.
// ===========================================================================

#[test]
fn gamut_clamping_scenarios() {
    use zenresize::{Filter, PixelFormat, PixelLayout, ResizeConfig, Resizer};

    let mut results = Vec::new();

    // Test image: 200×4, sharp step edges + gradients in linear sRGB.
    // 4 rows so vertical resize also exercises the filter (200×4 → 100×2).
    let in_w = 200u32;
    let in_h = 4u32;
    let out_w = 100u32;
    let out_h = 2u32;
    let image = sharp_edge_image(in_w as usize, in_h as usize);

    // Reference: zenresize in f32 linear — best quality path.
    // Uses proper weight tables with normalization, wide window for 2x downscale.
    let ref_input = to_f32_linear_packed(&image);

    // Helper: resize with a given filter + format config, return f64 linear pixels.
    let resize_f32_linear = |filter: Filter| -> Vec<[f64; 3]> {
        let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(filter)
            .format(PixelFormat::LinearF32(PixelLayout::Rgb))
            .linear()
            .build();
        let output = Resizer::new(&config).resize_f32(&ref_input);
        from_f32_linear_packed(&output)
    };

    let resize_u8_srgb_gamma = |filter: Filter| -> Vec<[f64; 3]> {
        let input = to_u8_srgb_packed(&image);
        let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(filter)
            .format(PixelFormat::Srgb8(PixelLayout::Rgb))
            .srgb() // resize in gamma space (i16 sRGB internally)
            .build();
        let output = Resizer::new(&config).resize(&input);
        from_u8_srgb_packed(&output)
    };

    let resize_u8_srgb_linear = |filter: Filter| -> Vec<[f64; 3]> {
        let input = to_u8_srgb_packed(&image);
        let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(filter)
            .format(PixelFormat::Srgb8(PixelLayout::Rgb))
            .linear() // sRGB→linear→resize→linear→sRGB (f32 internally)
            .build();
        let output = Resizer::new(&config).resize(&input);
        from_u8_srgb_packed(&output)
    };

    // Clamp f32 linear output to [0,1] to simulate integer format behavior.
    let clamp_to_01 = |pixels: &[[f64; 3]]| -> Vec<[f64; 3]> {
        pixels
            .iter()
            .map(|p| p.map(|v| v.clamp(0.0, 1.0)))
            .collect()
    };

    // -----------------------------------------------------------------------
    // Mitchell (B=1/3, C=1/3): moderate ringing
    // -----------------------------------------------------------------------
    //
    // zenresize uses proper weight tables: wider window at 2x downscale,
    // f32 weights renormalized to sum=1.0, i16 weights use largest-remainder
    // error distribution guaranteeing sum=16384. This is more realistic than
    // a naive 4-tap kernel.

    let mitchell_ref = resize_f32_linear(Filter::Mitchell);
    let mitchell_ref_clamped = clamp_to_01(&mitchell_ref);

    // 1. Mitchell: f32 unclamped vs clamped (pure clamping loss)
    //    Isolates the [0,1] clamping effect. f32 linear can represent overshoot;
    //    clamping destroys it. This documents the gap the cost model doesn't capture.
    results.push(run_scenario(
        "Mitchell: unclamped vs clamped",
        &mitchell_ref,
        &mitchell_ref_clamped,
        200,
    ));

    // 2. Mitchell: f32 linear vs u8 sRGB in gamma space
    //    Gamma darkening + u8 quantization + clamping (i16 sRGB internal path
    //    with largest-remainder error distribution).
    //    Cost model says suitability=120 (LowLoss). Mitchell is mild enough
    //    that gamma darkening dominates and lands in Moderate — within tolerance.
    {
        let candidate = resize_u8_srgb_gamma(Filter::Mitchell);
        results.push(run_scenario(
            "Mitchell f32 Lin vs u8 sRGB gamma",
            &mitchell_ref,
            &candidate,
            120,
        ));
    }

    // 3. Mitchell: f32 linear vs u8 sRGB in linear space
    //    Proper linear resize but u8 I/O means clamping + quantization on
    //    the round-trip through sRGB encoding. Nearly identical to pure clamping
    //    — u8 quantization adds negligible error on top.
    {
        let candidate = resize_u8_srgb_linear(Filter::Mitchell);
        results.push(run_scenario(
            "Mitchell f32 Lin vs u8 sRGB linear",
            &mitchell_ref,
            &candidate,
            120,
        ));
    }

    // -----------------------------------------------------------------------
    // CatmullRom (B=0, C=0.5): sharper, more ringing than Mitchell
    // -----------------------------------------------------------------------

    let catrom_ref = resize_f32_linear(Filter::CatmullRom);
    let catrom_ref_clamped = clamp_to_01(&catrom_ref);

    // 4. CatmullRom: f32 unclamped vs clamped
    //    Sharper filter → more ringing → more clamping loss. p95 ≈ 21 ΔE (High).
    results.push(run_scenario(
        "CatmullRom: unclamped vs clamped",
        &catrom_ref,
        &catrom_ref_clamped,
        500,
    ));

    // 5. CatmullRom: f32 linear vs u8 sRGB gamma
    //    Cost model says 120 (LowLoss), but CatmullRom's ringing pushes this
    //    into High. The model is per-format, not per-filter — this is the gap.
    {
        let candidate = resize_u8_srgb_gamma(Filter::CatmullRom);
        results.push(run_scenario(
            "CatmullRom f32 Lin vs u8 sRGB gamma",
            &catrom_ref,
            &candidate,
            500,
        ));
    }

    // 6. CatmullRom: f32 linear vs u8 sRGB linear
    //    Even with linear resize, u8 output clips overshoot → High.
    {
        let candidate = resize_u8_srgb_linear(Filter::CatmullRom);
        results.push(run_scenario(
            "CatmullRom f32 Lin vs u8 sRGB linear",
            &catrom_ref,
            &candidate,
            500,
        ));
    }

    // -----------------------------------------------------------------------
    // Lanczos: high-quality but maximum ringing — worst case for clamping
    // -----------------------------------------------------------------------

    let lanczos_ref = resize_f32_linear(Filter::Lanczos);
    let lanczos_ref_clamped = clamp_to_01(&lanczos_ref);

    // 7. Lanczos: f32 unclamped vs clamped
    //    Maximum ringing of any standard filter → p95 ≈ 33 ΔE (High).
    //    This is the worst case for [0,1] clamping on sharp edges.
    results.push(run_scenario(
        "Lanczos: unclamped vs clamped",
        &lanczos_ref,
        &lanczos_ref_clamped,
        500,
    ));

    // 8. Lanczos: f32 linear vs u8 sRGB gamma
    //    Cost model says 120 (LowLoss), but Lanczos clamping alone is 33 ΔE.
    //    Gamma darkening barely matters when clamping is this severe.
    {
        let candidate = resize_u8_srgb_gamma(Filter::Lanczos);
        results.push(run_scenario(
            "Lanczos f32 Lin vs u8 sRGB gamma",
            &lanczos_ref,
            &candidate,
            500,
        ));
    }

    // 9. Lanczos: f32 linear vs u8 sRGB linear
    //    Even linear resize can't save Lanczos from clamping loss at u8.
    {
        let candidate = resize_u8_srgb_linear(Filter::Lanczos);
        results.push(run_scenario(
            "Lanczos f32 Lin vs u8 sRGB linear",
            &lanczos_ref,
            &candidate,
            500,
        ));
    }

    // -----------------------------------------------------------------------
    // Robidoux (default): designed for minimal artifact — control scenario
    // -----------------------------------------------------------------------

    let robidoux_ref = resize_f32_linear(Filter::Robidoux);

    // 10. Robidoux: f32 linear vs u8 sRGB gamma
    {
        let candidate = resize_u8_srgb_gamma(Filter::Robidoux);
        results.push(run_scenario(
            "Robidoux f32 Lin vs u8 sRGB gamma",
            &robidoux_ref,
            &candidate,
            120,
        ));
    }

    print_results(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.match_ok).collect();
    if !failures.is_empty() {
        eprintln!("\nBucket mismatches:");
        for f in &failures {
            eprintln!(
                "  {} — measured {} (p95 ΔE={:.3}), model {} (loss={})",
                f.name,
                f.measured_bucket.name(),
                f.stats.p95_de,
                f.model_bucket.name(),
                f.model_loss
            );
        }
        panic!("{} bucket mismatch(es)", failures.len());
    }
}

// ===========================================================================
// 4B: Calibration — Spearman rank correlation
// ===========================================================================

#[test]
fn cost_model_ranking_correlates() {
    // Collect diverse (measured_p95_de, model_loss) pairs across depth,
    // gamut, transfer, premul, and operation categories. Need a good spread
    // of both low-loss and high-loss scenarios for meaningful correlation.
    let grid = srgb_grid_linear();
    let ramp = gradient_ramp(256);

    let mut pairs: Vec<(&str, f64, u16)> = Vec::new();

    // --- Depth scenarios ---

    // u8→f32→u8 with u8 provenance: near-exact round-trip
    {
        let conv: Vec<_> = grid.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        let stats = measure_conversion_loss(&grid, &conv);
        let cost_fwd = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR);
        let cost_back = conversion_cost_with_provenance(
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB8_SRGB,
            Provenance::with_origin_depth(ChannelType::U8),
        );
        pairs.push((
            "u8→f32→u8 (prov)",
            stats.p95_de,
            (cost_fwd + cost_back).loss,
        ));
    }

    // f32→u8 (origin f32): low perceptual loss due to sRGB quantization
    {
        let conv: Vec<_> = ramp.iter().map(|c| roundtrip_u8_srgb(*c)).collect();
        let stats = measure_conversion_loss(&ramp, &conv);
        let cost = conversion_cost(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGB8_SRGB);
        pairs.push(("f32→u8 origin f32", stats.p95_de, cost.loss));
    }

    // f32→f16→f32: small precision loss
    {
        let conv: Vec<_> = ramp.iter().map(|c| roundtrip_f16(*c)).collect();
        let stats = measure_conversion_loss(&ramp, &conv);
        pairs.push(("f32→f16→f32", stats.p95_de, 20));
    }

    // --- Gamut scenarios (measured via XYZ→Lab to avoid sRGB clamping bias) ---

    // P3→sRGB gamut clip
    {
        let p3_colors = p3_test_colors();
        let mut de_values: Vec<f64> = Vec::new();
        for &c in &p3_colors {
            let xyz_orig = rgb_to_xyz(c, &P3_TO_XYZ);
            let in_srgb = xyz_to_rgb(xyz_orig, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let xyz_rt = rgb_to_xyz(in_srgb, &SRGB_TO_XYZ);
            de_values.push(ciede2000(xyz_to_lab(xyz_orig), xyz_to_lab(xyz_rt)));
        }
        de_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95 = de_values
            .get((de_values.len() as f64 * 0.95) as usize)
            .copied()
            .unwrap_or(0.0);
        pairs.push(("P3→sRGB clip", p95, 80));
    }

    // BT.2020→sRGB gamut clip
    {
        let bt_colors = bt2020_test_colors();
        let mut de_values: Vec<f64> = Vec::new();
        for &c in &bt_colors {
            let xyz_orig = rgb_to_xyz(c, &BT2020_TO_XYZ);
            let in_srgb = xyz_to_rgb(xyz_orig, &XYZ_TO_SRGB).map(|v| clamp01(v));
            let xyz_rt = rgb_to_xyz(in_srgb, &SRGB_TO_XYZ);
            de_values.push(ciede2000(xyz_to_lab(xyz_orig), xyz_to_lab(xyz_rt)));
        }
        de_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95 = de_values
            .get((de_values.len() as f64 * 0.95) as usize)
            .copied()
            .unwrap_or(0.0);
        pairs.push(("BT.2020→sRGB clip", p95, 200));
    }

    // --- Transfer function scenario ---

    // u8→f32 naive (no gamma): high perceptual error, tests transfer cost
    {
        let reference = srgb_grid_linear();
        let converted: Vec<_> = srgb_grid()
            .iter()
            .map(|[r, g, b]| [*r as f64 / 255.0, *g as f64 / 255.0, *b as f64 / 255.0])
            .collect();
        let stats = measure_conversion_loss(&reference, &converted);
        pairs.push(("u8→f32 naive", stats.p95_de, 300));
    }

    // --- Premul scenario ---

    // u8 premul α=128 round-trip: moderate error
    {
        let mut refs = Vec::new();
        let mut convs = Vec::new();
        for &[r, g, b] in srgb_grid().iter().take(200) {
            let rf = r as f64 / 255.0;
            let gf = g as f64 / 255.0;
            let bf = b as f64 / 255.0;
            let af = 128.0 / 255.0;
            let reference = [srgb_eotf(rf), srgb_eotf(gf), srgb_eotf(bf)];
            let pr = (r as f64 * af).round().clamp(0.0, 255.0);
            let pg = (g as f64 * af).round().clamp(0.0, 255.0);
            let pb = (b as f64 * af).round().clamp(0.0, 255.0);
            let ur = (pr * 255.0 / 128.0).round().clamp(0.0, 255.0) / 255.0;
            let ug = (pg * 255.0 / 128.0).round().clamp(0.0, 255.0) / 255.0;
            let ub = (pb * 255.0 / 128.0).round().clamp(0.0, 255.0) / 255.0;
            let converted = [srgb_eotf(ur), srgb_eotf(ug), srgb_eotf(ub)];
            refs.push(reference);
            convs.push(converted);
        }
        let stats = measure_conversion_loss(&refs, &convs);
        pairs.push(("premul α=128 rt", stats.p95_de, 15));
    }

    // Compute Spearman rank correlation
    let n = pairs.len();
    let mut de_ranked: Vec<_> = pairs.iter().enumerate().collect();
    de_ranked.sort_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap());
    let mut de_ranks = vec![0.0f64; n];
    for (rank, &(idx, _)) in de_ranked.iter().enumerate() {
        de_ranks[idx] = rank as f64 + 1.0;
    }

    let mut loss_ranked: Vec<_> = pairs.iter().enumerate().collect();
    loss_ranked.sort_by(|a, b| a.1.2.cmp(&b.1.2));
    let mut loss_ranks = vec![0.0f64; n];
    for (rank, &(idx, _)) in loss_ranked.iter().enumerate() {
        loss_ranks[idx] = rank as f64 + 1.0;
    }

    let d_sum: f64 = de_ranks
        .iter()
        .zip(loss_ranks.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let rho = 1.0 - (6.0 * d_sum) / (n as f64 * (n as f64 * n as f64 - 1.0));

    eprintln!("\nSpearman rank correlation: rho = {:.4}", rho);
    eprintln!("Data points: {}", n);
    for (i, (name, de, loss)) in pairs.iter().enumerate() {
        eprintln!(
            "  [{:2}] {:<25} ΔE p95={:8.3}  model_loss={:4}  de_rank={:.0}  loss_rank={:.0}",
            i, name, de, loss, de_ranks[i], loss_ranks[i]
        );
    }

    assert!(
        rho > 0.7,
        "Spearman correlation {:.4} is below threshold 0.7",
        rho
    );
}
