//! Generate ICC hash table `.inc` files for `zenpixels::icc`.
//!
//! Scans ICC profiles, computes normalized FNV-1a hashes, identifies primaries
//! and TRC (measuring max u16 error against reference EOTFs for all 65536 values),
//! deduplicates by normalized hash, and runs moxcms transforms per rendering
//! intent to derive an empirical intent-safety mask. Writes Rust include files.
//!
//! Usage:
//!   cargo run -p icc-gen --release -- <icc-cache-dir> <bundled-profiles-dir> <out-dir>
//!
//! Optional lcms2 cross-check (requires Little CMS 2 C library):
//!   cargo run -p icc-gen --release --features lcms2-crosscheck -- ...

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use moxcms::{ColorProfile, Layout, RenderingIntent, TransformOptions};

// ── Normalized FNV-1a hash ───────────────────────────────────────────────
//
// Must match zenpixels/src/icc/mod.rs::fnv1a_64_normalized exactly.

fn fnv1a_64_normalized(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x100_0000_01b3;
    let mut hash = OFFSET;

    // Phase 1: first 100 bytes — zero metadata fields.
    let header_len = data.len().min(100);
    let mut i = 0;
    while i < header_len {
        let b = if (4..8).contains(&i)
            || (24..36).contains(&i)
            || (40..44).contains(&i)
            || (48..56).contains(&i)
            || (80..100).contains(&i)
        {
            0u8
        } else {
            data[i]
        };
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }

    // Phase 2: remaining bytes — straight hash.
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    hash
}

// ── Reference EOTFs (for TRC matching) ───────────────────────────────────

fn srgb_eotf(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn bt709_eotf(v: f64) -> f64 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}
fn gamma22_eotf(v: f64) -> f64 {
    v.powf(2.199_218_75)
}
fn gamma18_eotf(v: f64) -> f64 {
    v.powf(1.8)
}
fn gamma24_eotf(v: f64) -> f64 {
    v.powf(2.4)
}
fn gamma26_eotf(v: f64) -> f64 {
    v.powf(2.6)
}
fn linear_eotf(v: f64) -> f64 {
    v
}
fn pq_eotf(v: f64) -> f64 {
    const M1: f64 = 0.159_301_757_812_5;
    const M2: f64 = 78.843_75;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.687_5;
    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 {
        0.0
    } else {
        (num / den).powf(1.0 / M1)
    }
}
fn hlg_ootf_inv(v: f64) -> f64 {
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        const A: f64 = 0.178_832_77;
        const B: f64 = 1.0 - 4.0 * A;
        const C: f64 = 0.5 - A * (4.0 * A - B) / (4.0 * A);
        (((v - C) / A).exp() + B) / 12.0
    }
}

#[allow(dead_code)]
struct RefTrc {
    name: &'static str,
    tf_name: &'static str,
    eotf: fn(f64) -> f64,
}
const REFERENCE_TRCS: &[RefTrc] = &[
    RefTrc {
        name: "sRGB",
        tf_name: "Srgb",
        eotf: srgb_eotf,
    },
    RefTrc {
        name: "BT.709",
        tf_name: "Bt709",
        eotf: bt709_eotf,
    },
    RefTrc {
        name: "gamma2.2",
        tf_name: "Gamma22",
        eotf: gamma22_eotf,
    },
    RefTrc {
        name: "gamma1.8",
        tf_name: "Gamma18",
        eotf: gamma18_eotf,
    },
    RefTrc {
        name: "gamma2.4",
        tf_name: "Gamma24",
        eotf: gamma24_eotf,
    },
    RefTrc {
        name: "gamma2.6",
        tf_name: "Gamma26",
        eotf: gamma26_eotf,
    },
    RefTrc {
        name: "linear",
        tf_name: "Linear",
        eotf: linear_eotf,
    },
    RefTrc {
        name: "PQ",
        tf_name: "Pq",
        eotf: pq_eotf,
    },
    RefTrc {
        name: "HLG",
        tf_name: "Hlg",
        eotf: hlg_ootf_inv,
    },
];

// ── ICC TRC parsing ──────────────────────────────────────────────────────

enum Trc {
    Para(Vec<f64>),
    Lut(Vec<u16>),
    Gamma(f64),
}

fn eval_para(p: &[f64], x: f64) -> f64 {
    match p.len() {
        1 => x.powf(p[0]),
        3 => {
            let (g, a, b) = (p[0], p[1], p[2]);
            if x >= -b / a {
                (a * x + b).powf(g)
            } else {
                0.0
            }
        }
        5 => {
            let (g, a, b, c, d) = (p[0], p[1], p[2], p[3], p[4]);
            if x >= d { (a * x + b).powf(g) } else { c * x }
        }
        7 => {
            let (g, a, b, c, d, e, f) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
            if x >= d {
                (a * x + b).powf(g) + e
            } else {
                c * x + f
            }
        }
        _ => x,
    }
}

fn eval_trc(t: &Trc, x: f64) -> f64 {
    match t {
        Trc::Para(p) => eval_para(p, x),
        Trc::Gamma(g) => x.powf(*g),
        Trc::Lut(l) => {
            let p = x * (l.len() - 1) as f64;
            let i = p.floor() as usize;
            let f = p - i as f64;
            if i >= l.len() - 1 {
                l[l.len() - 1] as f64 / 65535.0
            } else {
                let a = l[i] as f64 / 65535.0;
                let b = l[i + 1] as f64 / 65535.0;
                a + f * (b - a)
            }
        }
    }
}

fn parse_trc(d: &[u8], o: usize) -> Option<Trc> {
    if o + 12 > d.len() {
        return None;
    }
    match &d[o..o + 4] {
        b"para" => {
            let ft = u16::from_be_bytes([d[o + 8], d[o + 9]]);
            let n = match ft {
                0 => 1,
                1 => 3,
                2 => 4,
                3 => 5,
                4 => 7,
                _ => return None,
            };
            let mut p = Vec::new();
            for i in 0..n {
                let q = o + 12 + i * 4;
                if q + 4 > d.len() {
                    return None;
                }
                p.push(i32::from_be_bytes([d[q], d[q + 1], d[q + 2], d[q + 3]]) as f64 / 65536.0);
            }
            Some(Trc::Para(p))
        }
        b"curv" => {
            let c = u32::from_be_bytes([d[o + 8], d[o + 9], d[o + 10], d[o + 11]]) as usize;
            if c == 0 {
                Some(Trc::Gamma(1.0))
            } else if c == 1 {
                Some(Trc::Gamma(
                    u16::from_be_bytes([d[o + 12], d[o + 13]]) as f64 / 256.0,
                ))
            } else {
                let mut l = Vec::with_capacity(c);
                for i in 0..c {
                    let q = o + 12 + i * 2;
                    if q + 2 > d.len() {
                        break;
                    }
                    l.push(u16::from_be_bytes([d[q], d[q + 1]]));
                }
                Some(Trc::Lut(l))
            }
        }
        _ => None,
    }
}

fn find_tag(d: &[u8], s: &[u8; 4]) -> Option<usize> {
    if d.len() < 132 {
        return None;
    }
    let tc = u32::from_be_bytes([d[128], d[129], d[130], d[131]]) as usize;
    for i in 0..tc.min(200) {
        let b = 132 + i * 12;
        if b + 12 > d.len() {
            break;
        }
        if &d[b..b + 4] == s {
            return Some(u32::from_be_bytes([d[b + 4], d[b + 5], d[b + 6], d[b + 7]]) as usize);
        }
    }
    None
}

fn max_u16_err(trc: &Trc, eotf: fn(f64) -> f64) -> u32 {
    let mut mx = 0u32;
    for i in 0..=65535u16 {
        let x = i as f64 / 65535.0;
        let a = (eval_trc(trc, x) * 65535.0).round() as i64;
        let b = (eotf(x) * 65535.0).round() as i64;
        let d = (a - b).unsigned_abs() as u32;
        if d > mx {
            mx = d;
        }
    }
    mx
}

// ── Primaries identification ─────────────────────────────────────────────

struct KP {
    rust_name: &'static str,
    rx: f64,
    ry: f64,
    gx: f64,
    gy: f64,
    bx: f64,
    by: f64,
}
const KNOWN_P: &[KP] = &[
    KP {
        rust_name: "Bt709",
        rx: 0.4361,
        ry: 0.2225,
        gx: 0.3851,
        gy: 0.7169,
        bx: 0.1431,
        by: 0.0606,
    },
    KP {
        rust_name: "DisplayP3",
        rx: 0.5151,
        ry: 0.2412,
        gx: 0.2919,
        gy: 0.6922,
        bx: 0.1572,
        by: 0.0666,
    },
    KP {
        rust_name: "DciP3",
        rx: 0.4862,
        ry: 0.2267,
        gx: 0.3239,
        gy: 0.7103,
        bx: 0.1542,
        by: 0.0630,
    },
    KP {
        rust_name: "Bt2020",
        rx: 0.6734,
        ry: 0.2790,
        gx: 0.1656,
        gy: 0.6753,
        bx: 0.1251,
        by: 0.0456,
    },
    KP {
        rust_name: "AdobeRgb",
        rx: 0.6097,
        ry: 0.3111,
        gx: 0.2053,
        gy: 0.6257,
        bx: 0.1492,
        by: 0.0632,
    },
    KP {
        rust_name: "ProPhoto",
        rx: 0.7977,
        ry: 0.2880,
        gx: 0.1352,
        gy: 0.7119,
        bx: 0.0313,
        by: 0.0001,
    },
    KP {
        rust_name: "Smpte170m",
        rx: 0.4163,
        ry: 0.2217,
        gx: 0.3932,
        gy: 0.7033,
        bx: 0.1547,
        by: 0.0750,
    },
    KP {
        rust_name: "Bt470Bg",
        rx: 0.4552,
        ry: 0.2323,
        gx: 0.3676,
        gy: 0.7078,
        bx: 0.1414,
        by: 0.0599,
    },
    KP {
        rust_name: "AppleRgb",
        rx: 0.4755,
        ry: 0.2552,
        gx: 0.3397,
        gy: 0.6726,
        bx: 0.1490,
        by: 0.0723,
    },
    KP {
        rust_name: "ColorMatch",
        rx: 0.5094,
        ry: 0.2749,
        gx: 0.3208,
        gy: 0.6581,
        bx: 0.1339,
        by: 0.0670,
    },
    KP {
        rust_name: "WideGamut",
        rx: 0.7165,
        ry: 0.2587,
        gx: 0.1010,
        gy: 0.7247,
        bx: 0.1467,
        by: 0.0166,
    },
    KP {
        rust_name: "EciRgbV2",
        rx: 0.6503,
        ry: 0.3203,
        gx: 0.1780,
        gy: 0.6021,
        bx: 0.1359,
        by: 0.0777,
    },
];

fn identify_primaries(data: &[u8]) -> Option<&'static str> {
    if data.len() < 132 {
        return None;
    }
    let tc = u32::from_be_bytes([data[128], data[129], data[130], data[131]]) as usize;
    let (mut r, mut g, mut b) = ((0.0f64, 0.0f64), (0.0, 0.0), (0.0, 0.0));
    for i in 0..tc.min(200) {
        let base = 132 + i * 12;
        if base + 12 > data.len() {
            break;
        }
        let sig = &data[base..base + 4];
        let off = u32::from_be_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]) as usize;
        if off + 20 > data.len() {
            continue;
        }
        let rd = |o: usize| {
            (
                i32::from_be_bytes([data[o + 8], data[o + 9], data[o + 10], data[o + 11]]) as f64
                    / 65536.0,
                i32::from_be_bytes([data[o + 12], data[o + 13], data[o + 14], data[o + 15]]) as f64
                    / 65536.0,
            )
        };
        match sig {
            b"rXYZ" => r = rd(off),
            b"gXYZ" => g = rd(off),
            b"bXYZ" => b = rd(off),
            _ => {}
        }
    }
    for k in KNOWN_P {
        const T: f64 = 0.003;
        if (r.0 - k.rx).abs() < T
            && (r.1 - k.ry).abs() < T
            && (g.0 - k.gx).abs() < T
            && (g.1 - k.gy).abs() < T
            && (b.0 - k.bx).abs() < T
            && (b.1 - k.by).abs() < T
        {
            return Some(k.rust_name);
        }
    }
    None
}

// ── TRC identification ───────────────────────────────────────────────────

/// Returns (tf_rust_name, max_u16_error) for the best-matching reference.
fn identify_trc(data: &[u8], trc_tag: &[u8; 4]) -> Option<(&'static str, u32)> {
    let trc_off = find_tag(data, trc_tag)?;
    let trc = parse_trc(data, trc_off)?;

    let mut best: Option<(&str, u32)> = None;
    for r in REFERENCE_TRCS {
        let err = max_u16_err(&trc, r.eotf);
        if best.is_none() || err < best.unwrap().1 {
            best = Some((r.tf_name, err));
        }
    }
    best
}

// ── Empirical intent-safety ──────────────────────────────────────────────
//
// The intent_mask tells runtime callers whether matrix+TRC substitution
// (with our canonical matrix for the identified primaries) matches what
// a full CMS would produce for a given intent.
//
// Static rule (fast path): a matrix-shaper profile with no A2B/B2A LUTs
// routes through moxcms's matrix+TRC path, so all three intents are
// structurally safe. A profile carrying an A2B_n or B2A_n LUT routes
// through that LUT for intent n, which may diverge from our canonical
// matrix+TRC.
//
// Empirical upgrade: a profile with LUTs can still be matrix+TRC-safe
// at an intent if the LUT happens to reproduce matrix+TRC output. We
// test this by running both the profile and a canonical synth reference
// through moxcms at that intent; if the outputs agree within the
// `COLORIMETRIC_VS_SYNTH_EPSILON_U16` budget, we grant the bit despite
// the LUT presence.
//
// Empirical downgrade: a profile that structurally "looks safe" (matrix
// shaper, no LUTs) may nonetheless carry non-canonical primaries — common
// in per-device calibration profiles. If its moxcms output diverges
// wildly from canonical synth, matrix+TRC substitution would produce
// visibly wrong pixels, so we drop the bit.

/// Bit flags matching zenpixels/src/icc/mod.rs.
const INTENT_COLORIMETRIC_SAFE: u8 = 1 << 0;
const INTENT_PERCEPTUAL_SAFE: u8 = 1 << 1;
const INTENT_SATURATION_SAFE: u8 = 1 << 2;

/// Max u16 deviation for the intent-vs-RelColor comparison to count as
/// "intent collapses to colorimetric." 64/65535 ≈ 0.1% — about ¼ of an
/// 8-bit step, matching the `Tolerance::Intent` TRC threshold.
const INTENT_VS_INTENT_EPSILON_U16: u32 = 64;

/// Max u16 deviation between `moxcms(icc, intent=RelColor)` and
/// `moxcms(synth_ref, intent=RelColor)` for the COLORIMETRIC bit to be
/// granted via empirical comparison.
///
/// 640/65535 ≈ 0.98% — about 2.5 8-bit code steps. Absorbs the inherent
/// rounding drift between an encoder's ICC primaries and moxcms's built-in
/// canonical chromaticities (typical 100-700 u16 in saturated regions)
/// while still rejecting profiles with meaningfully-divergent matrices
/// (Apple per-device calibration profiles drift 1000-2500 u16 and produce
/// visibly different output from canonical matrix+TRC substitution). Real
/// LUT-driven gamut mapping produces deviations in the thousands to tens
/// of thousands of u16.
const COLORIMETRIC_VS_SYNTH_EPSILON_U16: u32 = 640;

/// Number of pixel samples per ramp.
const RAMP_STEPS: usize = 64;

fn build_synth_ref(primaries: &str, transfer: &str) -> Option<ColorProfile> {
    // moxcms ships built-in canonical references for exactly these combos.
    // For others (Smpte170m, AppleRgb, WideGamut, ColorMatch, EciRgbV2,
    // Bt470Bg, DciP3 etc.) we fall back to the structural rule.
    match (primaries, transfer) {
        ("Bt709", "Srgb") => Some(ColorProfile::new_srgb()),
        ("DisplayP3", "Srgb") => Some(ColorProfile::new_display_p3()),
        ("Bt2020", "Bt709") | ("Bt2020", "Srgb") => Some(ColorProfile::new_bt2020()),
        ("Bt2020", "Pq") => Some(ColorProfile::new_bt2020_pq()),
        ("Bt2020", "Hlg") => Some(ColorProfile::new_bt2020_hlg()),
        ("DisplayP3", "Pq") => Some(ColorProfile::new_display_p3_pq()),
        ("AdobeRgb", "Gamma22") => Some(ColorProfile::new_adobe_rgb()),
        ("ProPhoto", "Gamma18") => Some(ColorProfile::new_pro_photo_rgb()),
        _ => None,
    }
}

fn make_ramp() -> Vec<u16> {
    let mut pixels = Vec::with_capacity(RAMP_STEPS * 5 * 3);
    // Gray ramp
    for i in 0..RAMP_STEPS {
        let v = ((i as f64 / (RAMP_STEPS - 1) as f64) * 65535.0) as u16;
        pixels.extend_from_slice(&[v, v, v]);
    }
    // Pure R/G/B ramps
    for ch in 0..3 {
        for i in 0..RAMP_STEPS {
            let v = ((i as f64 / (RAMP_STEPS - 1) as f64) * 65535.0) as u16;
            let mut px = [0u16; 3];
            px[ch] = v;
            pixels.extend_from_slice(&px);
        }
    }
    // Mid-white + mixed hues
    let mix: [[u16; 3]; RAMP_STEPS] = {
        let mut arr = [[0u16; 3]; RAMP_STEPS];
        for (i, p) in arr.iter_mut().enumerate() {
            let t = i as f64 / (RAMP_STEPS - 1) as f64;
            // Arbitrary mix to probe 3D interior.
            p[0] = ((t * 0.9 + 0.05) * 65535.0) as u16;
            p[1] = (((1.0 - t) * 0.8 + 0.1) * 65535.0) as u16;
            p[2] = ((((t * 2.0) % 1.0) * 0.7 + 0.15) * 65535.0) as u16;
        }
        arr
    };
    for p in &mix {
        pixels.extend_from_slice(p);
    }
    pixels
}

fn transform_ramp(
    src: &ColorProfile,
    dst: &ColorProfile,
    intent: RenderingIntent,
    ramp: &[u16],
) -> Option<Vec<u16>> {
    let opts = TransformOptions {
        rendering_intent: intent,
        ..Default::default()
    };
    let t = src
        .create_transform_16bit(Layout::Rgb, dst, Layout::Rgb, opts)
        .ok()?;
    let mut out = vec![0u16; ramp.len()];
    t.transform(ramp, &mut out).ok()?;
    Some(out)
}

fn max_pair_err(a: &[u16], b: &[u16]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// Result of the empirical intent-safety measurement.
enum EmpiricalMask {
    /// Authoritative mask derived from moxcms pixel comparison.
    Measured(u8),
    /// moxcms couldn't parse the profile or couldn't build the baseline
    /// transform. Fall back to structural check.
    NotAvailable,
}

/// Compute the empirical intent-safety mask.
///
/// * COLORIMETRIC — set iff moxcms(icc, RelColor) is close to
///   moxcms(synth, RelColor) within `COLORIMETRIC_VS_SYNTH_EPSILON_U16`.
///   This tells us the profile's RelColor pipeline is essentially
///   matrix+TRC (i.e., either no A2B0/B2A0 LUT, or the LUTs reproduce
///   canonical output).
/// * PERCEPTUAL / SATURATION — set iff moxcms(icc, Perceptual|Saturation)
///   matches moxcms(icc, RelColor) within `INTENT_VS_INTENT_EPSILON_U16`.
///   This tells us the other intents don't pick a different LUT.
fn measure_intent_mask(icc_data: &[u8], primaries: &str, transfer: &str) -> EmpiricalMask {
    let Ok(icc) = ColorProfile::new_from_slice(icc_data) else {
        return EmpiricalMask::NotAvailable;
    };
    // If there's no synth reference for this (primaries, transfer) we
    // can't empirically grant the COLORIMETRIC bit, and PERCEPTUAL /
    // SATURATION would be meaningless without it. Let the caller decide.
    let Some(synth) = build_synth_ref(primaries, transfer) else {
        return EmpiricalMask::NotAvailable;
    };
    let dst = ColorProfile::new_srgb();
    let ramp = make_ramp();

    let Some(rel_color) = transform_ramp(&icc, &dst, RenderingIntent::RelativeColorimetric, &ramp)
    else {
        return EmpiricalMask::NotAvailable;
    };
    let Some(synth_out) =
        transform_ramp(&synth, &dst, RenderingIntent::RelativeColorimetric, &ramp)
    else {
        return EmpiricalMask::NotAvailable;
    };

    let mut mask = 0u8;

    // COLORIMETRIC bit: icc(RelColor) vs synth(RelColor).
    let colorimetric_ok = max_pair_err(&rel_color, &synth_out) <= COLORIMETRIC_VS_SYNTH_EPSILON_U16;
    if colorimetric_ok {
        mask |= INTENT_COLORIMETRIC_SAFE;

        // PERCEPTUAL / SATURATION bits are only meaningful when
        // COLORIMETRIC is already set — if the RelColor path isn't
        // matrix+TRC-safe, no other intent is either.
        for (intent, bit) in [
            (RenderingIntent::Perceptual, INTENT_PERCEPTUAL_SAFE),
            (RenderingIntent::Saturation, INTENT_SATURATION_SAFE),
        ] {
            let Some(alt) = transform_ramp(&icc, &dst, intent, &ramp) else {
                continue;
            };
            if max_pair_err(&alt, &rel_color) <= INTENT_VS_INTENT_EPSILON_U16 {
                mask |= bit;
            }
        }
    }
    EmpiricalMask::Measured(mask)
}

/// Which A2B/B2A LUT tags does the profile carry?
///
/// Each bit indicates presence of the corresponding LUT — if set, a CMS
/// would prefer that LUT over matrix math for the matching intent, so
/// matrix+TRC substitution is unsafe for that intent.
#[derive(Clone, Copy, Default)]
struct LutTags {
    a2b0: bool,
    a2b1: bool,
    a2b2: bool,
    b2a0: bool,
    b2a1: bool,
    b2a2: bool,
}

fn scan_lut_tags(data: &[u8]) -> LutTags {
    let mut out = LutTags::default();
    if data.len() < 132 {
        return out;
    }
    let tc = u32::from_be_bytes([data[128], data[129], data[130], data[131]]) as usize;
    for i in 0..tc.min(200) {
        let b = 132 + i * 12;
        if b + 12 > data.len() {
            break;
        }
        match &data[b..b + 4] {
            b"A2B0" => out.a2b0 = true,
            b"A2B1" => out.a2b1 = true,
            b"A2B2" => out.a2b2 = true,
            b"B2A0" => out.b2a0 = true,
            b"B2A1" => out.b2a1 = true,
            b"B2A2" => out.b2a2 = true,
            _ => {}
        }
    }
    out
}

fn has_any_lut(t: LutTags) -> bool {
    t.a2b0 || t.a2b1 || t.a2b2 || t.b2a0 || t.b2a1 || t.b2a2
}

/// Structural intent-safety mask for grayscale (moxcms has no built-in
/// gray reference, so we can't do empirical comparison).
///
/// A bit is set iff the corresponding pair of A2B/B2A LUTs is absent.
/// Matches the convention used by the previous static-feature generator:
/// - bit 0 (colorimetric): no A2B0/B2A0
/// - bit 1 (perceptual):   no A2B1/B2A1
/// - bit 2 (saturation):   no A2B2/B2A2
fn gray_intent_mask(data: &[u8]) -> u8 {
    let t = scan_lut_tags(data);
    let mut mask = 0u8;
    if !t.a2b0 && !t.b2a0 {
        mask |= INTENT_COLORIMETRIC_SAFE;
    }
    if !t.a2b1 && !t.b2a1 {
        mask |= INTENT_PERCEPTUAL_SAFE;
    }
    if !t.a2b2 && !t.b2a2 {
        mask |= INTENT_SATURATION_SAFE;
    }
    mask
}

/// Compute the intent-safety mask for an RGB profile.
///
/// Structural rule (LUT tag presence) is the baseline, because moxcms's
/// matrix+TRC pipeline is what runtime substitution uses and any
/// matrix-shaper profile without LUTs routes through it.
///
/// Empirical moxcms pixel comparison can UPGRADE bits — a profile with
/// structurally-disqualifying features (Lab PCS, non-Bradford chad, or
/// LUTs) may still produce matrix+TRC-equivalent output through moxcms
/// in practice. When the empirical test confirms this, we grant the bit.
///
/// Empirical comparison does NOT downgrade bits: a matrix-shaper profile
/// is by definition a matrix+TRC profile, even if its encoded matrix
/// drifts slightly from our canonical. Runtime substitution still uses
/// canonical matrix+TRC, which is what users expect when selecting a
/// well-known color space via CICP.
fn compute_rgb_intent_mask(data: &[u8], primaries: &str, transfer: &str) -> u8 {
    let structural = structural_rgb_mask(data);
    let Some(empirical) = measure_empirical_intent_bits(data, primaries, transfer) else {
        return structural;
    };
    // OR combines signals: a bit is safe if either the structural rule
    // or the empirical test passes.
    structural | empirical
}

/// Structural fallback: derive intent safety from LUT tag presence alone.
///
/// Used when no synth reference is available for empirical measurement
/// (Smpte170m, AppleRgb, WideGamut, ColorMatch, EciRgbV2, Bt470Bg, etc.).
fn structural_rgb_mask(data: &[u8]) -> u8 {
    let luts = scan_lut_tags(data);
    if !has_any_lut(luts) {
        // No LUTs: a CMS has only one possible path (matrix+TRC via
        // colorants), so all intents are equivalent.
        return INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE;
    }
    let mut m = 0u8;
    if !luts.a2b0 && !luts.b2a0 {
        m |= INTENT_COLORIMETRIC_SAFE;
    }
    if !luts.a2b1 && !luts.b2a1 {
        m |= INTENT_PERCEPTUAL_SAFE;
    }
    if !luts.a2b2 && !luts.b2a2 {
        m |= INTENT_SATURATION_SAFE;
    }
    m
}

/// Run the empirical pixel comparison. Returns None if no synth reference
/// is available or moxcms fails to parse.
fn measure_empirical_intent_bits(icc_data: &[u8], primaries: &str, transfer: &str) -> Option<u8> {
    match measure_intent_mask(icc_data, primaries, transfer) {
        EmpiricalMask::Measured(m) => Some(m),
        EmpiricalMask::NotAvailable => None,
    }
}

// ── lcms2 cross-check (optional) ─────────────────────────────────────────

#[cfg(feature = "lcms2-crosscheck")]
fn lcms2_crosscheck_note(icc_data: &[u8]) -> String {
    // Currently a smoke test only — we verify the profile parses in lcms2.
    // Richer comparison could be added once we decide what signal we want.
    match lcms2::Profile::new_icc(icc_data) {
        Ok(_) => "lcms2:ok".into(),
        Err(e) => format!("lcms2:err({e})"),
    }
}

#[cfg(not(feature = "lcms2-crosscheck"))]
fn lcms2_crosscheck_note(_icc_data: &[u8]) -> String {
    String::new()
}

// ── .inc formatting ──────────────────────────────────────────────────────

fn fmt_hash(h: u64) -> String {
    let s = format!("{h:016x}");
    format!("0x{}_{}_{}_{}", &s[0..4], &s[4..8], &s[8..12], &s[12..16])
}

/// Format the intent mask as a numeric literal with a short comment.
fn fmt_mask(m: u8) -> String {
    format!("0b{m:03b}")
}

/// Human-readable mask breakdown for the comment column.
fn mask_breakdown(m: u8) -> &'static str {
    match m {
        0 => "-",
        0b001 => "C",
        0b010 => "P",
        0b100 => "S",
        0b011 => "CP",
        0b101 => "CS",
        0b110 => "PS",
        0b111 => "CPS",
        _ => "?",
    }
}

fn clean_desc(fname: &str) -> String {
    fname
        .replace(".icc", "")
        .replace('_', " ")
        .chars()
        .take(50)
        .collect()
}

// ── Main ─────────────────────────────────────────────────────────────────

struct RgbRow {
    hash: u64,
    cp: &'static str,
    tf: &'static str,
    max_err: u32,
    intent_mask: u8,
    desc: String,
}

struct GrayRow {
    hash: u64,
    tf: &'static str,
    max_err: u32,
    intent_mask: u8,
    desc: String,
}

fn collect_entries(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in dirs {
        if let Ok(rd) = std::fs::read_dir(dir) {
            out.extend(
                rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| {
                    matches!(p.extension().and_then(|e| e.to_str()), Some("icc" | "icm"))
                }),
            );
        }
    }
    out.sort();
    out
}

fn parse_args() -> (Vec<PathBuf>, PathBuf) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.len() {
        0 => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            let cache = PathBuf::from(&home).join(".cache/zenpixels-icc");
            let bundled = PathBuf::from("zenpixels-convert/src/profiles");
            let out = PathBuf::from("zenpixels/src/icc");
            (vec![cache, bundled], out)
        }
        1 => {
            let out = PathBuf::from("zenpixels/src/icc");
            (vec![PathBuf::from(&args[0])], out)
        }
        _ => {
            let out = PathBuf::from(args.last().unwrap());
            let inputs: Vec<PathBuf> = args[..args.len() - 1].iter().map(PathBuf::from).collect();
            (inputs, out)
        }
    }
}

fn write_rgb(rows: &BTreeMap<u64, RgbRow>, out_path: &Path) {
    let mut s = String::from("&[\n");
    for row in rows.values() {
        s += &format!(
            "    ({}, CP::{}, TF::{}, {:>2}, {}),  // [{}] {}\n",
            fmt_hash(row.hash),
            row.cp,
            row.tf,
            row.max_err,
            fmt_mask(row.intent_mask),
            mask_breakdown(row.intent_mask),
            clean_desc(&row.desc),
        );
    }
    s += "]\n";
    std::fs::write(out_path, s).unwrap();
}

fn write_gray(rows: &BTreeMap<u64, GrayRow>, out_path: &Path) {
    let mut s = String::from("&[\n");
    for row in rows.values() {
        s += &format!(
            "    ({}, TF::{}, {:>2}, {}),  // [{}] {}\n",
            fmt_hash(row.hash),
            row.tf,
            row.max_err,
            fmt_mask(row.intent_mask),
            mask_breakdown(row.intent_mask),
            clean_desc(&row.desc),
        );
    }
    s += "]\n";
    std::fs::write(out_path, s).unwrap();
}

fn main() {
    let (input_dirs, out_dir) = parse_args();

    for dir in &input_dirs {
        if !dir.exists() {
            eprintln!("Directory not found: {}", dir.display());
            eprintln!("Usage: icc-gen <dir1> [dir2 ...] <out-dir>");
            std::process::exit(1);
        }
    }

    let entries = collect_entries(&input_dirs);
    eprintln!(
        "Scanning {} files across {} dirs...",
        entries.len(),
        input_dirs.len()
    );

    let mut rgb: BTreeMap<u64, RgbRow> = BTreeMap::new();
    let mut gray: BTreeMap<u64, GrayRow> = BTreeMap::new();
    let mut scanned = 0u32;
    let mut skipped = 0u32;

    for (idx, path) in entries.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if data.len() < 132 {
            continue;
        }
        scanned += 1;

        if scanned % 50 == 0 {
            eprintln!(
                "  [{}/{}] {}",
                idx + 1,
                entries.len(),
                path.file_name().unwrap().to_string_lossy()
            );
        }

        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let cs = &data[16..20];
        let norm_hash = fnv1a_64_normalized(&data);

        if cs == b"RGB " {
            let Some(cp_name) = identify_primaries(&data) else {
                skipped += 1;
                continue;
            };
            let Some((tf_name, err)) = identify_trc(&data, b"rTRC") else {
                skipped += 1;
                continue;
            };
            if err > 56 {
                skipped += 1;
                continue;
            }

            // Empirical intent-safety check only once per unique hash — this
            // is where the cost is (moxcms builds + transforms two profiles
            // over a 320-pixel ramp for each of 3 intents, ~1-5ms).
            let mask = compute_rgb_intent_mask(&data, cp_name, tf_name);
            rgb.entry(norm_hash)
                .and_modify(|row| {
                    row.max_err = row.max_err.max(err);
                    // AND masks when multiple profiles collapse to the
                    // same normalized hash: the result must be safe for
                    // every collapsed input.
                    row.intent_mask &= mask;
                })
                .or_insert(RgbRow {
                    hash: norm_hash,
                    cp: cp_name,
                    tf: tf_name,
                    max_err: err,
                    intent_mask: mask,
                    desc: fname.clone(),
                });
        } else if cs == b"GRAY" {
            let trc = identify_trc(&data, b"kTRC").or_else(|| identify_trc(&data, b"gTRC"));
            let Some((tf_name, err)) = trc else {
                skipped += 1;
                continue;
            };
            if err > 56 {
                skipped += 1;
                continue;
            }

            let mask = gray_intent_mask(&data);
            gray.entry(norm_hash)
                .and_modify(|row| {
                    row.max_err = row.max_err.max(err);
                    row.intent_mask &= mask;
                })
                .or_insert(GrayRow {
                    hash: norm_hash,
                    tf: tf_name,
                    max_err: err,
                    intent_mask: mask,
                    desc: fname.clone(),
                });
        }
        let _ = lcms2_crosscheck_note(&data);
    }

    // ── Write tables ──────────────────────────────────────────────────

    let rgb_path = out_dir.join("icc_table_rgb.inc");
    let gray_path = out_dir.join("icc_table_gray.inc");
    write_rgb(&rgb, &rgb_path);
    write_gray(&gray, &gray_path);

    // ── Summary ───────────────────────────────────────────────────────

    eprintln!("Scanned: {scanned} profiles");
    eprintln!("Skipped: {skipped} (unknown primaries or TRC error >56)");
    eprintln!("RGB:  {} entries → {}", rgb.len(), rgb_path.display());
    eprintln!("Gray: {} entries → {}", gray.len(), gray_path.display());

    let mut combos: BTreeMap<String, usize> = BTreeMap::new();
    let mut mask_counts: BTreeMap<u8, usize> = BTreeMap::new();
    for row in rgb.values() {
        *combos.entry(format!("{}+{}", row.cp, row.tf)).or_default() += 1;
        *mask_counts.entry(row.intent_mask).or_default() += 1;
    }
    eprintln!("\nRGB by (primaries, transfer):");
    let mut sorted: Vec<_> = combos.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in sorted {
        eprintln!("  {k}: {v}");
    }
    eprintln!("\nRGB by intent_mask:");
    let mut sorted_masks: Vec<_> = mask_counts.iter().collect();
    sorted_masks.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in sorted_masks {
        eprintln!("  {} [{}]: {}", fmt_mask(*k), mask_breakdown(*k), v);
    }
}
