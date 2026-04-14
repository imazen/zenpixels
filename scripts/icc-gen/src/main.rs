//! Generate ICC hash table `.inc` files for `zenpixels::icc`.
//!
//! Scans ICC profiles, computes normalized FNV-1a hashes, identifies primaries
//! and TRC (measuring max u16 error against reference EOTFs for all 65536
//! values), deduplicates by normalized hash, and runs moxcms transforms per
//! rendering intent to derive an empirical intent-safety mask. Under the
//! default `lcms2-crosscheck` feature the mask is AND-gated against Little CMS
//! 2's interpretation of the same profile — both CMSs must agree the profile
//! matches canonical. Writes Rust include files.
//!
//! Usage:
//!   cargo run -p icc-gen --release -- <icc-cache-dir> <bundled-profiles-dir> <out-dir>
//!
//! `lcms2-crosscheck` is enabled by default and requires the Little CMS 2 C
//! library at system level (`apt install liblcms2-dev` on Debian/Ubuntu,
//! `brew install little-cms2` on macOS). To run without it:
//!   cargo run -p icc-gen --release --no-default-features -- ...

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
    TransformOptions,
};

/// Transform options for maximum-precision validation.
///
/// Mirrors `zenpixels_convert::cms_moxcms::transform_opts(
/// ColorPriority::PreferIcc, intent)` for the *correctness* knobs (no CICP
/// shortcut, Tetrahedral interpolation, High barycentric weights), but flips
/// `prefer_fixed_point` to `false` so the comparison measures true profile
/// divergence rather than Q2.13 quantization noise.
///
/// Runtime drift from this baseline is strictly additive — Q2.13 fixed-point
/// adds ~1-4 u16 noise on top of the float result without ever closing a
/// genuine divergence gap. A profile that passes the empirical comparison
/// here will also pass at runtime; one that fails here may still fail at
/// runtime by an even larger margin.
///
/// Critical: we DO NOT enable the `extended_range` feature on moxcms. That
/// feature's `try_extended_gamma_evaluator` (moxcms src/trc.rs:1293-1523)
/// detects sRGB / BT.709 parametric curves by 1e-4 parameter tolerance and
/// silently substitutes hardcoded canonical TRC code, which would mask any
/// real divergence in profile-encoded curves. See the icc-gen Cargo.toml.
fn runtime_opts(intent: RenderingIntent) -> TransformOptions {
    TransformOptions {
        rendering_intent: intent,
        allow_use_cicp_transfer: false,
        prefer_fixed_point: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
    }
}

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

/// Returns `true` if the given TRC tag is a `parametricCurveType` with
/// `funcType=3` (the ICC-v4 form with a linear toe segment near black).
///
/// Adobe RGB paraType-3 profiles are excluded from the identification table
/// because zenpixels-convert's matrix+TRC substitution uses pure gamma
/// 2.19921875 (matching the Adobe RGB 1998 encoding spec and ~85% of
/// real-world Adobe RGB profiles in the wild) — so a profile that encodes
/// the minority paraType-3 toe form would render visibly different in the
/// fast path than through a real CMS. Better to let the CMS honor the
/// exact encoded curve.
fn is_para_functype3(d: &[u8], trc_tag: &[u8; 4]) -> bool {
    let Some(off) = find_tag(d, trc_tag) else {
        return false;
    };
    if off + 10 > d.len() || &d[off..off + 4] != b"para" {
        return false;
    }
    let ft = u16::from_be_bytes([d[off + 8], d[off + 9]]);
    ft == 3
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

// ── Identification-table exclusion categories ────────────────────────────
//
// Named references into `KNOWN_P::rust_name`. The `rust_name` field is what
// the generator emits into the `.inc` table, so keeping the exclusion
// logic string-based on `rust_name` keeps these checks in sync with what
// the output declares. `assert_exclusion_names_valid()` verifies the names
// resolve against `KNOWN_P` at startup in debug builds.

/// Skip all profiles whose identified primaries appear in this list — no
/// fast-path acceleration; callers take the full-CMS path.
///
/// **ProPhoto / ROMM**: the in-the-wild encoding is too fragmented to
/// justify a single canonical form. Surveying real profiles:
/// - ~50% encode pure gamma 1.8 (Windows `ProPhoto.icm`,
///   `rommrgb`, Linux `rommrgb`/`ProPhotoRGB`, saucecontrol
///   pure-gamma variants)
/// - ~30% encode `paraType funcType=3` with a linear toe:
///   saucecontrol `ProPhoto-v4` uses the ISO 22028-2 form (`c=1/16,
///   d=1/32`); Apple's macOS `ROMM RGB` uses a non-standard
///   `c=1/16, d=1/512`
/// - 1 profile (`ProPhotoLin.icm`) encodes a linear TRC despite the name
/// - Two ISO 22028-2 v4 profiles are mAB/mBA LUTs with no rTRC at all
///
/// Picking any single "canonical" reference would misrender the others.
/// Dropping ProPhoto from the acceleration set entirely lets full CMS
/// honor each profile's exact encoded curve.
const EXCLUDED_PRIMARIES: &[&str] = &["ProPhoto"];

/// Skip profiles with these primaries *if* their rTRC is
/// `parametricCurveType funcType=3`.
///
/// **Adobe RGB**: the spec body (§4.3.4.2) defines pure gamma 2.19921875
/// with no toe. Annex C (Informative) *recommends* a slope limit of 1/32
/// when implementing the inverse transfer — `max(C'^2.19921875, C'/32)` —
/// which is mathematically equivalent to `paraType funcType=3` with
/// `c=1/32, d=0.05568`. Annex C explicitly states the slope limit is "an
/// implementation aspect, not an attribute of the Adobe RGB (1998) color
/// space encoding" and that "different implementations may use different
/// slope limits."
///
/// Our runtime (and the bundled `ADOBE_RGB` profile) both use pure gamma,
/// matching the spec body and ~85% of real-world Adobe RGB profiles
/// (Adobe CS4, Windows `ClayRGB1998` / `AdobeRGB1998`, macOS
/// `AdobeRGB1998`, Linux `AdobeRGB1998`/`compatibleWithAdobeRGB1998`,
/// Nikon, per-camera profiles). The ~15% that encode paraType-3 (notably
/// saucecontrol `AdobeCompat-v4`, ACE-generated profiles following Annex
/// C) would render with slight shadow lift (8-bit codes 1-14 per Annex C)
/// if we ran them through the pure-gamma fast path. We let them fall
/// through to full CMS so their exact encoded curve is honored.
const PARA3_EXCLUDED_PRIMARIES: &[&str] = &["AdobeRgb"];

fn is_excluded_primary(cp_name: &str) -> bool {
    EXCLUDED_PRIMARIES.contains(&cp_name)
}

fn is_para3_excluded_primary(cp_name: &str) -> bool {
    PARA3_EXCLUDED_PRIMARIES.contains(&cp_name)
}

#[cfg(debug_assertions)]
fn assert_exclusion_names_valid() {
    for name in EXCLUDED_PRIMARIES.iter().chain(PARA3_EXCLUDED_PRIMARIES) {
        assert!(
            KNOWN_P.iter().any(|kp| kp.rust_name == *name),
            "exclusion list references unknown primaries name {name:?}"
        );
    }
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
// Maximally-conservative policy: when moxcms can build a baseline transform
// for the identified (primaries, transfer), we trust the EMPIRICAL result
// alone. Structural inspection (LUT-tag presence, matrix-shaper layout)
// only acts as a fallback when no synth reference exists, because even a
// "clean" matrix-shaper profile can carry a non-Bradford `chad`, custom
// per-device chromaticities, or other state that makes runtime substitution
// disagree with what lcms2/moxcms would produce.
//
// Comparison: `moxcms(icc, intent)` (uses the profile's exact bytes,
// including chad/A2B_n/B2A_n) vs `moxcms(synth_canonical, intent)` (uses
// canonical primaries with Bradford chad — what zenpixels-convert's runtime
// matrix+TRC substitution actually does). Bit is granted only if the two
// agree within COLORIMETRIC_VS_SYNTH_EPSILON_U16 across the whole probe.
// We do not OR-merge with structural — that would mask legitimate divergences.

/// Bit flags matching zenpixels/src/icc/mod.rs.
const INTENT_COLORIMETRIC_SAFE: u8 = 1 << 0;
const INTENT_PERCEPTUAL_SAFE: u8 = 1 << 1;
const INTENT_SATURATION_SAFE: u8 = 1 << 2;

/// Max u16 deviation for the intent-vs-RelColor comparison to count as
/// "intent collapses to colorimetric." 64/65535 ≈ 0.1% — about ¼ of an
/// 8-bit step, matching the `Tolerance::Intent` TRC threshold.
const INTENT_VS_INTENT_EPSILON_U16: u32 = 64;

/// Max u16 deviation between `moxcms(icc, intent=RelColor)` and
/// `moxcms(synth_ref, intent=RelColor)` for the COLORIMETRIC bit.
///
/// 256/65535 ≈ 0.39% — exactly one 8-bit code step. Profiles whose CMS
/// output rounds to the same u8 value as our canonical matrix+TRC
/// substitution pass; profiles whose output would round to a different u8
/// (≈ visibly distinguishable in 8-bit pipelines) are rejected. This
/// catches non-Bradford `chad` adaptations, custom encoded primaries,
/// LUT-driven gamut remapping, and per-device calibration drift.
///
/// Note: for pure matrix-shaper profiles (no LUT tags), the s15.16
/// quantization floor can exceed this threshold for wider gamuts (~275
/// for BT.709, ~590 for BT.2020, ~930 for Display P3). That's handled
/// separately by OR-merging with the structural mask in
/// `compute_rgb_intent_mask` — the structural rule correctly grants all
/// bits when no LUTs are present, since the only CMS path is matrix+TRC.
const COLORIMETRIC_VS_SYNTH_EPSILON_U16: u32 = 256;

/// Looser epsilon for the lcms2 AND-gate comparison (when the
/// `lcms2-crosscheck` feature is enabled).
///
/// 512/65535 ≈ 0.78% — two 8-bit code steps. Rationale: even for
/// canonical sRGB profiles, the `paraType funcType=3` representation
/// used by lcms2's `Profile::new_srgb()` differs slightly from how the
/// same sRGB curve is encoded in third-party profiles (RawTherapee,
/// skcms, colord, etc.) — different parameter quantizations and
/// occasionally `curv count=N` LUT forms in place of paraType. Those
/// representation differences show up as 260-500 u16 deltas through
/// lcms2's transform, even though the profiles all describe the same
/// canonical curve. We use 2× the moxcms epsilon here so the AND-gate
/// rejects only *real* profile divergence (chad drift, non-canonical
/// primaries, LUT-driven gamut) and not parametric-representation noise.
const LCMS2_ANDGATE_EPSILON_U16: u32 = 2 * COLORIMETRIC_VS_SYNTH_EPSILON_U16;

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
        // ProPhoto deliberately omitted — ecosystem is too fragmented (pure
        // gamma, paraType-3 ISO d=1/32, Apple's d=1/512, linear variants,
        // mAB/mBA LUT profiles) to justify a single fast-path reference.
        // All ProPhoto profiles fall through to full CMS for faithful
        // rendering; see `exclude_adobe_non_pure_gamma` / `is_prophoto_*`
        // below and the notes in `TransferFunction::Gamma18`'s doc string.
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
    let t = src
        .create_transform_16bit(Layout::Rgb, dst, Layout::Rgb, runtime_opts(intent))
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
/// Bit is granted only when the profile's moxcms output matches the
/// canonical synth reference within epsilon. When the `lcms2-crosscheck`
/// feature is enabled, we additionally require lcms2's output from the
/// same profile to match moxcms's canonical synth within the same epsilon
/// (AND-gate). Rationale: moxcms and lcms2 should produce equivalent
/// results for well-formed matrix-shaper profiles; any divergence flags a
/// profile where one CMS is doing something the other isn't, and we'd
/// rather reject such profiles than ship an identification that could
/// render differently under a different CMS downstream.
///
/// * COLORIMETRIC — moxcms(icc, RelCol) ≈ moxcms(synth, RelCol), AND (if
///   lcms2 enabled) lcms2(icc, RelCol) ≈ moxcms(synth, RelCol), both
///   within `COLORIMETRIC_VS_SYNTH_EPSILON_U16`.
/// * PERCEPTUAL / SATURATION — moxcms says intent ≈ RelCol for this
///   profile, AND (if lcms2 enabled) lcms2 says the same, both within
///   `INTENT_VS_INTENT_EPSILON_U16`.
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

    // lcms2 AND-gate: if the feature is enabled, every bit must ALSO pass
    // an lcms2-side check (lcms2 comparing the profile against the same
    // canonical, through lcms2's own math). Comparing lcms2(icc) against
    // lcms2(synth_canonical) rather than moxcms(synth_out) keeps both
    // sides in the same CMS and cancels the ~225 u16 cross-CMS noise
    // floor (f32/f64 `pow` precision drift) so the gate measures real
    // profile divergence, not libm differences.
    //
    // For sRGB we use lcms2's built-in `Profile::new_srgb()` as the
    // reference — no moxcms encode round-trip. For other combos we fall
    // back to feeding lcms2 the moxcms-encoded bytes; the s15.16
    // fixed-point drift there is ~150-300 u16 and pushes a handful of
    // profiles over the threshold. See `lcms2_synth_ramp`.
    let synth_bytes = synth.encode().ok();
    let lcms2_rel_color = lcms2_transform_ramp(icc_data, RenderingIntent::RelativeColorimetric);
    let lcms2_synth = lcms2_synth_ramp(
        primaries,
        transfer,
        synth_bytes.as_deref(),
        RenderingIntent::RelativeColorimetric,
    );

    let mut mask = 0u8;

    // COLORIMETRIC bit: icc(RelColor) vs synth(RelColor).
    let mox_col_ok = max_pair_err(&rel_color, &synth_out) <= COLORIMETRIC_VS_SYNTH_EPSILON_U16;
    let lcms_col_ok = lcms2_agrees_with_synth(lcms2_rel_color.as_deref(), lcms2_synth.as_deref());
    if mox_col_ok && lcms_col_ok {
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
            let mox_intent_ok = max_pair_err(&alt, &rel_color) <= INTENT_VS_INTENT_EPSILON_U16;
            let lcms_alt = lcms2_transform_ramp(icc_data, intent);
            let lcms_intent_ok =
                lcms2_intent_agrees(lcms_alt.as_deref(), lcms2_rel_color.as_deref());
            if mox_intent_ok && lcms_intent_ok {
                mask |= bit;
            }
        }
    }
    EmpiricalMask::Measured(mask)
}

/// Check whether lcms2's output on the ICC profile agrees with lcms2's
/// output on the canonical synth reference within the colorimetric epsilon.
///
/// Both outputs go through lcms2, so the cross-CMS noise floor
/// (~225 u16 of `pow` precision drift between moxcms's f32 and lcms2's f64)
/// cancels and we measure the profile's actual divergence from canonical
/// as lcms2 sees it.
///
/// When the `lcms2-crosscheck` feature is disabled, both args are always
/// `None` and we return `true` (no extra gate). When the feature is
/// enabled, `None` on either side means we couldn't run lcms2 — be
/// conservative and withhold the bit to avoid granting on a single-CMS
/// signal.
fn lcms2_agrees_with_synth(lcms2_icc: Option<&[u16]>, lcms2_synth: Option<&[u16]>) -> bool {
    if cfg!(feature = "lcms2-crosscheck") {
        match (lcms2_icc, lcms2_synth) {
            (Some(icc), Some(synth)) => max_pair_err(icc, synth) <= LCMS2_ANDGATE_EPSILON_U16,
            _ => false,
        }
    } else {
        // No AND-gate when lcms2 is not compiled in.
        true
    }
}

/// Check whether lcms2 confirms `intent` collapses to RelCol for the same
/// profile (intra-profile intent equality, within the tighter ε).
fn lcms2_intent_agrees(lcms2_intent: Option<&[u16]>, lcms2_rel_col: Option<&[u16]>) -> bool {
    if cfg!(feature = "lcms2-crosscheck") {
        match (lcms2_intent, lcms2_rel_col) {
            (Some(alt), Some(rel)) => max_pair_err(alt, rel) <= INTENT_VS_INTENT_EPSILON_U16,
            // Either transform failed — can't confirm; be conservative.
            _ => false,
        }
    } else {
        true
    }
}

/// Run lcms2 at the given intent, returning the u16 ramp output.
/// Mirrors the configuration used in `lcms2_crosscheck_deltas`
/// (`NO_OPTIMIZE | HIGHRES_PRECALC`, float-internal, pure-analytic paths).
#[cfg(feature = "lcms2-crosscheck")]
fn lcms2_transform_ramp(icc_data: &[u8], intent: RenderingIntent) -> Option<Vec<u16>> {
    use lcms2::Profile;
    let lcms_src = Profile::new_icc(icc_data).ok()?;
    lcms2_transform_from_profile(&lcms_src, intent)
}

#[cfg(feature = "lcms2-crosscheck")]
fn lcms2_transform_from_profile(src: &lcms2::Profile, intent: RenderingIntent) -> Option<Vec<u16>> {
    use lcms2::{Flags, Intent, PixelFormat as LPx, Profile, Transform};

    let lcms_dst = Profile::new_srgb();
    let lcms_intent = match intent {
        RenderingIntent::RelativeColorimetric => Intent::RelativeColorimetric,
        RenderingIntent::AbsoluteColorimetric => Intent::AbsoluteColorimetric,
        RenderingIntent::Perceptual => Intent::Perceptual,
        RenderingIntent::Saturation => Intent::Saturation,
    };
    let flags = Flags::NO_OPTIMIZE | Flags::HIGHRES_PRECALC;
    let xform: Transform<[u16; 3], [u16; 3]> =
        Transform::new_flags(src, LPx::RGB_16, &lcms_dst, LPx::RGB_16, lcms_intent, flags).ok()?;
    let ramp = make_ramp();
    let mut out = vec![0u16; ramp.len()];
    let src_px: &[[u16; 3]] = bytemuck::cast_slice(&ramp);
    let dst_px: &mut [[u16; 3]] = bytemuck::cast_slice_mut(&mut out);
    xform.transform_pixels(src_px, dst_px);
    Some(out)
}

/// Build the lcms2-native canonical reference for a given (primaries,
/// transfer) combo, returning the same u16 ramp output. Prefers lcms2's
/// built-in constructors where available (sRGB) so the AND-gate doesn't
/// have to round-trip moxcms's encoded bytes through lcms2 (which adds
/// s15.16/fixed-point drift to the comparison). Falls back to parsing
/// moxcms-encoded synth bytes for combos where lcms2 has no built-in.
#[cfg(feature = "lcms2-crosscheck")]
fn lcms2_synth_ramp(
    primaries: &str,
    transfer: &str,
    moxcms_synth_bytes: Option<&[u8]>,
    intent: RenderingIntent,
) -> Option<Vec<u16>> {
    use lcms2::Profile;
    // lcms2 has a built-in sRGB constructor; prefer it for sRGB primaries
    // to avoid the encode/parse round-trip that introduces s15.16 noise.
    if (primaries, transfer) == ("Bt709", "Srgb") {
        return lcms2_transform_from_profile(&Profile::new_srgb(), intent);
    }
    // Other combos: feed lcms2 the moxcms-encoded synth bytes. Accepts the
    // s15.16 round-trip noise as the best we can do without handcrafting
    // lcms2-native canonicals for every combo.
    let bytes = moxcms_synth_bytes?;
    let profile = Profile::new_icc(bytes).ok()?;
    lcms2_transform_from_profile(&profile, intent)
}

#[cfg(not(feature = "lcms2-crosscheck"))]
fn lcms2_transform_ramp(_icc_data: &[u8], _intent: RenderingIntent) -> Option<Vec<u16>> {
    None
}

#[cfg(not(feature = "lcms2-crosscheck"))]
fn lcms2_synth_ramp(
    _primaries: &str,
    _transfer: &str,
    _moxcms_synth_bytes: Option<&[u8]>,
    _intent: RenderingIntent,
) -> Option<Vec<u16>> {
    None
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
/// Maximally-conservative policy: when the empirical moxcms comparison is
/// available (synth reference exists for the identified primaries+transfer),
/// it is the SOLE signal — we trust it over structural inspection. A profile
/// that "looks like" a clean matrix-shaper can still carry a non-Bradford
/// `chad`, custom per-device chromaticities, or LUTs that diverge from the
/// canonical matrix+TRC math used by runtime substitution.
///
/// Compute the intent-safety mask for an identified RGB profile.
///
/// The empirical check (`moxcms(icc) vs moxcms(synth)`) catches LUT-driven
/// intent divergence but is penalized by s15.16 quantization noise that
/// scales with gamut width (~275 u16 for BT.709, ~930 for Display P3).
/// The structural check (LUT tag presence) is immune to quantization but
/// can't detect behavioral differences inside LUTs.
///
/// We OR-merge both: the structural mask ensures that pure matrix-shaper
/// profiles (no LUT tags) get all bits — the only CMS path IS matrix+TRC,
/// so any delta is quantization noise, not a safety issue. The empirical
/// mask adds bits for profiles whose LUTs happen to produce the same
/// output as matrix math.
fn compute_rgb_intent_mask(data: &[u8], primaries: &str, transfer: &str) -> (u8, MaskProvenance) {
    let structural = structural_rgb_mask(data);
    match measure_empirical_intent_bits(data, primaries, transfer) {
        Some(empirical) => (empirical | structural, MaskProvenance::Empirical),
        None => (structural, MaskProvenance::Structural),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaskProvenance {
    Empirical,
    Structural,
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

/// Per-intent max u16 deltas: (relcol_vs_synth, perc_vs_relcol, sat_vs_relcol).
/// Each `Option` is None when the corresponding moxcms transform couldn't
/// be built. Used by --report mode for diagnostic output.
fn measure_intent_deltas(
    icc_data: &[u8],
    primaries: &str,
    transfer: &str,
) -> Option<(Option<u32>, Option<u32>, Option<u32>)> {
    let icc = ColorProfile::new_from_slice(icc_data).ok()?;
    let synth = build_synth_ref(primaries, transfer)?;
    let dst = ColorProfile::new_srgb();
    let ramp = make_ramp();

    let rel_color = transform_ramp(&icc, &dst, RenderingIntent::RelativeColorimetric, &ramp)?;
    let synth_out = transform_ramp(&synth, &dst, RenderingIntent::RelativeColorimetric, &ramp)?;
    let col_delta = max_pair_err(&rel_color, &synth_out);
    let perc_delta = transform_ramp(&icc, &dst, RenderingIntent::Perceptual, &ramp)
        .map(|alt| max_pair_err(&alt, &rel_color));
    let sat_delta = transform_ramp(&icc, &dst, RenderingIntent::Saturation, &ramp)
        .map(|alt| max_pair_err(&alt, &rel_color));
    Some((Some(col_delta), perc_delta, sat_delta))
}

// ── Report-mode rows ─────────────────────────────────────────────────────

struct ReportRow {
    name: String,
    cp: &'static str,
    tf: &'static str,
    final_mask: u8,
    provenance: MaskProvenance,
    structural_mask: u8,
    /// Per-intent moxcms-vs-canonical-synth deltas, or None if not measurable.
    deltas: Option<(Option<u32>, Option<u32>, Option<u32>)>,
    /// Per-intent moxcms-vs-lcms2 deltas (only with `lcms2-crosscheck` feature).
    /// Profiles where these differ from `deltas` indicate moxcms / lcms2
    /// disagreement and warrant manual inspection.
    lcms2_vs_mox: Option<CrossCheckDeltas>,
    luts: LutTags,
}

fn build_report_row(
    name: &str,
    cp: &'static str,
    tf: &'static str,
    data: &[u8],
    final_mask: u8,
    provenance: MaskProvenance,
) -> ReportRow {
    ReportRow {
        name: name.into(),
        cp,
        tf,
        final_mask,
        provenance,
        structural_mask: structural_rgb_mask(data),
        deltas: measure_intent_deltas(data, cp, tf),
        lcms2_vs_mox: lcms2_crosscheck_deltas(data),
        luts: scan_lut_tags(data),
    }
}

fn fmt_delta(d: Option<u32>) -> String {
    match d {
        Some(v) => format!("{v:>5}"),
        None => "  n/a".into(),
    }
}

fn fmt_lut_tags(t: LutTags) -> String {
    let mut parts = Vec::new();
    if t.a2b0 {
        parts.push("A2B0");
    }
    if t.a2b1 {
        parts.push("A2B1");
    }
    if t.a2b2 {
        parts.push("A2B2");
    }
    if t.b2a0 {
        parts.push("B2A0");
    }
    if t.b2a1 {
        parts.push("B2A1");
    }
    if t.b2a2 {
        parts.push("B2A2");
    }
    if parts.is_empty() {
        "(none)".into()
    } else {
        parts.join(",")
    }
}

fn print_report(rows: &[ReportRow]) {
    eprintln!("\n── ICC empirical-vs-structural disagreements ──────────────────────");
    eprintln!(
        "Threshold: COLORIMETRIC ≤{COLORIMETRIC_VS_SYNTH_EPSILON_U16} u16 vs synth, \
         PERC/SAT ≤{INTENT_VS_INTENT_EPSILON_U16} u16 vs RelCol\n"
    );
    let mut shown = 0usize;
    let mut by_kind: BTreeMap<&'static str, usize> = BTreeMap::new();
    for row in rows {
        if row.provenance != MaskProvenance::Empirical {
            continue;
        }
        if row.final_mask == row.structural_mask {
            continue;
        }
        // Disagreement: empirical and structural produce different masks.
        let kind = if row.final_mask & !row.structural_mask != 0 {
            // Empirical granted a bit structural rejected (LUT reproduces canonical).
            "empirical-upgraded"
        } else {
            // Structural granted a bit empirical rejected (chad/primaries drift).
            "empirical-downgraded"
        };
        *by_kind.entry(kind).or_default() += 1;
        let (cd, pd, sd) = row.deltas.unwrap_or_default();
        let xc = match row.lcms2_vs_mox {
            Some((c, p, s)) => format!(
                "  lcms2↔mox: c={} p={} s={}",
                fmt_delta(c),
                fmt_delta(p),
                fmt_delta(s)
            ),
            None => String::new(),
        };
        eprintln!(
            "  {:<12} {:<10}+{:<8} struct={} → final={} [{}]  Δcol={} Δperc={} Δsat={}  luts={}  {}{}",
            kind,
            row.cp,
            row.tf,
            fmt_mask(row.structural_mask),
            fmt_mask(row.final_mask),
            mask_breakdown(row.final_mask),
            fmt_delta(cd),
            fmt_delta(pd),
            fmt_delta(sd),
            fmt_lut_tags(row.luts),
            row.name,
            xc,
        );
        shown += 1;
    }
    eprintln!("\nDisagreement summary:");
    for (k, v) in &by_kind {
        eprintln!("  {k}: {v}");
    }
    eprintln!("Total disagreements shown: {shown}");
    let structural_only = rows
        .iter()
        .filter(|r| r.provenance == MaskProvenance::Structural)
        .count();
    if structural_only > 0 {
        eprintln!(
            "(Plus {structural_only} profiles with no synth reference — \
             structural-only verdict, see RGB-by-(cp,tf) summary for which combos.)"
        );
    }
}

// ── lcms2 cross-check (optional) ─────────────────────────────────────────
//
// When the `lcms2-crosscheck` feature is enabled, we run the same ramp through
// Little CMS 2 (the industry-standard reference) at each rendering intent and
// compare against moxcms's output. If the two CMSs agree, we're confident the
// profile is being interpreted correctly; if they disagree, the profile may
// have unusual content that exposes a bug or interpretation difference in one
// of them, and any decision based on either one's output should be flagged.

/// Per-intent (RelCol, Perceptual, Saturation) max u16 deltas between
/// `moxcms(icc, intent)` and `lcms2(icc, intent)` over the ramp. Each
/// `Option` is None when either CMS couldn't build a transform for that
/// intent on that profile.
type CrossCheckDeltas = (Option<u32>, Option<u32>, Option<u32>);

#[cfg(feature = "lcms2-crosscheck")]
fn lcms2_crosscheck_deltas(icc_data: &[u8]) -> Option<CrossCheckDeltas> {
    use lcms2::{Flags, Intent, PixelFormat as LPx, Profile, Transform};

    let lcms_src = Profile::new_icc(icc_data).ok()?;
    let lcms_dst = Profile::new_srgb();
    let mox_src = ColorProfile::new_from_slice(icc_data).ok()?;
    let mox_dst = ColorProfile::new_srgb();
    let ramp = make_ramp();

    let measure_one = |mox_intent: RenderingIntent, lcms_intent: Intent| -> Option<u32> {
        let mox_out = transform_ramp(&mox_src, &mox_dst, mox_intent, &ramp)?;
        // Use NO_OPTIMIZE | HIGHRES_PRECALC to maximize lcms2 precision and
        // disable any internal pipeline simplification that might mask
        // profile-specific behavior.
        let flags = Flags::NO_OPTIMIZE | Flags::HIGHRES_PRECALC;
        let xform: Transform<[u16; 3], [u16; 3]> = Transform::new_flags(
            &lcms_src,
            LPx::RGB_16,
            &lcms_dst,
            LPx::RGB_16,
            lcms_intent,
            flags,
        )
        .ok()?;
        let mut lcms_out = vec![0u16; ramp.len()];
        // lcms2's Transform is typed by *pixel*, so reinterpret the channel
        // slices as &[[u16; 3]] / &mut [[u16; 3]].
        let src_px: &[[u16; 3]] = bytemuck::cast_slice(&ramp);
        let dst_px: &mut [[u16; 3]] = bytemuck::cast_slice_mut(&mut lcms_out);
        xform.transform_pixels(src_px, dst_px);
        Some(max_pair_err(&mox_out, &lcms_out))
    };
    Some((
        measure_one(
            RenderingIntent::RelativeColorimetric,
            Intent::RelativeColorimetric,
        ),
        measure_one(RenderingIntent::Perceptual, Intent::Perceptual),
        measure_one(RenderingIntent::Saturation, Intent::Saturation),
    ))
}

#[cfg(not(feature = "lcms2-crosscheck"))]
fn lcms2_crosscheck_deltas(_icc_data: &[u8]) -> Option<CrossCheckDeltas> {
    None
}

// ── .inc formatting ──────────────────────────────────────────────────────

fn fmt_hash(h: u64) -> String {
    let s = format!("{h:016x}");
    format!("0x{}_{}_{}_{}", &s[0..4], &s[4..8], &s[8..12], &s[12..16])
}

/// Render the intent mask as a named `Safe::…` constant (see `icc/mod.rs`).
fn fmt_mask(m: u8) -> &'static str {
    match m {
        0b000 => "Safe::IdOnly",
        0b001 => "Safe::Colorimetric",
        0b010 => "Safe::Perceptual",
        0b011 => "Safe::ColorimetricPerceptual",
        0b100 => "Safe::Saturation",
        0b101 => "Safe::ColorimetricSaturation",
        0b110 => "Safe::PerceptualSaturation",
        0b111 => "Safe::AnyIntent",
        _ => unreachable!("intent mask is 3 bits"),
    }
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

struct Args {
    inputs: Vec<PathBuf>,
    out: PathBuf,
    /// When set, print structural-vs-empirical disagreements per profile.
    report: bool,
}

fn parse_args() -> Args {
    let mut positional: Vec<String> = Vec::new();
    let mut report = false;
    for arg in std::env::args().skip(1) {
        if arg == "--report" {
            report = true;
        } else {
            positional.push(arg);
        }
    }
    let (inputs, out) = match positional.len() {
        0 => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            let cache = PathBuf::from(&home).join(".cache/zenpixels-icc");
            let bundled = PathBuf::from("zenpixels-convert/src/profiles");
            let out = PathBuf::from("zenpixels/src/icc");
            (vec![cache, bundled], out)
        }
        1 => {
            let out = PathBuf::from("zenpixels/src/icc");
            (vec![PathBuf::from(&positional[0])], out)
        }
        _ => {
            let out = PathBuf::from(positional.last().unwrap());
            let inputs: Vec<PathBuf> = positional[..positional.len() - 1]
                .iter()
                .map(PathBuf::from)
                .collect();
            (inputs, out)
        }
    };
    Args {
        inputs,
        out,
        report,
    }
}

// ── Active enum variants ─────────────────────────────────────────────────
//
// Only (primaries, transfer) pairs whose enum variants exist in
// descriptor.rs are emitted as active table entries. Profiles using
// deferred variants are emitted as comments for reference. Keep these
// lists in sync with the enum definitions.

const ACTIVE_PRIMARIES: &[&str] = &["Bt709", "Bt2020", "DisplayP3", "AdobeRgb"];
const ACTIVE_TRANSFERS: &[&str] = &["Linear", "Srgb", "Bt709", "Pq", "Gamma22"];

/// Why a row is deferred (not an active table entry).
fn defer_reason(cp: &str, tf: &str) -> Option<&'static str> {
    if !ACTIVE_PRIMARIES.contains(&cp) && !ACTIVE_TRANSFERS.contains(&tf) {
        Some("primaries and transfer deferred")
    } else if !ACTIVE_PRIMARIES.contains(&cp) {
        Some("primaries deferred")
    } else if !ACTIVE_TRANSFERS.contains(&tf) {
        Some("transfer deferred")
    } else {
        None
    }
}

const RGB_HEADER: &str = "\
// Columns: (normalized_hash, primaries, transfer, max_u16_err, intent_mask)
//   max_u16_err — empirical max channel error (u16 units) vs a reference CMS;
//                 compared against Tolerance at query time.
//   intent_mask — Safe::* alias marking which rendering intents are
//                 matrix+TRC-equivalent; see icc/mod.rs for definitions.
//   Commented-out entries have deferred primaries or transfers — see
//   descriptor.rs for rationale. They are kept for reference.
// Generated by scripts/icc-gen — do not edit by hand.\n";

const GRAY_HEADER: &str = "\
// Columns: (normalized_hash, transfer, max_u16_err, intent_mask)
//   See icc_table_rgb.inc for column semantics.
// Generated by scripts/icc-gen — do not edit by hand.\n";

fn write_rgb(rows: &BTreeMap<u64, RgbRow>, out_path: &Path) {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut s = String::from(RGB_HEADER);
    s += "&[\n";
    for row in rows.values() {
        let mask = fmt_mask(row.intent_mask);
        let pad = " ".repeat(28usize.saturating_sub(mask.len()));
        let line = format!(
            "({}, CP::{}, TF::{}, {:>2}, {}),{}  // [{}] {}",
            fmt_hash(row.hash),
            row.cp,
            row.tf,
            row.max_err,
            mask,
            pad,
            mask_breakdown(row.intent_mask),
            clean_desc(&row.desc),
        );
        if let Some(reason) = defer_reason(row.cp, row.tf) {
            s += &format!("    // {line}  [{reason}]\n");
        } else {
            s += &format!("    {line}\n");
        }
    }
    s += "]\n";
    std::fs::write(out_path, s).unwrap();
}

fn write_gray(rows: &BTreeMap<u64, GrayRow>, out_path: &Path) {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut s = String::from(GRAY_HEADER);
    s += "&[\n";
    for row in rows.values() {
        let mask = fmt_mask(row.intent_mask);
        let pad = " ".repeat(28usize.saturating_sub(mask.len()));
        let line = format!(
            "({}, TF::{}, {:>2}, {}),{}  // [{}] {}",
            fmt_hash(row.hash),
            row.tf,
            row.max_err,
            mask,
            pad,
            mask_breakdown(row.intent_mask),
            clean_desc(&row.desc),
        );
        if !ACTIVE_TRANSFERS.contains(&row.tf) {
            s += &format!("    // {line}  [transfer deferred]\n");
        } else {
            s += &format!("    {line}\n");
        }
    }
    s += "]\n";
    std::fs::write(out_path, s).unwrap();
}

fn main() {
    #[cfg(debug_assertions)]
    assert_exclusion_names_valid();

    let args = parse_args();
    let input_dirs = args.inputs;
    let out_dir = args.out;
    let report_mode = args.report;

    for dir in &input_dirs {
        if !dir.exists() {
            eprintln!("Directory not found: {}", dir.display());
            eprintln!("Usage: icc-gen [--report] <dir1> [dir2 ...] <out-dir>");
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

    // Report-mode bookkeeping: one entry per unique normalized hash, capturing
    // every disagreement between structural and empirical signals.
    let mut report_rows: Vec<ReportRow> = Vec::new();

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

            // Compatibility-normalization exclusions — see constants above
            // for rationale. Skips categories of profile where matrix+TRC
            // substitution would diverge from full-CMS rendering; those
            // profiles still work, they just take the full-CMS path.
            if is_excluded_primary(cp_name)
                || (is_para3_excluded_primary(cp_name) && is_para_functype3(&data, b"rTRC"))
            {
                skipped += 1;
                continue;
            }

            // Empirical intent-safety check only once per unique hash — this
            // is where the cost is (moxcms builds + transforms two profiles
            // over a 320-pixel ramp for each of 3 intents, ~1-5ms).
            let (mask, provenance) = compute_rgb_intent_mask(&data, cp_name, tf_name);

            if report_mode {
                report_rows.push(build_report_row(
                    &fname, cp_name, tf_name, &data, mask, provenance,
                ));
            }

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

    if report_mode {
        print_report(&report_rows);
    }
}
