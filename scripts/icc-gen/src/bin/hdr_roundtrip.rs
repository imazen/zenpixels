//! HDR ICC round-trip validation: do moxcms-synthesized PQ / HLG / P3-PQ ICC
//! profiles reconstruct their EOTF faithfully when read back by a real CMS?
//!
//! Background
//! ----------
//! `zenpixels_convert::icc_profiles::synthesize_icc_for_cicp` lowers an HDR CICP
//! to an ICC profile via moxcms 0.8.1. moxcms cannot express PQ/HLG as an ICC
//! *parametric* curve, so it bakes the EOTF into a **1024-entry `curv` LUT**
//! (`build_trc_table(1024, pq_to_linear)`), output u16-quantized as
//! `(y*65535 + 0.5).floor()`. The sample grid is **uniform in the encoded
//! (PQ/HLG code) domain**, x = i/1023.
//!
//! Two error sources are baked in and this tool measures them end-to-end through
//! the CMS a real downstream consumer would use:
//!   1. **Input under-sampling** — 1024 uniform code-domain points + linear
//!      interpolation between them, against an EOTF whose curvature is extreme
//!      (PQ shoulder especially).
//!   2. **Output 16-bit quantization** — one u16 step is 1.5e-5 absolute linear,
//!      which is a *large relative* error for the PQ toe where decoded linear
//!      values fall below ~1e-3 (the "toe crush" concern).
//!
//! Method
//! ------
//! For each HDR CICP we:
//!   - synthesize the ICC bytes (must be `SynthesizedIcc::Profile`),
//!   - open them with **lcms2** (the authoritative downstream CMS) and build a
//!     float transform `ICC -> XYZ PCS` (`RGB_FLT -> XYZ_FLT`),
//!   - feed a **dense, log-spaced grey ramp** (R=G=B) down to 1e-4 in the toe,
//!     plus a uniform mid/shoulder sweep,
//!   - read back the PCS Y (luminance) — for a grey input Y *is* the decoded
//!     linear value — and compare against the analytic EOTF (a 1:1 port of
//!     moxcms `pq_to_linear` / `hlg_to_linear`, the very functions baked into
//!     the LUT, so the comparison isolates LUT-reconstruction error, not a
//!     formula mismatch).
//!
//! We normalize lcms2's Y by Y(x=1.0) to strip any constant gamut/white-point
//! matrix scale (relative-colorimetric grey luminance), isolating EOTF *shape*
//! error — and also report the raw (un-normalized) error so nothing is hidden.
//!
//! As a second, independent data point we round-trip through **moxcms itself**
//! (synth bytes -> `ColorProfile::new_from_slice` -> transform to a linear
//! sRGB-gamma profile) — but lcms2 is the verdict.
//!
//! Run: `cargo run --bin hdr_roundtrip` (needs liblcms2-dev + `cms-moxcms`).

use lcms2::{Flags, Intent, PixelFormat, Profile, Transform};
use zenpixels_convert::Cicp;
use zenpixels_convert::icc_profiles::{SynthesizedIcc, synthesize_icc_for_cicp};

// ---------------------------------------------------------------------------
// Analytic EOTFs — 1:1 ports of moxcms src/gamma.rs (the LUT source of truth).
// These are exactly the functions `build_trc_table` samples at 1024 points, so
// any divergence the CMS shows is LUT reconstruction error, not formula drift.
// ---------------------------------------------------------------------------

/// SMPTE ST 2084 (PQ) inverse EOTF, normalized so 1.0 == 10000 cd/m^2 peak.
/// Mirrors moxcms `pq_to_linear` (f64) exactly.
fn pq_to_linear(gamma: f64) -> f64 {
    if gamma > 0.0 {
        let pow_gamma = gamma.powf(1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.0);
        let den = (18.8515625 - 18.6875 * pow_gamma).max(f64::MIN);
        (num / den).powf(1.0 / 0.1593017578125)
    } else {
        0.0
    }
}

/// HLG inverse — OETF^-1 followed by the reference OOTF (system gamma 1.2),
/// matching moxcms `hlg_to_linear` (f64) exactly (note the `.powf(1.2)`).
fn hlg_to_linear(gamma: f64) -> f64 {
    if gamma < 0.0 {
        return 0.0;
    }
    if gamma <= 0.5 {
        ((gamma * gamma) * (1.0 / 3.0)).powf(1.2)
    } else {
        (((((gamma - 0.55991073) / 0.17883277).exp()) + 0.28466892) / 12.0).powf(1.2)
    }
}

#[derive(Clone, Copy)]
enum Transfer {
    Pq,
    Hlg,
}

impl Transfer {
    fn eval(self, x: f64) -> f64 {
        match self {
            Transfer::Pq => pq_to_linear(x),
            Transfer::Hlg => hlg_to_linear(x),
        }
    }
}

/// Reproduce moxcms' baked `curv` LUT exactly: 1024 u16 samples of the EOTF,
/// uniform in code space, output quantized `(y*65535 + 0.5).floor()`. This is
/// the byte-identical content of the ICC `curv` tag.
fn build_lut(tf: Transfer, n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            let y = tf.eval(x);
            (y * 65535.0 + 0.5).floor().clamp(0.0, 65535.0) as u16
        })
        .collect()
}

/// Reconstruct a value from a `curv` LUT the way an ICC CMS does: piecewise
/// **linear interpolation** between the two nearest u16 samples in code space,
/// output renormalized to [0,1]. This is the analytic model of what lcms2 must
/// do internally — if it predicts lcms2's measured error, the error source is
/// the 1024-entry LUT approximation itself (inherent), not a CMS artifact.
fn lut_reconstruct(lut: &[u16], x: f64) -> f64 {
    let n = lut.len();
    let pos = x.clamp(0.0, 1.0) * (n - 1) as f64;
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    let frac = pos - i0 as f64;
    let a = lut[i0] as f64 / 65535.0;
    let b = lut[i1] as f64 / 65535.0;
    a + (b - a) * frac
}

// ---------------------------------------------------------------------------
// Dense ramp: log-spaced in the toe (1e-4 .. ~0.05) where PQ linear values are
// tiny and quantization bites hardest, then uniform through mid + shoulder, and
// a guaranteed exact full-white anchor (x = 1.0) for normalization.
// ---------------------------------------------------------------------------
fn dense_ramp() -> Vec<f32> {
    let mut xs = Vec::new();
    // Log-spaced toe: 1e-4 .. 0.06, 400 points.
    let (lo, hi) = (1e-4_f64, 0.06_f64);
    let n_log = 400usize;
    for i in 0..n_log {
        let t = i as f64 / (n_log - 1) as f64;
        let x = lo * (hi / lo).powf(t);
        xs.push(x as f32);
    }
    // Uniform mid + shoulder: 0.06 .. 1.0, 600 points.
    let n_lin = 600usize;
    for i in 1..=n_lin {
        let x = 0.06 + (1.0 - 0.06) * (i as f64 / n_lin as f64);
        xs.push(x as f32);
    }
    // Exact endpoints.
    xs.push(0.0);
    xs.push(1.0);
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup();
    xs
}

/// Transform a grey ramp through `ICC -> XYZ PCS` with lcms2 (float in/out) and
/// return the decoded PCS Y (luminance) per input. For an R=G=B input Y is the
/// decoded linear value.
fn lcms2_decode_y(icc: &[u8], ramp: &[f32]) -> Result<Vec<f64>, String> {
    let src = Profile::new_icc(icc).map_err(|e| format!("lcms2 parse: {e:?}"))?;
    let pcs = Profile::new_xyz(); // XYZ PCS identity (D50), float-friendly.

    // NO_OPTIMIZE + HIGHRES_PRECALC: stop lcms2 collapsing the curv LUT into a
    // coarser device-link approximation; we want the profile's own LUT path,
    // evaluated at the highest precision lcms2 offers, so we measure the
    // *profile's* reconstruction, not an lcms2 optimization artifact.
    let flags = Flags::NO_OPTIMIZE | Flags::HIGHRES_PRECALC;
    let xform: Transform<[f32; 3], [f32; 3]> = Transform::new_flags(
        &src,
        PixelFormat::RGB_FLT,
        &pcs,
        PixelFormat::XYZ_FLT,
        Intent::RelativeColorimetric,
        flags,
    )
    .map_err(|e| format!("lcms2 transform build: {e:?}"))?;

    let input: Vec<[f32; 3]> = ramp.iter().map(|&v| [v, v, v]).collect();
    let mut output = vec![[0.0f32; 3]; ramp.len()];
    xform.transform_pixels(&input, &mut output);
    // XYZ_FLT in lcms2: encoded so that PCS white Y = 1.0 maps to f32 1.0.
    Ok(output.iter().map(|xyz| xyz[1] as f64).collect())
}

/// Independent second opinion: round-trip the *same* synthesized bytes through
/// moxcms's own CMS. We transform `ICC -> a linear (gamma 1.0) sRGB-primaries
/// profile`; for grey, the output channel value is the decoded linear (subject
/// to a gamut matrix, removed by the same full-white normalization).
fn moxcms_decode_lin(icc: &[u8], ramp: &[f32]) -> Result<Vec<f64>, String> {
    // `TransformExecutor` brings the `.transform` trait method into scope; the
    // concrete executor also exposes it inherently, so the import may read as
    // unused depending on resolution — keep it for clarity of the call site.
    #[allow(unused_imports)]
    use moxcms::TransformExecutor;
    use moxcms::{ColorProfile, Layout, RenderingIntent, TransformOptions, curve_from_gamma};

    let src = ColorProfile::new_from_slice(icc).map_err(|e| format!("moxcms parse: {e:?}"))?;
    // Linear-light sRGB-primaries destination (gamma 1.0): exposes decoded
    // linear directly on the grey diagonal. Overwrite the sRGB TRCs with a
    // gamma-1.0 (linear) curve on all three channels.
    let mut dst = ColorProfile::new_srgb();
    let linear = curve_from_gamma(1.0);
    dst.red_trc = Some(linear.clone());
    dst.green_trc = Some(linear.clone());
    dst.blue_trc = Some(linear);

    let opts = TransformOptions {
        rendering_intent: RenderingIntent::RelativeColorimetric,
        ..Default::default()
    };
    let xform = src
        .create_transform_f32(Layout::Rgb, &dst, Layout::Rgb, opts)
        .map_err(|e| format!("moxcms transform build: {e:?}"))?;

    let input: Vec<f32> = ramp.iter().flat_map(|&v| [v, v, v]).collect();
    let mut output = vec![0.0f32; input.len()];
    xform
        .transform(&input, &mut output)
        .map_err(|e| format!("moxcms transform run: {e:?}"))?;
    // Take the G channel (luminance-weighted middle) as the decoded linear.
    Ok(output.chunks_exact(3).map(|p| p[1] as f64).collect())
}

struct ErrStats {
    max_abs: f64,
    max_abs_x: f64,
    max_rel: f64,
    max_rel_x: f64,
    max_rel_ref: f64, // reference linear value at the worst relative error
    /// worst relative error restricted to refs >= 1e-3 (perceptually relevant
    /// magnitudes — below this is sub-noise even at 12-bit HDR signal depth)
    max_rel_above_1e3: f64,
    max_rel_above_1e3_x: f64,
}

/// Compare a decoded curve to the analytic EOTF after normalizing the decoded
/// curve so decoded(white) == eotf(white). `scale` is decoded value at x≈1.0.
fn compare(ramp: &[f32], decoded: &[f64], tf: Transfer, scale: f64) -> ErrStats {
    let mut s = ErrStats {
        max_abs: 0.0,
        max_abs_x: 0.0,
        max_rel: 0.0,
        max_rel_x: 0.0,
        max_rel_ref: 0.0,
        max_rel_above_1e3: 0.0,
        max_rel_above_1e3_x: 0.0,
    };
    for (&x, &d_raw) in ramp.iter().zip(decoded.iter()) {
        let x = x as f64;
        let reference = tf.eval(x);
        let decoded = if scale > 0.0 { d_raw / scale } else { d_raw };
        let abs = (decoded - reference).abs();
        if abs > s.max_abs {
            s.max_abs = abs;
            s.max_abs_x = x;
        }
        // Relative error against the reference; guard tiny denominators.
        if reference > 1e-9 {
            let rel = abs / reference;
            if rel > s.max_rel {
                s.max_rel = rel;
                s.max_rel_x = x;
                s.max_rel_ref = reference;
            }
            if reference >= 1e-3 && rel > s.max_rel_above_1e3 {
                s.max_rel_above_1e3 = rel;
                s.max_rel_above_1e3_x = x;
            }
        }
    }
    s
}

/// Find the decoded value at x closest to 1.0 (the normalization anchor).
fn value_at_white(ramp: &[f32], decoded: &[f64]) -> f64 {
    let mut best = (f64::INFINITY, 0.0);
    for (&x, &d) in ramp.iter().zip(decoded.iter()) {
        let dist = (1.0 - x as f64).abs();
        if dist < best.0 {
            best = (dist, d);
        }
    }
    best.1
}

fn region(x: f64) -> &'static str {
    if x < 0.05 {
        "TOE"
    } else if x < 0.6 {
        "MID"
    } else {
        "SHOULDER"
    }
}

fn run_case(name: &str, cicp: Cicp, tf: Transfer) {
    println!("\n================ {name} ================");
    println!("  CICP: primaries={} transfer={} matrix={} full_range={}",
        cicp.color_primaries, cicp.transfer_characteristics,
        cicp.matrix_coefficients, cicp.full_range);

    let icc = match synthesize_icc_for_cicp(cicp) {
        SynthesizedIcc::Profile(bytes) => {
            println!("  synthesize_icc_for_cicp -> Profile ({} bytes)", bytes.len());
            bytes.into_owned()
        }
        other => {
            println!("  !! synthesize_icc_for_cicp -> {other:?} (expected Profile) — SKIP");
            return;
        }
    };

    // Sanity: confirm the profile actually carries a curv LUT (not a parametric
    // curve), which is the whole reason this validation exists.
    report_trc_kind(&icc);

    let ramp = dense_ramp();

    // ---- lcms2 (authoritative) ----
    match lcms2_decode_y(&icc, &ramp) {
        Ok(y) => {
            let white = value_at_white(&ramp, &y);
            println!("  [lcms2] PCS Y at white (x~1.0) = {white:.6}  (normalization scale)");
            let raw = compare(&ramp, &y, tf, 1.0);
            let norm = compare(&ramp, &y, tf, white);
            print_stats("lcms2 (raw, no norm)", &raw);
            print_stats("lcms2 (white-normalized)", &norm);
            // Spot the reconstructed-vs-reference curve at representative code
            // points to show there is no localized cliff — error is smooth.
            print!("  [lcms2 spot rel-err]");
            for &xp in &[1e-4_f64, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                // nearest sampled ramp point
                let (mut bi, mut bd) = (0usize, f64::INFINITY);
                for (k, &rx) in ramp.iter().enumerate() {
                    let d = (rx as f64 - xp).abs();
                    if d < bd {
                        bd = d;
                        bi = k;
                    }
                }
                let refv = tf.eval(ramp[bi] as f64);
                let dec = y[bi] / white;
                let rel = if refv > 1e-12 { (dec - refv).abs() / refv } else { 0.0 };
                print!(" x={:.3}:{:.2}%", ramp[bi], rel * 100.0);
            }
            println!();
        }
        Err(e) => println!("  [lcms2] FAILED: {e}"),
    }

    // ---- moxcms (second opinion) ----
    match moxcms_decode_lin(&icc, &ramp) {
        Ok(lin) => {
            let white = value_at_white(&ramp, &lin);
            println!("  [moxcms] linear at white (x~1.0) = {white:.6}");
            let norm = compare(&ramp, &lin, tf, white);
            print_stats("moxcms (white-normalized)", &norm);
        }
        Err(e) => println!("  [moxcms] note: {e}"),
    }

    // ---- Analytic LUT model: isolate the *source* of the error ----
    // If the 1024-entry piecewise-linear reconstruction predicts lcms2's
    // measured error, the error is inherent to the LUT approximation (so a
    // denser LUT is the lever). Also show 2048/4096 to size that lever.
    for n in [1024usize, 2048, 4096] {
        let lut = build_lut(tf, n);
        let recon: Vec<f64> = ramp.iter().map(|&x| lut_reconstruct(&lut, x as f64)).collect();
        let s = compare(&ramp, &recon, tf, 1.0);
        println!(
            "  [analytic curv-LUT, {n} entries] max_abs={:.3e} @x={:.4} ({}) | max_rel(ref>=1e-3)={:.3e} @x={:.4} ({})",
            s.max_abs, s.max_abs_x, region(s.max_abs_x),
            s.max_rel_above_1e3, s.max_rel_above_1e3_x, region(s.max_rel_above_1e3_x),
        );
    }
}

fn print_stats(label: &str, s: &ErrStats) {
    println!(
        "  [{label}] max_abs={:.3e} @x={:.4} ({}) | max_rel={:.3e} @x={:.4} ({}, ref_lin={:.3e}) | max_rel(ref>=1e-3)={:.3e} @x={:.4} ({})",
        s.max_abs, s.max_abs_x, region(s.max_abs_x),
        s.max_rel, s.max_rel_x, region(s.max_rel_x), s.max_rel_ref,
        s.max_rel_above_1e3, s.max_rel_above_1e3_x, region(s.max_rel_above_1e3_x),
    );
}

/// Inspect the ICC and report whether the red-TRC tag is a curv LUT (and its
/// entry count) or a parametric curve. Best-effort tag walk; purely diagnostic.
fn report_trc_kind(icc: &[u8]) {
    // ICC: 128-byte header, then tag table: u32 count, then count*(sig,off,size).
    if icc.len() < 132 {
        println!("  [trc] profile too small to inspect");
        return;
    }
    let rd_u32 = |o: usize| -> u32 {
        u32::from_be_bytes([icc[o], icc[o + 1], icc[o + 2], icc[o + 3]])
    };
    let count = rd_u32(128) as usize;
    let mut found = None;
    for i in 0..count {
        let base = 132 + i * 12;
        if base + 12 > icc.len() {
            break;
        }
        if &icc[base..base + 4] == b"rTRC" {
            let off = rd_u32(base + 4) as usize;
            let size = rd_u32(base + 8) as usize;
            found = Some((off, size));
            break;
        }
    }
    match found {
        Some((off, size)) if off + 12 <= icc.len() => {
            let typ = &icc[off..off + 4];
            if typ == b"curv" {
                let n = rd_u32(off + 8) as usize;
                println!("  [trc] rTRC = curv LUT, {n} entries, tag {size} bytes  (the baked EOTF)");
            } else if typ == b"para" {
                println!("  [trc] rTRC = parametric (para) — not a LUT");
            } else {
                println!("  [trc] rTRC = type {:?}", core::str::from_utf8(typ).unwrap_or("????"));
            }
        }
        _ => println!("  [trc] rTRC tag not located (diagnostic only)"),
    }
}

fn main() {
    println!("HDR ICC round-trip validation (moxcms synth -> lcms2 read-back)");
    println!("Acceptable bar: white-normalized max relative error in linear < 1% at");
    println!("perceptually-relevant magnitudes (ref_lin >= 1e-3); deep-toe noise is");
    println!("expected from 16-bit LUT output quantization and reported separately.");

    // PQ / BT.2020 (HDR10): primaries=9, transfer=16, matrix=0, narrow=false.
    run_case("PQ  (BT.2020/SMPTE2084, HDR10)", Cicp::new(9, 16, 0, false), Transfer::Pq);
    // HLG / BT.2020: transfer=18.
    run_case("HLG (BT.2020/ARIB-STD-B67)", Cicp::new(9, 18, 0, false), Transfer::Hlg);
    // P3-PQ: primaries=12 (SMPTE RP431-2 / DCI-P3-D65), transfer=16.
    run_case("P3-PQ (DisplayP3/SMPTE2084)", Cicp::new(12, 16, 0, false), Transfer::Pq);
}
