// The POW24_* coefficients are kept at their full numpy-polyfit precision so
// the constant block matches the upstream zentone copy byte-for-byte; the
// trailing digits are below the f32 ULP for these values.
#![allow(clippy::excessive_precision)]

//! BT.2446 Method A tone mapper — the ITU's reference for graded
//! HDR↔SDR conversion with psychophysically verified round-trip fidelity.
//!
//! ITU-R BT.2446-1 §4. TMO: 1000 cd/m² HDR → 100 cd/m² SDR.
//!
//! Psychophysical evaluation (Annex 1): 12 participants, 115 images,
//! 2AFC on Sony BVM-X300 — imperceptible degradation after full
//! round-trip (p = 0.167 HDR, p = 0.196 SDR).
//!
//! The method operates in a perceptually linearized log-luminance domain
//! with a piecewise polynomial knee, Hunt-effect color correction in
//! Y'Cb'Cr', and BT.2020 luma weights.
//!
//! Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1.
//!
//! # Provenance
//!
//! Extracted from `zentone::Bt2446A` (zentone commit `2a7c86b1`). The
//! algorithm is byte-identical; the only difference is that this
//! crate's copy is the canonical home, and the `ToneMap` trait integration
//! is dropped (the trait lives in `zentone` and is not used here — the
//! `HdrToSdr` wrapper drives the curve directly through its inherent
//! methods).
//!
//! The kernel is a magetypes-defined `f32x16` strip that resolves to
//! AVX-512 on V4, 2× AVX2 on V3, NEON on aarch64, WASM-SIMD on wasm32, and
//! the polyfill scalar on everything else. The single-pixel
//! [`Bt2446A::map_rgb`] entry point is a 1-element wrapper around the
//! strip kernel — every code path through the curve goes through the same
//! kernel (the SIMD body on full 16-wide chunks, the scalar remainder tail
//! on the trailing 1–15 pixels and on `map_rgb`).

use libm::powf;

// BT.2020 luma weights (BT.2446 uses BT.2020, not BT.709) — shared with
// `measure` via the parent module's `BT2020_L*` constants.
use super::{BT2020_LB as LB, BT2020_LG as LG, BT2020_LR as LR};

/// Degree-7 monomial polynomial for `x^2.4` on `[0, 1]` (BT.1886 EOTF).
/// Max approximation error 5.88e-5 vs `libm::powf` over 20k uniform samples,
/// ~10× inside the SIMD parity tolerance (5e-4). Coefficients from
/// `numpy.polyfit`; order is Horner-form (highest degree first; constant
/// last). Costs 7 FMA vs ~24 ops for `pow_midp_unchecked(2.4)`. Used by the
/// SIMD body only — the scalar remainder tail (which also services
/// [`Bt2446A::map_rgb`]) keeps `libm::powf` for spec-bit-exact output.
const POW24_C7: f32 = 1.979_355_7e-1;
const POW24_C6: f32 = -8.261_85e-1;
const POW24_C5: f32 = 1.470_748_3;
const POW24_C4: f32 = -1.531_952;
const POW24_C3: f32 = 1.361_614_9;
const POW24_C2: f32 = 3.341_598e-1;
const POW24_C1: f32 = -6.362_703_7e-3;
const POW24_C0: f32 = 5.884_862_3e-5;

/// BT.2446 Method A tonemapper.
///
/// Construct with `new()`, then apply via [`map_rgb`](Self::map_rgb) or
/// [`map_strip_simd`](Self::map_strip_simd). Input is linear-light
/// BT.2020 RGB normalized so `1.0 = hdr_peak_nits`. Output is linear-light
/// BT.2020 RGB normalized so `1.0 = sdr_peak_nits` — the BT.2446-1 §4
/// pipeline natively emits gamma-encoded `R'_TMO` `G'_TMO` `B'_TMO`
/// (BT.1886 1/2.4); we apply the BT.1886 EOTF (`^2.4`) at the output to
/// deliver a linear-light contract (matching libplacebo's own `bt2446a`
/// which ends with `bt1886_eotf`).
///
/// # When to pick this
///
/// The most rigorously validated HDR → SDR curve published — the only one
/// with a peer-reviewed psychophysical study showing imperceptible
/// degradation after a full HDR → SDR → HDR round-trip on graded content.
/// Pick when broadcast-grade fidelity matters.
///
/// Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1 (12 participants,
/// 115 images, Sony BVM-X300, 2AFC, p ≈ 0.17 indistinguishability).
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "hdr-experimental")]
/// # {
/// use zenpixels_convert::hdr::Bt2446A;
///
/// let curve = Bt2446A::new(1000.0, 100.0);
/// let sdr = curve.map_rgb([0.6, 0.4, 0.2]);
/// assert!(sdr.iter().all(|&c| (0.0..=1.0).contains(&c)));
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Bt2446A {
    pub(crate) rho_hdr: f32,
    pub(crate) inv_log_rho_hdr: f32,
    pub(crate) rho_sdr: f32,
    pub(crate) inv_rho_sdr_minus_1: f32,
    /// Source peak in cd/m² as constructed. Stored so we can answer
    /// [`source_peak_nits`](Self::source_peak_nits) /
    /// [`peaks`](crate::hdr::ToneMapper::peaks) without inverting the
    /// derived `rho_hdr`.
    pub(crate) source_peak_nits: f32,
    /// Target peak in cd/m² as constructed. See `source_peak_nits`.
    pub(crate) target_peak_nits: f32,
}

impl Bt2446A {
    /// Create a new BT.2446 Method A tonemapper.
    ///
    /// `hdr_peak_nits`: peak luminance of HDR content (typically 1000).
    /// `sdr_peak_nits`: peak luminance of SDR target (typically 100).
    #[must_use]
    pub fn new(hdr_peak_nits: f32, sdr_peak_nits: f32) -> Self {
        // ρ_H = 1 + 32 · (L_HDR / 10 000)^(1/2.4) per ITU-R BT.2446-1 §4.
        // The exponent is the BT.1886 gamma reciprocal, not γ itself — the
        // pre-2025 zentone used `2.4` here, which collapsed ρ_H from 13.4
        // toward 1.13 at 1000 nits and turned the log compression into a
        // near-identity. Fixed against the libplacebo reference; matches
        // the well-known "ρ_H ≈ 13.2 at 1000 nit, 33 at 10 000 nit"
        // quoted in ITU-R BT.2446-1.
        let inv_gamma = 1.0_f32 / 2.4;
        let rho_hdr = 1.0 + 32.0 * powf(hdr_peak_nits / 10000.0, inv_gamma);
        let log_rho_hdr = libm::logf(rho_hdr);
        let rho_sdr = 1.0 + 32.0 * powf(sdr_peak_nits / 10000.0, inv_gamma);
        Self {
            rho_hdr,
            inv_log_rho_hdr: 1.0 / log_rho_hdr,
            rho_sdr,
            inv_rho_sdr_minus_1: 1.0 / (rho_sdr - 1.0),
            source_peak_nits: hdr_peak_nits,
            target_peak_nits: sdr_peak_nits,
        }
    }

    /// HDR source peak luminance in cd/m², as constructed.
    ///
    /// Mirrors the `hdr_peak_nits` passed to [`new`](Self::new); useful
    /// for diagnostics and for the
    /// [`ToneMapper::peaks`](crate::hdr::ToneMapper::peaks) implementation.
    #[must_use]
    pub fn source_peak_nits(&self) -> f32 {
        self.source_peak_nits
    }

    /// SDR target peak luminance in cd/m², as constructed.
    ///
    /// Mirrors the `sdr_peak_nits` passed to [`new`](Self::new).
    #[must_use]
    pub fn target_peak_nits(&self) -> f32 {
        self.target_peak_nits
    }

    /// Map a single HDR pixel (linear-light BT.2020 RGB, source-normalized)
    /// to an SDR pixel (linear-light BT.2020 RGB, target-normalized).
    ///
    /// Implemented as a 1-element strip through [`Self::map_strip_simd`].
    /// A single pixel never reaches the SIMD body (`chunks_exact(16)`
    /// yields zero chunks); it goes through the scalar remainder tail,
    /// which uses `libm::powf` for bit-exact reproducibility against the
    /// ITU-R BT.2446-1 §4 spec.
    ///
    /// See the struct-level docs for the input/output normalization
    /// contract.
    #[must_use]
    pub fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut buf = [rgb];
        self.map_strip_simd(&mut buf);
        buf[0]
    }

    /// Apply the curve to a strip of HDR pixels in place, dispatching to
    /// the widest available SIMD tier.
    pub fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            bt2446a_tier(
                strip,
                self.rho_hdr,
                self.inv_log_rho_hdr,
                self.rho_sdr,
                self.inv_rho_sdr_minus_1,
            ),
            [v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
    }
}

/// 16-wide BT.2446-A SIMD kernel. Polyfills per tier: AVX-512 (V4) → one
/// `f32x16` op per stage; AVX2 (V3) / NEON / WASM128 → 2× native width;
/// scalar → polyfill loop (runtime fallback AND the byte-exact reference
/// for the [`Bt2446A::map_rgb`] single-pixel entry via the remainder tail).
///
/// Key optimizations vs a naive scalar port: (1) f32x16 chunking; (2)
/// `pow/log2/exp2_midp_unchecked` (inputs pre-clamped to `[pos_eps, ∞)`, so
/// the polynomial-only paths are correct); (3) single `pos_eps` clamp
/// absorbing `max(zero)` + `max(pos_eps)`; (4) degree-7 polynomial for the
/// BT.1886 EOTF (`x^2.4`) on `[0, 1]` — 7 FMA vs ~24 ops, accuracy 5.88e-5
/// (see `POW24_*`). The forward gamma encode keeps `pow_midp_unchecked` —
/// its input range `[0, ~hdr_peak]` doesn't polynomial-approximate cleanly.
#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(crate) fn bt2446a_tier(
    token: Token,
    row: &mut [[f32; 3]],
    rho_hdr: f32,
    inv_log_rho_hdr: f32,
    rho_sdr: f32,
    inv_rho_sdr_minus_1: f32,
) {
    let zero = f32x16::zero(token);
    let one = f32x16::splat(token, 1.0);
    let lr = f32x16::splat(token, LR);
    let lg = f32x16::splat(token, LG);
    let lb = f32x16::splat(token, LB);
    let inv_24 = 1.0_f32 / 2.4;
    let rho_hdr_minus_1 = f32x16::splat(token, rho_hdr - 1.0);
    let inv_log_rho_hdr_v = f32x16::splat(token, inv_log_rho_hdr);
    let inv_rho_sdr_m1_v = f32x16::splat(token, inv_rho_sdr_minus_1);
    let log2_rho_sdr = f32x16::splat(token, libm::log2f(rho_sdr));
    let pos_eps = f32x16::splat(token, f32::MIN_POSITIVE);
    let ln2 = f32x16::splat(token, core::f32::consts::LN_2);

    let t1 = f32x16::splat(token, 0.7399);
    let t2 = f32x16::splat(token, 0.9909);
    let a1 = f32x16::splat(token, 1.0770);
    let a2_a = f32x16::splat(token, -1.1510);
    let a2_b = f32x16::splat(token, 2.7811);
    let a2_c = f32x16::splat(token, -0.6302);
    let a3_a = f32x16::splat(token, 0.5);
    let a3_b = f32x16::splat(token, 0.5);
    let one_p_one = f32x16::splat(token, 1.1);
    let inv_1_8814 = f32x16::splat(token, 1.0 / 1.8814);
    let inv_1_4746 = f32x16::splat(token, 1.0 / 1.4746);
    // 0.16455 = 2·Kb·(1-Kb)/Kg, 0.57135 = 2·Kr·(1-Kr)/Kg for BT.2020 (already
    // divided by Kg; do NOT divide again — that was the pre-fix bug).
    let mat_g_b = f32x16::splat(token, 0.16455);
    let mat_g_r = f32x16::splat(token, 0.57135);
    let mat_r_cr = f32x16::splat(token, 1.4746);
    let mat_b_cb = f32x16::splat(token, 1.8814);
    let zero_one = f32x16::splat(token, 0.1);

    // BT.1886 EOTF polynomial coefficients — see `POW24_*` constants above.
    let c7 = f32x16::splat(token, POW24_C7);
    let c6 = f32x16::splat(token, POW24_C6);
    let c5 = f32x16::splat(token, POW24_C5);
    let c4 = f32x16::splat(token, POW24_C4);
    let c3 = f32x16::splat(token, POW24_C3);
    let c2 = f32x16::splat(token, POW24_C2);
    let c1 = f32x16::splat(token, POW24_C1);
    let c0 = f32x16::splat(token, POW24_C0);

    let mut iter = row.chunks_exact_mut(16);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 16];
        let mut ga = [0.0_f32; 16];
        let mut ba = [0.0_f32; 16];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        // Pre-clamp to `pos_eps` so every downstream `*_unchecked` transcendental
        // sees only finite positive inputs. The `valid = y_p > 0` mask at the
        // end still snaps the original-zero pixels back to black, so the tiny
        // `pos_eps^...` placeholders never reach the output buffer.
        let r = f32x16::load(token, &ra).max(pos_eps);
        let g = f32x16::load(token, &ga).max(pos_eps);
        let b = f32x16::load(token, &ba).max(pos_eps);

        let r_p = r.pow_midp_unchecked(inv_24);
        let g_p = g.pow_midp_unchecked(inv_24);
        let b_p = b.pow_midp_unchecked(inv_24);

        let y_p = lr * r_p + lg * g_p + lb * b_p;

        let arg = (one + rho_hdr_minus_1 * y_p).max(pos_eps);
        let ln_arg = arg.log2_midp_unchecked() * ln2;
        let y_p_lin = ln_arg * inv_log_rho_hdr_v;

        let lo_branch = a1 * y_p_lin;
        let mid_branch = a2_a * y_p_lin * y_p_lin + a2_b * y_p_lin + a2_c;
        let hi_branch = a3_a * y_p_lin + a3_b;
        let in_lo = y_p_lin.simd_le(t1);
        let in_hi = y_p_lin.simd_ge(t2);
        let mid_or_hi = f32x16::blend(in_hi, hi_branch, mid_branch);
        let y_c = f32x16::blend(in_lo, lo_branch, mid_or_hi);

        let y_sdr = ((y_c * log2_rho_sdr).exp2_midp_unchecked() - one) * inv_rho_sdr_m1_v;

        let f = y_sdr / (one_p_one * y_p);
        let cb = f * (b_p - y_p) * inv_1_8814;
        let cr = f * (r_p - y_p) * inv_1_4746;
        let cr_pos = cr.max(zero);
        let y_tmo = y_sdr - zero_one * cr_pos;

        // Clamp into `[0, 1]` — the BT.1886 EOTF polynomial below is fit on
        // this domain. The final `valid` mask still zeroes pixels the spec
        // would emit as black.
        let r_prime_out = (y_tmo + mat_r_cr * cr).max(zero).min(one);
        let g_prime_out = (y_tmo - mat_g_b * cb - mat_g_r * cr).max(zero).min(one);
        let b_prime_out = (y_tmo + mat_b_cb * cb).max(zero).min(one);

        // BT.1886 EOTF (^2.4): degree-7 polynomial on `[0, 1]` via Estrin's
        // method (split into pairs of independent FMAs at depth 1, two at
        // depth 2, final merge). Critical path 3 FMA vs Horner's 7 FMA
        // chain; with 3 channels in parallel the pipeline stays fed.
        let r2 = r_prime_out * r_prime_out;
        let r4 = r2 * r2;
        let r_h7 = c7 * r_prime_out + c6;
        let r_h5 = c5 * r_prime_out + c4;
        let r_h3 = c3 * r_prime_out + c2;
        let r_h1 = c1 * r_prime_out + c0;
        let r_hi = r_h7 * r2 + r_h5;
        let r_lo = r_h3 * r2 + r_h1;
        let r_out = r_hi * r4 + r_lo;

        let g2 = g_prime_out * g_prime_out;
        let g4 = g2 * g2;
        let g_h7 = c7 * g_prime_out + c6;
        let g_h5 = c5 * g_prime_out + c4;
        let g_h3 = c3 * g_prime_out + c2;
        let g_h1 = c1 * g_prime_out + c0;
        let g_hi = g_h7 * g2 + g_h5;
        let g_lo = g_h3 * g2 + g_h1;
        let g_out = g_hi * g4 + g_lo;

        let b2 = b_prime_out * b_prime_out;
        let b4 = b2 * b2;
        let b_h7 = c7 * b_prime_out + c6;
        let b_h5 = c5 * b_prime_out + c4;
        let b_h3 = c3 * b_prime_out + c2;
        let b_h1 = c1 * b_prime_out + c0;
        let b_hi = b_h7 * b2 + b_h5;
        let b_lo = b_h3 * b2 + b_h1;
        let b_out = b_hi * b4 + b_lo;

        // Pre-clamp turns originally-zero pixels into `pos_eps`; the
        // `valid` mask snaps them back to true black on store.
        let valid = y_p.simd_gt(pos_eps);
        let or_arr = f32x16::blend(valid, r_out, zero).to_array();
        let og_arr = f32x16::blend(valid, g_out, zero).to_array();
        let ob_arr = f32x16::blend(valid, b_out, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }

    // Scalar remainder tail — also services the 1-element `map_rgb` entry
    // point. Uses `libm::powf` for both the input gamma encode (`^(1/2.4)`)
    // and the BT.1886 EOTF (`^2.4`) for bit-exact reproducibility against the
    // ITU-R BT.2446-1 §4 spec, where the SIMD body's polynomial
    // approximations diverge by up to 5.88e-5 (still inside the 5e-4 parity
    // bound).
    for px in iter.into_remainder().iter_mut() {
        let r_p = powf(px[0].max(0.0), 1.0 / 2.4);
        let g_p = powf(px[1].max(0.0), 1.0 / 2.4);
        let b_p = powf(px[2].max(0.0), 1.0 / 2.4);
        let y_p = LR * r_p + LG * g_p + LB * b_p;
        if y_p <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let y_p_lin = libm::logf(1.0 + (rho_hdr - 1.0) * y_p) * inv_log_rho_hdr;
        let y_c = if y_p_lin <= 0.7399 {
            1.0770 * y_p_lin
        } else if y_p_lin < 0.9909 {
            -1.1510 * y_p_lin * y_p_lin + 2.7811 * y_p_lin - 0.6302
        } else {
            0.5000 * y_p_lin + 0.5000
        };
        let y_sdr = (powf(rho_sdr, y_c) - 1.0) * inv_rho_sdr_minus_1;
        let f = y_sdr / (1.1 * y_p);
        let cb = f * (b_p - y_p) / 1.8814;
        let cr = f * (r_p - y_p) / 1.4746;
        let y_tmo = y_sdr - 0.1_f32.max(0.0) * cr.max(0.0);
        let r_prime_out = (y_tmo + 1.4746 * cr).clamp(0.0, 1.0);
        // 0.16455 / 0.57135 are already the BT.2020 G' coefficients (already
        // divided by Kg); see the `POW24_*` comment + struct docs above for
        // the derivation.
        let g_prime_out = (y_tmo - 0.16455 * cb - 0.57135 * cr).clamp(0.0, 1.0);
        let b_prime_out = (y_tmo + 1.8814 * cb).clamp(0.0, 1.0);
        // BT.1886 EOTF (^2.4) — see the struct docs above for the
        // linear-light contract.
        *px = [
            powf(r_prime_out, 2.4),
            powf(g_prime_out, 2.4),
            powf(b_prime_out, 2.4),
        ];
    }
}

// `Bt2446A` is the in-crate reference [`ToneMapper`]. The pipeline
// constructs it as the default when an HDR plan is built without an
// explicit mapper. `name()` is part of the implementation's public
// contract — see the trait docs.
impl crate::hdr::ToneMapper for Bt2446A {
    fn map_strip(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len(), "ToneMapper::map_strip: slice length mismatch");
        debug_assert!(input.len() % 3 == 0, "ToneMapper::map_strip: input not RGB-triple-aligned");
        // The SIMD body operates in-place on `[[f32; 3]]`, so copy
        // input → output first, then dispatch with a recast of the
        // destination. The copy is a single memcpy in the contiguous
        // (non-aliased) case and a no-op write when the caller passed
        // overlapping slices.
        if !core::ptr::eq(input.as_ptr(), output.as_ptr()) {
            output.copy_from_slice(input);
        }
        // Cast the flat RGB strip to `[[f32; 3]]` for the SIMD entry.
        // Safe (and panic-free) because the precondition guarantees
        // `len % 3 == 0`; release builds without the assertion will
        // truncate any rogue trailing scalars rather than panic.
        let triples = output.len() / 3;
        let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut output[..triples * 3]);
        self.map_strip_simd(strip);
    }

    fn working_primaries(&self) -> crate::ColorPrimaries {
        // BT.2446 Method A is defined in BT.2020 RGB — see the
        // struct-level docs and the planner's BT.2020-bracketing of
        // this step in `ConvertPlan::new_with_hdr_config`.
        crate::ColorPrimaries::Bt2020
    }

    fn peaks(&self) -> Option<(f32, f32)> {
        Some((self.source_peak_nits, self.target_peak_nits))
    }

    fn name(&self) -> &'static str {
        "bt2446a"
    }

    fn cost_ns_per_mp(&self) -> u32 {
        // BT.2446-A: ~250 Mpix/s on RGB f32 linear-light (Ryzen 9 7950X,
        // AVX2, 2026-06-20). 1 MP / 250 Mpix/s ≈ 4.194 ms/MP. Kept in
        // lockstep with the per-step cost in `crate::estimate` —
        // `ConvertStep::ToneMap` delegates to this method now, so the
        // two figures cannot drift.
        4_194_304
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_hdr_matches_itu_reference_values() {
        // ITU-R BT.2446-1 §4 cites ρ_H ≈ 13.2 at 1000 nits and 33 at
        // 10 000 nits. The pre-fix code used `(L/10000)^2.4` which gave
        // ρ_H ≈ 1.127 — essentially identity — breaking the entire
        // compression curve. Pin ρ to the spec values so the bug can't
        // return.
        let tm1k = Bt2446A::new(1000.0, 100.0);
        assert!(
            (tm1k.rho_hdr - 13.260).abs() < 0.02,
            "ρ_H at 1000 nits ≈ 13.26, got {}",
            tm1k.rho_hdr
        );
        let tm10k = Bt2446A::new(10_000.0, 100.0);
        assert!(
            (tm10k.rho_hdr - 33.0).abs() < 0.05,
            "ρ_H at 10 000 nits = 33.0, got {}",
            tm10k.rho_hdr
        );
        // Bound far enough below the correct value that any regression to
        // the pre-fix `^2.4` formula is immediately visible.
        assert!(
            tm1k.rho_hdr > 5.0,
            "ρ_H regressed toward the pre-fix value (~1.13); got {}",
            tm1k.rho_hdr
        );
    }

    #[test]
    fn libplacebo_parity_end_to_end_on_gray_ramp() {
        // End-to-end parity check against the published libplacebo
        // BT.2446-A formula (haasn/libplacebo, src/tone_mapping.c). Drives
        // the curve through `map_rgb` on a gray ramp (R=G=B=x): the Hunt
        // correction term (Cb, Cr) is zero for neutral input, so the
        // channel value reduces to the per-channel EETF + BT.1886 EOTF
        // chain. Must be bit-close (<1e-4) to libplacebo's numerics.
        fn libplacebo_eetf_then_eotf(x: f32, hdr_peak: f32, sdr_peak: f32) -> f32 {
            let x = x.clamp(0.0, 1.0);
            let p_hdr = 1.0 + 32.0 * powf(hdr_peak / 10000.0, 1.0 / 2.4);
            let p_sdr = 1.0 + 32.0 * powf(sdr_peak / 10000.0, 1.0 / 2.4);
            let x_p = powf(x, 1.0 / 2.4);
            let mut y = libm::logf(1.0 + (p_hdr - 1.0) * x_p) / libm::logf(p_hdr);
            y = if y <= 0.7399 {
                1.0770 * y
            } else if y < 0.9909 {
                -1.1510 * y * y + 2.7811 * y - 0.6302
            } else {
                0.5 * y + 0.5
            };
            let y_sdr_prime = (powf(p_sdr, y) - 1.0) / (p_sdr - 1.0);
            powf(y_sdr_prime, 2.4)
        }

        for &(hdr, sdr) in &[(1000.0_f32, 100.0_f32), (4000.0, 100.0), (10_000.0, 100.0)] {
            let tm = Bt2446A::new(hdr, sdr);
            for &x in &[0.05_f32, 0.18, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0] {
                let got = tm.map_rgb([x, x, x])[0];
                let want = libplacebo_eetf_then_eotf(x, hdr, sdr);
                assert!(
                    (got - want).abs() < 1e-4,
                    "libplacebo parity at (hdr={hdr}, sdr={sdr}, x={x}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn output_is_linear_light_not_gamma_encoded() {
        // The contract is "linear-light HDR in, linear-light SDR out". The
        // BT.2446-1 spec's pipeline emits gamma-encoded SDR; we MUST apply
        // the BT.1886 EOTF (`^2.4`) at the output (matching libplacebo's
        // `bt1886_eotf` close). Pre-fix returned gamma-encoded values,
        // which the consumer then double-gamma-encoded — every pixel came
        // out far too bright (median ΔE2000 ≈ 23 on the shootout corpus).
        let tm = Bt2446A::new(1000.0, 100.0);

        let black = tm.map_rgb([0.0, 0.0, 0.0]);
        assert_eq!(black, [0.0, 0.0, 0.0]);

        let peak = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in peak {
            assert!(
                (c - 1.0).abs() < 1e-4,
                "HDR peak should round-trip to SDR peak: {c}"
            );
        }

        // Mid-grey trace: 0.18 HDR → 0.660 gamma → 0.370 linear via ^2.4.
        // Pre-fix returned ≈0.660 (gamma-encoded); post-fix returns ≈0.370.
        let mid = tm.map_rgb([0.18, 0.18, 0.18]);
        for c in mid {
            assert!(
                (c - 0.370).abs() < 0.02,
                "mid-grey HDR 0.18 should map to linear-light SDR ≈ 0.37, got {c}"
            );
        }
        assert!(
            mid[0] < 0.55,
            "mid-grey output regressed toward the pre-fix gamma-encoded value (~0.66); got {}",
            mid[0]
        );
    }
}
