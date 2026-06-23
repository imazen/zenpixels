// The POW24_* coefficients are kept at their full numpy-polyfit precision so
// the constant block matches the upstream zentone copy byte-for-byte; the
// trailing digits are below the f32 ULP for these values.
#![allow(clippy::excessive_precision)]

//! SIMD strip kernel for `Bt2446A` — magetypes-defined `f32x16` kernel
//! that resolves to AVX-512 on V4, 2× AVX2 on V3, NEON on aarch64, WASM-SIMD
//! on wasm32, and scalar fallback otherwise.
//!
//! # Provenance
//!
//! Lifted verbatim from `zentone/src/simd/curves.rs::bt2446a_tier` (zentone
//! commit `2a7c86b1`). Only the module location and import paths changed —
//! the kernel body, constants, and approximation bounds are byte-identical
//! to the upstream version, and the `simd_parity` test moved with it.

use libm::powf;

// BT.2020 luma weights (used by BT.2446 Method A) — shared with the scalar
// `bt2446a` path via the parent module's `BT2020_L*` constants.
use super::{BT2020_LB as LB_2020, BT2020_LG as LG_2020, BT2020_LR as LR_2020};

/// Degree-7 monomial polynomial for `x^2.4` on `[0, 1]` (BT.1886 EOTF).
///
/// Max approximation error: **5.88e-5** vs `libm::powf(x, 2.4)` over 20k
/// uniformly-spaced samples on `[0, 1]`. This is ~10× inside the BT.2446-A
/// SIMD parity tolerance (5e-4) and ~3000× tighter than the perceptual delta
/// the `output_is_linear_light_not_gamma_encoded` regression test fences
/// (0.02). Coefficients are least-squares minima from `numpy.polyfit`.
///
/// Compared to `pow_midp_unchecked(2.4)` which expands to
/// `exp2_midp_unchecked(2.4 * log2_midp_unchecked(x))` — roughly **24 ops**
/// of polynomial math — this degree-7 evaluation costs only **7 FMA** (with
/// one mul/add at the top). The savings are 3× per `^2.4` call, applied to
/// 3 channels per pixel.
///
/// Order is Horner-form (highest degree first; constant last). Used by the
/// EOTF stage of the BT.2446-A SIMD kernel only — the scalar `map_rgb` path
/// keeps `libm::powf` for bit-exact reproducibility against the spec.
const POW24_C7: f32 = 1.979_355_7e-1;
const POW24_C6: f32 = -8.261_85e-1;
const POW24_C5: f32 = 1.470_748_3;
const POW24_C4: f32 = -1.531_952;
const POW24_C3: f32 = 1.361_614_9;
const POW24_C2: f32 = 3.341_598e-1;
const POW24_C1: f32 = -6.362_703_7e-3;
const POW24_C0: f32 = 5.884_862_3e-5;

/// 16-wide BT.2446-A SIMD kernel. Polyfills natively per tier:
/// **AVX-512 (V4)**: one `f32x16` op per stage; **AVX2 (V3) / NEON / WASM128**:
/// 2× their native width per stage.
///
/// Optimizations vs a naive scalar port:
/// 1. **f32x16 chunking** — 16 pixels per loop iteration, halving loop overhead
///    on every backend and lighting up the real AVX-512 path on V4 hardware.
/// 2. **`pow_midp_unchecked` / `log2_midp_unchecked` / `exp2_midp_unchecked`** —
///    skip the zero/negative/inf/NaN special-case blends. All inputs are
///    pre-clamped to `[pos_eps, ∞)`, so the polynomial-only path is correct.
///    Saves ~7 blendvps + 3 cmpps per call.
/// 3. **Single `pos_eps` clamp absorbs the `max(zero)` + `max(pos_eps)` two-step.**
/// 4. **Degree-7 monomial polynomial for the BT.1886 EOTF (`x^2.4`)** on the
///    post-clamp domain `[0, 1]`. Replaces 3 `pow_midp_unchecked(2.4)` calls
///    (~24 ops each: log2 polynomial + multiply + exp2 polynomial) with 3
///    degree-7 Horner evaluations (7 FMA each, fully pipelined). Approximation
///    error 5.88e-5 — 10× inside the SIMD parity tolerance (5e-4). This is
///    the dominant per-pixel saving. See `POW24_*` constants above for the
///    coefficient derivation.
///
///    The inverse direction (`x^(1/2.4)` for the input gamma encode) does NOT
///    polynomial-approximate cleanly because the input range is `[0, ~hdr_peak]`
///    (unclamped HDR, up to ~4× normalized) and `f(x) = x^0.417` has effectively
///    vertical tangent near 0. We keep `pow_midp_unchecked(1/2.4)` for the
///    input transfer.
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
    let lr = f32x16::splat(token, LR_2020);
    let lg = f32x16::splat(token, LG_2020);
    let lb = f32x16::splat(token, LB_2020);
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

        // BT.1886 EOTF (^2.4): degree-7 monomial polynomial on `[0, 1]`.
        // See `POW24_*` constants above for the coefficients + accuracy bound
        // (max approximation error 5.88e-5 vs `libm::powf`, well inside the
        // SIMD parity tolerance of 5e-4).
        //
        // **Estrin's method** (not Horner) — splits the polynomial into pairs:
        //   p(x) = ((c7·x + c6)·x² + (c5·x + c4))·x⁴
        //        + ((c3·x + c2)·x² + (c1·x + c0))
        // Four independent FMAs at depth 1, two at depth 2, then a final merge.
        // Critical-path latency: **3 FMA ≈ 12 cycles** vs Horner's 7 FMA chain
        // (~28 cycles). With 3 channels running in parallel, the pipeline stays
        // fed and the per-pixel EOTF cost drops to dependency-chain throughput
        // limits, not latency limits.
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

    for px in iter.into_remainder().iter_mut() {
        let r_p = powf(px[0].max(0.0), 1.0 / 2.4);
        let g_p = powf(px[1].max(0.0), 1.0 / 2.4);
        let b_p = powf(px[2].max(0.0), 1.0 / 2.4);
        let y_p = LR_2020 * r_p + LG_2020 * g_p + LB_2020 * b_p;
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
        // divided by Kg); see bt2446a.rs for the derivation.
        let g_prime_out = (y_tmo - 0.16455 * cb - 0.57135 * cr).clamp(0.0, 1.0);
        let b_prime_out = (y_tmo + 1.8814 * cb).clamp(0.0, 1.0);
        // BT.1886 EOTF (^2.4) — see bt2446a.rs for the linear-light contract.
        *px = [
            powf(r_prime_out, 2.4),
            powf(g_prime_out, 2.4),
            powf(b_prime_out, 2.4),
        ];
    }
}
