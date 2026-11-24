//! Bit-exact scalar + SIMD transfer function kernels (BT.709, Gamma 2.2, PQ, HLG Scene).
//!
//! Every SIMD kernel evaluates the exact same math as its `ScalarToken`
//! counterpart so `Accelerators::SCALAR` and `Accelerators::ALL` produce
//! **0-bit identical** output on every pixel.
//!
//! - `BT.709` and `Gamma 2.2` share a single const-configured `magemath::Powf`
//!   profile (`TRANSFER_POW`), keeping binary size tiny across all four curves.
//! - `PQ` uses the two-segment rational polynomial coefficients from
//!   `imazen/linear-srgb@ae83f194` (same copyright ownership) with plain
//!   mul+add Horner evaluation (0-bit portable across scalar, AVX2, NEON, WASM)
//!   and per-lane `f64` dark-toe patching.
//! - `HLG Scene` reuses the `magemath::Expf`/`Logf` profiles from `hlg_render`.

#![allow(clippy::excessive_precision)]

use archmage::ScalarToken;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfOptions};

const MATH_PORTABILITY: magemath::Portability = if cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
)) {
    magemath::Portability::BackendDependent
} else {
    magemath::Portability::BitExact
};

/// Single shared power profile for BT.709 (`1/0.45`, `0.45`) and Adobe Gamma 2.2
/// (`563/256`, `256/563`).
///
/// Domain proof: `|log2(1e-12) * 3.0| = 119.59 <= 125` and
/// `|log2(65536.0) * 3.0| = 48.0 <= 125`, so all range checks are statically
/// elided while covering every non-zero U8, U16, F16, and visible F32 pixel.
const POW_LO: f32 = 1.0e-12;
const POW_HI: f32 = 65_536.0;

const TRANSFER_POW: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveGamma)
    .base_range(POW_LO, POW_HI)
    .exponent_range(0.25, 3.0)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

const BT709_ALPHA: f32 = 0.099_296_826_809_44;
const BT709_ONE_PLUS_ALPHA: f32 = 1.099_296_826_809_44;
const BT709_INV_ONE_PLUS_ALPHA: f32 = 1.0 / 1.099_296_826_809_44;
const BT709_BETA: f32 = 0.018_053_968_510_807;
const BT709_BETA_LIN: f32 = 4.5 * BT709_BETA;
const BT709_INV_4_5: f32 = 1.0 / 4.5;
const BT709_DEC_EXP: f32 = 1.0 / 0.45;
const BT709_ENC_EXP: f32 = 0.45;

const GAMMA22_DEC_EXP: f32 = 563.0 / 256.0;
const GAMMA22_ENC_EXP: f32 = 256.0 / 563.0;

// ---------------------------------------------------------------------------
// Scalar exact per-lane fallbacks (only invoked on non-finite / |v| > 65536 /
// sub-visible 0 < |v| < 1e-12 / PQ dark toe).
// ---------------------------------------------------------------------------

#[cold]
#[inline(never)]
fn bt709_eotf_fallback(v: f32) -> f32 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA_LIN: f64 = 0.081_242_858_298_631_5;
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a < BETA_LIN {
        a / 4.5
    } else {
        libm::pow((a + (ALPHA - 1.0)) / ALPHA, 1.0 / 0.45)
    };
    (out as f32).copysign(v)
}

#[cold]
#[inline(never)]
fn bt709_oetf_fallback(v: f32) -> f32 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA: f64 = 0.018_053_968_510_807;
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a < BETA {
        4.5 * a
    } else {
        ALPHA * libm::pow(a, 0.45) - (ALPHA - 1.0)
    };
    (out as f32).copysign(v)
}

#[cold]
#[inline(never)]
fn gamma22_fallback(v: f32, exp: f64) -> f32 {
    if !v.is_finite() {
        return v;
    }
    (libm::pow(f64::from(v.abs()), exp) as f32).copysign(v)
}

/// Arbitrary validated decoding gamma EOTF (`L = sign(v) * |v|^gamma`) on an `f32` slice.
pub(crate) fn gamma_eotf_scalar_slice(buf: &mut [f32], gamma: f32) {
    let exp = f64::from(gamma);
    for v in buf.iter_mut() {
        let x = *v;
        if x == 0.0 || !x.is_finite() {
            continue;
        }
        *v = (libm::pow(f64::from(x.abs()), exp) as f32).copysign(x);
    }
}

/// Arbitrary validated encoding gamma OETF (`v = sign(L) * |L|^(1/gamma)`) on an `f32` slice.
pub(crate) fn gamma_oetf_scalar_slice(buf: &mut [f32], gamma: f32) {
    let inv = 1.0 / f64::from(gamma);
    for v in buf.iter_mut() {
        let x = *v;
        if x == 0.0 || !x.is_finite() {
            continue;
        }
        *v = (libm::pow(f64::from(x.abs()), inv) as f32).copysign(x);
    }
}

// ---------------------------------------------------------------------------
// BT.709 EOTF / OETF (odd-symmetric for extended negative values)
// ---------------------------------------------------------------------------

#[inline(always)]
fn bt709_eotf_one(v: f32) -> f32 {
    let a = v.abs();
    if a < BT709_BETA_LIN {
        return (a * BT709_INV_4_5).copysign(v);
    }
    let base = (a + BT709_ALPHA) * BT709_INV_ONE_PLUS_ALPHA;
    if base <= POW_HI {
        let p = Powf::<TRANSFER_POW>::scalar_with_token(ScalarToken, base, BT709_DEC_EXP)
            .unwrap_or_else(|_| bt709_eotf_fallback(a));
        p.copysign(v)
    } else {
        bt709_eotf_fallback(v)
    }
}

#[inline(always)]
fn bt709_oetf_one(v: f32) -> f32 {
    let a = v.abs();
    if a < BT709_BETA {
        return (a * 4.5).copysign(v);
    }
    if a <= POW_HI {
        let p = Powf::<TRANSFER_POW>::scalar_with_token(ScalarToken, a, BT709_ENC_EXP)
            .unwrap_or_else(|_| bt709_oetf_fallback(a));
        (BT709_ONE_PLUS_ALPHA * p - BT709_ALPHA).copysign(v)
    } else {
        bt709_oetf_fallback(v)
    }
}

#[inline(never)]
pub(crate) fn bt709_eotf_scalar_slice(vals: &mut [f32]) {
    for v in vals {
        *v = bt709_eotf_one(*v);
    }
}

#[inline(never)]
pub(crate) fn bt709_oetf_scalar_slice(vals: &mut [f32]) {
    for v in vals {
        *v = bt709_oetf_one(*v);
    }
}

macro_rules! bt709_eotf_simd_body {
    ($vec:ident, $lanes:literal, $token:ident, $vals:ident) => {{
        let beta_lin = $vec::splat_t($token, BT709_BETA_LIN);
        let inv_45 = $vec::splat_t($token, BT709_INV_4_5);
        let alpha = $vec::splat_t($token, BT709_ALPHA);
        let inv_1pa = $vec::splat_t($token, BT709_INV_ONE_PLUS_ALPHA);
        let one = $vec::splat_t($token, 1.0);
        let hi = $vec::splat_t($token, POW_HI);
        let neg_zero = $vec::splat_t($token, -0.0);

        let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
        for chunk in chunks {
            let v = $vec::load_t($token, chunk);
            let a = v.abs();
            let sign = v & neg_zero;
            let is_lin = a.simd_lt(beta_lin);
            let base = (a + alpha) * inv_1pa;
            let safe_base = $vec::blend(is_lin, one, base);
            if safe_base.simd_le(hi).bitcast_to_i32().all_true() {
                let lin = a * inv_45;
                let pow = safe_base.powf::<TRANSFER_POW>(BT709_DEC_EXP);
                let out = $vec::blend(is_lin, lin, pow) | sign;
                out.store(chunk);
            } else {
                for x in chunk.iter_mut() {
                    *x = bt709_eotf_one(*x);
                }
            }
        }
        for x in rem {
            *x = bt709_eotf_one(*x);
        }
    }};
}

macro_rules! bt709_oetf_simd_body {
    ($vec:ident, $lanes:literal, $token:ident, $vals:ident) => {{
        let beta = $vec::splat_t($token, BT709_BETA);
        let s45 = $vec::splat_t($token, 4.5);
        let alpha = $vec::splat_t($token, BT709_ALPHA);
        let one_pa = $vec::splat_t($token, BT709_ONE_PLUS_ALPHA);
        let one = $vec::splat_t($token, 1.0);
        let hi = $vec::splat_t($token, POW_HI);
        let neg_zero = $vec::splat_t($token, -0.0);

        let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
        for chunk in chunks {
            let v = $vec::load_t($token, chunk);
            let a = v.abs();
            let sign = v & neg_zero;
            let is_lin = a.simd_lt(beta);
            let safe_base = $vec::blend(is_lin, one, a);
            if safe_base.simd_le(hi).bitcast_to_i32().all_true() {
                let lin = a * s45;
                let pow = one_pa * safe_base.powf::<TRANSFER_POW>(BT709_ENC_EXP) - alpha;
                let out = $vec::blend(is_lin, lin, pow) | sign;
                out.store(chunk);
            } else {
                for x in chunk.iter_mut() {
                    *x = bt709_oetf_one(*x);
                }
            }
        }
        for x in rem {
            *x = bt709_oetf_one(*x);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn bt709_eotf_v3_t(token: archmage::X64V3Token, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    bt709_eotf_simd_body!(V, 8, token, vals);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn bt709_eotf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    bt709_eotf_simd_body!(V, 16, token, vals);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn bt709_eotf_neon_t(token: archmage::NeonToken, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    bt709_eotf_simd_body!(V, 4, token, vals);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn bt709_eotf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    bt709_eotf_simd_body!(V, 4, token, vals);
}

fn bt709_eotf_scalar_t(_token: ScalarToken, vals: &mut [f32]) {
    bt709_eotf_scalar_slice(vals);
}

pub(crate) fn bt709_eotf_simd_slice(vals: &mut [f32]) {
    archmage::attuned!(
        bt709_eotf(vals),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn bt709_oetf_v3_t(token: archmage::X64V3Token, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    bt709_oetf_simd_body!(V, 8, token, vals);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn bt709_oetf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    bt709_oetf_simd_body!(V, 16, token, vals);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn bt709_oetf_neon_t(token: archmage::NeonToken, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    bt709_oetf_simd_body!(V, 4, token, vals);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn bt709_oetf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    bt709_oetf_simd_body!(V, 4, token, vals);
}

fn bt709_oetf_scalar_t(_token: ScalarToken, vals: &mut [f32]) {
    bt709_oetf_scalar_slice(vals);
}

pub(crate) fn bt709_oetf_simd_slice(vals: &mut [f32]) {
    archmage::attuned!(
        bt709_oetf(vals),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// Adobe RGB Gamma 2.2 (563/256) EOTF / OETF — shares TRANSFER_POW!
// ---------------------------------------------------------------------------

#[inline(always)]
fn gamma_one(v: f32, exp_f32: f32, exp_f64: f64) -> f32 {
    let a = v.abs();
    if a == 0.0 {
        return 0.0_f32.copysign(v);
    }
    if (POW_LO..=POW_HI).contains(&a) {
        let p = Powf::<TRANSFER_POW>::scalar_with_token(ScalarToken, a, exp_f32)
            .unwrap_or_else(|_| gamma22_fallback(a, exp_f64));
        p.copysign(v)
    } else {
        gamma22_fallback(v, exp_f64)
    }
}

#[inline(never)]
pub(crate) fn gamma22_eotf_scalar_slice(vals: &mut [f32]) {
    for v in vals {
        *v = gamma_one(*v, GAMMA22_DEC_EXP, 563.0 / 256.0);
    }
}

#[inline(never)]
pub(crate) fn gamma22_oetf_scalar_slice(vals: &mut [f32]) {
    for v in vals {
        *v = gamma_one(*v, GAMMA22_ENC_EXP, 256.0 / 563.0);
    }
}

macro_rules! gamma_simd_body {
    ($vec:ident, $lanes:literal, $token:ident, $vals:ident, $exp_f32:ident, $exp_f64:ident) => {{
        let zero = $vec::zero_t($token);
        let one = $vec::splat_t($token, 1.0);
        let lo = $vec::splat_t($token, POW_LO);
        let hi = $vec::splat_t($token, POW_HI);
        let neg_zero = $vec::splat_t($token, -0.0);

        let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
        for chunk in chunks {
            let v = $vec::load_t($token, chunk);
            let a = v.abs();
            let sign = v & neg_zero;
            let is_zero = a.simd_eq(zero);
            let in_range = a.simd_ge(lo) & a.simd_le(hi);
            if (is_zero | in_range).bitcast_to_i32().all_true() {
                let safe = $vec::blend(is_zero, one, a);
                let p = safe.powf::<TRANSFER_POW>($exp_f32);
                let out = $vec::blend(is_zero, zero, p) | sign;
                out.store(chunk);
            } else {
                for x in chunk.iter_mut() {
                    *x = gamma_one(*x, $exp_f32, $exp_f64);
                }
            }
        }
        for x in rem {
            *x = gamma_one(*x, $exp_f32, $exp_f64);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gamma_slice_v3_t(token: archmage::X64V3Token, vals: &mut [f32], exp_f32: f32, exp_f64: f64) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    gamma_simd_body!(V, 8, token, vals, exp_f32, exp_f64);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn gamma_slice_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32], exp_f32: f32, exp_f64: f64) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    gamma_simd_body!(V, 16, token, vals, exp_f32, exp_f64);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn gamma_slice_neon_t(token: archmage::NeonToken, vals: &mut [f32], exp_f32: f32, exp_f64: f64) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    gamma_simd_body!(V, 4, token, vals, exp_f32, exp_f64);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn gamma_slice_wasm128_t(
    token: archmage::Wasm128Token,
    vals: &mut [f32],
    exp_f32: f32,
    exp_f64: f64,
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    gamma_simd_body!(V, 4, token, vals, exp_f32, exp_f64);
}

fn gamma_slice_scalar_t(_token: ScalarToken, vals: &mut [f32], exp_f32: f32, exp_f64: f64) {
    for v in vals {
        *v = gamma_one(*v, exp_f32, exp_f64);
    }
}

pub(crate) fn gamma22_eotf_simd_slice(vals: &mut [f32]) {
    let ef = GAMMA22_DEC_EXP;
    let ed = 563.0 / 256.0;
    archmage::attuned!(
        gamma_slice(vals, ef, ed),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

pub(crate) fn gamma22_oetf_simd_slice(vals: &mut [f32]) {
    let ef = GAMMA22_ENC_EXP;
    let ed = 256.0 / 563.0;
    archmage::attuned!(
        gamma_slice(vals, ef, ed),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// HDR transfers: PQ (SMPTE ST 2084) + HLG Scene (ARIB STD-B67)
// ---------------------------------------------------------------------------

#[cfg(feature = "hdr-transfers")]
pub(crate) mod hdr {
    use super::MATH_PORTABILITY;
    use archmage::ScalarToken;
    use magemath::SimdMath;
    use magemath::expf::{Expf, ExpfDomain, ExpfOptions};
    use magemath::logf::{Logf, LogfDomain, LogfOptions};

    // Coefficients from `imazen/linear-srgb@ae83f194` (same copyright ownership).
    const PQ_EOTF_P_SMALL: [f32; 5] = [
        4.36033203e-10,
        -1.53651077e-07,
        2.06864196e-06,
        2.80304711e-03,
        1.47809948e-02,
    ];
    const PQ_EOTF_Q_SMALL: [f32; 5] = [
        -9.89955022e-03,
        1.89911767e+00,
        -1.34828020e+00,
        -7.10547502e-01,
        1.00000000e+00,
    ];
    const PQ_EOTF_P_LARGE: [f32; 5] = [
        5.43046166e-04,
        -6.31017431e-03,
        2.90480734e-01,
        9.80363032e-01,
        2.22819483e-01,
    ];
    const PQ_EOTF_Q_LARGE: [f32; 5] = [
        1.58688561e+02,
        -1.60764060e+02,
        6.51370372e+01,
        -1.26438391e+01,
        1.00000000e+00,
    ];

    const PQ_INV_P_LARGE: [f32; 5] = [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
    const PQ_INV_Q_LARGE: [f32; 5] = [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];
    const PQ_INV_P_SMALL: [f32; 5] = [
        9.863406e-6,
        3.881234e-1,
        1.352821e2,
        6.889862e4,
        -2.864824e5,
    ];
    const PQ_INV_Q_SMALL: [f32; 5] = [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

    const PQ_EOTF_FLOOR: f32 = 0.02;
    const PQ_OETF_FLOOR: f32 = 1.0e-7;

    /// Exact f64 ST 2084 PQ EOTF (`[0, 1] -> [0, 1]`; NaN / `<= 0` -> `0.0`).
    #[cold]
    #[inline(never)]
    pub(crate) fn pq_eotf_exact(v: f32) -> f32 {
        const M1: f64 = 1305.0 / 8192.0;
        const M2: f64 = 2523.0 / 32.0;
        const C1: f64 = 107.0 / 128.0;
        const C2: f64 = 2413.0 / 128.0;
        const C3: f64 = 2392.0 / 128.0;
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        let e = f64::from(v.min(1.0));
        let ep = libm::pow(e, 1.0 / M2);
        let num = (ep - C1).max(0.0);
        let den = C2 - C3 * ep;
        if den <= 0.0 {
            return 1.0;
        }
        libm::pow(num / den, 1.0 / M1) as f32
    }

    /// Exact f64 ST 2084 PQ OETF (`[0, 1] -> [0, 1]`; NaN / `<= 0` -> `0.0`).
    #[cold]
    #[inline(never)]
    pub(crate) fn pq_oetf_exact(v: f32) -> f32 {
        const M1: f64 = 1305.0 / 8192.0;
        const M2: f64 = 2523.0 / 32.0;
        const C1: f64 = 107.0 / 128.0;
        const C2: f64 = 2413.0 / 128.0;
        const C3: f64 = 2392.0 / 128.0;
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        let y = f64::from(v.min(1.0));
        let ym = libm::pow(y, M1);
        libm::pow((C1 + C2 * ym) / (1.0 + C3 * ym), M2) as f32
    }

    #[inline(always)]
    fn eval_horner5_scalar(x: f32, c: [f32; 5]) -> f32 {
        (((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0]
    }

    /// Scalar PQ EOTF matching the SIMD rational-polynomial path bit-for-bit.
    #[inline(always)]
    pub(crate) fn pq_eotf_one(v: f32) -> f32 {
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        if v < PQ_EOTF_FLOOR {
            return pq_eotf_exact(v);
        }
        let x = v * v + v;
        if v < 0.25 {
            eval_horner5_scalar(x, PQ_EOTF_P_SMALL) / eval_horner5_scalar(x, PQ_EOTF_Q_SMALL)
        } else {
            eval_horner5_scalar(x, PQ_EOTF_P_LARGE) / eval_horner5_scalar(x, PQ_EOTF_Q_LARGE)
        }
    }

    /// Scalar PQ OETF matching the SIMD rational-polynomial path bit-for-bit.
    #[inline(always)]
    pub(crate) fn pq_oetf_one(v: f32) -> f32 {
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        if v < PQ_OETF_FLOOR {
            return pq_oetf_exact(v);
        }
        let a = libm::sqrtf(libm::sqrtf(v));
        if a < 0.1 {
            eval_horner5_scalar(a, PQ_INV_P_SMALL) / eval_horner5_scalar(a, PQ_INV_Q_SMALL)
        } else {
            eval_horner5_scalar(a, PQ_INV_P_LARGE) / eval_horner5_scalar(a, PQ_INV_Q_LARGE)
        }
    }

    #[inline(never)]
    pub(crate) fn pq_eotf_scalar_slice(vals: &mut [f32], pq_scale: f32) {
        for v in vals {
            *v = pq_eotf_one(*v) * pq_scale;
        }
    }

    #[inline(never)]
    pub(crate) fn pq_oetf_scalar_slice(vals: &mut [f32], pq_inv_scale: f32) {
        for v in vals {
            *v = pq_oetf_one(*v * pq_inv_scale);
        }
    }

    macro_rules! eval_blend5 {
        ($vec:ident, $token:ident, $x:ident, $small:ident, $ps:ident, $pl:ident) => {{
            let c4 = $vec::blend(
                $small,
                $vec::splat_t($token, $ps[4]),
                $vec::splat_t($token, $pl[4]),
            );
            let c3 = $vec::blend(
                $small,
                $vec::splat_t($token, $ps[3]),
                $vec::splat_t($token, $pl[3]),
            );
            let c2 = $vec::blend(
                $small,
                $vec::splat_t($token, $ps[2]),
                $vec::splat_t($token, $pl[2]),
            );
            let c1 = $vec::blend(
                $small,
                $vec::splat_t($token, $ps[1]),
                $vec::splat_t($token, $pl[1]),
            );
            let c0 = $vec::blend(
                $small,
                $vec::splat_t($token, $ps[0]),
                $vec::splat_t($token, $pl[0]),
            );
            (((c4 * $x + c3) * $x + c2) * $x + c1) * $x + c0
        }};
    }

    macro_rules! pq_eotf_simd_body {
        ($vec:ident, $lanes:literal, $token:ident, $vals:ident, $pq_scale:ident) => {{
            let zero = $vec::zero_t($token);
            let one = $vec::splat_t($token, 1.0);
            let floor = $vec::splat_t($token, PQ_EOTF_FLOOR);
            let split = $vec::splat_t($token, 0.25);
            let scale_v = $vec::splat_t($token, $pq_scale);

            let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
            for chunk in chunks {
                let v = $vec::load_t($token, chunk);
                let pos = v.simd_gt(zero);
                let ge_one = v.simd_ge(one);
                let is_dark = pos & v.simd_lt(floor);
                let a = $vec::blend(v.simd_gt(floor), v.min_fast(one), floor);
                let x = a * a + a;
                let small = a.simd_lt(split);
                let p = eval_blend5!($vec, $token, x, small, PQ_EOTF_P_SMALL, PQ_EOTF_P_LARGE);
                let q = eval_blend5!($vec, $token, x, small, PQ_EOTF_Q_SMALL, PQ_EOTF_Q_LARGE);
                let poly = p / q;
                let base = $vec::blend(ge_one, one, $vec::blend(pos, poly, zero));
                let out = base * scale_v;
                if is_dark.bitcast_to_i32().any_true() {
                    let src = *chunk;
                    let mut res = out.to_array();
                    let mask = is_dark.to_array();
                    for i in 0..$lanes {
                        if mask[i].to_bits() != 0 {
                            res[i] = pq_eotf_exact(src[i]) * $pq_scale;
                        }
                    }
                    *chunk = res;
                } else {
                    out.store(chunk);
                }
            }
            for x in rem {
                *x = pq_eotf_one(*x) * $pq_scale;
            }
        }};
    }

    macro_rules! pq_oetf_simd_body {
        ($vec:ident, $lanes:literal, $token:ident, $vals:ident, $pq_inv_scale:ident) => {{
            let zero = $vec::zero_t($token);
            let one = $vec::splat_t($token, 1.0);
            let floor = $vec::splat_t($token, PQ_OETF_FLOOR);
            let split = $vec::splat_t($token, 0.1);
            let inv_s = $vec::splat_t($token, $pq_inv_scale);

            let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
            for chunk in chunks {
                let v = $vec::load_t($token, chunk) * inv_s;
                let pos = v.simd_gt(zero);
                let ge_one = v.simd_ge(one);
                let is_dark = pos & v.simd_lt(floor);
                let y = $vec::blend(v.simd_gt(floor), v.min_fast(one), floor);
                let a = y.sqrt().sqrt();
                let small = a.simd_lt(split);
                let p = eval_blend5!($vec, $token, a, small, PQ_INV_P_SMALL, PQ_INV_P_LARGE);
                let q = eval_blend5!($vec, $token, a, small, PQ_INV_Q_SMALL, PQ_INV_Q_LARGE);
                let poly = p / q;
                let out = $vec::blend(ge_one, one, $vec::blend(pos, poly, zero));
                if is_dark.bitcast_to_i32().any_true() {
                    let scaled = v.to_array();
                    let mut res = out.to_array();
                    let mask = is_dark.to_array();
                    for i in 0..$lanes {
                        if mask[i].to_bits() != 0 {
                            res[i] = pq_oetf_exact(scaled[i]);
                        }
                    }
                    *chunk = res;
                } else {
                    out.store(chunk);
                }
            }
            for x in rem {
                *x = pq_oetf_one(*x * $pq_inv_scale);
            }
        }};
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn pq_eotf_v3_t(token: archmage::X64V3Token, vals: &mut [f32], pq_scale: f32) {
        type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
        pq_eotf_simd_body!(V, 8, token, vals, pq_scale);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn pq_eotf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32], pq_scale: f32) {
        type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
        pq_eotf_simd_body!(V, 16, token, vals, pq_scale);
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn pq_eotf_neon_t(token: archmage::NeonToken, vals: &mut [f32], pq_scale: f32) {
        type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
        pq_eotf_simd_body!(V, 4, token, vals, pq_scale);
    }

    #[cfg(target_arch = "wasm32")]
    #[archmage::arcane]
    fn pq_eotf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32], pq_scale: f32) {
        type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
        pq_eotf_simd_body!(V, 4, token, vals, pq_scale);
    }

    fn pq_eotf_scalar_t(_token: ScalarToken, vals: &mut [f32], pq_scale: f32) {
        pq_eotf_scalar_slice(vals, pq_scale);
    }

    pub(crate) fn pq_eotf_simd_slice(vals: &mut [f32], pq_scale: f32) {
        archmage::attuned!(
            pq_eotf(vals, pq_scale),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn pq_oetf_v3_t(token: archmage::X64V3Token, vals: &mut [f32], pq_inv_scale: f32) {
        type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
        pq_oetf_simd_body!(V, 8, token, vals, pq_inv_scale);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn pq_oetf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32], pq_inv_scale: f32) {
        type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
        pq_oetf_simd_body!(V, 16, token, vals, pq_inv_scale);
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn pq_oetf_neon_t(token: archmage::NeonToken, vals: &mut [f32], pq_inv_scale: f32) {
        type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
        pq_oetf_simd_body!(V, 4, token, vals, pq_inv_scale);
    }

    #[cfg(target_arch = "wasm32")]
    #[archmage::arcane]
    fn pq_oetf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32], pq_inv_scale: f32) {
        type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
        pq_oetf_simd_body!(V, 4, token, vals, pq_inv_scale);
    }

    fn pq_oetf_scalar_t(_token: ScalarToken, vals: &mut [f32], pq_inv_scale: f32) {
        pq_oetf_scalar_slice(vals, pq_inv_scale);
    }

    pub(crate) fn pq_oetf_simd_slice(vals: &mut [f32], pq_inv_scale: f32) {
        archmage::attuned!(
            pq_oetf(vals, pq_inv_scale),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    // -----------------------------------------------------------------------
    // HLG Scene EOTF / OETF — shares exact Expf / Logf options with hlg_render!
    // -----------------------------------------------------------------------

    const HLG_A: f32 = 0.17883277;
    const HLG_B: f32 = 0.28466892;
    const HLG_C: f32 = 0.55991073;
    const HLG_A_LN2: f32 = HLG_A * core::f32::consts::LN_2;
    const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A;

    const HLG_EXP_FAST: u128 = ExpfOptions::new()
        .domain(ExpfDomain::NormalOutput)
        .portability(MATH_PORTABILITY)
        .assume_in_bounds()
        .caller()
        .bits();

    const HLG_LOG_FAST: u128 = LogfOptions::new()
        .domain(LogfDomain::PositiveNormals)
        .reject_subnormals()
        .portability(MATH_PORTABILITY)
        .assume_in_bounds()
        .caller()
        .bits();

    #[cold]
    #[inline(never)]
    fn hlg_scene_eotf_fallback(v: f32) -> f32 {
        const A: f64 = 0.178_832_77;
        const B: f64 = 1.0 - 4.0 * A;
        let c: f64 = 0.5 - A * libm::log(4.0 * A);
        if !v.is_finite() {
            return v;
        }
        let a = f64::from(v.abs());
        let out = if a <= 0.5 {
            (a * a) / 3.0
        } else {
            (libm::exp((a - c) / A) + B) / 12.0
        };
        (out as f32).copysign(v)
    }

    #[cold]
    #[inline(never)]
    fn hlg_scene_oetf_fallback(v: f32) -> f32 {
        const A: f64 = 0.178_832_77;
        const B: f64 = 1.0 - 4.0 * A;
        let c: f64 = 0.5 - A * libm::log(4.0 * A);
        if !v.is_finite() {
            return v;
        }
        let a = f64::from(v.abs());
        let out = if a <= 1.0 / 12.0 {
            libm::sqrt(3.0 * a)
        } else {
            A * libm::log(12.0 * a - B) + c
        };
        (out as f32).copysign(v)
    }

    #[inline(always)]
    fn hlg_scene_eotf_one(v: f32) -> f32 {
        let a = v.abs();
        if a <= 0.5 {
            return ((a * a) * (1.0 / 3.0)).copysign(v);
        }
        let exp_arg = (a - HLG_C) * HLG_INV_A_LOG2E;
        if exp_arg <= 127.0 {
            let e = Expf::<HLG_EXP_FAST>::exp2_with_token(ScalarToken, exp_arg)
                .unwrap_or_else(|_| hlg_scene_eotf_fallback(a));
            ((e + HLG_B) * (1.0 / 12.0)).copysign(v)
        } else {
            hlg_scene_eotf_fallback(v)
        }
    }

    #[inline(always)]
    fn hlg_scene_oetf_one(v: f32) -> f32 {
        let a = v.abs();
        if a <= 1.0 / 12.0 {
            return libm::sqrtf(3.0 * a).copysign(v);
        }
        let log_arg = 12.0 * a - HLG_B;
        if log_arg.is_finite() {
            let l = Logf::<HLG_LOG_FAST>::log2_with_token(ScalarToken, log_arg)
                .unwrap_or_else(|_| hlg_scene_oetf_fallback(a));
            (HLG_A_LN2 * l + HLG_C).copysign(v)
        } else {
            hlg_scene_oetf_fallback(v)
        }
    }

    #[inline(never)]
    pub(crate) fn hlg_scene_eotf_scalar_slice(vals: &mut [f32]) {
        for v in vals {
            *v = hlg_scene_eotf_one(*v);
        }
    }

    #[inline(never)]
    pub(crate) fn hlg_scene_oetf_scalar_slice(vals: &mut [f32]) {
        for v in vals {
            *v = hlg_scene_oetf_one(*v);
        }
    }

    macro_rules! hlg_scene_eotf_simd_body {
        ($vec:ident, $lanes:literal, $token:ident, $vals:ident) => {{
            let zero = $vec::zero_t($token);
            let half = $vec::splat_t($token, 0.5);
            let inv3 = $vec::splat_t($token, 1.0 / 3.0);
            let inv12 = $vec::splat_t($token, 1.0 / 12.0);
            let b_c = $vec::splat_t($token, HLG_B);
            let c_c = $vec::splat_t($token, HLG_C);
            let inv_a = $vec::splat_t($token, HLG_INV_A_LOG2E);
            let exp_max = $vec::splat_t($token, 127.0);
            let neg_zero = $vec::splat_t($token, -0.0);

            let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
            for chunk in chunks {
                let v = $vec::load_t($token, chunk);
                let a = v.abs();
                let sign = v & neg_zero;
                let le_half = a.simd_le(half);
                let exp_arg = (a - c_c) * inv_a;
                let safe_arg = $vec::blend(le_half, zero, exp_arg);
                if safe_arg.simd_le(exp_max).bitcast_to_i32().all_true() {
                    let lo = (a * a) * inv3;
                    let hi = (safe_arg.exp2::<HLG_EXP_FAST>() + b_c) * inv12;
                    let out = $vec::blend(le_half, lo, hi) | sign;
                    out.store(chunk);
                } else {
                    for x in chunk.iter_mut() {
                        *x = hlg_scene_eotf_one(*x);
                    }
                }
            }
            for x in rem {
                *x = hlg_scene_eotf_one(*x);
            }
        }};
    }

    macro_rules! hlg_scene_oetf_simd_body {
        ($vec:ident, $lanes:literal, $token:ident, $vals:ident) => {{
            let zero = $vec::zero_t($token);
            let one = $vec::splat_t($token, 1.0);
            let inv12 = $vec::splat_t($token, 1.0 / 12.0);
            let three = $vec::splat_t($token, 3.0);
            let twelve = $vec::splat_t($token, 12.0);
            let b_c = $vec::splat_t($token, HLG_B);
            let c_c = $vec::splat_t($token, HLG_C);
            let a_ln2 = $vec::splat_t($token, HLG_A_LN2);
            let neg_zero = $vec::splat_t($token, -0.0);

            let (chunks, rem) = $vals.as_chunks_mut::<$lanes>();
            for chunk in chunks {
                let v = $vec::load_t($token, chunk);
                let a = v.abs();
                let sign = v & neg_zero;
                let le_12 = a.simd_le(inv12);
                let log_arg = twelve * a - b_c;
                let safe_arg = $vec::blend(le_12, one, log_arg);
                if (safe_arg * zero).simd_eq(zero).bitcast_to_i32().all_true() {
                    let lo = (three * a).sqrt();
                    let hi = a_ln2 * safe_arg.log2::<HLG_LOG_FAST>() + c_c;
                    let out = $vec::blend(le_12, lo, hi) | sign;
                    out.store(chunk);
                } else {
                    for x in chunk.iter_mut() {
                        *x = hlg_scene_oetf_one(*x);
                    }
                }
            }
            for x in rem {
                *x = hlg_scene_oetf_one(*x);
            }
        }};
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn hlg_scene_eotf_v3_t(token: archmage::X64V3Token, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
        hlg_scene_eotf_simd_body!(V, 8, token, vals);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn hlg_scene_eotf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
        hlg_scene_eotf_simd_body!(V, 16, token, vals);
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn hlg_scene_eotf_neon_t(token: archmage::NeonToken, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
        hlg_scene_eotf_simd_body!(V, 4, token, vals);
    }

    #[cfg(target_arch = "wasm32")]
    #[archmage::arcane]
    fn hlg_scene_eotf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
        hlg_scene_eotf_simd_body!(V, 4, token, vals);
    }

    fn hlg_scene_eotf_scalar_t(_token: ScalarToken, vals: &mut [f32]) {
        hlg_scene_eotf_scalar_slice(vals);
    }

    pub(crate) fn hlg_scene_eotf_simd_slice(vals: &mut [f32]) {
        archmage::attuned!(
            hlg_scene_eotf(vals),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn hlg_scene_oetf_v3_t(token: archmage::X64V3Token, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
        hlg_scene_oetf_simd_body!(V, 8, token, vals);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn hlg_scene_oetf_v4x_t(token: archmage::X64V4xToken, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
        hlg_scene_oetf_simd_body!(V, 16, token, vals);
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn hlg_scene_oetf_neon_t(token: archmage::NeonToken, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
        hlg_scene_oetf_simd_body!(V, 4, token, vals);
    }

    #[cfg(target_arch = "wasm32")]
    #[archmage::arcane]
    fn hlg_scene_oetf_wasm128_t(token: archmage::Wasm128Token, vals: &mut [f32]) {
        type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
        hlg_scene_oetf_simd_body!(V, 4, token, vals);
    }

    fn hlg_scene_oetf_scalar_t(_token: ScalarToken, vals: &mut [f32]) {
        hlg_scene_oetf_scalar_slice(vals);
    }

    pub(crate) fn hlg_scene_oetf_simd_slice(vals: &mut [f32]) {
        archmage::attuned!(
            hlg_scene_oetf(vals),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
}
