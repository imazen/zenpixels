//! Compact universal fast-FMA scalar conversion engine.
//!
//! [`UniversalPlan`] executes any valid native pixel conversion (`U8`, `U16`, `F16`,
//! `F32` × `Gray`/`GrayAlpha`/`Rgb`/`Bgr`/`Rgba`/`Bgra` + optional `Oklab`/`OklabA` ×
//! `Linear`/`Srgb`/`Bt709`/`Gamma22` + optional `Pq`/`Hlg` × all named primaries ×
//! straight/premultiplied/opaque/undefined alpha × matte compositing × luma collapse)
//! in a single pass with **zero scratch allocation** and **zero SIMD codegen**.
//!
//! # Bit-exact parity with SIMD accelerators
//!
//! Rather than canonicalizing a separate slow reference formula, every arithmetic
//! kernel here runs the exact same canonical fast-FMA math as the x86-64/AArch64
//! SIMD accelerators:
//!
//! - `sRGB` and `PQ` use the shared C0-continuous rational polynomials with single-rounding
//!   FMA (outlined so software FMA is emitted once, not duplicated across Horner loops).
//! - `BT.709`, `Gamma 2.2`, and `HLG` use `magemath` single-lane `ScalarToken` graphs
//!   with `.caller()` + `MATH_PORTABILITY` (`BitExact` on native and `precise-portable`
//!   WASM; `BackendDependent` on default WASM).
//! - 3×3 gamut matrices match `fast_gamut::fused_row` FMA association.
//! - Integer normalization (`/ 255.0`, `/ 65535.0`) and `f64` rounding match `float_simd`
//!   0-ULP / 0-mismatch quantization across all `f32` values in `[0, 1]`.
//! - `Oklab` uses `oklab::fast_cbrt` and `OklabSpace::for_primaries`.

#![allow(clippy::excessive_precision)]

use archmage::ScalarToken;
use magemath::powf::{Powf, PowfDomain, PowfOptions};
#[cfg(feature = "hdr-transfers")]
use magemath::{
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
};
use whereat::At;
use zenpixels::sample::FloatFacts;
use zenpixels::{
    AlphaMode, AlphaPolicy, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, ConvertOptions,
    DepthPolicy, GamutPolicy, LumaCoefficients, PixelDescriptor, SignalRange, TransferFunction,
};

use crate::ConvertError;
use crate::f16_scalar::{f16_bits_to_f32, f32_to_f16_bits};
use crate::route::{AccuracyContract, KernelTier, PlannedStepInfo, SemanticEffects};

// ---------------------------------------------------------------------------
// Canonical fast-FMA helpers (matches linear-srgb & fast_gamut bit-for-bit)
// ---------------------------------------------------------------------------

const USE_FUSED_MATH: bool = !cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
));

const MATH_PORTABILITY: magemath::Portability = if USE_FUSED_MATH {
    magemath::Portability::BitExact
} else {
    magemath::Portability::BackendDependent
};

/// Single-rounding FMA matching native hardware FMA (and `precise-portable` WASM).
/// Outlined so software FMA on scalar x86-64/WASM is emitted once instead of inlined
/// at every Horner step.
#[inline(never)]
fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    if USE_FUSED_MATH {
        magetypes::scalar::mul_add_fused_f32(a, b, c)
    } else {
        a * b + c
    }
}

/// Outlined rational polynomial evaluator P(x)/Q(x) using Horner's method with [`fma_f32`].
#[inline(never)]
fn eval_rational_poly(x: f32, p: &[f32], q: &[f32]) -> f32 {
    let mut yp = p[p.len() - 1];
    let mut i = p.len() - 1;
    while i > 0 {
        i -= 1;
        yp = fma_f32(yp, x, p[i]);
    }
    let mut yq = q[q.len() - 1];
    let mut j = q.len() - 1;
    while j > 0 {
        j -= 1;
        yq = fma_f32(yq, x, q[j]);
    }
    yp / yq
}

#[inline(never)]
fn mul3_fma(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        fma_f32(m[0][0], v[0], fma_f32(m[0][1], v[1], m[0][2] * v[2])),
        fma_f32(m[1][0], v[0], fma_f32(m[1][1], v[1], m[1][2] * v[2])),
        fma_f32(m[2][0], v[0], fma_f32(m[2][1], v[1], m[2][2] * v[2])),
    ]
}

// ---------------------------------------------------------------------------
// Transfer-function constants (exact match to linear-srgb SIMD kernels)
// ---------------------------------------------------------------------------

const SRGB_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const SRGB_A: f32 = 0.055_010_72;
const SRGB_A1: f32 = 1.0 + SRGB_A;

const S2L_P: [f32; 5] = [
    1.672_814_6e-2,
    8.089_536_4e-1,
    1.288_439_66e1,
    6.854_374_7e1,
    8.224_625_4e1,
];
const S2L_Q: [f32; 5] = [
    2.003_807_83e1,
    9.681_468_2e1,
    5.377_303_3e1,
    -7.125_638,
    1.0,
];
const L2S_P: [f32; 5] = [
    -1.356_440_97e-2,
    9.336_896_24e-2,
    1.157_270_53e1,
    5.002_124_79e1,
    2.792_420_96e1,
];
const L2S_Q: [f32; 5] = [
    2.633_812_73e-1,
    8.998_917_58,
    4.477_739_72e1,
    3.455_830_76e1,
    1.0,
];
const EXT_S2L_P: [f32; 7] = [
    1.802_136_5e1,
    9.110_411_4e2,
    1.570_602_1e4,
    1.020_638_2e5,
    2.199_931_2e5,
    1.338_269_2e5,
    1.706_519_4e4,
];
const EXT_S2L_Q: [f32; 7] = [
    2.159_401_7e4,
    1.508_555_1e5,
    2.303_299_0e5,
    8.239_410_8e4,
    4.473_249_1e3,
    -6.359_000_1e1,
    1.0,
];
const EXT_L2S_P: [f32; 7] = [
    -1.025_467_4,
    -3.075_361_5e-1,
    1.027_286e3,
    7.093_665e3,
    1.006_868_9e4,
    3.230_716e3,
    1.769_130_4e2,
];
const EXT_L2S_Q: [f32; 7] = [
    1.977_460_5e1,
    8.308_271e2,
    6.024_792_5e3,
    1.024_407_5e4,
    4.157_534e3,
    3.179_324_6e2,
    1.0,
];

const ADOBE_GAMMA: f32 = 563.0 / 256.0;
const INV_ADOBE_GAMMA: f32 = 1.0 / ADOBE_GAMMA;
const BT709_ALPHA: f32 = 0.09929682680944;
const BT709_BETA: f32 = 0.018053968510807;
const BT709_DECODE_MIN: f32 = (4.5 * BT709_BETA) * (1.0 / 4.5);

const GAMMA_DECODE_POW: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveUnit)
    .const_exponent(ADOBE_GAMMA)
    .reject_subnormals()
    .saturate_output()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
const GAMMA_ENCODE_POW: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveUnit)
    .const_exponent(INV_ADOBE_GAMMA)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
const GAMMA_EXT_DECODE_HDR: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveGamma)
    .base_range(1.0e-16, 65_536.0)
    .const_exponent(ADOBE_GAMMA)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
const GAMMA_EXT_ENCODE_HDR: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveGamma)
    .base_range(f32::MIN_POSITIVE, 65_536.0)
    .const_exponent(INV_ADOBE_GAMMA)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
const BT709_DECODE_FAST: u128 = PowfOptions::new()
    .positive_gamma()
    .reject_subnormals()
    .const_exponent(1.0 / 0.45)
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
const BT709_ENCODE_FAST: u128 = PowfOptions::new()
    .positive_gamma()
    .reject_subnormals()
    .const_exponent(0.45)
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

#[cfg(feature = "hdr-transfers")]
mod hdr_consts {
    use super::*;

    pub(super) const PQ_M1_F64: f64 = 2610.0 / 16384.0;
    pub(super) const PQ_M2_F64: f64 = 2523.0 / 32.0;
    pub(super) const PQ_C1_F64: f64 = 3424.0 / 4096.0;
    pub(super) const PQ_C2_F64: f64 = 2413.0 / 128.0;
    pub(super) const PQ_C3_F64: f64 = 2392.0 / 128.0;
    pub(super) const PQ_EOTF_POLY_FLOOR: f32 = 0.02;
    pub(super) const PQ_OETF_POLY_FLOOR: f32 = 1e-7;

    pub(super) const PQ_EOTF_P_SMALL: [f32; 5] = [
        4.36033203e-10,
        -1.53651077e-07,
        2.06864196e-06,
        2.80304711e-03,
        1.47809948e-02,
    ];
    pub(super) const PQ_EOTF_Q_SMALL: [f32; 5] = [
        -9.89955022e-03,
        1.89911767e+00,
        -1.34828020e+00,
        -7.10547502e-01,
        1.00000000e+00,
    ];
    pub(super) const PQ_EOTF_P_LARGE: [f32; 5] = [
        5.43046166e-04,
        -6.31017431e-03,
        2.90480734e-01,
        9.80363032e-01,
        2.22819483e-01,
    ];
    pub(super) const PQ_EOTF_Q_LARGE: [f32; 5] = [
        1.58688561e+02,
        -1.60764060e+02,
        6.51370372e+01,
        -1.26438391e+01,
        1.00000000e+00,
    ];
    pub(super) const PQ_INV_P_LARGE: [f32; 5] =
        [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
    pub(super) const PQ_INV_Q_LARGE: [f32; 5] =
        [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];
    pub(super) const PQ_INV_P_SMALL: [f32; 5] = [
        9.863406e-6,
        3.881234e-1,
        1.352821e2,
        6.889862e4,
        -2.864824e5,
    ];
    pub(super) const PQ_INV_Q_SMALL: [f32; 5] =
        [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

    pub(super) const HLG_A: f32 = 0.17883277;
    pub(super) const HLG_B: f32 = 0.28466892;
    pub(super) const HLG_C: f32 = 0.55991073;
    pub(super) const HLG_A_INV_LOG2E: f32 = HLG_A * core::f32::consts::LN_2;
    pub(super) const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A;

    pub(super) const HLG_EXP: u128 = ExpfOptions::new()
        .full_domain()
        .portability(MATH_PORTABILITY)
        .caller()
        .bits();
    pub(super) const HLG_EXP_FAST: u128 = ExpfOptions::new()
        .unclamped_exp2_range()
        .portability(MATH_PORTABILITY)
        .assume_in_bounds()
        .caller()
        .bits();
    pub(super) const HLG_LOG: u128 = LogfOptions::new()
        .full_domain()
        .portability(MATH_PORTABILITY)
        .caller()
        .bits();
    pub(super) const HLG_LOG_FAST: u128 = LogfOptions::new()
        .reject_subnormals()
        .portability(MATH_PORTABILITY)
        .assume_in_bounds()
        .caller()
        .bits();

    #[cold]
    #[inline(never)]
    pub(super) fn pq_to_linear_precise(v: f32) -> f32 {
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        let vp = libm::pow(f64::from(v), 1.0 / PQ_M2_F64);
        libm::pow(
            (vp - PQ_C1_F64).max(0.0) / (PQ_C2_F64 - PQ_C3_F64 * vp),
            1.0 / PQ_M1_F64,
        ) as f32
    }

    #[cold]
    #[inline(never)]
    pub(super) fn linear_to_pq_precise(v: f32) -> f32 {
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        let vp = libm::pow(f64::from(v), PQ_M1_F64);
        libm::pow(
            (PQ_C1_F64 + PQ_C2_F64 * vp) / (1.0 + PQ_C3_F64 * vp),
            PQ_M2_F64,
        ) as f32
    }
}

#[inline(always)]
fn gamma_decode_unit(v: f32) -> f32 {
    if v < f32::MIN_POSITIVE {
        return 0.0;
    }
    Powf::<GAMMA_DECODE_POW>::scalar_const_with_token(ScalarToken, v).unwrap_or(0.0)
}

#[inline(always)]
fn gamma_encode_unit(v: f32) -> f32 {
    if v < f32::MIN_POSITIVE {
        return 0.0;
    }
    Powf::<GAMMA_ENCODE_POW>::scalar_const_with_token(ScalarToken, v).unwrap_or(0.0)
}

#[cold]
#[inline(never)]
fn fallback_pow_f32(base: f32, exp: f32) -> f32 {
    libm::pow(f64::from(base), f64::from(exp)) as f32
}

#[inline(never)]
pub(crate) fn scalar_eotf(tf: TransferFunction, v: f32, clip: bool) -> f32 {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => v,
        TransferFunction::Srgb => {
            if clip {
                if v <= 0.0 || v.is_nan() {
                    return 0.0;
                }
                if v >= 1.0 {
                    return 1.0;
                }
                if v <= SRGB_THRESHOLD {
                    v * (1.0 / 12.92)
                } else {
                    eval_rational_poly(v, &S2L_P, &S2L_Q)
                }
            } else {
                let a = v.abs();
                let y = if a < SRGB_THRESHOLD {
                    a * (1.0 / 12.92)
                } else if a <= 8.0 {
                    eval_rational_poly(a, &EXT_S2L_P, &EXT_S2L_Q)
                } else {
                    fallback_pow_f32((a + SRGB_A) / SRGB_A1, 2.4)
                };
                y.abs().copysign(v)
            }
        }
        TransferFunction::Bt709 => {
            if v.is_nan() {
                v
            } else if v < 4.5 * BT709_BETA {
                v * (1.0 / 4.5)
            } else {
                let norm = (v + BT709_ALPHA) / (1.0 + BT709_ALPHA);
                let p = if norm <= 65_536.0 {
                    Powf::<BT709_DECODE_FAST>::scalar_const_with_token(ScalarToken, norm)
                        .unwrap_or_else(|_| fallback_pow_f32(norm, 1.0 / 0.45))
                } else {
                    fallback_pow_f32(norm, 1.0 / 0.45)
                };
                p.max(BT709_DECODE_MIN)
            }
        }
        TransferFunction::Gamma22 => {
            if clip {
                if v.is_nan() {
                    return v;
                }
                if v <= 0.0 {
                    return 0.0;
                }
                if v >= 1.0 {
                    return 1.0;
                }
                gamma_decode_unit(v)
            } else {
                let a = v.abs();
                let y = if a < f32::MIN_POSITIVE {
                    0.0
                } else if a < 1.0e-16 {
                    gamma_decode_unit(a)
                } else if a <= 65_536.0 {
                    Powf::<GAMMA_EXT_DECODE_HDR>::scalar_const_with_token(ScalarToken, a)
                        .unwrap_or_else(|_| libm::powf(a, ADOBE_GAMMA))
                } else {
                    libm::powf(a, ADOBE_GAMMA)
                };
                y.copysign(v)
            }
        }
        #[cfg(feature = "hdr-transfers")]
        TransferFunction::Pq => {
            use hdr_consts::*;
            if v.is_nan() || v <= 0.0 || v >= 1.0 || v < PQ_EOTF_POLY_FLOOR {
                return pq_to_linear_precise(v);
            }
            let x = fma_f32(v, v, v);
            if v < 0.25 {
                eval_rational_poly(x, &PQ_EOTF_P_SMALL, &PQ_EOTF_Q_SMALL)
            } else {
                eval_rational_poly(x, &PQ_EOTF_P_LARGE, &PQ_EOTF_Q_LARGE)
            }
        }
        #[cfg(feature = "hdr-transfers")]
        TransferFunction::Hlg => {
            use hdr_consts::*;
            if v <= 0.0 || v.is_nan() {
                0.0
            } else if v <= 0.5 {
                (v * v) * (1.0 / 3.0)
            } else {
                let arg = (v - HLG_C) * HLG_INV_A_LOG2E;
                let e = if arg <= 127.0 {
                    Expf::<HLG_EXP_FAST>::exp2_with_token(ScalarToken, arg).unwrap_or(0.0)
                } else {
                    Expf::<HLG_EXP>::exp2_with_token(ScalarToken, arg).unwrap_or(0.0)
                };
                (e + HLG_B) * (1.0 / 12.0)
            }
        }
        _ => v,
    }
}

#[inline(never)]
pub(crate) fn scalar_oetf(tf: TransferFunction, v: f32, clip: bool) -> f32 {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => v,
        TransferFunction::Srgb => {
            if clip {
                if v <= 0.0 || v.is_nan() {
                    return 0.0;
                }
                if v >= 1.0 {
                    return 1.0;
                }
                if v <= LINEAR_THRESHOLD {
                    v * 12.92
                } else {
                    eval_rational_poly(magetypes::scalar::sqrt_f32(v), &L2S_P, &L2S_Q)
                }
            } else {
                let a = v.abs();
                let y = if a < LINEAR_THRESHOLD {
                    a * 12.92
                } else if a <= 64.0 {
                    eval_rational_poly(magetypes::scalar::sqrt_f32(a), &EXT_L2S_P, &EXT_L2S_Q)
                } else {
                    fma_f32(SRGB_A1, fallback_pow_f32(a, 1.0 / 2.4), -SRGB_A)
                };
                y.abs().copysign(v)
            }
        }
        TransferFunction::Bt709 => {
            if v.is_nan() {
                v
            } else if v < BT709_BETA {
                4.5 * v
            } else {
                let p = if v <= 65_536.0 {
                    Powf::<BT709_ENCODE_FAST>::scalar_const_with_token(ScalarToken, v)
                        .unwrap_or_else(|_| fallback_pow_f32(v, 0.45))
                } else {
                    fallback_pow_f32(v, 0.45)
                };
                fma_f32(1.0 + BT709_ALPHA, p, -BT709_ALPHA)
            }
        }
        TransferFunction::Gamma22 => {
            if clip {
                if v.is_nan() {
                    return v;
                }
                if v <= 0.0 {
                    return 0.0;
                }
                if v >= 1.0 {
                    return 1.0;
                }
                gamma_encode_unit(v)
            } else {
                let a = v.abs();
                let y = if a < f32::MIN_POSITIVE {
                    0.0
                } else if a <= 65_536.0 {
                    Powf::<GAMMA_EXT_ENCODE_HDR>::scalar_const_with_token(ScalarToken, a)
                        .unwrap_or_else(|_| libm::powf(a, INV_ADOBE_GAMMA))
                } else {
                    libm::powf(a, INV_ADOBE_GAMMA)
                };
                y.copysign(v)
            }
        }
        #[cfg(feature = "hdr-transfers")]
        TransferFunction::Pq => {
            use hdr_consts::*;
            if v.is_nan() || v <= 0.0 || v >= 1.0 || v < PQ_OETF_POLY_FLOOR {
                return linear_to_pq_precise(v);
            }
            let a = magetypes::scalar::sqrt_f32(magetypes::scalar::sqrt_f32(v));
            if a < 0.1 {
                eval_rational_poly(a, &PQ_INV_P_SMALL, &PQ_INV_Q_SMALL)
            } else {
                eval_rational_poly(a, &PQ_INV_P_LARGE, &PQ_INV_Q_LARGE)
            }
        }
        #[cfg(feature = "hdr-transfers")]
        TransferFunction::Hlg => {
            use hdr_consts::*;
            if v <= 0.0 || v.is_nan() {
                0.0
            } else if v <= 1.0 / 12.0 {
                magetypes::scalar::sqrt_f32(3.0 * v)
            } else {
                let arg = 12.0 * v - HLG_B;
                let l = if arg.is_finite() {
                    Logf::<HLG_LOG_FAST>::log2_with_token(ScalarToken, arg).unwrap_or(0.0)
                } else {
                    Logf::<HLG_LOG>::log2_with_token(ScalarToken, arg).unwrap_or(0.0)
                };
                fma_f32(HLG_A_INV_LOG2E, l, HLG_C).max(0.5)
            }
        }
        _ => v,
    }
}

// ---------------------------------------------------------------------------
// Pixel unpack / pack (0-ULP match with float_simd)
// ---------------------------------------------------------------------------

#[inline]
fn read_sample(bytes: &[u8], ct: ChannelType, idx: usize) -> f32 {
    match ct {
        ChannelType::U8 => f32::from(bytes[idx]) / 255.0,
        ChannelType::U16 => {
            let o = idx * 2;
            f32::from(u16::from_ne_bytes([bytes[o], bytes[o + 1]])) / 65535.0
        }
        ChannelType::F16 => {
            let o = idx * 2;
            f16_bits_to_f32(u16::from_ne_bytes([bytes[o], bytes[o + 1]]))
        }
        ChannelType::F32 => {
            let o = idx * 4;
            f32::from_ne_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        }
        _ => 0.0,
    }
}

#[inline]
fn write_sample(bytes: &mut [u8], ct: ChannelType, idx: usize, v: f32, truncate: bool) {
    match ct {
        ChannelType::U8 => {
            let clamped = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            bytes[idx] = if truncate {
                (clamped * 255.0) as u8
            } else {
                (f64::from(clamped) * 255.0 + 0.5) as u8
            };
        }
        ChannelType::U16 => {
            let clamped = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            let val = if truncate {
                (clamped * 65535.0) as u16
            } else {
                (f64::from(clamped) * 65535.0 + 0.5) as u16
            };
            let o = idx * 2;
            bytes[o..o + 2].copy_from_slice(&val.to_ne_bytes());
        }
        ChannelType::F16 => {
            let o = idx * 2;
            bytes[o..o + 2].copy_from_slice(&f32_to_f16_bits(v).to_ne_bytes());
        }
        ChannelType::F32 => {
            let o = idx * 4;
            bytes[o..o + 4].copy_from_slice(&v.to_ne_bytes());
        }
        _ => {}
    }
}

#[inline(never)]
fn unpack_pixel(src: &[u8], d: PixelDescriptor) -> [f32; 4] {
    let ct = d.channel_type();
    match d.layout() {
        ChannelLayout::Gray => {
            let g = read_sample(src, ct, 0);
            [g, g, g, 1.0]
        }
        ChannelLayout::GrayAlpha => {
            let g = read_sample(src, ct, 0);
            let a = read_sample(src, ct, 1);
            [g, g, g, a]
        }
        ChannelLayout::Rgb | ChannelLayout::Oklab => [
            read_sample(src, ct, 0),
            read_sample(src, ct, 1),
            read_sample(src, ct, 2),
            1.0,
        ],
        ChannelLayout::Bgr => [
            read_sample(src, ct, 2),
            read_sample(src, ct, 1),
            read_sample(src, ct, 0),
            1.0,
        ],
        ChannelLayout::Rgba | ChannelLayout::OklabA => {
            let a = if d.alpha() == Some(AlphaMode::Undefined) {
                1.0
            } else {
                read_sample(src, ct, 3)
            };
            [
                read_sample(src, ct, 0),
                read_sample(src, ct, 1),
                read_sample(src, ct, 2),
                a,
            ]
        }
        ChannelLayout::Bgra => {
            let a = if d.alpha() == Some(AlphaMode::Undefined) {
                1.0
            } else {
                read_sample(src, ct, 3)
            };
            [
                read_sample(src, ct, 2),
                read_sample(src, ct, 1),
                read_sample(src, ct, 0),
                a,
            ]
        }
        _ => [0.0, 0.0, 0.0, 1.0],
    }
}

#[inline(never)]
fn pack_pixel(
    dst: &mut [u8],
    d: PixelDescriptor,
    rgba: [f32; 4],
    luma: [f32; 3],
    from_gray: bool,
    trunc: bool,
    u8_srgb_lut_out: bool,
) {
    let ct = d.channel_type();
    if u8_srgb_lut_out && ct == ChannelType::U8 {
        match d.layout() {
            ChannelLayout::Gray => {
                if from_gray {
                    dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                } else {
                    let r = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[0]));
                    let g = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[1]));
                    let b = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[2]));
                    dst[0] =
                        (r * luma[0] + g * luma[1] + b * luma[2] + 0.5).clamp(0.0, 255.0) as u8;
                }
                return;
            }
            ChannelLayout::GrayAlpha => {
                if from_gray {
                    dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                } else {
                    let r = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[0]));
                    let g = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[1]));
                    let b = f32::from(linear_srgb::default::linear_to_srgb_u8(rgba[2]));
                    dst[0] =
                        (r * luma[0] + g * luma[1] + b * luma[2] + 0.5).clamp(0.0, 255.0) as u8;
                }
                write_sample(dst, ct, 1, rgba[3], trunc);
                return;
            }
            ChannelLayout::Rgb => {
                dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                dst[1] = linear_srgb::default::linear_to_srgb_u8(rgba[1]);
                dst[2] = linear_srgb::default::linear_to_srgb_u8(rgba[2]);
                return;
            }
            ChannelLayout::Bgr => {
                dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[2]);
                dst[1] = linear_srgb::default::linear_to_srgb_u8(rgba[1]);
                dst[2] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                return;
            }
            ChannelLayout::Rgba => {
                dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                dst[1] = linear_srgb::default::linear_to_srgb_u8(rgba[1]);
                dst[2] = linear_srgb::default::linear_to_srgb_u8(rgba[2]);
                let a = if d.alpha() == Some(AlphaMode::Undefined) {
                    1.0
                } else {
                    rgba[3]
                };
                write_sample(dst, ct, 3, a, trunc);
                return;
            }
            ChannelLayout::Bgra => {
                dst[0] = linear_srgb::default::linear_to_srgb_u8(rgba[2]);
                dst[1] = linear_srgb::default::linear_to_srgb_u8(rgba[1]);
                dst[2] = linear_srgb::default::linear_to_srgb_u8(rgba[0]);
                let a = if d.alpha() == Some(AlphaMode::Undefined) {
                    1.0
                } else {
                    rgba[3]
                };
                write_sample(dst, ct, 3, a, trunc);
                return;
            }
            _ => {}
        }
    }
    match d.layout() {
        ChannelLayout::Gray => {
            let y = if from_gray {
                rgba[0]
            } else {
                luma[0] * rgba[0] + luma[1] * rgba[1] + luma[2] * rgba[2]
            };
            write_sample(dst, ct, 0, y, trunc);
        }
        ChannelLayout::GrayAlpha => {
            let y = if from_gray {
                rgba[0]
            } else {
                luma[0] * rgba[0] + luma[1] * rgba[1] + luma[2] * rgba[2]
            };
            write_sample(dst, ct, 0, y, trunc);
            write_sample(dst, ct, 1, rgba[3], trunc);
        }
        ChannelLayout::Rgb | ChannelLayout::Oklab => {
            write_sample(dst, ct, 0, rgba[0], trunc);
            write_sample(dst, ct, 1, rgba[1], trunc);
            write_sample(dst, ct, 2, rgba[2], trunc);
        }
        ChannelLayout::Bgr => {
            write_sample(dst, ct, 0, rgba[2], trunc);
            write_sample(dst, ct, 1, rgba[1], trunc);
            write_sample(dst, ct, 2, rgba[0], trunc);
        }
        ChannelLayout::Rgba | ChannelLayout::OklabA => {
            write_sample(dst, ct, 0, rgba[0], trunc);
            write_sample(dst, ct, 1, rgba[1], trunc);
            write_sample(dst, ct, 2, rgba[2], trunc);
            let a = if d.alpha() == Some(AlphaMode::Undefined) {
                1.0
            } else {
                rgba[3]
            };
            write_sample(dst, ct, 3, a, trunc);
        }
        ChannelLayout::Bgra => {
            write_sample(dst, ct, 0, rgba[2], trunc);
            write_sample(dst, ct, 1, rgba[1], trunc);
            write_sample(dst, ct, 2, rgba[0], trunc);
            let a = if d.alpha() == Some(AlphaMode::Undefined) {
                1.0
            } else {
                rgba[3]
            };
            write_sample(dst, ct, 3, a, trunc);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// UniversalPlan
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct ScalarJob {
    from: PixelDescriptor,
    to: PixelDescriptor,
    need_linear: bool,
    gamut_matrix: Option<[[f32; 3]; 3]>,
    matte_bg_linear: Option<[f32; 3]>,
    pq_scale_in: f32,
    pq_scale_out: f32,
    luma: [f32; 3],
    clip: bool,
    truncate: bool,
    from_oklab: bool,
    to_oklab: bool,
    #[cfg(feature = "oklab")]
    oklab_in_matrix: [[f32; 3]; 3],
    #[cfg(feature = "oklab")]
    oklab_out_matrix: [[f32; 3]; 3],
}

/// Self-contained, zero-scratch fast-FMA scalar conversion plan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UniversalPlan {
    job: ScalarJob,
    input_facts: FloatFacts,
    output_facts: FloatFacts,
    #[cfg_attr(not(feature = "hdr-transfers"), allow(dead_code))]
    pq_anchor_scale: f32,
    semantic: SemanticEffects,
    accuracy: AccuracyContract,
}

fn is_supported_ct(ct: ChannelType) -> bool {
    matches!(
        ct,
        ChannelType::U8 | ChannelType::U16 | ChannelType::F16 | ChannelType::F32
    )
}

fn is_native_layout(layout: ChannelLayout) -> bool {
    match layout {
        ChannelLayout::Gray
        | ChannelLayout::GrayAlpha
        | ChannelLayout::Rgb
        | ChannelLayout::Bgr
        | ChannelLayout::Rgba
        | ChannelLayout::Bgra => true,
        #[cfg(feature = "oklab")]
        ChannelLayout::Oklab | ChannelLayout::OklabA => true,
        _ => false,
    }
}

fn is_hdr_to_sdr(from: TransferFunction, to: TransferFunction) -> bool {
    matches!(from, TransferFunction::Pq | TransferFunction::Hlg)
        && matches!(
            to,
            TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22
        )
}

impl UniversalPlan {
    pub(crate) const fn from(&self) -> PixelDescriptor {
        self.job.from
    }

    pub(crate) const fn to(&self) -> PixelDescriptor {
        self.job.to
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.job.from == self.job.to
    }

    pub(crate) const fn input_facts(&self) -> FloatFacts {
        self.input_facts
    }

    pub(crate) const fn output_facts(&self) -> FloatFacts {
        self.output_facts
    }

    #[cfg(feature = "hdr-transfers")]
    pub(crate) fn scale_matte_bg(&mut self, factor: f32) {
        if let Some(bg) = &mut self.job.matte_bg_linear {
            for v in bg {
                *v *= factor;
            }
        }
    }

    pub(crate) fn step_info(&self) -> PlannedStepInfo {
        PlannedStepInfo {
            operation: if self.is_identity() {
                "Identity"
            } else {
                "UniversalScalar"
            },
            from: self.job.from,
            to: self.job.to,
            tier: KernelTier::Scalar,
            accuracy: self.accuracy,
            semantic: self.semantic,
        }
    }

    /// Build a universal scalar plan, enforcing the exact same descriptor, policy,
    /// feature-gate, and `preserve_samples` (`exact`) contracts as `NativePlan`.
    #[track_caller]
    pub(crate) fn try_build(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
        exact: bool,
    ) -> Result<Self, At<ConvertError>> {
        let hlg_boundary = from.transfer() != to.transfer()
            && (from.transfer() == TransferFunction::Hlg || to.transfer() == TransferFunction::Hlg);
        let hlg_composite = from.transfer() == TransferFunction::Hlg
            && to.transfer() == TransferFunction::Hlg
            && from.has_alpha()
            && !to.has_alpha()
            && matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. });
        if hlg_boundary || hlg_composite {
            #[cfg(not(feature = "hdr-transfers"))]
            return Err(whereat::at!(ConvertError::FeatureRequired {
                feature: "hdr-transfers",
                from,
                to,
            }));
            #[cfg(feature = "hdr-transfers")]
            {
                if options.hlg_rendering().is_none() {
                    return Err(whereat::at!(ConvertError::HlgRenderingRequired));
                }
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
        }

        #[cfg(not(feature = "perceptual"))]
        if from != to
            && (crate::convert::is_perceptual_model(from.color_model())
                || crate::convert::is_perceptual_model(to.color_model()))
        {
            return Err(whereat::at!(ConvertError::FeatureRequired {
                feature: "perceptual",
                from,
                to,
            }));
        }

        if crate::convert::needs_cms_for_color_model(&from, &to) {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }

        let pq_boundary = from.transfer() != to.transfer()
            && (from.transfer() == TransferFunction::Pq || to.transfer() == TransferFunction::Pq);
        #[cfg(not(feature = "hdr-transfers"))]
        if pq_boundary {
            return Err(whereat::at!(ConvertError::FeatureRequired {
                feature: "hdr-transfers",
                from,
                to,
            }));
        }
        if pq_boundary
            && !is_hdr_to_sdr(from.transfer(), to.transfer())
            && options.sdr_white().is_none()
        {
            return Err(whereat::at!(ConvertError::SdrWhiteRequired));
        }

        crate::convert::validate_descriptors(from, to)?;

        #[cfg(not(feature = "oklab"))]
        if from != to
            && (matches!(from.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA)
                || matches!(to.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA))
        {
            return Err(whereat::at!(ConvertError::FeatureRequired {
                feature: "oklab",
                from,
                to,
            }));
        }

        if !is_supported_ct(from.channel_type())
            || !is_supported_ct(to.channel_type())
            || !is_native_layout(from.layout())
            || !is_native_layout(to.layout())
        {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        let from_oklab = matches!(from.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA);
        let to_oklab = matches!(to.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA);
        if (from_oklab && from.channel_type() != ChannelType::F32)
            || (to_oklab && to.channel_type() != ChannelType::F32)
        {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        // Preserve-samples (`exact`) admission gate: reversible layout/depth widening only.
        if exact {
            let same_depth = from.channel_type() == to.channel_type();
            let exact_widen =
                from.channel_type() == ChannelType::U8 && to.channel_type() == ChannelType::U16;
            let drops_alpha = from.has_alpha() && !to.has_alpha();
            let changes_alpha_mode = from.has_alpha() && from.alpha() != to.alpha();
            let rgb_to_gray =
                from.color_model() == ColorModel::Rgb && to.color_model() == ColorModel::Gray;
            let undef_to_live = from.alpha() == Some(AlphaMode::Undefined) && to.has_alpha();
            if from.transfer() != to.transfer()
                || from.primaries != to.primaries
                || from.signal_range != to.signal_range
                || !(same_depth || exact_widen)
                || drops_alpha
                || changes_alpha_mode
                || rgb_to_gray
                || undef_to_live
                || from_oklab
                || to_oklab
            {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
        }

        // Alpha policy checks.
        let drops_alpha = from.has_alpha() && !to.has_alpha();
        if drops_alpha
            && from.signal_range == SignalRange::Narrow
            && matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. })
        {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }
        if drops_alpha && options.alpha_policy() == AlphaPolicy::DiscardIfOpaque {
            return Err(whereat::at!(ConvertError::AlphaCheckRequired));
        }
        if drops_alpha && options.alpha_policy() == AlphaPolicy::Forbid {
            return Err(whereat::at!(ConvertError::AlphaRemovalForbidden));
        }

        // Depth reduction policy.
        let reduces_depth = crate::route::channel_bits(from.channel_type())
            > crate::route::channel_bits(to.channel_type());
        if reduces_depth && options.depth_policy() == DepthPolicy::Forbid {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }

        crate::convert::validate_luma_policy(from, to, options)?;

        // HDR→SDR tone-mapping policy gates (never silently clip HDR→SDR).
        if is_hdr_to_sdr(from.transfer(), to.transfer()) {
            match options.hdr_peak() {
                zenpixels::hdr::HdrPeak::Nits(_) => {}
                _ => return Err(whereat::at!(ConvertError::HdrPeakRequired)),
            }
            if options.sdr_peak().is_none() {
                return Err(whereat::at!(ConvertError::SdrPeakRequired));
            }
            #[cfg(not(feature = "bt2446a"))]
            return Err(whereat::at!(ConvertError::HdrFeatureRequired));
            #[cfg(feature = "bt2446a")]
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        // Unknown-primaries gamut change: only named primaries have built-in matrices.
        // Gray has no chromaticity, so gray sources skip gamut conversion.
        let primaries_differ = !from_oklab
            && !to_oklab
            && from.primaries != to.primaries
            && from.primaries != ColorPrimaries::Unknown
            && to.primaries != ColorPrimaries::Unknown
            && !matches!(
                from.layout(),
                ChannelLayout::Gray | ChannelLayout::GrayAlpha
            );
        let gamut_matrix = if primaries_differ {
            let Some(m) = from.primaries.gamut_matrix_to(to.primaries) else {
                return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
            };
            Some(m)
        } else {
            None
        };
        if (from.primaries == ColorPrimaries::Unknown || to.primaries == ColorPrimaries::Unknown)
            && from.primaries != to.primaries
        {
            // Unknown primaries match any primaries without gamut conversion in zenpixels
            // unless Oklab needs concrete primaries.
        }
        #[cfg(not(feature = "hdr-transfers"))]
        if primaries_differ
            && (matches!(
                from.transfer(),
                TransferFunction::Pq | TransferFunction::Hlg
            ) || matches!(to.transfer(), TransferFunction::Pq | TransferFunction::Hlg))
        {
            return Err(whereat::at!(ConvertError::FeatureRequired {
                feature: "hdr-transfers",
                from,
                to,
            }));
        }

        // Oklab is CIE XYZ D65-anchored: Oklab↔Oklab needs no color transform,
        // and RGB↔Oklab always resolves its RGB↔LMS matrix from the RGB endpoint.
        let (from_oklab, to_oklab) = if from_oklab && to_oklab {
            (false, false)
        } else {
            (from_oklab, to_oklab)
        };
        #[cfg(feature = "oklab")]
        let (oklab_in_matrix, oklab_out_matrix) = {
            let rgb_primaries = |rgb_side: PixelDescriptor| {
                if matches!(
                    rgb_side.layout(),
                    ChannelLayout::Gray | ChannelLayout::GrayAlpha
                ) && rgb_side.primaries == ColorPrimaries::Unknown
                {
                    ColorPrimaries::Bt709
                } else {
                    rgb_side.primaries
                }
            };
            let in_m = if from_oklab {
                let Some(sp) = crate::oklab::OklabSpace::for_primaries(rgb_primaries(to)) else {
                    return Err(whereat::at!(ConvertError::NoPath { from, to }));
                };
                *sp.lms_to_rgb().rows()
            } else {
                [[0.0; 3]; 3]
            };
            let out_m = if to_oklab {
                let Some(sp) = crate::oklab::OklabSpace::for_primaries(rgb_primaries(from)) else {
                    return Err(whereat::at!(ConvertError::NoPath { from, to }));
                };
                *sp.rgb_to_lms().rows()
            } else {
                [[0.0; 3]; 3]
            };
            (in_m, out_m)
        };

        // Resolve matte background in source linear light.
        let clip = options.gamut_policy() == GamutPolicy::Clip;
        let matte_bg_linear = if drops_alpha
            && let AlphaPolicy::CompositeOnto {
                background,
                reference_white,
            } = options.alpha_policy()
        {
            let bg_d = background.descriptor();
            let hdr_bg = matches!(
                bg_d.transfer(),
                TransferFunction::Pq | TransferFunction::Hlg
            );
            if (bg_d.has_alpha() && bg_d.alpha() != Some(AlphaMode::Opaque))
                || (hdr_bg && bg_d.transfer() != from.transfer())
            {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
            #[cfg(not(feature = "hdr-transfers"))]
            if hdr_bg
                || matches!(
                    from.transfer(),
                    TransferFunction::Pq | TransferFunction::Hlg
                )
            {
                return Err(whereat::at!(ConvertError::FeatureRequired {
                    feature: "hdr-transfers",
                    from,
                    to,
                }));
            }
            if !is_supported_ct(bg_d.channel_type()) || !is_native_layout(bg_d.layout()) {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
            let bg_bytes = background.as_bytes();
            let mut bg = unpack_pixel(bg_bytes, bg_d);
            if bg_d.channel_type() == ChannelType::U8
                && bg_d.transfer() == TransferFunction::Srgb
                && !matches!(bg_d.layout(), ChannelLayout::Bgr | ChannelLayout::Bgra)
            {
                match bg_d.layout() {
                    ChannelLayout::Gray | ChannelLayout::GrayAlpha => {
                        let l = linear_srgb::default::srgb_u8_to_linear(bg_bytes[0]);
                        bg[0] = l;
                        bg[1] = l;
                        bg[2] = l;
                    }
                    _ => {
                        bg[0] = linear_srgb::default::srgb_u8_to_linear(bg_bytes[0]);
                        bg[1] = linear_srgb::default::srgb_u8_to_linear(bg_bytes[1]);
                        bg[2] = linear_srgb::default::srgb_u8_to_linear(bg_bytes[2]);
                    }
                }
            } else {
                bg[0] = scalar_eotf(bg_d.transfer(), bg[0], clip);
                bg[1] = scalar_eotf(bg_d.transfer(), bg[1], clip);
                bg[2] = scalar_eotf(bg_d.transfer(), bg[2], clip);
            }
            let mut rgb = [bg[0], bg[1], bg[2]];
            if bg_d.primaries != from.primaries
                && bg_d.primaries != ColorPrimaries::Unknown
                && from.primaries != ColorPrimaries::Unknown
            {
                let Some(m) = bg_d.primaries.gamut_matrix_to(from.primaries) else {
                    return Err(whereat::at!(ConvertError::NoPath { from, to }));
                };
                rgb = mul3_fma(&m, rgb);
            }
            if from.transfer() == TransferFunction::Pq && bg_d.transfer() != TransferFunction::Pq {
                let nits = reference_white
                    .unwrap_or(zenpixels::hdr::DiffuseWhite::BT2408)
                    .nits();
                let s = nits / 10_000.0;
                rgb = [rgb[0] * s, rgb[1] * s, rgb[2] * s];
            }
            Some(rgb)
        } else {
            None
        };

        let (pq_scale_in, pq_scale_out, pq_anchor_scale) = if pq_boundary {
            let anchor = options
                .sdr_white()
                .ok_or_else(|| whereat::at!(ConvertError::SdrWhiteRequired))?
                .nits();
            let scale = anchor / 10_000.0;
            if from.transfer() == TransferFunction::Pq {
                (1.0 / scale, 1.0, scale)
            } else {
                (1.0, scale, scale)
            }
        } else {
            (1.0, 1.0, 1.0)
        };

        let tf_differs = from.transfer() != to.transfer()
            && from.transfer() != TransferFunction::Unknown
            && to.transfer() != TransferFunction::Unknown;
        let need_linear = (tf_differs && !from_oklab && !to_oklab)
            || gamut_matrix.is_some()
            || matte_bg_linear.is_some()
            || from_oklab
            || to_oklab;

        let luma = options
            .luma()
            .unwrap_or(LumaCoefficients::Bt709)
            .coefficients();

        // Semantic effects & accuracy contract for RouteRank / introspection.
        let mut semantic = SemanticEffects::NONE;
        if drops_alpha {
            if matte_bg_linear.is_some() {
                semantic = semantic.union(SemanticEffects::MATTE_APPLIED);
            } else {
                semantic = semantic.union(SemanticEffects::ALPHA_DISCARDED);
            }
        }
        if reduces_depth {
            semantic = semantic.union(SemanticEffects::DEPTH_NARROWED);
        }
        if from.color_model() == ColorModel::Rgb && to.color_model() == ColorModel::Gray {
            semantic = semantic.union(SemanticEffects::LUMA_COLLAPSED);
        }
        if from.has_alpha() && to.has_alpha() && from.alpha() != to.alpha() {
            semantic = semantic.union(SemanticEffects::ALPHA_REASSOCIATED);
        }
        if !to.channel_type().is_float()
            && (from.channel_type().is_float()
                || gamut_matrix.is_some()
                || (clip && from.transfer() != to.transfer()))
        {
            semantic = semantic.union(SemanticEffects::CLAMPED);
        }

        let accuracy = if from == to
            || (!need_linear
                && !reduces_depth
                && !drops_alpha
                && from.color_model() == to.color_model()
                && (!from.has_alpha() || from.alpha() == to.alpha())
                && (from.channel_type() == to.channel_type()
                    || (from.channel_type() == ChannelType::U8
                        && to.channel_type() == ChannelType::U16)))
        {
            AccuracyContract::Exact
        } else if matte_bg_linear.is_some()
            || drops_alpha
            || (from.color_model() == ColorModel::Rgb && to.color_model() == ColorModel::Gray)
        {
            AccuracyContract::Rendered
        } else if need_linear {
            AccuracyContract::Computed
        } else {
            AccuracyContract::Quantized
        };

        let job = ScalarJob {
            from,
            to,
            need_linear,
            gamut_matrix,
            matte_bg_linear,
            pq_scale_in,
            pq_scale_out,
            luma,
            clip,
            truncate: options.depth_policy() == DepthPolicy::Truncate,
            from_oklab,
            to_oklab,
            #[cfg(feature = "oklab")]
            oklab_in_matrix,
            #[cfg(feature = "oklab")]
            oklab_out_matrix,
        };

        let mut plan = Self {
            job,
            input_facts: FloatFacts::NONE,
            output_facts: FloatFacts::NONE,
            pq_anchor_scale,
            semantic,
            accuracy,
        };
        plan.propagate_float_facts(FloatFacts::NONE);
        Ok(plan)
    }

    /// Propagate proven [`FloatFacts`] from input to output (matching `NativePlan`).
    pub(crate) fn propagate_float_facts(&mut self, initial: FloatFacts) {
        let from = self.job.from;
        let to = self.job.to;
        let initial = if from.channel_type().is_float() {
            initial
        } else {
            FloatFacts::NONE
        };
        self.input_facts = initial;
        if !to.channel_type().is_float() {
            self.output_facts = FloatFacts::NONE;
            return;
        }
        let unit_normal = FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE;
        let opaque_one = unit_normal | FloatFacts::POSITIVE;

        // Track `(color_facts, alpha_facts, color_min_exp, color_bounded)` through
        // each logical stage, mirroring `convert.rs::step_output_trust`.
        let mut cf: FloatFacts;
        let mut af: FloatFacts;
        let mut min_exp: Option<i16>;
        let mut bounded: bool;

        let alpha_defined = from.has_alpha() && from.alpha() != Some(AlphaMode::Undefined);
        match from.channel_type() {
            ChannelType::U8 => {
                cf = unit_normal;
                af = if alpha_defined {
                    unit_normal
                } else {
                    opaque_one
                };
                min_exp = Some(-12);
                bounded = true;
            }
            ChannelType::U16 => {
                cf = unit_normal;
                af = if alpha_defined {
                    unit_normal
                } else {
                    opaque_one
                };
                min_exp = Some(-34);
                bounded = true;
            }
            ChannelType::F16 => {
                if to.channel_type() == ChannelType::F32 {
                    cf = initial | FloatFacts::NO_SUBNORMALS;
                    af = if alpha_defined {
                        initial | FloatFacts::NO_SUBNORMALS
                    } else {
                        opaque_one
                    };
                    min_exp = Some(if initial.contains(FloatFacts::NO_SUBNORMALS) {
                        -14
                    } else {
                        -24
                    });
                    bounded = initial.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE);
                } else {
                    cf = initial;
                    af = if alpha_defined { initial } else { opaque_one };
                    min_exp = if initial.contains(FloatFacts::NO_SUBNORMALS) {
                        Some(-14)
                    } else {
                        Some(-24)
                    };
                    bounded = initial.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE);
                }
            }
            ChannelType::F32 => {
                cf = initial;
                af = if alpha_defined { initial } else { opaque_one };
                min_exp = if initial.contains(FloatFacts::NO_SUBNORMALS) {
                    Some(-126)
                } else {
                    None
                };
                bounded = initial.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE);
            }
            _ => {
                self.output_facts = FloatFacts::NONE;
                return;
            }
        }

        // Undefined source alpha upgraded to live alpha carries no proven alpha facts
        // when passthrough layout preserves undefined bytes; otherwise `unpack_pixel`
        // synthesizes 1.0 (opaque_one), matching NativePlan.
        if from.alpha() == Some(AlphaMode::Undefined)
            && to.has_alpha()
            && to.alpha() != Some(AlphaMode::Undefined)
            && from.channel_type() == to.channel_type()
            && from.layout() == to.layout()
        {
            self.output_facts = FloatFacts::NONE;
            return;
        }

        let from_premul = from.alpha() == Some(AlphaMode::Premultiplied);
        let to_premul = to.alpha() == Some(AlphaMode::Premultiplied);
        let from_linear = from.transfer() == TransferFunction::Linear;
        let stay_premul_for_matte =
            from_premul && from_linear && self.job.matte_bg_linear.is_some();
        let stay_premul_passthrough = from_premul && to_premul && !self.job.need_linear;
        let mut is_premul = from_premul;

        // 1. Unpremultiply (division by alpha drops all color bounds).
        if is_premul && !stay_premul_for_matte && !stay_premul_passthrough {
            cf = FloatFacts::NONE;
            min_exp = None;
            bounded = false;
            is_premul = false;
        }

        // 2. Oklab -> linear RGB.
        if self.job.from_oklab {
            cf = FloatFacts::NONE;
            min_exp = None;
            bounded = false;
        }

        // 3. EOTF (source -> linear).
        if self.job.need_linear && !self.job.from_oklab {
            let tf = from.transfer();
            match tf {
                TransferFunction::Linear | TransferFunction::Unknown => {}
                TransferFunction::Srgb if self.job.clip => {
                    if !cf.contains(FloatFacts::NO_NAN) {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    } else {
                        let mut out_cf =
                            FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE;
                        let mut next_min = None;
                        if cf.contains(FloatFacts::NON_NEGATIVE) {
                            out_cf |= FloatFacts::NON_NEGATIVE;
                            if let Some(k) = min_exp {
                                let nk = k.saturating_sub(4);
                                if nk >= -126 {
                                    out_cf |= FloatFacts::NO_SUBNORMALS;
                                    next_min = Some(nk);
                                    if cf.contains(FloatFacts::POSITIVE) {
                                        out_cf |= FloatFacts::POSITIVE;
                                    }
                                }
                            }
                        }
                        cf = out_cf;
                        min_exp = next_min;
                        bounded = true;
                    }
                }
                TransferFunction::Gamma22 if self.job.clip => {
                    if !cf.contains(FloatFacts::NO_NAN) {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    } else {
                        let mut out_cf =
                            FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE;
                        let mut next_min = None;
                        if cf.contains(FloatFacts::NON_NEGATIVE) {
                            out_cf |= FloatFacts::NON_NEGATIVE;
                            if let Some(k) = min_exp {
                                let nk = (k.saturating_mul(9)) / 4 - 2;
                                if nk >= -126 {
                                    out_cf |= FloatFacts::NO_SUBNORMALS;
                                    next_min = Some(nk);
                                    if cf.contains(FloatFacts::POSITIVE) {
                                        out_cf |= FloatFacts::POSITIVE;
                                    }
                                }
                            }
                        }
                        cf = out_cf;
                        min_exp = next_min;
                        bounded = true;
                    }
                }
                TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22 => {
                    if cf.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
                        let mut out_cf = FloatFacts::NO_NAN
                            | FloatFacts::NO_INFINITE
                            | FloatFacts::NON_NEGATIVE
                            | FloatFacts::AT_MOST_ONE;
                        let mut next_min = None;
                        if let Some(k) = min_exp {
                            let nk = match tf {
                                TransferFunction::Bt709 => k.saturating_sub(3),
                                TransferFunction::Srgb => k.saturating_sub(4),
                                _ => (k.saturating_mul(9)) / 4 - 2,
                            };
                            if nk >= -126 {
                                out_cf |= FloatFacts::NO_SUBNORMALS;
                                next_min = Some(nk);
                                if cf.contains(FloatFacts::POSITIVE) {
                                    out_cf |= FloatFacts::POSITIVE;
                                }
                            }
                        }
                        cf = out_cf;
                        min_exp = next_min;
                        bounded = true;
                    } else if tf == TransferFunction::Gamma22 {
                        let mut out_cf = FloatFacts::NONE;
                        if cf.contains(FloatFacts::NO_NAN) {
                            out_cf |= FloatFacts::NO_NAN;
                            if cf.contains(FloatFacts::NON_NEGATIVE) {
                                out_cf |= FloatFacts::NON_NEGATIVE;
                            }
                        }
                        cf = out_cf;
                        min_exp = None;
                        bounded = false;
                    } else {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    }
                }
                #[cfg(feature = "hdr-transfers")]
                TransferFunction::Pq => {
                    let s = self.pq_anchor_scale;
                    if s.is_finite() && s > 0.0 && (1.0 / s).is_finite() && (1.0 / s) <= 1e20 {
                        let mut out_cf = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE;
                        if s >= 1.0 {
                            out_cf |= FloatFacts::AT_MOST_ONE;
                        }
                        if matches!(from.channel_type(), ChannelType::U8 | ChannelType::U16) {
                            out_cf |= FloatFacts::NO_SUBNORMALS | FloatFacts::NON_NEGATIVE;
                            min_exp = Some(-64);
                        } else {
                            if cf.contains(FloatFacts::NON_NEGATIVE) {
                                out_cf |= FloatFacts::NON_NEGATIVE;
                            }
                            min_exp = None;
                        }
                        cf = out_cf;
                        bounded = true;
                    } else {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    }
                }
                #[cfg(feature = "hdr-transfers")]
                TransferFunction::Hlg
                    if cf.contains(
                        FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE,
                    ) =>
                {
                    let mut out_cf = FloatFacts::NO_NAN
                        | FloatFacts::NO_INFINITE
                        | FloatFacts::NON_NEGATIVE
                        | FloatFacts::AT_MOST_ONE;
                    let mut next_min = None;
                    if cf.contains(FloatFacts::NON_NEGATIVE)
                        && let Some(k) = min_exp
                    {
                        let nk = k.saturating_mul(2).saturating_sub(2);
                        if nk >= -126 {
                            out_cf |= FloatFacts::NO_SUBNORMALS;
                            next_min = Some(nk);
                            if cf.contains(FloatFacts::POSITIVE) {
                                out_cf |= FloatFacts::POSITIVE;
                            }
                        }
                    }
                    cf = out_cf;
                    min_exp = next_min;
                    bounded = true;
                }
                _ => {
                    cf = FloatFacts::NONE;
                    min_exp = None;
                    bounded = false;
                }
            }
        }

        // 4. Matte composite (conservative unless bounded; matches NativePlan default).
        if self.job.matte_bg_linear.is_some() {
            cf = FloatFacts::NONE;
            af = opaque_one;
            min_exp = None;
            bounded = false;
            is_premul = false;
        }

        // 5. Gamut matrix.
        if let Some(m) = &self.job.gamut_matrix {
            if !cf.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE) {
                cf = FloatFacts::NONE;
                min_exp = None;
                bounded = false;
            } else {
                let mut out_cf = FloatFacts::NO_NAN;
                let mut next_bounded = false;
                if bounded {
                    out_cf |= FloatFacts::NO_INFINITE;
                    next_bounded = true;
                }
                let mut next_min = None;
                let all_pos = m.iter().flatten().all(|&c| c >= (1.0 / 64.0));
                if all_pos && cf.contains(FloatFacts::NON_NEGATIVE) {
                    out_cf |= FloatFacts::NON_NEGATIVE;
                    if let Some(k) = min_exp
                        && k - 6 >= -126
                    {
                        out_cf |= FloatFacts::NO_SUBNORMALS;
                        next_min = Some(k - 6);
                    }
                }
                cf = out_cf;
                min_exp = next_min;
                bounded = next_bounded;
            }
        }

        // 6. Linear RGB -> Oklab.
        if self.job.to_oklab {
            if cf.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE) && bounded {
                cf = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE;
                min_exp = None;
                bounded = true;
            } else {
                cf = FloatFacts::NONE;
                min_exp = None;
                bounded = false;
            }
        }

        // 7. OETF (linear -> destination).
        if self.job.need_linear && !self.job.to_oklab {
            let tf = to.transfer();
            match tf {
                TransferFunction::Linear | TransferFunction::Unknown => {}
                TransferFunction::Srgb | TransferFunction::Gamma22 if self.job.clip => {
                    if !cf.contains(FloatFacts::NO_NAN) {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    } else {
                        let mut out_cf =
                            FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE;
                        let mut next_min = None;
                        if cf.contains(FloatFacts::NON_NEGATIVE) {
                            out_cf |= FloatFacts::NON_NEGATIVE;
                            if let Some(k) = min_exp {
                                let nk = k.min(0);
                                if nk >= -126 {
                                    out_cf |= FloatFacts::NO_SUBNORMALS;
                                    next_min = Some(nk);
                                    if cf.contains(FloatFacts::POSITIVE) {
                                        out_cf |= FloatFacts::POSITIVE;
                                    }
                                }
                            }
                        }
                        cf = out_cf;
                        min_exp = next_min;
                        bounded = true;
                    }
                }
                TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22 => {
                    if cf.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
                        let mut out_cf = FloatFacts::NO_NAN
                            | FloatFacts::NO_INFINITE
                            | FloatFacts::NON_NEGATIVE
                            | FloatFacts::AT_MOST_ONE;
                        let mut next_min = None;
                        if let Some(k) = min_exp {
                            let nk = k.min(0);
                            if nk >= -126 {
                                out_cf |= FloatFacts::NO_SUBNORMALS;
                                next_min = Some(nk);
                                if cf.contains(FloatFacts::POSITIVE) {
                                    out_cf |= FloatFacts::POSITIVE;
                                }
                            }
                        }
                        cf = out_cf;
                        min_exp = next_min;
                        bounded = true;
                    } else if tf == TransferFunction::Gamma22 {
                        let mut out_cf = FloatFacts::NONE;
                        if cf.contains(FloatFacts::NO_NAN) {
                            out_cf |= FloatFacts::NO_NAN;
                            if cf.contains(FloatFacts::NON_NEGATIVE) {
                                out_cf |= FloatFacts::NON_NEGATIVE;
                            }
                            if cf.contains(FloatFacts::NO_INFINITE) {
                                out_cf |= FloatFacts::NO_INFINITE;
                            }
                        }
                        bounded = bounded && out_cf.contains(FloatFacts::NO_INFINITE);
                        cf = out_cf;
                        min_exp = None;
                    } else {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    }
                }
                #[cfg(feature = "hdr-transfers")]
                TransferFunction::Pq => {
                    let s = self.pq_anchor_scale;
                    if s.is_finite() && s >= 0.0 {
                        let mut out_cf =
                            FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE;
                        if cf.contains(FloatFacts::NON_NEGATIVE) {
                            out_cf |= FloatFacts::NON_NEGATIVE;
                        }
                        cf = out_cf;
                        min_exp = None;
                        bounded = true;
                    } else {
                        cf = FloatFacts::NONE;
                        min_exp = None;
                        bounded = false;
                    }
                }
                #[cfg(feature = "hdr-transfers")]
                TransferFunction::Hlg
                    if cf.contains(
                        FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE,
                    ) =>
                {
                    let mut out_cf = FloatFacts::NO_NAN
                        | FloatFacts::NO_INFINITE
                        | FloatFacts::NON_NEGATIVE
                        | FloatFacts::AT_MOST_ONE;
                    let mut next_min = None;
                    if cf.contains(FloatFacts::NON_NEGATIVE)
                        && let Some(k) = min_exp
                    {
                        let nk = k.min(0);
                        if nk >= -126 {
                            out_cf |= FloatFacts::NO_SUBNORMALS;
                            next_min = Some(nk);
                            if cf.contains(FloatFacts::POSITIVE) {
                                out_cf |= FloatFacts::POSITIVE;
                            }
                        }
                    }
                    cf = out_cf;
                    min_exp = next_min;
                    bounded = true;
                }
                _ => {
                    cf = FloatFacts::NONE;
                    min_exp = None;
                    bounded = false;
                }
            }
        }

        // 8. RGB -> Gray luma collapse.
        if from.color_model() == ColorModel::Rgb && to.color_model() == ColorModel::Gray {
            let mut out_cf = FloatFacts::NONE;
            if cf.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE) && bounded {
                out_cf |= FloatFacts::NO_NAN | FloatFacts::NO_INFINITE;
            }
            if cf.contains(FloatFacts::NON_NEGATIVE) {
                out_cf |= FloatFacts::NON_NEGATIVE;
                if cf.contains(FloatFacts::AT_MOST_ONE) {
                    out_cf |= FloatFacts::AT_MOST_ONE;
                }
                if to.channel_type() == ChannelType::F32
                    && let Some(k) = min_exp
                    && k - 4 >= -126
                {
                    out_cf |= FloatFacts::NO_SUBNORMALS;
                    if cf.contains(FloatFacts::POSITIVE) && out_cf.contains(FloatFacts::NO_INFINITE)
                    {
                        out_cf |= FloatFacts::POSITIVE;
                    }
                }
            }
            cf = out_cf;
        }

        // 9. Straight -> Premultiplied alpha.
        if to_premul && !is_premul {
            let both = cf & af;
            if both.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
                cf = FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE;
                af = cf;
            } else {
                cf = FloatFacts::NONE;
                af = FloatFacts::NONE;
            }
        }

        // 10. Combine color + alpha facts, then filter for F16 narrowing if needed.
        let mut combined = if to.has_alpha() && to.alpha() != Some(AlphaMode::Undefined) {
            cf & af
        } else {
            cf
        };
        if to.channel_type() == ChannelType::F16 && from.channel_type() != ChannelType::F16 {
            combined &= FloatFacts::NO_NAN | FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE;
        }
        self.output_facts = combined;
    }

    /// Execute one row of `width` pixels in-place with zero heap allocation.
    #[inline(never)]
    pub(crate) fn execute(&self, src: &[u8], dst: &mut [u8], width: u32) {
        let width = width as usize;
        let job = &self.job;
        let sbpp = job.from.bytes_per_pixel();
        let dbpp = job.to.bytes_per_pixel();

        if job.from == job.to {
            let n = width * sbpp;
            dst[..n].copy_from_slice(&src[..n]);
            return;
        }

        let from_premul = job.from.alpha() == Some(AlphaMode::Premultiplied);
        let to_premul = job.to.alpha() == Some(AlphaMode::Premultiplied);
        let from_linear = job.from.transfer() == TransferFunction::Linear;
        let no_color_or_matte = !job.need_linear
            && !from_premul
            && !to_premul
            && !job.from_oklab
            && !job.to_oklab
            && job.matte_bg_linear.is_none();

        // Exact U8 <-> U16 integer fast paths (zero float drift)
        if no_color_or_matte && job.from.layout() == job.to.layout() {
            if job.from.channel_type() == ChannelType::U16
                && job.to.channel_type() == ChannelType::U8
            {
                let count = width * job.from.channels();
                for i in 0..count {
                    let v = u32::from(u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]));
                    dst[i] = if job.truncate {
                        (v >> 8) as u8
                    } else {
                        ((v * 255 + 32895) >> 16) as u8
                    };
                }
                return;
            }
            if job.from.channel_type() == ChannelType::U8
                && job.to.channel_type() == ChannelType::U16
            {
                let count = width * job.from.channels();
                for i in 0..count {
                    let v = u16::from(src[i]) * 257;
                    dst[i * 2..i * 2 + 2].copy_from_slice(&v.to_ne_bytes());
                }
                return;
            }
        }

        // Exact U16 -> U8 SDR transfer (`Linear`/`Srgb`/`Bt709`) matching `sdr_u16_to_u8`
        // across all 65,536 16-bit codes with 0 quantization error and zero LUT heap allocation.
        if job.from.channel_type() == ChannelType::U16
            && job.to.channel_type() == ChannelType::U8
            && job.from.transfer() != job.to.transfer()
            && matches!(
                job.from.transfer(),
                TransferFunction::Linear | TransferFunction::Srgb | TransferFunction::Bt709
            )
            && matches!(
                job.to.transfer(),
                TransferFunction::Linear | TransferFunction::Srgb | TransferFunction::Bt709
            )
            && job.gamut_matrix.is_none()
            && job.matte_bg_linear.is_none()
            && !job.truncate
            && !job.from_oklab
            && !job.to_oklab
            && !(job.from.color_model() == ColorModel::Rgb
                && job.to.color_model() == ColorModel::Gray)
        {
            const BT709_BETA_F64: f64 = 0.018053968510807;
            const BT709_ALPHA_F64: f64 = 0.09929682680944;
            let from_tf = job.from.transfer();
            let to_tf = job.to.transfer();
            let xfer = |code: u16| -> u8 {
                let x = f64::from(code) / 65535.0;
                let lin = match from_tf {
                    TransferFunction::Linear => x,
                    TransferFunction::Srgb => {
                        if x <= 0.04045 {
                            x / 12.92
                        } else {
                            libm::pow((x + 0.055) / 1.055, 2.4)
                        }
                    }
                    TransferFunction::Bt709 => {
                        if x < 4.5 * BT709_BETA_F64 {
                            x / 4.5
                        } else {
                            libm::pow((x + BT709_ALPHA_F64) / (1.0 + BT709_ALPHA_F64), 1.0 / 0.45)
                        }
                    }
                    _ => x,
                };
                let enc = match to_tf {
                    TransferFunction::Linear => lin,
                    TransferFunction::Srgb => {
                        if lin <= 0.0031308 {
                            lin * 12.92
                        } else {
                            1.055 * libm::pow(lin, 1.0 / 2.4) - 0.055
                        }
                    }
                    TransferFunction::Bt709 => {
                        if lin < BT709_BETA_F64 {
                            lin * 4.5
                        } else {
                            (1.0 + BT709_ALPHA_F64) * libm::pow(lin, 0.45) - BT709_ALPHA_F64
                        }
                    }
                    _ => lin,
                };
                libm::floor(enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            };
            let read_u16 =
                |s: &[u8], idx: usize| -> u16 { u16::from_ne_bytes([s[idx * 2], s[idx * 2 + 1]]) };
            for x in 0..width {
                let s = &src[x * sbpp..(x + 1) * sbpp];
                let d = &mut dst[x * dbpp..(x + 1) * dbpp];
                let (mut r16, mut g16, mut b16, a16) = match job.from.layout() {
                    ChannelLayout::Gray => {
                        let g = read_u16(s, 0);
                        (g, g, g, 65535u16)
                    }
                    ChannelLayout::GrayAlpha => {
                        let g = read_u16(s, 0);
                        (g, g, g, read_u16(s, 1))
                    }
                    ChannelLayout::Rgb => (read_u16(s, 0), read_u16(s, 1), read_u16(s, 2), 65535),
                    ChannelLayout::Bgr => (read_u16(s, 2), read_u16(s, 1), read_u16(s, 0), 65535),
                    ChannelLayout::Rgba => {
                        let a = if job.from.alpha() == Some(AlphaMode::Undefined) {
                            65535
                        } else {
                            read_u16(s, 3)
                        };
                        (read_u16(s, 0), read_u16(s, 1), read_u16(s, 2), a)
                    }
                    ChannelLayout::Bgra => {
                        let a = if job.from.alpha() == Some(AlphaMode::Undefined) {
                            65535
                        } else {
                            read_u16(s, 3)
                        };
                        (read_u16(s, 2), read_u16(s, 1), read_u16(s, 0), a)
                    }
                    _ => (0, 0, 0, 65535),
                };
                if from_premul {
                    if a16 == 0 {
                        r16 = 0;
                        g16 = 0;
                        b16 = 0;
                    } else {
                        let a32 = u32::from(a16);
                        r16 = ((u32::from(r16) * 65535 + a32 / 2) / a32).min(65535) as u16;
                        g16 = ((u32::from(g16) * 65535 + a32 / 2) / a32).min(65535) as u16;
                        b16 = ((u32::from(b16) * 65535 + a32 / 2) / a32).min(65535) as u16;
                    }
                }
                let mut r8 = xfer(r16);
                let mut g8 = xfer(g16);
                let mut b8 = xfer(b16);
                let a8 = ((u32::from(a16) * 255 + 32767) / 65535) as u8;
                if to_premul {
                    let a32 = u32::from(a8);
                    let pm = |c: u8| -> u8 {
                        let t = u32::from(c) * a32 + 128;
                        ((t + (t >> 8)) >> 8) as u8
                    };
                    r8 = pm(r8);
                    g8 = pm(g8);
                    b8 = pm(b8);
                }
                let oa = if job.to.alpha() == Some(AlphaMode::Undefined) {
                    255
                } else {
                    a8
                };
                match job.to.layout() {
                    ChannelLayout::Gray => d[0] = r8,
                    ChannelLayout::GrayAlpha => {
                        d[0] = r8;
                        d[1] = oa;
                    }
                    ChannelLayout::Rgb => {
                        d[0] = r8;
                        d[1] = g8;
                        d[2] = b8;
                    }
                    ChannelLayout::Bgr => {
                        d[0] = b8;
                        d[1] = g8;
                        d[2] = r8;
                    }
                    ChannelLayout::Rgba => {
                        d[0] = r8;
                        d[1] = g8;
                        d[2] = b8;
                        d[3] = oa;
                    }
                    ChannelLayout::Bgra => {
                        d[0] = b8;
                        d[1] = g8;
                        d[2] = r8;
                        d[3] = oa;
                    }
                    _ => {}
                }
            }
            return;
        }

        // Exact U8 / U16 integer straight <-> premul matching convert_kernels.rs bit-for-bit
        if !job.need_linear
            && !job.from_oklab
            && !job.to_oklab
            && job.matte_bg_linear.is_none()
            && job.from.layout() == job.to.layout()
            && from_premul != to_premul
        {
            if job.from.channel_type() == ChannelType::U8
                && job.to.channel_type() == ChannelType::U8
            {
                match job.from.layout() {
                    ChannelLayout::Rgba | ChannelLayout::Bgra => {
                        for x in 0..width {
                            let s = &src[x * 4..x * 4 + 4];
                            let d = &mut dst[x * 4..x * 4 + 4];
                            let a = s[3];
                            if to_premul {
                                for c in 0..3 {
                                    let t = u32::from(s[c]) * u32::from(a) + 128;
                                    d[c] = ((t + (t >> 8)) >> 8) as u8;
                                }
                                d[3] = a;
                            } else if a == 0 {
                                d.copy_from_slice(&[0, 0, 0, 0]);
                            } else {
                                let a32 = u32::from(a);
                                for c in 0..3 {
                                    d[c] = ((u32::from(s[c]) * 255 + a32 / 2) / a32).min(255) as u8;
                                }
                                d[3] = a;
                            }
                        }
                        return;
                    }
                    ChannelLayout::GrayAlpha => {
                        for x in 0..width {
                            let g = src[x * 2];
                            let a = src[x * 2 + 1];
                            let d = &mut dst[x * 2..x * 2 + 2];
                            if to_premul {
                                d[0] = ((u32::from(g) * u32::from(a) + 127) / 255) as u8;
                                d[1] = a;
                            } else if a == 0 {
                                d[0] = 0;
                                d[1] = 0;
                            } else {
                                let a32 = u32::from(a);
                                d[0] = ((u32::from(g) * 255 + a32 / 2) / a32).min(255) as u8;
                                d[1] = a;
                            }
                        }
                        return;
                    }
                    _ => {}
                }
            }
            if job.from.channel_type() == ChannelType::U16
                && job.to.channel_type() == ChannelType::U16
            {
                let ch = job.from.channels();
                if ch == 2 || ch == 4 {
                    for x in 0..width {
                        let base = x * ch * 2;
                        let a = u16::from_ne_bytes([
                            src[base + (ch - 1) * 2],
                            src[base + (ch - 1) * 2 + 1],
                        ]);
                        let a32 = u32::from(a);
                        for c in 0..(ch - 1) {
                            let v = u32::from(u16::from_ne_bytes([
                                src[base + c * 2],
                                src[base + c * 2 + 1],
                            ]));
                            let out = if to_premul {
                                ((v * a32 + 32767) / 65535) as u16
                            } else if a == 0 {
                                0
                            } else {
                                ((v * 65535 + a32 / 2) / a32).min(65535) as u16
                            };
                            dst[base + c * 2..base + c * 2 + 2].copy_from_slice(&out.to_ne_bytes());
                        }
                        dst[base + (ch - 1) * 2..base + ch * 2].copy_from_slice(&a.to_ne_bytes());
                    }
                    return;
                }
            }
            if job.from.channel_type() == ChannelType::F16
                && job.to.channel_type() == ChannelType::F16
            {
                let ch = job.from.channels();
                if ch == 2 || ch == 4 {
                    for x in 0..width {
                        let base = x * ch * 2;
                        let a_off = base + (ch - 1) * 2;
                        let raw_a = u16::from_ne_bytes([src[a_off], src[a_off + 1]]);
                        let a = f16_bits_to_f32(raw_a);
                        let inv_a = if !to_premul && a != 0.0 { 1.0 / a } else { 0.0 };
                        for c in 0..(ch - 1) {
                            let off = base + c * 2;
                            let v = f16_bits_to_f32(u16::from_ne_bytes([src[off], src[off + 1]]));
                            let out = if to_premul {
                                v * a
                            } else if a == 0.0 {
                                0.0
                            } else {
                                v * inv_a
                            };
                            dst[off..off + 2].copy_from_slice(&f32_to_f16_bits(out).to_ne_bytes());
                        }
                        // Alpha is a payload: preserve NaN bits and signed zero verbatim.
                        dst[a_off..a_off + 2].copy_from_slice(&src[a_off..a_off + 2]);
                    }
                    return;
                }
            }
        }

        // Exact U8 RGB/RGBA -> Gray/GrayAlpha integer formula matching convert_kernels.rs:
        // - `RgbToGray`/`RgbaToGray` (including `Rgb -> GrayA`, which plans as `RgbToGray`
        //   + `GrayToGrayAlpha`) with BT.709 uses the fixed-point `(54*r + 183*g + 19*b + 128) >> 8`.
        // - `RgbaToGrayAlpha` and non-BT.709 weights use `f32` dot + 0.5.
        if no_color_or_matte
            && job.from.channel_type() == ChannelType::U8
            && job.to.channel_type() == ChannelType::U8
            && job.from.color_model() == ColorModel::Rgb
            && job.to.color_model() == ColorModel::Gray
        {
            let w = job.luma;
            let bgr = matches!(job.from.layout(), ChannelLayout::Bgr | ChannelLayout::Bgra);
            let dst_has_alpha = job.to.layout() == ChannelLayout::GrayAlpha;
            let use_bt709_fixed = w == LumaCoefficients::Bt709.coefficients()
                && !(job.from.has_alpha() && dst_has_alpha);
            for x in 0..width {
                let s = &src[x * sbpp..(x + 1) * sbpp];
                let (r, g, b) = if bgr {
                    (s[2], s[1], s[0])
                } else {
                    (s[0], s[1], s[2])
                };
                dst[x * dbpp] = if use_bt709_fixed {
                    ((u16::from(r) * 54 + u16::from(g) * 183 + u16::from(b) * 19 + 128) >> 8) as u8
                } else {
                    let y = f32::from(r) * w[0] + f32::from(g) * w[1] + f32::from(b) * w[2];
                    (y + 0.5).clamp(0.0, 255.0) as u8
                };
                if dst_has_alpha {
                    dst[x * dbpp + 1] =
                        if sbpp == 4 && job.from.alpha() != Some(AlphaMode::Undefined) {
                            s[3]
                        } else {
                            255
                        };
                }
            }
            return;
        }

        // Exact U16 RGB/RGBA -> Gray/GrayAlpha formula matching `gray_u16_via_f32`
        // (widens raw u16 directly to f32 without `/ 65535.0`, rounds via `(y + 0.5) as u16`).
        if no_color_or_matte
            && job.from.channel_type() == ChannelType::U16
            && job.to.channel_type() == ChannelType::U16
            && job.from.color_model() == ColorModel::Rgb
            && job.to.color_model() == ColorModel::Gray
        {
            let w = job.luma;
            let bgr = matches!(job.from.layout(), ChannelLayout::Bgr | ChannelLayout::Bgra);
            let dst_has_alpha = job.to.layout() == ChannelLayout::GrayAlpha;
            let read16 = |s: &[u8], i: usize| u16::from_ne_bytes([s[i * 2], s[i * 2 + 1]]);
            for x in 0..width {
                let s = &src[x * sbpp..(x + 1) * sbpp];
                let d = &mut dst[x * dbpp..(x + 1) * dbpp];
                let (r, g, b) = if bgr {
                    (read16(s, 2), read16(s, 1), read16(s, 0))
                } else {
                    (read16(s, 0), read16(s, 1), read16(s, 2))
                };
                let y = f32::from(r) * w[0] + f32::from(g) * w[1] + f32::from(b) * w[2];
                let y16 = (y + 0.5).clamp(0.0, 65535.0) as u16;
                d[0..2].copy_from_slice(&y16.to_ne_bytes());
                if dst_has_alpha {
                    let a16 = if sbpp == 8 && job.from.alpha() != Some(AlphaMode::Undefined) {
                        read16(s, 3)
                    } else {
                        65535u16
                    };
                    d[2..4].copy_from_slice(&a16.to_ne_bytes());
                }
            }
            return;
        }

        // Exact `SrgbU16GamutRgb` parity: `fast_gamut` decodes via `linear_srgb::default::srgb_to_linear`
        // (`f64` Horner on `u * (1/65535)`) and encodes via `fma(srgb, 65535.0, 0.5) as u16`.
        if let Some(m) = &job.gamut_matrix
            && job.from.channel_type() == ChannelType::U16
            && job.to.channel_type() == ChannelType::U16
            && job.from.layout() == ChannelLayout::Rgb
            && job.to.layout() == ChannelLayout::Rgb
            && job.from.transfer() == TransferFunction::Srgb
            && job.to.transfer() == TransferFunction::Srgb
            && job.matte_bg_linear.is_none()
            && !job.truncate
        {
            let read16 = |s: &[u8], i: usize| u16::from_ne_bytes([s[i * 2], s[i * 2 + 1]]);
            for x in 0..width {
                let s = &src[x * 6..x * 6 + 6];
                let d = &mut dst[x * 6..x * 6 + 6];
                let r =
                    linear_srgb::default::srgb_to_linear(f32::from(read16(s, 0)) * (1.0 / 65535.0));
                let g =
                    linear_srgb::default::srgb_to_linear(f32::from(read16(s, 1)) * (1.0 / 65535.0));
                let b =
                    linear_srgb::default::srgb_to_linear(f32::from(read16(s, 2)) * (1.0 / 65535.0));
                let out = mul3_fma(m, [r, g, b]);
                for c in 0..3 {
                    let enc = scalar_oetf(TransferFunction::Srgb, out[c], true);
                    let q = fma_f32(enc, 65535.0, 0.5) as u16;
                    d[c * 2..c * 2 + 2].copy_from_slice(&q.to_ne_bytes());
                }
            }
            return;
        }

        // Exact same-depth U8/U16 pure swizzle / alpha add-drop / gray expand (zero float roundtrip)
        if no_color_or_matte
            && job.from.channel_type() == job.to.channel_type()
            && matches!(job.from.channel_type(), ChannelType::U8 | ChannelType::U16)
            && !(job.from.color_model() == ColorModel::Rgb
                && job.to.color_model() == ColorModel::Gray)
        {
            let cs = job.from.channel_type().byte_size();
            let opaque: [u8; 2] = [0xFF, 0xFF];
            for x in 0..width {
                let s = &src[x * sbpp..(x + 1) * sbpp];
                let d = &mut dst[x * dbpp..(x + 1) * dbpp];
                let (r, g, b, a): (&[u8], &[u8], &[u8], &[u8]) = match job.from.layout() {
                    ChannelLayout::Gray => (&s[0..cs], &s[0..cs], &s[0..cs], &opaque[..cs]),
                    ChannelLayout::GrayAlpha => (&s[0..cs], &s[0..cs], &s[0..cs], &s[cs..2 * cs]),
                    ChannelLayout::Rgb => {
                        (&s[0..cs], &s[cs..2 * cs], &s[2 * cs..3 * cs], &opaque[..cs])
                    }
                    ChannelLayout::Bgr => {
                        (&s[2 * cs..3 * cs], &s[cs..2 * cs], &s[0..cs], &opaque[..cs])
                    }
                    ChannelLayout::Rgba => {
                        let al = if job.from.alpha() == Some(AlphaMode::Undefined) {
                            &opaque[..cs]
                        } else {
                            &s[3 * cs..4 * cs]
                        };
                        (&s[0..cs], &s[cs..2 * cs], &s[2 * cs..3 * cs], al)
                    }
                    ChannelLayout::Bgra => {
                        let al = if job.from.alpha() == Some(AlphaMode::Undefined) {
                            &opaque[..cs]
                        } else {
                            &s[3 * cs..4 * cs]
                        };
                        (&s[2 * cs..3 * cs], &s[cs..2 * cs], &s[0..cs], al)
                    }
                    _ => (&opaque[..cs], &opaque[..cs], &opaque[..cs], &opaque[..cs]),
                };
                match job.to.layout() {
                    ChannelLayout::Gray => d[0..cs].copy_from_slice(r),
                    ChannelLayout::GrayAlpha => {
                        d[0..cs].copy_from_slice(r);
                        d[cs..2 * cs].copy_from_slice(a);
                    }
                    ChannelLayout::Rgb => {
                        d[0..cs].copy_from_slice(r);
                        d[cs..2 * cs].copy_from_slice(g);
                        d[2 * cs..3 * cs].copy_from_slice(b);
                    }
                    ChannelLayout::Bgr => {
                        d[0..cs].copy_from_slice(b);
                        d[cs..2 * cs].copy_from_slice(g);
                        d[2 * cs..3 * cs].copy_from_slice(r);
                    }
                    ChannelLayout::Rgba => {
                        d[0..cs].copy_from_slice(r);
                        d[cs..2 * cs].copy_from_slice(g);
                        d[2 * cs..3 * cs].copy_from_slice(b);
                        let oa = if job.to.alpha() == Some(AlphaMode::Undefined) {
                            &opaque[..cs]
                        } else {
                            a
                        };
                        d[3 * cs..4 * cs].copy_from_slice(oa);
                    }
                    ChannelLayout::Bgra => {
                        d[0..cs].copy_from_slice(b);
                        d[cs..2 * cs].copy_from_slice(g);
                        d[2 * cs..3 * cs].copy_from_slice(r);
                        let oa = if job.to.alpha() == Some(AlphaMode::Undefined) {
                            &opaque[..cs]
                        } else {
                            a
                        };
                        d[3 * cs..4 * cs].copy_from_slice(oa);
                    }
                    _ => {}
                }
            }
            return;
        }

        // General single-pass universal fast-FMA pixel loop.
        // Match `NativePlan`'s exact choice between `srgb_u8_to_linear` LUT vs
        // `NaiveU8ToF32 + SrgbF32ToLinearF32` polynomial, and between
        // `linear_to_srgb_u8` 4096-LUT vs `LinearF32ToSrgbF32 + NaiveF32ToU8`.
        let from_gray = job.from.color_model() == ColorModel::Gray;
        let is_srgb_u8_matte = job.from.channel_type() == ChannelType::U8
            && job.from.transfer() == TransferFunction::Srgb
            && !from_premul
            && job.matte_bg_linear.is_some()
            && job.from.layout() == ChannelLayout::Rgba;
        let u8_srgb_in = job.from.channel_type() == ChannelType::U8
            && job.from.transfer() == TransferFunction::Srgb
            && !from_premul
            && ((job.to.channel_type().is_float()
                && job.to.transfer() == TransferFunction::Linear)
                || is_srgb_u8_matte
                || (job.gamut_matrix.is_some()
                    && job.to.channel_type() == ChannelType::U8
                    && job.to.transfer() == TransferFunction::Srgb));
        let u8_srgb_lut_out = job.need_linear
            && job.to.channel_type() == ChannelType::U8
            && job.to.transfer() == TransferFunction::Srgb
            && !to_premul
            && !job.to_oklab
            && !job.truncate
            && job.gamut_matrix.is_none()
            && ((job.from.channel_type().is_float()
                && job.from.transfer() == TransferFunction::Linear
                && job.matte_bg_linear.is_none())
                || (is_srgb_u8_matte && job.to.channel_type() == ChannelType::U8));
        let srgb_u8_matte_alpha = is_srgb_u8_matte;

        let pq_u16_f32_alpha = job.from.channel_type().is_float()
            && job.from.transfer() == TransferFunction::Linear
            && job.to.channel_type() == ChannelType::U16
            && job.to.transfer() == TransferFunction::Pq
            && job.to.has_alpha()
            && job.to.alpha() != Some(AlphaMode::Undefined)
            && !job.truncate
            && job.gamut_matrix.is_none()
            && job.matte_bg_linear.is_none();

        let stay_premul_for_matte = from_premul && from_linear && job.matte_bg_linear.is_some();
        let stay_premul_passthrough = from_premul && to_premul && !job.need_linear;

        for x in 0..width {
            let spx = &src[x * sbpp..(x + 1) * sbpp];
            let mut px = unpack_pixel(spx, job.from);
            if srgb_u8_matte_alpha {
                // Matches `step_matte_srgb_u8` (`convert_kernels.rs:1800`) exhaustively.
                px[3] = f32::from(spx[3]) * (1.0 / 255.0);
            }

            // Unpremultiply in source space when needed, tracking live premul state
            // so we never double-premultiply or skip re-association.
            let mut is_premul = from_premul;
            if is_premul && !stay_premul_for_matte && !stay_premul_passthrough {
                let a = px[3];
                if a == 0.0 {
                    px[0] = 0.0;
                    px[1] = 0.0;
                    px[2] = 0.0;
                } else {
                    px[0] /= a;
                    px[1] /= a;
                    px[2] /= a;
                }
                is_premul = false;
            }

            #[cfg(feature = "oklab")]
            if job.from_oklab {
                let rgb = crate::oklab::oklab_to_rgb(px[0], px[1], px[2], &job.oklab_in_matrix);
                px[0] = rgb[0];
                px[1] = rgb[1];
                px[2] = rgb[2];
            }

            if job.need_linear {
                if !job.from_oklab {
                    if u8_srgb_in {
                        match job.from.layout() {
                            ChannelLayout::Gray | ChannelLayout::GrayAlpha => {
                                let l = linear_srgb::default::srgb_u8_to_linear(spx[0]);
                                px[0] = l;
                                px[1] = l;
                                px[2] = l;
                            }
                            ChannelLayout::Bgr | ChannelLayout::Bgra => {
                                px[0] = linear_srgb::default::srgb_u8_to_linear(spx[2]);
                                px[1] = linear_srgb::default::srgb_u8_to_linear(spx[1]);
                                px[2] = linear_srgb::default::srgb_u8_to_linear(spx[0]);
                            }
                            _ => {
                                px[0] = linear_srgb::default::srgb_u8_to_linear(spx[0]);
                                px[1] = linear_srgb::default::srgb_u8_to_linear(spx[1]);
                                px[2] = linear_srgb::default::srgb_u8_to_linear(spx[2]);
                            }
                        }
                    } else {
                        px[0] = scalar_eotf(job.from.transfer(), px[0], job.clip) * job.pq_scale_in;
                        px[1] = scalar_eotf(job.from.transfer(), px[1], job.clip) * job.pq_scale_in;
                        px[2] = scalar_eotf(job.from.transfer(), px[2], job.clip) * job.pq_scale_in;
                    }
                }
                if let Some(bg) = job.matte_bg_linear {
                    let raw_a = px[3];
                    let a = if raw_a >= 0.0 { raw_a.min(1.0) } else { 0.0 };
                    let inv = 1.0 - a;
                    if is_premul {
                        px[0] += bg[0] * inv;
                        px[1] += bg[1] * inv;
                        px[2] += bg[2] * inv;
                        is_premul = false;
                    } else {
                        px[0] = px[0] * a + bg[0] * inv;
                        px[1] = px[1] * a + bg[1] * inv;
                        px[2] = px[2] * a + bg[2] * inv;
                    }
                    px[3] = 1.0;
                }
                if let Some(m) = &job.gamut_matrix {
                    let r = mul3_fma(m, [px[0], px[1], px[2]]);
                    px[0] = r[0];
                    px[1] = r[1];
                    px[2] = r[2];
                }
                #[cfg(feature = "oklab")]
                if job.to_oklab {
                    let lab =
                        crate::oklab::rgb_to_oklab(px[0], px[1], px[2], &job.oklab_out_matrix);
                    px[0] = lab[0];
                    px[1] = lab[1];
                    px[2] = lab[2];
                }
                if !job.to_oklab && !u8_srgb_lut_out {
                    px[0] = scalar_oetf(job.to.transfer(), px[0] * job.pq_scale_out, job.clip);
                    px[1] = scalar_oetf(job.to.transfer(), px[1] * job.pq_scale_out, job.clip);
                    px[2] = scalar_oetf(job.to.transfer(), px[2] * job.pq_scale_out, job.clip);
                }
            }

            if to_premul && !is_premul {
                let a = px[3];
                if a == 0.0 {
                    px[0] = 0.0;
                    px[1] = 0.0;
                    px[2] = 0.0;
                } else {
                    px[0] *= a;
                    px[1] *= a;
                    px[2] *= a;
                }
            }

            let dpx = &mut dst[x * dbpp..(x + 1) * dbpp];
            pack_pixel(
                dpx,
                job.to,
                px,
                job.luma,
                from_gray,
                job.truncate,
                u8_srgb_lut_out,
            );
            if pq_u16_f32_alpha {
                // Match `restore_alpha_f32_u16` (`convert_kernels.rs:2728`) bit-for-bit.
                let a = (px[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                let off = (job.to.channels() - 1) * 2;
                dpx[off..off + 2].copy_from_slice(&a.to_ne_bytes());
            }
        }
    }
}
