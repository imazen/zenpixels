//! Fast gamut conversion between sRGB, Display P3, and BT.2020.
//!
//! Fused single-pass SIMD kernel: deinterleave → polynomial TRC → 3×3 matrix
//! FMA → polynomial TRC → interleave. No profile parsing, no LUT allocation,
//! higher precision than table-based CMS (~5e-7 max error vs f64).
//!
//! # Supported color spaces
//!
//! | Space | Primaries | TRC | Constant prefix |
//! |---|---|---|---|
//! | sRGB | BT.709 | sRGB | `SRGB_TO_*` |
//! | Display P3 | DCI-P3 | sRGB | `P3_TO_*` |
//! | BT.2020 SDR | BT.2020 | BT.709 | `BT2020_TO_*` |
//! | BT.2020 PQ | BT.2020 | PQ | `BT2020_TO_*` + PQ TRC |
//! | BT.2020 HLG | BT.2020 | HLG | `BT2020_TO_*` + HLG TRC |
//! | Linear * | any | identity | matrix-only functions |
//!
//! # Architecture
//!
//! Each TRC family (srgb, bt709, pq, hlg, linear) has a stamped SIMD kernel
//! that fuses linearization, matrix multiply, and re-encoding into one pass
//! over the data. The matrix is a runtime parameter — the same kernel handles
//! any primaries conversion (P3→sRGB, BT.2020→sRGB, etc.) for a given TRC pair.
//!
//! Source and destination TRC may differ (e.g., BT.2020 PQ → sRGB uses PQ
//! linearization and sRGB encoding). For same-TRC conversions (P3 ↔ sRGB),
//! the linearize/encode functions are identical.

use archmage::prelude::*;
#[cfg(target_arch = "x86_64")]
use linear_srgb::tokens::x8 as trc_x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8 as mt_f32x8;

// =========================================================================
// Shared helpers
// =========================================================================

/// Apply a 3×3 matrix to an RGB triple.
#[inline(always)]
pub(crate) fn mat3x3_scalar(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    mat3x3(m, r, g, b)
}

#[inline(always)]
fn mat3x3(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b)),
        m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b)),
        m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b)),
    )
}

/// SIMD matrix multiply: 3 channels × 8 pixels via FMA.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mat3x3_x8(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    rl: mt_f32x8,
    gl: mt_f32x8,
    bl: mt_f32x8,
) -> (mt_f32x8, mt_f32x8, mt_f32x8) {
    let or = mt_f32x8::splat(token, m[0][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[0][1]).mul_add(gl, mt_f32x8::splat(token, m[0][2]) * bl),
    );
    let og = mt_f32x8::splat(token, m[1][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[1][1]).mul_add(gl, mt_f32x8::splat(token, m[1][2]) * bl),
    );
    let ob = mt_f32x8::splat(token, m[2][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[2][1]).mul_add(gl, mt_f32x8::splat(token, m[2][2]) * bl),
    );
    (or, og, ob)
}

// =========================================================================
// Macro: stamp fused kernels per TRC pair
// =========================================================================

/// Stamps out a complete set of fused SIMD + scalar kernels for a given
/// source/dest TRC pair. Each invocation generates:
/// - `fused_8px_rgb_{name}` / `fused_8px_rgba_{name}` (#[rite] SIMD inner)
/// - `convert_rgb_{name}_v3` / `convert_rgba_{name}_v3` (#[arcane] dispatch)
/// - `convert_rgb_{name}_scalar` / `convert_rgba_{name}_scalar` (fallback)
macro_rules! stamp_trc_kernels {
    (
        $name:ident,
        simd_linearize: $simd_lin:path,
        simd_encode: $simd_enc:path,
        scalar_linearize: $scalar_lin:path,
        scalar_encode: $scalar_enc:path
    ) => {
        paste::paste! {
            #[cfg(target_arch = "x86_64")]
            #[rite]
            fn [<fused_8px_rgb_ $name>](token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
                let mut r = [0.0f32; 8];
                let mut g = [0.0f32; 8];
                let mut b = [0.0f32; 8];
                for i in 0..8 {
                    r[i] = data[i * 3];
                    g[i] = data[i * 3 + 1];
                    b[i] = data[i * 3 + 2];
                }
                let rl = mt_f32x8::from_array(token, $simd_lin(token, r));
                let gl = mt_f32x8::from_array(token, $simd_lin(token, g));
                let bl = mt_f32x8::from_array(token, $simd_lin(token, b));
                let (or, og, ob) = mat3x3_x8(token, m, rl, gl, bl);
                let ro = $simd_enc(token, or.to_array());
                let go = $simd_enc(token, og.to_array());
                let bo = $simd_enc(token, ob.to_array());
                for i in 0..8 {
                    data[i * 3] = ro[i];
                    data[i * 3 + 1] = go[i];
                    data[i * 3 + 2] = bo[i];
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[rite]
            fn [<fused_8px_rgba_ $name>](token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
                let mut r = [0.0f32; 8];
                let mut g = [0.0f32; 8];
                let mut b = [0.0f32; 8];
                for i in 0..8 {
                    r[i] = data[i * 4];
                    g[i] = data[i * 4 + 1];
                    b[i] = data[i * 4 + 2];
                }
                let rl = mt_f32x8::from_array(token, $simd_lin(token, r));
                let gl = mt_f32x8::from_array(token, $simd_lin(token, g));
                let bl = mt_f32x8::from_array(token, $simd_lin(token, b));
                let (or, og, ob) = mat3x3_x8(token, m, rl, gl, bl);
                let ro = $simd_enc(token, or.to_array());
                let go = $simd_enc(token, og.to_array());
                let bo = $simd_enc(token, ob.to_array());
                for i in 0..8 {
                    data[i * 4] = ro[i];
                    data[i * 4 + 1] = go[i];
                    data[i * 4 + 2] = bo[i];
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[arcane]
            fn [<convert_rgb_ $name _v3>](token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
                let bulk = (data.len() / 24) * 24;
                for off in (0..bulk).step_by(24) {
                    [<fused_8px_rgb_ $name>](token, m, &mut data[off..off + 24]);
                }
                for pixel in data[bulk..].chunks_exact_mut(3) {
                    let r = $scalar_lin(pixel[0]);
                    let g = $scalar_lin(pixel[1]);
                    let b = $scalar_lin(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = $scalar_enc(nr);
                    pixel[1] = $scalar_enc(ng);
                    pixel[2] = $scalar_enc(nb);
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[arcane]
            fn [<convert_rgba_ $name _v3>](token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
                let bulk = (data.len() / 32) * 32;
                for off in (0..bulk).step_by(32) {
                    [<fused_8px_rgba_ $name>](token, m, &mut data[off..off + 32]);
                }
                for pixel in data[bulk..].chunks_exact_mut(4) {
                    let r = $scalar_lin(pixel[0]);
                    let g = $scalar_lin(pixel[1]);
                    let b = $scalar_lin(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = $scalar_enc(nr);
                    pixel[1] = $scalar_enc(ng);
                    pixel[2] = $scalar_enc(nb);
                }
            }

            fn [<convert_rgb_ $name _scalar>](_token: ScalarToken, m: &[[f32; 3]; 3], data: &mut [f32]) {
                for pixel in data.chunks_exact_mut(3) {
                    let r = $scalar_lin(pixel[0]);
                    let g = $scalar_lin(pixel[1]);
                    let b = $scalar_lin(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = $scalar_enc(nr);
                    pixel[1] = $scalar_enc(ng);
                    pixel[2] = $scalar_enc(nb);
                }
            }

            fn [<convert_rgba_ $name _scalar>](_token: ScalarToken, m: &[[f32; 3]; 3], data: &mut [f32]) {
                for pixel in data.chunks_exact_mut(4) {
                    let r = $scalar_lin(pixel[0]);
                    let g = $scalar_lin(pixel[1]);
                    let b = $scalar_lin(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = $scalar_enc(nr);
                    pixel[1] = $scalar_enc(ng);
                    pixel[2] = $scalar_enc(nb);
                }
            }
        }
    };
}

// =========================================================================
// Stamp kernels for each TRC family
// =========================================================================

// sRGB TRC (sRGB ↔ P3, both use sRGB curve)
stamp_trc_kernels!(srgb,
    simd_linearize: trc_x8::srgb_to_linear_v3,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: linear_srgb::tf::srgb_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// BT.709 TRC (BT.2020 SDR)
stamp_trc_kernels!(bt709,
    simd_linearize: trc_x8::bt709_to_linear_v3,
    simd_encode: trc_x8::linear_to_bt709_v3,
    scalar_linearize: linear_srgb::tf::bt709_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_bt709
);

// PQ TRC (BT.2020 PQ / HDR10)
stamp_trc_kernels!(pq,
    simd_linearize: trc_x8::pq_to_linear_v3,
    simd_encode: trc_x8::linear_to_pq_v3,
    scalar_linearize: linear_srgb::tf::pq_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_pq
);

// HLG TRC (BT.2020 HLG)
stamp_trc_kernels!(hlg,
    simd_linearize: trc_x8::hlg_to_linear_v3,
    simd_encode: trc_x8::linear_to_hlg_v3,
    scalar_linearize: linear_srgb::tf::hlg_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_hlg
);

// =========================================================================
// Linear (identity TRC) — matrix only, no TRC
// =========================================================================

/// Convert linear f32 RGB pixels in-place using only the 3×3 matrix.
/// No TRC linearization or encoding — input/output are already linear.
pub fn convert_linear_rgb(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = nr;
        pixel[1] = ng;
        pixel[2] = nb;
    }
}

/// Convert linear f32 RGBA pixels in-place (alpha unchanged).
pub fn convert_linear_rgba(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    for pixel in data.chunks_exact_mut(4) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = nr;
        pixel[1] = ng;
        pixel[2] = nb;
    }
}

// =========================================================================
// Cross-TRC kernels (source and dest use different TRCs)
// =========================================================================

// PQ source → sRGB dest (BT.2020 PQ → sRGB: PQ linearize, sRGB encode)
stamp_trc_kernels!(pq_to_srgb,
    simd_linearize: trc_x8::pq_to_linear_v3,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: linear_srgb::tf::pq_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// HLG source → sRGB dest
stamp_trc_kernels!(hlg_to_srgb,
    simd_linearize: trc_x8::hlg_to_linear_v3,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: linear_srgb::tf::hlg_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// sRGB source → PQ dest
stamp_trc_kernels!(srgb_to_pq,
    simd_linearize: trc_x8::srgb_to_linear_v3,
    simd_encode: trc_x8::linear_to_pq_v3,
    scalar_linearize: linear_srgb::tf::srgb_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_pq
);

// BT.709 source → sRGB dest (BT.2020 SDR → sRGB)
stamp_trc_kernels!(bt709_to_srgb,
    simd_linearize: trc_x8::bt709_to_linear_v3,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: linear_srgb::tf::bt709_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// sRGB source → BT.709 dest (sRGB → BT.2020 SDR)
stamp_trc_kernels!(srgb_to_bt709,
    simd_linearize: trc_x8::srgb_to_linear_v3,
    simd_encode: trc_x8::linear_to_bt709_v3,
    scalar_linearize: linear_srgb::tf::srgb_to_linear,
    scalar_encode: linear_srgb::tf::linear_to_bt709
);

// =========================================================================
// Adobe RGB gamma 2.2 TRC wrappers (bind gamma parameter for macro compat)
// =========================================================================

const ADOBE_GAMMA: f32 = 2.19921875; // Adobe RGB spec: 563/256

#[cfg(target_arch = "x86_64")]
#[rite]
fn adobe_to_linear_x8(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    trc_x8::gamma_to_linear_v3(token, v, ADOBE_GAMMA)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn adobe_from_linear_x8(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    trc_x8::linear_to_gamma_v3(token, v, ADOBE_GAMMA)
}

#[inline(always)]
fn adobe_to_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::gamma_to_linear(v, ADOBE_GAMMA)
}

#[inline(always)]
fn adobe_from_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::linear_to_gamma(v, ADOBE_GAMMA)
}

// Adobe RGB same-TRC (both sides gamma 2.2 — pure power, no linear toe).
//
// This is the canonical Adobe RGB 1998 encoding per the Adobe RGB 1998
// spec §4.3.4.2 and matches ~85% of real-world Adobe RGB ICC profiles
// (Adobe CS4, Windows ClayRGB1998, macOS AdobeRGB1998, Linux
// `AdobeRGB1998`/`compatibleWithAdobeRGB1998`, Nikon, etc).
//
// The `parametricCurveType funcType=3` variant with a linear toe
// (`c=1/32, d=0.05568`) that some profiles use — saucecontrol's
// Compact-ICC AdobeCompat-v4 being the notable example — is deliberately
// NOT accelerated here. Profiles encoding that form fall through to the
// full CMS (moxcms) so they're rendered via their actual bytes. See
// `scripts/icc-gen/src/main.rs` for the identification-side policy and
// the rationale notes in `zenpixels-convert/src/icc_profiles.rs`.
stamp_trc_kernels!(adobe,
    simd_linearize: adobe_to_linear_x8,
    simd_encode: adobe_from_linear_x8,
    scalar_linearize: adobe_to_linear_scalar,
    scalar_encode: adobe_from_linear_scalar
);

// Adobe RGB source → sRGB dest (gamma 2.2 linearize, sRGB encode)
stamp_trc_kernels!(adobe_to_srgb,
    simd_linearize: adobe_to_linear_x8,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: adobe_to_linear_scalar,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// sRGB source → Adobe RGB dest (sRGB linearize, gamma 2.2 encode)
stamp_trc_kernels!(srgb_to_adobe,
    simd_linearize: trc_x8::srgb_to_linear_v3,
    simd_encode: adobe_from_linear_x8,
    scalar_linearize: linear_srgb::tf::srgb_to_linear,
    scalar_encode: adobe_from_linear_scalar
);

// =========================================================================
// Dispatch: pick the right kernel for a given (src_trc, dst_trc) pair
// =========================================================================

use crate::TransferFunction;

// =========================================================================
// Extended range (sign-preserving, no clamping — for HDR / out-of-gamut)
// =========================================================================

/// Sign-preserving sRGB linearization (extended range, scalar `powf`).
#[inline(always)]
fn linearize_srgb_extended(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::srgb_to_linear_extended(v)
    } else {
        -linear_srgb::precise::srgb_to_linear_extended(-v)
    }
}

/// Sign-preserving sRGB encoding (extended range, scalar `powf`).
#[inline(always)]
fn encode_srgb_extended(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::linear_to_srgb_extended(v)
    } else {
        -linear_srgb::precise::linear_to_srgb_extended(-v)
    }
}

/// Sign-preserving generic gamma linearization.
#[inline(always)]
fn linearize_gamma_extended(v: f32, gamma: f32) -> f32 {
    if v >= 0.0 {
        v.powf(gamma)
    } else {
        -((-v).powf(gamma))
    }
}

/// Sign-preserving generic gamma encoding.
#[inline(always)]
fn encode_gamma_extended(v: f32, inv_gamma: f32) -> f32 {
    if v >= 0.0 {
        v.powf(inv_gamma)
    } else {
        -((-v).powf(inv_gamma))
    }
}

/// Scalar linearization for extended range (sign-preserving, no clamping).
fn scalar_linearize_extended(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(linearize_srgb_extended),
        TransferFunction::Gamma22 => Some(|v| linearize_gamma_extended(v, ADOBE_GAMMA)),
        TransferFunction::Linear => Some(core::convert::identity),
        // PQ/HLG/BT.709 don't produce negative values in practice,
        // but use the clamped versions as fallback.
        _ => scalar_linearize(trc),
    }
}

/// Scalar encoding for extended range (sign-preserving, no clamping).
fn scalar_encode_extended(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(encode_srgb_extended),
        TransferFunction::Gamma22 => Some(|v| encode_gamma_extended(v, 1.0 / ADOBE_GAMMA)),
        TransferFunction::Linear => Some(core::convert::identity),
        _ => scalar_encode(trc),
    }
}

/// Convert f32 RGB data in-place with extended range (no clamping).
/// Scalar-only — sign preservation requires per-channel branching.
pub(crate) fn convert_f32_rgb_extended(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    let Some(lin) = scalar_linearize_extended(src_trc) else {
        return false;
    };
    let Some(enc) = scalar_encode_extended(dst_trc) else {
        return false;
    };
    for pixel in data.chunks_exact_mut(3) {
        let r = lin(pixel[0]);
        let g = lin(pixel[1]);
        let b = lin(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = enc(nr);
        pixel[1] = enc(ng);
        pixel[2] = enc(nb);
    }
    true
}

/// Convert f32 RGBA data in-place with extended range. Alpha unchanged.
pub(crate) fn convert_f32_rgba_extended(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    let Some(lin) = scalar_linearize_extended(src_trc) else {
        return false;
    };
    let Some(enc) = scalar_encode_extended(dst_trc) else {
        return false;
    };
    for pixel in data.chunks_exact_mut(4) {
        let r = lin(pixel[0]);
        let g = lin(pixel[1]);
        let b = lin(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = enc(nr);
        pixel[1] = enc(ng);
        pixel[2] = enc(nb);
    }
    true
}

// =========================================================================
// Scalar TRC lookup (clamped)
// =========================================================================

/// Whether a TRC has a SIMD x8 encode kernel available (x86_64 only).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_simd_encode(trc: TransferFunction) -> bool {
    matches!(
        trc,
        TransferFunction::Srgb
            | TransferFunction::Bt709
            | TransferFunction::Pq
            | TransferFunction::Hlg
            | TransferFunction::Gamma22
    )
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_simd_encode(_trc: TransferFunction) -> bool {
    false
}

/// SIMD x8 encode dispatch — must be called from a #[rite]/#[arcane] context.
#[cfg(target_arch = "x86_64")]
#[rite]
fn simd_encode_x8_dispatch(token: X64V3Token, trc: TransferFunction, v: [f32; 8]) -> [f32; 8] {
    match trc {
        TransferFunction::Srgb => trc_x8::linear_to_srgb_v3(token, v),
        TransferFunction::Bt709 => trc_x8::linear_to_bt709_v3(token, v),
        TransferFunction::Pq => trc_x8::linear_to_pq_v3(token, v),
        TransferFunction::Hlg => trc_x8::linear_to_hlg_v3(token, v),
        TransferFunction::Gamma22 => adobe_from_linear_x8(token, v),
        _ => v,
    }
}

/// Scalar linearization function for a given transfer function.
pub(crate) fn scalar_linearize(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(linear_srgb::tf::srgb_to_linear),
        TransferFunction::Bt709 => Some(linear_srgb::tf::bt709_to_linear),
        TransferFunction::Pq => Some(linear_srgb::tf::pq_to_linear),
        TransferFunction::Hlg => Some(linear_srgb::tf::hlg_to_linear),
        TransferFunction::Gamma22 => Some(adobe_to_linear_scalar),
        TransferFunction::Linear => Some(core::convert::identity),
        _ => None,
    }
}

/// Scalar encode function for a given transfer function.
pub(crate) fn scalar_encode(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(linear_srgb::tf::linear_to_srgb),
        TransferFunction::Bt709 => Some(linear_srgb::tf::linear_to_bt709),
        TransferFunction::Pq => Some(linear_srgb::tf::linear_to_pq),
        TransferFunction::Hlg => Some(linear_srgb::tf::linear_to_hlg),
        TransferFunction::Gamma22 => Some(adobe_from_linear_scalar),
        TransferFunction::Linear => Some(core::convert::identity),
        _ => None,
    }
}

/// Convert f32 RGB data in-place using the given gamut matrix and TRC pair.
///
/// Dispatches to fused SIMD kernels when a specialized kernel exists for the
/// (src_trc, dst_trc) pair. Falls back to scalar linearize → matrix → encode
/// for unsupported pairs. Returns `false` if either TRC is unknown.
pub(crate) fn convert_f32_rgb_dispatch(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 3, 0);
    // Linear is cross-platform — handle before SIMD dispatch.
    if src_trc == Linear && dst_trc == Linear {
        convert_linear_rgb(m, data);
        return true;
    }
    // SIMD fast path (x86_64 AVX2+FMA only). Each `incant!` does runtime
    // capability detection and falls through to a scalar implementation
    // when AVX2+FMA is unavailable.
    #[cfg(target_arch = "x86_64")]
    match (src_trc, dst_trc) {
        (Srgb, Srgb) => {
            incant!(convert_rgb_srgb(m, data));
            return true;
        }
        (Bt709, Bt709) => {
            incant!(convert_rgb_bt709(m, data));
            return true;
        }
        (Pq, Pq) => {
            incant!(convert_rgb_pq(m, data));
            return true;
        }
        (Hlg, Hlg) => {
            incant!(convert_rgb_hlg(m, data));
            return true;
        }
        (Gamma22, Gamma22) => {
            incant!(convert_rgb_adobe(m, data));
            return true;
        }
        (Pq, Srgb) => {
            incant!(convert_rgb_pq_to_srgb(m, data));
            return true;
        }
        (Hlg, Srgb) => {
            incant!(convert_rgb_hlg_to_srgb(m, data));
            return true;
        }
        (Srgb, Pq) => {
            incant!(convert_rgb_srgb_to_pq(m, data));
            return true;
        }
        (Bt709, Srgb) => {
            incant!(convert_rgb_bt709_to_srgb(m, data));
            return true;
        }
        (Srgb, Bt709) => {
            incant!(convert_rgb_srgb_to_bt709(m, data));
            return true;
        }
        (Gamma22, Srgb) => {
            incant!(convert_rgb_adobe_to_srgb(m, data));
            return true;
        }
        (Srgb, Gamma22) => {
            incant!(convert_rgb_srgb_to_adobe(m, data));
            return true;
        }
        _ => {} // fall through to scalar
    }
    // Scalar fallback — works on all platforms.
    let Some(lin) = scalar_linearize(src_trc) else {
        return false;
    };
    let Some(enc) = scalar_encode(dst_trc) else {
        return false;
    };
    for pixel in data.chunks_exact_mut(3) {
        let r = lin(pixel[0]);
        let g = lin(pixel[1]);
        let b = lin(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = enc(nr);
        pixel[1] = enc(ng);
        pixel[2] = enc(nb);
    }
    true
}

/// Convert f32 RGBA data in-place using the given gamut matrix and TRC pair.
/// Alpha channel is preserved unchanged.
pub(crate) fn convert_f32_rgba_dispatch(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 4, 0);
    if src_trc == Linear && dst_trc == Linear {
        convert_linear_rgba(m, data);
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    match (src_trc, dst_trc) {
        (Srgb, Srgb) => {
            incant!(convert_rgba_srgb(m, data));
            return true;
        }
        (Bt709, Bt709) => {
            incant!(convert_rgba_bt709(m, data));
            return true;
        }
        (Pq, Pq) => {
            incant!(convert_rgba_pq(m, data));
            return true;
        }
        (Hlg, Hlg) => {
            incant!(convert_rgba_hlg(m, data));
            return true;
        }
        (Gamma22, Gamma22) => {
            incant!(convert_rgba_adobe(m, data));
            return true;
        }
        (Pq, Srgb) => {
            incant!(convert_rgba_pq_to_srgb(m, data));
            return true;
        }
        (Hlg, Srgb) => {
            incant!(convert_rgba_hlg_to_srgb(m, data));
            return true;
        }
        (Srgb, Pq) => {
            incant!(convert_rgba_srgb_to_pq(m, data));
            return true;
        }
        (Bt709, Srgb) => {
            incant!(convert_rgba_bt709_to_srgb(m, data));
            return true;
        }
        (Srgb, Bt709) => {
            incant!(convert_rgba_srgb_to_bt709(m, data));
            return true;
        }
        (Gamma22, Srgb) => {
            incant!(convert_rgba_adobe_to_srgb(m, data));
            return true;
        }
        (Srgb, Gamma22) => {
            incant!(convert_rgba_srgb_to_adobe(m, data));
            return true;
        }
        _ => {}
    }
    let Some(lin) = scalar_linearize(src_trc) else {
        return false;
    };
    let Some(enc) = scalar_encode(dst_trc) else {
        return false;
    };
    for pixel in data.chunks_exact_mut(4) {
        let r = lin(pixel[0]);
        let g = lin(pixel[1]);
        let b = lin(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = enc(nr);
        pixel[1] = enc(ng);
        pixel[2] = enc(nb);
    }
    true
}

// =========================================================================
// u8 ↔ f32 wrappers
// =========================================================================

/// Build a 256-entry LUT for u8 → f32 linearization.
///
/// `lut[i] = linearize_fn(i as f32 / 255.0)`. Eliminates per-channel function
/// calls during conversion — just an array index.
pub(crate) fn build_linearize_lut(linearize_fn: fn(f32) -> f32) -> alloc::boxed::Box<[f32; 256]> {
    let mut lut = alloc::vec![0.0f32; 256].into_boxed_slice();
    for i in 0..256 {
        lut[i] = linearize_fn(i as f32 / 255.0);
    }
    lut.try_into().ok().unwrap()
}

/// Convert u8 RGB source to u8 RGB dest via gamut conversion.
///
/// Source u8 values are normalized to [0,1], linearized, matrix-transformed,
/// then re-encoded and quantized to u8 output.
pub(crate) fn convert_u8_rgb(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    linearize_fn: fn(f32) -> f32,
    encode_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = linearize_fn(src_px[0] as f32 / 255.0);
        let g = linearize_fn(src_px[1] as f32 / 255.0);
        let b = linearize_fn(src_px[2] as f32 / 255.0);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = (encode_fn(nr) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[1] = (encode_fn(ng) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[2] = (encode_fn(nb) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Scalar f32→u8 encode function for a given transfer function.
/// Uses LUT for sRGB (4096-entry const table, no math),
/// polynomial + quantize for others.
pub(crate) fn scalar_encode_u8(trc: TransferFunction) -> Option<fn(f32) -> u8> {
    fn srgb_u8(v: f32) -> u8 {
        linear_srgb::default::linear_to_srgb_u8(v)
    }
    fn quantize_with(enc: fn(f32) -> f32, v: f32) -> u8 {
        (enc(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }
    match trc {
        TransferFunction::Srgb => Some(srgb_u8),
        TransferFunction::Bt709 => Some(|v| quantize_with(linear_srgb::tf::linear_to_bt709, v)),
        TransferFunction::Pq => Some(|v| quantize_with(linear_srgb::tf::linear_to_pq, v)),
        TransferFunction::Hlg => Some(|v| quantize_with(linear_srgb::tf::linear_to_hlg, v)),
        TransferFunction::Gamma22 => Some(|v| quantize_with(adobe_from_linear_scalar, v)),
        TransferFunction::Linear => Some(|v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8),
        _ => None,
    }
}

/// Convert u8 RGB via LUT linearization AND LUT encoding (fastest u8→u8 path).
///
/// Linearize via 256-entry LUT, matrix multiply, encode via 4096-entry LUT.
/// No polynomial math — pure LUT + matrix.
pub(crate) fn convert_u8_rgb_lut_lut(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = enc_u8(nr);
        dst_px[1] = enc_u8(ng);
        dst_px[2] = enc_u8(nb);
    }
}

/// Fused u8→u8 RGB: LUT linearize → SIMD (matrix + polynomial encode) → quantize.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_u8_rgb_fused(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8; 24],
    dst: &mut [u8; 24],
    lin_lut: &[f32; 256],
    dst_trc: TransferFunction,
) {
    // u8 indices are always 0..255 → always in bounds for [f32; 256].
    // Fixed-size [u8; 24] input eliminates bounds checks on src.
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        r[i] = lin_lut[src[i * 3] as usize];
        g[i] = lin_lut[src[i * 3 + 1] as usize];
        b[i] = lin_lut[src[i * 3 + 2] as usize];
    }
    let rv = mt_f32x8::from_array(token, r);
    let gv = mt_f32x8::from_array(token, g);
    let bv = mt_f32x8::from_array(token, b);
    let (or, og, ob) = mat3x3_x8(token, m, rv, gv, bv);
    let ro = simd_encode_x8_dispatch(token, dst_trc, or.to_array());
    let go = simd_encode_x8_dispatch(token, dst_trc, og.to_array());
    let bo = simd_encode_x8_dispatch(token, dst_trc, ob.to_array());
    for i in 0..8 {
        dst[i * 3] = (ro[i] * 255.0 + 0.5) as u8;
        dst[i * 3 + 1] = (go[i] * 255.0 + 0.5) as u8;
        dst[i * 3 + 2] = (bo[i] * 255.0 + 0.5) as u8;
    }
}

/// 2-pixel kernel using safe magetypes APIs. Mirrors moxcms rgb_xyz_opt:
/// each 128-bit lane holds one pixel's [R,G,B,0] after FMA. Clamp+scale+
/// `to_i32_round` produces i32 indices; aligned store, scalar LUT gather.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_u8_rgb_matlut(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8; 24],
    dst: &mut [u8; 24],
    lin_lut: &[f32; 256],
    enc_lut: &[u8; 4096],
) {
    // Matrix rows pre-laid-out for the packed 2-pixel layout.
    // Each 256-bit vector holds the same 3×f32 row twice (low 128, high 128),
    // with padding in lane 3 and 7.
    let m0 = mt_f32x8::from_array(
        token,
        [
            m[0][0], m[1][0], m[2][0], 0.0, m[0][0], m[1][0], m[2][0], 0.0,
        ],
    );
    let m1 = mt_f32x8::from_array(
        token,
        [
            m[0][1], m[1][1], m[2][1], 0.0, m[0][1], m[1][1], m[2][1], 0.0,
        ],
    );
    let m2 = mt_f32x8::from_array(
        token,
        [
            m[0][2], m[1][2], m[2][2], 0.0, m[0][2], m[1][2], m[2][2], 0.0,
        ],
    );
    let scale = mt_f32x8::splat(token, 4095.0);
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);

    for pair in 0..4 {
        let off = pair * 6;
        let p0r = lin_lut[src[off] as usize];
        let p0g = lin_lut[src[off + 1] as usize];
        let p0b = lin_lut[src[off + 2] as usize];
        let p1r = lin_lut[src[off + 3] as usize];
        let p1g = lin_lut[src[off + 4] as usize];
        let p1b = lin_lut[src[off + 5] as usize];

        // Broadcast each channel to fill a 128-bit lane; pack 2 pixels.
        let r = mt_f32x8::from_array(token, [p0r, p0r, p0r, p0r, p1r, p1r, p1r, p1r]);
        let g = mt_f32x8::from_array(token, [p0g, p0g, p0g, p0g, p1g, p1g, p1g, p1g]);
        let b = mt_f32x8::from_array(token, [p0b, p0b, p0b, p0b, p1b, p1b, p1b, p1b]);

        // v = r*m0 + g*m1 + b*m2 — each 128-bit lane ends up as [R', G', B', 0].
        let v = r * m0 + g * m1 + b * m2;

        // Clamp [0,1], scale, SIMD f32→i32 round, extract as array (in-register).
        let clamped = v.max(zero).min(one);
        let scaled = clamped * scale;
        let idx = scaled.to_i32_round().to_array();

        dst[off] = enc_lut[idx[0] as usize & 0xFFF];
        dst[off + 1] = enc_lut[idx[1] as usize & 0xFFF];
        dst[off + 2] = enc_lut[idx[2] as usize & 0xFFF];
        dst[off + 3] = enc_lut[idx[4] as usize & 0xFFF];
        dst[off + 4] = enc_lut[idx[5] as usize & 0xFFF];
        dst[off + 5] = enc_lut[idx[6] as usize & 0xFFF];
    }
}

/// Build a 256-entry LUT mapping u8 sRGB → f32 linear.
pub(crate) fn srgb_lin_lut_u8() -> &'static [f32; 256] {
    use once_cell::race::OnceBox;
    static LUT: OnceBox<[f32; 256]> = OnceBox::new();
    LUT.get_or_init(|| {
        let mut t = alloc::boxed::Box::new([0.0f32; 256]);
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = linear_srgb::default::srgb_u8_to_linear(i as u8);
        }
        t
    })
}

/// Build a 4096-entry LUT mapping linear f32 → u8 sRGB.
pub(crate) fn srgb_enc_lut_u8() -> &'static [u8; 4096] {
    use once_cell::race::OnceBox;
    static LUT: OnceBox<[u8; 4096]> = OnceBox::new();
    LUT.get_or_init(|| {
        let mut t = alloc::boxed::Box::new([0u8; 4096]);
        for (i, slot) in t.iter_mut().enumerate() {
            let lin = i as f32 / 4095.0;
            *slot = linear_srgb::default::linear_to_srgb_u8(lin);
        }
        t
    })
}

/// Back-compat alias; prefer `srgb_enc_lut_u8`.
fn srgb_enc_lut_4096() -> &'static [u8; 4096] {
    srgb_enc_lut_u8()
}

/// Build a 65536-entry LUT mapping u16 sRGB → f32 linear.
pub(crate) fn srgb_lin_lut_u16() -> &'static [f32; 65536] {
    use once_cell::race::OnceBox;
    static LUT: OnceBox<[f32; 65536]> = OnceBox::new();
    LUT.get_or_init(|| {
        let mut t: alloc::boxed::Box<[f32; 65536]> = alloc::vec![0.0f32; 65536]
            .into_boxed_slice()
            .try_into()
            .ok()
            .unwrap();
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = linear_srgb::default::srgb_u16_to_linear(i as u16);
        }
        t
    })
}

/// Build a 65536-entry LUT mapping i32 index (scaled linear) → u16 sRGB.
/// Indexed by `(linear * 65535 + 0.5) as i32` after clamp.
/// Build a 65536-entry linearly-indexed u16 encode LUT. Lookup is
/// `idx = (linear * 65535 + 0.5) as usize & 0xFFFF`.
///
/// Sqrt-indexing (measured 2026-04-24, benchmarks/u16_sqrt_vs_linear_*)
/// was rejected for production: while the encode step in isolation is
/// tied at 13.1 GiB/s, the full matlut pipeline has a 2-pixel-per-vector
/// AoS layout that does 4 sqrts per 8-pixel chunk — 1.33× the sqrt
/// density of an isolated encode, and the cycles no longer hide behind
/// gather AGU dispatch. Ended up -17% on T7. See
/// benchmarks/u16_sqrt_aos_regression_2026-04-24.txt.
pub(crate) fn srgb_enc_lut_u16() -> &'static [u16; 65536] {
    use once_cell::race::OnceBox;
    static LUT: OnceBox<[u16; 65536]> = OnceBox::new();
    LUT.get_or_init(|| {
        let mut t: alloc::boxed::Box<[u16; 65536]> = alloc::vec![0u16; 65536]
            .into_boxed_slice()
            .try_into()
            .ok()
            .unwrap();
        for (i, slot) in t.iter_mut().enumerate() {
            let lin = i as f32 / 65535.0;
            *slot = linear_srgb::default::linear_to_srgb_u16(lin);
        }
        t
    })
}

/// 2-pixel u16 sRGB kernel using safe magetypes APIs.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_u16_rgb_matlut(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u16; 24],
    dst: &mut [u16; 24],
    lin_lut: &[f32; 65536],
    enc_lut: &[u16; 65536],
) {
    let m0 = mt_f32x8::from_array(
        token,
        [
            m[0][0], m[1][0], m[2][0], 0.0, m[0][0], m[1][0], m[2][0], 0.0,
        ],
    );
    let m1 = mt_f32x8::from_array(
        token,
        [
            m[0][1], m[1][1], m[2][1], 0.0, m[0][1], m[1][1], m[2][1], 0.0,
        ],
    );
    let m2 = mt_f32x8::from_array(
        token,
        [
            m[0][2], m[1][2], m[2][2], 0.0, m[0][2], m[1][2], m[2][2], 0.0,
        ],
    );
    let scale = mt_f32x8::splat(token, 65535.0);
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);

    let mut temp = [0i32; 8];

    for pair in 0..4 {
        let off = pair * 6;
        let p0r = lin_lut[src[off] as usize];
        let p0g = lin_lut[src[off + 1] as usize];
        let p0b = lin_lut[src[off + 2] as usize];
        let p1r = lin_lut[src[off + 3] as usize];
        let p1g = lin_lut[src[off + 4] as usize];
        let p1b = lin_lut[src[off + 5] as usize];

        let r = mt_f32x8::from_array(token, [p0r, p0r, p0r, p0r, p1r, p1r, p1r, p1r]);
        let g = mt_f32x8::from_array(token, [p0g, p0g, p0g, p0g, p1g, p1g, p1g, p1g]);
        let b = mt_f32x8::from_array(token, [p0b, p0b, p0b, p0b, p1b, p1b, p1b, p1b]);

        let v = r * m0 + g * m1 + b * m2;

        let clamped = v.max(zero).min(one);
        let scaled = clamped * scale;
        let idx = scaled.to_i32_round();
        idx.store(&mut temp);

        dst[off] = enc_lut[temp[0] as usize & 0xFFFF];
        dst[off + 1] = enc_lut[temp[1] as usize & 0xFFFF];
        dst[off + 2] = enc_lut[temp[2] as usize & 0xFFFF];
        dst[off + 3] = enc_lut[temp[4] as usize & 0xFFFF];
        dst[off + 4] = enc_lut[temp[5] as usize & 0xFFFF];
        dst[off + 5] = enc_lut[temp[6] as usize & 0xFFFF];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_u16_rgb_matlut_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    lin_lut: &[f32; 65536],
    enc_lut: &[u16; 65536],
) {
    let pixel_count = src.len() / 3;
    let bulk = (pixel_count / 8) * 8;
    let bulk_chans = bulk * 3;
    for off in (0..bulk_chans).step_by(24) {
        let s: &[u16; 24] = src[off..off + 24].try_into().unwrap();
        let d: &mut [u16; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
        convert_8px_u16_rgb_matlut(token, m, s, d, lin_lut, enc_lut);
    }
    // Scalar remainder.
    for i in bulk..pixel_count {
        let base = i * 3;
        let r = lin_lut[src[base] as usize];
        let g = lin_lut[src[base + 1] as usize];
        let b = lin_lut[src[base + 2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst[base] = enc_lut[(nr.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        dst[base + 1] = enc_lut[(ng.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        dst[base + 2] = enc_lut[(nb.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
    }
}

fn convert_u16_rgb_matlut_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    lin_lut: &[f32; 65536],
    enc_lut: &[u16; 65536],
) {
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = enc_lut[(nr.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        dst_px[1] = enc_lut[(ng.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        dst_px[2] = enc_lut[(nb.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
    }
}

/// Fused u16 RGB: LUT linearize → SIMD matrix → SIMD f32→i32 → LUT encode.
/// sRGB-specific.
pub(crate) fn convert_u16_rgb_simd_matlut(
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    lin_lut: &[f32; 65536],
    enc_lut: &[u16; 65536],
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_u16_rgb_matlut(m, src, dst, lin_lut, enc_lut));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_u16_rgb_matlut_scalar(ScalarToken, m, src, dst, lin_lut, enc_lut);
}

// =========================================================================
// u16 RGB fused-gamut hybrid kernels (bench-only, not on a hot path).
//
// Four combinations of decode and encode: {LUT, poly} × {LUT, poly}. Used
// to find the optimal point in the mix for sRGB u16 gamut conversion.
// All use mat3x3_x8 for the matrix step. Requires the u16↔f32 polynomial
// rites which ship in linear-srgb ≥ 0.6.12 (not yet released on
// crates.io). Gated behind `__bench_u16_hybrids` feature for now.
//
// When linear-srgb 0.6.12 lands on crates.io, remove this feature gate,
// bump zenpixels-convert's linear-srgb minimum to 0.6.12, and switch
// FusedSrgbU16GamutRgb in convert_kernels.rs to call
// `convert_u16_rgb_simd_lutdec_polyenc` (LUT+poly wins by +17% at 1080p
// and gets exact u16 roundtrip).
// =========================================================================

#[cfg(feature = "__bench_u16_hybrids")]
mod hybrids {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[rite]
    fn convert_8px_u16_rgb_lutdec_polyenc(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16; 24],
        dst: &mut [u16; 24],
        lin_lut: &[f32; 65536],
    ) {
        let mut rl_arr = [0.0f32; 8];
        let mut gl_arr = [0.0f32; 8];
        let mut bl_arr = [0.0f32; 8];
        for i in 0..8 {
            rl_arr[i] = lin_lut[src[i * 3] as usize];
            gl_arr[i] = lin_lut[src[i * 3 + 1] as usize];
            bl_arr[i] = lin_lut[src[i * 3 + 2] as usize];
        }
        let rl = mt_f32x8::from_array(token, rl_arr);
        let gl = mt_f32x8::from_array(token, gl_arr);
        let bl = mt_f32x8::from_array(token, bl_arr);
        let (or_v, og_v, ob_v) = mat3x3_x8(token, m, rl, gl, bl);
        let ro = trc_x8::linear_to_srgb_u16_v3(token, or_v.to_array());
        let go = trc_x8::linear_to_srgb_u16_v3(token, og_v.to_array());
        let bo = trc_x8::linear_to_srgb_u16_v3(token, ob_v.to_array());
        for i in 0..8 {
            dst[i * 3] = ro[i];
            dst[i * 3 + 1] = go[i];
            dst[i * 3 + 2] = bo[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[rite]
    fn convert_8px_u16_rgb_polydec_lutenc(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16; 24],
        dst: &mut [u16; 24],
        enc_lut: &[u16; 65536],
    ) {
        let mut r_u16 = [0u16; 8];
        let mut g_u16 = [0u16; 8];
        let mut b_u16 = [0u16; 8];
        for i in 0..8 {
            r_u16[i] = src[i * 3];
            g_u16[i] = src[i * 3 + 1];
            b_u16[i] = src[i * 3 + 2];
        }
        let rl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, r_u16));
        let gl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, g_u16));
        let bl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, b_u16));
        let (or_v, og_v, ob_v) = mat3x3_x8(token, m, rl, gl, bl);
        let scale = mt_f32x8::splat(token, 65535.0);
        let zero = mt_f32x8::zero(token);
        let one = mt_f32x8::splat(token, 1.0);
        let r_idx = (or_v.max(zero).min(one) * scale).to_i32_round().to_array();
        let g_idx = (og_v.max(zero).min(one) * scale).to_i32_round().to_array();
        let b_idx = (ob_v.max(zero).min(one) * scale).to_i32_round().to_array();
        for i in 0..8 {
            dst[i * 3] = enc_lut[r_idx[i] as usize & 0xFFFF];
            dst[i * 3 + 1] = enc_lut[g_idx[i] as usize & 0xFFFF];
            dst[i * 3 + 2] = enc_lut[b_idx[i] as usize & 0xFFFF];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[rite]
    fn convert_8px_u16_rgb_polydec_polyenc(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16; 24],
        dst: &mut [u16; 24],
    ) {
        let mut r_u16 = [0u16; 8];
        let mut g_u16 = [0u16; 8];
        let mut b_u16 = [0u16; 8];
        for i in 0..8 {
            r_u16[i] = src[i * 3];
            g_u16[i] = src[i * 3 + 1];
            b_u16[i] = src[i * 3 + 2];
        }
        let rl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, r_u16));
        let gl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, g_u16));
        let bl = mt_f32x8::from_array(token, trc_x8::srgb_u16_to_linear_v3(token, b_u16));
        let (or_v, og_v, ob_v) = mat3x3_x8(token, m, rl, gl, bl);
        let ro = trc_x8::linear_to_srgb_u16_v3(token, or_v.to_array());
        let go = trc_x8::linear_to_srgb_u16_v3(token, og_v.to_array());
        let bo = trc_x8::linear_to_srgb_u16_v3(token, ob_v.to_array());
        for i in 0..8 {
            dst[i * 3] = ro[i];
            dst[i * 3 + 1] = go[i];
            dst[i * 3 + 2] = bo[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn convert_u16_rgb_lutdec_polyenc_v3(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
        lin_lut: &[f32; 65536],
    ) {
        let pixel_count = src.len() / 3;
        let bulk = (pixel_count / 8) * 8;
        let bulk_chans = bulk * 3;
        for off in (0..bulk_chans).step_by(24) {
            let s: &[u16; 24] = src[off..off + 24].try_into().unwrap();
            let d: &mut [u16; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
            convert_8px_u16_rgb_lutdec_polyenc(token, m, s, d, lin_lut);
        }
        for i in bulk..pixel_count {
            let base = i * 3;
            let r = lin_lut[src[base] as usize];
            let g = lin_lut[src[base + 1] as usize];
            let b = lin_lut[src[base + 2] as usize];
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst[base] = linear_srgb::default::linear_to_srgb_u16(nr);
            dst[base + 1] = linear_srgb::default::linear_to_srgb_u16(ng);
            dst[base + 2] = linear_srgb::default::linear_to_srgb_u16(nb);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn convert_u16_rgb_polydec_lutenc_v3(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
        enc_lut: &[u16; 65536],
    ) {
        let pixel_count = src.len() / 3;
        let bulk = (pixel_count / 8) * 8;
        let bulk_chans = bulk * 3;
        for off in (0..bulk_chans).step_by(24) {
            let s: &[u16; 24] = src[off..off + 24].try_into().unwrap();
            let d: &mut [u16; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
            convert_8px_u16_rgb_polydec_lutenc(token, m, s, d, enc_lut);
        }
        for i in bulk..pixel_count {
            let base = i * 3;
            let r = linear_srgb::default::srgb_u16_to_linear(src[base]);
            let g = linear_srgb::default::srgb_u16_to_linear(src[base + 1]);
            let b = linear_srgb::default::srgb_u16_to_linear(src[base + 2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst[base] = enc_lut[(nr.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
            dst[base + 1] = enc_lut[(ng.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
            dst[base + 2] = enc_lut[(nb.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn convert_u16_rgb_polydec_polyenc_v3(
        token: X64V3Token,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        let pixel_count = src.len() / 3;
        let bulk = (pixel_count / 8) * 8;
        let bulk_chans = bulk * 3;
        for off in (0..bulk_chans).step_by(24) {
            let s: &[u16; 24] = src[off..off + 24].try_into().unwrap();
            let d: &mut [u16; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
            convert_8px_u16_rgb_polydec_polyenc(token, m, s, d);
        }
        for i in bulk..pixel_count {
            let base = i * 3;
            let r = linear_srgb::default::srgb_u16_to_linear(src[base]);
            let g = linear_srgb::default::srgb_u16_to_linear(src[base + 1]);
            let b = linear_srgb::default::srgb_u16_to_linear(src[base + 2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst[base] = linear_srgb::default::linear_to_srgb_u16(nr);
            dst[base + 1] = linear_srgb::default::linear_to_srgb_u16(ng);
            dst[base + 2] = linear_srgb::default::linear_to_srgb_u16(nb);
        }
    }

    fn convert_u16_rgb_lutdec_polyenc_scalar(
        _token: ScalarToken,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
        lin_lut: &[f32; 65536],
    ) {
        for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let r = lin_lut[src_px[0] as usize];
            let g = lin_lut[src_px[1] as usize];
            let b = lin_lut[src_px[2] as usize];
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst_px[0] = linear_srgb::default::linear_to_srgb_u16(nr);
            dst_px[1] = linear_srgb::default::linear_to_srgb_u16(ng);
            dst_px[2] = linear_srgb::default::linear_to_srgb_u16(nb);
        }
    }

    fn convert_u16_rgb_polydec_lutenc_scalar(
        _token: ScalarToken,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
        enc_lut: &[u16; 65536],
    ) {
        for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let r = linear_srgb::default::srgb_u16_to_linear(src_px[0]);
            let g = linear_srgb::default::srgb_u16_to_linear(src_px[1]);
            let b = linear_srgb::default::srgb_u16_to_linear(src_px[2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst_px[0] = enc_lut[(nr.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
            dst_px[1] = enc_lut[(ng.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
            dst_px[2] = enc_lut[(nb.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize & 0xFFFF];
        }
    }

    fn convert_u16_rgb_polydec_polyenc_scalar(
        _token: ScalarToken,
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let r = linear_srgb::default::srgb_u16_to_linear(src_px[0]);
            let g = linear_srgb::default::srgb_u16_to_linear(src_px[1]);
            let b = linear_srgb::default::srgb_u16_to_linear(src_px[2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            dst_px[0] = linear_srgb::default::linear_to_srgb_u16(nr);
            dst_px[1] = linear_srgb::default::linear_to_srgb_u16(ng);
            dst_px[2] = linear_srgb::default::linear_to_srgb_u16(nb);
        }
    }

    /// Bench-only dispatch wrappers. Exposed via `#[doc(hidden)] pub` so
    /// benches can reach them without promoting the kernels to permanent
    /// public API.
    /// Fused u16 RGB: 256 KB LUT decode → SIMD matrix → SIMD polynomial encode.
    /// sRGB-specific. The production path for sRGB u16 RGB gamut conversion
    /// as of 2026-04-23 — wins matlut (LUT+LUT) at 1080p by +7% and trades
    /// ~8% at smaller sizes for 100% exact u16 roundtrip (matlut's linear-
    /// indexed 128 KB encode LUT has ±6 u16 error; polynomial is ±0).
    pub(crate) fn convert_u16_rgb_simd_lutdec_polyenc(
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        let lin_lut = srgb_lin_lut_u16();
        #[cfg(target_arch = "x86_64")]
        {
            incant!(convert_u16_rgb_lutdec_polyenc(m, src, dst, lin_lut));
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        convert_u16_rgb_lutdec_polyenc_scalar(ScalarToken, m, src, dst, lin_lut);
    }

    #[doc(hidden)]
    pub(crate) fn __bench_u16_lut_decode_lut_encode(
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        convert_u16_rgb_simd_matlut(m, src, dst, srgb_lin_lut_u16(), srgb_enc_lut_u16());
    }

    #[doc(hidden)]
    pub(crate) fn __bench_u16_lut_decode_poly_encode(
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        convert_u16_rgb_simd_lutdec_polyenc(m, src, dst);
    }

    #[doc(hidden)]
    pub(crate) fn __bench_u16_poly_decode_lut_encode(
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        let enc_lut = srgb_enc_lut_u16();
        #[cfg(target_arch = "x86_64")]
        {
            incant!(convert_u16_rgb_polydec_lutenc(m, src, dst, enc_lut));
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        convert_u16_rgb_polydec_lutenc_scalar(ScalarToken, m, src, dst, enc_lut);
    }

    #[doc(hidden)]
    pub(crate) fn __bench_u16_poly_decode_poly_encode(
        m: &[[f32; 3]; 3],
        src: &[u16],
        dst: &mut [u16],
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            incant!(convert_u16_rgb_polydec_polyenc(m, src, dst));
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        convert_u16_rgb_polydec_polyenc_scalar(ScalarToken, m, src, dst);
    }
} // end of hybrids module

#[cfg(feature = "__bench_u16_hybrids")]
pub(crate) use hybrids::{
    __bench_u16_lut_decode_lut_encode, __bench_u16_lut_decode_poly_encode,
    __bench_u16_poly_decode_lut_encode, __bench_u16_poly_decode_poly_encode,
};

// ═════════════════════════════════════════════════════════════════════════
// Cross-depth matlut kernels: integer↔f32 with SIMD matrix. Output f32 is
// *linear* light (not sRGB-encoded). Input f32 is treated as linear light.
// Extended range (negatives, >1) flows through unclamped for f32 outputs;
// u8/u16 outputs implicitly clamp via cvtps_epi32 saturation + LUT gather.
// ═════════════════════════════════════════════════════════════════════════

/// 2-pixel u8-sRGB → linear-f32 kernel. Output is linear (NOT sRGB-encoded);
/// extended range flows through after the matrix.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_u8_to_f32_lin(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8; 24],
    dst: &mut [f32; 24],
    lin_lut: &[f32; 256],
) {
    let m0 = mt_f32x8::from_array(
        token,
        [
            m[0][0], m[1][0], m[2][0], 0.0, m[0][0], m[1][0], m[2][0], 0.0,
        ],
    );
    let m1 = mt_f32x8::from_array(
        token,
        [
            m[0][1], m[1][1], m[2][1], 0.0, m[0][1], m[1][1], m[2][1], 0.0,
        ],
    );
    let m2 = mt_f32x8::from_array(
        token,
        [
            m[0][2], m[1][2], m[2][2], 0.0, m[0][2], m[1][2], m[2][2], 0.0,
        ],
    );
    for pair in 0..4 {
        let off = pair * 6;
        let off_out = pair * 6;
        let p0r = lin_lut[src[off] as usize];
        let p0g = lin_lut[src[off + 1] as usize];
        let p0b = lin_lut[src[off + 2] as usize];
        let p1r = lin_lut[src[off + 3] as usize];
        let p1g = lin_lut[src[off + 4] as usize];
        let p1b = lin_lut[src[off + 5] as usize];
        let r = mt_f32x8::from_array(token, [p0r, p0r, p0r, p0r, p1r, p1r, p1r, p1r]);
        let g = mt_f32x8::from_array(token, [p0g, p0g, p0g, p0g, p1g, p1g, p1g, p1g]);
        let b = mt_f32x8::from_array(token, [p0b, p0b, p0b, p0b, p1b, p1b, p1b, p1b]);
        let v = r * m0 + g * m1 + b * m2;
        let out = v.to_array();
        dst[off_out] = out[0];
        dst[off_out + 1] = out[1];
        dst[off_out + 2] = out[2];
        dst[off_out + 3] = out[4];
        dst[off_out + 4] = out[5];
        dst[off_out + 5] = out[6];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_u8_to_f32_lin_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [f32],
    lin_lut: &[f32; 256],
) {
    let pixel_count = src.len() / 3;
    let bulk = (pixel_count / 8) * 8;
    for off in (0..bulk).step_by(8) {
        let s: &[u8; 24] = src[off * 3..off * 3 + 24].try_into().unwrap();
        let d: &mut [f32; 24] = (&mut dst[off * 3..off * 3 + 24]).try_into().unwrap();
        convert_8px_u8_to_f32_lin(token, m, s, d, lin_lut);
    }
    for i in bulk..pixel_count {
        let base = i * 3;
        let r = lin_lut[src[base] as usize];
        let g = lin_lut[src[base + 1] as usize];
        let b = lin_lut[src[base + 2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst[base] = nr;
        dst[base + 1] = ng;
        dst[base + 2] = nb;
    }
}

fn convert_u8_to_f32_lin_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [f32],
    lin_lut: &[f32; 256],
) {
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = nr;
        dst_px[1] = ng;
        dst_px[2] = nb;
    }
}

/// Fused u8-sRGB → linear-f32 RGB with gamut matrix.
pub(crate) fn convert_u8_to_f32_lin_simd(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [f32],
    lin_lut: &[f32; 256],
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_u8_to_f32_lin(m, src, dst, lin_lut));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_u8_to_f32_lin_scalar(ScalarToken, m, src, dst, lin_lut);
}

/// 2-pixel linear-f32 → u8-sRGB kernel with clamp + cvtps_epi32 + LUT gather.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_f32_lin_to_u8(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[f32; 24],
    dst: &mut [u8; 24],
    enc_lut: &[u8; 4096],
) {
    let m0 = mt_f32x8::from_array(
        token,
        [
            m[0][0], m[1][0], m[2][0], 0.0, m[0][0], m[1][0], m[2][0], 0.0,
        ],
    );
    let m1 = mt_f32x8::from_array(
        token,
        [
            m[0][1], m[1][1], m[2][1], 0.0, m[0][1], m[1][1], m[2][1], 0.0,
        ],
    );
    let m2 = mt_f32x8::from_array(
        token,
        [
            m[0][2], m[1][2], m[2][2], 0.0, m[0][2], m[1][2], m[2][2], 0.0,
        ],
    );
    let scale = mt_f32x8::splat(token, 4095.0);
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);

    for pair in 0..4 {
        let off = pair * 6;
        let r = mt_f32x8::from_array(
            token,
            [
                src[off],
                src[off],
                src[off],
                src[off],
                src[off + 3],
                src[off + 3],
                src[off + 3],
                src[off + 3],
            ],
        );
        let g = mt_f32x8::from_array(
            token,
            [
                src[off + 1],
                src[off + 1],
                src[off + 1],
                src[off + 1],
                src[off + 4],
                src[off + 4],
                src[off + 4],
                src[off + 4],
            ],
        );
        let b = mt_f32x8::from_array(
            token,
            [
                src[off + 2],
                src[off + 2],
                src[off + 2],
                src[off + 2],
                src[off + 5],
                src[off + 5],
                src[off + 5],
                src[off + 5],
            ],
        );
        let v = r * m0 + g * m1 + b * m2;
        let clamped = v.max(zero).min(one);
        let scaled = clamped * scale;
        let idx = scaled.to_i32_round().to_array();
        dst[off] = enc_lut[idx[0] as usize & 0xFFF];
        dst[off + 1] = enc_lut[idx[1] as usize & 0xFFF];
        dst[off + 2] = enc_lut[idx[2] as usize & 0xFFF];
        dst[off + 3] = enc_lut[idx[4] as usize & 0xFFF];
        dst[off + 4] = enc_lut[idx[5] as usize & 0xFFF];
        dst[off + 5] = enc_lut[idx[6] as usize & 0xFFF];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_f32_lin_to_u8_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[f32],
    dst: &mut [u8],
    enc_lut: &[u8; 4096],
) {
    let pixel_count = src.len() / 3;
    let bulk = (pixel_count / 8) * 8;
    for off in (0..bulk).step_by(8) {
        let s: &[f32; 24] = src[off * 3..off * 3 + 24].try_into().unwrap();
        let d: &mut [u8; 24] = (&mut dst[off * 3..off * 3 + 24]).try_into().unwrap();
        convert_8px_f32_lin_to_u8(token, m, s, d, enc_lut);
    }
    for i in bulk..pixel_count {
        let base = i * 3;
        let (nr, ng, nb) = mat3x3(m, src[base], src[base + 1], src[base + 2]);
        dst[base] = enc_lut[(nr.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
        dst[base + 1] = enc_lut[(ng.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
        dst[base + 2] = enc_lut[(nb.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
    }
}

fn convert_f32_lin_to_u8_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[f32],
    dst: &mut [u8],
    enc_lut: &[u8; 4096],
) {
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let (nr, ng, nb) = mat3x3(m, src_px[0], src_px[1], src_px[2]);
        dst_px[0] = enc_lut[(nr.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
        dst_px[1] = enc_lut[(ng.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
        dst_px[2] = enc_lut[(nb.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF];
    }
}

/// Fused linear-f32 → u8-sRGB RGB with gamut matrix (always clamps).
pub(crate) fn convert_f32_lin_to_u8_simd(
    m: &[[f32; 3]; 3],
    src: &[f32],
    dst: &mut [u8],
    enc_lut: &[u8; 4096],
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_f32_lin_to_u8(m, src, dst, enc_lut));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_f32_lin_to_u8_scalar(ScalarToken, m, src, dst, enc_lut);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_u8_rgb_matlut_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    let pixel_count = src.len() / 3;
    let bulk = (pixel_count / 8) * 8;
    let bulk_bytes = bulk * 3;
    let enc_lut = srgb_enc_lut_4096();
    for off in (0..bulk_bytes).step_by(24) {
        let s: &[u8; 24] = src[off..off + 24].try_into().unwrap();
        let d: &mut [u8; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
        convert_8px_u8_rgb_matlut(token, m, s, d, lin_lut, enc_lut);
    }
    for i in bulk..pixel_count {
        let base = i * 3;
        let r = lin_lut[src[base] as usize];
        let g = lin_lut[src[base + 1] as usize];
        let b = lin_lut[src[base + 2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst[base] = enc_u8(nr);
        dst[base + 1] = enc_u8(ng);
        dst[base + 2] = enc_u8(nb);
    }
}

fn convert_u8_rgb_matlut_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = enc_u8(nr);
        dst_px[1] = enc_u8(ng);
        dst_px[2] = enc_u8(nb);
    }
}

/// Fused u8 RGB: LUT linearize → SIMD matrix → SIMD f32→u8 via sRGB LUT.
/// Currently sRGB-specific (uses `linear_srgb::linear_to_srgb_u8_v3`).
/// Caller must ensure `enc_u8` is the sRGB encoder for correct remainder.
pub(crate) fn convert_u8_rgb_simd_matlut(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_u8_rgb_matlut(m, src, dst, lin_lut, enc_u8));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_u8_rgb_matlut_scalar(ScalarToken, m, src, dst, lin_lut, enc_u8);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_u8_rgb_fused_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    dst_trc: TransferFunction,
    scalar_enc: fn(f32) -> f32,
) {
    let pixel_count = src.len() / 3;
    let bulk = (pixel_count / 8) * 8;
    let bulk_bytes = bulk * 3;
    for off in (0..bulk_bytes).step_by(24) {
        let s: &[u8; 24] = src[off..off + 24].try_into().unwrap();
        let d: &mut [u8; 24] = (&mut dst[off..off + 24]).try_into().unwrap();
        convert_8px_u8_rgb_fused(token, m, s, d, lin_lut, dst_trc);
    }
    for i in bulk..pixel_count {
        let base = i * 3;
        let r = lin_lut[src[base] as usize];
        let g = lin_lut[src[base + 1] as usize];
        let b = lin_lut[src[base + 2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst[base] = (scalar_enc(nr) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst[base + 1] = (scalar_enc(ng) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst[base + 2] = (scalar_enc(nb) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

fn convert_u8_rgb_fused_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    _dst_trc: TransferFunction,
    scalar_enc: fn(f32) -> f32,
) {
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = (scalar_enc(nr) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[1] = (scalar_enc(ng) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[2] = (scalar_enc(nb) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Fused u8 RGB gamut conversion: LUT linearize → SIMD (matrix + polynomial encode) → quantize.
pub(crate) fn convert_u8_rgb_simd_fused(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    dst_trc: TransferFunction,
    scalar_enc: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_u8_rgb_fused(
            m, src, dst, lin_lut, dst_trc, scalar_enc
        ));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_u8_rgb_fused_scalar(ScalarToken, m, src, dst, lin_lut, dst_trc, scalar_enc);
}

/// SIMD-batched u8 RGBA: LUT linearize 8 pixels → SIMD matrix → LUT encode. Alpha copied.
#[cfg(target_arch = "x86_64")]
#[rite]
fn convert_8px_u8_rgba_simd(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        r[i] = lin_lut[src[i * 4] as usize];
        g[i] = lin_lut[src[i * 4 + 1] as usize];
        b[i] = lin_lut[src[i * 4 + 2] as usize];
    }
    let rv = mt_f32x8::from_array(token, r);
    let gv = mt_f32x8::from_array(token, g);
    let bv = mt_f32x8::from_array(token, b);
    let (or, og, ob) = mat3x3_x8(token, m, rv, gv, bv);
    let ro = or.to_array();
    let go = og.to_array();
    let bo = ob.to_array();
    for i in 0..8 {
        dst[i * 4] = enc_u8(ro[i]);
        dst[i * 4 + 1] = enc_u8(go[i]);
        dst[i * 4 + 2] = enc_u8(bo[i]);
        dst[i * 4 + 3] = src[i * 4 + 3]; // alpha passthrough
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn convert_u8_rgba_lut_simd_v3(
    token: X64V3Token,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    let pixel_count = src.len() / 4;
    let bulk = (pixel_count / 8) * 8;
    let bulk_bytes = bulk * 4;
    for off in (0..bulk_bytes).step_by(32) {
        convert_8px_u8_rgba_simd(
            token,
            m,
            &src[off..off + 32],
            &mut dst[off..off + 32],
            lin_lut,
            enc_u8,
        );
    }
    for i in bulk..pixel_count {
        let base = i * 4;
        let r = lin_lut[src[base] as usize];
        let g = lin_lut[src[base + 1] as usize];
        let b = lin_lut[src[base + 2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst[base] = enc_u8(nr);
        dst[base + 1] = enc_u8(ng);
        dst[base + 2] = enc_u8(nb);
        dst[base + 3] = src[base + 3];
    }
}

fn convert_u8_rgba_lut_simd_scalar(
    _token: ScalarToken,
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let r = lin_lut[src_px[0] as usize];
        let g = lin_lut[src_px[1] as usize];
        let b = lin_lut[src_px[2] as usize];
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = enc_u8(nr);
        dst_px[1] = enc_u8(ng);
        dst_px[2] = enc_u8(nb);
        dst_px[3] = src_px[3];
    }
}

/// SIMD-batched u8 RGBA gamut conversion. Alpha copied.
pub(crate) fn convert_u8_rgba_simd_lut(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    lin_lut: &[f32; 256],
    enc_u8: fn(f32) -> u8,
) {
    debug_assert_eq!(src.len() % 4, 0);
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        incant!(convert_u8_rgba_lut_simd(m, src, dst, lin_lut, enc_u8));
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    convert_u8_rgba_lut_simd_scalar(ScalarToken, m, src, dst, lin_lut, enc_u8);
}

/// Convert u8 RGBA source to u8 RGBA dest via gamut conversion. Alpha copied.
pub(crate) fn convert_u8_rgba(
    m: &[[f32; 3]; 3],
    src: &[u8],
    dst: &mut [u8],
    linearize_fn: fn(f32) -> f32,
    encode_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 4, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let r = linearize_fn(src_px[0] as f32 / 255.0);
        let g = linearize_fn(src_px[1] as f32 / 255.0);
        let b = linearize_fn(src_px[2] as f32 / 255.0);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = (encode_fn(nr) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[1] = (encode_fn(ng) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[2] = (encode_fn(nb) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        dst_px[3] = src_px[3];
    }
}

/// Convert u16 RGB source to u16 RGB dest via gamut conversion.
pub(crate) fn convert_u16_rgb(
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    linearize_fn: fn(f32) -> f32,
    encode_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = linearize_fn(src_px[0] as f32 / 65535.0);
        let g = linearize_fn(src_px[1] as f32 / 65535.0);
        let b = linearize_fn(src_px[2] as f32 / 65535.0);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        dst_px[0] = (encode_fn(nr) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[1] = (encode_fn(ng) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[2] = (encode_fn(nb) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorPrimaries;

    fn m(src: ColorPrimaries, dst: ColorPrimaries) -> [[f32; 3]; 3] {
        src.gamut_matrix_to(dst).unwrap()
    }

    // --- Dispatch: white/black/roundtrip via convert_f32_rgb_dispatch ---

    #[test]
    fn dispatch_p3_srgb_white() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut px = [1.0f32, 1.0, 1.0];
        convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );
        for c in &px {
            assert!((c - 1.0).abs() < 1e-4, "white: {px:?}");
        }
    }

    #[test]
    fn dispatch_p3_srgb_black() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut px = [0.0f32, 0.0, 0.0];
        convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );
        for c in &px {
            assert!(c.abs() < 1e-6, "black: {px:?}");
        }
    }

    #[test]
    fn dispatch_p3_srgb_roundtrip() {
        let fwd = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let inv = m(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3);
        let original = [0.5f32, 0.3, 0.7];
        let mut px = original;
        convert_f32_rgb_dispatch(
            &fwd,
            &mut px,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );
        convert_f32_rgb_dispatch(
            &inv,
            &mut px,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );
        for i in 0..3 {
            assert!((original[i] - px[i]).abs() < 1e-4, "ch{i}: {}", px[i]);
        }
    }

    #[test]
    fn dispatch_bt2020_sdr_srgb_white() {
        let mat = m(ColorPrimaries::Bt2020, ColorPrimaries::Bt709);
        let mut px = [1.0f32, 1.0, 1.0];
        convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Bt709,
            TransferFunction::Srgb,
        );
        for c in &px {
            assert!((c - 1.0).abs() < 1e-4, "white: {px:?}");
        }
    }

    #[test]
    fn dispatch_bt2020_pq_srgb_black() {
        let mat = m(ColorPrimaries::Bt2020, ColorPrimaries::Bt709);
        let mut px = [0.0f32, 0.0, 0.0];
        convert_f32_rgb_dispatch(&mat, &mut px, TransferFunction::Pq, TransferFunction::Srgb);
        for c in &px {
            assert!(c.abs() < 1e-5, "black: {px:?}");
        }
    }

    #[test]
    fn dispatch_adobe_srgb_white() {
        let mat = m(ColorPrimaries::AdobeRgb, ColorPrimaries::Bt709);
        let mut px = [1.0f32, 1.0, 1.0];
        convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Gamma22,
            TransferFunction::Srgb,
        );
        for c in &px {
            assert!((c - 1.0).abs() < 1e-4, "white: {px:?}");
        }
    }

    #[test]
    fn dispatch_rgba_alpha_passthrough() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut px = [0.5f32, 0.5, 0.5, 0.7];
        convert_f32_rgba_dispatch(
            &mat,
            &mut px,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );
        assert!((px[3] - 0.7).abs() < f32::EPSILON, "alpha: {px:?}");
    }

    #[test]
    fn dispatch_linear_white() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut px = [1.0f32, 1.0, 1.0];
        convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Linear,
            TransferFunction::Linear,
        );
        for c in &px {
            assert!((c - 1.0).abs() < 1e-6, "linear white: {px:?}");
        }
    }

    #[test]
    fn dispatch_returns_false_for_unknown_trc() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut px = [0.5f32, 0.5, 0.5];
        assert!(!convert_f32_rgb_dispatch(
            &mat,
            &mut px,
            TransferFunction::Unknown,
            TransferFunction::Srgb
        ));
    }

    // --- u8/u16 via internal helpers ---

    #[test]
    fn u8_white_black() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut dst = [0u8; 3];
        convert_u8_rgb(
            &mat,
            &[255, 255, 255],
            &mut dst,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        assert_eq!(dst, [255, 255, 255]);
        convert_u8_rgb(
            &mat,
            &[0, 0, 0],
            &mut dst,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        assert_eq!(dst, [0, 0, 0]);
    }

    #[test]
    fn u8_lut_lut_matches_scalar() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let lut = build_linearize_lut(linear_srgb::tf::srgb_to_linear);
        let enc_u8 = scalar_encode_u8(TransferFunction::Srgb).unwrap();
        let src = [128u8, 64, 200];
        let mut dst_scalar = [0u8; 3];
        let mut dst_lut = [0u8; 3];
        convert_u8_rgb(
            &mat,
            &src,
            &mut dst_scalar,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        convert_u8_rgb_lut_lut(&mat, &src, &mut dst_lut, &lut, enc_u8);
        assert_eq!(
            dst_scalar, dst_lut,
            "LUT-LUT and scalar should produce identical u8 output"
        );
    }

    #[test]
    fn u8_rgba_alpha_passthrough() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut dst = [0u8; 4];
        convert_u8_rgba(
            &mat,
            &[128, 64, 32, 200],
            &mut dst,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        assert_eq!(dst[3], 200);
    }

    #[test]
    fn u16_white_black() {
        let mat = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut dst = [0u16; 3];
        convert_u16_rgb(
            &mat,
            &[65535, 65535, 65535],
            &mut dst,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        assert_eq!(dst, [65535, 65535, 65535]);
        convert_u16_rgb(
            &mat,
            &[0, 0, 0],
            &mut dst,
            linear_srgb::tf::srgb_to_linear,
            linear_srgb::tf::linear_to_srgb,
        );
        assert_eq!(dst, [0, 0, 0]);
    }

    // --- Roundtrip accuracy ---

    #[test]
    fn f32_roundtrip_accuracy() {
        let fwd = m(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3);
        let inv = m(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709);
        let mut max_err: f32 = 0.0;
        for r in (0..=255).step_by(4) {
            for g in (0..=255).step_by(4) {
                for b in (0..=255).step_by(16) {
                    let original = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    let mut px = original;
                    convert_f32_rgb_dispatch(
                        &fwd,
                        &mut px,
                        TransferFunction::Srgb,
                        TransferFunction::Srgb,
                    );
                    convert_f32_rgb_dispatch(
                        &inv,
                        &mut px,
                        TransferFunction::Srgb,
                        TransferFunction::Srgb,
                    );
                    for i in 0..3 {
                        let err = (original[i] - px[i]).abs();
                        if err > max_err {
                            max_err = err;
                        }
                    }
                }
            }
        }
        assert!(max_err < 1e-4, "max roundtrip error: {max_err}");
    }

    // --- sRGB subset of wider gamuts ---

    #[test]
    fn srgb_subset_of_p3() {
        let mat = m(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3);
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let mut px = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    convert_f32_rgb_dispatch(
                        &mat,
                        &mut px,
                        TransferFunction::Srgb,
                        TransferFunction::Srgb,
                    );
                    for (i, c) in px.iter().enumerate() {
                        assert!(*c >= -1e-5 && *c <= 1.0 + 1e-5, "({r},{g},{b}) ch{i}: {c}");
                    }
                }
            }
        }
    }
}
