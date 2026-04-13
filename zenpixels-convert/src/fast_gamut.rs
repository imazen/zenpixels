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
use linear_srgb::tokens::x8 as trc_x8;
use magetypes::simd::f32x8 as mt_f32x8;

use crate::ColorPrimaries;

// =========================================================================
// Gamut matrices — computed at compile time via `ColorPrimaries::gamut_matrix_to`.
// Bradford chromatic adaptation is applied automatically when white points differ.
// =========================================================================

macro_rules! const_gamut_matrix {
    ($name:ident, $src:ident, $dst:ident, $doc:expr) => {
        #[doc = $doc]
        pub const $name: [[f32; 3]; 3] =
            match ColorPrimaries::$src.gamut_matrix_to(ColorPrimaries::$dst) {
                Some(m) => m,
                None => panic!(concat!(
                    "failed to compute ",
                    stringify!($src),
                    " → ",
                    stringify!($dst),
                    " matrix"
                )),
            };
    };
}

const_gamut_matrix!(
    SRGB_TO_P3,
    Bt709,
    DisplayP3,
    "sRGB linear → Display P3 linear."
);
const_gamut_matrix!(
    SRGB_TO_BT2020,
    Bt709,
    Bt2020,
    "sRGB linear → BT.2020 linear."
);
const_gamut_matrix!(
    P3_TO_SRGB,
    DisplayP3,
    Bt709,
    "Display P3 linear → sRGB linear."
);
const_gamut_matrix!(
    P3_TO_BT2020,
    DisplayP3,
    Bt2020,
    "Display P3 linear → BT.2020 linear."
);
const_gamut_matrix!(
    BT2020_TO_SRGB,
    Bt2020,
    Bt709,
    "BT.2020 linear → sRGB linear."
);
const_gamut_matrix!(
    BT2020_TO_P3,
    Bt2020,
    DisplayP3,
    "BT.2020 linear → Display P3 linear."
);

const_gamut_matrix!(
    ADOBERGB_TO_SRGB,
    AdobeRgb,
    Bt709,
    "Adobe RGB (1998) linear → sRGB linear."
);
const_gamut_matrix!(
    SRGB_TO_ADOBERGB,
    Bt709,
    AdobeRgb,
    "sRGB linear → Adobe RGB (1998) linear."
);
const_gamut_matrix!(
    ADOBERGB_TO_P3,
    AdobeRgb,
    DisplayP3,
    "Adobe RGB (1998) linear → Display P3 linear."
);
const_gamut_matrix!(
    P3_TO_ADOBERGB,
    DisplayP3,
    AdobeRgb,
    "Display P3 linear → Adobe RGB (1998) linear."
);
const_gamut_matrix!(
    ADOBERGB_TO_BT2020,
    AdobeRgb,
    Bt2020,
    "Adobe RGB (1998) linear → BT.2020 linear."
);
const_gamut_matrix!(
    BT2020_TO_ADOBERGB,
    Bt2020,
    AdobeRgb,
    "BT.2020 linear → Adobe RGB (1998) linear."
);

const_gamut_matrix!(
    DCIP3_TO_SRGB,
    DciP3,
    Bt709,
    "DCI-P3 (D50) linear → sRGB linear (Bradford adaptation)."
);
const_gamut_matrix!(
    SRGB_TO_DCIP3,
    Bt709,
    DciP3,
    "sRGB linear → DCI-P3 (D50) linear (Bradford adaptation)."
);
const_gamut_matrix!(
    DCIP3_TO_P3,
    DciP3,
    DisplayP3,
    "DCI-P3 (D50) linear → Display P3 (D65) linear (Bradford adaptation)."
);
const_gamut_matrix!(
    P3_TO_DCIP3,
    DisplayP3,
    DciP3,
    "Display P3 (D65) linear → DCI-P3 (D50) linear (Bradford adaptation)."
);
const_gamut_matrix!(
    DCIP3_TO_BT2020,
    DciP3,
    Bt2020,
    "DCI-P3 (D50) linear → BT.2020 linear (Bradford adaptation)."
);
const_gamut_matrix!(
    BT2020_TO_DCIP3,
    Bt2020,
    DciP3,
    "BT.2020 linear → DCI-P3 (D50) linear (Bradford adaptation)."
);

// =========================================================================
// Shared helpers
// =========================================================================

/// Apply a 3×3 matrix to an RGB triple (public entry point).
#[inline(always)]
pub fn mat3x3_pub(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    mat3x3(m, r, g, b)
}

/// Apply a 3×3 matrix to an RGB triple.
#[inline(always)]
fn mat3x3(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b)),
        m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b)),
        m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b)),
    )
}

/// SIMD matrix multiply: 3 channels × 8 pixels via FMA.
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

#[rite]
fn adobe_to_linear_x8(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    trc_x8::gamma_to_linear_v3(token, v, ADOBE_GAMMA)
}

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

// Adobe RGB same-TRC (both sides gamma 2.2)
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
// DCI-P3 gamma 2.6 TRC
// =========================================================================

const DCI_GAMMA: f32 = 2.6;

#[rite]
fn dci_to_linear_x8(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    trc_x8::gamma_to_linear_v3(token, v, DCI_GAMMA)
}

#[rite]
fn dci_from_linear_x8(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    trc_x8::linear_to_gamma_v3(token, v, DCI_GAMMA)
}

#[inline(always)]
fn dci_to_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::gamma_to_linear(v, DCI_GAMMA)
}

#[inline(always)]
fn dci_from_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::linear_to_gamma(v, DCI_GAMMA)
}

// DCI-P3 same-TRC (γ2.6 both sides — for DCI↔DCI with different primaries, unlikely)
stamp_trc_kernels!(dci,
    simd_linearize: dci_to_linear_x8,
    simd_encode: dci_from_linear_x8,
    scalar_linearize: dci_to_linear_scalar,
    scalar_encode: dci_from_linear_scalar
);

// DCI-P3 source → sRGB dest (γ2.6 linearize, sRGB encode)
stamp_trc_kernels!(dci_to_srgb,
    simd_linearize: dci_to_linear_x8,
    simd_encode: trc_x8::linear_to_srgb_v3,
    scalar_linearize: dci_to_linear_scalar,
    scalar_encode: linear_srgb::tf::linear_to_srgb
);

// sRGB source → DCI-P3 dest (sRGB linearize, γ2.6 encode)
stamp_trc_kernels!(srgb_to_dci,
    simd_linearize: trc_x8::srgb_to_linear_v3,
    simd_encode: dci_from_linear_x8,
    scalar_linearize: linear_srgb::tf::srgb_to_linear,
    scalar_encode: dci_from_linear_scalar
);

// =========================================================================
// Public API — named conversions
// =========================================================================

// --- P3 ↔ sRGB (same TRC: sRGB curve) ---

/// Display P3 → sRGB, f32 RGB in-place. Clamped to [0,1].
pub fn p3_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb(&P3_TO_SRGB, data));
}
/// sRGB → Display P3, f32 RGB in-place. Clamped to [0,1].
pub fn srgb_to_p3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb(&SRGB_TO_P3, data));
}
/// Display P3 → sRGB, f32 RGBA in-place. Alpha unchanged.
pub fn p3_to_srgb_f32_rgba(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_srgb(&P3_TO_SRGB, data));
}
/// sRGB → Display P3, f32 RGBA in-place. Alpha unchanged.
pub fn srgb_to_p3_f32_rgba(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_srgb(&SRGB_TO_P3, data));
}

// --- BT.2020 SDR ↔ sRGB (cross-TRC: BT.709 ↔ sRGB) ---

/// BT.2020 SDR → sRGB, f32 RGB in-place.
pub fn bt2020_sdr_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_bt709_to_srgb(&BT2020_TO_SRGB, data));
}
/// sRGB → BT.2020 SDR, f32 RGB in-place.
pub fn srgb_to_bt2020_sdr_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_bt709(&SRGB_TO_BT2020, data));
}

// --- BT.2020 PQ ↔ sRGB (cross-TRC: PQ ↔ sRGB) ---

/// BT.2020 PQ → sRGB, f32 RGB in-place. **No tone mapping** — values may clip.
pub fn bt2020_pq_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_pq_to_srgb(&BT2020_TO_SRGB, data));
}
/// sRGB → BT.2020 PQ, f32 RGB in-place.
pub fn srgb_to_bt2020_pq_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_pq(&SRGB_TO_BT2020, data));
}

// --- BT.2020 HLG ↔ sRGB (cross-TRC: HLG ↔ sRGB) ---

/// BT.2020 HLG → sRGB, f32 RGB in-place. **No tone mapping** — values may clip.
pub fn bt2020_hlg_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_hlg_to_srgb(&BT2020_TO_SRGB, data));
}

// --- P3 ↔ BT.2020 (same TRC for SDR: sRGB curve) ---

/// Display P3 → BT.2020, f32 RGB in-place (sRGB TRC on both sides).
pub fn p3_to_bt2020_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb(&P3_TO_BT2020, data));
}
/// BT.2020 → Display P3, f32 RGB in-place (sRGB TRC on both sides).
pub fn bt2020_to_p3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb(&BT2020_TO_P3, data));
}

// --- Adobe RGB ↔ sRGB (cross-TRC: gamma 2.2 ↔ sRGB) ---

/// Adobe RGB → sRGB, f32 RGB in-place.
pub fn adobergb_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_adobe_to_srgb(&ADOBERGB_TO_SRGB, data));
}
/// sRGB → Adobe RGB, f32 RGB in-place.
pub fn srgb_to_adobergb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_adobe(&SRGB_TO_ADOBERGB, data));
}
/// Adobe RGB → sRGB, f32 RGBA in-place. Alpha unchanged.
pub fn adobergb_to_srgb_f32_rgba(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_adobe_to_srgb(&ADOBERGB_TO_SRGB, data));
}

// --- Adobe RGB ↔ P3 (cross-TRC: gamma 2.2 ↔ sRGB) ---

/// Adobe RGB → Display P3, f32 RGB in-place.
pub fn adobergb_to_p3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_adobe_to_srgb(&ADOBERGB_TO_P3, data));
}
/// Display P3 → Adobe RGB, f32 RGB in-place.
pub fn p3_to_adobergb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_adobe(&P3_TO_ADOBERGB, data));
}

// --- Adobe RGB ↔ BT.2020 ---

/// Adobe RGB → BT.2020, f32 RGB in-place (gamma 2.2 both sides).
pub fn adobergb_to_bt2020_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_adobe(&ADOBERGB_TO_BT2020, data));
}

// --- Adobe RGB u8 ---

/// Adobe RGB → sRGB, u8 RGB → u8 RGB.
pub fn adobergb_to_srgb_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &ADOBERGB_TO_SRGB,
        src,
        dst,
        adobe_to_linear_scalar,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// sRGB → Adobe RGB, u8 RGB → u8 RGB.
pub fn srgb_to_adobergb_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &SRGB_TO_ADOBERGB,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        adobe_from_linear_scalar,
    );
}

// --- DCI-P3 ↔ sRGB (cross-TRC: γ2.6 ↔ sRGB, Bradford adaptation) ---

/// DCI-P3 → sRGB, f32 RGB in-place.
pub fn dcip3_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_dci_to_srgb(&DCIP3_TO_SRGB, data));
}
/// sRGB → DCI-P3, f32 RGB in-place.
pub fn srgb_to_dcip3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_dci(&SRGB_TO_DCIP3, data));
}

// --- DCI-P3 ↔ Display P3 (cross-TRC: γ2.6 ↔ sRGB, Bradford adaptation) ---

/// DCI-P3 → Display P3, f32 RGB in-place.
pub fn dcip3_to_p3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_dci_to_srgb(&DCIP3_TO_P3, data));
}
/// Display P3 → DCI-P3, f32 RGB in-place.
pub fn p3_to_dcip3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb_to_dci(&P3_TO_DCIP3, data));
}

// --- DCI-P3 ↔ BT.2020 ---

/// DCI-P3 → BT.2020, f32 RGB in-place (γ2.6 → BT.709 TRC).
pub fn dcip3_to_bt2020_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    // DCI γ2.6 linearize → matrix → BT.709 encode
    // Need a dci_to_bt709 stamp... use dci linearize + bt709 encode
    // For now, go through linear: linearize, matrix, encode
    for pixel in data.chunks_exact_mut(3) {
        let r = dci_to_linear_scalar(pixel[0]);
        let g = dci_to_linear_scalar(pixel[1]);
        let b = dci_to_linear_scalar(pixel[2]);
        let (nr, ng, nb) = mat3x3(&DCIP3_TO_BT2020, r, g, b);
        pixel[0] = linear_srgb::tf::linear_to_bt709(nr);
        pixel[1] = linear_srgb::tf::linear_to_bt709(ng);
        pixel[2] = linear_srgb::tf::linear_to_bt709(nb);
    }
}

// --- Generic: any matrix + any TRC (advanced users) ---

/// Generic conversion with sRGB TRC and custom matrix, f32 RGB in-place.
pub fn convert_srgb_trc_rgb(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_srgb(m, data));
}

/// Generic conversion with sRGB TRC and custom matrix, f32 RGBA in-place.
pub fn convert_srgb_trc_rgba(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_srgb(m, data));
}

// =========================================================================
// Dispatch: pick the right kernel for a given (src_trc, dst_trc) pair
// =========================================================================

use crate::TransferFunction;

/// Scalar linearization function for a given transfer function.
pub fn scalar_linearize(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(linear_srgb::tf::srgb_to_linear),
        TransferFunction::Bt709 => Some(linear_srgb::tf::bt709_to_linear),
        TransferFunction::Pq => Some(linear_srgb::tf::pq_to_linear),
        TransferFunction::Hlg => Some(linear_srgb::tf::hlg_to_linear),
        TransferFunction::Gamma22 => Some(adobe_to_linear_scalar),
        TransferFunction::Gamma26 => Some(dci_to_linear_scalar),
        TransferFunction::Linear => Some(core::convert::identity),
        _ => None,
    }
}

/// Scalar encode function for a given transfer function.
pub fn scalar_encode(trc: TransferFunction) -> Option<fn(f32) -> f32> {
    match trc {
        TransferFunction::Srgb => Some(linear_srgb::tf::linear_to_srgb),
        TransferFunction::Bt709 => Some(linear_srgb::tf::linear_to_bt709),
        TransferFunction::Pq => Some(linear_srgb::tf::linear_to_pq),
        TransferFunction::Hlg => Some(linear_srgb::tf::linear_to_hlg),
        TransferFunction::Gamma22 => Some(adobe_from_linear_scalar),
        TransferFunction::Gamma26 => Some(dci_from_linear_scalar),
        TransferFunction::Linear => Some(core::convert::identity),
        _ => None,
    }
}

/// Convert f32 RGB data in-place using the given gamut matrix and TRC pair.
///
/// Dispatches to fused SIMD kernels when a specialized kernel exists for the
/// (src_trc, dst_trc) pair. Falls back to scalar linearize → matrix → encode
/// for unsupported pairs. Returns `false` if either TRC is unknown.
pub fn convert_f32_rgb_dispatch(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 3, 0);
    match (src_trc, dst_trc) {
        // Same-TRC: single fused kernel
        (Srgb, Srgb) => incant!(convert_rgb_srgb(m, data)),
        (Bt709, Bt709) => incant!(convert_rgb_bt709(m, data)),
        (Pq, Pq) => incant!(convert_rgb_pq(m, data)),
        (Hlg, Hlg) => incant!(convert_rgb_hlg(m, data)),
        (Gamma22, Gamma22) => incant!(convert_rgb_adobe(m, data)),
        (Gamma26, Gamma26) => incant!(convert_rgb_dci(m, data)),
        (Linear, Linear) => convert_linear_rgb(m, data),
        // Cross-TRC: specialized kernels
        (Pq, Srgb) => incant!(convert_rgb_pq_to_srgb(m, data)),
        (Hlg, Srgb) => incant!(convert_rgb_hlg_to_srgb(m, data)),
        (Srgb, Pq) => incant!(convert_rgb_srgb_to_pq(m, data)),
        (Bt709, Srgb) => incant!(convert_rgb_bt709_to_srgb(m, data)),
        (Srgb, Bt709) => incant!(convert_rgb_srgb_to_bt709(m, data)),
        (Gamma22, Srgb) => incant!(convert_rgb_adobe_to_srgb(m, data)),
        (Srgb, Gamma22) => incant!(convert_rgb_srgb_to_adobe(m, data)),
        (Gamma26, Srgb) => incant!(convert_rgb_dci_to_srgb(m, data)),
        (Srgb, Gamma26) => incant!(convert_rgb_srgb_to_dci(m, data)),
        // Fallback: scalar path for any other TRC pair
        _ => {
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
            return true;
        }
    }
    true
}

/// Convert f32 RGBA data in-place using the given gamut matrix and TRC pair.
/// Alpha channel is preserved unchanged.
pub fn convert_f32_rgba_dispatch(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 4, 0);
    match (src_trc, dst_trc) {
        (Srgb, Srgb) => incant!(convert_rgba_srgb(m, data)),
        (Bt709, Bt709) => incant!(convert_rgba_bt709(m, data)),
        (Pq, Pq) => incant!(convert_rgba_pq(m, data)),
        (Hlg, Hlg) => incant!(convert_rgba_hlg(m, data)),
        (Gamma22, Gamma22) => incant!(convert_rgba_adobe(m, data)),
        (Gamma26, Gamma26) => incant!(convert_rgba_dci(m, data)),
        (Linear, Linear) => convert_linear_rgba(m, data),
        (Pq, Srgb) => incant!(convert_rgba_pq_to_srgb(m, data)),
        (Hlg, Srgb) => incant!(convert_rgba_hlg_to_srgb(m, data)),
        (Srgb, Pq) => incant!(convert_rgba_srgb_to_pq(m, data)),
        (Bt709, Srgb) => incant!(convert_rgba_bt709_to_srgb(m, data)),
        (Srgb, Bt709) => incant!(convert_rgba_srgb_to_bt709(m, data)),
        (Gamma22, Srgb) => incant!(convert_rgba_adobe_to_srgb(m, data)),
        (Srgb, Gamma22) => incant!(convert_rgba_srgb_to_adobe(m, data)),
        (Gamma26, Srgb) => incant!(convert_rgba_dci_to_srgb(m, data)),
        (Srgb, Gamma26) => incant!(convert_rgba_srgb_to_dci(m, data)),
        _ => {
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
            return true;
        }
    }
    true
}

// =========================================================================
// Extended range (sign-preserving scalar TRC, no clamping)
// =========================================================================

/// Linearize with sign-preserving sRGB TRC (extended range, scalar `powf`).
#[inline(always)]
fn linearize_extended(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::srgb_to_linear_extended(v)
    } else {
        -linear_srgb::precise::srgb_to_linear_extended(-v)
    }
}

/// Encode with sign-preserving sRGB TRC (extended range, scalar `powf`).
#[inline(always)]
fn encode_extended(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::linear_to_srgb_extended(v)
    } else {
        -linear_srgb::precise::linear_to_srgb_extended(-v)
    }
}

/// Display P3 → sRGB, f32 RGB in-place, **extended range** (no clamping).
/// Slower than clamped variant due to scalar `powf`.
pub fn p3_to_srgb_f32_extended(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let r = linearize_extended(pixel[0]);
        let g = linearize_extended(pixel[1]);
        let b = linearize_extended(pixel[2]);
        let (sr, sg, sb) = mat3x3(&P3_TO_SRGB, r, g, b);
        pixel[0] = encode_extended(sr);
        pixel[1] = encode_extended(sg);
        pixel[2] = encode_extended(sb);
    }
}

/// sRGB → Display P3, f32 RGB in-place, **extended range**.
pub fn srgb_to_p3_f32_extended(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let r = linearize_extended(pixel[0]);
        let g = linearize_extended(pixel[1]);
        let b = linearize_extended(pixel[2]);
        let (pr, pg, pb) = mat3x3(&SRGB_TO_P3, r, g, b);
        pixel[0] = encode_extended(pr);
        pixel[1] = encode_extended(pg);
        pixel[2] = encode_extended(pb);
    }
}

// =========================================================================
// u8 ↔ f32 wrappers
// =========================================================================

/// Convert u8 RGB source to u8 RGB dest via gamut conversion.
///
/// Source u8 values are normalized to [0,1], linearized, matrix-transformed,
/// then re-encoded and quantized to u8 output.
pub fn convert_u8_rgb(
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

/// Convert u8 RGBA source to u8 RGBA dest via gamut conversion. Alpha copied.
pub fn convert_u8_rgba(
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
pub fn convert_u16_rgb(
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

// --- u8 named conversions ---

/// Display P3 → sRGB, u8 RGB → u8 RGB.
pub fn p3_to_srgb_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &P3_TO_SRGB,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// sRGB → Display P3, u8 RGB → u8 RGB.
pub fn srgb_to_p3_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &SRGB_TO_P3,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// Display P3 → sRGB, u8 RGBA → u8 RGBA. Alpha copied.
pub fn p3_to_srgb_u8_rgba(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgba(
        &P3_TO_SRGB,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// sRGB → Display P3, u8 RGBA → u8 RGBA. Alpha copied.
pub fn srgb_to_p3_u8_rgba(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgba(
        &SRGB_TO_P3,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// BT.2020 SDR → sRGB, u8 RGB → u8 RGB.
pub fn bt2020_sdr_to_srgb_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &BT2020_TO_SRGB,
        src,
        dst,
        linear_srgb::tf::bt709_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// sRGB → BT.2020 SDR, u8 RGB → u8 RGB.
pub fn srgb_to_bt2020_sdr_u8_rgb(src: &[u8], dst: &mut [u8]) {
    convert_u8_rgb(
        &SRGB_TO_BT2020,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_bt709,
    );
}

// --- u16 named conversions ---

/// Display P3 → sRGB, u16 RGB → u16 RGB.
pub fn p3_to_srgb_u16_rgb(src: &[u16], dst: &mut [u16]) {
    convert_u16_rgb(
        &P3_TO_SRGB,
        src,
        dst,
        linear_srgb::tf::srgb_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

/// BT.2020 PQ → sRGB, u16 RGB → u16 RGB. **No tone mapping**.
pub fn bt2020_pq_to_srgb_u16_rgb(src: &[u16], dst: &mut [u16]) {
    convert_u16_rgb(
        &BT2020_TO_SRGB,
        src,
        dst,
        linear_srgb::tf::pq_to_linear,
        linear_srgb::tf::linear_to_srgb,
    );
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_white_roundtrip(convert: fn(&mut [f32])) {
        let mut px = [1.0_f32, 1.0, 1.0];
        convert(&mut px);
        for c in &px {
            assert!((c - 1.0).abs() < 1e-4, "white should map to white: {px:?}");
        }
    }

    fn assert_black_roundtrip(convert: fn(&mut [f32])) {
        let mut px = [0.0_f32, 0.0, 0.0];
        convert(&mut px);
        for c in &px {
            assert!(c.abs() < 1e-6, "black should map to black: {px:?}");
        }
    }

    fn assert_roundtrip(forward: fn(&mut [f32]), inverse: fn(&mut [f32]), tol: f32) {
        let original = [0.5_f32, 0.3, 0.7];
        let mut px = original;
        forward(&mut px);
        inverse(&mut px);
        for (i, (a, b)) in original.iter().zip(px.iter()).enumerate() {
            assert!((a - b).abs() < tol, "ch{i}: {a} → {b} (tol={tol})");
        }
    }

    // --- P3 ↔ sRGB ---
    #[test]
    fn p3_srgb_white() {
        assert_white_roundtrip(p3_to_srgb_f32);
    }
    #[test]
    fn srgb_p3_white() {
        assert_white_roundtrip(srgb_to_p3_f32);
    }
    #[test]
    fn p3_srgb_black() {
        assert_black_roundtrip(p3_to_srgb_f32);
    }
    #[test]
    fn p3_srgb_roundtrip() {
        assert_roundtrip(p3_to_srgb_f32, srgb_to_p3_f32, 1e-5);
    }

    // --- BT.2020 SDR ↔ sRGB ---
    #[test]
    fn bt2020_srgb_white() {
        assert_white_roundtrip(bt2020_sdr_to_srgb_f32);
    }
    #[test]
    fn srgb_bt2020_white() {
        assert_white_roundtrip(srgb_to_bt2020_sdr_f32);
    }
    #[test]
    fn bt2020_srgb_black() {
        assert_black_roundtrip(bt2020_sdr_to_srgb_f32);
    }
    #[test]
    fn bt2020_srgb_roundtrip() {
        assert_roundtrip(bt2020_sdr_to_srgb_f32, srgb_to_bt2020_sdr_f32, 1e-4);
    }

    // --- BT.2020 PQ ↔ sRGB ---
    #[test]
    fn bt2020pq_srgb_black() {
        assert_black_roundtrip(bt2020_pq_to_srgb_f32);
    }
    #[test]
    fn srgb_bt2020pq_black() {
        assert_black_roundtrip(srgb_to_bt2020_pq_f32);
    }

    // --- P3 ↔ BT.2020 ---
    #[test]
    fn p3_bt2020_white() {
        assert_white_roundtrip(p3_to_bt2020_f32);
    }
    #[test]
    fn bt2020_p3_white() {
        assert_white_roundtrip(bt2020_to_p3_f32);
    }
    #[test]
    fn p3_bt2020_roundtrip() {
        assert_roundtrip(p3_to_bt2020_f32, bt2020_to_p3_f32, 1e-5);
    }

    // --- Linear ---
    #[test]
    fn linear_white() {
        let mut px = [1.0_f32, 1.0, 1.0];
        convert_linear_rgb(&P3_TO_SRGB, &mut px);
        for c in &px {
            assert!((c - 1.0).abs() < 1e-6, "linear white: {px:?}");
        }
    }

    // --- RGBA alpha passthrough ---
    #[test]
    fn rgba_alpha_passthrough() {
        let mut px = [0.5_f32, 0.5, 0.5, 0.7];
        p3_to_srgb_f32_rgba(&mut px);
        assert!((px[3] - 0.7).abs() < f32::EPSILON, "alpha changed: {px:?}");
    }

    // --- Extended range ---
    #[test]
    fn extended_p3_green_negative_srgb_red() {
        let mut px = [0.0_f32, 1.0, 0.0];
        p3_to_srgb_f32_extended(&mut px);
        assert!(
            px[0] < 0.0,
            "P3 green should have negative sRGB red: {px:?}"
        );
        assert!(px[1] > 1.0, "P3 green should have >1.0 sRGB green: {px:?}");
    }

    #[test]
    fn clamped_p3_green_stays_in_range() {
        let mut px = [0.0_f32, 1.0, 0.0];
        p3_to_srgb_f32(&mut px);
        assert!(px[0] >= 0.0 && px[1] <= 1.0, "should be clamped: {px:?}");
    }

    // --- sRGB subset of wider gamuts ---
    #[test]
    fn srgb_subset_of_p3() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let mut px = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    srgb_to_p3_f32(&mut px);
                    for (i, c) in px.iter().enumerate() {
                        assert!(
                            *c >= -1e-5 && *c <= 1.0 + 1e-5,
                            "sRGB ({r},{g},{b}) ch{i} out of P3: {c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn srgb_subset_of_bt2020() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let mut px = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    srgb_to_bt2020_sdr_f32(&mut px);
                    for (i, c) in px.iter().enumerate() {
                        assert!(
                            *c >= -1e-4 && *c <= 1.0 + 1e-4,
                            "sRGB ({r},{g},{b}) ch{i} out of BT.2020: {c}"
                        );
                    }
                }
            }
        }
    }

    // --- u8 conversions ---

    #[test]
    fn u8_p3_srgb_white() {
        let src = [255u8, 255, 255];
        let mut dst = [0u8; 3];
        p3_to_srgb_u8_rgb(&src, &mut dst);
        assert_eq!(dst, [255, 255, 255]);
    }

    #[test]
    fn u8_p3_srgb_black() {
        let src = [0u8, 0, 0];
        let mut dst = [0u8; 3];
        p3_to_srgb_u8_rgb(&src, &mut dst);
        assert_eq!(dst, [0, 0, 0]);
    }

    #[test]
    fn u8_srgb_p3_roundtrip() {
        // Mid-gray should survive roundtrip within ±1 code value
        let original = [128u8, 128, 128];
        let mut mid = [0u8; 3];
        let mut result = [0u8; 3];
        srgb_to_p3_u8_rgb(&original, &mut mid);
        p3_to_srgb_u8_rgb(&mid, &mut result);
        for i in 0..3 {
            assert!(
                (original[i] as i16 - result[i] as i16).unsigned_abs() <= 1,
                "ch{i}: {original:?} → {mid:?} → {result:?}"
            );
        }
    }

    #[test]
    fn u8_rgba_alpha_passthrough() {
        let src = [128u8, 64, 32, 200];
        let mut dst = [0u8; 4];
        p3_to_srgb_u8_rgba(&src, &mut dst);
        assert_eq!(dst[3], 200, "alpha should be copied");
    }

    #[test]
    fn u8_bt2020_sdr_srgb_roundtrip() {
        let original = [100u8, 150, 200];
        let mut mid = [0u8; 3];
        let mut result = [0u8; 3];
        srgb_to_bt2020_sdr_u8_rgb(&original, &mut mid);
        bt2020_sdr_to_srgb_u8_rgb(&mid, &mut result);
        for i in 0..3 {
            assert!(
                (original[i] as i16 - result[i] as i16).unsigned_abs() <= 2,
                "ch{i}: {original:?} → {mid:?} → {result:?}"
            );
        }
    }

    // --- u16 conversions ---

    #[test]
    fn u16_p3_srgb_white() {
        let src = [65535u16, 65535, 65535];
        let mut dst = [0u16; 3];
        p3_to_srgb_u16_rgb(&src, &mut dst);
        assert_eq!(dst, [65535, 65535, 65535]);
    }

    #[test]
    fn u16_p3_srgb_black() {
        let src = [0u16, 0, 0];
        let mut dst = [0u16; 3];
        p3_to_srgb_u16_rgb(&src, &mut dst);
        assert_eq!(dst, [0, 0, 0]);
    }

    // --- Accuracy: f32 roundtrip max error ---

    #[test]
    fn f32_srgb_p3_srgb_roundtrip_accuracy() {
        // sRGB→P3→sRGB roundtrip: sRGB is a subset of P3, so no clamping.
        // This tests the pure polynomial + matrix precision.
        let mut max_err: f32 = 0.0;
        for r in (0..=255).step_by(4) {
            for g in (0..=255).step_by(4) {
                for b in (0..=255).step_by(16) {
                    let original = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    let mut px = original;
                    srgb_to_p3_f32(&mut px);
                    p3_to_srgb_f32(&mut px);
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

    // --- Matrix inverse verification ---

    #[test]
    fn matrix_inverse_pairs() {
        // P3_TO_SRGB × SRGB_TO_P3 ≈ identity
        let identity = |m1: &[[f32; 3]; 3], m2: &[[f32; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    let sum: f32 = (0..3).map(|k| m1[i][k] * m2[k][j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-5,
                        "M1×M2[{i}][{j}] = {sum}, expected {expected}"
                    );
                }
            }
        };
        identity(&P3_TO_SRGB, &SRGB_TO_P3);
        identity(&BT2020_TO_SRGB, &SRGB_TO_BT2020);
        identity(&BT2020_TO_P3, &P3_TO_BT2020);
    }

    // --- All matrices preserve white ---

    #[test]
    fn all_matrices_preserve_white() {
        let matrices: &[(&str, &[[f32; 3]; 3])] = &[
            ("P3_TO_SRGB", &P3_TO_SRGB),
            ("SRGB_TO_P3", &SRGB_TO_P3),
            ("BT2020_TO_SRGB", &BT2020_TO_SRGB),
            ("SRGB_TO_BT2020", &SRGB_TO_BT2020),
            ("P3_TO_BT2020", &P3_TO_BT2020),
            ("BT2020_TO_P3", &BT2020_TO_P3),
        ];
        for (name, m) in matrices {
            let (r, g, b) = mat3x3(m, 1.0, 1.0, 1.0);
            assert!(
                (r - 1.0).abs() < 1e-5 && (g - 1.0).abs() < 1e-5 && (b - 1.0).abs() < 1e-5,
                "{name}: white → ({r}, {g}, {b})"
            );
        }
    }

    // --- P3 subset of BT.2020 ---

    #[test]
    fn p3_subset_of_bt2020() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let mut px = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    p3_to_bt2020_f32(&mut px);
                    for (i, c) in px.iter().enumerate() {
                        assert!(
                            *c >= -1e-4 && *c <= 1.0 + 1e-4,
                            "P3 ({r},{g},{b}) ch{i} out of BT.2020: {c}"
                        );
                    }
                }
            }
        }
    }

    // --- Adobe RGB ---

    #[test]
    fn adobergb_srgb_white() {
        assert_white_roundtrip(adobergb_to_srgb_f32);
    }
    #[test]
    fn srgb_adobergb_white() {
        assert_white_roundtrip(srgb_to_adobergb_f32);
    }
    #[test]
    fn adobergb_srgb_black() {
        assert_black_roundtrip(adobergb_to_srgb_f32);
    }
    #[test]
    fn adobergb_srgb_roundtrip() {
        // sRGB is a subset of Adobe RGB, so sRGB→Adobe→sRGB roundtrips
        assert_roundtrip(srgb_to_adobergb_f32, adobergb_to_srgb_f32, 1e-4);
    }
    #[test]
    fn adobergb_p3_white() {
        assert_white_roundtrip(adobergb_to_p3_f32);
    }
    #[test]
    fn adobergb_rgba_alpha() {
        let mut px = [0.5_f32, 0.5, 0.5, 0.7];
        adobergb_to_srgb_f32_rgba(&mut px);
        assert!((px[3] - 0.7).abs() < f32::EPSILON, "alpha changed: {px:?}");
    }
    #[test]
    fn u8_adobergb_srgb_roundtrip() {
        let original = [128u8, 128, 128];
        let mut mid = [0u8; 3];
        let mut result = [0u8; 3];
        srgb_to_adobergb_u8_rgb(&original, &mut mid);
        adobergb_to_srgb_u8_rgb(&mid, &mut result);
        for i in 0..3 {
            assert!(
                (original[i] as i16 - result[i] as i16).unsigned_abs() <= 1,
                "ch{i}: {original:?} → {mid:?} → {result:?}"
            );
        }
    }
    #[test]
    fn all_matrices_preserve_white_with_adobe() {
        let matrices: &[(&str, &[[f32; 3]; 3])] = &[
            ("ADOBERGB_TO_SRGB", &ADOBERGB_TO_SRGB),
            ("SRGB_TO_ADOBERGB", &SRGB_TO_ADOBERGB),
            ("ADOBERGB_TO_P3", &ADOBERGB_TO_P3),
            ("P3_TO_ADOBERGB", &P3_TO_ADOBERGB),
            ("ADOBERGB_TO_BT2020", &ADOBERGB_TO_BT2020),
            ("BT2020_TO_ADOBERGB", &BT2020_TO_ADOBERGB),
        ];
        for (name, m) in matrices {
            let (r, g, b) = mat3x3(m, 1.0, 1.0, 1.0);
            assert!(
                (r - 1.0).abs() < 1e-5 && (g - 1.0).abs() < 1e-5 && (b - 1.0).abs() < 1e-5,
                "{name}: white → ({r}, {g}, {b})"
            );
        }
    }
    #[test]
    fn adobe_matrix_inverse_pairs() {
        let identity = |m1: &[[f32; 3]; 3], m2: &[[f32; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    let sum: f32 = (0..3).map(|k| m1[i][k] * m2[k][j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-4,
                        "M1×M2[{i}][{j}] = {sum}, expected {expected}"
                    );
                }
            }
        };
        identity(&ADOBERGB_TO_SRGB, &SRGB_TO_ADOBERGB);
        identity(&ADOBERGB_TO_P3, &P3_TO_ADOBERGB);
        identity(&ADOBERGB_TO_BT2020, &BT2020_TO_ADOBERGB);
    }

    // --- DCI-P3 ---

    #[test]
    fn dcip3_srgb_white() {
        assert_white_roundtrip(dcip3_to_srgb_f32);
    }
    #[test]
    fn srgb_dcip3_white() {
        assert_white_roundtrip(srgb_to_dcip3_f32);
    }
    #[test]
    fn dcip3_srgb_black() {
        assert_black_roundtrip(dcip3_to_srgb_f32);
    }
    #[test]
    fn dcip3_p3_white() {
        assert_white_roundtrip(dcip3_to_p3_f32);
    }
    #[test]
    fn p3_dcip3_white() {
        assert_white_roundtrip(p3_to_dcip3_f32);
    }
    #[test]
    fn dcip3_srgb_roundtrip() {
        assert_roundtrip(srgb_to_dcip3_f32, dcip3_to_srgb_f32, 1e-4);
    }
    #[test]
    fn dcip3_p3_roundtrip() {
        assert_roundtrip(p3_to_dcip3_f32, dcip3_to_p3_f32, 1e-4);
    }
    #[test]
    fn dci_matrix_inverse_pairs() {
        let identity = |m1: &[[f32; 3]; 3], m2: &[[f32; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    let sum: f32 = (0..3).map(|k| m1[i][k] * m2[k][j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-4,
                        "M1×M2[{i}][{j}] = {sum}, expected {expected}"
                    );
                }
            }
        };
        identity(&DCIP3_TO_SRGB, &SRGB_TO_DCIP3);
        identity(&DCIP3_TO_P3, &P3_TO_DCIP3);
        identity(&DCIP3_TO_BT2020, &BT2020_TO_DCIP3);
    }
    #[test]
    fn all_dci_matrices_preserve_white() {
        let matrices: &[(&str, &[[f32; 3]; 3])] = &[
            ("DCIP3_TO_SRGB", &DCIP3_TO_SRGB),
            ("SRGB_TO_DCIP3", &SRGB_TO_DCIP3),
            ("DCIP3_TO_P3", &DCIP3_TO_P3),
            ("P3_TO_DCIP3", &P3_TO_DCIP3),
            ("DCIP3_TO_BT2020", &DCIP3_TO_BT2020),
            ("BT2020_TO_DCIP3", &BT2020_TO_DCIP3),
        ];
        for (name, m) in matrices {
            let (r, g, b) = mat3x3(m, 1.0, 1.0, 1.0);
            assert!(
                (r - 1.0).abs() < 1e-4 && (g - 1.0).abs() < 1e-4 && (b - 1.0).abs() < 1e-4,
                "{name}: white → ({r}, {g}, {b})"
            );
        }
    }
}
