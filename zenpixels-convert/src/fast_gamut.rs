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

// =========================================================================
// Gamut matrices (all D65, no chromatic adaptation needed)
// =========================================================================

/// sRGB linear → Display P3 linear.
pub const SRGB_TO_P3: [[f32; 3]; 3] = [
    [0.8224619687, 0.1775380313, 0.0],
    [0.0331941989, 0.9668058011, 0.0],
    [0.0170826307, 0.0723974407, 0.9105199286],
];

/// sRGB linear → BT.2020 linear.
pub const SRGB_TO_BT2020: [[f32; 3]; 3] = [
    [0.6274038959, 0.3292830384, 0.0433130657],
    [0.0690972894, 0.9195403951, 0.0113623156],
    [0.0163914389, 0.0880133079, 0.8955952532],
];

/// Display P3 linear → sRGB linear.
pub const P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.2249401763, -0.2249401763, 0.0],
    [-0.0420569547, 1.0420569547, 0.0],
    [-0.0196375546, -0.0786360456, 1.0982736001],
];

/// Display P3 linear → BT.2020 linear.
pub const P3_TO_BT2020: [[f32; 3]; 3] = [
    [0.7538330344, 0.1985973691, 0.0475695966],
    [0.0457438490, 0.9417772198, 0.0124789312],
    [-0.0012103404, 0.0176017173, 0.9836086231],
];

/// BT.2020 linear → sRGB linear.
pub const BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.6604910021, -0.5876411388, -0.0728498633],
    [-0.1245504745, 1.1328998971, -0.0083494226],
    [-0.0181507634, -0.1005788980, 1.1187296614],
];

/// BT.2020 linear → Display P3 linear.
pub const BT2020_TO_P3: [[f32; 3]; 3] = [
    [1.3435782526, -0.2821796705, -0.0613985821],
    [-0.0652974528, 1.0757879158, -0.0104904631],
    [0.0028217873, -0.0195984945, 1.0167767073],
];

// =========================================================================
// Shared helpers
// =========================================================================

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
}
