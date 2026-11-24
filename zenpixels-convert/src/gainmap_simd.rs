//! Bit-exact scalar + padded SIMD-only kernels for ISO 21496-1 gain-map resampling and application.
//!
//! All scratch row slices owned by `GainMapApplier` are padded to a multiple of `8` elements
//! (`w_pad = width.next_multiple_of(8)`), which is a multiple of both `f32x8` (x86-64 V3) and
//! `f32x4` (AArch64 NEON / WASM128). Every SIMD kernel therefore executes a **pure SIMD-only
//! loop with zero scalar tail** inside `#[archmage::arcane]`, minimizing codegen size while
//! keeping **0-bit parity** against its scalar reference.

use archmage::ScalarToken;
use magemath::SimdMath;
use magemath::expf::{Expf, ExpfDomain, ExpfOptions};

const MATH_PORTABILITY: magemath::Portability = if cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
)) {
    magemath::Portability::BackendDependent
} else {
    magemath::Portability::BitExact
};

/// Defense-in-depth log₂ gain bound matching `ultrahdr-core`: `2^±30 ≈ [9.3e-10, 1.07e9]`.
pub(crate) const MAX_LOG2_GAIN: f32 = 30.0;

/// Portable bit-exact `exp2f` profile for log₂ gains in `[-30, +30]` (strictly inside `NormalOutput`).
const GAINMAP_EXP_OPT: u128 = ExpfOptions::new()
    .domain(ExpfDomain::NormalOutput)
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

#[inline(always)]
fn clamp_unit_scalar(v: f32) -> f32 {
    if v > 0.0 {
        if v < 1.0 { v } else { 1.0 }
    } else {
        0.0
    }
}

/// Ordered clamp to `[-MAX_LOG2_GAIN, +MAX_LOG2_GAIN]` matching SIMD `blend(lg.simd_gt(lo), lg.min_fast(hi), lo)`
/// identically for ALL floats (including `-0.0`, `±Inf`, and `NaN` → `-MAX_LOG2_GAIN`).
#[inline(always)]
fn clamp_log2_scalar(v: f32) -> f32 {
    if v > -MAX_LOG2_GAIN {
        if v < MAX_LOG2_GAIN { v } else { MAX_LOG2_GAIN }
    } else {
        -MAX_LOG2_GAIN
    }
}

/// Scalar `exp2(clamp_ordered(log2_gain, -30, +30))`, 0-bit identical to the SIMD lane calculation.
#[inline(never)]
pub(crate) fn exp2_gain_scalar(log2_gain: f32) -> f32 {
    let clamped = clamp_log2_scalar(log2_gain);
    Expf::<GAINMAP_EXP_OPT>::exp2_with_token(ScalarToken, clamped).unwrap_or(1.0)
}

// ---------------------------------------------------------------------------
// 1. Vertical separable resampling blends (padded to multiple of 8; SIMD-only!)
// ---------------------------------------------------------------------------

/// Scalar 2-row vertical blend with `[0.0, 1.0]` clamp (0-bit reference).
#[inline(never)]
pub(crate) fn blend_rows_2_scalar(r0: &[f32], r1: &[f32], w0: f32, w1: f32, out: &mut [f32]) {
    let n = out.len().min(r0.len()).min(r1.len());
    for i in 0..n {
        out[i] = clamp_unit_scalar(r0[i] * w0 + r1[i] * w1);
    }
}

macro_rules! blend_rows_2_body {
    ($V:ty, $LANES:literal, $token:ident, $r0:ident, $r1:ident, $w0:ident, $w1:ident, $out:ident) => {{
        let n = $out.len().min($r0.len()).min($r1.len());
        let vw0 = <$V>::splat_t($token, $w0);
        let vw1 = <$V>::splat_t($token, $w1);
        let zero = <$V>::zero_t($token);
        let one = <$V>::splat_t($token, 1.0);
        let (c0, _) = $r0[..n].as_chunks::<$LANES>();
        let (c1, _) = $r1[..n].as_chunks::<$LANES>();
        let (co, _) = $out[..n].as_chunks_mut::<$LANES>();
        for ((a, b), o) in c0.iter().zip(c1.iter()).zip(co.iter_mut()) {
            let sum = <$V>::load_t($token, a) * vw0 + <$V>::load_t($token, b) * vw1;
            let clamped = <$V>::blend(sum.simd_gt(zero), sum.min_fast(one), zero);
            clamped.store(o);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blend_rows_2_v3_t(
    token: archmage::X64V3Token,
    r0: &[f32],
    r1: &[f32],
    w0: f32,
    w1: f32,
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    blend_rows_2_body!(V, 8, token, r0, r1, w0, w1, out);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn blend_rows_2_neon_t(
    token: archmage::NeonToken,
    r0: &[f32],
    r1: &[f32],
    w0: f32,
    w1: f32,
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    blend_rows_2_body!(V, 4, token, r0, r1, w0, w1, out);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn blend_rows_2_wasm128_t(
    token: archmage::Wasm128Token,
    r0: &[f32],
    r1: &[f32],
    w0: f32,
    w1: f32,
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    blend_rows_2_body!(V, 4, token, r0, r1, w0, w1, out);
}

fn blend_rows_2_scalar_t(
    _token: ScalarToken,
    r0: &[f32],
    r1: &[f32],
    w0: f32,
    w1: f32,
    out: &mut [f32],
) {
    blend_rows_2_scalar(r0, r1, w0, w1, out);
}

pub(crate) fn simd_blend_rows_2(r0: &[f32], r1: &[f32], w0: f32, w1: f32, out: &mut [f32]) {
    archmage::attuned!(
        blend_rows_2(r0, r1, w0, w1, out),
        [_v3, _neon, _wasm128, _scalar]
    );
}

/// Scalar 4-row vertical blend (Catmull-Rom / Mitchell) with `[0.0, 1.0]` clamp (0-bit reference).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn blend_rows_4_scalar(
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    let n = out
        .len()
        .min(r0.len())
        .min(r1.len())
        .min(r2.len())
        .min(r3.len());
    for i in 0..n {
        let s = ((r0[i] * w[0] + r1[i] * w[1]) + r2[i] * w[2]) + r3[i] * w[3];
        out[i] = clamp_unit_scalar(s);
    }
}

macro_rules! blend_rows_4_body {
    ($V:ty, $LANES:literal, $token:ident, $r0:ident, $r1:ident, $r2:ident, $r3:ident, $w:ident, $out:ident) => {{
        let n = $out
            .len()
            .min($r0.len())
            .min($r1.len())
            .min($r2.len())
            .min($r3.len());
        let vw0 = <$V>::splat_t($token, $w[0]);
        let vw1 = <$V>::splat_t($token, $w[1]);
        let vw2 = <$V>::splat_t($token, $w[2]);
        let vw3 = <$V>::splat_t($token, $w[3]);
        let zero = <$V>::zero_t($token);
        let one = <$V>::splat_t($token, 1.0);
        let (c0, _) = $r0[..n].as_chunks::<$LANES>();
        let (c1, _) = $r1[..n].as_chunks::<$LANES>();
        let (c2, _) = $r2[..n].as_chunks::<$LANES>();
        let (c3, _) = $r3[..n].as_chunks::<$LANES>();
        let (co, _) = $out[..n].as_chunks_mut::<$LANES>();
        for i in 0..co.len() {
            let s = ((<$V>::load_t($token, &c0[i]) * vw0 + <$V>::load_t($token, &c1[i]) * vw1)
                + <$V>::load_t($token, &c2[i]) * vw2)
                + <$V>::load_t($token, &c3[i]) * vw3;
            let clamped = <$V>::blend(s.simd_gt(zero), s.min_fast(one), zero);
            clamped.store(&mut co[i]);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn blend_rows_4_v3_t(
    token: archmage::X64V3Token,
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    blend_rows_4_body!(V, 8, token, r0, r1, r2, r3, w, out);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn blend_rows_4_neon_t(
    token: archmage::NeonToken,
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    blend_rows_4_body!(V, 4, token, r0, r1, r2, r3, w, out);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn blend_rows_4_wasm128_t(
    token: archmage::Wasm128Token,
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    blend_rows_4_body!(V, 4, token, r0, r1, r2, r3, w, out);
}

#[allow(clippy::too_many_arguments)]
fn blend_rows_4_scalar_t(
    _token: ScalarToken,
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    blend_rows_4_scalar(r0, r1, r2, r3, w, out);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_blend_rows_4(
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    w: [f32; 4],
    out: &mut [f32],
) {
    archmage::attuned!(
        blend_rows_4(r0, r1, r2, r3, w, out),
        [_v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// 2. Decode normalized [0, 1] (gamma-applied) gain samples -> exp2 multipliers
//    Single Expf loop + padded SIMD-only body (no duplicate gamma branch / tail)!
// ---------------------------------------------------------------------------

/// Scalar gain decode: `exp2(clamp_ordered(base_w + slope_w * clamp_unit(s), -30, 30))`.
///
/// Any non-unit `1/gamma` power is applied to `samples` beforehand so `Expf::exp2` is
/// instantiated strictly once per ISA tier.
#[inline(never)]
pub(crate) fn decode_gain_1ch_scalar(
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    let n = samples.len().min(out_gains.len());
    for i in 0..n {
        let s = clamp_unit_scalar(samples[i]);
        let lg = base_w + slope_w * s;
        out_gains[i] = exp2_gain_scalar(lg);
    }
}

macro_rules! decode_gain_1ch_body {
    ($V:ty, $LANES:literal, $token:ident, $samples:ident, $base_w:ident, $slope_w:ident, $out_gains:ident) => {{
        let n = $samples.len().min($out_gains.len());
        let zero = <$V>::zero_t($token);
        let one = <$V>::splat_t($token, 1.0);
        let lo = <$V>::splat_t($token, -MAX_LOG2_GAIN);
        let hi = <$V>::splat_t($token, MAX_LOG2_GAIN);
        let v_base = <$V>::splat_t($token, $base_w);
        let v_slope = <$V>::splat_t($token, $slope_w);
        let (in_chunks, _) = $samples[..n].as_chunks::<$LANES>();
        let (out_chunks, _) = $out_gains[..n].as_chunks_mut::<$LANES>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let raw = <$V>::load_t($token, inp);
            let s = <$V>::blend(raw.simd_gt(zero), raw.min_fast(one), zero);
            let lg = v_base + v_slope * s;
            let clamped = <$V>::blend(lg.simd_gt(lo), lg.min_fast(hi), lo);
            let gain = clamped.exp2::<GAINMAP_EXP_OPT>();
            gain.store(out);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn decode_gain_1ch_v3_t(
    token: archmage::X64V3Token,
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    decode_gain_1ch_body!(V, 8, token, samples, base_w, slope_w, out_gains);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn decode_gain_1ch_neon_t(
    token: archmage::NeonToken,
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    decode_gain_1ch_body!(V, 4, token, samples, base_w, slope_w, out_gains);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn decode_gain_1ch_wasm128_t(
    token: archmage::Wasm128Token,
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    decode_gain_1ch_body!(V, 4, token, samples, base_w, slope_w, out_gains);
}

fn decode_gain_1ch_scalar_t(
    _token: ScalarToken,
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    decode_gain_1ch_scalar(samples, base_w, slope_w, out_gains);
}

pub(crate) fn simd_decode_gain_1ch(
    samples: &[f32],
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    archmage::attuned!(
        decode_gain_1ch(samples, base_w, slope_w, out_gains),
        [_v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// 3. Linear RGB(A) gain application (padded to multiple of 8; SIMD-only!)
// ---------------------------------------------------------------------------

/// Scalar linear RGB(A) gain combine (0-bit reference, preserves alpha untouched when `has_alpha`).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn combine_linear_f32_scalar(
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    let ch = if has_alpha { 4usize } else { 3usize };
    let pixels = (base_rgb.len() / ch)
        .min(out_rgb.len() / ch)
        .min(gr.len())
        .min(gg.len())
        .min(gb.len());
    for i in 0..pixels {
        let o = i * ch;
        out_rgb[o] = (base_rgb[o] + k_base[0]) * gr[i] - k_alt[0];
        out_rgb[o + 1] = (base_rgb[o + 1] + k_base[1]) * gg[i] - k_alt[1];
        out_rgb[o + 2] = (base_rgb[o + 2] + k_base[2]) * gb[i] - k_alt[2];
        if has_alpha {
            out_rgb[o + 3] = base_rgb[o + 3];
        }
    }
}

macro_rules! combine_linear_f32_body {
    ($V:ty, $LANES:literal, $token:ident, $base:ident, $gr:ident, $gg:ident, $gb:ident, $kb:ident, $ka:ident, $has_alpha:ident, $out:ident) => {{
        let ch = if $has_alpha { 4usize } else { 3usize };
        let pixels = ($base.len() / ch)
            .min($out.len() / ch)
            .min($gr.len())
            .min($gg.len())
            .min($gb.len());
        let kb0 = <$V>::splat_t($token, $kb[0]);
        let kb1 = <$V>::splat_t($token, $kb[1]);
        let kb2 = <$V>::splat_t($token, $kb[2]);
        let ka0 = <$V>::splat_t($token, $ka[0]);
        let ka1 = <$V>::splat_t($token, $ka[1]);
        let ka2 = <$V>::splat_t($token, $ka[2]);

        let mut i = 0usize;
        if $has_alpha {
            while i + $LANES <= pixels {
                let o = i * 4;
                let v0 = <$V>::load_t($token, (&$base[o..o + $LANES]).try_into().unwrap());
                let v1 = <$V>::load_t(
                    $token,
                    (&$base[o + $LANES..o + 2 * $LANES]).try_into().unwrap(),
                );
                let v2 = <$V>::load_t(
                    $token,
                    (&$base[o + 2 * $LANES..o + 3 * $LANES]).try_into().unwrap(),
                );
                let v3 = <$V>::load_t(
                    $token,
                    (&$base[o + 3 * $LANES..o + 4 * $LANES]).try_into().unwrap(),
                );
                let [r, g, b, a] = <$V>::deinterleave_4ch([v0, v1, v2, v3]);
                let vr = <$V>::load_t($token, (&$gr[i..i + $LANES]).try_into().unwrap());
                let vg = <$V>::load_t($token, (&$gg[i..i + $LANES]).try_into().unwrap());
                let vb = <$V>::load_t($token, (&$gb[i..i + $LANES]).try_into().unwrap());
                let [o0, o1, o2, o3] = <$V>::interleave_4ch([
                    (r + kb0) * vr - ka0,
                    (g + kb1) * vg - ka1,
                    (b + kb2) * vb - ka2,
                    a,
                ]);
                o0.store((&mut $out[o..o + $LANES]).try_into().unwrap());
                o1.store((&mut $out[o + $LANES..o + 2 * $LANES]).try_into().unwrap());
                o2.store(
                    (&mut $out[o + 2 * $LANES..o + 3 * $LANES])
                        .try_into()
                        .unwrap(),
                );
                o3.store(
                    (&mut $out[o + 3 * $LANES..o + 4 * $LANES])
                        .try_into()
                        .unwrap(),
                );
                i += $LANES;
            }
        } else {
            while i + $LANES <= pixels {
                let o = i * 3;
                let v0 = <$V>::load_t($token, (&$base[o..o + $LANES]).try_into().unwrap());
                let v1 = <$V>::load_t(
                    $token,
                    (&$base[o + $LANES..o + 2 * $LANES]).try_into().unwrap(),
                );
                let v2 = <$V>::load_t(
                    $token,
                    (&$base[o + 2 * $LANES..o + 3 * $LANES]).try_into().unwrap(),
                );
                let [r, g, b] = <$V>::deinterleave_3ch([v0, v1, v2]);
                let vr = <$V>::load_t($token, (&$gr[i..i + $LANES]).try_into().unwrap());
                let vg = <$V>::load_t($token, (&$gg[i..i + $LANES]).try_into().unwrap());
                let vb = <$V>::load_t($token, (&$gb[i..i + $LANES]).try_into().unwrap());
                let [o0, o1, o2] = <$V>::interleave_3ch([
                    (r + kb0) * vr - ka0,
                    (g + kb1) * vg - ka1,
                    (b + kb2) * vb - ka2,
                ]);
                o0.store((&mut $out[o..o + $LANES]).try_into().unwrap());
                o1.store((&mut $out[o + $LANES..o + 2 * $LANES]).try_into().unwrap());
                o2.store(
                    (&mut $out[o + 2 * $LANES..o + 3 * $LANES])
                        .try_into()
                        .unwrap(),
                );
                i += $LANES;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn combine_linear_f32_v3_t(
    token: archmage::X64V3Token,
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    combine_linear_f32_body!(
        V, 8, token, base_rgb, gr, gg, gb, k_base, k_alt, has_alpha, out_rgb
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn combine_linear_f32_neon_t(
    token: archmage::NeonToken,
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    combine_linear_f32_body!(
        V, 4, token, base_rgb, gr, gg, gb, k_base, k_alt, has_alpha, out_rgb
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn combine_linear_f32_wasm128_t(
    token: archmage::Wasm128Token,
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    combine_linear_f32_body!(
        V, 4, token, base_rgb, gr, gg, gb, k_base, k_alt, has_alpha, out_rgb
    );
}

#[allow(clippy::too_many_arguments)]
fn combine_linear_f32_scalar_t(
    _token: ScalarToken,
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    combine_linear_f32_scalar(base_rgb, gr, gg, gb, k_base, k_alt, has_alpha, out_rgb);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_combine_linear_f32(
    base_rgb: &[f32],
    gr: &[f32],
    gg: &[f32],
    gb: &[f32],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    has_alpha: bool,
    out_rgb: &mut [f32],
) {
    archmage::attuned!(
        combine_linear_f32(base_rgb, gr, gg, gb, k_base, k_alt, has_alpha, out_rgb),
        [_v3, _neon, _wasm128, _scalar]
    );
}
