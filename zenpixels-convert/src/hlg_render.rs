//! BT.2100-3 Table 5 display rendering, resolved by the plan or CMS adapter.
//! OOTF structure follows zentone f32e091c's hlg module (same ownership).
//! SIMD fast path uses const-configured `magemath` (`Expf`/`Logf` `Midp` +
//! `Powf` `Highp` packed binary64 for the luminance OOTF), falling back
//! per-lane to exact f64 scalar on non-finite, extreme-range, or exotic-gamma inputs.
use magemath::SimdMath;
use magemath::expf::{ExpfDomain, ExpfOptions};
use magemath::logf::{LogfDomain, LogfOptions};
use magemath::powf::{PowfDomain, PowfOptions};

use zenpixels::hdr::HlgRendering;

pub(crate) const BT2020_LUMA: [f64; 3] = [0.2627, 0.6780, 0.0593];

const HLG_A_F32: f32 = 0.17883277;
const HLG_B_F32: f32 = 0.28466892; // 1 - 4 * A
#[allow(clippy::excessive_precision)]
const HLG_C_F32: f32 = 0.55991073; // 0.5 - A * ln(4 * A)
const HLG_A_LN2_F32: f32 = HLG_A_F32 * core::f32::consts::LN_2;
const HLG_INV_A_LOG2E_F32: f32 = core::f32::consts::LOG2_E / HLG_A_F32;

const MATH_PORTABILITY: magemath::Portability = if cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
)) {
    magemath::Portability::BackendDependent
} else {
    magemath::Portability::BitExact
};

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

/// `Highp + BackendDependent` in `magemath` delegates to `std` `f32::powf`, so
/// `no_std` `wasm32` builds fall back to `BitExact` packed binary64.
const HLG_OOTF_PORTABILITY: magemath::Portability = if cfg!(all(
    target_arch = "wasm32",
    feature = "std",
    not(feature = "precise-portable")
)) {
    magemath::Portability::BackendDependent
} else {
    magemath::Portability::BitExact
};

/// Packed binary64 (`Highp`, <= 0.5 ULP) luminance OOTF power for `|Y| in [2^-24, 2^16]`
/// and system-gamma exponent in `[-1.0, 2.0]` (`|-24 * 2.0| = 48 <= 125`, eliding all
/// per-lane range checks).
const HLG_OOTF_LO: f32 = 1.0 / 16_777_216.0;
const HLG_OOTF_HI: f32 = 65_536.0;
const HLG_OOTF_HIGHP: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveGamma)
    .base_range(HLG_OOTF_LO, HLG_OOTF_HI)
    .exponent_range(-1.0, 2.0)
    .reject_subnormals()
    .highp()
    .portability(HLG_OOTF_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

fn decode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 0.5 {
        v * v.abs() / 3.0
    } else {
        v.signum() * (libm::exp((v.abs() - c) / a) + b) / 12.0
    }
}
fn encode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 1.0 / 12.0 {
        v.signum() * libm::sqrt(3.0 * v.abs())
    } else {
        v.signum() * (a * libm::log(12.0 * v.abs() - b) + c)
    }
}

// Extended RGB uses signed transfer curves and |Y| in the common luminance
// multiplier. Zero luminance maps to zero; nonfinite RGB maps to RGB NaN.
// Alpha is copied bit-for-bit. No per-row reduction or allocation occurs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedHlg {
    gamma: f64,
    beta: f64,
    scale: f64,
    luma: [f64; 3],
    gamma_m1_f32: f32,
    inv_ootf_exp_f32: f32,
    beta_f32: f32,
    one_minus_beta_f32: f32,
    inv_one_minus_beta_f32: f32,
    scale_f32: f32,
    inv_scale_f32: f32,
    luma_f32: [f32; 3],
    simd_fast_eligible: bool,
    pub(crate) inverse: bool,
    clip: bool,
}
impl PreparedHlg {
    pub(crate) fn new(rendering: HlgRendering, inverse: bool, white: f32, clip: bool) -> Self {
        Self::new_with_luma(rendering, inverse, white, clip, BT2020_LUMA)
    }

    pub(crate) fn new_with_luma(
        rendering: HlgRendering,
        inverse: bool,
        white: f32,
        clip: bool,
        luma: [f64; 3],
    ) -> Self {
        let gamma = f64::from(rendering.system_gamma());
        let peak = f64::from(rendering.peak().nits());
        let beta =
            libm::sqrt(3.0 * libm::pow(f64::from(rendering.black_nits()) / peak, 1.0 / gamma));
        let scale = peak / f64::from(white);
        let gamma_m1_f32 = (gamma - 1.0) as f32;
        let inv_ootf_exp_f32 = ((1.0 - gamma) / gamma) as f32;
        let beta_f32 = beta as f32;
        let one_minus_beta_f32 = (1.0 - beta) as f32;
        let inv_one_minus_beta_f32 = (1.0 / (1.0 - beta)) as f32;
        let scale_f32 = scale as f32;
        let inv_scale_f32 = (1.0 / scale) as f32;
        let luma_f32 = [luma[0] as f32, luma[1] as f32, luma[2] as f32];
        let active_exp = if inverse {
            inv_ootf_exp_f32
        } else {
            gamma_m1_f32
        };
        let simd_fast_eligible = (-1.0..=2.0).contains(&active_exp)
            && beta_f32.is_finite()
            && one_minus_beta_f32.is_finite()
            && inv_one_minus_beta_f32.is_finite()
            && scale_f32.is_finite()
            && scale_f32 > 0.0
            && inv_scale_f32.is_finite()
            && inv_scale_f32 > 0.0
            && luma_f32.iter().all(|v| v.is_finite());
        Self {
            gamma,
            beta,
            scale,
            luma,
            gamma_m1_f32,
            inv_ootf_exp_f32,
            beta_f32,
            one_minus_beta_f32,
            inv_one_minus_beta_f32,
            scale_f32,
            inv_scale_f32,
            luma_f32,
            simd_fast_eligible,
            inverse,
            clip,
        }
    }
}

#[cold]
#[inline(never)]
fn hlg_pixel_f64(rgb_in: [f32; 3], prepared: &PreparedHlg) -> [f32; 3] {
    let PreparedHlg {
        gamma,
        beta,
        scale,
        luma,
        inverse,
        clip,
        ..
    } = *prepared;
    let mut rgb = [
        f64::from(rgb_in[0]),
        f64::from(rgb_in[1]),
        f64::from(rgb_in[2]),
    ];
    if rgb.iter().any(|v| !v.is_finite()) {
        rgb = [f64::NAN; 3];
    } else if inverse {
        for v in &mut rgb {
            *v /= scale;
        }
        let y = luma[0] * rgb[0] + luma[1] * rgb[1] + luma[2] * rgb[2];
        let gain = if y == 0.0 {
            0.0
        } else {
            libm::pow(y.abs(), (1.0 - gamma) / gamma)
        };
        for v in &mut rgb {
            *v = (encode(*v * gain) - beta) / (1.0 - beta);
            if clip {
                *v = v.clamp(0.0, 1.0);
            }
        }
    } else {
        for v in &mut rgb {
            if clip {
                // ITU-R BT.2100-3 Table 5 Note 5h: clamp super-white at 1.0,
                // apply black-level lift first so PLUGE sub-blacks in
                // [-beta / (1 - beta), 0) decode smoothly to [0, L_B], and
                // clamp below display black at 0.0.
                let v_clamped = v.min(1.0);
                *v = decode(((1.0 - beta) * v_clamped + beta).max(0.0));
            } else {
                *v = decode((1.0 - beta) * *v + beta);
            }
        }
        let y = luma[0] * rgb[0] + luma[1] * rgb[1] + luma[2] * rgb[2];
        let gain = if y == 0.0 {
            0.0
        } else {
            scale * libm::pow(y.abs(), gamma - 1.0)
        };
        for v in &mut rgb {
            *v *= gain;
        }
    }
    [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
}

macro_rules! hlg_eval_chunk {
    ($vec:ident, $lanes:literal, $token:ident, $prepared:ident, $inverse:expr, $clip:expr, $ra:ident, $ga:ident, $ba:ident,
     $zero:ident, $one:ident, $half:ident, $neg_zero:ident, $inv_3:ident, $three:ident, $twelve:ident, $inv_12:ident,
     $b_const:ident, $c_const:ident, $a_ln2:ident, $inv_a_log2e:ident, $exp_max:ident, $ootf_lo:ident, $ootf_hi:ident,
     $lr:ident, $lg:ident, $lb:ident, $beta_v:ident, $one_minus_beta_v:ident, $inv_one_minus_beta_v:ident,
     $scale_v:ident, $inv_scale_v:ident, $gamma_m1:ident, $inv_ootf_exp:ident) => {{
        let r_in = $vec::from_array_t($token, $ra);
        let g_in = $vec::from_array_t($token, $ga);
        let b_in = $vec::from_array_t($token, $ba);
        let finite_in = (r_in * $zero).simd_eq($zero)
            & (g_in * $zero).simd_eq($zero)
            & (b_in * $zero).simd_eq($zero);

        let (mut ro, mut go, mut bo, ok) = if !$inverse {
            let decode_ch = |c: $vec| -> ($vec, $vec) {
                let lifted = if $clip {
                    ($one_minus_beta_v * c.min_fast($one) + $beta_v).max_fast($zero)
                } else {
                    $one_minus_beta_v * c + $beta_v
                };
                let abs_c = lifted.abs();
                let sign_c = lifted & $neg_zero;
                let lo = (abs_c * abs_c) * $inv_3;
                let exp_arg = (abs_c - $c_const) * $inv_a_log2e;
                let le_half = abs_c.simd_le($half);
                let hi_branch = abs_c.simd_gt($half);
                let exp_ok = exp_arg.simd_le($exp_max);
                let ch_ok = le_half | exp_ok;
                let safe_arg = $vec::blend(hi_branch & exp_ok, exp_arg, $zero);
                let hi = (safe_arg.exp2::<HLG_EXP_FAST>() + $b_const) * $inv_12;
                let mag = $vec::blend(hi_branch, hi, lo);
                (mag | sign_c, ch_ok)
            };
            let (dr, ok_r) = decode_ch(r_in);
            let (dg, ok_g) = decode_ch(g_in);
            let (db, ok_b) = decode_ch(b_in);
            let y = $lr * dr + $lg * dg + $lb * db;
            let y_abs = y.abs();
            let y_zero = y.simd_eq($zero);
            let y_in_range = y_abs.simd_ge($ootf_lo) & y_abs.simd_le($ootf_hi);
            let y_ok = y_zero | y_in_range;
            let y_safe = $vec::blend(y_in_range, y_abs, $one);
            let gain = $vec::blend(
                y_zero,
                $zero,
                $scale_v * y_safe.powf::<HLG_OOTF_HIGHP>($gamma_m1),
            );
            let ok = finite_in & y_ok & ok_r & ok_g & ok_b;
            (
                (dr * gain).to_array(),
                (dg * gain).to_array(),
                (db * gain).to_array(),
                ok,
            )
        } else {
            let rs = r_in * $inv_scale_v;
            let gs = g_in * $inv_scale_v;
            let bs = b_in * $inv_scale_v;
            let y = $lr * rs + $lg * gs + $lb * bs;
            let y_abs = y.abs();
            let y_zero = y.simd_eq($zero);
            let y_in_range = y_abs.simd_ge($ootf_lo) & y_abs.simd_le($ootf_hi);
            let y_ok = y_zero | y_in_range;
            let y_safe = $vec::blend(y_in_range, y_abs, $one);
            let gain = $vec::blend(y_zero, $zero, y_safe.powf::<HLG_OOTF_HIGHP>($inv_ootf_exp));

            let encode_ch = |c: $vec| -> ($vec, $vec) {
                let abs_c = c.abs();
                let sign_c = c & $neg_zero;
                let lo = ($three * abs_c).sqrt();
                let log_arg = $twelve * abs_c - $b_const;
                let le_inv12 = abs_c.simd_le($inv_12);
                let hi_branch = abs_c.simd_gt($inv_12);
                let log_ok = (log_arg * $zero).simd_eq($zero);
                let ch_ok = le_inv12 | log_ok;
                let safe_arg = $vec::blend(hi_branch & log_ok, log_arg, $one);
                let hi = $a_ln2 * safe_arg.log2::<HLG_LOG_FAST>() + $c_const;
                let mag = $vec::blend(hi_branch, hi, lo);
                let unlifted = ((mag | sign_c) - $beta_v) * $inv_one_minus_beta_v;
                let out = if $clip {
                    unlifted.max_fast($zero).min_fast($one)
                } else {
                    unlifted
                };
                (out, ch_ok)
            };
            let (er, ok_r) = encode_ch(rs * gain);
            let (eg, ok_g) = encode_ch(gs * gain);
            let (eb, ok_b) = encode_ch(bs * gain);
            let ok = finite_in & y_ok & ok_r & ok_g & ok_b;
            (er.to_array(), eg.to_array(), eb.to_array(), ok)
        };

        // Per-lane fallback preserves exact single-pixel vs whole-row partition parity:
        // valid neighbor lanes always keep their SIMD outputs unchanged.
        if !ok.bitcast_to_i32().all_true() {
            let ok_arr = ok.to_array();
            for i in 0..$lanes {
                if ok_arr[i].to_bits() == 0 {
                    let exact = hlg_pixel_f64([$ra[i], $ga[i], $ba[i]], $prepared);
                    ro[i] = exact[0];
                    go[i] = exact[1];
                    bo[i] = exact[2];
                }
            }
        }
        (ro, go, bo)
    }};
}

macro_rules! hlg_soa_loop {
    ($vec:ident, $lanes:literal, $token:ident, $r:ident, $g:ident, $b:ident, $len:ident,
     $prepared:ident, $inverse:expr, $clip:expr) => {{
        let zero = $vec::zero_t($token);
        let one = $vec::splat_t($token, 1.0);
        let half = $vec::splat_t($token, 0.5);
        let neg_zero = $vec::splat_t($token, -0.0);
        let inv_3 = $vec::splat_t($token, 1.0 / 3.0);
        let three = $vec::splat_t($token, 3.0);
        let twelve = $vec::splat_t($token, 12.0);
        let inv_12 = $vec::splat_t($token, 1.0 / 12.0);
        let b_const = $vec::splat_t($token, HLG_B_F32);
        let c_const = $vec::splat_t($token, HLG_C_F32);
        let a_ln2 = $vec::splat_t($token, HLG_A_LN2_F32);
        let inv_a_log2e = $vec::splat_t($token, HLG_INV_A_LOG2E_F32);
        let exp_max = $vec::splat_t($token, 127.0);
        let ootf_lo = $vec::splat_t($token, HLG_OOTF_LO);
        let ootf_hi = $vec::splat_t($token, HLG_OOTF_HI);

        let lr = $vec::splat_t($token, $prepared.luma_f32[0]);
        let lg = $vec::splat_t($token, $prepared.luma_f32[1]);
        let lb = $vec::splat_t($token, $prepared.luma_f32[2]);
        let beta_v = $vec::splat_t($token, $prepared.beta_f32);
        let one_minus_beta_v = $vec::splat_t($token, $prepared.one_minus_beta_f32);
        let inv_one_minus_beta_v = $vec::splat_t($token, $prepared.inv_one_minus_beta_f32);
        let scale_v = $vec::splat_t($token, $prepared.scale_f32);
        let inv_scale_v = $vec::splat_t($token, $prepared.inv_scale_f32);
        let gamma_m1 = $prepared.gamma_m1_f32;
        let inv_ootf_exp = $prepared.inv_ootf_exp_f32;

        let mut i = 0;
        while i + $lanes <= $len {
            let mut ra = [0.0f32; $lanes];
            let mut ga = [0.0f32; $lanes];
            let mut ba = [0.0f32; $lanes];
            ra.copy_from_slice(&$r[i..i + $lanes]);
            ga.copy_from_slice(&$g[i..i + $lanes]);
            ba.copy_from_slice(&$b[i..i + $lanes]);
            let (ro, go, bo) = hlg_eval_chunk!(
                $vec,
                $lanes,
                $token,
                $prepared,
                $inverse,
                $clip,
                ra,
                ga,
                ba,
                zero,
                one,
                half,
                neg_zero,
                inv_3,
                three,
                twelve,
                inv_12,
                b_const,
                c_const,
                a_ln2,
                inv_a_log2e,
                exp_max,
                ootf_lo,
                ootf_hi,
                lr,
                lg,
                lb,
                beta_v,
                one_minus_beta_v,
                inv_one_minus_beta_v,
                scale_v,
                inv_scale_v,
                gamma_m1,
                inv_ootf_exp
            );
            $r[i..i + $lanes].copy_from_slice(&ro);
            $g[i..i + $lanes].copy_from_slice(&go);
            $b[i..i + $lanes].copy_from_slice(&bo);
            i += $lanes;
        }
        let rem = $len - i;
        if rem > 0 {
            let mut ra = [0.5f32; $lanes];
            let mut ga = [0.5f32; $lanes];
            let mut ba = [0.5f32; $lanes];
            ra[..rem].copy_from_slice(&$r[i..$len]);
            ga[..rem].copy_from_slice(&$g[i..$len]);
            ba[..rem].copy_from_slice(&$b[i..$len]);
            let (ro, go, bo) = hlg_eval_chunk!(
                $vec,
                $lanes,
                $token,
                $prepared,
                $inverse,
                $clip,
                ra,
                ga,
                ba,
                zero,
                one,
                half,
                neg_zero,
                inv_3,
                three,
                twelve,
                inv_12,
                b_const,
                c_const,
                a_ln2,
                inv_a_log2e,
                exp_max,
                ootf_lo,
                ootf_hi,
                lr,
                lg,
                lb,
                beta_v,
                one_minus_beta_v,
                inv_one_minus_beta_v,
                scale_v,
                inv_scale_v,
                gamma_m1,
                inv_ootf_exp
            );
            $r[i..$len].copy_from_slice(&ro[..rem]);
            $g[i..$len].copy_from_slice(&go[..rem]);
            $b[i..$len].copy_from_slice(&bo[..rem]);
        }
    }};
}

macro_rules! hlg_soa_dispatch {
    ($vec:ident, $lanes:literal, $token:ident, $r:ident, $g:ident, $b:ident, $len:ident, $prepared:ident) => {{
        match ($prepared.inverse, $prepared.clip) {
            (false, false) => hlg_soa_loop!(
                $vec, $lanes, $token, $r, $g, $b, $len, $prepared, false, false
            ),
            (false, true) => hlg_soa_loop!(
                $vec, $lanes, $token, $r, $g, $b, $len, $prepared, false, true
            ),
            (true, false) => hlg_soa_loop!(
                $vec, $lanes, $token, $r, $g, $b, $len, $prepared, true, false
            ),
            (true, true) => hlg_soa_loop!(
                $vec, $lanes, $token, $r, $g, $b, $len, $prepared, true, true
            ),
        }
    }};
}

#[archmage::attune(_*_t, define(f32x8))]
fn hlg_soa_tier(
    token: Token,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    len: usize,
    prepared: &PreparedHlg,
) {
    hlg_soa_dispatch!(f32x8, 8, token, r, g, b, len, prepared);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn hlg_soa_tier_v4x_t(
    token: archmage::X64V4xToken,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    len: usize,
    prepared: &PreparedHlg,
) {
    type Vec16 = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    hlg_soa_dispatch!(Vec16, 16, token, r, g, b, len, prepared);
}

impl PreparedHlg {
    /// Apply BT.2100-3 display HLG EOTF or inverse EOTF directly on SoA `r, g, b` slices.
    ///
    /// When `simd == false`, runs the exact `ScalarToken` tier of the same `magemath`
    /// math so `Accelerators::SCALAR` and `Accelerators::ALL` match 0-bit identically.
    pub(crate) fn apply_soa(
        &self,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        len: usize,
        simd: bool,
    ) {
        if len == 0 {
            return;
        }
        if !self.simd_fast_eligible {
            for i in 0..len {
                let out = hlg_pixel_f64([r[i], g[i], b[i]], self);
                r[i] = out[0];
                g[i] = out[1];
                b[i] = out[2];
            }
            return;
        }
        if simd {
            archmage::attuned!(
                hlg_soa_tier(r, g, b, len, self),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        } else {
            hlg_soa_tier_scalar_t(archmage::ScalarToken, r, g, b, len, self);
        }
    }
}
