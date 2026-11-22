#[cfg(any(feature = "cms-moxcms", test))]
use archmage::ScalarToken;
#[cfg(any(feature = "cms-moxcms", test))]
use magemath::SimdMath;
#[cfg(any(feature = "cms-moxcms", test))]
use magemath::powf::{Powf, PowfDomain, PowfOptions};

#[cfg(any(feature = "cms-moxcms", test))]
const MAGNITUDE_POWER: u128 = PowfOptions::new()
    .reject_subnormals()
    .highp()
    .caller()
    .bits();

/// Elided `Highp` gamma profile for positive normal bases in `[2^-16, 2^16]`
/// and exponents in `[0.0, 4.0]` (`|n * log2(x)| <= 64 <= 125`), eliding both
/// per-lane input/output normal checks in `magemath`'s packed binary64 path.
#[cfg(any(feature = "cms-moxcms", test))]
const MAGNITUDE_POWER_GAMMA: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveGamma)
    .reject_subnormals()
    .highp()
    .assume_in_bounds()
    .caller()
    .bits();

/// High-precision power on positive normal magnitudes. The existing libm path
/// owns exceptional inputs and values outside the checked approximation range.
#[cfg(any(feature = "cms-moxcms", test))]
#[inline]
pub(crate) fn magnitude_pow(base: f32, exponent: f32) -> f32 {
    if (1.0 / 65_536.0..=65_536.0).contains(&base) && (0.0..=4.0).contains(&exponent) {
        return Powf::<MAGNITUDE_POWER_GAMMA>::scalar_with_token(ScalarToken, base, exponent)
            .unwrap_or_else(|_| libm::powf(base, exponent));
    }
    Powf::<MAGNITUDE_POWER>::scalar_with_token(ScalarToken, base, exponent)
        .unwrap_or_else(|_| libm::powf(base, exponent))
}

/// Evaluates `|v|^exponent * sign(v)` across RGB (`channels == 3`) or RGBA (`channels == 4`)
/// buffers using 8-wide SIMD `Powf` (dispatched once per buffer via `attuned!`, canonical
/// bit-exact `highp` with `.caller()`), falling back per-lane to [`magnitude_pow`] on
/// chunks containing subnormals or non-finites.
#[cfg(any(feature = "cms-moxcms", test))]
pub(crate) fn apply_signed_pow_rgb(target: &mut [f32], channels: usize, exponent: f32) {
    archmage::attuned!(
        apply_signed_pow_rgb_tier(target, channels, exponent),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[cfg(any(feature = "cms-moxcms", test))]
#[archmage::attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn apply_signed_pow_rgb_tier(token: Token, target: &mut [f32], channels: usize, exponent: f32) {
    let gamma_exp = (0.0..=4.0).contains(&exponent);
    let lo_g = f32x8::splat_t(token, 1.0 / 65_536.0);
    let hi_g = f32x8::splat_t(token, 65_536.0);
    let eval_vec = |v: f32x8| -> Option<f32x8> {
        if gamma_exp
            && (v.simd_ge(lo_g) & v.simd_le(hi_g))
                .bitcast_to_i32()
                .all_true()
        {
            Some(v.powf::<MAGNITUDE_POWER_GAMMA>(exponent))
        } else {
            v.try_powf_with::<MAGNITUDE_POWER>(exponent).ok()
        }
    };
    match channels {
        3 => {
            let usable = (target.len() / 3) * 3;
            let (chunks, tail) = target[..usable].as_chunks_mut::<8>();
            for chunk in chunks {
                let orig = *chunk;
                let abs = orig.map(f32::abs);
                if let Some(out_vec) = eval_vec(f32x8::from_array_t(token, abs)) {
                    let out = out_vec.to_array();
                    for i in 0..8 {
                        chunk[i] = out[i].copysign(orig[i]);
                    }
                } else {
                    for i in 0..8 {
                        chunk[i] = magnitude_pow(abs[i], exponent).copysign(orig[i]);
                    }
                }
            }
            for v in tail {
                *v = magnitude_pow(v.abs(), exponent).copysign(*v);
            }
        }
        4 => {
            let (pairs, tail) = target.as_chunks_mut::<8>();
            for pair in pairs {
                let orig = *pair;
                let abs = [
                    orig[0].abs(),
                    orig[1].abs(),
                    orig[2].abs(),
                    1.0,
                    orig[4].abs(),
                    orig[5].abs(),
                    orig[6].abs(),
                    1.0,
                ];
                if let Some(out_vec) = eval_vec(f32x8::from_array_t(token, abs)) {
                    let out = out_vec.to_array();
                    pair[0] = out[0].copysign(orig[0]);
                    pair[1] = out[1].copysign(orig[1]);
                    pair[2] = out[2].copysign(orig[2]);
                    pair[4] = out[4].copysign(orig[4]);
                    pair[5] = out[5].copysign(orig[5]);
                    pair[6] = out[6].copysign(orig[6]);
                } else {
                    for i in [0, 1, 2, 4, 5, 6] {
                        pair[i] = magnitude_pow(abs[i], exponent).copysign(orig[i]);
                    }
                }
            }
            for pixel in tail.as_chunks_mut::<4>().0 {
                for v in &mut pixel[..3] {
                    *v = magnitude_pow(v.abs(), exponent).copysign(*v);
                }
            }
        }
        _ => {
            for pixel in target.chunks_exact_mut(channels) {
                for v in &mut pixel[..3] {
                    *v = magnitude_pow(v.abs(), exponent).copysign(*v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_on_extended_gamma_magnitudes() {
        for exponent in [0.0, 1.0 / 2.2, 1.8, 2.2, 2.4, 5.0, f32::INFINITY] {
            for value in [
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                0.2,
                1.0,
                1.25,
                8.5,
                200.0,
                f32::MAX,
                f32::INFINITY,
                f32::NAN,
            ] {
                let result = magnitude_pow(value, exponent);
                let expected = libm::powf(value, exponent);
                if expected.is_finite() && expected != 0.0 {
                    assert!(
                        result.to_bits().abs_diff(expected.to_bits()) <= 1,
                        "{value}^{exponent}"
                    );
                } else if expected.is_nan() {
                    assert!(result.is_nan(), "{value}^{exponent}");
                } else {
                    assert_eq!(result.to_bits(), expected.to_bits(), "{value}^{exponent}");
                }
            }
        }
    }

    #[test]
    fn simd_apply_signed_pow_rgb_matches_scalar_bit_for_bit() {
        let samples = [
            -2.5f32,
            -1.0,
            -0.25,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            0.18,
            0.5,
            1.0,
            1.75,
            12.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.33,
            -0.77,
            0.91,
            0.04,
            2.2,
            -0.01,
            0.62,
            0.88,
            -1.4,
            0.12,
        ];
        for exponent in [1.0 / 2.2, 1.8, 2.2, 2.4] {
            for channels in [3usize, 4] {
                let mut got = samples;
                let mut expected = samples;
                for px in expected.chunks_exact_mut(channels) {
                    for v in &mut px[..3] {
                        *v = magnitude_pow(v.abs(), exponent).copysign(*v);
                    }
                }
                apply_signed_pow_rgb(&mut got, channels, exponent);
                for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                    if e.is_nan() {
                        assert!(g.is_nan(), "ch={channels} exp={exponent} idx={i}");
                    } else {
                        assert_eq!(
                            g.to_bits(),
                            e.to_bits(),
                            "ch={channels} exp={exponent} idx={i}: got {g} expected {e}"
                        );
                    }
                }
            }
        }
    }
}
