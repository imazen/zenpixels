use archmage::ScalarToken;
use magemath::powf::{Powf, PowfOptions};

const MAGNITUDE_POWER: u128 = PowfOptions::new()
    .reject_subnormals()
    .highp()
    .caller()
    .bits();

/// High-precision power on positive normal magnitudes. The existing libm path
/// owns exceptional inputs and values outside the checked approximation range.
#[inline]
pub(crate) fn magnitude_pow(base: f32, exponent: f32) -> f32 {
    Powf::<MAGNITUDE_POWER>::scalar_with_token(ScalarToken, base, exponent)
        .unwrap_or_else(|_| libm::powf(base, exponent))
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
}
