#![forbid(unsafe_code)]
macro_rules! compare {
    ($m:ident,$name:literal) => {{
        let mut values = Vec::new();
        for i in 0..=65535u32 {
            values.push(i as f32 / 65535.0);
        }
        for bits in [
            0u32,
            1,
            0x0080_0000,
            0x2980_0000,
            0x3f80_0000,
            0x5580_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0x7fc0_0000,
        ] {
            for offset in 0..=32 {
                values.push(f32::from_bits(bits.wrapping_add(offset)));
                values.push(f32::from_bits(bits.wrapping_sub(offset)));
            }
        }
        let mut state = 0x9e3779b9u32;
        for _ in 0..262144 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values.push(f32::from_bits(state));
        }
        let mut changed = 0u64;
        let mut pixel_difference = 0u64;
        for gamma in [1f32, 1.8, 563.0 / 256.0, 2.6, 2.8, 3., 4.] {
            for encode in [false, true] {
                let c = $m::curve_from_gamma(gamma);
                let r = review_range::curve_from_gamma(gamma);
                let (candidate, reference) = if encode {
                    (
                        c.make_gamma_evaluator().unwrap(),
                        r.make_gamma_evaluator().unwrap(),
                    )
                } else {
                    (
                        c.make_linear_evaluator().unwrap(),
                        r.make_linear_evaluator().unwrap(),
                    )
                };
                for &x in &values {
                    let actual = candidate.evaluate_value(x);
                    let expected = reference.evaluate_value(x);
                    changed += u64::from(
                        !(actual.is_nan() && expected.is_nan())
                            && actual.to_bits() != expected.to_bits(),
                    );
                    let pixel = candidate.evaluate_tristimulus($m::Rgb::new(x, 0.5, -0.5));
                    if !(pixel.r.is_nan() && actual.is_nan())
                        && pixel.r.to_bits() != actual.to_bits()
                    {
                        pixel_difference += 1;
                    }
                }
            }
        }
        println!(
            "{} samples={} reference_bit_changes={} pixel_scalar_disagreement={}",
            $name,
            values.len() * 14,
            changed,
            pixel_difference
        );
        assert_eq!(pixel_difference, 0, $name);
        if $name != "opt_wide" {
            assert_eq!(changed, 0, $name);
        }
    }};
}
fn main() {
    compare!(opt_or, "opt_or");
    compare!(opt_pixel, "opt_pixel");
    compare!(opt_wide, "opt_wide");
}
