#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken, X64V3Token, arcane};
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
fn mage<T: F32x8Convert>(token: T, values: [f32; 8], exponent: f32) -> [[f32; 8]; 2] {
    let v = f32x8::from_array_t(token, values);
    [
        v.pow_midp(exponent).to_array(),
        v.pow_lowp(exponent).to_array(),
    ]
}
#[arcane]
fn avx2(token: X64V3Token, values: [f32; 8], exponent: f32) -> [[f32; 8]; 2] {
    mage(token, values, exponent)
}
fn scalar(values: [f32; 8], exponent: f32) -> [[f32; 8]; 2] {
    mage(ScalarToken, values, exponent)
}
#[derive(Default)]
struct Stats {
    max_rel: f64,
    max_abs: f64,
    normal_refs: u64,
    nonfinite: u64,
    zero_refs: u64,
    wrong_zero: u64,
    subnormal_refs: u64,
    wrong_subnormal: u64,
    infinite_refs: u64,
    wrong_infinity: u64,
}
impl Stats {
    fn add(&mut self, value: f32, reference: f64) {
        let rounded = reference as f32;
        if rounded == 0.0 {
            self.zero_refs += 1;
            self.wrong_zero += u64::from(value != 0.0);
        } else if rounded.is_infinite() {
            self.infinite_refs += 1;
            self.wrong_infinity += u64::from(value != rounded);
        } else if rounded.is_subnormal() {
            self.subnormal_refs += 1;
            self.wrong_subnormal += u64::from(value != rounded);
        } else {
            self.normal_refs += 1;
            if value.is_finite() {
                let error = (f64::from(value) - reference).abs();
                self.max_abs = self.max_abs.max(error);
                self.max_rel = self.max_rel.max(error / reference.abs());
            } else {
                self.nonfinite += 1;
            }
        }
    }
    fn print(&self, name: &str, band: &str, exponent: f32) {
        println!(
            "{name}\t{band}\t{exponent:.9}\t{}\t{:.9e}\t{:.9e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.normal_refs,
            self.max_rel,
            self.max_abs,
            self.nonfinite,
            self.zero_refs,
            self.wrong_zero,
            self.subnormal_refs,
            self.wrong_subnormal,
            self.infinite_refs,
            self.wrong_infinity,
            self.normal_refs + self.zero_refs + self.subnormal_refs + self.infinite_refs
        );
    }
}
fn main() {
    let token = X64V3Token::summon().expect("caller must select an AVX2/FMA host");
    println!(
        "# Numeric comparison, not a timing benchmark; f64 powf reference of actual f32 x and exponent."
    );
    println!(
        "implementation\tband\texponent\tnormal_references\tmax_relative_error_normal\tmax_absolute_error_normal\tnonfinite_for_normal\tzero_references\tnonzero_for_zero\tsubnormal_references\twrong_subnormal\tinfinite_references\twrong_infinite\tinputs"
    );
    for gamma in [1.0f32, 1.8, 2.2, 2.4, 2.6, 2.8, 4.0] {
        for exponent in [gamma, 1.0 / gamma] {
            for band in ["unit-u16", "extended-16", "wide-positive-normal"] {
                let mut stats: [Stats; 5] = std::array::from_fn(|_| Stats::default());
                for first in (0..65536).step_by(8) {
                    let input = std::array::from_fn(|lane| {
                        let i = first + lane;
                        match band {
                            "unit-u16" => i as f32 / 65535.0,
                            "extended-16" => i as f32 * 16.0 / 65535.0,
                            _ => 2.0f64.powf(-126.0 + 253.0 * (i as f64 / 65535.0)) as f32,
                        }
                    });
                    let s = scalar(input, exponent);
                    let v = avx2(token, input, exponent);
                    for lane in 0..8 {
                        let reference = f64::from(input[lane]).powf(f64::from(exponent));
                        let outputs = [
                            pxfm::dirty_powf(input[lane], exponent),
                            s[0][lane],
                            s[1][lane],
                            v[0][lane],
                            v[1][lane],
                        ];
                        for (stat, output) in stats.iter_mut().zip(outputs) {
                            stat.add(output, reference);
                        }
                    }
                }
                for (name, stat) in [
                    "dirty",
                    "midp-scalar",
                    "lowp-scalar",
                    "midp-avx2",
                    "lowp-avx2",
                ]
                .into_iter()
                .zip(stats)
                {
                    stat.print(name, band, exponent);
                }
            }
        }
    }
    eprintln!(
        "EDGE\tx\texponent\tdirty\tmidp_scalar\tlowp_scalar\tmidp_avx2\tlowp_avx2\tf64_reference"
    );
    for exponent in [1.8f32, 1.0 / 1.8, 2.2, 1.0 / 2.2, 0.0, -1.0] {
        for x in [
            0.0,
            -0.0,
            f32::from_bits(1),
            1e-30,
            1e-20,
            0.5,
            -0.5,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ] {
            let s = scalar([x; 8], exponent);
            let v = avx2(token, [x; 8], exponent);
            eprintln!(
                "EDGE\t{x:?}\t{exponent:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}",
                pxfm::dirty_powf(x, exponent),
                s[0][0],
                s[1][0],
                v[0][0],
                v[1][0],
                f64::from(x).powf(f64::from(exponent))
            );
        }
    }
}
