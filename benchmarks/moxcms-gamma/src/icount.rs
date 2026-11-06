#![forbid(unsafe_code)]
// Compile the exact proposed PR revisions when validating a review report.
#[cfg(feature = "review-revisions")]
use review_before as m0;
#[cfg(feature = "review-revisions")]
use review_range as m4;
#[cfg(feature = "review-revisions")]
use review_sign as m2;
// Instruction counts under callgrind: `run_work` is the only collected function.
// usage: icount <before|copysign|branch|bitor> <eval_encode|eval_decode|p3_adobe_f32|srgb_adobe_f32|p3_adobe_u8> <neg_pct> <pixels>
use std::hint::black_box;
fn samples(n: usize, neg_pct: u32) -> Vec<f32> {
    let mut s = 0x9E3779B9u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    (0..n * 3)
        .map(|_| {
            let m = (next() >> 8) as f32 / (1u32 << 24) as f32 + 1e-4;
            if next() % 100 < neg_pct { -m } else { m }
        })
        .collect()
}
fn cube8(n: usize) -> Vec<u8> {
    let mut s = 0x2545F491u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    (0..n * 3).map(|_| (next() >> 24) as u8).collect()
}
#[inline(never)]
pub fn run_work(f: &mut dyn FnMut()) {
    f()
}

macro_rules! variant {
    ($m:ident, $load:expr, $neg:expr, $n:expr) => {{
        match $load {
            "eval_encode" | "eval_decode" => {
                let curve = $m::curve_from_gamma(563.0 / 256.0);
                let e = if $load == "eval_encode" {
                    curve.make_gamma_evaluator().unwrap()
                } else {
                    curve.make_linear_evaluator().unwrap()
                };
                let input = samples($n, $neg);
                let mut out = vec![0f32; input.len()];
                run_work(&mut || {
                    for (s, d) in black_box(&input[..])
                        .chunks_exact(3)
                        .zip(out.chunks_exact_mut(3))
                    {
                        let v = e.evaluate_tristimulus($m::Rgb::new(s[0], s[1], s[2]));
                        d[0] = v.r;
                        d[1] = v.g;
                        d[2] = v.b;
                    }
                });
                black_box(&out);
            }
            "p3_adobe_f32" | "srgb_adobe_f32" => {
                let s = if $load == "p3_adobe_f32" {
                    $m::ColorProfile::new_display_p3()
                } else {
                    $m::ColorProfile::new_srgb()
                };
                let t = $m::ColorProfile::new_adobe_rgb();
                let o = $m::TransformOptions {
                    allow_extended_range_rgb_xyz: true,
                    ..Default::default()
                };
                let tr = s
                    .create_transform_f32($m::Layout::Rgb, &t, $m::Layout::Rgb, o)
                    .unwrap();
                let input: Vec<f32> = cube8($n).iter().map(|&v| v as f32 / 255.0).collect();
                let mut out = vec![0f32; input.len()];
                run_work(&mut || tr.transform(black_box(&input), &mut out).unwrap());
                black_box(&out);
            }
            "prophoto_srgb_zero10"
            | "prophoto_srgb_zero100"
            | "cicp22_bt2020"
            | "cicp28_bt2020" => {
                let mut source = if $load.starts_with("prophoto") {
                    $m::ColorProfile::new_pro_photo_rgb()
                } else {
                    $m::ColorProfile::new_srgb()
                };
                if $load.starts_with("cicp") {
                    source.cicp = Some($m::CicpProfile {
                        color_primaries: $m::CicpColorPrimaries::Bt709,
                        transfer_characteristics: if $load == "cicp22_bt2020" {
                            $m::TransferCharacteristics::Bt470M
                        } else {
                            $m::TransferCharacteristics::Bt470Bg
                        },
                        matrix_coefficients: $m::MatrixCoefficients::Bt709,
                        full_range: true,
                    });
                }
                let target = if $load.starts_with("prophoto") {
                    $m::ColorProfile::new_srgb()
                } else {
                    $m::ColorProfile::new_bt2020()
                };
                let transform = source
                    .create_transform_f32(
                        $m::Layout::Rgb,
                        &target,
                        $m::Layout::Rgb,
                        $m::TransformOptions {
                            allow_extended_range_rgb_xyz: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let mut input = samples($n, $neg);
                if $load.ends_with("zero100") {
                    input.fill(0.0);
                } else if $load.ends_with("zero10") {
                    for x in input.iter_mut().step_by(10) {
                        *x = 0.0;
                    }
                }
                let mut out = vec![0f32; input.len()];
                run_work(&mut || transform.transform(black_box(&input), &mut out).unwrap());
                black_box(&out);
            }
            "p3_adobe_u8" => {
                let (s, t) = (
                    $m::ColorProfile::new_display_p3(),
                    $m::ColorProfile::new_adobe_rgb(),
                );
                let tr = s
                    .create_transform_8bit($m::Layout::Rgb, &t, $m::Layout::Rgb, Default::default())
                    .unwrap();
                let input = cube8($n);
                let mut out = vec![0u8; input.len()];
                run_work(&mut || tr.transform(black_box(&input), &mut out).unwrap());
                black_box(&out);
            }
            other => panic!("{other}"),
        }
    }};
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (neg, n): (u32, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    match a[1].as_str() {
        "before" => variant!(m0, a[2].as_str(), neg, n),
        "copysign" => variant!(m1, a[2].as_str(), neg, n),
        "branch" => variant!(m2, a[2].as_str(), neg, n),
        "bitor" => variant!(m3, a[2].as_str(), neg, n),
        "guarded" => variant!(m4, a[2].as_str(), neg, n),
        "always_f_powf" => variant!(m5, a[2].as_str(), neg, n),
        #[cfg(feature = "optimized")]
        "opt_or" => variant!(opt_or, a[2].as_str(), neg, n),
        #[cfg(feature = "optimized")]
        "opt_wide" => variant!(opt_wide, a[2].as_str(), neg, n),
        #[cfg(feature = "optimized")]
        "opt_pixel" => variant!(opt_pixel, a[2].as_str(), neg, n),
        #[cfg(feature = "outlined")]
        "opt_cold" => variant!(opt_cold, a[2].as_str(), neg, n),
        other => panic!("{other}"),
    }
}
