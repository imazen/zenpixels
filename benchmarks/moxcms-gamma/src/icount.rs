#![forbid(unsafe_code)]
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
                let curve = $m::curve_from_gamma(2.19921875);
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
        other => panic!("{other}"),
    }
}
