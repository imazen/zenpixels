#![forbid(unsafe_code)]
// Bit-level comparison of each candidate against the unchanged evaluator, plus the exact-math error.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
macro_rules! ev {
    ($m:ident, $g:expr, $enc:expr) => {{
        let c = $m::curve_from_gamma($g);
        let e = if $enc {
            c.make_gamma_evaluator().unwrap()
        } else {
            c.make_linear_evaluator().unwrap()
        };
        Box::new(move |x: f32| e.evaluate_value(x)) as Box<dyn Fn(f32) -> f32 + Send + Sync>
    }};
}
#[derive(Default)]
struct Class {
    n: AtomicU64,
    same: [AtomicU64; 3],
}
fn exhaustive(threads: u64) {
    for &(g, enc) in &[
        (2.19921875f32, true),
        (2.19921875, false),
        (1.8, true),
        (1.8, false),
        (2.6, true),
        (2.6, false),
        (3.0, true),
        (3.0, false),
    ] {
        let (e0, e1, e2, e3) = (
            ev!(m0, g, enc),
            ev!(m1, g, enc),
            ev!(m2, g, enc),
            ev!(m3, g, enc),
        );
        let names = [
            "+finite", "+0", "+inf", "+NaN", "-finite", "-0", "-inf", "-NaN",
        ];
        let classes: [Class; 8] = Default::default();
        let odd = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        let neg_flip = AtomicU64::new(0);
        std::thread::scope(|sc| {
            for t in 0..threads {
                let (e0, e1, e2, e3, classes, odd, neg_flip) =
                    (&e0, &e1, &e2, &e3, &classes, &odd, &neg_flip);
                sc.spawn(move || {
                    let span = (1u64 << 32) / threads;
                    let (lo, hi) = (
                        t * span,
                        if t + 1 == threads {
                            1u64 << 32
                        } else {
                            (t + 1) * span
                        },
                    );
                    let (mut cn, mut cs) = ([0u64; 8], [[0u64; 3]; 8]);
                    let (mut co, mut cf) = ([0u64; 3], 0u64);
                    for b in lo..hi {
                        let x = f32::from_bits(b as u32);
                        let neg = x.is_sign_negative();
                        let k = if x.is_nan() {
                            3
                        } else if x.is_infinite() {
                            2
                        } else if x == 0.0 {
                            1
                        } else {
                            0
                        } + if neg { 4 } else { 0 };
                        let v0 = e0(x).to_bits();
                        let v = [e1(x).to_bits(), e2(x).to_bits(), e3(x).to_bits()];
                        cn[k] += 1;
                        for i in 0..3 {
                            if v[i] == v0 {
                                cs[k][i] += 1;
                            }
                        }
                        if k == 4 {
                            let mirror = e0(-x);
                            for i in 0..3 {
                                if v[i] == (-mirror).to_bits() {
                                    co[i] += 1;
                                }
                            }
                            if v0 == mirror.to_bits() && mirror > 0.0 {
                                cf += 1;
                            }
                        }
                    }
                    for k in 0..8 {
                        classes[k].n.fetch_add(cn[k], Relaxed);
                        for i in 0..3 {
                            classes[k].same[i].fetch_add(cs[k][i], Relaxed);
                        }
                    }
                    for i in 0..3 {
                        odd[i].fetch_add(co[i], Relaxed);
                    }
                    neg_flip.fetch_add(cf, Relaxed);
                });
            }
        });
        println!(
            "--- all 2^32 f32 inputs, curve {g} {} ---",
            if enc { "encode" } else { "decode" }
        );
        for k in 0..8 {
            let c = &classes[k];
            println!(
                "{:>8} n={:>11} bit-identical to before: copysign {:>11} branch {:>11} bitor {:>11}",
                names[k],
                c.n.load(Relaxed),
                c.same[0].load(Relaxed),
                c.same[1].load(Relaxed),
                c.same[2].load(Relaxed)
            );
        }
        println!(
            " -finite: after == -(before at |x|): copysign {} branch {} bitor {}; before returned the positive mirror on {}",
            odd[0].load(Relaxed),
            odd[1].load(Relaxed),
            odd[2].load(Relaxed),
            neg_flip.load(Relaxed)
        );
    }
}
fn error_of_power() {
    // Relative error of the evaluator against exact f64 powers, over a log sweep of magnitudes.
    for &(g, enc) in &[
        (2.19921875f32, true),
        (2.19921875, false),
        (1.8, true),
        (1.8, false),
        (2.6, true),
        (2.6, false),
        (2.8, true),
        (2.8, false),
        (3.0, true),
        (3.0, false),
        (4.0, true),
        (4.0, false),
        (1.0, true),
        (1.0, false),
    ] {
        let e = ev!(m1, g, enc);
        let fixed = (g as f64 * 256.0 + 0.5).floor();
        let p = if enc { 256.0 / fixed } else { fixed / 256.0 };
        let (mut worst, mut at) = (0f64, 0f32);
        let (lo, hi) = (1e-4f32.to_bits(), 8f32.to_bits());
        for b in (lo..=hi).step_by(17) {
            let x = f32::from_bits(b);
            let exact = (x as f64).powf(p);
            let r = (((-e(-x)) as f64 - exact) / exact).abs();
            if r > worst {
                worst = r;
                at = x;
            }
        }
        println!(
            "power error curve {g} {} p={p:.6}: worst relative {worst:.3e} at |x|={at}",
            if enc { "encode" } else { "decode" }
        );
    }
}
fn underflow() {
    // Where the unchanged evaluator stops matching exact math on the positive axis.
    for &(g, enc) in &[
        (1.0f32, false),
        (1.8, false),
        (2.19921875, false),
        (2.6, false),
        (2.8, false),
        (3.0, false),
        (4.0, false),
        (1.8, true),
        (2.19921875, true),
        (2.6, true),
    ] {
        let e = ev!(m0, g, enc);
        let fixed = (g as f64 * 256.0 + 0.5).floor();
        let p = if enc { 256.0 / fixed } else { fixed / 256.0 };
        let (mut largest_bad, mut bad, mut negative, mut huge) = (0f32, 0u64, 0u64, 0u64);
        for b in 0..=1f32.to_bits() {
            let x = f32::from_bits(b);
            let v = e(x) as f64;
            let exact = (x as f64).powf(p);
            // Absolute test: an f32 result this far from a value in [0, 1] is wrong at any depth.
            if (v - exact).abs() > 1e-7_f64.max(exact * 1e-3) {
                bad += 1;
                largest_bad = x;
            }
            if v < 0.0 {
                negative += 1;
            }
            if v.abs() > 1.0 + 1e-3 {
                huge += 1;
            }
        }
        println!(
            "x in [0,1] curve {g} {}: wrong {bad} of {}; largest wrong x {largest_bad:e}; negative results {negative}; |result|>1 {huge}; f(0)={:e}",
            if enc { "encode" } else { "decode" },
            1f32.to_bits() as u64 + 1,
            e(0.0)
        );
    }
}
fn srgb_dec(x: f64) -> f64 {
    if x < 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
macro_rules! cube_run {
    ($m:ident, $input:expr) => {{
        let (s, t) = (
            $m::ColorProfile::new_display_p3(),
            $m::ColorProfile::new_adobe_rgb(),
        );
        let o = $m::TransformOptions {
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        };
        let tr = s
            .create_transform_f32($m::Layout::Rgb, &t, $m::Layout::Rgb, o)
            .unwrap();
        let mut out = vec![0f32; $input.len()];
        tr.transform($input, &mut out).unwrap();
        out
    }};
}
macro_rules! cube_u8 {
    ($m:ident, $input:expr) => {{
        let (s, t) = (
            $m::ColorProfile::new_display_p3(),
            $m::ColorProfile::new_adobe_rgb(),
        );
        let tr = s
            .create_transform_8bit($m::Layout::Rgb, &t, $m::Layout::Rgb, Default::default())
            .unwrap();
        let mut out = vec![0u8; $input.len()];
        tr.transform($input, &mut out).unwrap();
        out
    }};
}
fn cube() {
    let mut f = Vec::with_capacity(3 << 24);
    let mut b = Vec::with_capacity(3 << 24);
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for bl in 0..=255u8 {
                for v in [r, g, bl] {
                    f.push(v as f32 / 255.0);
                    b.push(v);
                }
            }
        }
    }
    let (s, t) = (
        m0::ColorProfile::new_display_p3(),
        m0::ColorProfile::new_adobe_rgb(),
    );
    let m = t
        .rgb_to_xyz_matrix()
        .inverse()
        .mat_mul(s.rgb_to_xyz_matrix())
        .v;
    let lin: Vec<f64> = (0..256).map(|i| srgb_dec(i as f64 / 255.0)).collect();
    let p = 256.0 / 563.0;
    let outs = [
        ("before", cube_run!(m0, &f)),
        ("copysign", cube_run!(m1, &f)),
        ("branch", cube_run!(m2, &f)),
        ("bitor", cube_run!(m3, &f)),
    ];
    for (name, o) in &outs {
        let (
            mut n,
            mut sum,
            mut worst,
            mut wrong_sign,
            mut changed,
            mut changed_nonneg,
            mut neg_exact,
        ) = (0u64, 0f64, 0f64, 0u64, 0u64, 0u64, 0u64);
        for (i, (src, px)) in b.chunks_exact(3).zip(o.chunks_exact(3)).enumerate() {
            let l = [
                lin[src[0] as usize],
                lin[src[1] as usize],
                lin[src[2] as usize],
            ];
            for c in 0..3 {
                let v = m[c][0] * l[0] + m[c][1] * l[1] + m[c][2] * l[2];
                let exact = v.clamp(0.0, 1.0).powf(p) * 255.0;
                let got = (px[c] as f64).clamp(0.0, 1.0) * 255.0;
                let d = got - exact;
                n += 1;
                sum += d;
                if d.abs() > worst {
                    worst = d.abs();
                }
                if v < -1e-6 {
                    neg_exact += 1;
                    if px[c] > 0.0 {
                        wrong_sign += 1;
                    }
                }
                let before = outs[0].1[i * 3 + c];
                if px[c].to_bits() != before.to_bits() {
                    changed += 1;
                    if before.is_nan() || px[c].to_bits() != (-before).to_bits() || v >= 0.0 {
                        changed_nonneg += 1;
                    }
                }
            }
        }
        println!(
            "2^24 P3->Adobe f32 extended, {name:>8}: clipped 8-bit error mean {:+.4} worst {worst:.3}; exact-negative channels {neg_exact}, of those positive {wrong_sign}; bits changed vs before {changed}, of those not a pure sign flip of an exact-negative channel {changed_nonneg}; n={n}",
            sum / n as f64
        );
    }
    let u0 = cube_u8!(m0, &b);
    for (name, o) in [
        ("copysign", cube_u8!(m1, &b)),
        ("branch", cube_u8!(m2, &b)),
        ("bitor", cube_u8!(m3, &b)),
    ] {
        println!(
            "2^24 P3->Adobe u8 default table route, {name}: {} of {} bytes differ from before",
            o.iter().zip(&u0).filter(|(a, b)| a != b).count(),
            o.len()
        );
    }
}
fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8u64)
        .min(8);
    println!("threads {threads}");
    error_of_power();
    underflow();
    cube();
    exhaustive(threads);
}
