#![forbid(unsafe_code)]
// Paired before/after: the pure-gamma evaluator and the public transforms that reach it.
use std::sync::Arc;
use zenbench::prelude::*;

/// Deterministic row of RGB f32 with `neg_pct` percent of samples negated, magnitudes in (0, 1].
fn samples(n: usize, neg_pct: u32) -> Arc<Vec<f32>> {
    let mut s = 0x9E3779B9u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    Arc::new(
        (0..n * 3)
            .map(|_| {
                let m = (next() >> 8) as f32 / (1u32 << 24) as f32 + 1e-4;
                if next() % 100 < neg_pct { -m } else { m }
            })
            .collect(),
    )
}
fn cube(n: usize) -> Arc<Vec<f32>> {
    let mut s = 0x2545F491u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    Arc::new((0..n * 3).map(|_| (next() >> 24) as f32 / 255.0).collect())
}
fn cube8(n: usize) -> Arc<Vec<u8>> {
    let mut s = 0x2545F491u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    Arc::new((0..n * 3).map(|_| (next() >> 24) as u8).collect())
}

macro_rules! eval_arm {
    ($g:ident, $name:literal, $m:ident, $input:expr, $encode:expr, $gamma:expr) => {{
        let input = $input.clone();
        let curve = $m::curve_from_gamma($gamma);
        let e = if $encode {
            curve.make_gamma_evaluator().unwrap()
        } else {
            curve.make_linear_evaluator().unwrap()
        };
        let e: Arc<dyn $m::ToneCurveEvaluator + Send + Sync> = Arc::from(e);
        $g.bench($name, move |b| {
            let (input, e) = (input.clone(), e.clone());
            let mut out = vec![0f32; input.len()];
            b.iter(move || {
                for (s, d) in black_box(&input[..])
                    .chunks_exact(3)
                    .zip(out.chunks_exact_mut(3))
                {
                    let v = e.evaluate_tristimulus($m::Rgb::new(s[0], s[1], s[2]));
                    d[0] = v.r;
                    d[1] = v.g;
                    d[2] = v.b;
                }
                black_box(out[0])
            })
        });
    }};
}
macro_rules! f32_arm {
    ($g:ident, $name:literal, $m:ident, $input:expr, $src:ident) => {{
        let input = $input.clone();
        let (s, t) = ($m::ColorProfile::$src(), $m::ColorProfile::new_adobe_rgb());
        let o = $m::TransformOptions {
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        };
        let tr = s
            .create_transform_f32($m::Layout::Rgb, &t, $m::Layout::Rgb, o)
            .unwrap();
        $g.bench($name, move |b| {
            let (input, tr) = (input.clone(), tr.clone());
            let mut out = vec![0f32; input.len()];
            b.iter(move || {
                tr.transform(black_box(&input), &mut out).unwrap();
                black_box(out[0])
            })
        });
    }};
}
macro_rules! u8_arm {
    ($g:ident, $name:literal, $m:ident, $input:expr) => {{
        let input = $input.clone();
        let (s, t) = (
            $m::ColorProfile::new_display_p3(),
            $m::ColorProfile::new_adobe_rgb(),
        );
        let tr = s
            .create_transform_8bit($m::Layout::Rgb, &t, $m::Layout::Rgb, Default::default())
            .unwrap();
        $g.bench($name, move |b| {
            let (input, tr) = (input.clone(), tr.clone());
            let mut out = vec![0u8; input.len()];
            b.iter(move || {
                tr.transform(black_box(&input), &mut out).unwrap();
                black_box(out[0])
            })
        });
    }};
}

fn gamma(suite: &mut Suite) {
    for n in [64usize, 1024, 4096] {
        for (dir, encode, g) in [
            ("encode", true, 2.19921875f32),
            ("decode", false, 2.19921875),
        ] {
            for neg in [0u32, 10, 50, 100] {
                let input = samples(n, neg);
                suite.group(format!("eval/{dir}/neg{neg}/w{n}"), move |g2| {
                    g2.throughput(Throughput::Elements(n as u64));
                    g2.baseline("before");
                    eval_arm!(g2, "before", m0, input, encode, g);
                    eval_arm!(g2, "copysign", m1, input, encode, g);
                    eval_arm!(g2, "branch", m2, input, encode, g);
                    eval_arm!(g2, "bitor", m3, input, encode, g);
                });
            }
        }
        let c = cube(n);
        for (name, _) in [("p3_to_adobe_extended", 0), ("srgb_to_adobe_extended", 1)] {
            let c = c.clone();
            suite.group(format!("transform/{name}/w{n}"), move |g2| {
                g2.throughput(Throughput::Elements(n as u64));
                g2.baseline("before");
                if name.starts_with("p3") {
                    f32_arm!(g2, "before", m0, c, new_display_p3);
                    f32_arm!(g2, "copysign", m1, c, new_display_p3);
                    f32_arm!(g2, "branch", m2, c, new_display_p3);
                    f32_arm!(g2, "bitor", m3, c, new_display_p3);
                } else {
                    f32_arm!(g2, "before", m0, c, new_srgb);
                    f32_arm!(g2, "copysign", m1, c, new_srgb);
                    f32_arm!(g2, "branch", m2, c, new_srgb);
                    f32_arm!(g2, "bitor", m3, c, new_srgb);
                }
            });
        }
        let c8 = cube8(n);
        suite.group(
            format!("control_table_route/p3_to_adobe_u8/w{n}"),
            move |g2| {
                g2.throughput(Throughput::Elements(n as u64));
                g2.baseline("before");
                u8_arm!(g2, "before", m0, c8);
                u8_arm!(g2, "copysign", m1, c8);
                u8_arm!(g2, "branch", m2, c8);
                u8_arm!(g2, "bitor", m3, c8);
            },
        );
    }
}
zenbench::main!(gamma);
