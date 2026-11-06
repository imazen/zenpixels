#![forbid(unsafe_code)]
// PR 2 cost: sign fix alone against the guarded evaluator and always-exact f_powf.
use std::sync::Arc;
use zenbench::prelude::*;
fn samples(n: usize, neg_pct: u32, zero_pct: u32) -> Arc<Vec<f32>> {
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
                if next() % 100 < zero_pct {
                    return 0.0;
                }
                if next() % 100 < neg_pct { -m } else { m }
            })
            .collect(),
    )
}
macro_rules! arm {
    ($g:ident, $name:literal, $m:ident, $input:expr, $encode:expr) => {{
        let input = $input.clone();
        let curve = $m::curve_from_gamma(2.19921875);
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
macro_rules! xf {
    ($g:ident, $name:literal, $m:ident, $input:expr) => {{
        let input = $input.clone();
        let (s, t) = (
            $m::ColorProfile::new_adobe_rgb(),
            $m::ColorProfile::new_pro_photo_rgb(),
        );
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
macro_rules! cicp {
    ($g:ident, $name:literal, $m:ident, $input:expr, $tc:ident) => {{
        let input = $input.clone();
        let mut s = $m::ColorProfile::new_srgb();
        s.cicp = Some($m::CicpProfile {
            color_primaries: $m::CicpColorPrimaries::Bt709,
            transfer_characteristics: $m::TransferCharacteristics::$tc,
            matrix_coefficients: $m::MatrixCoefficients::Bt709,
            full_range: true,
        });
        let t = $m::ColorProfile::new_bt2020();
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
fn underflow(suite: &mut Suite) {
    for n in [64usize, 1024, 4096] {
        for (dir, enc) in [("encode", true), ("decode", false)] {
            for (zero, neg) in [(0u32, 0u32), (0, 50), (10, 0), (100, 0)] {
                let input = samples(n, neg, zero);
                suite.group(format!("eval/{dir}/zero{zero}/neg{neg}/w{n}"), move |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    g.baseline("before");
                    arm!(g, "before", m0, input, enc);
                    arm!(g, "sign_fix", m2, input, enc);
                    arm!(g, "guarded", m4, input, enc);
                    arm!(g, "always_f_powf", m5, input, enc);
                });
            }
        }
        let input = samples(n, 0, 0);
        let i2 = input.clone();
        suite.group(format!("transform/cicp_gamma22_to_bt2020/w{n}"), move |g| {
            g.throughput(Throughput::Elements(n as u64));
            g.baseline("before");
            cicp!(g, "before", m0, i2, Bt470M);
            cicp!(g, "guarded", m4, i2, Bt470M);
        });
        let i3 = input.clone();
        suite.group(
            format!("transform/cicp_smpte428_to_bt2020/w{n}"),
            move |g| {
                g.throughput(Throughput::Elements(n as u64));
                g.baseline("before");
                cicp!(g, "before", m0, i3, Smpte428);
                cicp!(g, "guarded", m4, i3, Smpte428);
            },
        );
        let i4 = samples(n, 0, 0);
        suite.group(format!("transform/cicp_gamma28_to_bt2020/w{n}"), move |g| {
            g.throughput(Throughput::Elements(n as u64));
            g.baseline("before");
            cicp!(g, "before", m0, i4, Bt470Bg);
            cicp!(g, "guarded", m4, i4, Bt470Bg);
        });
        suite.group(
            format!("transform/adobe_to_prophoto_extended/w{n}"),
            move |g| {
                g.throughput(Throughput::Elements(n as u64));
                g.baseline("before");
                xf!(g, "before", m0, input);
                xf!(g, "sign_fix", m2, input);
                xf!(g, "guarded", m4, input);
                xf!(g, "always_f_powf", m5, input);
            },
        );
    }
}
zenbench::main!(underflow);
