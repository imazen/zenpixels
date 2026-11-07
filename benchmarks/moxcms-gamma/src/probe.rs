#![forbid(unsafe_code)]
// Behaviour table of the pure-gamma evaluator: before (m0) against each candidate.
macro_rules! ev {
    ($m:ident, $g:expr, $inv:expr) => {{
        let c = $m::curve_from_gamma($g);
        let e = if $inv {
            c.make_gamma_evaluator().unwrap()
        } else {
            c.make_linear_evaluator().unwrap()
        };
        Box::new(move |x: f32| e.evaluate_value(x)) as Box<dyn Fn(f32) -> f32 + Send + Sync>
    }};
}
fn main() {
    let specials: &[(&str, f32)] = &[
        ("+0", 0.0),
        ("-0", -0.0),
        ("+min_sub", f32::from_bits(1)),
        ("-min_sub", -f32::from_bits(1)),
        ("+min_norm", f32::MIN_POSITIVE),
        ("-min_norm", -f32::MIN_POSITIVE),
        ("+1e-30", 1e-30),
        ("-1e-30", -1e-30),
        ("+1e-10", 1e-10),
        ("-1e-10", -1e-10),
        ("+1e-4", 1e-4),
        ("-1e-4", -1e-4),
        ("-0.051", -0.051),
        ("+0.5", 0.5),
        ("-0.5", -0.5),
        ("+1", 1.0),
        ("-1", -1.0),
        ("+1.5", 1.5),
        ("-1.5", -1.5),
        ("+1e6", 1e6),
        ("-1e6", -1e6),
        ("+MAX", f32::MAX),
        ("-MAX", f32::MIN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("+NaN", f32::NAN),
        ("-NaN", -f32::NAN),
    ];
    for &(g, inv) in &[
        (2.19921875f32, true),
        (2.19921875, false),
        (1.8, true),
        (1.8, false),
        (2.6, true),
        (2.6, false),
        (3.0, false),
        (3.0, true),
        (1.0, false),
    ] {
        let e = [
            ev!(m0, g, inv),
            ev!(m1, g, inv),
            ev!(m2, g, inv),
            ev!(m3, g, inv),
        ];
        println!(
            "--- gamma curve {g} {} ---",
            if inv {
                "encode (x^1/g)"
            } else {
                "decode (x^g)"
            }
        );
        for &(name, x) in specials {
            let v: Vec<f32> = e.iter().map(|e| e(x)).collect();
            let exact = if x == 0.0 || x.is_nan() {
                f64::NAN
            } else {
                let p = if inv {
                    256.0 / ((g as f64 * 256.0).round())
                } else {
                    (g as f64 * 256.0).round() / 256.0
                };
                (x as f64).signum() * (x as f64).abs().powf(p)
            };
            println!(
                "{name:>10} before {:>14e} ({:08x}) c1 {:>14e} ({:08x}) c2 {:>14e} ({:08x}) c3 {:>14e} ({:08x}) exact {exact:>13.6e}",
                v[0],
                v[0].to_bits(),
                v[1],
                v[1].to_bits(),
                v[2],
                v[2].to_bits(),
                v[3],
                v[3].to_bits()
            );
        }
    }
    // The owner's reproduction through the public transform API.
    let (s, t) = (
        m0::ColorProfile::new_display_p3(),
        m0::ColorProfile::new_adobe_rgb(),
    );
    let o = m0::TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    let tr = s
        .create_transform_f32(m0::Layout::Rgb, &t, m0::Layout::Rgb, o)
        .unwrap();
    let src = [247.0f32 / 255.0, 1.0, 0.0];
    let mut d = [0f32; 3];
    tr.transform(&src, &mut d).unwrap();
    println!("before P3(247,255,0)->Adobe = {d:?}");
    let (s, t) = (
        m1::ColorProfile::new_display_p3(),
        m1::ColorProfile::new_adobe_rgb(),
    );
    let o = m1::TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    let tr = s
        .create_transform_f32(m1::Layout::Rgb, &t, m1::Layout::Rgb, o)
        .unwrap();
    tr.transform(&src, &mut d).unwrap();
    println!("c1     P3(247,255,0)->Adobe = {d:?}");
}
