#![forbid(unsafe_code)]
// CICP pure-power curves: master (m0) against the final fix (m4), dense over positive f32, both directions.
macro_rules! pair {
    ($m:ident, $tc:ident) => {{
        let with = |tc| {
            let mut p = $m::ColorProfile::new_srgb();
            p.cicp = Some($m::CicpProfile {
                color_primaries: $m::CicpColorPrimaries::Bt709,
                transfer_characteristics: tc,
                matrix_coefficients: $m::MatrixCoefficients::Bt709,
                full_range: true,
            });
            p
        };
        let (p, l) = (
            with($m::TransferCharacteristics::$tc),
            with($m::TransferCharacteristics::Linear),
        );
        let o = $m::TransformOptions {
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        };
        (
            p.create_transform_f32($m::Layout::Rgb, &l, $m::Layout::Rgb, o)
                .unwrap(),
            l.create_transform_f32($m::Layout::Rgb, &p, $m::Layout::Rgb, o)
                .unwrap(),
        )
    }};
}
macro_rules! check { ($name:literal, $tc:ident, $dec:expr, $enc:expr) => {{
    let (d0, e0) = pair!(m0, $tc);
    let (d4, e4) = pair!(m4, $tc);
    let inputs: Vec<f32> = (0..=f32::INFINITY.to_bits()).step_by(97).map(f32::from_bits).collect();
    let row: Vec<f32> = inputs.iter().flat_map(|&x| [x; 3]).collect();
    for (dir, t0, t4, exact) in [("decode", &d0, &d4, &$dec as &dyn Fn(f64) -> f64), ("encode", &e0, &e4, &$enc)] {
        let (mut a, mut b) = (vec![0f32; row.len()], vec![0f32; row.len()]);
        t0.transform(&row, &mut a).unwrap();
        t4.transform(&row, &mut b).unwrap();
        let (mut in_n, mut in_diff, mut out_n, mut out_old_bad, mut out_new_bad, mut worst_new, mut in_worst) = (0u64, 0u64, 0u64, 0u64, 0u64, 0f64, 0f64);
        for (i, &x) in inputs.iter().enumerate() {
            let (va, vb) = (a[i * 3], b[i * 3]);
            let e = exact(x as f64); let ef = e as f32;
            let rel = |v: f32| if v == ef { 0.0 } else if v.is_infinite() || ef.is_infinite() || !v.is_finite() { f64::INFINITY } else { (v as f64 - e).abs() / e.abs().max(1e-300) };
            // Identity is asked of inputs whose bounded_powf argument lands in the fast range.
            let fast = x.to_bits().wrapping_sub(0x2c00_0000) <= 0x2600_0000;
            if fast { in_n += 1; if va.to_bits() != vb.to_bits() { in_diff += 1; } if rel(vb) > in_worst { in_worst = rel(vb); } }
            else { out_n += 1; if rel(va) > 1e-3 && (va as f64 - e).abs() > 2e-45 { out_old_bad += 1; } if rel(vb) > 1e-3 && (vb as f64 - e).abs() > 2e-45 { out_new_bad += 1; if rel(vb) > worst_new { worst_new = rel(vb); } } }
        }
        println!("{} {dir}: inner range n={in_n} bits differ {in_diff} worst rel {in_worst:.2e}; outer n={out_n} wrong(>1e-3): master {out_old_bad}, fix {out_new_bad} (worst {worst_new:.2e})", $name);
    }
}}}
fn main() {
    check!("Bt470M 2.2", Bt470M, |x: f64| x.powf(2.2), |x: f64| x
        .powf(1. / 2.2));
    check!("Bt470Bg 2.8", Bt470Bg, |x: f64| x.powf(2.8), |x: f64| x
        .powf(1. / 2.8));
    const S: f64 = 0.91655527974030934;
    check!(
        "Smpte428 2.6",
        Smpte428,
        |x: f64| x.powf(2.6) / S,
        |x: f64| (x * S).powf(1. / 2.6)
    );
}
