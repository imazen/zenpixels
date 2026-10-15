//! Independent read-back of the shipped profiles; no generator/moxcms math.
//! Tables: ITU-T H.273 (07/2024), Tables 2/3, https://www.itu.int/rec/T-REC-H.273
//! ICC curve encodings: https://www.color.org/specification/ICC.1-2022-05.pdf
use zenpixels_convert::Cicp;
use zenpixels_convert::icc_profiles::{
    SynthesizedIcc, synthesize_gray_icc_for_cicp, synthesize_icc_for_cicp,
};

const PRIMARIES: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];
const TRANSFERS: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

fn u32be(b: &[u8]) -> usize {
    u32::from_be_bytes(b[..4].try_into().unwrap()) as usize
}
fn fixed(b: &[u8]) -> f64 {
    f64::from(i32::from_be_bytes(b[..4].try_into().unwrap())) / 65536.0
}
fn tag<'a>(icc: &'a [u8], sig: &[u8; 4]) -> &'a [u8] {
    for i in 0..u32be(&icc[128..]) {
        let e = &icc[132 + 12 * i..144 + 12 * i];
        if &e[..4] == sig {
            return &icc[u32be(&e[4..])..][..u32be(&e[8..])];
        }
    }
    panic!("missing tag {sig:?}")
}
fn xyz(icc: &[u8], sig: &[u8; 4]) -> [f64; 3] {
    let t = tag(icc, sig);
    assert_eq!(&t[..4], b"XYZ ");
    core::array::from_fn(|i| fixed(&t[8 + 4 * i..]))
}
fn xy_xyz([x, y]: [f64; 2]) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}
fn white(p: u8) -> [f64; 2] {
    match p {
        4 | 8 => [0.310, 0.316],
        10 => [1.0 / 3.0, 1.0 / 3.0],
        11 => [0.314, 0.351],
        _ => [0.3127, 0.3290],
    }
}
fn chromaticities(p: u8) -> [[f64; 2]; 3] {
    match p {
        1 => [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
        4 => [[0.670, 0.330], [0.210, 0.710], [0.140, 0.080]],
        5 => [[0.640, 0.330], [0.290, 0.600], [0.150, 0.060]],
        6 | 7 => [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]],
        8 => [[0.681, 0.319], [0.243, 0.692], [0.145, 0.049]],
        9 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        11 | 12 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        22 => [[0.630, 0.340], [0.295, 0.605], [0.155, 0.077]],
        _ => panic!("not an RGB chromaticity row: {p}"),
    }
}
// Gaussian elimination with pivoting, independently of the generator's
// determinant/Cramer construction. Input is row-major.
fn solve(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for c in 0..3 {
        let pivot = (c..3)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, pivot);
        b.swap(c, pivot);
        let divisor = a[c][c];
        assert!(divisor.abs() > 1e-10);
        for value in &mut a[c][c..] {
            *value /= divisor;
        }
        b[c] /= divisor;
        let pivot_row = a[c];
        for i in 0..3 {
            if i != c {
                let scale = a[i][c];
                for (value, pivot) in a[i][c..].iter_mut().zip(&pivot_row[c..]) {
                    *value -= scale * pivot;
                }
                b[i] -= scale * b[c];
            }
        }
    }
    b
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn adapt(v: [f64; 3], p: u8) -> [f64; 3] {
    let bradford = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let src = xy_xyz(white(p));
    // ICC PCS D50, not the rounded daylight xy approximation.
    let dst = [0.9642, 1.0, 0.8249];
    let cone = core::array::from_fn(|i| {
        dot(bradford[i], v) * dot(bradford[i], dst) / dot(bradford[i], src)
    });
    solve(bradford, cone)
}
fn expected_colorants(p: u8) -> [[f64; 3]; 3] {
    if p == 10 {
        return core::array::from_fn(|i| adapt(core::array::from_fn(|j| f64::from(i == j)), p));
    }
    let cols = chromaticities(p).map(xy_xyz);
    let rows = core::array::from_fn(|i| core::array::from_fn(|j| cols[j][i]));
    let scale = solve(rows, xy_xyz(white(p)));
    core::array::from_fn(|i| adapt(cols[i].map(|v| v * scale[i]), p))
}
fn determinant([a, b, c]: [[f64; 3]; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
}

#[test]
fn every_rgb_matrix_has_rank_and_h273_d50_colorants() {
    let mut checked = 0;
    for &p in PRIMARIES.iter().rev() {
        for &t in TRANSFERS {
            let result = synthesize_icc_for_cicp(Cicp::new(p, t, 0, true));
            let SynthesizedIcc::Profile(icc) = result else {
                assert!(
                    (p == 1 && t == 13) || (p == 10 && result == SynthesizedIcc::CmsUnsupported),
                    "unexpected outcome ({p},{t}): {result:?}"
                );
                continue;
            };
            let actual = [b"rXYZ", b"gXYZ", b"bXYZ"].map(|sig| xyz(&icc, sig));
            assert!(
                determinant(actual).abs() > 0.01,
                "({p},{t}) singular colorants: {actual:?}"
            );
            for (col, want) in actual.into_iter().zip(expected_colorants(p)) {
                for (a, b) in col.into_iter().zip(want) {
                    assert!(
                        (a - b).abs() < 0.001,
                        "({p},{t}) colorants {actual:?}, expected {:?}",
                        expected_colorants(p)
                    );
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 159);
}

fn curve(trc: &[u8], x: f64) -> f64 {
    match &trc[..4] {
        b"para" => {
            let kind = u16::from_be_bytes(trc[8..10].try_into().unwrap());
            let v: Vec<_> = (0..(trc.len() - 12) / 4)
                .map(|i| fixed(&trc[12 + 4 * i..]))
                .collect();
            match kind {
                0 => x.powf(v[0]),
                1 => {
                    if x >= -v[2] / v[1] {
                        (v[1] * x + v[2]).powf(v[0])
                    } else {
                        0.0
                    }
                }
                2 => {
                    if x >= -v[2] / v[1] {
                        (v[1] * x + v[2]).powf(v[0]) + v[3]
                    } else {
                        v[3]
                    }
                }
                3 => {
                    if x >= v[4] {
                        (v[1] * x + v[2]).powf(v[0])
                    } else {
                        v[3] * x
                    }
                }
                4 => {
                    if x >= v[4] {
                        (v[1] * x + v[2]).powf(v[0]) + v[5]
                    } else {
                        v[3] * x + v[6]
                    }
                }
                _ => panic!("unknown para function {kind}"),
            }
        }
        b"curv" => {
            let n = u32be(&trc[8..]);
            let sample = |i: usize| {
                f64::from(u16::from_be_bytes(
                    trc[12 + 2 * i..14 + 2 * i].try_into().unwrap(),
                ))
            };
            match n {
                0 => x,
                1 => x.powf(sample(0) / 256.0),
                _ => {
                    let position = x * (n - 1) as f64;
                    let i = (position.floor() as usize).min(n - 2);
                    (sample(i) + (sample(i + 1) - sample(i)) * (position - i as f64)) / 65535.0
                }
            }
        }
        other => panic!("unsupported TRC {other:?}"),
    }
}
fn inverse_oetf(x: f64, slope: f64, alpha: f64, beta: f64, power: f64) -> f64 {
    if x < slope * beta {
        x / slope
    } else {
        ((x + alpha - 1.0) / alpha).powf(power)
    }
}
fn analytic(t: u8, x: f64) -> f64 {
    let bt709 = || inverse_oetf(x, 4.5, 1.099296826809442, 0.018053968510807, 1.0 / 0.45);
    match t {
        1 | 6 | 11 | 12 | 14 | 15 => bt709(), // 11/12 agree on [0,1].
        4 => x.powf(2.2),
        5 => x.powf(2.8),
        7 => inverse_oetf(x, 4.0, 1.111572195921731, 0.022821585529445, 1.0 / 0.45),
        8 => x,
        9 => 10_f64.powf(2.0 * (x - 1.0)),
        10 => 10_f64.powf(2.5 * (x - 1.0)),
        13 => inverse_oetf(x, 12.92, 1.0550107189475866, 0.003_041_282_560_127_521, 2.4),
        16 => {
            let e = x.powf(32.0 / 2523.0);
            ((e - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * e))
                .powf(8192.0 / 1305.0)
        }
        // A u16 curv LUT saturates at 1; ST428's nominal endpoint is 52.37/48.
        17 => (x.powf(2.6) * 52.37 / 48.0).min(1.0),
        18 => {
            let scene = if x <= 0.5 {
                x * x / 3.0
            } else {
                (((x - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.0
            };
            // The shipped HLG ICC encodes BT.2100 reference-display neutral
            // EOTF (1000 cd/m², system gamma 1.2), including its OOTF.
            scene.powf(1.2)
        }
        _ => panic!("unassigned transfer {t}"),
    }
}

#[test]
fn every_rgb_and_gray_transfer_matches_analytic_h273() {
    let mut checked = [0, 0];
    for &p in PRIMARIES {
        for &t in TRANSFERS {
            for (kind, synth) in [synthesize_icc_for_cicp, synthesize_gray_icc_for_cicp]
                .into_iter()
                .enumerate()
            {
                let result = synth(Cicp::new(p, t, 0, true));
                if p == 1 && t == 13 {
                    assert_eq!(result, SynthesizedIcc::NotNeeded);
                    continue;
                }
                if kind == 0 && p == 10 {
                    assert_eq!(result, SynthesizedIcc::CmsUnsupported);
                    continue;
                }
                let SynthesizedIcc::Profile(icc) = result else {
                    panic!("missing profile ({p},{t}), kind {kind}: {result:?}");
                };
                let signatures: &[[u8; 4]] = if kind == 0 {
                    &[*b"rTRC", *b"gTRC", *b"bTRC"]
                } else {
                    &[*b"kTRC"]
                };
                for sig in signatures {
                    // Absolute linear-light tolerance: SDR 0.0005 (u8Fixed8
                    // gamma, s15Fixed16 coefficients and LUT quantization),
                    // PQ/HLG 0.001. This is NOT a relative deep-toe guarantee:
                    // PQ's u16 LUT cannot resolve tiny absolute luminances.
                    let tolerance = if matches!(t, 16 | 18) { 0.001 } else { 0.0005 };
                    for x in [
                        0.0, 0.001, 0.01, 0.04, 0.08, 0.10, 0.18, 0.25, 0.5, 0.75, 0.9, 1.0,
                    ] {
                        let actual = curve(tag(&icc, sig), x);
                        if matches!(t, 9 | 10) && x == 0.0 {
                            // H.273 maps [0, cutoff] to encoded zero. Its
                            // inverse here is an interval, not a single black
                            // value. Do not widen the pointwise tolerance.
                            let cutoff = if t == 9 { 0.01 } else { 10_f64.sqrt() / 1000.0 };
                            assert!(
                                (0.0..=cutoff).contains(&actual),
                                "({p},{t}) kind {kind} {sig:?} zero inverse {actual} outside [0,{cutoff}]"
                            );
                        } else {
                            let want = analytic(t, x);
                            assert!(
                                (actual - want).abs() <= tolerance,
                                "({p},{t}) kind {kind} {sig:?} x={x}: {actual} vs {want} (tolerance {tolerance})"
                            );
                        }
                    }
                }
                checked[kind] += 1;
            }
        }
    }
    assert_eq!(checked, [159, 175]);
}

#[test]
fn bt709_is_profiled_and_only_srgb_or_unspecified_defaults_are_omitted() {
    for synth in [synthesize_icc_for_cicp, synthesize_gray_icc_for_cicp] {
        let SynthesizedIcc::Profile(icc) = synth(Cicp::new(1, 1, 0, true)) else {
            panic!("BT.709 (1,1) needs its own curve");
        };
        let sig = if &icc[16..20] == b"GRAY" {
            b"kTRC"
        } else {
            b"rTRC"
        };
        assert!((curve(tag(&icc, sig), 0.5) - analytic(1, 0.5)).abs() < 0.0005);
        assert!((curve(tag(&icc, sig), 0.5) - analytic(13, 0.5)).abs() > 0.04);
        for (p, t) in [(1, 13), (2, 13), (1, 2), (2, 2), (2, 1)] {
            assert_eq!(synth(Cicp::new(p, t, 0, true)), SynthesizedIcc::NotNeeded);
        }
    }
}

#[test]
fn xyz_primaries_are_refused_for_rgb_in_every_build() {
    for t in 0..=255 {
        assert_eq!(
            synthesize_icc_for_cicp(Cicp::new(10, t, 0, true)),
            SynthesizedIcc::CmsUnsupported,
            "transfer {t}"
        );
    }
}

#[test]
fn gray_white_and_bradford_adaptation_are_independent_of_rgb_gamut() {
    let mut checked = 0;
    for &p in PRIMARIES {
        for &t in TRANSFERS {
            if p == 1 && t == 13 {
                continue;
            }
            let SynthesizedIcc::Profile(icc) =
                synthesize_gray_icc_for_cicp(Cicp::new(p, t, 0, true))
            else {
                panic!("missing gray ({p},{t})");
            };
            let want = xy_xyz(white(p));
            for (got, w) in xyz(&icc, b"wtpt").into_iter().zip(want) {
                assert!(
                    (got - w).abs() < 0.00002,
                    "gray ({p},{t}) white {got} != {w}"
                );
            }
            let chad = tag(&icc, b"chad");
            for i in 0..3 {
                let row = core::array::from_fn(|j| fixed(&chad[8 + 4 * (3 * i + j)..]));
                assert!(
                    // The gray recipe targets a 5003 K D50 approximation;
                    // compare to ICC D50 with the colorant tolerance, while
                    // preserving the tight source-white check above.
                    (dot(row, want) - [0.9642, 1.0, 0.8249][i]).abs() < 0.001,
                    "gray ({p},{t}) chad"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 175);
}
