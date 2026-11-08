//! Independent read-back of the shipped profiles; no generator/moxcms math.
//! Tables: ITU-T H.273 (07/2024), Tables 2/3, https://www.itu.int/rec/T-REC-H.273
//! ICC curve encodings: https://www.color.org/specification/ICC.1-2022-05.pdf
use crate::Cicp;
use crate::icc_profiles::{SynthesizedIcc, synthesize_gray_icc_for_cicp, synthesize_icc_for_cicp};

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
            if t == 18 {
                continue;
            }
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
                        (a - b).abs() < 0.00002,
                        "({p},{t}) colorants {actual:?}, expected {:?}",
                        expected_colorants(p)
                    );
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 149);
}

#[path = "transfer_reference.rs"]
mod transfer_reference;
use transfer_reference::{analytic, curve};

#[test]
fn every_rgb_and_gray_transfer_matches_analytic_h273() {
    let mut checked = [0, 0];
    for &p in PRIMARIES {
        for &t in TRANSFERS {
            if t == 18 {
                continue;
            }
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
                            let want = if t == 17 {
                                let peak = xyz(&icc, b"lumi")[1];
                                assert!((peak - 52.37).abs() < 0.00001);
                                analytic(t, x) * 48.0 / peak
                            } else {
                                analytic(t, x)
                            };
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
    assert_eq!(checked, [149, 164]);
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
            if t == 18 {
                continue;
            }
            if p == 1 && t == 13 {
                continue;
            }
            let SynthesizedIcc::Profile(icc) =
                synthesize_gray_icc_for_cicp(Cicp::new(p, t, 0, true))
            else {
                panic!("missing gray ({p},{t})");
            };
            let want = xy_xyz(white(p));
            for (got, w) in xyz(&icc, b"wtpt").into_iter().zip([0.9642, 1.0, 0.8249]) {
                assert!(
                    (got - w).abs() < 0.00002,
                    "gray ({p},{t}) white {got} != {w}"
                );
            }
            let chad = tag(&icc, b"chad");
            for i in 0..3 {
                let row = core::array::from_fn(|j| fixed(&chad[8 + 4 * (3 * i + j)..]));
                assert!(
                    (dot(row, want) - [0.9642, 1.0, 0.8249][i]).abs() < 0.00002,
                    "gray ({p},{t}) chad"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 164);
}

#[test]
fn hlg_synthesis_refuses_without_a_rendering_target_in_every_build() {
    for primaries in 0..=255 {
        let cicp = Cicp::new(primaries, 18, 0, true);
        assert_eq!(
            synthesize_icc_for_cicp(cicp),
            SynthesizedIcc::CmsUnsupported
        );
        assert_eq!(
            synthesize_gray_icc_for_cicp(cicp),
            SynthesizedIcc::CmsUnsupported
        );
    }
}

#[test]
fn rgb_media_white_and_full_adaptation_match_pcs() {
    for &p in PRIMARIES.iter().filter(|&&p| p != 10) {
        let SynthesizedIcc::Profile(icc) = synthesize_icc_for_cicp(Cicp::new(p, 8, 0, true)) else {
            panic!("linear RGB profile {p}");
        };
        for (actual, expected) in xyz(&icc, b"wtpt").into_iter().zip([0.9642, 1.0, 0.8249]) {
            assert!((actual - expected).abs() < 0.000008);
        }
        let chad = tag(&icc, b"chad");
        let native = xy_xyz(white(p));
        for i in 0..3 {
            let row = core::array::from_fn(|j| fixed(&chad[8 + 4 * (3 * i + j)..]));
            assert!((dot(row, native) - [0.9642, 1.0, 0.8249][i]).abs() < 0.00002);
        }
        assert_eq!(&icc[8..12], &[4, 0x40, 0, 0]);
    }
}

// Qualify the build-time recipe before replacing the stored export bytes.
// PQ's representation is evaluated separately by the backend precision harness.
#[cfg(feature = "cms-moxcms")]
#[test]
fn corrected_recipe_matches_independent_colorimetry_and_dense_sdr_curves() {
    for &p in PRIMARIES {
        for &t in TRANSFERS.iter().filter(|&&t| !matches!(t, 16 | 18)) {
            for (kind, synth) in [super::synthesis::rgb, super::synthesis::gray]
                .into_iter()
                .enumerate()
            {
                if kind == 0 && p == 10 {
                    continue;
                }
                let icc = synth(p, t).expect("supported explicit recipe");
                for (actual, expected) in xyz(&icc, b"wtpt").into_iter().zip([0.9642, 1.0, 0.8249])
                {
                    assert!(
                        (actual - expected).abs() < 0.000008,
                        "({p},{t}) kind {kind} wtpt"
                    );
                }
                if kind == 0 {
                    let actual = [b"rXYZ", b"gXYZ", b"bXYZ"].map(|sig| xyz(&icc, sig));
                    for (a, b) in actual
                        .into_iter()
                        .flatten()
                        .zip(expected_colorants(p).into_iter().flatten())
                    {
                        assert!((a - b).abs() < 0.00002, "({p},{t}) colorants");
                    }
                }
                let chad = tag(&icc, b"chad");
                for i in 0..3 {
                    let row = core::array::from_fn(|j| fixed(&chad[8 + 4 * (3 * i + j)..]));
                    assert!(
                        (dot(row, xy_xyz(white(p))) - [0.9642, 1.0, 0.8249][i]).abs() < 0.00002
                    );
                }
                // Four white groups share curves. One representative gamut per
                // transfer suffices for the dense scalar serialization check.
                if p != 1 {
                    continue;
                }
                let trc = tag(&icc, if kind == 0 { b"rTRC" } else { b"kTRC" });
                for code in 0..=65535 {
                    let x = f64::from(code) / 65535.0;
                    let actual = curve(trc, x);
                    if code == 0 && matches!(t, 9 | 10) {
                        let cutoff = if t == 9 { 0.01 } else { 10_f64.sqrt() / 1000.0 };
                        assert!((0.0..=cutoff).contains(&actual));
                        continue;
                    }
                    let expected = if t == 17 {
                        analytic(t, x) * 48.0 / xyz(&icc, b"lumi")[1]
                    } else {
                        analytic(t, x)
                    };
                    assert!(
                        (actual - expected).abs() <= 0.0005,
                        "TC{t} kind {kind} code {code}: {actual} versus {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn exported_pq_declares_absolute_scale_and_both_pipeline_directions() {
    for &p in PRIMARIES {
        for (gray, profile) in [
            (false, synthesize_icc_for_cicp(Cicp::new(p, 16, 0, true))),
            (
                true,
                synthesize_gray_icc_for_cicp(Cicp::new(p, 16, 0, true)),
            ),
        ] {
            if !gray && p == 10 {
                continue;
            }
            let SynthesizedIcc::Profile(icc) = profile else {
                panic!("PQ profile ({p},{gray})");
            };
            assert_eq!(xyz(&icc, b"lumi"), [0.0, 10000.0, 0.0]);
            for sig in [b"D2B0", b"D2B1", b"D2B2", b"B2D0", b"B2D1", b"B2D2"] {
                assert_eq!(&tag(&icc, sig)[..4], b"mpet");
            }
            for sig in [b"A2B0", b"A2B1", b"A2B2"] {
                assert_eq!(&tag(&icc, sig)[..4], b"mAB ");
            }
            for sig in [b"B2A0", b"B2A1", b"B2A2"] {
                assert_eq!(&tag(&icc, sig)[..4], b"mBA ");
            }
            let date: [u16; 6] = core::array::from_fn(|i| {
                u16::from_be_bytes(icc[24 + 2 * i..26 + 2 * i].try_into().unwrap())
            });
            assert!(date[0] >= 2000 && (1..=12).contains(&date[1]) && (1..=31).contains(&date[2]));
            if !gray {
                let pipe = tag(&icc, b"D2B1");
                let mut found = false;
                for i in 0..u32be(&pipe[12..]) {
                    let at = u32be(&pipe[16 + i * 8..]);
                    if &pipe[at..at + 4] != b"matf" {
                        continue;
                    }
                    let want = [b"rXYZ", b"gXYZ", b"bXYZ"].map(|sig| xyz(&icc, sig));
                    for (column, values) in want.iter().enumerate() {
                        for (row, &expected) in values.iter().enumerate() {
                            let at = at + 12 + (row * 3 + column) * 4;
                            let actual = f32::from_be_bytes(pipe[at..at + 4].try_into().unwrap());
                            assert_eq!(
                                actual, expected as f32,
                                "PQ matrix and declared primaries disagree"
                            );
                        }
                    }
                    found = true;
                }
                assert!(found, "missing float RGB-to-PCS matrix");
            }
        }
    }
}

#[cfg(feature = "cms-lcms")]
#[test]
fn exported_pq_dense_neutral_roundtrip_has_no_black_plateau() {
    use crate::{ColorProfileSource, ConvertOptions, PixelFormat, PluggableCms, cms_lcms::Lcms};
    for gray in [false, true] {
        let synth = if gray {
            synthesize_gray_icc_for_cicp
        } else {
            synthesize_icc_for_cicp
        };
        let SynthesizedIcc::Profile(pq) = synth(Cicp::new(9, 16, 0, true)) else {
            panic!("PQ ICC");
        };
        let SynthesizedIcc::Profile(linear) = synth(Cicp::new(9, 8, 0, true)) else {
            panic!("linear ICC");
        };
        let channels = if gray { 1 } else { 3 };
        let format = if gray {
            PixelFormat::GrayF32
        } else {
            PixelFormat::RgbF32
        };
        for reverse in [false, true] {
            let input: alloc::vec::Vec<f32> = (0..=65535)
                .flat_map(|code| {
                    let value = code as f32 / 65535.0;
                    let value = if reverse {
                        analytic(16, f64::from(value)) as f32
                    } else {
                        value
                    };
                    core::iter::repeat_n(value, channels)
                })
                .collect();
            let mut output = alloc::vec![0.0_f32; input.len()];
            let (src, dst) = if reverse {
                (&linear, &pq)
            } else {
                (&pq, &linear)
            };
            let mut transform = Lcms
                .build_source_transform(
                    ColorProfileSource::Icc(src),
                    ColorProfileSource::Icc(dst),
                    format,
                    format,
                    &ConvertOptions::permissive().with_clip_out_of_gamut(false),
                )
                .expect("qualified LCMS path")
                .expect("build transform");
            transform
                .try_transform_row(
                    bytemuck::cast_slice(&input),
                    bytemuck::cast_slice_mut(&mut output),
                    65536,
                )
                .expect("complete row");
            for (code, pixel) in output.chunks_exact(channels).enumerate() {
                let encoded = code as f32 / 65535.0;
                let expected = if reverse {
                    f64::from(encoded)
                } else {
                    analytic(16, f64::from(encoded))
                };
                for (c, &actual) in pixel.iter().enumerate() {
                    assert!(
                        actual.is_finite() && (f64::from(actual) - expected).abs() <= 0.0005,
                        "PQ {gray} reverse {reverse} code {code} actual {actual} expected {expected}"
                    );
                    if code == 0 {
                        assert_eq!(actual, 0.0);
                    } else {
                        assert!(actual > 0.0, "PQ positive tone became black at code {code}");
                        assert!(
                            actual >= output[(code - 1) * channels + c],
                            "nonmonotonic PQ tone"
                        );
                    }
                }
            }
        }
    }
}
