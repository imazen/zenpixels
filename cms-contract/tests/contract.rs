//! One behavioral contract for every `PluggableCms` backend, then the backends against each other.

use cms_contract::LcmsPort;
use zenpixels::{Cicp, ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::policy::ConvertOptions;

#[path = "../../zenpixels-convert/tests/common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;

const SRGB: Source<'static> = Source::Cicp(Cicp::new(1, 13, 0, true));
const ADOBE: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/AdobeCompat-v2.icc");
const P3: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/DisplayP3Compat-v4.icc");
const BT2020: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/Rec2020Compat-v4.icc");

fn backends() -> [(&'static str, &'static dyn PluggableCms); 2] {
    [("moxcms", &MoxCms), ("lcms", &LcmsPort)]
}

fn pairs() -> Vec<(&'static str, Source<'static>, Source<'static>)> {
    vec![
        ("adobe>srgb", Source::Icc(ADOBE), SRGB),
        ("p3>srgb", Source::Icc(P3), SRGB),
        ("2020>srgb", Source::Icc(BT2020), SRGB),
        ("srgb>p3", SRGB, Source::Icc(P3)),
        ("srgb>2020", SRGB, Source::Icc(BT2020)),
        ("p3>2020", Source::Icc(P3), Source::Icc(BT2020)),
    ]
}

fn opts() -> ConvertOptions {
    ConvertOptions::permissive()
}

fn build(
    cms: &dyn PluggableCms,
    src: Source<'_>,
    dst: Source<'_>,
    from: F,
    to: F,
) -> Box<dyn RowTransformMut> {
    cms.build_source_transform(src, dst, from, to, &opts())
        .expect("declined")
        .expect("refused")
}

/// A row of `n` pixels from a 3D ramp, in `format`, with alpha counting up.
fn row(format: F, n: usize) -> Vec<u8> {
    let color = |i: usize, c: usize| ((i * [7, 13, 29][c] + c * 11) % 256) as f32 / 255.0;
    let alpha = |i: usize| (i % 256) as f32 / 255.0;
    let mut out = Vec::new();
    for i in 0..n {
        for c in 0..format.channels() {
            let v = if c < 3 { color(i, c) } else { alpha(i) };
            match format {
                F::Rgb8 | F::Rgba8 => out.push((v * 255.0).round() as u8),
                F::Rgb16 | F::Rgba16 => {
                    out.extend_from_slice(&((v * 65535.0).round() as u16).to_ne_bytes())
                }
                _ => out.extend_from_slice(&v.to_ne_bytes()),
            }
        }
    }
    out
}

fn run(t: &mut Box<dyn RowTransformMut>, src: &[u8], n: usize, to: F) -> Vec<u8> {
    let mut dst = vec![0u8; n * to.bytes_per_pixel()];
    t.try_transform_row(src, &mut dst, n as u32).expect("row");
    dst
}

/// Decodes to 0..=255 scale, for comparisons across depths.
fn codes(format: F, bytes: &[u8]) -> Vec<f64> {
    match format {
        F::Rgb8 | F::Rgba8 => bytes.iter().map(|&v| f64::from(v)).collect(),
        F::Rgb16 | F::Rgba16 => bytes
            .chunks_exact(2)
            .map(|b| f64::from(u16::from_ne_bytes([b[0], b[1]])) / 257.0)
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_ne_bytes(b.try_into().unwrap())) * 255.0)
            .collect(),
    }
}

const FORMATS: [F; 6] = [
    F::Rgb8,
    F::Rgba8,
    F::Rgb16,
    F::Rgba16,
    F::RgbF32,
    F::RgbaF32,
];

#[test]
fn garbage_icc_is_never_accepted() {
    for (name, cms) in backends() {
        let mut truncated = P3.to_vec();
        truncated.truncate(200);
        for bad in [&b"not a profile"[..], &truncated, &[]] {
            let built =
                cms.build_source_transform(Source::Icc(bad), SRGB, F::Rgb8, F::Rgb8, &opts());
            assert!(
                !matches!(built, Some(Ok(_))),
                "{name} accepted {} bytes",
                bad.len()
            );
        }
    }
}

#[test]
fn rows_split_anywhere() {
    for (name, cms) in backends() {
        for (pair, src, dst) in pairs() {
            for f in FORMATS {
                let mut t = build(cms, src.clone(), dst.clone(), f, f);
                let input = row(f, 600);
                let whole = run(&mut t, &input, 600, f);
                let bpp = f.bytes_per_pixel();
                let mut parts = Vec::new();
                for chunk in input.chunks(7 * bpp) {
                    parts.extend(run(&mut t, chunk, chunk.len() / bpp, f));
                }
                assert_eq!(whole, parts, "{name} {pair} {f:?}");
                // Rows carry no state: the same row again gives the same result.
                assert_eq!(
                    whole,
                    run(&mut t, &input, 600, f),
                    "{name} {pair} {f:?} repeat"
                );
            }
        }
    }
}

#[test]
fn only_the_row_is_touched() {
    for (name, cms) in backends() {
        for f in FORMATS {
            let mut t = build(cms, Source::Icc(P3), SRGB, f, f);
            let bpp = f.bytes_per_pixel();
            let input = row(f, 40);
            let expected = run(&mut t, &input[..32 * bpp], 32, f);
            let mut dst = vec![0xA5u8; 40 * bpp];
            t.try_transform_row(&input, &mut dst, 32).expect("row");
            assert_eq!(&dst[..32 * bpp], &expected[..], "{name} {f:?}");
            assert!(
                dst[32 * bpp..].iter().all(|&b| b == 0xA5),
                "{name} {f:?} wrote past the row"
            );
            let before = dst.clone();
            t.try_transform_row(&input, &mut dst, 0).expect("empty row");
            assert_eq!(before, dst, "{name} {f:?} wrote on an empty row");
        }
    }
}

#[test]
fn alpha_is_carried_exactly() {
    for (name, cms) in backends() {
        for (pair, src, dst) in pairs() {
            for f in [F::Rgba8, F::Rgba16, F::RgbaF32] {
                let sample = f.bytes_per_pixel() / 4;
                let mut t = build(cms, src.clone(), dst.clone(), f, f);
                let input = row(f, 300);
                let out = run(&mut t, &input, 300, f);
                let alpha = |row: &[u8]| -> Vec<u8> {
                    row.chunks_exact(4 * sample)
                        .flat_map(|p| p[3 * sample..].to_vec())
                        .collect()
                };
                assert_eq!(alpha(&input), alpha(&out), "{name} {pair} {f:?}");
            }
        }
    }
}

#[test]
fn white_and_black_hold() {
    for (name, cms) in backends() {
        for (pair, src, dst) in pairs() {
            for f in [F::Rgb8, F::Rgb16, F::RgbF32] {
                let mut t = build(cms, src.clone(), dst.clone(), f, f);
                for (level, target) in [(1.0f32, 255.0), (0.0, 0.0)] {
                    let input: Vec<u8> = match f {
                        F::Rgb8 => vec![(level * 255.0) as u8; 3],
                        F::Rgb16 => {
                            let v = (level * 65535.0) as u16;
                            [v, v, v].iter().flat_map(|v| v.to_ne_bytes()).collect()
                        }
                        _ => [level; 3].iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    };
                    for v in codes(f, &run(&mut t, &input, 1, f)) {
                        assert!(
                            (v - target).abs() < 0.51,
                            "{name} {pair} {f:?} {level}: {v}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn f32_cmyk_ink_polarity_and_limits() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    for (name, cms) in backends() {
        let mut plain = build(cms, Source::Icc(&icc), SRGB, F::CmykF32, F::RgbF32);
        let mut flipped = build(cms, Source::Icc(&icc), SRGB, F::CmykF32Inverted, F::RgbF32);
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        let show = |t: &mut Box<dyn RowTransformMut>, ink: [f32; 4]| {
            codes(F::RgbF32, &run(t, &f32s(&ink), 1, F::RgbF32))
        };
        for (ink, target) in [([0.0; 4], 255.0), ([0.0, 0.0, 0.0, 1.0], 0.0)] {
            let got = show(&mut plain, ink);
            eprintln!("{name} ink {ink:?}: {got:?}");
            for v in &got {
                assert!((v - target).abs() < 1.0, "{name} ink {ink:?}: {got:?}");
            }
            let inverted = ink.map(|v| 1.0 - v);
            let other = show(&mut flipped, inverted);
            for (a, b) in got.iter().zip(&other) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "{name} polarity: {got:?} vs {other:?}"
                );
            }
        }
        assert!(matches!(
            cms.build_source_transform(SRGB, Source::Icc(&icc), F::RgbF32, F::CmykF32, &opts()),
            Some(Err(_))
        ));
    }
}

/// Worst color difference between the backends, in 8-bit codes.
fn worst(from: F, a: &[u8], b: &[u8]) -> f64 {
    codes(from, a)
        .iter()
        .zip(codes(from, b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn backends_agree() {
    let [(_, mox), (_, lcms)] = backends();
    let mut failed = Vec::new();
    for (pair, src, dst) in pairs() {
        for f in FORMATS {
            let input = row(f, 4096);
            let a = run(
                &mut build(mox, src.clone(), dst.clone(), f, f),
                &input,
                4096,
                f,
            );
            let b = run(
                &mut build(lcms, src.clone(), dst.clone(), f, f),
                &input,
                4096,
                f,
            );
            let d = worst(f, &a, &b);
            let exact = codes(f, &a)
                .iter()
                .zip(codes(f, &b))
                .filter(|(x, y)| **x != *y)
                .count();
            eprintln!(
                "{pair:<11} {f:?}: worst {d:.3}, differing {:.2}%",
                100.0 * exact as f64 / (a.len() / f.bytes_per_pixel() * f.channels()) as f64
            );
            // U8 can land on either side of a rounding.
            let limit = if matches!(f, F::Rgb8 | F::Rgba8) {
                1.0
            } else {
                0.2
            };
            if d > limit {
                failed.push(format!("{pair} {f:?}: {d:.3}"));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn f32_cmyk_backends_agree() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let [(_, mox), (_, lcms)] = backends();
    let ink: Vec<u8> = (0..4096 * 4)
        .flat_map(|i: u32| (((i * 37 + i / 4 * 11) % 101) as f32 / 100.0).to_ne_bytes())
        .collect();
    let a = run(
        &mut build(mox, Source::Icc(&icc), SRGB, F::CmykF32, F::RgbF32),
        &ink,
        4096,
        F::RgbF32,
    );
    let b = run(
        &mut build(lcms, Source::Icc(&icc), SRGB, F::CmykF32, F::RgbF32),
        &ink,
        4096,
        F::RgbF32,
    );
    let d = worst(F::RgbF32, &a, &b);
    eprintln!("synthetic CMYK F32: worst {d:.3}");
    // A coarse 9-point table: the backends order their 4D interpolation differently.
    assert!(d < 2.5, "{d}");
}

#[test]
fn gray_round_trips_and_carries_alpha() {
    let icc = cms_contract::gray_gamma22_icc();
    let [(_, mox), (_, lcms)] = backends();
    let ramp: Vec<u8> = (0..=255).collect();
    let a = run(
        &mut build(mox, Source::Icc(&icc), SRGB, F::Gray8, F::Rgb8),
        &ramp,
        256,
        F::Rgb8,
    );
    let b = run(
        &mut build(lcms, Source::Icc(&icc), SRGB, F::Gray8, F::Rgb8),
        &ramp,
        256,
        F::Rgb8,
    );
    let d = worst(F::Rgb8, &a, &b);
    eprintln!("gray 2.2>srgb Gray8>Rgb8: worst {d:.3}");
    assert!(d <= 1.0, "{d}");
    assert_eq!(a.chunks_exact(3).map(|p| p[0]).next_back(), Some(255));
    assert_eq!(a.chunks_exact(3).map(|p| p[0]).next(), Some(0));

    let with_alpha: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v]).collect();
    for (name, cms) in backends() {
        let mut t = build(cms, Source::Icc(&icc), SRGB, F::GrayA8, F::Rgba8);
        let out = run(&mut t, &with_alpha, 256, F::Rgba8);
        for (i, p) in out.chunks_exact(4).enumerate() {
            assert_eq!(p[3], 255 - i as u8, "{name} alpha at {i}");
        }
    }
}

#[test]
fn eight_bit_cmyk_agrees() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let [(_, mox), (_, lcms)] = backends();
    let ink: Vec<u8> = (0..4096 * 4)
        .map(|i: u32| ((i * 37 + i / 4 * 11) % 256) as u8)
        .collect();
    let a = run(
        &mut build(mox, Source::Icc(&icc), SRGB, F::Cmyk8, F::Rgb8),
        &ink,
        4096,
        F::Rgb8,
    );
    let b = run(
        &mut build(lcms, Source::Icc(&icc), SRGB, F::Cmyk8, F::Rgb8),
        &ink,
        4096,
        F::Rgb8,
    );
    let d = worst(F::Rgb8, &a, &b);
    eprintln!("synthetic CMYK Cmyk8: worst {d:.3}");
    assert!(d <= 6.0, "{d}");
}

/// The trait's real caller, on the same conversion from descriptors alone.
#[test]
fn row_converter_drives_every_backend() {
    use zenpixels::{ColorPrimaries, PixelDescriptor as D};
    use zenpixels_convert::RowConverter;
    for (from, to) in [
        (
            D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
            D::RGB8_SRGB,
        ),
        (
            D::RGB8_SRGB,
            D::RGB8_SRGB.with_primaries(ColorPrimaries::Bt2020),
        ),
        (
            D::RGB16_SRGB.with_primaries(ColorPrimaries::DisplayP3),
            D::RGB16_SRGB,
        ),
    ] {
        let f = from.pixel_format();
        let mut results = Vec::new();
        for (name, cms) in backends() {
            let input = row(f, 512);
            let mut c = RowConverter::new_explicit_with_cms(from, to, &opts(), Some(cms))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut out = vec![0u8; 512 * f.bytes_per_pixel()];
            c.try_convert_row(&input, &mut out, 512).expect("row");
            results.push(out);
        }
        let d = worst(f, &results[0], &results[1]);
        eprintln!("RowConverter {f:?}: worst {d:.3}");
        assert!(d <= 1.0, "{f:?}: {d}");
    }
}

/// Every profile in a directory, against the sRGB target. The caller chooses to run this, by naming
/// the directory in `CMS_CONTRACT_PROFILES`; see `just cms-contract`.
#[test]
#[ignore = "needs CMS_CONTRACT_PROFILES"]
fn real_profiles_agree() {
    let dir = std::env::var("CMS_CONTRACT_PROFILES").expect("CMS_CONTRACT_PROFILES");
    let [(_, mox), (_, lcms)] = backends();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("profile directory")
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{dir} holds no profiles");
    for path in paths {
        let icc = std::fs::read(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let space = &icc[16..20];
        let cases: &[(F, F)] = match space {
            b"RGB " => &[
                (F::Rgb8, F::Rgb8),
                (F::Rgb16, F::Rgb16),
                (F::RgbF32, F::RgbF32),
            ],
            b"CMYK" => &[(F::Cmyk8, F::Rgb8), (F::CmykF32, F::RgbF32)],
            b"GRAY" => &[(F::Gray8, F::Rgb8)],
            other => {
                eprintln!("{name}: {} not covered", String::from_utf8_lossy(other));
                continue;
            }
        };
        for &(from, to) in cases {
            let input: Vec<u8> = match from {
                F::Rgb8 | F::Rgb16 | F::RgbF32 => row(from, 4096),
                F::CmykF32 => (0..4096 * 4)
                    .flat_map(|i: u32| (((i * 37 + i / 4 * 11) % 101) as f32 / 100.0).to_ne_bytes())
                    .collect(),
                F::Cmyk8 => (0..4096 * 4)
                    .map(|i: u32| ((i * 37 + i / 4 * 11) % 256) as u8)
                    .collect(),
                _ => (0..4096).map(|i: u32| (i % 256) as u8).collect(),
            };
            let source = Source::Icc(&icc);
            let result = (
                mox.build_source_transform(source.clone(), SRGB, from, to, &opts()),
                lcms.build_source_transform(source, SRGB, from, to, &opts()),
            );
            match result {
                (Some(Ok(mut a)), Some(Ok(mut b))) => {
                    let d = worst(
                        to,
                        &run(&mut a, &input, 4096, to),
                        &run(&mut b, &input, 4096, to),
                    );
                    eprintln!("{name}: {from:?}>{to:?} worst {d:.3}");
                }
                (a, b) => eprintln!(
                    "{name}: {from:?}>{to:?} moxcms {} lcms {}",
                    a.as_ref().map_or("declined".to_string(), |r| r
                        .as_ref()
                        .map_or_else(|e| e.to_string(), |_| "built".to_string())),
                    b.as_ref().map_or("declined".to_string(), |r| r
                        .as_ref()
                        .map_or_else(|e| e.to_string(), |_| "built".to_string())),
                ),
            }
        }
    }
}

#[test]
fn k_only_ink_is_black() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    let ink: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    for (name, cms) in backends() {
        let mut t = build(cms, Source::Icc(&icc), SRGB, F::CmykF32, F::RgbF32);
        let got = codes(F::RgbF32, &run(&mut t, &ink, 1, F::RgbF32));
        assert!(got.iter().all(|v| v.abs() < 1.0), "{name}: {got:?}");
    }
}

/// 8-bit CMYK against the same profile's F32 path and against lcms2 (relative colorimetric with
/// black-point compensation at both depths). `CMS_CONTRACT_PROFILES` adds a directory of profiles.
/// Prints mean and worst differences in 8-bit codes, and the share more than one code apart.
#[test]
#[ignore = "measurement: run with --nocapture"]
fn cmyk8_against_f32_and_lcms() {
    let [(_, mox), (_, lcms)] = backends();
    let n = 16384;
    let ink: Vec<u8> = (0..n as u32 * 4)
        .map(|i| ((i * 37 + i / 4 * 11 + i / 64 * 3) % 256) as u8)
        .collect();
    let ink_f: Vec<u8> = ink
        .iter()
        .flat_map(|&v| (f32::from(v) / 255.0).to_ne_bytes())
        .collect();
    let mut profiles = vec![("synthetic".to_string(), synthetic_icc::synthetic_cmyk_icc())];
    if let Ok(dir) = std::env::var("CMS_CONTRACT_PROFILES") {
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .expect("profile directory")
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        for path in paths {
            let icc = std::fs::read(&path).unwrap();
            // Device links are CMYK to CMYK; they don't convert to RGB.
            if icc.get(16..20) == Some(b"CMYK".as_slice())
                && icc.get(12..16) != Some(b"link".as_slice())
            {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                profiles.push((name, icc));
            }
        }
    }
    let p3 = Source::Cicp(Cicp::new(12, 13, 0, true));
    let stats = |a: &[f64], b: &[f64]| {
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
        let over = d.iter().filter(|v| **v > 1.0).count();
        (
            d.iter().sum::<f64>() / d.len() as f64,
            d.iter().cloned().fold(0.0, f64::max),
            over as f64 * 100.0 / d.len() as f64,
        )
    };
    println!(
        "profile|target|layout|8-vs-f32 mean/worst/%>1|8-vs-lcms8 mean/worst/%>1|f32-vs-lcmsf32 mean/worst"
    );
    for (name, icc) in &profiles {
        for (tname, target) in [("sRGB", SRGB), ("P3", p3.clone())] {
            for (lname, to8, tof) in [("RGB", F::Rgb8, F::RgbF32), ("RGBA", F::Rgba8, F::RgbaF32)] {
                let run_one = |cms: &dyn PluggableCms, from: F, to: F, input: &[u8]| match cms
                    .build_source_transform(Source::Icc(icc), target.clone(), from, to, &opts())
                {
                    Some(Ok(mut t)) => {
                        let mut dst = vec![0u8; n * to.bytes_per_pixel()];
                        t.try_transform_row(input, &mut dst, n as u32)
                            .ok()
                            .map(|()| codes(to, &dst))
                    }
                    _ => None,
                };
                let cells = (
                    run_one(mox, F::Cmyk8, to8, &ink),
                    run_one(mox, F::CmykF32, tof, &ink_f),
                    run_one(lcms, F::Cmyk8, to8, &ink),
                    run_one(lcms, F::CmykF32, tof, &ink_f),
                );
                let (Some(m8), Some(mf), Some(l8), Some(lf)) = cells else {
                    println!("{name}|{tname}|{lname}|a backend refused");
                    continue;
                };
                let rounded: Vec<f64> = mf.iter().map(|v| v.clamp(0.0, 255.0).round()).collect();
                let (a, b, c) = stats(&m8, &rounded);
                let (d, e, f) = stats(&m8, &l8);
                let (g, h, _) = stats(&mf, &lf);
                println!(
                    "{name}|{tname}|{lname}|{a:.3}/{b:.1}/{c:.2}|{d:.3}/{e:.1}/{f:.2}|{g:.3}/{h:.2}"
                );
            }
        }
    }
}
