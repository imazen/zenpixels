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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f64::from(u16::from_ne_bytes([b[0], b[1]])) / 257.0)
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_ne_bytes(*b)) * 255.0)
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
fn relative_intent_maps_paper_white_to_destination_white() {
    let bytes = synthetic_icc::synthetic_cmyk_icc();
    let mut profile = lcms::profile::open_profile_from_mem(None, &bytes).unwrap();
    // A printing medium whose measured white is darker than PCS white.
    // Relative intent maps its relative A2B white to destination white;
    // absolute intent retains the darker medium white.
    assert!(lcms::profile::cms_write_tag(
        &mut profile,
        lcms::signatures::tag::SigMediaWhitePointTag,
        Some(lcms::tagtypes::TagObject::Xyz(lcms::types::CieXyz {
            X: 0.9642 * 0.8,
            Y: 0.8,
            Z: 0.8249 * 0.8,
        })),
    ));
    let bytes = lcms::profile::save_profile_to_mem(&mut profile).unwrap();
    let options = ConvertOptions::permissive().with_clip_out_of_gamut(false);
    let mut transform = LcmsPort
        .build_source_transform(Source::Icc(&bytes), SRGB, F::CMYKF32, F::RgbF32, &options)
        .unwrap()
        .unwrap();
    let input = [0.0f32; 4];
    let mut output = [0.0f32; 3];
    transform
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut output),
            1,
        )
        .unwrap();
    for value in output {
        assert!(
            (value - 1.0).abs() < 0.001,
            "relative paper white: {output:?}"
        );
    }
}

#[test]
fn f32_cmyk_ink_polarity_and_limits() {
    let icc = synthetic_icc::synthetic_cmyk_icc();
    for (name, cms) in backends() {
        let mut plain = build(cms, Source::Icc(&icc), SRGB, F::CMYKF32, F::RgbF32);
        let mut flipped = build(cms, Source::Icc(&icc), SRGB, F::CMYKF32_INVERTED, F::RgbF32);
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
            cms.build_source_transform(SRGB, Source::Icc(&icc), F::RgbF32, F::CMYKF32, &opts()),
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
        &mut build(mox, Source::Icc(&icc), SRGB, F::CMYKF32, F::RgbF32),
        &ink,
        4096,
        F::RgbF32,
    );
    let b = run(
        &mut build(lcms, Source::Icc(&icc), SRGB, F::CMYKF32, F::RgbF32),
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
    assert_eq!(
        a.as_chunks::<3>().0.iter().map(|p| p[0]).next_back(),
        Some(255)
    );
    assert_eq!(a.as_chunks::<3>().0.iter().map(|p| p[0]).next(), Some(0));

    let with_alpha: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v]).collect();
    for (name, cms) in backends() {
        let mut t = build(cms, Source::Icc(&icc), SRGB, F::GrayA8, F::Rgba8);
        let out = run(&mut t, &with_alpha, 256, F::Rgba8);
        for (i, p) in out.as_chunks::<4>().0.iter().enumerate() {
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
        &mut build(mox, Source::Icc(&icc), SRGB, F::CMYK8, F::Rgb8),
        &ink,
        4096,
        F::Rgb8,
    );
    let b = run(
        &mut build(lcms, Source::Icc(&icc), SRGB, F::CMYK8, F::Rgb8),
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
            b"CMYK" => &[(F::CMYK8, F::Rgb8), (F::CMYKF32, F::RgbF32)],
            b"GRAY" => &[(F::Gray8, F::Rgb8)],
            other => {
                eprintln!("{name}: {} not covered", String::from_utf8_lossy(other));
                continue;
            }
        };
        for &(from, to) in cases {
            let input: Vec<u8> = match from {
                F::Rgb8 | F::Rgb16 | F::RgbF32 => row(from, 4096),
                F::CMYKF32 => (0..4096 * 4)
                    .flat_map(|i: u32| (((i * 37 + i / 4 * 11) % 101) as f32 / 100.0).to_ne_bytes())
                    .collect(),
                F::CMYK8 => (0..4096 * 4)
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
        let mut t = build(cms, Source::Icc(&icc), SRGB, F::CMYKF32, F::RgbF32);
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
                    run_one(mox, F::CMYK8, to8, &ink),
                    run_one(mox, F::CMYKF32, tof, &ink_f),
                    run_one(lcms, F::CMYK8, to8, &ink),
                    run_one(lcms, F::CMYKF32, tof, &ink_f),
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

fn float_pipeline_profile(signatures: &[[u8; 4]]) -> Vec<u8> {
    use lcms::{lut, profile, tagtypes::TagObject};
    let mut altered = profile::open_profile_from_mem(None, P3).unwrap();
    let mut pipeline = lut::pipeline_alloc(None, 3, 3).unwrap();
    assert!(
        pipeline.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_matrix(
                None,
                3,
                3,
                &[0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25],
                None
            )
            .unwrap()
        )
    );
    for &signature in signatures {
        assert!(profile::cms_write_tag(
            &mut altered,
            u32::from_be_bytes(signature),
            Some(TagObject::Mpe(pipeline.clone()))
        ));
    }
    profile::save_profile_to_mem(&mut altered).unwrap()
}

/// A float pipeline for the requested intent takes precedence over matrix/TRC.
#[test]
fn float_pipeline_is_honored_or_explicitly_declined() {
    for signatures in [[*b"D2B0", *b"D2B1"], [*b"B2D0", *b"B2D1"]] {
        let altered = float_pipeline_profile(&signatures);
        let (source, target) = if signatures[0] == *b"D2B0" {
            (Source::Icc(&altered), SRGB)
        } else {
            (SRGB, Source::Icc(&altered))
        };
        let mut transform = build(
            &LcmsPort,
            source.clone(),
            target.clone(),
            F::RgbF32,
            F::RgbF32,
        );
        let input = [0.5f32; 3];
        let mut output = [0.0f32; 3];
        transform
            .try_transform_row(
                bytemuck::cast_slice(&input),
                bytemuck::cast_slice_mut(&mut output),
                1,
            )
            .unwrap();
        assert!(
            output.iter().any(|&value| (value - 0.5).abs() > 0.05),
            "float pipeline was ignored: {output:?}"
        );
        assert!(
            MoxCms
                .build_source_transform(source, target, F::RgbF32, F::RgbF32, &opts())
                .is_none(),
            "MoxCMS must decline unsupported float pipelines"
        );
    }
}

/// A float transform for another intent cannot disable relative colorimetry.
#[test]
fn relative_intent_ignores_float_tags_for_other_intents() {
    for prefix in [*b"D2B", *b"B2D"] {
        for intent in *b"023" {
            let altered = float_pipeline_profile(&[[prefix[0], prefix[1], prefix[2], intent]]);
            let (source, target, plain_source, plain_target) = if prefix == *b"D2B" {
                (Source::Icc(&altered), SRGB, Source::Icc(P3), SRGB)
            } else {
                (SRGB, Source::Icc(&altered), SRGB, Source::Icc(P3))
            };
            let input = [0.18f32, 0.5, 0.75];
            for (name, cms) in backends() {
                let mut with_tag = build(cms, source.clone(), target.clone(), F::RgbF32, F::RgbF32);
                let mut plain = build(
                    cms,
                    plain_source.clone(),
                    plain_target.clone(),
                    F::RgbF32,
                    F::RgbF32,
                );
                let mut actual = [0.0f32; 3];
                let mut expected = [0.0f32; 3];
                with_tag
                    .try_transform_row(
                        bytemuck::cast_slice(&input),
                        bytemuck::cast_slice_mut(&mut actual),
                        1,
                    )
                    .unwrap();
                plain
                    .try_transform_row(
                        bytemuck::cast_slice(&input),
                        bytemuck::cast_slice_mut(&mut expected),
                        1,
                    )
                    .unwrap();
                assert_eq!(
                    actual.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{name} unrelated intent{intent}"
                );
            }
        }
    }
}

#[test]
fn inverted_integer_cmyk_matches_negated_standard_cmyk_bit_exact() {
    let cmyk_icc = synthetic_icc::synthetic_cmyk_icc();
    let n = 64usize;

    // 8-bit inverted CMYK across both MoxCms and LcmsPort.
    let std8: Vec<u8> = (0..n * 4)
        .map(|i| ((i * 37 + (i / 4) * 13) % 256) as u8)
        .collect();
    let inv8: Vec<u8> = std8.iter().map(|&b| !b).collect();

    for (name, cms) in backends() {
        for to in [F::Rgb8, F::Rgba8] {
            let mut t_std = build(cms, Source::Icc(&cmyk_icc), SRGB, F::CMYK8, to);
            let mut t_inv = build(cms, Source::Icc(&cmyk_icc), SRGB, F::CMYK8_INVERTED, to);
            let out_std = run(&mut t_std, &std8, n, to);
            let out_inv = run(&mut t_inv, &inv8, n, to);
            assert_eq!(out_inv, out_std, "{name} Cmyk8Inverted -> {to:?}");
        }

        // Inverted CMYK is never accepted as an output format without an explicit inversion kernel.
        for (src_fmt, bad_dst) in [
            (F::Rgb8, F::CMYK8_INVERTED),
            (F::Rgb16, F::CMYK16_INVERTED),
            (F::RgbF32, F::CMYKF32_INVERTED),
        ] {
            let built =
                cms.build_source_transform(SRGB, Source::Icc(&cmyk_icc), src_fmt, bad_dst, &opts());
            assert!(
                !matches!(built, Some(Ok(_))),
                "{name} unexpectedly accepted destination {bad_dst:?}"
            );
        }
    }

    // 16-bit inverted CMYK in LcmsPort (MoxCms declines 16-bit CMYK).
    let std16_words: Vec<u16> = (0..n * 4)
        .map(|i| ((i * 2573 + (i / 4) * 977) & 0xffff) as u16)
        .collect();
    let std16: Vec<u8> = std16_words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let inv16: Vec<u8> = std16_words
        .iter()
        .flat_map(|w| (!*w).to_ne_bytes())
        .collect();
    for to in [F::Rgb16, F::Rgba16] {
        let mut t_std = build(&LcmsPort, Source::Icc(&cmyk_icc), SRGB, F::CMYK16, to);
        let mut t_inv = build(
            &LcmsPort,
            Source::Icc(&cmyk_icc),
            SRGB,
            F::CMYK16_INVERTED,
            to,
        );
        let out_std = run(&mut t_std, &std16, n, to);
        let out_inv = run(&mut t_inv, &inv16, n, to);
        assert_eq!(out_inv, out_std, "lcms Cmyk16Inverted -> {to:?}");
    }
    assert!(
        MoxCms
            .build_source_transform(
                Source::Icc(&cmyk_icc),
                SRGB,
                F::CMYK16_INVERTED,
                F::Rgb16,
                &opts()
            )
            .is_none()
    );
}

#[test]
fn parametric_color_source_and_destination_agree_across_backends() {
    use zenpixels::{
        CieXy, GammaCurve, ParametricColor, ParametricPrimaries, ParametricTransfer,
        RenderingIntent, WhitePoint,
    };

    let custom_rgb = ParametricColor::rgb(
        ParametricPrimaries::Custom {
            red: CieXy::new(0.68, 0.32),
            green: CieXy::new(0.265, 0.69),
            blue: CieXy::new(0.15, 0.06),
        },
        WhitePoint::D65,
        ParametricTransfer::Gamma(GammaCurve::from_eotf_exponent(2.4)),
    );
    let gray_22 = ParametricColor::gray(
        WhitePoint::D65,
        ParametricTransfer::Gamma(GammaCurve::GAMMA_22),
    );

    let input_rgb = row(F::Rgb8, 32);
    for param in [
        ParametricColor::DCI_P3,
        ParametricColor::ADOBE_RGB,
        custom_rgb,
        custom_rgb.with_rendering_intent(RenderingIntent::Perceptual),
    ] {
        let mut mox_to_srgb = build(&MoxCms, Source::Parametric(param), SRGB, F::Rgb8, F::Rgb8);
        let mut lcms_to_srgb = build(&LcmsPort, Source::Parametric(param), SRGB, F::Rgb8, F::Rgb8);
        let a = codes(F::Rgb8, &run(&mut mox_to_srgb, &input_rgb, 32, F::Rgb8));
        let b = codes(F::Rgb8, &run(&mut lcms_to_srgb, &input_rgb, 32, F::Rgb8));
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() <= 1.0,
                "mox vs lcms parametric->sRGB ({param:?}): {x} vs {y}"
            );
        }

        let mut mox_from_srgb = build(&MoxCms, SRGB, Source::Parametric(param), F::Rgb8, F::Rgb8);
        let mut lcms_from_srgb =
            build(&LcmsPort, SRGB, Source::Parametric(param), F::Rgb8, F::Rgb8);
        let c = codes(F::Rgb8, &run(&mut mox_from_srgb, &input_rgb, 32, F::Rgb8));
        let d = codes(F::Rgb8, &run(&mut lcms_from_srgb, &input_rgb, 32, F::Rgb8));
        for (x, y) in c.iter().zip(&d) {
            assert!(
                (x - y).abs() <= 1.0,
                "mox vs lcms sRGB->parametric ({param:?}): {x} vs {y}"
            );
        }
    }

    // Grayscale parametric source to sRGB across both backends.
    let gray_in: Vec<u8> = (0..16).map(|i| (i * 17) as u8).collect();
    let mut mox_g = build(
        &MoxCms,
        Source::Parametric(gray_22),
        SRGB,
        F::Gray8,
        F::Rgb8,
    );
    let mut lcms_g = build(
        &LcmsPort,
        Source::Parametric(gray_22),
        SRGB,
        F::Gray8,
        F::Rgb8,
    );
    let g_mox = codes(F::Rgb8, &run(&mut mox_g, &gray_in, 16, F::Rgb8));
    let g_lcms = codes(F::Rgb8, &run(&mut lcms_g, &gray_in, 16, F::Rgb8));
    for (x, y) in g_mox.iter().zip(&g_lcms) {
        assert!((x - y).abs() <= 1.0, "gray parametric mismatch: {x} vs {y}");
    }
}

#[test]
fn cmyka_five_channel_preserves_alpha_and_inverts_only_ink() {
    use zenpixels::{ChannelType, InkPolarity};

    let cmyk_icc = synthetic_icc::synthetic_cmyk_icc();
    let cmyka8 = F::cmyka(ChannelType::U8, InkPolarity::ZeroIsNoInk);
    let cmyka8_inv = F::cmyka(ChannelType::U8, InkPolarity::ZeroIsFullInk);
    let cmyka16 = F::cmyka(ChannelType::U16, InkPolarity::ZeroIsNoInk);
    let cmyka16_inv = F::cmyka(ChannelType::U16, InkPolarity::ZeroIsFullInk);
    let cmykaf32 = F::cmyka(ChannelType::F32, InkPolarity::ZeroIsNoInk);
    let cmykaf32_inv = F::cmyka(ChannelType::F32, InkPolarity::ZeroIsFullInk);

    // MoxCms declines 5-channel CmykAlpha cleanly so a plugin chain can hand off to Lcms.
    for fmt in [cmyka8, cmyka8_inv, cmyka16, cmykaf32] {
        assert!(
            MoxCms
                .build_source_transform(Source::Icc(&cmyk_icc), SRGB, fmt, F::Rgba8, &opts(),)
                .is_none()
        );
    }

    // U8 CMYKA: alpha must pass through untouched even when ink polarity is inverted.
    let cmyk4_u8 = [0u8, 0, 0, 0, 30, 80, 120, 40];
    let cmyka5_std_u8 = [0u8, 0, 0, 0, 37, 30, 80, 120, 40, 201];
    let cmyka5_inv_u8 = [255u8, 255, 255, 255, 37, !30, !80, !120, !40, 201];

    let mut t4 = build(&LcmsPort, Source::Icc(&cmyk_icc), SRGB, F::CMYK8, F::Rgb8);
    let mut t5_std = build(&LcmsPort, Source::Icc(&cmyk_icc), SRGB, cmyka8, F::Rgba8);
    let mut t5_inv = build(
        &LcmsPort,
        Source::Icc(&cmyk_icc),
        SRGB,
        cmyka8_inv,
        F::Rgba8,
    );
    let out4 = run(&mut t4, &cmyk4_u8, 2, F::Rgb8);
    let out5_std = run(&mut t5_std, &cmyka5_std_u8, 2, F::Rgba8);
    let out5_inv = run(&mut t5_inv, &cmyka5_inv_u8, 2, F::Rgba8);
    assert_eq!(out5_std, out5_inv);
    assert_eq!(&out5_std[0..3], &out4[0..3]);
    assert_eq!(&out5_std[4..7], &out4[3..6]);
    assert_eq!(out5_std[3], 37);
    assert_eq!(out5_std[7], 201);

    // U16 CMYKA: 16-bit ink inversion leaves 16-bit alpha exact.
    let cmyka5_std_u16: [u16; 5] = [1000, 12000, 24000, 4000, 50_123];
    let cmyka5_inv_u16: [u16; 5] = [!1000, !12000, !24000, !4000, 50_123];
    let mut t16_std = build(&LcmsPort, Source::Icc(&cmyk_icc), SRGB, cmyka16, F::Rgba16);
    let mut t16_inv = build(
        &LcmsPort,
        Source::Icc(&cmyk_icc),
        SRGB,
        cmyka16_inv,
        F::Rgba16,
    );
    let out16_std = run(
        &mut t16_std,
        bytemuck::cast_slice(&cmyka5_std_u16),
        1,
        F::Rgba16,
    );
    let out16_inv = run(
        &mut t16_inv,
        bytemuck::cast_slice(&cmyka5_inv_u16),
        1,
        F::Rgba16,
    );
    assert_eq!(out16_std, out16_inv);
    let out16_words: &[u16] = bytemuck::cast_slice(&out16_std);
    assert_eq!(out16_words[3], 50_123);

    // F32 CMYKA: 0..1 ink scales to 0..100% internally while 5th-channel alpha stays 0..1.
    let cmyka5_std_f32: [f32; 5] = [0.1, 0.3, 0.5, 0.2, 0.42];
    let cmyka5_inv_f32: [f32; 5] = [0.9, 0.7, 0.5, 0.8, 0.42];
    let mut tf_std = build(
        &LcmsPort,
        Source::Icc(&cmyk_icc),
        SRGB,
        cmykaf32,
        F::RgbaF32,
    );
    let mut tf_inv = build(
        &LcmsPort,
        Source::Icc(&cmyk_icc),
        SRGB,
        cmykaf32_inv,
        F::RgbaF32,
    );
    let outf_std = run(
        &mut tf_std,
        bytemuck::cast_slice(&cmyka5_std_f32),
        1,
        F::RgbaF32,
    );
    let outf_inv = run(
        &mut tf_inv,
        bytemuck::cast_slice(&cmyka5_inv_f32),
        1,
        F::RgbaF32,
    );
    let f_std: &[f32] = bytemuck::cast_slice(&outf_std);
    let f_inv: &[f32] = bytemuck::cast_slice(&outf_inv);
    for (a, b) in f_std[..3].iter().zip(&f_inv[..3]) {
        assert!((a - b).abs() < 1e-5, "{f_std:?} vs {f_inv:?}");
    }
    assert_eq!(f_std[3].to_bits(), 0.42f32.to_bits());
    assert_eq!(f_inv[3].to_bits(), 0.42f32.to_bits());
}

#[test]
fn lab_and_xyz_icc_transforms_and_backend_agreement() {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, ColorModel};
    use zenpixels_convert::{ConvertError, RowConverter};

    let mut lab_prof = lcms::virt::create_lab4_profile(None, None).unwrap();
    let lab_icc = lcms::profile::save_profile_to_mem(&mut lab_prof).unwrap();
    let mut xyz_prof = lcms::virt::create_xyz_profile(None).unwrap();
    let xyz_icc = lcms::profile::save_profile_to_mem(&mut xyz_prof).unwrap();

    assert_eq!(ColorModel::from_icc(&lab_icc), Some(ColorModel::Lab));
    assert_eq!(ColorModel::from_icc(&xyz_icc), Some(ColorModel::Xyz));

    let lab8 = F::from_parts(ChannelType::U8, ChannelLayout::Lab, None).unwrap();
    let laba8 = F::from_parts(
        ChannelType::U8,
        ChannelLayout::LabAlpha,
        Some(AlphaMode::Straight),
    )
    .unwrap();
    let lab16 = F::from_parts(ChannelType::U16, ChannelLayout::Lab, None).unwrap();
    let labf32 = F::from_parts(ChannelType::F32, ChannelLayout::Lab, None).unwrap();
    let xyz8 = F::from_parts(ChannelType::U8, ChannelLayout::Xyz, None).unwrap();
    let xyz16 = F::from_parts(ChannelType::U16, ChannelLayout::Xyz, None).unwrap();
    let xyzf32 = F::from_parts(ChannelType::F32, ChannelLayout::Xyz, None).unwrap();

    // Without a CMS, Lab and XYZ descriptors require CMS rather than reinterpreting channels.
    for fmt in [lab8, lab16, labf32, xyz16, xyzf32] {
        let err =
            RowConverter::new(fmt.descriptor(), zenpixels::PixelDescriptor::RGB8_SRGB).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::NeedsCms { .. }));
    }

    // 8-bit XYZ is intentionally declined by both backends (no standard u1Fixed15 byte packing).
    for (_, cms) in backends() {
        assert!(
            cms.build_source_transform(Source::Icc(&xyz_icc), SRGB, xyz8, F::Rgb8, &opts())
                .is_none()
        );
    }

    // Lcms LabF32 <-> RgbF32: sRGB white (1, 1, 1) maps to L*=100, a*=0, b*=0 without 0..1 clamping.
    let mut srgb_to_lab = build(&LcmsPort, SRGB, Source::Icc(&lab_icc), F::RgbF32, labf32);
    let lab_white_bytes = run(
        &mut srgb_to_lab,
        bytemuck::cast_slice(&[1.0f32, 1.0, 1.0]),
        1,
        labf32,
    );
    let lab_white: &[f32] = bytemuck::cast_slice(&lab_white_bytes);
    assert!(
        (lab_white[0] - 100.0).abs() < 0.05,
        "Lab white: {lab_white:?}"
    );
    assert!(
        lab_white[1].abs() < 0.05 && lab_white[2].abs() < 0.05,
        "Lab white: {lab_white:?}"
    );

    let mut lab_to_srgb = build(&LcmsPort, Source::Icc(&lab_icc), SRGB, labf32, F::RgbF32);
    let rgb_back_bytes = run(&mut lab_to_srgb, &lab_white_bytes, 1, F::RgbF32);
    let rgb_back: &[f32] = bytemuck::cast_slice(&rgb_back_bytes);
    for &c in rgb_back {
        assert!((c - 1.0).abs() < 1e-3, "LabF32->sRGB white: {rgb_back:?}");
    }

    // Lcms XyzF32 <-> RgbF32: D50 PCS white (0.9642, 1.0, 0.8249) maps to sRGB white (1, 1, 1).
    let mut xyz_to_srgb = build(&LcmsPort, Source::Icc(&xyz_icc), SRGB, xyzf32, F::RgbF32);
    let xyz_d50 = [0.9642f32, 1.0, 0.8249];
    let xyz_rgb_bytes = run(
        &mut xyz_to_srgb,
        bytemuck::cast_slice(&xyz_d50),
        1,
        F::RgbF32,
    );
    let xyz_rgb: &[f32] = bytemuck::cast_slice(&xyz_rgb_bytes);
    for &c in xyz_rgb {
        assert!((c - 1.0).abs() < 2e-3, "XyzF32 D50->sRGB: {xyz_rgb:?}");
    }

    // Lab8, LabA8, Lab16 in Lcms and MoxCms (using a generic v4 Lab->Lab mAB identity profile so both parse it).
    let mut mox_lab = moxcms::ColorProfile::new_lab();
    mox_lab.pcs = moxcms::DataColorSpace::Lab;
    let mox_lab_icc = mox_lab.encode().unwrap();
    // Neutral L*=100 (255, 128, 128) and L*=50 (128, 128, 128) in ICC v4 8-bit Lab.
    let lab8_in = [255u8, 128, 128, 128, 128, 128];
    let mut mox_lab8 = build(&MoxCms, Source::Icc(&mox_lab_icc), SRGB, lab8, F::Rgb8);
    let mut lcms_lab8 = build(&LcmsPort, Source::Icc(&mox_lab_icc), SRGB, lab8, F::Rgb8);
    let out_mox8 = run(&mut mox_lab8, &lab8_in, 2, F::Rgb8);
    let out_lcms8 = run(&mut lcms_lab8, &lab8_in, 2, F::Rgb8);
    for (&a, &b) in out_mox8.iter().zip(&out_lcms8) {
        assert!(
            a.abs_diff(b) <= 2,
            "Lab8 mox vs lcms: {out_mox8:?} vs {out_lcms8:?}"
        );
    }

    // LabA8 preserves alpha across both backends.
    let laba8_in = [255u8, 128, 128, 91];
    for (name, cms) in backends() {
        let mut t = build(cms, Source::Icc(&mox_lab_icc), SRGB, laba8, F::Rgba8);
        let out = run(&mut t, &laba8_in, 1, F::Rgba8);
        assert_eq!(out[3], 91, "{name} LabA8 alpha: {out:?}");
    }

    // Lab16 neutral white (0xffff, 0x8080, 0x8080) and mid-gray (0x8000, 0x8080, 0x8080) across both backends.
    // Note: moxcms bakes a 33^3 3D LUT whose grid nodes (16/32 and 17/32) straddle 128/255 and clip at the
    // L*=100 gamut apex by ~0.5% (~331 u16 codes, ~1.3 u8 codes); inside the gamut (L*=50) both match tightly.
    let lab16_neutrals: [u16; 6] = [0xffff, 0x8080, 0x8080, 0x8000, 0x8080, 0x8080];
    let mut mox_lab16 = build(&MoxCms, Source::Icc(&mox_lab_icc), SRGB, lab16, F::Rgb16);
    let mut lcms_lab16 = build(&LcmsPort, Source::Icc(&mox_lab_icc), SRGB, lab16, F::Rgb16);
    let out_mox16: [u16; 6] = bytemuck::cast_slice(&run(
        &mut mox_lab16,
        bytemuck::cast_slice(&lab16_neutrals),
        2,
        F::Rgb16,
    ))
    .try_into()
    .unwrap();
    let out_lcms16: [u16; 6] = bytemuck::cast_slice(&run(
        &mut lcms_lab16,
        bytemuck::cast_slice(&lab16_neutrals),
        2,
        F::Rgb16,
    ))
    .try_into()
    .unwrap();
    for c in &out_mox16[..3] {
        assert!(*c >= 65100, "mox Lab16 white: {out_mox16:?}");
    }
    for c in &out_lcms16[..3] {
        assert!(*c >= 65500, "lcms Lab16 white: {out_lcms16:?}");
    }
    for (&a, &b) in out_mox16[3..].iter().zip(&out_lcms16[3..]) {
        assert!(
            a.abs_diff(b) <= 200,
            "Lab16 mid-gray mox vs lcms (<1 u8 code): {out_mox16:?} vs {out_lcms16:?}"
        );
    }

    // Xyz16 u1Fixed15 D50 white (0.9642 * 32768 = 31595, 32768, 0.8249 * 32768 = 27030) in Lcms.
    let xyz16_d50: [u16; 3] = [31595, 32768, 27030];
    let mut lcms_xyz16 = build(&LcmsPort, Source::Icc(&xyz_icc), SRGB, xyz16, F::Rgb16);
    let out_xyz16: [u16; 3] = bytemuck::cast_slice(&run(
        &mut lcms_xyz16,
        bytemuck::cast_slice(&xyz16_d50),
        1,
        F::Rgb16,
    ))
    .try_into()
    .unwrap();
    for c in out_xyz16 {
        assert!(c >= 65450, "lcms Xyz16 D50->sRGB: {out_xyz16:?}");
    }
}

#[test]
fn working_spaces_and_icc_parametric_curves_agree_across_backends() {
    use zenpixels::{
        IccParametricCurve, ParametricColor, ParametricPrimaries, ParametricTransfer, WhitePoint,
    };

    // Continuous 7-parameter ICC type-4 curve with distinct (g, a, b, c, d, e, f) and e != f.
    let type4_g = 2.4_f32;
    let type4_e = 0.01_f32;
    let type4_f = 0.002_f32;
    let type4_b = 0.05_f32;
    let type4_a = (1.0_f32 - type4_e).powf(1.0 / type4_g) - type4_b;
    let type4_d = 0.08_f32;
    let type4_c = ((type4_a * type4_d + type4_b).powf(type4_g) + type4_e - type4_f) / type4_d;

    let curves = [
        IccParametricCurve::type0_gamma(2.2),
        IccParametricCurve::type1_cie122(2.2, 0.947_867_3, 0.052_132_7),
        IccParametricCurve::type2_iec61966_3(2.2, 0.947_867_3, 0.052_132_7, 0.01),
        IccParametricCurve::SRGB,
        IccParametricCurve::type4(
            type4_g, type4_a, type4_b, type4_c, type4_d, type4_e, type4_f,
        ),
    ];

    let mut spaces = vec![
        ParametricColor::LINEAR_SRGB,
        ParametricColor::PROPHOTO_RGB,
        ParametricColor::ACES_CG,
        ParametricColor::ACES_2065_1,
        ParametricColor::rgb(
            ParametricPrimaries::Bt601_525,
            WhitePoint::D65,
            ParametricTransfer::Bt709,
        ),
        ParametricColor::rgb(
            ParametricPrimaries::Bt601_625,
            WhitePoint::C,
            ParametricTransfer::Gamma(zenpixels::GammaCurve::GAMMA_22),
        ),
    ];
    for c in curves {
        spaces.push(ParametricColor::rgb(
            ParametricPrimaries::P3,
            WhitePoint::D65,
            ParametricTransfer::IccParametric(c),
        ));
    }

    let input_rgb = row(F::Rgb8, 64);
    for space in spaces {
        let mut mox_to = build(&MoxCms, Source::Parametric(space), SRGB, F::Rgb8, F::Rgb8);
        let mut lcms_to = build(&LcmsPort, Source::Parametric(space), SRGB, F::Rgb8, F::Rgb8);
        let a = codes(F::Rgb8, &run(&mut mox_to, &input_rgb, 64, F::Rgb8));
        let b = codes(F::Rgb8, &run(&mut lcms_to, &input_rgb, 64, F::Rgb8));
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() <= 1.0,
                "working/curve space -> sRGB ({space:?}): {x} vs {y}"
            );
        }

        let mut mox_from = build(&MoxCms, SRGB, Source::Parametric(space), F::Rgb8, F::Rgb8);
        let mut lcms_from = build(&LcmsPort, SRGB, Source::Parametric(space), F::Rgb8, F::Rgb8);
        let c = codes(F::Rgb8, &run(&mut mox_from, &input_rgb, 64, F::Rgb8));
        let d = codes(F::Rgb8, &run(&mut lcms_from, &input_rgb, 64, F::Rgb8));
        for (x, y) in c.iter().zip(&d) {
            assert!(
                (x - y).abs() <= 1.0,
                "sRGB -> working/curve space ({space:?}): {x} vs {y}"
            );
        }
    }

    // Discontinuous piecewise curve: forward (source -> sRGB) agrees across both backends,
    // while MoxCms declines inverse (sRGB -> dst) cleanly so LcmsPort handles it in a chain.
    let discontinuous = ParametricColor::rgb(
        ParametricPrimaries::P3,
        WhitePoint::D65,
        ParametricTransfer::IccParametric(IccParametricCurve::type4(
            2.4,
            1.0 / 1.055,
            0.055 / 1.055,
            1.0 / 12.92,
            0.04045,
            0.02,
            0.0,
        )),
    );
    let mut mox_fwd = build(
        &MoxCms,
        Source::Parametric(discontinuous),
        SRGB,
        F::Rgb8,
        F::Rgb8,
    );
    let mut lcms_fwd = build(
        &LcmsPort,
        Source::Parametric(discontinuous),
        SRGB,
        F::Rgb8,
        F::Rgb8,
    );
    let fwd_a = codes(F::Rgb8, &run(&mut mox_fwd, &input_rgb, 64, F::Rgb8));
    let fwd_b = codes(F::Rgb8, &run(&mut lcms_fwd, &input_rgb, 64, F::Rgb8));
    for (x, y) in fwd_a.iter().zip(&fwd_b) {
        assert!((x - y).abs() <= 1.0, "discontinuous forward: {x} vs {y}");
    }
    assert!(
        MoxCms
            .build_source_transform(
                SRGB,
                Source::Parametric(discontinuous),
                F::Rgb8,
                F::Rgb8,
                &opts(),
            )
            .is_none()
    );
    assert!(
        LcmsPort
            .build_source_transform(
                SRGB,
                Source::Parametric(discontinuous),
                F::Rgb8,
                F::Rgb8,
                &opts(),
            )
            .unwrap()
            .is_ok()
    );
}

#[test]
fn active_rendering_intent_in_convert_options_and_color_context() {
    use lcms::{lut, profile, signatures::tag, tagtypes::TagObject, virt};
    use std::sync::Arc;
    use zenpixels::{
        ColorContext, ParametricColor, PixelDescriptor, PixelEncoding, RenderingIntent,
    };
    use zenpixels_convert::RowConverter;

    // Build an RGB LUT ICC profile whose AToB0 (Perceptual) scales by 0.5 and AToB1 (RelativeColorimetric) is identity.
    let mut multi = virt::create_srgb_profile(None).unwrap();
    let make_lut = |scale: f64| {
        let mut pipe = lut::pipeline_alloc(None, 3, 3).unwrap();
        let curve = lcms::gamma::build_gamma(None, 1.0).unwrap();
        assert!(pipe.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_tone_curves(None, 3, Some(&[&curve, &curve, &curve])).unwrap()
        ));
        // 2x2x2 CLUT mapping (r, g, b) -> PCS XYZ D50 * scale
        let mut table = Vec::with_capacity(8 * 3);
        for r in [0.0f64, 1.0] {
            for g in [0.0f64, 1.0] {
                for b in [0.0f64, 1.0] {
                    let x = (0.4360 * r + 0.3851 * g + 0.1431 * b) * scale;
                    let y = (0.2225 * r + 0.7169 * g + 0.0606 * b) * scale;
                    let z = (0.0139 * r + 0.0971 * g + 0.7139 * b) * scale;
                    table.push((x * 32768.0).round() as u16);
                    table.push((y * 32768.0).round() as u16);
                    table.push((z * 32768.0).round() as u16);
                }
            }
        }
        assert!(pipe.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_clut16bit(None, 2, 3, 3, Some(&table)).unwrap()
        ));
        assert!(pipe.insert_stage(
            lut::StageLoc::AtEnd,
            lut::stage_alloc_tone_curves(None, 3, Some(&[&curve, &curve, &curve])).unwrap()
        ));
        pipe
    };
    multi.pcs = lcms::signatures::colorspace::SigXYZData;
    assert!(profile::cms_write_tag(
        &mut multi,
        tag::SigAToB0Tag,
        Some(TagObject::LutA2B(make_lut(0.25)))
    ));
    assert!(profile::cms_write_tag(
        &mut multi,
        tag::SigAToB1Tag,
        Some(TagObject::LutA2B(make_lut(1.0)))
    ));
    let multi_icc: Arc<[u8]> = Arc::from(profile::save_profile_to_mem(&mut multi).unwrap());

    let srgb_ctx = ColorContext::from_cicp(zenpixels::Cicp::SRGB);
    let dst_enc = PixelEncoding::new(PixelDescriptor::RGB8_SRGB, Some(&srgb_ctx));
    let white_in = [255u8, 255, 255];

    for (name, cms) in backends() {
        // 1. ConvertOptions::with_rendering_intent switches between Perceptual (AToB0) and RelativeColorimetric (AToB1).
        let src_ctx = ColorContext::from_icc(multi_icc.clone());
        let src_enc = PixelEncoding::new(PixelDescriptor::RGB8, Some(&src_ctx));

        let mut rel = RowConverter::new_with_encoding(
            src_enc,
            dst_enc,
            &opts().with_rendering_intent(RenderingIntent::RelativeColorimetric),
            Some(cms),
        )
        .unwrap();
        let mut perc = RowConverter::new_with_encoding(
            src_enc,
            dst_enc,
            &opts().with_rendering_intent(RenderingIntent::Perceptual),
            Some(cms),
        )
        .unwrap();
        let mut out_rel = [0u8; 3];
        let mut out_perc = [0u8; 3];
        rel.try_convert_row(&white_in, &mut out_rel, 1).unwrap();
        perc.try_convert_row(&white_in, &mut out_perc, 1).unwrap();
        assert!(
            out_rel[1] > 240 && out_perc[1] < 160,
            "{name} intent failed to switch AToB0 vs AToB1: rel={out_rel:?} perc={out_perc:?}"
        );

        // 2. ColorContext::with_rendering_intent flows automatically when ConvertOptions leaves rendering_intent unset.
        let src_perc_ctx = ColorContext::from_icc(multi_icc.clone())
            .with_rendering_intent(RenderingIntent::Perceptual);
        let mut from_ctx = RowConverter::new_with_encoding(
            PixelEncoding::new(PixelDescriptor::RGB8, Some(&src_perc_ctx)),
            dst_enc,
            &opts(),
            Some(cms),
        )
        .unwrap();
        let mut out_ctx = [0u8; 3];
        from_ctx
            .try_convert_row(&white_in, &mut out_ctx, 1)
            .unwrap();
        assert_eq!(out_ctx, out_perc, "{name} ColorContext rendering_intent");
    }

    // 3. AbsoluteColorimetric: MoxCms declines cleanly (no media-white unadaptation in moxcms 0.9.2),
    // while LcmsPort evaluates true ICC absolute colorimetric white shift (D65 sRGB -> D50 ProPhoto).
    let abs_opts = opts()
        .with_clip_out_of_gamut(false)
        .with_rendering_intent(RenderingIntent::AbsoluteColorimetric);
    assert!(
        MoxCms
            .build_source_transform(
                SRGB,
                Source::Parametric(ParametricColor::PROPHOTO_RGB),
                F::RgbF32,
                F::RgbF32,
                &abs_opts,
            )
            .is_none()
    );
    let mut lcms_rel = LcmsPort
        .build_source_transform(
            SRGB,
            Source::Parametric(ParametricColor::PROPHOTO_RGB),
            F::RgbF32,
            F::RgbF32,
            &opts().with_clip_out_of_gamut(false),
        )
        .unwrap()
        .unwrap();
    let mut lcms_abs = LcmsPort
        .build_source_transform(
            SRGB,
            Source::Parametric(ParametricColor::PROPHOTO_RGB),
            F::RgbF32,
            F::RgbF32,
            &abs_opts,
        )
        .unwrap()
        .unwrap();
    let white_f32 = [1.0f32, 1.0, 1.0];
    let rel_out: [f32; 3] = bytemuck::cast_slice(&run(
        &mut lcms_rel,
        bytemuck::cast_slice(&white_f32),
        1,
        F::RgbF32,
    ))
    .try_into()
    .unwrap();
    let abs_out: [f32; 3] = bytemuck::cast_slice(&run(
        &mut lcms_abs,
        bytemuck::cast_slice(&white_f32),
        1,
        F::RgbF32,
    ))
    .try_into()
    .unwrap();
    assert!(
        (rel_out[2] - 1.0).abs() < 1e-3,
        "rel D65->D50 white: {rel_out:?}"
    );
    assert!(
        abs_out[2] > 1.05,
        "abs D65->D50 white must preserve D65 blue excess (>1.0): {abs_out:?}"
    );
}
