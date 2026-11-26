//! One behavioral contract for every `CmsBackend` (`MoxCms`, `Lcms`), then the backends against each other.

use std::sync::Arc;

use cms_contract::LcmsPort;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, PixelFormat, Primaries, RenderingIntent, SampleLayout,
    SampleType, SignalRange, Transfer,
};
use zenpixels_convert::{CmsBackend, CmsTransform, ConvertPolicy, MoxCms, RowPlan};

const ADOBE: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/AdobeCompat-v2.icc");
const P3: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/DisplayP3Compat-v4.icc");
const BT2020: &[u8] = include_bytes!("../../zenpixels-convert/src/profiles/Rec2020Compat-v4.icc");

/// Build a small synthetic CMYK `A2B0` `mft2` ICC profile in memory (9^4 grid -> Lab).
fn synthetic_cmyk_icc() -> Vec<u8> {
    const GRID: u32 = 9;
    let node_lab = |c: f64, m: f64, y: f64, k: f64| -> [u16; 3] {
        let lit = 1.0 - k;
        let l = 100.0 * lit * (1.0 - 0.3 * c - 0.4 * m - 0.1 * y);
        let a = 60.0 * lit * (m - c);
        let b = 70.0 * lit * (y - 0.5 * (c + m));
        [
            (l * 652.8).round() as u16,
            ((a + 128.0) * 256.0).round() as u16,
            ((b + 128.0) * 256.0).round() as u16,
        ]
    };

    let mut lut = Vec::new();
    lut.extend_from_slice(b"mft2");
    lut.extend_from_slice(&[0; 4]);
    lut.extend_from_slice(&[4, 3, GRID as u8, 0]);
    for row in 0..3 {
        for col in 0..3 {
            let one: i32 = if row == col { 0x10000 } else { 0 };
            lut.extend_from_slice(&one.to_be_bytes());
        }
    }
    lut.extend_from_slice(&2u16.to_be_bytes());
    lut.extend_from_slice(&2u16.to_be_bytes());
    let ramp = [0u16, 0xFFFF];
    for _ in 0..4 {
        for v in ramp {
            lut.extend_from_slice(&v.to_be_bytes());
        }
    }
    for node in 0..GRID.pow(4) {
        let ink = |place: u32| f64::from(node / GRID.pow(place) % GRID) / f64::from(GRID - 1);
        for v in node_lab(ink(3), ink(2), ink(1), ink(0)) {
            lut.extend_from_slice(&v.to_be_bytes());
        }
    }
    for _ in 0..3 {
        for v in ramp {
            lut.extend_from_slice(&v.to_be_bytes());
        }
    }

    const TAG_START: usize = 128 + 4 + 12;
    let size = TAG_START + lut.len();
    let mut icc = vec![0u8; 128];
    icc[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    icc[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    icc[12..16].copy_from_slice(b"prtr");
    icc[16..20].copy_from_slice(b"CMYK");
    icc[20..24].copy_from_slice(b"Lab ");
    icc[36..40].copy_from_slice(b"acsp");
    for (i, v) in [0x0000_F6D6u32, 0x0001_0000, 0x0000_D32D]
        .iter()
        .enumerate()
    {
        icc[68 + i * 4..72 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    icc.extend_from_slice(&1u32.to_be_bytes());
    icc.extend_from_slice(b"A2B0");
    icc.extend_from_slice(&(TAG_START as u32).to_be_bytes());
    icc.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    icc.extend_from_slice(&lut);
    icc
}

fn backends() -> [(&'static str, &'static dyn CmsBackend); 2] {
    [("moxcms", &MoxCms), ("lcms", &LcmsPort)]
}

#[derive(Clone)]
enum ProfileKind {
    Srgb,
    Icc(Arc<[u8]>),
}

impl ProfileKind {
    fn spec(&self, layout: SampleLayout, intent: RenderingIntent) -> PixelFormat {
        let alpha = if layout.order.has_alpha_slot() {
            AlphaMode::Straight
        } else {
            AlphaMode::None
        };
        let color = match self {
            Self::Srgb => ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha,
                white: None,
            },
            Self::Icc(profile) => ColorSpace::Icc {
                profile: Arc::clone(profile),
                channels: if matches!(layout.order, ChannelOrder::Gray | ChannelOrder::GrayAlpha) {
                    1
                } else {
                    3
                },
                alpha,
                intent,
            },
        };
        PixelFormat::new(layout, color)
    }
}

fn pairs() -> Vec<(&'static str, ProfileKind, ProfileKind)> {
    let adobe = ProfileKind::Icc(Arc::from(ADOBE));
    let p3 = ProfileKind::Icc(Arc::from(P3));
    let bt2020 = ProfileKind::Icc(Arc::from(BT2020));
    vec![
        ("adobe>srgb", adobe, ProfileKind::Srgb),
        ("p3>srgb", p3.clone(), ProfileKind::Srgb),
        ("2020>srgb", bt2020.clone(), ProfileKind::Srgb),
        ("srgb>p3", ProfileKind::Srgb, p3.clone()),
        ("srgb>2020", ProfileKind::Srgb, bt2020.clone()),
        ("p3>2020", p3, bt2020),
    ]
}

const LAYOUTS: [SampleLayout; 6] = [
    SampleLayout::RGB8,
    SampleLayout::RGBA8,
    SampleLayout::RGB16,
    SampleLayout::RGBA16,
    SampleLayout::RGBF32,
    SampleLayout::RGBAF32,
];

fn build(cms: &dyn CmsBackend, src: &PixelFormat, dst: &PixelFormat) -> Arc<dyn CmsTransform> {
    cms.build_transform(src, dst)
        .unwrap_or_else(|e| panic!("build_transform failed: {e:?}"))
}

fn row(layout: SampleLayout, n: usize) -> Vec<u8> {
    let color = |i: usize, c: usize| ((i * [7, 13, 29][c] + c * 11) % 256) as f32 / 255.0;
    let alpha = |i: usize| (i % 256) as f32 / 255.0;
    let mut out = Vec::with_capacity(n * layout.bytes_per_pixel());
    for i in 0..n {
        for c in 0..layout.channels() {
            let v = if c < 3 { color(i, c) } else { alpha(i) };
            match layout.sample {
                SampleType::U8 => out.push((v * 255.0).round() as u8),
                SampleType::U16 => {
                    out.extend_from_slice(&((v * 65535.0).round() as u16).to_ne_bytes());
                }
                SampleType::F32 => out.extend_from_slice(&v.to_ne_bytes()),
                _ => unreachable!(),
            }
        }
    }
    out
}

fn run(t: &dyn CmsTransform, src: &[u8], n: usize, to: SampleLayout) -> Vec<u8> {
    let mut dst = vec![0u8; n * to.bytes_per_pixel()];
    t.transform_row(src, &mut dst, n as u32).expect("row");
    dst
}

fn codes(layout: SampleLayout, bytes: &[u8]) -> Vec<f64> {
    match layout.sample {
        SampleType::U8 => bytes.iter().map(|&v| f64::from(v)).collect(),
        SampleType::U16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f64::from(u16::from_ne_bytes(*b)) / 257.0)
            .collect(),
        SampleType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_ne_bytes(*b)) * 255.0)
            .collect(),
        _ => unreachable!(),
    }
}

fn worst_with_witness(layout: SampleLayout, a: &[u8], b: &[u8]) -> (f64, usize, f64, f64) {
    let mut best = (0.0f64, 0usize, 0.0f64, 0.0f64);
    for (i, (x, y)) in codes(layout, a)
        .into_iter()
        .zip(codes(layout, b))
        .enumerate()
    {
        let d = (x - y).abs();
        if d > best.0 {
            best = (d, i, x, y);
        }
    }
    best
}

fn worst(layout: SampleLayout, a: &[u8], b: &[u8]) -> f64 {
    worst_with_witness(layout, a, b).0
}

#[test]
fn garbage_icc_is_never_accepted() {
    let dst = PixelFormat::RGB8_SRGB;
    let mut truncated = P3.to_vec();
    truncated.truncate(200);
    for (name, cms) in backends() {
        for bad in [&b"not a profile"[..], &truncated, &[]] {
            let src = PixelFormat::new(
                SampleLayout::RGB8,
                ColorSpace::Icc {
                    profile: Arc::from(bad),
                    channels: 3,
                    alpha: AlphaMode::None,
                    intent: RenderingIntent::RelativeColorimetric,
                },
            );
            assert!(
                cms.build_transform(&src, &dst).is_err(),
                "{name} accepted {} garbage ICC bytes",
                bad.len()
            );
        }
    }
}

#[test]
fn rows_split_anywhere() {
    for (name, cms) in backends() {
        for (pair, src_p, dst_p) in pairs() {
            for layout in LAYOUTS {
                let src = src_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let dst = dst_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let t = build(cms, &src, &dst);
                let input = row(layout, 600);
                let whole = run(t.as_ref(), &input, 600, layout);
                let bpp = layout.bytes_per_pixel();
                let mut parts = Vec::new();
                for chunk in input.chunks(7 * bpp) {
                    parts.extend(run(t.as_ref(), chunk, chunk.len() / bpp, layout));
                }
                assert_eq!(whole, parts, "{name} {pair} {layout:?}");
                assert_eq!(
                    whole,
                    run(t.as_ref(), &input, 600, layout),
                    "{name} {pair} {layout:?} repeat"
                );
            }
        }
    }
}

#[test]
fn only_the_row_is_touched() {
    let p3 = ProfileKind::Icc(Arc::from(P3));
    for (name, cms) in backends() {
        for layout in LAYOUTS {
            let src = p3.spec(layout, RenderingIntent::RelativeColorimetric);
            let dst = ProfileKind::Srgb.spec(layout, RenderingIntent::RelativeColorimetric);
            let t = build(cms, &src, &dst);
            let bpp = layout.bytes_per_pixel();
            let input = row(layout, 40);
            let expected = run(t.as_ref(), &input[..32 * bpp], 32, layout);
            let mut out = vec![0xA5u8; 40 * bpp];
            t.transform_row(&input, &mut out, 32).expect("row");
            assert_eq!(&out[..32 * bpp], &expected[..], "{name} {layout:?}");
            assert!(
                out[32 * bpp..].iter().all(|&b| b == 0xA5),
                "{name} {layout:?} wrote past the row"
            );
            let before = out.clone();
            t.transform_row(&input, &mut out, 0).expect("empty row");
            assert_eq!(before, out, "{name} {layout:?} wrote on an empty row");
        }
    }
}

#[test]
fn alpha_is_carried_exactly() {
    for (name, cms) in backends() {
        for (pair, src_p, dst_p) in pairs() {
            for layout in [
                SampleLayout::RGBA8,
                SampleLayout::RGBA16,
                SampleLayout::RGBAF32,
            ] {
                let sample = layout.sample.byte_size();
                let src = src_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let dst = dst_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let t = build(cms, &src, &dst);
                let input = row(layout, 300);
                let out = run(t.as_ref(), &input, 300, layout);
                let alpha = |r: &[u8]| -> Vec<u8> {
                    r.chunks_exact(4 * sample)
                        .flat_map(|p| p[3 * sample..].to_vec())
                        .collect()
                };
                assert_eq!(alpha(&input), alpha(&out), "{name} {pair} {layout:?}");
            }
        }
    }
}

#[test]
fn white_and_black_hold() {
    for (name, cms) in backends() {
        for (pair, src_p, dst_p) in pairs() {
            for layout in [
                SampleLayout::RGB8,
                SampleLayout::RGB16,
                SampleLayout::RGBF32,
            ] {
                let src = src_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let dst = dst_p.spec(layout, RenderingIntent::RelativeColorimetric);
                let t = build(cms, &src, &dst);
                for (level, target) in [(1.0f32, 255.0), (0.0, 0.0)] {
                    let input: Vec<u8> = match layout.sample {
                        SampleType::U8 => vec![(level * 255.0) as u8; 3],
                        SampleType::U16 => {
                            let v = (level * 65535.0) as u16;
                            [v, v, v].iter().flat_map(|v| v.to_ne_bytes()).collect()
                        }
                        SampleType::F32 => {
                            [level; 3].iter().flat_map(|v| v.to_ne_bytes()).collect()
                        }
                        _ => unreachable!(),
                    };
                    for v in codes(layout, &run(t.as_ref(), &input, 1, layout)) {
                        assert!(
                            (v - target).abs() < 0.51,
                            "{name} {pair} {layout:?} {level}: {v}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn relative_intent_maps_paper_white_to_destination_white() {
    // Append a `wtpt` tag representing a printing medium whose measured white is 0.8 * D50.
    // Relative intent maps its relative A2B white to destination white (1.0);
    // absolute intent retains the darker medium white.
    let mut bytes = synthetic_cmyk_icc();
    let count = u32::from_be_bytes(bytes[128..132].try_into().unwrap());
    let dir_end = 132 + (count as usize) * 12;
    let mut wtpt = Vec::with_capacity(20);
    wtpt.extend_from_slice(b"XYZ ");
    wtpt.extend_from_slice(&[0u8; 4]);
    let s15f16 = |v: f64| ((v * 65536.0).round() as i32).to_be_bytes();
    wtpt.extend_from_slice(&s15f16(0.9642 * 0.8));
    wtpt.extend_from_slice(&s15f16(0.8));
    wtpt.extend_from_slice(&s15f16(0.8249 * 0.8));

    let payload_off = bytes.len() as u32 + 12;
    for i in 0..count as usize {
        let off = 132 + i * 12 + 4;
        let old = u32::from_be_bytes(bytes[off..off + 4].try_into().unwrap());
        bytes[off..off + 4].copy_from_slice(&(old + 12).to_be_bytes());
    }
    bytes[128..132].copy_from_slice(&(count + 1).to_be_bytes());
    let mut entry = [0u8; 12];
    entry[0..4].copy_from_slice(b"wtpt");
    entry[4..8].copy_from_slice(&payload_off.to_be_bytes());
    entry[8..12].copy_from_slice(&(wtpt.len() as u32).to_be_bytes());
    bytes.splice(dir_end..dir_end, entry);
    bytes.extend_from_slice(&wtpt);
    let total = bytes.len() as u32;
    bytes[0..4].copy_from_slice(&total.to_be_bytes());

    let bytes: Arc<[u8]> = Arc::from(bytes);
    let src = PixelFormat::new(
        SampleLayout::CMYKF32,
        ColorSpace::Cmyk {
            icc: Some(bytes),
            alpha: AlphaMode::None,
            inverted: false,
        },
    );
    let dst = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let t = LcmsPort.build_transform(&src, &dst).unwrap();
    let input = [0.0f32; 4];
    let mut output = [0.0f32; 3];
    t.transform_row(
        bytemuck::cast_slice(&input),
        bytemuck::cast_slice_mut(&mut output),
        1,
    )
    .unwrap();
    for value in output {
        assert!(
            (value - 1.0).abs() < 0.002,
            "relative paper white: {output:?}"
        );
    }
}

fn rgbf32_srgb() -> PixelFormat {
    PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    )
}

#[test]
fn f32_cmyk_ink_polarity_and_limits() {
    let icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());
    let dst = rgbf32_srgb();
    let plain_spec = PixelFormat::new(
        SampleLayout::CMYKF32,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&icc)),
            alpha: AlphaMode::None,
            inverted: false,
        },
    );
    let flipped_spec = PixelFormat::new(
        SampleLayout::new(SampleType::F32, ChannelOrder::CmykInverted),
        ColorSpace::Cmyk {
            icc: Some(icc),
            alpha: AlphaMode::None,
            inverted: true,
        },
    );
    for (name, cms) in backends() {
        let plain = build(cms, &plain_spec, &dst);
        let flipped = build(cms, &flipped_spec, &dst);
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        let show = |t: &dyn CmsTransform, ink: [f32; 4]| {
            codes(
                SampleLayout::RGBF32,
                &run(t, &f32s(&ink), 1, SampleLayout::RGBF32),
            )
        };
        for (ink, target) in [([0.0; 4], 255.0), ([0.0, 0.0, 0.0, 1.0], 0.0)] {
            let got = show(plain.as_ref(), ink);
            for v in &got {
                assert!((v - target).abs() < 1.0, "{name} ink {ink:?}: {got:?}");
            }
            let inverted = ink.map(|v| 1.0 - v);
            let other = show(flipped.as_ref(), inverted);
            for (a, b) in got.iter().zip(&other) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "{name} polarity: {got:?} vs {other:?}"
                );
            }
        }
        // Synthetic profile has only A2B0 (no B2A0): RGB -> CMYK must be rejected!
        assert!(cms.build_transform(&dst, &plain_spec).is_err());
    }
}

#[test]
fn backends_agree() {
    let [(_, mox), (_, lcms)] = backends();
    let mut failed = Vec::new();
    for (pair, src_p, dst_p) in pairs() {
        for layout in LAYOUTS {
            let src = src_p.spec(layout, RenderingIntent::RelativeColorimetric);
            let dst = dst_p.spec(layout, RenderingIntent::RelativeColorimetric);
            let input = row(layout, 4096);
            let a = run(build(mox, &src, &dst).as_ref(), &input, 4096, layout);
            let b = run(build(lcms, &src, &dst).as_ref(), &input, 4096, layout);
            let (d, idx, vx, vy) = worst_with_witness(layout, &a, &b);
            let limit = if layout.sample == SampleType::U8 {
                1.0
            } else {
                0.2
            };
            if d > limit {
                failed.push(format!(
                    "{pair} {layout:?}: {d:.3} > {limit} at ch {idx} (mox={vx:.4}, lcms={vy:.4})"
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn f32_cmyk_backends_agree() {
    let icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());
    let src = PixelFormat::new(
        SampleLayout::CMYKF32,
        ColorSpace::Cmyk {
            icc: Some(icc),
            alpha: AlphaMode::None,
            inverted: false,
        },
    );
    let dst = rgbf32_srgb();
    let ink: Vec<u8> = (0..4096 * 4)
        .flat_map(|i: u32| (((i * 37 + i / 4 * 11) % 101) as f32 / 100.0).to_ne_bytes())
        .collect();
    // Drive through RowPlan with default GamutPolicy::Clip (matching 0.4 ConvertOptions defaults).
    let pm = RowPlan::build(
        src.clone(),
        dst.clone(),
        &ConvertPolicy::permissive().with_cms(Arc::new(MoxCms)),
        None,
    )
    .unwrap();
    let pl = RowPlan::build(
        src,
        dst,
        &ConvertPolicy::permissive().with_cms(Arc::new(LcmsPort)),
        None,
    )
    .unwrap();
    let mut a = vec![0u8; 4096 * SampleLayout::RGBF32.bytes_per_pixel()];
    let mut b = vec![0u8; 4096 * SampleLayout::RGBF32.bytes_per_pixel()];
    pm.convert_row(&ink, &mut a, 4096).unwrap();
    pl.convert_row(&ink, &mut b, 4096).unwrap();
    let (d, idx, vx, vy) = worst_with_witness(SampleLayout::RGBF32, &a, &b);
    assert!(
        d < 2.5,
        "synthetic CMYK F32 worst diff {d:.3} >= 2.5 at ch {idx} (mox={vx:.4}, lcms={vy:.4})"
    );
}

#[test]
fn gray_round_trips_and_carries_alpha() {
    let icc: Arc<[u8]> = Arc::from(cms_contract::gray_gamma22_icc());
    let src_gray = PixelFormat::new(
        SampleLayout::GRAY8,
        ColorSpace::Icc {
            profile: Arc::clone(&icc),
            channels: 1,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_rgb = PixelFormat::RGB8_SRGB;
    let [(_, mox), (_, lcms)] = backends();
    let ramp: Vec<u8> = (0..=255).collect();
    let a = run(
        build(mox, &src_gray, &dst_rgb).as_ref(),
        &ramp,
        256,
        SampleLayout::RGB8,
    );
    let b = run(
        build(lcms, &src_gray, &dst_rgb).as_ref(),
        &ramp,
        256,
        SampleLayout::RGB8,
    );
    let d = worst(SampleLayout::RGB8, &a, &b);
    assert!(d <= 1.0, "gray 2.2>srgb diff {d:.3}");
    assert_eq!(a.as_chunks::<3>().0.first().map(|p| p[0]), Some(0));
    assert_eq!(a.as_chunks::<3>().0.last().map(|p| p[0]), Some(255));

    let src_ga = PixelFormat::new(
        SampleLayout::GRAYA8,
        ColorSpace::Icc {
            profile: icc,
            channels: 1,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_rgba = PixelFormat::RGBA8_SRGB;
    let with_alpha: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v]).collect();
    for (name, cms) in backends() {
        let t = build(cms, &src_ga, &dst_rgba);
        let out = run(t.as_ref(), &with_alpha, 256, SampleLayout::RGBA8);
        for (i, p) in out.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(p[3], (255 - i) as u8, "{name} gray alpha at {i}");
        }
    }
}

#[test]
fn eight_bit_cmyk_and_inverted_polarity_agree() {
    let icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());
    let src_std = PixelFormat::new(
        SampleLayout::CMYK8,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&icc)),
            alpha: AlphaMode::None,
            inverted: false,
        },
    );
    let src_inv = PixelFormat::new(
        SampleLayout::new(SampleType::U8, ChannelOrder::CmykInverted),
        ColorSpace::Cmyk {
            icc: Some(icc),
            alpha: AlphaMode::None,
            inverted: true,
        },
    );
    let dst = PixelFormat::RGB8_SRGB;
    let ink: Vec<u8> = (0..1024 * 4)
        .map(|i| ((i * 53 + 17) & 0xFF) as u8)
        .collect();
    let inv_ink: Vec<u8> = ink.iter().map(|&b| 255 - b).collect();

    let [(_, mox), (_, lcms)] = backends();
    let a = run(
        build(mox, &src_std, &dst).as_ref(),
        &ink,
        1024,
        SampleLayout::RGB8,
    );
    let a_inv = run(
        build(mox, &src_inv, &dst).as_ref(),
        &inv_ink,
        1024,
        SampleLayout::RGB8,
    );
    let b = run(
        build(lcms, &src_std, &dst).as_ref(),
        &ink,
        1024,
        SampleLayout::RGB8,
    );
    let b_inv = run(
        build(lcms, &src_inv, &dst).as_ref(),
        &inv_ink,
        1024,
        SampleLayout::RGB8,
    );
    assert_eq!(
        a, a_inv,
        "MoxCms inverted CMYK8 must match negated standard bit-exact"
    );
    assert_eq!(
        b, b_inv,
        "Lcms inverted CMYK8 must match negated standard bit-exact"
    );
    let d = worst(SampleLayout::RGB8, &a, &b);
    assert!(d <= 3.0, "CMYK8 backend diff {d:.3}");
}

#[test]
fn row_plan_drives_every_backend_without_substitution() {
    let p3_spec = ProfileKind::Icc(Arc::from(P3))
        .spec(SampleLayout::RGB8, RenderingIntent::RelativeColorimetric);
    let srgb_spec = PixelFormat::RGB8_SRGB;
    let input = row(SampleLayout::RGB8, 256);

    let pol_mox = ConvertPolicy::permissive()
        .with_icc_substitution(false)
        .with_cms(Arc::new(MoxCms));
    let pol_lcms = ConvertPolicy::permissive()
        .with_icc_substitution(false)
        .with_cms(Arc::new(LcmsPort));

    let pm = RowPlan::build(p3_spec.clone(), srgb_spec.clone(), &pol_mox, None).unwrap();
    let pl = RowPlan::build(p3_spec, srgb_spec, &pol_lcms, None).unwrap();
    let mut om = vec![0u8; 256 * 3];
    let mut ol = vec![0u8; 256 * 3];
    pm.convert_row(&input, &mut om, 256).unwrap();
    pl.convert_row(&input, &mut ol, 256).unwrap();
    let d = worst(SampleLayout::RGB8, &om, &ol);
    assert!(d <= 1.0, "RowPlan MoxCms vs Lcms diff {d:.3}");
}

#[test]
fn float_mpe_and_absolute_intent_guards_hold() {
    // Patch P3 ICC with a synthetic D2B1 directory tag: MoxCms must refuse RelativeColorimetric
    // while still accepting Perceptual (which uses D2B0, absent).
    let mut mpe = P3.to_vec();
    let count = u32::from_be_bytes(mpe[128..132].try_into().unwrap());
    let dir_end = 132 + (count as usize) * 12;
    let payload_off = mpe.len() as u32;
    mpe.extend_from_slice(&[0u8; 16]);
    mpe[128..132].copy_from_slice(&(count + 1).to_be_bytes());
    let mut entry = [0u8; 12];
    entry[0..4].copy_from_slice(b"D2B1");
    entry[4..8].copy_from_slice(&(payload_off + 12).to_be_bytes());
    entry[8..12].copy_from_slice(&16u32.to_be_bytes());
    for i in 0..count as usize {
        let off = 132 + i * 12 + 4;
        let old = u32::from_be_bytes(mpe[off..off + 4].try_into().unwrap());
        mpe[off..off + 4].copy_from_slice(&(old + 12).to_be_bytes());
    }
    mpe.splice(dir_end..dir_end, entry);
    let new_size = mpe.len() as u32;
    mpe[0..4].copy_from_slice(&new_size.to_be_bytes());

    let src_rel = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(mpe.clone()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let src_perc = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(mpe),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    let dst = PixelFormat::RGB8_SRGB;
    assert!(!MoxCms.supports_formats(&src_rel, &dst));
    assert!(MoxCms.build_transform(&src_rel, &dst).is_err());
    assert!(MoxCms.supports_formats(&src_perc, &dst));
    assert!(MoxCms.build_transform(&src_perc, &dst).is_ok());

    // AbsoluteColorimetric: MoxCms must refuse; Lcms must succeed!
    let src_abs = ProfileKind::Icc(Arc::from(P3))
        .spec(SampleLayout::RGB8, RenderingIntent::AbsoluteColorimetric);
    assert!(!MoxCms.supports_formats(&src_abs, &dst));
    assert!(MoxCms.build_transform(&src_abs, &dst).is_err());
    assert!(LcmsPort.supports_formats(&src_abs, &dst));
    assert!(LcmsPort.build_transform(&src_abs, &dst).is_ok());
}
