//! CMS shootout: `imazen/lcms` (`main`), `moxcms`, and `zenpixels-convert` 0.5 native SoA.
//!
//! Compares:
//! 1. Display P3 ICC (`DISPLAY_P3_V4`, matrix+TRC) -> sRGB at 4K UHD (3840x2160)
//!    across RGB8, RGBA8, RGB16, RGBF32 (`Lcms`, `LcmsUnoptimized`, `MoxCms`,
//!    native `RowPlan` ALL & SCALAR).
//! 2. Adobe RGB ICC (`ADOBE_RGB`) -> sRGB at 4K UHD (`Lcms`, `MoxCms`, native auto-sub).
//! 3. 3D RGB CLUT ICC (17^3 `mft2`) -> sRGB at 4K UHD (`Lcms`, `LcmsUnoptimized`, `MoxCms`).
//! 4. CMYK 4D CLUT ICC (9^4 `mft2`) -> sRGB at 4096-px rows & 4K UHD
//!    (`CMYK8`, `CMYK8_INVERTED`, `CMYK16`, `CMYKF32`).
//!
//! Run:
//!   cargo bench -p zenpixels-convert --all-features --bench bench_cms_shootout -- --no-busy-gate

use std::sync::Arc;
use std::time::Duration;
use zenbench::prelude::*;
use zenpixels_convert::{
    Accelerators, AlphaMode, CmsBackend, ColorSpace, ConvertPolicy, Lcms, LcmsUnoptimized, MoxCms,
    PixelFormat, Primaries, RenderingIntent, RowPlan, SampleLayout, SampleType, SignalRange,
    Transfer, icc_profiles,
};

const W_4K: u32 = 3840;
const H_4K: u32 = 2160;
const ROW_W: u32 = 4096;

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

fn synthetic_rgb_lut_icc() -> Vec<u8> {
    const NODES: u32 = 17;
    const M: [[f64; 3]; 3] = [
        [0.4360, 0.3851, 0.1431],
        [0.2225, 0.7169, 0.0606],
        [0.0139, 0.0971, 0.7139],
    ];
    const WHITE: [f64; 3] = [0.9642, 1.0, 0.8249];
    let lab_f = |t: f64| {
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    };
    let decode = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut lut = Vec::new();
    lut.extend_from_slice(b"mft2");
    lut.extend_from_slice(&[0; 4]);
    lut.extend_from_slice(&[3, 3, NODES as u8, 0]);
    for row in 0..3 {
        for col in 0..3 {
            let one: i32 = if row == col { 0x10000 } else { 0 };
            lut.extend_from_slice(&one.to_be_bytes());
        }
    }
    lut.extend_from_slice(&2u16.to_be_bytes());
    lut.extend_from_slice(&2u16.to_be_bytes());
    let ramp = [0u16, 0xFFFF];
    for _ in 0..3 {
        for v in ramp {
            lut.extend_from_slice(&v.to_be_bytes());
        }
    }
    for node in 0..NODES.pow(3) {
        let ch =
            |place: u32| decode(f64::from(node / NODES.pow(place) % NODES) / f64::from(NODES - 1));
        let rgb = [ch(2), ch(1), ch(0)];
        let fxyz = [0, 1, 2].map(|i| {
            let v = M[i][0] * rgb[0] + M[i][1] * rgb[1] + M[i][2] * rgb[2];
            lab_f(v / WHITE[i])
        });
        let l = 116.0 * fxyz[1] - 16.0;
        let a = 500.0 * (fxyz[0] - fxyz[1]);
        let b = 200.0 * (fxyz[1] - fxyz[2]);
        for v in [l * 652.8, (a + 128.0) * 256.0, (b + 128.0) * 256.0] {
            lut.extend_from_slice(&(v.round().clamp(0.0, 65535.0) as u16).to_be_bytes());
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
    icc[12..16].copy_from_slice(b"spac");
    icc[16..20].copy_from_slice(b"RGB ");
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

fn fill_u8(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(37).wrapping_add(i / 7 * 13)) % 251) as u8)
        .collect()
}

fn fill_u16(samples: usize) -> Vec<u16> {
    (0..samples)
        .map(|i| ((i.wrapping_mul(9973).wrapping_add(12345)) & 0xFFFF) as u16)
        .collect()
}

fn fill_f32(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| ((i.wrapping_mul(37).wrapping_add(17)) % 1009) as f32 / 1008.0)
        .collect()
}

fn icc_spec(layout: SampleLayout, profile: Arc<[u8]>, alpha: AlphaMode) -> PixelFormat {
    PixelFormat::new(
        layout,
        ColorSpace::Icc {
            profile,
            color_channels: 3,
            alpha,
            intent: RenderingIntent::RelativeColorimetric,
        },
    )
}

fn rgb_spec(
    layout: SampleLayout,
    primaries: Primaries,
    transfer: Transfer,
    alpha: AlphaMode,
) -> PixelFormat {
    PixelFormat::new(
        layout,
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            alpha,
            white: None,
        },
    )
}

fn bench_4k_rgb_u8(
    suite: &mut zenbench::Suite,
    label: &str,
    icc: Arc<[u8]>,
    layout: SampleLayout,
    alpha: AlphaMode,
    native_src: Option<(Primaries, Transfer)>,
) {
    let source_format = icc_spec(layout, Arc::clone(&icc), alpha);
    let destination_format = rgb_spec(layout, Primaries::Bt709, Transfer::Srgb, alpha);
    let bpp = layout.bytes_per_pixel();
    let row_bytes = W_4K as usize * bpp;
    let total_bytes = row_bytes * H_4K as usize;

    let lcms_t = Lcms
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let lcms_unopt_t = LcmsUnoptimized
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let mox_t = MoxCms
        .build_transform(&source_format, &destination_format)
        .unwrap();

    let src = Arc::new(fill_u8(total_bytes));

    suite.compare(label, |g| {
        g.throughput(Throughput::Bytes((total_bytes * 2) as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(6)
            .max_rounds(6)
            .max_time(Duration::from_millis(1200));

        {
            let s = Arc::clone(&src);
            let t = Arc::clone(&lcms_t);
            let mut dst = vec![0u8; total_bytes];
            g.bench("lcms_main_opt", move |b| {
                b.iter(|| {
                    for y in 0..H_4K as usize {
                        let off = y * row_bytes;
                        t.transform_row(
                            &s[off..off + row_bytes],
                            &mut dst[off..off + row_bytes],
                            W_4K,
                        )
                        .unwrap();
                    }
                    black_box(dst[total_bytes - 1])
                })
            });
        }
        {
            let s = Arc::clone(&src);
            let t = Arc::clone(&lcms_unopt_t);
            let mut dst = vec![0u8; total_bytes];
            g.bench("lcms_unopt", move |b| {
                b.iter(|| {
                    for y in 0..H_4K as usize {
                        let off = y * row_bytes;
                        t.transform_row(
                            &s[off..off + row_bytes],
                            &mut dst[off..off + row_bytes],
                            W_4K,
                        )
                        .unwrap();
                    }
                    black_box(dst[total_bytes - 1])
                })
            });
        }
        {
            let s = Arc::clone(&src);
            let t = Arc::clone(&mox_t);
            let mut dst = vec![0u8; total_bytes];
            g.bench("moxcms_0.9.2", move |b| {
                b.iter(|| {
                    for y in 0..H_4K as usize {
                        let off = y * row_bytes;
                        t.transform_row(
                            &s[off..off + row_bytes],
                            &mut dst[off..off + row_bytes],
                            W_4K,
                        )
                        .unwrap();
                    }
                    black_box(dst[total_bytes - 1])
                })
            });
        }
        if let Some((prim, tf)) = native_src {
            let nat_src = rgb_spec(layout, prim, tf, alpha);
            let plan_simd = RowPlan::build(
                nat_src.clone(),
                destination_format.clone(),
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap();
            let s = Arc::clone(&src);
            let mut dst = vec![0u8; total_bytes];
            g.bench("native_05_ALL", move |b| {
                b.iter(|| {
                    for y in 0..H_4K {
                        let off = y as usize * row_bytes;
                        plan_simd
                            .convert_row(
                                &s[off..off + row_bytes],
                                &mut dst[off..off + row_bytes],
                                W_4K,
                            )
                            .unwrap();
                    }
                    black_box(dst[total_bytes - 1])
                })
            });

            let plan_approx = RowPlan::build(
                nat_src,
                destination_format,
                &ConvertPolicy::permissive()
                    .with_approximate(true)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap();
            let s2 = Arc::clone(&src);
            let mut dst2 = vec![0u8; total_bytes];
            g.bench("native_05_approx_q14", move |b| {
                b.iter(|| {
                    for y in 0..H_4K {
                        let off = y as usize * row_bytes;
                        plan_approx
                            .convert_row(
                                &s2[off..off + row_bytes],
                                &mut dst2[off..off + row_bytes],
                                W_4K,
                            )
                            .unwrap();
                    }
                    black_box(dst2[total_bytes - 1])
                })
            });
        }
    });
}

fn bench_4k_rgb_u16(
    suite: &mut zenbench::Suite,
    label: &str,
    icc: Arc<[u8]>,
    native_src: Option<(Primaries, Transfer)>,
) {
    let layout = SampleLayout::RGB16;
    let source_format = icc_spec(layout, Arc::clone(&icc), AlphaMode::None);
    let destination_format = rgb_spec(layout, Primaries::Bt709, Transfer::Srgb, AlphaMode::None);
    let row_elems = W_4K as usize * 3;
    let total_elems = row_elems * H_4K as usize;
    let total_bytes = total_elems * 2;

    let lcms_t = Lcms
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let lcms_unopt_t = LcmsUnoptimized
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let mox_t = MoxCms
        .build_transform(&source_format, &destination_format)
        .unwrap();

    let src = Arc::new(fill_u16(total_elems));

    suite.compare(label, |g| {
        g.throughput(Throughput::Bytes((total_bytes * 2) as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(6)
            .max_rounds(6)
            .max_time(Duration::from_millis(1200));

        for (name, t) in [
            ("lcms_main_opt", lcms_t),
            ("lcms_unopt", lcms_unopt_t),
            ("moxcms_0.9.2", mox_t),
        ] {
            let s = Arc::clone(&src);
            let mut dst = vec![0u16; total_elems];
            g.bench(name, move |b| {
                b.iter(|| {
                    for y in 0..H_4K as usize {
                        let off = y * row_elems;
                        t.transform_row(
                            bytemuck::cast_slice(&s[off..off + row_elems]),
                            bytemuck::cast_slice_mut(&mut dst[off..off + row_elems]),
                            W_4K,
                        )
                        .unwrap();
                    }
                    black_box(dst[total_elems - 1])
                })
            });
        }
        if let Some((prim, tf)) = native_src {
            let nat_src = rgb_spec(layout, prim, tf, AlphaMode::None);
            let plan_simd = RowPlan::build(
                nat_src,
                destination_format,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap();
            let s = Arc::clone(&src);
            let mut dst = vec![0u16; total_elems];
            g.bench("native_05_ALL", move |b| {
                b.iter(|| {
                    for y in 0..H_4K {
                        let off = y as usize * row_elems;
                        plan_simd
                            .convert_row(
                                bytemuck::cast_slice(&s[off..off + row_elems]),
                                bytemuck::cast_slice_mut(&mut dst[off..off + row_elems]),
                                W_4K,
                            )
                            .unwrap();
                    }
                    black_box(dst[total_elems - 1])
                })
            });
        }
    });
}

fn bench_4k_rgb_f32(
    suite: &mut zenbench::Suite,
    label: &str,
    icc: Arc<[u8]>,
    native_src: Option<(Primaries, Transfer)>,
) {
    let layout = SampleLayout::RGBF32;
    let source_format = icc_spec(layout, Arc::clone(&icc), AlphaMode::None);
    let destination_format = rgb_spec(layout, Primaries::Bt709, Transfer::Srgb, AlphaMode::None);
    let row_elems = W_4K as usize * 3;
    let total_elems = row_elems * H_4K as usize;
    let total_bytes = total_elems * 4;

    let lcms_t = Lcms
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let mox_t = MoxCms
        .build_transform(&source_format, &destination_format)
        .unwrap();
    let src = Arc::new(fill_f32(total_elems));

    suite.compare(label, |g| {
        g.throughput(Throughput::Bytes((total_bytes * 2) as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(6)
            .max_rounds(6)
            .max_time(Duration::from_millis(1200));

        for (name, t) in [("lcms_main_opt", lcms_t), ("moxcms_0.9.2", mox_t)] {
            let s = Arc::clone(&src);
            let mut dst = vec![0.0f32; total_elems];
            g.bench(name, move |b| {
                b.iter(|| {
                    for y in 0..H_4K as usize {
                        let off = y * row_elems;
                        t.transform_row(
                            bytemuck::cast_slice(&s[off..off + row_elems]),
                            bytemuck::cast_slice_mut(&mut dst[off..off + row_elems]),
                            W_4K,
                        )
                        .unwrap();
                    }
                    black_box(dst[total_elems - 1])
                })
            });
        }
        if let Some((prim, tf)) = native_src {
            let nat_src = rgb_spec(layout, prim, tf, AlphaMode::None);
            let plan_simd = RowPlan::build(
                nat_src,
                destination_format,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap();
            let s = Arc::clone(&src);
            let mut dst = vec![0.0f32; total_elems];
            g.bench("native_05_ALL", move |b| {
                b.iter(|| {
                    for y in 0..H_4K {
                        let off = y as usize * row_elems;
                        plan_simd
                            .convert_row(
                                bytemuck::cast_slice(&s[off..off + row_elems]),
                                bytemuck::cast_slice_mut(&mut dst[off..off + row_elems]),
                                W_4K,
                            )
                            .unwrap();
                    }
                    black_box(dst[total_elems - 1])
                })
            });
        }
    });
}

fn bench_cmyk_rows(suite: &mut zenbench::Suite, cmyk_icc: Arc<[u8]>) {
    // 4096-px single-row CMYK8 -> RGB8 (comparable to benchmarks/cmyk8_bpc_2026-11-05.md!)
    let cmyk8 = PixelFormat::new(
        SampleLayout::CMYK8,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&cmyk_icc)),
            inverted: false,
            alpha: AlphaMode::None,
        },
    );
    let cmyk8_inv = PixelFormat::new(
        SampleLayout::CMYK8_INVERTED,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&cmyk_icc)),
            inverted: true,
            alpha: AlphaMode::None,
        },
    );
    let rgb8 = PixelFormat::RGB8_SRGB;

    let s_row = Arc::new(fill_u8(ROW_W as usize * 4));
    suite.compare("cmyk8_to_rgb8_4096px", |g| {
        g.throughput(Throughput::Elements(ROW_W as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(20)
            .max_rounds(20)
            .max_time(Duration::from_millis(800));

        let lcms_t = Lcms.build_transform(&cmyk8, &rgb8).unwrap();
        let lcms_inv = Lcms.build_transform(&cmyk8_inv, &rgb8).unwrap();
        let lcms_unopt = LcmsUnoptimized.build_transform(&cmyk8, &rgb8).unwrap();
        let mox_t = MoxCms.build_transform(&cmyk8, &rgb8).unwrap();
        let mox_inv = MoxCms.build_transform(&cmyk8_inv, &rgb8).unwrap();

        for (name, t) in [
            ("lcms_main_opt", lcms_t),
            ("lcms_inv_opt", lcms_inv),
            ("lcms_unopt", lcms_unopt),
            ("moxcms_0.9.2", mox_t),
            ("moxcms_inv", mox_inv),
        ] {
            let s = Arc::clone(&s_row);
            let mut d = vec![0u8; ROW_W as usize * 3];
            g.bench(name, move |b| {
                b.iter(|| {
                    t.transform_row(&s, &mut d, ROW_W).unwrap();
                    black_box(d[d.len() - 1])
                })
            });
        }
    });

    // 4096-px CMYK16 -> RGB16
    let cmyk16 = PixelFormat::new(
        SampleLayout::new(SampleType::U16, zenpixels_convert::ChannelOrder::Cmyk),
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&cmyk_icc)),
            inverted: false,
            alpha: AlphaMode::None,
        },
    );
    let rgb16 = PixelFormat::RGB16_SRGB;
    let s16 = Arc::new(fill_u16(ROW_W as usize * 4));
    suite.compare("cmyk16_to_rgb16_4096px", |g| {
        g.throughput(Throughput::Elements(ROW_W as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(20)
            .max_rounds(20)
            .max_time(Duration::from_millis(800));
        let lcms_t = Lcms.build_transform(&cmyk16, &rgb16).unwrap();
        let mox_t = MoxCms.build_transform(&cmyk16, &rgb16).unwrap();
        for (name, t) in [("lcms_main_opt", lcms_t), ("moxcms_0.9.2", mox_t)] {
            let s = Arc::clone(&s16);
            let mut d = vec![0u16; ROW_W as usize * 3];
            g.bench(name, move |b| {
                b.iter(|| {
                    t.transform_row(
                        bytemuck::cast_slice(&s),
                        bytemuck::cast_slice_mut(&mut d),
                        ROW_W,
                    )
                    .unwrap();
                    black_box(d[d.len() - 1])
                })
            });
        }
    });

    // 4096-px CMYKF32 -> RGBF32
    let cmykf = PixelFormat::new(
        SampleLayout::new(SampleType::F32, zenpixels_convert::ChannelOrder::Cmyk),
        ColorSpace::Cmyk {
            icc: Some(cmyk_icc),
            inverted: false,
            alpha: AlphaMode::None,
        },
    );
    let rgbf = rgb_spec(
        SampleLayout::RGBF32,
        Primaries::Bt709,
        Transfer::Srgb,
        AlphaMode::None,
    );
    let sf = Arc::new(fill_f32(ROW_W as usize * 4));
    suite.compare("cmykf32_to_rgbf32_4096px", |g| {
        g.throughput(Throughput::Elements(ROW_W as u64));
        g.config()
            .auto_rounds(false)
            .min_rounds(20)
            .max_rounds(20)
            .max_time(Duration::from_millis(800));
        let lcms_t = Lcms.build_transform(&cmykf, &rgbf).unwrap();
        let mox_t = MoxCms.build_transform(&cmykf, &rgbf).unwrap();
        for (name, t) in [("lcms_main_opt", lcms_t), ("moxcms_0.9.2", mox_t)] {
            let s = Arc::clone(&sf);
            let mut d = vec![0.0f32; ROW_W as usize * 3];
            g.bench(name, move |b| {
                b.iter(|| {
                    t.transform_row(
                        bytemuck::cast_slice(&s),
                        bytemuck::cast_slice_mut(&mut d),
                        ROW_W,
                    )
                    .unwrap();
                    black_box(d[d.len() - 1])
                })
            });
        }
    });
}

fn main() {
    let p3_icc: Arc<[u8]> = Arc::from(icc_profiles::DISPLAY_P3_V4);
    let adobe_icc: Arc<[u8]> = Arc::from(icc_profiles::ADOBE_RGB);
    let rgb_lut_icc: Arc<[u8]> = Arc::from(synthetic_rgb_lut_icc());
    let cmyk_icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());

    zenbench::run(|suite| {
        // 1. Display P3 ICC -> sRGB (4K UHD)
        bench_4k_rgb_u8(
            suite,
            "4k_p3_icc_to_srgb_rgb8",
            Arc::clone(&p3_icc),
            SampleLayout::RGB8,
            AlphaMode::None,
            Some((Primaries::DisplayP3, Transfer::Srgb)),
        );
        bench_4k_rgb_u8(
            suite,
            "4k_p3_icc_to_srgb_rgba8",
            Arc::clone(&p3_icc),
            SampleLayout::RGBA8,
            AlphaMode::Straight,
            Some((Primaries::DisplayP3, Transfer::Srgb)),
        );
        bench_4k_rgb_u16(
            suite,
            "4k_p3_icc_to_srgb_rgb16",
            Arc::clone(&p3_icc),
            Some((Primaries::DisplayP3, Transfer::Srgb)),
        );
        bench_4k_rgb_f32(
            suite,
            "4k_p3_icc_to_srgb_rgbf32",
            p3_icc,
            Some((Primaries::DisplayP3, Transfer::Srgb)),
        );

        // 2. Adobe RGB ICC -> sRGB (4K UHD)
        bench_4k_rgb_u8(
            suite,
            "4k_adobe_icc_to_srgb_rgb8",
            Arc::clone(&adobe_icc),
            SampleLayout::RGB8,
            AlphaMode::None,
            Some((Primaries::AdobeRgb, Transfer::Gamma22)),
        );
        bench_4k_rgb_f32(
            suite,
            "4k_adobe_icc_to_srgb_rgbf32",
            adobe_icc,
            Some((Primaries::AdobeRgb, Transfer::Gamma22)),
        );

        // 3. 3D LUT ICC (17^3 mft2) -> sRGB (4K UHD)
        bench_4k_rgb_u8(
            suite,
            "4k_lut3d_icc_to_srgb_rgb8",
            Arc::clone(&rgb_lut_icc),
            SampleLayout::RGB8,
            AlphaMode::None,
            None,
        );
        bench_4k_rgb_u16(
            suite,
            "4k_lut3d_icc_to_srgb_rgb16",
            Arc::clone(&rgb_lut_icc),
            None,
        );
        bench_4k_rgb_f32(suite, "4k_lut3d_icc_to_srgb_rgbf32", rgb_lut_icc, None);

        // 4. CMYK 4D CLUT ICC -> sRGB (4096-px rows)
        bench_cmyk_rows(suite, cmyk_icc);
    });
}
