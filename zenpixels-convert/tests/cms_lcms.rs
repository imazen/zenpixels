#![cfg(feature = "cms-lcms")]
//! Integration and parity tests for the pure-Rust `imazen/lcms` (`Lcms`) backend.

extern crate alloc;

use alloc::sync::Arc;
use zenpixels_convert::{
    AlphaMode, CmsBackend, ColorSpace, ConvertPolicy, Lcms, LcmsUnoptimized, PixelFormat,
    Primaries, RenderingIntent, RowPlan, SampleLayout, SignalRange, Transfer, icc_profiles,
};

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

/// Build an RGB CLUT (`A2B0` `mft2`) ICC profile approximating sRGB -> Lab.
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

fn icc_spec(
    layout: SampleLayout,
    profile: Arc<[u8]>,
    channels: u8,
    alpha: AlphaMode,
) -> PixelFormat {
    PixelFormat::new(
        layout,
        ColorSpace::Icc {
            profile,
            channels,
            alpha,
            intent: RenderingIntent::RelativeColorimetric,
        },
    )
}

#[test]
fn lcms_p3_icc_to_srgb_rgb8_rgb16_rgbf32_zero_alloc() {
    let p3_icc: Arc<[u8]> = Arc::from(icc_profiles::DISPLAY_P3_V4);
    let policy = ConvertPolicy::permissive().with_cms(Arc::new(Lcms));

    // RGB8
    let plan8 = RowPlan::build(
        icc_spec(SampleLayout::RGB8, Arc::clone(&p3_icc), 3, AlphaMode::None),
        PixelFormat::RGB8_SRGB,
        &policy,
        None,
    )
    .unwrap();
    let src8 = [255u8, 255, 255, 0, 0, 0, 128, 128, 128, 255, 0, 0];
    let mut dst8 = [0u8; 12];
    let allocs = allocation_counter::measure(|| {
        plan8.convert_row(&src8, &mut dst8, 4, 0).unwrap();
    });
    assert_eq!(allocs.count_total, 0, "Lcms RGB8 row must not allocate");
    assert_eq!(&dst8[0..3], &[255, 255, 255]);
    assert_eq!(&dst8[3..6], &[0, 0, 0]);
    assert!((dst8[6] as i32 - 128).abs() <= 1);
    assert_eq!(dst8[9], 255); // P3 pure red clips to sRGB R=255

    // RGB16
    let plan16 = RowPlan::build(
        icc_spec(SampleLayout::RGB16, Arc::clone(&p3_icc), 3, AlphaMode::None),
        PixelFormat::RGB16_SRGB,
        &policy,
        None,
    )
    .unwrap();
    let src16 = [65535u16, 65535, 65535, 0, 0, 0, 32768, 32768, 32768];
    let mut dst16 = [0u16; 9];
    let allocs16 = allocation_counter::measure(|| {
        plan16
            .convert_row(
                bytemuck::cast_slice(&src16),
                bytemuck::cast_slice_mut(&mut dst16),
                3,
                0,
            )
            .unwrap();
    });
    assert_eq!(allocs16.count_total, 0, "Lcms RGB16 row must not allocate");
    assert_eq!(&dst16[0..3], &[65535, 65535, 65535]);
    assert_eq!(&dst16[3..6], &[0, 0, 0]);
    assert!((dst16[6] as i32 - 32768).abs() <= 8);

    // RGBF32
    let srgb_f32 = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let plan_f32 = RowPlan::build(
        icc_spec(SampleLayout::RGBF32, p3_icc, 3, AlphaMode::None),
        srgb_f32,
        &policy,
        None,
    )
    .unwrap();
    let src_f = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
    let mut dst_f = [0.0f32; 9];
    let allocs_f = allocation_counter::measure(|| {
        plan_f32
            .convert_row(
                bytemuck::cast_slice(&src_f),
                bytemuck::cast_slice_mut(&mut dst_f),
                3,
                0,
            )
            .unwrap();
    });
    assert_eq!(allocs_f.count_total, 0, "Lcms RGBF32 row must not allocate");
    for c in 0..3 {
        assert!((dst_f[c] - 1.0).abs() < 1e-4, "white: {dst_f:?}");
        assert!(dst_f[3 + c].abs() < 1e-4, "black: {dst_f:?}");
        assert!((dst_f[6 + c] - 0.5).abs() < 2e-3, "midgray: {dst_f:?}");
    }
}

#[test]
fn lcms_preserves_alpha_exact_on_matrix_and_clut_profiles_including_f32() {
    let lut_icc: Arc<[u8]> = Arc::from(synthetic_rgb_lut_icc());
    let p3_icc: Arc<[u8]> = Arc::from(icc_profiles::DISPLAY_P3_V4);

    for icc in [p3_icc, lut_icc] {
        // RGBA8
        let t8 = Lcms
            .build_transform(
                &icc_spec(
                    SampleLayout::RGBA8,
                    Arc::clone(&icc),
                    3,
                    AlphaMode::Straight,
                ),
                &PixelFormat::RGBA8_SRGB,
            )
            .unwrap();
        let src8 = [200u8, 120, 40, 73, 10, 220, 90, 211];
        let mut dst8 = [0u8; 8];
        t8.transform_row(&src8, &mut dst8, 2).unwrap();
        assert_eq!(dst8[3], 73);
        assert_eq!(dst8[7], 211);

        // RGBA16
        let t16 = Lcms
            .build_transform(
                &icc_spec(
                    SampleLayout::RGBA16,
                    Arc::clone(&icc),
                    3,
                    AlphaMode::Straight,
                ),
                &PixelFormat::RGBA16_SRGB,
            )
            .unwrap();
        let src16 = [40000u16, 20000, 10000, 12345, 5000, 60000, 30000, 54321];
        let mut dst16 = [0u16; 8];
        t16.transform_row(
            bytemuck::cast_slice(&src16),
            bytemuck::cast_slice_mut(&mut dst16),
            2,
        )
        .unwrap();
        assert_eq!(dst16[3], 12345);
        assert_eq!(dst16[7], 54321);

        // RGBAF32 (exercises both matrix-shaper and ClutOnly RGBA_FLT alpha guard!)
        let dst_spec_f32 = PixelFormat::new(
            SampleLayout::RGBAF32,
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Straight,
                white: None,
            },
        );
        let tf = Lcms
            .build_transform(
                &icc_spec(SampleLayout::RGBAF32, icc, 3, AlphaMode::Straight),
                &dst_spec_f32,
            )
            .unwrap();
        let srcf = [0.7f32, 0.4, 0.2, 0.375, 0.1, 0.8, 0.5, 0.875];
        let mut dstf = [0.0f32; 8];
        tf.transform_row(
            bytemuck::cast_slice(&srcf),
            bytemuck::cast_slice_mut(&mut dstf),
            2,
        )
        .unwrap();
        assert_eq!(dstf[3].to_bits(), 0.375f32.to_bits());
        assert_eq!(dstf[7].to_bits(), 0.875f32.to_bits());
    }
}

#[test]
fn lcms_cmyk_u8_u16_f32_standard_and_inverted_zero_alloc() {
    let cmyk_icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());
    let policy = ConvertPolicy::permissive().with_cms(Arc::new(Lcms));

    // Standard CMYK8 -> RGBA8 (adds opaque alpha!)
    let cmyk8 = PixelFormat::new(
        SampleLayout::CMYK8,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&cmyk_icc)),
            inverted: false,
            alpha: AlphaMode::None,
        },
    );
    let plan8 = RowPlan::build(cmyk8, PixelFormat::RGBA8_SRGB, &policy, None).unwrap();
    // 0% ink = paper white; 100% K = black.
    let src8 = [0u8, 0, 0, 0, 0, 0, 0, 255];
    let mut dst8 = [0u8; 8];
    let allocs = allocation_counter::measure(|| {
        plan8.convert_row(&src8, &mut dst8, 2, 0).unwrap();
    });
    assert_eq!(allocs.count_total, 0);
    assert!(
        dst8[0] >= 254 && dst8[1] >= 254 && dst8[2] >= 254,
        "white: {dst8:?}"
    );
    assert_eq!(dst8[3], 255);
    assert!(
        dst8[4] <= 2 && dst8[5] <= 2 && dst8[6] <= 2,
        "black: {dst8:?}"
    );
    assert_eq!(dst8[7], 255);

    // Inverted CMYK8 (Adobe JPEG convention: 255 = no ink) -> RGB8 matches standard!
    let cmyk8_inv = PixelFormat::new(
        SampleLayout::CMYK8_INVERTED,
        ColorSpace::Cmyk {
            icc: Some(Arc::clone(&cmyk_icc)),
            inverted: true,
            alpha: AlphaMode::None,
        },
    );
    let plan8_inv = RowPlan::build(cmyk8_inv, PixelFormat::RGB8_SRGB, &policy, None).unwrap();
    let src8_inv = [255u8, 255, 255, 255, 255, 255, 255, 0];
    let mut dst8_inv = [0u8; 6];
    let allocs_inv = allocation_counter::measure(|| {
        plan8_inv
            .convert_row(&src8_inv, &mut dst8_inv, 2, 0)
            .unwrap();
    });
    assert_eq!(allocs_inv.count_total, 0);
    assert_eq!(&dst8_inv[0..3], &dst8[0..3]);
    assert_eq!(&dst8_inv[3..6], &dst8[4..7]);

    // Inverted CMYK F32 ([0, 1] -> percent inside Lcms!) -> RGBF32 zero-alloc!
    let cmyk_f32_inv = PixelFormat::new(
        SampleLayout::new(
            zenpixels_convert::SampleType::F32,
            zenpixels_convert::ChannelOrder::CmykInverted,
        ),
        ColorSpace::Cmyk {
            icc: Some(cmyk_icc),
            inverted: true,
            alpha: AlphaMode::None,
        },
    );
    let dst_rgbf = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let plan_f = RowPlan::build(cmyk_f32_inv, dst_rgbf, &policy, None).unwrap();
    let src_f = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let mut dst_f = [0.0f32; 6];
    let allocs_f = allocation_counter::measure(|| {
        plan_f
            .convert_row(
                bytemuck::cast_slice(&src_f),
                bytemuck::cast_slice_mut(&mut dst_f),
                2,
                0,
            )
            .unwrap();
    });
    assert_eq!(allocs_f.count_total, 0);
    assert!(
        dst_f[0] > 0.99 && dst_f[1] > 0.99 && dst_f[2] > 0.99,
        "{dst_f:?}"
    );
    assert!(
        dst_f[3] < 0.01 && dst_f[4] < 0.01 && dst_f[5] < 0.01,
        "{dst_f:?}"
    );
}

#[test]
fn lcms_bgr_bgra_and_unoptimized_mode() {
    let p3_icc: Arc<[u8]> = Arc::from(icc_profiles::DISPLAY_P3_V4);
    let src = icc_spec(
        SampleLayout::RGBA8,
        Arc::clone(&p3_icc),
        3,
        AlphaMode::Straight,
    );
    let t_bgra = Lcms
        .build_transform(&src, &PixelFormat::BGRA8_SRGB)
        .unwrap();
    let t_rgba = Lcms
        .build_transform(&src, &PixelFormat::RGBA8_SRGB)
        .unwrap();
    let t_unopt = LcmsUnoptimized
        .build_transform(&src, &PixelFormat::RGBA8_SRGB)
        .unwrap();

    let inp = [200u8, 100, 40, 177, 30, 180, 240, 42];
    let mut out_bgra = [0u8; 8];
    let mut out_rgba = [0u8; 8];
    let mut out_unopt = [0u8; 8];
    t_bgra.transform_row(&inp, &mut out_bgra, 2).unwrap();
    t_rgba.transform_row(&inp, &mut out_rgba, 2).unwrap();
    t_unopt.transform_row(&inp, &mut out_unopt, 2).unwrap();

    for px in 0..2 {
        let i = px * 4;
        assert_eq!(out_bgra[i], out_rgba[i + 2]);
        assert_eq!(out_bgra[i + 1], out_rgba[i + 1]);
        assert_eq!(out_bgra[i + 2], out_rgba[i]);
        assert_eq!(out_bgra[i + 3], inp[i + 3]);
        for c in 0..3 {
            assert!(
                (out_unopt[i + c] as i32 - out_rgba[i + c] as i32).abs() <= 1,
                "opt vs unopt mismatch: {out_rgba:?} vs {out_unopt:?}"
            );
        }
        assert_eq!(out_unopt[i + 3], inp[i + 3]);
    }
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn lcms_and_moxcms_agree_on_p3_and_cmyk() {
    let p3_icc: Arc<[u8]> = Arc::from(icc_profiles::DISPLAY_P3_V4);
    let src8 = icc_spec(SampleLayout::RGB8, Arc::clone(&p3_icc), 3, AlphaMode::None);
    let dst8 = PixelFormat::RGB8_SRGB;

    let t_lcms = Lcms.build_transform(&src8, &dst8).unwrap();
    let t_mox = zenpixels_convert::MoxCms
        .build_transform(&src8, &dst8)
        .unwrap();

    let mut row = Vec::with_capacity(64 * 3);
    for i in 0..64u8 {
        row.push(i.wrapping_mul(4));
        row.push(i.wrapping_mul(7).wrapping_add(13));
        row.push(255 - i.wrapping_mul(3));
    }
    let mut out_l = vec![0u8; row.len()];
    let mut out_m = vec![0u8; row.len()];
    t_lcms.transform_row(&row, &mut out_l, 64).unwrap();
    t_mox.transform_row(&row, &mut out_m, 64).unwrap();

    let max_diff = out_l
        .iter()
        .zip(&out_m)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    assert!(
        max_diff <= 2,
        "Lcms vs MoxCms P3->sRGB RGB8 max diff {max_diff} > 2"
    );

    // CMYK8 synthetic profile parity.
    let cmyk_icc: Arc<[u8]> = Arc::from(synthetic_cmyk_icc());
    let cmyk_src = PixelFormat::new(
        SampleLayout::CMYK8,
        ColorSpace::Cmyk {
            icc: Some(cmyk_icc),
            inverted: false,
            alpha: AlphaMode::None,
        },
    );
    let cl = Lcms.build_transform(&cmyk_src, &dst8).unwrap();
    let cm = zenpixels_convert::MoxCms
        .build_transform(&cmyk_src, &dst8)
        .unwrap();
    let mut cmyk_row = Vec::with_capacity(64 * 4);
    for i in 0..64u8 {
        cmyk_row.push(i.wrapping_mul(4));
        cmyk_row.push(i.wrapping_mul(3));
        cmyk_row.push((63 - i).wrapping_mul(4));
        cmyk_row.push(i.wrapping_mul(2));
    }
    let mut cl_out = vec![0u8; 64 * 3];
    let mut cm_out = vec![0u8; 64 * 3];
    cl.transform_row(&cmyk_row, &mut cl_out, 64).unwrap();
    cm.transform_row(&cmyk_row, &mut cm_out, 64).unwrap();
    let cmyk_max_diff = cl_out
        .iter()
        .zip(&cm_out)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    assert!(
        cmyk_max_diff <= 4,
        "Lcms vs MoxCms CMYK8->RGB8 max diff {cmyk_max_diff} > 4"
    );
}
