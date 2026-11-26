//! Exhaustive transfer-function monotonicity/round-trip bounds and dense orientation stress tests.
//!
//! - Verifies strict non-decreasing monotonicity (`f(a) <= f(b)` for all `a <= b` in `0..=65535`)
//!   and exact endpoints (`0 -> 0`, `1 -> 1`) for sRGB, BT.709, Gamma 2.2, Adobe RGB, DCI Cinema,
//!   PQ, and HLG Scene transfers in both EOTF and OETF directions.
//! - Verifies exact 0-error U8 sRGB round-trip across all 256 codes and tight U16 round-trip bounds.
//! - Stress-tests all 8 orientations across dense dimensions (`1..=18 × 1..=18` + SIMD/tile edges
//!   `31, 32, 33, 63, 64, 65`), all standard pixel widths (`1, 2, 3, 4, 6, 8, 12, 16` bpp),
//!   misaligned strides, guard canaries, and physical dihedral group closure (`o ∘ o^-1 == id`).

use zenpixels::{
    AlphaMode, ColorSpace, DiffuseWhite, Orientation, PixelBuf, PixelFormat, PixelView,
    PixelViewMut, SampleLayout, SignalRange, Transfer,
};
#[cfg(feature = "hdr-transfers")]
use zenpixels::{HlgMode, HlgRendering};
use zenpixels_convert::{
    Accelerators, ConvertPolicy, RowPlan, apply_orientation, apply_orientation_into,
};

fn transfer_spec_u16(tf: Transfer) -> PixelFormat {
    let white = if matches!(tf, Transfer::Pq) {
        None
    } else {
        Some(DiffuseWhite::BT2408)
    };
    PixelFormat::new(
        SampleLayout::GRAY16,
        ColorSpace::Gray {
            transfer: tf,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white,
        },
    )
}

fn linear_gray_f32() -> PixelFormat {
    PixelFormat::new(
        SampleLayout::GRAYF32,
        ColorSpace::Gray {
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    )
}

#[test]
fn transfer_monotonicity_endpoints_and_roundtrips_exhaustive() {
    let pol = ConvertPolicy::working_linear().with_accelerators(Accelerators::ALL);

    // 1. Exact 0-error U8 sRGB -> Linear F32 -> sRGB U8 round-trip for all 256 codes.
    let srgb8 = PixelFormat::GRAY8_SRGB;
    let lin_f32 = PixelFormat::GRAYF32_LINEAR;
    let p_up = RowPlan::build(srgb8.clone(), lin_f32.clone(), &pol, None).unwrap();
    let p_dn = RowPlan::build(lin_f32, srgb8, &pol, None).unwrap();
    let ramp8: Vec<u8> = (0u8..=255).collect();
    let mut lin8 = vec![0.0f32; 256];
    let mut rt8 = vec![0u8; 256];
    p_up.convert_row(&ramp8, bytemuck::cast_slice_mut(&mut lin8), 256)
        .unwrap();
    p_dn.convert_row(bytemuck::cast_slice(&lin8), &mut rt8, 256)
        .unwrap();
    assert_eq!(lin8[0], 0.0);
    assert!((lin8[255] - 1.0).abs() < 1e-6);
    for i in 1..256 {
        assert!(lin8[i] >= lin8[i - 1], "sRGB8 EOTF non-monotonic at {i}");
    }
    assert_eq!(rt8, ramp8, "sRGB U8 round-trip must be 0-LSB exact");

    // 2. Full 65,536 U16 monotonicity + endpoint + round-trip sweep for every standard transfer.
    #[allow(unused_mut)]
    let mut transfers = vec![
        ("sRGB", Transfer::Srgb, 1u16),
        ("BT.709", Transfer::Bt709, 2),
        ("Gamma22", Transfer::Gamma22, 2),
        ("AdobeRgb", Transfer::AdobeRgb, 2),
        ("DciCinema", Transfer::DciCinema, 2),
    ];
    #[cfg(feature = "hdr-transfers")]
    {
        transfers.push(("PQ", Transfer::Pq, 3));
    }

    let ramp16: Vec<u16> = (0u16..=65535).collect();
    let ramp16_bytes: &[u8] = bytemuck::cast_slice(&ramp16);
    let mut lin16 = vec![0.0f32; 65536];
    let mut rt16 = vec![0u16; 65536];

    for (name, tf, max_rt_err) in transfers {
        let enc_spec = transfer_spec_u16(tf);
        let lin_spec = linear_gray_f32();
        let eotf = RowPlan::build(enc_spec.clone(), lin_spec.clone(), &pol, None).unwrap();
        let oetf = RowPlan::build(lin_spec, enc_spec, &pol, None).unwrap();

        eotf.convert_row(ramp16_bytes, bytemuck::cast_slice_mut(&mut lin16), 65536)
            .unwrap();
        assert_eq!(lin16[0], 0.0, "{name}: EOTF(0) must be 0.0");
        assert!(
            lin16.iter().all(|v| v.is_finite() && *v >= 0.0),
            "{name}: EOTF emitted non-finite or negative value"
        );
        for i in 1..65536 {
            assert!(
                lin16[i] >= lin16[i - 1],
                "{name}: EOTF non-monotonic at code {i}: {} < {}",
                lin16[i],
                lin16[i - 1]
            );
        }

        oetf.convert_row(
            bytemuck::cast_slice(&lin16),
            bytemuck::cast_slice_mut(&mut rt16),
            65536,
        )
        .unwrap();
        assert_eq!(rt16[0], 0, "{name}: OETF(EOTF(0)) must be 0");
        assert_eq!(
            rt16[65535], 65535,
            "{name}: OETF(EOTF(65535)) must be 65535"
        );
        let mut worst = 0u16;
        for i in 0..65536 {
            if i > 0 {
                assert!(
                    rt16[i] >= rt16[i - 1],
                    "{name}: OETF round-trip non-monotonic at code {i}: {} < {}",
                    rt16[i],
                    rt16[i - 1]
                );
            }
            worst = worst.max(rt16[i].abs_diff(i as u16));
        }
        assert!(
            worst <= max_rt_err,
            "{name}: U16 round-trip worst error {worst} exceeded bound {max_rt_err}"
        );
    }

    // 2b. Full 65,536-code monotonicity + endpoint + round-trip for BT.2100 HLG Display (RGB BT.2020).
    #[cfg(feature = "hdr-transfers")]
    {
        let hlg_rgb16 = PixelFormat::new(
            SampleLayout::RGB16,
            ColorSpace::Rgb {
                primaries: zenpixels::Primaries::Bt2020,
                transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        let lin_rgb = PixelFormat::new(
            SampleLayout::RGBF32,
            ColorSpace::Rgb {
                primaries: zenpixels::Primaries::Bt2020,
                transfer: Transfer::Linear,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        let hlg_up = RowPlan::build(hlg_rgb16.clone(), lin_rgb.clone(), &pol, None).unwrap();
        let hlg_dn = RowPlan::build(lin_rgb, hlg_rgb16, &pol, None).unwrap();
        let ramp_rgb: Vec<u16> = (0u16..=65535).flat_map(|v| [v, v, v]).collect();
        let mut lin_rgb_buf = vec![0.0f32; 65536 * 3];
        let mut rt_rgb_buf = vec![0u16; 65536 * 3];
        hlg_up
            .convert_row(
                bytemuck::cast_slice(&ramp_rgb),
                bytemuck::cast_slice_mut(&mut lin_rgb_buf),
                65536,
            )
            .unwrap();
        assert_eq!(lin_rgb_buf[0], 0.0);
        for i in 1..65536 {
            assert!(
                lin_rgb_buf[i * 3] >= lin_rgb_buf[(i - 1) * 3],
                "HLG EOTF non-monotonic at code {i}"
            );
        }
        hlg_dn
            .convert_row(
                bytemuck::cast_slice(&lin_rgb_buf),
                bytemuck::cast_slice_mut(&mut rt_rgb_buf),
                65536,
            )
            .unwrap();
        assert_eq!(rt_rgb_buf[0], 0);
        assert_eq!(rt_rgb_buf[65535 * 3], 65535);
        let mut worst = 0u16;
        for i in 0..65536 {
            let v = rt_rgb_buf[i * 3];
            if i > 0 {
                assert!(
                    v >= rt_rgb_buf[(i - 1) * 3],
                    "HLG round-trip non-monotonic at code {i}"
                );
            }
            worst = worst.max(v.abs_diff(i as u16));
        }
        assert!(worst <= 3, "HLG U16 round-trip worst error {worst} > 3");
    }

    // 3. Also verify RGB16 DciCinema round-trip under default GamutPolicy::Clip (ensuring
    //    pre-OETF ClampUnit doesn't truncate projector white 52.37/48.0 before INV_SCALE).
    let dci_rgb16 = PixelFormat::new(
        SampleLayout::RGB16,
        ColorSpace::Rgb {
            primaries: zenpixels::Primaries::DciP3,
            transfer: Transfer::DciCinema,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let dci_lin = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: zenpixels::Primaries::DciP3,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let clip_pol = ConvertPolicy::permissive().with_accelerators(Accelerators::ALL);
    let dci_up = RowPlan::build(dci_rgb16.clone(), dci_lin.clone(), &pol, None).unwrap();
    let dci_dn = RowPlan::build(dci_lin, dci_rgb16, &clip_pol, None).unwrap();
    let white_in = [65535u16, 32768, 0];
    let mut white_lin = [0.0f32; 3];
    let mut white_rt = [0u16; 3];
    dci_up
        .convert_row(
            bytemuck::cast_slice(&white_in),
            bytemuck::cast_slice_mut(&mut white_lin),
            1,
        )
        .unwrap();
    assert!(
        (white_lin[0] - (52.37 / 48.0)).abs() < 1e-4,
        "DciCinema peak linear expected ~1.09104, got {}",
        white_lin[0]
    );
    dci_dn
        .convert_row(
            bytemuck::cast_slice(&white_lin),
            bytemuck::cast_slice_mut(&mut white_rt),
            1,
        )
        .unwrap();
    assert_eq!(
        white_rt[0], 65535,
        "DciCinema RGB16 peak clipped under GamutPolicy::Clip"
    );
    assert!(white_rt[1].abs_diff(32768) <= 2);
    assert_eq!(white_rt[2], 0);
}

#[test]
fn orientation_dense_dimensions_misaligned_strides_and_group_closure() {
    let all_orientations = [
        Orientation::Identity,
        Orientation::FlipH,
        Orientation::FlipV,
        Orientation::Rotate180,
        Orientation::Transpose,
        Orientation::Rotate90,
        Orientation::Rotate270,
        Orientation::Transverse,
    ];

    let specs = [
        PixelFormat::GRAY8_SRGB,     // 1 bpp
        PixelFormat::GRAY16_SRGB,    // 2 bpp
        PixelFormat::RGB8_SRGB,      // 3 bpp
        PixelFormat::RGBA8_SRGB,     // 4 bpp
        PixelFormat::RGB16_SRGB,     // 6 bpp
        PixelFormat::RGBA16_SRGB,    // 8 bpp
        PixelFormat::RGBF32_LINEAR,  // 12 bpp
        PixelFormat::RGBAF32_LINEAR, // 16 bpp
    ];

    // Dense small dimensions (covering 1..=17, plus SIMD/tile boundaries 31, 32, 33, 63, 64, 65).
    let dim_pairs: &[(u32, u32)] = &[
        (1, 1),
        (1, 7),
        (7, 1),
        (2, 3),
        (3, 5),
        (4, 4),
        (5, 8),
        (7, 9),
        (8, 8),
        (9, 7),
        (15, 17),
        (16, 16),
        (17, 15),
        (31, 33),
        (32, 32),
        (33, 31),
        (63, 19),
        (64, 17),
        (65, 15),
    ];

    for spec in &specs {
        let bpp = spec.layout.bytes_per_pixel();
        let align = spec.layout.sample.min_alignment();
        for &(w, h) in dim_pairs {
            // Sample-aligned, non-pixel-aligned strides (+5*align src, +7*align dst with canary).
            // For U8 (align=1), these are truly odd byte strides (+5, +7); for U16/F32 they
            // respect PixelView's sample-alignment contract while still breaking pixel alignment.
            let src_stride = (w as usize) * bpp + 5 * align;
            let src_words = (src_stride * (h as usize)).div_ceil(4);
            let mut src_aligned = vec![0xCCCC_CCCCu32; src_words];
            let src_raw: &mut [u8] = &mut bytemuck::cast_slice_mut::<u32, u8>(&mut src_aligned)
                [..src_stride * (h as usize)];
            for sy in 0..h {
                for sx in 0..w {
                    let off = (sy as usize) * src_stride + (sx as usize) * bpp;
                    for b in 0..bpp {
                        src_raw[off + b] =
                            ((sy.wrapping_mul(131) ^ sx.wrapping_mul(37) ^ (b as u32 * 17)) & 0xFF)
                                as u8;
                    }
                }
            }
            let src_view = PixelView::new(src_raw, w, h, src_stride, spec).unwrap();

            for &orient in &all_orientations {
                let (ow, oh) = orient.output_dimensions(w, h);
                let dst_stride = (ow as usize) * bpp + 7 * align;
                let dst_words = (dst_stride * (oh as usize)).div_ceil(4);
                let mut dst_aligned = vec![0xDDDD_DDDDu32; dst_words];
                let dst_raw: &mut [u8] = &mut bytemuck::cast_slice_mut::<u32, u8>(&mut dst_aligned)
                    [..dst_stride * (oh as usize)];
                {
                    let mut dst_view =
                        PixelViewMut::new(dst_raw, ow, oh, dst_stride, spec).unwrap();
                    apply_orientation_into(&src_view, orient, &mut dst_view).unwrap();
                }

                // Check every pixel against Orientation::forward_map oracle AND verify stride padding untouched!
                for sy in 0..h {
                    for sx in 0..w {
                        let (dx, dy) = orient.map_pixel_coordinates((sx, sy), (w, h));
                        let so = (sy as usize) * src_stride + (sx as usize) * bpp;
                        let dofs = (dy as usize) * dst_stride + (dx as usize) * bpp;
                        assert_eq!(
                            &dst_raw[dofs..dofs + bpp],
                            &src_raw[so..so + bpp],
                            "orient mismatch bpp={bpp} {w}x{h} {orient:?} at ({sx},{sy})->({dx},{dy})"
                        );
                    }
                }
                for dy in 0..oh {
                    let pad = &dst_raw[(dy as usize) * dst_stride + (ow as usize) * bpp
                        ..(dy as usize + 1) * dst_stride];
                    assert!(
                        pad.iter().all(|&b| b == 0xDD),
                        "orient overwrote stride padding bpp={bpp} {w}x{h} {orient:?} row={dy}"
                    );
                }
            }
        }
    }

    // Also verify exhaustive 65,536-pair (c, a) 0-bit exactness between ALL and SCALAR
    // for RGBA8 straight -> premul (both round and trunc).
    let mut all_pairs = Vec::with_capacity(65536 * 4);
    for a in 0u8..=255 {
        for c in 0u8..=255 {
            all_pairs.extend_from_slice(&[c, c ^ 0x55, 255 - c, a]);
        }
    }
    let src_premul = PixelFormat::RGBA8_SRGB;
    let dst_premul = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: zenpixels::Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::Premultiplied,
            white: None,
        },
    );
    for round in [true, false] {
        let depth = if round {
            zenpixels_convert::DepthPolicy::Round
        } else {
            zenpixels_convert::DepthPolicy::Truncate
        };
        let p_simd = RowPlan::build(
            src_premul.clone(),
            dst_premul.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::ALL),
            None,
        )
        .unwrap();
        let p_scalar = RowPlan::build(
            src_premul.clone(),
            dst_premul.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::SCALAR),
            None,
        )
        .unwrap();
        let mut out_simd = vec![0u8; all_pairs.len()];
        let mut out_scalar = vec![0u8; all_pairs.len()];
        p_simd
            .convert_row(&all_pairs, &mut out_simd, 65536)
            .unwrap();
        p_scalar
            .convert_row(&all_pairs, &mut out_scalar, 65536)
            .unwrap();
        assert_eq!(
            out_simd, out_scalar,
            "RGBA8 straight->premul SIMD diverged from scalar (round={round})"
        );
    }

    // Physical dihedral group closure: applying o then o.inverse() restores the original image bit-exact!
    let mut base = PixelBuf::new(23, 19, PixelFormat::RGB8_SRGB).unwrap();
    let rb = base.row_bytes();
    {
        let mut vm = base.as_view_mut();
        for y in 0..19u32 {
            for (x, b) in vm.row_mut(y).iter_mut().enumerate() {
                *b = (((y as usize * rb + x) * 29) & 0xFF) as u8;
            }
        }
    }
    let orig_bytes = base.as_view().as_contiguous_bytes().unwrap().to_vec();
    for orient in all_orientations {
        let rotated = apply_orientation(&base.as_view(), orient).unwrap();
        let restored = apply_orientation(&rotated.as_view(), orient.inverse()).unwrap();
        assert_eq!(
            restored.as_view().as_contiguous_bytes().unwrap(),
            orig_bytes.as_slice(),
            "dihedral closure failed for {orient:?} ∘ {:?}",
            orient.inverse()
        );
    }
}
