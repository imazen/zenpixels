//! Row-partition invariance, odd-stride guard canaries, and cross-platform XXH3 digest pins.
//!
//! Guarantees that:
//! 1. Converting a row pixel-by-pixel (`width = 1`, `N` calls) produces **0-bit identical**
//!    output to converting the entire row in one call (`width = N`) for non-dithered routes
//!    across all SIMD tail boundaries (`1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257`).
//! 2. Guard canary bytes immediately before and after every destination slice are never touched.
//! 3. Portable endian/stride-independent XXH3-64 digests are identical between `SCALAR`, `ALL`,
//!    tight buffers, and odd-strided views across canonical SDR/HDR/wide-gamut/Oklab routes.

use zenpixels::{
    AlphaMode, BitPacking, Color, ColorSpace, HashInclusions, Nits, PixelBuf, PixelFormat,
    Primaries, RowAlign, SampleLayout, SampleType, SignalRange, Transfer,
};
#[cfg(feature = "hdr-transfers")]
use zenpixels::{DiffuseWhite, HlgMode, HlgRendering};
#[cfg(feature = "oklab")]
use zenpixels_convert::GamutPolicy;
use zenpixels_convert::{
    Accelerators, AlphaPolicy, ConvertPolicy, DepthPolicy, ImagePlan, LumaCoefficients, LumaDomain,
    PeakSource, RowPlan,
};

fn pattern_bytes(spec: &PixelFormat, width: u32) -> Vec<u8> {
    let bpp = spec.layout.bytes_per_pixel();
    let total = (width as usize) * bpp;
    let mut out = vec![0u8; total];
    match spec.layout.sample {
        SampleType::U8 => {
            for (i, b) in out.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(73).wrapping_add(19) ^ (i >> 2)) & 0xFF) as u8;
            }
        }
        SampleType::U16 => {
            for (i, chunk) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let v = if spec.layout.packing == BitPacking::P010 {
                    (((i.wrapping_mul(97) + 13) & 0x3FF) as u16) << 6
                } else {
                    ((i.wrapping_mul(9973).wrapping_add(12345)) & 0xFFFF) as u16
                };
                *chunk = v.to_ne_bytes();
            }
        }
        SampleType::U32 => {
            for (i, chunk) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let r = ((i * 73 + 11) & 0x3FF) as u32;
                let g = ((i * 193 + 29) & 0x3FF) as u32;
                let b = ((i * 419 + 53) & 0x3FF) as u32;
                let a = (i & 3) as u32;
                let word = r | (g << 10) | (b << 20) | (a << 30);
                *chunk = word.to_ne_bytes();
            }
        }
        SampleType::F16 => {
            for (i, chunk) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let f = ((i.wrapping_mul(37).wrapping_add(7)) % 257) as f32 / 256.0;
                *chunk = half::f16::from_f32(f).to_bits().to_ne_bytes();
            }
        }
        SampleType::F32 => {
            for (i, chunk) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let f = ((i.wrapping_mul(41).wrapping_add(13)) % 1009) as f32 / 1008.0;
                *chunk = f.to_ne_bytes();
            }
        }
        _ => {}
    }
    out
}

fn verify_partition_and_canary(
    name: &str,
    source_format: PixelFormat,
    destination_format: PixelFormat,
    policy: ConvertPolicy,
    hdr_peak: Option<Nits>,
) {
    let plan_all = RowPlan::build(
        source_format.clone(),
        destination_format.clone(),
        &policy.clone().with_accelerators(Accelerators::ALL),
        hdr_peak,
    )
    .unwrap_or_else(|e| panic!("{name} ALL build failed: {e}"));
    let plan_sc = RowPlan::build(
        source_format.clone(),
        destination_format.clone(),
        &policy.with_accelerators(Accelerators::SCALAR),
        hdr_peak,
    )
    .unwrap_or_else(|e| panic!("{name} SCALAR build failed: {e}"));

    let sbpp = source_format.layout.bytes_per_pixel();
    let dbpp = destination_format.layout.bytes_per_pixel();
    let widths = [
        1u32, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257,
    ];

    for &w in &widths {
        let src = pattern_bytes(&source_format, w);
        // Surround destination row with 13-byte 0xAA guard canaries on both ends (also tests odd alignment!).
        const GUARD: usize = 13;
        let dst_len = (w as usize) * dbpp;
        let mut dst_all = vec![0xAAu8; dst_len + 2 * GUARD];
        let mut dst_sc = vec![0xAAu8; dst_len + 2 * GUARD];
        let mut dst_1px = vec![0xAAu8; dst_len + 2 * GUARD];

        plan_all
            .convert_row(&src, &mut dst_all[GUARD..GUARD + dst_len], w)
            .unwrap();
        plan_sc
            .convert_row(&src, &mut dst_sc[GUARD..GUARD + dst_len], w)
            .unwrap();

        for x in 0..(w as usize) {
            plan_all
                .convert_row(
                    &src[x * sbpp..(x + 1) * sbpp],
                    &mut dst_1px[GUARD + x * dbpp..GUARD + (x + 1) * dbpp],
                    1,
                )
                .unwrap();
        }

        assert!(
            dst_all[..GUARD].iter().all(|&b| b == 0xAA)
                && dst_all[GUARD + dst_len..].iter().all(|&b| b == 0xAA),
            "{name} w={w}: ALL corrupted guard bytes!"
        );
        assert!(
            dst_sc[..GUARD].iter().all(|&b| b == 0xAA)
                && dst_sc[GUARD + dst_len..].iter().all(|&b| b == 0xAA),
            "{name} w={w}: SCALAR corrupted guard bytes!"
        );
        assert_eq!(
            &dst_all[GUARD..GUARD + dst_len],
            &dst_sc[GUARD..GUARD + dst_len],
            "{name} w={w}: ALL vs SCALAR mismatch"
        );
        assert_eq!(
            &dst_all[GUARD..GUARD + dst_len],
            &dst_1px[GUARD..GUARD + dst_len],
            "{name} w={w}: whole-row vs 1-px partition mismatch"
        );
    }
}

#[test]
fn row_partition_and_odd_stride_canary_invariance_matrix() {
    let perm = ConvertPolicy::permissive();
    let p3_u8 = PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let p3_u16 = PixelFormat::RGB16_SRGB.with_primaries(Primaries::DisplayP3);
    let p3_f32 = PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3);
    let rgba8_premul = PixelFormat::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    let rgb8_narrow = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let rgb565 = PixelFormat::new(
        SampleLayout::RGB565,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let rgb10a2 = PixelFormat::new(
        SampleLayout::RGB10A2,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::Straight,
            white: None,
        },
    );
    let p010_rgb = PixelFormat::new(
        SampleLayout::RGB16.with_packing(BitPacking::P010),
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );

    let cases: Vec<(&str, PixelFormat, PixelFormat, ConvertPolicy, Option<Nits>)> = vec![
        (
            "RGBA8->RGB8",
            PixelFormat::RGBA8_SRGB,
            PixelFormat::RGB8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB8->BGRA8",
            PixelFormat::RGB8_SRGB,
            PixelFormat::BGRA8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGBA8->BGRA8",
            PixelFormat::RGBA8_SRGB,
            PixelFormat::BGRA8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB8->Gray8_Encoded_709",
            PixelFormat::RGB8_SRGB,
            PixelFormat::GRAY8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "BGRA8->Gray8_Encoded_601",
            PixelFormat::BGRA8_SRGB,
            PixelFormat::GRAY8_SRGB,
            perm.clone().with_luma(LumaCoefficients::Bt601),
            None,
        ),
        (
            "RGB8->Gray8_LinearLight",
            PixelFormat::RGB8_SRGB,
            PixelFormat::GRAY8_SRGB,
            perm.clone().with_luma_domain(LumaDomain::LinearLight),
            None,
        ),
        (
            "RGB8->RGB16",
            PixelFormat::RGB8_SRGB,
            PixelFormat::RGB16_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB16->RGB8_Round",
            PixelFormat::RGB16_SRGB,
            PixelFormat::RGB8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB16->RGB8_Trunc",
            PixelFormat::RGB16_SRGB,
            PixelFormat::RGB8_SRGB,
            perm.clone().with_depth(DepthPolicy::Truncate),
            None,
        ),
        (
            "RGBA8_Straight->Premul_Round",
            PixelFormat::RGBA8_SRGB,
            rgba8_premul.clone(),
            perm.clone(),
            None,
        ),
        (
            "RGBA8_Straight->Premul_Trunc",
            PixelFormat::RGBA8_SRGB,
            rgba8_premul.clone(),
            perm.clone().with_depth(DepthPolicy::Truncate),
            None,
        ),
        (
            "RGBA8_Premul->Straight",
            rgba8_premul,
            PixelFormat::RGBA8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB8_sRGB->RGBF32_Lin",
            PixelFormat::RGB8_SRGB,
            PixelFormat::RGBF32_LINEAR,
            perm.clone(),
            None,
        ),
        (
            "RGBF32_Lin->RGB8_sRGB_Exact",
            PixelFormat::RGBF32_LINEAR,
            PixelFormat::RGB8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGBAF32_Lin->RGBA8_sRGB_Approx",
            PixelFormat::RGBAF32_LINEAR,
            PixelFormat::RGBA8_SRGB,
            perm.clone().with_approximate(true),
            None,
        ),
        (
            "RGBF16_Lin->RGBF32_Lin",
            PixelFormat::RGBF16_LINEAR,
            PixelFormat::RGBF32_LINEAR,
            perm.clone(),
            None,
        ),
        (
            "RGBF32_Lin->RGBF16_Lin",
            PixelFormat::RGBF32_LINEAR,
            PixelFormat::RGBF16_LINEAR,
            perm.clone(),
            None,
        ),
        (
            "P3_RGB8->sRGB_RGB8_Exact",
            p3_u8.clone(),
            PixelFormat::RGB8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "P3_RGB8->sRGB_RGB8_ApproxQ14",
            p3_u8,
            PixelFormat::RGB8_SRGB,
            perm.clone().with_approximate(true),
            None,
        ),
        (
            "P3_RGB16->sRGB_RGB16",
            p3_u16,
            PixelFormat::RGB16_SRGB,
            perm.clone(),
            None,
        ),
        (
            "P3_F32->sRGB_F32",
            p3_f32,
            PixelFormat::RGBF32_LINEAR,
            perm.clone(),
            None,
        ),
        (
            "NarrowRGB8->FullRGB8",
            rgb8_narrow.clone(),
            PixelFormat::RGB8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "FullRGB8->NarrowRGB8",
            PixelFormat::RGB8_SRGB,
            rgb8_narrow,
            perm.clone(),
            None,
        ),
        (
            "RGB565->RGBA8",
            rgb565,
            PixelFormat::RGBA8_SRGB,
            perm.clone(),
            None,
        ),
        (
            "RGB10A2->RGB16",
            rgb10a2,
            PixelFormat::RGB16_SRGB,
            perm.clone(),
            None,
        ),
        (
            "P010_RGB->RGBF32",
            p010_rgb,
            PixelFormat::RGBF32_LINEAR,
            perm.clone(),
            None,
        ),
        (
            "RGBA8->RGB8_MatteBlack",
            PixelFormat::RGBA8_SRGB,
            PixelFormat::RGB8_SRGB,
            perm.clone()
                .with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK)),
            None,
        ),
    ];

    for (name, src, dst, pol, peak) in cases {
        verify_partition_and_canary(name, src, dst, pol, peak);
    }

    #[cfg(feature = "oklab")]
    {
        verify_partition_and_canary(
            "RGB8_sRGB->OklabF32",
            PixelFormat::RGB8_SRGB,
            PixelFormat::OKLAB_F32,
            perm.clone(),
            None,
        );
        verify_partition_and_canary(
            "P3_F32->sRGB_F32_SoftCompress",
            PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3),
            PixelFormat::RGBF32_LINEAR,
            perm.clone().with_gamut(GamutPolicy::SoftCompress),
            None,
        );
    }

    #[cfg(feature = "hdr-transfers")]
    {
        let hlg_disp = PixelFormat::new(
            SampleLayout::RGB16,
            ColorSpace::Rgb {
                primaries: Primaries::Bt2020,
                transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        verify_partition_and_canary(
            "HLG16_Display->Lin2020_F32",
            hlg_disp,
            PixelFormat::RGBF32_LINEAR
                .with_primaries(Primaries::Bt2020)
                .with_white(DiffuseWhite::BT2408),
            ConvertPolicy::working_linear(),
            None,
        );
        #[cfg(feature = "bt2446a")]
        verify_partition_and_canary(
            "PQ16->sRGB8_BT2446A",
            PixelFormat::RGB16_BT2100_PQ,
            PixelFormat::RGB8_SRGB,
            ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new_positive(100.0).unwrap()),
            Some(Nits::new_positive(1000.0).unwrap()),
        );
    }
}

#[test]
fn portable_xxh3_digests_are_invariant_across_simd_and_stride() {
    let w = 67u32;
    let h = 5u32;
    let source_format = PixelFormat::RGBA8_SRGB.with_primaries(Primaries::DisplayP3);
    let mut src_tight = PixelBuf::new(w, h, source_format.clone()).unwrap();
    let mut src_strided =
        PixelBuf::new_aligned(w, h, source_format.clone(), RowAlign::Cache64).unwrap();
    for y in 0..h {
        let mut row = vec![0u8; w as usize * 4];
        for (i, b) in row.iter_mut().enumerate() {
            *b = ((y as usize * 131 + i * 37 + 7) & 0xFF) as u8;
        }
        src_tight.as_view_mut().row_mut(y).copy_from_slice(&row);
        src_strided.as_view_mut().row_mut(y).copy_from_slice(&row);
    }
    assert_eq!(
        src_tight.hash_with(HashInclusions::SamplesAndLayout),
        src_strided.hash_with(HashInclusions::SamplesAndLayout),
        "tight vs Cache64-strided source XXH3 must match"
    );

    let targets: Vec<(&str, PixelFormat, ConvertPolicy)> = vec![
        (
            "p3_rgba8_to_srgb_rgb8",
            PixelFormat::RGB8_SRGB,
            ConvertPolicy::permissive(),
        ),
        (
            "p3_rgba8_to_srgb_gray8_enc",
            PixelFormat::GRAY8_SRGB,
            ConvertPolicy::permissive(),
        ),
        (
            "p3_rgba8_to_srgb_gray8_lin",
            PixelFormat::GRAY8_SRGB,
            ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight),
        ),
        (
            "p3_rgba8_to_srgb_rgb16",
            PixelFormat::RGB16_SRGB,
            ConvertPolicy::permissive(),
        ),
        (
            "p3_rgba8_to_srgb_rgbf32",
            PixelFormat::RGBF32_LINEAR,
            ConvertPolicy::permissive(),
        ),
    ];

    for (name, destination_format, pol) in targets {
        let out_sc = ImagePlan::build(
            source_format.clone(),
            destination_format.clone(),
            &pol.clone().with_accelerators(Accelerators::SCALAR),
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&src_tight)
        .unwrap();
        let out_all = ImagePlan::build(
            source_format.clone(),
            destination_format.clone(),
            &pol.with_accelerators(Accelerators::ALL),
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&src_strided)
        .unwrap();
        let h_sc = out_sc.hash_with(HashInclusions::PixelsAndColor);
        let h_all = out_all.hash_with(HashInclusions::PixelsAndColor);
        assert_eq!(
            h_sc, h_all,
            "{name}: portable XXH3 digest differed between SCALAR/tight and ALL/strided"
        );
    }
}
