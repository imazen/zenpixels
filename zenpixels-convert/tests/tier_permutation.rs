//! Isolated CPU-tier permutation test using `archmage::testing::for_each_token_permutation`.
//!
//! Runs in its own integration test binary so process-wide token disabling exercises
//! every compiled SIMD tier (`_v4x`, `_v3`, `_neon`, `_wasm128`, `_scalar`) independently
//! without racing unrelated parallel tests, asserting **0-bit exact parity** across all tiers.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenpixels::{AlphaMode, Orientation, PixelBuf, PixelFormat, Primaries, Transfer};
#[cfg(feature = "hdr-transfers")]
use zenpixels::{ColorSpace, DiffuseWhite, HlgMode, HlgRendering, SampleLayout, SignalRange};
#[cfg(feature = "oklab")]
use zenpixels_convert::GamutPolicy;
use zenpixels_convert::{
    Accelerators, ConvertPolicy, DepthPolicy, LumaCoefficients, RowPlan, apply_orientation,
};

#[test]
fn all_simd_kernels_and_orient_are_bit_identical_across_every_cpu_tier() {
    let w = 97u32;
    let rgb8: Vec<u8> = (0..(w as usize * 3))
        .map(|i| ((i * 73 + 17) & 0xFF) as u8)
        .collect();
    let rgba8: Vec<u8> = (0..(w as usize * 4))
        .map(|i| ((i * 59 + 11) & 0xFF) as u8)
        .collect();
    let rgb16: Vec<u8> = (0..(w as usize * 3))
        .flat_map(|i| (((i * 9973 + 321) & 0xFFFF) as u16).to_ne_bytes())
        .collect();
    let rgbf32: Vec<u8> = (0..(w as usize * 3))
        .flat_map(|i| (((i * 37 + 13) % 1009) as f32 / 1008.0).to_ne_bytes())
        .collect();
    let rgbaf32: Vec<u8> = (0..(w as usize * 4))
        .flat_map(|i| (((i * 29 + 7) % 1009) as f32 / 1008.0).to_ne_bytes())
        .collect();
    let rgbf16: Vec<u8> = (0..(w as usize * 3))
        .flat_map(|i| {
            half::f16::from_f32(((i * 31 + 5) % 257) as f32 / 256.0)
                .to_bits()
                .to_ne_bytes()
        })
        .collect();

    let p3_u8 = PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let p3_f32 = PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3);
    let rgba8_premul = PixelFormat::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    let bt709_f32 = PixelFormat::RGBF32_LINEAR.with_transfer(Transfer::Bt709);
    let g22_f32 = PixelFormat::RGBF32_LINEAR.with_transfer(Transfer::Gamma22);
    let adobe_f32 = PixelFormat::RGBF32_LINEAR
        .with_primaries(Primaries::AdobeRgb)
        .with_transfer(Transfer::AdobeRgb);

    #[allow(unused_mut)]
    let mut routes: Vec<(&str, RowPlan, &[u8])> = vec![
        (
            "rgb8_to_gray8_bt709",
            RowPlan::build(
                PixelFormat::RGB8_SRGB,
                PixelFormat::GRAY8_SRGB,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb8,
        ),
        (
            "rgba8_to_gray8_bt601",
            RowPlan::build(
                PixelFormat::RGBA8_SRGB,
                PixelFormat::GRAY8_SRGB,
                &ConvertPolicy::permissive()
                    .with_luma(LumaCoefficients::Bt601)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgba8,
        ),
        (
            "rgba8_premul_round",
            RowPlan::build(
                PixelFormat::RGBA8_SRGB,
                rgba8_premul.clone(),
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgba8,
        ),
        (
            "rgba8_premul_trunc",
            RowPlan::build(
                PixelFormat::RGBA8_SRGB,
                rgba8_premul,
                &ConvertPolicy::permissive()
                    .with_depth(DepthPolicy::Truncate)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgba8,
        ),
        (
            "lin_f32_to_srgb8_exact",
            RowPlan::build(
                PixelFormat::RGBAF32_LINEAR,
                PixelFormat::RGBA8_SRGB,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbaf32,
        ),
        (
            "lin_f32_to_srgb8_approx",
            RowPlan::build(
                PixelFormat::RGBAF32_LINEAR,
                PixelFormat::RGBA8_SRGB,
                &ConvertPolicy::permissive()
                    .with_approximate(true)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbaf32,
        ),
        (
            "p3_u8_to_srgb8_exact",
            RowPlan::build(
                p3_u8.clone(),
                PixelFormat::RGB8_SRGB,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb8,
        ),
        (
            "p3_u8_to_srgb8_q14",
            RowPlan::build(
                p3_u8,
                PixelFormat::RGB8_SRGB,
                &ConvertPolicy::permissive()
                    .with_approximate(true)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb8,
        ),
        (
            "p3_f32_to_srgb_f32",
            RowPlan::build(
                p3_f32,
                PixelFormat::RGBF32_LINEAR,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ),
        (
            "f16_to_f32",
            RowPlan::build(
                PixelFormat::RGBF16_LINEAR,
                PixelFormat::RGBF32_LINEAR,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf16,
        ),
        (
            "f32_to_f16",
            RowPlan::build(
                PixelFormat::RGBF32_LINEAR,
                PixelFormat::RGBF16_LINEAR,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ),
        (
            "bt709_eotf",
            RowPlan::build(
                bt709_f32,
                PixelFormat::RGBF32_LINEAR,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ),
        (
            "gamma22_oetf",
            RowPlan::build(
                PixelFormat::RGBF32_LINEAR,
                g22_f32,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ),
        (
            "adobe_eotf",
            RowPlan::build(
                adobe_f32,
                PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::AdobeRgb),
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ),
        (
            "u16_p3_to_u16_srgb",
            RowPlan::build(
                PixelFormat::RGB16_SRGB.with_primaries(Primaries::DisplayP3),
                PixelFormat::RGB16_SRGB,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb16,
        ),
    ];

    #[cfg(feature = "oklab")]
    {
        routes.push((
            "rgb8_to_oklab_f32",
            RowPlan::build(
                PixelFormat::RGB8_SRGB,
                PixelFormat::OKLAB_F32,
                &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb8,
        ));
        routes.push((
            "p3_soft_compress",
            RowPlan::build(
                PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3),
                PixelFormat::RGBF32_LINEAR,
                &ConvertPolicy::permissive()
                    .with_gamut(GamutPolicy::SoftCompress)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgbf32,
        ));
    }

    #[cfg(feature = "hdr-transfers")]
    {
        routes.push((
            "pq16_to_lin2020_f32",
            RowPlan::build(
                PixelFormat::RGB16_BT2100_PQ,
                PixelFormat::RGBF32_LINEAR
                    .with_primaries(Primaries::Bt2020)
                    .with_white(DiffuseWhite::BT2408),
                &ConvertPolicy::working_linear().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb16,
        ));
        let hlg = PixelFormat::new(
            SampleLayout::RGB16,
            ColorSpace::Rgb {
                primaries: Primaries::Bt2020,
                transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        routes.push((
            "hlg16_display_to_lin2020_f32",
            RowPlan::build(
                hlg,
                PixelFormat::RGBF32_LINEAR
                    .with_primaries(Primaries::Bt2020)
                    .with_white(DiffuseWhite::BT2408),
                &ConvertPolicy::working_linear().with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap(),
            &rgb16,
        ));
    }

    // Orient transpose buffers for 1, 2, 3, 4, 6, 8, 12, 16 bpp.
    let orient_specs = [
        PixelFormat::GRAY8_SRGB,
        PixelFormat::GRAY16_SRGB,
        PixelFormat::RGB8_SRGB,
        PixelFormat::RGBA8_SRGB,
        PixelFormat::RGB16_SRGB,
        PixelFormat::RGBA16_SRGB,
        PixelFormat::RGBF32_LINEAR,
        PixelFormat::RGBAF32_LINEAR,
    ];
    let orient_bufs: Vec<PixelBuf> = orient_specs
        .iter()
        .map(|spec| {
            let mut b = PixelBuf::new(35, 29, spec.clone()).unwrap();
            let rb = b.row_bytes();
            let mut vm = b.as_view_mut();
            for y in 0..29u32 {
                for (x, byte) in vm.row_mut(y).iter_mut().enumerate() {
                    *byte = (((y as usize) * rb + x).wrapping_mul(47) & 0xFF) as u8;
                }
            }
            b
        })
        .collect();

    // Capture baseline outputs on highest available tier.
    let baselines: Vec<Vec<u8>> = routes
        .iter()
        .map(|(_, plan, src)| {
            let mut dst =
                vec![0u8; w as usize * plan.destination_format().layout.bytes_per_pixel()];
            plan.convert_row(src, &mut dst, w).unwrap();
            dst
        })
        .collect();
    let orient_baselines: Vec<Vec<u8>> = orient_bufs
        .iter()
        .map(|b| {
            apply_orientation(&b.as_view(), Orientation::Rotate90)
                .unwrap()
                .as_view()
                .as_contiguous_bytes()
                .unwrap()
                .to_vec()
        })
        .collect();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for ((name, plan, src), expected) in routes.iter().zip(baselines.iter()) {
            let mut got = vec![0u8; expected.len()];
            plan.convert_row(src, &mut got, w).unwrap();
            assert_eq!(
                &got, expected,
                "tier permutation mismatch on {name} under tier {perm}"
            );
        }
        for (idx, (buf, expected)) in orient_bufs.iter().zip(orient_baselines.iter()).enumerate() {
            let got = apply_orientation(&buf.as_view(), Orientation::Rotate90).unwrap();
            assert_eq!(
                got.as_view().as_contiguous_bytes().unwrap(),
                expected.as_slice(),
                "orient Rotate90 mismatch on spec[{idx}] under tier {perm}"
            );
        }
    });
    assert!(
        report.permutations_run >= 1,
        "expected at least 1 CPU tier permutation"
    );
}
