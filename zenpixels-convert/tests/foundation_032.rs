//! End-to-end tests for the 0.3.2 `zenpixels-convert` additions:
//! - Item 2.1: Parametric HDR (`Pq` / `Hlg`) + CMS (`crate::test_conversion::encoded_worker`,
//!   `OutputPlan`, `MoxCms`, `Lcms`) and honest missing-anchor errors.
//! - Item 2.2: `CodeMap` `U32` (`1..=32` bits) + signed two's-complement `I8`/`I16`/`I32`,
//!   plus `codes::to_canonical` layout parity (`Bgr`, `Bgra`, `Rgbx`, `Bgrx`, `Cmyk*`, `CmykAlpha*`).
//! - Item 2.3a: `ChromaSubsampling::Yuv440` centered vertical 2× upsampler.
//! - Item 2.4a: `PlaneConvertExt` (`oriented`, `oriented_into`) across EXIF orientations `1..=8`.

use zenpixels::{
    ChannelType, ColorContext, ColorOrigin, ConvertOptions, ParametricColor, PixelDescriptor,
    PixelEncoding, PixelFormat, SignalRange,
    hdr::{DiffuseWhite, HlgRendering},
    sample::SampleEncoding,
};
use zenpixels_convert::{
    ConvertError, OutputProfile,
    codes::{CodeMap, CodeMapError, CodeRole},
};

#[test]
fn codemap_u32_up_to_32_bits_and_signed_twos_complement() {
    // 1. Unsigned U32 12 -> 24 -> 32 -> 12 round-trip without u64 overflow at u32::MAX.
    let u12 = SampleEncoding::new(ChannelType::U32, 12, 0).unwrap();
    let u24_shifted = SampleEncoding::new(ChannelType::U32, 24, 8).unwrap();
    let u32_full = SampleEncoding::new(ChannelType::U32, 32, 0).unwrap();

    let widen_12_32 = CodeMap::new(u12, u32_full, CodeRole::Intensity, SignalRange::Full).unwrap();
    let narrow_32_12 = CodeMap::new(u32_full, u12, CodeRole::Intensity, SignalRange::Full).unwrap();
    assert_eq!(widen_12_32.code_u32(0), 0);
    assert_eq!(widen_12_32.code_u32(4095), u32::MAX);
    for q in 0..=4095u32 {
        let w = widen_12_32.code_u32(q);
        assert_eq!(narrow_32_12.code_u32(w), q, "12->32->12 failed at {q}");
    }

    // u16 slice helper rejects U32 storage cleanly.
    let mut dummy16 = [0u16; 2];
    assert_eq!(
        widen_12_32.map_words(&[0, 4095], &mut dummy16),
        Err(CodeMapError::UnsupportedStorage)
    );

    // u32 word helper shifts and masks padding.
    let pack_24 = CodeMap::new(u12, u24_shifted, CodeRole::Intensity, SignalRange::Full).unwrap();
    let mut dst32 = [0u32; 3];
    pack_24.map_words_u32(&[0, 2048, 4095], &mut dst32).unwrap();
    assert_eq!(dst32[0], 0);
    assert_eq!(dst32[2], 0xFFFF_FF00);

    // Narrow-range U32 scaling (10-bit -> 20-bit -> 10-bit).
    let n10 = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
    let n20 = SampleEncoding::new(ChannelType::U32, 20, 4).unwrap();
    let nw = CodeMap::new(n10, n20, CodeRole::Intensity, SignalRange::Narrow).unwrap();
    let nn = CodeMap::new(n20, n10, CodeRole::Intensity, SignalRange::Narrow).unwrap();
    assert_eq!(nw.word_u32(64), (64u32 << 10) << 4);
    assert_eq!(nn.word_u32((940u32 << 10) << 4), 940);

    // 2. Signed two's-complement I8 / I16 / I32 round-trip and endpoint preservation.
    let i8_enc = SampleEncoding::new(ChannelType::I8, 8, 0).unwrap();
    let i10_enc = SampleEncoding::new(ChannelType::I16, 10, 4).unwrap();
    let i16_enc = SampleEncoding::new(ChannelType::I16, 16, 0).unwrap();
    let i32_enc = SampleEncoding::new(ChannelType::I32, 32, 0).unwrap();

    // Mixed signed/unsigned and narrow signed are rejected.
    assert_eq!(
        CodeMap::new(i8_enc, u12, CodeRole::Intensity, SignalRange::Full),
        Err(CodeMapError::UnsupportedStorage)
    );
    assert_eq!(
        CodeMap::new(i8_enc, i16_enc, CodeRole::Intensity, SignalRange::Narrow),
        Err(CodeMapError::UnsupportedRange(SignalRange::Narrow))
    );

    let s_widen = CodeMap::new(i8_enc, i16_enc, CodeRole::Intensity, SignalRange::Full).unwrap();
    let s_narrow = CodeMap::new(i16_enc, i8_enc, CodeRole::Intensity, SignalRange::Full).unwrap();
    assert_eq!(s_widen.code_i32(-128), -32768);
    assert_eq!(s_widen.code_i32(0), 0);
    assert_eq!(s_widen.code_i32(127), 32767);
    for q in -128..=127i32 {
        let w = s_widen.code_i32(q);
        assert_eq!(s_narrow.code_i32(w), q, "I8->I16->I8 failed at {q}");
    }

    // Exhaustive 10-bit shifted I16 -> 32-bit I32 -> 10-bit shifted I16.
    let s_10_32 = CodeMap::new(i10_enc, i32_enc, CodeRole::Intensity, SignalRange::Full).unwrap();
    let s_32_10 = CodeMap::new(i32_enc, i10_enc, CodeRole::Intensity, SignalRange::Full).unwrap();
    assert_eq!(s_10_32.code_i32(-512), i32::MIN);
    assert_eq!(s_10_32.code_i32(0), 0);
    assert_eq!(s_10_32.code_i32(511), i32::MAX);
    for q in -512..=511i32 {
        let packed = (((q as u32) & 0x3FF) << 4) as i32;
        let w32 = s_10_32.word_i32(packed);
        let back_packed = s_32_10.word_i32(w32);
        let unpacked = ((back_packed as u32 >> 4) << 22) as i32 >> 22;
        assert_eq!(unpacked, q, "signed 10->32->10 failed at {q}");
    }

    let src_i16 = [-128i16, -1, 0, 1, 127];
    let mut mid_i16 = [0i16; 5];
    let mut back_i16 = [0i16; 5];
    s_widen.map_words_i16(&src_i16, &mut mid_i16).unwrap();
    s_narrow.map_words_i16(&mid_i16, &mut back_i16).unwrap();
    assert_eq!(back_i16, src_i16);
}

#[cfg(feature = "planar")]
mod planar_032 {
    use super::*;
    use zenpixels::{
        AlphaMode, ChannelLayout, F16Sample, Orientation, PixelSlice, PixelSliceMut,
        plane::{AnyPlane, PlaneRef},
        sample::{ChromaSiting, ChromaSubsampling, CodePlane, ExactDepth},
    };
    use zenpixels_convert::{PlaneConvertExt, codes::to_canonical, ycbcr::YcbcrToRgbF32};

    #[test]
    fn to_canonical_supports_bgr_rgbx_bgrx_cmyk_cmyka_and_u32_code_planes() {
        // 1. U32-backed 12-bit CodePlane -> Bgr16
        let bgr16 = PixelFormat::from_parts(ChannelType::U16, ChannelLayout::Bgr, None)
            .unwrap()
            .descriptor();
        let u32_12 = SampleEncoding::new(ChannelType::U32, 12, 4).unwrap();
        let b_src = [0u32 << 4, 2048 << 4, 4095 << 4];
        let g_src = [1024u32 << 4, 4095 << 4, 0];
        let r_src = [4095u32 << 4, 0, 100 << 4];
        let planes = [&b_src, &g_src, &r_src].map(|s| {
            CodePlane::new(
                AnyPlane::U32(PlaneRef::new(&s[..], 3, 1, 3).unwrap()),
                u32_12,
            )
            .unwrap()
        });
        let mut out_bgr = [0u16; 9];
        let (_, exact) = to_canonical(
            &planes,
            SignalRange::Full,
            0,
            0,
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut out_bgr), 3, 1, 18, bgr16).unwrap(),
        )
        .unwrap();
        assert_eq!((exact.color_bits(), exact.alpha_bits()), (12, 16));
        assert_eq!(out_bgr[0], 0);
        assert_eq!(out_bgr[2], 65535);
        let slice = PixelSlice::new(bytemuck::cast_slice(&out_bgr), 3, 1, 18, bgr16).unwrap();
        assert!(exact.holds_for(&slice).unwrap());
        assert_eq!(ExactDepth::measure(&slice).unwrap(), exact);

        // 2. RGBX8 and BGRX8 take 3 color planes and leave the 4th padding slot untouched.
        for desc in [PixelDescriptor::RGBX8_SRGB, PixelDescriptor::BGRX8_SRGB] {
            let u8_4 = SampleEncoding::new(ChannelType::U8, 4, 0).unwrap();
            let c0 = [0u8, 15];
            let c1 = [5u8, 10];
            let c2 = [15u8, 1];
            let planes = [&c0, &c1, &c2].map(|s| {
                CodePlane::new(AnyPlane::U8(PlaneRef::new(&s[..], 2, 1, 2).unwrap()), u8_4).unwrap()
            });
            let mut bytes = [0xAAu8; 8];
            let (_, exact) = to_canonical(
                &planes,
                SignalRange::Full,
                0,
                0,
                PixelSliceMut::new(&mut bytes, 2, 1, 8, desc).unwrap(),
            )
            .unwrap();
            assert_eq!(bytes, [0, 85, 255, 0xAA, 255, 170, 17, 0xAA]);
            assert_eq!((exact.color_bits(), exact.alpha_bits()), (4, 8));
            let view = PixelSlice::new(&bytes, 2, 1, 8, desc).unwrap();
            assert!(exact.holds_for(&view).unwrap());
        }

        // 3. CMYK16_INVERTED and 5-channel CMYKA16 (10-bit ink + 1-bit alpha).
        let u16_10 = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
        let u16_1 = SampleEncoding::new(ChannelType::U16, 1, 0).unwrap();
        let c = [0u16, 1023];
        let m = [512u16, 256];
        let y = [1023u16, 0];
        let k = [128u16, 64];
        let a = [0u16, 1];
        let cmyka16 = PixelFormat::from_parts(
            ChannelType::U16,
            ChannelLayout::CmykInvertedAlpha,
            Some(AlphaMode::Straight),
        )
        .unwrap()
        .descriptor();
        let planes_5 = [
            CodePlane::new(AnyPlane::U16(PlaneRef::new(&c, 2, 1, 2).unwrap()), u16_10).unwrap(),
            CodePlane::new(AnyPlane::U16(PlaneRef::new(&m, 2, 1, 2).unwrap()), u16_10).unwrap(),
            CodePlane::new(AnyPlane::U16(PlaneRef::new(&y, 2, 1, 2).unwrap()), u16_10).unwrap(),
            CodePlane::new(AnyPlane::U16(PlaneRef::new(&k, 2, 1, 2).unwrap()), u16_10).unwrap(),
            CodePlane::new(AnyPlane::U16(PlaneRef::new(&a, 2, 1, 2).unwrap()), u16_1).unwrap(),
        ];
        let mut out_cmyka = [0u16; 10];
        let (_, exact) = to_canonical(
            &planes_5,
            SignalRange::Full,
            0,
            0,
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut out_cmyka), 2, 1, 20, cmyka16)
                .unwrap(),
        )
        .unwrap();
        assert_eq!((exact.color_bits(), exact.alpha_bits()), (10, 1));
        assert_eq!(out_cmyka[4], 0);
        assert_eq!(out_cmyka[9], 65535);
        let view = PixelSlice::new(bytemuck::cast_slice(&out_cmyka), 2, 1, 20, cmyka16).unwrap();
        assert!(exact.holds_for(&view).unwrap());
        assert_eq!(ExactDepth::measure(&view).unwrap(), exact);
    }

    #[test]
    fn yuv440_upsamples_vertically_without_horizontal_blur() {
        let enc = SampleEncoding::new(ChannelType::U8, 8, 0).unwrap();
        // 2x4 luma neutral 128, chroma 2x2 with distinct left/right columns and top/bottom rows.
        let y = [128u8; 8];
        // Row 0: [64, 192], Row 1: [192, 64]
        let cb = [64u8, 192, 192, 64];
        let cr = [128u8; 4];
        let planes = [
            CodePlane::new(AnyPlane::U8(PlaneRef::new(&y, 2, 4, 2).unwrap()), enc).unwrap(),
            CodePlane::new(AnyPlane::U8(PlaneRef::new(&cb, 2, 2, 2).unwrap()), enc).unwrap(),
            CodePlane::new(AnyPlane::U8(PlaneRef::new(&cr, 2, 2, 2).unwrap()), enc).unwrap(),
        ];
        let plan = YcbcrToRgbF32::new(
            planes,
            ChromaSubsampling::Yuv440,
            ChromaSiting::Center,
            zenpixels::Cicp::new(1, 13, 1, true),
        )
        .unwrap();
        let mut out = [[0.0f32; 3]; 8];
        plan.convert_rows(
            0,
            0,
            PixelSliceMut::new(
                bytemuck::cast_slice_mut(&mut out),
                2,
                4,
                24,
                plan.output_descriptor(),
            )
            .unwrap(),
        )
        .unwrap();
        // Top row (y=0) clamps at row 0: left blue < right blue (no horizontal mixing).
        assert!(out[0][2] < out[1][2]);
        // Row 1 (3/4 row0 + 1/4 row1) vs Row 2 (1/4 row0 + 3/4 row1) swap left/right dominance!
        assert!(out[2][2] < out[3][2]);
        assert!(out[4][2] > out[5][2]);
        // Bottom row (y=3) clamps at row 1: left blue > right blue.
        assert!(out[6][2] > out[7][2]);
    }

    #[test]
    fn plane_convert_ext_orients_all_eight_exif_orientations_and_preserves_padding() {
        let (w, h) = (7usize, 4usize);
        let mut plane = zenpixels::plane::AlignedPlane::<u32>::alloc(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                plane.row_mut(y).unwrap()[x] = ((y * 100 + x) as u32) ^ 0xA55A_0000;
            }
        }
        for orient in Orientation::ALL {
            let (ow, oh) = orient.output_dimensions(w as u32, h as u32);
            let out = plane.oriented(orient).unwrap();
            assert_eq!((out.width(), out.height()), (ow as usize, oh as usize));
            for sy in 0..h as u32 {
                for sx in 0..w as u32 {
                    let (dx, dy) = orient.forward_map(sx, sy, w as u32, h as u32);
                    assert_eq!(
                        out.row(dy as usize).unwrap()[dx as usize],
                        plane.row(sy as usize).unwrap()[sx as usize],
                        "u32 plane mismatch for {orient:?} at ({sx}, {sy})"
                    );
                }
            }

            // Test oriented_into with explicit trailing row padding sentinel.
            let dst_stride = ow as usize + 5;
            let mut backing = vec![0xDEAD_BEEFu32; dst_stride * oh as usize];
            let dst_mut =
                zenpixels::plane::PlaneMut::new(&mut backing, ow as usize, oh as usize, dst_stride)
                    .unwrap();
            plane.view().oriented_into(orient, dst_mut).unwrap();
            for dy in 0..oh as usize {
                assert_eq!(
                    &backing[dy * dst_stride..dy * dst_stride + ow as usize],
                    out.row(dy).unwrap()
                );
                assert!(
                    backing[dy * dst_stride + ow as usize..(dy + 1) * dst_stride]
                        .iter()
                        .all(|&v| v == 0xDEAD_BEEF)
                );
            }
        }

        // Also exercise 2-byte F16Sample and 8-byte u64 planes.
        let mut f16_plane = zenpixels::plane::AlignedPlane::alloc(3, 2).unwrap();
        for y in 0..2 {
            for x in 0..3 {
                f16_plane.row_mut(y).unwrap()[x] = F16Sample::from_bits((y * 10 + x + 1) as u16);
            }
        }
        let rot90 = f16_plane.oriented(Orientation::Rotate90).unwrap();
        assert_eq!((rot90.width(), rot90.height()), (2, 3));
        assert_eq!(rot90.row(0).unwrap()[1], F16Sample::from_bits(1));

        let mut u64_plane = zenpixels::plane::AlignedPlane::alloc(3, 2).unwrap();
        u64_plane
            .row_mut(0)
            .unwrap()
            .copy_from_slice(&[11u64, 12, 13]);
        u64_plane
            .row_mut(1)
            .unwrap()
            .copy_from_slice(&[21u64, 22, 23]);
        let rot270 = u64_plane.oriented(Orientation::Rotate270).unwrap();
        assert_eq!((rot270.width(), rot270.height()), (2, 3));
        assert_eq!(rot270.row(0).unwrap(), &[13u64, 23]);
    }
}

#[test]
fn parametric_hdr_honest_errors_and_cms_free_resolved_paths() {
    let bt2100_hlg = ParametricColor::from_cicp(zenpixels::Cicp::BT2100_HLG_RGB).unwrap();
    let bt2100_pq = ParametricColor::from_cicp(zenpixels::Cicp::BT2100_PQ_RGB).unwrap();
    let srgb = ParametricColor::from_cicp(zenpixels::Cicp::SRGB).unwrap();
    let lin_srgb = ParametricColor::from_cicp(zenpixels::Cicp::LINEAR_SRGB).unwrap();

    let hlg_ctx = ColorContext::from_parametric(bt2100_hlg);
    let pq_ctx = ColorContext::from_parametric(bt2100_pq);
    let srgb_ctx = ColorContext::from_parametric(srgb);
    let lin_ctx = ColorContext::from_parametric(lin_srgb);

    let hlg_desc = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(zenpixels::ColorPrimaries::Bt2020)
        .with_transfer(zenpixels::TransferFunction::Hlg);
    let pq_desc = PixelDescriptor::RGBF32_LINEAR
        .with_primaries(zenpixels::ColorPrimaries::Bt2020)
        .with_transfer(zenpixels::TransferFunction::Pq);
    let hlg_enc = PixelEncoding::new(hlg_desc, Some(&hlg_ctx));
    let pq_enc = PixelEncoding::new(pq_desc, Some(&pq_ctx));
    let srgb_enc = PixelEncoding::new(
        PixelDescriptor::RGBF32_LINEAR.with_transfer(zenpixels::TransferFunction::Srgb),
        Some(&srgb_ctx),
    );
    let lin_enc = PixelEncoding::new(PixelDescriptor::RGBF32_LINEAR, Some(&lin_ctx));

    // 1. Missing HLG rendering returns HlgRenderingRequired (NOT NeedsCms!).
    let err = crate::test_conversion::encoded_worker(
        hlg_enc,
        lin_enc,
        &ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
        None,
    )
    .unwrap_err();
    assert_eq!(err.error(), &ConvertError::HlgRenderingRequired);

    // 2. Missing SDR white returns SdrWhiteRequired (NOT NeedsCms!).
    let err = crate::test_conversion::encoded_worker(
        pq_enc,
        lin_enc,
        &ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
        None,
    )
    .unwrap_err();
    assert_eq!(err.error(), &ConvertError::SdrWhiteRequired);

    // 3. Missing HDR peak on PQ -> sRGB returns HdrPeakRequired (NOT NeedsCms!).
    let err = crate::test_conversion::encoded_worker(
        pq_enc,
        srgb_enc,
        &ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
        None,
    )
    .unwrap_err();
    assert_eq!(err.error(), &ConvertError::HdrPeakRequired);

    // 4. With explicit anchors, Parametric BT2100_HLG -> Parametric LINEAR_SRGB runs natively without CMS!
    let opts = ConvertOptions::retain_channels()
        .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
        .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
        .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
        .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
        .with_sdr_white(DiffuseWhite::BT2408)
        .with_gamut_policy(zenpixels_convert::GamutPolicy::PreserveExtended);
    let mut plan = crate::test_conversion::output_plan(
        hlg_enc,
        &ColorOrigin::from_parametric(bt2100_hlg),
        OutputProfile::Parametric(lin_srgb),
        PixelFormat::RgbF32,
        &opts,
        None,
    )
    .unwrap();
    let src = [0.75f32, 0.75, 0.75];
    let mut dst = [0.0f32; 3];
    let src_slice =
        zenpixels::PixelSlice::new(bytemuck::cast_slice(&src), 1, 1, 12, hlg_enc.descriptor())
            .unwrap()
            .with_color_context(std::sync::Arc::new(hlg_ctx));
    let dst_slice = zenpixels::PixelSliceMut::new(
        bytemuck::cast_slice_mut(&mut dst),
        1,
        1,
        12,
        plan.output_descriptor(),
    )
    .unwrap();
    plan.convert_rows(src_slice, dst_slice).unwrap();
    // 75% HLG signal on 1000-nit reference display (~203 nits) -> ~1.0 relative to 203-nit diffuse white!
    for c in dst {
        assert!(
            (c - 1.0).abs() < 0.02,
            "expected ~1.0 diffuse white, got {c}"
        );
    }
}

#[cfg(all(feature = "cms-moxcms", feature = "cms-lcms"))]
#[test]
fn moxcms_and_lcms_handle_parametric_hlg_and_hdr_icc_pivots() {
    use zenpixels::{CieXy, ParametricPrimaries, ParametricTransfer, WhitePoint};
    use zenpixels_convert::{Lcms, MoxCms, cms::PluggableCms};

    let custom_hlg = ParametricColor::rgb(
        ParametricPrimaries::Custom {
            red: CieXy::new(0.680, 0.320),
            green: CieXy::new(0.265, 0.690),
            blue: CieXy::new(0.150, 0.060),
        },
        WhitePoint::D65,
        ParametricTransfer::Hlg,
    );
    let opts = ConvertOptions::retain_channels()
        .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
        .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
        .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
        .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
        .with_sdr_white(DiffuseWhite::BT2408)
        .with_gamut_policy(zenpixels_convert::GamutPolicy::PreserveExtended);

    // Both MoxCms and Lcms accept ParametricTransfer::Hlg when HlgRendering + sdr_white are set,
    // and decline (`None`) when omitted.
    for cms in [&MoxCms as &dyn PluggableCms, &Lcms as &dyn PluggableCms] {
        assert!(
            cms.build_source_transform(
                zenpixels::ColorProfileSource::Parametric(custom_hlg),
                zenpixels::ColorProfileSource::Parametric(ParametricColor::ACES_CG),
                PixelFormat::RgbF32,
                PixelFormat::RgbF32,
                &ConvertOptions::retain_channels()
                    .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
                    .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
                    .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
            )
            .is_none()
        );
    }

    let mut mox_xf = MoxCms
        .build_source_transform(
            zenpixels::ColorProfileSource::Parametric(custom_hlg),
            zenpixels::ColorProfileSource::Parametric(ParametricColor::ACES_CG),
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
            &opts,
        )
        .expect("MoxCms should accept Parametric HLG with conditions")
        .unwrap();
    let mut lcms_xf = Lcms
        .build_source_transform(
            zenpixels::ColorProfileSource::Parametric(custom_hlg),
            zenpixels::ColorProfileSource::Parametric(ParametricColor::ACES_CG),
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
            &opts,
        )
        .expect("Lcms should accept Parametric HLG with conditions")
        .unwrap();

    let input = [0.75f32, 0.5, 0.25];
    let (mut mox_out, mut lcms_out) = ([0.0f32; 3], [0.0f32; 3]);
    mox_xf.prepare(1).unwrap();
    lcms_xf.prepare(1).unwrap();
    mox_xf
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut mox_out),
            1,
        )
        .unwrap();
    lcms_xf
        .try_transform_row(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice_mut(&mut lcms_out),
            1,
        )
        .unwrap();
    for i in 0..3 {
        assert!(
            (mox_out[i] - lcms_out[i]).abs() < 2e-3,
            "ch{i}: mox={} vs lcms={}",
            mox_out[i],
            lcms_out[i]
        );
    }

    // Also verify crate::test_conversion::encoded_worker on custom-primaries HLG <-> ProPhoto RGB ICC/Parametric!
    let custom_ctx = ColorContext::from_parametric(custom_hlg);
    let acescg_ctx = ColorContext::from_parametric(ParametricColor::ACES_CG);
    let unknown_rgb = PixelFormat::RgbF32
        .descriptor()
        .with_primaries(zenpixels::ColorPrimaries::Unknown)
        .with_transfer(zenpixels::TransferFunction::Unknown);
    let mut conv = crate::test_conversion::encoded_worker(
        PixelEncoding::new(unknown_rgb, Some(&custom_ctx)),
        PixelEncoding::new(unknown_rgb, Some(&acescg_ctx)),
        &opts,
        Some(&MoxCms),
    )
    .unwrap();
    let mut pivot_out = [0.0f32; 3];
    conv.try_convert_row(
        bytemuck::cast_slice(&input),
        bytemuck::cast_slice_mut(&mut pivot_out),
        1,
    )
    .unwrap();
    for i in 0..3 {
        assert!(
            (pivot_out[i] - mox_out[i]).abs() < 2e-3,
            "pivot ch{i}: {} vs {}",
            pivot_out[i],
            mox_out[i]
        );
    }

    // With bt2446a, tone-mapping BT.2100 PQ -> DCI-P3 Parametric via CMS succeeds!
    #[cfg(feature = "bt2446a")]
    {
        use zenpixels::hdr::{HdrPeak, Nits};
        let tm_opts = opts
            .with_hdr_peak(HdrPeak::Nits(Nits::new(1000.0)))
            .with_sdr_peak(Nits::new(100.0));
        let pq_ctx = ColorContext::from_parametric(
            ParametricColor::from_cicp(zenpixels::Cicp::BT2100_PQ_RGB).unwrap(),
        );
        let dci_ctx = ColorContext::from_parametric(ParametricColor::DCI_P3);
        let mut tm_conv = crate::test_conversion::encoded_worker(
            PixelEncoding::new(PixelDescriptor::RGB16_BT2100_PQ, Some(&pq_ctx)),
            PixelEncoding::new(
                PixelDescriptor::RGB8.with_primaries(zenpixels::ColorPrimaries::Unknown),
                Some(&dci_ctx),
            ),
            &tm_opts,
            Some(&MoxCms),
        )
        .unwrap();
        let src_pq16 = [38000u16, 38000, 38000];
        let mut dst_dci8 = [0u8; 3];
        tm_conv
            .try_convert_row(bytemuck::cast_slice(&src_pq16), &mut dst_dci8, 1)
            .unwrap();
        assert!(dst_dci8.iter().all(|&c| c > 100));
    }
}

#[allow(dead_code)]
#[path = "support/cms.rs"]
mod test_cms;
#[allow(unused_imports)]
use test_cms::TestCmsExt as _;

#[path = "support/conversion.rs"]
mod test_conversion;
