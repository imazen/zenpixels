//! Core 0.5 tests covering algebraic `ColorSpace`, `ImageSpec` validation,
//! `Primaries::gamut_matrix_to`, `Cicp`, `BitPacking` (P010 / U10_LSB / RGB565 / RGB10A2),
//! `PixelView<'a>` chained lifetimes, `SampleFacts` invalidation, `BandedBuf`,
//! `PlanarBuf`, and `DualLayerHdrBuf`.

use zenpixels::{
    AlphaMode, BandedBuf, BitPacking, ChannelOrder, Cicp, CicpError, ColorError, ColorSpace,
    DiffuseWhite, DualLayerHdrBuf, FloatFacts, GainMapMetadata, HlgMode, HlgRendering, ImageSpec,
    InteropTarget, Nits, PixelBuf, PixelSink, PixelSource, PlanarBuf, PlaneRole, PlaneSpec,
    Primaries, SampleFacts, SampleLayout, SampleType, SignalRange, Transfer,
};

#[test]
fn algebraic_color_space_and_spec_validation() {
    assert!(ImageSpec::RGB8_SRGB.validate().is_ok());
    assert!(ImageSpec::RGBA8_SRGB.validate().is_ok());
    assert!(ImageSpec::RGBX8_SRGB.validate().is_ok());
    assert!(ImageSpec::OKLAB_F32.validate().is_ok());
    assert!(ImageSpec::OKLABA_F32.validate().is_ok());
    assert!(ImageSpec::RGB16_BT2100_PQ.validate().is_ok());
    assert!(ImageSpec::RGB16_BT2100_HLG_1000.validate().is_ok());

    // Alpha / layout slot mismatch rejected.
    let bad_alpha = ImageSpec::new(SampleLayout::RGB8, ColorSpace::SRGB_ALPHA);
    assert_eq!(
        *bad_alpha.validate().unwrap_err().error(),
        ColorError::AlphaLayoutMismatch
    );

    let bad_padding = ImageSpec::new(SampleLayout::RGBX8, ColorSpace::SRGB_ALPHA);
    assert_eq!(
        *bad_padding.validate().unwrap_err().error(),
        ColorError::AlphaLayoutMismatch
    );

    // Narrow range requires integer samples.
    let bad_narrow = ImageSpec::RGBF32_LINEAR.with_range(SignalRange::Narrow);
    assert_eq!(
        *bad_narrow.validate().unwrap_err().error(),
        ColorError::NarrowRangeRequiresInteger
    );

    // HLG Display requires full-range BT.2020.
    let bad_hlg = ImageSpec::RGB16_BT2100_HLG_1000.with_primaries(Primaries::Bt709);
    assert_eq!(
        *bad_hlg.validate().unwrap_err().error(),
        ColorError::HlgDisplayRequiresBt2020Full
    );

    // Oklab requires F32 + Oklab channel order.
    let bad_oklab = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::OKLAB);
    assert_eq!(
        *bad_oklab.validate().unwrap_err().error(),
        ColorError::ColorLayoutMismatch
    );
}

#[test]
fn primaries_gamut_matrices_roundtrip_accurately() {
    let all = [
        Primaries::Bt709,
        Primaries::DisplayP3,
        Primaries::Bt2020,
        Primaries::AdobeRgb,
    ];
    for &a in &all {
        for &b in &all {
            let m_ab = a.gamut_matrix_to(b).unwrap();
            let m_ba = b.gamut_matrix_to(a).unwrap();
            let sample = [0.73f32, 0.41, 0.19];
            let mid = [
                m_ab[0][0] * sample[0] + m_ab[0][1] * sample[1] + m_ab[0][2] * sample[2],
                m_ab[1][0] * sample[0] + m_ab[1][1] * sample[1] + m_ab[1][2] * sample[2],
                m_ab[2][0] * sample[0] + m_ab[2][1] * sample[1] + m_ab[2][2] * sample[2],
            ];
            let back = [
                m_ba[0][0] * mid[0] + m_ba[0][1] * mid[1] + m_ba[0][2] * mid[2],
                m_ba[1][0] * mid[0] + m_ba[1][1] * mid[1] + m_ba[1][2] * mid[2],
                m_ba[2][0] * mid[0] + m_ba[2][1] * mid[1] + m_ba[2][2] * mid[2],
            ];
            for c in 0..3 {
                assert!(
                    (back[c] - sample[c]).abs() < 1e-5,
                    "{a:?}->{b:?}->{a:?} ch {c}: {} vs {}",
                    back[c],
                    sample[c]
                );
            }
            // D65 white [1, 1, 1] stays [1, 1, 1] across every gamut matrix.
            for row in m_ab {
                let sum = row[0] + row[1] + row[2];
                assert!((sum - 1.0).abs() < 1e-5, "{a:?}->{b:?} white row sum {sum}");
            }
        }
        assert!(a.gamut_matrix_to(Primaries::Unknown).is_none());
    }
}

#[test]
fn cicp_maps_rgb_and_refuses_ycbcr_matrix() {
    let cs = Cicp::SRGB.to_color_space(AlphaMode::None).unwrap();
    assert_eq!(cs, ColorSpace::SRGB);
    assert_eq!(cs.to_cicp(), Some(Cicp::SRGB));

    let pq = Cicp::BT2100_PQ_RGB.to_color_space(AlphaMode::None).unwrap();
    assert_eq!(pq, ColorSpace::BT2100_PQ);

    let hlg = Cicp::BT2100_HLG_RGB
        .to_color_space(AlphaMode::None)
        .unwrap();
    assert_eq!(hlg, ColorSpace::BT2100_HLG_SCENE);

    // Non-identity matrix_coefficients (MC=9, BT.2020 YCbCr) cannot become RGB directly!
    let ycbcr_cicp = Cicp::new(9, 16, 9, false);
    assert_eq!(
        *ycbcr_cicp
            .to_color_space(AlphaMode::None)
            .unwrap_err()
            .error(),
        CicpError::NonIdentityMatrix(9)
    );
}

#[test]
fn bit_packing_p010_u10_lsb_rgb565_and_rgb10a2() {
    // P010: 10-bit code in MSBs of u16
    assert_eq!(BitPacking::P010.max_code(SampleType::U16, 0), 1023);
    let packed = BitPacking::P010.pack_code(SampleType::U16, 0, 512);
    assert_eq!(packed, 512 << 6);
    assert_eq!(
        BitPacking::P010.extract_code(SampleType::U16, packed, 0),
        512
    );

    // 10-bit LSB (AVIF/HEVC)
    let lsb = BitPacking::U10_LSB.pack_code(SampleType::U16, 0, 1023);
    assert_eq!(lsb, 1023);
    assert_eq!(
        BitPacking::U10_LSB.extract_code(SampleType::U16, lsb, 0),
        1023
    );

    // RGB565 packed u16
    let mut word565 = 0u32;
    word565 |= BitPacking::RGB565.pack_code(SampleType::U16, 0, 31); // R=31
    word565 |= BitPacking::RGB565.pack_code(SampleType::U16, 1, 42); // G=42
    word565 |= BitPacking::RGB565.pack_code(SampleType::U16, 2, 17); // B=17
    assert_eq!(
        BitPacking::RGB565.extract_code(SampleType::U16, word565, 0),
        31
    );
    assert_eq!(
        BitPacking::RGB565.extract_code(SampleType::U16, word565, 1),
        42
    );
    assert_eq!(
        BitPacking::RGB565.extract_code(SampleType::U16, word565, 2),
        17
    );

    // RGB10A2 packed u32
    let mut word10 = 0u32;
    word10 |= BitPacking::RGB10A2.pack_code(SampleType::U32, 0, 700);
    word10 |= BitPacking::RGB10A2.pack_code(SampleType::U32, 1, 512);
    word10 |= BitPacking::RGB10A2.pack_code(SampleType::U32, 2, 1023);
    word10 |= BitPacking::RGB10A2.pack_code(SampleType::U32, 3, 3);
    assert_eq!(
        BitPacking::RGB10A2.extract_code(SampleType::U32, word10, 0),
        700
    );
    assert_eq!(
        BitPacking::RGB10A2.extract_code(SampleType::U32, word10, 1),
        512
    );
    assert_eq!(
        BitPacking::RGB10A2.extract_code(SampleType::U32, word10, 2),
        1023
    );
    assert_eq!(
        BitPacking::RGB10A2.extract_code(SampleType::U32, word10, 3),
        3
    );

    // SampleLayout with RGB565 validates cleanly as 2 bytes per pixel!
    assert_eq!(SampleLayout::RGB565.bytes_per_pixel(), 2);
    assert!(SampleLayout::RGB565.validate().is_ok());
    assert_eq!(SampleLayout::RGB10A2.bytes_per_pixel(), 4);
    assert!(SampleLayout::RGB10A2.validate().is_ok());
}

#[test]
fn view_chained_lifetimes_and_sample_facts_invalidation() {
    let mut buf = PixelBuf::new(8, 6, ImageSpec::RGBF32_LINEAR).unwrap();
    // Newly zeroed F32 buffer automatically carries UNIT_INTERVAL + FINITE + NON_NEGATIVE facts!
    assert!(buf.facts().float.is_finite());
    assert!(buf.facts().float.is_unit_interval());
    assert!(buf.facts().all_grayscale);

    // Taking mutable view MUST invalidate cached SampleFacts.
    {
        let mut mv = buf.as_view_mut();
        mv.row_as_mut::<f32>(3).unwrap()[6] = 1.75; // pixel (2, 3) R channel
    }
    assert_eq!(buf.facts(), SampleFacts::NONE);

    // Chained crop/sub_rows return slices tied to `'a` (not temporary view!).
    let row_slice: &[u8] = buf
        .as_view()
        .sub_rows(2, 3)
        .unwrap()
        .crop(2, 1, 3, 2)
        .unwrap()
        .row(0);
    let floats: &[f32] = bytemuck::cast_slice(row_slice);
    assert_eq!(floats[0], 1.75);

    // Re-measuring float facts detects >1.0 HDR values accurately.
    let measured = buf.as_view().measure_float_facts();
    assert!(measured.is_finite());
    assert!(measured.contains(FloatFacts::NON_NEGATIVE));
    assert!(!measured.is_unit_interval());
}

#[test]
fn banded_buf_planar_buf_and_dual_layer_hdr() {
    // BandedBuf with tiny 64-byte band cap forces multiple bands and shares PixelSource/PixelSink.
    let mut banded = BandedBuf::new(8, 10, ImageSpec::RGBA8_SRGB, 64).unwrap();
    assert!(banded.bands().len() >= 5);
    assert_eq!(banded.source_width(), 8);
    assert_eq!(banded.source_height(), 10);
    banded.sink_row_mut(7)[0..4].copy_from_slice(&[11, 22, 33, 255]);
    assert_eq!(&banded.source_row(7)[0..4], &[11, 22, 33, 255]);

    // PlanarBuf P010 (Y 10-bit MSB + semi-planar CbCr 4:2:0 10-bit MSB)
    let p010 = [
        PlaneSpec::new(PlaneRole::Y, SampleType::U16).with_packing(BitPacking::P010),
        PlaneSpec::new(PlaneRole::CbCr, SampleType::U16)
            .with_packing(BitPacking::P010)
            .with_subsampling(2, 2),
    ];
    let mut planar = PlanarBuf::new(16, 12, &p010).unwrap();
    assert_eq!(planar.planes()[0].width(), 16);
    assert_eq!(planar.planes()[0].height(), 12);
    assert_eq!(planar.planes()[1].width(), 8);
    assert_eq!(planar.planes()[1].height(), 6);

    let y_val = BitPacking::P010.pack_code(SampleType::U16, 0, 940) as u16;
    planar.plane_view_mut(0).unwrap().row_mut(2)[6..8].copy_from_slice(&y_val.to_ne_bytes());
    assert_eq!(planar.planes()[0].code_u32(3, 2, 0), Some(940));

    // DualLayerHdrBuf (ISO 21496-1 / UltraHDR)
    let base = PixelBuf::new(16, 16, ImageSpec::RGB8_SRGB).unwrap();
    let gain = PixelBuf::new(
        8,
        8,
        ImageSpec::new(
            SampleLayout::GRAY8,
            ColorSpace::Untagged {
                channels: 1,
                alpha: AlphaMode::None,
            },
        ),
    )
    .unwrap();
    let meta = GainMapMetadata::symmetric(2.3);
    let dual = DualLayerHdrBuf::new(base, gain, meta).unwrap();
    assert_eq!(dual.base().width(), 16);
    assert_eq!(dual.gain_map().width(), 8);
    assert!((dual.metadata().alternate_hdr_headroom.to_f64() - 2.3).abs() < 1e-4);

    // InteropTarget checks
    assert!(ImageSpec::RGB8_SRGB.supports_interop(InteropTarget::DynamicImage));
    assert!(!ImageSpec::GRAYF32_LINEAR.supports_interop(InteropTarget::DynamicImage));
    assert!(ImageSpec::GRAYF32_LINEAR.supports_interop(InteropTarget::ImageBuffer));
    assert!(
        !ImageSpec::new(SampleLayout::RGB565, ColorSpace::SRGB)
            .supports_interop(InteropTarget::Imgref)
    );

    // DiffuseWhite + HlgRendering invariants
    assert_eq!(DiffuseWhite::default(), DiffuseWhite::BT2408);
    let hlg_disp = HlgRendering::new(Nits::new(2000.0), 0.01, 1.32).unwrap();
    let cs = ColorSpace::BT2100_HLG_SCENE.with_transfer(Transfer::Hlg(HlgMode::Display(hlg_disp)));
    assert!(cs.is_hdr());
    assert_eq!(cs.to_cicp().unwrap().transfer_characteristics, 18);
    assert_eq!(ChannelOrder::Rgb.channels(), 3);
}
