//! Allocation gates (zero-alloc `RowPlan::convert_row`, O(1) streaming HDR peak
//! measurement with ZERO full-image F32 temp buffer), `BandedBuf` parity,
//! `DualLayerHdr` ISO 21496-1 gain map roundtrips, and YCbCr P010 reconstruction.

use zenpixels_convert::{
    BitPacking, ColorSpace, ConvertPolicy, DiffuseWhite, GainMapMetadata, ImageSpec, PixelBuf,
    PlanarBuf, PlaneRole, PlaneSpec, Primaries, RowPlan, SampleType, SignalRange, Transfer,
    YcbcrMatrix, apply_gain_map, extract_gain_map, ycbcr_to_rgb_f32,
};

#[test]
fn row_plan_convert_row_allocates_zero_bytes() {
    let plan = RowPlan::build(
        ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let src = vec![128u8; 256 * 3];
    let mut dst = vec![0u8; 256 * 3];

    let info = allocation_counter::measure(|| {
        for y in 0..8 {
            plan.convert_row(&src, &mut dst, 256, y).unwrap();
        }
    });
    assert_eq!(
        info.count_total, 0,
        "RowPlan::convert_row must perform ZERO heap allocations (uses 1 KiB stack StripBuf!)"
    );
}

#[cfg(all(feature = "hdr-transfers", feature = "bt2446a"))]
#[test]
fn hdr_to_sdr_measured_peak_streams_without_full_image_allocation_and_matches_banded() {
    use zenpixels_convert::{
        BandedBuf, ContentLightLevel, ConvertError, HdrMetadata, ImagePlan, LightLevelMethod,
        MasteringDisplay, Nits, PeakSource, PixelSink, PixelSource, measure_cll,
    };
    let (w, h) = (128u32, 128u32); // 128x128 RGBF32 would be 196,608 B
    let mut hdr_buf = PixelBuf::new(w, h, ImageSpec::rgb16_bt2100_pq()).unwrap();
    let mut hdr_banded = BandedBuf::new(w, h, ImageSpec::rgb16_bt2100_pq(), 16 * 1024).unwrap();
    assert!(hdr_banded.bands().len() >= 4);

    // Fill both with identical HDR PQ ramp (peak ~1000 nits ≈ PQ code 0.7518 * 65535 ≈ 49270)
    for y in 0..h {
        let row_dst = hdr_buf.as_view_mut().row_mut(y).len();
        let mut tmp = vec![0u16; row_dst / 2];
        for x in 0..w as usize {
            let v = (((x * 380 + y as usize * 15) % 49270) as u16).max(500);
            tmp[x * 3] = v;
            tmp[x * 3 + 1] = v / 2;
            tmp[x * 3 + 2] = v / 4;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&tmp);
        hdr_buf.as_view_mut().row_mut(y).copy_from_slice(bytes);
        hdr_banded.sink_row_mut(y).copy_from_slice(bytes);
    }

    let policy = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0));
    let plan = ImagePlan::build(
        ImageSpec::rgb16_bt2100_pq(),
        ImageSpec::rgb8_srgb(),
        policy.clone(),
        PeakSource::Measured,
    )
    .unwrap();

    let mut out_buf = PixelBuf::new(w, h, ImageSpec::rgb8_srgb()).unwrap();
    let mut out_banded = BandedBuf::new(w, h, ImageSpec::rgb8_srgb(), 8 * 1024).unwrap();

    // Measure allocations during execute_into (PeakSource::Measured on PQ U16):
    // MUST NOT allocate a full-image F32 intermediate (196 KiB)!
    let alloc_info = allocation_counter::measure(|| {
        plan.execute_into(&hdr_buf, &mut out_buf).unwrap();
    });
    assert!(
        alloc_info.bytes_max < 32 * 1024,
        "streaming HDR peak measurement allocated {} bytes (must stay O(1), well under 32 KiB!)",
        alloc_info.bytes_max
    );

    // BandedBuf produces 0-bit identical pixels to contiguous PixelBuf!
    plan.execute_into(&hdr_banded, &mut out_banded).unwrap();
    for y in 0..h {
        assert_eq!(out_buf.as_view().row(y), out_banded.source_row(y));
    }

    // PeakSource::Declared uses MaxCLL or ST 2086 mastering display, refuses if both absent.
    let plan_decl = ImagePlan::build(
        ImageSpec::rgb16_bt2100_pq(),
        ImageSpec::rgb8_srgb(),
        policy,
        PeakSource::Declared,
    )
    .unwrap();
    let err = plan_decl.execute(&hdr_buf).unwrap_err();
    assert_eq!(*err.error(), ConvertError::MissingDeclaredPeak);

    let with_cll = hdr_buf.clone().with_hdr(HdrMetadata {
        cll: Some(ContentLightLevel::new(1000, 250)),
        mastering: Some(MasteringDisplay::HDR10_REFERENCE),
    });
    assert!(plan_decl.execute(&with_cll).is_ok());

    // CLL measurement agrees with ~1000-nit peak!
    let cll = measure_cll(
        &hdr_buf,
        DiffuseWhite::BT2408,
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    assert!(
        (900..=1100).contains(&cll.max_content_light_level),
        "measured MaxCLL should be ~1000 nits, got {}",
        cll.max_content_light_level
    );
}

#[test]
fn dual_layer_hdr_gain_map_extract_and_apply_roundtrip() {
    let (w, h) = (16u32, 12u32);
    let mut sdr = PixelBuf::new(w, h, ImageSpec::rgb8_srgb()).unwrap();
    let mut hdr = PixelBuf::new(
        w,
        h,
        ImageSpec::rgbf32_linear().with_white(DiffuseWhite::BT2408),
    )
    .unwrap();

    // Base = sRGB mid-gray (~0.214 linear), HDR = 4x brighter (~0.856 linear = 2.0 stops boost)
    for y in 0..h {
        for px in sdr.as_view_mut().row_mut(y).as_chunks_mut::<3>().0 {
            px.copy_from_slice(&[128, 128, 128]);
        }
        let lin_sdr = linear_srgb::precise::srgb_to_linear(128.0 / 255.0);
        for px in hdr
            .as_view_mut()
            .row_as_mut::<f32>(y)
            .unwrap()
            .as_chunks_mut::<3>()
            .0
        {
            px.copy_from_slice(&[lin_sdr * 4.0, lin_sdr * 4.0, lin_sdr * 4.0]);
        }
    }

    let meta = GainMapMetadata::symmetric(2.5);
    let dual = extract_gain_map(&sdr.as_view(), &hdr.as_view(), meta).unwrap();

    // At headroom = 0.0 stops -> exact SDR base luminance!
    let at_sdr = apply_gain_map(&dual, 0.0).unwrap();
    let expected_sdr = linear_srgb::precise::srgb_to_linear(128.0 / 255.0);
    let got_sdr = at_sdr.as_view().row_as::<f32>(0).unwrap()[0];
    assert!(
        (got_sdr - expected_sdr).abs() < 1e-4,
        "at headroom 0: got {got_sdr}, want {expected_sdr}"
    );

    // At headroom = 2.5 stops -> reconstructed 4x HDR luminance!
    let at_hdr = apply_gain_map(&dual, 2.5).unwrap();
    let got_hdr = at_hdr.as_view().row_as::<f32>(0).unwrap()[0];
    assert!(
        (got_hdr - expected_sdr * 4.0).abs() < 0.02,
        "at full headroom: got {got_hdr}, want {}",
        expected_sdr * 4.0
    );
}

#[test]
fn ycbcr_p010_and_8bit_420_to_rgb_f32() {
    // 10-bit P010 semi-planar 4:2:0 narrow-range BT.2020 pure white: Y=940, Cb=512, Cr=512 -> RGB=(1,1,1)
    let specs = [
        PlaneSpec::new(PlaneRole::Y, SampleType::U16).with_packing(BitPacking::P010),
        PlaneSpec::new(PlaneRole::CbCr, SampleType::U16)
            .with_packing(BitPacking::P010)
            .with_subsampling(2, 2),
    ];
    let mut planar = PlanarBuf::new(8, 6, &specs).unwrap();
    let y_packed = BitPacking::P010.pack_code(SampleType::U16, 0, 940) as u16;
    let c_packed = BitPacking::P010.pack_code(SampleType::U16, 0, 512) as u16;
    for y in 0..6 {
        let mut pv = planar.plane_view_mut(0).unwrap();
        let r = pv.row_mut(y);
        for x in 0..8 {
            r[x * 2..x * 2 + 2].copy_from_slice(&y_packed.to_ne_bytes());
        }
    }
    for y in 0..3 {
        let mut pv = planar.plane_view_mut(1).unwrap();
        let r = pv.row_mut(y);
        for x in 0..4 {
            r[x * 4..x * 4 + 2].copy_from_slice(&c_packed.to_ne_bytes());
            r[x * 4 + 2..x * 4 + 4].copy_from_slice(&c_packed.to_ne_bytes());
        }
    }

    let rgb = ycbcr_to_rgb_f32(
        &planar.as_view(),
        YcbcrMatrix::Bt2020,
        SignalRange::Narrow,
        Primaries::Bt2020,
        Transfer::Pq,
    )
    .unwrap();
    let px = rgb.as_view().row_as::<f32>(0).unwrap();
    assert!((px[0] - 1.0).abs() < 1e-4 && (px[1] - 1.0).abs() < 1e-4 && (px[2] - 1.0).abs() < 1e-4);
    assert!(matches!(
        rgb.spec().color,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Pq,
            ..
        }
    ));
}
