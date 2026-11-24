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

#[test]
fn gainmap_streaming_row_cache_efficiency_zero_alloc_and_banded_parity() {
    use enough::Unstoppable;
    use std::cell::Cell;
    use zenpixels_convert::{
        AlphaMode, BandedBuf, DualLayerHdrView, GainMapApplier, GainMapApplyConfig,
        GainMapResample, Orientation, PeakSource, PixelSource, SampleLayout, apply_gain_map_with,
        convert_buf,
    };

    let (w, h) = (32u32, 32u32);
    let (gm_w, gm_h) = (8u32, 8u32);

    let mut base = PixelBuf::new(w, h, ImageSpec::rgba8_srgb()).unwrap();
    {
        let mut bv = base.as_view_mut();
        for y in 0..h {
            let row = bv.row_mut(y);
            for x in 0..w as usize {
                row[x * 4] = ((x * 7 + y as usize * 5) % 256) as u8;
                row[x * 4 + 1] = ((x * 11 + y as usize * 3 + 40) % 256) as u8;
                row[x * 4 + 2] = ((x * 13 + y as usize * 9 + 90) % 256) as u8;
                row[x * 4 + 3] = ((x * 17 + 64) % 256) as u8;
            }
        }
    }
    let gm_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let mut gm = PixelBuf::new(gm_w, gm_h, gm_spec.clone()).unwrap();
    for y in 0..gm_h {
        for (x, b) in gm.as_view_mut().row_mut(y).iter_mut().enumerate() {
            *b = ((y as usize * 31 + x * 23 + 17) % 256) as u8;
        }
    }
    let meta = GainMapMetadata::symmetric(3.0);
    let dual = DualLayerHdrView::new(
        base.as_view(),
        gm.as_view(),
        meta,
        None,
        Orientation::Identity,
    )
    .unwrap();

    // 1. Sequential Bilinear pull fetches each ¼-res gain-map row AT MOST ONCE (8 total fetches for 32 output rows)!
    let cfg_bi = GainMapApplyConfig::new(2.5).with_resample(GainMapResample::Bilinear);
    let mut applier = GainMapApplier::for_view(&dual, &cfg_bi).unwrap();
    let bpp = applier.dst_spec().layout.bytes_per_pixel();
    let mut seq_rows: Vec<Vec<u8>> = Vec::new();
    let fetches = Cell::new(0u32);
    let gm_view = gm.as_view();
    for y in 0..h {
        let mut dst = vec![0u8; w as usize * bpp];
        applier
            .apply_row_with_provider(
                y,
                base.as_view().row(y),
                |gy| {
                    fetches.set(fetches.get() + 1);
                    gm_view.row(gy)
                },
                &mut dst,
            )
            .unwrap();
        seq_rows.push(dst);
    }
    assert_eq!(
        fetches.get(),
        gm_h,
        "Bilinear sequential pull must fetch each gain-map row exactly once ({gm_h}), got {}",
        fetches.get()
    );

    // 2. Non-monotonic & shuffled row access (including Bicubic 4-tap!) must match sequential 0-bit
    //    (proving safe active-range cache eviction never clobbers a tap needed by the current row!).
    for resample in [GainMapResample::Bilinear, GainMapResample::Bicubic] {
        let cfg = GainMapApplyConfig::new(2.5).with_resample(resample);
        let ref_buf = apply_gain_map_with(&dual, &cfg, &Unstoppable).unwrap();
        let mut app = GainMapApplier::for_view(&dual, &cfg).unwrap();
        let mut dst = vec![0u8; w as usize * bpp];

        // Reverse order then pseudo-random order
        for y in (0..h).rev() {
            app.apply_row(y, base.as_view().row(y), &gm_view, &mut dst)
                .unwrap();
            assert_eq!(
                &dst[..],
                ref_buf.as_view().row(y),
                "reverse row {y} mismatch for {resample:?}"
            );
        }
        let shuffled = [17u32, 0, 31, 5, 18, 16, 1, 30, 15, 19, 14, 2, 29, 8, 9, 10];
        for y in shuffled {
            app.apply_row(y, base.as_view().row(y), &gm_view, &mut dst)
                .unwrap();
            assert_eq!(
                &dst[..],
                ref_buf.as_view().row(y),
                "shuffled row {y} mismatch for {resample:?}"
            );
        }

        // 3. Zero heap allocations per apply_row call after construction!
        let allocs = allocation_counter::measure(|| {
            for y in 0..h {
                app.apply_row(y, base.as_view().row(y), &gm_view, &mut dst)
                    .unwrap();
            }
        });
        assert_eq!(
            allocs.count_total, 0,
            "GainMapApplier::apply_row ({resample:?}) must perform ZERO heap allocations, did {}",
            allocs.count_total
        );
    }

    // 4. BandedBuf streaming via apply_into matches PixelBuf 0-bit!
    let mut banded = BandedBuf::new(w, h, applier.dst_spec().clone(), 4096).unwrap();
    assert!(banded.bands().len() >= 4);
    applier
        .apply_into(&base, &gm_view, &mut banded, &Unstoppable)
        .unwrap();
    for y in 0..h {
        assert_eq!(banded.source_row(y), &seq_rows[y as usize][..]);
    }

    // 5. Fused dst_spec (RGBAF16, RGBA8 sRGB, and RGB16 PQ when enabled) matches 2-step convert!
    let ref_lin = apply_gain_map_with(&dual, &cfg_bi, &Unstoppable).unwrap();
    for target in [ImageSpec::rgbaf16_linear(), ImageSpec::rgba8_srgb()] {
        let cfg_fused = cfg_bi.clone().with_dst_spec(target.clone());
        let fused = apply_gain_map_with(&dual, &cfg_fused, &Unstoppable).unwrap();
        let two_step = convert_buf(
            &ref_lin.as_view(),
            target.clone(),
            &ConvertPolicy::working_linear(),
            PeakSource::Measured,
        )
        .unwrap();
        for y in 0..h {
            assert_eq!(
                fused.as_view().row(y),
                two_step.as_view().row(y),
                "fused dst_spec {target:?} diverged from 2-step at row {y}"
            );
        }
    }
}

#[test]
fn gainmap_resampling_unit_weights_clamping_and_adversarial_guard() {
    use core::num::NonZeroU32;
    use enough::Unstoppable;
    use zenpixels_convert::{
        AlphaMode, DualLayerHdrView, GainMapApplyConfig, GainMapChannels, GainMapColorSpace,
        GainMapComponent, GainMapResample, Orientation, SampleLayout, SignedFraction,
        UnsignedFraction, apply_gain_map_with,
        gainmap::{
            display_boost_to_headroom_stops, full_reconstruction_headroom_stops, signed_weight,
        },
    };

    // Helper checks
    assert_eq!(display_boost_to_headroom_stops(1.0), 0.0);
    assert_eq!(display_boost_to_headroom_stops(0.5), 0.0);
    assert_eq!(display_boost_to_headroom_stops(f32::NAN), 0.0);
    assert!((display_boost_to_headroom_stops(4.0) - 2.0).abs() < 1e-6);

    let meta = GainMapMetadata::symmetric(2.0);
    assert!((full_reconstruction_headroom_stops(&meta) - 2.0).abs() < 1e-6);
    assert_eq!(signed_weight(&meta, 0.0), Some(0.0));
    assert_eq!(signed_weight(&meta, 1.0), Some(0.5));
    assert_eq!(signed_weight(&meta, 4.0), Some(1.0));
    assert_eq!(signed_weight(&meta, f32::NAN), None);

    // 1. Constant gain map (all 128) on odd prime dimensions must reconstruct constant across ALL pixels
    //    (interior + all 4 borders) for Bilinear, Bicubic, and Mitchell (unit-sum weight proof!).
    let mut base = PixelBuf::new(23, 19, ImageSpec::rgb8_srgb()).unwrap();
    for y in 0..19 {
        base.as_view_mut().row_mut(y).fill(180);
    }
    let gm_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let mut gm_const = PixelBuf::new(7, 5, gm_spec.clone()).unwrap();
    for y in 0..5 {
        gm_const.as_view_mut().row_mut(y).fill(128);
    }
    let dual_const = DualLayerHdrView::new(
        base.as_view(),
        gm_const.as_view(),
        meta,
        None,
        Orientation::Identity,
    )
    .unwrap();
    for resample in [
        GainMapResample::Bilinear,
        GainMapResample::Bicubic,
        GainMapResample::Mitchell,
    ] {
        let out = apply_gain_map_with(
            &dual_const,
            &GainMapApplyConfig::new(2.0).with_resample(resample),
            &Unstoppable,
        )
        .unwrap();
        let ref_v = out.as_view().row_as::<f32>(0).unwrap()[0];
        for y in 0..19 {
            for &v in out.as_view().row_as::<f32>(y).unwrap() {
                assert!(
                    (v - ref_v).abs() < 1e-5,
                    "constant gain map drifted at {resample:?}: {v} vs {ref_v}"
                );
            }
        }
    }

    // 2. Sharp step edge (0 left, 255 right) with Bicubic/Mitchell must clamp overshoot to [0, 1]
    //    so reconstructed luminance stays strictly within [base_at_min, base_at_max]!
    let mut gm_step = PixelBuf::new(8, 8, gm_spec.clone()).unwrap();
    {
        let mut gv = gm_step.as_view_mut();
        for y in 0..8 {
            let r = gv.row_mut(y);
            for (x, px) in r.iter_mut().enumerate().take(8) {
                *px = if x < 4 { 0 } else { 255 };
            }
        }
    }
    let dual_step = DualLayerHdrView::new(
        base.as_view(),
        gm_step.as_view(),
        meta,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let lin_180 = linear_srgb::precise::srgb_to_linear(180.0 / 255.0);
    let k = 1.0 / 64.0;
    let min_expected = lin_180 - 1e-4;
    let max_expected = (lin_180 + k) * 4.0 - k + 1e-4;
    for resample in [GainMapResample::Bicubic, GainMapResample::Mitchell] {
        let out = apply_gain_map_with(
            &dual_step,
            &GainMapApplyConfig::new(2.0).with_resample(resample),
            &Unstoppable,
        )
        .unwrap();
        for y in 0..19 {
            for &v in out.as_view().row_as::<f32>(y).unwrap() {
                assert!(
                    v >= min_expected && v <= max_expected,
                    "{resample:?} overshoot escaped [min, max]: {v} not in [{min_expected}, {max_expected}]"
                );
            }
        }
    }

    // 3. Adversarial ±40-stop metadata must be clamped to ±30 stops (finite output, no Inf/NaN).
    let adv_comp = GainMapComponent {
        min_gain_stops: SignedFraction::new(-40, NonZeroU32::new(1).unwrap()),
        max_gain_stops: SignedFraction::new(40, NonZeroU32::new(1).unwrap()),
        gamma: UnsignedFraction::ONE,
        base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
        alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
    };
    let adv_meta = GainMapMetadata::try_new(
        GainMapChannels::Shared(adv_comp),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(40, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let dual_adv = DualLayerHdrView::new(
        base.as_view(),
        gm_step.as_view(),
        adv_meta,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let out_adv = apply_gain_map_with(
        &dual_adv,
        &GainMapApplyConfig::new(40.0).with_resample(GainMapResample::Bilinear),
        &Unstoppable,
    )
    .unwrap();
    for y in 0..19 {
        for &v in out_adv.as_view().row_as::<f32>(y).unwrap() {
            assert!(
                v.is_finite(),
                "adversarial ±40-stop metadata produced non-finite {v}"
            );
        }
    }
}

#[test]
fn gainmap_row_extractor_streaming_and_contract_33() {
    use zenpixels_convert::{GainMapExtractConfig, GainMapRowExtractor, SampleLayout};

    // Configure grid [0, 4] stops with alternate_hdr_headroom = 2.0 stops, then feed 3.0-stop pixels:
    // contract #33: `metadata()` keeps exact grid, `build_metadata_widening_headroom()` widens
    // alternate_hdr_headroom to ~3.0 stops while leaving min_gain/max_gain untouched!
    let cfg = GainMapExtractConfig::default()
        .with_gain_range_stops(0.0, 4.0)
        .with_headroom_stops(0.0, 2.0)
        .with_offsets(0.0, 0.0);
    let mut ext = GainMapRowExtractor::from_config(&cfg, Primaries::Bt709).unwrap();
    assert_eq!(ext.observed_gain_stops(), None);

    // 4-channel RGBA linear input: pixel 0 = 1x (0 stops), pixel 1 = 8x (3.0 stops)
    let base_rgba = [0.25f32, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 1.0];
    let hdr_rgba = [0.25f32, 0.25, 0.25, 0.5, 2.0, 2.0, 2.0, 1.0];
    let mut gm_row = [0u8; 2];
    ext.encode_row_linear_f32(&base_rgba, &hdr_rgba, 4, SampleLayout::GRAY8, &mut gm_row)
        .unwrap();

    assert_eq!(gm_row[0], 0);
    // 3.0 / 4.0 * 255 = 191.25 -> 191
    assert!((i32::from(gm_row[1]) - 191).abs() <= 1, "got {}", gm_row[1]);

    let (obs_min, obs_max) = ext.observed_gain_stops().unwrap();
    assert!(obs_min.abs() < 1e-4 && (obs_max - 3.0).abs() < 1e-4);

    let exact_meta = ext.metadata();
    assert!((exact_meta.alternate_hdr_headroom.to_f32() - 2.0).abs() < 1e-5);
    assert!((exact_meta.channel(0).max_gain_stops.to_f32() - 4.0).abs() < 1e-5);

    let widened = ext.build_metadata_widening_headroom();
    assert!(
        (widened.alternate_hdr_headroom.to_f32() - 3.0).abs() < 1e-3,
        "contract #33 should widen alternate headroom to ~3.0, got {}",
        widened.alternate_hdr_headroom.to_f32()
    );
    // Grid bounds MUST remain exact (4.0) so quantized bytes stay faithful!
    assert!((widened.channel(0).max_gain_stops.to_f32() - 4.0).abs() < 1e-5);

    // 3-channel Reverse mode (HDR base -> SDR alternate): log2(sdr/hdr) negated -> positive stops
    let rev_cfg = GainMapExtractConfig::default()
        .with_multi_channel(true)
        .with_gain_range_stops(0.0, 3.0)
        .with_headroom_stops(3.0, 0.0)
        .with_offsets(0.0, 0.0);
    let mut rev_ext = GainMapRowExtractor::from_config(&rev_cfg, Primaries::DisplayP3).unwrap();
    let hdr_base = [2.0f32, 1.0, 0.5];
    let sdr_alt = [0.5f32, 0.5, 0.5]; // ratios: 1/4 (-2 stops -> +2), 1/2 (-1 -> +1), 1 (0)
    let mut rev_row = [0u8; 3];
    rev_ext
        .encode_row_linear_f32(&hdr_base, &sdr_alt, 3, SampleLayout::RGB8, &mut rev_row)
        .unwrap();
    assert!(
        (i32::from(rev_row[0]) - 170).abs() <= 1,
        "R: {}",
        rev_row[0]
    ); // 2/3 * 255 = 170
    assert!((i32::from(rev_row[1]) - 85).abs() <= 1, "G: {}", rev_row[1]); // 1/3 * 255 = 85
    assert_eq!(rev_row[2], 0);
}

#[test]
fn gainmap_extract_only_subsampling_content_fit_10bit_alternate_and_stop() {
    use enough::{Stop, StopReason, Unstoppable};
    use zenpixels_convert::{
        ConvertError, DualLayerHdrView, GainMapApplyConfig, GainMapColorSpace, GainMapDownsample,
        GainMapExtractConfig, GainMapResample, Orientation, apply_gain_map_with,
        extract_gain_map_only, gainmap::CONTENT_FIT_MIN_SPAN_STOPS,
    };

    use core::sync::atomic::{AtomicU32, Ordering};
    struct CancelAfter(AtomicU32);
    impl Stop for CancelAfter {
        fn check(&self) -> Result<(), StopReason> {
            let prev = self.0.load(Ordering::Relaxed);
            if prev == 0 {
                Err(StopReason::Cancelled)
            } else {
                self.0.store(prev - 1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    let (w, h) = (32u32, 24u32);
    let mut sdr = PixelBuf::new(w, h, ImageSpec::rgb8_srgb()).unwrap();
    let mut hdr = PixelBuf::new(
        w,
        h,
        ImageSpec::rgbf32_linear().with_white(DiffuseWhite::BT2408),
    )
    .unwrap();
    let lin_128 = linear_srgb::precise::srgb_to_linear(128.0 / 255.0);
    {
        let mut sv = sdr.as_view_mut();
        let mut hv = hdr.as_view_mut();
        for y in 0..h {
            sv.row_mut(y).fill(128);
            let hr = hv.row_as_mut::<f32>(y).unwrap();
            for x in 0..w as usize {
                // Left half = 1x (0 stops), right half = 4x (2.0 stops)
                let boost = if x < (w as usize) / 2 { 1.0 } else { 4.0 };
                hr[x * 3] = lin_128 * boost;
                hr[x * 3 + 1] = lin_128 * boost;
                hr[x * 3 + 2] = lin_128 * boost;
            }
        }
    }

    // 1. Subsampled ¼-res extraction (Box, Bilinear, Point) -> 8x6 gain map + roundtrip apply!
    for ds in [
        GainMapDownsample::Box,
        GainMapDownsample::Bilinear,
        GainMapDownsample::Point,
    ] {
        let cfg = GainMapExtractConfig::ultrahdr().with_downsample(ds);
        let (gm_buf, meta) =
            extract_gain_map_only(&sdr.as_view(), &hdr.as_view(), &cfg, &Unstoppable).unwrap();
        assert_eq!((gm_buf.width(), gm_buf.height()), (8, 6));
        let dual = DualLayerHdrView::new(
            sdr.as_view(),
            gm_buf.as_view(),
            meta,
            None,
            Orientation::Identity,
        )
        .unwrap();
        let rec = apply_gain_map_with(
            &dual,
            &GainMapApplyConfig::new(meta.alternate_hdr_headroom.to_f32())
                .with_resample(GainMapResample::Bilinear),
            &Unstoppable,
        )
        .unwrap();
        // Far left (x=0) should be ~1x, far right (x=31) should be ~4x!
        let r0 = rec.as_view().row_as::<f32>(12).unwrap();
        assert!((r0[0] - lin_128).abs() < 0.02, "{ds:?} left={}", r0[0]);
        assert!(
            (r0[31 * 3] - lin_128 * 4.0).abs() < 0.05,
            "{ds:?} right={}",
            r0[31 * 3]
        );
    }

    // 2. 2-pass content_fit narrows a wide [0, 5]-stop grid down to the observed ~[0, 2]-stop range!
    let cfg_fit = GainMapExtractConfig::default()
        .with_gain_range_stops(0.0, 5.0)
        .with_headroom_stops(0.0, 5.0)
        .with_offsets(0.0, 0.0)
        .with_content_fit(true);
    let (_gm_fit, meta_fit) =
        extract_gain_map_only(&sdr.as_view(), &hdr.as_view(), &cfg_fit, &Unstoppable).unwrap();
    assert!(
        (meta_fit.channel(0).min_gain_stops.to_f32() - 0.0).abs() < 1e-3,
        "min={}",
        meta_fit.channel(0).min_gain_stops.to_f32()
    );
    assert!(
        (meta_fit.channel(0).max_gain_stops.to_f32() - 2.0).abs() < 1e-2,
        "content_fit should narrow 5.0 grid to ~2.0 stops, got {}",
        meta_fit.channel(0).max_gain_stops.to_f32()
    );

    // Uniform image with content_fit enforces CONTENT_FIT_MIN_SPAN_STOPS!
    let mut hdr_flat = PixelBuf::new(
        8,
        8,
        ImageSpec::rgbf32_linear().with_white(DiffuseWhite::BT2408),
    )
    .unwrap();
    let mut sdr_flat = PixelBuf::new(8, 8, ImageSpec::rgb8_srgb()).unwrap();
    for y in 0..8 {
        sdr_flat.as_view_mut().row_mut(y).fill(128);
        hdr_flat
            .as_view_mut()
            .row_as_mut::<f32>(y)
            .unwrap()
            .fill(lin_128);
    }
    let (_, meta_flat) = extract_gain_map_only(
        &sdr_flat.as_view(),
        &hdr_flat.as_view(),
        &cfg_fit,
        &Unstoppable,
    )
    .unwrap();
    let span =
        meta_flat.channel(0).max_gain_stops.to_f32() - meta_flat.channel(0).min_gain_stops.to_f32();
    assert!(
        span >= CONTENT_FIT_MIN_SPAN_STOPS - 1e-5,
        "uniform content_fit span {span} must be >= {CONTENT_FIT_MIN_SPAN_STOPS}"
    );

    // 3. 10-bit BitPacking::Lsb { bits: 10 } U16 gain map round-trip!
    let cfg_10 = GainMapExtractConfig::default()
        .with_sample_storage(SampleType::U16, BitPacking::Lsb { bits: 10 })
        .with_gain_range_stops(0.0, 2.5)
        .with_headroom_stops(0.0, 2.5);
    let (gm_10, meta_10) =
        extract_gain_map_only(&sdr.as_view(), &hdr.as_view(), &cfg_10, &Unstoppable).unwrap();
    assert_eq!(gm_10.spec().layout.sample, SampleType::U16);
    assert_eq!(gm_10.spec().layout.packing, BitPacking::Lsb { bits: 10 });
    // Verify top 6 bits are strictly zero on every 10-bit code!
    for y in 0..h {
        for &code in gm_10.as_view().row_as::<u16>(y).unwrap() {
            assert!(code <= 1023, "10-bit Lsb gain code overflowed: {code}");
        }
    }
    let dual_10 = DualLayerHdrView::new(
        sdr.as_view(),
        gm_10.as_view(),
        meta_10,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let rec_10 =
        apply_gain_map_with(&dual_10, &GainMapApplyConfig::new(2.5), &Unstoppable).unwrap();
    let p_right = rec_10.as_view().row_as::<f32>(0).unwrap()[31 * 3];
    assert!(
        (p_right - lin_128 * 4.0).abs() < 0.005,
        "10-bit gain map should reconstruct within 0.005, got {p_right} vs {}",
        lin_128 * 4.0
    );

    // 4. GainMapColorSpace::Alternate (sRGB base + Display P3 HDR alternate)!
    let p3_hdr_spec = ImageSpec::rgbf32_linear()
        .with_primaries(Primaries::DisplayP3)
        .with_white(DiffuseWhite::BT2408);
    let mut hdr_p3 = PixelBuf::new(8, 8, p3_hdr_spec.clone()).unwrap();
    {
        let mut hv = hdr_p3.as_view_mut();
        for y in 0..8 {
            let r = hv.row_as_mut::<f32>(y).unwrap();
            for x in 0..8 {
                // Pure Display P3 saturated HDR red (outside sRGB gamut!)
                r[x * 3] = 2.0;
                r[x * 3 + 1] = 0.25;
                r[x * 3 + 2] = 0.25;
            }
        }
    }
    let cfg_alt = GainMapExtractConfig::default()
        .with_multi_channel(true)
        .with_application_space(GainMapColorSpace::Alternate)
        .with_gain_range_stops(-1.0, 4.0)
        .with_headroom_stops(0.0, 4.0);
    let (gm_alt, meta_alt) = extract_gain_map_only(
        &sdr_flat.as_view(),
        &hdr_p3.as_view(),
        &cfg_alt,
        &Unstoppable,
    )
    .unwrap();
    let dual_alt = DualLayerHdrView::new(
        sdr_flat.as_view(),
        gm_alt.as_view(),
        meta_alt,
        Some(&p3_hdr_spec.color),
        Orientation::Identity,
    )
    .unwrap();
    let rec_alt =
        apply_gain_map_with(&dual_alt, &GainMapApplyConfig::new(4.0), &Unstoppable).unwrap();
    assert!(matches!(
        rec_alt.spec().color,
        ColorSpace::Rgb {
            primaries: Primaries::DisplayP3,
            transfer: Transfer::Linear,
            ..
        }
    ));
    let px_alt = rec_alt.as_view().row_as::<f32>(0).unwrap();
    assert!(
        (px_alt[0] - 2.0).abs() < 0.04
            && (px_alt[1] - 0.25).abs() < 0.02
            && (px_alt[2] - 0.25).abs() < 0.02,
        "Alternate P3 gain map reconstructed {px_alt:?}, expected [2.0, 0.25, 0.25]"
    );

    // 5. Cooperative enough::Stop cancellation on both apply and extract!
    let err_ext = extract_gain_map_only(
        &sdr.as_view(),
        &hdr.as_view(),
        &GainMapExtractConfig::default(),
        &CancelAfter(AtomicU32::new(3)),
    )
    .unwrap_err();
    assert_eq!(
        *err_ext.error(),
        ConvertError::Cancelled(StopReason::Cancelled)
    );

    let err_app = apply_gain_map_with(
        &dual_10,
        &GainMapApplyConfig::new(2.5),
        &CancelAfter(AtomicU32::new(3)),
    )
    .unwrap_err();
    assert_eq!(
        *err_app.error(),
        ConvertError::Cancelled(StopReason::Cancelled)
    );
}
