//! Contract tests for HDR photometry, scene-linear tone mapping, native-primaries CLL & histogram,
//! YCbCr (odd-origin crop phase, Identity GBR, monochrome, exact narrow white, NV12/NV21/I420),
//! independent 8-orientation coordinate oracle, and ISO 21496-1 / UltraHDR gain maps.
//!
//! Directly gates audit findings:
//! - B16 (HLG separate src/dst white + relative white epsilon)
//! - B17 (scene-linear HDR tone mapping dispatch + identity HDR metadata retention)
//! - B18 (native-primaries MaxRGB BT.709/P3 + H.274 ceiling 100.4->101)
//! - B24 / B39 (5x3 4:2:0 asymmetric Cb/Cr odd-origin crop == full->crop, NV12/NV21/I420, GBR, Mono, exact white)
//! - B25 / D06 (independent 8-orientation coordinate oracle across skinny/tile/padded canary buffers + origin)
//! - B28 / B29 / B30 / B31 / B40 (gain-map analytic oracle, 31-stop exactness, 128->4 64-tap inclusive, gamma!=1 SIMD==scalar 0-bit)
//! - B41 (LightLevelHistogram zero_count, nearest-rank median, fallible try_new)

extern crate alloc;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

use zenpixels_convert::{
    AlphaMode, At, Cicp, ColorOrigin, ColorSpace, ContentLightLevel, ConvertError, ConvertPolicy,
    DiffuseWhite, HdrMetadata, ImagePlan, LightLevelHistogram, LightLevelMethod, MasteringDisplay,
    Nits, Orientation, PeakSource, PixelBuf, PixelFormat, PixelRect, PlanarReconstructionOptions,
    PlanarView, PlaneFormat, PlaneRole, PlaneView, PreparedMapper, Primaries, Requirements,
    SampleLayout, SampleType, SignalRange, StripMapper, ToneMapContext, ToneMapper, Transfer,
    YcbcrMatrix, apply_orientation, apply_orientation_into, convert_to_buf,
    convert_to_buf_with_peak, measure_content_light_level, measure_histogram, measure_peak,
    reconstruct_planar_f32, reconstruct_planar_into,
};

#[cfg(feature = "hdr-transfers")]
#[test]
fn b16_hlg_white_anchor_no_double_scale_and_relative_white_epsilon() {
    use zenpixels_convert::{ChannelOrder, HlgMode, HlgRendering, HlgRenderingParams, RowPlan};
    let hlg_1000 = HlgRendering::new(HlgRenderingParams {
        peak: Nits::new_positive(1000.0).unwrap(),
        black_nits: 0.0,
        system_gamma: 1.2,
    })
    .unwrap();
    let src_lin = PixelFormat::RGBF32_LINEAR
        .with_primaries(Primaries::Bt2020)
        .with_white(DiffuseWhite::new(100.0).unwrap());
    let dst_hlg_100 = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::BT2100_HLG_SCENE
            .with_transfer(Transfer::Hlg(HlgMode::Display(hlg_1000)))
            .with_white(DiffuseWhite::new(100.0).unwrap()),
    );
    let dst_hlg_200 = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::BT2100_HLG_SCENE
            .with_transfer(Transfer::Hlg(HlgMode::Display(hlg_1000)))
            .with_white(DiffuseWhite::new(200.0).unwrap()),
    );
    let dst_hlg_200_bgr = PixelFormat::new(
        SampleLayout::new(SampleType::F32, ChannelOrder::Bgr),
        dst_hlg_200.color.clone(),
    );

    let one = [1.0f32, 1.0, 1.0];
    let mut o100 = [0f32; 3];
    let mut o200 = [0f32; 3];
    let mut o200_bgr = [0f32; 3];
    RowPlan::build(
        src_lin.clone(),
        dst_hlg_100,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap()
    .convert_row(
        bytemuck::cast_slice(&one),
        bytemuck::cast_slice_mut(&mut o100),
        1,
    )
    .unwrap();
    RowPlan::build(
        src_lin.clone(),
        dst_hlg_200,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap()
    .convert_row(
        bytemuck::cast_slice(&one),
        bytemuck::cast_slice_mut(&mut o200),
        1,
    )
    .unwrap();
    RowPlan::build(src_lin, dst_hlg_200_bgr, &ConvertPolicy::permissive(), None)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&one),
            bytemuck::cast_slice_mut(&mut o200_bgr),
            1,
        )
        .unwrap();

    for (label, v) in [("w=100", o100), ("w=200", o200), ("w=200 BGR", o200_bgr)] {
        for (c, &val) in v.iter().enumerate() {
            assert!(
                (val - 0.629_620_3).abs() < 1e-3,
                "HLG {label} ch {c} got {val}, expected ~0.6296203"
            );
        }
    }
    assert!(
        (o100[0] - o200[0]).abs() < 1e-5,
        "w=100 {} vs w=200 {}",
        o100[0],
        o200[0]
    );

    // B16: Tiny white ratio 0.001 -> 0.00105 must apply exact physical scale 0.001/0.00105 ~= 0.95238!
    let src_small_w = PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::new(0.001).unwrap());
    let dst_small_w = PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::new(0.00105).unwrap());
    let mut o_small = [0f32; 3];
    RowPlan::build(src_small_w, dst_small_w, &ConvertPolicy::permissive(), None)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&one),
            bytemuck::cast_slice_mut(&mut o_small),
            1,
        )
        .unwrap();
    let want_ratio = 0.001_f32 / 0.00105_f32;
    assert!(
        (o_small[0] - want_ratio).abs() < 1e-5,
        "relative white 0.001->0.00105 got {}, expected {want_ratio}",
        o_small[0]
    );
}

#[test]
fn b17_scene_linear_hdr_triggers_tone_mapper_and_identity_retains_hdr_metadata() {
    #[derive(Debug)]
    struct SpyMapper {
        calls: Arc<AtomicUsize>,
    }
    #[derive(Debug)]
    struct SpyStrip {
        calls: Arc<AtomicUsize>,
    }
    impl StripMapper for SpyStrip {
        fn map_strip(&self, input: &[f32], output: &mut [f32]) {
            // ToneMapper receives source-peak-normalized linear RGB in [0, 1] where 1.0 == source_peak.
            // Linear pass-through maps [812, 203, 0] nits -> [1.0, 0.25, 0.0] -> sRGB8 [255, 137, 0]!
            self.calls.fetch_add(1, Ordering::Relaxed);
            for (o, &v) in output.iter_mut().zip(input.iter()) {
                *o = v.clamp(0.0, 1.0);
            }
        }
    }
    impl ToneMapper for SpyMapper {
        fn requirements(&self) -> Requirements {
            Requirements::SOURCE_PEAK_ONLY
        }
        fn prepare(&self, _ctx: &ToneMapContext) -> Result<PreparedMapper, At<ConvertError>> {
            Ok(PreparedMapper::Streaming(Box::new(SpyStrip {
                calls: Arc::clone(&self.calls),
            })))
        }
        fn name(&self) -> &'static str {
            "spy-quarter"
        }
        fn working_primaries_for(&self, target: Primaries) -> Primaries {
            target
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let src_hdr_lin = PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::BT2408);
    let hdr_meta = HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(812, 400)),
        mastering: Some(MasteringDisplay::new(
            [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            [0.3127, 0.3290],
            0.005,
            1000.0,
        )),
    };
    let src_buf = PixelBuf::from_typed(vec![4.0f32, 1.0, 0.0], 1, 1, src_hdr_lin.clone())
        .unwrap()
        .with_hdr(hdr_meta);

    // 1. Scene-linear HDR [4,1,0] @ 203 nits with Declared CLL 812 -> sRGB8 MUST invoke tone mapper!
    let pol = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new_positive(203.0).unwrap())
        .with_tone_mapper(Arc::new(SpyMapper {
            calls: Arc::clone(&calls),
        }));
    let out = convert_to_buf_with_peak(
        &src_buf.as_view(),
        PixelFormat::RGB8_SRGB,
        &pol,
        PeakSource::Declared,
    )
    .unwrap();
    assert!(
        calls.load(Ordering::Relaxed) >= 1,
        "scene-linear HDR input must actually invoke the tone mapper!"
    );
    // Spy scales [4, 1, 0] * 0.25 -> [1.0, 0.25, 0.0] -> sRGB8 [255, ~137, 0] (NOT clipped [255,255,0]!).
    let px = out.as_view().row(0);
    assert_eq!(px[0], 255);
    assert!(
        px[1] >= 134 && px[1] <= 140,
        "expected tone-mapped G~137, got {px:?}"
    );
    assert_eq!(px[2], 0);
    assert!(out.hdr().content_light_level.is_none());

    // 2. Identity ImagePlan on HDR linear F32 MUST preserve CLL & MDCV metadata!
    let id_plan = ImagePlan::build(
        src_hdr_lin.clone(),
        src_hdr_lin,
        &ConvertPolicy::lossless(),
        PeakSource::Measured,
    )
    .unwrap();
    let copied = id_plan.execute(&src_buf).unwrap();
    assert_eq!(copied.hdr(), hdr_meta);

    // 3. Ordinary SDR linear F32 [0.5, 0.25, 0.0] (no HDR metadata, peak <= ceiling) MUST NOT invoke mapper!
    let sdr_calls = Arc::new(AtomicUsize::new(0));
    let sdr_pol =
        ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new_positive(203.0).unwrap())
            .with_tone_mapper(Arc::new(SpyMapper {
                calls: Arc::clone(&sdr_calls),
            }));
    let sdr_buf =
        PixelBuf::from_typed(vec![0.5f32, 0.25, 0.0], 1, 1, PixelFormat::RGBF32_LINEAR).unwrap();
    let _ = convert_to_buf(&sdr_buf.as_view(), PixelFormat::RGB8_SRGB, &sdr_pol).unwrap();
    assert_eq!(
        sdr_calls.load(Ordering::Relaxed),
        0,
        "ordinary SDR linear F32 must NOT invoke HDR tone mapper"
    );
}

#[test]
fn b18_b41_native_primaries_cll_ceiling_and_histogram_ranks() {
    // B18: BT.709 linear red [4,0,0] @ 203 nits -> native MaxRGB = 812/812 (NOT 509 from BT.2020 conversion!).
    let bt709_red = PixelBuf::from_typed(
        vec![4.0f32, 0.0, 0.0],
        1,
        1,
        PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::BT2408),
    )
    .unwrap();
    let cll709 = measure_content_light_level(
        &bt709_red,
        DiffuseWhite::BT2408,
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    assert_eq!(cll709.max_content_light_level, 812);
    assert_eq!(cll709.max_frame_average_light_level, 812);

    // Display P3 linear red [1,0,0] @ 203 nits -> native MaxRGB = 203/203 (NOT 153!).
    let p3_red = PixelBuf::from_typed(
        vec![1.0f32, 0.0, 0.0],
        1,
        1,
        PixelFormat::RGBF32_LINEAR
            .with_primaries(Primaries::DisplayP3)
            .with_white(DiffuseWhite::BT2408),
    )
    .unwrap();
    let cll_p3 = measure_content_light_level(
        &p3_red,
        DiffuseWhite::BT2408,
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    assert_eq!(cll_p3.max_content_light_level, 203);
    assert_eq!(cll_p3.max_frame_average_light_level, 203);

    // H.274 upper-bound ceiling: 100.4 nits -> 101!
    let frac = PixelBuf::from_typed(
        vec![100.4f32, 0.0, 0.0],
        1,
        1,
        PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::new(1.0).unwrap()),
    )
    .unwrap();
    let cll_frac = measure_content_light_level(
        &frac,
        DiffuseWhite::new(1.0).unwrap(),
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    assert_eq!(cll_frac.max_content_light_level, 101);
    let peak = measure_peak(&frac, DiffuseWhite::new(1.0).unwrap(), None).unwrap();
    assert!((peak - 100.4).abs() < 1e-3);

    // B41: LightLevelHistogram zero_count, nearest-rank median, and fallible `new`!
    let mut h_zeros = LightLevelHistogram::new(LightLevelMethod::MaxRgb).unwrap();
    h_zeros.record(0.0);
    h_zeros.record(0.0);
    h_zeros.record(1.0);
    assert_eq!(h_zeros.quantile_nits(0.0), 0.0);
    assert_eq!(h_zeros.quantile_nits(0.5), 0.0);
    assert!((h_zeros.quantile_nits(1.0) - 1.0).abs() < 0.05);

    let mut h_med = LightLevelHistogram::new(LightLevelMethod::MaxRgb).unwrap();
    h_med.record(0.0);
    h_med.record(1.0);
    h_med.record(100.0);
    let p50 = h_med.quantile_nits(0.5);
    assert!(
        (p50 - 1.0).abs() < 0.15,
        "median of [0, 1, 100] must be ~1.0 nits, got {p50}"
    );

    let hist_buf = PixelBuf::from_typed(
        vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0],
        3,
        1,
        PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::new(1.0).unwrap()),
    )
    .unwrap();
    let measured_h = measure_histogram(
        &hist_buf,
        DiffuseWhite::new(1.0).unwrap(),
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    assert_eq!(measured_h.quantile_nits(0.0), 0.0);
    assert!((measured_h.quantile_nits(0.5) - 1.0).abs() < 0.15);
}

#[test]
fn b24_b39_ycbcr_odd_crop_phase_nv12_nv21_i420_gbr_monochrome_and_exact_white() {
    let y_spec = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    let cb_spec = PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2);
    let cr_spec = PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2);
    let a_spec = PlaneFormat::new(PlaneRole::Alpha, SampleType::U8);

    let y_plane: [u8; 18] = [
        40, 80, 120, 160, 200, 0xEE, 50, 90, 130, 170, 210, 0xEE, 60, 100, 140, 180, 220, 0xEE,
    ];
    let cb_plane: [u8; 8] = [64, 140, 210, 0xDD, 96, 176, 48, 0xDD];
    let cr_plane: [u8; 8] = [200, 110, 40, 0xDD, 180, 72, 224, 0xDD];
    let a_plane: [u8; 18] = [
        255, 200, 150, 100, 50, 0xCC, 240, 190, 140, 90, 40, 0xCC, 230, 180, 130, 80, 30, 0xCC,
    ];

    let bt709_narrow_alpha = ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Bt709,
        range: SignalRange::Narrow,
        alpha: AlphaMode::Straight,
        white: None,
    };
    let bt709_narrow = ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Bt709,
        range: SignalRange::Narrow,
        alpha: AlphaMode::None,
        white: None,
    };

    let full = PlanarView::new(
        5,
        3,
        &[
            PlaneView::new(&y_plane, 5, 3, 6, y_spec).unwrap(),
            PlaneView::new(&cb_plane, 3, 2, 4, cb_spec).unwrap(),
            PlaneView::new(&cr_plane, 3, 2, 4, cr_spec).unwrap(),
            PlaneView::new(&a_plane, 5, 3, 6, a_spec).unwrap(),
        ],
    )
    .unwrap()
    .with_color(bt709_narrow_alpha.clone())
    .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());

    let full_rgb = reconstruct_planar_f32(&full, &PlanarReconstructionOptions::nearest()).unwrap();
    let cropped_view = full.crop(PixelRect::new(1, 1, 3, 2)).unwrap();
    let cropped_rgb =
        reconstruct_planar_f32(&cropped_view, &PlanarReconstructionOptions::nearest()).unwrap();
    let full_then_crop = full_rgb.as_view().crop(PixelRect::new(1, 1, 3, 2)).unwrap();
    for row in 0..2 {
        assert_eq!(
            cropped_rgb.as_view().row(row),
            full_then_crop.row(row),
            "odd-origin (1,1) 4:2:0 crop phase mismatch on row {row}"
        );
    }

    // B39: NV12 (CbCr) and NV21 (CrCb) produce bit-exact same RGB as I420 on the same asymmetric frame!
    let mut cbcr = [0u8; 3 * 2 * 2];
    let mut crcb = [0u8; 3 * 2 * 2];
    for cy in 0..2 {
        for cx in 0..3 {
            let cb = cb_plane[cy * 4 + cx];
            let cr = cr_plane[cy * 4 + cx];
            cbcr[(cy * 3 + cx) * 2] = cb;
            cbcr[(cy * 3 + cx) * 2 + 1] = cr;
            crcb[(cy * 3 + cx) * 2] = cr;
            crcb[(cy * 3 + cx) * 2 + 1] = cb;
        }
    }
    let nv12 = PlanarView::new(
        5,
        3,
        &[
            PlaneView::new(&y_plane, 5, 3, 6, y_spec).unwrap(),
            PlaneView::new(
                &cbcr,
                3,
                2,
                6,
                PlaneFormat::new(PlaneRole::CbCr, SampleType::U8).with_subsampling(2, 2),
            )
            .unwrap(),
            PlaneView::new(&a_plane, 5, 3, 6, a_spec).unwrap(),
        ],
    )
    .unwrap()
    .with_color(bt709_narrow_alpha.clone())
    .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let nv21 = PlanarView::new(
        5,
        3,
        &[
            PlaneView::new(&y_plane, 5, 3, 6, y_spec).unwrap(),
            PlaneView::new(
                &crcb,
                3,
                2,
                6,
                PlaneFormat::new(PlaneRole::CrCb, SampleType::U8).with_subsampling(2, 2),
            )
            .unwrap(),
            PlaneView::new(&a_plane, 5, 3, 6, a_spec).unwrap(),
        ],
    )
    .unwrap()
    .with_color(bt709_narrow_alpha)
    .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let rgb_nv12 = reconstruct_planar_f32(&nv12, &PlanarReconstructionOptions::nearest()).unwrap();
    let rgb_nv21 = reconstruct_planar_f32(&nv21, &PlanarReconstructionOptions::nearest()).unwrap();
    for row in 0..3 {
        assert_eq!(rgb_nv12.as_view().row(row), full_rgb.as_view().row(row));
        assert_eq!(rgb_nv21.as_view().row(row), full_rgb.as_view().row(row));
    }

    // Exact narrow white Y=235, Cb=128, Cr=128 -> exact 1.0f32 via zero-alloc `reconstruct_planar_into`!
    let w_y = [235u8];
    let w_c = [128u8];
    let white_pv = PlanarView::new(
        1,
        1,
        &[
            PlaneView::new(&w_y, 1, 1, 1, y_spec).unwrap(),
            PlaneView::new(
                &w_c,
                1,
                1,
                1,
                PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
            )
            .unwrap(),
            PlaneView::new(
                &w_c,
                1,
                1,
                1,
                PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .with_color(bt709_narrow.clone())
    .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let mut dst_w = PixelBuf::new(
        1,
        1,
        PixelFormat::new(
            SampleLayout::RGBF32,
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Bt709,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: None,
            },
        ),
    )
    .unwrap();
    reconstruct_planar_into(
        &white_pv,
        &mut dst_w.as_view_mut(),
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(dst_w.as_view().row_as::<f32>(0).unwrap(), &[1.0, 1.0, 1.0]);

    // Identity (GBR, MC=0) maps Y=G, Cb=B, Cr=R!
    let gbr_pv = PlanarView::new(
        1,
        1,
        &[
            PlaneView::new(&[128u8], 1, 1, 1, y_spec).unwrap(),
            PlaneView::new(
                &[255u8],
                1,
                1,
                1,
                PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
            )
            .unwrap(),
            PlaneView::new(
                &[0u8],
                1,
                1,
                1,
                PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .with_color(ColorSpace::SRGB)
    .with_matrix_coefficients(YcbcrMatrix::Identity.to_matrix_coefficients());
    let gbr_out = reconstruct_planar_f32(&gbr_pv, &PlanarReconstructionOptions::nearest()).unwrap();
    let gp = gbr_out.as_view().row_as::<f32>(0).unwrap();
    assert_eq!(gp[0], 0.0);
    assert!((gp[1] - 128.0 / 255.0).abs() < 1e-6);
    assert_eq!(gp[2], 1.0);

    // Monochrome Y-only produces neutral RGB `[y, y, y]`!
    let mono_pv = PlanarView::new(1, 1, &[PlaneView::new(&[235u8], 1, 1, 1, y_spec).unwrap()])
        .unwrap()
        .with_color(bt709_narrow);
    let mono_out =
        reconstruct_planar_f32(&mono_pv, &PlanarReconstructionOptions::nearest()).unwrap();
    assert_eq!(
        mono_out.as_view().row_as::<f32>(0).unwrap(),
        &[1.0, 1.0, 1.0]
    );
}

#[test]
fn b25_d06_independent_8_orientation_coordinate_oracle_with_canaries_and_origin() {
    let independent_map =
        |orient: Orientation, x: u32, y: u32, w: u32, h: u32| -> (u32, u32, u32, u32) {
            match orient {
                Orientation::Identity => (w, h, x, y),
                Orientation::FlipH => (w, h, w - 1 - x, y),
                Orientation::Rotate180 => (w, h, w - 1 - x, h - 1 - y),
                Orientation::FlipV => (w, h, x, h - 1 - y),
                Orientation::Transpose => (h, w, y, x),
                Orientation::Rotate90 => (h, w, h - 1 - y, x),
                Orientation::Transverse => (h, w, h - 1 - y, w - 1 - x),
                Orientation::Rotate270 => (h, w, y, w - 1 - x),
                _ => unreachable!(),
            }
        };

    let origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    let hdr = HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(1000, 300)),
        mastering: None,
    };

    for (w, h) in [(1u32, 17u32), (17, 1), (31, 33), (64, 65), (65, 64)] {
        let mut src = PixelBuf::new(w, h, PixelFormat::RGBA8_SRGB)
            .unwrap()
            .with_origin(origin.clone())
            .with_hdr(hdr);
        {
            let mut vm = src.as_view_mut();
            for y in 0..h {
                let row = vm.row_mut(y);
                for x in 0..w {
                    let off = (x as usize) * 4;
                    row[off] = (x & 0xFF) as u8;
                    row[off + 1] = (y & 0xFF) as u8;
                    row[off + 2] = ((x ^ y) & 0xFF) as u8;
                    row[off + 3] = 255;
                }
            }
        }

        for orient in Orientation::ALL {
            let (ew, eh, _, _) = independent_map(orient, 0, 0, w, h);
            let out = apply_orientation(&src.as_view(), orient).unwrap();
            assert_eq!((out.width(), out.height()), (ew, eh));
            assert_eq!(
                out.origin(),
                &origin,
                "apply_orientation must preserve ColorOrigin"
            );
            assert_eq!(
                out.hdr(),
                hdr,
                "apply_orientation must preserve HdrMetadata"
            );

            let dst_stride = (ew as usize) * 4 + 12;
            let mut dst_raw = vec![0xA5u8; dst_stride * (eh as usize)];
            {
                let mut dst_view = zenpixels_convert::PixelViewMut::new(
                    &mut dst_raw,
                    ew,
                    eh,
                    dst_stride,
                    &zenpixels_convert::formats::RGBA8_SRGB,
                )
                .unwrap();
                apply_orientation_into(&src.as_view(), orient, &mut dst_view).unwrap();
            }

            for y in 0..h {
                for x in 0..w {
                    let (_, _, dx, dy) = independent_map(orient, x, y, w, h);
                    let want = [
                        (x & 0xFF) as u8,
                        (y & 0xFF) as u8,
                        ((x ^ y) & 0xFF) as u8,
                        255,
                    ];
                    let off = (dx as usize) * 4;
                    assert_eq!(&out.as_view().row(dy)[off..off + 4], &want);
                    let po = (dy as usize) * dst_stride + off;
                    assert_eq!(&dst_raw[po..po + 4], &want);
                }
            }
            for dy in 0..eh as usize {
                let tail = &dst_raw[dy * dst_stride + (ew as usize) * 4..(dy + 1) * dst_stride];
                assert!(
                    tail.iter().all(|&b| b == 0xA5),
                    "orient {orient:?} overwrote row {dy} canary!"
                );
            }
        }
    }
}

#[cfg(feature = "gainmap")]
#[test]
fn b28_b29_b30_b40_gainmap_analytic_oracle_31stops_64tap_inclusive_and_gamma_simd_parity() {
    use core::num::NonZeroU32;
    use enough::Unstoppable;
    use zenpixels_convert::gainmap::{GainMapApplyConfig, GainMapResample, apply_gain_map_with};
    use zenpixels_convert::{
        Accelerators, DualLayerHdrView, GainMapChannels, GainMapColorSpace, GainMapComponent,
        GainMapMetadata, SignedFraction, UnsignedFraction,
    };
    let gm_spec = PixelFormat::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            total_channels: 1,
            alpha: AlphaMode::None,
        },
    );

    // B28: Independent analytic oracle: base=1.0, min=0, max=2, gamma=1, offsets=0, U8 code 128
    // => exact gain = 2^(2 * 128 / 255)!
    let base1 =
        PixelBuf::from_typed(vec![1.0f32, 1.0, 1.0], 1, 1, PixelFormat::RGBF32_LINEAR).unwrap();
    let gm128 = PixelBuf::from_typed(vec![128u8], 1, 1, gm_spec.clone()).unwrap();
    let meta2 = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::ZERO,
            max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            gamma: UnsignedFraction::ONE,
            base_offset: SignedFraction::ZERO,
            alternate_offset: SignedFraction::ZERO,
        }),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let dv = DualLayerHdrView::new(
        base1.as_view(),
        gm128.as_view(),
        meta2,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let out = apply_gain_map_with(
        &dv,
        &GainMapApplyConfig::from_headroom_stops(2.0),
        &Unstoppable,
    )
    .unwrap();
    let expected_val = (2.0_f64).powf(2.0 * 128.0 / 255.0) as f32;
    let got_px = out.as_view().row_as::<f32>(0).unwrap();
    for (c, &val) in got_px.iter().enumerate().take(3) {
        assert!(
            (val - expected_val).abs() < 1e-4,
            "analytic gain oracle ch {c}: got={val} expected={expected_val}"
        );
    }

    // B29: Exact 31-stop reconstruction (NOT silently clamped to 30 stops!):
    // (1 + 1/64) * 2^31 - 1/64 ~= 2.181038e9 (whereas 30 stops would be ~1.0905e9!).
    let meta31 = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::ZERO,
            max_gain_stops: SignedFraction::new(31, NonZeroU32::new(1).unwrap()),
            gamma: UnsignedFraction::ONE,
            base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
            alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
        }),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(31, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let gm255 = PixelBuf::from_typed(vec![255u8], 1, 1, gm_spec.clone()).unwrap();
    let dv31 = DualLayerHdrView::new(
        base1.as_view(),
        gm255.as_view(),
        meta31,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let out31_scalar = apply_gain_map_with(
        &dv31,
        &GainMapApplyConfig::from_headroom_stops(31.0)
            .with_policy(ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR)),
        &Unstoppable,
    )
    .unwrap();
    let out31_simd = apply_gain_map_with(
        &dv31,
        &GainMapApplyConfig::from_headroom_stops(31.0)
            .with_policy(ConvertPolicy::permissive().with_accelerators(Accelerators::ALL)),
        &Unstoppable,
    )
    .unwrap();
    let want31 = ((1.0_f64 + 1.0 / 64.0) * 2.0_f64.powi(31) - 1.0 / 64.0) as f32;
    let got31_s = out31_scalar.as_view().row_as::<f32>(0).unwrap()[0];
    let got31_v = out31_simd.as_view().row_as::<f32>(0).unwrap()[0];
    assert!(
        ((got31_s - want31) / want31).abs() < 1e-5,
        "31-stop scalar got {got31_s}, expected {want31} (30-stop clamp would be ~1.09e9!)"
    );
    assert_eq!(
        got31_s.to_bits(),
        got31_v.to_bits(),
        "31-stop scalar vs SIMD must match bit-exact"
    );

    // B30: Bilinear 128x1 -> 4x1 has exact max actual tap count = 64 (inclusive!) -> MUST SUCCEED
    // and equal 1x1 control; 132x1 -> 4x1 has >64 taps -> MUST FAIL with UnsupportedRoute!
    let base4 =
        PixelBuf::from_typed(vec![1.0f32; 4 * 3], 4, 1, PixelFormat::RGBF32_LINEAR).unwrap();
    let gm128w = PixelBuf::from_typed(vec![128u8; 128], 128, 1, gm_spec.clone()).unwrap();
    let dv128 = DualLayerHdrView::new(
        base4.as_view(),
        gm128w.as_view(),
        meta2,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let cfg_bi =
        GainMapApplyConfig::from_headroom_stops(2.0).with_resample(GainMapResample::Bilinear);
    let out128 = apply_gain_map_with(&dv128, &cfg_bi, &Unstoppable).unwrap();
    for px in out128
        .as_view()
        .row_as::<f32>(0)
        .unwrap()
        .as_chunks::<3>()
        .0
    {
        assert!(
            (px[0] - expected_val).abs() < 1e-4,
            "128->4 bilinear got {}",
            px[0]
        );
    }
    let gm132w = PixelBuf::from_typed(vec![128u8; 132], 132, 1, gm_spec.clone()).unwrap();
    let dv132 = DualLayerHdrView::new(
        base4.as_view(),
        gm132w.as_view(),
        meta2,
        None,
        Orientation::Identity,
    )
    .unwrap();
    assert_eq!(
        *apply_gain_map_with(&dv132, &cfg_bi, &Unstoppable)
            .unwrap_err()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // Non-unit gamma (1.5) bilinear SIMD (`Accelerators::ALL`) vs `Accelerators::SCALAR`:
    // 0-bit identical across a 67-pixel ramp (covers SIMD lanes + scalar tail)!
    let meta_g15 = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::ZERO,
            max_gain_stops: SignedFraction::new(3, NonZeroU32::new(1).unwrap()),
            gamma: UnsignedFraction::new(3, NonZeroU32::new(2).unwrap()),
            base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
            alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
        }),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let base67 = PixelBuf::from_typed(
        (0..67 * 3).map(|i| (i % 19) as f32 / 18.0).collect(),
        67,
        1,
        PixelFormat::RGBF32_LINEAR,
    )
    .unwrap();
    let gm34 = PixelBuf::from_typed(
        (0..34u8).map(|i| i.wrapping_mul(7)).collect(),
        34,
        1,
        gm_spec,
    )
    .unwrap();
    let dv_g15 = DualLayerHdrView::new(
        base67.as_view(),
        gm34.as_view(),
        meta_g15,
        None,
        Orientation::Identity,
    )
    .unwrap();
    let o_g15_scalar = apply_gain_map_with(
        &dv_g15,
        &GainMapApplyConfig::from_headroom_stops(2.0)
            .with_resample(GainMapResample::Bilinear)
            .with_policy(ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR)),
        &Unstoppable,
    )
    .unwrap();
    let o_g15_simd = apply_gain_map_with(
        &dv_g15,
        &GainMapApplyConfig::from_headroom_stops(2.0)
            .with_resample(GainMapResample::Bilinear)
            .with_policy(ConvertPolicy::permissive().with_accelerators(Accelerators::ALL)),
        &Unstoppable,
    )
    .unwrap();
    assert_eq!(
        o_g15_scalar.as_view().row(0),
        o_g15_simd.as_view().row(0),
        "gamma=1.5 bilinear SIMD must be 0-bit identical to scalar!"
    );
}
