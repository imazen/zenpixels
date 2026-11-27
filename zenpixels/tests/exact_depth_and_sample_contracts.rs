//! Comprehensive contract tests for `zenpixels` 0.5 core invariants:
//! - ExactDepth grid nesting (full & narrow, independent color/alpha, padding exclusion) [B02, B06, D03]
//! - SampleEncoding & CodePlane [D10]
//! - TypedElement strict adoption vs explicit raw POD reinterpretation [B19]
//! - `with_spec` / `with_color` / `reinterpret` provenance & safety [B20]
//! - `PixelView` / `PixelBuf` origin carriage across borrow, crop, sub_rows, tight copy [B21]
//! - `imgref` active-extent trim (trailing bytes) + consuming `into_crop`/`into_sub_rows` [B22]
//! - `PlaneFormat` role arity, `PlanarView::crop` odd-origin phase, alignment checks [B23, B24]
//! - `DualLayerHdrBuf::new_with_alternate`, `into_parts` roundtrip, `GainMapMetadata::try_symmetric` [B26, B27]
//! - `Cicp::try_from_bytes` range validation [B38]
//! - `PixelBuf::new` `AlphaMode::Opaque` stored max alpha + `image` interop canonicalization [A09, B07, D06]
//! - `CustomPrimaries::ACES_AP0` + singularity refusal [B09]
//! - `Transfer::DciCinema` TC17 vs `Gamma::DCI_2_6` and `Transfer::AdobeRgb` vs `Transfer::Gamma22` [A01, A02]

extern crate alloc;
use core::num::NonZeroU32;

use zenpixels::{
    AlphaMode, BufferError, ChannelOrder, ChromaSiting, ChromaSubsampling, Chromaticity, Cicp,
    CodePlane, ColorOrigin, ColorSpace, ContentLightLevel, CustomPrimaries, DualLayerHdrBuf,
    DualLayerHdrView, ExactDepth, ExactDepthError, F16Sample, GainMapChannels, GainMapColorSpace,
    GainMapComponent, GainMapMetadata, Gamma, HdrMetadata, MasteringDisplay, Orientation, PixelBuf,
    PixelFormat, PixelRect, PixelStorage, PixelView, PlanarBuf, PlanarView, PlaneFormat, PlaneRole,
    PlaneView, Primaries, RenderingIntent, SampleEncoding, SampleFacts, SampleLayout, SampleType,
    SignalRange, SignedFraction, Transfer, UnsignedFraction, full_range_grid_nests,
    narrow_range_grid_nests,
};

#[test]
fn exact_depth_grid_nesting_and_independent_color_alpha_measurement() {
    // B02 / D03: Full-range grid inclusion holds iff (2^dst - 1) is divisible by (2^src - 1).
    assert!(full_range_grid_nests(1, 8));
    assert!(full_range_grid_nests(2, 8));
    assert!(full_range_grid_nests(4, 8));
    assert!(full_range_grid_nests(8, 8));
    assert!(full_range_grid_nests(8, 16));
    assert!(!full_range_grid_nests(3, 8)); // 255 % 7 != 0 (LSB3 code 1 = 1/7 != 36/255!)
    assert!(!full_range_grid_nests(5, 8));
    assert!(!full_range_grid_nests(6, 8));
    assert!(!full_range_grid_nests(10, 12)); // 4095 % 1023 != 0
    assert!(!full_range_grid_nests(10, 16)); // 65535 % 1023 != 0
    assert!(full_range_grid_nests(4, 12)); // 4095 % 15 == 0
    assert!(full_range_grid_nests(6, 12)); // 4095 % 63 == 0

    // Narrow-range ITU-R BT.2100-3 bit-shift nesting holds iff src_bits <= dst_bits for 8..=16.
    assert!(narrow_range_grid_nests(8, 10));
    assert!(narrow_range_grid_nests(8, 12));
    assert!(narrow_range_grid_nests(8, 16));
    assert!(narrow_range_grid_nests(10, 12));
    assert!(!narrow_range_grid_nests(12, 10));
    assert!(!narrow_range_grid_nests(7, 8));

    // Measure full-range U16 RGBA where color is 8-bit replicated (0x1010, 0x8080, 0xEBEB)
    // and alpha is binary 1-bit ({0, 0xFFFF}), with off-grid poison in row padding!
    let mut raw_u16 = vec![0x1234u16; 2 * 6]; // width=1, height=2, stride=12 bytes (2 extra u16 pad per row!)
    raw_u16[0..4].copy_from_slice(&[0x1010, 0x8080, 0xEBEB, 0xFFFF]);
    raw_u16[4..6].copy_from_slice(&[0x1234, 0x5678]);
    raw_u16[6..10].copy_from_slice(&[0x0000, 0xFFFF, 0x4242, 0x0000]);
    raw_u16[10..12].copy_from_slice(&[0x9999, 0xABCD]);

    let view = PixelView::new(
        bytemuck::cast_slice(&raw_u16),
        1,
        2,
        12,
        &zenpixels::formats::RGBA16_SRGB,
    )
    .unwrap();
    let d = ExactDepth::measure(&view).unwrap();
    assert_eq!(d.storage(), SampleType::U16);
    assert_eq!(d.range(), SignalRange::Full);
    assert_eq!(d.color_bits(), 8);
    assert_eq!(d.alpha_bits(), 1);
    assert!(d.color_exact_at(8));
    assert!(d.alpha_exact_at(1));
    assert!(d.alpha_exact_at(8));
    assert!(!d.color_exact_at(4));

    // Narrow-range U16 with replicated bytes [0x1010, 0x8080, 0xEBEB] is NOT exact at 8-bit narrow
    // (narrow 8-bit codes live at multiples of 256: [0x1000, 0x8000, 0xEB00]!).
    let narrow_spec = PixelFormat::RGB16_SRGB.with_range(SignalRange::Narrow);
    let rep_narrow: [u16; 3] = [0x1010, 0x8080, 0xEBEB];
    let rep_view =
        PixelView::new(bytemuck::cast_slice(&rep_narrow), 1, 1, 6, &narrow_spec).unwrap();
    let d_rep = ExactDepth::measure(&rep_view).unwrap();
    assert!(
        !d_rep.color_exact_at(8),
        "0x1010 is off the narrow 8-bit grid (must be multiple of 256), got {d_rep:?}"
    );
    assert_eq!(d_rep.color_bits(), 16); // 0xEBEB is odd -> full 16-bit narrow depth!

    let exact_narrow: [u16; 3] = [0x1000, 0x8000, 0xEB00];
    let exact_view =
        PixelView::new(bytemuck::cast_slice(&exact_narrow), 1, 1, 6, &narrow_spec).unwrap();
    let d_ex = ExactDepth::measure(&exact_view).unwrap();
    assert_eq!(d_ex.color_bits(), 8);
    assert!(d_ex.color_exact_at(8));

    // RGBX undefined slot poison MUST NOT disqualify RGB color exactness!
    let rgbx_spec = PixelFormat::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Rgbx),
        ColorSpace::SRGB.with_alpha(AlphaMode::Undefined),
    );
    let rgbx_words: [u16; 4] = [0x2323, 0x7878, 0xCDCD, 0x1357];
    let rgbx_view = PixelView::new(bytemuck::cast_slice(&rgbx_words), 1, 1, 8, &rgbx_spec).unwrap();
    let d_rgbx = ExactDepth::measure(&rgbx_view).unwrap();
    assert_eq!(d_rgbx.color_bits(), 8);
    // And 4-bit replicated RGB [0x2222, 0x7777, 0xCCCC] with off-grid X=0x1357 measures tightest 4-bit!
    let rgbx_4b: [u16; 4] = [0x2222, 0x7777, 0xCCCC, 0x1357];
    let d_rgbx_4b = ExactDepth::measure(
        &PixelView::new(bytemuck::cast_slice(&rgbx_4b), 1, 1, 8, &rgbx_spec).unwrap(),
    )
    .unwrap();
    assert_eq!(d_rgbx_4b.color_bits(), 4);
    assert!(d_rgbx_4b.color_exact_at(8));

    // Invalid depths rejected cleanly, and 3-bit full-range does NOT nest into 8-bit!
    assert_eq!(
        ExactDepth::new(SampleType::F32, SignalRange::Full, 8, 8),
        Err(ExactDepthError::UnsupportedStorage)
    );
    assert_eq!(
        ExactDepth::new(SampleType::U8, SignalRange::Full, 0, 8),
        Err(ExactDepthError::InvalidBits)
    );
    assert_eq!(
        ExactDepth::new(SampleType::U8, SignalRange::Full, 9, 8),
        Err(ExactDepthError::InvalidBits)
    );
    assert_eq!(
        ExactDepth::new(SampleType::U16, SignalRange::Narrow, 7, 8),
        Err(ExactDepthError::NarrowBelowEightBits)
    );
    let d3_u16 = ExactDepth::new(SampleType::U16, SignalRange::Full, 3, 8).unwrap();
    assert!(!d3_u16.color_exact_at(8));
    assert!(d3_u16.color_exact_at(6));
    assert!(d3_u16.color_exact_at(16));
}

#[test]
fn sample_encoding_and_code_plane_contracts() {
    let enc10 = SampleEncoding::new(SampleType::U16, 10, 0).unwrap();
    assert_eq!(enc10.storage(), SampleType::U16);
    assert_eq!(enc10.code_bits(), 10);
    assert_eq!(enc10.bit_shift(), 0);
    assert!(SampleEncoding::new(SampleType::U16, 0, 0).is_err());
    assert!(SampleEncoding::new(SampleType::U16, 17, 0).is_err());
    assert!(SampleEncoding::new(SampleType::F32, 16, 0).is_err());
    assert!(SampleEncoding::new(SampleType::U16, 10, 7).is_err());

    // CodePlane with strided row & short final row (stride=6 words, width=4, height=2 -> 10 words!).
    let words: [u16; 10] = [100, 200, 300, 400, 0xDEAD, 0xBEEF, 500, 600, 700, 800];
    let plane = CodePlane::from_u16(&words, 4, 2, 6, enc10).unwrap();
    assert_eq!(plane.width(), 4);
    assert_eq!(plane.height(), 2);
    let mut r0 = [0u16; 4];
    let mut r1 = [0u16; 4];
    plane.read_codes(0, 0, &mut r0).unwrap();
    plane.read_codes(0, 1, &mut r1).unwrap();
    assert_eq!(r0, [100, 200, 300, 400]);
    assert_eq!(r1, [500, 600, 700, 800]);
    assert_eq!(plane.code(2, 1), Some(700));

    // ChromaSubsampling & ChromaSiting dimensions (odd dimensions round up!).
    assert_eq!(ChromaSubsampling::Yuv420.chroma_size(5, 3), Some((3, 2)));
    assert_eq!(ChromaSubsampling::Yuv422.chroma_size(5, 3), Some((3, 3)));
    assert_eq!(ChromaSubsampling::Yuv444.chroma_size(5, 3), Some((5, 3)));
    assert_eq!(ChromaSubsampling::Mono.chroma_size(5, 3), None);
    let _ = ChromaSiting::TopLeft;
}

#[test]
fn typed_from_vec_rejects_mismatched_types_while_from_raw_pod_explicitly_reinterprets() {
    // B19: `vec![0x3f800000_i32; 3]` MUST NOT adopt as RGBF32 via typed `PixelBuf::from_typed`!
    let bad_i32 = vec![0x3f80_0000_i32; 3];
    let err = PixelBuf::from_typed(bad_i32, 1, 1, PixelFormat::RGBF32_LINEAR).unwrap_err();
    assert_eq!(*err.error(), BufferError::IncompatibleLayout);

    // U32 rejected as RGBF32, U16 rejected as RGBF16, F16Sample rejected as RGB16.
    assert!(PixelBuf::from_typed(vec![0u32; 3], 1, 1, PixelFormat::RGBF32_LINEAR).is_err());
    assert!(PixelBuf::from_typed(vec![0u16; 3], 1, 1, PixelFormat::RGBF16_LINEAR).is_err());
    assert!(PixelBuf::from_typed(vec![F16Sample::ZERO; 3], 1, 1, PixelFormat::RGB16_SRGB).is_err());

    // Matching typed adoption succeeds zero-copy at identical pointer.
    let good_f32 = vec![1.0f32, 0.5, 0.25];
    let good_ptr = good_f32.as_ptr() as usize;
    let buf = PixelBuf::from_typed(good_f32, 1, 1, PixelFormat::RGBF32_LINEAR).unwrap();
    assert_eq!(buf.as_view().row(0).as_ptr() as usize, good_ptr);

    // Explicit `from_pod_bytes_unchecked` permits deliberate native-endian byte reinterpretation.
    let raw_i32 = vec![0x3f80_0000_i32; 3];
    let reinterp =
        PixelBuf::from_pod_bytes_unchecked(raw_i32, 1, 1, PixelFormat::RGBF32_LINEAR).unwrap();
    assert_eq!(
        reinterp.as_view().row_as::<f32>(0).unwrap(),
        &[1.0, 1.0, 1.0]
    );
}

#[test]
fn with_spec_with_color_reinterpret_and_origin_preservation_contracts() {
    // B20: `with_spec` refuses numeric sample/packing changes (U8 -> I8), while `reinterpret`
    // explicitly allows same-bpp relabel AND clears stale ColorOrigin!
    let p3_origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    let buf = PixelBuf::from_typed(vec![255u8, 128, 127], 1, 1, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_origin(p3_origin.clone());
    let i8_spec = PixelFormat::new(
        SampleLayout::new(SampleType::I8, ChannelOrder::Rgb),
        ColorSpace::Untagged {
            total_channels: 3,
            alpha: AlphaMode::None,
        },
    );
    let err = buf.clone().relabel_format(i8_spec.clone()).unwrap_err();
    assert_eq!(*err.error(), BufferError::IncompatibleLayout);
    assert_eq!(buf.origin(), &p3_origin);

    // Same sample & packing channel-order relabel (RGB8 -> BGR8) succeeds via `relabel_format`.
    let bgr_srgb = PixelFormat::new(SampleLayout::BGR8, ColorSpace::SRGB);
    let bgr_buf = buf.clone().relabel_format(bgr_srgb).unwrap();
    assert_eq!(bgr_buf.origin(), &p3_origin);

    // `relabel_color` changing color space clears stale `origin` and `facts`!
    let p3_buf = buf.clone().relabel_color(ColorSpace::DISPLAY_P3).unwrap();
    assert_eq!(p3_buf.origin(), &ColorOrigin::assumed());
    assert_eq!(p3_buf.facts(), SampleFacts::NONE);

    // Explicit `reinterpret_bytes` permits U8 -> I8 and clears stale `origin` and `facts`!
    let re = buf.reinterpret_bytes(i8_spec).unwrap();
    assert_eq!(re.origin(), &ColorOrigin::assumed());
    assert_eq!(re.facts(), SampleFacts::NONE);
    assert_eq!(re.as_view().row_as::<i8>(0).unwrap(), &[-1i8, -128, 127]);

    // B21: Owned -> View -> Crop -> SubRows -> TightCopy MUST preserve `origin` and `hdr`!
    let hdr = HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(1000, 400)),
        mastering: Some(MasteringDisplay::new(
            [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            [0.3127, 0.3290],
            0.005,
            1000.0,
        )),
    };
    let mut owned = PixelBuf::new(4, 4, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_origin(p3_origin.clone())
        .with_hdr(hdr);
    let v = owned.as_view();
    assert_eq!(v.origin(), Some(&p3_origin));
    assert_eq!(v.hdr(), hdr);
    let cropped_v = v.crop(PixelRect::new(1, 1, 2, 2)).unwrap();
    assert_eq!(cropped_v.origin(), Some(&p3_origin));
    assert_eq!(cropped_v.hdr(), hdr);
    let tight = cropped_v.to_tight_buf().unwrap();
    assert_eq!(tight.origin(), &p3_origin);
    assert_eq!(tight.hdr(), hdr);
    // B22: Consuming mutable `into_sub_rows` + `into_crop` preserves backing `'a` lifetime!
    let mut sub = owned
        .as_view_mut()
        .into_sub_rows(1, 3)
        .unwrap()
        .into_crop(PixelRect::new(1, 0, 2, 2))
        .unwrap();
    sub.row_mut(0)[0] = 77;
    assert_eq!(owned.as_view().row(1)[3], 77);
}

#[cfg(feature = "imgref")]
#[test]
fn imgref_bridge_trims_trailing_bytes_and_rejects_misaligned_stride() {
    // B22: Direct PixelView with extra trailing storage byte `[1,2,3,99]` MUST borrow as 1x1
    // ImgRef<rgb::Rgb<u8>> at the exact same pointer (NOT fail with IncompatibleLayout!).
    let storage = [1u8, 2, 3, 99];
    let v = PixelView::new(&storage, 1, 1, 3, &zenpixels::formats::RGB8_SRGB).unwrap();
    let img: imgref::ImgRef<'_, rgb::Rgb<u8>> = v.as_imgref().unwrap();
    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
    assert_eq!(img.buf().as_ptr() as *const u8, storage.as_ptr());
    assert_eq!(img[(0usize, 0usize)], rgb::Rgb::new(1, 2, 3));

    // Strided 3x2 stride=12 (21 active bytes = 12 + 9) plus 1 trailing byte -> valid ImgRef!
    let mut pitched = vec![0u8; 22];
    pitched[0..9].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    pitched[12..21].copy_from_slice(&[10, 11, 12, 13, 14, 15, 16, 17, 18]);
    pitched[21] = 0xAA;
    let pv = PixelView::new(&pitched, 3, 2, 12, &zenpixels::formats::RGB8_SRGB).unwrap();
    let pimg: imgref::ImgRef<'_, rgb::Rgb<u8>> = pv.as_imgref().unwrap();
    assert_eq!(pimg.stride(), 4);
    assert_eq!(pimg[(2usize, 1usize)], rgb::Rgb::new(16, 17, 18));

    // Non-whole-pixel stride (10 bytes for 3-byte RGB) must still be refused!
    let bad_stride = vec![0u8; 20];
    let bv = PixelView::new(&bad_stride, 3, 2, 10, &zenpixels::formats::RGB8_SRGB).unwrap();
    assert!(bv.as_imgref::<rgb::Rgb<u8>>().is_err());
}

#[test]
fn planar_role_arity_odd_crop_origin_and_alignment() {
    // B23: Known PlaneRole component arity enforced by PlaneFormat::validate!
    let mut bad_y = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    bad_y.components = 2;
    assert!(
        bad_y.validate().is_err(),
        "Y plane cannot have 2 components"
    );
    let mut bad_cbcr = PlaneFormat::new(PlaneRole::CbCr, SampleType::U8);
    bad_cbcr.components = 1;
    assert!(
        bad_cbcr.validate().is_err(),
        "CbCr plane cannot have 1 component"
    );
    let mut ok_extra = PlaneFormat::new(PlaneRole::Extra(0), SampleType::U8);
    ok_extra.components = 2;
    assert!(ok_extra.validate().is_ok());

    // B24: PlanarView::crop records odd crop_origin `(1, 1)` for 4:2:0 phase tracking!
    let y_spec = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    let cb_spec = PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2);
    let cr_spec = PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2);
    let y_data = [0u8; 5 * 3];
    let cb_data = [0u8; 3 * 2];
    let cr_data = [0u8; 3 * 2];
    let py = PlaneView::new(&y_data, 5, 3, 5, y_spec).unwrap();
    let pcb = PlaneView::new(&cb_data, 3, 2, 3, cb_spec).unwrap();
    let pcr = PlaneView::new(&cr_data, 3, 2, 3, cr_spec).unwrap();
    let full = PlanarView::new(5, 3, &[py, pcb, pcr]).unwrap();
    assert_eq!(full.crop_origin(), (0, 0));
    let cropped = full.crop(PixelRect::new(1, 1, 3, 2)).unwrap();
    assert_eq!(cropped.crop_origin(), (1, 1));
    assert_eq!(cropped.width(), 3);
    assert_eq!(cropped.height(), 2);

    // Alignment check uses target `SampleType::min_alignment()` (`align_of::<u64>()`).
    let u64_spec = PlaneFormat::new(PlaneRole::Extra(1), SampleType::U64);
    let mut aligned = vec![0u64; 4];
    let bytes = bytemuck::cast_slice_mut::<u64, u8>(&mut aligned);
    assert!(PlaneView::new(&bytes[0..8], 1, 1, 8, u64_spec).is_ok());
    assert!(PlaneView::new(&bytes[1..9], 1, 1, 8, u64_spec).is_err());

    let _ = (
        PlanarBuf::new(4, 2, &[y_spec]).unwrap(),
        PixelStorage::U8(vec![0u8; 4]),
    );
}

#[test]
fn dual_layer_hdr_strict_alternate_into_parts_and_try_symmetric() {
    // B27: `GainMapMetadata::sdr_to_hdr_stops` rejects <= 0, NaN, and infinity; accepts positive.
    assert!(GainMapMetadata::sdr_to_hdr_stops(-1.0).is_none());
    assert!(GainMapMetadata::sdr_to_hdr_stops(0.0).is_none());
    assert!(GainMapMetadata::sdr_to_hdr_stops(f32::NAN).is_none());
    assert!(GainMapMetadata::sdr_to_hdr_stops(f32::INFINITY).is_none());
    assert!(GainMapMetadata::sdr_to_hdr_stops(f32::NEG_INFINITY).is_none());
    let pos_meta = GainMapMetadata::sdr_to_hdr_stops(2.5).unwrap();
    assert!(pos_meta.is_valid());
    assert!((pos_meta.alternate_hdr_headroom.to_f32() - 2.5).abs() < 1e-4);

    // B26 + B27: `DualLayerHdrBuf::new` refuses `Alternate` without alternate_color_space;
    // `new_with_alternate` succeeds and `into_parts()` round-trips every single field!
    let alt_meta = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::ZERO,
            max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            gamma: UnsignedFraction::ONE,
            base_offset: SignedFraction::ZERO,
            alternate_offset: SignedFraction::ZERO,
        }),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Alternate,
    )
    .unwrap();
    let gm_spec = PixelFormat::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            total_channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let base = PixelBuf::new(4, 4, PixelFormat::RGB8_SRGB).unwrap();
    let gm = PixelBuf::new(2, 2, gm_spec).unwrap();
    assert!(DualLayerHdrBuf::new(base.clone(), gm.clone(), alt_meta).is_err());
    assert!(
        DualLayerHdrView::new(
            base.as_view(),
            gm.as_view(),
            alt_meta,
            None,
            Orientation::Identity
        )
        .is_err()
    );

    let dual =
        DualLayerHdrBuf::new_with_alternate_color(base, gm, alt_meta, ColorSpace::DISPLAY_P3)
            .unwrap()
            .with_orientation(Orientation::Rotate270);
    let parts = dual.into_parts();
    assert_eq!(parts.orientation, Orientation::Rotate270);
    assert_eq!(parts.alternate_color_space, Some(ColorSpace::DISPLAY_P3));
    assert_eq!(parts.metadata, alt_meta);
    let rebuilt = DualLayerHdrBuf::from_parts(parts).unwrap();
    assert_eq!(rebuilt.orientation(), Orientation::Rotate270);
    assert_eq!(
        rebuilt.alternate_color_space(),
        Some(&ColorSpace::DISPLAY_P3)
    );
}

#[test]
fn cicp_wire_range_aces_ap0_opaque_init_and_transfer_distinctness() {
    // B38: `Cicp::try_from_bytes` rejects invalid full_range byte 2, accepts 0 and 1.
    assert!(Cicp::try_from_bytes([9, 16, 0, 2]).is_err());
    assert!(Cicp::try_from_bytes([9, 16, 0, 255]).is_err());
    let pq_narrow = Cicp::try_from_bytes([9, 16, 0, 0]).unwrap();
    assert!(!pq_narrow.full_range);
    let pq_full = Cicp::try_from_bytes([9, 16, 0, 1]).unwrap();
    assert!(pq_full.full_range);

    // B09: ACES AP0 imaginary/boundary primaries (B.y = -0.077) accepted, singular rejected!
    let ap0 = CustomPrimaries::ACES_AP0;
    let m = Primaries::Custom(ap0)
        .gamut_matrix_to(Primaries::Bt709)
        .unwrap();
    for row in m {
        assert!(
            (row[0] + row[1] + row[2] - 1.0).abs() < 1e-4,
            "AP0 white row={row:?}"
        );
    }
    assert!(
        CustomPrimaries::new(
            (0.64, 0.33),
            (0.64, 0.33),
            (0.15, 0.06),
            Chromaticity::D65.xy(),
        )
        .is_none()
    );

    // A01 / A02: AdobeRgb != Gamma22 (TC4), DciCinema (TC17) != Gamma::DCI_2_6!
    assert_eq!(Transfer::Gamma22.to_cicp(), Some(4));
    assert_eq!(Transfer::from_cicp(4), Some(Transfer::Gamma22));
    assert_eq!(Transfer::AdobeRgb.to_cicp(), None);
    assert_eq!(Transfer::DciCinema.to_cicp(), Some(17));
    assert_eq!(Transfer::from_cicp(17), Some(Transfer::DciCinema));
    assert_eq!(Transfer::Gamma(Gamma::DCI_2_6).to_cicp(), None);

    // A09 / B07: Freshly allocated `AlphaMode::Opaque` buffers initialize stored alpha to max!
    let op_u8 = PixelBuf::new(2, 2, PixelFormat::RGBA8_SRGB.with_alpha(AlphaMode::Opaque)).unwrap();
    assert!(op_u8.facts().all_opaque);
    for y in 0..2 {
        for px in op_u8.as_view().row(y).chunks_exact(4) {
            assert_eq!(px, &[0, 0, 0, 255]);
        }
    }
    let op_u16 =
        PixelBuf::new(2, 1, PixelFormat::RGBA16_SRGB.with_alpha(AlphaMode::Opaque)).unwrap();
    assert_eq!(
        op_u16.as_view().row_as::<u16>(0).unwrap(),
        &[0, 0, 0, 65535, 0, 0, 0, 65535]
    );
    let op_f32 = PixelBuf::new(
        1,
        1,
        PixelFormat::RGBAF32_LINEAR.with_alpha(AlphaMode::Opaque),
    )
    .unwrap();
    assert_eq!(
        op_f32.as_view().row_as::<f32>(0).unwrap(),
        &[0.0, 0.0, 0.0, 1.0]
    );

    // `image` crate interop canonicalizes poisoned AlphaMode::Opaque alpha=0 to 255 on export!
    #[cfg(feature = "image")]
    {
        let poisoned = PixelBuf::from_typed(
            vec![10u8, 20, 30, 0, 40, 50, 60, 17],
            2,
            1,
            PixelFormat::RGBA8_SRGB.with_alpha(AlphaMode::Opaque),
        )
        .unwrap();
        let img = poisoned.into_image_buffer::<image::Rgba<u8>>().unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [40, 50, 60, 255]);
    }
    let _ = RenderingIntent::Perceptual;
}

#[test]
fn exact_depth_union_disjoint_strips_and_icc_can_represent_hdr() {
    // Disjoint-strip union on U16 full range: 3-bit + 5-bit -> 15-bit; 4-bit + 8-bit -> 8-bit;
    // 8-bit + 12-bit -> lcm(8,12)=24 > 16 -> 16-bit storage depth!
    let u3 = ExactDepth::new(SampleType::U16, SignalRange::Full, 3, 1).unwrap();
    let u5 = ExactDepth::new(SampleType::U16, SignalRange::Full, 5, 4).unwrap();
    let u35 = ExactDepth::union_disjoint_regions(u3, u5).unwrap();
    assert_eq!(u35.color_bits(), 15);
    assert_eq!(u35.alpha_bits(), 4);

    let u8_in_16 = ExactDepth::new(SampleType::U16, SignalRange::Full, 8, 8).unwrap();
    let u12_in_16 = ExactDepth::new(SampleType::U16, SignalRange::Full, 12, 12).unwrap();
    let u8_12 = ExactDepth::union_disjoint_regions(u8_in_16, u12_in_16).unwrap();
    assert_eq!(u8_12.color_bits(), 16);
    assert_eq!(u8_12.alpha_bits(), 16);

    // Narrow range: monotonic max (`max(10, 12) = 12`).
    let n10 = ExactDepth::new(SampleType::U16, SignalRange::Narrow, 10, 8).unwrap();
    let n12 = ExactDepth::new(SampleType::U16, SignalRange::Narrow, 12, 4).unwrap();
    let n10_12 = ExactDepth::union_disjoint_regions(n10, n12).unwrap();
    assert_eq!(n10_12.color_bits(), 12);
    assert_eq!(n10_12.alpha_bits(), 8);

    // Exhaustive mathematical verification over ALL U16 full-range pairs (1..=16 x 1..=16):
    // every widened code from grid `a` and every widened code from grid `b` MUST hold on `union_disjoint_regions(a, b)`!
    for a in 1u8..=16 {
        let da = ExactDepth::new(SampleType::U16, SignalRange::Full, a, 16).unwrap();
        let max_a = (1u32 << a) - 1;
        let samples_a: Vec<u16> = (0..=max_a)
            .step_by(((max_a / 31).max(1)) as usize)
            .map(|q| ((q * 65535 + max_a / 2) / max_a) as u16)
            .collect();
        for b in a..=16 {
            let db = ExactDepth::new(SampleType::U16, SignalRange::Full, b, 16).unwrap();
            let merged = ExactDepth::union_disjoint_regions(da, db).unwrap();
            assert!(merged.color_bits() >= a && merged.color_bits() >= b);
            let buf_a = PixelBuf::from_typed(
                samples_a.clone(),
                samples_a.len() as u32,
                1,
                PixelFormat::GRAY16_SRGB,
            )
            .unwrap();
            assert!(
                merged.holds_for(&buf_a.as_view()).unwrap(),
                "union({a}, {b}) = {} must hold for every {a}-bit sample",
                merged.color_bits()
            );
            let max_b = (1u32 << b) - 1;
            let samples_b: Vec<u16> = (0..=max_b)
                .step_by(((max_b / 31).max(1)) as usize)
                .map(|q| ((q * 65535 + max_b / 2) / max_b) as u16)
                .collect();
            let buf_b = PixelBuf::from_typed(
                samples_b.clone(),
                samples_b.len() as u32,
                1,
                PixelFormat::GRAY16_SRGB,
            )
            .unwrap();
            assert!(
                merged.holds_for(&buf_b.as_view()).unwrap(),
                "union({a}, {b}) = {} must hold for every {b}-bit sample",
                merged.color_bits()
            );
        }
    }

    // `PixelFormat::can_represent_hdr()` returns true for floating-point ICC and PQ/HLG-tagged ICC.
    let fake_icc: std::sync::Arc<[u8]> = std::sync::Arc::from(vec![0u8; 128]);
    let icc_f32 = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Icc {
            profile: fake_icc.clone(),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(icc_f32.can_represent_hdr());
    let icc_u16 = PixelFormat::new(
        SampleLayout::RGB16,
        ColorSpace::Icc {
            profile: fake_icc,
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(!icc_u16.can_represent_hdr());
}

#[test]
fn opus_review_p1_5_p2_4_p2_12_gainmap_direction_shifted_and_banded_propagation() {
    // P1-5: Contradictory `backward_direction` flag MUST fail `is_valid()` and return `direction() == None`!
    let base_hdr = UnsignedFraction::new(3, NonZeroU32::new(1).unwrap());
    let alt_sdr = UnsignedFraction::ZERO;
    let rev_meta = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::new(-3, NonZeroU32::new(1).unwrap()),
            max_gain_stops: SignedFraction::ZERO,
            gamma: UnsignedFraction::ONE,
            base_offset: SignedFraction::ONE_64TH,
            alternate_offset: SignedFraction::ONE_64TH,
        }),
        base_hdr,
        alt_sdr,
        GainMapColorSpace::Base,
    )
    .unwrap();
    assert!(rev_meta.is_valid());
    assert_eq!(
        rev_meta.direction(),
        Some(zenpixels::GainMapDirection::Reverse)
    );
    let ok_flag = rev_meta.with_backward_direction(true);
    assert!(ok_flag.is_valid());
    assert_eq!(
        ok_flag.direction(),
        Some(zenpixels::GainMapDirection::Reverse)
    );
    let bad_flag = rev_meta.with_backward_direction(false);
    assert!(!bad_flag.is_valid());
    assert_eq!(bad_flag.direction(), None);

    let fwd_bad_flag = GainMapMetadata::sdr_to_hdr_stops(2.0)
        .unwrap()
        .with_backward_direction(true);
    assert!(!fwd_bad_flag.is_valid());
    assert_eq!(fwd_bad_flag.direction(), None);

    // P2-4: DualLayerHdrBuf rejects BitPacking::Shifted gain map (consistent with gainmap applier).
    let base = PixelBuf::new(2, 2, PixelFormat::RGB8_SRGB).unwrap();
    let shifted_gm_spec = PixelFormat::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Gray)
            .with_packing(zenpixels::BitPacking::Shifted { bits: 10, shift: 2 }),
        ColorSpace::Untagged {
            total_channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let shifted_gm = PixelBuf::new(2, 2, shifted_gm_spec).unwrap();
    assert!(
        DualLayerHdrBuf::new(
            base,
            shifted_gm,
            GainMapMetadata::sdr_to_hdr_stops(2.0).unwrap()
        )
        .is_err()
    );

    // P2-12: BandedBuf::with_hdr and with_origin propagate to child bands.
    let hdr = HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(1200, 350)),
        mastering: None,
    };
    let orig = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    let banded = zenpixels::BandedBuf::new(
        4,
        4,
        PixelFormat::RGB8_SRGB,
        zenpixels::BandSizing::MaxBandBytes(24),
    )
    .unwrap()
    .with_hdr(hdr)
    .with_origin(orig.clone());
    assert_eq!(banded.bands().len(), 2);
    for band in banded.bands() {
        assert_eq!(band.hdr(), hdr);
        assert_eq!(band.origin(), &orig);
    }
}
