//! Integration tests for ImageSpec/PixelFormat/SpecCow, lossless H.273 CICP,
//! ICC structural validation, ClliCodes, Orientation on buffers/views/bands/planar,
//! sound Opaque construction/new_zeroed/retag, recovery constructors, and zero-copy planar APIs.

use std::sync::Arc;
use zenpixels::{
    AlphaMode, BandedBuf, BitPacking, BufFill, BufferError, Chromaticity, Cicp, ClliCodes,
    ColorError, ColorOrigin, ColorSpace, ContentLightLevel, DiffuseWhite, DualLayerHdr,
    DualLayerHdrView, GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapMetadata, Gamma,
    HdrMetadata, IccValidationError, ImageSpec, Nits, Orientation, OwnedPlane, ParametricColor,
    PixelBuf, PixelCow, PixelFormat, PixelSink, PixelSource, PixelView, PlanarBuf, PlanarView,
    PlaneRole, PlaneSpec, PlaneView, PlaneViewMut, Primaries, RenderingIntent, RowAlign,
    SampleLayout, SampleType, SignalRange, SignedFraction, Transfer, UnsignedFraction, WhitePoint,
    validate_icc_structure,
};

fn minimal_valid_icc(class: &[u8; 4]) -> Vec<u8> {
    let mut icc = vec![0u8; 132];
    icc[0..4].copy_from_slice(&132u32.to_be_bytes());
    icc[12..16].copy_from_slice(class);
    icc[16..20].copy_from_slice(b"RGB ");
    icc[20..24].copy_from_slice(b"XYZ ");
    icc[36..40].copy_from_slice(b"acsp");
    icc[128..132].copy_from_slice(&0u32.to_be_bytes()); // tag_count = 0
    icc
}

fn sample_gm_meta() -> GainMapMetadata {
    let comp = GainMapComponent {
        min_gain_stops: SignedFraction::try_new(0, 1).unwrap(),
        max_gain_stops: SignedFraction::try_new(2, 1).unwrap(),
        gamma: UnsignedFraction::try_new(1, 1).unwrap(),
        base_offset: SignedFraction::try_new(1, 64).unwrap(),
        alternate_offset: SignedFraction::try_new(1, 64).unwrap(),
    };
    GainMapMetadata::try_new(
        GainMapChannels::Shared(comp),
        UnsignedFraction::try_new(0, 1).unwrap(),
        UnsignedFraction::try_new(2, 1).unwrap(),
        GainMapColorSpace::Base,
    )
    .unwrap()
}

#[test]
fn imagespec_const_promotion_pixelformat_copy_and_niches() {
    // 1. Static slice promotion via `const` and inline `const { ... }` on ImageSpec,
    //    plus direct static promotion on Copy/!Drop PixelFormat!
    const PROMOTED: &[ImageSpec] = &[ImageSpec::RGB8_SRGB, ImageSpec::RGBA8_SRGB];
    let promoted_in_fn: &'static [ImageSpec] = const {
        &[
            ImageSpec::RGB8_SRGB,
            ImageSpec::RGBA8_SRGB,
            ImageSpec::RGB16_BT2100_PQ,
        ]
    };
    let promoted_formats: &'static [PixelFormat] = &[
        PixelFormat::RGB8_SRGB,
        PixelFormat::RGBA8_SRGB,
        PixelFormat::RGB16_BT2100_PQ,
    ];
    assert_eq!(PROMOTED.len(), 2);
    assert_eq!(promoted_in_fn[0], ImageSpec::rgb8_srgb());
    assert_eq!(promoted_in_fn[2], ImageSpec::rgb16_bt2100_pq());
    assert_eq!(promoted_formats[0].to_spec(), ImageSpec::RGB8_SRGB);

    // 2. PixelFormat is Copy + !Drop + const.
    fn assert_copy<T: Copy>() {}
    assert_copy::<PixelFormat>();
    assert_copy::<ParametricColor>();
    assert!(!core::mem::needs_drop::<PixelFormat>());
    assert!(!core::mem::needs_drop::<ParametricColor>());

    let fmt = PixelFormat::RGB8_SRGB;
    let spec: ImageSpec = fmt.into();
    assert_eq!(spec, ImageSpec::RGB8_SRGB);
    assert_eq!(spec.as_format(), Some(fmt));

    let icc_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(minimal_valid_icc(b"mntr")),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert_eq!(icc_spec.as_format(), None);

    // 3. NonZeroU32 niches keep Option<DiffuseWhite>, Option<Nits>, Option<Gamma> at 4 bytes!
    assert_eq!(core::mem::size_of::<Option<DiffuseWhite>>(), 4);
    assert_eq!(core::mem::size_of::<Option<Nits>>(), 4);
    assert_eq!(core::mem::size_of::<Option<Gamma>>(), 4);
}

#[test]
fn speccow_zero_arc_bumps_across_views_crops_and_mut_views() {
    let profile: Arc<[u8]> = Arc::from(minimal_valid_icc(b"mntr"));
    let icc_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::clone(&profile),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert_eq!(Arc::strong_count(&profile), 2);

    let mut buf = PixelBuf::new(8, 8, icc_spec).unwrap();
    // `icc_spec` moved into `buf`; strong count stays 2 (`profile` + `buf.spec`).
    assert_eq!(Arc::strong_count(&profile), 2);

    {
        let view = buf.as_view();
        assert_eq!(Arc::strong_count(&profile), 2);
        let cropped = view.crop(1, 1, 4, 4).unwrap();
        assert_eq!(Arc::strong_count(&profile), 2);
        let strip = cropped.sub_rows(1, 2).unwrap();
        assert_eq!(Arc::strong_count(&profile), 2);
        let cow = PixelCow::Borrowed(strip);
        let cow_view = cow.as_view();
        assert_eq!(Arc::strong_count(&profile), 2);
        assert_eq!(cow_view.rows(), 2);
    }

    {
        let mut mv = buf.try_as_view_mut().unwrap();
        assert_eq!(Arc::strong_count(&profile), 2);
        let sub = mv.crop_mut(0, 0, 4, 4).unwrap();
        assert_eq!(Arc::strong_count(&profile), 2);
        let imm = sub.as_view();
        assert_eq!(Arc::strong_count(&profile), 2);
        assert_eq!(imm.width(), 4);
    }

    // Passing `&ImageSpec` to `PixelView::new` ALSO avoids any Arc bump!
    let raw = [0u8; 4 * 4 * 3];
    let borrowed = PixelView::new(&raw, 4, 4, 12, buf.spec()).unwrap();
    assert_eq!(Arc::strong_count(&profile), 2);
    assert_eq!(borrowed.width(), 4);
}

#[test]
fn lossless_h273_cicp_all_primaries_transfers_projective_and_clli() {
    // All 11 ITU-T H.273 Table 2 colour_primaries codes round-trip 100% losslessly.
    for cp in [1u8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22] {
        let p = Primaries::from_cicp(cp).expect("supported CP");
        assert_eq!(p.to_cicp(), Some(cp), "CP {cp} must round-trip losslessly");
        let mat = p.rgb_to_xyz_f64().expect("valid RGB->XYZ matrix");
        // Every row sum of RGB->XYZ must equal white XYZ (with Y_w == 1.0)!
        let sum_y = mat[1][0] + mat[1][1] + mat[1][2];
        assert!(
            (sum_y - 1.0).abs() < 1e-4,
            "CP {cp}: Y row sum {sum_y} should be 1.0"
        );
    }
    assert_eq!(Primaries::from_cicp(2), None);
    assert_eq!(
        Primaries::Bt470M.white_point(),
        Some((Chromaticity::C.x(), Chromaticity::C.y()))
    );
    assert_eq!(
        WhitePoint::C.xy(),
        (Chromaticity::C.x(), Chromaticity::C.y())
    );
    assert!(Primaries::Bt601.same_chromaticity(Primaries::Smpte240M));
    assert!(!Primaries::Bt709.same_chromaticity(Primaries::Bt601));

    // CIE 1931 XYZ (CP=10) has r_y == 0 and b_y == 0; projective solver produces identity matrix!
    let xyz_mat = Primaries::Xyz.rgb_to_xyz_f64().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((xyz_mat[i][j] - want).abs() < 1e-6);
        }
    }

    // All 16 ITU-T H.273 Table 3 transfer_characteristics codes round-trip 100% losslessly.
    for tc in [1u8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] {
        let t = Transfer::from_cicp(tc).expect("supported TC");
        assert_eq!(t.to_cicp(), Some(tc), "TC {tc} must round-trip losslessly");
    }
    assert!(Transfer::Bt709.same_curve(Transfer::Bt601));
    assert!(Transfer::Bt709.same_curve(Transfer::Bt2020_10bit));
    assert!(Transfer::Bt709.same_curve(Transfer::Bt2020_12bit));
    assert!(!Transfer::Bt709.same_curve(Transfer::Bt1361));

    // ColorSpace -> Cicp round-trip preserves exact codes (CP=6/TC=6 stays 6/6, not 1/1).
    let cicp_601 = Cicp::new(6, 6, 0, true);
    let cs_601 = ColorSpace::Rgb {
        primaries: Primaries::from_cicp(cicp_601.color_primaries).unwrap(),
        transfer: Transfer::from_cicp(cicp_601.transfer_characteristics).unwrap(),
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    assert_eq!(cs_601.to_cicp(), Some(cicp_601));

    // ClliCodes lossless 0.0001-nit PNG cLLI wire carrier.
    let codes = ClliCodes::from_ten_thousandths(12_345_678, 2_502_500);
    let bytes = codes.to_be_bytes();
    let decoded = ClliCodes::from_be_bytes(bytes);
    assert_eq!(decoded, codes);
    assert!((decoded.max_cll_nits() - 1234.5678).abs() < 1e-6);
    assert!((decoded.max_fall_nits() - 250.25).abs() < 1e-6);
    let cll = decoded.semantic();
    assert_eq!(cll.max_content_light_level, 1235);
    assert_eq!(cll.max_frame_average_light_level, 250);

    // Checked `try_with_*` helpers detect no-op variants.
    assert!(
        ImageSpec::OKLAB_F32
            .try_with_primaries(Primaries::DisplayP3)
            .is_none()
    );
    assert!(
        ImageSpec::OKLAB_F32
            .try_with_transfer(Transfer::Srgb)
            .is_none()
    );
    assert!(
        ImageSpec::OKLAB_F32
            .try_with_range(SignalRange::Narrow)
            .is_none()
    );
    assert!(
        ImageSpec::OKLAB_F32
            .try_with_white(DiffuseWhite::BT2408)
            .is_none()
    );
    assert!(
        ImageSpec::RGB8_SRGB
            .try_with_primaries(Primaries::DisplayP3)
            .is_some()
    );
    assert!(
        ImageSpec::GRAY8_SRGB
            .try_with_transfer(Transfer::Linear)
            .is_some()
    );
}

#[test]
fn icc_structural_validation_wired_into_imagespec() {
    // Truncated (<132B) rejected.
    assert_eq!(
        validate_icc_structure(&[0u8; 128]),
        Err(IccValidationError::TooShort)
    );
    // Bad magic rejected.
    let mut bad_magic = minimal_valid_icc(b"mntr");
    bad_magic[36..40].copy_from_slice(b"nope");
    assert_eq!(
        validate_icc_structure(&bad_magic),
        Err(IccValidationError::MissingSignature)
    );
    // Bad profile class rejected.
    let bad_class = minimal_valid_icc(b"xxxx");
    assert_eq!(
        validate_icc_structure(&bad_class),
        Err(IccValidationError::UnsupportedProfileClass)
    );
    // Declared size > bytes length rejected.
    let mut trunc = minimal_valid_icc(b"mntr");
    trunc[0..4].copy_from_slice(&512u32.to_be_bytes());
    assert_eq!(
        validate_icc_structure(&trunc),
        Err(IccValidationError::InvalidDeclaredSize)
    );
    // Tag span out of bounds rejected.
    let mut oob_tag = vec![0u8; 144];
    oob_tag[0..4].copy_from_slice(&144u32.to_be_bytes());
    oob_tag[12..16].copy_from_slice(b"mntr");
    oob_tag[36..40].copy_from_slice(b"acsp");
    oob_tag[128..132].copy_from_slice(&1u32.to_be_bytes()); // 1 tag
    oob_tag[132..136].copy_from_slice(b"desc");
    oob_tag[136..140].copy_from_slice(&140u32.to_be_bytes()); // offset 140
    oob_tag[140..144].copy_from_slice(&32u32.to_be_bytes()); // size 32 -> 172 > 144!
    assert_eq!(
        validate_icc_structure(&oob_tag),
        Err(IccValidationError::TagSpanOutOfBounds)
    );
    let bad_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(oob_tag),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert_eq!(
        *bad_spec.validate().unwrap_err().error(),
        ColorError::InvalidIccProfile
    );

    // Valid profile passes.
    let ok_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(minimal_valid_icc(b"mntr")),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(ok_spec.validate().is_ok());
}

#[test]
fn opaque_construction_white_fill_new_zeroed_retag_and_recovery() {
    // 1. `new` black-fills and sets Opaque alpha to MAX.
    let op_rgba8 = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque);
    let black = PixelBuf::new(2, 2, op_rgba8.clone()).unwrap();
    assert!(black.facts().all_opaque);
    assert_eq!(black.as_view().row(0), &[0, 0, 0, 255, 0, 0, 0, 255]);

    // 2. `new_white` fills reference white across Full, Narrow, Bitfields, Oklab, Xyz, CMYK.
    let white_rgba8 = PixelBuf::new_white(2, 1, op_rgba8.clone()).unwrap();
    assert!(white_rgba8.facts().all_opaque);
    assert_eq!(
        white_rgba8.as_view().row(0),
        &[255, 255, 255, 255, 255, 255, 255, 255]
    );

    let narrow_rgb8 = ImageSpec::RGB8_SRGB.with_range(SignalRange::Narrow);
    let white_narrow = PixelBuf::new_white(2, 1, narrow_rgb8).unwrap();
    assert_eq!(
        white_narrow.as_view().row(0),
        &[235, 235, 235, 235, 235, 235]
    );

    let rgb10a2_spec = ImageSpec::new(
        SampleLayout::RGB10A2,
        ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
    );
    let white_10a2 = PixelBuf::new_white(1, 1, rgb10a2_spec.clone()).unwrap();
    let w10 = u32::from_ne_bytes(white_10a2.as_view().row(0).try_into().unwrap());
    assert_eq!(w10, 0xFFFF_FFFF);

    let white_oklab = PixelBuf::new_white(1, 1, ImageSpec::OKLABA_F32).unwrap();
    let ok_px: &[f32] = white_oklab.as_view().row_as(0).unwrap();
    assert_eq!(ok_px, &[1.0, 0.0, 0.0, 1.0]);

    let white_cmyk = PixelBuf::new_white(1, 1, ImageSpec::cmyk8(None)).unwrap();
    assert_eq!(white_cmyk.as_view().row(0), &[0, 0, 0, 0]);

    // 3. `new_zeroed` refuses `AlphaMode::Opaque` on alpha-bearing layouts, accepts `Undefined`.
    assert_eq!(
        *PixelBuf::new_zeroed(4, 4, op_rgba8.clone())
            .unwrap_err()
            .error(),
        BufferError::UnverifiedOpaqueAlpha
    );
    assert_eq!(
        *PixelBuf::new_zeroed(4, 4, rgb10a2_spec)
            .unwrap_err()
            .error(),
        BufferError::UnverifiedOpaqueAlpha
    );

    let straight_rgba8 = ImageSpec::RGBA8_SRGB;
    let mut fast = PixelBuf::new_zeroed(2, 1, straight_rgba8).unwrap();
    assert_eq!(fast.as_view().row(0), &[0, 0, 0, 0, 0, 0, 0, 0]);

    // Filling with alpha != 255 fails `into_opaque_verified`.
    fast.as_view_mut()
        .row_mut(0)
        .copy_from_slice(&[10, 20, 30, 255, 40, 50, 60, 254]);
    let fast_clone = fast.clone();
    assert_eq!(
        *fast_clone.into_opaque_verified().unwrap_err().error(),
        BufferError::UnverifiedOpaqueAlpha
    );

    // `into_opaque_filled` repairs alpha to 255 and retags as Opaque.
    let filled = fast.clone().into_opaque_filled().unwrap();
    assert_eq!(filled.spec().color.alpha(), AlphaMode::Opaque);
    assert!(filled.facts().all_opaque);
    assert_eq!(filled.as_view().row(0), &[10, 20, 30, 255, 40, 50, 60, 255]);

    // Fixing alpha then calling `into_opaque_verified` succeeds zero-copy.
    fast.as_view_mut().row_mut(0)[7] = 255;
    let verified = fast.into_opaque_verified().unwrap();
    assert_eq!(verified.spec().color.alpha(), AlphaMode::Opaque);
    assert!(verified.facts().all_opaque);

    // `assume_opaque` retags in O(1) without scanning.
    let mut fast2 =
        PixelBuf::new_with_fill(2, 1, ImageSpec::RGBA8_SRGB, RowAlign::Tight, BufFill::White)
            .unwrap();
    fast2 = fast2.assume_opaque().unwrap();
    assert_eq!(fast2.spec().color.alpha(), AlphaMode::Opaque);
    assert!(fast2.facts().all_opaque);

    // No-alpha layout rejects all three opaque retags.
    let rgb = PixelBuf::new(2, 2, PixelFormat::RGB8_SRGB).unwrap();
    assert_eq!(
        *rgb.assume_opaque().unwrap_err().error(),
        BufferError::IncompatibleLayout
    );

    // 4. Recovery constructors return caller Vec untouched on error.
    let short_bytes = vec![7u8; 5];
    let ptr = short_bytes.as_ptr();
    let (err, recovered) =
        PixelBuf::try_from_bytes_recover(short_bytes, 4, 4, PixelFormat::RGB8_SRGB).unwrap_err();
    assert_eq!(*err.error(), BufferError::InsufficientData);
    assert_eq!(recovered.as_ptr(), ptr);

    let wrong_vec = vec![123u16; 12];
    let vptr = wrong_vec.as_ptr();
    let (err, recovered_v) =
        PixelBuf::try_from_vec_recover(wrong_vec, 2, 2, PixelFormat::RGB8_SRGB).unwrap_err();
    assert_eq!(*err.error(), BufferError::IncompatibleLayout);
    assert_eq!(recovered_v.as_ptr(), vptr);
}

#[test]
fn orientation_carriage_across_buffers_views_bands_planar_and_dual_layer() {
    let mut buf = PixelBuf::new(6, 4, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    assert_eq!(buf.orientation(), Orientation::Rotate90);
    assert_eq!(buf.display_dimensions(), (4, 6));
    assert_eq!(buf.as_view().orientation(), Orientation::Rotate90);
    assert_eq!(buf.as_view().display_dimensions(), (4, 6));
    assert_eq!(
        buf.as_view().crop(0, 0, 3, 2).unwrap().orientation(),
        Orientation::Rotate90
    );
    assert_eq!(buf.source_orientation(), Orientation::Rotate90);

    // PixelBufParts preserves orientation losslessly.
    let parts = buf.clone().into_parts();
    assert_eq!(parts.orientation, Orientation::Rotate90);
    let restored = PixelBuf::try_from_parts(parts).unwrap();
    assert_eq!(restored.orientation(), Orientation::Rotate90);

    // rotate_metadata composes in O(1).
    buf = buf.rotate_metadata(Orientation::Rotate90);
    assert_eq!(buf.orientation(), Orientation::Rotate180);
    assert_eq!(buf.display_dimensions(), (6, 4));

    // BandedBuf propagates orientation to bands and back through into_buf.
    let banded = BandedBuf::from_buf(buf.clone(), 64).unwrap();
    assert_eq!(banded.orientation(), Orientation::Rotate180);
    assert_eq!(banded.bands()[0].orientation(), Orientation::Rotate180);
    let mut dst_band = BandedBuf::new(6, 4, PixelFormat::RGB8_SRGB, 64).unwrap();
    dst_band.sink_set_orientation(Orientation::Rotate270);
    assert_eq!(dst_band.orientation(), Orientation::Rotate270);
    assert_eq!(
        dst_band.into_buf().unwrap().orientation(),
        Orientation::Rotate270
    );

    // DualLayerHdr orientation validation, inheritance, sync, and InvalidGainMapMetadata.
    let gm_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let base_r90 = PixelBuf::new(4, 2, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    let gm_id = PixelBuf::new(4, 2, gm_spec.clone()).unwrap();
    let meta = sample_gm_meta();

    // Misregistered layer orientations must fail!
    assert_eq!(
        *DualLayerHdr::new(base_r90.clone(), gm_id.clone(), meta)
            .unwrap_err()
            .error(),
        BufferError::IncompatibleLayout
    );

    // Matching layer orientations inherited by `DualLayerHdr::new` and synced on `rotate_metadata`.
    let gm_r90 = gm_id.clone().with_orientation(Orientation::Rotate90);
    let dual = DualLayerHdr::new(base_r90.clone(), gm_r90.clone(), meta)
        .unwrap()
        .rotate_metadata(Orientation::Rotate90);
    assert_eq!(dual.orientation(), Orientation::Rotate180);
    assert_eq!(dual.base().orientation(), Orientation::Rotate180);
    assert_eq!(dual.gain_map().orientation(), Orientation::Rotate180);
    let dual_parts = dual.into_parts();
    let dual_rt = DualLayerHdr::try_from_parts(dual_parts).unwrap();
    assert_eq!(dual_rt.orientation(), Orientation::Rotate180);

    // Contradictory explicit orientation must fail!
    assert_eq!(
        *DualLayerHdr::new_with_alternate(
            base_r90.clone(),
            gm_r90.clone(),
            meta,
            None,
            Orientation::Rotate270,
        )
        .unwrap_err()
        .error(),
        BufferError::IncompatibleLayout
    );
    assert_eq!(
        *DualLayerHdrView::new(
            base_r90.as_view(),
            gm_id.as_view(),
            meta,
            None,
            Orientation::Rotate90,
        )
        .unwrap_err()
        .error(),
        BufferError::IncompatibleLayout
    );

    // Invalid gain-map metadata returns dedicated `BufferError::InvalidGainMapMetadata`.
    let mut bad_meta = meta;
    bad_meta.alternate_hdr_headroom = bad_meta.base_hdr_headroom; // H_base == H_alt invalid!
    assert_eq!(
        *DualLayerHdr::new(
            PixelBuf::new(4, 2, PixelFormat::RGB8_SRGB).unwrap(),
            gm_id,
            bad_meta,
        )
        .unwrap_err()
        .error(),
        BufferError::InvalidGainMapMetadata
    );
}

#[test]
fn planar_zero_copy_codec_apis_and_strict_role_validation() {
    // 1. Zero-copy typed sample plane construction + row_as + code_u32 + CodePlane bridge.
    let mut y_samples: Vec<u16> = vec![100, 200, 300, 400, 500, 600, 700, 800];
    let y_ptr = y_samples.as_ptr();
    let y_spec = PlaneSpec::new(PlaneRole::Y, SampleType::U16).with_packing(BitPacking::U10_LSB);

    {
        let mut pvm = PlaneViewMut::from_samples(y_spec, 4, 2, 4, &mut y_samples).unwrap();
        assert_eq!(pvm.row_as::<u16>(0).unwrap(), &[100, 200, 300, 400]);
        pvm.row_as_mut::<u16>(1).unwrap()[0] = 512;
        let mut cp = pvm.into_code_plane_mut(0).unwrap();
        cp.set_code(3, 1, 1023).unwrap();
    }
    assert_eq!(y_samples[4], 512);
    assert_eq!(y_samples[7], 1023);

    let pv = PlaneView::from_samples(y_spec, 4, 2, 4, &y_samples).unwrap();
    assert_eq!(pv.data().as_ptr() as *const u16, y_ptr);
    assert_eq!(pv.row_as::<u16>(1).unwrap(), &[512, 600, 700, 1023]);
    assert_eq!(pv.code_u32(3, 1, 0), Some(1023));
    let cp = pv.as_code_plane(0).unwrap();
    assert_eq!(cp.code(3, 1), Some(1023));

    let owned = OwnedPlane::from_samples(y_spec, 4, 2, 4, y_samples).unwrap();
    assert_eq!(owned.as_view().data().as_ptr() as *const u16, y_ptr);
    assert_eq!(owned.row_as::<u16>(0).unwrap(), &[100, 200, 300, 400]);

    // 2. PlanarBuf::new_aligned + auxiliary roles + plane_by_role + hdr/origin/orientation propagation.
    let specs = [
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
        PlaneSpec::new(PlaneRole::Depth, SampleType::U16),
        PlaneSpec::new(PlaneRole::Matte, SampleType::U8),
        PlaneSpec::new(PlaneRole::Thermal, SampleType::F32),
    ];
    let hdr = HdrMetadata {
        cll: Some(ContentLightLevel::new(1000, 400)),
        mastering: None,
    };
    let origin = ColorOrigin::from_cicp(Cicp::SRGB);
    let pbuf = PlanarBuf::new_aligned(8, 6, &specs, RowAlign::Cache64)
        .unwrap()
        .with_hdr(hdr)
        .with_origin(origin.clone())
        .with_orientation(Orientation::Rotate270);
    assert_eq!(pbuf.plane_by_role(PlaneRole::Y).unwrap().stride() % 64, 0);
    assert!(pbuf.plane_by_role(PlaneRole::Depth).is_some());
    assert!(pbuf.plane_by_role(PlaneRole::Matte).is_some());
    assert!(pbuf.plane_by_role(PlaneRole::Thermal).is_some());

    let pview = pbuf.as_view();
    assert_eq!(pview.hdr(), hdr);
    assert_eq!(pview.origin(), Some(&origin));
    assert_eq!(pview.orientation(), Orientation::Rotate270);
    assert_eq!(pview.display_dimensions(), (6, 8));
    assert!(pview.plane_by_role(PlaneRole::Cb).is_some());

    let cropped = pview.crop(2, 2, 4, 2).unwrap();
    assert_eq!(cropped.hdr(), hdr);
    assert_eq!(cropped.origin(), Some(&origin));
    assert_eq!(cropped.orientation(), Orientation::Rotate270);

    // 3. Strict `validate_plane_role_set` rejections.
    let bad_sets: &[&[PlaneSpec]] = &[
        // Mix planar Cb with semi-planar CbCr
        &[
            PlaneSpec::new(PlaneRole::Y, SampleType::U8),
            PlaneSpec::new(PlaneRole::Cb, SampleType::U8),
            PlaneSpec::new(PlaneRole::CbCr, SampleType::U8),
        ],
        // Mix CbCr and CrCb
        &[
            PlaneSpec::new(PlaneRole::Y, SampleType::U8),
            PlaneSpec::new(PlaneRole::CbCr, SampleType::U8),
            PlaneSpec::new(PlaneRole::CrCb, SampleType::U8),
        ],
        // Orphan Cb without Cr
        &[
            PlaneSpec::new(PlaneRole::Y, SampleType::U8),
            PlaneSpec::new(PlaneRole::Cb, SampleType::U8),
        ],
        // Chroma without Y
        &[
            PlaneSpec::new(PlaneRole::Cb, SampleType::U8),
            PlaneSpec::new(PlaneRole::Cr, SampleType::U8),
        ],
        // Mix YCbCr with RGB
        &[
            PlaneSpec::new(PlaneRole::Y, SampleType::U8),
            PlaneSpec::new(PlaneRole::R, SampleType::U8),
            PlaneSpec::new(PlaneRole::G, SampleType::U8),
            PlaneSpec::new(PlaneRole::B, SampleType::U8),
        ],
        // Partial RGB (missing B)
        &[
            PlaneSpec::new(PlaneRole::R, SampleType::U8),
            PlaneSpec::new(PlaneRole::G, SampleType::U8),
        ],
        // Partial CMYK (missing Key)
        &[
            PlaneSpec::new(PlaneRole::Cyan, SampleType::U8),
            PlaneSpec::new(PlaneRole::Magenta, SampleType::U8),
            PlaneSpec::new(PlaneRole::Yellow, SampleType::U8),
        ],
        // Mix Packed with Y
        &[
            PlaneSpec::new(PlaneRole::Packed, SampleType::U8),
            PlaneSpec::new(PlaneRole::Y, SampleType::U8),
        ],
        // Duplicate Extra(0)
        &[
            PlaneSpec::new(PlaneRole::Extra(0), SampleType::U8),
            PlaneSpec::new(PlaneRole::Extra(0), SampleType::U8),
        ],
    ];
    for (idx, set) in bad_sets.iter().enumerate() {
        let err = PlanarBuf::new(4, 4, set).unwrap_err();
        assert_eq!(
            *err.error(),
            BufferError::IncompatibleLayout,
            "bad plane set #{idx} should fail with IncompatibleLayout"
        );
    }

    // Valid complete PlanarRgb (R + G + B + Alpha) succeeds!
    let rgb_planes = [
        PlaneSpec::new(PlaneRole::R, SampleType::F32),
        PlaneSpec::new(PlaneRole::G, SampleType::F32),
        PlaneSpec::new(PlaneRole::B, SampleType::F32),
        PlaneSpec::new(PlaneRole::Alpha, SampleType::F32),
    ];
    let rgb_pbuf = PlanarBuf::new(4, 4, &rgb_planes).unwrap();
    assert_eq!(rgb_pbuf.planes().len(), 4);
    assert!(PlanarView::new(4, 4, rgb_pbuf.as_view().planes()).is_ok());
}
