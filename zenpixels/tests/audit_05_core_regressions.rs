//! Comprehensive regression tests for zenpixels 0.5 core invariants
//! (allocation alignment, zero-copy error recovery, strided gather, BandedBuf,
//! BitPacking, ImageSpec validation, ICC intent matching, DualLayerHdrBuf,
//! sealed planes, and image-crate CICP interop).

extern crate alloc;
use alloc::sync::Arc;

#[cfg(feature = "icc")]
use zenpixels::IccColorSpace;
use zenpixels::{
    AlignedSamples, AlphaMode, BandedBuf, BitPacking, ColorSpace, DualLayerHdrBuf, GainMapMetadata,
    HdrMetadata, HlgMode, ImageSpec, PixelBuf, PixelSink, PixelSource, PixelStorage, PlanarBuf,
    PlanarView, PlaneBuf, PlaneRole, PlaneSpec, PlaneView, RenderingIntent, SampleFacts,
    SampleLayout, SampleType, SignalRange, Transfer,
};

#[test]
fn pixel_buf_clone_preserves_alignment_and_with_spec_checks_bpp() {
    let mut buf = PixelBuf::new(4, 2, ImageSpec::RGBAF32_LINEAR).unwrap();
    buf.as_view_mut().row_as_mut::<f32>(0).unwrap()[0] = 0.75;
    let cloned = buf.clone();
    assert_eq!(cloned.as_view().row_as::<f32>(0).unwrap()[0], 0.75);
    assert_eq!(cloned.spec(), buf.spec());

    // Aligned64 backing for wide RGBAF32 layout: Clone must also satisfy f32 alignment.
    let aligned_bytes = AlignedSamples::zeroed_bytes(4 * 2 * 16).unwrap();
    let from_aligned = PixelBuf::from_storage_parts(
        PixelStorage::Aligned64(aligned_bytes),
        0,
        4,
        2,
        4 * 16,
        ImageSpec::RGBAF32_LINEAR,
    )
    .unwrap();
    let cloned_aligned = from_aligned.clone();
    let row0 = cloned_aligned.as_view().row_as::<f32>(0).unwrap();
    assert_eq!(row0.len(), 16);

    // U8-backed wide layout (constructed from a Vec<u8> whose ptr happens to be 4-aligned
    // via slice offset): Clone must preserve sample alignment rather than allocating unaligned U8!
    let raw_u8 = vec![0u8; 4 * 2 * 16 + 16];
    let ptr_addr = raw_u8.as_ptr() as usize;
    let off = (16 - (ptr_addr % 16)) % 16;
    let from_u8 = PixelBuf::from_storage_parts(
        PixelStorage::U8(raw_u8),
        off,
        4,
        2,
        4 * 16,
        ImageSpec::RGBAF32_LINEAR,
    )
    .unwrap();
    let cloned_u8 = from_u8.clone();
    assert!(
        cloned_u8.as_view().row_as::<f32>(0).is_ok(),
        "cloning U8-backed F32 buffer must preserve F32 alignment"
    );

    // relabel_spec must reject mismatched bytes_per_pixel (RGB8 -> RGBA8).
    let rgb = PixelBuf::new(2, 2, ImageSpec::RGB8_SRGB).unwrap();
    assert!(
        rgb.relabel_spec(ImageSpec::RGBA8_SRGB).is_err(),
        "relabel_spec must reject mismatched bytes_per_pixel"
    );

    // relabel_spec accepts identical bytes_per_pixel (RGB8 -> BGR8).
    let rgb2 = PixelBuf::new(2, 2, ImageSpec::RGB8_SRGB).unwrap();
    let bgr_spec = ImageSpec::new(SampleLayout::BGR8, ColorSpace::SRGB);
    let bgr = rgb2.relabel_spec(bgr_spec).unwrap();
    assert_eq!(bgr.spec().layout, SampleLayout::BGR8);
}

#[test]
fn try_into_vec_zero_copy_preserves_self_on_error_and_strided_gather_preserves_all_bytes() {
    // 1. Padded-stride U8 RGB8 buffer: try_into_vec_zero_copy::<[u8; 3]> must fail
    // AND leave the returned PixelBuf completely intact (no premature truncation!).
    let raw = vec![
        10, 20, 30, 40, 50, 60, 0xEE, 0xFF, // row 0 (2 px = 6B, stride = 8B)
        70, 80, 90, 100, 110, 120, 0xEE, 0xFF, // row 1
    ];
    let buf = PixelBuf::from_storage_parts(PixelStorage::U8(raw), 0, 2, 2, 8, ImageSpec::RGB8_SRGB)
        .unwrap();
    let recovered = match buf.into_vec_zero_copy::<[u8; 3]>() {
        Ok(_) => panic!("strided buffer must not zero-copy cast to tight Vec<[u8; 3]>"),
        Err(b) => b,
    };
    assert_eq!(recovered.as_view().row(0), &[10, 20, 30, 40, 50, 60]);
    assert_eq!(recovered.as_view().row(1), &[70, 80, 90, 100, 110, 120]);

    // 2. Tight Vec<u8> -> Vec<u32> must fail (alignment mismatch) AND preserve self untouched!
    let tight_rgba = PixelBuf::from_typed(
        vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        2,
        2,
        ImageSpec::RGBA8_SRGB,
    )
    .unwrap();
    let recovered2 = match tight_rgba.into_vec_zero_copy::<u32>() {
        Ok(_) => panic!("Vec<u8> must never zero-copy cast to Vec<u32> (alignment mismatch)"),
        Err(b) => b,
    };
    assert_eq!(recovered2.as_view().row(0), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        recovered2.as_view().row(1),
        &[9, 10, 11, 12, 13, 14, 15, 16]
    );

    // 3. Strided try_into_vec::<[u16; 2]> where row_bytes = 6 (not a multiple of 4!)
    // must sequentially gather ALL row bytes without dropping trailing bytes.
    // 2 rows x 6 bytes = 12 bytes = 3 elements of [u16; 2].
    let mut raw2 = Vec::new();
    for v in [1u16, 2, 3] {
        raw2.extend_from_slice(&v.to_ne_bytes());
    }
    raw2.extend_from_slice(&[0xAA, 0xBB]);
    for v in [4u16, 5, 6] {
        raw2.extend_from_slice(&v.to_ne_bytes());
    }
    raw2.extend_from_slice(&[0xCC, 0xDD]);
    let strided =
        PixelBuf::from_storage_parts(PixelStorage::U8(raw2), 0, 2, 2, 8, ImageSpec::RGB8_SRGB)
            .unwrap();
    let gathered: Vec<[u16; 2]> = strided.into_vec::<[u16; 2]>().unwrap();
    assert_eq!(gathered, vec![[1, 2], [3, 4], [5, 6]]);
}

#[test]
#[should_panic]
fn pixel_buf_source_row_panics_out_of_bounds_even_with_trailing_capacity() {
    let buf = PixelBuf::from_storage_parts(
        PixelStorage::U8(vec![0u8; 2 * 4 * 3]),
        0,
        2,
        1,
        6,
        ImageSpec::RGB8_SRGB,
    )
    .unwrap();
    let _ = PixelSource::source_row(&buf, 1);
}

#[test]
#[should_panic]
fn pixel_buf_sink_row_mut_panics_out_of_bounds_even_with_trailing_capacity() {
    let mut buf = PixelBuf::from_storage_parts(
        PixelStorage::U8(vec![0u8; 2 * 4 * 3]),
        0,
        2,
        1,
        6,
        ImageSpec::RGB8_SRGB,
    )
    .unwrap();
    let _ = PixelSink::sink_row_mut(&mut buf, 1);
}

#[test]
fn banded_buf_rejects_zero_band_bytes_and_tracks_facts_across_child_bands() {
    assert!(BandedBuf::new(4, 4, ImageSpec::RGBA8_SRGB, 0).is_err());

    let mut banded = BandedBuf::new(4, 4, ImageSpec::RGBA8_SRGB, 32).unwrap();
    assert!(banded.bands().len() >= 2);

    // Attach facts via sink_finish -> propagates to parent AND child bands.
    let facts = SampleFacts::NONE.with_all_opaque(true);
    banded.sink_finish(facts, HdrMetadata::default());
    assert!(banded.facts().all_opaque);
    assert!(banded.bands().iter().all(|b| b.facts().all_opaque));

    // Mutating row 2 via PixelSink::sink_row_mut must clear BOTH parent and mutated child band facts!
    let r = PixelSink::sink_row_mut(&mut banded, 2);
    r[3] = 0;
    assert!(!banded.facts().all_opaque);
    let mut y_acc = 0u32;
    for b in banded.bands() {
        let bh = b.height();
        if 2 >= y_acc && 2 < y_acc + bh {
            assert!(
                !b.facts().all_opaque,
                "child band containing mutated row 2 must have cleared facts"
            );
        }
        y_acc += bh;
    }
}

#[test]
fn bit_packing_validate_rejects_overlapping_masks_and_unused_nonzero_slots() {
    // Overlapping bitfields: ch0 = bits 0..10, ch1 = bits 8..18 (overlaps bits 8..10!)
    let overlap = BitPacking::Bitfields {
        bits: [10, 10, 10, 0],
        shifts: [0, 8, 20, 0],
    };
    assert!(overlap.validate(SampleType::U32, 3).is_err());

    // Unused channel slot (components = 3, but slot 3 has bits=2 / shift=30)
    let unused_nonzero = BitPacking::Bitfields {
        bits: [10, 10, 10, 2],
        shifts: [0, 10, 20, 30],
    };
    assert!(unused_nonzero.validate(SampleType::U32, 3).is_err());

    let unused_shift = BitPacking::Bitfields {
        bits: [10, 10, 10, 0],
        shifts: [0, 10, 20, 30],
    };
    assert!(unused_shift.validate(SampleType::U32, 3).is_err());

    // Valid RGB10A2 (4 components) still passes.
    assert!(BitPacking::RGB10A2.validate(SampleType::U32, 4).is_ok());
}

#[test]
fn image_spec_validate_rejects_hlg_display_on_gray_and_icc_signature_mismatch() {
    let gray_hlg_display = ImageSpec::new(
        SampleLayout::GRAYF32,
        ColorSpace::Gray {
            transfer: Transfer::Hlg(HlgMode::Display(
                zenpixels::HlgRendering::REFERENCE_1000_NIT,
            )),
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    assert!(
        gray_hlg_display.validate().is_err(),
        "HLG Display OOTF is undefined on Gray"
    );

    // Minimal structurally valid 132-byte ICC header with 'GRAY' at 16..20, 'mntr' at 12..16,
    // 'acsp' at 36..40, declared size = 132, and tag count = 0.
    let mut gray_icc = vec![0u8; 132];
    gray_icc[0..4].copy_from_slice(&132u32.to_be_bytes());
    gray_icc[12..16].copy_from_slice(b"mntr");
    gray_icc[16..20].copy_from_slice(b"GRAY");
    gray_icc[36..40].copy_from_slice(b"acsp");
    #[cfg(feature = "icc")]
    assert_eq!(
        zenpixels::icc::profile_color_space(&gray_icc),
        Some(IccColorSpace::Gray)
    );

    // Truncated 128-byte ICC (missing 4-byte tag count / zero declared size) MUST be rejected!
    let truncated_icc = vec![0u8; 128];
    let trunc_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Icc {
            profile: Arc::from(truncated_icc.as_slice()),
            channels: 1,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(
        trunc_spec.validate().is_err(),
        "truncated 128-byte ICC must fail structural validation"
    );

    // Gray ICC attached to RGB8 layout must be rejected!
    let mismatch = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(gray_icc.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(
        mismatch.validate().is_err(),
        "GRAY ICC profile must be rejected on 3-channel RGB layout"
    );

    // Matching Gray ICC on Gray8 layout succeeds.
    let matched = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Icc {
            profile: Arc::from(gray_icc.as_slice()),
            channels: 1,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(matched.validate().is_ok());
}

#[cfg(feature = "icc")]
#[test]
fn icc_from_icc_for_intent_rejects_absolute_colorimetric() {
    let fake_icc = vec![0u8; 128];
    assert!(
        zenpixels::icc::IccIdentification::from_icc_for_intent(
            &fake_icc,
            RenderingIntent::AbsoluteColorimetric,
            false,
        )
        .is_none()
    );
}

#[test]
fn dual_layer_hdr_and_planes_enforce_invariants() {
    use core::num::NonZeroU32;
    use zenpixels::{
        ChannelOrder, Chromaticity, Cicp, CustomPrimaries, GainMapChannels, GainMapColorSpace,
        GainMapComponent, GainMapDirection, Gamma, Orientation, Primaries, SignedFraction,
        UnsignedFraction, WhitePoint,
    };

    let untagged_gm_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let valid_base = PixelBuf::new(4, 4, ImageSpec::RGB8_SRGB).unwrap();
    let valid_gm = PixelBuf::new(2, 2, untagged_gm_spec.clone()).unwrap();
    let valid_meta = GainMapMetadata::symmetric(2.0);
    let hdr = DualLayerHdrBuf::new(valid_base, valid_gm, valid_meta)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    assert_eq!(hdr.base().width(), 4);
    assert_eq!(hdr.gain_map().width(), 2);
    assert_eq!(hdr.orientation(), Orientation::Rotate90);
    let view = hdr.as_view();
    assert_eq!(view.base().width(), 4);
    assert_eq!(view.gain_map().width(), 2);
    assert_eq!(view.orientation(), Orientation::Rotate90);
    let parts = hdr.into_parts();
    assert_eq!(parts.base.height(), 4);
    assert_eq!(parts.gain_map.height(), 2);
    assert_eq!(parts.orientation, Orientation::Rotate90);
    assert!(parts.alternate_color_space.is_none());

    // Non-RGB/Gray/ICC base (CMYK) rejected; transfer-tagged gain map rejected; bitfields gain map rejected.
    let cmyk_base = PixelBuf::new(4, 4, ImageSpec::cmyk8(None)).unwrap();
    let ok_gm = PixelBuf::new(2, 2, untagged_gm_spec.clone()).unwrap();
    assert!(DualLayerHdrBuf::new(cmyk_base, ok_gm, GainMapMetadata::symmetric(2.0)).is_err());

    let ok_base = PixelBuf::new(4, 4, ImageSpec::RGB8_SRGB).unwrap();
    let srgb_gm = PixelBuf::new(2, 2, ImageSpec::GRAY8_SRGB).unwrap();
    assert!(
        DualLayerHdrBuf::new(ok_base, srgb_gm, GainMapMetadata::symmetric(2.0)).is_err(),
        "ISO 21496-1 gain map must be ColorSpace::Untagged (numeric, no transfer curve)"
    );

    let ok_base2 = PixelBuf::new(4, 4, ImageSpec::RGB8_SRGB).unwrap();
    let rgb565_gm = PixelBuf::new(
        2,
        2,
        ImageSpec::new(
            SampleLayout::RGB565,
            ColorSpace::Untagged {
                channels: 3,
                alpha: AlphaMode::None,
            },
        ),
    )
    .unwrap();
    assert!(DualLayerHdrBuf::new(ok_base2, rgb565_gm, GainMapMetadata::symmetric(2.0)).is_err());

    // Exact unreduced rationals preserved lossless without GCD reduction, and invalid metadata rejected.
    let unreduced = SignedFraction::new(2, NonZeroU32::new(4).unwrap());
    assert_eq!(unreduced.num, 2);
    assert_eq!(unreduced.den.get(), 4);
    let zero_gamma = GainMapComponent {
        min_gain_stops: SignedFraction::ZERO,
        max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
        gamma: UnsignedFraction::ZERO,
        base_offset: SignedFraction::ZERO,
        alternate_offset: SignedFraction::ZERO,
    };
    assert!(
        GainMapMetadata::new(
            GainMapChannels::Shared(zero_gamma),
            UnsignedFraction::ZERO,
            UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            GainMapColorSpace::Base,
        )
        .is_none()
    );
    let inverted_range = GainMapComponent {
        min_gain_stops: SignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        max_gain_stops: SignedFraction::new(1, NonZeroU32::new(1).unwrap()),
        gamma: UnsignedFraction::ONE,
        base_offset: SignedFraction::ZERO,
        alternate_offset: SignedFraction::ZERO,
    };
    assert!(
        GainMapMetadata::new(
            GainMapChannels::Shared(inverted_range),
            UnsignedFraction::ZERO,
            UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            GainMapColorSpace::Base,
        )
        .is_none()
    );
    // Reverse direction (HDR base -> SDR alternate) accepted!
    let rev = GainMapMetadata::new(
        GainMapChannels::Shared(GainMapComponent {
            min_gain_stops: SignedFraction::new(-2, NonZeroU32::new(1).unwrap()),
            max_gain_stops: SignedFraction::ZERO,
            gamma: UnsignedFraction::ONE,
            base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
            alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
        }),
        UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
        UnsignedFraction::ZERO,
        GainMapColorSpace::Base,
    )
    .unwrap();
    assert_eq!(rev.direction(), Some(GainMapDirection::Reverse));

    // DCI-P3 (CICP CP 11) vs Display P3 (CICP CP 12) + Bradford adaptation + CustomPrimaries + Gamma + XYZ.
    assert_eq!(Primaries::DciP3.to_cicp(), Some(11));
    assert_eq!(Primaries::DisplayP3.to_cicp(), Some(12));
    assert_eq!(Cicp::DCI_P3.color_primaries, 11);
    let dci_to_srgb = Primaries::DciP3.gamut_matrix_to(Primaries::Bt709).unwrap();
    let dp3_to_srgb = Primaries::DisplayP3
        .gamut_matrix_to(Primaries::Bt709)
        .unwrap();
    // D65 white in DCI-P3 (adapted from DCI theater white) differs from Display P3 matrix!
    assert_ne!(dci_to_srgb, dp3_to_srgb);
    // White (1,1,1) in DCI-P3 maps to white (1,1,1) in sRGB under Bradford adaptation!
    for row in dci_to_srgb {
        let sum = row[0] + row[1] + row[2];
        assert!((sum - 1.0).abs() < 1e-4, "DCI-P3 white sum={sum}");
    }
    let prophoto = CustomPrimaries::new(
        (0.7347, 0.2653),
        (0.1596, 0.8404),
        (0.0366, 0.0001),
        Chromaticity::D50.xy(),
    )
    .unwrap();
    let pp_to_srgb = Primaries::Custom(prophoto)
        .gamut_matrix_to(Primaries::Bt709)
        .unwrap();
    for row in pp_to_srgb {
        let sum = row[0] + row[1] + row[2];
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "ProPhoto D50->D65 white sum={sum}"
        );
    }
    // A01: H.273 TC17 (SMPTE ST 428-1 DCI Cinema) has 52.37/48 white scale inside the curve,
    // distinct from pure Gamma::DCI_2_6 (power 2.6 with unit white).
    assert_eq!(Transfer::DciCinema.to_cicp(), Some(17));
    assert_eq!(Transfer::from_cicp(17), Some(Transfer::DciCinema));
    assert_eq!(Transfer::Gamma(Gamma::DCI_2_6).to_cicp(), None);
    assert_eq!(Transfer::Gamma(Gamma::GAMMA_1_8).to_cicp(), None);

    // ColorSpace::Xyz & CMYKA validation + U32 20-bit Lsb packing.
    let xyz_spec = ImageSpec::new(
        SampleLayout::XYZF32,
        ColorSpace::Xyz {
            white_point: WhitePoint::E,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    assert!(xyz_spec.validate().is_ok());
    let cmyka_spec = ImageSpec::new(
        SampleLayout::CMYKA8,
        ColorSpace::Cmyk {
            icc: None,
            inverted: false,
            alpha: AlphaMode::Straight,
        },
    );
    assert!(cmyka_spec.validate().is_ok());
    assert_eq!(ChannelOrder::Cmyka.channels(), 5);
    assert!(
        BitPacking::Lsb { bits: 20 }
            .validate(SampleType::U32, 3)
            .is_ok()
    );
    assert!(
        BitPacking::Lsb { bits: 32 }
            .validate(SampleType::U32, 3)
            .is_err()
    );

    // Sealed PlaneView & PlaneBuf validation, plus 5-plane CMYKA+Extra spillover (>4 inline).
    let y_spec = PlaneSpec::new(PlaneRole::Y, SampleType::U8);
    assert!(PlaneView::new(&[0u8; 4], 4, 2, 4, y_spec).is_err());
    let pv = PlaneView::new(&[0u8; 8], 4, 2, 4, y_spec).unwrap();
    let planar = PlanarView::new(4, 2, &[pv]).unwrap();
    assert_eq!(planar.planes().len(), 1);

    let op = PlaneBuf::from_storage(PixelStorage::U8(vec![0u8; 8]), 4, 2, 4, y_spec).unwrap();
    let pbuf = PlanarBuf::from_planes(4, 2, vec![op]).unwrap();
    assert_eq!(pbuf.planes().len(), 1);

    let six_specs = [
        PlaneSpec::new(PlaneRole::Cyan, SampleType::U8),
        PlaneSpec::new(PlaneRole::Magenta, SampleType::U8),
        PlaneSpec::new(PlaneRole::Yellow, SampleType::U8),
        PlaneSpec::new(PlaneRole::Key, SampleType::U8),
        PlaneSpec::new(PlaneRole::Alpha, SampleType::U8),
        PlaneSpec::new(PlaneRole::Extra(0), SampleType::U16),
    ];
    let six_buf = PlanarBuf::new(4, 2, &six_specs).unwrap();
    assert_eq!(six_buf.planes().len(), 6);
    let six_view = six_buf.as_view();
    assert_eq!(six_view.planes().len(), 6);
    assert_eq!(six_view.planes()[5].spec().role, PlaneRole::Extra(0));
}

#[cfg(feature = "image")]
#[test]
fn image_crate_interop_propagates_p3_cicp_and_rejects_mismatches() {
    let p3_spec = ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
    let mut buf = PixelBuf::new(2, 1, p3_spec).unwrap();
    buf.as_view_mut()
        .row_mut(0)
        .copy_from_slice(&[255, 0, 0, 0, 255, 0]);

    // Exporting RGBA PixelBuf into image::Rgb<u8> must fail!
    let rgba_buf = PixelBuf::new(2, 1, ImageSpec::RGBA8_SRGB).unwrap();
    assert!(rgba_buf.into_image_buffer::<image::Rgb<u8>>().is_err());

    // Exporting Display P3 RGB8 sets Display P3 CICP on the resulting ImageBuffer.
    let img = buf.into_image_buffer::<image::Rgb<u8>>().unwrap();
    assert_eq!(img.color_space(), image::metadata::Cicp::DISPLAY_P3);

    // from_rgb8_image_buffer rejects contradictory ColorSpace::SRGB on a Display P3 ImageBuffer!
    assert!(PixelBuf::from_rgb8_image_with_color(img.clone(), ColorSpace::SRGB).is_err());

    // from_rgb8_image preserves Display P3!
    let roundtripped = PixelBuf::from_rgb8_image(img).unwrap();
    assert_eq!(
        roundtripped.spec().color.to_cicp(),
        Some(zenpixels::Cicp::DISPLAY_P3)
    );
    assert_eq!(roundtripped.as_view().row(0), &[255, 0, 0, 0, 255, 0]);
}
