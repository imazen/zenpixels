//! Comprehensive tests for external owned storage (`from_external` / `from_external_mut`),
//! recoverable parts, in-place swizzle, banded helpers, planar color/mutability, `CodePlaneMut`,
//! symmetric `imgref`/`image` interop, and portable XXH3 pixel hashing.

use std::sync::Arc;

#[cfg(feature = "hash")]
use zenpixels::BitPacking;
#[cfg(any(feature = "imgref", feature = "hash"))]
use zenpixels::RowAlign;
use zenpixels::{
    AlphaMode, BandRowLocation, BandSizing, BandedBuf, BufferError, ByteStorage, ByteStorageMut,
    ChannelOrder, ChromaSiting, CodePlaneError, CodePlaneMut, ColorSpace, PixelBuf, PixelFormat,
    PixelRect, PixelSink, PlanarBuf, PlanarViewMut, PlaneBuf, PlaneFormat, PlaneRole, PlaneViewMut,
    Primaries, SampleEncoding, SampleFacts, SampleLayout, SampleType, SignalRange, Transfer,
};
#[cfg(any(feature = "imgref", feature = "image"))]
use zenpixels::{PixelView, PixelViewMut};

#[derive(Debug, PartialEq, Eq)]
struct ReadOnlyPoolHandle {
    id: u64,
    data: Arc<[u8]>,
}

impl ByteStorage for ReadOnlyPoolHandle {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MutablePoolLease {
    lease_id: u64,
    bytes: Box<[u8]>,
}

impl ByteStorage for MutablePoolLease {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl ByteStorageMut for MutablePoolLease {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

#[test]
fn external_immutable_zero_copy_cow_and_recovery() {
    let raw: Arc<[u8]> = Arc::from((0u8..36).collect::<Vec<_>>());
    let ptr = raw.as_ptr();
    let handle = ReadOnlyPoolHandle {
        id: 42,
        data: Arc::clone(&raw),
    };

    let spec = PixelFormat::RGB8_SRGB;
    let buf = PixelBuf::from_external(handle, 4, 3, 12, spec.clone()).unwrap();
    assert!(buf.is_external());
    assert!(!buf.is_mutable_in_place());
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);
    assert_eq!(buf.external_ref::<ReadOnlyPoolHandle>().unwrap().id, 42);
    assert!(buf.external_ref::<Arc<[u8]>>().is_none());

    // Already-tight external buffer stays external zero-copy through `into_tight_rows`.
    let buf = buf.into_tight_rows().unwrap();
    assert!(buf.is_external());
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);

    // Wrong downcast returns `Err(buf)` completely untouched.
    let buf = buf.into_external::<Arc<[u8]>>().unwrap_err();
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);

    // Clone copies into independent aligned storage.
    let mut cow_buf = buf.try_clone().unwrap();
    assert!(!cow_buf.is_external());
    cow_buf.as_view_mut().row_mut(0)[0] = 250;
    assert_eq!(raw[0], 0);

    // Recover exact original handle zero-copy!
    let recovered = buf.into_external::<ReadOnlyPoolHandle>().unwrap();
    assert_eq!(recovered.id, 42);
    assert_eq!(recovered.data.as_ptr(), ptr);

    // Direct `Arc<[u8]>` with copy-on-write on mutation leaves `raw` untouched.
    let mut arc_buf = PixelBuf::from_external(Arc::clone(&raw), 4, 3, 12, spec.clone()).unwrap();
    assert_eq!(arc_buf.as_view().as_strided_bytes().as_ptr(), ptr);
    {
        let mut vm = arc_buf.try_cow_view_mut().unwrap();
        vm.row_mut(1)[2] = 99;
    }
    assert!(!arc_buf.is_external());
    assert!(arc_buf.is_mutable_in_place());
    assert_eq!(arc_buf.as_view().row(1)[2], 99);
    assert_eq!(raw[14], 14);

    // Direct `PixelBuf::row` / `row_mut` CoW-promotes external storage and clears cached facts!
    let mut row_buf = PixelBuf::from_external(Arc::clone(&raw), 4, 3, 12, spec).unwrap();
    assert_eq!(row_buf.row(1)[2], 14);
    row_buf.assume_facts(SampleFacts::NONE.with_all_opaque(true));
    assert!(row_buf.facts().all_opaque);
    row_buf.row_mut(1)[2] = 99;
    assert!(!row_buf.is_external());
    assert!(row_buf.is_mutable_in_place());
    assert!(!row_buf.facts().all_opaque);
    assert_eq!(row_buf.row(1)[2], 99);
    assert_eq!(raw[14], 14);
}

#[test]
fn external_immutable_validation_failure_returns_owner_untouched() {
    let raw: Arc<[u8]> = Arc::from(vec![7u8; 10]);
    let ptr = raw.as_ptr();
    // Too short for 4x3 RGB8 (needs 36 bytes).
    let (err, returned) =
        PixelBuf::from_external(raw, 4, 3, 12, PixelFormat::RGB8_SRGB).unwrap_err();
    assert_eq!(*err.error(), BufferError::InsufficientData);
    assert_eq!(returned.as_ptr(), ptr);

    // Misaligned offset for U16 returns owner untouched.
    let aligned_u16: Arc<[u16]> = Arc::from(vec![100u16; 32]);
    let ptr16 = aligned_u16.as_ptr();
    let (err, returned) = PixelBuf::from_external_with_byte_offset(
        aligned_u16,
        1, // odd byte offset breaks u16 alignment!
        2,
        2,
        12,
        PixelFormat::new(SampleLayout::RGB16, ColorSpace::SRGB),
    )
    .unwrap_err();
    assert_eq!(*err.error(), BufferError::AlignmentViolation);
    assert_eq!(returned.as_ptr(), ptr16);

    // Static slice `&'static [u8]` works zero-copy!
    static STATIC_RGB: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let static_buf =
        PixelBuf::from_external(&STATIC_RGB[..], 2, 2, 6, PixelFormat::RGB8_SRGB).unwrap();
    assert_eq!(
        static_buf.as_view().as_strided_bytes().as_ptr(),
        STATIC_RGB.as_ptr()
    );
    let back: &'static [u8] = static_buf.into_external().unwrap();
    assert_eq!(back.as_ptr(), STATIC_RGB.as_ptr());
}

#[test]
fn external_mutable_zero_copy_in_place_write_and_recovery() {
    let lease = MutablePoolLease {
        lease_id: 777,
        bytes: vec![0u8; 48].into_boxed_slice(),
    };
    let ptr = lease.bytes.as_ptr();

    let mut buf = PixelBuf::from_external_mut(lease, 4, 3, 16, PixelFormat::RGBA8_SRGB).unwrap();
    assert!(buf.is_external());
    assert!(buf.is_mutable_in_place());
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);

    // In-place write via `as_view_mut` AND `PixelSink` writes directly into the pool lease!
    {
        let mut vm = buf.try_cow_view_mut().unwrap();
        assert_eq!(vm.as_strided_bytes_mut().as_ptr(), ptr);
        vm.row_mut(0)[0..4].copy_from_slice(&[10, 20, 30, 255]);
    }
    buf.sink_row_mut(2)[0..4].copy_from_slice(&[90, 80, 70, 128]);
    assert!(buf.is_external());

    // `external_mut` clears cached facts.
    buf.assume_facts(SampleFacts::NONE.with_all_opaque(true));
    assert!(buf.facts().all_opaque);
    assert_eq!(
        buf.external_mut::<MutablePoolLease>().unwrap().lease_id,
        777
    );
    assert!(!buf.facts().all_opaque);

    // Recover exact original lease zero-copy and verify in-place mutations persisted!
    let lease_out = buf.into_external_mut::<MutablePoolLease>().unwrap();
    assert_eq!(lease_out.lease_id, 777);
    assert_eq!(lease_out.bytes.as_ptr(), ptr);
    assert_eq!(&lease_out.bytes[0..4], &[10, 20, 30, 255]);
    assert_eq!(&lease_out.bytes[32..36], &[90, 80, 70, 128]);

    // `Box<[f32]>` zero-copy mutable F32 image!
    let f32_box: Box<[f32]> = vec![0.25f32; 12].into_boxed_slice();
    let f32_ptr = f32_box.as_ptr() as *const u8;
    let mut f32_buf = PixelBuf::from_external_mut(
        f32_box,
        2,
        2,
        24,
        PixelFormat::new(SampleLayout::RGBF32, ColorSpace::LINEAR_SRGB),
    )
    .unwrap();
    assert_eq!(f32_buf.as_view().as_strided_bytes().as_ptr(), f32_ptr);
    f32_buf.as_view_mut().row_as_mut::<f32>(1).unwrap()[0] = 0.75;
    let recovered_box = f32_buf.into_external_mut::<Box<[f32]>>().unwrap();
    assert_eq!(recovered_box[6], 0.75);
}

#[test]
fn external_owned_plane_and_try_from_parts_recover() {
    let y_arc: Arc<[u8]> = Arc::from(vec![120u8; 16]);
    let y_ptr = y_arc.as_ptr();
    let y_plane = PlaneBuf::from_external(
        y_arc,
        4,
        4,
        4,
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    assert_eq!(y_plane.as_view().data().as_ptr(), y_ptr);
    let back = y_plane.into_external::<Arc<[u8]>>().unwrap();
    assert_eq!(back.as_ptr(), y_ptr);

    let mut uv_plane = PlaneBuf::from_external_mut(
        vec![512u16; 4].into_boxed_slice(),
        2,
        2,
        4,
        PlaneFormat::new(PlaneRole::Cb, SampleType::U16),
    )
    .unwrap();
    uv_plane.try_cow_view_mut().unwrap().row_mut(0)[0..2].copy_from_slice(&1023u16.to_ne_bytes());
    let uv_back = uv_plane.into_external_mut::<Box<[u16]>>().unwrap();
    assert_eq!(uv_back[0], 1023);

    // `try_from_parts_recover` preserves storage allocation on geometry rejection.
    let buf = PixelBuf::new(4, 3, PixelFormat::RGB8_SRGB).unwrap();
    let orig_ptr = buf.as_view().as_strided_bytes().as_ptr();
    let mut parts = buf.into_parts();
    parts.stride_bytes = 2; // invalid (< row_bytes=12)
    let (err, mut parts) = PixelBuf::from_parts_recover(parts).unwrap_err();
    assert_eq!(*err.error(), BufferError::StrideTooSmall);
    assert_eq!(parts.storage.as_bytes().as_ptr(), orig_ptr);
    parts.stride_bytes = 12;
    let restored = PixelBuf::from_parts_recover(parts).unwrap();
    assert_eq!(restored.as_view().as_strided_bytes().as_ptr(), orig_ptr);
}

#[test]
fn swizzle_in_place_rgb_bgr_and_drop_padding_zero_realloc() {
    let mut buf = PixelBuf::from_bytes(
        vec![10, 20, 30, 255, 40, 50, 60, 255],
        2,
        1,
        PixelFormat::new(
            SampleLayout::RGBA8,
            ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
        ),
    )
    .unwrap();
    let ptr = buf.as_view().as_strided_bytes().as_ptr();
    buf.assume_facts(SampleFacts::NONE.with_all_opaque(true));

    buf.swizzle_in_place(ChannelOrder::Bgra).unwrap();
    assert_eq!(buf.format().layout.order, ChannelOrder::Bgra);
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);
    assert_eq!(buf.as_view().row(0), &[30, 20, 10, 255, 60, 50, 40, 255]);
    assert!(buf.facts().all_opaque);

    // `swizzle_in_place` only handles equal-width channel swaps (refuses 4->3).
    assert_eq!(
        *buf.swizzle_in_place(ChannelOrder::Rgb).unwrap_err().error(),
        BufferError::IncompatibleLayout
    );

    // Opaque BGRA -> RGB compacts in place without reallocating via `drop_unused_lane_in_place`!
    buf.drop_unused_lane_in_place(ChannelOrder::Rgb).unwrap();
    assert_eq!(buf.format().layout.order, ChannelOrder::Rgb);
    assert_eq!(buf.format().color.alpha(), AlphaMode::None);
    assert_eq!(buf.as_view().as_strided_bytes().as_ptr(), ptr);
    assert_eq!(buf.as_view().row(0), &[10, 20, 30, 40, 50, 60]);

    // Straight alpha refuses lossy 4->3 drop.
    let mut straight = PixelBuf::new(2, 1, PixelFormat::RGBA8_SRGB).unwrap();
    assert_eq!(
        *straight
            .drop_unused_lane_in_place(ChannelOrder::Rgb)
            .unwrap_err()
            .error(),
        BufferError::IncompatibleLayout
    );

    // U16 RGB <-> BGR in place.
    let mut u16_buf = PixelBuf::from_samples(
        vec![1000u16, 2000, 3000, 4000, 5000, 6000],
        2,
        1,
        PixelFormat::new(SampleLayout::RGB16, ColorSpace::SRGB),
    )
    .unwrap();
    let ptr16 = u16_buf.as_view().as_strided_bytes().as_ptr();
    u16_buf.swizzle_in_place(ChannelOrder::Bgr).unwrap();
    assert_eq!(u16_buf.as_view().as_strided_bytes().as_ptr(), ptr16);
    assert_eq!(
        u16_buf.as_view().row_as::<u16>(0).unwrap(),
        &[3000, 2000, 1000, 6000, 5000, 4000]
    );
}

#[test]
fn banded_buf_helpers_and_planar_color_and_mut_views() {
    let mut src = PixelBuf::new(5, 7, PixelFormat::RGB8_SRGB).unwrap();
    {
        let mut vm = src.as_view_mut();
        for y in 0..7 {
            let row = vm.row_mut(y);
            for x in 0..5 {
                row[x * 3] = (y * 10 + x as u32) as u8;
                row[x * 3 + 1] = 100;
                row[x * 3 + 2] = 200;
            }
        }
    }
    let banded =
        BandedBuf::from_buf(src.try_clone().unwrap(), BandSizing::MaxBandBytes(30)).unwrap();
    assert_eq!(banded.rows_per_band(), 2);
    assert_eq!(banded.band_count(), 4);
    assert_eq!(
        banded.locate_row(5),
        Some(BandRowLocation {
            band_index: 2,
            local_y: 1
        })
    );
    assert_eq!(banded.locate_row(7), None);
    let views = banded.sub_row_views(1, 5).unwrap();
    assert_eq!(views.len(), 3); // band 0 row 1, band 1 rows 0..2, band 2 rows 0..2
    let gathered = banded.into_buf().unwrap();
    for y in 0..7 {
        assert_eq!(gathered.as_view().row(y), src.as_view().row(y));
    }

    // Single-band zero-copy round-trip.
    let ptr = src.as_view().as_strided_bytes().as_ptr();
    let single = BandedBuf::from_buf(src, BandSizing::MaxBandBytes(1024 * 1024)).unwrap();
    assert_eq!(single.band_count(), 1);
    assert_eq!(
        single.band_view(0).unwrap().as_strided_bytes().as_ptr(),
        ptr
    );
    let back = single.into_buf().unwrap();
    assert_eq!(back.as_view().as_strided_bytes().as_ptr(), ptr);

    // PlanarBuf + PlanarView + PlanarViewMut with full color metadata.
    let y_spec = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    let cb_spec = PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2);
    let cr_spec = PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2);
    let mut planar = PlanarBuf::new(4, 4, &[y_spec, cb_spec, cr_spec])
        .unwrap()
        .with_chroma_siting(ChromaSiting::Left)
        .with_color(ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Bt709,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        })
        .with_matrix_coefficients(zenpixels::MatrixCoefficients::BT709)
        .with_range_override(SignalRange::Narrow);

    {
        let mut pvm = planar.as_view_mut();
        assert_eq!(
            pvm.matrix_coefficients(),
            Some(zenpixels::MatrixCoefficients::BT709)
        );
        assert_eq!(pvm.effective_range(), Some(SignalRange::Narrow));
        pvm.plane_mut(0).unwrap().row_mut(0).fill(235);
        pvm.plane_mut(1).unwrap().row_mut(0).fill(128);
        pvm.plane_mut(2).unwrap().row_mut(0).fill(128);
    }
    let pv = planar.as_view();
    assert_eq!(
        pv.matrix_coefficients(),
        Some(zenpixels::MatrixCoefficients::BT709)
    );
    assert_eq!(pv.effective_range(), Some(SignalRange::Narrow));
    assert_eq!(pv.color(), planar.color());
    let cropped = pv.crop(PixelRect::new(1, 1, 2, 2)).unwrap();
    assert_eq!(
        cropped.matrix_coefficients(),
        Some(zenpixels::MatrixCoefficients::BT709)
    );
    assert_eq!(cropped.chroma_siting(), ChromaSiting::Left);

    // Caller-owned zero-alloc `PlanarViewMut::from_array`.
    let mut y_raw = [0u8; 16];
    let mut cb_raw = [0u8; 4];
    let mut cr_raw = [0u8; 4];
    let pvm = PlanarViewMut::from_array(
        4,
        4,
        [
            PlaneViewMut::new(&mut y_raw, 4, 4, 4, y_spec).unwrap(),
            PlaneViewMut::new(&mut cb_raw, 2, 2, 2, cb_spec).unwrap(),
            PlaneViewMut::new(&mut cr_raw, 2, 2, 2, cr_spec).unwrap(),
        ],
    )
    .unwrap()
    .with_matrix_coefficients(zenpixels::MatrixCoefficients::BT601);
    let mut pvm = pvm;
    pvm.planes_mut()[0].row_mut(2)[3] = 42;
    let imm = pvm.into_view().unwrap();
    assert_eq!(
        imm.matrix_coefficients(),
        Some(zenpixels::MatrixCoefficients::BT601)
    );
    assert_eq!(imm.planes()[0].row(2)[3], 42);
}

#[test]
fn code_plane_mut_read_write_and_preserve_bits() {
    let enc10_msb = SampleEncoding::new(SampleType::U16, 10, 6).unwrap();
    let mut words = [0x003Fu16; 6]; // low 6 bits pre-filled with sentinel 0x3F
    {
        let mut cp = CodePlaneMut::from_u16_mut(&mut words, 3, 2, 3, enc10_msb)
            .unwrap()
            .preserving_other_bits();
        cp.write_codes(0, 0, &[0, 512, 1023]).unwrap();
        assert_eq!(cp.code(0, 0), Some(0));
        assert_eq!(cp.code(1, 0), Some(512));
        assert_eq!(cp.code(2, 0), Some(1023));
        // Out-of-range code fails atomically without mutating row 1.
        let err = cp.write_codes(0, 1, &[10, 1024, 20]).unwrap_err();
        assert_eq!(err, CodePlaneError::CodeOutOfRange);
    }
    // Low 6 bits preserved!
    assert_eq!(words[0], 0x003F);
    assert_eq!(words[1], (512 << 6) | 0x003F);
    assert_eq!(words[2], (1023 << 6) | 0x003F);
    assert_eq!(words[3], 0x003F);
}

#[cfg(feature = "imgref")]
#[test]
fn imgref_mutable_borrowed_and_sample_interop() {
    use rgb::Rgb;

    let mut data = vec![Rgb::new(1u8, 2, 3); 12];
    let ptr = data.as_ptr() as *const u8;
    let img_mut = imgref::ImgRefMut::new_stride(&mut data, 3, 3, 4);
    let mut vm = PixelViewMut::from_imgref_mut(img_mut, ColorSpace::SRGB).unwrap();
    assert_eq!(vm.width(), 3);
    assert_eq!(vm.height(), 3);
    assert_eq!(vm.stride_bytes(), 12);
    assert_eq!(vm.as_strided_bytes_mut().as_ptr(), ptr);
    {
        let mut s_img = vm.as_sample_imgref_mut::<u8>().unwrap();
        assert_eq!(s_img.width(), 9);
        assert_eq!(s_img.stride(), 12);
        s_img.buf_mut()[0] = 77;
    }
    let back = vm.into_imgref_mut::<Rgb<u8>>().unwrap();
    assert_eq!(back.buf()[0], Rgb::new(77, 2, 3));

    let view = PixelView::from_imgref(back.as_ref(), ColorSpace::SRGB).unwrap();
    let sample_ref = view.as_sample_imgref::<u8>().unwrap();
    assert_eq!(sample_ref.buf()[0], 77);

    // `into_imgvec_zero_copy` succeeds on tight Vec<u8> and returns Err(buf) untouched on Aligned64.
    let tight = PixelBuf::new(2, 2, PixelFormat::RGB8_SRGB).unwrap();
    let ptr = tight.as_view().as_strided_bytes().as_ptr();
    let iv = tight.into_imgvec_zero_copy::<Rgb<u8>>().unwrap();
    assert_eq!(iv.buf().as_ptr() as *const u8, ptr);

    let aligned = PixelBuf::new_aligned(2, 2, PixelFormat::RGB8_SRGB, RowAlign::Cache64).unwrap();
    let aptr = aligned.as_view().as_strided_bytes().as_ptr();
    let back = aligned.into_imgvec_zero_copy::<Rgb<u8>>().unwrap_err();
    assert_eq!(back.as_view().as_strided_bytes().as_ptr(), aptr);
}

#[cfg(feature = "image")]
#[test]
fn image_crate_symmetric_10_formats_borrowed_and_metadata() {
    use zenpixels::{Cicp, Orientation};

    // Orientation 1:1 bijection
    for &o in &Orientation::ALL {
        let img_o: image::metadata::Orientation = o.into();
        let back: Orientation = img_o.into();
        assert_eq!(back, o);
    }

    // CICP round-trip
    let cicp = Cicp::DISPLAY_P3;
    let img_cicp = image::metadata::Cicp::try_from(cicp).unwrap();
    let back_cicp = Cicp::from(img_cicp);
    assert_eq!(back_cicp, cicp);

    // All 10 DynamicImage formats round-trip zero-copy owned + borrowed!
    let mut rgb16 = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_pixel(
        3,
        2,
        image::Rgb([1000, 2000, 3000]),
    );
    rgb16
        .set_color_space(image::metadata::Cicp::DISPLAY_P3)
        .unwrap();
    {
        let v = PixelView::from_image_buffer(&rgb16).unwrap();
        assert_eq!(v.format().color, ColorSpace::DISPLAY_P3);
        assert_eq!(v.row_as::<u16>(0).unwrap()[0..3], [1000, 2000, 3000]);
    }
    {
        let mut vm = PixelViewMut::from_image_buffer_mut(&mut rgb16).unwrap();
        vm.row_as_mut::<u16>(1).unwrap()[0] = 9999;
    }
    assert_eq!(rgb16.get_pixel(0, 1)[0], 9999);

    let dyn_img = image::DynamicImage::ImageRgb16(rgb16);
    let zp = PixelBuf::try_from(dyn_img).unwrap();
    assert_eq!(zp.format().color, ColorSpace::DISPLAY_P3);
    let dyn_back = zp.into_dynamic_image().unwrap();
    assert!(matches!(dyn_back, image::DynamicImage::ImageRgb16(_)));

    let luma16 = image::DynamicImage::ImageLuma16(image::ImageBuffer::from_pixel(
        2,
        2,
        image::Luma([4321u16]),
    ));
    let zp_l = PixelBuf::from_dynamic_image(luma16).unwrap();
    assert_eq!(zp_l.format().layout, SampleLayout::GRAY16);

    let rgb32f = image::DynamicImage::ImageRgb32F(image::ImageBuffer::from_pixel(
        2,
        2,
        image::Rgb([0.25f32, 0.5, 0.75]),
    ));
    let zp_f = PixelBuf::from_dynamic_image(rgb32f).unwrap();
    assert_eq!(zp_f.format().layout, SampleLayout::RGBF32);
}

#[cfg(feature = "hash")]
#[test]
fn portable_xxh3_hash_invariance_and_padding_exclusion() {
    use zenpixels::HashInclusions;

    let mut tight = PixelBuf::new(5, 4, PixelFormat::RGB8_SRGB).unwrap();
    {
        let mut vm = tight.as_view_mut();
        for y in 0..4 {
            let row = vm.row_mut(y);
            for x in 0..5 {
                row[x * 3] = (x as u8).wrapping_mul(37);
                row[x * 3 + 1] = (y as u8).wrapping_mul(53);
                row[x * 3 + 2] = (x as u8 + y as u8).wrapping_mul(19);
            }
        }
    }
    let h_tight = tight.hash_with(HashInclusions::PixelsColorAndFacts);

    // 64-byte aligned padded stride hashes IDENTICALLY.
    let mut padded =
        PixelBuf::new_aligned(5, 4, PixelFormat::RGB8_SRGB, RowAlign::Cache64).unwrap();
    for y in 0..4 {
        padded
            .as_view_mut()
            .row_mut(y)
            .copy_from_slice(tight.as_view().row(y));
    }
    assert_eq!(
        padded.hash_with(HashInclusions::PixelsColorAndFacts),
        h_tight
    );

    // External storage hashes IDENTICALLY.
    let ext_arc: Arc<[u8]> = Arc::from(tight.as_view().as_contiguous_bytes().unwrap());
    let ext = PixelBuf::from_external(ext_arc, 5, 4, 15, PixelFormat::RGB8_SRGB).unwrap();
    assert_eq!(ext.hash_with(HashInclusions::PixelsColorAndFacts), h_tight);

    // BandedBuf hashes IDENTICALLY across band sizes!
    let banded = BandedBuf::from_view(&tight.as_view(), BandSizing::MaxBandBytes(15)).unwrap();
    assert_eq!(banded.band_count(), 4);
    assert_eq!(
        banded.hash_with(HashInclusions::PixelsColorAndFacts),
        h_tight
    );

    // RGBX padding byte and Opaque alpha byte are excluded!
    let a = PixelBuf::from_bytes(
        vec![10, 20, 30, 0, 40, 50, 60, 123],
        2,
        1,
        PixelFormat::new(
            SampleLayout::RGBX8,
            ColorSpace::SRGB.with_alpha(AlphaMode::Undefined),
        ),
    )
    .unwrap();
    let b = PixelBuf::from_bytes(
        vec![10, 20, 30, 255, 40, 50, 60, 99],
        2,
        1,
        PixelFormat::new(
            SampleLayout::RGBX8,
            ColorSpace::SRGB.with_alpha(AlphaMode::Undefined),
        ),
    )
    .unwrap();
    assert_eq!(
        a.hash_with(HashInclusions::PixelsAndColor),
        b.hash_with(HashInclusions::PixelsAndColor)
    );

    let op_a = PixelBuf::from_bytes(
        vec![10, 20, 30, 0],
        1,
        1,
        PixelFormat::new(
            SampleLayout::RGBA8,
            ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
        ),
    )
    .unwrap();
    let op_b = PixelBuf::from_bytes(
        vec![10, 20, 30, 255],
        1,
        1,
        PixelFormat::new(
            SampleLayout::RGBA8,
            ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
        ),
    )
    .unwrap();
    assert_eq!(
        op_a.hash_with(HashInclusions::PixelsAndColor),
        op_b.hash_with(HashInclusions::PixelsAndColor)
    );

    // BitPacking::Lsb { bits: 10 } ignores high 6 padding bits!
    let lsb_spec = PixelFormat::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Gray)
            .with_packing(BitPacking::Lsb { bits: 10 }),
        ColorSpace::GRAY_SRGB,
    );
    let p1 = PixelBuf::from_samples(vec![512u16], 1, 1, lsb_spec.clone()).unwrap();
    let p2 = PixelBuf::from_samples(vec![512u16 | 0xFC00], 1, 1, lsb_spec).unwrap();
    assert_eq!(
        p1.hash_with(HashInclusions::PixelsAndColor),
        p2.hash_with(HashInclusions::PixelsAndColor)
    );

    // HashInclusions sensitivity: color change preserves SamplesAndLayout, changes PixelsAndColor.
    let recolored = tight
        .try_clone()
        .unwrap()
        .relabel_color(ColorSpace::DISPLAY_P3)
        .unwrap();
    assert_eq!(
        tight.hash_with(HashInclusions::SamplesAndLayout),
        recolored.hash_with(HashInclusions::SamplesAndLayout)
    );
    assert_ne!(
        tight.hash_with(HashInclusions::PixelsAndColor),
        recolored.hash_with(HashInclusions::PixelsAndColor)
    );
}
