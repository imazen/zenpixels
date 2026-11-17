use zenpixels::ByteStorage;
#[cfg(feature = "rgb")]
use zenpixels::ByteStorageMut;
use zenpixels::{
    AlphaMode, ColorProfileSource, NamedProfile, PixelBuffer, PixelDescriptor, PixelFormat,
    PixelSlice, PixelSliceMut,
};

#[test]
fn reject_contradictory_descriptors_at_boundaries() {
    for d in [
        PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Premultiplied)),
        PixelDescriptor::RGBX8.with_alpha(Some(AlphaMode::Straight)),
        PixelDescriptor::RGBA8.with_alpha(None),
    ] {
        assert!(d.validate().is_err());
        assert!(PixelBuffer::try_new(1, 1, d).is_err());
        assert!(PixelSlice::new(&[0; 16], 1, 1, 4, d).is_err());
        let mut parts = PixelBuffer::new(1, 1, PixelDescriptor::RGBA8).into_parts();
        parts.descriptor = d;
        let mut error = PixelBuffer::try_from_parts(parts).unwrap_err();
        assert_eq!(error.take_parts().unwrap().data.as_bytes().len(), 4);
    }
}

#[test]
fn reinterpret_checks_new_alignment() {
    let mut data = [0u8; 16];
    let offset = (0..4)
        .find(|i| (data.as_ptr() as usize + i) % 4 != 0)
        .unwrap();
    let slice =
        PixelSlice::new(&data[offset..offset + 4], 1, 1, 4, PixelDescriptor::RGBA8).unwrap();
    assert!(
        slice
            .reinterpret(PixelFormat::GrayF32.descriptor())
            .is_err()
    );
    let slice = PixelSliceMut::new(
        &mut data[offset..offset + 4],
        1,
        1,
        4,
        PixelDescriptor::RGBA8,
    )
    .unwrap();
    assert!(
        slice
            .reinterpret(PixelFormat::GrayF32.descriptor())
            .is_err()
    );
}

#[test]
fn empty_views_do_not_compute_unused_offsets() {
    let slice = PixelSlice::new(&[], 0, u32::MAX, usize::MAX, PixelDescriptor::RGB8).unwrap();
    assert!(slice.row(u32::MAX - 1).is_empty());
    assert!(slice.sub_rows(u32::MAX - 1, 1).row(0).is_empty());
    assert_eq!(slice.as_contiguous_bytes(), Some(&[][..]));
    assert!(slice.as_strided_bytes().is_empty());
    let mut empty = [];
    let mut slice =
        PixelSliceMut::new(&mut empty, 0, u32::MAX, usize::MAX, PixelDescriptor::RGB8).unwrap();
    assert!(slice.row_mut(u32::MAX - 1).is_empty());
}

#[cfg(feature = "icc")]
#[test]
fn named_hdr_rgb_profiles_match_their_cicp() {
    for named in [NamedProfile::Bt2020Pq, NamedProfile::Bt2020Hlg] {
        assert_eq!(
            ColorProfileSource::Named(named).resolve(),
            ColorProfileSource::Cicp(named.to_cicp().unwrap()).resolve()
        );
    }
}

#[cfg(feature = "rgb")]
#[test]
fn typed_layout_changes_require_erasure() {
    let data = [1, 2, 3, 4];
    let typed = PixelSlice::<rgb::RGBA<u8>>::new_typed(&data, 1, 1, 1).unwrap();
    assert!(typed.clone().reinterpret(PixelDescriptor::BGRA8).is_err());
    let changed = typed.erase().reinterpret(PixelDescriptor::BGRA8).unwrap();
    assert!(changed.try_typed::<rgb::Bgra<u8>>().is_some());
}

#[cfg(feature = "rgb")]
#[test]
fn padded_u8_export_reuses_allocation() {
    let mut parts = PixelBuffer::new(1, 2, PixelDescriptor::RGBA8).into_parts();
    parts.data = vec![91u8; 20].into();
    parts.offset = 4;
    parts.stride_bytes = 8;
    parts.data.as_bytes_mut()[4..8].copy_from_slice(&[1, 2, 3, 4]);
    parts.data.as_bytes_mut()[12..16].copy_from_slice(&[5, 6, 7, 8]);
    let ptr = parts.data.as_bytes().as_ptr();
    let b = PixelBuffer::try_from_parts(parts).unwrap();
    let pixels = b.into_contiguous_pixels::<rgb::RGBA<u8>>().unwrap();
    assert_eq!(pixels.as_ptr().cast::<u8>(), ptr);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&pixels),
        &[1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[cfg(feature = "imgref")]
#[test]
fn zero_width_imgref_export_keeps_geometry() {
    let mut buffer = PixelBuffer::<rgb::RGBA<u8>>::new_typed(0, 7);
    assert_eq!(buffer.as_imgref().width(), 0);
    assert_eq!(buffer.as_imgref_mut().height(), 7);
    let img = imgref::ImgVec::try_from(buffer).unwrap();
    assert_eq!((img.width(), img.height()), (0, 7));

    // Also test Wx0 with stride == 0 adopted via PixelBufferParts.
    let parts = zenpixels::PixelBufferParts::new(
        Vec::<u8>::new().into(),
        7,
        0,
        0,
        PixelDescriptor::RGB8_SRGB,
    );
    let mut wz = PixelBuffer::try_from_parts(parts)
        .unwrap()
        .try_typed::<rgb::RGB8>()
        .unwrap();
    assert_eq!((wz.as_imgref().width(), wz.as_imgref().height()), (7, 0));
    assert_eq!(
        (wz.as_imgref_mut().width(), wz.as_imgref_mut().height()),
        (7, 0)
    );
    let wz_img = imgref::ImgVec::try_from(wz).unwrap();
    assert_eq!((wz_img.width(), wz_img.height()), (7, 0));
}

#[test]
fn interop_target_contracts_validate_upfront_with_zero_padding() {
    use zenpixels::{
        BufferError, ChannelLayout, ChannelType, InteropTarget, SignalRange, TransferFunction,
    };

    // Tight accepts any valid descriptor, including CMYK8 and unknown transfer.
    assert!(InteropTarget::Tight.supports(PixelDescriptor::CMYK8));
    assert!(InteropTarget::Tight.supports(PixelDescriptor::RGB8));
    let tight =
        PixelBuffer::try_new_for(5, 3, PixelDescriptor::RGB8, InteropTarget::Tight).unwrap();
    assert_eq!(tight.stride(), 15);

    // Imgref accepts typed Pixel layouts (including F16, BGR, BGRA, RGBX, and unknown transfer),
    // and rejects CMYK / Oklab / F64.
    assert!(InteropTarget::Imgref.supports(PixelDescriptor::RGB8));
    assert!(InteropTarget::Imgref.supports(PixelDescriptor::RGBF16_LINEAR));
    assert!(InteropTarget::Imgref.supports(PixelDescriptor::BGR8));
    assert!(InteropTarget::Imgref.supports(PixelDescriptor::RGBX8_SRGB));
    assert!(!InteropTarget::Imgref.supports(PixelDescriptor::CMYK8));
    assert!(!InteropTarget::Imgref.supports(PixelDescriptor::OKLABF32));
    assert_eq!(
        *PixelBuffer::try_new_for(4, 4, PixelDescriptor::CMYK8, InteropTarget::Imgref)
            .unwrap_err()
            .error(),
        BufferError::IncompatibleDescriptor
    );

    // FlatSamples accepts Gray/GrayAlpha/Rgb/Rgba in U8/U16/F16/F32, rejects BGR/BGRA/CMYK.
    assert!(InteropTarget::FlatSamples.supports(PixelDescriptor::RGBF16));
    assert!(InteropTarget::FlatSamples.supports(PixelDescriptor::GRAYF32_LINEAR));
    assert!(!InteropTarget::FlatSamples.supports(PixelDescriptor::BGR8));
    assert!(!InteropTarget::FlatSamples.supports(PixelDescriptor::BGRA8_SRGB));

    // ImageBuffer requires known CICP color, full range, non-premultiplied/non-undefined alpha, U8/U16/F32.
    assert!(InteropTarget::ImageBuffer.supports(PixelDescriptor::RGB8_SRGB));
    assert!(InteropTarget::ImageBuffer.supports(PixelDescriptor::GRAYF32_LINEAR));
    assert!(!InteropTarget::ImageBuffer.supports(PixelDescriptor::RGB8)); // Unknown transfer
    assert!(!InteropTarget::ImageBuffer.supports(PixelDescriptor::RGBF16_LINEAR)); // F16 unsupported by ImageBuffer
    assert!(
        !InteropTarget::ImageBuffer
            .supports(PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)))
    );
    assert!(!InteropTarget::ImageBuffer.supports(PixelDescriptor::RGBX8_SRGB));
    assert!(
        !InteropTarget::ImageBuffer
            .supports(PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow))
    );

    // DynamicImage matches ImageBuffer except GrayF32 / GrayAF32 have no DynamicImage variant.
    assert!(InteropTarget::DynamicImage.supports(PixelDescriptor::RGBF32_LINEAR));
    assert!(!InteropTarget::DynamicImage.supports(PixelDescriptor::GRAYF32_LINEAR));
    assert!(!InteropTarget::DynamicImage.supports(PixelDescriptor::GRAYAF32_LINEAR));

    // Invalid descriptor still surfaces InvalidDescriptor.
    let bad = PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Straight));
    assert!(!InteropTarget::Tight.supports(bad));
    assert!(matches!(
        *PixelBuffer::try_new_for(2, 2, bad, InteropTarget::ImageBuffer)
            .unwrap_err()
            .error(),
        BufferError::InvalidDescriptor(_)
    ));

    // Deep BGR16 is accepted by InteropTarget::Imgref.
    let bgr16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb,
    );
    assert!(InteropTarget::Imgref.supports(bgr16));
}

#[cfg(feature = "imgref")]
#[test]
fn imgvec_handles_unpadded_final_row_and_capacity_slop_without_mutating_on_error() {
    use zenpixels::PixelBufferParts;

    // 1. Strided 2x2 RGB8 with stride = 12 B (4 px), backing = 18 B ((2-1)*4 + 2 = 6 px, no trailing row-1 pad).
    let mut raw = vec![0u8; 18];
    raw[0..6].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    raw[12..18].copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    raw.shrink_to_fit();
    assert_eq!(raw.capacity() % 3, 0);
    let ptr = raw.as_ptr();
    let typed = PixelBuffer::try_from_parts(PixelBufferParts::new(
        raw.into(),
        2,
        2,
        12,
        PixelDescriptor::RGB8_SRGB,
    ))
    .unwrap()
    .try_typed::<rgb::RGB8>()
    .unwrap();
    let img = imgref::ImgVec::try_from(typed).unwrap();
    assert_eq!(img.buf().as_ptr().cast::<u8>(), ptr, "zero-copy move");
    assert_eq!((img.width(), img.height(), img.stride()), (2, 2, 4));
    let rows: Vec<&[rgb::RGB8]> = img.rows().collect();
    assert_eq!(rows[0], &[rgb::RGB8::new(1, 2, 3), rgb::RGB8::new(4, 5, 6)]);
    assert_eq!(
        rows[1],
        &[rgb::RGB8::new(7, 8, 9), rgb::RGB8::new(10, 11, 12)]
    );

    // 2. Capacity % 3 != 0 with trailing byte: TryFrom refuses and leaves buffer bytes completely untouched,
    //    while into_imgvec compacts cleanly.
    let mut bad_cap: Vec<u8> = Vec::with_capacity(19);
    bad_cap.resize(19, 0xCC);
    bad_cap[0..6].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    bad_cap[12..18].copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    assert_ne!(bad_cap.capacity() % 3, 0);
    let typed_bad = PixelBuffer::try_from_parts(PixelBufferParts::new(
        bad_cap.into(),
        2,
        2,
        12,
        PixelDescriptor::RGB8_SRGB,
    ))
    .unwrap()
    .try_typed::<rgb::RGB8>()
    .unwrap();
    let refused = imgref::ImgVec::try_from(typed_bad).unwrap_err();
    let check_parts = refused.erase().into_parts();
    assert_eq!(
        check_parts.data.as_bytes().len(),
        19,
        "refused TryFrom must not truncate trailing byte"
    );
    assert_eq!(check_parts.data.as_bytes()[18], 0xCC);
    let recovered = PixelBuffer::try_from_parts(check_parts)
        .unwrap()
        .try_typed::<rgb::RGB8>()
        .unwrap();
    let compacted = recovered.into_imgvec();
    assert_eq!(
        (compacted.width(), compacted.height(), compacted.stride()),
        (2, 2, 2)
    );
    assert_eq!(
        compacted.buf(),
        &[
            rgb::RGB8::new(1, 2, 3),
            rgb::RGB8::new(4, 5, 6),
            rgb::RGB8::new(7, 8, 9),
            rgb::RGB8::new(10, 11, 12),
        ]
    );

    // 3. Capacity % 3 == 0 (21 B) with 2 trailing slop bytes (len == 20): TryFrom trims slop to 18 B and moves zero-copy!
    let mut slop_ok: Vec<u8> = Vec::with_capacity(21);
    assert_eq!(slop_ok.capacity(), 21);
    slop_ok.resize(20, 0xEE);
    slop_ok[0..6].copy_from_slice(&[9, 8, 7, 6, 5, 4]);
    slop_ok[12..18].copy_from_slice(&[3, 2, 1, 0, 11, 22]);
    let slop_ptr = slop_ok.as_ptr();
    let typed_slop = PixelBuffer::try_from_parts(PixelBufferParts::new(
        slop_ok.into(),
        2,
        2,
        12,
        PixelDescriptor::RGB8_SRGB,
    ))
    .unwrap()
    .try_typed::<rgb::RGB8>()
    .unwrap();
    let moved = imgref::ImgVec::try_from(typed_slop).unwrap();
    assert_eq!(moved.buf().as_ptr().cast::<u8>(), slop_ptr);
    assert_eq!(moved.buf().len(), 6);
}

#[cfg(feature = "imgref")]
#[test]
fn symmetric_slice_imgref_sample_imgref_and_bgr_f16_types_roundtrip_zero_copy() {
    use zenpixels::{F16Sample, GrayAlphaF16, InteropTarget, RowAlign};

    // 1. Typed & erased PixelSlice / PixelSliceMut <-> ImgRef / ImgRefMut (including into_imgref_mut lifetime 'a).
    let mut backing = vec![0u8; 4 * 3 * 3];
    let img_mut_long = {
        let mut slice_mut = PixelSliceMut::<rgb::RGB8>::new_typed(&mut backing, 4, 3, 4).unwrap();
        slice_mut.as_imgref_mut()[(1usize, 2usize)] = rgb::RGB8::new(10, 20, 30);
        assert_eq!(
            slice_mut.as_imgref()[(1usize, 2usize)],
            rgb::RGB8::new(10, 20, 30)
        );
        // Consume slice_mut and keep ImgRefMut<'a, RGB8> alive past slice_mut's scope!
        slice_mut.into_imgref_mut()
    };
    assert_eq!(img_mut_long[(1usize, 2usize)], rgb::RGB8::new(10, 20, 30));

    let erased_slice = PixelSlice::new(&backing, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    let img_ref = erased_slice.try_as_imgref::<rgb::RGB8>().unwrap();
    assert_eq!(img_ref.buf().as_ptr().cast::<u8>(), backing.as_ptr());
    let back_erased = PixelSlice::from_imgref(img_ref);
    assert_eq!(back_erased.descriptor(), PixelDescriptor::RGB8_SRGB);

    // 2. [T; N] and bare sample ImgRef via try_from_imgref_with / try_from_imgref_mut_with.
    let mut arr_pixels = vec![[1u8, 2, 3]; 8];
    let arr_ptr = arr_pixels.as_ptr().cast::<u8>();
    let arr_img = imgref::Img::new_stride(arr_pixels.as_mut_slice(), 3, 2, 4);
    let mut arr_slice =
        PixelSliceMut::try_from_imgref_mut_with(arr_img, PixelDescriptor::RGB8_SRGB).unwrap();
    arr_slice.row_mut(1)[0..3].copy_from_slice(&[99, 88, 77]);
    assert_eq!(arr_slice.as_strided_bytes().as_ptr(), arr_ptr);
    assert_eq!(arr_pixels[4], [99, 88, 77]);

    // 3. 64-byte SIMD-padded 3-channel RGB8 & RGBF16 via try_new_with_align + try_as_sample_imgref.
    let mut simd_rgb8 =
        PixelBuffer::try_new_with_align(5, 3, PixelDescriptor::RGB8_SRGB, RowAlign::Cache64)
            .unwrap();
    assert_eq!(simd_rgb8.stride(), 64);
    assert!(simd_rgb8.try_as_imgref::<rgb::RGB8>().is_none());
    {
        let mut s_mut = simd_rgb8.try_as_sample_imgref_mut::<u8>().unwrap();
        assert_eq!((s_mut.width(), s_mut.height(), s_mut.stride()), (15, 3, 64));
        s_mut[(14usize, 2usize)] = 0x7F;
    }
    let s_ref = simd_rgb8.try_as_sample_imgref::<u8>().unwrap();
    assert_eq!(s_ref[(14usize, 2usize)], 0x7F);
    let wrapped =
        PixelSlice::try_from_sample_imgref_with(s_ref, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(
        (wrapped.width(), wrapped.rows(), wrapped.stride()),
        (5, 3, 64)
    );
    assert_eq!(wrapped.row(2)[14], 0x7F);

    // PixelCache64 (192 B stride) moves zero-copy into ImgVec<RGB8>!
    let px_cache =
        PixelBuffer::try_new_with_align(5, 3, PixelDescriptor::RGB8_SRGB, RowAlign::PixelCache64)
            .unwrap();
    assert_eq!(px_cache.stride(), 192);
    let px_ptr = px_cache.as_slice().as_strided_bytes().as_ptr();
    let px_img = px_cache.try_into_imgvec_zero_copy::<rgb::RGB8>().unwrap();
    assert_eq!(px_img.buf().as_ptr().cast::<u8>(), px_ptr);
    assert_eq!(px_img.stride(), 64);

    // 4. Bgr<u8|u16|F16Sample|f32>, Bgra<u16|F16Sample|f32>, and GrayAlphaF16 zero-copy round-trips.
    let bgr_f16_desc = <rgb::Bgr<F16Sample> as zenpixels::Pixel>::DESCRIPTOR
        .with_transfer(zenpixels::TransferFunction::Linear);
    let mut bgr_buf = PixelBuffer::try_new_for(3, 2, bgr_f16_desc, InteropTarget::Imgref).unwrap();
    let bgr_ptr = bgr_buf.as_slice().as_strided_bytes().as_ptr();
    {
        let mut v = bgr_buf.try_as_imgref_mut::<rgb::Bgr<F16Sample>>().unwrap();
        v[(0usize, 0usize)] = rgb::Bgr {
            b: F16Sample::from_f32(0.25),
            g: F16Sample::from_f32(0.5),
            r: F16Sample::from_f32(1.0),
        };
    }
    let bgr_img = bgr_buf
        .try_into_imgvec_zero_copy::<rgb::Bgr<F16Sample>>()
        .unwrap();
    assert_eq!(bgr_img.buf().as_ptr().cast::<u8>(), bgr_ptr);
    let bgr_slice: PixelSlice<'_, rgb::Bgr<F16Sample>> = bgr_img.as_ref().into();
    assert_eq!(
        bgr_slice.as_imgref()[(0usize, 0usize)].r,
        F16Sample::from_f32(1.0)
    );
    let bgr_back = PixelBuffer::from_imgvec_erased(bgr_img);
    assert_eq!(bgr_back.as_slice().as_strided_bytes().as_ptr(), bgr_ptr);

    let ga_img = imgref::ImgVec::new(
        vec![GrayAlphaF16::new(F16Sample::ONE, F16Sample::ZERO); 6],
        3,
        2,
    );
    let ga_ptr = ga_img.buf().as_ptr().cast::<u8>();
    let ga_slice: PixelSlice<'_, GrayAlphaF16> = ga_img.as_ref().into();
    assert_eq!(ga_slice.descriptor(), PixelDescriptor::GRAYAF16_LINEAR);
    let ga_buf = PixelBuffer::from_imgvec(ga_img);
    assert_eq!(ga_buf.as_slice().as_strided_bytes().as_ptr(), ga_ptr);
}
