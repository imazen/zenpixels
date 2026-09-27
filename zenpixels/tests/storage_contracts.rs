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
        assert_eq!(error.take_parts().unwrap().data.len(), 4);
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
    parts.data = vec![91; 20];
    parts.offset = 4;
    parts.stride_bytes = 8;
    parts.data[4..8].copy_from_slice(&[1, 2, 3, 4]);
    parts.data[12..16].copy_from_slice(&[5, 6, 7, 8]);
    let ptr = parts.data.as_ptr();
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
}
