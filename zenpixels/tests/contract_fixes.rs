use zenpixels::*;
#[test]
fn empty_crop_positive_rows_are_empty() {
    let b = PixelBuffer::new(4, 4, PixelDescriptor::RGB8);
    let crop = b.crop_view(0, 0, 0, 2);
    assert!(crop.row(1).is_empty());
}
#[test]
fn cicp_to_descriptor_preserves_padding() {
    let d = Cicp::SRGB.try_to_descriptor(PixelFormat::Rgbx8).unwrap();
    assert_eq!(d.alpha, PixelFormat::Rgbx8.default_alpha());
}
#[cfg(feature = "imgref")]
#[test]
fn layout_helper_preserves_color_and_alpha() {
    let mut data = [1u8, 2, 3, 128];
    let s = PixelSliceMut::<rgb::RGBA<u8>>::new_typed(&mut data, 1, 1, 1)
        .unwrap()
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_transfer(TransferFunction::Linear)
        .with_alpha_mode(Some(AlphaMode::Premultiplied));
    let s = s.swap_to_bgra();
    assert_eq!(s.descriptor().primaries, ColorPrimaries::DisplayP3);
    assert_eq!(s.descriptor().transfer(), TransferFunction::Linear);
    assert_eq!(s.descriptor().alpha(), Some(AlphaMode::Premultiplied));
}
#[cfg(feature = "imgref")]
#[test]
fn from_imgvec_preserves_strided_allocation() {
    let px = rgb::RGB8::new(1, 2, 3);
    let img = imgref::Img::new_stride(vec![px; 6], 2, 2, 3);
    let ptr = img.buf().as_ptr().cast::<u8>();
    let b = PixelBuffer::<rgb::RGB8>::from_imgvec(img);
    assert_eq!(b.stride(), 9);
    assert_eq!(b.as_slice().row(1), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(b.as_slice().row(0).as_ptr(), ptr);
}

#[test]
fn containment_agrees_with_primary_vertices() {
    let known = [
        ColorPrimaries::Bt709,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::AdobeRgb,
        ColorPrimaries::Bt2020,
    ];
    for outer in known {
        for inner in known {
            let m = inner.gamut_matrix_to(outer).unwrap();
            let fits = m.iter().flatten().all(|v| *v >= -1e-4 && *v <= 1.0001);
            assert_eq!(outer.contains(inner), fits, "{outer:?} contains {inner:?}");
        }
        assert!(!outer.contains(ColorPrimaries::Unknown));
        assert!(!ColorPrimaries::Unknown.contains(outer));
    }
}
