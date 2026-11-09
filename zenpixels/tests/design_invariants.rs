use zenpixels::*;

#[test]
fn constructors_share_alpha_validation() {
    for layout in [
        ChannelLayout::Rgb,
        ChannelLayout::Rgba,
        ChannelLayout::Gray,
        ChannelLayout::GrayAlpha,
    ] {
        for alpha in [None, Some(AlphaMode::Straight), Some(AlphaMode::Opaque)] {
            let short =
                PixelDescriptor::try_new(ChannelType::U8, layout, alpha, TransferFunction::Srgb);
            let full = PixelDescriptor::try_new_full(
                ChannelType::U8,
                layout,
                alpha,
                TransferFunction::Srgb,
                ColorPrimaries::Bt709,
            );
            assert_eq!(short, full);
            assert_eq!(full.is_some(), layout.has_alpha() == alpha.is_some());
            if let Some(d) = full {
                d.validate().unwrap();
            }
        }
    }
}

#[test]
fn final_row_split_does_not_address_trailing_padding() {
    let mut bytes = [4u8, 9];
    let mut plane = plane::Plane::new(&mut bytes[..], 2, 1, usize::MAX).unwrap();
    let crop = plane.sub_mut(1, 0, 1, 1).unwrap();
    let (top, bottom) = crop.split_at_row(1).unwrap();
    assert_eq!(top.row(0), Some(&[9][..]));
    assert_eq!(bottom.height(), 0);
}

#[test]
fn empty_channel_reorders_do_not_address_rows() {
    let mut bytes = [];
    let px = PixelSliceMut::new(&mut bytes, 0, 2, 4, PixelDescriptor::RGBX8_SRGB).unwrap();
    let out = px.try_typed::<Rgbx>().unwrap().swap_to_bgrx();
    assert!(out.row(1).is_empty());
}
