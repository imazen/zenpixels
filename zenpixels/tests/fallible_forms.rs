//! Each `try_` form agrees with its panicking twin in bounds and refuses out of bounds.
use zenpixels::{
    AlphaMode, BufferError, ChannelLayout, ChannelType, PixelBuffer, PixelDescriptor, PixelSlice,
    TransferFunction, hdr::DiffuseWhite,
};

const D: PixelDescriptor = PixelDescriptor::RGB8_SRGB;

/// 3 by 2 with one pixel of padding per row.
fn strided() -> Vec<u8> {
    (0u8..24).collect()
}

#[test]
fn slice_forms_agree_in_bounds_and_refuse_out_of_bounds() {
    let data = strided();
    let s = PixelSlice::new(&data, 3, 2, 12, D).unwrap();
    assert_eq!(s.try_row(1), Some(s.row(1)));
    assert_eq!(s.try_row(2), None);
    assert_eq!(s.try_row_with_stride(0), Some(s.row_with_stride(0)));
    assert_eq!(s.try_row_with_stride(2), None);
    assert_eq!(
        s.try_sub_rows(1, 1).unwrap().row(0),
        s.sub_rows(1, 1).row(0)
    );
    assert!(s.try_sub_rows(1, 2).is_none());
    assert!(s.try_sub_rows(u32::MAX, 2).is_none());
    assert_eq!(
        s.try_crop_view(1, 1, 2, 1).unwrap().row(0),
        s.crop_view(1, 1, 2, 1).row(0)
    );
    assert!(s.try_crop_view(2, 0, 2, 1).is_none());
    assert!(s.try_crop_view(0, 0, 1, u32::MAX).is_none());
    assert_eq!(
        s.try_contiguous_bytes().unwrap().as_ref(),
        s.contiguous_bytes().as_ref()
    );
}

#[test]
fn slice_mut_forms_agree_and_refuse() {
    let mut b = PixelBuffer::try_new(3, 2, D).unwrap();
    let mut m = b.as_slice_mut();
    assert_eq!(m.try_row(1).map(<[u8]>::len), Some(9));
    assert!(m.try_row(2).is_none());
    m.try_row_mut(1).unwrap()[0] = 7;
    assert_eq!(m.row(1)[0], 7);
    assert!(m.try_row_mut(2).is_none());
    assert_eq!(m.try_sub_rows_mut(1, 1).unwrap().row(0)[0], 7);
    assert!(m.try_sub_rows_mut(1, 2).is_none());
}

#[test]
fn buffer_forms_agree_and_refuse() {
    let mut b = PixelBuffer::try_new(3, 2, D).unwrap();
    b.as_slice_mut()
        .row_mut(1)
        .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(b.try_rows(1, 1).unwrap().row(0), b.rows(1, 1).row(0));
    assert!(b.try_rows(2, 1).is_none());
    assert!(b.try_rows_mut(1, 2).is_none());
    assert_eq!(b.try_crop_view(1, 1, 1, 1).unwrap().row(0), &[4, 5, 6]);
    assert!(b.try_crop_view(3, 0, 1, 1).is_none());
    assert_eq!(
        b.try_copy_to_contiguous_bytes().unwrap(),
        b.copy_to_contiguous_bytes()
    );
    let copy = b.crop_copy(1, 1, 2, 1).unwrap();
    assert_eq!(copy.as_slice().row(0), &[4, 5, 6, 7, 8, 9]);
    assert_eq!(
        *b.crop_copy(2, 0, 2, 1).unwrap_err().error(),
        BufferError::InvalidDimensions
    );
}

#[test]
fn descriptor_changes_refuse_a_new_layout_or_a_bad_alpha() {
    let data = strided();
    let s = PixelSlice::new(&data, 3, 2, 12, D).unwrap();
    assert_eq!(
        *s.clone()
            .try_with_descriptor(PixelDescriptor::RGBA8_SRGB)
            .unwrap_err()
            .error(),
        BufferError::IncompatibleDescriptor
    );
    let relabeled = s
        .clone()
        .try_with_descriptor(D.with_transfer(TransferFunction::Linear))
        .unwrap();
    assert_eq!(relabeled.descriptor().transfer(), TransferFunction::Linear);
    assert!(
        s.clone()
            .try_with_alpha_mode(Some(AlphaMode::Straight))
            .is_err()
    );
    assert!(s.try_with_alpha_mode(None).is_ok());

    let b = PixelBuffer::try_new(1, 1, D).unwrap();
    assert!(b.try_with_descriptor(PixelDescriptor::RGBA8_SRGB).is_err());
    let b = PixelBuffer::try_new(1, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert!(b.try_with_alpha_mode(None).is_err());
}

#[test]
fn constructors() {
    assert_eq!(
        PixelDescriptor::try_new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb
        ),
        Some(PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb
        ))
    );
    assert!(
        PixelDescriptor::try_new(
            ChannelType::U16,
            ChannelLayout::Bgra,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb
        )
        .is_none()
    );
    assert_eq!(DiffuseWhite::try_new(203.0), Some(DiffuseWhite::new(203.0)));
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(DiffuseWhite::try_new(bad).is_none());
    }
}
