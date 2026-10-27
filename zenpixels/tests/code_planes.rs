use zenpixels::{
    ChannelType,
    sample::{CodePlane, CodePlaneError, SampleEncoding},
};

#[test]
fn borrowed_codes_keep_stride_placement_and_padding_distinct() {
    let encoding = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
    let words = [63, 127, 65535, 0xdead, 1151, 32767, 64063];
    let plane = CodePlane::from_u16(&words, 3, 2, 8, encoding).unwrap();
    assert_eq!(
        (plane.width(), plane.height(), plane.encoding()),
        (3, 2, encoding)
    );
    assert_eq!(
        (plane.code(0, 0), plane.code(1, 0), plane.code(2, 0)),
        (Some(0), Some(1), Some(1023))
    );
    assert_eq!(
        (plane.code(0, 1), plane.code(1, 1), plane.code(2, 1)),
        (Some(17), Some(511), Some(1000))
    );
    assert_eq!(plane.code(3, 0), None);
    assert_eq!(plane.code(0, 2), None);
    let bytes = [0xf0, 0xff, 0xaa, 0xf1, 0xf2];
    let p = CodePlane::from_u8(
        &bytes,
        2,
        2,
        3,
        SampleEncoding::new(ChannelType::U8, 4, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        (p.code(0, 0), p.code(1, 0), p.code(0, 1), p.code(1, 1)),
        (Some(0), Some(15), Some(1), Some(2))
    );
}

#[test]
fn invalid_storage_and_extent_refuse_without_reading() {
    let enc = SampleEncoding::new(ChannelType::U16, 12, 0).unwrap();
    assert_eq!(
        CodePlane::from_u8(&[], 1, 1, 2, enc).unwrap_err(),
        CodePlaneError::StorageMismatch
    );
    assert_eq!(
        CodePlane::from_u16(&[0; 4], 1, 2, 3, enc).unwrap_err(),
        CodePlaneError::MisalignedStride
    );
    assert_eq!(
        CodePlane::from_u16(&[0; 4], 2, 2, 2, enc).unwrap_err(),
        CodePlaneError::ShortStride
    );
    assert_eq!(
        CodePlane::from_u16(&[0; 3], 2, 2, 4, enc).unwrap_err(),
        CodePlaneError::ShortBuffer
    );
    assert_eq!(
        CodePlane::from_u16(&[], 1, usize::MAX, usize::MAX - 1, enc).unwrap_err(),
        CodePlaneError::GeometryOverflow
    );
    let empty = CodePlane::from_u16(&[], 0, usize::MAX, 0, enc).unwrap();
    assert_eq!(empty.code(0, usize::MAX - 1), None);
}
