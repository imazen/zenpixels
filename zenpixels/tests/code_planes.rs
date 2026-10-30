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

#[test]
fn interleaved_components_borrow_semi_planar_and_packed_layouts() {
    // P012 chroma: Cb/Cr interleaved, 12 high bits, nonzero low padding.
    let p012 = SampleEncoding::new(ChannelType::U16, 12, 4).unwrap();
    let uv: Vec<u16> = (0..12u16).map(|i| ((i * 300) << 4) | 0xf).collect(); // 2 rows x (3 pairs) + 0 pad
    let cb = CodePlane::from_u16_interleaved(&uv, 3, 2, 12, 2, p012).unwrap();
    let cr = CodePlane::from_u16_interleaved(&uv[1..], 3, 2, 12, 2, p012).unwrap();
    assert_eq!(cb.step_words(), 2);
    for y in 0..2 {
        for x in 0..3 {
            let i = y * 6 + x * 2;
            assert_eq!(cb.code(x, y), Some((i as u16 * 300) & 0xfff));
            assert_eq!(cr.code(x, y), Some(((i as u16 + 1) * 300) & 0xfff));
        }
    }
    let mut row = [0u16; 2];
    cr.read_codes(1, 1, &mut row).unwrap();
    assert_eq!(row, [cr.code(1, 1).unwrap(), cr.code(2, 1).unwrap()]);
    // The last row of the second component needs only its own last sample.
    assert_eq!(
        CodePlane::from_u16_interleaved(&uv[1..11], 3, 2, 12, 2, p012).unwrap_err(),
        CodePlaneError::ShortBuffer
    );
    assert_eq!(
        CodePlane::from_u16_interleaved(&uv, 3, 2, 8, 2, p012).unwrap_err(),
        CodePlaneError::ShortStride
    );
    assert_eq!(
        CodePlane::from_u16_interleaved(&uv, 3, 2, 12, 0, p012).unwrap_err(),
        CodePlaneError::ZeroStep
    );
    assert_eq!(
        CodePlane::from_u16_interleaved(&uv, usize::MAX, 1, 12, 2, p012).unwrap_err(),
        CodePlaneError::GeometryOverflow
    );
}

#[test]
fn row_reads_match_single_codes_and_refuse_outside_the_plane() {
    let ten = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
    let words: Vec<u16> = (0..40u16).map(|i| (i * 29) | 0xfc00).collect();
    // RGBA-interleaved green, rows of 4 pixels with 2 words of row padding.
    let green = CodePlane::from_u16_interleaved(&words[1..], 4, 2, 36, 4, ten).unwrap();
    for y in 0..2 {
        let mut row = [0u16; 4];
        green.read_codes(0, y, &mut row).unwrap();
        let single: Vec<u16> = (0..4).map(|x| green.code(x, y).unwrap()).collect();
        assert_eq!(row.to_vec(), single);
        assert!(row.iter().all(|c| *c < 1024));
    }
    let mut untouched = [7u16; 3];
    assert_eq!(
        green.read_codes(2, 0, &mut untouched),
        Err(CodePlaneError::OutsideActivePlane)
    );
    assert_eq!(
        green.read_codes(0, 2, &mut untouched[..1]),
        Err(CodePlaneError::OutsideActivePlane)
    );
    assert_eq!(untouched, [7; 3]);
    assert_eq!(green.read_codes(4, 1, &mut []), Ok(()));
    let bytes = [0xab_u8, 0, 0xcd, 0];
    let u8_plane = CodePlane::from_u8_interleaved(
        &bytes,
        2,
        1,
        4,
        2,
        SampleEncoding::new(ChannelType::U8, 8, 0).unwrap(),
    )
    .unwrap();
    let mut codes = [0u16; 2];
    u8_plane.read_codes(0, 0, &mut codes).unwrap();
    assert_eq!(codes, [0xab, 0xcd]);
}
