use zenpixels::{
    ChannelType,
    plane::{AnyPlane, PlaneRef},
    sample::{ChromaSiting, ChromaSubsampling, CodePlane, CodePlaneError, SampleEncoding},
};

fn u16_plane(words: &[u16], width: usize, height: usize, stride: usize) -> AnyPlane<'_> {
    AnyPlane::U16(PlaneRef::new(words, width, height, stride).unwrap())
}

#[test]
fn borrowed_codes_keep_stride_placement_and_padding_distinct() {
    let encoding = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
    let words = [63, 127, 65535, 0xdead, 1151, 32767, 64063];
    let plane = CodePlane::new(u16_plane(&words, 3, 2, 4), encoding).unwrap();
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
    let p = CodePlane::new(
        AnyPlane::U8(PlaneRef::new(&bytes[..], 2, 2, 3).unwrap()),
        SampleEncoding::new(ChannelType::U8, 4, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        (p.code(0, 0), p.code(1, 0), p.code(0, 1), p.code(1, 1)),
        (Some(0), Some(15), Some(1), Some(2))
    );
}

#[test]
fn storage_and_component_mismatches_refuse_without_reading() {
    let enc = SampleEncoding::new(ChannelType::U16, 12, 0).unwrap();
    let bytes: [u8; 0] = [];
    assert_eq!(
        CodePlane::new(
            AnyPlane::U8(PlaneRef::new(&bytes[..], 1, 0, 1).unwrap()),
            enc
        )
        .unwrap_err(),
        CodePlaneError::StorageMismatch
    );
    let floats = [0.0f32; 4];
    assert_eq!(
        CodePlane::new(
            AnyPlane::F32(PlaneRef::new(&floats[..], 2, 2, 2).unwrap()),
            enc
        )
        .unwrap_err(),
        CodePlaneError::StorageMismatch
    );
    let words = [0u16; 6];
    let plane = u16_plane(&words, 6, 1, 6);
    assert_eq!(
        CodePlane::component(plane, 2, 2, enc).unwrap_err(),
        CodePlaneError::Components
    );
    assert_eq!(
        CodePlane::component(plane, 0, 0, enc).unwrap_err(),
        CodePlaneError::Components
    );
    assert_eq!(
        CodePlane::component(plane, 4, 0, enc).unwrap_err(),
        CodePlaneError::Components
    );
    let empty = CodePlane::new(u16_plane(&[], 0, usize::MAX, 0), enc).unwrap();
    assert_eq!(empty.code(0, usize::MAX - 1), None);
    let empty_cr = CodePlane::component(u16_plane(&[], 0, 3, 0), 2, 1, enc).unwrap();
    assert_eq!((empty_cr.width(), empty_cr.code(0, 0)), (0, None));
}

#[test]
fn components_of_semi_planar_and_interleaved_planes() {
    // P012 chroma: Cb/Cr interleaved, 12 high bits, nonzero low padding.
    let p012 = SampleEncoding::new(ChannelType::U16, 12, 4).unwrap();
    let uv: Vec<u16> = (0..12u16).map(|i| ((i * 300) << 4) | 0xf).collect(); // 2 rows x 3 pairs
    let plane = u16_plane(&uv, 6, 2, 6);
    let cb = CodePlane::component(plane, 2, 0, p012).unwrap();
    let cr = CodePlane::component(plane, 2, 1, p012).unwrap();
    assert_eq!((cb.width(), cb.height()), (3, 2));
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
}

#[test]
fn row_reads_match_single_codes_and_refuse_outside_the_plane() {
    let ten = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
    let words: Vec<u16> = (0..40u16).map(|i| (i * 29) | 0xfc00).collect();
    // Green of interleaved RGBA, rows of 4 pixels with 2 words of row padding.
    let green = CodePlane::component(u16_plane(&words, 16, 2, 18), 4, 1, ten).unwrap();
    for y in 0..2 {
        let mut row = [0u16; 4];
        green.read_codes(0, y, &mut row).unwrap();
        let single: Vec<u16> = (0..4).map(|x| green.code(x, y).unwrap()).collect();
        assert_eq!(row.to_vec(), single);
        assert_eq!(single[2], (words[y * 18 + 2 * 4 + 1] & 0x3ff));
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
    let u8_plane = CodePlane::component(
        AnyPlane::U8(PlaneRef::new(&bytes[..], 4, 1, 4).unwrap()),
        2,
        0,
        SampleEncoding::new(ChannelType::U8, 8, 0).unwrap(),
    )
    .unwrap();
    let mut codes = [0u16; 2];
    u8_plane.read_codes(0, 0, &mut codes).unwrap();
    assert_eq!(codes, [0xab, 0xcd]);
}

#[test]
fn chroma_size_rounds_up_and_mono_has_no_chroma() {
    use ChromaSubsampling as S;
    assert_eq!(S::Mono.chroma_size(5, 3), None);
    assert_eq!(S::Yuv444.chroma_size(5, 3), Some((5, 3)));
    assert_eq!(S::Yuv422.chroma_size(5, 3), Some((3, 3)));
    assert_eq!(S::Yuv420.chroma_size(5, 3), Some((3, 2)));
    assert_eq!(
        S::Yuv420.chroma_size(u32::MAX, 1),
        Some((u32::MAX / 2 + 1, 1))
    );
    assert_eq!(S::Yuv420.chroma_size(0, 0), Some((0, 0)));
    assert_eq!(ChromaSiting::default(), ChromaSiting::Unknown);
}
#[test]
fn bulk_codes_preserve_every_encoding_and_component_step() {
    use zenpixels::ChannelType;
    use zenpixels::plane::{AnyPlane, PlaneRef};
    use zenpixels::sample::{CodePlane, SampleEncoding};
    for ty in [ChannelType::U8, ChannelType::U16] {
        let storage_bits = if ty == ChannelType::U8 { 8 } else { 16 };
        for bits in 1..=storage_bits {
            for shift in 0..=storage_bits - bits {
                let encoding = SampleEncoding::new(ty, bits, shift).unwrap();
                let mask = ((1u32 << bits) - 1) as u16;
                for components in [1, 2] {
                    for width in [0, 1, 7, 16, 17, 127, 128, 129] {
                        let stride = width * components + 7;
                        let wide: Vec<u16> = (0..stride * 3)
                            .map(|i| (i as u16).wrapping_mul(1297) ^ 0xa57c)
                            .collect();
                        let narrow: Vec<u8> = wide.iter().map(|v| *v as u8).collect();
                        let plane = match ty {
                            ChannelType::U8 => AnyPlane::U8(
                                PlaneRef::new(&narrow[..], width * components, 3, stride).unwrap(),
                            ),
                            ChannelType::U16 => AnyPlane::U16(
                                PlaneRef::new(&wide[..], width * components, 3, stride).unwrap(),
                            ),
                            _ => unreachable!(),
                        };
                        for component in 0..components {
                            let codes =
                                CodePlane::component(plane, components, component, encoding)
                                    .unwrap();
                            for y in 0..3 {
                                for start in [0, width / 2, width] {
                                    let mut actual = vec![0xffff; width - start];
                                    codes.read_codes(start, y, &mut actual).unwrap();
                                    let expected: Vec<_> = (start..width)
                                        .map(|x| {
                                            let i = y * stride + x * components + component;
                                            let word = if ty == ChannelType::U8 {
                                                u16::from(narrow[i])
                                            } else {
                                                wide[i]
                                            };
                                            (word >> shift) & mask
                                        })
                                        .collect();
                                    assert_eq!(
                                        actual, expected,
                                        "{ty:?} bits={bits} shift={shift} components={components} width={width} start={start}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
