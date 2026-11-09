use zenpixels::{
    ChannelLayout as L, ChannelType as T,
    hdr::St2086Codes,
    sample::{NativeCodeSlice, SampleEncoding},
};
#[test]
fn payloads_and_storage_are_distinct() {
    let e = [
        SampleEncoding::new(T::U16, 10, 0).unwrap(),
        SampleEncoding::new(T::U16, 10, 0).unwrap(),
        SampleEncoding::new(T::U16, 10, 0).unwrap(),
        SampleEncoding::new(T::U16, 1, 0).unwrap(),
    ];
    let words = [0xfc00u16 | 1023, 0xfc00 | 512, 0xfc00, 0xfffe | 1, 123, 456];
    let bytes = bytemuck::cast_slice(&words);
    let view = NativeCodeSlice::new(bytes, 1, 1, 12, L::Rgba, &e).unwrap();
    assert_eq!(view.row(0).unwrap(), &bytes[..8]);
    assert_eq!(
        (0..4)
            .map(|c| view.code(0, 0, c).unwrap())
            .collect::<Vec<_>>(),
        [1023, 512, 0, 1]
    );
    assert_eq!(view.code(1, 0, 0), None);
}
#[test]
fn signed_and_wide_are_exact() {
    for (t, bits, word, want) in [
        (T::I16, 10, 0xfe00u64, -512i128),
        (T::U64, 64, u64::MAX, u64::MAX as i128),
        (T::I64, 64, 1 << 63, i64::MIN as i128),
        (T::U32, 32, u32::MAX as u64, u32::MAX as i128),
    ] {
        let e = [SampleEncoding::new(t, bits, 0).unwrap()];
        let bytes = match t.byte_size() {
            2 => (word as u16).to_ne_bytes().to_vec(),
            4 => (word as u32).to_ne_bytes().to_vec(),
            _ => word.to_ne_bytes().to_vec(),
        };
        let v = NativeCodeSlice::new(&bytes[..t.byte_size()], 1, 1, t.byte_size(), L::Gray, &e)
            .unwrap();
        assert_eq!(v.code(0, 0, 0), Some(want));
    }
    let e = [SampleEncoding::new(T::U64, 64, 0).unwrap()];
    let empty = NativeCodeSlice::new(&[], 0, 2, usize::MAX, L::Gray, &e).unwrap();
    assert_eq!(empty.row(1), Some(&[][..]));
    assert!(NativeCodeSlice::new(&[], 1, 2, usize::MAX, L::Gray, &e).is_err());
}
#[test]
fn metadata_preserves_each_byte_and_float_loss_is_visible() {
    for i in 0..24 {
        for byte in 0..=255 {
            let mut bytes = [0; 24];
            bytes[i] = byte;
            assert_eq!(St2086Codes::from_be_bytes(bytes).to_be_bytes(), bytes);
        }
    }
    let mut bytes = [0; 24];
    bytes[16..20].copy_from_slice(&40_000_001u32.to_be_bytes());
    let codes = St2086Codes::from_be_bytes(bytes);
    assert_ne!(
        (codes.semantic().max_luminance as f64 * 10000.0).round() as u32,
        40_000_001
    );
    assert_eq!(codes.to_be_bytes(), bytes);
}

#[test]
fn rgb565_wire_words_require_unpacking() {
    let enc = [
        SampleEncoding::new(T::U16, 5, 11).unwrap(),
        SampleEncoding::new(T::U16, 6, 5).unwrap(),
        SampleEncoding::new(T::U16, 5, 0).unwrap(),
    ];
    assert!(NativeCodeSlice::new(&[0xff, 0xff], 1, 1, 2, L::Rgb, &enc).is_err());
}
