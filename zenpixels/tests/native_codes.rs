use zenpixels::{
    ChannelType as T,
    hdr::St2086Codes,
    plane::{AnyPlane, PlaneRef},
    sample::{CodePlane, SampleEncoding},
};

#[test]
fn payloads_and_storage_are_distinct() {
    let words = [0xffffu16, 0xfe00, 0xfc00, 0xffff, 123, 456];
    let plane = AnyPlane::U16(PlaneRef::new(&words[..], 4, 1, 6).unwrap());
    let codes: Vec<_> = [10, 10, 10, 1]
        .into_iter()
        .enumerate()
        .map(|(index, bits)| {
            let encoding = SampleEncoding::new(T::U16, bits, 0).unwrap();
            let component = CodePlane::component(plane, 4, index, encoding).unwrap();
            assert_eq!(component.code(1, 0), None);
            component.code(0, 0).unwrap()
        })
        .collect();
    assert_eq!(codes, [1023, 512, 0, 1]);
    assert_eq!(words, [0xffff, 0xfe00, 0xfc00, 0xffff, 123, 456]);
}

#[test]
fn signed_and_wide_encodings_remain_descriptions() {
    for (storage, bits) in [(T::I16, 10), (T::U64, 64), (T::I64, 64), (T::U32, 32)] {
        let encoding = SampleEncoding::new(storage, bits, 0).unwrap();
        assert_eq!(encoding.storage(), storage);
        assert_eq!(encoding.code_bits(), bits);
    }
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
    let words = [0xffffu16];
    let plane = AnyPlane::U16(PlaneRef::new(&words[..], 1, 1, 1).unwrap());
    let encoding = SampleEncoding::new(T::U16, 5, 11).unwrap();
    assert!(CodePlane::component(plane, 3, 0, encoding).is_err());
}
