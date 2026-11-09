use zenpixels::{
    ChannelType as T, PixelDescriptor as D, PixelSliceMut,
    plane::{AnyPlane, PlaneRef},
    sample::{CodePlane, SampleEncoding},
};
use zenpixels_convert::codes::to_canonical;

#[test]
fn per_channel_canonicalization_preserves_padding() {
    let words = [
        0xffffu16, 0xfc00, 0xffff, 0xffff, 99, 99, 0xfc00, 0xffff, 0xfc00, 0xfffe,
    ];
    let plane = AnyPlane::U16(PlaneRef::new(&words[..], 4, 2, 6).unwrap());
    let components: Vec<_> = [10, 10, 10, 1]
        .into_iter()
        .enumerate()
        .map(|(i, bits)| {
            CodePlane::component(plane, 4, i, SampleEncoding::new(T::U16, bits, 0).unwrap())
                .unwrap()
        })
        .collect();
    let mut out = [42u16; 10];
    let dst = PixelSliceMut::new(bytemuck::cast_slice_mut(&mut out), 1, 2, 12, D::RGBA16).unwrap();
    to_canonical(&components, zenpixels::SignalRange::Full, 0, 0, dst).unwrap();
    assert_eq!(out, [65535, 0, 65535, 65535, 42, 42, 0, 65535, 0, 0]);
}

#[test]
fn wide_and_signed_carriage_does_not_grant_code_arithmetic() {
    use zenpixels_convert::codes::{CodeMap, CodeMapError, CodeRole};
    let canonical = SampleEncoding::new(T::U16, 16, 0).unwrap();
    for storage in [T::I8, T::I16, T::U32, T::I32, T::U64, T::I64] {
        let encoding = SampleEncoding::new(storage, 8, 0).unwrap();
        assert!(matches!(
            CodeMap::new(
                encoding,
                canonical,
                CodeRole::Intensity,
                zenpixels::SignalRange::Full
            ),
            Err(CodeMapError::UnsupportedStorage)
        ));
        assert!(matches!(
            CodeMap::new(
                canonical,
                encoding,
                CodeRole::Intensity,
                zenpixels::SignalRange::Full
            ),
            Err(CodeMapError::UnsupportedStorage)
        ));
    }
}
