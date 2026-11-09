use zenpixels::{
    ChannelLayout as L, ChannelType as T, PixelDescriptor as D, PixelSliceMut,
    sample::{NativeCodeSlice, NativeEncoding},
};
use zenpixels_convert::codes::{NativeConversionError, native_to_u16};
#[test]
fn per_channel_canonicalization_preserves_padding_and_refuses_signed() {
    let enc = [
        NativeEncoding::new(T::U16, 10, 0).unwrap(),
        NativeEncoding::new(T::U16, 10, 0).unwrap(),
        NativeEncoding::new(T::U16, 10, 0).unwrap(),
        NativeEncoding::new(T::U16, 1, 0).unwrap(),
    ];
    let words = [
        0xffffu16, 0xfc00, 0xffff, 0xffff, 99, 99, 0xfc00, 0xffff, 0xfc00, 0xfffe,
    ];
    let src = NativeCodeSlice::new(bytemuck::cast_slice(&words), 1, 2, 12, L::Rgba, &enc).unwrap();
    let mut out = [42u16; 10];
    {
        let mut dst =
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut out), 1, 2, 12, D::RGBA16).unwrap();
        native_to_u16(src, &mut dst, zenpixels::SignalRange::Full).unwrap();
    }
    assert_eq!(out, [65535, 0, 65535, 65535, 42, 42, 0, 65535, 0, 0]);
    let signed = [NativeEncoding::new(T::I16, 10, 0).unwrap(); 4];
    let src =
        NativeCodeSlice::new(bytemuck::cast_slice(&words), 1, 2, 12, L::Rgba, &signed).unwrap();
    let before = out;
    {
        let mut dst =
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut out), 1, 2, 12, D::RGBA16).unwrap();
        assert_eq!(
            native_to_u16(src, &mut dst, zenpixels::SignalRange::Full),
            Err(NativeConversionError::UnsupportedEncoding)
        );
    }
    assert_eq!(out, before);
}

#[test]
fn encoding_back_obeys_padding_policy_and_recovers_every_ten_bit_code() {
    use zenpixels_convert::codes::{NativePadding, u16_to_native};
    let enc = [NativeEncoding::new(T::U16, 10, 0).unwrap()];
    let words: Vec<u16> = (0..1024).map(|v| v | 0xfc00).collect();
    let src =
        NativeCodeSlice::new(bytemuck::cast_slice(&words), 1024, 1, 2048, L::Gray, &enc).unwrap();
    let mut canonical = vec![0u16; 1024];
    {
        let mut dst = PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut canonical),
            1024,
            1,
            2048,
            D::GRAY16,
        )
        .unwrap();
        native_to_u16(src, &mut dst, zenpixels::SignalRange::Full).unwrap();
    }
    let src =
        zenpixels::PixelSlice::new(bytemuck::cast_slice(&canonical), 1024, 1, 2048, D::GRAY16)
            .unwrap();
    let mut dest = vec![0xa800u16; 1024];
    assert_eq!(
        u16_to_native(
            src.clone(),
            bytemuck::cast_slice_mut(&mut dest),
            2048,
            &enc,
            NativePadding::Zero,
            zenpixels::DepthPolicy::Forbid
        ),
        Err(NativeConversionError::DepthReduction)
    );
    assert!(dest.iter().all(|&v| v == 0xa800));
    u16_to_native(
        src.clone(),
        bytemuck::cast_slice_mut(&mut dest),
        2048,
        &enc,
        NativePadding::Preserve,
        zenpixels::DepthPolicy::Round,
    )
    .unwrap();
    for (i, &v) in dest.iter().enumerate() {
        assert_eq!(v, 0xa800 | i as u16);
    }
    u16_to_native(
        src,
        bytemuck::cast_slice_mut(&mut dest),
        2048,
        &enc,
        NativePadding::Zero,
        zenpixels::DepthPolicy::Round,
    )
    .unwrap();
    for (i, &v) in dest.iter().enumerate() {
        assert_eq!(v, i as u16);
    }
}
