use zenpixels::PixelBuffer;
#[test]
fn matching_typed_allocations_survive_erasure_and_extraction() {
    macro_rules! check {
        ($sample:ty, $value:expr) => {{
            let pixels = vec![rgb::RGB::<$sample>::new($value, $value, $value); 2];
            let pointer = pixels.as_ptr();
            let buffer: PixelBuffer = PixelBuffer::from_pixels(pixels, 2, 1).unwrap().into();
            let result = buffer
                .into_contiguous_pixels::<rgb::RGB<$sample>>()
                .unwrap();
            assert_eq!(pointer, result.as_ptr());
            assert_eq!(result.len(), 2);
        }};
    }
    check!(u8, 17);
    check!(u16, 12345);
    check!(zenpixels::F16Sample, zenpixels::F16Sample::from_f32(0.25));
    check!(f32, 0.25);

    // Typed F16 RGBA, BGRA, Gray, and GrayAlphaF16 round-trip zero-copy as well.
    let ga = vec![
        zenpixels::GrayAlphaF16::new(
            zenpixels::F16Sample::from_f32(0.5),
            zenpixels::F16Sample::ONE
        );
        2
    ];
    let ga_mismatch: PixelBuffer = PixelBuffer::from_pixels(ga.clone(), 2, 1).unwrap().into();
    let ga_ptr = ga.as_ptr();
    let ga_buf: PixelBuffer = PixelBuffer::from_pixels(ga, 2, 1).unwrap().into();
    assert_eq!(
        ga_buf.descriptor(),
        zenpixels::PixelFormat::GrayAF16.descriptor()
    );
    // Extracting as GrayAlpha16 must fail cleanly (no silent F16 -> U16 confusion).
    assert!(
        ga_mismatch
            .into_contiguous_pixels::<zenpixels::GrayAlpha16>()
            .is_none()
    );
    assert!(
        ga_buf
            .as_slice()
            .try_typed::<zenpixels::GrayAlpha16>()
            .is_none()
    );
    assert!(
        ga_buf
            .as_slice()
            .try_typed::<zenpixels::GrayAlphaF16>()
            .is_some()
    );
    let ga_out = ga_buf
        .into_contiguous_pixels::<zenpixels::GrayAlphaF16>()
        .unwrap();
    assert_eq!(ga_ptr, ga_out.as_ptr());
}

#[test]
fn byte_allocation_copies_when_typed_allocator_layout_differs() {
    let bytes = vec![0x11u8; 12];
    let buffer = PixelBuffer::from_vec(bytes, 2, 1, zenpixels::PixelDescriptor::RGB16).unwrap();
    let pointer = buffer.as_slice().as_strided_bytes().as_ptr();
    let result = buffer.into_contiguous_pixels::<rgb::RGB<u16>>().unwrap();
    assert_ne!(pointer, result.as_ptr().cast::<u8>());
    assert_eq!(result, vec![rgb::RGB::new(0x1111, 0x1111, 0x1111); 2]);
}
