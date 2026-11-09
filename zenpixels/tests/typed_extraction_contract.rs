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
    check!(f32, 0.25);
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
