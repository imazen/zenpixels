use std::sync::Arc;
use zenpixels::{
    BufferError, ByteStorage, ByteStorageMut, Cicp, ColorContext, PixelBuffer, PixelBufferParts,
    PixelDescriptor, PixelSliceMut, PixelStorage,
};

fn byte_capacity(data: &PixelStorage) -> usize {
    match data {
        PixelStorage::U8(v) => v.capacity(),
        PixelStorage::U16(v) => v.capacity() * 2,
        _ => panic!("these tests hold u8 or u16 samples"),
    }
}

#[test]
fn adoption_roundtrips_storage_and_context_without_cloning() {
    let context = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    for descriptor in [PixelDescriptor::RGB8, PixelDescriptor::RGB16] {
        // A prefix that reaches a 64-byte boundary, a stride that is the LCM of 64 and
        // the pixel size, and a further prefix regardless of allocator alignment.
        let stride = descriptor.simd_aligned_stride(3, 64);
        let prefix = descriptor.bytes_per_pixel();
        let data = vec![0u8; stride * 2 + 63 + prefix];
        let boundary = data.as_ptr().align_offset(64);
        let mut parts: PixelBufferParts =
            PixelBufferParts::new(data.into(), 3, 2, stride, descriptor);
        parts.color_context = Some(context.clone());
        parts.offset = boundary + prefix;
        let original = (
            parts.data.as_bytes().as_ptr(),
            parts.data.as_bytes().len(),
            byte_capacity(&parts.data),
            parts.offset,
            parts.stride_bytes,
        );
        let mut buffer = PixelBuffer::try_from_parts(parts).unwrap();
        buffer.as_slice_mut().row_mut(1).fill(7);
        assert!(buffer.as_slice().row(1).iter().all(|&b| b == 7));
        let parts = buffer.into_parts();
        assert_eq!(
            (
                parts.data.as_bytes().as_ptr(),
                parts.data.as_bytes().len(),
                byte_capacity(&parts.data),
                parts.offset,
                parts.stride_bytes
            ),
            original
        );
        assert!(Arc::ptr_eq(parts.color_context.as_ref().unwrap(), &context));
        assert_eq!(Arc::strong_count(&context), 2);
    }
}

#[test]
fn minimal_final_row_extent_supports_views_mutation_and_compaction() {
    let mut parts = PixelBuffer::new(4, 1, PixelDescriptor::RGB8).into_parts();
    parts
        .data
        .as_bytes_mut()
        .copy_from_slice(&[1, 2, 3, 99, 99, 99, 99, 99, 99, 4, 5, 6]);
    parts.width = 1;
    parts.height = 2;
    parts.stride_bytes = 9;
    let pointer = parts.data.as_bytes().as_ptr();
    let mut buffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(buffer.as_slice().row(1), &[4, 5, 6]);
    assert_eq!(buffer.rows(1, 1).row(0), &[4, 5, 6]);
    buffer.as_slice_mut().row_mut(1)[0] = 7;
    buffer.transform_in_place(|p| {
        assert_eq!(p.bytes.len(), 12);
        PixelSliceMut::new(p.bytes, p.width, p.rows, p.stride, p.descriptor).unwrap()
    });
    assert_eq!(buffer.copy_to_contiguous_bytes(), &[1, 2, 3, 7, 5, 6]);
    let parts = buffer.into_contiguous().into_parts();
    assert_eq!(parts.data.as_bytes(), &[1, 2, 3, 7, 5, 6]);
    assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
}

#[test]
fn rejection_preserves_allocation_and_can_be_repaired() {
    let mut parts = PixelBuffer::new(2, 2, PixelDescriptor::RGB8).into_parts();
    let pointer = parts.data.as_bytes().as_ptr();
    let capacity = byte_capacity(&parts.data);
    parts.offset = usize::MAX;
    let mut error = PixelBuffer::try_from_parts(parts).unwrap_err();
    assert_eq!(*error.error().error(), BufferError::InsufficientData);
    let mut parts = error.take_parts().unwrap();
    assert_eq!(
        (parts.data.as_bytes().as_ptr(), byte_capacity(&parts.data)),
        (pointer, capacity)
    );
    assert_eq!(parts.offset, usize::MAX);
    assert!(error.take_parts().is_none());
    assert_eq!(
        *error.without_buffer().error(),
        BufferError::InsufficientData
    );
    parts.offset = 0;
    assert!(PixelBuffer::try_from_parts(parts).is_ok());
}

#[test]
fn rejection_validates_stride_extent_overflow_and_alignment() {
    for (stride, height, expected) in [
        (2, 2, BufferError::StrideTooSmall),
        (4, 2, BufferError::InsufficientData),
        (9, 3, BufferError::InsufficientData),
        (
            usize::MAX - usize::MAX % 3,
            3,
            BufferError::InvalidDimensions,
        ),
    ] {
        let mut parts = PixelBuffer::new(1, 2, PixelDescriptor::RGB8).into_parts();
        parts.stride_bytes = stride;
        parts.height = height;
        assert_eq!(
            *PixelBuffer::try_from_parts(parts)
                .unwrap_err()
                .without_buffer()
                .error(),
            expected
        );
    }
    let mut parts = PixelBuffer::new(2, 2, PixelDescriptor::RGB16).into_parts();
    parts.height = 1;
    parts.offset = (0..2)
        .find(|&offset| (parts.data.as_bytes().as_ptr() as usize + offset) % 2 != 0)
        .unwrap();
    assert_eq!(
        *PixelBuffer::try_from_parts(parts)
            .unwrap_err()
            .without_buffer()
            .error(),
        BufferError::AlignmentViolation
    );
}

#[test]
fn stripping_error_releases_context_and_does_not_dump_pixels() {
    let context = Arc::new(ColorContext::from_cicp(Cicp::SRGB));
    let mut parts = PixelBuffer::new(10, 10, PixelDescriptor::RGB8)
        .with_color_context(context.clone())
        .into_parts();
    parts.offset = usize::MAX;
    let error = PixelBuffer::try_from_parts(parts).unwrap_err();
    assert_eq!(Arc::strong_count(&context), 2);
    assert!(!format!("{error:?}").contains("data:"));
    let stripped = error.without_buffer();
    assert_eq!(Arc::strong_count(&context), 1);
    assert_eq!(*stripped.error(), BufferError::InsufficientData);
}

#[test]
fn empty_packed_buffers_roundtrip() {
    for (width, height) in [(0, 0), (0, 3), (3, 0)] {
        let parts = PixelBuffer::new(width, height, PixelDescriptor::RGB8).into_parts();
        let buffer = PixelBuffer::try_from_parts(parts).unwrap();
        assert_eq!(buffer.as_slice().as_strided_bytes(), &[]);
        assert_eq!((buffer.width(), buffer.height()), (width, height));
    }
}
