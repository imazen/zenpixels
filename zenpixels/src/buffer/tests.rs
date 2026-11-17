use super::*;
use crate::descriptor::PixelFormat;

/// A buffer adopted with whatever prefix makes its rows `align`-aligned, on a
/// stride that is the LCM of `align` and the pixel size. No public constructor
/// allocates this today; `offset` and `try_from_parts` keep it available.
fn simd_padded(width: u32, height: u32, descriptor: PixelDescriptor, align: usize) -> PixelBuffer {
    let stride = descriptor.simd_aligned_stride(width, align);
    let total = stride * height as usize;
    let data = alloc::vec![0u8; total + align - 1];
    let mut parts: PixelBufferParts =
        PixelBufferParts::new(data.into(), width, height, stride, descriptor);
    parts.offset = align_offset(parts.data.as_bytes().as_ptr(), align);
    PixelBuffer::try_from_parts(parts).unwrap()
}

use crate::descriptor::{ChannelLayout, ChannelType};
use alloc::format;

/// The buffer-level relabel builders set their field and **preserve** the
/// rest of the `ColorContext` (clone-override-reattach), and chain.
#[test]
fn allocator_zeroed_storage_boundaries() {
    for size in [0, 1, 4095, 4096, 4097, 65536] {
        let storage = PixelStorage::try_zeroed(ChannelType::U8, size).unwrap();
        assert_eq!(ByteStorage::as_bytes(&storage).len(), size);
        assert!(
            ByteStorage::as_bytes(&storage)
                .iter()
                .all(|&byte| byte == 0)
        );
    }
    assert!(matches!(
        PixelStorage::try_zeroed(ChannelType::U8, usize::MAX),
        Err(BufferError::AllocationFailed)
    ));
}

#[test]
fn relabel_builders_set_and_preserve_color_context() {
    let data = alloc::vec![0u8; 4 * 4 * 3];
    let slice = PixelSlice::new(&data, 4, 4, 12, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .with_cicp(Cicp::BT2100_PQ)
        .with_diffuse_white(DiffuseWhite::new(120.0));
    let ctx = slice.color_context().expect("context attached");
    assert_eq!(ctx.diffuse_white(), Some(DiffuseWhite::new(120.0)));
    assert_eq!(
        ctx.cicp(),
        Some(Cicp::BT2100_PQ),
        "cicp preserved across diffuse_white relabel"
    );
    assert_eq!(
        slice.encoding(),
        crate::PixelEncoding::new(PixelDescriptor::RGB8_SRGB, Some(ctx.as_ref()))
    );

    // Relabeling with an ICC profile replaces the single-authority profile
    // (clearing CICP) while preserving diffuse_white.
    let slice = slice.with_icc(alloc::vec![1u8, 2, 3]);
    let ctx = slice.color_context().unwrap();
    assert_eq!(ctx.icc(), Some(&[1u8, 2, 3][..]));
    assert_eq!(ctx.cicp(), None);
    assert_eq!(ctx.diffuse_white(), Some(DiffuseWhite::new(120.0)));
}

#[test]
fn try_row_with_stride_handles_cropped_and_minimal_extent_final_rows() {
    // 2x2 RGBA8 with stride 16, minimal-extent backing (row 0 = 16 B, row 1 = 8 B -> 24 B total).
    let data = alloc::vec![7u8; 24];
    let slice = PixelSlice::new(&data, 2, 2, 16, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert_eq!(slice.try_row_with_stride(0).map(|r| r.len()), Some(16));
    assert_eq!(slice.try_row_with_stride(1), None);
    assert_eq!(slice.try_row_with_stride(2), None);

    // Full-extent padded buffer works on all rows, then sub_rows / crop_view trims the final row.
    let buf = simd_padded(2, 3, PixelDescriptor::RGBA8_SRGB, 64);
    let full = buf.as_slice();
    assert_eq!(
        full.try_row_with_stride(2).map(|r| r.len()),
        Some(full.stride())
    );

    let sub = full.sub_rows(0, 2);
    assert_eq!(
        sub.try_row_with_stride(0).map(|r| r.len()),
        Some(full.stride())
    );
    assert_eq!(sub.try_row_with_stride(1), None);

    let cropped = full.crop_view(0, 0, 1, 2);
    assert_eq!(
        cropped.try_row_with_stride(0).map(|r| r.len()),
        Some(full.stride())
    );
    assert_eq!(cropped.try_row_with_stride(1), None);
}

// --- PixelBuffer allocation and row access ---

#[test]
fn into_contiguous_preserves_pixels_across_offsets_padding_and_empty_extents() {
    for descriptor in [
        PixelDescriptor::GRAY8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA16_SRGB,
    ] {
        let bpp = descriptor.bytes_per_pixel();
        for width in [0, 1, 2, 5] {
            for height in [0, 1, 2, 5] {
                for padding_pixels in [0, 1, 3] {
                    for prefix_samples in [0, 3] {
                        for final_padding in [false, true] {
                            let row_bytes = width as usize * bpp;
                            let stride = row_bytes + padding_pixels * bpp;
                            let visible_len = if width == 0 || height == 0 {
                                0
                            } else {
                                (height as usize - 1) * stride + row_bytes
                            };
                            // Reserve enough first: computing the offset must
                            // precede initialization but follow allocation.
                            let mut data = Vec::with_capacity(visible_len + stride + 128);
                            let align = descriptor.min_alignment();
                            let offset =
                                align_offset(data.as_ptr(), align) + prefix_samples * align;
                            let len =
                                offset + visible_len + if final_padding { stride + 7 } else { 0 };
                            data.resize(len, 0xEE);
                            let mut expected = Vec::new();
                            for y in 0..height as usize {
                                for x in 0..row_bytes {
                                    let value = (y * 13 + x) as u8;
                                    data[offset + y * stride + x] = value;
                                    expected.push(value);
                                }
                            }
                            let pointer = data.as_ptr();
                            let capacity = data.capacity();
                            let buffer = PixelBuffer::<()> {
                                data: data.into(),
                                offset: offset as u32,
                                width,
                                height,
                                stride_samples: checked_stride_samples(stride, &descriptor)
                                    .unwrap(),
                                descriptor,
                                color: None,
                                exact: None,
                                _pixel: PhantomData,
                            };
                            let packed = buffer.into_contiguous();
                            assert_eq!(packed.as_contiguous_bytes(), Some(expected.as_slice()));
                            assert_eq!(packed.stride(), row_bytes);
                            assert_eq!((packed.width(), packed.height()), (width, height));
                            assert_eq!(packed.descriptor(), descriptor);
                            let parts = packed.into_parts();
                            assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
                            assert_eq!(parts.data.capacity_bytes(), capacity);
                            assert_eq!(parts.offset, offset);
                            assert_eq!(parts.data.as_bytes().len(), offset + expected.len());
                            assert!(parts.data.as_bytes()[..offset].iter().all(|&v| v == 0xEE));
                            assert_eq!(parts.data.as_bytes()[offset..], expected);
                            assert!(parts.color_context.is_none());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn into_parts_moves_the_complete_uncompacted_allocation_and_context() {
    let context = Arc::new(
        ColorContext::from_icc([1u8, 2, 3, 4]).with_diffuse_white(DiffuseWhite::new(120.0)),
    );
    let mut buffer =
        simd_padded(3, 2, PixelDescriptor::RGB8_SRGB, 64).with_color_context(context.clone());
    buffer.data.as_bytes_mut().fill(0xAB);
    let (pointer, len, capacity, offset, stride) = (
        buffer.data.as_bytes().as_ptr(),
        buffer.data.as_bytes().len(),
        buffer.data.capacity_bytes(),
        buffer.offset as usize,
        buffer.stride(),
    );
    let parts = buffer.into_parts();
    assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
    assert_eq!(
        (parts.data.as_bytes().len(), parts.data.capacity_bytes()),
        (len, capacity)
    );
    assert_eq!((parts.offset, parts.stride_bytes), (offset, stride));
    assert_eq!((parts.width, parts.height), (3, 2));
    assert_eq!(parts.descriptor, PixelDescriptor::RGB8_SRGB);
    assert!(parts.data.as_bytes().iter().all(|&v| v == 0xAB));
    assert!(Arc::ptr_eq(parts.color_context.as_ref().unwrap(), &context));
    assert_eq!(Arc::strong_count(&context), 2);
}

#[test]
fn into_contiguous_preserves_type_and_moves_color_context() {
    let context = Arc::new(ColorContext::from_cicp(Cicp::SRGB));
    let mut buffer = simd_padded(2, 3, Rgbx::DESCRIPTOR, 16)
        .try_typed::<Rgbx>()
        .unwrap()
        .with_color_context(context.clone());
    for y in 0..3 {
        buffer.as_slice_mut().row_mut(y).fill(y as u8 + 1);
    }
    let pointer = buffer.data.as_bytes().as_ptr();
    let packed: PixelBuffer<Rgbx> = buffer.into_contiguous();
    assert_eq!(packed.data.as_bytes().as_ptr(), pointer);
    assert_eq!(packed.descriptor(), Rgbx::DESCRIPTOR);
    for y in 0..3 {
        assert_eq!(packed.as_slice().row(y), &[y as u8 + 1; 8]);
    }
    assert!(Arc::ptr_eq(packed.color_context().unwrap(), &context));
    assert_eq!(Arc::strong_count(&context), 2);
    let parts = packed.into_parts();
    assert!(Arc::ptr_eq(parts.color_context.as_ref().unwrap(), &context));
    assert_eq!(Arc::strong_count(&context), 2);
}

#[test]
fn into_contiguous_is_idempotent_and_trims_trailing_storage() {
    let mut bytes = vec![1u8, 2, 3, 0xEE, 0xEE];
    bytes.reserve(128);
    let buffer = PixelBuffer::from_vec(bytes, 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let (pointer, capacity) = (
        buffer.data.as_bytes().as_ptr(),
        buffer.data.capacity_bytes(),
    );
    let packed = buffer.into_contiguous().into_contiguous();
    assert_eq!(packed.as_contiguous_bytes(), Some([1, 2, 3].as_slice()));
    let parts = packed.into_parts();
    assert_eq!(parts.data.as_bytes(), [1, 2, 3]);
    assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
    assert_eq!(parts.data.capacity_bytes(), capacity);
}

#[test]
fn pixel_buffer_new_rgb8() {
    let buf = PixelBuffer::new(10, 5, PixelDescriptor::RGB8_SRGB);
    assert_eq!(buf.width(), 10);
    assert_eq!(buf.height(), 5);
    assert_eq!(buf.stride(), 30);
    assert_eq!(buf.descriptor(), PixelDescriptor::RGB8_SRGB);
    // All zeros
    let slice = buf.as_slice();
    assert_eq!(slice.row(0), &[0u8; 30]);
    assert_eq!(slice.row(4), &[0u8; 30]);
}

#[test]
fn in_place_pixels_try_new_rejects_invalid_geometry() {
    let mut bytes = [0u8; 5];
    let error = InPlacePixels::try_new(&mut bytes, 2, 1, 5, PixelDescriptor::RGB8_SRGB, None)
        .err()
        .unwrap();
    assert_eq!(*error.error(), BufferError::StrideTooSmall);
}

#[test]
fn pixel_cow_clones_only_when_mutated() {
    let bytes = [1u8, 2, 3];
    let slice = PixelSlice::new(&bytes, 1, 1, 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut pixels = PixelCow::Borrowed(slice);

    assert!(pixels.is_borrowed());
    pixels.to_mut().as_slice_mut().row_mut(0)[0] = 9;

    assert!(pixels.is_owned());
    assert_eq!(pixels.as_slice().row(0), &[9, 2, 3]);
    assert_eq!(bytes, [1, 2, 3]);
}

#[test]
fn pixel_buffer_from_vec() {
    let data = vec![0u8; 30 * 5];
    let buf = PixelBuffer::from_vec(data, 10, 5, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(buf.width(), 10);
    assert_eq!(buf.height(), 5);
}

#[test]
fn pixel_buffer_from_vec_too_small() {
    let data = vec![0u8; 10];
    let err = PixelBuffer::from_vec(data, 10, 5, PixelDescriptor::RGB8_SRGB);
    assert_eq!(*err.unwrap_err().error(), BufferError::InsufficientData);
}

#[test]
fn pixel_buffer_into_vec_roundtrip() {
    let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);
    let v = buf.into_parts().data.try_into_vec::<u8>().unwrap();
    // Can re-wrap it
    let buf2 = PixelBuffer::from_vec(v, 4, 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert_eq!(buf2.width(), 4);
}

#[test]
fn pixel_buffer_write_and_read() {
    let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    {
        let mut slice = buf.as_slice_mut();
        let row = slice.row_mut(0);
        row[0] = 255;
        row[1] = 128;
        row[2] = 64;
    }
    let slice = buf.as_slice();
    assert_eq!(&slice.row(0)[..3], &[255, 128, 64]);
    assert_eq!(&slice.row(1)[..3], &[0, 0, 0]);
}

#[test]
fn pixel_buffer_simd_aligned() {
    let buf = simd_padded(10, 5, PixelDescriptor::RGBA8_SRGB, 64);
    assert_eq!(buf.width(), 10);
    assert_eq!(buf.height(), 5);
    // RGBA8 bpp=4, lcm(4,64)=64, raw=40 -> stride=64
    assert_eq!(buf.stride(), 64);
    // First row should be 64-byte aligned
    let slice = buf.as_slice();
    assert_eq!(slice.data.as_bytes().as_ptr() as usize % 64, 0);
}

// --- PixelSlice crop_view ---

#[test]
fn pixel_slice_crop_view() {
    // 4x4 RGB8 buffer, fill each row with row index
    let mut buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
    {
        let mut slice = buf.as_slice_mut();
        for y in 0..4u32 {
            let row = slice.row_mut(y);
            for byte in row.iter_mut() {
                *byte = y as u8;
            }
        }
    }
    // Crop 2x2 starting at (1, 1)
    let crop = buf.crop_view(1, 1, 2, 2);
    assert_eq!(crop.width(), 2);
    assert_eq!(crop.rows(), 2);
    // Row 0 of crop = row 1 of original, should be all 1s
    assert_eq!(crop.row(0), &[1, 1, 1, 1, 1, 1]);
    // Row 1 of crop = row 2 of original, should be all 2s
    assert_eq!(crop.row(1), &[2, 2, 2, 2, 2, 2]);
}

#[test]
fn pixel_slice_crop_copy() {
    let mut buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
    {
        let mut slice = buf.as_slice_mut();
        for y in 0..4u32 {
            let row = slice.row_mut(y);
            for (i, byte) in row.iter_mut().enumerate() {
                *byte = (y * 100 + i as u32) as u8;
            }
        }
    }
    let cropped = buf.crop_copy(1, 1, 2, 2).unwrap();
    assert_eq!(cropped.width(), 2);
    assert_eq!(cropped.height(), 2);
    // Row 0: original row 1, pixels 1-2 -> bytes [103,104,105, 106,107,108]
    assert_eq!(cropped.as_slice().row(0), &[103, 104, 105, 106, 107, 108]);
}

#[test]
fn pixel_slice_sub_rows() {
    let mut buf = PixelBuffer::new(2, 4, PixelDescriptor::GRAY8_SRGB);
    {
        let mut slice = buf.as_slice_mut();
        for y in 0..4u32 {
            let row = slice.row_mut(y);
            row[0] = y as u8 * 10;
            row[1] = y as u8 * 10 + 1;
        }
    }
    let sub = buf.rows(1, 2);
    assert_eq!(sub.rows(), 2);
    assert_eq!(sub.row(0), &[10, 11]);
    assert_eq!(sub.row(1), &[20, 21]);
}

// --- PixelSlice validation ---

#[test]
fn pixel_slice_stride_too_small() {
    let data = [0u8; 100];
    let err = PixelSlice::new(&data, 10, 1, 2, PixelDescriptor::RGB8_SRGB);
    assert_eq!(*err.unwrap_err().error(), BufferError::StrideTooSmall);
}

#[test]
fn pixel_slice_insufficient_data() {
    let data = [0u8; 10];
    let err = PixelSlice::new(&data, 10, 1, 30, PixelDescriptor::RGB8_SRGB);
    assert_eq!(*err.unwrap_err().error(), BufferError::InsufficientData);
}

#[test]
fn pixel_slice_zero_rows() {
    let data = [0u8; 0];
    let slice = PixelSlice::new(&data, 10, 0, 30, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(slice.rows(), 0);
}

#[test]
fn stride_need_not_be_whole_pixels() {
    // RGB8 bpp=3, stride=32 is not a multiple of 3, but every sample still fits.
    let data = [0u8; 128];
    let slice = PixelSlice::new(&data, 10, 2, 32, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(slice.stride(), 32);
    assert_eq!(slice.row(1).len(), 30);
    assert_eq!(slice.row(1).as_ptr(), data[32..].as_ptr());
}

#[test]
fn stride_must_be_whole_samples() {
    let words = [0u32; 32];
    let data: &[u8] = bytemuck::cast_slice(&words);
    for (descriptor, width, stride) in [
        (PixelDescriptor::RGB16_SRGB, 2, 13),
        (PixelDescriptor::RGBF32_LINEAR, 1, 14),
        (PixelDescriptor::RGBF32_LINEAR, 1, 15),
    ] {
        for rows in [1, 2] {
            assert_eq!(
                *PixelSlice::new(data, width, rows, stride, descriptor)
                    .unwrap_err()
                    .error(),
                BufferError::AlignmentViolation,
                "{descriptor:?} stride {stride} rows {rows}"
            );
        }
    }
    assert!(PixelSlice::new(data, 2, 2, 14, PixelDescriptor::RGB16_SRGB).is_ok());
    assert!(PixelSlice::new(data, 1, 2, 16, PixelDescriptor::RGBF32_LINEAR).is_ok());
}

#[test]
fn stride_pixel_aligned_accepted() {
    // RGBA8 bpp=4, stride=48 is a multiple of 4
    let data = [0u8; 256];
    let ok = PixelSlice::new(&data, 10, 2, 48, PixelDescriptor::RGBA8_SRGB);
    assert!(ok.is_ok());
    let s = ok.unwrap();
    assert_eq!(s.stride(), 48);
}

// --- Debug formatting ---

#[test]
fn debug_formats() {
    let buf = PixelBuffer::new(10, 5, PixelDescriptor::RGB8_SRGB);
    assert_eq!(format!("{buf:?}"), "PixelBuffer(10x5, Rgb U8)");

    let slice = buf.as_slice();
    assert_eq!(format!("{slice:?}"), "PixelSlice(10x5, Rgb U8)");

    let mut buf = PixelBuffer::new(3, 3, PixelDescriptor::RGBA16_SRGB);
    let slice_mut = buf.as_slice_mut();
    assert_eq!(format!("{slice_mut:?}"), "PixelSliceMut(3x3, Rgba U16)");
}

// --- BufferError Display ---

#[test]
fn buffer_error_display() {
    let msg = format!("{}", BufferError::StrideTooSmall);
    assert!(msg.contains("stride"));
}

// --- Edge cases ---

#[test]
fn bgrx8_srgb_properties() {
    let d = PixelDescriptor::BGRX8_SRGB;
    assert_eq!(d.channel_type(), ChannelType::U8);
    assert_eq!(d.layout(), ChannelLayout::Bgra);
    assert_eq!(d.alpha(), Some(AlphaMode::Undefined));
    assert_eq!(d.transfer(), TransferFunction::Srgb);
    assert_eq!(d.bytes_per_pixel(), 4);
    assert_eq!(d.min_alignment(), 1);
    // Layout-compatible with BGRA8
    assert!(d.layout_compatible(PixelDescriptor::BGRA8_SRGB));
    // BGRX has no meaningful alpha -- the fourth byte is padding
    assert!(!d.has_alpha());
    // BGRA does have meaningful alpha
    assert!(PixelDescriptor::BGRA8_SRGB.has_alpha());
    // The layout itself reports an alpha-position channel
    assert!(d.layout().has_alpha());
}

#[test]
fn zero_size_buffer() {
    let buf = PixelBuffer::new(0, 0, PixelDescriptor::RGB8_SRGB);
    assert_eq!(buf.width(), 0);
    assert_eq!(buf.height(), 0);
    let slice = buf.as_slice();
    assert_eq!(slice.rows(), 0);
}

#[test]
fn crop_empty() {
    let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
    let crop = buf.crop_view(0, 0, 0, 0);
    assert_eq!(crop.width(), 0);
    assert_eq!(crop.rows(), 0);
}

#[test]
fn sub_rows_empty() {
    let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
    let sub = buf.rows(2, 0);
    assert_eq!(sub.rows(), 0);
}

// --- with_descriptor assertion ---

#[test]
fn with_descriptor_metadata_change_succeeds() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    // Changing transfer function is metadata-only -- should succeed
    let buf2 = buf.with_descriptor(PixelDescriptor::RGB8);
    assert_eq!(buf2.descriptor(), PixelDescriptor::RGB8);
}

#[test]
#[should_panic(expected = "with_descriptor() cannot change physical layout")]
fn with_descriptor_layout_change_panics() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
    // Trying to change from RGB8 to RGBA8 -- different layout, should panic
    let _ = buf.with_descriptor(PixelDescriptor::RGBA8);
}

#[test]
fn with_descriptor_slice_assertion() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    let slice = buf.as_slice();
    // Metadata change OK
    let s2 = slice.with_descriptor(PixelDescriptor::RGB8);
    assert_eq!(s2.descriptor(), PixelDescriptor::RGB8);
}

#[test]
#[should_panic(expected = "with_descriptor() cannot change physical layout")]
fn with_descriptor_slice_layout_change_panics() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
    let slice = buf.as_slice();
    let _ = slice.with_descriptor(PixelDescriptor::RGBA8);
}

// --- reinterpret ---

#[test]
fn reinterpret_same_bpp_succeeds() {
    // RGBA8 -> BGRA8: same 4 bpp, different layout
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8);
    let buf2 = buf.reinterpret(PixelFormat::Bgra8.descriptor()).unwrap();
    assert_eq!(buf2.descriptor().layout(), ChannelLayout::Bgra);
}

#[test]
fn reinterpret_different_bpp_fails() {
    // RGB8 (3 bpp) -> RGBA8 (4 bpp): different bytes_per_pixel
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
    let err = buf.reinterpret(PixelDescriptor::RGBA8);
    assert_eq!(
        *err.unwrap_err().error(),
        BufferError::IncompatibleDescriptor
    );
}

#[test]
fn reinterpret_rgbx_to_rgba() {
    // RGBX8 -> RGBA8: same bpp (4), reinterpret padding as alpha
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBX8);
    let buf2 = buf.reinterpret(PixelDescriptor::RGBA8).unwrap();
    assert!(buf2.descriptor().has_alpha());
}

// --- Per-field metadata setters ---

#[test]
fn per_field_setters() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8);
    let buf = buf.with_transfer(TransferFunction::Srgb);
    assert_eq!(buf.descriptor().transfer(), TransferFunction::Srgb);
    let buf = buf.with_primaries(ColorPrimaries::DisplayP3);
    assert_eq!(buf.descriptor().primaries, ColorPrimaries::DisplayP3);
    let buf = buf.with_signal_range(SignalRange::Narrow);
    assert!(matches!(buf.descriptor().signal_range, SignalRange::Narrow));
    let buf = buf.with_alpha_mode(Some(AlphaMode::Premultiplied));
    assert_eq!(buf.descriptor().alpha(), Some(AlphaMode::Premultiplied));
}

// --- copy_to_contiguous_bytes ---

#[test]
fn copy_to_contiguous_bytes_tight() {
    let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    {
        let mut s = buf.as_slice_mut();
        s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        s.row_mut(1).copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    }
    let bytes = buf.copy_to_contiguous_bytes();
    assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

#[test]
fn copy_to_contiguous_bytes_padded() {
    // Use SIMD-aligned buffer which will have stride padding for small widths
    let mut buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
    let stride = buf.stride();
    // Stride should be >= 6 (2 pixels * 3 bytes) and aligned to lcm(3, 16) = 48
    assert!(stride >= 6);
    {
        let mut s = buf.as_slice_mut();
        s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        s.row_mut(1).copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    }
    let bytes = buf.copy_to_contiguous_bytes();
    // Should only contain the actual pixel data, no padding
    assert_eq!(bytes.len(), 12);
    assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

// --- BufferError Display for all variants ---

#[test]
fn buffer_error_display_alignment_violation() {
    let msg = format!("{}", BufferError::AlignmentViolation);
    assert_eq!(msg, "data is not aligned for the channel type");
}

#[test]
fn buffer_error_display_insufficient_data() {
    let msg = format!("{}", BufferError::InsufficientData);
    assert_eq!(msg, "data slice is too small for the given dimensions");
}

#[test]
fn buffer_error_display_invalid_dimensions() {
    let msg = format!("{}", BufferError::InvalidDimensions);
    assert_eq!(msg, "width or height is zero or causes overflow");
}

#[test]
fn buffer_error_display_incompatible_descriptor() {
    let msg = format!("{}", BufferError::IncompatibleDescriptor);
    assert_eq!(
        msg,
        "descriptor conflicts with pixel layout, type, or alpha semantics"
    );
}

#[test]
fn buffer_error_display_allocation_failed() {
    let msg = format!("{}", BufferError::AllocationFailed);
    assert_eq!(msg, "buffer allocation failed");
}

// --- PixelSlice::is_contiguous ---

#[test]
fn pixel_slice_is_contiguous_tight() {
    // Tight buffer: stride == width * bpp
    let buf = PixelBuffer::new(4, 3, PixelDescriptor::RGBA8_SRGB);
    let slice = buf.as_slice();
    // stride should be 4 * 4 = 16
    assert_eq!(slice.stride(), 16);
    assert!(slice.is_contiguous());
}

// --- PixelSlice::as_contiguous_bytes ---

#[test]
fn pixel_slice_as_contiguous_bytes_tight() {
    let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
    {
        let mut s = buf.as_slice_mut();
        s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        s.row_mut(1)
            .copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
    }
    let slice = buf.as_slice();
    let bytes = slice.as_contiguous_bytes();
    assert!(bytes.is_some());
    assert_eq!(
        bytes.unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[test]
fn pixel_slice_as_contiguous_bytes_padded_returns_none() {
    // SIMD-aligned buffer will have stride padding for small widths
    let buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
    let slice = buf.as_slice();
    // stride > width * bpp, so not contiguous
    assert!(slice.stride() > 6);
    assert!(!slice.is_contiguous());
    assert!(slice.as_contiguous_bytes().is_none());
}

// --- PixelSlice::as_strided_bytes ---

#[test]
fn pixel_slice_as_strided_bytes_tight() {
    let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
    {
        let mut s = buf.as_slice_mut();
        s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        s.row_mut(1)
            .copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
    }
    let slice = buf.as_slice();
    let bytes = slice.as_strided_bytes();
    // Tight layout: strided bytes == contiguous bytes
    assert_eq!(
        bytes,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[test]
fn pixel_slice_as_strided_bytes_padded() {
    let buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
    let slice = buf.as_slice();
    let stride = slice.stride();
    assert!(stride > 6, "expected padding for SIMD alignment");
    let bytes = slice.as_strided_bytes();
    // Total length includes stride padding between and after rows
    assert_eq!(bytes.len(), stride * 2);
}

#[test]
fn pixel_slice_as_strided_bytes_sub_rows() {
    let buf = simd_padded(2, 4, PixelDescriptor::RGB8_SRGB, 16);
    let slice = buf.as_slice();
    let stride = slice.stride();
    let sub = slice.sub_rows(1, 2);
    let bytes = sub.as_strided_bytes();
    // sub_rows trims trailing padding on last row: (count-1)*stride + width*bpp
    let expected_len = stride + 2 * 3; // 1 stride + 6 pixel bytes
    assert_eq!(bytes.len(), expected_len);
}

#[test]
fn pixel_slice_mut_as_strided_bytes() {
    let mut buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
    let mut slice = buf.as_slice_mut();
    let stride = slice.stride();
    // Write via as_strided_bytes_mut
    let bytes = slice.as_strided_bytes_mut();
    assert_eq!(bytes.len(), stride * 2);
    bytes[0] = 42;
    // Verify through row accessor
    assert_eq!(slice.row(0)[0], 42);
}

// --- PixelSlice::row_with_stride ---

#[test]
fn pixel_slice_row_with_stride_padded() {
    // Create a padded buffer via SIMD alignment
    let buf = simd_padded(2, 2, PixelDescriptor::RGBA8_SRGB, 64);
    let slice = buf.as_slice();
    let stride = slice.stride();
    // stride should be >= 8 (2 * 4 bpp) and aligned
    assert!(stride >= 8);
    // row_with_stride returns the full stride bytes including padding
    let full_row = slice.row_with_stride(0);
    assert_eq!(full_row.len(), stride);
    // row() returns only the pixel data (no padding)
    let pixel_row = slice.row(0);
    assert_eq!(pixel_row.len(), 8); // 2 pixels * 4 bytes
}

// --- PixelBuffer::has_alpha ---

#[test]
fn pixel_buffer_has_alpha_rgba8() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
    assert!(buf.has_alpha());
}

#[test]
fn pixel_buffer_has_alpha_rgb8_false() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    assert!(!buf.has_alpha());
}

// --- PixelBuffer::is_grayscale ---

#[test]
fn pixel_buffer_is_grayscale_gray8() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::GRAY8_SRGB);
    assert!(buf.is_grayscale());
}

#[test]
fn pixel_buffer_is_grayscale_rgb8_false() {
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    assert!(!buf.is_grayscale());
}

// --- PixelSlice::try_typed (requires "rgb" feature) ---

#[cfg(feature = "rgb")]
#[test]
fn pixel_slice_try_typed_success() {
    use rgb::Rgba;

    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
    let slice = buf.as_slice();
    let typed: Option<PixelSlice<'_, Rgba<u8>>> = slice.try_typed();
    assert!(typed.is_some());
    let typed = typed.unwrap();
    assert_eq!(typed.width(), 2);
    assert_eq!(typed.rows(), 2);
}

// --- PixelBuffer::try_typed ---

#[cfg(feature = "rgb")]
#[test]
fn pixel_buffer_try_typed_success() {
    use rgb::Rgba;

    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
    let typed: Option<PixelBuffer<Rgba<u8>>> = buf.try_typed();
    assert!(typed.is_some());
    let typed = typed.unwrap();
    assert_eq!(typed.width(), 2);
    assert_eq!(typed.height(), 2);
}

#[cfg(feature = "rgb")]
#[test]
fn pixel_buffer_try_typed_failure_wrong_layout() {
    use rgb::Rgba;

    // RGB8 buffer cannot be typed as Rgba<u8>
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    let typed: Option<PixelBuffer<Rgba<u8>>> = buf.try_typed();
    assert!(typed.is_none());
}

// --- Security audit 2026-05-06: 32-bit overflow safety regressions ---
//
// These dimensions overflow `usize` arithmetic on 32-bit targets but are
// representable on 64-bit, so the tests are gated to run under `cross`
// or any other 32-bit build (e.g. i686-unknown-linux-gnu). On 64-bit the
// overflow path is unreachable for `u32 x u32 x small-const` inputs;
// running the suite under 32-bit exercises the actual guard.

#[cfg(target_pointer_width = "32")]
#[test]
fn from_vec_rejects_stride_height_overflow_32bit() {
    // RGB8: bpp=3. stride * height overflows 32-bit usize.
    // 0xFFFF * 3 = 0x2FFFD, * 0xFFFF heights overflows.
    let data = alloc::vec::Vec::<u8>::new();
    let err = PixelBuffer::from_vec(data, 0xFFFF, 0xFFFF, PixelDescriptor::RGB8_SRGB)
        .expect_err("expected overflow to be reported");
    assert_eq!(*err.error(), BufferError::InvalidDimensions);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[cfg(feature = "rgb")]
fn from_pixels_rejects_width_height_overflow_32bit() {
    use rgb::Rgb;
    // 2^16 * 2^16 == 2^32, wraps to 0 on 32-bit. With the fix this
    // returns InvalidDimensions instead of accepting an empty vec
    // for a (65536, 65536) buffer.
    let pixels: alloc::vec::Vec<Rgb<u8>> = alloc::vec::Vec::new();
    let err = PixelBuffer::<Rgb<u8>>::from_pixels(pixels, 1 << 16, 1 << 16)
        .expect_err("expected width*height overflow to be reported");
    assert_eq!(*err.error(), BufferError::InvalidDimensions);
}

#[cfg(feature = "rgb")]
#[test]
fn from_pixels_length_mismatch_does_not_panic() {
    // On 64-bit this is the normal length-mismatch path; on 32-bit the
    // overflow guard activates first. Either way, we get a clean error
    // and never index out of bounds.
    use rgb::Rgb;
    let pixels: alloc::vec::Vec<Rgb<u8>> = alloc::vec![Rgb { r: 0, g: 0, b: 0 }; 4];
    // (3, 3) wants 9 pixels; we provided 4.
    let err =
        PixelBuffer::<Rgb<u8>>::from_pixels(pixels, 3, 3).expect_err("length mismatch must error");
    assert_eq!(*err.error(), BufferError::InvalidDimensions);
}

#[test]
fn from_vec_insufficient_data_clean_error() {
    // Standard insufficient-data path -- guards offset+total before the
    // length check post-fix. Verify the existing happy-path negative
    // still surfaces as InsufficientData (not panic) under the new
    // checked_add guard.
    let data = alloc::vec::Vec::<u8>::new();
    let err = PixelBuffer::from_vec(data, 4, 4, PixelDescriptor::RGB8_SRGB)
        .expect_err("empty data must error for non-zero dims");
    assert_eq!(*err.error(), BufferError::InsufficientData);
}

#[test]
fn crop_copy_round_trip_does_not_overflow() {
    // Sanity check that crop_copy still works for normal inputs after
    // the overflow guard was added. The H1 fix only changes overflow
    // behavior (deferred slice OOB -> eager panic with named message);
    // we exercise that the happy path still returns a tightly-packed
    // buffer of the requested size.
    let buf = PixelBuffer::new(8, 4, PixelDescriptor::RGB8_SRGB);
    let cropped = buf.crop_copy(1, 1, 3, 2).unwrap();
    assert_eq!(cropped.width(), 3);
    assert_eq!(cropped.height(), 2);
    assert_eq!(cropped.stride(), 3 * 3); // tightly packed RGB8
}

#[cfg(all(test, feature = "imgref"))]
mod buffer_tests {
    use super::*;
    use alloc::vec::Vec;
    use rgb::{Gray, Rgb};

    // --- ImgRef -> PixelSlice -> row access ---

    #[test]
    fn imgref_to_pixel_slice_rgb8() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 10,
                g: 20,
                b: 30,
            },
            Rgb {
                r: 40,
                g: 50,
                b: 60,
            },
            Rgb {
                r: 70,
                g: 80,
                b: 90,
            },
            Rgb {
                r: 100,
                g: 110,
                b: 120,
            },
        ];
        let img = imgref::Img::new(pixels.as_slice(), 2, 2);
        let slice: PixelSlice<'_, Rgb<u8>> = img.into();
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.rows(), 2);
        assert_eq!(slice.row(0), &[10, 20, 30, 40, 50, 60]);
        assert_eq!(slice.row(1), &[70, 80, 90, 100, 110, 120]);
    }

    #[test]
    fn imgref_to_pixel_slice_gray16() {
        let pixels = vec![Gray::new(1000u16), Gray::new(2000u16)];
        let img = imgref::Img::new(pixels.as_slice(), 2, 1);
        let slice: PixelSlice<'_, Gray<u16>> = img.into();
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.rows(), 1);
        assert_eq!(slice.descriptor(), PixelDescriptor::GRAY16);
        // Bytes should be native-endian u16
        let row = slice.row(0);
        assert_eq!(row.len(), 4);
        let v0 = u16::from_ne_bytes([row[0], row[1]]);
        let v1 = u16::from_ne_bytes([row[2], row[3]]);
        assert_eq!(v0, 1000);
        assert_eq!(v1, 2000);
    }

    // --- from_pixels_erased ---

    #[test]
    fn from_pixels_erased_matches_manual() {
        let pixels1: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 10,
                g: 20,
                b: 30,
            },
            Rgb {
                r: 40,
                g: 50,
                b: 60,
            },
        ];
        let pixels2 = pixels1.clone();

        // Manual: from_pixels + into
        let manual: PixelBuffer = PixelBuffer::from_pixels(pixels1, 2, 1).unwrap().into();

        // Erased: from_pixels_erased
        let erased = PixelBuffer::from_pixels_erased(pixels2, 2, 1).unwrap();

        assert_eq!(manual.width(), erased.width());
        assert_eq!(manual.height(), erased.height());
        assert_eq!(manual.descriptor(), erased.descriptor());
        assert_eq!(manual.as_slice().row(0), erased.as_slice().row(0));
    }

    #[test]
    fn from_pixels_erased_dimension_mismatch() {
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 1, g: 2, b: 3 }];
        let err = PixelBuffer::from_pixels_erased(pixels, 2, 1);
        assert_eq!(*err.unwrap_err().error(), BufferError::InvalidDimensions);
    }
}
