use super::*;
use crate::descriptor::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, SignalRange, TransferFunction,
};
use alloc::format;
use alloc::vec;

// --- PixelBuffer allocation and row access ---

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
    assert_eq!(err.unwrap_err(), BufferError::InsufficientData);
}

#[test]
fn pixel_buffer_into_vec_roundtrip() {
    let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);
    let v = buf.into_vec();
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
    let buf = PixelBuffer::new_simd_aligned(10, 5, PixelDescriptor::RGBA8_SRGB, 64);
    assert_eq!(buf.width(), 10);
    assert_eq!(buf.height(), 5);
    // RGBA8 bpp=4, lcm(4,64)=64, raw=40 -> stride=64
    assert_eq!(buf.stride(), 64);
    // First row should be 64-byte aligned
    let slice = buf.as_slice();
    assert_eq!(slice.data.as_ptr() as usize % 64, 0);
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
    let cropped = buf.crop_copy(1, 1, 2, 2);
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
    assert_eq!(err.unwrap_err(), BufferError::StrideTooSmall);
}

#[test]
fn pixel_slice_insufficient_data() {
    let data = [0u8; 10];
    let err = PixelSlice::new(&data, 10, 1, 30, PixelDescriptor::RGB8_SRGB);
    assert_eq!(err.unwrap_err(), BufferError::InsufficientData);
}

#[test]
fn pixel_slice_zero_rows() {
    let data = [0u8; 0];
    let slice = PixelSlice::new(&data, 10, 0, 30, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(slice.rows(), 0);
}

#[test]
fn stride_not_pixel_aligned_rejected() {
    // RGB8 bpp=3, stride=32 is not a multiple of 3
    let data = [0u8; 128];
    let err = PixelSlice::new(&data, 10, 1, 32, PixelDescriptor::RGB8_SRGB);
    assert_eq!(err.unwrap_err(), BufferError::StrideNotPixelAligned);

    // stride=33 IS a multiple of 3 -> accepted
    let ok = PixelSlice::new(&data, 10, 1, 33, PixelDescriptor::RGB8_SRGB);
    assert!(ok.is_ok());
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
    let buf2 = buf.reinterpret(PixelDescriptor::BGRA8).unwrap();
    assert_eq!(buf2.descriptor().layout(), ChannelLayout::Bgra);
}

#[test]
fn reinterpret_different_bpp_fails() {
    // RGB8 (3 bpp) -> RGBA8 (4 bpp): different bytes_per_pixel
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
    let err = buf.reinterpret(PixelDescriptor::RGBA8);
    assert_eq!(err.unwrap_err(), BufferError::IncompatibleDescriptor);
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
    let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
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
    let mut buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGB8_SRGB, 16);
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

// ---------------------------------------------------------------------------
// imgref interop tests
// ---------------------------------------------------------------------------

#[cfg(feature = "imgref")]
mod imgref_tests {
    use super::super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use rgb::{Gray, Rgb};

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
        assert_eq!(err.unwrap_err(), BufferError::InvalidDimensions);
    }
}
