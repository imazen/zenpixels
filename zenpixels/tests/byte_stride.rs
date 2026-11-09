//! Rows whose stride isn't a whole number of pixels (imazen/zenpixels#91).
//!
//! Every view of such rows must give the same pixels as the packed form and leave the padding
//! alone. Whatever counts its stride in pixels must refuse them rather than misread them.

use image::flat::FlatSamples;
#[cfg(feature = "imgref")]
use imgref::ImgVec;
#[cfg(feature = "rgb")]
use rgb::{Rgb, Rgba};
use zenpixels::sample::ExactDepth;
use zenpixels::{
    BufferError, ByteStorage, PixelBuffer, PixelBufferParts, PixelDescriptor as D, PixelSlice,
    PixelSliceMut, TransferFunction,
};

const PAD: u8 = 0xEE;

/// `(descriptor, width, height, stride)`: strides that leave a part of a pixel at the end of each row.
fn cases() -> Vec<(D, u32, u32, usize)> {
    let mut cases = vec![
        // RGB8 two pixels, 8 bytes apart: the case in the issue.
        (D::RGB8_SRGB, 2, 3, 8),
        // Native JXL f32 RGB with 16 bytes of row padding.
        (D::RGBF32_LINEAR, 5, 3, 76),
        (D::RGBF32_LINEAR, 3, 4, 40),
        (D::RGB16_SRGB, 5, 3, 32),
        (D::RGB16_SRGB, 1, 2, 8),
        (D::RGBA16_SRGB, 2, 2, 18),
        (D::RGBA8_SRGB, 3, 2, 13),
        (D::GRAYA16_SRGB, 3, 2, 14),
        (D::BGRA8_SRGB, 3, 2, 15),
        (D::CMYK8, 2, 3, 9),
        (D::CMYK16, 2, 3, 18),
        (
            D::from_pixel_format(zenpixels::PixelFormat::RgbF16),
            2,
            3,
            14,
        ),
    ];
    for extra in [1usize, 2, 4, 13, 49] {
        cases.push((D::RGB8_SRGB, 7, 5, 21 + extra));
    }
    cases
}

/// Packed pixels and the same pixels at `stride`, with `PAD` in every byte that isn't a sample.
fn fixture(d: D, width: u32, height: u32, stride: usize) -> (Vec<u8>, Vec<u32>) {
    let row = width as usize * d.bytes_per_pixel();
    let packed: Vec<u8> = (0..row * height as usize)
        .map(|i| (i * 7 + 3) as u8)
        .collect();
    let total = (height as usize - 1) * stride + row;
    let mut words = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
    for (y, source) in packed.chunks_exact(row).enumerate() {
        bytes[y * stride..y * stride + row].copy_from_slice(source);
    }
    (packed, words)
}

fn assert_padding_is_pad(bytes: &[u8], row: usize, stride: usize, what: &str) {
    for (y, line) in bytes.chunks(stride).enumerate() {
        if line.len() > row {
            assert!(
                line[row..].iter().all(|&b| b == PAD),
                "{what}: row {y} padding changed"
            );
        }
    }
}

#[test]
fn every_row_view_matches_the_packed_pixels() {
    for (d, w, h, stride) in cases() {
        assert_ne!(
            stride % d.bytes_per_pixel(),
            0,
            "{d:?}: fixture must be byte-strided"
        );
        let (packed, words) = fixture(d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let bytes = &bytes[..(h as usize - 1) * stride + row];
        let slice = PixelSlice::new(bytes, w, h, stride, d).unwrap();
        assert_eq!(slice.stride(), stride);
        assert!(!slice.is_contiguous());
        assert!(slice.as_contiguous_bytes().is_none());
        assert_eq!(&*slice.contiguous_bytes(), &packed[..]);
        assert_eq!(&*slice.try_contiguous_bytes().unwrap(), &packed[..]);
        for y in 0..h {
            let start = y as usize * row;
            assert_eq!(slice.row(y), &packed[start..start + row], "{d:?} row {y}");
            assert_eq!(slice.try_row(y).unwrap(), &packed[start..start + row]);
            let sub = slice.sub_rows(y, h - y);
            assert_eq!(sub.stride(), stride);
            assert_eq!(sub.row(0), &packed[start..start + row]);
            for x in 0..w {
                let bpp = d.bytes_per_pixel();
                let crop = slice.crop_view(x, y, w - x, h - y);
                let at = start + x as usize * bpp;
                assert_eq!(crop.row(0), &packed[at..start + row], "{d:?} crop {x},{y}");
                assert_eq!(
                    slice.get_pixel(x, y).unwrap(),
                    PixelSlice::new(&packed[at..at + bpp], 1, 1, bpp, d)
                        .unwrap()
                        .get_pixel(0, 0)
                        .unwrap()
                );
            }
        }
        assert_eq!(slice.get_pixel(w, 0), None);
        let tight = slice.try_to_tight_buffer().unwrap();
        assert_eq!(tight.stride(), row);
        assert_eq!(tight.as_contiguous_bytes().unwrap(), &packed[..]);
        assert_eq!(slice.as_strided_bytes().len(), bytes.len());
    }
}

/// Reading a row, with the view split at every row boundary, gives the same bytes.
#[test]
fn splitting_rows_does_not_change_them() {
    for (d, w, h, stride) in cases() {
        let (packed, words) = fixture(d, w, h, stride);
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let row = w as usize * d.bytes_per_pixel();
        let slice =
            PixelSlice::new(&bytes[..(h as usize - 1) * stride + row], w, h, stride, d).unwrap();
        for split in 0..=h {
            let (top, bottom) = (slice.sub_rows(0, split), slice.sub_rows(split, h - split));
            let mut joined = top.contiguous_bytes().into_owned();
            joined.extend_from_slice(&bottom.contiguous_bytes());
            assert_eq!(joined, packed, "{d:?} split at {split}");
        }
    }
}

#[test]
fn final_row_needs_no_padding_and_a_short_buffer_is_refused() {
    for (d, w, h, stride) in cases() {
        let row = w as usize * d.bytes_per_pixel();
        let (_, words) = fixture(d, w, h, stride);
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let exact = (h as usize - 1) * stride + row;
        assert!(PixelSlice::new(&bytes[..exact], w, h, stride, d).is_ok());
        assert_eq!(
            *PixelSlice::new(&bytes[..exact - 1], w, h, stride, d)
                .unwrap_err()
                .error(),
            BufferError::InsufficientData
        );
        assert_eq!(
            *PixelSlice::new(bytes, w, h, row - 1, d)
                .unwrap_err()
                .error(),
            BufferError::StrideTooSmall
        );
    }
}

#[test]
fn stride_must_be_whole_samples_everywhere() {
    let words = vec![0u32; 64];
    for (d, width, stride) in [
        (D::RGB16_SRGB, 2, 13usize),
        (D::RGBA16_SRGB, 1, 9),
        (D::GRAY16_SRGB, 3, 7),
        (D::RGBF32_LINEAR, 1, 14),
        (D::RGBAF32_LINEAR, 1, 17),
        (D::GRAYF32_LINEAR, 1, 6),
    ] {
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        assert_eq!(
            *PixelSlice::new(bytes, width, 2, stride, d)
                .unwrap_err()
                .error(),
            BufferError::AlignmentViolation,
            "{d:?}"
        );
        let mut copy = words.clone();
        assert_eq!(
            *PixelSliceMut::new(bytemuck::cast_slice_mut(&mut copy), width, 2, stride, d)
                .unwrap_err()
                .error(),
            BufferError::AlignmentViolation
        );
        let parts: PixelBufferParts =
            PixelBufferParts::new(words.clone().into(), width, 2, stride, d);
        assert_eq!(
            *PixelBuffer::try_from_parts(parts)
                .unwrap_err()
                .without_buffer()
                .error(),
            BufferError::AlignmentViolation
        );
    }
}

/// A descriptor with wider samples must not reinterpret rows that don't start on one.
#[test]
fn reinterpret_rechecks_stride_alignment() {
    let words = vec![0u32; 16];
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(bytes, 2, 2, 9, D::RGBA8_SRGB).unwrap();
    let wider = D::GRAYA16_SRGB;
    assert_eq!(
        *slice.reinterpret(wider).unwrap_err().error(),
        BufferError::AlignmentViolation
    );
    let slice = PixelSlice::new(bytes, 2, 2, 10, D::RGBA8_SRGB).unwrap();
    assert_eq!(slice.reinterpret(wider).unwrap().stride(), 10);
    // A byte swap on the same samples keeps its stride.
    let slice = PixelSlice::new(bytes, 2, 2, 9, D::RGBA8_SRGB).unwrap();
    assert_eq!(slice.reinterpret(D::BGRA8_SRGB).unwrap().stride(), 9);
}

#[test]
fn mutable_rows_leave_padding_alone() {
    for (d, w, h, stride) in cases() {
        let (_, mut words) = fixture(d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
        {
            let mut view = PixelSliceMut::new(bytes, w, h, stride, d).unwrap();
            for y in 0..h {
                view.row_mut(y).fill(y as u8 + 1);
            }
            let mut sub = view.sub_rows_mut(h - 1, 1);
            sub.row_mut(0).fill(0x77);
        }
        assert_padding_is_pad(
            &bytes[..(h as usize - 1) * stride + row],
            row,
            stride,
            "row_mut",
        );
        for y in 0..h as usize {
            let want = if y + 1 == h as usize {
                0x77
            } else {
                y as u8 + 1
            };
            assert!(
                bytes[y * stride..y * stride + row]
                    .iter()
                    .all(|&b| b == want)
            );
        }
    }
}

#[cfg(feature = "rgb")]
#[test]
fn byte_swaps_stay_in_their_rows() {
    let (packed, mut words) = fixture(D::RGBA8_SRGB, 3, 2, 13);
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
    let view: PixelSliceMut<'_, Rgba<u8>> = PixelSliceMut::new(bytes, 3, 2, 13, D::RGBA8_SRGB)
        .unwrap()
        .try_typed()
        .unwrap();
    let swapped = view.swap_to_bgra();
    for y in 0..2u32 {
        for (x, pixel) in swapped.row(y).chunks_exact(4).enumerate() {
            let source = &packed[y as usize * 12 + x * 4..][..4];
            assert_eq!(pixel, [source[2], source[1], source[0], source[3]]);
        }
    }
    drop(swapped);
    assert_padding_is_pad(&bytes[..13 + 12], 12, 13, "swap_to_bgra");
}

#[test]
fn owned_buffers_adopt_compact_and_keep_their_storage() {
    for (d, w, h, stride) in cases() {
        let (packed, words) = fixture(d, w, h, stride);
        let pointer = words.as_ptr() as *const u8;
        let parts: PixelBufferParts = PixelBufferParts::new(words.into(), w, h, stride, d);
        let buffer = PixelBuffer::try_from_parts(parts).unwrap();
        assert_eq!(buffer.stride(), stride);
        assert!(buffer.as_contiguous_bytes().is_none());
        assert_eq!(buffer.copy_to_contiguous_bytes(), packed);
        assert_eq!(buffer.try_copy_to_contiguous_bytes().unwrap(), packed);
        assert_eq!(buffer.rows(0, h).contiguous_bytes().as_ref(), &packed[..]);
        assert_eq!(
            buffer
                .crop_copy(0, 0, w, h)
                .unwrap()
                .copy_to_contiguous_bytes(),
            packed
        );
        let parts = buffer.into_parts();
        assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
        assert_eq!(parts.stride_bytes, stride);

        let packed_buffer = PixelBuffer::try_from_parts(parts)
            .unwrap()
            .into_contiguous();
        assert_eq!(packed_buffer.stride(), w as usize * d.bytes_per_pixel());
        assert_eq!(packed_buffer.as_contiguous_bytes().unwrap(), &packed[..]);
    }
}

#[cfg(feature = "rgb")]
#[test]
fn into_contiguous_pixels_compacts_byte_strides() {
    let (packed, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let buffer = PixelBuffer::try_from_parts(parts).unwrap();
    let pixels = buffer.into_contiguous_pixels::<Rgb<u8>>().unwrap();
    assert_eq!(bytemuck::cast_slice::<Rgb<u8>, u8>(&pixels), &packed[..]);
}

/// `imgref` counts its stride in pixels, so it refuses rows it cannot represent.
#[cfg(feature = "imgref")]
#[test]
fn imgref_preserves_optional_checked_conversion() {
    let (_, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let mut buffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert!(buffer.try_as_imgref::<Rgb<u8>>().is_none());
    assert!(buffer.try_as_imgref_mut::<Rgb<u8>>().is_none());
    assert!(buffer.try_as_imgref::<Rgba<u8>>().is_none());
    assert!(buffer.try_as_imgref_mut::<Rgb<f32>>().is_none());
}

#[cfg(feature = "imgref")]
#[test]
fn imgref_still_works_on_whole_pixel_strides() {
    let (packed, words) = fixture(D::RGB8_SRGB, 2, 3, 9);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 9, D::RGB8_SRGB);
    let mut buffer = PixelBuffer::try_from_parts(parts).unwrap();
    let image = buffer.try_as_imgref::<Rgb<u8>>().unwrap();
    assert_eq!(image.stride(), 3);
    let read: Vec<u8> = image
        .rows()
        .flat_map(|row| bytemuck::cast_slice::<Rgb<u8>, u8>(row).to_vec())
        .collect();
    assert_eq!(read, packed);
    assert_eq!(buffer.try_as_imgref_mut::<Rgb<u8>>().unwrap().stride(), 3);
}

#[cfg(feature = "imgref")]
#[test]
#[should_panic(expected = "ImgRef requires a stride in whole pixels")]
fn typed_as_imgref_panics_with_the_reason() {
    let (_, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let typed = PixelBuffer::try_from_parts(parts)
        .unwrap()
        .try_typed::<Rgb<u8>>()
        .unwrap();
    let _ = typed.as_imgref();
}

#[cfg(feature = "imgref")]
#[test]
#[should_panic(expected = "ImgRef requires a stride in whole pixels")]
fn typed_as_imgref_mut_panics_with_the_reason() {
    let (_, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let mut typed = PixelBuffer::try_from_parts(parts)
        .unwrap()
        .try_typed::<Rgb<u8>>()
        .unwrap();
    let _ = typed.as_imgref_mut();
}

/// The error from moving into an `ImgVec` is the buffer, unchanged.
#[cfg(feature = "imgref")]
#[test]
fn imgvec_refuses_and_returns_the_buffer() {
    let (packed, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let typed = PixelBuffer::try_from_parts(parts)
        .unwrap()
        .try_typed::<Rgb<u8>>()
        .unwrap();
    let back = ImgVec::<Rgb<u8>>::try_from(typed).unwrap_err();
    assert_eq!(back.stride(), 8);
    assert_eq!(back.copy_to_contiguous_bytes(), packed);
}

/// Exact-depth scans read rows, not padding, and a refused mutable view keeps the evidence.
#[test]
fn exact_depth_scans_rows_and_survives_a_refusal() {
    // 8-bit codes widened to 16 bits, with padding that is on no lower grid.
    let (w, h, stride) = (5u32, 3u32, 32usize);
    let d = D::RGB16_SRGB;
    let mut samples = vec![0x1234u16; (h as usize - 1) * stride / 2 + 15];
    for y in 0..h as usize {
        for i in 0..15 {
            samples[y * stride / 2 + i] = ((y * 40 + i * 13) as u16 & 0xff) * 257;
        }
    }
    let bytes = bytemuck::cast_slice::<u16, u8>(&samples);
    let slice = PixelSlice::new(bytes, w, h, stride, d).unwrap();
    let measured = ExactDepth::measure(&slice).unwrap();
    assert_eq!(measured.color_bits(), 8);
    assert!(measured.holds_for(&slice).unwrap());
    let tight = slice.try_to_tight_buffer().unwrap();
    assert_eq!(ExactDepth::measure(&tight.as_slice()).unwrap(), measured);

    let parts: PixelBufferParts = PixelBufferParts::new(samples.into(), w, h, stride, d);
    let mut buffer = PixelBuffer::try_from_parts(parts).unwrap();
    buffer.set_exact_depth(Some(measured)).unwrap();
    #[cfg(feature = "imgref")]
    {
        assert!(buffer.try_as_imgref_mut::<Rgb<u16>>().is_none());
        assert_eq!(buffer.exact_depth(), Some(measured));
    }
    buffer.as_slice_mut();
    assert_eq!(buffer.exact_depth(), None);
}

/// A typed row cast works on every row of a byte-strided view.
#[cfg(feature = "rgb")]
#[test]
fn typed_views_read_each_row() {
    let (packed, words) = fixture(D::RGBF32_LINEAR, 5, 3, 76);
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(bytes, 5, 3, 76, D::RGBF32_LINEAR).unwrap();
    let typed = slice.try_typed::<Rgb<f32>>().unwrap();
    assert_eq!(typed.stride(), 76);
    let mut read = Vec::new();
    for y in 0..3 {
        let row: &[Rgb<f32>] = bytemuck::cast_slice(typed.row(y));
        assert_eq!(row.len(), 5);
        read.extend_from_slice(bytemuck::cast_slice::<Rgb<f32>, u8>(row));
    }
    assert_eq!(read, packed);
}

#[cfg(feature = "rgb")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
struct Aligned4Rgba([u8; 4]);

#[cfg(feature = "rgb")]
impl zenpixels::Pixel for Aligned4Rgba {
    const DESCRIPTOR: D = D::RGBA8_SRGB;
}

/// A custom pixel more aligned than its samples can't view rows that don't start on one, so a
/// row cast never panics on a view `try_typed` handed out.
#[cfg(feature = "rgb")]
#[test]
fn try_typed_requires_every_row_aligned_for_the_pixel() {
    let words = vec![0u32; 16];
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(bytes, 2, 2, 9, D::RGBA8_SRGB).unwrap();
    assert!(slice.clone().try_typed::<Rgba<u8>>().is_some());
    assert!(slice.try_typed::<Aligned4Rgba>().is_none());
    let slice = PixelSlice::new(bytes, 2, 2, 12, D::RGBA8_SRGB).unwrap();
    let typed = slice.try_typed::<Aligned4Rgba>().unwrap();
    for y in 0..2 {
        assert_eq!(
            bytemuck::cast_slice::<u8, Aligned4Rgba>(typed.row(y)).len(),
            2
        );
    }

    let mut copy = words.clone();
    let view =
        PixelSliceMut::new(bytemuck::cast_slice_mut(&mut copy), 2, 2, 9, D::RGBA8_SRGB).unwrap();
    assert!(view.try_typed::<Aligned4Rgba>().is_none());

    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 2, 9, D::RGBA8_SRGB);
    let buffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert!(buffer.try_typed::<Aligned4Rgba>().is_none());
}

/// `image` itself reads the strided view back.
#[test]
fn image_flat_samples_read_byte_strided_rows() {
    let (packed, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(&bytes[..8 * 2 + 6], 2, 3, 8, D::RGB8_SRGB).unwrap();
    let flat = FlatSamples::<&[u8]>::try_from(slice).unwrap();
    assert_eq!(flat.layout.height_stride, 8);
    let view = flat.as_view::<image::Rgb<u8>>().unwrap();
    for y in 0..3u32 {
        for x in 0..2u32 {
            use image::GenericImageView;
            let at = (y * 6 + x * 3) as usize;
            assert_eq!(
                view.get_pixel(x, y).0,
                [packed[at], packed[at + 1], packed[at + 2]]
            );
        }
    }

    let (packed, words) = fixture(D::RGB16_SRGB, 5, 3, 32);
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(&bytes[..32 * 2 + 30], 5, 3, 32, D::RGB16_SRGB).unwrap();
    let flat = FlatSamples::<&[u16]>::try_from(slice).unwrap();
    assert_eq!(flat.layout.height_stride, 16);
    let view = flat.as_view::<image::Rgb<u16>>().unwrap();
    use image::GenericImageView;
    let last = view.get_pixel(4, 2).0;
    let at = 2 * 30 + 4 * 6;
    let want: [u16; 3] = bytemuck::pod_read_unaligned(&packed[at..at + 6]);
    assert_eq!(last, want);
}

/// `image::ImageBuffer` has no stride, so conversion into one needs tight rows.
#[test]
fn image_buffer_refuses_byte_strided_rows() {
    let (_, words) = fixture(D::RGB8_SRGB, 2, 3, 8);
    let parts: PixelBufferParts = PixelBufferParts::new(words.into(), 2, 3, 8, D::RGB8_SRGB);
    let buffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert!(image::RgbImage::try_from(buffer).is_err());
}

#[test]
fn relabeling_keeps_a_byte_stride() {
    let words = vec![0u32; 16];
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let slice = PixelSlice::new(bytes, 2, 2, 8, D::RGB8_SRGB)
        .unwrap()
        .with_transfer(TransferFunction::Linear);
    assert_eq!(slice.stride(), 8);
    assert_eq!(slice.row(1).as_ptr(), bytes[8..].as_ptr());
}
