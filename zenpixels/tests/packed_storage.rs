//! `PixelBuffer` over storage other than `Vec<u8>`: nothing is copied, and
//! layout validation matches the `Vec<u8>` path.

use zenpixels::plane::{AlignedSamples, SampleStorage, SampleStorageMut};
use zenpixels::{PixelBuffer, PixelBufferParts, PixelDescriptor};

const W: u32 = 37;
const H: u32 = 11;
const N: usize = (W * H * 3) as usize;

fn adopt<S: SampleStorage>(data: S, stride: usize, d: PixelDescriptor) -> PixelBuffer<(), S> {
    PixelBuffer::try_from_parts(PixelBufferParts::new(data, W, H, stride, d)).unwrap()
}

#[test]
fn vec_u16_moves_in_and_out_without_copying() {
    let raw: Vec<u16> = (0..N).map(|i| i as u16 * 3).collect();
    let p0 = raw.as_ptr();
    let cap = raw.capacity();
    let pb = adopt(raw, W as usize * 6, PixelDescriptor::RGB16_SRGB);
    let bytes = pb.as_slice();
    assert_eq!(bytes.as_strided_bytes().as_ptr().cast::<u16>(), p0);
    let row = bytes.row(1);
    assert_eq!(
        row[..6],
        *bytemuck::cast_slice::<u16, u8>(&[333u16, 336, 339])
    );
    drop(bytes);
    let parts = pb.into_parts();
    assert_eq!((parts.data.as_ptr(), parts.data.capacity()), (p0, cap));
}

#[test]
fn vec_f32_holds_rgbf32() {
    let raw = vec![0.25f32; N];
    let p0 = raw.as_ptr();
    let pb = adopt(raw, W as usize * 12, PixelDescriptor::RGBF32_LINEAR);
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr().cast::<f32>(), p0);
}

#[test]
fn offset_and_stride_address_the_same_samples_as_the_byte_path() {
    let stride_samples = 126;
    let start = 4;
    let len = start + stride_samples * (H as usize - 1) + W as usize * 3;
    let mut raw = vec![0u16; len];
    raw[start + stride_samples] = 9;
    let as_bytes: Vec<u8> = bytemuck::cast_slice(&raw).to_vec();

    let mut wide =
        PixelBufferParts::new(raw, W, H, stride_samples * 2, PixelDescriptor::RGB16_SRGB);
    wide.offset = start * 2;
    let wide = PixelBuffer::try_from_parts(wide).unwrap();
    let copy = wide.copy_to_contiguous_bytes();

    let misalign = as_bytes.as_ptr().align_offset(2);
    let mut realigned = vec![0u8; as_bytes.len() + misalign + 2];
    let a = realigned.as_ptr().align_offset(2);
    realigned[a..a + as_bytes.len()].copy_from_slice(&as_bytes);
    let mut narrow = PixelBufferParts::new(
        realigned,
        W,
        H,
        stride_samples * 2,
        PixelDescriptor::RGB16_SRGB,
    );
    narrow.offset = a + start * 2;
    let narrow = PixelBuffer::try_from_parts(narrow).unwrap();
    assert_eq!(copy, narrow.copy_to_contiguous_bytes());
    assert_eq!(wide.as_slice().row(1)[..2], 9u16.to_ne_bytes());
}

#[test]
fn writes_reach_the_owner() {
    let mut pb = adopt(vec![0u16; N], W as usize * 6, PixelDescriptor::RGB16_SRGB);
    pb.as_slice_mut().row_mut(2)[0] = 0xAB;
    pb.rows_mut(3, 1).row_mut(0)[1] = 0xCD;
    let data = pb.into_parts().data;
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    assert_eq!(bytes[2 * W as usize * 6], 0xAB);
    assert_eq!(bytes[3 * W as usize * 6 + 1], 0xCD);
}

#[test]
fn boxes_slices_and_custom_owners_adopt() {
    let boxed: Box<[u8]> = vec![5u8; N].into_boxed_slice();
    let p0 = boxed.as_ptr();
    let pb = adopt(boxed, W as usize * 3, PixelDescriptor::RGB8_SRGB);
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr(), p0);

    let v = vec![1u8; N];
    let pb = adopt(&v[..], W as usize * 3, PixelDescriptor::RGB8_SRGB);
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr(), v.as_ptr());

    struct Pool {
        slot: u32,
        buf: Vec<u16>,
    }
    impl SampleStorage for Pool {
        type Sample = u16;
        fn samples(&self) -> &[u16] {
            &self.buf
        }
    }
    impl SampleStorageMut for Pool {
        fn samples_mut(&mut self) -> &mut [u16] {
            &mut self.buf
        }
    }
    let pool = Pool {
        slot: 7,
        buf: vec![2; N],
    };
    let p0 = pool.buf.as_ptr();
    let mut pb = adopt(pool, W as usize * 6, PixelDescriptor::RGB16_SRGB);
    pb.as_slice_mut().row_mut(0)[0] = 1;
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr().cast::<u16>(), p0);
    assert_eq!(pb.into_parts().data.slot, 7);
}

#[cfg(feature = "rgb")]
#[test]
fn storage_survives_erase_and_typing() {
    let raw = vec![0u16; N];
    let p0 = raw.as_ptr();
    let pb = adopt(raw, W as usize * 6, PixelDescriptor::RGB16_SRGB);
    let typed = pb.try_typed::<rgb::Rgb<u16>>().unwrap();
    let erased = typed.erase();
    assert_eq!(erased.into_parts().data.as_ptr(), p0);
}

#[test]
fn validation_matches_and_returns_rejected_parts() {
    // Too short.
    let short = PixelBufferParts::new(
        vec![0u16; N - 1],
        W,
        H,
        W as usize * 6,
        PixelDescriptor::RGB16_SRGB,
    );
    let mut e = PixelBuffer::try_from_parts(short).unwrap_err();
    assert_eq!(e.take_parts().unwrap().data.len(), N - 1);
    // Misaligned first sample in byte storage.
    let bytes = vec![0u8; N * 2 + 2];
    let odd = usize::from(bytes.as_ptr() as usize % 2 == 0);
    let mut parts = PixelBufferParts::new(bytes, W, H, W as usize * 6, PixelDescriptor::RGB16_SRGB);
    parts.offset = odd;
    assert!(PixelBuffer::try_from_parts(parts).is_err());
    // Storage smaller than the offset must not panic.
    let mut parts = PixelBufferParts::new(
        vec![0u16; 1],
        W,
        H,
        W as usize * 6,
        PixelDescriptor::RGB16_SRGB,
    );
    parts.offset = 100;
    assert!(PixelBuffer::try_from_parts(parts).is_err());
}

#[cfg(feature = "imgref")]
#[test]
fn imgref_views_cover_any_storage() {
    let mut pb = adopt(vec![3u16; N], W as usize * 6, PixelDescriptor::RGB16_SRGB)
        .try_typed::<rgb::Rgb<u16>>()
        .unwrap();
    assert_eq!(pb.as_imgref().width(), W as usize);
    pb.as_imgref_mut().buf_mut()[0].r = 8;
    assert_eq!(pb.as_imgref().buf()[0].r, 8);
}

#[test]
fn default_storage_is_pixel_storage() {
    fn takes_default(_: PixelBuffer) {}
    takes_default(PixelBuffer::new(1, 1, PixelDescriptor::RGB8_SRGB));
    #[cfg(feature = "rgb")]
    {
        fn takes_typed(_: PixelBuffer<rgb::Rgb<u8>>) {}
        takes_typed(PixelBuffer::new_typed(1, 1));
    }
    let parts: PixelBufferParts = PixelBuffer::new(1, 1, PixelDescriptor::RGB8_SRGB).into_parts();
    assert_eq!(zenpixels::ByteStorage::as_bytes(&parts.data).len(), 3);
}

#[test]
fn aligned_buffers_start_at_index_zero_on_64_bytes_for_every_format() {
    for d in [
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGBA16_SRGB,
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGBAF32_LINEAR,
        PixelDescriptor::GRAY8_SRGB,
        PixelDescriptor::GRAY16_SRGB,
        PixelDescriptor::GRAYF32_LINEAR,
        PixelDescriptor::GRAYA8_SRGB,
        PixelDescriptor::GRAYA16_SRGB,
        PixelDescriptor::GRAYAF32_LINEAR,
    ] {
        for (w, h) in [(1u32, 1u32), (2, 3), (37, 11), (64, 2), (4096, 2)] {
            let buffer: PixelBuffer<(), AlignedSamples<u8>> =
                PixelBuffer::try_new_aligned(w, h, d).unwrap();
            let bpp = d.bytes_per_pixel();
            assert_eq!(buffer.stride() % bpp, 0);
            assert!(buffer.stride() >= w as usize * bpp);
            let slice = buffer.as_slice();
            let bytes = slice.as_strided_bytes();
            assert_eq!(bytes.as_ptr() as usize % 64, 0);
            assert!(bytes.iter().all(|&b| b == 0));
            drop(slice);
            let parts = buffer.into_parts();
            assert_eq!(parts.offset, 0);
            assert_eq!(parts.data.as_ptr() as usize % 64, 0);
        }
    }
}

#[cfg(feature = "rgb")]
#[test]
fn aligned_buffers_view_as_their_pixel_type_on_every_row() {
    let mut buffer: PixelBuffer<(), AlignedSamples<u8>> =
        PixelBuffer::try_new_aligned(W, H, PixelDescriptor::RGBA16_SRGB).unwrap();
    buffer.as_slice_mut().row_mut(3)[0] = 0xAA;
    let slice = buffer.as_slice();
    for y in 0..H {
        let row: &[rgb::Rgba<u16>] = bytemuck::cast_slice(slice.row(y));
        assert_eq!(row.len(), W as usize);
    }
    drop(slice);
    let typed = buffer.try_typed::<rgb::Rgba<u16>>().unwrap();
    assert_eq!(typed.into_parts().data.as_ptr() as usize % 64, 0);
}

#[test]
fn aligned_buffers_can_hold_the_channel_type_directly() {
    let wide: PixelBuffer<(), AlignedSamples<u16>> =
        PixelBuffer::try_new_aligned(W, H, PixelDescriptor::RGB16_SRGB).unwrap();
    assert_eq!(wide.as_slice().row(0).len(), W as usize * 6);
    let mismatch = PixelBuffer::<(), AlignedSamples<u16>>::try_new_aligned(
        W,
        H,
        PixelDescriptor::RGBF32_LINEAR,
    );
    assert!(mismatch.is_err());
    let overflow = PixelBuffer::<(), AlignedSamples<u8>>::try_new_aligned(
        u32::MAX,
        u32::MAX,
        PixelDescriptor::RGBAF32_LINEAR,
    );
    assert!(overflow.is_err());
}

use zenpixels::{ByteStorage, ByteStorageMut, PixelStorage};

#[test]
fn allocation_uses_the_sample_type_of_the_format() {
    let cases = [
        (PixelDescriptor::RGB8_SRGB, 1usize),
        (PixelDescriptor::RGB16_SRGB, 2),
        (PixelDescriptor::RGBAF32_LINEAR, 4),
    ];
    for (d, align) in cases {
        let buffer = PixelBuffer::new(5, 3, d);
        let parts = buffer.into_parts();
        assert_eq!(parts.offset, 0);
        assert_eq!(parts.data.as_bytes().len(), 5 * 3 * d.bytes_per_pixel());
        match (parts.data, align) {
            (PixelStorage::U8(_), 1) | (PixelStorage::U16(_), 2) | (PixelStorage::F32(_), 4) => {}
            (other, _) => panic!("{d:?}: {other:?}"),
        }
    }
}

#[test]
fn storage_gives_back_only_its_own_sample_type() {
    let v: Vec<u16> = vec![1, 2, 3];
    let ptr = v.as_ptr();
    let storage = PixelStorage::from(v);
    assert_eq!(storage.as_bytes().len(), 6);
    let storage = storage.try_into_vec::<u8>().unwrap_err();
    let storage = storage.try_into_vec::<f32>().unwrap_err();
    let back = storage.try_into_vec::<u16>().unwrap();
    assert_eq!(back.as_ptr(), ptr);
}

#[test]
fn every_sample_type_round_trips_through_storage() {
    macro_rules! round_trip {
        ($($t:ty),*) => {$(
            let v: Vec<$t> = vec![<$t>::default(); 4];
            let ptr = v.as_ptr();
            let mut s = PixelStorage::from(v);
            assert_eq!(s.as_bytes().len(), 4 * size_of::<$t>());
            s.as_bytes_mut()[0] = 0;
            let back = s.try_into_vec::<$t>().unwrap();
            assert_eq!(back.as_ptr(), ptr);
        )*};
    }
    round_trip!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
}

#[test]
fn decoder_vecs_move_through_the_default_buffer() {
    let raw: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let ptr = raw.as_ptr();
    let parts = PixelBufferParts::new(
        raw.into(),
        W,
        H,
        W as usize * 12,
        PixelDescriptor::RGBF32_LINEAR,
    );
    let buffer: PixelBuffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
    let back = buffer.into_parts().data.try_into_vec::<f32>().unwrap();
    assert_eq!(back.as_ptr(), ptr);
}

#[test]
fn offset_past_u32_is_refused_not_truncated() {
    let mut parts: PixelBufferParts =
        PixelBufferParts::new(vec![0u8; 12].into(), 2, 2, 6, PixelDescriptor::RGB8_SRGB);
    if let Ok(offset) = usize::try_from(u64::from(u32::MAX) + 1) {
        parts.offset = offset;
        assert!(PixelBuffer::try_from_parts(parts).is_err());
    }
}

#[test]
fn tight_byte_handoff_is_fallible_on_storage_offset_and_stride() {
    let tight = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
    let ptr = tight.as_slice().as_strided_bytes().as_ptr();
    let bytes = tight.try_into_tight_u8_vec().unwrap();
    assert_eq!((bytes.as_ptr(), bytes.len()), (ptr, 12));

    // Wider storage.
    assert!(
        PixelBuffer::new(2, 2, PixelDescriptor::RGB16_SRGB)
            .try_into_tight_u8_vec()
            .is_err()
    );
    // A prefix before the first pixel.
    let mut parts =
        PixelBufferParts::new(vec![0u8; 15].into(), 2, 2, 6, PixelDescriptor::RGB8_SRGB);
    parts.offset = 3;
    let offset: PixelBuffer = PixelBuffer::try_from_parts(parts).unwrap();
    let offset = offset.try_into_tight_u8_vec().unwrap_err();
    assert_eq!(offset.into_parts().offset, 3);
    // Padded rows, then packing them makes the handoff succeed in place.
    let parts = PixelBufferParts::new(vec![0u8; 18].into(), 2, 2, 9, PixelDescriptor::RGB8_SRGB);
    let padded: PixelBuffer = PixelBuffer::try_from_parts(parts).unwrap();
    let padded = padded.try_into_tight_u8_vec().unwrap_err();
    assert_eq!(
        padded
            .into_contiguous()
            .try_into_tight_u8_vec()
            .unwrap()
            .len(),
        12
    );
}

#[cfg(feature = "imgref")]
#[test]
fn typed_buffers_move_into_imgvec() {
    use imgref::ImgVec;
    use rgb::Rgb;

    let buffer = PixelBuffer::<Rgb<u16>>::new_typed(W, H);
    let ptr = buffer.as_slice().as_strided_bytes().as_ptr();
    let img = ImgVec::try_from(buffer).unwrap();
    assert_eq!(img.buf().as_ptr().cast(), ptr);
    assert_eq!((img.width(), img.height()), (W as usize, H as usize));

    // `from_pixels` moves too, in both directions.
    let pixels = vec![Rgb::new(1u16, 2, 3); (W * H) as usize];
    let ptr = pixels.as_ptr();
    let buffer = PixelBuffer::from_pixels(pixels, W, H).unwrap();
    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
    let back = ImgVec::try_from(buffer).unwrap();
    assert_eq!(back.buf().as_ptr(), ptr);
}

#[test]
fn slices_copy_to_tight_typed_buffers() {
    use zenpixels::{Cicp, ColorContext};
    let context = std::sync::Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    // Padded rows with non-zero padding, in three sample types.
    for (d, sample) in [
        (PixelDescriptor::RGB8_SRGB, 1usize),
        (PixelDescriptor::RGB16_SRGB, 2),
        (PixelDescriptor::RGBAF32_LINEAR, 4),
    ] {
        let row = 5 * d.bytes_per_pixel();
        let stride = row + d.bytes_per_pixel();
        let mut source = PixelBuffer::new((stride / d.bytes_per_pixel()) as u32, 3, d).into_parts();
        source.width = 5;
        source.color_context = Some(context.clone());
        let mut source = PixelBuffer::try_from_parts(source).unwrap();
        source.as_slice_mut().as_strided_bytes_mut().fill(0xEE);
        for y in 0..3 {
            source.as_slice_mut().row_mut(y).fill(y as u8 + 1);
        }
        let copy = source.as_slice().try_to_tight_buffer().unwrap();
        assert_eq!(copy.stride(), row);
        assert_eq!(copy.descriptor(), d);
        assert!(std::sync::Arc::ptr_eq(
            copy.color_context().unwrap(),
            &context
        ));
        for y in 0..3 {
            assert_eq!(copy.as_slice().row(y), source.as_slice().row(y));
        }
        let parts = copy.into_parts();
        assert_eq!(parts.offset, 0);
        assert_eq!(parts.data.as_bytes().len(), row * 3);
        match (parts.data, sample) {
            (PixelStorage::U8(_), 1) | (PixelStorage::U16(_), 2) | (PixelStorage::F32(_), 4) => {}
            (other, _) => panic!("{d:?}: {other:?}"),
        }
    }
    // Empty.
    let empty = PixelBuffer::new(0, 0, PixelDescriptor::RGB16_SRGB);
    let copy = empty.as_slice().try_to_tight_buffer().unwrap();
    assert_eq!((copy.width(), copy.height()), (0, 0));
}

/// Empty views are valid whatever their stride, and may have no bytes at any row start
/// and no alignment for a wider sample.
#[test]
fn empty_views_copy_to_empty_buffers() {
    use zenpixels::{PixelDescriptor, PixelSlice};
    for (width, rows, stride, descriptor) in [
        (0, 2, 1, PixelDescriptor::GRAY8),
        (0, 1, 0, PixelDescriptor::GRAY16),
        (0, 3, 64, PixelDescriptor::RGBAF32),
        (0, 0, 0, PixelDescriptor::GRAY16),
    ] {
        let view = PixelSlice::new(&[], width, rows, stride, descriptor).unwrap();
        let copy = view.try_to_tight_buffer().unwrap();
        assert_eq!((copy.width(), copy.height()), (width, rows));
        assert_eq!(copy.descriptor(), descriptor);
    }
    // A width with no rows, over bytes that wouldn't hold a row.
    let view = PixelSlice::new(&[], 4, 0, 16, PixelDescriptor::RGBAF32).unwrap();
    let copy = view.try_to_tight_buffer().unwrap();
    assert_eq!((copy.width(), copy.height()), (4, 0));
}
