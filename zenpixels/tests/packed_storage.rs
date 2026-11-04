//! `PixelBuffer` over storage other than `Vec<u8>`: nothing is copied, and
//! layout validation matches the `Vec<u8>` path.

use zenpixels::plane::{SampleStorage, SampleStorageMut};
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
fn default_storage_type_is_unchanged() {
    fn takes_default(_: PixelBuffer) {}
    takes_default(PixelBuffer::new(1, 1, PixelDescriptor::RGB8_SRGB));
    #[cfg(feature = "rgb")]
    {
        fn takes_typed(_: PixelBuffer<rgb::Rgb<u8>>) {}
        takes_typed(PixelBuffer::new_typed(1, 1));
    }
    let parts: PixelBufferParts = PixelBuffer::new(1, 1, PixelDescriptor::RGB8_SRGB).into_parts();
    assert_eq!(parts.data.len(), 3);
}
