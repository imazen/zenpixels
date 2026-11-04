//! Packed buffers over any sample storage, on stable. Pointer identity, heap
//! allocations during the conversion, and pixels are all checked.
use image::{ImageBuffer, Rgb};
use std::alloc::{GlobalAlloc, Layout, System};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use zenpixels::plane::{SampleStorage, SampleStorageMut};
use zenpixels::{PixelBuffer, PixelBufferParts, PixelDescriptor};
struct C;
static N: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for C {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        N.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, s: usize) -> *mut u8 {
        N.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, s) }
    }
}
#[global_allocator]
static G: C = C;
fn counted<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let b = N.load(Ordering::Relaxed);
    let r = std::hint::black_box(f());
    (r, N.load(Ordering::Relaxed) - b)
}
fn report(n: &str, same: bool, a: usize) {
    println!(
        "{n:<62} {:<9} allocs={a}",
        if same { "same-ptr" } else { "COPIED" }
    );
    assert!(same && a == 0, "{n}");
}
const W: u32 = 37;
const H: u32 = 11;
struct Pool {
    id: u32,
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
fn main() {
    let n = (W * H * 3) as usize;
    let (w, us) = (W, W as usize);
    // u16: was a forced copy with Vec<u8> storage.
    let raw: Vec<u16> = (0..n).map(|i| i as u16 * 3).collect();
    let img = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w, H, raw).unwrap();
    let p0 = img.as_raw().as_ptr();
    let (pb, a) = counted(|| {
        PixelBuffer::try_from_parts(PixelBufferParts::new(
            img.into_raw(),
            w,
            H,
            us * 6,
            PixelDescriptor::RGB16_SRGB,
        ))
        .unwrap()
    });
    report(
        "move ImageBuffer<Rgb<u16>, Vec<u16>> -> PixelBuffer",
        pb.as_slice().as_strided_bytes().as_ptr() as *const u16 == p0,
        a,
    );
    assert_eq!(
        &pb.as_slice().row(1)[..6],
        bytemuck::cast_slice::<u16, u8>(&[333u16, 336, 339])
    );
    let (back, a) = counted(|| {
        let p = pb.into_parts();
        ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w, H, p.data).unwrap()
    });
    report(
        "move PixelBuffer<RGB16, Vec<u16>> -> ImageBuffer<Rgb<u16>>",
        back.as_raw().as_ptr() == p0,
        a,
    );
    // f32.
    let img = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(w, H);
    let p0 = img.as_raw().as_ptr();
    let (pb, a) = counted(|| {
        PixelBuffer::try_from_parts(PixelBufferParts::new(
            img.into_raw(),
            w,
            H,
            us * 12,
            PixelDescriptor::RGBF32_LINEAR,
        ))
        .unwrap()
    });
    report(
        "move ImageBuffer<Rgb<f32>, Vec<f32>> -> PixelBuffer",
        pb.as_slice().as_strided_bytes().as_ptr() as *const f32 == p0,
        a,
    );
    // Strided, with an offset, mutable through the owner.
    let stride_samples = 126usize;
    let start = 4usize;
    let mut raw16 = vec![0u16; start + stride_samples * (H as usize - 1) + us * 3];
    let q0 = raw16[start..].as_ptr();
    raw16[start + stride_samples] = 9;
    let mut parts =
        PixelBufferParts::new(raw16, w, H, stride_samples * 2, PixelDescriptor::RGB16_SRGB);
    parts.offset = start * 2;
    let (mut pb, a) = counted(|| PixelBuffer::try_from_parts(parts).unwrap());
    report(
        "adopt strided Vec<u16> with offset",
        pb.as_slice().as_strided_bytes().as_ptr() as *const u16 == q0,
        a,
    );
    assert_eq!(&pb.as_slice().row(1)[..2], &9u16.to_ne_bytes());
    let (s, a) = counted(|| {
        let m = pb.as_slice_mut();
        m.stride()
    });
    report("as_slice_mut over Vec<u16>", s == stride_samples * 2, a);
    // Box, borrowed slices, custom owner.
    let b: Box<[u8]> = vec![5u8; n].into_boxed_slice();
    let b0 = b.as_ptr();
    let (pb, a) = counted(|| {
        PixelBuffer::try_from_parts(PixelBufferParts::new(
            b,
            w,
            H,
            us * 3,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap()
    });
    report(
        "adopt Box<[u8]>",
        pb.as_slice().as_strided_bytes().as_ptr() == b0,
        a,
    );
    let v = vec![1u8; n];
    let (pb, a) = counted(|| {
        PixelBuffer::try_from_parts(PixelBufferParts::new(
            &v[..],
            w,
            H,
            us * 3,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap()
    });
    report(
        "adopt &[u8]",
        pb.as_slice().as_strided_bytes().as_ptr() == v.as_ptr(),
        a,
    );
    let pool = Pool {
        id: 7,
        buf: vec![2u16; n],
    };
    let b0 = pool.buf.as_ptr();
    let (pb, a) = counted(|| {
        PixelBuffer::try_from_parts(PixelBufferParts::new(
            pool,
            w,
            H,
            us * 6,
            PixelDescriptor::RGB16_SRGB,
        ))
        .unwrap()
    });
    report(
        "adopt custom owner (pool slot)",
        pb.as_slice().as_strided_bytes().as_ptr() as *const u16 == b0,
        a,
    );
    let p = pb.into_parts();
    assert_eq!(p.data.id, 7);
    // The image feature's conversions.
    let img = ImageBuffer::<Rgb<u16>, Vec<u16>>::new(w, H);
    let p0 = img.as_raw().as_ptr();
    let (pb, a) = counted(|| PixelBuffer::try_from(img).unwrap());
    report(
        "TryFrom ImageBuffer<Rgb<u16>, Vec<u16>> -> PixelBuffer",
        pb.as_slice().as_strided_bytes().as_ptr() as *const u16 == p0,
        a,
    );
    let (img, a) = counted(|| ImageBuffer::<Rgb<u16>, Vec<u16>>::try_from(pb).unwrap());
    report(
        "TryFrom PixelBuffer -> ImageBuffer<Rgb<u16>, Vec<u16>>",
        img.as_raw().as_ptr() == p0,
        a,
    );
    let (s, a) = counted(|| zenpixels::PixelSlice::try_from(&img).unwrap());
    report(
        "TryFrom &ImageBuffer<Rgb<u16>> -> PixelSlice",
        s.as_strided_bytes().as_ptr() as *const u16 == p0,
        a,
    );
    // Aligned allocation, viewed as its pixel type with no alignment failure.
    let (aligned, a) = counted(|| {
        PixelBuffer::<(), zenpixels::plane::AlignedSamples<u8>>::try_new_aligned(
            w,
            H,
            PixelDescriptor::RGB16_SRGB,
        )
        .unwrap()
    });
    let first = aligned.as_slice().as_strided_bytes().as_ptr() as usize;
    println!(
        "    try_new_aligned RGB16: addr%64={} stride={} allocs={a}",
        first % 64,
        aligned.stride()
    );
    let slice = aligned.as_slice();
    let row: &[[u16; 3]] = bytemuck::cast_slice(slice.row(H - 1));
    println!("    last row as &[[u16; 3]]: {} pixels", row.len());
    // Misaligned u8 storage for u16 samples still refused; rejected parts come back.
    let bytes = vec![0u8; n * 2 + 2];
    let odd = if bytes.as_ptr() as usize % 2 == 0 {
        1
    } else {
        0
    };
    let mut parts = PixelBufferParts::new(bytes, w, H, us * 6, PixelDescriptor::RGB16_SRGB);
    parts.offset = odd;
    let mut e = PixelBuffer::try_from_parts(parts).unwrap_err();
    println!(
        "    misaligned u8 storage for RGB16: refused, parts_returned={}",
        e.take_parts().is_some()
    );
    println!(
        "    size_of PixelBuffer: Vec<u8>={} Vec<u16>={} Box<[u8]>={} &[u8]={}",
        size_of::<PixelBuffer>(),
        size_of::<PixelBuffer<(), Vec<u16>>>(),
        size_of::<PixelBuffer<(), Box<[u8]>>>(),
        size_of::<PixelBuffer<(), &[u8]>>()
    );
}
