//! Allocator-backed storage, on Rust 1.100 or newer: planes and packed buffers
//! hold `Vec<T, Align64>` directly, and it moves on to `image`.
use image::{ImageBuffer, Rgb};
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::mem::size_of;
use std::ptr::NonNull;
use zenpixels::plane::Plane;
use zenpixels::{PixelBuffer, PixelBufferParts, PixelDescriptor};

#[derive(Clone, Copy, Default)]
struct Align64;
fn up(l: Layout) -> Result<Layout, AllocError> {
    l.align_to(64).map_err(|_| AllocError)
}
unsafe impl Allocator for Align64 {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(up(l)?)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        unsafe { Global.deallocate(p, up(l).unwrap()) }
    }
}
fn aligned<T: Copy + Default>(n: usize) -> Vec<T, Align64> {
    let mut v = Vec::with_capacity_in(n, Align64);
    v.resize(n, T::default());
    v
}
fn main() {
    const W: u32 = 37;
    const H: u32 = 11;
    let n = (W * H * 3) as usize;
    let d = |c: u8| match c {
        8 => PixelDescriptor::RGB8_SRGB,
        16 => PixelDescriptor::RGB16_SRGB,
        _ => PixelDescriptor::RGBF32_LINEAR,
    };

    let v8 = aligned::<u8>(n);
    let p8 = v8.as_ptr();
    let parts = PixelBufferParts::new(v8, W, H, W as usize * 3, d(8));
    let pb = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr(), p8);
    println!(
        "Vec<u8, Align64> adopted: offset=0 addr%64={}",
        p8 as usize % 64
    );
    let img = ImageBuffer::<Rgb<u8>, Vec<u8, Align64>>::try_from(pb).unwrap();
    assert_eq!(img.as_raw().as_ptr(), p8);
    println!("-> ImageBuffer<Rgb<u8>, Vec<u8, Align64>>: same-ptr");

    let v16 = aligned::<u16>(n);
    let p16 = v16.as_ptr();
    let parts = PixelBufferParts::new(v16, W, H, W as usize * 6, d(16));
    let pb = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr().cast::<u16>(), p16);
    println!(
        "Vec<u16, Align64> adopted: offset=0 addr%64={}",
        p16 as usize % 64
    );

    let vf = aligned::<f32>(n);
    let pf = vf.as_ptr();
    let parts = PixelBufferParts::new(vf, W, H, W as usize * 12, d(32));
    let pb = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr().cast::<f32>(), pf);
    println!(
        "Vec<f32, Align64> adopted: offset=0 addr%64={}",
        pf as usize % 64
    );

    let plane = Plane::new(aligned::<u16>(16), 3, 4, 4).unwrap();
    let back: Vec<u16, Align64> = plane.try_into_vec().unwrap();
    println!(
        "Plane<Vec<u16, Align64>>::try_into_vec: addr%64={}",
        back.as_ptr() as usize % 64
    );
    println!(
        "size_of PixelBuffer: default={} Vec<u8, Align64>={}",
        size_of::<PixelBuffer>(),
        size_of::<PixelBuffer<(), Vec<u8, Align64>>>()
    );
}
