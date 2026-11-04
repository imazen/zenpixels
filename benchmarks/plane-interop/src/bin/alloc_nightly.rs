//! Allocator-backed packed buffers. Needs the allocator API (nightly here;
//! Rust 1.100 stable). A real impl in zenpixels would be `Vec<T, A>`; a newtype
//! stands in because the orphan rule refuses that impl from this crate.
#![feature(allocator_api)]
use image::{ImageBuffer, Rgb};
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::mem::size_of;
use std::ptr::NonNull;
use zenpixels::plane::{PlaneSample, SampleStorage, SampleStorageMut};
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
struct AV<T>(Vec<T, Align64>);
impl<T: PlaneSample> SampleStorage for AV<T> {
    type Sample = T;
    fn samples(&self) -> &[T] {
        &self.0
    }
}
impl<T: PlaneSample> SampleStorageMut for AV<T> {
    fn samples_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}
fn main() {
    const W: u32 = 37;
    const H: u32 = 11;
    let n = (W * H * 3) as usize;
    // Same parameter, three sample types, each row 0 at offset 0 and 64-aligned.
    let mut v8 = Vec::with_capacity_in(n, Align64);
    v8.resize(n, 3u8);
    let p8 = v8.as_ptr();
    let pb = PixelBuffer::try_from_parts(PixelBufferParts::new(
        AV(v8),
        W,
        H,
        W as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    ))
    .unwrap();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr(), p8);
    println!(
        "Vec<u8, Align64> adopted: offset=0 addr%64={}",
        p8 as usize % 64
    );
    // Hand over to image, whose Container only needs Deref<Target=[u8]>.
    let img =
        ImageBuffer::<Rgb<u8>, Vec<u8, Align64>>::from_raw(W, H, pb.into_parts().data.0).unwrap();
    assert_eq!(img.as_raw().as_ptr(), p8);
    println!("-> ImageBuffer<Rgb<u8>, Vec<u8, Align64>>: same-ptr");
    let mut v16 = Vec::with_capacity_in(n, Align64);
    v16.resize(n, 3u16);
    let p16 = v16.as_ptr();
    let pb = PixelBuffer::try_from_parts(PixelBufferParts::new(
        AV(v16),
        W,
        H,
        W as usize * 6,
        PixelDescriptor::RGB16_SRGB,
    ))
    .unwrap();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr() as *const u16, p16);
    println!(
        "Vec<u16, Align64> adopted: offset=0 addr%64={}",
        p16 as usize % 64
    );
    let mut vf = Vec::with_capacity_in(n, Align64);
    vf.resize(n, 0.5f32);
    let pf = vf.as_ptr();
    let mut pb = PixelBuffer::try_from_parts(PixelBufferParts::new(
        AV(vf),
        W,
        H,
        W as usize * 12,
        PixelDescriptor::RGBF32_LINEAR,
    ))
    .unwrap();
    pb.as_slice_mut();
    assert_eq!(pb.as_slice().as_strided_bytes().as_ptr() as *const f32, pf);
    println!(
        "Vec<f32, Align64> adopted: offset=0 addr%64={}",
        pf as usize % 64
    );
    println!(
        "size_of PixelBuffer: Vec<u8>={} Vec<u8,Align64>={}",
        size_of::<PixelBuffer>(),
        size_of::<PixelBuffer<(), AV<u8>>>()
    );
}
