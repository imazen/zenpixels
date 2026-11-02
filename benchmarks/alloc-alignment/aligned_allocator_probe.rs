//! Nightly prototype (not crate code): a 64-byte aligning `Allocator` over `Global`,
//! as stabilized for Rust 1.100 (rust-lang/rust#156882). Checks that `Vec<T, Align64>`
//! starts on 64 bytes after new, clone, growth and try_reserve. Run:
//! `rustup run nightly-2026-07-13 rustc -O --edition 2024 aligned_allocator_probe.rs && ./aligned_allocator_probe`
//! Results: see README.md in this directory.
#![feature(allocator_api)]
// Prototype: a 64-byte aligning allocator over Global, as stabilized for 1.100.
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

#[derive(Clone, Copy, Default)]
struct Align64;

fn widen(layout: Layout) -> Layout {
    layout.align_to(64).expect("64 is a power of two and small")
}

// SAFETY: every block is allocated and freed by Global with the same widened
// layout, so deallocate and the default grow/shrink (which route through
// allocate/deallocate) always pass Global the layout it allocated with.
unsafe impl Allocator for Align64 {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(widen(layout))
    }
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate_zeroed(widen(layout))
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, widen(layout)) }
    }
}

fn main() {
    let mut bad = 0;
    for bytes in [64usize, 1000, 16 << 10, 128 << 10, 1 << 20, 32 << 20] {
        let mut v: Vec<u16, Align64> = Vec::with_capacity_in(bytes / 2, Align64);
        v.resize(bytes / 2, 7);
        let a = v.as_ptr() as usize % 64;
        let c = v.clone();
        let b = c.as_ptr() as usize % 64;
        v.extend(std::iter::repeat(1).take(bytes)); // grow (realloc path)
        let g = v.as_ptr() as usize % 64;
        let mut z: Vec<f32, Align64> = Vec::new_in(Align64);
        z.try_reserve_exact(bytes / 4).unwrap();
        let t = z.as_ptr() as usize % 64;
        println!("{bytes:>9} B: new %64={a} clone %64={b} after-grow %64={g} try_reserve %64={t}");
        bad += (a | b | g | t != 0) as u32;
    }
    println!("misaligned: {bad}");
}
