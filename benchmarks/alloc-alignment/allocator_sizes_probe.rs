//! Nightly prototype (not crate code): size_of Vec with ZST, enum and dyn allocators.
//! `rustup run nightly-2026-07-13 rustc -O --edition 2024 allocator_sizes_probe.rs`
#![feature(allocator_api)]
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct Align64;
fn widen(l: Layout) -> Layout { l.align_to(64).unwrap() }
unsafe impl Allocator for Align64 {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> { Global.allocate(widen(l)) }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) { unsafe { Global.deallocate(p, widen(l)) } }
}

/// A closed set of ZST allocators behind one 1-byte tag.
#[derive(Clone, Copy)]
enum PlaneAlloc { Global, Align64 }
unsafe impl Allocator for PlaneAlloc {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self { Self::Global => Global.allocate(l), Self::Align64 => Align64.allocate(l) }
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        unsafe { match self { Self::Global => Global.deallocate(p, l), Self::Align64 => Align64.deallocate(p, l) } }
    }
}

fn main() {
    println!("Vec<u8>                                  {}", size_of::<Vec<u8>>());
    println!("Vec<u8, Global>                          {}", size_of::<Vec<u8, Global>>());
    println!("Vec<u8, Align64>                         {}", size_of::<Vec<u8, Align64>>());
    println!("Vec<u8, PlaneAlloc> (enum of ZSTs)       {}", size_of::<Vec<u8, PlaneAlloc>>());
    println!("Vec<u8, &'static dyn Allocator>          {}", size_of::<Vec<u8, &'static dyn Allocator>>());
    println!("Vec<u8, Arc<dyn Allocator+Send+Sync>>    {}", size_of::<Vec<u8, Arc<dyn Allocator + Send + Sync>>>());
    println!("Box<dyn Fn()> (a box of a trait object)   {}", size_of::<Box<dyn Fn()>>());
    let mut v: Vec<u16, PlaneAlloc> = Vec::with_capacity_in(1 << 20, PlaneAlloc::Align64);
    v.resize(1 << 20, 1);
    println!("PlaneAlloc::Align64 1 MiB vec %64 = {}", v.as_ptr() as usize % 64);
}
