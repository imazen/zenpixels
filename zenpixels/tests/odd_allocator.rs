//! A `Vec<u8>` has no alignment guarantee beyond 1. This allocator returns odd
//! addresses for byte allocations, which is legal, and which glibc never does, so it
//! exposes code that quietly assumes a `Vec<u8>` is aligned for wider samples.

use std::alloc::{GlobalAlloc, Layout, System};
use zenpixels::{PixelBuffer, PixelDescriptor};

struct OddBytes;

fn widened(layout: Layout) -> Layout {
    Layout::from_size_align(layout.size() + 16, 16).unwrap()
}

// SAFETY: byte allocations come from a 16-aligned block offset by one, and dealloc
// undoes the same offset with the same layout. Others go straight to System.
unsafe impl GlobalAlloc for OddBytes {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() == 1 {
            let base = unsafe { System.alloc(widened(layout)) };
            if base.is_null() {
                base
            } else {
                unsafe { base.add(1) }
            }
        } else {
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() == 1 {
            unsafe { System.dealloc(ptr.sub(1), widened(layout)) }
        } else {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}

#[global_allocator]
static ODD: OddBytes = OddBytes;

/// Samples `0, 1, 2, ...` as native-endian bytes in a `Vec<u8>`, at an odd address.
fn odd_u16_bytes(samples: usize, extra_bytes: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples * 2 + extra_bytes);
    for i in 0..samples as u16 {
        bytes.extend_from_slice(&i.to_ne_bytes());
    }
    bytes.resize(samples * 2 + extra_bytes, 0xEE);
    assert_eq!(
        bytes.as_ptr() as usize % 2,
        1,
        "the test needs an odd address"
    );
    bytes
}

fn assert_rgb16_counts_up(buffer: &PixelBuffer) {
    let slice = buffer.as_slice();
    let mut next = 0u16;
    for y in 0..slice.rows() {
        for sample in slice.row(y).chunks_exact(2) {
            assert_eq!(sample, next.to_ne_bytes(), "row {y}");
            next += 1;
        }
    }
}

/// An exact-length buffer used to fail with `InsufficientData`.
#[test]
fn from_vec_accepts_an_exact_length_misaligned_vec() {
    let bytes = odd_u16_bytes(3 * 4 * 2, 0);
    let buffer = PixelBuffer::from_vec(bytes, 4, 2, PixelDescriptor::RGB16_SRGB).unwrap();
    assert_rgb16_counts_up(&buffer);
}

/// A longer buffer used to be accepted with every sample read one byte late.
#[test]
fn from_vec_reads_the_first_byte_as_the_first_pixel() {
    let bytes = odd_u16_bytes(3 * 4 * 2, 9);
    let buffer = PixelBuffer::from_vec(bytes, 4, 2, PixelDescriptor::RGB16_SRGB).unwrap();
    assert_rgb16_counts_up(&buffer);
}

/// Allocating constructors don't depend on what a byte allocation returns.
#[test]
fn allocating_constructors_are_aligned_for_their_samples() {
    for d in [
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGBAF32_LINEAR,
        PixelDescriptor::GRAY16_SRGB,
    ] {
        let buffer = PixelBuffer::new(5, 3, d);
        let first = buffer.as_slice().as_strided_bytes().as_ptr() as usize;
        assert_eq!(first % d.min_alignment(), 0, "{d:?}");
        assert_eq!(buffer.into_parts().offset, 0);
    }
}

/// `Vec<u8>` still holds byte formats without moving, at any address.
#[test]
fn byte_formats_adopt_at_an_odd_address() {
    let mut bytes = Vec::with_capacity(12);
    bytes.resize(12, 3u8);
    assert_eq!(bytes.as_ptr() as usize % 2, 1);
    let ptr = bytes.as_ptr();
    let buffer = PixelBuffer::from_vec(bytes, 2, 2, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr(), ptr);
}
