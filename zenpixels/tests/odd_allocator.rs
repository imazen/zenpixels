//! Integration test with a custom `GlobalAlloc` that deliberately returns odd
//! addresses (`addr % 2 == 1`) for every `align == 1` allocation (`Vec<u8>`).
//!
//! Proves that `zenpixels 0.5`:
//! - Never assumes `Vec<u8>` is 2/4/8-byte aligned.
//! - Adopts `U8`/`I8` `Vec<u8>` zero-copy even at odd addresses.
//! - Safely copies misaligned `Vec<u8>` into native-aligned storage for wider
//!   samples (`U16`, `F16`, `F32`, `F64`) with first byte = first pixel.
//! - Never violates alignment on `PixelBuf::new`, `new_aligned`, or
//!   `from_parts` + `into_contiguous`.

use std::alloc::{GlobalAlloc, Layout, System};
use zenpixels::{
    ColorSpace, F16Sample, ImageSpec, PixelBuf, PixelStorage, RowAlign, SampleLayout, SampleType,
};

struct OddBytes;

#[global_allocator]
static ALLOC: OddBytes = OddBytes;

// SAFETY: Every align==1 allocation reserves 1 extra leading byte and returns
// `base + 1` (always odd because `System` returns >=2-byte aligned pointers for
// size > 0). Dealloc/realloc adjust back by 1 byte with the exact original Layout.
unsafe impl GlobalAlloc for OddBytes {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() == 1 && layout.size() > 0 {
            let real = Layout::from_size_align(layout.size() + 1, 2).unwrap();
            let base = unsafe { System.alloc(real) };
            if base.is_null() {
                return base;
            }
            unsafe { base.add(1) }
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() == 1 && layout.size() > 0 {
            let real = Layout::from_size_align(layout.size() + 1, 2).unwrap();
            unsafe { System.dealloc(ptr.sub(1), real) };
        } else {
            unsafe { System.dealloc(ptr, layout) };
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() == 1 && layout.size() > 0 {
            let real = Layout::from_size_align(layout.size() + 1, 2).unwrap();
            let base = unsafe { System.alloc_zeroed(real) };
            if base.is_null() {
                return base;
            }
            unsafe { base.add(1) }
        } else {
            unsafe { System.alloc_zeroed(layout) }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() == 1 && layout.size() > 0 && new_size > 0 {
            let real = Layout::from_size_align(layout.size() + 1, 2).unwrap();
            let base = unsafe { System.realloc(ptr.sub(1), real, new_size + 1) };
            if base.is_null() {
                return base;
            }
            unsafe { base.add(1) }
        } else {
            // Fall back to alloc + copy + dealloc when transitioning zero <-> non-zero.
            let new_layout = Layout::from_size_align(new_size, layout.align()).unwrap();
            let new_ptr = unsafe { self.alloc(new_layout) };
            if !new_ptr.is_null() {
                let copy = layout.size().min(new_size);
                unsafe {
                    core::ptr::copy_nonoverlapping(ptr, new_ptr, copy);
                    self.dealloc(ptr, layout);
                }
            }
            new_ptr
        }
    }
}

#[test]
fn odd_allocator_actually_returns_odd_u8_pointers() {
    #[allow(clippy::useless_vec)]
    let v = vec![0xAAu8; 32];
    assert_eq!(
        (v.as_ptr() as usize) % 2,
        1,
        "expected odd address from OddBytes"
    );
}

#[test]
fn from_bytes_u8_adopts_zero_copy_at_odd_address() {
    let raw = vec![10u8, 20, 30, 40, 50, 60];
    let in_ptr = raw.as_ptr();
    assert_eq!((in_ptr as usize) % 2, 1);

    let buf = PixelBuf::from_bytes(raw, 2, 1, ImageSpec::rgb8_srgb()).unwrap();
    assert_eq!(buf.as_view().row(0).as_ptr(), in_ptr);
    assert_eq!(buf.as_view().row(0), &[10, 20, 30, 40, 50, 60]);
}

#[test]
fn from_bytes_u16_copies_safely_preserving_first_byte_equals_first_pixel() {
    let samples = [0x1122u16, 0x3344, 0x5566, 0x7788, 0x99AA, 0xBBCC];
    let mut raw = Vec::with_capacity(12);
    for s in samples {
        raw.extend_from_slice(&s.to_ne_bytes());
    }
    assert_eq!((raw.as_ptr() as usize) % 2, 1);

    // Exact-length odd input
    let buf = PixelBuf::from_bytes(raw.clone(), 2, 1, ImageSpec::rgb16_srgb()).unwrap();
    let view = buf.as_view();
    assert_eq!((view.row(0).as_ptr() as usize) % 2, 0);
    assert_eq!(view.row_as::<u16>(0).unwrap(), &samples);

    // Oversized odd input (extra trailing bytes)
    let mut padded = raw;
    padded.extend_from_slice(&[0xFF; 7]);
    assert_eq!((padded.as_ptr() as usize) % 2, 1);
    let buf2 = PixelBuf::from_bytes(padded, 2, 1, ImageSpec::rgb16_srgb()).unwrap();
    assert_eq!(buf2.as_view().row_as::<u16>(0).unwrap(), &samples);
}

#[test]
fn from_bytes_f16_and_f32_and_f64_copy_safely_from_odd_address() {
    // F16
    let h = [F16Sample::ONE, F16Sample::from_f32(0.5), F16Sample::ZERO];
    let raw_f16: Vec<u8> = bytemuck::cast_slice(&h).to_vec();
    assert_eq!((raw_f16.as_ptr() as usize) % 2, 1);
    let buf_f16 = PixelBuf::from_bytes(raw_f16, 1, 1, ImageSpec::rgbf16_linear()).unwrap();
    assert_eq!(buf_f16.as_view().row_as::<F16Sample>(0).unwrap(), &h);

    // F32
    let f = [0.25f32, 0.5, 0.75];
    let raw_f32: Vec<u8> = bytemuck::cast_slice(&f).to_vec();
    assert_eq!((raw_f32.as_ptr() as usize) % 2, 1);
    let buf_f32 = PixelBuf::from_bytes(raw_f32, 1, 1, ImageSpec::rgbf32_linear()).unwrap();
    assert_eq!((buf_f32.as_view().row(0).as_ptr() as usize) % 4, 0);
    assert_eq!(buf_f32.as_view().row_as::<f32>(0).unwrap(), &f);

    // F64 untagged
    let d = [1.5f64, -2.25f64];
    let raw_f64: Vec<u8> = bytemuck::cast_slice(&d).to_vec();
    assert_eq!((raw_f64.as_ptr() as usize) % 2, 1);
    let spec_f64 = ImageSpec::new(
        SampleLayout::new(SampleType::F64, zenpixels::ChannelOrder::Gray),
        ColorSpace::Untagged {
            channels: 1,
            alpha: zenpixels::AlphaMode::None,
        },
    );
    let buf_f64 = PixelBuf::from_bytes(raw_f64, 2, 1, spec_f64).unwrap();
    assert_eq!(
        (buf_f64.as_view().row(0).as_ptr() as usize) % core::mem::align_of::<f64>(),
        0
    );
    assert_eq!(buf_f64.as_view().row_as::<f64>(0).unwrap(), &d);
}

#[test]
fn allocating_constructors_and_into_contiguous_preserve_alignment() {
    for align in [
        RowAlign::Tight,
        RowAlign::Cache64,
        RowAlign::PixelCache64,
        RowAlign::Samples(8),
    ] {
        let buf = PixelBuf::new_aligned(7, 3, ImageSpec::rgbf32_linear(), align).unwrap();
        let view = buf.as_view();
        for y in 0..3 {
            let ptr = view.row(y).as_ptr() as usize;
            assert_eq!(ptr % 4, 0);
            if matches!(align, RowAlign::Cache64 | RowAlign::PixelCache64) {
                assert_eq!(ptr % 64, 0);
            }
            assert_eq!(view.row_as::<f32>(y).unwrap().len(), 21);
        }
    }

    // Odd Vec<u8> adopted via from_parts at offset=1 (even pointer) must stay aligned after into_contiguous!
    let mut backing = vec![0u8; 1 + 2 * 10]; // offset=1, stride=10, row_bytes=6 (2 rows of 1 RGB16 pixel)
    assert_eq!((backing.as_ptr() as usize) % 2, 1);
    let row0 = bytemuck::cast_slice::<u16, u8>(&[100u16, 200, 300]);
    let row1 = bytemuck::cast_slice::<u16, u8>(&[400u16, 500, 600]);
    backing[1..7].copy_from_slice(row0);
    backing[11..17].copy_from_slice(row1);

    let buf = PixelBuf::from_storage_parts(
        PixelStorage::U8(backing),
        1,
        1,
        2,
        10,
        ImageSpec::rgb16_srgb(),
    )
    .unwrap();
    let tight = buf.into_contiguous();
    assert_eq!(tight.offset(), 1, "cannot shift to odd offset 0 for U16");
    assert_eq!(tight.as_view().row_as::<u16>(0).unwrap(), &[100, 200, 300]);
    assert_eq!(tight.as_view().row_as::<u16>(1).unwrap(), &[400, 500, 600]);
}
