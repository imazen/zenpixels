//! Allocation-failure injection for the typed `Pixel` extraction paths
//! (issue #151), gated on the `rgb` feature because
//! [`PixelBuffer::try_into_contiguous_pixels`] lives there.
//!
//! Same threshold-allocator approach as `alloc_failure.rs` — this file is its
//! own test binary, so its `#[global_allocator]` affects only these tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use zenpixels::{BufferError, PixelBuffer, PixelDescriptor};

static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
static FAIL_LOCK: Mutex<()> = Mutex::new(());

struct FailOver;

// SAFETY: allocations past the threshold return null; smaller ones and every
// deallocation delegate to `System` with the same layout.
unsafe impl GlobalAlloc for FailOver {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > THRESHOLD.load(Ordering::Relaxed) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > THRESHOLD.load(Ordering::Relaxed) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > THRESHOLD.load(Ordering::Relaxed) {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static FAIL_OVER: FailOver = FailOver;

struct ThresholdGuard;

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        THRESHOLD.store(usize::MAX, Ordering::Relaxed);
    }
}

fn failing_above<T>(bytes: usize, body: impl FnOnce() -> T) -> T {
    THRESHOLD.store(bytes, Ordering::Relaxed);
    let _guard = ThresholdGuard;
    body()
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    FAIL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Byte storage + an align-2 pixel forces the copy arm of
/// `try_into_contiguous_pixels`: `try_cast_vec::<u8, Rgb<u16>>` can never
/// increase alignment. The copy reserves a `Vec<Rgb<u16>>`.
#[test]
fn try_into_contiguous_pixels_reports_copy_allocation_failure() {
    let _g = lock();
    let buf =
        PixelBuffer::from_vec(vec![0u8; 64 * 32 * 6], 64, 32, PixelDescriptor::RGB16_SRGB).unwrap();
    failing_above(4096, || {
        let err = buf
            .try_into_contiguous_pixels::<rgb::Rgb<u16>>()
            .unwrap_err();
        assert_eq!(*err.error(), BufferError::AllocationFailed);
    });
    let pixels = PixelBuffer::from_vec(vec![0u8; 64 * 32 * 6], 64, 32, PixelDescriptor::RGB16_SRGB)
        .unwrap()
        .try_into_contiguous_pixels::<rgb::Rgb<u16>>()
        .unwrap()
        .unwrap();
    assert_eq!(pixels.len(), 64 * 32);
}

/// The same call on an incompatible descriptor refuses with `Ok(None)`, not
/// an allocation error.
#[test]
fn try_into_contiguous_pixels_refuses_wrong_layout() {
    let _g = lock();
    let buf =
        PixelBuffer::from_vec(vec![0u8; 4 * 4 * 3], 4, 4, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(matches!(
        buf.try_into_contiguous_pixels::<rgb::Rgb<u16>>(),
        Ok(None)
    ));
}
