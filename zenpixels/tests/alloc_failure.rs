//! Deterministic allocation-failure injection for the fallible constructors
//! and copy paths audited under imazen/zenpixels#151.
//!
//! A `#[global_allocator]` returns null for every allocation larger than a
//! run-time threshold, so each `try_*` path can be pushed over its own reserve
//! without depending on host memory pressure. This file is its own test
//! binary, so the allocator affects only these tests; every test that lowers
//! the threshold holds `FAIL_LOCK` so a neighbor's big allocation can't fail
//! spuriously.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use zenpixels::plane::{AlignedSamples, PlaneError};
use zenpixels::{BufferError, PixelBuffer, PixelDescriptor, PixelSlice};

/// Allocations larger than this many bytes fail; `usize::MAX` means never.
static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Serializes threshold manipulation across tests in this binary.
static FAIL_LOCK: Mutex<()> = Mutex::new(());

struct FailOver;

// SAFETY: `alloc`/`alloc_zeroed`/`realloc` return null past the threshold
// (callers observe failure, the contract's only requirement); smaller
// allocations and all `dealloc`s delegate to `System` with the same layout.
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

/// Restore the threshold on drop, even when a test assertion panics while it
/// is lowered.
struct ThresholdGuard;

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        THRESHOLD.store(usize::MAX, Ordering::Relaxed);
    }
}

/// Fail every allocation above `bytes` while `body` runs, then restore.
fn failing_above<T>(bytes: usize, body: impl FnOnce() -> T) -> T {
    THRESHOLD.store(bytes, Ordering::Relaxed);
    let _guard = ThresholdGuard;
    body()
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    FAIL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn assert_alloc_failed(err: &whereat::At<BufferError>) {
    assert_eq!(
        *err.error(),
        BufferError::AllocationFailed,
        "expected AllocationFailed, got {err:?}"
    );
}

const RGB8: PixelDescriptor = PixelDescriptor::RGB8_SRGB;

/// 64x32 RGB8 with stride 256 (rows are 192 bytes): the tight copies below are
/// 6 KiB, over the 4 KiB gate.
fn padded() -> (Vec<u8>, usize, u32, u32) {
    (vec![0xABu8; 256 * 32], 256, 64, 32)
}

#[test]
fn try_new_reports_allocation_failure_and_recovers() {
    let _g = lock();
    // 64x64 RGBA8 = 16 KiB, over the 4 KiB gate.
    failing_above(4096, || {
        let err = PixelBuffer::try_new(64, 64, PixelDescriptor::RGBA8_SRGB).unwrap_err();
        assert_alloc_failed(&err);
    });
    // The allocator restored, the identical call succeeds — refusal leaks nothing.
    let buf = PixelBuffer::try_new(64, 64, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert_eq!(buf.width(), 64);
}

#[test]
fn try_new_aligned_reports_allocation_failure() {
    let _g = lock();
    failing_above(4096, || {
        let err = PixelBuffer::<(), AlignedSamples<u16>>::try_new_aligned(
            64,
            64,
            PixelDescriptor::RGB16_SRGB,
        )
        .unwrap_err();
        assert_alloc_failed(&err);
    });
}

#[test]
fn try_new_reports_overflow_not_allocation() {
    let _g = lock();
    // stride * height overflows usize on every target — a dimension error,
    // not an allocation the host may or may not satisfy.
    let err = PixelBuffer::try_new(u32::MAX, u32::MAX, RGB8).unwrap_err();
    assert_eq!(*err.error(), BufferError::InvalidDimensions);
}

#[test]
fn zero_extents_allocate_nothing() {
    let _g = lock();
    // Even with every allocation above 4 KiB refused, empty buffers succeed.
    let buf = failing_above(4096, || PixelBuffer::try_new(0, 0, RGB8).unwrap());
    assert_eq!(buf.width(), 0);
}

#[test]
fn try_to_tight_buffer_reports_allocation_failure() {
    let _g = lock();
    let (data, stride, w, h) = padded();
    let slice = PixelSlice::new(&data, w, h, stride, RGB8).unwrap();
    failing_above(4096, || {
        let err = slice.try_to_tight_buffer().unwrap_err();
        assert_alloc_failed(&err);
    });
    // Borrowed input is untouched: the same call succeeds after refusal.
    let tight = slice.try_to_tight_buffer().unwrap();
    assert_eq!(tight.as_slice().row(3), &data[3 * 256..3 * 256 + 192]);
}

#[test]
fn try_to_tight_buffer_on_short_final_row() {
    let _g = lock();
    // A view whose backing slice ends inside the last stride's padding:
    // 64x32 RGB8, stride 256, data covering (32-1)*256 + 192 = 8128 bytes.
    let data = vec![7u8; 31 * 256 + 192];
    let slice = PixelSlice::new(&data, 64, 32, 256, RGB8).unwrap();
    failing_above(4096, || {
        let err = slice.try_to_tight_buffer().unwrap_err();
        assert_alloc_failed(&err);
    });
    let tight = slice.try_to_tight_buffer().unwrap();
    assert_eq!(tight.as_slice().row(31), &data[31 * 256..31 * 256 + 192]);
}

#[test]
fn try_contiguous_bytes_reports_allocation_failure() {
    let _g = lock();
    let (data, stride, w, h) = padded();
    let slice = PixelSlice::new(&data, w, h, stride, RGB8).unwrap();
    failing_above(4096, || {
        let err = slice.try_contiguous_bytes().unwrap_err();
        assert_alloc_failed(&err);
    });
    assert!(slice.try_contiguous_bytes().is_ok());
}

#[test]
fn try_contiguous_bytes_stays_borrowed_when_tight() {
    let _g = lock();
    let data = vec![0x55u8; 64 * 3 * 4];
    let slice = PixelSlice::new(&data, 64, 4, 64 * 3, RGB8).unwrap();
    // Tight rows borrow — no allocation happens even with every big one refused.
    let cow = failing_above(4096, || slice.try_contiguous_bytes().unwrap());
    assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn try_copy_to_contiguous_bytes_reports_allocation_failure() {
    let _g = lock();
    let buf = PixelBuffer::try_new(64, 64, RGB8).unwrap();
    failing_above(4096, || {
        let err = buf.try_copy_to_contiguous_bytes().unwrap_err();
        assert_alloc_failed(&err);
    });
    assert_eq!(
        buf.try_copy_to_contiguous_bytes().unwrap().len(),
        64 * 64 * 3
    );
}

#[test]
fn crop_copy_reports_allocation_failure() {
    let _g = lock();
    let buf = PixelBuffer::try_new(96, 96, RGB8).unwrap();
    failing_above(4096, || {
        let err = buf.crop_copy(0, 0, 64, 64).unwrap_err();
        assert_alloc_failed(&err);
    });
    assert_eq!(buf.crop_copy(0, 0, 64, 64).unwrap().width(), 64);
}

#[test]
fn aligned_samples_try_zeroed_reports_allocation_failure() {
    let _g = lock();
    failing_above(4096, || {
        let err = AlignedSamples::<u16>::try_zeroed(64 * 64 * 3).unwrap_err();
        assert_eq!(err, PlaneError::AllocationFailed);
    });
    assert!(AlignedSamples::<u16>::try_zeroed(64 * 64 * 3).is_ok());
}
