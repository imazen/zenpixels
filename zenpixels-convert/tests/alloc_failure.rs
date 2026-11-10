//! Deterministic allocation-failure injection for the fallible conversion
//! paths audited under imazen/zenpixels#151.
//!
//! A `#[global_allocator]` refuses every allocation larger than a run-time
//! threshold, pushing each `Result`-returning path over its own reserve
//! without touching host memory pressure. This file is its own test binary,
//! so the allocator affects only these tests; every threshold change holds
//! `FAIL_LOCK` so a neighbor test's big allocation can't fail spuriously.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use zenpixels::{BufferError, PixelBuffer, PixelDescriptor, PixelSlice};
use zenpixels_convert::{
    ConvertError, ConvertPlan, PixelBufferConvertExt, PixelSliceConvertExt, RowConverter,
};

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

const RGB8: PixelDescriptor = PixelDescriptor::RGB8_SRGB;

/// Restore the threshold on drop, even when a test assertion panics while it
/// is lowered.
struct ThresholdGuard;

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        THRESHOLD.store(usize::MAX, Ordering::Relaxed);
    }
}

/// Fail every allocation above `threshold` bytes while `body` runs, then restore.
fn failing_above<T>(threshold: usize, body: impl FnOnce() -> T) -> T {
    THRESHOLD.store(threshold, Ordering::Relaxed);
    let _guard = ThresholdGuard;
    body()
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    FAIL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn assert_alloc_failed(err: &whereat::At<ConvertError>) {
    assert!(
        matches!(
            *err.error(),
            ConvertError::AllocationFailed | ConvertError::Buffer(BufferError::AllocationFailed)
        ),
        "expected an allocation failure, got {err:?}"
    );
}

/// A multi-step plan: RGBA8 → linear F32 RGB emits `DropAlpha` followed by
/// `SrgbU8ToLinearF32`, so row conversion needs the ping-pong scratch.
fn multi_op_plan() -> ConvertPlan {
    ConvertPlan::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBF32_LINEAR)
        .expect("plan RGBA8 -> RGBF32_LINEAR")
}

#[test]
fn scratch_prepare_reports_allocation_failure() {
    let _g = lock();
    let mut converter = RowConverter::from_plan(multi_op_plan());
    failing_above(4096, || {
        let err = converter.prepare(512).unwrap_err();
        assert_alloc_failed(&err);
    });
    // The allocator restored, the identical prepare succeeds.
    converter.prepare(512).unwrap();
}

#[test]
fn unprepared_row_conversion_reports_allocation_failure() {
    let _g = lock();
    let width = 256u32;
    let mut converter = RowConverter::from_plan(multi_op_plan());
    let src = vec![0x40u8; width as usize * 4];
    let mut dst = vec![0u8; width as usize * 12];
    failing_above(4096, || {
        let err = converter
            .try_convert_row(&src, &mut dst, width)
            .unwrap_err();
        assert_alloc_failed(&err);
    });
    // After refusal, the same conversion succeeds — the scratch reserve left
    // no half-charged state.
    converter.try_convert_row(&src, &mut dst, width).unwrap();
}

#[test]
fn convert_to_reports_allocation_failure() {
    let _g = lock();
    let data = vec![0x77u8; 64 * 64 * 3];
    let slice = PixelSlice::new(&data, 64, 64, 64 * 3, RGB8).unwrap();
    failing_above(4096, || {
        let err = slice.convert_to(PixelDescriptor::RGBA8_SRGB).unwrap_err();
        assert_alloc_failed(&err);
    });
    assert!(slice.convert_to(PixelDescriptor::RGBA8_SRGB).is_ok());
}

#[test]
fn into_oriented_reports_allocation_failure() {
    let _g = lock();
    // 400x300 non-square: the in-place transpose's `moved` bitvec needs
    // 120000 bits (15 KiB); the allocating fallback needs ~360 KiB.
    let buf = PixelBuffer::try_new(400, 300, RGB8).unwrap();
    failing_above(4096, || {
        let err = buf
            .into_oriented(zenpixels::Orientation::Transpose)
            .unwrap_err();
        assert_alloc_failed(&err);
    });
    let out = PixelBuffer::try_new(400, 300, RGB8)
        .unwrap()
        .into_oriented(zenpixels::Orientation::Transpose)
        .unwrap();
    assert_eq!((out.width(), out.height()), (300, 400));
}

#[test]
fn oriented_reports_allocation_failure() {
    let _g = lock();
    let buf = PixelBuffer::try_new(96, 48, RGB8).unwrap();
    let slice = buf.as_slice();
    failing_above(4096, || {
        let err = slice
            .oriented(zenpixels::Orientation::Rotate90)
            .unwrap_err();
        assert_alloc_failed(&err);
    });
    let out = slice.oriented(zenpixels::Orientation::Rotate90).unwrap();
    assert_eq!((out.width(), out.height()), (48, 96));
}

/// A square transpose needs no `moved` scratch: it succeeds with any
/// bitvec-sized reserve refused. 200x200 makes the scratch a 5 KiB
/// allocation if it were wrongly allocated, so the 4 KiB gate catches it.
#[test]
fn square_in_place_transpose_needs_no_scratch() {
    let _g = lock();
    let buf = PixelBuffer::try_new(200, 200, RGB8).unwrap();
    let out = failing_above(4096, || {
        buf.into_oriented(zenpixels::Orientation::Transpose)
            .unwrap()
    });
    assert_eq!((out.width(), out.height()), (200, 200));
}
