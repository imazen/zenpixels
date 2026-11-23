//! Integration test with a thread-local failing `GlobalAlloc`.
//!
//! Proves that every allocating constructor and fallback copy path in `zenpixels 0.5`
//! returns `Err(BufferError::AllocationFailed)` cleanly on OOM without panicking.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use zenpixels::{
    BandedBuf, BufferError, ImageSpec, PixelBuf, PlanarBuf, PlaneRole, PlaneSpec, RowAlign,
    SampleType,
};

struct FailingAlloc;

thread_local! {
    static FAIL_THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[global_allocator]
static ALLOC: FailingAlloc = FailingAlloc;

unsafe impl GlobalAlloc for FailingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= FAIL_THRESHOLD.with(Cell::get) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= FAIL_THRESHOLD.with(Cell::get) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= FAIL_THRESHOLD.with(Cell::get) {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn with_oom_above<T>(threshold: usize, f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FAIL_THRESHOLD.with(|c| c.set(usize::MAX));
        }
    }
    FAIL_THRESHOLD.with(|c| c.set(threshold));
    let _guard = Reset;
    f()
}

#[test]
fn constructors_and_copies_return_allocation_failed_on_oom() {
    // PixelBuf::new (tight)
    let err = with_oom_above(4096, || PixelBuf::new(64, 64, ImageSpec::rgba8_srgb())).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // PixelBuf::new_aligned (Cache64 / AlignedSamples)
    let err = with_oom_above(4096, || {
        PixelBuf::new_aligned(64, 64, ImageSpec::rgbf32_linear(), RowAlign::Cache64)
    })
    .unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // BandedBuf::new
    let err = with_oom_above(4096, || {
        BandedBuf::new(64, 64, ImageSpec::rgba8_srgb(), 8192)
    })
    .unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // PlanarBuf::new
    let specs = [
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    ];
    let err = with_oom_above(4096, || PlanarBuf::new(128, 128, &specs)).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // View fallback copy paths: try_to_tight_buf, try_contiguous_bytes
    let padded = PixelBuf::new_aligned(33, 64, ImageSpec::rgb8_srgb(), RowAlign::Cache64).unwrap();
    let err = with_oom_above(2048, || padded.as_view().try_to_tight_buf()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    let err = with_oom_above(2048, || padded.as_view().try_contiguous_bytes()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // try_into_vec fallback copy path (Aligned64 cannot zero-copy to Vec<u8>)
    let err = with_oom_above(2048, || padded.try_into_vec::<u8>()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
}
