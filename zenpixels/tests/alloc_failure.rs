//! Integration test with a thread-local failing `GlobalAlloc`.
//!
//! Proves that every allocating constructor and fallback copy path in `zenpixels 0.5`
//! returns `Err(BufferError::AllocationFailed)` cleanly on OOM without panicking.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use zenpixels::{
    BandedBuf, BufferError, PixelBuf, PixelFormat, PixelRect, PlanarBuf, PlaneBuf, PlaneFormat,
    PlaneRole, RowAlign, SampleType,
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
    let err = with_oom_above(4096, || PixelBuf::new(64, 64, PixelFormat::RGBA8_SRGB)).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // PixelBuf::new_aligned (Cache64 / AlignedSamples)
    let err = with_oom_above(4096, || {
        PixelBuf::new_aligned(64, 64, PixelFormat::RGBF32_LINEAR, RowAlign::Cache64)
    })
    .unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // BandedBuf::new
    let err = with_oom_above(4096, || {
        BandedBuf::new(64, 64, PixelFormat::RGBA8_SRGB, 8192)
    })
    .unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // PlanarBuf::new
    let specs = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    ];
    let err = with_oom_above(4096, || PlanarBuf::new(128, 128, &specs)).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // View fallback copy paths: try_to_tight_buf, try_contiguous_bytes
    let padded = PixelBuf::new_aligned(33, 64, PixelFormat::RGB8_SRGB, RowAlign::Cache64).unwrap();
    let err = with_oom_above(2048, || padded.as_view().to_tight_buf()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    let err = with_oom_above(2048, || padded.as_view().to_contiguous_bytes()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // try_into_vec fallback copy path (Aligned64 cannot zero-copy to Vec<u8>)
    let err = with_oom_above(2048, || padded.into_vec::<u8>()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // External immutable storage CoW promotion (`try_make_mut`) on OOM leaves buffer intact!
    let ext_backing = vec![0x5Au8; 64 * 64 * 4];
    let mut ext_buf = PixelBuf::from_external(
        std::sync::Arc::<[u8]>::from(ext_backing.into_boxed_slice()),
        64,
        64,
        64 * 4,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap();
    let err = with_oom_above(4096, || ext_buf.make_mut()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
    assert!(ext_buf.is_external());
    assert_eq!(ext_buf.as_view().row(0)[0], 0x5A);

    // Fallible CoW promotion on `sub_rows_mut` / `crop_view_mut` returns `AllocationFailed`
    // without panicking, and out-of-bounds fails BEFORE attempting CoW allocation.
    let err = with_oom_above(4096, || ext_buf.sub_rows_mut(0, 4)).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
    let err =
        with_oom_above(4096, || ext_buf.crop_view_mut(PixelRect::new(0, 0, 4, 4))).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
    let err =
        with_oom_above(4096, || ext_buf.crop_view_mut(PixelRect::new(100, 0, 4, 4))).unwrap_err();
    assert_eq!(*err.error(), BufferError::OutOfBounds);
    assert!(ext_buf.is_external());

    // Padded external immutable storage `into_tight_buf` returns `AllocationFailed` cleanly on OOM!
    let padded_ext = PixelBuf::from_external(
        std::sync::Arc::<[u8]>::from(vec![0x42u8; 64 * 260].into_boxed_slice()),
        64,
        64,
        260,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap();
    let err = with_oom_above(4096, || padded_ext.into_tight_buf()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    // PlaneBuf::from_external CoW promotion (`try_as_view_mut`) on OOM leaves plane intact!
    let mut ext_plane = PlaneBuf::from_external(
        std::sync::Arc::<[u8]>::from(vec![0x33u8; 128 * 64].into_boxed_slice()),
        128,
        64,
        128,
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let err = with_oom_above(4096, || ext_plane.try_as_view_mut()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
    let err =
        with_oom_above(4096, || ext_plane.crop_view_mut(PixelRect::new(0, 0, 4, 4))).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
    assert_eq!(ext_plane.as_view().row(0)[0], 0x33);

    // BandedBuf::from_view and multi-band into_buf OOM paths
    let src_big = PixelBuf::new(64, 64, PixelFormat::RGBA8_SRGB).unwrap();
    let err = with_oom_above(4096, || BandedBuf::from_view(&src_big.as_view(), 8192)).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);

    let banded = BandedBuf::from_view(&src_big.as_view(), 4096).unwrap();
    assert!(banded.band_count() > 1);
    let err = with_oom_above(8192, || banded.into_buf()).unwrap_err();
    assert_eq!(*err.error(), BufferError::AllocationFailed);
}
