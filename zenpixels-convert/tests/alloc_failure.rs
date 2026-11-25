//! Integration test with a thread-local failing `GlobalAlloc` for `zenpixels-convert`.
//!
//! 1. Proves allocating APIs (`ImagePlan::execute`, `measure_histogram`, `measure_max_smoothed`,
//!    `apply_orientation`, `apply_orientation_banded`, `reconstruct_planar_f32`) return
//!    `Err(ConvertError::AllocationFailed)` cleanly on OOM without panicking.
//! 2. Proves zero-allocation APIs (`RowPlan::convert_row`, `apply_orientation_into`,
//!    `apply_orientation_banded_into`, `reconstruct_planar_into`) perform **zero heap allocations**
//!    even when every allocation >= 1 byte fails!

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use zenpixels::{
    BandedBuf, ColorSpace, DiffuseWhite, ImageSpec, Orientation, PixelBuf, PlanarBuf, PlaneBuf,
    PlaneRole, PlaneSpec, Primaries, SampleType,
};
use zenpixels_convert::{
    Accelerators, ConvertError, ConvertPolicy, ImagePlan, LightLevelMethod, PeakSource,
    PlanarReconstructionOptions, RowPlan, YcbcrMatrix, apply_orientation, apply_orientation_banded,
    apply_orientation_banded_into, apply_orientation_into, measure_histogram, measure_max_smoothed,
    reconstruct_planar_f32, reconstruct_planar_into,
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
fn converter_allocating_paths_return_allocation_failed_on_oom() {
    let src = PixelBuf::new(64, 64, ImageSpec::RGBA8_SRGB).unwrap();

    // 1. ImagePlan::execute OOM
    let plan = ImagePlan::build(
        ImageSpec::RGBA8_SRGB,
        ImageSpec::RGBF32_LINEAR,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    let err = with_oom_above(4096, || plan.execute(&src)).unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);

    // 2. measure_histogram & measure_max_smoothed OOM (1024*8 = 8192 B histogram or scratch row)
    let big_w = PixelBuf::new(1024, 2, ImageSpec::RGB8_SRGB).unwrap();
    let err = with_oom_above(4096, || {
        measure_histogram(&big_w, DiffuseWhite::BT2408, None, LightLevelMethod::MaxRgb)
    })
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);

    let err = with_oom_above(4096, || {
        measure_max_smoothed(&big_w, DiffuseWhite::BT2408, None, LightLevelMethod::MaxRgb)
    })
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);

    // 3. apply_orientation & apply_orientation_banded OOM
    let err = with_oom_above(4096, || {
        apply_orientation(&src.as_view(), Orientation::Rotate90)
    })
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);

    let banded = BandedBuf::from_view(&src.as_view(), 4096).unwrap();
    let err = with_oom_above(2048, || {
        apply_orientation_banded(&banded, Orientation::Rotate90, 4096)
    })
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);

    // 4. ycbcr_to_rgb_f32 OOM
    let y = PlaneBuf::from_external(
        vec![128u8; 64 * 64],
        64,
        64,
        64,
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let cb = PlaneBuf::from_external(
        vec![128u8; 32 * 32],
        32,
        32,
        32,
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let cr = PlaneBuf::from_external(
        vec![128u8; 32 * 32],
        32,
        32,
        32,
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let planar = PlanarBuf::from_planes(64, 64, vec![y, cb, cr])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let err = with_oom_above(4096, || reconstruct_planar_f32(&planar.as_view())).unwrap_err();
    assert_eq!(*err.error(), ConvertError::AllocationFailed);
}

#[test]
fn into_and_row_apis_execute_with_zero_heap_allocations() {
    let pol = ConvertPolicy::permissive().with_accelerators(Accelerators::ALL);
    let row_plan = RowPlan::build(
        ImageSpec::RGBA8_SRGB.with_primaries(Primaries::DisplayP3),
        ImageSpec::RGB16_SRGB,
        &pol,
        None,
    )
    .unwrap();
    let src_row = vec![123u8; 97 * 4];
    let mut dst_row = vec![0u8; 97 * 6];

    // Pre-allocate buffers for orient_into, orient_banded_into, and ycbcr_to_rgb_into.
    let src_img = PixelBuf::new(33, 29, ImageSpec::RGB8_SRGB).unwrap();
    let mut dst_img = PixelBuf::new(29, 33, ImageSpec::RGB8_SRGB).unwrap();

    let src_banded = BandedBuf::from_view(&src_img.as_view(), 33 * 3 * 7).unwrap();
    let mut dst_banded = BandedBuf::new(29, 33, ImageSpec::RGB8_SRGB, 29 * 3 * 5).unwrap();

    let y = PlaneBuf::from_external(
        vec![180u8; 256],
        16,
        16,
        16,
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let cb = PlaneBuf::from_external(
        vec![128u8; 64],
        8,
        8,
        8,
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let cr = PlaneBuf::from_external(
        vec![128u8; 64],
        8,
        8,
        8,
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let planar = PlanarBuf::from_planes(16, 16, vec![y, cb, cr])
        .unwrap()
        .with_color(ColorSpace::LINEAR_SRGB)
        .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let mut ycbcr_dst = PixelBuf::new(16, 16, ImageSpec::RGBF32_LINEAR).unwrap();

    // FAIL_THRESHOLD = 1 forbids ANY heap allocation >= 1 byte!
    with_oom_above(1, || {
        row_plan.convert_row(&src_row, &mut dst_row, 97, 0).unwrap();
        apply_orientation_into(
            &src_img.as_view(),
            Orientation::Rotate90,
            &mut dst_img.as_view_mut(),
        )
        .unwrap();
        apply_orientation_banded_into(&src_banded, Orientation::Rotate270, &mut dst_banded)
            .unwrap();
        reconstruct_planar_into(
            &planar.as_view(),
            &PlanarReconstructionOptions::nearest(),
            &mut ycbcr_dst.as_view_mut(),
        )
        .unwrap();
    });
}
