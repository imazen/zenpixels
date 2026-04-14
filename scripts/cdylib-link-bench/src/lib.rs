//! cdylib link-time benchmark surface.
//!
//! Exports `extern "C"` symbols that pull in real codegen from
//! `zenpixels` and `zenpixels-convert` so the linker has to resolve and ship
//! a representative slice of the public API. Use it to time the full
//! compile + link path:
//!
//! ```text
//! cargo clean
//! cargo build --release -p cdylib-link-bench                              # default (incl. cms-moxcms)
//! cargo build --release -p cdylib-link-bench --no-default-features        # minimum surface
//! cargo build --release -p cdylib-link-bench --features zencms-lite       # add zencms-lite path
//! ```
//!
//! Pair with `cargo bloat --release -p cdylib-link-bench --crates` to see
//! per-crate `.text` contribution to the resulting `libcdylib_link_bench.so`.

use core::ffi::c_void;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Cicp, ColorPrimaries, PixelDescriptor, PixelFormat,
    TransferFunction,
};
use zenpixels_convert::{ColorPrimariesExt, RowConverter, TransferFunctionExt};

#[unsafe(no_mangle)]
pub extern "C" fn zpx_descriptor_size(t: u32) -> usize {
    match t {
        0 => core::mem::size_of::<PixelDescriptor>(),
        1 => core::mem::size_of::<PixelFormat>(),
        2 => core::mem::size_of::<Cicp>(),
        3 => core::mem::size_of::<RowConverter>(),
        _ => 0,
    }
}

/// Force monomorphisation of the planning + per-row conversion path
/// for a common sRGB u8 -> linear f32 RGBA conversion.
/// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn zpx_plan_srgb_u8_to_linear_f32() -> u32 {
    let src = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let dst = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    RowConverter::new(src, dst).is_ok() as u32
}

/// Force a real per-row conversion to drag the conversion kernels in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zpx_convert_one_row(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_len: usize,
    width: u32,
) -> u32 {
    if src.is_null() || dst.is_null() {
        return 0;
    }
    let src_slice = unsafe { core::slice::from_raw_parts(src, src_len) };
    let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, dst_len) };
    let from = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let to = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let Ok(mut rc) = RowConverter::new(from, to) else {
        return 0;
    };
    rc.convert_row(src_slice, dst_slice, width);
    1
}

/// Force monomorphisation of the transfer-function fast path.
#[unsafe(no_mangle)]
pub extern "C" fn zpx_srgb_decode_sample(value: f32) -> f32 {
    TransferFunction::Srgb.linearize(value)
}

/// Force monomorphisation of a primaries-to-XYZ matrix lookup.
#[unsafe(no_mangle)]
pub extern "C" fn zpx_p3_to_xyz_matrix_into(out: *mut f32) -> u32 {
    if out.is_null() {
        return 0;
    }
    let Some(m) = ColorPrimaries::DisplayP3.to_xyz_matrix() else {
        return 0;
    };
    unsafe {
        for (i, v) in m.iter().flatten().enumerate() {
            out.add(i).write(*v);
        }
    }
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn zpx_touch(p: *mut c_void) -> usize {
    p as usize
}
