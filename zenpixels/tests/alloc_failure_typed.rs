//! Allocation-failure injection for the typed `Pixel` extraction paths
//! (issue #151), gated on the `rgb` feature because
//! [`PixelBuffer::try_into_contiguous_pixels`] lives there.
//!
//! Same threshold-allocator approach as `alloc_failure.rs` — this file is its
//! own test binary, so its `#[global_allocator]` affects only these tests.

#[path = "common/fail_allocator.rs"]
mod fail_allocator;
use fail_allocator::{failing_above, lock};

use zenpixels::{BufferError, PixelBuffer, PixelDescriptor};

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
