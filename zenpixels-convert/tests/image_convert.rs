//! Integration tests for the `PixelBufferImageExt` converting helpers
//! (feature = "image"): `to_image_rgb8` / `to_image_rgba8`.
//!
//! Unlike zenpixels' byte-level `to_dynamic_image` (a format-preserving
//! reinterpret), these *convert* any source format into a standard sRGB
//! `image` buffer. They also exercise the convert-crate `prelude`.
#![cfg(feature = "image")]

use zenpixels::{PixelBuffer, PixelDescriptor};
// The prelude brings the ext traits into scope so the methods are callable.
use zenpixels_convert::prelude::*;

#[test]
fn to_image_rgba8_adds_opaque_alpha_from_rgb() {
    let data = vec![10u8, 20, 30, 40, 50, 60];
    let pb = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();

    let img = pb.to_image_rgba8().expect("rgb -> rgba8");

    assert_eq!(img.dimensions(), (2, 1));
    assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [40, 50, 60, 255]);
}

#[test]
fn to_image_rgb8_drops_alpha_from_rgba() {
    let data = vec![10u8, 20, 30, 128, 40, 50, 60, 200];
    let pb = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGBA8_SRGB).unwrap();

    let img = pb.to_image_rgb8().expect("rgba -> rgb8");

    assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30]);
    assert_eq!(img.get_pixel(1, 0).0, [40, 50, 60]);
}

#[test]
fn to_image_rgb8_expands_gray() {
    let pb = PixelBuffer::from_vec(vec![123u8, 200], 2, 1, PixelDescriptor::GRAY8_SRGB).unwrap();

    let img = pb.to_image_rgb8().expect("gray -> rgb8");

    assert_eq!(img.get_pixel(0, 0).0, [123, 123, 123]);
    assert_eq!(img.get_pixel(1, 0).0, [200, 200, 200]);
}

#[test]
fn to_image_rgb8_narrows_16bit() {
    let px: [u16; 3] = [65535, 0, 257];
    let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let pb = PixelBuffer::from_vec(bytes, 1, 1, PixelDescriptor::RGB16_SRGB).unwrap();

    let img = pb.to_image_rgb8().expect("rgb16 -> rgb8");

    // 16-bit narrows by ÷257: 65535→255, 0→0, 257→1.
    assert_eq!(img.get_pixel(0, 0).0, [255, 0, 1]);
}

#[test]
fn to_image_rgb8_returns_needs_cms_for_cmyk() {
    let pb = PixelBuffer::from_vec(vec![0u8; 4], 1, 1, PixelDescriptor::CMYK8).unwrap();

    let err = pb.to_image_rgb8().expect_err("CMYK needs a CMS plugin");
    assert!(matches!(
        *err.error(),
        zenpixels_convert::ConvertError::NeedsCms { .. }
    ));
}

#[test]
fn roundtrip_image_through_pixelbuffer_is_identity() {
    // image RgbaImage → PixelBuffer (zenpixels bridge) → to_image_rgba8.
    let src = image::RgbaImage::from_fn(5, 4, |x, y| image::Rgba([x as u8, y as u8, 7, 255]));

    let pb: PixelBuffer = src.clone().into();
    let back = pb.to_image_rgba8().expect("roundtrip");

    assert_eq!(back, src);
}
