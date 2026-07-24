//! Integration tests for the `PixelBufferImageExt` converting helpers
//! (feature = "image"): `to_image_rgb8` / `to_image_rgba8`.
//!
//! Unlike zenpixels' byte-level `to_dynamic_image` (a format-preserving
//! reinterpret), these *convert* any source format into a standard sRGB
//! `image` buffer.
#![cfg(feature = "image")]

use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};
use zenpixels_convert::{DynamicImageExt, ImageBufferExt, PixelBufferImageExt};

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

    let pb = src
        .clone()
        .try_into_pixel_buffer()
        .expect("image buffer is valid");
    let back = pb.to_image_rgba8().expect("roundtrip");

    assert_eq!(back, src);
}

#[test]
fn owned_and_borrowed_image_interop_preserve_layout() {
    let image = RgbImage::from_fn(4, 3, |x, y| Rgb([x as u8, y as u8, 7]));
    let expected = image.clone().into_raw();

    let slice: PixelSlice<'_> = image.try_as_pixel_slice().expect("valid image layout");
    assert_eq!(slice.descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!(slice.as_contiguous_bytes(), Some(image.as_raw().as_slice()));

    let buffer = image
        .try_into_pixel_buffer()
        .expect("valid image allocation");
    assert_eq!(buffer.copy_to_contiguous_bytes(), expected);
}

#[test]
fn dynamic_image_dispatches_and_wide_formats_roundtrip() {
    let dynamic = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(2, 2, Luma([42])));
    let gray = dynamic
        .try_into_pixel_buffer()
        .expect("known DynamicImage variant");
    assert_eq!(gray.descriptor().format, PixelFormat::Gray8);

    let wide: ImageBuffer<Rgba<f32>, Vec<f32>> =
        ImageBuffer::from_fn(3, 2, |x, y| Rgba([x as f32, y as f32, 0.5, 1.0]));
    let expected = wide.clone();
    let buffer = wide.try_into_pixel_buffer().expect("wide image");
    assert_eq!(buffer.descriptor().format, PixelFormat::RgbaF32);
    assert_eq!(
        buffer
            .to_dynamic_image()
            .expect("RgbaF32 is representable")
            .into_rgba32f(),
        expected
    );
}

#[test]
fn to_dynamic_image_strips_wide_row_padding_in_one_output_allocation() {
    let (width, height) = (3u32, 2u32);
    let mut buffer = PixelBuffer::new_simd_aligned(width, height, PixelDescriptor::RGB16, 64);
    for y in 0..height {
        let mut rows = buffer.rows_mut(y, 1);
        let values = bytemuck::cast_slice_mut::<u8, u16>(rows.row_mut(0));
        for (i, value) in values.iter_mut().enumerate() {
            *value = y as u16 * 100 + i as u16;
        }
    }

    let image = buffer
        .to_dynamic_image()
        .expect("Rgb16 is representable")
        .into_rgb16();
    assert_eq!(
        image.into_raw(),
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103, 104, 105, 106, 107, 108
        ]
    );
}

#[test]
fn to_dynamic_image_rejects_unrepresentable_layout() {
    let buffer = PixelBuffer::new(2, 2, PixelDescriptor::BGRA8_SRGB);
    assert!(buffer.to_dynamic_image().is_none());
}
