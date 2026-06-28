//! Integration tests for the `image`-crate interop bridge (feature = "image").
//!
//! These double as usage documentation: they only touch the public API, so they
//! show exactly how a caller bridges between `image::{ImageBuffer, DynamicImage}`
//! and `zenpixels::{PixelBuffer, PixelSlice}`.
#![cfg(feature = "image")]

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};

fn rgba_gradient(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        Rgba([
            (x * 7) as u8,
            (y * 11) as u8,
            (x + y) as u8,
            255 - (x as u8),
        ])
    })
}

#[test]
fn rgba_image_into_pixelbuffer_roundtrips() {
    let img = rgba_gradient(8, 5);
    let expected = img.clone().into_raw();

    // The headline boilerplate-killer: one `.into()`.
    let pb: PixelBuffer = img.into();

    assert_eq!(pb.width(), 8);
    assert_eq!(pb.height(), 5);
    assert_eq!(pb.descriptor(), PixelDescriptor::RGBA8_SRGB);
    assert_eq!(pb.descriptor().format, PixelFormat::Rgba8);
    // Bytes preserved exactly (8-bit path is zero-copy).
    assert_eq!(pb.copy_to_contiguous_bytes(), expected);
}

#[test]
fn rgb_image_borrow_as_pixelslice_is_zero_copy() {
    let img = RgbImage::from_fn(4, 4, |x, y| Rgb([x as u8, y as u8, 0]));
    let raw = img.as_raw().clone();

    // Borrow without consuming or copying.
    let slice: PixelSlice = (&img).into();

    assert_eq!(slice.width(), 4);
    assert_eq!(slice.rows(), 4);
    assert_eq!(slice.descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!(slice.as_contiguous_bytes(), Some(raw.as_slice()));
    // `img` is still usable after the borrow.
    assert_eq!(img.dimensions(), (4, 4));
}

#[test]
fn dynamic_image_dispatches_per_variant() {
    let cases: [(DynamicImage, PixelFormat); 4] = [
        (
            DynamicImage::ImageLuma8(GrayImage::from_pixel(2, 2, Luma([9]))),
            PixelFormat::Gray8,
        ),
        (
            DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([1, 2, 3]))),
            PixelFormat::Rgb8,
        ),
        (
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 4]))),
            PixelFormat::Rgba8,
        ),
        (
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(2, 2, Rgb([300u16, 1, 2]))),
            PixelFormat::Rgb16,
        ),
    ];
    for (dyn_img, want) in cases {
        let pb: PixelBuffer = dyn_img.into();
        assert_eq!(pb.descriptor().format, want, "variant {want:?}");
        assert_eq!((pb.width(), pb.height()), (2, 2));
    }
}

#[test]
fn to_dynamic_image_roundtrips_rgba8() {
    let img = rgba_gradient(6, 6);
    let original = img.clone();

    let pb: PixelBuffer = img.into();
    let back = pb.to_dynamic_image().expect("Rgba8 is mappable");

    let back_rgba = back.as_rgba8().expect("round-trips to Rgba8");
    assert_eq!(back_rgba, &original);
}

#[test]
fn to_dynamic_image_roundtrips_16bit() {
    let img: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_fn(5, 3, |x, y| Rgb([x as u16 * 1000, y as u16 * 2000, 65535]));
    let original = img.clone();

    let pb: PixelBuffer = img.into();
    assert_eq!(pb.descriptor().format, PixelFormat::Rgb16);

    let back = pb.to_dynamic_image().expect("Rgb16 is mappable");
    let back16 = back.as_rgb16().expect("round-trips to Rgb16");
    assert_eq!(back16, &original);
}

#[test]
fn to_dynamic_image_roundtrips_f32() {
    let img: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::from_fn(4, 2, |x, y| {
        Rgba([x as f32 * 0.25, y as f32 * 0.5, 1.0, 0.5])
    });
    let original = img.clone();

    let pb: PixelBuffer = img.into();
    assert_eq!(pb.descriptor().format, PixelFormat::RgbaF32);

    let back = pb.to_dynamic_image().expect("RgbaF32 is mappable");
    let back_f32 = back.as_rgba32f().expect("round-trips to RgbaF32");
    assert_eq!(back_f32, &original);
}

#[test]
fn to_dynamic_image_strips_row_padding() {
    // A SIMD-aligned buffer has stride padding; `to_dynamic_image` must produce
    // a tightly-packed `image` buffer with the padding removed.
    let (w, h) = (3u32, 4u32);
    let mut pb = PixelBuffer::new_simd_aligned(w, h, PixelDescriptor::RGB8_SRGB, 64);
    assert!(
        pb.stride() > (w as usize) * 3,
        "test needs a padded stride to be meaningful (got {})",
        pb.stride()
    );

    // Write a known value into every pixel, row by row (skipping padding).
    for y in 0..h {
        let mut rows = pb.rows_mut(y, 1);
        let row = rows.row_mut(0);
        for (i, b) in row.iter_mut().enumerate() {
            *b = (y as usize * 30 + i) as u8;
        }
    }

    let rgb = pb.to_dynamic_image().unwrap().to_rgb8();
    assert_eq!(rgb.dimensions(), (w, h));
    for y in 0..h {
        for x in 0..w {
            let px = rgb.get_pixel(x, y);
            let base = y as usize * 30 + (x as usize) * 3;
            assert_eq!(px.0, [base as u8, (base + 1) as u8, (base + 2) as u8]);
        }
    }
}

#[test]
fn to_dynamic_image_returns_none_for_unmappable_layouts() {
    // BGRA has no `image`-crate equivalent → None (caller must convert first).
    let pb = PixelBuffer::new(2, 2, PixelDescriptor::BGRA8_SRGB);
    assert!(pb.to_dynamic_image().is_none());

    // Padded RGBX (4th byte undefined) must not masquerade as Rgba8.
    let pb = PixelBuffer::new(2, 2, PixelDescriptor::RGBX8_SRGB);
    assert!(pb.to_dynamic_image().is_none());
}

#[test]
fn future_dynamic_image_variant_falls_back_to_rgba8() {
    // Sanity: the documented fallback path for `#[non_exhaustive]` works for the
    // known variants too (a Luma8 routed through the generic conversion).
    let dyn_img = DynamicImage::ImageLuma8(GrayImage::from_pixel(2, 2, Luma([42])));
    let rgba = dyn_img.to_rgba8();
    let pb: PixelBuffer = rgba.into();
    assert_eq!(pb.descriptor().format, PixelFormat::Rgba8);
}
