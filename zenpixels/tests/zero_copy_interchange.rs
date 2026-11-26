//! Verifies the exact zero-copy vs safe-copy contracts for `PixelBuf`, `PixelView`,
//! `PixelStorage`, `rgb`, `imgref`, and `image` 0.25 under Rust's `GlobalAlloc::dealloc`
//! alignment-matching and `bytemuck::allocation::try_cast_vec` rules.

#[cfg(feature = "imgref")]
use zenpixels::RowAlign;
use zenpixels::{ColorSpace, F16Sample, GrayAlphaF16, PixelBuf, PixelFormat, SampleLayout};

#[test]
fn vec_sample_and_pixel_zero_copy_roundtrips_preserve_pointer_identity() {
    // Vec<u16> -> PixelBuf (RGB16) -> Vec<u16>
    let mut raw_u16 = Vec::with_capacity(12);
    raw_u16.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let in_ptr = raw_u16.as_ptr();
    let buf = PixelBuf::from_typed(raw_u16, 2, 2, PixelFormat::RGB16_SRGB).unwrap();
    assert_eq!(buf.as_view().row_as::<u16>(0).unwrap().as_ptr(), in_ptr);
    let back = buf.into_vec_zero_copy::<u16>().unwrap();
    assert_eq!(back.as_ptr(), in_ptr);

    // Vec<F16Sample> -> PixelBuf (RGBF16) -> Vec<F16Sample>
    let f16s = vec![
        F16Sample::ZERO,
        F16Sample::ONE,
        F16Sample::from_f32(0.5),
        F16Sample::from_f32(2.0),
        F16Sample::NEG_ONE,
        F16Sample::from_f32(0.25),
    ];
    let f16_ptr = f16s.as_ptr();
    let buf = PixelBuf::from_typed(f16s, 2, 1, PixelFormat::RGBF16_LINEAR).unwrap();
    let back = buf.into_vec_zero_copy::<F16Sample>().unwrap();
    assert_eq!(back.as_ptr(), f16_ptr);

    // Vec<GrayAlphaF16> -> PixelBuf -> Vec<GrayAlphaF16> (align=2, size=4)
    let ga = vec![
        GrayAlphaF16::new(F16Sample::from_f32(0.25), F16Sample::ONE),
        GrayAlphaF16::new(F16Sample::from_f32(0.75), F16Sample::from_f32(0.5)),
    ];
    let ga_ptr = ga.as_ptr();
    let spec_ga = PixelFormat::new(
        SampleLayout::GRAYAF16,
        ColorSpace::GRAY_LINEAR.with_alpha(zenpixels::AlphaMode::Straight),
    );
    let buf = PixelBuf::from_typed(ga, 2, 1, spec_ga).unwrap();
    let back = buf.into_vec_zero_copy::<GrayAlphaF16>().unwrap();
    assert_eq!(back.as_ptr(), ga_ptr);
}

#[cfg(feature = "rgb")]
#[test]
fn rgb_crate_zero_copy_moves_and_capacity_slop_guards() {
    use rgb::{Rgb, Rgba};

    // Vec<Rgb<u8>> zero-copy roundtrip
    let px8 = vec![Rgb::new(1u8, 2, 3), Rgb::new(4, 5, 6)];
    let p8 = px8.as_ptr();
    let buf = PixelBuf::from_typed(px8, 2, 1, PixelFormat::RGB8_SRGB).unwrap();
    assert_eq!(
        buf.as_view()
            .as_contiguous_typed::<Rgb<u8>>()
            .unwrap()
            .as_ptr(),
        p8
    );
    let back8 = buf.into_vec_zero_copy::<Rgb<u8>>().unwrap();
    assert_eq!(back8.as_ptr(), p8);

    // Vec<Rgb<u16>> zero-copy roundtrip (align=2 == align_of::<u16>(), cap_bytes % 6 == 0)
    let px16 = vec![Rgb::new(100u16, 200, 300), Rgb::new(400, 500, 600)];
    let p16 = px16.as_ptr();
    let buf = PixelBuf::from_typed(px16, 2, 1, PixelFormat::RGB16_SRGB).unwrap();
    assert_eq!(buf.as_view().row_as::<Rgb<u16>>(0).unwrap().as_ptr(), p16);
    let back16 = buf.into_vec_zero_copy::<Rgb<u16>>().unwrap();
    assert_eq!(back16.as_ptr(), p16);

    // Vec<Rgba<f32>> zero-copy roundtrip (align=4 == align_of::<f32>())
    let pxf32 = vec![
        Rgba::new(0.1f32, 0.2, 0.3, 1.0),
        Rgba::new(0.4, 0.5, 0.6, 0.8),
    ];
    let pf32 = pxf32.as_ptr();
    let buf = PixelBuf::from_typed(pxf32, 2, 1, PixelFormat::RGBAF32_LINEAR).unwrap();
    let backf32 = buf.into_vec_zero_copy::<Rgba<f32>>().unwrap();
    assert_eq!(backf32.as_ptr(), pf32);

    // Capacity slop guard: Vec<u16> with cap=7 (14 bytes, NOT divisible by 6 = size_of::<Rgb<u16>>())
    let mut sloppy = Vec::with_capacity(7);
    sloppy.extend_from_slice(&[10u16, 20, 30, 40, 50, 60]);
    assert_eq!(sloppy.capacity(), 7);
    let sloppy_ptr = sloppy.as_ptr();
    let buf = PixelBuf::from_typed(sloppy, 2, 1, PixelFormat::RGB16_SRGB).unwrap();
    // Strict zero-copy MUST refuse (dealloc layout capacity mismatch) and return buf untouched.
    let buf = buf.into_vec_zero_copy::<Rgb<u16>>().unwrap_err();
    // Safe fallback copies cleanly with exact values and a fresh pointer.
    let copied = buf.into_vec::<Rgb<u16>>().unwrap();
    assert_ne!(copied.as_ptr() as usize, sloppy_ptr as usize);
    assert_eq!(copied, vec![Rgb::new(10, 20, 30), Rgb::new(40, 50, 60)]);

    // Vec<u8> holding RGB16 bytes: even if pointer is even on system malloc, dealloc align is 1 != 2,
    // so zero-copy into Vec<Rgb<u16>> MUST refuse and try_into_vec MUST copy safely!
    let mut bytes_rgb16 = Vec::with_capacity(12);
    for v in [11u16, 22, 33, 44, 55, 66] {
        bytes_rgb16.extend_from_slice(&v.to_ne_bytes());
    }
    let bytes_ptr = bytes_rgb16.as_ptr();
    let buf = PixelBuf::from_bytes(bytes_rgb16, 2, 1, PixelFormat::RGB16_SRGB).unwrap();
    assert_eq!(buf.storage().allocation_alignment(), 1);
    let buf = buf.into_vec_zero_copy::<Rgb<u16>>().unwrap_err();
    let copied = buf.into_vec::<Rgb<u16>>().unwrap();
    assert_ne!(copied.as_ptr() as usize, bytes_ptr as usize);
    assert_eq!(copied, vec![Rgb::new(11, 22, 33), Rgb::new(44, 55, 66)]);
}

#[cfg(feature = "rgb")]
#[test]
fn aligned64_borrows_zero_copy_as_typed_slice_but_copies_for_vec() {
    use rgb::Rgba;

    let mut buf =
        PixelBuf::new_aligned(4, 2, PixelFormat::RGBAF32_LINEAR, RowAlign::Cache64).unwrap();
    assert_eq!(buf.storage().allocation_alignment(), 64);
    assert_eq!(buf.stride(), 64); // 4 * 16 = 64 bytes -> contiguous AND 64-byte aligned!

    buf.as_view_mut().row_as_mut::<Rgba<f32>>(0).unwrap()[1] = Rgba::new(0.5, 0.25, 0.75, 1.0);
    let slice = buf.as_view().as_contiguous_typed::<Rgba<f32>>().unwrap();
    assert_eq!((slice.as_ptr() as usize) % 64, 0);
    assert_eq!(slice[1], Rgba::new(0.5, 0.25, 0.75, 1.0));
    let slice_ptr = slice.as_ptr();

    // Aligned64 has dealloc align=64 != align_of::<Rgba<f32>>()=4 -> strict zero-copy refuses.
    let buf = buf.into_vec_zero_copy::<Rgba<f32>>().unwrap_err();
    // Safe fallback copies into Vec<Rgba<f32>> cleanly.
    let vec = buf.into_vec::<Rgba<f32>>().unwrap();
    assert_ne!(vec.as_ptr(), slice_ptr);
    assert_eq!(vec[1], Rgba::new(0.5, 0.25, 0.75, 1.0));
}

#[cfg(feature = "imgref")]
#[test]
fn imgref_zero_copy_roundtrip_and_strided_view() {
    use rgb::Rgb;

    let data = vec![Rgb::new(10u16, 20, 30); 6];
    let ptr = data.as_ptr();
    let img = imgref::ImgVec::new(data, 3, 2);

    let buf = PixelBuf::from_imgvec(img, ColorSpace::SRGB).unwrap();
    let imgref_view = buf.as_view().as_imgref::<Rgb<u16>>().unwrap();
    assert_eq!(imgref_view.buf().as_ptr(), ptr);
    assert_eq!((imgref_view.width(), imgref_view.height()), (3, 2));

    let out = buf.into_imgvec::<Rgb<u16>>().unwrap();
    assert_eq!(out.buf().as_ptr(), ptr);
}

#[cfg(feature = "image")]
#[test]
fn image_crate_zero_copy_roundtrip() {
    let raw = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let ptr = raw.as_ptr();
    let rgb_img = image::RgbImage::from_raw(2, 2, raw).unwrap();

    let buf = PixelBuf::from_rgb8_image_with_color(rgb_img, ColorSpace::SRGB).unwrap();
    assert_eq!(buf.as_view().row(0).as_ptr(), ptr);

    let dyn_img = buf.into_dynamic_image().unwrap();
    match dyn_img {
        image::DynamicImage::ImageRgb8(img) => {
            assert_eq!(img.as_raw().as_ptr(), ptr);
        }
        other => panic!("expected ImageRgb8, got {other:?}"),
    }
}
