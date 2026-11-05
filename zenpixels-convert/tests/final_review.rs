//! Regressions from the final API review, 2026-11-05: each failed before its fix.

#![cfg(feature = "cms-moxcms")]

use std::sync::Arc;

use zenpixels::{
    ColorContext, ColorPrimaries as P, PixelBuffer, PixelDescriptor as D, PixelSlice,
    TransferFunction as T, buffer::BufferError,
};
use zenpixels_convert::adapt::adapt_raw_rows_for_encode;
use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
use zenpixels_convert::{ConvertOptions, PixelSliceConvertExt, RowConverter};

fn f16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// F16 cross-gamut took the U8 bridge, reading half the samples and writing half the row.
#[test]
fn f16_cross_gamut_converts_whole_rows() {
    const ONE: u16 = 0x3C00;
    let linear = D::new(
        zenpixels::ChannelType::F16,
        zenpixels::ChannelLayout::Rgb,
        None,
        T::Linear,
    );
    let from = linear.with_primaries(P::DisplayP3);
    let to = linear.with_primaries(P::Bt709);
    let src = f16_bytes(&[ONE; 6]);
    let mut dst = vec![0xAA; 12];
    RowConverter::new(from, to)
        .unwrap()
        .convert_row(&src, &mut dst, 2);
    for i in 0..6 {
        let code = u16::from_ne_bytes([dst[i * 2], dst[i * 2 + 1]]);
        assert!(code.abs_diff(ONE) <= 2, "{code:#x}");
    }
}

fn p3_tagged<'a>(pixel: &'a [u8]) -> PixelSlice<'a> {
    PixelSlice::new(pixel, 1, 1, 3, D::RGB8_SRGB)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(DISPLAY_P3_V4.to_vec())))
}

/// `adapt_for_encode` used the descriptor, ignored the ICC, then wrote a label for its target.
#[test]
fn adapt_for_encode_honors_the_current_profile_as_convert_to_does() {
    let target = D::RGB8_SRGB
        .with_transfer(T::Linear)
        .with_primaries(P::Bt709);
    let pixel = [128u8, 64, 32];
    let slice = p3_tagged(&pixel);
    let direct = slice.convert_to(target).unwrap();
    let adapted = slice
        .adapt_for_encode(&[target], &ConvertOptions::permissive())
        .unwrap();
    let adapted = adapted.as_slice();
    assert_eq!(adapted.row(0), direct.as_slice().row(0));
    assert_eq!(
        adapted.color_context().map(|c| c.cicp),
        direct.as_slice().color_context().map(|c| c.cicp)
    );
}

/// `convert_into` kept what the destination said about its old pixels.
#[test]
fn convert_into_replaces_the_destination_color_context() {
    let pixel = [128u8, 64, 32];
    let source = PixelSlice::new(&pixel, 1, 1, 3, D::RGB8_SRGB).unwrap();
    let target = D::RGB8_SRGB.with_transfer(T::Linear);
    let mut buffer = PixelBuffer::try_new(1, 1, target).unwrap();
    let stale = Arc::new(ColorContext::from_cicp(zenpixels::Cicp::DISPLAY_P3));
    let dst = buffer.as_slice_mut().with_color_context(stale);
    let converted = source.convert_into(dst).unwrap();
    let cicp = converted.color_context().and_then(|c| c.cicp);
    assert_ne!(cicp, Some(zenpixels::Cicp::DISPLAY_P3));
}

/// 8-byte samples passed `Color::try_new` into 4-byte-aligned storage, and a fallible conversion panicked.
#[test]
fn color_refuses_samples_its_storage_cannot_align() {
    let f64_gray = D::new(
        zenpixels::ChannelType::F64,
        zenpixels::ChannelLayout::Gray,
        None,
        T::Linear,
    );
    assert_eq!(
        zenpixels::Color::try_new(f64_gray, &0.5f64.to_ne_bytes()).unwrap_err(),
        BufferError::InvalidDimensions
    );
    let any = zenpixels::AnyColor::try_new(f64_gray, &0.5f64.to_ne_bytes()).unwrap();
    assert!(any.as_color().is_none());
}

/// A final row needs no padding, and a stride shorter than a row is an error rather than a panic.
#[test]
fn raw_adapter_geometry() {
    let options = ConvertOptions::permissive();
    let view =
        adapt_raw_rows_for_encode(&[0u8; 9], D::RGB8_SRGB, 1, 2, 6, &[D::RGB8_SRGB], &options);
    assert!(view.is_ok(), "{:?}", view.err().map(|e| e.to_string()));
    let short = adapt_raw_rows_for_encode(
        &[1u8, 2, 3],
        D::RGB8_SRGB,
        2,
        1,
        3,
        &[D::RGB16_SRGB],
        &options,
    );
    assert!(short.is_err());
}

fn close(got: &[f32], want: &[f32], tolerance: f32) {
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < tolerance, "{got:?} vs {want:?}");
    }
}

fn to_oklab(from: D, to: D, src: &[u8], pixels: u32) -> Vec<f32> {
    let mut out = vec![0f32; pixels as usize * to.channels()];
    RowConverter::new(from, to)
        .unwrap()
        .try_convert_row(src, bytemuck::cast_slice_mut(&mut out), pixels)
        .unwrap();
    out
}

/// Oklab kernels take linear RGB. Non-linear input skipped the transfer function, and float sRGB input
/// ran it after Oklab, clamping a and b.
#[test]
fn oklab_conversions_pass_through_linear_rgb() {
    let gray = to_oklab(D::RGB8_SRGB, D::OKLABF32, &[128, 128, 128], 1);
    close(&gray, &[0.5998752, 0.0, 0.0], 1e-4);

    let mut back = [0u8; 3];
    RowConverter::new(D::OKLABF32, D::RGB8_SRGB)
        .unwrap()
        .try_convert_row(
            bytemuck::cast_slice(&[0.5998752f32, 0.0, 0.0]),
            &mut back,
            1,
        )
        .unwrap();
    assert_eq!(back, [128, 128, 128]);

    let green = to_oklab(
        D::RGBF32_LINEAR.with_transfer(T::Srgb),
        D::OKLABF32.with_transfer(T::Linear),
        bytemuck::cast_slice(&[0.0f32, 1.0, 0.0]),
        1,
    );
    close(&green, &[0.8664396, -0.23388702, 0.1794982], 1e-3);
}

/// `Some(&MoxCms)` must not make a conversion fail, or lose alpha, that works without it.
#[test]
fn a_cms_handles_depth_alpha_and_gray_like_the_plan_does() {
    use zenpixels::{AlphaPolicy, Color};
    let cms = zenpixels_convert::cms_moxcms::MoxCms;
    let cms: &dyn zenpixels_convert::cms::PluggableCms = &cms;
    let p3 = |d: D| d.with_primaries(P::DisplayP3);

    // Across depths.
    let mut c = RowConverter::new_explicit_with_cms(
        p3(D::RGB16_SRGB),
        D::RGB8_SRGB,
        &ConvertOptions::permissive(),
        Some(cms),
    )
    .unwrap();
    let mut out = [0u8; 3];
    c.try_convert_row(bytemuck::cast_slice(&[65535u16; 3]), &mut out, 1)
        .unwrap();
    assert_eq!(out, [255; 3]);

    // Compositing is the plan's to do, not a plugin's to skip.
    let white = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(255, 255, 255),
        reference_white: None,
    });
    let mut c =
        RowConverter::new_explicit_with_cms(p3(D::RGBA8_SRGB), D::RGB8_SRGB, &white, Some(cms))
            .unwrap();
    let mut out = [0u8; 3];
    c.try_convert_row(&[0, 0, 0, 0], &mut out, 1).unwrap();
    assert_eq!(out, [255; 3]);
}

#[test]
fn orientation_refuses_a_different_format_of_the_same_size() {
    let pixel = [255u8, 0, 0, 255];
    let source = PixelSlice::new(&pixel, 1, 1, 4, D::RGBA8_SRGB).unwrap();
    let mut buffer = PixelBuffer::try_new(1, 1, D::BGRA8_SRGB).unwrap();
    let dst = buffer.as_slice_mut();
    assert!(
        source
            .oriented_into(zenpixels::Orientation::Identity, dst)
            .is_err()
    );
}

/// One conversion of a row and of its pixels one at a time must give the same bytes, through
/// `RowConverter::new`, which takes the fast gamut kernels.
#[test]
fn fast_gamut_kernels_are_row_split_invariant() {
    use zenpixels::ChannelType as C;
    let mut failures = Vec::new();
    for base in [
        D::RGB8_SRGB,
        D::RGBA8_SRGB,
        D::BGRA8_SRGB,
        D::RGB16_SRGB,
        D::RGBA16_SRGB,
        D::RGBF32_LINEAR,
        D::RGBAF32_LINEAR,
    ] {
        for (transfer_in, transfer_out) in [
            (T::Srgb, T::Srgb),
            (T::Linear, T::Srgb),
            (T::Srgb, T::Linear),
        ] {
            for (a, b) in [
                (P::DisplayP3, P::Bt709),
                (P::Bt709, P::DisplayP3),
                (P::Bt2020, P::Bt709),
            ] {
                let from = base.with_transfer(transfer_in).with_primaries(a);
                let to = base.with_transfer(transfer_out).with_primaries(b);
                let Ok(mut c) = RowConverter::new(from, to) else {
                    continue;
                };
                let n = 65_537usize;
                // A hash, so neighboring channels don't move together.
                let value = |i: usize| {
                    let h = (i as u32).wrapping_mul(0x9E37_79B1);
                    (h ^ (h >> 15)).wrapping_mul(0x85EB_CA77) >> 16
                };
                let sample = from.channel_type().byte_size();
                let bpp = from.bytes_per_pixel();
                let src: Vec<u8> = match from.channel_type() {
                    C::U8 => (0..n * bpp).map(|i| value(i) as u8).collect(),
                    C::U16 => (0..n * bpp / sample)
                        .flat_map(|i| (value(i) as u16).to_ne_bytes())
                        .collect(),
                    _ => (0..n * bpp / sample)
                        .flat_map(|i| ((value(i) as f32) / 65535.0).to_ne_bytes())
                        .collect(),
                };
                let mut whole = vec![0u8; n * bpp];
                c.try_convert_row(&src, &mut whole, n as u32).unwrap();
                let mut parts = vec![0u8; n * bpp];
                for i in 0..n {
                    c.try_convert_row(
                        &src[i * bpp..(i + 1) * bpp],
                        &mut parts[i * bpp..(i + 1) * bpp],
                        1,
                    )
                    .unwrap();
                }
                if whole != parts {
                    let wrong = whole.iter().zip(&parts).filter(|(x, y)| x != y).count();
                    failures.push(format!(
                        "{:?} {transfer_in:?}>{transfer_out:?} {a:?}>{b:?}: {wrong} bytes",
                        base.pixel_format()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");

    // The pixel that first showed it.
    let from = D::RGB8_SRGB.with_primaries(P::DisplayP3);
    let mut c = RowConverter::new(from, D::RGB8_SRGB).unwrap();
    let src: Vec<u8> = [12u8, 153, 195].repeat(9);
    let mut out = vec![0u8; 27];
    c.try_convert_row(&src, &mut out, 9).unwrap();
    assert_eq!(out[..3], out[24..27]);
}

#[test]
fn unknown_input_still_encodes_as_the_target_says() {
    let from = D::RGB8_SRGB
        .with_transfer(T::Unknown)
        .with_primaries(P::DisplayP3);
    let to = D::RGB8_SRGB
        .with_transfer(T::Linear)
        .with_primaries(P::Bt709);
    let mut out = [0u8; 3];
    RowConverter::new(from, to)
        .unwrap()
        .try_convert_row(&[128, 128, 128], &mut out, 1)
        .unwrap();
    assert_eq!(out, [55, 55, 55]);
}

#[test]
fn opposite_ink_polarity_is_not_layout_compatible() {
    use zenpixels::PixelFormat as F;
    assert!(
        !F::CmykF32
            .descriptor()
            .layout_compatible(F::CmykF32Inverted.descriptor())
    );
    assert!(
        F::CmykF32
            .descriptor()
            .layout_compatible(F::CmykF32.descriptor())
    );
}

#[test]
fn same_color_sees_out_of_range_channels() {
    use zenpixels_convert::ColorConvertExt;
    let color = |r: f32| {
        zenpixels::Color::try_new(
            D::RGBF32_LINEAR.with_primaries(P::Bt709),
            bytemuck::cast_slice(&[r, 0.0, 0.0]),
        )
        .unwrap()
    };
    assert!(!color(-0.5).same_color(&color(0.0)));
}
