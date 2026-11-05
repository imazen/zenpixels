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
    for sample in dst.chunks_exact(2) {
        let code = u16::from_ne_bytes([sample[0], sample[1]]);
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
