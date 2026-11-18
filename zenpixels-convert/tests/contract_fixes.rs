use zenpixels_convert::PixelSliceConvertExt;
use zenpixels_convert::adapt::adapt_raw_rows_for_encode;
use zenpixels_convert::*;
#[test]
fn known_transfer_is_converted() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let pixels = [128u8; 3];
    let out = adapt_raw_rows_for_encode(
        &pixels,
        src,
        1,
        1,
        3,
        &[PixelDescriptor::RGB8_SRGB],
        &zenpixels_convert::ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardUnchecked),
    )
    .unwrap();
    assert_eq!(
        out.as_slice().descriptor().transfer(),
        TransferFunction::Srgb
    );
    assert!(out.as_slice().row(0)[0] >= 187);
    let mut real = crate::test_conversion::new_worker(src, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut converted = [0; 3];
    real.try_convert_row(&pixels, &mut converted, 1).unwrap();
    assert!(converted[0] >= 187, "actual linear→sRGB must encode ~188");
}
#[test]
fn composition_preserves_final_samples_of_materialized_stages() {
    let f = PixelDescriptor::RGBF32_LINEAR;
    let u = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let mut a = crate::test_conversion::new_worker(f, u).unwrap();
    let mut b = crate::test_conversion::new_worker(u, f).unwrap();
    let mut composed = crate::test_conversion::new_plan(f, u)
        .unwrap()
        .then(crate::test_conversion::new_plan(u, f).unwrap())
        .unwrap()
        .prepare(1)
        .unwrap();
    let src = [0.1234567f32; 3];
    let mut quantized = [0u8; 3];
    let mut separate = [0f32; 3];
    let mut together = [0f32; 3];
    a.try_convert_row(bytemuck::cast_slice(&src), &mut quantized, 1)
        .unwrap();
    b.try_convert_row(&quantized, bytemuck::cast_slice_mut(&mut separate), 1)
        .unwrap();
    composed
        .try_convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut together),
            1,
        )
        .unwrap();
    assert!(!composed.is_identity());
    assert_ne!(together, src);
    assert_eq!(together, separate);
}
#[test]
fn gamma22_scalar_matches_adobe_gamma() {
    let g22 = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
        ColorPrimaries::AdobeRgb,
    );
    let lin = g22.with_transfer(TransferFunction::Linear);
    let mut to_lin = crate::test_conversion::new_worker(g22, lin).unwrap();
    let mut to_g22 = crate::test_conversion::new_worker(lin, g22).unwrap();
    let src = [0.5f32; 3];
    let mut mid = [0.0f32; 3];
    let mut back = [0.0f32; 3];
    to_lin
        .try_convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut mid),
            1,
        )
        .unwrap();
    to_g22
        .try_convert_row(
            bytemuck::cast_slice(&mid),
            bytemuck::cast_slice_mut(&mut back),
            1,
        )
        .unwrap();
    assert!((mid[0] - 0.5f32.powf(563.0 / 256.0)).abs() < 1e-6);
    assert!((back[0] - 0.5).abs() < 1e-6);
}
#[test]
fn f16_premultiply_rounds_below_midpoint_to_zero() {
    let src = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let dst = src.with_alpha(Some(AlphaMode::Premultiplied));
    let mut converter = crate::test_conversion::new_worker(src, dst).unwrap();
    let samples = [1u16, 0, 0, 0x3600]; // minimum subnormal color; alpha .375
    let mut out = [0u16; 4];
    converter
        .try_convert_row(
            bytemuck::cast_slice(&samples),
            bytemuck::cast_slice_mut(&mut out),
            1,
        )
        .unwrap();
    assert_eq!(
        out[0], 0,
        "minimum-subnormal times .375 is below midpoint and should round to zero"
    );
}
#[test]
fn orientation_preserves_color_context() {
    use std::sync::Arc;
    use zenpixels::{Cicp, ColorContext, Orientation, PixelBuffer, PixelDescriptor};
    use zenpixels_convert::PixelBufferConvertExt;
    let ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    let src = PixelBuffer::new(2, 3, PixelDescriptor::RGB8).with_color_context(ctx);
    let out = src.as_slice().oriented(Orientation::Rotate90).unwrap();
    assert!(Arc::ptr_eq(
        out.color_context().unwrap(),
        src.color_context().unwrap()
    ));
    let src = src.into_oriented(Orientation::Rotate90).unwrap();
    assert!(src.color_context().is_some());
}

#[test]
fn explicit_adapter_encodes_known_transfer() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let pixels = [128u8; 3];
    let out = adapt_raw_rows_for_encode(
        &pixels,
        src,
        1,
        1,
        3,
        &[PixelDescriptor::RGB8_SRGB],
        &policy::ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
    )
    .unwrap();
    assert!(out.as_slice().row(0)[0] >= 187);
}

#[path = "support/conversion.rs"]
mod test_conversion;
