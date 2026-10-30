use zenpixels_convert::*;
#[test]
fn known_transfer_is_converted() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let pixels = [128u8; 3];
    let out =
        adapt::adapt_for_encode_cow(&pixels, src, 1, 1, 3, &[PixelDescriptor::RGB8_SRGB]).unwrap();
    assert_eq!(
        out.as_slice().descriptor().transfer(),
        TransferFunction::Srgb
    );
    assert!(out.as_slice().row(0)[0] >= 187);
    let mut real = RowConverter::new(src, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut converted = [0; 3];
    real.convert_row(&pixels, &mut converted, 1);
    assert!(converted[0] >= 187, "actual linear→sRGB must encode ~188");
}
#[test]
fn compose_optimizes_final_output_by_default() {
    let f = PixelDescriptor::RGBF32_LINEAR;
    let u = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let mut a = RowConverter::new(f, u).unwrap();
    let mut b = RowConverter::new(u, f).unwrap();
    let mut composed = a.compose(&b).unwrap();
    let src = [0.1234567f32; 3];
    let mut quantized = [0u8; 3];
    let mut separate = [0f32; 3];
    let mut together = [0f32; 3];
    a.convert_row(bytemuck::cast_slice(&src), &mut quantized, 1);
    b.convert_row(&quantized, bytemuck::cast_slice_mut(&mut separate), 1);
    composed.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut together),
        1,
    );
    assert!(composed.is_identity());
    assert_eq!(together, src);
    assert_ne!(together, separate);
}
#[test]
fn gamma22_scalar_matches_adobe_gamma() {
    let linear = TransferFunction::Gamma22.linearize(0.5);
    assert!((linear - 0.5f32.powf(563.0 / 256.0)).abs() < 1e-6);
    assert!((TransferFunction::Gamma22.delinearize(linear) - 0.5).abs() < 1e-6);
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
    let mut converter = RowConverter::new(src, dst).unwrap();
    let samples = [1u16, 0, 0, 0x3600]; // minimum subnormal color; alpha .375
    let mut out = [0u16; 4];
    converter.convert_row(
        bytemuck::cast_slice(&samples),
        bytemuck::cast_slice_mut(&mut out),
        1,
    );
    assert_eq!(
        out[0], 0,
        "minimum-subnormal times .375 is below midpoint and should round to zero"
    );
}
#[test]
fn orientation_preserves_color_context() {
    use std::sync::Arc;
    use zenpixels::{Cicp, ColorContext, Orientation, PixelBuffer, PixelDescriptor};
    use zenpixels_convert::orient::{apply_orientation, apply_orientation_in_place};
    let ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    let mut src = PixelBuffer::new(2, 3, PixelDescriptor::RGB8).with_color_context(ctx);
    let out = apply_orientation(src.as_slice(), Orientation::Rotate90);
    assert!(Arc::ptr_eq(
        out.color_context().unwrap(),
        src.color_context().unwrap()
    ));
    apply_orientation_in_place(&mut src, Orientation::Rotate90).unwrap();
    assert!(src.color_context().is_some());
}

#[test]
fn explicit_adapter_encodes_known_transfer() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let pixels = [128u8; 3];
    let out = adapt::adapt_for_encode_explicit_cow(
        &pixels,
        src,
        1,
        1,
        3,
        &[PixelDescriptor::RGB8_SRGB],
        &policy::ConvertOptions::permissive(),
    )
    .unwrap();
    assert!(out.as_slice().row(0)[0] >= 187);
}
