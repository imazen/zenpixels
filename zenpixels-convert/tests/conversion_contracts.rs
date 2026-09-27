// Regression cases from the API contract review.
#[cfg(test)]
mod tests {
    use zenpixels_convert::*;
    #[test]
    fn known_transfer_is_converted() {
        let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
        let pixels = [128u8; 3];
        let out = adapt::adapt_for_encode_cow(&pixels, src, 1, 1, 3, &[PixelDescriptor::RGB8_SRGB])
            .unwrap();
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
    fn rgba_to_grayalpha_preserves_alpha() {
        let src = PixelDescriptor::RGBA8_SRGB;
        let dst = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let mut converter = RowConverter::new(src, dst).unwrap();
        let mut out = [0u8; 2];
        converter.convert_row(&[200, 30, 10, 99], &mut out, 1);
        assert_eq!(out, [65, 99]);
        assert!(!converter.is_identity());
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
    fn opaque_only_policy_requires_preflight() {
        let options = policy::ConvertOptions::permissive()
            .with_alpha_policy(policy::AlphaPolicy::DiscardIfOpaque);
        assert!(
            RowConverter::new_explicit(
                PixelDescriptor::RGBA8_SRGB,
                PixelDescriptor::RGB8_SRGB,
                &options
            )
            .is_err()
        );
    }
    #[test]
    fn composite_to_gray_includes_background() {
        let options = policy::ConvertOptions::permissive().with_alpha_policy(
            policy::AlphaPolicy::CompositeOnto {
                r: 255,
                g: 255,
                b: 255,
            },
        );
        let dst = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let mut converter =
            RowConverter::new_explicit(PixelDescriptor::RGBA8_SRGB, dst, &options).unwrap();
        let mut out = [123u8; 1];
        converter.convert_row(&[0, 0, 0, 0], &mut out, 1);
        assert_eq!(
            out,
            [255],
            "transparent black over white must produce white"
        );
    }
    #[test]
    fn premul_transfer_unassociates_in_source_domain() {
        let src = PixelDescriptor::RGBAF32_LINEAR
            .with_transfer(TransferFunction::Srgb)
            .with_alpha(Some(AlphaMode::Premultiplied));
        let dst = PixelDescriptor::RGBAF32_LINEAR;
        let mut converter = RowConverter::new(src, dst).unwrap();
        let values = [0.25f32, 0.25, 0.25, 0.5];
        let mut out = [0f32; 4];
        converter.convert_row(
            bytemuck::cast_slice(&values),
            bytemuck::cast_slice_mut(&mut out),
            1,
        );
        assert!((out[0] - 0.214041).abs() < 1e-5, "got {}", out[0]);
        assert!((TransferFunction::Srgb.linearize(0.5) - 0.214041).abs() < 1e-5);
    }
    #[test]
    fn adobe_oklab_plan_refuses_unsupported_primaries() {
        let src = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb);
        let dst = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::AdobeRgb);
        assert!(RowConverter::new(src, dst).is_err());
    }
    #[cfg(feature = "cms-moxcms")]
    #[test]
    fn moxcms_crossdepth_refused_at_planning() {
        let from = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        assert!(
            RowConverter::new_explicit_with_cms(
                from,
                PixelDescriptor::RGBF32_LINEAR,
                &policy::ConvertOptions::permissive(),
                Some(&MoxCms)
            )
            .is_err()
        );
    }
}
