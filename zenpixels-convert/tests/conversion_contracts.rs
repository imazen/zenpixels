// Regression cases from the API contract review.
#[cfg(test)]
mod tests {
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
    fn rgba_to_grayalpha_preserves_alpha() {
        let src = PixelDescriptor::RGBA8_SRGB;
        let dst = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let mut converter = crate::test_conversion::new_worker(src, dst).unwrap();
        let mut out = [0u8; 2];
        converter
            .try_convert_row(&[200, 30, 10, 99], &mut out, 1)
            .unwrap();
        assert_eq!(out, [65, 99]);
        assert!(!converter.is_identity());
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
    fn opaque_only_policy_requires_preflight() {
        let options = policy::ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
            .with_alpha_policy(policy::AlphaPolicy::DiscardIfOpaque);
        assert!(
            crate::test_conversion::explicit_worker(
                PixelDescriptor::RGBA8_SRGB,
                PixelDescriptor::RGB8_SRGB,
                &options
            )
            .is_err()
        );
    }
    #[test]
    fn composite_to_gray_includes_background() {
        let options = policy::ConvertOptions::retain_channels()
            .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
            .with_alpha_policy(policy::AlphaPolicy::CompositeOnto {
                background: zenpixels::Color::srgb8(255, 255, 255),
                reference_white: None,
            });
        let dst = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let mut converter =
            crate::test_conversion::explicit_worker(PixelDescriptor::RGBA8_SRGB, dst, &options)
                .unwrap();
        let mut out = [123u8; 1];
        converter
            .try_convert_row(&[0, 0, 0, 0], &mut out, 1)
            .unwrap();
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
        let mut converter = crate::test_conversion::new_worker(src, dst).unwrap();
        let values = [0.25f32, 0.25, 0.25, 0.5];
        let mut out = [0f32; 4];
        converter
            .try_convert_row(
                bytemuck::cast_slice(&values),
                bytemuck::cast_slice_mut(&mut out),
                1,
            )
            .unwrap();
        assert!((out[0] - 0.214041).abs() < 1e-5, "got {}", out[0]);
        assert!((linear_srgb::precise::srgb_to_linear(0.5) - 0.214041).abs() < 1e-5);
    }
    #[test]
    fn unknown_oklab_plan_refuses_unsupported_primaries() {
        let src = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Unknown);
        let dst = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::Unknown);
        assert!(crate::test_conversion::new_worker(src, dst).is_err());

        // Adobe RGB is supported natively via exact rational XYZ/Oklab matrices.
        let adobe = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::AdobeRgb);
        let adobe_oklab = dst.with_primaries(ColorPrimaries::AdobeRgb);
        assert!(crate::test_conversion::new_worker(adobe, adobe_oklab).is_ok());
    }
    #[cfg(feature = "cms-moxcms")]
    #[test]
    fn moxcms_crossdepth_gives_the_plans_result() {
        // A backend with U8 and F32 paths has no U8-to-F32 path, so it declines, and the bytes must
        // never land in float storage unconverted.
        let from = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        let to = PixelDescriptor::RGBF32_LINEAR;
        let src = [255u8, 128, 0];
        let convert = |cms: Option<&dyn zenpixels_convert::cms::PluggableCms>| {
            let mut out = [0f32; 3];
            crate::test_conversion::cms_worker(
                from,
                to,
                &policy::ConvertOptions::retain_channels()
                    .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
                    .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
                    .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
                cms,
            )
            .unwrap()
            .try_convert_row(&src, bytemuck::cast_slice_mut(&mut out), 1)
            .unwrap();
            out
        };
        assert_eq!(convert(Some(&MoxCms)), convert(None));
    }
}

#[path = "support/conversion.rs"]
mod test_conversion;
