use zenpixels::*;
use zenpixels_convert::PixelBufferConvertExt;

#[test]
fn wide_channel_removal_preserves_every_payload_bit() {
    let from = PixelDescriptor::new(
        ChannelType::U64,
        ChannelLayout::Rgba,
        Some(AlphaMode::Opaque),
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::new(
        ChannelType::U64,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let words = [u64::MAX, 1 << 63, 42, 0, 77, 88, 99, u64::MAX];
    let mut buffer =
        PixelBuffer::from_vec(bytemuck::cast_slice(&words).to_vec(), 2, 1, from).unwrap();
    buffer.try_adapt_in_place(to).unwrap();
    assert_eq!(
        buffer.as_slice().row(0),
        bytemuck::cast_slice(&[u64::MAX, 1 << 63, 42, 77, 88, 99])
    );
}

#[test]
fn layout_cannot_bypass_luma_policy() {
    for layout in [
        ChannelLayout::Rgb,
        ChannelLayout::Bgr,
        ChannelLayout::Rgba,
        ChannelLayout::Bgra,
    ] {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            layout,
            layout.has_alpha().then_some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let options =
            ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        assert!(
            zenpixels_convert::RowConverter::new_explicit(
                from,
                PixelDescriptor::GRAY8_SRGB,
                &options
            )
            .is_err()
        );
    }
}

#[test]
fn composed_matte_to_gray_has_complete_operation_contracts() {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(255, 255, 255),
        reference_white: None,
    });
    for from in [
        PixelDescriptor::BGRA8_SRGB,
        PixelDescriptor::RGBA8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
    ] {
        for to in [
            PixelDescriptor::GRAY8_SRGB,
            PixelFormat::GrayF32
                .descriptor()
                .with_transfer(TransferFunction::Srgb),
        ] {
            let mut converter =
                zenpixels_convert::RowConverter::new_explicit(from, to, &options).unwrap();
            let mut output = [0f32; 4];
            converter
                .try_convert_row(&[0, 0, 0, 0], bytemuck::cast_slice_mut(&mut output), 1)
                .unwrap();
            if to.channel_type() == ChannelType::F32 {
                assert!((output[0] - 1.0).abs() < 0.0001);
            } else {
                assert_eq!(bytemuck::cast_slice::<_, u8>(&output)[0], 255);
            }
        }
    }
}

#[test]
fn allocating_into_and_adaptation_share_output_interpretation() {
    use std::sync::Arc;
    use zenpixels_convert::PixelSliceConvertExt;
    let pq = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Pq);
    let data = [0.5f32; 6];
    for white in [hdr::DiffuseWhite::BT2408, hdr::DiffuseWhite::new(1000.0)] {
        let source = PixelSlice::new(bytemuck::cast_slice(&data), 1, 2, 12, pq)
            .unwrap()
            .with_color_context(Arc::new(ColorContext::default().with_diffuse_white(white)));
        let target = PixelDescriptor::RGBF32_LINEAR;
        let owned = source.convert_to(target).unwrap();
        let mut into = [0f32; 6];
        let destination =
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut into), 1, 2, 12, target).unwrap();
        let destination = source.convert_into(destination).unwrap();
        let adapted = source
            .adapt_for_encode(&[target], &ConvertOptions::permissive())
            .unwrap();
        for y in 0..2 {
            assert_eq!(owned.as_slice().row(y), destination.row(y));
            assert_eq!(owned.as_slice().row(y), adapted.as_slice().row(y));
        }
        assert_eq!(owned.color_context().unwrap().diffuse_white, Some(white));
        assert_eq!(
            destination.color_context().unwrap().diffuse_white,
            Some(white)
        );
        assert_eq!(
            adapted.as_slice().color_context().unwrap().diffuse_white,
            Some(white)
        );
    }
}

#[test]
fn f16_alpha_payload_survives_association_at_every_position() {
    for layout in [ChannelLayout::Rgba, ChannelLayout::GrayAlpha] {
        let channels = layout.channels();
        for alpha in [0x7c01u16, 0x7e55, 0x8000, 0, 0x3800, 0x3c00] {
            for (a, b) in [
                (AlphaMode::Straight, AlphaMode::Premultiplied),
                (AlphaMode::Premultiplied, AlphaMode::Straight),
            ] {
                let from = PixelDescriptor::new(
                    ChannelType::F16,
                    layout,
                    Some(a),
                    TransferFunction::Linear,
                );
                let to = from.with_alpha(Some(b));
                let mut source = vec![0x3800u16; channels * 19];
                for pixel in source.chunks_exact_mut(channels) {
                    pixel[channels - 1] = alpha;
                }
                let mut output = vec![0u16; source.len()];
                let mut converter = zenpixels_convert::RowConverter::new(from, to).unwrap();
                converter
                    .try_convert_row(
                        bytemuck::cast_slice(&source),
                        bytemuck::cast_slice_mut(&mut output),
                        19,
                    )
                    .unwrap();
                for pixel in output.chunks_exact(channels) {
                    assert_eq!(pixel[channels - 1], alpha);
                }
            }
        }
    }
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn histogram_shape_and_small_positive_ranks_are_consistent() {
    use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
    for value in [0.01f32, 0.1, 0.3, 0.7, 1.0] {
        let data = [value; 24];
        let bytes = bytemuck::cast_slice(&data);
        let wide = PixelSlice::new(bytes, 8, 1, 96, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let tall = PixelSlice::new(bytes, 1, 8, 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
        for method in [LightLevelMethod::MaxRgb, LightLevelMethod::LuminanceBt2020] {
            let a = hdr::ContentLightLevel::measure_histogram(
                wide.clone(),
                hdr::DiffuseWhite::BT2408,
                method,
            )
            .unwrap();
            let b = hdr::ContentLightLevel::measure_histogram(
                tall.clone(),
                hdr::DiffuseWhite::BT2408,
                method,
            )
            .unwrap();
            assert_eq!(a.bins(), b.bins());
            assert!(a.percentile(0.01) > value * 203.0 * 0.9);
            assert!(a.percentile(0.01) <= value * 203.0 * 1.01);
        }
    }
}
