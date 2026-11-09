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
