//! Native TIFF/IMIJ carriage must not normalize, narrow, or canonicalize samples.
use zenpixels::{ChannelLayout, ChannelType, PixelBuffer, PixelBufferParts, PixelFormat};

fn native_patterns(channel: ChannelType) -> Vec<u8> {
    match channel {
        ChannelType::I8 => [i8::MIN, -1, 0, i8::MAX]
            .into_iter()
            .map(|v| v as u8)
            .collect(),
        ChannelType::I16 => [i16::MIN, -1, 0, i16::MAX]
            .into_iter()
            .flat_map(i16::to_ne_bytes)
            .collect(),
        ChannelType::U32 => [0, 1, 0x0100_0001, u32::MAX]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect(),
        ChannelType::I32 => [i32::MIN, -1, 0, i32::MAX]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect(),
        ChannelType::U64 => [0, 1, 0x0020_0000_0000_0001, u64::MAX]
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect(),
        ChannelType::I64 => [i64::MIN, -1, 0, i64::MAX]
            .into_iter()
            .flat_map(i64::to_ne_bytes)
            .collect(),
        // Signed zero, signaling/quiet NaN payloads and a subnormal are data.
        ChannelType::F64 => [
            0x8000_0000_0000_0000u64,
            0x7ff0_0000_0000_0001,
            0x7ff8_1234_5678_9abc,
            1,
        ]
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .collect(),
        _ => panic!("not a native-sample fixture"),
    }
}

#[test]
fn external_strided_allocation_adopts_exactly_and_is_recoverable_on_refusal() {
    let descriptor = PixelFormat::from_parts(ChannelType::U64, ChannelLayout::Rgb, None)
        .unwrap()
        .descriptor();
    let stride = 8 * 3 * 5;
    let bytes: Vec<u8> = (0..stride * 3).map(|i| (i * 73 + 19) as u8).collect();
    let pointer = bytes.as_ptr();
    let expected = bytes.clone();
    let mut parts = PixelBufferParts::new(bytes, 3, 3, stride, descriptor);
    // Deliberately impossible, then recover and correct the same allocation.
    parts.stride_bytes = 1;
    let mut error = PixelBuffer::try_from_parts(parts).unwrap_err();
    let mut parts = error.take_parts().unwrap();
    assert_eq!(parts.data.as_ptr(), pointer);
    assert_eq!(parts.data, expected);
    parts.stride_bytes = stride;
    let buffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr(), pointer);
    for y in 0..3 {
        let at = y as usize * stride;
        assert_eq!(buffer.as_slice().row(y), &expected[at..at + 8 * 3 * 3]);
    }
}

#[test]
fn native_storage_survives_strides_subviews_and_owned_handoff() {
    for channel in [
        ChannelType::I8,
        ChannelType::I16,
        ChannelType::U32,
        ChannelType::I32,
        ChannelType::U64,
        ChannelType::I64,
        ChannelType::F64,
    ] {
        for layout in [
            ChannelLayout::Gray,
            ChannelLayout::GrayAlpha,
            ChannelLayout::Rgb,
            ChannelLayout::Rgba,
        ] {
            let alpha = layout.has_alpha().then_some(zenpixels::AlphaMode::Straight);
            let format = PixelFormat::from_parts(channel, layout, alpha).unwrap();
            let descriptor = format.descriptor();
            assert_eq!(descriptor.channel_type(), channel);
            assert_eq!(descriptor.layout(), layout);
            assert_eq!(descriptor.alpha(), alpha);
            let mut buffer = PixelBuffer::try_new_simd_aligned(5, 3, descriptor, 64).unwrap();
            let patterns = native_patterns(channel);
            let row_bytes = 5 * layout.channels() * channel.byte_size();
            let expected: Vec<u8> = (0..row_bytes)
                .map(|i| patterns[i % patterns.len()])
                .collect();
            for y in 0..3 {
                buffer.as_slice_mut().row_mut(y).copy_from_slice(&expected);
            }
            let parts = buffer.into_parts();
            assert!(parts.stride_bytes >= row_bytes);
            let buffer = PixelBuffer::try_from_parts(parts).unwrap();
            for y in 0..3 {
                assert_eq!(buffer.as_slice().row(y), expected);
            }
            let cropped = buffer.crop_copy(1, 1, 3, 2);
            let pixel = layout.channels() * channel.byte_size();
            for y in 0..2 {
                assert_eq!(cropped.as_slice().row(y), &expected[pixel..4 * pixel]);
            }
        }
    }
}

#[test]
fn cmyk_f32_carries_strides_profiles_and_every_sample_bit() {
    use std::sync::Arc;
    use zenpixels::{ColorContext, ColorModel, ColorPrimaries, PixelDescriptor, TransferFunction};
    let descriptor = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Cmyk,
        None,
        TransferFunction::Unknown,
    )
    .with_primaries(ColorPrimaries::Unknown);
    assert_eq!(descriptor.format, PixelFormat::CmykF32);
    assert_eq!(descriptor.color_model(), ColorModel::Cmyk);
    assert_eq!(descriptor.bytes_per_pixel(), 16);
    assert!(!descriptor.has_alpha());
    assert_eq!(PixelFormat::CmykF32.default_alpha(), None);
    assert_eq!(PixelFormat::CmykF32.descriptor().format, descriptor.format);
    for descriptor in [descriptor, PixelFormat::CmykF32Inverted.descriptor()] {
        assert_eq!(descriptor.layout(), ChannelLayout::Cmyk);
        assert_eq!(descriptor.channel_type(), ChannelType::F32);
        assert_eq!(descriptor.color_model(), ColorModel::Cmyk);
        assert_eq!(descriptor.bytes_per_pixel(), 16);
        assert_eq!(descriptor.alpha(), None);
        let patterns = [0x80000000u32, 0x7fc01234, 1, 0x3f800001];
        let row: Vec<u8> = patterns
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .cycle()
            .take(48)
            .collect();
        let mut data = vec![0xa5; 64 * 3];
        for y in 0..3 {
            data[y * 64..y * 64 + 48].copy_from_slice(&row);
        }
        let address = data.as_ptr();
        let expected = data.clone();
        let profile: Arc<[u8]> = Arc::from(&b"opaque-native-CMYK-profile"[..]);
        let mut parts = PixelBufferParts::new(data, 3, 3, 64, descriptor);
        parts.color_context = Some(Arc::new(ColorContext::from_icc(profile.clone())));
        let image = PixelBuffer::try_from_parts(parts).unwrap();
        assert_eq!(image.as_slice().row(0).as_ptr(), address);
        for y in 0..3 {
            assert_eq!(image.as_slice().row(y), row);
        }
        let parts = image.into_parts();
        assert_eq!(parts.data, expected);
        assert_eq!(parts.data.as_ptr(), address);
        assert_eq!(
            parts.color_context.unwrap().icc.as_ref().unwrap().as_ref(),
            profile.as_ref()
        );
    }
}

#[test]
fn native_combinations_have_one_generic_representation() {
    use zenpixels::AlphaMode;
    const NATIVE: [ChannelType; 7] = [
        ChannelType::I8,
        ChannelType::I16,
        ChannelType::U32,
        ChannelType::I32,
        ChannelType::U64,
        ChannelType::I64,
        ChannelType::F64,
    ];
    const LAYOUTS: [(ChannelLayout, &str); 4] = [
        (ChannelLayout::Gray, "Gray"),
        (ChannelLayout::GrayAlpha, "GrayA"),
        (ChannelLayout::Rgb, "Rgb"),
        (ChannelLayout::Rgba, "Rgba"),
    ];
    let mut seen = Vec::new();
    for channel in NATIVE {
        for (layout, prefix) in LAYOUTS {
            for alpha in [None, Some(AlphaMode::Straight), Some(AlphaMode::Undefined)] {
                let format = PixelFormat::from_parts(channel, layout, alpha).unwrap();
                assert!(matches!(format, PixelFormat::Generic(_)), "{format:?}");
                assert_eq!(format.channel_type(), channel);
                assert_eq!(format.layout(), layout);
                assert_eq!(format.name(), format!("{prefix}{channel}"));
                assert_eq!(format.to_string(), format.name());
                assert_eq!(
                    format.bytes_per_pixel(),
                    layout.channels() * channel.byte_size()
                );
                assert_eq!(
                    format.default_alpha(),
                    layout.has_alpha().then_some(AlphaMode::Straight)
                );
                let descriptor = format.descriptor();
                assert_eq!(descriptor.format, format);
                assert!(descriptor.layout_compatible(format.descriptor()));
                if alpha.is_none() {
                    assert!(!seen.contains(&format));
                    seen.push(format);
                }
            }
        }
    }
    assert_eq!(seen.len(), 28);
    // Combinations with a named variant never resolve generically.
    for channel in [
        ChannelType::U8,
        ChannelType::U16,
        ChannelType::F16,
        ChannelType::F32,
    ] {
        for (layout, _) in LAYOUTS {
            let format = PixelFormat::from_parts(channel, layout, None).unwrap();
            assert!(!matches!(format, PixelFormat::Generic(_)), "{format:?}");
        }
    }
    // Layouts without generic carriage stay unrepresentable.
    for channel in NATIVE {
        for layout in [
            ChannelLayout::Bgra,
            ChannelLayout::Cmyk,
            ChannelLayout::Oklab,
        ] {
            assert_eq!(PixelFormat::from_parts(channel, layout, None), None);
        }
    }
}
