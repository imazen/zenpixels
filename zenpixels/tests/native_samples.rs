//! Native TIFF/IMIJ carriage must not normalize, narrow, or canonicalize samples.
use zenpixels::{ChannelLayout, ChannelType, PixelBuffer, PixelFormat};

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
