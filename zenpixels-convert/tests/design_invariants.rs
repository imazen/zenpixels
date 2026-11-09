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
