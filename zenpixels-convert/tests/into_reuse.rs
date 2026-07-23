use zenpixels::{Orientation, PixelBuffer, PixelDescriptor};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::load_bearing::PixelBufferLoadBearingExt;
use zenpixels_convert::orient::into_oriented;

#[test]
fn into_converted_returns_the_rewritten_buffer() {
    let buffer = PixelBuffer::new(2, 1, PixelDescriptor::RGBA8_SRGB);
    let converted = buffer.into_converted(PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(converted.descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!((converted.width(), converted.height()), (2, 1));
}

#[test]
fn into_oriented_returns_adopted_geometry() {
    let buffer = PixelBuffer::new(2, 3, PixelDescriptor::RGB8_SRGB);
    let oriented = into_oriented(buffer, Orientation::Rotate90).unwrap();
    assert_eq!((oriented.width(), oriented.height()), (3, 2));
}

#[test]
fn into_load_bearing_returns_reduced_buffer() {
    let mut buffer = PixelBuffer::new(1, 1, PixelDescriptor::RGBA8_SRGB);
    buffer
        .as_slice_mut()
        .row_mut(0)
        .copy_from_slice(&[40, 40, 40, 255]);
    let reduced = buffer.into_load_bearing_format(true);
    assert_eq!(reduced.descriptor(), PixelDescriptor::GRAY8_SRGB);
    assert_eq!(reduced.as_slice().row(0), &[40]);
}
