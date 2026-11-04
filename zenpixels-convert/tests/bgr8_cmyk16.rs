//! The formats that the kernels don't handle must refuse, never pass through as RGB.

use zenpixels::{PixelDescriptor as D, TransferFunction as T};
use zenpixels_convert::RowConverter;

const SRGB_BGR: D = D::BGR8.with_transfer(T::Srgb);

/// A converter either refuses, or swaps the channels. Copying B into R silently is the
/// failure this guards against.
#[test]
fn bgr8_conversions_refuse_or_swap_channels() {
    let bgr = [10u8, 20, 30, 40, 50, 60];
    for (from, to) in [(SRGB_BGR, D::RGB8_SRGB), (SRGB_BGR, D::RGBA8_SRGB)] {
        match RowConverter::new(from, to) {
            Err(_) => {}
            Ok(mut converter) => {
                let channels = to.bytes_per_pixel();
                let mut out = vec![0u8; 2 * channels];
                converter.convert_row(&bgr, &mut out, 2);
                assert_eq!(&out[..3], &[30, 20, 10], "{from:?} to {to:?}");
                assert_eq!(
                    &out[channels..channels + 3],
                    &[60, 50, 40],
                    "{from:?} to {to:?}"
                );
            }
        }
    }
    let rgb = [30u8, 20, 10, 60, 50, 40];
    match RowConverter::new(D::RGB8_SRGB, SRGB_BGR) {
        Err(_) => {}
        Ok(mut converter) => {
            let mut out = [0u8; 6];
            converter.convert_row(&rgb, &mut out, 2);
            assert_eq!(out, bgr);
        }
    }
}

/// Changing only the transfer has no kernels either, and must refuse rather than panic.
#[test]
fn bgr8_refuses_a_transfer_change() {
    let linear = D::BGR8.with_transfer(T::Linear);
    assert!(RowConverter::new(SRGB_BGR, linear).is_err());
    assert!(RowConverter::new(linear, SRGB_BGR).is_err());
    let mut out = [0u8; 3];
    RowConverter::new(SRGB_BGR, SRGB_BGR)
        .unwrap()
        .convert_row(&[1, 2, 3], &mut out, 1);
    assert_eq!(out, [1, 2, 3]);
}

#[test]
fn cmyk16_has_no_numeric_conversion() {
    assert!(RowConverter::new(D::CMYK16, D::RGB16_SRGB).is_err());
    assert!(RowConverter::new(D::RGB16_SRGB, D::CMYK16).is_err());
}
