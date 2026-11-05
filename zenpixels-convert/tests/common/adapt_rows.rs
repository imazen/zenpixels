//! `adapt_for_encode` from raw rows and geometry, as the adapter tests state their inputs.

use whereat::At;
use zenpixels::{ConvertOptions, PixelCow, PixelDescriptor, PixelSlice};
use zenpixels_convert::{ConvertError, PixelSliceConvertExt};

pub fn adapt_rows<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
    options: &ConvertOptions,
) -> Result<PixelCow<'a>, At<ConvertError>> {
    PixelSlice::new(data, width, rows, stride, descriptor)
        .map_err(|e| e.map_error(ConvertError::from))?
        .adapt_for_encode(supported, options)
}
