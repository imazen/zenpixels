//! Whole-frame pixel conversion with transfer function awareness.
//!
//! This module provides [`convert_pixels`], a transfer-function-aware
//! pixel format converter that operates on raw byte buffers.

use alloc::vec::Vec;

use crate::adapt::convert_buffer;
use crate::{ConvertError, PixelDescriptor, TransferFunction};

/// Convert raw pixel data to a target [`PixelDescriptor`], applying correct
/// transfer functions when crossing depth boundaries.
///
/// # Arguments
///
/// * `data` - Source pixel bytes (packed, stride = width * bpp).
/// * `descriptor` - Format of the source data (without transfer info).
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `source_transfer` - The transfer function of the source data, resolved
///   from CICP/ICC metadata.
/// * `target` - The desired output format (including transfer function).
///
/// # Returns
///
/// The converted pixel data with format metadata.
pub fn convert_pixels(
    data: &[u8],
    descriptor: PixelDescriptor,
    width: u32,
    height: u32,
    source_transfer: TransferFunction,
    target: PixelDescriptor,
) -> Result<ConvertedPixels, ConvertError> {
    let src_desc = descriptor.with_transfer(source_transfer);

    if src_desc == target {
        return Ok(ConvertedPixels {
            data: data.to_vec(),
            descriptor: target,
            width,
            height,
        });
    }

    let converted = convert_buffer(data, width, height, src_desc, target)?;

    Ok(ConvertedPixels {
        data: converted,
        descriptor: target,
        width,
        height,
    })
}

/// Result of a whole-frame pixel conversion.
pub struct ConvertedPixels {
    /// Raw pixel bytes in the target format.
    pub data: Vec<u8>,
    /// The pixel format of `data`.
    pub descriptor: PixelDescriptor,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}
