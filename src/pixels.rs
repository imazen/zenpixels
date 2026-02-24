//! Whole-frame [`PixelData`] conversion with transfer function awareness.
//!
//! This module provides [`convert_pixels`], a transfer-function-aware
//! replacement for `PixelData::to_rgb8()` and similar methods.

use zencodec_types::{PixelDescriptor, PixelData};

use crate::adapt::convert_buffer;
use crate::error::ConvertError;

/// Convert [`PixelData`] to a target [`PixelDescriptor`], applying correct
/// transfer functions when crossing depth boundaries.
///
/// Unlike `PixelData::to_rgb8()` which does naive `v * 255` clamping,
/// this function applies the sRGB OETF when converting f32 linear → u8 sRGB,
/// and the sRGB EOTF when converting u8 sRGB → f32 linear.
///
/// # Arguments
///
/// * `pixels` - Source pixel data.
/// * `source_transfer` - The transfer function of the source data. Since
///   `PixelData::descriptor()` always returns `Unknown`, the caller must
///   resolve this from CICP/ICC metadata.
/// * `target` - The desired output format (including transfer function).
///
/// # Returns
///
/// The raw bytes of the converted image. Width/height are preserved
/// from the source.
pub fn convert_pixels(
    pixels: &PixelData,
    source_transfer: zencodec_types::TransferFunction,
    target: PixelDescriptor,
) -> Result<ConvertedPixels, ConvertError> {
    let width = pixels.width();
    let height = pixels.height();
    let src_desc = pixels.descriptor().with_transfer(source_transfer);

    if src_desc == target {
        // Already the right format — just copy the bytes.
        return Ok(ConvertedPixels {
            data: pixels.to_bytes(),
            descriptor: target,
            width,
            height,
        });
    }

    let src_bytes = pixels.to_bytes();
    let converted = convert_buffer(&src_bytes, width, height, src_desc, target)?;

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
    pub data: alloc::vec::Vec<u8>,
    /// The pixel format of `data`.
    pub descriptor: PixelDescriptor,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}
