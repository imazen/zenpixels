//! Codec adapter functions.
//!
//! High-level helpers that combine format negotiation with conversion,
//! replacing the per-codec format dispatch if-chains.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use zencodec_types::{PixelDescriptor, PixelSlice};

use crate::converter::RowConverter;
use crate::error::ConvertError;
use crate::negotiate::{best_match, ConvertIntent};

/// Result of format adaptation: the converted data and its descriptor.
pub struct Adapted<'a> {
    /// Pixel data — borrowed if no conversion was needed, owned otherwise.
    pub data: Cow<'a, [u8]>,
    /// The pixel format of `data`.
    pub descriptor: PixelDescriptor,
    /// Width of the pixel data.
    pub width: u32,
    /// Number of rows.
    pub rows: u32,
}

/// Negotiate format and convert a [`PixelSlice`] for encoding.
///
/// Uses [`ConvertIntent::Fastest`] — minimizes conversion cost, which is the
/// right default for codec encoding where the codec knows what format it wants.
///
/// If the input already matches one of the `supported` formats, returns
/// `Cow::Borrowed` (zero-copy). Otherwise, converts to the best match
/// and returns `Cow::Owned`.
///
/// # Errors
///
/// Returns [`ConvertError::EmptyFormatList`] if `supported` is empty.
/// Returns [`ConvertError::NoPath`] if no conversion exists to any supported format.
///
/// # Example
///
/// ```rust,ignore
/// fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
///     let adapted = zenpixels::adapt::adapt_for_encode(
///         &pixels,
///         &Self::SUPPORTED_FORMATS,
///     )?;
///     self.do_encode_native(&adapted.data, adapted.descriptor, adapted.width, adapted.rows)
/// }
/// ```
pub fn adapt_for_encode<'a>(
    pixels: &PixelSlice<'a>,
    supported: &[PixelDescriptor],
) -> Result<Adapted<'a>, ConvertError> {
    adapt_for_encode_with_intent(pixels, supported, ConvertIntent::Fastest)
}

/// Negotiate format and convert a [`PixelSlice`] with intent awareness.
///
/// Like [`adapt_for_encode`], but lets the caller specify a [`ConvertIntent`]
/// to shift format preferences. For example, `ConvertIntent::LinearLight` will
/// prefer f32 Linear targets for gamma-correct resize operations.
///
/// # Errors
///
/// Returns [`ConvertError::EmptyFormatList`] if `supported` is empty.
/// Returns [`ConvertError::NoPath`] if no conversion exists to any supported format.
pub fn adapt_for_encode_with_intent<'a>(
    pixels: &PixelSlice<'a>,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Result<Adapted<'a>, ConvertError> {
    if supported.is_empty() {
        return Err(ConvertError::EmptyFormatList);
    }

    let src_desc = pixels.descriptor();

    // Check for exact match first (zero-copy path).
    if supported.contains(&src_desc) {
        return Ok(Adapted {
            data: pixels.contiguous_bytes(),
            descriptor: src_desc,
            width: pixels.width(),
            rows: pixels.rows(),
        });
    }

    // Also check for transfer-agnostic match: if source has Unknown transfer
    // and a supported format matches on everything except transfer, it's still
    // a zero-copy path (the codec doesn't care about the transfer tag).
    for &target in supported {
        if src_desc.channel_type == target.channel_type
            && src_desc.layout == target.layout
            && src_desc.alpha == target.alpha
        {
            // Layout-compatible, only transfer differs — zero-copy.
            return Ok(Adapted {
                data: pixels.contiguous_bytes(),
                descriptor: target,
                width: pixels.width(),
                rows: pixels.rows(),
            });
        }
    }

    // Need conversion — pick best target and convert.
    let target =
        best_match(src_desc, supported, intent).ok_or(ConvertError::EmptyFormatList)?;

    let converter = RowConverter::new(src_desc, target)?;

    let width = pixels.width();
    let rows = pixels.rows();
    let dst_bpp = target.bytes_per_pixel();
    let dst_stride = (width as usize) * dst_bpp;
    let mut output = vec![0u8; dst_stride * rows as usize];

    for y in 0..rows {
        let src_row = pixels.row(y);
        let dst_start = y as usize * dst_stride;
        let dst_end = dst_start + dst_stride;
        converter.convert_row(src_row, &mut output[dst_start..dst_end], width);
    }

    Ok(Adapted {
        data: Cow::Owned(output),
        descriptor: target,
        width,
        rows,
    })
}

/// Convert a raw byte buffer from one format to another.
///
/// Useful for decode-side adaptation where you have raw decoded bytes
/// and need them in a specific target format.
///
/// Returns the converted buffer.
pub fn convert_buffer(
    src: &[u8],
    width: u32,
    rows: u32,
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<Vec<u8>, ConvertError> {
    if from == to {
        return Ok(src.to_vec());
    }

    let converter = RowConverter::new(from, to)?;
    let src_bpp = from.bytes_per_pixel();
    let dst_bpp = to.bytes_per_pixel();
    let src_stride = (width as usize) * src_bpp;
    let dst_stride = (width as usize) * dst_bpp;
    let mut output = vec![0u8; dst_stride * rows as usize];

    for y in 0..rows {
        let src_start = y as usize * src_stride;
        let src_end = src_start + src_stride;
        let dst_start = y as usize * dst_stride;
        let dst_end = dst_start + dst_stride;
        converter.convert_row(
            &src[src_start..src_end],
            &mut output[dst_start..dst_end],
            width,
        );
    }

    Ok(output)
}
