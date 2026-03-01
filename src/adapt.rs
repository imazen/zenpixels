//! Codec adapter functions.
//!
//! High-level helpers that combine format negotiation with conversion,
//! replacing the per-codec format dispatch if-chains.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use crate::PixelDescriptor;
use crate::converter::RowConverter;
use crate::error::ConvertError;
use crate::negotiate::{ConvertIntent, best_match};

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

/// Negotiate format and convert pixel data for encoding.
///
/// Uses [`ConvertIntent::Fastest`] — minimizes conversion cost.
///
/// If the input already matches one of the `supported` formats, returns
/// `Cow::Borrowed` (zero-copy). Otherwise, converts to the best match.
///
/// # Arguments
///
/// * `data` - Raw pixel bytes, `rows * stride` bytes minimum.
/// * `descriptor` - Format of the input data.
/// * `width` - Pixels per row.
/// * `rows` - Number of rows.
/// * `stride` - Bytes between row starts (use `width * descriptor.bytes_per_pixel()` for packed).
/// * `supported` - Formats the encoder accepts.
pub fn adapt_for_encode<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
) -> Result<Adapted<'a>, ConvertError> {
    adapt_for_encode_with_intent(
        data,
        descriptor,
        width,
        rows,
        stride,
        supported,
        ConvertIntent::Fastest,
    )
}

/// Negotiate format and convert with intent awareness.
///
/// Like [`adapt_for_encode`], but lets the caller specify a [`ConvertIntent`].
pub fn adapt_for_encode_with_intent<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Result<Adapted<'a>, ConvertError> {
    if supported.is_empty() {
        return Err(ConvertError::EmptyFormatList);
    }

    // Check for exact match (zero-copy path).
    if supported.contains(&descriptor) {
        return Ok(Adapted {
            data: contiguous_from_strided(data, width, rows, stride, descriptor.bytes_per_pixel()),
            descriptor,
            width,
            rows,
        });
    }

    // Check for transfer-agnostic match: if source has Unknown transfer
    // and a supported format matches on everything except transfer, it's
    // still a zero-copy path.
    for &target in supported {
        if descriptor.channel_type == target.channel_type
            && descriptor.layout == target.layout
            && descriptor.alpha == target.alpha
        {
            return Ok(Adapted {
                data: contiguous_from_strided(
                    data,
                    width,
                    rows,
                    stride,
                    descriptor.bytes_per_pixel(),
                ),
                descriptor: target,
                width,
                rows,
            });
        }
    }

    // Need conversion — pick best target.
    let target = best_match(descriptor, supported, intent).ok_or(ConvertError::EmptyFormatList)?;

    let converter = RowConverter::new(descriptor, target)?;

    let src_bpp = descriptor.bytes_per_pixel();
    let dst_bpp = target.bytes_per_pixel();
    let dst_stride = (width as usize) * dst_bpp;
    let mut output = vec![0u8; dst_stride * rows as usize];

    for y in 0..rows {
        let src_start = y as usize * stride;
        let src_end = src_start + (width as usize * src_bpp);
        let dst_start = y as usize * dst_stride;
        let dst_end = dst_start + dst_stride;
        converter.convert_row(
            &data[src_start..src_end],
            &mut output[dst_start..dst_end],
            width,
        );
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
/// Assumes packed (stride = width * bpp) layout.
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

/// Extract contiguous packed rows from potentially strided data.
fn contiguous_from_strided<'a>(
    data: &'a [u8],
    width: u32,
    rows: u32,
    stride: usize,
    bpp: usize,
) -> Cow<'a, [u8]> {
    let row_bytes = width as usize * bpp;
    if stride == row_bytes {
        // Already packed.
        let total = row_bytes * rows as usize;
        Cow::Borrowed(&data[..total])
    } else {
        // Need to strip padding.
        let mut packed = Vec::with_capacity(row_bytes * rows as usize);
        for y in 0..rows as usize {
            let start = y * stride;
            packed.extend_from_slice(&data[start..start + row_bytes]);
        }
        Cow::Owned(packed)
    }
}
