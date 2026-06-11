//! Codec adapter functions — the fastest path to a compliant encoder.
//!
//! These functions combine format negotiation with pixel conversion in a
//! single call, replacing the per-codec format dispatch if-chains that
//! every encoder would otherwise need to write.
//!
//! # Which function to use
//!
//! | Function | Negotiation | Policy | Use case |
//! |----------|-------------|--------|----------|
//! | [`adapt_for_encode`] | `Fastest` intent | Permissive | Simple encode path |
//! | [`adapt_for_encode_with_intent`] | Caller-specified | Permissive | Encode after processing |
//! | [`adapt_for_encode_explicit`] | `Fastest` intent | [`ConvertOptions`] | Policy-sensitive encode |
//! | [`convert_buffer`] | None (caller picks) | Permissive | Direct format→format |
//!
//! # Zero-copy fast path
//!
//! All `adapt_for_encode*` functions check for an exact match first. If the
//! source descriptor matches one of the supported formats, the function
//! returns `Cow::Borrowed` — no allocation, no copy, no conversion. This
//! means the common case (JPEG u8 sRGB → JPEG u8 sRGB) has zero overhead.
//!
//! A second fast path handles transfer-agnostic matches: if the source has
//! `TransferFunction::Unknown` and a supported format matches on everything
//! else (depth, layout, alpha), it's also zero-copy. This covers codecs
//! that don't tag their output with a transfer function.
//!
//! # Strided buffers
//!
//! The `stride` parameter allows adapting buffers with row padding (common
//! when rows are SIMD-aligned or when working with sub-regions of a larger
//! buffer). If `stride > width * bpp`, the padding is stripped during
//! conversion and the output is always packed (stride = width * bpp).
//!
//! # Example
//!
//! ```rust,ignore
//! use zenpixels_convert::adapt::adapt_for_encode;
//!
//! let supported = &[
//!     PixelDescriptor::RGB8_SRGB,
//!     PixelDescriptor::GRAY8_SRGB,
//! ];
//!
//! let adapted = adapt_for_encode(
//!     raw_bytes, source_desc, width, rows, stride, supported,
//! )?;
//!
//! match &adapted.data {
//!     Cow::Borrowed(data) => {
//!         // Fast path: source was already in a supported format.
//!         encoder.write_direct(data, adapted.descriptor)?;
//!     }
//!     Cow::Owned(data) => {
//!         // Converted: write the new data with the new descriptor.
//!         encoder.write_converted(data, adapted.descriptor)?;
//!     }
//! }
//! ```

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use crate::convert::ConvertPlan;
use crate::converter::RowConverter;
use crate::negotiate::{ConvertIntent, best_match};
use crate::policy::{AlphaPolicy, ConvertOptions};
use crate::{ChannelLayout, ChannelType, ColorModel, ConvertError, PixelDescriptor, PixelSliceMut};
use whereat::{At, ResultAtExt};

/// Assert that a descriptor is not CMYK.
///
/// CMYK is device-dependent and cannot be adapted by zenpixels-convert.
/// Use a CMS (e.g., moxcms) with an ICC profile for CMYK↔RGB conversion.
fn assert_not_cmyk(desc: &PixelDescriptor) {
    assert!(
        desc.color_model() != ColorModel::Cmyk,
        "CMYK pixel data cannot be processed by zenpixels-convert. \
         Use a CMS (e.g., moxcms) with an ICC profile for CMYK↔RGB conversion."
    );
}

/// Result of format adaptation: the converted data and its descriptor.
#[derive(Clone, Debug)]
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
#[track_caller]
pub fn adapt_for_encode<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
) -> Result<Adapted<'a>, At<ConvertError>> {
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
#[track_caller]
pub fn adapt_for_encode_with_intent<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Result<Adapted<'a>, At<ConvertError>> {
    assert_not_cmyk(&descriptor);
    if supported.is_empty() {
        return Err(whereat::at!(ConvertError::EmptyFormatList));
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
    // still a zero-copy path. Primaries and signal range must also match
    // — relabeling BT.2020 as BT.709 without gamut conversion is wrong.
    for &target in supported {
        if descriptor.channel_type() == target.channel_type()
            && descriptor.layout() == target.layout()
            && descriptor.alpha() == target.alpha()
            && descriptor.primaries == target.primaries
            && descriptor.signal_range == target.signal_range
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
    let target = best_match(descriptor, supported, intent)
        .ok_or_else(|| whereat::at!(ConvertError::EmptyFormatList))?;

    let mut converter = RowConverter::new(descriptor, target).at()?;

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
#[track_caller]
pub fn convert_buffer(
    src: &[u8],
    width: u32,
    rows: u32,
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<Vec<u8>, At<ConvertError>> {
    assert_not_cmyk(&from);
    assert_not_cmyk(&to);
    if from == to {
        return Ok(src.to_vec());
    }

    let mut converter = RowConverter::new(from, to).at()?;
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

/// Attempt to adapt a buffer to `target` **in place** — no allocation, no
/// copy. Succeeds when the transition keeps every pixel's byte size:
///
/// * **identical byte layout** (same [`PixelFormat`](zenpixels::PixelFormat)):
///   descriptor re-tag only (transfer / primaries / signal range / alpha
///   mode) — zero data movement;
/// * **`Rgba8`-family ↔ `Bgra8`-family** (the X-padding forms included):
///   garb's SIMD B↔R swap, per row — strided buffers handled, padding
///   bytes untouched.
///
/// On success the returned slice views the **same memory** under the
/// target descriptor. `Err` hands the slice back **unchanged** so the
/// caller can fall through to the allocating [`adapt_for_encode`] path:
///
/// ```rust,ignore
/// let slice = match try_adapt_in_place(slice, target) {
///     Ok(adapted) => adapted,
///     Err(unchanged) => {
///         // shape-changing conversion — allocate via adapt_for_encode
///         fall_back(unchanged.as_pixel_slice())?
///     }
/// };
/// ```
///
/// Everything that changes the buffer's shape — channel-count changes
/// (RGBA→RGB, RGB→Gray), bit-depth changes (U16→U8, F32→U8) — returns
/// `Err`; those need [`adapt_for_encode`] /
/// [`convert_buffer`]. For content-driven narrowing (drop an opaque
/// alpha lane, collapse gray pixels) see
/// [`reduce_to_load_bearing_format_in_place`](crate::PixelSliceMutLoadBearingExt::reduce_to_load_bearing_format_in_place).
///
/// Like the other in-place APIs, the returned slice is the only valid
/// view of the rewritten bytes — a `PixelSliceMut` borrowed from a
/// descriptor-carrying owner leaves the owner's descriptor stale.
pub fn try_adapt_in_place<'a>(
    slice: PixelSliceMut<'a>,
    target: PixelDescriptor,
) -> Result<PixelSliceMut<'a>, PixelSliceMut<'a>> {
    let src = slice.descriptor();

    // Same byte layout: metadata-only re-tag, no pixel work.
    if src.format == target.format {
        return Ok(slice.with_descriptor(target));
    }
    if src.bytes_per_pixel() != target.bytes_per_pixel() {
        return Err(slice);
    }

    // The one same-size physical reorder the format set has: the 4-byte
    // B<->R swap between the Rgba and Bgra channel orders (U8 only — no
    // 16-bit Bgra format exists).
    let swappable = src.channel_type() == ChannelType::U8
        && target.channel_type() == ChannelType::U8
        && matches!(
            (src.layout(), target.layout()),
            (ChannelLayout::Rgba, ChannelLayout::Bgra) | (ChannelLayout::Bgra, ChannelLayout::Rgba)
        );
    if !swappable {
        return Err(slice);
    }

    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    let stride = slice.stride();
    let mut slice = slice;
    // Geometry was validated when the slice was constructed, so the swap
    // can't actually fail; stay total and give the slice back (untouched —
    // garb validates before writing) rather than panic.
    if garb::bytes::rgba_to_bgra_inplace_strided(
        slice.as_strided_bytes_mut(),
        width,
        height,
        stride,
    )
    .is_err()
    {
        return Err(slice);
    }
    // Same-bpp layout change: reinterpret cannot fail (bpp equality was
    // checked above, and that is reinterpret's only gate).
    Ok(slice
        .reinterpret(target)
        .expect("same-bpp reinterpret cannot fail"))
}

/// Negotiate format and convert with explicit policies.
///
/// Like [`adapt_for_encode`], but enforces [`ConvertOptions`] policies
/// on the conversion. Returns an error if a policy forbids the required
/// conversion.
#[track_caller]
pub fn adapt_for_encode_explicit<'a>(
    data: &'a [u8],
    descriptor: PixelDescriptor,
    width: u32,
    rows: u32,
    stride: usize,
    supported: &[PixelDescriptor],
    options: &ConvertOptions,
) -> Result<Adapted<'a>, At<ConvertError>> {
    assert_not_cmyk(&descriptor);
    if supported.is_empty() {
        return Err(whereat::at!(ConvertError::EmptyFormatList));
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

    // Check for transfer-agnostic match (primaries and signal range must match).
    for &target in supported {
        if descriptor.channel_type() == target.channel_type()
            && descriptor.layout() == target.layout()
            && descriptor.alpha() == target.alpha()
            && descriptor.primaries == target.primaries
            && descriptor.signal_range == target.signal_range
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

    // Need conversion — pick best target, then validate policies.
    let target = best_match(descriptor, supported, ConvertIntent::Fastest)
        .ok_or_else(|| whereat::at!(ConvertError::EmptyFormatList))?;

    // Validate policies before doing work.
    let plan = ConvertPlan::new_explicit(descriptor, target, options).at()?;

    // Runtime opacity check for DiscardIfOpaque.
    let drops_alpha = descriptor.alpha().is_some() && target.alpha().is_none();
    if drops_alpha && options.alpha_policy == AlphaPolicy::DiscardIfOpaque {
        let src_bpp = descriptor.bytes_per_pixel();
        if !is_fully_opaque(data, width, rows, stride, src_bpp, &descriptor) {
            return Err(whereat::at!(ConvertError::AlphaNotOpaque));
        }
    }

    let mut converter = RowConverter::from_plan(plan);
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

/// Check if all alpha values in a strided buffer are fully opaque.
fn is_fully_opaque(
    data: &[u8],
    width: u32,
    rows: u32,
    stride: usize,
    bpp: usize,
    desc: &PixelDescriptor,
) -> bool {
    if desc.alpha().is_none() {
        return true;
    }
    let cs = desc.channel_type().byte_size();
    let alpha_offset = (desc.layout().channels() - 1) * cs;
    for y in 0..rows {
        let row_start = y as usize * stride;
        for x in 0..width as usize {
            let off = row_start + x * bpp + alpha_offset;
            match desc.channel_type() {
                crate::ChannelType::U8 => {
                    if data[off] != 255 {
                        return false;
                    }
                }
                crate::ChannelType::U16 => {
                    let v = u16::from_ne_bytes([data[off], data[off + 1]]);
                    if v != 65535 {
                        return false;
                    }
                }
                crate::ChannelType::F32 => {
                    let v = f32::from_ne_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]);
                    if v < 1.0 {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
    true
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

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::descriptor::{ColorPrimaries, SignalRange};
    use zenpixels::policy::{AlphaPolicy, DepthPolicy};

    /// 2×1 RGB8 pixel data (6 bytes).
    fn test_rgb8_data() -> Vec<u8> {
        vec![255, 0, 0, 0, 255, 0]
    }

    // ── try_adapt_in_place ─────────────────────────────────────────

    #[test]
    fn in_place_bgra_to_rgba_swaps_bytes_without_alloc() {
        // Bgra8 stores B,G,R,A. Two pixels.
        let mut bytes = [10u8, 20, 30, 255, 40, 50, 60, 128];
        let slice = PixelSliceMut::new(&mut bytes, 2, 1, 8, PixelDescriptor::BGRA8_SRGB).unwrap();
        let out = try_adapt_in_place(slice, PixelDescriptor::RGBA8_SRGB)
            .expect("4bpp B<->R swap is in-place");
        assert_eq!(out.descriptor(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(out.row(0), &[30u8, 20, 10, 255, 60, 50, 40, 128]);
    }

    #[test]
    fn in_place_rgba_to_bgra_roundtrips() {
        let original = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut bytes = original;
        let slice = PixelSliceMut::new(&mut bytes, 2, 1, 8, PixelDescriptor::RGBA8_SRGB).unwrap();
        let bgra = try_adapt_in_place(slice, PixelDescriptor::BGRA8_SRGB).expect("to bgra");
        assert_eq!(bgra.row(0), &[3u8, 2, 1, 4, 7, 6, 5, 8]);
        let rgba = try_adapt_in_place(bgra, PixelDescriptor::RGBA8_SRGB).expect("back to rgba");
        assert_eq!(rgba.row(0), &original[..]);
    }

    #[test]
    fn in_place_swap_respects_stride_padding() {
        // 1 pixel per row, stride 8: 4 padding bytes per row must be
        // untouched.
        let mut bytes = [10u8, 20, 30, 255, 0xAA, 0xAB, 0xAC, 0xAD, 40, 50, 60, 128];
        let slice = PixelSliceMut::new(&mut bytes, 1, 2, 8, PixelDescriptor::BGRA8_SRGB).unwrap();
        let out = try_adapt_in_place(slice, PixelDescriptor::RGBA8_SRGB).expect("strided swap");
        assert_eq!(out.row(0), &[30u8, 20, 10, 255]);
        assert_eq!(out.row(1), &[60u8, 50, 40, 128]);
        drop(out);
        assert_eq!(
            &bytes[4..8],
            &[0xAA, 0xAB, 0xAC, 0xAD],
            "padding bytes must be untouched"
        );
    }

    #[test]
    fn in_place_metadata_retag_moves_no_bytes() {
        let original = [1u8, 2, 3, 4, 5, 6];
        let mut bytes = original;
        let slice = PixelSliceMut::new(&mut bytes, 2, 1, 6, PixelDescriptor::RGB8).unwrap();
        let target = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        let out = try_adapt_in_place(slice, target).expect("same-format retag");
        assert_eq!(out.descriptor(), target);
        assert_eq!(out.row(0), &original[..]);
    }

    #[test]
    fn in_place_rejects_shape_changes_unchanged() {
        // RGBA -> RGB shrinks the pixel; must hand the slice back untouched.
        let original = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut bytes = original;
        let slice = PixelSliceMut::new(&mut bytes, 2, 1, 8, PixelDescriptor::RGBA8_SRGB).unwrap();
        let back = try_adapt_in_place(slice, PixelDescriptor::RGB8_SRGB)
            .expect_err("channel-count change cannot be in-place");
        assert_eq!(back.descriptor(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(back.row(0), &original[..]);

        // Bit-depth change likewise.
        let mut bytes = original;
        let slice = PixelSliceMut::new(&mut bytes, 2, 1, 8, PixelDescriptor::RGBA8_SRGB).unwrap();
        let back = try_adapt_in_place(slice, PixelDescriptor::RGBA16_SRGB)
            .expect_err("depth change cannot be in-place");
        assert_eq!(back.row(0), &original[..]);
    }

    #[test]
    fn transfer_agnostic_match_requires_same_primaries() {
        let data = test_rgb8_data();
        let source = PixelDescriptor::RGB8.with_primaries(ColorPrimaries::Bt2020);
        let target = PixelDescriptor::RGB8_SRGB; // BT.709 primaries

        let result = adapt_for_encode(&data, source, 2, 1, 6, &[target]).unwrap();

        // Must NOT zero-copy relabel — primaries differ, conversion is needed.
        // Before the fix, this would return Cow::Borrowed (zero-copy) via the
        // transfer-agnostic match, silently relabeling BT.2020 as BT.709.
        assert!(
            matches!(result.data, Cow::Owned(_)),
            "different primaries must trigger conversion, not zero-copy relabel"
        );
    }

    #[test]
    fn transfer_agnostic_match_requires_same_signal_range() {
        let data = test_rgb8_data();
        let source = PixelDescriptor::RGB8.with_signal_range(SignalRange::Narrow);
        let target = PixelDescriptor::RGB8_SRGB; // Full range

        let result = adapt_for_encode(&data, source, 2, 1, 6, &[target]).unwrap();

        // Must not zero-copy relabel — signal ranges differ.
        assert!(
            matches!(result.data, Cow::Owned(_)),
            "different signal range must trigger conversion, not zero-copy relabel"
        );
    }

    #[test]
    fn transfer_agnostic_match_allows_zero_copy_when_all_match() {
        let data = test_rgb8_data();
        // Source: RGB8 with unknown transfer, BT.709, Full range.
        let source = PixelDescriptor::RGB8.with_primaries(ColorPrimaries::Bt709);
        // Target: RGB8 sRGB with same primaries and range.
        let target = PixelDescriptor::RGB8_SRGB;

        let result = adapt_for_encode(&data, source, 2, 1, 6, &[target]).unwrap();

        // Should zero-copy (only transfer differs, which is the agnostic part).
        assert!(
            matches!(result.data, Cow::Borrowed(_)),
            "should be zero-copy when only transfer differs"
        );
        assert_eq!(result.descriptor, target);
    }

    #[test]
    fn exact_match_is_zero_copy() {
        let data = test_rgb8_data();
        let desc = PixelDescriptor::RGB8_SRGB;

        let result = adapt_for_encode(&data, desc, 2, 1, 6, &[desc]).unwrap();

        assert!(matches!(result.data, Cow::Borrowed(_)));
        assert_eq!(result.descriptor, desc);
    }

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_by_adapt_for_encode() {
        let cmyk_data = vec![0u8; 4 * 4]; // 4 pixels
        let _ = adapt_for_encode(
            &cmyk_data,
            PixelDescriptor::CMYK8,
            2,
            2,
            8,
            &[PixelDescriptor::RGB8_SRGB],
        );
    }

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_by_convert_buffer() {
        let cmyk_data = vec![0u8; 4 * 4];
        let _ = convert_buffer(
            &cmyk_data,
            2,
            2,
            PixelDescriptor::CMYK8,
            PixelDescriptor::RGB8_SRGB,
        );
    }

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_by_convert_buffer_as_target() {
        let rgb_data = vec![0u8; 3 * 4];
        let _ = convert_buffer(
            &rgb_data,
            2,
            2,
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::CMYK8,
        );
    }

    #[test]
    fn explicit_variant_also_checks_primaries() {
        let data = test_rgb8_data();
        let source = PixelDescriptor::RGB8.with_primaries(ColorPrimaries::Bt2020);
        let target = PixelDescriptor::RGB8_SRGB;
        let options = ConvertOptions::forbid_lossy()
            .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
            .with_depth_policy(DepthPolicy::Round);

        let result =
            adapt_for_encode_explicit(&data, source, 2, 1, 6, &[target], &options).unwrap();

        assert!(
            matches!(result.data, Cow::Owned(_)),
            "explicit variant: different primaries must trigger conversion"
        );
    }
}
