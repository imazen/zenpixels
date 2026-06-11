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
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ColorModel, ConvertError, PixelBuffer, PixelDescriptor,
    PixelSliceMut,
};
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
/// A [`SignalRange`](zenpixels::SignalRange) mismatch with every supported
/// format fails with [`ConvertError::NoPath`]: no Narrow↔Full kernels
/// exist, and neither the zero-copy paths nor the planner will relabel a
/// range without rescaling. Offer a same-range target to accept narrow
/// input verbatim.
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

/// Attempt to adapt a [`PixelBuffer`] to `target` **in place** — no
/// allocation, no copy of the frame, and the buffer's descriptor /
/// geometry / color context are updated **atomically** via
/// [`PixelBuffer::transform_in_place`] (this is deliberately the only
/// in-place adaptation entry point; a re-described view over an owner
/// carrying the old descriptor is unrepresentable). Three transition
/// classes succeed:
///
/// * **identical byte layout** (same [`PixelFormat`](zenpixels::PixelFormat)):
///   descriptor re-tag only (transfer / primaries / signal range / alpha
///   mode) — zero data movement;
/// * **`Rgba8`-family ↔ `Bgra8`-family** (the X-padding forms included):
///   garb's SIMD B↔R swap, per row — strided buffers handled, padding
///   bytes untouched;
/// * **alpha-lane removal with contract-droppable alpha** — RGBA→RGB,
///   BGRA→RGB (with the B↔R reorder), GrayAlpha→Gray, at U8/U16/F32 as
///   the formats exist — **only** when the source's alpha mode is
///   [`AlphaMode::Undefined`] (X padding) or [`AlphaMode::Opaque`]
///   (declared all-max): in those modes dropping the lane is value-exact
///   by contract. The stride is kept as close as the slice rules allow —
///   rounded down to a whole number of target pixels — so when the input
///   stride already divides evenly rows compact **at their own bases**
///   with zero cross-row movement and the freed bytes become row padding.
///   Straight or premultiplied alpha returns `Err`: a blind discard would
///   silently diverge from [`adapt_for_encode`]'s alpha policy (which
///   mattes); for *measured*-opaque alpha use
///   [`reduce_to_load_bearing_format_in_place`](crate::PixelBufferLoadBearingExt::reduce_to_load_bearing_format_in_place),
///   which scans and proves it first.
///
/// `Err(ConvertError::NoPath)` means the transition needs a real
/// conversion (bit-depth change, chroma removal, lane addition, live
/// alpha) — the buffer is **untouched**; fall through to the allocating
/// [`adapt_for_encode`] / [`convert_buffer`]:
///
/// ```rust,ignore
/// if try_adapt_in_place(&mut buf, target).is_err() {
///     // needs a real conversion — allocate via adapt_for_encode
/// }
/// ```
pub fn try_adapt_in_place(
    buf: &mut PixelBuffer,
    target: PixelDescriptor,
) -> Result<(), At<ConvertError>> {
    let src = buf.descriptor();
    let no_path = || {
        Err(whereat::at!(ConvertError::NoPath {
            from: src,
            to: target
        }))
    };

    // Same byte layout: metadata-only re-tag, no pixel work.
    if src.format == target.format {
        buf.transform_in_place(|px| {
            rewrap(px.bytes, px.width, px.rows, px.stride, target, px.color)
        });
        return Ok(());
    }

    // Same-size physical reorder: the 4-byte B<->R swap between the Rgba
    // and Bgra channel orders (U8 only — no 16-bit Bgra format exists).
    if src.bytes_per_pixel() == target.bytes_per_pixel() {
        let swappable = src.channel_type() == ChannelType::U8
            && target.channel_type() == ChannelType::U8
            && matches!(
                (src.layout(), target.layout()),
                (ChannelLayout::Rgba, ChannelLayout::Bgra)
                    | (ChannelLayout::Bgra, ChannelLayout::Rgba)
            );
        if !swappable {
            return no_path();
        }
        buf.transform_in_place(|px| {
            let width = px.width as usize;
            let rows = px.rows as usize;
            // Geometry was validated at construction; the impossible
            // size-mismatch error keeps the closure total.
            let _ = garb::bytes::rgba_to_bgra_inplace_strided(px.bytes, width, rows, px.stride);
            rewrap(px.bytes, px.width, px.rows, px.stride, target, px.color)
        });
        return Ok(());
    }

    // Shrinking alpha-lane removal. Allowed only when the source's alpha
    // mode makes the drop value-exact BY CONTRACT (Undefined padding /
    // declared Opaque) — discarding live Straight/Premultiplied alpha
    // here would silently diverge from adapt_for_encode's matting policy.
    if !matches!(
        src.alpha,
        Some(AlphaMode::Undefined) | Some(AlphaMode::Opaque)
    ) {
        return no_path();
    }
    if src.channel_type() != target.channel_type() {
        return no_path();
    }
    // Channel-selection map in element units; mirrors the load-bearing
    // rewrite's transition table for the lane-drop subset.
    let map: &'static [usize] = match (src.layout(), target.layout()) {
        (ChannelLayout::Rgba, ChannelLayout::Rgb) => &[0, 1, 2],
        // Bgra stores B,G,R,A — dropping the lane into Rgb needs the
        // B<->R reorder (U8 only; no Bgra16/F32 formats exist).
        (ChannelLayout::Bgra, ChannelLayout::Rgb) => &[2, 1, 0],
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => &[0],
        _ => return no_path(),
    };

    let in_bpp = src.bytes_per_pixel();
    let out_bpp = target.bytes_per_pixel();
    let elem = src.bytes_per_channel();

    buf.transform_in_place(|px| drop_lane_impl(px, target, map, in_bpp, out_bpp, elem));
    Ok(())
}

/// The lane-drop transform behind [`try_adapt_in_place`], split out so
/// arbitrary-stride geometries (not constructible through the public
/// buffer constructors) stay unit-testable.
fn drop_lane_impl<'a>(
    px: zenpixels::InPlacePixels<'a>,
    target: PixelDescriptor,
    map: &'static [usize],
    in_bpp: usize,
    out_bpp: usize,
    elem: usize,
) -> PixelSliceMut<'a> {
    let width = px.width as usize;
    // Output stride: the input stride rounded down to a whole number
    // of target pixels (PixelSlice requires stride % bpp == 0). When
    // the input stride is already a multiple of the narrower pixel,
    // rows stay at their own bases (zero cross-row movement, freed
    // bytes become row padding); otherwise rows shift up slightly.
    // Either way dst(y, x) <= src(y, x) for every pixel, so a
    // forward pass staged through a fixed temp never clobbers unread
    // source.
    let out_stride = px.stride - (px.stride % out_bpp);
    for y in 0..px.rows as usize {
        let sbase = y * px.stride;
        let dbase = y * out_stride;
        for x in 0..width {
            let s = sbase + x * in_bpp;
            let mut tmp = [0u8; 16];
            tmp[..in_bpp].copy_from_slice(&px.bytes[s..s + in_bpp]);
            let d = dbase + x * out_bpp;
            for (k, &c) in map.iter().enumerate() {
                px.bytes[d + k * elem..d + (k + 1) * elem]
                    .copy_from_slice(&tmp[c * elem..(c + 1) * elem]);
            }
        }
    }
    rewrap(px.bytes, px.width, px.rows, out_stride, target, px.color)
}

/// Re-wrap transform output bytes under a new description, carrying the
/// color context (in-place adaptations are color-class-preserving).
fn rewrap<'a>(
    bytes: &'a mut [u8],
    width: u32,
    rows: u32,
    stride: usize,
    descriptor: PixelDescriptor,
    color: Option<alloc::sync::Arc<zenpixels::ColorContext>>,
) -> PixelSliceMut<'a> {
    let out = PixelSliceMut::new(bytes, width, rows, stride, descriptor)
        .expect("in-place adaptation geometry is always valid");
    match color {
        Some(c) => out.with_color_context(c),
        None => out,
    }
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

    fn buf_from(bytes: &[u8], w: u32, h: u32, desc: PixelDescriptor) -> zenpixels::PixelBuffer {
        zenpixels::PixelBuffer::from_vec(bytes.to_vec(), w, h, desc).unwrap()
    }

    #[test]
    fn in_place_bgra_to_rgba_swaps_bytes_and_updates_buffer() {
        // Bgra8 stores B,G,R,A. Two pixels.
        let mut buf = buf_from(
            &[10u8, 20, 30, 255, 40, 50, 60, 128],
            2,
            1,
            PixelDescriptor::BGRA8_SRGB,
        );
        try_adapt_in_place(&mut buf, PixelDescriptor::RGBA8_SRGB)
            .expect("4bpp B<->R swap is in-place");
        assert_eq!(buf.descriptor(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(buf.as_slice().row(0), &[30u8, 20, 10, 255, 60, 50, 40, 128]);
    }

    #[test]
    fn in_place_rgba_to_bgra_roundtrips() {
        let original = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut buf = buf_from(&original, 2, 1, PixelDescriptor::RGBA8_SRGB);
        try_adapt_in_place(&mut buf, PixelDescriptor::BGRA8_SRGB).expect("to bgra");
        assert_eq!(buf.as_slice().row(0), &[3u8, 2, 1, 4, 7, 6, 5, 8]);
        try_adapt_in_place(&mut buf, PixelDescriptor::RGBA8_SRGB).expect("back to rgba");
        assert_eq!(buf.as_slice().row(0), &original[..]);
    }

    #[test]
    fn in_place_swap_respects_stride_padding() {
        // SIMD-aligned buffer: 1 px/row at simd_align 16 → stride 16,
        // 12 padding bytes per row that must be untouched.
        let mut buf =
            zenpixels::PixelBuffer::new_simd_aligned(1, 2, PixelDescriptor::BGRA8_SRGB, 16);
        assert_eq!(buf.stride(), 16, "fixture must be strided");
        {
            let mut view = buf.as_slice_mut();
            view.row_mut(0).copy_from_slice(&[10, 20, 30, 255]);
            view.row_mut(1).copy_from_slice(&[40, 50, 60, 128]);
            let backing = view.as_strided_bytes_mut();
            backing[4..16].fill(0xAA);
            backing[20..32].fill(0xBB);
        }
        try_adapt_in_place(&mut buf, PixelDescriptor::RGBA8_SRGB).expect("strided swap");
        assert_eq!(buf.as_slice().row(0), &[30u8, 20, 10, 255]);
        assert_eq!(buf.as_slice().row(1), &[60u8, 50, 40, 128]);
        let view = buf.as_slice();
        let backing = view.as_strided_bytes();
        assert!(
            backing[4..16].iter().all(|&b| b == 0xAA),
            "row-0 padding must be untouched"
        );
        assert!(
            backing[20..28].iter().all(|&b| b == 0xBB),
            "row-1 padding must be untouched"
        );
    }

    #[test]
    fn in_place_metadata_retag_moves_no_bytes() {
        let original = [1u8, 2, 3, 4, 5, 6];
        let mut buf = buf_from(&original, 2, 1, PixelDescriptor::RGB8);
        let target = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        try_adapt_in_place(&mut buf, target).expect("same-format retag");
        assert_eq!(buf.descriptor(), target);
        assert_eq!(buf.as_slice().row(0), &original[..]);
    }

    #[test]
    fn in_place_rejects_live_alpha_drop_and_depth_changes_unchanged() {
        // RGBA(Straight) -> RGB would discard live alpha; must leave the
        // buffer untouched (adapt_for_encode mattes instead).
        let original = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut buf = buf_from(&original, 2, 1, PixelDescriptor::RGBA8_SRGB);
        try_adapt_in_place(&mut buf, PixelDescriptor::RGB8_SRGB)
            .expect_err("straight-alpha drop is not contract-exact");
        assert_eq!(buf.descriptor(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(buf.as_slice().row(0), &original[..]);

        // Bit-depth change likewise.
        try_adapt_in_place(&mut buf, PixelDescriptor::RGBA16_SRGB)
            .expect_err("depth change cannot be in-place");
        assert_eq!(buf.as_slice().row(0), &original[..]);
    }

    #[test]
    fn in_place_rgbx_to_rgb_compacts_and_buffer_adopts_geometry() {
        // RGBX (Undefined padding): the X byte is contract-droppable.
        // 2 px/row, 2 rows, tight 4bpp stride 8 → out stride rounds down
        // to 6 (tight for 3bpp) and the buffer's own stride/descriptor
        // update atomically.
        let mut buf = buf_from(
            &[
                1u8, 2, 3, 0xEE, 4, 5, 6, 0xEE, // row 0
                7, 8, 9, 0xEE, 10, 11, 12, 0xEE, // row 1
            ],
            2,
            2,
            PixelDescriptor::RGBX8_SRGB,
        );
        try_adapt_in_place(&mut buf, PixelDescriptor::RGB8_SRGB)
            .expect("padding drop is contract-exact");
        assert_eq!(buf.descriptor(), PixelDescriptor::RGB8_SRGB);
        assert_eq!(buf.stride(), 6);
        assert_eq!(buf.as_slice().row(0), &[1u8, 2, 3, 4, 5, 6]);
        assert_eq!(buf.as_slice().row(1), &[7u8, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn drop_lane_impl_keeps_divisible_stride_rows_in_place() {
        // Stride 12 (divisible by 3): rows stay at their own bases —
        // zero cross-row movement, freed bytes become padding. Exercised
        // at the transform level because the public constructors only
        // produce pixel-tight or simd-aligned strides.
        let mut bytes = [
            1u8, 2, 3, 0xEE, 4, 5, 6, 0xEE, 0xAA, 0xAA, 0xAA, 0xAA, // row 0 + pad
            7, 8, 9, 0xEE, 10, 11, 12, 0xEE, 0xBB, 0xBB, 0xBB, 0xBB, // row 1 + pad
        ];
        let px = zenpixels::InPlacePixels {
            bytes: &mut bytes,
            width: 2,
            rows: 2,
            stride: 12,
            descriptor: PixelDescriptor::RGBX8_SRGB,
            color: None,
        };
        let out = drop_lane_impl(px, PixelDescriptor::RGB8_SRGB, &[0, 1, 2], 4, 3, 1);
        assert_eq!(out.stride(), 12, "divisible stride preserved verbatim");
        assert_eq!(out.row(0), &[1u8, 2, 3, 4, 5, 6]);
        assert_eq!(out.row(1), &[7u8, 8, 9, 10, 11, 12]);
        drop(out);
        assert_eq!(&bytes[8..12], &[0xAA; 4], "row-0 tail padding untouched");
        assert_eq!(&bytes[20..24], &[0xBB; 4], "row-1 tail padding untouched");
    }

    #[test]
    fn in_place_opaque_bgra_to_rgb_reorders_while_dropping() {
        // Declared-Opaque BGRA -> RGB: lane drop + B<->R reorder.
        let mut buf = buf_from(
            &[10u8, 20, 30, 255, 40, 50, 60, 255],
            2,
            1,
            PixelDescriptor::BGRA8_SRGB.with_alpha_mode(Some(AlphaMode::Opaque)),
        );
        try_adapt_in_place(&mut buf, PixelDescriptor::RGB8_SRGB).expect("opaque drop allowed");
        assert_eq!(buf.as_slice().row(0), &[30u8, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn in_place_opaque_rgba16_to_rgb16_drops_lane() {
        // U16 lane drop: element-wise (2-byte) selection.
        let px16 = |r: u16, g: u16, b: u16| {
            [r, g, b, 0xFFFF]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>()
        };
        let bytes: Vec<u8> = [px16(0x1234, 0x5678, 0x9ABC), px16(0x1111, 0x2222, 0x3333)].concat();
        let mut buf = buf_from(
            &bytes,
            2,
            1,
            PixelDescriptor::RGBA16_SRGB.with_alpha_mode(Some(AlphaMode::Opaque)),
        );
        try_adapt_in_place(&mut buf, PixelDescriptor::RGB16_SRGB).expect("u16 lane drop");
        let expected: Vec<u8> = [0x1234u16, 0x5678, 0x9ABC, 0x1111, 0x2222, 0x3333]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(buf.as_slice().row(0), &expected[..]);
        assert_eq!(buf.stride(), 12, "16-px input stride rounds to 12");
    }

    #[test]
    fn in_place_opaque_graya_to_gray_matches_allocating_path() {
        // Differential vs convert_buffer on the same transition.
        let original = [10u8, 255, 20, 255, 30, 255, 40, 255];
        let src = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Opaque),
            zenpixels::TransferFunction::Srgb,
        );
        let target = PixelDescriptor::GRAY8_SRGB;

        let mut buf = buf_from(&original, 4, 1, src);
        try_adapt_in_place(&mut buf, target).expect("graya drop");
        let in_place_row = buf.as_slice().row(0).to_vec();

        let allocated = convert_buffer(&original, 4, 1, src, target).expect("allocating path");
        assert_eq!(in_place_row, allocated, "in-place must match allocating");
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

    /// A signal-range mismatch refuses loudly. No Narrow↔Full kernels
    /// exist, so neither zero-copy relabeling nor an allocating
    /// "conversion" is acceptable — the latter would emit narrow-coded
    /// values under a full-range label (this test's predecessor codified
    /// exactly that bug by asserting only that an allocation happened).
    #[test]
    fn signal_range_mismatch_refuses_not_relabels() {
        let data = test_rgb8_data();
        let source = PixelDescriptor::RGB8.with_signal_range(SignalRange::Narrow);
        let target = PixelDescriptor::RGB8_SRGB; // Full range

        let err = adapt_for_encode(&data, source, 2, 1, 6, &[target]).unwrap_err();
        assert!(
            matches!(*err.error(), ConvertError::NoPath { .. }),
            "range crossing must refuse (no kernels), got: {}",
            err.error()
        );
    }

    /// Narrow data is accepted verbatim when a same-range target is offered:
    /// the transfer-agnostic zero-copy arm applies as usual once the signal
    /// ranges agree.
    #[test]
    fn signal_range_match_zero_copies_narrow_verbatim() {
        let data = test_rgb8_data();
        let source = PixelDescriptor::RGB8
            .with_primaries(ColorPrimaries::Bt709)
            .with_signal_range(SignalRange::Narrow);
        let full_target = PixelDescriptor::RGB8_SRGB;
        let narrow_target = PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow);

        let result =
            adapt_for_encode(&data, source, 2, 1, 6, &[full_target, narrow_target]).unwrap();
        assert!(
            matches!(result.data, Cow::Borrowed(_)),
            "same-range target must zero-copy"
        );
        assert_eq!(result.descriptor.signal_range, SignalRange::Narrow);
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
