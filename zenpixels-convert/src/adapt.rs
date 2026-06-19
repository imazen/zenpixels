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

/// Reject CMYK on either side of a `from`/`to` pair at the public-API
/// boundary, surfacing it as a [`ConvertError::NoPath`] (which carries
/// both descriptors and gets a CMYK-aware hint from its `Display` impl).
///
/// **CMYK contract.** CMYK is a device-dependent subtractive colour
/// model: faithful CMYK↔RGB conversion requires an ICC profile and a
/// CMS, neither of which lives in this crate. Silently reinterpreting
/// C/M/Y/K as R/G/B/A would produce visibly wrong colours AND wrong
/// transparency — much worse than a clean rejection. Use moxcms (or
/// another CMS) for CMYK conversion.
///
/// Replaces the pre-#44 `assert_not_cmyk` that panicked the process.
/// We fold the CMYK case into the existing `NoPath { from, to }` variant
/// rather than introducing a `CmykUnsupported` arm: zero workspace
/// consumers match anything but `AllocationFailed` today, and a
/// dedicated variant would be future-facing surface no caller earns
/// (see imazen/zenpixels#44 discussion). Callers wanting to route to
/// a CMS can inspect `from.color_model() == ColorModel::Cmyk` on the
/// returned `NoPath` — information-equivalent.
fn reject_cmyk(from: PixelDescriptor, to: PixelDescriptor) -> Result<(), At<ConvertError>> {
    if from.color_model() == ColorModel::Cmyk || to.color_model() == ColorModel::Cmyk {
        return Err(whereat::at!(ConvertError::NoPath { from, to }));
    }
    Ok(())
}

/// Validate that a raw input buffer is long enough to hold the declared
/// geometry: `rows × stride` bytes minimum (the contract documented on
/// every raw-bytes entry point on this module).
///
/// The multiplication is `checked_mul` so a 32-bit `usize` wrap on i686 /
/// wasm32 surfaces as [`ConvertError::AllocationFailed`] (the same class
/// `PixelBuffer::from_pixels` uses) instead of silently passing through
/// with a small number that then OOBs in the slice loop. Pre-#44 the raw
/// entry points slice into `data` with no length check at all and panic
/// with index-OOB on a truncated buffer.
fn ensure_src_buffer_fits(
    data_len: usize,
    rows: u32,
    stride: usize,
) -> Result<(), At<ConvertError>> {
    if rows == 0 {
        return Ok(());
    }
    let needed = (rows as usize)
        .checked_mul(stride)
        .ok_or_else(|| whereat::at!(ConvertError::AllocationFailed))?;
    if data_len < needed {
        return Err(whereat::at!(ConvertError::BufferSize {
            expected: needed,
            actual: data_len,
        }));
    }
    Ok(())
}

/// Compute `rows × stride` as a `usize` byte count for an output
/// allocation, surfacing 32-bit overflow as
/// [`ConvertError::AllocationFailed`] instead of a silent wrap (which
/// would `vec![0u8; small]` and then OOB inside the conversion loop).
fn checked_byte_alloc(rows: u32, stride: usize) -> Result<usize, At<ConvertError>> {
    (rows as usize)
        .checked_mul(stride)
        .ok_or_else(|| whereat::at!(ConvertError::AllocationFailed))
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
    ensure_src_buffer_fits(data.len(), rows, stride)?;
    if supported.is_empty() {
        return Err(whereat::at!(ConvertError::EmptyFormatList));
    }
    // CMYK check after EmptyFormatList so we always have a concrete target
    // descriptor to pair into `NoPath { from, to }`.
    reject_cmyk(descriptor, supported[0])?;

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
    let mut output = vec![0u8; checked_byte_alloc(rows, dst_stride)?];

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
    reject_cmyk(from, to)?;
    let src_bpp = from.bytes_per_pixel();
    let src_stride = (width as usize) * src_bpp;
    ensure_src_buffer_fits(src.len(), rows, src_stride)?;
    if from == to {
        return Ok(src.to_vec());
    }

    let mut converter = RowConverter::new(from, to).at()?;
    let dst_bpp = to.bytes_per_pixel();
    let dst_stride = (width as usize) * dst_bpp;
    let mut output = vec![0u8; checked_byte_alloc(rows, dst_stride)?];

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

/// Like [`convert_buffer`] but anchors the **PQ** transfer steps to an
/// absolute-luminance white point — the cd/m² that relative-linear `1.0`
/// represents (e.g. [`DiffuseWhite::BT2408`](zenpixels::hdr::DiffuseWhite) =
/// 203). The PQ kernels then scale by `anchor / 10000` across the
/// relative-linear ↔ PQ-absolute boundary, so a relative-linear buffer encodes
/// to PQ at the right brightness with no caller-side pre-scale. Conversions
/// without a PQ step are identical to [`convert_buffer`].
///
/// Runs on the built-in plan path (so the anchored PQ steps are never bypassed
/// by a CMS matlut fast path). Honors a **strided** source — `src_stride` is the
/// bytes between row starts (`width * from.bytes_per_pixel()` for packed) — and
/// converts row-by-row with no pre-pack. Returns a freshly-allocated
/// [`PixelBuffer`] (its row stride is the buffer's own — no hand-rolled `Vec`).
/// The plan drives any channel change (e.g. `DropAlpha` for an RGB target, or
/// alpha-preserving passthrough for an RGBA one), so the caller hands the source
/// straight in. Alpha, if kept, is never PQ-encoded or anchor-scaled.
#[track_caller]
pub(crate) fn convert_buffer_with_anchor(
    src: &[u8],
    width: u32,
    rows: u32,
    src_stride: usize,
    from: PixelDescriptor,
    to: PixelDescriptor,
    anchor: zenpixels::hdr::DiffuseWhite,
) -> Result<PixelBuffer, At<ConvertError>> {
    // Reject CMYK and validate the src buffer *before* allocating the dst —
    // failing fast saves the alloc on a broken request, and points the
    // error at the outer entry point instead of the inner helper.
    reject_cmyk(from, to)?;
    ensure_src_buffer_fits(src.len(), rows, src_stride)?;
    // Allocate through the existing PixelBuffer machinery (start-aligned,
    // fallible) rather than a hand-rolled `vec![0u8; …]`, and convert into its
    // backing at the buffer's own row stride.
    let mut buf = PixelBuffer::try_new(width, rows, to)
        .map_err(|_| whereat::at!(ConvertError::AllocationFailed))?;
    let dst_stride = buf.stride();
    {
        let mut slice = buf.as_slice_mut();
        convert_into_with_anchor(
            src,
            width,
            rows,
            src_stride,
            from,
            to,
            anchor,
            slice.as_strided_bytes_mut(),
            dst_stride,
        )?;
    }
    Ok(buf)
}

/// Like [`convert_buffer_with_anchor`] but writes into a caller-provided `dst` —
/// no output allocation. Honors a **strided destination**: `dst_stride` is the
/// bytes between output row starts (pass `width * to.bytes_per_pixel()` for
/// packed). `dst` must hold `(rows - 1) * dst_stride + width * to.bpp` bytes and
/// `dst_stride` must be ≥ the packed row width; otherwise
/// [`ConvertError::BufferSize`]. The same strided-source / alpha rules apply.
#[track_caller]
#[allow(clippy::too_many_arguments)] // mirrors convert_buffer_with_anchor + strided dst
pub(crate) fn convert_into_with_anchor(
    src: &[u8],
    width: u32,
    rows: u32,
    src_stride: usize,
    from: PixelDescriptor,
    to: PixelDescriptor,
    anchor: zenpixels::hdr::DiffuseWhite,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<(), At<ConvertError>> {
    reject_cmyk(from, to)?;
    ensure_src_buffer_fits(src.len(), rows, src_stride)?;

    let dst_row = (width as usize) * to.bytes_per_pixel();
    if rows > 0 {
        // A strided dst must hold each row's content without rows overlapping:
        // `dst_stride >= dst_row`, and the last row ends at
        // `(rows-1) * dst_stride + dst_row`.
        let needed = (rows as usize - 1) * dst_stride + dst_row;
        if dst_stride < dst_row || dst.len() < needed {
            return Err(whereat::at!(ConvertError::BufferSize {
                expected: needed.max(dst_row),
                actual: dst.len().min(dst_stride),
            }));
        }
    }

    // `from == to` yields an Identity plan, whose row step copies src → dst — so
    // the strided loop handles it too (no packed-only special case needed).
    let plan = ConvertPlan::new(from, to).at()?.with_pq_anchor(anchor);
    let mut converter = RowConverter::from_plan(plan);
    let src_row = (width as usize) * from.bytes_per_pixel();

    for y in 0..rows as usize {
        let src_start = y * src_stride;
        let dst_start = y * dst_stride;
        converter.convert_row(
            &src[src_start..src_start + src_row],
            &mut dst[dst_start..dst_start + dst_row],
            width,
        );
    }

    Ok(())
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
    ensure_src_buffer_fits(data.len(), rows, stride)?;
    if supported.is_empty() {
        return Err(whereat::at!(ConvertError::EmptyFormatList));
    }
    reject_cmyk(descriptor, supported[0])?;

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
    let mut output = vec![0u8; checked_byte_alloc(rows, dst_stride)?];

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
mod anchor_tests {
    //! `convert_buffer_with_anchor` — proof the absolute-luminance anchor
    //! threads through the PQ `ConvertStep`s (not via any caller pre-scale),
    //! that it honors a strided source, and that it preserves alpha.
    use super::convert_buffer_with_anchor;
    use crate::{PixelDescriptor, TransferFunction};
    use alloc::vec;
    use alloc::vec::Vec;
    use zenpixels::hdr::DiffuseWhite;

    /// f64 SMPTE ST 2084 inverse-EOTF (linear-light fraction → PQ code [0,1]).
    fn pq_oetf(x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let m1 = 2610.0 / 16384.0;
        let m2 = 2523.0 / 4096.0 * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = 2413.0 / 4096.0 * 32.0;
        let c3 = 2392.0 / 4096.0 * 32.0;
        let xp = x.powf(m1);
        ((c1 + c2 * xp) / (1.0 + c3 * xp)).powf(m2)
    }

    /// Tight RGB f32 bytes from per-pixel gray values.
    fn gray_rgb_f32(values: &[f32]) -> Vec<u8> {
        let mut v = Vec::with_capacity(values.len() * 12);
        for &g in values {
            for _ in 0..3 {
                v.extend_from_slice(&g.to_ne_bytes());
            }
        }
        v
    }

    /// Tight RGBA f32 bytes: per-pixel gray RGB plus its alpha.
    fn gray_rgba_f32(pixels: &[(f32, f32)]) -> Vec<u8> {
        let mut v = Vec::with_capacity(pixels.len() * 16);
        for &(g, a) in pixels {
            for _ in 0..3 {
                v.extend_from_slice(&g.to_ne_bytes());
            }
            v.extend_from_slice(&a.to_ne_bytes());
        }
        v
    }

    /// Packed bytes-per-row for `desc` at `width` pixels.
    fn packed_stride(width: usize, desc: PixelDescriptor) -> usize {
        width * desc.bytes_per_pixel()
    }

    fn pq16_target() -> (PixelDescriptor, PixelDescriptor) {
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        // Tag the source gamut as the target's so no gamut step is inserted.
        let lin = PixelDescriptor::RGBF32_LINEAR.with_primaries(target.primaries);
        (lin, target)
    }

    #[test]
    fn anchor_pq16_encode_matches_st2084_oracle() {
        let values = [0.001f32, 0.1, 1.0, 2.0, 49.0];
        let (lin, target) = pq16_target();
        let out = convert_buffer_with_anchor(
            &gray_rgb_f32(&values),
            values.len() as u32,
            1,
            packed_stride(values.len(), lin),
            lin,
            target,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        let codes: &[u16] = bytemuck::cast_slice(out.as_slice().as_strided_bytes());
        for (i, &v) in values.iter().enumerate() {
            let got = i64::from(codes[i * 3]);
            // 1.0 of relative-linear sits at 203 / 10000 of the PQ-absolute range.
            let want = (pq_oetf(f64::from(v) * 203.0 / 10_000.0) * 65535.0).round() as i64;
            assert!(
                (got - want).abs() <= 1,
                "@203 at {v}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn anchor_changes_pq_output_in_kernel() {
        let (lin, target) = pq16_target();
        let src = gray_rgb_f32(&[1.0]);
        let enc = |w: DiffuseWhite| {
            let o = convert_buffer_with_anchor(&src, 1, 1, packed_stride(1, lin), lin, target, w)
                .unwrap();
            let ob = o.as_slice().as_strided_bytes();
            i64::from(u16::from_ne_bytes([ob[0], ob[1]]))
        };
        let c100 = enc(DiffuseWhite::new(100.0));
        let c203 = enc(DiffuseWhite::BT2408);
        // The same relative-linear 1.0 lands at a different PQ code per anchor —
        // i.e. the scale is applied inside the kernel, not by a caller.
        assert_ne!(c100, c203);
        let want100 = (pq_oetf(100.0 / 10_000.0) * 65535.0).round() as i64;
        assert!(
            (c100 - want100).abs() <= 1,
            "@100: got {c100} want {want100}"
        );
    }

    #[test]
    fn anchor_pq16_decode_divides_and_roundtrips() {
        // linear @ 203 → PQ16 → linear @ 203 recovers the input: the decode
        // kernel's ÷scale is the exact inverse of the encode's ×scale.
        let values = [0.05f32, 0.2, 1.0, 5.0];
        let (lin, target) = pq16_target();
        let pq = convert_buffer_with_anchor(
            &gray_rgb_f32(&values),
            values.len() as u32,
            1,
            packed_stride(values.len(), lin),
            lin,
            target,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        let back = convert_buffer_with_anchor(
            pq.as_slice().as_strided_bytes(),
            values.len() as u32,
            1,
            pq.stride(),
            target,
            lin,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        let backf: &[f32] = bytemuck::cast_slice(back.as_slice().as_strided_bytes());
        for (i, &v) in values.iter().enumerate() {
            let got = backf[i * 3];
            let rel = ((f64::from(got) - f64::from(v)) / f64::from(v)).abs();
            assert!(rel < 0.02, "roundtrip @203 at {v}: got {got} (rel {rel})");
        }
    }

    #[test]
    fn anchor_threads_through_f32_pq_slice_kernel() {
        // RGBF32 linear → RGBF32 PQ exercises the SIMD slice kernel
        // (LinearF32ToPqF32), a different code path than the u16 kernel.
        let values = [0.1f32, 1.0, 4.0];
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        let lin = PixelDescriptor::RGBF32_LINEAR.with_primaries(target.primaries);
        let pqf32 = lin.with_transfer(TransferFunction::Pq);
        let out = convert_buffer_with_anchor(
            &gray_rgb_f32(&values),
            values.len() as u32,
            1,
            packed_stride(values.len(), lin),
            lin,
            pqf32,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        let encoded: &[f32] = bytemuck::cast_slice(out.as_slice().as_strided_bytes());
        for (i, &v) in values.iter().enumerate() {
            let got = f64::from(encoded[i * 3]);
            let want = pq_oetf(f64::from(v) * 203.0 / 10_000.0);
            assert!(
                (got - want).abs() < 1e-3,
                "f32 PQ @203 at {v}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn no_anchor_default_is_unscaled() {
        // DiffuseWhite at 10000 nits ⇒ scale 1.0 ⇒ the kernel treats linear as
        // already PQ-absolute (the prior behavior). 1.0 linear → PQ code 65535.
        let (lin, target) = pq16_target();
        let out = convert_buffer_with_anchor(
            &gray_rgb_f32(&[1.0]),
            1,
            1,
            packed_stride(1, lin),
            lin,
            target,
            DiffuseWhite::new(10_000.0),
        )
        .unwrap();
        let ob = out.as_slice().as_strided_bytes();
        assert_eq!(u16::from_ne_bytes([ob[0], ob[1]]), 65535);
    }

    #[test]
    fn anchor_preserves_alpha_through_rgba_pq16() {
        // RGBA f32 linear → RGBA16 PQ: the RGB lanes take the PQ OETF + anchor;
        // alpha rides through linearly (never PQ-encoded, never anchor-scaled) —
        // the alpha-preserving `_rgba_slice` kernel path.
        let rgb_pq = PixelDescriptor::RGB16_BT2100_PQ;
        let src = PixelDescriptor::RGBAF32_LINEAR.with_primaries(rgb_pq.primaries);
        let target = PixelDescriptor::RGBA16
            .with_transfer(TransferFunction::Pq)
            .with_primaries(rgb_pq.primaries);
        let pixels = [(1.0f32, 0.5f32), (2.0, 0.25)];
        let out = convert_buffer_with_anchor(
            &gray_rgba_f32(&pixels),
            pixels.len() as u32,
            1,
            packed_stride(pixels.len(), src),
            src,
            target,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        let codes: &[u16] = bytemuck::cast_slice(out.as_slice().as_strided_bytes());
        for (i, &(g, a)) in pixels.iter().enumerate() {
            let r = i64::from(codes[i * 4]);
            let want_rgb = (pq_oetf(f64::from(g) * 203.0 / 10_000.0) * 65535.0).round() as i64;
            assert!(
                (r - want_rgb).abs() <= 1,
                "rgb @203 at {g}: got {r} want {want_rgb}"
            );
            // Alpha is linear → u16; PQ-encoding it would give a wildly wrong code.
            let alpha = codes[i * 4 + 3];
            let want_a = (f64::from(a) * 65535.0).round() as u16;
            assert_eq!(
                alpha, want_a,
                "alpha must pass through linearly: got {alpha} want {want_a}"
            );
        }
    }

    #[test]
    fn anchor_honors_source_stride() {
        // A padded source stride with sentinel padding (999.0, which would clip
        // to 65535 if it leaked) must convert identically to the packed source.
        let (lin, target) = pq16_target();
        let row_vals = [0.1f32, 1.0, 3.0];
        let width = row_vals.len() as u32;
        let rows = 2u32;
        let row = packed_stride(row_vals.len(), lin);
        let stride = row + 2 * 12; // two sentinel pixels of padding per row

        let mut packed = gray_rgb_f32(&row_vals);
        packed.extend_from_slice(&gray_rgb_f32(&row_vals));
        let want = convert_buffer_with_anchor(
            &packed,
            width,
            rows,
            row,
            lin,
            target,
            DiffuseWhite::BT2408,
        )
        .unwrap();

        let mut strided = vec![0u8; stride * rows as usize];
        for y in 0..rows as usize {
            let s = y * stride;
            strided[s..s + row].copy_from_slice(&gray_rgb_f32(&row_vals));
            for b in strided[s + row..s + stride].chunks_exact_mut(4) {
                b.copy_from_slice(&999.0f32.to_ne_bytes());
            }
        }
        let got = convert_buffer_with_anchor(
            &strided,
            width,
            rows,
            stride,
            lin,
            target,
            DiffuseWhite::BT2408,
        )
        .unwrap();
        assert_eq!(
            got.as_slice().as_strided_bytes(),
            want.as_slice().as_strided_bytes(),
            "strided source must convert identically to packed"
        );
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
        let px =
            zenpixels::InPlacePixels::new(&mut bytes, 2, 2, 12, PixelDescriptor::RGBX8_SRGB, None);
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

    // Pre-#44 these were `#[should_panic]` tests that pinned the
    // `assert_not_cmyk` ABORT behaviour. After the typed-error fix they
    // assert the same rejection — `ConvertError::NoPath { from, to }` with
    // one side being CMYK — without killing the process. The newer
    // `cmyk_input_returns_typed_error_*` tests cover the same surface plus
    // the inverse direction.
    #[test]
    fn cmyk_rejected_by_adapt_for_encode() {
        let cmyk_data = vec![0u8; 4 * 4]; // 4 pixels
        let err = adapt_for_encode(
            &cmyk_data,
            PixelDescriptor::CMYK8,
            2,
            2,
            8,
            &[PixelDescriptor::RGB8_SRGB],
        )
        .unwrap_err();
        assert!(matches!(
            *err.error(),
            ConvertError::NoPath { from, .. } if from.color_model() == ColorModel::Cmyk
        ));
    }

    #[test]
    fn cmyk_rejected_by_convert_buffer() {
        let cmyk_data = vec![0u8; 4 * 4];
        let err = convert_buffer(
            &cmyk_data,
            2,
            2,
            PixelDescriptor::CMYK8,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap_err();
        assert!(matches!(
            *err.error(),
            ConvertError::NoPath { from, .. } if from.color_model() == ColorModel::Cmyk
        ));
    }

    #[test]
    fn cmyk_rejected_by_convert_buffer_as_target() {
        let rgb_data = vec![0u8; 3 * 4];
        let err = convert_buffer(
            &rgb_data,
            2,
            2,
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::CMYK8,
        )
        .unwrap_err();
        assert!(matches!(
            *err.error(),
            ConvertError::NoPath { to, .. } if to.color_model() == ColorModel::Cmyk
        ));
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

    // ── #44.1 — CMYK rejection produces typed errors, not panics ──────────

    #[test]
    fn cmyk_input_returns_typed_error_from_adapt_for_encode() {
        // 2×1 CMYK8 = 8 bytes; the value pattern is arbitrary, it never
        // reaches the conversion loop because CMYK is rejected up front.
        let data = [0u8; 8];
        let cmyk = PixelDescriptor::CMYK8;
        let target = PixelDescriptor::RGB8_SRGB;
        let err = adapt_for_encode(&data, cmyk, 2, 1, 8, &[target]).unwrap_err();
        assert!(
            matches!(
                *err.error(),
                ConvertError::NoPath { from, .. } if from.color_model() == ColorModel::Cmyk
            ),
            "got: {:?}",
            err.error()
        );
        // Display message must carry the moxcms hint so server-side error
        // logs are actionable without pattern-matching the variant.
        let msg = format!("{}", err.error());
        assert!(
            msg.contains("CMYK") && msg.contains("moxcms"),
            "Display message lost CMYK hint: {msg}"
        );
    }

    #[test]
    fn cmyk_input_returns_typed_error_from_convert_buffer() {
        let data = [0u8; 8];
        let err = convert_buffer(
            &data,
            2,
            1,
            PixelDescriptor::CMYK8,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap_err();
        assert!(matches!(
            *err.error(),
            ConvertError::NoPath { from, .. } if from.color_model() == ColorModel::Cmyk
        ));
    }

    #[test]
    fn cmyk_target_returns_typed_error_from_convert_buffer() {
        // RGB→CMYK should also be rejected (no inverse direction either).
        let data = [0u8; 6];
        let err = convert_buffer(
            &data,
            2,
            1,
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::CMYK8,
        )
        .unwrap_err();
        assert!(matches!(
            *err.error(),
            ConvertError::NoPath { to, .. } if to.color_model() == ColorModel::Cmyk
        ));
    }

    // ── #44.2 — truncated src buffer returns typed BufferSize ────────────

    #[test]
    fn truncated_src_returns_buffer_size_error_from_adapt_for_encode() {
        // Declared 2×2 RGB8 = 12 bytes needed, only provide 6 (one row).
        let data = [255u8, 0, 0, 0, 255, 0];
        let err = adapt_for_encode(
            &data,
            PixelDescriptor::RGB8_SRGB,
            2,
            2,
            6, // packed stride; 2 rows × 6 = 12 needed
            &[PixelDescriptor::RGB8_SRGB],
        )
        .unwrap_err();
        assert!(
            matches!(
                *err.error(),
                ConvertError::BufferSize {
                    expected: 12,
                    actual: 6
                }
            ),
            "got: {:?}",
            err.error()
        );
    }

    #[test]
    fn truncated_src_returns_buffer_size_error_from_convert_buffer() {
        // Declared 4×1 RGBA8 = 16 bytes, provide 8.
        let data = [0u8; 8];
        let err = convert_buffer(
            &data,
            4,
            1,
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap_err();
        assert!(matches!(*err.error(), ConvertError::BufferSize { .. }));
    }

    // ── #44.3 — 32-bit allocation overflow surfaces typed error ─────────

    #[test]
    fn zero_rows_does_not_trigger_size_check() {
        // Empty inputs are well-defined: rows=0 means no data needed.
        let data: &[u8] = &[];
        let result = adapt_for_encode(
            data,
            PixelDescriptor::RGB8_SRGB,
            0,
            0,
            0,
            &[PixelDescriptor::RGB8_SRGB],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn extreme_rows_stride_returns_allocation_failed_not_panic() {
        // u32::MAX rows × non-trivial stride will overflow usize on 32-bit.
        // On 64-bit, `data.len()` is then the gating check (we don't actually
        // have a u32::MAX-row buffer to feed it). The discriminator is that
        // the function returns a typed error in *both* cases rather than
        // panicking with index-OOB or producing a wrap-and-corrupt allocation.
        let data = [0u8; 1];
        let err = convert_buffer(
            &data,
            1,
            u32::MAX,
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB, // identity (early-return), but src check still runs
        )
        .unwrap_err();
        assert!(
            matches!(
                *err.error(),
                ConvertError::AllocationFailed | ConvertError::BufferSize { .. }
            ),
            "got: {:?}",
            err.error()
        );
    }
}
