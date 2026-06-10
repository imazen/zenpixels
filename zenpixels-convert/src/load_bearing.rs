//! Descriptor-aware load-bearing analysis: "what parts of this buffer's
//! declared descriptor are actually carrying information?"
//!
//! Each predicate in [`crate::scan`] answers a single byte-level question
//! ("is this alpha lane all 0xFF?"); this module assembles those answers
//! into a [`LoadBearingReport`] keyed off the buffer's [`PixelDescriptor`]
//! and provides a one-call extension method on [`PixelSlice`] that runs
//! the right predicates for the descriptor and folds the results into a
//! narrower target descriptor.
//!
//! The entry points:
//!   * [`PixelSliceLoadBearingExt::determine_load_bearing`] -- analysis,
//!     no buffer modification
//!   * [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
//!     -- analysis + buffer rewrite, `None` when no narrowing is possible
//!
//! Every reduction this module reports is **bit-exact invertible**: drop
//! an all-max alpha lane and a decoder resynthesizes it; collapse
//! `R==G==B` to gray and the expansion is exact; narrow bit-replicated
//! U16 to U8 and `u8 * 0x0101` reconstructs every sample. Primaries /
//! gamut narrowing is deliberately **not** part of this analysis: it is
//! a re-encoding (EOTF decode → 3×3 matrix in linear light →
//! re-quantize) that rewrites stored pixel values, so it belongs to an
//! explicit opt-in conversion API that pairs the descriptor re-tag with
//! the buffer rewrite -- never to a descriptor-level "reduction" where
//! a re-tag without the rewrite would silently misinterpret pixels.

#[cfg(test)]
use alloc::vec::Vec;

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice,
};

use crate::scan::{self, FusedRequest};

/// What a buffer's content actually exercises about its declared
/// descriptor. Each field is `Option<T>` so it self-reports whether
/// the predicate was actually measured against this buffer:
///
/// - `Some(value)` -- the predicate ran; `value` reflects measured truth
/// - `None` -- the predicate didn't run (channel type unsupported, or the
///   field doesn't apply to this layout). Codecs should treat `None` as
///   "I don't know -- keep the conservative interpretation".
///
/// For boolean fields, the interesting signal for codecs is `Some(false)`:
/// "this dimension isn't load-bearing, it's safe to narrow". `Some(true)`
/// or `None` both mean "leave it alone".
///
/// `Default::default()` produces an all-`None` report -- the safe starting
/// state when no analysis has run.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct LoadBearingReport {
    /// `Some(true)` → at least one alpha sample is not channel-max
    /// (alpha is load-bearing). `Some(false)` → alpha can be dropped
    /// (every sample is channel-max, OR the layout has no alpha
    /// channel -- codec drops alpha either way). `None` → predicate
    /// didn't run (unsupported channel type).
    pub uses_alpha: Option<bool>,

    /// `Some(true)` → at least one pixel has differing chroma channels
    /// (R != G or G != B). `Some(false)` → no chroma variation (either
    /// R==G==B everywhere or the layout is already grayscale). `None`
    /// → predicate didn't run.
    pub uses_chroma: Option<bool>,

    /// `Some(true)` → at least one U16 sample has its low byte differ
    /// from its high byte. `Some(false)` → no information lost in
    /// U16 → U8 narrowing (either bit-replicated samples or the
    /// buffer is already at U8). `None` → predicate didn't run (F32,
    /// F16, etc.).
    pub uses_low_bits: Option<bool>,
}

impl LoadBearingReport {
    /// True if the analysis returned at least one non-`None` field --
    /// i.e. some predicate ran (or answered structurally). Codecs that
    /// need a quick "is there anything actionable here" check before
    /// consulting individual fields; `false` means the buffer's
    /// layout × channel-type combination isn't wired and the report
    /// carries no information.
    #[inline]
    pub const fn any_analyzed(&self) -> bool {
        self.uses_alpha.is_some() || self.uses_chroma.is_some() || self.uses_low_bits.is_some()
    }

    /// Produce the narrowest descriptor justified by this report.
    ///
    /// Order of reduction (each step's outcome feeds the next):
    ///   1. Channel-type narrowing (U16 → U8 when `uses_low_bits` is
    ///      false)
    ///   2. Alpha drop (when `uses_alpha` is false and the layout has
    ///      alpha)
    ///   3. Chroma drop (when `uses_chroma` is false and the layout
    ///      has chroma)
    ///
    /// Color signaling (primaries, transfer, signal range) carries over
    /// from `src` untouched -- a load-bearing reduction never re-tags
    /// color, because every reduction here keeps stored values exact.
    ///
    /// Alpha drop from `Bgra` narrows to `Rgb` -- there is no `Bgr`
    /// layout, so the buffer rewrite in
    /// [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
    /// reorders channels (B,G,R,A → R,G,B). Callers applying this
    /// descriptor with their own rewrite must do the same reorder.
    ///
    /// If a step would yield an unrepresentable `(channel_type, layout,
    /// alpha)` triple, the source format is kept.
    #[must_use]
    pub fn apply_to(&self, src: &PixelDescriptor) -> PixelDescriptor {
        let mut channel_type = src.channel_type();
        let mut layout = src.layout();
        let mut alpha = src.alpha;

        // Each step triggers ONLY on Some(false) -- the explicit
        // "not load-bearing" signal. Some(true) and None both mean
        // "leave this dimension alone".

        // 1. Channel-type narrowing.
        if matches!(self.uses_low_bits, Some(false)) && channel_type == ChannelType::U16 {
            channel_type = ChannelType::U8;
        }

        // 2. Alpha drop. Bgra → Rgb implies the B,G,R,A → R,G,B
        // channel reorder in the buffer rewrite.
        if matches!(self.uses_alpha, Some(false)) {
            layout = match layout {
                ChannelLayout::Rgba | ChannelLayout::Bgra => ChannelLayout::Rgb,
                ChannelLayout::GrayAlpha => ChannelLayout::Gray,
                other => other,
            };
            if layout != src.layout() {
                alpha = None;
            }
        }

        // 3. Chroma drop.
        if matches!(self.uses_chroma, Some(false)) {
            layout = match layout {
                ChannelLayout::Rgb => ChannelLayout::Gray,
                ChannelLayout::Rgba | ChannelLayout::Bgra => ChannelLayout::GrayAlpha,
                other => other,
            };
        }

        // Assemble the new format. PixelFormat::from_parts returns None
        // for unrepresentable triples; in that case keep the source.
        let format = PixelFormat::from_parts(channel_type, layout, alpha).unwrap_or(src.format);

        PixelDescriptor::from_pixel_format(format)
            .with_transfer(src.transfer)
            .with_primaries(src.primaries)
            .with_alpha(alpha)
            .with_signal_range(src.signal_range)
    }
}

// ── Extension trait on PixelSlice ──────────────────────────────────────

mod sealed {
    /// Seals [`super::PixelSliceLoadBearingExt`] to `PixelSlice` -- the
    /// analysis is keyed off `PixelSlice`'s descriptor + row iteration
    /// contract, so external impls have nothing valid to implement.
    pub trait Sealed {}
    impl<P> Sealed for zenpixels::PixelSlice<'_, P> {}
}

/// Run all relevant load-bearing predicates against a [`PixelSlice`] and
/// (optionally) produce a narrower buffer.
///
/// Sealed: implemented for [`PixelSlice`] only.
pub trait PixelSliceLoadBearingExt: sealed::Sealed {
    /// Run all relevant predicates and return the report. Pure analysis
    /// -- no buffer rewrite, no descriptor changes.
    ///
    /// Use [`LoadBearingReport::apply_to`] on the slice's descriptor to
    /// see what the buffer could become; use
    /// [`Self::try_reduce_to_load_bearing_format`] to actually build it.
    fn determine_load_bearing(&self) -> LoadBearingReport;

    /// Run analysis and return the rewritten buffer if any reduction is
    /// available; `None` if the buffer is already at its load-bearing
    /// minimum, the predicates couldn't run, or allocation failed.
    ///
    /// The returned [`PixelBuffer`] carries the narrowed descriptor and
    /// the buffer's standard SIMD-aligned row stride (it is not
    /// byte-tightly packed; use the buffer's own accessors or
    /// [`PixelBuffer::as_slice`] downstream).
    fn try_reduce_to_load_bearing_format(&self) -> Option<PixelBuffer>;
}

impl<P> PixelSliceLoadBearingExt for PixelSlice<'_, P> {
    fn determine_load_bearing(&self) -> LoadBearingReport {
        let descriptor = self.descriptor();
        let layout = descriptor.layout();
        let channel_type = descriptor.channel_type();

        // ── Descriptor-level alpha answers ───────────────────────
        // Two `AlphaMode`s answer the alpha question without touching
        // a single pixel:
        //   * `Undefined` (RGBX/BGRX padding): the lane bytes are
        //     meaningless -- scanning them would derive answers from
        //     garbage. Structurally droppable.
        //   * `Opaque`: the descriptor *contracts* every sample is
        //     channel-max. Trust it -- same answer a scan of a
        //     genuinely all-opaque buffer produces.
        // `Straight` and `Premultiplied` scan normally. (All
        // reductions here stay valid under premultiplication: alpha
        // only drops when uniformly max, where premultiplied ==
        // straight; `R==G==B` and bit-replication are value-exact
        // tests unaffected by what the values encode.)
        let alpha_structural: Option<Option<bool>> = if layout.has_alpha() {
            match descriptor.alpha {
                Some(AlphaMode::Undefined) | Some(AlphaMode::Opaque) => Some(Some(false)),
                _ => None,
            }
        } else {
            None
        };
        let scan_alpha = alpha_structural.is_none();

        // ── Per-pixel byte-level predicates ──────────────────────
        // Each branch returns `Some(value)` when the predicate ran
        // (or the answer is structurally trivial -- e.g. `uses_alpha
        // == Some(false)` for a layout with no alpha channel) and
        // `None` when the predicate isn't wired for this channel
        // type. Codecs treat `Some(false)` as the actionable
        // "drop this" signal.
        let (mut uses_alpha, uses_chroma) = match (layout, channel_type) {
            (ChannelLayout::Rgba | ChannelLayout::Bgra, ChannelType::U8) => {
                let fused = fused_rgba8_over_rows(
                    self,
                    FusedRequest {
                        check_opaque: scan_alpha,
                        check_grayscale: true,
                    },
                );
                (Some(!fused.is_opaque), Some(!fused.is_grayscale))
            }
            (ChannelLayout::Rgba, ChannelType::U16) => (
                Some(scan_alpha && !rows_all(self, cast_u16, scan::is_opaque_rgba16)),
                Some(!rows_all(self, cast_u16, scan::is_grayscale_rgba16)),
            ),
            (ChannelLayout::Rgb, ChannelType::U8) => (
                Some(false), // no alpha channel -- structurally not load-bearing
                Some(!rows_all(self, cast_u8, scan::is_grayscale_rgb8)),
            ),
            (ChannelLayout::Rgb, ChannelType::U16) => (
                Some(false),
                Some(!rows_all(self, cast_u16, scan::is_grayscale_rgb16)),
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U8) => (
                Some(scan_alpha && !rows_all(self, cast_u8, scan::is_opaque_ga8)),
                Some(false), // already grayscale -- no chroma to be load-bearing
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U16) => (
                Some(scan_alpha && !rows_all(self, cast_u16, scan::is_opaque_ga16)),
                Some(false),
            ),

            // Gray-anything: structurally no alpha and no chroma to
            // test. Both fields are `Some(false)` regardless of the
            // channel-type-specific predicate availability.
            (ChannelLayout::Gray, _) => (Some(false), Some(false)),

            // F32 RGB(A) / GrayAlpha -- predicates wired.
            (ChannelLayout::Rgba, ChannelType::F32) => (
                Some(scan_alpha && !rows_all(self, cast_f32, scan::is_opaque_rgba_f32)),
                Some(!rows_all(self, cast_f32, scan::is_grayscale_rgba_f32)),
            ),
            (ChannelLayout::Rgb, ChannelType::F32) => (
                Some(false),
                Some(!rows_all(self, cast_f32, scan::is_grayscale_rgb_f32)),
            ),
            (ChannelLayout::GrayAlpha, ChannelType::F32) => (
                Some(scan_alpha && !rows_all(self, cast_f32, scan::is_opaque_ga_f32)),
                Some(false),
            ),

            // F16 / Oklab / CMYK with non-Gray layout -- predicates
            // not yet wired. All fields stay `None`.
            _ => (None, None),
        };

        // Overlay the structural alpha answer (the scan, when one ran
        // at all, was told not to compute it). `uses_alpha.is_some()`
        // limits the overlay to layout × channel-type combos whose
        // predicates are wired -- unanalyzed combos stay all-`None`.
        if let Some(structural_uses) = alpha_structural
            && uses_alpha.is_some()
        {
            uses_alpha = structural_uses;
        }

        // ── Low bits (U16 → U8) ──────────────────────────────────
        let uses_low_bits = match channel_type {
            ChannelType::U16 => Some(!rows_all(
                self,
                cast_u16,
                scan::bit_replication_lossless_u16,
            )),
            // U8 is already at minimum integer depth -- structurally
            // not load-bearing in the U16-narrowing sense.
            ChannelType::U8 => Some(false),
            // F32 / F16 -- no defined narrowing without lossy
            // quantization. `None` = predicate doesn't apply.
            _ => None,
        };

        LoadBearingReport {
            uses_alpha,
            uses_chroma,
            uses_low_bits,
        }
    }

    fn try_reduce_to_load_bearing_format(&self) -> Option<PixelBuffer> {
        let src = self.descriptor();
        let report = self.determine_load_bearing();
        let target = report.apply_to(&src);
        if target == src {
            return None;
        }
        // One fallible zeroed allocation (calloc path) at the target's
        // standard aligned stride; rows are then written in place --
        // no per-pixel Vec growth anywhere in the rewrite.
        let mut out = PixelBuffer::try_new(self.width(), self.rows(), target).ok()?;
        transform_into(self, &src, &target, &mut out)?;
        Some(out)
    }
}

// ── Strided row iteration helpers ──────────────────────────────────────
//
// "Every function that operates on rows of pixels MUST natively support
// strided rows, at no additional runtime cost on the tightly-packed
// path." (Per global CLAUDE.md.) These helpers implement that contract:
// when the slice is contiguous, ONE call to the inner predicate; when
// strided, one call per row. Output of the predicate AND-reduces across
// rows with early-exit on first false.

/// AND-reduce a slice-level predicate across rows: one call on the
/// contiguous fast path, one call per row when strided, early-exit on
/// the first `false`. `cast` reinterprets each row's bytes as the
/// predicate's element type (`cast_u8` / `cast_u16` / `cast_f32`).
#[inline]
fn rows_all<P, T, F>(slice: &PixelSlice<'_, P>, cast: fn(&[u8]) -> &[T], predicate: F) -> bool
where
    T: 'static,
    F: Fn(&[T]) -> bool,
{
    if let Some(bytes) = slice.as_contiguous_bytes() {
        predicate(cast(bytes))
    } else {
        for y in 0..slice.rows() {
            if !predicate(cast(slice.row(y))) {
                return false;
            }
        }
        true
    }
}

/// Row-aware fused predicate for RGBA8/Bgra8. Drops finished checks
/// from the next row's request so per-row work shrinks as flags flip.
/// Single fused call on contiguous buffers. Unrequested checks come
/// back `false` ("not computed"), mirroring `FusedResult` semantics.
fn fused_rgba8_over_rows<P>(slice: &PixelSlice<'_, P>, request: FusedRequest) -> scan::FusedResult {
    if let Some(bytes) = slice.as_contiguous_bytes() {
        return scan::fused_predicates_rgba8_cg(bytes, request);
    }
    let mut req = request;
    let mut total = scan::FusedResult {
        is_opaque: req.check_opaque,
        is_grayscale: req.check_grayscale,
    };
    for y in 0..slice.rows() {
        if !req.check_opaque && !req.check_grayscale {
            break;
        }
        let row = slice.row(y);
        let r = scan::fused_predicates_rgba8_cg(row, req);
        if req.check_opaque && !r.is_opaque {
            total.is_opaque = false;
            req.check_opaque = false;
        }
        if req.check_grayscale && !r.is_grayscale {
            total.is_grayscale = false;
            req.check_grayscale = false;
        }
    }
    total
}

// ── Helpers ────────────────────────────────────────────────────────────

fn cast_u8(bytes: &[u8]) -> &[u8] {
    bytes
}

fn cast_u16(bytes: &[u8]) -> &[u16] {
    bytemuck::cast_slice(bytes)
}

fn cast_f32(bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(bytes)
}

/// Fill `out` (pre-zeroed, target descriptor, aligned stride) from
/// `slice`, row by row. Strided input costs nothing extra -- the loop
/// is per-row either way. Returns `None` for descriptor pairs this
/// module doesn't know how to rewrite.
///
/// Every transition is a pure byte selection -- no sample value
/// changes. The two RGBA-family alpha drops delegate to `garb`'s SIMD
/// swizzles; the remaining selections are fixed-stride copy loops that
/// LLVM turns into shuffles (and they only run when the corresponding
/// scan proved the dropped bytes redundant).
fn transform_into<P>(
    slice: &PixelSlice<'_, P>,
    src: &PixelDescriptor,
    dst: &PixelDescriptor,
    out: &mut PixelBuffer,
) -> Option<()> {
    let src_ct = src.channel_type();
    let dst_ct = dst.channel_type();
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    // U16 → U8 narrowing is the only channel-type transition. The
    // bit-replication precondition (`uses_low_bits == Some(false)`)
    // proves both bytes of every sample are equal, so byte 0 is the
    // (replicated) high byte regardless of endianness.
    let narrow16 = src_ct == ChannelType::U16 && dst_ct == ChannelType::U8;
    if !narrow16 && src_ct != dst_ct {
        return None;
    }

    // Source-channel selection map for the layout transition. Identity
    // when only the channel type narrows.
    const IDENTITY: [usize; 4] = [0, 1, 2, 3];
    let in_ch = src_layout.channels();
    let map: &[usize] = match (src_layout, dst_layout) {
        _ if src_layout == dst_layout => &IDENTITY[..in_ch],
        (ChannelLayout::Rgba, ChannelLayout::Rgb) => &[0, 1, 2],
        // Bgra stores B,G,R,A -- dropping alpha into the Rgb layout
        // requires the B↔R reorder, not a prefix copy.
        (ChannelLayout::Bgra, ChannelLayout::Rgb) => &[2, 1, 0],
        // Channel 0 is R for Rgba and B for Bgra; either is the gray
        // value because these transitions only fire when R == G == B
        // held for every pixel.
        (ChannelLayout::Rgba | ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => &[0, 3],
        (ChannelLayout::Rgba | ChannelLayout::Bgra, ChannelLayout::Gray) => &[0],
        (ChannelLayout::Rgb, ChannelLayout::Gray) => &[0],
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => &[0],
        _ => return None,
    };

    let mut out_rows = out.as_slice_mut();
    for y in 0..slice.rows() {
        let row_in = slice.row(y);
        let row_out = out_rows.row_mut(y);
        if narrow16 {
            select_row_u16_to_u8(row_in, row_out, in_ch, map);
        } else {
            match (dst_ct.byte_size(), src_layout, dst_layout) {
                (1, ChannelLayout::Rgba, ChannelLayout::Rgb) => {
                    garb::bytes::rgba_to_rgb(row_in, row_out).ok()?;
                }
                (1, ChannelLayout::Bgra, ChannelLayout::Rgb) => {
                    garb::bytes::bgra_to_rgb(row_in, row_out).ok()?;
                }
                (1, ..) => select_row::<1>(row_in, row_out, in_ch, map),
                (2, ..) => select_row::<2>(row_in, row_out, in_ch, map),
                (4, ..) => select_row::<4>(row_in, row_out, in_ch, map),
                _ => return None,
            }
        }
    }
    Some(())
}

/// Copy the `map`-selected channels (element size `E` bytes) of each
/// pixel in `row_in` into `row_out`. Fixed `E` + `chunks_exact` keeps
/// the loop bounds-check-free and auto-vectorizable.
#[inline]
fn select_row<const E: usize>(row_in: &[u8], row_out: &mut [u8], in_ch: usize, map: &[usize]) {
    let out_px = map.len() * E;
    let in_px = in_ch * E;
    for (dst, src) in row_out
        .chunks_exact_mut(out_px)
        .zip(row_in.chunks_exact(in_px))
    {
        for (k, &c) in map.iter().enumerate() {
            dst[k * E..(k + 1) * E].copy_from_slice(&src[c * E..c * E + E]);
        }
    }
}

/// Like [`select_row`] but narrows each selected u16 sample to u8 by
/// taking byte 0 (valid because bit-replication was proven first).
#[inline]
fn select_row_u16_to_u8(row_in: &[u8], row_out: &mut [u8], in_ch: usize, map: &[usize]) {
    let in_px = in_ch * 2;
    for (dst, src) in row_out
        .chunks_exact_mut(map.len())
        .zip(row_in.chunks_exact(in_px))
    {
        for (k, &c) in map.iter().enumerate() {
            dst[k] = src[c * 2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::{ColorPrimaries, PixelSlice, TransferFunction};

    fn make_slice<'a>(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> PixelSlice<'a> {
        let descriptor =
            PixelDescriptor::from_pixel_format(format).with_transfer(TransferFunction::Srgb);
        let stride = width as usize * format.bytes_per_pixel();
        PixelSlice::new(bytes, width, height, stride, descriptor).unwrap()
    }

    fn make_slice_with_primaries<'a>(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
        primaries: ColorPrimaries,
    ) -> PixelSlice<'a> {
        let descriptor = PixelDescriptor::from_pixel_format(format)
            .with_transfer(TransferFunction::Srgb)
            .with_primaries(primaries);
        let stride = width as usize * format.bytes_per_pixel();
        PixelSlice::new(bytes, width, height, stride, descriptor).unwrap()
    }

    /// Analysis + combiner: the descriptor this buffer would reduce to.
    fn reduced(slice: &PixelSlice<'_>) -> PixelDescriptor {
        slice.determine_load_bearing().apply_to(&slice.descriptor())
    }

    // ── Reductions on common channel types ────────────────────────

    #[test]
    fn rgba8_all_opaque_gray_reduces_to_gray8() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let g = (i * 30) as u8;
                [g, g, g, 255]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        // analyzed bool removed
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(false));

        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::Gray8);
    }

    #[test]
    fn rgba8_with_real_color_keeps_rgba_drops_alpha() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                [
                    (i * 60 + 10) as u8,
                    (i * 30 + 50) as u8,
                    (i * 90 + 20) as u8,
                    255,
                ]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        // analyzed bool removed
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(true));

        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::Rgb8);
    }

    #[test]
    fn rgba8_alpha_mix_0_and_255_reports_binary() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let a = if i & 1 == 0 { 0 } else { 255 };
                [50, 50, 50, a]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(true), "alpha varies → load-bearing");
        assert_eq!(r.uses_chroma, Some(false));
    }

    #[test]
    fn rgba16_bit_replicated_reduces_to_rgba8() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let r = (i * 60) as u8;
                let g = (i * 30 + 10) as u8;
                let b = (i * 80 + 5) as u8;
                let a = 0xFF;
                [r, r, g, g, b, b, a, a]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba16);
        let r = slice.determine_load_bearing();
        // analyzed bool removed
        assert_eq!(r.uses_low_bits, Some(false));
        assert_eq!(r.uses_alpha, Some(false));
        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::Rgb8);
    }

    #[test]
    fn rgba16_actual_high_precision_keeps_u16() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let r_lo = (i * 17 + 1) as u8;
                let r_hi = (i * 60) as u8;
                [r_hi, r_lo, 0, 0, 0, 0, 0xFF, 0xFF]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba16);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_low_bits, Some(true));
    }

    // ── Sub-byte gray detection ──────────────────────────────────

    // ── try_reduce ─────────────────────────────────────────────

    #[test]
    fn try_reduce_returns_some_when_reduction_available() {
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let result = slice.try_reduce_to_load_bearing_format();
        let out = result.expect("should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        assert_eq!(out.as_slice().row(0), &[0u8, 30, 60, 90]);
    }

    #[test]
    fn try_reduce_returns_none_when_already_minimal() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| [i * 60, 100, 200, i * 40 + 1])
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        assert!(slice.try_reduce_to_load_bearing_format().is_none());
    }

    // ── analyzed flag ─────────────────────────────────────────────

    #[test]
    fn default_report_is_fully_unanalyzed() {
        // With per-field Option semantics, the default report is
        // "nothing was checked" — every field is None. apply_to on a
        // None-only report is identity (no Some(false) signals).
        let r = LoadBearingReport::default();
        assert_eq!(r.uses_alpha, None);
        assert_eq!(r.uses_chroma, None);
        assert_eq!(r.uses_low_bits, None);
        assert!(!r.any_analyzed());
    }

    #[test]
    fn any_analyzed_fires_when_at_least_one_field_set() {
        let mut r = LoadBearingReport::default();
        assert!(!r.any_analyzed());
        r.uses_alpha = Some(true);
        assert!(r.any_analyzed(), "any_analyzed fires for any Some");
        r.uses_alpha = None;
        r.uses_low_bits = Some(false);
        assert!(r.any_analyzed(), "any_analyzed fires on low-bits too");
    }

    // ── Color signaling is never re-tagged ───────────────────────

    #[test]
    fn wide_primaries_tag_is_preserved_and_ignored_by_analysis() {
        // A P3-tagged buffer analyzes exactly like an sRGB-tagged one
        // (the analysis is value-exact and color-space-blind), and the
        // reduced descriptor keeps the P3 tag — load-bearing reduction
        // never re-tags primaries, because a re-tag without a pixel
        // rewrite would reinterpret the buffer in the wrong space.
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let g = (i * 30) as u8;
                [g, g, g, 255]
            })
            .collect();
        let p3 =
            make_slice_with_primaries(&bytes, 4, 1, PixelFormat::Rgba8, ColorPrimaries::DisplayP3);
        let srgb =
            make_slice_with_primaries(&bytes, 4, 1, PixelFormat::Rgba8, ColorPrimaries::Bt709);

        let r_p3 = p3.determine_load_bearing();
        let r_srgb = srgb.determine_load_bearing();
        assert_eq!(r_p3.uses_alpha, r_srgb.uses_alpha);
        assert_eq!(r_p3.uses_chroma, r_srgb.uses_chroma);

        let out = p3
            .try_reduce_to_load_bearing_format()
            .expect("gray+opaque should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        assert_eq!(
            out.descriptor().primaries,
            ColorPrimaries::DisplayP3,
            "primaries tag must carry over untouched"
        );
        // Bit-exact: the gray bytes are the original channel values.
        assert_eq!(out.as_slice().row(0), &[0u8, 30, 60, 90]);
    }

    // ── Apply combiner ──────────────────────────────────────────

    #[test]
    fn apply_to_no_op_on_fully_load_bearing() {
        let src = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8);
        let r = LoadBearingReport::default();
        assert_eq!(r.apply_to(&src), src);
    }

    #[test]
    fn ga8_opaque_reduces_to_gray8() {
        let bytes = [10u8, 255, 50, 255, 100, 255];
        let slice = make_slice(&bytes, 3, 1, PixelFormat::GrayA8);
        assert_eq!(reduced(&slice).format, PixelFormat::Gray8);
    }

    #[test]
    fn rgba16_grayscale_alpha_replicated_reduces_to_gray8() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let g = (i * 60) as u8;
                [g, g, g, g, g, g, 0xFF, 0xFF]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba16);
        assert_eq!(reduced(&slice).format, PixelFormat::Gray8);
    }

    // ── AlphaMode-driven structural answers ─────────────────────

    #[test]
    fn undefined_alpha_padding_is_structurally_droppable() {
        // RGBX-style buffer: lane 3 is garbage padding (0x7B), NOT an
        // alpha channel. The analysis must not scan it — uses_alpha
        // answers from the descriptor and the padding never poisons
        // the result; alpha_is_binary doesn't apply.
        let bytes = [10u8, 20, 30, 0x7B, 40, 50, 60, 0x01];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_alpha(Some(AlphaMode::Undefined));
        let slice = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_alpha,
            Some(false),
            "padding lane is never load-bearing"
        );
        assert_eq!(r.uses_chroma, Some(true), "chroma still measured");
        // try_reduce drops the padding lane.
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("padding drop is a reduction");
        assert_eq!(out.descriptor().format, PixelFormat::Rgb8);
        assert_eq!(out.as_slice().row(0), &[10u8, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn declared_opaque_alpha_is_trusted_without_scanning() {
        // AlphaMode::Opaque is a descriptor-level contract: every alpha
        // sample is channel-max. The analysis trusts it (mirroring what
        // a scan of a genuinely all-opaque buffer reports) instead of
        // re-verifying per pixel.
        let bytes = [10u8, 10, 10, 255, 20, 20, 20, 255];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_alpha(Some(AlphaMode::Opaque));
        let slice = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(false), "chroma still measured");
    }

    #[test]
    fn premultiplied_alpha_scans_like_straight() {
        // Premultiplied buffers run the same value-exact predicates:
        // alpha only drops when uniformly max (premul == straight
        // there), and varying premultiplied alpha stays load-bearing.
        let bytes = [10u8, 10, 10, 128, 20, 20, 20, 64];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_alpha(Some(AlphaMode::Premultiplied));
        let slice = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_alpha,
            Some(true),
            "varying premul alpha is load-bearing"
        );
    }

    // ── Strided-row tests ──────────────────────────────────────
    //
    // These build a buffer with stride > width × bpp (i.e. padding
    // between rows) and verify that:
    //   1. `determine_load_bearing` runs the predicates per-row and
    //      reaches the same answer as the equivalent contiguous buffer
    //   2. `try_reduce_to_load_bearing_format` produces the same
    //      tightly-packed output regardless of input stride
    //   3. The padding bytes (which contain garbage that would poison
    //      a contiguous-only predicate) don't affect the result

    /// Build a strided RGBA8 buffer: each row's `width × 4` pixel bytes
    /// are followed by `padding_bytes` of garbage. Returns the byte
    /// buffer and the stride in bytes.
    fn build_strided_rgba8(
        width: u32,
        height: u32,
        padding_bytes: usize,
        mut pixel_at: impl FnMut(u32, u32) -> [u8; 4],
    ) -> (Vec<u8>, usize) {
        let row_pixels = width as usize * 4;
        let stride = row_pixels + padding_bytes;
        let mut buf = vec![0xAAu8; stride * height as usize]; // 0xAA garbage
        for y in 0..height {
            for x in 0..width {
                let p = pixel_at(x, y);
                let off = y as usize * stride + x as usize * 4;
                buf[off..off + 4].copy_from_slice(&p);
            }
            // Stamp obvious garbage in the padding to catch leaks.
            for k in row_pixels..stride {
                buf[y as usize * stride + k] = 0xCD;
            }
        }
        (buf, stride)
    }

    fn slice_from_strided<'a>(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
    ) -> PixelSlice<'a> {
        let descriptor =
            PixelDescriptor::from_pixel_format(format).with_transfer(TransferFunction::Srgb);
        PixelSlice::new(bytes, width, height, stride, descriptor).unwrap()
    }

    #[test]
    fn strided_rgba8_all_opaque_gray_reduces_correctly() {
        // 4 rows × 4 pixels, 32 bytes of garbage per row of padding.
        let (buf, stride) = build_strided_rgba8(4, 4, 32, |x, y| {
            let g = ((x + y) * 30) as u8;
            [g, g, g, 255]
        });
        let slice = slice_from_strided(&buf, 4, 4, stride, PixelFormat::Rgba8);
        assert!(!slice.is_contiguous(), "test fixture must be strided");
        let r = slice.determine_load_bearing();
        // analyzed bool removed
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(false));
        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::Gray8);
    }

    #[test]
    fn strided_rgba8_garbage_padding_doesnt_poison_predicates() {
        // Pixel content makes the buffer NOT all-opaque (alpha=128).
        // The padding bytes (0xCD) would falsely look like "alpha != 255"
        // if the predicate accidentally read them. Verify the trait
        // dispatch reads only pixel bytes, not stride.
        let (buf, stride) = build_strided_rgba8(8, 3, 16, |_x, _y| [50, 50, 50, 255]);
        let slice = slice_from_strided(&buf, 8, 3, stride, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_alpha,
            Some(false),
            "alpha is uniformly 255 -- must not be confused by 0xCD padding"
        );
        // Same buffer but with one real non-opaque pixel -- predicate should fire.
        let (buf, stride) = build_strided_rgba8(8, 3, 16, |x, y| {
            if x == 2 && y == 1 {
                [10, 10, 10, 0]
            } else {
                [50, 50, 50, 255]
            }
        });
        let slice = slice_from_strided(&buf, 8, 3, stride, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_alpha,
            Some(true),
            "real transparent pixel must be detected"
        );
    }

    #[test]
    fn strided_rgba8_try_reduce_produces_tight_output() {
        // 4 rows × 4 pixels grayscale opaque → reduces to Gray8 tight.
        let (buf, stride) = build_strided_rgba8(4, 4, 16, |x, y| {
            let g = ((x + y) * 20) as u8;
            [g, g, g, 255]
        });
        let slice = slice_from_strided(&buf, 4, 4, stride, PixelFormat::Rgba8);
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("strided buffer should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        // Logical content survives independent of the output stride.
        let view = out.as_slice();
        for y in 0..4u32 {
            let row = view.row(y);
            for (x, &g) in row.iter().enumerate() {
                let expected = ((x as u32 + y) * 20) as u8;
                assert_eq!(g, expected, "gray byte at ({x},{y}) wrong");
            }
        }
    }

    #[test]
    fn strided_rgba8_matches_contiguous_result() {
        // Build the same logical content as a contiguous and a strided
        // slice; verify the report is identical.
        fn fill(x: u32, y: u32) -> [u8; 4] {
            [(x * 30) as u8, (y * 50) as u8, ((x + y) * 11) as u8, 255]
        }
        let width = 6;
        let height = 5;

        // Contiguous version
        let mut contig = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                contig.extend_from_slice(&fill(x, y));
            }
        }
        let contig_slice = make_slice(&contig, width, height, PixelFormat::Rgba8);

        // Strided version (with garbage padding)
        let (strided, stride) = build_strided_rgba8(width, height, 24, fill);
        let strided_slice = slice_from_strided(&strided, width, height, stride, PixelFormat::Rgba8);

        let r_contig = contig_slice.determine_load_bearing();
        let r_strided = strided_slice.determine_load_bearing();

        // Compare every analytical field.
        assert_eq!(r_contig.any_analyzed(), r_strided.any_analyzed());
        assert_eq!(r_contig.uses_alpha, r_strided.uses_alpha);
        assert_eq!(r_contig.uses_chroma, r_strided.uses_chroma);
        assert_eq!(r_contig.uses_low_bits, r_strided.uses_low_bits);
    }

    // ── F32 load_bearing tests ────────────────────────────────

    fn make_f32_slice<'a>(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
        transfer: TransferFunction,
    ) -> PixelSlice<'a> {
        let descriptor = PixelDescriptor::from_pixel_format(format).with_transfer(transfer);
        let stride = width as usize * format.bytes_per_pixel();
        PixelSlice::new(bytes, width, height, stride, descriptor).unwrap()
    }

    #[test]
    fn rgba_f32_all_opaque_gray_reduces_to_gray_f32() {
        // 4 RGBA f32 pixels: gray + opaque.
        let pixels: [f32; 16] = [
            0.1, 0.1, 0.1, 1.0, //
            0.5, 0.5, 0.5, 1.0, //
            0.9, 0.9, 0.9, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 4, 1, PixelFormat::RgbaF32, TransferFunction::Linear);
        let r = slice.determine_load_bearing();
        // analyzed bool removed
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(false));

        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::GrayF32);
    }

    #[test]
    fn rgba_f32_with_real_color_reduces_to_rgb_f32() {
        let pixels: [f32; 16] = [
            0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5, 1.0, 1.0,
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 4, 1, PixelFormat::RgbaF32, TransferFunction::Linear);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(true));

        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::RgbF32);
    }

    #[test]
    fn rgba_f32_with_intermediate_alpha_keeps_alpha() {
        let pixels: [f32; 12] = [0.5, 0.5, 0.5, 0.25, 0.7, 0.7, 0.7, 0.5, 0.3, 0.3, 0.3, 0.75];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 3, 1, PixelFormat::RgbaF32, TransferFunction::Linear);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(true));
        assert_eq!(r.uses_chroma, Some(false));

        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::GrayAF32);
    }

    #[test]
    fn try_reduce_rgba_f32_to_gray_f32() {
        let pixels: [f32; 16] = [
            0.1, 0.1, 0.1, 1.0, //
            0.5, 0.5, 0.5, 1.0, //
            0.9, 0.9, 0.9, 1.0, //
            0.4, 0.4, 0.4, 1.0,
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 4, 1, PixelFormat::RgbaF32, TransferFunction::Linear);
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::GrayF32);
        let view = out.as_slice();
        let gray: &[f32] = bytemuck::cast_slice(view.row(0));
        assert_eq!(gray, &[0.1, 0.5, 0.9, 0.4]);
    }

    #[test]
    fn linear_f32_wide_primaries_reduce_keeps_tag_and_values() {
        // P3-tagged linear f32 gray+opaque: reduces structurally
        // (alpha drop + chroma collapse) with values untouched and the
        // P3 tag carried over -- no primaries re-tag, no matrix.
        let pixels: [f32; 16] = [
            0.5, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 1.0, 0.75, 0.75, 0.75, 1.0, 0.1, 0.1, 0.1, 1.0,
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::RgbaF32)
            .with_transfer(TransferFunction::Linear)
            .with_primaries(ColorPrimaries::DisplayP3);
        let slice = PixelSlice::new(bytes, 4, 1, 4 * 16, descriptor).unwrap();
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::GrayF32);
        assert_eq!(out.descriptor().primaries, ColorPrimaries::DisplayP3);
        let view = out.as_slice();
        let gray: &[f32] = bytemuck::cast_slice(view.row(0));
        assert_eq!(gray, &[0.5_f32, 0.25, 0.75, 0.1], "values bit-exact");
    }

    #[test]
    fn ga_f32_opaque_reduces_to_gray_f32() {
        let pixels: [f32; 6] = [0.1, 1.0, 0.5, 1.0, 0.9, 1.0];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 3, 1, PixelFormat::GrayAF32, TransferFunction::Linear);
        assert_eq!(reduced(&slice).format, PixelFormat::GrayF32);
    }

    #[test]
    fn rgb_f32_grayscale_reduces_to_gray_f32() {
        let pixels: [f32; 9] = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 3, 1, PixelFormat::RgbF32, TransferFunction::Linear);
        assert_eq!(reduced(&slice).format, PixelFormat::GrayF32);
    }

    // ── Edge cases: idempotency ───────────────────────────────
    //
    // apply_to a report twice should be idempotent -- running the
    // narrower descriptor through the same report shouldn't narrow
    // further (it's already at the report's target). This catches
    // bugs where apply_to has hidden state or order-dependent loops.

    #[test]
    fn apply_to_is_idempotent() {
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        let target_a = r.apply_to(&slice.descriptor());
        let target_b = r.apply_to(&target_a);
        assert_eq!(
            target_a, target_b,
            "apply_to twice must equal apply_to once"
        );
    }

    #[test]
    fn apply_to_no_op_on_already_minimal_gray8() {
        // Gray8 has nothing to reduce -- report says everything is
        // false / None, apply_to should return the source unchanged.
        let bytes = [50u8, 100, 150, 200];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(false));
        assert_eq!(r.uses_low_bits, Some(false));
        let target = r.apply_to(&slice.descriptor());
        assert_eq!(target, slice.descriptor());
    }

    // ── Edge cases: trait method consistency ──────────────────
    //
    // try_reduce_to_load_bearing_format's returned descriptor should
    // match report.apply_to(descriptor). Running them independently
    // must produce the same target.

    #[test]
    fn try_reduce_descriptor_matches_determine_reduced() {
        let bytes: Vec<u8> = (0..8).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 8, 1, PixelFormat::Rgba8);
        let determined = reduced(&slice);
        let out = slice.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(determined, out.descriptor());
    }

    #[test]
    fn try_reduce_returns_none_when_descriptor_unchanged() {
        // Gray8 with 8-bit-needing values -- nothing to reduce.
        let bytes = [50u8, 100, 150, 200];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        assert!(slice.try_reduce_to_load_bearing_format().is_none());
        // The combiner agrees: same descriptor back.
        assert_eq!(reduced(&slice), slice.descriptor());
    }

    // ── Edge cases: 1×1 / single-row / single-column inputs ───

    #[test]
    fn single_pixel_inputs_for_each_layout() {
        // 1×1 Rgba8: opaque + gray → reduces to Gray8.
        let s = make_slice(&[100u8, 100, 100, 255], 1, 1, PixelFormat::Rgba8);
        assert_eq!(reduced(&s).format, PixelFormat::Gray8);

        // 1×1 Rgb8 with R=G=B → reduces to Gray8.
        let s = make_slice(&[42u8, 42, 42], 1, 1, PixelFormat::Rgb8);
        assert_eq!(reduced(&s).format, PixelFormat::Gray8);

        // 1×1 GrayA8 opaque → Gray8.
        let s = make_slice(&[42u8, 255], 1, 1, PixelFormat::GrayA8);
        assert_eq!(reduced(&s).format, PixelFormat::Gray8);

        // 1×1 Gray8 -- no reduction available.
        let s = make_slice(&[42u8], 1, 1, PixelFormat::Gray8);
        assert_eq!(reduced(&s), s.descriptor());
    }

    #[test]
    fn single_row_tall_buffer() {
        // 1 row, many cols -- exercises the per-row loop with one pass.
        let bytes: Vec<u8> = (0..32).flat_map(|i| [i * 7, i * 7, i * 7, 255]).collect();
        let s = make_slice(&bytes, 32, 1, PixelFormat::Rgba8);
        assert_eq!(reduced(&s).format, PixelFormat::Gray8);
    }

    #[test]
    fn single_col_tall_buffer() {
        // 1 col, many rows -- heavily strided territory.
        let height = 16u32;
        let width = 1u32;
        let stride = 32; // 1 byte content + 31 bytes padding per row
        let mut buf = vec![0xAAu8; stride * height as usize];
        for y in 0..height {
            buf[y as usize * stride] = (y * 7) as u8;
        }
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Gray8)
            .with_transfer(TransferFunction::Srgb);
        let s = PixelSlice::new(&buf, width, height, stride, descriptor).unwrap();
        assert!(!s.is_contiguous());
        // Should run without panicking; at least the structural
        // bools (alpha, chroma) populate for any U8 layout.
        assert!(s.determine_load_bearing().any_analyzed());
    }

    // ── Edge cases: full PixelFormat matrix ──────────────────
    //
    // Every PixelFormat is either analyzed=true (predicates run) or
    // analyzed=false (explicit unsupported). No format should panic.

    fn dummy_bytes_for(format: PixelFormat) -> Vec<u8> {
        // 1×1 buffer of the right byte size, all zeros.
        vec![0u8; format.bytes_per_pixel()]
    }

    #[test]
    fn analyzed_status_for_every_pixel_format() {
        // U8 layouts: should analyze (all have SIMD predicate paths).
        for fmt in [
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Bgra8,
            PixelFormat::Gray8,
            PixelFormat::GrayA8,
        ] {
            let bytes = dummy_bytes_for(fmt);
            let s = make_slice(&bytes, 1, 1, fmt);
            assert!(
                s.determine_load_bearing().any_analyzed(),
                "{fmt:?} should produce at least one Some field"
            );
        }
        // U16 layouts: should analyze.
        for fmt in [
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::Gray16,
            PixelFormat::GrayA16,
        ] {
            let bytes = dummy_bytes_for(fmt);
            let s = make_slice(&bytes, 1, 1, fmt);
            assert!(
                s.determine_load_bearing().any_analyzed(),
                "{fmt:?} should produce at least one Some field"
            );
        }
        // F32 RGB(A) / GA -- should analyze.
        for fmt in [
            PixelFormat::RgbF32,
            PixelFormat::RgbaF32,
            PixelFormat::GrayAF32,
        ] {
            let bytes = dummy_bytes_for(fmt);
            let s = make_slice(&bytes, 1, 1, fmt);
            assert!(
                s.determine_load_bearing().any_analyzed(),
                "{fmt:?} should produce at least one Some field"
            );
        }
        // Gray-layout formats analyze trivially regardless of channel
        // type -- there's no chroma or alpha to test (those fields are
        // structurally absent), so the report's bools are valid even
        // for channel types whose byte-level predicates aren't wired.
        for fmt in [PixelFormat::GrayF32, PixelFormat::GrayF16] {
            let bytes = dummy_bytes_for(fmt);
            let s = make_slice(&bytes, 1, 1, fmt);
            // Gray-layout formats produce Some(false) for both
            // alpha and chroma regardless of channel type -- the
            // structural answer is valid even when channel-type
            // predicates aren't wired.
            let r = s.determine_load_bearing();
            assert_eq!(r.uses_alpha, Some(false), "{fmt:?} alpha");
            assert_eq!(r.uses_chroma, Some(false), "{fmt:?} chroma");
        }

        // F16 / Oklab / CMYK with non-trivial layouts -- unanalyzed for
        // v0 because their byte-level predicates aren't wired yet.
        for fmt in [
            PixelFormat::RgbF16,
            PixelFormat::RgbaF16,
            PixelFormat::GrayAF16,
            PixelFormat::OklabF32,
            PixelFormat::OklabaF32,
            PixelFormat::Cmyk8,
        ] {
            let bytes = dummy_bytes_for(fmt);
            let s = make_slice(&bytes, 1, 1, fmt);
            let r = s.determine_load_bearing();
            // No predicate ran for this layout × channel-type combo --
            // every field stays None.
            assert_eq!(r.uses_alpha, None, "{fmt:?} alpha should be None");
            assert_eq!(r.uses_chroma, None, "{fmt:?} chroma should be None");
        }
    }

    // ── Edge cases: Bgra alpha-drop reorder ───────────────────

    #[test]
    fn bgra8_opaque_color_reduces_to_rgb8_with_reorder() {
        // Bgra stores B,G,R,A. Alpha-drop narrows to Rgb -- and the
        // buffer rewrite must reorder channels, not prefix-copy.
        // Pixel 0: B=50, G=100, R=150; pixel 1: B=60, G=110, R=160.
        let bytes = [50u8, 100, 150, 255, 60, 110, 160, 255];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Bgra8)
            .with_transfer(TransferFunction::Srgb);
        let s = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = s.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(false));
        assert_eq!(r.uses_chroma, Some(true));
        let target = r.apply_to(&s.descriptor());
        assert_eq!(target.format, PixelFormat::Rgb8);

        let out = s
            .try_reduce_to_load_bearing_format()
            .expect("opaque Bgra8 should reduce");
        assert_eq!(out.descriptor().format, PixelFormat::Rgb8);
        assert_eq!(
            out.as_slice().row(0),
            &[150u8, 100, 50, 160, 110, 60],
            "B,G,R,A → R,G,B requires the B↔R swap"
        );
    }

    #[test]
    fn bgra8_grayscale_collapses_to_gray_alpha8() {
        // R==G==B, alpha varying -- should collapse to GrayA8 even
        // for Bgra8 source (chroma drop, alpha kept).
        let bytes = [42u8, 42, 42, 100, 99, 99, 99, 200];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Bgra8)
            .with_transfer(TransferFunction::Srgb);
        let s = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = s.determine_load_bearing();
        assert_eq!(reduced(&s).format, PixelFormat::GrayA8);
        assert_eq!(r.uses_chroma, Some(false));
        // Rewrite keeps gray + alpha pairs.
        let out = s.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(out.descriptor().format, PixelFormat::GrayA8);
        assert_eq!(out.as_slice().row(0), &[42u8, 100, 99, 200]);
    }

    // ── Edge cases: report.fully_load_bearing as starting state ─

    #[test]
    fn fully_load_bearing_apply_to_is_identity() {
        // Default report → no narrowing. apply_to produces input.
        let r = LoadBearingReport::default();
        for fmt in [
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgba16,
            PixelFormat::GrayAF32,
        ] {
            let src = PixelDescriptor::from_pixel_format(fmt);
            assert_eq!(r.apply_to(&src), src, "{fmt:?} identity broke");
        }
    }

    // ── Edge cases: zero-row buffers ─────────────────────────
    //
    // 0×0 / 0×N / N×0 buffers -- width or rows = 0 means no pixels.
    // Predicates should return vacuous-true; report should still run.

    #[test]
    fn zero_pixel_buffer_analyzes_with_vacuous_truth() {
        // Empty bytes via a 0×0 image (no rows, no width).
        let bytes: [u8; 0] = [];
        // PixelSlice may not allow width=0 directly; build a 1-row
        // slice with 0 effective width via stride.
        // Use rows=1, width=0 if validate_slice allows.
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb);
        if let Ok(s) = PixelSlice::new(&bytes, 0, 0, 0, descriptor) {
            let r = s.determine_load_bearing();
            // No pixels: every "uses" is vacuously false (nothing to
            // demand the channel).
            // analyzed bool removed
            assert_eq!(r.uses_alpha, Some(false));
            assert_eq!(r.uses_chroma, Some(false));
        }
        // Some validators reject 0-dimensional descriptors. If so,
        // we don't lose semantics -- codecs won't see this case in
        // practice. The test passes either way.
    }

    // ── Sanity: every layout's reduced format round-trips ────

    #[test]
    fn every_reduction_target_is_constructable() {
        // For each non-trivial reduction, build a buffer that triggers
        // it and verify try_reduce produces a Vec<u8> of the right size.
        struct Case {
            src: PixelFormat,
            bytes: Vec<u8>,
            width: u32,
            height: u32,
            expect_format: PixelFormat,
            expect_size: usize,
        }
        let cases = vec![
            Case {
                src: PixelFormat::Rgba8,
                bytes: vec![10, 10, 10, 255, 20, 20, 20, 255],
                width: 2,
                height: 1,
                expect_format: PixelFormat::Gray8,
                expect_size: 2,
            },
            Case {
                src: PixelFormat::Rgba8,
                bytes: vec![10, 20, 30, 255, 40, 50, 60, 255],
                width: 2,
                height: 1,
                expect_format: PixelFormat::Rgb8,
                expect_size: 6,
            },
            Case {
                src: PixelFormat::GrayA8,
                bytes: vec![10, 255, 50, 255],
                width: 2,
                height: 1,
                expect_format: PixelFormat::Gray8,
                expect_size: 2,
            },
            Case {
                src: PixelFormat::Rgba16,
                bytes: vec![
                    10, 10, 10, 10, 10, 10, 0xFF, 0xFF, // gray opaque (bit-rep)
                    20, 20, 20, 20, 20, 20, 0xFF, 0xFF,
                ],
                width: 2,
                height: 1,
                expect_format: PixelFormat::Gray8,
                expect_size: 2,
            },
        ];
        for c in cases {
            let s = make_slice(&c.bytes, c.width, c.height, c.src);
            let out = s
                .try_reduce_to_load_bearing_format()
                .unwrap_or_else(|| panic!("{:?} should reduce", c.src));
            assert_eq!(
                out.descriptor().format,
                c.expect_format,
                "format from {:?}",
                c.src
            );
            assert_eq!(
                out.as_slice().row(0).len(),
                c.expect_size,
                "row size from {:?}",
                c.src
            );
        }
    }
}
