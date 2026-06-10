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

use alloc::vec::Vec;

use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, PixelFormat, PixelSlice};

use crate::scan::{self, FusedRequest};

/// Sub-byte grayscale bit depths a codec encoder may pack to. Only
/// meaningful when the buffer has been narrowed to grayscale at U8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrayBitDepth {
    /// 1 bit per sample (every value ∈ {0, 255}).
    One,
    /// 2 bits per sample (every value is a multiple of 85).
    Two,
    /// 4 bits per sample (every value is a multiple of 17).
    Four,
}

impl GrayBitDepth {
    /// Bit count.
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
        }
    }
}

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

    /// `Some(true)` → alpha varies but stays in `{0, channel_max}`;
    /// codec can use binary-mask alpha (PNG `tRNS`, GIF transparency).
    /// `Some(false)` → alpha varies through intermediate values; full
    /// alpha channel needed. `None` → either no alpha channel, or
    /// predicate didn't run.
    pub alpha_is_binary: Option<bool>,

    /// `Some(One/Two/Four)` → grayscale buffer can be sub-byte-packed
    /// at this depth without loss. `None` → no sub-byte reduction
    /// (either not grayscale, not U8, or doesn't fit any sub-byte
    /// depth, or predicate didn't run).
    pub uses_gray_bit_depth: Option<GrayBitDepth>,
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
        self.uses_alpha.is_some()
            || self.uses_chroma.is_some()
            || self.uses_low_bits.is_some()
            || self.alpha_is_binary.is_some()
            || self.uses_gray_bit_depth.is_some()
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
    /// Sub-byte gray (`uses_gray_bit_depth`) is **not** applied here --
    /// `zenpixels` doesn't model sub-byte channel types. Codec encoders
    /// that support sub-byte (e.g. PNG indexed/grayscale) read the
    /// field directly off the report and apply their own bit-packing.
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

    /// Run analysis and return the rewritten contiguous buffer if any
    /// reduction is available; `None` if the buffer is already at its
    /// load-bearing minimum (or the predicates couldn't run).
    fn try_reduce_to_load_bearing_format(&self) -> Option<(PixelDescriptor, Vec<u8>)>;
}

impl<P> PixelSliceLoadBearingExt for PixelSlice<'_, P> {
    fn determine_load_bearing(&self) -> LoadBearingReport {
        let descriptor = self.descriptor();
        let layout = descriptor.layout();
        let channel_type = descriptor.channel_type();

        // ── Descriptor-level alpha answers ───────────────────────
        // Two `AlphaMode`s answer the alpha questions without touching
        // a single pixel:
        //   * `Undefined` (RGBX/BGRX padding): the lane bytes are
        //     meaningless -- scanning them would derive answers from
        //     garbage. Structurally droppable, binary-alpha N/A.
        //   * `Opaque`: the descriptor *contracts* every sample is
        //     channel-max. Trust it -- same answers a scan of a
        //     genuinely all-opaque buffer produces.
        // `Straight` and `Premultiplied` scan normally. (All
        // reductions here stay valid under premultiplication: alpha
        // only drops when uniformly max, where premultiplied ==
        // straight; `R==G==B` and bit-replication are value-exact
        // tests unaffected by what the values encode.)
        let alpha_structural: Option<(Option<bool>, Option<bool>)> = if layout.has_alpha() {
            match descriptor.alpha {
                Some(AlphaMode::Undefined) => Some((Some(false), None)),
                Some(AlphaMode::Opaque) => Some((Some(false), Some(true))),
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
        let (mut uses_alpha, uses_chroma, mut alpha_is_binary) = match (layout, channel_type) {
            (ChannelLayout::Rgba | ChannelLayout::Bgra, ChannelType::U8) => {
                let fused = fused_rgba8_over_rows(
                    self,
                    FusedRequest {
                        check_opaque: scan_alpha,
                        check_grayscale: true,
                        check_binary_alpha: scan_alpha,
                    },
                );
                (
                    Some(!fused.is_opaque),
                    Some(!fused.is_grayscale),
                    Some(fused.is_binary_alpha),
                )
            }
            (ChannelLayout::Rgba, ChannelType::U16) => (
                Some(scan_alpha && !rows_all(self, cast_u16, scan::is_opaque_rgba16)),
                Some(!rows_all(self, cast_u16, scan::is_grayscale_rgba16)),
                Some(scan_alpha && rows_all(self, cast_u16, scan::alpha_is_binary_rgba16)),
            ),
            (ChannelLayout::Rgb, ChannelType::U8) => (
                Some(false), // no alpha channel -- structurally not load-bearing
                Some(!rows_all(self, cast_u8, scan::is_grayscale_rgb8)),
                None, // no alpha channel -- alpha-binary doesn't apply
            ),
            (ChannelLayout::Rgb, ChannelType::U16) => (
                Some(false),
                Some(!rows_all(self, cast_u16, scan::is_grayscale_rgb16)),
                None,
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U8) => (
                Some(scan_alpha && !rows_all(self, cast_u8, scan::is_opaque_ga8)),
                Some(false), // already grayscale -- no chroma to be load-bearing
                Some(scan_alpha && rows_all(self, cast_u8, scan::alpha_is_binary_ga8)),
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U16) => (
                Some(scan_alpha && !rows_all(self, cast_u16, scan::is_opaque_ga16)),
                Some(false),
                Some(scan_alpha && rows_all(self, cast_u16, scan::alpha_is_binary_ga16)),
            ),

            // Gray-anything: structurally no alpha and no chroma to
            // test. Both fields are `Some(false)` regardless of the
            // channel-type-specific predicate availability.
            (ChannelLayout::Gray, _) => (Some(false), Some(false), None),

            // F32 RGB(A) / GrayAlpha -- predicates wired.
            (ChannelLayout::Rgba, ChannelType::F32) => (
                Some(scan_alpha && !rows_all(self, cast_f32, scan::is_opaque_rgba_f32)),
                Some(!rows_all(self, cast_f32, scan::is_grayscale_rgba_f32)),
                Some(scan_alpha && rows_all(self, cast_f32, scan::alpha_is_binary_rgba_f32)),
            ),
            (ChannelLayout::Rgb, ChannelType::F32) => (
                Some(false),
                Some(!rows_all(self, cast_f32, scan::is_grayscale_rgb_f32)),
                None,
            ),
            (ChannelLayout::GrayAlpha, ChannelType::F32) => (
                Some(scan_alpha && !rows_all(self, cast_f32, scan::is_opaque_ga_f32)),
                Some(false),
                Some(scan_alpha && rows_all(self, cast_f32, scan::alpha_is_binary_ga_f32)),
            ),

            // F16 / Oklab / CMYK with non-Gray layout -- predicates
            // not yet wired. All fields stay `None`.
            _ => (None, None, None),
        };

        // Overlay the structural alpha answers (the scan, when one ran
        // at all, was told not to compute these). `uses_alpha.is_some()`
        // limits the overlay to layout × channel-type combos whose
        // predicates are wired -- unanalyzed combos stay all-`None`.
        if let Some((structural_uses, structural_binary)) = alpha_structural
            && uses_alpha.is_some()
        {
            uses_alpha = structural_uses;
            alpha_is_binary = structural_binary;
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

        // ── Sub-byte gray ───────────────────────────────────────
        // Only meaningful when the buffer is (or becomes) grayscale
        // at U8 channel-type and the analysis ran for chroma.
        let uses_gray_bit_depth =
            if matches!(uses_chroma, Some(false)) && channel_type == ChannelType::U8 {
                sub_byte_gray_over_rows(self, layout)
            } else {
                None
            };

        LoadBearingReport {
            uses_alpha,
            uses_chroma,
            uses_low_bits,
            alpha_is_binary,
            uses_gray_bit_depth,
        }
    }

    fn try_reduce_to_load_bearing_format(&self) -> Option<(PixelDescriptor, Vec<u8>)> {
        let src = self.descriptor();
        let report = self.determine_load_bearing();
        let target = report.apply_to(&src);
        if target == src {
            return None;
        }
        // Output is tightly packed (no stride padding) -- we control
        // the new buffer's layout. Caller can always re-stride later.
        let out = transform_over_rows(self, &src, &target)?;
        Some((target, out))
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
        is_binary_alpha: req.check_binary_alpha,
    };
    for y in 0..slice.rows() {
        if !req.check_opaque && !req.check_grayscale && !req.check_binary_alpha {
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
        if req.check_binary_alpha && !r.is_binary_alpha {
            total.is_binary_alpha = false;
            req.check_binary_alpha = false;
        }
    }
    total
}

/// Row-aware sub-byte gray detection.
fn sub_byte_gray_over_rows<P>(
    slice: &PixelSlice<'_, P>,
    layout: ChannelLayout,
) -> Option<GrayBitDepth> {
    let stride = match layout {
        ChannelLayout::Gray => 1,
        ChannelLayout::GrayAlpha => 2,
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba | ChannelLayout::Bgra => 4,
        _ => 1,
    };
    let mut can_1 = true;
    let mut can_2 = true;
    let mut can_4 = true;
    let process_row =
        |bytes: &[u8], can_1: &mut bool, can_2: &mut bool, can_4: &mut bool| -> bool {
            for i in (0..bytes.len()).step_by(stride) {
                let v = bytes[i];
                if *can_4 && !v.is_multiple_of(17) {
                    return false; // signals bail
                }
                if *can_2 && !v.is_multiple_of(85) {
                    *can_2 = false;
                    *can_1 = false;
                }
                if *can_1 && v != 0 && v != 255 {
                    *can_1 = false;
                }
            }
            true
        };
    if let Some(bytes) = slice.as_contiguous_bytes() {
        if !process_row(bytes, &mut can_1, &mut can_2, &mut can_4) {
            return None;
        }
    } else {
        for y in 0..slice.rows() {
            if !process_row(slice.row(y), &mut can_1, &mut can_2, &mut can_4) {
                return None;
            }
        }
    }
    if can_1 {
        Some(GrayBitDepth::One)
    } else if can_2 {
        Some(GrayBitDepth::Two)
    } else if can_4 {
        Some(GrayBitDepth::Four)
    } else {
        None
    }
}

/// Row-aware transform: produce a tightly-packed output buffer from a
/// (possibly strided) source by transforming row-by-row and appending.
fn transform_over_rows<P>(
    slice: &PixelSlice<'_, P>,
    src: &PixelDescriptor,
    dst: &PixelDescriptor,
) -> Option<Vec<u8>> {
    if let Some(bytes) = slice.as_contiguous_bytes() {
        return transform_to(bytes, src, dst);
    }
    // Strided: process each row independently and concatenate. Each
    // row is tightly-packed in the output.
    let mut out = Vec::new();
    for y in 0..slice.rows() {
        let row_out = transform_to(slice.row(y), src, dst)?;
        out.extend_from_slice(&row_out);
    }
    Some(out)
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

/// Build the rewritten buffer for the supported descriptor transitions.
/// Returns `None` for descriptor pairs we don't know how to convert.
/// Every transition is a pure byte shuffle -- no sample value changes.
fn transform_to(bytes: &[u8], src: &PixelDescriptor, dst: &PixelDescriptor) -> Option<Vec<u8>> {
    let src_ct = src.channel_type();
    let dst_ct = dst.channel_type();
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    // Step 1: channel-type narrowing.
    //   U16 → U8 (bit-replicated): both bytes of each sample are equal
    //   (that's the precondition `uses_low_bits == Some(false)` proves),
    //   so taking byte 0 is the high byte regardless of endianness.
    //   F32 → F32, U8 → U8, U16 → U16: pass through.
    //   Anything else: not yet supported.
    let post_ct: Vec<u8> = if src_ct == ChannelType::U16 && dst_ct == ChannelType::U8 {
        bytes.chunks_exact(2).map(|p| p[0]).collect()
    } else if src_ct == dst_ct {
        bytes.to_vec()
    } else {
        return None;
    };

    // Step 2: layout narrowing -- element-size-parameterized so the
    // same code shape handles U8, U16, and F32 alike. `elem` is the
    // bytes per channel for the post-step-1 buffer.
    let elem = dst_ct.byte_size();
    let in_pixel = src_layout.channels() * elem;
    let n = post_ct.len() / src_layout.channels(); // total channel-samples

    if src_layout == dst_layout {
        return Some(post_ct);
    }

    let copy_channels = |px: &[u8], out: &mut Vec<u8>, indices: &[usize]| {
        for &c in indices {
            out.extend_from_slice(&px[c * elem..(c + 1) * elem]);
        }
    };

    let out: Vec<u8> = match (src_layout, dst_layout) {
        (ChannelLayout::Rgba, ChannelLayout::Rgb) => {
            let mut out = Vec::with_capacity(n / 4 * 3 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0, 1, 2]);
            }
            out
        }
        (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            // Bgra stores B,G,R,A -- dropping alpha into the Rgb layout
            // requires the B↔R reorder, not a prefix copy.
            let mut out = Vec::with_capacity(n / 4 * 3 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[2, 1, 0]);
            }
            out
        }
        (ChannelLayout::Rgba, ChannelLayout::GrayAlpha)
        | (ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => {
            // Channel 0 is R for Rgba and B for Bgra; either is the
            // gray value because this transition only fires when
            // R == G == B held for every pixel.
            let mut out = Vec::with_capacity(n / 4 * 2 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0, 3]); // gray + alpha
            }
            out
        }
        (ChannelLayout::Rgba, ChannelLayout::Gray) | (ChannelLayout::Bgra, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n / 4 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0]);
            }
            out
        }
        (ChannelLayout::Rgb, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n / 3 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0]);
            }
            out
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n / 2 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0]);
            }
            out
        }
        _ => return None,
    };

    Some(out)
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
        assert_eq!(
            r.alpha_is_binary,
            Some(true),
            "all-opaque qualifies as binary"
        );

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
        assert_eq!(
            r.alpha_is_binary,
            Some(true),
            "but the variation is 0/255 only"
        );
        assert_eq!(r.uses_chroma, Some(false));
    }

    #[test]
    fn rgba8_alpha_with_intermediate_reports_not_binary() {
        let bytes = [10u8, 10, 10, 128, 20, 20, 20, 64];
        let slice = make_slice(&bytes, 2, 1, PixelFormat::Rgba8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(true));
        assert_eq!(
            r.alpha_is_binary,
            Some(false),
            "128 and 64 are intermediate"
        );
    }

    #[test]
    fn rgb8_no_alpha_reports_alpha_is_binary_none() {
        let bytes = [10u8, 20, 30, 40, 50, 60];
        let slice = make_slice(&bytes, 2, 1, PixelFormat::Rgb8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.alpha_is_binary, None, "no alpha channel → None");
        assert_eq!(r.uses_alpha, Some(false));
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
        assert_eq!(r.alpha_is_binary, Some(true));
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

    #[test]
    fn pure_white_grayscale_detects_1bit_depth() {
        let bytes = [0u8, 255, 0, 255];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gray_bit_depth, Some(GrayBitDepth::One));
    }

    #[test]
    fn quarter_levels_grayscale_detects_2bit_depth() {
        let bytes = [0u8, 85, 170, 255];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gray_bit_depth, Some(GrayBitDepth::Two));
    }

    #[test]
    fn sixteen_levels_grayscale_detects_4bit_depth() {
        let bytes: Vec<u8> = (0..16).map(|i| i * 17).collect();
        let slice = make_slice(&bytes, 16, 1, PixelFormat::Gray8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gray_bit_depth, Some(GrayBitDepth::Four));
    }

    #[test]
    fn arbitrary_grayscale_keeps_8bit_depth() {
        let bytes = [0u8, 1, 2, 3, 4, 5];
        let slice = make_slice(&bytes, 6, 1, PixelFormat::Gray8);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gray_bit_depth, None, "no sub-byte reduction");
    }

    // ── try_reduce ─────────────────────────────────────────────

    #[test]
    fn try_reduce_returns_some_when_reduction_available() {
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let result = slice.try_reduce_to_load_bearing_format();
        let (target, out) = result.expect("should reduce");
        assert_eq!(target.format, PixelFormat::Gray8);
        assert_eq!(out.len(), 4);
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
        assert_eq!(r.alpha_is_binary, None);
        assert_eq!(r.uses_gray_bit_depth, None);
        assert!(!r.any_analyzed());
    }

    #[test]
    fn any_analyzed_fires_when_at_least_one_field_set() {
        let mut r = LoadBearingReport::default();
        assert!(!r.any_analyzed());
        r.uses_alpha = Some(true);
        assert!(r.any_analyzed(), "any_analyzed fires for any Some");
        r.uses_alpha = None;
        r.uses_gray_bit_depth = Some(GrayBitDepth::Two);
        assert!(r.any_analyzed(), "any_analyzed fires on gray depth too");
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
        assert_eq!(r_p3.alpha_is_binary, r_srgb.alpha_is_binary);

        let (target, out) = p3
            .try_reduce_to_load_bearing_format()
            .expect("gray+opaque should reduce");
        assert_eq!(target.format, PixelFormat::Gray8);
        assert_eq!(
            target.primaries,
            ColorPrimaries::DisplayP3,
            "primaries tag must carry over untouched"
        );
        // Bit-exact: the gray bytes are the original channel values.
        assert_eq!(out, &[0u8, 30, 60, 90]);
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
        assert_eq!(r.alpha_is_binary, None, "no real alpha channel to classify");
        assert_eq!(r.uses_chroma, Some(true), "chroma still measured");
        // try_reduce drops the padding lane.
        let (target, out) = slice
            .try_reduce_to_load_bearing_format()
            .expect("padding drop is a reduction");
        assert_eq!(target.format, PixelFormat::Rgb8);
        assert_eq!(out, &[10u8, 20, 30, 40, 50, 60]);
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
        assert_eq!(
            r.alpha_is_binary,
            Some(true),
            "all-max is binary-compatible"
        );
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
        assert_eq!(r.alpha_is_binary, Some(false));
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
        assert_eq!(r.alpha_is_binary, Some(true));
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
        let (target, out) = slice
            .try_reduce_to_load_bearing_format()
            .expect("strided buffer should reduce");
        assert_eq!(target.format, PixelFormat::Gray8);
        // Tightly-packed output: 4 × 4 = 16 grayscale bytes, no stride padding.
        assert_eq!(out.len(), 16);
        for y in 0..4 {
            for x in 0..4 {
                let expected = ((x + y) * 20) as u8;
                assert_eq!(
                    out[y * 4 + x],
                    expected,
                    "tight gray byte at ({x},{y}) wrong"
                );
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
        assert_eq!(r_contig.alpha_is_binary, r_strided.alpha_is_binary);
        assert_eq!(r_contig.uses_gray_bit_depth, r_strided.uses_gray_bit_depth);
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
        assert_eq!(r.alpha_is_binary, Some(true));

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
        assert_eq!(r.alpha_is_binary, Some(false));
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
        let (target, out) = slice
            .try_reduce_to_load_bearing_format()
            .expect("should reduce");
        assert_eq!(target.format, PixelFormat::GrayF32);
        assert_eq!(out.len(), 4 * 4); // 4 f32 grayscale samples = 16 bytes
        let gray: &[f32] = bytemuck::cast_slice(&out);
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
        let (target, out) = slice
            .try_reduce_to_load_bearing_format()
            .expect("should reduce");
        assert_eq!(target.format, PixelFormat::GrayF32);
        assert_eq!(target.primaries, ColorPrimaries::DisplayP3);
        let gray: &[f32] = bytemuck::cast_slice(&out);
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

    #[test]
    fn strided_gray8_sub_byte_detection_works() {
        // Gray8 strided buffer where every value is in {0, 255}.
        // Sub-byte gray detection must iterate rows correctly.
        let width = 8u32;
        let height = 4u32;
        let stride = width as usize + 12; // 12 bytes padding per row
        let mut buf = vec![0xAAu8; stride * height as usize];
        for y in 0..height {
            for x in 0..width {
                let v = if (x + y) & 1 == 0 { 0u8 } else { 255 };
                buf[y as usize * stride + x as usize] = v;
            }
        }
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Gray8)
            .with_transfer(TransferFunction::Srgb);
        let slice = PixelSlice::new(&buf, width, height, stride, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_gray_bit_depth,
            Some(GrayBitDepth::One),
            "strided buffer with 0/255 only must detect 1-bit depth"
        );
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
        let (reduced_target, _) = slice.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(determined, reduced_target);
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
            assert_eq!(
                r.alpha_is_binary, None,
                "{fmt:?} alpha-binary should be None"
            );
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

        let (reduced_target, out) = s
            .try_reduce_to_load_bearing_format()
            .expect("opaque Bgra8 should reduce");
        assert_eq!(reduced_target.format, PixelFormat::Rgb8);
        assert_eq!(
            out,
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
        let (target, out) = s.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(target.format, PixelFormat::GrayA8);
        assert_eq!(out, &[42u8, 100, 99, 200]);
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
            assert_eq!(
                r.alpha_is_binary,
                Some(true),
                "vacuous: every alpha is in {{0,255}}"
            );
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
            let (target, out) = s
                .try_reduce_to_load_bearing_format()
                .unwrap_or_else(|| panic!("{:?} should reduce", c.src));
            assert_eq!(target.format, c.expect_format, "format from {:?}", c.src);
            assert_eq!(out.len(), c.expect_size, "size from {:?}", c.src);
        }
    }
}
