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
//!   * [`PixelSliceLoadBearingExt::determine_load_bearing_reduced_descriptor`]
//!     -- analysis + descriptor combiner
//!   * [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
//!     -- analysis + buffer rewrite, `None` when no narrowing is possible

use alloc::vec::Vec;

use linear_srgb::tf::{linear_to_srgb, srgb_to_linear};
use zenpixels::{
    ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, PixelFormat, PixelSlice,
    TransferFunction,
};

use crate::gamut::{
    DEFAULT_GAMUT_EPSILON, GamutFit, check_fits_in_gamut_linear_f32_rgb,
    check_fits_in_gamut_linear_f32_rgba,
};
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

    /// `Some(narrower)` → every linear-light pixel fits inside that
    /// narrower primary set after the matrix. `None` → primaries are
    /// load-bearing OR predicate didn't run (non-Linear non-sRGB
    /// transfer, F16, etc.). When `Some`, the rewrite path in
    /// [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
    /// applies the matrix and re-encodes at the narrower primaries.
    pub uses_gamut: Option<ColorPrimaries>,
}

impl LoadBearingReport {
    /// Convenience: true if alpha can be losslessly dropped from the
    /// descriptor (`uses_alpha == Some(false)`). Equivalent to the
    /// `matches!` check; provided for readability at codec sites.
    #[inline]
    pub const fn alpha_droppable(&self) -> bool {
        matches!(self.uses_alpha, Some(false))
    }

    /// Convenience: true if chroma can be losslessly dropped (the
    /// buffer can be re-encoded as grayscale without loss).
    #[inline]
    pub const fn chroma_droppable(&self) -> bool {
        matches!(self.uses_chroma, Some(false))
    }

    /// Convenience: true if `ChannelType::U16` can be losslessly
    /// narrowed to `ChannelType::U8` via bit-replication.
    #[inline]
    pub const fn low_bits_droppable(&self) -> bool {
        matches!(self.uses_low_bits, Some(false))
    }

    /// Convenience: true if the analysis returned at least one
    /// non-`None` field -- i.e. some predicate ran successfully.
    /// Codecs that need a quick "is there anything actionable here"
    /// check before consulting individual fields.
    #[inline]
    pub const fn any_analyzed(&self) -> bool {
        self.uses_alpha.is_some()
            || self.uses_chroma.is_some()
            || self.uses_low_bits.is_some()
            || self.alpha_is_binary.is_some()
            || self.uses_gray_bit_depth.is_some()
            || self.uses_gamut.is_some()
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
    ///   4. Primaries narrowing (when `uses_gamut` is `Some`)
    ///
    /// Sub-byte gray (`uses_gray_bit_depth`) is **not** applied here --
    /// `zenpixels` doesn't model sub-byte channel types. Codec encoders
    /// that support sub-byte (e.g. PNG indexed/grayscale) read the
    /// field directly off the report and apply their own bit-packing.
    ///
    /// If a step would yield an unrepresentable `(channel_type, layout,
    /// alpha)` triple (e.g. dropping alpha from `Bgra8` -- there's no
    /// `Bgr8` layout), the layout is silently kept. Bgra-with-no-alpha
    /// callers should note this and reorder channels themselves if
    /// they want narrower output.
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

        // 2. Alpha drop.
        if matches!(self.uses_alpha, Some(false)) {
            layout = match layout {
                ChannelLayout::Rgba => ChannelLayout::Rgb,
                ChannelLayout::GrayAlpha => ChannelLayout::Gray,
                // Bgra has no clean Bgr counterpart in this enum; keep
                // alpha and let codecs reorder if they want narrower.
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

        // 4. Primaries narrowing.
        let primaries = self.uses_gamut.unwrap_or(src.primaries);

        PixelDescriptor::from_pixel_format(format)
            .with_transfer(src.transfer)
            .with_primaries(primaries)
            .with_alpha(alpha)
            .with_signal_range(src.signal_range)
    }
}

// ── Extension trait on PixelSlice ──────────────────────────────────────

/// Run all relevant load-bearing predicates against a [`PixelSlice`] and
/// (optionally) produce a narrower buffer.
pub trait PixelSliceLoadBearingExt {
    /// Run all relevant predicates and return the report. Pure analysis
    /// -- no buffer rewrite, no descriptor changes.
    fn determine_load_bearing(&self) -> LoadBearingReport;

    /// Run analysis and return the descriptor the buffer would have
    /// after every justified reduction.
    fn determine_load_bearing_reduced_descriptor(&self) -> PixelDescriptor;

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

        // ── Per-pixel byte-level predicates ──────────────────────
        // Each branch returns `Some(value)` when the predicate ran
        // (or the answer is structurally trivial -- e.g. `uses_alpha
        // == Some(false)` for a layout with no alpha channel) and
        // `None` when the predicate isn't wired for this channel
        // type. Codecs treat `Some(false)` as the actionable
        // "drop this" signal.
        let (uses_alpha, uses_chroma, alpha_is_binary) = match (layout, channel_type) {
            (ChannelLayout::Rgba | ChannelLayout::Bgra, ChannelType::U8) => {
                let fused = fused_rgba8_over_rows(self);
                (
                    Some(!fused.is_opaque),
                    Some(!fused.is_grayscale),
                    Some(fused.is_binary_alpha),
                )
            }
            (ChannelLayout::Rgba, ChannelType::U16) => (
                Some(!rows_all_u16(self, scan::is_opaque_rgba16)),
                Some(!rows_all_u16(self, scan::is_grayscale_rgba16)),
                Some(rows_all_u16(self, scan::alpha_is_binary_rgba16)),
            ),
            (ChannelLayout::Rgb, ChannelType::U8) => (
                Some(false), // no alpha channel -- structurally not load-bearing
                Some(!rows_all_u8(self, scan::is_grayscale_rgb8)),
                None, // no alpha channel -- alpha-binary doesn't apply
            ),
            (ChannelLayout::Rgb, ChannelType::U16) => (
                Some(false),
                Some(!rows_all_u16(self, scan::is_grayscale_rgb16)),
                None,
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U8) => (
                Some(!rows_all_u8(self, scan::is_opaque_ga8)),
                Some(false), // already grayscale -- no chroma to be load-bearing
                Some(rows_all_u8(self, scan::alpha_is_binary_ga8)),
            ),
            (ChannelLayout::GrayAlpha, ChannelType::U16) => (
                Some(!rows_all_u16(self, scan::is_opaque_ga16)),
                Some(false),
                Some(rows_all_u16(self, scan::alpha_is_binary_ga16)),
            ),

            // Gray-anything: structurally no alpha and no chroma to
            // test. Both fields are `Some(false)` regardless of the
            // channel-type-specific predicate availability.
            (ChannelLayout::Gray, _) => (Some(false), Some(false), None),

            // F32 RGB(A) / GrayAlpha -- predicates wired.
            (ChannelLayout::Rgba, ChannelType::F32) => (
                Some(!rows_all_f32(self, scan::is_opaque_rgba_f32)),
                Some(!rows_all_f32(self, scan::is_grayscale_rgba_f32)),
                Some(rows_all_f32(self, scan::alpha_is_binary_rgba_f32)),
            ),
            (ChannelLayout::Rgb, ChannelType::F32) => (
                Some(false),
                Some(!rows_all_f32(self, scan::is_grayscale_rgb_f32)),
                None,
            ),
            (ChannelLayout::GrayAlpha, ChannelType::F32) => (
                Some(!rows_all_f32(self, scan::is_opaque_ga_f32)),
                Some(false),
                Some(rows_all_f32(self, scan::alpha_is_binary_ga_f32)),
            ),

            // F16 / Oklab / CMYK with non-Gray layout -- predicates
            // not yet wired. All fields stay `None`.
            _ => (None, None, None),
        };

        // ── Low bits (U16 → U8) ──────────────────────────────────
        let uses_low_bits = match channel_type {
            ChannelType::U16 => Some(!rows_all_u16(self, scan::bit_replication_lossless_u16)),
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

        // ── Gamut narrowing ─────────────────────────────────────
        let uses_gamut = detect_gamut_narrowing_over_rows(self, &descriptor);

        LoadBearingReport {
            uses_alpha,
            uses_chroma,
            uses_low_bits,
            alpha_is_binary,
            uses_gray_bit_depth,
            uses_gamut,
        }
    }

    fn determine_load_bearing_reduced_descriptor(&self) -> PixelDescriptor {
        let report = self.determine_load_bearing();
        report.apply_to(&self.descriptor())
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

#[inline]
fn rows_all_u8<P, F>(slice: &PixelSlice<'_, P>, predicate: F) -> bool
where
    F: Fn(&[u8]) -> bool,
{
    if let Some(bytes) = slice.as_contiguous_bytes() {
        predicate(bytes)
    } else {
        for y in 0..slice.rows() {
            if !predicate(slice.row(y)) {
                return false;
            }
        }
        true
    }
}

#[inline]
fn rows_all_u16<P, F>(slice: &PixelSlice<'_, P>, predicate: F) -> bool
where
    F: Fn(&[u16]) -> bool,
{
    if let Some(bytes) = slice.as_contiguous_bytes() {
        predicate(cast_u16(bytes))
    } else {
        for y in 0..slice.rows() {
            if !predicate(cast_u16(slice.row(y))) {
                return false;
            }
        }
        true
    }
}

#[inline]
fn rows_all_f32<P, F>(slice: &PixelSlice<'_, P>, predicate: F) -> bool
where
    F: Fn(&[f32]) -> bool,
{
    if let Some(bytes) = slice.as_contiguous_bytes() {
        predicate(cast_f32(bytes))
    } else {
        for y in 0..slice.rows() {
            if !predicate(cast_f32(slice.row(y))) {
                return false;
            }
        }
        true
    }
}

/// Row-aware fused predicate for RGBA8/Bgra8. Drops dropped checks
/// from the next row's request so per-row work shrinks as flags flip.
/// Single fused call on contiguous buffers.
fn fused_rgba8_over_rows<P>(slice: &PixelSlice<'_, P>) -> scan::FusedResult {
    if let Some(bytes) = slice.as_contiguous_bytes() {
        return scan::fused_predicates_rgba8_cg(bytes, FusedRequest::all());
    }
    let mut req = FusedRequest::all();
    let mut total = scan::FusedResult {
        is_opaque: true,
        is_grayscale: true,
        is_binary_alpha: true,
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

/// Row-aware gamut narrowing detection. Two supported source paths:
///
///   1. U8 + sRGB transfer → linearize per row → bounds check
///   2. F32 + Linear transfer → call the bounds check directly (data
///      is already linear-light)
///
/// Other transfers / depths bail (`None`), keeping the source primaries
/// load-bearing. Adding more transfers (PQ, HLG, BT.709) is a clean
/// follow-up -- dispatch the right EOTF in the linearization step.
fn detect_gamut_narrowing_over_rows<P>(
    slice: &PixelSlice<'_, P>,
    descriptor: &PixelDescriptor,
) -> Option<ColorPrimaries> {
    let target = ColorPrimaries::Bt709;
    if descriptor.primaries == target {
        return None;
    }
    let m = descriptor.primaries.gamut_matrix_to(target)?;
    let layout = descriptor.layout();
    let supports_rgb_layout = matches!(
        layout,
        ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
    );
    if !supports_rgb_layout {
        return None;
    }

    let channel_type = descriptor.channel_type();
    let transfer = descriptor.transfer;

    // F32 + Linear: zero-decode path -- pass the raw f32 data straight
    // to the bounds-check helper.
    if channel_type == ChannelType::F32 && transfer == TransferFunction::Linear {
        let check_row_f32 = |bytes: &[u8]| -> bool {
            let f32_row = cast_f32(bytes);
            let fit = match layout {
                ChannelLayout::Rgb => {
                    check_fits_in_gamut_linear_f32_rgb(f32_row, &m, DEFAULT_GAMUT_EPSILON)
                }
                ChannelLayout::Rgba | ChannelLayout::Bgra => {
                    check_fits_in_gamut_linear_f32_rgba(f32_row, &m, DEFAULT_GAMUT_EPSILON)
                }
                _ => return false,
            };
            matches!(fit, GamutFit::AllInside)
        };
        if let Some(bytes) = slice.as_contiguous_bytes() {
            if !check_row_f32(bytes) {
                return None;
            }
        } else {
            for y in 0..slice.rows() {
                if !check_row_f32(slice.row(y)) {
                    return None;
                }
            }
        }
        return Some(target);
    }

    // U8 + sRGB transfer: linearize per row, then bounds-check.
    if channel_type != ChannelType::U8 || transfer != TransferFunction::Srgb {
        return None;
    }

    let check_row_u8 = |bytes: &[u8]| -> bool {
        let linear = linearize_row(bytes, layout);
        let fit = match layout {
            ChannelLayout::Rgb => {
                check_fits_in_gamut_linear_f32_rgb(&linear, &m, DEFAULT_GAMUT_EPSILON)
            }
            ChannelLayout::Rgba | ChannelLayout::Bgra => {
                check_fits_in_gamut_linear_f32_rgba(&linear, &m, DEFAULT_GAMUT_EPSILON)
            }
            _ => return false,
        };
        matches!(fit, GamutFit::AllInside)
    };

    if let Some(bytes) = slice.as_contiguous_bytes() {
        if !check_row_u8(bytes) {
            return None;
        }
    } else {
        for y in 0..slice.rows() {
            if !check_row_u8(slice.row(y)) {
                return None;
            }
        }
    }
    Some(target)
}

/// Linearize a u8 sRGB-encoded row into a Vec<f32>. Alpha (when
/// present) is converted as plain unit-range, not gamma.
fn linearize_row(bytes: &[u8], layout: ChannelLayout) -> Vec<f32> {
    match layout {
        ChannelLayout::Rgb => bytes
            .iter()
            .map(|&b| srgb_to_linear(b as f32 / 255.0))
            .collect(),
        ChannelLayout::Rgba | ChannelLayout::Bgra => {
            let mut v = Vec::with_capacity(bytes.len());
            for chunk in bytes.chunks_exact(4) {
                v.push(srgb_to_linear(chunk[0] as f32 / 255.0));
                v.push(srgb_to_linear(chunk[1] as f32 / 255.0));
                v.push(srgb_to_linear(chunk[2] as f32 / 255.0));
                v.push(chunk[3] as f32 / 255.0);
            }
            v
        }
        _ => Vec::new(),
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

fn cast_u16(bytes: &[u8]) -> &[u16] {
    bytemuck::cast_slice(bytes)
}

fn cast_f32(bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(bytes)
}

/// Apply the gamut matrix in-place to a linear-light RGB or RGBA f32
/// buffer.
fn apply_matrix_inplace(linear: &mut [f32], m: &[[f32; 3]; 3], layout: ChannelLayout) {
    let stride = match layout {
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba | ChannelLayout::Bgra => 4,
        _ => return,
    };
    for px in linear.chunks_exact_mut(stride) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        px[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        px[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        px[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        // alpha (px[3] for RGBA/Bgra) untouched.
    }
}

/// Encode a transformed linear-light f32 buffer back to u8 sRGB bytes.
fn encode_linear_to_srgb_u8(linear: &[f32], layout: ChannelLayout) -> Vec<u8> {
    let stride = match layout {
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba | ChannelLayout::Bgra => 4,
        _ => 1,
    };
    let mut out = Vec::with_capacity(linear.len());
    for chunk in linear.chunks_exact(stride) {
        // RGB channels: linear → sRGB → u8 byte.
        for &c in &chunk[..3.min(stride)] {
            out.push(byte_from_unit_f32(linear_to_srgb(c.clamp(0.0, 1.0))));
        }
        if stride == 4 {
            // Alpha: not gamma-encoded; clamp + scale.
            out.push(byte_from_unit_f32(chunk[3].clamp(0.0, 1.0)));
        }
    }
    out
}

#[inline]
fn byte_from_unit_f32(v: f32) -> u8 {
    (v * 255.0 + 0.5) as u8
}

/// Build the rewritten buffer for the supported descriptor transitions.
/// Returns `None` for descriptor pairs we don't know how to convert.
fn transform_to(bytes: &[u8], src: &PixelDescriptor, dst: &PixelDescriptor) -> Option<Vec<u8>> {
    // Step 0: gamut narrowing first when requested. A successful step
    // rewrites pixel values in-place but keeps layout / channel-type;
    // the subsequent steps operate on the rewritten buffer.
    let gamut_owned;
    let mut working_bytes: &[u8] = bytes;
    let mut working_descriptor = *src;
    if src.primaries != dst.primaries {
        let m = src.primaries.gamut_matrix_to(dst.primaries)?;
        let layout = src.layout();
        let ct = src.channel_type();
        let transfer = src.transfer;

        // F32 + Linear: matrix on the raw f32 data, no transfer
        // function involved.
        if ct == ChannelType::F32 && transfer == TransferFunction::Linear {
            let mut linear: Vec<f32> = cast_f32(bytes).to_vec();
            apply_matrix_inplace(&mut linear, &m, layout);
            gamut_owned = bytemuck::cast_slice::<f32, u8>(&linear).to_vec();
            working_bytes = &gamut_owned;
            working_descriptor = src.with_primaries(dst.primaries);
        } else if ct == ChannelType::U8 && transfer == TransferFunction::Srgb {
            let mut linear: Vec<f32> = match layout {
                ChannelLayout::Rgb => bytes
                    .iter()
                    .map(|&b| srgb_to_linear(b as f32 / 255.0))
                    .collect(),
                ChannelLayout::Rgba | ChannelLayout::Bgra => {
                    let mut v = Vec::with_capacity(bytes.len());
                    for chunk in bytes.chunks_exact(4) {
                        v.push(srgb_to_linear(chunk[0] as f32 / 255.0));
                        v.push(srgb_to_linear(chunk[1] as f32 / 255.0));
                        v.push(srgb_to_linear(chunk[2] as f32 / 255.0));
                        v.push(chunk[3] as f32 / 255.0);
                    }
                    v
                }
                _ => return None,
            };
            apply_matrix_inplace(&mut linear, &m, layout);
            gamut_owned = encode_linear_to_srgb_u8(&linear, layout);
            working_bytes = &gamut_owned;
            working_descriptor = src.with_primaries(dst.primaries);
        } else {
            return None; // other transfer / depth combos not yet wired
        }
    }

    let src_ct = working_descriptor.channel_type();
    let dst_ct = dst.channel_type();
    let src_layout = working_descriptor.layout();
    let dst_layout = dst.layout();

    // Step 1: channel-type narrowing.
    //   U16 → U8 (bit-replicated): take the high byte of each pair.
    //   F32 → F32, U8 → U8, U16 → U16: pass through.
    //   Anything else: not yet supported.
    let post_ct: Vec<u8> = if src_ct == ChannelType::U16 && dst_ct == ChannelType::U8 {
        working_bytes.chunks_exact(2).map(|p| p[0]).collect()
    } else if src_ct == dst_ct {
        working_bytes.to_vec()
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
        (ChannelLayout::Rgba, ChannelLayout::Rgb) | (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            let mut out = Vec::with_capacity(n / 4 * 3 * elem);
            for px in post_ct.chunks_exact(in_pixel) {
                copy_channels(px, &mut out, &[0, 1, 2]);
            }
            out
        }
        (ChannelLayout::Rgba, ChannelLayout::GrayAlpha)
        | (ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => {
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
    use zenpixels::{Cicp, ColorPrimaries, PixelSlice, TransferFunction};

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
        assert_eq!(r.uses_gamut, None);
        assert!(!r.any_analyzed());
        assert!(!r.alpha_droppable());
        assert!(!r.chroma_droppable());
        assert!(!r.low_bits_droppable());
    }

    #[test]
    fn any_analyzed_fires_when_at_least_one_field_set() {
        let mut r = LoadBearingReport::default();
        assert!(!r.any_analyzed());
        r.uses_alpha = Some(true);
        assert!(r.any_analyzed(), "any_analyzed fires for any Some");
        r.uses_alpha = None;
        r.uses_gamut = Some(ColorPrimaries::Bt709);
        assert!(r.any_analyzed(), "any_analyzed fires on uses_gamut too");
    }

    // ── Gamut detection + transformation ────────────────────────

    #[test]
    fn p3_with_neutral_gray_detects_srgb_fit() {
        // (128,128,128) in P3 with sRGB transfer fits sRGB primaries
        // (white points coincide so gray maps to gray).
        let bytes: Vec<u8> = (0..4).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let slice =
            make_slice_with_primaries(&bytes, 4, 1, PixelFormat::Rgba8, ColorPrimaries::DisplayP3);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, Some(ColorPrimaries::Bt709));
    }

    #[test]
    fn p3_with_saturated_red_keeps_p3() {
        // P3 (255, 0, 0) is brighter than sRGB max red -- out of gamut.
        let bytes = [255u8, 0, 0, 255];
        let slice =
            make_slice_with_primaries(&bytes, 1, 1, PixelFormat::Rgba8, ColorPrimaries::DisplayP3);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, None, "saturated P3 red is load-bearing");
    }

    #[test]
    fn p3_buffer_with_neutral_gray_transforms_to_srgb() {
        // Detection succeeds → try_reduce should produce a buffer
        // tagged sRGB primaries.
        let bytes: Vec<u8> = (0..4).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let slice =
            make_slice_with_primaries(&bytes, 4, 1, PixelFormat::Rgba8, ColorPrimaries::DisplayP3);
        let (target, out) = slice
            .try_reduce_to_load_bearing_format()
            .expect("should reduce");
        // Reduces all the way: P3+RGBA → sRGB+Gray8 (gray + opaque).
        assert_eq!(target.primaries, ColorPrimaries::Bt709);
        assert_eq!(target.format, PixelFormat::Gray8);
        // Gray content stays mid-gray after the chromatically-aligned
        // P3→sRGB matrix.
        for &b in &out {
            assert!((127..=129).contains(&b), "drift after gamut transform: {b}");
        }
    }

    #[test]
    fn srgb_buffer_unaffected_by_gamut_check() {
        let bytes = [128u8, 128, 128, 255];
        let slice =
            make_slice_with_primaries(&bytes, 1, 1, PixelFormat::Rgba8, ColorPrimaries::Bt709);
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, None, "already at sRGB");
    }

    #[test]
    fn pq_transfer_skips_gamut_detection() {
        // Non-sRGB transfer: detection bails (v0 doesn't dispatch
        // the matching EOTF yet).
        let bytes = [128u8, 128, 128, 255];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Pq)
            .with_primaries(ColorPrimaries::Bt2020);
        let slice = PixelSlice::new(&bytes, 1, 1, 4, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, None, "PQ unsupported for v0 gamut detection");
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
        let target = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::Gray8);
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
        let target = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::Gray8);
    }

    // Suppress unused-import warning if Cicp isn't otherwise referenced
    // (forward-compat, in case the user runs only a subset of tests).
    #[allow(dead_code)]
    fn _cicp_reference() -> Cicp {
        Cicp::SRGB
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
        assert_eq!(r_contig.uses_gamut, r_strided.uses_gamut);
    }

    #[test]
    fn strided_p3_with_neutral_gray_detects_srgb_fit() {
        // P3-tagged grayscale buffer, strided. Gamut detection must
        // run per-row.
        let (buf, stride) = build_strided_rgba8(4, 3, 12, |_, _| [128, 128, 128, 255]);
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_primaries(ColorPrimaries::DisplayP3);
        let slice = PixelSlice::new(&buf, 4, 3, stride, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_gamut,
            Some(ColorPrimaries::Bt709),
            "strided P3 gray must still detect sRGB fit"
        );
    }

    #[test]
    fn strided_p3_with_one_oog_row_keeps_p3() {
        // Row 1 has a saturated red pixel; rows 0 and 2 are gray.
        // Per-row gamut check must early-exit at row 1.
        let (buf, stride) = build_strided_rgba8(4, 3, 12, |x, y| {
            if y == 1 && x == 2 {
                [255, 0, 0, 255]
            } else {
                [100, 100, 100, 255]
            }
        });
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_primaries(ColorPrimaries::DisplayP3);
        let slice = PixelSlice::new(&buf, 4, 3, stride, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, None, "OOG pixel must keep P3 tagging");
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
    fn p3_linear_f32_with_neutral_gray_detects_srgb_fit() {
        // F32 + Linear: gamut detection runs without any TRC decode.
        let pixels: [f32; 16] = [
            0.5, 0.5, 0.5, 1.0, //
            0.25, 0.25, 0.25, 1.0, //
            0.75, 0.75, 0.75, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::RgbaF32)
            .with_transfer(TransferFunction::Linear)
            .with_primaries(ColorPrimaries::DisplayP3);
        let slice = PixelSlice::new(bytes, 4, 1, 4 * 16, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(
            r.uses_gamut,
            Some(ColorPrimaries::Bt709),
            "linear-light f32 gray must fit sRGB"
        );
    }

    #[test]
    fn p3_linear_f32_saturated_red_keeps_p3() {
        // (1.0, 0, 0) in P3 linear → out of sRGB gamut.
        let pixels: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::RgbaF32)
            .with_transfer(TransferFunction::Linear)
            .with_primaries(ColorPrimaries::DisplayP3);
        let slice = PixelSlice::new(bytes, 1, 1, 16, descriptor).unwrap();
        let r = slice.determine_load_bearing();
        assert_eq!(r.uses_gamut, None);
    }

    #[test]
    fn p3_linear_f32_neutral_gray_transforms_to_srgb() {
        // Detection succeeds → transform produces sRGB-tagged f32.
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
        // P3 linear gray + opaque → sRGB linear gray (chromatically
        // aligned matrix preserves gray) + drops to GrayF32.
        assert_eq!(target.primaries, ColorPrimaries::Bt709);
        assert_eq!(target.format, PixelFormat::GrayF32);
        let gray: &[f32] = bytemuck::cast_slice(&out);
        assert_eq!(gray.len(), 4);
        // Each output value should be very close to the input (gray
        // doesn't shift across white-point-aligned matrices).
        for (got, exp) in gray.iter().zip(&[0.5_f32, 0.25, 0.75, 0.1]) {
            assert!(
                (got - exp).abs() < 1e-3,
                "gray drift: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn ga_f32_opaque_reduces_to_gray_f32() {
        let pixels: [f32; 6] = [0.1, 1.0, 0.5, 1.0, 0.9, 1.0];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 3, 1, PixelFormat::GrayAF32, TransferFunction::Linear);
        let target = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::GrayF32);
    }

    #[test]
    fn rgb_f32_grayscale_reduces_to_gray_f32() {
        let pixels: [f32; 9] = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        let slice = make_f32_slice(bytes, 3, 1, PixelFormat::RgbF32, TransferFunction::Linear);
        let target = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::GrayF32);
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
    // match determine_load_bearing_reduced_descriptor. Running them
    // independently must produce the same target.

    #[test]
    fn try_reduce_descriptor_matches_determine_reduced() {
        let bytes: Vec<u8> = (0..8).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 8, 1, PixelFormat::Rgba8);
        let determined = slice.determine_load_bearing_reduced_descriptor();
        let (reduced_target, _) = slice.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(determined, reduced_target);
    }

    #[test]
    fn try_reduce_returns_none_when_descriptor_unchanged() {
        // Gray8 with 8-bit-needing values -- nothing to reduce.
        let bytes = [50u8, 100, 150, 200];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        assert!(slice.try_reduce_to_load_bearing_format().is_none());
        // determine_reduced_descriptor returns the same descriptor.
        let determined = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(determined, slice.descriptor());
    }

    // ── Edge cases: 1×1 / single-row / single-column inputs ───

    #[test]
    fn single_pixel_inputs_for_each_layout() {
        // 1×1 Rgba8: opaque + gray → reduces to Gray8.
        let s = make_slice(&[100u8, 100, 100, 255], 1, 1, PixelFormat::Rgba8);
        assert_eq!(
            s.determine_load_bearing_reduced_descriptor().format,
            PixelFormat::Gray8
        );

        // 1×1 Rgb8 with R=G=B → reduces to Gray8.
        let s = make_slice(&[42u8, 42, 42], 1, 1, PixelFormat::Rgb8);
        assert_eq!(
            s.determine_load_bearing_reduced_descriptor().format,
            PixelFormat::Gray8
        );

        // 1×1 GrayA8 opaque → Gray8.
        let s = make_slice(&[42u8, 255], 1, 1, PixelFormat::GrayA8);
        assert_eq!(
            s.determine_load_bearing_reduced_descriptor().format,
            PixelFormat::Gray8
        );

        // 1×1 Gray8 -- no reduction available.
        let s = make_slice(&[42u8], 1, 1, PixelFormat::Gray8);
        assert_eq!(
            s.determine_load_bearing_reduced_descriptor(),
            s.descriptor()
        );
    }

    #[test]
    fn single_row_tall_buffer() {
        // 1 row, many cols -- exercises the per-row loop with one pass.
        let bytes: Vec<u8> = (0..32).flat_map(|i| [i * 7, i * 7, i * 7, 255]).collect();
        let s = make_slice(&bytes, 32, 1, PixelFormat::Rgba8);
        assert_eq!(
            s.determine_load_bearing_reduced_descriptor().format,
            PixelFormat::Gray8
        );
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

    // ── Edge cases: gamut boundary values ────────────────────

    #[test]
    fn pure_white_fits_every_target_gamut() {
        // (255, 255, 255, 255) is white in any RGB primary set.
        // Should fit sRGB regardless of source primaries.
        for src_primaries in [
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
            ColorPrimaries::AdobeRgb,
        ] {
            let bytes = [255u8, 255, 255, 255];
            let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
                .with_transfer(TransferFunction::Srgb)
                .with_primaries(src_primaries);
            let s = PixelSlice::new(&bytes, 1, 1, 4, descriptor).unwrap();
            let r = s.determine_load_bearing();
            assert_eq!(
                r.uses_gamut,
                Some(ColorPrimaries::Bt709),
                "white must fit sRGB from {src_primaries:?}"
            );
        }
    }

    #[test]
    fn pure_black_fits_every_target_gamut() {
        // (0, 0, 0, 255) -- pure black, fits any gamut by definition.
        for src_primaries in [
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
            ColorPrimaries::AdobeRgb,
        ] {
            let bytes = [0u8, 0, 0, 255];
            let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
                .with_transfer(TransferFunction::Srgb)
                .with_primaries(src_primaries);
            let s = PixelSlice::new(&bytes, 1, 1, 4, descriptor).unwrap();
            let r = s.determine_load_bearing();
            assert_eq!(
                r.uses_gamut,
                Some(ColorPrimaries::Bt709),
                "black must fit sRGB from {src_primaries:?}"
            );
        }
    }

    #[test]
    fn srgb_source_doesnt_attempt_gamut_narrowing() {
        // sRGB → sRGB is a no-op, uses_gamut should be None.
        let bytes = [50u8, 100, 200, 255];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_primaries(ColorPrimaries::Bt709);
        let s = PixelSlice::new(&bytes, 1, 1, 4, descriptor).unwrap();
        assert_eq!(s.determine_load_bearing().uses_gamut, None);
    }

    // ── Edge cases: Bgra alpha-drop quirk ────────────────────

    #[test]
    fn bgra8_with_no_alpha_load_bearing_keeps_bgra_format() {
        // No Bgr8 enum variant -- apply_to keeps Bgra8 even though
        // alpha isn't load-bearing. Documented behavior.
        let bytes = [50u8, 100, 150, 255, 60, 110, 160, 255];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Bgra8)
            .with_transfer(TransferFunction::Srgb);
        let s = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let r = s.determine_load_bearing();
        assert_eq!(r.uses_alpha, Some(false));
        let target = r.apply_to(&s.descriptor());
        // apply_to keeps Bgra8 -- no Bgr8 to narrow into.
        assert_eq!(target.format, PixelFormat::Bgra8);
    }

    #[test]
    fn bgra8_grayscale_collapses_to_gray_alpha8() {
        // R==G==B, alpha varying -- should collapse to GrayA8 even
        // for Bgra8 source (chroma drop, alpha kept).
        let bytes = [42u8, 42, 42, 100, 99, 99, 99, 200];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Bgra8)
            .with_transfer(TransferFunction::Srgb);
        let s = PixelSlice::new(&bytes, 2, 1, 8, descriptor).unwrap();
        let target = s.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::GrayA8);
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
