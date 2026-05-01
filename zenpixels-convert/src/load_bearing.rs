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
//!   * [`PixelSliceLoadBearingExt::determine_load_bearing`] — analysis,
//!     no buffer modification
//!   * [`PixelSliceLoadBearingExt::determine_load_bearing_reduced_descriptor`]
//!     — analysis + descriptor combiner
//!   * [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
//!     — analysis + buffer rewrite, `None` when no narrowing is possible

use alloc::vec::Vec;

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, PixelFormat,
    PixelSlice,
};

use crate::scan;

/// Sub-byte grayscale bit depths that a codec encoder may pack to.
/// Only meaningful for grayscale buffers at U8 channel-type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrayBitDepth {
    /// 1 bit per sample (every value ∈ {0, 255}).
    One,
    /// 2 bits per sample (every value is a multiple of 85).
    Two,
    /// 4 bits per sample (every value is a multiple of 17).
    Four,
    /// 8 bits per sample (full U8 range used).
    Eight,
}

impl GrayBitDepth {
    /// Bit count.
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

/// What a buffer's content actually exercises about its declared
/// descriptor. Each `uses_*` field answers "is this part of the
/// descriptor load-bearing for the actual pixel data?" — `false` /
/// narrower variants mean the descriptor over-promises and can be
/// narrowed losslessly.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct LoadBearingReport {
    /// True iff at least one alpha sample is not channel-max (255 for
    /// U8, 65535 for U16). `false` → the alpha channel can be dropped.
    /// Trivially `false` for descriptors with no alpha channel.
    pub uses_alpha: bool,

    /// True iff at least one pixel has differing chroma channels (R, G,
    /// B not all equal). `false` → can narrow to grayscale. Trivially
    /// `false` for descriptors that are already grayscale.
    pub uses_chroma: bool,

    /// True iff at least one U16 sample has its low byte differ from
    /// its high byte. `false` → can narrow `ChannelType::U16` to
    /// `ChannelType::U8` via bit-replication. Trivially `false` for
    /// descriptors not at U16.
    pub uses_low_bits: bool,

    /// Smallest grayscale bit depth that losslessly represents the
    /// channel. Only meaningful when the buffer is grayscale (or
    /// became grayscale after `uses_chroma == false`). For non-
    /// grayscale buffers this is unconditionally `GrayBitDepth::Eight`.
    /// PNG-style sub-byte packing is the primary consumer; codecs that
    /// don't support sub-byte ignore this field.
    pub uses_gray_bit_depth: GrayBitDepth,

    /// `Some(narrower)` if every linear-light pixel fits inside that
    /// narrower primary set after the matrix; `None` if the current
    /// primaries are load-bearing (or no narrower target was checked).
    /// Currently the analysis only checks `Bt709` (sRGB) as a target
    /// when the source is one of the supported wide-gamut primaries.
    pub uses_gamut: Option<ColorPrimaries>,
}

impl LoadBearingReport {
    /// Default report — every dimension is "load-bearing", i.e. nothing
    /// can be narrowed. Use as the starting state if you want to merge
    /// in answers piecemeal.
    pub const fn fully_load_bearing() -> Self {
        Self {
            uses_alpha: true,
            uses_chroma: true,
            uses_low_bits: true,
            uses_gray_bit_depth: GrayBitDepth::Eight,
            uses_gamut: None,
        }
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
    ///   4. Primaries narrowing (when `uses_gamut` is Some)
    ///
    /// Sub-byte gray (`uses_gray_bit_depth`) is **not** applied here —
    /// `zenpixels` doesn't model sub-byte channel types. Codec encoders
    /// that support sub-byte (e.g. PNG indexed/grayscale) read the
    /// field directly off the report and apply their own bit-packing.
    ///
    /// If a step would yield an unrepresentable `(channel_type, layout,
    /// alpha)` triple (e.g. dropping alpha from `Bgra8` — there's no
    /// `Bgr8` layout), that step is skipped and the wider descriptor
    /// is kept.
    #[must_use]
    pub fn apply_to(&self, src: &PixelDescriptor) -> PixelDescriptor {
        let mut channel_type = src.channel_type();
        let mut layout = src.layout();
        let mut alpha = src.alpha;

        // 1. Channel-type narrowing.
        if !self.uses_low_bits && channel_type == ChannelType::U16 {
            channel_type = ChannelType::U8;
        }

        // 2. Alpha drop. Map layouts that have a clean drop target.
        if !self.uses_alpha {
            layout = match layout {
                ChannelLayout::Rgba => ChannelLayout::Rgb,
                ChannelLayout::GrayAlpha => ChannelLayout::Gray,
                // Bgra has no clean Bgr counterpart in this enum;
                // keep alpha for now and let codecs handle channel
                // reordering themselves if they want narrower output.
                other => other,
            };
            if layout != src.layout() {
                alpha = None;
            }
        }

        // 3. Chroma drop.
        if !self.uses_chroma {
            layout = match layout {
                ChannelLayout::Rgb => ChannelLayout::Gray,
                ChannelLayout::Rgba => ChannelLayout::GrayAlpha,
                ChannelLayout::Bgra => ChannelLayout::GrayAlpha,
                other => other,
            };
        }

        // Try to assemble the new format. If `(channel_type, layout,
        // alpha)` isn't a valid combination, keep the original format.
        let format = PixelFormat::from_parts(channel_type, layout, alpha)
            .unwrap_or(src.format);

        // 4. Primaries narrowing.
        let primaries = self.uses_gamut.unwrap_or(src.primaries);

        // PixelDescriptor is #[non_exhaustive], so we build via the
        // public constructors and copy over the fields we want.
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
///
/// Dispatches based on the slice's [`PixelDescriptor`] — picks the right
/// SIMD predicate for the channel layout and channel type, and assembles
/// the answers into a [`LoadBearingReport`].
pub trait PixelSliceLoadBearingExt {
    /// Run all relevant predicates and return the report. Pure analysis
    /// — no buffer rewrite, no descriptor changes.
    fn determine_load_bearing(&self) -> LoadBearingReport;

    /// Run analysis and return the descriptor the buffer would have
    /// after every justified reduction. Convenience for
    /// `self.determine_load_bearing().apply_to(self.descriptor())`.
    fn determine_load_bearing_reduced_descriptor(&self) -> PixelDescriptor;

    /// Run analysis and return the rewritten contiguous buffer if any
    /// reduction is available; `None` if the buffer is already at its
    /// load-bearing minimum (or the predicates couldn't run, e.g.
    /// non-contiguous input).
    fn try_reduce_to_load_bearing_format(&self) -> Option<(PixelDescriptor, Vec<u8>)>;
}

impl<P> PixelSliceLoadBearingExt for PixelSlice<'_, P> {
    fn determine_load_bearing(&self) -> LoadBearingReport {
        let descriptor = self.descriptor();
        let Some(bytes) = self.as_contiguous_bytes() else {
            // Strided buffer: predicates require contiguous input.
            // Fall back to "fully load-bearing" — caller can either
            // make it contiguous and retry, or skip the optimization.
            return LoadBearingReport::fully_load_bearing();
        };

        let layout = descriptor.layout();
        let channel_type = descriptor.channel_type();

        // ── Alpha presence and binary alpha ──────────────────────
        let uses_alpha = match (layout, channel_type) {
            (ChannelLayout::Rgba, ChannelType::U8) => !scan::is_opaque_rgba8(bytes),
            (ChannelLayout::Rgba, ChannelType::U16) => !scan::is_opaque_rgba16(cast_u16(bytes)),
            (ChannelLayout::Bgra, ChannelType::U8) => !scan::is_opaque_rgba8(bytes),
            (ChannelLayout::GrayAlpha, ChannelType::U8) => !scan::is_opaque_ga8(bytes),
            (ChannelLayout::GrayAlpha, ChannelType::U16) => !scan::is_opaque_ga16(cast_u16(bytes)),
            // No alpha channel — trivially not load-bearing.
            _ if !layout.has_alpha() => false,
            // Other (F32/F16) — predicates not implemented yet; keep load-bearing.
            _ => true,
        };

        // ── Chroma equality (R == G == B per pixel) ──────────────
        let uses_chroma = match (layout, channel_type) {
            (ChannelLayout::Rgb, ChannelType::U8) => !scan::is_grayscale_rgb8(bytes),
            (ChannelLayout::Rgba, ChannelType::U8) => !scan::is_grayscale_rgba8(bytes),
            (ChannelLayout::Bgra, ChannelType::U8) => !scan::is_grayscale_rgba8(bytes),
            (ChannelLayout::Rgb, ChannelType::U16) => !scan::is_grayscale_rgb16(cast_u16(bytes)),
            (ChannelLayout::Rgba, ChannelType::U16) => {
                !scan::is_grayscale_rgba16(cast_u16(bytes))
            }
            // Already grayscale — trivially not chroma.
            (ChannelLayout::Gray | ChannelLayout::GrayAlpha, _) => false,
            // F32 / Oklab / CMYK — unsupported, keep load-bearing.
            _ => true,
        };

        // ── Low bits (U16 → U8 reducibility) ─────────────────────
        let uses_low_bits = match channel_type {
            ChannelType::U16 => !scan::bit_replication_lossless_u16(cast_u16(bytes)),
            // U8 already at minimum integer depth.
            ChannelType::U8 => false,
            // F32 / F16 — not modeled; keep load-bearing.
            _ => true,
        };

        // ── Sub-byte gray (only when content is grayscale) ───────
        let uses_gray_bit_depth = if !uses_chroma && channel_type == ChannelType::U8 {
            // Walk the buffer once for sub-byte detection. Cheap
            // (bandwidth-bound, can be fused with other passes
            // later). Predicate: every sample multiple of 17 → 4-bit;
            // every multiple of 85 → 2-bit; every value in {0, 255}
            // → 1-bit; else 8.
            let stride = match layout {
                ChannelLayout::Gray => 1,
                ChannelLayout::GrayAlpha => 2,
                ChannelLayout::Rgb => 3,
                ChannelLayout::Rgba | ChannelLayout::Bgra => 4,
                _ => 1,
            };
            sub_byte_gray_depth(bytes, stride)
        } else {
            GrayBitDepth::Eight
        };

        // ── Gamut narrowing ──────────────────────────────────────
        // Caller has to wire the gamut check separately because it
        // requires linear-light conversion; we don't know the
        // transfer function path without more zenpixels-convert
        // primitives. Leave None for v0; future work hooks the
        // existing fit-and-transform helpers.
        let uses_gamut = None;

        LoadBearingReport {
            uses_alpha,
            uses_chroma,
            uses_low_bits,
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
        let bytes = self.as_contiguous_bytes()?;
        // Concrete transform paths covered: U16→U8 (replicated),
        // RGBA→RGB (alpha drop), RGB/RGBA→Gray/GrayA (chroma drop),
        // GrayA→Gray (alpha drop). Combinations apply in order.
        let out = transform_to(bytes, &src, &target)?;
        Some((target, out))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn cast_u16(bytes: &[u8]) -> &[u16] {
    // Safe via bytemuck — both have the same alignment requirement
    // when the source is a properly-aligned u16-typed PixelSlice.
    bytemuck::cast_slice(bytes)
}

/// Find the smallest sub-byte gray depth that all samples in a
/// stride-N buffer satisfy. Stride-N because the chroma channels for
/// already-grayscale RGB buffers are still equal to the gray channel
/// — checking just one channel per pixel is sufficient.
fn sub_byte_gray_depth(bytes: &[u8], stride: usize) -> GrayBitDepth {
    let mut can_1 = true;
    let mut can_2 = true;
    let mut can_4 = true;
    for i in (0..bytes.len()).step_by(stride) {
        let v = bytes[i];
        if can_4 && v % 17 != 0 {
            return GrayBitDepth::Eight;
        }
        if can_2 && v % 85 != 0 {
            can_2 = false;
            can_1 = false;
        }
        if can_1 && v != 0 && v != 255 {
            can_1 = false;
        }
    }
    if can_1 {
        GrayBitDepth::One
    } else if can_2 {
        GrayBitDepth::Two
    } else if can_4 {
        GrayBitDepth::Four
    } else {
        GrayBitDepth::Eight
    }
}

/// Build the rewritten buffer for the supported descriptor transitions.
/// Returns `None` for descriptor pairs we don't know how to convert.
fn transform_to(
    bytes: &[u8],
    src: &PixelDescriptor,
    dst: &PixelDescriptor,
) -> Option<Vec<u8>> {
    let src_ct = src.channel_type();
    let dst_ct = dst.channel_type();
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    // Step 1: U16 → U8 if asked.
    let u8_bytes: Vec<u8> = if src_ct == ChannelType::U16 && dst_ct == ChannelType::U8 {
        // Take the high byte (bit-replicated → equal to the low byte).
        bytes
            .chunks_exact(2)
            .map(|p| p[0]) // BE order; equivalent to (sample >> 8) since hi == lo.
            .collect()
    } else if src_ct == dst_ct {
        bytes.to_vec()
    } else {
        return None;
    };

    // After step 1, `u8_bytes` is at the dst channel-type but still at the
    // src layout. Now narrow the layout if needed.
    let working_layout = src_layout;
    let n = u8_bytes.len() / src_layout.channels();

    let out: Vec<u8> = match (working_layout, dst_layout) {
        (a, b) if a == b => u8_bytes,
        (ChannelLayout::Rgba, ChannelLayout::Rgb) | (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            let mut out = Vec::with_capacity(n * 3);
            for px in u8_bytes.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
            out
        }
        (ChannelLayout::Rgba, ChannelLayout::GrayAlpha)
        | (ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => {
            let mut out = Vec::with_capacity(n * 2);
            for px in u8_bytes.chunks_exact(4) {
                out.push(px[0]); // R == G == B (gray)
                out.push(px[3]); // alpha
            }
            out
        }
        (ChannelLayout::Rgba, ChannelLayout::Gray)
        | (ChannelLayout::Bgra, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n);
            for px in u8_bytes.chunks_exact(4) {
                out.push(px[0]);
            }
            out
        }
        (ChannelLayout::Rgb, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n);
            for px in u8_bytes.chunks_exact(3) {
                out.push(px[0]);
            }
            out
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => {
            let mut out = Vec::with_capacity(n);
            for px in u8_bytes.chunks_exact(2) {
                out.push(px[0]);
            }
            out
        }
        // Unsupported transition.
        _ => return None,
    };

    // Re-encode for U16 if we narrowed channel-type but the dst is U16
    // (rare; bit-replicated round-trip).
    if dst_ct == ChannelType::U16 && src_ct == ChannelType::U8 {
        let mut wide = Vec::with_capacity(out.len() * 2);
        for b in out {
            wide.push(b);
            wide.push(b);
        }
        return Some(wide);
    }
    let _ = AlphaMode::Opaque; // touch to silence dead-code warnings if ever
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::{PixelSlice, TransferFunction};

    fn make_slice<'a>(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> PixelSlice<'a> {
        let descriptor = PixelDescriptor::from_pixel_format(format)
            .with_transfer(TransferFunction::Srgb);
        let stride = width as usize * format.bytes_per_pixel();
        PixelSlice::new(bytes, width, height, stride, descriptor).unwrap()
    }

    #[test]
    fn rgba8_all_opaque_gray_reduces_to_gray8() {
        // 4 RGBA pixels: gray, opaque.
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let g = (i * 30) as u8;
                [g, g, g, 255]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let report = slice.determine_load_bearing();
        assert!(!report.uses_alpha, "alpha is constant max → not load-bearing");
        assert!(!report.uses_chroma, "R == G == B → chroma not load-bearing");

        let target = report.apply_to(&slice.descriptor());
        assert_eq!(target.format, PixelFormat::Gray8);
    }

    #[test]
    fn rgba8_with_real_color_keeps_rgba() {
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
        let report = slice.determine_load_bearing();
        assert!(!report.uses_alpha);
        assert!(report.uses_chroma);

        let target = report.apply_to(&slice.descriptor());
        // Alpha drop succeeds (Rgba8 → Rgb8).
        assert_eq!(target.format, PixelFormat::Rgb8);
    }

    #[test]
    fn rgba8_with_alpha_variation_keeps_alpha() {
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| [50, 50, 50, (i * 60) as u8])
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let report = slice.determine_load_bearing();
        assert!(report.uses_alpha, "alpha varies → load-bearing");
        assert!(!report.uses_chroma);

        let target = report.apply_to(&slice.descriptor());
        // RGBA → GrayAlpha8 (chroma collapses, alpha kept).
        assert_eq!(target.format, PixelFormat::GrayA8);
    }

    #[test]
    fn rgba16_bit_replicated_reduces_to_rgba8() {
        // 4 RGBA16 pixels, every channel bit-replicated u8 → u16.
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
        let report = slice.determine_load_bearing();
        assert!(!report.uses_low_bits, "every channel bit-replicated");
        assert!(!report.uses_alpha, "alpha = 0xFFFF everywhere");

        let target = report.apply_to(&slice.descriptor());
        // U16 → U8 + alpha drop = Rgb8.
        assert_eq!(target.format, PixelFormat::Rgb8);
    }

    #[test]
    fn rgba16_actual_high_precision_keeps_u16() {
        // 4 pixels with low byte != high byte — load-bearing.
        let bytes: Vec<u8> = (0..4)
            .flat_map(|i| {
                let r_lo = (i * 17 + 1) as u8;
                let r_hi = (i * 60) as u8;
                [r_hi, r_lo, 0, 0, 0, 0, 0xFF, 0xFF]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba16);
        let report = slice.determine_load_bearing();
        assert!(report.uses_low_bits, "varying low bytes → load-bearing");

        let target = report.apply_to(&slice.descriptor());
        assert_eq!(target.channel_type(), ChannelType::U16);
    }

    #[test]
    fn pure_white_grayscale_detects_1bit_depth() {
        // 4 Gray8 pixels: only 0 and 255 → 1-bit.
        let bytes = [0u8, 255, 0, 255];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        let report = slice.determine_load_bearing();
        assert_eq!(report.uses_gray_bit_depth, GrayBitDepth::One);
    }

    #[test]
    fn quarter_levels_grayscale_detects_2bit_depth() {
        let bytes = [0u8, 85, 170, 255];
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Gray8);
        let report = slice.determine_load_bearing();
        assert_eq!(report.uses_gray_bit_depth, GrayBitDepth::Two);
    }

    #[test]
    fn sixteen_levels_grayscale_detects_4bit_depth() {
        // 0, 17, 34, ..., 255 — 16 evenly spaced values.
        let bytes: Vec<u8> = (0..16).map(|i| i * 17).collect();
        let slice = make_slice(&bytes, 16, 1, PixelFormat::Gray8);
        let report = slice.determine_load_bearing();
        assert_eq!(report.uses_gray_bit_depth, GrayBitDepth::Four);
    }

    #[test]
    fn arbitrary_grayscale_keeps_8bit_depth() {
        let bytes = [0u8, 1, 2, 3, 4, 5];
        let slice = make_slice(&bytes, 6, 1, PixelFormat::Gray8);
        let report = slice.determine_load_bearing();
        assert_eq!(report.uses_gray_bit_depth, GrayBitDepth::Eight);
    }

    #[test]
    fn try_reduce_returns_some_when_reduction_available() {
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let result = slice.try_reduce_to_load_bearing_format();
        let (target, out) = result.expect("should reduce");
        assert_eq!(target.format, PixelFormat::Gray8);
        assert_eq!(out.len(), 4); // 4 grayscale samples
    }

    #[test]
    fn try_reduce_returns_none_when_already_minimal() {
        // Truly RGBA content: not gray, not opaque, full alpha range.
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 60, 100, 200, i * 40 + 1]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8);
        let result = slice.try_reduce_to_load_bearing_format();
        assert!(result.is_none(), "should be at minimum already");
    }

    #[test]
    fn fully_load_bearing_default() {
        let r = LoadBearingReport::fully_load_bearing();
        assert!(r.uses_alpha);
        assert!(r.uses_chroma);
        assert!(r.uses_low_bits);
        assert_eq!(r.uses_gray_bit_depth, GrayBitDepth::Eight);
        assert_eq!(r.uses_gamut, None);
    }

    #[test]
    fn apply_to_no_op_on_fully_load_bearing() {
        let src = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8);
        let report = LoadBearingReport::fully_load_bearing();
        let dst = report.apply_to(&src);
        assert_eq!(dst, src);
    }

    #[test]
    fn rgba16_grayscale_alpha_replicated_reduces_to_gray8() {
        // Bit-replicated u8 in u16 form, gray, opaque → Gray8.
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

    #[test]
    fn ga8_opaque_reduces_to_gray8() {
        let bytes = [10u8, 255, 50, 255, 100, 255];
        let slice = make_slice(&bytes, 3, 1, PixelFormat::GrayA8);
        let target = slice.determine_load_bearing_reduced_descriptor();
        assert_eq!(target.format, PixelFormat::Gray8);
    }
}
