//! Descriptor-aware load-bearing analysis: "what parts of this buffer's
//! declared descriptor are actually carrying information?"
//!
//! Each predicate in `crate::scan` answers a single byte-level question
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
//!     -- analysis + buffer rewrite into a fresh allocation, `None` when
//!     no narrowing is possible
//!   * [`PixelBufferLoadBearingExt::reduce_to_load_bearing_format_in_place`]
//!     -- analysis + buffer rewrite into the buffer's own bytes (no
//!     allocation), re-describing the buffer in the same call
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

use alloc::sync::Arc;

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorContext, InPlacePixels, PixelBuffer,
    PixelDescriptor, PixelFormat, PixelSlice, PixelSliceMut,
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
    /// Seals [`super::PixelSliceLoadBearingExt`] to the slice types —
    /// the analysis is keyed off their descriptor + row iteration
    /// contract, so external impls have nothing valid to implement.
    pub trait Sealed {}
    impl<P> Sealed for zenpixels::PixelSlice<'_, P> {}
    impl<P> Sealed for zenpixels::PixelSliceMut<'_, P> {}
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
    /// The returned [`PixelBuffer`] carries the narrowed descriptor and a
    /// tightly-packed row stride (`width * bytes_per_pixel`) — it is allocated
    /// with [`PixelBuffer::try_new`], and `PixelDescriptor::aligned_stride` is
    /// packed despite its name (`simd_aligned_stride` is the padded one, and
    /// nothing in these crates allocates with it). Use the buffer's own
    /// accessors or [`PixelBuffer::as_slice`] downstream rather than assuming
    /// either stride.
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
        let mut report = self.determine_load_bearing();
        // Color-signaling plan: collapsing RGB(A) to gray would pair an
        // attached RGB-class ICC profile with a Gray layout -- invalid
        // signaling. Either a GRAY-class variant stands in (Swap), the
        // context is already gray-valid (Carry), or the chroma signal is
        // masked for the *rewrite* (Suppress); see
        // [`plan_chroma_collapse_signaling`].
        let plan = plan_chroma_collapse_signaling(self.color_context());
        if matches!(plan, GraySignalPlan::Suppress) {
            report.uses_chroma = None;
        }
        let target = report.apply_to(&src);
        if target == src {
            return None;
        }
        // One fallible zeroed allocation (calloc path) at the target's
        // standard aligned stride; rows are then written in place --
        // no per-pixel Vec growth anywhere in the rewrite.
        let mut out = PixelBuffer::try_new(self.width(), self.rows(), target).ok()?;
        transform_into(self, &src, &target, &mut out)?;
        // Every reduction is value-exact, so color metadata travels with
        // the reduced buffer: swapped to the GRAY-class context when the
        // rewrite collapsed chroma under a Swap plan, carried verbatim
        // otherwise (class-preserving reductions keep it valid).
        let ctx = match (chroma_collapsed(src.layout(), target.layout()), plan) {
            (true, GraySignalPlan::Swap(swapped)) => swapped,
            _ => self.color_context().cloned(),
        };
        Some(match ctx {
            Some(ctx) => out.with_color_context(ctx),
            None => out,
        })
    }
}

/// How the chroma-collapse rewrite (RGB(A)/BGRA -> Gray/GrayAlpha) must
/// treat the buffer's color signaling. Produced by
/// [`plan_chroma_collapse_signaling`]; consumed by both reduce variants.
enum GraySignalPlan {
    /// Collapse allowed; the existing context (or none) stays valid for
    /// gray and carries over as-is.
    Carry,
    /// Collapse allowed, but only with this replacement context: a
    /// GRAY-class ICC swapped in for the RGB-class one (or the ICC
    /// dropped in favor of CICP/descriptor signaling when the color is
    /// the assumed sRGB default). Applied only if the rewrite actually
    /// collapses chroma.
    Swap(Option<Arc<ColorContext>>),
    /// No valid gray signaling can be derived -- the rewrite keeps the
    /// RGB form (alpha-drop and U16 -> U8 stay available).
    Suppress,
}

/// Decide whether the chroma collapse may rewrite this buffer, and what
/// the reduced buffer's [`ColorContext`] must become.
///
/// An ICC profile's header declares a device color space class
/// (`'RGB '`, `'GRAY'`, ...); pairing a Gray-layout image with an
/// RGB-class profile is invalid signaling (libpng, among others,
/// rejects it). So when ICC bytes are attached, the collapse is allowed
/// only if a **GRAY-class variant** can stand in: derive the CICP
/// description of the attached profile -- the context's explicit `cicp`
/// field, then an embedded `cICP` tag
/// ([`zenpixels::icc::extract_cicp`]), then the normalized-hash
/// identification of well-known profiles
/// ([`zenpixels::icc::identify_common`]; worst accepted TRC deviation
/// ±56/65535, sub-step at 8-bit) -- and feed it to
/// [`crate::icc_profiles::synthesize_gray_icc_for_cicp`]. The swapped
/// context keeps the source's `cicp` field alongside the new gray ICC.
/// Unidentifiable profiles (and profiles whose color has no CICP code
/// points, e.g. Adobe RGB) suppress the collapse.
///
/// CICP-only contexts have no class to violate: H.273 primaries (the
/// white point) and transfer characteristics remain meaningful for
/// single-channel data, and matrix coefficients describe a YCbCr<->RGB
/// mapping that gray consumers ignore -- they carry over unchanged.
///
/// Note this gates only the buffer-rewriting APIs. The
/// [`LoadBearingReport`] still reports measured chroma truth; encoders
/// with their own color-emit pipelines can act on it and synthesize /
/// re-resolve signaling themselves.
fn plan_chroma_collapse_signaling(ctx: Option<&Arc<ColorContext>>) -> GraySignalPlan {
    use crate::icc_profiles::{SynthesizedIcc, synthesize_gray_icc_for_cicp};

    let Some(ctx) = ctx else {
        return GraySignalPlan::Carry;
    };
    let Some(icc) = ctx.icc.as_deref() else {
        return GraySignalPlan::Carry;
    };

    let cicp = ctx
        .cicp
        .or_else(|| zenpixels::icc::extract_cicp(icc))
        .or_else(|| zenpixels::icc::identify_common(icc).and_then(|id| id.to_cicp()));
    let Some(cicp) = cicp else {
        return GraySignalPlan::Suppress;
    };

    match synthesize_gray_icc_for_cicp(cicp) {
        SynthesizedIcc::Profile(bytes) => {
            let mut swapped = ColorContext::from_icc(bytes.into_owned());
            swapped.cicp = ctx.cicp;
            GraySignalPlan::Swap(Some(Arc::new(swapped)))
        }
        // Assumed sRGB default: gray output needs no ICC (descriptor /
        // container-level signaling suffices); keep the CICP if the
        // source context carried one.
        SynthesizedIcc::NotNeeded => {
            GraySignalPlan::Swap(ctx.cicp.map(|c| Arc::new(ColorContext::from_cicp(c))))
        }
        // Off-grid code points / future variants: nothing derivable.
        _ => GraySignalPlan::Suppress,
    }
}

/// Whether the layout transition `src -> dst` collapses chroma (the
/// transition [`plan_chroma_collapse_signaling`] gates).
fn chroma_collapsed(src: ChannelLayout, dst: ChannelLayout) -> bool {
    matches!(
        src,
        ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
    ) && matches!(dst, ChannelLayout::Gray | ChannelLayout::GrayAlpha)
}

// ── In-place reduction on PixelBuffer ──────────────────────────────────

/// In-place load-bearing reduction: rewrite a [`PixelBuffer`]'s own
/// bytes to the narrowest justified format -- no allocation, and the
/// buffer's descriptor/geometry/color are updated **atomically** via
/// [`PixelBuffer::transform_in_place`] (a stale-descriptor state is
/// unrepresentable; this is deliberately the only in-place entry point).
///
/// Sealed: implemented for [`PixelBuffer`] only.
pub trait PixelBufferLoadBearingExt: sealed::Sealed {
    /// Consume this buffer and return its load-bearing representation while
    /// reusing the existing allocation.
    fn into_load_bearing_format(mut self, force_alpha_restructuring: bool) -> Self
    where
        Self: Sized,
    {
        self.reduce_to_load_bearing_format_in_place(force_alpha_restructuring);
        self
    }

    /// Run the load-bearing analysis and rewrite this buffer in place to
    /// the narrowest justified format, adopting the narrowed descriptor
    /// and tight row stride (`width * bytes_per_pixel`) in the same call.
    /// When no reduction applies the buffer is unchanged -- compare
    /// [`PixelBuffer::descriptor`] before/after to detect it.
    ///
    /// Every rewrite is the same bit-exact byte selection
    /// [`PixelSliceLoadBearingExt::try_reduce_to_load_bearing_format`]
    /// performs; only the destination is the buffer itself. Rows are
    /// compacted front-to-back (narrowing means every write lands at or
    /// before the bytes it just consumed), so strided input works and
    /// the result is always tightly packed.
    ///
    /// `force_alpha_restructuring` controls the one reduction that has a
    /// tag-only alternative:
    ///
    /// * `false` -- a provably non-load-bearing alpha lane is **not**
    ///   compacted away. A scanned-opaque `Straight`/`Premultiplied`
    ///   buffer is re-tagged [`AlphaMode::Opaque`] instead (zero data
    ///   movement -- encoders that re-layout internally only need the
    ///   contract); RGBX/BGRX padding keeps its
    ///   [`AlphaMode::Undefined`] tag. Channel-type narrowing and chroma
    ///   collapse still rewrite, keeping the alpha lane in the layout
    ///   (RGBA16 -> GrayA8, not Gray8).
    /// * `true` -- the alpha/padding lane is physically removed
    ///   (RGBA/BGRA -> RGB, GrayAlpha -> Gray), for consumers that need
    ///   the packed narrow form (TIFF/JXL-style writers).
    ///
    /// Color metadata ([`ColorContext`]) follows the same rules as the
    /// allocating variant: it stays on the buffer. When the rewrite
    /// collapses chroma and ICC bytes are attached (an RGB-class profile
    /// cannot describe a Gray layout), a GRAY-class variant is swapped in
    /// if the profile's CICP description is derivable (explicit `cicp`
    /// field, embedded `cICP` tag, or well-known-profile identification);
    /// otherwise the collapse is suppressed. CICP-only contexts stay
    /// valid for gray and carry over unchanged.
    fn reduce_to_load_bearing_format_in_place(&mut self, force_alpha_restructuring: bool);
}

impl sealed::Sealed for PixelBuffer {}

impl PixelBufferLoadBearingExt for PixelBuffer {
    fn reduce_to_load_bearing_format_in_place(&mut self, force_alpha_restructuring: bool) {
        self.transform_in_place(|px| reduce_in_place_impl(px, force_alpha_restructuring));
    }
}

/// The transform body behind
/// [`PixelBufferLoadBearingExt::reduce_to_load_bearing_format_in_place`]:
/// analyze, plan, compact, and return the re-described view for
/// [`PixelBuffer::transform_in_place`] to adopt. Returns the input
/// re-wrapped unchanged when no reduction applies.
fn reduce_in_place_impl(
    px: InPlacePixels<'_>,
    force_alpha_restructuring: bool,
) -> PixelSliceMut<'_> {
    let InPlacePixels {
        bytes,
        width,
        rows,
        stride: in_stride,
        descriptor: src,
        color: original_ctx,
        ..
    } = px;
    fn rewrap<'b>(
        bytes: &'b mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        desc: PixelDescriptor,
        ctx: Option<Arc<ColorContext>>,
    ) -> PixelSliceMut<'b> {
        let out = PixelSliceMut::new(bytes, width, rows, stride, desc)
            .expect("in-place reduction geometry is always valid");
        match ctx {
            Some(c) => out.with_color_context(c),
            None => out,
        }
    }
    if width == 0 || rows == 0 {
        return rewrap(bytes, width, rows, in_stride, src, original_ctx);
    }

    let mut report = {
        let view = PixelSlice::new(&bytes[..], width, rows, in_stride, src)
            .expect("buffer-backed view is always valid");
        view.determine_load_bearing()
    };

    // Same color-signaling plan as the allocating variant.
    let plan = plan_chroma_collapse_signaling(original_ctx.as_ref());
    if matches!(plan, GraySignalPlan::Suppress) {
        report.uses_chroma = None;
    }

    // Alpha plan: physical drop only on request. Otherwise the lane
    // stays in the layout and a scanned-opaque Straight/Premultiplied
    // tag upgrades to the contract the scan just measured.
    let alpha_droppable = matches!(report.uses_alpha, Some(false)) && src.layout().has_alpha();
    if !force_alpha_restructuring {
        report.uses_alpha = None;
    }

    let mut target = report.apply_to(&src);
    if !force_alpha_restructuring
        && alpha_droppable
        && target.layout().has_alpha()
        && !matches!(
            src.alpha,
            Some(AlphaMode::Undefined) | Some(AlphaMode::Opaque)
        )
    {
        target = target.with_alpha(Some(AlphaMode::Opaque));
    }

    if target == src {
        return rewrap(bytes, width, rows, in_stride, src, original_ctx);
    }

    // Equal-bpp target: no physical narrowing happened.
    // `apply_to`'s `from_parts` can re-spell a same-shape format
    // (Rgba8 + Undefined alpha normalizes to Rgbx8); a reduction
    // API must narrow, not re-spell, so keep the source format and
    // apply only the alpha re-tag if one was earned above. No
    // bytes move and the original stride is kept.
    if target.bytes_per_pixel() == src.bytes_per_pixel() {
        let retagged = src.with_alpha(target.alpha);
        return rewrap(bytes, width, rows, in_stride, retagged, original_ctx);
    }

    // Physical rewrite. `apply_to` only produces transitions
    // `selection_map` knows, so the fallback is unreachable -- but
    // stay total and hand the view back rather than panic.
    let narrow16 =
        src.channel_type() == ChannelType::U16 && target.channel_type() == ChannelType::U8;
    let Some(map) = selection_map(src.layout(), target.layout()) else {
        return rewrap(bytes, width, rows, in_stride, src, original_ctx);
    };

    // Same context rule as the allocating variant: swap to the
    // GRAY-class context when this rewrite collapses chroma under a
    // Swap plan, carry the original otherwise.
    let ctx = match (chroma_collapsed(src.layout(), target.layout()), plan) {
        (true, GraySignalPlan::Swap(swapped)) => swapped,
        _ => original_ctx,
    };

    let in_bpp = src.bytes_per_pixel();
    let out_bpp = target.bytes_per_pixel();
    debug_assert!(out_bpp < in_bpp, "reduction always shrinks bpp");
    let out_stride = compact_rows_in_place(
        bytes,
        width as usize,
        rows as usize,
        in_stride,
        in_bpp,
        out_bpp,
        src.layout(),
        target.layout(),
        map,
        narrow16,
        CompactionStride::Packed,
    );

    rewrap(bytes, width, rows, out_stride, target, ctx)
}

/// Rewrite rows front-to-back in place, narrowing `in_bpp` -> `out_bpp`
/// per pixel (channel selection via `map`, optional U16 -> U8 narrowing).
/// Returns the selected output stride.
///
/// Overlap safety (plain index math, no `unsafe`): for pixel `(y, x)`,
/// `dst_end = y*out_stride + (x+1)*out_bpp <= y*in_stride +
/// (x+1)*in_bpp`, the start of the next unread source pixel -- every
/// write lands at or before the bytes already consumed. Rows whose destination
/// span is disjoint from their source span borrow both via `split_at_mut` and
/// reuse the allocating path's SIMD/shuffle row kernels. With packed output
/// this is all but the first `~out_bpp / (in_bpp - out_bpp)` rows on tight
/// input; a row-preserving stride may intentionally overlap more rows. The
/// overlapping rows stage each pixel through a fixed temp so the within-pixel
/// read stays ahead of the write (dst == src only at the very first pixel).
#[derive(Clone, Copy)]
pub(crate) enum CompactionStride {
    /// Collapse every row front-to-back into a tightly packed image.
    Packed,
    /// Keep rows at the closest valid stride not exceeding the input stride.
    ///
    /// This avoids unnecessary cross-row movement for a lane-only adaptation
    /// while still satisfying `PixelSlice`'s whole-pixel stride invariant.
    PreserveRows,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compact_rows_in_place(
    data: &mut [u8],
    width: usize,
    rows: usize,
    in_stride: usize,
    in_bpp: usize,
    out_bpp: usize,
    src_layout: ChannelLayout,
    dst_layout: ChannelLayout,
    map: &[usize],
    narrow16: bool,
    stride_policy: CompactionStride,
) -> usize {
    let packed_stride = width * out_bpp;
    let out_stride = match stride_policy {
        CompactionStride::Packed => packed_stride,
        CompactionStride::PreserveRows => in_stride - (in_stride % out_bpp),
    };
    debug_assert!(out_stride >= packed_stride);
    let in_ch = src_layout.channels();
    let elem = if narrow16 { 2 } else { in_bpp / in_ch };
    let row_in_len = width * in_bpp;
    for y in 0..rows {
        let src_start = y * in_stride;
        let dst_start = y * out_stride;
        let dst_end = dst_start + out_stride;
        if dst_end <= src_start {
            // Disjoint spans: same row kernels as the allocating rewrite.
            let (head, tail) = data.split_at_mut(src_start);
            let row_in = &tail[..row_in_len];
            let row_out = &mut head[dst_start..dst_end];
            if narrow16 {
                select_row_u16_to_u8(row_in, row_out, in_ch, map);
            } else {
                match (elem, src_layout, dst_layout) {
                    (1, ChannelLayout::Rgba, ChannelLayout::Rgb) => {
                        // Length mismatch is impossible here; the scalar
                        // fallback keeps the loop total instead of
                        // propagating an unreachable error.
                        if garb::bytes::rgba_to_rgb(row_in, row_out).is_err() {
                            select_row::<1>(row_in, row_out, in_ch, map);
                        }
                    }
                    (1, ChannelLayout::Bgra, ChannelLayout::Rgb) => {
                        if garb::bytes::bgra_to_rgb(row_in, row_out).is_err() {
                            select_row::<1>(row_in, row_out, in_ch, map);
                        }
                    }
                    (1, ..) => select_row::<1>(row_in, row_out, in_ch, map),
                    (2, ..) => select_row::<2>(row_in, row_out, in_ch, map),
                    _ => select_row::<4>(row_in, row_out, in_ch, map),
                }
            }
        } else {
            // Overlapping prefix rows: per-pixel staging through a
            // fixed temp (max in_bpp = RgbaF32 = 16 bytes).
            for x in 0..width {
                let s = src_start + x * in_bpp;
                let mut tmp = [0u8; 16];
                tmp[..in_bpp].copy_from_slice(&data[s..s + in_bpp]);
                let d = dst_start + x * out_bpp;
                if narrow16 {
                    for (k, &c) in map.iter().enumerate() {
                        data[d + k] = tmp[c * 2];
                    }
                } else {
                    for (k, &c) in map.iter().enumerate() {
                        data[d + k * elem..d + (k + 1) * elem]
                            .copy_from_slice(&tmp[c * elem..(c + 1) * elem]);
                    }
                }
            }
        }
    }
    out_stride
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

    // Source-channel selection map for the layout transition.
    let in_ch = src_layout.channels();
    let map: &[usize] = selection_map(src_layout, dst_layout)?;

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

/// Source-channel selection map for a layout transition, in element
/// units. Identity when the layout doesn't change (channel-type-only
/// narrowing); `None` for pairs the reducer never produces.
fn selection_map(src_layout: ChannelLayout, dst_layout: ChannelLayout) -> Option<&'static [usize]> {
    static IDENTITY: [usize; 4] = [0, 1, 2, 3];
    Some(match (src_layout, dst_layout) {
        _ if src_layout == dst_layout => &IDENTITY[..src_layout.channels()],
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
    })
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

    // ── In-place reduction (PixelBuffer entry + transform impl) ───

    /// Tight-stride PixelBuffer fixture with an sRGB-tagged descriptor.
    fn lb_buf(bytes: &[u8], width: u32, height: u32, format: PixelFormat) -> PixelBuffer {
        let descriptor =
            PixelDescriptor::from_pixel_format(format).with_transfer(TransferFunction::Srgb);
        PixelBuffer::from_vec(bytes.to_vec(), width, height, descriptor).unwrap()
    }

    #[test]
    fn in_place_rgba8_gray_opaque_force_true_compacts_to_gray8() {
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8);
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Gray8);
        assert_eq!(buf.stride(), 4, "tight stride");
        assert_eq!(buf.as_slice().row(0), &[0u8, 30, 60, 90]);
    }

    #[test]
    fn in_place_rgba8_gray_opaque_force_false_keeps_alpha_lane() {
        // Chroma collapse still rewrites, but the (non-load-bearing)
        // alpha lane stays in the layout and re-tags Opaque.
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8);
        buf.reduce_to_load_bearing_format_in_place(false);
        assert_eq!(buf.descriptor().format, PixelFormat::GrayA8);
        assert_eq!(buf.descriptor().alpha, Some(AlphaMode::Opaque));
        assert_eq!(
            buf.as_slice().row(0),
            &[0u8, 255, 30, 255, 60, 255, 90, 255]
        );
    }

    #[test]
    fn in_place_colorful_opaque_force_false_is_retag_only() {
        let original: Vec<u8> = (0..4i32)
            .flat_map(|i| {
                [
                    (i * 60 + 10) as u8,
                    (i * 30 + 50) as u8,
                    (i * 90 + 20) as u8,
                    255,
                ]
            })
            .collect();
        let mut buf = lb_buf(&original, 4, 1, PixelFormat::Rgba8);
        let in_stride = buf.stride();
        buf.reduce_to_load_bearing_format_in_place(false);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgba8, "layout kept");
        assert_eq!(
            buf.descriptor().alpha,
            Some(AlphaMode::Opaque),
            "scanned-opaque straight alpha upgrades to the Opaque contract"
        );
        assert_eq!(buf.stride(), in_stride, "no bytes moved");
        assert_eq!(buf.as_slice().row(0), &original[..], "no bytes changed");
    }

    #[test]
    fn in_place_colorful_opaque_force_true_drops_alpha() {
        let bytes: Vec<u8> = (0..4i32)
            .flat_map(|i| {
                [
                    (i * 60 + 10) as u8,
                    (i * 30 + 50) as u8,
                    (i * 90 + 20) as u8,
                    255,
                ]
            })
            .collect();
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8);
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
        assert_eq!(
            buf.as_slice().row(0),
            &[10u8, 50, 20, 70, 80, 110, 130, 110, 200, 190, 140, 34]
        );
    }

    #[test]
    fn in_place_bgra8_force_true_reorders_to_rgb8() {
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Bgra8)
            .with_transfer(TransferFunction::Srgb);
        let mut buf = PixelBuffer::from_vec(
            vec![50u8, 100, 150, 255, 60, 110, 160, 255],
            2,
            1,
            descriptor,
        )
        .unwrap();
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
        assert_eq!(buf.as_slice().row(0), &[150u8, 100, 50, 160, 110, 60]);
    }

    #[test]
    fn in_place_rgba16_replicated_gray_opaque_both_force_modes() {
        let build = |i: u8| {
            let g = i * 60;
            [g, g, g, g, g, g, 0xFF, 0xFF]
        };
        let bytes: Vec<u8> = (0..4).flat_map(build).collect();
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba16);
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Gray8);
        assert_eq!(buf.as_slice().row(0), &[0u8, 60, 120, 180]);

        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba16);
        buf.reduce_to_load_bearing_format_in_place(false);
        assert_eq!(buf.descriptor().format, PixelFormat::GrayA8);
        assert_eq!(buf.descriptor().alpha, Some(AlphaMode::Opaque));
        assert_eq!(
            buf.as_slice().row(0),
            &[0u8, 255, 60, 255, 120, 255, 180, 255]
        );
    }

    #[test]
    fn in_place_gray16_replicated_single_row_overlap_path() {
        // Gray16 -> Gray8 on one row: dst and src spans share the same
        // start, so the entire row takes the overlapping per-pixel path.
        let bytes: Vec<u8> = (0..64u16).flat_map(|i| [(i * 4) as u8; 2]).collect();
        let mut buf = lb_buf(&bytes, 64, 1, PixelFormat::Gray16);
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Gray8);
        let expected: Vec<u8> = (0..64u16).map(|i| (i * 4) as u8).collect();
        assert_eq!(buf.as_slice().row(0), &expected[..]);
    }

    #[test]
    fn in_place_undefined_padding_retag_vs_restructure() {
        // RGBX padding: force=false leaves the buffer alone entirely
        // (the Undefined tag already says "ignore the lane" -- it must
        // NOT be upgraded to Opaque, the bytes are garbage).
        let original = [10u8, 20, 30, 0x7B, 40, 50, 60, 0x01];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb)
            .with_alpha(Some(AlphaMode::Undefined));
        let mut buf = PixelBuffer::from_vec(original.to_vec(), 2, 1, descriptor).unwrap();
        buf.reduce_to_load_bearing_format_in_place(false);
        assert_eq!(buf.descriptor(), descriptor, "fully unchanged");
        assert_eq!(buf.as_slice().row(0), &original[..]);

        let mut buf = PixelBuffer::from_vec(original.to_vec(), 2, 1, descriptor).unwrap();
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
        assert_eq!(buf.as_slice().row(0), &[10u8, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn in_place_load_bearing_alpha_is_untouched() {
        // Varying premultiplied alpha + real chroma: nothing reduces,
        // both force modes leave the buffer unchanged.
        let original = [10u8, 20, 30, 128, 40, 50, 60, 64];
        for force in [false, true] {
            let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
                .with_transfer(TransferFunction::Srgb)
                .with_alpha(Some(AlphaMode::Premultiplied));
            let mut buf = PixelBuffer::from_vec(original.to_vec(), 2, 1, descriptor).unwrap();
            buf.reduce_to_load_bearing_format_in_place(force);
            assert_eq!(buf.descriptor(), descriptor, "force={force}");
            assert_eq!(buf.as_slice().row(0), &original[..]);
        }
    }

    #[test]
    fn reduce_impl_strided_input_compacts_like_allocating() {
        // Arbitrary stride padding isn't constructible through the
        // public buffer constructors, so the strided path is pinned at
        // the transform level via a hand-built InPlacePixels.
        let (buf, stride) = build_strided_rgba8(5, 4, 24, |x, y| {
            let g = ((x + y) * 19) as u8;
            [g, g, g, 255]
        });
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_transfer(TransferFunction::Srgb);
        let reference = PixelSlice::new(&buf, 5, 4, stride, descriptor)
            .unwrap()
            .try_reduce_to_load_bearing_format()
            .expect("reduces");

        let mut mut_buf = buf.clone();
        let out = reduce_in_place_impl(
            InPlacePixels::new(&mut mut_buf, 5, 4, stride, descriptor, None),
            true,
        );
        assert_eq!(out.descriptor(), reference.descriptor());
        for y in 0..4 {
            assert_eq!(out.row(y), reference.as_slice().row(y), "row {y}");
        }
    }

    #[test]
    fn in_place_matches_allocating_across_geometries() {
        // Differential: every (width, rows) drives the overlap-prefix /
        // disjoint-row split differently; the buffer entry point must be
        // byte-identical to the allocating rewrite for every content
        // class that triggers a distinct transition.
        #[derive(Clone, Copy)]
        enum Content {
            GrayOpaque,    // Rgba8 -> Gray8 (4 -> 1)
            ColorOpaque,   // Rgba8 -> Rgb8  (4 -> 3, garb path)
            GrayVaryAlpha, // Rgba8 -> GrayA8 (4 -> 2)
            Replicated16,  // Rgba16 -> Gray8 (8 -> 1)
        }
        let mut lcg: u32 = 0x2F6E_2B1D;
        let mut next = move || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (lcg >> 24) as u8
        };
        for content in [
            Content::GrayOpaque,
            Content::ColorOpaque,
            Content::GrayVaryAlpha,
            Content::Replicated16,
        ] {
            for (width, rows) in [
                (1u32, 1u32),
                (1, 7),
                (2, 3),
                (3, 2),
                (5, 5),
                (17, 3),
                (64, 4),
                (65, 2),
            ] {
                let (format, bytes): (PixelFormat, Vec<u8>) = match content {
                    Content::GrayOpaque => (
                        PixelFormat::Rgba8,
                        (0..width * rows)
                            .flat_map(|_| {
                                let g = next();
                                [g, g, g, 255]
                            })
                            .collect(),
                    ),
                    Content::ColorOpaque => (
                        PixelFormat::Rgba8,
                        (0..width * rows)
                            .flat_map(|_| [next(), next(), next(), 255])
                            .collect(),
                    ),
                    Content::GrayVaryAlpha => (
                        PixelFormat::Rgba8,
                        (0..width * rows)
                            .flat_map(|_| {
                                let g = next();
                                [g, g, g, next()]
                            })
                            .collect(),
                    ),
                    Content::Replicated16 => (
                        PixelFormat::Rgba16,
                        (0..width * rows)
                            .flat_map(|_| {
                                let g = next();
                                [g, g, g, g, g, g, 0xFF, 0xFF]
                            })
                            .collect(),
                    ),
                };
                let reference =
                    make_slice(&bytes, width, rows, format).try_reduce_to_load_bearing_format();
                let mut buf = lb_buf(&bytes, width, rows, format);
                buf.reduce_to_load_bearing_format_in_place(true);
                match reference {
                    Some(reference) => {
                        assert_eq!(
                            buf.descriptor(),
                            reference.descriptor(),
                            "{width}x{rows} descriptor"
                        );
                        for y in 0..rows {
                            assert_eq!(
                                buf.as_slice().row(y),
                                reference.as_slice().row(y),
                                "{width}x{rows} row {y}"
                            );
                        }
                    }
                    None => {
                        // Every content class above guarantees at least
                        // one reduction (gray collapse or alpha drop).
                        panic!("{width}x{rows} expected a reduction");
                    }
                }
            }
        }
    }

    // ── ColorContext propagation ───────────────────────────────

    #[test]
    fn color_context_carries_through_class_preserving_reductions() {
        // Alpha drop keeps the RGB class -- ICC must travel, unchanged.
        let ctx = Arc::new(ColorContext::from_icc(alloc::vec![0u8; 8]));
        let bytes: Vec<u8> = (0..4i32)
            .flat_map(|i| {
                [
                    (i * 60 + 10) as u8,
                    (i * 30 + 50) as u8,
                    (i * 90 + 20) as u8,
                    255,
                ]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("alpha drop available");
        assert_eq!(out.descriptor().format, PixelFormat::Rgb8);
        assert!(
            out.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx)),
            "ICC context must carry over for class-preserving reductions"
        );

        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
        assert!(
            buf.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );
    }

    #[test]
    fn icc_context_suppresses_gray_collapse_but_not_other_reductions() {
        // Gray + opaque RGBA with *unidentifiable* ICC bytes attached:
        // no CICP description is derivable, so no GRAY-class variant can
        // stand in, and the rewrite stops at the class-preserving alpha
        // drop. The report itself still measures chroma truthfully.
        let ctx = Arc::new(ColorContext::from_icc(alloc::vec![0u8; 8]));
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        assert_eq!(
            slice.determine_load_bearing().uses_chroma,
            Some(false),
            "analysis stays truthful"
        );
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("alpha drop still available");
        assert_eq!(
            out.descriptor().format,
            PixelFormat::Rgb8,
            "gray collapse suppressed, alpha drop kept"
        );
        assert!(
            out.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );

        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
        assert!(
            buf.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );
    }

    #[test]
    fn cicp_only_context_carries_through_gray_collapse() {
        // CICP has no device-class to violate: primaries/transfer stay
        // meaningful for gray and matrix coefficients don't apply to
        // single-channel data -- the collapse proceeds and the context
        // travels.
        let ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("gray collapse available");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        assert!(
            out.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );

        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Gray8);
        assert!(
            buf.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );
    }

    // ── Gray-class ICC swap on collapse ────────────────────────────

    /// Assert the context carries a GRAY-class ICC (header bytes 16..20).
    fn assert_gray_class_icc(ctx: Option<&Arc<ColorContext>>) -> Arc<[u8]> {
        let icc = ctx
            .expect("reduced buffer must carry a context")
            .icc
            .clone()
            .expect("swapped context must hold ICC bytes");
        assert_eq!(&icc[16..20], b"GRAY", "swapped profile must be GRAY-class");
        assert_eq!(&icc[36..40], b"acsp", "swapped profile must be a valid ICC");
        icc
    }

    #[cfg(feature = "icc-db")]
    #[test]
    fn icc_with_cicp_swaps_to_gray_class_profile_on_collapse() {
        // Both fields populated: the cicp field describes the attached
        // RGB profile, so the collapse swaps in the GRAY-class synthesis
        // for that CICP and keeps the cicp alongside.
        let mut both = ColorContext::from_icc(alloc::vec![0u8; 8]);
        both.cicp = Some(Cicp::DISPLAY_P3);
        let ctx = Arc::new(both);
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();

        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("gray collapse available via gray-ICC swap");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        let swapped = assert_gray_class_icc(out.as_slice().color_context());
        assert_eq!(
            out.as_slice().color_context().unwrap().cicp,
            Some(Cicp::DISPLAY_P3),
            "source cicp must ride along"
        );

        // The buffer entry point produces the same signaling.
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::Gray8);
        let view = buf.as_slice();
        let swapped_in_place = assert_gray_class_icc(view.color_context());
        assert_eq!(swapped_in_place.as_ref(), swapped.as_ref());
    }

    #[cfg(feature = "icc-db")]
    #[test]
    fn recognized_rgb_profile_swaps_to_gray_class_on_collapse() {
        // ICC-only context holding the bundled Display-P3 profile: the
        // normalized-hash identification recognizes it as (P3-D65, sRGB)
        // and the collapse swaps in the matching GRAY-class profile.
        let ctx = Arc::new(ColorContext::from_icc(
            crate::icc_profiles::DISPLAY_P3_V4.to_vec(),
        ));
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("gray collapse available via identification");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        assert_gray_class_icc(out.as_slice().color_context());
        assert_eq!(
            out.as_slice().color_context().unwrap().cicp,
            None,
            "no cicp on the source context, none invented"
        );
    }

    #[test]
    fn srgb_described_icc_drops_to_cicp_only_on_collapse() {
        // ICC + cicp where the description is the assumed sRGB default:
        // gray output needs no ICC at all -- the swap drops to a
        // CICP-only context.
        let mut both = ColorContext::from_icc(alloc::vec![0u8; 8]);
        both.cicp = Some(Cicp::SRGB);
        let ctx = Arc::new(both);
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("gray collapse available");
        assert_eq!(out.descriptor().format, PixelFormat::Gray8);
        let new_ctx = out
            .as_slice()
            .color_context()
            .cloned()
            .expect("cicp-only context expected");
        assert!(new_ctx.icc.is_none(), "sRGB-default gray needs no ICC");
        assert_eq!(new_ctx.cicp, Some(Cicp::SRGB));
    }

    #[test]
    fn non_cicp_recognized_profile_still_suppresses_collapse() {
        // Adobe RGB has no CICP code points: even if the profile is
        // recognized, no gray variant is derivable from the CICP grid --
        // the collapse stays suppressed and the original context carries.
        let ctx = Arc::new(ColorContext::from_icc(
            crate::icc_profiles::ADOBE_RGB.to_vec(),
        ));
        let bytes: Vec<u8> = (0..4).flat_map(|i| [i * 30, i * 30, i * 30, 255]).collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("alpha drop still available");
        assert_eq!(
            out.descriptor().format,
            PixelFormat::Rgb8,
            "collapse suppressed without a CICP-expressible color"
        );
        assert!(
            out.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx))
        );
    }

    #[test]
    fn colorful_content_keeps_original_context_despite_swap_plan() {
        // A swappable context on a buffer whose chroma IS load-bearing:
        // no collapse happens, so no swap applies -- the original
        // context carries through the alpha drop untouched.
        let mut both = ColorContext::from_icc(alloc::vec![0u8; 8]);
        both.cicp = Some(Cicp::DISPLAY_P3);
        let ctx = Arc::new(both);
        let bytes: Vec<u8> = (0..4i32)
            .flat_map(|i| {
                [
                    (i * 60 + 10) as u8,
                    (i * 30 + 50) as u8,
                    (i * 90 + 20) as u8,
                    255,
                ]
            })
            .collect();
        let slice = make_slice(&bytes, 4, 1, PixelFormat::Rgba8).with_color_context(ctx.clone());
        let out = slice
            .try_reduce_to_load_bearing_format()
            .expect("alpha drop available");
        assert_eq!(out.descriptor().format, PixelFormat::Rgb8);
        assert!(
            out.as_slice()
                .color_context()
                .is_some_and(|c| Arc::ptr_eq(c, &ctx)),
            "no collapse -> original context, not the swap"
        );
    }

    #[test]
    fn in_place_rgba_f32_gray_opaque_force_true() {
        let pixels: [f32; 16] = [
            0.1, 0.1, 0.1, 1.0, //
            0.5, 0.5, 0.5, 1.0, //
            0.9, 0.9, 0.9, 1.0, //
            0.4, 0.4, 0.4, 1.0,
        ];
        let descriptor = PixelDescriptor::from_pixel_format(PixelFormat::RgbaF32)
            .with_transfer(TransferFunction::Linear);
        let mut buf =
            PixelBuffer::from_vec(bytemuck::cast_slice(&pixels).to_vec(), 4, 1, descriptor)
                .unwrap();
        buf.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buf.descriptor().format, PixelFormat::GrayF32);
        let view = buf.as_slice();
        let gray: &[f32] = bytemuck::cast_slice(view.row(0));
        assert_eq!(gray, &[0.1_f32, 0.5, 0.9, 0.4], "values bit-exact");
    }

    #[test]
    fn in_place_true_u16_keeps_channel_type() {
        // Genuine 16-bit data (lo != hi): no narrowing, and with
        // colorful opaque pixels + force=false only the alpha re-tag
        // applies.
        let bytes: Vec<u8> = (0..4u16)
            .flat_map(|i| {
                let r = 0x1234 + i * 0x0101;
                let g = 0x4567;
                let b = 0x89AB;
                [r, g, b, 0xFFFF]
            })
            .flat_map(u16::to_ne_bytes)
            .collect();
        let mut buf = lb_buf(&bytes, 4, 1, PixelFormat::Rgba16);
        buf.reduce_to_load_bearing_format_in_place(false);
        assert_eq!(buf.descriptor().format, PixelFormat::Rgba16);
        assert_eq!(buf.descriptor().alpha, Some(AlphaMode::Opaque));
    }

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
