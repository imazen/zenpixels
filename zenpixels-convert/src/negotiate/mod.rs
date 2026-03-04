//! Format negotiation — pick the best conversion target.
//!
//! The cost model separates **effort** (computational work) from **loss**
//! (information destroyed). The [`ConvertIntent`] controls how these axes
//! are weighted: `Fastest` prioritizes low effort, while `LinearLight` and
//! `Blend` prioritize low loss.
//!
//! Consumers that can perform conversions internally (e.g., a JPEG encoder
//! with a fused f32→u8+DCT path) can express this via [`FormatOption`]
//! with a custom [`ConversionCost`], so the negotiation picks their fast
//! path instead of doing a redundant conversion.
//!
//! # How negotiation works
//!
//! For each candidate target in the supported format list, the cost model
//! computes five independent cost components:
//!
//! 1. **Transfer cost**: Effort of applying/removing EOTF (sRGB→linear,
//!    PQ→linear, etc.). Unknown↔anything is free (relabeling).
//! 2. **Depth cost**: Effort of depth conversion (u8→f32, u16→u8). Loss
//!    considers provenance — if the data was originally u8 and was widened
//!    to f32 for processing, converting back to u8 has zero loss.
//! 3. **Layout cost**: Effort of adding/removing/swizzling channels.
//!    RGB→Gray is very lossy (500); BGRA→RGBA is cheap (5, swizzle only).
//! 4. **Alpha cost**: Effort of alpha mode conversion. Straight→premul is
//!    cheap; premul→straight involves division (worse rounding at low alpha).
//! 5. **Primaries cost**: Effort of gamut matrix (3×3 multiply). Loss
//!    considers provenance — narrowing to a gamut that contains the origin
//!    gamut has near-zero loss.
//!
//! These five costs are summed, then the consumer's cost override is added,
//! then a suitability penalty (e.g., "u8 sRGB is bad for linear-light
//! resize") is added to the loss axis. Finally, effort and loss are
//! combined into a single score via [`ConvertIntent`]-specific weights.
//! The candidate with the lowest score wins.
//!
//! # Provenance
//!
//! [`Provenance`] is the key to avoiding unnecessary quality loss. Without
//! it, converting f32→u8 always reports high loss. With it, the cost model
//! knows the data was originally u8 (e.g., from JPEG) and the round-trip
//! is lossless.
//!
//! Update provenance when operations change the data's effective precision
//! or gamut:
//!
//! - JPEG u8 → f32 for resize → back to u8: **No update needed.** Origin
//!   is still u8.
//! - sRGB data → convert to BT.2020 → saturation boost → back to sRGB:
//!   **Call `invalidate_primaries(Bt2020)`** because the data now genuinely
//!   uses the wider gamut.
//! - 16-bit PNG → process in f32 → back to u16: **No update needed.**
//!   Origin is still u16.
//!
//! # Consumer cost overrides
//!
//! [`FormatOption::with_cost`] lets codecs advertise fast internal paths.
//! Example: a JPEG encoder with a fused f32→u8+DCT kernel can accept
//! f32 data directly, saving the caller a separate f32→u8 conversion:
//!
//! ```rust,ignore
//! let options = &[
//!     FormatOption::from(PixelDescriptor::RGB8_SRGB),    // native: zero cost
//!     FormatOption::with_cost(
//!         PixelDescriptor::RGBF32_LINEAR,
//!         ConversionCost::new(5, 0),  // fast fused path
//!     ),
//! ];
//! ```
//!
//! Without the override, the negotiator would prefer RGB8_SRGB (no
//! conversion needed) even when the source is already f32. With the
//! override, it sees that delivering f32 directly costs only 5 effort
//! (the encoder's fused path) vs. 40+ effort (our f32→u8 conversion).
//!
//! # Suitability penalties
//!
//! The cost model adds quality-of-operation penalties independent of
//! conversion cost. For example, bilinear resize in sRGB u8 produces
//! gamma-darkening artifacts (measured ΔE ≈ 13.7) regardless of how
//! cheap the u8→u8 identity "conversion" is. `LinearLight` intent
//! penalizes non-linear formats by 120 loss points, steering the
//! negotiator toward f32 linear even when u8 sRGB is cheaper to deliver.

mod cost;
#[cfg(test)]
mod tests;

use core::ops::Add;

use crate::{AlphaMode, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction};

use cost::best_of;

// Re-export pub(crate) items used by other modules (path.rs, op_format.rs).
pub(crate) use cost::{channel_bits, suitability_loss, weighted_score};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Tracks where pixel data came from, so the cost model can distinguish
/// "f32 that was widened from u8 JPEG" (lossless round-trip back to u8)
/// from "f32 that was decoded from a 16-bit EXR" (lossy truncation to u8).
///
/// Without provenance, `depth_cost(f32 → u8)` always reports high loss.
/// With provenance, it can see that the data's *true* precision is u8,
/// so the round-trip is lossless.
///
/// # Gamut provenance
///
/// `origin_primaries` tracks the gamut of the original source, enabling
/// lossless round-trip detection for gamut conversions. For example,
/// sRGB data placed in BT.2020 for processing can round-trip back to
/// sRGB losslessly — but only if no operations expanded the actual color
/// usage (e.g., saturation boost filling the wider gamut). When an
/// operation does expand gamut usage, the caller must update provenance
/// via [`invalidate_primaries`](Self::invalidate_primaries) to reflect
/// that the data now genuinely uses the wider gamut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// The channel depth of the original source data.
    ///
    /// For a JPEG (u8 sRGB) decoded into f32 for resize, this is `U8`.
    /// For an EXR (f32) loaded directly, this is `F32`.
    /// For a 16-bit PNG, this is `U16`.
    pub origin_depth: ChannelType,

    /// The color primaries of the original source data.
    ///
    /// For a standard sRGB JPEG, this is `Bt709`. For a Display P3 image,
    /// this is `DisplayP3`. Used to detect when converting to a narrower
    /// gamut is lossless (the source fits entirely within the target).
    ///
    /// **Important:** If an operation expands the data's gamut usage
    /// (e.g., saturation boost in BT.2020 that pushes colors outside
    /// the original sRGB gamut), call [`invalidate_primaries`](Self::invalidate_primaries)
    /// to update this to the current working primaries. Otherwise the
    /// cost model will incorrectly report the gamut narrowing as lossless.
    pub origin_primaries: ColorPrimaries,
}

impl Provenance {
    /// Assume the descriptor's properties *are* the true origin characteristics.
    ///
    /// This is the conservative default: if you don't know the data's history,
    /// assume its current format is its true origin.
    #[inline]
    pub fn from_source(desc: PixelDescriptor) -> Self {
        Self {
            origin_depth: desc.channel_type(),
            origin_primaries: desc.primaries,
        }
    }

    /// Create provenance with an explicit origin depth. Primaries default to BT.709.
    ///
    /// Use this when the data has been widened from a known source depth.
    /// For example, a JPEG (u8) decoded into f32 for resize:
    ///
    /// ```rust,ignore
    /// let provenance = Provenance::with_origin_depth(ChannelType::U8);
    /// ```
    #[inline]
    pub const fn with_origin_depth(origin_depth: ChannelType) -> Self {
        Self {
            origin_depth,
            origin_primaries: ColorPrimaries::Bt709,
        }
    }

    /// Create provenance with explicit origin depth and primaries.
    #[inline]
    pub const fn with_origin(origin_depth: ChannelType, origin_primaries: ColorPrimaries) -> Self {
        Self {
            origin_depth,
            origin_primaries,
        }
    }

    /// Create provenance with an explicit origin primaries. Depth defaults to
    /// the descriptor's current channel type.
    #[inline]
    pub fn with_origin_primaries(desc: PixelDescriptor, primaries: ColorPrimaries) -> Self {
        Self {
            origin_depth: desc.channel_type(),
            origin_primaries: primaries,
        }
    }

    /// Mark the gamut provenance as invalid (matches current format).
    ///
    /// Call this after any operation that expands the data's color usage
    /// beyond the original gamut. For example, if sRGB data is converted
    /// to BT.2020 and then saturation is boosted to fill the wider gamut,
    /// the origin is no longer sRGB — the data genuinely uses BT.2020.
    ///
    /// After this call, converting to a narrower gamut (e.g., back to sRGB)
    /// will correctly report gamut clipping loss.
    #[inline]
    pub fn invalidate_primaries(&mut self, current: ColorPrimaries) {
        self.origin_primaries = current;
    }
}

/// What the consumer plans to do with the converted pixels.
///
/// Shifts the format negotiation cost model to prefer formats
/// suited for the intended operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvertIntent {
    /// Minimize conversion effort. Good for encoding — the codec
    /// already knows what it wants, just get there cheaply.
    #[default]
    Fastest,

    /// Pixel-accurate operations that need linear light:
    /// resize, blur, anti-aliasing, mipmap generation.
    /// Prefers f32 Linear. Straight alpha is fine.
    LinearLight,

    /// Compositing and blending (Porter-Duff, layer merge).
    /// Prefers f32 Linear with Premultiplied alpha.
    Blend,

    /// Perceptual adjustments: sharpening, contrast, saturation.
    /// Prefers sRGB space (perceptually uniform).
    Perceptual,
}

/// Two-axis conversion cost: computational effort vs. information loss.
///
/// These are independent concerns:
/// - A fast conversion can be very lossy (f32 HDR → u8 sRGB clamp).
/// - A slow conversion can be lossless (u8 sRGB → f32 Linear).
/// - A consumer's fused path can be fast with the same loss as our path.
///
/// [`ConvertIntent`] controls how the two axes are weighted for ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConversionCost {
    /// Computational work (cycles, not quality). Lower is faster.
    pub effort: u16,
    /// Information destroyed (precision, gamut, channels). Lower is more faithful.
    pub loss: u16,
}

impl ConversionCost {
    /// Zero cost — identity conversion.
    pub const ZERO: Self = Self { effort: 0, loss: 0 };

    /// Create a cost with explicit effort and loss.
    pub const fn new(effort: u16, loss: u16) -> Self {
        Self { effort, loss }
    }
}

impl Add for ConversionCost {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            effort: self.effort.saturating_add(rhs.effort),
            loss: self.loss.saturating_add(rhs.loss),
        }
    }
}

/// A supported format with optional consumer-provided cost override.
///
/// Use this with [`best_match_with`] when the consumer can handle some
/// formats more efficiently than the default conversion path.
///
/// # Example
///
/// A JPEG encoder with a fast internal f32→u8 path:
///
/// ```rust,ignore
/// use zenpixels::{FormatOption, ConversionCost};
/// use zenpixels::PixelDescriptor;
///
/// let options = &[
///     FormatOption::from(PixelDescriptor::RGB8_SRGB),     // native, zero cost
///     FormatOption::with_cost(
///         PixelDescriptor::RGBF32_LINEAR,
///         ConversionCost::new(5, 0),  // fast fused path, no extra loss
///     ),
/// ];
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FormatOption {
    /// The pixel format the consumer can accept.
    pub descriptor: PixelDescriptor,
    /// Additional cost the consumer incurs to handle this format
    /// after we deliver it. Zero for native formats.
    pub consumer_cost: ConversionCost,
}

impl FormatOption {
    /// Create an option with explicit consumer cost.
    pub const fn with_cost(descriptor: PixelDescriptor, consumer_cost: ConversionCost) -> Self {
        Self {
            descriptor,
            consumer_cost,
        }
    }
}

impl From<PixelDescriptor> for FormatOption {
    fn from(descriptor: PixelDescriptor) -> Self {
        Self {
            descriptor,
            consumer_cost: ConversionCost::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Pick the best conversion target from `supported` for the given source.
///
/// Returns `None` only if `supported` is empty.
///
/// This is the simple API — all consumer costs are assumed zero,
/// and provenance is inferred from the source descriptor (conservative).
/// Use [`best_match_with`] for consumer cost overrides, or
/// [`negotiate`] for full control over provenance and consumer costs.
pub fn best_match(
    source: PixelDescriptor,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    negotiate(
        source,
        Provenance::from_source(source),
        supported.iter().map(|&d| FormatOption::from(d)),
        intent,
    )
}

/// Pick the best conversion target from `options`, accounting for
/// consumer-provided cost overrides.
///
/// Returns `None` only if `options` is empty.
///
/// Each [`FormatOption`] specifies a format the consumer can accept
/// and what it costs the consumer to handle that format internally.
/// The total cost is `our_conversion + consumer_cost`, weighted by intent.
///
/// Provenance is inferred from the source descriptor. Use [`negotiate`]
/// when the data has been widened from a lower-precision origin.
pub fn best_match_with(
    source: PixelDescriptor,
    options: &[FormatOption],
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    negotiate(
        source,
        Provenance::from_source(source),
        options.iter().copied(),
        intent,
    )
}

/// Fully-parameterized format negotiation.
///
/// This is the most flexible entry point: it takes explicit provenance
/// (so the cost model knows the data's true origin precision) and
/// consumer cost overrides (so fused conversion paths are accounted for).
///
/// # When to use
///
/// Use this when the current pixel format doesn't represent the data's
/// true precision. For example, a JPEG image (u8 sRGB) decoded into f32
/// for gamma-correct resize: the data is *currently* f32, but its origin
/// precision is u8. Converting back to u8 for JPEG encoding is lossless
/// (within ±1 LSB), and the cost model should reflect that.
///
/// ```rust,ignore
/// let provenance = Provenance::with_origin_depth(ChannelType::U8);
/// let target = negotiate(
///     current_f32_desc,
///     provenance,
///     encoder_options.iter().copied(),
///     ConvertIntent::Fastest,
/// );
/// ```
pub fn negotiate(
    source: PixelDescriptor,
    provenance: Provenance,
    options: impl Iterator<Item = FormatOption>,
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    best_of(
        source,
        provenance,
        options.map(|o| (o.descriptor, o.consumer_cost)),
        intent,
    )
}

/// Recommend the ideal working format for a given intent, based on the source format.
///
/// Unlike [`best_match`], this isn't constrained to a list — it returns what
/// the consumer *should* be working in for optimal results.
///
/// Key principles:
/// - **Fastest** preserves the source format (identity).
/// - **LinearLight** upgrades to f32 Linear for gamma-correct operations.
/// - **Blend** upgrades to f32 Linear with premultiplied alpha.
/// - **Perceptual** keeps u8 sRGB for SDR-8 sources (LUT-fast), upgrades others to f32 sRGB.
/// - Never downgrades precision — a u16 source won't be recommended as u8.
pub fn ideal_format(source: PixelDescriptor, intent: ConvertIntent) -> PixelDescriptor {
    match intent {
        ConvertIntent::Fastest => source,

        ConvertIntent::LinearLight => {
            if source.channel_type() == ChannelType::F32
                && source.transfer() == TransferFunction::Linear
            {
                return source;
            }
            PixelDescriptor::new(
                ChannelType::F32,
                source.layout(),
                source.alpha(),
                TransferFunction::Linear,
            )
        }

        ConvertIntent::Blend => {
            let alpha = if source.layout().has_alpha() {
                Some(AlphaMode::Premultiplied)
            } else {
                source.alpha()
            };
            if source.channel_type() == ChannelType::F32
                && source.transfer() == TransferFunction::Linear
                && source.alpha() == alpha
            {
                return source;
            }
            PixelDescriptor::new(
                ChannelType::F32,
                source.layout(),
                alpha,
                TransferFunction::Linear,
            )
        }

        ConvertIntent::Perceptual => {
            let tier = precision_tier(source);
            match tier {
                PrecisionTier::Sdr8 => {
                    if source.transfer() == TransferFunction::Srgb
                        || source.transfer() == TransferFunction::Unknown
                    {
                        return source;
                    }
                    PixelDescriptor::new(
                        ChannelType::U8,
                        source.layout(),
                        source.alpha(),
                        TransferFunction::Srgb,
                    )
                }
                _ => PixelDescriptor::new(
                    ChannelType::F32,
                    source.layout(),
                    source.alpha(),
                    TransferFunction::Srgb,
                ),
            }
        }
    }
}

/// Compute the two-axis conversion cost for `from` → `to`.
///
/// This is the cost of *our* conversion kernels — it doesn't include
/// any consumer-side cost. Intent-independent.
///
/// Provenance is inferred from `from` (conservative: assumes current
/// depth is the true origin). Use [`conversion_cost_with_provenance`]
/// when the data has been widened from a lower-precision source.
pub fn conversion_cost(from: PixelDescriptor, to: PixelDescriptor) -> ConversionCost {
    conversion_cost_with_provenance(from, to, Provenance::from_source(from))
}

/// Compute the two-axis conversion cost with explicit provenance.
///
/// Like [`conversion_cost`], but uses the provided [`Provenance`] to
/// determine whether a depth narrowing is actually lossy. For example,
/// `f32 → u8` reports zero loss when `provenance.origin_depth == U8`,
/// because the data was originally u8 and the round-trip is lossless.
pub fn conversion_cost_with_provenance(
    from: PixelDescriptor,
    to: PixelDescriptor,
    provenance: Provenance,
) -> ConversionCost {
    cost::transfer_cost(from.transfer(), to.transfer())
        + cost::depth_cost(
            from.channel_type(),
            to.channel_type(),
            provenance.origin_depth,
        )
        + cost::layout_cost(from.layout(), to.layout())
        + cost::alpha_cost(from.alpha(), to.alpha(), from.layout(), to.layout())
        + cost::primaries_cost(from.primaries, to.primaries, provenance.origin_primaries)
}

// ---------------------------------------------------------------------------
// Precision tier (used by ideal_format only)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PrecisionTier {
    Sdr8 = 0,
    Sdr16 = 1,
    LinearF32 = 2,
    Hdr = 3,
}

#[allow(unreachable_patterns)] // non_exhaustive: future variants
fn precision_tier(desc: PixelDescriptor) -> PrecisionTier {
    if matches!(
        desc.transfer(),
        TransferFunction::Pq | TransferFunction::Hlg
    ) {
        return PrecisionTier::Hdr;
    }
    match desc.channel_type() {
        ChannelType::U8 => PrecisionTier::Sdr8,
        ChannelType::U16 | ChannelType::F16 => PrecisionTier::Sdr16,
        ChannelType::F32 => PrecisionTier::LinearF32,
        _ => PrecisionTier::Sdr8,
    }
}
