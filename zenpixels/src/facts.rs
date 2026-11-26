//! Unified sample evidence: [`FloatFacts`], [`ExactDepth`], and [`SampleFacts`].
//!
//! Owned buffers cache proven facts about their current samples and automatically
//! clear them on every `&mut` borrow so stale bounds can never mislead SIMD fast
//! paths or lossless negotiation.

use crate::color::{AlphaMode, ColorSpace, PixelFormat, SignalRange};
use crate::layout::SampleType;
pub use crate::sample::ExactDepth;

/// 6-bit bitmask of proven properties on floating-point samples.
///
/// Mirrors the exact invariants that `magemath` and SIMD quantization kernels
/// use to elide per-lane NaN, infinity, subnormal, sign, and range guards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FloatFacts(u8);

impl FloatFacts {
    /// No proven properties (untrusted / arbitrary floats).
    pub const NONE: Self = Self(0);
    /// No channel sample is `NaN`.
    pub const NO_NAN: Self = Self(1 << 0);
    /// No channel sample is `+∞` or `-∞`.
    pub const NO_INFINITE: Self = Self(1 << 1);
    /// No channel sample is a subnormal float.
    pub const NO_SUBNORMALS: Self = Self(1 << 2);
    /// Every sample is `>= 0.0` (including `-0.0` flushed or non-negative).
    pub const NON_NEGATIVE: Self = Self(1 << 3);
    /// Every sample is strictly `> 0.0`.
    pub const POSITIVE: Self = Self(1 << 4);
    /// Every sample is `<= 1.0`.
    pub const AT_MOST_ONE: Self = Self(1 << 5);

    /// Finite floats (`NO_NAN | NO_INFINITE`).
    pub const FINITE: Self = Self(Self::NO_NAN.0 | Self::NO_INFINITE.0);
    /// Clean unit interval `[0.0, 1.0]` (`NO_NAN | NO_INFINITE | NO_SUBNORMALS | NON_NEGATIVE | AT_MOST_ONE`).
    pub const UNIT_INTERVAL: Self = Self(
        Self::NO_NAN.0
            | Self::NO_INFINITE.0
            | Self::NO_SUBNORMALS.0
            | Self::NON_NEGATIVE.0
            | Self::AT_MOST_ONE.0,
    );

    /// Raw bitmask representation.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether `self` contains all flags in `other`.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether all samples are proven finite (`NO_NAN | NO_INFINITE`).
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.contains(Self::FINITE)
    }

    /// Whether all samples are proven inside `[0.0, 1.0]` without NaN, Inf, or subnormals.
    #[inline]
    #[must_use]
    pub const fn is_unit_interval(self) -> bool {
        self.contains(Self::UNIT_INTERVAL)
    }

    /// Combine two fact sets (union of proven properties).
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersect two fact sets (properties guaranteed by both).
    #[inline]
    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Measure exact [`FloatFacts`] across a slice of `f32` samples.
    #[must_use]
    pub fn measure_f32_slice(samples: &[f32]) -> Self {
        let mut no_nan = true;
        let mut no_inf = true;
        let mut no_sub = true;
        let mut non_neg = true;
        let mut positive = true;
        let mut at_most_one = true;
        for &v in samples {
            if v.is_nan() {
                no_nan = false;
                non_neg = false;
                positive = false;
                at_most_one = false;
                continue;
            }
            if v.is_infinite() {
                no_inf = false;
            }
            if v.is_subnormal() {
                no_sub = false;
            }
            if v < 0.0 {
                non_neg = false;
                positive = false;
            } else if v == 0.0 {
                positive = false;
            }
            if v > 1.0 {
                at_most_one = false;
            }
        }
        let mut bits = 0u8;
        if no_nan {
            bits |= Self::NO_NAN.0;
        }
        if no_inf {
            bits |= Self::NO_INFINITE.0;
        }
        if no_sub {
            bits |= Self::NO_SUBNORMALS.0;
        }
        if non_neg {
            bits |= Self::NON_NEGATIVE.0;
        }
        if positive {
            bits |= Self::POSITIVE.0;
        }
        if at_most_one {
            bits |= Self::AT_MOST_ONE.0;
        }
        Self(bits)
    }
}

impl core::ops::BitOr for FloatFacts {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl core::ops::BitAnd for FloatFacts {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self.intersect(rhs)
    }
}

/// Unified sample evidence cached on owned buffers and propagated through conversion plans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[doc(alias = "content_facts")]
#[doc(alias = "proven_opaque")]
pub struct SampleFacts {
    /// Proven floating-point invariants.
    pub float: FloatFacts,
    /// Proven exact integer code-grid depth (range-aware, separate color & alpha grids).
    pub depth: Option<ExactDepth>,
    /// Proven that every active alpha sample equals channel max (or layout has no meaningful alpha).
    pub all_opaque: bool,
    /// Proven that `R == G == B` for every pixel (or layout is already grayscale).
    pub all_grayscale: bool,
    /// Proven that every active alpha sample is either `0` or channel max (binary transparency).
    pub binary_alpha: bool,
}

impl SampleFacts {
    /// Unknown / unproven sample properties.
    pub const NONE: Self = Self {
        float: FloatFacts::NONE,
        depth: None,
        all_opaque: false,
        all_grayscale: false,
        binary_alpha: false,
    };

    /// Set floating-point invariants.
    #[inline]
    #[must_use]
    pub const fn with_float(mut self, float: FloatFacts) -> Self {
        self.float = float;
        self
    }

    /// Attach (or clear with `None`) proven [`ExactDepth`].
    ///
    /// Accepts either [`ExactDepth`] directly or [`Option<ExactDepth>`].
    #[inline]
    #[must_use]
    pub fn with_depth(mut self, depth: impl Into<Option<ExactDepth>>) -> Self {
        self.depth = depth.into();
        self
    }

    /// Facts guaranteed for a freshly allocated buffer of `format` (with color channels zeroed
    /// and [`AlphaMode::Opaque`] alpha slots initialized to channel maximum).
    #[inline]
    #[must_use]
    pub const fn for_zeroed_format(format: &PixelFormat) -> Self {
        let spec = format;
        let layout = spec.layout;
        let sample = layout.sample;
        let alpha_mode = spec.color.alpha();
        let all_opaque = matches!(
            alpha_mode,
            AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
        );
        // Freshly allocated Straight/Premul buffers have alpha=0 everywhere, which is binary {0, max}.
        let binary_alpha = true;
        let all_grayscale = layout.order.is_rgb_family() || layout.order.is_gray_family();

        if sample.is_float() {
            return Self {
                float: FloatFacts::UNIT_INTERVAL,
                depth: None,
                all_opaque,
                all_grayscale,
                binary_alpha,
            };
        }

        let depth = if layout.packing.is_full()
            && matches!(sample, SampleType::U8 | SampleType::U16)
            && all_grayscale
        {
            let range = match spec.color {
                ColorSpace::Rgb { range, .. } | ColorSpace::Gray { range, .. } => range,
                ColorSpace::Icc { .. } => SignalRange::Full,
                _ => {
                    return Self {
                        float: FloatFacts::NONE,
                        depth: None,
                        all_opaque,
                        all_grayscale: false,
                        binary_alpha,
                    };
                }
            };
            let color_bits = match range {
                SignalRange::Full => 1,
                SignalRange::Narrow => 8,
            };
            match ExactDepth::new(sample, range, color_bits, 1) {
                Ok(d) => Some(d),
                Err(_) => None,
            }
        } else {
            None
        };

        Self {
            float: FloatFacts::NONE,
            depth,
            all_opaque,
            all_grayscale,
            binary_alpha,
        }
    }

    /// Facts guaranteed for a freshly allocated white-filled buffer of `format`
    /// (color channels filled to reference white and any alpha slot filled to channel maximum).
    #[inline]
    #[must_use]
    pub const fn for_white_format(format: &PixelFormat) -> Self {
        let spec = format;
        let layout = spec.layout;
        let sample = layout.sample;
        let all_opaque = true;
        let binary_alpha = true;
        let all_grayscale = layout.order.is_rgb_family() || layout.order.is_gray_family();

        if sample.is_float() {
            let float = if matches!(
                spec.color,
                ColorSpace::Rgb { .. } | ColorSpace::Gray { .. } | ColorSpace::Icc { .. }
            ) {
                FloatFacts::UNIT_INTERVAL.union(FloatFacts::POSITIVE)
            } else {
                FloatFacts::FINITE
            };
            return Self {
                float,
                depth: None,
                all_opaque,
                all_grayscale,
                binary_alpha,
            };
        }

        let depth = if layout.packing.is_full()
            && matches!(sample, SampleType::U8 | SampleType::U16)
            && all_grayscale
        {
            let range = match spec.color {
                ColorSpace::Rgb { range, .. } | ColorSpace::Gray { range, .. } => range,
                ColorSpace::Icc { .. } => SignalRange::Full,
                _ => {
                    return Self {
                        float: FloatFacts::NONE,
                        depth: None,
                        all_opaque,
                        all_grayscale: false,
                        binary_alpha,
                    };
                }
            };
            let color_bits = match range {
                SignalRange::Full => 1,
                SignalRange::Narrow => 8,
            };
            match ExactDepth::new(sample, range, color_bits, 1) {
                Ok(d) => Some(d),
                Err(_) => None,
            }
        } else {
            None
        };

        Self {
            float: FloatFacts::NONE,
            depth,
            all_opaque,
            all_grayscale,
            binary_alpha,
        }
    }

    /// Construct facts carrying only proven [`FloatFacts`].
    #[inline]
    #[must_use]
    pub const fn from_float(float: FloatFacts) -> Self {
        Self {
            float,
            depth: None,
            all_opaque: false,
            all_grayscale: false,
            binary_alpha: false,
        }
    }

    /// Construct facts carrying proven [`ExactDepth`].
    #[inline]
    #[must_use]
    pub const fn from_depth(depth: ExactDepth) -> Self {
        Self {
            float: FloatFacts::NONE,
            depth: Some(depth),
            all_opaque: false,
            all_grayscale: false,
            binary_alpha: false,
        }
    }

    /// Attach proven `all_opaque` flag (also implies `binary_alpha` when `true`).
    #[inline]
    #[must_use]
    pub const fn with_all_opaque(mut self, all_opaque: bool) -> Self {
        self.all_opaque = all_opaque;
        if all_opaque {
            self.binary_alpha = true;
        }
        self
    }

    /// Attach proven `all_grayscale` flag.
    #[inline]
    #[must_use]
    pub const fn with_all_grayscale(mut self, all_grayscale: bool) -> Self {
        self.all_grayscale = all_grayscale;
        self
    }

    /// Attach proven `binary_alpha` flag.
    #[inline]
    #[must_use]
    pub const fn with_binary_alpha(mut self, binary_alpha: bool) -> Self {
        self.binary_alpha = binary_alpha;
        self
    }

    /// Whether current full-range/narrow integer samples are proven exact at `color_bits` and `alpha_bits`.
    #[inline]
    #[must_use]
    pub const fn is_exact_at(self, color_bits: u8, alpha_bits: u8) -> bool {
        match self.depth {
            Some(d) => d.color_exact_at(color_bits) && d.alpha_exact_at(alpha_bits),
            None => false,
        }
    }
}
