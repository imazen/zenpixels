//! Unified sample evidence: [`FloatFacts`], [`ExactDepth`], and [`SampleFacts`].
//!
//! Owned buffers cache proven facts about their current samples and automatically
//! clear them on every `&mut` borrow so stale bounds can never mislead SIMD fast
//! paths or lossless negotiation.

use crate::layout::SampleType;

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

/// Proven integer/content evidence about current samples (bit depth, opacity, grayscale).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExactDepth {
    /// Proven effective integer bit depth (`0` = unknown, e.g. `8` when every `U16` sample satisfies `v == (v >> 8) * 257`).
    pub effective_bits: u8,
    /// Proven that every alpha sample is fully opaque.
    pub all_opaque: bool,
    /// Proven that `R == G == B` for every pixel.
    pub all_grayscale: bool,
}

/// Unified sample evidence cached on owned buffers and propagated through conversion plans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SampleFacts {
    /// Proven floating-point invariants.
    pub float: FloatFacts,
    /// Proven integer/content invariants.
    pub exact: ExactDepth,
}

impl SampleFacts {
    /// Unknown / unproven sample properties.
    pub const NONE: Self = Self {
        float: FloatFacts::NONE,
        exact: ExactDepth {
            effective_bits: 0,
            all_opaque: false,
            all_grayscale: false,
        },
    };

    /// Facts guaranteed for any freshly zero-allocated buffer of `sample`.
    #[inline]
    #[must_use]
    pub const fn for_zeroed(sample: SampleType) -> Self {
        if sample.is_float() {
            Self {
                float: FloatFacts::UNIT_INTERVAL,
                exact: ExactDepth {
                    effective_bits: 0,
                    all_opaque: false,
                    all_grayscale: true,
                },
            }
        } else {
            Self {
                float: FloatFacts::NONE,
                exact: ExactDepth {
                    effective_bits: 8,
                    all_opaque: false,
                    all_grayscale: true,
                },
            }
        }
    }

    /// Construct facts carrying only proven [`FloatFacts`].
    #[inline]
    #[must_use]
    pub const fn from_float(float: FloatFacts) -> Self {
        Self {
            float,
            exact: ExactDepth {
                effective_bits: 0,
                all_opaque: false,
                all_grayscale: false,
            },
        }
    }
}
