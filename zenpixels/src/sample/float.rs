use crate::banded::BandedPixelBuffer;
use crate::{AlphaMode, ChannelType, F16Sample, PixelDescriptor, PixelSlice};
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

/// Verified or asserted properties of the active floating-point samples in a view (`F16`, `F32`, `F64`).
///
/// Each flag is an atomic universal statement over active samples (excluding row stride padding
/// and [`AlphaMode::Undefined`] padding channels). Subnormal exclusion ([`Self::NO_SUBNORMALS`]) is
/// signaled independently of sign and positivity ([`Self::NON_NEGATIVE`], [`Self::POSITIVE`]), and
/// no bundled preset combinations are exposed; combine individual flags with `|` or [`Self::union`].
///
/// Like [`ExactDepth`](super::ExactDepth), these facts describe the *current* samples only: any
/// mutable view (`as_slice_mut`, `rows_mut`, `bands_mut`, `row_mut`) clears them, and descriptor
/// changes that alter [`ChannelType`] or promote [`AlphaMode::Undefined`] padding into an active
/// alpha channel drop them.
///
/// ```
/// use zenpixels::{PixelDescriptor, PixelSlice, sample::FloatFacts};
///
/// let samples = [0.0f32, 0.25, 1.0];
/// let slice = PixelSlice::new(
///     bytemuck::cast_slice(&samples),
///     1,
///     1,
///     12,
///     PixelDescriptor::RGBF32_LINEAR,
/// )?;
/// let facts = FloatFacts::measure(&slice)?;
/// assert!(facts.contains(
///     FloatFacts::NO_NAN
///         | FloatFacts::NO_INFINITE
///         | FloatFacts::NO_SUBNORMALS
///         | FloatFacts::NON_NEGATIVE
///         | FloatFacts::AT_MOST_ONE
/// ));
/// assert!(!facts.contains(FloatFacts::POSITIVE)); // 0.0 is non-negative, not positive
/// assert!(facts.holds_for(&slice)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct FloatFacts(u8);

const ALL_BITS: u8 = (1 << 6) - 1;

impl FloatFacts {
    /// No floating-point sample properties are claimed.
    pub const NONE: Self = Self(0);

    /// No active sample is `NaN` (`!x.is_nan()`).
    pub const NO_NAN: Self = Self(1 << 0);

    /// No active sample is `+∞` or `-∞` (`!x.is_infinite()`).
    pub const NO_INFINITE: Self = Self(1 << 1);

    /// No active sample is subnormal (`!x.is_subnormal()`).
    ///
    /// Zero (`±0.0`) is not subnormal under IEEE 754, so zero samples and signed
    /// normal values both satisfy `NO_SUBNORMALS`.
    pub const NO_SUBNORMALS: Self = Self(1 << 2);

    /// Every active sample is `+0.0` or positive (`!x.is_nan() && !x.is_sign_negative()`).
    ///
    /// Excludes negative values, `-0.0`, and `NaN` so sign-preserving math (`copysign`,
    /// signed cube root, non-negative power domains) can elide sign restoration bit-identically.
    /// Implies [`Self::NO_NAN`].
    pub const NON_NEGATIVE: Self = Self((1 << 3) | (1 << 0));

    /// Every active sample is strictly positive (`x > 0.0`).
    ///
    /// Excludes `±0.0`, negative values, and `NaN`. Does *not* imply [`Self::NO_SUBNORMALS`]
    /// or [`Self::NO_INFINITE`]. Implies [`Self::NON_NEGATIVE`] and [`Self::NO_NAN`].
    pub const POSITIVE: Self = Self((1 << 4) | (1 << 3) | (1 << 0));

    /// Every active sample satisfies `x <= 1.0`.
    ///
    /// Excludes values `> 1.0`, `+∞`, and `NaN`. Implies [`Self::NO_NAN`], and when combined
    /// with [`Self::NON_NEGATIVE`] also implies [`Self::NO_INFINITE`].
    pub const AT_MOST_ONE: Self = Self((1 << 5) | (1 << 0));

    #[inline]
    pub(crate) const fn bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub(crate) const fn from_raw_normalized(bits: u8) -> Self {
        Self(bits & ALL_BITS).normalized()
    }

    #[inline]
    const fn normalized(self) -> Self {
        let mut b = self.0 & ALL_BITS;
        if (b & (1 << 4)) != 0 {
            // POSITIVE => NON_NEGATIVE | NO_NAN
            b |= (1 << 3) | (1 << 0);
        }
        if (b & ((1 << 3) | (1 << 5))) != 0 {
            // NON_NEGATIVE or AT_MOST_ONE => NO_NAN
            b |= 1 << 0;
        }
        if (b & ((1 << 3) | (1 << 5))) == ((1 << 3) | (1 << 5)) {
            // 0.0 <= x <= 1.0 => NO_INFINITE
            b |= 1 << 1;
        }
        Self(b)
    }

    /// True when no properties are claimed (`self == FloatFacts::NONE`).
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True when every property in `required` is present in `self`.
    #[inline]
    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        let s = self.normalized();
        let r = required.normalized();
        (s.0 & r.0) == r.0
    }

    /// Combine all properties asserted by either `self` or `other` (closing logical implications).
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0).normalized()
    }

    /// Retain only properties guaranteed by *both* `self` and `other` (e.g. across two row bands).
    #[inline]
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.normalized().0 & other.normalized().0).normalized()
    }

    /// Scan active floating-point samples in `pixels` and return all satisfied [`FloatFacts`].
    ///
    /// Row padding and [`AlphaMode::Undefined`] channels are ignored. Existing facts attached to
    /// `pixels` are not consulted. An empty (`0`-width or `0`-row) float view vacuously satisfies
    /// every fact. Returns [`FloatFactsError::UnsupportedStorage`] when `pixels` is not floating-point.
    #[inline]
    pub fn measure<P>(pixels: &PixelSlice<'_, P>) -> Result<Self, FloatFactsError> {
        Self::measure_erased(&pixels.reborrow_erased_no_color())
    }

    fn measure_erased(pixels: &PixelSlice<'_>) -> Result<Self, FloatFactsError> {
        let desc = pixels.descriptor();
        if !desc.channel_type().is_float() {
            return Err(FloatFactsError::UnsupportedStorage);
        }
        let mut mask = ALL_BITS;
        scan_slice_into(pixels, desc, &mut mask);
        Ok(Self::from_raw_normalized(mask))
    }

    /// Scan active floating-point samples across all bands in `pixels`.
    ///
    /// Returns the intersection of facts satisfied by every band.
    pub fn measure_banded(pixels: &BandedPixelBuffer) -> Result<Self, FloatFactsError> {
        let desc = pixels.descriptor();
        if !desc.channel_type().is_float() {
            return Err(FloatFactsError::UnsupportedStorage);
        }
        let mut mask = ALL_BITS;
        for band in pixels.bands() {
            scan_slice_into(&band, desc, &mut mask);
            if mask == 0 {
                break;
            }
        }
        Ok(Self::from_raw_normalized(mask))
    }

    /// Verify that every active sample in `rows` satisfies `self`.
    ///
    /// Stops at the first counterexample (`Ok(false)`). Ignores row padding,
    /// [`AlphaMode::Undefined`] channels, and any facts already attached to `rows`.
    /// Returns [`FloatFactsError::UnsupportedStorage`] when `rows` is not floating-point.
    #[inline]
    pub fn holds_for<P>(self, rows: &PixelSlice<'_, P>) -> Result<bool, FloatFactsError> {
        self.holds_for_erased(&rows.reborrow_erased_no_color())
    }

    fn holds_for_erased(self, rows: &PixelSlice<'_>) -> Result<bool, FloatFactsError> {
        let desc = rows.descriptor();
        if !desc.channel_type().is_float() {
            return Err(FloatFactsError::UnsupportedStorage);
        }
        let required = self.normalized().0;
        if required == 0 || rows.width() == 0 || rows.rows() == 0 {
            return Ok(true);
        }
        let mut accum = ALL_BITS;
        Ok(check_slice(rows, desc, required, &mut accum))
    }
}

impl BitOr for FloatFacts {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl BitOrAssign for FloatFacts {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

impl BitAnd for FloatFacts {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl BitAndAssign for FloatFacts {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(rhs);
    }
}

impl fmt::Debug for FloatFacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return f.write_str("FloatFacts::NONE");
        }
        f.write_str("FloatFacts(")?;
        let flags = [
            (1 << 0, "NO_NAN"),
            (1 << 1, "NO_INFINITE"),
            (1 << 2, "NO_SUBNORMALS"),
            (1 << 3, "NON_NEGATIVE"),
            (1 << 4, "POSITIVE"),
            (1 << 5, "AT_MOST_ONE"),
        ];
        let mut first = true;
        for (bit, name) in flags {
            if (self.0 & bit) != 0 {
                if !first {
                    f.write_str(" | ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        f.write_str(")")
    }
}

/// Floating-point sample facts cannot be attached to or measured on integer storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FloatFactsError {
    /// Floating-point sample facts require `F16`, `F32`, or `F64` storage.
    UnsupportedStorage,
}

impl fmt::Display for FloatFactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedStorage => {
                "floating-point sample facts require F16, F32, or F64 storage"
            }
        })
    }
}

impl core::error::Error for FloatFactsError {}

#[inline(always)]
fn sample_facts_f16(bits: u16) -> u8 {
    let h = F16Sample::from_bits(bits);
    let abs_bits = bits & 0x7fff;
    let is_nan = abs_bits > 0x7c00;
    let is_inf = abs_bits == 0x7c00;
    let is_sub = abs_bits != 0 && abs_bits < 0x0400;
    let non_neg = !is_nan && (bits & 0x8000) == 0;
    let vf = h.to_f32();
    let pos = vf > 0.0;
    let le_one = vf <= 1.0;
    u8::from(!is_nan)
        | (u8::from(!is_inf) << 1)
        | (u8::from(!is_sub) << 2)
        | (u8::from(non_neg) << 3)
        | (u8::from(pos) << 4)
        | (u8::from(le_one) << 5)
}

#[inline(always)]
fn sample_facts_f32(x: f32) -> u8 {
    let bits = x.to_bits();
    let abs_bits = bits & 0x7fff_ffff;
    let is_nan = abs_bits > 0x7f80_0000;
    let is_inf = abs_bits == 0x7f80_0000;
    let is_sub = abs_bits != 0 && abs_bits < 0x0080_0000;
    let non_neg = !is_nan && (bits & 0x8000_0000) == 0;
    let pos = x > 0.0;
    let le_one = x <= 1.0;
    u8::from(!is_nan)
        | (u8::from(!is_inf) << 1)
        | (u8::from(!is_sub) << 2)
        | (u8::from(non_neg) << 3)
        | (u8::from(pos) << 4)
        | (u8::from(le_one) << 5)
}

#[inline(always)]
fn sample_facts_f64(x: f64) -> u8 {
    let bits = x.to_bits();
    let abs_bits = bits & 0x7fff_ffff_ffff_ffff;
    let is_nan = abs_bits > 0x7ff0_0000_0000_0000;
    let is_inf = abs_bits == 0x7ff0_0000_0000_0000;
    let is_sub = abs_bits != 0 && abs_bits < 0x0010_0000_0000_0000;
    let non_neg = !is_nan && (bits & 0x8000_0000_0000_0000) == 0;
    let pos = x > 0.0;
    let le_one = x <= 1.0;
    u8::from(!is_nan)
        | (u8::from(!is_inf) << 1)
        | (u8::from(!is_sub) << 2)
        | (u8::from(non_neg) << 3)
        | (u8::from(pos) << 4)
        | (u8::from(le_one) << 5)
}

fn scan_slice_into(pixels: &PixelSlice<'_>, desc: PixelDescriptor, mask: &mut u8) {
    if pixels.width() == 0 || pixels.rows() == 0 || *mask == 0 {
        return;
    }
    let channels = desc.channels();
    let has_pad = desc.alpha() == Some(AlphaMode::Undefined);
    let active = channels - usize::from(has_pad);
    match desc.channel_type() {
        ChannelType::F16 => {
            for y in 0..pixels.rows() {
                let row: &[u16] = bytemuck::cast_slice(pixels.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *mask &= sample_facts_f16(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                } else {
                    for chunk in row.chunks(64) {
                        for &s in chunk {
                            *mask &= sample_facts_f16(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                }
            }
        }
        ChannelType::F32 => {
            for y in 0..pixels.rows() {
                let row: &[f32] = bytemuck::cast_slice(pixels.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *mask &= sample_facts_f32(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                } else {
                    for chunk in row.chunks(64) {
                        for &s in chunk {
                            *mask &= sample_facts_f32(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                }
            }
        }
        ChannelType::F64 => {
            for y in 0..pixels.rows() {
                let row: &[f64] = bytemuck::cast_slice(pixels.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *mask &= sample_facts_f64(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                } else {
                    for chunk in row.chunks(64) {
                        for &s in chunk {
                            *mask &= sample_facts_f64(s);
                        }
                        if *mask == 0 {
                            return;
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

fn check_slice(rows: &PixelSlice<'_>, desc: PixelDescriptor, required: u8, accum: &mut u8) -> bool {
    let channels = desc.channels();
    let has_pad = desc.alpha() == Some(AlphaMode::Undefined);
    let active = channels - usize::from(has_pad);
    match desc.channel_type() {
        ChannelType::F16 => {
            for y in 0..rows.rows() {
                let row: &[u16] = bytemuck::cast_slice(rows.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *accum &= sample_facts_f16(s);
                            if (*accum & required) != required {
                                return false;
                            }
                        }
                    }
                } else {
                    for &s in row {
                        *accum &= sample_facts_f16(s);
                        if (*accum & required) != required {
                            return false;
                        }
                    }
                }
            }
        }
        ChannelType::F32 => {
            for y in 0..rows.rows() {
                let row: &[f32] = bytemuck::cast_slice(rows.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *accum &= sample_facts_f32(s);
                            if (*accum & required) != required {
                                return false;
                            }
                        }
                    }
                } else {
                    for &s in row {
                        *accum &= sample_facts_f32(s);
                        if (*accum & required) != required {
                            return false;
                        }
                    }
                }
            }
        }
        ChannelType::F64 => {
            for y in 0..rows.rows() {
                let row: &[f64] = bytemuck::cast_slice(rows.row(y));
                if has_pad {
                    for px in row.chunks_exact(channels) {
                        for &s in &px[..active] {
                            *accum &= sample_facts_f64(s);
                            if (*accum & required) != required {
                                return false;
                            }
                        }
                    }
                } else {
                    for &s in row {
                        *accum &= sample_facts_f64(s);
                        if (*accum & required) != required {
                            return false;
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    true
}
