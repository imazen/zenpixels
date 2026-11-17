//! Custom sample and pixel types (`F16Sample` and gray+alpha formats).
//!
//! These types are `#[repr(transparent)]` or `#[repr(C)]` and derive
//! `Pod`/`Zeroable` for sound, zero-cost byte/word reinterpretation via
//! `bytemuck` under `#![forbid(unsafe_code)]`.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

/// IEEE 754 `binary16` (half-precision float) interchange sample.
///
/// The internal representation is intentionally opaque (`#[repr(transparent)]`
/// over a 16-bit IEEE 754 `binary16` payload, 2-byte size and 2-byte alignment)
/// so a future release can back it with `core::primitive::f16` once Rust
/// stabilizes native half-precision floats, without exposing raw integer field
/// access or allowing accidental `F16Sample(1)` (`6e-8` vs `1.0`) bugs.
///
/// Every 16-bit pattern is a valid `F16Sample` (`Zeroable` + `Pod`, where
/// all-zero bits represent `+0.0`), so slices and `Vec`s convert zero-copy to
/// and from `u16` (e.g. for `magetypes::F16Convert`), `half::f16`, or `[u8]`
/// via [`bytemuck`] or the inherent [`as_bits_slice`](Self::as_bits_slice) and
/// [`vec_into_bits`](Self::vec_into_bits) helpers.
///
/// ```compile_fail
/// use zenpixels::F16Sample;
/// let _ = F16Sample(1); // inner bits are private; use from_f32 / from_bits / ONE
/// ```
///
/// ```compile_fail
/// use zenpixels::F16Sample;
/// let x = F16Sample::ONE;
/// let _ = x.0; // inner field is private; use to_bits() or to_f32()
/// ```
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16Sample(u16);

impl F16Sample {
    /// Positive zero (`+0.0`, `0x0000`).
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero (`-0.0`, `0x8000`).
    pub const NEG_ZERO: Self = Self(0x8000);
    /// Multiplicative identity (`1.0`, `0x3c00`).
    pub const ONE: Self = Self(0x3c00);
    /// Negative one (`-1.0`, `0xbc00`).
    pub const NEG_ONE: Self = Self(0xbc00);
    /// Smallest finite `binary16` value (`-65504.0`, `0xfbff`).
    pub const MIN: Self = Self(0xfbff);
    /// Largest finite `binary16` value (`65504.0`, `0x7bff`).
    pub const MAX: Self = Self(0x7bff);
    /// Smallest positive normal `binary16` value (`2^-14`, `0x0400`).
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// Machine epsilon (`2^-10`, `0x1400`).
    pub const EPSILON: Self = Self(0x1400);
    /// Positive infinity (`+inf`, `0x7c00`).
    pub const INFINITY: Self = Self(0x7c00);
    /// Negative infinity (`-inf`, `0xfc00`).
    pub const NEG_INFINITY: Self = Self(0xfc00);
    /// Canonical quiet `NaN` (`0x7e00`).
    pub const NAN: Self = Self(0x7e00);

    /// Construct from raw IEEE 754 `binary16` bits without modifying payloads.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Extract the raw IEEE 754 `binary16` bits without modifying payloads.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Construct from native-endian bytes.
    #[inline]
    #[must_use]
    pub const fn from_ne_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_ne_bytes(bytes))
    }

    /// Return native-endian bytes.
    #[inline]
    #[must_use]
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        self.0.to_ne_bytes()
    }

    /// Construct from little-endian bytes.
    #[inline]
    #[must_use]
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// Return little-endian bytes.
    #[inline]
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Construct from big-endian bytes.
    #[inline]
    #[must_use]
    pub const fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    /// Return big-endian bytes.
    #[inline]
    #[must_use]
    pub const fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }

    /// Convert losslessly to `f32`, preserving sign, subnormals, infinities, and `NaN` payloads.
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp_mant = h & 0x7fff;
        if exp_mant == 0 {
            return f32::from_bits(sign);
        }
        let exp = (exp_mant >> 10) & 0x1f;
        let mant = exp_mant & 0x03ff;
        if exp == 0x1f {
            return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
        }
        if exp == 0 {
            let p = 31 - mant.leading_zeros();
            let biased_exp_f32 = p + 103;
            let stored_mant_f32 = (mant << (23 - p)) & 0x7f_ffff;
            return f32::from_bits(sign | (biased_exp_f32 << 23) | stored_mant_f32);
        }
        let biased_exp_f32 = exp + 112;
        f32::from_bits(sign | (biased_exp_f32 << 23) | (mant << 13))
    }

    /// Convert `f32` to `F16Sample` using IEEE 754 round-to-nearest-even.
    ///
    /// - Values beyond `±65504.0` saturate to `±INFINITY`.
    /// - Values below the smallest subnormal (`2^-24`, ties to even/zero) underflow to `±0.0`.
    /// - `NaN` inputs preserve sign and high-order payload bits with the quiet bit set.
    #[inline]
    #[must_use]
    pub const fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp_f32 = ((bits >> 23) & 0xff) as i32;
        let frac_f32 = bits & 0x7f_ffff;

        if exp_f32 == 0xff {
            if frac_f32 == 0 {
                return Self(sign | 0x7c00);
            }
            let payload = (frac_f32 >> 13) as u16;
            return Self(sign | 0x7c00 | (payload | 0x0200));
        }
        if exp_f32 == 0 {
            return Self(sign);
        }

        let target_exp = exp_f32 - 127 + 15;
        if target_exp >= 0x1f {
            return Self(sign | 0x7c00);
        }
        if target_exp > 0 {
            let truncated = (frac_f32 >> 13) as u16;
            let round_bit = (frac_f32 >> 12) & 1;
            let sticky = (frac_f32 & 0xfff) != 0;
            let round_up = round_bit != 0 && (sticky || (truncated & 1) != 0);
            if !round_up {
                return Self(sign | ((target_exp as u16) << 10) | truncated);
            }
            let new_mant = truncated + 1;
            if new_mant < 0x0400 {
                return Self(sign | ((target_exp as u16) << 10) | new_mant);
            }
            let new_exp = target_exp + 1;
            if new_exp >= 0x1f {
                return Self(sign | 0x7c00);
            }
            return Self(sign | ((new_exp as u16) << 10));
        }

        if target_exp < -10 {
            return Self(sign);
        }

        let shift = (14 - target_exp) as u32;
        let full_mant = (1u32 << 23) | frac_f32;
        let round_bit = (full_mant >> (shift - 1)) & 1;
        let sticky_mask = (1u32 << (shift - 1)) - 1;
        let sticky = (full_mant & sticky_mask) != 0;
        let truncated = full_mant >> shift;
        let round_up = round_bit != 0 && (sticky || (truncated & 1) != 0);
        let result = if round_up { truncated + 1 } else { truncated };
        Self(sign | (result as u16))
    }

    /// Returns `true` if this value is `NaN`.
    #[inline]
    #[must_use]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7fff) > 0x7c00
    }

    /// Returns `true` if this value is `+inf` or `-inf`.
    #[inline]
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7c00
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        (self.0 & 0x7c00) != 0x7c00
    }

    /// Returns `true` if the number is subnormal.
    #[inline]
    #[must_use]
    pub const fn is_subnormal(self) -> bool {
        (self.0 & 0x7c00) == 0 && (self.0 & 0x03ff) != 0
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal, nor `NaN`.
    #[inline]
    #[must_use]
    pub const fn is_normal(self) -> bool {
        let exp = self.0 & 0x7c00;
        exp != 0 && exp != 0x7c00
    }

    /// Returns `true` if the sign bit is set (including `-0.0`, `-inf`, and negative `NaN`).
    #[inline]
    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    /// Returns `true` if the sign bit is clear (including `+0.0`, `+inf`, and positive `NaN`).
    #[inline]
    #[must_use]
    pub const fn is_sign_positive(self) -> bool {
        (self.0 & 0x8000) == 0
    }

    /// IEEE 754 `totalOrder` comparison (`-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`).
    #[inline]
    #[must_use]
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        let mut left = self.0 as i16;
        let mut right = other.0 as i16;
        left ^= (((left >> 15) as u16) >> 1) as i16;
        right ^= (((right >> 15) as u16) >> 1) as i16;
        left.cmp(&right)
    }

    /// Borrow a slice of `F16Sample` as raw `u16` bit patterns without copying
    /// (e.g. for `magetypes::F16Convert::f16_to_f32_slice`).
    #[inline]
    #[must_use]
    pub fn as_bits_slice(slice: &[Self]) -> &[u16] {
        bytemuck::cast_slice(slice)
    }

    /// Mutably borrow a slice of `F16Sample` as raw `u16` bit patterns without copying
    /// (e.g. for `magetypes::F16Convert::f32_to_f16_slice`).
    #[inline]
    #[must_use]
    pub fn as_bits_slice_mut(slice: &mut [Self]) -> &mut [u16] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Borrow a slice of raw `u16` bit patterns as `F16Sample` without copying.
    #[inline]
    #[must_use]
    pub fn from_bits_slice(slice: &[u16]) -> &[Self] {
        bytemuck::cast_slice(slice)
    }

    /// Mutably borrow a slice of raw `u16` bit patterns as `F16Sample` without copying.
    #[inline]
    #[must_use]
    pub fn from_bits_slice_mut(slice: &mut [u16]) -> &mut [Self] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Convert an owned `Vec<F16Sample>` into `Vec<u16>` without reallocating or copying.
    #[inline]
    #[must_use]
    pub fn vec_into_bits(vec: Vec<Self>) -> Vec<u16> {
        bytemuck::allocation::cast_vec(vec)
    }

    /// Convert an owned `Vec<u16>` of IEEE 754 `binary16` bits into `Vec<F16Sample>`
    /// without reallocating or copying.
    #[inline]
    #[must_use]
    pub fn vec_from_bits(vec: Vec<u16>) -> Vec<Self> {
        bytemuck::allocation::cast_vec(vec)
    }
}

impl PartialEq for F16Sample {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}

impl PartialOrd for F16Sample {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl fmt::Debug for F16Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

impl fmt::Display for F16Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

impl From<F16Sample> for f32 {
    #[inline]
    fn from(value: F16Sample) -> Self {
        value.to_f32()
    }
}

impl From<F16Sample> for f64 {
    #[inline]
    fn from(value: F16Sample) -> Self {
        f64::from(value.to_f32())
    }
}

/// Grayscale + alpha, 8-bit.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GrayAlpha8 {
    /// Gray value.
    pub v: u8,
    /// Alpha value.
    pub a: u8,
}

/// Grayscale + alpha, 16-bit.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GrayAlpha16 {
    /// Gray value.
    pub v: u16,
    /// Alpha value.
    pub a: u16,
}

/// Grayscale + alpha, f16.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GrayAlphaF16 {
    /// Gray value.
    pub v: F16Sample,
    /// Alpha value.
    pub a: F16Sample,
}

/// Grayscale + alpha, f32.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GrayAlphaF32 {
    /// Gray value.
    pub v: f32,
    /// Alpha value.
    pub a: f32,
}

impl GrayAlpha8 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: u8, a: u8) -> Self {
        Self { v, a }
    }
}

impl GrayAlpha16 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: u16, a: u16) -> Self {
        Self { v, a }
    }
}

impl GrayAlphaF16 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: F16Sample, a: F16Sample) -> Self {
        Self { v, a }
    }
}

impl GrayAlphaF32 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: f32, a: f32) -> Self {
        Self { v, a }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn gray_alpha8_new_and_fields() {
        let px = GrayAlpha8::new(128, 200);
        assert_eq!(px.v, 128);
        assert_eq!(px.a, 200);
    }

    #[test]
    fn gray_alpha16_new_and_fields() {
        let px = GrayAlpha16::new(1000, 65535);
        assert_eq!(px.v, 1000);
        assert_eq!(px.a, 65535);
    }

    #[test]
    fn gray_alpha_f32_new_and_fields() {
        let px = GrayAlphaF32::new(0.5, 1.0);
        assert_eq!(px.v, 0.5);
        assert_eq!(px.a, 1.0);
    }

    #[test]
    fn gray_alpha8_default() {
        let px = GrayAlpha8::default();
        assert_eq!(px.v, 0);
        assert_eq!(px.a, 0);
    }

    #[test]
    fn gray_alpha16_default() {
        let px = GrayAlpha16::default();
        assert_eq!(px.v, 0);
        assert_eq!(px.a, 0);
    }

    #[test]
    fn gray_alpha_f32_default() {
        let px = GrayAlphaF32::default();
        assert_eq!(px.v, 0.0);
        assert_eq!(px.a, 0.0);
    }

    #[test]
    fn gray_alpha8_eq() {
        let a = GrayAlpha8::new(100, 200);
        let b = GrayAlpha8::new(100, 200);
        let c = GrayAlpha8::new(100, 201);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    #[cfg(feature = "std")]
    fn gray_alpha8_hash() {
        use core::hash::{Hash, Hasher};
        let a = GrayAlpha8::new(100, 200);
        let b = GrayAlpha8::new(100, 200);
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn gray_alpha_f32_partial_eq() {
        let a = GrayAlphaF32::new(0.5, 1.0);
        let b = GrayAlphaF32::new(0.5, 1.0);
        let c = GrayAlphaF32::new(0.5, 0.9);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn gray_alpha8_debug() {
        let s = format!("{:?}", GrayAlpha8::new(10, 20));
        assert!(s.contains("10"));
        assert!(s.contains("20"));
    }

    #[test]
    fn gray_alpha8_clone_copy() {
        let a = GrayAlpha8::new(50, 100);
        let b = a;
        #[allow(clippy::clone_on_copy)]
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn gray_alpha8_bytemuck_roundtrip() {
        let pixels = [GrayAlpha8::new(10, 20), GrayAlpha8::new(30, 40)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes, &[10, 20, 30, 40]);
        let back: &[GrayAlpha8] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &pixels);
    }

    #[test]
    fn gray_alpha16_bytemuck_roundtrip() {
        let pixels = [GrayAlpha16::new(1000, 2000)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 4);
        let back: &[GrayAlpha16] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], pixels[0]);
    }

    #[test]
    fn gray_alpha_f32_bytemuck_roundtrip() {
        let pixels = [GrayAlphaF32::new(0.5, 1.0)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 8);
        let back: &[GrayAlphaF32] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], pixels[0]);
    }

    #[test]
    fn f16_sample_exhaustive_65536_roundtrip_and_total_cmp() {
        assert_eq!(core::mem::size_of::<F16Sample>(), 2);
        assert_eq!(core::mem::align_of::<F16Sample>(), 2);
        assert_eq!(core::mem::size_of::<GrayAlphaF16>(), 4);
        assert_eq!(core::mem::align_of::<GrayAlphaF16>(), 2);

        assert_eq!(F16Sample::default().to_bits(), 0x0000);
        assert_eq!(F16Sample::ZERO, F16Sample::NEG_ZERO);
        assert_ne!(F16Sample::NAN, F16Sample::NAN);
        assert_eq!(
            F16Sample::NEG_ZERO.total_cmp(&F16Sample::ZERO),
            Ordering::Less
        );

        for bits in 0u16..=u16::MAX {
            let h = F16Sample::from_bits(bits);
            assert_eq!(h.to_bits(), bits);
            let f = h.to_f32();
            assert_eq!(h.is_nan(), f.is_nan());
            assert_eq!(h.is_infinite(), f.is_infinite());
            assert_eq!(h.is_finite(), f.is_finite());
            assert_eq!(h.is_sign_negative(), f.is_sign_negative());
            assert_eq!(h.is_sign_positive(), f.is_sign_positive());
            if !h.is_nan() {
                assert_eq!(
                    F16Sample::from_f32(f).to_bits(),
                    bits,
                    "bits={bits:#06x} f={f}"
                );
            } else {
                let rt = F16Sample::from_f32(f);
                assert!(rt.is_nan());
                assert_eq!(rt.is_sign_negative(), h.is_sign_negative());
            }
        }

        // Round-to-nearest-even ties and overflow/underflow boundaries.
        assert_eq!(F16Sample::from_f32(1.0 + 2.0f32.powi(-11)), F16Sample::ONE);
        assert_eq!(
            F16Sample::from_f32(1.0 + 3.0 * 2.0f32.powi(-11)).to_bits(),
            0x3c02
        );
        assert_eq!(F16Sample::from_f32(65520.0), F16Sample::INFINITY);
        assert_eq!(F16Sample::from_f32(-65520.0), F16Sample::NEG_INFINITY);
        assert_eq!(F16Sample::from_f32(2.0f32.powi(-25)).to_bits(), 0x0000);
        assert_eq!(F16Sample::from_f32(-2.0f32.powi(-25)).to_bits(), 0x8000);
        assert_eq!(F16Sample::from_f32(2.0f32.powi(-24)).to_bits(), 0x0001);

        // Monotone total_cmp across all 65,536 bit patterns sorted by total_cmp.
        let mut all: Vec<F16Sample> = (0u16..=u16::MAX).map(F16Sample::from_bits).collect();
        all.sort_by(F16Sample::total_cmp);
        assert!(all[0].is_nan() && all[0].is_sign_negative());
        assert!(all.last().unwrap().is_nan() && all.last().unwrap().is_sign_positive());
        for pair in all.windows(2) {
            assert_eq!(pair[0].total_cmp(&pair[1]), Ordering::Less);
            if !pair[0].is_nan() && !pair[1].is_nan() {
                assert_eq!(
                    pair[0].to_f32().total_cmp(&pair[1].to_f32()),
                    Ordering::Less
                );
            }
        }
    }

    #[test]
    fn f16_sample_zero_copy_slice_and_vec_casts() {
        let mut samples = alloc::vec![F16Sample::NEG_ONE, F16Sample::ZERO, F16Sample::ONE];
        let ptr = samples.as_ptr();
        assert_eq!(
            F16Sample::as_bits_slice(&samples),
            &[0xbc00u16, 0x0000, 0x3c00]
        );
        F16Sample::as_bits_slice_mut(&mut samples)[1] = 0x3800; // 0.5
        assert_eq!(samples[1].to_f32(), 0.5);

        let raw_vec = F16Sample::vec_into_bits(samples);
        assert_eq!(raw_vec.as_ptr(), ptr.cast::<u16>());
        let back = F16Sample::vec_from_bits(raw_vec);
        assert_eq!(back.as_ptr(), ptr);
        assert_eq!(back[1], F16Sample::from_f32(0.5));

        let ga = [GrayAlphaF16::new(F16Sample::from_f32(0.25), F16Sample::ONE)];
        let bytes: &[u8] = bytemuck::cast_slice(&ga);
        assert_eq!(bytes.len(), 4);
        let ga_back: &[GrayAlphaF16] = bytemuck::cast_slice(bytes);
        assert_eq!(ga_back[0], ga[0]);
    }
}
