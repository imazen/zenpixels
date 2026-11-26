//! Integer sample interpretation, exact code-depth evidence, and native code planes.
//!
//! Storage width, code depth, and bit placement are independent: an unshifted
//! 10-bit code in a `u16` sample is not a normalized 16-bit value.
//!
//! - [`SampleEncoding`] describes the integer storage type, significant bit
//!   depth, and low-bit shift of one coded component without implying range or
//!   color meaning.
//! - [`ExactDepth`] carries proven evidence that the *current* samples of a
//!   canonical `U8`/`U16` image survive re-quantization to a lower depth under
//!   either the full-range (divisibility-nesting) or narrow-range (power-of-two)
//!   canonical mapping, tracking color and alpha depths independently.
//! - [`CodePlane`] reads unpacked integer codes from borrowed native codec
//!   planes (`u8` or `u16`, planar or interleaved) without copying or allocating.
//! - [`ChromaSiting`] and [`ChromaSubsampling`] describe declared chroma sample
//!   locations and subsampling ratios without silently assuming defaults.

use crate::layout::SampleType;
use core::fmt;

mod exact;
mod plane;

pub use exact::{ExactDepth, ExactDepthError, full_range_grid_nests, narrow_range_grid_nests};
pub use plane::{
    ChromaSiting, ChromaSubsampling, CodePlane, CodePlaneError, CodePlaneMut, InterleavedCodeLayout,
};

/// Checked interpretation of one integer code in a stored sample.
///
/// Extract a code with `(sample >> bit_shift) & ((1 << code_bits) - 1)` using
/// sufficiently wide arithmetic. Bits outside that payload are padding: ignored
/// on read and written as zero on output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleEncoding {
    storage: SampleType,
    code_bits: u8,
    bit_shift: u8,
}

impl SampleEncoding {
    /// Describe an integer component without examining or changing samples.
    ///
    /// `storage` must be an integer [`SampleType`]. `code_bits` must be in
    /// `1..=storage.bits()`, and `code_bits + bit_shift <= storage.bits()`.
    pub const fn new(
        storage: SampleType,
        code_bits: u8,
        bit_shift: u8,
    ) -> Result<Self, SampleEncodingError> {
        if !storage.is_integer() {
            return Err(SampleEncodingError::UnsupportedStorage);
        }
        let storage_bits = storage.bits();
        if code_bits == 0 || code_bits > storage_bits {
            return Err(SampleEncodingError::InvalidBitDepth);
        }
        if bit_shift > storage_bits - code_bits {
            return Err(SampleEncodingError::InvalidBitShift);
        }
        Ok(Self {
            storage,
            code_bits,
            bit_shift,
        })
    }

    /// Integer storage type; this does not determine the code depth.
    #[inline]
    #[must_use]
    pub const fn storage(self) -> SampleType {
        self.storage
    }

    /// Number of significant code bits in the sample representation.
    #[inline]
    #[must_use]
    pub const fn code_bits(self) -> u8 {
        self.code_bits
    }

    /// Number of low padding bits below the code in its stored sample.
    #[inline]
    #[must_use]
    pub const fn bit_shift(self) -> u8 {
        self.bit_shift
    }

    /// Convert to [`crate::BitPacking`] when `storage` is unsigned (`U8`, `U16`, `U32`).
    #[inline]
    #[must_use]
    pub const fn to_bit_packing(self) -> Option<crate::BitPacking> {
        if !matches!(
            self.storage,
            SampleType::U8 | SampleType::U16 | SampleType::U32
        ) {
            return None;
        }
        let storage_bits = self.storage.bits();
        if self.code_bits == storage_bits && self.bit_shift == 0 {
            Some(crate::BitPacking::Full)
        } else if self.bit_shift == 0 {
            Some(crate::BitPacking::Lsb {
                bits: self.code_bits,
            })
        } else if self.bit_shift == storage_bits - self.code_bits {
            Some(crate::BitPacking::Msb {
                bits: self.code_bits,
            })
        } else {
            Some(crate::BitPacking::Shifted {
                bits: self.code_bits,
                shift: self.bit_shift,
            })
        }
    }
}

/// An integer sample description cannot fit its declared storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleEncodingError {
    /// The storage type is not an integer.
    UnsupportedStorage,
    /// Code depth is zero or exceeds storage width.
    InvalidBitDepth,
    /// Code depth plus low padding exceeds storage width.
    InvalidBitShift,
}

impl fmt::Display for SampleEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedStorage => "sample encoding requires integer storage",
            Self::InvalidBitDepth => "sample code depth must be positive and fit its storage",
            Self::InvalidBitShift => "sample code bits and low padding exceed storage width",
        })
    }
}

impl core::error::Error for SampleEncodingError {}
