//! Integer sample interpretation for native codec and video planes.
//!
//! Storage width, code depth and bit placement are independent. An unshifted
//! ten-bit code in a `u16` sample is not a normalized sixteen-bit value.
//! This description does not change [`crate::PixelDescriptor`] or the existing
//! RGB/gray U16 numerical contract. A raw plane must retain this encoding with
//! its storage and obtain component roles, range and color from its enclosing
//! signal. Original source precision is separate provenance.
//!
//! [`ExactDepth`] is different from both: evidence that the *current* samples
//! of an ordinary U8/U16 image survive re-quantization to a lower depth. Native
//! codes become canonical samples through `zenpixels_convert::codes`.
//!
//! For floating-point views (`F16`, `F32`, `F64`), [`FloatFacts`] records atomic
//! active-sample invariants (`NO_NAN`, `NO_INFINITE`, `NO_SUBNORMALS`,
//! `NON_NEGATIVE`, `POSITIVE`, `AT_MOST_ONE`) with the same mutation-clearing
//! lifetime as [`ExactDepth`].
//!
//! Stored samples use native endianness. A byte-backed adapter must decode its
//! declared byte order before applying the bit placement. Each description selects
//! one component; multiple views can select different bitfields of a shared word.
//! Signed integer payloads use two's complement within the declared code bits.

use crate::{AlphaMode, ChannelType, PixelDescriptor};
use core::fmt;

mod exact;
mod float;
mod plane;
pub use exact::{ExactDepth, ExactDepthError};
pub use float::{FloatFacts, FloatFactsError};
pub use plane::{ChromaSiting, ChromaSubsampling};
#[cfg(feature = "planar")]
pub use plane::{CodePlane, CodePlaneError, CodePlaneMut};

/// Two-byte internal union of integer [`ExactDepth`] and floating-point [`FloatFacts`],
/// keeping `PixelBuffer` at 64 bytes and `PixelSlice` at 48 bytes on 64-bit targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SampleEvidence {
    // None = empty; bit 7 set (0x80 | ..) = ExactDepth; 0x40 = non-empty FloatFacts.
    tag: Option<core::num::NonZeroU8>,
    payload: u8,
}

const _: () = assert!(core::mem::size_of::<SampleEvidence>() == 2);

impl SampleEvidence {
    pub(crate) const NONE: Self = Self {
        tag: None,
        payload: 0,
    };

    #[inline]
    pub(crate) const fn from_exact(exact: Option<ExactDepth>) -> Self {
        match exact {
            Some(e) => {
                let (tag, payload) = e.into_raw();
                Self {
                    tag: Some(tag),
                    payload,
                }
            }
            None => Self::NONE,
        }
    }

    #[inline]
    pub(crate) const fn from_float(facts: FloatFacts) -> Self {
        if facts.is_empty() {
            Self::NONE
        } else {
            Self {
                tag: core::num::NonZeroU8::new(0x40),
                payload: facts.bits(),
            }
        }
    }

    #[inline]
    pub(crate) const fn exact_depth(self) -> Option<ExactDepth> {
        match self.tag {
            Some(t) if (t.get() & 0x80) != 0 => Some(ExactDepth::from_raw(t, self.payload)),
            _ => None,
        }
    }

    #[inline]
    pub(crate) const fn float_facts(self) -> FloatFacts {
        match self.tag {
            Some(t) if t.get() == 0x40 => FloatFacts::from_raw_normalized(self.payload),
            _ => FloatFacts::NONE,
        }
    }

    #[inline]
    pub(crate) fn survives_relabel(self, old: PixelDescriptor, new: PixelDescriptor) -> Self {
        if self.tag.is_none() {
            return Self::NONE;
        }
        // Promoting uninspected Undefined padding into an active alpha channel invalidates sample proofs.
        if old.alpha() == Some(AlphaMode::Undefined) && new.alpha() != Some(AlphaMode::Undefined) {
            return Self::NONE;
        }
        if self.exact_depth().is_some() {
            if old.channel_type() == new.channel_type()
                && old.layout() == new.layout()
                && old.signal_range == new.signal_range
            {
                self
            } else {
                Self::NONE
            }
        } else if old.channel_type() == new.channel_type() && new.channel_type().is_float() {
            self
        } else {
            Self::NONE
        }
    }
}

/// Checked interpretation of one integer code in a stored sample.
///
/// Extract a code with `(sample >> bit_shift) & ((1 << code_bits) - 1)`, using
/// sufficiently wide arithmetic. Bits outside that payload are padding: ignore
/// them on read and write zero padding on output. Strict format-conformance
/// checks may additionally require zero padding, but are separate sample scans.
/// This value validates the description, not every sample in a pixel allocation.
///
/// No range, normalization or transfer conversion is implied. For example,
/// packing code 1023 at shift 6 produces 65472, whereas rescaling full-range
/// ten-bit intensity into the full sixteen-bit domain produces 65535.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleEncoding {
    storage: ChannelType,
    code_bits: u8,
    bit_shift: u8,
}

impl SampleEncoding {
    /// Describe an integer component without examining or changing samples.
    ///
    /// `storage` must be an integer type. Code depth must be positive and fit in the
    /// storage type; adding the low padding bits must also fit. The check accepts
    /// all such placements, including 8-bit codes in U16 and 10/12-bit codes
    /// aligned to either end of U16. A consuming codec can support a narrower
    /// subset and must check that capability separately.
    pub const fn new(
        storage: ChannelType,
        code_bits: u8,
        bit_shift: u8,
    ) -> Result<Self, SampleEncodingError> {
        let storage_bits = match storage {
            ChannelType::U8 | ChannelType::I8 => 8,
            ChannelType::U16 | ChannelType::I16 => 16,
            ChannelType::U32 | ChannelType::I32 => 32,
            ChannelType::U64 | ChannelType::I64 => 64,
            _ => return Err(SampleEncodingError::UnsupportedStorage),
        };
        if code_bits == 0 || code_bits > storage_bits {
            return Err(SampleEncodingError::InvalidBitDepth);
        }
        // Subtract after checking depth instead of overflowing an unchecked
        // `code_bits + bit_shift` supplied by an external format description.
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
    pub const fn storage(self) -> ChannelType {
        self.storage
    }

    /// Number of code bits in the current sample representation.
    pub const fn code_bits(self) -> u8 {
        self.code_bits
    }

    /// Number of low padding bits below the code in its stored sample.
    pub const fn bit_shift(self) -> u8 {
        self.bit_shift
    }

    /// True when the backing integer storage is signed two's complement
    /// (`I8`, `I16`, `I32`, or `I64`).
    pub const fn is_signed(self) -> bool {
        matches!(
            self.storage,
            ChannelType::I8 | ChannelType::I16 | ChannelType::I32 | ChannelType::I64
        )
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
