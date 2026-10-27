use crate::{ChannelType, SignalRange};
use core::fmt;

/// Proven exact code depth of the current samples of a canonical U8/U16 image.
///
/// `color_bits = N` states that every current color sample `v` survives
/// re-quantization to N-bit codes: narrowing it to N bits and widening back to
/// the storage depth reproduces `v` exactly, using the canonical rule for the
/// signal range (nearest full-scale rescaling for full range; power-of-two
/// anchor scaling for narrow range). `alpha_bits` states the same for alpha,
/// which is always full scale. Storage depth itself is always exact, so the
/// weakest evidence is `color_bits == alpha_bits == storage bits`.
///
/// This describes current values, never history. Decoding a ten-bit file does
/// not establish ten-bit evidence for RGB samples produced by a YCbCr matrix,
/// and resizing, filtering, color conversion or any mutable access ends it.
/// It is established only by the code that wrote the samples (for example an
/// exact widening of native codes), by an explicit sample scan, or by an
/// operation that provably preserves it.
///
/// Higher depth is not automatically exact. Full-range N-bit grids nest only
/// when N divides the depth: canonical U16 samples holding eight-bit content
/// are exact at 8 and 16 bits but not at 10 or 12, where 252 and 240 of the 256
/// levels change. The original N-bit codes nevertheless remain recoverable at
/// every higher depth; [`color_codes_recoverable_at`](Self::color_codes_recoverable_at)
/// states that weaker guarantee.
///
/// ```
/// use zenpixels::{ChannelType, SignalRange, sample::ExactDepth};
/// let ten = ExactDepth::new(ChannelType::U16, SignalRange::Full, 10, 16).unwrap();
/// assert!(ten.color_exact_at(10) && ten.color_exact_at(16));
/// assert!(!ten.color_exact_at(12)); // 10 does not divide 12
/// assert!(ten.color_codes_recoverable_at(12));
/// let narrow = ExactDepth::new(ChannelType::U16, SignalRange::Narrow, 10, 16).unwrap();
/// assert!(narrow.color_exact_at(12)); // narrow grids nest by powers of two
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExactDepth {
    storage: ChannelType,
    range: SignalRange,
    color_bits: u8,
    alpha_bits: u8,
}

/// An exact-depth statement is impossible for its storage or range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExactDepthError {
    /// Evidence is defined for canonical U8 and U16 integer storage only.
    UnsupportedStorage,
    /// Depth is zero or exceeds the storage width.
    InvalidBits,
    /// Narrow-range codes need at least eight bits.
    NarrowBelowEightBits,
    /// The evidence names a different storage type or signal range than the pixels.
    DescriptorMismatch,
}

impl fmt::Display for ExactDepthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedStorage => "exact-depth evidence requires U8 or U16 storage",
            Self::InvalidBits => "exact depth must be positive and fit its storage",
            Self::NarrowBelowEightBits => "narrow-range exact depth needs at least 8 bits",
            Self::DescriptorMismatch => "exact-depth evidence disagrees with the pixel descriptor",
        })
    }
}

impl core::error::Error for ExactDepthError {}

impl ExactDepth {
    /// State that current color and alpha samples are exact at the given depths.
    ///
    /// This is an assertion, checked only for consistency with the storage and
    /// range; it does not read samples. Use it where the caller wrote the samples
    /// itself, or after a scan. `alpha_bits` is ignored for layouts without alpha;
    /// pass the storage depth when nothing more is known.
    pub const fn new(
        storage: ChannelType,
        range: SignalRange,
        color_bits: u8,
        alpha_bits: u8,
    ) -> Result<Self, ExactDepthError> {
        let storage_bits = match storage {
            ChannelType::U8 => 8,
            ChannelType::U16 => 16,
            _ => return Err(ExactDepthError::UnsupportedStorage),
        };
        if color_bits == 0
            || color_bits > storage_bits
            || alpha_bits == 0
            || alpha_bits > storage_bits
        {
            return Err(ExactDepthError::InvalidBits);
        }
        match range {
            SignalRange::Full => {}
            SignalRange::Narrow if color_bits >= 8 => {}
            SignalRange::Narrow => return Err(ExactDepthError::NarrowBelowEightBits),
        }
        Ok(Self {
            storage,
            range,
            color_bits,
            alpha_bits,
        })
    }

    /// Storage the statement applies to.
    pub const fn storage(self) -> ChannelType {
        self.storage
    }

    /// Signal range whose canonical rule defines the color grid.
    pub const fn range(self) -> SignalRange {
        self.range
    }

    /// Smallest proven exact color depth.
    pub const fn color_bits(self) -> u8 {
        self.color_bits
    }

    /// Smallest proven exact alpha depth.
    pub const fn alpha_bits(self) -> u8 {
        self.alpha_bits
    }

    /// Whether quantizing color to `bits` and widening back reproduces every sample.
    pub const fn color_exact_at(self, bits: u8) -> bool {
        grid_nests(
            self.storage_bits(),
            self.color_bits,
            bits,
            matches!(self.range, SignalRange::Narrow),
        )
    }

    /// Whether quantizing alpha to `bits` and widening back reproduces every sample.
    pub const fn alpha_exact_at(self, bits: u8) -> bool {
        grid_nests(self.storage_bits(), self.alpha_bits, bits, false)
    }

    /// Whether the N-bit color codes survive a `bits`-deep encoding, recoverable
    /// by quantizing the decoded samples back to N bits. Weaker than exactness.
    pub const fn color_codes_recoverable_at(self, bits: u8) -> bool {
        bits >= self.color_bits
    }

    const fn storage_bits(self) -> u8 {
        match self.storage {
            ChannelType::U8 => 8,
            _ => 16,
        }
    }
}

/// Whether the `base`-bit grid in `storage_bits` storage is a subset of the
/// `depth`-bit grid. Widening to at least the storage depth always round-trips;
/// narrow (power-of-two) grids nest upward; full-range grids nest exactly when
/// `base` divides `depth`, because `2^base - 1` divides `2^depth - 1`.
const fn grid_nests(storage_bits: u8, base: u8, depth: u8, narrow: bool) -> bool {
    if depth == 0 {
        false
    } else if depth >= storage_bits {
        true
    } else if narrow {
        depth >= base
    } else {
        depth % base == 0
    }
}
