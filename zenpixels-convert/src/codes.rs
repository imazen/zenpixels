//! Exact depth, placement and range conversion of native integer codes.
//!
//! A *code* is the unsigned integer that a [`SampleEncoding`] extracts from a
//! storage word. Ordinary (canonical) `U16` and `U8` image samples are codes of
//! the full word width: a canonical `U16` white is 65535, never a ten-bit 1023.
//! Native codecs keep their codes in `CodePlane` views; this module converts
//! between those codes and canonical samples, or between two native encodings.
//!
//! The rule depends on what the component means, not on its storage:
//!
//! | Role and range | Depth change from `N` to `M` bits | Inverse of a widening |
//! |---|---|---|
//! | Intensity (R′G′B′, gray, Y′), full range | Nearest `(q·max(M) + max(N)/2) / max(N)` | Exact: every code returns |
//! | Intensity or chroma, narrow range | `q << (M−N)`; narrowing rounds half up, clamped to `max(M)` | Exact |
//! | Alpha, any color range | Nearest full-scale, as full-range intensity | Exact |
//! | Chroma, full range | Refused: neutral and span have no exact relation between depths | — |
//!
//! `max(N) = 2^N − 1`; the odd denominators make nearest rounding tie-free.
//! Narrow range needs at least eight bits so its anchors (16·2^(N−8) black,
//! 128·2^(N−8) neutral) are integers. Placement (bit shift) and storage width
//! are applied after the code conversion; padding bits are ignored on reads and
//! written as zero.
//!
//! Widening and then narrowing recovers every code. Narrowing arbitrary codes
//! is lossy. A higher depth is not automatically exact for lower-depth content:
//! canonical `U16` samples widened from eight-bit codes do not survive
//! quantization to ten bits and back (252 of 256 change). Exact full-range
//! re-quantization requires `M = N`, `N | M`, or the full storage width.
//!
//! ```
//! use zenpixels::{ChannelType, SignalRange, sample::SampleEncoding};
//! use zenpixels_convert::codes::{CodeMap, CodeRole};
//!
//! let canonical = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
//! let ten = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
//! let p010 = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
//!
//! let widen = CodeMap::new(ten, canonical, CodeRole::Intensity, SignalRange::Full).unwrap();
//! let narrow = CodeMap::new(canonical, ten, CodeRole::Intensity, SignalRange::Full).unwrap();
//! assert_eq!(widen.word(1023), 65535); // full-range white stays white
//! assert_eq!(narrow.word(widen.word(512)), 512);
//!
//! // Packing is not normalization: P010 white is 1023 << 6 = 65472.
//! let pack = CodeMap::new(ten, p010, CodeRole::Intensity, SignalRange::Full).unwrap();
//! assert_eq!(pack.word(1023), 65472);
//!
//! // Narrow-range anchors scale by powers of two: ten-bit black 64 → 4096.
//! let narrow_widen = CodeMap::new(ten, canonical, CodeRole::Intensity, SignalRange::Narrow).unwrap();
//! assert_eq!(narrow_widen.word(64), 4096);
//! ```

use core::fmt;
#[cfg(feature = "planar")]
use whereat::At;
#[cfg(feature = "planar")]
use zenpixels::{
    AlphaMode, ChannelLayout, PixelDescriptor, PixelSliceMut,
    sample::{CodePlane, CodePlaneError, ExactDepth},
};
use zenpixels::{ChannelType, SignalRange, sample::SampleEncoding};

/// What a component's codes mean, which selects its depth-conversion rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CodeRole {
    /// R′G′B′, gray or Y′: zero is black; the signal range selects white.
    Intensity,
    /// Cb or Cr: a signed offset from a neutral code that scales with depth.
    Chroma,
    /// Coverage: always full scale, whatever the color signal range.
    Alpha,
}

/// A code conversion cannot be performed exactly as requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodeMapError {
    /// Numerical mapping requires matching-signedness 8/16/32-bit integer words,
    /// or a word-slice helper was called with a narrower word type than the encoding.
    UnsupportedStorage,
    /// The signal range has no defined code rule here.
    UnsupportedRange(SignalRange),
    /// Narrow-range anchors are not integers below eight bits.
    NarrowBelowEightBits {
        /// The offending code depth.
        bits: u8,
    },
    /// Full-range chroma has no exact neutral/span relation between depths.
    FullRangeChromaDepth {
        /// Source code depth.
        from_bits: u8,
        /// Destination code depth.
        to_bits: u8,
    },
    /// Source and destination word slices differ in length.
    LengthMismatch {
        /// Source words.
        source: usize,
        /// Destination words.
        destination: usize,
    },
}

impl fmt::Display for CodeMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStorage => f.write_str(
                "code mapping requires matching unsigned (U8/U16/U32) or signed (I8/I16/I32) integer words",
            ),
            Self::UnsupportedRange(range) => write!(f, "no code rule for signal range {range}"),
            Self::NarrowBelowEightBits { bits } => {
                write!(f, "narrow-range codes need at least 8 bits, got {bits}")
            }
            Self::FullRangeChromaDepth { from_bits, to_bits } => write!(
                f,
                "full-range chroma has no exact {from_bits}-bit to {to_bits}-bit rule"
            ),
            Self::LengthMismatch {
                source,
                destination,
            } => write!(
                f,
                "code map source has {source} words, destination {destination}"
            ),
        }
    }
}

impl core::error::Error for CodeMapError {}

#[inline]
const fn mask_u32(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[inline]
const fn max_u64(bits: u8) -> u64 {
    mask_u32(bits) as u64
}

#[inline]
const fn sign_extend_i32(raw: u32, bits: u8) -> i32 {
    let shift = 32 - (bits as u32);
    ((raw << shift) as i32) >> shift
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Same,
    Nearest { from_max: u64, to_max: u64 },
    ShiftLeft(u32),
    RoundShiftRight { shift: u32, max: u32 },
    SignedSame,
    SignedScale { from_bits: u8, to_bits: u8 },
}

/// A checked conversion of one component from one sample encoding to another.
///
/// Construction selects the rule once from the role, signal range and both
/// encodings; mapping performs integer arithmetic only and never allocates.
/// Supports unsigned (`U8`, `U16`, `U32`, `1..=32` bits) and signed
/// two's-complement (`I8`, `I16`, `I32`, `1..=32` bits) sample encodings when
/// both endpoints share signedness. See the [module documentation](self) for
/// the rules and their guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeMap {
    from: SampleEncoding,
    to: SampleEncoding,
    rule: Rule,
}

impl CodeMap {
    /// Prepare the conversion of `role` codes from `from` to `to`.
    pub const fn new(
        from: SampleEncoding,
        to: SampleEncoding,
        role: CodeRole,
        range: SignalRange,
    ) -> Result<Self, CodeMapError> {
        let unsigned_pair = matches!(
            from.storage(),
            ChannelType::U8 | ChannelType::U16 | ChannelType::U32
        ) && matches!(
            to.storage(),
            ChannelType::U8 | ChannelType::U16 | ChannelType::U32
        );
        let signed_pair = matches!(
            from.storage(),
            ChannelType::I8 | ChannelType::I16 | ChannelType::I32
        ) && matches!(
            to.storage(),
            ChannelType::I8 | ChannelType::I16 | ChannelType::I32
        );
        if !unsigned_pair && !signed_pair {
            return Err(CodeMapError::UnsupportedStorage);
        }
        let (n, m) = (from.code_bits(), to.code_bits());
        let full_scale = match (role, range) {
            (CodeRole::Alpha, _) => true,
            (_, SignalRange::Full) => true,
            (_, SignalRange::Narrow) => false,
            (_, other) => return Err(CodeMapError::UnsupportedRange(other)),
        };
        if signed_pair {
            if !full_scale {
                return Err(CodeMapError::UnsupportedRange(range));
            }
            let rule = if n == m {
                Rule::SignedSame
            } else {
                Rule::SignedScale {
                    from_bits: n,
                    to_bits: m,
                }
            };
            return Ok(Self { from, to, rule });
        }
        let rule = if n == m {
            Rule::Same
        } else if full_scale {
            if matches!(role, CodeRole::Chroma) {
                return Err(CodeMapError::FullRangeChromaDepth {
                    from_bits: n,
                    to_bits: m,
                });
            }
            Rule::Nearest {
                from_max: max_u64(n),
                to_max: max_u64(m),
            }
        } else if n < 8 || m < 8 {
            return Err(CodeMapError::NarrowBelowEightBits {
                bits: if n < m { n } else { m },
            });
        } else if m > n {
            Rule::ShiftLeft((m - n) as u32)
        } else {
            Rule::RoundShiftRight {
                shift: (n - m) as u32,
                max: mask_u32(m),
            }
        };
        Ok(Self { from, to, rule })
    }

    /// Encoding of the words this map reads.
    pub const fn input(&self) -> SampleEncoding {
        self.from
    }

    /// Encoding of the words this map writes.
    pub const fn output(&self) -> SampleEncoding {
        self.to
    }

    /// Convert one unsigned `1..=16`-bit source code (bits above its depth are ignored)
    /// to a destination code. For `>16`-bit unsigned encodings, use [`code_u32`](Self::code_u32).
    pub const fn code(&self, code: u16) -> u16 {
        self.code_u32(code as u32) as u16
    }

    /// Convert one unsigned `1..=32`-bit source code (bits above its depth are ignored)
    /// to a destination code.
    pub const fn code_u32(&self, code: u32) -> u32 {
        let q = code & mask_u32(self.from.code_bits());
        match self.rule {
            Rule::Same => q,
            // q ≤ 2^32 - 1, to_max ≤ 2^32 - 1: q·to_max + from_max/2 < 2^64.
            Rule::Nearest { from_max, to_max } => {
                (((q as u64) * to_max + from_max / 2) / from_max) as u32
            }
            Rule::ShiftLeft(s) => q << s,
            Rule::RoundShiftRight { shift, max } => {
                let r = (((q as u64) + (1u64 << (shift - 1))) >> shift) as u32;
                if r > max { max } else { r }
            }
            Rule::SignedSame | Rule::SignedScale { .. } => {
                (self.code_i32(q as i32) as u32) & mask_u32(self.to.code_bits())
            }
        }
    }

    /// Convert one signed two's-complement `1..=32`-bit source code (bits above its depth
    /// are ignored via sign extension) to a sign-extended destination code.
    ///
    /// Preserves neutral `0`, maps positive codes `[0, 2^(N-1)-1]` by tie-free odd-denominator
    /// nearest scaling onto `[0, 2^(M-1)-1]`, and scales negative codes `[-2^(N-1), -1]` onto
    /// `[-2^(M-1), -1]` with exact widening/narrowing round-trip.
    pub const fn code_i32(&self, code: i32) -> i32 {
        match self.rule {
            Rule::SignedSame => sign_extend_i32(code as u32, self.from.code_bits()),
            Rule::SignedScale {
                from_bits: n,
                to_bits: m,
            } => {
                let q = sign_extend_i32(code as u32, n);
                if q >= 0 {
                    if n <= 1 || m <= 1 {
                        0
                    } else {
                        let from_pos = (1u64 << (n - 1)) - 1;
                        let to_pos = (1u64 << (m - 1)) - 1;
                        (((q as u64) * to_pos + from_pos / 2) / from_pos) as i32
                    }
                } else {
                    let mag = q.unsigned_abs() as u64;
                    let scaled = if m >= n {
                        mag << (m - n)
                    } else {
                        let shift = (n - m) as u32;
                        let max_neg = 1u64 << (m - 1);
                        let r = (mag + (1u64 << (shift - 1))) >> shift;
                        if r > max_neg { max_neg } else { r }
                    };
                    0i32.wrapping_sub(scaled as i32)
                }
            }
            _ => self.code_u32(code as u32) as i32,
        }
    }

    /// Extract the code from a `u16` source word, convert it and place it in a `u16` destination word.
    ///
    /// Source padding is ignored; destination padding is zero. A destination with
    /// `U8` storage always fits in the low byte of the result.
    pub const fn word(&self, word: u16) -> u16 {
        self.word_u32(word as u32) as u16
    }

    /// Extract the code from a `u32` source word, convert it and place it in a `u32` destination word.
    ///
    /// Source padding is ignored; destination padding is zero. Destinations with
    /// `U8` or `U16` storage fit in the low 8 or 16 bits of the result.
    pub const fn word_u32(&self, word: u32) -> u32 {
        self.code_u32(word >> self.from.bit_shift()) << self.to.bit_shift()
    }

    /// Extract the signed code from an `i16` source word, convert it and place it in an `i16`
    /// destination word (`I8` destinations are sign-extended from the low byte when top-aligned).
    pub const fn word_i16(&self, word: i16) -> i16 {
        let raw = ((word as u16) as u32) >> self.from.bit_shift();
        let c = self.code_i32(raw as i32);
        let placed = ((c as u32) & mask_u32(self.to.code_bits())) << self.to.bit_shift();
        if matches!(self.to.storage(), ChannelType::I8)
            && self.to.code_bits() + self.to.bit_shift() == 8
        {
            (placed as u8 as i8) as i16
        } else {
            placed as u16 as i16
        }
    }

    /// Extract the signed code from an `i32` source word, convert it and place it in an `i32`
    /// destination word (`I8`/`I16` destinations are sign-extended when top-aligned in their container).
    pub const fn word_i32(&self, word: i32) -> i32 {
        let raw = (word as u32) >> self.from.bit_shift();
        let c = self.code_i32(raw as i32);
        let placed = ((c as u32) & mask_u32(self.to.code_bits())) << self.to.bit_shift();
        if matches!(self.to.storage(), ChannelType::I8)
            && self.to.code_bits() + self.to.bit_shift() == 8
        {
            (placed as u8 as i8) as i32
        } else if matches!(self.to.storage(), ChannelType::I16)
            && self.to.code_bits() + self.to.bit_shift() == 16
        {
            (placed as u16 as i16) as i32
        } else {
            placed as i32
        }
    }

    /// Map equal-length runs of `U8`/`U16` source words to destination words.
    ///
    /// Returns [`CodeMapError::UnsupportedStorage`] if either encoding is signed or uses `U32` storage
    /// (use [`map_words_u32`](Self::map_words_u32), [`map_words_i16`](Self::map_words_i16), or
    /// [`map_words_i32`](Self::map_words_i32)). The lengths are checked before any write.
    pub fn map_words(&self, source: &[u16], destination: &mut [u16]) -> Result<(), CodeMapError> {
        if !matches!(self.from.storage(), ChannelType::U8 | ChannelType::U16)
            || !matches!(self.to.storage(), ChannelType::U8 | ChannelType::U16)
        {
            return Err(CodeMapError::UnsupportedStorage);
        }
        if source.len() != destination.len() {
            return Err(CodeMapError::LengthMismatch {
                source: source.len(),
                destination: destination.len(),
            });
        }
        for (d, s) in destination.iter_mut().zip(source) {
            *d = self.word(*s);
        }
        Ok(())
    }

    /// Map equal-length runs of unsigned `U8`/`U16`/`U32` source words to destination words.
    ///
    /// Returns [`CodeMapError::UnsupportedStorage`] on signed encodings. The lengths are checked before any write.
    pub fn map_words_u32(
        &self,
        source: &[u32],
        destination: &mut [u32],
    ) -> Result<(), CodeMapError> {
        if self.from.is_signed() || self.to.is_signed() {
            return Err(CodeMapError::UnsupportedStorage);
        }
        if source.len() != destination.len() {
            return Err(CodeMapError::LengthMismatch {
                source: source.len(),
                destination: destination.len(),
            });
        }
        for (d, s) in destination.iter_mut().zip(source) {
            *d = self.word_u32(*s);
        }
        Ok(())
    }

    /// Map equal-length runs of signed `I8`/`I16` source words to destination words.
    ///
    /// Returns [`CodeMapError::UnsupportedStorage`] if either encoding is unsigned or uses `I32` storage.
    pub fn map_words_i16(
        &self,
        source: &[i16],
        destination: &mut [i16],
    ) -> Result<(), CodeMapError> {
        if !matches!(self.from.storage(), ChannelType::I8 | ChannelType::I16)
            || !matches!(self.to.storage(), ChannelType::I8 | ChannelType::I16)
        {
            return Err(CodeMapError::UnsupportedStorage);
        }
        if source.len() != destination.len() {
            return Err(CodeMapError::LengthMismatch {
                source: source.len(),
                destination: destination.len(),
            });
        }
        for (d, s) in destination.iter_mut().zip(source) {
            *d = self.word_i16(*s);
        }
        Ok(())
    }

    /// Map equal-length runs of signed `I8`/`I16`/`I32` source words to destination words.
    ///
    /// Returns [`CodeMapError::UnsupportedStorage`] on unsigned encodings. The lengths are checked before any write.
    pub fn map_words_i32(
        &self,
        source: &[i32],
        destination: &mut [i32],
    ) -> Result<(), CodeMapError> {
        if !self.from.is_signed() || !self.to.is_signed() {
            return Err(CodeMapError::UnsupportedStorage);
        }
        if source.len() != destination.len() {
            return Err(CodeMapError::LengthMismatch {
                source: source.len(),
                destination: destination.len(),
            });
        }
        for (d, s) in destination.iter_mut().zip(source) {
            *d = self.word_i32(*s);
        }
        Ok(())
    }
}

/// Native components cannot be widened into the requested canonical image.
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodesError {
    /// The destination is not a supported canonical U8/U16 layout
    /// (gray, gray-alpha, RGB/BGR, RGBA/BGRA/RGBX/BGRX, or CMYK/CMYKA).
    Destination {
        /// The destination descriptor.
        actual: PixelDescriptor,
    },
    /// The destination's signal range differs from the components' range.
    RangeMismatch {
        /// Range of the native components.
        source: SignalRange,
        /// Range declared by the destination.
        destination: SignalRange,
    },
    /// The number of components differs from the destination's active channels.
    ComponentCount {
        /// Destination active channels.
        expected: usize,
        /// Supplied components.
        actual: usize,
    },
    /// A component has more code bits than the destination; widening only.
    Narrowing {
        /// Component index.
        component: usize,
        /// Its code depth.
        bits: u8,
        /// The destination's storage depth.
        destination_bits: u8,
    },
    /// A component's encoding has no exact rule for its role and range.
    Map {
        /// Component index.
        component: usize,
        /// The refusal.
        error: CodeMapError,
    },
    /// The requested region extends beyond a component plane.
    Extent {
        /// Component index.
        component: usize,
        /// Region origin in plane coordinates.
        origin: (usize, usize),
        /// Region width and height.
        extent: (usize, usize),
        /// The plane's width and height.
        plane: (usize, usize),
    },
    /// A plane read failed after validation; an internal consistency error.
    Read(CodePlaneError),
}

#[cfg(feature = "planar")]
impl fmt::Display for CodesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Destination { actual } => write!(
                f,
                "canonical destination must be U8/U16 gray, gray-alpha, RGB/BGR, RGBA/BGRA/RGBX/BGRX or CMYK/CMYKA, got {actual:?}"
            ),
            Self::RangeMismatch {
                source,
                destination,
            } => write!(
                f,
                "components are {source} range, destination is {destination}"
            ),
            Self::ComponentCount { expected, actual } => {
                write!(
                    f,
                    "destination has {expected} channels, got {actual} components"
                )
            }
            Self::Narrowing {
                component,
                bits,
                destination_bits,
            } => write!(
                f,
                "component {component} has {bits} code bits; a {destination_bits}-bit destination would quantize"
            ),
            Self::Map { component, error } => write!(f, "component {component}: {error}"),
            Self::Extent {
                component,
                origin,
                extent,
                plane,
            } => write!(
                f,
                "component {component}: region {extent:?} at {origin:?} exceeds plane {plane:?}"
            ),
            Self::Read(error) => write!(f, "code plane read failed: {error}"),
        }
    }
}

#[cfg(feature = "planar")]
impl core::error::Error for CodesError {}

/// Codes converted per bounded batch; the stack buffer is 512 bytes.
#[cfg(feature = "planar")]
const BATCH: usize = 256;

/// Widen borrowed native components into a canonical strided image region.
///
/// `components` are in destination active channel order (gray; gray, alpha;
/// R, G, B; B, G, R; R, G, B, A; B, G, R, A; C, M, Y, K; or C, M, Y, K, A;
/// `RGBX`/`BGRX` padding slots take 3 color components and leave the 4th
/// padding channel untouched); the alpha component is always full scale. Each component's
/// codes are widened exactly with the rule for its role and `range`, which must
/// equal the destination's declared signal range; this never converts range.
/// `(x, y)` is the region origin in plane coordinates, so crops and row
/// partitions read the original planes. Components may use different code
/// depths and placements but must not exceed the destination's storage depth:
/// narrowing is a lossy choice made explicitly with [`CodeMap`].
///
/// Every check precedes the first write; row padding is untouched. The color
/// context already attached to `destination` is kept: components carry no color.
/// Execution reads each requested sample once and allocates nothing.
///
/// Returns the destination and the [`ExactDepth`] this widening established for
/// the written samples: the code depth (for mixed full-range color depths, the
/// smallest depth every component's grid nests in) and the alpha code depth.
/// The caller owns that evidence; it stops describing the samples as soon as
/// they are changed.
#[cfg(feature = "planar")]
pub fn to_canonical<'b>(
    components: &[CodePlane<'_>],
    range: SignalRange,
    x: usize,
    y: usize,
    mut destination: PixelSliceMut<'b>,
) -> Result<(PixelSliceMut<'b>, ExactDepth), At<CodesError>> {
    let descriptor = destination.descriptor();
    let storage_bits: u8 = match descriptor.channel_type() {
        ChannelType::U8 => 8,
        ChannelType::U16 => 16,
        _ => return Err(whereat::at!(CodesError::Destination { actual: descriptor })),
    };
    let channels = descriptor.channels();
    let (has_alpha, expected_components) = match (descriptor.layout(), descriptor.alpha()) {
        (
            ChannelLayout::Gray
            | ChannelLayout::Rgb
            | ChannelLayout::Bgr
            | ChannelLayout::Cmyk
            | ChannelLayout::CmykInverted,
            None,
        ) => (false, channels),
        (ChannelLayout::Rgba | ChannelLayout::Bgra, Some(AlphaMode::Undefined)) => (false, 3),
        (
            ChannelLayout::GrayAlpha
            | ChannelLayout::Rgba
            | ChannelLayout::Bgra
            | ChannelLayout::CmykAlpha
            | ChannelLayout::CmykInvertedAlpha,
            Some(_),
        ) => (true, channels),
        _ => return Err(whereat::at!(CodesError::Destination { actual: descriptor })),
    };
    if descriptor.signal_range != range {
        return Err(whereat::at!(CodesError::RangeMismatch {
            source: range,
            destination: descriptor.signal_range,
        }));
    }
    if components.len() != expected_components {
        return Err(whereat::at!(CodesError::ComponentCount {
            expected: expected_components,
            actual: components.len(),
        }));
    }
    let canonical = match SampleEncoding::new(descriptor.channel_type(), storage_bits, 0) {
        Ok(encoding) => encoding,
        Err(_) => return Err(whereat::at!(CodesError::Destination { actual: descriptor })),
    };
    let (width, height) = (destination.width() as usize, destination.rows() as usize);
    let mut maps = [None; 5];
    for (component, plane) in components.iter().enumerate() {
        let bits = plane.encoding().code_bits();
        if bits > storage_bits {
            return Err(whereat::at!(CodesError::Narrowing {
                component,
                bits,
                destination_bits: storage_bits,
            }));
        }
        let role = if has_alpha && component + 1 == expected_components {
            CodeRole::Alpha
        } else {
            CodeRole::Intensity
        };
        maps[component] = Some(
            CodeMap::new(plane.encoding(), canonical, role, range)
                .map_err(|error| whereat::at!(CodesError::Map { component, error }))?,
        );
        if x.checked_add(width).is_none_or(|end| end > plane.width())
            || y.checked_add(height).is_none_or(|end| end > plane.height())
        {
            return Err(whereat::at!(CodesError::Extent {
                component,
                origin: (x, y),
                extent: (width, height),
                plane: (plane.width(), plane.height()),
            }));
        }
    }

    let exact = established_depth(
        components,
        range,
        has_alpha,
        descriptor.channel_type(),
        storage_bits,
    )
    .ok_or_else(|| whereat::at!(CodesError::Destination { actual: descriptor }))?;

    let mut codes = [0u16; BATCH];
    for row in 0..height {
        let out = destination.row_mut(row as u32);
        for (component, (plane, map)) in components.iter().zip(maps).enumerate() {
            let Some(map) = map else { continue };
            let mut start = 0;
            while start < width {
                let count = (width - start).min(BATCH);
                let batch = &mut codes[..count];
                plane
                    .read_codes(x + start, y + row, batch)
                    .map_err(|e| whereat::at!(CodesError::Read(e)))?;
                let first = start * channels + component;
                if storage_bits == 8 {
                    for (i, code) in batch.iter().enumerate() {
                        out[first + i * channels] = map.code(*code) as u8;
                    }
                } else {
                    for (i, code) in batch.iter().enumerate() {
                        let at = (first + i * channels) * 2;
                        out[at..at + 2].copy_from_slice(&map.code(*code).to_ne_bytes());
                    }
                }
                start += count;
            }
        }
    }
    Ok((destination, exact))
}

/// Evidence established by exactly widening `components` into canonical storage.
#[cfg(feature = "planar")]
fn established_depth(
    components: &[CodePlane<'_>],
    range: SignalRange,
    has_alpha: bool,
    storage: ChannelType,
    storage_bits: u8,
) -> Option<ExactDepth> {
    let color_count = components.len() - usize::from(has_alpha);
    let narrow = range == SignalRange::Narrow;
    // Full-range grids nest in multiples of their depth: combine by lcm.
    let mut color: u32 = 1;
    for plane in &components[..color_count] {
        let bits = u32::from(plane.encoding().code_bits());
        color = if narrow {
            color.max(bits)
        } else {
            lcm(color, bits)
        };
    }
    let color = if color > u32::from(storage_bits) {
        storage_bits
    } else {
        color as u8
    };
    let alpha = if has_alpha {
        components[color_count].encoding().code_bits()
    } else {
        storage_bits
    };
    ExactDepth::new(storage, range, color, alpha).ok()
}

#[cfg(feature = "planar")]
fn lcm(a: u32, b: u32) -> u32 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}
