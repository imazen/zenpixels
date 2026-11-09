//! Exact depth, placement and range conversion of native integer codes.
//!
//! A *code* is the unsigned integer that a [`SampleEncoding`] extracts from a
//! storage word. Ordinary (canonical) `U16` and `U8` image samples are codes of
//! the full word width: a canonical `U16` white is 65535, never a ten-bit 1023.
//! Native codecs keep their codes in [`CodePlane`] views; this module converts
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
use whereat::At;
use whereat::ResultAtExt as _;
use zenpixels::{
    ChannelLayout, ChannelType, PixelDescriptor, PixelSliceMut, SignalRange,
    sample::{CodePlane, CodePlaneError, ExactDepth, SampleEncoding},
};

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
    /// Numerical mapping requires unsigned U8/U16 words.
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
            Self::UnsupportedStorage => f.write_str("code mapping requires unsigned U8/U16 words"),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Same,
    Nearest { from_max: u32, to_max: u32 },
    ShiftLeft(u32),
    RoundShiftRight { shift: u32, max: u32 },
}

/// A checked conversion of one component from one sample encoding to another.
///
/// Construction selects the rule once from the role, signal range and both
/// encodings; mapping performs integer arithmetic only and never allocates.
/// See the [module documentation](self) for the rules and their guarantees.
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
        if !matches!(from.storage(), ChannelType::U8 | ChannelType::U16)
            || !matches!(to.storage(), ChannelType::U8 | ChannelType::U16)
        {
            return Err(CodeMapError::UnsupportedStorage);
        }
        let (n, m) = (from.code_bits(), to.code_bits());
        let full_scale = match (role, range) {
            (CodeRole::Alpha, _) => true,
            (_, SignalRange::Full) => true,
            (_, SignalRange::Narrow) => false,
            (_, other) => return Err(CodeMapError::UnsupportedRange(other)),
        };
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
                from_max: (1u32 << n) - 1,
                to_max: (1u32 << m) - 1,
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
                max: (1u32 << m) - 1,
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

    /// Convert one source code (bits above its depth are ignored) to a destination code.
    pub const fn code(&self, code: u16) -> u16 {
        let q = (code as u32) & ((1u32 << self.from.code_bits()) - 1);
        (match self.rule {
            Rule::Same => q,
            // max ≤ 65535: q·to_max + from_max/2 < 2^32.
            Rule::Nearest { from_max, to_max } => (q * to_max + from_max / 2) / from_max,
            Rule::ShiftLeft(s) => q << s,
            Rule::RoundShiftRight { shift, max } => {
                let r = (q + (1 << (shift - 1))) >> shift;
                if r > max { max } else { r }
            }
        }) as u16
    }

    /// Extract the code from a source word, convert it and place it in a destination word.
    ///
    /// Source padding is ignored; destination padding is zero. A destination with
    /// `U8` storage always fits in the low byte of the result.
    pub const fn word(&self, word: u16) -> u16 {
        self.code(word >> self.from.bit_shift()) << self.to.bit_shift()
    }

    /// Map equal-length runs of source words to destination words.
    ///
    /// Use this for contiguous plane rows or interleaved rows whose components share
    /// one encoding. The lengths are checked before any write.
    pub fn map_words(&self, source: &[u16], destination: &mut [u16]) -> Result<(), CodeMapError> {
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
}

/// Native components cannot be widened into the requested canonical image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodesError {
    /// The destination is not canonical U8/U16 gray, gray-alpha, RGB or RGBA.
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
    /// The number of components differs from the destination's channels.
    ComponentCount {
        /// Destination channels.
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

impl fmt::Display for CodesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Destination { actual } => write!(
                f,
                "canonical destination must be U8/U16 gray, gray-alpha, RGB or RGBA, got {actual:?}"
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

impl core::error::Error for CodesError {}

/// Codes converted per bounded batch; the stack buffer is 512 bytes.
const BATCH: usize = 256;

/// Widen borrowed native components into a canonical strided image region.
///
/// `components` are in destination channel order (gray; gray, alpha; R, G, B;
/// or R, G, B, A); the alpha component is always full scale. Each component's
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
    let has_alpha = match descriptor.layout() {
        ChannelLayout::Gray | ChannelLayout::Rgb => false,
        ChannelLayout::GrayAlpha | ChannelLayout::Rgba => true,
        _ => return Err(whereat::at!(CodesError::Destination { actual: descriptor })),
    };
    if descriptor.signal_range != range {
        return Err(whereat::at!(CodesError::RangeMismatch {
            source: range,
            destination: descriptor.signal_range,
        }));
    }
    let channels = descriptor.channels();
    if components.len() != channels {
        return Err(whereat::at!(CodesError::ComponentCount {
            expected: channels,
            actual: components.len(),
        }));
    }
    let canonical = match SampleEncoding::new(descriptor.channel_type(), storage_bits, 0) {
        Ok(encoding) => encoding,
        Err(_) => return Err(whereat::at!(CodesError::Destination { actual: descriptor })),
    };
    let (width, height) = (destination.width() as usize, destination.rows() as usize);
    let mut maps = [None; 4];
    for (component, plane) in components.iter().enumerate() {
        let bits = plane.encoding().code_bits();
        if bits > storage_bits {
            return Err(whereat::at!(CodesError::Narrowing {
                component,
                bits,
                destination_bits: storage_bits,
            }));
        }
        let role = if has_alpha && component + 1 == channels {
            CodeRole::Alpha
        } else {
            CodeRole::Intensity
        };
        maps[component] = Some(
            CodeMap::new(plane.encoding(), canonical, role, range)
                .map_err(|error| CodesError::Map { component, error })
                .map_err(whereat::At::wrap)
                .at_crate(crate::at_crate_info())
                .at()?,
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
    .ok_or(CodesError::Destination { actual: descriptor })
    .map_err(whereat::At::wrap)
    .at_crate(crate::at_crate_info())
    .at()?;

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
                    .map_err(CodesError::Read)
                    .map_err(whereat::At::wrap)
                    .at_crate(crate::at_crate_info())
                    .at()?;
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

fn lcm(a: u32, b: u32) -> u32 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// Canonicalize unsigned native RGB/gray words into U16 samples.
///
/// Layout and dimensions must match. The destination descriptor supplies color
/// meaning: this operation does no color conversion. Signed or >16-bit payloads
/// refuse before any writes. Alpha uses full range regardless of color range.
/// Native sample padding is ignored; destination row padding is untouched.
pub fn native_to_u16(
    source: zenpixels::sample::NativeCodeSlice<'_>,
    destination: &mut PixelSliceMut<'_>,
    range: SignalRange,
) -> Result<(), NativeConversionError> {
    let d = destination.descriptor();
    if d.signal_range != range
        || d.channel_type() != ChannelType::U16
        || d.layout() != source.layout()
        || destination.width() != source.width()
        || destination.rows() != source.height()
    {
        return Err(NativeConversionError::Destination);
    }
    let target = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
    let mut maps = [None; 4];
    let channels = source.encodings().len();
    for (i, e) in source.encodings().iter().enumerate() {
        if !matches!(
            e.storage(),
            ChannelType::U8 | ChannelType::U16 | ChannelType::U32 | ChannelType::U64
        ) || e.code_bits() > 16
        {
            return Err(NativeConversionError::UnsupportedEncoding);
        }
        let encoding = SampleEncoding::new(ChannelType::U16, e.code_bits(), 0).unwrap();
        let role = if d.has_alpha() && i == channels - 1 {
            CodeRole::Alpha
        } else {
            CodeRole::Intensity
        };
        maps[i] = Some(
            CodeMap::new(encoding, target, role, d.signal_range)
                .map_err(NativeConversionError::Map)?,
        );
    }
    if source.width() == 0 {
        return Ok(());
    }
    for y in 0..source.height() {
        let row = destination.row_mut(y);
        for x in 0..source.width() {
            for (c, map) in maps[..channels].iter().enumerate() {
                let code = source.code(x, y, c).unwrap() as u16;
                let out = map.unwrap().code(code).to_ne_bytes();
                let offset = (x as usize * channels + c) * 2;
                row[offset..offset + 2].copy_from_slice(&out);
            }
        }
    }
    Ok(())
}

/// Native canonicalization refuses unsupported interpretation before writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeConversionError {
    /// Native payload is narrower than U16 and policy forbids reducing depth.
    DepthReduction,
    /// Destination must be matching U16 geometry and layout.
    Destination,
    /// Signed or wider-than-U16 payload needs a separately qualified conversion.
    UnsupportedEncoding,
    /// Range/depth mapping is unsupported.
    Map(CodeMapError),
}
impl core::fmt::Display for NativeConversionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "native canonicalization refused: {self:?}")
    }
}
impl core::error::Error for NativeConversionError {}

/// Treatment of bits outside native payloads when encoding canonical samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativePadding {
    /// Clear all padding bits in each written sample word.
    Zero,
    /// Keep the destination's existing padding bits. Row padding is always kept.
    Preserve,
}

/// Encode canonical U16 samples into separate native integer component words.
///
/// The source supplies geometry, layout and signal range. Signed and >16-bit
/// payloads refuse. Narrowing requires the explicit round depth policy;
/// all validation precedes writes. The destination's byte stride can include
/// arbitrary row padding. This does not pack RGB565 or other shared-word formats.
pub fn u16_to_native(
    source: zenpixels::PixelSlice<'_>,
    destination: &mut [u8],
    stride: usize,
    encodings: &[zenpixels::sample::SampleEncoding],
    padding: NativePadding,
    depth: zenpixels::DepthPolicy,
) -> Result<(), NativeConversionError> {
    let d = source.descriptor();
    if d.channel_type() != ChannelType::U16 {
        return Err(NativeConversionError::Destination);
    }
    zenpixels::sample::NativeCodeSlice::new(
        destination,
        source.width(),
        source.rows(),
        stride,
        d.layout(),
        encodings,
    )
    .map_err(|_| NativeConversionError::Destination)?;
    let mut maps = [None; 4];
    let input = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
    for (i, e) in encodings.iter().enumerate() {
        if !matches!(
            e.storage(),
            ChannelType::U8 | ChannelType::U16 | ChannelType::U32 | ChannelType::U64
        ) || e.code_bits() > 16
        {
            return Err(NativeConversionError::UnsupportedEncoding);
        }
        if e.code_bits() < 16 && depth == zenpixels::DepthPolicy::Forbid {
            return Err(NativeConversionError::DepthReduction);
        }
        // Only the established round mapping is qualified for arbitrary native
        // payloads; truncate is a different contract and refuses for now.
        if e.code_bits() < 16 && depth != zenpixels::DepthPolicy::Round {
            return Err(NativeConversionError::UnsupportedEncoding);
        }
        let target = SampleEncoding::new(ChannelType::U16, e.code_bits(), 0).unwrap();
        let role = if d.has_alpha() && i == encodings.len() - 1 {
            CodeRole::Alpha
        } else {
            CodeRole::Intensity
        };
        maps[i] = Some(
            CodeMap::new(input, target, role, d.signal_range)
                .map_err(NativeConversionError::Map)?,
        );
    }
    if source.width() == 0 {
        return Ok(());
    }
    let size = encodings[0].storage().byte_size();
    for y in 0..source.rows() {
        let src = source.row(y);
        for x in 0..source.width() as usize {
            for (c, e) in encodings.iter().enumerate() {
                let i = x * encodings.len() + c;
                let code = u16::from_ne_bytes(src[i * 2..i * 2 + 2].try_into().unwrap());
                let value = u64::from(maps[c].unwrap().code(code)) << e.bit_shift();
                let mask = ((1u64 << e.code_bits()) - 1) << e.bit_shift();
                let start = y as usize * stride + i * size;
                let bytes = &mut destination[start..start + size];
                let old = match size {
                    1 => u64::from(bytes[0]),
                    2 => u64::from(u16::from_ne_bytes(bytes.try_into().unwrap())),
                    4 => u64::from(u32::from_ne_bytes(bytes.try_into().unwrap())),
                    8 => u64::from_ne_bytes(bytes.try_into().unwrap()),
                    _ => unreachable!(),
                };
                let word = match padding {
                    NativePadding::Zero => value,
                    NativePadding::Preserve => (old & !mask) | value,
                };
                match size {
                    1 => bytes[0] = word as u8,
                    2 => bytes.copy_from_slice(&(word as u16).to_ne_bytes()),
                    4 => bytes.copy_from_slice(&(word as u32).to_ne_bytes()),
                    8 => bytes.copy_from_slice(&word.to_ne_bytes()),
                    _ => unreachable!(),
                }
            }
        }
    }
    Ok(())
}
