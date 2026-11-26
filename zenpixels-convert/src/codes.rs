//! Exact depth, placement, and range conversion of native integer codes.
//!
//! A *code* is the unsigned integer that a [`SampleEncoding`] extracts from a
//! storage word. Canonical `U16` and `U8` image samples are codes of the full
//! word width: a canonical `U16` white is `65535`, never a ten-bit `1023`.
//! Native codecs keep their codes in [`CodePlane`] views; this module converts
//! between those codes and canonical samples, or between two native encodings.
//!
//! | Role and range | Depth change from `N` to `M` bits | Inverse of a widening |
//! |---|---|---|
//! | Intensity (R′G′B′, gray, Y′), full range | Nearest `(q·max(M) + max(N)/2) / max(N)` | Exact: every code returns |
//! | Intensity or chroma, narrow range | `q << (M−N)`; narrowing rounds half up, clamped to `max(M)` | Exact |
//! | Alpha, any color range | Nearest full-scale, as full-range intensity | Exact |
//! | Chroma, full range | Refused: neutral and span have no exact relation between depths | — |
//!
//! `max(N) = 2^N − 1`; odd denominators make nearest rounding tie-free.
//! Narrow range needs at least 8 bits so its anchors (`16·2^(N−8)` black,
//! `128·2^(N−8)` neutral) are integers.

use core::fmt;
use whereat::At;
use zenpixels::sample::{CodePlane, CodePlaneError, ExactDepth, SampleEncoding};
use zenpixels::{ChannelOrder, PixelFormat, PixelViewMut, SampleType, SignalRange};

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
    /// Numerical mapping requires unsigned `U8`/`U16` words.
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
            Self::UnsupportedRange(range) => write!(f, "no code rule for signal range {range:?}"),
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

/// A checked conversion of one component from one [`SampleEncoding`] to another.
///
/// Construction selects the rule once from the role, signal range, and both
/// encodings; mapping performs integer arithmetic only and never allocates.
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
        if !matches!(from.storage(), SampleType::U8 | SampleType::U16)
            || !matches!(to.storage(), SampleType::U8 | SampleType::U16)
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
    #[inline]
    #[must_use]
    pub const fn input(&self) -> SampleEncoding {
        self.from
    }

    /// Encoding of the words this map writes.
    #[inline]
    #[must_use]
    pub const fn output(&self) -> SampleEncoding {
        self.to
    }

    /// Convert one source code (bits above its depth are ignored) to a destination code.
    #[inline]
    #[must_use]
    pub const fn map_code(&self, code: u16) -> u16 {
        let q = (code as u32) & ((1u32 << self.from.code_bits()) - 1);
        (match self.rule {
            Rule::Same => q,
            // max <= 65535: q * to_max + from_max / 2 < 2^32.
            Rule::Nearest { from_max, to_max } => (q * to_max + from_max / 2) / from_max,
            Rule::ShiftLeft(s) => q << s,
            Rule::RoundShiftRight { shift, max } => {
                let r = (q + (1 << (shift - 1))) >> shift;
                if r > max { max } else { r }
            }
        }) as u16
    }

    /// Extract the code from a source word, convert it, and place it in a destination word.
    ///
    /// Source padding is ignored; destination padding is zero.
    #[inline]
    #[must_use]
    pub const fn map_word(&self, word: u16) -> u16 {
        self.map_code(word >> self.from.bit_shift()) << self.to.bit_shift()
    }

    /// Map equal-length runs of source words to destination words.
    ///
    /// Lengths are checked before any write.
    pub fn map_words(&self, source: &[u16], destination: &mut [u16]) -> Result<(), CodeMapError> {
        if source.len() != destination.len() {
            return Err(CodeMapError::LengthMismatch {
                source: source.len(),
                destination: destination.len(),
            });
        }
        for (d, s) in destination.iter_mut().zip(source) {
            *d = self.map_word(*s);
        }
        Ok(())
    }
}

/// Native components cannot be widened into the requested canonical image.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodesError {
    /// The destination is not canonical full-packing `U8`/`U16` Gray, GrayAlpha, RGB, or RGBA.
    Destination {
        /// The destination spec.
        actual: PixelFormat,
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
                "canonical destination must be full-packing U8/U16 Gray, GrayAlpha, RGB, or RGBA, got {actual:?}"
            ),
            Self::RangeMismatch {
                source,
                destination,
            } => write!(
                f,
                "components are {source:?} range, destination is {destination:?}"
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

/// Unpack and widen borrowed native component [`CodePlane`]s into a canonical strided
/// [`PixelViewMut`] region starting at `source_origin = (x, y)` in each source plane.
///
/// `components` are in destination channel order (`Gray`; `Gray, Alpha`; `R, G, B`;
/// or `R, G, B, A`); alpha is always full scale. Every check precedes the first write,
/// row padding is untouched, and execution allocates nothing.
///
/// Returns the [`ExactDepth`] established by this exact widening.
pub fn unpack_code_planes_into(
    components: &[CodePlane<'_>],
    range: SignalRange,
    source_origin: (usize, usize),
    destination: &mut PixelViewMut<'_>,
) -> Result<ExactDepth, At<CodesError>> {
    let (x, y) = source_origin;
    let spec = destination.format().clone();
    if !spec.layout.packing.is_full() {
        return Err(whereat::at!(CodesError::Destination { actual: spec }));
    }
    let storage_bits: u8 = match spec.layout.sample {
        SampleType::U8 => 8,
        SampleType::U16 => 16,
        _ => return Err(whereat::at!(CodesError::Destination { actual: spec })),
    };
    let has_alpha = match spec.layout.order {
        ChannelOrder::Gray | ChannelOrder::Rgb => false,
        ChannelOrder::GrayAlpha | ChannelOrder::Rgba => true,
        _ => return Err(whereat::at!(CodesError::Destination { actual: spec })),
    };
    let dst_range = spec.color.range().unwrap_or(SignalRange::Full);
    if dst_range != range {
        return Err(whereat::at!(CodesError::RangeMismatch {
            source: range,
            destination: dst_range,
        }));
    }
    let channels = spec.layout.channels();
    if components.len() != channels {
        return Err(whereat::at!(CodesError::ComponentCount {
            expected: channels,
            actual: components.len(),
        }));
    }
    let canonical = match SampleEncoding::new(spec.layout.sample, storage_bits, 0) {
        Ok(encoding) => encoding,
        Err(_) => return Err(whereat::at!(CodesError::Destination { actual: spec })),
    };
    let (width, height) = (destination.width() as usize, destination.height() as usize);
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
        spec.layout.sample,
        storage_bits,
    )
    .ok_or_else(|| whereat::at!(CodesError::Destination { actual: spec }))?;

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
                        out[first + i * channels] = map.map_code(*code) as u8;
                    }
                } else {
                    for (i, code) in batch.iter().enumerate() {
                        let at = (first + i * channels) * 2;
                        out[at..at + 2].copy_from_slice(&map.map_code(*code).to_ne_bytes());
                    }
                }
                start += count;
            }
        }
    }
    Ok(exact)
}

fn established_depth(
    components: &[CodePlane<'_>],
    range: SignalRange,
    has_alpha: bool,
    storage: SampleType,
    storage_bits: u8,
) -> Option<ExactDepth> {
    let color_count = components.len() - usize::from(has_alpha);
    let narrow = range == SignalRange::Narrow;
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
