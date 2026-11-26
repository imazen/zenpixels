//! Pure memory layout and bit-packing geometry for pixel storage.
//!
//! Layout describes **how bytes and bits sit in memory** without making any
//! claim about color space, transfer curve, or primaries. Pair a
//! [`SampleLayout`] with a [`ColorSpace`](crate::ColorSpace) in an
//! [`PixelFormat`](crate::PixelFormat) to describe real pixels.

use core::fmt;

/// Native numeric type of each stored channel sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// Unsigned 8-bit integer (`0..=255`).
    U8,
    /// Signed 8-bit integer (`-128..=127`).
    I8,
    /// Unsigned 16-bit integer (`0..=65535`).
    U16,
    /// Signed 16-bit integer (`-32768..=32767`).
    I16,
    /// IEEE 754 `binary16` half-precision float ([`F16Sample`](crate::F16Sample)).
    F16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// IEEE 754 `binary32` single-precision float (`f32`).
    F32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 `binary64` double-precision float (`f64`).
    F64,
}

impl SampleType {
    /// Size of one sample in bytes.
    #[inline]
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Required byte alignment for a slice of samples of this type on this target.
    #[inline]
    #[must_use]
    pub const fn min_alignment(self) -> usize {
        match self {
            Self::U8 | Self::I8 => core::mem::align_of::<u8>(),
            Self::U16 | Self::I16 | Self::F16 => core::mem::align_of::<u16>(),
            Self::U32 | Self::I32 => core::mem::align_of::<u32>(),
            Self::F32 => core::mem::align_of::<f32>(),
            Self::U64 | Self::I64 => core::mem::align_of::<u64>(),
            Self::F64 => core::mem::align_of::<f64>(),
        }
    }

    /// Bit width of one sample.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        (self.byte_size() * 8) as u8
    }

    /// Whether this sample type is IEEE floating point (`F16`, `F32`, or `F64`).
    #[inline]
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Whether this sample type is an integer type (signed or unsigned).
    #[inline]
    #[must_use]
    pub const fn is_integer(self) -> bool {
        !self.is_float()
    }

    /// Whether this sample type is signed (integers `I*` or floats `F*`).
    #[inline]
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::F16 | Self::I32 | Self::F32 | Self::I64 | Self::F64
        )
    }

    /// Whether this sample type is one of the 4 standard processing types (`U8`, `U16`, `F16`, `F32`).
    #[inline]
    #[must_use]
    pub const fn is_standard_color(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::F16 | Self::F32)
    }
}

/// Bit-level packing of integer codes inside storage words.
///
/// Ordinary full-width samples use [`BitPacking::Full`]. Codec-native 10/12-bit
/// codes inside `U16` words (such as P010 or HEIF/AVIF 10-bit planes) and
/// bitfield words such as `RGB565` or `RGB10A1` are first-class storage
/// descriptions rather than a separate buffer universe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BitPacking {
    /// All bits of each [`SampleType`] word hold the channel value (`0..=255` in `U8`, `0..=65535` in `U16`).
    #[default]
    Full,
    /// Right-aligned `bits`-wide unsigned code in each `U8`/`U16`/`U32` sample (unused high bits ignored on read).
    Lsb {
        /// Significant bit depth (`1..=8` for `U8`, `1..=16` for `U16`, `1..=31` for `U32`).
        bits: u8,
    },
    /// Left-aligned `bits`-wide unsigned code in each `U8`/`U16`/`U32` sample (e.g., P010: 10-bit code shifted left by 6).
    Msb {
        /// Significant bit depth (`1..=8` for `U8`, `1..=16` for `U16`, `1..=31` for `U32`).
        bits: u8,
    },
    /// Arbitrary bit-shifted `bits`-wide unsigned code sitting at bit offset `shift` in each
    /// `U8`/`U16`/`U32` sample (`bits >= 1`, `bits + shift <= sample.bits()`).
    Shifted {
        /// Significant bit depth (`1..=8` for `U8`, `1..=16` for `U16`, `1..=31` for `U32`).
        bits: u8,
        /// Number of low padding bits below the code.
        shift: u8,
    },
    /// Multiple channels bit-packed into a single [`SampleType`] word (e.g. `RGB565` in `U16`, `RGB10A1` in `U32`).
    Bitfields {
        /// Bit widths per channel (0 for unused slots).
        bits: [u8; 4],
        /// Right-shift from bit 0 for each channel.
        shifts: [u8; 4],
    },
}

impl BitPacking {
    /// P010 / 10-bit MSB-aligned code in `U16` (code sits in bits `6..16`).
    pub const P010: Self = Self::Msb { bits: 10 };
    /// 10-bit LSB-aligned code in `U16` (code sits in bits `0..10`).
    pub const U10_LSB: Self = Self::Lsb { bits: 10 };
    /// 12-bit LSB-aligned code in `U16` (code sits in bits `0..12`).
    pub const U12_LSB: Self = Self::Lsb { bits: 12 };
    /// 16-bit `RGB565`: R in `11..16`, G in `5..11`, B in `0..5`.
    pub const RGB565: Self = Self::Bitfields {
        bits: [5, 6, 5, 0],
        shifts: [11, 5, 0, 0],
    };
    /// 32-bit `RGB10A2`: R in `0..10`, G in `10..20`, B in `20..30`, A in `30..32`.
    pub const RGB10A2: Self = Self::Bitfields {
        bits: [10, 10, 10, 2],
        shifts: [0, 10, 20, 30],
    };

    /// Whether every bit of every sample word is a full-width value.
    #[inline]
    #[must_use]
    pub const fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Whether multiple channels share one storage word (`Bitfields`).
    #[inline]
    #[must_use]
    pub const fn is_packed_word(self) -> bool {
        matches!(self, Self::Bitfields { .. })
    }

    /// Validate that this packing fits inside `sample` and matches `channels`.
    pub const fn validate(self, sample: SampleType, channels: usize) -> Result<(), LayoutError> {
        let word_bits = sample.bits();
        match self {
            Self::Full => Ok(()),
            Self::Lsb { bits } | Self::Msb { bits } => {
                if !matches!(sample, SampleType::U8 | SampleType::U16 | SampleType::U32) {
                    return Err(LayoutError::UnsupportedBitPacking);
                }
                let max_bits = if matches!(sample, SampleType::U32) {
                    31
                } else {
                    word_bits
                };
                if bits == 0 || bits > max_bits {
                    return Err(LayoutError::InvalidBitDepth);
                }
                Ok(())
            }
            Self::Shifted { bits, shift } => {
                if !matches!(sample, SampleType::U8 | SampleType::U16 | SampleType::U32) {
                    return Err(LayoutError::UnsupportedBitPacking);
                }
                let max_bits = if matches!(sample, SampleType::U32) {
                    31
                } else {
                    word_bits
                };
                if bits == 0 || bits > max_bits || (bits as u16) + (shift as u16) > word_bits as u16
                {
                    return Err(LayoutError::InvalidBitDepth);
                }
                Ok(())
            }
            Self::Bitfields { bits, shifts } => {
                if !matches!(sample, SampleType::U16 | SampleType::U32) {
                    return Err(LayoutError::UnsupportedBitPacking);
                }
                if channels == 0 || channels > 4 {
                    return Err(LayoutError::InvalidChannelCount);
                }
                let mut used_mask: u32 = 0;
                let mut i = 0;
                while i < channels {
                    let b = bits[i];
                    let s = shifts[i];
                    if b == 0 || (b as u16) + (s as u16) > (word_bits as u16) {
                        return Err(LayoutError::InvalidBitDepth);
                    }
                    let field_mask = if b >= 32 {
                        u32::MAX
                    } else {
                        ((1u32 << b) - 1) << s
                    };
                    if (used_mask & field_mask) != 0 {
                        return Err(LayoutError::UnsupportedBitPacking);
                    }
                    used_mask |= field_mask;
                    i += 1;
                }
                while i < 4 {
                    if bits[i] != 0 || shifts[i] != 0 {
                        return Err(LayoutError::InvalidBitDepth);
                    }
                    i += 1;
                }
                Ok(())
            }
        }
    }

    /// Extract the unsigned integer code for `component` from a `u32`-widened word.
    #[inline]
    #[must_use]
    pub const fn extract_code(self, sample: SampleType, word: u32, component: usize) -> u32 {
        match self {
            Self::Full => word,
            Self::Lsb { bits } => {
                if bits == 0 {
                    return 0;
                }
                let mask = if bits >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                };
                word & mask
            }
            Self::Msb { bits } => {
                if bits == 0 {
                    return 0;
                }
                let shift = sample.bits().saturating_sub(bits);
                if shift >= 32 {
                    return 0;
                }
                let mask = if bits >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                };
                (word >> shift) & mask
            }
            Self::Shifted { bits, shift } => {
                if bits == 0 || shift >= 32 {
                    return 0;
                }
                let mask = if bits >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                };
                (word >> shift) & mask
            }
            Self::Bitfields { bits, shifts } => {
                if component >= 4 {
                    return 0;
                }
                let b = bits[component];
                let s = shifts[component];
                if b == 0 || s >= 32 {
                    return 0;
                }
                let mask = if b >= 32 { u32::MAX } else { (1u32 << b) - 1 };
                (word >> s) & mask
            }
        }
    }

    /// Pack an unsigned integer `code` into its bit position for `component`.
    #[inline]
    #[must_use]
    pub const fn pack_code(self, sample: SampleType, component: usize, code: u32) -> u32 {
        let clamped = {
            let m = self.max_code(sample, component);
            if code > m { m } else { code }
        };
        match self {
            Self::Full | Self::Lsb { .. } => clamped,
            Self::Msb { bits } => {
                let shift = sample.bits().saturating_sub(bits);
                if shift >= 32 { 0 } else { clamped << shift }
            }
            Self::Shifted { shift, .. } => {
                if shift >= 32 {
                    0
                } else {
                    clamped << shift
                }
            }
            Self::Bitfields { shifts, .. } => {
                if component >= 4 || shifts[component] >= 32 {
                    0
                } else {
                    clamped << shifts[component]
                }
            }
        }
    }

    /// Maximum representable unsigned code for `component` under this packing.
    #[inline]
    #[must_use]
    pub const fn max_code(self, sample: SampleType, component: usize) -> u32 {
        let b = match self {
            Self::Full => sample.bits(),
            Self::Lsb { bits } | Self::Msb { bits } | Self::Shifted { bits, .. } => bits,
            Self::Bitfields { bits, .. } => {
                if component < 4 {
                    bits[component]
                } else {
                    0
                }
            }
        };
        if b == 0 {
            0
        } else if b >= 32 {
            u32::MAX
        } else {
            (1u32 << b) - 1
        }
    }
}

/// Memory order of interleaved channels within a single pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChannelOrder {
    /// Single-channel grayscale (`[Y]`).
    Gray,
    /// Two-channel grayscale + alpha (`[Y, A]`).
    GrayAlpha,
    /// Three-channel red-green-blue (`[R, G, B]`).
    Rgb,
    /// Three-channel blue-green-red (`[B, G, R]`).
    Bgr,
    /// Four-channel red-green-blue-alpha (`[R, G, B, A]`).
    Rgba,
    /// Four-channel blue-green-red-alpha (`[B, G, R, A]`).
    Bgra,
    /// Four-channel red-green-blue + unused padding (`[R, G, B, X]`).
    Rgbx,
    /// Four-channel blue-green-red + unused padding (`[B, G, R, X]`).
    Bgrx,
    /// Three-channel Oklab (`[L, a, b]`).
    Oklab,
    /// Four-channel Oklab + alpha (`[L, a, b, A]`).
    OklabA,
    /// Four-channel CMYK (`[C, M, Y, K]`, `0` = no ink).
    Cmyk,
    /// Four-channel inverted CMYK (`[C, M, Y, K]`, `max` = no ink, Adobe JPEG/JXL).
    CmykInverted,
    /// Five-channel CMYK + alpha (`[C, M, Y, K, A]`, `0` = no ink, e.g. TIFF CMYKA).
    Cmyka,
    /// Five-channel inverted CMYK + alpha (`[C, M, Y, K, A]`, `max` = no ink).
    CmykaInverted,
    /// Three-channel CIE XYZ (`[X, Y, Z]`, e.g. TIFF LogLuv).
    Xyz,
    /// Four-channel CIE XYZ + alpha (`[X, Y, Z, A]`).
    XyzA,
    /// Three-channel perceptual/cylindrical space (`Oklch`, `Lab`, `Lch`, `Jzazbz`, `HSL`, `HSV`).
    Perceptual3,
    /// Four-channel perceptual/cylindrical space + alpha.
    Perceptual4,
    /// Generic `N`-channel untagged layout (`1..=16`).
    Generic(u8),
}

impl ChannelOrder {
    /// Number of stored channels per pixel.
    #[inline]
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb | Self::Bgr | Self::Oklab | Self::Xyz | Self::Perceptual3 => 3,
            Self::Rgba
            | Self::Bgra
            | Self::Rgbx
            | Self::Bgrx
            | Self::OklabA
            | Self::Cmyk
            | Self::CmykInverted
            | Self::XyzA
            | Self::Perceptual4 => 4,
            Self::Cmyka | Self::CmykaInverted => 5,
            Self::Generic(n) => n as usize,
        }
    }

    /// Whether this layout has a dedicated alpha channel slot (not padding).
    #[inline]
    #[must_use]
    pub const fn has_alpha_slot(self) -> bool {
        matches!(
            self,
            Self::GrayAlpha
                | Self::Rgba
                | Self::Bgra
                | Self::OklabA
                | Self::Cmyka
                | Self::CmykaInverted
                | Self::XyzA
                | Self::Perceptual4
        )
    }

    /// Whether this layout has an unused padding channel slot (`Rgbx` / `Bgrx`).
    #[inline]
    #[must_use]
    pub const fn has_padding_slot(self) -> bool {
        matches!(self, Self::Rgbx | Self::Bgrx)
    }

    /// Whether this layout stores blue before red (`Bgr`, `Bgra`, `Bgrx`).
    #[inline]
    #[must_use]
    pub const fn is_bgr(self) -> bool {
        matches!(self, Self::Bgr | Self::Bgra | Self::Bgrx)
    }

    /// Whether this layout is an RGB-family pixel arrangement.
    #[inline]
    #[must_use]
    pub const fn is_rgb_family(self) -> bool {
        matches!(
            self,
            Self::Rgb | Self::Bgr | Self::Rgba | Self::Bgra | Self::Rgbx | Self::Bgrx
        )
    }

    /// Whether this layout is grayscale (`Gray` or `GrayAlpha`).
    #[inline]
    #[must_use]
    pub const fn is_gray_family(self) -> bool {
        matches!(self, Self::Gray | Self::GrayAlpha)
    }

    /// Whether this layout is Oklab (`Oklab` or `OklabA`).
    #[inline]
    #[must_use]
    pub const fn is_oklab_family(self) -> bool {
        matches!(self, Self::Oklab | Self::OklabA)
    }

    /// Whether this layout is CIE XYZ (`Xyz` or `XyzA`).
    #[inline]
    #[must_use]
    pub const fn is_xyz_family(self) -> bool {
        matches!(self, Self::Xyz | Self::XyzA)
    }

    /// Whether this layout is CMYK (`Cmyk`, `CmykInverted`, `Cmyka`, `CmykaInverted`).
    #[inline]
    #[must_use]
    pub const fn is_cmyk_family(self) -> bool {
        matches!(
            self,
            Self::Cmyk | Self::CmykInverted | Self::Cmyka | Self::CmykaInverted
        )
    }
}

/// Complete interleaved storage description for a single plane of pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SampleLayout {
    /// Underlying numeric sample storage type.
    pub sample: SampleType,
    /// Interleaved channel arrangement.
    pub order: ChannelOrder,
    /// Bit-level packing inside each storage word.
    pub packing: BitPacking,
}

impl SampleLayout {
    /// Packed 8-bit RGB (`3 B/px`).
    pub const RGB8: Self = Self::new(SampleType::U8, ChannelOrder::Rgb);
    /// Packed 8-bit RGBA (`4 B/px`).
    pub const RGBA8: Self = Self::new(SampleType::U8, ChannelOrder::Rgba);
    /// Packed 8-bit BGR (`3 B/px`).
    pub const BGR8: Self = Self::new(SampleType::U8, ChannelOrder::Bgr);
    /// Packed 8-bit BGRA (`4 B/px`).
    pub const BGRA8: Self = Self::new(SampleType::U8, ChannelOrder::Bgra);
    /// Packed 8-bit RGBx (`4 B/px`, 4th byte padding).
    pub const RGBX8: Self = Self::new(SampleType::U8, ChannelOrder::Rgbx);
    /// Packed 8-bit BGRx (`4 B/px`, 4th byte padding).
    pub const BGRX8: Self = Self::new(SampleType::U8, ChannelOrder::Bgrx);
    /// Packed 8-bit grayscale (`1 B/px`).
    pub const GRAY8: Self = Self::new(SampleType::U8, ChannelOrder::Gray);
    /// Packed 8-bit grayscale + alpha (`2 B/px`).
    pub const GRAYA8: Self = Self::new(SampleType::U8, ChannelOrder::GrayAlpha);

    /// Packed 16-bit RGB (`6 B/px`).
    pub const RGB16: Self = Self::new(SampleType::U16, ChannelOrder::Rgb);
    /// Packed 16-bit RGBA (`8 B/px`).
    pub const RGBA16: Self = Self::new(SampleType::U16, ChannelOrder::Rgba);
    /// Packed 16-bit grayscale (`2 B/px`).
    pub const GRAY16: Self = Self::new(SampleType::U16, ChannelOrder::Gray);
    /// Packed 16-bit grayscale + alpha (`4 B/px`).
    pub const GRAYA16: Self = Self::new(SampleType::U16, ChannelOrder::GrayAlpha);

    /// Packed `f16` RGB (`6 B/px`).
    pub const RGBF16: Self = Self::new(SampleType::F16, ChannelOrder::Rgb);
    /// Packed `f16` RGBA (`8 B/px`).
    pub const RGBAF16: Self = Self::new(SampleType::F16, ChannelOrder::Rgba);
    /// Packed `f16` grayscale (`2 B/px`).
    pub const GRAYF16: Self = Self::new(SampleType::F16, ChannelOrder::Gray);
    /// Packed `f16` grayscale + alpha (`4 B/px`).
    pub const GRAYAF16: Self = Self::new(SampleType::F16, ChannelOrder::GrayAlpha);

    /// Packed `f32` RGB (`12 B/px`).
    pub const RGBF32: Self = Self::new(SampleType::F32, ChannelOrder::Rgb);
    /// Packed `f32` RGBA (`16 B/px`).
    pub const RGBAF32: Self = Self::new(SampleType::F32, ChannelOrder::Rgba);
    /// Packed `f32` grayscale (`4 B/px`).
    pub const GRAYF32: Self = Self::new(SampleType::F32, ChannelOrder::Gray);
    /// Packed `f32` grayscale + alpha (`8 B/px`).
    pub const GRAYAF32: Self = Self::new(SampleType::F32, ChannelOrder::GrayAlpha);

    /// Packed `f32` CIE XYZ (`12 B/px`).
    pub const XYZF32: Self = Self::new(SampleType::F32, ChannelOrder::Xyz);
    /// Packed `f32` CIE XYZ + alpha (`16 B/px`).
    pub const XYZAF32: Self = Self::new(SampleType::F32, ChannelOrder::XyzA);

    /// Packed `f32` Oklab (`12 B/px`).
    pub const OKLABF32: Self = Self::new(SampleType::F32, ChannelOrder::Oklab);
    /// Packed `f32` Oklab + alpha (`16 B/px`).
    pub const OKLABAF32: Self = Self::new(SampleType::F32, ChannelOrder::OklabA);

    /// Packed 8-bit CMYK (`4 B/px`, `0` = no ink).
    pub const CMYK8: Self = Self::new(SampleType::U8, ChannelOrder::Cmyk);
    /// Packed 8-bit inverted CMYK (`4 B/px`, `255` = no ink).
    pub const CMYK8_INVERTED: Self = Self::new(SampleType::U8, ChannelOrder::CmykInverted);
    /// Packed 8-bit CMYK + alpha (`5 B/px`, `0` = no ink).
    pub const CMYKA8: Self = Self::new(SampleType::U8, ChannelOrder::Cmyka);
    /// Packed 8-bit inverted CMYK + alpha (`5 B/px`, `255` = no ink).
    pub const CMYKA8_INVERTED: Self = Self::new(SampleType::U8, ChannelOrder::CmykaInverted);
    /// Packed 16-bit CMYK (`8 B/px`).
    pub const CMYK16: Self = Self::new(SampleType::U16, ChannelOrder::Cmyk);
    /// Packed `f32` CMYK (`16 B/px`).
    pub const CMYKF32: Self = Self::new(SampleType::F32, ChannelOrder::Cmyk);

    /// Packed 16-bit `RGB565` bitfield word (`2 B/px`).
    pub const RGB565: Self = Self {
        sample: SampleType::U16,
        order: ChannelOrder::Rgb,
        packing: BitPacking::RGB565,
    };

    /// Packed 32-bit `RGB10A2` bitfield word (`4 B/px`).
    pub const RGB10A2: Self = Self {
        sample: SampleType::U32,
        order: ChannelOrder::Rgba,
        packing: BitPacking::RGB10A2,
    };

    /// Construct a full-width interleaved sample layout.
    #[inline]
    #[must_use]
    pub const fn new(sample: SampleType, order: ChannelOrder) -> Self {
        Self {
            sample,
            order,
            packing: BitPacking::Full,
        }
    }

    /// Attach a bit-packing description (e.g. [`BitPacking::P010`], [`BitPacking::RGB565`]).
    #[inline]
    #[must_use]
    pub const fn with_packing(mut self, packing: BitPacking) -> Self {
        self.packing = packing;
        self
    }

    /// Number of channels per pixel.
    #[inline]
    #[must_use]
    pub const fn channels(self) -> usize {
        self.order.channels()
    }

    /// Bytes occupied by one pixel in memory.
    ///
    /// For [`BitPacking::Bitfields`] (such as `RGB565`), all channels fit inside
    /// a single storage word, so this equals `self.sample.byte_size()`.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        if self.packing.is_packed_word() {
            self.sample.byte_size()
        } else {
            self.order.channels() * self.sample.byte_size()
        }
    }

    /// Minimum byte alignment required for row starts and slice pointers.
    #[inline]
    #[must_use]
    pub const fn min_alignment(self) -> usize {
        self.sample.min_alignment()
    }

    /// Validate that channel count and bit packing are well-formed.
    pub const fn validate(self) -> Result<(), LayoutError> {
        let ch = self.order.channels();
        if ch == 0 || ch > 16 {
            return Err(LayoutError::InvalidChannelCount);
        }
        self.packing.validate(self.sample, ch)
    }

    /// Compute the row byte stride for `width` pixels under `align`.
    pub const fn aligned_stride(self, width: u32, align: RowAlign) -> Result<usize, LayoutError> {
        let bpp = self.bytes_per_pixel();
        let Some(row_bytes) = (width as usize).checked_mul(bpp) else {
            return Err(LayoutError::GeometryOverflow);
        };
        if row_bytes == 0 {
            return Ok(0);
        }
        let sample_bytes = self.sample.byte_size();
        match align {
            RowAlign::Tight => Ok(row_bytes),
            RowAlign::Cache64 => {
                let step = if sample_bytes > 64 { sample_bytes } else { 64 };
                let Some(aligned) = round_up_checked(row_bytes, step) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                if aligned % 4096 == 0 {
                    match aligned.checked_add(step) {
                        Some(v) => Ok(v),
                        None => Err(LayoutError::GeometryOverflow),
                    }
                } else {
                    Ok(aligned)
                }
            }
            RowAlign::PixelCache64 => {
                let step = lcm(bpp, 64);
                let Some(aligned) = round_up_checked(row_bytes, step) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                if aligned % 4096 == 0 {
                    match aligned.checked_add(step) {
                        Some(v) => Ok(v),
                        None => Err(LayoutError::GeometryOverflow),
                    }
                } else {
                    Ok(aligned)
                }
            }
            RowAlign::Samples(n) => {
                let mult = if n == 0 { 1 } else { n as usize };
                let Some(step) = mult.checked_mul(sample_bytes) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                match round_up_checked(row_bytes, step) {
                    Some(v) => Ok(v),
                    None => Err(LayoutError::GeometryOverflow),
                }
            }
        }
    }
}

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

const fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)) * b
    }
}

const fn round_up_checked(val: usize, step: usize) -> Option<usize> {
    if step <= 1 {
        return Some(val);
    }
    let rem = val % step;
    if rem == 0 {
        Some(val)
    } else {
        val.checked_add(step - rem)
    }
}

/// Role of a single plane inside a multi-plane image ([`PlanarBuf`](crate::PlanarBuf)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PlaneRole {
    /// Interleaved packed plane.
    Packed,
    /// Luma (`Y'`).
    Y,
    /// Blue-difference chroma (`Cb` / `U`).
    Cb,
    /// Red-difference chroma (`Cr` / `V`).
    Cr,
    /// Semi-planar interleaved chroma (`[Cb, Cr]`, e.g. NV12 / P010).
    CbCr,
    /// Semi-planar reversed interleaved chroma (`[Cr, Cb]`, e.g. NV21).
    CrCb,
    /// Separate red plane (`GBR` / planar RGB).
    R,
    /// Separate green plane.
    G,
    /// Separate blue plane.
    B,
    /// Separate cyan ink plane (planar CMYK/CMYKA).
    Cyan,
    /// Separate magenta ink plane.
    Magenta,
    /// Separate yellow ink plane.
    Yellow,
    /// Separate black/key ink plane.
    Key,
    /// Separate alpha plane.
    Alpha,
    /// HDR gain-map plane (ISO 21496-1 / UltraHDR).
    GainMap,
    /// Depth / disparity map plane (HEIF/AVIF/JXL/TIFF depth auxiliary image).
    Depth,
    /// Segmentation / foreground matte plane (HEIF/AVIF portrait matte).
    Matte,
    /// Thermal / infrared radiance plane.
    Thermal,
    /// Indexed auxiliary / extra channel plane (JXL extra channels, TIFF extra/multispectral bands).
    Extra(u16),
}

/// Rectangular pixel region specified by top-left origin `(x, y)` and `(width, height)` in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelRect {
    /// Left edge in pixels (`0`-based).
    pub x: u32,
    /// Top edge in pixels (`0`-based).
    pub y: u32,
    /// Region width in pixels.
    pub width: u32,
    /// Region height in pixels.
    pub height: u32,
}

impl PixelRect {
    /// Construct a pixel rectangle from `(x, y, width, height)`.
    #[inline]
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Storage specification for one plane in a multi-plane image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[doc(alias = "PlaneSpec")]
pub struct PlaneFormat {
    /// Semantic role of this plane.
    pub role: PlaneRole,
    /// Sample numeric storage type.
    pub sample: SampleType,
    /// Number of interleaved components per position (1 for `Y`/`Cb`/`Cr`, 2 for `CbCr`/`CrCb`).
    pub components: u8,
    /// Bit packing within each sample word.
    pub packing: BitPacking,
    /// Horizontal and vertical subsampling divisors `(hx, vy)` relative to frame dimensions.
    pub subsampling: (u8, u8),
}

impl PlaneFormat {
    /// Construct a full-resolution single-component plane specification.
    #[inline]
    #[must_use]
    pub const fn new(role: PlaneRole, sample: SampleType) -> Self {
        let components = match role {
            PlaneRole::CbCr | PlaneRole::CrCb => 2,
            _ => 1,
        };
        Self {
            role,
            sample,
            components,
            packing: BitPacking::Full,
            subsampling: (1, 1),
        }
    }

    /// Set bit packing (e.g. [`BitPacking::P010`]).
    #[inline]
    #[must_use]
    pub const fn with_packing(mut self, packing: BitPacking) -> Self {
        self.packing = packing;
        self
    }

    /// Set subsampling divisors `(horizontal_divisor, vertical_divisor)` (e.g. `(2, 2)` for 4:2:0 chroma).
    #[inline]
    #[must_use]
    pub const fn with_subsampling(mut self, horizontal_divisor: u8, vertical_divisor: u8) -> Self {
        self.subsampling = (horizontal_divisor, vertical_divisor);
        self
    }

    /// Bytes per plane element (one luma sample or one `CbCr` pair).
    #[inline]
    #[must_use]
    pub const fn bytes_per_element(self) -> usize {
        if self.packing.is_packed_word() {
            self.sample.byte_size()
        } else {
            (self.components as usize) * self.sample.byte_size()
        }
    }

    /// Compute this plane's `(width, height)` in elements for a full luma frame of `(frame_width, frame_height)`.
    #[inline]
    #[must_use]
    pub const fn plane_dimensions_for_frame(
        self,
        frame_width: u32,
        frame_height: u32,
    ) -> (u32, u32) {
        let hx = if self.subsampling.0 == 0 {
            1
        } else {
            self.subsampling.0 as u32
        };
        let vy = if self.subsampling.1 == 0 {
            1
        } else {
            self.subsampling.1 as u32
        };
        (frame_width.div_ceil(hx), frame_height.div_ceil(vy))
    }

    /// Compute this plane's `(width, height)` in elements for a cropped luma frame of
    /// `(frame_width, frame_height)` starting at luma-grid origin `crop_origin = (origin_x, origin_y)`.
    ///
    /// Preserves chroma phase at odd crop origins: a `2×2` subsampled plane for a crop
    /// `[origin_x .. origin_x + frame_width)` covers chroma columns `[origin_x / hx .. ceil((origin_x + frame_width) / hx))`.
    #[inline]
    #[must_use]
    pub const fn cropped_plane_dimensions(
        self,
        frame_width: u32,
        frame_height: u32,
        crop_origin: (u32, u32),
    ) -> Option<(u32, u32)> {
        let hx = if self.subsampling.0 == 0 {
            1
        } else {
            self.subsampling.0 as u32
        };
        let vy = if self.subsampling.1 == 0 {
            1
        } else {
            self.subsampling.1 as u32
        };
        let Some(end_x) = crop_origin.0.checked_add(frame_width) else {
            return None;
        };
        let Some(end_y) = crop_origin.1.checked_add(frame_height) else {
            return None;
        };
        let pw = end_x.div_ceil(hx) - (crop_origin.0 / hx);
        let ph = end_y.div_ceil(vy) - (crop_origin.1 / vy);
        Some((pw, ph))
    }

    /// Compute the row byte stride for `plane_width` plane elements (post-subsampling) under `align`.
    pub const fn aligned_stride_for_plane_width(
        self,
        plane_width: u32,
        align: RowAlign,
    ) -> Result<usize, LayoutError> {
        let bpe = self.bytes_per_element();
        let Some(row_bytes) = (plane_width as usize).checked_mul(bpe) else {
            return Err(LayoutError::GeometryOverflow);
        };
        if row_bytes == 0 {
            return Ok(0);
        }
        let sample_bytes = self.sample.byte_size();
        match align {
            RowAlign::Tight => Ok(row_bytes),
            RowAlign::Cache64 => {
                let step = if sample_bytes > 64 { sample_bytes } else { 64 };
                let Some(aligned) = round_up_checked(row_bytes, step) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                if aligned % 4096 == 0 {
                    match aligned.checked_add(step) {
                        Some(v) => Ok(v),
                        None => Err(LayoutError::GeometryOverflow),
                    }
                } else {
                    Ok(aligned)
                }
            }
            RowAlign::PixelCache64 => {
                let step = lcm(bpe, 64);
                let Some(aligned) = round_up_checked(row_bytes, step) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                if aligned % 4096 == 0 {
                    match aligned.checked_add(step) {
                        Some(v) => Ok(v),
                        None => Err(LayoutError::GeometryOverflow),
                    }
                } else {
                    Ok(aligned)
                }
            }
            RowAlign::Samples(n) => {
                let mult = if n == 0 { 1 } else { n as usize };
                let Some(step) = mult.checked_mul(sample_bytes) else {
                    return Err(LayoutError::GeometryOverflow);
                };
                match round_up_checked(row_bytes, step) {
                    Some(v) => Ok(v),
                    None => Err(LayoutError::GeometryOverflow),
                }
            }
        }
    }

    /// Derive the [`SampleEncoding`](crate::SampleEncoding) for unpacked integer planes
    /// (`U8`/`U16`/`U32` with `Full`, `Lsb`, `Msb`, or `Shifted` packing), or `None` for
    /// floating-point or packed bitfield words.
    #[inline]
    #[must_use]
    pub const fn sample_encoding(self) -> Option<crate::SampleEncoding> {
        if !self.sample.is_integer() {
            return None;
        }
        let word_bits = self.sample.bits();
        let (code_bits, bit_shift) = match self.packing {
            BitPacking::Full => (word_bits, 0),
            BitPacking::Lsb { bits } => (bits, 0),
            BitPacking::Msb { bits } => {
                if bits > word_bits {
                    return None;
                }
                (bits, word_bits - bits)
            }
            BitPacking::Shifted { bits, shift } => (bits, shift),
            BitPacking::Bitfields { .. } => return None,
        };
        match crate::SampleEncoding::new(self.sample, code_bits, bit_shift) {
            Ok(enc) => Some(enc),
            Err(_) => None,
        }
    }

    /// Validate component count (including role arity), subsampling, and bit packing.
    pub const fn validate(self) -> Result<(), LayoutError> {
        let expected_arity = match self.role {
            PlaneRole::CbCr | PlaneRole::CrCb => Some(2u8),
            PlaneRole::Y
            | PlaneRole::Cb
            | PlaneRole::Cr
            | PlaneRole::R
            | PlaneRole::G
            | PlaneRole::B
            | PlaneRole::Cyan
            | PlaneRole::Magenta
            | PlaneRole::Yellow
            | PlaneRole::Key
            | PlaneRole::Alpha
            | PlaneRole::GainMap
            | PlaneRole::Depth
            | PlaneRole::Matte
            | PlaneRole::Thermal => Some(1u8),
            PlaneRole::Packed | PlaneRole::Extra(_) => None,
        };
        if let Some(want) = expected_arity {
            if self.components != want {
                return Err(LayoutError::InvalidChannelCount);
            }
        } else if self.components == 0 || self.components > 4 {
            return Err(LayoutError::InvalidChannelCount);
        }
        if self.subsampling.0 == 0 || self.subsampling.1 == 0 {
            return Err(LayoutError::InvalidSubsampling);
        }
        self.packing.validate(self.sample, self.components as usize)
    }
}

/// Row-start alignment policy for allocated buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowAlign {
    /// Tightly packed rows (`stride == width * bytes_per_pixel`).
    #[default]
    Tight,
    /// 64-byte row alignment (widest SIMD vector & cache line), with a +64 B
    /// anti-aliasing bump when the aligned stride is a nonzero multiple of 4096 bytes.
    Cache64,
    /// Whole-pixel 64-byte alignment (`lcm(bytes_per_pixel, 64)`), plus one step
    /// on 4096-byte multiples so whole-pixel stride consumers like `imgref` stay aligned.
    PixelCache64,
    /// Align row stride to a multiple of `N` samples (`0` treated as `1`).
    Samples(u16),
}

impl RowAlign {
    /// Convert a byte pitch (such as `wgpu`'s 256-byte alignment) into [`RowAlign::Samples`].
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: u16, sample: SampleType) -> Option<Self> {
        let sz = sample.byte_size();
        if bytes == 0 || (bytes as usize) % sz != 0 {
            None
        } else {
            Some(Self::Samples((bytes as usize / sz) as u16))
        }
    }
}

/// Row stride measured in **scalar samples** (not bytes).
///
/// Used by [`PlaneBuf::from_samples_strided`](crate::PlaneBuf::from_samples_strided),
/// [`PlaneView::from_samples_strided`](crate::PlaneView::from_samples_strided), and
/// [`PlaneViewMut::from_samples_strided`](crate::PlaneViewMut::from_samples_strided) so
/// sample-count strides can never be confused with byte strides at the call site.
///
/// For tightly packed planes (`stride == width * spec.components`), use the 4-argument
/// `from_samples` constructors instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SampleStride(pub usize);

impl SampleStride {
    /// Construct a row stride measured in scalar samples.
    #[inline]
    #[must_use]
    pub const fn new(samples: usize) -> Self {
        Self(samples)
    }

    /// Row stride in scalar samples.
    #[inline]
    #[must_use]
    pub const fn samples(self) -> usize {
        self.0
    }
}

impl fmt::Display for SampleStride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} samples", self.0)
    }
}

/// Downstream container contract for [`PixelBuf::new_for_interop`](crate::PixelBuf::new_for_interop).
///
/// Validates upfront that an [`PixelFormat`](crate::PixelFormat) can move into the
/// target container **without copying or silently relabeling pixels**.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InteropTarget {
    /// Tightly packed native storage (`stride == width * bytes_per_pixel`, `offset == 0`).
    #[default]
    Tight,
    /// Zero-copy `imgref::ImgVec<P>` / `ImgRef<P>` handoff.
    Imgref,
    /// Zero-copy `image::flat::FlatSamples<Vec<S>>` handoff.
    FlatSamples,
    /// Zero-copy `image::ImageBuffer<P, Vec<S>>` handoff with lossless CICP color preservation.
    ImageBuffer,
    /// Zero-copy `image::DynamicImage` handoff with lossless CICP color preservation.
    DynamicImage,
}

/// Layout validation or stride computation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayoutError {
    /// Channel count is zero or exceeds the supported maximum.
    InvalidChannelCount,
    /// Bit depth or shift exceeds the underlying sample word.
    InvalidBitDepth,
    /// Bit packing is not supported for the chosen [`SampleType`].
    UnsupportedBitPacking,
    /// Plane subsampling divisor is zero.
    InvalidSubsampling,
    /// Stride or byte-size computation overflowed `usize`.
    GeometryOverflow,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidChannelCount => "invalid channel count for layout",
            Self::InvalidBitDepth => "bit-packing depth or shift exceeds storage word",
            Self::UnsupportedBitPacking => "bit packing is unsupported for this sample type",
            Self::InvalidSubsampling => "plane subsampling divisor must be non-zero",
            Self::GeometryOverflow => "row stride or byte size overflow",
        })
    }
}

impl core::error::Error for LayoutError {}
