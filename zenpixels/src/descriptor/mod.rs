//! Pixel format descriptor types.
//!
//! These types describe the format of pixel data: channel type, layout,
//! alpha handling, transfer function, color primaries, and signal range.
//!
//! Standalone definitions — no dependency on zencodec-types.

use core::fmt;

mod pixel_descriptor;
mod pixel_format;
#[cfg(feature = "planar")]
pub mod planar;
#[cfg(test)]
mod tests;

pub use pixel_descriptor::*;
pub use pixel_format::*;
#[cfg(feature = "planar")]
pub use planar::*;

// ---------------------------------------------------------------------------
// Channel type
// ---------------------------------------------------------------------------

/// Channel storage type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ChannelType {
    /// 8-bit unsigned integer (1 byte per channel).
    U8 = 1,
    /// 16-bit unsigned integer (2 bytes per channel).
    U16 = 2,
    /// 32-bit floating point (4 bytes per channel).
    F32 = 4,
    /// IEEE 754 half-precision float (2 bytes per channel).
    F16 = 5,
}

impl ChannelType {
    /// Byte size of a single channel value.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 => 2,
            Self::F32 => 4,
            _ => 0,
        }
    }

    /// Whether this is [`U8`](Self::U8).
    #[inline]
    pub const fn is_u8(self) -> bool {
        matches!(self, Self::U8)
    }

    /// Whether this is [`U16`](Self::U16).
    #[inline]
    pub const fn is_u16(self) -> bool {
        matches!(self, Self::U16)
    }

    /// Whether this is [`F32`](Self::F32).
    #[inline]
    pub const fn is_f32(self) -> bool {
        matches!(self, Self::F32)
    }

    /// Whether this is [`F16`](Self::F16).
    #[inline]
    pub const fn is_f16(self) -> bool {
        matches!(self, Self::F16)
    }

    /// Whether this is an integer type.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16)
    }

    /// Whether this is a floating-point type.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16)
    }
}

impl fmt::Display for ChannelType {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8 => f.write_str("U8"),
            Self::U16 => f.write_str("U16"),
            Self::F32 => f.write_str("F32"),
            Self::F16 => f.write_str("F16"),
            _ => write!(f, "ChannelType({})", *self as u8),
        }
    }
}

// ---------------------------------------------------------------------------
// Channel layout
// ---------------------------------------------------------------------------

/// Channel layout (number and meaning of channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ChannelLayout {
    /// Single luminance channel.
    Gray = 1,
    /// Luminance + alpha.
    GrayAlpha = 2,
    /// Red, green, blue.
    Rgb = 3,
    /// Red, green, blue, alpha.
    Rgba = 4,
    /// Blue, green, red, alpha (Windows/DirectX byte order).
    Bgra = 5,
    /// Oklab perceptual color: L, a, b.
    Oklab = 6,
    /// Oklab perceptual color with alpha: L, a, b, alpha.
    OklabA = 7,
}

impl ChannelLayout {
    /// Number of channels in this layout.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb | Self::Oklab => 3,
            Self::Rgba | Self::Bgra | Self::OklabA => 4,
            _ => 0,
        }
    }

    /// Whether this layout includes an alpha channel.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::GrayAlpha | Self::Rgba | Self::Bgra | Self::OklabA
        )
    }
}

impl fmt::Display for ChannelLayout {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gray => f.write_str("Gray"),
            Self::GrayAlpha => f.write_str("GrayAlpha"),
            Self::Rgb => f.write_str("RGB"),
            Self::Rgba => f.write_str("RGBA"),
            Self::Bgra => f.write_str("BGRA"),
            Self::Oklab => f.write_str("Oklab"),
            Self::OklabA => f.write_str("OklabA"),
            _ => write!(f, "ChannelLayout({})", *self as u8),
        }
    }
}

// ---------------------------------------------------------------------------
// Alpha mode
// ---------------------------------------------------------------------------

/// Alpha channel interpretation.
///
/// Wrapped in `Option<AlphaMode>` on [`PixelDescriptor`]: `None` means no
/// alpha channel exists, while `Some(AlphaMode::Straight)` etc. describe
/// the semantics of a present alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum AlphaMode {
    /// Alpha bytes exist but values are undefined padding (RGBX, BGRX).
    Undefined = 1,
    /// Straight (unassociated) alpha.
    Straight = 2,
    /// Premultiplied (associated) alpha.
    Premultiplied = 3,
    /// Alpha channel present, all values fully opaque.
    Opaque = 4,
}

impl AlphaMode {
    /// Whether this mode represents a real alpha channel (not Undefined padding).
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Straight | Self::Premultiplied | Self::Opaque)
    }

    /// Whether this mode has any alpha-position bytes (including padding).
    #[inline]
    pub const fn has_alpha_bytes(self) -> bool {
        true
    }
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undefined => f.write_str("undefined"),
            Self::Straight => f.write_str("straight"),
            Self::Premultiplied => f.write_str("premultiplied"),
            Self::Opaque => f.write_str("opaque"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer function
// ---------------------------------------------------------------------------

/// Electro-optical transfer function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum TransferFunction {
    /// Linear light (gamma 1.0).
    Linear = 0,
    /// sRGB transfer curve (IEC 61966-2-1).
    Srgb = 1,
    /// BT.709 transfer curve.
    Bt709 = 2,
    /// Perceptual Quantizer (SMPTE ST 2084, HDR10).
    Pq = 3,
    /// Hybrid Log-Gamma (ARIB STD-B67, HLG).
    Hlg = 4,
    /// Transfer function is not known.
    Unknown = 255,
}

impl TransferFunction {
    /// Map CICP `transfer_characteristics` code to a [`TransferFunction`].
    pub const fn from_cicp(tc: u8) -> Option<Self> {
        match tc {
            1 => Some(Self::Bt709),
            8 => Some(Self::Linear),
            13 => Some(Self::Srgb),
            16 => Some(Self::Pq),
            18 => Some(Self::Hlg),
            _ => None,
        }
    }

    /// Reference white luminance in nits.
    ///
    /// - SDR (sRGB, BT.709, Linear, Unknown): `1.0` (relative/scene-referred)
    /// - PQ: `203.0` (ITU-R BT.2408 reference white)
    /// - HLG: `1.0` (scene-referred)
    #[allow(unreachable_patterns)]
    pub fn reference_white_nits(&self) -> f32 {
        match self {
            Self::Pq => 203.0,
            _ => 1.0,
        }
    }
}

impl fmt::Display for TransferFunction {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => f.write_str("linear"),
            Self::Srgb => f.write_str("sRGB"),
            Self::Bt709 => f.write_str("BT.709"),
            Self::Pq => f.write_str("PQ"),
            Self::Hlg => f.write_str("HLG"),
            Self::Unknown => f.write_str("unknown"),
            _ => write!(f, "TransferFunction({})", *self as u8),
        }
    }
}

// ---------------------------------------------------------------------------
// Color primaries
// ---------------------------------------------------------------------------

/// Color primaries (CIE xy chromaticities of R, G, B).
///
/// Discriminant values match CICP `ColorPrimaries` codes (ITU-T H.273).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ColorPrimaries {
    /// BT.709 / sRGB (CICP 1).
    #[default]
    Bt709 = 1,
    /// BT.2020 / BT.2100 (CICP 9). Wide gamut for HDR.
    Bt2020 = 9,
    /// Display P3 (CICP 12). Apple ecosystem, wide gamut SDR.
    DisplayP3 = 12,
    /// Primaries not known.
    Unknown = 255,
}

impl ColorPrimaries {
    /// Map a CICP `color_primaries` code to a [`ColorPrimaries`].
    pub const fn from_cicp(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bt709),
            9 => Some(Self::Bt2020),
            12 => Some(Self::DisplayP3),
            _ => None,
        }
    }

    /// Convert to the CICP `color_primaries` code.
    #[allow(unreachable_patterns)]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Bt2020 => Some(9),
            Self::DisplayP3 => Some(12),
            Self::Unknown => None,
            _ => None,
        }
    }

    /// Whether `self` fully contains the gamut of `other`.
    ///
    /// Gamut hierarchy: BT.2020 > Display P3 > BT.709.
    pub const fn contains(self, other: Self) -> bool {
        self.gamut_width() >= other.gamut_width()
            && !matches!(self, Self::Unknown)
            && !matches!(other, Self::Unknown)
    }

    #[allow(unreachable_patterns)]
    const fn gamut_width(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::DisplayP3 => 2,
            Self::Bt2020 => 3,
            Self::Unknown => 0,
            _ => 0,
        }
    }
}

impl fmt::Display for ColorPrimaries {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bt709 => f.write_str("BT.709"),
            Self::Bt2020 => f.write_str("BT.2020"),
            Self::DisplayP3 => f.write_str("Display P3"),
            Self::Unknown => f.write_str("unknown"),
            _ => write!(f, "ColorPrimaries({})", *self as u8),
        }
    }
}

// ---------------------------------------------------------------------------
// Signal range
// ---------------------------------------------------------------------------

/// Signal range for pixel values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum SignalRange {
    /// Full range: 0-2^N-1 (e.g. 0-255 for 8-bit).
    #[default]
    Full = 0,
    /// Narrow (limited/studio) range: 16-235 luma, 16-240 chroma (for 8-bit).
    Narrow = 1,
}

impl fmt::Display for SignalRange {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => f.write_str("full"),
            Self::Narrow => f.write_str("narrow"),
            _ => write!(f, "SignalRange({})", *self as u8),
        }
    }
}

// ---------------------------------------------------------------------------
// Color model — what the channels represent
// ---------------------------------------------------------------------------

/// What the channels represent, independent of channel count or byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ColorModel {
    /// Single grayscale channel.
    Gray = 0,
    /// Red, green, blue (or BGR when [`ByteOrder::Bgr`]).
    Rgb = 1,
    /// Luma + chroma (Y, Cb, Cr).
    YCbCr = 2,
    /// Oklab perceptual color space (L, a, b).
    Oklab = 3,
    /// CIE XYZ.
    Xyz = 4,
    /// CIE L*a*b*.
    Lab = 5,
}

impl ColorModel {
    /// Number of color channels (excluding alpha).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn color_channels(self) -> u8 {
        match self {
            Self::Gray => 1,
            _ => 3,
        }
    }
}

impl fmt::Display for ColorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gray => f.write_str("Gray"),
            Self::Rgb => f.write_str("RGB"),
            Self::YCbCr => f.write_str("YCbCr"),
            Self::Oklab => f.write_str("Oklab"),
            Self::Xyz => f.write_str("XYZ"),
            Self::Lab => f.write_str("L*a*b*"),
        }
    }
}

// ---------------------------------------------------------------------------
// Byte order
// ---------------------------------------------------------------------------

/// RGB-family byte order. Only meaningful when color model is RGB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ByteOrder {
    /// Standard order: R, G, B (+ A if present).
    #[default]
    Native = 0,
    /// Windows/DirectX order: B, G, R (+ A if present).
    Bgr = 1,
}

impl fmt::Display for ByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::Bgr => f.write_str("BGR"),
        }
    }
}

// ---------------------------------------------------------------------------
// Deprecated alias
// ---------------------------------------------------------------------------

/// Deprecated alias — use [`PixelFormat`] instead.
#[deprecated(note = "renamed to PixelFormat")]
pub type InterleaveFormat = PixelFormat;

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

pub(crate) const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub(crate) const fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

pub(crate) const fn align_up_general(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let rem = value % align;
    if rem == 0 { value } else { value + align - rem }
}
