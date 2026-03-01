//! Pixel format descriptor types.
//!
//! These types describe the format of pixel data: channel type, layout,
//! alpha handling, transfer function, color primaries, and signal range.
//!
//! Standalone definitions — no dependency on zencodec-types.

use core::fmt;

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
    /// Signed 16-bit integer (2 bytes per channel).
    I16 = 6,
}

impl ChannelType {
    /// Byte size of a single channel value.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 | Self::I16 => 2,
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

    /// Whether this is [`I16`](Self::I16).
    #[inline]
    pub const fn is_i16(self) -> bool {
        matches!(self, Self::I16)
    }

    /// Whether this is an integer type.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::I16)
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
            Self::I16 => f.write_str("I16"),
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
}

impl ChannelLayout {
    /// Number of channels in this layout.
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    /// Whether this layout includes an alpha channel.
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba | Self::Bgra)
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gray => f.write_str("Gray"),
            Self::GrayAlpha => f.write_str("GrayAlpha"),
            Self::Rgb => f.write_str("RGB"),
            Self::Rgba => f.write_str("RGBA"),
            Self::Bgra => f.write_str("BGRA"),
        }
    }
}

// ---------------------------------------------------------------------------
// Alpha mode
// ---------------------------------------------------------------------------

/// Alpha channel interpretation.
///
/// Unlike `Option<AlphaMode>`, this uses a flat enum for ergonomic matching:
/// `AlphaMode::None` means no alpha channel, vs `Some(AlphaMode::Straight)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum AlphaMode {
    /// No alpha channel exists.
    #[default]
    None = 0,
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
    /// Whether this mode represents a real alpha channel (not None or Undefined).
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Straight | Self::Premultiplied | Self::Opaque)
    }

    /// Whether this mode has any alpha-position bytes (including padding).
    #[inline]
    pub const fn has_alpha_bytes(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
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
// PixelDescriptor
// ---------------------------------------------------------------------------

/// Compact pixel format descriptor.
///
/// Describes the format of pixel data without carrying the data itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub struct PixelDescriptor {
    /// Channel storage type (u8, u16, f16, i16, f32).
    pub channel_type: ChannelType,
    /// Channel layout (gray, RGB, RGBA, etc.).
    pub layout: ChannelLayout,
    /// Alpha interpretation.
    pub alpha: AlphaMode,
    /// Transfer function (sRGB, linear, PQ, etc.).
    pub transfer: TransferFunction,
    /// Color primaries (gamut). Defaults to BT.709/sRGB.
    pub primaries: ColorPrimaries,
    /// Signal range (full vs narrow/limited).
    pub signal_range: SignalRange,
}

impl PixelDescriptor {
    /// Create a descriptor with default primaries (BT.709) and full range.
    pub const fn new(
        channel_type: ChannelType,
        layout: ChannelLayout,
        alpha: AlphaMode,
        transfer: TransferFunction,
    ) -> Self {
        Self {
            channel_type,
            layout,
            alpha,
            transfer,
            primaries: ColorPrimaries::Bt709,
            signal_range: SignalRange::Full,
        }
    }

    /// Create a descriptor with explicit primaries.
    pub const fn new_full(
        channel_type: ChannelType,
        layout: ChannelLayout,
        alpha: AlphaMode,
        transfer: TransferFunction,
        primaries: ColorPrimaries,
    ) -> Self {
        Self {
            channel_type,
            layout,
            alpha,
            transfer,
            primaries,
            signal_range: SignalRange::Full,
        }
    }

    // -- sRGB constants -------------------------------------------------------

    /// 8-bit sRGB RGB.
    pub const RGB8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB RGBA with straight alpha.
    pub const RGBA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB RGB.
    pub const RGB16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB RGBA with straight alpha.
    pub const RGBA16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    /// Linear-light f32 RGB.
    pub const RGBF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Linear,
    );
    /// Linear-light f32 RGBA with straight alpha.
    pub const RGBAF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Linear,
    );
    /// 8-bit sRGB grayscale.
    pub const GRAY8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB grayscale.
    pub const GRAY16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    /// Linear-light f32 grayscale.
    pub const GRAYF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Linear,
    );
    /// 8-bit sRGB grayscale with straight alpha.
    pub const GRAYA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB grayscale with straight alpha.
    pub const GRAYA16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    /// Linear-light f32 grayscale with straight alpha.
    pub const GRAYAF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Linear,
    );
    /// 8-bit sRGB BGRA with straight alpha.
    pub const BGRA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        AlphaMode::Straight,
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB RGBX (padding byte, not alpha).
    pub const RGBX8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Undefined,
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB BGRX (padding byte, not alpha).
    pub const BGRX8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        AlphaMode::Undefined,
        TransferFunction::Srgb,
    );

    // -- Transfer-agnostic constants ------------------------------------------

    /// 8-bit RGB, transfer unknown.
    pub const RGB8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// 8-bit RGBA, transfer unknown.
    pub const RGBA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// 16-bit RGB, transfer unknown.
    pub const RGB16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// 16-bit RGBA, transfer unknown.
    pub const RGBA16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// f32 RGB, transfer unknown.
    pub const RGBF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// f32 RGBA, transfer unknown.
    pub const RGBAF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// 8-bit grayscale, transfer unknown.
    pub const GRAY8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// 16-bit grayscale, transfer unknown.
    pub const GRAY16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// f32 grayscale, transfer unknown.
    pub const GRAYF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Gray,
        AlphaMode::None,
        TransferFunction::Unknown,
    );
    /// 8-bit grayscale with alpha, transfer unknown.
    pub const GRAYA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// 16-bit grayscale with alpha, transfer unknown.
    pub const GRAYA16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// f32 grayscale with alpha, transfer unknown.
    pub const GRAYAF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::GrayAlpha,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// 8-bit BGRA, transfer unknown.
    pub const BGRA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        AlphaMode::Straight,
        TransferFunction::Unknown,
    );
    /// 8-bit RGBX, transfer unknown.
    pub const RGBX8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        AlphaMode::Undefined,
        TransferFunction::Unknown,
    );
    /// 8-bit BGRX, transfer unknown.
    pub const BGRX8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        AlphaMode::Undefined,
        TransferFunction::Unknown,
    );

    // -- Methods --------------------------------------------------------------

    /// Number of channels.
    #[inline]
    pub const fn channels(self) -> usize {
        self.layout.channels()
    }

    /// Bytes per pixel.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        self.layout.channels() * self.channel_type.byte_size()
    }

    /// Whether this descriptor has meaningful alpha data.
    #[inline]
    pub const fn has_alpha(self) -> bool {
        self.alpha.has_alpha()
    }

    /// Whether this descriptor is grayscale.
    #[inline]
    pub const fn is_grayscale(self) -> bool {
        matches!(self.layout, ChannelLayout::Gray | ChannelLayout::GrayAlpha)
    }

    /// Whether this descriptor uses BGR byte order.
    #[inline]
    pub const fn is_bgr(self) -> bool {
        matches!(self.layout, ChannelLayout::Bgra)
    }

    /// Return a copy with a different transfer function.
    #[inline]
    pub const fn with_transfer(self, transfer: TransferFunction) -> Self {
        Self { transfer, ..self }
    }

    /// Return a copy with different primaries.
    #[inline]
    pub const fn with_primaries(self, primaries: ColorPrimaries) -> Self {
        Self { primaries, ..self }
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    pub const fn with_alpha(self, alpha: AlphaMode) -> Self {
        Self { alpha, ..self }
    }

    /// Return a copy with a different signal range.
    #[inline]
    pub const fn with_signal_range(self, signal_range: SignalRange) -> Self {
        Self {
            signal_range,
            ..self
        }
    }

    /// Whether this descriptor's channel type and layout are compatible with `other`.
    ///
    /// "Compatible" means the raw bytes can be reinterpreted as `other`
    /// without any pixel transformation — same channel type, same layout.
    #[inline]
    pub const fn layout_compatible(self, other: Self) -> bool {
        self.channel_type as u8 == other.channel_type as u8
            && self.layout as u8 == other.layout as u8
    }
}

impl fmt::Display for PixelDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.layout, self.channel_type, self.transfer)?;
        if self.alpha.has_alpha() {
            write!(f, " alpha={}", self.alpha)?;
        }
        Ok(())
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
// Chroma subsampling
// ---------------------------------------------------------------------------

/// Chroma subsampling ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Subsampling {
    /// 4:4:4 — no subsampling, full resolution chroma.
    #[default]
    S444 = 0,
    /// 4:2:2 — horizontal half resolution chroma.
    S422 = 1,
    /// 4:2:0 — both horizontal and vertical half resolution chroma.
    S420 = 2,
    /// 4:1:1 — quarter horizontal resolution chroma.
    S411 = 3,
}

impl Subsampling {
    /// Horizontal subsampling factor (1 = full, 2 = half, 4 = quarter).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn h_factor(self) -> u8 {
        match self {
            Self::S444 => 1,
            Self::S422 | Self::S420 => 2,
            Self::S411 => 4,
            _ => 1,
        }
    }

    /// Vertical subsampling factor (1 = full, 2 = half).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn v_factor(self) -> u8 {
        match self {
            Self::S420 => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for Subsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S444 => f.write_str("4:4:4"),
            Self::S422 => f.write_str("4:2:2"),
            Self::S420 => f.write_str("4:2:0"),
            Self::S411 => f.write_str("4:1:1"),
        }
    }
}

// ---------------------------------------------------------------------------
// YUV matrix coefficients
// ---------------------------------------------------------------------------

/// YCbCr matrix coefficients for luma/chroma conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum YuvMatrix {
    /// Identity / not applicable (RGB, Gray, Oklab, etc.).
    #[default]
    Identity = 0,
    /// BT.601: Y = 0.299R + 0.587G + 0.114B (JPEG, WebP, SD video).
    Bt601 = 1,
    /// BT.709: Y = 0.2126R + 0.7152G + 0.0722B (AVIF, HEIC, HD video).
    Bt709 = 2,
    /// BT.2020: Y = 0.2627R + 0.6780G + 0.0593B (4K/8K HDR).
    Bt2020 = 3,
}

impl YuvMatrix {
    /// RGB to Y luma coefficients [Kr, Kg, Kb].
    #[allow(unreachable_patterns)]
    pub const fn rgb_to_y_coeffs(self) -> [f64; 3] {
        match self {
            Self::Identity => [1.0, 0.0, 0.0],
            Self::Bt601 => [0.299, 0.587, 0.114],
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
            _ => [0.2126, 0.7152, 0.0722],
        }
    }

    /// Map a CICP `matrix_coefficients` code to a [`YuvMatrix`].
    pub const fn from_cicp(mc: u8) -> Option<Self> {
        match mc {
            0 => Some(Self::Identity),
            1 => Some(Self::Bt709),
            5 | 6 => Some(Self::Bt601),
            9 => Some(Self::Bt2020),
            _ => None,
        }
    }
}

impl fmt::Display for YuvMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => f.write_str("Identity"),
            Self::Bt601 => f.write_str("BT.601"),
            Self::Bt709 => f.write_str("BT.709"),
            Self::Bt2020 => f.write_str("BT.2020"),
        }
    }
}

// ---------------------------------------------------------------------------
// PixelFormat — match-friendly physical layout enum
// ---------------------------------------------------------------------------

/// Physical pixel layout for match-based format dispatch.
///
/// Captures channel type and layout only — NOT transfer function, primaries,
/// or signal range. Every variant corresponds to a named [`PixelDescriptor`]
/// constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Rgb16,
    Rgba16,
    RgbF32,
    RgbaF32,
    Gray8,
    Gray16,
    GrayF32,
    GrayA8,
    GrayA16,
    GrayAF32,
    Bgra8,
    Rgbx8,
    Bgrx8,
}

impl PixelFormat {
    /// Base descriptor with `Unknown` transfer and BT.709 primaries.
    #[allow(unreachable_patterns)]
    pub const fn descriptor(self) -> PixelDescriptor {
        match self {
            Self::Rgb8 => PixelDescriptor::RGB8,
            Self::Rgba8 => PixelDescriptor::RGBA8,
            Self::Rgb16 => PixelDescriptor::RGB16,
            Self::Rgba16 => PixelDescriptor::RGBA16,
            Self::RgbF32 => PixelDescriptor::RGBF32,
            Self::RgbaF32 => PixelDescriptor::RGBAF32,
            Self::Gray8 => PixelDescriptor::GRAY8,
            Self::Gray16 => PixelDescriptor::GRAY16,
            Self::GrayF32 => PixelDescriptor::GRAYF32,
            Self::GrayA8 => PixelDescriptor::GRAYA8,
            Self::GrayA16 => PixelDescriptor::GRAYA16,
            Self::GrayAF32 => PixelDescriptor::GRAYAF32,
            Self::Bgra8 => PixelDescriptor::BGRA8,
            Self::Rgbx8 => PixelDescriptor::RGBX8,
            Self::Bgrx8 => PixelDescriptor::BGRX8,
            _ => PixelDescriptor::RGB8,
        }
    }

    /// Short human-readable name.
    #[allow(unreachable_patterns)]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rgb8 => "RGB8",
            Self::Rgba8 => "RGBA8",
            Self::Rgb16 => "RGB16",
            Self::Rgba16 => "RGBA16",
            Self::RgbF32 => "RgbF32",
            Self::RgbaF32 => "RgbaF32",
            Self::Gray8 => "Gray8",
            Self::Gray16 => "Gray16",
            Self::GrayF32 => "GrayF32",
            Self::GrayA8 => "GrayA8",
            Self::GrayA16 => "GrayA16",
            Self::GrayAF32 => "GrayAF32",
            Self::Bgra8 => "BGRA8",
            Self::Rgbx8 => "RGBX8",
            Self::Bgrx8 => "BGRX8",
            _ => "Unknown",
        }
    }

    /// Bytes per pixel.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        self.descriptor().bytes_per_pixel()
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Planar descriptors
// ---------------------------------------------------------------------------

/// Maximum number of planes in a planar layout.
pub const MAX_PLANES: usize = 8;

/// Semantic meaning of a single plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum PlaneSemantic {
    Luma = 0,
    ChromaCb = 1,
    ChromaCr = 2,
    Red = 3,
    Green = 4,
    Blue = 5,
    Alpha = 6,
    OklabL = 7,
    OklabA = 8,
    OklabB = 9,
    GainMap = 10,
    Gray = 11,
}

impl fmt::Display for PlaneSemantic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Luma => f.write_str("Y"),
            Self::ChromaCb => f.write_str("Cb"),
            Self::ChromaCr => f.write_str("Cr"),
            Self::Red => f.write_str("R"),
            Self::Green => f.write_str("G"),
            Self::Blue => f.write_str("B"),
            Self::Alpha => f.write_str("A"),
            Self::OklabL => f.write_str("L"),
            Self::OklabA => f.write_str("a"),
            Self::OklabB => f.write_str("b"),
            Self::GainMap => f.write_str("GainMap"),
            Self::Gray => f.write_str("Gray"),
        }
    }
}

/// Descriptor for a single plane in a planar layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneSpec {
    /// What this plane represents.
    pub semantic: PlaneSemantic,
    /// Horizontal subsampling (1=full, 2=half, 4=quarter).
    pub h_subsample: u8,
    /// Vertical subsampling (1=full, 2=half).
    pub v_subsample: u8,
    /// Per-plane channel type (usually same for all).
    pub channel_type: ChannelType,
}

impl PlaneSpec {
    /// Compute plane width from reference (luma) width.
    #[inline]
    pub const fn plane_width(self, ref_w: u32) -> u32 {
        ref_w.div_ceil(self.h_subsample as u32)
    }

    /// Compute plane height from reference (luma) height.
    #[inline]
    pub const fn plane_height(self, ref_h: u32) -> u32 {
        ref_h.div_ceil(self.v_subsample as u32)
    }

    /// Whether this plane is subsampled.
    #[inline]
    pub const fn is_subsampled(self) -> bool {
        self.h_subsample > 1 || self.v_subsample > 1
    }
}

/// Fixed-size planar format descriptor. Up to 8 planes, no heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanarDescriptor {
    planes: [PlaneSpec; MAX_PLANES],
    plane_count: u8,
    /// YUV matrix coefficients (if YCbCr).
    pub yuv_matrix: YuvMatrix,
    /// Transfer function for all planes.
    pub transfer: TransferFunction,
}

impl PlanarDescriptor {
    /// Number of planes.
    #[inline]
    pub const fn plane_count(&self) -> usize {
        self.plane_count as usize
    }

    /// Access all plane specs.
    #[inline]
    pub fn planes(&self) -> &[PlaneSpec] {
        &self.planes[..self.plane_count as usize]
    }

    /// Width of a specific plane given reference width.
    #[inline]
    pub fn plane_width(&self, idx: usize, ref_w: u32) -> u32 {
        self.planes[idx].plane_width(ref_w)
    }

    /// Height of a specific plane given reference height.
    #[inline]
    pub fn plane_height(&self, idx: usize, ref_h: u32) -> u32 {
        self.planes[idx].plane_height(ref_h)
    }

    // -- Factory methods ------------------------------------------------------

    const fn spec(semantic: PlaneSemantic, h: u8, v: u8, ct: ChannelType) -> PlaneSpec {
        PlaneSpec {
            semantic,
            h_subsample: h,
            v_subsample: v,
            channel_type: ct,
        }
    }

    const EMPTY_SPEC: PlaneSpec = PlaneSpec {
        semantic: PlaneSemantic::Gray,
        h_subsample: 1,
        v_subsample: 1,
        channel_type: ChannelType::U8,
    };

    fn from_specs(specs: &[PlaneSpec], matrix: YuvMatrix, transfer: TransferFunction) -> Self {
        let mut planes = [Self::EMPTY_SPEC; MAX_PLANES];
        let count = specs.len().min(MAX_PLANES);
        planes[..count].copy_from_slice(&specs[..count]);
        Self {
            planes,
            plane_count: count as u8,
            yuv_matrix: matrix,
            transfer,
        }
    }

    /// YCbCr 4:2:0.
    pub fn ycbcr_420(ct: ChannelType, matrix: YuvMatrix, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Luma, 1, 1, ct),
                Self::spec(PlaneSemantic::ChromaCb, 2, 2, ct),
                Self::spec(PlaneSemantic::ChromaCr, 2, 2, ct),
            ],
            matrix,
            transfer,
        )
    }

    /// YCbCr 4:2:2.
    pub fn ycbcr_422(ct: ChannelType, matrix: YuvMatrix, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Luma, 1, 1, ct),
                Self::spec(PlaneSemantic::ChromaCb, 2, 1, ct),
                Self::spec(PlaneSemantic::ChromaCr, 2, 1, ct),
            ],
            matrix,
            transfer,
        )
    }

    /// YCbCr 4:4:4.
    pub fn ycbcr_444(ct: ChannelType, matrix: YuvMatrix, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Luma, 1, 1, ct),
                Self::spec(PlaneSemantic::ChromaCb, 1, 1, ct),
                Self::spec(PlaneSemantic::ChromaCr, 1, 1, ct),
            ],
            matrix,
            transfer,
        )
    }

    /// YCbCr 4:1:1.
    pub fn ycbcr_411(ct: ChannelType, matrix: YuvMatrix, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Luma, 1, 1, ct),
                Self::spec(PlaneSemantic::ChromaCb, 4, 1, ct),
                Self::spec(PlaneSemantic::ChromaCr, 4, 1, ct),
            ],
            matrix,
            transfer,
        )
    }

    /// Planar RGB.
    pub fn planar_rgb(ct: ChannelType, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Red, 1, 1, ct),
                Self::spec(PlaneSemantic::Green, 1, 1, ct),
                Self::spec(PlaneSemantic::Blue, 1, 1, ct),
            ],
            YuvMatrix::Identity,
            transfer,
        )
    }

    /// Planar RGBA.
    pub fn planar_rgba(ct: ChannelType, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::Red, 1, 1, ct),
                Self::spec(PlaneSemantic::Green, 1, 1, ct),
                Self::spec(PlaneSemantic::Blue, 1, 1, ct),
                Self::spec(PlaneSemantic::Alpha, 1, 1, ct),
            ],
            YuvMatrix::Identity,
            transfer,
        )
    }

    /// Planar Oklab.
    pub fn oklab(ct: ChannelType, transfer: TransferFunction) -> Self {
        Self::from_specs(
            &[
                Self::spec(PlaneSemantic::OklabL, 1, 1, ct),
                Self::spec(PlaneSemantic::OklabA, 1, 1, ct),
                Self::spec(PlaneSemantic::OklabB, 1, 1, ct),
            ],
            YuvMatrix::Identity,
            transfer,
        )
    }

    /// Create from a [`Subsampling`] value.
    pub fn from_subsampling(
        sub: Subsampling,
        ct: ChannelType,
        matrix: YuvMatrix,
        transfer: TransferFunction,
    ) -> Self {
        match sub {
            Subsampling::S444 => Self::ycbcr_444(ct, matrix, transfer),
            Subsampling::S422 => Self::ycbcr_422(ct, matrix, transfer),
            Subsampling::S420 => Self::ycbcr_420(ct, matrix, transfer),
            Subsampling::S411 => Self::ycbcr_411(ct, matrix, transfer),
        }
    }
}

// ---------------------------------------------------------------------------
// Plane mask
// ---------------------------------------------------------------------------

/// Bitfield selecting which planes to process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneMask {
    bits: u8,
}

impl PlaneMask {
    pub const ALL: Self = Self { bits: 0xFF };
    pub const NONE: Self = Self { bits: 0 };
    /// Plane 0 only (typically luma).
    pub const LUMA: Self = Self { bits: 1 };
    /// Planes 1 and 2 (typically chroma Cb/Cr).
    pub const CHROMA: Self = Self { bits: 0b110 };
    /// Plane 3 (typically alpha in a 4-plane layout).
    pub const ALPHA: Self = Self { bits: 0b1000 };

    /// Select a single plane by index.
    #[inline]
    pub const fn single(idx: usize) -> Self {
        Self {
            bits: 1u8 << (idx as u8),
        }
    }

    /// Whether plane `idx` is included.
    #[inline]
    pub const fn includes(self, idx: usize) -> bool {
        self.bits & (1u8 << (idx as u8)) != 0
    }

    /// Union of two masks.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Intersection of two masks.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    /// Number of selected planes.
    #[inline]
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;

    #[test]
    fn channel_type_byte_size() {
        assert_eq!(ChannelType::U8.byte_size(), 1);
        assert_eq!(ChannelType::U16.byte_size(), 2);
        assert_eq!(ChannelType::F16.byte_size(), 2);
        assert_eq!(ChannelType::I16.byte_size(), 2);
        assert_eq!(ChannelType::F32.byte_size(), 4);
    }

    #[test]
    fn descriptor_bytes_per_pixel() {
        assert_eq!(PixelDescriptor::RGB8.bytes_per_pixel(), 3);
        assert_eq!(PixelDescriptor::RGBA8.bytes_per_pixel(), 4);
        assert_eq!(PixelDescriptor::GRAY8.bytes_per_pixel(), 1);
        assert_eq!(PixelDescriptor::RGBAF32.bytes_per_pixel(), 16);
        assert_eq!(PixelDescriptor::GRAYA8.bytes_per_pixel(), 2);
    }

    #[test]
    fn descriptor_has_alpha() {
        assert!(!PixelDescriptor::RGB8.has_alpha());
        assert!(PixelDescriptor::RGBA8.has_alpha());
        assert!(!PixelDescriptor::RGBX8.has_alpha());
        assert!(PixelDescriptor::GRAYA8.has_alpha());
    }

    #[test]
    fn descriptor_is_grayscale() {
        assert!(PixelDescriptor::GRAY8.is_grayscale());
        assert!(PixelDescriptor::GRAYA8.is_grayscale());
        assert!(!PixelDescriptor::RGB8.is_grayscale());
    }

    #[test]
    fn layout_compatible() {
        assert!(PixelDescriptor::RGB8_SRGB.layout_compatible(PixelDescriptor::RGB8));
        assert!(!PixelDescriptor::RGB8.layout_compatible(PixelDescriptor::RGBA8));
    }

    #[test]
    fn pixel_format_roundtrip() {
        let desc = PixelFormat::Rgba8.descriptor();
        assert_eq!(desc.layout, ChannelLayout::Rgba);
        assert_eq!(desc.channel_type, ChannelType::U8);
    }

    #[test]
    fn alpha_mode_none_default() {
        assert_eq!(AlphaMode::default(), AlphaMode::None);
        assert!(!AlphaMode::None.has_alpha());
        assert!(!AlphaMode::Undefined.has_alpha());
        assert!(AlphaMode::Straight.has_alpha());
        assert!(AlphaMode::Premultiplied.has_alpha());
    }

    #[test]
    fn color_primaries_containment() {
        assert!(ColorPrimaries::Bt2020.contains(ColorPrimaries::DisplayP3));
        assert!(ColorPrimaries::Bt2020.contains(ColorPrimaries::Bt709));
        assert!(ColorPrimaries::DisplayP3.contains(ColorPrimaries::Bt709));
        assert!(!ColorPrimaries::Bt709.contains(ColorPrimaries::DisplayP3));
        assert!(!ColorPrimaries::Unknown.contains(ColorPrimaries::Bt709));
    }

    #[test]
    fn descriptor_size() {
        assert!(size_of::<PixelDescriptor>() <= 8);
    }

    #[test]
    fn color_model_channels() {
        assert_eq!(ColorModel::Gray.color_channels(), 1);
        assert_eq!(ColorModel::Rgb.color_channels(), 3);
        assert_eq!(ColorModel::YCbCr.color_channels(), 3);
        assert_eq!(ColorModel::Oklab.color_channels(), 3);
    }

    #[test]
    fn subsampling_factors() {
        assert_eq!(Subsampling::S444.h_factor(), 1);
        assert_eq!(Subsampling::S444.v_factor(), 1);
        assert_eq!(Subsampling::S422.h_factor(), 2);
        assert_eq!(Subsampling::S422.v_factor(), 1);
        assert_eq!(Subsampling::S420.h_factor(), 2);
        assert_eq!(Subsampling::S420.v_factor(), 2);
        assert_eq!(Subsampling::S411.h_factor(), 4);
        assert_eq!(Subsampling::S411.v_factor(), 1);
    }

    #[test]
    fn yuv_matrix_cicp() {
        assert_eq!(YuvMatrix::from_cicp(1), Some(YuvMatrix::Bt709));
        assert_eq!(YuvMatrix::from_cicp(5), Some(YuvMatrix::Bt601));
        assert_eq!(YuvMatrix::from_cicp(9), Some(YuvMatrix::Bt2020));
        assert_eq!(YuvMatrix::from_cicp(99), None);
    }

    #[test]
    fn planar_descriptor_420() {
        let pd =
            PlanarDescriptor::ycbcr_420(ChannelType::U8, YuvMatrix::Bt601, TransferFunction::Srgb);
        assert_eq!(pd.plane_count(), 3);
        assert_eq!(pd.plane_width(0, 1920), 1920); // luma
        assert_eq!(pd.plane_width(1, 1920), 960); // chroma
        assert_eq!(pd.plane_height(0, 1080), 1080);
        assert_eq!(pd.plane_height(1, 1080), 540);
    }

    #[test]
    fn planar_descriptor_411() {
        let pd =
            PlanarDescriptor::ycbcr_411(ChannelType::U8, YuvMatrix::Bt601, TransferFunction::Srgb);
        assert_eq!(pd.plane_count(), 3);
        assert_eq!(pd.plane_width(1, 1920), 480); // quarter horizontal
        assert_eq!(pd.plane_height(1, 1080), 1080); // no vertical subsampling
    }

    #[test]
    fn plane_mask_operations() {
        let luma_chroma = PlaneMask::LUMA.union(PlaneMask::CHROMA);
        assert!(luma_chroma.includes(0));
        assert!(luma_chroma.includes(1));
        assert!(luma_chroma.includes(2));
        assert!(!luma_chroma.includes(3));
        assert_eq!(luma_chroma.count(), 3);
    }

    #[test]
    fn plane_spec_rounding() {
        let spec = PlaneSpec {
            semantic: PlaneSemantic::ChromaCb,
            h_subsample: 2,
            v_subsample: 2,
            channel_type: ChannelType::U8,
        };
        // Odd dimensions round up
        assert_eq!(spec.plane_width(1921), 961);
        assert_eq!(spec.plane_height(1081), 541);
    }

    #[test]
    fn planar_descriptor_size() {
        assert!(size_of::<PlanarDescriptor>() <= 40);
        assert_eq!(size_of::<PlaneSpec>(), 4);
        assert_eq!(size_of::<PlaneMask>(), 1);
    }
}
