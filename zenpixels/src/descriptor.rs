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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
/// Combines a [`PixelFormat`] (physical pixel layout) with transfer function,
/// alpha mode, color primaries, and signal range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PixelDescriptor {
    /// Physical pixel format (channel type + layout as a flat enum).
    pub format: PixelFormat,
    /// Electro-optical transfer function.
    pub transfer: TransferFunction,
    /// Alpha interpretation. `None` = no alpha channel.
    pub alpha: Option<AlphaMode>,
    /// Color primaries (gamut). Defaults to BT.709/sRGB.
    pub primaries: ColorPrimaries,
    /// Signal range (full vs narrow/limited).
    pub signal_range: SignalRange,
}

impl PixelDescriptor {
    // -- Forwarding accessors -------------------------------------------------

    /// The pixel format variant (layout + depth, no transfer or alpha semantics).
    #[inline]
    pub const fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    /// Channel storage type.
    #[inline]
    pub const fn channel_type(&self) -> ChannelType {
        self.format.channel_type()
    }

    /// Alpha interpretation. `None` = no alpha channel.
    #[inline]
    pub const fn alpha(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Transfer function.
    #[inline]
    pub const fn transfer(&self) -> TransferFunction {
        self.transfer
    }

    /// Byte order.
    #[inline]
    pub const fn byte_order(&self) -> ByteOrder {
        self.format.byte_order()
    }

    /// Color model.
    #[inline]
    pub const fn color_model(&self) -> ColorModel {
        self.format.color_model()
    }

    /// Channel layout (derived from the [`PixelFormat`] variant).
    #[inline]
    pub const fn layout(&self) -> ChannelLayout {
        self.format.layout()
    }

    // -- Constructors ---------------------------------------------------------

    /// Create a descriptor with default primaries (BT.709) and full range.
    ///
    /// # Panics
    ///
    /// Panics if the `(channel_type, layout, alpha)` combination has no
    /// corresponding [`PixelFormat`] variant (e.g. `(U16, Bgra, _)`).
    #[inline]
    pub const fn new(
        channel_type: ChannelType,
        layout: ChannelLayout,
        alpha: Option<AlphaMode>,
        transfer: TransferFunction,
    ) -> Self {
        let format = match PixelFormat::from_parts(channel_type, layout, alpha) {
            Some(f) => f,
            None => panic!("unsupported PixelFormat combination"),
        };
        Self {
            format,
            transfer,
            alpha,
            primaries: ColorPrimaries::Bt709,
            signal_range: SignalRange::Full,
        }
    }

    /// Create a descriptor with explicit primaries.
    ///
    /// # Panics
    ///
    /// Panics if the `(channel_type, layout, alpha)` combination has no
    /// corresponding [`PixelFormat`] variant.
    #[inline]
    pub const fn new_full(
        channel_type: ChannelType,
        layout: ChannelLayout,
        alpha: Option<AlphaMode>,
        transfer: TransferFunction,
        primaries: ColorPrimaries,
    ) -> Self {
        let format = match PixelFormat::from_parts(channel_type, layout, alpha) {
            Some(f) => f,
            None => panic!("unsupported PixelFormat combination"),
        };
        Self {
            format,
            transfer,
            alpha,
            primaries,
            signal_range: SignalRange::Full,
        }
    }

    /// Create from a [`PixelFormat`] with default alpha, unknown transfer,
    /// BT.709 primaries, and full range.
    #[inline]
    pub const fn from_pixel_format(format: PixelFormat) -> Self {
        Self {
            format,
            transfer: TransferFunction::Unknown,
            alpha: format.default_alpha(),
            primaries: ColorPrimaries::Bt709,
            signal_range: SignalRange::Full,
        }
    }

    // -- sRGB constants -------------------------------------------------------

    /// 8-bit sRGB RGB.
    pub const RGB8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB RGBA with straight alpha.
    pub const RGBA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB RGB.
    pub const RGB16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB RGBA with straight alpha.
    pub const RGBA16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    /// Linear-light f32 RGB.
    pub const RGBF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    /// Linear-light f32 RGBA with straight alpha.
    pub const RGBAF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    /// 8-bit sRGB grayscale.
    pub const GRAY8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Gray,
        None,
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB grayscale.
    pub const GRAY16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Gray,
        None,
        TransferFunction::Srgb,
    );
    /// Linear-light f32 grayscale.
    pub const GRAYF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear,
    );
    /// 8-bit sRGB grayscale with straight alpha.
    pub const GRAYA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    /// 16-bit sRGB grayscale with straight alpha.
    pub const GRAYA16_SRGB: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    /// Linear-light f32 grayscale with straight alpha.
    pub const GRAYAF32_LINEAR: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    /// 8-bit sRGB BGRA with straight alpha.
    pub const BGRA8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB RGBX (padding byte, not alpha).
    pub const RGBX8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Undefined),
        TransferFunction::Srgb,
    );
    /// 8-bit sRGB BGRX (padding byte, not alpha).
    pub const BGRX8_SRGB: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        Some(AlphaMode::Undefined),
        TransferFunction::Srgb,
    );

    // -- Transfer-agnostic constants ------------------------------------------

    /// 8-bit RGB, transfer unknown.
    pub const RGB8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Unknown,
    );
    /// 8-bit RGBA, transfer unknown.
    pub const RGBA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// 16-bit RGB, transfer unknown.
    pub const RGB16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Unknown,
    );
    /// 16-bit RGBA, transfer unknown.
    pub const RGBA16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// f32 RGB, transfer unknown.
    pub const RGBF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Unknown,
    );
    /// f32 RGBA, transfer unknown.
    pub const RGBAF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// 8-bit grayscale, transfer unknown.
    pub const GRAY8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Gray,
        None,
        TransferFunction::Unknown,
    );
    /// 16-bit grayscale, transfer unknown.
    pub const GRAY16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::Gray,
        None,
        TransferFunction::Unknown,
    );
    /// f32 grayscale, transfer unknown.
    pub const GRAYF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::Gray,
        None,
        TransferFunction::Unknown,
    );
    /// 8-bit grayscale with alpha, transfer unknown.
    pub const GRAYA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// 16-bit grayscale with alpha, transfer unknown.
    pub const GRAYA16: Self = Self::new(
        ChannelType::U16,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// f32 grayscale with alpha, transfer unknown.
    pub const GRAYAF32: Self = Self::new(
        ChannelType::F32,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// 8-bit BGRA, transfer unknown.
    pub const BGRA8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        Some(AlphaMode::Straight),
        TransferFunction::Unknown,
    );
    /// 8-bit RGBX, transfer unknown.
    pub const RGBX8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Undefined),
        TransferFunction::Unknown,
    );
    /// 8-bit BGRX, transfer unknown.
    pub const BGRX8: Self = Self::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        Some(AlphaMode::Undefined),
        TransferFunction::Unknown,
    );

    // -- Oklab constants ------------------------------------------------------

    /// Oklab f32 (L, a, b), transfer unknown.
    pub const OKLABF32: Self = Self {
        format: PixelFormat::OklabF32,
        transfer: TransferFunction::Unknown,
        alpha: None,
        primaries: ColorPrimaries::Bt709,
        signal_range: SignalRange::Full,
    };
    /// Oklab+alpha f32 (L, a, b, alpha), transfer unknown.
    pub const OKLABAF32: Self = Self {
        format: PixelFormat::OklabaF32,
        transfer: TransferFunction::Unknown,
        alpha: Some(AlphaMode::Straight),
        primaries: ColorPrimaries::Bt709,
        signal_range: SignalRange::Full,
    };

    // -- Methods --------------------------------------------------------------

    /// Number of channels.
    #[inline]
    pub const fn channels(self) -> usize {
        self.format.channels()
    }

    /// Bytes per pixel.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        self.format.bytes_per_pixel()
    }

    /// Whether this descriptor has meaningful alpha data.
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(
            self.alpha,
            Some(AlphaMode::Straight) | Some(AlphaMode::Premultiplied) | Some(AlphaMode::Opaque)
        )
    }

    /// Whether this descriptor is grayscale.
    #[inline]
    pub const fn is_grayscale(self) -> bool {
        self.format.is_grayscale()
    }

    /// Whether this descriptor uses BGR byte order.
    #[inline]
    pub const fn is_bgr(self) -> bool {
        matches!(self.format.byte_order(), ByteOrder::Bgr)
    }

    /// Return a copy with a different transfer function.
    #[inline]
    #[must_use]
    pub const fn with_transfer(self, transfer: TransferFunction) -> Self {
        Self { transfer, ..self }
    }

    /// Return a copy with different primaries.
    #[inline]
    #[must_use]
    pub const fn with_primaries(self, primaries: ColorPrimaries) -> Self {
        Self { primaries, ..self }
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub const fn with_alpha(self, alpha: Option<AlphaMode>) -> Self {
        Self { alpha, ..self }
    }

    /// Alias for [`with_alpha`](Self::with_alpha).
    #[inline]
    #[must_use]
    pub const fn with_alpha_mode(self, alpha: Option<AlphaMode>) -> Self {
        self.with_alpha(alpha)
    }

    /// Return a copy with a different signal range.
    #[inline]
    #[must_use]
    pub const fn with_signal_range(self, signal_range: SignalRange) -> Self {
        Self {
            signal_range,
            ..self
        }
    }

    /// Whether this format is fully opaque (no transparency possible).
    ///
    /// Returns `true` when there is no alpha channel (`None`), the alpha
    /// bytes are undefined padding (`Undefined`), or alpha is all-255 (`Opaque`).
    #[inline]
    pub const fn is_opaque(self) -> bool {
        matches!(
            self.alpha,
            None | Some(AlphaMode::Undefined | AlphaMode::Opaque)
        )
    }

    /// Whether this format may contain transparent pixels.
    ///
    /// Returns `true` for [`Straight`](AlphaMode::Straight) and
    /// [`Premultiplied`](AlphaMode::Premultiplied).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn may_have_transparency(self) -> bool {
        matches!(
            self.alpha,
            Some(AlphaMode::Straight | AlphaMode::Premultiplied)
        )
    }

    /// Whether the transfer function is [`Linear`](TransferFunction::Linear).
    #[inline]
    pub const fn is_linear(self) -> bool {
        matches!(self.transfer, TransferFunction::Linear)
    }

    /// Whether the transfer function is [`Unknown`](TransferFunction::Unknown).
    #[inline]
    pub const fn is_unknown_transfer(self) -> bool {
        matches!(self.transfer, TransferFunction::Unknown)
    }

    /// Minimum byte alignment required for the channel type (1, 2, or 4).
    #[inline]
    pub const fn min_alignment(self) -> usize {
        self.format.channel_type().byte_size()
    }

    /// Tightly-packed byte stride for a given width.
    #[inline]
    pub const fn aligned_stride(self, width: u32) -> usize {
        width as usize * self.bytes_per_pixel()
    }

    /// SIMD-friendly byte stride for a given width.
    ///
    /// The stride is a multiple of `lcm(bytes_per_pixel, simd_align)`,
    /// ensuring every row start is both pixel-aligned and SIMD-aligned.
    /// `simd_align` must be a power of 2.
    #[inline]
    pub const fn simd_aligned_stride(self, width: u32, simd_align: usize) -> usize {
        let bpp = self.bytes_per_pixel();
        let raw = width as usize * bpp;
        let align = lcm(bpp, simd_align);
        align_up_general(raw, align)
    }

    /// Whether this descriptor's channel type and layout are compatible with `other`.
    ///
    /// "Compatible" means the raw bytes can be reinterpreted as `other`
    /// without any pixel transformation — same channel type, same layout.
    #[inline]
    pub const fn layout_compatible(self, other: Self) -> bool {
        self.format.channel_type() as u8 == other.format.channel_type() as u8
            && self.layout() as u8 == other.layout() as u8
    }
}

// Alignment helpers.

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

const fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

const fn align_up_general(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let rem = value % align;
    if rem == 0 { value } else { value + align - rem }
}

impl fmt::Display for PixelDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.format,
            self.format.channel_type(),
            self.transfer
        )?;
        if let Some(alpha) = self.alpha
            && alpha.has_alpha()
        {
            write!(f, " alpha={alpha}")?;
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
// Chroma subsampling (planar feature)
// ---------------------------------------------------------------------------

/// Chroma subsampling ratio.
#[cfg(feature = "planar")]
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

#[cfg(feature = "planar")]
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

    /// Map horizontal and vertical subsampling factors to a named pattern.
    ///
    /// Returns `None` for factor combinations that don't match a standard
    /// subsampling pattern.
    #[inline]
    pub const fn from_factors(h: u8, v: u8) -> Option<Self> {
        match (h, v) {
            (1, 1) => Some(Self::S444),
            (2, 1) => Some(Self::S422),
            (2, 2) => Some(Self::S420),
            (4, 1) => Some(Self::S411),
            _ => None,
        }
    }
}

#[cfg(feature = "planar")]
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
// YUV matrix coefficients (planar feature)
// ---------------------------------------------------------------------------

/// YCbCr matrix coefficients for luma/chroma conversion.
#[cfg(feature = "planar")]
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

#[cfg(feature = "planar")]
impl YuvMatrix {
    /// RGB to Y luma coefficients [Kr, Kg, Kb].
    #[allow(unreachable_patterns)]
    #[inline]
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
    #[inline]
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

#[cfg(feature = "planar")]
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
// PixelFormat — flat enum for physical pixel layout
// ---------------------------------------------------------------------------

/// Physical pixel layout for match-based format dispatch.
///
/// Each variant encodes the channel type (U8/U16/F32) and layout (RGB/RGBA/
/// Gray/etc.) in one discriminant. Transfer function and alpha mode live on
/// [`PixelDescriptor`], not here.
///
/// Use this enum when you need exhaustive `match` dispatch over known
/// pixel layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum PixelFormat {
    Rgb8 = 1,
    Rgba8 = 2,
    Rgb16 = 3,
    Rgba16 = 4,
    RgbF32 = 5,
    RgbaF32 = 6,
    Gray8 = 7,
    Gray16 = 8,
    GrayF32 = 9,
    GrayA8 = 10,
    GrayA16 = 11,
    GrayAF32 = 12,
    Bgra8 = 13,
    Rgbx8 = 14,
    Bgrx8 = 15,
    OklabF32 = 16,
    OklabaF32 = 17,
}

impl PixelFormat {
    /// Channel storage type.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn channel_type(self) -> ChannelType {
        match self {
            Self::Rgb8
            | Self::Rgba8
            | Self::Gray8
            | Self::GrayA8
            | Self::Bgra8
            | Self::Rgbx8
            | Self::Bgrx8 => ChannelType::U8,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 | Self::GrayA16 => ChannelType::U16,
            Self::RgbF32
            | Self::RgbaF32
            | Self::GrayF32
            | Self::GrayAF32
            | Self::OklabF32
            | Self::OklabaF32 => ChannelType::F32,
            _ => ChannelType::U8,
        }
    }

    /// Channel layout.
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn layout(self) -> ChannelLayout {
        match self {
            Self::Rgb8 | Self::Rgb16 | Self::RgbF32 => ChannelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 | Self::Rgbx8 => ChannelLayout::Rgba,
            Self::Gray8 | Self::Gray16 | Self::GrayF32 => ChannelLayout::Gray,
            Self::GrayA8 | Self::GrayA16 | Self::GrayAF32 => ChannelLayout::GrayAlpha,
            Self::Bgra8 | Self::Bgrx8 => ChannelLayout::Bgra,
            Self::OklabF32 => ChannelLayout::Oklab,
            Self::OklabaF32 => ChannelLayout::OklabA,
            _ => ChannelLayout::Rgb,
        }
    }

    /// Color model (what the channels represent).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn color_model(self) -> ColorModel {
        match self {
            Self::Gray8
            | Self::Gray16
            | Self::GrayF32
            | Self::GrayA8
            | Self::GrayA16
            | Self::GrayAF32 => ColorModel::Gray,
            Self::OklabF32 | Self::OklabaF32 => ColorModel::Oklab,
            _ => ColorModel::Rgb,
        }
    }

    /// Byte order (Native or BGR).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn byte_order(self) -> ByteOrder {
        match self {
            Self::Bgra8 | Self::Bgrx8 => ByteOrder::Bgr,
            _ => ByteOrder::Native,
        }
    }

    /// Number of channels (including alpha/padding if present).
    #[inline]
    pub const fn channels(self) -> usize {
        self.layout().channels()
    }

    /// Bytes per pixel.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.channel_type().byte_size()
    }

    /// Whether this format has alpha or padding bytes (4th channel).
    #[inline]
    pub const fn has_alpha_bytes(self) -> bool {
        self.layout().has_alpha()
    }

    /// Whether this format is grayscale.
    #[inline]
    pub const fn is_grayscale(self) -> bool {
        matches!(self.color_model(), ColorModel::Gray)
    }

    /// Default alpha mode for this format.
    ///
    /// - Formats with no alpha bytes → `None`
    /// - Formats with padding (Rgbx8, Bgrx8) → `Some(AlphaMode::Undefined)`
    /// - Formats with alpha → `Some(AlphaMode::Straight)`
    #[allow(unreachable_patterns)]
    #[inline]
    pub const fn default_alpha(self) -> Option<AlphaMode> {
        match self {
            Self::Rgb8
            | Self::Rgb16
            | Self::RgbF32
            | Self::Gray8
            | Self::Gray16
            | Self::GrayF32
            | Self::OklabF32 => None,
            Self::Rgbx8 | Self::Bgrx8 => Some(AlphaMode::Undefined),
            _ => Some(AlphaMode::Straight),
        }
    }

    /// Short human-readable name.
    #[allow(unreachable_patterns)]
    #[inline]
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
            Self::OklabF32 => "OklabF32",
            Self::OklabaF32 => "OklabaF32",
            _ => "Unknown",
        }
    }

    /// Resolve a format from channel type, layout, and alpha presence.
    ///
    /// Returns `None` for combinations that have no `PixelFormat` variant
    /// (e.g. `(U16, Bgra, _)`).
    #[inline]
    pub const fn from_parts(
        channel_type: ChannelType,
        layout: ChannelLayout,
        alpha: Option<AlphaMode>,
    ) -> Option<Self> {
        let is_padding = matches!(alpha, Some(AlphaMode::Undefined));
        match (channel_type, layout, is_padding) {
            (ChannelType::U8, ChannelLayout::Rgb, _) => Some(Self::Rgb8),
            (ChannelType::U16, ChannelLayout::Rgb, _) => Some(Self::Rgb16),
            (ChannelType::F32, ChannelLayout::Rgb, _) => Some(Self::RgbF32),

            (ChannelType::U8, ChannelLayout::Rgba, true) => Some(Self::Rgbx8),
            (ChannelType::U8, ChannelLayout::Rgba, false) => Some(Self::Rgba8),
            (ChannelType::U16, ChannelLayout::Rgba, _) => Some(Self::Rgba16),
            (ChannelType::F32, ChannelLayout::Rgba, _) => Some(Self::RgbaF32),

            (ChannelType::U8, ChannelLayout::Gray, _) => Some(Self::Gray8),
            (ChannelType::U16, ChannelLayout::Gray, _) => Some(Self::Gray16),
            (ChannelType::F32, ChannelLayout::Gray, _) => Some(Self::GrayF32),

            (ChannelType::U8, ChannelLayout::GrayAlpha, _) => Some(Self::GrayA8),
            (ChannelType::U16, ChannelLayout::GrayAlpha, _) => Some(Self::GrayA16),
            (ChannelType::F32, ChannelLayout::GrayAlpha, _) => Some(Self::GrayAF32),

            (ChannelType::U8, ChannelLayout::Bgra, true) => Some(Self::Bgrx8),
            (ChannelType::U8, ChannelLayout::Bgra, false) => Some(Self::Bgra8),

            (ChannelType::F32, ChannelLayout::Oklab, _) => Some(Self::OklabF32),
            (ChannelType::F32, ChannelLayout::OklabA, _) => Some(Self::OklabaF32),

            _ => None,
        }
    }

    /// Base descriptor with `Unknown` transfer and BT.709 primaries.
    #[allow(unreachable_patterns)]
    #[inline]
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
            Self::OklabF32 => PixelDescriptor::OKLABF32,
            Self::OklabaF32 => PixelDescriptor::OKLABAF32,
            _ => PixelDescriptor::RGB8,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Deprecated alias — use [`PixelFormat`] instead.
#[deprecated(note = "renamed to PixelFormat")]
pub type InterleaveFormat = PixelFormat;

// ---------------------------------------------------------------------------
// Multi-plane types (planar feature)
// ---------------------------------------------------------------------------

/// Semantic label for a plane in a multi-plane image.
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum PlaneSemantic {
    /// Luma (Y) — brightness in YCbCr color spaces.
    Luma = 0,
    /// Chroma blue-difference (Cb / U).
    ChromaCb = 1,
    /// Chroma red-difference (Cr / V).
    ChromaCr = 2,
    /// Red channel.
    Red = 3,
    /// Green channel.
    Green = 4,
    /// Blue channel.
    Blue = 5,
    /// Alpha (transparency) channel.
    Alpha = 6,
    /// Depth map.
    Depth = 7,
    /// Gain map (e.g., Ultra HDR).
    GainMap = 8,
    /// Grayscale (single-channel luminance, not part of YCbCr).
    Gray = 9,
    /// Oklab lightness (L).
    OklabL = 10,
    /// Oklab green-red axis (a).
    OklabA = 11,
    /// Oklab blue-yellow axis (b).
    OklabB = 12,
}

#[cfg(feature = "planar")]
impl PlaneSemantic {
    /// Whether this semantic represents a luminance-like channel.
    #[inline]
    pub const fn is_luminance(self) -> bool {
        matches!(self, Self::Luma | Self::Gray | Self::OklabL)
    }

    /// Whether this semantic represents a chroma channel.
    #[inline]
    pub const fn is_chroma(self) -> bool {
        matches!(
            self,
            Self::ChromaCb | Self::ChromaCr | Self::OklabA | Self::OklabB
        )
    }

    /// Whether this semantic is an RGB color channel.
    #[inline]
    pub const fn is_rgb(self) -> bool {
        matches!(self, Self::Red | Self::Green | Self::Blue)
    }

    /// Whether this semantic is the alpha channel.
    #[inline]
    pub const fn is_alpha(self) -> bool {
        matches!(self, Self::Alpha)
    }
}

#[cfg(feature = "planar")]
impl fmt::Display for PlaneSemantic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Luma => f.write_str("Luma"),
            Self::ChromaCb => f.write_str("Cb"),
            Self::ChromaCr => f.write_str("Cr"),
            Self::Red => f.write_str("R"),
            Self::Green => f.write_str("G"),
            Self::Blue => f.write_str("B"),
            Self::Alpha => f.write_str("A"),
            Self::Depth => f.write_str("Depth"),
            Self::GainMap => f.write_str("GainMap"),
            Self::Gray => f.write_str("Gray"),
            Self::OklabL => f.write_str("Oklab.L"),
            Self::OklabA => f.write_str("Oklab.a"),
            Self::OklabB => f.write_str("Oklab.b"),
        }
    }
}

// ---------------------------------------------------------------------------
// PlaneDescriptor — per-plane metadata (no pixel data)
// ---------------------------------------------------------------------------

/// Metadata describing a single plane in a multi-plane image.
///
/// This is a pure descriptor — no pixel data, no heap allocation.
/// Subsample factors are explicit per-plane rather than inferred from
/// dimension ratios, because strip-based pipelines don't carry global
/// image dimensions.
///
/// # Examples
///
/// ```
/// use zenpixels::{PlaneDescriptor, PlaneSemantic, ChannelType};
///
/// let luma = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
/// assert!(!luma.is_subsampled());
/// assert_eq!(luma.plane_width(1920), 1920);
///
/// let chroma = PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::F32)
///     .with_subsampling(2, 2); // 4:2:0
/// assert!(chroma.is_subsampled());
/// assert_eq!(chroma.plane_width(1920), 960);
/// assert_eq!(chroma.plane_height(1080), 540);
/// ```
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneDescriptor {
    /// What this plane represents.
    pub semantic: PlaneSemantic,
    /// Storage type for each sample in this plane.
    pub channel_type: ChannelType,
    /// Horizontal subsampling factor (1 = full resolution, 2 = half, 4 = quarter).
    pub h_subsample: u8,
    /// Vertical subsampling factor (1 = full resolution, 2 = half).
    pub v_subsample: u8,
}

#[cfg(feature = "planar")]
impl PlaneDescriptor {
    /// Create a full-resolution plane descriptor (no subsampling).
    #[inline]
    pub const fn new(semantic: PlaneSemantic, channel_type: ChannelType) -> Self {
        Self {
            semantic,
            channel_type,
            h_subsample: 1,
            v_subsample: 1,
        }
    }

    /// Builder: set subsampling factors.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `h` and `v` are non-zero powers of two.
    #[inline]
    #[must_use]
    pub const fn with_subsampling(mut self, h: u8, v: u8) -> Self {
        debug_assert!(
            h > 0 && h.is_power_of_two(),
            "h_subsample must be a power of 2"
        );
        debug_assert!(
            v > 0 && v.is_power_of_two(),
            "v_subsample must be a power of 2"
        );
        self.h_subsample = h;
        self.v_subsample = v;
        self
    }

    /// Compute the width of this plane given a reference (luma) width.
    ///
    /// Uses ceiling division so subsampled planes always cover the full image.
    #[inline]
    pub const fn plane_width(&self, ref_width: u32) -> u32 {
        ref_width.div_ceil(self.h_subsample as u32)
    }

    /// Compute the height of this plane given a reference (luma) height.
    ///
    /// Uses ceiling division so subsampled planes always cover the full image.
    #[inline]
    pub const fn plane_height(&self, ref_height: u32) -> u32 {
        ref_height.div_ceil(self.v_subsample as u32)
    }

    /// Whether this plane is subsampled (either axis).
    #[inline]
    pub const fn is_subsampled(&self) -> bool {
        self.h_subsample > 1 || self.v_subsample > 1
    }

    /// Bytes per sample in this plane.
    #[inline]
    pub const fn bytes_per_sample(&self) -> usize {
        self.channel_type.byte_size()
    }
}

#[cfg(feature = "planar")]
impl fmt::Display for PlaneDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.semantic, self.channel_type)?;
        if self.is_subsampled() {
            write!(f, " (1/{}×1/{})", self.h_subsample, self.v_subsample)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PlaneMask — bitmask for plane selection
// ---------------------------------------------------------------------------

/// Bitmask selecting which planes to process in a multi-plane operation.
///
/// Supports up to 8 planes. Operations that only affect certain channels
/// (e.g., luma sharpening) use this to skip untouched planes.
///
/// # Examples
///
/// ```
/// use zenpixels::PlaneMask;
///
/// let mask = PlaneMask::LUMA.union(PlaneMask::ALPHA);
/// assert!(mask.includes(0));  // luma
/// assert!(!mask.includes(1)); // chroma Cb
/// assert!(mask.includes(3));  // alpha
/// assert_eq!(mask.count(), 2);
/// ```
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneMask {
    bits: u8,
}

#[cfg(feature = "planar")]
impl PlaneMask {
    /// All planes (bits 0–7).
    pub const ALL: Self = Self { bits: 0xFF };
    /// No planes.
    pub const NONE: Self = Self { bits: 0 };
    /// Plane 0 only (luma / lightness / red / gray).
    pub const LUMA: Self = Self { bits: 0b0001 };
    /// Planes 1 and 2 (chroma Cb + Cr, or Oklab a + b).
    pub const CHROMA: Self = Self { bits: 0b0110 };
    /// Plane 3 (alpha).
    pub const ALPHA: Self = Self { bits: 0b1000 };

    /// Mask for a single plane by index (0–7).
    #[inline]
    pub const fn single(idx: usize) -> Self {
        debug_assert!(idx < 8, "PlaneMask supports at most 8 planes");
        Self {
            bits: 1u8 << (idx as u8),
        }
    }

    /// Whether the plane at `idx` is included in this mask.
    #[inline]
    pub const fn includes(&self, idx: usize) -> bool {
        if idx >= 8 {
            return false;
        }
        (self.bits >> (idx as u8)) & 1 != 0
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

    /// Number of planes selected.
    #[inline]
    pub const fn count(&self) -> u32 {
        self.bits.count_ones()
    }

    /// Whether no planes are selected.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// The raw bitmask value.
    #[inline]
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// Construct from a raw bitmask value.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self { bits }
    }
}

#[cfg(feature = "planar")]
impl fmt::Display for PlaneMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::ALL {
            return f.write_str("ALL");
        }
        if self.is_empty() {
            return f.write_str("NONE");
        }
        let mut first = true;
        for i in 0..8 {
            if self.includes(i) {
                if !first {
                    f.write_str("|")?;
                }
                write!(f, "{i}")?;
                first = false;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plane — buffer + semantic (legacy, retained for backward compat)
// ---------------------------------------------------------------------------

/// A single plane with its semantic label.
///
/// Each plane is an independent [`PixelBuffer`](crate::buffer::PixelBuffer)
/// (opaque gray channel) that can be a different size from other planes
/// (e.g., subsampled chroma).
///
/// **Prefer [`PlaneLayout`] + separate buffers** for new code. This type
/// is retained for backward compatibility.
#[cfg(feature = "planar")]
pub struct Plane {
    /// The pixel data for this plane.
    pub buffer: crate::buffer::PixelBuffer,
    /// What this plane represents.
    pub semantic: PlaneSemantic,
}

/// How planes in a multi-plane image relate to each other.
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PlaneRelationship {
    /// Independent channels (e.g., split R, G, B).
    Independent,
    /// YCbCr with a specific matrix. Subsampling per-plane in [`PlaneDescriptor`].
    YCbCr {
        /// The YCbCr matrix coefficients.
        matrix: YuvMatrix,
    },
    /// Oklab perceptual color space (L/a/b). Fixed transform, no matrix parameter.
    Oklab,
    /// Gain map (base rendition + gain plane).
    GainMap,
}

// ---------------------------------------------------------------------------
// PlaneLayout — complete spatial layout descriptor
// ---------------------------------------------------------------------------

/// Complete layout of a multi-plane image.
///
/// Separates the *metadata* (how many planes, what they represent, their
/// subsampling) from any pixel buffers. Every consumer that needs to know
/// about planar organization works with `PlaneLayout`; only code that
/// allocates or reads pixels needs actual buffers.
///
/// # Examples
///
/// ```
/// use zenpixels::{PlaneLayout, ChannelType, PlaneSemantic};
///
/// let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
/// assert!(layout.is_planar());
/// assert!(layout.is_ycbcr());
/// assert!(layout.has_subsampling());
/// assert_eq!(layout.plane_count(), 3);
///
/// // Check plane semantics
/// let planes = layout.planes();
/// assert_eq!(planes[0].semantic, PlaneSemantic::Luma);
/// assert_eq!(planes[1].semantic, PlaneSemantic::ChromaCb);
/// assert!(!planes[0].is_subsampled());
/// assert!(planes[1].is_subsampled());
/// ```
#[cfg(feature = "planar")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneLayout {
    /// Interleaved pixel data (channels packed per-pixel).
    Interleaved {
        /// Number of channels per pixel (e.g., 3 for RGB, 4 for RGBA).
        channels: u8,
    },
    /// Planar pixel data (one buffer per plane).
    Planar {
        /// Descriptor for each plane.
        planes: alloc::vec::Vec<PlaneDescriptor>,
        /// How the planes relate to each other.
        relationship: PlaneRelationship,
    },
}

#[cfg(feature = "planar")]
impl PlaneLayout {
    // --- Factory methods ---

    /// YCbCr 4:4:4 (no subsampling).
    pub fn ycbcr_444(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// YCbCr 4:4:4 with a specific matrix.
    pub fn ycbcr_444_matrix(ct: ChannelType, matrix: YuvMatrix) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct),
            ],
            relationship: PlaneRelationship::YCbCr { matrix },
        }
    }

    /// YCbCr 4:2:2 (horizontal half chroma).
    pub fn ycbcr_422(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct).with_subsampling(2, 1),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct).with_subsampling(2, 1),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// YCbCr 4:2:0 (half chroma in both axes).
    pub fn ycbcr_420(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct).with_subsampling(2, 2),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct).with_subsampling(2, 2),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// Planar RGB (3 independent planes, no subsampling).
    pub fn rgb(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Red, ct),
                PlaneDescriptor::new(PlaneSemantic::Green, ct),
                PlaneDescriptor::new(PlaneSemantic::Blue, ct),
            ],
            relationship: PlaneRelationship::Independent,
        }
    }

    /// Planar RGBA (4 independent planes, no subsampling).
    pub fn rgba(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Red, ct),
                PlaneDescriptor::new(PlaneSemantic::Green, ct),
                PlaneDescriptor::new(PlaneSemantic::Blue, ct),
                PlaneDescriptor::new(PlaneSemantic::Alpha, ct),
            ],
            relationship: PlaneRelationship::Independent,
        }
    }

    /// Oklab (L/a/b, 3 planes, no subsampling).
    pub fn oklab(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::OklabL, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabA, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabB, ct),
            ],
            relationship: PlaneRelationship::Oklab,
        }
    }

    /// Oklab with alpha (L/a/b/A, 4 planes, no subsampling).
    pub fn oklab_alpha(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::OklabL, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabA, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabB, ct),
                PlaneDescriptor::new(PlaneSemantic::Alpha, ct),
            ],
            relationship: PlaneRelationship::Oklab,
        }
    }

    /// Grayscale (single plane, no subsampling).
    pub fn gray(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![PlaneDescriptor::new(PlaneSemantic::Gray, ct)],
            relationship: PlaneRelationship::Independent,
        }
    }

    // --- Queries ---

    /// Number of planes (or interleaved channels).
    #[inline]
    pub fn plane_count(&self) -> usize {
        match self {
            Self::Interleaved { channels } => *channels as usize,
            Self::Planar { planes, .. } => planes.len(),
        }
    }

    /// Plane descriptors. Empty slice for interleaved layout.
    #[inline]
    pub fn planes(&self) -> &[PlaneDescriptor] {
        match self {
            Self::Interleaved { .. } => &[],
            Self::Planar { planes, .. } => planes,
        }
    }

    /// Index of the luma/lightness plane, if any.
    pub fn luma_plane_index(&self) -> Option<usize> {
        match self {
            Self::Interleaved { .. } => None,
            Self::Planar { planes, .. } => planes.iter().position(|p| p.semantic.is_luminance()),
        }
    }

    /// Whether any plane is subsampled.
    pub fn has_subsampling(&self) -> bool {
        match self {
            Self::Interleaved { .. } => false,
            Self::Planar { planes, .. } => planes.iter().any(|p| p.is_subsampled()),
        }
    }

    /// Whether this is a YCbCr layout.
    pub fn is_ycbcr(&self) -> bool {
        matches!(
            self,
            Self::Planar {
                relationship: PlaneRelationship::YCbCr { .. },
                ..
            }
        )
    }

    /// Whether this is an Oklab layout.
    pub fn is_oklab(&self) -> bool {
        matches!(
            self,
            Self::Planar {
                relationship: PlaneRelationship::Oklab,
                ..
            }
        )
    }

    /// Whether this layout is planar (not interleaved).
    #[inline]
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::Planar { .. })
    }

    /// The plane relationship, if planar.
    pub fn relationship(&self) -> Option<PlaneRelationship> {
        match self {
            Self::Interleaved { .. } => None,
            Self::Planar { relationship, .. } => Some(*relationship),
        }
    }

    /// Maximum vertical subsampling factor across all planes.
    ///
    /// Returns 1 for interleaved or non-subsampled layouts.
    pub fn max_v_subsample(&self) -> u8 {
        match self {
            Self::Interleaved { .. } => 1,
            Self::Planar { planes, .. } => planes.iter().map(|p| p.v_subsample).max().unwrap_or(1),
        }
    }

    /// Maximum horizontal subsampling factor across all planes.
    ///
    /// Returns 1 for interleaved or non-subsampled layouts.
    pub fn max_h_subsample(&self) -> u8 {
        match self {
            Self::Interleaved { .. } => 1,
            Self::Planar { planes, .. } => planes.iter().map(|p| p.h_subsample).max().unwrap_or(1),
        }
    }

    /// Build a [`PlaneMask`] from a predicate on plane semantics.
    ///
    /// Returns [`PlaneMask::NONE`] for interleaved layouts.
    pub fn mask_where(&self, f: impl Fn(PlaneSemantic) -> bool) -> PlaneMask {
        match self {
            Self::Interleaved { .. } => PlaneMask::NONE,
            Self::Planar { planes, .. } => {
                let mut bits = 0u8;
                for (i, p) in planes.iter().enumerate() {
                    if i < 8 && f(p.semantic) {
                        bits |= 1 << (i as u8);
                    }
                }
                PlaneMask::from_bits(bits)
            }
        }
    }

    /// Mask of luminance-like planes (Luma, Gray, OklabL).
    pub fn luma_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_luminance())
    }

    /// Mask of chroma planes (Cb, Cr, OklabA, OklabB).
    pub fn chroma_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_chroma())
    }

    /// Mask of alpha planes.
    pub fn alpha_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_alpha())
    }
}

#[cfg(feature = "planar")]
impl fmt::Display for PlaneLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interleaved { channels } => write!(f, "Interleaved({channels}ch)"),
            Self::Planar {
                planes,
                relationship,
            } => {
                write!(f, "{relationship:?}[")?;
                for (i, p) in planes.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{p}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// A multi-plane image where each plane is an independent pixel buffer.
///
/// Combines a [`PlaneLayout`] (metadata describing plane count, semantics,
/// subsampling, and relationship) with actual pixel buffers. Each buffer
/// corresponds to one plane in the layout.
///
/// # Examples
///
/// ```
/// use zenpixels::{MultiPlaneImage, PlaneLayout, ChannelType, PixelBuffer, PixelDescriptor};
///
/// let layout = PlaneLayout::ycbcr_444(ChannelType::U8);
/// let y = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
/// let cb = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
/// let cr = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
///
/// let img = MultiPlaneImage::new(layout, vec![y, cb, cr]);
/// assert_eq!(img.plane_count(), 3);
/// assert!(img.layout().is_ycbcr());
/// ```
#[cfg(feature = "planar")]
pub struct MultiPlaneImage {
    layout: PlaneLayout,
    buffers: alloc::vec::Vec<crate::buffer::PixelBuffer>,
    origin: Option<alloc::sync::Arc<crate::color::ColorContext>>,
}

#[cfg(feature = "planar")]
impl MultiPlaneImage {
    /// Create a new multi-plane image from a layout and corresponding buffers.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the number of buffers matches the planar plane count.
    pub fn new(layout: PlaneLayout, buffers: alloc::vec::Vec<crate::buffer::PixelBuffer>) -> Self {
        debug_assert!(
            layout.is_planar(),
            "MultiPlaneImage requires a Planar layout"
        );
        debug_assert_eq!(
            layout.plane_count(),
            buffers.len(),
            "buffer count ({}) must match plane count ({})",
            buffers.len(),
            layout.plane_count(),
        );
        Self {
            layout,
            buffers,
            origin: None,
        }
    }

    /// Attach a color context.
    pub fn with_origin(mut self, ctx: alloc::sync::Arc<crate::color::ColorContext>) -> Self {
        self.origin = Some(ctx);
        self
    }

    /// The layout describing this image's plane organization.
    #[inline]
    pub fn layout(&self) -> &PlaneLayout {
        &self.layout
    }

    /// Number of planes.
    #[inline]
    pub fn plane_count(&self) -> usize {
        self.buffers.len()
    }

    /// Access a single buffer by index.
    #[inline]
    pub fn buffer(&self, idx: usize) -> Option<&crate::buffer::PixelBuffer> {
        self.buffers.get(idx)
    }

    /// Access a single buffer mutably by index.
    #[inline]
    pub fn buffer_mut(&mut self, idx: usize) -> Option<&mut crate::buffer::PixelBuffer> {
        self.buffers.get_mut(idx)
    }

    /// Access all buffers.
    #[inline]
    pub fn buffers(&self) -> &[crate::buffer::PixelBuffer] {
        &self.buffers
    }

    /// Access all buffers mutably.
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [crate::buffer::PixelBuffer] {
        &mut self.buffers
    }

    /// The optional color context shared across all planes.
    #[inline]
    pub fn origin(&self) -> Option<&alloc::sync::Arc<crate::color::ColorContext>> {
        self.origin.as_ref()
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
    fn pixel_format_descriptor_roundtrip() {
        let desc = PixelFormat::Rgba8.descriptor();
        assert_eq!(desc.layout(), ChannelLayout::Rgba);
        assert_eq!(desc.channel_type(), ChannelType::U8);
    }

    #[test]
    fn pixel_format_enum_basics() {
        assert_eq!(PixelFormat::Rgb8.channels(), 3);
        assert_eq!(PixelFormat::Rgba8.channels(), 4);
        assert!(PixelFormat::Rgba8.has_alpha_bytes());
        assert!(!PixelFormat::Rgb8.has_alpha_bytes());
        assert_eq!(PixelFormat::RgbF32.bytes_per_pixel(), 12);
        assert_eq!(PixelFormat::RgbaF32.bytes_per_pixel(), 16);
        assert_eq!(PixelFormat::Gray8.channels(), 1);
        assert!(PixelFormat::Gray8.is_grayscale());
        assert!(!PixelFormat::Rgb8.is_grayscale());
        assert_eq!(PixelFormat::Bgra8.byte_order(), ByteOrder::Bgr);
        assert_eq!(PixelFormat::Rgb8.byte_order(), ByteOrder::Native);
    }

    #[test]
    fn pixel_format_enum_size() {
        // Single-byte discriminant — much smaller than old 5-field struct.
        assert!(size_of::<PixelFormat>() <= 2);
    }

    #[test]
    fn pixel_format_from_parts_roundtrip() {
        let fmt = PixelFormat::Rgba8;
        let rebuilt =
            PixelFormat::from_parts(fmt.channel_type(), fmt.layout(), fmt.default_alpha());
        assert_eq!(rebuilt, Some(fmt));

        let fmt2 = PixelFormat::Bgra8;
        let rebuilt2 =
            PixelFormat::from_parts(fmt2.channel_type(), fmt2.layout(), fmt2.default_alpha());
        assert_eq!(rebuilt2, Some(fmt2));

        let fmt3 = PixelFormat::Gray8;
        let rebuilt3 =
            PixelFormat::from_parts(fmt3.channel_type(), fmt3.layout(), fmt3.default_alpha());
        assert_eq!(rebuilt3, Some(fmt3));
    }

    #[test]
    fn alpha_mode_semantics() {
        // None (Option) = no alpha channel
        assert!(!PixelDescriptor::RGB8.has_alpha());
        // Undefined = padding bytes, not real alpha
        assert!(!AlphaMode::Undefined.has_alpha());
        // Straight and Premultiplied = real alpha
        assert!(AlphaMode::Straight.has_alpha());
        assert!(AlphaMode::Premultiplied.has_alpha());
        assert!(AlphaMode::Opaque.has_alpha());
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
        // PixelFormat (1 byte enum) + transfer (1) + alpha (2) + primaries (1) + signal_range (1) = ~6
        assert!(size_of::<PixelDescriptor>() <= 8);
    }

    #[test]
    fn color_model_channels() {
        assert_eq!(ColorModel::Gray.color_channels(), 1);
        assert_eq!(ColorModel::Rgb.color_channels(), 3);
        assert_eq!(ColorModel::YCbCr.color_channels(), 3);
        assert_eq!(ColorModel::Oklab.color_channels(), 3);
    }

    #[cfg(feature = "planar")]
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

    #[cfg(feature = "planar")]
    #[test]
    fn yuv_matrix_cicp() {
        assert_eq!(YuvMatrix::from_cicp(1), Some(YuvMatrix::Bt709));
        assert_eq!(YuvMatrix::from_cicp(5), Some(YuvMatrix::Bt601));
        assert_eq!(YuvMatrix::from_cicp(9), Some(YuvMatrix::Bt2020));
        assert_eq!(YuvMatrix::from_cicp(99), None);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn subsampling_from_factors() {
        assert_eq!(Subsampling::from_factors(1, 1), Some(Subsampling::S444));
        assert_eq!(Subsampling::from_factors(2, 1), Some(Subsampling::S422));
        assert_eq!(Subsampling::from_factors(2, 2), Some(Subsampling::S420));
        assert_eq!(Subsampling::from_factors(4, 1), Some(Subsampling::S411));
        assert_eq!(Subsampling::from_factors(3, 1), None);
        assert_eq!(Subsampling::from_factors(1, 2), None);
    }

    // --- PlaneSemantic tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn plane_semantic_classification() {
        // Luminance-like
        assert!(PlaneSemantic::Luma.is_luminance());
        assert!(PlaneSemantic::Gray.is_luminance());
        assert!(PlaneSemantic::OklabL.is_luminance());
        assert!(!PlaneSemantic::Red.is_luminance());

        // Chroma
        assert!(PlaneSemantic::ChromaCb.is_chroma());
        assert!(PlaneSemantic::ChromaCr.is_chroma());
        assert!(PlaneSemantic::OklabA.is_chroma());
        assert!(PlaneSemantic::OklabB.is_chroma());
        assert!(!PlaneSemantic::Luma.is_chroma());

        // RGB
        assert!(PlaneSemantic::Red.is_rgb());
        assert!(PlaneSemantic::Green.is_rgb());
        assert!(PlaneSemantic::Blue.is_rgb());
        assert!(!PlaneSemantic::Luma.is_rgb());

        // Alpha
        assert!(PlaneSemantic::Alpha.is_alpha());
        assert!(!PlaneSemantic::Luma.is_alpha());
        assert!(!PlaneSemantic::Depth.is_alpha());
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_semantic_display() {
        assert_eq!(format!("{}", PlaneSemantic::Luma), "Luma");
        assert_eq!(format!("{}", PlaneSemantic::ChromaCb), "Cb");
        assert_eq!(format!("{}", PlaneSemantic::Gray), "Gray");
        assert_eq!(format!("{}", PlaneSemantic::OklabL), "Oklab.L");
    }

    // --- PlaneDescriptor tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn plane_descriptor_full_resolution() {
        let d = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
        assert_eq!(d.semantic, PlaneSemantic::Luma);
        assert_eq!(d.channel_type, ChannelType::F32);
        assert!(!d.is_subsampled());
        assert_eq!(d.h_subsample, 1);
        assert_eq!(d.v_subsample, 1);
        assert_eq!(d.plane_width(1920), 1920);
        assert_eq!(d.plane_height(1080), 1080);
        assert_eq!(d.bytes_per_sample(), 4);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_descriptor_subsampled() {
        let d =
            PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::U8).with_subsampling(2, 2);
        assert!(d.is_subsampled());
        assert_eq!(d.plane_width(1920), 960);
        assert_eq!(d.plane_height(1080), 540);
        // Odd dimensions use ceiling division
        assert_eq!(d.plane_width(1921), 961);
        assert_eq!(d.plane_height(1081), 541);
        assert_eq!(d.bytes_per_sample(), 1);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_descriptor_quarter_h() {
        let d =
            PlaneDescriptor::new(PlaneSemantic::ChromaCr, ChannelType::U16).with_subsampling(4, 1);
        assert!(d.is_subsampled());
        assert_eq!(d.plane_width(1920), 480);
        assert_eq!(d.plane_height(1080), 1080);
        assert_eq!(d.bytes_per_sample(), 2);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_descriptor_display() {
        let d = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
        assert_eq!(format!("{d}"), "Luma:F32");

        let d =
            PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::U8).with_subsampling(2, 2);
        assert_eq!(format!("{d}"), "Cb:U8 (1/2\u{00d7}1/2)");
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_descriptor_size() {
        // Should be small — semantic (1) + channel_type (1) + h (1) + v (1) = 4
        assert!(size_of::<PlaneDescriptor>() <= 4);
    }

    // --- PlaneMask tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_constants() {
        assert!(PlaneMask::ALL.includes(0));
        assert!(PlaneMask::ALL.includes(7));
        assert!(!PlaneMask::NONE.includes(0));
        assert!(PlaneMask::NONE.is_empty());
        assert!(PlaneMask::LUMA.includes(0));
        assert!(!PlaneMask::LUMA.includes(1));
        assert!(PlaneMask::CHROMA.includes(1));
        assert!(PlaneMask::CHROMA.includes(2));
        assert!(!PlaneMask::CHROMA.includes(0));
        assert!(PlaneMask::ALPHA.includes(3));
        assert!(!PlaneMask::ALPHA.includes(0));
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_single() {
        let m = PlaneMask::single(5);
        assert!(m.includes(5));
        assert!(!m.includes(4));
        assert_eq!(m.count(), 1);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_union_intersection() {
        let luma_alpha = PlaneMask::LUMA.union(PlaneMask::ALPHA);
        assert!(luma_alpha.includes(0));
        assert!(luma_alpha.includes(3));
        assert!(!luma_alpha.includes(1));
        assert_eq!(luma_alpha.count(), 2);

        let intersect = luma_alpha.intersection(PlaneMask::LUMA);
        assert!(intersect.includes(0));
        assert!(!intersect.includes(3));
        assert_eq!(intersect.count(), 1);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_out_of_range() {
        assert!(!PlaneMask::ALL.includes(8));
        assert!(!PlaneMask::ALL.includes(100));
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_bits_roundtrip() {
        let m = PlaneMask::LUMA.union(PlaneMask::CHROMA);
        let bits = m.bits();
        assert_eq!(PlaneMask::from_bits(bits), m);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_mask_display() {
        assert_eq!(format!("{}", PlaneMask::ALL), "ALL");
        assert_eq!(format!("{}", PlaneMask::NONE), "NONE");
        assert_eq!(format!("{}", PlaneMask::LUMA), "0");
        assert_eq!(
            format!("{}", PlaneMask::LUMA.union(PlaneMask::ALPHA)),
            "0|3"
        );
    }

    // --- PlaneLayout tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_interleaved() {
        let layout = PlaneLayout::Interleaved { channels: 4 };
        assert!(!layout.is_planar());
        assert_eq!(layout.plane_count(), 4);
        assert!(layout.planes().is_empty());
        assert!(!layout.has_subsampling());
        assert!(!layout.is_ycbcr());
        assert!(!layout.is_oklab());
        assert_eq!(layout.luma_plane_index(), None);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_ycbcr_444() {
        let layout = PlaneLayout::ycbcr_444(ChannelType::F32);
        assert!(layout.is_planar());
        assert!(layout.is_ycbcr());
        assert!(!layout.is_oklab());
        assert_eq!(layout.plane_count(), 3);
        assert!(!layout.has_subsampling());
        assert_eq!(layout.luma_plane_index(), Some(0));
        assert_eq!(layout.max_v_subsample(), 1);
        assert_eq!(layout.max_h_subsample(), 1);

        let planes = layout.planes();
        assert_eq!(planes[0].semantic, PlaneSemantic::Luma);
        assert_eq!(planes[1].semantic, PlaneSemantic::ChromaCb);
        assert_eq!(planes[2].semantic, PlaneSemantic::ChromaCr);
        assert_eq!(planes[0].channel_type, ChannelType::F32);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_ycbcr_420() {
        let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
        assert!(layout.is_planar());
        assert!(layout.is_ycbcr());
        assert!(layout.has_subsampling());
        assert_eq!(layout.max_v_subsample(), 2);
        assert_eq!(layout.max_h_subsample(), 2);

        let planes = layout.planes();
        assert!(!planes[0].is_subsampled());
        assert!(planes[1].is_subsampled());
        assert_eq!(planes[1].h_subsample, 2);
        assert_eq!(planes[1].v_subsample, 2);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_ycbcr_422() {
        let layout = PlaneLayout::ycbcr_422(ChannelType::U8);
        assert!(layout.has_subsampling());
        assert_eq!(layout.max_v_subsample(), 1);
        assert_eq!(layout.max_h_subsample(), 2);

        let planes = layout.planes();
        assert_eq!(planes[1].h_subsample, 2);
        assert_eq!(planes[1].v_subsample, 1);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_ycbcr_444_matrix() {
        let layout = PlaneLayout::ycbcr_444_matrix(ChannelType::U8, YuvMatrix::Bt709);
        assert_eq!(
            layout.relationship(),
            Some(PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt709
            })
        );
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_oklab() {
        let layout = PlaneLayout::oklab(ChannelType::F32);
        assert!(layout.is_planar());
        assert!(layout.is_oklab());
        assert!(!layout.is_ycbcr());
        assert_eq!(layout.plane_count(), 3);
        assert!(!layout.has_subsampling());
        assert_eq!(layout.luma_plane_index(), Some(0));

        let planes = layout.planes();
        assert_eq!(planes[0].semantic, PlaneSemantic::OklabL);
        assert_eq!(planes[1].semantic, PlaneSemantic::OklabA);
        assert_eq!(planes[2].semantic, PlaneSemantic::OklabB);
        assert_eq!(planes[0].channel_type, ChannelType::F32);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_oklab_alpha() {
        let layout = PlaneLayout::oklab_alpha(ChannelType::F32);
        assert!(layout.is_oklab());
        assert_eq!(layout.plane_count(), 4);
        assert_eq!(layout.planes()[3].semantic, PlaneSemantic::Alpha);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_rgb() {
        let layout = PlaneLayout::rgb(ChannelType::U8);
        assert!(layout.is_planar());
        assert!(!layout.is_ycbcr());
        assert!(!layout.is_oklab());
        assert_eq!(layout.plane_count(), 3);
        assert_eq!(layout.relationship(), Some(PlaneRelationship::Independent));

        let planes = layout.planes();
        assert_eq!(planes[0].semantic, PlaneSemantic::Red);
        assert_eq!(planes[1].semantic, PlaneSemantic::Green);
        assert_eq!(planes[2].semantic, PlaneSemantic::Blue);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_rgba() {
        let layout = PlaneLayout::rgba(ChannelType::U8);
        assert_eq!(layout.plane_count(), 4);
        assert_eq!(layout.planes()[3].semantic, PlaneSemantic::Alpha);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_gray() {
        let layout = PlaneLayout::gray(ChannelType::U8);
        assert!(layout.is_planar());
        assert_eq!(layout.plane_count(), 1);
        assert_eq!(layout.planes()[0].semantic, PlaneSemantic::Gray);
        assert_eq!(layout.luma_plane_index(), Some(0));
    }

    #[cfg(feature = "planar")]
    #[test]
    fn plane_layout_display() {
        let layout = PlaneLayout::Interleaved { channels: 3 };
        assert_eq!(format!("{layout}"), "Interleaved(3ch)");

        let layout = PlaneLayout::oklab(ChannelType::F32);
        let s = format!("{layout}");
        assert!(s.starts_with("Oklab["), "got: {s}");
        assert!(s.contains("Oklab.L:F32"), "got: {s}");
    }

    // --- MultiPlaneImage tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn multi_plane_image_basic() {
        let layout = PlaneLayout::ycbcr_444(ChannelType::U8);
        let y = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);
        let cb = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);
        let cr = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);

        let img = MultiPlaneImage::new(layout, alloc::vec![y, cb, cr]);
        assert_eq!(img.plane_count(), 3);
        assert!(img.layout().is_ycbcr());
        assert!(img.buffer(0).is_some());
        assert!(img.buffer(2).is_some());
        assert!(img.buffer(3).is_none());
        assert!(img.origin().is_none());
    }

    #[cfg(feature = "planar")]
    #[test]
    fn multi_plane_image_with_origin() {
        let layout = PlaneLayout::gray(ChannelType::U8);
        let buf = crate::buffer::PixelBuffer::new(32, 32, PixelDescriptor::GRAY8);

        let ctx = alloc::sync::Arc::new(crate::color::ColorContext::from_cicp(
            crate::cicp::Cicp::new(1, 13, 0, false),
        ));
        let img = MultiPlaneImage::new(layout, alloc::vec![buf]).with_origin(ctx.clone());
        assert!(img.origin().is_some());
    }

    // --- PlaneRelationship tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn plane_relationship_variants() {
        let r = PlaneRelationship::Oklab;
        assert_eq!(r, PlaneRelationship::Oklab);

        let r = PlaneRelationship::YCbCr {
            matrix: YuvMatrix::Bt709,
        };
        assert_ne!(r, PlaneRelationship::Independent);

        // Copy
        let r2 = r;
        assert_eq!(r, r2);
    }

    #[test]
    fn reference_white_nits_values() {
        assert_eq!(TransferFunction::Pq.reference_white_nits(), 203.0);
        assert_eq!(TransferFunction::Srgb.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Hlg.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Linear.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Unknown.reference_white_nits(), 1.0);
    }

    // --- PlaneLayout mask tests ---

    #[cfg(feature = "planar")]
    #[test]
    fn mask_where_oklab() {
        let layout = PlaneLayout::oklab_alpha(ChannelType::F32);
        let luma = layout.luma_mask();
        assert!(luma.includes(0));
        assert!(!luma.includes(1));
        assert_eq!(luma.count(), 1);

        let chroma = layout.chroma_mask();
        assert!(chroma.includes(1));
        assert!(chroma.includes(2));
        assert!(!chroma.includes(0));
        assert_eq!(chroma.count(), 2);

        let alpha = layout.alpha_mask();
        assert!(alpha.includes(3));
        assert_eq!(alpha.count(), 1);
    }

    #[cfg(feature = "planar")]
    #[test]
    fn mask_where_ycbcr_420() {
        let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
        let luma = layout.luma_mask();
        assert!(luma.includes(0));
        assert_eq!(luma.count(), 1);

        let chroma = layout.chroma_mask();
        assert!(chroma.includes(1));
        assert!(chroma.includes(2));
        assert_eq!(chroma.count(), 2);

        let alpha = layout.alpha_mask();
        assert!(alpha.is_empty());
    }

    #[cfg(feature = "planar")]
    #[test]
    fn mask_where_gray() {
        let layout = PlaneLayout::gray(ChannelType::U8);
        let luma = layout.luma_mask();
        assert!(luma.includes(0));
        assert_eq!(luma.count(), 1);
        assert!(layout.chroma_mask().is_empty());
        assert!(layout.alpha_mask().is_empty());
    }

    #[cfg(feature = "planar")]
    #[test]
    fn mask_where_interleaved_returns_none() {
        let layout = PlaneLayout::Interleaved { channels: 4 };
        assert!(layout.luma_mask().is_empty());
        assert!(layout.chroma_mask().is_empty());
        assert!(layout.alpha_mask().is_empty());
    }
}
