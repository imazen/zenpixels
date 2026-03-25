//! Pixel format descriptor types.
//!
//! These types describe the format of pixel data: channel type, layout,
//! alpha handling, transfer function, color primaries, and signal range.
//!
//! Standalone definitions — no dependency on zencodec.

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

    /// Convert to the CICP `transfer_characteristics` code.
    #[allow(unreachable_patterns)]
    #[inline]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Linear => Some(8),
            Self::Srgb => Some(13),
            Self::Pq => Some(16),
            Self::Hlg => Some(18),
            Self::Unknown => None,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::format;
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

    // --- PlaneSemantic tests ---

    // --- PlaneDescriptor tests ---

    // --- PlaneMask tests ---

    // --- PlaneLayout tests ---

    // --- MultiPlaneImage tests ---

    // --- PlaneRelationship tests ---

    #[test]
    fn reference_white_nits_values() {
        assert_eq!(TransferFunction::Pq.reference_white_nits(), 203.0);
        assert_eq!(TransferFunction::Srgb.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Hlg.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Linear.reference_white_nits(), 1.0);
        assert_eq!(TransferFunction::Unknown.reference_white_nits(), 1.0);
    }

    // --- PlaneLayout mask tests ---

    // --- Display impl tests ---

    #[test]
    fn channel_type_display() {
        assert_eq!(format!("{}", ChannelType::U8), "U8");
        assert_eq!(format!("{}", ChannelType::U16), "U16");
        assert_eq!(format!("{}", ChannelType::F32), "F32");
        assert_eq!(format!("{}", ChannelType::F16), "F16");
    }

    #[test]
    fn channel_layout_display() {
        assert_eq!(format!("{}", ChannelLayout::Gray), "Gray");
        assert_eq!(format!("{}", ChannelLayout::GrayAlpha), "GrayAlpha");
        assert_eq!(format!("{}", ChannelLayout::Rgb), "RGB");
        assert_eq!(format!("{}", ChannelLayout::Rgba), "RGBA");
        assert_eq!(format!("{}", ChannelLayout::Bgra), "BGRA");
        assert_eq!(format!("{}", ChannelLayout::Oklab), "Oklab");
        assert_eq!(format!("{}", ChannelLayout::OklabA), "OklabA");
    }

    #[test]
    fn alpha_mode_display() {
        assert_eq!(format!("{}", AlphaMode::Undefined), "undefined");
        assert_eq!(format!("{}", AlphaMode::Straight), "straight");
        assert_eq!(format!("{}", AlphaMode::Premultiplied), "premultiplied");
        assert_eq!(format!("{}", AlphaMode::Opaque), "opaque");
    }

    #[test]
    fn transfer_function_display() {
        assert_eq!(format!("{}", TransferFunction::Linear), "linear");
        assert_eq!(format!("{}", TransferFunction::Srgb), "sRGB");
        assert_eq!(format!("{}", TransferFunction::Bt709), "BT.709");
        assert_eq!(format!("{}", TransferFunction::Pq), "PQ");
        assert_eq!(format!("{}", TransferFunction::Hlg), "HLG");
        assert_eq!(format!("{}", TransferFunction::Unknown), "unknown");
    }

    #[test]
    fn color_primaries_display() {
        assert_eq!(format!("{}", ColorPrimaries::Bt709), "BT.709");
        assert_eq!(format!("{}", ColorPrimaries::Bt2020), "BT.2020");
        assert_eq!(format!("{}", ColorPrimaries::DisplayP3), "Display P3");
        assert_eq!(format!("{}", ColorPrimaries::Unknown), "unknown");
    }

    #[test]
    fn signal_range_display() {
        assert_eq!(format!("{}", SignalRange::Full), "full");
        assert_eq!(format!("{}", SignalRange::Narrow), "narrow");
    }

    #[test]
    fn pixel_descriptor_display() {
        let s = format!("{}", PixelDescriptor::RGB8_SRGB);
        assert!(s.contains("U8"), "expected U8 in: {s}");
        assert!(s.contains("sRGB"), "expected sRGB in: {s}");

        let s = format!("{}", PixelDescriptor::RGBA8_SRGB);
        assert!(s.contains("alpha=straight"), "expected alpha in: {s}");
    }

    #[test]
    fn pixel_format_display() {
        let s = format!("{}", PixelFormat::Rgb8);
        assert!(s.contains("RGB8"));
        let s = format!("{}", PixelFormat::Bgra8);
        assert!(s.contains("BGRA8"));
    }

    // --- from_cicp / to_cicp tests ---

    #[test]
    fn transfer_function_from_cicp() {
        assert_eq!(
            TransferFunction::from_cicp(1),
            Some(TransferFunction::Bt709)
        );
        assert_eq!(
            TransferFunction::from_cicp(8),
            Some(TransferFunction::Linear)
        );
        assert_eq!(
            TransferFunction::from_cicp(13),
            Some(TransferFunction::Srgb)
        );
        assert_eq!(TransferFunction::from_cicp(16), Some(TransferFunction::Pq));
        assert_eq!(TransferFunction::from_cicp(18), Some(TransferFunction::Hlg));
        assert_eq!(TransferFunction::from_cicp(99), None);
    }

    #[test]
    fn transfer_function_to_cicp() {
        assert_eq!(TransferFunction::Bt709.to_cicp(), Some(1));
        assert_eq!(TransferFunction::Linear.to_cicp(), Some(8));
        assert_eq!(TransferFunction::Srgb.to_cicp(), Some(13));
        assert_eq!(TransferFunction::Pq.to_cicp(), Some(16));
        assert_eq!(TransferFunction::Hlg.to_cicp(), Some(18));
        assert_eq!(TransferFunction::Unknown.to_cicp(), None);
    }

    #[test]
    fn transfer_function_cicp_roundtrip() {
        for tf in [
            TransferFunction::Bt709,
            TransferFunction::Linear,
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg,
        ] {
            let code = tf.to_cicp().unwrap();
            assert_eq!(TransferFunction::from_cicp(code), Some(tf));
        }
    }

    #[test]
    fn color_primaries_from_cicp() {
        assert_eq!(ColorPrimaries::from_cicp(1), Some(ColorPrimaries::Bt709));
        assert_eq!(ColorPrimaries::from_cicp(9), Some(ColorPrimaries::Bt2020));
        assert_eq!(
            ColorPrimaries::from_cicp(12),
            Some(ColorPrimaries::DisplayP3)
        );
        assert_eq!(ColorPrimaries::from_cicp(99), None);
    }

    #[test]
    fn color_primaries_to_cicp() {
        assert_eq!(ColorPrimaries::Bt709.to_cicp(), Some(1));
        assert_eq!(ColorPrimaries::Bt2020.to_cicp(), Some(9));
        assert_eq!(ColorPrimaries::DisplayP3.to_cicp(), Some(12));
        assert_eq!(ColorPrimaries::Unknown.to_cicp(), None);
    }

    // --- ChannelType helpers ---

    #[test]
    fn channel_type_helpers() {
        assert!(ChannelType::U8.is_u8());
        assert!(!ChannelType::U8.is_u16());
        assert!(ChannelType::U16.is_u16());
        assert!(ChannelType::F32.is_f32());
        assert!(ChannelType::F16.is_f16());
        assert!(ChannelType::U8.is_integer());
        assert!(ChannelType::U16.is_integer());
        assert!(!ChannelType::F32.is_integer());
        assert!(ChannelType::F32.is_float());
        assert!(ChannelType::F16.is_float());
        assert!(!ChannelType::U8.is_float());
    }

    // --- ChannelLayout helpers ---

    #[test]
    fn channel_layout_channels() {
        assert_eq!(ChannelLayout::Gray.channels(), 1);
        assert_eq!(ChannelLayout::GrayAlpha.channels(), 2);
        assert_eq!(ChannelLayout::Rgb.channels(), 3);
        assert_eq!(ChannelLayout::Rgba.channels(), 4);
        assert_eq!(ChannelLayout::Bgra.channels(), 4);
        assert_eq!(ChannelLayout::Oklab.channels(), 3);
        assert_eq!(ChannelLayout::OklabA.channels(), 4);
    }

    #[test]
    fn channel_layout_has_alpha() {
        assert!(!ChannelLayout::Gray.has_alpha());
        assert!(ChannelLayout::GrayAlpha.has_alpha());
        assert!(!ChannelLayout::Rgb.has_alpha());
        assert!(ChannelLayout::Rgba.has_alpha());
        assert!(ChannelLayout::Bgra.has_alpha());
        assert!(!ChannelLayout::Oklab.has_alpha());
        assert!(ChannelLayout::OklabA.has_alpha());
    }

    // --- PixelDescriptor builder methods ---

    #[test]
    fn with_transfer() {
        let desc = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
        assert_eq!(desc.transfer(), TransferFunction::Linear);
        assert_eq!(desc.layout(), ChannelLayout::Rgb);
    }

    #[test]
    fn with_primaries() {
        let desc = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        assert_eq!(desc.primaries, ColorPrimaries::DisplayP3);
    }

    #[test]
    fn with_signal_range() {
        let desc = PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow);
        assert_eq!(desc.signal_range, SignalRange::Narrow);
    }

    #[test]
    fn with_alpha_mode() {
        let desc = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
        assert_eq!(desc.alpha(), Some(AlphaMode::Premultiplied));
    }

    // --- PixelDescriptor predicates ---

    #[test]
    fn is_opaque_and_may_have_transparency() {
        assert!(PixelDescriptor::RGB8_SRGB.is_opaque());
        assert!(!PixelDescriptor::RGB8_SRGB.may_have_transparency());
        assert!(!PixelDescriptor::RGBA8_SRGB.is_opaque());
        assert!(PixelDescriptor::RGBA8_SRGB.may_have_transparency());

        let rgbx = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Undefined),
            TransferFunction::Srgb,
        );
        assert!(rgbx.is_opaque());
        assert!(!rgbx.may_have_transparency());
    }

    #[test]
    fn is_linear_and_is_unknown_transfer() {
        assert!(!PixelDescriptor::RGB8_SRGB.is_linear());
        assert!(PixelDescriptor::RGBF32_LINEAR.is_linear());
        assert!(!PixelDescriptor::RGB8_SRGB.is_unknown_transfer());
        let desc = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Unknown);
        assert!(desc.is_unknown_transfer());
    }

    #[test]
    fn min_alignment() {
        assert_eq!(PixelDescriptor::RGB8_SRGB.min_alignment(), 1);
        assert_eq!(PixelDescriptor::RGBF32_LINEAR.min_alignment(), 4);
    }

    #[test]
    fn aligned_stride() {
        assert_eq!(PixelDescriptor::RGB8_SRGB.aligned_stride(100), 300);
        assert_eq!(PixelDescriptor::RGBA8_SRGB.aligned_stride(100), 400);
        assert_eq!(PixelDescriptor::RGBF32_LINEAR.aligned_stride(10), 120);
    }

    #[test]
    fn simd_aligned_stride() {
        let stride = PixelDescriptor::RGB8_SRGB.simd_aligned_stride(100, 16);
        assert!(stride >= 300);
        assert_eq!(stride % 16, 0);
        assert_eq!(stride % 3, 0); // pixel-aligned
    }

    // --- new_full and from_pixel_format ---

    #[test]
    fn new_full_constructor() {
        let desc = PixelDescriptor::new_full(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
            ColorPrimaries::DisplayP3,
        );
        assert_eq!(desc.primaries, ColorPrimaries::DisplayP3);
        assert_eq!(desc.transfer(), TransferFunction::Srgb);
    }

    #[test]
    fn from_pixel_format_constructor() {
        let desc = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8);
        assert_eq!(desc.layout(), ChannelLayout::Rgba);
        assert_eq!(desc.transfer(), TransferFunction::Unknown);
        assert_eq!(desc.primaries, ColorPrimaries::Bt709);
        assert_eq!(desc.signal_range, SignalRange::Full);
    }

    // --- PixelFormat::name ---

    #[test]
    fn pixel_format_name() {
        assert_eq!(PixelFormat::Rgb8.name(), "RGB8");
        assert_eq!(PixelFormat::Bgra8.name(), "BGRA8");
        assert_eq!(PixelFormat::Gray8.name(), "Gray8");
    }

    // --- ColorModel ---

    #[test]
    fn color_model_display() {
        assert_eq!(format!("{}", ColorModel::Gray), "Gray");
        assert_eq!(format!("{}", ColorModel::Rgb), "RGB");
        assert_eq!(format!("{}", ColorModel::YCbCr), "YCbCr");
        assert_eq!(format!("{}", ColorModel::Oklab), "Oklab");
    }

    // --- SignalRange default ---

    #[test]
    fn signal_range_default() {
        assert_eq!(SignalRange::default(), SignalRange::Full);
    }

    // --- ColorPrimaries default ---

    #[test]
    fn color_primaries_default() {
        assert_eq!(ColorPrimaries::default(), ColorPrimaries::Bt709);
    }
}
