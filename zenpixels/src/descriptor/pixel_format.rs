use core::fmt;

use super::{
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, PixelDescriptor,
};

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
