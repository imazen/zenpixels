use core::fmt;

use super::{
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, PixelFormat,
    SignalRange, TransferFunction, align_up_general, lcm,
};

/// Compact pixel format descriptor.
///
/// Combines a [`PixelFormat`] (physical pixel layout) with transfer function,
/// alpha mode, color primaries, and signal range.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    pub const fn with_alpha(self, alpha: Option<AlphaMode>) -> Self {
        Self { alpha, ..self }
    }

    /// Alias for [`with_alpha`](Self::with_alpha).
    #[inline]
    pub const fn with_alpha_mode(self, alpha: Option<AlphaMode>) -> Self {
        self.with_alpha(alpha)
    }

    /// Return a copy with a different signal range.
    #[inline]
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

    /// The alpha mode, if any.
    #[inline]
    pub const fn alpha_mode(self) -> Option<AlphaMode> {
        self.alpha
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
