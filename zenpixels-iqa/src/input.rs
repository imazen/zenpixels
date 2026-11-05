//! Borrowed actual pixels and native planes. No conversion is implicit.
use crate::{ErrorKind, MetricError, MetricResult, fail};
use alloc::{borrow::Cow, format, sync::Arc};
use zenpixels::plane::{AnyPlane, PlanarLayout};
use zenpixels::{Cicp, ColorContext, PixelDescriptor, PixelSlice};

/// One still image, with its actual color and sample interpretation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Image<'a> {
    /// Packed pixels; attached color context remains authoritative.
    Packed(PixelSlice<'a>),
    /// Validated native planar samples, without repacking or RGB conversion.
    Planar(Planes<'a>),
}
impl<'a> From<PixelSlice<'a>> for Image<'a> {
    fn from(value: PixelSlice<'a>) -> Self {
        Self::Packed(value)
    }
}
impl<'a> From<Planes<'a>> for Image<'a> {
    fn from(value: Planes<'a>) -> Self {
        Self::Planar(value)
    }
}
impl Image<'_> {
    /// Active width in full-resolution positions.
    pub fn width(&self) -> u32 {
        match self {
            Self::Packed(p) => p.width(),
            Self::Planar(p) => p.width,
        }
    }
    /// Active height in full-resolution positions.
    pub fn height(&self) -> u32 {
        match self {
            Self::Packed(p) => p.rows(),
            Self::Planar(p) => p.height,
        }
    }
    /// Pixel-free capability description. This does not certify color context.
    pub fn format(&self) -> InputFormat {
        match self {
            Self::Packed(p) => InputFormat::Packed(p.descriptor()),
            Self::Planar(p) => InputFormat::Planar(PlanarFormat {
                layout: p.layout.clone(),
                cicp: p.cicp,
            }),
        }
    }
    /// Borrow packed input, when present.
    pub fn as_packed(&self) -> Option<&PixelSlice<'_>> {
        match self {
            Self::Packed(p) => Some(p),
            _ => None,
        }
    }
    /// Borrow planar input, when present.
    pub fn as_planar(&self) -> Option<&Planes<'_>> {
        match self {
            Self::Planar(p) => Some(p),
            _ => None,
        }
    }
}
/// A pixel-free format query. Implementations still validate actual input context.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputFormat {
    /// Packed samples.
    Packed(PixelDescriptor),
    /// Native planes and signal interpretation.
    Planar(PlanarFormat),
}
impl InputFormat {
    /// Packed descriptor, when present.
    pub fn packed(&self) -> Option<PixelDescriptor> {
        match self {
            Self::Packed(d) => Some(*d),
            _ => None,
        }
    }
}
/// Native plane layout and declared CICP signal. Plane encodings include code
/// depth/shift or float scale; chroma siting and subsampling belong to the layout.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlanarFormat {
    /// Shared zenpixels layout, without an IQA-specific plane type hierarchy.
    pub layout: PlanarLayout,
    /// Primaries, transfer, matrix coefficients and range of the current samples.
    pub cicp: Cicp,
}
/// Borrowed planes validated against a zenpixels layout. Dimensions and strides
/// describe current samples. ICC/display interpretation is never inferred.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Planes<'a> {
    layout: &'a PlanarLayout,
    planes: &'a [AnyPlane<'a>],
    width: u32,
    height: u32,
    cicp: Cicp,
    context: Option<Arc<ColorContext>>,
}
impl<'a> Planes<'a> {
    /// Validate sample types, plane count, subsampling, semi-planar component
    /// widths and odd edges through zenpixels. No sample arithmetic is performed.
    #[track_caller]
    pub fn new(
        layout: &'a PlanarLayout,
        planes: &'a [AnyPlane<'a>],
        width: u32,
        height: u32,
        cicp: Cicp,
    ) -> MetricResult<Self> {
        use whereat::ErrorAtExt as _;
        if width == 0 || height == 0 {
            return fail(MetricError::new(ErrorKind::InvalidInput));
        }
        layout
            .check(planes, width, height)
            .map_err(|e| MetricError::from_detail(ErrorKind::InvalidInput, e).start_at())?;
        Ok(Self {
            layout,
            planes,
            width,
            height,
            cicp,
            context: None,
        })
    }
    /// Retain authoritative ICC/CICP and absolute-luminance context. A consumer
    /// must resolve conflicting or unsupported claims before scoring.
    pub fn with_color_context(mut self, context: Arc<ColorContext>) -> Self {
        self.context = Some(context);
        self
    }
    /// The native layout.
    pub fn layout(&self) -> &PlanarLayout {
        self.layout
    }
    /// Actual strided samples, in layout order.
    pub fn planes(&self) -> &[AnyPlane<'a>] {
        self.planes
    }
    /// Declared CICP signal.
    pub fn cicp(&self) -> Cicp {
        self.cicp
    }
    /// Additional authoritative context, without conversion or relabeling.
    pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
        self.context.as_ref()
    }
}
/// A strided sRGB8 view for native byte APIs.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Rgb8In<'a> {
    /// Active samples and intervening padding.
    pub pixels: &'a [u8],
    /// Pixels per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Bytes between row starts.
    pub stride: usize,
}
/// Whether the packed descriptor is sRGB8; actual context is checked separately.
pub fn is_rgb8_srgb(format: &InputFormat) -> bool {
    format.packed() == Some(PixelDescriptor::RGB8_SRGB)
}
/// Borrow sRGB8 while refusing a conflicting/opaque current profile or luminance
/// anchor. No ICC conversion is attempted by this helper.
#[track_caller]
pub fn rgb8_view<'a>(image: &'a Image<'_>) -> MetricResult<Rgb8In<'a>> {
    let Some(p) = image.as_packed() else {
        return fail(unsupported(&image.format(), "packed sRGB8"));
    };
    if p.descriptor() != PixelDescriptor::RGB8_SRGB
        || p.color_context()
            .is_some_and(|c| c.icc.is_some() || !c.is_srgb() || c.diffuse_white.is_some())
    {
        return fail(unsupported(
            &image.format(),
            "sRGB8 with compatible current color context",
        ));
    }
    Ok(Rgb8In {
        pixels: p.as_strided_bytes(),
        width: p.width() as usize,
        height: p.rows() as usize,
        stride: p.stride(),
    })
}
/// Borrow an equal-size pair without flattening rows.
#[track_caller]
pub fn rgb8_pair<'a>(r: &'a Image<'_>, d: &'a Image<'_>) -> MetricResult<(Rgb8In<'a>, Rgb8In<'a>)> {
    check_same_size(r, d)?;
    Ok((rgb8_view(r)?, rgb8_view(d)?))
}
/// Explicit unsupported-input error carrying the expected native format.
pub fn unsupported(format: &InputFormat, wanted: &str) -> MetricError {
    MetricError::new(ErrorKind::UnsupportedInput)
        .with_message(format!("wanted {wanted}, got {format:?}"))
}
/// Check active geometry before arithmetic.
#[track_caller]
pub fn check_same_size(r: &Image<'_>, d: &Image<'_>) -> MetricResult<()> {
    if (r.width(), r.height()) != (d.width(), d.height()) {
        return fail(MetricError::new(ErrorKind::DimensionMismatch));
    }
    Ok(())
}
/// Explicitly pack active rows for a backend requiring contiguous input. Tightly
/// packed input is borrowed, including a minimally sized final row.
pub fn contiguous_rgb8<'a>(r: &Rgb8In<'a>) -> Cow<'a, [u8]> {
    let row = r.width * 3;
    if r.stride == row {
        Cow::Borrowed(&r.pixels[..row * r.height])
    } else {
        Cow::Owned(
            (0..r.height)
                .flat_map(|y| r.pixels[y * r.stride..y * r.stride + row].iter().copied())
                .collect(),
        )
    }
}
