//! # zenpixels 0.5 — Pixel & Color Interchange Types
//!
//! Pure, `no_std + alloc`, `#![forbid(unsafe_code)]` interchange foundation for
//! Rust image codecs and pipelines.
//!
//! ## Start Here (The 4 Core Types)
//!
//! - [`ImageSpec`]: pairs a physical [`SampleLayout`] with an algebraic [`ColorSpace`]
//!   (`Rgb`, `Gray`, `Oklab`, `Xyz`, `Perceptual`, `Cmyk`, `Icc`, `Untagged`). Use
//!   owned presets like [`ImageSpec::RGB8_SRGB`] for buffers and `&'static` refs like
//!   [`ImageSpec::rgb8_srgb_ref()`] for zero-allocation borrowed views.
//! - [`PixelBuf`]: owned, stride-preserving, alignment-exact 2D buffer (supporting
//!   zero-copy external storage via [`ByteStorage`]/[`ByteStorageMut`]) carrying
//!   [`Orientation`], [`HdrMetadata`], [`ColorOrigin`], and [`SampleFacts`].
//! - [`PixelView<'a>`] / [`PixelViewMut<'a>`]: zero-copy borrowed strided views and
//!   rectangular crops (`width()`, `height()` / `rows()`, `stride()`, `row(y)`).
//! - [`PixelFormat`]: `Copy + const` layout + [`ParametricColor`] for static codec
//!   capability tables without heap ownership.
//!
//! ## Quick Example
//!
//! ```rust
//! use zenpixels::{ImageSpec, PixelBuf, PixelView};
//!
//! let buf = PixelBuf::from_typed(vec![255u8, 0, 0, 0, 255, 0], 2, 1, ImageSpec::RGB8_SRGB)?;
//! let right = buf.crop_view(1, 0, 1, 1)?;
//! assert_eq!(right.row(0), &[0, 255, 0]);
//!
//! let raw = [10u8, 20, 30];
//! let view = PixelView::from_bytes(&raw, 1, 1, ImageSpec::rgb8_srgb_ref())?;
//! assert_eq!(view.row(0), &[10, 20, 30]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Additional Carriers
//!
//! - [`PlanarBuf`] / [`PlanarView<'a>`] / [`PlaneBuf`] / [`PlaneView<'a>`]: multi-plane
//!   YCbCr and planar RGB with uniform `(data, width, height, stride, spec)` constructors
//!   and bit-packed [`CodePlane`] access.
//! - [`BandedBuf`]: bounded row-band storage for gigapixel images.
//! - [`DualLayerHdrBuf`] / [`DualLayerHdrView<'a>`]: ISO 21496-1 / UltraHDR gain maps.
//! - **Error contract:** carrier/pipeline operations return [`At<E>`] (`*err.error()` /
//!   `err.into_inner()`); const-capable layout, storage, code-plane, and CICP-projection
//!   primitives return bare `E`.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels/");

pub mod buffer;
pub mod cicp;
pub mod color;
pub mod facts;
#[cfg(feature = "hash")]
pub mod hashing;
pub mod hdr;
#[cfg(feature = "icc")]
pub mod icc;
pub mod interop;
pub mod layout;
pub mod orientation;
pub mod pixel_types;
pub mod sample;
pub mod storage;

// Clean, flat re-exports of the core interchange surface.
pub use buffer::{
    BandedBuf, Bgrx, BufFill, BufferError, DEFAULT_MAX_BAND_BYTES, DualLayerHdrBuf,
    DualLayerHdrBufParts, DualLayerHdrView, PixelBuf, PixelBufParts, PixelCow, PixelSink,
    PixelSource, PixelView, PixelViewMut, PlanarBuf, PlanarView, PlanarViewMut, PlaneBuf,
    PlaneView, PlaneViewMut, Rgbx, TypedElement,
};
pub use cicp::{
    Cicp, CicpError, CicpProjectionError, CicpTag, color_primaries_name, matrix_coefficients_name,
    transfer_characteristics_name,
};
pub use color::{
    AlphaMode, Chromaticity, Color, ColorError, ColorOrigin, ColorSpace, CustomPrimaries,
    GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapDirection, GainMapMetadata, Gamma,
    HdrMetadata, HlgMode, IccColorSpace, IccValidationError, ImageSpec, OriginKind,
    ParametricColor, PerceptualSpace, PixelFormat, Primaries, RenderingIntent, SignalRange,
    SignedFraction, SpecCow, Transfer, UnsignedFraction, WhitePoint, validate_icc_structure,
};
pub use facts::{ExactDepth, FloatFacts, SampleFacts};
#[cfg(feature = "hash")]
pub use hashing::HashInclusions;
pub use hdr::{
    ClliCodes, ContentLightLevel, DiffuseWhite, HlgRendering, MasteringDisplay, Nits, PeakSource,
    St2086Codes,
};
#[cfg(feature = "rgb")]
pub use interop::PixelType;
pub use layout::{
    BitPacking, ChannelOrder, InteropTarget, LayoutError, PlaneRole, PlaneSpec, RowAlign,
    SampleLayout, SampleType, Subsampling,
};
pub use orientation::Orientation;
pub use pixel_types::{F16Sample, GrayAlpha8, GrayAlpha16, GrayAlphaF16, GrayAlphaF32};
pub use sample::{
    ChromaSiting, ChromaSubsampling, CodePlane, CodePlaneError, CodePlaneMut, ExactDepthError,
    SampleEncoding, SampleEncodingError, full_range_grid_nests, narrow_range_grid_nests,
};
pub use storage::{
    AlignedSamples, ByteStorage, ByteStorageMut, ExternalMutStorage, ExternalStorage, PixelStorage,
    StorageError, StorageSample,
};
pub use whereat::At;
