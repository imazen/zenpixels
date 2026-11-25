//! # zenpixels 0.5 — Greenfield Pixel & Color Interchange Types
//!
//! Pure, `no_std + alloc`, `#![forbid(unsafe_code)]` interchange foundation for
//! Rust image codecs and pipelines:
//!
//! - **Algebraic [`ColorSpace`]:** mutually exclusive variants (`Rgb`, `Gray`,
//!   `Oklab`, `Perceptual`, `Cmyk`, `Icc`, `Untagged`) make invalid states such
//!   as Oklab carrying BT.709 primaries or an ICC profile paired with conflicting
//!   CICP primaries unrepresentable by construction.
//! - **Unified [`SampleLayout`] + [`BitPacking`]:** full-width samples, 10/12-bit
//!   LSB/MSB codec planes (e.g. P010), and bitfield words (RGB565, RGB10A2).
//! - **Unparameterized, alignment-exact [`PixelBuf`] / [`PixelView<'a>`] /
//!   [`PixelViewMut<'a>`]:** fallible by default, true `'a` lifetimes, zero-copy
//!   `bytemuck` typed access and `Vec<P>` moves that strictly preserve Rust's
//!   allocation/deallocation alignment contract.
//! - **Unified multi-band, multi-plane, and HDR carriers:** [`BandedBuf`],
//!   [`PlanarBuf`], [`DualLayerHdr`] (ISO 21496-1 / UltraHDR gain maps), and
//!   [`SampleFacts`] (automatically cleared on `&mut`).

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
    BandedBuf, Bgrx, BufFill, BufferError, DEFAULT_MAX_BAND_BYTES, DualLayerHdr, DualLayerHdrParts,
    DualLayerHdrView, OwnedPlane, PixelBuf, PixelBufParts, PixelCow, PixelSink, PixelSource,
    PixelView, PixelViewMut, PlanarBuf, PlanarView, PlanarViewMut, PlaneView, PlaneViewMut, Rgbx,
    TypedElement,
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
