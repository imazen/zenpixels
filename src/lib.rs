//! Pixel format negotiation and transfer-function-aware conversion.
//!
//! Every image codec needs format dispatch: converting input pixel data
//! to a format the codec natively supports. This crate centralizes that logic.
//!
//! # Core concepts
//!
//! - **Format negotiation**: [`best_match`] picks the cheapest conversion
//!   target from a codec's supported formats for a given source descriptor.
//!
//! - **Row conversion**: [`RowConverter`] pre-computes a conversion plan and
//!   converts rows with no per-row allocation, using SIMD where available.
//!
//! - **Codec helpers**: [`adapt::adapt_for_encode`] negotiates format and converts
//!   pixel data in one call, returning `Cow::Borrowed` when the input
//!   already matches a supported format.
//!
//! # Transfer function rules
//!
//! Conversions between depth classes apply the correct transfer function:
//!
//! | Source | Destination | Action |
//! |--------|-------------|--------|
//! | u8 Srgb | f32 Linear | sRGB EOTF via `linear-srgb` |
//! | f32 Linear | u8 Srgb | sRGB OETF via `linear-srgb` |
//! | u16 PQ | f32 Linear | PQ EOTF (SMPTE ST 2084) |
//! | f32 Linear | u16 PQ | PQ OETF (inverse EOTF) |
//! | u16 HLG | f32 Linear | HLG EOTF (ARIB STD-B67) |
//! | f32 Linear | u16 HLG | HLG OETF |
//! | u8 Unknown | f32 Unknown | Naive `v / 255.0` |
//! | Same depth | Same depth | No transfer math |

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod descriptor;
pub mod policy;

pub mod cicp;
pub mod color;
pub mod pixel_types;

mod convert;
mod error;
pub(crate) mod negotiate;

pub mod adapt;
pub mod buffer;
pub mod converter;
pub mod gamut;
pub mod hdr;
pub mod oklab;
pub mod op_format;
pub mod path;
pub mod pixels;
pub mod registry;

pub use adapt::adapt_for_encode_explicit;
pub use convert::{ConvertPlan, convert_row};
pub use converter::RowConverter;
pub use error::ConvertError;
pub use negotiate::{
    ConversionCost, ConvertIntent, FormatOption, Provenance, best_match, best_match_with,
    conversion_cost, conversion_cost_with_provenance, ideal_format, negotiate,
};
pub use op_format::{OpCategory, OpRequirement};
pub use path::{
    ConversionPath, LossBucket, MatrixStats, PathEntry, QualityThreshold, generate_path_matrix,
    matrix_stats, optimal_path,
};
pub use policy::{AlphaPolicy, ConvertOptions, DepthPolicy, GrayExpand, LumaCoefficients};
pub use registry::{CodecFormats, FormatEntry};

// Re-export key descriptor types at crate root for ergonomics.
pub use descriptor::{
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, InterleaveFormat,
    PixelDescriptor, PixelFormat, SignalRange, TransferFunction,
};

// Re-export planar types when the `planar` feature is enabled.
#[cfg(feature = "planar")]
pub use descriptor::{
    MultiPlaneImage, Plane, PlaneDescriptor, PlaneLayout, PlaneMask, PlaneRelationship,
    PlaneSemantic, Subsampling, YuvMatrix,
};

// Re-export buffer types at crate root.
pub use buffer::{Bgrx, BufferError, Pixel, PixelBuffer, PixelSlice, PixelSliceMut, Rgbx};

// Re-export color types at crate root.
pub use cicp::Cicp;
pub use color::{ColorContext, ColorProfileSource, NamedProfile};

// Re-export GrayAlpha pixel types at crate root.
pub use pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

// Re-export gamut conversion utilities.
pub use gamut::{
    GamutMatrix, apply_matrix_f32, apply_matrix_row_f32, apply_matrix_row_rgba_f32,
    conversion_matrix, mat3_mul,
};

// Re-export HDR types and tone mapping.
pub use hdr::{
    ContentLightLevel, HdrMetadata, MasteringDisplay, exposure_tonemap, reinhard_inverse,
    reinhard_tonemap,
};
