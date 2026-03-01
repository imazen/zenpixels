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
//! - **Codec helpers**: [`adapt_for_encode`] negotiates format and converts
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
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, PixelDescriptor,
    PixelFormat, PlanarDescriptor, PlaneMask, PlaneSemantic, PlaneSpec, SignalRange, Subsampling,
    TransferFunction, YuvMatrix,
};

// Re-export buffer types at crate root.
pub use buffer::{BufferError, Pixel, PixelBuffer, PixelSlice, PixelSliceMut, Rgbx, Bgrx};

// Re-export color types at crate root.
pub use cicp::Cicp;
pub use color::{ColorContext, ColorProfileSource, NamedProfile, WorkingColorSpace};

// Re-export GrayAlpha pixel types at crate root.
pub use pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};
