//! Pixel format negotiation and transfer-function-aware conversion.
//!
//! Every codec that implements [`zencodec_types::Encoder`] needs to handle
//! format dispatch: converting input pixel data to a format the codec
//! natively supports. This crate centralizes that logic.
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
//!   a [`PixelSlice`](zencodec_types::PixelSlice) in one call, returning
//!   `Cow::Borrowed` when the input already matches a supported format.
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

mod convert;
mod error;
pub(crate) mod negotiate;

pub mod adapt;
pub mod converter;
pub mod op_format;
pub mod path;
pub mod pixels;
pub mod registry;

pub use convert::{convert_row, ConvertPlan};
pub use converter::RowConverter;
pub use error::ConvertError;
pub use negotiate::{
    best_match, best_match_with, conversion_cost, conversion_cost_with_provenance, ideal_format,
    negotiate, ConversionCost, ConvertIntent, FormatOption, Provenance,
};
pub use op_format::{OpCategory, OpRequirement};
pub use path::{
    generate_path_matrix, matrix_stats, optimal_path, ConversionPath, LossBucket, MatrixStats,
    PathEntry, QualityThreshold,
};
pub use registry::{CodecFormats, FormatEntry};

// Re-export key types from zencodec-types for convenience.
pub use zencodec_types::{ColorPrimaries, PixelDescriptor};
