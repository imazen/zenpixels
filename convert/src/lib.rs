//! Transfer-function-aware pixel conversion for zenpixels.
//!
//! This crate provides all the conversion logic that was split out of the
//! `zenpixels` interchange crate: row-level format conversion, gamut mapping,
//! codec format negotiation, and HDR tone mapping.
//!
//! # Re-exports
//!
//! All interchange types from `zenpixels` are re-exported at the crate root,
//! so downstream code can depend on `zenpixels-convert` alone.
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
//! - **Extension traits**: [`TransferFunctionExt`], [`ColorPrimariesExt`],
//!   and [`PixelBufferConvertExt`] add conversion methods to interchange types.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

// Re-export all interchange types from zenpixels.
pub use zenpixels::*;

// Conversion modules.
pub(crate) mod convert;
pub mod error;
pub(crate) mod negotiate;

pub mod adapt;
pub mod cms;
pub mod converter;
pub mod ext;
pub mod gamut;
pub mod hdr;
pub mod oklab;
pub mod op_format;
pub mod output;
pub mod path;
pub mod pixels;
pub mod registry;

// Re-export key conversion types at crate root.
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
pub use registry::{CodecFormats, FormatEntry};

// Re-export extension traits.
#[cfg(feature = "buffer")]
pub use ext::PixelBufferConvertExt;
pub use ext::{ColorPrimariesExt, TransferFunctionExt};

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

// Re-export CMS traits.
pub use cms::{ColorManagement, RowTransform};

// Re-export output types.
pub use output::{EncodeReady, OutputMetadata, OutputProfile, finalize_for_output};
