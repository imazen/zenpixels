//! Pipeline planner — operation requirements, codec format tables, and path solver.
//!
//! This module is gated behind the `pipeline` feature. It provides:
//!
//! - [`op_format`]: Operation category requirements (what working format each
//!   operation class needs).
//! - [`registry`]: Static codec format tables (what each codec can decode/encode).
//! - [`path`]: Conversion path solver (find the cheapest source→op→output chain).

pub mod op_format;
pub mod path;
pub mod registry;

pub use op_format::{OpCategory, OpRequirement};
pub use path::{
    ConversionPath, LossBucket, MatrixStats, PathEntry, QualityThreshold, generate_path_matrix,
    matrix_stats, optimal_path,
};
pub use registry::{CodecFormats, FormatEntry};
