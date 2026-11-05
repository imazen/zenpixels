//! Still-image quality contracts over actual zenpixels samples.
//!
//! Multiple native scalars, spatial maps, configured identities and optional
//! real prepared-reference work. No codec, conversion, kernel, runtime, video
//! session or implicit replay is included. External metric implementations are
//! intentional extension points; optional methods must default to refusal.
//!
//! Public records are extended through constructors, never downstream literals:
//! ```compile_fail
//! let opts = zenpsychovisual_api::EvalOpts {
//!     pulse: &zenpsychovisual_api::how_far::NoPulse,
//!     map: None,
//!     max_pixels: None,
//! };
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
extern crate alloc;
pub mod conformance;
mod error;
mod eval;
mod identity;
mod input;
mod map;
mod metric;
mod no_reference;
pub use error::{
    ErrorKind, MetricError, MetricResult, ResultMetricExt, fail, find_metric_error, is_stopped,
    stop_reason,
};
pub use eval::{
    Evaluation, Extras, NonFinite, Resampled, ScalarEval, ScoreKind, ScoreSpec, Scores,
    apply_map_request,
};
pub use how_far;
pub use identity::{Backend, Implementation, MetricId, is_sanitized_column};
pub use input::{
    Image, InputFormat, PlanarFormat, Planes, Rgb8In, check_same_size, contiguous_rgb8,
    is_rgb8_srgb, rgb8_pair, rgb8_view, unsupported,
};
pub use map::{
    MapFold, MapGeometry, MapPolarity, MapReduce, MapRequest, MapView, PixelRect, Support,
    fold_values, gray_f32_buffer, gray_f32_slice, map_fold, map_subsample_f32,
};
pub use metric::{
    EvalOpts, Evaluated, Metric, PreparedReference, ViewingConditions, WarmReference,
    prepare_reference, score, score_no_reference,
};
pub use no_reference::{NoReferenceMetric, PairedEval, PairedNoReference, REFERENCE_SUFFIX};
pub use whereat;
pub use zenpixels;
