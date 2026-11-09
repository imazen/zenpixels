//! Typed still-image quality results over colour-described zenpixels views.
//!
//! [`Metric::Output`] preserves native structured assessments. [`Evaluation`]
//! exposes named scalars; [`PrepareReference`] optionally reuses reference work.
//! Diagnostic arrays remain native data, explicitly requested in [`EvalOpts`].
//! [`Erased`] composes with typed wrappers and supports heterogeneous metric
//! collections. Persistence, map processing, scheduling and video composition
//! belong to consumers. Worker adapters supply any required
//! Send/Sync bounds; the core imposes none.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
extern crate alloc;
mod erased;
mod error;
mod eval;
mod identity;
mod metric;
pub use erased::{DynMetric, DynPrepareReference, Erased, ErasedEvaluation};
pub use error::{
    ErrorKind, MetricError, MetricResult, ResultMetricExt, fail, find_metric_error, is_stopped,
    stop_reason,
};
pub use eval::{Direction, Evaluation, Score, ScoreSpec};
pub use how_far;
pub use identity::{Backend, Implementation, MetricId};
pub use metric::{
    EvalOpts, Metric, PrepareReference, PreparedReference, WarmReference, prepare_reference, score,
};
pub use whereat;
pub use zenpixels;
