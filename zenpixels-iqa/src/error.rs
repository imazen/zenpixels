//! The unified metric error: one concrete envelope, located with whereat.
//!
//! Metrics return [`MetricResult<T>`] = `Result<T, At<MetricError>>`, the
//! same shape zencodec uses for `At<CodecError>`. The `At` carries the
//! location trace; [`MetricError`] carries a coarse [`ErrorKind`] fixed at
//! construction, the originating metric family, an optional static
//! message, and the metric's own error boxed as [`MetricError::detail`]
//! (reachable through `source()`).
//!
//! Cancellation interop: `MetricError` implements `From<StopReason>` and
//! [`how_far::AsStopReason`], so how-far / enough code classifies it.
//! Rust's orphan rules forbid implementing `AsStopReason` for the located
//! `At<MetricError>` here; pass [`stop_reason`] to how-far's
//! `*_classified` entry points, or call `.error().as_stop_reason()`.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use core::fmt;

use how_far::{AsStopReason, StopReason};
use whereat::At;

/// `Result<T, At<MetricError>>` — the return type of every fallible
/// generic-surface call.
pub type MetricResult<T> = Result<T, At<MetricError>>;

/// Coarse, metric-agnostic classification of a failure. Fixed when the
/// error is created, so routing (retry, skip cell, alert) never parses
/// messages.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The input's pixel format/layout/transfer isn't one the metric
    /// accepts natively ([`crate::Metric::accepts`] would say `false`).
    UnsupportedInput,
    /// Reference and distorted (or successive frames, or a warm reference
    /// and a candidate) disagree on dimensions.
    DimensionMismatch,
    /// The image is below the metric's minimum size.
    TooSmall,
    /// Input values the metric can't score (non-finite samples, a buffer
    /// shorter than its declared geometry).
    InvalidInput,
    /// Well-formed input whose content the metric can't characterise —
    /// e.g. a featureless image that makes a fitted statistic NaN where
    /// the reference implementation raises.
    Degenerate,
    /// The metric's configuration or a request parameter is invalid
    /// (a zero subsample factor, a non-positive p-norm).
    InvalidConfig,
    /// A map was requested from a metric that publishes none.
    NoMap,
    /// A well-formed request this metric doesn't implement (an unknown
    /// `MapFold::Custom` name, a temporal call on a still-image metric).
    UnsupportedRequest,
    /// Stopped through the caller's `Stop`/`Pulse` — cancelled or past
    /// its deadline.
    Stopped(StopReason),
    /// Allocation failure or a resource ceiling.
    Resource,
    /// A compute backend (GPU device, thread pool) failed.
    Backend,
    /// A broken invariant inside the metric. Always a code defect.
    Internal,
}

impl ErrorKind {
    /// Stable lowercase name — `"unsupported_input"`, `"stopped"`, … —
    /// for ledgers and logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorKind::UnsupportedInput => "unsupported_input",
            ErrorKind::DimensionMismatch => "dimension_mismatch",
            ErrorKind::TooSmall => "too_small",
            ErrorKind::InvalidInput => "invalid_input",
            ErrorKind::Degenerate => "degenerate",
            ErrorKind::InvalidConfig => "invalid_config",
            ErrorKind::NoMap => "no_map",
            ErrorKind::UnsupportedRequest => "unsupported_request",
            ErrorKind::Stopped(_) => "stopped",
            ErrorKind::Resource => "resource",
            ErrorKind::Backend => "backend",
            ErrorKind::Internal => "internal",
        }
    }

    /// `true` when the same call with the same input will fail again —
    /// everything except `Stopped`, `Resource` and `Backend`.
    pub const fn is_deterministic(self) -> bool {
        !matches!(
            self,
            ErrorKind::Stopped(_) | ErrorKind::Resource | ErrorKind::Backend
        )
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Stopped(r) => write!(f, "stopped: {r}"),
            other => f.write_str(other.as_str()),
        }
    }
}

/// The error every generic-surface call returns, inside `At<…>`.
///
/// One word wide (the fields live behind a `Box`), so
/// `At<MetricError>` is two words and `MetricResult<T>` stays cheap on
/// the `?` path.
#[non_exhaustive]
pub struct MetricError(Box<Repr>);

struct Repr {
    kind: ErrorKind,
    metric: Option<&'static str>,
    message: Option<Cow<'static, str>>,
    detail: Option<Box<dyn core::error::Error + Send + Sync>>,
}

impl MetricError {
    /// Transfer an existing native location trace into this envelope.
    #[track_caller]
    pub fn from_located<E>(kind: ErrorKind, detail: At<E>) -> At<Self>
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        detail.map_error(|e| Self::from_detail(kind, e)).at()
    }

    /// An error of `kind` with no detail.
    pub fn new(kind: ErrorKind) -> Self {
        MetricError(Box::new(Repr {
            kind,
            metric: None,
            message: None,
            detail: None,
        }))
    }

    /// An error of `kind` wrapping the metric's own error.
    pub fn from_detail<E>(kind: ErrorKind, detail: E) -> Self
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        Self::new(kind).with_detail(detail)
    }

    /// The `Stopped` error for `reason`.
    pub fn stopped(reason: StopReason) -> Self {
        Self::new(ErrorKind::Stopped(reason))
    }

    /// Attach the metric's own error (boxed; reachable via `source()`).
    #[must_use]
    pub fn with_detail<E>(mut self, detail: E) -> Self
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        self.0.detail = Some(Box::new(detail));
        self
    }

    /// Attach an already-boxed detail error.
    #[must_use]
    pub fn with_boxed_detail(mut self, detail: Box<dyn core::error::Error + Send + Sync>) -> Self {
        self.0.detail = Some(detail);
        self
    }

    /// Stamp the originating metric family (`"fsim"`, `"cvvdp"`).
    #[must_use]
    pub fn with_metric(mut self, family: &'static str) -> Self {
        self.0.metric = Some(family);
        self
    }

    /// Attach a human-readable message (shown when there is no detail).
    #[must_use]
    pub fn with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.0.message = Some(message.into());
        self
    }

    /// The coarse classification.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// The originating metric family, if stamped.
    pub fn metric(&self) -> Option<&'static str> {
        self.0.metric
    }

    /// The attached message, if any.
    pub fn message(&self) -> Option<&str> {
        self.0.message.as_deref()
    }

    /// The metric's own error, if any.
    pub fn detail(&self) -> Option<&(dyn core::error::Error + 'static)> {
        self.0
            .detail
            .as_deref()
            .map(|d| d as &(dyn core::error::Error + 'static))
    }
}

impl fmt::Debug for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetricError")
            .field("kind", &self.0.kind)
            .field("metric", &self.0.metric)
            .field("message", &self.0.message)
            .field("detail", &self.0.detail)
            .finish()
    }
}

impl fmt::Display for MetricError {
    /// `"{metric}: {kind}: {message}: {detail}"`, omitting absent parts.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(m) = self.0.metric {
            write!(f, "{m}: ")?;
        }
        write!(f, "{}", self.0.kind)?;
        if let Some(msg) = &self.0.message {
            write!(f, ": {msg}")?;
        }
        if let Some(d) = &self.0.detail {
            write!(f, ": {d}")?;
        }
        Ok(())
    }
}

impl core::error::Error for MetricError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        self.detail()
    }
}

impl From<StopReason> for MetricError {
    fn from(reason: StopReason) -> Self {
        MetricError::stopped(reason)
    }
}

impl AsStopReason for MetricError {
    fn as_stop_reason(&self) -> Option<StopReason> {
        match self.0.kind {
            ErrorKind::Stopped(r) => Some(r),
            _ => None,
        }
    }
}

/// The stop reason of a located error, if it is a stop (orphan rules keep
/// `AsStopReason` off `At<MetricError>` itself).
pub fn stop_reason(error: &At<MetricError>) -> Option<StopReason> {
    error.error().as_stop_reason()
}

/// The how-far classifier: pass `is_stopped` to `Stages::run_classified`
/// and friends so a stopped metric records `Outcome::Stopped`, not a
/// failure.
pub fn is_stopped(error: &At<MetricError>) -> bool {
    stop_reason(error).is_some()
}

/// Recover the [`MetricError`] from any error chain — after erasure to
/// `Box<dyn Error>` or wrapping by a consumer. Looks for `At<MetricError>`,
/// `MetricError` and one `Box` layer at every level of `source()`.
pub fn find_metric_error<'a>(
    error: &'a (dyn core::error::Error + 'static),
) -> Option<&'a MetricError> {
    let mut cur: Option<&'a (dyn core::error::Error + 'static)> = Some(error);
    while let Some(e) = cur {
        if let Some(at) = e.downcast_ref::<At<MetricError>>() {
            return Some(at.error());
        }
        if let Some(m) = e.downcast_ref::<MetricError>() {
            return Some(m);
        }
        if let Some(b) = e.downcast_ref::<Box<At<MetricError>>>() {
            return Some(b.error());
        }
        cur = e.source();
    }
    None
}

/// Locate a new error at the caller.
#[track_caller]
pub fn fail<T>(error: MetricError) -> MetricResult<T> {
    Err(whereat::at(error))
}

/// Convert a metric crate's own `Result` into a [`MetricResult`],
/// locating the error at the caller.
pub trait ResultMetricExt<T> {
    /// Wrap the error as the detail of a `kind` [`MetricError`] stamped
    /// with `family`.
    #[track_caller]
    fn metric_err(self, family: &'static str, kind: ErrorKind) -> MetricResult<T>;

    /// Like [`Self::metric_err`], choosing the kind from the error value.
    #[track_caller]
    fn metric_err_with(
        self,
        family: &'static str,
        classify: impl FnOnce(&Self::Error) -> ErrorKind,
    ) -> MetricResult<T>
    where
        Self: Sized;

    /// The wrapped error type.
    type Error;
}

impl<T, E> ResultMetricExt<T> for Result<T, E>
where
    E: core::error::Error + Send + Sync + 'static,
{
    type Error = E;

    #[track_caller]
    fn metric_err(self, family: &'static str, kind: ErrorKind) -> MetricResult<T> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(whereat::at(
                MetricError::from_detail(kind, e).with_metric(family),
            )),
        }
    }

    #[track_caller]
    fn metric_err_with(
        self,
        family: &'static str,
        classify: impl FnOnce(&E) -> ErrorKind,
    ) -> MetricResult<T> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => {
                let kind = classify(&e);
                Err(whereat::at(
                    MetricError::from_detail(kind, e).with_metric(family),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[derive(Debug)]
    struct Leaf;
    impl fmt::Display for Leaf {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("leaf failed")
        }
    }
    impl core::error::Error for Leaf {}

    #[test]
    fn display_kind_detail_and_source() {
        let e = MetricError::from_detail(ErrorKind::TooSmall, Leaf).with_metric("gmsd");
        assert_eq!(e.to_string(), "gmsd: too_small: leaf failed");
        assert!(core::error::Error::source(&e).is_some());
        assert_eq!(e.kind(), ErrorKind::TooSmall);
        assert!(e.kind().is_deterministic());
    }

    #[test]
    fn stop_reason_round_trips_through_as_stop_reason() {
        let e: MetricError = StopReason::Cancelled.into();
        assert_eq!(e.as_stop_reason(), Some(StopReason::Cancelled));
        assert!(!e.kind().is_deterministic());
        let located = whereat::at(e);
        assert_eq!(stop_reason(&located), Some(StopReason::Cancelled));
        let other = whereat::at(MetricError::new(ErrorKind::NoMap));
        assert_eq!(stop_reason(&other), None);
    }

    #[test]
    fn recoverable_after_erasure() {
        let r: Result<(), Leaf> = Err(Leaf);
        let err = r.metric_err("fsim", ErrorKind::InvalidInput).unwrap_err();
        let boxed: Box<dyn core::error::Error + Send + Sync> = Box::new(err);
        let found = find_metric_error(&*boxed).expect("envelope recoverable");
        assert_eq!(found.kind(), ErrorKind::InvalidInput);
        assert_eq!(found.metric(), Some("fsim"));
    }

    #[test]
    fn located_at_the_caller() {
        let r: Result<(), Leaf> = Err(Leaf);
        let line = line!() + 1;
        let err = r.metric_err("fsim", ErrorKind::InvalidInput).unwrap_err();
        let frame = err.frames().next().expect("one frame");
        let loc = frame.location().expect("location recorded");
        assert_eq!(loc.line(), line);
        assert!(loc.file().ends_with("error.rs"));
    }

    #[test]
    fn error_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricError>();
        assert_send_sync::<At<MetricError>>();
    }
}
