//! Still-image scoring with owned observation phases and located errors.
use crate::{
    ErrorKind, Evaluation, Image, InputFormat, MapRequest, MetricError, MetricId, MetricResult,
    ScoreKind, ScoreSpec, check_same_size, fail,
};
use alloc::{borrow::Cow, boxed::Box, vec::Vec};
use core::fmt;
use how_far::{Complete, NoPulse, Pulse, Stop};
use whereat::ResultAtExt as _;

/// What one evaluation should compute, and how the caller can stop it.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct EvalOpts<'a> {
    /// `None` = scores only; `Some` = also produce maps.
    pub map: Option<MapRequest>,
    /// Maximum active pixels in each image, excluding scratch/retained work.
    pub max_pixels: Option<u64>,
    /// Cancellation and progress. Metrics call [`Self::checkpoint`] (or
    /// pass this to how-far stages) between phases. Default: `&NoPulse`.
    pub pulse: &'a dyn Pulse,
}

impl Default for EvalOpts<'_> {
    fn default() -> Self {
        Self::SCORE_ONLY
    }
}

impl fmt::Debug for EvalOpts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalOpts")
            .field("map", &self.map)
            .finish_non_exhaustive()
    }
}

impl<'a> EvalOpts<'a> {
    /// Scores only, never stopped.
    pub const SCORE_ONLY: EvalOpts<'static> = EvalOpts {
        max_pixels: None,
        map: None,
        pulse: &NoPulse,
    };

    /// Same as [`Self::SCORE_ONLY`].
    pub const fn new() -> EvalOpts<'static> {
        Self::SCORE_ONLY
    }

    /// Request maps.
    #[must_use]
    pub const fn with_map(mut self, request: MapRequest) -> Self {
        self.map = Some(request);
        self
    }

    /// Request native maps.
    #[must_use]
    pub const fn with_native_map(self) -> Self {
        self.with_map(MapRequest::Native)
    }

    /// Scores only (drop any map request).
    #[must_use]
    pub const fn without_map(mut self) -> Self {
        self.map = None;
        self
    }

    /// Observe and stop through `pulse`.
    #[must_use]
    pub fn with_pulse<'b>(self, pulse: &'b dyn Pulse) -> EvalOpts<'b> {
        EvalOpts {
            max_pixels: self.max_pixels,
            map: self.map,
            pulse,
        }
    }

    /// Enforce a per-image active-pixel admission ceiling.
    pub const fn with_max_pixels(mut self, limit: u64) -> Self {
        self.max_pixels = Some(limit);
        self
    }

    /// Report actual native work units and check cancellation.
    #[track_caller]
    pub fn step(&self, completed: u64) -> MetricResult<()> {
        use how_far::ProgressExt as _;
        self.pulse
            .step(completed)
            .map_err(|r| whereat::at(MetricError::stopped(r)))
    }

    /// The pulse as a [`Stop`], for metric internals that take one.
    pub fn stop(&self) -> &'a dyn Stop {
        self.pulse
    }

    /// Return [`ErrorKind::Stopped`] if the caller stopped.
    ///
    /// # Errors
    /// As described.
    #[track_caller]
    pub fn checkpoint(&self) -> MetricResult<()> {
        match self.pulse.check() {
            Ok(()) => Ok(()),
            Err(r) => fail(MetricError::stopped(r)),
        }
    }

    /// For metrics without maps: fail before any work when a map was
    /// requested.
    ///
    /// # Errors
    /// [`ErrorKind::NoMap`] with `family` stamped.
    #[track_caller]
    pub fn refuse_map(&self, family: &'static str) -> MetricResult<()> {
        match self.map {
            None => Ok(()),
            Some(_) => fail(MetricError::new(ErrorKind::NoMap).with_metric(family)),
        }
    }
}

/// Typed viewing conditions of a viewing-conditioned metric. Richer
/// settings (surround, observer age, EOTF) stay metric-specific and appear
/// in [`MetricId::config`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewingConditions {
    /// Pixels per visual degree.
    pub ppd: f32,
    /// Display peak luminance, cd/m², when the metric models display
    /// light.
    pub display_peak_nits: Option<f32>,
}

impl ViewingConditions {
    /// `ppd` and an optional peak; extend with the `with_*` builders.
    pub const fn new(ppd: f32, display_peak_nits: Option<f32>) -> Self {
        Self {
            ppd,
            display_peak_nits,
        }
    }

    /// Set the display peak.
    #[must_use]
    pub const fn with_display_peak_nits(mut self, nits: f32) -> Self {
        self.display_peak_nits = Some(nits);
        self
    }
}

/// An externally implementable configured scorer. Future optional operations
/// must default to explicit refusal. Call `score` for checked result envelopes.
pub trait Metric: Send {
    /// Complete arithmetic identity, including all score-affecting settings.
    fn id(&self) -> MetricId;
    /// Every native scalar, headline first; stable for this configured instance.
    fn outputs(&self) -> Cow<'static, [ScoreSpec]>;
    /// Headline semantics.
    fn score_kind(&self) -> ScoreKind {
        self.outputs()
            .first()
            .map_or(ScoreKind::Feature, |s| s.kind)
    }
    /// Fully qualified persistent columns, in output order.
    fn columns(&self) -> Vec<alloc::string::String> {
        let id = self.id();
        self.outputs().iter().map(|s| id.column(&s.key)).collect()
    }
    /// Typed viewing conditions when applicable; additional settings belong in identity.
    fn viewing_conditions(&self) -> Option<ViewingConditions> {
        None
    }
    /// Native format support. Actual ICC/CICP, alpha and luminance context must
    /// still be checked on input; this query never authorizes conversion.
    fn accepts(&self, format: &InputFormat) -> bool;
    /// Native operation. Read active strided rows/planes, poll/report through
    /// the full Pulse, and join any owned workers before returning. Do not
    /// retain the pulse in results or later prepared-reference operations.
    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>>;
    /// Prepare real reference work reuse with bit-identical cold results.
    /// Default refuses: replay is not silently advertised as preparation.
    fn warm_reference<'m>(
        &'m mut self,
        _r: Image<'_>,
        _opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        fail(MetricError::new(ErrorKind::UnsupportedRequest))
    }
}
/// Actual prepared reference work; subsequent calls receive fresh controls.
pub trait WarmReference: Send {
    /// Score one candidate. Errors must leave this session reusable or make
    /// subsequent calls refuse; never serve stale results.
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>>;
    /// Whether actual reference work is retained, rather than cold replay.
    fn reuses_reference_work(&self) -> bool;
}
/// Result bound to its complete pre-execution identity and output specification.
#[non_exhaustive]
pub struct Evaluated {
    /// Arithmetic and configuration identity.
    pub id: MetricId,
    /// Declared outputs, headline first.
    pub outputs: Vec<ScoreSpec>,
    /// Native result including scores, maps, extras and concrete downcast.
    pub result: Box<dyn Evaluation>,
}
impl fmt::Debug for Evaluated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Evaluated")
            .field("id", &self.id)
            .field("outputs", &self.outputs)
            .field("score", &self.result.score())
            .finish_non_exhaustive()
    }
}
#[track_caller]
fn admit(image: &Image<'_>, opts: EvalOpts<'_>) -> MetricResult<()> {
    opts.checkpoint()?;
    if image.width() == 0 || image.height() == 0 {
        return fail(MetricError::new(ErrorKind::InvalidInput));
    }
    if opts
        .max_pixels
        .is_some_and(|limit| u64::from(image.width()) * u64::from(image.height()) > limit)
    {
        return fail(MetricError::new(ErrorKind::Resource));
    }
    Ok(())
}
#[track_caller]
fn finish(
    id: MetricId,
    outputs: Vec<ScoreSpec>,
    result: Box<dyn Evaluation>,
    opts: EvalOpts<'_>,
) -> MetricResult<Evaluated> {
    opts.checkpoint()?;
    let actual: Vec<_> = result.scores().collect();
    if outputs.is_empty()
        || outputs.len() != actual.len()
        || result.score().to_bits() != actual[0].1.to_bits()
        || outputs.iter().zip(&actual).any(|(s, (key, v))| {
            s.key != *key
                || (!v.is_finite()
                    && !(s.nonfinite == crate::NonFinite::PositiveInfinity && *v == f64::INFINITY))
        })
    {
        return fail(
            MetricError::new(ErrorKind::Internal)
                .with_message("native scalar outputs violate their declaration"),
        );
    }
    if opts.map.is_some() && result.maps().is_empty() {
        return fail(MetricError::new(ErrorKind::NoMap));
    }
    Ok(Evaluated {
        id,
        outputs,
        result,
    })
}
/// Run work under one owned child phase and hand off the actual outcome. The
/// caller's borrowed parent remains its responsibility. Planning rejection uses
/// how-far's diagnostic, cancellation-preserving untracked fallback.
#[track_caller]
pub(crate) fn operation<T>(
    opts: EvalOpts<'_>,
    label: &'static str,
    work: impl FnOnce(EvalOpts<'_>) -> MetricResult<T>,
) -> MetricResult<T> {
    let mut stages = how_far::Stages::new(
        opts.pulse,
        &[how_far::PhaseSpec::new(label, 1, how_far::Total::Unknown)],
    );
    let result = stages.run_classified(crate::is_stopped, |pulse| work(opts.with_pulse(pulse)));
    stages.complete_classified(result, crate::is_stopped).at()
}
/// Admit, evaluate and validate every declared scalar while preserving native
/// maps and causes. Backend failures take precedence over a later stop.
#[track_caller]
pub fn score(
    metric: &mut dyn Metric,
    r: Image<'_>,
    d: Image<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<Evaluated> {
    operation(opts, "score image pair", |opts| {
        admit(&r, opts).at()?;
        admit(&d, opts).at()?;
        check_same_size(&r, &d).at()?;
        if !metric.accepts(&r.format()) || !metric.accepts(&d.format()) {
            return fail(MetricError::new(ErrorKind::UnsupportedInput));
        }
        let id = metric.id();
        let outputs = metric.outputs().into_owned();
        let result = metric.evaluate(r, d, opts).at()?;
        finish(id, outputs, result, opts).at()
    })
}
/// Admit and score a native no-reference metric without a synthetic reference.
#[track_caller]
pub fn score_no_reference(
    metric: &mut dyn crate::NoReferenceMetric,
    image: Image<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<Evaluated> {
    operation(opts, "score image", |opts| {
        admit(&image, opts).at()?;
        if !metric.accepts(&image.format()) {
            return fail(MetricError::new(ErrorKind::UnsupportedInput));
        }
        let id = metric.id();
        let outputs = metric.outputs().into_owned();
        let result = metric.evaluate_image(image, opts).at()?;
        finish(id, outputs, result, opts).at()
    })
}
/// Prepared work holding original identity and geometry, without retaining its
/// preparation pulse. Each evaluation creates an independent observation phase.
#[non_exhaustive]
pub struct PreparedReference<'a> {
    id: MetricId,
    outputs: Vec<ScoreSpec>,
    size: (u32, u32),
    inner: Box<dyn WarmReference + 'a>,
}
impl PreparedReference<'_> {
    /// Admit a candidate and return the complete native result.
    #[track_caller]
    pub fn score(&mut self, image: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Evaluated> {
        operation(opts, "score prepared image", |opts| {
            admit(&image, opts).at()?;
            if (image.width(), image.height()) != self.size {
                return fail(MetricError::new(ErrorKind::DimensionMismatch));
            }
            let result = self.inner.evaluate(image, opts).at()?;
            finish(self.id.clone(), self.outputs.clone(), result, opts).at()
        })
    }
}
/// Request real prepared work; explicit refusal is allowed. A cold replay must
/// remain a caller-selected fallback, with its own identity/resource accounting.
#[track_caller]
pub fn prepare_reference<'a>(
    metric: &'a mut dyn Metric,
    image: Image<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<PreparedReference<'a>> {
    operation(opts, "prepare image reference", |opts| {
        admit(&image, opts).at()?;
        if !metric.accepts(&image.format()) {
            return fail(MetricError::new(ErrorKind::UnsupportedInput));
        }
        let id = metric.id();
        let outputs = metric.outputs().into_owned();
        let size = (image.width(), image.height());
        let inner = metric.warm_reference(image, opts).at()?;
        if !inner.reuses_reference_work() {
            return fail(
                MetricError::new(ErrorKind::UnsupportedRequest)
                    .with_message("preparation returned cold replay"),
            );
        }
        opts.checkpoint().at()?;
        Ok(PreparedReference {
            id,
            outputs,
            size,
            inner,
        })
    })
}
