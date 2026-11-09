//! Typed comparisons and optional reusable reference work.
use crate::{ErrorKind, Evaluation, MetricError, MetricId, MetricResult, fail};
use core::fmt;
use how_far::{Complete, NoPulse, Pulse, Stop};
use whereat::ResultAtExt as _;
use zenpixels::PixelSlice;
/// Explicit native diagnostic request. A named request requires a common map
/// with this exact key; absence is an error, never successful empty output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapRequest<'a> {
    /// Score-only work.
    None,
    /// Native diagnostics; richer native results may omit common map views.
    Native,
    /// A specific common map, in its native units and placement.
    Named(&'a str),
}

/// Operation-local controls. Diagnostic arrays are off by default.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct EvalOpts<'a> {
    /// Cancellation/progress, never retained beyond the operation.
    pulse: &'a dyn Pulse,
    /// Active-pixel ceiling per input, excluding native scratch.
    max_pixels: Option<u64>,
    maps: MapRequest<'a>,
}
impl fmt::Debug for EvalOpts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalOpts")
            .field("maps", &self.maps)
            .field("max_pixels", &self.max_pixels)
            .finish_non_exhaustive()
    }
}
impl Default for EvalOpts<'_> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a> EvalOpts<'a> {
    /// Score-only work, without a cancellation source or pixel ceiling.
    pub const fn new() -> EvalOpts<'static> {
        EvalOpts {
            pulse: &NoPulse,
            max_pixels: None,
            maps: MapRequest::None,
        }
    }
    /// Opt into native diagnostic arrays. The caller bounds the witness set
    /// and storage budget. Unsupported diagnostics must be refused.
    pub const fn with_native_map(mut self) -> Self {
        self.maps = MapRequest::Native;
        self
    }
    pub(crate) const fn without_diagnostics(mut self) -> Self {
        self.maps = MapRequest::None;
        self
    }
    /// Require a common native map by its stable key. Implementations may
    /// compute shared components, but must not substitute another map.
    pub fn with_map<'b>(self, key: &'b str) -> EvalOpts<'b>
    where
        'a: 'b,
    {
        EvalOpts {
            pulse: self.pulse,
            max_pixels: self.max_pixels,
            maps: MapRequest::Named(key),
        }
    }
    /// Requested diagnostic selection.
    pub const fn map_request(&self) -> MapRequest<'a> {
        self.maps
    }
    /// Whether diagnostics were explicitly requested.
    pub const fn diagnostics(&self) -> bool {
        !matches!(self.maps, MapRequest::None)
    }
    /// Replace operation-local observation; never complete its parent.
    pub fn with_pulse<'b>(self, pulse: &'b dyn Pulse) -> EvalOpts<'b>
    where
        'a: 'b,
    {
        EvalOpts {
            pulse,
            max_pixels: self.max_pixels,
            maps: self.maps,
        }
    }
    /// Limit active pixels per input.
    pub const fn with_max_pixels(mut self, limit: u64) -> Self {
        self.max_pixels = Some(limit);
        self
    }
    /// Operation-local progress and cancellation source.
    pub const fn pulse(&self) -> &'a dyn Pulse {
        self.pulse
    }
    /// Active-pixel limit, when configured.
    pub const fn max_pixels(&self) -> Option<u64> {
        self.max_pixels
    }
    /// Borrow cancellation for native kernels accepting only Stop.
    pub fn stop(&self) -> &'a dyn Stop {
        self.pulse
    }
    /// Check cancellation between native phases.
    #[track_caller]
    pub fn checkpoint(&self) -> MetricResult<()> {
        self.pulse
            .check()
            .map_err(|r| whereat::at(MetricError::stopped(r)))
    }
}
/// Configured full-reference metric over actual strided, colour-described
/// inputs. Reject unsupported precision, alpha and HDR interpretation explicitly.
pub trait Metric {
    /// Native structured assessment; no mandatory erasure or boxing.
    type Output: Evaluation;
    /// Effective scoring configuration and actual implementation provenance.
    fn id(&self) -> MetricId;
    /// Native comparison. Poll controls and join owned workers before return.
    /// Use [`score`] for admission and returned-score checks.
    fn evaluate(
        &mut self,
        reference: &PixelSlice<'_>,
        candidate: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output>;
}
/// Optional actual reference precomputation, never implicit cold replay.
pub trait PrepareReference: Metric {
    /// Retained work may borrow the scorer, but not the input or preparation
    /// pulse. Each subsequent evaluation receives fresh controls.
    type Prepared<'a>: WarmReference<Output = Self::Output>
    where
        Self: 'a;
    /// Prepare work preserving cold-result scalar bits.
    fn warm_reference(
        &mut self,
        reference: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>>;
}
/// Native prepared work. Errors leave it reusable or make later calls refuse;
/// never return stale results after failure.
pub trait WarmReference {
    /// Same structured result as the corresponding cold scorer.
    type Output: Evaluation;
    /// Score with fresh operation-local controls.
    fn evaluate(
        &mut self,
        candidate: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output>;
}
/// Checked prepared work with retained geometry and no per-call identity copies.
pub struct PreparedReference<P> {
    size: (u32, u32),
    inner: P,
}
impl<P: WarmReference> PreparedReference<P> {
    /// Check candidate dimensions and return a validated native assessment.
    #[track_caller]
    pub fn score(&mut self, image: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<P::Output> {
        operation(opts, "score prepared image", |opts| {
            admit(image, opts)?;
            if (image.width(), image.rows()) != self.size {
                return fail(MetricError::new(ErrorKind::DimensionMismatch));
            }
            let result = self.inner.evaluate(image, opts).at()?;
            validate(&result)?;
            validate_maps(&result, image.width(), image.rows(), opts)?;
            opts.checkpoint()?;
            Ok(result)
        })
    }
}
#[track_caller]
fn admit(image: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<()> {
    admit_geometry(image.width(), image.rows(), opts)
}
/// Shared operation admission for packed and native planar views. Storage and
/// interpretation have already been bound by the respective zenpixels view.
#[track_caller]
pub(crate) fn admit_geometry(width: u32, height: u32, opts: EvalOpts<'_>) -> MetricResult<()> {
    opts.checkpoint()?;
    if matches!(opts.maps, MapRequest::Named("")) {
        return fail(MetricError::new(ErrorKind::InvalidConfig).with_message("empty map request"));
    }
    if width == 0 || height == 0 {
        return fail(MetricError::new(ErrorKind::InvalidInput));
    }
    if opts
        .max_pixels
        .is_some_and(|limit| u64::from(width) * u64::from(height) > limit)
    {
        return fail(MetricError::new(ErrorKind::Resource));
    }
    Ok(())
}
#[track_caller]
fn validate(result: &impl Evaluation) -> MetricResult<()> {
    validate_scores(crate::EvaluationRef::new(result))
}

// Most native output sets fit inline. Reserve before pushing so overflow
// allocation remains fallible; TinyVec owns the storage mechanics.
#[track_caller]
fn insert_name<'a>(names: &mut tinyvec::TinyVec<[&'a str; 64]>, key: &'a str) -> MetricResult<()> {
    if key.is_empty() || names.contains(&key) {
        return fail(
            MetricError::new(ErrorKind::Internal).with_message("empty or duplicate output key"),
        );
    }
    names
        .try_reserve(1)
        .map_err(|e| whereat::at(MetricError::from_detail(ErrorKind::Resource, e)))?;
    names.push(key);
    Ok(())
}
#[track_caller]
pub(crate) fn validate_scores(result: crate::EvaluationRef<'_>) -> MetricResult<()> {
    let mut names = tinyvec::TinyVec::new();
    let mut status = Ok(());
    let mut count = 0;
    result.visit_scores(|score| {
        if status.is_err() {
            return;
        }
        count += 1;
        if !score.value.is_finite()
            && !(score.spec.positive_infinity() && score.value == f64::INFINITY)
        {
            status =
                fail(MetricError::new(ErrorKind::Internal).with_message("invalid native score"));
        } else {
            status = insert_name(&mut names, score.spec.key());
        }
    });
    status?;
    if count == 0 {
        return fail(MetricError::new(ErrorKind::Internal).with_message("empty native score set"));
    }
    Ok(())
}
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
/// Admit image geometry and validate scores, preserving typed native results
/// and located failures. Native errors take precedence over a later stop.
#[track_caller]
pub fn score<M: Metric + ?Sized>(
    metric: &mut M,
    reference: &PixelSlice<'_>,
    candidate: &PixelSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<M::Output> {
    operation(opts, "score image pair", |opts| {
        admit(reference, opts)?;
        admit(candidate, opts)?;
        if (reference.width(), reference.rows()) != (candidate.width(), candidate.rows()) {
            return fail(MetricError::new(ErrorKind::DimensionMismatch));
        }
        let result = metric.evaluate(reference, candidate, opts).at()?;
        validate(&result)?;
        validate_maps(&result, candidate.width(), candidate.rows(), opts)?;
        opts.checkpoint()?;
        Ok(result)
    })
}
/// Prepare native reference work with admission and an owned child phase.
#[track_caller]
pub fn prepare_reference<'a, M: PrepareReference + ?Sized>(
    metric: &'a mut M,
    reference: &PixelSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<PreparedReference<M::Prepared<'a>>> {
    operation(opts, "prepare image reference", |opts| {
        admit(reference, opts)?;
        let inner = metric.warm_reference(reference, opts).at()?;
        opts.checkpoint()?;
        Ok(PreparedReference {
            size: (reference.width(), reference.rows()),
            inner,
        })
    })
}

/// Admit one image and validate its complete native no-reference assessment.
#[track_caller]
pub fn score_image<M: crate::NoReferenceMetric + ?Sized>(
    metric: &mut M,
    image: &PixelSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<M::Output> {
    operation(opts, "score one image", |opts| {
        admit(image, opts)?;
        let result = metric.evaluate_image(image, opts).at()?;
        validate(&result)?;
        validate_maps(&result, image.width(), image.rows(), opts)?;
        opts.checkpoint()?;
        Ok(result)
    })
}
#[track_caller]
fn validate_maps(
    result: &impl Evaluation,
    width: u32,
    height: u32,
    opts: EvalOpts<'_>,
) -> MetricResult<()> {
    validate_map_views(crate::EvaluationRef::new(result), width, height, opts)
}
#[track_caller]
pub(crate) fn validate_map_views(
    result: crate::EvaluationRef<'_>,
    width: u32,
    height: u32,
    opts: EvalOpts<'_>,
) -> MetricResult<()> {
    if !opts.diagnostics() {
        return Ok(());
    }
    let mut names = tinyvec::TinyVec::new();
    let mut found = false;
    let mut status = Ok(());
    result.visit_maps(|map| {
        if status.is_err() {
            return;
        }
        if map.values.width() == 0
            || map.values.height() == 0
            || map.grid.input_width() != width
            || map.grid.input_height() != height
        {
            status =
                fail(MetricError::new(ErrorKind::Internal).with_message("invalid native map view"));
            return;
        }
        found |= matches!(opts.maps, MapRequest::Named(key) if key == map.spec.key());
        status = insert_name(&mut names, map.spec.key());
    });
    status?;
    if matches!(opts.maps, MapRequest::Named(_)) && !found {
        return fail(
            MetricError::new(ErrorKind::NoMap)
                .with_message("requested common map was not returned"),
        );
    }
    Ok(())
}
