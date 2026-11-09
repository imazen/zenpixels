//! Typed comparisons and optional reusable reference work.
use crate::{ErrorKind, Evaluation, MetricError, MetricId, MetricResult, fail};
use core::fmt;
use how_far::{Complete, NoPulse, Pulse, Stop};
use whereat::ResultAtExt as _;
use zenpixels::PixelSlice;
/// Operation-local controls. Diagnostic arrays are off by default.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct EvalOpts<'a> {
    /// Cancellation/progress, never retained beyond the operation.
    pub pulse: &'a dyn Pulse,
    /// Active-pixel ceiling per input, excluding native scratch.
    pub max_pixels: Option<u64>,
    diagnostics: bool,
}
impl fmt::Debug for EvalOpts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalOpts")
            .field("diagnostics", &self.diagnostics)
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
            diagnostics: false,
        }
    }
    /// Opt into native diagnostic arrays. The caller bounds the witness set
    /// and storage budget. Unsupported diagnostics must be refused.
    pub const fn with_native_map(mut self) -> Self {
        self.diagnostics = true;
        self
    }
    /// Whether diagnostics were explicitly requested.
    pub const fn diagnostics(&self) -> bool {
        self.diagnostics
    }
    /// Replace operation-local observation; never complete its parent.
    pub fn with_pulse<'b>(self, pulse: &'b dyn Pulse) -> EvalOpts<'b> {
        EvalOpts {
            pulse,
            max_pixels: self.max_pixels,
            diagnostics: self.diagnostics,
        }
    }
    /// Limit active pixels per input.
    pub const fn with_max_pixels(mut self, limit: u64) -> Self {
        self.max_pixels = Some(limit);
        self
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
            opts.checkpoint()?;
            Ok(result)
        })
    }
}
#[track_caller]
fn admit(image: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<()> {
    opts.checkpoint()?;
    if image.width() == 0 || image.rows() == 0 {
        return fail(MetricError::new(ErrorKind::InvalidInput));
    }
    if opts
        .max_pixels
        .is_some_and(|limit| u64::from(image.width()) * u64::from(image.rows()) > limit)
    {
        return fail(MetricError::new(ErrorKind::Resource));
    }
    Ok(())
}
#[track_caller]
fn validate(result: &impl Evaluation) -> MetricResult<()> {
    let mut count = 0;
    for (index, score) in result.scores().enumerate() {
        count += 1;
        if score.spec.key.is_empty()
            || (!score.value.is_finite()
                && !(score.spec.positive_infinity && score.value == f64::INFINITY))
            || result
                .scores()
                .take(index)
                .any(|other| other.spec.key == score.spec.key)
        {
            return fail(
                MetricError::new(ErrorKind::Internal).with_message("invalid native score set"),
            );
        }
    }
    if count == 0 {
        return fail(MetricError::new(ErrorKind::Internal).with_message("empty native score set"));
    }
    Ok(())
}
#[track_caller]
fn operation<T>(
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
