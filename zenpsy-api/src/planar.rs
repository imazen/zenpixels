//! Optional planar ingress and owned preparation with lending comparisons.
use crate::metric::{operation, validate_map_views, validate_scores};
use crate::{
    ErrorKind, EvalOpts, Evaluation, EvaluationRef, Metric, MetricError, MetricResult, fail,
};
use whereat::ResultAtExt as _;
use zenpixels::{
    AlphaMode, ColorContext,
    plane::{PlanarLayout, PlanarSlice},
};

/// Optional native planar input. The packed [`Metric`] contract stays usable
/// independently; no implicit conversion or packed replay is supplied.
pub trait PlanarMetric: Metric {
    /// Score complete color-described planes, preserving native output.
    fn evaluate_planar(
        &mut self,
        reference: &PlanarSlice<'_>,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output>;
}
/// Reusable native planar work owned independently of its factory scorer.
/// The preparation input and its Pulse are not retained. Configuration borrows
/// already held by the factory may be carried in the concrete prepared type.
pub trait PreparePlanarReference: PlanarMetric {
    /// Actual owned reference work, suitable for storage in an encoder callback.
    type PlanarPrepared: PlanarWarmReference<Output = Self::Output>;
    /// Build native reference work. Subsequent calls use fresh controls.
    fn warm_planar_reference(
        &mut self,
        reference: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::PlanarPrepared>;
}
/// Prepared native planar scoring. Implementors may override lending to let
/// consumers inspect native scratch before it is recycled, avoiding map export.
/// Workers must join before return. After failure, reuse must succeed correctly
/// or refuse explicitly; stale assessments must never be returned.
pub trait PlanarWarmReference {
    /// Complete owned native assessment.
    type Output: Evaluation;
    /// Evaluate a candidate using actual prepared reference work.
    fn evaluate_planar(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output>;
    /// Invoke `consumer` exactly once on success, while all borrowed native
    /// arrays remain valid. The default borrows an owned result; an override
    /// can lend scratch directly. A failed operation need not invoke it.
    fn with_evaluation(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        consumer: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        let output = self.evaluate_planar(candidate, opts)?;
        consumer(EvaluationRef::new(&output));
        Ok(())
    }
}
/// Checked owned planar work, binding image geometry, plane interpretation and
/// color/luminance to preparation. Strides may differ for every candidate.
pub struct PreparedPlanarReference<P> {
    inner: P,
    size: (u32, u32),
    layout: PlanarLayout,
    color: ColorContext,
    alpha: Option<AlphaMode>,
}
impl<P: PlanarWarmReference> PreparedPlanarReference<P> {
    fn admit(&self, candidate: &PlanarSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<()> {
        admit(candidate, opts)?;
        if (candidate.width(), candidate.height()) != self.size {
            return fail(MetricError::new(ErrorKind::DimensionMismatch));
        }
        if candidate.layout() != &self.layout
            || candidate.color_context() != &self.color
            || candidate.alpha_mode() != self.alpha
        {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_message("prepared planar interpretation changed"),
            );
        }
        Ok(())
    }
    /// Produce a checked owned native assessment.
    pub fn score(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<P::Output> {
        operation(opts, "score prepared planar image", |opts| {
            self.admit(candidate, opts)?;
            let output = self.inner.evaluate_planar(candidate, opts).at()?;
            let view = EvaluationRef::new(&output);
            validate_scores(view)?;
            validate_map_views(view, candidate.width(), candidate.height(), opts)?;
            opts.checkpoint()?;
            Ok(output)
        })
    }
    /// Consume a checked borrowed assessment before native scratch is recycled.
    /// Returned `T` cannot borrow the temporary view. The callback is invoked
    /// once at most; callers commit side effects only after this call succeeds.
    pub fn with_evaluation<T>(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        consume: impl FnOnce(EvaluationRef<'_>) -> MetricResult<T>,
    ) -> MetricResult<T> {
        operation(opts, "inspect prepared planar image", |opts| {
            self.admit(candidate, opts)?;
            let mut consume = Some(consume);
            let mut output = None;
            let mut repeated = false;
            self.inner
                .with_evaluation(candidate, opts, &mut |view| {
                    let Some(consume) = consume.take() else {
                        repeated = true;
                        return;
                    };
                    output = Some((|| {
                        validate_scores(view)?;
                        validate_map_views(view, candidate.width(), candidate.height(), opts)?;
                        opts.checkpoint()?;
                        consume(view)
                    })());
                })
                .at()?;
            if repeated {
                return fail(
                    MetricError::new(ErrorKind::Internal).with_message("lending callback repeated"),
                );
            }
            let output = output.ok_or_else(|| {
                whereat::at(
                    MetricError::new(ErrorKind::Internal).with_message("lending callback omitted"),
                )
            })??;
            opts.checkpoint()?;
            Ok(output)
        })
    }
}
fn admit(image: &PlanarSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<()> {
    opts.checkpoint()?;
    if image.width() == 0
        || image.height() == 0
        || matches!(opts.map_request(), crate::MapRequest::Named(""))
    {
        return fail(MetricError::new(ErrorKind::InvalidInput));
    }
    if opts
        .max_pixels()
        .is_some_and(|n| u64::from(image.width()) * u64::from(image.height()) > n)
    {
        return fail(MetricError::new(ErrorKind::Resource));
    }
    Ok(())
}
/// Checked native planar pair scoring. Color/encoding support is decided by
/// the metric, which must refuse unsupported interpretation explicitly.
pub fn score_planar<M: PlanarMetric + ?Sized>(
    metric: &mut M,
    reference: &PlanarSlice<'_>,
    candidate: &PlanarSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<M::Output> {
    operation(opts, "score planar image pair", |opts| {
        admit(reference, opts)?;
        admit(candidate, opts)?;
        if (reference.width(), reference.height()) != (candidate.width(), candidate.height()) {
            return fail(MetricError::new(ErrorKind::DimensionMismatch));
        }
        let output = metric.evaluate_planar(reference, candidate, opts).at()?;
        let view = EvaluationRef::new(&output);
        validate_scores(view)?;
        validate_map_views(view, candidate.width(), candidate.height(), opts)?;
        opts.checkpoint()?;
        Ok(output)
    })
}
/// Prepare owned planar work, binding candidate interpretation and geometry.
pub fn prepare_planar_reference<M: PreparePlanarReference + ?Sized>(
    metric: &mut M,
    reference: &PlanarSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<PreparedPlanarReference<M::PlanarPrepared>> {
    operation(opts, "prepare planar image reference", |opts| {
        admit(reference, opts)?;
        let inner = metric.warm_planar_reference(reference, opts).at()?;
        opts.checkpoint()?;
        Ok(PreparedPlanarReference {
            inner,
            size: (reference.width(), reference.height()),
            layout: reference.layout().clone(),
            color: reference.color_context().clone(),
            alpha: reference.alpha_mode(),
        })
    })
}

impl<P: PlanarWarmReference> PlanarWarmReference for PreparedPlanarReference<P> {
    type Output = P::Output;
    fn evaluate_planar(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.score(candidate, opts)
    }
    fn with_evaluation(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        consume: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        PreparedPlanarReference::with_evaluation(self, candidate, opts, |view| {
            consume(view);
            Ok(())
        })
    }
}
