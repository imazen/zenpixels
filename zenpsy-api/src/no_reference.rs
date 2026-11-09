//! One-image scoring and explicit reference/candidate pairing.
use crate::{
    EvalOpts, Evaluation, Metric, MetricId, MetricResult, PrepareReference, Score, ScoreSpec,
    WarmReference, score_image,
};
use alloc::{format, sync::Arc, vec::Vec};
use zenpixels::PixelSlice;

/// A configured one-image quality metric. Uses the same native assessments,
/// optional maps and operation controls as full-reference scoring.
pub trait NoReferenceMetric {
    /// Complete native result, without mandatory boxing.
    type Output: Evaluation;
    /// Effective configuration and actual implementation provenance.
    fn id(&self) -> MetricId;
    /// Score one image; use [`crate::score_image`] for checked calls.
    fn evaluate_image(
        &mut self,
        image: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output>;
}

/// Explicitly run a one-image metric on both inputs. Candidate outputs keep
/// their keys; reference outputs append `_ref`. Only candidate maps are
/// requested. Preparation caches the actual reference assessment once.
pub struct PairedNoReference<M>(M);
impl<M> PairedNoReference<M> {
    /// Wrap a one-image metric for pair consumers.
    pub const fn new(inner: M) -> Self {
        Self(inner)
    }
}

/// Both original native assessments, sharing cached reference work without
/// cloning native arrays. The common view exposes candidate scores/maps first.
pub struct PairedEvaluation<T> {
    reference: Arc<T>,
    candidate: T,
    reference_specs: Arc<[ScoreSpec]>,
}
impl<T> PairedEvaluation<T> {
    /// Original reference assessment (score-only).
    pub fn reference(&self) -> &T {
        &self.reference
    }
    /// Original candidate assessment, including requested native diagnostics.
    pub fn candidate(&self) -> &T {
        &self.candidate
    }
}
impl<T: Evaluation> Evaluation for PairedEvaluation<T> {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.candidate.scores().chain(
            self.reference_specs
                .iter()
                .zip(self.reference.scores())
                .map(|(spec, score)| Score::new(spec, score.value)),
        )
    }
    fn maps(&self) -> impl Iterator<Item = crate::MapView<'_>> {
        self.candidate.maps()
    }
}
type Reference<T> = (Arc<T>, Arc<[ScoreSpec]>);
fn reference<M: NoReferenceMetric>(
    metric: &mut M,
    image: &PixelSlice<'_>,
    opts: EvalOpts<'_>,
) -> MetricResult<Reference<M::Output>> {
    let result = score_image(metric, image, opts.without_diagnostics())?;
    let specs = result
        .scores()
        .map(|score| {
            score
                .spec
                .clone()
                .with_key(format!("{}_ref", score.spec.key()))
        })
        .collect::<Vec<_>>()
        .into();
    Ok((Arc::new(result), specs))
}
impl<M: NoReferenceMetric> Metric for PairedNoReference<M> {
    type Output = PairedEvaluation<M::Output>;
    fn id(&self) -> MetricId {
        self.0.id()
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let (reference, reference_specs) = reference(&mut self.0, r, opts)?;
        let candidate = score_image(&mut self.0, d, opts)?;
        Ok(PairedEvaluation {
            reference,
            candidate,
            reference_specs,
        })
    }
}
/// Actual cached reference result and exclusive borrow of a one-image scorer.
/// Construct through [`crate::prepare_reference`] on [`PairedNoReference`].
pub struct PairedReference<'a, M: NoReferenceMetric> {
    metric: &'a mut M,
    reference: Arc<M::Output>,
    reference_specs: Arc<[ScoreSpec]>,
}
impl<M: NoReferenceMetric> PrepareReference for PairedNoReference<M> {
    type Prepared<'a>
        = PairedReference<'a, M>
    where
        Self: 'a;
    fn warm_reference(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        let (reference, reference_specs) = reference(&mut self.0, r, opts)?;
        Ok(PairedReference {
            metric: &mut self.0,
            reference,
            reference_specs,
        })
    }
}
impl<M: NoReferenceMetric> WarmReference for PairedReference<'_, M> {
    type Output = PairedEvaluation<M::Output>;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        let candidate = score_image(self.metric, d, opts)?;
        Ok(PairedEvaluation {
            reference: Arc::clone(&self.reference),
            candidate,
            reference_specs: Arc::clone(&self.reference_specs),
        })
    }
}
