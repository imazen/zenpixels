//! Optional erasure at collection boundaries, preserving native results.
use crate::{
    EvalOpts, Evaluation, MapView, Metric, MetricId, MetricResult, NoReferenceMetric,
    PrepareReference, PreparedReference, Score, WarmReference,
};
use alloc::boxed::Box;
use core::any::{Any, TypeId};
use zenpixels::PixelSlice;

trait NativeEvaluation {
    fn scores(&self) -> Box<dyn Iterator<Item = Score<'_>> + '_>;
    fn maps(&self) -> Box<dyn Iterator<Item = MapView<'_>> + '_>;
    fn as_any(&self) -> &dyn Any;
}
impl<T: Evaluation + 'static> NativeEvaluation for T {
    fn scores(&self) -> Box<dyn Iterator<Item = Score<'_>> + '_> {
        Box::new(Evaluation::scores(self))
    }
    fn maps(&self) -> Box<dyn Iterator<Item = MapView<'_>> + '_> {
        Box::new(Evaluation::maps(self))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An owned native assessment with common borrowed score access. The native
/// result is moved into a box; scalar descriptors and diagnostic arrays are not
/// copied. Score iteration uses a short-lived erased iterator allocation.
pub struct ErasedEvaluation {
    inner: Box<dyn NativeEvaluation>,
}
impl ErasedEvaluation {
    /// Erase an assessment. Repeated erasure preserves the original native
    /// downcast target rather than retaining nested erased-result envelopes.
    pub fn new<T: Evaluation + 'static>(native: T) -> Self {
        if TypeId::of::<T>() == TypeId::of::<Self>() {
            let native: Box<dyn Any> = Box::new(native);
            return match native.downcast::<Self>() {
                Ok(native) => *native,
                Err(_) => unreachable!("matching erased result type"),
            };
        }
        Self {
            inner: Box::new(native),
        }
    }
    /// Borrow the original concrete assessment, including any explicitly
    /// requested native diagnostic data. Returns None for a different type.
    pub fn native<T: 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref()
    }
}
impl Evaluation for ErasedEvaluation {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.inner.scores()
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.inner.maps()
    }
}

/// A heterogeneous metric object with a common structured result. The lifetime
/// permits a scorer to borrow configuration; only its owned result must be
/// static for native downcasting. Use [`crate::score`] for checked evaluation.
///
/// ```
/// use zenpsy_api::{DynMetric, ErasedEvaluation, EvalOpts, MetricResult, score};
/// use zenpsy_api::zenpixels::PixelSlice;
/// fn run_all(metrics: &mut [Box<DynMetric<'_>>], r: &PixelSlice<'_>, d: &PixelSlice<'_>)
///     -> MetricResult<Vec<ErasedEvaluation>>
/// {
///     metrics.iter_mut().map(|m| score(m, r, d, EvalOpts::new())).collect()
/// }
/// ```
pub type DynMetric<'a> = dyn Metric<Output = ErasedEvaluation> + 'a;

/// Stackable wrapper normalizing only the result type. It forwards identity,
/// actual pixel views and operation controls. Typed callers keep their native
/// result by omitting this wrapper. No Send/Sync bounds are added.
pub struct Erased<M>(M);
impl<M> Erased<M> {
    /// Wrap a typed metric or native prepared reference.
    pub const fn new(inner: M) -> Self {
        Self(inner)
    }
}
impl<M: Metric> Metric for Erased<M>
where
    M::Output: 'static,
{
    type Output = ErasedEvaluation;
    fn id(&self) -> MetricId {
        self.0.id()
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.0.evaluate(r, d, opts).map(ErasedEvaluation::new)
    }
}
impl<P: WarmReference> WarmReference for Erased<P>
where
    P::Output: 'static,
{
    type Output = ErasedEvaluation;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        self.0.evaluate(d, opts).map(ErasedEvaluation::new)
    }
}
impl<M: PrepareReference> PrepareReference for Erased<M>
where
    M::Output: 'static,
{
    type Prepared<'a>
        = Erased<M::Prepared<'a>>
    where
        Self: 'a;
    fn warm_reference(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        self.0.warm_reference(r, opts).map(Erased::new)
    }
}

/// Object-safe optional preparation. Only metrics with native prepared work
/// implement this extension; a cold metric is not given a replay fallback.
pub trait DynPrepareReference: Metric<Output = ErasedEvaluation> {
    /// Prepare checked reference work. Returned objects retain dimensions and
    /// use fresh candidate controls, without retaining the preparation pulse.
    fn prepare(
        &mut self,
        reference: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference<Output = ErasedEvaluation> + '_>>;
}
struct Checked<P>(PreparedReference<P>);
impl<P: WarmReference> WarmReference for Checked<P> {
    type Output = P::Output;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        self.0.score(d, opts)
    }
}
impl<M: PrepareReference> DynPrepareReference for Erased<M>
where
    M::Output: 'static,
{
    fn prepare(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference<Output = ErasedEvaluation> + '_>> {
        crate::prepare_reference(self, r, opts)
            .map(|p| Box::new(Checked(p)) as Box<dyn WarmReference<Output = ErasedEvaluation> + '_>)
    }
}

// Forwarding lets boxes participate in generic wrapper stacks without adding
// observation phases, copying pixels or reconstructing native assessments.
impl<M: Metric + ?Sized> Metric for Box<M> {
    type Output = M::Output;
    fn id(&self) -> MetricId {
        (**self).id()
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        (**self).evaluate(r, d, opts)
    }
}
impl<P: WarmReference + ?Sized> WarmReference for Box<P> {
    type Output = P::Output;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        (**self).evaluate(d, opts)
    }
}
impl<M: PrepareReference + ?Sized> PrepareReference for Box<M> {
    type Prepared<'a>
        = M::Prepared<'a>
    where
        Self: 'a;
    fn warm_reference(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        (**self).warm_reference(r, opts)
    }
}

/// A heterogeneous one-image metric object. Scorer configuration may be borrowed;
/// the erased owned native assessment must be static for downcasting.
pub type DynNoReferenceMetric<'a> = dyn NoReferenceMetric<Output = ErasedEvaluation> + 'a;
impl<M: NoReferenceMetric> NoReferenceMetric for Erased<M>
where
    M::Output: 'static,
{
    type Output = ErasedEvaluation;
    fn id(&self) -> MetricId {
        self.0.id()
    }
    fn evaluate_image(
        &mut self,
        image: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.0
            .evaluate_image(image, opts)
            .map(ErasedEvaluation::new)
    }
}
impl<M: NoReferenceMetric + ?Sized> NoReferenceMetric for Box<M> {
    type Output = M::Output;
    fn id(&self) -> MetricId {
        (**self).id()
    }
    fn evaluate_image(
        &mut self,
        image: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        (**self).evaluate_image(image, opts)
    }
}
