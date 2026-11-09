//! Borrowed score views without prescribing native result storage.
use alloc::borrow::Cow;
/// Quality ordering; diagnostic features need not have one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Direction {
    /// Higher indicates better quality.
    Higher,
    /// Lower indicates better quality.
    Lower,
    /// No quality ordering is defined.
    Unordered,
}
/// Native output semantics, shared by scalars and maps. No universal range
/// or normalization is imposed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ScoreSpec {
    /// Stable identifier, unique within an assessment.
    key: Cow<'static, str>,
    /// Quality ordering.
    direction: Direction,
    /// Native unit or scale name.
    unit: Cow<'static, str>,
    /// Defined identical-pair value, if any.
    perfect: Option<f64>,
    /// Whether positive infinity is valid. NaN is never valid.
    positive_infinity: bool,
}
impl ScoreSpec {
    /// Declare a scalar without an assumed identity value or range.
    pub const fn new(key: &'static str, direction: Direction) -> Self {
        Self {
            key: Cow::Borrowed(key),
            direction,
            unit: Cow::Borrowed(""),
            perfect: None,
            positive_infinity: false,
        }
    }
    /// Replace the stable identifier, including a name computed at runtime.
    pub fn with_key(mut self, key: impl Into<Cow<'static, str>>) -> Self {
        self.key = key.into();
        self
    }
    /// Stable output identifier.
    pub fn key(&self) -> &str {
        &self.key
    }
    /// Native quality ordering.
    pub const fn direction(&self) -> Direction {
        self.direction
    }
    /// Native unit or scale name.
    pub fn unit(&self) -> &str {
        &self.unit
    }
    /// Defined identical-pair value, if any.
    pub const fn perfect(&self) -> Option<f64> {
        self.perfect
    }
    /// Whether positive infinity is accepted as a scalar value.
    pub const fn positive_infinity(&self) -> bool {
        self.positive_infinity
    }
    /// Configure infinity independently of any identical-pair value.
    pub const fn with_positive_infinity(mut self, allowed: bool) -> Self {
        self.positive_infinity = allowed;
        self
    }
    /// Name the native unit or scale.
    pub fn with_unit(mut self, unit: impl Into<Cow<'static, str>>) -> Self {
        self.unit = unit.into();
        self
    }
    /// Set the identical-pair value, permitting positive infinity if specified.
    ///
    /// # Panics
    /// Panics for NaN or negative infinity, which cannot be valid scores.
    pub const fn with_perfect(mut self, value: f64) -> Self {
        assert!(
            !value.is_nan() && value != f64::NEG_INFINITY,
            "invalid identical-pair value"
        );
        self.perfect = Some(value);
        self.positive_infinity = value == f64::INFINITY;
        self
    }
}
/// Unmodified scalar with borrowed semantics.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Score<'a> {
    /// Scalar meaning.
    pub spec: &'a ScoreSpec,
    /// Native value.
    pub value: f64,
}
impl<'a> Score<'a> {
    /// Borrow a descriptor for a native value.
    pub const fn new(spec: &'a ScoreSpec, value: f64) -> Self {
        Self { spec, value }
    }
}
/// Common access to a concrete assessment. Additional native fields remain
/// accessible on its concrete type. Iteration need not allocate. A successful
/// assessment has at least one uniquely named score, in stable output order.
pub trait Evaluation {
    /// All scalar outputs and their semantics, headline first when applicable.
    fn scores(&self) -> impl Iterator<Item = Score<'_>>;
    /// Visit scores without allocating an erased iterator. Implementations
    /// overriding this method must expose exactly the same order and values.
    fn visit_scores<'a>(&'a self, visitor: &mut dyn FnMut(Score<'a>)) {
        self.scores().for_each(visitor);
    }
    /// Visit maps without allocating an erased iterator, with the same views
    /// and order as [`Self::maps`].
    fn visit_maps<'a>(&'a self, visitor: &mut dyn FnMut(crate::MapView<'a>)) {
        self.maps().for_each(visitor);
    }
    /// Borrow native map planes when diagnostics were requested. The default
    /// publishes none; a metric without diagnostics must refuse a request.
    /// Native fields remain accessible even when no common map view is exposed.
    fn maps(&self) -> impl Iterator<Item = crate::MapView<'_>> {
        core::iter::empty()
    }
}

/// A temporary assessment borrowed during a lending comparison. The view
/// cannot outlive the callback or retain metric scratch. Native typed results
/// remain available from the ordinary owned scoring interface.
#[derive(Clone, Copy)]
pub struct EvaluationRef<'a> {
    inner: &'a dyn EvaluationAccess,
}
impl<'a> EvaluationRef<'a> {
    /// Borrow a concrete assessment without boxing or copying its maps.
    pub fn new<T: Evaluation>(value: &'a T) -> Self {
        Self { inner: value }
    }
    /// Visit every named scalar in native output order.
    pub fn visit_scores(self, mut visitor: impl FnMut(Score<'a>)) {
        self.inner.visit_scores(&mut visitor);
    }
    /// Visit every native map in output order.
    pub fn visit_maps(self, mut visitor: impl FnMut(crate::MapView<'a>)) {
        self.inner.visit_maps(&mut visitor);
    }
}
pub(crate) trait EvaluationAccess {
    fn visit_scores<'a>(&'a self, visitor: &mut dyn FnMut(Score<'a>));
    fn visit_maps<'a>(&'a self, visitor: &mut dyn FnMut(crate::MapView<'a>));
}
impl<T: Evaluation> EvaluationAccess for T {
    fn visit_scores<'a>(&'a self, visitor: &mut dyn FnMut(Score<'a>)) {
        Evaluation::visit_scores(self, visitor);
    }
    fn visit_maps<'a>(&'a self, visitor: &mut dyn FnMut(crate::MapView<'a>)) {
        Evaluation::visit_maps(self, visitor);
    }
}
