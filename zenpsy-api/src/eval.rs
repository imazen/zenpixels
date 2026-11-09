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
/// Native scalar semantics. No universal range or normalization is imposed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ScoreSpec {
    /// Stable identifier, unique within an assessment.
    pub key: Cow<'static, str>,
    /// Quality ordering.
    pub direction: Direction,
    /// Native unit or scale name.
    pub unit: Cow<'static, str>,
    /// Defined identical-pair value, if any.
    pub perfect: Option<f64>,
    /// Whether positive infinity is valid. NaN is never valid.
    pub positive_infinity: bool,
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
    /// Name the native unit or scale.
    pub fn with_unit(mut self, unit: impl Into<Cow<'static, str>>) -> Self {
        self.unit = unit.into();
        self
    }
    /// Set the identical-pair value, permitting positive infinity if specified.
    pub const fn with_perfect(mut self, value: f64) -> Self {
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
}
