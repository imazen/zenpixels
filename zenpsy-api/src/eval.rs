//! The result object a metric returns, and what its numbers mean.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;

use zenpixels::PixelCow;

use crate::error::{ErrorKind, MetricError, MetricResult, fail};
use crate::map::{
    MapFold, MapGeometry, MapPolarity, MapReduce, MapRequest, MapView, gray_f32_buffer, map_fold,
    map_subsample_f32,
};

/// What a scalar output means: direction and scale.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoreKind {
    /// `[0, 1]`-ish similarity; `1` = identical (SSIM family, FSIM,
    /// HaarPSI, VSI, IW-SSIM, VIF).
    Similarity,
    /// `0` = identical, higher = worse (MAD, GMSD, NLPD, butteraugli).
    Difference,
    /// Just-objectionable-difference scale; `10` = identical (cvvdp,
    /// HDR-VDP-3).
    Jod,
    /// `100` = identical (HDR-VDP-2's `Q`).
    Quality100,
    /// Higher = better, decibels (PSNR family).
    Decibels,
    /// Detection probability `[0, 1]`; `0` = identical.
    Probability,
    /// A diagnostic or intermediate feature with no quality direction
    /// (VMAF's elementary features, per-scale terms).
    Feature,
    /// No-reference quality of one image, higher = better (MUSIQ,
    /// ARNIQA). The scale is the metric's own; no value means "identical".
    Quality,
    /// No-reference distortion of one image, higher = worse (NIQE, PIQE,
    /// BRISQUE). The scale is the metric's own.
    Distortion,
}

impl ScoreKind {
    /// `Some(true)` when larger means closer to the reference (for the
    /// no-reference kinds: better quality); `None` for [`ScoreKind::Feature`].
    pub const fn higher_is_better(self) -> Option<bool> {
        match self {
            ScoreKind::Similarity
            | ScoreKind::Jod
            | ScoreKind::Quality100
            | ScoreKind::Decibels
            | ScoreKind::Quality => Some(true),
            ScoreKind::Difference | ScoreKind::Probability | ScoreKind::Distortion => Some(false),
            ScoreKind::Feature => None,
        }
    }

    /// The value an identical pair scores, when the scale has one
    /// (`Decibels` is unbounded: identical is `+∞`).
    pub const fn perfect(self) -> Option<f64> {
        match self {
            ScoreKind::Similarity => Some(1.0),
            ScoreKind::Difference | ScoreKind::Probability => Some(0.0),
            ScoreKind::Jod => Some(10.0),
            ScoreKind::Quality100 => Some(100.0),
            ScoreKind::Decibels => Some(f64::INFINITY),
            ScoreKind::Feature | ScoreKind::Quality | ScoreKind::Distortion => None,
        }
    }
}

/// Successful nonfinite behavior of one scalar output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NonFinite {
    /// Successful values must be finite.
    Refuse,
    /// Positive infinity is meaningful, for example identical-image PSNR.
    PositiveInfinity,
}
/// One declared scalar output: its key and meaning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ScoreSpec {
    /// Stable key, `[a-z0-9_]` (`"jod"`, `"fsimc"`, `"wmcs_0"`).
    pub key: Cow<'static, str>,
    /// What the number means.
    pub kind: ScoreKind,
    /// Native unit; empty when the metric defines no separate physical unit.
    pub unit: Cow<'static, str>,
    /// Whether positive infinity is meaningful; NaN is always refused.
    pub nonfinite: NonFinite,
}

impl ScoreSpec {
    /// A spec with a static key.
    pub const fn new(key: &'static str, kind: ScoreKind) -> Self {
        Self {
            key: Cow::Borrowed(key),
            kind,
            unit: Cow::Borrowed(""),
            nonfinite: if matches!(kind, ScoreKind::Decibels) {
                NonFinite::PositiveInfinity
            } else {
                NonFinite::Refuse
            },
        }
    }

    /// Attach a native output unit without normalizing the score.
    pub fn with_unit(mut self, unit: impl Into<Cow<'static, str>>) -> Self {
        self.unit = unit.into();
        self
    }

    /// Specify meaningful positive infinity independently of score kind.
    pub const fn with_nonfinite(mut self, policy: NonFinite) -> Self {
        self.nonfinite = policy;
        self
    }

    /// A spec with a computed key.
    pub fn owned(key: alloc::string::String, kind: ScoreKind) -> Self {
        Self {
            key: Cow::Owned(key),
            kind,
            unit: Cow::Borrowed(""),
            nonfinite: if matches!(kind, ScoreKind::Decibels) {
                NonFinite::PositiveInfinity
            } else {
                NonFinite::Refuse
            },
        }
    }
}

/// `(key, value)` pairs: the scalar outputs of one comparison.
pub type Scores<'a> = Box<dyn Iterator<Item = (Cow<'static, str>, f64)> + 'a>;

/// `(key, value)` pairs of metric metadata.
pub type Extras<'a> = Box<dyn Iterator<Item = (Cow<'static, str>, Cow<'static, str>)> + 'a>;

/// One comparison's output, owned by the caller.
///
/// `Send + Sync`, so results move to and are read from other threads.
/// Every method past [`Self::score`] and [`Self::as_any`] has a default;
/// adding methods is additive.
pub trait Evaluation: Send + Sync {
    /// The headline value: the first of [`crate::Metric::outputs`].
    fn score(&self) -> f64;

    /// Every scalar output, headline first, keyed as
    /// [`crate::Metric::outputs`] declares. Default: `("score", score())`.
    fn scores(&self) -> Scores<'_> {
        Box::new(core::iter::once((Cow::Borrowed("score"), self.score())))
    }

    /// The primary map, when one was requested and the metric has one.
    /// Its resolution is the metric's, not the input's: read
    /// [`MapView::geometry`].
    fn map(&self) -> Option<MapView<'_>> {
        None
    }

    /// Every map, primary first. Default: [`Self::map`] alone.
    fn maps(&self) -> Vec<MapView<'_>> {
        self.map().into_iter().collect()
    }

    /// Another scalar from the primary map, on demand. Always folds the
    /// metric's native map, whatever resolution [`Self::map`] serves.
    ///
    /// # Errors
    /// [`ErrorKind::NoMap`] without a map; otherwise as [`map_fold`].
    fn map_score(&self, fold: MapFold<'_>) -> MetricResult<f64> {
        match self.map() {
            Some(m) => map_fold(&m, fold),
            None => fail(MetricError::new(ErrorKind::NoMap)),
        }
    }

    /// Metric metadata with stable keys. Default: empty.
    fn extras(&self) -> Extras<'_> {
        Box::new(core::iter::empty())
    }

    /// Downcast to the concrete result type for its typed API.
    fn as_any(&self) -> &dyn core::any::Any;
}

/// A scalar-only result.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ScalarEval {
    /// `(key, value)` pairs, headline first.
    pub named: Vec<(Cow<'static, str>, f64)>,
}

impl ScalarEval {
    /// One output.
    pub fn single(key: &'static str, value: f64) -> Self {
        Self {
            named: alloc::vec![(Cow::Borrowed(key), value)],
        }
    }

    /// Append an output.
    pub fn push(&mut self, key: impl Into<Cow<'static, str>>, value: f64) {
        self.named.push((key.into(), value));
    }

    /// Builder form of [`Self::push`].
    #[must_use]
    pub fn with(mut self, key: impl Into<Cow<'static, str>>, value: f64) -> Self {
        self.push(key, value);
        self
    }
}

impl Evaluation for ScalarEval {
    fn score(&self) -> f64 {
        self.named.first().map_or(f64::NAN, |(_, v)| *v)
    }

    fn scores(&self) -> Scores<'_> {
        Box::new(self.named.iter().cloned())
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Serve `request` from an evaluation that computed its native maps.
///
/// `Native` passes through; `Subsampled` wraps the evaluation so
/// [`Evaluation::map`]/[`Evaluation::maps`] return reduced maps while
/// scores, extras, [`Evaluation::map_score`] and [`Evaluation::as_any`]
/// still reach the native result.
///
/// # Errors
/// [`ErrorKind::NoMap`] when the evaluation has no map;
/// [`ErrorKind::InvalidConfig`] for a zero factor.
#[track_caller]
pub fn apply_map_request(
    eval: Box<dyn Evaluation>,
    request: MapRequest,
) -> MetricResult<Box<dyn Evaluation>> {
    let (factor, reduce) = match request {
        MapRequest::Native => (1, MapReduce::Mean),
        MapRequest::Subsampled { factor, reduce } => (factor, reduce),
    };
    if factor == 0 {
        return fail(
            MetricError::new(ErrorKind::InvalidConfig).with_message("map subsample factor 0"),
        );
    }
    let mut maps = Vec::new();
    {
        let native = eval.maps();
        if native.is_empty() {
            return fail(MetricError::new(ErrorKind::NoMap));
        }
        if factor == 1 {
            drop(native);
            return Ok(eval);
        }
        for m in native {
            let (vals, w, h) =
                map_subsample_f32(&m.values(), m.width(), m.height(), factor, reduce)?;
            maps.push(OwnedMap {
                name: m.name,
                pixels: gray_f32_buffer(&vals, w, h)?,
                polarity: m.polarity,
                geometry: m.geometry.subsampled(factor),
            });
        }
    }
    Ok(Box::new(Resampled {
        inner: eval,
        maps,
        served: request,
    }))
}

struct OwnedMap {
    name: &'static str,
    pixels: zenpixels::PixelBuffer,
    polarity: MapPolarity,
    geometry: MapGeometry,
}

/// An evaluation serving reduced maps over its native result. Reach the
/// native result with [`Resampled::native`], or downcast through
/// [`Evaluation::as_any`] (which returns the native result).
#[non_exhaustive]
pub struct Resampled {
    inner: Box<dyn Evaluation>,
    maps: Vec<OwnedMap>,
    served: MapRequest,
}

impl Resampled {
    /// The native evaluation.
    pub fn native(&self) -> &dyn Evaluation {
        &*self.inner
    }
}

impl Evaluation for Resampled {
    fn score(&self) -> f64 {
        self.inner.score()
    }

    fn scores(&self) -> Scores<'_> {
        self.inner.scores()
    }

    fn map(&self) -> Option<MapView<'_>> {
        self.maps.first().map(|m| self.view(m))
    }

    fn maps(&self) -> Vec<MapView<'_>> {
        self.maps.iter().map(|m| self.view(m)).collect()
    }

    fn map_score(&self, fold: MapFold<'_>) -> MetricResult<f64> {
        self.inner.map_score(fold)
    }

    fn extras(&self) -> Extras<'_> {
        self.inner.extras()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self.inner.as_any()
    }
}

impl Resampled {
    fn view<'s>(&'s self, m: &'s OwnedMap) -> MapView<'s> {
        MapView {
            name: m.name,
            pixels: PixelCow::Borrowed(m.pixels.as_slice()),
            polarity: m.polarity,
            geometry: m.geometry,
            served: self.served,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WithMap(zenpixels::PixelBuffer);

    impl Evaluation for WithMap {
        fn score(&self) -> f64 {
            7.0
        }
        fn map(&self) -> Option<MapView<'_>> {
            MapView::new(
                "diff",
                self.0.as_slice(),
                MapPolarity::Difference,
                MapGeometry::identity(4, 2),
            )
            .ok()
        }
        fn as_any(&self) -> &dyn core::any::Any {
            self
        }
    }

    fn eval() -> Box<dyn Evaluation> {
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        Box::new(WithMap(gray_f32_buffer(&v, 4, 2).unwrap()))
    }

    #[test]
    fn subsampled_serves_small_map_but_folds_native() {
        let e = apply_map_request(eval(), MapRequest::subsampled(2)).unwrap();
        let m = e.map().unwrap();
        assert_eq!((m.width(), m.height()), (2, 1));
        assert_eq!(&*m.values(), &[2.5, 4.5]);
        assert_eq!(m.served, MapRequest::subsampled(2));
        assert_eq!(m.geometry, MapGeometry::identity(4, 2).subsampled(2));
        assert_eq!(e.map_score(MapFold::Max).unwrap(), 7.0);
        assert_eq!(e.score(), 7.0);
        assert!(e.as_any().downcast_ref::<WithMap>().is_some());
    }

    #[test]
    fn requests_fail_loudly() {
        let k = |e: Box<dyn Evaluation>, r| apply_map_request(e, r).err().unwrap().error().kind();
        assert_eq!(
            k(Box::new(ScalarEval::single("x", 1.0)), MapRequest::Native),
            ErrorKind::NoMap
        );
        assert_eq!(
            k(eval(), MapRequest::subsampled(0)),
            ErrorKind::InvalidConfig
        );
        assert!(
            apply_map_request(eval(), MapRequest::subsampled(1))
                .unwrap()
                .as_any()
                .is::<WithMap>()
        );
    }

    #[test]
    fn score_kind_semantics() {
        assert_eq!(ScoreKind::Difference.higher_is_better(), Some(false));
        assert_eq!(ScoreKind::Jod.perfect(), Some(10.0));
        assert_eq!(ScoreKind::Feature.higher_is_better(), None);
    }
}
