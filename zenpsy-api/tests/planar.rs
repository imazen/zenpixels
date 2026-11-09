use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zenpsy_api::{
    zenpixels::{plane::*, *},
    *,
};

fn layout() -> PlanarLayout {
    PlanarLayout::new(
        [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue]
            .map(|r| PlaneSpec::new(r, PlaneEncoding::Float(FloatEncoding::NOMINAL)))
            .to_vec(),
        sample::ChromaSubsampling::Yuv444,
        sample::ChromaSiting::Unknown,
    )
    .unwrap()
}
fn color() -> ColorContext {
    ColorContext::from_cicp(Cicp::new(1, 8, 0, true))
}
fn planes(data: &[f32]) -> [AnyPlane<'_>; 3] {
    [AnyPlane::F32(PlaneRef::new(data, 2, 2, 3).unwrap()); 3]
}
struct Output {
    values: [f32; 4],
    sum: f64,
}
static SCORE: ScoreSpec = ScoreSpec::new("sum", Direction::Lower);
static MAP: ScoreSpec = ScoreSpec::new("error", Direction::Lower);
impl Evaluation for Output {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        std::iter::once(Score::new(&SCORE, self.sum))
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        std::iter::once(MapView::new(
            &MAP,
            AnyPlane::F32(PlaneRef::new(&self.values[..], 2, 2, 2).unwrap()),
            MapGrid::new(2, 2).unwrap(),
        ))
    }
}
struct MetricFactory {
    preparations: Arc<AtomicUsize>,
    behavior: usize,
}
impl Metric for MetricFactory {
    type Output = Output;
    fn id(&self) -> MetricId {
        MetricId::new("control", Implementation::new("planar-control", "1", 1))
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Output> {
        fail(MetricError::new(ErrorKind::UnsupportedInput))
    }
}
impl PlanarMetric for MetricFactory {
    fn evaluate_planar(
        &mut self,
        r: &PlanarSlice<'_>,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Output> {
        self.warm_planar_reference(r, opts)?
            .evaluate_planar(d, opts)
    }
}
struct Prepared {
    reference: f64,
    behavior: usize,
}
impl PreparePlanarReference for MetricFactory {
    type PlanarPrepared = Prepared;
    fn warm_planar_reference(
        &mut self,
        r: &PlanarSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Prepared> {
        self.preparations.fetch_add(1, Ordering::SeqCst);
        Ok(Prepared {
            reference: sum(r),
            behavior: self.behavior,
        })
    }
}
fn sum(p: &PlanarSlice<'_>) -> f64 {
    let AnyPlane::F32(red) = p.planes()[0] else {
        panic!("f32")
    };
    red.rows().flatten().map(|v| f64::from(*v)).sum()
}
impl PlanarWarmReference for Prepared {
    type Output = Output;
    fn evaluate_planar(&mut self, p: &PlanarSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Output> {
        opts.checkpoint()?;
        let sum = sum(p) - self.reference;
        Ok(Output {
            sum,
            values: [sum as f32; 4],
        })
    }
    fn with_evaluation(
        &mut self,
        p: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        visitor: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        let output = self.evaluate_planar(p, opts)?;
        if self.behavior != 1 {
            visitor(EvaluationRef::new(&output));
        }
        if self.behavior == 2 {
            visitor(EvaluationRef::new(&output));
        }
        Ok(())
    }
}
#[test]
fn owned_preparation_survives_factory_drop_and_erasure_lends_views() {
    let l = layout();
    let c = color();
    let data = [1.0, 2.0, f32::NAN, 3.0, 4.0];
    let planes = planes(&data);
    let r = PlanarSlice::new(&planes, &l, 2, 2, &c, None).unwrap();
    let preparations = Arc::new(AtomicUsize::new(0));
    let mut m = Erased::new(Erased::new(MetricFactory {
        preparations: preparations.clone(),
        behavior: 0,
    }));
    let cold = score_planar(&mut m, &r, &r, EvalOpts::new().with_map("error")).unwrap();
    let mut cold_scores = Vec::new();
    cold.visit_scores(&mut |s| cold_scores.push(s.value.to_bits()));
    let mut prepared = prepare_planar_reference(&mut m, &r, EvalOpts::default()).unwrap();
    drop(m);
    for _ in 0..2 {
        let result = prepared
            .with_evaluation(&r, EvalOpts::new().with_map("error"), |view| {
                let mut scores = Vec::new();
                view.visit_scores(|s| scores.push(s.value.to_bits()));
                assert_eq!(scores, cold_scores);
                let mut map_count = 0;
                view.visit_maps(|m| {
                    map_count += 1;
                    assert_eq!(m.spec.key(), "error");
                    let AnyPlane::F32(p) = m.values else {
                        panic!("f32")
                    };
                    assert_eq!(p.row(0).unwrap(), [0.0, 0.0]);
                });
                assert_eq!(map_count, 1);
                Ok(scores)
            })
            .unwrap();
        assert_eq!(result, cold_scores);
    }
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    let other_color = ColorContext::from_cicp(Cicp::new(9, 8, 0, true));
    let changed = PlanarSlice::new(&planes, &l, 2, 2, &other_color, None).unwrap();
    assert_eq!(
        prepared
            .score(&changed, EvalOpts::new())
            .err()
            .unwrap()
            .error()
            .kind(),
        ErrorKind::UnsupportedInput
    );
    assert_eq!(
        prepared
            .score(&r, EvalOpts::new().with_max_pixels(3))
            .err()
            .unwrap()
            .error()
            .kind(),
        ErrorKind::Resource
    );
    assert_eq!(
        prepared
            .with_evaluation::<()>(&r, EvalOpts::new().with_map("missing"), |_| panic!(
                "missing map must refuse before consumer"
            ))
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::NoMap
    );
}
#[test]
fn lending_callback_count_and_original_consumer_errors_are_checked() {
    let l = layout();
    let c = color();
    let data = [0.0; 5];
    let planes = planes(&data);
    let p = PlanarSlice::new(&planes, &l, 2, 2, &c, None).unwrap();
    for behavior in [1, 2] {
        let mut m = MetricFactory {
            preparations: Arc::default(),
            behavior,
        };
        let mut warm = prepare_planar_reference(&mut m, &p, EvalOpts::new()).unwrap();
        assert_eq!(
            warm.with_evaluation(&p, EvalOpts::new(), |_| Ok(()))
                .unwrap_err()
                .error()
                .kind(),
            ErrorKind::Internal
        );
    }
    let mut m = MetricFactory {
        preparations: Arc::default(),
        behavior: 0,
    };
    let mut warm = prepare_planar_reference(&mut m, &p, EvalOpts::new()).unwrap();
    let e = warm
        .with_evaluation(&p, EvalOpts::new(), |_| {
            fail::<()>(MetricError::new(ErrorKind::Degenerate).with_message("consumer"))
        })
        .unwrap_err();
    assert_eq!(e.error().message(), Some("consumer"));
}
#[test]
fn borrowed_picture_checks_geometry_and_alpha_without_inspecting_padding() {
    let l = layout();
    let c = color();
    let data = [0.0; 5];
    let planes = planes(&data);
    assert!(PlanarSlice::new(&planes, &l, 3, 2, &c, None).is_err());
    assert!(PlanarSlice::new(&planes, &l, 2, 2, &c, Some(AlphaMode::Straight)).is_err());
    let p = PlanarSlice::new(&planes, &l, 2, 2, &c, None).unwrap();
    let AnyPlane::F32(r) = p.planes()[0] else {
        panic!("f32")
    };
    assert_eq!(r.strided_samples().as_ptr(), data.as_ptr());
}
