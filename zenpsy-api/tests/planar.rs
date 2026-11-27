use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zenpsy_api::{zenpixels::*, *};

fn rgb_planes(data: &[f32], stride_samples: usize) -> [PlaneView<'_>; 3] {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let stride = stride_samples * 4;
    [
        PlaneView::new(
            bytes,
            2,
            2,
            stride,
            PlaneFormat::new(PlaneRole::R, SampleType::F32),
        )
        .unwrap(),
        PlaneView::new(
            bytes,
            2,
            2,
            stride,
            PlaneFormat::new(PlaneRole::G, SampleType::F32),
        )
        .unwrap(),
        PlaneView::new(
            bytes,
            2,
            2,
            stride,
            PlaneFormat::new(PlaneRole::B, SampleType::F32),
        )
        .unwrap(),
    ]
}

fn color() -> ColorSpace {
    ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    }
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
            PlaneView::new(
                bytemuck::cast_slice(&self.values),
                2,
                2,
                8,
                PlaneFormat::new(PlaneRole::Y, SampleType::F32),
            )
            .unwrap(),
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
        _: &PixelView<'_>,
        _: &PixelView<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Output> {
        fail(MetricError::new(ErrorKind::UnsupportedInput))
    }
}
impl PlanarMetric for MetricFactory {
    fn evaluate_planar(
        &mut self,
        r: &PlanarView<'_>,
        d: &PlanarView<'_>,
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
        r: &PlanarView<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Prepared> {
        self.preparations.fetch_add(1, Ordering::SeqCst);
        Ok(Prepared {
            reference: sum(r),
            behavior: self.behavior,
        })
    }
}
fn sum(p: &PlanarView<'_>) -> f64 {
    let red = p.planes()[0];
    assert_eq!(red.format().sample, SampleType::F32);
    (0..red.height())
        .flat_map(|y| bytemuck::cast_slice::<u8, f32>(red.row(y)).iter().copied())
        .map(f64::from)
        .sum()
}
impl PlanarWarmReference for Prepared {
    type Output = Output;
    fn evaluate_planar(&mut self, p: &PlanarView<'_>, opts: EvalOpts<'_>) -> MetricResult<Output> {
        opts.checkpoint()?;
        let sum = sum(p) - self.reference;
        Ok(Output {
            sum,
            values: [sum as f32; 4],
        })
    }
    fn with_evaluation(
        &mut self,
        p: &PlanarView<'_>,
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
    let c = color();
    let data = [1.0, 2.0, f32::NAN, 3.0, 4.0];
    let planes = rgb_planes(&data, 3);
    let r = PlanarView::new(2, 2, &planes).unwrap().with_color(c);
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
                    assert_eq!(m.values.format().sample, SampleType::F32);
                    assert_eq!(bytemuck::cast_slice::<u8, f32>(m.values.row(0)), [0.0, 0.0]);
                });
                assert_eq!(map_count, 1);
                Ok(scores)
            })
            .unwrap();
        assert_eq!(result, cold_scores);
    }
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    let other_color = ColorSpace::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    let changed = PlanarView::new(2, 2, &planes)
        .unwrap()
        .with_color(other_color);
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
    let c = color();
    let data = [0.0; 5];
    let planes = rgb_planes(&data, 3);
    let p = PlanarView::new(2, 2, &planes).unwrap().with_color(c);
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
fn borrowed_picture_checks_geometry_without_inspecting_padding() {
    let c = color();
    let data = [0.0; 5];
    let planes = rgb_planes(&data, 3);
    assert!(PlanarView::new(3, 2, &planes).is_err());
    let p = PlanarView::new(2, 2, &planes).unwrap().with_color(c);
    let r = p.planes()[0];
    assert_eq!(
        r.data().as_ptr(),
        bytemuck::cast_slice::<f32, u8>(&data).as_ptr()
    );
}

#[test]
fn heterogeneous_planar_objects_retain_checked_lending() {
    let c = color();
    let data = [0.0; 5];
    let planes = rgb_planes(&data, 3);
    let p = PlanarView::new(2, 2, &planes).unwrap().with_color(c);
    let mut metrics: Vec<Box<dyn DynPreparePlanarReference>> = vec![
        Box::new(Erased::new(MetricFactory {
            preparations: Arc::default(),
            behavior: 0,
        })),
        Box::new(Erased::new(Erased::new(MetricFactory {
            preparations: Arc::default(),
            behavior: 0,
        }))),
    ];
    for metric in &mut metrics {
        let mut prepared = metric.prepare_planar(&p, EvalOpts::new()).unwrap();
        let mut n = 0;
        prepared
            .with_evaluation(&p, EvalOpts::new().with_map("error"), &mut |view| {
                view.visit_maps(|m| {
                    n += 1;
                    assert_eq!(m.spec.key(), "error");
                })
            })
            .unwrap();
        assert_eq!(n, 1);
        assert!(
            prepared
                .evaluate_planar(&p, EvalOpts::new())
                .unwrap()
                .native::<Output>()
                .is_some()
        );
    }
}

#[test]
fn prepared_interpretation_retains_authority_and_rejects_changes() {
    let icc: Arc<[u8]> = Arc::from([1, 2, 3, 4]);
    let c = ColorSpace::Icc {
        profile: icc.clone(),
        color_channels: 3,
        alpha: AlphaMode::None,
        intent: RenderingIntent::RelativeColorimetric,
    };
    let data = [1.0, 2.0, f32::NAN, 3.0, 4.0];
    let p = rgb_planes(&data, 3);
    let image = PlanarView::new(2, 2, &p)
        .unwrap()
        .with_color(c.clone())
        .with_matrix_coefficients(MatrixCoefficients::BT709)
        .with_range_override(SignalRange::Narrow);
    let mut metric = Erased::new(Erased::new(MetricFactory {
        preparations: Arc::default(),
        behavior: 0,
    }));
    let mut prepared = prepare_planar_reference(&mut metric, &image, EvalOpts::new()).unwrap();
    let independently_padded = [1.0, 2.0, f32::NAN, f32::NAN, 3.0, 4.0];
    let q = rgb_planes(&independently_padded, 4);
    let candidate = PlanarView::new(2, 2, &q)
        .unwrap()
        .with_color(c.clone())
        .with_matrix_coefficients(MatrixCoefficients::BT709)
        .with_range_override(SignalRange::Narrow);
    let result = prepared.score(&candidate, EvalOpts::new()).unwrap();
    assert_eq!(
        result.scores().next().unwrap().value.to_bits(),
        0.0f64.to_bits()
    );
    let changed_candidates = [
        PlanarView::new(2, 2, &q)
            .unwrap()
            .with_color(ColorSpace::Icc {
                profile: Arc::from([4, 3, 2, 1]),
                color_channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::RelativeColorimetric,
            })
            .with_matrix_coefficients(MatrixCoefficients::BT709)
            .with_range_override(SignalRange::Narrow),
        PlanarView::new(2, 2, &q)
            .unwrap()
            .with_color(c.clone())
            .with_matrix_coefficients(MatrixCoefficients::BT2020_NCL)
            .with_range_override(SignalRange::Narrow),
        PlanarView::new(2, 2, &q)
            .unwrap()
            .with_color(c.clone())
            .with_matrix_coefficients(MatrixCoefficients::BT709)
            .with_range_override(SignalRange::Full),
        PlanarView::new(2, 2, &q)
            .unwrap()
            .with_color(c)
            .with_matrix_coefficients(MatrixCoefficients::BT709)
            .with_range_override(SignalRange::Narrow)
            .with_chroma_siting(ChromaSiting::Left),
    ];
    for candidate in &changed_candidates {
        assert_eq!(
            prepared
                .score(candidate, EvalOpts::new())
                .err()
                .unwrap()
                .error()
                .kind(),
            ErrorKind::UnsupportedInput
        );
        assert_eq!(
            prepared
                .with_evaluation::<()>(candidate, EvalOpts::new(), |_| panic!(
                    "changed interpretation reached consumer"
                ))
                .unwrap_err()
                .error()
                .kind(),
            ErrorKind::UnsupportedInput
        );
    }
    prepared
        .with_evaluation(&image, EvalOpts::new(), |_| Ok(()))
        .unwrap();
}

#[test]
fn packed_and_planar_operation_admission_share_request_classification() {
    let c = color();
    let data = [0.0; 5];
    let p = rgb_planes(&data, 3);
    let planar = PlanarView::new(2, 2, &p).unwrap().with_color(c);
    let bytes = [0; 12];
    let packed = PixelView::new(&bytes, 2, 2, 6, &formats::RGB8_SRGB).unwrap();
    let mut metric = MetricFactory {
        preparations: Arc::default(),
        behavior: 0,
    };
    for (opts, expected) in [
        (EvalOpts::new().with_map(""), ErrorKind::InvalidConfig),
        (EvalOpts::new().with_max_pixels(3), ErrorKind::Resource),
    ] {
        assert_eq!(
            score(&mut metric, &packed, &packed, opts)
                .err()
                .unwrap()
                .error()
                .kind(),
            expected
        );
        assert_eq!(
            score_planar(&mut metric, &planar, &planar, opts)
                .err()
                .unwrap()
                .error()
                .kind(),
            expected
        );
        assert_eq!(
            prepare_planar_reference(&mut metric, &planar, opts)
                .err()
                .unwrap()
                .error()
                .kind(),
            expected
        );
    }
    assert_eq!(metric.preparations.load(Ordering::SeqCst), 0);
}
