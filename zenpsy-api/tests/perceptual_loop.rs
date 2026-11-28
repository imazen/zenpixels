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

struct SimilarityOutput {
    score: f64,
    aux: f64,
    map_u16: [u16; 4],
}

static HEADLINE: ScoreSpec = ScoreSpec::new("similarity", Direction::Higher).with_perfect(100.0);
static AUX: ScoreSpec = ScoreSpec::new("aux", Direction::Unordered);
static MAP_HIGHER: ScoreSpec = ScoreSpec::new("local_sim", Direction::Higher).with_perfect(100.0);

impl Evaluation for SimilarityOutput {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        [
            Score::new(&HEADLINE, self.score),
            Score::new(&AUX, self.aux),
        ]
        .into_iter()
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        core::iter::once(MapView::new(
            &MAP_HIGHER,
            AnyPlane::U16(PlaneRef::new(&self.map_u16[..], 2, 2, 2).unwrap()),
            MapGrid::new(4, 4).unwrap().with_cells(0, 0, 2, 2).unwrap(),
        ))
    }
}

struct LoopMetric;
struct LoopPrepared;

impl Metric for LoopMetric {
    type Output = SimilarityOutput;
    fn id(&self) -> MetricId {
        MetricId::new("sim-loop", Implementation::new("loop-test", "0.1.0", 1))
    }
    fn headline_score_key(&self) -> Option<&str> {
        Some("similarity")
    }
    fn steering_map_key(&self) -> Option<&str> {
        Some("local_sim")
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        Ok(SimilarityOutput {
            score: 88.5,
            aux: 3.0,
            map_u16: [100, 90, 80, 75],
        })
    }
}

impl PrepareReference for LoopMetric {
    type Prepared<'a> = LoopPrepared;
    fn estimated_reference_bytes(&self, width: u32, height: u32) -> Option<u64> {
        Some(u64::from(width) * u64::from(height) * 16)
    }
    fn warm_reference(
        &mut self,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        Ok(LoopPrepared)
    }
}

impl WarmReference for LoopPrepared {
    type Output = SimilarityOutput;
    fn is_deterministic(&self) -> bool {
        true
    }
    fn evaluate(&mut self, _: &PixelSlice<'_>, _: EvalOpts<'_>) -> MetricResult<Self::Output> {
        Ok(SimilarityOutput {
            score: 88.5,
            aux: 3.0,
            map_u16: [100, 90, 80, 75],
        })
    }
}

impl PlanarMetric for LoopMetric {
    fn evaluate_planar(
        &mut self,
        r: &PlanarSlice<'_>,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.warm_planar_reference(r, opts)?
            .evaluate_planar(d, opts)
    }
}

impl PreparePlanarReference for LoopMetric {
    type PlanarPrepared = LoopPrepared;
    fn estimated_reference_bytes(&self, width: u32, height: u32) -> Option<u64> {
        Some(u64::from(width) * u64::from(height) * 12)
    }
    fn warm_planar_reference(
        &mut self,
        _: &PlanarSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Self::PlanarPrepared> {
        Ok(LoopPrepared)
    }
}

impl PlanarWarmReference for LoopPrepared {
    type Output = SimilarityOutput;
    fn is_deterministic(&self) -> bool {
        true
    }
    fn evaluate_planar(
        &mut self,
        _: &PlanarSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        Ok(SimilarityOutput {
            score: 88.5,
            aux: 3.0,
            map_u16: [100, 90, 80, 75],
        })
    }
}

#[test]
fn score_loss_handles_both_directions_and_refuses_uncalibrated() {
    let lower = ScoreSpec::new("d", Direction::Lower).with_perfect(0.0);
    assert_eq!(lower.loss(1.25), Some(1.25));
    assert_eq!(lower.loss(-0.2), Some(0.0));
    assert_eq!(lower.loss(f64::NAN), None);
    assert_eq!(lower.loss(f64::INFINITY), None);

    let higher = ScoreSpec::new("z", Direction::Higher).with_perfect(100.0);
    assert_eq!(Score::new(&higher, 87.5).loss(), Some(12.5));
    assert_eq!(Score::new(&higher, 100.2).loss(), Some(0.0));

    let psnr = ScoreSpec::new("psnr", Direction::Higher).with_perfect(f64::INFINITY);
    assert_eq!(psnr.loss(40.0), None);

    let no_perfect = ScoreSpec::new("raw", Direction::Lower);
    assert_eq!(no_perfect.loss(1.0), None);

    let unordered = ScoreSpec::new("diag", Direction::Unordered).with_perfect(0.0);
    assert_eq!(unordered.loss(0.5), None);
}

#[test]
fn loop_metadata_and_accessors_forward_through_erased_and_box() {
    let mut erased = Erased::new(Erased::new(LoopMetric));
    assert_eq!(erased.headline_score_key(), Some("similarity"));
    assert_eq!(erased.steering_map_key(), Some("local_sim"));
    assert_eq!(
        PrepareReference::estimated_reference_bytes(&erased, 10, 20),
        Some(3200)
    );
    assert_eq!(
        PreparePlanarReference::estimated_reference_bytes(&erased, 10, 20),
        Some(2400)
    );

    let boxed: Box<dyn DynPreparePlanarReference> = Box::new(Erased::new(LoopMetric));
    assert_eq!(boxed.headline_score_key(), Some("similarity"));
    assert_eq!(boxed.steering_map_key(), Some("local_sim"));
    assert_eq!(boxed.estimated_reference_bytes(10, 20), Some(2400));

    let l = layout();
    let c = color();
    let data = [0.25f32; 16];
    let p = [AnyPlane::F32(PlaneRef::new(&data, 4, 4, 4).unwrap()); 3];
    let slice = PlanarSlice::new(&p, &l, 4, 4, &c, None).unwrap();

    let mut prepared = prepare_planar_reference(&mut erased, &slice, EvalOpts::new()).unwrap();
    assert_eq!(prepared.size(), (4, 4));
    assert_eq!(prepared.layout(), &l);
    assert_eq!(prepared.color_context(), &c);
    assert_eq!(prepared.alpha_mode(), None);
    assert!(prepared.is_deterministic());

    let map_key = erased.steering_map_key().unwrap();
    let score_key = erased.headline_score_key().unwrap();
    let (loss, cell_losses) = prepared
        .with_evaluation(&slice, EvalOpts::new().with_map(map_key), |view| {
            let s = view.score(score_key).unwrap();
            assert_eq!(view.primary_score().unwrap().spec.key(), score_key);
            assert!(view.score("missing").is_none());
            let m = view.map(map_key).unwrap();
            assert_eq!(view.primary_map().unwrap().spec.key(), map_key);
            assert!(!m.grid.is_pixel_aligned(m.values.width(), m.values.height()));
            assert_eq!(m.grid.cell_bounds(0, 0), (0, 0, 2, 2));
            assert_eq!(m.grid.cell_bounds(1, 1), (2, 2, 4, 4));
            assert!(m.as_f32().is_none());
            let losses = [
                m.loss_at(0, 0).unwrap(),
                m.loss_at(1, 0).unwrap(),
                m.loss_at(0, 1).unwrap(),
                m.loss_at(1, 1).unwrap(),
            ];
            assert!(m.sample_f64(2, 0).is_none());
            Ok((s.loss().unwrap(), losses))
        })
        .unwrap();
    assert_eq!(loss, 11.5);
    assert_eq!(cell_losses, [0.0, 10.0, 20.0, 25.0]);
}

#[test]
fn packed_prepared_accessors_and_pixel_aligned_grid() {
    let bytes = [0u8; 4 * 4 * 3];
    let img = PixelSlice::new(&bytes, 4, 4, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut erased = Erased::new(LoopMetric);
    let prepared = prepare_reference(&mut erased, &img, EvalOpts::new()).unwrap();
    assert_eq!(prepared.size(), (4, 4));
    assert!(prepared.is_deterministic());

    let grid = MapGrid::new(8, 6).unwrap();
    assert!(grid.is_pixel_aligned(8, 6));
    assert!(!grid.is_pixel_aligned(4, 6));
    assert_eq!(grid.cell_bounds(3, 2), (3, 2, 4, 3));
}
