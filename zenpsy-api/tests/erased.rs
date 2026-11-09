use std::{cell::Cell, rc::Rc};
use zenpsy_api::{
    zenpixels::{PixelDescriptor, PixelSlice},
    *,
};

#[derive(Debug)]
struct NativeA {
    values: [f64; 2],
    // Local state and diagnostics survive without Send/Sync or array copies.
    local: Rc<u32>,
    map: Option<Box<[f32]>>,
}
impl Evaluation for NativeA {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        static SPECS: [ScoreSpec; 2] = [
            ScoreSpec::new("max", Direction::Lower),
            ScoreSpec::new("p3", Direction::Lower),
        ];
        SPECS.iter().zip(self.values).map(|(s, v)| Score::new(s, v))
    }
}
struct A<'a> {
    label: &'a str,
    cold: Rc<Cell<usize>>,
    preparations: Rc<Cell<usize>>,
    map_pointer: Rc<Cell<*const f32>>,
}
fn sum(r: &PixelSlice<'_>) -> u64 {
    (0..r.rows())
        .flat_map(|y| r.row(y))
        .map(|&v| u64::from(v))
        .sum()
}
impl A<'_> {
    fn result(
        &self,
        reference: u64,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<NativeA> {
        opts.checkpoint()?;
        let map = opts
            .diagnostics()
            .then(|| vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice());
        if let Some(map) = &map {
            self.map_pointer.set(map.as_ptr());
        }
        Ok(NativeA {
            values: [reference as f64, sum(d) as f64],
            local: Rc::new(17),
            map,
        })
    }
}
impl Metric for A<'_> {
    type Output = NativeA;
    fn id(&self) -> MetricId {
        MetricId::new("a", Implementation::new("control", "1", 1)).with_config("label", self.label)
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<NativeA> {
        self.cold.set(self.cold.get() + 1);
        self.result(sum(r), d, opts)
    }
}
struct PreparedA<'a, 'c> {
    scorer: &'a A<'c>,
    reference: u64,
}
impl<'c> PrepareReference for A<'c> {
    type Prepared<'a>
        = PreparedA<'a, 'c>
    where
        Self: 'a;
    fn warm_reference(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        opts.checkpoint()?;
        self.preparations.set(self.preparations.get() + 1);
        Ok(PreparedA {
            scorer: self,
            reference: sum(r),
        })
    }
}
impl WarmReference for PreparedA<'_, '_> {
    type Output = NativeA;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<NativeA> {
        self.scorer.result(self.reference, d, opts)
    }
}
struct B;
#[derive(Debug)]
struct NativeB(f64);
impl Evaluation for NativeB {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        static SPEC: ScoreSpec = ScoreSpec::new("quality", Direction::Higher);
        core::iter::once(Score::new(&SPEC, self.0))
    }
}
impl Metric for B {
    type Output = NativeB;
    fn id(&self) -> MetricId {
        MetricId::new("b", Implementation::new("control", "1", 1))
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<NativeB> {
        Ok(NativeB(3.5))
    }
}
fn view(rows: u32) -> PixelSlice<'static> {
    PixelSlice::new(
        &[1, 2, 3, 4, 5, 6, 255, 255, 255, 7, 8, 9, 10, 11, 12],
        2,
        rows,
        9,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}
fn a<'a>(label: &'a str) -> A<'a> {
    A {
        label,
        cold: Rc::new(Cell::new(0)),
        preparations: Rc::new(Cell::new(0)),
        map_pointer: Rc::new(Cell::new(core::ptr::null())),
    }
}

#[test]
fn heterogeneous_stack_runs_every_metric_and_keeps_original_native_data() {
    let label = String::from("borrowed config");
    let first = a(&label);
    let cold = first.cold.clone();
    let pointer = first.map_pointer.clone();
    let mut metrics: [Box<DynMetric<'_>>; 2] = [
        Box::new(Erased::new(Erased::new(Box::new(first)))),
        Box::new(Erased::new(B)),
    ];
    let mut families = Vec::new();
    for metric in &mut metrics {
        families.push(metric.id().family());
        let result = score(
            metric,
            &view(2),
            &view(2),
            EvalOpts::new().with_native_map(),
        )
        .unwrap();
        if metric.id().family() == "a" {
            let n = result.native::<NativeA>().unwrap();
            assert_eq!(*n.local, 17);
            assert_eq!(n.map.as_ref().unwrap().as_ptr(), pointer.get());
            assert_eq!(
                result
                    .scores()
                    .map(|s| (s.spec.key.as_ref(), s.value))
                    .collect::<Vec<_>>(),
                [("max", 78.0), ("p3", 78.0)]
            );
            assert!(
                result.native::<ErasedEvaluation>().is_none(),
                "repeated erasure must flatten"
            );
            assert!(result.native::<NativeB>().is_none());
        } else {
            assert_eq!(result.native::<NativeB>().unwrap().0, 3.5);
        }
    }
    assert_eq!(families, ["a", "b"]);
    assert_eq!(cold.get(), 1);
}
#[test]
fn prepared_object_retains_borrowed_scorer_geometry_and_fresh_controls() {
    let label = String::from("prepared borrowed config");
    let first = a(&label);
    let cold = first.cold.clone();
    let preparations = first.preparations.clone();
    let mut metrics: Vec<Box<dyn DynPrepareReference + '_>> =
        vec![Box::new(Erased::new(Erased::new(Box::new(first))))];
    for metric in &mut metrics {
        let mut warm = metric.prepare(&view(2), EvalOpts::new()).unwrap();
        let error = warm.evaluate(&view(1), EvalOpts::new()).err().unwrap();
        assert_eq!(error.error().kind(), ErrorKind::DimensionMismatch);
        let error = warm
            .evaluate(&view(2), EvalOpts::new().with_max_pixels(3))
            .err()
            .unwrap();
        assert_eq!(error.error().kind(), ErrorKind::Resource);
        let plain = warm.evaluate(&view(2), EvalOpts::new()).unwrap();
        assert!(plain.native::<NativeA>().unwrap().map.is_none());
        let diagnostic = warm
            .evaluate(&view(2), EvalOpts::new().with_native_map())
            .unwrap();
        assert!(diagnostic.native::<NativeA>().unwrap().map.is_some());
        assert_eq!(
            plain
                .scores()
                .map(|s| s.value.to_bits())
                .collect::<Vec<_>>(),
            diagnostic
                .scores()
                .map(|s| s.value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(cold.get(), 0);
    assert_eq!(preparations.get(), 1);
}
#[test]
fn checked_generic_code_accepts_erased_trait_objects_without_another_wrapper() {
    let mut metric: Box<DynMetric<'_>> = Box::new(Erased::new(B));
    let out = score(metric.as_mut(), &view(2), &view(2), EvalOpts::new()).unwrap();
    assert_eq!(out.native::<NativeB>().unwrap().0, 3.5);
}
