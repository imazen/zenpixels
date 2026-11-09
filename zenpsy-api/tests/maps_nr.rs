use std::{cell::Cell, rc::Rc};
use zenpsy_api::{
    zenpixels::{
        PixelDescriptor, PixelSlice,
        plane::{AnyPlane, PlaneBuf, PlaneRef},
    },
    *,
};

#[derive(Debug)]
struct Native {
    scores: [f64; 2],
    map: Option<PlaneBuf<f32>>,
    local: Rc<u32>,
    grid: MapGrid,
}
impl Evaluation for Native {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        static SPECS: [ScoreSpec; 2] = [
            ScoreSpec::new("sum", Direction::Unordered),
            ScoreSpec::new("max", Direction::Unordered),
        ];
        SPECS.iter().zip(self.scores).map(|(s, v)| Score::new(s, v))
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        static SPEC: ScoreSpec = ScoreSpec::new("native", Direction::Unordered);
        self.map
            .iter()
            .map(|p| MapView::new(&SPEC, AnyPlane::F32(p.view()), self.grid))
    }
}
struct ImageMetric {
    calls: Rc<Cell<usize>>,
    maps: bool,
    malformed: bool,
}
impl NoReferenceMetric for ImageMetric {
    type Output = Native;
    fn id(&self) -> MetricId {
        MetricId::new("image", Implementation::new("native-control", "1", 1))
    }
    fn evaluate_image(&mut self, p: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Native> {
        if opts.diagnostics() && !self.maps {
            return fail(MetricError::new(ErrorKind::UnsupportedRequest));
        }
        self.calls.set(self.calls.get() + 1);
        let values = (0..p.rows()).flat_map(|y| p.row(y).iter().copied());
        let sum = values.clone().map(u64::from).sum::<u64>();
        let max = values.max().unwrap();
        let map = opts.diagnostics().then(|| {
            // A cropped/coarse native grid with row padding and a payload NaN.
            PlaneBuf::new(
                vec![1.0, f32::from_bits(0x7fc0_1234), -999.0, 3.0, 4.0],
                2,
                2,
                3,
            )
            .unwrap()
        });
        let grid = MapGrid::new(p.width() + u32::from(self.malformed), p.rows())?
            .with_cells(-1, 1, 2, 2)?;
        Ok(Native {
            scores: [sum as f64, f64::from(max)],
            map,
            grid,
            local: Rc::new(17),
        })
    }
}
fn metric(maps: bool) -> ImageMetric {
    ImageMetric {
        calls: Rc::new(Cell::new(0)),
        maps,
        malformed: false,
    }
}
fn image(bytes: &[u8]) -> PixelSlice<'_> {
    PixelSlice::new(bytes, 2, 2, 9, PixelDescriptor::RGB8_SRGB).unwrap()
}
fn bits(result: &impl Evaluation) -> Vec<(String, u64)> {
    result
        .scores()
        .map(|s| (s.spec.key().to_string(), s.value.to_bits()))
        .collect()
}
#[test]
fn native_maps_borrow_the_original_strided_plane_through_stacked_erasure() {
    let input = [1, 2, 3, 4, 5, 6, 255, 255, 255, 7, 8, 9, 10, 11, 12];
    let mut m = metric(true);
    let cold = score_image(&mut m, &image(&input), EvalOpts::new()).unwrap();
    assert!(cold.maps().next().is_none());
    assert_eq!(cold.scores, [78.0, 12.0]); // Padding was not scored.
    let mut erased: Box<DynNoReferenceMetric<'_>> = Box::new(Erased::new(Erased::new(m)));
    let out = score_image(
        &mut erased,
        &image(&input),
        EvalOpts::new().with_native_map(),
    )
    .unwrap();
    assert_eq!(bits(&out), bits(&cold));
    let native = out.native::<Native>().unwrap();
    assert_eq!(*native.local, 17);
    let map = out.maps().next().unwrap();
    let AnyPlane::F32(plane) = map.values else {
        panic!("native f32 samples")
    };
    assert_eq!(
        plane.strided_samples().as_ptr(),
        native.map.as_ref().unwrap().strided_samples().as_ptr()
    );
    assert_eq!((plane.width(), plane.height(), plane.stride()), (2, 2, 3));
    assert_eq!(plane.row(1).unwrap(), [3.0, 4.0]);
    assert_eq!(plane.row(0).unwrap()[1].to_bits(), 0x7fc0_1234);
    assert_eq!(
        map.grid,
        MapGrid::new(2, 2).unwrap().with_cells(-1, 1, 2, 2).unwrap()
    );
    assert!(out.native::<ErasedEvaluation>().is_none());
}
#[test]
fn pairing_preserves_both_native_results_and_reuses_reference_work() {
    let r = [1; 15];
    let d = [2; 15];
    let m = metric(true);
    let calls = m.calls.clone();
    let mut pair = PairedNoReference::new(m);
    let cold = score(&mut pair, &image(&r), &image(&d), EvalOpts::new()).unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(
        bits(&cold)
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>(),
        ["sum", "max", "sum_ref", "max_ref"]
    );
    assert_eq!(cold.reference().scores, [12.0, 1.0]);
    assert_eq!(cold.candidate().scores, [24.0, 2.0]);
    let mut erased = Erased::new(Erased::new(pair));
    let mut warm = erased.prepare(&image(&r), EvalOpts::new()).unwrap();
    assert_eq!(calls.get(), 3);
    let a = warm
        .evaluate(&image(&d), EvalOpts::new().with_native_map())
        .unwrap();
    let b = warm.evaluate(&image(&d), EvalOpts::new()).unwrap();
    assert_eq!(calls.get(), 5);
    assert_eq!(bits(&a), bits(&cold));
    assert_eq!(bits(&b), bits(&cold));
    let a_native = a.native::<PairedEvaluation<Native>>().unwrap();
    let b_native = b.native::<PairedEvaluation<Native>>().unwrap();
    assert!(core::ptr::eq(a_native.reference(), b_native.reference()));
    assert!(a_native.reference().maps().next().is_none());
    assert_eq!(a.maps().count(), 1);
    assert_eq!(b.maps().count(), 0);
    let AnyPlane::F32(plane) = a.maps().next().unwrap().values else {
        panic!("native f32 samples")
    };
    assert_eq!(
        plane.strided_samples().as_ptr(),
        a_native
            .candidate()
            .map
            .as_ref()
            .unwrap()
            .strided_samples()
            .as_ptr()
    );
}
#[test]
fn unsupported_maps_bad_grids_and_admission_fail_explicitly() {
    let input = [1; 15];
    let p = image(&input);
    let mut m = metric(false);
    assert_eq!(
        score_image(&mut m, &p, EvalOpts::new().with_native_map())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::UnsupportedRequest
    );
    assert_eq!(
        score_image(&mut m, &p, EvalOpts::new().with_max_pixels(3))
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::Resource
    );
    assert_eq!(m.calls.get(), 0);
    let mut malformed = metric(true);
    malformed.malformed = true;
    assert_eq!(
        score_image(&mut malformed, &p, EvalOpts::new().with_native_map())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::Internal
    );
    assert!(MapGrid::new(0, 2).is_err());
    assert!(MapGrid::new(2, 2).unwrap().with_cells(0, 0, 0, 1).is_err());
    // Borrowed plane construction independently rejects short strides/storage.
    assert!(PlaneRef::new(&[1.0f32; 4][..], 2, 2, 1).is_err());
}
