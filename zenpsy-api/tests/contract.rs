use std::{borrow::Cow, sync::Arc};
use zenpsy_api::zenpixels::plane::{
    AnyPlane, PlanarLayout, PlaneEncoding, PlaneRef, PlaneRole, PlaneSpec,
};
use zenpsy_api::zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
use zenpsy_api::zenpixels::{ChannelType, Cicp, ColorContext, PixelDescriptor, PixelSlice};
use zenpsy_api::*;

struct Control {
    calls: usize,
    value: f64,
    nonfinite: NonFinite,
}
impl Metric for Control {
    fn id(&self) -> MetricId {
        MetricId::new("control", Implementation::new("contract", "1", 1))
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(vec![
            ScoreSpec::new("score", ScoreKind::Feature).with_nonfinite(self.nonfinite),
        ])
    }
    fn accepts(&self, _: &InputFormat) -> bool {
        true
    }
    fn evaluate(
        &mut self,
        _: Image<'_>,
        _: Image<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        self.calls += 1;
        Ok(Box::new(ScalarEval::single("score", self.value)))
    }
}
fn image() -> Image<'static> {
    PixelSlice::new(&[0; 18], 3, 2, 9, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}
#[test]
fn admission_precedes_native_work_and_nonfinite_policy_is_enforced() {
    let mut m = Control {
        calls: 0,
        value: f64::INFINITY,
        nonfinite: NonFinite::Refuse,
    };
    assert_eq!(
        score(&mut m, image(), image(), EvalOpts::new().with_max_pixels(5))
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::Resource
    );
    assert_eq!(m.calls, 0);
    assert_eq!(
        score(&mut m, image(), image(), EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::Internal
    );
    m.nonfinite = NonFinite::PositiveInfinity;
    assert_eq!(
        score(&mut m, image(), image(), EvalOpts::new())
            .unwrap()
            .result
            .score(),
        f64::INFINITY
    );
    m.value = f64::NAN;
    assert!(score(&mut m, image(), image(), EvalOpts::new()).is_err());
    assert!(prepare_reference(&mut m, image(), EvalOpts::new()).is_err());
}
#[test]
fn rgb_helper_preserves_minimal_strides_and_refuses_conflicting_context() {
    let bytes = [0u8; 21];
    let p = PixelSlice::new(&bytes, 3, 2, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    let image = Image::from(p.clone());
    let view = rgb8_view(&image).unwrap();
    assert_eq!(view.pixels.as_ptr(), bytes.as_ptr());
    assert_eq!(view.stride, 12);
    assert_eq!(contiguous_rgb8(&view).len(), 18);
    let p = p.with_color_context(Arc::new(ColorContext::from_cicp(Cicp::new(9, 16, 0, true))));
    assert_eq!(
        rgb8_view(&p.into()).unwrap_err().error().kind(),
        ErrorKind::UnsupportedInput
    );
}
#[test]
fn native_odd_subsampled_planes_use_shared_layout_without_copying() {
    let enc = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U16, 10, 6).unwrap());
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, enc),
            PlaneSpec::new(PlaneRole::Cb, enc),
            PlaneSpec::new(PlaneRole::Cr, enc),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    let y = [0u16; 17];
    let uv = [0u16; 7];
    let planes = [
        AnyPlane::U16(PlaneRef::new(&y[..], 5, 3, 6).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uv[..], 3, 2, 4).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uv[..], 3, 2, 4).unwrap()),
    ];
    let p = Planes::new(&layout, &planes, 5, 3, Cicp::new(9, 16, 9, false)).unwrap();
    assert_eq!(p.layout().planes()[0].encoding(), enc);
    match p.planes()[0] {
        AnyPlane::U16(p) => assert_eq!(p.strided_samples().as_ptr(), y.as_ptr()),
        _ => panic!("sample width changed"),
    }
    let image = Image::from(p);
    assert_eq!((image.width(), image.height()), (5, 3));
    assert!(Planes::new(&layout, &planes, 6, 3, Cicp::new(9, 16, 9, false)).is_err());
}
struct Nr;
impl NoReferenceMetric for Nr {
    fn id(&self) -> MetricId {
        MetricId::new("nr", Implementation::new("contract", "1", 1))
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(vec![ScoreSpec::new("score", ScoreKind::Quality)])
    }
    fn accepts(&self, f: &InputFormat) -> bool {
        is_rgb8_srgb(f)
    }
    fn evaluate_image(
        &mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        opts.refuse_map("nr")?;
        let v = rgb8_view(&r)?;
        Ok(Box::new(ScalarEval::single(
            "score",
            f64::from(v.pixels[0]),
        )))
    }
}
#[test]
fn paired_no_reference_retains_both_named_outputs_and_real_reference_work() {
    let mut m = PairedNoReference::new(Nr);
    let r = image();
    let dbytes = [7; 18];
    let d = Image::from(PixelSlice::new(&dbytes, 3, 2, 9, PixelDescriptor::RGB8_SRGB).unwrap());
    conformance::assert_conforms(&mut m, &r, &d);
    let mut warm = prepare_reference(&mut m, r, EvalOpts::new()).unwrap();
    let out = warm.score(d, EvalOpts::new()).unwrap();
    assert_eq!(
        out.result.scores().collect::<Vec<_>>(),
        vec![("score".into(), 7.0), ("score_ref".into(), 0.0)]
    );
}
