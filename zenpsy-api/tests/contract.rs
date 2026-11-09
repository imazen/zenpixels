use std::rc::Rc;
use zenpsy_api::{
    zenpixels::{PixelDescriptor, PixelSlice},
    *,
};

#[derive(Debug)]
struct Native {
    values: [f64; 2],
    keys: [ScoreSpec; 2],
    // Proves the core does not impose Send/Sync on native assessments.
    detail: Rc<u32>,
}
impl Evaluation for Native {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.keys
            .iter()
            .zip(self.values)
            .map(|(s, v)| Score::new(s, v))
    }
}
struct Scorer {
    values: [f64; 2],
    duplicate: bool,
}
impl Metric for Scorer {
    type Output = Native;
    fn id(&self) -> MetricId {
        MetricId::new("control", Implementation::new("test", "1", 1)).with_config("display", "hdr")
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Native> {
        opts.checkpoint()?;
        assert_eq!(r.descriptor(), d.descriptor());
        assert!(!opts.diagnostics());
        Ok(Native {
            values: self.values,
            keys: [
                ScoreSpec::new("max", Direction::Lower),
                ScoreSpec::new(
                    if self.duplicate { "max" } else { "pnorm" },
                    Direction::Lower,
                ),
            ],
            detail: Rc::new(17),
        })
    }
}
struct Prepared {
    values: [f64; 2],
}
impl PrepareReference for Scorer {
    type Prepared<'a> = Prepared;
    fn warm_reference(&mut self, _: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Prepared> {
        opts.checkpoint()?;
        Ok(Prepared {
            values: self.values,
        })
    }
}
impl WarmReference for Prepared {
    type Output = Native;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Native> {
        Scorer {
            values: self.values,
            duplicate: false,
        }
        .evaluate(d, d, opts)
    }
}
fn image(rows: u32) -> PixelSlice<'static> {
    PixelSlice::new(&[0; 12], 2, rows, 6, PixelDescriptor::RGB8_SRGB).unwrap()
}
#[test]
fn preserves_multiple_outputs_native_data_and_prepared_geometry() {
    let mut m = Scorer {
        values: [1.25, 0.375],
        duplicate: false,
    };
    let cold = score(&mut m, &image(2), &image(2), EvalOpts::new()).unwrap();
    assert_eq!(*cold.detail, 17);
    assert_eq!(
        cold.scores()
            .map(|s| (s.spec.key(), s.value))
            .collect::<Vec<_>>(),
        [("max", 1.25), ("pnorm", 0.375)]
    );
    let mut warm = prepare_reference(&mut m, &image(2), EvalOpts::new()).unwrap();
    assert_eq!(
        warm.score(&image(1), EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::DimensionMismatch
    );
    assert_eq!(
        warm.score(&image(2), EvalOpts::new())
            .unwrap()
            .values
            .map(f64::to_bits),
        cold.values.map(f64::to_bits)
    );
}
#[test]
fn refuses_duplicate_nonfinite_and_over_budget_inputs() {
    for (values, duplicate) in [
        ([1.0, 2.0], true),
        ([f64::NAN, 2.0], false),
        ([1.0, f64::INFINITY], false),
    ] {
        assert_eq!(
            score(
                &mut Scorer { values, duplicate },
                &image(2),
                &image(2),
                EvalOpts::new()
            )
            .unwrap_err()
            .error()
            .kind(),
            ErrorKind::Internal
        );
    }
    assert_eq!(
        score(
            &mut Scorer {
                values: [1.0, 2.0],
                duplicate: false
            },
            &image(2),
            &image(2),
            EvalOpts::new().with_max_pixels(3)
        )
        .unwrap_err()
        .error()
        .kind(),
        ErrorKind::Resource
    );
}
#[test]
fn identity_retains_config_and_actual_implementation_without_column_policy() {
    let a = Scorer {
        values: [1.0, 2.0],
        duplicate: false,
    }
    .id();
    assert_eq!(a.config_value("display"), Some("hdr"));
    assert_ne!(a.clone().with_backend(Backend::gpu("cuda")), a);
    assert_eq!(
        a.clone().with_config("x", 1).with_config("y", 2),
        a.with_config("y", 2).with_config("x", 1)
    );
}

#[test]
fn descriptors_support_owned_keys_and_infinity_without_identity_defaults() {
    let spec = ScoreSpec::new("placeholder", Direction::Lower)
        .with_key(String::from("runtime-key"))
        .with_positive_infinity(true);
    assert_eq!(spec.key(), "runtime-key");
    assert_eq!(spec.perfect(), None);
    assert!(spec.positive_infinity());
    assert!(!EvalOpts::default().diagnostics());
    assert_eq!(
        MapGrid::new(0, 1).unwrap_err().error().kind(),
        ErrorKind::InvalidConfig
    );
    assert_eq!(
        MapGrid::new(1, 1)
            .unwrap()
            .with_cells(0, 0, 0, 1)
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::InvalidConfig
    );
}
