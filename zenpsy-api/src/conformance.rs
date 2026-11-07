//! Reusable still-image implementation controls. Algorithm/reference parity
//! remains the implementing crate's responsibility.
use crate::{Image, Metric};
use alloc::{string::String, vec::Vec};
/// One violated declared contract.
#[derive(Debug)]
#[non_exhaustive]
pub struct Violation {
    /// Failed contract.
    pub check: &'static str,
    /// Observed failure.
    pub detail: String,
}
/// Validate identity, scalar declarations, maps, cancellation and optional real
/// warm-reference parity on a caller-supplied non-identical representative pair.
pub fn check(metric: &mut dyn Metric, r: &Image<'_>, d: &Image<'_>) -> Vec<Violation> {
    use crate::{ErrorKind, EvalOpts, MapReduce, MapRequest};
    use alloc::format;
    let mut violations = Vec::new();
    let mut fail = |check, detail| violations.push(Violation { check, detail });
    let columns = metric.columns();
    for (i, c) in columns.iter().enumerate() {
        if !crate::is_sanitized_column(c) || columns[..i].contains(c) {
            fail("identity", format!("invalid or duplicate column {c}"));
        }
    }
    let cold = match crate::score(metric, r.clone(), d.clone(), EvalOpts::new()) {
        Ok(e) => e,
        Err(e) => {
            fail("score", format!("{e}"));
            return violations;
        }
    };
    let bits: Vec<_> = cold
        .result
        .scores()
        .map(|(k, v)| (k, v.to_bits()))
        .collect();
    match crate::score(
        metric,
        r.clone(),
        d.clone(),
        EvalOpts::new().with_native_map(),
    ) {
        Ok(mapped) => {
            if mapped
                .result
                .scores()
                .map(|(k, v)| (k, v.to_bits()))
                .collect::<Vec<_>>()
                != bits
            {
                fail(
                    "map_score_identity",
                    String::from("map request changed scalars"),
                );
            }
            for m in mapped.result.maps() {
                if m.geometry.input_width != r.width()
                    || m.geometry.input_height != r.height()
                    || m.width() != m.geometry.cells_x
                    || m.height() != m.geometry.cells_y
                {
                    fail("map_geometry", format!("invalid geometry for {}", m.name));
                }
            }
            match crate::score(
                metric,
                r.clone(),
                d.clone(),
                EvalOpts::new().with_map(MapRequest::Subsampled {
                    factor: 2,
                    reduce: MapReduce::Mean,
                }),
            ) {
                Ok(small) => {
                    if small
                        .result
                        .scores()
                        .map(|(k, v)| (k, v.to_bits()))
                        .collect::<Vec<_>>()
                        != bits
                    {
                        fail(
                            "subsampled_score_identity",
                            String::from("reduced map changed scalars"),
                        );
                    }
                    let native = mapped.result.maps();
                    let served = small.result.maps();
                    if native.len() != served.len()
                        || native
                            .iter()
                            .zip(served)
                            .any(|(a, b)| b.geometry != a.geometry.subsampled(2))
                    {
                        fail(
                            "subsampled_geometry",
                            String::from("reduced geometry differs"),
                        );
                    }
                }
                Err(e) => fail("map_subsample", format!("{e}")),
            }
        }
        Err(e) if e.error().kind() == ErrorKind::NoMap => {}
        Err(e) => fail("map", format!("{e}")),
    }
    match crate::prepare_reference(metric, r.clone(), EvalOpts::new()) {
        Ok(mut warm) => match warm.score(d.clone(), EvalOpts::new()) {
            Ok(result)
                if result
                    .result
                    .scores()
                    .map(|(k, v)| (k, v.to_bits()))
                    .collect::<Vec<_>>()
                    == bits => {}
            Ok(_) => fail("warm", String::from("warm scalars differ from cold")),
            Err(e) => fail("warm", format!("{e}")),
        },
        Err(e) if e.error().kind() == ErrorKind::UnsupportedRequest => {}
        Err(e) => fail("prepare", format!("{e}")),
    }
    struct Stopped;
    impl how_far::Stop for Stopped {
        fn check(&self) -> Result<(), how_far::StopReason> {
            Err(how_far::StopReason::Cancelled)
        }
    }
    let pulse = how_far::StopOnly::new(Stopped);
    match crate::score(
        metric,
        r.clone(),
        d.clone(),
        EvalOpts::new().with_pulse(&pulse),
    ) {
        Err(e) if crate::stop_reason(&e) == Some(how_far::StopReason::Cancelled) => {}
        Err(e) => fail("cancellation", format!("{e}")),
        Ok(_) => fail("cancellation", String::from("stopped call succeeded")),
    }
    violations
}
/// Assert the caller-supplied native implementation obeys the shared contract.
pub fn assert_conforms(metric: &mut dyn Metric, r: &Image<'_>, d: &Image<'_>) {
    let errors = check(metric, r, d);
    assert!(errors.is_empty(), "{errors:#?}");
}
