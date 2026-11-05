//! Cross-crate controls for Pulse ownership and located error propagation.
use how_far::{
    Child, ChildPulse, Complete, Execution, Outcome, PhaseSpec, PlanError, Report, Stages, Stop,
    Total,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use zenpixels_iqa::how_far::{Pulse, SharedPulse, StopReason};
use zenpixels_iqa::whereat::{At, ErrorAtExt};
use zenpixels_iqa::*;
type Error = MetricError;
type Result<T> = MetricResult<T>;
use zenpixels::{PixelDescriptor, PixelSlice};

#[derive(Default)]
struct Log {
    outcomes: Mutex<BTreeMap<String, Outcome>>,
    reports: Mutex<BTreeMap<String, u64>>,
    totals: Mutex<BTreeMap<String, Total>>,
    issues: Mutex<Vec<PlanError>>,
    stopped: AtomicBool,
    reject_plan: AtomicBool,
}
struct Recorder {
    path: String,
    log: Arc<Log>,
    owns_completion: bool,
    finished: bool,
}
impl Recorder {
    fn root(log: &Arc<Log>) -> Self {
        Self {
            path: "caller".into(),
            log: log.clone(),
            owns_completion: false,
            finished: false,
        }
    }
}
impl Stop for Recorder {
    fn check(&self) -> core::result::Result<(), StopReason> {
        if self.log.stopped.load(Ordering::Acquire) {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}
impl Report for Recorder {
    fn advance(&self, completed: u64) {
        *self
            .log
            .reports
            .lock()
            .unwrap()
            .entry(self.path.clone())
            .or_default() += completed;
    }
}
impl Pulse for Recorder {
    fn record_issue(&self, issue: PlanError) {
        self.log.issues.lock().unwrap().push(issue);
    }
    fn split(
        &self,
        _: Execution,
        parts: &[PhaseSpec<'_>],
    ) -> core::result::Result<Vec<Child<'_>>, PlanError> {
        PhaseSpec::validate_split(parts)?;
        if self.log.reject_plan.load(Ordering::Relaxed) {
            return Err(PlanError::Unsupported);
        }
        Ok(parts
            .iter()
            .map(|part| {
                Child::new(Self {
                    path: format!("{}/{}", self.path, part.name),
                    log: self.log.clone(),
                    owns_completion: true,
                    finished: false,
                })
            })
            .collect())
    }
    fn set_total(&self, total: Total) -> core::result::Result<(), PlanError> {
        self.log
            .totals
            .lock()
            .unwrap()
            .insert(self.path.clone(), total);
        Ok(())
    }
    fn share(&self) -> core::result::Result<SharedPulse, PlanError> {
        Ok(SharedPulse::new(Self {
            path: self.path.clone(),
            log: self.log.clone(),
            owns_completion: false,
            finished: false,
        }))
    }
}
impl ChildPulse for Recorder {
    fn finish(mut self: Box<Self>, outcome: Outcome) -> core::result::Result<(), PlanError> {
        assert!(
            self.owns_completion,
            "a shared view cannot acquire completion rights"
        );
        assert!(
            self.log
                .outcomes
                .lock()
                .unwrap()
                .insert(self.path.clone(), outcome)
                .is_none(),
            "complete a child exactly once"
        );
        self.finished = true;
        Ok(())
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        if self.owns_completion && !self.finished {
            self.log
                .outcomes
                .lock()
                .unwrap()
                .insert(self.path.clone(), Outcome::Abandoned);
        }
    }
}

fn image() -> Image<'static> {
    PixelSlice::new(&[0; 12], 2, 2, 6, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}
struct Worker {
    stop_at_return: bool,
    error: Option<At<Error>>,
    log: Arc<Log>,
}
impl Metric for Worker {
    fn id(&self) -> MetricId {
        MetricId::new("control", Implementation::new("protocol", "1", 1))
    }
    fn outputs(&self) -> std::borrow::Cow<'static, [ScoreSpec]> {
        std::borrow::Cow::Owned(vec![ScoreSpec::new("score", ScoreKind::Difference)])
    }
    fn accepts(&self, f: &InputFormat) -> bool {
        is_rgb8_srgb(f)
    }
    fn evaluate(
        &mut self,
        _: Image<'_>,
        _: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> Result<Box<dyn Evaluation>> {
        opts.pulse.set_total(Total::Exact(2)).unwrap();
        let shared = opts.pulse.share().unwrap();
        // An owned worker gets the complete view, including progress and stop.
        std::thread::spawn(move || EvalOpts::new().with_pulse(shared.as_pulse()).step(2))
            .join()
            .unwrap()?;
        if self.stop_at_return {
            self.log.stopped.store(true, Ordering::Release);
        }
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        Ok(Box::new(ScalarEval::single("score", 1.25)))
    }
}
fn worker(log: &Arc<Log>) -> Worker {
    Worker {
        stop_at_return: false,
        error: None,
        log: log.clone(),
    }
}

#[test]
fn owned_worker_reporting_and_parent_result_handoffs_compose() {
    let log = Arc::new(Log::default());
    let pulse = Recorder::root(&log);
    let mut metric = worker(&log);
    let mut caller = Stages::new(&pulse, &[PhaseSpec::new("compare", 1, Total::Unknown)]);
    let result = caller.run_classified(
        |e| stop_reason(e).is_some(),
        |pulse| {
            score(
                &mut metric,
                image(),
                image(),
                EvalOpts::new().with_pulse(pulse),
            )
        },
    );
    assert_eq!(
        caller
            .complete_classified(result, |e| stop_reason(e).is_some())
            .unwrap()
            .result
            .score(),
        1.25
    );
    assert_eq!(
        *log.outcomes.lock().unwrap(),
        BTreeMap::from([
            ("caller/compare".into(), Outcome::Succeeded),
            ("caller/compare/score image pair".into(), Outcome::Succeeded),
        ])
    );
    assert_eq!(
        log.reports.lock().unwrap()["caller/compare/score image pair"],
        2
    );
    assert_eq!(
        log.totals.lock().unwrap()["caller/compare/score image pair"],
        Total::Exact(2)
    );
    assert!(log.issues.lock().unwrap().is_empty());
}

#[test]
fn cancellation_and_failure_complete_owned_children_without_finishing_parent() {
    for stopped in [false, true] {
        let log = Arc::new(Log::default());
        let pulse = Recorder::root(&log);
        let mut metric = worker(&log);
        if stopped {
            metric.stop_at_return = true;
        } else {
            metric.error = Some(Error::new(ErrorKind::Backend).start_at());
        }
        let error = score(
            &mut metric,
            image(),
            image(),
            EvalOpts::new().with_pulse(&pulse),
        )
        .unwrap_err();
        assert_eq!(stop_reason(&error), stopped.then_some(StopReason::TimedOut));
        assert_eq!(
            *log.outcomes.lock().unwrap(),
            BTreeMap::from([(
                "caller/score image pair".into(),
                if stopped {
                    Outcome::Cancelled
                } else {
                    Outcome::Failed
                }
            ),])
        );
        assert!(!error.is_empty());
    }
}

#[test]
fn rejected_observation_plan_retains_cancellation_without_inventing_progress() {
    let log = Arc::new(Log::default());
    log.reject_plan.store(true, Ordering::Relaxed);
    let pulse = Recorder::root(&log);
    let mut metric = worker(&log);
    assert_eq!(
        score(
            &mut metric,
            image(),
            image(),
            EvalOpts::new().with_pulse(&pulse)
        )
        .unwrap()
        .result
        .score(),
        1.25
    );
    // Rejected tracking follows how-far's cancellation-only fallback. Reports
    // must not be credited to a parent whose plan could not be applied.
    assert!(log.reports.lock().unwrap().is_empty());
    assert_eq!(*log.issues.lock().unwrap(), vec![PlanError::Unsupported]);
    metric.stop_at_return = true;
    assert_eq!(
        stop_reason(
            &score(
                &mut metric,
                image(),
                image(),
                EvalOpts::new().with_pulse(&pulse)
            )
            .unwrap_err()
        ),
        Some(StopReason::TimedOut)
    );
    assert!(log.outcomes.lock().unwrap().is_empty());
}

#[derive(Debug)]
struct Native(Box<u32>);
impl std::fmt::Display for Native {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native sentinel")
    }
}
impl std::error::Error for Native {}

#[test]
fn native_trace_cause_and_error_priority_survive_the_envelope_and_scoring_boundary() {
    let native = Native(Box::new(17));
    let allocation = &*native.0 as *const u32;
    let origin = line!() + 1;
    let native = native.start_at().at_str("native context");
    let error = Error::from_located(ErrorKind::Backend, native);
    let log = Arc::new(Log::default());
    let pulse = Recorder::root(&log);
    let mut metric = worker(&log);
    metric.error = Some(error);
    metric.stop_at_return = true;
    let error = score(
        &mut metric,
        image(),
        image(),
        EvalOpts::new().with_pulse(&pulse),
    )
    .unwrap_err();
    assert_eq!(
        error.error().kind(),
        ErrorKind::Backend,
        "the actual native failure precedes the later stop"
    );
    assert_eq!(stop_reason(&error), None);
    let first = error.frames().find_map(|frame| frame.location()).unwrap();
    assert_eq!(first.file(), file!());
    assert_eq!(first.line(), origin);
    assert!(
        error
            .contexts()
            .any(|context| context.as_text() == Some("native context"))
    );
    assert!(
        error.frames().count() >= 3,
        "conversion and public propagation append frames"
    );
    let cause = std::error::Error::source(error.error())
        .unwrap()
        .downcast_ref::<Native>()
        .unwrap();
    assert_eq!(&*cause.0 as *const u32, allocation);
    assert_eq!(
        log.outcomes.lock().unwrap()["caller/score image pair"],
        Outcome::Failed
    );
}
