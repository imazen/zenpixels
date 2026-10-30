//! Internal: runtime op tracer for in-repo tests. Not stable public API —
//! the `__` module prefix and `#[doc(hidden)]` mark it as internal-only,
//! same convention as `__bench_u16_hybrids`. Do not depend on this
//! externally; the surface may change without a semver bump.
//!
//! When the `__trace_ops` feature is enabled, every `ConvertStep` dispatched
//! through `apply_step_u8` is recorded by name to a thread-local
//! `Vec<&'static str>`. Tests use this to assert that conversions execute
//! the expected sequence of kernels without redundant work or silent skips.
//!
//! When the feature is off (the default), `record_step` (crate-internal,
//! so not linkable here) is an `#[inline(always)]` empty function — the
//! call site lowers to no instructions and the recording infrastructure
//! compiles out entirely.
//! Production builds pay literally nothing.
//!
//! Step *parameters* (luma coefficients, matte color, etc.) are not
//! recorded — those are verified by inspecting the `ConvertPlan` via its
//! `Debug` impl, which already shows the resolved parameters and is not
//! feature-gated.
//!
//! Usage in tests (gated on `cfg(feature = "__trace_ops")`):
//!
//! ```ignore
//! use zenpixels_convert::__trace_ops;
//! __trace_ops::start_recording();
//! conv.convert_row(&src, &mut dst, width);
//! let steps = __trace_ops::stop_recording();
//! assert_eq!(steps, vec!["RgbToGray"]);
//! ```
//!
//! `start_recording` / `stop_recording` are paired per thread; nested
//! recording overwrites the inner buffer (last writer wins). Tests should
//! avoid nesting and run with `--test-threads=1` if multiple recording
//! tests would otherwise race on the thread-local.

use crate::convert::ConvertStep;
use alloc::vec::Vec;

#[cfg(feature = "__trace_ops")]
mod inner {
    use super::*;
    use std::cell::RefCell;

    std::thread_local! {
        static TRACE: RefCell<Option<Vec<&'static str>>> =
            const { RefCell::new(None) };
    }

    /// Begin recording dispatched step names on this thread to a fresh
    /// buffer. Any prior buffer is dropped.
    pub fn start_recording() {
        TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
    }

    /// Stop recording and return the captured step-name sequence. Returns
    /// an empty Vec if [`start_recording`] wasn't called on this thread.
    pub fn stop_recording() -> Vec<&'static str> {
        TRACE.with(|t| t.borrow_mut().take().unwrap_or_default())
    }

    /// Record a step name. Called from the kernel dispatch.
    ///
    /// The name source is [`ConvertStep::variant_name`] — a single source of
    /// truth shared with [`crate::estimate::step_name`]. Adding a variant
    /// to `ConvertStep` and forgetting to extend `variant_name` is a
    /// compile error (exhaustive match in `convert.rs`), so the tracer
    /// can't silently miss new ops.
    #[inline]
    pub(crate) fn record_step(step: &ConvertStep) {
        TRACE.with(|t| {
            if let Some(v) = t.borrow_mut().as_mut() {
                v.push(step.variant_name());
            }
        });
    }
}

#[cfg(not(feature = "__trace_ops"))]
mod inner {
    use super::*;

    /// No-op when `trace_ops` is disabled.
    pub fn start_recording() {}

    /// Returns an empty Vec when `trace_ops` is disabled.
    pub fn stop_recording() -> Vec<&'static str> {
        Vec::new()
    }

    /// No-op. Empty body is `#[inline(always)]` so call sites lower to nothing.
    #[inline(always)]
    pub(crate) fn record_step(_step: &ConvertStep) {}
}

pub(crate) use inner::record_step;
pub use inner::{start_recording, stop_recording};
