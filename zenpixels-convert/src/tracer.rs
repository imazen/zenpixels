//! Runtime op tracer for `convert_row`.
//!
//! When the `trace_ops` feature is enabled, every `ConvertStep` dispatched
//! through `apply_step_u8` is recorded by name to a thread-local
//! `Vec<&'static str>`. Tests use this to assert that conversions execute
//! the expected sequence of kernels without redundant work or silent skips.
//!
//! When the feature is off (the default), [`record_step`] is an
//! `#[inline(always)]` empty function — the call site lowers to no
//! instructions and the recording infrastructure compiles out entirely.
//! Production builds pay literally nothing.
//!
//! Step *parameters* (luma coefficients, matte color, etc.) are not
//! recorded — those are verified by inspecting the `ConvertPlan` via its
//! `Debug` impl, which already shows the resolved parameters and is not
//! feature-gated.
//!
//! Usage in tests (gated on `cfg(feature = "trace_ops")`):
//!
//! ```ignore
//! use zenpixels_convert::tracer;
//! tracer::start_recording();
//! conv.convert_row(&src, &mut dst, width);
//! let steps = tracer::stop_recording();
//! assert_eq!(steps, vec!["RgbToGray"]);
//! ```
//!
//! `start_recording` / `stop_recording` are paired per thread; nested
//! recording overwrites the inner buffer (last writer wins). Tests should
//! avoid nesting and run with `--test-threads=1` if multiple recording
//! tests would otherwise race on the thread-local.

use crate::convert::ConvertStep;
use alloc::vec::Vec;

/// The variant name of a `ConvertStep` — used by the tracer. Exhaustive
/// match: adding a variant to `ConvertStep` without adding it here is a
/// compile error, so the tracer can't silently miss new ops.
#[cfg(feature = "trace_ops")]
fn step_name(step: &ConvertStep) -> &'static str {
    match step {
        ConvertStep::Identity => "Identity",
        ConvertStep::SwizzleBgraRgba => "SwizzleBgraRgba",
        ConvertStep::RgbToBgra => "RgbToBgra",
        ConvertStep::AddAlpha => "AddAlpha",
        ConvertStep::DropAlpha => "DropAlpha",
        ConvertStep::MatteComposite { .. } => "MatteComposite",
        ConvertStep::GrayToRgb => "GrayToRgb",
        ConvertStep::GrayToRgba => "GrayToRgba",
        ConvertStep::RgbToGray { .. } => "RgbToGray",
        ConvertStep::RgbaToGray { .. } => "RgbaToGray",
        ConvertStep::GrayAlphaToRgba => "GrayAlphaToRgba",
        ConvertStep::GrayAlphaToRgb => "GrayAlphaToRgb",
        ConvertStep::GrayToGrayAlpha => "GrayToGrayAlpha",
        ConvertStep::GrayAlphaToGray => "GrayAlphaToGray",
        ConvertStep::SrgbU8ToLinearF32 => "SrgbU8ToLinearF32",
        ConvertStep::LinearF32ToSrgbU8 => "LinearF32ToSrgbU8",
        ConvertStep::NaiveU8ToF32 => "NaiveU8ToF32",
        ConvertStep::NaiveF32ToU8 => "NaiveF32ToU8",
        ConvertStep::U16ToU8 => "U16ToU8",
        ConvertStep::U8ToU16 => "U8ToU16",
        ConvertStep::U16ToF32 => "U16ToF32",
        ConvertStep::F32ToU16 => "F32ToU16",
        ConvertStep::F16ToF32 => "F16ToF32",
        ConvertStep::F32ToF16 => "F32ToF16",
        ConvertStep::PqU16ToLinearF32 => "PqU16ToLinearF32",
        ConvertStep::LinearF32ToPqU16 => "LinearF32ToPqU16",
        ConvertStep::PqF32ToLinearF32 => "PqF32ToLinearF32",
        ConvertStep::LinearF32ToPqF32 => "LinearF32ToPqF32",
        ConvertStep::HlgU16ToLinearF32 => "HlgU16ToLinearF32",
        ConvertStep::LinearF32ToHlgU16 => "LinearF32ToHlgU16",
        ConvertStep::HlgF32ToLinearF32 => "HlgF32ToLinearF32",
        ConvertStep::LinearF32ToHlgF32 => "LinearF32ToHlgF32",
        ConvertStep::SrgbF32ToLinearF32 => "SrgbF32ToLinearF32",
        ConvertStep::LinearF32ToSrgbF32 => "LinearF32ToSrgbF32",
        ConvertStep::SrgbF32ToLinearF32Extended => "SrgbF32ToLinearF32Extended",
        ConvertStep::LinearF32ToSrgbF32Extended => "LinearF32ToSrgbF32Extended",
        ConvertStep::Bt709F32ToLinearF32 => "Bt709F32ToLinearF32",
        ConvertStep::LinearF32ToBt709F32 => "LinearF32ToBt709F32",
        ConvertStep::Gamma22F32ToLinearF32 => "Gamma22F32ToLinearF32",
        ConvertStep::LinearF32ToGamma22F32 => "LinearF32ToGamma22F32",
        ConvertStep::StraightToPremul => "StraightToPremul",
        ConvertStep::PremulToStraight => "PremulToStraight",
        ConvertStep::LinearRgbToOklab => "LinearRgbToOklab",
        ConvertStep::OklabToLinearRgb => "OklabToLinearRgb",
        ConvertStep::LinearRgbaToOklaba => "LinearRgbaToOklaba",
        ConvertStep::OklabaToLinearRgba => "OklabaToLinearRgba",
        ConvertStep::GamutMatrixRgbF32(_) => "GamutMatrixRgbF32",
        ConvertStep::GamutMatrixRgbaF32(_) => "GamutMatrixRgbaF32",
        ConvertStep::FusedSrgbU8GamutRgb(_) => "FusedSrgbU8GamutRgb",
        ConvertStep::FusedSrgbU8GamutRgba(_) => "FusedSrgbU8GamutRgba",
        ConvertStep::FusedSrgbU16GamutRgb(_) => "FusedSrgbU16GamutRgb",
        ConvertStep::FusedSrgbU8ToLinearF32Rgb(_) => "FusedSrgbU8ToLinearF32Rgb",
        ConvertStep::FusedLinearF32ToSrgbU8Rgb(_) => "FusedLinearF32ToSrgbU8Rgb",
    }
}

#[cfg(feature = "trace_ops")]
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
    #[inline]
    pub(crate) fn record_step(step: &ConvertStep) {
        TRACE.with(|t| {
            if let Some(v) = t.borrow_mut().as_mut() {
                v.push(step_name(step));
            }
        });
    }
}

#[cfg(not(feature = "trace_ops"))]
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

pub use inner::{start_recording, stop_recording};
pub(crate) use inner::record_step;
