//! Pluggable HDR→SDR tone mapping for [`ConvertPlan`](crate::ConvertPlan).
//!
//! [`ToneMapper`] lets callers inject their own HDR curve into a conversion plan. It has two phases:
//!
//! 1. **Prepare.** The mapper receives a [`ToneMapContext`] with the peaks and working primaries, and
//!    produces a [`PreparedMapper`]: typically a [`StripMapper`] with its coefficients baked in.
//! 2. **Apply.** The plan calls [`StripMapper::map_strip`] per strip during row conversion.
//!
//! The caller supplies the source peak through [`HdrConfig`](crate::HdrConfig). The plan measures nothing:
//! PQ's peak needs a pass over the whole image first, as
//! [`convert_to_sdr_measuring_peak`](crate::ext::PixelBufferHdrConvertExt::convert_to_sdr_measuring_peak)
//! does. The trait is object-safe, so a plan holds `Arc<dyn ToneMapper>` without monomorphization.
//!
//! [`Bt2446A`](crate::hdr::Bt2446A) is this crate's implementation, the ITU-R BT.2446-1 Method A curve.
//! Any other curve implements the trait and goes to
//! [`ConvertPlan::new_with_tone_mapper`](crate::ConvertPlan::new_with_tone_mapper).
//!
//! Gated behind `hdr-experimental` together with the rest of the HDR surface in this module.

use alloc::boxed::Box;
use core::fmt::Debug;
use core::panic::{RefUnwindSafe, UnwindSafe};

use crate::ColorPrimaries;

/// Pluggable HDR→SDR tone mapper for [`ConvertPlan`](crate::ConvertPlan).
///
/// Implementations declare their analysis needs via
/// [`requirements`](Self::requirements), then receive the gathered
/// [`ToneMapContext`] in [`prepare`](Self::prepare) and produce a
/// [`PreparedMapper`] — typically a [`StripMapper`] with image-specific
/// coefficients precomputed.
///
/// The trait is **object-safe**: [`ConvertPlan`](crate::ConvertPlan)
/// stores `Arc<dyn ToneMapper>` internally, so the dispatch table never
/// has to know the concrete mapper type.
///
/// # Invariants
///
/// - **Pure and deterministic.** Given `&self` and a `ToneMapContext`,
///   every method must be a pure function of its inputs. Interior
///   mutability that perturbs output is forbidden.
/// - **`Send + Sync + UnwindSafe + RefUnwindSafe + Debug`.** The
///   supertrait bounds are required so plans (which are `Clone + Debug`)
///   keep their auto-traits across the `Arc<dyn ToneMapper>` field
///   stored on `ConvertStep::ToneMap`. Mappers are pure functions of
///   their inputs and pose no panic-safety hazard; concrete
///   implementations virtually always satisfy this without thought
///   (any `#[derive(Debug)]` struct of plain numeric / `Arc` fields
///   does).
/// - **Stable [`name`](Self::name).** The string is consumed by
///   diagnostics, trace recorders, and estimate / oracle cache keys.
///   Treat it as part of the implementation's public contract: once
///   shipped, do not rename across releases.
///
/// # Working space
///
/// The strip handed to [`StripMapper::map_strip`] is **linear-light** RGB
/// in the primaries declared by
/// [`working_primaries_for`](Self::working_primaries_for) (BT.2020 by
/// default — the canonical HDR working space). Mappers that operate in a
/// different gamut (a few Filmic variants prefer DCI-P3) override
/// `working_primaries_for`; the pipeline inserts an extra gamut matrix
/// on each side of the strip to honor that.
pub trait ToneMapper: Send + Sync + UnwindSafe + RefUnwindSafe + Debug {
    /// What pipeline-gathered context does this mapper need before it
    /// can produce a [`PreparedMapper`]?
    ///
    /// Examples:
    ///
    /// - A pointwise curve with peaks pre-baked at construction returns
    ///   [`Requirements::NONE`] — no analysis needed.
    /// - A pointwise curve that wants the pipeline to manage peaks
    ///   returns [`Requirements::SOURCE_PEAK_ONLY`].
    /// - A local operator (Mantiuk, Fattal, bilateral) returns
    ///   `Requirements { full_image: true, ..Requirements::NONE }` —
    ///   currently rejected by the builder (reserved for a future
    ///   release; see [`PreparedMapper::Streaming`]).
    fn requirements(&self) -> Requirements;

    /// Build a per-image strip executor from the gathered context.
    ///
    /// Called **once** by the pipeline after the analysis pass.
    /// Implementations should precompute curve coefficients / LUTs /
    /// adaptation fields here, then return a [`PreparedMapper`] that the
    /// pipeline applies strip-by-strip during row dispatch.
    ///
    /// `ctx` carries every field the mapper declared in
    /// [`requirements`](Self::requirements), plus the always-populated
    /// `source_peak_nits` / `target_peak_nits` / `working_primaries`.
    fn prepare(&self, ctx: &ToneMapContext) -> PreparedMapper;

    /// Short kebab-case identifier for diagnostics, trace logs, and
    /// estimate / oracle cache keys.
    ///
    /// Examples: `"bt2446a"`, `"filmic-spline"`, `"aces-rrt"`,
    /// `"itu2408"`, `"adaptive"`. Must be stable across releases of the
    /// implementing crate — treat changing it as a breaking change.
    fn name(&self) -> &'static str;

    /// The working color primaries this mapper expects on its input
    /// strip, given a target primaries hint.
    ///
    /// Default: [`ColorPrimaries::Bt2020`] — the canonical HDR working
    /// space. Mappers operating in a different gamut override this; the
    /// pipeline inserts a matrix-multiply on each side of the strip to
    /// reach the working primaries.
    ///
    /// The `target` hint lets adaptive mappers pick a working space
    /// matched to the destination (e.g., DCI-P3 for a P3 target). The
    /// default implementation ignores it and always returns BT.2020.
    fn working_primaries_for(&self, _target: ColorPrimaries) -> ColorPrimaries {
        ColorPrimaries::Bt2020
    }
}

/// Per-image strip executor produced by [`ToneMapper::prepare`].
///
/// `Streaming` is the dominant case: a precomputed curve / LUT that the
/// pipeline calls strip-by-strip via
/// [`StripMapper::map_strip`].
///
/// `Buffered` is reserved for local operators that need to write their
/// entire output during `prepare()` (e.g., Mantiuk gradient processing,
/// Fattal, bilateral). The pipeline would read the buffer strip-by-strip
/// during the apply pass. **Reserved for a future release; v0.2.16
/// pipeline asserts `!requirements.full_image` at builder time** — the
/// variant exists in the enum under `#[non_exhaustive]` so callers can
/// exhaustively match without breaking when the variant lands.
#[non_exhaustive]
#[derive(Debug)]
pub enum PreparedMapper {
    /// Strip-by-strip executor. The pipeline calls
    /// [`StripMapper::map_strip`] per strip with
    /// image-specific coefficients already baked into the boxed
    /// implementation.
    Streaming(Box<dyn StripMapper>),
    // Buffered(Box<[f32]>),  // reserved — see enum-level docs
}

/// Compiled-for-this-image strip executor.
///
/// Produced by [`ToneMapper::prepare`] with image-specific coefficients
/// (peak-aware curve coeffs, adaptation field, LUT rebuild) already baked
/// in. The pipeline calls [`map_strip`](Self::map_strip) for each strip
/// in the convert plan.
///
/// # Invariants
///
/// - **`Send + Sync + UnwindSafe + RefUnwindSafe + Debug`** — same
///   rationale as on [`ToneMapper`]: the strip mapper rides inside a
///   `OnceCell<Arc<dyn StripMapper>>` cached on the plan, which must
///   carry its auto-traits through.
/// - **No allocation, no panic on well-formed inputs.** Hot path —
///   implementations should avoid heap allocation in
///   [`map_strip`](Self::map_strip) and must not panic when the input
///   and output slices satisfy the documented precondition.
pub trait StripMapper: Send + Sync + UnwindSafe + RefUnwindSafe + Debug {
    /// Apply tone mapping to an interleaved RGB strip in linear-light
    /// working space.
    ///
    /// **Precondition.** Both slices are interleaved RGB triplets, i.e.
    /// `input.len() % 3 == 0` and `input.len() == output.len()`. The
    /// strip is interpreted in the primaries the parent
    /// [`ToneMapper`] reports from
    /// [`working_primaries_for`](ToneMapper::working_primaries_for)
    /// (default BT.2020).
    ///
    /// **In-place permitted.** Implementations MAY support
    /// `input.as_ptr() == output.as_ptr()` (the same slice passed twice
    /// reinterpreted as immutable + mutable views by the caller), but
    /// MUST behave correctly either way.
    fn map_strip(&self, input: &[f32], output: &mut [f32]);
}

/// Pipeline analysis requirements declared by a [`ToneMapper`].
///
/// The pipeline reads this **before** building the execution plan and
/// inserts analysis steps for whichever fields are `true`:
///
/// - [`source_peak_nits`](Self::source_peak_nits) — pipeline measures via
///   the in-crate [`CllMeasure`](crate::hdr::CllMeasure) at the
///   buffer-level entry when
///   [`HdrConfig::source_peak_nits`](crate::HdrConfig::source_peak_nits)
///   is `0.0` (the sentinel for "not provided"); otherwise the value
///   from `HdrConfig` is forwarded into [`ToneMapContext`].
/// - [`full_image`](Self::full_image) — pipeline buffers the entire
///   source image before calling `prepare`. **Reserved for a future
///   release.** Setting `true` in v0.2.16 returns
///   [`FullImageMapperNotSupported`](crate::ConvertError::FullImageMapperNotSupported)
///   from the builder.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Requirements {
    /// The mapper reads [`ToneMapContext::source_peak_nits`], which the plan forwards from
    /// [`HdrConfig`](crate::HdrConfig).
    pub source_peak_nits: bool,
    /// The pipeline must buffer the entire source image before calling
    /// `prepare`. **Reserved for a future release.** Setting `true` returns
    /// [`FullImageMapperNotSupported`](crate::ConvertError::FullImageMapperNotSupported)
    /// from
    /// [`ConvertPlan::new_with_tone_mapper`](crate::ConvertPlan::new_with_tone_mapper).
    pub full_image: bool,
}

impl Requirements {
    /// No analysis required. The mapper is fully self-contained (peaks
    /// pre-baked at construction, no image stats needed).
    pub const NONE: Self = Self {
        source_peak_nits: false,
        full_image: false,
    };

    /// Only source peak is needed — the common case for any pointwise
    /// tone curve that wants the pipeline to measure or supply the
    /// peak (e.g., [`Bt2446A::pipeline_managed`](crate::hdr::Bt2446A::pipeline_managed)).
    pub const SOURCE_PEAK_ONLY: Self = Self {
        source_peak_nits: true,
        full_image: false,
    };
}

/// Gathered analysis context handed to [`ToneMapper::prepare`].
///
/// Fields are populated by the pipeline based on [`Requirements`] flags.
/// Optional fields are `Some` iff the corresponding flag was `true`.
/// Required fields ([`source_peak_nits`](Self::source_peak_nits),
/// [`target_peak_nits`](Self::target_peak_nits),
/// [`working_primaries`](Self::working_primaries)) are always populated.
///
/// Constructed by the pipeline; downstream code generally only reads
/// this. The struct is `#[non_exhaustive]` so additional optional stats
/// can be added in patch releases.
#[non_exhaustive]
#[derive(Debug)]
pub struct ToneMapContext {
    /// Resolved source peak luminance in cd/m². Either supplied via
    /// [`HdrConfig::source_peak_nits`](crate::HdrConfig::source_peak_nits)
    /// or measured by the pipeline using
    /// [`CllMeasure::measure_max`](crate::hdr::CllMeasure::measure_max).
    pub source_peak_nits: f32,
    /// Target peak luminance in cd/m². Sourced from
    /// [`HdrConfig::target_peak_nits`](crate::HdrConfig::target_peak_nits)
    /// (default `100.0`).
    pub target_peak_nits: f32,
    /// Working primaries the strip will be in. Default
    /// [`ColorPrimaries::Bt2020`].
    pub working_primaries: ColorPrimaries,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    /// Object-safety pin for [`ToneMapper`]. If the trait stops being
    /// dyn-compatible — e.g. someone adds a generic method or a
    /// `Self`-by-value return — this fails to compile, which is exactly
    /// the breakage we want to catch at the trait-edit site.
    #[allow(dead_code)]
    fn _assert_obj_safe_tone_mapper(_: &dyn ToneMapper) {}

    /// Object-safety pin for [`StripMapper`]. Same justification as
    /// `_assert_obj_safe_tone_mapper`.
    #[allow(dead_code)]
    fn _assert_obj_safe_strip_mapper(_: &dyn StripMapper) {}

    #[test]
    fn requirements_none_constant_is_all_false() {
        let r = Requirements::NONE;
        assert!(!r.source_peak_nits);
        assert!(!r.full_image);
        assert_eq!(r, Requirements::default());
    }

    #[test]
    fn requirements_source_peak_only_constant() {
        let r = Requirements::SOURCE_PEAK_ONLY;
        assert!(r.source_peak_nits);
        assert!(!r.full_image);
    }

    #[test]
    fn requirements_clone_and_debug_round_trip() {
        let r = Requirements::SOURCE_PEAK_ONLY;
        let r2 = r;
        assert_eq!(r, r2);
        let s = alloc::format!("{r:?}");
        assert!(s.contains("Requirements"), "Debug shape: {s}");
    }

    /// Mock that records what context fields it observed at
    /// `prepare()` time, using atomic storage so the `&self` method
    /// can mutate state without interior unsafe (the crate is
    /// `#![forbid(unsafe_code)]`).
    #[derive(Debug, Default)]
    struct MockMapper {
        observed_source_peak_bits: core::sync::atomic::AtomicU32,
        observed_target_peak_bits: core::sync::atomic::AtomicU32,
        // `u64::MAX` sentinel = unobserved; otherwise low 32 bits = the
        // observed f32 bits, high 32 bits = 1 if `Some`, 0 if `None`.
    }

    impl MockMapper {
        const LOG_AVG_UNSEEN: u64 = u64::MAX;
    }

    impl ToneMapper for MockMapper {
        fn requirements(&self) -> Requirements {
            Requirements {
                source_peak_nits: true,
                ..Requirements::NONE
            }
        }
        fn prepare(&self, ctx: &ToneMapContext) -> PreparedMapper {
            use core::sync::atomic::Ordering;
            self.observed_source_peak_bits
                .store(ctx.source_peak_nits.to_bits(), Ordering::Relaxed);
            self.observed_target_peak_bits
                .store(ctx.target_peak_nits.to_bits(), Ordering::Relaxed);
            PreparedMapper::Streaming(Box::new(IdentityStripMapper))
        }
        fn name(&self) -> &'static str {
            "mock"
        }
    }

    #[derive(Debug)]
    struct IdentityStripMapper;
    impl StripMapper for IdentityStripMapper {
        fn map_strip(&self, input: &[f32], output: &mut [f32]) {
            output.copy_from_slice(input);
        }
    }

    #[test]
    fn mock_mapper_receives_populated_context() {
        use core::sync::atomic::Ordering;
        let mapper = MockMapper {
            ..MockMapper::default()
        };
        // Drive `prepare` directly to verify the field plumbing: the
        // pipeline integration test in `tests/builder.rs` exercises the
        // end-to-end flow.
        let ctx = ToneMapContext {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            working_primaries: ColorPrimaries::Bt2020,
        };
        let _prepared = mapper.prepare(&ctx);
        assert_eq!(
            f32::from_bits(mapper.observed_source_peak_bits.load(Ordering::Relaxed)),
            1000.0
        );
        assert_eq!(
            f32::from_bits(mapper.observed_target_peak_bits.load(Ordering::Relaxed)),
            100.0
        );
    }

    #[test]
    fn dyn_dispatch_round_trip() {
        let mapper: Arc<dyn ToneMapper> = Arc::new(MockMapper::default());
        assert_eq!(mapper.name(), "mock");
        let r = mapper.requirements();
        assert!(r.source_peak_nits);
        assert!(!r.full_image);
    }
}
