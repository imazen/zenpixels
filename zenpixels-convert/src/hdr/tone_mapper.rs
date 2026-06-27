//! Pluggable HDR→SDR tone mapping for [`ConvertPlan`](crate::ConvertPlan).
//!
//! [`ToneMapper`] is the permanent dependency-injection surface that lets
//! callers inject their own HDR curves into the conversion pipeline. The
//! shape is intentionally two-phase to fit every real tone-mapping
//! operator in the literature:
//!
//! 1. **Analysis** — the pipeline gathers whatever stats the mapper
//!    declares via [`requirements`](ToneMapper::requirements): source peak
//!    nits (measured via [`CllMeasure`](crate::hdr::CllMeasure) when not
//!    supplied), log-average luminance, luminance histogram, full-image
//!    buffer.
//! 2. **Prepare** — the mapper receives the gathered [`ToneMapContext`]
//!    and produces a [`PreparedMapper`] — typically a precomputed
//!    [`StripMapper`] with image-specific coefficients (peak-aware curve
//!    coefficients, adaptation field, LUT) already baked in.
//! 3. **Apply** — the pipeline calls
//!    [`StripMapper::map_strip`](StripMapper::map_strip) per strip during
//!    row-by-row conversion.
//!
//! The split mirrors how every real tone-mapping operator works in
//! practice: an analysis pass over the whole image (when needed), then a
//! fast curve or LUT applied strip-by-strip. The trait is object-safe so
//! the pipeline holds `Arc<dyn ToneMapper>` without monomorphization.
//!
//! # Implementations
//!
//! - **This crate.** [`Bt2446A`](crate::hdr::Bt2446A) — the ITU-R
//!   BT.2446-1 §4 Method A reference curve.
//! - **`zentone`.** FilmicSpline, AcesRrt, ITU-R BT.2408, Möbius, Yrg,
//!   Jzazbz, AdaptiveTonemapper, DetectedCurve, ProfileToneCurve, … —
//!   injected via [`ConvertPlanBuilder::with_tone_mapper`](crate::ConvertPlanBuilder::with_tone_mapper)
//!   plus a zentone-side extension trait that constructs the mapper from
//!   the builder's staged [`HdrConfig`](crate::HdrConfig).
//! - **Downstream callers.** Any custom curve — bespoke film emulation,
//!   research operators, etc. — implements this trait and feeds it into
//!   the builder.
//!
//! Gated behind `hdr-experimental` together with the rest of the HDR
//! surface in this module. The trait shape is intentionally locked.

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
    /// - An adaptive mapper returns
    ///   `Requirements { source_peak_nits: true, luminance_histogram: true,
    ///   ..Requirements::NONE }`.
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
    fn prepare(&self, ctx: &ToneMapContext<'_>) -> PreparedMapper;

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

    /// Estimated wall-clock nanoseconds per megapixel of the
    /// **strip-apply** cost on a baseline AVX2 host (Ryzen 9 7950X,
    /// single-threaded).
    ///
    /// Consumed by
    /// [`ConvertPlan::estimate`](crate::ConvertPlan::estimate) and its
    /// `_in` sibling so a scheduler can budget the mapper alongside the
    /// rest of the plan. The unit is the same as every other
    /// `*_ns_per_mp` value the estimate module uses internally.
    ///
    /// Includes only the per-strip cost (the [`StripMapper`]). Analysis-
    /// phase costs (peak measurement, histogram building) are accounted
    /// separately by the pipeline using the existing
    /// [`CllMeasure`](crate::hdr::CllMeasure) calibration. The
    /// `4_000_000` default (≈250 Mpix/s — a typical SIMD tone-curve
    /// throughput) is a conservative middle-of-the-road estimate;
    /// override when your mapper is materially faster or slower.
    fn cost_ns_per_mp(&self) -> u32 {
        4_000_000
    }
}

/// Per-image strip executor produced by [`ToneMapper::prepare`].
///
/// `Streaming` is the dominant case: a precomputed curve / LUT that the
/// pipeline calls strip-by-strip via
/// [`StripMapper::map_strip`](StripMapper::map_strip).
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
    /// [`StripMapper::map_strip`](StripMapper::map_strip) per strip with
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
    /// [`ToneMapper`](ToneMapper) reports from
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
/// - [`log_avg_luminance`](Self::log_avg_luminance) — pipeline computes
///   `exp(mean(log(luminance + ε)))` over the source image during the
///   analysis pass. Reserved for adaptive mappers.
/// - [`luminance_histogram`](Self::luminance_histogram) — pipeline
///   builds a [`LuminanceHistogram::NUM_BINS`]-bin log-luminance
///   histogram. Reserved for histogram-equalization mappers.
/// - [`full_image`](Self::full_image) — pipeline buffers the entire
///   source image before calling `prepare`. **Reserved for a future
///   release.** Setting `true` in v0.2.16 returns
///   [`ConvertError::FullImageMapperNotSupported`](crate::ConvertError::FullImageMapperNotSupported)
///   from the builder.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Requirements {
    /// The pipeline must populate
    /// [`ToneMapContext::source_peak_nits`](ToneMapContext::source_peak_nits) —
    /// measuring via [`CllMeasure`](crate::hdr::CllMeasure) when
    /// `HdrConfig::source_peak_nits == 0.0`, or forwarding the
    /// caller-supplied value otherwise.
    pub source_peak_nits: bool,
    /// The pipeline must compute
    /// `exp(mean(log(luminance + ε)))` over the source image and supply
    /// it in [`ToneMapContext::log_avg_luminance`](ToneMapContext::log_avg_luminance).
    /// Reserved for a future release (the wiring is not active in
    /// v0.2.16; the field always reads `None`).
    pub log_avg_luminance: bool,
    /// The pipeline must build a log-luminance histogram and supply it
    /// in [`ToneMapContext::luminance_histogram`](ToneMapContext::luminance_histogram).
    /// Reserved for a future release.
    pub luminance_histogram: bool,
    /// The pipeline must buffer the entire source image before calling
    /// `prepare`. **Reserved for a future release.** Setting `true` in
    /// v0.2.16 returns
    /// [`ConvertError::FullImageMapperNotSupported`](crate::ConvertError::FullImageMapperNotSupported)
    /// from
    /// [`ConvertPlanBuilder::build`](crate::ConvertPlanBuilder::build).
    pub full_image: bool,
}

impl Requirements {
    /// No analysis required. The mapper is fully self-contained (peaks
    /// pre-baked at construction, no image stats needed).
    pub const NONE: Self = Self {
        source_peak_nits: false,
        log_avg_luminance: false,
        luminance_histogram: false,
        full_image: false,
    };

    /// Only source peak is needed — the common case for any pointwise
    /// tone curve that wants the pipeline to measure or supply the
    /// peak (e.g., [`Bt2446A::pipeline_managed`](crate::hdr::Bt2446A::pipeline_managed)).
    pub const SOURCE_PEAK_ONLY: Self = Self {
        source_peak_nits: true,
        log_avg_luminance: false,
        luminance_histogram: false,
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
pub struct ToneMapContext<'a> {
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
    /// `Some` iff [`Requirements::log_avg_luminance`] was `true`.
    /// Reserved for a future release.
    pub log_avg_luminance: Option<f32>,
    /// `Some` iff [`Requirements::luminance_histogram`] was `true`.
    /// Reserved for a future release.
    pub luminance_histogram: Option<&'a LuminanceHistogram>,
}

/// 1024-bin log-luminance histogram used for adaptive tone mappers
/// (Reinhard global, Drago, Mantiuk08, histogram-adjustment).
///
/// Bins span `log2(MIN_NITS)..=log2(MAX_NITS)` with `MIN_NITS = 0.005`,
/// `MAX_NITS = 10000.0`. Counts are normalized so that
/// `bins.iter().sum::<f32>() == 1.0` — i.e. bins represent a probability
/// mass function over per-pixel luminance.
///
/// Reserved for a future release: built by the pipeline only when the
/// mapper requests [`Requirements::luminance_histogram`]. The constants
/// and bin count are stable surface; consumers can rely on them.
#[derive(Debug, Clone)]
pub struct LuminanceHistogram {
    /// Normalized counts (sum to 1.0) of per-pixel log-luminance.
    pub bins: [f32; Self::NUM_BINS],
    /// Lower bin edge in log2 nits. Pinned at `log2(0.005)`.
    pub min_log2_nits: f32,
    /// Upper bin edge in log2 nits. Pinned at `log2(10_000.0)`.
    pub max_log2_nits: f32,
}

impl LuminanceHistogram {
    /// Number of bins in the histogram. Reserved as a stable constant
    /// so callers can dimension their adapters.
    pub const NUM_BINS: usize = 1024;
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
        assert!(!r.log_avg_luminance);
        assert!(!r.luminance_histogram);
        assert!(!r.full_image);
        assert_eq!(r, Requirements::default());
    }

    #[test]
    fn requirements_source_peak_only_constant() {
        let r = Requirements::SOURCE_PEAK_ONLY;
        assert!(r.source_peak_nits);
        assert!(!r.log_avg_luminance);
        assert!(!r.luminance_histogram);
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

    #[test]
    fn luminance_histogram_num_bins_pinned() {
        assert_eq!(LuminanceHistogram::NUM_BINS, 1024);
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
        observed_log_avg: core::sync::atomic::AtomicU64,
    }

    impl MockMapper {
        const LOG_AVG_UNSEEN: u64 = u64::MAX;
        fn pack_log_avg(v: Option<f32>) -> u64 {
            match v {
                Some(x) => (1u64 << 32) | u64::from(x.to_bits()),
                None => 0,
            }
        }
        fn unpack_log_avg(bits: u64) -> Option<Option<f32>> {
            if bits == Self::LOG_AVG_UNSEEN {
                return None;
            }
            if bits == 0 {
                return Some(None);
            }
            #[allow(clippy::cast_possible_truncation)]
            let lo = bits as u32;
            Some(Some(f32::from_bits(lo)))
        }
    }

    impl ToneMapper for MockMapper {
        fn requirements(&self) -> Requirements {
            Requirements {
                source_peak_nits: true,
                log_avg_luminance: true,
                ..Requirements::NONE
            }
        }
        fn prepare(&self, ctx: &ToneMapContext<'_>) -> PreparedMapper {
            use core::sync::atomic::Ordering;
            self.observed_source_peak_bits
                .store(ctx.source_peak_nits.to_bits(), Ordering::Relaxed);
            self.observed_target_peak_bits
                .store(ctx.target_peak_nits.to_bits(), Ordering::Relaxed);
            self.observed_log_avg
                .store(Self::pack_log_avg(ctx.log_avg_luminance), Ordering::Relaxed);
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
            observed_log_avg: core::sync::atomic::AtomicU64::new(MockMapper::LOG_AVG_UNSEEN),
            ..MockMapper::default()
        };
        // Drive `prepare` directly to verify the field plumbing: the
        // pipeline integration test in `tests/builder.rs` exercises the
        // end-to-end flow.
        let ctx = ToneMapContext {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            working_primaries: ColorPrimaries::Bt2020,
            log_avg_luminance: Some(0.18),
            luminance_histogram: None,
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
        let log_avg =
            MockMapper::unpack_log_avg(mapper.observed_log_avg.load(Ordering::Relaxed)).unwrap();
        assert_eq!(log_avg, Some(0.18));
    }

    #[test]
    fn dyn_dispatch_round_trip() {
        let mapper: Arc<dyn ToneMapper> = Arc::new(MockMapper::default());
        assert_eq!(mapper.name(), "mock");
        let r = mapper.requirements();
        assert!(r.source_peak_nits);
        assert!(r.log_avg_luminance);
        assert!(!r.full_image);
    }
}
