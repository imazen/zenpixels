//! Pluggable HDR→SDR tone mapping for [`ConvertPlan`](crate::ConvertPlan).
//!
//! [`ToneMapper`] is the permanent dynamic-dispatch surface that lets
//! callers inject their own HDR curves into the conversion pipeline. The
//! in-crate reference implementation is [`Bt2446A`](crate::hdr::Bt2446A);
//! downstream crates (notably `zentone`) ship richer mapper menus
//! (FilmicSpline, ACES, ITU-R BT.2408, Möbius, Yrg, Jzazbz, …) and inject
//! them via an extension trait on
//! [`ConvertPlanBuilder`](crate::ConvertPlanBuilder).
//!
//! See the crate-level
//! ["Pluggable HDR tone mapping" section](crate#pluggable-hdr-tone-mapping)
//! for the full usage story.
//!
//! Gated behind `hdr-experimental` together with the rest of the HDR
//! surface in this module. The trait shape itself is intentionally locked
//! once landed — see the type-level docs for the invariants every
//! implementation must uphold.

use core::fmt::Debug;
use core::panic::{RefUnwindSafe, UnwindSafe};

use crate::ColorPrimaries;

/// Pluggable HDR→SDR tone mapper for
/// [`ConvertPlan`](crate::ConvertPlan).
///
/// Implementations live in three places:
///
/// - **This crate.** [`Bt2446A`](crate::hdr::Bt2446A) — the ITU-R BT.2446-1
///   §4 Method A reference curve, the default mapper used when a plan is
///   built without explicit injection.
/// - **`zentone`.** FilmicSpline, AcesRrt, ITU-R BT.2408, Möbius, Yrg,
///   Jzazbz, … injected through
///   [`ConvertPlanBuilder::with_tone_mapper`](crate::ConvertPlanBuilder::with_tone_mapper)
///   plus a zentone-side extension trait.
/// - **Downstream callers.** Any custom curve — bespoke film emulation,
///   research operators, etc. — implements this trait and feeds it into the
///   builder.
///
/// The trait is **object-safe**:
/// [`ConvertPlan`](crate::ConvertPlan) stores `Arc<dyn ToneMapper>`
/// internally, so the dispatch table never has to know the concrete
/// mapper type.
///
/// [`ConvertPlan::new_with_hdr_peak`]: crate::ConvertPlan::new_with_hdr_peak
/// [`ConvertPlan::new_with_hdr_config`]: crate::ConvertPlan::new_with_hdr_config
///
/// # Invariants
///
/// - **Pure and deterministic.** Given `&self`, every method must be a
///   pure function of its inputs. Interior mutability that perturbs
///   output is forbidden.
/// - **Hot-loop safe.** [`map_strip`](Self::map_strip) is called from
///   strip-encoder loops; it must not allocate on its own, must not
///   panic on well-formed inputs (see the precondition list on the
///   method), and should be inlinable around the loop.
/// - **`Send + Sync + UnwindSafe + RefUnwindSafe + Debug`.** The
///   supertrait bounds are required so plans (which are `Clone + Debug`)
///   keep their auto-traits across the `Arc<dyn ToneMapper>` field
///   stored on `ConvertStep::ToneMap`. Mappers are pure functions of
///   their inputs and pose no panic-safety hazard; concrete
///   implementations virtually always satisfy this without thought
///   (any `#[derive(Debug)]` struct of plain numeric / `Arc` fields
///   does). Implementations should keep their `Debug` short — a
///   single-line `"MyMapper { … }"` is plenty.
/// - **Stable [`name`](Self::name).** The string is consumed by
///   diagnostics, trace recorders, and estimate / oracle cache keys.
///   Treat it as part of the implementation's public contract: once
///   shipped, do not rename across releases.
///
/// # Working space
///
/// The strip handed to [`map_strip`](Self::map_strip) is **linear-light**
/// RGB in the primaries reported by
/// [`working_primaries`](Self::working_primaries) (BT.2020 by default —
/// the canonical HDR working space). Mappers that operate in a different
/// gamut (a few Filmic variants prefer DCI-P3) override
/// `working_primaries`; the pipeline inserts an extra gamut matrix on
/// each side of the strip to honor that.
pub trait ToneMapper: Send + Sync + UnwindSafe + RefUnwindSafe + Debug {
    /// Apply tone mapping to one strip of interleaved RGB pixels.
    ///
    /// **Precondition.** Both slices are interleaved RGB triplets, i.e.
    /// `input.len() % 3 == 0` and `input.len() == output.len()`. The
    /// strip is interpreted in the primaries returned by
    /// [`working_primaries`](Self::working_primaries).
    ///
    /// **In-place permitted.** Implementations MAY support
    /// `input.as_ptr() == output.as_ptr()` (the same slice passed twice
    /// reinterpreted as immutable + mutable views by the caller), but
    /// MUST behave correctly either way.
    ///
    /// **No allocation, no panic.** Hot path — implementations should
    /// avoid heap allocation and must not panic on well-formed inputs.
    fn map_strip(&self, input: &[f32], output: &mut [f32]);

    /// The working color primaries this mapper expects on its input
    /// strip.
    ///
    /// Default: [`ColorPrimaries::Bt2020`] — the canonical HDR working
    /// space. Mappers operating in a different gamut override this; the
    /// pipeline inserts a matrix-multiply on each side of the strip to
    /// reach the working primaries.
    fn working_primaries(&self) -> ColorPrimaries {
        ColorPrimaries::Bt2020
    }

    /// `(source_peak_nits, target_peak_nits)` the mapper was constructed
    /// with, if it is peak-aware.
    ///
    /// Returns `None` when the mapper is peak-agnostic (a curve defined
    /// purely in normalized space). The pipeline uses this for
    /// diagnostics only — the mapper is the source of truth for its own
    /// normalization.
    fn peaks(&self) -> Option<(f32, f32)> {
        None
    }

    /// Short kebab-case identifier for diagnostics, trace logs, and
    /// estimate / oracle cache keys.
    ///
    /// Examples: `"bt2446a"`, `"filmic-spline"`, `"aces-rrt"`,
    /// `"itu2408"`. Must be stable across releases of the implementing
    /// crate — treat changing it as a breaking change.
    fn name(&self) -> &'static str;

    /// Estimated wall-clock nanoseconds per megapixel on a baseline
    /// AVX2 host (Ryzen 9 7950X, single-threaded).
    ///
    /// Consumed by
    /// [`ConvertPlan::estimate`](crate::ConvertPlan::estimate) and its
    /// `_in` sibling so a scheduler can budget the mapper alongside the
    /// rest of the plan.
    ///
    /// The default of `1500` is a conservative middle-of-the-road
    /// estimate; override when your mapper is materially faster or
    /// slower than that baseline.
    fn cost_ns_per_mp(&self) -> u32 {
        1500
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    /// Object-safety pin. If the trait stops being dyn-compatible — e.g.
    /// someone adds a generic method or a `Self`-by-value return — this
    /// fails to compile, which is exactly the breakage we want to catch
    /// at the trait-edit site.
    #[allow(dead_code)]
    fn _assert_obj_safe(_: &dyn ToneMapper) {}

    /// Identity mapper used to verify dyn-dispatch round-trips through
    /// the builder. Constant-zero on construction; behaves like the
    /// canonical no-op curve.
    #[derive(Debug)]
    struct IdentityMapper;

    impl ToneMapper for IdentityMapper {
        fn map_strip(&self, input: &[f32], output: &mut [f32]) {
            output.copy_from_slice(input);
        }
        fn name(&self) -> &'static str {
            "identity-test"
        }
    }

    #[test]
    fn default_working_primaries_is_bt2020() {
        let m = IdentityMapper;
        assert_eq!(m.working_primaries(), ColorPrimaries::Bt2020);
    }

    #[test]
    fn default_peaks_is_none() {
        let m = IdentityMapper;
        assert_eq!(m.peaks(), None);
    }

    #[test]
    fn default_cost_is_reasonable() {
        let m = IdentityMapper;
        // Sanity-check the default cost is in the right order of
        // magnitude for a per-megapixel ns budget; the exact value is
        // documented on the trait.
        assert_eq!(m.cost_ns_per_mp(), 1500);
    }

    #[test]
    fn identity_strip_roundtrips_through_dyn() {
        let mapper: Arc<dyn ToneMapper> = Arc::new(IdentityMapper);
        let input = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut output = [0.0f32; 6];
        mapper.map_strip(&input, &mut output);
        assert_eq!(input, output);
        assert_eq!(mapper.name(), "identity-test");
    }
}
