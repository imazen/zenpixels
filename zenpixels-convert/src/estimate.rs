//! Resource-estimation primitive for [`ConvertPlan`].
//!
//! [`ConvertPlan::estimate`](crate::ConvertPlan::estimate) and
//! [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in) walk a plan's
//! steps and return a [`ResourceEstimate`] (peak memory, wall-ms,
//! intermediate-buffer count) with no allocation or row work, so schedulers
//! can decide whether an op fits a memory budget / SLA before it runs.
//!
//! Accuracy contract, calibration source, threading model, and the
//! foundation-crate / `zencodec::estimate::*` shape-compatibility rationale
//! live in [`docs/ESTIMATE.md`](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/docs/ESTIMATE.md).
//!
//! Every field is `Option`, all structs are `#[non_exhaustive]`, and builders
//! are growable: future fields land additively at every match-bind site.

use crate::PixelDescriptor;
use crate::convert::{ConvertPlan, ConvertStep, FusedKind};

/// SIMD instruction tier the codec will dispatch to. Optional hint on
/// [`ComputeEnvironment`] — a wider/newer tier generally means faster
/// encode/decode, so estimates can apply a per-tier time factor. Variants
/// mirror the `x86-64-vN` microarchitecture levels and the archmage /
/// magetypes token vocabulary, so an archmage-detected tier maps trivially.
///
/// ```rust,ignore
/// use zenpixels_convert::{ComputeEnvironment, SimdTier};
/// let tier = if archmage::X64V4Token::summon().is_some() { SimdTier::X86V4 }
///     else if archmage::X64V3Token::summon().is_some() { SimdTier::X86V3 }
///     else if archmage::X64V2Token::summon().is_some() { SimdTier::X86V2 }
///     else { SimdTier::X86V1 };
/// let env = ComputeEnvironment::new().with_cores(8).with_simd_tier(tier);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimdTier {
    /// SIMD tier unknown — estimates use a conservative cross-tier baseline.
    /// Use [`CurrentHost`](SimdTier::CurrentHost) for the local machine.
    Unknown,
    /// Host running the estimate (≈ calibration host's native tier). Distinct
    /// from [`Unknown`](SimdTier::Unknown) which is a cross-tier average.
    CurrentHost,
    /// WebAssembly, no SIMD128 (scalar wasm).
    Wasm,
    /// WebAssembly SIMD128.
    Wasm128,
    /// AArch64 / ARM NEON (archmage `NeonToken`).
    Neon,
    /// x86-64-v1 — SSE2 baseline.
    X86V1,
    /// x86-64-v2 — SSE4.2 (archmage `X64V2Token`).
    X86V2,
    /// x86-64-v3 — AVX2 + FMA (archmage `X64V3Token`).
    X86V3,
    /// x86-64-v4 — AVX-512 (archmage `X64V4Token`).
    X86V4,
}

/// Hardware + runtime conditions for a resource estimate. `#[non_exhaustive]`
/// and shape-compatible with `zencodec::estimate::ComputeEnvironment`:
/// construct via [`new`](Self::new), refine with `with_*` setters, read with
/// the accessors. Carries cores + optional [`SimdTier`] + optional RAM today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ComputeEnvironment {
    available_cores: usize,
    available_ram_bytes: Option<u64>,
    simd_tier: Option<SimdTier>,
}

#[rustfmt::skip]
impl ComputeEnvironment {
    /// Single-core, unknown RAM, unspecified SIMD tier (conservative default).
    #[must_use] pub fn new() -> Self { Self { available_cores: 1, available_ram_bytes: None, simd_tier: None } }
    /// CPU cores available, clamped to ≥ 1. `std` callers typically pass
    /// `std::thread::available_parallelism()`.
    #[must_use] pub fn with_cores(mut self, cores: usize) -> Self { self.available_cores = cores.max(1); self }
    /// Physical RAM available, for memory-ceiling decisions.
    #[must_use] pub fn with_available_ram_bytes(mut self, bytes: u64) -> Self { self.available_ram_bytes = Some(bytes); self }
    /// SIMD instruction tier the codec will dispatch to.
    #[must_use] pub fn with_simd_tier(mut self, tier: SimdTier) -> Self { self.simd_tier = Some(tier); self }
    /// Available CPU cores (≥ 1).
    #[must_use] pub fn cores(&self) -> usize { self.available_cores }
    /// Available RAM in bytes, if known.
    #[must_use] pub fn available_ram_bytes(&self) -> Option<u64> { self.available_ram_bytes }
    /// The SIMD tier hint, if specified.
    #[must_use] pub fn simd_tier(&self) -> Option<SimdTier> { self.simd_tier }
}

impl Default for ComputeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Characteristics of the image being encoded/decoded. `#[non_exhaustive]`
/// and shape-compatible with `zencodec::estimate::ImageCharacteristics`.
/// Carries dimensions + pixel format today; per-frame, so animation fields
/// belong on codec types, not here.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageCharacteristics {
    width: u32,
    height: u32,
    descriptor: PixelDescriptor,
}

#[rustfmt::skip]
impl ImageCharacteristics {
    /// A still image of `width` × `height` with the given pixel format.
    #[must_use] pub fn new(width: u32, height: u32, descriptor: PixelDescriptor) -> Self { Self { width, height, descriptor } }
    /// Image width in pixels.
    #[must_use] pub fn width(&self) -> u32 { self.width }
    /// Image height in pixels.
    #[must_use] pub fn height(&self) -> u32 { self.height }
    /// The pixel format of the source/decoded buffer.
    #[must_use] pub fn descriptor(&self) -> &PixelDescriptor { &self.descriptor }
}

/// Predicted resources for a conversion plan. Every field is `Option` —
/// fillers may model what they can and leave the rest `None`. `wall_ms` is
/// already scaled to `compute.cores()` when produced by
/// [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in); the internal
/// threading-bottleneck model is in `estimate_plan` (`docs/ESTIMATE.md`).
/// `#[non_exhaustive]`, shape-compatible with
/// `zencodec::estimate::ResourceEstimate`; build via [`new`](Self::new) /
/// [`unknown`](Self::unknown) + the `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ResourceEstimate {
    peak_memory_bytes_est: Option<u64>,
    wall_ms: Option<u64>,
    intermediate_buffer_count: Option<u32>,
}

#[rustfmt::skip]
impl ResourceEstimate {
    /// All-`None` estimate — for stages that don't model their resource use.
    #[must_use] pub fn unknown() -> Self { Self { peak_memory_bytes_est: None, wall_ms: None, intermediate_buffer_count: None } }
    /// Estimate from the two essentials: typical peak memory + wall time.
    /// Buffer count left `None`; refine with
    /// [`with_intermediate_buffer_count`](Self::with_intermediate_buffer_count).
    #[must_use] pub fn new(peak_memory_bytes_est: u64, wall_ms: u64) -> Self {
        Self { peak_memory_bytes_est: Some(peak_memory_bytes_est), wall_ms: Some(wall_ms), intermediate_buffer_count: None }
    }
    /// Full-image intermediate buffers held simultaneously, NOT counting input
    /// or output. Distinguishes 1-giant-buffer from N-medium-buffer plans for
    /// paging-pressure decisions.
    #[must_use] pub fn with_intermediate_buffer_count(mut self, n: u32) -> Self { self.intermediate_buffer_count = Some(n); self }
    /// Typical (≈ p50) estimated peak memory, bytes.
    #[must_use] pub fn peak_memory_bytes_est(&self) -> Option<u64> { self.peak_memory_bytes_est }
    /// Predicted **wall-clock** ms. Already scaled to
    /// [`ComputeEnvironment::cores`] when produced by
    /// [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in).
    #[must_use] pub fn wall_ms(&self) -> Option<u64> { self.wall_ms }
    /// Full-image intermediate buffers held simultaneously (input/output
    /// excluded). `None` when the planner can't determine it.
    #[must_use] pub fn intermediate_buffer_count(&self) -> Option<u32> { self.intermediate_buffer_count }
}

// Per-step calibration: ns/MP at 4096-pixel rows, AVX2/V3, Ryzen 9 7950X,
// from the 2026-04-23 bench suite under `zenpixels/benchmarks/`. See
// docs/ESTIMATE.md for the per-file list.

const ONE_MP: f64 = 1_048_576.0;
const GIB: f64 = 1_073_741_824.0;

/// `ns/MP = MP * bpp * 1e9 / (throughput_gib_s * GIB)`.
const fn gib_to_ns_per_mp(throughput_gib_s: f64, bytes_per_pixel: f64) -> f64 {
    bytes_per_pixel * ONE_MP * 1.0e9 / (throughput_gib_s * GIB)
}

/// Per-megapixel cost (ns) for a [`ConvertStep`], multiplied by
/// `pixels / 1 MP` for the runtime contribution. Bench values from the
/// 2026-04-23 suite (HDR tone-map: 2026-06-20); see `docs/ESTIMATE.md` for
/// per-file sources. `gib(g)` returns the GiB/s→ns/MP cost at the step's
/// source bpp; `bucketed` picks a per-bpp throughput with a fallback.
#[rustfmt::skip]
fn step_cost_ns_per_mp(step: &ConvertStep, current_bpp: usize) -> f64 {
    let bpp = current_bpp as f64;
    let gib = |g: f64| gib_to_ns_per_mp(g, bpp);
    let bucketed = |bs: &[(usize, f64)], fallback: f64| -> f64 {
        gib(bs.iter().copied().find_map(|(b, g)| (b == current_bpp).then_some(g)).unwrap_or(fallback))
    };
    match step {
        ConvertStep::Identity => 0.0,
        // Layout (t1).
        ConvertStep::SwizzleBgraRgba => bucketed(&[(4, 116.42)], 75.0),
        ConvertStep::RgbToBgra => gib(80.0),
        ConvertStep::AddAlpha => bucketed(&[(3, 125.06), (6, 40.59), (12, 104.01)], 30.0),
        ConvertStep::DropAlpha => bucketed(&[(4, 95.90), (8, 133.63), (16, 148.81)], 80.0),
        ConvertStep::MatteComposite { .. } => gib(5.0), // EOTF-blend-OETF, 3-8 GiB/s by TF
        ConvertStep::GrayToRgb => bucketed(&[(1, 12.85)], 60.0),
        ConvertStep::GrayToRgba => gib(8.6),
        ConvertStep::RgbToGray { .. } => gib(12.0),
        ConvertStep::RgbaToGray { .. } => gib(10.0),
        ConvertStep::GrayAlphaToRgba => bucketed(&[(2, 95.30), (4, 119.80), (8, 149.72)], 60.0),
        ConvertStep::GrayAlphaToRgb | ConvertStep::GrayAlphaToGray => gib(80.0),
        ConvertStep::GrayToGrayAlpha => gib(100.0),
        // Depth (t2). 4096-row, RGB.
        ConvertStep::U8ToU16 => gib(112.82),
        ConvertStep::U16ToU8 => gib(34.39),
        ConvertStep::NaiveU8ToF32 => gib(95.21),
        ConvertStep::NaiveF32ToU8 => gib(52.99),
        ConvertStep::U16ToF32 => gib(88.68),
        ConvertStep::F32ToU16 => gib(64.33),
        ConvertStep::F16ToF32 => gib(7.09),
        ConvertStep::F32ToF16 => gib(3.25),
        // Transfer functions (t3 fused, t4 f32). 4096-row, RGB.
        ConvertStep::SrgbU8ToLinearF32 => gib(24.26),
        ConvertStep::LinearF32ToSrgbU8 => gib(4.56),
        ConvertStep::PqU16ToLinearF32 => gib(2.68),
        ConvertStep::LinearF32ToPqU16 => gib(1.39),
        ConvertStep::HlgU16ToLinearF32 => gib(6.16),
        ConvertStep::LinearF32ToHlgU16 => gib(4.44),
        ConvertStep::PqF32ToLinearF32 => gib(3.0),
        ConvertStep::LinearF32ToPqF32 => gib(2.72),
        ConvertStep::HlgF32ToLinearF32 => gib(6.0),
        ConvertStep::LinearF32ToHlgF32 => gib(4.0),
        ConvertStep::SrgbF32ToLinearF32 | ConvertStep::SrgbF32ToLinearF32Extended => gib(24.95),
        ConvertStep::LinearF32ToSrgbF32 | ConvertStep::LinearF32ToSrgbF32Extended => gib(8.0),
        ConvertStep::Bt709F32ToLinearF32 | ConvertStep::Gamma22F32ToLinearF32 => gib(6.0),
        ConvertStep::LinearF32ToBt709F32 | ConvertStep::LinearF32ToGamma22F32 => gib(4.5),
        // Alpha mode (t5). f32 4-channel.
        ConvertStep::StraightToPremul => gib(13.74),
        ConvertStep::PremulToStraight => gib(7.51), // divide is slow
        // Oklab (t6). cbrt forward, cubed inverse.
        ConvertStep::LinearRgbToOklab => gib(1.61),
        ConvertStep::OklabToLinearRgb => gib(53.25),
        ConvertStep::LinearRgbaToOklaba => gib(2.14),
        ConvertStep::OklabaToLinearRgba => gib(58.91),
        // Gamut matrices (t7). 3×3 linear-F32 matmul.
        ConvertStep::GamutMatrixRgbF32(_) => gib(21.84),
        ConvertStep::GamutMatrixRgbaF32(_) => gib(20.0),
        ConvertStep::Fused { kind, .. } => match kind {
            FusedKind::SrgbU8GamutRgb => gib(3.79),
            FusedKind::SrgbU8GamutRgba => gib(3.5),
            FusedKind::SrgbU16GamutRgb => gib(5.84),
            FusedKind::SrgbU8ToLinearF32Rgb => gib(11.19),
            FusedKind::LinearF32ToSrgbU8Rgb => gib(3.11),
        },
        // HDR. Pluggable: the mapper reports its own per-MP cost via
        // `ToneMapper::cost_ns_per_mp` (default 4_000_000, BT.2446-A
        // 4_194_304 — empirically ~250 Mpix/s on RGB f32 linear-light,
        // Ryzen 9 7950X, AVX2, 2026-06-20). The default keeps the
        // scheduler conservative for unknown custom curves. Units
        // (ns / MP) match the rest of this table — `mapper.cost_ns_per_mp`
        // returns the same unit as `gib(...)` outputs.
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMap(step) => f64::from(step.mapper.cost_ns_per_mp()),
        // Hue-preserving rational knee in OKLch; ~3 GiB/s (cbrt-dominated).
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { .. } => gib(3.0),
    }
}

/// Every row-stride SIMD kernel is parallel. HDR steps (pluggable
/// `ToneMap`, OKLch soft-compress) read per-image scalars and are
/// scheduled serially. Bias for ambiguous future steps: SERIAL
/// (over-estimate wall time).
#[rustfmt::skip]
fn step_is_parallelizable(step: &ConvertStep) -> bool {
    #[cfg(feature = "hdr-experimental")]
    if matches!(step, ConvertStep::ToneMap(_) | ConvertStep::SoftCompressOklch { .. }) {
        return false;
    }
    let _ = step;
    true
}

/// Wall-time multiplier on the AVX2 (`X86V3`) calibration baseline. Coarse
/// "Δkernel" estimates pending a per-tier sweep. Absent hint + future
/// unhandled tiers fall back to baseline (1.0).
#[rustfmt::skip]
fn simd_tier_multiplier(tier: SimdTier) -> f64 {
    match tier {
        SimdTier::X86V4 => 0.85,                                                              // AVX-512
        SimdTier::X86V2 | SimdTier::X86V1 => 1.4,                                             // SSE2/SSE4.2
        SimdTier::Wasm128 => 1.3,
        SimdTier::Wasm => 2.0,
        SimdTier::X86V3 | SimdTier::Neon | SimdTier::Unknown | SimdTier::CurrentHost => 1.0,
    }
}

/// Walk the plan once, summing time and tracking peak + live intermediate-
/// buffer counts. Memory + threading model: `docs/ESTIMATE.md`.
//
// TODO: per-tier calibration tables once the t-series bench sweep re-runs on
// AVX-512 / SSE / NEON / WASM hosts.
#[rustfmt::skip]
pub(crate) fn estimate_plan(plan: &ConvertPlan, image: &ImageCharacteristics, compute: &ComputeEnvironment) -> ResourceEstimate {
    let (width, height) = (image.width(), image.height());
    let tier_mul = compute.simd_tier().map(simd_tier_multiplier).unwrap_or(1.0);
    let pixels = u64::from(width) * u64::from(height);
    let dst_bytes = pixels * plan.to().bytes_per_pixel() as u64;
    // Identity: memcpy-only ~30 GB/s midpoint, SERIAL, no scratch.
    if plan.is_identity() {
        let ms = (dst_bytes as f64) / (30.0 * GIB) * 1_000.0 * tier_mul;
        return finalize(dst_bytes, ms as u64, 1, 0, compute);
    }
    let pixels_mp = (pixels as f64) / ONE_MP;
    // Multi-step: 2 ping-pong row halves sized to the widest intermediate bpp.
    // Single-step: kernel writes src→dst directly (no scratch).
    let multi = plan.steps().len() > 1;
    let (mut max_bpp, mut desc) = (plan.from().bytes_per_pixel(), plan.from());
    // Bottleneck: SERIAL step → 1 thread; else min knee (rows/64 ∈ [1,16]).
    let knee = (u64::from(height) / 64).clamp(1, 16) as u32;
    let (mut total_time_ms, mut any_serial, mut min_knee) = (0.0_f64, false, u32::MAX);
    for step in plan.steps() {
        total_time_ms += step_cost_ns_per_mp(step, desc.bytes_per_pixel()) * pixels_mp / 1e6;
        if step_is_parallelizable(step) { min_knee = min_knee.min(knee); } else { any_serial = true; }
        desc = intermediate_after(desc, step);
        max_bpp = max_bpp.max(desc.bytes_per_pixel());
    }
    let scratch_bytes = if multi { (u64::from(width) * max_bpp as u64).saturating_mul(2) } else { 0 };
    let buffer_count: u32 = if multi { 2 } else { 0 };
    let bottleneck = if any_serial || min_knee == u32::MAX { 1 } else { min_knee };
    finalize(dst_bytes.saturating_add(scratch_bytes), (total_time_ms * tier_mul) as u64, bottleneck, buffer_count, compute)
}

/// Divide single-thread wall by `min(cores, bottleneck)` + attach buffer count.
#[rustfmt::skip]
fn finalize(peak: u64, wall_st: u64, bottleneck: u32, buffers: u32, compute: &ComputeEnvironment) -> ResourceEstimate {
    let eff = (compute.cores() as u64).max(1).min(bottleneck.max(1) as u64);
    ResourceEstimate::new(peak, wall_st / eff).with_intermediate_buffer_count(buffers)
}

/// Re-call of `crate::convert::intermediate_desc_for_estimate` to keep the
/// two helpers from drifting.
fn intermediate_after(current: PixelDescriptor, step: &ConvertStep) -> PixelDescriptor {
    crate::convert::intermediate_desc_for_estimate(current, step)
}

#[cfg(test)]
#[rustfmt::skip]
mod local_type_contract_tests {
    //! Mirrors `zencodec::estimate` contract tests so shape drift trips a
    //! unit test, not a downstream compile failure.
    use super::*;
    const D: PixelDescriptor = PixelDescriptor::RGB8_SRGB;

    #[test]
    fn compute_environment_builder_clamps_and_defaults() {
        assert_eq!(ComputeEnvironment::new().cores(), 1);
        assert_eq!(ComputeEnvironment::new().with_cores(0).cores(), 1);
        assert_eq!(ComputeEnvironment::default().with_cores(16).cores(), 16);
        let e = ComputeEnvironment::new().with_available_ram_bytes(1 << 30);
        assert_eq!(e.available_ram_bytes(), Some(1 << 30));
        assert_eq!(ComputeEnvironment::new().simd_tier(), None);
        let t = ComputeEnvironment::new().with_simd_tier(SimdTier::X86V3);
        assert_eq!(t.simd_tier(), Some(SimdTier::X86V3));
    }
    #[test]
    fn image_characteristics_fields() {
        let im = ImageCharacteristics::new(1024, 768, D);
        assert_eq!((im.width(), im.height(), *im.descriptor()), (1024, 768, D));
    }
    #[test]
    fn resource_estimate_new_unknown_and_buffer_count() {
        let est = ResourceEstimate::new(200, 1000);
        assert_eq!(est.peak_memory_bytes_est(), Some(200));
        assert_eq!(est.wall_ms(), Some(1000));
        assert_eq!(est.intermediate_buffer_count(), None);
        let u = ResourceEstimate::unknown();
        assert_eq!(u.peak_memory_bytes_est(), None);
        assert_eq!(u.wall_ms(), None);
        assert_eq!(u.intermediate_buffer_count(), None);
        let withbuf = ResourceEstimate::new(200, 1000).with_intermediate_buffer_count(2);
        assert_eq!(withbuf.intermediate_buffer_count(), Some(2));
    }
}
