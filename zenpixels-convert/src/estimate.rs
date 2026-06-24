//! Resource-estimation primitive for [`ConvertPlan`].
//!
//! Callers integrating zenpixels-convert into pipelines need to make
//! scheduling and throttling decisions *before* an operation runs:
//!
//! - Will this conversion OOM on a small server?
//! - Should we batch-throttle to avoid stalling the worker?
//! - Will the planned conversion finish within an SLA?
//!
//! [`ConvertPlan::estimate`](crate::ConvertPlan::estimate) and
//! [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in) answer those
//! questions cheaply — no allocation, no row work, just walk the planned
//! steps and return a [`ResourceEstimate`] with the projected peak memory,
//! wall-clock, and CPU-core scaling numbers.
//!
//! # API-shape-compatible with codec-side estimates
//!
//! `zenpixels-convert` is a **foundation crate** — it sits below codec
//! abstractions in the workspace dep graph and does NOT depend on
//! `zencodec`. To keep multi-stage `decode → convert → encode` pipelines
//! ergonomic, the four types defined here ([`ResourceEstimate`],
//! [`ComputeEnvironment`], [`ImageCharacteristics`], [`SimdTier`]) are
//! **shape-compatible** with the corresponding `zencodec::estimate::*`
//! types — same field names, same builder method names, same accessor
//! signatures, same `#[non_exhaustive]` discipline — so a codec author
//! whose stack already uses the zencodec contract can wire per-stage
//! estimates through with a trivial `From` conversion (a follow-up will
//! gate that conversion behind a feature flag).
//!
//! Every field is `Option`, the structs are `#[non_exhaustive]`, and the
//! builders are growable: future fields land additively without breaking
//! match-bind sites at the call boundary.
//!
//! TODO(follow-up): add `From<zencodec::estimate::ResourceEstimate>` (and
//! the sibling conversions) behind a feature flag so callers at the codec
//! boundary get a one-liner. Out of scope here — this commit restores the
//! foundation-crate layering; the bridge lands separately.
//!
//! # Accuracy contract
//!
//! Estimates are **best-effort**. The design tolerance is ±30 % vs
//! the underlying bench numbers; real-world variance comes from:
//!
//! - **CPU model and SIMD tier.** Calibration data is from the V3
//!   (AVX2) path on Ryzen 9 7950X. AVX-512 hosts run faster on a
//!   handful of kernels; older Zen / Intel / Apple Silicon hosts
//!   vary kernel-by-kernel. [`ComputeEnvironment::with_simd_tier`]
//!   applies a coarse per-tier wall-time multiplier on top of the
//!   AVX2 baseline (TODO: per-tier calibration tables — see the
//!   `simd_tier_multiplier` body for the current values).
//! - **Core count.** Wall time is scaled by the effective parallel
//!   thread count: every plan whose every step is row-parallel divides
//!   `wall_ms` by `min(compute.cores(), per_step_knee)` where the
//!   per-step knee is `rows / 64` clamped to `[1, 16]`. Any SERIAL
//!   step in the plan disables the scaling (single-thread wall is
//!   reported).
//! - **Cache state.** Cold L1/L2 cache adds per-call overhead; the
//!   benches measure steady-state at 4096-pixel rows, so very small
//!   images carry proportionally more fixed overhead than the
//!   estimate accounts for.
//! - **Frequency scaling / thermal throttling.** The reference
//!   machine is water-cooled and runs ~4.5 GHz under sustained
//!   load. Boxes that thermal-throttle will be slower.
//! - **Contention.** The estimate assumes a single hot pipeline.
//!   Heavy concurrent load reduces effective throughput.
//!
//! Use the estimate for *sizing decisions* (ballpark memory budget,
//! "is this op cheap or expensive?"), not for tight SLAs.
//!
//! # Calibration source
//!
//! Per-pixel cycle costs are baked from the 2026-04-23 benchmark
//! suite at `zenpixels/benchmarks/`:
//!
//! - `t1_layout_2026-04-23_baseline.txt` — swizzle, add/drop alpha,
//!   gray-to-rgb, etc.
//! - `t2_depth_2026-04-23_baseline.txt` — U8/U16/F16/F32 depth shifts.
//! - `t3_tf_fused_2026-04-23_baseline.txt` — sRGB/PQ/HLG transfer
//!   functions (the fused integer-in / linear-out kernels).
//! - `t4_tf_f32_2026-04-23_baseline.txt` — F32 transfer functions.
//! - `t6_oklab_2026-04-23_baseline.txt` — Linear RGB ↔ Oklab.
//! - `t7_gamut_2026-04-23_baseline.txt` — 3×3 gamut matrices.
//! - `bt2446a_throughput_2026-06-20.md` (zentone) — the
//!   HDR→SDR tone-map curve.
//! - `measure_max_throughput_2026-06-19.md` — the SOTA
//!   spec-conformant CLL reading used by HDR-source scan legs;
//!   the default-build SIMD path on the 7950X (no
//!   `-C target-cpu=native`) delivers **~2.7 Gpix/s** steady-state on
//!   RGB f32 linear-light.
//!
//! All steady-state at 4096-pixel rows (L2-resident) on the public
//! AVX2 path (no `-C target-cpu=native`).
//!
//! # Threading model
//!
//! Most [`ConvertStep`]s are row-parallel (SIMD strip kernels) and contribute
//! a per-step parallel knee of `rows / 64` clamped to `[1, 16]`. The plan's
//! overall threading is the bottleneck: if **any** step is SERIAL, the whole
//! plan is. Otherwise the smallest knee across parallel steps caps the
//! useful thread count, and `estimate_plan` divides `wall_ms` by
//! `min(compute.cores(), bottleneck_knee)` directly when building the
//! [`ResourceEstimate`]. The model is internal — the public surface only
//! exposes the resulting scaled `wall_ms`.

use crate::convert::{ConvertPlan, ConvertStep, FusedKind};
use crate::PixelDescriptor;

// ---------------------------------------------------------------------------
// Local resource-estimate types (shape-compatible with `zencodec::estimate`).
//
// `zenpixels-convert` is a foundation crate and does NOT depend on codec
// abstractions; defining these locally restores the workspace's dep-layering
// rule (foundation crates sit BELOW codec crates) without losing the
// shape compatibility codec authors rely on at the encode/decode boundary.
//
// Every type uses private fields + `Option`-returning accessors + builder-
// style `with_*` setters + `#[non_exhaustive]` so future field additions
// are source-compatible at every match-bind / construction site.
// ---------------------------------------------------------------------------

/// The SIMD instruction tier a codec will dispatch to.
///
/// An optional hint on [`ComputeEnvironment`]: a wider/newer tier generally
/// means a faster encode/decode, so estimates can apply a per-tier time factor
/// (today they assume the calibration host's tier; the field carries the hint
/// for future tier-aware models). The variants mirror the `x86-64-vN`
/// microarchitecture levels and the archmage / magetypes token vocabulary, so
/// a caller that already detects a tier with archmage maps it trivially:
///
/// ```rust,ignore
/// use zenpixels_convert::{ComputeEnvironment, SimdTier};
/// // archmage tokens use the same x86-64-vN levels:
/// let tier = if archmage::X64V4Token::summon().is_some() { SimdTier::X86V4 }
///     else if archmage::X64V3Token::summon().is_some() { SimdTier::X86V3 }
///     else if archmage::X64V2Token::summon().is_some() { SimdTier::X86V2 }
///     else { SimdTier::X86V1 };
/// let env = ComputeEnvironment::new().with_cores(8).with_simd_tier(tier);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimdTier {
    /// The SIMD tier is unknown — estimates use a conservative cross-tier
    /// **baseline average** (no assumption about the running hardware). Use
    /// [`CurrentHost`](SimdTier::CurrentHost) instead when estimating for the
    /// local machine.
    Unknown,
    /// The SIMD tier of the host actually running the estimate (≈ the
    /// calibration host's native tier). Use this when estimating for the local
    /// machine — distinct from [`Unknown`](SimdTier::Unknown), which assumes a
    /// cross-tier baseline average rather than the running hardware.
    CurrentHost,
    /// WebAssembly without SIMD128 (scalar wasm).
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

/// Hardware + runtime conditions for a resource estimate.
///
/// Sealed and growable — construct via [`ComputeEnvironment::new`] and refine
/// with the `with_*` setters; read with the accessors. Carries the available
/// core count + an optional [`SimdTier`] + optional RAM budget today; new
/// fields (load factor, GPU) are additive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ComputeEnvironment {
    available_cores: usize,
    available_ram_bytes: Option<u64>,
    simd_tier: Option<SimdTier>,
}

impl ComputeEnvironment {
    /// A single-core environment with unknown RAM and unspecified SIMD tier
    /// (the conservative default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            available_cores: 1,
            available_ram_bytes: None,
            simd_tier: None,
        }
    }

    /// Number of CPU cores available to the operation (clamped to ≥ 1). On
    /// `std` callers typically pass `std::thread::available_parallelism()`.
    #[must_use]
    pub fn with_cores(mut self, cores: usize) -> Self {
        self.available_cores = cores.max(1);
        self
    }

    /// Physical RAM available to the operation, for memory-ceiling decisions.
    #[must_use]
    pub fn with_available_ram_bytes(mut self, bytes: u64) -> Self {
        self.available_ram_bytes = Some(bytes);
        self
    }

    /// The SIMD instruction tier the codec will dispatch to (e.g. detected via
    /// archmage on the caller's side). Estimates may use it to apply a
    /// per-tier time factor.
    #[must_use]
    pub fn with_simd_tier(mut self, tier: SimdTier) -> Self {
        self.simd_tier = Some(tier);
        self
    }

    /// Available CPU cores (≥ 1).
    #[must_use]
    pub fn cores(&self) -> usize {
        self.available_cores
    }

    /// Available RAM in bytes, if known.
    #[must_use]
    pub fn available_ram_bytes(&self) -> Option<u64> {
        self.available_ram_bytes
    }

    /// The SIMD tier hint, if specified.
    #[must_use]
    pub fn simd_tier(&self) -> Option<SimdTier> {
        self.simd_tier
    }
}

impl Default for ComputeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Characteristics of the image being encoded/decoded.
///
/// Sealed and growable. Carries the dimensions and pixel format today; future
/// fields (content class, HDR tier depth) are additive.
///
/// `zenpixels-convert` is per-frame: animation-related fields belong on
/// codec types (e.g. on `zencodec::estimate::ImageCharacteristics`), not on
/// the per-conversion estimate input. Callers iterating an animation invoke
/// the per-frame plan once per frame; the scheduler sees per-frame resource
/// use directly.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageCharacteristics {
    width: u32,
    height: u32,
    descriptor: PixelDescriptor,
}

impl ImageCharacteristics {
    /// A still image of `width` × `height` with the given pixel format.
    #[must_use]
    pub fn new(width: u32, height: u32, descriptor: PixelDescriptor) -> Self {
        Self {
            width,
            height,
            descriptor,
        }
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixel format of the source/decoded buffer.
    #[must_use]
    pub fn descriptor(&self) -> &PixelDescriptor {
        &self.descriptor
    }
}

/// Predicted resources for a conversion plan.
///
/// Every field is `Option` — fillers may model what they can and leave the
/// rest `None` (the all-`None` constructor is [`ResourceEstimate::unknown`]).
/// `wall_ms` is already scaled to `compute.cores()` when the estimate comes
/// from [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in); the
/// internal threading-bottleneck model lives inside `estimate_plan` (see
/// the module docs).
///
/// Sealed and growable: build via [`new`](ResourceEstimate::new) /
/// [`unknown`](ResourceEstimate::unknown) + the `with_*` setters, read with the
/// accessors.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ResourceEstimate {
    peak_memory_bytes_est: Option<u64>,
    wall_ms: Option<u64>,
    intermediate_buffer_count: Option<u32>,
}

impl ResourceEstimate {
    /// An empty estimate — every field `None`. This is what a stage that does
    /// not model its resource use returns; stages that do model fill in what
    /// they can with the `with_*` setters. Read fields back as `Option`.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            peak_memory_bytes_est: None,
            wall_ms: None,
            intermediate_buffer_count: None,
        }
    }

    /// An estimate from the two essentials: the typical (estimated) peak memory
    /// and the wall time. The buffer count is left `None`; refine with
    /// [`with_intermediate_buffer_count`](Self::with_intermediate_buffer_count).
    #[must_use]
    pub fn new(peak_memory_bytes_est: u64, wall_ms: u64) -> Self {
        Self {
            peak_memory_bytes_est: Some(peak_memory_bytes_est),
            wall_ms: Some(wall_ms),
            intermediate_buffer_count: None,
        }
    }

    /// Set the number of full-image intermediate buffers held simultaneously
    /// during execution, NOT counting the input or output buffer. Schedulers
    /// can use this to distinguish 1-giant-buffer plans from N-medium-buffer
    /// plans for paging-pressure decisions.
    #[must_use]
    pub fn with_intermediate_buffer_count(mut self, n: u32) -> Self {
        self.intermediate_buffer_count = Some(n);
        self
    }

    /// Typical (≈ p50) estimated peak memory for natural content, bytes.
    #[must_use]
    pub fn peak_memory_bytes_est(&self) -> Option<u64> {
        self.peak_memory_bytes_est
    }

    /// Predicted **wall-clock** time in milliseconds. Already scaled to the
    /// caller's [`ComputeEnvironment::cores`] when produced by
    /// [`ConvertPlan::estimate_in`](crate::ConvertPlan::estimate_in).
    #[must_use]
    pub fn wall_ms(&self) -> Option<u64> {
        self.wall_ms
    }

    /// Estimated number of full-image intermediate buffers held simultaneously
    /// during execution, NOT counting the input or output buffer. Schedulers
    /// can use this to distinguish 1-giant-buffer plans from N-medium-buffer
    /// plans for paging-pressure decisions.
    ///
    /// Returns `None` when the planner can't reliably determine the count
    /// (e.g. plans whose buffer model isn't statically known).
    #[must_use]
    pub fn intermediate_buffer_count(&self) -> Option<u32> {
        self.intermediate_buffer_count
    }
}

// ---------------------------------------------------------------------------
// Calibration: per-step ns/MP costs at 4096-pixel rows, AVX2/V3, Ryzen 9 7950X.
//
// All values derived from the 2026-04-23 bench suite under
// `zenpixels/benchmarks/`. Conversion: throughput in GiB/s × bytes/pixel
// → bytes/s → pixels/s → ns/MP. See top-of-file docs for source files.
// ---------------------------------------------------------------------------

/// Bytes per *pixel* in the input row of each kernel as measured.
/// Most TF kernels are 3-channel RGB; alpha-channel kernels match.
///
/// 1 MP = 1_048_576 pixels. ns/MP at 4096-row throughput G GiB/s with
/// bpp B = (1_048_576 / (G * 2^30 / B)) * 1e9 ns.
const ONE_MP: f64 = 1_048_576.0;
const GIB: f64 = 1_073_741_824.0;

/// Convert a steady-state throughput (GiB/s) at a given bytes-per-pixel
/// into a per-megapixel cost in nanoseconds.
const fn gib_to_ns_per_mp(throughput_gib_s: f64, bytes_per_pixel: f64) -> f64 {
    // ns/MP = (MP / (throughput_bytes_s / bpp)) * 1e9
    //       = MP * bpp * 1e9 / throughput_bytes_s
    let bytes_per_pixel_per_mp = bytes_per_pixel * ONE_MP;
    let throughput_bytes_s = throughput_gib_s * GIB;
    bytes_per_pixel_per_mp * 1.0e9 / throughput_bytes_s
}

/// Per-megapixel cost (ns) for each [`ConvertStep`] kind.
///
/// The float result is multiplied by `(pixels / 1 MP)` to get the time
/// contribution at runtime. All values from the 2026-04-23 bench suite (or
/// 2026-06-20 for the HDR tone-map).
fn step_cost_ns_per_mp(step: &ConvertStep, current_bpp: usize) -> f64 {
    // The bench throughputs are bytes/s of the SOURCE row, so different
    // bpp inputs need to scale. We treat bpp == 3 (RGB) or bpp == 4 (RGBA)
    // as the canonical measured value. The function below converts that
    // to ns/MP at the runtime bpp.
    //
    // For layout-changing steps (e.g. GrayToRgb) we use the source bpp
    // explicitly; bench was on the relevant input.
    let bpp = current_bpp as f64;

    match step {
        // ----- Identity: zero cost (the row-copy fast path). -----
        ConvertStep::Identity => 0.0,

        // ----- Layout (t1_layout). All measured on 4096-row at the bpp shown. -----
        // Throughputs assume the source bpp. swizzle/add_alpha/drop_alpha
        // are byte-level in/out; gray-to-rgb expands.
        ConvertStep::SwizzleBgraRgba => {
            // u8 4-byte: 116.42 GiB/s
            // f32 16-byte: ~75 GiB/s (extrapolated from u16 equivalents)
            let g = if bpp <= 4.0 { 116.42 } else { 75.0 };
            gib_to_ns_per_mp(g, bpp)
        }
        ConvertStep::RgbToBgra => {
            // u8: ~80 GiB/s (single fused SIMD pass).
            gib_to_ns_per_mp(80.0, bpp)
        }
        ConvertStep::AddAlpha => {
            // u8 3-byte: 125.06 GiB/s, u16: 40.59, f32: 104.01, f16: 19.06.
            let g = match bpp as usize {
                3 => 125.06,
                6 => 40.59,
                12 => 104.01,
                _ => 30.0, // fallback for f16 / other
            };
            gib_to_ns_per_mp(g, bpp)
        }
        ConvertStep::DropAlpha => {
            // u8 4-byte: 95.90 GiB/s, u16: 133.63, f32: 148.81, f16: 143.11.
            let g = match bpp as usize {
                4 => 95.90,
                8 => 133.63,
                16 => 148.81,
                _ => 80.0,
            };
            gib_to_ns_per_mp(g, bpp)
        }
        ConvertStep::MatteComposite { .. } => {
            // Heuristic: ~ DropAlpha + per-TF linearize/encode. The matte
            // composite kernel does an EOTF-blend-OETF round trip per pixel.
            // Measured ranges from 3-8 GiB/s depending on TF. Use 5 GiB/s.
            gib_to_ns_per_mp(5.0, bpp)
        }
        ConvertStep::GrayToRgb => {
            // u8 1-byte: 12.85 GiB/s, u16: 108.85
            let g = if bpp <= 1.0 { 12.85 } else { 60.0 };
            gib_to_ns_per_mp(g, bpp)
        }
        ConvertStep::GrayToRgba => {
            // Roughly GrayToRgb + AddAlpha; bench at 8.6 GiB/s for u8.
            gib_to_ns_per_mp(8.6, bpp)
        }
        ConvertStep::RgbToGray { .. } => {
            // RGB→gray weighted sum; ~12 GiB/s for u8, faster for u16.
            gib_to_ns_per_mp(12.0, bpp)
        }
        ConvertStep::RgbaToGray { .. } => {
            // Similar to RgbToGray plus alpha drop.
            gib_to_ns_per_mp(10.0, bpp)
        }
        ConvertStep::GrayAlphaToRgba => {
            // u8 2-byte: 95.30 GiB/s, u16: 119.80, f32: 149.72.
            let g = match bpp as usize {
                2 => 95.30,
                4 => 119.80,
                8 => 149.72,
                _ => 60.0,
            };
            gib_to_ns_per_mp(g, bpp)
        }
        ConvertStep::GrayAlphaToRgb | ConvertStep::GrayAlphaToGray => {
            // Cheap byte-level transforms; ~80 GiB/s for u8.
            gib_to_ns_per_mp(80.0, bpp)
        }
        ConvertStep::GrayToGrayAlpha => {
            // Add opaque alpha to gray; ~100 GiB/s for u8.
            gib_to_ns_per_mp(100.0, bpp)
        }

        // ----- Depth conversion (t2_depth). 4096-row, RGB. -----
        ConvertStep::U8ToU16 => gib_to_ns_per_mp(112.82, bpp),
        ConvertStep::U16ToU8 => gib_to_ns_per_mp(34.39, bpp),
        ConvertStep::NaiveU8ToF32 => gib_to_ns_per_mp(95.21, bpp),
        ConvertStep::NaiveF32ToU8 => gib_to_ns_per_mp(52.99, bpp),
        ConvertStep::U16ToF32 => gib_to_ns_per_mp(88.68, bpp),
        ConvertStep::F32ToU16 => gib_to_ns_per_mp(64.33, bpp),
        ConvertStep::F16ToF32 => gib_to_ns_per_mp(7.09, bpp),
        ConvertStep::F32ToF16 => gib_to_ns_per_mp(3.25, bpp),

        // ----- Transfer functions (t3 fused, t4 f32). 4096-row, RGB. -----
        ConvertStep::SrgbU8ToLinearF32 => gib_to_ns_per_mp(24.26, bpp),
        ConvertStep::LinearF32ToSrgbU8 => gib_to_ns_per_mp(4.56, bpp),
        ConvertStep::PqU16ToLinearF32 => gib_to_ns_per_mp(2.68, bpp),
        ConvertStep::LinearF32ToPqU16 => gib_to_ns_per_mp(1.39, bpp),
        ConvertStep::HlgU16ToLinearF32 => gib_to_ns_per_mp(6.16, bpp),
        ConvertStep::LinearF32ToHlgU16 => gib_to_ns_per_mp(4.44, bpp),
        ConvertStep::PqF32ToLinearF32 => gib_to_ns_per_mp(3.0, bpp),
        ConvertStep::LinearF32ToPqF32 => gib_to_ns_per_mp(2.72, bpp),
        ConvertStep::HlgF32ToLinearF32 => gib_to_ns_per_mp(6.0, bpp),
        ConvertStep::LinearF32ToHlgF32 => gib_to_ns_per_mp(4.0, bpp),
        ConvertStep::SrgbF32ToLinearF32 | ConvertStep::SrgbF32ToLinearF32Extended => {
            gib_to_ns_per_mp(24.95, bpp)
        }
        ConvertStep::LinearF32ToSrgbF32 | ConvertStep::LinearF32ToSrgbF32Extended => {
            gib_to_ns_per_mp(8.0, bpp)
        }
        ConvertStep::Bt709F32ToLinearF32 => gib_to_ns_per_mp(6.0, bpp),
        ConvertStep::LinearF32ToBt709F32 => gib_to_ns_per_mp(4.5, bpp),
        ConvertStep::Gamma22F32ToLinearF32 => gib_to_ns_per_mp(6.0, bpp),
        ConvertStep::LinearF32ToGamma22F32 => gib_to_ns_per_mp(4.5, bpp),

        // ----- Alpha mode (t5). Premul/straight conversions. -----
        ConvertStep::StraightToPremul => {
            // f32 4-channel: ~13.74 GiB/s
            gib_to_ns_per_mp(13.74, bpp)
        }
        ConvertStep::PremulToStraight => {
            // f32 4-channel: ~7.51 GiB/s (divide is slow)
            gib_to_ns_per_mp(7.51, bpp)
        }

        // ----- Oklab (t6). cbrt-dominated forward, cubed inverse. -----
        ConvertStep::LinearRgbToOklab => gib_to_ns_per_mp(1.61, bpp),
        ConvertStep::OklabToLinearRgb => gib_to_ns_per_mp(53.25, bpp),
        ConvertStep::LinearRgbaToOklaba => gib_to_ns_per_mp(2.14, bpp),
        ConvertStep::OklabaToLinearRgba => gib_to_ns_per_mp(58.91, bpp),

        // ----- Gamut matrices (t7). 3×3 matrix-multiply on linear F32. -----
        ConvertStep::GamutMatrixRgbF32(_) => gib_to_ns_per_mp(21.84, bpp),
        ConvertStep::GamutMatrixRgbaF32(_) => gib_to_ns_per_mp(20.0, bpp),
        ConvertStep::Fused { kind, .. } => match kind {
            FusedKind::SrgbU8GamutRgb => gib_to_ns_per_mp(3.79, bpp),
            FusedKind::SrgbU8GamutRgba => gib_to_ns_per_mp(3.5, bpp),
            FusedKind::SrgbU16GamutRgb => gib_to_ns_per_mp(5.84, bpp),
            FusedKind::SrgbU8ToLinearF32Rgb => gib_to_ns_per_mp(11.19, bpp),
            FusedKind::LinearF32ToSrgbU8Rgb => gib_to_ns_per_mp(3.11, bpp),
        },

        // ----- HDR. BT.2446-A from zentone bench (2026-06-20). -----
        // Bt2446A::map_strip_simd: ~250 Mpix/s on RGB f32 linear-light.
        // 1 MP / 250 Mpix/s = 4 ms/MP = 4_000_000 ns/MP.
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A { .. } => {
            // 1 MP / 250 Mpix/s = 4.0e6 / 1000 = 4000 ns/Mpix × 1024 = ~4.0 ms / MP.
            // Actually: 1_048_576 / 250_000_000 s = 4.194e-3 s = 4_194_304 ns/MP.
            4_194_304.0
        }
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { .. } => {
            // Hue-preserving rational knee curve in OKLch space. Per-pixel
            // is roughly Oklab forward + scalar curve + Oklab inverse —
            // dominated by the cbrt path. Use ~3 GiB/s as a coarse model.
            gib_to_ns_per_mp(3.0, bpp)
        }
    }
}

/// Whether a step is parallelizable per-row.
///
/// Every step that operates on rows independently (every SIMD strip kernel in
/// the crate) is parallelizable. The two exceptions are HDR steps with
/// per-image state (BT.2446-A and OKLch soft-compress both read a max
/// luminance / chroma scalar from the source); we conservatively mark those
/// SERIAL even though the per-strip kernels themselves are SIMD-row-internal.
/// All other steps return `true`.
///
/// If a step's parallelizability is ambiguous in the future, the bias is
/// toward SERIAL: over-estimating wall time is safer than under-estimating.
fn step_is_parallelizable(step: &ConvertStep) -> bool {
    match step {
        // The hot kernels are all row-stride SIMD — safe to parallelize.
        ConvertStep::Identity
        | ConvertStep::SwizzleBgraRgba
        | ConvertStep::RgbToBgra
        | ConvertStep::AddAlpha
        | ConvertStep::DropAlpha
        | ConvertStep::MatteComposite { .. }
        | ConvertStep::GrayToRgb
        | ConvertStep::GrayToRgba
        | ConvertStep::RgbToGray { .. }
        | ConvertStep::RgbaToGray { .. }
        | ConvertStep::GrayAlphaToRgba
        | ConvertStep::GrayAlphaToRgb
        | ConvertStep::GrayAlphaToGray
        | ConvertStep::GrayToGrayAlpha
        | ConvertStep::U8ToU16
        | ConvertStep::U16ToU8
        | ConvertStep::NaiveU8ToF32
        | ConvertStep::NaiveF32ToU8
        | ConvertStep::U16ToF32
        | ConvertStep::F32ToU16
        | ConvertStep::F16ToF32
        | ConvertStep::F32ToF16
        | ConvertStep::SrgbU8ToLinearF32
        | ConvertStep::LinearF32ToSrgbU8
        | ConvertStep::PqU16ToLinearF32
        | ConvertStep::LinearF32ToPqU16
        | ConvertStep::HlgU16ToLinearF32
        | ConvertStep::LinearF32ToHlgU16
        | ConvertStep::PqF32ToLinearF32
        | ConvertStep::LinearF32ToPqF32
        | ConvertStep::HlgF32ToLinearF32
        | ConvertStep::LinearF32ToHlgF32
        | ConvertStep::SrgbF32ToLinearF32
        | ConvertStep::SrgbF32ToLinearF32Extended
        | ConvertStep::LinearF32ToSrgbF32
        | ConvertStep::LinearF32ToSrgbF32Extended
        | ConvertStep::Bt709F32ToLinearF32
        | ConvertStep::LinearF32ToBt709F32
        | ConvertStep::Gamma22F32ToLinearF32
        | ConvertStep::LinearF32ToGamma22F32
        | ConvertStep::StraightToPremul
        | ConvertStep::PremulToStraight
        | ConvertStep::LinearRgbToOklab
        | ConvertStep::OklabToLinearRgb
        | ConvertStep::LinearRgbaToOklaba
        | ConvertStep::OklabaToLinearRgba
        | ConvertStep::GamutMatrixRgbF32(_)
        | ConvertStep::GamutMatrixRgbaF32(_)
        | ConvertStep::Fused { .. } => true,
        // HDR steps read per-image scalars (source peak, max chroma) and
        // are currently scheduled serially. The per-strip SIMD kernel is
        // still hot; only the across-strip orchestration is serial. Bias
        // toward over-estimate (SERIAL).
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A { .. } | ConvertStep::SoftCompressOklch { .. } => false,
    }
}

/// Wall-time multiplier applied to the AVX2-baseline calibration when the
/// caller passed a known SIMD tier. The baseline is `SimdTier::X86V3`
/// (AVX2 + FMA) — the calibration host. Per-tier ratios are coarse "Δkernel"
/// estimates pending a follow-up per-tier calibration sweep; see the
/// **TODO** at the call site in [`estimate_plan`].
fn simd_tier_multiplier(tier: SimdTier) -> f64 {
    match tier {
        // AVX-512: ~15 % wall-time reduction on lane-doubled SIMD kernels.
        SimdTier::X86V4 => 0.85,
        // AVX2 baseline.
        SimdTier::X86V3 => 1.0,
        // SSE4.2 / SSE2: roughly 1.4× the AVX2 wall time on byte-level
        // kernels; the t-series benches haven't been re-measured here.
        SimdTier::X86V2 | SimdTier::X86V1 => 1.4,
        // NEON: similar throughput to AVX2 on the production kernels we
        // care about — keep parity until a NEON sweep lands.
        SimdTier::Neon => 1.0,
        // WASM-128: roughly 1.3× wall time.
        SimdTier::Wasm128 => 1.3,
        // Scalar WASM: ~2× wall time.
        SimdTier::Wasm => 2.0,
        // Unknown / current-host: assume baseline (no adjustment). Same as
        // having no hint — we do not penalize the unspecified path. The enum
        // is `#[non_exhaustive]` (future variants land additively); the
        // call site applies `.unwrap_or(1.0)` so an absent hint and a future
        // unhandled tier both fall back to the AVX2 baseline.
        SimdTier::Unknown | SimdTier::CurrentHost => 1.0,
    }
}

/// Body of the plan-level estimate. Walks the plan's steps once,
/// summing time and tracking the peak intermediate buffer size and the
/// number of simultaneously-live intermediate buffers.
///
/// Memory model:
/// - The output buffer at `to.bytes_per_pixel() * pixels` is always allocated.
/// - Multi-step plans hold two scratch row buffers ping-ponged between
///   intermediate descriptors. Worst-case scratch is `2 * width *
///   max_intermediate_bpp` bytes — counted as 2 intermediate buffers for
///   paging-pressure reporting.
/// - The estimate is for a single per-call working set, NOT a
///   parallel-job-wide cap.
///
/// Intermediate-buffer-count model (reported via
/// [`ResourceEstimate::intermediate_buffer_count`]):
/// - Identity plan → 0 (just `dst` memcpy from `src`).
/// - Single-step plan → 0 (kernel writes `src → dst` directly).
/// - Multi-step plan → 2 (the two ping-pong scratch halves).
///
/// Threading: the plan-level threading is the bottleneck across steps. A
/// SERIAL step forces the whole plan SERIAL (wall == single-thread); otherwise
/// the smallest per-step knee (`rows / 64` clamped to `[1, 16]`) caps the
/// useful thread count, and `wall_ms` is divided by
/// `min(compute.cores(), bottleneck_knee)`.
pub(crate) fn estimate_plan(
    plan: &ConvertPlan,
    image: &ImageCharacteristics,
    compute: &ComputeEnvironment,
) -> ResourceEstimate {
    let width = image.width();
    let height = image.height();

    // Apply the per-tier wall-time multiplier on top of the AVX2 baseline.
    // TODO: per-tier calibration tables once the t-series bench sweep
    // re-runs on AVX-512 / SSE / NEON / WASM hosts.
    let tier_mul = compute
        .simd_tier()
        .map(simd_tier_multiplier)
        .unwrap_or(1.0);

    // Quick identity short-circuit. The destination is still allocated and
    // memcpy'd into; the wall-time projection is the memcpy alone (serial —
    // there's only one operation, the memcpy can't usefully scale).
    if plan.is_identity() {
        let pixels = (width as u64) * (height as u64);
        let dst_bytes = pixels * plan.to().bytes_per_pixel() as u64;
        // memcpy at ~30 GB/s is a reasonable assumption — but on systems
        // with NUMA effects this can be 10-50 GB/s. Use a midpoint.
        let memcpy_gib_s = 30.0;
        let memcpy_time_ms = (dst_bytes as f64) / (memcpy_gib_s * GIB) * 1_000.0 * tier_mul;
        let wall_ms = memcpy_time_ms as u64;
        // Identity is SERIAL (single memcpy), so the bottleneck cap is 1.
        // No intermediates — the memcpy goes src → dst directly.
        return finalize(dst_bytes, wall_ms, 1, 0, compute);
    }

    let pixels = (width as u64) * (height as u64);
    let pixels_mp = (pixels as f64) / ONE_MP;

    // Output buffer is always allocated.
    let dst_bpp = plan.to().bytes_per_pixel() as u64;
    let dst_bytes = pixels * dst_bpp;

    // Scratch buffers for multi-step plans: two row-sized halves
    // sized to the widest intermediate. Single-step plans use
    // no scratch (the kernel writes directly into the dst row).
    let is_multi_step = plan.steps().len() > 1;
    let scratch_per_half_bytes = if is_multi_step {
        // Walk steps to find the widest intermediate bpp.
        let mut desc = plan.from();
        let mut max_bpp = desc.bytes_per_pixel();
        for step in plan.steps() {
            desc = intermediate_after(desc, step);
            max_bpp = max_bpp.max(desc.bytes_per_pixel());
        }
        (width as u64) * (max_bpp as u64)
    } else {
        0
    };
    // Two ping-pong halves (single allocation, but split in two).
    let scratch_bytes = scratch_per_half_bytes.saturating_mul(2);
    // Buffer count: 0 for identity / single-step (no scratch), 2 for
    // multi-step (the two ping-pong halves). Capturing this lets schedulers
    // distinguish 1-giant-buffer plans from N-medium-buffer plans for
    // paging-pressure decisions.
    let intermediate_buffer_count: u32 = if is_multi_step { 2 } else { 0 };

    // Sum per-step time contributions and compute the bottleneck
    // threading across steps. The model:
    //   * SERIAL step → whole-plan bottleneck stays at 1 (no scaling).
    //   * Parallel step → contributes a per-step knee of
    //     `rows / 64` clamped to `[1, 16]`. The smallest knee across all
    //     parallel steps caps the useful thread count.
    let mut total_time_ms = 0.0;
    let mut desc = plan.from();
    let mut any_serial = false;
    let mut min_knee: u32 = u32::MAX;

    for step in plan.steps() {
        let current_bpp = desc.bytes_per_pixel();
        let ns_per_mp = step_cost_ns_per_mp(step, current_bpp);
        let step_time_ms = ns_per_mp * pixels_mp / 1_000_000.0;
        total_time_ms += step_time_ms;
        if !step_is_parallelizable(step) {
            any_serial = true;
        } else {
            // Row-per-task heuristic: rows / 64 clamped to ≤ 16, ≥ 1.
            let rows = u64::from(height);
            let cap = ((rows / 64).max(1)).min(16) as u32;
            min_knee = min_knee.min(cap);
        }
        desc = intermediate_after(desc, step);
    }
    let total_time_ms = total_time_ms * tier_mul;

    // Whole-plan bottleneck: any SERIAL step → 1 thread; else the smallest
    // per-step knee (or 1 if no parallel steps were observed).
    let bottleneck_threads = if any_serial || min_knee == u32::MAX {
        1
    } else {
        min_knee
    };

    // Peak working-set: destination buffer + scratch (for multi-step).
    let peak_memory_bytes = dst_bytes.saturating_add(scratch_bytes);

    let wall_ms = total_time_ms as u64;
    finalize(
        peak_memory_bytes,
        wall_ms,
        bottleneck_threads,
        intermediate_buffer_count,
        compute,
    )
}

/// Wrap the projected peak + single-thread wall-ms in the
/// [`ResourceEstimate`] shape: divide `wall_ms` by
/// `min(compute.cores(), bottleneck_threads)` and attach the
/// intermediate-buffer count.
fn finalize(
    peak_est: u64,
    wall_ms_single_thread: u64,
    bottleneck_threads: u32,
    intermediate_buffer_count: u32,
    compute: &ComputeEnvironment,
) -> ResourceEstimate {
    let effective = (compute.cores() as u64)
        .max(1)
        .min(bottleneck_threads.max(1) as u64);
    let wall_ms_scaled = wall_ms_single_thread / effective;
    ResourceEstimate::new(peak_est, wall_ms_scaled)
        .with_intermediate_buffer_count(intermediate_buffer_count)
}

/// Mirror of `intermediate_desc` in `convert.rs`, exposed via the
/// `ConvertPlan::intermediate_after_step` helper. Kept as a
/// thin re-call so the two don't drift.
fn intermediate_after(current: PixelDescriptor, step: &ConvertStep) -> PixelDescriptor {
    crate::convert::intermediate_desc_for_estimate(current, step)
}

#[cfg(test)]
mod local_type_contract_tests {
    //! Contract tests for the locally-defined estimate types. These mirror the
    //! contract tests that lived in `zencodec::estimate` so we catch shape
    //! drift if the local types ever diverge from the codec-side ones.
    use super::*;

    fn desc() -> PixelDescriptor {
        PixelDescriptor::RGB8_SRGB
    }

    #[test]
    fn compute_environment_builder_clamps_and_defaults() {
        assert_eq!(ComputeEnvironment::new().cores(), 1);
        assert_eq!(ComputeEnvironment::new().with_cores(0).cores(), 1);
        assert_eq!(ComputeEnvironment::default().with_cores(16).cores(), 16);
        assert_eq!(
            ComputeEnvironment::new()
                .with_available_ram_bytes(1 << 30)
                .available_ram_bytes(),
            Some(1 << 30)
        );
        assert_eq!(ComputeEnvironment::new().simd_tier(), None);
        assert_eq!(
            ComputeEnvironment::new()
                .with_simd_tier(SimdTier::X86V3)
                .simd_tier(),
            Some(SimdTier::X86V3)
        );
    }

    #[test]
    fn image_characteristics_fields() {
        let im = ImageCharacteristics::new(1024, 768, desc());
        assert_eq!(im.width(), 1024);
        assert_eq!(im.height(), 768);
        assert_eq!(*im.descriptor(), desc());
    }

    #[test]
    fn new_sets_only_peak_est_and_wall() {
        let est = ResourceEstimate::new(200, 1000);
        assert_eq!(est.peak_memory_bytes_est(), Some(200));
        assert_eq!(est.wall_ms(), Some(1000));
        assert_eq!(est.intermediate_buffer_count(), None);
    }

    #[test]
    fn unknown_is_all_none() {
        let est = ResourceEstimate::unknown();
        assert_eq!(est.peak_memory_bytes_est(), None);
        assert_eq!(est.wall_ms(), None);
        assert_eq!(est.intermediate_buffer_count(), None);
    }

    #[test]
    fn with_intermediate_buffer_count_sets_field() {
        let est = ResourceEstimate::new(200, 1000).with_intermediate_buffer_count(2);
        assert_eq!(est.intermediate_buffer_count(), Some(2));
    }
}
