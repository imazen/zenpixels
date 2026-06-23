//! Resource-estimation API for [`ConvertPlan`].
//!
//! Callers integrating zenpixels-convert into pipelines need to make
//! scheduling and throttling decisions *before* an operation runs:
//!
//! - Will this conversion OOM on a small server?
//! - Should we batch-throttle to avoid stalling the worker?
//! - Will the planned conversion finish within an SLA?
//!
//! [`ConvertPlan::estimate_resources`] answers those questions cheaply
//! (no allocation, no row work — just walks the planned steps). The
//! returned [`ResourceEstimate`] reports a **peak memory upper bound**
//! and a **median wall-clock projection** on the reference machine
//! (AMD Ryzen 9 7950X, AVX2/V3 tier).
//!
//! # Accuracy contract
//!
//! Estimates are **best-effort**. The design tolerance is ±30 % vs
//! the underlying bench numbers; real-world variance comes from:
//!
//! - **CPU model and SIMD tier.** Calibration data is from the V3
//!   (AVX2) path on Ryzen 9 7950X. AVX-512 hosts run faster on a
//!   handful of kernels; older Zen / Intel / Apple Silicon hosts
//!   vary kernel-by-kernel.
//! - **Cache state.** Cold L1/L2 cache adds per-call overhead; the
//!   benches measure steady-state at 4096-pixel rows, so very small
//!   images carry proportionally more fixed overhead than the
//!   estimate accounts for. See `EstimateConfidence::Heuristic`.
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
//!
//! All steady-state at 4096-pixel rows (L2-resident) on the public
//! AVX2 path (no `-C target-cpu=native`).

use alloc::vec::Vec;

use crate::convert::{ConvertPlan, ConvertStep};
use crate::PixelDescriptor;

/// Peak resource cost projection for executing a [`ConvertPlan`].
///
/// Returned by [`ConvertPlan::estimate_resources`] and
/// [`PixelBufferConvertExt::estimate_convert_to`]
/// (see [`crate::ext::PixelBufferConvertExt`]). All values are
/// best-effort projections — see the module-level
/// [accuracy contract](crate::estimate#accuracy-contract).
///
/// # Memory
///
/// `peak_memory_bytes` is an **upper bound** on the working-set
/// allocations required to execute the plan, NOT including the
/// caller's persistent state. It covers:
///
/// 1. The **destination buffer** (always allocated — the plan
///    writes into a fresh buffer rather than mutating input).
/// 2. **Scratch buffers** for multi-step plans: the converter
///    holds two row-sized intermediate buffers (ping-pong) sized
///    to the widest intermediate format the plan passes through.
///
/// The number does *not* model rayon thread-local scratch or
/// allocator overhead — it's the per-call working-set ceiling.
///
/// # Time
///
/// `wall_time_ms` is a **median projection** on the reference
/// machine (AMD Ryzen 9 7950X, AVX2/V3 tier, no contention).
/// Real-world variance can exceed ±30 % from the benches that
/// fed the calibration — see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceEstimate {
    /// Peak working-set memory in bytes. Upper bound; includes
    /// input + intermediate + output buffers. Does **not** include
    /// the caller's persistent state.
    pub peak_memory_bytes: u64,

    /// Wall-clock projection in milliseconds, median on the
    /// reference machine.
    pub wall_time_ms: f64,

    /// Per-step breakdown for diagnostics. Empty for trivial
    /// (identity) plans.
    pub breakdown: Vec<StepEstimate>,

    /// Calibration confidence — see [`EstimateConfidence`].
    pub confidence: EstimateConfidence,
}

/// Per-step contribution to a [`ResourceEstimate`].
///
/// The breakdown lets callers see which step dominates so they
/// can target optimization or routing changes ("this op is 90 %
/// linear-f32-to-srgb-u8 — consider a different output format").
#[derive(Debug, Clone, PartialEq)]
pub struct StepEstimate {
    /// Static step name — e.g. `"SrgbU8ToLinearF32"`,
    /// `"ToneMapBt2446A"`, `"GamutMatrixRgbF32"`.
    pub name: &'static str,

    /// Memory contribution at this step in bytes. The plan
    /// peaks at `max(running_total_after_each_step)`, NOT the
    /// sum — see the module docs.
    pub memory_bytes: u64,

    /// Time contribution in milliseconds.
    pub time_ms: f64,
}

/// How confident the estimate is in its calibration.
///
/// Used for sanity-checking and reporting. Always prefer the
/// numeric estimate over the confidence value for sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateConfidence {
    /// Every step in the plan has calibrated per-pixel data from
    /// the 2026-04-23 / 2026-06-20 bench suites.
    Calibrated,
    /// One or more steps fell back to a generic per-pixel cost
    /// (no exact bench available). The total may be ±50 % off for
    /// those steps.
    Heuristic,
    /// The estimate could not be computed (descriptor unknown or
    /// plan rejected). All fields are zero.
    Unknown,
}

impl ResourceEstimate {
    /// Zero estimate. Used by the trait-side fallback when a plan
    /// cannot be built (descriptor incompatibility, CMYK input, etc.).
    ///
    /// `confidence` is typically [`EstimateConfidence::Unknown`].
    #[must_use]
    pub fn zero(confidence: EstimateConfidence) -> Self {
        Self {
            peak_memory_bytes: 0,
            wall_time_ms: 0.0,
            breakdown: Vec::new(),
            confidence,
        }
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
/// Returned by [`step_cost_ns_per_mp`]. The float result is multiplied
/// by `(pixels / 1 MP)` to get the time contribution at runtime.
///
/// All values from the 2026-04-23 bench suite (or 2026-06-20 for the
/// HDR tone-map). The function returns a (`ns_per_mp`, `calibrated`)
/// pair — `calibrated == false` flips the plan's overall confidence to
/// [`EstimateConfidence::Heuristic`].
fn step_cost_ns_per_mp(step: &ConvertStep, current_bpp: usize) -> (f64, bool) {
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
        ConvertStep::Identity => (0.0, true),

        // ----- Layout (t1_layout). All measured on 4096-row at the bpp shown. -----
        // Throughputs assume the source bpp. swizzle/add_alpha/drop_alpha
        // are byte-level in/out; gray-to-rgb expands.
        ConvertStep::SwizzleBgraRgba => {
            // u8 4-byte: 116.42 GiB/s
            // f32 16-byte: ~75 GiB/s (extrapolated from u16 equivalents)
            let g = if bpp <= 4.0 { 116.42 } else { 75.0 };
            (gib_to_ns_per_mp(g, bpp), bpp <= 4.0)
        }
        ConvertStep::RgbToBgra => {
            // u8: ~80 GiB/s (single fused SIMD pass).
            (gib_to_ns_per_mp(80.0, bpp), bpp == 3.0)
        }
        ConvertStep::AddAlpha => {
            // u8 3-byte: 125.06 GiB/s, u16: 40.59, f32: 104.01, f16: 19.06.
            let g = match bpp as usize {
                3 => 125.06,
                6 => 40.59,
                12 => 104.01,
                _ => 30.0, // fallback for f16 / other
            };
            (
                gib_to_ns_per_mp(g, bpp),
                matches!(bpp as usize, 3 | 6 | 12),
            )
        }
        ConvertStep::DropAlpha => {
            // u8 4-byte: 95.90 GiB/s, u16: 133.63, f32: 148.81, f16: 143.11.
            let g = match bpp as usize {
                4 => 95.90,
                8 => 133.63,
                16 => 148.81,
                _ => 80.0,
            };
            (
                gib_to_ns_per_mp(g, bpp),
                matches!(bpp as usize, 4 | 8 | 16),
            )
        }
        ConvertStep::MatteComposite { .. } => {
            // Heuristic: ~ DropAlpha + per-TF linearize/encode. The matte
            // composite kernel does an EOTF-blend-OETF round trip per pixel.
            // Measured ranges from 3-8 GiB/s depending on TF. Use 5 GiB/s.
            (gib_to_ns_per_mp(5.0, bpp), false)
        }
        ConvertStep::GrayToRgb => {
            // u8 1-byte: 12.85 GiB/s, u16: 108.85
            let g = if bpp <= 1.0 { 12.85 } else { 60.0 };
            (gib_to_ns_per_mp(g, bpp), bpp <= 2.0)
        }
        ConvertStep::GrayToRgba => {
            // Roughly GrayToRgb + AddAlpha; bench at 8.6 GiB/s for u8.
            (gib_to_ns_per_mp(8.6, bpp), bpp <= 1.0)
        }
        ConvertStep::RgbToGray { .. } => {
            // RGB→gray weighted sum; ~12 GiB/s for u8, faster for u16.
            (gib_to_ns_per_mp(12.0, bpp), bpp == 3.0)
        }
        ConvertStep::RgbaToGray { .. } => {
            // Similar to RgbToGray plus alpha drop.
            (gib_to_ns_per_mp(10.0, bpp), bpp == 4.0)
        }
        ConvertStep::GrayAlphaToRgba => {
            // u8 2-byte: 95.30 GiB/s, u16: 119.80, f32: 149.72.
            let g = match bpp as usize {
                2 => 95.30,
                4 => 119.80,
                8 => 149.72,
                _ => 60.0,
            };
            (gib_to_ns_per_mp(g, bpp), matches!(bpp as usize, 2 | 4 | 8))
        }
        ConvertStep::GrayAlphaToRgb | ConvertStep::GrayAlphaToGray => {
            // Cheap byte-level transforms; ~80 GiB/s for u8.
            (gib_to_ns_per_mp(80.0, bpp), bpp <= 2.0)
        }
        ConvertStep::GrayToGrayAlpha => {
            // Add opaque alpha to gray; ~100 GiB/s for u8.
            (gib_to_ns_per_mp(100.0, bpp), bpp <= 1.0)
        }

        // ----- Depth conversion (t2_depth). 4096-row, RGB. -----
        ConvertStep::U8ToU16 => (gib_to_ns_per_mp(112.82, bpp), true),
        ConvertStep::U16ToU8 => (gib_to_ns_per_mp(34.39, bpp), true),
        ConvertStep::NaiveU8ToF32 => (gib_to_ns_per_mp(95.21, bpp), true),
        ConvertStep::NaiveF32ToU8 => (gib_to_ns_per_mp(52.99, bpp), true),
        ConvertStep::U16ToF32 => (gib_to_ns_per_mp(88.68, bpp), true),
        ConvertStep::F32ToU16 => (gib_to_ns_per_mp(64.33, bpp), true),
        ConvertStep::F16ToF32 => (gib_to_ns_per_mp(7.09, bpp), true),
        ConvertStep::F32ToF16 => (gib_to_ns_per_mp(3.25, bpp), true),

        // ----- Transfer functions (t3 fused, t4 f32). 4096-row, RGB. -----
        ConvertStep::SrgbU8ToLinearF32 => (gib_to_ns_per_mp(24.26, bpp), true),
        ConvertStep::LinearF32ToSrgbU8 => (gib_to_ns_per_mp(4.56, bpp), true),
        ConvertStep::PqU16ToLinearF32 => (gib_to_ns_per_mp(2.68, bpp), true),
        ConvertStep::LinearF32ToPqU16 => (gib_to_ns_per_mp(1.39, bpp), true),
        ConvertStep::HlgU16ToLinearF32 => (gib_to_ns_per_mp(6.16, bpp), true),
        ConvertStep::LinearF32ToHlgU16 => (gib_to_ns_per_mp(4.44, bpp), true),
        ConvertStep::PqF32ToLinearF32 => (gib_to_ns_per_mp(3.0, bpp), false),
        ConvertStep::LinearF32ToPqF32 => (gib_to_ns_per_mp(2.72, bpp), true),
        ConvertStep::HlgF32ToLinearF32 => (gib_to_ns_per_mp(6.0, bpp), false),
        ConvertStep::LinearF32ToHlgF32 => (gib_to_ns_per_mp(4.0, bpp), false),
        ConvertStep::SrgbF32ToLinearF32 | ConvertStep::SrgbF32ToLinearF32Extended => {
            (gib_to_ns_per_mp(24.95, bpp), true)
        }
        ConvertStep::LinearF32ToSrgbF32 | ConvertStep::LinearF32ToSrgbF32Extended => {
            (gib_to_ns_per_mp(8.0, bpp), false)
        }
        ConvertStep::Bt709F32ToLinearF32 => (gib_to_ns_per_mp(6.0, bpp), false),
        ConvertStep::LinearF32ToBt709F32 => (gib_to_ns_per_mp(4.5, bpp), false),
        ConvertStep::Gamma22F32ToLinearF32 => (gib_to_ns_per_mp(6.0, bpp), false),
        ConvertStep::LinearF32ToGamma22F32 => (gib_to_ns_per_mp(4.5, bpp), false),

        // ----- Alpha mode (t5). Premul/straight conversions. -----
        ConvertStep::StraightToPremul => {
            // f32 4-channel: ~13.74 GiB/s
            (gib_to_ns_per_mp(13.74, bpp), bpp == 16.0)
        }
        ConvertStep::PremulToStraight => {
            // f32 4-channel: ~7.51 GiB/s (divide is slow)
            (gib_to_ns_per_mp(7.51, bpp), bpp == 16.0)
        }

        // ----- Oklab (t6). cbrt-dominated forward, cubed inverse. -----
        ConvertStep::LinearRgbToOklab => (gib_to_ns_per_mp(1.61, bpp), bpp == 12.0),
        ConvertStep::OklabToLinearRgb => (gib_to_ns_per_mp(53.25, bpp), bpp == 12.0),
        ConvertStep::LinearRgbaToOklaba => (gib_to_ns_per_mp(2.14, bpp), bpp == 16.0),
        ConvertStep::OklabaToLinearRgba => (gib_to_ns_per_mp(58.91, bpp), bpp == 16.0),

        // ----- Gamut matrices (t7). 3×3 matrix-multiply on linear F32. -----
        ConvertStep::GamutMatrixRgbF32(_) => (gib_to_ns_per_mp(21.84, bpp), true),
        ConvertStep::GamutMatrixRgbaF32(_) => (gib_to_ns_per_mp(20.0, bpp), false),
        ConvertStep::FusedSrgbU8GamutRgb(_) => (gib_to_ns_per_mp(3.79, bpp), true),
        ConvertStep::FusedSrgbU8GamutRgba(_) => (gib_to_ns_per_mp(3.5, bpp), false),
        ConvertStep::FusedSrgbU16GamutRgb(_) => (gib_to_ns_per_mp(5.84, bpp), true),
        ConvertStep::FusedSrgbU8ToLinearF32Rgb(_) => (gib_to_ns_per_mp(11.19, bpp), true),
        ConvertStep::FusedLinearF32ToSrgbU8Rgb(_) => (gib_to_ns_per_mp(3.11, bpp), true),

        // ----- HDR. BT.2446-A from zentone bench (2026-06-20). -----
        // Bt2446A::map_strip_simd: ~250 Mpix/s on RGB f32 linear-light.
        // 1 MP / 250 Mpix/s = 4 ms/MP = 4_000_000 ns/MP.
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A { .. } => {
            // 1 MP / 250 Mpix/s = 4.0e6 / 1000 = 4000 ns/Mpix × 1024 = ~4.0 ms / MP.
            // Actually: 1_048_576 / 250_000_000 s = 4.194e-3 s = 4_194_304 ns/MP.
            (4_194_304.0, true)
        }
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { .. } => {
            // Hue-preserving rational knee curve in OKLch space. Per-pixel
            // is roughly Oklab forward + scalar curve + Oklab inverse —
            // dominated by the cbrt path. Use ~3 GiB/s as a coarse model.
            (gib_to_ns_per_mp(3.0, bpp), false)
        }
    }
}

/// Step name for [`StepEstimate::name`].
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
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A { .. } => "ToneMapBt2446A",
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { .. } => "SoftCompressOklch",
    }
}

/// Body of the plan-level estimate. Walks the plan's steps once,
/// summing time and tracking the peak intermediate buffer size.
///
/// Memory model:
/// - The output buffer at `to.bytes_per_pixel() * pixels` is always allocated.
/// - Multi-step plans hold two scratch row buffers ping-ponged between
///   intermediate descriptors. Worst-case scratch is `2 * width *
///   max_intermediate_bpp` bytes.
/// - The estimate is for a single per-call working set, NOT a
///   parallel-job-wide cap.
pub(crate) fn estimate_plan(plan: &ConvertPlan, width: u32, height: u32) -> ResourceEstimate {
    // Quick identity short-circuit.
    if plan.is_identity() {
        // Even identity allocates the destination buffer + a memcpy.
        let pixels = (width as u64) * (height as u64);
        let dst_bytes = pixels * plan.to().bytes_per_pixel() as u64;
        // memcpy at ~30 GB/s is a reasonable assumption — but on systems
        // with NUMA effects this can be 10-50 GB/s. Use a midpoint.
        let memcpy_gib_s = 30.0;
        let memcpy_time_ms = (dst_bytes as f64) / (memcpy_gib_s * GIB) * 1_000.0;
        return ResourceEstimate {
            peak_memory_bytes: dst_bytes,
            wall_time_ms: memcpy_time_ms,
            breakdown: Vec::new(),
            confidence: EstimateConfidence::Calibrated,
        };
    }

    let pixels = (width as u64) * (height as u64);
    let pixels_mp = (pixels as f64) / ONE_MP;

    // Output buffer is always allocated.
    let dst_bpp = plan.to().bytes_per_pixel() as u64;
    let dst_bytes = pixels * dst_bpp;

    // Scratch buffers for multi-step plans: two row-sized halves
    // sized to the widest intermediate. Single-step plans use
    // no scratch (the kernel writes directly into the dst row).
    let scratch_per_half_bytes = if plan.steps().len() > 1 {
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

    // Per-step time + breakdown. Memory contribution per step
    // models whether the step allocates a new buffer or operates
    // in-place on the existing one — but since scratch is pre-sized
    // and reused, the per-step memory_bytes is the *cumulative*
    // working set after this step runs, not a delta.
    let mut breakdown: Vec<StepEstimate> = Vec::with_capacity(plan.steps().len());
    let mut total_time_ms = 0.0;
    let mut all_calibrated = true;
    let mut desc = plan.from();

    for step in plan.steps() {
        let current_bpp = desc.bytes_per_pixel();
        let (ns_per_mp, calibrated) = step_cost_ns_per_mp(step, current_bpp);
        let step_time_ms = ns_per_mp * pixels_mp / 1_000_000.0;
        if !calibrated {
            all_calibrated = false;
        }
        total_time_ms += step_time_ms;

        // Working-set after this step ran. The output of this step
        // sits in scratch (or the destination). Use the next-desc
        // bpp times width for the per-step row delta.
        let next = intermediate_after(desc, step);
        let step_mem = (width as u64) * (next.bytes_per_pixel() as u64);

        breakdown.push(StepEstimate {
            name: step_name(step),
            memory_bytes: step_mem,
            time_ms: step_time_ms,
        });

        desc = next;
    }

    // Peak working-set: destination buffer + scratch (for multi-step).
    // The scratch is reused, not added per step.
    let peak_memory_bytes = dst_bytes.saturating_add(scratch_bytes);

    let confidence = if all_calibrated {
        EstimateConfidence::Calibrated
    } else {
        EstimateConfidence::Heuristic
    };

    ResourceEstimate {
        peak_memory_bytes,
        wall_time_ms: total_time_ms,
        breakdown,
        confidence,
    }
}

/// Mirror of `intermediate_desc` in `convert.rs`, exposed via the
/// `ConvertPlan::intermediate_after_step` helper. Kept as a
/// thin re-call so the two don't drift.
fn intermediate_after(current: PixelDescriptor, step: &ConvertStep) -> PixelDescriptor {
    crate::convert::intermediate_desc_for_estimate(current, step)
}

