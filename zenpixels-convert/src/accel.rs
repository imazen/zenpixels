//! Opt-in SIMD accelerator registry and compile-time [`Accelerators`] overrides.
//!
//! By default, [`ConvertPlan::build`](crate::ConvertPlan::build) executes every valid
//! native conversion through a compact universal fast-FMA scalar core (`+87 KB` lean
//! Wasm / `+159 KB` lean x86-64 stripped delta; zero scratch allocation, **0-bit
//! identical** to the x86-64/AArch64 SIMD kernels), so codecs and library crates never
//! bloat binaries with unused SIMD tables and never fail with `NoPath` for missing
//! capability flags.
//!
//! # Enabling fast mode
//!
//! Pick whichever level of opt-in fits your crate:
//!
//! 1. **One-liner process-wide startup (recommended for apps, CLIs, servers, benches):**
//!    ```
//!    zenpixels_convert::accel::enable_all();
//!    ```
//!    Every subsequent [`ConvertPlan::build`](crate::ConvertPlan::build),
//!    [`ConvertPlan::select`](crate::ConvertPlan::select),
//!    [`OutputPlan::build`](crate::OutputPlan::build), and extension-trait call across the
//!    entire process (including inside codec crates) automatically uses full SIMD.
//!
//! 2. **Selective granular opt-in (unused SIMD kernels dead-strip to `0 B`):**
//!    ```
//!    zenpixels_convert::accel::enable_layout(); // garb swizzle / alpha / gray / int premul
//!    zenpixels_convert::accel::enable_srgb();   // linear-srgb sRGB u8/u16/f32
//!    // Or one call for web SDR pipelines (layout + srgb + float + gamut):
//!    zenpixels_convert::accel::enable_web_sdr();
//!    ```
//!
//! 3. **Per-spec `const` override (zero global state; ignores the global registry):**
//!    ```
//!    use zenpixels_convert::{Accelerators, ConversionSpec, ConvertOptions};
//!
//!    let fast = ConversionSpec::fast(ConvertOptions::retain_channels());
//!    let web = ConversionSpec::new(ConvertOptions::retain_channels())
//!        .with_accelerators(&Accelerators::WEB_SDR);
//!    // Hermetic scalar-only override for 0-bit parity tests even after `enable_all()`:
//!    let scalar = ConversionSpec::new(ConvertOptions::retain_channels())
//!        .with_accelerators(&Accelerators::SCALAR);
//!    ```
//!
//! 4. **Cargo feature `fast` (default-off, for top-level binaries):**
//!    Enable `zenpixels-convert = { ..., features = ["fast"] }` in your binary's
//!    `Cargo.toml` to make all unregistered plans default to [`Accelerators::ALL`]
//!    without modifying `main()`. Library crates should not enable `fast`.

use alloc::boxed::Box;
use core::fmt;
use once_cell::race::OnceBox;
use whereat::At;
use zenpixels::sample::FloatFacts;
use zenpixels::{
    AlphaMode, AlphaPolicy, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, ConvertOptions,
    DepthPolicy, GamutPolicy, PixelDescriptor, SignalRange, TransferFunction,
};

use crate::ConvertError;
use crate::convert::{ConvertStep, NativePlan};

/// Internal kernel capability bitmask used solely to specialize [`convert_kernels`]
/// step binders for [`Accelerators::WEB_SDR`] vs [`Accelerators::ALL`].
#[allow(dead_code)]
pub(crate) mod kernel_caps {
    pub(crate) const U8: u64 = 1 << 0;
    pub(crate) const U16: u64 = 1 << 1;
    pub(crate) const F16: u64 = 1 << 2;
    pub(crate) const F32: u64 = 1 << 3;
    pub(crate) const LAYOUT: u64 = 1 << 4;
    pub(crate) const GRAY: u64 = 1 << 5;
    pub(crate) const PREMUL: u64 = 1 << 6;
    pub(crate) const MATTE: u64 = 1 << 7;
    pub(crate) const DEPTH_TRUNCATE: u64 = 1 << 8;
    pub(crate) const TF_SRGB: u64 = 1 << 9;
    pub(crate) const TF_EXTENDED: u64 = 1 << 10;
    pub(crate) const TF_BT709: u64 = 1 << 11;
    pub(crate) const TF_GAMMA22: u64 = 1 << 12;
    pub(crate) const TF_PQ: u64 = 1 << 13;
    pub(crate) const TF_HLG: u64 = 1 << 14;
    pub(crate) const GAMUT_MATRIX: u64 = 1 << 15;
    pub(crate) const GAMUT_FUSED: u64 = 1 << 16;
    pub(crate) const OKLAB: u64 = 1 << 17;
    pub(crate) const SOFT_COMPRESS: u64 = 1 << 18;
    pub(crate) const TONE_MAP: u64 = 1 << 19;
    pub(crate) const PERCEPTUAL_POLAR: u64 = 1 << 20;
    pub(crate) const PERCEPTUAL_HSL_HSV: u64 = 1 << 21;
    pub(crate) const PERCEPTUAL_LCH: u64 = 1 << 22;
    pub(crate) const PERCEPTUAL_JZAZBZ: u64 = 1 << 23;
    pub(crate) const PERCEPTUAL_INT: u64 = 1 << 24;
    pub(crate) const FLOAT_FACTS_SPEC: u64 = 1 << 25;

    pub(crate) const ALL: u64 = !0u64;
    pub(crate) const LAYOUT_ONLY: u64 =
        U8 | U16 | F16 | F32 | LAYOUT | GRAY | PREMUL | DEPTH_TRUNCATE;
    pub(crate) const WEB_SDR: u64 = U8
        | F32
        | LAYOUT
        | GRAY
        | PREMUL
        | MATTE
        | DEPTH_TRUNCATE
        | TF_SRGB
        | TF_EXTENDED
        | GAMUT_MATRIX
        | GAMUT_FUSED
        | FLOAT_FACTS_SPEC;
}

/// Internal planner sub-hooks passed into `NativePlan::new_explicit_core`.
type PlanWithOptsFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    &ConvertOptions,
    &PlannerHooks,
) -> Result<NativePlan, At<ConvertError>>;

#[cfg(feature = "bt2446a")]
type Bt2446AFn =
    fn(PixelDescriptor, PixelDescriptor, crate::HdrConfig) -> Result<NativePlan, At<ConvertError>>;

#[cfg(feature = "oklab")]
type OklabPivotFn = fn(PixelDescriptor, PixelDescriptor) -> Option<PixelDescriptor>;

#[cfg(feature = "perceptual")]
type PerceptualPlanFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    &PlannerHooks,
) -> Result<Option<NativePlan>, At<ConvertError>>;

#[derive(Clone, Copy)]
pub(crate) struct PlannerHooks {
    pub(crate) hlg_render: Option<PlanWithOptsFn>,
    #[cfg(feature = "bt2446a")]
    pub(crate) bt2446a: Option<Bt2446AFn>,
    #[cfg(feature = "oklab")]
    pub(crate) soft_compress: Option<PlanWithOptsFn>,
    #[cfg(feature = "oklab")]
    pub(crate) oklab_pivot: Option<OklabPivotFn>,
    #[cfg(feature = "perceptual")]
    pub(crate) perceptual: Option<PerceptualPlanFn>,
}

impl PlannerHooks {
    pub(crate) const ALL: Self = Self {
        hlg_render: {
            #[cfg(feature = "hdr-transfers")]
            {
                Some(NativePlan::plan_hlg_rendering_core)
            }
            #[cfg(not(feature = "hdr-transfers"))]
            {
                None
            }
        },
        #[cfg(feature = "bt2446a")]
        bt2446a: Some(NativePlan::new_with_hdr_config_unbound),
        #[cfg(feature = "oklab")]
        soft_compress: Some(NativePlan::new_soft_compressed_core),
        #[cfg(feature = "oklab")]
        oklab_pivot: Some(crate::convert::oklab_pivot_pub),
        #[cfg(feature = "perceptual")]
        perceptual: Some(crate::convert::plan_perceptual_core),
    };

    pub(crate) const WEB_SDR: Self = Self {
        hlg_render: None,
        #[cfg(feature = "bt2446a")]
        bt2446a: None,
        #[cfg(feature = "oklab")]
        soft_compress: None,
        #[cfg(feature = "oklab")]
        oklab_pivot: None,
        #[cfg(feature = "perceptual")]
        perceptual: None,
    };
}

/// Bound direct-slot accelerator builder. Returns `Some(Ok(bound_plan))` when the
/// slot owns this `(from, to, options)` conversion, `Some(Err(e))` on an explicit
/// policy/validation error, or `None` to fall through to other slots / universal scalar.
pub(crate) type DirectAccelFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    &ConvertOptions,
    bool,
    FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>>;

/// Full multi-step SIMD pipeline accelerator builder.
pub(crate) type PipelineAccelFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    &ConvertOptions,
    bool,
    FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>>;

/// Compile-time or runtime set of opt-in SIMD conversion accelerators.
///
/// Pass to [`ConversionSpec::with_accelerators`](crate::ConversionSpec::with_accelerators)
/// for a scoped override, or install process-wide via [`enable`]. Any conversion not
/// covered by an installed accelerator seamlessly executes on the 0-bit-identical
/// universal scalar core — never returning `NoPath` for valid formats.
#[derive(Clone, Copy)]
pub struct Accelerators {
    layout: Option<DirectAccelFn>,
    srgb: Option<DirectAccelFn>,
    float: Option<DirectAccelFn>,
    gamut: Option<DirectAccelFn>,
    hdr: Option<DirectAccelFn>,
    oklab: Option<DirectAccelFn>,
    web_pipeline: Option<PipelineAccelFn>,
    all_pipeline: Option<PipelineAccelFn>,
}

impl fmt::Debug for Accelerators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Accelerators")
            .field("layout", &self.layout.is_some())
            .field("srgb", &self.srgb.is_some())
            .field("float", &self.float.is_some())
            .field("gamut", &self.gamut.is_some())
            .field("hdr", &self.hdr.is_some())
            .field("oklab", &self.oklab.is_some())
            .field("web_pipeline", &self.web_pipeline.is_some())
            .field("all_pipeline", &self.all_pipeline.is_some())
            .finish()
    }
}

impl PartialEq for Accelerators {
    fn eq(&self, other: &Self) -> bool {
        self.layout.is_some() == other.layout.is_some()
            && self.srgb.is_some() == other.srgb.is_some()
            && self.float.is_some() == other.float.is_some()
            && self.gamut.is_some() == other.gamut.is_some()
            && self.hdr.is_some() == other.hdr.is_some()
            && self.oklab.is_some() == other.oklab.is_some()
            && self.web_pipeline.is_some() == other.web_pipeline.is_some()
            && self.all_pipeline.is_some() == other.all_pipeline.is_some()
    }
}

impl Eq for Accelerators {}

impl Default for Accelerators {
    fn default() -> Self {
        Self::SCALAR
    }
}

impl Accelerators {
    /// No SIMD accelerators installed: execute everything through the compact
    /// universal fast-FMA scalar core (`0 B` of SIMD codegen).
    ///
    /// Passing `&Accelerators::SCALAR` to
    /// [`ConversionSpec::with_accelerators`](crate::ConversionSpec::with_accelerators)
    /// forces universal scalar even when [`enable_all`] has been called or the `fast`
    /// Cargo feature is enabled — useful for hermetic 0-bit parity tests.
    pub const SCALAR: Self = Self {
        layout: None,
        srgb: None,
        float: None,
        gamut: None,
        hdr: None,
        oklab: None,
        web_pipeline: None,
        all_pipeline: None,
    };

    /// Same-depth byte swizzle, alpha add/drop, gray expand/collapse, integer
    /// premultiplication, and `U8 ↔ U16` depth conversion (`garb` SIMD).
    pub const LAYOUT: Self = Self::SCALAR.with_layout();

    /// Web SDR preset: layout, sRGB transfers, `U8 ↔ F32` float normalization/quantization
    /// and `F32` premul, fused SDR gamut conversion, and composed multi-step SDR SIMD pipelines
    /// (excludes `BT.709`, `Gamma 2.2`, `PQ`/`HLG`, `U16`/`F16` float, and `Oklab` SIMD,
    /// which remain available via universal scalar).
    pub const WEB_SDR: Self = Self::SCALAR
        .with_layout()
        .with_srgb()
        .with_float_web_sdr()
        .with_gamut()
        .with_web_pipeline();

    /// Every SIMD accelerator plus the full composed multi-step SIMD pipeline.
    pub const ALL: Self = Self::SCALAR
        .with_layout()
        .with_srgb()
        .with_float()
        .with_gamut()
        .with_hdr()
        .with_oklab()
        .with_all_pipeline();

    /// Alias for [`Self::SCALAR`], starting a fluent `const` builder chain.
    #[inline]
    pub const fn new() -> Self {
        Self::SCALAR
    }

    /// Add `garb` / integer layout, alpha add/drop, gray, integer premul, and `U8 ↔ U16` SIMD.
    #[inline]
    pub const fn with_layout(mut self) -> Self {
        self.layout = Some(accel_layout);
        self
    }

    /// Add `linear-srgb` SIMD sRGB transfer kernels (`U8`/`U16`/`F32`, clamped + extended,
    /// and sRGB `U8` matte).
    #[inline]
    pub const fn with_srgb(mut self) -> Self {
        self.srgb = Some(accel_srgb);
        self
    }

    /// Add `float_simd` (`U8`/`U16 ↔ F32`, `FloatFacts`-specialized quantization, `F32` premul)
    /// and hardware `F16C`/`NEON-fp16` `F16 ↔ F32` SIMD kernels.
    #[inline]
    pub const fn with_float(mut self) -> Self {
        self.float = Some(accel_float);
        self
    }

    /// Internal `U8 ↔ F32` + `F32` premul float slot for [`Self::WEB_SDR`] (excludes `U16`/`F16` SIMD).
    #[inline]
    const fn with_float_web_sdr(mut self) -> Self {
        self.float = Some(accel_float_web_sdr);
        self
    }

    /// Add `fast_gamut` SIMD 3×3 matrix and fused sRGB `U8`/`U16`/`F32` gamut kernels.
    #[inline]
    pub const fn with_gamut(mut self) -> Self {
        self.gamut = Some(accel_gamut);
        self
    }

    /// Add `BT.709`, `Gamma 2.2`, and (when `hdr-transfers` is enabled) `PQ` / `HLG` SIMD kernels.
    #[inline]
    pub const fn with_hdr(mut self) -> Self {
        self.hdr = Some(accel_hdr);
        self
    }

    /// Add `Oklab` / `OklabA` (`FloatFacts`-specialized) SIMD kernels (no-op when `oklab` feature is off).
    #[inline]
    #[allow(unused_mut)]
    pub const fn with_oklab(mut self) -> Self {
        #[cfg(feature = "oklab")]
        {
            self.oklab = Some(accel_oklab);
        }
        self
    }

    /// Add composed multi-step web-SDR SIMD pipelines (`U8`/`F32` + sRGB + gamut).
    #[inline]
    const fn with_web_pipeline(mut self) -> Self {
        self.web_pipeline = Some(accel_pipeline_web_sdr);
        self
    }

    /// Add the full composed multi-step SIMD pipeline across all depths, transfers, and models.
    #[inline]
    const fn with_all_pipeline(mut self) -> Self {
        self.all_pipeline = Some(accel_pipeline_all);
        self
    }

    /// Combine two accelerator sets, taking every accelerator present in either set.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            layout: if self.layout.is_some() {
                self.layout
            } else {
                other.layout
            },
            srgb: if self.srgb.is_some() {
                self.srgb
            } else {
                other.srgb
            },
            float: if self.float.is_some() {
                self.float
            } else {
                other.float
            },
            gamut: if self.gamut.is_some() {
                self.gamut
            } else {
                other.gamut
            },
            hdr: if self.hdr.is_some() {
                self.hdr
            } else {
                other.hdr
            },
            oklab: if self.oklab.is_some() {
                self.oklab
            } else {
                other.oklab
            },
            web_pipeline: if self.web_pipeline.is_some() {
                self.web_pipeline
            } else {
                other.web_pipeline
            },
            all_pipeline: if self.all_pipeline.is_some() {
                self.all_pipeline
            } else {
                other.all_pipeline
            },
        }
    }

    /// True if no SIMD accelerators are installed (universal scalar only).
    #[inline]
    pub const fn is_scalar(&self) -> bool {
        self.layout.is_none()
            && self.srgb.is_none()
            && self.float.is_none()
            && self.gamut.is_none()
            && self.hdr.is_none()
            && self.oklab.is_none()
            && self.web_pipeline.is_none()
            && self.all_pipeline.is_none()
    }

    /// True if layout/swizzle/gray/integer-premul SIMD is enabled.
    #[inline]
    pub const fn has_layout(&self) -> bool {
        self.layout.is_some()
    }

    /// True if sRGB SIMD transfer kernels are enabled.
    #[inline]
    pub const fn has_srgb(&self) -> bool {
        self.srgb.is_some()
    }

    /// True if `float_simd` and `F16` SIMD kernels are enabled.
    #[inline]
    pub const fn has_float(&self) -> bool {
        self.float.is_some()
    }

    /// True if `fast_gamut` SIMD matrix/fused kernels are enabled.
    #[inline]
    pub const fn has_gamut(&self) -> bool {
        self.gamut.is_some()
    }

    /// True if BT.709 / Gamma 2.2 / PQ / HLG SIMD transfer kernels are enabled.
    #[inline]
    pub const fn has_hdr(&self) -> bool {
        self.hdr.is_some()
    }

    /// True if Oklab SIMD kernels are enabled.
    #[inline]
    pub const fn has_oklab(&self) -> bool {
        self.oklab.is_some()
    }

    /// True if full composed multi-step SIMD pipelines are enabled.
    #[inline]
    pub const fn is_all(&self) -> bool {
        self.all_pipeline.is_some()
    }

    /// Attempt to build a bound SIMD [`NativePlan`] for `(from, to, options)` using
    /// whichever slots are installed in `self`. Returns `None` when no installed slot
    /// covers the pair (caller then falls through to [`UniversalPlan`](crate::universal::UniversalPlan)).
    pub(crate) fn try_accelerate(
        &self,
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
        exact: bool,
        input_facts: FloatFacts,
    ) -> Option<Result<NativePlan, At<ConvertError>>> {
        // Prefer full composed SIMD pipeline when installed (covers multi-step + fused peepholes).
        if let Some(f) = self.all_pipeline
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        // Direct slots (order: cheap single-step checks first).
        if let Some(f) = self.layout
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.srgb
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.float
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.gamut
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.hdr
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.oklab
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        if let Some(f) = self.web_pipeline
            && let Some(res) = f(from, to, options, exact, input_facts)
        {
            return Some(res);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Process-wide safe `OnceBox` registry (no_std + alloc, lock-free reads)
// ---------------------------------------------------------------------------

static SLOT_LAYOUT: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_SRGB: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_FLOAT: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_GAMUT: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_HDR: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_OKLAB: OnceBox<DirectAccelFn> = OnceBox::new();
static SLOT_WEB_PIPELINE: OnceBox<PipelineAccelFn> = OnceBox::new();
static SLOT_ALL_PIPELINE: OnceBox<PipelineAccelFn> = OnceBox::new();

#[inline]
fn install_slot<T: Copy + 'static>(slot: &OnceBox<T>, val: T) {
    if slot.get().is_none() {
        let _ = slot.set(Box::new(val));
    }
}

/// Register `garb` / integer layout, alpha add/drop, gray, integer premul, and `U8 ↔ U16`
/// SIMD accelerators process-wide. Idempotent and thread-safe.
pub fn enable_layout() {
    install_slot(&SLOT_LAYOUT, accel_layout);
}

/// Register `linear-srgb` sRGB SIMD transfer kernels (`U8`/`U16`/`F32`, clamped + extended,
/// and sRGB `U8` matte) process-wide. Idempotent and thread-safe.
pub fn enable_srgb() {
    install_slot(&SLOT_SRGB, accel_srgb);
}

/// Register `float_simd` (`U8`/`U16 ↔ F32`, `FloatFacts`-specialized quantization, `F32` premul)
/// and hardware `F16C`/`NEON-fp16` `F16 ↔ F32` SIMD accelerators process-wide.
pub fn enable_float() {
    install_slot(&SLOT_FLOAT, accel_float);
}

/// Register `fast_gamut` SIMD 3×3 matrix and fused sRGB `U8`/`U16`/`F32` gamut kernels
/// process-wide.
pub fn enable_gamut() {
    install_slot(&SLOT_GAMUT, accel_gamut);
}

/// Register `BT.709`, `Gamma 2.2`, and (when `hdr-transfers` is enabled) `PQ` / `HLG` SIMD
/// transfer kernels process-wide.
pub fn enable_hdr() {
    install_slot(&SLOT_HDR, accel_hdr);
}

/// Register `Oklab` / `OklabA` (`FloatFacts`-specialized) SIMD kernels process-wide
/// (no-op when the `oklab` Cargo feature is disabled).
pub fn enable_oklab() {
    #[cfg(feature = "oklab")]
    install_slot(&SLOT_OKLAB, accel_oklab);
}

/// Register all web SDR SIMD accelerators (`layout + srgb + float + gamut` + composed
/// web-SDR multi-step pipelines) process-wide in one call.
///
/// Unused HDR, `U16`/`F16`, and `Oklab` SIMD kernels remain dead-stripped while still
/// working via the 0-bit-identical universal scalar core.
pub fn enable_web_sdr() {
    enable_layout();
    enable_srgb();
    install_slot(&SLOT_FLOAT, accel_float_web_sdr);
    enable_gamut();
    install_slot(&SLOT_WEB_PIPELINE, accel_pipeline_web_sdr);
}

/// Register **every** SIMD accelerator and composed multi-step SIMD pipeline process-wide.
///
/// Call once at application, server, CLI, or benchmark startup so all libraries and codecs
/// in the process run at full SIMD speed.
pub fn enable_all() {
    enable_layout();
    enable_srgb();
    enable_float();
    enable_gamut();
    enable_hdr();
    enable_oklab();
    install_slot(&SLOT_ALL_PIPELINE, accel_pipeline_all);
}

/// Register every accelerator present in `accels` into the process-wide registry.
pub fn enable(accels: Accelerators) {
    if let Some(f) = accels.layout {
        install_slot(&SLOT_LAYOUT, f);
    }
    if let Some(f) = accels.srgb {
        install_slot(&SLOT_SRGB, f);
    }
    if let Some(f) = accels.float {
        install_slot(&SLOT_FLOAT, f);
    }
    if let Some(f) = accels.gamut {
        install_slot(&SLOT_GAMUT, f);
    }
    if let Some(f) = accels.hdr {
        install_slot(&SLOT_HDR, f);
    }
    if let Some(f) = accels.oklab {
        install_slot(&SLOT_OKLAB, f);
    }
    if let Some(f) = accels.web_pipeline {
        install_slot(&SLOT_WEB_PIPELINE, f);
    }
    if let Some(f) = accels.all_pipeline {
        install_slot(&SLOT_ALL_PIPELINE, f);
    }
}

/// Snapshot of the accelerators currently installed in the process-wide registry.
///
/// When the `fast` Cargo feature is enabled, returns [`Accelerators::ALL`].
#[inline]
pub fn enabled() -> Accelerators {
    #[cfg(feature = "fast")]
    {
        Accelerators::ALL
    }
    #[cfg(not(feature = "fast"))]
    {
        snapshot()
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn snapshot() -> Accelerators {
    Accelerators {
        layout: SLOT_LAYOUT.get().copied(),
        srgb: SLOT_SRGB.get().copied(),
        float: SLOT_FLOAT.get().copied(),
        gamut: SLOT_GAMUT.get().copied(),
        hdr: SLOT_HDR.get().copied(),
        oklab: SLOT_OKLAB.get().copied(),
        web_pipeline: SLOT_WEB_PIPELINE.get().copied(),
        all_pipeline: SLOT_ALL_PIPELINE.get().copied(),
    }
}

// ---------------------------------------------------------------------------
// Direct accelerator slot implementations (each links ONLY its own SIMD crate!)
// ---------------------------------------------------------------------------

fn bind_single(
    from: PixelDescriptor,
    to: PixelDescriptor,
    step: ConvertStep,
    input_facts: FloatFacts,
    bind_fn: fn(&mut NativePlan) -> Result<(), At<ConvertError>>,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    let mut plan = NativePlan::build(from, to, alloc::vec![step]);
    plan.propagate_float_facts(input_facts);
    match bind_fn(&mut plan) {
        Ok(()) => Some(Ok(plan)),
        Err(_) => None,
    }
}

/// Slot 1: `garb` / integer layout, gray, integer premul, and same-layout `U8 ↔ U16`.
fn accel_layout(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if from.transfer() != to.transfer()
        || from.primaries != to.primaries
        || from.signal_range != to.signal_range
    {
        return None;
    }
    // Same-layout, same-alpha U8 <-> U16 depth conversion.
    if from.layout() == to.layout()
        && from.alpha() == to.alpha()
        && from.signal_range == SignalRange::Full
        && matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
    {
        match (from.channel_type(), to.channel_type()) {
            (ChannelType::U8, ChannelType::U16) => {
                return bind_single(
                    from,
                    to,
                    ConvertStep::U8ToU16,
                    input_facts,
                    NativePlan::bind::<{ kernel_caps::LAYOUT_ONLY }>,
                );
            }
            (ChannelType::U16, ChannelType::U8) if !exact => {
                if options.depth_policy() == DepthPolicy::Forbid {
                    return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
                }
                let step = if options.depth_policy() == DepthPolicy::Truncate {
                    ConvertStep::U16ToU8Truncate
                } else {
                    ConvertStep::U16ToU8
                };
                return bind_single(
                    from,
                    to,
                    step,
                    input_facts,
                    NativePlan::bind::<{ kernel_caps::LAYOUT_ONLY }>,
                );
            }
            _ => {}
        }
    }
    // Same-depth layout / gray / integer premul (no matte, no F32 premul so float_simd isn't pulled).
    if from.channel_type() != to.channel_type()
        || (from.has_alpha()
            && !to.has_alpha()
            && matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. }))
    {
        return None;
    }
    if from.channel_type() == ChannelType::F32
        && from.has_alpha()
        && to.has_alpha()
        && from.alpha() != to.alpha()
        && (from.alpha() == Some(AlphaMode::Premultiplied)
            || to.alpha() == Some(AlphaMode::Premultiplied))
    {
        // Leave F32 premul to `accel_float` so `enable_layout()` never links `float_simd`.
        return None;
    }
    let mut plan = match NativePlan::plan_layout_only(from, to, options, exact) {
        Ok(p) => p,
        Err(_) => return None,
    };
    plan.propagate_float_facts(input_facts);
    const LAYOUT_NO_F32_PREMUL: u64 = kernel_caps::U8
        | kernel_caps::U16
        | kernel_caps::F16
        | kernel_caps::F32
        | kernel_caps::LAYOUT
        | kernel_caps::GRAY;
    const LAYOUT_INT_PREMUL: u64 = kernel_caps::U8 | kernel_caps::U16 | kernel_caps::PREMUL;
    let res = if from.channel_type() == ChannelType::F32 {
        plan.bind::<LAYOUT_NO_F32_PREMUL>()
    } else {
        plan.bind::<{ LAYOUT_NO_F32_PREMUL | LAYOUT_INT_PREMUL }>()
    };
    match res {
        Ok(()) => Some(Ok(plan)),
        Err(_) => None,
    }
}

/// Slot 2: `linear-srgb` sRGB SIMD (`U8 ↔ F32`, `F32 ↔ F32` clamped/ext, `RGBA8` sRGB matte).
fn accel_srgb(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.primaries != to.primaries
        || from.signal_range != SignalRange::Full
        || to.signal_range != SignalRange::Full
    {
        return None;
    }
    const SRGB_CAPS: u64 = kernel_caps::U8
        | kernel_caps::F32
        | kernel_caps::TF_SRGB
        | kernel_caps::TF_EXTENDED
        | kernel_caps::MATTE;

    // Straight RGBA8 sRGB -> RGB8 sRGB matte composite (uses linear-srgb LUTs, zero float_simd).
    if from.channel_type() == ChannelType::U8
        && to.channel_type() == ChannelType::U8
        && from.layout() == ChannelLayout::Rgba
        && to.layout() == ChannelLayout::Rgb
        && from.transfer() == TransferFunction::Srgb
        && to.transfer() == TransferFunction::Srgb
        && from.alpha() != Some(AlphaMode::Premultiplied)
        && from.alpha() != Some(AlphaMode::Undefined)
        && let AlphaPolicy::CompositeOnto {
            background,
            reference_white,
        } = options.alpha_policy()
    {
        fn noop(_: &mut [f32]) {}
        let step =
            ConvertStep::MatteComposite(alloc::boxed::Box::new(crate::convert::MatteState {
                background,
                reference_white,
                scale: 1.0,
                clip: options.gamut_policy() == GamutPolicy::Clip,
                linear: [0.0; 3],
                to_linear: noop,
                from_linear: noop,
            }));
        return bind_single(from, to, step, input_facts, NativePlan::bind::<SRGB_CAPS>);
    }

    if from.layout() != to.layout()
        || from.alpha() != to.alpha()
        || from.alpha() == Some(AlphaMode::Premultiplied)
        || !matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
    {
        return None;
    }
    let ext = options.gamut_policy() == GamutPolicy::PreserveExtended;
    let step = match (
        from.channel_type(),
        from.transfer(),
        to.channel_type(),
        to.transfer(),
    ) {
        (ChannelType::U8, TransferFunction::Srgb, ChannelType::F32, TransferFunction::Linear) => {
            ConvertStep::SrgbU8ToLinearF32
        }
        (ChannelType::F32, TransferFunction::Linear, ChannelType::U8, TransferFunction::Srgb) => {
            if options.depth_policy() == DepthPolicy::Forbid {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            if options.depth_policy() == DepthPolicy::Truncate {
                // Truncation splits into `[LinearF32ToSrgbF32, NaiveF32ToU8Truncate]`
                // in the pipeline / universal path instead of rounded `LinearF32ToSrgbU8`.
                return None;
            }
            ConvertStep::LinearF32ToSrgbU8
        }
        (ChannelType::F32, TransferFunction::Srgb, ChannelType::F32, TransferFunction::Linear) => {
            if ext {
                ConvertStep::SrgbF32ToLinearF32Extended
            } else {
                ConvertStep::SrgbF32ToLinearF32
            }
        }
        (ChannelType::F32, TransferFunction::Linear, ChannelType::F32, TransferFunction::Srgb) => {
            if ext {
                ConvertStep::LinearF32ToSrgbF32Extended
            } else {
                ConvertStep::LinearF32ToSrgbF32
            }
        }
        _ => return None,
    };
    bind_single(from, to, step, input_facts, NativePlan::bind::<SRGB_CAPS>)
}

/// Slot 3: `float_simd` depth normalization/quantization (`FloatFacts`-specialized), `F32` premul,
/// and hardware `F16C`/`NEON-fp16` `F16 ↔ F32`.
fn accel_float(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.primaries != to.primaries
        || from.transfer() != to.transfer()
        || from.signal_range != SignalRange::Full
        || to.signal_range != SignalRange::Full
        || from.layout() != to.layout()
        || !matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
    {
        return None;
    }
    const FLOAT_CAPS: u64 = kernel_caps::U8
        | kernel_caps::U16
        | kernel_caps::F16
        | kernel_caps::F32
        | kernel_caps::PREMUL
        | kernel_caps::DEPTH_TRUNCATE
        | kernel_caps::FLOAT_FACTS_SPEC;

    // F32 straight <-> premul.
    if from.channel_type() == ChannelType::F32
        && to.channel_type() == ChannelType::F32
        && from.has_alpha()
        && to.has_alpha()
        && from.alpha() != to.alpha()
    {
        let step = match (from.alpha(), to.alpha()) {
            (Some(AlphaMode::Straight | AlphaMode::Opaque), Some(AlphaMode::Premultiplied)) => {
                ConvertStep::StraightToPremul
            }
            (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => {
                ConvertStep::PremulToStraight
            }
            _ => return None,
        };
        return bind_single(from, to, step, input_facts, NativePlan::bind::<FLOAT_CAPS>);
    }

    if from.alpha() != to.alpha() {
        return None;
    }
    let trunc = options.depth_policy() == DepthPolicy::Truncate;
    let forbid_narrow = options.depth_policy() == DepthPolicy::Forbid;
    let step = match (from.channel_type(), to.channel_type()) {
        (ChannelType::U8, ChannelType::F32) => ConvertStep::NaiveU8ToF32,
        (ChannelType::F32, ChannelType::U8) => {
            if forbid_narrow {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            if trunc {
                ConvertStep::NaiveF32ToU8Truncate
            } else {
                ConvertStep::NaiveF32ToU8
            }
        }
        (ChannelType::U16, ChannelType::F32) => ConvertStep::U16ToF32,
        (ChannelType::F32, ChannelType::U16) => {
            if forbid_narrow {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            if trunc {
                ConvertStep::F32ToU16Truncate
            } else {
                ConvertStep::F32ToU16
            }
        }
        (ChannelType::F16, ChannelType::F32) => ConvertStep::F16ToF32,
        (ChannelType::F32, ChannelType::F16) => {
            if forbid_narrow {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            ConvertStep::F32ToF16
        }
        _ => return None,
    };
    bind_single(from, to, step, input_facts, NativePlan::bind::<FLOAT_CAPS>)
}

/// Slot 3 (Web SDR): `U8 ↔ F32` float normalization/quantization and `F32` premul only
/// (leaves `U16`/`F16` SIMD dead-stripped; universal scalar handles them identically).
fn accel_float_web_sdr(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.primaries != to.primaries
        || from.transfer() != to.transfer()
        || from.signal_range != SignalRange::Full
        || to.signal_range != SignalRange::Full
        || from.layout() != to.layout()
        || !matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
    {
        return None;
    }
    const FLOAT_WEB_CAPS: u64 = kernel_caps::U8
        | kernel_caps::F32
        | kernel_caps::PREMUL
        | kernel_caps::DEPTH_TRUNCATE
        | kernel_caps::FLOAT_FACTS_SPEC;

    if from.channel_type() == ChannelType::F32
        && to.channel_type() == ChannelType::F32
        && from.has_alpha()
        && to.has_alpha()
        && from.alpha() != to.alpha()
    {
        let step = match (from.alpha(), to.alpha()) {
            (Some(AlphaMode::Straight | AlphaMode::Opaque), Some(AlphaMode::Premultiplied)) => {
                ConvertStep::StraightToPremul
            }
            (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => {
                ConvertStep::PremulToStraight
            }
            _ => return None,
        };
        return bind_single(
            from,
            to,
            step,
            input_facts,
            NativePlan::bind::<FLOAT_WEB_CAPS>,
        );
    }

    if from.alpha() != to.alpha() {
        return None;
    }
    let trunc = options.depth_policy() == DepthPolicy::Truncate;
    let forbid_narrow = options.depth_policy() == DepthPolicy::Forbid;
    let step = match (from.channel_type(), to.channel_type()) {
        (ChannelType::U8, ChannelType::F32) => ConvertStep::NaiveU8ToF32,
        (ChannelType::F32, ChannelType::U8) => {
            if forbid_narrow {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            if trunc {
                ConvertStep::NaiveF32ToU8Truncate
            } else {
                ConvertStep::NaiveF32ToU8
            }
        }
        _ => return None,
    };
    bind_single(
        from,
        to,
        step,
        input_facts,
        NativePlan::bind::<FLOAT_WEB_CAPS>,
    )
}

/// Slot 4: `fast_gamut` SIMD 3×3 matrix and fused sRGB `U8`/`U16`/`F32` gamut kernels.
fn accel_gamut(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.primaries == to.primaries
        || from.primaries == ColorPrimaries::Unknown
        || to.primaries == ColorPrimaries::Unknown
        || from.signal_range != SignalRange::Full
        || to.signal_range != SignalRange::Full
        || from.layout() != to.layout()
        || from.alpha() != to.alpha()
        || from.alpha() == Some(AlphaMode::Premultiplied)
        || options.gamut_policy() == GamutPolicy::SoftCompress
    {
        return None;
    }
    let m = from.primaries.gamut_matrix_to(to.primaries)?;
    let flat = [
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ];
    const GAMUT_CAPS: u64 = kernel_caps::U8
        | kernel_caps::U16
        | kernel_caps::F32
        | kernel_caps::TF_SRGB
        | kernel_caps::GAMUT_MATRIX
        | kernel_caps::GAMUT_FUSED;

    let step = match (
        from.channel_type(),
        from.layout(),
        from.transfer(),
        to.channel_type(),
        to.transfer(),
    ) {
        (
            ChannelType::F32,
            ChannelLayout::Rgb,
            TransferFunction::Linear,
            ChannelType::F32,
            TransferFunction::Linear,
        ) => ConvertStep::GamutMatrixRgbF32(flat, to.primaries),
        (
            ChannelType::F32,
            ChannelLayout::Rgba,
            TransferFunction::Linear,
            ChannelType::F32,
            TransferFunction::Linear,
        ) => ConvertStep::GamutMatrixRgbaF32(flat, to.primaries),
        (
            ChannelType::U8,
            ChannelLayout::Rgb,
            TransferFunction::Srgb,
            ChannelType::U8,
            TransferFunction::Srgb,
        ) => ConvertStep::Fused {
            kind: crate::convert::FusedKind::SrgbU8GamutRgb,
            matrix: flat,
            primaries: to.primaries,
        },
        (
            ChannelType::U8,
            ChannelLayout::Rgba,
            TransferFunction::Srgb,
            ChannelType::U8,
            TransferFunction::Srgb,
        ) => ConvertStep::Fused {
            kind: crate::convert::FusedKind::SrgbU8GamutRgba,
            matrix: flat,
            primaries: to.primaries,
        },
        (
            ChannelType::U16,
            ChannelLayout::Rgb,
            TransferFunction::Srgb,
            ChannelType::U16,
            TransferFunction::Srgb,
        ) => ConvertStep::Fused {
            kind: crate::convert::FusedKind::SrgbU16GamutRgb,
            matrix: flat,
            primaries: to.primaries,
        },
        (
            ChannelType::U8,
            ChannelLayout::Rgb,
            TransferFunction::Srgb,
            ChannelType::F32,
            TransferFunction::Linear,
        ) => ConvertStep::Fused {
            kind: crate::convert::FusedKind::SrgbU8ToLinearF32Rgb,
            matrix: flat,
            primaries: to.primaries,
        },
        (
            ChannelType::F32,
            ChannelLayout::Rgb,
            TransferFunction::Linear,
            ChannelType::U8,
            TransferFunction::Srgb,
        ) => {
            if options.depth_policy() == DepthPolicy::Forbid {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            ConvertStep::Fused {
                kind: crate::convert::FusedKind::LinearF32ToSrgbU8Rgb,
                matrix: flat,
                primaries: to.primaries,
            }
        }
        _ => return None,
    };
    bind_single(from, to, step, input_facts, NativePlan::bind::<GAMUT_CAPS>)
}

/// Slot 5: `BT.709`, `Gamma 2.2`, and (when `hdr-transfers` is enabled) `PQ` / `HLG` SIMD.
fn accel_hdr(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.primaries != to.primaries
        || from.signal_range != SignalRange::Full
        || to.signal_range != SignalRange::Full
        || from.layout() != to.layout()
        || from.alpha() != to.alpha()
        || from.alpha() == Some(AlphaMode::Premultiplied)
        || !matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
    {
        return None;
    }
    const HDR_CAPS: u64 = kernel_caps::U16
        | kernel_caps::F32
        | kernel_caps::TF_BT709
        | kernel_caps::TF_GAMMA22
        | kernel_caps::TF_EXTENDED
        | kernel_caps::TF_PQ
        | kernel_caps::TF_HLG;

    let ext = options.gamut_policy() == GamutPolicy::PreserveExtended;
    let step = match (
        from.channel_type(),
        from.transfer(),
        to.channel_type(),
        to.transfer(),
    ) {
        (ChannelType::F32, TransferFunction::Bt709, ChannelType::F32, TransferFunction::Linear) => {
            ConvertStep::Bt709F32ToLinearF32
        }
        (ChannelType::F32, TransferFunction::Linear, ChannelType::F32, TransferFunction::Bt709) => {
            ConvertStep::LinearF32ToBt709F32
        }
        (
            ChannelType::F32,
            TransferFunction::Gamma22,
            ChannelType::F32,
            TransferFunction::Linear,
        ) => {
            if ext {
                ConvertStep::Gamma22F32ToLinearF32Extended
            } else {
                ConvertStep::Gamma22F32ToLinearF32
            }
        }
        (
            ChannelType::F32,
            TransferFunction::Linear,
            ChannelType::F32,
            TransferFunction::Gamma22,
        ) => {
            if ext {
                ConvertStep::LinearF32ToGamma22F32Extended
            } else {
                ConvertStep::LinearF32ToGamma22F32
            }
        }
        #[cfg(feature = "hdr-transfers")]
        (ChannelType::U16, TransferFunction::Pq, ChannelType::F32, TransferFunction::Linear) => {
            ConvertStep::PqU16ToLinearF32
        }
        #[cfg(feature = "hdr-transfers")]
        (ChannelType::F32, TransferFunction::Linear, ChannelType::U16, TransferFunction::Pq) => {
            if options.depth_policy() == DepthPolicy::Forbid {
                return Some(Err(whereat::at!(ConvertError::DepthReductionForbidden)));
            }
            if options.depth_policy() == DepthPolicy::Truncate {
                return None;
            }
            ConvertStep::LinearF32ToPqU16
        }
        #[cfg(feature = "hdr-transfers")]
        (ChannelType::F32, TransferFunction::Pq, ChannelType::F32, TransferFunction::Linear) => {
            ConvertStep::PqF32ToLinearF32
        }
        #[cfg(feature = "hdr-transfers")]
        (ChannelType::F32, TransferFunction::Linear, ChannelType::F32, TransferFunction::Pq) => {
            ConvertStep::LinearF32ToPqF32
        }
        _ => return None,
    };

    let pq_boundary = from.transfer() != to.transfer()
        && (from.transfer() == TransferFunction::Pq || to.transfer() == TransferFunction::Pq);
    let mut plan = NativePlan::build(from, to, alloc::vec![step]);
    if pq_boundary {
        let Some(anchor) = options.sdr_white() else {
            return Some(Err(whereat::at!(ConvertError::SdrWhiteRequired)));
        };
        plan = plan.with_pq_anchor(anchor);
    }
    plan.propagate_float_facts(input_facts);
    match plan.bind::<HDR_CAPS>() {
        Ok(()) => Some(Ok(plan)),
        Err(_) => None,
    }
}

/// Slot 6: `Oklab` / `OklabA` (`FloatFacts`-specialized) SIMD kernels.
#[cfg(feature = "oklab")]
fn accel_oklab(
    from: PixelDescriptor,
    to: PixelDescriptor,
    _options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    if exact
        || from.channel_type() != ChannelType::F32
        || to.channel_type() != ChannelType::F32
        || from.alpha() != to.alpha()
        || from.alpha() == Some(AlphaMode::Premultiplied)
    {
        return None;
    }
    let (step, rgb_side, oklab_side) = match (from.layout(), to.layout()) {
        (ChannelLayout::Rgb, ChannelLayout::Oklab) => (ConvertStep::LinearRgbToOklab, from, to),
        (ChannelLayout::Oklab, ChannelLayout::Rgb) => (ConvertStep::OklabToLinearRgb, to, from),
        (ChannelLayout::Rgba, ChannelLayout::OklabA) => (ConvertStep::LinearRgbaToOklaba, from, to),
        (ChannelLayout::OklabA, ChannelLayout::Rgba) => (ConvertStep::OklabaToLinearRgba, to, from),
        _ => return None,
    };
    if rgb_side.transfer() != TransferFunction::Linear
        || !matches!(
            oklab_side.transfer(),
            TransferFunction::Unknown | TransferFunction::Linear
        )
    {
        return None;
    }
    let _ = crate::oklab::OklabSpace::for_primaries(rgb_side.primaries)?;
    const OKLAB_CAPS: u64 = kernel_caps::F32 | kernel_caps::OKLAB | kernel_caps::FLOAT_FACTS_SPEC;
    bind_single(from, to, step, input_facts, NativePlan::bind::<OKLAB_CAPS>)
}

/// Pipeline slot (Web SDR): multi-step composed SIMD for `U8`/`F32` + sRGB + gamut;
/// gracefully returns `None` (falling back to universal scalar) for out-of-subset steps.
fn accel_pipeline_web_sdr(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    let mut plan = if exact {
        NativePlan::new_preserving_samples_core(from, to, &PlannerHooks::WEB_SDR).ok()?
    } else {
        match NativePlan::new_explicit_core(from, to, options, &PlannerHooks::WEB_SDR) {
            Ok(p) => p,
            Err(e) if matches!(e.error(), ConvertError::NoPath { .. }) => return None,
            Err(e) => return Some(Err(e)),
        }
    };
    plan.propagate_float_facts(input_facts);
    match plan.bind::<{ kernel_caps::WEB_SDR }>() {
        Ok(()) => Some(Ok(plan)),
        Err(_) => None,
    }
}

/// Pipeline slot (All): full multi-step composed SIMD across every depth, transfer,
/// gamut, Oklab, HLG render, soft-compress, and perceptual kernel.
fn accel_pipeline_all(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    exact: bool,
    input_facts: FloatFacts,
) -> Option<Result<NativePlan, At<ConvertError>>> {
    let mut plan = if exact {
        match NativePlan::new_preserving_samples_core(from, to, &PlannerHooks::ALL) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        }
    } else {
        match NativePlan::new_explicit_core(from, to, options, &PlannerHooks::ALL) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        }
    };
    plan.propagate_float_facts(input_facts);
    match plan.bind::<{ kernel_caps::ALL }>() {
        Ok(()) => Some(Ok(plan)),
        Err(e) => Some(Err(e)),
    }
}
