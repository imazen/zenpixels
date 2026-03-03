# What zenfilters Needs from zenpixels

zenfilters operates on planar Oklab f32 data. Filters see f32 slices — they never see primaries, transfer functions, or CICP. All color management lives in the conversion boundary.

The pipeline is:

```
pixels in → zenpixels converts to planar Oklab → filters run → zenpixels converts back
```

This is the same thing zenpixels already does for every format conversion. Planar Oklab is just another target format.

## What Already Landed

The `5d4cc57` commit added the descriptor foundation:

- `PlaneSemantic::OklabL`, `OklabA`, `OklabB` with `is_luminance()`, `is_chroma()` classification
- `PlaneRelationship::Oklab`
- `PlaneDescriptor` — per-plane metadata (semantic, channel_type, subsampling), 4 bytes, Copy
- `PlaneMask` — bitmask with LUMA/CHROMA/ALPHA constants
- `PlaneLayout` — `Interleaved` or `Planar` with factory methods: `oklab()`, `oklab_alpha()`, `ycbcr_444/422/420()`, `rgb()`, `rgba()`, `gray()`
- `MultiPlaneImage` — `PlaneLayout` + `Vec<PixelBuffer>`, index access

This is the right shape. What follows builds on it.

## What's Missing

### 1. Scalar Reference Functions on Existing Enums

**Highest priority. Unlocks everything else.**

The transfer function kernels (`pq_eotf`, `hlg_oetf`, `srgb_eotf_f32`, etc.) exist in `convert.rs` but are private. They need to be the canonical reference implementations that any downstream SIMD code tests against.

```rust
impl TransferFunction {
    /// Encode → linear light (EOTF). THE scalar reference for correctness testing.
    pub fn linearize(&self, v: f32) -> f32;
    /// Linear light → encode (OETF).
    pub fn delinearize(&self, v: f32) -> f32;
}
```

This is just making existing private functions public behind a method on the enum. No new math, no new dependencies.

Similarly, `ColorPrimaries` should expose XYZ matrices:

```rust
impl ColorPrimaries {
    /// 3x3 row-major: linear RGB in these primaries → CIE XYZ (D65 white).
    pub fn to_xyz_matrix(&self) -> [[f32; 3]; 3];
    /// 3x3 row-major: CIE XYZ (D65 white) → linear RGB in these primaries.
    pub fn from_xyz_matrix(&self) -> [[f32; 3]; 3];
}
```

These are compile-time constants derived from the CIE xy chromaticities in the ITU-R specs. The existing `gamut.rs` matrices (BT709_TO_BT2020, etc.) are *products* of these — `to_xyz(A)` composed with `from_xyz(B)` gives the gamut conversion. Having the building blocks means we can compute any conversion, including RGB→LMS for Oklab.

### 2. Oklab Conversion Constants

Pre-computed matrices for the Oklab scatter/gather path:

```rust
/// Oklab conversion utilities.
pub mod oklab {
    use crate::{ColorPrimaries, GamutMatrix};

    /// LMS-from-XYZ matrix (Hunt-Pointer-Estevez variant, Ottosson 2020).
    pub const LMS_FROM_XYZ: GamutMatrix = [ /* ... */ ];
    /// XYZ-from-LMS (inverse).
    pub const XYZ_FROM_LMS: GamutMatrix = [ /* ... */ ];

    /// LMS^(1/3) → Oklab (M2 matrix, universal — does not depend on primaries).
    pub const OKLAB_FROM_LMS_CBRT: GamutMatrix = [ /* ... */ ];
    /// Oklab → LMS^(1/3) (M2 inverse).
    pub const LMS_CBRT_FROM_OKLAB: GamutMatrix = [ /* ... */ ];

    /// Compute the combined RGB→LMS matrix for a given set of primaries.
    ///
    /// This is `LMS_FROM_XYZ × XYZ_from_primaries`. The result is a
    /// compile-time constant for the three supported primaries sets.
    pub fn rgb_to_lms_matrix(primaries: ColorPrimaries) -> GamutMatrix;

    /// Compute the combined LMS→RGB matrix for a given set of primaries.
    pub fn lms_to_rgb_matrix(primaries: ColorPrimaries) -> GamutMatrix;

    /// Scalar reference: cbrt approximation (two Newton-Raphson iterations).
    /// ~22 bits of precision, sufficient for f32.
    pub fn fast_cbrt(x: f32) -> f32;

    /// Scalar reference: linear RGB → Oklab L/a/b.
    pub fn rgb_to_oklab(r: f32, g: f32, b: f32, m1: &GamutMatrix) -> [f32; 3];

    /// Scalar reference: Oklab L/a/b → linear RGB.
    pub fn oklab_to_rgb(l: f32, a: f32, b: f32, m1_inv: &GamutMatrix) -> [f32; 3];
}
```

zenfilters tests its SIMD scatter/gather against `oklab::rgb_to_oklab()`. zenpixels stays small — this is ~100 lines of const arrays and pure scalar math with zero dependencies.

### 3. HDR Reference White

PQ linear values are absolute luminance (0–10,000 nits). Oklab expects relative luminance (~0–1 for SDR). Without normalization, HDR L values will be ~0–49 and break filter parameter ranges.

```rust
impl TransferFunction {
    /// Reference white luminance in nits.
    /// SDR transfers return 1.0 (relative). PQ returns 203.0 (ITU-R BT.2408).
    /// HLG returns 1.0 (scene-referred, already relative).
    pub fn reference_white_nits(&self) -> f32;
}
```

The scatter path divides linear values by this before Oklab conversion. The gather path multiplies by it after. Filters see relative luminance where 1.0 = reference white; HDR highlights exceed 1.0. Filter parameters work identically for SDR and HDR.

This is one method returning a constant. No new types needed — the `ReferenceWhite` enum from the previous draft was overengineered. It's just `transfer.reference_white_nits()`.

### 4. PlaneMask Semantic Lookup

The current `PlaneMask` uses positional bit indices (0=luma, 1=Cb/OklabA, 2=Cr/OklabB, 3=alpha). This works for the factory-method layouts, but breaks if anyone constructs a custom `PlaneLayout` with different ordering.

```rust
impl PlaneLayout {
    /// Build a mask selecting all planes matching a predicate.
    pub fn mask_where(&self, f: impl Fn(PlaneSemantic) -> bool) -> PlaneMask;

    /// Mask of all luminance-like planes.
    pub fn luma_mask(&self) -> PlaneMask { self.mask_where(|s| s.is_luminance()) }

    /// Mask of all chroma planes.
    pub fn chroma_mask(&self) -> PlaneMask { self.mask_where(|s| s.is_chroma()) }

    /// Mask of the alpha plane.
    pub fn alpha_mask(&self) -> PlaneMask { self.mask_where(|s| s.is_alpha()) }
}
```

Now `PlaneMask::LUMA` is a shorthand for the common case, and `layout.luma_mask()` is the correct way for code that handles arbitrary layouts.

## What zenfilters Owns (NOT in zenpixels)

### ChannelAccess

This is a filter concept, not a format concept:

```rust
// In zenfilters, not zenpixels
pub struct ChannelAccess {
    pub reads: PlaneMask,
    pub writes: PlaneMask,
}
```

Each filter declares its `ChannelAccess`. The zenfilters pipeline uses it to skip unchanged planes. zenpixels provides `PlaneMask`; zenfilters composes it.

### SIMD Scatter/Gather Kernels

The SIMD implementation of RGB→planarOklab and back. Uses matrices from `zenpixels::oklab::rgb_to_lms_matrix()`, tests against `zenpixels::oklab::rgb_to_oklab()`, but the archmage `#[arcane]`/`#[rite]` kernels live in zenfilters.

### Multi-Op Coalescing

The previous draft put a `coalesce_operations()` solver in zenpixels. That's wrong — zenpixels doesn't know about filter pipelines. zenfilters knows its operation sequence and uses zenpixels' `PlaneMask` + `PlaneLayout` to decide where to insert scatter/gather boundaries. The logic is:

```
for each adjacent pair of filters:
    if both prefer planar Oklab: keep planar, no boundary
    if layout preference changes: insert gather + scatter
```

This is 20 lines of code in zenfilters, not a generic solver in zenpixels.

### OpCategory Refinements

The existing `OklabSharpen` and `OklabAdjust` are sufficient for zenpixels' path solver. zenfilters can define finer-grained categories internally (L-only vs chroma-only vs mixed) without pushing them into zenpixels. zenpixels just needs to know "this op wants linear f32 Oklab" — which `OklabSharpen`/`OklabAdjust` already express.

## Alpha

Unpremultiply before Oklab, re-premultiply after. `PremulToStraight` and `StraightToPremul` already exist as `ConvertStep` variants. Alpha becomes the 4th plane via `PlaneLayout::oklab_alpha()`, and is passthrough for all Oklab filters (clarity, brilliance, contrast, sharpening, saturation, vibrance).

## Primaries

The scatter/gather path goes `Linear RGB → LMS → LMS^(1/3) → Oklab`. The RGB→LMS matrix (M1) depends on primaries. The LMS→Oklab matrix (M2) is universal. zenpixels provides `oklab::rgb_to_lms_matrix(primaries)` which pre-multiplies `LMS_FROM_XYZ × primaries.to_xyz_matrix()`. The filter pipeline calls this once at init. Three supported primaries = three constant 3x3 matrices.

If source and target primaries differ, the gather uses a different LMS→RGB matrix. Free gamut conversion — no separate pass.

## HDR

The pipeline linearizes first (existing `ConvertStep`s: `PqU16ToLinearF32`, `SrgbU8ToLinearF32`, etc.), then normalizes to reference white (`÷ transfer.reference_white_nits()`), then scatters to planar Oklab. Gather reverses it. Filters never know they're processing HDR.

## D50 Chromatic Adaptation

Only matters for ICC profile workflows. Low priority. When needed, it's a Bradford 3x3 matrix applied after linearization, before scatter. Could be a `ConvertStep` variant or handled externally.

## Gamut Mapping at Output

When the gather produces out-of-gamut values (e.g., saturation boost pushed a P3 color outside sRGB), the options are clip (fast, can shift hue) or perceptual chroma reduction (preserves hue, iteratively reduces Oklch chroma). This belongs in zenfilters' gather kernel, not in zenpixels — it's a per-pixel operation using Oklab math that zenpixels doesn't need to know about.

## Summary

| Item | Where | Size | Priority |
|------|-------|------|----------|
| Scalar `linearize`/`delinearize` on `TransferFunction` | zenpixels | ~20 lines (expose existing private fns) | **Now** |
| XYZ matrices on `ColorPrimaries` | zenpixels `gamut.rs` | ~60 lines (6 const matrices) | **Now** |
| `oklab` module (M1/M2 matrices, scalar reference fns) | zenpixels | ~100 lines | **Now** |
| `reference_white_nits()` on `TransferFunction` | zenpixels | ~10 lines | **Now** |
| `mask_where()` on `PlaneLayout` | zenpixels | ~15 lines | Soon |
| `ChannelAccess` struct | zenfilters | ~10 lines | When building zenfilters |
| SIMD scatter/gather kernels | zenfilters | ~200 lines | When building zenfilters |
| Multi-op coalescing | zenfilters | ~30 lines | When building zenfilters |
| Gamut mapping (chroma reduce) | zenfilters | ~50 lines | When building zenfilters |
| D50 chromatic adaptation | zenpixels | ~30 lines | Low |

Total new code in zenpixels: ~200 lines. Everything else lives in zenfilters.

## Benchmark Context

Planar SIMD separable blur vs interleaved scalar 2D blur at 1920×1080, single-threaded, AVX2+FMA:

| Filter | Interleaved Oklab | Planar Oklab SIMD | Speedup |
|--------|-------------------|-------------------|---------|
| Clarity (σ=10) | 4.71 s | 25.0 ms | 188× |
| Brilliance (σ=10) | 4.74 s | 27.7 ms | 171× |
| Noise Reduction (bilateral, σ=2) | 1.24 s | 1.69 s | 0.73× |
| Oklab round-trip conversion | — | 15.5 ms | — |

A 4-filter chain (clarity + brilliance + contrast + sharpening) at 1080p: **~108ms** total including conversion. Full results: `/home/lilith/work/planar_filter_analysis.md`.
