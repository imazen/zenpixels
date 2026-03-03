# Planar Oklab Support for zenfilters

## The Ask

`ConvertPlan` learns planar Oklab as a target format. A filter author calls one function, gets f32 slices back. Everything else — linearization, primaries, reference white, alpha — is internal to the plan.

```rust
// This is the entire external API surface for zenfilters:
let plan = ConvertPlan::to_planar_oklab(&source_descriptor)?;
let planes: PlanarOklab = plan.convert(&pixel_data);
// ... filters operate on planes.l, planes.a, planes.b ...
let plan_back = ConvertPlan::from_planar_oklab(&target_descriptor)?;
let pixels = plan_back.convert(&planes);
```

Filters never see `ColorPrimaries`, `TransferFunction`, or `ConvertStep`. They get `&mut [f32]`.

## What ConvertPlan Does Today

For interleaved→interleaved conversions, `ConvertPlan::new(from, to)` already builds a chain of `ConvertStep`s:

1. Layout change (add/drop alpha, gray→rgb, swizzle)
2. Transfer function (sRGB EOTF, PQ EOTF, HLG EOTF — all implemented as private fns in convert.rs)
3. Depth change (u8→f32, u16→f32, etc.)
4. Alpha mode (premul→straight, straight→premul)

It composes these into a `Vec<ConvertStep>`, applies them per-row. No allocation per-row.

## What ConvertPlan Needs to Learn

### The Scatter Path (interleaved → planar Oklab)

Internally, the plan assembles these steps:

```
1. Existing steps: layout normalize + linearize + depth→f32 + unpremultiply
2. NEW: ÷ reference_white (1.0 for SDR, 203.0 for PQ)
3. NEW: 3×3 matrix multiply: linear RGB → LMS (matrix depends on source primaries)
4. NEW: cbrt per component: LMS → LMS^(1/3)
5. NEW: 3×3 matrix multiply: LMS^(1/3) → Oklab L/a/b
6. NEW: scatter: deinterleave L/a/b into separate planes
   (alpha goes to its own plane if present)
```

Steps 1 already exist. Steps 2–6 are new, but 2 is a multiply, 3 and 5 are the same matrix-multiply pattern already in `gamut.rs`, 4 is a scalar function, and 6 is a memcpy-with-stride.

### The Gather Path (planar Oklab → interleaved)

The reverse:

```
1. NEW: gather: interleave L/a/b from planes (+ alpha if present)
2. NEW: 3×3 matrix: Oklab → LMS^(1/3)
3. NEW: cube per component: LMS^(1/3) → LMS
4. NEW: 3×3 matrix: LMS → linear RGB (matrix depends on target primaries)
5. NEW: × reference_white
6. Existing steps: premultiply + f32→depth + delinearize + layout
```

### What Internal Building Blocks Are Needed

**A. XYZ matrices per primaries set.**

The existing gamut matrices (e.g., `BT709_TO_BT2020`) are pre-multiplied products. ConvertPlan needs the building blocks:

```rust
// In gamut.rs — const arrays, ~60 lines
impl ColorPrimaries {
    pub fn to_xyz_matrix(&self) -> &'static GamutMatrix;
    pub fn from_xyz_matrix(&self) -> &'static GamutMatrix;
}
```

Six 3×3 const matrices (to_xyz and from_xyz for BT.709, Display P3, BT.2020). Derived from the CIE xy chromaticities in ITU-R BT.709, BT.2020, and SMPTE RP 431-2 (P3). D65 white point for all three.

These also let us *verify* the existing gamut matrices: `BT709_TO_BT2020` should equal `from_xyz(BT2020) × to_xyz(BT709)`.

**B. Oklab-specific matrices and scalar math.**

```rust
// New module: oklab.rs, ~80 lines
// LMS ← XYZ (Hunt-Pointer-Estevez, Ottosson 2020)
pub const LMS_FROM_XYZ: GamutMatrix = [/* ... */];
pub const XYZ_FROM_LMS: GamutMatrix = [/* ... */];

// Oklab ← LMS^(1/3) (M2, universal)
pub const OKLAB_FROM_LMS_CBRT: GamutMatrix = [/* ... */];
pub const LMS_CBRT_FROM_OKLAB: GamutMatrix = [/* ... */];

// Pre-multiplied convenience: RGB → LMS for each primaries set
pub fn rgb_to_lms(primaries: ColorPrimaries) -> GamutMatrix {
    mat3_mul(&LMS_FROM_XYZ, &primaries.to_xyz_matrix())
}
pub fn lms_to_rgb(primaries: ColorPrimaries) -> GamutMatrix {
    mat3_mul(&primaries.from_xyz_matrix(), &XYZ_FROM_LMS)
}

// Scalar reference (for testing SIMD implementations against)
pub fn fast_cbrt(x: f32) -> f32;
pub fn rgb_to_oklab(r: f32, g: f32, b: f32, rgb_to_lms: &GamutMatrix) -> [f32; 3];
pub fn oklab_to_rgb(l: f32, a: f32, b: f32, lms_to_rgb: &GamutMatrix) -> [f32; 3];
```

**C. Reference white accessor.**

```rust
// On existing TransferFunction enum, ~10 lines
impl TransferFunction {
    /// Reference white in nits. SDR returns 1.0, PQ returns 203.0.
    pub fn reference_white_nits(&self) -> f32;
}
```

**D. Public scalar EOTF/OETF.**

The `pq_eotf`, `hlg_oetf`, etc. already exist as private functions in `convert.rs`. Make them public methods on `TransferFunction`:

```rust
impl TransferFunction {
    /// Scalar EOTF: encoded signal → linear light.
    pub fn linearize(&self, v: f32) -> f32;
    /// Scalar OETF: linear light → encoded signal.
    pub fn delinearize(&self, v: f32) -> f32;
}
```

These serve as test oracles. Any SIMD implementation (in zenpixels or downstream) validates against them.

**E. PlaneLayout semantic mask lookup.**

```rust
// ~15 lines on existing PlaneLayout
impl PlaneLayout {
    pub fn mask_where(&self, f: impl Fn(PlaneSemantic) -> bool) -> PlaneMask;
}
```

### New ConvertStep Variants

```rust
// Internal to convert.rs
enum ConvertStep {
    // ... existing variants ...

    /// Multiply all channels by a constant (for reference white normalization).
    ScaleF32 { factor: f32 },

    /// 3×3 matrix on RGB channels (for RGB→LMS or LMS→RGB).
    /// Alpha preserved. Operates on f32 data only.
    MatrixF32 { m: [[f32; 3]; 3] },

    /// Per-component cbrt (LMS → LMS^(1/3)).
    CbrtF32,

    /// Per-component cube (LMS^(1/3) → LMS).
    CubeF32,

    /// Deinterleave 3 or 4 channels into separate plane buffers.
    ScatterToPlanes,

    /// Interleave separate plane buffers into 3 or 4 channels.
    GatherFromPlanes,
}
```

`MatrixF32` is the same operation as `apply_matrix_row_rgba_f32` in gamut.rs, just as a ConvertStep. `ScaleF32` is a trivial per-element multiply. `CbrtF32`/`CubeF32` are per-element scalar ops (SIMD-friendly). `ScatterToPlanes`/`GatherFromPlanes` are the layout change.

These are all *internal* (`pub(crate)`). The public API is just `ConvertPlan::to_planar_oklab()` and `ConvertPlan::from_planar_oklab()`.

### ConvertPlan Output Type

Today `ConvertPlan` converts row-by-row into a same-layout destination buffer. For planar output, it needs to write into multiple separate buffers. Two options:

**Option A: ConvertPlan writes into a MultiPlaneImage.**

```rust
impl ConvertPlan {
    /// Execute the plan, writing planar output into a MultiPlaneImage.
    pub fn convert_to_planar(&self, src_row: &[u8], dst: &mut MultiPlaneImage, y: usize);
    /// Execute the plan, reading planar input.
    pub fn convert_from_planar(&self, src: &MultiPlaneImage, y: usize, dst_row: &mut [u8]);
}
```

**Option B: ConvertPlan produces a PlanarConvertPlan wrapper.**

```rust
pub struct PlanarConvertPlan {
    interleaved_steps: Vec<ConvertStep>,  // linearize, depth, unpremul
    scatter_steps: Vec<ConvertStep>,       // matrix, cbrt, matrix, scatter
    layout: PlaneLayout,
}
```

Option A is simpler. The `MultiPlaneImage` already exists and owns the plane buffers. The plan just needs to know "step N writes to planes instead of interleaved output."

## What This Doesn't Include (And Why)

**Multi-op coalescing.** Not zenpixels' problem. The caller decides when to scatter and gather. If zenfilters wants to batch 4 filters between one scatter/gather pair, it just... does that. No solver needed. Call `convert_to_planar`, run 4 filters, call `convert_from_planar`. The pipeline author can see the filter list and trivially group adjacent Oklab ops.

**OpCategory refinements.** The existing `OklabSharpen`/`OklabAdjust` already say "needs linear f32, Oklab space." That's enough for the path solver. Whether a filter is L-only or chroma-only is a filter implementation detail, not a format negotiation concern.

**ChannelAccess / read-write masks.** A filter-level optimization. The filter knows which planes it touches. It skips the ones it doesn't. zenpixels provides `PlaneMask` as the vocabulary type; the filter uses it however it wants.

**Gamut mapping.** A per-pixel operation that happens after gather when target gamut is narrower than source. It uses Oklab math (reduce chroma until in-gamut). It can live anywhere — in the gather step, in a post-processing filter, or as a ConvertStep. Not blocking.

**D50 chromatic adaptation.** Only needed for ICC-profiled input. `MatrixF32` step with a Bradford matrix. Add it when ICC support ships.

**JzAzBz / ICtCp.** Structurally identical to Oklab (different matrices, PQ nonlinearity instead of cbrt). If needed later, add `CbrtF32` → `PerceptualNonlinearityF32 { kind: Cbrt | Pq }` and different matrix constants. The architecture doesn't change.

## Size

| Change | Where | Lines |
|--------|-------|-------|
| XYZ matrices on `ColorPrimaries` | gamut.rs | ~60 |
| `oklab.rs` module (matrices + scalar ref) | new file | ~80 |
| `reference_white_nits()` | descriptor.rs | ~10 |
| `linearize()`/`delinearize()` on `TransferFunction` | convert.rs | ~20 (expose existing) |
| `mask_where()` on `PlaneLayout` | descriptor.rs | ~15 |
| New ConvertStep variants + plan construction | convert.rs | ~150 |
| Tests | various | ~100 |
| **Total** | | **~435 lines** |

No new dependencies. No new public types (except `oklab` module contents). No architecture changes. ConvertPlan gets a new target format, same as if we were adding YCbCr planar output.

## Benchmark Context

1920×1080, single-threaded, AVX2+FMA. Planar SIMD separable blur vs interleaved scalar 2D blur, both pre-converted to Oklab:

| | Interleaved | Planar SIMD | Speedup |
|---|---|---|---|
| Clarity (σ=10) | 4.71 s | 25 ms | 188× |
| Brilliance (σ=10) | 4.74 s | 28 ms | 171× |
| Bilateral NR (σ=2) | 1.24 s | 1.69 s | 0.73× |
| Oklab conversion round-trip | — | 15.5 ms | — |

4-filter chain at 1080p: **~108ms** total including conversion.

Full analysis: `/home/lilith/work/planar_filter_analysis.md`
