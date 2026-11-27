# Migrating from `zenpixels` 0.2 / 0.4 to 0.5

`zenpixels 0.5` and `zenpixels-convert 0.5` consolidate the 0.2 and 0.4 APIs around a single unified descriptor (`PixelFormat`), type-erased zero-copy buffers with external storage ownership (`PixelBuf` + `ByteStorage`), zero-allocation borrowed views (`PixelView<'a>` / `PixelViewMut<'a>` taking `&'a PixelFormat`), and stateless `Send + Sync` row/image conversion plans (`RowPlan`, `ImagePlan`, `OutputPlan`).

Both crates remain `#![forbid(unsafe_code)]` and support `no_std + alloc` (disable default `std` feature).

---

## 1. Descriptor & Concept Map

**Important:** In 0.5, the old layout-only `PixelFormat` enum became `SampleLayout`, while the new `Copy + const` `ParametricFormat` struct carries **both layout and parametric color** (replacing color-bearing `PixelDescriptor` in `const` codec tables). Two `ParametricFormat` values can share `SampleLayout::RGB8` while differing in primaries or transfer, and format negotiation compares color as well as layout.

| 0.2 (`e96e4e9c`) / 0.4 (`666814ab`) | 0.5 | Notes |
|---|---|---|
| `PixelFormat::Rgb8` + `ChannelType` + `ChannelLayout` | `SampleLayout::RGB8` (`SampleType`, `ChannelOrder`, `BitPacking`) | Physical byte/sample layout only; cannot authorize a color conversion |
| `PixelDescriptor::RGB8_SRGB` | `ParametricFormat::RGB8_SRGB` (`Copy`/`const` codec tables), `PixelFormat::RGB8_SRGB` (owned), or `&formats::RGB8_SRGB` (`'static` ref) | Both include color; use `PixelFormat::from_layout_assuming_color` only when intentionally assuming default color for a raw layout |
| `PixelDescriptor` + `ColorContext` (ICC) | `PixelFormat { layout, color: ColorSpace::Icc { ... } }` + optional `ColorOrigin` | Arbitrary ICC profiles live in `PixelFormat` (not `Copy` `ParametricFormat`). `PixelFormat::new` / `ParametricFormat::new` are unchecked `const` constructors; call `validate()` or pass to a carrier constructor |
| `TransferFunction` / `ColorPrimaries` | `Transfer` / `Primaries` + `CicpTag` | Lossless H.273 (11 CP, 16 TC, 9 MC) + ICC.1 `cicpType` 44-byte custom-primaries amendment (`CicpTag`, `CustomPrimaries::{ACES_AP0, ACES_CG, PROPHOTO_RGB}`, Annex 1 `to_icc_d50_matrix_f64`); explicit `Transfer::Hlg(HlgMode::Scene \| HlgMode::Display(HlgRendering))` |
| `PixelBuffer<P, S>` | `PixelBuf` | Type-erased, stride-preserving, fallible (`into_tight_rows`), O(1) zero-copy `into_crop`, symmetric `crop_view`/`crop_view_mut`/`sub_rows`/`sub_rows_mut`, direct `row`/`row_mut`, carries `Orientation`; supports external storage (`from_external*`), `new_for_interop`, `new_white`, fast `new_raw_zeroed` + `into_opaque_verified`/`assume_opaque`, and explicit `relabel_color`/`relabel_format`/`reinterpret_bytes` |
| `PixelSlice<'a>` / `PixelSliceMut<'a>` | `PixelView<'a>` / `PixelViewMut<'a>` | Borrowed strided views taking `&'a PixelFormat` (0 heap allocations; borrow `&formats::*` or any local spec; explicit `new_with_format_cow` when owning/sharing a `FormatCow`) |
| `BandedPixelBuffer` (0.4) | `BandedBuf` | Bounded row-band storage (`BandSizing::{MaxBandBytes, RowsPerBand}`, `RowAlign`, `DEFAULT_MAX_BAND_BYTES`) carrying `Orientation` for gigapixel images |
| `MultiPlaneImage` (0.2) / `PlanarBuffer` / `PlanarSlice` (0.4) | `PlanarBuf` / `PlanarView` / `PlanarViewMut` | Always built (no `planar` feature); symmetric `crop_view`/`crop_view_mut`/`crop`/`crop_mut`/`into_crop`/`sub_rows*`; carry `ColorSpace`, typed `MatrixCoefficients` (`matrix_coefficients`), `range_override`/`effective_range`, `alpha_override`/`effective_alpha`, `chroma_siting`, `crop_origin` (`new_with_crop_origin`), `hdr`, `origin`, and `orientation` |
| `AnyPlane` / `PlaneRef<T>` / `PlaneBuf<T>` (0.4) | `PlaneView` / `PlaneViewMut` / `PlaneBuf` | Zero-copy typed codec APIs: tight 4-arg `from_samples(samples, width, height, spec)` and strided 5-arg `from_samples_strided(..., SampleStride(n), spec)`; raw byte `new` / `from_external` take byte stride (`stride_bytes()` returns **bytes**) |
| `CodePlane` | `CodePlane` + `CodePlaneMut` | Unconditional in `zenpixels`; read/write arbitrary bit-packed or P010 codes (`InterleavedCodeLayout`) |
| Dual-layer gain maps | `DualLayerHdrBuf` / `DualLayerHdrView` / `DualLayerHdrBufParts` | Unified base + gain-map + ISO 21496-1 metadata + orientation carrier (`new_with_alternate_color` for `Alternate` space) |
| `ConvertOptions` (0.2/0.4) + `ConversionSpec` (0.4) | `ConvertPolicy` | `Default` is **`lossless()`** (never silently drops alpha/depth/range); explicit `permissive()`, `working_linear()`, `display_sdr(...)`. All planners borrow `&ConvertPolicy` |
| `RowConverter` (0.2) / `ConvertPlan` + `RowConverter` (0.4) | `RowPlan` + `ImagePlan` | `RowPlan` is stateless, `Send + Sync`, zero heap allocation per `convert_row` / `convert_row_at`, takes resolved `Option<Nits>`; `ImagePlan` takes `PeakSource` |
| `buf.to_rgb8()` / `slice.convert_to(...)` | `to_rgb8_if_opaque(&view)?` / `convert_to_buf(&view, &dst, &policy)?` | `to_rgb8_if_opaque` / `to_gray8_if_opaque` use `DiscardIfOpaque` (refuse unproven alpha discard); `convert_to_buf`/`convert_into` default to `PeakSource::Measured` when HDR needs it (`*_with_peak` for explicit `PeakSource`) |
| `OutputPlan` / `adapt_for_encode` | `OutputPlan` / `adapt_for_encode` / `adapt_for_encode_parametric` (`OutputSignaling`, `ResolvedSignaling`) | Canonicalizes unverified `AlphaMode::Opaque`; rejects `MetadataOnly` ICC on structured RGB/Gray; rejects `HlgMode::Display` on `dst` under `CicpOnly`/`Auto` |
| `PluggableCms` + `RowTransformMut` | `CmsBackend` + `CmsTransform` | Stateless `Send + Sync` row transforms; built-in `MoxCms` and `Lcms` |
| `CllMeasure` / `LightLevelHistogram` | `measure_content_light_level`, `measure_histogram`, `measure_max_smoothed` | `LightLevelHistogram::{bins, merge, quantile_nits, to_robust_content_light_level}` + 3-tap smoothed peak |

---

## 2. Cargo Feature Changes

### `zenpixels`

| Feature | 0.2 / 0.4 | 0.5 |
|---|---|---|
| `default` | `["std", "icc"]` | `["std", "icc"]` |
| `planar` | opt-in | **Removed** (planar, banded, and `CodePlane` types are always built) |
| `perceptual` | opt-in | **Removed** (`ColorSpace::Perceptual` is always representable; math stays gated in `zenpixels-convert/perceptual`) |
| `rgb`, `imgref`, `image` | `rgb` / `imgref` / `image` | Unchanged (`imgref` implies `rgb`; `image` uses `image 0.25`) |
| `hash` | opt-in | Portable endian/stride-independent XXH3-64 (`xxhash-rust`) |

### `zenpixels-convert`

| Feature | 0.2 / 0.4 | 0.5 |
|---|---|---|
| `default` | `std`, `icc-db`, `oklab`, `hdr-transfers`, ... | `["std", "fast-transpose", "oklab", "hdr-transfers", "gainmap"]` |
| `icc-db` | default-on | **Default-off** (opt-in for full CICP -> ICC LZ4 bundle; compact P3/Rec.2020 profiles remain built-in) |
| `gainmap` | separate / experimental | **Default-on** (ISO 21496-1 / UltraHDR gain-map reconstruction & extraction) |
| `hdr-experimental` | opt-in | **Removed** (use `hdr-transfers`, `bt2446a`, `gainmap`) |
| `planar` / `introspection` / `zencms-lite` | opt-in | **Removed** |
| `fast` | opt-in | Default-off (registers full SIMD kernels globally; library crates should leave off to preserve dead-stripping) |
| `cms-moxcms` / `cms-lcms` | opt-in | Pure-Rust `MoxCms` and `Lcms` CMS backends |

---

## 3. Policy Presets & Fallibility / Error Contracts

### `ConvertPolicy` Presets (Safe Default!)

Neither pinned 0.2 nor pinned 0.4 implemented `Default` on `ConvertOptions`. In 0.5, `ConvertPolicy::default()` equals **`ConvertPolicy::lossless()`** so callers never silently drop alpha, quantize bit depth, or rescale narrow/full range.

| Preset | `depth` | `alpha` | `gamut` | `range` | `tone_map_hdr_to_sdr` | `allow_icc_substitution` | Notes |
|---|---|---|---|---|---|---|---|
| `lossless()` / `default()` | `Forbid` | `Forbid` | `PreserveExtended` | `Forbid` | `false` | `false` | Loss-permission preset (`allow_approximate: false`); forbids RGB→Gray unless proven grayscale; does not claim universal mathematical invertibility of float transforms |
| `permissive()` | `Round` | `DiscardUnchecked` | `Clip` | `Convert` | `false` | `true` | Explicit opt-in for unchecked lossy conversion |
| `working_linear()` | `Round` | `DiscardIfOpaque` | `PreserveExtended` | `Convert` | `false` | `true` | Defaults `sdr_white` to BT.2408 (`203 cd/m²`) |
| `display_sdr(white, peak)` | `Round` | `CompositeOnto(BLACK)` | `Clip` | `Convert` | `true` | `true` | Enables HDR→SDR tone mapping + HLG 1000-nit reference rendering |

> **`to_rgb8_if_opaque` / `to_gray8_if_opaque`:** Both convenience helpers target full-range sRGB `RGB8` / encoded-Y' `GRAY8` under `ConvertPolicy::permissive().with_alpha(AlphaPolicy::DiscardIfOpaque)`. Unproven straight/premultiplied alpha returns `ConvertError::AlphaLossForbidden`; pass proven-opaque inputs (via `scan_facts` / `into_opaque_verified`) or call `convert_to_buf` with `AlphaPolicy::CompositeOnto` / `DiscardUnchecked`.

### Fallibility Naming & Error Shapes

- **Naming rule:** Standalone fallible operations use ordinary descriptive names (`from_typed`, `from_bytes`, `from_parts`, `from_storage_parts`, `into_tight_rows`, `into_vec_zero_copy`, `into_external`, `make_mut`, `merge`, `project_cicp`, `Nits::new_positive`, `DiffuseWhite::new`, `LightLevelHistogram::new`, etc.). The `try_` prefix is used **only** when the same type already exposes a direct value-returning twin (e.g. `try_clone` / `clone`, `try_make_view_mut` / `as_view_mut`, `try_with_*` / `with_*`, `Cicp::try_from_bytes` / `Cicp::from_bytes_lenient`).
- **Error shape boundary (not every fallible API uses `At<E>`!):**
  - **Located `whereat::At<E>`** on carrier/pipeline boundaries (`PixelBuf`, `PixelView`, `PlanarBuf`, `BandedBuf`, `DualLayerHdrBuf`, `PixelFormat::validate`, `ParametricFormat::validate`, `Cicp::try_from_bytes`, `CicpTag::from_tag_bytes`, `RowPlan`, `ImagePlan`, `OutputPlan`, `apply_orientation*`, `reconstruct_planar_*`, `compact_to_buf`, `gainmap`, `CmsBackend`, `measure_*`, `unpack_code_planes_into`, `TryFrom<DynamicImage> for PixelBuf`):
    ```rust
    match PixelBuf::from_typed(data, w, h, PixelFormat::RGB8_SRGB) {
        Ok(buf) => buf,
        Err(err) => match *err.error() { // or `err.decompose()` to take `(E, trace)`
            BufferError::IncompatibleLayout => { /* ... */ }
            other => return Err(other.into()),
        },
    }
    ```
  - **Bare `E`** on const-capable layout, storage, code-plane, validation, and projection primitives (`LayoutError`, `StorageError`, `SampleEncodingError`, `ExactDepthError`, `CodePlaneError`, `CicpProjectionError`, `IccValidationError`, `CodeMapError`, plus `TryFrom<Cicp> for image::metadata::Cicp` returning bare `BufferError`):
    ```rust
    match spec.color.project_cicp_tag() {
        Ok(tag) => tag,
        Err(CicpProjectionError::HlgDisplayRenderingLost) => { /* ... */ }
        Err(other) => { /* ... */ }
    }
    ```
  - **Recovery shapes:** Plain `from_typed` / `from_bytes` drop the input `Vec` on error. Use `from_typed_recover`, `from_bytes_recover`, `from_parts_recover`, or `from_external*` to recover `(At<BufferError>, original_input)` on failure; zero-copy extractors (`into_vec_zero_copy`, `into_external`, `into_imgvec_zero_copy`) return `Result<T, Self>` returning the untouched buffer on mismatch.

---

## 4. Code Migration Examples

### Constructing Specs & Converting Rows

**Before — `zenpixels 0.2` (`e96e4e9c`):**
```rust
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenpixels_convert::{ConvertOptions, RowConverter};

let src = PixelBuffer::try_new(w, h, PixelDescriptor::RGB8_SRGB)?;
let mut conv = RowConverter::new_explicit(
    src.descriptor(),
    PixelDescriptor::RGBA8_SRGB,
    &ConvertOptions::permissive(),
)?;
conv.try_convert_row(src.as_slice().row(0), dst_row, w)?;
```

**Before — `zenpixels 0.4` (`666814ab` — note `ConvertOptions` had no `permissive()` or `Default`):**
```rust
use zenpixels::{ConvertOptions, DepthPolicy, PixelBuffer, PixelDescriptor};
use zenpixels_convert::{ConversionSpec, ConvertPlan};

let src = PixelBuffer::try_new(w, h, PixelDescriptor::RGB8_SRGB)?;
let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
let plan = ConvertPlan::build(
    src.descriptor().into(),
    PixelDescriptor::RGBA8_SRGB.into(),
    &ConversionSpec::new(opts),
)?;
let mut conv = plan.prepare(w)?;
conv.try_convert_row(src.as_slice().row(0), dst_row, w)?;
```

**After — `zenpixels 0.5`:**
```rust
use zenpixels::{PixelFormat, PixelBuf, formats};
use zenpixels_convert::{ConvertPolicy, RowPlan};

let src = PixelBuf::new(w, h, PixelFormat::RGB8_SRGB)?;
let plan = RowPlan::build(
    src.format(),
    &formats::RGBA8_SRGB,
    &ConvertPolicy::lossless(),
    None, // resolved_source_peak: Option<Nits>
)?;
plan.convert_row(src.row(0), dst_row, w)?;
```

For whole-image conversion, use `convert_to_buf` (defaults to `PeakSource::Measured` only when HDR tone mapping needs a peak; use `convert_to_buf_with_peak` for explicit `PeakSource`) or the safe 8-bit helpers:
```rust
let linear = zenpixels_convert::convert_to_buf(
    &src.as_view(),
    &zenpixels_convert::formats::RGBF32_LINEAR,
    &ConvertPolicy::lossless(),
)?;
let rgb = zenpixels_convert::to_rgb8_if_opaque(&linear.as_view())?;
let gray = zenpixels_convert::to_gray8_if_opaque(&linear.as_view())?;
```

### Zero-Allocation Borrowed Views (`&'a PixelFormat` vs `FormatCow`)

Normal `PixelView` / `PixelViewMut` constructors take `&'a PixelFormat` so view construction **never allocates**:

```rust
use zenpixels::{ParametricFormat, PixelView, FormatCow, formats};

// Zero heap allocation: borrow a static preset (or any local `PixelFormat`):
let view = PixelView::from_bytes(rgb_bytes, w, h, &formats::RGB8_SRGB)?;

// Only when a view must carry an owned/shared spec (e.g. a dynamically built custom spec),
// opt in explicitly via `new_with_format_cow` + `FormatCow`:
let view_cow = PixelView::new_with_format_cow(
    rgb_bytes, w, h, stride, FormatCow::from_parametric(ParametricFormat::RGB8_SRGB), // 0-alloc for presets!
)?;
```

### Zero-Copy External Owned Data into `PixelBuf` / `PlaneBuf`

External buffers (`Vec<T>`, `Box<[T]>`, `Arc<[T]>`, FFI allocations, memory-mapped slices, or custom pools) move into `PixelBuf` or `PlaneBuf` with **zero copies** via `ByteStorage` / `ByteStorageMut`:

```rust
use std::sync::Arc;
use zenpixels::{PixelFormat, PixelBuf};

// Built-in impls exist for Vec<u8>/Vec<u16>/Vec<f32>, Box<[T]>, Arc<[T]>, Arc<Vec<T>>, etc.
let shared: Arc<[u8]> = Arc::from(vec![255u8; 4 * 4 * 3]);
let mut buf = PixelBuf::from_external(shared, 4, 4, 12, PixelFormat::RGB8_SRGB)
    .map_err(|(err, _orig)| err)?;

// Read zero-copy:
let view = buf.as_view();

// Mutate via fallible copy-on-write (only copies if storage is immutable / shared):
let mut mut_view = buf.try_make_view_mut()?;
```

For mutable zero-copy external ownership (in-place mutation without CoW), implement `ByteStorageMut` and construct via `PixelBuf::from_external_mut` or `PlaneBuf::from_external_mut`.

### Sound `AlphaMode::Opaque` Allocation & Fast Fill-Then-Retag

`PixelBuf::new` fills black (setting alpha to channel max on `AlphaMode::Opaque`), and `PixelBuf::new_white` fills reference white. When a decoder or renderer overwrites every pixel and wants to skip the upfront alpha-fill pass:

```rust
use zenpixels::{PixelFormat, PixelBuf};

let mut buf = PixelBuf::new_raw_zeroed(w, h, PixelFormat::RGBA8_SRGB)?; // refuses AlphaMode::Opaque
// ... write all pixels including A = 255 ...
let opaque = buf.into_opaque_verified()?; // scans active pixels itself; or `assume_opaque()` / `into_opaque_filled()`
```

### RGB -> Gray Luma Domain (`Encoded` vs `LinearLight`)

`ConvertPolicy` defaults to `LumaDomain::Encoded`, matching JPEG/video `Y'` and `zenpixels 0.2/0.4` defaults (neutral `(v, v, v)` maps bit-for-bit to `v`, sRGB `(255, 0, 0)` maps to `54`, and `U8` RGB/BGR/RGBA/BGRA -> Gray8 takes the direct SIMD fast path).

To request physical linear-light luminance (`Y = w_r*R + w_g*G + w_b*B`, where sRGB `(255, 0, 0)` maps to `127`):
```rust
use zenpixels_convert::{ConvertPolicy, LumaDomain};

let policy = ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight);
```

### Encoder Output Adaptation (`OutputPlan` / `adapt_for_encode`)

```rust
use zenpixels::{ParametricFormat, formats};
use zenpixels_convert::{
    ConvertPolicy, OutputPlan, OutputSignaling, PeakSource, adapt_for_encode_parametric,
};

// Quick one-shot adaptation over Copy `ParametricFormat` slices (borrows zero-copy when compatible):
let (chosen, ready) = adapt_for_encode_parametric(
    buf.as_view(),
    buf.origin(),
    &[ParametricFormat::RGB8_SRGB, ParametricFormat::RGBA8_SRGB],
    &ConvertPolicy::permissive(),
    PeakSource::Measured,
    OutputSignaling::Auto,
)?;

// Or build a reusable OutputPlan borrowing specs & policy (cloned internally into the plan):
let out_plan = OutputPlan::build(
    buf.format(),
    &formats::RGB8_SRGB,
    &ConvertPolicy::permissive(),
    PeakSource::Measured,
    OutputSignaling::Auto,
)?;
```

### Orientation Verb Table

| Operation | 0.5 API | Contract |
|---|---|---|
| Replace metadata | `with_orientation(self, o)` / `set_orientation(&mut self, o)` | Sets pending orientation to `o` in O(1); no pixel movement |
| Compose metadata | `compose_orientation(self, delta)` (on all 9 carriers/views) | Sets pending orientation to `old.then(delta)` in O(1) (includes flips) |
| Explicit physical transform | `apply_orientation(&view, o)` / `apply_orientation_owned(buf, o)` / `apply_orientation_banded(&bands, o, sizing)` | Physically transforms stored pixels by **exactly `o`** (does not compose pending orientation); output carries `Orientation::Identity` (`apply_orientation_owned(_, Identity)` stays zero-copy) |
| Explicit into caller sink | `apply_orientation_into(&view, o, &mut dst)` / `apply_orientation_banded_into(...)` | Writes transformed pixels into caller-owned destination with zero allocation (does not change `dst.orientation()`) |
| Bake pending orientation | `bake_orientation(&view)` / `bake_orientation_owned(buf)` / `bake_orientation_banded(&bands, sizing)` | Consumes pending `orientation()`, transforms pixels, emits `Orientation::Identity` (owned fast path is zero-copy on `Identity`) |
| Dual-layer physical bake | `gainmap::bake_dual_layer_orientation(&view)` / `gainmap::bake_dual_layer_orientation_after(&view, delta)` | Transforms both base & gain map by `pending` (or `pending.then(delta)`), emitting `Orientation::Identity` |

### Pluggable CMS (`MoxCms` and `Lcms`)

```rust
use std::sync::Arc;
use zenpixels_convert::{ConvertPolicy, Lcms, MoxCms};

// Pure-Rust MoxCms (fast default; declines AbsoluteColorimetric, float D2B*/B2D* MPE,
// and unequal adjacent RGB TRCs so callers can fall back explicitly):
let mox_pol = ConvertPolicy::permissive().with_cms(Arc::new(MoxCms));

// Pure-Rust Lcms (full ICC.1 support including AbsoluteColorimetric, float MPE, Lab/XYZ, CMYK):
let lcms_pol = ConvertPolicy::permissive().with_cms(Arc::new(Lcms));
```

### Zero-Copy Planar Reconstruction, Compaction & Orientation

```rust
use zenpixels::{
    BandSizing, ChromaSiting, ColorSpace, MatrixCoefficients, PlanarView, PlaneFormat, PlaneRole,
    PlaneView, SampleLayout, SampleStride, SampleType,
};
use zenpixels_convert::{
    AlphaCompaction, PlanarReconstructionOptions, apply_orientation_banded,
    apply_orientation_banded_into, bake_orientation_owned, compact_in_place,
    reconstruct_planar_f32, reconstruct_planar_to_buf,
};

// Borrow codec-native typed planes zero-copy: tight 4-arg `from_samples(samples, w, h, spec)` or
// strided 5-arg `from_samples_strided(samples, w, h, SampleStride(stride_samples), spec)`.
// `plane.code_u32(x, y, 0)` takes component index `0` for 1-channel planes:
let py = PlaneView::from_samples_strided(y_slice, w, h, SampleStride(y_stride), PlaneFormat::new(PlaneRole::Y, SampleType::U8))?;
let pcb = PlaneView::from_samples_strided(cb_slice, cw, ch, SampleStride(c_stride), PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2))?;
let pcr = PlaneView::from_samples_strided(cr_slice, cw, ch, SampleStride(c_stride), PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2))?;
let planar = PlanarView::new(w, h, &[py, pcb, pcr])?
    .with_color(ColorSpace::SRGB)
    .with_matrix_coefficients(MatrixCoefficients::BT709)
    .with_chroma_siting(ChromaSiting::Center);

// Reconstruct directly to RGB8/RGBA8/BGR8/RGB16/RGBF32 with siting-exact Bilinear upsampling:
let rgb8 = reconstruct_planar_to_buf(&planar, SampleLayout::RGB8, &PlanarReconstructionOptions::bilinear())?;
let rgbf32 = reconstruct_planar_f32(&planar, &PlanarReconstructionOptions::bilinear())?;

// Bake pending buffer orientation (zero-copy when already Identity) or rotate banded buffers:
let mut display_buf = bake_orientation_owned(rgb8)?;
compact_in_place(&mut display_buf, AlphaCompaction::DropUnusedLanes)?;
let oriented = apply_orientation_banded(&src_bands, orient, BandSizing::DEFAULT)?;
apply_orientation_banded_into(&src_bands, orient, &mut dst_bands)?;
```

### Additional 0.5 Safety & Ergonomics Notes

- **Opaque alpha canonicalization:** Converting unverified `RGBA8` (`AlphaMode::Straight` or unverified `Opaque`, including `RGB10A2`) to `AlphaMode::Opaque` canonicalizes alpha to channel max instead of bypassing; it stays zero-copy when `view.facts().all_opaque` or `policy.source_facts.all_opaque` (set via `ConvertPolicy::with_source_facts`) is proven.
- **`MetadataOnly` ICC profiles:** Rejected when targeting structured `ColorSpace::Rgb` / `ColorSpace::Gray` without a CMS, while remaining valid on explicit `ColorSpace::Icc`.
- **HLG display signaling:** `OutputPlan` refuses `Transfer::Hlg(HlgMode::Display(_))` on `dst` under `OutputSignaling::CicpOnly` and `OutputSignaling::Auto` (`ConvertError::OutputSignalingUnavailable`) because CICP cannot signal display rendering parameters (converting *from* display HLG to sRGB/PQ succeeds).
- **`CicpTag` 44-byte custom primaries:** Encodes `(x, y)` in signed `s15Fixed16Number` (`1 / 65536`); compare tag bytes or `CustomPrimaries::to_s15fixed16()` after wire round-trip rather than raw decimal `f32` presets.
- **`DualLayerHdrBuf`:** Use `DualLayerHdrBuf::new_with_alternate_color(base, gain_map, metadata, color)` for `GainMapColorSpace::Alternate` (inherits layer orientation). Rebuilding via `DualLayerHdrBuf::from_parts(DualLayerHdrBufParts)` requires `base.orientation()` and `gain_map.orientation()` to match `parts.orientation` (or both `Identity`). Gain-map conversion/extraction APIs are namespaced under `zenpixels_convert::gainmap::*`: apply via `gainmap::apply_gain_map_with` / `gainmap::apply_gain_map_oriented` with `&zenpixels_convert::Unstoppable` (re-exported from `enough`) and explicit `gainmap::GainMapApplyConfig::from_headroom_stops` (stops) or `from_display_boost_linear` (linear multiplier).
- **Content compaction (`compact`):** Free functions `analyze_compaction(&view)` → `CompactionReport { requires_alpha, requires_chroma, requires_u16_precision }` (`candidate_format`), `compact_to_buf(&view)`, and fallible `compact_in_place(&mut buf, AlphaCompaction::{RetagOpaque, DropUnusedLanes}) -> Result<bool, At<ConvertError>>`. `RetagOpaque` retags provably opaque alpha zero-move (while still compacting depth/chroma in-place); `DropUnusedLanes` also drops unused alpha/padding lanes in place.
- **`Bt2446A` / `Bt2446APipelineManaged`:** Use `Bt2446A::pipeline_managed()` (returns `Bt2446APipelineManaged`) for planner-resolved peaks or `Bt2446A::with_peaks(Nits::new_positive(hdr).unwrap(), Nits::new_positive(sdr).unwrap())` for baked typed peaks (`map_rgb_bt2020_peak_normalized` / `map_strip_bt2020_peak_normalized`).
- **`imgref` & `image` interop:** `imgref::ImgVec::new(pixels, width, height)` takes `(pixels, w, h)` and indexes via `img[(x_usize, y_usize)]`; `PixelView` exposes `height()` and `stride_bytes()`; explicit-color `image` adapters are named `from_dynamic_image_with_color` / `from_rgb8_image_with_color` / `from_rgba8_image_with_color` and typed `rgb` crate adapters `from_pixels_with_color`.
