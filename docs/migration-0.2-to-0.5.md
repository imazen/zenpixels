# Migrating from `zenpixels` 0.2 / 0.4 to 0.5

`zenpixels 0.5` and `zenpixels-convert 0.5` are a clean-sheet consolidation of the 0.2/0.4 APIs around a single unified descriptor (`ImageSpec`), type-erased zero-copy buffers with external storage ownership (`PixelBuf` + `ByteStorage`), and stateless `Send + Sync` row/image conversion plans (`RowPlan`, `ImagePlan`, `OutputPlan`).

Both crates remain `#![forbid(unsafe_code)]` and `no_std + alloc` by default.

---

## 1. Quick Type & Concept Map

| 0.2 / 0.4 | 0.5 | Notes |
|---|---|---|
| `PixelFormat` + `ChannelType` + `ChannelLayout` | `SampleLayout` (`SampleType`, `ChannelOrder`, `BitPacking`) | Orthogonal physical layout; presets like `SampleLayout::RGB8`, `RGBAF32`, `P010` |
| `PixelDescriptor` + `ColorContext` | `ImageSpec` (`SampleLayout` + `ColorSpace`) + optional `ColorOrigin` | Color authority lives on `ColorSpace` (`Rgb`, `Gray`, `Oklab`, `Xyz`, `Perceptual`, `Icc`, `Cmyk`) |
| `TransferFunction` / `ColorPrimaries` | `Transfer` / `Primaries` | `Transfer::Hlg(HlgMode::Scene \| HlgMode::Display(HlgRendering))` makes HLG display rendering explicit |
| `PixelBuffer<P, S>` | `PixelBuf` | Type-erased, stride-preserving, fallible; supports external storage via `PixelBuf::from_external*` |
| `PixelSlice<'a>` / `PixelSliceMut<'a>` | `PixelView<'a>` / `PixelViewMut<'a>` | Borrowed strided views with `SampleFacts` and `HdrMetadata` |
| `BandedPixelBuffer` | `BandedBuf` | Bounded row-band storage (`BandLayout`, `RowAlign`) for gigapixel images |
| `PlanarBuffer` / `PlanarSlice` / `PlanarSliceMut` | `PlanarBuf` / `PlanarView` / `PlanarViewMut` | Always available (no `planar` Cargo feature); carry optional `ColorSpace`, `ycbcr_matrix_code`, `range`, `alpha`, `chroma_siting` |
| `AnyPlane` / `PlaneRef` / `PlaneBuf` | `PlaneView` / `PlaneViewMut` / `OwnedPlane` | Unified byte-addressed strided planes with `PlaneSpec` (`PlaneRole`, `SampleType`, subsampling) |
| `CodePlane` | `CodePlane` + `CodePlaneMut` | Unconditional in `zenpixels`; read/write arbitrary bit-packed or P010 codes |
| `ConvertOptions` + `ConversionSpec` | `ConvertPolicy` | Presets: `lossless()`, `permissive()`, `working_linear()`, `display_sdr()` |
| `ConvertPlan` + `RowConverter` | `RowPlan` + `ImagePlan` | `RowPlan` is stateless, `Send + Sync`, and allocates zero heap bytes per `convert_row` |
| `OutputPlan` / `adapt_for_encode` | `OutputPlan` / `adapt_for_encode` | Canonicalizes unverified `AlphaMode::Opaque`; rejects `MetadataOnly` ICC on structured RGB/Gray |
| `PluggableCms` + `RowTransformMut` | `CmsBackend` + `CmsTransform` | Stateless `Send + Sync` row transforms; built-in `MoxCms` and `Lcms` |
| `CllMeasure` / `LightLevelHistogram` | `measure_cll`, `measure_histogram`, `measure_max_smoothed` | `LightLevelHistogram::{bins, try_merge}` + 3-tap smoothed peak |

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

## 3. Code Migration Examples

### Constructing Specs & Converting Rows

**Before (0.2 / 0.4):**
```rust
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenpixels_convert::{ConvertOptions, ConvertPlan, ConversionSpec};

let src = PixelBuffer::try_new(w, h, PixelDescriptor::RGB8_SRGB)?;
let plan = ConvertPlan::build(
    src.descriptor().into(),
    PixelDescriptor::RGBA8_SRGB.into(),
    &ConversionSpec::new(ConvertOptions::permissive()),
)?;
let mut conv = plan.prepare(w)?;
conv.try_convert_row(src.as_slice().row(0), dst_row, w)?;
```

**After (0.5):**
```rust
use zenpixels::{ImageSpec, PixelBuf};
use zenpixels_convert::{ConvertPolicy, RowPlan};

let src = PixelBuf::new(w, h, ImageSpec::rgb8_srgb())?;
let plan = RowPlan::build(
    src.spec().clone(),
    ImageSpec::rgba8_srgb(),
    &ConvertPolicy::permissive(),
    None,
)?;
plan.convert_row(src.as_view().row(0), dst_row, w, 0)?;
```

For common 8-bit outputs, single-call helpers are available directly on `zenpixels_convert`:
```rust
let rgb = zenpixels_convert::to_rgb8(&src.as_view(), &ConvertPolicy::permissive())?;
let gray = zenpixels_convert::to_gray8(&src.as_view(), &ConvertPolicy::permissive())?;
```

### Zero-Copy External Owned Data into `PixelBuf` / `OwnedPlane`

In 0.5, external buffers (`Vec<T>`, `Box<[T]>`, `Arc<[T]>`, FFI allocations, memory-mapped slices, or custom pools) move into `PixelBuf` or `OwnedPlane` with **zero copies** via `ByteStorage` / `ByteStorageMut`:

```rust
use std::sync::Arc;
use zenpixels::{ByteStorage, ImageSpec, PixelBuf};

// Built-in impls exist for Vec<u8>/Vec<u16>/Vec<f32>, Box<[T]>, Arc<[T]>, Arc<Vec<T>>, etc.
let shared: Arc<[u8]> = Arc::from(vec![255u8; 4 * 4 * 3]);
let mut buf = PixelBuf::from_external(shared, 4, 4, 12, ImageSpec::rgb8_srgb())?;

// Read zero-copy:
let view = buf.as_view();

// Mutate via fallible copy-on-write (only copies if storage is immutable / shared):
let mut_view = buf.try_make_mut()?;
```

For mutable zero-copy external ownership (in-place mutation without CoW), implement `ByteStorageMut` and construct via `PixelBuf::from_external_mut` or `OwnedPlane::from_external_mut`.

### RGB -> Gray Luma Domain (`Encoded` vs `LinearLight`)

`ConvertPolicy` defaults to `LumaDomain::Encoded`, matching JPEG/video `Y'` and `zenpixels 0.2/0.4` defaults (neutral `(v, v, v)` maps bit-for-bit to `v`, sRGB `(255, 0, 0)` maps to `54`, and `U8` RGB/BGR/RGBA/BGRA -> Gray8 takes the direct SIMD fast path).

To request physical linear-light luminance (`Y = w_r*R + w_g*G + w_b*B`, where sRGB `(255, 0, 0)` maps to `127`):
```rust
use zenpixels_convert::{ConvertPolicy, LumaDomain};

let policy = ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight);
```

### Encoder Output Adaptation (`OutputPlan` / `adapt_for_encode`)

```rust
use zenpixels::{ImageSpec, PixelCow};
use zenpixels_convert::{ConvertPolicy, OutputPlan, adapt_for_encode};

// Quick one-shot adaptation (borrows zero-copy when already compatible):
let ready: PixelCow<'_> = adapt_for_encode(
    &view,
    &[ImageSpec::rgb8_srgb(), ImageSpec::rgba8_srgb()],
    &ConvertPolicy::permissive(),
)?;

// Or build once and reuse:
let out_plan = OutputPlan::build(
    view.spec().clone(),
    ImageSpec::rgb8_srgb(),
    &ConvertPolicy::permissive(),
    None,
)?;
```

Notes on 0.5 safety fixes:
- Converting unverified `RGBA8` (`AlphaMode::Straight`) to `AlphaMode::Opaque` canonicalizes alpha to `255` instead of bypassing; it remains zero-copy whenever `view.facts().all_opaque` or `policy.source_facts.all_opaque` is proven.
- `MetadataOnly` ICC profiles (profiles recognized for identification but not safe for matrix+TRC substitution) are rejected when targeting structured `ColorSpace::Rgb` / `ColorSpace::Gray` without a CMS, while remaining valid on explicit `ColorSpace::Icc`.

### Pluggable CMS (`MoxCms` and `Lcms`)

```rust
use std::sync::Arc;
use zenpixels_convert::{ConvertPolicy, Lcms, MoxCms, RowPlan};

// Pure-Rust MoxCms (fast default; declines AbsoluteColorimetric, float D2B*/B2D* MPE,
// and unequal adjacent RGB TRCs so callers can fall back explicitly):
let mox_pol = ConvertPolicy::permissive().with_cms(Arc::new(MoxCms));

// Pure-Rust Lcms (full ICC.1 support including AbsoluteColorimetric, float MPE, Lab/XYZ, CMYK):
let lcms_pol = ConvertPolicy::permissive().with_cms(Arc::new(Lcms));
```

### Planar YCbCr & Banded Orientation

```rust
// Planar YCbCr -> packed RGB/RGBA F32:
let rgb = zenpixels_convert::ycbcr_to_rgb_from_planar(
    &planar_view,
    zenpixels_convert::YcbcrMatrix::from_cicp_code(1).unwrap(),
    zenpixels::SignalRange::Narrow,
    zenpixels::ColorSpace::SRGB_LINEAR,
)?;

// Gigapixel banded orientation (allocating or zero-alloc into pre-allocated bands):
let oriented = zenpixels_convert::orient_banded(&src_bands, orient)?;
zenpixels_convert::apply_orientation_banded_into(&src_bands, orient, &mut dst_bands)?;
```
