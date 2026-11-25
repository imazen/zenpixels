# `zenpixels` 0.5 User Guide & Cookbook

A task-oriented guide to storing, borrowing, and converting pixels in Rust with `zenpixels` and `zenpixels-convert`. Both crates are `#![forbid(unsafe_code)]` and support `no_std + alloc` (disable default `std` features for `no_std`).

If you are upgrading from `zenpixels` 0.2 or 0.4, see [migration-0.2-to-0.5.md](migration-0.2-to-0.5.md).

---

## Your First 60 Seconds (Wrap → Crop → Convert)

```rust
use zenpixels_convert::{ImageSpec, PixelBuf, to_rgba8};

// 1. Own tightly packed sRGB RGB8 pixels (annotated literals so Rust infers u8):
let img = PixelBuf::from_typed(
    vec![255u8, 0, 0,  0, 255, 0,  0, 0, 255,  255, 255, 255],
    2, 2,
    ImageSpec::RGB8_SRGB,
)?;

// 2. Borrow a zero-copy rectangular crop (keeps parent stride = 6 bytes):
let right_col = img.crop_view(1, 0, 1, 2)?;
assert_eq!(right_col.row(0), &[0, 255, 0]);

// 3. Convert to straight-alpha RGBA8 in one call:
let rgba = to_rgba8(&right_col)?;
assert_eq!(rgba.as_view().row(0), &[0, 255, 0, 255]);
```

---

## 1. The 4-Type Mental Model

You only need four types on day one:

| Type | What it is | Owns memory? | `Copy`? | When to use |
|---|---|---|---|---|
| `ImageSpec` | Full image description: `SampleLayout` (bytes/order/depth) + `ColorSpace` (RGB/Gray/Oklab/XYZ/CMYK/ICC/Untagged) | Only if `ColorSpace::Icc` (`Arc<[u8]>`) | No | Describing any buffer or conversion endpoint |
| `PixelBuf` | Owned, stride-preserving, alignment-exact 2D pixel buffer (+ `Orientation`, `HdrMetadata`, `ColorOrigin`, `SampleFacts`) | Yes (`Vec`, aligned, or external `Arc`/`Box`/FFI) | No | Holding decoded or converted images |
| `PixelView<'a>` / `PixelViewMut<'a>` | Borrowed strided 2D view or rectangular crop (zero copies) | No | No | Passing images or tiles into functions |
| `PixelFormat` | Compact `SampleLayout` + `ParametricColor` (no ICC) | No | **Yes (`const`)** | `const` codec capability tables (`&[PixelFormat]`) |

**Presets on `ImageSpec`:**
- Owned values for `PixelBuf` constructors: `ImageSpec::RGB8_SRGB`, `ImageSpec::RGBA8_SRGB`, `ImageSpec::RGBF32_LINEAR`, `ImageSpec::GRAY8_SRGB`, `ImageSpec::RGB16_BT2100_PQ`, `ImageSpec::RGB16_BT2100_HLG_1000`, etc. (also available as `ImageSpec::rgb8_srgb()`).
- Borrowed `'static` refs for zero-allocation `PixelView` / `PixelViewMut` constructors: `ImageSpec::rgb8_srgb_ref()`, `ImageSpec::rgba8_srgb_ref()`, `ImageSpec::rgbf32_linear_ref()`, `ImageSpec::gray8_srgb_ref()`, etc.
  *(Because `ImageSpec` implements `Drop` for its `Icc` variant, `&ImageSpec::RGB8_SRGB` creates a local temporary; `ImageSpec::rgb8_srgb_ref()` returns a true `&'static ImageSpec` with zero heap allocation.)*
- Note on defaults: `ImageSpec::RGBA8_SRGB` and `BGRA8_SRGB` default to `AlphaMode::Straight`; modify any spec with `.with_alpha(...)`, `.with_primaries(...)`, `.with_transfer(...)`, `.with_range(...)`, `.with_white(...)` (or checked `.try_with_*()`).
- Note on validation: `ImageSpec::new(layout, color)` and `PixelFormat::new(layout, color)` are unchecked `const` constructors; call `.validate()` or pass them to any buffer/plan constructor (which validates automatically).

**Auxiliary metadata carried on buffers/views:**
- `Orientation` — EXIF orientation (`Identity`, `Rotate90`, `FlipH`, etc.), carried directly on all image/planar/banded/dual-layer buffers and views.
- `ColorOrigin` — Provenance of the file's original color signaling (`ColorOrigin::assumed()`, `from_cicp`, `from_icc`, `from_icc_and_cicp`), separate from the pixel's active `ColorSpace`.
- `HdrMetadata` — Optional `ContentLightLevel` (MaxCLL / MaxFALL) and ST 2086 `MasteringDisplay`.
- `SampleFacts` — Content facts (`all_opaque`, `all_grayscale`, `binary_alpha`, `float`, `depth`), either measured by `scan_facts` / seeded by freshly filled buffers (`PixelBuf::new`, `new_white`, `new_zeroed`), or asserted by callers via `set_facts` / `with_facts` / `with_source_facts`. Automatically cleared whenever mutable pixel access is taken.

---

## 2. Error Handling (3 Exact Shapes)

Not every fallible API returns `whereat::At<E>`. The boundary is consistent:

1. **Located `whereat::At<E>`** — carrier constructors, plans, conversions, orientation, YCbCr, gain maps, CMS, measurements, and `ImageSpec::validate` / `Cicp::try_from_bytes` / `CicpTag::from_tag_bytes`:
   ```rust
   use zenpixels::{BufferError, ImageSpec, PixelBuf};

   match PixelBuf::from_bytes(bytes, w, h, ImageSpec::RGB8_SRGB) {
       Ok(buf) => buf,
       Err(err) => match *err.error() { // or `err.into_inner()`
           BufferError::InsufficientData => { /* ... */ }
           BufferError::UnverifiedOpaqueAlpha => { /* ... */ }
           other => return Err(other.into()),
       },
   }
   ```
2. **Bare `E`** — const-capable layout, storage, code-plane, validation, and CICP projection primitives (`LayoutError`, `StorageError`, `SampleEncodingError`, `ExactDepthError`, `CodePlaneError`, `CicpProjectionError`, `IccValidationError`, `CodeMapError`, plus `TryFrom<Cicp> for image::metadata::Cicp`):
   ```rust
   use zenpixels::{CicpProjectionError, ColorSpace};

   match color.project_cicp_tag() {
       Ok(tag) => tag,
       Err(CicpProjectionError::HlgDisplayRenderingLost) => { /* ... */ }
       Err(other) => { /* ... */ }
   }
   ```
3. **Recovery returns** — when you want your input buffer back on error:
   - `from_bytes_recover`, `from_typed_recover`, `from_parts_recover`, `from_external`, `from_external_mut` return `Result<PixelBuf, (At<BufferError>, OriginalInput)>`.
   - Zero-copy extractors (`into_vec_zero_copy`, `into_external`, `into_external_mut`, `into_imgvec_zero_copy`) return `Result<T, Self>` (returning the untouched `PixelBuf` on mismatch).

**Naming rule for `try_`:** Standalone fallible methods use ordinary names (`from_typed`, `from_bytes`, `into_vec_zero_copy`, `make_mut`, `merge`, `project_cicp`). The `try_` prefix appears **only** when the same type already has a direct value-returning twin (`try_clone`/`clone`, `try_as_view_mut`/`as_view_mut`, `try_with_*`/`with_*`, `Cicp::try_from_bytes`/`Cicp::from_bytes`).

---

## 3. Buffers, Views, Cropping & External Storage

### Choosing a `PixelBuf` Constructor

| Goal | Constructor | Notes |
|---|---|---|
| Fresh black buffer (max alpha if `AlphaMode::Opaque`) | `PixelBuf::new(w, h, spec)` | Safe default; sets `facts.all_opaque = true` when opaque |
| Fresh reference-white buffer | `PixelBuf::new_white(w, h, spec)` | `255` for U8, `65535` for U16, `1.0` for F32 |
| Fast zeroed buffer for decoders that overwrite every pixel | `PixelBuf::new_zeroed(w, h, spec)` | Skips alpha-fill pass; **refuses** `AlphaMode::Opaque` on alpha-bearing layouts (`BufferError::UnverifiedOpaqueAlpha`) |
| Adopt tightly packed `Vec<u8>` | `PixelBuf::from_bytes(vec, w, h, spec)` | Handles odd-address allocators; use `from_bytes_recover` to get `vec` back on error |
| Adopt typed `Vec<P>` (`u8`, `u16`, `f32`, `[u8; 3]`, `Rgb<u8>`, …) | `PixelBuf::from_typed(vec, w, h, spec)` | Validates `P` against `spec.layout` (annotate literals like `vec![255u8, ...]`); use `from_typed_recover` to recover on error |
| Adopt scalar channel `Vec<S>` (`u8`/`u16`/`f32`/`F16Sample`) | `PixelBuf::from_samples(vec, w, h, spec)` | Checks sample type only |
| Adopt external owned storage zero-copy (`Arc<[u8]>`, `Box<[T]>`, FFI) | `PixelBuf::from_external(storage, w, h, stride, spec)` | Mutable zero-copy: `from_external_mut`; returns `(err, storage)` on error |
| Rebuild from `PixelBufParts` / raw `PixelStorage` | `PixelBuf::from_parts(parts)` / `from_storage_parts(...)` | Use `from_parts_recover` to keep parts on error |

### Fast Opaque Decode (Zero-Overhead Fill-Then-Retag)

```rust
use zenpixels::{AlphaMode, BufferError, ImageSpec, PixelBuf};

// 1. `new_zeroed` refuses unverified `AlphaMode::Opaque` on alpha-bearing layouts:
let opaque_rgba = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque);
let err = PixelBuf::new_zeroed(4, 4, opaque_rgba).unwrap_err();
assert_eq!(*err.error(), BufferError::UnverifiedOpaqueAlpha);

// 2. Allocate as Straight, write every pixel (with A = 255), then verify & retag zero-copy:
let mut buf = PixelBuf::new_zeroed(4, 4, ImageSpec::RGBA8_SRGB)?;
for y in 0..buf.height() {
    for px in buf.as_view_mut().row_mut(y).chunks_exact_mut(4) {
        px.copy_from_slice(&[10, 20, 30, 255]);
    }
}
// `into_opaque_verified` scans all active pixels itself (no prior `set_facts` needed):
let opaque = buf.into_opaque_verified()?;
assert!(opaque.facts().all_opaque);
assert_eq!(opaque.spec().color.alpha(), AlphaMode::Opaque);
// (Or `into_opaque_filled()` to fill alpha to max, or `assume_opaque()` if caller guarantees it.)
```

### Zero-Copy External `Arc<[u8]>` + Crop + Fallible Copy-on-Write

```rust
use std::sync::Arc;
use zenpixels::{ImageSpec, PixelBuf};

let shared: Arc<[u8]> = Arc::from(vec![42u8; 8 * 6 * 3]);
let mut buf = PixelBuf::from_external(shared.clone(), 8, 6, 8 * 3, ImageSpec::RGB8_SRGB)
    .map_err(|(err, _orig)| err)?;

// Rectangular zero-copy crop preserves parent stride (24 bytes):
let tile = buf.crop_view(2, 1, 4, 3)?;
assert_eq!((tile.width(), tile.height()), (4, 3)); // `rows()` and `height()` are aliases on views
assert_eq!(tile.stride(), 24);
assert_eq!(tile.row(0).len(), 4 * 3);

// Fallible CoW mutation promotes `buf` to owned storage without touching `shared`:
buf.try_as_view_mut()?.row_mut(0)[0] = 99;
assert_eq!(buf.as_view().row(0)[0], 99);
assert_eq!(shared[0], 42);
```

### Zero-Allocation Borrowed Views (`PixelView` / `PixelViewMut`)

Primary view constructors take `&'a ImageSpec` so borrowing raw bytes **never allocates**:

```rust
use zenpixels::{ImageSpec, PixelFormat, PixelView, PixelViewMut, SpecCow};

// Tightly packed bytes (zero allocations):
let view = PixelView::from_bytes(rgb_bytes, w, h, ImageSpec::rgb8_srgb_ref())?;

// Strided bytes (zero allocations):
let strided = PixelView::new(rgb_bytes, w, h, stride_bytes, ImageSpec::rgb8_srgb_ref())?;
let mut_view = PixelViewMut::new(rgb_bytes_mut, w, h, stride_bytes, ImageSpec::rgb8_srgb_ref())?;

// Explicit owned/shared spec on a view (0-alloc for standard presets, Arc for custom/ICC):
let cow_view = PixelView::new_with_spec_cow(
    rgb_bytes, w, h, stride_bytes, SpecCow::from_format(PixelFormat::RGB8_SRGB),
)?;
```

### Error Recovery & Zero-Copy `Vec` Extraction

```rust
use zenpixels::{ImageSpec, PixelBuf};

// Recover original Vec intact on failure:
let too_small = vec![1u8, 2, 3];
let (_err, recovered) = PixelBuf::from_typed_recover(too_small, 2, 2, ImageSpec::RGB8_SRGB)
    .unwrap_err();
assert_eq!(recovered, vec![1, 2, 3]);

// Zero-copy extraction back to Vec<u8> (fails returning `Err(buf)` if alignment/stride differs;
// use `into_vec::<P>()` to copy if zero-copy isn't possible):
let data = vec![10u8; 2 * 2 * 3];
let ptr = data.as_ptr();
let buf = PixelBuf::from_typed(data, 2, 2, ImageSpec::RGB8_SRGB)?;
let back: Vec<u8> = buf.into_vec_zero_copy::<u8>().expect("tight u8 Vec");
assert_eq!(back.as_ptr(), ptr);
```

---

## 4. Converting Pixels (`zenpixels-convert`)

`zenpixels-convert` re-exports all of `zenpixels::*` at its root with zero module collisions.

### Which Conversion API Should I Pick?

| Task | Function / Type | Policy & Peak |
|---|---|---|
| Quick full-range sRGB 8-bit output | `to_rgb8(&view)`, `to_gray8(&view)` | `permissive().with_alpha(DiscardIfOpaque)` (refuses unproven alpha loss!) |
| Quick straight-alpha sRGB 8-bit output | `to_rgba8(&view)`, `to_bgra8(&view)` | `permissive()` |
| One-shot conversion into a new `PixelBuf` | `convert_to_buf(&view, dst_spec, &policy, peak)` | Explicit `&ConvertPolicy` + `PeakSource` |
| One-shot conversion into an existing sink | `convert_into(&src, &mut dst, &policy, peak)` | Works with `PixelBuf`, `PixelViewMut`, `BandedBuf` |
| Reusable whole-image converter | `ImagePlan::build(src, dst, &policy, peak)` | `.execute(&src)` or `.execute_into(&src, &mut dst)` |
| Streaming row-by-row converter (zero heap scratch) | `RowPlan::build(src, dst, &policy, resolved_source_peak)` | Takes `Option<Nits>` (cannot measure a whole image from one row); `plan.convert_row(src_row, dst_row, width, y)` |
| Negotiate + convert + resolve CICP/ICC for an encoder | `adapt_for_encode_format` / `adapt_for_encode` / `OutputPlan` | Returns `EncodeReady { pixels: PixelCow, cicp, icc, hdr }` |

### `ConvertPolicy` Presets (Safe by Default)

`ConvertPolicy::default()` equals **`ConvertPolicy::lossless()`** — it refuses unprovable depth narrowing, alpha discard, and range scaling, and preserves extended float gamut. Note that `lossless()` describes **loss-permission settings** and is **content-aware** on whole-image plans (`ImagePlan` / `convert_to_buf`): freshly allocated all-black/white buffers (`PixelBuf::new`, `new_white`) seed exact 1-bit `SampleFacts`, so narrowing an untouched black `RGB16` buffer to `RGB8` is provably lossless and succeeds, whereas non-exact 16-bit content fails with `ConvertError::DepthReductionForbidden` (and `RowPlan::build`, which cannot see image content, refuses unconditionally unless facts are attached on the policy). Opt in explicitly when loss is desired:

| Preset | Depth | Alpha | Gamut | Range | HDR→SDR Tone Map | ICC Sub |
|---|---|---|---|---|---|---|
| `ConvertPolicy::lossless()` / `default()` | `Forbid` | `Forbid` | `PreserveExtended` | `Forbid` | `false` | `false` |
| `ConvertPolicy::permissive()` | `Round` | `DiscardUnchecked` | `Clip` | `Convert` | `false` | `true` |
| `ConvertPolicy::working_linear()` | `Round` | `DiscardIfOpaque` | `PreserveExtended` | `Convert` | `false` (`BT2408` white) | `true` |
| `ConvertPolicy::display_sdr(white, peak)` | `Round` | `CompositeOnto(BLACK)` | `Clip` | `Convert` | `true` | `true` |

**RGB → Gray luma domain:** Defaults to `LumaDomain::Encoded` (`Y'` on encoded samples, matching JPEG/video — sRGB `(255, 0, 0)` → `54`). Switch to physical luminance with `.with_luma_domain(LumaDomain::LinearLight)` (sRGB `(255, 0, 0)` → `127`).

### HDR PQ → SDR Tone Mapping & SIMD/Scalar 0-Bit Parity

Plain `ConvertPolicy::permissive()` refuses unanchored PQ→SDR routes (`ConvertError::MissingHdrAnchor`) until you configure a diffuse white anchor and tone mapper (`Bt2446A`, enabled via the `bt2446a` Cargo feature):

```rust
use std::sync::Arc;
use zenpixels_convert::{
    Accelerators, Bt2446A, ConvertPolicy, DiffuseWhite, ImagePlan, ImageSpec, Nits, PeakSource,
    RowPlan,
};

let policy = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0))
    .with_tone_mapper(Arc::new(Bt2446A::pipeline_managed()));

// Whole-image streaming peak measurement (O(1) row memory):
let plan = ImagePlan::build(
    ImageSpec::rgb16_bt2100_pq_ref(),
    ImageSpec::rgb8_srgb_ref(),
    &policy,
    PeakSource::Measured,
)?;
let sdr_buf = plan.execute(&pq16_buf)?;

// Streaming row-by-row with explicit resolved source peak (e.g. 1000 nits) + SIMD vs scalar parity:
let peak = Some(Nits::try_new(1000.0).unwrap());
let simd_row = RowPlan::build(
    ImageSpec::rgb16_bt2100_pq_ref(),
    ImageSpec::rgb8_srgb_ref(),
    &policy.clone().with_accelerators(Accelerators::ALL),
    peak,
)?;
let scalar_row = RowPlan::build(
    ImageSpec::rgb16_bt2100_pq_ref(),
    ImageSpec::rgb8_srgb_ref(),
    &policy.with_accelerators(Accelerators::SCALAR),
    peak,
)?;
// `simd_row.convert_row(...)` and `scalar_row.convert_row(...)` produce 0-bit identical bytes!
```

### Explicit ICC Conversion via `MoxCms` and `Lcms`

Enable Cargo features `cms-moxcms` and/or `cms-lcms` on `zenpixels-convert`:

```rust
use std::sync::Arc;
use zenpixels_convert::{
    AlphaMode, ColorSpace, ConvertPolicy, ImageSpec, Lcms, MoxCms, PeakSource, RenderingIntent,
    SampleLayout, convert_to_buf, icc_profiles,
};

let adobe_spec = ImageSpec::new(
    SampleLayout::RGB8,
    ColorSpace::Icc {
        profile: Arc::from(icc_profiles::ADOBE_RGB),
        channels: 3,
        alpha: AlphaMode::None,
        intent: RenderingIntent::RelativeColorimetric,
    },
);

// Disable matrix+TRC fast substitution so the CMS backend actually executes:
let mox_pol = ConvertPolicy::permissive()
    .with_icc_substitution(false)
    .with_cms(Arc::new(MoxCms));
let lcms_pol = ConvertPolicy::permissive()
    .with_icc_substitution(false)
    .with_cms(Arc::new(Lcms));

let out_mox = convert_to_buf(&src.as_view(), ImageSpec::rgb8_srgb_ref(), &mox_pol, PeakSource::Measured)?;
let out_lcms = convert_to_buf(&src.as_view(), ImageSpec::rgb8_srgb_ref(), &lcms_pol, PeakSource::Measured)?;
```

### Encoder Format Negotiation & Output Signaling (`OutputPlan` / `adapt_for_encode_format`)

```rust
use zenpixels_convert::{
    ConvertError, ConvertPolicy, ImageSpec, OutputPlan, OutputProfile, PeakSource, PixelCow,
    PixelFormat, adapt_for_encode_format,
};

const SUPPORTED: &[PixelFormat] = &[PixelFormat::RGB8_SRGB, PixelFormat::RGBA8_SRGB];

// Borrows zero-copy (`PixelCow::Borrowed`) when input already matches a supported format:
let (chosen, ready) = adapt_for_encode_format(
    buf.as_view(),
    buf.origin(),
    SUPPORTED,
    &ConvertPolicy::permissive(),
    PeakSource::Measured,
    OutputProfile::Auto,
)?;
assert!(matches!(ready.pixels, PixelCow::Borrowed(_)));

// Reusable OutputPlan (borrows specs & policy; checks DESTINATION signaling eagerly):
let plan = OutputPlan::build(
    ImageSpec::rgb8_srgb_ref(),
    ImageSpec::rgb8_srgb_ref(),
    &ConvertPolicy::permissive(),
    PeakSource::Measured,
    OutputProfile::Auto,
)?;
assert!(plan.is_identity());
let ready_owned = plan.execute_buf(buf)?; // Zero-copy move on identity!

// Destination signaling guard: CICP cannot carry `HlgMode::Display(_)`, so targeting
// `RGB16_BT2100_HLG_1000` as `dst` under `CicpOnly` or `Auto` fails eagerly:
let err = OutputPlan::build(
    ImageSpec::rgb8_srgb_ref(),
    ImageSpec::rgb16_bt2100_hlg_1000_ref(),
    &ConvertPolicy::permissive(),
    PeakSource::Measured,
    OutputProfile::CicpOnly,
)
.unwrap_err();
assert_eq!(*err.error(), ConvertError::OutputSignalingUnavailable);
```

---

## 5. Planar YCbCr & Planar RGB

All plane constructors (`PlaneView`, `PlaneViewMut`, `PlaneBuf`) share the uniform argument order **`(data_or_samples, width, height, stride, plane_spec)`**:
- `from_samples` takes `stride` in **sample elements** (`S`); `new` / `from_external` take `stride` in **bytes**; `.stride()` always returns **bytes**.
- `code_u32(x, y, component)` and `as_code_plane(component)` take `component = 0` for 1-channel planes (`Y`, `Cb`, `Cr`, `R`, `G`, `B`, `Alpha`) and `0` or `1` for semi-planar `CbCr`/`CrCb`.

### 10-bit 4:2:0 YCbCr → Pre-Allocated `BGRA8` (Zero Allocation)

```rust
use zenpixels_convert::{
    BitPacking, ChromaSiting, ColorSpace, ImageSpec, PixelBuf, PlanarView, PlaneRole, PlaneSpec,
    PlaneView, SampleType, SignalRange, YcbcrMatrix, YcbcrOptions, ycbcr_to_rgb_with_options_into,
};

let y_spec = PlaneSpec::new(PlaneRole::Y, SampleType::U16).with_packing(BitPacking::U10_LSB);
let cb_spec = PlaneSpec::new(PlaneRole::Cb, SampleType::U16)
    .with_packing(BitPacking::U10_LSB)
    .with_subsampling(2, 2);
let cr_spec = PlaneSpec::new(PlaneRole::Cr, SampleType::U16)
    .with_packing(BitPacking::U10_LSB)
    .with_subsampling(2, 2);

let py = PlaneView::from_samples(&y_u16, 4, 4, 4, y_spec)?;   // element stride = 4
let pcb = PlaneView::from_samples(&cb_u16, 2, 2, 2, cb_spec)?; // element stride = 2
let pcr = PlaneView::from_samples(&cr_u16, 2, 2, 2, cr_spec)?;

let planar = PlanarView::new(4, 4, &[py, pcb, pcr])?
    .with_color(ColorSpace::SRGB)
    .with_ycbcr_matrix_code(1) // BT.709
    .with_range(SignalRange::Narrow)
    .with_chroma_siting(ChromaSiting::Left);

// Inspect 10-bit luma code directly (component 0):
let y_plane = planar.plane_by_role(PlaneRole::Y).unwrap();
let code = y_plane.code_u32(0, 0, 0).unwrap();
let code_plane = y_plane.as_code_plane(0)?;
assert_eq!(code_plane.code(0, 0), Some(code as u16));

// Zero-alloc reconstruction into pre-allocated BGRA8 buffer:
let mut dst = PixelBuf::new(4, 4, ImageSpec::BGRA8_SRGB)?;
ycbcr_to_rgb_with_options_into(
    &planar,
    YcbcrMatrix::Bt709,
    SignalRange::Narrow,
    &YcbcrOptions::bilinear(),
    &mut dst.as_view_mut(),
)?;
```

### Planar F32 RGB+Alpha → Interleaved `RGBA8`

`ycbcr_to_rgb_with_options` also reconstructs **planar RGB / RGBA** (`PlaneRole::{R, G, B, Alpha}`) directly into any supported interleaved `SampleLayout`, propagating `Orientation`:

```rust
use zenpixels_convert::{
    ColorSpace, Orientation, PlanarBuf, PlaneRole, PlaneSpec, SampleLayout, SampleType,
    YcbcrOptions, bake_orientation_owned, ycbcr_to_rgb_with_options,
};

let specs = [
    PlaneSpec::new(PlaneRole::R, SampleType::F32),
    PlaneSpec::new(PlaneRole::G, SampleType::F32),
    PlaneSpec::new(PlaneRole::B, SampleType::F32),
    PlaneSpec::new(PlaneRole::Alpha, SampleType::F32),
];
let mut planar = PlanarBuf::new(4, 2, &specs)?
    .with_color(ColorSpace::LINEAR_SRGB_ALPHA)
    .with_orientation(Orientation::Rotate90);

// Write pixels via typed row slices:
planar.plane_view_mut_by_role(PlaneRole::R).unwrap().row_as_mut::<f32>(0)?[0] = 1.0;

// Reconstruct to interleaved RGBA8 (preserves Orientation::Rotate90):
let rgba = ycbcr_to_rgb_with_options(&planar.as_view(), SampleLayout::RGBA8, &YcbcrOptions::nearest())?;
assert_eq!(rgba.display_dimensions(), (2, 4));

// Physically bake pending orientation -> dimensions become (2, 4), orientation becomes Identity:
let baked = bake_orientation_owned(rgba)?;
assert_eq!((baked.width(), baked.height()), (2, 4));
assert_eq!(baked.orientation(), Orientation::Identity);
```

---

## 6. Orientation & Gigapixel `BandedBuf`

### Orientation Verb Reference

| Intent | API | Pixel movement? | Resulting `orientation()` |
|---|---|---|---|
| Replace EXIF metadata | `with_orientation(self, o)` / `set_orientation(&mut self, o)` | None (O(1)) | `o` |
| Compose EXIF metadata (`old.then(delta)`) | `compose_orientation(self, delta)` | None (O(1)) | `old.then(delta)` |
| Physically apply **explicit** `o` into a new/owned buffer (ignores pending metadata) | `apply_orientation(&view, o)` / `apply_orientation_owned(buf, o)` / `apply_orientation_banded(&bands, o, max_bytes)` | Yes (`apply_orientation_owned(_, Identity)` is zero-copy) | `Identity` |
| Physically apply **explicit** `o` into a caller sink | `apply_orientation_into(&view, o, &mut dst)` / `apply_orientation_banded_into(&bands, o, &mut dst_bands)` | Yes (zero allocation) | Unchanged on caller's `dst` (pixel write only) |
| Physically **bake** pending metadata | `bake_orientation(&view)` / `bake_orientation_owned(buf)` / `bake_orientation_banded(&bands, max_bytes)` | Yes (`bake_orientation_owned` is zero-copy when `Identity`) | `Identity` |
| Dual-layer gain-map physical bake | `bake_dual_layer_orientation(&view)` / `bake_dual_layer_orientation_after(&view, delta)` | Both layers | `Identity` |

### Gigapixel `BandedBuf`

`BandedBuf` splits huge images into row bands bounded by `max_band_bytes` (default `DEFAULT_MAX_BAND_BYTES`) and implements both `PixelSource` and `PixelSink`:

```rust
use zenpixels_convert::{
    BandedBuf, ConvertPolicy, ImagePlan, ImageSpec, Orientation, PeakSource,
    apply_orientation_banded,
};

let mut src = BandedBuf::new(8, 6, ImageSpec::RGB8_SRGB, 48)?;
assert!(src.band_count() > 1);
src.row_mut(0)[0..3].copy_from_slice(&[200, 100, 50]);

let mut dst = BandedBuf::new(8, 6, ImageSpec::RGBA8_SRGB, 64)?;
ImagePlan::build(src.spec(), dst.spec(), &ConvertPolicy::permissive(), PeakSource::Measured)?
    .execute_into(&src, &mut dst)?;

let rotated = apply_orientation_banded(&src, Orientation::Rotate90, 48)?;
assert_eq!((rotated.width(), rotated.height()), (6, 8));
```

---

## 7. Lossless H.273 CICP & 44-Byte ICC.1 `cicpType`

```rust
use zenpixels::{AlphaMode, Cicp, CicpTag, CustomPrimaries};

// Standard 4-byte H.273 CICP round-trip:
let bt601 = Cicp::new(6, 6, 0, true);
let cs = bt601.to_color_space(AlphaMode::None)?;
assert_eq!(cs.project_cicp()?, bt601);

// 44-byte ICC.1 cicpType with custom primaries (negative y_B supported via signed s15Fixed16):
let tag = CicpTag::with_custom_primaries(CustomPrimaries::ACES_AP0, 16, 0, true);
let bytes = tag.to_tag_bytes_vec();
assert_eq!(bytes.len(), 44);
let decoded = CicpTag::from_tag_bytes(&bytes)?;
// Wire encoding quantizes chromaticities to signed s15Fixed16 (1/65536), so raw `f32`
// constants differ slightly from decoded floats. Compare serialized bytes or `to_s15fixed16()`:
assert_eq!(decoded.to_tag_bytes_vec(), bytes);
assert_eq!(decoded.cicp(), tag.cicp());
assert_eq!(
    decoded.custom_primaries().unwrap().to_s15fixed16(),
    CustomPrimaries::ACES_AP0.to_s15fixed16(),
);
let _cs_custom = decoded.to_color_space(AlphaMode::None)?;

// ICC Annex 1 D50 Bradford matrix (rows sum to ICC PCS D50 = [0.9642, 1.0, 0.8249]):
let m = CustomPrimaries::ACES_AP0.to_icc_d50_matrix_f64().unwrap();
```

---

## 8. UltraHDR / ISO 21496-1 Gain Maps (`DualLayerHdrBuf`)

- Cancellable gain-map APIs accept `&dyn Stop`. Pass **`&zenpixels_convert::Unstoppable`** (re-exported from `enough`) when no cancellation is needed.
- `apply_gain_map(&dual, headroom_stops)` takes log₂ **stops**; for linear display boost (e.g. `2.5×`) or custom orientation/resampling, use `GainMapApplyConfig::from_display_boost(2.5)` / `from_headroom_stops(stops)` with `apply_gain_map_with` or `apply_gain_map_oriented`.
- When reconstructing via `DualLayerHdrBuf::from_parts(DualLayerHdrBufParts)`, both `parts.base.orientation()` and `parts.gain_map.orientation()` must equal `parts.orientation` (or both be `Identity`).

```rust
use zenpixels_convert::{
    DualLayerHdrBuf, GainMapApplyConfig, GainMapExtractConfig, GainMapOrientation, Orientation,
    Unstoppable, apply_gain_map_oriented, extract_gain_map_only,
};

let cfg = GainMapExtractConfig::default();
let (gain_map, meta) = extract_gain_map_only(&sdr_base.as_view(), &hdr_alt.as_view(), &cfg, &Unstoppable)?;
let dual = DualLayerHdrBuf::new(sdr_base, gain_map, meta)?
    .compose_orientation(Orientation::Rotate90); // O(1) EXIF orientation update

let apply_bake = GainMapApplyConfig::from_display_boost(2.5)
    .with_orientation(GainMapOrientation::Bake);
let (hdr_baked, baked_orient) = apply_gain_map_oriented(&dual.as_view(), &apply_bake, &Unstoppable)?;
assert_eq!(baked_orient, Orientation::Identity);

let apply_pass = GainMapApplyConfig::from_display_boost(2.5)
    .with_orientation(GainMapOrientation::Passthrough);
let (hdr_stored, pending_orient) = apply_gain_map_oriented(&dual.as_view(), &apply_pass, &Unstoppable)?;
assert_eq!(pending_orient, Orientation::Rotate90);
```

---

## 9. Content Inspection & In-Place Lossless Compaction (`load_bearing`)

Inspect actual pixel data to find whether alpha, chroma, or U16 low bytes carry real information, and compact without loss:

```rust
use zenpixels_convert::{
    AlphaCompaction, AlphaMode, PixelBufLoadBearingExt, PixelViewLoadBearingExt, SampleLayout,
};

// Allocating compaction (e.g. neutral opaque RGBA16 -> GRAY8):
let report = rgba16_neutral.as_view().determine_load_bearing();
assert_eq!(report.uses_alpha, Some(false));
assert_eq!(report.uses_chroma, Some(false));
assert_eq!(report.uses_low_bits, Some(false));
let compact = rgba16_neutral.as_view().reduce_to_load_bearing_format()?.unwrap();
assert_eq!(compact.spec().layout, SampleLayout::GRAY8);

// Zero-reallocation in-place compaction on opaque RGBA8:
// 1. Retag alpha to AlphaMode::Opaque without shifting lanes:
rgba8_opaque.reduce_to_load_bearing_format_in_place(AlphaCompaction::RetagOpaque);
assert_eq!(rgba8_opaque.spec().color.alpha(), AlphaMode::Opaque);
assert_eq!(rgba8_opaque.spec().layout, SampleLayout::RGBA8);

// 2. Physically drop unused alpha lane in-place (reusing the same allocation):
rgba8_opaque.reduce_to_load_bearing_format_in_place(AlphaCompaction::DropUnusedLanes);
assert_eq!(rgba8_opaque.spec().layout, SampleLayout::RGB8);
```

---

## 10. Interop (`rgb`, `imgref`, `image 0.25`)

Enable `features = ["imgref", "image"]` on `zenpixels`:

```rust
use imgref::ImgVec;
use rgb::Rgb;
use zenpixels::{ColorSpace, ImageSpec, PixelBuf};

// `imgref` round-trip (note `ImgVec::new(pixels, width, height)` and `img[(x_usize, y_usize)]`):
let img_ref = buf.as_view().as_imgref::<Rgb<u8>>()?;
let top_left = img_ref[(0usize, 0usize)];
let owned_img = ImgVec::new(vec![top_left; 4], 2, 2);
let back_from_imgref = PixelBuf::from_imgvec(owned_img, ColorSpace::SRGB)?;

// `image` 0.25 round-trip (propagates CICP color space automatically):
let dyn_img = buf.into_dynamic_image()?;
let back_from_image = PixelBuf::from_dynamic_image(dyn_img)?;
```
