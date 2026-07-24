<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenpixels

Pixel format types and transfer-function-aware conversion for Rust image codecs.

A JPEG decoder gives you `RGB8` in sRGB. An AVIF decoder gives you `RGBA16` in BT.2020 PQ. A resize library wants `RGBF32` in linear light. Without shared types, every codec pair needs hand-rolled conversion — and gets transfer functions wrong, silently drops alpha, or writes "sRGB" in the ICC profile while the pixels are linear.

zenpixels makes pixel format descriptions first-class types that travel with the data. The conversion crate handles transfer functions, gamut matrices, depth scaling, and alpha compositing so codecs don't have to.

Two crates: **zenpixels** (types, buffers, metadata) and **zenpixels-convert** (all the math). Both are `no_std + alloc`, `forbid(unsafe_code)`, no system dependencies.

```toml
# Types only — for codec crates
zenpixels = "0.2.16"

# Types + conversion — for processing pipelines
zenpixels-convert = "0.2.16"
```

## Quick start

`zenpixels` gives you a [`PixelBuffer`] that carries its [`PixelDescriptor`]
(format + color semantics) so the bytes never travel without a description of
what they mean. Allocate one, hand the rows to a codec, recover the `Vec` when
done. (`Rgba<u8>` comes from the [`rgb`](https://crates.io/crates/rgb) crate
and needs the `rgb` feature; the descriptor-based path below works without it.)

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};
use rgb::Rgba;

// Typed buffer — format enforced at compile time, SIMD-aligned rows.
// A typed buffer knows its byte layout but leaves color semantics
// Unknown: the `rgb` crate's `Rgba<u8>` says nothing about sRGB vs
// linear, so stamp the encoding when you know it. `with_descriptor`
// keeps the layout and only changes the transfer/primaries.
let pixels = vec![Rgba::new(255u8, 128, 0, 255); 64 * 48];
let buf = PixelBuffer::<Rgba<u8>>::from_pixels(pixels, 64, 48)?
    .with_descriptor(PixelDescriptor::RGBA8_SRGB);

// Row and byte access live on the view; `as_slice()` is the entry point.
// `row(y)` is unpadded; `as_strided_bytes()` is the whole backing buffer
// (stride included) — zero-copy passthrough to a codec or GPU upload.
let view = buf.as_slice();
let first_row: &[u8] = view.row(0);
let strided: &[u8]   = view.as_strided_bytes();

// Recover the allocation for pool reuse.
let raw: Vec<u8> = buf.into_vec();
# Ok::<(), zenpixels::At<zenpixels::BufferError>>(())
```

Don't know the format at compile time? Build a [`PixelDescriptor`] from a
[`PixelFormat`] plus the color semantics, then use a type-erased
`PixelBuffer`:

```rust
use zenpixels::{PixelBuffer, PixelDescriptor, PixelFormat, TransferFunction, ColorPrimaries};

let desc = PixelDescriptor::RGB8_SRGB
    .with_transfer(TransferFunction::Linear)   // same bytes, linear light
    .with_primaries(ColorPrimaries::DisplayP3);

let buf = PixelBuffer::new(64, 48, desc);      // panics on OOM; try_new for fallible
assert_eq!(buf.descriptor().format, PixelFormat::Rgb8);
```

### Wrapping a decoder's `Vec<u8>` (no copy)

`PixelBuffer::new` *allocates*. A codec that already decoded into its own
`Vec<u8>` wants the opposite: take ownership of that allocation and stamp a
descriptor on it, without a copy. That is [`PixelBuffer::from_vec`] — the
constructor every zen codec reaches for at the end of decode.

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};

let (width, height) = (64u32, 48u32);
// A decoder produced these — tightly packed RGBA8, 4 bytes/pixel.
let decoded: Vec<u8> = vec![0u8; width as usize * height as usize * 4];

// Consumes `decoded` (no copy); the descriptor's tight stride
// (width * bytes_per_pixel) is assumed — `from_vec` does NOT pad rows.
let buf = PixelBuffer::from_vec(decoded, width, height, PixelDescriptor::RGBA8_SRGB)?;
assert_eq!(buf.stride(), 64 * 4);              // bytes, not pixels — see below
# Ok::<(), zenpixels::At<zenpixels::BufferError>>(())
```

`from_vec` returns [`BufferError::InsufficientData`] (wrapped as
`At<BufferError>`) if the `Vec` is shorter than `aligned_stride(width) * height`
plus the leading bytes skipped for channel alignment. It does **not** accept an
explicit stride: rows are assumed tightly packed at
`width * bytes_per_pixel`. For padded/strided bytes you don't own — a decoder's
scratch row buffer, a crop of a parent image, a GPU-readback strip — borrow a
view with an explicit stride instead (next section).

> **Stride is always measured in BYTES**, never pixels or elements
> (`stride()` returns `usize`; `aligned_stride(width) = width * bytes_per_pixel`).
> So for 64-pixel-wide RGBA8 the tight stride is `64 * 4 = 256`, and for
> 16-bit RGB it is `64 * 6 = 384`. Passing a pixel count where a byte stride is
> expected is the single most common silent-corruption bug at the codec
> boundary.

### Borrowing strided bytes (explicit stride)

When the bytes have row padding (SIMD alignment, a sub-region of a larger
image) or you only want to borrow rather than own them, wrap a
[`PixelSlice`] with an explicit byte stride:

```rust
use zenpixels::{PixelSlice, PixelDescriptor};

let (width, rows) = (64u32, 48u32);
let stride_bytes = 256usize;                   // e.g. SIMD-padded; >= width*bpp
let backing: Vec<u8> = vec![0u8; stride_bytes * rows as usize];

// Borrows `backing` (lifetime-bound, zero-copy). `stride_bytes` is BYTES
// between row starts, and must be a multiple of bytes_per_pixel.
let view = PixelSlice::new(&backing, width, rows, stride_bytes, PixelDescriptor::RGBA8_SRGB)?;
assert_eq!(view.width(), 64);
# Ok::<(), zenpixels::At<zenpixels::BufferError>>(())
```

Multi-byte samples (U16/F16/F32) are read in **native byte order** — a decoder
holding big-endian container data (e.g. 16-bit PNG) must byte-swap before
wrapping. `PixelSlice::new` validates length, stride, and channel alignment,
returning `At<BufferError>` (`InsufficientData` / `StrideTooSmall` /
`StrideNotPixelAligned` / `AlignmentViolation`) on a mismatch.

## Format conversion (zenpixels-convert)

```rust
use zenpixels::PixelBuffer;
use zenpixels_convert::{RowConverter, best_match, ConvertIntent};

// `src: PixelBuffer` — pick the cheapest target format the encoder supports
let source_desc = src.descriptor();
let target = best_match(source_desc, &encoder_formats, ConvertIntent::Fastest)
    .ok_or("no compatible format")?;

// Allocate the destination, then convert row by row — no per-row allocation
let (w, h) = (src.width(), src.height());
let mut dst = PixelBuffer::new(w, h, target);
let src_view = src.as_slice();
let mut dst_view = dst.as_slice_mut();
let mut converter = RowConverter::new(source_desc, target)?;
for y in 0..h {
    converter.convert_row(src_view.row(y), dst_view.row_mut(y), w);
}
```

> Most callers want the high-level `to_rgba8()` / `to_rgb8()` extension (shown in
> the codec READMEs); reach for `RowConverter` only when you need a specific
> target format or zero per-row allocation.

### Color profile conversion

Built-in: named-profile pairs (sRGB ↔ Display P3 ↔ BT.2020 ↔ Adobe RGB) use
hardcoded matrices with fused SIMD kernels — LUT-decode + SIMD matrix +
SIMD polynomial encode for u16, fused matlut for u8. No CMS backend needed.

```rust
use zenpixels::{PixelDescriptor, ColorPrimaries};
use zenpixels_convert::{RowConverter, ConvertOptions};

let p3  = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
let srgb = PixelDescriptor::RGB8_SRGB;

let mut conv = RowConverter::new_explicit(
    p3, srgb, &ConvertOptions::permissive(),
)?;
conv.convert_row(p3_row, srgb_row, width);
```

Custom ICC profiles (vendor-specific, LUT-based, perceptual intent) need a
CMS backend. Pass one via `PluggableCms` — the plan delegates to the plugin
when the profiles differ, or uses the built-in matlut fast path when both
sides are well-known.

```rust
use zenpixels_convert::{RowConverter, ConvertOptions, cms::PluggableCms};

let cms: &dyn PluggableCms = &MoxCms;  // or any backend
let mut conv = RowConverter::new_explicit_with_cms(
    source_desc, target_desc,
    &ConvertOptions::permissive(),
    Some(cms),
)?;
```

For HDR and wide-gamut pipelines that defer tone/gamut mapping, use
`with_clip_out_of_gamut(false)` — f32 sRGB transfers then preserve
negative and supernormal values through the conversion.

## Type system

The core split: *what the bytes are* vs. *what the bytes mean*.

**`PixelFormat`** is a flat enum of byte layouts — channel count, depth, memory order. No color semantics.

```
Rgb8, Rgba8, Rgb16, Rgba16, RgbF16, RgbaF16, RgbF32, RgbaF32,
Gray8, Gray16, GrayF16, GrayAF16, GrayF32, GrayA8, GrayA16, GrayAF32,
Bgra8, Rgbx8, Bgrx8, Cmyk8, OklabF32, OklabaF32
```

F16 variants are descriptor-only today — typed `Pixel` impls land when Rust
stable ships native `f16`.

**`PixelDescriptor`** wraps a `PixelFormat` with everything needed to interpret the color data:

```rust
pub struct PixelDescriptor {
    pub format: PixelFormat,
    pub transfer: TransferFunction,    // Linear, Srgb, Bt709, Pq, Gamma22, Hlg, Unknown
    pub alpha: Option<AlphaMode>,      // Undefined, Straight, Premultiplied, Opaque
    pub primaries: ColorPrimaries,     // Bt709, DisplayP3, Bt2020, AdobeRgb, Unknown
    pub signal_range: SignalRange,     // Full or Narrow
}
```

Every field's type is re-exported at the crate root — there is no submodule to
reach into. The canonical import for hand-building a descriptor:

```rust
use zenpixels::{
    PixelDescriptor,        // the struct itself
    PixelFormat,            // the byte-layout enum
    TransferFunction,       // Linear, Srgb, Bt709, Pq, Gamma22, Hlg, Unknown
    AlphaMode,              // Undefined, Straight, Premultiplied, Opaque
    ColorPrimaries,         // Bt709, DisplayP3, Bt2020, AdobeRgb, Unknown
    SignalRange,            // Full, Narrow
};
```

Every buffer carries one. Every codec declares which ones it produces and consumes. Predefined constants cover the common cases — each expands to a complete descriptor, so check the **alpha** column before handing bytes to a codec: a codec that trusts the wrong alpha mode silently corrupts pixels (treating straight alpha as premultiplied, or reading the padding byte of an `X` format as coverage).

| Constant | format | transfer | alpha | primaries | range |
|---|---|---|---|---|---|
| `RGB8_SRGB` | `Rgb8` | `Srgb` | `None` (no alpha channel) | `Bt709` | `Full` |
| `RGBA8_SRGB` | `Rgba8` | `Srgb` | `Some(Straight)` | `Bt709` | `Full` |
| `RGB16_SRGB` | `Rgb16` | `Srgb` | `None` | `Bt709` | `Full` |
| `RGBA16_SRGB` | `Rgba16` | `Srgb` | `Some(Straight)` | `Bt709` | `Full` |
| `RGBF32_LINEAR` | `RgbF32` | `Linear` | `None` | `Bt709` | `Full` |
| `RGBAF32_LINEAR` | `RgbaF32` | `Linear` | `Some(Straight)` | `Bt709` | `Full` |
| `GRAY8_SRGB` | `Gray8` | `Srgb` | `None` | `Bt709` | `Full` |
| `GRAYA8_SRGB` | `GrayA8` | `Srgb` | `Some(Straight)` | `Bt709` | `Full` |
| `BGRA8_SRGB` | `Bgra8` | `Srgb` | `Some(Straight)` | `Bt709` | `Full` |
| `RGBX8_SRGB` | `Rgbx8` | `Srgb` | `Some(Undefined)` (padding byte, **not** coverage) | `Bt709` | `Full` |
| `BGRX8_SRGB` | `Bgrx8` | `Srgb` | `Some(Undefined)` (padding byte) | `Bt709` | `Full` |
| `RGB16_BT2100_PQ` | `Rgb16` | `Pq` | `None` | `Bt2020` | `Full` |
| `RGB16_BT2100_HLG` | `Rgb16` | `Hlg` | `None` | `Bt2020` | `Full` |
| `OKLABF32` | `OklabF32` | `Unknown` | `None` | `Bt709` | `Full` |
| `OKLABAF32` | `OklabaF32` | `Unknown` | `Some(Straight)` | `Bt709` | `Full` |
| `CMYK8` | `Cmyk8` | `Unknown` | `None` | `Bt709` | `Full` |

`None` means the format has no alpha channel at all; `Some(Undefined)` (the `X` formats) means the lane is present but its bytes are padding, not coverage. Every constant that does carry alpha defaults to **straight** (unassociated) — never premultiplied. Transfer-agnostic siblings (`RGB8`, `RGBA8`, `RGB16`, … without the `_SRGB`/`_LINEAR` suffix) are identical except `transfer = Unknown`; reach for those when you don't yet know the encoding and want to stamp it later with `with_transfer(...)`.

### CICP and ICC

`Cicp` carries ITU-T H.273 code points (used by AVIF, HEIF, JPEG XL, AV1). Named constants for `SRGB`, `DISPLAY_P3`, `BT2100_PQ`, `BT2100_HLG`. Human-readable name lookups via `color_primaries_name()` etc.

`ColorContext` bundles ICC profile bytes and/or CICP codes. Travels with pixel data via `Arc` — cheap to clone, cheap to share across pipeline stages.

`ColorOrigin` is the immutable provenance record: *how the source file described its color*, not what the pixels currently are. Used at encode time to decide whether to re-embed the original profile.

**CICP → ICC synthesis** (zenpixels-convert): `synthesize_icc_for_cicp(Cicp)` resolves an embeddable ICC profile for any assigned H.273 combination — bundled compressed database (`icc-db` feature, on by default; ~36 KB asset, lazily decoded per transfer group, no CMS required), with PQ/HLG HDR included. `synthesize_gray_icc_for_cicp` is the GRAY-class sibling for single-channel output (libpng rejects gray images paired with RGB-class profiles). The sRGB default answers `NotNeeded`; everything the database serves round-trips back through `extract_cicp` / `identify_common`.

### Orientation

`Orientation` is the canonical EXIF orientation enum for the zen ecosystem. `#[repr(u8)]` with EXIF values 1-8, so `o as u8` gives the tag value directly.

All 8 elements of the D4 dihedral group, with full composition algebra:

```rust
use zenpixels::Orientation;

let combined = Orientation::Rotate90.then(Orientation::FlipH);
assert_eq!(combined, Orientation::Transpose);

let undone = Orientation::Rotate90.compose(Orientation::Rotate90.inverse());
assert_eq!(undone, Orientation::Identity);

let (w, h) = Orientation::Rotate90.output_dimensions(1920, 1080);
assert_eq!((w, h), (1080, 1920));
```

The buffer-baking half lives in zenpixels-convert: `apply_orientation`
(fresh buffer; SIMD-tiled transpose for 4-byte pixels behind the
`fast-transpose` feature, portable scalar tiled path otherwise),
`apply_orientation_into` (caller-provided target, no allocation), and
`apply_orientation_in_place` (reuses the buffer's own allocation; see below).

## Pixel buffers

`PixelBuffer`, `PixelSlice`, and `PixelSliceMut` carry their `PixelDescriptor` and optional `ColorContext`. Generic over `P: Pixel` for compile-time type safety, with zero-cost `.erase()` / `.try_typed::<Q>()` for dynamic dispatch.

```rust
// Typed buffer — format enforced at compile time
let buf = PixelBuffer::<Rgba<u8>>::from_pixels(pixels, width, height)?;

// Type-erased for codec dispatch
let erased = buf.erase();

// Recover the type
let typed = erased.try_typed::<Rgba<u8>>().unwrap();
```

### Data access

Row and byte accessors live on `PixelSlice` / `PixelSliceMut`; reach them from a buffer via `as_slice()` / `as_slice_mut()`. `as_contiguous_bytes()`, `copy_to_contiguous_bytes()`, and the `rows(y, count)` / `crop_view(...)` windowing methods are also available directly on `PixelBuffer`.

Row-level: `row(y)` returns pixel bytes without padding. `row_with_stride(y)` includes padding.

Bulk: `as_strided_bytes()` returns the full backing `&[u8]` including stride padding — zero-copy passthrough to GPU uploads, codec writers, or anything that takes a buffer + stride. `as_contiguous_bytes()` returns `Some` only when rows are tightly packed. `contiguous_bytes()` returns `Cow` — borrows when tight, copies to strip padding otherwise.

Views: `sub_rows(y, count)` and `crop_view(x, y, w, h)` are zero-copy. `crop_copy()` allocates.

### Dimensions and descriptor

Read a buffer's geometry and color semantics directly — these accessors exist
on `PixelBuffer`, `PixelSlice`, and `PixelSliceMut` alike:

```rust
let w: u32          = buf.width();        // pixels
let h: u32          = buf.height();       // pixels
let s: usize        = buf.stride();       // BYTES between row starts
let d: PixelDescriptor = buf.descriptor();   // returned BY VALUE (it is Copy)
```

`descriptor()` returns `PixelDescriptor` **by value**, not `&PixelDescriptor` —
the descriptor is a small `Copy` struct, so read its fields freely
(`buf.descriptor().format`, `.transfer`, `.alpha`, …).

### Allocation

`PixelBuffer::new(w, h, desc)` / `try_new()` allocate a zero-filled buffer with
tight stride; `new_simd_aligned()` / `try_new_simd_aligned()` pad rows for SIMD;
[`from_vec(data, w, h, desc)`](#wrapping-a-decoders-vecu8-no-copy) wraps an
existing `Vec<u8>` (tight stride, no copy). The `try_*` variants return
`Result<_, At<BufferError>>` (a [`whereat`](https://crates.io/crates/whereat)
location wrapper around [`BufferError`] — `AllocationFailed`, `InvalidDimensions`,
`InsufficientData`, `StrideTooSmall`, …); the un-prefixed forms panic on failure
(see [allocation policy](https://docs.rs/zenpixels/latest/zenpixels/#allocation-policy)).
All constructors validate dimensions, stride, and alignment. `into_vec()`
recovers the allocation for pool reuse.

### In-place layout transforms

Layout-changing in-place work goes through one atomic primitive:
`PixelBuffer::transform_in_place` runs a transform over the buffer's own
bytes and adopts the resulting geometry/descriptor/color context in the
same call — the buffer and its bytes can never disagree (there is
deliberately no slice-level equivalent). zenpixels-convert builds three
operations on it, all allocation-free:

- `buf.reduce_to_load_bearing_format_in_place(force)` — bit-exact narrowing
  (drop an all-opaque alpha lane, collapse `R==G==B` to gray, narrow
  bit-replicated U16 to U8) driven by a SIMD content scan;
- `try_adapt_in_place(&mut buf, target)` — metadata re-tags, `Rgba8`↔`Bgra8`
  SIMD B↔R swap, and contract-exact alpha-lane drops (RGBX→RGB and friends);
- `apply_orientation_in_place(&mut buf, orientation)` — orientation baking
  without a second pixel buffer.

The analysis half is allocation-free too: `slice.determine_load_bearing()`
reports `uses_alpha` / `uses_chroma` / `uses_low_bits` (~75 GiB/s fused
scan) so encoders can route formats without rewriting anything.

### Interop

With `imgref` feature: `From<ImgRef<P>>`, `From<ImgVec<P>>`, `as_imgref()`, `try_as_imgref::<P>()` and mutable counterparts. With `rgb` feature: `Pixel` impls for `Rgb<u8>`, `Rgba<u8>`, `Gray<u8>`, `BGRA<u8>`, and their `u16`/`f32` variants.

## Conversion

`zenpixels-convert` re-exports everything from `zenpixels`, so downstream code can depend on it alone.

### Row conversion

`RowConverter` pre-computes a conversion plan from a source/target descriptor pair. Three tiers:

1. **Direct kernels** for common pairs (byte swizzle, depth shift, transfer function LUTs)
2. **Composed plans** for less common pairs (e.g., `RGB8_SRGB` to `RGBA16_LINEAR`)
3. **Hub path** through linear sRGB f32 as universal fallback

### Format negotiation

The cost model separates **effort** (CPU work) from **loss** (information destroyed). `ConvertIntent` controls weighting:

| Intent | Effort | Loss | Use case |
|---|---|---|---|
| `Fastest` | 4x | 1x | Encoding — get there fast |
| `LinearLight` | 1x | 4x | Resize, blur — need linear math |
| `Blend` | 1x | 4x | Compositing — premultiplied alpha |
| `Perceptual` | 1x | 3x | Color grading, sharpening |

`Provenance` tracking lets the cost model know that f32 data decoded from a u8 JPEG has zero loss converting back to u8.

Three entry points: `best_match()` (simple), `best_match_with()` (with consumer costs), `negotiate()` (full control with provenance).

### No silent lossy conversions

Every operation that destroys information requires an explicit policy via `ConvertOptions`:

- **Alpha removal**: `DiscardIfOpaque`, `CompositeOnto { r, g, b }`, `DiscardUnchecked`, or `Forbid`
- **Depth reduction**: `Round`, `Truncate`, or `Forbid`
- **RGB to gray**: requires explicit luma coefficients (`Bt709`, `Bt601`, `Bt2020`, or `DisplayP3`), or `None` to forbid. Y' (encoded luma) semantic — round-trips bit-exactly for `R==G==B`.

Convenience constructors: `ConvertOptions::forbid_lossy()` (safe default) and `ConvertOptions::permissive()` (sensible lossy defaults), with `with_alpha_policy()`, `with_depth_policy()`, etc. for customization.

### Atomic output assembly

`finalize_for_output` couples converted pixels with matching encoder metadata in one step. Prevents the bug where pixel values don't match the embedded ICC/CICP profile.

### Gamut, HDR, Oklab

**Gamut matrices** — 3x3 row-major f32 between BT.709, Display P3, BT.2020. No CMS needed for named-profile conversions.

**HDR** — production HDR→SDR display mapping is a native `ConvertPlan` step behind the `hdr-experimental` feature (`ConvertPlan::new_with_hdr_peak` / `new_with_hdr_config` → run through the same `RowConverter`): an ITU-R BT.2446 Method A curve (`Bt2446A`) plus an OKLch soft-compress knee (`SoftCompress`). Source-peak measurement is the `CllMeasure` trait (`measure_max`, a SIMD CTA-861.3 reading). `quantize_to` is the anchor-aware linear→PQ16 quantizer (reads `DiffuseWhite`, BT.2408 default 203); `ContentLightLevel` and `MasteringDisplay` carry the metadata. The older global `reinhard_*` / `exposure_tonemap` helpers are `#[deprecated]` — reach for the [zentone](https://github.com/imazen/zentone) crate for standalone tone-mapping curves.

**Oklab** — primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`, scalar `rgb_to_oklab()` / `oklab_to_rgb()`, public LMS/XYZ/Oklab matrices. Non-sRGB sources get correct LMS matrices without an intermediate sRGB step.

**CMS** — `PluggableCms` trait (dyn-compatible, accepts `ColorProfileSource` directly — CICP, named profiles, or raw ICC bytes) plugs an external backend into `RowConverter`. `RowTransformMut` is the `&mut self` row-level transform returned by plugins. The `cms-moxcms` feature provides a concrete backend using [moxcms](https://crates.io/crates/moxcms), supporting u8/u16/f32 transforms with automatic profile identification. The older `ColorManagement` / `RowTransform` traits (ICC-bytes-only, `&self`) are retained for backward compatibility.

**ICC identification** — `zenpixels::icc::identify_common(icc_bytes)` recognizes 298 well-known RGB + 69 grayscale profiles via normalized FNV-1a hash lookup (~100ns), including every profile the CICP database itself embeds in encoded output. Returns primaries, transfer function, and `IdentificationUse` (whether matrix+TRC substitution is safe vs CMS-only). Covers sRGB, Display P3, BT.2020, Adobe RGB variants across ICC v2–v5; `extract_cicp` reads embedded `cICP` tags directly.

### Pipeline planner

`CodecFormats` declares each codec's decode outputs and encode inputs, ICC/CICP support, effective bits, and overshoot behavior. The `pipeline` feature enables the format registry, operation requirements, and path solver for multi-step conversion planning.

### Resource estimation

`ConvertPlan::estimate(w, h)` (and `estimate_in(&ImageCharacteristics, &ComputeEnvironment)`) predicts a plan's `peak_memory_bytes_est`, `wall_ms` (already scaled to core count), and `intermediate_buffer_count` *before* running it — cheap to call (walks the planned steps, no row work, no allocation), so schedulers and throttlers can budget ahead. The estimate types are shape-compatible with `zencodec::estimate::*` for wiring `decode → convert → encode` across the boundary, without `zenpixels-convert` depending on `zencodec`. See the [zenpixels-convert README](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md#resource-estimation) for the full contract.

## Planar support

With the `planar` feature: `PlaneLayout`, `PlaneDescriptor`, `PlaneSemantic`, `Subsampling` (4:2:0/4:2:2/4:4:4/4:1:1), `YuvMatrix`, and `MultiPlaneImage` container. Handles YCbCr, Oklab planes, gain maps, and separate alpha planes.

## Features

### zenpixels

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library (currently a no-op; everything is `no_std + alloc`) |
| `icc` | yes | `icc` module — hash-based ICC profile identification (~100ns) |
| `rgb` | | `Pixel` impls for `rgb` crate types, typed `from_pixels()` constructors |
| `imgref` | | `From<ImgRef>` / `From<ImgVec>` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types (YCbCr, Oklab, gain maps) |
| `serde` | | No-op stub (soft-removed in 0.2.16, queued for removal); previously added `Serialize`/`Deserialize` derives on the core types — a workspace-wide sweep found zero consumers |

### zenpixels-convert

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library |
| `icc-db` | yes | Bundled CICP→ICC profile database (~36 KB asset) behind `synthesize_icc_for_cicp` / `synthesize_gray_icc_for_cicp`; disable for size-sensitive builds (they answer `NeedsCms` instead) |
| `fast-transpose` | | SIMD transpose kernels (x86-64 AVX2 + aarch64 NEON) for `apply_orientation*`; portable scalar otherwise (output is bit-identical) |
| `avx512` | | 16-wide AVX-512F f16 conversion kernels (runtime-dispatched) |
| `rgb` | | `Pixel` impls for `rgb` crate types, typed convenience methods (`to_rgb8()`, `to_rgba8()`, etc.) |
| `imgref` | | `ImgRef`/`ImgVec` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types |
| `pipeline` | | Pipeline planner: format registry, operation requirements, path solver |
| `hdr-experimental` | | Native HDR→SDR display mapping inside `ConvertPlan` (BT.2446 Method A + OKLch soft compress + CTA-861.3 CLL measurement); API shape may move ahead of 0.3.0 |
| `cms-moxcms` | | ICC profile transforms via [moxcms](https://crates.io/crates/moxcms) (implies `std`) |
| `serde` | | No-op stub (soft-removed in 0.2.15, queued for removal); previously forwarded to `zenpixels/serde` |

## Build time

`zenpixels` itself compiles in **~0.28s** (release, 7950X). The cold `cargo build --release -p zenpixels` wall is ~1.9s, but 1.6s of that is the serial prerequisite chain `proc-macro2 → syn → bytemuck_derive → bytemuck` — costs most real Rust projects already pay for something else. Edit-rebuild cycles only pay the 0.3s.

## MSRV

`zenpixels` requires Rust 1.85+. `zenpixels-convert` requires Rust 1.89+ (for the safe SIMD intrinsics it uses). 2024 edition.

## License

Apache-2.0 OR MIT

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | **zenpixels** · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
