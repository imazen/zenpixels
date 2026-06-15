# zenpixels [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels?style=flat-square)](https://crates.io/crates/zenpixels) [![lib.rs](https://img.shields.io/crates/v/zenpixels?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels) [![docs.rs](https://img.shields.io/docsrs/zenpixels?style=flat-square)](https://docs.rs/zenpixels) [![license](https://img.shields.io/crates/l/zenpixels?style=flat-square)](https://github.com/imazen/zenpixels#license)

Pixel format types and transfer-function-aware conversion for Rust image codecs.

A JPEG decoder gives you `RGB8` in sRGB. An AVIF decoder gives you `RGBA16` in BT.2020 PQ. A resize library wants `RGBF32` in linear light. Without shared types, every codec pair needs hand-rolled conversion — and gets transfer functions wrong, silently drops alpha, or writes "sRGB" in the ICC profile while the pixels are linear.

zenpixels makes pixel format descriptions first-class types that travel with the data. The conversion crate handles transfer functions, gamut matrices, depth scaling, and alpha compositing so codecs don't have to.

Two crates: **zenpixels** (types, buffers, metadata) and **zenpixels-convert** (all the math). Both are `no_std + alloc`, `forbid(unsafe_code)`, no system dependencies.

```toml
# Types only — for codec crates
zenpixels = "0.2.14"

# Types + conversion — for processing pipelines
zenpixels-convert = "0.2.14"
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
    pub alpha: Option<AlphaMode>,      // Straight, Premultiplied, Opaque, Undefined
    pub primaries: ColorPrimaries,     // Bt709, DisplayP3, Bt2020, AdobeRgb, Unknown
    pub signal_range: SignalRange,     // Full or Narrow
}
```

Every buffer carries one. Every codec declares which ones it produces and consumes. Predefined constants for the common cases:

```rust
PixelDescriptor::RGB8_SRGB        // u8 RGB, sRGB transfer, BT.709 primaries
PixelDescriptor::RGBAF32_LINEAR   // f32 RGBA, linear light
PixelDescriptor::BGRA8_SRGB       // u8 BGRA (Windows/DirectX order)
PixelDescriptor::RGB16_BT2100_PQ  // u16 RGB, BT.2020 + PQ — HDR10-style stills
PixelDescriptor::OKLABF32         // f32 Oklab L,a,b
```

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
(fresh buffer, SIMD-tiled transpose for 4-byte pixels), `apply_orientation_into`
(caller-provided target, no allocation), and `apply_orientation_in_place`
(reuses the buffer's own allocation; see below).

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

### Allocation

`try_new()` for tight stride, `try_new_simd_aligned()` for SIMD-aligned rows, `from_vec()` to wrap an existing allocation. All constructors validate dimensions, stride, and alignment. `into_vec()` recovers the allocation for pool reuse.

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

**HDR** — Reinhard and exposure tone mapping, `ContentLightLevel` and `MasteringDisplay` metadata.

**Oklab** — primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`, scalar `rgb_to_oklab()` / `oklab_to_rgb()`, public LMS/XYZ/Oklab matrices. Non-sRGB sources get correct LMS matrices without an intermediate sRGB step.

**CMS** — `PluggableCms` trait (dyn-compatible, accepts `ColorProfileSource` directly — CICP, named profiles, or raw ICC bytes) plugs an external backend into `RowConverter`. `RowTransformMut` is the `&mut self` row-level transform returned by plugins. The `cms-moxcms` feature provides a concrete backend using [moxcms](https://crates.io/crates/moxcms), supporting u8/u16/f32 transforms with automatic profile identification. The older `ColorManagement` / `RowTransform` traits (ICC-bytes-only, `&self`) are retained for backward compatibility.

**ICC identification** — `zenpixels::icc::identify_common(icc_bytes)` recognizes 298 well-known RGB + 69 grayscale profiles via normalized FNV-1a hash lookup (~100ns), including every profile the CICP database itself embeds in encoded output. Returns primaries, transfer function, and `IdentificationUse` (whether matrix+TRC substitution is safe vs CMS-only). Covers sRGB, Display P3, BT.2020, Adobe RGB variants across ICC v2–v5; `extract_cicp` reads embedded `cICP` tags directly.

### Pipeline planner

`CodecFormats` declares each codec's decode outputs and encode inputs, ICC/CICP support, effective bits, and overshoot behavior. The `pipeline` feature enables the format registry, operation requirements, and path solver for multi-step conversion planning.

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
| `serde` | | `Serialize`/`Deserialize` derives on all core types |

### zenpixels-convert

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library |
| `icc-db` | yes | Bundled CICP→ICC profile database (~36 KB asset) behind `synthesize_icc_for_cicp` / `synthesize_gray_icc_for_cicp`; disable for size-sensitive builds (they answer `NeedsCms` instead) |
| `avx512` | | 16-wide AVX-512F f16 conversion kernels (runtime-dispatched) |
| `rgb` | | `Pixel` impls for `rgb` crate types, typed convenience methods (`to_rgb8()`, `to_rgba8()`, etc.) |
| `imgref` | | `ImgRef`/`ImgVec` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types |
| `pipeline` | | Pipeline planner: format registry, operation requirements, path solver |
| `cms-moxcms` | | ICC profile transforms via [moxcms](https://crates.io/crates/moxcms) (implies `std`) |
| `serde` | | Forwards to `zenpixels/serde` |

## Build time

`zenpixels` itself compiles in **~0.28s** (release, 7950X). The cold `cargo build --release -p zenpixels` wall is ~1.9s, but 1.6s of that is the serial prerequisite chain `proc-macro2 → syn → bytemuck_derive → bytemuck` — costs most real Rust projects already pay for something else. Edit-rebuild cycles only pay the 0.3s.

## MSRV

`zenpixels` requires Rust 1.85+. `zenpixels-convert` requires Rust 1.89+ (for the safe SIMD intrinsics it uses). 2024 edition.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | **zenpixels** · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Apache-2.0 OR MIT

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
