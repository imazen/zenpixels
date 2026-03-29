# zenpixels ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpixels?style=flat-square) ![docs.rs](https://img.shields.io/docsrs/zenpixels?style=flat-square) ![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square) ![license](https://img.shields.io/crates/l/zenpixels?style=flat-square)

Pixel format types and transfer-function-aware conversion for Rust image codecs.

A JPEG decoder gives you `RGB8` in sRGB. An AVIF decoder gives you `RGBA16` in BT.2020 PQ. A resize library wants `RGBF32` in linear light. Without shared types, every codec pair needs hand-rolled conversion — and gets transfer functions wrong, silently drops alpha, or writes "sRGB" in the ICC profile while the pixels are linear.

zenpixels makes pixel format descriptions first-class types that travel with the data. The conversion crate handles transfer functions, gamut matrices, depth scaling, and alpha compositing so codecs don't have to.

Two crates: **zenpixels** (types, buffers, metadata) and **zenpixels-convert** (all the math). Both are `no_std + alloc`, `forbid(unsafe_code)`, no system dependencies.

```toml
# Types only — for codec crates
zenpixels = "0.2"

# Types + conversion — for processing pipelines
zenpixels-convert = "0.2"
```

## Quick start

```rust
use zenpixels_convert::{RowConverter, best_match, ConvertIntent};

// Pick the cheapest target format the encoder supports
let target = best_match(source_desc, &encoder_formats, ConvertIntent::Fastest)
    .ok_or("no compatible format")?;

// Pre-compute the plan, then convert row by row — no per-row allocation
let mut converter = RowConverter::new(source_desc, target)?;
for y in 0..height {
    converter.convert_row(src_row, dst_row, width);
}
```

## Type system

The core split: *what the bytes are* vs. *what the bytes mean*.

**`PixelFormat`** is a flat enum of byte layouts — channel count, depth, memory order. No color semantics.

```
Rgb8, Rgba8, Rgb16, Rgba16, RgbF32, RgbaF32,
Gray8, Gray16, GrayF32, GrayA8, GrayA16, GrayAF32,
Bgra8, Rgbx8, Bgrx8, OklabF32, OklabaF32
```

**`PixelDescriptor`** wraps a `PixelFormat` with everything needed to interpret the color data:

```rust
pub struct PixelDescriptor {
    pub format: PixelFormat,
    pub transfer: TransferFunction,    // Srgb, Linear, Pq, Hlg, Bt709, Unknown
    pub alpha: Option<AlphaMode>,      // Straight, Premultiplied, Opaque, Undefined
    pub primaries: ColorPrimaries,     // Bt709, DisplayP3, Bt2020, Unknown
    pub signal_range: SignalRange,     // Full or Narrow
}
```

Every buffer carries one. Every codec declares which ones it produces and consumes. 40+ predefined constants:

```rust
PixelDescriptor::RGB8_SRGB        // u8 RGB, sRGB transfer, BT.709 primaries
PixelDescriptor::RGBAF32_LINEAR   // f32 RGBA, linear light
PixelDescriptor::BGRA8_SRGB       // u8 BGRA (Windows/DirectX order)
PixelDescriptor::OKLABF32         // f32 Oklab L,a,b
```

### CICP and ICC

`Cicp` carries ITU-T H.273 code points (used by AVIF, HEIF, JPEG XL, AV1). Named constants for `SRGB`, `DISPLAY_P3`, `BT2100_PQ`, `BT2100_HLG`. Human-readable name lookups via `color_primaries_name()` etc.

`ColorContext` bundles ICC profile bytes and/or CICP codes. Travels with pixel data via `Arc` — cheap to clone, cheap to share across pipeline stages.

`ColorOrigin` is the immutable provenance record: *how the source file described its color*, not what the pixels currently are. Used at encode time to decide whether to re-embed the original profile.

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

Row-level: `row(y)` returns pixel bytes without padding. `row_with_stride(y)` includes padding.

Bulk: `as_strided_bytes()` returns the full backing `&[u8]` including stride padding — zero-copy passthrough to GPU uploads, codec writers, or anything that takes a buffer + stride. `as_contiguous_bytes()` returns `Some` only when rows are tightly packed. `contiguous_bytes()` returns `Cow` — borrows when tight, copies to strip padding otherwise.

Views: `sub_rows(y, count)` and `crop_view(x, y, w, h)` are zero-copy. `crop_copy()` allocates.

### Allocation

`try_new()` for tight stride, `try_new_simd_aligned()` for SIMD-aligned rows, `from_vec()` to wrap an existing allocation. All constructors validate dimensions, stride, and alignment. `into_vec()` recovers the allocation for pool reuse.

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
- **RGB to gray**: requires explicit luma coefficients (`Bt709` or `Bt601`), or `None` to forbid

Convenience constructors: `ConvertOptions::forbid_lossy()` (safe default) and `ConvertOptions::permissive()` (sensible lossy defaults), with `with_alpha_policy()`, `with_depth_policy()`, etc. for customization.

### Atomic output assembly

`finalize_for_output` couples converted pixels with matching encoder metadata in one step. Prevents the bug where pixel values don't match the embedded ICC/CICP profile.

### Gamut, HDR, Oklab

**Gamut matrices** — 3x3 row-major f32 between BT.709, Display P3, BT.2020. No CMS needed for named-profile conversions.

**HDR** — Reinhard and exposure tone mapping, `ContentLightLevel` and `MasteringDisplay` metadata.

**Oklab** — primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`, scalar `rgb_to_oklab()` / `oklab_to_rgb()`, public LMS/XYZ/Oklab matrices. Non-sRGB sources get correct LMS matrices without an intermediate sRGB step.

**CMS** — `ColorManagement` and `RowTransform` traits for ICC-to-ICC transforms. The `cms-moxcms` feature provides a concrete backend using [moxcms](https://crates.io/crates/moxcms), supporting u8/u16/f32 transforms with automatic profile identification.

### Pipeline planner

`CodecFormats` declares each codec's decode outputs and encode inputs, ICC/CICP support, effective bits, and overshoot behavior. The `pipeline` feature enables the format registry, operation requirements, and path solver for multi-step conversion planning.

## Planar support

With the `planar` feature: `PlaneLayout`, `PlaneDescriptor`, `PlaneSemantic`, `Subsampling` (4:2:0/4:2:2/4:4:4/4:1:1), `YuvMatrix`, and `MultiPlaneImage` container. Handles YCbCr, Oklab planes, gain maps, and separate alpha planes.

## Features

### zenpixels

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library (currently a no-op; everything is `no_std + alloc`) |
| `rgb` | | `Pixel` impls for `rgb` crate types, typed `from_pixels()` constructors |
| `imgref` | | `From<ImgRef>` / `From<ImgVec>` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types (YCbCr, Oklab, gain maps) |
| `serde` | | `Serialize`/`Deserialize` derives on all core types |

### zenpixels-convert

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library |
| `rgb` | | `Pixel` impls for `rgb` crate types, typed convenience methods (`to_rgb8()`, `to_rgba8()`, etc.) |
| `imgref` | | `ImgRef`/`ImgVec` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types |
| `pipeline` | | Pipeline planner: format registry, operation requirements, path solver |
| `cms-moxcms` | | ICC profile transforms via [moxcms](https://crates.io/crates/moxcms) (implies `std`) |
| `serde` | | Forwards to `zenpixels/serde` |

## MSRV

Rust 1.93+, 2024 edition.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs<sup>[1]</sup> | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | **zenpixels** · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sup>[1]</sup> <sub>as of 2026</sub>

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
