# zenpixels

[![CI](https://github.com/imazen/zenpixels/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/zenpixels?style=for-the-badge)](https://crates.io/crates/zenpixels)
[![docs.rs](https://img.shields.io/docsrs/zenpixels?style=for-the-badge)](https://docs.rs/zenpixels)
[![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=for-the-badge)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field)
[![license](https://img.shields.io/crates/l/zenpixels?style=for-the-badge)](LICENSE-MIT)

Pixel format interchange types and transfer-function-aware conversion for Rust image codecs.

Two crates, one split: **zenpixels** defines the types, **zenpixels-convert** does the math. Codecs depend on the interchange crate for zero-cost format descriptions. Processing pipelines pull in the conversion crate when they need to change pixel formats, apply transfer functions, or negotiate the cheapest encode path.

Both crates are `no_std + alloc`, `forbid(unsafe_code)`, and have no system dependencies.

## Why

Image codecs produce pixels in different formats. A JPEG decoder gives you `RGB8` in sRGB. An AVIF decoder might give you `RGBA16` in BT.2020 PQ. A resize library wants `RGBF32` in linear light. A PNG encoder takes `RGBA8` or `RGBA16`.

Without a shared type system, every codec pair needs hand-rolled conversion code. Transfer functions get applied in the wrong order (or not at all). Alpha gets silently discarded. Metadata says sRGB but the pixels are linear. These bugs are subtle, hard to test, and widespread.

zenpixels fixes this by making pixel format descriptions first-class types that travel with the pixel data. The conversion crate handles all the math — transfer functions, gamut matrices, depth scaling, alpha compositing — so codecs don't have to.

## The type system

The core design separates *what the bytes are* from *what the bytes mean*.

### PixelFormat — byte layout

`PixelFormat` is a flat enum describing the physical pixel layout: channel count, channel depth, and byte order. Nothing about color interpretation.

```
Rgb8, Rgba8, Rgb16, Rgba16, RgbF32, RgbaF32,
Gray8, Gray16, GrayF32, GrayA8, GrayA16, GrayAF32,
Bgra8, Rgbx8, Bgrx8, OklabF32, OklabaF32
```

Use this for exhaustive `match` dispatch over known layouts. It answers questions like "how many bytes per pixel?" and "is there an alpha channel?" — but not "is this sRGB or linear?" or "are these BT.709 or BT.2020 primaries?"

### PixelDescriptor — full meaning

`PixelDescriptor` wraps a `PixelFormat` with everything needed to interpret the color data correctly:

```rust
pub struct PixelDescriptor {
    pub format: PixelFormat,           // byte layout
    pub transfer: TransferFunction,    // sRGB, Linear, PQ, HLG, BT.709, Unknown
    pub alpha: Option<AlphaMode>,      // None, Straight, Premultiplied, Opaque, Undefined
    pub primaries: ColorPrimaries,     // BT.709, BT.2020, Display P3, Unknown
    pub signal_range: SignalRange,     // Full (0-255) or Narrow (16-235)
}
```

This is the unit of currency in the zen ecosystem. Every buffer carries one. Every codec declares which ones it produces and consumes. The conversion system uses pairs of them to build conversion plans.

40+ predefined constants follow a naming convention: `{FORMAT}_{TRANSFER}` for concrete descriptors, `{FORMAT}` for transfer-agnostic ones.

```rust
// Concrete — transfer function is known
PixelDescriptor::RGB8_SRGB        // u8 RGB, sRGB transfer, BT.709 primaries
PixelDescriptor::RGBAF32_LINEAR   // f32 RGBA, linear light, BT.709 primaries
PixelDescriptor::BGRA8_SRGB       // u8 BGRA, sRGB transfer (Windows/DirectX order)

// Transfer-agnostic — for negotiation when transfer doesn't matter
PixelDescriptor::RGB8             // u8 RGB, transfer unknown
PixelDescriptor::RGBA16           // u16 RGBA, transfer unknown

// Perceptual color
PixelDescriptor::OKLABF32         // f32 Oklab L,a,b
PixelDescriptor::OKLABAF32        // f32 Oklab L,a,b + alpha
```

### Building blocks

Each axis of `PixelDescriptor` is its own enum:

**`ChannelType`** — storage per channel: `U8`, `U16`, `F16`, `F32`.

**`ChannelLayout`** — what the channels represent: `Gray`, `GrayAlpha`, `Rgb`, `Rgba`, `Bgra`, `Oklab`, `OklabA`.

**`TransferFunction`** — the electro-optical transfer function: `Linear`, `Srgb`, `Bt709`, `Pq` (HDR10), `Hlg`, `Unknown`.

**`ColorPrimaries`** — RGB chromaticities: `Bt709` (sRGB), `DisplayP3`, `Bt2020`, `Unknown`. Discriminant values match CICP codes. Supports gamut containment queries (`Bt2020.contains(DisplayP3)` is true).

**`AlphaMode`** — how to interpret the alpha channel: `Straight` (unassociated), `Premultiplied` (associated), `Opaque` (all 0xFF), `Undefined` (padding, as in RGBX/BGRX). Wrapped in `Option` — `None` means no alpha channel exists at all.

**`SignalRange`** — `Full` (0–255 for u8) or `Narrow` (16–235 luma, 16–240 chroma). Matters for video formats.

### Cicp

ITU-T H.273 code points, used by HEIF, AVIF, JPEG XL, and AV1 to signal color space in-band.

```rust
pub struct Cicp {
    pub color_primaries: u8,           // 1=BT.709, 9=BT.2020, 12=Display P3
    pub transfer_characteristics: u8,  // 1=BT.709, 13=sRGB, 16=PQ, 18=HLG
    pub matrix_coefficients: u8,       // 0=Identity, 1=BT.709, 6=BT.601, 9=BT.2020
    pub full_range: bool,              // true=0-255, false=16-235
}

// Named constants for common profiles
Cicp::SRGB          // (1, 13, 6, true)
Cicp::DISPLAY_P3    // (12, 13, 0, true)
Cicp::BT2100_PQ     // (9, 16, 9, true)
Cicp::BT2100_HLG    // (9, 18, 9, true)
```

Human-readable name lookups: `color_primaries_name()`, `transfer_characteristics_name()`, `matrix_coefficients_name()`.

### ColorContext

Bundles ICC profile bytes and/or CICP codes. Travels with pixel data via `Option<Arc<ColorContext>>` on buffers. Cheap to clone, cheap to share across pipeline stages.

```rust
let ctx = ColorContext::from_icc_and_cicp(icc_bytes, cicp);
assert!(ctx.is_srgb());
let tf = ctx.transfer_function(); // derived from CICP if available
```

`ColorOrigin` is the immutable provenance record — it tracks *how the source file described its color* (ICC, CICP, gAMA+cHRM, or assumed), not what the pixels currently are. Used at encode time to decide whether to re-embed the original profile.

## Pixel buffers

`PixelBuffer`, `PixelSlice`, and `PixelSliceMut` are always available (no feature gate). They provide format-aware pixel storage that carries its own `PixelDescriptor` and optional `ColorContext`.

### Typed vs. type-erased

Buffers are generic over `P: Pixel`. When `P` is a concrete type, format correctness is enforced at compile time. Call `.erase()` to get a type-erased buffer for dynamic dispatch, and `.try_typed::<Q>()` to recover the type.

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};
use rgb::Rgba;

// Typed — format enforced at compile time
let buf = PixelBuffer::<Rgba<u8>>::from_pixels(pixels, width, height);

// Type-erased for codec dispatch
let erased = buf.erase();
assert_eq!(erased.descriptor(), PixelDescriptor::RGBA8);

// Recover the type
let typed = erased.try_typed::<Rgba<u8>>().unwrap();
```

### The Pixel trait

Maps concrete pixel types to their `PixelDescriptor`. Open trait — custom types can implement it.

```rust
pub trait Pixel: bytemuck::Pod {
    const DESCRIPTOR: PixelDescriptor;
}
```

Built-in impls for `Rgbx` and `Bgrx` (32-bit SIMD-friendly padded types) and `GrayAlpha8`/`GrayAlpha16`/`GrayAlphaF32` are always available. With the `rgb` feature, you also get impls for `Rgb<u8>`, `Rgba<u8>`, `Gray<u8>`, `BGRA<u8>`, and their `u16`/`f32` variants.

### Buffer operations

**`PixelBuffer<P>`** (owned):
- `try_new(w, h, desc)` — tight stride
- `try_new_simd_aligned(w, h, desc, align)` — SIMD-aligned rows (e.g., 32-byte)
- `from_vec(data, w, h, desc)` — wrap existing allocation
- `from_pixels(pixels, w, h)` — from typed `Vec<P>` (requires `rgb` feature)
- `as_slice()` / `as_slice_mut()` — borrow as `PixelSlice` / `PixelSliceMut`
- `rows(y, count)` / `rows_mut(y, count)` — borrow a sub-range of rows
- `crop_view(x, y, w, h)` — zero-copy rectangle view
- `crop_copy(x, y, w, h)` — copy a rectangle into a new buffer
- `into_vec()` — recover the allocation for pool reuse

**`PixelSlice<'a, P>`** (borrowed, immutable) and **`PixelSliceMut<'a, P>`** (borrowed, mutable):
- `row(y)` / `row_mut(y)` — pixel bytes for a single row (no padding)
- `row_with_stride(y)` — full stride bytes including padding
- `as_strided_bytes()` — raw backing bytes including stride padding (zero-copy)
- `as_contiguous_bytes()` — raw bytes when rows are tightly packed (`None` if padded)
- `contiguous_bytes()` — `Cow::Borrowed` when tight, copies to strip padding otherwise
- `sub_rows(y, count)` — zero-copy vertical slice
- `crop_view(x, y, w, h)` — zero-copy rectangle view
- `is_contiguous()` — true when `stride == width * bpp`

With the `imgref` feature, `From<ImgRef<P>>` and `From<ImgVec<P>>` conversions are available for interop with the `imgref` ecosystem. `PixelBuffer` also provides `as_imgref()`, `try_as_imgref::<P>()`, and their mutable counterparts for type-erased buffers.

## Conversion (zenpixels-convert)

All the pixel math lives here. Re-exports everything from `zenpixels`, so downstream code can depend on this crate alone.

### Row conversion

`RowConverter` pre-computes a conversion plan from a source/target `PixelDescriptor` pair. No per-row allocation or dispatch.

```rust
use zenpixels_convert::{RowConverter, best_match, ConvertIntent};

// Pick the cheapest format the encoder supports
let target = best_match(source_desc, &encoder_formats, ConvertIntent::Fastest)
    .ok_or("no compatible format")?;

// Convert row by row
let converter = RowConverter::new(source_desc, target)?;
for y in 0..height {
    converter.convert_row(src_row, dst_row, width);
}
```

Three tiers, from fastest to most general:

1. **Direct SIMD kernels** for common pairs (byte swizzle, depth shift, transfer function LUTs).
2. **Composed multi-step plans** for less common pairs (e.g., `RGB8_SRGB` to `RGBA16_LINEAR`).
3. **Hub path** through linear sRGB f32 as a universal fallback.

The converter picks the fastest tier that covers the requested pair.

### Format negotiation

The cost model separates **effort** (CPU work) from **loss** (information destroyed). `ConvertIntent` controls how they're weighted:

| Intent | Effort | Loss | Use case |
|---|---|---|---|
| `Fastest` | 4x | 1x | Encoding — get there fast |
| `LinearLight` | 1x | 4x | Resize, blur — need linear math |
| `Blend` | 1x | 4x | Compositing — need premultiplied alpha |
| `Perceptual` | 1x | 3x | Color grading, sharpening |

Provenance tracking lets the cost model know that f32 data decoded from a u8 JPEG has zero loss converting back to u8. Without this, the model would penalize the round-trip as lossy.

### No silent lossy conversions

Every operation that destroys information requires an explicit policy via `ConvertOptions`:

- **Alpha removal**: `DiscardIfOpaque` (error if not opaque), `CompositeOnto { r, g, b }` (flatten onto background), `DiscardUnchecked`, or `Forbid`.
- **Depth reduction**: `Round`, `Truncate`, or `Forbid`.
- **RGB to gray**: requires explicit luma coefficients (`Bt709` or `Bt601`), or `None` to forbid.

### Atomic output assembly

`finalize_for_output` couples converted pixels with matching encoder metadata in one step. Prevents the most common color management bug: pixel values that don't match the embedded ICC/CICP profile.

### Additional capabilities

- **Gamut matrices** — 3x3 row-major f32 conversion matrices between named primaries. No CMS needed for sRGB/Display P3/BT.2020 conversions.
- **HDR** — Reinhard and exposure tone mapping, content light level metadata.
- **Oklab** — primaries-aware `rgb_to_lms_matrix()` and `lms_to_rgb_matrix()`, scalar `rgb_to_oklab()` and `oklab_to_rgb()` functions, and public LMS/XYZ/Oklab matrices (`LMS_FROM_XYZ`, `OKLAB_FROM_LMS_CBRT`, etc.). Non-sRGB sources get correct LMS matrices without an intermediate sRGB step.
- **CMS traits** — `ColorManagement` and `RowTransform` for ICC-to-ICC transforms via external CMS backends.
- **Codec format registry** — `CodecFormats` struct where each codec declares its decode outputs and encode inputs, ICC/CICP support, effective bits, and whether values can overshoot `[0.0, 1.0]`.

## Planar support

With the `planar` feature, zenpixels handles multi-plane images: YCbCr 4:2:0/4:2:2/4:4:4, Oklab planes, gain maps, and separate alpha planes.

- **`PlaneLayout`** — `Interleaved { channels }` or `Planar { planes, relationship }`.
- **`PlaneDescriptor`** — per-plane semantic label, channel type, and subsampling factors.
- **`PlaneSemantic`** — `Luma`, `ChromaCb`, `ChromaCr`, `OklabL`, `OklabA`, `OklabB`, `Alpha`, `GainMap`, `Red`, `Green`, `Blue`, `Depth`.
- **`Subsampling`** — `S444`, `S422`, `S420`, `S411` with `h_factor()`/`v_factor()` accessors.
- **`YuvMatrix`** — `Identity`, `Bt601`, `Bt709`, `Bt2020` with `rgb_to_y_coeffs()`.
- **`MultiPlaneImage`** — bundles a `PlaneLayout`, per-plane `PixelBuffer`s, and shared `ColorContext`.

## Features

### zenpixels

| Feature | What it enables |
|---|---|
| `std` | Standard library (default; currently a no-op, everything is `no_std + alloc`) |
| `rgb` | `Pixel` impls for `rgb` crate types, typed `from_pixels()` constructors |
| `imgref` | `From<ImgRef>` / `From<ImgVec>` conversions (implies `rgb`) |
| `buffer` | Convenience: enables both `rgb` and `imgref` |
| `planar` | Multi-plane image types (YCbCr, Oklab, gain maps) |

### zenpixels-convert

| Feature | What it enables |
|---|---|
| `std` | Standard library (default) |
| `rgb` | Enables `rgb` crate `Pixel` impls (forwarded to zenpixels) |
| `imgref` | `ImgRef`/`ImgVec` conversions (implies `rgb`) |
| `buffer` | `PixelBufferConvertExt` — `convert_to()`, `to_rgb8()`, `to_rgba8()`, etc. |
| `planar` | Multi-plane image types (forwarded to zenpixels) |
| `codec` | Format registry and negotiation (implies `buffer`) |

## Ecosystem

zenpixels is the pixel format layer for the zen codec family:

- **zenjpeg** — JPEG
- **zenpng** — PNG
- **zenwebp** — WebP
- **zenjxl** — JPEG XL
- **zenavif** — AVIF
- **zengif** — GIF
- **zenbitmaps** — BMP/TIFF
- **zenresize** — image resizing
- **zencodecs** — unified codec dispatch

All zen codecs use `PixelDescriptor` to describe their output and `CodecFormats` to declare their capabilities. The conversion crate handles all format bridging between them.

## MSRV

Rust 1.93+, 2024 edition.

## License

Apache-2.0 OR MIT
