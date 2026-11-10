<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenpixels

Pixel types for Rust image code. Keep samples, dimensions, stride and color
meaning together.

**Start with the [pixel types guide](https://imazen.github.io/zenpixels/guide.html).**
It goes from owning an image to borrowing a subimage, then covers color,
conversion and native samples. The [interactive explorer](https://imazen.github.io/zenpixels/)
shows the bytes and runs the real converter.

This is the **unpublished 0.3.1 candidate**. The guide follows this source;
[docs.rs](https://docs.rs/zenpixels) describes the published crate. Publication
still requires release approval.

## Quick start

`PixelBuffer` owns the storage. `PixelSlice` borrows an image or rectangular
subimage. `PixelDescriptor` explains the samples.

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};

let decoded = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
let image = PixelBuffer::from_vec(decoded, 2, 2, PixelDescriptor::RGB8_SRGB)?;
let view = image.as_slice();
let right_column = view.crop_view(1, 0, 1, 2);

assert_eq!(right_column.width(), 1);
assert_eq!(right_column.stride(), image.stride()); // Parent stride: six bytes.
assert_eq!(right_column.row(0), &[0, 255, 0]);      // Only the cropped pixel.
# Ok::<(), Box<dyn std::error::Error>>(())
```

No pixels are copied when you borrow or crop. For a fresh zero-filled image,
use `PixelBuffer::try_new(width, height, descriptor)`.

## Two crates

```toml
# Storage, views and descriptions
zenpixels = "0.3.1"

# Conversion; also re-exports the core types
zenpixels-convert = "0.3.1"
```

zenpixels describes and carries samples. zenpixels-convert changes sample
values: transfer curves, gamut, depth, layout, alpha and explicit HDR rendering.
Both support `no_std + alloc` and forbid unsafe code.

## Pixel buffers and views

| Need | API |
|---|---|
| Own tightly packed decoded bytes | `PixelBuffer::from_vec` |
| Allocate a zero-filled image | `PixelBuffer::try_new` |
| Allocate aligned storage with padded rows | `PixelBuffer::<(), AlignedSamples<u8>>::try_new_aligned` |
| Borrow someone else's storage | `PixelSlice::new` |
| Borrow a rectangular subimage | `crop_view` |
| Borrow full-width rows | `sub_rows` on a view; `rows` on a buffer |
| Write pixels | `as_slice_mut().row_mut(y)` |
| Copy a view into packed owned storage | `try_to_tight_buffer` |
| Move storage and metadata together | `into_parts` / `try_from_parts` |

Width and height count pixels. **Stride counts bytes between row starts.**
`row(y)` returns only active pixels, including for crops. An erased byte view
requires sample-aligned row starts, not a whole number of pixels per stride.
The final row needs no trailing padding. Typed constructors also enforce their
Rust pixel type's alignment; `new_typed` takes its stride in pixels.

`from_vec` accepts tight rows. U8 storage can be adopted without a pixel copy;
wide-sample bytes are copied into typed storage when necessary for alignment.
Use `try_from_parts` for an explicit offset/stride and rejected-ownership
recovery. Packed rows do not necessarily imply a zero offset.

Use `as_contiguous_bytes()` when you specifically require packed rows; it
returns `None` for a strided view. Use `try_contiguous_bytes()` to borrow when
packed or make a fallible packed copy otherwise.

## Type system

A `PixelFormat` describes storage and layout. A `PixelDescriptor` adds transfer
function, primaries, alpha and signal range. For example, `RGB8_SRGB` is ordinary
web RGB; `RGBF32_LINEAR` is linear F32 RGB. Declaring a different descriptor
**does not convert samples**.

Most code can use dynamic `PixelBuffer` and `PixelSlice`. With `rgb`, concrete
pixel types are available too, such as `PixelBuffer<rgb::Rgb<u8>>`. Their Rust
type establishes layout; declare the color separately. `.erase()` retains the
storage and metadata while dropping the concrete Rust pixel type.

Signed and wide native samples use `PixelFormat::Generic`, constructed by
`PixelFormat::from_parts`. The existing owners/views carry their exact bytes;
general signed/wide numerical conversion remains unsupported. See the
[sample contract](https://github.com/imazen/zenpixels/blob/main/docs/sample-encoding-contract.md).

## Color and native samples

- `ColorContext` describes the **current** samples, including an attached ICC
  or CICP and luminance anchors. An attached ICC is authoritative even if a
  companion disagrees or the ICC is unusable; unsupported operations refuse or
  preserve the original interpretation.
- `ColorOrigin` keeps original file declarations separately. Converted samples
  must be emitted with matching current metadata.
- `SampleEncoding` describes native integer code depth and bit placement.
  `CodePlane` reads U8/U16 component codes from typed planes. Ten-bit white 1023
  is not ordinary full-range U16 white 65535: canonicalization is explicit.
- `ExactDepth` is evidence about the **current** U8/U16 code grid. Mutable
  access invalidates it. Original source depth is not a lossless-output proof.

`plane::Plane` carries a strided sample plane. `PlanarLayout` / `PlanarSlice`
combine validated roles, geometry and interpretation. The old `planar` module
is removed; its feature spelling remains a no-op. Native code extraction is
separate from chroma reconstruction, range expansion and color conversion.

## Conversion

```rust
use zenpixels_convert::{PixelBuffer, PixelDescriptor, PixelSliceConvertExt};

let image = PixelBuffer::from_vec(vec![128, 128, 128], 1, 1, PixelDescriptor::RGB8_SRGB)?;
let linear = image.as_slice().convert_to(PixelDescriptor::RGBF32_LINEAR)?;
assert_eq!(linear.descriptor(), PixelDescriptor::RGBF32_LINEAR);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `RowConverter` for rows and prepared scratch, or `OutputPlan` for
caller-supplied output rows and matching encoder metadata. `EncodeReady::finalize`
creates independent owned output. Known descriptor-only conversions need no
CMS; arbitrary ICC transforms need a selected backend. HDR rendering requires
explicit luminance/rendering inputs.

Loss policies are explicit with `ConvertOptions`. `retain_channels` prevents
alpha/depth loss but still defaults to gamut clipping; it is not a general
lossless guarantee. `RowConverter::new` has legacy permissive defaults, including
unchecked alpha discard. Use explicit options or `new_preserving_samples` when
preservation is required.

[Conversion guide](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md) ·
[Design contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md)

## Allocation and ownership

Fallible image constructors return located `BufferError` errors. `new` and
other documented convenience paths panic on failure. Some control objects,
lazy caches and dependency allocations remain infallible: see the
[allocation audit](https://github.com/imazen/zenpixels/blob/main/docs/allocation-audit.md).

`into_contiguous()` compacts rows in the current allocation. Orientation can
reuse the pixel allocation, but rectangular in-place transpose needs scratch.
Reusing pixel storage is not a promise that every operation allocates nothing.

## Features

| Core feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `icc` | Default; paired-fingerprint recognition with scoped substitution permission |
| `rgb` | Concrete `rgb` pixel types |
| `imgref` | imgref adapters; enables `rgb` |
| `image` | image 0.25 interop; requires Rust 1.88 |
| `hash` | Pixel `Hash` and configurable inclusions via zenhash; requires Rust 1.89 |
| `allocator_api` | Allocator-parameterized storage on supported compilers |
| `planar`, `serde` | Retained compatibility spellings; no API |

The converter's [feature list](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md#features)
covers the ICC database, CMS backends, interop and HDR options. Resource-estimation
APIs are retired; their compatibility feature does not restore them.

## MSRV

Core requires Rust 1.85. The converter and core `hash` feature require 1.89;
core `image` requires 1.88. Both crates use edition 2024.

## License

Apache-2.0 OR MIT

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | **zenpixels** · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
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
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
