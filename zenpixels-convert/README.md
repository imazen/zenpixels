# zenpixels-convert [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square)](https://crates.io/crates/zenpixels-convert) [![lib.rs](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels-convert) [![docs.rs](https://img.shields.io/docsrs/zenpixels-convert?style=flat-square)](https://docs.rs/zenpixels-convert) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenpixels-convert?style=flat-square)](#license)

Convert pixel samples while keeping their description and output metadata in
agreement. This crate re-exports zenpixels, so most applications need just this
one dependency.

New to the types? Start with the [image and subimage guide](https://imazen.github.io/zenpixels/guide.html).
Use the [interactive explorer](https://imazen.github.io/zenpixels/) to inspect
bytes and try actual conversion routes.

## Install

These examples follow the **unpublished 0.3.1 candidate**. The
[published reference](https://docs.rs/zenpixels-convert) may describe an earlier API.

```toml
zenpixels-convert = "0.3.1"
```

## Quick start

For known descriptor-based color, convert an image or a borrowed subimage:

```rust
use zenpixels_convert::{PixelBuffer, PixelDescriptor, PixelSliceConvertExt};

let image = PixelBuffer::from_vec(
    vec![255, 0, 0, 0, 255, 0], 2, 1, PixelDescriptor::RGB8_SRGB,
)?;
let green = image.crop_view(1, 0, 1, 1);
let linear = green.convert_to(PixelDescriptor::RGBF32_LINEAR)?;

assert_eq!(linear.width(), 1);
assert_eq!(linear.descriptor(), PixelDescriptor::RGBF32_LINEAR);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The crop borrows the parent's storage and stride. The conversion creates a new
output buffer. Changing a descriptor alone would not perform this conversion.

## Convert rows with explicit options

Build the converter once. `prepare` reserves built-in scratch and initializes
selected tables before the loop. Rows may come from a crop or padded storage:

```rust
use zenpixels_convert::{ConvertOptions, PixelBuffer, PixelDescriptor, RowConverter};

let source = PixelBuffer::from_vec(
    vec![64, 128, 192, 255, 0, 0], 2, 1, PixelDescriptor::RGB8_SRGB,
)?;
let mut output = PixelBuffer::try_new(2, 1, PixelDescriptor::RGBF32_LINEAR)?;
let options = ConvertOptions::retain_channels().with_clip_out_of_gamut(false);
let mut converter = RowConverter::new_explicit(
    source.descriptor(), output.descriptor(), &options,
)?;
converter.prepare(source.width())?;

for y in 0..source.height() {
    converter.try_convert_row(
        source.as_slice().row(y),
        output.as_slice_mut().row_mut(y),
        source.width(),
    )?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Within prepared capacity, built-in row execution needs no new scratch
allocation. CMS backends have their own preparation contract. Execution validates
row extents/alignment before writes; a backend failure may leave partial output.
Use `try_clone()` for a fallible independent converter worker.

## Choose what may change

- `RowConverter::new` keeps legacy permissive defaults, including unchecked
  alpha discard. Naming a target is not a general lossless guarantee.
- `ConvertOptions::retain_channels()` forbids alpha/depth loss, but defaults
  to gamut clipping. Add `.with_clip_out_of_gamut(false)` for extended floats.
- Use explicit alpha/depth/luma policies when requesting lossy operations.
  RGB-to-gray requires chosen luma coefficients.
- `ConvertPlan::new_preserving_samples` accepts proven representation changes
  or refuses. It does not infer preservation from original source depth.
- `compose` may remove incidental intermediate quantization;
  `compose_preserving` retains intentional stage boundaries.

## Keep encoder metadata with the result

`EncodeReady::finalize` creates independent owned output and its matching color
metadata together:

```rust
use zenpixels_convert::{
    Cicp, ColorOrigin, EncodeReady, OutputProfile,
    PixelBuffer, PixelDescriptor, PixelFormat,
};

let source = PixelBuffer::from_vec(vec![64, 128, 192], 1, 1, PixelDescriptor::RGB8_SRGB)?;
let ready = EncodeReady::finalize(
    &source, &ColorOrigin::assumed(), OutputProfile::Named(Cicp::SRGB),
    PixelFormat::Rgb16, None,
)?;
assert_eq!(ready.pixels().descriptor().pixel_format(), PixelFormat::Rgb16);
assert_eq!(ready.metadata().cicp, Some(Cicp::SRGB));
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `OutputPlan` when an encoder can borrow unchanged rows or you supply the
output storage. `borrow_rows` can avoid identity pixel copies; `convert_rows`
writes to the supplied destination. Configure the encoder from the plan's
metadata. Original `ColorOrigin` declarations cannot override converted current
pixels. [Output contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md).

## Color profiles and HDR

Named descriptor-based primaries/transfer pairs use the built-in paths.
Arbitrary ICC profiles need a selected `PluggableCms`: enable `cms-moxcms` for
`MoxCms` or `cms-lcms` for `Lcms`. Enabling a feature does not select a backend
for you. Metadata-only ICC recognition does not authorize substitution.

`SynthesizedIcc::for_cicp` describes already-decoded RGB; it does not apply
YCbCr matrices or expand limited-range codes. Unsupported synthesis reports
its refusal. HLG ICC synthesis remains unsupported.

Relative/absolute light conversions need an explicit white anchor. Display HLG
rendering needs explicit display peak, black and effective gamma; an opt-in
named reference preset is available. Tone mapping also needs an explicit
rendering policy and appropriate peak inputs. Source metadata does not silently
select a display. [HDR contract](https://github.com/imazen/zenpixels/blob/main/docs/hdr-anchor-api-design.md).

Native integer code extraction is separate from color conversion.
`codes::to_canonical` handles supported component encodings; generic signed/wide
storage does not grant numerical conversion support.

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `icc-db` | Default; qualified CICP-to-ICC recipes with lazy group decoding |
| `cms-moxcms`, `cms-lcms` | Explicitly selected CMS implementations |
| `rgb`, `imgref` | Core typed-pixel and imgref interop |
| `fast-transpose` | Additional runtime-dispatched orientation kernels |
| `avx512` | Opt-in wider row-kernel tiers |
| `hdr-experimental`, `bt2446a` | HDR measurement/rendering and BT.2446A |
| `planar`, `serde`, `estimation-experimental` | Retained compatibility spellings |

Disabling `icc-db` retains supported compact exports and explicit synthesis
refusal elsewhere. It does not restore the legacy P3 fallback. Resource-estimation
APIs are retired.

## MSRV

Rust 1.89, edition 2024. `no_std + alloc`; unsafe code is forbidden.

## See also

[Pixel types guide](https://imazen.github.io/zenpixels/guide.html) ·
[Core types](https://github.com/imazen/zenpixels) ·
[API reference](https://docs.rs/zenpixels-convert) ·
[Design contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md)

## License

Apache-2.0 OR MIT

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · **zenpixels-convert** · [linear-srgb] · [garb] · [zenyuv] |
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
[zenpixels]: https://github.com/imazen/zenpixels
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
