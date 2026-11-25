# zenpixels [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels?style=flat-square)](https://crates.io/crates/zenpixels) [![lib.rs](https://img.shields.io/crates/v/zenpixels?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels) [![docs.rs](https://img.shields.io/docsrs/zenpixels?style=flat-square)](https://docs.rs/zenpixels) [![MSRV](https://img.shields.io/badge/MSRV-1.85-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenpixels?style=flat-square)](#license)

Pixel types for Rust image pipelines. Keep samples, dimensions, stride, bit
packing, and color meaning together — `no_std + alloc`, `#![forbid(unsafe_code)]`.

This is the **0.5.0 greenfield candidate**. [docs.rs](https://docs.rs/zenpixels)
describes the latest published crate.

## Quick start

`PixelBuf` owns alignment-exact storage. `PixelView<'a>` / `PixelViewMut<'a>`
borrow an image or rectangular subimage. `ImageSpec` pairs a `SampleLayout` with
an algebraic `ColorSpace`.

```rust
use zenpixels::{ImageSpec, PixelBuf};

let decoded = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
let image = PixelBuf::from_vec(decoded, 2, 2, ImageSpec::rgb8_srgb())?;
let right_column = image.crop_view(1, 0, 1, 2)?;

assert_eq!(right_column.width(), 1);
assert_eq!(right_column.stride(), image.stride()); // Parent stride: 6 bytes.
assert_eq!(right_column.row(0), &[0, 255, 0]);     // Only the cropped pixel.
# Ok::<(), Box<dyn std::error::Error>>(())
```

No pixels are copied when you borrow or crop. Both `image.crop_view(...)` and
`image.as_view().crop(...)` preserve the buffer's `'a` lifetime. For a fresh
zero-filled image, use `PixelBuf::new(width, height, spec)`.

## Two crates

```toml
# Storage, views, color spaces, HDR metadata, planes (lightweight — use in codecs)
zenpixels = "0.5.0"

# Conversion; re-exports zenpixels::* with zero module name collisions
zenpixels-convert = "0.5.0"
```

## Core architecture (0.5)

- **Algebraic `ColorSpace` + `Copy`/`const` `PixelFormat`:** mutually exclusive
  variants (`Rgb`, `Gray`, `Oklab`, `Xyz`, `Perceptual`, `Cmyk`, `Icc`,
  `Untagged`) make invalid states (such as Oklab carrying BT.709 primaries or
  ICC contradicting CICP) unrepresentable. `ImageSpec` provides `const` presets
  (`ImageSpec::RGB8_SRGB`, etc.), `SpecCow<'a>` avoids `Arc` bumps across views
  and crops, and `PixelFormat` (`Copy + const + !Drop`) serves codec tables and
  `const` slices without heap ownership.
- **Lossless ITU-T H.273 CICP & ICC.1 `cicpType` amendment:** all 11 Table 2
  primaries (including projective `y == 0` CIE XYZ), all 16 Table 3 transfers,
  all 9 Table 4 NCL matrix codes, ICC.1 `cicpType` 44-byte custom-primaries
  extension (`CicpTag`, signed `s15Fixed16Number` supporting negative
  chromaticities such as `CustomPrimaries::ACES_AP0`, `ACES_CG`, `PROPHOTO_RGB`,
  and Annex 1 `to_icc_d50_matrix_f64`), exact `0.0001 cd/m²` `ClliCodes`, and
  ISO 15076-1 structural ICC validation (`validate_icc_structure`) wired into
  `ImageSpec::validate()`.
- **Unified `SampleLayout` + `BitPacking`:** full-width samples (`U8`, `U16`,
  `F16`, `F32`, `I16`, `I32`, `U32`), 10/12-bit codec planes (`P010`, `U10_LSB`,
  `U12_LSB`), and bitfield words (`RGB565`, `RGB10A2`).
- **Sound `AlphaMode::Opaque` & fast fill-then-retag:** `PixelBuf::new` fills
  black with canonical max alpha, `PixelBuf::new_white` / `new_with_fill` fill
  reference white, and `PixelBuf::new_zeroed` skips the alpha-fill pass (refusing
  `AlphaMode::Opaque` on alpha-bearing layouts) so decoders can write all pixels
  and retag zero-copy via `into_opaque_verified`, `into_opaque_filled`, or
  `assume_opaque`.
- **Alignment-exact zero-copy interchange & external ownership:** `PixelBuf::from_vec<P>`,
  `try_from_vec_recover`, `from_external` / `from_external_mut`, and
  `try_into_vec_zero_copy<P>` preserve Rust's allocation alignment contract via
  `PixelStorage` (`U8`, `U16`, `U32`, `F32`, `Aligned64`, `External`, `ExternalMut`).
- **Orientation on buffers & multi-plane/dual-layer carriers:** `Orientation`
  travels directly on `PixelBuf`, `PixelView`, `PixelViewMut`, `BandedBuf`,
  `PlanarBuf`, `PlanarView`, `PlanarViewMut`, and `DualLayerHdr` (ISO 21496-1 /
  UltraHDR gain maps) with O(1) `rotate_metadata` and `display_dimensions()`.
  Planar views expose zero-copy typed codec APIs (`PlaneView::from_samples`,
  `row_as`, `code_u32`, `as_code_plane`, `plane_by_role`, `PlanarBuf::new_aligned`).

## Conversion

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PeakSource, PixelBuf, convert_buf};

let src = PixelBuf::from_vec(vec![128, 128, 128], 1, 1, ImageSpec::rgb8_srgb())?;
let linear = convert_buf(
    &src.as_view(), ImageSpec::rgbf32_linear(), &ConvertPolicy::permissive(), PeakSource::Measured,
)?;
assert_eq!(linear.spec(), &ImageSpec::rgbf32_linear());
# Ok::<(), Box<dyn std::error::Error>>(())
```

See [zenpixels-convert/README.md](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md)
for `RowPlan`, `ImagePlan`, `OutputPlan`, `Negotiate`, `const Accelerators`,
HDR tone mapping, and CMS backends (`MoxCms`, `Lcms`).

## Features (`zenpixels`)

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `icc` | Default; paired-fingerprint ICC recognition with intent-scoped substitution |
| `rgb` | `PixelType` trait & typed `rgb` crate interop |
| `imgref` | `imgref` adapters; enables `rgb` |
| `image` | `image` 0.25 interop with CICP color-space propagation; requires Rust 1.88 |

## MSRV

`zenpixels` requires Rust 1.85 (`image` feature: 1.88). `zenpixels-convert`
requires Rust 1.89. Both crates use edition 2024.

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
