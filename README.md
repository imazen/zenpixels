# zenpixels [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels?style=flat-square)](https://crates.io/crates/zenpixels) [![lib.rs](https://img.shields.io/crates/v/zenpixels?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels) [![docs.rs](https://img.shields.io/docsrs/zenpixels?style=flat-square)](https://docs.rs/zenpixels) [![MSRV](https://img.shields.io/badge/MSRV-1.85-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenpixels?style=flat-square)](#license)

Pixel types for Rust image pipelines. Keep samples, dimensions, stride, bit
packing, orientation, and color meaning together — `#![forbid(unsafe_code)]`,
supporting `no_std + alloc` (disable default `std` feature).

This is the **0.5.0 greenfield candidate**. [docs.rs](https://docs.rs/zenpixels)
describes the latest published crate.

- **Start by struct/enum name?** Browse the [Type Atlas & Visual Drilldown](docs/types-and-relationships.md) (`PixelFormat`, `PixelBuf`, `PixelView`, `PlanarBuf`, `ConvertPolicy`, …).
- **New to `zenpixels`?** Read the task-oriented [User Guide & Cookbook](docs/guide.md).
- **Upgrading from 0.2 / 0.4?** See [docs/migration-0.2-to-0.5.md](docs/migration-0.2-to-0.5.md).
- **Need color conversion, HDR, orientation, or planar reconstruction?** See [zenpixels-convert/README.md](zenpixels-convert/README.md).

## The 4 types you need on day one

| Type | Role |
|---|---|
| `PixelFormat` | Pairs physical `SampleLayout` + algebraic `ColorSpace` (RGB, Gray, Oklab, XYZ, CMYK, ICC, Untagged). Use `PixelFormat::RGB8_SRGB` for owned specs and `&formats::RGB8_SRGB` for `'static` borrowed views. |
| `PixelBuf` | Owned, stride-preserving, alignment-exact 2D buffer (can adopt `Vec<T>`, `Arc<[u8]>`, `Box<[T]>`, or custom FFI storage zero-copy). Exposes direct `row(y)` / `row_mut(y)` access and carries `Orientation`, `HdrMetadata`, `ColorOrigin`, and `SampleFacts`. |
| `PixelView<'a>` / `PixelViewMut<'a>` | Zero-copy borrowed strided views and rectangular crops. Primary constructors take `&'a PixelFormat` for guaranteed zero heap allocation. |
| `ParametricFormat` | `Copy + const` layout + parametric color pair (no ICC) for static codec capability tables (`&[ParametricFormat]`). |

## Quick start

```rust
use zenpixels::{PixelBuf, PixelFormat, PixelRect, PixelView, formats};

// Wrap typed pixel data and take a zero-copy crop:
let decoded = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
let image = PixelBuf::from_typed(decoded, 2, 2, PixelFormat::RGB8_SRGB)?;
let right_col = image.crop_view(PixelRect::new(1, 0, 1, 2))?;

assert_eq!((right_col.width(), right_col.height()), (1, 2));
assert_eq!(right_col.stride_bytes(), image.stride_bytes()); // Parent stride: 6 bytes.
assert_eq!(right_col.row(0), &[0, 255, 0]);                 // Only the cropped pixel.

// Borrow raw bytes with zero heap allocation using a static preset:
let raw = [10u8, 20, 30, 40, 50, 60];
let view = PixelView::from_bytes(&raw, 2, 1, &formats::RGB8_SRGB)?;
assert_eq!(view.row(0), &[10, 20, 30, 40, 50, 60]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

> **Quick tips:**
> - `PixelBuf::from_typed` checks the Rust element type against `spec.layout` (annotate literals like `vec![255u8, ...]`, or use `PixelBuf::from_bytes` for raw byte buffers).
> - Carrier/pipeline errors return `whereat::At<E>` — inspect the inner enum with `*err.error()` (or `err.decompose()` / `err.map_error(...)`). Const-capable layout/storage/code/CICP-projection primitives return bare `E`.
> - Need your buffer back on error? Use `from_bytes_recover`, `from_typed_recover`, `from_parts_recover`, or `from_external*` (`Err((err, original))`).

## Two crates

```toml
# Storage, views, color spaces, CICP, HDR metadata, planes (lightweight — use in codecs)
zenpixels = "0.5.0"

# Conversion, HDR tone mapping, CMS, gain maps, planar reconstruction, orientation (re-exports zenpixels::*)
zenpixels-convert = "0.5.0"
```

## What else is included?

See [docs/guide.md](docs/guide.md) for copy-pasteable recipes covering every area:

- **External zero-copy ownership, cropping & fallible tightening:** `PixelBuf::from_external` / `from_external_mut` (`Arc<[u8]>`, `Box<[T]>`, FFI), O(1) zero-copy `PixelBuf::into_crop`, symmetric `crop_view` / `crop_view_mut` / `sub_rows` / `sub_rows_mut` across carriers, fallible copy-on-write `try_as_view_mut` / `into_tight_buf`, `into_vec_zero_copy`, and explicit metadata relabeling (`relabel_color`, `relabel_format`, `reinterpret_bytes`).
- **Sound `AlphaMode::Opaque` & fast fill-then-retag:** `PixelBuf::new` (black + max alpha), `new_white`, and `new_zeroed` (refuses unverified `Opaque`) paired with `into_opaque_verified` / `into_opaque_filled` / `assume_opaque`.
- **Planar, banded & dual-layer HDR:** `PlanarBuf` / `PlanarView` / `PlaneBuf` / `PlaneView` (tight typed `from_samples(samples, w, h, spec)`, strided `from_samples_strided(..., SampleStride(n), spec)`, typed 1-byte `MatrixCoefficients`, `CodePlane` bit-packed inspection), `BandedBuf` for gigapixel images, and `DualLayerHdrBuf` for ISO 21496-1 / UltraHDR gain maps.
- **Lossless CICP & ICC.1 `cicpType`:** all H.273 codes plus 44-byte signed `s15Fixed16` custom primaries (`CicpTag`, `CustomPrimaries::ACES_AP0` / `ACES_CG` / `PROPHOTO_RGB`, Annex 1 D50 matrices) and ISO 15076-1 structural ICC validation (`validate_icc_structure`).
- **Orientation on every carrier:** O(1) `with_orientation` / `compose_orientation` / `display_dimensions()`.

## Conversion

```rust
use zenpixels_convert::{PixelFormat, PixelBuf, to_rgba8};

let src = PixelBuf::from_typed(vec![128u8, 128, 128], 1, 1, PixelFormat::RGB8_SRGB)?;
let rgba = to_rgba8(&src.as_view())?;
assert_eq!(rgba.row(0), &[128, 128, 128, 255]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For custom destinations, linear-light working spaces, HDR tone mapping, or ICC
profiles, use `convert_to_buf`, `ImagePlan`, `RowPlan`, or `OutputPlan` — see
[docs/guide.md](docs/guide.md) and [zenpixels-convert/README.md](zenpixels-convert/README.md).

## Features (`zenpixels`)

| Feature | Effect |
|---|---|
| `std` | Default; disable default features for `no_std + alloc` |
| `icc` | Default; paired-fingerprint ICC recognition with intent-scoped substitution |
| `rgb` | `PixelType` trait & typed `rgb` crate interop |
| `imgref` | `imgref` adapters; enables `rgb` |
| `image` | `image` 0.25 interop with CICP color-space propagation; requires Rust 1.88 |
| `hash` | Portable endian/stride-independent XXH3-64 pixel hashing |

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
