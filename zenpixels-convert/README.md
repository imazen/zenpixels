# zenpixels-convert [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square)](https://crates.io/crates/zenpixels-convert) [![lib.rs](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels-convert) [![docs.rs](https://img.shields.io/docsrs/zenpixels-convert?style=flat-square)](https://docs.rs/zenpixels-convert) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenpixels-convert?style=flat-square)](#license)

Transfer-function-aware pixel conversion, HDR tone mapping, gamut compression,
ICC/CMS routing, gain-map reconstruction, EXIF orientation, and codec format
negotiation for [`zenpixels`](https://github.com/imazen/zenpixels). Re-exports
`zenpixels::*` with zero module name collisions.

This is the **0.5.0 greenfield candidate**.

## Quick start

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PeakSource, PixelBuf, convert_buf};

let src = PixelBuf::from_vec(
    vec![255, 0, 0, 0, 255, 0], 2, 1, ImageSpec::rgb8_srgb(),
)?;
let linear = convert_buf(
    &src.as_view(), ImageSpec::rgbf32_linear(), &ConvertPolicy::permissive(), PeakSource::Measured,
)?;
assert_eq!(linear.spec(), &ImageSpec::rgbf32_linear());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Row-streaming (`RowPlan`) vs image-level (`ImagePlan`)

`RowPlan::build` runs rows through a stack-allocated 64-pixel SoA L1 strip with
zero per-row heap allocation (and statically accepts only `Option<Nits>` for HDR
peaks). `ImagePlan::build` accepts `PeakSource` (including streaming
`PeakSource::Measured`) and runs across `PixelView`, `PixelBuf`, and `BandedBuf`.

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PixelBuf, RowPlan};

let src = PixelBuf::from_vec(vec![64, 128, 192, 255, 0, 0], 2, 1, ImageSpec::RGB8_SRGB)?;
let mut dst = PixelBuf::new(2, 1, ImageSpec::RGBF32_LINEAR)?;
let plan = RowPlan::build(
    src.spec(), dst.spec(), &ConvertPolicy::permissive(), None,
)?;
for y in 0..src.height() {
    plan.convert_row(src.as_view().row(y), dst.as_view_mut().row_mut(y), src.width(), y)?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Opt-in SIMD (`accel` & `const Accelerators`)

By default, every native pair executes on a compact universal scalar core that
maintains 0-bit parity with x86-64/AArch64 SIMD kernels so libraries and Wasm
builds stay light. Enable SIMD globally (`accel::enable_all()`,
`accel::enable_web_sdr()`) or per-policy via `ConvertPolicy::with_accelerators(Accelerators::ALL)`.

## Orientation, planar YCbCr & encoder signaling

- **Orientation:** `ImagePlan` and `load_bearing` propagate pending buffer
  `Orientation` metadata; call `bake_orientation` / `bake_orientation_buf`
  (zero-copy on `Identity`) / `bake_orientation_banded` when physically rotating
  pixels, or `rotate_metadata` for O(1) EXIF passthrough.
- **Zero-copy planar YCbCr / RGB reconstruction:** `ycbcr_to_rgb_with_options`
  and zero-alloc `ycbcr_to_rgb_with_options_into` handle all 9 H.273 Table 4 NCL
  matrices (including `YCgCo` and `ChromaDerivedNcl`), siting-exact
  `ChromaUpsample::Bilinear` (9:3:3:1 `Center`, `Left`, `TopLeft`, odd crop
  origins), planar RGB (`U8`/`U16`/`F32`), and direct `U8`/`U16`/`F32`
  `RGB`/`RGBA`/`BGR`/`BGRA`/`RGBX`/`BGRX`/`Gray`/`GrayAlpha` outputs.
- **Encoder signaling (`OutputPlan` & `EncodeReady`):** `OutputPlan::build`
  accepts borrowed `&ImageSpec` or owned `ImageSpec`, validates output signaling
  (`Auto`, `CicpOnly`, `EmbedIcc`, `SameAsOrigin`, `Named(Cicp)`, `CustomIcc`)
  eagerly (refusing `HlgMode::Display` on `CicpOnly`/`Auto`), reuses owned
  storage on identity via `EncodeReady::from_owned`, and strips stale HDR
  metadata when photometry changes.

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `fast-transpose` | Default; runtime-dispatched SIMD orientation kernels |
| `oklab` | Default; CIE XYZ D65 Oklab conversions and OKLch soft gamut compression |
| `hdr-transfers` | Default; PQ (ST 2084) and HLG (ARIB STD-B67 + BT.2100-3 display OOTF) |
| `perceptual` | Default-off; Oklch, CIE Lab/Lch, Jzazbz/Jzczhz, HSL, HSV |
| `bt2446a` | Built-in ITU-R BT.2446 Method A HDR→SDR tone mapper |
| `icc-db` | Committed LZ4-compressed CICP→ICC profile bundle |
| `cms-moxcms`, `cms-lcms` | Pure-Rust `MoxCms` and `Lcms` ICC CMS backends |
| `fast` | Default-off (binaries only); defaults plans to `Accelerators::ALL` |
| `avx512` | Opt-in AVX-512F F16C & vector tiers |
| `precise-portable` | Portable bit-exact FMA on strict WASM engines |
| `rgb`, `imgref` | Core `PixelType` and `imgref` interop |

## MSRV

Rust 1.89, edition 2024. `no_std + alloc`, `#![forbid(unsafe_code)]`.

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
