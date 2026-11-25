# zenpixels-convert [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square)](https://crates.io/crates/zenpixels-convert) [![lib.rs](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels-convert) [![docs.rs](https://img.shields.io/docsrs/zenpixels-convert?style=flat-square)](https://docs.rs/zenpixels-convert) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenpixels-convert?style=flat-square)](#license)

Transfer-function-aware pixel conversion, HDR tone mapping, gamut compression,
ICC/CMS routing, gain-map reconstruction, EXIF orientation, planar YCbCr/RGB
reconstruction, and codec format negotiation for [`zenpixels`](https://github.com/imazen/zenpixels).
Re-exports `zenpixels::*` at the crate root with zero module name collisions.

This is the **0.5.0 greenfield candidate**. For full end-to-end recipes, see
[docs/guide.md](../docs/guide.md) and [docs/migration-0.2-to-0.5.md](../docs/migration-0.2-to-0.5.md).

## Which API should I pick?

| Need | API |
|---|---|
| Quick 8-bit sRGB output | `to_rgb8(&view)`, `to_gray8(&view)` (refuse unproven alpha discard), `to_rgba8(&view)`, `to_bgra8(&view)` |
| One-shot image conversion | `convert_to_buf(&view, dst_spec, &policy, PeakSource::Measured)` / `convert_into(&src, &mut dst, &policy, peak)` |
| Reusable whole-image plan (`PixelBuf` / `PixelView` / `BandedBuf`) | `ImagePlan::build(src, dst, &policy, peak)` → `.execute(&src)` / `.execute_into(&src, &mut dst)` |
| Streaming row loop (zero per-row heap scratch) | `RowPlan::build(src, dst, &policy, resolved_source_peak)` → `.convert_row(src_row, dst_row, width, y)` |
| Encoder negotiation + CICP/ICC signaling | `adapt_for_encode_format(view, origin, &[PixelFormat], &policy, peak, profile)` or reusable `OutputPlan::build(...)` |

## Quick start

```rust
use zenpixels_convert::{
    ConvertPolicy, ImageSpec, PeakSource, PixelBuf, convert_to_buf, to_rgba8,
};

let src = PixelBuf::from_typed(
    vec![255u8, 0, 0, 0, 255, 0], 2, 1, ImageSpec::RGB8_SRGB,
)?;

// 1. One-call 8-bit sRGB conversion (no policy/HDR setup needed):
let rgba = to_rgba8(&src.as_view())?;
assert_eq!(rgba.as_view().row(0), &[255, 0, 0, 255, 0, 255, 0, 255]);

// 2. Arbitrary destination with explicit policy (safe default = lossless):
let linear = convert_to_buf(
    &src.as_view(),
    ImageSpec::rgbf32_linear_ref(),
    &ConvertPolicy::working_linear(),
    PeakSource::Measured,
)?;
assert_eq!(linear.spec(), ImageSpec::rgbf32_linear_ref());
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Safe-by-default `ConvertPolicy`

`ConvertPolicy::default()` equals **`ConvertPolicy::lossless()`** — it refuses
depth narrowing, alpha removal, and range scaling, and preserves extended float
gamut. Opt in explicitly to `ConvertPolicy::permissive()`,
`ConvertPolicy::working_linear()`, or `ConvertPolicy::display_sdr(white, peak)`
when lossy conversion is desired. All planners (`RowPlan::build`,
`ImagePlan::build`, `OutputPlan::build`) borrow `&ConvertPolicy` and accept
either `&ImageSpec` or `ImageSpec`.

## Streaming rows (`RowPlan`)

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PixelBuf, RowPlan};

let src = PixelBuf::from_typed(vec![64u8, 128, 192, 255, 0, 0], 2, 1, ImageSpec::RGB8_SRGB)?;
let mut dst = PixelBuf::new(2, 1, ImageSpec::RGBF32_LINEAR)?;
let plan = RowPlan::build(src.spec(), dst.spec(), &ConvertPolicy::permissive(), None)?;
for y in 0..src.height() {
    plan.convert_row(src.as_view().row(y), dst.as_view_mut().row_mut(y), src.width(), y)?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Opt-in SIMD (`accel` & `const Accelerators`)

By default, every native pair runs on a compact scalar core with 0-bit parity
against x86-64/AArch64 SIMD kernels so libraries and Wasm builds stay small.
Enable SIMD globally (`accel::enable_all()`, `accel::enable_web_sdr()`) or
per-policy via `ConvertPolicy::with_accelerators(Accelerators::ALL)`.

## Key capabilities (see [docs/guide.md](../docs/guide.md) for recipes)

- **Orientation & `DualLayerHdrBuf`:** `compose_orientation` updates metadata in
  O(1); `apply_orientation*` physically applies an explicit transform (emitting
  `Identity`); `bake_orientation*` / `bake_dual_layer_orientation*` bake pending
  orientation (zero-copy when already `Identity`). Pass
  `&zenpixels_convert::Unstoppable` (re-exported from `enough`) to gain-map APIs
  accepting `&dyn Stop`, and configure headroom with
  `GainMapApplyConfig::from_headroom_stops` or `from_display_boost`.
- **Planar YCbCr & planar RGB:** `ycbcr_to_rgb_with_options` and zero-alloc
  `ycbcr_to_rgb_with_options_into` reconstruct all 9 H.273 NCL matrices and
  planar RGB/RGBA directly into `U8`/`U16`/`F32` interleaved layouts with
  siting-exact `YcbcrOptions::bilinear()`.
- **HDR & ICC/CMS:** `Bt2446A::pipeline_managed()` tone mapping with streaming
  `PeakSource::Measured`, plus pure-Rust `MoxCms` and `Lcms` CMS backends
  (`with_icc_substitution(false)` forces CMS execution).
- **Load-bearing compaction:** `determine_load_bearing()`,
  `reduce_to_load_bearing_format()`, and zero-reallocation
  `reduce_to_load_bearing_format_in_place(AlphaCompaction::RetagOpaque | DropUnusedLanes)`.
- **Encoder signaling (`OutputPlan` & `EncodeReady`):** validates destination
  signaling eagerly (`OutputProfile::{Auto, SameAsOrigin, CicpOnly, EmbedIcc, CustomIcc, Strip}`),
  refuses `HlgMode::Display` on `dst` under `CicpOnly`/`Auto`, and stays
  zero-copy on identity.

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `fast-transpose` | Default; runtime-dispatched SIMD orientation kernels |
| `oklab` | Default; CIE XYZ D65 Oklab conversions and OKLch soft gamut compression |
| `hdr-transfers` | Default; PQ (ST 2084) and HLG (ARIB STD-B67 + BT.2100-3 display OOTF) |
| `gainmap` | Default; ISO 21496-1 / UltraHDR gain-map extraction & reconstruction |
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
