<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenpixels-convert

Convert pixel samples while keeping their description and output metadata in
agreement. This crate re-exports `zenpixels::*`, so most applications need just
this one dependency.

**Start with the guides:**

- **[Pixel types guide](https://imazen.github.io/zenpixels/guide.html)** — own buffers, borrow crops, and convert rows.
- **[Transfer functions & live Rust codegen](https://imazen.github.io/zenpixels/transfers.html)** — interactive EOTF/OETF & bit-allocation charts with live conversion snippets.
- **[Primaries, HDR headroom, CLL & CICP vs ICC](https://imazen.github.io/zenpixels/color-hdr.html)** — CIE `xy`/`u'v'` gamut renderings, HDR headroom/BT.2446A charts, signaling range, and CICP vs ICC.
- **[Conversion kernel browser](https://imazen.github.io/zenpixels/kernels.html)** — inspect all 181 live-extracted steps & SIMD kernels by `const Accelerators` preset.
- **[Balance speed and light compilation](https://imazen.github.io/zenpixels/speed-and-size.html)** — how `accel` and `const Accelerators` keep builds light while unlocking full SIMD.
- **[Plain-English ELI5 mental model](https://imazen.github.io/zenpixels/eli5.html)** — core concepts and real-world gotchas.
- **[Interactive byte explorer](https://imazen.github.io/zenpixels/)** — inspect bytes and run live converter plans in your browser.

## Install

These examples follow the **unpublished 0.4.0 candidate**. The
[published reference](https://docs.rs/zenpixels-convert) may describe an earlier API.

```toml
zenpixels-convert = "0.4.0"
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

## Convert rows, views, buffers, or bands with explicit options

Build the converter once. `prepare` reserves built-in scratch and initializes
selected tables before the loop. Rows may come from a crop or padded storage:

```rust
use zenpixels_convert::{ConversionSpec, ConvertOptions, ConvertPlan, GamutPolicy, PixelBuffer, PixelDescriptor};

let source = PixelBuffer::from_vec(
    vec![64, 128, 192, 255, 0, 0], 2, 1, PixelDescriptor::RGB8_SRGB,
)?;
let mut output = PixelBuffer::try_new(2, 1, PixelDescriptor::RGBF32_LINEAR)?;
let options = ConvertOptions::retain_channels().with_gamut_policy(GamutPolicy::PreserveExtended);
let plan = ConvertPlan::build(
    source.encoding(), output.encoding(), &ConversionSpec::new(options),
)?;
let mut converter = plan.prepare(source.width())?;

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
Use `try_clone()` for a fallible independent converter worker, or pass whole
views/buffers/bands directly to `converter.convert(&src, &mut dst)` (`PackedSource`
/ `PackedSink`: `&PixelSlice`, `&PixelBuffer`, `&BandedPixelBuffer` → `&mut PixelSliceMut`,
`&mut PixelBuffer`, `&mut BandedPixelBuffer`). Owned sinks receive updated
`ColorContext` and sample evidence (`ExactDepth`/`FloatFacts`); `&mut PixelSliceMut`
receives updated `ColorContext` only.

## Balance speed and light compilation (`accel` & `const Accelerators`)

By default, every native conversion runs through a compact, zero-scratch
**universal fast-FMA scalar core** that is bit-for-bit identical to the
x86-64/AArch64 SIMD kernels, so libraries and Wasm builds never pay for unused
SIMD tables. Opt into SIMD at whichever granularity fits your crate:

```rust
use zenpixels_convert::{Accelerators, ConversionSpec, ConvertOptions};

// 1. One-liner for apps, CLIs, servers, and benchmarks (process-wide):
zenpixels_convert::accel::enable_all();

// 2. Selective opt-in (unused SIMD kernels dead-strip to 0 B):
zenpixels_convert::accel::enable_web_sdr(); // or enable_layout/srgb/float/gamut/hdr/oklab

// 3. Per-spec const helper (zero global state; ignores global registry):
const SDR_AND_OKLAB: Accelerators = Accelerators::WEB_SDR.with_oklab();
let fast   = ConversionSpec::fast(ConvertOptions::retain_channels());
let scoped = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&SDR_AND_OKLAB);
let scalar = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&Accelerators::SCALAR); // hermetic parity testing
# let _ = (fast, scoped, scalar);
```

Top-level binaries can alternatively enable the default-off Cargo feature `fast`
(library crates should never enable `fast`). Attach measured or proven float
bounds via `ConversionSpec::with_float_facts(FloatFacts::measure(&slice)?)` and
inspect propagated guarantees with `plan.output_float_facts()`. Read the
**[Speed and light compilation guide](https://imazen.github.io/zenpixels/speed-and-size.html)**
for measured binary sizes, feature budgets, and workload recipes.

## Choose what may change

- Choose alpha, depth, luma and gamut policies on `ConvertOptions`.
  `retain_channels()` forbids alpha and depth loss and selects gamut clipping.
- `ConversionSpec::preserve_samples()` admits proven reversible representation
  changes. Provide source interpretation for unknown transfer or request a
  relabel when you know the actual interpretation.
- **Wide-gamut Oklab works directly:** `PixelDescriptor::OKLABF32` is
  CIE XYZ D65-anchored, so roundtripping Display P3 or BT.2020 through
  `OKLABF32` uses that gamut's own `RGB ↔ LMS` matrix directly — bit-exact
  with zero BT.709 clipping and no extra `.with_primaries(...)` call.
- `ConvertPlan::select` returns a candidate index and an executable plan.
  `ConvertPlan::then` preserves the samples and interpretation at a materialized
  boundary, including intermediate quantization.

## Keep encoder metadata with the result

`EncodeReady::from_owned` reuses the source allocation for identity output and
publishes matching color metadata. Use `copy_from` to create independent output
from borrowed pixels.

```rust
use zenpixels_convert::{
    Cicp, ConversionSpec, EncodeReady, OutputPlan, OutputProfile, OutputTarget,
    PixelBuffer, PixelDescriptor, PixelFormat,
};

let source = PixelBuffer::from_vec(vec![64, 128, 192], 1, 1, PixelDescriptor::RGB8_SRGB)?;
let plan = OutputPlan::build(
    source.encoding(),
    OutputTarget::new(OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgb16),
    &ConversionSpec::preserve_samples(),
)?;
let ready = EncodeReady::from_owned(source, plan)?;
let view = ready.pixels();
assert_eq!(view.descriptor().pixel_format(), PixelFormat::Rgb16);
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

Explicit HDR rules:

- **Import path:** `zenpixels_convert::hdr` re-exports all core HDR anchor
  types (`Nits`, `DiffuseWhite`, `HdrPeak`, `HlgRendering`,
  `ContentLightLevel`, `MasteringDisplay`, `St2086Codes`) from
  `zenpixels::hdr`, so apps only need `zenpixels-convert`.
- **Feature gate:** Built-in HDR→SDR tone mapping requires
  `features = ["bt2446a"]` (or an injected feature-gated `ToneMapper`); without
  it, explicit HDR→SDR plans fail with `ConvertError::HdrFeatureRequired`.
- **Plan vs. buffer peaks:** `ConvertPlan::build` / `RowConverter` see encodings
  only and require concrete `HdrPeak::Nits(...)` (`HdrPeak::Measured` /
  `Declared` resolve only on buffer-level entry points that see pixel data).
- **HLG→SDR:** `with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)` supplies
  the OOTF shape only; you must still set `with_sdr_white(...)`,
  `with_hdr_peak(HdrPeak::Nits(...))`, and `with_sdr_peak(...)`
  (`HlgRendering.peak()` is not reused as the content peak).
- Source metadata never silently selects a display.
  [HDR contract](https://github.com/imazen/zenpixels/blob/main/docs/hdr-anchor-api-design.md).

Native integer code extraction is separate from color conversion.
`codes::to_canonical` handles supported component encodings; generic signed/wide
storage does not grant numerical conversion support.

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `fast-transpose` | Default; runtime-dispatched SIMD orientation kernels |
| `oklab` | Default; Oklab conversions and soft gamut compression |
| `hdr-transfers` | Default; PQ/HLG transfer curves and HLG display rendering |
| `perceptual` | Default-off; closed-form `Oklch`, `Lch`, `Jzazbz`, `Jzczhz`, `Hsl`, `Hsv` (forwards to `zenpixels/perceptual`) |
| `fast` | Default-off (binaries only); defaults plans to `Accelerators::ALL` without calling `accel::enable_all()` |
| `precise-portable` | Default-off; single-rounding software FMA on non-fusing WASM engines |
| `icc-db` | Qualified CICP-to-ICC recipes with lazy group decoding |
| `css` | CSS Color 4 string parser (`css` module and `ColorConvertExt::parse_css`) |
| `planar` | Planar YCbCr conversion, N-bit code extraction, and plane orientation |
| `cms-moxcms`, `cms-lcms` | Explicitly selected CMS implementations |
| `rgb`, `imgref` | Core typed-pixel and imgref interop |
| `introspection` | Opt-in compiled step, accuracy and dispatch diagnostics |
| `avx512` | Opt-in wider row-kernel tiers |
| `hdr-experimental`, `bt2446a` | HDR measurement/rendering and BT.2446A tone mapping |

Converting across a compiled-out feature (`oklab`, `hdr-transfers`, or
`perceptual` when `zenpixels/perceptual` is unified on) returns
`ConvertError::FeatureRequired { feature, from, to }` instead of silently
dropping transforms (`ChannelLayout::Lab` returns `NeedsCms`). Without
`icc-db`, synthesis retains supported compact exports and explicit refusal
elsewhere; it does not restore the legacy P3 fallback.

## MSRV

Rust 1.89, edition 2024. `no_std + alloc`; unsafe code is forbidden.

## See also

[Pixel types guide](https://imazen.github.io/zenpixels/guide.html) ·
[Transfer functions](https://imazen.github.io/zenpixels/transfers.html) ·
[Primaries, HDR & CICP/ICC](https://imazen.github.io/zenpixels/color-hdr.html) ·
[Kernel browser](https://imazen.github.io/zenpixels/kernels.html) ·
[Speed & light builds](https://imazen.github.io/zenpixels/speed-and-size.html) ·
[Plain-English ELI5](https://imazen.github.io/zenpixels/eli5.html) ·
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
