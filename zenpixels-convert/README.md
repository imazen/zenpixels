# zenpixels-convert [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenpixels/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenpixels/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square)](https://crates.io/crates/zenpixels-convert) [![lib.rs](https://img.shields.io/crates/v/zenpixels-convert?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpixels-convert) [![docs.rs](https://img.shields.io/docsrs/zenpixels-convert?style=flat-square)](https://docs.rs/zenpixels-convert) [![license](https://img.shields.io/crates/l/zenpixels-convert?style=flat-square)](https://github.com/imazen/zenpixels#license)

Transfer-function-aware pixel conversion, gamut mapping, and codec format negotiation for Rust image pipelines.

A JPEG decoder gives you `RGB8` in sRGB. An AVIF decoder gives you `RGBA16` in BT.2020 PQ. A resize library wants `RGBF32` in linear light. Without shared types, every codec pair needs hand-rolled conversion — and gets transfer functions wrong, silently drops alpha, or writes "sRGB" in the ICC profile while the pixels are linear.

`zenpixels-convert` is the math half of the [`zenpixels`](https://crates.io/crates/zenpixels) ecosystem. `zenpixels` carries the format/color types ([`PixelBuffer`], [`PixelDescriptor`], [`PixelFormat`], `TransferFunction`, `ColorPrimaries`, `ColorContext`); this crate handles transfer functions, gamut matrices, depth scaling, alpha compositing, format negotiation, and HDR tone mapping so codecs don't have to. It **re-exports everything from `zenpixels`** at its crate root, so downstream pipelines can depend on `zenpixels-convert` alone.

`no_std + alloc`, `#![forbid(unsafe_code)]`, no system dependencies. SIMD where available, runtime-dispatched.

## Install

```toml
[dependencies]
zenpixels-convert = "0.2.15"
```

Common feature combinations:

```toml
# Plus the rgb crate's typed buffers and convenience methods
zenpixels-convert = { version = "0.2.15", features = ["rgb"] }

# Plus a pluggable CMS backend (moxcms) for arbitrary ICC profiles
zenpixels-convert = { version = "0.2.15", features = ["cms-moxcms"] }

# Plus the experimental HDR→SDR display-mapping surface
zenpixels-convert = { version = "0.2.15", features = ["hdr-experimental"] }

# Size-sensitive / wasm builds (drops the bundled CICP->ICC database)
zenpixels-convert = { version = "0.2.15", default-features = false, features = ["std"] }
```

## Quick start

### Negotiate a format, then convert row by row

[`best_match`] picks the cheapest target format an encoder supports for a given source descriptor; [`RowConverter`] pre-computes the conversion plan once and converts rows with no per-row allocation.

```rust
use zenpixels_convert::{RowConverter, best_match, ConvertIntent};

// Pick the cheapest target format the encoder supports.
let target = best_match(source_desc, &encoder_formats, ConvertIntent::Fastest)
    .ok_or("no compatible format")?;

// Pre-compute the plan, then convert row by row — no per-row allocation.
let mut converter = RowConverter::new(source_desc, target)?;
for y in 0..height {
    converter.convert_row(src_row, dst_row, width);
}
```

### Color profile conversion (no CMS needed for named profiles)

Named-profile pairs (sRGB ↔ Display P3 ↔ BT.2020 ↔ Adobe RGB) use hardcoded matrices with fused SIMD kernels — LUT-decode + SIMD matrix + SIMD polynomial encode for u16, fused matlut for u8. No CMS backend required.

```rust
use zenpixels::{PixelDescriptor, ColorPrimaries};
use zenpixels_convert::{RowConverter, ConvertOptions};

let p3   = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
let srgb = PixelDescriptor::RGB8_SRGB;

let mut conv = RowConverter::new_explicit(
    p3, srgb, &ConvertOptions::permissive(),
)?;
conv.convert_row(p3_row, srgb_row, width);
```

Custom ICC profiles (vendor-specific, LUT-based, perceptual intent) need a CMS backend. Pass one via [`PluggableCms`] — the plan delegates to the plugin when the profiles differ, and uses the built-in matlut fast path when both sides are well-known. The `cms-moxcms` feature provides a concrete backend ([`MoxCms`]) built on [moxcms](https://crates.io/crates/moxcms).

```rust
use zenpixels_convert::{RowConverter, ConvertOptions, MoxCms, cms::PluggableCms};

let cms: &dyn PluggableCms = &MoxCms;
let mut conv = RowConverter::new_explicit_with_cms(
    source_desc, target_desc,
    &ConvertOptions::permissive(),
    Some(cms),
)?;
```

For HDR and wide-gamut pipelines that defer tone/gamut mapping, build `ConvertOptions` with gamut clipping disabled so f32 sRGB transfers preserve negative and supernormal values through the conversion.

## No silent lossy conversions

Every operation that destroys information requires an explicit policy via [`ConvertOptions`] (from `zenpixels`, re-exported here):

- **Alpha removal**: `DiscardIfOpaque`, `CompositeOnto { r, g, b }`, `DiscardUnchecked`, or `Forbid`.
- **Depth reduction**: `Round`, `Truncate`, or `Forbid`.
- **RGB → gray**: requires explicit luma coefficients (`Bt709`, `Bt601`, `Bt2020`, or `DisplayP3`), or `None` to forbid. Uses Y' (encoded luma) semantics — round-trips bit-exactly when `R == G == B`.

Convenience constructors: `ConvertOptions::forbid_lossy()` (safe default) and `ConvertOptions::permissive()` (sensible lossy defaults), plus `with_alpha_policy()` / `with_depth_policy()` for customization.

## Format negotiation

The cost model separates **effort** (CPU work) from **loss** (information destroyed). [`ConvertIntent`] controls the weighting:

| Intent | Effort | Loss | Use case |
|---|---|---|---|
| `Fastest` | 4× | 1× | Encoding — get there fast |
| `LinearLight` | 1× | 4× | Resize, blur — need linear math |
| `Blend` | 1× | 4× | Compositing — premultiplied alpha |
| `Perceptual` | 1× | 3× | Color grading, sharpening |

[`Provenance`] tracking lets the cost model know that f32 data decoded from a u8 JPEG has zero loss converting back to u8. Three entry points: [`best_match`] (simple), [`best_match_with`] (with consumer costs), [`negotiate`] (full control with provenance).

## Row conversion tiers

[`RowConverter`] pre-computes a conversion plan from a source/target descriptor pair, choosing one of three tiers:

1. **Direct kernels** for common pairs (byte swizzle, depth shift, transfer-function LUTs).
2. **Composed plans** for less common pairs (e.g. `RGB8_SRGB` → `RGBA16_LINEAR`).
3. **Hub path** through linear sRGB f32 as the universal fallback.

## Gamut, HDR, Oklab

- **Gamut matrices** — 3×3 row-major f32 between BT.709, Display P3, BT.2020 via [`conversion_matrix`] / [`GamutMatrix`] and `apply_matrix_*` row kernels. No CMS needed for named-profile conversions.
- **HDR (`hdr-experimental` feature)** — production HDR→SDR is a native step inside [`ConvertPlan`]: build via [`ConvertPlan::new_with_hdr_peak(from, to, source_peak_nits)`](https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html#method.new_with_hdr_peak) or [`ConvertPlan::new_with_hdr_config(from, to, HdrConfig)`](https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html#method.new_with_hdr_config), then run through the same [`RowConverter`]. The pipeline inserts the ITU-R BT.2446 Method A curve ([`Bt2446A`]) + an OKLch [`SoftCompress`] knee — confirmed best in the 2026-06-22 audited shootout (2-5× lower mean ΔE2000 than channel-independent curves). Source-peak measurement uses [`CllMeasure::measure_max`] (the spec-conformant CTA-861.3 reading, ~2.7 Gpix/s on RGB f32 on a Ryzen 9 7950X / AVX2 — the SOTA spec-conformant CLL reading in the workspace). Plain `ConvertPlan::new` now **refuses** HDR-encoded → SDR-encoded conversions with [`ConvertError::HdrSourceRequiresPeak`] — no more silent saturating routes. Other HDR primitives: [`quantize_to`] (anchor-aware linear→PQ16, reads `DiffuseWhite` from the source's `ColorContext`, BT.2408 default of 203), [`ContentLightLevel`] and [`MasteringDisplay`] metadata. The legacy global Reinhard helpers (`reinhard_tonemap` / `reinhard_inverse` / `exposure_tonemap`) are `#[deprecated]` + `#[doc(hidden)]` since 0.2.15 and queued for removal in 0.3.0.
- **Oklab** — primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`, scalar `rgb_to_oklab()` / `oklab_to_rgb()`. Non-sRGB sources get correct LMS matrices without an intermediate sRGB step.

## CICP / ICC

- **ICC identification** — `zenpixels::icc::identify_common(icc_bytes)` recognizes hundreds of well-known RGB + grayscale profiles via a normalized hash lookup (~100ns), returning primaries, transfer function, and whether matrix+TRC substitution is safe vs CMS-only. `extract_cicp` reads embedded `cICP` tags directly.
- **CICP → ICC synthesis** — `icc_profiles::synthesize_icc_for_cicp(Cicp)` resolves an embeddable ICC profile for any assigned H.273 combination from the bundled compressed database (`icc-db` feature, on by default; ~36 KB asset, lazily decoded, no CMS required, PQ/HLG HDR included). `synthesize_gray_icc_for_cicp` is the GRAY-class sibling for single-channel output. The sRGB default answers `NotNeeded`. With `icc-db` off, these answer `NeedsCms` outside the sRGB default and the bundled consts.

## Orientation

The buffer-baking half of the zen orientation story lives here (the `Orientation` enum and its composition algebra are in `zenpixels`):

- [`orient::apply_orientation`] — fresh buffer; SIMD-tiled transpose for 4-byte pixels (behind the `fast-transpose` feature; portable scalar otherwise).
- [`orient::apply_orientation_into`] — caller-provided target, no allocation.
- [`orient::apply_orientation_in_place`] — reuses the buffer's own allocation.

## Atomic output assembly

[`finalize_for_output`] couples converted pixels with matching encoder metadata in one step, preventing the bug where pixel values don't match the embedded ICC/CICP profile. Returns an [`EncodeReady`] bundle.

## Pipeline planner (`pipeline` feature)

[`CodecFormats`] declares each codec's decode outputs and encode inputs, ICC/CICP support, effective bits, and overshoot behavior. The `pipeline` feature enables the format registry, operation requirements, and a path solver ([`optimal_path`], [`generate_path_matrix`]) for multi-step conversion planning.

## Load-bearing analysis

[`PixelSliceLoadBearingExt`] / [`PixelBufferLoadBearingExt`] run SIMD predicates over a buffer to report which channels actually carry information (e.g. is the alpha fully opaque, is the image grayscale), returning a [`LoadBearingReport`] used to drive lossless format-reduction decisions.

## Key types and functions

| Item | What it does |
|---|---|
| [`RowConverter`] | Pre-computed per-row conversion plan (`new`, `new_explicit`, `new_explicit_with_cms`, `convert_row`, `convert_rows`) |
| [`best_match`] / [`best_match_with`] / [`negotiate`] | Format negotiation entry points |
| [`ConvertIntent`] | Effort-vs-loss weighting for negotiation |
| [`Provenance`] / [`ConversionCost`] / [`FormatOption`] | Cost-model inputs and results |
| `ConvertOptions` (re-export) | Explicit lossy-operation policy |
| [`ConvertPlan`] / [`convert_row`] | Lower-level plan + stateless row convert |
| [`adapt::adapt_for_encode`] / [`adapt_for_encode_explicit`] / [`try_adapt_in_place`] | One-call negotiate-and-convert helpers |
| [`PluggableCms`] / [`RowTransformMut`] / [`MoxCms`] | Pluggable CMS backend interface + moxcms impl |
| [`GamutMatrix`] / [`conversion_matrix`] / `apply_matrix_*` | Gamut matrix construction and application |
| [`ConvertPlan::new_with_hdr_peak`] / [`ConvertPlan::new_with_hdr_config`] | HDR→SDR conversion plan constructors (`hdr-experimental` feature) |
| [`Bt2446A`] / [`SoftCompress`] / [`CllMeasure`] | ITU-R BT.2446 Method A curve + OKLch knee + CTA-861.3 CLL measurement (`hdr-experimental` feature) |
| [`quantize_to`] | Anchor-aware linear→PQ16 quantizer |
| [`finalize_for_output`] / [`EncodeReady`] / [`OutputMetadata`] | Atomic pixels-plus-metadata output assembly |
| [`TransferFunctionExt`] / [`ColorPrimariesExt`] / [`PixelBufferConvertExt`] | Conversion methods bolted onto interchange types |
| [`LoadBearingReport`] / [`PixelSliceLoadBearingExt`] | Channel-information analysis |
| [`ConvertError`] | Conversion error type |

## Features

| Feature | Default | What it enables |
|---|---|---|
| `std` | yes | Standard library |
| `icc-db` | yes | Bundled CICP→ICC profile database (~36 KB asset) behind `synthesize_icc_for_cicp` / `synthesize_gray_icc_for_cicp`; disable for size-sensitive builds (they answer `NeedsCms` instead) |
| `fast-transpose` | | SIMD transpose kernels (x86-64 AVX2 + aarch64 NEON) for `apply_orientation*`; portable scalar otherwise |
| `avx512` | | 16-wide AVX-512F f16 conversion kernels (runtime-dispatched) |
| `rgb` | | `Pixel` impls for the [`rgb`](https://crates.io/crates/rgb) crate's types + typed convenience methods (`to_rgb8()`, `to_rgba8()`, …) |
| `imgref` | | `ImgRef` / `ImgVec` conversions (implies `rgb`) |
| `planar` | | Multi-plane image types (YCbCr, Oklab planes, gain maps) |
| `pipeline` | | Pipeline planner: format registry, operation requirements, path solver |
| `hdr-experimental` | | HDR→SDR display mapping (ITU-R BT.2446 Method A curve + OKLch soft compress + CTA-861.3 CLL measurement). API shape may move ahead of 0.3.0 (in particular `measure_robust` → `measure` rename); scan kernels and accuracy contracts are stable. See [HDR section](#gamut-hdr-oklab) above. |
| `cms-moxcms` | | ICC profile transforms via [moxcms](https://crates.io/crates/moxcms) (implies `std`) |
| `serde` | | Forwards to `zenpixels/serde` (`hdr-experimental` types implement `Serialize`/`Deserialize` when both features are on) |

### `no_std`

`zenpixels-convert` is `no_std + alloc` by default-construction (`std` is on by default but everything works without it). For `no_std`, set `default-features = false` and re-enable only what you need (note that `cms-moxcms` requires `std`):

```toml
zenpixels-convert = { version = "0.2.15", default-features = false }
```

## MSRV

Rust 1.89+ (for the safe SIMD intrinsics it uses), 2024 edition. The companion `zenpixels` crate requires 1.85+.

## See also

- [`zenpixels`](https://crates.io/crates/zenpixels) — the pixel format/color type system this crate operates on.
- [`linear-srgb`](https://crates.io/crates/linear-srgb) — the linear↔sRGB transfer math used internally.

## License

Apache-2.0 OR MIT.

[`PixelBuffer`]: https://docs.rs/zenpixels/latest/zenpixels/struct.PixelBuffer.html
[`PixelDescriptor`]: https://docs.rs/zenpixels/latest/zenpixels/struct.PixelDescriptor.html
[`PixelFormat`]: https://docs.rs/zenpixels/latest/zenpixels/enum.PixelFormat.html
[`RowConverter`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.RowConverter.html
[`best_match`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.best_match.html
[`best_match_with`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.best_match_with.html
[`negotiate`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.negotiate.html
[`ConvertIntent`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/enum.ConvertIntent.html
[`Provenance`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/enum.Provenance.html
[`ConversionCost`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConversionCost.html
[`FormatOption`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.FormatOption.html
[`ConvertOptions`]: https://docs.rs/zenpixels/latest/zenpixels/struct.ConvertOptions.html
[`ConvertPlan`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html
[`convert_row`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.convert_row.html
[`adapt::adapt_for_encode`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/adapt/fn.adapt_for_encode.html
[`adapt_for_encode_explicit`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.adapt_for_encode_explicit.html
[`try_adapt_in_place`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.try_adapt_in_place.html
[`PluggableCms`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/cms/trait.PluggableCms.html
[`RowTransformMut`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/cms/trait.RowTransformMut.html
[`MoxCms`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.MoxCms.html
[`GamutMatrix`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/gamut/type.GamutMatrix.html
[`conversion_matrix`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/gamut/fn.conversion_matrix.html
[`quantize_to`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/fn.quantize_to.html
[`Bt2446A`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/struct.Bt2446A.html
[`SoftCompress`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/struct.SoftCompress.html
[`CllMeasure`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/trait.CllMeasure.html
[`CllMeasure::measure_max`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/trait.CllMeasure.html#tymethod.measure_max
[`ConvertPlan`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html
[`ConvertPlan::new_with_hdr_peak`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html#method.new_with_hdr_peak
[`ConvertPlan::new_with_hdr_config`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.ConvertPlan.html#method.new_with_hdr_config
[`ConvertError::HdrSourceRequiresPeak`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/enum.ConvertError.html#variant.HdrSourceRequiresPeak
[`ContentLightLevel`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/struct.ContentLightLevel.html
[`MasteringDisplay`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/struct.MasteringDisplay.html
[`finalize_for_output`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/fn.finalize_for_output.html
[`EncodeReady`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/output/struct.EncodeReady.html
[`OutputMetadata`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/output/struct.OutputMetadata.html
[`TransferFunctionExt`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/trait.TransferFunctionExt.html
[`ColorPrimariesExt`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/trait.ColorPrimariesExt.html
[`PixelBufferConvertExt`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/trait.PixelBufferConvertExt.html
[`LoadBearingReport`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/struct.LoadBearingReport.html
[`PixelSliceLoadBearingExt`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/trait.PixelSliceLoadBearingExt.html
[`PixelBufferLoadBearingExt`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/trait.PixelBufferLoadBearingExt.html
[`orient::apply_orientation`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/orient/fn.apply_orientation.html
[`orient::apply_orientation_into`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/orient/fn.apply_orientation_into.html
[`orient::apply_orientation_in_place`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/orient/fn.apply_orientation_in_place.html
[`CodecFormats`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/pipeline/struct.CodecFormats.html
[`optimal_path`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/pipeline/fn.optimal_path.html
[`generate_path_matrix`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/pipeline/fn.generate_path_matrix.html
[`ConvertError`]: https://docs.rs/zenpixels-convert/latest/zenpixels_convert/enum.ConvertError.html
