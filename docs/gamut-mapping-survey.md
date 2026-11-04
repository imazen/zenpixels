# Perceptual gamut mapping: what it means and what to do

Surveyed 2026-11-04 at `4322c4dd`, from source. The harness and CSV results are in [`benchmarks/gamut-survey`](../benchmarks/gamut-survey). No images were viewed, so this doesn't cover preference or visible banding. It also doesn't cover real JPEG decode, BPC conformance, or arbitrary ICC pairs. Timings: Ryzen 9 7900X (Zen 4), 1024-pixel rows, not measured on Zen 5 or other CPUs.

## "Perceptual" means three different things

- **ICC perceptual intent** runs tables a profile author wrote: A2B0 and B2A0. Matrix profiles, including ordinary P3 and BT.2020, have none, so the flag adds no compression. moxcms' matrix path never receives the intent. All four intents gave exactly equal output on 4,913 colors from each gamut, in moxcms 0.9.1 and our fork's 0.9.0.
- **Profile-free compression** is a chosen formula, such as [Ottosson's adaptive clipping](https://bottosson.github.io/posts/gamutclipping/), [ACES Reference Gamut Compression](https://github.com/aces-aswf/aces-core/blob/1256fee50ee35548c6eab8eca854ff3349008489/transforms/ctl/lmt/LMT.Academy.GamutCompress.ctl) or our `SoftCompress`. It runs per pixel and works on images. [BT.2407](https://www.itu.int/pub/R-REP-BT.2407-2017) is a survey of six methods, not one method.
- **[CSS gamut mapping](https://www.w3.org/TR/css-color-4/#gamut-mapping)** searches for one color. It keeps in-gamut colors, reduces OkLCh chroma, then accepts a final clip within a small ΔEOK.

## What other implementations do

| Implementation | What it does for a matrix P3 image to sRGB |
|---|---|
| moxcms 0.9.1 | Matrix and curves, then bounds. A profile with tables but not the requested one can return an error. No BPC option; absolute uses the same tables as relative, with no media-white step |
| Little CMS | Matrix and curves, then bounds. A missing table falls back to table 0, then the matrix. Has media-white handling for absolute and optional BPC, neither of which compresses chroma |
| qcms, skcms | Curves and matrix, or the tables present. qcms ignores its intent argument; skcms picks tables by parser priority |
| palette, kolor, Linebender `color`, `oklab` | Numerical conversion, then clamp or out-of-range values; Oklab is a coordinate system, not a policy |
| culori, Color.js, culors, farg | Single-color searches, in the CSS style or similar |
| gainforge, zentone | Tone mapping, and filmlike or RGB soft clips that don't hold Oklab hue |

## Measurements

Linear source to encoded sRGB. Hue and ΔEOK use CSS's f64 Oklab, before `u8` rounding.

| Method | P3 green hue | BT.2020 blue hue | BT.2020 blue ΔEOK | P3 in-gamut colors changed | P3 / BT.2020 saturation-ramp reversals | ns/pixel P3 / BT.2020 |
|---|---:|---:|---:|---:|---:|---:|
| Clip (`RowConverter`) | -3.15° | +18.99° | 0.137 | 0 of 2,855 | 1,779 / 2,270 | 3.13 / 3.11 |
| CSS search (f64 scalar) | -2.32° | +6.43° | 0.257 | 0 | 5,596 / 24,039 | 201 / 516 |
| Ottosson adaptive (f64 scalar) | 0.00° | 0.00° | 0.275 | 0 | 16,544 / 28,181 | 66 / 87 |
| `SoftCompress` 0.96 | -0.01° | -0.01° | 0.276 | 456, by up to 0.024 ΔEOK | 0 / 1 | 65 / 66 |
| moxcms, any intent | -3.15° | +18.99° | 0.137 | Numerical only: worst 1.3e-4 | 1,840 / 2,563 | 2.8 / 3.0 |

Rows differ in precision and dispatch. Ramps are 288 per gamut at 1024 samples, counting chroma decreases over 1e-7. That's a diagnostic, not a visibility measure. `SoftCompress` also left 32 of 4,913 P3 and 50 of 4,913 BT.2020 colors out of range, by up to 0.0025, so it needs a final clamp. Standalone probes added 9,225 `.text` bytes for CSS, 5,897 for adaptive and 9,359 for `SoftCompress`.

A generated 33³ CLUT carries CSS mapping through unmodified moxcms at 7.6 ns/pixel, in a 216 KB profile per pair. Its worst off-grid distance from direct CSS is 0.024 ΔEOK for P3 and 0.042 for BT.2020, so it's a demonstration rather than a solution. It also needs a backend that requests Perceptual.

The CSS search here agrees with Color.js 0.7.1 within 1.55e-14 on 2,048 samples, and the adaptive clip with Ottosson's C++ within 1.94e-6.

## Where each belongs

- **`Color`:** a private CSS search once a CSS parser or similar single-color output exists. zenpipe's parser takes named and hex colors today, so nothing calls for it yet.
- **Images:** an explicit output choice reusing `SoftCompress`, after the matrix and before the target transfer, with a final clamp. zenfilters already calls it (`GamutMapping::SoftCompress`) and the HDR plan emits it. No SDR export path asks for it yet.
- **CMS:** ICC tables, media white and BPC stay there. BPC doesn't address gamut, and adding it or absolute media-white handling means changing moxcms' connection stage, not setting a flag. Making `MoxCms` request other intents is a small adapter change, and matters only for profiles with tables.
