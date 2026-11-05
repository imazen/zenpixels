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

## A faster soft compressor, 2026-11-04

`benchmarks/gamut-compress` tests a two-step mapper for images, against the boundary-table `SoftCompress` it replaced, with the same 0.96 knee. Results are on the Ryzen 9 7900X (Zen 4) with AVX2 dispatch; not measured on Zen 5.

1. **Gate in linear RGB.** A pixel with `min > 0.0875 * max` and `max < 0.95` can't be past the knee, so it stays bit-identical with no color math. A 257³ sRGB grid had 12,020,103 passing colors and none the exact mapper would change. These thresholds are for sRGB output; other primaries need their own.
2. **Solve the boundary directly.** Scaling OKLab chroma by `s` at constant L and hue is a lerp in cube-root LMS, `x'(s) = L + s (x' - L)`, because the a and b rows of M2 sum to zero and the L row sums to one. So each RGB channel is a cubic in `s`, needing no atan2, square root or table. Start on the chord from gray to the input, take 2 or 3 Newton steps on packed SIMD batches, and apply the same knee.

| ns/pixel, kernel only | Clip | Table | Solve, 3 steps |
|---|---:|---:|---:|
| Saturation-heavy P3 | 0.55 | 40.80 | 5.60 |
| Saturation-heavy BT.2020 | 0.55 | 42.28 | 7.19 |
| 10 real P3 images, range | 0.67 to 0.94 | 36.15 to 44.33 | 0.64 to 7.20 |

Whole P3 u8 to sRGB u8 rows on those images took 2.65 to 9.00 ns/pixel with 3 steps, against 1.23 to 1.25 for the fused clipping row. About 1.4 ns of that gap is the unfused route through f32, measured on the image where almost nothing reaches the solve. A fused kernel could remove it, but that hasn't been built.

| Quality on the survey's grids | Clip | Table | Solve, 3 steps |
|---|---:|---:|---:|
| BT.2020 blue hue shift | +18.985° | 0° | 0° |
| Max hue shift, BT.2020 grid | 22.84° | 0.16° | 0.0012° |
| In-gamut colors changed, P3 grid | 0 of 2,855 | 433 | 198 |
| Max change to an in-gamut color, P3 | 0 | 0.0244 | 0.0063 |
| Largest raw excursion past the cube, BT.2020 | 0 | 0.0025 | 0.000000 |
| Ramp reversals over 1e-7, P3 / BT.2020 | 1,780 / 2,270 | 288 / 345 | 0 / 380 |
| Largest reversal, P3 / BT.2020 | 0.0000038 / 0.0000039 | 0.178 / 0.179 | 0 / 0.00027 |

The table's large reversals came from a hue-seam bug, since fixed; its figures here predate the fix. It also needed a 64 KiB table per primaries and 2.72 ms to build one, and neither remains. On real images the solve and the table differed by at most 0.0064 in OKLab.

The solve keeps more chroma than bisection on the dark P3 blues, by up to 0.042 in OKLab, because the sRGB boundary there leaves and re-enters at fixed L and hue. Which is preferable hasn't been judged, and no visual comparison has been done.

moxcms doesn't need to change to use this: for recognizable matrix-shaper curves it can produce unclipped linear target RGB, and the mapper then runs before the target curve. A hook inside moxcms would touch its matrix kernels, 16 files and 5,625 lines across SSE, AVX, AVX-512 and NEON and the fixed-point variants. LUT profiles keep their authored tables.

### In the crate

`SoftCompress` now uses this directly, and the table is gone. Every SIMD tier gives identical bits, because it uses no fused operations. `ConvertOptions::gamut_mapping` opts images in, with clipping staying the default. It refuses, rather than clips, for ICC sources, HDR transfers and targets with no OKLab matrices.

| ns/pixel on the 10 real P3 images | Clip or table | Two-step |
|---|---:|---:|
| Whole u8 row, `GamutMapping::SoftCompress` | 1.24 to 1.25 (clip) | 2.81 to 9.10 |
| Planar OKLab, as zenfilters holds it | 12.02 to 18.58 (table) | 0.94 to 5.73 |

### zenfilters

- `GamutMapping::SoftCompress` used the table through `compress_planes`, building one per `Pipeline`. `compress_oklab_planes` is the replacement.
- `GamutMapping::ChromaReduce` is declared but never applied, so selecting it gives clipping. A knee of 1.0 is the same mapping.
- Its docs suggest a knee of 0.9, against the 0.96 default the June sweep chose.
- The June sweep chose 0.96 as the largest knee keeping pre-clamp out-of-gamut pixels under 0.1 % at corpus p90, measured through the table. An exact boundary leaves almost nothing out of gamut at any knee, so that criterion no longer separates knees. On a BT.2020 grid, knees from 0.90 to 0.99 left no pixels past 1e-7, and 1.0 left 11.8 % from roundoff at the boundary. On the 10 real P3 images, moving from 0.96 to 0.99 changed about 1 % fewer pixels, 9.27 % against 10.29 % on average, and mean OKLab error barely moved, 0.00059 against 0.00065. So 0.96 stays. The gain-mapped HDR corpus wasn't re-run, and any new criterion would need a visual judgment.
- `GamutLut` and `GamutMLP` expand sRGB toward P3, the other direction, so nothing here touches them.
