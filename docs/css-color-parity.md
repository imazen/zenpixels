# zenpixels against CSS color (2026-11-04)

Compared with the editor's drafts of [CSS Color 4](https://drafts.csswg.org/css-color-4/) (3 Nov 2026), [CSS Color 5](https://drafts.csswg.org/css-color-5/) and [CSS Color HDR 1](https://drafts.csswg.org/css-color-hdr-1/) (26 Oct 2026). Numbers come from `benchmarks/css-parity`, using the matrices and formulas in the spec's [conversions.js](https://drafts.csswg.org/css-color-4/conversions.js), in f64. No browser was run.

## Result

- `Color` can hold every CSS color once a parser resolves it. Spaces we don't name convert at parse time to linear BT.2020 `f32`, which keeps out-of-gamut values.
- sRGB, Display P3, Adobe RGB, BT.2020 primaries and the 203 cd/m² HDR anchors match to rounding.
- Five real differences, in [Differences](#differences).

zenpixels has no CSS parser. zenpipe parses hex and the CSS named colors (`src/bridge/parse.rs`); nothing parses the functional forms.

## Syntax

| CSS | In `Color` | Status |
|---|---|---|
| `#rgb`, named colors, integer `rgb()` | `Color::srgb8`, or `RGBA8_SRGB` with alpha | Direct |
| fractional `rgb()`, `hsl()`, `hwb()` | F32 `Srgb`, converted at parse | Direct; no HSL layout, and none needed |
| `color(srgb …)`, `color(srgb-linear …)` | F32 `Srgb`, F32 `Linear` | Direct |
| `color(display-p3 …)`, `display-p3-linear` | F32 `Srgb` or `Linear`, `DisplayP3` | Direct |
| `color(a98-rgb …)` | F32 `Gamma22`, `AdobeRgb` | Direct: same 563/256 exponent |
| `color(rec2100-pq …)`, `rec2100-hlg` | F32 `Pq` or `Hlg`, `Bt2020` | Direct |
| `oklab()` | F32 `Oklab` | Direct |
| `oklch()` | F32 `Oklab`, polar to rectangular at parse | Hue is gone once chroma is 0 |
| `color(rec2100-linear …)` | F32 `Linear`, `Bt2020` | **Anchor differs**: below |
| `color(rec2020 …)` | Linear `Bt2020`, decoded at parse | Gamma 2.4 has no `TransferFunction`; `Bt709` isn't it |
| `color(prophoto-rgb …)`, `xyz`, `xyz-d50`, `xyz-d65`, `lab()`, `lch()` | Linear `Bt2020`, converted at parse | No named space; D50 needs Bradford at parse |
| `ictcp()`, `jzazbz()`, `jzczhz()` | Linear `Bt2020`, converted at parse | No named space |
| `device-cmyk()` | F32 `Cmyk` holds it | Numeric conversion refused; CSS's naive formula isn't implemented |
| `color(--profile …)` with `@color-profile` | Converted through a CMS at parse | `Color` is `Copy` and can't hold ICC |
| `none` | Not representable | Parser's job; renders as 0, only matters for interpolation |
| `currentcolor`, `light-dark()`, `contrast-color()`, `color-mix()`, relative colors | Resolved before `Color` | Engine level |

Parsing to linear loses the original space, so a color can't be serialized back as written.

## Color management

| Topic | CSS | zenpixels |
|---|---|---|
| Conversion | Linearize, XYZ, Bradford if whites differ, destination | Same; all named primaries are D65 |
| Clamping during conversion | ["The results of color conversion are not clamped"](https://drafts.csswg.org/css-color-4/), §11.1 | Float output keeps them only with `with_clip_out_of_gamut(false)`; `permissive()` clips |
| Out of gamut at output | OkLCh chroma reduction for single colors; a perceptual intent suggested for images | Clips; `SoftCompressOklch` only on the experimental HDR path |
| Alpha | Unpremultiplied values, premultiplied for interpolation | Descriptor says which |
| Interpolation | Default Oklab; legacy sRGB in encoded sRGB; premultiplied | No color interpolation API; resampling is linear |
| [Compositing](https://drafts.csswg.org/compositing-1/) | The text read here names no space. Browsers are reported to use encoded sRGB ([csswg-drafts #6093](https://github.com/w3c/csswg-drafts/issues/6093), [public-colorweb, 2021](https://lists.w3.org/Archives/Public/public-colorweb/2021Jan/0044.html)); not verified | Linear; zenpipe's `MatteBlend::Encoded` opts into the other |
| SDR white in HDR | 203 cd/m²: 58% PQ, 75% HLG | `DiffuseWhite::BT2408`, for mattes: 0.5807 and 0.7499 measured |
| Headroom, `dynamic-range-limit`, `hdr-color()` | Defined | None |
| Rendering intents | `@color-profile`, relative colorimetric default | `cms::RenderingIntent` |
| Serialization | Precision rules per space | None |

## Measured agreement

| Check | Result |
|---|---|
| sRGB decode, 2M samples | Worst 4.93e-6; 1,583 differ as `u8`, all at rounding boundaries |
| sRGB encode, 2M samples | Worst 9.77e-6; 1,625 differ as `u8` |
| sRGB `u8` table | Worst 4.79e-6; 0 of 256 fail to round trip |
| Adobe RGB decode of 0.5 | 0.21775553, against 0.2177555 |
| Gamut matrices, 12 pairs of 4 spaces | Worst element off by 4.42e-7 |
| Oklab, 52³ sRGB colors, same matrices as CSS | Worst ΔEOK 2.31e-6; neutrals' chroma under 6e-7 in `f32` |

## Differences

**1. `rec2100-linear` means something else.** CSS puts 1.0 at 203 cd/m². Our linear puts 1.0 at PQ's peak, 10,000 cd/m². `color(rec2100-linear 1 1 1)` needs scaling by 203/10,000 at parse.

**2. SDR doesn't convert straight to PQ or HLG.** `Color::srgb8(255, 255, 255).try_convert_to(pq)` is refused, where CSS gives 0.58. `CompositeOnto` places white through `reference_white`. `RowConverter::new` still puts it at the peak, 10,000 cd/m².

**3. Clipping shifts hue.** To sRGB `u8`, against the spec's binary search implemented here. Hue and ΔEOK are measured before rounding, with the rounded result beside them. [The gamut survey](gamut-mapping-survey.md) adds other methods and a tolerance-free search:

| Source | Ours | Hue shift | CSS mapping | Hue shift |
|---|---|---:|---|---:|
| `display-p3 1 0 0` | 255, 0, 0 | +0.3° | 255, 11, 12 | 0.0° |
| `display-p3 0 1 0` | 0, 255, 0 | -3.1° | 0, 251, 41 | -2.3° |
| `display-p3 0 0 1` | 0, 0, 255 | 0.0° | 0, 0, 255 | 0.0° |
| linear BT.2020 `1 0 0` | 255, 0, 0 | +5.0° | 255, 73, 79 | -0.4° |
| linear BT.2020 `0 1 0` | 0, 255, 0 | -10.1° | 0, 242, 114 | -2.6° |
| linear BT.2020 `0 0 1` | 0, 0, 255 | +19.0° | 0, 81, 147 | +6.4° |

CSS applies this to individually specified colors, not images. Clipping is closer in ΔEOK in every case but blue P3, where they tie, and CSS gives up that distance to hold hue.

**4. Mattes blend in linear light, on purpose.** 50% red on white is green 187 here and 127 on encoded values, which is what browsers are reported to do.
