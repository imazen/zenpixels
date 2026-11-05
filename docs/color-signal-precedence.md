# ICC and CICP precedence: what readers do

From imazen/image-compat, round 14, 2026-10-17: synthetic fixtures through browsers, OS image stacks, libraries and tools, on Linux, macOS 27 and Windows 11. A verdict compares each reader's output with renderings of each hypothesis. Counts are reader modes, so one program can count more than once, and modes that don't apply color at all appear with the "stored" outcomes. Only measured behavior is here; the specification text is quoted on that project's feature pages.

| Fixture | Outcome | Modes | Who |
|---|---|---:|---|
| PNG: cICP P3 alone | Honored | 13 | Chrome, Edge, Firefox (color mode 2, and shipped), ImageIO, NSImage, sips with conversion, zen |
| | Treated as sRGB | 27 | Every library and tool otherwise, WIC, WebKitGTK |
| PNG: iCCP sRGB + cICP P3 | cICP wins | 13 | The same 13 |
| | iCCP or nothing | 27 | The same 27 |
| PNG: iCCP P3 + sRGB chunk | iCCP wins | 22 | Browsers, ImageIO, color-managed library modes, WIC, zen |
| PNG: `gAMA` + `cHRM` P3 | Honored | 13 | Chrome, Edge, Firefox, ImageIO, NSImage, sips, Qt with conversion |
| | Treated as sRGB | 27 | Including zen |
| AVIF: `prof` P3 + `nclx` sRGB | ICC wins | 16 | Chrome, Edge, Firefox, ImageIO, NSImage, sips, color-managed libraries, zen |
| | No conversion | 14 | avifdec, ffmpeg, plain library modes, WebKitGTK. All 14 also ignore `nclx` alone |
| | Reader chose `nclx` over ICC | 0 | |
| HEIC: `prof` + `nclx` that disagree | ICC wins | 11 | Color-managed GraphicsMagick, ImageMagick and libvips, zen, plus ImageIO, NSImage, sips with conversion and WIC, shown below |
| | `nclx` or no conversion | 7 | heif-dec and plain library modes |
| | No HEIC support | 4 | Chrome, Edge, Firefox |
| JPEG: P3 v4 and v2 profile | Same verdicts for both | 22 honored | ICC version made no difference |

## Reading HEIC on Apple and Windows

The verdicts read `other` for ImageIO, NSImage, sips and WIC, but that comes from the scoring. A verdict needs the runner-up 2x further than the best, measured by the p99 of each pixel's worst channel against renders from one decoder. HEVC decoders differ near sharp edges in this test pattern: ImageIO against libheif on the same stream gives p99 23 and max 72, with mean 0.87 on flat pixels and 5.12 near edges. That swamps the gap between hypotheses, 29 against 38.

Comparing a reader with itself cancels that difference:

| Reader | `conflict` vs `icc-p3` | `conflict` vs its own sRGB decode | vs its own decode converted from P3 |
|---|---|---|---|
| ImageIO, NSImage | 0 of 76,800 pixels differ | mean 1.52, max 104 | mean 0.33, max 2 |
| Quick Look (output stays P3) | 0 of 76,800 | mean 0.25, max 11 | n/a |
| WIC | 0 of 76,800 | mean 1.87, max 100 | mean 1.19, max 24 |

ImageIO's raw mode also reports `Display P3 v4` as the image's color space on the conflict fixture, and CICP 12/13/0 on the `nclx`-only one. So ICC wins on both. On range they differ: WIC applies `nclx` limited range next to the ICC profile (raw mean 9.56), and ImageIO doesn't (0 pixels change).

## What this settles

1. **Decoders.** `ColorAuthority` follows the table in `zenpixels::color`: PNG prefers cICP, as its Third Edition ranks and as every reader that reads cICP does. AVIF prefers ICC when both exist: no reader chose `nclx`. HEIC matches AVIF: both take their `colr` rule from MIAF, and no color-managed reader measured here chose `nclx`, including Apple's and Windows'.
2. **Encoders.** A cICP chunk alone loses wide gamut in 27 of 41 PNG reader modes. Write iCCP and cICP together, describing the same space. Neither may contradict the pixels, because readers split on which they use.
3. **Profile versions.** v2 and v4 matrix profiles behaved the same in every reader, so either is safe for matrix spaces.

## Not measured

An ICC profile with its own `cicp` tag, or a CICP-only transfer such as PQ against an SDR ICC profile in a conflict fixture. On-screen HDR isn't measured, because headless screenshots are SDR.
