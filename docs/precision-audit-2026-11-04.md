# Rounding and precision audit

Measured 2026-11-04 on r7900x (AVX-512 double-pumped; not measured on Zen 5 or ARM), with
`benchmarks/precision-audit/run.sh`. Each path is checked against `f64` math. Encode inputs are 2^22
values uniform in linear light plus 2^20 log-spaced from 1e-6 to 1. "Library constants" means
`linear-srgb`'s C0-continuous sRGB (`a = 0.0550107…`). "IEC" means 61966-2-1 (`a = 0.055`).

## Versions in use

| Where | `linear-srgb` | `garb` |
|---|---|---|
| zenpixels #108 | git `52896f4b` (unreleased) | git `605276d7` (unreleased) |
| zenpixels main, zenpipe, zenresize, zencodec, zenjpeg, zenpng | 0.6.12 | 0.2.8, where used |
| archived standalone zenfilters | 0.6.8 | 0.2.5 |

`52896f4b` doesn't compile without its `transfer` feature: `src/lanes.rs` uses `tf::bt709` and `tf::hlg`
unconditionally.

## Differences between versions

| Path | 0.6.8 + 0.2.5 | 0.6.12 + 0.2.8 | `52896f4b` + `605276d7` |
|---|---|---|---|
| `linear_to_srgb_u16_slice`, row split in 7s vs whole | 0 changed | **164 of 65,536 by 1** | 0 changed |
| `garb::convert_u16_to_f32`, row vs single | 512 by 1 ULP | 512 by 1 ULP | 0 |
| `garb::convert_u8_to_f32` | 1 by 1 ULP | 1 by 1 ULP | 0 |

## The same at every version

| Path | Result | Against |
|---|---|---|
| `linear_to_srgb_u8` (12-bit table), uniform | 64,806 of 4,194,304 (1.55%) off by 1 | Library constants |
| `linear_to_srgb_u8` (12-bit table), dark log | 84,213 of 1,048,576 (8.03%) off by 1 | Library constants |
| `linear_to_srgb_u16_slice` | 44,579 of 5,242,880 (0.85%) off by 1 | Library constants |
| `linear_to_srgb_u16_slice` | About 20% off by 1 | IEC |
| `srgb_u8_to_linear` | Exact | Library constants |
| `srgb_u8_to_linear` | 244 of 256 differ, up to 2.28e-4 relative (code 12) | IEC |
| `srgb_u16_to_linear` | 59,538 of 65,536 differ, up to 9 ULP | Library constants |
| u8 → linear → u8 | Every code returns | |
| u16 → linear → u16 | Every code returns | |
| 8, 10, 12, 14-bit code → canonical u16 → linear → u16 → code | Every code returns | |
| `garb::convert_u16_to_u8`, `(v*255+32768)>>16` | 127 of 65,536 off by 1 (33025 gives 128, nearest 129) | `round(v/257)` |
| `garb::convert_f32_to_u8` | 32 of 4,194,304 off by 1 | `f64` nearest |
| `garb::convert_f32_to_u16` | 5,460 of 4,194,304 off by 1 | `f64` nearest |

## What reaches output

- `linear_to_srgb_u8` is on every `f32` to sRGB `u8` output in `zenresize` (`src/color.rs:91`), so on every
  8-bit resize in linear light, and on zenpixels' `LinearF32ToSrgbU8`. Grid values survive, which is why
  round trips hide this. Values produced by filtering land between them.
- zenpixels' U16 to U8 uses its own `round(v/257)` kernel, not `garb::convert_u16_to_u8`. No other caller
  of the garb function was found in the crates searched.
- The 10-bit `LinearTable10`, the 12-bit `LinearTable12`, and `garb`'s `packed_1010102` had no callers
  in the crates searched.
- `zenresize`'s `i12` path needs RGBX-like input with no premultiply. An RGBX8 linear resize reported
  `F32` when measured, and zenpipe's RGBA8 path premultiplies.
- `linear-srgb`'s `iec` module doc says the constant sets differ only near the threshold at f32. In
  f32 they differ at 244 of 256 codes, and they agree after rounding to 8 bits.

## Not audited

Rounding in `zentone`, in the metric crates' local `linear_to_srgb_u8` helpers
(`zenmetrics-api/src/alpha.rs:58`, `cvvdp-gpu/src/heatmap.rs:127`), and decoder scaling in `heic`, `zenjxl`,
`zentiff`, `zenjp2`, and `zenraw`'s sensor depth. `zenavif` widens with bit replication, which differs from
canonical by 1 on 234 of 1,024 10-bit and 952 of 4,096 12-bit codes.
