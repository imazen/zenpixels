# Row kernels, 2026-11-04: tiers, transfer math and size

`cargo bench -p zenpixels-convert --bench row_kernel_tiers`, 1024-pixel rows, median of the group's 20 rounds,
source and destination at fixed page offsets, floats uniform in `[0, 1]`. x86-64 on r7900x (Ryzen 9 7900X, Zen 4,
double-pumped AVX-512), fat LTO, one codegen unit. **Not measured on Zen 5**, where 512-bit kernels may gain more.

"Base" is the real PR base `5bd0d3aa`, on archmage 0.9.30, with this bench copied in. "Now" is this branch on
archmage 0.9.31-beta.2, garb `605276d7` and linear-srgb `2145ecbc`, measured before reference white, which adds one
multiply per row chunk to matte. `baseline` turns the runtime tiers off, so it runs what a plain function compiles to:
SSE2 with autovectorization. `v4x` needs `--features avx512`.

## Size

Bytes `RowConverter` adds to a stripped binary, from `benchmarks/converter-size`, fat LTO. An independent review
measured the x86-64 opt-level 3 pair as 523,440 and 319,672; small differences come from fresh dependency resolution.

| build | before | now |
|---|---:|---:|
| x86-64 opt-level 3 | 522,784 | 350,096 |
| x86-64 opt-level s | 401 KB | 198,920 |
| x86-64 opt-level 3, `avx512` | not measured | 399,776 |
| wasm32 opt-level 3, simd128 | 299 KB | 248,690 |
| wasm32 opt-level z, simd128 | 256 KB | 229,711 |
| aarch64 opt-level 3 (section sums) | 245,328 | 194,968 |

## Compile time

Clean build of the whole graph with default features, `-j 2`: `cargo check` 4.5 s, debug 5.3 s, release 7.6 s (8.1 s
before). In release, by crate: zenpixels-convert 2.6 s, garb 2.4 s, magetypes 1.4 s, syn 1.3 s, linear-srgb 0.9 s.
Measured before the last tier changes. Not rerun by the independent review.

## Tier rule

- Keep a v3 copy only if it beats `baseline` by at least 1.25x in both build profiles (fat LTO and default).
- Add `+v4x(cfg(avx512))` only if v4 beats v3 by at least 1.2x in both profiles.
- No neon or wasm128 copies of autovectorized kernels: those are their targets' baseline.

Applied to the base with uniform floats, in both profiles. Exceptions, judged on this branch's numbers because the base
doesn't represent them: `add alpha`, whose U16 and F16 kernels were merged, and Oklab encode, which had no tier gain
without LTO until it gained `#[inline]`.

## Kernels

| kernel | base v4 | base v3 | base `baseline` | now v3 | now `baseline` | now v4x | tiers |
|---|---:|---:|---:|---:|---:|---:|---|
| add alpha u16 | 323 ns | 549 ns | 344 ns | 203 ns | 460 ns | 107 ns | v4x, v3 |
| drop alpha u16 | 66 ns | 89 ns | 210 ns | 88 ns | 213 ns | 66 ns | v4x, v3 |
| gray to rgb u16 | 62 ns | 52 ns | 83 ns | 52 ns | 84 ns | 53 ns | v3 |
| gray to rgba u16 | 82 ns | 90 ns | 167 ns | 71 ns | 168 ns | 64 ns | v3 |
| rgb to gray u16 | 752 ns | 860 ns | 1.26 µs | 864 ns | 1.28 µs | 857 ns | v3 |
| rgba to gray u16 | 774 ns | 1.12 µs | 1.37 µs | 1.30 µs | 1.40 µs | 765 ns | v4x |
| ga to rgba u16 | 68 ns | 61 ns | 206 ns | 61 ns | 206 ns | 63 ns | v3 |
| ga to rgb u16 | 66 ns | 55 ns | 298 ns | 53 ns | 299 ns | 53 ns | v3 |
| gray to ga u16 | 79 ns | 43 ns | 55 ns | 47 ns | 61 ns | 46 ns | v3 |
| ga to gray u16 | 68 ns | 43 ns | 78 ns | 43 ns | 77 ns | 43 ns | v3 |
| add alpha f16 | 676 ns | 733 ns | 344 ns | 202 ns | 457 ns | 107 ns | v4x, v3 |
| drop alpha f16 | 66 ns | 90 ns | 210 ns | 88 ns | 210 ns | 66 ns | v4x, v3 |
| gray to rgb f16 | 62 ns | 52 ns | 84 ns | 53 ns | 83 ns | 52 ns | v3 |
| gray to rgba f16 | 80 ns | 88 ns | 166 ns | 61 ns | 162 ns | 63 ns | v3 |
| rgb to gray f16 | 1.20 µs | 2.20 µs | 3.90 µs | 329 ns | 1.66 µs | 331 ns | v4x, v3 |
| rgba to gray f16 | 1.20 µs | 2.20 µs | 3.80 µs | 434 ns | 2.05 µs | 361 ns | v4x, v3 |
| ga to rgba f16 | 68 ns | 60 ns | 206 ns | 62 ns | 209 ns | 63 ns | v3 |
| ga to rgb f16 | 64 ns | 54 ns | 300 ns | 54 ns | 301 ns | 54 ns | v3 |
| gray to ga f16 | 79 ns | 42 ns | 62 ns | 46 ns | 60 ns | 46 ns | v3 |
| ga to gray f16 | 68 ns | 43 ns | 78 ns | 43 ns | 76 ns | 43 ns | v3 |
| add alpha f32 | 240 ns | 232 ns | 393 ns | 234 ns | 394 ns | 233 ns | v3 |
| drop alpha f32 | 113 ns | 93 ns | 294 ns | 96 ns | 301 ns | 94 ns | v3 |
| gray to rgb f32 | 89 ns | 83 ns | 148 ns | 84 ns | 148 ns | 83 ns | v3 |
| gray to rgba f32 | 137 ns | 112 ns | 200 ns | 113 ns | 201 ns | 114 ns | v3 |
| rgb to gray f32 | 107 ns | 139 ns | 309 ns | 142 ns | 304 ns | 106 ns | v4x, v3 |
| rgba to gray f32 | 137 ns | 219 ns | 301 ns | 221 ns | 306 ns | 142 ns | v4x, v3 |
| ga to rgba f32 | 109 ns | 108 ns | 198 ns | 108 ns | 198 ns | 104 ns | v3 |
| ga to rgb f32 | 94 ns | 84 ns | 151 ns | 85 ns | 152 ns | 86 ns | v3 |
| gray to ga f32 | 83 ns | 65 ns | 112 ns | 65 ns | 102 ns | 65 ns | v3 |
| ga to gray f32 | 56 ns | 49 ns | 72 ns | 49 ns | 72 ns | 49 ns | v3 |
| premul ga u8 | 63 ns | 78 ns | 496 ns | 78 ns | 502 ns | 57 ns | v4x, v3 |
| premul rgba u8 | 316 ns | 316 ns | 1.30 µs | 317 ns | 1.31 µs | 314 ns | garb |
| unpremul ga u8 | 953 ns | 954 ns | 845 ns | 859 ns | 859 ns | 856 ns | plain |
| unpremul rgba u8 | 2.30 µs | 2.30 µs | 2.30 µs | 2.30 µs | 2.30 µs | 2.30 µs | plain |
| premul ga u16 | 105 ns | 111 ns | 338 ns | 117 ns | 340 ns | 112 ns | v3 |
| premul rgba u16 | 334 ns | 693 ns | 1.69 µs | 697 ns | 1.70 µs | 334 ns | v4x, v3 |
| unpremul ga u16 | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | plain |
| unpremul rgba u16 | 3.30 µs | 3.30 µs | 3.40 µs | 3.40 µs | 3.40 µs | 3.40 µs | plain |
| premul ga f32 | 99 ns | 101 ns | 172 ns | 102 ns | 183 ns | 101 ns | v3 |
| premul rgba f32 | 261 ns | 262 ns | 498 ns | 262 ns | 503 ns | 265 ns | garb |
| unpremul ga f32 | 404 ns | 605 ns | 584 ns | 588 ns | 588 ns | 402 ns | v4x |
| unpremul rgba f32 | 329 ns | 328 ns | 663 ns | 332 ns | 665 ns | 331 ns | garb |
| matte u8 srgb | 4.00 µs | 4.00 µs | 4.00 µs | 2.40 µs | 5.00 µs | 2.40 µs | slices |
| matte u8 linear | 2.80 µs | 2.80 µs | 2.90 µs | 1.10 µs | 3.10 µs | 1.10 µs | slices |
| matte u8 bt709 | 96.50 µs | 97.50 µs | 97.10 µs | 3.80 µs | 95.90 µs | 3.00 µs | slices |
| matte u8 pq | 73.20 µs | 73.30 µs | 73.30 µs | 5.30 µs | 79.70 µs | 4.60 µs | slices |
| matte u8 hlg | 33.70 µs | 33.70 µs | 33.70 µs | 2.50 µs | 34.40 µs | 2.30 µs | slices |
| matte u16 srgb | 36.50 µs | 37.60 µs | 37.50 µs | 2.40 µs | 72.40 µs | 2.30 µs | slices |
| matte u16 linear | 3.00 µs | 3.00 µs | 3.00 µs | 1.10 µs | 3.70 µs | 1.10 µs | slices |
| matte u16 bt709 | 97.50 µs | 97.40 µs | 97.20 µs | 3.90 µs | 98.60 µs | 3.10 µs | slices |
| matte u16 pq | 73.00 µs | 73.20 µs | 73.00 µs | 4.90 µs | 78.70 µs | 4.40 µs | slices |
| matte u16 hlg | 33.80 µs | 33.70 µs | 33.80 µs | 2.60 µs | 36.00 µs | 2.30 µs | slices |
| matte f16 srgb | 67.70 µs | 67.80 µs | 68.60 µs | 2.10 µs | 71.00 µs | 2.00 µs | slices |
| matte f16 linear | 2.00 µs | 2.00 µs | 3.70 µs | 884 ns | 3.66 µs | 874 ns | slices |
| matte f16 bt709 | 96.80 µs | 96.80 µs | 97.80 µs | 3.60 µs | 99.00 µs | 2.80 µs | slices |
| matte f16 pq | 75.10 µs | 75.20 µs | 76.10 µs | 4.70 µs | 78.40 µs | 4.10 µs | slices |
| matte f16 hlg | 32.00 µs | 32.00 µs | 33.00 µs | 2.40 µs | 34.80 µs | 2.00 µs | slices |
| matte f32 srgb | 66.60 µs | 66.70 µs | 66.80 µs | 2.20 µs | 70.40 µs | 2.00 µs | slices |
| matte f32 linear | 649 ns | 648 ns | 653 ns | 839 ns | 1.39 µs | 873 ns | slices |
| matte f32 bt709 | 94.30 µs | 95.00 µs | 94.20 µs | 3.60 µs | 97.30 µs | 2.80 µs | slices |
| matte f32 pq | 73.50 µs | 73.30 µs | 73.50 µs | 4.60 µs | 74.20 µs | 4.20 µs | slices |
| matte f32 hlg | 30.80 µs | 30.90 µs | 30.80 µs | 2.30 µs | 31.40 µs | 2.00 µs | slices |
| rgb to gray u8 | 733 ns | 806 ns | 1.08 µs | 811 ns | 1.09 µs | 798 ns | v3 |
| rgba to gray u8 | 747 ns | 757 ns | 1.07 µs | 768 ns | 1.10 µs | 759 ns | v3 |
| hlg u16 to linear | 2.60 µs | 3.20 µs | 16.90 µs | 921 ns | 17.50 µs | 727 ns | slices |
| linear to hlg u16 | 3.20 µs | 3.10 µs | 18.80 µs | 1.10 µs | 22.20 µs | 1.10 µs | slices |
| bt709 f32 to linear | 3.60 µs | 4.90 µs | 48.80 µs | 1.50 µs | 48.40 µs | 1.10 µs | slices |
| linear to bt709 f32 | 3.20 µs | 4.00 µs | 51.90 µs | 1.40 µs | 47.50 µs | 886 ns | slices |
| hlg f32 to linear | 802 ns | 805 ns | 5.37 µs | 804 ns | 16.79 µs | 588 ns | slices |
| linear to hlg f32 | 782 ns | 785 ns | 12.22 µs | 784 ns | 20.67 µs | 659 ns | slices |
| pq u16 to linear | 5.10 µs | 5.10 µs | 23.90 µs | 2.80 µs | 35.30 µs | 2.00 µs | slices |
| linear to pq u16 | 5.40 µs | 5.40 µs | 23.80 µs | 2.00 µs | 45.50 µs | 1.80 µs | slices |
| pq f32 to linear | 5.10 µs | 5.10 µs | 23.50 µs | 2.40 µs | 34.30 µs | 1.70 µs | slices |
| linear to pq f32 | 5.10 µs | 5.10 µs | 21.70 µs | 1.60 µs | 40.70 µs | 1.50 µs | slices |
| pq f32 to linear near black | 5.10 µs | 5.10 µs | 23.50 µs | 6.20 µs | 30.80 µs | 4.30 µs | slices |
| linear to pq f32 near black | 5.10 µs | 5.10 µs | 25.00 µs | 6.70 µs | 30.00 µs | 4.90 µs | slices |
| rgb to oklab | 1.20 µs | 1.90 µs | 3.80 µs | 2.10 µs | 4.20 µs | 1.20 µs | v4x, v3 |
| oklab to rgb | 427 ns | 664 ns | 1.04 µs | 662 ns | 1.05 µs | 419 ns | v4x, v3 |
| rgba to oklaba | 1.30 µs | 2.20 µs | 4.00 µs | 2.20 µs | 4.00 µs | 1.30 µs | v4x, v3 |
| oklaba to rgba | 518 ns | 778 ns | 1.22 µs | 772 ns | 1.24 µs | 513 ns | v4x, v3 |

## Costs

Worst ratios against the base, across the kernels above:

| Where | Worst | Example |
|---|---:|---|
| AVX-512 CPU, default build | 2.09x | `premul rgba u16`, 334 ns to 697 ns (v4 to v3) |
| AVX2 CPU | 1.31x | near-black `linear to pq f32` |
| No AVX2 | 3.13x | `hlg f32 to linear`, 5.37 µs to 16.79 µs |
| AVX-512 CPU, `avx512` build | 1.35x | `matte f32 linear` |

The largest gains at v3: matte up to 32x (`matte f16 srgb`; `matte f32 srgb` 66.7 µs to 2.2 µs), `hlg u16 to linear` 3.5x, BT.709 3.3x,
`linear to pq u16` 2.7x, `rgb to gray f16` 6.7x.

## PQ accuracy

Against ST 2084 in f64, a million samples each way, rounded to `u16`. Decode uses uniform signal. Encode percentages
depend on the grid: the review's uniform grid measured 16.7% and 0.8%, and this one uses `t^3`.

| | decode: wrong, worst | encode: wrong, worst |
|---|---:|---:|
| vendored kernel, base | 10.7%, 4 codes | 12.5%, 1 code |
| linear-srgb slice, now | 1.1%, 1 code | 0.7%, 1 code |

## Monotonicity

Every f32 in `[0, 1]`, 1,065,353,217 values in order, through `RowConverter` at v3, counting where a larger input gives a
smaller output. All `u16` reversals are 1 code. `tier-bits/monotonic.rs`.

| Curve | `u16`, base | `u16`, now | `f32`, base | `f32`, now |
|---|---:|---:|---:|---:|
| PQ decode | 0 | 5,185 | 7,051 | 533,912 |
| PQ encode | 886,928 | 17,934 | 6,478,185 | 6,684,199 |
| HLG decode | 0 | 0 | 0 | 0 |
| HLG encode | 1 | 0 | 1 | 0 |
| BT.709 decode | 0 | 0 | 1 | 0 |
| BT.709 encode | 14 | 8 | 2,007 | 2,316 |
| sRGB decode | 875 | 875 | 594,354 | 594,354 |
| sRGB encode | 10,835 | 10,835 | 3,315,023 | 3,315,023 |
| Gamma 2.2 decode | 0 | 0 | 0 | 0 |
| Gamma 2.2 encode | 11 | 11 | 3,712 | 3,712 |

## Position and tiers

- **Position:** 572 tested conversions gave a different result for a pixel depending on where it sat in a row, up to 4
  `u16` codes on PQ. Now none of 1792, at every tier (`tests/row_partition.rs`). That suite doesn't cover matte or layout
  changes.
- **Tiers:** every conversion now gives the same bytes at every tier, including gamut, matte and Oklab. See `tier-bits/README.md`.

## aarch64

`zen-arm-xl`, Neoverse N1, shared with a niced fuzz service, so differences up to about 20% on move kernels are noise. The
workspace tests pass there. Earlier measurements on this branch, not rerun after the last tier changes: `rgb to gray f16` 10.4 µs to
1.4 µs, `hlg u16 to linear` 8.6 µs to 5.9 µs, `linear to bt709 f32` 10.7 µs to 7.4 µs, the rest flat.

## Limits

- Harness bugs, fixed: enabling V3 re-enables V4, so setting V4 first made every "v3" run v4, and `set_tier` ran inside the timed
  loop. It now runs outside it and asserts the active tier. An earlier float pattern full of tiny values also misled
  several conclusions. Tier tables measured before this file are wrong.
- 16-pixel rows were measured but aren't listed, and microbenchmarks don't establish end-to-end codec speed.
