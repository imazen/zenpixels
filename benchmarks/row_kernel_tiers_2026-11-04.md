# Row kernel speed per dispatch tier, 2026-11-04

`cargo bench -p zenpixels-convert --bench row_kernel_tiers`, 1024-pixel rows, median of the group's
20 rounds, source and destination at fixed page offsets. x86-64 on r7900x (Ryzen 9 7900X, Zen 4,
double-pumped AVX-512). **Not measured on Zen 5**, where 512-bit kernels may gain more. "Old" is this
branch's source with every kernel still on the `#[autoversion]` default tiers, so it already includes
the gray f16 port and merged kernels; "new" is the same source with the tiers below, as `#[attune]` on
archmage 0.9.31-beta.2. Build: fat LTO, one codegen unit, except the last two columns (default bench
profile). `base` turns the runtime tiers off, so it shows what a
plain function compiles to: SSE2 with autovectorization. `new v4x` needs `--features avx512`.

## Rule

- Keep a v3 copy only if it beats `base` by at least 1.25x in both build profiles.
- Add `+v4x(cfg(avx512))` only if v4 beats v3 by at least 1.2x in both profiles.
- bt709 stays plain: `base` beats v3 there, and v4 by 19% in one direction and within 3% in the other.
- Kernels with no neon or wasm128 copy: those tiers are the baseline of their targets.

## x86-64

| kernel | old v4 | old v3 | old base | new v3 | new base | new v4x | tiers | old v4, default | new v3, default |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| add alpha u16 | 111 ns | 204 ns | 462 ns | 202 ns | 459 ns | 115 ns | v4x, v3 | 116 ns | 209 ns |
| drop alpha u16 | 71 ns | 93 ns | 217 ns | 89 ns | 211 ns | 73 ns | v4x, v3 | 73 ns | 92 ns |
| gray to rgb u16 | 66 ns | 57 ns | 84 ns | 56 ns | 84 ns | 58 ns | v3 | 69 ns | 63 ns |
| gray to rgba u16 | 92 ns | 67 ns | 139 ns | 66 ns | 206 ns | 66 ns | v3 | 98 ns | 70 ns |
| rgb to gray u16 | 761 ns | 870 ns | 1.28 µs | 867 ns | 1.28 µs | 856 ns | v3 | 755 ns | 867 ns |
| rgba to gray u16 | 772 ns | 1.12 µs | 1.36 µs | 1.40 µs | 1.40 µs | 771 ns | v4x | 774 ns | 1.40 µs |
| ga to rgba u16 | 74 ns | 66 ns | 133 ns | 66 ns | 130 ns | 66 ns | v3 | 77 ns | 68 ns |
| ga to rgb u16 | 68 ns | 57 ns | 297 ns | 57 ns | 302 ns | 62 ns | v3 | 72 ns | 61 ns |
| gray to ga u16 | 85 ns | 50 ns | 59 ns | 61 ns | 61 ns | 63 ns | plain | 88 ns | 66 ns |
| ga to gray u16 | 83 ns | 46 ns | 77 ns | 46 ns | 76 ns | 48 ns | v3 | 77 ns | 51 ns |
| add alpha f16 | 111 ns | 204 ns | 461 ns | 203 ns | 485 ns | 116 ns | v4x, v3 | 115 ns | 215 ns |
| drop alpha f16 | 70 ns | 95 ns | 210 ns | 88 ns | 211 ns | 73 ns | v4x, v3 | 73 ns | 96 ns |
| gray to rgb f16 | 66 ns | 56 ns | 84 ns | 56 ns | 84 ns | 58 ns | v3 | 69 ns | 60 ns |
| gray to rgba f16 | 92 ns | 68 ns | 139 ns | 66 ns | 163 ns | 67 ns | v3 | 96 ns | 71 ns |
| rgb to gray f16 | 327 ns | 333 ns | 2.52 µs | 320 ns | 2.55 µs | 330 ns | v4x, v3 | 327 ns | 334 ns |
| rgba to gray f16 | 370 ns | 451 ns | 3.11 µs | 445 ns | 3.10 µs | 376 ns | v4x, v3 | 377 ns | 444 ns |
| ga to rgba f16 | 75 ns | 65 ns | 129 ns | 65 ns | 130 ns | 67 ns | v3 | 76 ns | 68 ns |
| ga to rgb f16 | 68 ns | 57 ns | 299 ns | 57 ns | 302 ns | 58 ns | v3 | 72 ns | 61 ns |
| gray to ga f16 | 85 ns | 50 ns | 59 ns | 61 ns | 61 ns | 63 ns | plain | 88 ns | 65 ns |
| ga to gray f16 | 71 ns | 46 ns | 75 ns | 46 ns | 76 ns | 48 ns | v3 | 67 ns | 51 ns |
| add alpha f32 | 245 ns | 236 ns | 390 ns | 238 ns | 394 ns | 238 ns | v3 | 248 ns | 264 ns |
| drop alpha f32 | 122 ns | 101 ns | 297 ns | 100 ns | 300 ns | 105 ns | v3 | 123 ns | 107 ns |
| gray to rgb f32 | 94 ns | 88 ns | 150 ns | 89 ns | 153 ns | 89 ns | v3 | 98 ns | 93 ns |
| gray to rgba f32 | 141 ns | 117 ns | 195 ns | 116 ns | 197 ns | 119 ns | v3 | 150 ns | 121 ns |
| rgb to gray f32 | 272 ns | 371 ns | 898 ns | 350 ns | 792 ns | 277 ns | v4x, v3 | 257 ns | 381 ns |
| rgba to gray f32 | 302 ns | 473 ns | 689 ns | 472 ns | 747 ns | 339 ns | v4x, v3 | 328 ns | 472 ns |
| ga to rgba f32 | 114 ns | 112 ns | 197 ns | 110 ns | 198 ns | 115 ns | v3 | 116 ns | 117 ns |
| ga to rgb f32 | 98 ns | 88 ns | 151 ns | 88 ns | 152 ns | 89 ns | v3 | 102 ns | 99 ns |
| gray to ga f32 | 88 ns | 69 ns | 103 ns | 69 ns | 112 ns | 70 ns | v3 | 93 ns | 73 ns |
| ga to gray f32 | 60 ns | 53 ns | 73 ns | 52 ns | 72 ns | 55 ns | v3 | 65 ns | 57 ns |
| premul ga u8 | 68 ns | 82 ns | 495 ns | 84 ns | 493 ns | 65 ns | v4x, v3 | 71 ns | 86 ns |
| premul rgba u8 | 320 ns | 324 ns | 1.30 µs | 319 ns | 1.29 µs | 324 ns | garb | 324 ns | 324 ns |
| unpremul ga u8 | 946 ns | 956 ns | 778 ns | 788 ns | 787 ns | 783 ns | plain | 960 ns | 805 ns |
| unpremul rgba u8 | 2.20 µs | 2.20 µs | 2.20 µs | 2.20 µs | 2.20 µs | 2.20 µs | plain | 2.20 µs | 2.20 µs |
| premul ga u16 | 112 ns | 117 ns | 347 ns | 118 ns | 340 ns | 119 ns | v3 | 113 ns | 118 ns |
| premul rgba u16 | 338 ns | 702 ns | 1.69 µs | 701 ns | 1.70 µs | 339 ns | v4x, v3 | 339 ns | 700 ns |
| unpremul ga u16 | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | plain | 1.10 µs | 1.10 µs |
| unpremul rgba u16 | 3.30 µs | 3.30 µs | 3.30 µs | 3.30 µs | 3.30 µs | 3.30 µs | plain | 3.30 µs | 3.30 µs |
| premul ga f32 | 272 ns | 347 ns | 744 ns | 352 ns | 766 ns | 230 ns | v4x, v3 | 221 ns | 333 ns |
| premul rgba f32 | 421 ns | 426 ns | 1.47 µs | 412 ns | 1.46 µs | 418 ns | garb | 417 ns | 572 ns |
| unpremul ga f32 | 895 ns | 1.05 µs | 1.11 µs | 1.10 µs | 1.10 µs | 1.10 µs | plain | 908 ns | 1.10 µs |
| unpremul rgba f32 | 806 ns | 804 ns | 1.47 µs | 855 ns | 1.57 µs | 845 ns | garb | 842 ns | 822 ns |
| rgb to gray u8 | 735 ns | 801 ns | 1.08 µs | 808 ns | 1.09 µs | 803 ns | v3 | 739 ns | 811 ns |
| rgba to gray u8 | 758 ns | 767 ns | 1.11 µs | 769 ns | 1.10 µs | 758 ns | v3 | 756 ns | 777 ns |
| hlg u16 to linear | 2.60 µs | 3.20 µs | 3.50 µs | 3.50 µs | 3.50 µs | 2.70 µs | v4x | 2.60 µs | 3.50 µs |
| linear to hlg u16 | 3.80 µs | 3.70 µs | 4.70 µs | 3.60 µs | 4.70 µs | 3.70 µs | v3 | 3.80 µs | 3.70 µs |
| bt709 f32 to linear | 5.30 µs | 6.40 µs | 4.30 µs | 4.50 µs | 4.50 µs | 4.40 µs | plain | 5.40 µs | 4.50 µs |
| linear to bt709 f32 | 3.90 µs | 4.70 µs | 4.00 µs | 4.00 µs | 4.00 µs | 3.90 µs | plain | 3.90 µs | 3.80 µs |
| rgb to oklab | 2.50 µs | 3.40 µs | 5.50 µs | 3.60 µs | 5.80 µs | 2.50 µs | v4x, v3 | 2.50 µs | 3.50 µs |
| oklab to rgb | 589 ns | 1.35 µs | 2.66 µs | 1.40 µs | 2.70 µs | 606 ns | v4x, v3 | 656 ns | 1.40 µs |
| rgba to oklaba | 2.20 µs | 3.60 µs | 5.80 µs | 3.50 µs | 5.80 µs | 2.20 µs | v4x, v3 | 2.20 µs | 3.70 µs |
| oklaba to rgba | 968 ns | 1.76 µs | 3.62 µs | 1.80 µs | 3.70 µs | 952 ns | v4x, v3 | 957 ns | 1.70 µs |

The u16 and f16 rows of `add alpha`, `gray to rgba` and `gray to ga` now run one kernel, which takes the
alpha value as an argument.

Gray f16 before the port (`83839ef3`): `rgb to gray f16` took 1.30 µs at v4 and 3.90 µs at `base`. It
goes through the F16C slice converters and the f32 kernel now. NaN stays NaN, but which payload survives
can differ from per-sample software conversion, as in the f32 kernel. Other values match bit for bit.

## aarch64

`zen-arm-xl`, Neoverse N1, shared with a niced fuzz service, so differences up to about 20% on move
kernels are noise: `ga to rgb f16` measured 527 ns and 715 ns in the same old build. "Old" `neon` and
`base` matched throughout, because NEON is that target's baseline. Removing the neon copies changes no
kernel beyond that noise, and gray f16 is the one large change.

| kernel | old neon | new |
|---|---:|---:|
| rgb to gray f16 | 10.40 µs | 1.40 µs |
| rgba to gray f16 | 10.90 µs | 1.90 µs |
| add alpha u16 | 427 ns | 426 ns |
| rgb to gray f32 | 485 ns | 494 ns |
| premul rgba u16 | 1.80 µs | 1.80 µs |
| rgb to oklab | 10.20 µs | 10.30 µs |
| hlg u16 to linear | 8.60 µs | 8.60 µs |
| linear to bt709 f32 | 10.70 µs | 7.60 µs |

## Binary size

Bytes `RowConverter` adds to a stripped binary, from `benchmarks/converter-size`, fat LTO.

| build | before | tiers on 0.9.30 | attune on the beta |
|---|---:|---:|---:|
| x86-64 opt-level 3 | 522,784 | 432,464 | 436,416 |
| x86-64 opt-level s | 401 KB | 367,944 | 357,200 |
| x86-64 opt-level 3, `avx512` | not measured | 471,624 | 475,512 |
| wasm32 opt-level 3, simd128 | 299 KB | 295,888 | 295,633 |
| wasm32 opt-level z, simd128 | 256 KB | 256,226 | 250,311 |
| aarch64 opt-level 3 (section sums) | 245,328 | 246,696 | 253,672 |

aarch64 did not shrink. Its neon and baseline copies were already the same code. The last column also
picks up newer garb and linear-srgb `main`, so the 7 KB there isn't attributed to attune alone.

## Limits

- Harness bug, fixed: enabling V3 re-enables V4, so setting V4 first made every "v3" run v4. Tier
  tables measured before this file are wrong.
- `rgba to gray u16` and `hlg u16 to linear` have no v3 copy under the rule, so AVX2 machines get `base`.
- `premul rgba f32` varied by 30% between identical runs of the default profile.
- 16-pixel rows were measured but aren't listed: `base` is usually fastest there, by about 5 ns.

