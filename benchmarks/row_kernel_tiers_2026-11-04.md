# Row kernels, 2026-11-04: tiers, transfer math and size

`cargo bench -p zenpixels-convert --bench row_kernel_tiers`, 1024-pixel rows, median of the group's 20
rounds, source and destination at fixed page offsets, floats uniform in `[0, 1]`. x86-64 on r7900x (Ryzen
9 7900X, Zen 4, double-pumped AVX-512), fat LTO, one codegen unit. **Not measured on Zen 5**, where 512-bit
kernels may gain more. Built on archmage 0.9.31-beta.2, garb `605276d7`, linear-srgb `2145ecbc`.

`base` turns the runtime tiers off, so it shows what a plain function compiles to: SSE2 with
autovectorization. `v4x` needs `--features avx512`.

## Size

Bytes `RowConverter` adds to a stripped binary, from `benchmarks/converter-size`, fat LTO.

| build | before | after |
|---|---:|---:|
| x86-64 opt-level 3 | 522,784 | 319,592 |
| x86-64 opt-level s | 401 KB | 194,640 |
| x86-64 opt-level 3, `avx512` | not measured | 366,600 |
| wasm32 opt-level 3, simd128 | 299 KB | 248,117 |
| wasm32 opt-level z, simd128 | 256 KB | 229,141 |
| aarch64 opt-level 3 (section sums) | 245,328 | 194,688 |

Where the x86-64 opt-level 3 reduction came from, in order:

| step | added bytes |
|---|---:|
| start | 522,784 |
| tier rule, merged kernels, gray f16 | 432,464 |
| `#[attune]`, newer garb and linear-srgb | 436,416 |
| HLG and BT.709 through slices | 427,632 |
| tails through the vector kernels | 412,560 |
| PQ through linear-srgb | 395,264 |
| one matte kernel | 339,120 |
| hand-written scalar tiers in linear-srgb | 319,592 |

New code by crate at the end, from symbol names: zenpixels-convert 209,018, linear-srgb 14,414, garb
11,252, others 21,479. Inlining counts toward the caller, so dependency code inlined into
zenpixels-convert appears under it. What remains largest: `ConvertPlan::new` 14 KB, gamut matrix kernels
about 27 KB, `matte_composite` 10 KB, `apply_step_u8` 7 KB.

## Compile time

Clean build of the whole graph with default features, `-j 2`: `cargo check` 4.5 s, debug 5.3 s, release
7.6 s. In release, by crate: zenpixels-convert 2.6 s, garb 2.4 s, magetypes 1.4 s, syn 1.3 s, linear-srgb
0.9 s. Fat LTO adds about 1.4 s to a final binary that uses `RowConverter`. These changes left it about
the same: 8.1 s before.

## Rule

- Keep a v3 copy only if it beats `base` by at least 1.25x in both build profiles.
- Add `+v4x(cfg(avx512))` only if v4 beats v3 by at least 1.2x in both profiles.
- No neon or wasm128 copies of autovectorized kernels: those are their targets' baseline.

The rule ran before the bench used uniform floats. Rechecked with uniform floats: every kept v3 copy and
v4x tier still passes or sits at the threshold. Plain kernels were not re-measured with a v3 copy.

## Kernels

"Before" is v3 before the transfer and matte changes.

| kernel | before | v3 | `base` | v4x | tiers |
|---|---:|---:|---:|---:|---|
| add alpha u16 | 204 ns | 203 ns | 461 ns | 113 ns | v4x, v3 |
| drop alpha u16 | 89 ns | 93 ns | 209 ns | 74 ns | v4x, v3 |
| gray to rgb u16 | 56 ns | 56 ns | 84 ns | 58 ns | v3 |
| gray to rgba u16 | 65 ns | 66 ns | 206 ns | 68 ns | v3 |
| rgb to gray u16 | 867 ns | 868 ns | 1.28 µs | 860 ns | v3 |
| rgba to gray u16 | 1.40 µs | 1.40 µs | 1.40 µs | 775 ns | v4x |
| ga to rgba u16 | 64 ns | 66 ns | 132 ns | 66 ns | v3 |
| ga to rgb u16 | 58 ns | 57 ns | 299 ns | 58 ns | v3 |
| gray to ga u16 | 55 ns | 62 ns | 62 ns | 62 ns | plain |
| ga to gray u16 | 46 ns | 46 ns | 76 ns | 48 ns | v3 |
| add alpha f16 | 201 ns | 203 ns | 463 ns | 113 ns | v4x, v3 |
| drop alpha f16 | 89 ns | 88 ns | 211 ns | 72 ns | v4x, v3 |
| gray to rgb f16 | 56 ns | 56 ns | 85 ns | 58 ns | v3 |
| gray to rgba f16 | 67 ns | 72 ns | 168 ns | 67 ns | v3 |
| rgb to gray f16 | 326 ns | 324 ns | 1.67 µs | 331 ns | v4x, v3 |
| rgba to gray f16 | 436 ns | 441 ns | 1.94 µs | 370 ns | v4x, v3 |
| ga to rgba f16 | 65 ns | 65 ns | 129 ns | 66 ns | v3 |
| ga to rgb f16 | 57 ns | 56 ns | 299 ns | 58 ns | v3 |
| gray to ga f16 | 55 ns | 61 ns | 61 ns | 62 ns | plain |
| ga to gray f16 | 46 ns | 46 ns | 75 ns | 48 ns | v3 |
| add alpha f32 | 246 ns | 237 ns | 395 ns | 241 ns | v3 |
| drop alpha f32 | 98 ns | 100 ns | 303 ns | 100 ns | v3 |
| gray to rgb f32 | 87 ns | 88 ns | 148 ns | 89 ns | v3 |
| gray to rgba f32 | 116 ns | 116 ns | 201 ns | 117 ns | v3 |
| rgb to gray f32 | 138 ns | 138 ns | 305 ns | 116 ns | v4x, v3 |
| rgba to gray f32 | 222 ns | 222 ns | 306 ns | 145 ns | v4x, v3 |
| ga to rgba f32 | 110 ns | 112 ns | 199 ns | 112 ns | v3 |
| ga to rgb f32 | 90 ns | 89 ns | 151 ns | 90 ns | v3 |
| gray to ga f32 | 68 ns | 68 ns | 112 ns | 70 ns | v3 |
| ga to gray f32 | 52 ns | 56 ns | 72 ns | 54 ns | v3 |
| premul ga u8 | 84 ns | 82 ns | 492 ns | 65 ns | v4x, v3 |
| premul rgba u8 | 319 ns | 318 ns | 1.31 µs | 321 ns | garb |
| unpremul ga u8 | 854 ns | 854 ns | 854 ns | 855 ns | plain |
| unpremul rgba u8 | 2.30 µs | 2.30 µs | 2.30 µs | 2.30 µs | plain |
| premul ga u16 | 114 ns | 115 ns | 352 ns | 116 ns | v3 |
| premul rgba u16 | 698 ns | 703 ns | 1.70 µs | 339 ns | v4x, v3 |
| unpremul ga u16 | 1.10 µs | 1.10 µs | 1.10 µs | 1.10 µs | plain |
| unpremul rgba u16 | 3.40 µs | 3.40 µs | 3.40 µs | 3.40 µs | plain |
| premul ga f32 | 105 ns | 105 ns | 172 ns | 105 ns | v4x, v3 |
| premul rgba f32 | 272 ns | 268 ns | 502 ns | 267 ns | garb |
| unpremul ga f32 | 589 ns | 590 ns | 590 ns | 588 ns | plain |
| unpremul rgba f32 | 333 ns | 334 ns | 686 ns | 334 ns | garb |
| matte u8 srgb | 4.10 µs | 2.20 µs | 4.80 µs | 2.50 µs | slices |
| matte u8 linear | 2.90 µs | 1.10 µs | 3.10 µs | 1.00 µs | slices |
| matte u8 bt709 | 96.60 µs | 3.80 µs | 101.90 µs | 3.00 µs | slices |
| matte u8 pq | 73.90 µs | 5.30 µs | 78.30 µs | 4.50 µs | slices |
| matte u8 hlg | 34.20 µs | 2.60 µs | 36.00 µs | 2.20 µs | slices |
| matte u16 srgb | 35.40 µs | 2.40 µs | 73.20 µs | 2.20 µs | slices |
| matte u16 linear | 3.00 µs | 1.10 µs | 3.60 µs | 1.10 µs | slices |
| matte u16 bt709 | 94.00 µs | 3.90 µs | 102.50 µs | 3.00 µs | slices |
| matte u16 pq | 74.20 µs | 5.00 µs | 77.00 µs | 4.30 µs | slices |
| matte u16 hlg | 33.80 µs | 2.60 µs | 36.00 µs | 2.30 µs | slices |
| matte f16 srgb | 67.40 µs | 2.10 µs | 72.10 µs | 1.90 µs | slices |
| matte f16 linear | 2.00 µs | 877 ns | 3.64 µs | 856 ns | slices |
| matte f16 bt709 | 95.90 µs | 3.60 µs | 98.60 µs | 2.70 µs | slices |
| matte f16 pq | 75.90 µs | 4.70 µs | 77.20 µs | 4.00 µs | slices |
| matte f16 hlg | 31.30 µs | 2.40 µs | 35.40 µs | 2.00 µs | slices |
| matte f32 srgb | 66.50 µs | 2.10 µs | 72.70 µs | 2.00 µs | slices |
| matte f32 linear | 654 ns | 853 ns | 1.39 µs | 909 ns | slices |
| matte f32 bt709 | 96.30 µs | 3.70 µs | 96.80 µs | 2.70 µs | slices |
| matte f32 pq | 74.90 µs | 4.70 µs | 78.20 µs | 4.10 µs | slices |
| matte f32 hlg | 30.70 µs | 2.30 µs | 33.10 µs | 2.00 µs | slices |
| rgb to gray u8 | 809 ns | 808 ns | 1.08 µs | 809 ns | v3 |
| rgba to gray u8 | 771 ns | 764 ns | 1.10 µs | 767 ns | v3 |
| hlg u16 to linear | 16.40 µs | 924 ns | 17.15 µs | 733 ns | slices |
| linear to hlg u16 | 3.10 µs | 1.10 µs | 22.70 µs | 1.10 µs | slices |
| bt709 f32 to linear | 47.70 µs | 1.50 µs | 48.20 µs | 1.10 µs | slices |
| linear to bt709 f32 | 48.80 µs | 1.40 µs | 51.70 µs | 897 ns | slices |
| hlg f32 to linear | 805 ns | 810 ns | 16.31 µs | 598 ns | slices |
| linear to hlg f32 | 797 ns | 793 ns | 20.44 µs | 668 ns | slices |
| pq u16 to linear | 5.20 µs | 2.80 µs | 36.60 µs | 2.00 µs | slices |
| linear to pq u16 | 5.50 µs | 1.90 µs | 41.00 µs | 1.80 µs | slices |
| pq f32 to linear | 5.10 µs | 2.40 µs | 36.80 µs | 1.80 µs | slices |
| linear to pq f32 | 5.10 µs | 1.60 µs | 41.90 µs | 1.50 µs | slices |
| pq f32 to linear near black | 5.10 µs | 6.30 µs | 30.70 µs | 4.40 µs | slices |
| linear to pq f32 near black | 5.10 µs | 6.50 µs | 29.70 µs | 4.90 µs | slices |
| rgb to oklab | 1.90 µs | 1.90 µs | 3.80 µs | 1.20 µs | v4x, v3 |
| oklab to rgb | 664 ns | 667 ns | 1.05 µs | 432 ns | v4x, v3 |
| rgba to oklaba | 2.30 µs | 2.30 µs | 4.00 µs | 1.30 µs | v4x, v3 |
| oklaba to rgba | 769 ns | 773 ns | 1.23 µs | 517 ns | v4x, v3 |

## PQ accuracy

Against ST 2084 in f64, a million samples each way, rounded to `u16`:

| | decode: wrong, worst | encode: wrong, worst | decode speed |
|---|---:|---:|---:|
| vendored "precise" kernel, before | 10.7%, 4 codes | 12.5%, 1 code | 5.1 µs |
| linear-srgb slice, after | 1.1%, 1 code | 0.7%, 1 code | 2.4 µs |

Near-black rows, the slice's worst case, cost 6.2 µs against 5.1 µs before.

## Position and tiers

- **Position:** 572 of the conversions tested gave a different result for a pixel depending on where it
  sat in a row, up to 4 `u16` codes on PQ. Now zero of 1792, at every tier (`tests/row_partition.rs`).
- **Tiers:** 528 of 1792 conversions still differ between v3 and `base`, because the vector kernels fuse
  multiply-add where the hardware does. 189 of those have integer output: 853 of 517,562 samples differ,
  by at most 1 code. Run `measure_differences_between_tiers` to reproduce.

## aarch64

`zen-arm-xl`, Neoverse N1, shared with a niced fuzz service, so differences up to about 20% on move kernels
are noise. 1011 tests pass there.

| kernel | before | after |
|---|---:|---:|
| rgb to gray f16 | 10.40 µs | 1.30 µs |
| rgba to gray f16 | 10.90 µs | 1.80 µs |
| add alpha u16 | 427 ns | 422 ns |
| rgb to gray f32 | 485 ns | 475 ns |
| premul rgba u16 | 1.80 µs | 1.80 µs |
| hlg u16 to linear | 8.60 µs | 5.90 µs |
| linear to hlg u16 | 6.20 µs | 6.10 µs |
| bt709 f32 to linear | 10.70 µs | 8.00 µs |
| linear to bt709 f32 | 10.70 µs | 7.40 µs |

## Costs

- Default builds have no v4 tier, so AVX-512 CPUs run some kernels up to 2.4x slower than before unless
  built with `avx512` (`add alpha`, `premul rgba u16`, `oklab to rgb`).
- Non-AVX2 x86-64 is slower on the slices' scalar tiers, which are smaller: up to 3x on `hlg f32 to linear`.
- `matte f32 linear` is 30% slower: it now splits channels and goes through scratch rows.
- Near-black PQ rows are about 25% slower at v3.

## Limits

- Harness bug, fixed: enabling V3 re-enables V4, so setting V4 first made every "v3" run v4. Tier tables
  measured before this file are wrong.
- `premul rgba f32` varied by 30% between identical runs.
- 16-pixel rows were measured but aren't listed.

