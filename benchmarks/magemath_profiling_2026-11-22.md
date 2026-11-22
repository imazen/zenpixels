# `zenpixels-convert` Real Hardware Profiling & `magemath 0.1.1` Const Configuration (2026-11-22)

- **Host:** `r7900x` (AMD Ryzen 9 7900X, Zen 4, 24T, 30.0 GiB RAM, AVX2+FMA & AVX-512; **not measured on Zen 5**)
- **Baseline:** `051921c8` (`feat/0.4-float-facts`), built without `-C target-cpu=native`
- **Harness:** `bench_4k_paths` (4K UHD `3840 × 2160`, 8.29M px), `examples/measure_histogram_throughput`, `perf stat` / `perf record -F 999 -g --call-graph dwarf` / `perf report` / `perf annotate`
- **Resource envelope (`run-heavy --mem 16G`):** peak RSS `0.99 GiB` (benchmarks/perf), `2.68 GiB` (full test + clippy + `no_std` + `wasm32` matrix)
- **Determinism gate:** `kernel_tests::row_partition::EXPECTED_DIGEST = 0x922a875a7231ea47`, `every_tier_gives_the_same_bytes`, and `single_pixel_rows_match_a_whole_row` all preserved bit-for-bit (`0` digest change).

## 1. Before / After 4K UHD `zenbench` Results (`r7900x`, Zen 4)

### `v3` (AVX2 + FMA, `--features fast,bt2446a`)

| Benchmark (4K UHD) | Before | After | Speedup |
|---|---:|---:|---:|
| `transfer_sdr/rgbf32_gamma22_to_lin_clip` | 31.4 ms (5.91 GiB/s) | **12.5 ms (14.8 GiB/s)** | **2.51×** |
| `transfer_sdr/rgbf32_lin_to_gamma22_clip` | 31.6 ms (5.87 GiB/s) | **12.3 ms (15.1 GiB/s)** | **2.57×** |
| `transfer_sdr/rgbf32_gamma22_to_lin_ext_untrusted` | 147.8 ms (1.25 GiB/s) | **13.7 ms (13.5 GiB/s)** | **10.8×** |
| `transfer_sdr/rgbf32_gamma22_to_lin_ext_unit_clean` | 31.4 ms (5.90 GiB/s) | **12.5 ms (14.8 GiB/s)** | **2.51×** |
| `transfer_sdr/rgbf32_bt709_to_lin` | 24.2 ms (7.65 GiB/s) | **16.9 ms (11.0 GiB/s)** | **1.43×** |
| `transfer_sdr/rgbf32_lin_to_bt709` | 22.2 ms (8.36 GiB/s) | **15.4 ms (12.0 GiB/s)** | **1.44×** |
| `transfer_hdr/rgbf32_hlg_to_lin` (BT.2100-3 OOTF) | 439.5 ms (432 MiB/s) | **44.6 ms (4.15 GiB/s)** | **9.85×** |
| `transfer_hdr/rgbf32_lin_to_hlg` (BT.2100-3 OOTF) | 539.6 ms (352 MiB/s) | **47.5 ms (3.90 GiB/s)** | **11.4×** |
| `transfer_hdr/rgb16_hlg_to_rgbf32_lin` | 593.9 ms (240 MiB/s) | **48.5 ms (2.87 GiB/s)** | **12.2×** |
| `transfer_hdr/rgbf32_lin_to_rgb16_hlg` | 543.7 ms (262 MiB/s) | **50.2 ms (2.77 GiB/s)** | **10.8×** |
| `gamut/adobe_g22_to_srgb_rgbf32` | 45.4 ms (4.09 GiB/s) | **28.5 ms (6.51 GiB/s)** | **1.59×** |
| `gamut/bt2020_bt709_to_srgb_rgbf32` | 38.6 ms (4.80 GiB/s) | **32.0 ms (5.79 GiB/s)** | **1.21×** |
| `gamut/p3_to_srgb_soft_compress_f32` | 58.7 ms (3.16 GiB/s) | **57.6 ms (3.22 GiB/s)** | **1.02×** |
| `measure_histogram_throughput` 4K UHD `hist` | 358 Mpix/s | **366 Mpix/s** | **1.02×** |

### `v4x` (AVX-512, `--features fast,bt2446a,avx512`, `BENCH_4K_TIER=v4x`)

| Benchmark (4K UHD) | Before | After | Speedup |
|---|---:|---:|---:|
| `transfer_sdr/rgbf32_gamma22_to_lin_clip` | 16.8 ms (11.0 GiB/s) | **8.1 ms (22.9 GiB/s)** | **2.07×** |
| `transfer_sdr/rgbf32_lin_to_gamma22_clip` | 16.6 ms (11.2 GiB/s) | **8.1 ms (22.9 GiB/s)** | **2.05×** |
| `transfer_sdr/rgbf32_gamma22_to_lin_ext_untrusted` | 133.2 ms (1.39 GiB/s) | **8.5 ms (21.8 GiB/s)** | **15.7×** |
| `transfer_sdr/rgbf32_gamma22_to_lin_ext_unit_clean` | 17.2 ms (10.8 GiB/s) | **7.8 ms (23.8 GiB/s)** | **2.21×** |
| `transfer_sdr/rgbf32_bt709_to_lin` | 15.1 ms (12.3 GiB/s) | **9.9 ms (18.8 GiB/s)** | **1.53×** |
| `transfer_sdr/rgbf32_lin_to_bt709` | 14.0 ms (13.2 GiB/s) | **9.1 ms (20.3 GiB/s)** | **1.54×** |
| `transfer_hdr/rgbf32_hlg_to_lin` (BT.2100-3 OOTF) | ~440 ms (scalar) | **37.6 ms (4.93 GiB/s)** | **~11.7×** |
| `transfer_hdr/rgbf32_lin_to_hlg` (BT.2100-3 OOTF) | ~540 ms (scalar) | **38.7 ms (4.79 GiB/s)** | **~13.9×** |
| `gamut/adobe_g22_to_srgb_rgbf32` | — | **21.9 ms (8.47 GiB/s)** | — |
| `gamut/bt2020_bt709_to_srgb_rgbf32` | — | **23.8 ms (7.77 GiB/s)** | — |

### Controlled Micro-A/B Measurements (`v3`, `r7900x`)

- `gamma_math.rs` (`Highp` `PositiveGamma` elided vs unchecked full `Highp`): `15.8 ms → 11.2 ms` (**1.41×**, 0 bit diffs).
- `universal.rs` scalar (`ScalarToken`, 256K floats): Gamma22 encode `5.4 → 3.5 ms` (**1.55×**), BT.709 encode `4.0 → 3.5 ms` (**1.14×**), HLG exp `2.0 → 1.5 ms` (**1.33×**), HLG log `1.2 → 1.1 ms` (**1.12×**) — all 0 bit diffs.
- `hdr/measure.rs` `log2` micro (4M floats): legacy `log2_midp` `28.1 GiB/s` → `magemath` `Logf<PositiveNormals, assume_in_bounds>` `34.0 GiB/s` (**+21%**); full histogram is scatter-bound (`358 → 366 Mpix/s`).
- `hdr/bt2446a.rs` (1M px): legacy `magetypes` `*_midp_unchecked` `7.5 ms` → const-configured `magemath` `7.1 ms` (**1.06×**).

## 2. Hardware `perf` Root-Cause Findings

1. **Gamma 2.2 clipped (`convert_kernels.rs`):** `perf annotate` showed **44.35%–44.68%** of all cycles in `linear_srgb::default::{gamma_to_linear,linear_to_gamma}_slice` were spent on a single microcoded AVX2 `vmaskmovps` store emitted by LLVM for in-place `blend(v.simd_ne(v), v, result)`, plus `4.9% __memmove` + `restore_alpha`. Switching to out-of-place `src -> dst` `magemath 0.1.1` kernels with `const_exponent(ADOBE_GAMMA / INV_ADOBE_GAMMA)`, `elides_range_check == true` fast paths, in-register alpha masking, and native `f32x16` on `v4x` reduced `vmaskmovps` and `__memmove` to **0%**.
2. **Gamma 2.2 extended (`convert_kernels.rs`, `universal.rs`):** `perf report` showed `30.06% libm::powf` + `13.20%` scalar fallback loop for `|v| > 1.0`. Using `PowfDomain::PositiveGamma` with `base_range(1e-16, 65_536)` (EOTF) and `base_range(MIN_POSITIVE, 65_536)` (OETF) handles the entire HDR range in SIMD (`0% libm::powf`), staying 0-bit continuous with clipped `[0, 1]`.
3. **HLG display rendering (`hlg_render.rs`):** `perf report` showed 100% scalar `f64` (`28% libm::pow` + `15.8% libm::exp`). Vectorizing with `Expf::<HLG_EXP_FAST>` + `Logf::<HLG_LOG_FAST>` + packed binary64 `Powf::<HLG_OOTF_HIGHP>` (`<= 0.5 ULP`, verified `7.3e-9` at `0.5` gray and `5.96e-8` max error across full range) with per-lane `f64` fallback + padded tails preserves exact partition independence while delivering `9.9×–13.9×` speedup.
4. **BT.709 (`convert_kernels.rs`):** `perf annotate` showed `11.8%` `%rsp` stack spill/reload and `4.4–5.4% __memmove` in `linear_srgb` `v3`. Out-of-place `magemath` with `FloatFacts::NO_NAN | NO_INFINITE | AT_MOST_ONE` check elision + in-register alpha masking + `f32x16` on `v4x` achieved `1.43×–1.54×` speedup with `0` bit diffs.
5. **`recip` / `rsqrt` hardware verdict:**
   - `premul_to_straight` at 4K is **8.4 ms (29.3 GiB/s)** vs `straight_to_premul` **8.4 ms (29.5 GiB/s)** — 100% DRAM-bandwidth bound at 4K (`4.0 µs` vs `3.1 µs` in L2 `4096px`).
   - `vdivps` cycle share in `perf annotate`: `2.26%` (sRGB), `<= 1.2%` (BT.709 after), `1.92%` (soft_compress), `10.31%` (unpremul, DRAM-bound), `3.2–6.0%` (Oklab `fast_cbrt`, eliminated entirely by division-free `magemath::Cbrtf`).
   - **`rsqrt` has 0% use; `recip` (`vrcpps`) is not worth the accuracy loss anywhere.**
6. **Out-of-scope (`linear-srgb`) bottlenecks observed in `perf report`:**
   - **PQ (`rgbf32_pq_to_lin_uniform`, 105.9 ms v3; near-black 1.16 s):** `18.14% pq_to_linear_slice_x8_cfg::slow` + `6.74% libm __ieee754_pow_fma` + `4.64%` non-inlined `blend` (2% of uniform `[0,1]` samples `< 0.02` trigger 8× scalar `f64` fallback on ~15% of 8-lane vectors).
   - **sRGB (`rgbf32_srgb_to_lin_clip`, 9.5 ms v3):** `7.19% __memmove` exceeds `5.16% srgb_v3` compute (out-of-place slice API in `linear-srgb` would eliminate the copy).

## 3. Pending Decision: Oklab `fast_cbrt` → `magemath::Cbrtf<Midp>` (Not Landed)

Measured in scratch A/B harness (`v3`, 4K RGB→Oklab, `r7900x`; accuracy swept across `[1e-6, 2.0]`):

| Variant | 4K Time (`v3`) | Throughput | Speedup vs untrusted | Max ULP |
|---|---:|---:|---:|---:|
| Current `fast_cbrt` (untrusted) | 22.7 ms | 8.16 GiB/s | 1.00× | 14 ULP |
| Current `fast_cbrt` (`unit_finite`) | 18.3 ms | 10.1 GiB/s | 1.24× | 14 ULP |
| `magemath::Cbrtf<AllFloats, Midp>` | **15.4 ms** | **12.1 GiB/s** | **1.47×** | **1 ULP** |
| `magemath::Cbrtf<Unit, admit_subnormals, assume_in_bounds, Midp>` | **14.6 ms** | **12.7 GiB/s** | **1.55×** | **1 ULP** |
| `magemath::Cbrtf<PositiveNormals, Midp>` | **13.2 ms** | **14.0 GiB/s** | **1.72×** | **1 ULP** |
| `magemath::Cbrtf<PositiveNormals, Lowp>` | 12.2 ms | 15.2 GiB/s | 1.86× | 179 ULP |

*Soundness note:* `FloatFacts::NO_SUBNORMALS | POSITIVE` on linear RGB does **not** imply subnormal-free LMS (`m1[0][2] * f32::MIN_POSITIVE ≈ 0.0514 * MIN_POSITIVE` is subnormal), so `admit_subnormals()` is required for soundness on unit-clean RGB. Switching Oklab requires updating `row_partition::EXPECTED_DIGEST` and `fast_cbrt_spec` bit-identity tests, so it is held pending user confirmation.
