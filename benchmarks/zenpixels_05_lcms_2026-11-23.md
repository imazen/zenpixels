# `zenpixels` 0.5 + `imazen/lcms` (`main`) Benchmark Report (2026-11-23)

- **Tree:** `feat/0.5-greenfield` (parent `43b73b67`, `zenpixels` 0.5.0 / `zenpixels-convert` 0.5.0)
- **Dependencies:** `lcms` `https://github.com/imazen/lcms` `main` @ `e1d336ba`, `moxcms` `0.9.2`, `linear-srgb` `ae83f194`, `garb` `457ff8b9`, `archmage`/`magetypes` `0.9.33`, `magemath` `0.1.1`
- **Host:** `r7900x` — AMD Ryzen 9 7900X (Zen 4, 24T, 30.0 GiB RAM). **Not measured on Zen 5.**
- **Flags:** Default release bench profile (**no `-C target-cpu=native`**), wrapped in `run-heavy --mem 16G`.
- **Commands & resource receipts:**
  - `cargo bench -p zenpixels-convert --all-features --bench bench_cms_shootout -- --no-busy-gate` — `rc=0`, `301s` (incl. build), **peak RSS `1.98 GiB`**
  - `cargo bench -p zenpixels-convert --all-features --bench bench_05_4k -- --no-busy-gate` — `rc=0`, `246s`, **peak RSS `0.36 GiB`**
- **0.4 Reference:** `benchmarks/baseline_4k_2026-11-21.md` (measured on the same `r7900x` host at `1c08f781`, `v3` tier).

---

## 1. CMS Shootout: `imazen/lcms` (`main`) vs `moxcms 0.9.2` vs 0.5 Native SoA

### 4K UHD (`3840 × 2160` = `8,294,400` px/frame, row-by-row, 6 rounds)

| Route | `Lcms` (`main` opt, `NO_CACHE`) | `LcmsUnoptimized` (`NO_OPTIMIZE`) | `MoxCms` `0.9.2` | 0.5 Native SoA (`ALL`) | `Lcms` vs Unopt | `MoxCms` vs `Lcms` |
|---|---:|---:|---:|---:|---:|---:|
| Display P3 ICC → sRGB `RGB8` | `35.9 ms` (`1.29 GiB/s`) | `2,893.9 ms` (`16.4 MiB/s`) | `8.0 ms` (`5.76 GiB/s`) | `71.1 ms` (`668 MiB/s`) | `80.6×` | `4.5×` |
| Display P3 ICC → sRGB `RGBA8` | `64.3 ms` (`984 MiB/s`) | `2,933.1 ms` (`21.6 MiB/s`) | `9.0 ms` (`6.85 GiB/s`) | `78.2 ms` (`809 MiB/s`) | `45.6×` | `7.1×` |
| Display P3 ICC → sRGB `RGB16` | `103.6 ms` (`916 MiB/s`) | `2,858.1 ms` (`33.2 MiB/s`) | `31.3 ms` (`2.97 GiB/s`) | `77.0 ms` (`1.20 GiB/s`) | `27.6×` | `3.3×` |
| Display P3 ICC → sRGB `RGBF32` | `2,990 ms` (`63.5 MiB/s`) | — | `~30 ms` (`6.35 GiB/s`) | `~68 ms` (`2.76 GiB/s`) | — | `~100×` |
| Adobe RGB ICC → sRGB `RGB8` | `35.9 ms` (`1.29 GiB/s`) | `2,955.2 ms` (`16.1 MiB/s`) | `8.1 ms` (`5.75 GiB/s`) | `725.3 ms` (`65.4 MiB/s`) | `82.3×` | `4.4×` |
| Adobe RGB ICC → sRGB `RGBF32` | `3,000 ms` (`63.3 MiB/s`) | — | `~30 ms` (`6.20 GiB/s`) | `720 ms` (`263 MiB/s`) | — | `~98×` |
| 3D LUT ICC (`17³ mft2`) → sRGB `RGB8` | `66.4 ms` (`715 MiB/s`) | `3,426.7 ms` (`13.9 MiB/s`) | `21.4 ms` (`2.16 GiB/s`) | — | `51.6×` | `3.1×` |
| 3D LUT ICC (`17³ mft2`) → sRGB `RGB16` | `103.0 ms` (`921 MiB/s`) | `3,421.2 ms` (`27.7 MiB/s`) | `24.2 ms` (`3.83 GiB/s`) | — | `33.2×` | `4.3×` |
| 3D LUT ICC (`17³ mft2`) → sRGB `RGBF32` | `3,470 ms` (`54.7 MiB/s`) | — | `~50 ms` (`3.86 GiB/s`) | — | — | `~71×` |

### CMYK 4D CLUT ICC (`9⁴ mft2`, `4096`-px rows, 20 rounds)

| Route | `Lcms` (`main` opt) | `Lcms` (inverted) | `LcmsUnoptimized` | `MoxCms` `0.9.2` | `MoxCms` (inverted) |
|---|---:|---:|---:|---:|---:|
| `CMYK8 → RGB8` | `93.0 ±0.8 µs` (`44.0 Mpx/s`) | `93.0 ±0.6 µs` (`44.0 Mpx/s`) | `1,571.3 ±8.2 µs` (`2.61 Mpx/s`) | `22.3 ±0.2 µs` (`183 Mpx/s`) | `24.7 ±0.2 µs` (`166 Mpx/s`) |
| `CMYK16 → RGB16` | `91.6 ±1.4 µs` (`44.7 Mpx/s`) | — | — | `25.0 ±0.2 µs` (`164 Mpx/s`) | — |
| `CMYKF32 → RGBF32` | `1.6 ms` (`2.53 Mpx/s`) | — | — | `~38.6 µs` (`106 Mpx/s`) | — |

### Actionable Findings in `imazen/lcms` (`e1d336ba`)

1. **`RGBA_FLT -> RGBA_FLT` `COPY_ALPHA` clobber (`xform.rs:2340`):** `handle_extra_channels` runs before `xform_f32`, then `clutflt_tetra_loop_v3` / scalar fallback writes `0.0f32` into lane 3. `zenpixels-convert`'s `Lcms` adapter guards this with a post-pass alpha copy and tests it on both matrix-shaper and CLUT `RGBAF32` profiles.
2. **Missing `F32` SIMD fast paths for `RGB_FLT -> RGB_FLT` and `CMYK_FLT -> RGB_FLT`:** `lcms` only vectorizes `RGBA_FLT -> RGBA_FLT` (ClutOnly) and `CMYK_FLT -> CMYK_FLT` (ClutOnly); `RGB_FLT` and `CMYK_FLT -> RGB_FLT` drop to scalar `cpow` (`black_box(exp)`), causing the ~42–100× gap vs `moxcms`.
3. **`MatShaper` fast path only matches `RGB8 -> RGB8` (`xform.rs:1304`):** `RGBA8 -> RGBA8` drops from `35.9 ms` to `64.3 ms`.
4. **`clut8_eval4_loop_v3` / `clut16_eval4_loop_v3` only match `CMYK -> CMYK`:** `CMYK8/16 -> RGB8/16` falls through to generic unroll/pack loops (`93 µs` vs `moxcms` `22.3 µs`).

---

## 2. `zenpixels-convert` 0.5 4K UHD Engine (`ALL` vs `SCALAR` vs 0.4 Baseline)

All rows are full `3840 × 2160` (`8,294,400` px) frames on `r7900x` (`6` rounds).

| Category / Path | 0.5 `ALL` | 0.5 `SCALAR` | `ALL` vs `SCALAR` | 0.4 Baseline (`v3`) | 0.5 vs 0.4 |
|---|---:|---:|---:|---:|---:|
| `layout/rgba8_to_rgb8` | `1.6 ms` (`33.8 GiB/s`) | `3.2 ms` | `2.0×` | `1.4 ms` | `1.14×` |
| `layout/rgb8_to_bgra8` | `1.6 ms` (`33.1 GiB/s`) | `4.4 ms` | `2.7×` | `1.4 ms` | `1.14×` |
| `layout/rgba8_to_bgra8` | `1.8 ms` (`33.8 GiB/s`) | `2.2 ms` | `1.2×` | `1.7 ms` | `1.06×` |
| `layout/rgb8_to_bgr8` | `1.4 ms` (`34.1 GiB/s`) | `3.6 ms` | `2.7×` | `3.2 ms` | **`0.44×` (2.3× faster)** |
| `layout/rgba16_to_rgb16` | `49.8 ms` (`2.17 GiB/s`) | `49.9 ms` | `1.0×` | `3.4 ms` | `14.6×` (needs wide `ByteFastPath`) |
| `layout/rgbaf32_to_rgbf32` | `43.2 ms` (`5.01 GiB/s`) | `42.9 ms` | `1.0×` | `6.9 ms` | `6.3×` (needs wide `ByteFastPath`) |
| `depth/rgb8_to_rgb16` | `2.4 ms` (`28.6 GiB/s`) | `2.4 ms` | `1.0×` | `2.1 ms` | `1.14×` |
| `depth/rgb16_to_rgb8_round` | `46.0 ms` (`1.51 GiB/s`) | `46.0 ms` | `1.0×` | `13.4 ms` | `3.4×` (needs int `ByteFastPath`) |
| `depth/rgb16_to_rgb8_trunc` | `47.6 ms` (`1.46 GiB/s`) | `47.8 ms` | `1.0×` | `2.0 ms` | `23.8×` (needs int `ByteFastPath`) |
| `depth/rgb16_to_rgb8_dither` | `52.6 ms` (`1.32 GiB/s`) | `52.6 ms` | `1.0×` | — | new in 0.5 |
| `depth/rgbf16_to_rgbf32` | `32.8 ms` (`4.24 GiB/s`) | `51.9 ms` | `1.6×` | `4.9 ms` | `6.7×` (needs direct F16↔F32) |
| `depth/rgbf32_to_rgbf16` | `37.3 ms` (`3.73 GiB/s`) | `70.8 ms` | `1.9×` | `4.1 ms` | `9.1×` (needs direct F16↔F32) |
| `depth/rgbf32_to_rgb8_round` | `55.4 ms` (`2.09 GiB/s`) | `55.4 ms` | `1.0×` | `4.2 ms` | `13.2×` (scalar strip pack) |
| `depth/rgb8_narrow_to_full` | `39.5 ms` (`1.17 GiB/s`) | `39.5 ms` | `1.0×` | — | new in 0.5 |
| `depth/p010_to_rgb16` | `72.3 ms` (`1.28 GiB/s`) | `72.1 ms` | `1.0×` | — | new in 0.5 |
| `depth/rgb565_to_rgb8` | `51.9 ms` (`763 MiB/s`) | `52.0 ms` | `1.0×` | — | new in 0.5 |
| `transfer_sdr/rgb8_srgb_to_rgbf32_lin` | `34.8 ms` (`3.33 GiB/s`) | `354.0 ms` | `10.2×` | `6.2 ms` | `5.6×` (no U8 LUT fast path yet) |
| `transfer_sdr/rgbf32_lin_to_rgb8_srgb` | `70.1 ms` (`1.65 GiB/s`) | `397.5 ms` | `5.7×` | `7.2 ms` | `9.7×` (scalar strip unpack/pack) |
| `transfer_sdr/rgbf32_srgb_to_lin` | `52.0 ms` (`3.56 GiB/s`) | `374.1 ms` | `7.2×` | `9.9 ms` | `5.3×` (scalar strip unpack/pack) |
| `transfer_sdr/rgbf32_lin_to_srgb` | `52.1 ms` (`3.56 GiB/s`) | `378.4 ms` | `7.3×` | `11.5 ms` | `4.5×` (scalar strip unpack/pack) |
| `transfer_sdr/rgbf32_bt709_to_lin` | `696.5 ms` (`273 MiB/s`) | `698.0 ms` | `1.0×` | `23.3 ms` | `29.9×` (scalar `libm::powf`) |
| `transfer_sdr/rgbf32_gamma22_to_lin` | `705.2 ms` (`269 MiB/s`) | `706.3 ms` | `1.0×` | `33.1 ms` | `21.3×` (scalar `libm::powf`) |
| `gamut/p3_to_srgb_rgb8` | `67.9 ms` (`699 MiB/s`) | `721.5 ms` | `10.6×` | `20.1 ms` | `3.4×` (no matlut yet) |
| `gamut/p3_to_srgb_rgb16` | `76.0 ms` (`1.22 GiB/s`) | `731.4 ms` | `9.6×` | `18.9 ms` | `4.0×` (no matlut yet) |
| `gamut/p3_to_srgb_lin_rgbf32` | `40.9 ms` (`4.54 GiB/s`) | `41.7 ms` | `1.02×` | `6.2 ms` | `6.6×` (scalar SoA deinterleave) |
| `gamut/p3_to_srgb_srgb_rgbf32` | `69.2 ms` (`2.68 GiB/s`) | `722.5 ms` | `10.4×` | `23.7 ms` | `2.9×` (scalar SoA deinterleave) |
| `gamut/p3_to_srgb_soft_compress_f32` | `95.0 ms` (`1.95 GiB/s`) | `740.6 ms` | `7.8×` | `59.5 ms` | `1.6×` |
| `transfer_hdr/rgb16_pq_to_rgbf32_lin` | `1.75 s` (`81.3 MiB/s`) | `1.75 s` | `1.0×` | `637.7 ms` | `2.7×` (scalar `libm::powf`) |
| `transfer_hdr/rgbf32_lin_to_rgb16_pq` | `1.85 s` (`76.8 MiB/s`) | `1.86 s` | `1.0×` | `18.7 ms` | `99×` (scalar `libm::powf`) |
| `transfer_hdr/rgb16_hlg_1000nit_to_lin` | `67.7 ms` (`2.05 GiB/s`) | `291.8 ms` | `4.3×` | `592.7 ms` | **`0.11×` (8.8× faster!)** |
| `transfer_hdr/pq_u16_to_srgb_u8_bt2446a` | `1.83 s` (`39.0 MiB/s`) | `2.21 s` | `1.2×` | — | PQ EOTF dominates (`1.75 s`) |
| `gray_alpha/rgb8_to_gray8_bt709` | `52.2 ms` (`606 MiB/s`) | `694.5 ms` | `13.3×` | `0.84 ms` | 0.5 does true linear-light luma |
| `gray_alpha/rgba8_straight_to_premul` | `50.4 ms` (`1.23 GiB/s`) | `50.4 ms` | `1.0×` | `2.8 ms` | `18×` (needs int `ByteFastPath`) |
| `gray_alpha/rgba8_premul_to_straight` | `54.8 ms` (`1.13 GiB/s`) | `54.7 ms` | `1.0×` | `18.4 ms` | `3.0×` |
| `gray_alpha/rgba8_srgb_matte_black` | `72.0 ms` (`770 MiB/s`) | `714.4 ms` | `9.9×` | `17.2 ms` | `4.2×` |
| `oklab/rgb_lin_to_oklab_f32` | `48.1 ms` (`3.86 GiB/s`) | `248.6 ms` | `5.2×` | `21.0 ms` | `2.3×` (scalar SoA deinterleave) |
| `oklab/oklab_to_rgb_lin_f32` | `39.8 ms` (`4.66 GiB/s`) | `42.2 ms` | `1.06×` | `7.1 ms` | `5.6×` (scalar SoA deinterleave) |
| `oklab/rgb8_srgb_to_oklab_f32` | `45.1 ms` (`2.57 GiB/s`) | `563.0 ms` | `12.5×` | — | new direct strip route |
| `perceptual/lin_rgb_to_jzazbz_f32` | `1.94 s` (`97.9 MiB/s`) | `1.94 s` | `1.0×` | `1.98 s` | `0.98×` (parity) |
| `orient_4k/rotate90_gray8_1bpp` | `3.4 ms` (`4.61 GiB/s`) | — | — | `0.53 ms` | `6.4×` (scalar 16×16 tile) |
| `orient_4k/rotate90_rgb8_3bpp` | `10.8 ms` (`4.28 GiB/s`) | — | — | `4.5 ms` | `2.4×` (scalar 16×16 tile) |
| `orient_4k/rotate90_rgba8_4bpp` | `14.4 ms` (`4.29 GiB/s`) | — | — | `4.6 ms` | `3.1×` (scalar 16×16 tile) |
| `orient_4k/rotate90_rgba16_8bpp` | `28.9 ms` (`4.28 GiB/s`) | — | — | `8.5 ms` | `3.4×` (scalar 16×16 tile) |
| `orient_4k/rotate90_rgbf32_12bpp` | `44.9 ms` (`4.13 GiB/s`) | — | — | `15.2 ms` | `3.0×` (scalar 16×16 tile) |

### Top 4 Next-Phase Optimization Levers for 0.5 (Measured Impact)

1. **Vectorize `unpack_strip` / `pack_strip` (3×/4× AoS↔SoA deinterleave/interleave):** Eliminates the ~35–40 ms floor shared by *every* 4K strip route (F32, U16, U8, Oklab, Gamut, Matte).
2. **SIMD `powf` via `magemath` for `PQ`, `BT.709`, and `Gamma22`:** Replaces scalar `libm::powf` in `strip.rs` — recovers `21×–99×` on `Gamma22`, `BT.709`, `PQ`, and `BT.2446A` (just like HLG already jumped **8.8× faster than 0.4**!).
3. **Extend `ByteFastPath` for wide layouts (`RGBA16↔RGB16`, `RGBAF32↔RGBF32`), integer `U16→U8` trunc/round, direct `F16↔F32`, and integer U8/U16 `Straight↔Premul`:** Zero-float fast paths recovering `3.4×–24×`.
4. **Port 0.4's AVX2/NEON `fast-transpose` block kernels into `orient/`:** Recovers `2.4×–6.4×` on `orient_4k`.
