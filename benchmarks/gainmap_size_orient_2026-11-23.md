# Gain-Map Separation, Padded SIMD-Only Codegen & Orientation Models (2026-11-23)

Measured on `r7900x` (AMD Ryzen 9 7900X, Zen 4, 24T, 30.0 GiB RAM) under
`run-heavy --mem 16G` (`peak-RSS 0.74–0.83 GiB`), without `-C target-cpu=native`.
Baseline commit: `c4626b22eff6be78c30c4dfaeea082d709e68e96`.

## 1. Stripped Binary Size (`benchmarks/converter-size/measure.sh` & `gainmap-size-probe`)

### `RowPlan`-only binaries (`gainmap` feature off in Lean; `Accelerators::ALL` excludes `gainmap_*`)

| Configuration | Tier | Before (`c4626b22`) | After | Delta |
|---|---|---:|---:|---:|
| Lean `x86_64 -O3` (`base=297,792 B`) | `SCALAR` | `+145,824 B` | **`+142,896 B`** | `-2,928 B` |
| Lean `x86_64 -O3` | `LAYOUT` | `+157,624 B` | **`+154,664 B`** | `-2,960 B` |
| Lean `x86_64 -O3` | `WEB_SDR` | `+231,536 B` | **`+201,408 B`** | **`-30,128 B` (-13.0%)** |
| Lean `x86_64 -O3` | `ALL` | `+270,424 B` | **`+208,760 B`** | **`-61,664 B` (-22.8%)** |
| Default `x86_64 -O3` (`base=297,792 B`) | `SCALAR` | `+196,704 B` | **`+193,792 B`** | `-2,912 B` |
| Default `x86_64 -O3` | `WEB_SDR` | `+285,504 B` | **`+255,408 B`** | **`-30,096 B` (-10.5%)** |
| Default `x86_64 -O3` | `ALL` | `+419,640 B` | **`+356,136 B`** | **`-63,504 B` (-15.1%)** |
| Default `wasm32 -Oz +simd128` (`base=32,751 B`) | `ALL` | `+281,178 B` | **`+268,258 B`** | **`-12,920 B`** |

- **`conv_fast` gainmap symbol check:** `0` gainmap symbols linked (`11,398 B` leak eliminated).

### Gain-map caller probe (`Lean + features = ["rgb", "gainmap"]`, delta over `base`)

| Binary | Target | Before (`c4626b22`) | After | Delta |
|---|---|---:|---:|---:|
| `gm_apply_scalar` (`SCALAR`) | `x86_64 -O3` | `+240,960 B` | **`+227,584 B`** | `-13,376 B` |
| `gm_apply_web_sdr` (`WEB_SDR.with_gainmap()`) | `x86_64 -O3` | — | **`+289,544 B`** | `-69,024 B` vs prior `ALL` |
| `gm_apply_all` (`ALL.with_gainmap()`) | `x86_64 -O3` | `+358,568 B` | **`+296,928 B`** | **`-61,640 B`** |
| `gm_extract` (`extract_gain_map_only`) | `x86_64 -O3` | `+236,240 B` | **`+236,504 B`** | `+264 B` |
| `gm_apply_scalar.wasm` | `wasm32 -Oz +simd128` | `+180,201 B` | **`+193,025 B`** | `+12,824 B` (orientation/P1 validation) |
| `gm_apply_web_sdr.wasm` | `wasm32 -Oz +simd128` | — | **`+230,303 B`** | — |
| `gm_apply_all.wasm` | `wasm32 -Oz +simd128` | `+229,947 B` | **`+235,592 B`** | `+5,645 B` |
| `gm_extract.wasm` | `wasm32 -Oz +simd128` | `+170,725 B` | **`+190,558 B`** | `+19,833 B` |

### Key Symbol Reductions (`nm -C --size-sort`, unstripped Lean `x86_64 -O3`)

| Symbol | Before | After | Reduction |
|---|---:|---:|---|
| `strip_simd::__arcane_unpack_strip_v3_t` | `17,503 B` | **`5,085 B`** | **-12,418 B (3.4×)** |
| `strip_simd::__arcane_srgb8_matrix_exact_v3_t` | `13,008 B` | **`5,619 B`** | **-7,389 B (2.3×)** |
| `strip_simd::__arcane_pack_strip_v3_t` | `8,263 B` | **`3,745 B`** | **-4,518 B (2.2×)** |
| `transfers::__arcane_{bt709_eotf,bt709_oetf,gamma}_v3_t` | `18,001 B` | **`~2,400 B`** | **-15,600 B (7.5×)** |
| `gainmap_simd` total (`ALL` + `SCALAR`) | `11,398 B` | **`4,820 B`** | **-6,578 B (2.4×)** |
| `gainmap_simd::__arcane_decode_gain_1ch_v3_t` | `1,914 B` | **`259 B`** | **-1,655 B (7.4×)** |
| `gainmap_simd::decode_gain_1ch_scalar` | `3,486 B` | **`147 B`** | **-3,339 B (23.7×)** |
| `gainmap::decode_gm_row_planar` | `4,676 B` | **`1,594 B`** | **-3,082 B (2.9×)** |
| `gainmap::apply_gain_map_with` → `GainMapApplier::new` + helpers | `36,279 B` (1 fn) | **`14,985 B`** (`new`, flat buffers) | de-inlined & flat scratch |

## 2. Clean Crate Compile Times (`zenpixels-convert`, warm deps)

| Feature Set | `cargo check` | `cargo build --release` | Before (`c4626b22`) release |
|---|---:|---:|---:|
| `--no-default-features` | `0.20s` | **`1.09s`** | `1.44s` (**-24%**) |
| `--no-default-features --features gainmap` | `0.21s` (`+0.01s`) | **`1.27s`** (`+0.18s`) | — |
| default (`std`, `rgb`, `gainmap`, `fast-transpose`, `oklab`, `hdr-transfers`) | `0.29s` | **`2.12s`** | `2.49s` (**-15%**) |

## 3. 4K UHD (3840×2160) Performance Verification (`bench_05_4k`)

Peak RSS during `bench_05_4k`: `0.74–0.83 GiB` (`run-heavy --mem 16G`).

| Benchmark | Before (`c4626b22`) | After | Throughput | Speedup vs SCALAR |
|---|---:|---:|---:|---:|
| `gainmap_4k/apply_qres_bilinear_u8_to_rgbf32` | `17.3 ms` | **`17.0 ms`** | `6.86 GiB/s` | `4.8×` (`81.9 ms`) |
| `gainmap_4k/apply_qres_bicubic_u8_to_rgbf32` | `18.8 ms` | **`18.4 ms`** | `6.32 GiB/s` | `4.5×` (`83.3 ms`) |
| `gainmap_4k/apply_qres_bilinear_u8_to_rgbaf16` | `23.7 ms` | **`23.5 ms`** | `3.96 GiB/s` | `8.1×` (`190.0 ms`) |
| `gainmap_4k/apply_1to1_lut_u8_to_rgbf32` | `14.8 ms` | **`13.7 ms`** | `9.02 GiB/s` | `1.33×` (`18.2 ms`) |
| `gainmap_4k/extract_qres_box_u8` | `26.1 ms` | **`28.5 ms`** | `4.08 GiB/s` | (`f64` signed-offset log₂) |
| `gainmap_4k/extract_qres_point_u8` | `8.6 ms` | **`10.3 ms`** | `11.3 GiB/s` | (`f64` signed-offset log₂) |
| `gamut/p3_to_srgb_rgb8` (Option A padded SIMD) | `22.6 ms` | **`22.6 ms`** | `2.05 GiB/s` | `41.3×` (`933.2 ms`) |
| `gamut/p3_to_srgb_rgb8_approx_q14` (Option B) | `9.8 ms` | **`9.8 ms`** | `4.72 GiB/s` | `2.6×` (`25.6 ms`) |
| `transfer_sdr/rgbf32_bt709_to_lin` (padded SIMD) | `17.9 ms` | **`17.9 ms`** | `10.4 GiB/s` | `21.4×` (`382.6 ms`) |
| `depth/rgb16_to_rgb8_round` | `2.8 ms` | **`2.8 ms`** | `25.2 GiB/s` | `1.0×` |
| `depth/p010_to_rgb16` | `47.1 ms` | **`47.1 ms`** | `1.97 GiB/s` | `1.52×` (`71.7 ms`) |
