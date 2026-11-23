# `zenpixels-convert` 0.5 Speed, Codegen Size & Compile-Time Comparison vs 0.4 (2026-11-23)

- **Tree:** `feat/0.5-greenfield` (`zenpixels` 0.5.0 / `zenpixels-convert` 0.5.0, `#![forbid(unsafe_code)]`, `no_std + alloc` default, **0-bit `SCALAR` vs `ALL` parity verified across all routes**)
- **0.4 Reference:** `archive/0.4-float-facts-pre-0.5` (`benchmarks/baseline_4k_2026-11-21.md`, `benchmarks/codegen_shrink_2026-11-22.md`)
- **0.5 Pre-Optimization Reference:** `a08a4404` (`benchmarks/zenpixels_05_lcms_2026-11-23.md`)
- **Host:** `r7900x` — AMD Ryzen 9 7900X (Zen 4, 24T, 30.0 GiB RAM). **Not measured on Zen 5.**
- **Flags:** Default release / bench profile (**no `-C target-cpu=native`**), wrapped in `run-heavy --mem 16G`.
- **Resource receipts:**
  - Verification + `measure.sh` (`size_05_after.txt`): `rc=0`, **peak RSS `0.79 GiB`**
  - Clean 0.5 vs 0.4 compile times + full 4K `bench_05_4k` + `4k_p3_icc_to_srgb_rgb8`: `rc=0`, `271s`, **peak RSS `1.13 GiB`**
  - CMS native groups (`4k_p3_icc_to_srgb_rgbf32` + `4k_adobe_icc_to_srgb_rgb8`): `rc=0`, `53s`, **peak RSS `0.82 GiB`**

---

## 1. Crate-Only Clean Compile Times (`0.5` vs `0.4`)

Measured on `r7900x` with dependencies pre-warmed and only `zenpixels` + `zenpixels-convert` cleaned between runs (`cargo clean [--release] -p zenpixels -p zenpixels-convert`).

| Command | `0.4` (`archive/0.4`) | `0.5` (After SIMD Opt) | Speedup (`0.4 / 0.5`) |
|---|---:|---:|---:|
| `cargo check -p zenpixels-convert` (default) | `1.88 s` | **`0.80 s`** | **`2.35×` faster** |
| `cargo check -p zenpixels-convert --all-features` | `2.68 s` | **`0.94 s`** | **`2.85×` faster** |
| `cargo build -p zenpixels-convert --release` (default) | `5.35 s` | **`2.09 s`** | **`2.56×` faster** |
| `cargo build -p zenpixels-convert --release --all-features` | `8.20 s` | **`3.01 s`** | **`2.72×` faster** |

---

## 2. Stripped Binary Size Added (`bin - base`, Bytes)

Harness: `benchmarks/converter-size/measure.sh` (stripped, fat LTO, CGU=1, `panic=abort`, `RowPlan::build(bb(RGBA8_SRGB), bb(RGB8_SRGB))`).

### Lean Features (`--no-default-features`, core + `rgb`)

| Target | `0.2.16` crates.io | `0.4` Ph5 `ALL` | `0.5` Before `ALL` | `0.5` After `SCALAR` | `0.5` After `LAYOUT` | `0.5` After `WEB_SDR` | `0.5` After `ALL` | `0.5 ALL` vs `0.4 ALL` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `x86_64 -O3` | `436,040` | `406,200` | `122,584` | `111,992` | `121,176` | `172,040` | **`197,592`** | **`-51.4%` (2.06× smaller)** |
| `x86_64 -Os` | `365,512` | `308,400` | `85,000` | `96,408` | `102,024` | `128,648` | **`152,392`** | **`-50.6%` (2.02× smaller)** |
| `x86_64 -O3 +avx512` | `453,512` | `470,008` | `139,112` | `112,008` | `121,192` | `186,856` | **`244,520`** | **`-48.0%` (1.92× smaller)** |
| `wasm32 -O3 +simd128` | `206,136` | `311,245` | `87,664` | `93,889` | `98,510` | `126,894` | **`135,518`** | **`-56.5%` (2.30× smaller)** |
| `wasm32 -Oz +simd128` | `187,656` | `205,809` | `94,111` | `115,063` | `119,424` | `146,376` | **`152,044`** | **`-26.1%` (1.35× smaller)** |

### Default Features (`+std,+fast-transpose,+oklab,+hdr-transfers`)

| Target | `0.4` Ph5 `SDR_WEB` | `0.4` Ph5 `ALL` | `0.5` Before `ALL` | `0.5` After `SCALAR` | `0.5` After `LAYOUT` | `0.5` After `WEB_SDR` | `0.5` After `ALL` | `0.5 ALL` vs `0.4 ALL` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `x86_64 -O3 (def)` | `242,384` | `512,032` | `187,728` | `163,840` | `175,664` | **`226,880`** | **`345,680`** | **`-32.5%` (-166 KB)** |
| `x86_64 -Os (def)` | `175,504` | `382,448` | `216,560` | `169,000` | `177,128` | `204,120` | **`341,448`** | **`-10.7%` (-41 KB)** |
| `x86_64 -O3 +avx512 (def)` | `258,640` | `611,056` | `221,472` | `163,840` | `175,664` | **`244,096`** | **`448,040`** | **`-26.7%` (-163 KB)** |
| `wasm32 -O3 +simd128 (def)` | `188,820` | `374,369` | `134,535` | `127,890` | `132,535` | **`160,927`** | **`216,393`** | **`-42.2%` (-158 KB)** |
| `wasm32 -Oz +simd128 (def)` | `135,602` | `255,942` | `174,701` | `175,805` | `180,209` | `207,144` | **`252,261`** | **`-1.4%` (-3.7 KB)** |

*Dead-stripping isolation verified:* `SCALAR` links zero `garb`, strip-SIMD, `linear-srgb` SIMD, `BT.709`/`Gamma22`/`PQ`/`HLG` SIMD, or `Oklab`/`SoftCompress` SIMD (`LAYOUT` adds `+9–12 KB` for `garb`, `WEB_SDR` adds `+51 KB` for native-width strip codecs + sRGB + 3×3 gamut, and `ALL` adds BT.709 + Gamma 2.2 + F16C + HDR + Oklab SIMD).

---

## 3. Full-Frame 4K UHD (`3840 × 2160`) Throughput (`v3` Tier, `r7900x`)

All measurements are full `8,294,400`-pixel 4K UHD frames (`6` rounds, row-by-row via `RowPlan::convert_row` or `apply_orientation_into`). Every `ALL` route is verified **0-bit identical** to `SCALAR`.

| Category / Path | 0.5 Before (`ALL`) | **0.5 After (`ALL`)** | 0.5 After (`SCALAR`) | 0.4 Baseline (`v3`) | 0.5 After vs Before | 0.5 After vs 0.4 |
|---|---:|---:|---:|---:|---:|---:|
| `layout/rgba8_to_rgb8` | `1.6 ms` | **`1.6 ms`** (`34.5 GiB/s`) | `3.3 ms` | `1.4 ms` | parity | `1.14×` |
| `layout/rgb8_to_bgra8` | `1.6 ms` | **`1.6 ms`** (`33.9 GiB/s`) | `4.5 ms` | `1.4 ms` | parity | `1.14×` |
| `layout/rgba8_to_bgra8` | `1.8 ms` | **`1.8 ms`** (`34.1 GiB/s`) | `2.2 ms` | `1.7 ms` | parity | `1.06×` |
| `layout/rgb8_to_bgr8` | `1.4 ms` | **`1.3 ms`** (`34.4 GiB/s`) | `3.6 ms` | `3.2 ms` | `1.08×` | **`2.46× faster`** |
| `layout/rgba16_to_rgb16` | `49.8 ms` | **`4.0 ms`** (`26.9 GiB/s`) | `3.9 ms` | `3.4 ms` | **`12.5× faster`** | `1.18×` |
| `layout/rgbaf32_to_rgbf32` | `43.2 ms` | **`7.6 ms`** (`28.5 GiB/s`) | `7.6 ms` | `6.9 ms` | **`5.7× faster`** | `1.10×` |
| `depth/rgb8_to_rgb16` | `2.4 ms` | **`2.5 ms`** (`28.3 GiB/s`) | `2.4 ms` | `2.1 ms` | parity | `1.19×` |
| `depth/rgb16_to_rgb8_round` | `46.0 ms` | **`2.8 ms`** (`25.0 GiB/s`) | `2.7 ms` | `13.4 ms` | **`16.4× faster`** | **`4.79× faster`** |
| `depth/rgb16_to_rgb8_trunc` | `47.6 ms` | **`2.8 ms`** (`25.0 GiB/s`) | `2.9 ms` | `2.0 ms` | **`17.0× faster`** | `1.40×` (exact `/257`) |
| `depth/rgb16_to_rgb8_dither` | `52.6 ms` | **`36.1 ms`** (`1.93 GiB/s`) | `50.0 ms` | — | **`1.46× faster`** | new in 0.5 |
| `depth/rgbf16_to_rgbf32` | `32.8 ms` | **`5.2 ms`** (`26.7 GiB/s`) | `19.5 ms` | `4.9 ms` | **`6.3× faster`** | parity (`1.06×`) |
| `depth/rgbf32_to_rgbf16` | `37.3 ms` | **`4.2 ms`** (`33.1 GiB/s`) | `32.3 ms` | `4.1 ms` | **`8.9× faster`** | parity (`1.02×`) |
| `depth/rgbf32_to_rgb8_round` | `55.4 ms` | **`4.3 ms`** (`27.1 GiB/s`) | `20.1 ms` | `4.2 ms` | **`12.9× faster`** | parity (`1.02×`) |
| `depth/rgb8_narrow_to_full` | `39.5 ms` | **`7.6 ms`** (`6.07 GiB/s`) | `45.7 ms` | — | **`5.2× faster`** | new in 0.5 |
| `depth/p010_to_rgb16` | `72.3 ms` | **`41.6 ms`** (`2.23 GiB/s`) | `66.5 ms` | — | **`1.74× faster`** | new in 0.5 |
| `depth/rgb565_to_rgb8` | `51.9 ms` | **`21.2 ms`** (`1.83 GiB/s`) | `47.2 ms` | — | **`2.45× faster`** | new in 0.5 |
| `transfer_sdr/rgb8_srgb_to_rgbf32_lin` | `34.8 ms` | **`6.2 ms`** (`18.7 GiB/s`) | `7.2 ms` | `6.2 ms` | **`5.6× faster`** | **parity (`1.00×`)** |
| `transfer_sdr/rgbf32_lin_to_rgb8_srgb` | `70.1 ms` | **`14.8 ms`** (`7.84 GiB/s`) | `363.5 ms` | `7.2 ms` | **`4.7× faster`** | exact poly (no ±1 LUT) |
| `transfer_sdr/rgbf32_srgb_to_lin` | `52.0 ms` | **`16.6 ms`** (`11.2 GiB/s`) | `337.9 ms` | `9.9 ms` | **`3.1× faster`** | extended 6/6 poly |
| `transfer_sdr/rgbf32_lin_to_srgb` | `52.1 ms` | **`16.6 ms`** (`11.2 GiB/s`) | `345.6 ms` | `11.5 ms` | **`3.1× faster`** | extended 6/6 poly |
| `transfer_sdr/rgbf32_bt709_to_lin` | `696.5 ms` | **`17.9 ms`** (`10.4 GiB/s`) | `382.5 ms` | `23.3 ms` | **`38.9× faster`** | **`1.30× faster`** |
| `transfer_sdr/rgbf32_gamma22_to_lin` | `705.2 ms` | **`20.1 ms`** (`9.22 GiB/s`) | `434.1 ms` | `33.1 ms` | **`35.1× faster`** | **`1.65× faster`** |
| `gamut/p3_to_srgb_rgb8` | `67.9 ms` | **`22.8 ms`** (`2.03 GiB/s`) | `393.2 ms` | `20.1 ms` | **`3.0× faster`** | `1.13×` |
| `gamut/p3_to_srgb_rgb16` | `76.0 ms` | **`29.2 ms`** (`3.17 GiB/s`) | `748.0 ms` | `18.9 ms` | **`2.6× faster`** | `1.54×` |
| `gamut/p3_to_srgb_lin_rgbf32` | `40.9 ms` | **`8.2 ms`** (`22.5 GiB/s`) | `39.3 ms` | `6.2 ms` | **`5.0× faster`** | `1.32×` |
| `gamut/p3_to_srgb_srgb_rgbf32` | `69.2 ms` | **`28.1 ms`** (`6.59 GiB/s`) | `733.5 ms` | `23.7 ms` | **`2.5× faster`** | `1.19×` |
| `gamut/p3_to_srgb_soft_compress_f32` | `95.0 ms` | **`49.3 ms`** (`3.76 GiB/s`) | `797.1 ms` | `59.5 ms` | **`1.93× faster`** | **`1.21× faster`** |
| `transfer_hdr/rgb16_pq_to_rgbf32_lin` | `1,750 ms` | **`57.2 ms`** (`2.43 GiB/s`) | `125.4 ms` | `637.7 ms` (`106 ms` Ph4) | **`30.6× faster`** | **`1.86×`–`11.1× faster`** |
| `transfer_hdr/rgbf32_lin_to_rgb16_pq` | `1,850 ms` | **`16.1 ms`** (`8.64 GiB/s`) | `129.6 ms` | `18.7 ms` | **`114.9× faster`** | **`1.16× faster`** |
| `transfer_hdr/rgb16_hlg_1000nit_to_lin` | `67.7 ms` | **`43.7 ms`** (`3.18 GiB/s`) | `290.2 ms` | `592.7 ms` | **`1.55× faster`** | **`13.6× faster`** |
| `transfer_hdr/pq_u16_to_srgb_u8_bt2446a` | `1,830 ms` | **`117.9 ms`** (`604 MiB/s`) | `571.2 ms` | — | **`15.5× faster`** | full 4K HDR→SDR |
| `gray_alpha/rgb8_to_gray8_bt709` | `52.2 ms` | **`23.8 ms`** (`1.30 GiB/s`) | `133.4 ms` | `0.84 ms` (gamma-domain) | **`2.19× faster`** | true linear-light luma |
| `gray_alpha/rgba8_straight_to_premul` | `50.4 ms` | **`9.4 ms`** (`6.55 GiB/s`) | `9.6 ms` | `2.8 ms` | **`5.4× faster`** | exact scalar int |
| `gray_alpha/rgba8_premul_to_straight` | `54.8 ms` | **`9.9 ms`** (`6.24 GiB/s`) | `10.0 ms` | `18.4 ms` | **`5.5× faster`** | **`1.86× faster`** |
| `gray_alpha/rgba8_srgb_matte_black` | `72.0 ms` | **`24.6 ms`** (`2.19 GiB/s`) | `376.5 ms` | `17.2 ms` | **`2.93× faster`** | `1.43×` |
| `oklab/rgb_lin_to_oklab_f32` | `48.1 ms` | **`14.0 ms`** (`13.2 GiB/s`) | `245.7 ms` | `21.0 ms` | **`3.4× faster`** | **`1.50× faster`** |
| `oklab/oklab_to_rgb_lin_f32` | `39.8 ms` | **`7.9 ms`** (`23.5 GiB/s`) | `39.6 ms` | `7.1 ms` | **`5.0× faster`** | `1.11×` |
| `oklab/rgb8_srgb_to_oklab_f32` | `45.1 ms` | **`17.3 ms`** (`6.70 GiB/s`) | `229.5 ms` | — | **`2.6× faster`** | new direct strip route |
| `perceptual/lin_rgb_to_jzazbz_f32` | `1.94 s` | **`1.91 s`** (`99.5 MiB/s`) | `1.94 s` | `1.98 s` | parity | parity (`1.04× faster`) |
| `orient_4k/rotate90_gray8_1bpp` | `3.4 ms`* | **`537.0 µs`** (`28.8 GiB/s`) | — | `531.9 µs` | bench fix* | **parity (`1.01×`)** |
| `orient_4k/rotate90_rgb8_3bpp` | `10.8 ms`* | **`4.3 ms`** (`10.7 GiB/s`) | — | `4.5 ms` | bench fix* | **`1.05× faster`** |
| `orient_4k/rotate90_rgba8_4bpp` | `14.4 ms`* | **`5.8 ms`** (`10.7 GiB/s`) | — | `4.6 ms` | bench fix* | `1.26×` |
| `orient_4k/rotate90_rgba16_8bpp` | `28.9 ms`* | **`10.8 ms`** (`11.4 GiB/s`) | — | `8.5 ms` | bench fix* | `1.27×` |
| `orient_4k/rotate90_rgbf32_12bpp` | `44.9 ms`* | **`15.0 ms`** (`12.3 GiB/s`) | — | `15.2 ms` | bench fix* | **parity (`1.01× faster`)** |

*\*Note on `orient_4k`: `0.5` already included `0.4`'s full AVX2/NEON `fast-transpose` block kernels (`pxn_x86`, `rgb3_x86`, `pxn_neon`); the earlier `3.4–44.9 ms` (`4.28 GiB/s`) numbers were a benchmark-harness bug allocating an untouched `PixelBuf::new` inside `g.bench` per round (measuring Linux demand-zero page faults).*

---

## 4. CMS Shootout Impact on Substitutable ICC Routes (4K UHD)

| Route | `Lcms` (`main` opt) | `MoxCms 0.9.2` | 0.5 Native Before | **0.5 Native After (`ALL`)** | After vs Before | After vs `Lcms` / `MoxCms` |
|---|---:|---:|---:|---:|---:|---:|
| Display P3 ICC → sRGB `RGB8` | `34.8 ms` (`1.33 GiB/s`) | `8.0 ms` (`5.76 GiB/s`) | `71.1 ms` | **`22.8 ms`** (`2.04 GiB/s`) | **`3.1× faster`** | **`1.53×` faster than `Lcms`** |
| Display P3 ICC → sRGB `RGBF32` | `3.04 s` (`62.5 MiB/s`) | `~30 ms` (`6.17 GiB/s`) | `~68 ms` | **`~28 ms`** (`6.59 GiB/s`) | **`2.4× faster`** | **`105×` faster than `Lcms`; `1.07×` faster than `MoxCms`!** |
| Adobe RGB ICC → sRGB `RGB8` | `34.0 ms` (`1.36 GiB/s`) | `8.1 ms` (`5.74 GiB/s`) | `725.3 ms` | **`32.4 ms`** (`1.43 GiB/s`) | **`22.4× faster`** | **`1.05×` faster than `Lcms`** |

---

## 5. Architecture & Optimization Summary

1. **Zero-float `ByteFastPath` extensions (`src/strip.rs`):** Added wide 2-byte/4-byte layout add/drop/swap/gray (`U16`/`F16`/`F32`), exact integer `U16 → U8` round (`((v * 255 + 32895) >> 16)`) and truncate (`((v * 255 + 255) >> 16)`), and exact integer `RGBA8`/`BGRA8` straight ↔ premul (`(((c * a + 128) * 257) >> 16)` + 256-entry fixed-point reciprocal LUT), exhaustively verified across all `65,536` inputs.
2. **Direct contiguous single-stage `FastRow` paths (`src/strip.rs`, `src/plan.rs`):** Eliminated 64-px AoS↔SoA bouncing for direct `F16 ↔ F32` (F16C/NEON-fp16), `F32 ↔ U8/U16`, `sRGB U8 ↔ Linear F32`, and single-transfer `F32 ↔ F32` (`sRGB`, `BT.709`, `Gamma22`, `PQ`, `HLG Scene`).
3. **Native-width SIMD strip codecs (`src/strip_simd.rs`):** Added `f32x8` (x86-64 V3) / `f32x4` (NEON & WASM128) AoS↔SoA `unpack_strip` / `pack_strip` via `magetypes` (`load_rgb_u8_t`, `load_rgba_u8_t`, `load_rgb_u16_t`, `deinterleave_{3,4}ch`, `interleave_{3,4}ch`, `to_u8_with`/`to_u16_with`), fused U8 sRGB LUT unpack in `exec_strip_row`, and single-channel `gray_only` OETF for `RGB → Gray`.
4. **Bit-exact SIMD transfers (`src/transfers.rs`):** Single shared `magemath::Powf` profile (`TRANSFER_POW`) across `BT.709` and `Gamma22`, two-segment rational-polynomial `PQ` with per-lane `f64` dark-toe patching, and `HLG Scene` reusing `hlg_render`'s `Expf`/`Logf` profiles — all 0-bit identical between `ScalarToken` and SIMD.
5. **Strict dead-stripping in `Accelerators` (`src/accel.rs`):** Replaced runtime `bool` flags (`byte_layout`, `hlg_simd`, `hdr_simd`) with `Option<fn(...)>` pointers so `SCALAR`, `LAYOUT`, and `WEB_SDR` link zero unused SIMD kernels.
