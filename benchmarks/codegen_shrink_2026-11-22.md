# Codegen Shrink, Feature Gating & Capability Masking Results (2026-11-22)

- **Host:** `r7900x` — AMD Ryzen 9 7900X (Zen 4, 24T, 30.0 GiB RAM). **Not measured on Zen 5.**
- **Commits:**
  - Baseline: `da787b7f` (`benchmarks/baseline_4k_2026-11-21.md`, `linear-srgb` `f2430c90`)
  - `linear-srgb` PQ row-bailout fix: `ae83f194` (`fix/transfer-rails-magemath`)
  - Phase 2 (zero-functionality-loss padded-chunk tail elimination + cold scalar trim): `01cf0bed`
  - Phase 3 (Option B Cargo features: `oklab` default-on, `hdr-transfers` default-on, `perceptual` default-off): `0a238b57`
  - Phase 4 (const-generic capability bitmasking `caps` + `ConverterProfile<const CAPS: u64>`): this commit
- **Size harness:** `benchmarks/converter-size/` (`base`, `conv`, `conv_sdr_web`, `conv_layout_u8`; stripped, fat LTO, CGU=1, `panic=abort`, `run-heavy` peak RSS `0.78 GiB`)
- **4K UHD benchmark:** `cargo bench -p zenpixels-convert --all-features --bench bench_4k_paths` (`3840 × 2160` = `8,294,400` px/frame, `v3` tier, `run-heavy` peak RSS `0.35 GiB`)

## Stripped Binary Size Added (`bin - base`, bytes)

### Progression across Phases 1–4 (Option B Default & Lean SDR)

| Target | Baseline (`da787b7f`) | Phase 2 (`01cf0bed`) | Phase 3 Default (`+oklab,+hdr-tf`) | Phase 3 Lean SDR | Phase 4 `ALL` Default | Phase 4 `SDR_WEB` Default | Phase 4 `LAYOUT_U8` Default | Phase 4 `LAYOUT_U8` Lean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `x86_64 -O3` | `615,880` | `541,992` (-12.0%) | `466,840` (-24.2%) | `365,360` (-40.7%) | `512,176` (-16.8%) | `289,232` (-53.0%) | `189,648` (-69.2%) | `173,520` (-71.8%) |
| `x86_64 -Os` | `546,128` | `430,816` (-21.1%) | `378,520` (-30.7%) | `314,496` (-42.4%) | `383,936` (-29.7%) | `201,720` (-63.1%) | `134,432` (-75.4%) | `128,320` (-76.5%) |
| `x86_64 -O3 +avx512` | `725,016` | `630,488` (-13.0%) | `556,136` (-23.3%) | `419,744` (-42.1%) | `611,104` (-15.7%) | `305,504` (-57.9%) | `189,648` (-73.8%) | `173,536` (-76.1%) |
| `wasm32 -O3 +simd128` | `434,473` | `394,028` (-9.3%) | `336,926` (-22.5%) | `270,076` (-37.8%) | `371,090` (-14.6%) | `228,222` (-47.5%) | `167,642` (-61.4%) | `160,429` (-63.1%) |
| `wasm32 -Oz +simd128` | `337,495` | `296,676` (-12.1%) | `263,950` (-21.8%) | `213,422` (-36.8%) | `270,664` (-19.8%) | `164,336` (-51.3%) | `118,058` (-65.0%) | `104,609` (-69.0%) |

### Multi-Codec Feature Union Resilience (All 4 Heavy Features Enabled: `+oklab,+hdr-transfers,+perceptual,+bt2446a`)

| Target | Unmasked `ALL` (`conv`) | `ConverterProfile::<SDR_WEB>` | `ConverterProfile::<LAYOUT_U8>` |
|---|---:|---:|---:|
| `x86_64 -O3` | `663,360 B` | `311,496 B` (-53.0%) | `212,152 B` (-68.0%) |
| `x86_64 -Os` | `528,144 B` | `220,256 B` (-58.3%) | `153,144 B` (-71.0%) |
| `x86_64 -O3 +avx512` | `766,128 B` | `327,864 B` (-57.2%) | `212,136 B` (-72.3%) |
| `wasm32 -O3 +simd128` | `502,810 B` | `256,445 B` (-49.0%) | `196,324 B` (-61.0%) |
| `wasm32 -Oz +simd128` | `362,561 B` | `177,597 B` (-51.0%) | `131,402 B` (-63.8%) |

Unstripped `nm -C` audit on `conv_sdr_web` and `conv_layout_u8` with all 4 heavy features enabled confirmed zero row-kernel leakage for PQ, HLG, Oklab, `SoftCompress`, `Bt2446A`, perceptual spaces (`Jzazbz`, `Lch`, `Hsl`, `Hsv`, polar), U16, or F16.

## 4K UHD (`3840 × 2160`) Throughput Verification (`v3` tier, Zen 4 `r7900x`)

| Path | Baseline (`da787b7f`) | Phase 4 | Speedup / Delta |
|---|---:|---:|---:|
| `layout_u8/rgba8_to_rgb8` | `1.4 ms` (`39.1 GiB/s`) | `1.4 ms` (`39.7 GiB/s`) | +1.5% |
| `layout_u8/rgb8_to_bgra8` | `1.4 ms` (`38.2 GiB/s`) | `1.4 ms` (`39.3 GiB/s`) | +2.9% |
| `layout_wide/rgba16_to_rgb16` | `3.4 ms` (`32.1 GiB/s`) | `3.4 ms` (`31.6 GiB/s`) | parity |
| `gray_u8/rgb8_to_gray8_bt709` | `844.7 µs` (`36.6 GiB/s`) | `861.1 µs` (`35.9 GiB/s`) | parity |
| `gray_u8/rgba8_to_graya8` | `58.0 ms` (`818 MiB/s`) | `7.9 ms` (`5.83 GiB/s`) | **7.3× faster** |
| `gray_wide/rgb16_to_gray16` | `7.1 ms` (`8.76 GiB/s`) | `6.7 ms` (`9.24 GiB/s`) | **+5.6%** |
| `gray_wide/rgbf32_to_grayf32` | `3.9 ms` (`31.3 GiB/s`) | `3.9 ms` (`31.3 GiB/s`) | parity |
| `depth/rgb8_to_rgbf32_naive` | `5.3 ms` (`22.0 GiB/s`) | `4.4 ms` (`26.3 GiB/s`) | **+20%** |
| `depth/rgbf32_to_rgb8_round_untrusted` | `4.2 ms` (`27.6 GiB/s`) | `3.8 ms` (`30.2 GiB/s`) | **+9.5%** |
| `depth/rgbf32_to_rgb8_round_unit_clean` | `3.6 ms` (`32.5 GiB/s`) | `3.5 ms` (`33.5 GiB/s`) | +3.1% |
| `depth/rgbf32_to_rgb16_round_untrusted` | `4.5 ms` (`31.2 GiB/s`) | `4.4 ms` (`31.2 GiB/s`) | parity |
| `transfer_sdr/rgb8_srgb_to_rgbf32_lin` | `6.2 ms` (`18.8 GiB/s`) | `6.1 ms` (`19.0 GiB/s`) | +1.1% |
| `transfer_sdr/rgbf32_lin_to_rgb8_srgb` | `7.2 ms` (`16.2 GiB/s`) | `6.6 ms` (`17.4 GiB/s`) | **+7.4%** |
| `transfer_sdr/rgbf32_srgb_to_lin_clip` | `9.9 ms` (`18.8 GiB/s`) | `9.4 ms` (`19.6 GiB/s`) | **+4.3%** |
| `transfer_hdr/rgbf32_pq_to_lin_uniform` | `616.5 ms` (`308 MiB/s`) | `106.2 ms` (`1.75 GiB/s`) | **5.8× faster** |
| `transfer_hdr/rgbf32_lin_to_pq_uniform` | `22.6 ms` (`8.21 GiB/s`) | `23.4 ms` (`7.92 GiB/s`) | -3.5% (noise) |
| `gamut/p3_to_srgb_rgb8_matlut` | `20.1 ms` (`2.30 GiB/s`) | `20.5 ms` (`2.26 GiB/s`) | parity |
| `gamut/p3_to_srgb_lin_rgbf32` | `6.2 ms` (`30.0 GiB/s`) | `6.2 ms` (`29.9 GiB/s`) | parity |
| `premul/rgba8_straight_to_premul` | `2.8 ms` (`22.3 GiB/s`) | `2.8 ms` (`22.2 GiB/s`) | parity |
| `premul/rgbaf16_straight_to_premul` | `14.3 ms` (`8.65 GiB/s`) | `14.6 ms` (`8.47 GiB/s`) | parity |
| `premul/rgbaf32_straight_to_premul_untrusted` | `8.0 ms` (`30.9 GiB/s`) | `8.3 ms` (`29.9 GiB/s`) | parity |
| `matte/rgba8_srgb` | `17.2 ms` (`3.15 GiB/s`) | `15.2 ms` (`3.56 GiB/s`) | **+13.0%** |
| `matte/rgbaf32_srgb` | `25.0 ms` (`8.64 GiB/s`) | `14.4 ms` (`15.1 GiB/s`) | **+74.8%** |
| `oklab/rgb_to_oklab_untrusted` | `21.0 ms` (`8.83 GiB/s`) | `22.1 ms` (`8.37 GiB/s`) | parity (95% CI `[20.8–24.7]ms`) |
| `perceptual/srgb_rgb_to_hsl` | `36.9 ms` (`5.02 GiB/s`) | `40.0 ms` (`4.64 GiB/s`) | parity (95% CI `[38.5–41.7]ms`) |
