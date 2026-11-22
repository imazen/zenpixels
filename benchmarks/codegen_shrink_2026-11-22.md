# Codegen Shrink, Feature Gating & Capability Masking Results (2026-11-22)

- **Host:** `r7900x` — AMD Ryzen 9 7900X (Zen 4, 24T, 30.0 GiB RAM). **Not measured on Zen 5.**
- **Commits:**
  - Baseline: `da787b7f` (`benchmarks/baseline_4k_2026-11-21.md`, `linear-srgb` `f2430c90`)
  - `linear-srgb` PQ row-bailout fix: `ae83f194` (`fix/transfer-rails-magemath`)
  - Phase 2 (zero-functionality-loss padded-chunk tail elimination + cold scalar trim): `01cf0bed`
  - Phase 3 (Option B Cargo features: `oklab` default-on, `hdr-transfers` default-on, `perceptual` default-off): `0a238b57`
  - Phase 4 (const-generic capability bitmasking `caps` + `ConverterProfile<const CAPS: u64>`): `90ac13d8`
  - Phase 5 (compact split SoA ICC hash tables, single-constructor `intermediate_desc`, `caps::PROFILES`, `PlannerHooks` + `plan_layout_only`): this commit
- **Size harness:** `benchmarks/converter-size/` (`base`, `conv`, `conv_sdr_web`, `conv_sdr_web_desc`, `conv_layout_u8`; stripped, fat LTO, CGU=1, `panic=abort`, `run-heavy` peak RSS `0.77 GiB`)
- **Reference baselines (same profile, stripped):**
  - Empty `fn main() {}`: `285,264 B` (`x86_64 -O3`), `278,224 B` (`x86_64 -Os`), `16,544 B` (`wasm32 -O3 +simd128`), `15,171 B` (`wasm32 -Oz +simd128`)
  - Crates.io `zenpixels-convert 0.2.16` (`RowConverter::new`, `default-features = false, features = ["rgb"]`, `bin - base_0216`; const desc / `black_box` dyn desc): `+436,040 / +490,976 B` (`x86_64 -O3`), `+365,512 / +391,352 B` (`x86_64 -Os`), `+453,512 / +508,496 B` (`x86_64 -O3 +avx512`), `+206,136 / +239,757 B` (`wasm32 -O3 +simd128`), `+187,656 / +208,423 B` (`wasm32 -Oz +simd128`)
- **4K UHD benchmark:** `cargo bench -p zenpixels-convert --all-features --bench bench_4k_paths` (`3840 × 2160` = `8,294,400` px/frame, `v3` tier, `run-heavy` peak RSS `0.35 GiB`)

## Stripped Binary Size Added (`bin - base`, bytes)

### Progression across Phases 1–5 (Option B Default & Lean SDR)

| Target | `0.2.16` crates.io | Baseline (`da787b7f`) | Phase 4 `ALL` Default | Phase 5 `ALL` Default | Phase 5 `ALL` Lean | Phase 5 `SDR_WEB` Default | Phase 5 `SDR_WEB & !PROFILES` Default | Phase 5 `LAYOUT_U8` Default | Phase 5 `LAYOUT_U8` Lean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `x86_64 -O3` | `436,040` | `615,880` | `512,176` | `512,032` (-16.9%) | `406,200` (-34.0%) | `242,384` (-60.6%) | `220,192` (-64.2%) | `60,352` (-90.2%) | `58,912` (-90.4%) |
| `x86_64 -Os` | `365,512` | `546,128` | `383,936` | `382,448` (-30.0%) | `308,400` (-43.5%) | `175,504` (-67.9%) | `162,968` (-70.2%) | `35,096` (-93.6%) | `34,264` (-93.7%) |
| `x86_64 -O3 +avx512` | `453,512` | `725,016` | `611,104` | `611,056` (-15.7%) | `470,008` (-35.2%) | `258,640` (-64.3%) | `236,512` (-67.4%) | `60,336` (-91.7%) | `58,912` (-91.9%) |
| `wasm32 -O3 +simd128` | `206,136` | `434,473` | `371,090` | `374,369` (-13.8%) | `311,245` (-28.4%) | `188,820` (-56.5%) | `163,616` (-62.3%) | `56,806` (-86.9%) | `55,600` (-87.2%) |
| `wasm32 -Oz +simd128` | `187,656` | `337,495` | `270,664` | `255,942` (-24.2%) | `205,809` (-39.0%) | `135,602` (-59.8%) | `124,514` (-63.1%) | `33,702` (-90.0%) | `32,498` (-90.4%) |

### Multi-Codec Feature Union Resilience (All 4 Heavy Features Enabled: `+oklab,+hdr-transfers,+perceptual,+bt2446a`)

| Target | Unmasked `ALL` (`conv`) | `ConverterProfile::<SDR_WEB>` | `SDR_WEB & !PROFILES` | `ConverterProfile::<LAYOUT_U8>` |
|---|---:|---:|---:|---:|
| `x86_64 -O3` | `691,672 B` | `244,864 B` (-64.6%) | `220,632 B` (-68.1%) | `62,560 B` (-91.0%) |
| `x86_64 -Os` | `552,328 B` | `187,328 B` (-66.1%) | `172,984 B` (-68.7%) | `37,560 B` (-93.2%) |
| `x86_64 -O3 +avx512` | `794,488 B` | `261,184 B` (-67.1%) | `237,000 B` (-70.2%) | `62,544 B` (-92.1%) |
| `wasm32 -O3 +simd128` | `503,008 B` | `197,458 B` (-60.7%) | `172,110 B` (-65.8%) | `59,659 B` (-88.1%) |
| `wasm32 -Oz +simd128` | `363,235 B` | `140,284 B` (-61.4%) | `128,901 B` (-64.5%) | `35,604 B` (-90.2%) |

### Dynamic ICC Context & Multi-Codec Union (Option B Default, `bin - base`)

| Target | `SDR_WEB` + runtime ICC `ColorContext` | `LAYOUT_U8` + `CODEC_JPEG` + `CODEC_PNG` (1 binary, with ICC) | `ALL` (for comparison) |
|---|---:|---:|---:|
| `x86_64 -O3` | `+291,152 B` | `+475,984 B` | `+512,032 B` |
| `x86_64 -Os` | `+201,832 B` | `+359,992 B` | `+382,448 B` |
| `wasm32 -O3 +simd128` | `+236,059 B` | `+332,522 B` | `+374,369 B` |
| `wasm32 -Oz +simd128` | `+151,298 B` | `+244,344 B` | `+255,942 B` |

### `imazen/archmage` PR #203 (`ea5d86e0`, `magemath` scalar & highp codegen) A/B on Phase 5 `ALL`

Measured with `[patch.crates-io]` overriding `archmage`/`magetypes`/`magemath` to PR #203 `ea5d86e0` (unmerged; not committed to `Cargo.toml`; `run-heavy` peak RSS `0.83 GiB`):

| Target | `ALL` Lean (before → PR #203) | `ALL` Default `+oklab,+hdr` (before → PR #203) | `ALL` All-4-Features (before → PR #203) |
|---|---:|---:|---:|
| `x86_64 -O3` | `+406,200 → +382,184 B` (**-24.0 KB**) | `+512,032 → +486,400 B` (**-25.6 KB**) | `+691,672 → +651,032 B` (**-40.6 KB**) |
| `x86_64 -Os` | `+308,400 → +272,080 B` (**-36.3 KB**) | `+382,448 → +341,296 B` (**-41.2 KB**) | `+552,328 → +501,208 B` (**-51.1 KB**) |
| `x86_64 -O3 +avx512` | `+470,008 → +441,832 B` (**-28.2 KB**) | `+611,056 → +581,072 B` (**-30.0 KB**) | `+794,488 → +749,560 B` (**-44.9 KB**) |
| `wasm32 -O3 +simd128` | `+311,245 → +298,116 B` (**-13.1 KB**) | `+374,369 → +360,883 B` (**-13.5 KB**) | `+503,008 → +489,272 B` (**-13.7 KB**) |
| `wasm32 -Oz +simd128` | `+205,809 → +189,796 B` (**-16.0 KB**) | `+255,942 → +235,757 B` (**-20.2 KB**) | `+363,235 → +343,010 B` (**-20.2 KB**) |

`SDR_WEB`, `SDR_WEB & !PROFILES`, and `LAYOUT_U8` changed by `<0.4 KB` under PR #203, independently confirming they already link zero `magemath` transcendental kernels.

Unstripped `nm -C` audit confirmed:
- Compact split SoA ICC tables (`SUBSTITUTABLE_{FNV64,MURMUR32,META}`): `2,870 B` (`205 × 14 B`, down from `10,584 B` = `-72.9%`) when ICC resolution is reachable, `0 B` in `conv_layout_u8` and `conv_sdr_web_desc`; `METADATA_ONLY_*` (`236 × 14 B = 3,304 B`) dead-strips unless `IccIdentification::from_icc` is called.
- `intermediate_desc`: `1,095 B` `.text` on `x86_64 -Os` (down from `10,398 B` `.text` + `5,700 B` `.rodata`), `0 B` in `conv_layout_u8`.
- Zero row-kernel or planner leakage for PQ, HLG, Oklab, `SoftCompress`, `Bt2446A`, perceptual spaces, U16, or F16 in `conv_sdr_web` / `conv_layout_u8`.

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
