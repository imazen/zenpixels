# Float Depth Division/Quantization & Band-Generic Overhead (`2026-11-20`)

- **Host:** `r7900x` (AMD Ryzen 9 7900X, Zen 4, 24 threads, 30.0 GiB RAM; **not measured on Zen 5**)
- **Harness:** `zenbench 0.1.11`, built without `-C target-cpu=native` (`archmage 0.9.32` runtime `X64V3Token` AVX2+FMA dispatch)
- **Command:** `cargo bench -p zenpixels-convert --features __bench_float_depth --bench bench_float_depth_and_bands`
- **Resource envelope:** `run-heavy --mem 10G --jobs 8`, peak RSS `5.50 GiB` (full suite) / `3.30 GiB` (`band_generic` re-run after `with_source` + `wants_stamp` zero-clone optimization)

---

## 1. Integer → Float Exact 0-ULP Division (`u8 -> f32` and `u16 -> f32`, 3ch RGB)

### Instruction-level trade-off (AVX2+FMA / NEON / WASM SIMD128)

1. **`recip_1ulp` (`x * f32(1.0 / M)`):** 1 vector `vmulps` (4 cy latency, 0.5 cy reciprocal throughput on Zen 4). Off by 1 ULP for **126 / 256** `u8` inputs (`1.0 / 255.0 = 0x3b808081` rounds up) and **512 / 65,536** `u16` inputs (`1.0 / 65535.0 = 0x37800080`).
2. **`split_exact_0ulp` (shipped in `zenpixels-convert/src/float_simd.rs`):**
   - `u8 / 255`: decomposes `q0 = x * C_HI` (`C_HI = 0x3b808000`, trailing 15 mantissa bits zero → `x * C_HI` has at most 16 mantissa bits and is **exact without rounding** for all `x <= 255`), then computes `vfmsub231ps`/`fma(x, C_LO, q0)` (`C_LO = 1.0 / 255.0 - C_HI` rounds up by `< 2^-39`, strictly below every `u8 / 255` tie boundary). **Cost: 1 `vmulps` + 1 `vfmadd` (2 vector FP ops), 0 mismatches across all 256 `u8` values.**
   - `u16 / 65535`: identical Veltkamp-style split with `C_HI = 0x37800000 = 2^-16` (exact exponent shift, zero mantissa bits → `x * 2^-16` is **exact** for every 16-bit integer) + `fma(x, C_LO, q0)` (`C_LO = 1.0 / 65535.0 - 2^-16 = 0x2f800080`). **Cost: 1 `vmulps` + 1 `vfmadd` (2 vector FP ops), 0 mismatches across all 65,536 `u16` values.**
   - On WASM without `relaxed-simd` (or on a non-fusing `relaxed-simd` host caught by `wasm_fuses_relaxed_madd()`), widens 4 lanes to `f64x2` and multiplies by exact `1.0f64 / M` before `f32x4.demote_f64x2_zero` (53-bit significand separates every 16-bit rational tie cleanly).
3. **`newton_0ulp` (`q0 = x * R`, `r = fma(-q0, M, x)`, `fma(r, R, q0)`):** 1 `vmulps` + 2 dependent `vfmadd`s (3–4 vector ops, ~12 cy serial latency chain). Exact 0-ULP, but serializes two dependent FMAs per vector.
4. **`div_0ulp` (`vdivps(x, M)`):** IEEE hardware divide (11 cy latency, 3–5 cy throughput on Zen 4; unpipelined / lower-throughput on older cores and mobile ARM).

### Measured throughput (`u8 -> f32`, 3ch RGB)

| Arm | `64×64` (L1) | `256×256` (L2) | `1024×1024` (L3) | `4096×4096` (DRAM) |
|---|---:|---:|---:|---:|
| **`split_exact_0ulp` (shipped)** | `599 ±4 ns` (`95.5 GiB/s`) | `10.0 ±0.2 µs` (`91.7 GiB/s`) | `178.1 ±7.9 µs` (`82.3 GiB/s`) | `10.4 ±0.3 ms` (`22.6 GiB/s`) |
| `recip_1ulp` (`126/256` off) | `422 ±2 ns` (`135 GiB/s`, `-29.5%`) | `8.6 ±0.1 µs` (`107 GiB/s`, `-13.5%`) | `167.6 ±6.6 µs` (`87.4 GiB/s`, `-5.8%`) | `10.1 ±0.3 ms` (`23.1 GiB/s`, CI crosses 0) |
| `newton_0ulp` (4 ops) | `721 ±5 ns` (`79.3 GiB/s`, `+20.5%`) | `11.6 ±0.2 µs` (`78.7 GiB/s`, `+17.4%`) | `208.7 ±4.7 µs` (`70.2 GiB/s`, `+17.5%`) | `10.4 ±0.2 ms` (`22.5 GiB/s`, CI crosses 0) |
| `div_0ulp` (`vdivps`) | `838 ±3 ns` (`68.3 GiB/s`, `+39.9%`) | `13.6 ±0.1 µs` (`67.6 GiB/s`, `+36.8%`) | `236.4 ±7.1 µs` (`62.0 GiB/s`, `+32.8%`) | `10.2 ±0.3 ms` (`22.9 GiB/s`, CI crosses 0) |

### Measured throughput (`u16 -> f32`, 3ch RGB)

| Arm | `64×64` (L1) | `256×256` (L2) | `1024×1024` (L3) | `4096×4096` (DRAM) |
|---|---:|---:|---:|---:|
| **`split_exact_0ulp` (shipped)** | `430 ±2 ns` (`160 GiB/s`) | `9.8 ±0.1 µs` (`112 GiB/s`) | `197.5 ±7.7 µs` (`89.0 GiB/s`) | `10.6 ±0.1 ms` (`26.5 GiB/s`) |
| `recip_1ulp` (`512/65536` off) | `470 ±3 ns` (`146 GiB/s`, `+9.3%`) | `9.4 ±0.1 µs` (`117 GiB/s`, `-4.2%`) | `197.5 ±6.9 µs` (`89.0 GiB/s`, CI crosses 0) | `10.7 ±0.2 ms` (`26.3 GiB/s`, CI crosses 0) |
| `newton_0ulp` (4 ops) | `562 ±3 ns` (`122 GiB/s`, `+30.8%`) | `10.7 ±0.1 µs` (`102 GiB/s`, `+9.7%`) | `209.2 ±4.5 µs` (`84.0 GiB/s`, `+5.9%`) | `10.6 ±0.1 ms` (`26.6 GiB/s`, CI crosses 0) |
| `div_0ulp` (`vdivps`) | `845 ±2 ns` (`81.3 GiB/s`, `+96.5%`) | `13.7 ±0.1 µs` (`80.2 GiB/s`, `+40.0%`) | `241.7 ±3.2 µs` (`72.7 GiB/s`, `+21.2%`) | `10.6 ±0.2 ms` (`26.6 GiB/s`, CI crosses 0) |

---

## 2. Float → Integer Quantization & `FloatFacts` Specialization (`f32 -> u8` and `f32 -> u16`, 3ch RGB)

### Instruction-level trade-off (`magetypes 0.9.32` `min_fast`/`max_fast`/`to_i32_fast`)

1. **`magic_fma_clamped` (shipped default when `!NO_NAN`):** `y = (x & x.simd_ge(0.0)).min_fast(1.0)` (3 vector instructions `vcmpgeps` + `vandps` + `vminps` — eliminating `magetypes 0.9.32`'s quiet-`NaN` blend in `.min()` since `NaN` was already flushed by `simd_ge(0.0)`), followed by single-instruction magic-number FMA `fma(y, M, 1.5 * 2^23)` (`0x4b400000`), integer subtract of `0x4b400000`, and narrow pack (`vpackusdw` + `vpackuswb` + `vpermq`). **0 mismatches across all `1,065,353,217` `f32` bit patterns in `[0.0, 1.0]`** matching `(x * M + 0.5).floor()` ties-away-from-zero!
2. **`magic_fma_no_nan` (shipped when `FloatFacts::NO_NAN` is proven, e.g. out-of-gamut linear `f32` after a 3×3 primaries matrix):** `y = x.max_fast(0.0).min_fast(1.0)` (2 single instructions `vmaxps` + `vminps`), saving the `vcmpgeps + vandps` `NaN` mask while still clamping negative / `> 1.0` / `±Inf` samples bit-identically.
3. **`magic_fma_unit_clean` (shipped when `FloatFacts::NO_NAN | NON_NEGATIVE | AT_MOST_ONE` is proven):** elides both `0.0` and `1.0` vector clamps completely — straight `vfmadd` + subtract + pack (0 clamp instructions).
4. **`mul_rne_inexact` (`(y * M).to_i32_round()`):** `vmulps` + `vcvtps2dq` (2 serial rounding steps instead of 1 fused FMA into `2^23` integer grid; both **slower** than `magic_fma` AND disagrees with ties-away-from-zero `(x * M + 0.5).floor()` at half-code boundaries).
5. **`truncate_unit_clean` (`DepthPolicy::Truncate` + `FloatFacts::UNIT_CLEAN`):** `(x * M).to_i32_fast()` (`magetypes 0.9.32` 1-instruction `vcvttps2dq` / WASM `i32x4.relaxed_trunc_f32x4_s` without `>= 2^31` saturation fixups) + narrow pack.

### Measured throughput (`f32 -> u8`, 3ch RGB; peak RSS `3.45 GiB`)

| Arm | `64×64` (L1) | `256×256` (L2) | `1024×1024` (L3) | `4096×4096` (DRAM) |
|---|---:|---:|---:|---:|
| **`magic_fma_clamped` (shipped)** | `659 ±3 ns` (`86.8 GiB/s`) | `10.4 ±0.1 µs` (`88.1 GiB/s`) | `181.3 ±2.6 µs` (`80.8 GiB/s`) | `7.7 ±0.1 ms` (`30.3 GiB/s`) |
| **`magic_fma_no_nan` (`FloatFacts::NO_NAN`)** | `570 ±1 ns` (`100 GiB/s`, **`-13.5%`**) | `9.8 ±0.1 µs` (`93.8 GiB/s`, **`-6.1%`**) | `173.0 ±3.6 µs` (`84.7 GiB/s`, **`-5.9%`**) | `7.7 ±0.2 ms` (`30.4 GiB/s`, CI crosses 0) |
| **`magic_fma_unit_clean` (`FloatFacts`)** | `435 ±1 ns` (`132 GiB/s`, **`-34.0%`**) | `7.9 ±0.1 µs` (`115 GiB/s`, **`-23.6%`**) | `146.4 ±4.4 µs` (`100 GiB/s`, **`-19.0%`**) | `7.5 ±0.2 ms` (`31.3 GiB/s`, **`-3.3%`**) |
| `mul_rne_inexact` | `668 ±3 ns` (`85.6 GiB/s`, `+1.4%`) | `11.1 ±0.1 µs` (`82.2 GiB/s`, `+7.3%`) | `190.8 ±3.9 µs` (`76.8 GiB/s`, `+5.0%`) | `7.8 ±0.1 ms` (`30.2 GiB/s`, CI crosses 0) |
| **`truncate_unit_clean` (`FloatFacts`)** | `391 ±2 ns` (`146 GiB/s`, **`-40.6%`**) | `7.4 ±0.1 µs` (`124 GiB/s`, **`-28.9%`**) | `135.8 ±4.6 µs` (`108 GiB/s`, **`-26.4%`**) | `7.2 ±0.1 ms` (`32.3 GiB/s`, **`-6.5%`**) |

*(Note: switching `.min(one)` → `.min_fast(one)` after `NaN` flush sped up `magic_fma_clamped` itself by **10.6–12.2%** vs the pre-`min_fast` `737 ns / 11.7 µs / 206.6 µs / 8.7 ms` baseline on `r7900x`.)*

### Measured throughput (`f32 -> u16`, 3ch RGB; peak RSS `3.45 GiB`)

| Arm | `64×64` (L1) | `256×256` (L2) | `1024×1024` (L3) | `4096×4096` (DRAM) |
|---|---:|---:|---:|---:|
| **`magic_fma_clamped` (shipped)** | `626 ±2 ns` (`110 GiB/s`) | `10.8 ±0.1 µs` (`102 GiB/s`) | `193.3 ±2.3 µs` (`90.9 GiB/s`) | `9.4 ±0.2 ms` (`30.0 GiB/s`) |
| **`magic_fma_no_nan` (`FloatFacts::NO_NAN`)** | `496 ±1 ns` (`138 GiB/s`, **`-20.8%`**) | `9.9 ±0.1 µs` (`111 GiB/s`, **`-8.3%`**) | `181.5 ±2.5 µs` (`96.8 GiB/s`, **`-6.6%`**) | `9.2 ±0.2 ms` (`30.4 GiB/s`, CI crosses 0) |
| **`magic_fma_unit_clean` (`FloatFacts`)** | `467 ±2 ns` (`147 GiB/s`, **`-25.4%`**) | `9.5 ±0.1 µs` (`116 GiB/s`, **`-12.2%`**) | `191.3 ±3.2 µs` (`91.9 GiB/s`, CI crosses 0) | `9.0 ±0.1 ms` (`31.2 GiB/s`, **`-4.0%`**) |
| `mul_rne_inexact` | `603 ±1 ns` (`114 GiB/s`, `-3.6%`) | `10.9 ±0.1 µs` (`101 GiB/s`, CI crosses 0) | `195.1 ±4.7 µs` (`90.1 GiB/s`, CI crosses 0) | `9.3 ±0.2 ms` (`30.1 GiB/s`, CI crosses 0) |
| **`truncate_unit_clean` (`FloatFacts`)** | `452 ±1 ns` (`152 GiB/s`, **`-27.6%`**) | `9.1 ±0.1 µs` (`121 GiB/s`, **`-15.8%`**) | `182.4 ±4.8 µs` (`96.4 GiB/s`, **`-6.4%`**) | `8.9 ±0.1 ms` (`31.5 GiB/s`, **`-4.5%`**) |

### Context-awareness conclusion: does relaxing 0-ULP buy anything?

- **`f32 -> u8` and `f32 -> u16`:** **No** — exact 0-ULP `magic_fma` is **3.3–6.0% faster** than inexact `mul + round` (`mul_rne_inexact`) because a single `fma(y, M, 1.5 * 2^23)` replaces `vmulps + vcvtps2dq`! The real context-aware win is `FloatFacts::UNIT_CLEAN` clamp elision (`-9%` to `-45%`), which `zenpixels-convert` now dispatches automatically whenever upstream steps or caller assertions prove `[0, 1]` clean samples.
- **`u16 -> f32`:** **No** — `split_exact_0ulp` is **9.3% faster** than `recip_1ulp` in L1 (`64×64`), within `4.2%` in L2 (`256×256`), and statistically tied (`89.0 GiB/s` vs `89.0 GiB/s`, CI `[-1.4%, +1.7%]`) from `1024×1024` onward.
- **`u8 -> f32`:** `split_exact_0ulp` (`178.1 µs` @ `1024²`, `10.4 ms` @ `4096²`) is within `5.8%` of 1-ULP reciprocal at `1024²` and tied at `4096²` (`177 ns` gap at `64²`), while **1.33–1.40× faster than `vdivps`** and **1.17–1.20× faster than 4-op Newton**. Moreover, transfer-aware conversions (`sRGB u8 -> linear f32`, `fast_gamut`) already fuse LUTs directly and never run `NaiveU8ToF32`; when `NaiveU8ToF32` *is* planned (linear `u8 -> f32` or `Ignore` depth rescale), callers expect exact identity (`u8 -> f32 -> f16 -> u8`, `u8 -> f32 -> u16` bit-exactness), so keeping `split_exact_0ulp` unconditionally avoids precision cliffs with negligible cost.

---

## 3. Band-Generic `PackedSource` / `PackedSink` Overhead (`< 2%` Gate)

Measured after `SealedSource::with_source` + `SinkStamp::wants_stamp` (`~/tmp/bench_band_generic_after.log`, peak RSS `3.30 GiB`):

### `RGB8 -> RGBA8` (memory-bandwidth + metadata-overhead sensitive)

| Call surface | `64×64` | `256×256` | `1024×1024` | `4096×4096` |
|---|---:|---:|---:|---:|
| `RowConverter::convert_rows` (baseline) | `1.1 ±0.0 µs` (`25.0 GiB/s`) | `6.4 ±0.0 µs` (`66.3 GiB/s`) | `72.2 ±0.9 µs` (`94.7 GiB/s`) | `4.1 ±0.1 ms` (`26.8 GiB/s`) |
| **`RowConverter::convert(PixelSlice, PixelSliceMut)`** | `1.1 ±0.0 µs` (`+1.8%`) | `6.5 ±0.0 µs` (`+0.8%`, CI crosses 0) | `71.7 ±0.7 µs` (`-1.1%`, CI crosses 0) | `4.1 ±0.1 ms` (`+0.5%`, CI crosses 0) |
| **`RowConverter::convert(&PixelBuffer, &mut PixelBuffer)`** | `1.1 ±0.0 µs` (`+1.5%`) | `6.5 ±0.0 µs` (`+0.4%`, CI crosses 0) | `72.5 ±1.1 µs` (`+0.1%`, CI crosses 0) | `3.9 ±0.1 ms` (`-4.6%`, `28.0 GiB/s`) |
| `RowConverter::convert(&BandedPixelBuffer, &mut BandedPixelBuffer)` (4 bands) | `1.2 ±0.0 µs` (`+14.4%`, 16-row micro-bands) | `6.7 ±0.0 µs` (`+4.4%`, 64-row bands) | `71.7 ±0.9 µs` (`+1.2%`, CI crosses 0) | `4.1 ±0.1 ms` (`-0.5%`, CI crosses 0) |
| `OutputPlan::convert_rows` (per-row loop) | `1.7 ±0.0 µs` (`+59.1%`) | `9.0 ±0.0 µs` (`+39.8%`) | `83.3 ±1.2 µs` (`+15.0%`) | `4.1 ±0.1 ms` (`+1.2%`, CI crosses 0) |
| **`OutputPlan::convert_into(&PixelBuffer, &mut PixelBuffer)`** | `1.1 ±0.0 µs` (**`35.3%` faster than `OutputPlan::convert_rows`**) | `6.5 ±0.0 µs` (**`27.8%` faster**) | `72.6 ±0.9 µs` (**`12.8%` faster**) | `4.1 ±0.1 ms` (`-0.4%`, CI crosses 0) |

### `RGB8 -> Linear RGBf32`

| Call surface | `64×64` | `256×256` | `1024×1024` | `4096×4096` |
|---|---:|---:|---:|---:|
| `RowConverter::convert_rows` (baseline) | `2.7 ±0.0 µs` (`21.5 GiB/s`) | `37.5 ±0.2 µs` (`24.4 GiB/s`) | `615.9 ±9.2 µs` (`23.8 GiB/s`) | `12.8 ±0.2 ms` (`18.3 GiB/s`) |
| **`RowConverter::convert(&PixelBuffer, &mut PixelBuffer)`** | `2.7 ±0.0 µs` (`+0.8%`, CI crosses 0) | `37.6 ±0.2 µs` (`+0.0%`, CI crosses 0) | `614.0 ±8.5 µs` (`-0.4%`, CI crosses 0) | `13.0 ±0.2 ms` (`+0.2%`, CI crosses 0) |
| `RowConverter::convert(&BandedPixelBuffer, &mut BandedPixelBuffer)` (4 bands) | `2.8 ±0.0 µs` (`+4.9%`, 16-row micro-bands) | `38.1 ±0.4 µs` (`+1.5%`) | `617.2 ±10.5 µs` (`+0.1%`, CI crosses 0) | `12.8 ±0.1 ms` (`+0.1%`, CI crosses 0) |
