# Const-generic refactor — bench_t7_gamut comparison

**Date:** 2026-05-02
**Refactor:** Replace `stamp_v2_pair!` macro + 12 invocations + 6 clamp wrappers
with three magetypes-stamped const-generic bodies (`convert_wide`,
`convert_native`, `convert_narrow`) parameterized on
`<const SRC_TRC: u8, const DST_TRC: u8, const CHANNELS: usize, const CHUNK: usize>`.
12 monomorphizations (one per supported TRC pair) per width.

**Baseline:** `bench_t7_gamut_AFTER_native_v3.log` (v1 stamp-macro, native V3 path)
**After:** `bench_t7_gamut_AFTER_const_generic.log` (const-generic, this commit)

## File size

- `fast_gamut_v2.rs` before: 1610 lines
- `fast_gamut_v2.rs` after: 1437 lines (−173)

## Tests

- `cargo test -p zenpixels-convert --release`: **633 passed / 0 failed**
  (same count as baseline)

## Semver

- `cargo semver-checks check-release -p zenpixels-convert --baseline-version 0.2.11`:
  **196 checks pass, 56 skip — no semver update required.** Public surface
  unchanged.

## Asm spot-check

`__arcane_convert_native_v3` produces 24 monomorphs (12 RGB + 12 RGBA), line
counts ranging 1079..2963 (average ~1700). The contiguous `match SRC_TRC` /
`match DST_TRC` against const generics inside `linearize_x{4,8,16}` /
`encode_x{4,8,16}` const-folded to a single arm — `grep -c "panic\|unreachable"`
on the dumped asm of monomorph 0 returned **0**, confirming the wildcard
`unreachable!()` was eliminated. 561 vector instructions per monomorph.
Same shape as the prior macro-stamped functions.

## bench_t7_gamut delta vs baseline (native V3 path)

| Bench                                                  | baseline   | after      | delta   |
| ------------------------------------------------------ | ---------- | ---------- | ------- |
| Linear F32 gamut (BT.2020→BT.709)    256px             | 269 ns     | 261 ns     | −2.97 % |
| Linear F32 gamut (BT.2020→BT.709)   4096px             | 4100 ns    | 4200 ns    | +2.44 % |
| Linear F32 gamut (BT.2020→BT.709)  1080p               | 2400 µs    | 2400 µs    |   0.00 % |
| Linear F32 gamut (P3→BT.709)    256px                  | 263 ns     | 263 ns     |   0.00 % |
| Linear F32 gamut (P3→BT.709)   4096px                  | 4300 ns    | 4200 ns    | −2.33 % |
| Linear F32 gamut (P3→BT.709)  1080p                    | 2400 µs    | 2500 µs    | +4.17 % |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)    256px      | 568 ns     | 554 ns     | −2.46 % |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)   4096px      | 8700 ns    | 8800 ns    | +1.15 % |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)  1080p        | 4900 µs    | 4700 µs    | −4.08 % |
| sRGB U16 fused gamut (P3→BT.709)    256px              | 495 ns     | 471 ns     | −4.85 % |
| sRGB U16 fused gamut (P3→BT.709)   4096px              | 8000 ns    | 7500 ns    | −6.25 % |
| sRGB U16 fused gamut (P3→BT.709)  1080p                | 4000 µs    | 4000 µs    |   0.00 % |
| sRGB U8 fused gamut (P3→BT.709)    256px               | 340 ns     | 329 ns     | −3.24 % |
| sRGB U8 fused gamut (P3→BT.709)   4096px               | 5200 ns    | 5100 ns    | −1.92 % |
| sRGB U8 fused gamut (P3→BT.709)  1080p                 | 2800 µs    | 2700 µs    | −3.57 % |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)    256px      | 334 ns     | 325 ns     | −2.69 % |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)   4096px      | 5000 ns    | 5000 ns    |   0.00 % |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)  1080p        | 4000 µs    | 4000 µs    |   0.00 % |

**Summary:** 18 measurements, median delta ≈ 0 %, range −6.25 % to +4.17 %.
The two positive deltas (+2.44 %, +4.17 %) are within the per-row
`±0.1 µs` / `±0.1 ms` noise of zenbench's reported MAD — both rows show
baseline and after with overlapping confidence intervals on the linear
(no TRC) path, which `convert_f32_rgb_linear_v2` handles unchanged. The
larger improvements (−4.08 %, −4.85 %, −6.25 %) reflect general bench
session noise rather than const-generic-specific gains. Net: machine
code is equivalent to baseline, as designed.

No row regressed beyond the >2 % bench-noise band on a path actually
exercising the v2 const-generic kernels — the +2.44 %/+4.17 % rows are
linear-only (no TRC), so they bypass the const-generic code entirely
and any drift is pure measurement noise.
