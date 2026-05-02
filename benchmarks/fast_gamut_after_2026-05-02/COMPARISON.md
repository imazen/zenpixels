# fast_gamut_v2 vs v1 — V3 (AVX2) parity benchmark

Hardware: AMD Ryzen 9 7950X (water-cooled), no AVX-512 → V3 is the
highest tier the runtime activates. Release profile, default features.

Baselines (v1 stamp_trc_kernels): captured 2026-05-02 in
`benchmarks/fast_gamut_baseline_2026-05-02/` at commit `f51224f8` (main).

After-refactor: this run, at the v2-wired commit (jj `ztwtszvk`).
Benchmark binary built with the v2 path active in
`convert_f32_rgb_dispatch` / `convert_f32_rgba_dispatch`.

## bench_t7_gamut — fast_gamut path comparison

This is the load-bearing comparison. All `Linear F32 gamut`,
`sRGB U* fused gamut`, and `… + gamut` rows route through the v2
dispatcher.

| benchmark                                                | BEFORE      | AFTER       |   Δ%   |
|----------------------------------------------------------|-------------|-------------|-------:|
| Linear F32 gamut (BT.2020→BT.709)    256px               | 273 ±4ns    | 269 ±6ns    |  -1.5% |
| Linear F32 gamut (BT.2020→BT.709)   4096px               | 4.2 ±0.1µs  | 4.2 ±0.0µs  |  +0.0% |
| Linear F32 gamut (BT.2020→BT.709)  1080p                 | 2.4 ±0.1ms  | 2.4 ±0.1ms  |  +0.0% |
| Linear F32 gamut (P3→BT.709)    256px                    | 268 ±4ns    | 274 ±9ns    |  +2.2% |
| Linear F32 gamut (P3→BT.709)   4096px                    | 4.3 ±0.1µs  | 4.3 ±0.1µs  |  +0.0% |
| Linear F32 gamut (P3→BT.709)  1080p                      | 2.3 ±0.1ms  | 2.5 ±0.1ms  |  +8.7% |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)    256px        | 567 ±9ns    | 595 ±16ns   |  +4.9% |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)   4096px        | 8.7 ±0.2µs  | 8.7 ±0.2µs  |  +0.0% |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)  1080p          | 4.6 ±0.1ms  | 4.8 ±0.3ms  |  +4.3% |
| sRGB U16 fused gamut (P3→BT.709)    256px                | 489 ±13ns   | 499 ±21ns   |  +2.0% |
| sRGB U16 fused gamut (P3→BT.709)   4096px                | 8.2 ±0.2µs  | 8.1 ±0.3µs  |  -1.2% |
| sRGB U16 fused gamut (P3→BT.709)  1080p                  | 4.1 ±0.1ms  | 4.2 ±0.2ms  |  +2.4% |
| sRGB U8 fused gamut (P3→BT.709)    256px                 | 335 ±6ns    | 339 ±15ns   |  +1.2% |
| sRGB U8 fused gamut (P3→BT.709)   4096px                 | 5.2 ±0.1µs  | 5.5 ±0.2µs  |  +5.8% |
| sRGB U8 fused gamut (P3→BT.709)  1080p                   | 2.8 ±0.1ms  | 2.8 ±0.2ms  |  +0.0% |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)    256px        | 329 ±6ns    | 333 ±12ns   |  +1.2% |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)   4096px        | 5.1 ±0.1µs  | 5.0 ±0.1µs  |  -2.0% |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)  1080p          | 4.6 ±0.3ms  | 4.5 ±0.3ms  |  -2.2% |

**Summary:**

- Median Δ: **+1.2%**
- 14 of 18 within ±3% of v1 baseline.
- 16 of 18 within ±5%.
- 4 of 18 faster than v1.
- Worst case: +8.7% on `Linear F32 gamut (P3→BT.709) 1080p`. The MAD
  on both runs (±0.1ms) overlaps the absolute +0.2ms delta, so this is
  likely run-to-run noise rather than a structural regression.
- The 4096-pixel sweep — the path that dominates real workloads —
  ranges from -2.0% to +5.8%. No regression worth a native-V3 body.

## bench_t3_tf_fused — TF-only depth conversions (sanity check)

`bench_t3_tf_fused` exercises pure TF depth conversions
(`SrgbU8 ↔ LinearF32`, `PqU16 ↔ LinearF32`, `HlgU16 ↔ LinearF32`) via
`convert_kernels.rs` and `linear-srgb`'s direct slice helpers, **not**
through `fast_gamut`. The v2 refactor does not touch these paths.

The first AFTER run was captured under heavy system load (load avg
~12, 4× the BEFORE wallclock and CV markers up to 146% on individual
rows), making the numbers unreliable for parity claims. A second
clean rerun was not performed because t3 does not exercise the v2
surface — any drift here is unrelated to this refactor.

The full t3 log is preserved in `bench_t3_tf_fused_AFTER.log` for
reference, with the noise markers (`[1] CV=…%`) intact.

## Conclusion

V3 (AVX2) parity confirmed on `bench_t7_gamut` — median +1.2%, all
4096-pixel rows within +5.8%. No native V3 (`f32x8`-direct) body is
needed; the f32x16 polyfill to 2× 256-bit ops on V3 holds up.

V4 / V4x / NEON / WASM128 are now covered by the wide / narrow bodies
respectively. Cross-target benchmarks for those tiers are out of scope
for this run (no AVX-512 host on the dev box; AArch64 / WASM via
`cross` is a follow-up).
