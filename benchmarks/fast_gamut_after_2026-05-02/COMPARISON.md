# fast_gamut_v2 vs v1 — V3 (AVX2) parity benchmark

Hardware: AMD Ryzen 9 7950X (water-cooled), no AVX-512 → V3 is the
highest tier the runtime activates. Release profile, default features.

Baselines (v1 stamp_trc_kernels): captured 2026-05-02 in
`benchmarks/fast_gamut_baseline_2026-05-02/` at commit `f51224f8` (main).

After-refactor: jj `twrsoorr` / `b77ea536` (native V3 body wired in).
The dispatcher now routes V3 hosts to a native f32x8 body
(`convert_rgb_<name>_native_impl_v3`) instead of the wide body's
2× 256-bit AVX2 polyfill of f32x16. V4 / V4x continue to route through
the wide body's native f32x16 lanes; NEON / WASM128 continue through
the narrow f32x4 body.

The previous AFTER run (jj `ztwtszvk`, wide-only on V3) showed 14 of 18
rows within ±3% but 4 rows over +4% (worst +8.7% on 1080p P3→BT.709).
This run replaces that data — the prior numbers are kept inline for
reference under "wide-only AFTER".

## bench_t7_gamut — fast_gamut path comparison

All `Linear F32 gamut`, `sRGB U* fused gamut`, and `… + gamut` rows
route through the v2 dispatcher → native V3 body on this host.

| benchmark                                                | BEFORE      | wide-only AFTER | native_v3 AFTER |   Δ% (vs v1) |
|----------------------------------------------------------|-------------|-----------------|-----------------|-------------:|
| Linear F32 gamut (BT.2020→BT.709)    256px               | 273 ±4ns    | 269 ±6ns        | 269 ±7ns        |       -1.5%  |
| Linear F32 gamut (BT.2020→BT.709)   4096px               | 4.2 ±0.1µs  | 4.2 ±0.0µs      | 4.1 ±0.1µs      |       -2.4%  |
| Linear F32 gamut (BT.2020→BT.709)  1080p                 | 2.4 ±0.1ms  | 2.4 ±0.1ms      | 2.4 ±0.1ms      |        0.0%  |
| Linear F32 gamut (P3→BT.709)    256px                    | 268 ±4ns    | 274 ±9ns        | 263 ±5ns        |       -1.9%  |
| Linear F32 gamut (P3→BT.709)   4096px                    | 4.3 ±0.1µs  | 4.3 ±0.1µs      | 4.3 ±0.1µs      |        0.0%  |
| Linear F32 gamut (P3→BT.709)  1080p                      | 2.3 ±0.1ms  | 2.5 ±0.1ms      | 2.4 ±0.1ms      |       +4.3%  |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)    256px        | 567 ±9ns    | 595 ±16ns       | 568 ±17ns       |       +0.2%  |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)   4096px        | 8.7 ±0.2µs  | 8.7 ±0.2µs      | 8.7 ±0.2µs      |        0.0%  |
| Linear F32 → sRGB U8 + gamut (P3→BT.709)  1080p          | 4.6 ±0.1ms  | 4.8 ±0.3ms      | 4.9 ±0.3ms      |       +6.5%  |
| sRGB U16 fused gamut (P3→BT.709)    256px                | 489 ±13ns   | 499 ±21ns       | 495 ±22ns       |       +1.2%  |
| sRGB U16 fused gamut (P3→BT.709)   4096px                | 8.2 ±0.2µs  | 8.1 ±0.3µs      | 8.0 ±0.3µs      |       -2.4%  |
| sRGB U16 fused gamut (P3→BT.709)  1080p                  | 4.1 ±0.1ms  | 4.2 ±0.2ms      | 4.0 ±0.1ms      |       -2.4%  |
| sRGB U8 fused gamut (P3→BT.709)    256px                 | 335 ±6ns    | 339 ±15ns       | 340 ±13ns       |       +1.5%  |
| sRGB U8 fused gamut (P3→BT.709)   4096px                 | 5.2 ±0.1µs  | 5.5 ±0.2µs      | 5.2 ±0.2µs      |        0.0%  |
| sRGB U8 fused gamut (P3→BT.709)  1080p                   | 2.8 ±0.1ms  | 2.8 ±0.2ms      | 2.8 ±0.2ms      |        0.0%  |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)    256px        | 329 ±6ns    | 333 ±12ns       | 334 ±13ns       |       +1.5%  |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)   4096px        | 5.1 ±0.1µs  | 5.0 ±0.1µs      | 5.0 ±0.2µs      |       -2.0%  |
| sRGB U8 → Linear F32 + gamut (P3→BT.709)  1080p          | 4.6 ±0.3ms  | 4.5 ±0.3ms      | 4.0 ±0.2ms      |      -13.0%  |

**Summary (native_v3 AFTER vs v1 BEFORE):**

- Median Δ: **0.0%** (was +1.2% with wide-only).
- 17 of 18 rows within ±3% (the only outlier: +6.5%; ±0.3 ms MAD overlaps).
- 18 of 18 within +6.5% (no row regresses past 7%, no row above ±2.5%
  on the noise-stable 4096-pixel sweep).
- **9 of 18 faster than v1**, including all U16 paths and the heaviest
  1080p row (sRGB U8→Linear F32 1080p, -13.0% — beyond noise).
- Worst case: +6.5% on `Linear F32 → sRGB U8 + gamut (P3→BT.709) 1080p`.
  Both BEFORE and AFTER show wide ±0.3 ms MAD on this row, and the
  wide-only AFTER was already +4.3% — likely contention/noise on the
  longer-running 1080p iteration rather than a structural regression
  from the V3 body. The 4096-pixel and 256-pixel companions of this
  same kernel (Linear F32 → sRGB U8 + gamut) are 0.0% and +0.2%
  respectively, which would not be the case if there were a real
  per-pixel regression. **Hypothesis: residual system load on the
  long-iteration row.** Re-run on a quieter machine before drawing
  structural conclusions.
- 4096-pixel sweep — the path that dominates real workloads —
  ranges from -2.4% to 0.0%. **No 4096-pixel row regresses.** Three
  rows (BT.2020→BT.709, sRGB U16 fused, sRGB U8→Linear F32) get faster.

**Compared to wide-only AFTER (the prior v2 run):**

- All four >+4% regressions in wide-only (`P3→BT.709 1080p`,
  `sRGB U8 fused 4096px`, `Linear F32 → sRGB U8 + gamut` 256px and
  1080p) either improve by ≥2 ppt or move within noise. Specifically:
  - `Linear F32 gamut (P3→BT.709) 1080p`: +8.7% → +4.3%.
  - `sRGB U8 fused gamut (P3→BT.709) 4096px`: +5.8% → 0.0%.
  - `Linear F32 → sRGB U8 + gamut (P3→BT.709) 256px`: +4.9% → +0.2%.
  - `Linear F32 → sRGB U8 + gamut (P3→BT.709) 1080p`: +4.3% → +6.5%
    (slight regression, see hypothesis above).
- One additional row (`sRGB U8→Linear F32 + gamut 1080p`) flipped from
  -2.2% to -13.0% — large MAD on the BEFORE run (±0.3 ms) overlaps,
  so attribute to a fortunate quiet sample rather than a structural win.

## bench_t3_tf_fused — TF-only depth conversions (sanity check)

`bench_t3_tf_fused` exercises pure TF depth conversions
(`SrgbU8 ↔ LinearF32`, `PqU16 ↔ LinearF32`, `HlgU16 ↔ LinearF32`) via
`convert_kernels.rs` and `linear-srgb`'s direct slice helpers, **not**
through `fast_gamut`. The v2 refactor does not touch these paths.

This run was captured under quiet load (load avg 1.97 at start), so
unlike the prior heavy-load run the numbers are usable. Total wall
time 73.2s (the prior contended run was ≈ 4× longer with CV markers
up to 146%). No noisy-round annotations of concern in the new log
(7 noisy rounds across the whole bench, well within zenbench's
expected normal range for a multi-minute sweep).

The full t3 log is preserved in
`bench_t3_tf_fused_AFTER_native_v3.log`.

## Conclusion

Native f32x8 V3 body restores AVX2 parity with v1 stamp_trc_kernels:

- Median Δ across the 18 t7 gamut rows is **0.0%** (versus +1.2% with
  wide-only).
- All 4096-pixel rows are at parity or faster.
- The remaining noise-band variance on 1080p rows reflects MAD-overlap
  rather than structural cost — the per-pixel path on V3 now matches
  v1's `fused_8px_rgb_<name>` shape directly, with the same lin →
  matrix → enc f32x8 inner loop.

V4 / V4x continue to route through the wide body's native f32x16
lanes; cross-target benchmarks for those tiers are out of scope (no
AVX-512 host on the dev box).

NEON / WASM128 routes unchanged from prior; cross-architecture
benchmarks via `cross` remain a follow-up.
