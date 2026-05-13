# fast_gamut_v2 — fresh bench on HEAD (post-garb-0.2.8 migration)

## What this run measures

`bench_t7_gamut` on commit `<HEAD>` after the garb 0.2.8 `_scalar` migration
(see `fast_gamut_v2.rs` calling `garb::deinterleave::*_chunk*_to_planes_scalar`
instead of the now-deleted `_v3 / _neon / _wasm128` hand-written chunk SIMD).

## What this run does NOT measure

This is a **single-axis throughput snapshot of the v2 path on HEAD**.
It is not a v1↔v2 paired comparison — v1 (`stamp_trc_kernels!`) was
deleted in this PR. The original `PAIRED_COMPARISON.md` numbers were
measured against:

  1. v1 (deleted in this PR), and
  2. garb 0.2.7's hand-written 128-bit-XMM f32 chunk SIMD
     (`_v3 / _neon / _wasm128` — never published; dropped from garb
     0.2.8 because LLVM autovec under target_feature avx2,fma was
     +26-37% faster at 1024px per `imazen/garb`'s own
     `benchmarks/deinterleave_autovec_vs_chunk_2026-05-07`).

Neither baseline exists on HEAD. The paired-comparison logs are
**historical** — useful for understanding the design journey, not for
asserting current-HEAD perf. See SUPERSEDED banners on COMPARISON.md
and PAIRED_COMPARISON.md.

## Sanity check this run does provide

- v2 + garb-0.2.8 still hits expected throughput buckets at 256px /
  4096px / 1080p (no obvious regression from the migration).
- The `convert_*_linear_v2` scalar mat3x3 paths and the const-generic
  SIMD body for Srgb→Srgb both produce within-bench-noise numbers.

## Hardware / harness

  host:      AMD Ryzen 9 7950X (Zen 4, water-cooled, 128 GB DDR5)
  cmd:       cargo bench --bench bench_t7_gamut -p zenpixels-convert
  zenbench:  whatever's pinned in workspace
  date:      2026-05-13 UTC

## Headline numbers (from log)

  sRGB U8 fused gamut (P3→BT.709)    256px →  227ns
                                    4096px → 3.36µs
                                    1080p  → 1.79ms
  sRGB U16 fused gamut (P3→BT.709)   256px →  485ns  
                                    4096px → 8.0µs
                                    1080p  → 4.1ms
  sRGB U8 → Linear F32 + gamut       256px →  332ns
                                    4096px → 4.9µs
                                    1080p  → 4.1ms
  Linear F32 → sRGB U8 + gamut       256px →  574ns
                                    4096px → 8.9µs
                                    1080p  → 4.7ms
