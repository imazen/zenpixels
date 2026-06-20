# `ContentLightLevel::measure_max_smoothed` throughput

**Date:** 2026-06-19
**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads, AVX-512)
**Build:** `RUSTFLAGS="-C target-cpu=native" cargo run -p zenpixels-convert --example measure_histogram_throughput --release`
**Source:** `zenpixels-convert/examples/measure_histogram_throughput.rs`

`measure_max_smoothed` applies a 3×1 horizontal box filter to the per-pixel
`max(R, G, B)` (or BT.2020 luma) reduction before tracking the running max.
Single-pixel defects average with their cold neighbours and drop below the
actual content peak; legitimate features that span ≥3 horizontal pixels are
preserved at full magnitude. MaxFALL is the literal arithmetic mean of the
unsmoothed input (linearity of expectation; CTA-861.3 spec-literal regardless).

Implementation is a scalar single-pass streaming kernel — `prev / curr / next`
sliding window over `m_raw`, no row scratch buffer, no allocation. LLVM
auto-vectorises the short data dependency chain (4-5 ops/pixel) into a tight
inner loop.

## Results

| Path | 256² | 1024² | 2048² (4 MP) | 3840×2160 (4K UHD) | 7680×4320 (8K) |
|------|------|-------|--------------|--------------------|----------------|
| `measure_max` (spec-strict, SIMD baseline) | 3 431 | 3 313 | 2 508 | 2 033 | 2 644 Mpix/s |
| **`measure_max_smoothed` (3×1 box, scalar)** | **1 285** | **1 295** | **1 039** | **1 277** | **1 133 Mpix/s** |
| `measure_histogram` (full histogram + bins) | 477 | 478 | 471 | 305 | 417 Mpix/s |
| smoothed / measure_max ratio | 0.37 × | 0.39 × | 0.41 × | 0.63 × | 0.43 × |

The 3×1 box-filtered scan stays above 1 Gpix/s at every realistic image size
and ~2-3× faster than the histogram path. The SIMD measure_max baseline is
hand-vectorised with archmage/magetypes and reflects what's possible with
no cross-pixel data dependency; the smoothed kernel's left/right neighbour
dependency is what keeps it ~40-60% of that ceiling under auto-vectorisation.

## Why no `archmage::magetypes` SIMD path here

`magetypes` `f32x8` exposes arithmetic, load, store, splat, zero, and reductions
— no lane-shift / `vpalignr` / `vext` / generic shuffle. The natural SIMD design
for a 3×1 box (deinterleave RGB → `f32x8` `m_curr`, store back to a 9-element
f32 array with a 1-pixel scalar peek-ahead, build shifted-left and shifted-right
arrays via copy, reload as `f32x8`, sum the three) measured **1.0 Gpix/s** — a
~20 % regression vs the auto-vectorised scalar. The store→load forwarding round
trip dominates the per-iter cost: the streaming dependency chain has no work the
SIMD can amortise away.

A two-pass design — pass 1 (existing SIMD `scan_row_max_rgb_tier`-style)
deinterleaves + reduces into a per-row scratch `[f32]` while accumulating the
unsmoothed sum, pass 2 streams the 3-tap box-max over the scratch — would
trade one extra row-scratch round trip (4 B/pixel) for losing the store→load
chain. Memory bandwidth budget: 12 (RGB) + 4 (scratch write) + 4 (scratch
read) = 20 B/pixel = ~3.0 Gpix/s ceiling at DDR5 60 GB/s. That's the next
SIMD step if the scalar 1.3 Gpix/s here turns out to be too slow for a real
use case. Filed as task #45-adjacent future work.

## Reproduce

```bash
RUSTFLAGS="-C target-cpu=native" cargo run \
    --example measure_histogram_throughput \
    --release --features simd
```

Output format: per-row label, dimensions, iteration count, then Mpix/s for
each of the three readouts (`measure_max`, `measure_max_smoothed`,
`measure_histogram`) plus the smoothed/max ratio.
