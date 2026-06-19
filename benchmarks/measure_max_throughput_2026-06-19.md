# `ContentLightLevel::measure_max` throughput — SOTA spec-conformant CLL

**Date:** 2026-06-19
**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads, AVX-512, DDR5)
**Build:** `cargo run --example measure_histogram_throughput --release`
**Source:** `zenpixels/examples/measure_histogram_throughput.rs`

`measure_max` is the **spec-conformant** CLL reading (CTA-861.3 strict — literal MaxCLL + arithmetic MaxFALL, no percentile). The dominant production case for HDR-aware encoders, and the one that runs on every frame of every encode. We re-routed it off the histogram path so it does only the work that CTA-861.3 requires: per-pixel `max(R, G, B)` (or BT.2020 luma) + running max + f64 sum, with no histogram store. Reaches **SOTA throughput** — above the documented ~1-2 Gpix/s of `libplacebo`'s analogous path.

The histogram path stays for `measure_percentile` / `measure_histogram` callers who actually need the bin distribution.

## Results

Pixel source: deterministic per-pixel ramp + sparkle (1024² → 8K). Warmup × 3 before each timed run.

### `RUSTFLAGS="-C target-cpu=native"` (real AVX2 / AVX-512 intrinsics on this CPU)

| Image size | measure_max scalar | measure_max SIMD | measure_histogram SIMD | measure_max / measure_histogram |
|---|---|---|---|---|
| 256² thumb | 1717 Mpix/s | **3439 Mpix/s** | 489 Mpix/s | 7.03 × |
| 1024² HD | 1740 Mpix/s | **3447 Mpix/s** | 490 Mpix/s | 7.04 × |
| 2048² 4 MP | 1717 Mpix/s | **2906 Mpix/s** | 472 Mpix/s | 6.16 × |
| 3840×2160 4K | 1719 Mpix/s | **2766 Mpix/s** | 460 Mpix/s | 6.02 × |
| 7680×4320 8K | 1713 Mpix/s | **2558 Mpix/s** | 481 Mpix/s | 5.31 × |

### Default `cargo run --release` (no `target-cpu=native`, baseline v2)

| Image size | measure_max SIMD | measure_histogram SIMD | measure_max / measure_histogram |
|---|---|---|---|
| 256² | 3360 Mpix/s | 320 Mpix/s | 10.5 × |
| 1024² | 3328 Mpix/s | 328 Mpix/s | 10.1 × |
| 2048² | 2735 Mpix/s | 312 Mpix/s | 8.8 × |
| 3840×2160 | 2624 Mpix/s | 254 Mpix/s | 10.3 × |
| 7680×4320 | 2606 Mpix/s | 317 Mpix/s | 8.2 × |

## What we hit

**`measure_max` is gigapixel-class on both scalar and SIMD paths**, and 5–10× faster than the histogram path:

- **Scalar measure_max: 1.7 Gpix/s** — LLVM auto-vectorises the simple max+sum loop given the fixed-size-array pattern, so even users on the foundational no-`simd` build path are at gigapixel speed.
- **SIMD measure_max: 2.5–3.4 Gpix/s** — adds another 1.5–2 × over scalar via the tier-dispatched `f32x8` kernel.
- **Above libplacebo** (~1–2 Gpix/s documented) — SOTA territory for CTA-861.3 CLL analysis in the workspace.

The throughput drops at 4K → 8K because we're hitting the DDR5 memory wall: 3 Gpix/s × 12 B/px = **36 GB/s sequential read**, against ~40 GB/s sustained on this DDR5-5200 configuration. The kernel is now memory-bound at large sizes — there's no further compute optimisation that meaningfully helps without changing the input encoding (e.g. PQ16 instead of RGBF32).

## Comparison to `measure_histogram`

| Path | Why it's slower | Use it when |
|---|---|---|
| `measure_max` | — | CTA-861.3 spec-strict delivery (Netflix, broadcast). The hot path for HDR-aware encoders. |
| `measure_percentile` | Builds 1024-bin log histogram, then walks the CDF for one percentile. | Defect-rejection at p99 / p99.9 / p99.99, or any content policy that needs a percentile MaxCLL. |
| `measure_histogram` | Builds 1024-bin log histogram. Caller does multiple readouts. | Want both literal max AND percentile, or want to plot / inspect the distribution. |

The 5–10 × gap between `measure_max` and `measure_histogram` is the cost of the histogram scatter (8 scalar increments per SIMD chunk, each with a load-add-store dependency on the previous iteration). That step is structurally necessary for histogram-based readouts and out of scope here.

## Reproduce

```bash
cargo run --example measure_histogram_throughput --features simd --release
RUSTFLAGS="-C target-cpu=native" cargo run --example measure_histogram_throughput --features simd --release
```

Output is one line per image size, two throughputs (measure_max + measure_histogram) and the ratio.
