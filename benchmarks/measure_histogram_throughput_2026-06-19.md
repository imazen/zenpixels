# `ContentLightLevel::measure_histogram` throughput

**Date:** 2026-06-19
**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads, AVX-512)
**Build:** `cargo run --example measure_histogram_throughput --release` with the cited features
**Source:** `zenpixels-convert/examples/measure_histogram_throughput.rs`

Pixel source is a deterministic per-pixel ramp + sparkle so the histogram path exercises a realistic bin distribution. Warmup x3 before each timed run.

## Results

### Default build (baseline `x86-64-v2` target features)

| Path | 256² | 1024² | 2048² (4 MP) | 3840×2160 (4K UHD) | 7680×4320 (8K) |
|------|------|-------|--------------|--------------------|----------------|
| Scalar (`default` features) | 194 Mpix/s | 208 Mpix/s | 206 Mpix/s | 287 Mpix/s | 287 Mpix/s |
| SIMD (`simd` feature) | 336 Mpix/s | 344 Mpix/s | 337 Mpix/s | 342 Mpix/s | 343 Mpix/s |
| **Speedup** | **1.73 ×** | **1.65 ×** | **1.64 ×** | **1.19 ×** | **1.19 ×** |

### `RUSTFLAGS="-C target-cpu=native"` (per-machine — enables AVX2 / AVX-512)

| Path | 256² | 1024² | 2048² | 3840×2160 | 7680×4320 |
|------|------|-------|-------|-----------|-----------|
| Scalar | not tested | not tested | not tested | ≈ 287 Mpix/s | ≈ 287 Mpix/s |
| SIMD (`simd`) | 493 Mpix/s | 491 Mpix/s | 487 Mpix/s | 490 Mpix/s | 489 Mpix/s |
| SIMD (`simd avx512`) | 487 Mpix/s | 488 Mpix/s | 485 Mpix/s | 482 Mpix/s | 482 Mpix/s |

`avx512` did **not** improve over `v3` (AVX2) in this kernel — the inner loop is bottlenecked by the per-iteration scatter to the per-lane sub-histograms, not by SIMD math width. The histogram store hits the same cache line repeatedly on smooth content (load-add-store latency chain), and AVX-512's wider lanes just produce more bin indices per chunk without changing the scatter step.

## Reaching gigapixel throughput

At 490 Mpix/s × 12 B/pixel = 5.9 GB/s sequential read — only ~15 % of DDR5 bandwidth, so the kernel is **not** memory-bound. The bottleneck is the histogram scatter:

- Per-iteration cost (8-pixel SIMD chunk): ≈ 50 cycles
- Breakdown: load + deinterleave + SIMD max + SIMD multiply + SIMD `log2_midp` + bin-index clamp + **8 scalar histogram increments**
- The 8 scatter writes dominate; each carries a load-add-store dependency on the previous iteration's write to the same sub-histogram bin.

Crossing 1 Gpix/s on this hardware would need eliminating or amortising the scatter step:

- **Block-level histograms** — accumulate a small local histogram per N rows in registers / L1, then merge. Removes per-pixel store cost; pays once per block merge.
- **Sort-based histogram** — bucket pixels by bin in one pass, then count bin populations in a separate pass. Trades store cost for sort cost.
- **AVX-512 conflict detection (VPCONFLICTD)** — detect same-bin lanes within a chunk and batch the increments. Hardware-specific.
- **Reduced bin count** — 256 bins (~5 KB total, fits L1 trivially) at the cost of ~0.08-stops/bin resolution instead of ~0.02. Loses the JND-below-cone resolution we aimed for.

All four are larger restructurings than this PR; the 1.7× win from the current SIMD path is the easy gain. Profile-guided tuning on a real-content corpus (vs the synthetic ramp here) is the next step before we restructure.

## Reproduce

```bash
# V3 / AVX2 + emulated NEON & WASM128 (default tier on zenpixels-convert)
cargo run -p zenpixels-convert --example measure_histogram_throughput \
    --release --features hdr-experimental

# With -C target-cpu=native (compiles V3 path with AVX2 intrinsics)
RUSTFLAGS="-C target-cpu=native" cargo run -p zenpixels-convert \
    --example measure_histogram_throughput \
    --release --features hdr-experimental

# AVX-512 tier on top
RUSTFLAGS="-C target-cpu=native" cargo run -p zenpixels-convert \
    --example measure_histogram_throughput \
    --release --features "hdr-experimental avx512"
```

Output format: per-row label, dimensions, iteration count, elapsed ms, computed Mpix/s.
