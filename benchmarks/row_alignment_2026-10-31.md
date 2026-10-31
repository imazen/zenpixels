# Row-start alignment cost per SIMD tier (2026-10-31)

Bench: `zenpixels-convert/benches/bench_row_alignment.rs`. In-place multiply-add
over strided F32 RGBA rows (load, mul, add, store; the access pattern of the
conversion kernels), each tier called directly with its token. Every row
starts 0, 4, 16 or 32 bytes past a 64-byte boundary; stride is a multiple of 64
bytes. 20 fixed rounds per group, zenbench 0.1.11, `--no-busy-gate` (shared
hosts; load recorded below). Tree: main `43c74ea0` plus the bench.

```sh
cargo bench -p zenpixels-convert --features avx512 --bench bench_row_alignment -- --no-busy-gate   # x86
cargo bench -p zenpixels-convert --bench bench_row_alignment -- --no-busy-gate                     # aarch64
```

Values: slowdown against the 64-byte-aligned row start, as zenbench's 95% CI.

## AMD Ryzen 9 7900X (Zen 4, AVX-512 executed as 2×256-bit), rustc 1.99.1

Two runs; this is the second (load 1.4 before, 1.5 after). The first run
(load 3.4 from a preceding build) agrees within a few percent.

| Tier | Size | +4 B | +16 B | +32 B |
|---|---|---|---|---|
| v4 (64-byte vectors) | L1, 8 KB | +39.2%–+42.5% | +41.2%–+44.0% | −0.3%–+1.1% |
| v4 | L2, 256 KB | +53.3%–+53.9% | +53.2%–+53.9% | +8.8%–+9.5% |
| v4 | memory, 32 MB | +2.8%–+4.2% | +0.9%–+4.7% | +2.6%–+5.2% |
| v3 (32-byte vectors) | L1 | +28.9%–+30.0% | +28.2%–+29.0% | −0.3%–+0.3% |
| v3 | L2 | +48.1%–+48.8% | +48.3%–+49.4% | −1.1%–−0.7% |
| v3 | memory | +1.1%–+4.2% | +1.1%–+6.8% | −1.9%–+1.8% |
| scalar (compiler-vectorized SSE2, 16 bytes) | L1 | +12.1%–+12.6% | −0.2%–+0.1% | −0.2%–+0.1% |
| scalar | L2 | +22.5%–+22.9% | −0.3%–0.0% | −0.4%–−0.2% |
| scalar | memory | +6.4%–+12.5% | −8.3%–−1.3% | −2.8%–+1.1% |

Aligned throughput: v4 158 GiB/s (L1), 149 GiB/s (L2), 32 GiB/s (memory);
v3 157 / 154 / 32 GiB/s; scalar 81 / 78 / 29 GiB/s.

## Neoverse-N1 (zen-arm-xl, Hetzner cax41), rustc 1.97.1

The niced fuzz service occupied up to 6 of 16 cores (load 1.1 before, 4.5
after); the bench ran at nice 10.

| Tier | Size | +4 B | +16 B | +32 B |
|---|---|---|---|---|
| neon (16-byte vectors) | L1, 8 KB | +1.7%–+2.6% | −0.2%–+0.2% | −0.2%–+0.4% |
| neon | L2, 256 KB | +6.0%–+7.5% | −5.6%–−3.9% | −7.4%–−5.3% |
| neon | memory, 32 MB | +1.6%–+4.2% | −0.4%–+1.8% | −1.9%–+0.8% |
| scalar | L1 | −0.2%–+1.1% | −0.8%–+0.2% | −1.0%–−0.5% |
| scalar | L2 | +7.9%–+9.7% | −2.4%–0.0% | −3.6%–−1.8% |
| scalar | memory | +1.1%–+3.5% | −0.4%–+2.7% | −0.2%–+2.4% |

Aligned throughput: neon 25 GiB/s (L1), 23 GiB/s (L2), 9.8 GiB/s (memory).

## Reading

- A row start costs only when it is not a multiple of the vector width: v3
  pays at +4 and +16, not +32; 16-byte SSE2 and NEON pay only at +4.
- Cache-resident rows (the strip sizes zenpipe streams) pay 28–54% on AVX2 and
  AVX-512 for a misaligned start on Zen 4. From memory the cost is under 7%
  for v3 and v4 (up to 12.5% for 16-byte SSE2 at +4).
- AVX-512 at +32 costs nothing in L1 and 9% in L2 on Zen 4. That fits Zen 4
  executing 512-bit operations as two 256-bit halves (an interpretation, not
  measured). Full-width AVX-512 (Zen 5, Intel) may behave like +4 at +32; not
  measured (benchmarks do not run on dev).
- One load-bound kernel only; kernels with more arithmetic per load were not
  measured. Also not measured: wasm128, Apple silicon, Intel.
