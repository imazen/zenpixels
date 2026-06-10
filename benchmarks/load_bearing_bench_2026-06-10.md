# load_bearing / scan predicate benchmarks — 2026-06-10

- **Host:** AMD Ryzen 9 7950X (Zen 4), Linux WSL2
- **Base commit:** f2ae919d4de5 (PR #30 restructure)
- **Command:** `cargo bench --bench bench_load_bearing --features __bench_scan -p zenpixels-convert`
- **Build:** release, runtime SIMD dispatch (no `-C target-cpu=native`), default features → AVX2 (v4) tier
- **Harness:** zenbench (interleaved rounds, paired stats)
- **Content:** worst-case gray+opaque RGBA8 (no early exit — every byte scanned)

## Fused kernel pass structures (`fused_predicates_rgba8_cg`)

| size | fused 3-check 1-pass | three 1-check passes | scalar 3-check 1-pass |
|---|---|---|---|
|   64×64 (16 KB, L1) | 479 ns — 31.9 GiB/s | +3.4% | 4.73 µs — 3.23 GiB/s (9.9×) |
|  256×256 (256 KB, L2) | 7.4 µs — 33.0 GiB/s | +5.5% | 75.4 µs — 3.24 GiB/s (10.2×) |
| 1024×1024 (4 MB, L3) | 120 µs — 32.5 GiB/s | +11.9% | 1.19 ms — 3.27 GiB/s (9.9×) |
| 4096×4096 (64 MB, DRAM) | 2.8 ms — 22.4 GiB/s | **+176% (2.76×)** | 19.1 ms — 3.27 GiB/s (6.8×) |

Fusion pays where it was designed to: cache-resident buffers re-read for
free, so three passes cost only loop overhead (+3–12%); at DRAM-bound
sizes the single pass is 2.76× faster (one memory sweep instead of three).
SIMD vs scalar is ~10× cache-resident, 6.8× at DRAM bandwidth.

## Public trait entry (`determine_load_bearing`, Rgba8)

| size | straight alpha (3-check scan) | AlphaMode::Opaque (gray check only) |
|---|---|---|
|   64×64 | 480 ns | 211 ns (−56%) |
|  256×256 | 7.4 µs | 2.9 µs (−60%) |
| 1024×1024 (1 MP) | **120 µs** | 54 µs (−55%) |
| 4096×4096 (16 MP) | 2.4 ms | 2.2 ms (−7%) |

The full analysis at 1 MP costs 0.12 ms — cheap enough to run at every
encoder entry. AlphaMode-based elision halves cost at compute-bound
sizes and converges to bandwidth parity at DRAM-bound sizes (same bytes
read either way).

Raw log: captured at /tmp/pr30-bench.log during the run; regenerate with
the command above.
