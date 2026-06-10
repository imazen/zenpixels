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

---

# Blocked-reduction kernel (same day, optimization pass)

The tables above showed the kernel compute-limited at ~33 GiB/s
cache-resident: 1–3 `any_true` mask reductions + branches per 64-byte
chunk were the ceiling, not memory. The kernel was rewritten to
OR-accumulate violations into one register and reduce once per
8-chunk (512 B) block, with a rare ≤512 B scalar re-scan when the
accumulator fires (content transition → resume in the narrower
const-generic specialization).

## After (same command, same content, same host)

| size | fused 3-check 1-pass | vs before | three 1-check passes | scalar |
|---|---|---|---|---|
|   64×64 | 203 ns — 75.2 GiB/s | **2.36×** | +72% | 27× slower |
|  256×256 | 3.1 µs — 78.3 GiB/s | **2.37×** | +108% | 29× slower |
| 1024×1024 | 57.9 µs — 67.4 GiB/s | **2.07×** | +127% | 22× slower |
| 4096×4096 | 2.7 ±0.4 ms — 23.1 GiB/s | ~parity | +206% | 6.6× slower |

`determine_load_bearing` (Rgba8 straight): 222 ns at 64×64 (was 480),
2.6 µs at 256×256 (was 7.4), **59 µs at 1 MP (was 120)**. At 16 MP both
kernels converge on the single-core DRAM read wall — 20–26 GiB/s run to
run on this box (2.4–3.2 ms for 64 MB, high inter-run variance under
WSL2; zenbench flagged the noisy rounds). A cold 16 MP buffer cannot go
faster single-threaded; the fix at that size is analyzing strips while
they are cache-warm inside the pipeline (the per-row API composes for
free) or caller-side threading, not more kernel work.

The `AlphaMode::Opaque` elision now matters only at tiny sizes (177 vs
222 ns at 64×64); past L2 both variants saturate the same load
bandwidth.

## Unified partial block (follow-up simplification)

The trailing-chunks loop (per-chunk reductions, separate verdict code)
was replaced by letting the blocked loop's final block run short
(`block_end = min(i + 512, chunkable)`): one verdict shape, and narrow
strided rows now get one reduction per row instead of one per chunk.
Re-measured back-to-back: 1 MP identical (57.2 vs 57.9 µs), 256×256
within noise (75.8 vs 78.3 GiB/s), 64×64 possibly ~10% slower
(226–251 ns vs 203 ns, heavily noise-flagged; ambient variance moved
the *unchanged* scalar baseline ±35% between the same runs). Accepted:
≤40 ns/call at 16 KB for one code shape fewer.

---

# Post-ablation (2-check kernel, same day)

After the per-encoder consumer audit, `alpha_is_binary` and
`uses_gray_bit_depth` were removed (no concrete consumers; see PR #30).
The fused kernel drops to 2 checks (opaque + grayscale). Throughput is
unchanged within noise — the blocked kernel was load-bound, not
check-bound: 71–74 GiB/s cache-resident, ~22 GiB/s at 16 MP DRAM;
`determine_load_bearing` 55 µs at 1 MP, ~2.5 ms at 16 MP. The ablation
is a pure code-size/API win (−~700 lines, 3-field report).
