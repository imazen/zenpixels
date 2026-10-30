# Fused scan predicates: per-tier NEON isolation — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/kernel_tiers.rs` (zenbench, interleaved arms), 1 MP RGBA8

`fused_predicates_rgba8_cg` is the production one-pass predicate driving format negotiation
(is-opaque / is-grayscale) on every image. The other benches here measure conversion
pipelines; an aggregate cannot reveal one kernel losing to its own scalar fallback. That
failure mode was real in this same sweep — three zenfilters NEON kernels were measurably
slower than their scalar tier.

## Result: no losers, NEON wins decisively

| case | NEON | scalar | speedup |
|---|---|---|---|
| opaque only, all-true | 85.8 µs | 306.4 µs | **3.57×** |
| grayscale only, all-true | 112.8 µs | 506.0 µs | **4.49×** |
| both, all-true | 171.1 µs | 840.5 µs | **4.91×** |
| both, fails on first pixel | 132 ns | 239 ns | 1.81× |

Two input shapes are measured on purpose. These predicates early-exit on the first failing
pixel, so an all-true buffer measures steady-state throughput while a fails-on-first-pixel
buffer measures entry and dispatch cost. Those are very different regimes (µs vs ns) and a
single number would hide one of them.

The all-true ratios are among the highest in this sweep because the work is a wide
compare-and-reduce — exactly the shape LLVM's autovectorizer handles worst and hand-written
SIMD handles best, unlike the elementwise passes elsewhere that sit at the memory-bandwidth
wall.

## Note

The `is_opaque_rgba8` / `is_grayscale_rgba8` single-purpose functions in `scan.rs` are
`#[cfg(test)]` reference implementations, not the production path — they exist to validate the
shared mask constants at every dispatch tier. Benchmarking them would have measured code that
does not ship. The fused entry point is what production calls and what is measured here.

The bench needs `--features __bench_scan` (a pre-existing bench-only feature that exposes the
crate-private `scan` module) plus `archmage/testable_dispatch`, added as a dev-dependency so
the baseline NEON token can be disabled. Without the latter the bench skips loudly rather than
reporting the SIMD path under both labels.
