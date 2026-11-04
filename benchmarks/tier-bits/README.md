# Do tiers and platforms produce the same bits?

Measured 2026-11-04 on linear-srgb's 12 plain slices, 1,200,002 samples each: 1,000,001 uniform in `[0, 1]`
and 200,001 of `t^4`. Hosts: r7900x (Zen 4) and `zen-arm-xl` (Neoverse N1). WASM under wasmtime on r7900x.
Not measured: Zen 5, Apple, ARMv7, JavaScriptCore, V8, or NaN and out-of-range input.

`tier_diff.rs` goes in linear-srgb's `tests/` and dumps each slice per tier to `$FMA_OUT`. `compare.rs` is a
standalone binary: `compare <dump-root> <build>:<tier>:<build>:<tier>...`.

Builds, all from linear-srgb `2145ecbc` unless stated:

- **current:** `mul_add_relaxed` in vector kernels, hand-written scalar tiers.
- **fused:** the same with `mul_add_fused`, and `mlaf` always fused.
- **fused kernels:** fused, and the scalar tier runs the vector kernels instead of the scalar functions.
- **unfused kernels:** `c47ee9f6`, whose scalar tier runs the vector kernels with separate multiply and add.

| Comparison | Samples differing of 14.4M | Largest ULP | As `u16` | Largest codes |
|---|---:|---:|---:|---:|
| v3 and v4x, current | 0 | 0 | 0 | 0 |
| v3, current and fused | 0 | 0 | 0 | 0 |
| NEON, current and fused | 0 | 0 | 0 | 0 |
| x86-64 v3 and aarch64 NEON, current | 0 | 0 | 0 | 0 |
| v3 and WASM relaxed, current (wasmtime fuses) | 0 | 0 | 0 | 0 |
| v3 and WASM strict, current | 5,536,533 | 114,827 | 22,689 | 1 |
| v3 and scalar, unfused kernels | 5,519,873 | 81 | 22,689 | 1 |
| v3 and scalar, current | 5,318,178 | 4,124,044 | 16,988 | 1 |
| v3 and scalar, fused | 5,287,855 | 4,124,044 | 16,390 | 1 |
| v3 and scalar, fused kernels | 338,229 | 114,827 | 772 | 1 |
| x86-64 scalar and aarch64 scalar, current | 412,679 | 1 | 1,304 | 1 |
| x86-64 scalar and aarch64 scalar, fused | 0 | 0 | 0 | 0 |

So fusing doesn't close the v3 and scalar gap on its own, because the hand-written scalar tiers run different
algorithms. With fused kernels, 8 of 12 slices match exactly. The other 4 are the gamma and PQ slices, the
ones that call `pow_midp`; magetypes' transcendentals call `mul_add` internally, the likely but unproved cause.

The x86-64 scalar difference is `mlaf`, which chooses fused or separate at compile time.

## Cost of fused kernels

ns per sample, range over the slices: 12 natively, 11 on WASM (test output overwrote one timing). One run
each, so treat small differences as noise.

| Target | Current | Fused kernels |
|---|---:|---:|
| x86-64 v3 | 0.18-0.57 | 0.17-0.58 |
| aarch64 NEON | 1.41-2.99 | 1.29-3.00 |
| x86-64 without FMA (scalar tier) | 3.09-18.78 | 3.50-13.88 |
| WASM strict | 0.80-3.35 | 3.33-34.17 |
| WASM relaxed, wasmtime | 0.64-2.83 | 1.87-7.56 |

The unfused-kernels scalar tier on x86-64 ran at 0.58-8.47 ns.

`RowConverter` size at x86-64 opt-level 3, fat LTO: 319,592 bytes added today, 654,680 with fused kernels.
