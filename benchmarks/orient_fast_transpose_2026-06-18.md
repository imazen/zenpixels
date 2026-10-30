# `orient` — `fast-transpose` compile-time vs runtime vs size

**Date:** 2026-06-18
**Base commit:** `208bded9` (zenpixels 0.2.15 dev) + the `__bench_orient` decouple
**Host:** `lilith` — AMD Ryzen 9 7950X, WSL2, 24 cores allocated; release builds, **no**
`-C target-cpu=native` (the SIMD kernels dispatch at runtime via archmage/`incant!`).
**Harness:** zenbench (`benches/bench_orient.rs`), run under `~/work/zen/scripts/run-heavy`.

## Question

`fast-transpose` is opt-in (off by default). Its comment claimed the opt-in exists
to avoid "pulling archmage/magetypes." Is that true, and is the runtime win worth
the compile/size cost — i.e., should it be default-on?

## Compile time (cold `cargo build -p zenpixels-convert --release`)

| | without ft | with ft | Δ |
|---|---|---|---|
| cold full build | 7.22s | 7.11s | **~0 (noise)** |
| incremental `orient` recompile | 2.65s | 2.68s | **~0 (noise)** |

**archmage + magetypes compile in BOTH builds** — they are unconditional deps (the
gamut / f16 SIMD paths already pull them). The "pulls archmage/magetypes" rationale
for opt-in is **false**; `fast-transpose` adds no measurable compile time.

## Binary size (`bench_orient` release binary, `size(1)` `.text`)

| | without ft | with ft | Δ |
|---|---|---|---|
| `.text` (code) | 967 KB | 1004 KB | **+36 KB** |
| total file | 1.265 MB | 1.306 MB | +40 KB |

The +36 KB is the AVX2 (`pxn_x86` / `rgb3_x86`) transpose kernels. This is the *only*
real cost of the feature — relevant solely for wasm / size-extreme builds.

## Runtime — SIMD `apply_orientation` (ft) vs scalar tiled gather (no ft)

zenbench, same input, mean of 4 rounds. Output is **bit-identical** (parity-tested
against the `forward_map` scatter oracle). Speedup = scalar ÷ SIMD.

### Rotate90 (the dominant real-world case — portrait phone photos)

| size | RGBA8 scalar→SIMD | × | RGB8 scalar→SIMD | × |
|---|---|---|---|---|
| 256²  | 28.8µs → 13.2µs | **2.2** | 31.2µs → 14.1µs | **2.2** |
| 1024² | 3.10ms → 0.65ms | **4.7** | 1.30ms → 0.87ms | 1.5 |
| 2048² | 12.1ms → 3.4ms  | **3.6** | 6.8ms → 2.9ms   | **2.3** |
| 12MP  | 42.5ms → 37.6ms | 1.13    | 41.6ms → 28.1ms | 1.48 |

### Transpose (pure-transpose baseline)

| size | RGBA8 scalar→SIMD | × | RGB8 scalar→SIMD | × |
|---|---|---|---|---|
| 256²  | 31.5µs → 12.8µs | **2.5** | 28.2µs → 13.4µs | **2.1** |
| 1024² | 2.70ms → 0.65ms | **4.1** | 1.00ms → 0.70ms | 1.4 |
| 2048² | 10.8ms → 3.3ms  | **3.3** | 7.6ms → 2.5ms   | **3.0** |
| 12MP  | 41.5ms → 35.1ms | 1.18    | 39.2ms → 26.9ms | 1.46 |

**Shape:** the SIMD win is largest at thumbnail→mid sizes (2–4.7×) and shrinks at
12MP (1.1–1.5×), where the transpose is memory-bandwidth-bound and the scalar tiled
gather is already efficient.

## Caveat

4 rounds only (1 call/round at ≥1024²), so the ≥1024² CIs are wide (CV up to ~22%);
the *direction* and order-of-magnitude are robust, but a denser round count would
tighten the mid-size multipliers. Transpose throughput is content-independent (a
memory shuffle), so no content sweep is needed; there is no quality knob.

## Conclusion

`fast-transpose` costs **~0 compile time + ~36 KB code** for a **1.1–4.7× faster**
transpose (biggest at small/mid sizes). The opt-in's stated compile rationale is
false. The only argument for opt-in is the 36 KB (wasm/size-extreme).

**Decision (2026-06-18):** queue **default-on for 0.3.0** (the output is bit-identical,
but flipping a default feature is batched into the next breaking release per the
0.2.x policy). 0.2.15 keeps it opt-in, with the false comment corrected and this
finding documented so consumers can enable it now.

## Reproduce

```sh
# compile time
run-heavy -- bash -c 'cargo clean; time cargo build -p zenpixels-convert --release;
  cargo clean; time cargo build -p zenpixels-convert --release --features fast-transpose'
# runtime (scalar vs SIMD)
run-heavy -- cargo bench --bench bench_orient --features __bench_orient
run-heavy -- cargo bench --bench bench_orient --features __bench_orient,fast-transpose
# size
size target/release/deps/bench_orient-*
```
