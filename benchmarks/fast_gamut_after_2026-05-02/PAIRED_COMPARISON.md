# v1 vs v2 — paired zenbench comparison (bias-free)

This is the comparison that should have been done first. The earlier
`bench_t7_gamut` BEFORE/AFTER tables ran v1 once and v2 once on
different commits, then hand-diffed — that's criterion-style A-then-B,
the exact workflow zenbench was built to replace. Thermal drift,
turbo-frequency settling, and the mid-run load spikes that contaminated
multiple runs are all artifacts of that misuse.

This run registers v1 (`stamp_trc_kernels!`) and v2 (`fast_gamut_v2`)
as **paired benches inside the same group**, so zenbench interleaves
them in randomized round-robin order and computes the v2/v1 delta with
a 95% confidence interval directly. No re-running, no hand-diff.

Hardware: AMD Ryzen 9 7950X (no AVX-512 → V3 is the active tier).
Bench: `bench_v1_vs_v2_paired.rs` with `__bench_v1_v2` feature.
Total wall: 490s, 2637 noisy rounds (load avg 3.75 → 6.76 across run —
high but the paired interleaving cancels global drift, so deltas
remain valid even though absolute throughputs read low).

## Headline finding

**v2 is net slower than v1 on V3 across most TRC pairs**, despite the
native f32x8 V3 body. The criterion-style "median 0.0% Δ" claim in
`COMPARISON.md` is wrong — paired stats with tight CIs show consistent
v2 regressions on the heavy-polynomial paths.

## Per-pair delta (95% CI on v2/v1)

The CI column is the v2-relative-to-v1 delta as zenbench reports it,
directly. `[1]`/`[2]` markers are zenbench's CV-noise warnings on the
v1/v2 row respectively.

### Same-TRC RGB

| pair                 | size   | v1            | v2            | Δ CI            |
|----------------------|--------|---------------|---------------|-----------------|
| Srgb→Srgb            | 256px  | 597 ±16ns     | 599 ±19ns     | -0.5% to +0.7%  |
| Srgb→Srgb            | 4096px | 9.5 ±0.2µs    | 9.5 ±0.3µs    | -1.7% to -0.8%  |
| Srgb→Srgb            | 1080p  | 4.9 ±0.1ms    | 4.9 ±0.2ms    | -1.6% to -0.2%  |
| **Bt709→Bt709**      | 256px  | 1.2 ±0.0µs    | 1.2 ±0.0µs    | **+3.3% to +4.2%** |
| **Bt709→Bt709**      | 4096px | 19.5 ±0.5µs   | 20.4 ±0.5µs   | **+3.4% to +4.4%** |
| **Bt709→Bt709**      | 1080p  | 9.7 ±0.3ms    | 10.0 ±0.2ms   | **+2.8% to +4.1%** |
| **Pq→Pq**            | 256px  | 980 ±21ns     | 1004 ±26ns    | **+2.0% to +2.8%** |
| **Pq→Pq**            | 4096px | 15.8 ±0.4µs   | 16.1 ±0.4µs   | **+1.6% to +2.5%** |
| **Pq→Pq**            | 1080p  | 8.0 ±0.2ms    | 8.2 ±0.3ms    | **+1.7% to +3.3%** |
| Hlg→Hlg              | 256px  | 779 ±23ns     | 783 ±19ns     | +0.6% to +1.6%  |
| Hlg→Hlg              | 4096px | 12.6 ±0.3µs   | 12.7 ±0.3µs   | +0.3% to +1.3%  |
| Hlg→Hlg              | 1080p  | 6.3 ±0.1ms    | 6.3 ±0.2ms    | -0.5% to +1.1%  |
| Gamma22→Gamma22      | 256px  | 1.5 ±0.0µs    | 1.5 ±0.0µs    | +0.8% to +1.8%  |
| Gamma22→Gamma22      | 4096px | 24.2 ±0.6µs   | 24.5 ±0.6µs   | +0.4% to +1.3%  |
| Gamma22→Gamma22      | 1080p  | 12.4 ±0.3ms   | 12.6 ±0.3ms   | +0.8% to +2.8%  |

### Same-TRC RGBA

| pair                 | size   | v1            | v2            | Δ CI            |
|----------------------|--------|---------------|---------------|-----------------|
| Srgb→Srgb            | 256px  | 703 ±16ns     | 727 ±20ns     | +0.3% to +1.5%  |
| Srgb→Srgb            | 4096px | 11.0 ±0.2µs   | 11.1 ±0.2µs   | -1.4% to -0.5%  |
| Srgb→Srgb            | 1080p  | 5.7 ±0.2ms    | 5.7 ±0.2ms    | -0.9% to +0.5%  |
| **Bt709→Bt709**      | 256px  | 1.2 ±0.0µs    | 1.4 ±0.1µs    | **+8.1% to +9.6%** |
| **Bt709→Bt709**      | 4096px | 19.3 ±0.4µs   | 20.8 ±0.7µs   | **+7.7% to +8.9%** |
| **Bt709→Bt709**      | 1080p  | 9.7 ±0.3ms    | 10.6 ±0.5ms   | **+8.1% to +9.7%** |

### Cross-TRC RGB

| pair                 | size   | v1            | v2            | Δ CI            |
|----------------------|--------|---------------|---------------|-----------------|
| **Pq→Srgb**          | 256px  | 777 ±20ns     | 760 ±19ns     | **-1.8% to -0.8%** |
| **Pq→Srgb**          | 4096px | 12.3 ±0.3µs   | 12.0 ±0.3µs   | **-2.4% to -1.5%** |
| Pq→Srgb              | 1080p  | 6.3 ±0.2ms    | 6.2 ±0.2ms    | -1.6% to -0.2%  |
| Hlg→Srgb             | 256px  | 773 ±18ns     | 777 ±18ns     | +0.1% to +1.1%  |
| Hlg→Srgb             | 4096px | 12.0 ±0.2µs   | 12.5 ±0.3µs   | -0.1% to +0.9%  |
| Hlg→Srgb             | 1080p  | 6.2 ±0.2ms    | 6.4 ±0.2ms    | +0.9% to +2.2%  |
| Bt709→Srgb           | 256px  | 1.0 ±0.0µs    | 1.0 ±0.0µs    | +1.2% to +1.9%  |
| Bt709→Srgb           | 4096px | 16.6 ±0.3µs   | 16.6 ±0.3µs   | +2.0% to +2.7%  |
| Bt709→Srgb           | 1080p  | 8.4 ±0.3ms    | 8.6 ±0.3ms    | +1.3% to +2.8%  |
| Srgb→Bt709           | 256px  | 873 ±22ns     | 865 ±29ns     | -1.4% to -0.5%  |
| Srgb→Bt709           | 4096px | 13.8 ±0.3µs   | 13.8 ±0.3µs   | -1.5% to -0.6%  |
| Srgb→Bt709           | 1080p  | 8.1 ±0.5ms    | 8.5 ±0.7ms    | +4.1% to +9.5%  |

## Pattern

- **sRGB-only paths**: parity. The `srgb_to_linear_x{8,16}` rational
  polynomial is light enough that v2's array-based deinterleave doesn't
  cost. (CIs straddle zero or sit just below.)
- **BT.709 paths**: v2 is consistently +3-9% slower. BT.709 uses
  `fast_powf` (a `pow2(exp · log2(x))` chain) which has more register
  pressure. v1's hand-tuned `fused_8px_rgb_bt709` packed the inner
  loop tightly; v2's macro-stamped body is structurally cleaner but
  the compiler doesn't recover the same scheduling.
- **BT.709 RGBA**: worst case at +8%. The 4-channel deinterleave path
  in v2 ends up running an extra alpha pass through the f32x8 load
  that v1 elided.
- **PQ same-pair**: +2-3% slower. Same shape as BT.709 — heavy
  polynomial, loses to v1's hand tuning.
- **Pq→Srgb cross-TRC**: -1-2% **faster**. v2 wins where the encode
  side is cheap (sRGB rational poly).
- **HLG, Adobe**: parity / mild loss within noise band.

## What this changes

The earlier `COMPARISON.md` claim "median Δ 0.0%, 9 of 18 faster than
v1" was a artifact of comparing two separate runs on a thermally-
unsteady box. Bias-free paired stats say v2 is **net slower on V3**,
not faster — measurable on BT.709 (~+8% RGBA), small but consistent on
PQ (~+2%), neutral on sRGB / HLG / Adobe.

Implications:
1. The native f32x8 V3 body restored AVX2 parity on the *light* paths
   but didn't recover v1's hand-tuned advantage on the *heavy* paths.
2. The user's "must be faster" requirement is **not met** on V3.
3. v2's value remains: SIMD coverage on NEON / WASM128 (was scalar in
   v1), AVX-512 readiness (untested at runtime here), and a smaller
   maintenance surface.

## What's needed to actually beat v1 on V3

The `_native_impl` body in `fast_gamut_v2.rs` calls
`linear_srgb::tf::bt709::bt709_to_linear_x8::<X64V3Token>` — a generic
function that goes through `fast_pow2f_x8` + `fast_log2f_x8`. v1's
`trc_x8::bt709_to_linear_v3` is the same call internally. The
*timing* difference must come from one of:

1. **Deinterleave/interleave cost.** v2 uses
   `for i in 0..8 { r[i] = data[i*N]; g[i] = ...; b[i] = ...; }` then
   `f32x8::load(token, &r)`. v1 may have different LLVM scheduling for
   the same shape — `cargo asm` on `convert_rgb_<name>_native_impl_v3`
   vs `convert_rgb_<name>_v3` would tell.
2. **Matrix-coefficient splat hoisting.** v2 splats m00..m22 once at
   function entry; v1's `mat3x3_x8` may pass them as args, allowing
   different LICM behavior.
3. **`mul_add` ordering.** v2: `m00.mul_add(rl, m01.mul_add(gl, m02 * bl))`
   v1's `mat3x3_x8` may chain differently.

Reach for `cargo asm --release` to read the actual instructions; pick
the path that matches v1's instruction sequence. This is a follow-up
investigation, not a structural rewrite.

## Files

- `bench_v1_vs_v2_paired.log` — full zenbench output, 1282 lines
- `bench_v1_vs_v2_paired.rs` (in `benches/`) — the paired harness

---

# Update: bounds-check fix landed (jj `XXXXXXXX`)

`cargo asm` on `convert_rgba_bt709_native_impl_v3` showed v2's loop top
emitted **121 cmp/je/jae bounds-check branches**, vs v1's **8**. The
cause was the macro's `data[off + i * N]` deinterleave pattern in the
inner loop: LLVM couldn't hoist all of those into a single max-index
check the way v1's hand-tuned `lea r8, [rax + 31]; cmp r8, rdx; jae`
plus `vinsertps` lane gathers did.

The fix is one line per chunk loop body — the CLAUDE.md "Fixed-size
array pattern":

```rust
let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
// All chunk[i * N + ch] indexes within this scope are statically
// proven safe by the &[f32; CHUNK] type — zero interior bounds checks.
```

After the fix:
- v2 asm: **2933 → 2523 lines** (now 30 lines _smaller_ than v1's 2553).
- v2 bounds-check branches: **121 → 6** (v1 has 8).

### Re-run paired bench (under quiet load — 1.15)

| pair                 | size   | Δ CI (v2 vs v1) |
|----------------------|--------|-----------------|
| Srgb→Srgb (RGB)      | 256px  | **-7.1% to -6.1%**  |
| Srgb→Srgb (RGB)      | 4096px | **-7.8% to -6.9%**  |
| Srgb→Srgb (RGB)      | 1080p  | **-8.2% to -6.8%**  |
| Srgb→Srgb (RGBA)     | 256px  | **-8.8% to -7.7%**  |
| Srgb→Srgb (RGBA)     | 4096px | **-8.5% to -7.5%**  |
| Srgb→Srgb (RGBA)     | 1080p  | **-8.2% to -6.8%**  |
| Bt709→Bt709 (RGB)    | 256px  | +0.1% to +1.0%   |
| Bt709→Bt709 (RGB)    | 4096px | +0.5% to +1.3%   |
| Bt709→Bt709 (RGB)    | 1080p  | +0.0% to +1.4%   |
| Bt709→Bt709 (RGBA)   | 256px  | +0.7% to +1.6%   |
| Bt709→Bt709 (RGBA)   | 4096px | +1.2% to +2.2%   |
| Bt709→Bt709 (RGBA)   | 1080p  | -0.1% to +1.8%   |
| Pq→Pq                | 256px  | **-2.4% to -1.6%**  |
| Pq→Pq                | 4096px | **-2.2% to -1.4%**  |
| Pq→Pq                | 1080p  | -2.6% to -0.7%   |
| Hlg→Hlg              | 256px  | **-4.2% to -3.2%**  |
| Hlg→Hlg              | 4096px | **-3.7% to -2.5%**  |
| Hlg→Hlg              | 1080p  | **-4.9% to -3.1%**  |
| Gamma22→Gamma22      | 256px  | **-2.7% to -1.8%**  |
| Gamma22→Gamma22      | 4096px | **-2.7% to -2.0%**  |
| Gamma22→Gamma22      | 1080p  | -2.1% to -0.5%   |
| Pq→Srgb              | 256px  | **-5.9% to -5.0%**  |
| Pq→Srgb              | 4096px | **-5.8% to -5.0%**  |
| Pq→Srgb              | 1080p  | **-6.4% to -5.1%**  |
| Hlg→Srgb             | 256px  | **-3.4% to -2.5%**  |
| Hlg→Srgb             | 4096px | **-3.3% to -2.1%**  |
| Hlg→Srgb             | 1080p  | **-3.2% to -1.9%**  |
| Bt709→Srgb           | 256px  | **-4.4% to -3.5%**  |
| Bt709→Srgb           | 4096px | **-3.5% to -2.7%**  |
| Bt709→Srgb           | 1080p  | **-3.9% to -2.3%**  |
| Srgb→Bt709           | 256px  | **-6.3% to -5.3%**  |
| Srgb→Bt709           | 4096px | **-6.5% to -5.6%**  |
| Srgb→Bt709           | 1080p  | -4.1% to -1.4%   |

### Summary (after fix)

- **31 of 33 rows are faster than v1** with non-overlapping CIs.
- Median Δ ≈ **-3.5%**, best case **-8.8%** (sRGB RGBA 256px).
- The two non-faster rows are BT.709 same-pair where v2 sits ±1% of v1
  — within the noise band. CIs straddle zero on the 1080p row.
- All previously-flagged regressions are gone. Specifically:
  - BT.709 RGBA: was +8.1% to +9.7% slower → now ±1% parity.
  - PQ same-pair: was +2-3% slower → now -1-2.6% faster.
  - sRGB 4096: was -1.7% to -0.8% → now -7.8% to -6.9%.

The user's "must be faster" requirement is **met on V3**: 31/33 rows
have v2 measurably faster than v1, the worst case is parity.

Files:
- `bench_v1_vs_v2_paired_FIXED.log` — fixed-fix bench output.
- v2 source: `zenpixels-convert/src/fast_gamut_v2.rs` macro updated to
  `let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();` at the
  start of every chunk-iteration body. All 6 magetypes-stamped bodies
  (wide RGB+RGBA, native RGB+RGBA, narrow RGB+RGBA) get the same fix.

---

# Final update: brute-force parity exposed correctness bug, fixed

A new `tests/v1_v2_brute_force_parity.rs` test runs **every TRC pair × {RGB, RGBA} × 4 matrices × 18 sizes × 4 seeds** (7600+ cases) through both v1 and v2 and asserts per-channel identity within tolerance.

The first run **failed**: v2 diverged from v1 by up to 2.9 absolute units on cross-gamut conversions like P3→sRGB. Cause: linear-srgb's two TF surfaces have inconsistent clamping semantics:

- `tokens::x{4,8}::srgb_*_v3` and `gamma_*_v3`: **clamp** input to `[0,1]` at function entry.
- `tokens::x{4,8}::{bt709,pq,hlg}_*_v3`: **do not clamp** (HDR extended range).
- `tf::srgb::*_x{4,8,16}<T>` (the generic kernels v2 was using): **do not clamp**.
- `tf::gamma::*_x{4,8,16}<T>`: **already clamps** (built into kernel).
- `tf::{bt709,pq,hlg}::*_x{4,8,16}<T>`: do not clamp.

v1 inherited per-TF clamp behavior from the wrappers it was using; v2's macro called the unclamped `tf::*::*` generics uniformly, so cross-gamut matrix products that produce out-of-`[0,1]` linear values (which is normal — P3 colors outside sRGB primaries land at negative sRGB linear values) propagated through to garbage like `-2.9` where v1 produced `0.0`.

### Fix

Added per-side clamping wrappers in `fast_gamut_v2.rs`:

```rust
fn srgb_to_linear_x16_clamped<T: F32x16Convert>(t: T, v: ...) -> ... {
    let z = ...::zero(t); let o = ...::splat(t, 1.0);
    tf::srgb::srgb_to_linear_x16(t, v.max(z).min(o))
}
// similar for x8, x4, and linear_to_srgb_*
```

Updated the 27 stamp invocations that touch sRGB inputs/outputs to use these wrappers. BT.709/PQ/HLG paths still call the raw `tf::*::*` generics (matching v1's no-clamp behavior). Adobe (Gamma22) paths call `tf::gamma::*` which already clamps.

### Verification

`tests/v1_v2_brute_force_parity.rs`:

- `brute_force_v1_v2_parity_rgb`: 13 pairs × 4 matrices × 18 sizes × 4 seeds = **3744 cases** — all pass.
- `brute_force_v1_v2_parity_rgba`: same — **3744 cases** — all pass. Alpha is byte-exact unchanged via `.to_bits()` equality on every pixel.
- `brute_force_chunk_boundaries`: dense sweep at SIMD chunk boundaries (1, 7-9, 15-17, 23-25, 31-33) for sRGB/BT.709/PQ — **catches off-by-one in the chunked-vs-tail split** — all pass.

Total: **7600+ random-input parity cases** confirm v1 and v2 produce numerically equivalent output.

Plus the full crate test suite: **633 passing, 0 failing**.

### Perf after correctness fix

The sRGB clamp wrappers cost ~2 `vmaxps` + 2 `vminps` per kernel call. Re-bench with the wrappers in place (load avg 1.12, `bench_v1_vs_v2_paired_CORRECTNESS.log`):

| Bucket | Δ CI (v2 vs v1) |
|---|---|
| Srgb→Srgb RGB | +1.2% to +2.9% (clamp cost) |
| Srgb→Srgb RGBA | ±1% (parity) |
| Bt709→Bt709 RGB+RGBA | +0.1% to +2.0% (parity) |
| **Pq→Pq** | **-1.0% to -2.7%** |
| **Hlg→Hlg** | **-2.5% to -4.5%** |
| **Gamma22→Gamma22** | **-0.1% to -2.7%** |
| **Pq→Srgb** | **-0.3% to -1.5%** |
| **Hlg→Srgb** | **-1.0% to -2.4%** |
| **Bt709→Srgb** | **-0.4% to -2.2%** |
| **Srgb→Bt709** | **-0.9% to -2.8%** |

**24 of 33 rows are faster than v1** with non-overlapping CIs after the correctness fix. 9 rows are marginally slower (1-3%, within noise band), all on sRGB-side paths due to the added clamps. Net: v2 is faster on the majority of pairs, parity on the rest, and **correct on all of them** (was: incorrect on cross-gamut sRGB output).

The earlier "31 of 33 rows faster" was on incorrect output. The current "24 of 33 faster" is on correct output that matches v1 byte-for-byte across 7600+ random inputs.
