# fast_gamut.rs redesign — design document

Status: **planned**, not implemented. Baselines captured under
`benchmarks/fast_gamut_baseline_2026-05-02/`.

## Why

Today's `fast_gamut.rs` has ~110 functions across four families
(stamp_trc_kernels float, matlut integer, hybrid LUT/poly, cross-depth)
and the SIMD coverage is **V3 + scalar, x86-only**. AArch64 / WASM /
AVX-512 systems all fall through to scalar gamut conversion.

The `stamp_trc_kernels!` macro reinvents `#[magetypes]` worse than
`#[magetypes]`. Its parameter shape doesn't generalize; the
`#[cfg(target_arch = "x86_64")]` gates silently amputate non-x86 SIMD.
The integer matlut family has the same problem and is hand-rolled per
function.

## What

Replace the four families with two const-generic magetypes bodies per
family, dispatched by `incant!` across all six tiers:

```rust
// Body 1 — wide tiers via f32x16 (the only width V4 / V4x impl).
#[magetypes(define(f32x16), v4x, v4, v3, scalar)]
fn convert_f32_impl<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
>(token: Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    // Calls into linear_srgb::tf::*::*_x16<T: F32x16Convert>.
    // …
}

// Body 2 — narrow tiers via f32x4 (works for NEON + WASM128 + scalar).
#[magetypes(define(f32x4), neon, wasm128)]
fn convert_f32_impl<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
>(token: Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    // Calls into linear_srgb::tf::*::*_x4<T: F32x4Convert>.
    // Same logical shape as Body 1, half the SIMD width.
    // …
}

pub fn convert<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
>(m: &[[f32; 3]; 3], data: &mut [f32]) {
    incant!(
        convert_f32_impl::<SRC_TRC, DST_TRC, CHANNELS>(m, data),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}
```

Const-generic axes:

- `SRC_TRC` and `DST_TRC` — integer-tag enum (`SRGB=0, BT709=1, PQ=2,
  HLG=3, ADOBE=4, LINEAR=5`). The body's `match SRC_TRC { … }` and
  `match DST_TRC { … }` const-prop to a single arm per specialization,
  so each monomorphization is a hand-tuned-equivalent kernel.
- `CHANNELS` — 3 (RGB) or 4 (RGBA, alpha passthrough). Replaces the
  `_rgb_` / `_rgba_` duplication.

For the integer pipeline (matlut + hybrid):

- Add `LIN_LUT: bool` and `ENC_LUT: bool` const generics. Collapses the
  4 hybrid kernels into 1 body with 4 specializations. LUT path uses
  per-arch gather (native on V3 / V4 / V4x; per-lane fallback on NEON /
  WASM128). Polynomial path uses linear-srgb's TRC kernels at the
  appropriate width.

For cross-depth (u8 ↔ f32):

- linear-srgb already provides `srgb_u8_to_linear_v3` / `_v4` / slice
  variants at every width. Replace the hand-rolled cross-depth functions
  with thin wrappers that call into linear-srgb's public APIs.

## Why two bodies, not one

Verified by compile: `F32x4Convert` and `F32x8Convert` are **not
implemented** for `X64V4Token` / `X64V4xToken`. Only `F32x16Convert` is
tier-uniform across all six tier tokens. So:

- A single f32x16 body covers V4x / V4 / V3 / scalar via the `*_x16<T>`
  family. Verified by cross-compile on x86_64 / aarch64 / wasm32 / i686.
- f32x16 polyfilled to NEON / WASM128 = 4× 128-bit ops, with register
  pressure issues in the polynomial inner loops (likely 30–60% slower
  than native f32x4 hand-tuned).
- A separate f32x4 body covers NEON + WASM128 (both impl F32x4Convert)
  via the `*_x4<T>` family at native register width. No spills.

Two bodies, one dispatcher, every architecture native-ish.

## Upstream prerequisites

### linear-srgb (unblocker)

The `tf::*::*_x16<T: F32x16Convert>` and `tf::*::*_x4<T: F32x4Convert>`
families are currently `pub(crate)`. **Flip them to `pub`** so the
generic backends can be called from zenpixels-convert.

Concrete patch (~12 lines):

```rust
// In tf/srgb.rs, tf/bt709.rs, tf/pq.rs, tf/hlg.rs:
- pub(crate) fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> { … }
+ pub        fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> { … }
// (Same for x16 variant. x8 variant can stay pub(crate) — V4 doesn't
// impl F32x8Convert, so x8 is V3-only and the existing public x8
// per-tier functions cover it.)
```

This is the only blocker. After it ships in a linear-srgb release (or
locally via path dep), the redesign can proceed.

### garb (optional optimization)

garb 0.2.7's deinterleave is f32x8-shaped (`[f32; 8]` planar arrays).
For the f32x16 body's per-batch deinterleave at 16 pixels, garb would
need either a generic-backend deinterleave or new f32x16-shaped
kernels. **Not blocking** — the body can use in-place magetypes shuffle
primitives for v0; garb upgrade is a perf optimization.

## Coverage matrix (after redesign)

|  | V4x | V4 | V3 | NEON | WASM128 | scalar |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| sRGB | ✅ wide | ✅ wide | ✅ wide | ✅ narrow | ✅ narrow | ✅ wide |
| BT.709 | ✅ wide | ✅ wide | ✅ wide | ✅ narrow | ✅ narrow | ✅ wide |
| PQ | ✅ wide | ✅ wide | ✅ wide | ✅ narrow | ✅ narrow | ✅ wide |
| HLG | ✅ wide | ✅ wide | ✅ wide | ✅ narrow | ✅ narrow | ✅ wide |
| Adobe (gamma) | ✅ wide | ✅ wide | ✅ wide | ✅ narrow | ✅ narrow | ✅ wide |
| u8 matlut | ✅ wide-LUT | ✅ wide-LUT | ✅ wide-LUT | ✅ narrow-poly | ✅ narrow-poly | ✅ scalar |
| u16 hybrids | ✅ wide-LUT/poly | ✅ wide-LUT/poly | ✅ wide-LUT/poly | ✅ narrow-poly | ✅ narrow-poly | ✅ scalar |

`wide` = f32x16 body. `narrow` = f32x4 body. The `narrow` path on x86 is
unused (V3 / V4 / V4x prefer the wide body).

## File structure (proposed)

```
zenpixels-convert/src/
├─ fast_gamut/
│   ├─ mod.rs               # public surface (today's pub functions)
│   ├─ trc.rs               # SRC_TRC / DST_TRC integer constants + match dispatchers
│   ├─ float_wide.rs        # #[magetypes(define(f32x16), v4x, v4, v3, scalar)] body
│   ├─ float_narrow.rs      # #[magetypes(define(f32x4), neon, wasm128)] body
│   ├─ integer_wide.rs       # u16 hybrid body, f32x16 width
│   ├─ integer_narrow.rs     # u16 hybrid body, f32x4 width
│   └─ cross_depth.rs       # thin wrappers around linear-srgb's u8 ↔ f32 APIs
└─ fast_gamut.rs            # delete after migration
```

Total lines projected: ~700 (vs today's ~2700).

## Migration plan

1. **Open upstream linear-srgb PR** with the visibility flip. ~12-line
   diff, one round of review. Block until it merges + a release ships.
   (Or use a local path dep for development.)
2. **Add `"avx512"` to linear-srgb's feature list in
   zenpixels-convert/Cargo.toml.** Unblocks the `*_x16<T>` family.
3. **Implement `float_wide.rs` and `float_narrow.rs` for sRGB-only**
   (one TRC) as a proof of concept. Re-run bench_t3_tf_fused; compare
   against baseline.
4. **Extend to BT.709 / PQ / HLG / Adobe.** Re-run bench_t3_tf_fused;
   bench_t7_gamut.
5. **Implement `integer_wide.rs` and `integer_narrow.rs` for the matlut
   family** with const-generic LIN_LUT / ENC_LUT.
6. **Replace the hybrid lutdec_polyenc / polydec_lutenc / polydec_polyenc
   functions** with const-generic specializations on the integer body.
7. **Replace cross-depth u8 ↔ f32 functions** with thin wrappers around
   linear-srgb's public APIs.
8. **Delete `stamp_trc_kernels!` macro and the old `fast_gamut.rs`.**

Each step is independently shippable. Steps 3–7 can each be merged
individually, gated by feature flag during transition if needed.

## Benchmarks

Baselines captured 2026-05-02:

```
benchmarks/fast_gamut_baseline_2026-05-02/
├─ bench_t3_tf_fused_BEFORE.log    # TF-fused depth conversions
├─ bench_t7_gamut_BEFORE.log       # P3 ↔ BT.709 gamut + TF
├─ bench_matlut_vs_poly_BEFORE.log # Integer matlut paths
└─ META.txt
```

After-refactor benchmarks should match these (same hardware, same
release profile) and demonstrate:

- **AVX-512 hosts** (V4 / V4x): 1.5–2× over baseline (V3 → V4x = 2× wider register)
- **AVX2 hosts** (V3): within 10% of baseline (f32x16 polyfilled to
  2× 256-bit; register pressure is the risk)
- **AArch64 hosts** (NEON): currently scalar; after = 4–8× faster
- **WASM hosts** (WASM128): currently scalar; after = 4–8× faster

Cross-arch numbers will need cross-target benchmarking via `cross` or
QEMU-user.

## Out of scope

- Hand-tuned `_v3` slot-in (using f32x8 native instead of f32x16
  polyfilled). Would recover ~10% on AVX2 but adds another body. Defer
  until benchmarks justify.
- New TRCs (Display-P3-PQ, BT.2100-HLG-narrow-range, etc). The const-
  generic enum is extensible — adding a new TRC tag is one match arm.
- Automatic LUT/poly selection per-arch. Today the dispatcher hard-codes
  LUT on x86, poly on NEON / WASM. A "smart auto" pickier could measure
  per host. Out of scope for v1.
