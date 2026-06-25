# ConvertPlan resource estimates

Reference for `zenpixels_convert::estimate` — what the numbers mean, how they
are calibrated, and how the threading model works. The module docstring on
[`zenpixels_convert::estimate`] points here; the API itself is documented
inline.

## Entry points

- [`ConvertPlan::estimate(width, height)`](../src/convert.rs) — quick estimate
  with default [`ComputeEnvironment`].
- [`ConvertPlan::estimate_in(image, compute)`](../src/convert.rs) — caller
  supplies the [`ImageCharacteristics`] and [`ComputeEnvironment`].

Both return a [`ResourceEstimate`] with projected peak memory, wall-clock
milliseconds (already core-scaled in `estimate_in`), and the number of
simultaneously-live intermediate buffers.

## API-shape-compatible with codec-side estimates

`zenpixels-convert` is a **foundation crate** — it sits below codec
abstractions in the workspace dep graph and does NOT depend on `zencodec`.
To keep multi-stage `decode → convert → encode` pipelines ergonomic, the four
types defined here (`ResourceEstimate`, `ComputeEnvironment`,
`ImageCharacteristics`, `SimdTier`) are **shape-compatible** with the
corresponding `zencodec::estimate::*` types — same field names, same builder
method names, same accessor signatures, same `#[non_exhaustive]` discipline —
so a codec author whose stack already uses the zencodec contract can wire
per-stage estimates through with a trivial `From` conversion (a follow-up
will gate that conversion behind a feature flag).

Every field is `Option`, the structs are `#[non_exhaustive]`, and the builders
are growable: future fields land additively without breaking match-bind sites
at the call boundary.

TODO(follow-up): add `From<zencodec::estimate::ResourceEstimate>` (and the
sibling conversions) behind a feature flag so callers at the codec boundary
get a one-liner. Out of scope for the foundation-crate layer; the bridge
lands separately.

## Accuracy contract

Estimates are **best-effort**. The design tolerance is ±30 % vs the
underlying bench numbers; real-world variance comes from:

- **CPU model and SIMD tier.** Calibration data is from the V3 (AVX2) path
  on Ryzen 9 7950X. AVX-512 hosts run faster on a handful of kernels; older
  Zen / Intel / Apple Silicon hosts vary kernel-by-kernel.
  [`ComputeEnvironment::with_simd_tier`] applies a coarse per-tier
  wall-time multiplier on top of the AVX2 baseline (TODO: per-tier
  calibration tables — see the `simd_tier_multiplier` body for the current
  values).
- **Core count.** Wall time is scaled by the effective parallel thread
  count: every plan whose every step is row-parallel divides `wall_ms` by
  `min(compute.cores(), per_step_knee)` where the per-step knee is
  `rows / 64` clamped to `[1, 16]`. Any SERIAL step in the plan disables
  the scaling (single-thread wall is reported).
- **Cache state.** Cold L1/L2 cache adds per-call overhead; the benches
  measure steady-state at 4096-pixel rows, so very small images carry
  proportionally more fixed overhead than the estimate accounts for.
- **Frequency scaling / thermal throttling.** The reference machine is
  water-cooled and runs ~4.5 GHz under sustained load. Boxes that
  thermal-throttle will be slower.
- **Contention.** The estimate assumes a single hot pipeline. Heavy
  concurrent load reduces effective throughput.

Use the estimate for *sizing decisions* (ballpark memory budget, "is this op
cheap or expensive?"), not for tight SLAs.

## Calibration source

Per-pixel cycle costs are baked from the 2026-04-23 benchmark suite at
`zenpixels/benchmarks/`:

- `t1_layout_2026-04-23_baseline.txt` — swizzle, add/drop alpha,
  gray-to-rgb, etc.
- `t2_depth_2026-04-23_baseline.txt` — U8/U16/F16/F32 depth shifts.
- `t3_tf_fused_2026-04-23_baseline.txt` — sRGB/PQ/HLG transfer functions
  (the fused integer-in / linear-out kernels).
- `t4_tf_f32_2026-04-23_baseline.txt` — F32 transfer functions.
- `t6_oklab_2026-04-23_baseline.txt` — Linear RGB ↔ Oklab.
- `t7_gamut_2026-04-23_baseline.txt` — 3×3 gamut matrices.
- `bt2446a_throughput_2026-06-20.md` (zentone) — the HDR→SDR tone-map
  curve.
- `measure_max_throughput_2026-06-19.md` — the SOTA spec-conformant CLL
  reading used by HDR-source scan legs; the default-build SIMD path on
  the 7950X (no `-C target-cpu=native`) delivers **~2.7 Gpix/s**
  steady-state on RGB f32 linear-light.

All steady-state at 4096-pixel rows (L2-resident) on the public AVX2 path
(no `-C target-cpu=native`).

## Threading model

Most `ConvertStep`s are row-parallel (SIMD strip kernels) and contribute a
per-step parallel knee of `rows / 64` clamped to `[1, 16]`. The plan's
overall threading is the bottleneck: if **any** step is SERIAL, the whole
plan is. Otherwise the smallest knee across parallel steps caps the useful
thread count, and `estimate_plan` divides `wall_ms` by
`min(compute.cores(), bottleneck_knee)` directly when building the
[`ResourceEstimate`]. The model is internal — the public surface only
exposes the resulting scaled `wall_ms`.

The two HDR steps (`ToneMapBt2446A`, `SoftCompressOklch`, both gated behind
the `hdr-experimental` feature) read per-image scalars (source peak, max
chroma) and are currently scheduled serially. The per-strip SIMD kernel is
still hot; only the across-strip orchestration is serial. The bias is
toward over-estimate (SERIAL) — if a step's parallelizability is ambiguous
in the future, mark it SERIAL: over-estimating wall time is safer than
under-estimating.

## Memory model

- The output buffer at `to.bytes_per_pixel() * pixels` is always allocated.
- Multi-step plans hold two scratch row buffers ping-ponged between
  intermediate descriptors. Worst-case scratch is
  `2 * width * max_intermediate_bpp` bytes — counted as 2 intermediate
  buffers for paging-pressure reporting.
- The estimate is for a single per-call working set, NOT a parallel-job-wide
  cap.

Intermediate-buffer-count model (reported via
[`ResourceEstimate::intermediate_buffer_count`]):

- Identity plan → 0 (just `dst` memcpy from `src`).
- Single-step plan → 0 (kernel writes `src → dst` directly).
- Multi-step plan → 2 (the two ping-pong scratch halves).
