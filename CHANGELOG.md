# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for
     0.x) release. Add items here as you discover them. Do NOT ship these
     piecemeal — batch them. -->

- **Remove `zenpixels-convert::hdr::HdrMetadata`** (struct + methods) and its
  re-export — deprecated in 0.2.14; superseded by carrying `ContentLightLevel`
  / `MasteringDisplay` directly (or `zencodec::Metadata` at the codec layer).
- **Remove `zenpixels-convert::hdr::{reinhard_tonemap, reinhard_inverse,
  exposure_tonemap}`** and their re-exports — naive global Reinhard + a bare
  `v · 2^stops` clamp, neither display-adaptive (no diffuse-white anchor, no
  peak luminance, no chroma correction); same outlier-driven failure mode that
  got `ContentLightLevel::measure` deprecated in 0.2.15. `#[deprecated]` +
  `#[doc(hidden)]` in this release; production HDR→SDR mapping lives in the
  `zentone` crate (`zentone::Bt2446A` for ITU-R BT.2446 Method A,
  `Bt2408Tonemapper`, ACES, AgX, filmic-spline, gain-map, plus SIMD strip
  processing).
- **Remove the deprecated `ContentLightLevel::measure`** — the literal-maximum
  MaxCLL shipped in 0.2.14 is outlier-sensitive (a single specular/noise pixel
  inflates it, making displays over-tone-map); `#[deprecated]` + `#[doc(hidden)]`
  in 0.2.15, to be replaced by a percentile-aware measure (#54). Delete the
  literal-max method here.
- **Rename `CllMeasure::measure_robust(px, white, method)` →
  `CllMeasure::measure(px, white, method)`** in `zenpixels-convert` at the
  same release that deletes the deprecated 2-arg `zenpixels::ContentLightLevel::measure`.
  The 0.2.x `measure_robust` slot is defect-tolerant via
  [`ContentLightLevel::DEFAULT_PERCENTILE`] (0.9999) — the industry-default
  reading every production HDR tool already uses. Promoting it to the
  obvious `measure` name in 0.3.0 makes the "I don't know which to pick"
  entry point give the production-correct answer. Keep `measure_robust`
  as a `#[deprecated]` alias for one release to ease migration.
- **Demote the unadopted registry *lookup* surface to `pub(crate)`** —
  `zenpixels::registry::{KnownColorSpace, REGISTRY, find_by_cicp,
  find_by_primaries_transfer, find_by_named}` are `#[allow(dead_code)]`
  groundwork (added 0.2.7, #8) for a `from_cicp`/`to_cicp` derivation that never
  landed — those mappings are still hardcoded `match`es in `color.rs`, and the
  lookup surface has zero consumers. `#[doc(hidden)]`'d in 0.2.14. **Disposition:
  demote to `pub(crate)`** (keep the groundwork for the eventual derivation
  refactor), *not* delete — it is unadopted, not replaced. NB the
  matrix-computation half of the same module (`gamut_matrix`, `rgb_to_xyz`,
  Bradford) **is** adopted (`descriptor.rs`) and stays `pub`.
- **Remove the legacy `zenpixels::planar::Plane`** — superseded by `PlaneLayout`
  + separate `PixelBuffer`s (what `MultiPlaneImage` actually holds); zero
  consumers anywhere. `#[doc(hidden)]`'d in 0.2.14; delete here.
- **Drop `OutputMetadata::hdr`** — remove the unwired `Option<HdrMetadata>`
  field; **nothing replaces it.** `OutputMetadata { icc, cicp }` is correct by
  design: it is the lowering target of a codec's *color* plan
  (`zencodec::ColorEmitPlan` is itself `{ cicp, icc }`) and mirrors that shape.
  The HDR content descriptors — content light level, mastering display, and the
  `diffuse_white` anchor — are not color-profile data; they ride the
  codec-boundary carrier `zencodec::Metadata`, which already carries all three
  as sibling fields. (`HdrMetadata`'s zero consumers across `~/work`, and
  zencodec routing around it from the start, confirm it was never on that path.)
- **`zenpixels::ColorAuthority`: add `#[non_exhaustive]`.** The `Cicp`/`Icc`
  authority set can plausibly grow (gain-map / embedded-HDR / merged authority),
  so it shouldn't be a hard exhaustive match. Unlike the struct seals and
  `ConvertError` (both cargo-copter-verified 0-victim and shipped in 0.2.14),
  sealing `ColorAuthority` has **2** measured exhaustive-match victims —
  `zenanalyze` (its own `match`) and `zenfilters` (via
  `zenpixels-convert::output`'s match) — so it batches here, and the `_ =>`
  migrations (`zenanalyze`, `zenpixels-convert::output`) land together at 0.3.0.
- **`zenpixels-convert`: make `fast-transpose` default-on.** It costs ~0 compile
  time (archmage/magetypes are already unconditional deps) + ~36 KB code for a
  1.1–4.7× faster transpose, output bit-identical
  (`benchmarks/orient_fast_transpose_2026-06-18.md`). Flipping a default feature
  batches here; size-sensitive builds opt out via `default-features = false`.

## [0.2.15] - 2026-06-23

### zenpixels-convert — fixed (publish-prep audit)

- **`serde` feature now compiles with `hdr-experimental`.** The
  `serde` Cargo feature previously only forwarded to `zenpixels/serde`
  while `hdr::measure::LightLevelMethod` used
  `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
  — building with `--features serde,hdr-experimental` failed with
  `E0433: cannot find module or crate \`serde\`` because zenpixels-convert
  itself had no `serde` dep. Added `serde = { version = "1.0",
  default-features = false, optional = true, features = ["derive"] }` and
  changed the `serde` feature to `["dep:serde", "zenpixels/serde"]`. Caught
  by `cargo semver-checks` during the 0.2.15 publish audit.
- **`PixelBufferConvertExt::estimate_*` family ships with default impls.** The
  six estimator methods added in 0.2.15 (`estimate_convert_to`,
  `estimate_try_add_alpha`, `estimate_try_widen_to_u16`,
  `estimate_try_narrow_to_u8`, `estimate_linearize`, `estimate_delinearize`)
  were initially trait-required-with-no-default, which `cargo semver-checks`
  caught as a major break (`trait_method_added`) — downstream impls of the
  trait predating 0.2.15 would fail to compile. Each method now has a default
  that returns `ResourceEstimate::zero(EstimateConfidence::Unknown)`; the
  crate's own `PixelBuffer` impl overrides with the real plan-walking
  estimator. Downstream code that already implements `PixelBufferConvertExt`
  for a custom type keeps building unchanged.
- **Workspace `zenpixels` dep bumped to `0.2.15`.** `zenpixels-convert`'s
  `hdr-experimental` surface uses `ContentLightLevel::DEFAULT_PERCENTILE`
  (new in `zenpixels` 0.2.15). The published `zenpixels-convert` 0.2.15
  thus requires `zenpixels ^0.2.15` — publishing must wait until
  `zenpixels` 0.2.15 is on crates.io.
- **All 43 rustdoc warnings cleared.** Escaped `[0,1]` interval notation as
  code spans (~20 sites), unlinked or redirected references to private items
  (`ConvertStep::ExternalTransform`, `ToneMapBt2446A`, `SoftCompressOklch`),
  qualified ambiguous links (`negotiate` was both a function and a module),
  fixed redundant explicit link targets in `hdr/` and `output.rs`, removed
  the stale `PixelSliceMutLoadBearingExt` reference in `load_bearing.rs`.
  `cargo doc --no-deps --document-private-items
  --features hdr-experimental,pipeline,cms-moxcms,rgb` now reports 0
  warnings.

### Workspace — deps

- **Migrate to published `zencodec 0.1.24`; drop the git-rev/path patch.**
  Removed the `[patch.crates-io] zencodec = { path = "../zencodec" }` entry now
  that `zencodec 0.1.24` is on crates.io. The patch only affected the
  `__hdr-e2e-test` transitive graph (no zencodec dep in default builds);
  that graph now resolves zencodec from the registry. `ultrahdr-core` stays
  patched to the imazen git main (still unpublished). No code changes — no
  member implements or reads the `estimate` / `ResourceEstimate` API.

### zenpixels — added

- **`ContentLightLevel::DEFAULT_PERCENTILE = 0.9999`** — public constant for
  the industry-default percentile used by `CllMeasure::measure_robust` in
  `zenpixels-convert`. Surfaced on the type itself (not the trait) so any
  caller — including ones that don't pull `zenpixels-convert` — can refer
  to the same anchor.

### zenpixels — changed

- **HDR measurement (the `measure_*` API surface, `LightLevelHistogram`,
  `LightLevelMethod`, and the SIMD scan kernel) moves to `zenpixels-convert`**
  as the new `measure` module + [`CllMeasure`] extension trait. `zenpixels`
  is now structural-only HDR: `DiffuseWhite`, `ContentLightLevel` (struct +
  `new` + `DEFAULT_PERCENTILE` + the deprecated 2-arg `measure(px, white)`),
  `MasteringDisplay`. Knock-on effects:
  - **`simd` and `avx512` Cargo features removed** from `zenpixels`. The
    `archmage` and `magetypes` optional deps are gone. `cargo tree -p
    zenpixels` shows zero SIMD-runtime dependencies in any configuration.
  - **Migration**: callers using `ContentLightLevel::measure_max(...)` /
    `measure_robust(...)` / `measure_max_smoothed(...)` / `measure_percentile(...)`
    / `measure_histogram(...)` now `use zenpixels_convert::hdr::CllMeasure;` and
    keep the same call syntax (the extension trait preserves
    `ContentLightLevel::measure_*(...)` ergonomics). `LightLevelHistogram`
    and `LightLevelMethod` re-export from `zenpixels_convert::hdr` (also
    gated on `hdr-experimental`).
  - **Rationale**: the SIMD kernels (V3 / NEON / WASM128 / scalar tiers via
    `archmage::magetypes`) and the histogram + percentile logic are pixel-
    iteration code; that lives in `zenpixels-convert` alongside the gamut /
    f16 / orientation kernels that already use the same SIMD machinery.
    `zenpixels` remains a structural interchange crate, which keeps every
    codec consumer (zenjpeg / zenpng / zenavif / zenjxl / zenwebp /
    zentiff / zengif / zencodec) free of optional-feature surfaces and
    SIMD deps for the metadata-pass-through path.

- **`LightLevelHistogram::percentile` interpolates within the bin** (now in
  `zenpixels-convert::measure`) — previously returned the lower edge of the
  bin where the cumulative CDF first crossed the threshold (~0.02 stops /
  ~2 % below the true percentile, ~13 nits at 1015 cd/m²). Now does linear
  interpolation in log2 space within that bin: `fraction = (threshold −
  count_before) / count_in_bin`, `value = 2^(log2_lower + fraction ·
  log2_step)`. Capped at `literal_max_nits` so f32 rounding in the interp
  can never overshoot the spec-literal max by one u16-nits code. On dense
  bright content (1 MP solid) the `measure_percentile` readout now lands
  within ~1 nit of the literal max instead of ~13 nits below.

### zenpixels-convert — added

- **`ConvertPlan::estimate_resources(width, height) -> ResourceEstimate`** —
  predict peak working-set memory + median wall-clock time for a plan
  before running it. Returns
  `ResourceEstimate { peak_memory_bytes, wall_time_ms, breakdown, confidence }`
  where `breakdown` is per-`ConvertStep` (`StepEstimate { name, memory_bytes,
  time_ms }`) and `confidence` is `EstimateConfidence::{Calibrated,
  Heuristic, Unknown}`. Cheap to call (no allocation, no row work — just
  walks the planned steps). Calibration drives from the
  `zenpixels/benchmarks/{t1_layout, t2_depth, t3_tf_fused, t4_tf_f32,
  t5_alpha, t6_oklab, t7_gamut}_2026-04-23_baseline.txt` suite (steady-state
  4096-pixel-row throughput on AMD Ryzen 9 7950X, AVX2/V3 tier) plus the
  zentone `bt2446a_throughput_2026-06-20.md` cell for `ToneMapBt2446A`
  (~250 Mpix/s). Memory model: destination buffer always counted; multi-step
  plans add two row-sized scratch halves sized to the widest intermediate
  bpp; identity plans count the memcpy. Documented tolerance is **±30 %**
  vs the underlying bench numbers — wider on hosts with a different SIMD
  tier, very small images (per-call overhead dominates), or thermal-
  throttled environments. Companion shortcut: **`PixelBufferConvertExt::
  estimate_convert_to(&target) -> ResourceEstimate`** on `PixelBuffer`.
  Steps with no exact bench (BT.709 / Gamma22 F32 OETF/EOTF, premul on F16,
  fused gamut variants without a calibration row) flip the plan's
  `confidence` to `Heuristic` so callers can detect lower-precision cells.
  New public surface (`estimate` module + `ResourceEstimate` /
  `StepEstimate` / `EstimateConfidence` re-exported at crate root) gated
  on neither `hdr-experimental` nor any other feature: ships with default
  features.
- **`ConvertPlan::new_with_hdr_peak(from, to, source_peak_nits)` +
  `ConvertPlan::new_with_hdr_config(from, to, HdrConfig)`** — HDR→SDR
  conversions are now native steps in the standard `ConvertPlan`
  infrastructure (new `ConvertStep::ToneMapBt2446A` and
  `ConvertStep::SoftCompressOklch` variants), composed alongside the
  existing transfer / depth / gamut steps. The plan still runs through the
  same `RowConverter` / `convert_row` pipeline as every other conversion
  in the crate — no parallel API surface. **`HdrConfig { source_peak_nits,
  target_peak_nits: 100, gamut_knee: 0.9 }`** carries the BT.2446-A and
  soft-compress knobs. Wide-gamut output mode (target primaries = BT.2020)
  skips the post-curve gamut matrix and the OKLch compress step
  automatically. The end-to-end producer-SDR ΔE2000 against the imazen-26
  UltraHDR sample is **3.16** (matches the deleted `HdrToSdr` pipeline).
  Gated behind the existing `hdr-experimental` feature.
- **`PixelBufferHdrConvertExt`** — extension trait on `PixelBuffer` with
  `convert_to_sdr(target)` (auto-measures source peak via
  `CllMeasure::measure_robust`) and `convert_to_with_hdr_config(target,
  HdrConfig)` (explicit knobs). Both route through the new
  `ConvertPlan::new_with_hdr_config` path. Gated behind `hdr-experimental`.
- **Eleven new `estimate_*` companions on the convert extension traits.**
  Each of the previously-allocating public methods now has a non-allocating
  estimator that returns a [`ResourceEstimate`] sharing the same ±30 %
  design tolerance and confidence semantics as the
  [`ConvertPlan::estimate_resources`] primitive shipped in commit
  `183af66`. On `PixelBufferConvertExt`: `estimate_try_add_alpha`,
  `estimate_try_widen_to_u16` (same-channel-type → identity-memcpy fast
  path), `estimate_try_narrow_to_u8` (same fast path), `estimate_linearize`,
  `estimate_delinearize(transfer)`. On `PixelBufferConvertTypedExt`
  (`rgb` feature): `estimate_to_rgb8`, `estimate_to_rgba8`,
  `estimate_to_gray8`, `estimate_to_bgra8` — each estimates the underlying
  type-erased `convert_to(&PixelDescriptor::*)` call (the typed wrapper
  itself is a zero-cost cast). On `PixelBufferHdrConvertExt`
  (`hdr-experimental`): `estimate_convert_to_with_hdr_config` (delegates to
  [`ConvertPlan::new_with_hdr_config`]) and `estimate_convert_to_sdr` —
  the latter accounts for **all three legs** of `convert_to_sdr`: source
  linearization, the
  [`CllMeasure::measure_max`](crate::hdr::CllMeasure::measure_max) scan
  (new calibration constant from `benchmarks/measure_max_throughput_2026-06-19.md`,
  default-build SIMD column at 2048² = **2735 Mpix/s** on Ryzen 9 7950X
  AVX2), and the downstream HDR plan. The estimator sums wall-clock
  across legs, takes `max(peak_memory_bytes)` (linear scratch frees
  before the HDR plan allocates), concatenates breakdowns with a new
  `"MeasureMaxCll"` step, and reports the most-conservative confidence
  tier across legs. Non-HDR sources short-circuit to plain
  `estimate_convert_to(target)`, matching the runtime path. Tests cover
  delegation equivalence (peak_memory exact, wall_time within 1 %) plus
  the measure_max-scan accounting on a 4 MP buffer.

### zenpixels-convert — removed

- **`zenpixels_convert::hdr::HdrToSdr`** and its `buffer_dispatch.rs` glue
  — ~2,400 LOC of parallel API surface folded into the standard
  `ConvertPlan` infrastructure (above). Net deletion: ~1,300 LOC. The
  underlying math primitives (`Bt2446A`, `SoftCompress`,
  `GamutBoundaryLut`) stay public for advanced callers who want to drive
  the curves directly.

### zenpixels-convert — changed (BREAKING, gated by `hdr-experimental`)

- **`ConvertPlan::new` now refuses HDR-encoded → SDR-encoded conversions**
  (`Pq` / `Hlg` source → `Srgb` / `Bt709` / `Gamma22` target) with a new
  `ConvertError::HdrSourceRequiresPeak` variant, and points the caller at
  `ConvertPlan::new_with_hdr_peak`. Pre-refactor the plain plan silently
  routed those through a linear intermediate with no tone-mapping,
  saturating every HDR sample above SDR diffuse-white to peak. HDR→Linear
  paths (lossless transfer decode) stay allowed via the regular `new`
  entry. Only fires under `hdr-experimental`; without the feature the
  historic pass-through behavior is preserved.

- **`HdrToSdr::convert_buffer` / `convert_into`** — buffer-level dispatch
  wrappers around the existing linear-light `apply_strip` primitive. The
  buffer methods auto-handle transfer (Linear / sRGB / BT.709 / PQ / HLG /
  Gamma22), channel format (u8 / u16 / f16 / f32 × RGB | RGBA), alpha mode
  (Straight / Premultiplied / None with α==0 short-circuited to RGB=0),
  signal range (Full / Narrow, linear-space approximation), and source
  diffuse-white anchor (buffer's `ColorContext::diffuse_white` wins over the
  struct's stored value via the new `HdrToSdr::with_source_diffuse_white_nits`
  setter). Wraps `RowConverter` for transfer + format + alpha + primaries;
  signal-range handling is approximate in linear space (no Narrow↔Full
  kernels exist in zenpixels-convert yet) and will tighten when those land.
  The strip-level `apply_strip` math is unchanged — only the byte-format
  dispatch around it is new. End-to-end producer-SDR match against a real
  UltraHDR sample still lands at ΔE2000 = 3.16 (≤ 5.0 budget). Gated behind
  the existing `hdr-experimental` feature.
- **`zenpixels_convert::hdr::Bt2446A`**, **`HdrToSdr`**, **`SoftCompress`**,
  and **`GamutBoundaryLut`** — gated behind the existing `hdr-experimental`
  Cargo feature. Extracted into a single home so the canonical HDR → SDR
  pipeline lives next to the CLL measurement primitives that feed it:
  `HdrToSdr::new(source_peak_nits)` composes a BT.2020 → BT.709 matrix step,
  the BT.2446 Method A tone curve, and an OKLch soft chroma-compression
  knee into one strip API. `Bt2446A` and the SIMD strip kernel `bt2446a_tier`
  (degree-7 monomial BT.1886 EOTF approximation, Estrin's-method evaluation)
  moved from `zentone::Bt2446A`. `SoftCompress` and `GamutBoundaryLut` moved
  from `zenfilters::gamut_lut::GamutBoundaryLut`. Both upstream copies are
  removed; this crate is the single source of truth.
  trait, gated behind the new **`hdr-experimental`** Cargo feature (default-
  off). The new home for HDR content-light-level measurement, with the
  `LightLevelHistogram` and `LightLevelMethod` types alongside. Implementing
  `CllMeasure` for `zenpixels::ContentLightLevel` means callers write
  `use zenpixels_convert::hdr::CllMeasure; ContentLightLevel::measure_robust(...)`
  — the same call signature the methods had on `ContentLightLevel` directly
  in pre-publication 0.2.15 main. The feature flag covers expected pre-0.3.0
  shape churn (the queued `measure_robust` → `measure` rename when the
  deprecated 2-arg `measure(px, white)` ships its 0.3.0 removal, plus
  potential trait-vs-free-fn restructuring); the scan kernels and accuracy
  contracts underneath are stable.

  Full surface ([#54](https://github.com/imazen/zenpixels/issues/54)):

  - **`CllMeasure::measure_max(px, white, method)`** — CTA-861.3 spec-strict
    (literal MaxCLL + arithmetic mean). The hot path for delivery-mandate
    HDR metadata; runs at **2.5-3.4 Gpix/s** with the V3 / AVX2 tier of the
    `archmage::magetypes` SIMD kernel on a Ryzen 9 7950X, **above libplacebo**.
    Three accuracy tests pin against an independent f64
    Psychtoolbox-3 / x265 / libplacebo / Dolby Vision L1 / libultrahdr
    oracle. Reproducer + full table in
    [`benchmarks/measure_max_throughput_2026-06-19.md`](benchmarks/measure_max_throughput_2026-06-19.md).

  - **`CllMeasure::measure_robust(px, white, method)`** — the recommended-
    default reader, equivalent to `measure_percentile(_, _,
    ContentLightLevel::DEFAULT_PERCENTILE, _)` with `DEFAULT_PERCENTILE` =
    0.9999. Drops the top 0.01 % of pixels as the outlier budget — the
    production-correct answer for content with possible defect-driven hot
    pixels (sensor noise, stuck pixels, specular blowouts). Matches what
    libplacebo (`pl_color_space_infer`), DaVinci Resolve, x265's
    `--master-display` helpers, and most HDR10+ mastering tools do.
    Tests cover the alias contract (bit-exact vs explicit
    `measure_percentile`), the defect-rejection path on 100k pixels with
    one stuck pixel, and the sparse-bright cliff (1-in-100 bright pixel
    dropped at small sizes — astrophotography should pick `measure_max`).

  - **`CllMeasure::measure_max_smoothed(px, white, method)`** — MaxCLL via
    a 3×1 horizontal box-filtered max. Robust against single-pixel defects
    and math-weird outliers without committing to an explicit percentile;
    closer in shape to Dolby Vision L1's per-block analysis. A single
    stuck pixel at 10 000 cd/m² in a 0.005 cd/m² background reads as ~3 333
    after the box filter; clusters of ≥3 pixels preserve at full magnitude.
    MaxFALL stays the literal arithmetic mean (linearity of expectation;
    CTA-861.3 unchanged). Single-pass streaming scan auto-vectorises to
    ~1.0-1.3 Gpix/s on Zen 4 (2-3× faster than the histogram path).
    Reproducer + table in
    [`benchmarks/measure_max_smoothed_throughput_2026-06-19.md`](benchmarks/measure_max_smoothed_throughput_2026-06-19.md).

  - **`CllMeasure::measure_percentile(px, white, percentile, method)`** —
    explicit-percentile convenience. No default — caller commits per
    content policy (sparse-bright legitimate content like astrophotography
    / fireworks / candle-in-dark-room has opposite needs from defect
    rejection). Issue #54's docstring spells out the trade-off.

  - **`CllMeasure::measure_histogram(px, white, method)`** — builds a
    [`LightLevelHistogram`] in a single SIMD-friendly pass via a tiered
    `#[archmage::magetypes]` kernel (V3 / NEON / WASM128 / scalar) with
    `f32x8::log2_midp` for the per-pixel log2, eight per-lane sub-histograms
    (32 KiB total L1-resident) to side-step the cross-lane scatter
    conflict, and SIMD `reduce_max` / `reduce_add` for the running max +
    f64 sum at end-of-row. ~470-490 Mpix/s on Zen 4. The histogram is the
    primitive — callers can derive `max()`, `mean()`, `percentile(p)`, and
    arbitrary CDF readouts after one scan. Reproducer +
    perf-bottleneck notes in
    [`benchmarks/measure_histogram_throughput_2026-06-19.md`](benchmarks/measure_histogram_throughput_2026-06-19.md).

  - **[`LightLevelHistogram`]** — log-scale histogram of per-pixel light
    levels over `[0.005, 10000]` cd/m² across 1024 bins (~0.02 stops/bin,
    well below the cone JND; 4 KiB cache-resident state). Exposes `max()`,
    `mean()`, `percentile(p)` (with linear interpolation in log2 space —
    see the `zenpixels — changed` note above), `method()`, `total_pixels()`,
    `bins()` for custom CDF walks.

  - **[`LightLevelMethod`]** enum (`MaxRgb` default, `LuminanceBt2020`) —
    explicit per-pixel reduction choice. CTA-861-G Annex P leaves this
    normatively open; `MaxRgb` matches x265 / DaVinci / Dolby Vision L1 /
    Psychtoolbox, `LuminanceBt2020` matches some Netflix / Apple TV+
    pipelines. `#[non_exhaustive]`.

  Tests cover: defect-spike (lone outlier → p99 drops it, spec-literal
  keeps it); astrophotography night-stars (1100 px with 100 stars →
  spec-literal & p99.99 keep them); BT.2020 vs MaxRgb method divergence on
  saturated red; percentile boundary handling (0, 1, NaN, out-of-range);
  bin interpolation precision (1 MP solid → within ~1 nit of literal max);
  bin interpolation cap (no overshoot of literal max under f32 rounding);
  3×1 box-filter suppression / cluster preservation / mirror-pad edge /
  degenerate widths; `DEFAULT_PERCENTILE` constant pin; input rejection
  contract.

### zenpixels — fixed

- **`PlaneDescriptor::plane_width` / `plane_height` no longer panic with
  divide-by-zero on `h_subsample == 0` / `v_subsample == 0`** (`planar.rs`,
  [#43](https://github.com/imazen/zenpixels/issues/43) item 1). The
  subsample fields are public, so any code path that struct-literals a
  `PlaneDescriptor` or maps an out-of-range codec subsampling code to
  factor 0 could set them to 0 — the next `plane_width` call then
  panicked in release. Now `0` is treated as `1` (full resolution) at the
  use-site, so production servers never see the abort. The
  [`with_subsampling`](https://docs.rs/zenpixels/0.2/zenpixels/struct.PlaneDescriptor.html#method.with_subsampling)
  builder's `debug_assert!` still catches the misuse in dev / test.
  Non-breaking; signature unchanged.
- **`PixelDescriptor::aligned_stride` / `simd_aligned_stride` overflow-safe
  on 32-bit targets (i686, wasm32)** (`descriptor.rs`,
  [#43](https://github.com/imazen/zenpixels/issues/43) item 2). The plain
  `width as usize * bpp` multiply could wrap silently on 32-bit `usize` —
  producing a small stride that passed the subsequent `checked_mul(height)`
  guard and yielded an under-sized allocation paired with the full-size
  `width` field (silent buffer corruption / deferred OOB, instead of the
  documented `InvalidDimensions` typed error). Both methods now use
  `saturating_mul`, and the `align_up_general` helper saturates too so
  the upstream guard isn't undone by a wrap in `value + (align − rem)`.
  Strides saturate to `usize::MAX`, which downstream `checked_mul` then
  surfaces as `InvalidDimensions`. Matches the `PixelBuffer::from_pixels`
  guard that already had this treatment. Non-breaking.

### zenpixels-convert — fixed

- **`hdr::HdrToSdr` pipeline order — corrected.** The shipped pipeline
  applied the BT.2020 → BT.709 gamut matrix *before* the BT.2446 Method A
  tone-map. That was wrong: ITU-R BT.2446-1 specifies the curve in
  BT.2020 R'G'B' / Y'Cb'Cr', so its luma weights
  (`0.2627 / 0.6780 / 0.0593`) and YCbCr↔RGB coefficients are BT.2020-
  specific. Feeding the curve BT.709 RGB produced a systematic hue shift
  on saturated content. The new order is **`(source → BT.2020 if needed)
  → Bt2446A → BT.2020 → BT.709 matrix → SoftCompress (BT.709 OKLch)`**:
  the curve sees the BT.2020 input it was derived against, then the
  gamut step projects into the BT.709 working space, and soft chroma
  compression keeps the final pixels hue-preserved in unit cube. Verified
  on a 12 MP imazen-26 UltraHDR sample: mean ΔE2000 = **3.16** against
  producer SDR — matches zentone's published BT.2446-A median of 3.17.
- **`hdr::HdrToSdr` accepts arbitrary source primaries** via the new
  `with_source_primaries(source_peak_nits, source_primaries)` and
  `with_params(source_peak_nits, source_primaries, target_peak_nits,
  gamut_knee)` constructors. When `source_primaries != ColorPrimaries::Bt2020`
  the pipeline inserts a `source → BT.2020` matrix step (cached at
  construction) before the BT.2446 curve so the curve still sees the
  BT.2020 RGB it was designed for. `HdrToSdr::new(...)` keeps the
  BT.2020-source default. Supports Display P3 HDR (e.g. Apple ProRAW)
  and BT.709 HDR (UltraHDR JPEG / HEIC gain-map reconstruction outputs)
  without distorting the curve.
- **New color-space verification tests in
  `src/hdr/hdr_to_sdr.rs::tests`** pin the math against the pipeline-
  order regression: `neutral_grey_stays_neutral`,
  `hdr_diffuse_white_maps_to_sdr_diffuse_white`,
  `saturated_bt2020_{red,green,blue}_lands_*_dominant_in_bt709`,
  `p3_source_pipeline_produces_neutral_grey_from_p3_grey`,
  `bt709_source_is_identity_matrix_to_bt2020`,
  `pipeline_order_regression_test` (compares OLD vs NEW order — they
  must differ measurably), and the SIMD parity property test was
  extended to run under each of BT.2020 / BT.709 / Display P3 source
  primaries.
- **New `__hdr-e2e-test` Cargo feature** gates an end-to-end producer-
  SDR ΔE2000 regression test
  (`tests/hdr_producer_sdr_match.rs`) that decodes a known imazen-26
  UltraHDR sample, runs HDR through the pipeline, and pins mean ΔE2000
  < 5.0 against the producer's SDR base. The feature pulls `zenjpeg` +
  `anyhow` as test-only deps via the workspace's `[patch.crates-io]`
  table (which routes `ultrahdr-core` to the imazen git main; `zencodec`
  resolves from crates.io now that 0.1.24 is published). The test
  silently skips when the imazen-26
  corpus is absent (CI), and MUST be run on `lilith`'s machine before
  merging any pipeline-touching change. Invocation:
  `cargo test -p zenpixels-convert@0.2.15 --features __hdr-e2e-test
  --test hdr_producer_sdr_match -- --nocapture`.

### zenpixels-convert — changed

- **`hdr::HdrToSdr` now takes `PixelDescriptor` for source and target**
  (replacing the previous `ColorPrimaries` parameters). `PixelDescriptor`
  carries primaries, transfer function, format, alpha mode, and signal
  range — the descriptors are the single source of truth for buffer color
  metadata. Required transfer is `Linear` for both ends (validated at
  construction); encoding the output to BT.1886/sRGB/PQ/HLG remains the
  caller's job for now. This is a breaking change inside the
  `hdr-experimental` gate. New surface:
  ```
  HdrToSdr::new(source, target, source_peak_nits)
  HdrToSdr::with_params(source, target, source_peak_nits, target_peak_nits, gamut_knee)
  ```

### Earlier 0.2.15 prep (2026-06-18 commit batch)

The items below were the initial cut of 0.2.15 prepared on 2026-06-18; they ship
in the same 0.2.15 release as the larger blocks above.

#### zenpixels — added

- **Buffer-level colour-relabel builders** on `PixelSlice` / `PixelSliceMut` /
  `PixelBuffer`: `with_diffuse_white(DiffuseWhite)`, `with_cicp(Cicp)`,
  `with_icc(impl Into<Arc<[u8]>>)`. **Reinterpretation only** — pixels untouched;
  each clones any existing `ColorContext`, overrides the one field, and
  re-attaches, matching the existing `with_transfer` / `with_primaries` pattern
  so the anchor (and CICP/ICC) is reframed as ergonomically as the transfer.
  Plus `ColorContext::with_cicp` / `with_icc` to match `with_diffuse_white`.

#### zenpixels — deprecated

- **`ContentLightLevel::measure` is deprecated and `#[doc(hidden)]`.** It computes
  the *literal* CTA-861.3 MaxCLL — the absolute maximum over all pixels of
  `max(R, G, B)` — which is outlier-sensitive: a single specular highlight or
  noise/stuck pixel inflates MaxCLL, and displays then over-tone-map (dimming the
  image). Production HDR-metadata generators use a high percentile (~99.99th)
  instead. A percentile-aware replacement is tracked in
  [#54](https://github.com/imazen/zenpixels/issues/54); the literal-max method is
  queued for removal in 0.3.0. MaxFALL (the mean) was never affected, and there
  were no consumers — nothing breaks.

#### zenpixels-convert — deprecated

- **`hdr::{reinhard_tonemap, reinhard_inverse, exposure_tonemap}` are deprecated
  and `#[doc(hidden)]`.** Naive global Reinhard `v / (1 + v)` and a bare
  `v · 2^stops` clamp; neither uses a diffuse-white anchor, peak-luminance
  metadata, or chroma correction. On a 1000-nit HDR pixel Reinhard returns
  ~1.0, which writes out as full-scale SDR — same outlier-driven failure mode
  that got `ContentLightLevel::measure` deprecated. Use the `zentone` crate
  (`zentone::Bt2446A` for the ITU-R BT.2446 Method A display mapping;
  `Bt2408Tonemapper`, ACES, AgX, filmic-spline, gain-map curves, plus SIMD
  strip processing) for production HDR→SDR work. Removal queued for 0.3.0.

#### zenpixels-convert — internal

- **`orient`: fixed public-doc links + corrected the `fast-transpose` comment.**
  The module-level docs linked private `transpose_*` fns (dead links on docs.rs)
  → converted to code spans. The `fast-transpose` comment falsely claimed it
  "pulls archmage/magetypes" — those are unconditional deps, so the feature is
  ~free to compile (+36 KB code) for a 1.1–4.7× faster transpose; see
  `benchmarks/orient_fast_transpose_2026-06-18.md`.
- **`bench_orient`: decoupled `__bench_orient` from `fast-transpose`** so it A/Bs
  the scalar tiled gather vs the SIMD kernels (`--features __bench_orient` vs
  `,fast-transpose`); the staged shim + its parity test now require both features.

## [0.2.14] - 2026-06-18

### zenpixels — docs

- **`docs(readme)`: document the `from_vec` raw-bytes+stride codec-interop
  path.** An insulated external-developer usability test (README only) found
  the #1 codec constructor — `PixelBuffer::from_vec` wrapping a decoded
  `Vec<u8>` without copying — was named in prose but never shown, and the
  stride **unit** (bytes, not pixels/elements) was unstated anywhere. Added a
  worked `from_vec` example (tight stride) plus the explicit-stride
  `PixelSlice::new` borrow path, tabulated every predefined constant's full
  descriptor (notably the **alpha mode** column — `None` vs straight vs the
  `X`-format `Undefined` padding lane), added the canonical crate-root `use`
  block for the `PixelDescriptor` field types, and documented the `width()` /
  `height()` / `stride()` accessors and the `descriptor()` by-value return /
  `At<BufferError>` error type.

### zenpixels-convert — docs

- **`docs(readme)`: add a crate-local `README.md`.** The crate previously set
  `readme = "../README.md"` (the combined workspace README), but that file
  lives outside the package root and so was never packaged — crates.io showed
  no README for `zenpixels-convert`. Added a focused conversion-API README,
  pointed `readme = "README.md"`, and added `/README.md` to `include` so it
  ships with the crate.

### zenpixels — added

- **`hdr::DiffuseWhite`** — a typed absolute-luminance anchor: the cd/m²
  (nits) that relative-linear `1.0` represents (OpenEXR `whiteLuminance` /
  JPEG XL `intensity_target` / libheif `ndwt` / libplacebo SDR-white). Newtype
  on purpose — HDR mixes nits, PQ-encoded `[0,1]`, log2 gain, and headroom
  ratios. `DiffuseWhite::BT2408` (203, the cross-vendor default) is the
  `Default`.
- **`ContentLightLevel::measure(PixelSlice, DiffuseWhite) -> Option<Self>`** —
  MaxCLL/MaxFALL (CTA-861.3-A stills) from relative-linear RGB(A) f32, a
  constructor on the type it returns (replacing the withdrawn
  `zenpixels-convert::hdr::compute_content_light_level` free function).
  Negative/NaN clamp to 0, alpha ignored, strided rows handled; `None` for
  non-`RgbF32`/`RgbaF32` or non-`Linear` input.
- **`ColorContext::diffuse_white` field + `with_diffuse_white` builder, and
  `DiffuseWhite` re-exported at the crate root** — the absolute-luminance
  anchor now travels on the existing `PixelBuffer.color: Arc<ColorContext>`
  sidecar (propagated by every clone / `with_descriptor` at no per-strip cost),
  so the HDR converter can map relative-linear light to absolute luminance for
  PQ/HLG encode and tone-mapping. `None` = unsignaled (consumers default to
  `DiffuseWhite::BT2408`, 203). First slice of the M×N HDR pipeline (#45).

### zenpixels — fixed

- **`no_std` build of `ContentLightLevel::measure`** — `nits_to_u16` called
  `f64::round`, which lives in `std` (libm); the crate is `no_std`, so
  `--no-default-features` failed to compile (`E0599`). Replaced with a
  saturating round-half-up on the non-negative nits domain — byte-identical
  output, no libm. Caught by the feature-powerset CI job before release.

#### Changed (BREAKING, tolerated in 0.2.x)

- **`ColorContext` is now `#[non_exhaustive]`** (`PartialEq` **and `Eq` both
  retained**). `#[non_exhaustive]` lets the next HDR carrier fields (mastering
  display, content light level) extend it without another break. **`Eq` is
  preserved**: `DiffuseWhite` wraps `f32` but implements a bit-exact `Eq` (a
  luminance anchor is never `NaN`/`-0.0`, so a bitwise compare is reflexive and
  consistent), so adding the `diffuse_white` field did not cost `ColorContext`
  its `Eq` — avoiding a removed-trait break vs 0.2.13. Tolerated per the 0.2.x
  policy: the only external struct-literal constructor is `zencodec`, which
  builds via `ColorContext::from_cicp`/`from_icc` (not a literal), so sealing it
  is harmless, and no in-tree `Eq`/`Hash` use depends on `ColorContext`
  (verified). `cargo semver-checks` now reports a single remaining major item
  (`struct_marked_non_exhaustive`) — a tolerated-bucket item; the `Eq`-removal
  failure is gone.
- **Three more types are now `#[non_exhaustive]`: `planar::MultiPlaneImage`,
  `planar::PlaneLayout`, and `buffer::InPlacePixels`** (the last gains
  `InPlacePixels::new`) — only the genuinely-extensible ones (planar geometry /
  YUV layouts grow; the transform bundle accretes fields). Each keeps its valid
  `Eq`/`Hash`; construction via `::new`/builders. **cargo-copter (all 21
  published reverse-deps): 0 victims** — nobody downstream struct-literals or
  exhaustively-matches them; `InPlacePixels` is transform-internal (no published
  constructor).
- **The fixed-spec data bags stay open** — `hdr::ContentLightLevel`
  (CTA-861.3), `hdr::MasteringDisplay` (SMPTE ST 2086), `planar::PlaneDescriptor`,
  `planar::PlaneMask`, and the POD pixel types `Bgrx`/`Rgbx`/`GrayAlpha*`.
  Struct-literal is their natural interface and the field sets are standardized
  /won't grow, so sealing would be breakage risk for ~zero future-proofing (the
  `Rect`/`Size` carve-out). `registry::KnownColorSpace` (+ `REGISTRY`,
  `find_by_*`) and the legacy `planar::Plane` are now `#[doc(hidden)]` — dead /
  positioned-for-future public API with no consumers, pending 0.3.0 removal (see
  QUEUED). `ColorAuthority` is likewise deferred to 0.3.0.
- **`zenpixels-convert::ConvertError` is now `#[non_exhaustive]`, with a new
  `Buffer(zenpixels::BufferError)` variant + `From<BufferError>`** — promoted
  from the 0.3.0 queue on cargo-copter evidence: sealing it produced **0
  `E0004` regressions** across zpc's published reverse-deps (a `non_exhaustive`
  enum only breaks exhaustive `match`es lacking a `_` arm, and nobody matches
  `ConvertError` that way). The seal being free, the `Buffer` variant rides
  along (adding a variant to an already-sealed enum is not a break). The 8
  `PixelBuffer::{try_new,from_vec}` construction sites now
  `map_err_at(ConvertError::from)`, so a `StrideTooSmall` / `InvalidDimensions`
  layout error keeps its real cause instead of being mislabeled
  `AllocationFailed` (OOM) — the *classification* half of #52 (the
  trace-preservation half landed earlier).

### Workspace — build

- Public-API snapshots migrated to the `zenutils-apidoc` 0.1.0 runner package
  at `apidoc/` (workspace-excluded, CI-free): three snapshot files per crate
  under `docs/public-api/`, regenerated via `just api-doc`. Replaces the
  in-crate `zenpixels/tests/public_api_doc.rs` copy, its `serde_json` dev-dep,
  and every `ZEN_API_DOC` / cargo-public-api trace in CI.

### zenpixels-convert — performance

- **The SIMD transpose kernels are now behind an opt-in `fast-transpose`
  Cargo feature** (default-off). Without it, `apply_orientation*` use the
  portable scalar tiled/blocked paths (correct, slower); enabling
  `fast-transpose` pulls in the x86-64 AVX2 and aarch64 NEON kernels. The
  ~3k lines of kernels were split out of `orient.rs` into
  `orient/{pxn_x86,pxn_neon,rgb3_x86}.rs` (feature- and arch-gated);
  `orient/mod.rs` keeps the public API, dispatch, and always-on scalar
  core. The corruption gate (`orient/tests.rs`) now validates both the
  scalar default path and the SIMD path. The `transpose-shootout`
  measurement crate moved to `benchmarks/transpose-shootout/`
  (workspace-excluded, dev-box only).

- **NEON transpose kernels for all eight pixel widths + 12/16-byte
  maximization on both architectures** (zenjpeg#150, continued): aarch64
  tiers (vzip cascades, tbl expand/compress, tuple/block-move kernels —
  all `forbid(unsafe_code)`) developed and hardware-validated on a
  Neoverse-N1 Hetzner box; they **beat or tie the C++ Simd library's NEON
  tier at every width it supports** at 12MP (Rotate90 4ch +43%, 3ch +25%,
  1ch +14%). x86 RGBF32 gains a 3-store intra-run-slop path (+34% over
  fast_transpose, was +17%); RGBAF32 goes all-32-byte loads/stores
  (−7% vs fast_transpose's raw-pointer loops — the documented residual).
  Remaining ARM gaps vs fast_transpose's unsafe NEON kernels (1/4/6-16
  bpp, 8–35%) are recorded with next designs in
  `docs/transpose-research-2026-06-11.md`; records in
  `benchmarks/transpose_shootout_{x86_final,arm_r1,arm_final}_2026-06-12.*`.

- **x86-64 SIMD transpose kernels for 1/2/3/4-byte pixels — up to 9.5×
  over the tiled gather, matching the C++ Simd library at 12MP in most
  cells** (zenjpeg#150 follow-through; all `#![forbid(unsafe_code)]` via
  archmage value-mode intrinsics + safe_unaligned_simd reference wrappers,
  x86-64-v3/AVX2 baseline with scalar/NEON-magetypes fallbacks unchanged):
  gray8 16×16 SSSE3 cascade with full-width row-band sweep; 2-byte 8×8
  cascade; RGB8 AVX2 8-row expand 3→4 / dword-transpose / compress with
  contiguous 24-byte stores (the only SIMD RGB24 transpose in the Rust
  ecosystem); 4-byte AVX2 8×8; column-stripe macro-blocking (64 dst rows
  written to completion) on the 2/3/4-byte kernels; per-kernel measured
  band/stripe iteration order under reflections. 12MP Rotate90 RGB8:
  87.2 → 9.2 ms across the two sessions; Transpose gray8 9.2 → 4.4 ms
  (ties zune-imageprocs, the prior fastest Rust impl). Full cross-library
  shootout records in `benchmarks/transpose_shootout_*_2026-06-12.*`;
  measurement harness in `transpose-shootout/` (workspace-excluded).
  Remaining known gaps vs the C++ reference, with designs noted in
  `docs/transpose-research-2026-06-11.md`: gray8 ~2× (their AVX2 16×16),
  Rotate90 1/2/3ch −14…−37%, and ~2× at L2-resident sizes (per-tile
  overhead); non-temporal stores blocked on a safe archmage wrapper.

- **Transposing orientations (`apply_orientation*`) are ~1.8–7× faster for
  every non-4-byte pixel size** (zenjpeg#150 part a): 1/2/3/6/8/12/16-byte
  pixels now route through a monomorphised cache-blocked gather
  (`transpose_tiled::<BPP>`) that precomputes the orientation's separable
  inverse map per destination row — sequential destination writes, ±stride
  source stepping, one fixed-size copy per pixel — replacing the per-element
  `forward_map` + `row_mut` + variable-length copy. RGB8 Rotate90 12MP:
  87.2 → 48.1 ms (now also ~1.5× faster than zenjpeg's internal naive
  gather, unblocking full delegation); RGB8 Transpose 1024²: 4.36 → 0.85 ms
  (Ryzen 9 7950X, `benchmarks/orient_tiled_gather_2026-06-11.txt`). 4-byte
  pixels keep the SIMD register transpose; the `forward_map` scatter remains
  the parity oracle (new gate:
  `tiled_transpose_matches_blocked_reference_across_bpp`) and the fallback
  for future `#[non_exhaustive]` orientation variants.

### zenpixels-convert — changed

- **The absolute-luminance anchor now threads through the PQ `ConvertStep`s
  themselves (#45 S2).** The PQ kernels (u16 + f32, encode + decode) take a
  `diffuse_white / 10000` scale carried on the plan and apply it to the **RGB
  lanes only**: encode multiplies relative-linear into PQ-absolute before the
  OETF, decode divides after the EOTF, so a relative-linear buffer maps to PQ at
  the right brightness with no caller pre-scale. The unsignaled default is `1.0`
  — "linear is already PQ-absolute (1.0 = 10000 cd/m²)", the exact prior
  behavior. HLG is intentionally excluded (scene-referred anchoring differs).

- **The PQ kernels (u16 + f32, encode + decode) now use a vendored *precise*
  SIMD exact-ST 2084 transfer**, alpha-preserving. `pq_eotf_slice` /
  `pq_oetf_slice` evaluate the exact SMPTE ST 2084 formula in SIMD via magetypes'
  `pow_midp_precise`. This replaces **both** the prior scalar u16 path **and** the
  `linear_srgb::default` rational-poly slice: that fit is only valid above
  v≈0.02 and, applied as a slice (no exact-below-threshold branch), extrapolated
  to black and drifted the tight u16 → f32 → u16 round-trip up to **256 codes**.
  The vendored kernel is precise across the full range (round-trip **≤1**, ST
  2084 oracle **±1**) and stays SIMD. u16↔f32 depth scaling uses the SIMD
  `garb::bytes` primitives; the EOTF/OETF run over every lane and the alpha lane
  is then restored linearly (never transferred or anchored). The RGB-only anchor
  multiply is `multiply_color_channels`, a generic `#[magetypes]` `f32x16` method
  (with-alpha `[f,f,f,1]` pattern / without-alpha uniform).

- **The no-allocation convert path honors a strided destination.**
  `convert_into_with_anchor` (and `quantize_into`) take a `dst_stride`, writing
  each row at the caller's stride instead of assuming packed output — so the
  result can land in a sub-region of a larger buffer, and the `BufferSize` check
  validates `dst_stride ≥ row` + `(rows-1)·dst_stride + row` bytes.

- **`quantize_to` no longer repacks; it honors strides and preserves alpha.**
  It hands the (possibly strided, possibly RGBA) source straight to the anchored
  pipeline — no caller-side pre-scale or contiguous repack. Alpha follows the
  **target**: an RGB PQ target drops it (as before), an RGBA PQ target
  (`RGBA16.with_transfer(Pq)…`) preserves it linearly. Codes still match the f64
  ST 2084 oracle within ±1. Internally `convert_buffer_with_anchor` is now
  strided and split over a `convert_into_with_anchor` primitive; a `pub(crate)`
  `quantize_into` (no-allocation — writes into a caller buffer) is staged behind
  it, ready to promote when a concrete consumer or the §3.2 public HDR-convert
  surface lands.

- **`quantize_to` now carries the diffuse-white anchor onto its output.** The
  result `PixelBuffer` gets a `ColorContext` with the applied `diffuse_white`
  (the `ndwt` a downstream encoder signals) plus the target's CICP, instead of a
  context-less buffer — the anchor is a *reference* that survives the encode, so
  the output self-describes it rather than silently dropping it. (`quantize_into`
  writes raw bytes, so its caller owns the envelope.)

- **HLG↔PQ now refuses at plan time (`ConvertError::NoPath`)** instead of
  emitting wrong pixels. The HLG kernels carry only the scene-referred OETF (no
  OOTF, no `Lw`), while PQ is absolute display light — routing one to the other
  through the shared linear intermediate conflates the two luminance domains
  (deterministic but photometrically wrong brightness). `ConvertPlan::new` (and
  everything above it) refuses the cross, the same posture as the signal-range
  refusal; HLG↔SDR/linear and PQ↔SDR/linear are unaffected. Correct HLG↔PQ needs
  the OOTF + `(diffuse_white, Lw)` threading (#45 S2). Uses the existing `NoPath`
  variant — no new public API, no semver break.

- **Signal-range crossings now refuse at plan time instead of mislabeling.**
  `ConvertPlan::new` (and everything above it: `new_explicit`,
  `RowConverter`, `convert_buffer`, `adapt_for_encode*`) returns
  `ConvertError::NoPath` when source and target
  `SignalRange` differ. No Narrow↔Full expand/contract kernels exist, and
  the previous behavior planned the *other* descriptor differences and
  emitted the source's range-coded values under the target's range label —
  mislabeled pixels (lifted blacks narrow→full, crush on the way back),
  not a conversion. Same-range plans, including Narrow→Narrow
  value-preserving steps, are unaffected; narrow data still zero-copies to
  same-range targets. The `NoPath` display names the range crossing so the
  refusal isn't two identical-looking descriptors. (Real expand/contract
  kernels remain future work, tracked as zencodec's cross-repo
  Known-Issue 3.)

- **Tone-map helpers gained explicit out-of-domain contracts** (zenpixels#39
  Rung 1): `reinhard_tonemap` and `reinhard_inverse` clamp negative and NaN
  inputs to 0.0, and `reinhard_tonemap(+∞)` returns 1.0 (the limit).
  Previously `reinhard_tonemap(-1.0)` → `-inf` and `(-2.0)` → `+2.0` —
  silently outside the documented range, and reachable because linear HDR
  buffers can carry small negatives from gamut-mapping ringing.
  `exposure_tonemap` now maps NaN to 0.0 for consistency. Only
  out-of-contract inputs change; in-domain values are bit-identical.

### zenpixels-convert — added

- **`hdr::quantize_to(PixelSlice, target: PixelDescriptor) -> Result<PixelBuffer>`**
  — anchor-aware linear→PQ16 quantizer, the canonical successor to the
  withdrawn `encode_pq16`. Reads the absolute-luminance anchor from the
  source's `ColorContext::diffuse_white` (S1a; default `DiffuseWhite::BT2408`
  = 203), pre-scales by `anchor / 10000`, then reuses `convert_buffer` for the
  linear→PQ + f32→u16 quantization. PQ codes match the f64 ST 2084 oracle
  within ±1; CLL is decoupled (`ContentLightLevel::measure`). Real consumer:
  `~/work/hdr-corpus-convert` (M×N HDR epic #45, slice S2).
- Property/oracle tests for the tone-map helpers (output range,
  monotonicity, f64-oracle agreement, round-trip relative-error bound) and
  a pipeline pin in `tests/output_finalize.rs` that a PQ source finalized
  `SameAsOrigin` passes origin CICP through while `OutputMetadata::hdr`
  stays `None` — the documented not-yet-wired contract
  (`output.rs` `TODO(0.3.0)`; wiring it must consciously update the pin).

### zenpixels-convert — deprecated

- **`hdr::HdrMetadata`** (struct + `is_hdr`/`is_sdr`/`hdr10`/`hlg`) and the
  **`OutputMetadata::hdr` field** are deprecated. `HdrMetadata` is a
  redundant, weaker, frozen-shape duplicate of the codec-layer carrier
  `zencodec::Metadata` (which the codecs actually populate): it bundles
  `transfer` with CLL/mastering — which all surveyed prior art (libavif,
  libheif, libjxl, FFmpeg, libplacebo, ICC, CSS, Chrome, …) keeps separate —
  and its public, non-`#[non_exhaustive]` fields can't grow the
  absolute-luminance anchor or gain-map data HDR needs. `hdr10()` also
  synthesizes a placeholder ST 2086 mastering volume that was never measured.
  Carry [`ContentLightLevel`] / [`MasteringDisplay`] directly. Removal +
  the `OutputMetadata::hdr` retype to sibling optional fields are queued for
  0.3.0 (see QUEUED BREAKING CHANGES). Design rationale:
  `docs/hdr-design-survey-2026-06-13.md`.

  (The in-development `hdr::compute_content_light_level` / `encode_pq16` /
  `REFERENCE_DIFFUSE_WHITE_NITS` — never released — were withdrawn and
  replaced: the bare-`f32` anchor the prior art uniformly rejects became the
  typed public `DiffuseWhite` + the `ContentLightLevel::measure` constructor,
  and the encode is the anchor-aware `hdr::quantize_to` above, which reads its
  anchor from `ColorContext` (S1a) rather than a hand-passed arg. The real
  consumer — missed in the first `~/work/zen/`-only audit — is
  `~/work/hdr-corpus-convert` (its own repo one level up), which encodes
  gain-map sources to 16-bit PQ PNG. The longer-term move (thread the anchor
  into the PQ/HLG `ConvertStep`s so the encode collapses fully into
  `convert_buffer`) is tracked in epic #45; see
  `docs/hdr-design-survey-2026-06-13.md`.)

## [0.2.13] - 2026-06-11

Both crates release as 0.2.13 (zenpixels skips 0.2.12 to keep the pair in lockstep).

### Workspace — added

- Versioned public-API surface snapshots at `docs/public-api/{zenpixels,zenpixels-convert}.txt` (389420b), regenerated by `zenpixels/tests/public_api_doc.rs` on every `cargo test` (`ZEN_API_DOC=check` verifies in CI's clippy job, `=off` skips; `just api-doc` / `just api-doc-check` locally).

### zenpixels — added

- **`PixelDescriptor::RGB16_BT2100_PQ` / `RGB16_BT2100_HLG` presets + `new`↔`new_full` doc cross-references** (#28, cce44476) — BT.2100 HDR descriptors (BT.2020 primaries, PQ/HLG transfer, full range — CICP `(9, 16|18, 0, full)`) without spelling out the enums, and `# See also` sections so `new`'s docs surface `new_full` for non-default primaries (and vice versa).
- **`PixelDescriptor::bytes_per_channel()`** (#29, cce44476) — per-sample byte width (`channel_type().byte_size()` shorthand); the usual high-bit-depth gate is `bytes_per_channel() > 1`.
- **Endianness contract documented on `PixelSlice::new` / `PixelSliceMut::new`** (#29, cce44476) — multi-byte samples (U16/F16/F32) are interpreted in native byte order; `ByteOrder` is *channel* order (BGR vs RGB), not byte order within a sample, so decoders reading big-endian container data must swap before wrapping. (Issue #29's constructor-validation option targets an endianness tag that doesn't exist on `PixelDescriptor`; adding one is spec work, tracked with the #36 cluster.)
- **`PixelBuffer::transform_in_place(FnOnce(InPlacePixels) -> PixelSliceMut)` + `InPlacePixels`** — the one atomic primitive for layout-changing in-place transforms: hands the transform the backing bytes plus the current description, validates the returned view (same aligned base, fits the allocation), and adopts width/height/stride/descriptor/color **in the same call** — a buffer whose descriptor disagrees with its bytes is unrepresentable. Deliberately the *only* in-place entry: the slice-level escape hatch (`into_strided_bytes`, briefly on this branch) was removed because a re-described view leaves a borrowing `PixelBuffer` stale. `InPlacePixels` is a plain constructible parameter bag so transforms stay unit-testable at arbitrary strides. (be75ee22)

### zenpixels — changed

- **`SignalRange` semantics fully specified in docs** — the enum now pins the
  ITU "limited/studio swing" definition: anchors scale by `×2^(N−8)` (8-bit
  16–235 / 16–240, 10-bit 64–940 / 64–960, 12-bit 256–3760 / 256–3840), the
  luma span applies to every RGB/gray channel, excursions (sub-blacks,
  super-whites, xvYCC) are legal and an eventual expand kernel must choose
  clamp-vs-preserve, the 1:1 mapping to CICP `video_full_range_flag` / AV1
  `color_range`, and the no-relabeling rule (`with_signal_range` describes,
  never rescales). Also documents the cross-depth caveat: full-scale depth
  rescaling preserves ITU anchors only approximately (8-bit 235 → 60 395 vs
  the ITU 16-bit anchor 60 160). Docs only; no behavior change in zenpixels.
- **ICC identification tables regenerated to include the workspace's own committed CICP-bundle profiles** (855d8e48) — `icc-gen` now decodes `zenpixels-convert`'s `cicp_bundle.lz4` (both the RGB and GRAY locator tables) and feeds every unique profile through the same classification pipeline as the on-disk corpus. +28 live RGB rows and +40 live gray rows (every white point of gray transfers 1/4/6/8/11/12/13/14/15/16); 40 enum-unrepresentable RGB gamuts emitted as `[primaries deferred]` comments; pure additions, no existing corpus entries changed. Profiles this workspace embeds in encoded output are now hash-recognizable on the way back in.

### zenpixels — fixed

- 32-bit overflow safety in `PixelBuffer` constructors and `crop_copy` — checked arithmetic guards `width × height × bytes_per_pixel` from wrapping on 32-bit targets. (#31, 4ebad3e)

### zenpixels-convert — added

- `orient::apply_orientation(PixelSlice, Orientation) -> PixelBuffer` — physically bakes any of the eight EXIF orientations into a fresh buffer (rotate/flip). Descriptor-, channel-, and bit-depth-agnostic (moves whole `bpp`-sized pixels); strided input handled. Flips are row memcpy / element-reverse; the four transposing orientations use a cache-blocked (loop-tiled) transpose with the reflection folded into the destination address. For 4-byte pixels the per-tile transpose is SIMD on x86 (SSE), aarch64 (NEON), and wasm (SIMD128) — magetypes `f32x4::transpose_4x4` generated once from a single body via `#[magetypes(v3, neon, wasm128, scalar)]` and dispatched by `incant!` (scalar tier when no SIMD is present); each pixel rides as one f32 lane (bit-exact — the kernel only shuffles whole 32-bit lanes). Edges / other element widths stay on the cache-blocked scalar path, which is also the parity oracle. This is the buffer-baking half of the zen orientation model — codecs that decode to a raster buffer call it when `OrientationHint::bakes()` is true (the coordinate math stays in `zenpixels::Orientation`).
- `orient::apply_orientation_into(PixelSlice, Orientation, PixelSliceMut) -> Result<(), ConvertError>` — the no-allocation variant: writes into a caller-provided target (must already have the oriented geometry + matching bytes-per-pixel, else `ConvertError::BufferSize`), so callers can reuse / pool the buffer across many calls (a codec `decode_into`, an image proxy on same-size images). `apply_orientation` is now a thin allocating wrapper over it — matching the crate's caller-provides-`dst` convention (`convert_row`).
- `orient::apply_orientation_in_place(&mut PixelBuffer, Orientation) -> Result<(), ConvertError>` — bakes orientation **reusing the buffer's own allocation**, no second pixel buffer (the transposing orientations would otherwise hold a 2× transient). Square transpose = diagonal swap; non-square = cycle-following with an `n`-element visited scratch; flips = row/element swaps; padded input is compacted to tight first. The buffer's dims/stride/descriptor update atomically via `PixelBuffer::transform_in_place`, and the color context is carried through. Trade-off: non-square transpose is cache-hostile (no SIMD) — choose it when peak memory matters more than speed; `bpp > 16` returns `ConvertError::BufferSize`. Parity-gated against the out-of-place path across square/non-square × tight/padded × 5 formats × all 8 orientations. (34a50bd1, reworked be75ee22)
- **`adapt::try_adapt_in_place(&mut PixelBuffer, PixelDescriptor) -> Result<(), At<ConvertError>>`** (#11, cce44476 + 9fc66112; re-targeted to `PixelBuffer` in be75ee22) — allocation-free format adaptation: metadata-only re-tags move zero bytes; `Rgba8`↔`Bgra8` (X-padding forms included) runs garb's SIMD strided B↔R swap in the existing buffer; contract-exact alpha-lane drops (RGBA→RGB, BGRA→RGB with the B↔R reorder, GrayAlpha→Gray at U8/U16/F32) when the alpha mode is `Undefined`/`Opaque`, with the stride rounded down to a whole number of target pixels (rows stay at their own bases when it already divides evenly). `Err(NoPath)` leaves the buffer **untouched** for the allocating `adapt_for_encode` fallback; live `Straight`/`Premultiplied` alpha always errs (matting is `adapt_for_encode`'s job; measured-opaque drops are `reduce_to_load_bearing_format_in_place`'s). Eliminates the full-frame allocation for BGRA-native pipelines (imageflow) feeding RGBA-preferring encoders.
- **`load_bearing` module — bit-exact descriptor narrowing analysis** (#30, 288a6833). `PixelSliceLoadBearingExt` (sealed, on `PixelSlice`) answers "which parts of this buffer's declared descriptor actually carry information?": `determine_load_bearing() -> LoadBearingReport` (`uses_alpha` / `uses_chroma` / `uses_low_bits` — each field validated against a concrete consumer site in zenwebp/zenavif/zenjxl/zentiff via per-encoder audit), `LoadBearingReport::apply_to` (descriptor combiner), and `try_reduce_to_load_bearing_format() -> Option<PixelBuffer>` (rows written in place into one fallible zeroed allocation; alpha drop, gray collapse, bit-replicated U16→U8, Bgra→Rgb with channel reorder — the two RGBA-family drops delegate to garb's SIMD swizzles). Backed by a crate-private `scan` module of magetypes SIMD predicates on 256-bit types over the crate's standard v3/neon/wasm128/scalar tier set — no `w512` feature, no 512-bit types (single-pass fused RGBA8/BGRA8 kernel with blocked deferred reduction; strided rows supported per-row at no cost to the contiguous path). `AlphaMode::Undefined`/`Opaque` answer the alpha question from the descriptor without scanning. Every reduction is bit-exact invertible — primaries/gamut narrowing is deliberately excluded (it re-encodes pixel values; a future explicit opt-in conversion API owns that), and `alpha_is_binary`/`uses_gray_bit_depth` were dropped pre-merge after the audit found no encoder consumer (recoverable from PR history if one materializes).
- **`PixelBufferLoadBearingExt::reduce_to_load_bearing_format_in_place(&mut self, force_alpha_restructuring)`** (#30, 288a6833; re-targeted to `PixelBuffer` in be75ee22) — allocation-free sibling of `try_reduce_to_load_bearing_format`: compacts rows forward into the buffer's own bytes (overlap-safe by construction; the few overlapping prefix rows stage per-pixel, all later rows reuse the allocating path's garb/shuffle row kernels via `split_at_mut`) and adopts the narrowed descriptor + tight stride atomically. `force_alpha_restructuring=false` keeps the alpha lane and re-tags scanned-opaque `Straight`/`Premultiplied` buffers `AlphaMode::Opaque` (tag-only, zero data movement; `Undefined` padding stays untouched); `true` physically drops the lane for packed-form consumers (TIFF/JXL-style writers).
- **`ColorContext` propagation through both reduce variants** (#30, 288a6833) — previously the reduced `PixelBuffer` silently dropped the source's ICC/CICP context. It now carries over for class-preserving reductions (alpha drop, U16→U8, Bgra reorder) and for CICP-only contexts through gray collapse (H.273 primaries/transfer stay meaningful for gray; matrix coefficients don't apply to single-channel data). The report itself always measures chroma truthfully. ICC-bearing sources are handled by the gray-class swap below.
- **`icc_profiles::synthesize_gray_icc_for_cicp(Cicp) -> SynthesizedIcc`** (75eb953e) — GRAY-class sibling of `synthesize_icc_for_cicp`, for single-channel (Gray/GrayAlpha) output: `kTRC` = the CICP transfer's tone curve (byte-identical to the RGB synthesis' curve for the same transfer, PQ/HLG `curv` LUTs included), media white point = the primaries' H.273 white (the only thing primaries contribute to gray — D65/C/E/DCI per Rec. ITU-T H.273 Table 2), and a per-white Bradford white→D50 `chad` (computed per white, not moxcms's scaffold cone matrix). Served from the **same** build-time bundle as the RGB profiles (64 unique gray profiles after white-point dedup cover all 174 grid combos, packed gray-first into the shared per-transfer LZ4 groups so the gray `kTRC` payload — byte-identical to the RGB TRC for the same transfer — compresses away; the whole gray side adds ~7.6 KB to `cicp_bundle.lz4` (28.4 → 36.0 KB) where a separately-compressed gray blob measured 24.4 KB. Gray-first ordering matters: lz4_flex's fast-mode match finder loses distant positions, so RGB-first compressed the gray section at standalone ratio. Same lazy per-group decode + one shared cache, golden-sha256-pinned, `cms-moxcms`-gated byte-roundtrip vs fresh moxcms, content-pinned cross-arch in icc-gen with the byte-pin x86_64-only per the lz4 arch-stability lesson). sRGB default → `NotNeeded` (gray viewers assume sRGB gamma; PNG carries `gAMA`/`sRGB`); off-grid code points → `CmsUnsupported` in every build.
- **Gray collapse now proceeds for ICC-bearing sources when a GRAY-class swap is derivable** (75eb953e) — completes the follow-up queued in the entry above: when the load-bearing rewrite collapses chroma and ICC bytes are attached, the reduce variants derive the profile's CICP description (explicit `ColorContext::cicp` field → embedded `cICP` tag via `zenpixels::icc::extract_cicp` → well-known-profile identification via `zenpixels::icc::identify_common`, worst accepted TRC deviation ±56/65535 — sub-step at 8-bit) and swap in `synthesize_gray_icc_for_cicp`'s GRAY-class profile, keeping the source `cicp` alongside; an sRGB-described ICC drops to CICP/descriptor-only signaling. Unidentifiable profiles — and colors with no CICP code points (Adobe RGB) — still suppress the collapse, and a swap never applies when chroma turns out load-bearing.

### zenpixels-convert — changed

- **`synthesize_icc_for_cicp` now gives full CICP→ICC coverage from a bundled compressed blob — no CMS required.** A default (no-feature) build resolves a profile for the entire assigned ITU-T H.273 grid (11 primaries × 16 transfers = 174 profile-yielding combos, the 2 sRGB-default pairs excluded as `NotNeeded`), where it previously returned `NeedsCms` for anything outside the 4 `&'static` consts. The profiles are bundled as one transfer-grouped LZ4 asset (`src/profiles/cicp_bundle.lz4`, ~28 KB; 596 KB raw → 21.5×) generated build-time by the `icc-gen` `cicp_bundle_gen` tool from moxcms; the runtime decodes only the touched transfer group (lazily, once, into a per-group cache via the existing `once_cell::race::OnceBox`) and slices the profile out as a zero-copy `Cow::Borrowed`. Both the build-time compress and the runtime decode use pure-Rust `lz4_flex` (`no_std` `safe-decode` at runtime) — no C dependency, and `#![forbid(unsafe_code)]` holds. This makes moxcms a **build-time generator only**: `cms-moxcms` is no longer involved in synthesis at all — the feature remains for the separate `MoxCms` *transform* engine that applies profiles to pixels. Internal implementation detail — `synthesize_icc_for_cicp`'s signature and the `SynthesizedIcc` enum are unchanged. `synthesize_icc_for_cicp` is now feature-independent: a CICP outside the assigned grid (e.g. a reserved code point) returns `CmsUnsupported` in every build (a no-CMS build previously returned `NeedsCms`); there is no moxcms synthesis fallback, and the `NeedsCms` variant is retained only for API stability. The committed blob is reproducible (deterministic regen; ICC creation-timestamp zeroed) and pinned by a golden sha256 test; a `cms-moxcms`-gated test asserts every bundled profile is byte-identical to a fresh moxcms generation. The grid includes HDR PQ and HLG: HLG round-trips cleanly through a CMS, while the PQ / P3-PQ profiles (ICC `curv`-LUT encodings) are faithful above ~10 nits but soften in the deep toe (≈8% relative at ~1 nit) — prefer CICP-native PQ/HLG signalling where the container carries it (see the `synthesize_icc_for_cicp` HDR caveat). (87153e1, fca0c87, c0e2607, 02362f3)

### zenpixels-convert — tests

- **Recognition round-trip guards on the embedded profiles** (855d8e48) — `every_bundled_profile_roundtrips_through_extract_cicp` pins: every RGB bundle profile's `cICP` tag recovers its exact code points; every GRAY profile has **no** cicp tag (ICC.1 restricts the tag to RGB/YCbCr/XYZ data colour spaces — the timeless reason; attempting one additionally tripped a moxcms 0.8.1 tag-count/emission mismatch that corrupted the tag table, since fixed upstream in [moxcms#182](https://github.com/awxkee/moxcms/issues/182), which now strips cicp from gray on encode) and is hash-identified exactly per pinned transfer sets (recognized: 1/4/6/8/11/12/13/14/15/16; gaps: 5/7/9/10/17/18 — HLG's curv-LUT exceeds the ±56/65535 identification tolerance). Both directions pinned, so any profile regeneration that gains or loses recognition fails loudly. `bundled_consts_are_hash_identified` guards the four curated `&'static` profiles the same way.

### zenpixels-convert — changed (features)

- **`icc-db` feature (default ON)** gates `lz4_flex` and the committed CICP→ICC bundle (~36 KB asset + lazy per-group decode). `default-features = false` builds drop the asset; `synthesize_icc_for_cicp` / `synthesize_gray_icc_for_cicp` answer `NeedsCms` past the sRGB default and the four bundled consts, and the load-bearing gray-ICC swap degrades to suppressing the collapse for ICC-bearing sources. (be75ee22)

### zenpixels-convert — docs

- **U16 widening reality survey** (#30, 288a6833, `benchmarks/u16_widening_survey_2026-06-10.md`) — measured/source-verified what "secretly 8-bit" 16-bit data looks like: libpng `expand_16`, ImageMagick `-depth 16`, and Rust `image` all emit exact `v*257` byte replication (65535 = 255×257 — correct scaling *is* replication), while `v<<8`, ffmpeg's near-shift-with-YUV-noise, and raw-8-bit-in-16 (`astype`) forms exist in the wild but are value-rescaling, not bit-exact. `bit_replication_lossless_u16` therefore stays `lo == hi` only; a new test pins rejection of all three wild patterns at every dispatch tier.

## [0.2.12] - 2026-06-10

### zenpixels-convert — added

- **`icc_profiles::synthesize_icc_for_cicp(Cicp) -> SynthesizedIcc`** — transfer-aware ICC synthesis for a full CICP, with a typed `SynthesizedIcc` outcome (`#[non_exhaustive]`: `Profile(Cow)` / `NotNeeded` / `NeedsCms` / `CmsUnsupported`) that distinguishes "got bytes" from "needs a CMS", "CMS couldn't", and "sRGB default, none needed". The verb name reflects that it *creates* a profile (bundled fast-path, else generated via the `cms-moxcms` feature). Unlike `icc_profile_for_primaries` (primaries-only, which would hand a BT.2020-**PQ** source the SDR-TRC Rec.2020 profile) it matches the TRC and never mis-tags: the `cms-moxcms` path gates on a populated `red_trc` so a `Reserved`/`Unspecified` code yields `CmsUnsupported` rather than a degenerate profile. (#37)

### zenpixels-convert — deprecated

- **`icc_profiles::icc_profile_for_primaries`** — transfer-blind (returns a gamut's bundled SDR profile regardless of the actual TRC, so it can mis-tag an HDR source). Use `synthesize_icc_for_cicp` for transfer-aware synthesis, or embed a bundled const (`DISPLAY_P3_V4` / `REC2020_V4` / `ADOBE_RGB`) directly. (#37)

### zenpixels-convert — performance

- **garb chunk SIMD for `invert_8px` deinterleave** — the 8-pixel kernel of the inverse-XYB → sRGB u8 pipeline now uses `garb::deinterleave::rgb24_chunk8_to_planes_tokenless_v3` (`_mm_shuffle_epi8`-based) for the u8 stride-3 deinterleave that LLVM autovec wouldn't emit a precomputed-mask shuffle for. (#32, aa570b5)
- **Hardware f16 conversion via magetypes `F16Convert`** — f16↔f32 slice converters use native F16C (x86-64-v3) and NEON-fp16 (aarch64), replacing the prior x86-only F16C dispatch that fell back to scalar on aarch64. (1d45ba4)
- **Optional `avx512` feature** (off by default) — bumps archmage/magetypes to 0.9.26; when enabled and AVX-512F is proven at runtime, f16 conversion runs 16-wide `_mm512_cvt{ph_ps,ps_ph}` (8-wide F16C otherwise). (a128c29)

### zenpixels-convert — changed

- Exclude `tests/` from the published package (540 KB of integration tests); benches remain (required by declared `[[bench]]` targets). (b60bf694)

### zenpixels-convert — docs

- Refresh README for the current zenpixels API. (c1d9787)

## [0.2.11] - 2026-04-25

### zenpixels — added

- **`LumaCoefficients::DisplayP3` variant.** Adds the DisplayP3 luma recipe
  (`0.2289746R + 0.6917385G + 0.0792869B`), derived from the DCI-P3 primaries
  with a D65 whitepoint (the middle row of the DisplayP3→XYZ matrix).
  Unlike BT.709/BT.2020 there is no ITU recommendation prescribing these —
  they match what libultrahdr (`gainmapmath.cpp:162`) and other HDR tooling
  use for RGB→luma on DisplayP3 content. Verified against zenpixels' own
  `rgb_to_xyz(DisplayP3)` Y row within 1 f32 ULP. Non-breaking: new variant
  on `#[non_exhaustive]` enum, new arm on `coefficients()`.

- **`LumaCoefficients::Bt2020` variant and `coefficients()` accessor.**
  Adds the UHDTV BT.2020 luma recipe (`0.2627R + 0.6780G + 0.0593B`, same
  primaries as BT.2100 — the HDR case shares this variant). The new
  `pub const fn coefficients(self) -> [f32; 3]` returns the `[R, G, B]`
  weights for any variant so downstream crates can pull numeric values
  directly instead of maintaining their own tables. Motivated by
  ultrahdr-core's `luma_coefficients()` table, which duplicates this data
  for its BT.2100 HDR base-image path — this landing lets ultrahdr-core
  delete its copy and pass the enum through instead. Non-breaking: new
  variant on `#[non_exhaustive]` enum, new inherent method. `ConvertStep`
  RGB→Gray kernels still hardcode BT.709 internally; wiring `options.luma`
  through the planner is a separate follow-up (7562dd0).

- **F16 (IEEE 754 half-precision) pixel format variants.**
  `PixelFormat::RgbF16`, `RgbaF16`, `GrayF16`, `GrayAF16` on the
  `#[non_exhaustive]` enum. Construct via `PixelDescriptor::new()` —
  preset constants stay `pub(crate)` per YAGNI until an external codec
  consumer requires them. Descriptor-level only — no new `Pixel` trait
  impls, no `half` crate dependency in zenpixels core. Typed
  `impl Pixel for Rgb<f16>` etc. will land when Rust stable ships the
  native `f16` primitive (tracked in #23). Non-breaking: new variants on
  `#[non_exhaustive]`. (#23)

### zenpixels-convert — added

- **`__trace_ops` internal feature flag** (gated module
  `pub mod __trace_ops` with `#[doc(hidden)]`). Runtime op tracer for
  in-repo plan validation tests. With the feature on, every
  `ConvertStep` dispatched through the kernel is recorded by name to a
  thread-local `Vec<&'static str>`. With the feature off (default),
  `record_step` is an `#[inline(always)]` empty function — call sites
  lower to no instructions, production builds pay literally nothing.
  Surface (`start_recording`, `stop_recording`) is internal-only; the
  `__` prefix and `#[doc(hidden)]` follow the same convention as
  `__bench_u16_hybrids`. Tests in `tests/plan_validation.rs` use this
  to lock plan shape for representative `(from, to, options)` tuples
  (no waste, no skips). CI runs the trace-gated tests on each platform
  via `cargo test -p zenpixels-convert --features __trace_ops --test plan_validation`.
  Feature requires `std`.

- **`icc_profiles::icc_profile_for(primaries, transfer)` accessor**
  (`pub(crate)` — internal, currently consumed only by unit tests; will
  be promoted to `pub` when an external codec consumer such as
  ultrahdr-rs lands). Companion to the primaries-only
  `icc_profile_for_primaries`; matches a `(ColorPrimaries, TransferFunction)`
  pair against the bundled profile set and returns `None` if nothing
  matches the requested TRC exactly. Currently routes:
  `(DisplayP3, Srgb|Bt709)` → `DISPLAY_P3_V4`,
  `(Bt2020, Bt709|Srgb)` → `REC2020_V4`,
  `(AdobeRgb, Gamma22)` → `ADOBE_RGB`. HDR transfers (`Pq`, `Hlg`) and
  `Linear` return `None` on every primaries set — no PQ/HLG ICC profiles
  are bundled, and those workflows should signal color via CICP or
  generate the profile through a CMS. (47eb81e)

- **F16 conversion kernels.** `ConvertStep::F16ToF32` / `F32ToF16`
  (private enum variants) with scalar implementations. Planner routes
  F16 ↔ F32, F16 ↔ U8, F16 ↔ U16, and same-F16-TF-change paths through
  F32 linear intermediate — never passes F16 bytes through unchanged on
  TF changes. F16 arms added to all layout/swizzle kernels
  (`swizzle_bgra_rgba`, `rgb_to_bgra`, `add_alpha`, `drop_alpha`,
  `matte_composite`, `gray_to_rgb`, `gray_to_rgba`, `gray_alpha_to_rgba`,
  `gray_alpha_to_rgb`, `gray_to_gray_alpha`, `gray_alpha_to_gray`,
  `straight_to_premul`, `premul_to_straight`). SIMD dispatch for
  F16C / AVX-512 FP16 / ARMv8.2 FP16 is a deferred optimization.
  5 round-trip tests in `tests/roundtrip.rs`. (#23)

- **Local scalar f16 conversion (`src/f16_scalar.rs`, `pub(crate)`).**
  Two functions, `f16_bits_to_f32` and `f32_to_f16_bits`, cover every
  production call site that previously used `half::f16`. Correct IEEE
  754 binary16 semantics: zero, subnormals, normals, infinity, NaN,
  overflow, underflow, round-to-nearest-even. Cross-validated against
  the `half` crate exhaustively for all 65,536 f16 bit patterns
  (f16→f32 is bit-exact) and sampled broadly for f32→f16 rounding.
  The `half` crate is no longer a production dependency — demoted to
  `[dev-dependencies]` for cross-validation and the perceptual-loss
  suite only. (#23)

### zenpixels-convert — performance

- **Fused sRGB U16 RGB gamut conversion now uses LUT-decode + SIMD
  polynomial encode** (was LUT-decode + linear-indexed LUT-encode).
  +17% throughput at 1080p in T7 gamut benchmarks, and 100% bit-exact
  u16 roundtrip (was ~71% ±6 with the old 128 KB encode LUT — see
  `benchmarks/u16_hybrid_matrix_2026-04-23.txt`). Unblocked by
  `linear-srgb` 0.6.12 shipping the SIMD `linear_to_srgb_u16_v3` rite.
  `ConvertStep::FusedSrgbU16GamutRgb` now dispatches through
  `convert_u16_rgb_simd_lutdec_polyenc`. Bumped `linear-srgb` minimum
  to 0.6.12.

### zenpixels-convert — fixed

- **MatteComposite integer (U8/U16) arms now honor the source TF.**
  Pre-fix the U8 and U16 `matte_composite` kernel arms always linearized
  via `srgb_u8_to_linear` / `srgb_u16_to_linear` regardless of the
  descriptor's transfer function — silently wrong for every non-sRGB TF,
  including `Linear` (sRGB EOTF applied to already-linear data). The
  `MatteTf` trait gained `eotf_u8`/`oetf_u8`/`eotf_u16`/`oetf_u16` default
  methods (route through f32 EOTF/OETF); `SrgbTf` overrides them with the
  LUT-based fast path so the dominant case keeps its speed. New
  `dispatch_matte_u8_rgba` / `dispatch_matte_u16_rgba` mirror the float
  side. Tests in `matte_composite_linearize.rs` exercise sRGB, Linear,
  and BT.709 on both U8 and U16, with the Linear u8 case hard-rejecting
  the pre-fix buggy output. (d7965b1)

- **`ConvertOptions::luma` is now actually honored in `RgbToGray` /
  `RgbaToGray`.** Pre-fix the kernels hardcoded BT.709 via garb regardless
  of the user's coefficient choice — DisplayP3, BT.601, BT.2020 silently
  ignored. Additionally the U8-only kernel ran on cast bytes for
  U16/F32/F16 inputs, producing garbage. Now: `ConvertStep::RgbToGray` and
  `RgbaToGray` carry resolved `LumaCoefficients`; `ConvertPlan::new_explicit`
  walks the plan after build and substitutes the user's choice. New per-
  channel-type kernels (`rgb/rgba_to_gray_{u8_generic, u16, f32, f16}`)
  apply the configured coefficients. BT.709 u8 keeps garb's fixed-point
  fast path; other coefficients on u8 + all U16/F32/F16 use the generic
  f32 path. Y' (encoded luma) semantic preserved — round-trip identity
  invariant tested in `ulp_exhaustive.rs:561-564` still holds. New
  `tests/luma_coefficients.rs` (8 cases) verifies each coefficient
  produces distinct output. `ConvertStep` is `pub(crate)` so the variant
  shape change is internal only. (d8c28fc)

- **MatteComposite now linearizes non-Linear F32/F16 pixel data
  correctly** (fixes #25). Previously the F32 and F16 kernel arms
  assumed pixel color channels were already in linear light — true
  when the planner inserted a linearize step before the kernel, but
  false for same-TF same-depth float plans (e.g. F32 sRGB RGBA →
  F32 sRGB RGB with `AlphaPolicy::CompositeOnto`), where the plan
  degenerated to `[MatteComposite]`. Error on mid-gray sRGB pixels
  over a grey matte was up to ~12% of the normalized range. The
  kernel now reads the source descriptor's `TransferFunction` and
  linearizes only the RGB channels inline (alpha stays linear as it
  always was), mirroring the pattern the U8/U16 arms already use for
  sRGB. Integer arms continue to hardcode sRGB EOTF/OETF — correct
  for the common sRGB case, latently wrong for BT.709 u8 / PQ u16 /
  HLG u16; extending the integer arms is a separate follow-up (also
  tracked under #25). 5 regression tests in
  `tests/matte_composite_linearize.rs`.

- **Depth-reduction policy now catches U16 → F16.** Previous gate used
  `ChannelType::byte_size()`, which misses U16 → F16 since both are 2
  bytes — but U16 carries 16 bits of precision in [0,1] vs F16's ~11,
  so U16 → F16 is a precision reduction. Switched to
  `channel_bits()` (already used by the cost model in `negotiate.rs`)
  so `DepthPolicy::Forbid` correctly rejects U16 → F16 and U16 → F16
  inside RGBA → RGBA plans. No API change.

## [0.2.10] - 2026-04-20

### zenpixels — docs

- **Clarified `PixelDescriptor::with_transfer` / `with_primaries` /
  `with_alpha` as relabel-only operations** (docs-only). Previously the
  doc comments said "return a copy with a different X" — literally true
  but understated that no pixel math happens. Updated to explicitly call
  out that these are metadata-only operations and that the way to
  actually re-encode pixels is to pass the new descriptor as the
  destination to `RowConverter::new`. Also notes that built-in premul
  kernels operate in encoded byte space (Canvas 2D semantics), not
  linear light. (#21, da5ba60)

### zenpixels-convert — fixed

- **`AlphaPolicy::CompositeOnto` now produces correct pixels for
  premultiplied source** (fixes issue [#19][] [F]). The `matte_composite`
  kernel uses the straight-alpha over operator `fg*a + bg*(1-a)` after
  decoding to linear light. If the source descriptor declared
  `AlphaMode::Premultiplied`, feeding its bytes into the straight kernel
  multiplied by `a` a second time, producing
  `straight*a² + bg*(1-a)` — silently wrong by up to ~24 u8 codes at
  a ≈ 0.25. Fix: planner inserts `PremulToStraight` before
  `MatteComposite` when the source alpha mode is `Premultiplied`,
  recovering straight sRGB bytes (in our library's encoded-space premul
  convention) that the kernel handles correctly. No API change; kernel
  unchanged. Straight-source path unaffected. 6 regression tests in
  `tests/matte_composite_premul.rs`. (#22, ae0ebd8)

- **Planner no longer silently passes bytes through on TF changes** (fixes
  issue [#19][] [A] and [B]). Previously, several descriptor pairs emitted
  `[Identity]` or a naked depth-scale step labeled with the target TF but
  applying no EOTF/OETF — producing wrong pixels with no error. Affected:
  - Same-depth integer TF changes (U8 / U16): `Gamma22 → Srgb`, `Srgb →
    Bt709`, `Pq → Srgb`, every other cross-TF pair. Now routes through an
    F32 linear intermediate.
  - Integer↔F32 cross-TF combinations without a fused kernel: `U8 Gamma22
    → F32 Linear`, `U8 Bt709 → F32 Linear`, `U16 Gamma22 → F32 Linear`,
    etc. Now composes `NaiveU8ToF32 / U16ToF32` with the appropriate F32
    EOTF/OETF steps. Mid-gray error was up to 55× off for PQ inputs.
  - `U8 Bt709 → F32 Linear` and `F32 Linear → U8 Bt709` previously used
    the sRGB-specific fused kernel (`SrgbU8ToLinearF32` /
    `LinearF32ToSrgbU8`), producing ~17% linear-light error vs the correct
    BT.709 EOTF. Fused path now narrowed to sRGB; BT.709 composes through
    the correct step.
  - U16↔U8 and U8↔U16 cross-depth cross-TF combinations now compose
    through F32 linear when a fused kernel doesn't exist.
  Unknown TF on either side continues to pass bytes through as before (the
  explicit-intent API is tracked in #19 [C]/[D] for deprecate-and-add).
  Adds 54 regression tests in `tests/planner_silent_passthrough.rs`
  covering every TF × depth combination, HDR (PQ/HLG), extended-range
  out-of-gamut (`with_clip_out_of_gamut(false)`), and cross-primaries
  crossings. (#20, a9cb1c3)

### zenpixels-convert — added

- **First-class Gamma 2.2 (Adobe RGB 1998) transfer in the fast path.** New
  `ConvertStep::Gamma22F32ToLinearF32` / `LinearF32ToGamma22F32` variants plus
  `depth_steps` arms for Gamma22 ↔ {Linear, sRGB, BT.709, PQ, HLG} same-depth
  F32. The primaries-conversion injection now routes Gamma22 through the
  correct EOTF/OETF instead of falling through to the sRGB approximation.
  Lets AdobeRGB ↔ PQ / HLG / BT.2020 / Linear compose in the built-in planner
  without hitting the moxcms CMS fallback. SIMD via
  `linear_srgb::default::{gamma_to_linear,linear_to_gamma}_slice` with
  `ADOBE_GAMMA = 563/256 ≈ 2.19921875`. `ConvertStep` is `pub(crate)`;
  no public API change. (#18, 2238309)

### zenpixels — internal

- Regenerated ICC hash tables with 23 new canonical/test profile entries
  (jpegli/libjxl testdata V4 canonicals, Rec2020 PQ CICP, RawTherapee
  working spaces, misc test profiles). RGB table: 209 → 240 lines;
  gray table: 31 → 34 lines. No existing entries removed or modified.
  Three skcms iccMAX (ICC v5) sRGB profiles were evaluated but deferred:
  moxcms rejects them with `InvalidProfile`, and the
  `imazen/moxcms#permissive-flagged-profiles` branch that attempts to
  parse them has a mpet matrix-shaper miscalibration producing ~13% FSR
  channel imbalance — see imazen/moxcms#2. Hard-rejection at the CMS
  layer is strictly safer than fast-path misrendering. (#17, c876b94)

[#19]: https://github.com/imazen/zenpixels/issues/19

## [0.2.9] - 2026-04-16

### zenpixels-convert — internal

- **`builtin_profiles` module (internal, `pub(crate)`)** — hand-coded XYB ICC
  inverse for jpegli's 720-byte profile. SIMD-accelerated via magetypes on
  x86_64. Internal only — no external consumers yet per YAGNI policy. (a5fdf9f, a3d924f)

### zenpixels-convert — performance

- **Fused `RgbToBgra` conversion step** — planner now emits a single
  `ConvertStep::RgbToBgra` for `(Rgb → Bgra)` u8 conversions instead of the
  two-pass `[AddAlpha, SwizzleBgraRgba]` sequence. Delegates to
  `garb::bytes::rgb_to_bgra` (8 px/iter AVX2 with R/B swap and `alpha=255`
  in one pass), halving destination-buffer write traffic for u8. u16/f32
  continue to use the existing two-step scalar path. (baa6214)

### zenpixels-convert — fixes

- **Raise `linear-srgb` minimum version to `0.6.10`** (was `0.6.7`).
  `srgb_to_linear_extended_slice` / `linear_to_srgb_extended_slice` were
  added in `linear-srgb` 0.6.10; downstream builds resolving to 0.6.7 (e.g.,
  zenpng fuzz targets) failed to compile. The workspace lockfile already
  resolved to 0.6.10 — this codifies the actual minimum. (9c53fe0)

### Docs & internal

- Fix 12 `cargo doc` warnings across `zenpixels`, `zenpixels-convert`, and
  `scripts/icc-gen`: fully qualified intra-doc links for cross-module
  references (`ColorModel`, `ColorProfileSource`, `ConvertPlan`,
  `RowConverter::new_explicit_with_cms`, `PluggableCms`), corrected
  `identify → identify_common` reference in the `icc` module preamble,
  dropped intra-doc links to private items (`Tolerance`, `ZenCmsLite`),
  fixed `Self::IccOnly` mislabel in `ColorPriority` docs, and escaped
  `<icc-cache-dir>` placeholders in `icc-gen` module docs. `cargo doc
  --no-deps` is now warning-free. Also bumps `[workspace.package]` version
  `0.2.2 → 0.2.8` to match the member crates. (b58212a)

## zenpixels 0.2.8 + zenpixels-convert 0.2.8 (2026-04-15)

Ships `PluggableCms`, `RowTransformMut`, fused matlut kernels,
`ConvertOptions::clip_out_of_gamut`, and `ZenCmsLite` as the default
CMS backend. Carries a set of **tolerated technical breaks** (see
[`CLAUDE.md`](CLAUDE.md) §0.2.x versioning policy) that
`cargo semver-checks` flags as major but which have no known external
impact. A 0.3.0 bump for these alone was judged too disruptive to the
`zen*` sibling dependency graph.

### zenpixels

#### Added

- **`ConvertOptions::clip_out_of_gamut: bool`** field (default `true`)
  plus `with_clip_out_of_gamut(bool)` builder. Set to `false` to emit
  sign-preserving extended-range f32 sRGB transfers — preserves
  negative and supernormal values for HDR / wide-gamut pipelines that
  defer tone or gamut mapping to a later stage.
- **`ConvertOptions::forbid_lossy()`** / **`::permissive()`** presets
  plus `with_alpha_policy`, `with_depth_policy`, `with_gray_expand`,
  `with_luma`, `with_clip_out_of_gamut` builders. Required since
  `ConvertOptions` became `#[non_exhaustive]`.

#### Changed (tolerated technical breaks)

- **`ConvertOptions` → `#[non_exhaustive]`**. External struct-literal
  construction breaks; in-tree callers migrated to builder pattern.
  Audited: no external struct-literal users across `~/work/zen/`.

### zenpixels-convert

#### Added

- **`PluggableCms` trait** with `build_source_transform` (owned
  `Box<dyn RowTransformMut>`) and `build_shared_source_transform`
  (shared `Arc<dyn RowTransform>`, default `None`). Dyn-compatible,
  accepts `ColorProfileSource` (ICC / CICP / Named / PrimariesTransferPair),
  carries `&ConvertOptions`.
- **`CmsPluginError`** — type-erased error wrapper for plugins, wraps
  any `core::error::Error + Send + Sync`. Plugin methods return
  `Option<Result<T, whereat::At<CmsPluginError>>>`: `None` = declined
  (chain tries next plugin), `Some(Ok)` = accepted, `Some(Err)` =
  tried-and-failed (error propagates immediately — the chain does not
  continue past a failed plugin to avoid silently substituting different
  color math). The `At<_>` envelope records the plugin's internal
  failure point via `whereat::at!` / `ResultAtExt::at()`; the receive
  site in `RowConverter::new_explicit_with_cms` adds a second stamp when
  wrapping into `ConvertError::CmsError`.
- **`RowTransformMut` trait** (`&mut self`, `Send`) for owned, stateful
  transforms that need scratch buffers without interior mutability.
  `RowTransform` (`&self`, `Send + Sync`) remains for stateless/shareable
  transforms (e.g., moxcms `TransformExecutor`).
- **`RowConverter::new_explicit_with_cms`** with ordered dispatch:
  user-supplied plugin first, then `ZenCmsLite` default (named-profile
  matlut fast path). Integer profile matches use fused SIMD kernels.
- **`finalize_for_output_with(...)`** — dyn-safe replacement for
  `finalize_for_output<C>`. Takes `Option<&dyn PluggableCms>` and routes
  through `RowConverter::new_explicit_with_cms` so the CMS dispatch
  chain is honored.
- **`SourceColor::to_color_context()`** (zencodec) — drops the
  non-authoritative color field based on `color_authority` so
  `ColorContext::as_profile_source()` naturally returns the
  authoritative source without a separate parameter.
- **Fused u8/u16 matlut SIMD kernels** on `RowConverter`'s default
  path. u8 sRGB: ~3× speedup vs prior; u16 sRGB: ~49× speedup.

#### Changed (tolerated technical breaks)

- **`RowTransform: Send + Sync`** (was `Send`-only). In-tree impls
  (`MoxRowTransform`, `LiteTransform`) already satisfy `Sync` because
  their inner state does. External impls that were intentionally
  `!Sync` would break; none are known.
- **`RowConverter` is no longer `Sync` / `UnwindSafe` /
  `RefUnwindSafe`**. Mechanical consequence of the new
  `external: Option<Box<dyn RowTransformMut: Send>>` field.
  `convert_row` takes `&mut self`, so cross-thread shared-reference
  use was never useful.
- **`RowConverter` no longer derives `Debug`**. Plan contents and
  external transform have no meaningful `Debug` representation.
- **Feature `zencms-lite` removed**. Functionality became unconditional
  — LUTs use `OnceBox` for no_std compatibility without a feature gate.

#### Deprecated

- **`ColorManagement` trait** — use `PluggableCms` instead.
  `ColorManagement` is non-dyn-safe (generic `type Error`), takes raw
  ICC byte pairs, and has no options channel. Existing impls
  (`MoxCms`, `ZenCmsLite`) are preserved and still work; they gain
  `#[allow(deprecated)]` on the impl block.
- **`finalize_for_output<C: ColorManagement>`** — use
  `finalize_for_output_with(..., cms: Option<&dyn PluggableCms>)`.

---

## Queued breaking changes (for 0.3.0)

Non-tolerated breaks (see [`CLAUDE.md`](CLAUDE.md) §0.2.x versioning
policy) — these require a proper 0.3.0 bump. This section accumulates
across 0.2.N patches and only clears when the 0.3.0 release cuts.

### zenpixels

- **`repr(u8)` removal** from `ColorPrimaries` and `TransferFunction`.
- **`ColorContext` → `#[non_exhaustive]`**. Construct via
  `from_icc()` / `from_cicp()` + builders. Direct struct literal
  construction is already discouraged (fields are `Option` with no
  authority signal); non-exhaustive makes it enforceable.
- **Remove `ColorContext::from_icc_and_cicp()`** (deprecated since 0.2.6).
  Use `from_icc()` or `from_cicp()` — codecs should populate only the
  authoritative field via `SourceColor::to_color_context()`.
- **Remove `ColorPrimaries` / `TransferFunction` commented-out variants**
  (deferred AppleRgb, Smpte170m, Bt470Bg, WideGamut, ColorMatch,
  EciRgbV2, DciP3, Gamma18, Gamma24, Gamma28) — clean up after the
  `repr(u8)` removal frees discriminant assignment.

### zenpixels-convert

- **`ConvertError` → `#[non_exhaustive]`** + new
  `HdrTransferRequiresToneMapping` variant. See imazen/zenpixels#10 for
  HDR provenance plan.
- **Remove `ColorManagement` trait** (deprecated in 0.2.8). Callers
  migrate to `PluggableCms`.
- **Remove `finalize_for_output<C: ColorManagement>`** (deprecated in
  0.2.8). Callers migrate to `finalize_for_output_with(..., cms:
  Option<&dyn PluggableCms>)`.
- **Remove `ZenCmsLite::extended` field and `::extended()` constructor**
  (deprecated; use `ConvertOptions::clip_out_of_gamut` via
  `RowConverter` instead).
- **Remove `lut_transform_opts()` and `cicp_transform_opts()`** in
  `cms_moxcms` (deprecated since 0.2.3; use `transform_opts()` with
  explicit `ColorPriority` + `RenderingIntent`).
- **Remove `ADOBE_RGB_COMPAT` and `PROPHOTO_RGB`** ICC profile constants
  in `icc_profiles` (deprecated since 0.2.4).

---

## zenpixels 0.2.7 (2026-04-14)

### Additions

- **`ColorPrimaries::AdobeRgb`** and **`TransferFunction::Gamma22`** enum
  variants for Adobe RGB (1998) identification and conversion.
- **`icc` module** — lightweight ICC profile identification (~100ns):
  - `identify_common(icc_bytes)` — hash-based lookup against 163 known RGB
    + 18 grayscale profiles from a corpus of 1,065 real-world ICC profiles.
  - `is_common_srgb(icc_bytes)` — convenience sRGB check.
  - `extract_cicp(data)` — read CICP tag from ICC v4.4+ profiles.
  - `IccIdentification` struct with `primaries`, `transfer`, `valid_use`.
  - `IdentificationUse` enum: `MetadataOnly` vs `MatrixTrcSubstitution` —
    tells callers whether matrix+TRC math is safe or a full CMS is needed.
- **`ColorPrimaries` methods**: `chromaticity()`, `white_point()`,
  `gamut_matrix_to()` (const-computed 3×3 Bradford-adapted gamut matrices),
  `WHITE_D65` constant.
- **`ColorProfileSource::PrimariesTransferPair`** variant +
  `from_primaries_transfer()`, `primaries_transfer()`, `resolve()`.
- **`NamedProfile`**: `from_primaries_transfer()`, `to_primaries_transfer()`.
- **`PixelDescriptor::color_profile_source()`**.
- **`ColorAuthority`** enum (`Icc` | `Cicp`) on `ColorContext` / `ColorOrigin`.
- **`NamedProfile::Bt2020Hlg`** + `TransferFunction::Hlg` CICP 18 round-trip.

### Behavior changes

- **`ColorContext::as_profile_source()`** now respects `color_authority` instead
  of hardcoding CICP preference.
- Enum variants trimmed to those with backing conversion math. Removed variants
  preserved as comments in `descriptor.rs` with chromaticities and rationale.

### Internal (not public API)

- `scripts/icc-gen` crate for regenerating ICC hash tables with empirical
  CMS validation via moxcms + lcms2 cross-check.
- `Safe::AnyIntent` / `Safe::IdOnly` named constants replace magic bitfields
  in `.inc` table files.
- `ProfileFeatures`, `inspect_profile()`, `CoalesceForUse`,
  `identify_common_for()` — kept `pub(crate)` for future use.
- Color registry with const-computed gamut matrices.

## zenpixels-convert 0.2.7 (2026-04-14)

### Additions

- **`icc_profiles::ADOBE_RGB`** — bundled CC0 Adobe RGB (1998) ICC profile
  (v2, pure gamma 2.19921875, matching ~85% of real-world profiles).
- **`ADOBE_RGB_V4`** deprecated alias → `ADOBE_RGB`.
- **`PROPHOTO_V4`** deprecated (returns empty bytes — ProPhoto not bundled
  due to TRC fragmentation).

### Behavior changes

- **`finalize_for_output()` respects `ColorAuthority`** on the `ColorOrigin`.
- **`SameAsOrigin` no longer invokes the CMS.** Only pixel format changes
  are applied. Previously built a wasteful same-profile-to-same-profile
  CMS transform.
- **`conversion_matrix()` returns `Option<GamutMatrix>`** (owned) instead of
  `Option<&'static GamutMatrix>`. `GamutMatrix` is `Copy` — callers drop
  the `&`.

### Internal (not public API)

- `ZenCmsLite` + `fast_gamut` — fused SIMD gamut conversion kernels
  (sRGB ↔ Display P3 ↔ BT.2020 ↔ Adobe RGB). Kept `pub(crate)` pending
  aarch64 SIMD and benchmarking against moxcms.
- Bundled Adobe RGB profile switched from v4 paraType-3 to v2 pure-gamma
  form (matches the spec and ecosystem majority).

## 0.2.3

### zenpixels-convert — additions

- **`RenderingIntent`** enum — `Perceptual`, `RelativeColorimetric` (default),
  `Saturation`, `AbsoluteColorimetric`. Backend-agnostic ICC rendering intent
  with thorough documentation of LUT fallback behavior, profile compatibility,
  and the moxcms/lcms2 perceptual intent mismatch.
- **`ColorPriority`** enum — `PreferIcc` (default), `PreferCicp`. Controls
  whether the CMS trusts ICC `curv`/`para` TRCs or CICP transfer characteristics.
  Documented: precision tradeoffs, advisory vs. authoritative semantics, and
  when each setting is correct.
- **`transform_opts(priority, intent)`** — single entry point for building
  moxcms `TransformOptions`. Replaces `lut_transform_opts()` and
  `cicp_transform_opts()` with explicit control over rendering intent.

### zenpixels-convert — breaking behavior change

- **Default rendering intent is now `RelativeColorimetric`**, not `Perceptual`.
  The previous default inherited moxcms's `Perceptual`, but moxcms's perceptual
  intent does not match lcms2 and may produce inaccurate results. Most display
  profiles only ship a relative colorimetric LUT, making the two intents
  identical in practice — but for profiles that do have perceptual tables, this
  is a visible change.

### zenpixels-convert — deprecations

- **`lut_transform_opts()`** — use `transform_opts(ColorPriority::PreferIcc, intent)`.
- **`cicp_transform_opts()`** — use `transform_opts(ColorPriority::PreferCicp, intent)`.

## 0.2.2

### zenpixels-convert — additions

- **`lut_transform_opts()`** (public) — canonical moxcms `TransformOptions` for
  standard ICC LUT transforms: `allow_use_cicp_transfer: false`,
  `BarycentricWeightScale::High`, `InterpolationMethod::Tetrahedral`.
- **`cicp_transform_opts()`** — same quality settings as `lut_transform_opts` but
  `allow_use_cicp_transfer: true` for CICP-native source formats (JXL, HEIF).

### zenpixels-convert — improvements

- **`InterpolationMethod::Tetrahedral`** added to the internal `lut_transform_opts`
  used by `MoxCms::build_transform_for_format`. Improves accuracy of 3D CLUT
  transforms over the previous trilinear default.
- **`BarycentricWeightScale::High`** was already set; now documented in the public
  function with the rationale (max LUT interpolation error ≤2 vs ≤14, no perf cost).

## 0.2.1

### zenpixels — additions

- **`serde` feature** — optional `Serialize`/`Deserialize` derives on all core
  types: `PixelDescriptor`, `PixelFormat`, `ChannelType`, `ChannelLayout`,
  `AlphaMode`, `TransferFunction`, `ColorPrimaries`, `SignalRange`, `ColorModel`,
  `ByteOrder`, `Cicp`, `ContentLightLevel`, and `MasteringDisplay`. Off by default.

### zenpixels-convert — additions

- **Gamut matrix in `RowConverter`** — primaries conversion (e.g., BT.709 ↔
  Display P3 ↔ BT.2020) is now automatic. `RowConverter` injects a 3×3 matrix
  step in linear f32 space when source and destination primaries differ.
  Previously callers had to apply gamut matrices manually.
- **Embedded ICC profiles** — CC0-licensed ICC profiles for Display P3, AdobeRGB,
  Rec2020, and ProPhoto are now bundled. `icc_profile_for_primaries()` returns
  the appropriate profile bytes for a given `ColorPrimaries` value.
- **`serde` feature** — forwards to `zenpixels/serde`.

### Bug fixes

- **Display P3 TRC correction** — `identify_by_colorants` now correctly maps
  Display P3 to sRGB transfer characteristic (code 13) instead of BT.709 (code 1).
- **`allow_use_cicp_transfer` disabled** in the moxcms CMS path. CICP transfer
  function override is for applications, not CMMs. The zen conversion pipeline
  handles transfer functions explicitly via `RowConverter`, so the CMS should
  only apply the ICC profile's gamut mapping. Matches the moxcms v2 path fix.
- **Linear-space matte compositing** — `matte_composite` now blends in linear
  light instead of gamma-encoded space, fixing visible darkening artifacts at
  semi-transparent edges.

## 0.2.0

This is a **breaking release** — see "Breaking changes" below.

### zenpixels — breaking changes

- **Removed `buffer` feature.** Its functionality (`rgb` + `imgref`) is now always
  available via the `imgref` feature, which implies `rgb`.
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, BufferError>` now return `Result<T, At<BufferError>>`.
  Call `.error()` to inspect, `.into_inner()` to unwrap, or use `whereat::ResultAtExt`
  for ergonomic chaining. Affected: `PixelSlice::new`, `PixelSliceMut::new`,
  `PixelBuffer::try_new`, `from_vec`, `from_pixels`, `reinterpret`, and all
  `_typed` constructors.

### zenpixels — additions

- **`Orientation` enum** — canonical EXIF orientation type (D4 dihedral group).
  `#[repr(u8)]` with EXIF values 1-8. Includes D4 group algebra (`compose`,
  `inverse`, `then`), geometry helpers (`output_dimensions`, `forward_map`,
  `swaps_axes`, `is_row_local`), and EXIF conversion (`from_exif`, `to_exif`).
  All core methods are `const`. Re-exported at crate root.
- `PixelSlice::as_strided_bytes()` — zero-copy access to raw backing bytes including
  inter-row stride padding. For GPU uploads, codec writers, and other buffer+stride APIs.
- `PixelSliceMut::as_strided_bytes()` / `as_strided_bytes_mut()` — return the full
  backing `&[u8]` / `&mut [u8]` including any trailing bytes beyond the image extent.
- `PixelSliceMut::as_pixel_slice()` and `From<PixelSliceMut> for PixelSlice` —
  zero-copy immutable borrow/move from a mutable slice.
- `ContentLightLevel` and `MasteringDisplay` moved here from `zenpixels-convert::hdr`.
  Re-exported at crate root.
- `MasteringDisplay::HDR10_REFERENCE` and `DISPLAY_P3_1000` — predefined constants
  for common mastering display configurations.
- `Cicp::from_descriptor()`, `Cicp::to_descriptor()` — round-trip between CICP codes
  and `PixelDescriptor`.
- `NamedProfile::from_cicp()` — identify named profiles from CICP codes.
- `TransferFunction::to_cicp()`, `ColorPrimaries::to_cicp()` — convert enum variants
  to CICP code points.
- `ConvertOptions` convenience constructors: `forbid_lossy()`, `permissive()`,
  plus `with_alpha_policy()`, `with_depth_policy()`, `with_gray_expand()`,
  `with_luma()` builders.
- `#[track_caller]` on all fallible constructors for better error diagnostics.
- `whereat::At`, `ResultAtExt`, and `at` re-exported at crate root.

### zenpixels-convert — breaking changes

- **`RowConverter::convert_row()` and `convert_rows()` changed from `&self` to
  `&mut self`**. This enables internal scratch buffer reuse (no per-row heap allocation).
  Callers must use `let mut converter`.
- **`RowConverter` no longer auto-derives `Clone`.** A manual `Clone` impl creates
  fresh (empty) scratch buffers. Behavior is unchanged but the clone is not a
  bitwise copy.
- **`RowTransform` trait now requires `Send`.** Non-`Send` implementors will no longer
  compile.
- **`PixelBufferConvertExt` trait split.** `to_rgb8()`, `to_rgba8()`, `to_gray8()`,
  `to_bgra8()` moved to new `PixelBufferConvertTypedExt` trait (requires `rgb` feature).
  `linearize()` and `delinearize()` added to `PixelBufferConvertExt` (always available).
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, ConvertError>` now return `Result<T, At<ConvertError>>`.
  Affected: `RowConverter::new`, `new_explicit`, `convert_rows`,
  `adapt_for_encode`, `adapt_for_encode_explicit`, `convert_buffer`,
  `PixelBufferConvertExt` methods.
- **`codec` feature renamed to `pipeline`.** `CodecFormats`, `FormatEntry`,
  `ConversionPath`, `PathEntry`, etc. moved from root to `pipeline::` submodule.
  Import paths changed from `zenpixels_convert::registry::*` to
  `zenpixels_convert::pipeline::*`.
- **`Cicp::SRGB.matrix_coefficients` changed from `6` to `0`** (correct per ITU-T H.273
  — sRGB is an RGB color space, not YCbCr, so Identity matrix is correct).

### zenpixels-convert — additions

- **Streaming perf: zero per-row allocation.** `ConvertScratch` ping-pong buffers
  replace heap allocation in multi-step row conversions.
- `ConvertPlan::compose()` and `RowConverter::compose()` — chain two converters.
  Peephole optimization cancels inverse pairs (e.g., premultiply + unpremultiply).
- `RowConverter::new_explicit()` — explicit conversion plan with `ConvertOptions`
  policy validation before creating the plan.
- `MatteComposite` conversion step — flatten alpha against a matte color
  (used by `AlphaPolicy::CompositeOnto`).
- `linearize()` / `delinearize()` on `PixelBufferConvertExt` — buffer-level
  transfer function conversion.
- F32-to-F32 transfer function kernels: `SrgbF32ToLinearF32`, `LinearF32ToSrgbF32`,
  `Bt709F32ToLinearF32`, `LinearF32ToBt709F32`. Previously only u8/u16↔f32 TF
  conversions existed; these enable f32→f32 re-encoding without a depth roundtrip.
  PQ and HLG f32↔f32 kernels also added. All SIMD-dispatched via `linear-srgb`.
- **moxcms CMS backend** (behind `cms-moxcms` feature). `MoxCms` implements
  `ColorManagement` for ICC profile transforms via the `moxcms` crate. Supports
  u8, u16, and f32 transforms. F16 input routes to the f32 path.
- `garb` 0.2 for SIMD-accelerated pixel swizzle, layout conversions, depth scaling,
  and BT.709 luma.
- Public Oklab constants and functions: `LMS_FROM_XYZ`, `XYZ_FROM_LMS`,
  `OKLAB_FROM_LMS_CBRT`, `LMS_CBRT_FROM_OKLAB`, `rgb_to_oklab()`, `oklab_to_rgb()`,
  `fast_cbrt()`.

### Bug fixes

- F16 data no longer incorrectly routed to u16 CMS transform path. F16 now uses
  the f32 transform (IEEE 754 half-floats are not integer-encoded).
- Fixed ICC profile identification to use D50-adapted PCS colorants.

## 0.1.0

Initial release.

### zenpixels (interchange types)

**Pixel format description:**
- `PixelFormat` flat enum: `Rgb8`, `Rgba8`, `Rgb16`, `Rgba16`, `RgbF32`, `RgbaF32`, `Gray8`, `Gray16`, `GrayF32`, `GrayA8`, `GrayA16`, `GrayAF32`, `Bgra8`, `Rgbx8`, `Bgrx8`, `OklabF32`, `OklabaF32`
- `PixelDescriptor` with transfer function, alpha mode, color primaries, signal range
- 40+ predefined descriptor constants (`RGB8_SRGB`, `RGBAF32_LINEAR`, `BGRA8_SRGB`, etc.)
- `ChannelType`, `ChannelLayout`, `TransferFunction`, `ColorPrimaries`, `AlphaMode`, `SignalRange` enums
- `Cicp` struct with ITU-T H.273 code points and human-readable name lookups

**Pixel buffers:**
- `PixelBuffer<P>` (owned), `PixelSlice<'a, P>` (borrowed), `PixelSliceMut<'a, P>` (mutable borrowed)
- Phantom-typed `P: Pixel` for compile-time format safety, zero-cost `.erase()` / `.try_typed::<Q>()`
- SIMD-aligned allocation via `try_new_simd_aligned()`
- Row access: `row()`, `row_mut()`, `row_with_stride()`
- Contiguous access: `as_contiguous_bytes()`, `contiguous_bytes()` (Cow)
- Zero-copy views: `sub_rows()`, `crop_view()`, `crop_copy()`
- `Rgbx` and `Bgrx` 32-bit SIMD-friendly padded pixel types
- `GrayAlpha8`, `GrayAlpha16`, `GrayAlphaF32` pixel types

**Color metadata:**
- `ColorContext` (ICC + CICP, `Arc`-shared)
- `ColorOrigin`, `ColorProvenance`, `ColorProfileSource`, `NamedProfile`

**Conversion policies:**
- `ConvertOptions` with `AlphaPolicy`, `DepthPolicy`, `LumaCoefficients`, `GrayExpand`

**Multi-plane images** (behind `planar` feature):
- `PlaneLayout`, `PlaneDescriptor`, `PlaneSemantic`, `Subsampling`, `YuvMatrix`
- `MultiPlaneImage` container with per-plane `PixelBuffer`s
- YCbCr 4:2:0/4:2:2/4:4:4, Oklab planes, gain maps, separate alpha planes

**Interop** (behind feature gates):
- `rgb` feature: `Pixel` impls for `rgb` crate types
- `imgref` feature: `From<ImgRef>` / `From<ImgVec>` conversions, `as_imgref()` / `try_as_imgref::<P>()`

### zenpixels-convert (pixel math)

**Row conversion:**
- `RowConverter` with pre-computed conversion plan, no per-row allocation
- Three-tier dispatch: direct SIMD kernels, composed multi-step plans, hub path through linear sRGB f32
- Transfer function kernels: sRGB, BT.709, PQ (HDR10), HLG
- Depth scaling (u8/u16/f32), alpha mode changes, byte swizzle

**Format negotiation:**
- Two-axis cost model (effort vs. loss) with `ConvertIntent` weighting
- `best_match()`, `best_match_with()`, `negotiate()` entry points
- `Provenance` tracking for lossless round-trip detection
- `ideal_format()` for operation-aware format selection

**Gamut mapping:**
- 3x3 row-major f32 gamut matrices between BT.709, Display P3, BT.2020
- `conversion_matrix()`, `apply_matrix_row_f32()`, `apply_matrix_row_rgba_f32()`

**Oklab:**
- Primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`

**HDR:**
- Reinhard and exposure tone mapping
- `ContentLightLevel`, `MasteringDisplay`, `HdrMetadata`

**Codec integration:**
- `CodecFormats` registry with `FormatEntry` (effective bits, overshoot flag)
- `finalize_for_output()` for atomic pixel + metadata assembly
- `adapt_for_encode_explicit()` for policy-validated conversion
- `ConvertError` with specific variants (`NoMatch`, `NoPath`, `AlphaNotOpaque`, `DepthReductionForbidden`, `CmsError`)
- `ColorManagement` and `RowTransform` traits for external CMS backends

**Operation format requirements:**
- `OpCategory` and `OpRequirement` for operation-specific format suitability
- Conversion path analysis: `ConversionPath`, `LossBucket`, `generate_path_matrix()`
