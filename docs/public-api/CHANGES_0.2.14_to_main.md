# Public API delta: v0.2.14 → main (zenpixels 0.2.16 / zenpixels-convert 0.2.15, both unreleased)

Generated 2026-06-23 by diffing the committed `docs/public-api/v0.2.14/*.txt`
snapshots (frozen at the `zenpixels-v0.2.14` / `zenpixels-convert-v0.2.14`
tags) against the regenerated `docs/public-api/*.txt` snapshots at the
current `main` tip. Both sides are byte-comparable: same `zenutils-apidoc`
runner, same encoding, same feature-axis bucketing.

Run `just api-doc` to refresh the current-main snapshots; the
`v0.2.14/` baseline is frozen.

> **Snapshot format note (2026-06-25):** the current-main snapshots are
> now generated with `zenutils-apidoc 0.1.2`'s `no_file_meta_header()` +
> `no_autotraits_summary()` builders enabled, which strip two
> perpetually-churning lines from each rendered file: the `# files: …`
> line-count header at the top of `<crate>.txt`, and the `X types
> implement all of: Freeze, RefUnwindSafe, Send, Sync, Unpin,
> UnwindSafe` summary line at the top of each `## auto traits` block.
> Both counters shifted on every regen as the API grew without
> conveying any semver signal. The `v0.2.14/` baseline still carries
> those lines (frozen format from the original `0.1.0` runner) — so the
> first few lines of every `main`-side header block and the auto-traits
> block will appear as net-removals in a raw `diff -u` against the
> baseline. The per-`pub`-line counts in the tables below already
> account for this; ignore the format-noise lines when reviewing.

## Summary

Per file, counting unique `pub …` lines:

| Crate | File | Added | Removed |
|---|---|---:|---:|
| `zenpixels` | `zenpixels.txt` (default surface) | 12 | 2 |
| `zenpixels` | `zenpixels.features.txt` (feature additions) | 0 | 7 |
| `zenpixels` | `zenpixels.internal.txt` (hidden / excluded) | 7 | 0 |
| `zenpixels-convert` | `zenpixels-convert.txt` (default surface) | 53 | 4 |
| `zenpixels-convert` | `zenpixels-convert.features.txt` (feature additions) | 46 | 0 |
| `zenpixels-convert` | `zenpixels-convert.internal.txt` (hidden / excluded) | 28 | 0 |

Per crate (sum over the three files):

| Crate | Added (net `pub` lines) | Removed (net `pub` lines) | Version on main |
|---|---:|---:|---|
| `zenpixels` | 19 | 9 | 0.2.16 (was 0.2.14, unreleased) |
| `zenpixels-convert` | 127 | 4 | 0.2.15 (was 0.2.14) |

Header-block line counts on `main` vs `v0.2.14` (after the 2026-06-25
noise-reduction regen — see the format note above):

| File | v0.2.14 | main |
|---|---:|---:|
| `zenpixels.txt` | 855 | 874 |
| `zenpixels.features.txt` | 302 | 293 |
| `zenpixels.internal.txt` | 105 | 117 |
| `zenpixels-convert.txt` | 353 | 426 |
| `zenpixels-convert.features.txt` | 279 | 317 |
| `zenpixels-convert.internal.txt` | 90 | 128 |

## Most notable additions

### `zenpixels-convert` — default surface (`zenpixels-convert.txt`)

- **`ConvertError::NeedsCms { from, to }`** — new typed-error variant for
  `(from, to)` pairs that are structurally valid but need a CMS plugin
  (CMYK today; Lab / XYZ / spot inks in the future). Replaces the
  pre-0.2.15 `assert_not_cmyk` panic on six entry points; makes the
  documented `RowConverter::new_explicit_with_cms(_, _, _, Some(&MoxCms))`
  escape hatch reachable. Distinct from `NoPath` (no architecturally
  possible conversion).
- **`ConvertError::Buffer(BufferError)` + `ConvertError::HdrSourceRequiresPeak { from, to }`**
  — two more typed-error variants surfacing previously panicking raw-bytes
  paths and HDR plans that lacked a peak luminance.
- **`pub fn requires_cms(&PixelDescriptor, &PixelDescriptor) -> bool`** —
  predicate for schedulers to probe (source, target) pairs before
  attaching a CMS plugin to a batch. Returns true iff either side's
  `ColorModel` is outside the native `{Gray, Rgb, Oklab}` set.
- **`pub mod estimate` + `ConvertPlan::estimate(w, h) -> ResourceEstimate`
  and `ConvertPlan::estimate_in(&image, &compute) -> ResourceEstimate`** —
  resource projection for any conversion plan. The shipped shape is the
  `estimate` module with `ResourceEstimate { peak_memory_bytes_est, wall_ms,
  intermediate_buffer_count }` + `ComputeEnvironment` + `ImageCharacteristics`
  + `SimdTier` (all `#[non_exhaustive]`, builder-constructed, locally defined
  and shape-compatible with `zencodec::estimate::*`). The work cycle churned
  through two discarded intermediates before this: a bare
  `estimate(w, h) -> (u64, f64)` tuple, and a wider `ResourceEstimate {
  breakdown, confidence, … }` + `StepEstimate` + `EstimateConfidence` + a
  family of `estimate_*` shadow methods on the `PixelBuffer*ConvertExt`
  traits. Both were scratched; the module + two `ConvertPlan` methods are
  what actually ship (CHANGELOG "changed" entries for the full arc).

### `zenpixels-convert` — feature additions (`zenpixels-convert.features.txt`, mostly behind `hdr-experimental`)

- **`hdr::Bt2446A { map_rgb, map_strip_simd, new }`** — ITU-R BT.2446
  Method A HDR→SDR tonemapper as a typed struct, with SIMD strip
  processing.
- **`hdr::SoftCompress { new, from_matrices, apply_strip, knee, lut,
  DEFAULT_KNEE }` + `hdr::GamutBoundaryLut { new, max_chroma,
  compress_planes }`** — gamut-aware compress pipeline used by the
  `ConvertPlan` HDR tone-map path (the `SoftCompressOklch` step; the earlier
  standalone `HdrToSdr` type was folded into `ConvertPlan`). `DEFAULT_KNEE`
  shipped at `0.96` (corpus-calibrated 2026-06-23) vs the prior
  un-calibrated `0.9`.
- **`pub trait CllMeasure` (also reachable as `hdr::measure::CllMeasure`)
  with `measure_max(px, white, method)`** — promoted (production-default)
  ContentLightLevel reading. The percentile / smoothed / histogram
  variants live in `internal.txt` as `#[doc(hidden)]` opt-ins.
- **`pub trait PixelBufferHdrConvertExt { convert_to_sdr,
  convert_to_with_hdr_config }`** — fluent HDR-source ext methods on
  `PixelBuffer`. (The `estimate_convert_to_*` shadow methods from an earlier
  iteration were removed with the rest of the `estimate_*` family; estimation
  is now via `ConvertPlan::estimate`.)
- **`ConvertPlan::new_with_hdr_peak` + `ConvertPlan::new_with_hdr_config`**
  — plan-builders for HDR sources with explicit peak luminance or full
  `HdrConfig`.

### `zenpixels` — default surface (`zenpixels.txt`)

- **Buffer-level relabel builders: `PixelBuffer::with_{cicp,icc,diffuse_white}`,
  `PixelSlice::with_{cicp,icc,diffuse_white}`, `PixelSliceMut::with_{cicp,icc,diffuse_white}`,
  `ColorContext::with_{cicp,icc,diffuse_white}`** — chainable mutators
  that attach CICP / ICC / DiffuseWhite to an existing buffer without
  copying.
- **`InPlacePixels::new(buf, w, h, stride, descriptor, ColorContext)`**
  — convenience constructor on the existing in-place borrow.

### `zenpixels-convert` — internal / hidden additions (`zenpixels-convert.internal.txt`)

- **`hdr::measure` module** — `LightLevelHistogram { bins, max, mean,
  percentile, total_pixels, method, BIN_MIN_NITS, BIN_MAX_NITS,
  NUM_BINS }`, `LightLevelMethod { LuminanceBt2020, MaxRgb }`, plus
  `CllMeasure::{measure_histogram, measure_percentile,
  measure_max_smoothed, measure_robust}` and the mirror
  `ContentLightLevel::measure_*` re-exports. Behind `hdr-experimental`
  +  `#[doc(hidden)]` so the production default
  (`CllMeasure::measure_max`) stays visible while the alternative
  measures remain accessible.
- **`hdr::{reinhard_tonemap, reinhard_inverse, exposure_tonemap}`** —
  previously default-surface free functions, now `#[doc(hidden)]` and
  queued for deletion (CHANGELOG `QUEUED BREAKING CHANGES` — naive
  global Reinhard / bare-exposure, both superseded by `zentone`).

## Most notable removals or visibility changes

### `zenpixels` — default surface

- **`ContentLightLevel::measure(PixelSlice<'_>, DiffuseWhite) -> Option<Self>`**
  moved off the default surface to `zenpixels.internal.txt`: it is now
  `#[deprecated]` + `#[doc(hidden)]` but **still present and fully
  functional** (an earlier pass on this unreleased line replaced its body
  with an `unimplemented!()` shim and mislabeled the removal as a shipped
  0.3.0; commit 6019aeef restored the working 0.2.14 body). The maintained
  replacement is `zenpixels-convert`'s `CllMeasure::measure_max` /
  `measure_robust` family; the actual removal stays in the CHANGELOG's
  QUEUED BREAKING CHANGES for the next breaking release.

### `zenpixels` — feature surface (`zenpixels.features.txt`)

- **`planar::Plane` (struct, plus `buffer` and `semantic` fields) and
  `planar::PlaneLayout`'s `[also: planar]` re-export annotation, plus
  `planar::MultiPlaneImage`'s `[also: planar]` annotation** moved to
  `zenpixels.internal.txt` (still reachable but `#[doc(hidden)]`).
  `PlaneLayout` itself remains in the default features surface.

### `zenpixels-convert` — default surface

- **`enum ConvertError [also: error]`** — the `[also: error]` re-export
  annotation dropped because `ConvertError` is now exposed at one
  canonical path only.
- **`exposure_tonemap`, `reinhard_inverse`, `reinhard_tonemap`** — moved
  from the default surface to `internal.txt` (`#[doc(hidden)]`), see
  above.

## Backwards compatibility

`cargo semver-checks check-release` against the v0.2.14 baseline, with
default features:

- **`zenpixels-convert`**: 196 checks pass, 0 fail, 56 skip — the
  v0.2.14 → 0.2.15 (minor) bump is correct. All new variants land on the
  `#[non_exhaustive] ConvertError` enum; no inherent methods, free
  functions, or trait items were removed.
- **`zenpixels`**: the v0.2.14 → 0.2.16 bump stays inside the 0.2.x
  line. `ContentLightLevel::measure` is NOT removed — it is
  `#[deprecated]` + `#[doc(hidden)]` with its working 0.2.14 body
  restored (6019aeef), so the signature and behavior both survive.
  Everything else on the zenpixels surface is additive (new `with_*`
  relabel builders, new in-place ctor).

Net: no hard break ships on this line. The `ContentLightLevel::measure`
*removal* — which WOULD be a semver break requiring the leading-digit
bump — remains queued in the CHANGELOG's QUEUED BREAKING CHANGES for
the next breaking release.
