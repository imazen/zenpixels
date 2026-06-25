# Public API delta: v0.2.14 → main (zenpixels 0.3.0 / zenpixels-convert 0.2.15)

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
| `zenpixels` | 19 | 9 | 0.3.0 (was 0.2.14) |
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
- **`ConvertPlan::estimate(w, h) -> (u64, f64)`** — predicting
  `(peak_memory_bytes, wall_time_ms)` for any conversion plan. The single
  tuple-returning method that survives pre-publish YAGNI cleanup; an
  earlier iteration in this work cycle elaborated `pub mod estimate` +
  `struct ResourceEstimate { peak_memory_bytes, wall_time_ms, breakdown,
  confidence }` + `struct StepEstimate` + `enum EstimateConfidence` and
  11 `estimate_*` shadow methods across `PixelBufferConvertExt` /
  `PixelBufferConvertTypedExt` / `PixelBufferHdrConvertExt` — all of that
  was scratched in favor of the one method on `ConvertPlan` (CHANGELOG
  "changed" entry for details).

### `zenpixels-convert` — feature additions (`zenpixels-convert.features.txt`, mostly behind `hdr-experimental`)

- **`hdr::Bt2446A { map_rgb, map_strip_simd, new }`** — ITU-R BT.2446
  Method A HDR→SDR tonemapper as a typed struct, with SIMD strip
  processing.
- **`hdr::SoftCompress { new, from_matrices, apply_strip, knee, lut,
  DEFAULT_KNEE }` + `hdr::GamutBoundaryLut { new, max_chroma,
  compress_planes }`** — gamut-aware compress pipeline used by
  `HdrToSdr`. `DEFAULT_KNEE` shipped at `0.96` (corpus-calibrated
  2026-06-23) vs the prior un-calibrated `0.9`.
- **`pub trait CllMeasure` (also reachable as `hdr::measure::CllMeasure`)
  with `measure_max(px, white, method)`** — promoted (production-default)
  ContentLightLevel reading. The percentile / smoothed / histogram
  variants live in `internal.txt` as `#[doc(hidden)]` opt-ins.
- **`pub trait PixelBufferHdrConvertExt { convert_to_sdr,
  convert_to_with_hdr_config, estimate_convert_to_sdr,
  estimate_convert_to_with_hdr_config }`** — fluent HDR-source ext
  methods on `PixelBuffer`.
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
  removed. The literal-maximum MaxCLL was outlier-sensitive (one bright
  pixel inflated it) and is replaced in `zenpixels-convert` by the
  `CllMeasure::measure_max` / `measure_robust` family. This is the only
  semver-breaking removal — it's what drove the `zenpixels` 0.2.14 →
  0.3.0 major bump.

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
- **`zenpixels`**: 0 fails reported against the v0.2.14 → 0.3.0
  (major) bump — the `ContentLightLevel::measure` removal IS a semver
  break, and the major version bump accommodates it. Everything else on
  the zenpixels surface is additive (new `with_*` relabel builders, new
  in-place ctor).

Net: the only hard break is `ContentLightLevel::measure`; the major
bump is required and sufficient to absorb it. All other deltas are
additive or visibility-changes inside the same major.
