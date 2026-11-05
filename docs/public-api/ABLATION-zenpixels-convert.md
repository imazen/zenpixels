# Free functions and other cut candidates, 2026-11-04

Every public free function in `zenpixels` and `zenpixels-convert`, plus other candidates, with where each went. All of it is applied on `feat/api-ablation`, except the two rows marked **not yet**.

`zpc` = `zenpixels_convert`, `zp` = `zenpixels`, `Desc` = `zp::PixelDescriptor`, `E` = `whereat::At<ConvertError>`.

Callers are text matches of call syntax in local checkouts under `~/work`, not compiler output: shipped + other, where other means tests, examples, benches, tools and `perf/`. `zensim` and `zensimx` count separately, so some totals overstate real sites. Oklab rows are totals.

| Old item | Callers | Canonical path now |
|---|---:|---|
| `zpc::adapt::adapt_for_encode_cow` | 8+4 | `PixelSliceConvertExt::adapt_for_encode(&self, &[Desc], &ConvertOptions) -> Result<PixelCow<'a>, E>`. The old behavior is `ConvertOptions::permissive().with_alpha_policy(DiscardUnchecked)` |
| `zpc::adapt::adapt_for_encode_with_intent_cow` | 0 | Removed. Encode paths use the fastest intent |
| `zpc::adapt::adapt_for_encode_explicit_cow` | 2+0 | `PixelSliceConvertExt::adapt_for_encode` |
| `zpc::adapt::check_opaque(PixelSlice)` | 0+1 | `PixelSliceConvertExt::check_opaque(&self) -> Result<(), E>` |
| `zpc::adapt::try_adapt_in_place` | 0 | `PixelBufferConvertExt::try_adapt_in_place(&mut self, Desc) -> Result<(), E>`. Kept: it's the only format change with no allocation |
| `zpc::convert_row` | 6+0 | `RowConverter::from_plan(plan).convert_row`. Whole images: `PixelSliceConvertExt::convert_into(&self, PixelSliceMut) -> Result<PixelSliceMut, E>` and `convert_to` |
| `zpc::best_match` | 1+0 | `ConvertIntent::best_match(self, Desc, &[Desc]) -> Option<Desc>` |
| `zpc::best_match_with` | 0 | Removed: `ConvertIntent::negotiate` with `Provenance::from_source` |
| `zpc::negotiate` | 0 | `ConvertIntent::negotiate(self, Desc, Provenance, impl Iterator<Item = FormatOption>) -> Option<Desc>`. No caller yet, kept for the unique job: origin depth and consumer fused-path costs |
| `zpc::ideal_format` | 0 | Removed, with its private precision tier |
| `zpc::conversion_cost` | 1+0 | `ConversionCost::between(Desc, Desc) -> ConversionCost` |
| `zpc::conversion_cost_with_provenance` | 0 | `ConversionCost::between_with_provenance(Desc, Desc, Provenance)`. Kept: nothing else says narrowing back to the origin depth is free |
| `zpc::Provenance`, `zpc::FormatOption` | 0 | Kept public, as inputs to the two above |
| `zpc::gamut::conversion_matrix` | 0+3 | `GamutMatrix::between(from, to) -> Option<GamutMatrix>`. Equal primaries now give the identity, not `None` |
| `zpc::gamut::apply_matrix_f32` | 1+0 | `GamutMatrix::apply(&self, [f32; 3]) -> [f32; 3]` |
| `zpc::gamut::apply_matrix_row_f32` | 0+4 | `GamutMatrix::apply_row_rgb(&self, &mut [[f32; 3]])`, in place |
| `zpc::gamut::apply_matrix_row_rgba_f32` | 0 | `GamutMatrix::apply_row_rgba(&self, &mut [[f32; 4]])`, in place, alpha unchanged |
| `zpc::gamut::mat3_mul(a, b)` | 0 | `b.then(&a)` on `GamutMatrix`. Also new: `GamutMatrix::inverse`. The array is private: `from_rows` and `rows()` |
| `zpc::output::finalize_for_output` | 0 | Removed: `EncodeReady::finalize` |
| `zpc::output::finalize_for_output_with` | 2+6 | `EncodeReady::finalize(&PixelBuffer, &ColorOrigin, OutputProfile, PixelFormat, Option<&dyn PluggableCms>)`. Streaming stays `OutputPlan` |
| `zpc::hdr::quantize_to` | 1+0 | `PixelSliceConvertExt::quantize_to(&self, Desc) -> Result<PixelBuffer, E>` |
| `zpc::codes::measure_exact_depth` | 0 | Removed: `zp::sample::ExactDepth::measure` |
| `zpc::codes::to_canonical` | 0 | Stays free until 0.3, then cut if no codec calls it. Its main input is a slice of planes, so no type owns it |
| `zpc::oklab::fast_cbrt` | 0+5 | `pub(crate)`. zentone's examples use `f32::cbrt` |
| `zpc::oklab::rgb_to_oklab` | 17 | `OklabSpace::from_rgb(&self, [f32; 3]) -> [f32; 3]` |
| `zpc::oklab::oklab_to_rgb` | 14 | `OklabSpace::to_rgb(&self, [f32; 3]) -> [f32; 3]` |
| `zpc::oklab::rgb_to_lms_matrix` | 35 | `OklabSpace::for_primaries(p)?.rgb_to_lms() -> &GamutMatrix` |
| `zpc::oklab::lms_to_rgb_matrix` | 27 | `OklabSpace::for_primaries(p)?.lms_to_rgb() -> &GamutMatrix` |
| `zpc::hdr::SoftCompress::{new, from_matrices}` | 0 | `SoftCompress::new(&OklabSpace, knee)`, which can't panic or take a mismatched pair |
| `zpc::icc_profiles::display_p3_icc` | 0+7 | Removed: `DISPLAY_P3_V4`, `DISPLAY_P3_V2` |
| `zpc::icc_profiles::icc_profile_for_primaries` | 0 | Removed: `SynthesizedIcc::for_cicp` or the constants |
| `zpc::icc_profiles::synthesize_icc_for_cicp` | 12+5 | `SynthesizedIcc::for_cicp(Cicp) -> SynthesizedIcc` |
| `zpc::icc_profiles::synthesize_gray_icc_for_cicp` | 0 | `pub(crate)`. Used by `load_bearing` |
| `zpc::icc_profiles::REC2020_V4` | 0 | `pub(crate)`. `SynthesizedIcc::for_cicp` accounts for transfer |
| `zp::icc::extract_cicp` | 9+6 | `zp::Cicp::from_icc(&[u8]) -> Option<Cicp>` |
| `zp::icc::identify_common` | 15+8 | `zp::icc::IccIdentification::from_icc(&[u8]) -> Option<IccIdentification>` |
| `zp::icc::is_common_srgb` | 7+0 | `zp::icc::IccIdentification::is_common_srgb(&[u8]) -> bool` |
| `zp::icc::profile_color_space` | 5+0 | `zp::ColorModel::from_icc(&[u8]) -> Option<ColorModel>` |
| `zp::icc::normalized_hash` | 0 | Stays free: `zpc` calls it across crates and no type owns it. It stays FNV-1a, because the identification tables hold its values |
| `zpc::orient::apply_orientation` | 14+7 | `PixelSliceConvertExt::oriented(&self, Orientation) -> Result<PixelBuffer, E>`, which no longer panics on a failed allocation |
| `zpc::orient::apply_orientation_into` | 4+0 | `PixelSliceConvertExt::oriented_into(&self, Orientation, PixelSliceMut) -> Result<(), E>` |
| `zpc::orient::apply_orientation_in_place` | 3+0 | `PixelBufferConvertExt::into_oriented(self, Orientation) -> Result<PixelBuffer, E>`: untouched for identity, in place where possible, copies for pixels over 16 bytes |
| `zpc::pipeline::path::LossBucket` | 0 | `zpc::LossBucket`, with `ConversionCost::loss_bucket()`. Variants renamed, because `Lossless` covered small nonzero losses |
| `zpc::pipeline::path::QualityThreshold` | 0 | `ConvertIntent::best_match_within(self, Desc, &[Desc], LossBucket) -> Option<Desc>` |
| `zpc::pipeline::*` otherwise, feature `pipeline` | 0 | Removed. See below |
| `zpc::hdr::GamutBoundaryLut` | 1+0 | Removed: `SoftCompress::compress_oklab_planes(&self, &[f32], &mut [f32], &mut [f32]) -> Result<(), E>` |
| `zpc::cms::RenderingIntent`, `ColorPriority` | 0 | Removed. See below |
| `zpc::cms::ColorManagement` | 1+0 | Removed: `PluggableCms`. zenpipe migrates at the pin bump |

## What replaced the removed types

- **`pipeline`** chose *which* working format to use for a source, an operation category and an output, from static tables of 13 operation categories and 10 codec format lists, and from a cost it scored but never executed. Conversions never ran through it, and nothing outside this repo enabled it. `ConvertIntent::best_match`, `::best_match_within` and `::negotiate` do the choosing, with each codec's live `supported_descriptors()` in place of the copied table. Still gone: the 13 operation categories, the whole-chain solver `optimal_path`, and `FormatEntry`'s `effective_bits` and `can_overshoot`.
- **`RenderingIntent`** was never wired to the moxcms backend, which always used relative colorimetric. Matrix profiles have no rendering tables, and all four moxcms intents gave identical output on 4,913 colors per gamut. Image gamut policy is `ConvertOptions::gamut_mapping`; authored tables stay a `PluggableCms` matter.
- **`ColorPriority`** duplicated `zp::ColorAuthority`, which is what decoders set per image. See `docs/color-signal-precedence.md`.

# ABLATION REPORT — zenpixels-convert

**Date:** 2026-06-11  
**Snapshot commit:** 288a6833 (main; PR #30 merged)  
**Snapshot file:** `docs/public-api/zenpixels-convert.txt`  
**Total public items (default features):** 740 (per snapshot header count)  
**Total public items (all features):** 1,173  
**Grep template:**
```
ugrep -rn "TERM" /home/lilith/work/zen/ \
  --exclude-dir=target --exclude-dir=.jj --exclude-dir=retired \
  --exclude-dir=perm-corpus --include="*.rs" \
  | grep -v "^/home/lilith/work/zen/zenpixels/"
```

---

## Summary

| Verdict | Count | Notes |
|---------|-------|-------|
| KEEP (confirmed consumers) | ~710 | Core conversion plumbing, icc_profiles staples, oklab, gamut, orient, output, cms, negotiate, converter, ext, pipeline, error |
| FLAG A — `#[deprecated]` already | 1 | `icc_profiles::icc_profile_for_primaries` (deprecated in 0.2.12 CHANGELOG) — already marked, nothing to do |
| FLAG A — `#[deprecated]` recommend | 1 | `icc_profiles::ADOBE_RGB_V4` — already marked `#[deprecated]` in source |
| FLAG B — demote to `pub(crate)` or remove | 5–6 | `hdr::HdrMetadata`, `hdr::reinhard_tonemap`, `hdr::reinhard_inverse`, `hdr::exposure_tonemap`; `icc_profiles::PROPHOTO_V4` (empty `&[]`); `icc_profiles::REC2020_V4`; `icc_profiles::display_p3_icc`; `load_bearing` module |
| Abstain (zero-consumer, NOT flagged) | 0 | load_bearing is flagged because CHANGELOG explicitly named intended consumers that grep confirms are NOT yet wired |

**Flag rate (conservative):** ~7–8 items out of 740 = ~1%. Consistent with expectation for an org-interchange crate.

---

## Flagged Items — Action Required

### B: `hdr` module — `HdrMetadata`, `reinhard_tonemap`, `reinhard_inverse`, `exposure_tonemap`

**Location:** `zenpixels-convert/src/hdr.rs`

**Evidence:** 0 external consumers in any active `.rs` file across `/home/lilith/work/zen/`.
- `HdrMetadata` struct: 0 hits outside zenpixels. `retired/zenimage` has its own unrelated `HdrMetadata` struct with a different shape and different source (that one is about extracting HDR metadata from file bytes, not a pixel-level HDR property).
- `reinhard_tonemap(f32) -> f32`: 0 external callers. `retired/zenimage` has its own `reinhard_tonemap` with a different signature (`gamma_correction: f32` parameter) at a different path.
- `reinhard_inverse(f32) -> f32`: 0 external callers.
- `exposure_tonemap(f32, f32) -> f32` (`std`-gated): 0 external callers.
- zenmetrics's `hdr` references are all `zenmetrics_api::hdr`, a completely separate module.

**Recommendation:** Demote `hdr` module to `pub(crate)`. These are plumbing that serves internal conversion steps. The `exposure_tonemap` std-gated fn is particularly speculative. When an external HDR-math caller materializes, wire through zenpixels-convert's `finalize_for_output_with` or a dedicated public API designed around the actual use case.

**Note on `HdrMetadata` struct fields:** The struct has `pub transfer`, `pub content_light_level`, `pub mastering_display`. These are distinct from `zenpixels::ContentLightLevel` and `zenpixels::MasteringDisplay` which are confirmed KEEP (widely used via `zencodec` re-exports). The `zenpixels_convert::hdr::HdrMetadata` aggregation type is zero-consumer.

---

### B: `icc_profiles::PROPHOTO_V4` — empty stub

**Location:** `zenpixels-convert/src/icc_profiles.rs:126`

```rust
pub const PROPHOTO_V4: &[u8] = &[];
```

**Evidence:** The constant is literally an empty byte slice — no ICC profile is actually embedded. 0 external consumers found (only `retired/zenimage` has its own `PROPHOTO_V4` pointing at a real `.icc` file). Shipping an `&[]` as a public const misleads any caller that assumes it's a valid ICC profile.

**Recommendation:** Either (a) embed the actual ProPhoto ICC profile bytes and find/add a real consumer, or (b) demote to `pub(crate)` (or remove entirely, queued for 0.3.x). The empty bytes are a footgun.

---

### B: `icc_profiles::REC2020_V4` — zero external consumers

**Location:** `zenpixels-convert/src/icc_profiles.rs:134`

**Evidence:** 0 external callers in active `.rs` files. The constant exists and points at a real `.icc` file, but nothing uses it. The `synthesize_icc_for_cicp` path (which produces Rec.2020 profiles) uses the bundled blob, not this constant directly.

**Recommendation:** Demote to `pub(crate)`. If a caller needs a Rec.2020 ICC profile, they should use `synthesize_icc_for_cicp(Cicp::BT2020_PQ)` or similar — the transfer-aware path. A primaries-only Rec.2020 const will mis-tag SDR vs HDR.

---

### B: `icc_profiles::display_p3_icc(prefer_v2: bool) -> &'static [u8]` — zero consumers

**Location:** `zenpixels-convert/src/icc_profiles.rs:194`

**Evidence:** 0 external `.rs` callers. Referenced in `zenpipe/tests/wide_gamut_pipeline_gap.rs:258-260` only as a comment noting it could be used if the function existed in `zencodecs::cms`. Not called anywhere.

**Recommendation:** Demote to `pub(crate)`. `DISPLAY_P3_V2` and `DISPLAY_P3_V4` consts are the well-used interface (confirmed by zenjpeg tests, zenpng tests, zentiff tests). The selector function adds no value externally since callers know which version they want.

---

### A: `icc_profiles::ADOBE_RGB_V4` — deprecated alias (already done)

**Location:** `zenpixels-convert/src/icc_profiles.rs:118`

```rust
#[deprecated(note = "renamed to ADOBE_RGB")]
pub const ADOBE_RGB_V4: &[u8] = ADOBE_RGB;
```

**Status:** Already correctly marked `#[deprecated]`. No action needed. Listed here for completeness.

---

### A: `icc_profiles::icc_profile_for_primaries` — deprecated (already done)

**Location:** `zenpixels-convert/src/icc_profiles.rs:178`

**Status:** CHANGELOG 0.2.12 marks this deprecated, recommending `synthesize_icc_for_cicp` instead. Confirm the `#[deprecated]` attribute is in source (verified: the CHANGELOG entry says "deprecated"). No additional action needed here.

---

### B: `load_bearing` module — designed consumers not yet wired

**Location:** `zenpixels-convert/src/load_bearing.rs`; re-exported at lib.rs:499

**Exported items:**
- `LoadBearingReport`
- `PixelSliceLoadBearingExt` (sealed trait)
- `PixelSliceMutLoadBearingExt` (sealed trait)

**Evidence:** 0 external `.rs` callers across all zen repos. CHANGELOG PR #30 explicitly names intended consumer codecs: "per-encoder audit ... zenwebp/zenavif/zenjxl/zentiff". Grepping those repos' `src/` confirms none have wired it yet.

**Context:** CHANGELOG describes this as "designed for" those four codecs. The API is fresh (PR #30, 2026-06-10). The sealed-trait pattern means external implementation is not the intent — the module is consumed via the extension methods, not implemented.

**Recommendation:** This is a judgment call. Two options:
1. **Keep public** — the design intent is to wire four codecs imminently, and the sealed-trait structure means there's no accidental impl risk. The API is sound.
2. **Demote to `pub(crate)` until first codec wires it** — consistent with CLAUDE.md's "no pub without a concrete current consumer" rule. Promote back to `pub` simultaneously with the first codec PR.

**Conservative ablation vote: FLAG B** — demote to `pub(crate)` before next release, promote simultaneously with the first codec consumer. If the codec wiring is imminent (within the same 0.2.x series), this may be skipped; document the intent.

---

## Confirmed-KEEP Modules (high-confidence, grep-verified)

| Module / Item | Evidence |
|---|---|
| `oklab::*` (all 4 constants + 5 functions) | zenfilters 7+ files: `lms_to_rgb_matrix`, `OKLAB_FROM_LMS_CBRT`, `LMS_CBRT_FROM_OKLAB`, `rgb_to_oklab`, `fast_cbrt` (implicit) |
| `gamut::GamutMatrix`, `mat3_mul`, `apply_matrix_*` | zenfilters 14+ files; `_dbg/wv.rs` |
| `icc_profiles::SynthesizedIcc`, `synthesize_icc_for_cicp` | zenjpeg, zenpng, zenavif, zentiff active `.rs` |
| `icc_profiles::DISPLAY_P3_V4`, `DISPLAY_P3_V2` | zenjpeg tests, zenpng tests, zentiff tests |
| `icc_profiles::ADOBE_RGB` | zenjpeg `codec.rs` (AdobeRGB ICC embed path) |
| `orient::apply_orientation`, `apply_orientation_into` | Fresh deliberate API (PR #30, June 2026); KEEP per task instructions |
| `converter::RowConverter` | zenanalyze `row_stream.rs` (multiple worktrees + main); zenmetrics `decode.rs` |
| `ConvertPlan`, `convert_row` | zenmetrics cvvdp pipeline, zenmetrics-api, zenmetrics-gpu-core |
| `PixelBufferConvertTypedExt` | zenpng, zenavif, zenanalyze, `_dbg/wv.rs` |
| `PixelBufferConvertExt` | zenpng, zenpipe/zencodecs (re-exported publicly), zencodecs tests |
| `ColorPrimariesExt`, `TransferFunctionExt` | zenpipe ext traits |
| `MoxCms` | zenpipe graph.rs, zenpipe/tests/cms.rs, zenpipe lib.rs re-export; zenjxl-decoder |
| `adapt::adapt_for_encode`, `Adapted`, `adapt_for_encode_explicit` | zenpipe zencodecs docs + dispatch; referenced in several encode paths |
| `negotiate::Provenance`, `best_match`, `best_match_with`, `negotiate`, `ideal_format`, `ConvertIntent`, `ConversionCost`, `FormatOption`, `conversion_cost` | Internal plumbing for format negotiation; used throughout zenpixels-convert itself and by zenpipe ops |
| `output::finalize_for_output`, `finalize_for_output_with`, `EncodeReady`, `OutputMetadata`, `OutputProfile` | zencodec docs reference as canonical lowering path; zenpipe uses |
| `pipeline::*` (RowPipeline etc.) | Internal pipeline plumbing |
| `error::ConvertError` | Re-exported at crate root; zenmetrics uses it in return types |
| `cms::*` | CMS trait abstractions used by MoxCms, zenpipe |
| `ext::PixelBufferConvertExt` | Confirmed above |
| `hdr::HdrMetadata::is_hdr`, `is_sdr` | (ONLY relevant if `HdrMetadata` stays pub; if demoted, moot) |
| `__bench_*` modules | Internal benchmark harness, double-underscore = explicitly internal |
| `__trace_ops` | Internal tracing, double-underscore = explicitly internal |

---

## Items Requiring No Action (already correct)

- `builtin_profiles` — correctly `pub(crate)` in `lib.rs:403`
- `registry` (in zenpixels core) — correctly `pub(crate)`
- All sealed traits — sealed pattern is correct
- `__bench_u16_hybrids`, `__bench_scan`, `__trace_ops` — double-underscore convention signals internal; keep as-is
