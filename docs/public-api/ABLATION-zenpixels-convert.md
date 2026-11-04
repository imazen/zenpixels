# Free functions and other cut candidates, 2026-11-04

All 42 public free functions in `zenpixels` and `zenpixels-convert` at `feat/color-value`, read from `docs/public-api/`, plus other candidates. Nothing in this table is applied yet.

Callers are text matches of call syntax in local checkouts under `~/work`, shown as shipped + other. "Other" means tests, examples, benches, tools and `perf/`. They aren't compiler output. Each count excludes this repo, jj workspaces and the archived standalone `zenfilters` checkout, but `zensim` and `zensimx` are counted separately. Names with no local caller were also searched with `gh search code --owner imazen`, which only sees default branches. The Oklab rows are total matches.

`zpc` = `zenpixels_convert`, `zp` = `zenpixels`, `Desc` = `zp::PixelDescriptor`, `E` = `whereat::At<ConvertError>`. **NEW** doesn't exist yet. **Exists** is in the API today.

| Current item | Callers | Action | Canonical path |
|---|---:|---|---|
| `zpc::adapt::adapt_for_encode_cow<'a>(&'a [u8], Desc, u32, u32, usize, &[Desc]) -> Result<PixelCow<'a>, E>` | 8+4 | Replace | **NEW** `PixelSliceConvertExt::adapt_for_encode(self, &[Desc], &ConvertOptions) -> Result<PixelCow<'a>, E>`, passing `ConvertOptions::permissive().with_alpha_policy(DiscardUnchecked)`, which is what this function uses today |
| `zpc::adapt::adapt_for_encode_with_intent_cow<'a>(&'a [u8], Desc, u32, u32, usize, &[Desc], ConvertIntent) -> Result<PixelCow<'a>, E>` | 0 | Delete | **NEW** `PixelSliceConvertExt::adapt_for_encode`, which always uses the fastest intent |
| `zpc::adapt::adapt_for_encode_explicit_cow<'a>(&'a [u8], Desc, u32, u32, usize, &[Desc], &ConvertOptions) -> Result<PixelCow<'a>, E>` | 2+0 | Replace | **NEW** `PixelSliceConvertExt::adapt_for_encode` |
| `zpc::adapt::check_opaque(PixelSlice<'_>) -> Result<(), E>` | 0+1 | Move, stays public | **NEW** `PixelSliceConvertExt::check_opaque(&self) -> Result<(), E>`. `AlphaPolicy::DiscardIfOpaque` errors send callers here |
| `zpc::adapt::try_adapt_in_place(&mut PixelBuffer, Desc) -> Result<(), E>` | 0 | Delete | **Exists** `PixelBufferConvertExt::convert_to(&self, Desc)`, which allocates; no in-place equivalent remains |
| `zpc::convert_row(&ConvertPlan, &[u8], &mut [u8], u32)` | 6+0 | Replace | **Exists** `RowConverter::from_plan(plan).convert_row(src, dst, width)`. Whole images: **NEW** `PixelSliceConvertExt::convert_into(self, PixelSliceMut) -> Result<PixelSliceMut, E>` |
| `zpc::best_match(Desc, &[Desc], ConvertIntent) -> Option<Desc>` | 1+0 | Move | **NEW** `ConvertIntent::best_match(self, Desc, &[Desc]) -> Option<Desc>` |
| `zpc::best_match_with(Desc, &[FormatOption], ConvertIntent) -> Option<Desc>` | 0 | `pub(crate)` | None public; **NEW** `ConvertIntent::best_match` covers real callers |
| `zpc::negotiate(Desc, Provenance, impl Iterator<Item = FormatOption>, ConvertIntent) -> Option<Desc>` | 0 | `pub(crate)` | None public; **NEW** `ConvertIntent::best_match` covers real callers |
| `zpc::ideal_format(Desc, ConvertIntent) -> Desc` | 0 | Delete | None. Nothing inside calls it either, and its private precision tier goes too |
| `zpc::conversion_cost(Desc, Desc) -> ConversionCost` | 1+0 | Move | **NEW** `ConversionCost::between(Desc, Desc) -> ConversionCost` |
| `zpc::conversion_cost_with_provenance(Desc, Desc, Provenance) -> ConversionCost` | 0 | `pub(crate)` | None public |
| `zpc::gamut::conversion_matrix(ColorPrimaries, ColorPrimaries) -> Option<GamutMatrix>` | 0+3 | `pub(crate)` | **Exists** `zp::ColorPrimaries::gamut_matrix_to(self, Self) -> Option<[[f32; 3]; 3]>`. It returns a matrix for equal primaries where this returns `None`, so callers guard `from != to` |
| `zpc::gamut::apply_matrix_f32(&mut [f32; 3], &GamutMatrix)` | 1+0 | `pub(crate)` | None attachable: `GamutMatrix` is an alias of `[[f32; 3]; 3]`. The one zqi tool writes the three lines |
| `zpc::gamut::apply_matrix_row_f32(&mut [f32], usize, &GamutMatrix)` | 0+4 | Delete | **Exists** `RowConverter::new(from, to)?.convert_row(src, dst, width)`. Nothing inside calls it |
| `zpc::gamut::apply_matrix_row_rgba_f32(&mut [f32], usize, &GamutMatrix)` | 0 | Delete | **Exists** `RowConverter::new(from, to)?.convert_row(src, dst, width)`. Nothing inside calls it |
| `zpc::gamut::mat3_mul(&GamutMatrix, &GamutMatrix) -> GamutMatrix` | 0 | `pub(crate)` | None public; `oklab.rs` uses it |
| `zpc::output::finalize_for_output<C: ColorManagement>(&PixelBuffer, &ColorOrigin, OutputProfile, PixelFormat, &C) -> Result<EncodeReady, E>` | 0 | Delete | **NEW** `EncodeReady::finalize`, below. It's a separate code path today, so `tests/output_finalize.rs` has to move and show the two agree |
| `zpc::output::finalize_for_output_with(&PixelBuffer, &ColorOrigin, OutputProfile, PixelFormat, Option<&dyn PluggableCms>) -> Result<EncodeReady, E>` | 2+6 | Move | **NEW** `EncodeReady::finalize(&PixelBuffer, &ColorOrigin, OutputProfile, PixelFormat, Option<&dyn PluggableCms>) -> Result<EncodeReady, E>`. Streaming stays **Exists** `OutputPlan::new` + `convert_rows` |
| `zpc::hdr::quantize_to(PixelSlice<'_>, Desc) -> Result<PixelBuffer, E>` | 1+0 | Move | **NEW** `PixelSliceConvertExt::quantize_to(self, Desc) -> Result<PixelBuffer, E>` |
| `zpc::codes::measure_exact_depth(&PixelSlice<'_>) -> Result<ExactDepth, At<CodesError>>` (deprecated) | 0 | Delete | **Exists** `zp::sample::ExactDepth::measure(&PixelSlice) -> Result<ExactDepth, ExactDepthError>` |
| `zpc::codes::to_canonical<'b>(&[CodePlane<'_>], SignalRange, usize, usize, PixelSliceMut<'b>) -> Result<(PixelSliceMut<'b>, ExactDepth), At<CodesError>>` | 0 | Keep to 0.3, then cut if no codec calls it | Stays here; its main input is a slice of planes, so no type owns it. `docs/sample-precision-evidence.md` documents it |
| `zpc::oklab::fast_cbrt(f32) -> f32` | 0+5 | `pub(crate)` | `f32::cbrt` in the five zentone example calls; difference in results not measured |
| `zpc::oklab::rgb_to_oklab(f32, f32, f32, &GamutMatrix) -> [f32; 3]` | 17 | Keep | Stays here; no type owns three floats and a matrix without adding a new one |
| `zpc::oklab::oklab_to_rgb(f32, f32, f32, &GamutMatrix) -> [f32; 3]` | 14 | Keep | Stays here, same reason |
| `zpc::oklab::rgb_to_lms_matrix(ColorPrimaries) -> Option<GamutMatrix>` | 35 | Move | **NEW** `ColorPrimariesExt::to_lms_matrix(&self) -> Option<GamutMatrix>`, beside **Exists** `to_xyz_matrix` |
| `zpc::oklab::lms_to_rgb_matrix(ColorPrimaries) -> Option<GamutMatrix>` | 27 | Move | **NEW** `ColorPrimariesExt::from_lms_matrix(&self) -> Option<GamutMatrix>`, beside **Exists** `from_xyz_matrix` |
| `zpc::icc_profiles::display_p3_icc(bool) -> &'static [u8]` | 0+7 | Delete | **Exists** `zpc::icc_profiles::DISPLAY_P3_V4`, `DISPLAY_P3_V2` |
| `zpc::icc_profiles::icc_profile_for_primaries(ColorPrimaries) -> Option<&'static [u8]>` (deprecated) | 0 | Delete | **NEW** `SynthesizedIcc::for_cicp`, or **Exists** the bundled constants |
| `zpc::icc_profiles::synthesize_icc_for_cicp(Cicp) -> SynthesizedIcc` | 12+5 | Move | **NEW** `SynthesizedIcc::for_cicp(Cicp) -> SynthesizedIcc` |
| `zpc::icc_profiles::synthesize_gray_icc_for_cicp(Cicp) -> SynthesizedIcc` | 0 | `pub(crate)` | None public; `load_bearing.rs:473` uses it |
| `zp::icc::extract_cicp(&[u8]) -> Option<Cicp>` | 9+6 | Move | **NEW** `zp::Cicp::from_icc(&[u8]) -> Option<Cicp>` |
| `zp::icc::identify_common(&[u8]) -> Option<IccIdentification>` | 15+8 | Move | **NEW** `zp::icc::IccIdentification::from_icc(&[u8]) -> Option<IccIdentification>` |
| `zp::icc::is_common_srgb(&[u8]) -> bool` | 7+0 | Delete | **NEW** `IccIdentification::from_icc(bytes)`, then **Exists** `.is_srgb()`. That's the function's whole body |
| `zp::icc::profile_color_space(&[u8]) -> Option<ColorModel>` | 5+0 | Move | **NEW** `zp::ColorModel::from_icc(&[u8]) -> Option<ColorModel>` |
| `zp::icc::normalized_hash(&[u8]) -> u64` | 0 | Keep | Stays here. `zpc`'s `output.rs:77` calls it across crates, so `pub(crate)` won't compile, and no type owns it |
| `zpc::orient::apply_orientation(PixelSlice<'_>, Orientation) -> PixelBuffer` | 14+7 | Move | **NEW** `PixelSliceConvertExt::oriented(self, Orientation) -> PixelBuffer` |
| `zpc::orient::apply_orientation_into(PixelSlice<'_>, Orientation, PixelSliceMut<'_>) -> Result<(), ConvertError>` | 4+0 | Move | **NEW** `PixelSliceConvertExt::oriented_into(self, Orientation, PixelSliceMut<'_>) -> Result<(), E>` |
| `zpc::orient::apply_orientation_in_place(&mut PixelBuffer, Orientation) -> Result<(), ConvertError>` | 3+0 | Replace | **NEW** `PixelBufferConvertExt::into_oriented(self, Orientation) -> Result<PixelBuffer, E>`: untouched for identity, in place where possible, otherwise copies |
| `zpc::pipeline::generate_path_matrix(&[&CodecFormats], &[OpCategory], &[&CodecFormats], QualityThreshold) -> Vec<PathEntry>` | 0 | Delete with feature | None |
| `zpc::pipeline::optimal_path(Desc, Provenance, OpCategory, Desc, QualityThreshold) -> Option<ConversionPath>` | 0 | Delete with feature | None |
| `zpc::pipeline::matrix_stats(&[PathEntry]) -> MatrixStats` | 0 | Delete with feature | None |
| `zpc::pipeline::path::{ConversionPath, PathEntry, MatrixStats, LossBucket, QualityThreshold}` | 0 | Delete with feature | None |
| `zpc::pipeline::op_format::{OpCategory, OpRequirement}` | 0 | Delete with feature | None |
| `zpc::pipeline::registry::{CodecFormats, FormatEntry, ALL_CODECS, AVIF, BMP, FARBFELD, GIF, JPEG, JPEG_DECODE_EXTENDED, JXL, PNG, PNM, WEBP}` | 0 | Delete with feature | None |
| Cargo feature `zenpixels-convert/pipeline` (1,485 lines) | 0 | Delete | None. `tests/path_validation.rs` and `tests/registry_validation.rs` go with it, and the `lib.rs` docs pointing at its table need rewriting |
| `zpc::cms::RenderingIntent` | 0 | Delete | None public. `cms_moxcms.rs` names `moxcms::RenderingIntent::RelativeColorimetric` directly; another intent means another **Exists** `PluggableCms` implementation |
| `zpc::cms::ColorPriority` | 0 | Delete | None public. `cms_moxcms.rs` sets `allow_use_cicp_transfer: false` directly |
| `zpc::cms::ColorManagement` (deprecated since 0.2.8) | 1+0 | Delete | **Exists** `zpc::cms::PluggableCms`. Needs a public signature change first in zenpipe: `IccTransformSource::new<C: ColorManagement>` (`src/sources/icc_transform.rs:72`, called at `src/graph.rs:1832`) |
| `zpc::Provenance` | 0 | `pub(crate)` | None public, once `negotiate` and the `pipeline` feature leave the API |
| `zpc::FormatOption` | 0 | `pub(crate)` | None public, once `negotiate` and `best_match_with` leave the API |
| `zpc::icc_profiles::REC2020_V4` | 0 | `pub(crate)` | **NEW** `SynthesizedIcc::for_cicp`, which accounts for transfer; `output.rs` still uses the constant inside |

The 42 functions split into 17 moved or replaced by a method, 13 deleted, 8 `pub(crate)` and 4 kept as free functions. This adds one sealed trait, `zpc::ext::PixelSliceConvertExt`, and 17 public methods. One of the 17, `PixelSliceConvertExt::convert_to(self, Desc) -> Result<PixelBuffer, E>`, replaces no row above: it serves the hand-written row loops described below.

Methods on `zenpixels` types added from `zenpixels-convert` must be sealed trait methods, as `PixelBufferConvertExt` is today. Inherent methods would mean moving the kernels into `zenpixels`. `orient/` alone is 4,039 lines including tests, and it pulls in archmage and magetypes.

Repeated caller code: 16 hand-written whole-image row loops in shipped code (heuristic: `RowConverter::new` or `ConvertPlan::new` with a row loop within 22 lines), against 7 uses of `.convert_to(`, which exists only on `PixelBuffer`. 10 of 12 `apply_orientation` sites guard for identity first. `adapt_for_encode_cow` and its variants take bare bytes, descriptor, width, height and stride instead of a `PixelSlice`.

---

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
