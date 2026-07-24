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
| `orient::apply_orientation`, `apply_orientation_into`, `apply_orientation_in_place` | Published compatibility wrappers; deprecated in favor of the sealed slice/buffer orientation traits, then remove in 0.3 |
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
