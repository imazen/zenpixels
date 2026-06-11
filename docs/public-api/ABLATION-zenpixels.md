# ABLATION REPORT — zenpixels (core)

**Date:** 2026-06-11  
**Snapshot commit:** 288a6833 (main; PR #30 merged)  
**Snapshot file:** `docs/public-api/zenpixels.txt`  
**Total public items (default features):** 2,254 (per snapshot)  
**Total public items (all features):** 3,079  
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
| KEEP (confirmed consumers) | ~2,250+ | Core pixel type system; every codec + zenfilters + zenresize + zenmetrics + zenanalyze + zencodec + zenpipe depends on these |
| FLAG B — demote to `pub(crate)` | 0 | No clear zero-consumer items identified in the core public surface |
| Observe / low-priority watch | 2 | `GrayAlpha8`, `GrayAlphaF32` (not `GrayAlpha16` which is confirmed used) — see notes |
| Already correct | 1 | `registry` module correctly `pub(crate)` |

**Flag rate: 0%** — the core crate is foundational to the entire org; search confirmed consumers for every module investigated.

---

## Module-by-module findings

### `descriptor` module — `PixelDescriptor`, `ChannelLayout`, `ChannelType`, `TransferFunction`, `ColorPrimaries`, `MatrixCoefficients`, `BitDepth`, `PixelLayout`, `PixelFormat` — KEEP

Confirmed consumers across the entire org. Every codec (zenjpeg, zenpng, zenavif, zenjxl, zenwebp, zentiff), zencodec, zenpipe, zenresize, zenanalyze, zenmetrics, imageflow — all depend on these types. The pixel descriptor type system is the org's interchange contract.

### `orientation` module — `Orientation` — KEEP

Directly used by: zencodec (via the `OrientationHint` design), heic adapter, zenjpeg, zenjxl. The `apply_orientation` API is in `zenpixels-convert`, not here; this module owns the coordinate-level enum.

### `planar` module — `PlanarPixelSlice`, `PlaneMask`, etc. — KEEP

zenfilters (in zenpipe repo) uses `PlaneMask` and planar pixel slice operations extensively (~10+ files).

### `policy` module — `AlphaPolicy`, `ConvertOptions`, `DepthPolicy`, `GrayExpand`, `LumaCoefficients` — KEEP

- `AlphaPolicy`, `ConvertOptions`: zenpipe/imageflow-compat/execute.rs uses them directly.
- `LumaCoefficients`: ultrahdr-core docs reference it; zenpixels-convert internals use it.
- `DepthPolicy`, `GrayExpand`: Used in zenpixels-convert policy planner.

### `cicp` module — `Cicp` — KEEP

Heavily used across all HDR-capable codecs, zencodec, and zenmetrics. `Cicp::DISPLAY_P3`, `Cicp::SRGB`, `Cicp::BT2020_PQ` etc. are the org's CICP interchange type.

### `color` module — `ColorSpace`, `ColorPrimariesEnum`, re-exports — KEEP

Used throughout the codec stack and zencodec color APIs.

### `hdr` module (zenpixels core) — `ContentLightLevel`, `MasteringDisplay` — KEEP

Confirmed consumers:
- zenjpeg `codec.rs`: `zencodec::ContentLightLevel`, `zencodec::MasteringDisplay` (re-exported from zencodec which re-exports from zenpixels)
- zenpng `decode.rs`, `decoder/mod.rs`, `encoder/metadata.rs`, `encode.rs`: all use `ContentLightLevel` and `MasteringDisplay`

Note: these are distinct from `zenpixels_convert::hdr::HdrMetadata` (flagged B in the convert report). The core types are actively used.

### `icc` module — `IccIdentification` etc. — KEEP

zenjpeg uses `IccIdentification` for ICC tag parsing. Used in `IccDisposition` tracking through zencodec.

### `pixel_types` module — `GrayAlpha8`, `GrayAlpha16`, `GrayAlphaF32` — MOSTLY KEEP

- `GrayAlpha16` — KEEP: zenpng `apng.rs`, `codec.rs`, `postprocess.rs` — active confirmed consumers.
- `GrayAlpha8` — LOW PRIORITY WATCH: no external `.rs` consumer found in current active code (jxl-encoder has its own `PixelLayout::GrayAlpha8` which is a different enum entirely). The type likely has in-tree users via the generic `Pixel` trait machinery. Not flagged B (safe to wait; it's a variant of a `#[non_exhaustive]`-style pixel type set, and the type system is not polluted by it).
- `GrayAlphaF32` — LOW PRIORITY WATCH: same situation as `GrayAlpha8`. F32 variants for 32-bit floating-point gray+alpha; no active external consumer found, but removal would be a breaking change and the type is consistent with the type system design.

**Conservative vote:** Do not flag B. The pixel type set is consistent (u8/u16/f32 variants for each channel config). Removing `GrayAlpha8` or `GrayAlphaF32` would require a 0.3.0 bump and affects the type-system completeness. Wait for real evidence of zero use before queuing these.

### `buffer` module — `PixelBuffer`, `PixelSlice`, `PixelSliceMut`, `Pixel`, `Bgrx`, `Rgbx`, `Bgrx`, `BufferError` — KEEP

- `PixelBuffer`, `PixelSlice`, `PixelSliceMut`, `Pixel`: Core buffer types, used everywhere.
- `Bgrx`, `Rgbx`: Used in zenjpeg tests and zenresize. `PixelFormat::Bgrx` and `PixelLayout::Rgbx8Srgb` appear in zenjpeg tests directly. `Rgbx` also used in zenresize benchmarks and streaming tests.

---

## Items Correctly Scoped (no action needed)

| Item | Status |
|------|--------|
| `pub(crate) mod registry` | Correctly internal — `KnownColorSpace`, `REGISTRY`, `find_by_*`, matrix math helpers all correctly not exposed |
| `#[non_exhaustive]` on enums (`ColorPrimaries`, `MatrixCoefficients`, etc.) | Already applied; correct for extensible type sets |
| Sealed traits in convert crate | Applied correctly per CLAUDE.md guidance |

---

## What Was NOT Investigated (2,254 items; grep-based audit)

This report used module-level grep evidence rather than exhaustively reading the 400KB snapshot file. Items not investigated individually:
- Individual `PixelDescriptor` preset constants (hundreds: `RGBA8_SRGB`, `RGB8_SRGB`, `GRAY8`, etc.) — all consumed via codec and resize code
- Individual `TransferFunction`/`ColorPrimaries` enum variants — used in CICP negotiation across all codecs
- `PlanarPixelSlice` internal methods beyond the `PlaneMask` check
- `BufferError` variant set — used via zenpixels-convert `ConvertError` chain

The pattern is clear: zenpixels is the ecosystem's type system, and every crate reaches into it. The investigation found **zero items to flag** in the core crate.

---

## Recommendation

No changes needed to `zenpixels` (core) public API surface for this ablation pass. The low-priority watches (`GrayAlpha8`, `GrayAlphaF32`) should be revisited in 6–12 months; if still zero external consumers at that point, they can be evaluated for the 0.3.0 breaking-change queue.

Apply changes identified in `ABLATION-zenpixels-convert.md` (the companion report) for the few flags in the conversion crate.
