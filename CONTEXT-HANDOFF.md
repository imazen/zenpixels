# Context Handoff — zenpixels color model

Last updated: 2026-04-15. Post 0.2.8 release.

## What shipped 2026-04-14/15

- **zenpixels 0.2.8** — `ConvertOptions::clip_out_of_gamut` + `#[non_exhaustive]` (tolerated technical break per `CLAUDE.md` §0.2.x policy).
- **zenpixels-convert 0.2.8** — `PluggableCms`, `RowTransformMut`, `CmsPluginError` wrapped in `whereat::At`, `finalize_for_output_with`, `ZenCmsLite` as default CMS (no feature gate), fused u8/u16 matlut kernels (3×/49× vs prior).
- **zencodec 0.1.17** — `SourceColor::to_color_context()` (drop-dupe authority), `descriptor_for_decoded_pixels_v2` (widened `corrected_to`, authority-aware, placebo param dropped), `resolve_color` primitive.

## Open tracking issues (read these first)

- **imazen/zenpixels#16** — `HdrProvenance` design for 0.3.0. Luma-only gain-map-first HDR with mirror-split encode. Unifies CLL/MDCV/tone-map/gain-map history into one `ColorContext::hdr` slot.
- **imazen/zencodec#11** — Cross-codec decoder audit. SourceColor/ImageInfo wiring gaps across all zen codecs. Tables comparing what each codec populates.
- **imazen/zenjpeg#87** — zenjpeg migration to `_v2` (5 call sites) + UltraHDR detection roadmap.

## The mental model — how color metadata flows

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
│ raw file     │   │ codec decoder    │   │ zencodec     │   │ pipeline /   │
│ bytes        │──▶│ (zenjpeg, etc)   │──▶│ types        │──▶│ CMS / encode │
└──────────────┘   └──────────────────┘   └──────────────┘   └──────────────┘
                                                  │
                         SourceColor ─────────────┤
                         (codec boundary)         │
                                                  ▼
                                          ColorContext
                                          (travels with
                                          PixelBuffer via Arc)
```

### Two types — `SourceColor` vs `ColorContext`

- **`SourceColor`** — lives at the codec boundary. Rich: `{icc, cicp, color_authority, bit_depth, channel_count, content_light_level, mastering_display}`. Captures everything the file said, including roundtrip metadata.
- **`ColorContext`** — lives on `PixelBuffer` via `Arc<ColorContext>`. Narrow: `{icc, cicp}`. Travels with pixel data through pipeline stages. HDR fields (CLL/MDCV) are a known gap — see zenpixels#16.

Bridge: `SourceColor::to_color_context()` — uses `color_authority` to drop the non-authoritative field so `ColorContext::as_profile_source()` naturally returns the right source. **This is the one place authority semantics live**; downstream code doesn't re-implement them.

### The four naming dimensions

A color space can be named in four ways. All are expressible as `ColorProfileSource` variants:

```rust
pub enum ColorProfileSource<'a> {
    Icc(&'a [u8]),      // raw ICC profile bytes
    Cicp(Cicp),         // H.273 code points (primaries, transfer, matrix, range)
    Named(NamedProfile),  // sRGB, DisplayP3, BT.2020, AdobeRgb, ...
    PrimariesTransferPair { primaries: ColorPrimaries, transfer: TransferFunction },
}
```

**When to use which**:
- `Icc` — custom profiles, printer profiles, calibrated displays, JPEG/PNG embedded ICC
- `Cicp` — AVIF/HEIF nclx, PNG cICP chunk, JXL CICP metadata. Maps 1:1 to a small canonical set.
- `Named` — when you want `AdobeRgb` (v2-gamma variant, no CICP code) or similar "known but no-CICP" combos.
- `PrimariesTransferPair` — arbitrary combos like `Bt709 + Linear` that neither CICP nor Named can express.

## CICP vs ICC vs TRC — the three-way confusion

**These describe different things even when they overlap**:

- **ICC profile** — a bundle of color-pipeline data: primaries matrix, transfer characteristic (TRC), chromatic adaptation (`chad`), optional LUTs per intent, rendering descriptions, creator metadata, tag table. Self-contained; CMS can transform through it.
- **CICP** (ITU-T H.273 Coding-Independent Code Points) — a 4-tuple `(primaries, transfer, matrix, range)` of u8 codes. Names a standard color space, no pipeline data. Can't carry custom primaries or a non-standard TRC.
- **TRC** (Tone Reproduction Curve) — just the transfer function. An ICC carries one; a CICP references one by code (sRGB = 13, BT.709 = 1, PQ = 16, HLG = 18).

**Overlap and authority**: some formats carry both ICC and CICP for the same image. Who wins depends on the format spec:

| Format | Spec says | Source of truth |
|---|---|---|
| JPEG | ICC (APP2); no CICP in spec | ICC |
| PNG | cICP > iCCP when both present | CICP if cICP chunk, else ICC |
| WebP | ICC (ICCP); no CICP in spec | ICC |
| AVIF / MIAF | ICC (`colr`-Restricted) > CICP (nclx) | ICC if colr, else CICP |
| HEIF | nclx primary (ISO 23008-12) | CICP |
| JXL | CICP > ICC (ISO 21496-1) | CICP if CICP present, else ICC |
| TIFF | ICC (tag 34675); no CICP | ICC |
| GIF / BMP / PNM | no metadata, assumed sRGB | n/a |

This per-format logic lives in each codec's decoder, which sets `SourceColor::color_authority` accordingly. Downstream code reads authority, not format type.

### Why not just parse the ICC and ignore CICP?

- **Precision**: ICC TRC may be a `curv` LUT approximating the true sRGB function with ~0.08% error. CICP tells the CMS the exact mathematical form.
- **Speed**: closed-form sRGB EOTF is ~5× faster than LUT interpolation.
- **Trust**: container-level CICP is more authoritative for container formats (JXL, AVIF) that emit ICC for backwards compat.
- **Ambiguity**: some encoders embed a generic sRGB ICC alongside a PQ CICP. The ICC is wrong (compatibility placeholder). Preferring ICC produces wrong color.

`ColorPriority` enum in zenpixels-convert (`PreferIcc` / `PreferCicp`) threads this decision through the CMS layer independently of `color_authority`. Authority says "which field is canonical for THIS file"; priority says "given both, which do I trust for transform math".

## `ZenCmsLite` — the default fast path

Shipped as the automatic dispatch in `RowConverter::new_explicit_with_cms` (no user opt-in needed). Handles all named-profile conversions via fused matlut kernels. Falls through to user-supplied `PluggableCms` (e.g., moxcms) when profiles aren't recognizable.

**How it identifies profiles**:

1. Hash-based (`zenpixels::icc::identify_common`) — 163 RGB + 18 gray profiles, FNV-1a hash with metadata fields zeroed.
2. Falls back to CICP-in-ICC tag (ICC v4.4+ `cicp` tag) for profiles not in the hash table.
3. Returns `IccIdentification { primaries, transfer, valid_use }`.

`valid_use`:
- `MatrixTrcSubstitution` — safe for our matlut fast path (all intent bits set, bit-exact-at-u8 vs canonical).
- `MetadataOnly` — recognized but needs a full CMS (LUT-based perceptual, non-Bradford chad, Lab PCS).

Every live hash table entry is currently `Safe::AnyIntent` — `MetadataOnly` path exists but has no production data yet.

**Table generation** — `scripts/icc-gen` empirically validates each corpus profile: runs `moxcms(icc, intent=RelCol)` vs `moxcms(synth_canonical, intent=RelCol)` across a 64-step probe, grants `INTENT_COLORIMETRIC_SAFE` only when they agree within `COLORIMETRIC_VS_SYNTH_EPSILON_U16 = 256/65535 ≈ 0.39%` (one u8 code step). Optional lcms2 AND-gate.

`max_u16_err` numeric is still in the table but never queried at runtime (every caller uses `Tolerance::Intent` = 56 u16). User directed to keep the numeric in case future strict-tolerance callers need it.

## `PluggableCms` — the CMS override point

```rust
pub trait PluggableCms: Send + Sync {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        options: &ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>>;

    fn build_shared_source_transform(
        &self,
        ...
    ) -> Option<Result<Arc<dyn RowTransform>, whereat::At<CmsPluginError>>> { None }
}
```

**Three semantics on return**:
- `None` — plugin declines, dispatch chain tries next plugin (user plugin → ZenCmsLite default).
- `Some(Ok(t))` — plugin accepted, chain stops.
- `Some(Err(e))` — plugin tried and failed. Error propagates, chain does NOT continue (avoids silently substituting different color math from another backend).

**Error wrapping**: `whereat::At<CmsPluginError>` records the plugin's internal failure location via `whereat::at!` / `ResultAtExt::at()`; the receive site in `RowConverter` adds its own stamp via `whereat::at_crate!`. Two location points per failure, zero backtrace runtime cost.

### Two trait split: `RowTransform` vs `RowTransformMut`

- **`RowTransform`** (`&self`, `Send + Sync`) — stateless/shareable. Natural fit for moxcms `TransformExecutor` (already `&self`). Stored as `Arc<dyn RowTransform>`, clonable across threads.
- **`RowTransformMut`** (`&mut self`, `Send`) — owned/stateful. Holds scratch buffers without interior mutability. Stored as `Box<dyn RowTransformMut>`.

`LiteTransformMut` wraps `RowConverter` directly — no `Mutex`. The old `Mutex<RowConverter>` wrapper exists only for the deprecated `ColorManagement` API which needs `&self`.

## The `descriptor_for_decoded_pixels` migration

**Old API** (zencodec 0.1.16 and earlier, deprecated in 0.1.17):
```rust
fn descriptor_for_decoded_pixels(
    format: PixelFormat,
    source_color: &SourceColor,
    corrected_to: Option<&Cicp>,              // CICP targets only
    tolerance: IccMatchTolerance,             // placebo
) -> PixelDescriptor
```

**New API** (zencodec 0.1.17):
```rust
fn descriptor_for_decoded_pixels_v2(
    format: PixelFormat,
    source_color: &SourceColor,
    corrected_to: Option<&ColorProfileSource<'_>>,  // any target
) -> PixelDescriptor

fn resolve_color(
    source_color: &SourceColor,
    corrected_to: Option<&ColorProfileSource<'_>>,
) -> (ColorPrimaries, TransferFunction)
```

**Priority chain** (both functions):
```
corrected_to.is_some()
  └─→ use the target directly

else — SourceColor::to_color_context().as_profile_source()
  (honors color_authority, drops non-authoritative field)
  ├─→ Cicp → (cicp.primaries, cicp.transfer)
  ├─→ Icc  → identify_common → (primaries, transfer) OR (Unknown, Unknown)
  ├─→ Named → named.to_primaries_transfer()
  └─→ PrimariesTransferPair → as-is

else — no color metadata
  └─→ (Bt709, Srgb)  [assumed, legacy web default]
```

**Key migration patterns**:

1. **Simplest swap** — existing `corrected_cicp.as_ref()` callers:
   ```rust
   let corrected = corrected_cicp.map(zenpixels::ColorProfileSource::Cicp);
   let desc = descriptor_for_decoded_pixels_v2(format, &source_color, corrected.as_ref());
   ```

2. **Resolve once, build per-format** — better for multi-descriptor codecs (zenjpeg pattern, issue #87):
   ```rust
   let corrected = corrected_cicp.map(ColorProfileSource::Cicp);
   let (primaries, transfer) = resolve_color(&info.source_color, corrected.as_ref());
   // ... inspect, then build descriptors per-format ...
   ```

3. **Non-CICP targets** — widened API enables:
   ```rust
   // Adobe RGB v2-gamma (no CICP code for this variant):
   let target = ColorProfileSource::Named(NamedProfile::AdobeRgb);

   // Linear BT.709:
   let target = ColorProfileSource::PrimariesTransferPair {
       primaries: ColorPrimaries::Bt709,
       transfer: TransferFunction::Linear,
   };

   // Custom printer ICC:
   let target = ColorProfileSource::Icc(proof_profile_bytes);
   ```

## 0.3.0 queue — non-tolerated breaks

See `CHANGELOG.md` "Queued breaking changes (for 0.3.0)" section. Summary:

- **zenpixels**: `repr(u8)` removal from `ColorPrimaries`/`TransferFunction`; `ColorContext` → `#[non_exhaustive]`; remove deprecated `ColorContext::from_icc_and_cicp`; clean up commented-out enum variants. Add `ColorContext::hdr: Option<HdrProvenance>` (see zenpixels#16).
- **zenpixels-convert**: `ConvertError` → `#[non_exhaustive]` + `HdrTransferRequiresToneMapping` variant; remove deprecated `ColorManagement` / `finalize_for_output<C>` / `ZenCmsLite::extended` / `lut_transform_opts` / `cicp_transform_opts` / `ADOBE_RGB_COMPAT` / `PROPHOTO_RGB`.
- **zencodec**: remove deprecated ICC shims (`icc_extract_cicp`, `identify_well_known_icc`, `icc_profile_is_srgb`, `IccMatchTolerance`); remove `descriptor_for_decoded_pixels` (the old one); remove deprecated gainmap items; remove `SourceColor::has_hdr_transfer()`.

## Open design: HDR provenance (zenpixels#16)

Proposal in the issue: one `Option<HdrProvenance>` on `ColorContext`, with `HdrOrigin` enum that distinguishes:
- `Native` — decoded natively from HDR format (PQ/HLG), no gain map.
- `GainMap(GainMapProvenance)` — reconstructed from base SDR + gain map. Encode path is **mirror-split**: compute `log2(HDR.Y / base.Y)` to reproduce the gain map, emit alongside base SDR in gain map's original format.
- `ToneMapped { algorithm, target_peak_nits }` — pixels tone-mapped during decode. Informational; can't reconstruct HDR.

**Why luma-only matters**: ~95%+ of real-world HDR on the web is luma-only gain-map HDR (UltraHDR JPEG, Adobe Gain Map, Apple HDR). Mirror-split is bit-accurate for luma-only sources. `GainMapChannels::Luma` gets a specialized fast path.

Dependencies before 0.3.0:
- zencodec#11 decoder gaps — WebP/GIF bit_depth, HEIC authority, HEIC gain-map parse, UltraHDR JPEG APP11/APP2 detection.
- `ColorContext` → `#[non_exhaustive]` before adding the `hdr` field.

## Key files and their purpose

### zenpixels
- `src/color.rs` — `ColorProfileSource`, `NamedProfile`, `ColorAuthority`, `ColorContext`, `ColorOrigin`, `ColorProvenance`, `Cicp`.
- `src/descriptor.rs` — `PixelDescriptor`, `PixelFormat`, `ColorPrimaries`, `TransferFunction`.
- `src/icc/mod.rs` — hash-based profile identification. `identify_common()`, `extract_cicp()`, `Tolerance` enum (pub(crate)), `Safe::*` constants for intent masks.
- `src/icc/icc_table_rgb.inc`, `icc_table_gray.inc` — generated hash tables (199 RGB + 25 gray active, all `Safe::AnyIntent`).
- `src/policy.rs` — `ConvertOptions`, `AlphaPolicy`, `DepthPolicy`, `GrayExpand`, `LumaCoefficients`.

### zenpixels-convert
- `src/cms.rs` — `PluggableCms`, `RowTransform`, `RowTransformMut`, `CmsPluginError`, `ColorManagement` (deprecated), `RenderingIntent`, `ColorPriority`.
- `src/cms_lite.rs` — `ZenCmsLite` (default CMS), `LiteTransformMut` (new path), `LiteTransform` (std-only Mutex wrapper for deprecated `ColorManagement` API).
- `src/cms_moxcms.rs` — `MoxCms` backend, `transform_opts(priority, intent)`, `source_to_moxcms_profile` (private, used for future `build_source_transform`).
- `src/converter.rs` — `RowConverter`, `new_explicit_with_cms` with dispatch chain.
- `src/convert.rs` — `ConvertPlan`, `ConvertStep` enum, peephole fusion.
- `src/fast_gamut.rs` — fused matlut SIMD kernels for u8/u16/f32 RGB(A).
- `src/output.rs` — `finalize_for_output_with` (new), `finalize_for_output<C>` (deprecated), `OutputProfile`, `EncodeReady`.

### zencodec
- `src/info.rs` — `ImageInfo`, `SourceColor`, `GainMapPresence`, `Orientation`, `Resolution`, `EmbeddedMetadata`, `ContentLightLevel`, `MasteringDisplay`.
- `src/helpers/icc.rs` — `descriptor_for_decoded_pixels_v2`, `resolve_color`, `descriptor_for_decoded_pixels` (deprecated), `IccMatchTolerance` (deprecated placebo).

### scripts/icc-gen (internal)
- Walks corpus, empirically validates each profile against moxcms canonical synth. Optional lcms2 cross-check. Emits `.inc` files consumed by `zenpixels::icc`.

## `CLAUDE.md` policy recap

0.2.x tolerates narrow technical semver breaks when (a) no known external victims exist after auditing `~/work/zen/`, (b) a 0.3.0 bump would ripple through the whole zen dependency graph disproportionate to the change. Categories accepted in 0.2.N:
- Adding `#[non_exhaustive]` to structs/enums
- Adding fields to non-non-exhaustive structs (if all in-tree callers migrated)
- Adding auto-trait supertraits when all impls satisfy them
- Removing Cargo features that folded into default
- Losing auto-trait impls as mechanical consequence of new fields
- Dropping derive(Debug) when the type has no useful Debug repr

NOT tolerated in 0.2.N (require 0.3.0):
- Removing public items / renaming / signature changes / semantic behavior changes.

CHANGELOG entries for tolerated breaks go under `#### Changed (tolerated technical breaks)`.

## Pointers for the next contributor

1. **Adding HDR support end-to-end**: read zenpixels#16 first. Implementation order: land `HdrProvenance` types (additive, zenpixels), then `ColorContext::hdr` field (breaking, 0.3.0), then decoder population (per-codec PRs tracked in zencodec#11), then encode-side mirror-split in `finalize_for_output_with`.

2. **Fixing a specific codec's metadata gap**: check the table in zencodec#11. Each cell has a file:line reference to the offending code. PNG is the reference implementation — compare against it.

3. **Adding a new CMS backend**: implement `PluggableCms`, not `ColorManagement`. Use `whereat::at!` on errors. Ship a conformance test: verify `build_source_transform(src, dst, ..., &ConvertOptions::permissive())` produces matching output across your backend, moxcms, and ZenCmsLite for the named-profile set (see `cms_moxcms.rs` tests for patterns).

4. **Tolerated vs non-tolerated 0.2.x break decision**: run `cargo semver-checks`. If every failure falls into the categories in `CLAUDE.md`, audit external callers in `~/work/zen/` and ship. If any fails outside — stop, queue for 0.3.0.

5. **Adding a profile to the ICC hash table**: don't edit `.inc` files by hand. Run `scripts/icc-gen` against a corpus containing the profile; it does the empirical validation.
