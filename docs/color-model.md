# zenpixels color model — architecture reference

> **Provenance & staleness.** This document is the durable architecture
> content salvaged from the repo-root `CONTEXT-HANDOFF.md` (last substantively
> updated **2026-04-15**, post-0.2.8) when that banned handoff file was
> removed (2026-07-14). The *design* here — the two-type split, the CICP / ICC
> / TRC distinction, the per-format authority table, the CMS dispatch chain —
> is stable and still describes the crate. **Version-specific numbers**
> (table entry counts, exact deprecation lists, "shipped in 0.2.N") were
> current as of 0.2.8 and may have drifted: verify any such claim against
> current source before relying on it (per the repo's SEARCH-before-acting
> rule). Fix claims here in-place when you find drift — don't spawn a new
> handoff.

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
- **`ColorContext`** — lives on `PixelBuffer` via `Arc<ColorContext>`. Narrow: `{icc, cicp}` plus a `diffuse_white` anchor. Travels with pixel data through pipeline stages. Full HDR provenance (CLL/MDCV) is a known gap — see zenpixels#16.

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

1. Hash-based (`zenpixels::icc::identify_common`) — FNV-1a hash with metadata fields zeroed, over the committed RGB + gray profile tables.
2. Falls back to CICP-in-ICC tag (ICC v4.4+ `cicp` tag) for profiles not in the hash table.
3. Returns `IccIdentification { primaries, transfer, valid_use }`.

`valid_use`:
- `MatrixTrcSubstitution` — safe for our matlut fast path (all intent bits set, bit-exact-at-u8 vs canonical).
- `MetadataOnly` — recognized but needs a full CMS (LUT-based perceptual, non-Bradford chad, Lab PCS).

**Table generation** — `scripts/icc-gen` empirically validates each corpus profile: runs `moxcms(icc, intent=RelCol)` vs `moxcms(synth_canonical, intent=RelCol)` across a 64-step probe, grants `INTENT_COLORIMETRIC_SAFE` only when they agree within `COLORIMETRIC_VS_SYNTH_EPSILON_U16 = 256/65535 ≈ 0.39%` (one u8 code step). Optional lcms2 AND-gate. Don't edit the `.inc` files by hand — regenerate.

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

    fn build_shared_source_transform(&self, /* … */)
        -> Option<Result<Arc<dyn RowTransform>, whereat::At<CmsPluginError>>> { None }
}
```

**Three semantics on return**:
- `None` — plugin declines, dispatch chain tries next plugin (user plugin → ZenCmsLite default).
- `Some(Ok(t))` — plugin accepted, chain stops.
- `Some(Err(e))` — plugin tried and failed. Error propagates, chain does NOT continue (avoids silently substituting different color math from another backend).

**Error wrapping**: `whereat::At<CmsPluginError>` records the plugin's internal failure location via `whereat::at!` / `ResultAtExt::at()`; the receive site in `RowConverter` adds its own stamp. Two location points per failure, zero backtrace runtime cost.

### Two-trait split: `RowTransform` vs `RowTransformMut`

- **`RowTransform`** (`&self`, `Send + Sync`) — stateless/shareable. Natural fit for moxcms `TransformExecutor` (already `&self`). Stored as `Arc<dyn RowTransform>`, clonable across threads.
- **`RowTransformMut`** (`&mut self`, `Send`) — owned/stateful. Holds scratch buffers without interior mutability. Stored as `Box<dyn RowTransformMut>`.

> Note (2026-07): both `transform_row` methods currently return `()`, forcing the moxcms wrapper to `.expect()` on a transform failure. A `Result`-returning signature is queued for 0.3.0 (see `CHANGELOG.md` QUEUED BREAKING CHANGES).

## Related design work

- **zenpixels#16** — `HdrProvenance` for 0.3.0: one `Option<HdrProvenance>` on `ColorContext`, with an origin enum distinguishing `Native` (PQ/HLG), `GainMap` (reconstructed from base SDR + gain map; encode path is mirror-split — compute `log2(HDR.Y / base.Y)`), and `ToneMapped { algorithm, target_peak_nits }`. Luma-only matters: ~95%+ of real-world web HDR is luma-only gain-map (UltraHDR JPEG, Adobe Gain Map, Apple HDR), where mirror-split is bit-accurate.
- **zencodec#11** — cross-codec decoder audit: which `SourceColor`/`ImageInfo` fields each codec populates, with per-cell file:line references. PNG is the reference implementation.

## Key files

### zenpixels
- `src/color.rs` — `ColorProfileSource`, `NamedProfile`, `ColorAuthority`, `ColorContext`, `ColorOrigin`, `ColorProvenance`.
- `src/cicp.rs` — `Cicp`.
- `src/descriptor.rs` — `PixelDescriptor`, `PixelFormat`, `ColorPrimaries`, `TransferFunction`.
- `src/hdr.rs` — `DiffuseWhite`, `ContentLightLevel`, `MasteringDisplay`.
- `src/icc/mod.rs` — hash-based profile identification (`identify_common()`, `extract_cicp()`).
- `src/icc/icc_table_rgb.inc`, `icc_table_gray.inc` — generated hash tables.
- `src/policy.rs` — `ConvertOptions`, `AlphaPolicy`, `DepthPolicy`, `GrayExpand`, `LumaCoefficients`.

### zenpixels-convert
- `src/cms.rs` — `PluggableCms`, `RowTransform`, `RowTransformMut`, `CmsPluginError`, `RenderingIntent`, `ColorPriority`.
- `src/cms_lite.rs` — `ZenCmsLite` (default CMS) + `LiteTransformMut`.
- `src/cms_moxcms.rs` — `MoxCms` backend.
- `src/converter.rs` — `RowConverter`, `new_explicit_with_cms` dispatch chain.
- `src/convert.rs` — `ConvertPlan`, `ConvertStep`, peephole fusion, HDR tone-map plan.
- `src/estimate.rs` — `ResourceEstimate` / `ComputeEnvironment` / `ImageCharacteristics` / `SimdTier` (see `docs/ESTIMATE.md`).
- `src/fast_gamut.rs` — fused matlut SIMD kernels for u8/u16/f32 RGB(A).
- `src/hdr/` — `Bt2446A`, `SoftCompress`, `GamutBoundaryLut`, `measure::CllMeasure` (behind `hdr-experimental`).
- `src/output.rs` — `finalize_for_output_with`, `OutputProfile`, `EncodeReady`.

### zencodec (sibling crate)
- `src/info.rs` — `ImageInfo`, `SourceColor`, `GainMapPresence`, `ContentLightLevel`, `MasteringDisplay`.
- `src/helpers/icc.rs` — `descriptor_for_decoded_pixels_v2`, `resolve_color`.
