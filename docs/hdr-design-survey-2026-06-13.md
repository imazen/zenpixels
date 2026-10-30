# HDR representation: zen inventory, team intent, external prior art (2026-06-13)

Triggered by the design question "why is `encode_pq16` a free function, and what is
the superior long-term contract?" raised while reviewing the 0.2.14 release (which
first-ships `encode_pq16` + `compute_content_light_level`). Six parallel read-only
agents surveyed: (A) zen code, (B) the team's GH issues, (C) external OSS prior art.
This is the durable distillation; the full agent transcripts are in the session log.

Treat issue-body claims with skepticism — see "Skepticism flags" at the end.

---

## 1. What zen has today

### Color math (solid)
- `TransferFunction { Linear, Srgb, Bt709, Pq, Hlg, Gamma22, Unknown }`
  (`zenpixels/src/descriptor.rs`). PQ (ST 2084) + HLG (ARIB B67) EOTF/OETF live in
  `linear-srgb/src/tf/{pq,hlg}.rs` — exact + rational-poly + SIMD (x4/x8/x16),
  sub-u16 round-trip tested.
- **The conversion pipeline already does `Linear ↔ PQ` and `Linear ↔ HLG`** (u16 and
  f32) as first-class `ConvertStep`s (`convert.rs:563-571`,
  `convert_kernels.rs:1555-1688`). This is the load-bearing fact: `encode_pq16`
  duplicates machinery `ConvertPlan` already has.

### Absolute luminance (the gap)
- Only a **fixed per-TF constant**: `TransferFunction::reference_white_nits()` →
  PQ=203, else 1.0. Not settable per buffer.
- The only place a per-instance anchor exists is the **positional `f32`** argument
  `diffuse_white_nits` to the two `hdr.rs` free functions, plus the
  `REFERENCE_DIFFUSE_WHITE_NITS = 203.0` const.
- No luminance field on `PixelDescriptor`, `ColorContext`, or `Cicp`.
- Scene-vs-display is implicit in the TF choice (HLG/Linear=scene, PQ=display).

### Signaling + metadata carriers
- `Cicp { color_primaries, transfer_characteristics, matrix_coefficients, full_range }`
  (H.273), `ColorContext { icc, cicp }`, `SignalRange { Full, Narrow }` (no
  Narrow↔Full kernels — `NoPath`), ICC bundle (~135 profiles, lazy decode).
- **Two homes for HDR metadata**: `zenpixels::hdr` (`ContentLightLevel`,
  `MasteringDisplay`, `HdrMetadata`) AND `zencodec::info` (`Cicp`,
  `ContentLightLevel`, `MasteringDisplay`, `Metadata` aggregate). Codecs funnel
  HDR signaling into `zencodec::Metadata`. **`zencodec::gainmap` is a full
  ISO 21496-1 gain-map implementation** (per-channel log2/linear params,
  `BaseIsHdr`/`BaseIsSdr`, JPEG-APP2 / AVIF-tmap / JXL-jhgm wire forms).

### Codecs are HDR-capable; the conversion layer is the disconnect
- zenpng: cICP + cLLi + mDCV r/w. zenavif: nclx + clli + mdcv + gain-map decode.
  zenjpeg + ultrahdr: full Ultra HDR (gain maps, XMP, MPF). zenjxl: jhgm bundle.
  zenraw: **scene-referred linear f32 with NO nits anchor** (camera-dependent).
- The codecs read/write rich HDR into `zencodec::Metadata`, but `ConvertPlan`
  operates on `PixelDescriptor` (transfer/primaries/range only — no luminance,
  CLL, mastering, or gain-map). `hdr.rs`'s free functions bridge **neither** —
  they bypass both the metadata carrier and the pipeline.

### Planned-but-deferred (TODO(0.3.0) in code)
- `HdrPolicy` enum, HDR→SDR tone-map gate, `origin/target_has_hdr_transfer`
  checks, `OutputMetadata::hdr` wiring (`output.rs:107,180,344`, `error.rs:40-43`).

---

## 2. Team intent (GH issues — read skeptically)

- **`encode_pq16` + `compute_content_light_level` are a *decided* minimal design**
  (zenpixels#38 = #39 "Rung 2"), already shipped (commit `4399f93`). The posture:
  "output side belongs to zenpixels-convert; input adapters (P010/f16/1010102) do
  NOT (→ ultrahdr-rs)."
- **The grand pipeline-integrated contract is being argued DOWN, not built.**
  zenpixels#16 (`ColorContext.hdr: Option<HdrProvenance>`, gain-map-first, 0.3.0)
  is the most-detailed artifact — but the team's own same-day scope audit
  (zenpixels#39 comment 3) found its premise ("the pipeline loses CLL/MDCV")
  **factually wrong** (zenpipe carries both on `zencodec::Metadata`), noted CLL has
  "exactly one consumer," and demoted Rung 4 from "gated rung" to "hypothesis
  needing a named use case." YAGNI is being used to *block* it.
- **Absolute luminance was deliberately routed to zensim, not zenpixels**
  (zenpixels#34, dispersed): PU + reference/peak luminance live in the perceptual-
  metric crate (zensim#38: `reference_luminance_cd_m2`, default 203), explicitly
  off the "decade-frozen" zenpixels surface.
- Durable, decided posture: **CICP-first signaling, container-owns-emission,
  naive-tone-maps-only in core, quality tone-mapping quarantined in AGPL zentone,
  IQA math vendored into zensim, gain-maps rich in zencodec, aggressive YAGNI gate.**
- Genuinely open/contested: mirror-split vs layer-transit at encode (the newer
  opinion contradicts #16's spec); `Cicp::resolve_matrix` (#36, designed, unshipped,
  gated on a codec consumer); zencodec#24 Phases 1-4 (demand-gated).

---

## 3. External prior art (codec + color-management + Rust)

### The unanimous structural pattern
**No production system bundles absolute luminance into the color-space / transfer /
primaries type.** Verified across libavif, libheif, libjxl, FFmpeg, libplacebo,
Vulkan, OCIO, OIIO, ICC, CSS/color.js, Skia, Chrome gfx, Apple, Android (14 systems).
Luminance is always a *separate* optional component — a side struct, a nested HDR
block, or a standalone basic-info field. **This validates zen keeping luminance off
`ColorContext`/`PixelDescriptor`.**

### Three distinct absolute-luminance concepts (name them separately — Chrome models all 3)
- **(a) nominal diffuse / SDR white in nits** — the relative↔absolute anchor.
  ICC `lumi`, Chrome `NDWL`/`kDefaultSDRWhiteLevel`, CSS/color.js `Yw`, libheif
  **`ndwt`** (added 2026 — "nominal diffuse white luminance", 0 = ISO/TS 22028-5
  default), JXL **`intensity_target`** ("luminance at which (1,1,1) displays… for
  conversions between PQ and relative colorspaces"), libplacebo `PL_COLOR_SDR_WHITE`.
  **Default = 203 (BT.2408) — universal.**
- **(b) mastering display luminance** (ST 2086) + CLL/FALL (CTA-861.3) — content
  ceiling/floor. Everywhere: avif clli, Chrome `HdrMetadataSmpteSt2086` +
  `HdrMetadataCta861_3`, FFmpeg side-data, libplacebo `pl_hdr_metadata`.
- **(c) headroom ratio** — runtime display capability, 1.0 = SDR. Apple
  `currentEDRHeadroom`, Android `getHdrSdrRatio`, Chrome `HdrMetadataExtendedRange`.
  (Compositor/display-facing only.)

### Best models to study
- **libplacebo `pl_color_space { primaries, transfer, hdr: pl_hdr_metadata }`** +
  the **`pl_hdr_scaling { NORM(1.0=203), NITS, PQ, SQRT }`** enum + idempotent
  inference + zero-as-unknown. The reference design.
- **JXL `ToneMapping { intensity_target, min_nits, relative_to_max_display,
  linear_below }`** — the cleanest codestream nits model.
- **OpenEXR `whiteLuminance: Option<f32>`** ("nits of RGB (1,1,1)") — cleanest
  standard scalar anchor in Rust today.
- **colstodian `Color<Space, State>`, `State ∈ {Scene, Display}`** — the only
  type-level scene/display in Rust (but leaves luminance untyped).
- **moxcms** (which zen already depends on) — CICP + PQ/HLG + nits-parameterized
  transform `from_xyz_with_display_luminance(nits)`.

### Cross-cutting lessons
- **Type the units.** Libraries mix nits / PQ-codes (0-1) / log2-boost / ratios,
  often in one struct (libplacebo, FFmpeg DV). "Newtypes (`Nits`, `Pq`, `Norm`,
  `Log2Boost`) are not optional." → **a bare `diffuse_white_nits: f32` positional
  is the clearest anti-pattern.**
- **`Option`, not sentinel-zero** (libplacebo's `PL_COLOR_HDR_BLACK=1e-6` is direct
  evidence sentinel-zero is a smell).
- **PQ/HLG don't fit parametric transfer forms** → use a type-tagged enum (zen
  already does).
- **Scene-vs-display split**: the color-management/web world (OCIO `encoding`, ICC
  `ciis`, color.js `referred`) makes it an *explicit* flag and argues it is NOT
  derivable from the transfer (HLG=scene is the exception). The codec/compositor
  world (libplacebo *removed* its `light` enum, FFmpeg/avif/heif never had one)
  leaves it implicit in the TF. **zen's implicit approach matches the codec world it
  lives in; the explicit flag matters most on the IQA side, where zensim#38 handles
  it.**
- **Gain maps are a parallel additive layer** (ISO 21496-1 converging — avif 1.2,
  libultrahdr, JXL jhgm). Near green-field in Rust (only awxkee/gainforge +
  imazen/ultrahdr). zen's `zencodec` gain-map work is *ahead* of the ecosystem.
  The nits↔headroom bridge (`peak ≈ sdr_white × content_boost`) is lossy — type it.

---

## 4. Verdict on `encode_pq16` and the superior contract

### Why it's a free function
Not an accident — it's the team's decided "Rung 2" minimal output-side helper. It
exists separately from `ConvertPlan` only because the pipeline can't express its two
extras: a **custom diffuse-white anchor** (the pipeline hardcodes PQ's 203 via the
fixed per-TF constant) and **CLL as a side-output**. The grand pipeline integration
(luminance flowing through the type system) is the *deliberately deferred* Rung 4.

### What the prior art says the team got RIGHT
Keeping luminance off the color-signaling struct (unanimous external split);
CICP-first; 203 default; gain-maps as a parallel zencodec layer. The conservative
split is well-supported.

### What the prior art flags as fixable, in priority order
1. **Type the anchor.** `diffuse_white_nits: f32` positional → a `Nits`/`DiffuseWhite`
   newtype (or a small options struct). The single strongest, lowest-risk
   improvement; every external model that mixes luminance kinds types them.
2. **CLL measurement → a constructor**, `ContentLightLevel::measure(px, anchor)`,
   not a free function. Aligns it with the metadata type it returns (and with how
   every library models CLL as a small struct + an analysis).
3. **The anchor's long-term home is `zencodec::Metadata`**, not a positional arg —
   an explicit optional nominal-diffuse-white field mirroring libheif `ndwt` / JXL
   `intensity_target` / libplacebo `pl_hdr_metadata`. The team's own audit pointed
   here ("zenpipe carries it on zencodec::Metadata"). This is the natural seam, and
   it keeps luminance off the frozen zenpixels core (matching the universal split).
4. **`encode_pq16` should ultimately dissolve into the pipeline.** Since
   `ConvertPlan` already does linear→PQ, once the anchor lives in Metadata,
   `convert_buffer(linear_with_metadata, RGB16_BT2100_PQ)` + a separate `measure()`
   replaces it — avoiding the per-(transfer × depth) free-function sprawl
   (`encode_pq16`/`_hlg16`/`_pq10`/`_pq12`…) a descriptor-driven pipeline collapses
   to one call.

### Immediate 0.2.14 options
Both `hdr.rs` free functions first-ship in 0.2.14, freezing their shape forever
(zenpixels "every pub is forever"). They have one real consumer (the HDR corpus /
zenmetrics `--hdr` tooling), so the YAGNI "no consumer" worry is softer than first
stated — but the positional-`f32` shape is the freeze-forever liability.
1. **Ship Rung 2 as-is** (team's plan; documented provisional). Freezes the shape.
2. **Minimal realignment** (recommended): type the anchor + `ContentLightLevel::measure`
   constructor; keep `encode_pq16` as the interim output helper with the
   pipeline-integration path documented. Respects the decision, fixes the
   freeze-forever shape.
3. **Hold `encode_pq16`** (`pub(crate)`/experimental), ship only CLL-as-constructor;
   defer encode to the planned pipeline work. YAGNI-purest.

---

## 5. Skepticism flags (issue bodies are confidently wrong in places)
- **Phantom "PR #35 landed":** ≥6 comments across zensim/zenmetrics/zenpixels assert
  `zenpixels_convert::hdr_iqa::pu_encode`, `Cicp::resolve_matrix`, and
  `TransferFunction::peak_luminance_nits` shipped via PR #35. **PR #35 is CLOSED, not
  merged; none exist in zenpixels.** The IQA arc survived by *vendoring* into zensim
  (`zensim/src/pu21.rs`). zenpixels' actual HDR surface = `hdr.rs` + CICP consts +
  icc-db bundle. No `resolve_matrix`, no `hdr_iqa`.
- **Inverted H.273 claim:** zenpixels#34's body states "NEVER auto-derive
  BT.2020-NCL from CP=9" — backwards; CP=9+MC=9 is the canonical-correct pair. The
  correction lives in #36's comments.
- **Rung 4 / `HdrProvenance` is the most-documented, least-committed piece** — read
  it as speculative, explicitly not decided.

---

*Full agent transcripts (6 streams, ~700k tokens) in the 2026-06-13 session log.*
