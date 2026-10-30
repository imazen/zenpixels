# HDR anchor & convert API — user stories + what auto-converts

Status: design / proposal. Decides which of the HDR-convert items are worth a
**public** commitment (each `pub` in zenpixels is a decade-long contract — see
`CLAUDE.md` "Public API surface"), what shape they take, and — the load-bearing
question — **which conversions happen automatically vs. which the caller must
ask for**. Companion to the prior-art survey (`hdr-design-survey-2026-06-13.md`)
and the M×N epic (#45).

Personas referenced throughout:

- **Codec** — an encoder/decoder (`zenjpeg`, `zenavif`, `zenjxl`, `heic`).
  Populates `ColorContext` / `zencodec::Metadata` at decode; reads them at encode.
- **Pipeline** — `zenpipe` / imageflow. Holds `PixelBuffer`s with a
  `ColorContext` and moves them decode → process → encode.
- **Transcoder** — `zencodecs::transcode` (M×N). Gain-map JPEG → HDR JXL/AVIF.
- **HDR tool** — batch utility, e.g. `~/work/hdr-corpus-convert`.
- **App** — downstream consumer via `imageflow_abi` ("resize this, keep it HDR").

---

## 1. The principle: data-driven converts auto, intent-driven converts are explicit

A conversion **auto-applies** (the pipeline just does it, no policy argument) iff
it is **both**:

1. **lossless or precision-bounded** (reversible, or only loses sub-ULP), and
2. **fully determined by data the pixels already carry** (descriptor +
   `ColorContext`) — there is exactly one correct result.

Everything else is **explicit**: the caller must opt in through a policy, and the
**default policy is the safe / non-destructive one**. This is the existing
`ConvertOptions` philosophy ("no silent defaults for lossy operations") extended
to HDR.

The clean restatement, and the test to apply to any new step:

> **The absolute-luminance anchor is a property of the _data_** (it travels on
> `ColorContext`), so applying it is **auto**. **Tone-mapping is a property of the
> _output intent_** (target display, chosen operator), so it is **explicit**.
> Ask: "is this determined by what the pixels *are*, or by what the caller
> *wants*?" The first auto-converts; the second needs a policy.

---

## 2. Taxonomy — what auto-converts and what doesn't

| Operation | Tier | Why |
|---|---|---|
| Transfer encode/decode (sRGB / BT.709 / PQ / HLG / γ2.2 ↔ linear) | **AUTO** | deterministic, precision-bounded |
| **Absolute-luminance anchor** (relative-linear ↔ PQ-absolute via `diffuse_white`) | **AUTO** | the anchor is data on `ColorContext`; placing diffuse white is lossless |
| Gamut **matrix** between known primaries, in-gamut (709 ↔ P3 ↔ 2020) | **AUTO** | deterministic 3×3 in linear light |
| HLG OOTF for a **signaled** display peak | **AUTO** | peak is signaled data; OOTF is deterministic given it |
| Lossless layout/depth (u8→u16, add opaque alpha, premul ↔ straight) | **AUTO** | reversible |
| Anchor **value** when unsignaled → `DiffuseWhite::BT2408` (203) | **AUTO (default)** | cross-vendor convention; safe, overridable |
| — | — | — |
| Depth **reduction** (u16→u8), alpha **removal** | **EXPLICIT** — `DepthPolicy` / `AlphaPolicy` | lossy |
| RGB → Gray luma recipe | **EXPLICIT** — `LumaCoefficients` | recipe choice |
| **Out-of-gamut** handling (clip / perceptual / soft-clip) | **EXPLICIT** | opinionated; no single answer |
| **HDR → SDR tone-map** / peak-adaptation `(content_nits → display_nits)` | **EXPLICIT** — `HdrPolicy` + operator | lossy, opinionated |
| **Gain-map reconstruction** (`BaseOnly` vs `ReconstructHdr`) | **EXPLICIT** — `GainMapRender`, default `BaseOnly` | changes the image's nature; an SDR consumer must not get surprise HDR |
| Signal range narrow ↔ full | **EXPLICIT — refuse** (until kernels exist) | relabel-without-rescale = corrupt pixels |

Two consequences worth stating outright:

- **Encoding relative-linear → PQ at the anchor is auto and lossless** — it only
  *places* diffuse white at its nits. It is **not** tone-mapping. Tone-mapping
  only enters when the content's luminance exceeds what the target can show; that
  is a separate, explicit step.
- **`ReconstructHdr` is explicit and defaults off** for the same reason
  `AlphaPolicy::DiscardIfOpaque` is conservative: a caller asking for "the image"
  must not silently receive a re-natured (HDR) image or a surprise f32 buffer.

---

## 3. User stories for the proposed APIs & data structures

Each story states the persona, the need, the **auto-or-explicit** tier, and the
**surface recommendation** (the decade commitment).

### 3.1 `ColorContext.diffuse_white` — the anchor travels with the pixels *(shipped 0.2.14)*

> *As the **pipeline**, when I reconstruct linear HDR from a gain-map JPEG I tag
> the buffer's `ColorContext` with `diffuse_white = 203`. When I later convert
> that buffer to PQ for AVIF, the converter must place diffuse white at 203 nits
> **without my re-specifying it** — the anchor I already attached is the anchor.*

- **Tier: AUTO.** Surface: **shipped `pub`** (field + `with_diffuse_white`).
- Acceptance: a `PixelBuffer` carrying `diffuse_white = N` converts to PQ with
  `1.0 → N nits`; clones/strips preserve it; `None` ⇒ 203.

### 3.2 PixelBuffer-level anchor-aware convert — *the_ public HDR entry* (PROPOSED)

> *As the **pipeline / transcoder**, I hold a `PixelBuffer` with a `ColorContext`
> and want `convert(&buffer, target)` to honor the anchor automatically — the
> common case is "the pixels know their own anchor".*

- **Tier: AUTO** (reads `ColorContext.diffuse_white`).
- **Surface recommendation:** this — a buffer-level convert that auto-reads the
  anchor — is the **right public HDR-convert surface**, *not* the byte-level
  explicit variant. It is the only one with a broad consumer (every pipeline
  stage). **Not yet built**; it supersedes `quantize_to` (§3.4) when it lands.
- Open shape question: a free `fn` taking `&PixelBuffer`, vs. an inherent
  `PixelBuffer::convert_to`. Prefer the free fn in `zenpixels-convert` (keeps the
  core `PixelBuffer` free of convert deps).

### 3.3 `convert_buffer_with_anchor` (byte-level, explicit anchor) — *internal* (PROPOSED `pub`, recommend **NO**)

> *As **`quantize_to`** (and any byte-level caller that has raw HDR bytes but no
> `ColorContext`), I encode relative-linear → PQ at an anchor I pass explicitly.*

- **Tier: explicit** (no `ColorContext` to read → parameter).
- **Surface recommendation: keep `pub(crate)`.** It exists today only as
  `quantize_to`'s engine. No *external* consumer needs the byte-level explicit
  form — the sweep of `~/work` found none, and the auto path (§3.2) covers the
  buffer case. **Do not promote** until a concrete external caller appears; this
  is exactly the speculative-`pub` the API policy forbids. *(This answers the
  promotion you gated: the byte-level explicit fn is not the surface to commit —
  the buffer-level auto fn is.)*

### 3.4 `quantize_to` — the current PQ entry *(shipped, `pub`)*

> *As an **HDR tool**, I encode a reconstructed linear-HDR buffer to PQ16 and the
> default 203 anchor matches the reconstruction's `1.0 = SDR white`.*

- **Tier: AUTO** (reads `ColorContext`, defaults 203).
- **Surface: keep `pub`** while it has a live consumer (`hdr-corpus-convert`).
  **Deprecation path:** once §3.2 lands (general, any-target), `quantize_to`
  becomes a thin alias or is deprecated — it is PQ16-only and a special case of
  "convert a buffer honoring its anchor".

### 3.5 HLG anchor carrier — `(diffuse_white, display_peak_nits)` + OOTF (PROPOSED)

> *As a **codec** decoding HLG, I convert HLG → linear → sRGB honoring the system
> gamma for the signaled nominal peak `Lw` (default 1000 nits).*

- **Tier: AUTO** for the OOTF given a signaled/defaulted peak (the math is
  deterministic — `zentone::hlg::{hlg_system_gamma, hlg_ootf, hlg_inverse_ootf}`;
  `γ = 1.2 + 0.42·log10(Lw/1000)`). **Explicit** only if the caller wants a
  *non-default* render intent (scene- vs display-referred) or a non-signaled peak.
- **Why it's not the PQ scalar:** PQ's anchor is one multiply; HLG needs a peak
  **and** a power curve. So the HLG step carries **two** values, not the single
  `pq_anchor_scale` f32, and `zenpixels-convert` must depend on `zentone` (a leaf
  crate — acyclic). Bigger than PQ; deliberately out of the PQ-only PR (#48).
- **Surface:** internal first (thread `(white, Lw)` into the HLG `ConvertStep`s
  the same `pub(crate)` way PQ's scale threads); a public HLG entry only via §3.2
  once a consumer needs HLG output.

### 3.6 Tone-map / peak-adaptation policy — `HdrPolicy` (PROPOSED, the `output.rs` TODO)

> *As the **pipeline**, converting 4000-nit content to a 600-nit display, I must
> tone-map — and which operator (Reinhard / BT.2390 EETF / Hable …) and target
> peak are **my** decision, not a silent default.*

- **Tier: EXPLICIT.** This is the canonical "not an auto-convert": lossy and
  opinionated. Surface: an `HdrPolicy` enum on `ConvertOptions`/a new
  `ConvertOutputOptions`, **default = do not tone-map** (refuse, or pass through
  with a typed `HdrTransferRequiresToneMapping` error — gated on `ConvertError`
  becoming `#[non_exhaustive]`). The operators live in `zentone`; zenpixels-convert
  only *dispatches*.
- Acceptance: a plain `convert` from PQ-4000 to an SDR target **errors** rather
  than guessing an operator; tone-mapping happens only when `HdrPolicy` selects one.

### 3.7 `OutputMetadata` 0.3.0 reshape — drop `hdr`, add nothing *(decided)*

> *As a **codec** emitting AVIF, the atomic color-emit bundle gives me `icc` /
> `cicp`; the CLL / mastering / `diffuse_white` I read from the `Metadata`
> carrier, not from here.*

- **Tier: n/a** (carrier shape, not a conversion). Decision (grounded in the
  `~/work` sweep + `zencodec::ColorEmitPlan { cicp, icc }`): `OutputMetadata` stays
  **color-only**; HDR descriptors ride `zencodec::Metadata` (which already has
  them as siblings). See `output.rs` field docs + CHANGELOG queue.

### 3.8 `ContentLightLevel::measure` generic front-end (PROPOSED, deferred)

> *As an **HDR tool**, I want MaxCLL/MaxFALL of a decoded **PQ** image without
> hand-linearizing it first.*

- **Tier: AUTO** (the convert-crate wrapper linearizes, then calls the core
  primitive). **Deferred:** no consumer needs it yet (`hdr-corpus-convert`
  measures already-linear f32). Core `measure` stays f32-linear-only by design
  (cd/m² is defined only in linear light, and core has no linearizer). Add the
  wrapper to `zenpixels-convert::hdr` **when** S4/transcode needs CLL of a
  not-yet-linear buffer — co-landed with its consumer.

---

## 4. What we are deliberately **not** adding (YAGNI ledger)

- **Anchor on `ConvertOptions`.** Considered for full-pipeline threading;
  rejected for now — it drops `ConvertOptions: Eq/Hash` (f32) for a *hypothetical*
  future `RowConverter`-anchor consumer. The buffer-level §3.2 path covers real
  consumers without the break. Revisit only when a non-PixelBuffer caller needs it.
- **Public `convert_buffer_with_anchor` / `with_pq_anchor`.** `pub(crate)` until
  an external byte-level consumer exists (§3.3).
- **A tone-map operator zoo in zenpixels-convert.** The operators are `zentone`'s;
  zenpixels-convert dispatches via `HdrPolicy` (§3.6), it does not reimplement.
- **Auto HDR→SDR or auto gain-map reconstruction.** Both are explicit and default
  off (§2) — an SDR caller never gets surprise HDR, and an HDR→SDR convert never
  silently picks an operator.

---

## 5. Summary — the public surface this justifies

| Item | Tier | Surface (decade commitment) |
|---|---|---|
| `ColorContext.diffuse_white` | auto | **`pub`** (shipped) |
| `quantize_to` | auto | **`pub`** (shipped; deprecate after §3.2) |
| PixelBuffer-level anchor convert (§3.2) | auto | **`pub`** — *the* HDR entry, when built |
| `convert_buffer_with_anchor`, `with_pq_anchor` | explicit | **`pub(crate)`** — no external consumer |
| HLG `(white, Lw)` threading | auto | **`pub(crate)`** first; public via §3.2 |
| `HdrPolicy` tone-map | explicit | **`pub`** when the gate (`ConvertError` non_exhaustive) + a consumer land |
| `OutputMetadata` | n/a | color-only; HDR rides `zencodec::Metadata` |
| generic `measure` front-end | auto | **`pub`** in convert, co-landed with its consumer |

The one-line rule to carry forward: **auto-convert what the pixels already
determine; require an explicit, safe-defaulted policy for everything that loses
information or encodes the caller's taste.**
