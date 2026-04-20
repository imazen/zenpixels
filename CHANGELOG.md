# Changelog

## [Unreleased]

### zenpixels-convert — fixed

- **`AlphaPolicy::CompositeOnto` now produces correct pixels for
  premultiplied source** (fixes issue [#19][] [F]). The `matte_composite`
  kernel uses the straight-alpha over operator `fg*a + bg*(1-a)` after
  decoding to linear light. If the source descriptor declared
  `AlphaMode::Premultiplied`, feeding its bytes into the straight kernel
  multiplied by `a` a second time, producing
  `straight*a² + bg*(1-a)` — silently wrong by up to ~24 u8 codes at
  a ≈ 0.25. Fix: planner inserts `PremulToStraight` before
  `MatteComposite` when the source alpha mode is `Premultiplied`,
  recovering straight sRGB bytes (in our library's encoded-space premul
  convention) that the kernel handles correctly. No API change; kernel
  unchanged. Straight-source path unaffected. 6 regression tests in
  `tests/matte_composite_premul.rs`.

- **Planner no longer silently passes bytes through on TF changes** (fixes
  issue [#19][] [A] and [B]). Previously, several descriptor pairs emitted
  `[Identity]` or a naked depth-scale step labeled with the target TF but
  applying no EOTF/OETF — producing wrong pixels with no error. Affected:
  - Same-depth integer TF changes (U8 / U16): `Gamma22 → Srgb`, `Srgb →
    Bt709`, `Pq → Srgb`, every other cross-TF pair. Now routes through an
    F32 linear intermediate.
  - Integer↔F32 cross-TF combinations without a fused kernel: `U8 Gamma22
    → F32 Linear`, `U8 Bt709 → F32 Linear`, `U16 Gamma22 → F32 Linear`,
    etc. Now composes `NaiveU8ToF32 / U16ToF32` with the appropriate F32
    EOTF/OETF steps. Mid-gray error was up to 55× off for PQ inputs.
  - `U8 Bt709 → F32 Linear` and `F32 Linear → U8 Bt709` previously used
    the sRGB-specific fused kernel (`SrgbU8ToLinearF32` /
    `LinearF32ToSrgbU8`), producing ~17% linear-light error vs the correct
    BT.709 EOTF. Fused path now narrowed to sRGB; BT.709 composes through
    the correct step.
  - U16↔U8 and U8↔U16 cross-depth cross-TF combinations now compose
    through F32 linear when a fused kernel doesn't exist.
  Unknown TF on either side continues to pass bytes through as before (the
  explicit-intent API is tracked in #19 [C]/[D] for deprecate-and-add).
  Adds 54 regression tests in `tests/planner_silent_passthrough.rs`
  covering every TF × depth combination, HDR (PQ/HLG), extended-range
  out-of-gamut (`with_clip_out_of_gamut(false)`), and cross-primaries
  crossings.

### zenpixels-convert — added

- **First-class Gamma 2.2 (Adobe RGB 1998) transfer in the fast path.** New
  `ConvertStep::Gamma22F32ToLinearF32` / `LinearF32ToGamma22F32` variants plus
  `depth_steps` arms for Gamma22 ↔ {Linear, sRGB, BT.709, PQ, HLG} same-depth
  F32. The primaries-conversion injection now routes Gamma22 through the
  correct EOTF/OETF instead of falling through to the sRGB approximation.
  Lets AdobeRGB ↔ PQ / HLG / BT.2020 / Linear compose in the built-in planner
  without hitting the moxcms CMS fallback. SIMD via
  `linear_srgb::default::{gamma_to_linear,linear_to_gamma}_slice` with
  `ADOBE_GAMMA = 563/256 ≈ 2.19921875`.

[#19]: https://github.com/imazen/zenpixels/issues/19

## [0.2.9] - 2026-04-16

### zenpixels-convert — internal

- **`builtin_profiles` module (internal, `pub(crate)`)** — hand-coded XYB ICC
  inverse for jpegli's 720-byte profile. SIMD-accelerated via magetypes on
  x86_64. Internal only — no external consumers yet per YAGNI policy. (a5fdf9f, a3d924f)

### zenpixels-convert — performance

- **Fused `RgbToBgra` conversion step** — planner now emits a single
  `ConvertStep::RgbToBgra` for `(Rgb → Bgra)` u8 conversions instead of the
  two-pass `[AddAlpha, SwizzleBgraRgba]` sequence. Delegates to
  `garb::bytes::rgb_to_bgra` (8 px/iter AVX2 with R/B swap and `alpha=255`
  in one pass), halving destination-buffer write traffic for u8. u16/f32
  continue to use the existing two-step scalar path. (baa6214)

### zenpixels-convert — fixes

- **Raise `linear-srgb` minimum version to `0.6.10`** (was `0.6.7`).
  `srgb_to_linear_extended_slice` / `linear_to_srgb_extended_slice` were
  added in `linear-srgb` 0.6.10; downstream builds resolving to 0.6.7 (e.g.,
  zenpng fuzz targets) failed to compile. The workspace lockfile already
  resolved to 0.6.10 — this codifies the actual minimum. (9c53fe0)

### Docs & internal

- Fix 12 `cargo doc` warnings across `zenpixels`, `zenpixels-convert`, and
  `scripts/icc-gen`: fully qualified intra-doc links for cross-module
  references (`ColorModel`, `ColorProfileSource`, `ConvertPlan`,
  `RowConverter::new_explicit_with_cms`, `PluggableCms`), corrected
  `identify → identify_common` reference in the `icc` module preamble,
  dropped intra-doc links to private items (`Tolerance`, `ZenCmsLite`),
  fixed `Self::IccOnly` mislabel in `ColorPriority` docs, and escaped
  `<icc-cache-dir>` placeholders in `icc-gen` module docs. `cargo doc
  --no-deps` is now warning-free. Also bumps `[workspace.package]` version
  `0.2.2 → 0.2.8` to match the member crates. (b58212a)

## zenpixels 0.2.8 + zenpixels-convert 0.2.8 (2026-04-15)

Ships `PluggableCms`, `RowTransformMut`, fused matlut kernels,
`ConvertOptions::clip_out_of_gamut`, and `ZenCmsLite` as the default
CMS backend. Carries a set of **tolerated technical breaks** (see
[`CLAUDE.md`](CLAUDE.md) §0.2.x versioning policy) that
`cargo semver-checks` flags as major but which have no known external
impact. A 0.3.0 bump for these alone was judged too disruptive to the
`zen*` sibling dependency graph.

### zenpixels

#### Added

- **`ConvertOptions::clip_out_of_gamut: bool`** field (default `true`)
  plus `with_clip_out_of_gamut(bool)` builder. Set to `false` to emit
  sign-preserving extended-range f32 sRGB transfers — preserves
  negative and supernormal values for HDR / wide-gamut pipelines that
  defer tone or gamut mapping to a later stage.
- **`ConvertOptions::forbid_lossy()`** / **`::permissive()`** presets
  plus `with_alpha_policy`, `with_depth_policy`, `with_gray_expand`,
  `with_luma`, `with_clip_out_of_gamut` builders. Required since
  `ConvertOptions` became `#[non_exhaustive]`.

#### Changed (tolerated technical breaks)

- **`ConvertOptions` → `#[non_exhaustive]`**. External struct-literal
  construction breaks; in-tree callers migrated to builder pattern.
  Audited: no external struct-literal users across `~/work/zen/`.

### zenpixels-convert

#### Added

- **`PluggableCms` trait** with `build_source_transform` (owned
  `Box<dyn RowTransformMut>`) and `build_shared_source_transform`
  (shared `Arc<dyn RowTransform>`, default `None`). Dyn-compatible,
  accepts `ColorProfileSource` (ICC / CICP / Named / PrimariesTransferPair),
  carries `&ConvertOptions`.
- **`CmsPluginError`** — type-erased error wrapper for plugins, wraps
  any `core::error::Error + Send + Sync`. Plugin methods return
  `Option<Result<T, whereat::At<CmsPluginError>>>`: `None` = declined
  (chain tries next plugin), `Some(Ok)` = accepted, `Some(Err)` =
  tried-and-failed (error propagates immediately — the chain does not
  continue past a failed plugin to avoid silently substituting different
  color math). The `At<_>` envelope records the plugin's internal
  failure point via `whereat::at!` / `ResultAtExt::at()`; the receive
  site in `RowConverter::new_explicit_with_cms` adds a second stamp when
  wrapping into `ConvertError::CmsError`.
- **`RowTransformMut` trait** (`&mut self`, `Send`) for owned, stateful
  transforms that need scratch buffers without interior mutability.
  `RowTransform` (`&self`, `Send + Sync`) remains for stateless/shareable
  transforms (e.g., moxcms `TransformExecutor`).
- **`RowConverter::new_explicit_with_cms`** with ordered dispatch:
  user-supplied plugin first, then `ZenCmsLite` default (named-profile
  matlut fast path). Integer profile matches use fused SIMD kernels.
- **`finalize_for_output_with(...)`** — dyn-safe replacement for
  `finalize_for_output<C>`. Takes `Option<&dyn PluggableCms>` and routes
  through `RowConverter::new_explicit_with_cms` so the CMS dispatch
  chain is honored.
- **`SourceColor::to_color_context()`** (zencodec) — drops the
  non-authoritative color field based on `color_authority` so
  `ColorContext::as_profile_source()` naturally returns the
  authoritative source without a separate parameter.
- **Fused u8/u16 matlut SIMD kernels** on `RowConverter`'s default
  path. u8 sRGB: ~3× speedup vs prior; u16 sRGB: ~49× speedup.

#### Changed (tolerated technical breaks)

- **`RowTransform: Send + Sync`** (was `Send`-only). In-tree impls
  (`MoxRowTransform`, `LiteTransform`) already satisfy `Sync` because
  their inner state does. External impls that were intentionally
  `!Sync` would break; none are known.
- **`RowConverter` is no longer `Sync` / `UnwindSafe` /
  `RefUnwindSafe`**. Mechanical consequence of the new
  `external: Option<Box<dyn RowTransformMut: Send>>` field.
  `convert_row` takes `&mut self`, so cross-thread shared-reference
  use was never useful.
- **`RowConverter` no longer derives `Debug`**. Plan contents and
  external transform have no meaningful `Debug` representation.
- **Feature `zencms-lite` removed**. Functionality became unconditional
  — LUTs use `OnceBox` for no_std compatibility without a feature gate.

#### Deprecated

- **`ColorManagement` trait** — use `PluggableCms` instead.
  `ColorManagement` is non-dyn-safe (generic `type Error`), takes raw
  ICC byte pairs, and has no options channel. Existing impls
  (`MoxCms`, `ZenCmsLite`) are preserved and still work; they gain
  `#[allow(deprecated)]` on the impl block.
- **`finalize_for_output<C: ColorManagement>`** — use
  `finalize_for_output_with(..., cms: Option<&dyn PluggableCms>)`.

---

## Queued breaking changes (for 0.3.0)

Non-tolerated breaks (see [`CLAUDE.md`](CLAUDE.md) §0.2.x versioning
policy) — these require a proper 0.3.0 bump. This section accumulates
across 0.2.N patches and only clears when the 0.3.0 release cuts.

### zenpixels

- **`repr(u8)` removal** from `ColorPrimaries` and `TransferFunction`.
- **`ColorContext` → `#[non_exhaustive]`**. Construct via
  `from_icc()` / `from_cicp()` + builders. Direct struct literal
  construction is already discouraged (fields are `Option` with no
  authority signal); non-exhaustive makes it enforceable.
- **Remove `ColorContext::from_icc_and_cicp()`** (deprecated since 0.2.6).
  Use `from_icc()` or `from_cicp()` — codecs should populate only the
  authoritative field via `SourceColor::to_color_context()`.
- **Remove `ColorPrimaries` / `TransferFunction` commented-out variants**
  (deferred AppleRgb, Smpte170m, Bt470Bg, WideGamut, ColorMatch,
  EciRgbV2, DciP3, Gamma18, Gamma24, Gamma28) — clean up after the
  `repr(u8)` removal frees discriminant assignment.

### zenpixels-convert

- **`ConvertError` → `#[non_exhaustive]`** + new
  `HdrTransferRequiresToneMapping` variant. See imazen/zenpixels#10 for
  HDR provenance plan.
- **Remove `ColorManagement` trait** (deprecated in 0.2.8). Callers
  migrate to `PluggableCms`.
- **Remove `finalize_for_output<C: ColorManagement>`** (deprecated in
  0.2.8). Callers migrate to `finalize_for_output_with(..., cms:
  Option<&dyn PluggableCms>)`.
- **Remove `ZenCmsLite::extended` field and `::extended()` constructor**
  (deprecated; use `ConvertOptions::clip_out_of_gamut` via
  `RowConverter` instead).
- **Remove `lut_transform_opts()` and `cicp_transform_opts()`** in
  `cms_moxcms` (deprecated since 0.2.3; use `transform_opts()` with
  explicit `ColorPriority` + `RenderingIntent`).
- **Remove `ADOBE_RGB_COMPAT` and `PROPHOTO_RGB`** ICC profile constants
  in `icc_profiles` (deprecated since 0.2.4).

---

## zenpixels 0.2.7 (2026-04-14)

### Additions

- **`ColorPrimaries::AdobeRgb`** and **`TransferFunction::Gamma22`** enum
  variants for Adobe RGB (1998) identification and conversion.
- **`icc` module** — lightweight ICC profile identification (~100ns):
  - `identify_common(icc_bytes)` — hash-based lookup against 163 known RGB
    + 18 grayscale profiles from a corpus of 1,065 real-world ICC profiles.
  - `is_common_srgb(icc_bytes)` — convenience sRGB check.
  - `extract_cicp(data)` — read CICP tag from ICC v4.4+ profiles.
  - `IccIdentification` struct with `primaries`, `transfer`, `valid_use`.
  - `IdentificationUse` enum: `MetadataOnly` vs `MatrixTrcSubstitution` —
    tells callers whether matrix+TRC math is safe or a full CMS is needed.
- **`ColorPrimaries` methods**: `chromaticity()`, `white_point()`,
  `gamut_matrix_to()` (const-computed 3×3 Bradford-adapted gamut matrices),
  `WHITE_D65` constant.
- **`ColorProfileSource::PrimariesTransferPair`** variant +
  `from_primaries_transfer()`, `primaries_transfer()`, `resolve()`.
- **`NamedProfile`**: `from_primaries_transfer()`, `to_primaries_transfer()`.
- **`PixelDescriptor::color_profile_source()`**.
- **`ColorAuthority`** enum (`Icc` | `Cicp`) on `ColorContext` / `ColorOrigin`.
- **`NamedProfile::Bt2020Hlg`** + `TransferFunction::Hlg` CICP 18 round-trip.

### Behavior changes

- **`ColorContext::as_profile_source()`** now respects `color_authority` instead
  of hardcoding CICP preference.
- Enum variants trimmed to those with backing conversion math. Removed variants
  preserved as comments in `descriptor.rs` with chromaticities and rationale.

### Internal (not public API)

- `scripts/icc-gen` crate for regenerating ICC hash tables with empirical
  CMS validation via moxcms + lcms2 cross-check.
- `Safe::AnyIntent` / `Safe::IdOnly` named constants replace magic bitfields
  in `.inc` table files.
- `ProfileFeatures`, `inspect_profile()`, `CoalesceForUse`,
  `identify_common_for()` — kept `pub(crate)` for future use.
- Color registry with const-computed gamut matrices.

## zenpixels-convert 0.2.7 (2026-04-14)

### Additions

- **`icc_profiles::ADOBE_RGB`** — bundled CC0 Adobe RGB (1998) ICC profile
  (v2, pure gamma 2.19921875, matching ~85% of real-world profiles).
- **`ADOBE_RGB_V4`** deprecated alias → `ADOBE_RGB`.
- **`PROPHOTO_V4`** deprecated (returns empty bytes — ProPhoto not bundled
  due to TRC fragmentation).

### Behavior changes

- **`finalize_for_output()` respects `ColorAuthority`** on the `ColorOrigin`.
- **`SameAsOrigin` no longer invokes the CMS.** Only pixel format changes
  are applied. Previously built a wasteful same-profile-to-same-profile
  CMS transform.
- **`conversion_matrix()` returns `Option<GamutMatrix>`** (owned) instead of
  `Option<&'static GamutMatrix>`. `GamutMatrix` is `Copy` — callers drop
  the `&`.

### Internal (not public API)

- `ZenCmsLite` + `fast_gamut` — fused SIMD gamut conversion kernels
  (sRGB ↔ Display P3 ↔ BT.2020 ↔ Adobe RGB). Kept `pub(crate)` pending
  aarch64 SIMD and benchmarking against moxcms.
- Bundled Adobe RGB profile switched from v4 paraType-3 to v2 pure-gamma
  form (matches the spec and ecosystem majority).

## 0.2.3

### zenpixels-convert — additions

- **`RenderingIntent`** enum — `Perceptual`, `RelativeColorimetric` (default),
  `Saturation`, `AbsoluteColorimetric`. Backend-agnostic ICC rendering intent
  with thorough documentation of LUT fallback behavior, profile compatibility,
  and the moxcms/lcms2 perceptual intent mismatch.
- **`ColorPriority`** enum — `PreferIcc` (default), `PreferCicp`. Controls
  whether the CMS trusts ICC `curv`/`para` TRCs or CICP transfer characteristics.
  Documented: precision tradeoffs, advisory vs. authoritative semantics, and
  when each setting is correct.
- **`transform_opts(priority, intent)`** — single entry point for building
  moxcms `TransformOptions`. Replaces `lut_transform_opts()` and
  `cicp_transform_opts()` with explicit control over rendering intent.

### zenpixels-convert — breaking behavior change

- **Default rendering intent is now `RelativeColorimetric`**, not `Perceptual`.
  The previous default inherited moxcms's `Perceptual`, but moxcms's perceptual
  intent does not match lcms2 and may produce inaccurate results. Most display
  profiles only ship a relative colorimetric LUT, making the two intents
  identical in practice — but for profiles that do have perceptual tables, this
  is a visible change.

### zenpixels-convert — deprecations

- **`lut_transform_opts()`** — use `transform_opts(ColorPriority::PreferIcc, intent)`.
- **`cicp_transform_opts()`** — use `transform_opts(ColorPriority::PreferCicp, intent)`.

## 0.2.2

### zenpixels-convert — additions

- **`lut_transform_opts()`** (public) — canonical moxcms `TransformOptions` for
  standard ICC LUT transforms: `allow_use_cicp_transfer: false`,
  `BarycentricWeightScale::High`, `InterpolationMethod::Tetrahedral`.
- **`cicp_transform_opts()`** — same quality settings as `lut_transform_opts` but
  `allow_use_cicp_transfer: true` for CICP-native source formats (JXL, HEIF).

### zenpixels-convert — improvements

- **`InterpolationMethod::Tetrahedral`** added to the internal `lut_transform_opts`
  used by `MoxCms::build_transform_for_format`. Improves accuracy of 3D CLUT
  transforms over the previous trilinear default.
- **`BarycentricWeightScale::High`** was already set; now documented in the public
  function with the rationale (max LUT interpolation error ≤2 vs ≤14, no perf cost).

## 0.2.1

### zenpixels — additions

- **`serde` feature** — optional `Serialize`/`Deserialize` derives on all core
  types: `PixelDescriptor`, `PixelFormat`, `ChannelType`, `ChannelLayout`,
  `AlphaMode`, `TransferFunction`, `ColorPrimaries`, `SignalRange`, `ColorModel`,
  `ByteOrder`, `Cicp`, `ContentLightLevel`, and `MasteringDisplay`. Off by default.

### zenpixels-convert — additions

- **Gamut matrix in `RowConverter`** — primaries conversion (e.g., BT.709 ↔
  Display P3 ↔ BT.2020) is now automatic. `RowConverter` injects a 3×3 matrix
  step in linear f32 space when source and destination primaries differ.
  Previously callers had to apply gamut matrices manually.
- **Embedded ICC profiles** — CC0-licensed ICC profiles for Display P3, AdobeRGB,
  Rec2020, and ProPhoto are now bundled. `icc_profile_for_primaries()` returns
  the appropriate profile bytes for a given `ColorPrimaries` value.
- **`serde` feature** — forwards to `zenpixels/serde`.

### Bug fixes

- **Display P3 TRC correction** — `identify_by_colorants` now correctly maps
  Display P3 to sRGB transfer characteristic (code 13) instead of BT.709 (code 1).
- **`allow_use_cicp_transfer` disabled** in the moxcms CMS path. CICP transfer
  function override is for applications, not CMMs. The zen conversion pipeline
  handles transfer functions explicitly via `RowConverter`, so the CMS should
  only apply the ICC profile's gamut mapping. Matches the moxcms v2 path fix.
- **Linear-space matte compositing** — `matte_composite` now blends in linear
  light instead of gamma-encoded space, fixing visible darkening artifacts at
  semi-transparent edges.

## 0.2.0

This is a **breaking release** — see "Breaking changes" below.

### zenpixels — breaking changes

- **Removed `buffer` feature.** Its functionality (`rgb` + `imgref`) is now always
  available via the `imgref` feature, which implies `rgb`.
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, BufferError>` now return `Result<T, At<BufferError>>`.
  Call `.error()` to inspect, `.into_inner()` to unwrap, or use `whereat::ResultAtExt`
  for ergonomic chaining. Affected: `PixelSlice::new`, `PixelSliceMut::new`,
  `PixelBuffer::try_new`, `from_vec`, `from_pixels`, `reinterpret`, and all
  `_typed` constructors.

### zenpixels — additions

- **`Orientation` enum** — canonical EXIF orientation type (D4 dihedral group).
  `#[repr(u8)]` with EXIF values 1-8. Includes D4 group algebra (`compose`,
  `inverse`, `then`), geometry helpers (`output_dimensions`, `forward_map`,
  `swaps_axes`, `is_row_local`), and EXIF conversion (`from_exif`, `to_exif`).
  All core methods are `const`. Re-exported at crate root.
- `PixelSlice::as_strided_bytes()` — zero-copy access to raw backing bytes including
  inter-row stride padding. For GPU uploads, codec writers, and other buffer+stride APIs.
- `PixelSliceMut::as_strided_bytes()` / `as_strided_bytes_mut()` — return the full
  backing `&[u8]` / `&mut [u8]` including any trailing bytes beyond the image extent.
- `PixelSliceMut::as_pixel_slice()` and `From<PixelSliceMut> for PixelSlice` —
  zero-copy immutable borrow/move from a mutable slice.
- `ContentLightLevel` and `MasteringDisplay` moved here from `zenpixels-convert::hdr`.
  Re-exported at crate root.
- `MasteringDisplay::HDR10_REFERENCE` and `DISPLAY_P3_1000` — predefined constants
  for common mastering display configurations.
- `Cicp::from_descriptor()`, `Cicp::to_descriptor()` — round-trip between CICP codes
  and `PixelDescriptor`.
- `NamedProfile::from_cicp()` — identify named profiles from CICP codes.
- `TransferFunction::to_cicp()`, `ColorPrimaries::to_cicp()` — convert enum variants
  to CICP code points.
- `ConvertOptions` convenience constructors: `forbid_lossy()`, `permissive()`,
  plus `with_alpha_policy()`, `with_depth_policy()`, `with_gray_expand()`,
  `with_luma()` builders.
- `#[track_caller]` on all fallible constructors for better error diagnostics.
- `whereat::At`, `ResultAtExt`, and `at` re-exported at crate root.

### zenpixels-convert — breaking changes

- **`RowConverter::convert_row()` and `convert_rows()` changed from `&self` to
  `&mut self`**. This enables internal scratch buffer reuse (no per-row heap allocation).
  Callers must use `let mut converter`.
- **`RowConverter` no longer auto-derives `Clone`.** A manual `Clone` impl creates
  fresh (empty) scratch buffers. Behavior is unchanged but the clone is not a
  bitwise copy.
- **`RowTransform` trait now requires `Send`.** Non-`Send` implementors will no longer
  compile.
- **`PixelBufferConvertExt` trait split.** `to_rgb8()`, `to_rgba8()`, `to_gray8()`,
  `to_bgra8()` moved to new `PixelBufferConvertTypedExt` trait (requires `rgb` feature).
  `linearize()` and `delinearize()` added to `PixelBufferConvertExt` (always available).
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, ConvertError>` now return `Result<T, At<ConvertError>>`.
  Affected: `RowConverter::new`, `new_explicit`, `convert_rows`,
  `adapt_for_encode`, `adapt_for_encode_explicit`, `convert_buffer`,
  `PixelBufferConvertExt` methods.
- **`codec` feature renamed to `pipeline`.** `CodecFormats`, `FormatEntry`,
  `ConversionPath`, `PathEntry`, etc. moved from root to `pipeline::` submodule.
  Import paths changed from `zenpixels_convert::registry::*` to
  `zenpixels_convert::pipeline::*`.
- **`Cicp::SRGB.matrix_coefficients` changed from `6` to `0`** (correct per ITU-T H.273
  — sRGB is an RGB color space, not YCbCr, so Identity matrix is correct).

### zenpixels-convert — additions

- **Streaming perf: zero per-row allocation.** `ConvertScratch` ping-pong buffers
  replace heap allocation in multi-step row conversions.
- `ConvertPlan::compose()` and `RowConverter::compose()` — chain two converters.
  Peephole optimization cancels inverse pairs (e.g., premultiply + unpremultiply).
- `RowConverter::new_explicit()` — explicit conversion plan with `ConvertOptions`
  policy validation before creating the plan.
- `MatteComposite` conversion step — flatten alpha against a matte color
  (used by `AlphaPolicy::CompositeOnto`).
- `linearize()` / `delinearize()` on `PixelBufferConvertExt` — buffer-level
  transfer function conversion.
- F32-to-F32 transfer function kernels: `SrgbF32ToLinearF32`, `LinearF32ToSrgbF32`,
  `Bt709F32ToLinearF32`, `LinearF32ToBt709F32`. Previously only u8/u16↔f32 TF
  conversions existed; these enable f32→f32 re-encoding without a depth roundtrip.
  PQ and HLG f32↔f32 kernels also added. All SIMD-dispatched via `linear-srgb`.
- **moxcms CMS backend** (behind `cms-moxcms` feature). `MoxCms` implements
  `ColorManagement` for ICC profile transforms via the `moxcms` crate. Supports
  u8, u16, and f32 transforms. F16 input routes to the f32 path.
- `garb` 0.2 for SIMD-accelerated pixel swizzle, layout conversions, depth scaling,
  and BT.709 luma.
- Public Oklab constants and functions: `LMS_FROM_XYZ`, `XYZ_FROM_LMS`,
  `OKLAB_FROM_LMS_CBRT`, `LMS_CBRT_FROM_OKLAB`, `rgb_to_oklab()`, `oklab_to_rgb()`,
  `fast_cbrt()`.

### Bug fixes

- F16 data no longer incorrectly routed to u16 CMS transform path. F16 now uses
  the f32 transform (IEEE 754 half-floats are not integer-encoded).
- Fixed ICC profile identification to use D50-adapted PCS colorants.

## 0.1.0

Initial release.

### zenpixels (interchange types)

**Pixel format description:**
- `PixelFormat` flat enum: `Rgb8`, `Rgba8`, `Rgb16`, `Rgba16`, `RgbF32`, `RgbaF32`, `Gray8`, `Gray16`, `GrayF32`, `GrayA8`, `GrayA16`, `GrayAF32`, `Bgra8`, `Rgbx8`, `Bgrx8`, `OklabF32`, `OklabaF32`
- `PixelDescriptor` with transfer function, alpha mode, color primaries, signal range
- 40+ predefined descriptor constants (`RGB8_SRGB`, `RGBAF32_LINEAR`, `BGRA8_SRGB`, etc.)
- `ChannelType`, `ChannelLayout`, `TransferFunction`, `ColorPrimaries`, `AlphaMode`, `SignalRange` enums
- `Cicp` struct with ITU-T H.273 code points and human-readable name lookups

**Pixel buffers:**
- `PixelBuffer<P>` (owned), `PixelSlice<'a, P>` (borrowed), `PixelSliceMut<'a, P>` (mutable borrowed)
- Phantom-typed `P: Pixel` for compile-time format safety, zero-cost `.erase()` / `.try_typed::<Q>()`
- SIMD-aligned allocation via `try_new_simd_aligned()`
- Row access: `row()`, `row_mut()`, `row_with_stride()`
- Contiguous access: `as_contiguous_bytes()`, `contiguous_bytes()` (Cow)
- Zero-copy views: `sub_rows()`, `crop_view()`, `crop_copy()`
- `Rgbx` and `Bgrx` 32-bit SIMD-friendly padded pixel types
- `GrayAlpha8`, `GrayAlpha16`, `GrayAlphaF32` pixel types

**Color metadata:**
- `ColorContext` (ICC + CICP, `Arc`-shared)
- `ColorOrigin`, `ColorProvenance`, `ColorProfileSource`, `NamedProfile`

**Conversion policies:**
- `ConvertOptions` with `AlphaPolicy`, `DepthPolicy`, `LumaCoefficients`, `GrayExpand`

**Multi-plane images** (behind `planar` feature):
- `PlaneLayout`, `PlaneDescriptor`, `PlaneSemantic`, `Subsampling`, `YuvMatrix`
- `MultiPlaneImage` container with per-plane `PixelBuffer`s
- YCbCr 4:2:0/4:2:2/4:4:4, Oklab planes, gain maps, separate alpha planes

**Interop** (behind feature gates):
- `rgb` feature: `Pixel` impls for `rgb` crate types
- `imgref` feature: `From<ImgRef>` / `From<ImgVec>` conversions, `as_imgref()` / `try_as_imgref::<P>()`

### zenpixels-convert (pixel math)

**Row conversion:**
- `RowConverter` with pre-computed conversion plan, no per-row allocation
- Three-tier dispatch: direct SIMD kernels, composed multi-step plans, hub path through linear sRGB f32
- Transfer function kernels: sRGB, BT.709, PQ (HDR10), HLG
- Depth scaling (u8/u16/f32), alpha mode changes, byte swizzle

**Format negotiation:**
- Two-axis cost model (effort vs. loss) with `ConvertIntent` weighting
- `best_match()`, `best_match_with()`, `negotiate()` entry points
- `Provenance` tracking for lossless round-trip detection
- `ideal_format()` for operation-aware format selection

**Gamut mapping:**
- 3x3 row-major f32 gamut matrices between BT.709, Display P3, BT.2020
- `conversion_matrix()`, `apply_matrix_row_f32()`, `apply_matrix_row_rgba_f32()`

**Oklab:**
- Primaries-aware `rgb_to_lms_matrix()` / `lms_to_rgb_matrix()`

**HDR:**
- Reinhard and exposure tone mapping
- `ContentLightLevel`, `MasteringDisplay`, `HdrMetadata`

**Codec integration:**
- `CodecFormats` registry with `FormatEntry` (effective bits, overshoot flag)
- `finalize_for_output()` for atomic pixel + metadata assembly
- `adapt_for_encode_explicit()` for policy-validated conversion
- `ConvertError` with specific variants (`NoMatch`, `NoPath`, `AlphaNotOpaque`, `DepthReductionForbidden`, `CmsError`)
- `ColorManagement` and `RowTransform` traits for external CMS backends

**Operation format requirements:**
- `OpCategory` and `OpRequirement` for operation-specific format suitability
- Conversion path analysis: `ConversionPath`, `LossBucket`, `generate_path_matrix()`
