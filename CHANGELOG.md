# Changelog

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
