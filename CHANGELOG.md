# Changelog

## 0.2.0

This is a **breaking release** — see "Breaking changes" below.

### zenpixels — breaking changes

- **Removed `buffer` feature.** Its functionality (`rgb` + `imgref`) is now always
  available via the `imgref` feature, which implies `rgb`.
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, BufferError>` now return `Result<T, At<BufferError>>`.
  Call `.into_inner()` to unwrap, or use `whereat::ResultAtExt` for ergonomic chaining.

### zenpixels — additions

- `PixelSlice::as_strided_bytes()` — zero-copy access to raw backing bytes including
  inter-row stride padding. For GPU uploads, codec writers, and other buffer+stride APIs.
- `PixelSliceMut::as_strided_bytes()` / `as_strided_bytes_mut()` — mutable equivalents.
  Now clips to actual data extent (matching `PixelSlice` behavior).
- `PixelSliceMut::as_pixel_slice()` and `From<PixelSliceMut> for PixelSlice` —
  zero-copy immutable borrow from a mutable slice.
- `ContentLightLevel` and `MasteringDisplay` moved here from `zenpixels-convert::hdr`.
  Re-exported at crate root.
- `Cicp::from_descriptor()`, `Cicp::to_descriptor()` — round-trip between CICP codes
  and `PixelDescriptor`.
- `NamedProfile::from_cicp()` — identify named profiles from CICP codes.
- `TransferFunction::to_cicp()` — convert transfer function enum to CICP code.
- `ConvertOptions` convenience constructors: `forbid_lossy()`, `permissive()`,
  plus `with_alpha_policy()`, `with_depth_policy()`, `with_luma()` builders.

### zenpixels-convert — breaking changes

- **`RowConverter::convert_row()` and `convert_rows()` changed from `&self` to
  `&mut self`**. This enables internal scratch buffer reuse (no per-row heap allocation).
  Callers must use `let mut converter`.
- **`RowTransform` trait now requires `Send`.** Non-`Send` implementors will no longer
  compile.
- **`PixelBufferConvertExt` trait split.** `to_rgb8()`, `to_rgba8()`, `to_gray8()`,
  `to_bgra8()` moved to new `PixelBufferConvertTypedExt` trait.
  `linearize()` and `delinearize()` added to `PixelBufferConvertExt`.
- **Error types now wrapped in `At<>`** (from `whereat` crate). All public functions
  returning `Result<T, ConvertError>` now return `Result<T, At<ConvertError>>`.
- **Pipeline modules gated behind `pipeline` feature.** `CodecFormats`, `FormatEntry`,
  `ConversionPath`, `PathEntry`, etc. moved from root to `pipeline::` submodule.
- **`Cicp::SRGB.matrix_coefficients` changed from `6` to `0`** (correct per ITU-T H.273).

### zenpixels-convert — additions

- **Streaming perf: zero per-row allocation.** `ConvertScratch` ping-pong buffers
  replace heap allocation in multi-step row conversions.
- `ConvertPlan::compose()` and `RowConverter::compose()` — chain two converters.
  Peephole optimization cancels inverse pairs (e.g., premultiply + unpremultiply).
- `RowConverter::new_explicit()` — explicit conversion plan without format negotiation.
- `MatteComposite` conversion step — flatten alpha against a matte color.
- `linearize()` / `delinearize()` on `PixelBufferConvertExt` — buffer-level TF conversion.
- F32 transfer function kernels for sRGB, BT.709, PQ, HLG (SIMD-dispatched via
  `linear-srgb`).
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
