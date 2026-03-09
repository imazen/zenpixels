# Changelog

## 0.1.1

### zenpixels

- Add `PixelSlice::as_strided_bytes()` — zero-copy access to raw backing bytes including inter-row stride padding, for passthrough to GPU uploads, codec writers, and other APIs that accept a buffer + stride.
- Add `PixelSliceMut::as_strided_bytes()` and `as_strided_bytes_mut()` — same for mutable slices.

### zenpixels-convert

- Make Oklab LMS/XYZ matrices public: `LMS_FROM_XYZ`, `XYZ_FROM_LMS`, `OKLAB_FROM_LMS_CBRT`, `LMS_CBRT_FROM_OKLAB`.
- Make scalar Oklab functions public: `rgb_to_oklab()`, `oklab_to_rgb()`, `fast_cbrt()`.

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
