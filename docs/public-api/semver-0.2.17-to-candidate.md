# 0.2.17 bridge to 0.3 candidate: semver audit

Baseline: `e96e4e9c65020a0c7fac4937819fc044886ae4df` (release/0.2, 0.2.17).
Candidate source: `a6874185` plus documentation/HLG backend changes with no new
public signatures. Checker 0.51.0, Rust 1.99, default features, forced patch
comparison. The checker reports 15 major lint families for core and 16 for the
converter; exit 100 is expected for the authorized breaking 0.3 release.
This is not a semver-clean patch claim.

## Classification and migration

The symbol findings below are approved breaking-release migrations under the
finalization contract and the current API cleanup. Enum repr/cast removal and
native named-variant retirement move to explicit mappings and Generic/from_parts.
ByteOrder moves to layout; constructors move to fallible forms. Root HDR aliases
move to hdr paths; computation and estimation move to converter functionality or
are removed. Free operations move to the receivers in the current ablation map.
Options/cost fields use builders/accessors; matte fields now carry their color
space and white anchor. Typed storage handoff uses into_parts or canonical byte
extraction. Concrete extension traits are sealed/Sized; CMS plugins remain open.
Compatibility constants/empty profile placeholders are not replacement profiles;
use SynthesizedIcc and handle refusal. CodePlane construction uses new/component.

Non-exhaustive/hidden-field findings are intentional API encapsulation. Changed
implicit error-enum discriminants are not wire codes; match variants instead.
Required methods on sealed concrete extensions are part of that sealing change.
No finding below is dismissed as a tool limitation or silently accepted as a
0.2 patch. Historical release-plan language retaining into_vec is superseded by
the later finalization contract's retired handoffs and current storage API.

The installed 0.49 checker cannot parse rustdoc v61. The initial all-feature
heuristic also loses workspace patches and selects the unpublished precise-portable
linear-srgb feature against the registry. These are tooling failures, not passing
checks. Explicit default-feature auditing succeeded with 0.51. Feature-enabled
compilation and strict Clippy run against the patched workspace separately; that
is not a claim of an all-feature semver diff. The precise-portable dependency
patch remains a release packaging gate.

## Recorded findings (deduplicated)

### enum_discriminants_undefined_non_unit_variant

- `enum BufferError`

### enum_marked_non_exhaustive

- `enum ColorAuthority`

### enum_missing

- `enum descriptor::ByteOrder`
- `enum ByteOrder`

### enum_no_repr_variant_discriminant_changed

- `variant CodePlaneError::OutsideActivePlane 6 -> 2`

### enum_repr_int_removed

- `enum ChannelType`
- `enum PixelFormat`
- `enum ChannelLayout`
- `enum AlphaMode`
- `enum ColorPrimaries`
- `enum ColorModel`
- `enum SignalRange`
- `enum TransferFunction`

### enum_struct_variant_field_added

- `field background of variant AlphaPolicy::CompositeOnto`
- `field reference_white of variant AlphaPolicy::CompositeOnto`

### enum_struct_variant_field_missing

- `field r of variant AlphaPolicy::CompositeOnto`
- `field g of variant AlphaPolicy::CompositeOnto`
- `field b of variant AlphaPolicy::CompositeOnto`

### enum_variant_missing

- `variant PixelFormat::RgbI8`
- `variant PixelFormat::RgbaI8`
- `variant PixelFormat::GrayI8`
- `variant PixelFormat::GrayAI8`
- `variant PixelFormat::RgbI16`
- `variant PixelFormat::RgbaI16`
- `variant PixelFormat::GrayI16`
- `variant PixelFormat::GrayAI16`
- `variant PixelFormat::RgbU32`
- `variant PixelFormat::RgbaU32`
- `variant PixelFormat::GrayU32`
- `variant PixelFormat::GrayAU32`
- `variant PixelFormat::RgbI32`
- `variant PixelFormat::RgbaI32`
- `variant PixelFormat::GrayI32`
- `variant PixelFormat::GrayAI32`
- `variant PixelFormat::RgbU64`
- `variant PixelFormat::RgbaU64`
- `variant PixelFormat::GrayU64`
- `variant PixelFormat::GrayAU64`
- `variant PixelFormat::RgbI64`
- `variant PixelFormat::RgbaI64`
- `variant PixelFormat::GrayI64`
- `variant PixelFormat::GrayAI64`
- `variant PixelFormat::RgbF64`
- `variant PixelFormat::RgbaF64`
- `variant PixelFormat::GrayF64`
- `variant PixelFormat::GrayAF64`
- `variant CodePlaneError::MisalignedStride`
- `variant CodePlaneError::ShortStride`
- `variant CodePlaneError::ShortBuffer`
- `variant CodePlaneError::GeometryOverflow`
- `variant CodePlaneError::ZeroStep`

### function_missing

- `function icc::identify_common`
- `function icc::is_common_srgb`
- `function icc::profile_color_space`
- `function icc::extract_cicp`

### inherent_associated_pub_const_missing

- `ContentLightLevel::DEFAULT_PERCENTILE`

### inherent_method_missing

- `ContentLightLevel::measure`
- `PixelBuffer::new_simd_aligned`
- `PixelBuffer::try_new_simd_aligned`
- `PixelBuffer::into_vec`
- `PixelDescriptor::byte_order`
- `Cicp::to_descriptor`
- `ConvertOptions::forbid_lossy`
- `ColorContext::from_icc_and_cicp`
- `CodePlane::from_u8`
- `CodePlane::from_u16`
- `CodePlane::from_u8_interleaved`
- `CodePlane::from_u16_interleaved`
- `CodePlane::step_words`
- `PixelFormat::byte_order`

### struct_marked_non_exhaustive

- `struct MasteringDisplay`

### struct_missing

- `struct ContentLightLevel`
- `struct MasteringDisplay`
- `struct DiffuseWhite`

### struct_pub_field_missing

- `field gray_expand of struct ConvertOptions`
- `field alpha_policy of struct ConvertOptions`
- `field depth_policy of struct ConvertOptions`
- `field luma of struct ConvertOptions`
- `field clip_out_of_gamut of struct ConvertOptions`

### struct_pub_field_now_doc_hidden

- `field ConvertOptions.gray_expand`
- `field ConvertOptions.alpha_policy`
- `field ConvertOptions.depth_policy`
- `field ConvertOptions.luma`
- `field ConvertOptions.clip_out_of_gamut`
- `Summary semver requires new major version: 15 major and 0 minor checks failed`
- `Finished [   3.540s] zenpixels`
- `Building zenpixels-convert v0.3.1 (current)`
- `Built [   3.986s] (current)`
- `Parsing zenpixels-convert v0.3.1 (current)`
- `Parsed [   0.007s] (current)`
- `Building zenpixels-convert v0.2.17 (baseline)`
- `Built [   3.638s] (baseline)`
- `Parsing zenpixels-convert v0.2.17 (baseline)`
- `Parsed [   0.006s] (baseline)`
- `Checking zenpixels-convert v0.2.17 -> v0.3.1 (assume patch change)`
- `Checked [   0.014s] 229 checks: 213 pass, 16 fail, 0 warn, 31 skip`

### enum_missing

- `enum estimate::SimdTier`
- `enum SimdTier`
- `enum cms::ColorPriority`
- `enum ColorPriority`
- `enum cms::RenderingIntent`
- `enum RenderingIntent`

### enum_variant_missing

- `variant CodesError::Unmeasurable`

### feature_missing

- `feature pipeline in the package's Cargo.toml`

### function_missing

- `function output::finalize_for_output_with`
- `function finalize_for_output_with`
- `function gamut::apply_matrix_f32`
- `function apply_matrix_f32`
- `function orient::apply_orientation`
- `function oklab::fast_cbrt`
- `function conversion_cost`
- `function adapt::adapt_for_encode_with_intent`
- `function adapt::check_opaque`
- `function orient::apply_orientation_in_place`
- `function hdr::reinhard_inverse`
- `function reinhard_inverse`
- `function adapt::try_adapt_in_place`
- `function try_adapt_in_place`
- `function icc_profiles::synthesize_gray_icc_for_cicp`
- `function icc_profiles::icc_profile_for_primaries`
- `function negotiate`
- `function oklab::rgb_to_lms_matrix`
- `function ideal_format`
- `function adapt::adapt_for_encode`
- `function hdr::quantize_to`
- `function quantize_to`
- `function adapt::adapt_for_encode_explicit`
- `function adapt_for_encode_explicit`
- `function gamut::apply_matrix_row_rgba_f32`
- `function apply_matrix_row_rgba_f32`
- `function oklab::rgb_to_oklab`
- `function adapt::adapt_for_encode_with_intent_cow`
- `function gamut::conversion_matrix`
- `function conversion_matrix`
- `function requires_cms`
- `function hdr::exposure_tonemap`
- `function exposure_tonemap`
- `function convert_row`
- `function icc_profiles::display_p3_icc`
- `function best_match`
- `function gamut::mat3_mul`
- `function orient::apply_orientation_into`
- `function oklab::lms_to_rgb_matrix`
- `function conversion_cost_with_provenance`
- `function adapt::adapt_for_encode_cow`
- `function gamut::apply_matrix_row_f32`
- `function apply_matrix_row_f32`
- `function adapt::adapt_for_encode_explicit_cow`
- `function adapt_for_encode_explicit_cow`
- `function best_match_with`
- `function hdr::reinhard_tonemap`
- `function reinhard_tonemap`
- `function oklab::oklab_to_rgb`
- `function adapt::convert_buffer`
- `function icc_profiles::synthesize_icc_for_cicp`
- `function output::finalize_for_output`
- `function finalize_for_output`
- `function codes::measure_exact_depth`

### inherent_method_missing

- `ConvertPlan::estimate_in`
- `ConvertPlan::estimate`

### module_missing

- `mod estimate`
- `mod orient`

### pub_module_level_const_missing

- `REC2020_V4`
- `ADOBE_RGB_V4`
- `PROPHOTO_V4`

### struct_marked_non_exhaustive

- `struct FormatOption`
- `struct ConversionCost`

### struct_missing

- `struct estimate::ComputeEnvironment`
- `struct ComputeEnvironment`
- `struct adapt::Adapted`
- `struct estimate::ImageCharacteristics`
- `struct ImageCharacteristics`
- `struct hdr::HdrMetadata`
- `struct HdrMetadata`
- `struct estimate::ResourceEstimate`
- `struct ResourceEstimate`

### struct_pub_field_missing

- `field effort of struct ConversionCost`
- `field loss of struct ConversionCost`
- `field hdr of struct OutputMetadata`

### struct_pub_field_now_doc_hidden

- `field ConversionCost.effort`
- `field ConversionCost.loss`

### trait_added_supertrait

- `trait ext::PixelBufferConvertExt gained Sealed`
- `trait ext::PixelBufferConvertExt gained Sized`
- `trait PixelBufferConvertExt gained Sealed`
- `trait PixelBufferConvertExt gained Sized`
- `trait ext::ColorPrimariesExt gained Sealed`
- `trait ColorPrimariesExt gained Sealed`
- `trait ext::TransferFunctionExt gained Sealed`
- `trait TransferFunctionExt gained Sealed`

### trait_method_added

- `trait method ext::PixelBufferConvertExt::convert_to_with_options`
- `trait method ext::PixelBufferConvertExt::try_adapt_in_place`
- `trait method ext::PixelBufferConvertExt::into_oriented`
- `trait method PixelBufferConvertExt::convert_to_with_options`
- `trait method PixelBufferConvertExt::try_adapt_in_place`
- `trait method PixelBufferConvertExt::into_oriented`

### trait_missing

- `trait cms::ColorManagement`
- `trait ColorManagement`

### trait_newly_sealed

- `trait ext::PixelBufferConvertExt`
- `trait PixelBufferConvertExt`
- `trait ext::ColorPrimariesExt`
- `trait ColorPrimariesExt`
- `trait ext::TransferFunctionExt`
- `trait TransferFunctionExt`

### trait_no_longer_dyn_compatible

- `trait PixelBufferConvertExt`
- `Summary semver requires new major version: 16 major and 0 minor checks failed`
- `Finished [   8.179s] zenpixels-convert`
