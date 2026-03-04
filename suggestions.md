# Suggestions for zenpixels

From a research rewrite that rebuilt the crate from scratch, then compared
against production. These are concrete, actionable changes — not theoretical.

## Replace PixelFormat struct with a flat enum

The current `PixelFormat` struct has 5 orthogonal fields (`channel_type`,
`color_model`, `alpha`, `transfer`, `byte_order`). This creates ~1,800 valid
field combinations, but only ~23 are meaningful. The `InterleaveFormat` enum
exists because you can't match on a struct — but then `interleaved_format()`
is fallible because some valid `PixelFormat` values have no `InterleaveFormat`.

A `#[repr(u8)]` flat enum with ~18 variants eliminates:

- `InterleaveFormat` entirely (the format IS the dispatch enum)
- `ChannelLayout`, `ByteOrder`, `ColorModel` enums (queries become methods)
- The `Option<InterleaveFormat>` fallibility in dispatch paths
- ~1,777 impossible states

All derived properties (`channels()`, `has_alpha()`, `bytes_per_pixel()`,
`is_linear()`, `component_type()`) become match arms on the enum. The research
crate's `format.rs` is 174 lines vs production's 2,894-line `descriptor.rs`.

### PixelFormat should encode byte layout, not gamut

Primaries (sRGB vs P3 vs BT.2020) don't change the byte layout. A P3 image
and an sRGB image at 8-bit RGB have identical byte streams — the difference
is which triangle on the CIE diagram those values map to. Primaries belong
on `PixelDescriptor` (via `ColorPrimaries`) or CICP, not on `PixelFormat`.

The research crate got this wrong with `DisplayP3_8`, `Bt601_8`, `Bt709_8`,
`Bt2020_16` as separate format variants. These should be `Rgb8` or `Rgb16`
with different primaries on the descriptor.

Naming matters here: calling a P3 buffer `Srgb8` invites accidental clamping.
Use gamut-neutral names: `Rgb8`, `Rgba8`, `Rgb16`, `LinRgbF32`, etc. The
name says "3 channels, u8, gamma-encoded" without claiming a gamut. Which
gamma curve and which gamut are on the descriptor.

### What about HDR / future formats?

The flat enum gets a new variant when the byte-level encoding differs.
PQ and HLG have different transfer curves than sRGB, so `RgbPq16` or
`RgbHlg16` would be new variants (if the transfer function isn't already
a separate field on `PixelDescriptor`). But P3 at 8-bit sRGB transfer
is just `Rgb8` — no new variant needed.

If variant count becomes unwieldy (50+), revisit — but real codecs produce
maybe 20 distinct byte-level formats total.

### TransferFunction as a separate queryable type

Even with a flat enum, a `TransferFunction` enum (`Linear`, `Srgb`, `Bt709`,
`Pq`, `Hlg`) is useful for generic code that applies/removes gamma. Add a
`transfer_function()` method on `PixelFormat` that returns it. The transfer
function is derived from the variant, not stored separately.

## Drop the rgb crate dependency

Production uses `Rgb<u8>`, `Rgba<u8>`, etc. from the `rgb` crate behind a
feature gate. Problems:

- `Rgb<u8>` says nothing about transfer function or color space
- Feature-gates `Pixel` trait impls, so basic typed access requires `buffer` feature
- The `rgb` crate's Pod story has had soundness issues historically

Own types with gamut-neutral names — `Rgb8`, `LinRgbF32`, `OklabF32` — are
`#[repr(C)]` + `bytemuck::Pod` unconditionally. `Rgb8` means "3×u8,
gamma-encoded" without claiming sRGB gamut. The gamut lives on the descriptor.

## Make `Pixel: bytemuck::Pod` a supertrait

Currently `Pixel: Copy + 'static`, with bytemuck bounds added per-method
behind the `buffer` feature. This means `row_pixels()`, `from_pixels()`, and
`cast_slice()` require extra trait bounds at every call site.

With `Pixel: bytemuck::Pod + Copy + 'static`, any function that takes
`P: Pixel` gets zero-copy typed access for free. No feature gates, no
conditional bounds. One trait, consistent everywhere.

## Remove feature gates for core buffer functionality

The `buffer` feature gates `Pixel` impls, conversion methods on slices, and
imgref interop. Without `buffer`, the crate's buffer types exist but can't do
typed pixel access — which defeats their purpose.

Core typed access (`row_pixels()`, `from_pixels()`, `Pixel` impls for concrete
types) should always be available. Gate external interop (imgref, rgb) behind
features, not the crate's own functionality.

## Move codec-specific modules to zencodec-types

`registry`, `adapt`, `op_format`, and `path` are codec integration modules.
They couple the pixel crate to codec capabilities (which formats JPEG supports,
what AV1's effective bit depth is, etc.). This is a layering violation — pixel
interchange types shouldn't know about codecs.

Move to zencodec-types where the codec traits and format negotiation already
live. The pixel crate stays focused on representing and accessing pixels.

## Move conversion methods off buffer types

`PixelBufferConvertExt` (`convert_to()`, `try_add_alpha()`, `try_widen_to_u16()`,
etc.) lives in the convert crate but adds methods to the pixel crate's buffer
types. This creates a circular concern — buffer types conceptually depend on
conversion logic.

Keep conversion as free functions or on a dedicated converter type. The buffer
types do one thing: hold pixels with metadata. The convert crate does one thing:
transform between formats.

## Color metadata: origin + current state

For ICC-aware pipelines (decode P3/AdobeRGB, process in Oklab, re-encode with
target profile), the buffer needs to carry:

1. **Origin** — what the source file said (ICC bytes, CICP, provenance)
2. **Current state** — what color space the pixels are in now (PixelDescriptor)

The conversion to an output profile and the metadata generation for the encoder
should be a single atomic operation. A `finalize_for_output(target_profile,
pixel_format, &cms)` function converts pixels to the target profile's space
AND produces matching metadata. This prevents the pixels-don't-match-metadata
bug where sRGB values get encoded with a P3 ICC profile.

The CMS should be pluggable via a trait in the convert crate. Core types carry
raw ICC bytes (`Arc<[u8]>`); the convert crate defines `ColorManagement` trait;
moxcms or lcms2 implements it. Named profiles (sRGB, P3, BT.2020) use
hardcoded matrices and skip the CMS entirely.

### ICC-to-CICP equivalence

Many ICC profiles describe named color spaces. Detection should be two-tier:
hash table of known ICC byte sequences for instant lookup, then CMS-based
semantic comparison (extract matrix + TRC, compare against known values) as
fallback. This belongs in the CMS trait, not in core.

## ConvertOptions — keep and expand

The `AlphaPolicy`, `DepthPolicy`, and `LumaCoefficients` types are good.
Making lossy operations explicit prevents silent data loss. The research crate
lacks these and silently strips alpha / truncates depth. Port them as-is.

## Provenance tracking — keep

`Provenance` with origin depth and primaries is a smart optimization. Knowing
pixels originated as u8 means u8 -> f32 -> u8 is lossless and shouldn't be
penalized. Keep this in the negotiation system.

## Buffer type simplification

Both codebases use the same phantom-typed `PixelSlice<P>`, `PixelSliceMut<P>`,
`PixelBuffer<P>` pattern. The research version is ~3x smaller (963 vs 2,715
lines) for the same core functionality, mainly because it doesn't carry
`Option<Arc<ColorContext>>`, doesn't have imgref interop, and doesn't have
conversion methods on the types themselves.

If color metadata moves to the buffer (see above), the size difference shrinks.
But the principle holds: buffer types should provide data access, not conversion.

## Drop imgref dependency

`PixelSlice` serves the same purpose as `ImgRef` (borrowed 2D strided view).
Having both creates confusion about which to use. The `as_imgref()` /
`from_imgvec()` bridges exist because external crates use imgref, but for new
code the native types are cleaner.

Migration path: keep imgref interop behind a feature gate for backward
compatibility, but don't use it internally. New code uses `PixelSlice`.
