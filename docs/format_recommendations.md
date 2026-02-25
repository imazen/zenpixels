# Pixel Format Recommendation Matrix

For each input image type: recommended transfer, working, and encoding representations
with clamp rules and websafe codec cross-reference.

Interactive browser: [format_browser.html](format_browser.html) | Machine-readable: [format_recommendations.json](format_recommendations.json)

---

## Input Scenarios

### Standard SDR (8-bit sRGB)

| ID | Format | Layout | Source Codecs | Bytes/px |
|----|--------|--------|---------------|----------|
| `rgb8_srgb` | RGB u8 sRGB | 3ch | JPEG PNG WebP AVIF JXL BMP GIF PNM | 3 |
| `rgba8_srgb` | RGBA u8 sRGB | 4ch straight | PNG WebP AVIF JXL GIF | 4 |
| `gray8_srgb` | Gray u8 sRGB | 1ch | JPEG PNG JXL GIF PNM | 1 |
| `graya8_srgb` | GrayAlpha u8 sRGB | 2ch straight | PNG | 2 |
| `bgra8_srgb` | BGRA u8 sRGB | 4ch straight | JPEG PNG JXL BMP PNM | 4 |

### High-Precision SDR (16-bit sRGB)

| ID | Format | Layout | Source Codecs | Bytes/px |
|----|--------|--------|---------------|----------|
| `rgb16_srgb` | RGB u16 sRGB | 3ch | PNG PNM | 6 |
| `rgba16_srgb` | RGBA u16 sRGB | 4ch straight | PNG Farbfeld PNM | 8 |
| `gray16_srgb` | Gray u16 sRGB | 1ch | PNG PNM | 2 |
| `graya16_srgb` | GrayAlpha u16 sRGB | 2ch straight | PNG | 4 |

### Linear / HDR (f32 Linear)

| ID | Format | Layout | Source Codecs | Bytes/px |
|----|--------|--------|---------------|----------|
| `rgbf32_linear` | RGB f32 Linear | 3ch | PNG JXL PNM | 12 |
| `rgbaf32_linear` | RGBA f32 Linear | 4ch straight | PNG JXL PNM | 16 |
| `grayf32_linear` | Gray f32 Linear | 1ch | PNG JXL PNM | 4 |
| `grayaf32_linear` | GrayAlpha f32 Linear | 2ch straight | PNG JXL PNM | 8 |

### JPEG Extended Decode (Internal API)

These use zenjpeg's internal decode paths, not the zencodec API.
They produce f32 output with **overshoot** (values outside \[0,1\]).

| ID | Format | Eff. Bits | Overshoot | Decode Mode |
|----|--------|-----------|-----------|-------------|
| `jpeg_f32_srgb_deblock` | RGB f32 sRGB | 8 | yes | Integer IDCT, sRGB gamma |
| `jpeg_f32_linear_debias` | RGB f32 Linear | 8 | yes | Debiased dequant, linear |
| `jpeg_f32_srgb_precise` | RGB f32 sRGB | 10 | yes | Laplacian bias, ~10 bits recovered |
| `jpeg_f32_linear_precise` | RGB f32 Linear | 10 | yes | Laplacian bias + linear, best quality |
| `jpeg_gray_f32_linear` | Gray f32 Linear | 8 | yes | Grayscale debiased |

---

## Transfer Representation

The recommended internal storage format — how to hold the decoded data before operations.

| Input Class | Transfer Rep | Rationale |
|-------------|--------------|-----------|
| u8 sRGB (any layout) | Same as source | Compact, no conversion needed |
| u16 sRGB (any layout) | Same as source | Full 16-bit precision preserved |
| f32 Linear (any layout) | Same as source | Maximum precision and dynamic range |
| JPEG f32 unclamped | Same as source (**unclamped**) | Overshoot contains deblocking quality; clamping destroys it |

**Rule:** Keep data in its native format until an operation requires conversion.
Never convert "just in case" — every conversion adds latency; some add loss.

---

## Working Format by Operation Group

### Identity Operations: Passthrough, ColorMatrix, Arithmetic

**Rule:** Use source format directly. No conversion.

| Input | Working Format | Clamp | Notes |
|-------|---------------|-------|-------|
| Any | Same as source | N/A | Zero cost |
| JPEG unclamped | Same (unclamped) | unclamped | Preserve overshoot through passthrough |

### Linear-Light Math: ResizeGentle, ResizeSharp, OklabSharpen, OklabAdjust, Tonemap, IccTransform

**Rule:** Convert to f32 Linear.

| Input | Working Format | Clamp | Notes |
|-------|---------------|-------|-------|
| RGB u8 sRGB | RGB f32 Linear | **unclamped** | u8→f32→u8 round-trip lossless (ULP proven, 256³ values) |
| RGBA u8 sRGB | RGBA f32 Linear Straight | **unclamped** | Alpha preserved, round-trip lossless |
| Gray u8 sRGB | Gray f32 Linear | **unclamped** | Single-channel, lossless |
| GrayAlpha u8 sRGB | GrayAlpha f32 Linear Straight | **unclamped** | Luma + alpha, lossless round-trip |
| BGRA u8 sRGB | RGBA f32 Linear | **unclamped** | Swizzle B↔R during conversion |
| RGB u16 sRGB | RGB f32 Linear | **unclamped** | u16→f32 lossless (16 bits ⊂ 23-bit mantissa) |
| RGBA u16 sRGB | RGBA f32 Linear | **unclamped** | u16 fits exactly in f32 |
| GrayAlpha u16 sRGB | GrayAlpha f32 Linear Straight | **unclamped** | u16→f32 lossless, alpha preserved |
| f32 Linear (any) | Same (already f32 Linear) | **unclamped** | Zero conversion cost |
| JPEG f32 sRGB unclamped | RGB f32 Linear (unclamped) | **unclamped** | Apply EOTF; overshoot transforms correctly |
| JPEG f32 Linear unclamped | Same | **unclamped** | Already ideal |

**Clamp rules for linear_math:**
- **ResizeSharp:** MUST be unclamped. Lanczos/CatmullRom produce overshoot (ringing).
  Clamping causes visible banding at high-contrast edges.
- **ResizeGentle:** Unclamped OK. Mitchell has minor overshoot.
- **Tonemap, IccTransform, OklabSharpen, OklabAdjust:** Clamp output to \[0,1\] after operation.

### Soft Filters: Blur, Sharpen

**Rule:** sRGB is acceptable. Gamma error is negligible for convolution.

| Input | Working Format | Clamp | Notes |
|-------|---------------|-------|-------|
| u8 sRGB (any layout) | Same as source | clamped \[0,255\] | No conversion overhead |
| u16 sRGB (any layout) | Same as source | clamped \[0,65535\] | 16-bit precision |
| f32 Linear (any) | Same as source | clamped \[0,1\] post-op | Don't convert to sRGB just for blur |
| JPEG f32 unclamped | Same (unclamped) | unclamped | Preserve overshoot through filter |

### Compositing: Composite (Porter-Duff, blend modes)

**Rule:** RGBA f32 Linear Premultiplied. Always.

| Input | Working Format | Clamp | Notes |
|-------|---------------|-------|-------|
| Any RGB/Gray | RGBA f32 Linear Premul | clamped \[0,1\] | Add opaque alpha (1.0). Expand Gray→RGB if needed. |
| Any RGBA/GrayAlpha Straight | RGBA f32 Linear Premul | clamped \[0,1\] | Expand Gray→RGB if needed. Straight→Premul: ≤2 ULP rounding at low alpha |
| JPEG unclamped | RGBA f32 Linear Premul | clamped \[0,1\] | Clamp overshoot before compositing |

### Palette Quantization: Quantize (GIF encoding)

**Rule:** sRGB u8 required. Perceptual distance metrics need gamma space.

| Input | Working Format | Clamp | Notes |
|-------|---------------|-------|-------|
| u8 sRGB (any) | Same as source | clamped \[0,255\] | Already in target format |
| u16 sRGB | u8 sRGB | clamped \[0,255\] | 8 bits depth lost |
| f32 Linear | u8 sRGB | clamped \[0,255\] | Apply OETF + quantize. Major depth loss. |
| JPEG f32 unclamped | u8 sRGB | clamped \[0,255\] | Clamp overshoot, then quantize |

---

## Clamp / Unclamped Rules Summary

| Rule | When | Why |
|------|------|-----|
| **Unclamped required** | ResizeSharp working format | Lanczos/CatmullRom overshoot is deliberate ringing; clamping creates banding |
| **Unclamped required** | JPEG f32 extended decode | IDCT ringing preserved for smoother re-encoding |
| **Unclamped OK** | ResizeGentle, any f32 transfer | Mitchell has minor overshoot; no harm in preserving |
| **Clamp after op** | Tonemap, IccTransform, OklabAdjust, Composite | Operations produce bounded output; clamp to \[0,1\] |
| **Clamp before encode** | All integer-backed codecs | u8/u16 encoders silently clamp; explicit clamp is cleaner |
| **No clamp needed** | JXL f32, PNM PFM | Can store arbitrary float values |

---

## Encoding Representation

Target format for equivalent-quality encoding. Preserves source precision.

| Input | Encoding Rep | ICC | CICP | Notes |
|-------|-------------|-----|------|-------|
| u8 sRGB (any) | Same (u8 sRGB) | Preserve | sRGB default | All codecs accept natively |
| u16 sRGB (any) | u16 sRGB | Preserve | Custom profile | PNG, Farbfeld, PNM native. JXL via f32. |
| f32 Linear (any) | f32 Linear | Preserve | Signal transfer+primaries | PNG, JXL, PNM only |
| JPEG f32 8-bit eff | u8 sRGB | Preserve | sRGB default | 8-bit effective → u8 is sufficient |
| JPEG f32 10-bit eff | u16 sRGB or f32 | Preserve | Custom | u8 loses 2 bits of recovered precision |

---

## Encode Format Candidates

Which codecs can encode at equivalent quality for each input class.

| Input Class | Lossless | Lossy | Notes |
|-------------|----------|-------|-------|
| RGB u8 sRGB | PNG JXL BMP PNM | JPEG WebP AVIF JXL | All codecs accept natively |
| RGBA u8 sRGB | PNG JXL PNM | WebP AVIF JXL | JPEG strips alpha |
| Gray u8 sRGB | PNG JXL PNM | JPEG JXL | WebP/AVIF need RGB expansion |
| GrayAlpha u8 sRGB | PNG | JXL | JPEG drops alpha; WebP/AVIF need Gray→RGB expansion |
| RGB u16 sRGB | PNG Farbfeld PNM | JXL (f32) | Web codecs truncate to u8 |
| RGBA u16 sRGB | PNG Farbfeld PNM | JXL (f32) | JPEG: alpha + depth loss |
| Gray u16 sRGB | PNG PNM | JXL (f32) | Limited codec support |
| GrayAlpha u16 sRGB | PNG | JXL (f32) | Very limited native support |
| RGB f32 Linear | PNG JXL PNM | JXL | Web codecs: catastrophic depth loss |
| RGBA f32 Linear | PNG JXL PNM | JXL | Web codecs: catastrophic loss |
| Gray f32 Linear | PNG JXL PNM | JXL | Very limited |
| GrayAlpha f32 Linear | PNG JXL PNM | JXL | Very limited |
| JPEG extended 8-bit | PNG JXL BMP PNM | JPEG WebP AVIF JXL | 8-bit eff → same as u8 sRGB |
| JPEG extended 10-bit | PNG(u16) JXL(f32) PNM | JXL | u8 codecs lose 2 bits |

---

## Websafe Cross-Reference

What happens when you encode each input type to the five major web codecs.

**Legend:** ✅ native (zero loss) · ⚠️ conversion needed (minor/noted loss) · ❌ significant loss

**Note:** JXL uses gamma 1/2.2 (power 2.2), not the sRGB transfer curve. The sRGB curve
has a linear segment near black then ~2.4 power; gamma 2.2 is a pure power function.
At u8 precision this causes ±1 value error in the darkest tones. JXL f32 Linear is
unaffected (no transfer function conversion). For truly lossless sRGB u8 round-trips,
prefer PNG or use JXL via the f32 Linear path.

### RGB Formats

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| RGB u8 sRGB | ✅ native | ✅ native | ✅ lossless | ✅ native | ⚠️ gamma 2.2 (±1 dark) |
| RGB u16 sRGB | ⚠️ u8 truncation | ⚠️ u8 truncation | ✅ lossless u16 | ⚠️ u8 truncation | ✅ f32 (lossless) |
| RGB f32 Linear | ❌ u8+OETF | ❌ u8+OETF | ✅ lossless f32 | ❌ u8+OETF | ✅ native f32 |

### RGBA Formats

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| RGBA u8 sRGB | ⚠️ alpha dropped | ✅ native | ✅ lossless | ✅ native | ⚠️ gamma 2.2 (±1 dark) |
| RGBA u16 sRGB | ❌ alpha+u8 | ⚠️ u8 truncation | ✅ lossless u16 | ⚠️ u8 truncation | ✅ f32 (lossless) |
| RGBA f32 Linear | ❌ alpha+u8+OETF | ⚠️ u8+OETF | ✅ lossless f32 | ⚠️ u8+OETF | ✅ native f32 |

### Grayscale Formats

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| Gray u8 sRGB | ✅ native gray | ⚠️ →RGB (3x) | ✅ lossless gray | ⚠️ →RGB (3x) | ⚠️ gamma 2.2 (±1 dark) |
| Gray u16 sRGB | ⚠️ u8 gray | ⚠️ →RGB+u8 | ✅ lossless u16 gray | ⚠️ →RGB+u8 | ✅ f32 gray |
| Gray f32 Linear | ❌ u8+OETF gray | ❌ →RGB+u8+OETF | ✅ lossless f32 gray | ❌ →RGB+u8+OETF | ✅ native f32 gray |

### GrayAlpha Formats

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| GrayAlpha u8 sRGB | ⚠️ alpha dropped | ⚠️ →RGBA (3x) | ✅ native GA8 | ⚠️ →RGBA (3x) | ⚠️ gamma 2.2 (±1 dark) |
| GrayAlpha u16 sRGB | ❌ alpha+u8 | ⚠️ →RGBA+u8 | ✅ lossless GA16 | ⚠️ →RGBA+u8 | ✅ f32 GA |
| GrayAlpha f32 Linear | ❌ alpha+u8+OETF | ❌ →RGBA+u8+OETF | ✅ lossless f32 GA | ❌ →RGBA+u8+OETF | ✅ native f32 GA |

### BGRA Format

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| BGRA u8 sRGB | ⚠️ swizzle+alpha | ✅ swizzle only | ✅ swizzle only | ✅ swizzle only | ⚠️ gamma 2.2 + swizzle |

### JPEG Extended Decode

| Input | JPEG | WebP | PNG | AVIF | JXL |
|-------|------|------|-----|------|-----|
| f32 sRGB deblock (8b) | ✅ clamp→u8 | ✅ clamp→u8 | ✅ clamp→u8 | ✅ clamp→u8 | ⚠️ gamma 2.2 (±1 dark) |
| f32 Linear debias (8b) | ✅ OETF→u8 | ✅ OETF→u8 | ✅ OETF→u8 | ✅ OETF→u8 | ⚠️ gamma 2.2 (±1 dark) |
| f32 sRGB precise (10b) | ⚠️ -2 bits | ⚠️ -2 bits | ✅ u16 (lossless) | ⚠️ -2 bits | ✅ f32 (lossless) |
| f32 Linear precise (10b) | ⚠️ -2 bits | ⚠️ -2 bits | ✅ u16 (lossless) | ⚠️ -2 bits | ✅ f32 (lossless) |
| Gray f32 Linear (8b) | ✅ OETF→u8 | ⚠️ →RGB | ✅ OETF→u8 | ⚠️ →RGB | ⚠️ gamma 2.2 (±1 dark) |

### ICC / CICP Handling

| Codec | ICC Decode | ICC Encode | CICP | Notes |
|-------|-----------|-----------|------|-------|
| JPEG | ✅ | ✅ | ❌ | APP2 ICC marker |
| WebP | ✅ | ✅ | ❌ | RIFF ICCP chunk |
| PNG | ✅ | ✅ | ✅ | iCCP chunk + cICP chunk |
| AVIF | ✅ | ❌ | ✅ | ICC decode only; CICP in ISOBMFF/AV1 |
| JXL | ✅ | ✅ | ✅ | ICC in codestream; CICP via metadata |
| GIF | ❌ | ❌ | ❌ | No color management |
| BMP | ❌ | ❌ | ❌ | No color management |
| Farbfeld | ❌ | ❌ | ❌ | No color management |
| PNM | ❌ | ❌ | ❌ | No color management |

---

## ULP Proofs Referenced

The working format recommendations above are backed by exhaustive numerical proofs:

| Claim | Test | Values Tested |
|-------|------|---------------|
| u8 sRGB → f32 Linear → u8 sRGB = lossless | `ulp_u8_srgb_f32_linear_roundtrip_full_cube` | 16,777,216 (256³) |
| u8 → u16 → u8 = lossless | `ulp_u8_u16_roundtrip` | 256 |
| u16 → f32 → u16 = lossless | `ulp_u16_f32_roundtrip_exhaustive` | 65,536 |
| u16 → u8 max error = 1 | `ulp_u16_to_u8_max_error` | 65,536 |
| u8 → u16 = v×257 exact | `ulp_u8_to_u16_exact` | 256 |
| Premul round-trip u8 ≤1 ULP at low alpha | `ulp_premul_roundtrip_u8_exhaustive` | 65,536 (256×256) |
| Premul round-trip f32 ≤2 ULP | `ulp_premul_roundtrip_f32` | 1,000,000 |
| BGRA↔RGBA swizzle = lossless | `ulp_bgra_rgba_swizzle_roundtrip` | 1,000,000 |
| sRGB EOTF is monotonic | `ulp_srgb_eotf_monotonic` | 256 |
| Gray→RGB→Gray = lossless (for pure gray) | `ulp_gray_rgb_gray_roundtrip` | 256 |
| GrayAlpha→RGBA→GrayAlpha: gray lossless, alpha lossless | (via gray_rgb + add_drop_alpha proofs) | 256 |
| RGB→RGBA→RGB = lossless | `ulp_add_drop_alpha_roundtrip` | 256 |
