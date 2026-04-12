# Tone Mapping Example Design

How to handle `HdrRequiresToneMapping` by converting any HDR content to sRGB
using `ultrahdr-core`'s tone mapping before `finalize_for_output()`.

## The problem

`finalize_for_output()` rejects HDR (PQ/HLG) sources targeting SDR outputs:

```rust
let result = finalize_for_output(&buffer, &origin, OutputProfile::Icc(srgb), Rgb8, &cms);
// → Err(HdrRequiresToneMapping)
```

The caller must tone map first, then retry with SDR metadata.

## ultrahdr-core API surface

### One-shot (simplest)

```rust
// Takes RawImage, returns RGBA8 sRGB bytes
ultrahdr_core::color::tonemap::tonemap_image_to_srgb8(
    img: &RawImage,
    target_gamut: ColorGamut,  // ColorGamut::Bt709 for sRGB
) -> Result<Vec<u8>>
```

Handles PQ, HLG, sRGB, and linear transfers. Does gamut conversion + tone curve + sRGB OETF + quantize in one pass. Output is always RGBA8 sRGB.

Pixel format support: `Rgba8`, `Rgb8`, `Rgba32F`, `Rgba1010102`, `P010`.

### Per-pixel (for custom pipelines)

```rust
// PQ linear RGB → SDR linear RGB (BT.709 gamut, 0-1 range)
tonemap_pq_to_sdr(pq_rgb: [f32; 3], config: &ToneMapConfig) -> [f32; 3]

// HLG linear RGB → SDR linear RGB
tonemap_hlg_to_sdr(hlg_rgb: [f32; 3], config: &ToneMapConfig) -> [f32; 3]
```

### Streaming (for zenpipe integration)

```rust
StreamingTonemapper::new(width, height, config) -> Result<Self>
  .push_rows(data: &[f32], stride, num_rows) -> Result<Vec<TonemapOutput>>
  .finish() -> Result<Vec<TonemapOutput>>
```

`TonemapOutput` contains linear f32 rows. Caller applies sRGB OETF via
`StreamingTonemapper::linear_to_srgb8_rgba()` or `linear_to_srgb8_rgb()`.

### BT.2408 (broadcast standard)

```rust
Bt2408Tonemapper::new(content_max_nits: f32, display_max_nits: f32) -> Self
  .tonemap_rgb(rgb: [f32; 3]) -> [f32; 3]
  .tonemap_luminance(nits: f32) -> f32
```

Uses ITU-R BT.2408 reference EETF. Takes scene-linear nits, outputs display-linear nits.

## Bridge: PixelBuffer ↔ RawImage

The key challenge. `PixelBuffer` uses `PixelDescriptor` (format + transfer + primaries).
`RawImage` uses `PixelFormat` + `ColorGamut` + `ColorTransfer` enums.

### Cicp → ultrahdr-core mapping

```rust
fn cicp_to_gamut(cicp: &Cicp) -> Option<ColorGamut> {
    match cicp.color_primaries {
        1  => Some(ColorGamut::Bt709),
        9  => Some(ColorGamut::Bt2100),
        12 => Some(ColorGamut::P3),
        _  => None,
    }
}

fn cicp_to_transfer(cicp: &Cicp) -> Option<ColorTransfer> {
    match cicp.transfer_characteristics {
        1 | 6 | 13 => Some(ColorTransfer::Srgb),
        8          => Some(ColorTransfer::Linear),
        16         => Some(ColorTransfer::Pq),
        18         => Some(ColorTransfer::Hlg),
        _          => None,
    }
}
```

### PixelBuffer → RawImage

```rust
fn pixel_buffer_to_raw_image(
    buffer: &PixelBuffer,
    cicp: &Cicp,
) -> Result<RawImage, &'static str> {
    let format = match buffer.descriptor().format {
        PixelFormat::Rgba8  => ultrahdr_core::PixelFormat::Rgba8,
        PixelFormat::Rgb8   => ultrahdr_core::PixelFormat::Rgb8,
        PixelFormat::RgbaF32 => ultrahdr_core::PixelFormat::Rgba32F,
        _ => return Err("unsupported pixel format for tone mapping"),
    };
    let gamut = cicp_to_gamut(cicp).ok_or("unsupported gamut")?;
    let transfer = cicp_to_transfer(cicp).ok_or("unsupported transfer")?;

    let slice = buffer.as_slice();
    let data = slice.contiguous_bytes().into_owned();
    let stride = buffer.width() * buffer.descriptor().bytes_per_pixel() as u32;

    RawImage::from_data(buffer.width(), buffer.height(), format, data)
        .map(|mut img| { img.gamut = gamut; img.transfer = transfer; img.stride = stride; img })
        .map_err(|_| "invalid image dimensions")
}
```

### SDR output → PixelBuffer + ColorOrigin

After `tonemap_image_to_srgb8()`, the output is RGBA8 sRGB:

```rust
let sdr_bytes = tonemap_image_to_srgb8(&raw_image, ColorGamut::Bt709)?;
let sdr_buffer = PixelBuffer::from_vec(
    sdr_bytes, width, height, PixelDescriptor::RGBA8_SRGB,
)?;
let sdr_origin = ColorOrigin::from_cicp(Cicp::SRGB);
```

## Complete workflow

```rust
fn convert_any_to_srgb(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    cms: &impl ColorManagement,
) -> Result<EncodeReady, ConvertError> {
    // Try direct conversion first
    match finalize_for_output(buffer, origin, OutputProfile::Named(Cicp::SRGB), Rgb8, cms) {
        Ok(ready) => return Ok(ready),
        Err(e) if *e.error() == ConvertError::HdrRequiresToneMapping => {
            // Fall through to tone mapping
        }
        Err(e) => return Err(e),
    }

    // HDR path: tone map first
    let cicp = origin.cicp.ok_or(ConvertError::CmsError("HDR without CICP".into()))?;
    let raw = pixel_buffer_to_raw_image(buffer, &cicp)?;
    let sdr_bytes = tonemap_image_to_srgb8(&raw, ColorGamut::Bt709)?;
    let sdr_buffer = PixelBuffer::from_vec(
        sdr_bytes, buffer.width(), buffer.height(), PixelDescriptor::RGBA8_SRGB,
    )?;
    let sdr_origin = ColorOrigin::from_cicp(Cicp::SRGB);

    // Now it's SDR → finalize succeeds
    finalize_for_output(&sdr_buffer, &sdr_origin, OutputProfile::Named(Cicp::SRGB), Rgb8, cms)
}
```

## Test plan

### Integration test: `tests/tonemap_integration.rs`

Feature-gated behind `tonemap` (pulls `ultrahdr-core` with `transfer` feature).

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | `pq_bt2020_to_srgb8` | Synthetic PQ RGBA32F gradient | sRGB RGBA8, values in 0-255, not clipped to 0/255 |
| 2 | `hlg_bt2020_to_srgb8` | Synthetic HLG RGBA32F | sRGB RGBA8, midtones preserved |
| 3 | `pq_p3_to_srgb8` | PQ Display P3 | sRGB RGBA8, gamut mapped |
| 4 | `sdr_passthrough` | sRGB RGBA8 | Unchanged (no tone mapping) |
| 5 | `convert_any_pq` | PQ RGBA32F | `convert_any_to_srgb` succeeds |
| 6 | `convert_any_sdr` | sRGB RGB8 | `convert_any_to_srgb` succeeds (direct path) |
| 7 | `convert_any_p3_sdr` | Display P3 sRGB-TRC RGB8 | CMS path (gamut only, no tone map) |
| 8 | `streaming_pq_tonemap` | PQ f32 rows | StreamingTonemapper produces valid sRGB8 |

### Synthetic test data

No real HDR images needed. Generate gradient ramps:

```rust
fn pq_gradient(width: u32, height: u32) -> Vec<f32> {
    // Linear-light values 0..10000 nits, encoded as PQ
    // PQ OETF: signal = ((c1 + c2 * Y^m1) / (1 + c3 * Y^m1))^m2
    // where Y = nits / 10000
}
```

Values should span 0-1000 nits (typical HDR content) so tone mapping
produces visible midtone detail (not just clipped white).

## Where this lives

- Bridge functions (`cicp_to_gamut`, `pixel_buffer_to_raw_image`): `zenpixels-convert/src/hdr.rs` (already exists, has `HdrMetadata`)
- `convert_any_to_srgb` helper: `zenpixels-convert/src/output.rs` or a new `src/tonemap.rs`
- Tests: `zenpixels-convert/tests/tonemap_integration.rs`
- Feature: `tonemap` = `["dep:ultrahdr-core", "ultrahdr-core/transfer"]`

## Dependency

```toml
[dependencies]
ultrahdr-core = { version = "0.4.1", optional = true, default-features = false, features = ["transfer"] }

[features]
tonemap = ["dep:ultrahdr-core"]
```

The `transfer` feature on ultrahdr-core enables the tone mapping and streaming
tonemap modules (`color::tonemap`, `color::streaming_tonemap`). Without it,
only gain map math and metadata are available.
