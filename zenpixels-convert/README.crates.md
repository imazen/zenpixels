# zenpixels-convert

Transfer-function-aware pixel conversion, HDR tone mapping, gamut compression,
ICC/CMS routing, gain-map reconstruction, EXIF orientation, and codec format
negotiation for `zenpixels`. Re-exports `zenpixels::*` with zero module name
collisions.

This is the **0.5.0 greenfield candidate**.

## Quick start

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PeakSource, PixelBuf, convert_buf};

let src = PixelBuf::from_vec(
    vec![255u8, 0, 0, 0, 255, 0], 2, 1, ImageSpec::rgb8_srgb(),
)?;
let linear = convert_buf(
    &src.as_view(), ImageSpec::rgbf32_linear(), &ConvertPolicy::permissive(), PeakSource::Measured,
)?;
assert_eq!(linear.spec(), &ImageSpec::rgbf32_linear());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `fast-transpose` | Default; runtime-dispatched SIMD orientation kernels |
| `oklab` | Default; CIE XYZ D65 Oklab conversions and OKLch soft gamut compression |
| `hdr-transfers` | Default; PQ (ST 2084) and HLG (ARIB STD-B67 + BT.2100-3 display OOTF) |
| `perceptual` | Default-off; Oklch, CIE Lab/Lch, Jzazbz/Jzczhz, HSL, HSV |
| `bt2446a` | Built-in ITU-R BT.2446 Method A HDR→SDR tone mapper |
| `icc-db` | Committed LZ4-compressed CICP→ICC profile bundle |
| `cms-moxcms`, `cms-lcms` | Pure-Rust `MoxCms` and `Lcms` ICC CMS backends |
| `fast` | Default-off (binaries only); defaults plans to `Accelerators::ALL` |
| `avx512` | Opt-in AVX-512F F16C & vector tiers |
| `precise-portable` | Portable bit-exact FMA on strict WASM engines |
| `rgb`, `imgref` | Core `PixelType` and `imgref` interop |

## MSRV

Rust 1.89, edition 2024. `no_std + alloc`, `#![forbid(unsafe_code)]`.

## License

Apache-2.0 OR MIT
