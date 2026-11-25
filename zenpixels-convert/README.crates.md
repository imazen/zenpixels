# zenpixels-convert

Transfer-function-aware pixel conversion, HDR tone mapping, gamut compression,
ICC/CMS routing, gain-map reconstruction, EXIF orientation, planar reconstruction,
and codec format negotiation for `zenpixels`. Re-exports `zenpixels::*` with zero
module name collisions.

This is the **0.5.0 greenfield candidate**.

## Quick start

```rust
use zenpixels_convert::{ConvertPolicy, ImageSpec, PixelBuf, convert_to_buf, specs};

let src = PixelBuf::from_typed(
    vec![255u8, 0, 0, 0, 255, 0], 2, 1, ImageSpec::RGB8_SRGB,
)?;
let linear = convert_to_buf(
    &src.as_view(), &specs::RGBF32_LINEAR, &ConvertPolicy::lossless(),
)?;
assert_eq!(linear.spec(), &specs::RGBF32_LINEAR);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Features

| Feature | Effect |
|---|---|
| `std` | Default; disable default features for `no_std + alloc` |
| `fast-transpose` | Default; runtime-dispatched SIMD orientation kernels |
| `oklab` | Default; CIE XYZ D65 Oklab conversions and OKLch soft gamut compression |
| `hdr-transfers` | Default; PQ (ST 2084) and HLG (ARIB STD-B67 + BT.2100-3 display OOTF) |
| `gainmap` | Default; ISO 21496-1 / UltraHDR gain-map extraction & reconstruction |
| `perceptual` | Default-off; Oklch, CIE Lab/Lch, Jzazbz/Jzczhz, HSL, HSV |
| `bt2446a` | Built-in ITU-R BT.2446 Method A HDR→SDR tone mapper |
| `icc-db` | Committed LZ4-compressed CICP→ICC profile bundle |
| `cms-moxcms`, `cms-lcms` | Pure-Rust `MoxCms` and `Lcms` ICC CMS backends |
| `fast` | Default-off (binaries only); defaults plans to `Accelerators::ALL` |
| `avx512` | Opt-in AVX-512F F16C & vector tiers |
| `precise-portable` | Portable bit-exact FMA on strict WASM engines |
| `rgb`, `imgref` | Core `PixelType` and `imgref` interop |

## MSRV

Rust 1.89, edition 2024. `#![forbid(unsafe_code)]`; supports `no_std + alloc`
(disable default `std` feature).

## License

Apache-2.0 OR MIT
