# zenpixels

Pixel types for Rust image pipelines. Keep samples, dimensions, stride, bit
packing, and color meaning together — `no_std + alloc`, `#![forbid(unsafe_code)]`.

This is the **0.5.0 greenfield candidate**. [docs.rs](https://docs.rs/zenpixels)
describes the latest published crate.

## Quick start

`PixelBuf` owns alignment-exact storage. `PixelView<'a>` / `PixelViewMut<'a>`
borrow an image or rectangular subimage. `PixelFormat` pairs a `SampleLayout` with
an algebraic `ColorSpace`.

```rust
use zenpixels::{PixelFormat, PixelBuf};

let decoded = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
let image = PixelBuf::from_typed(decoded, 2, 2, PixelFormat::RGB8_SRGB)?;
let right_column = image.crop_view(1, 0, 1, 2)?;

assert_eq!(right_column.width(), 1);
assert_eq!(right_column.height(), 2);
assert_eq!(right_column.stride(), image.stride()); // Parent stride: 6 bytes.
assert_eq!(right_column.row(0), &[0, 255, 0]);     // Only the cropped pixel.
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Two crates

```toml
zenpixels = "0.5.0"
zenpixels-convert = "0.5.0"
```

## Features

| Feature | Effect |
|---|---|
| `std` | Default; omit for `no_std + alloc` |
| `icc` | Default; paired-fingerprint ICC recognition with intent-scoped substitution |
| `rgb` | `PixelType` trait & typed `rgb` crate interop |
| `imgref` | `imgref` adapters; enables `rgb` |
| `image` | `image` 0.25 interop with CICP color-space propagation; requires Rust 1.88 |

## MSRV

`zenpixels` requires Rust 1.85 (`image` feature: 1.88). `zenpixels-convert`
requires Rust 1.89. Both crates use edition 2024.

## License

Apache-2.0 OR MIT
