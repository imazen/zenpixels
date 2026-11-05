# Why `Color` looks the way it does

`Color` is one pixel of a descriptor's format, inline and `Copy`. `AnyColor` also holds wider pixels. Measured 2026-11-04 on Zen 4; not measured on Zen 5.

## What it has to serve

| Use | Needs |
|---|---|
| Matte, in `AlphaPolicy::CompositeOnto` | `Copy`, because `ConvertOptions` is `Copy` with `const` builders; any space and depth |
| Canvas fill, `zenlayout::CanvasColor`, `zenpipe::CanvasFill::Solid` | `Copy` inside layout types; today loses its color space |
| CSS parsing | Out-of-gamut `f32` |
| `get_pixel` | Every format, with no allocation for common ones |
| `bKGD`, GIF background | The image's own format and depth |
| Palettes, gradients | Not `Color`: one descriptor over many samples is a `PixelBuffer` row |

Size hardly matters for singletons. It matters in arrays, where `Color` is the wrong type anyway.

## Layouts considered

| Layout | Bytes | Holds | `Copy` | Against |
|---|---:|---|---|---|
| 4 B samples, packed meaning | 8 | 14 of 26 named formats, no RGB in 16 bits or `f32` | Yes | Every pipeline's working format, and every wide CSS color, doesn't fit |
| `Box` always | 8 | Everything | No | 5.74 ns per read against 0.54 inline, fallible, no `const`, no `Copy` |
| Inline up to 8 B or box | 16 | Everything | No | RGB in `f32` allocates; `ConvertOptions` loses `Copy` |
| 4 × `f16` plus meaning | 12 | Colors, not formats | Yes | 58,367 of 65,536 `u16` values don't round trip |
| 4 × `f32` plus meaning, format forgotten | 20 | Colors at up to 32-bit float | Yes | No byte access, no wider samples. `u8` and `u16` round trip exactly |
| 16 B plus descriptor (current) | 24 | Every named format | Yes | Largest of the `Copy` options |
| 16 B plus 3 B packed descriptor | 20 | Every named format | Yes | Saves 4 B for a pack and unpack step |

`PixelDescriptor` is 6 bytes. In safe Rust, an enum holding a `Box` and inline data needs a separate tag, so inline-or-box can't reach 8.
