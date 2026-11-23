# Images and views in Rust

An image is samples, a shape, and a description. **zenpixels** keeps them
together so you can pass images, cropped subimages, and color metadata through
your pipeline without juggling loose byte slices and parallel arguments.

This guide follows the **0.4.0 candidate**. Start here to own, borrow, and
convert pixels, then jump straight to what you need:

- **[Transfer functions & live Rust codegen](transfers.html)** — interactive
  EOTF/OETF and bit-allocation charts for sRGB, BT.709, Gamma 2.2, PQ, and HLG,
  plus a live `zenpixels-convert` code generator.
- **[Primaries, HDR headroom, CLL & CICP vs ICC](color-hdr.html)** — CIE `xy`
  and `u'v'` gamut renderings, `DiffuseWhite` / `MaxCLL` / `ST.2086` / BT.2446A
  roll-off charts, signaling range, and CICP vs ICC comparison.
- **[Conversion kernel browser](kernels.html)** — explore every real step and
  SIMD kernel extracted from `zenpixels-convert` and see what each
  `const Accelerators` preset links.
- **[Balance speed and light compilation](speed-and-size.md)** — keep libraries
  and Wasm small with the universal fast-FMA scalar core, then turn on SIMD
  cleanly with `accel` and `const Accelerators`.
- **[Plain-English ELI5 mental model](eli5.md)** — how `garb`, `linear-srgb`,
  `zenpixels`, and `zenpixels-convert` fit together and the gotchas to avoid.
- **[Interactive byte explorer](index.html)** — inspect individual bytes, row
  strides, and live converter plans in your browser.
- **[Source on GitHub](https://github.com/imazen/zenpixels)**

[Own pixels](#put-decoded-pixels-in-a-buffer) ·
[Borrow a subimage](#borrow-the-part-you-need) ·
[Describe and convert](#the-format-is-the-layout-the-descriptor-adds-meaning) ·
[Speed & light builds](#keep-compilation-light-turn-on-simd-when-you-want-it)

## Three types get you started

| What you need | Use |
|---|---|
| Own an image | `PixelBuffer` |
| Borrow an image or a rectangular subimage | `PixelSlice` |
| Describe the samples and their color meaning | `PixelDescriptor` |

A `PixelSliceMut` is the writable view. The buffer owns the allocation; views
borrow it. Making a view or a cropped view does not copy the pixel data.

## Put decoded pixels in a buffer

If a decoder gives you packed RGB bytes, give the vector and its description
to the buffer:

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};

let decoded = vec![255, 0, 0, 0, 255, 0]; // Red, green.
let image = PixelBuffer::from_vec(
    decoded, 2, 1, PixelDescriptor::RGB8_SRGB,
)?;

assert_eq!(image.width(), 2);
assert_eq!(image.height(), 1);
# assert_eq!(image.as_slice().row(0), &[255, 0, 0, 0, 255, 0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Here the descriptor says **RGB, eight-bit samples, sRGB color**. It declares
what those existing bytes mean. It does not convert them.

For a fresh zero-filled image, use `PixelBuffer::try_new(width, height,
descriptor)`. Ordinary rows are packed. Use `try_new_aligned` when you
specifically need a 64-byte-aligned owner and padded rows.

## Borrow the part you need

A slice borrows an image or a rectangular subimage. Cropping changes its starting
position, width and row count, while retaining the parent stride:

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};

let image = PixelBuffer::try_new(
    8, 6, PixelDescriptor::RGB8_SRGB,
)?;
let crop = image.crop_view(2, 1, 3, 2); // x, y, width, height

# assert_eq!((crop.width(), crop.rows()), (3, 2));
# assert_eq!(crop.stride(), image.stride());
assert_eq!(crop.stride(), 24);    // Parent stride.
assert_eq!(crop.row(0).len(), 9); // Three RGB pixels.
# assert_eq!(crop.row(0).as_ptr(), image.as_slice().row(1)[6..].as_ptr());

for y in 0..crop.rows() {
    let row = crop.row(y);
    assert_eq!(row.len(), 3 * 3);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The rows belong to the original image. No repacking is necessary. Take crops
directly from the buffer (`image.crop_view(...)`) or chain through a borrowed
view (`image.as_slice().crop_view(...)` / `sub_rows(...)` return views tied to
the underlying buffer's `'a` lifetime). Use `sub_rows(y, count)` for a
full-width strip, or `try_to_tight_buffer()` when you need an independent
packed copy. Crops and strips must be in bounds.

## Stride is the distance between row starts

**Width and height count pixels. Stride counts bytes.** `row(y)` excludes the
gaps between rows. A cropped view can have a large stride even when it has
only a few pixels.

You can also borrow someone else's strided storage directly:

```rust
use zenpixels::{PixelDescriptor, PixelSlice};

let bytes = [
    255, 0, 0, 0, 255, 0, 99, 99, // First row, then two padding bytes.
    0, 0, 255, 255, 255, 255,     // Last row needs no trailing padding.
];
let view = PixelSlice::new(&bytes, 2, 2, 8, PixelDescriptor::RGB8_SRGB)?;

assert_eq!(view.stride(), 8); // Not a multiple of the three-byte pixel.
assert_eq!(view.row(1), &[0, 0, 255, 255, 255, 255]);
assert!(view.as_contiguous_bytes().is_none());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The first sample and row starts must have the required sample alignment:
one byte for U8, two for U16, four for F32. Padding does **not** have to be
a whole pixel. `as_strided_bytes()` includes gaps; it is not a packed image.

## Change pixels through a writable view

```rust
use zenpixels::{PixelBuffer, PixelDescriptor};

let mut image = PixelBuffer::try_new(2, 1, PixelDescriptor::RGB8_SRGB)?;
image.as_slice_mut().row_mut(0).copy_from_slice(&[255, 0, 0, 0, 255, 0]);
assert_eq!(image.as_slice().row(0)[..3], [255, 0, 0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Rust's borrowing rules protect the owner while a view is in use. Mutable access
also clears cached `ExactDepth` and `FloatFacts` evidence; the library cannot
assume your edits kept the old sample bounds.

## The format is the layout; the descriptor adds meaning

`PixelFormat::Rgb8` says how the samples are stored. A `PixelDescriptor` adds
transfer function, primaries, alpha interpretation and signal range.

For example, `RGB8_SRGB` describes ordinary web RGB. `RGBF32_LINEAR` describes
linear-light F32 RGB. Changing a descriptor changes the declaration, not the
samples. Use **zenpixels-convert** when the sample values must change:

```rust
use zenpixels_convert::{PixelBuffer, PixelDescriptor, PixelSliceConvertExt};

let image = PixelBuffer::from_vec(vec![128, 128, 128], 1, 1, PixelDescriptor::RGB8_SRGB)?;
let linear = image.as_slice().convert_to(PixelDescriptor::RGBF32_LINEAR)?;

assert_eq!(linear.descriptor(), PixelDescriptor::RGBF32_LINEAR);
assert_eq!(linear.as_slice().row(0).len(), 3 * 4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For repeated images, crops, or gigapixel-safe `BandedPixelBuffer`s, build a
reusable `ConvertPlan` once, prepare a `RowConverter` for your max row width,
and pass views, buffers, or bands uniformly through `converter.convert(&src, &mut dst)`:

```rust
use zenpixels_convert::{
    ConversionSpec, ConvertOptions, ConvertPlan, PixelBuffer, PixelDescriptor,
};

let src = PixelBuffer::from_vec(
    vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128],
    2, 2, PixelDescriptor::RGB8_SRGB,
)?;
let crop = src.crop_view(1, 0, 1, 2);
let mut dst = PixelBuffer::try_new(1, 2, PixelDescriptor::RGBF32_LINEAR)?;

let plan = ConvertPlan::build(
    crop.encoding(),
    dst.encoding(),
    &ConversionSpec::new(ConvertOptions::retain_channels()),
)?;
let mut conv = plan.prepare(crop.width())?;
conv.convert(&crop, &mut dst)?;

assert_eq!(dst.as_slice().row(0).len(), 3 * 4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A few rules save hours on your first real pipeline:

- **Wide-gamut Oklab works directly:** `PixelDescriptor::OKLABF32` is
  CIE XYZ D65-anchored, so roundtripping Display P3 or BT.2020 through
  `OKLABF32` uses that gamut's own `RGB ↔ LMS` matrix directly — bit-exact
  with zero BT.709 clipping and no extra `.with_primaries(...)` call needed.
- **Explicit HDR & ICC:** Custom ICC transforms need an explicitly passed CMS
  (`spec.with_cms(&MoxCms)`). HDR→SDR tone mapping requires explicit luminance
  anchors (`zenpixels_convert::hdr::{Nits, DiffuseWhite, HdrPeak, HlgRendering}`)
  plus `features = ["bt2446a"]`. See
  [Primaries, HDR & CICP/ICC](color-hdr.html), the
  [conversion guide](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md),
  and [plain-English ELI5](eli5.md) for every detail.

## Keep compilation light, turn on SIMD when you want it

By default, `zenpixels-convert` executes every valid native conversion through
a tiny, zero-scratch **universal fast-FMA scalar core** that is bit-for-bit
identical to the x86-64/AArch64 SIMD kernels. Library crates and tiny Wasm
modules compile fast and never link unused SIMD tables.

When you are building an application, CLI, server, or benchmark, turn on SIMD
at the exact granularity you want — either process-wide at startup or via
compile-time `const Accelerators` helpers:

```rust
use zenpixels_convert::{Accelerators, ConversionSpec, ConvertOptions, accel};

// Option A: one line in main() for full SIMD everywhere (including inside codecs)
accel::enable_all();

// Option B: compile-time const helper — only links the SIMD families you name
const WEB_PLUS_OKLAB: Accelerators = Accelerators::WEB_SDR.with_oklab();
let spec = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&WEB_PLUS_OKLAB);

// Option C: hermetic scalar-only spec for 0-bit parity tests (even after enable_all)
let scalar = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&Accelerators::SCALAR);
# assert!(accel::enabled().is_all());
# let _ = (spec, scalar);
```

Read **[Balancing speed and light compilation](speed-and-size.md)** for the full
decision table, live-measured stripped binary sizes (x86-64 and Wasm), Cargo
feature budgets, and `FloatFacts` fast paths.

## Use a concrete pixel type when it helps

With the `rgb` feature, a typed buffer can carry `rgb::Rgb<u8>` rather than
leaving its pixel layout dynamic. Add `rgb` as a dependency too:

```toml
zenpixels = { version = "0.4.0", features = ["rgb"] }
rgb = "0.8.53"
```

```rust
use rgb::Rgb;
use zenpixels::{PixelBuffer, PixelDescriptor};

let pixels = vec![Rgb::new(255u8, 0, 0), Rgb::new(0, 255, 0)];
let typed = PixelBuffer::<Rgb<u8>>::from_pixels(pixels, 2, 1)?
    .with_descriptor(PixelDescriptor::RGB8_SRGB);
let dynamic = typed.erase();

assert_eq!(dynamic.as_slice().row(0), &[255, 0, 0, 0, 255, 0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The concrete type establishes the layout. It does not establish a color space;
the example explicitly declares sRGB. `erase()` changes the Rust type while
keeping the storage and metadata. You can use the dynamic types throughout
your program if that is simpler.

## Native codes and wide samples are separate choices

An ordinary full-range U16 image uses **65535 for white**. A codec's ten-bit
code in a U16 word has a different meaning. Keep that interpretation with
the native plane until you explicitly convert it:

```rust
use zenpixels::{ChannelType, plane::{AnyPlane, PlaneRef}, sample::{CodePlane, SampleEncoding}};

let words = [1023u16, 512];
let plane = AnyPlane::U16(PlaneRef::new(&words[..], 2, 1, 2)?);
let ten_bit = SampleEncoding::new(ChannelType::U16, 10, 0)?;
let codes = CodePlane::new(plane, ten_bit)?;
assert_eq!(codes.code(0, 0), Some(1023)); // A code, not normalized U16 white.
# Ok::<(), Box<dyn std::error::Error>>(())
```

`codes::to_canonical` in the converter performs the explicit widening. Existing
component readers also handle interleaved CbCr, RGB10 + alpha1, and RGB565
bitfields. [Try the storage examples](index.html#h-storage).

Signed and wide types such as I16, U32 and F64 already fit
`PixelFormat::Generic`, constructed through `PixelFormat::from_parts`.
The existing owners and views preserve their bits; unsupported numerical
conversion refuses. You do not need another buffer family to carry them.

## Carry color and ownership together

`ColorContext` holds the current pixels' single `ColorProfile` (`Cicp` or `Icc`)
and luminance declarations; `PixelEncoding` borrows both a `PixelDescriptor` and
an optional `ColorContext` without cloning `Arc`s. `ColorOrigin` retains what the
original file said, including dual ICC + CICP container claims. Resolve source
claims at the codec boundary: an attached ICC remains authoritative even if a
CICP companion disagrees or the ICC is unusable, and an unusable ICC is a reason
to refuse an unsupported operation rather than substitute CICP silently. After a
conversion, original metadata must not override the current samples.

`into_parts()` moves storage, offset, dimensions, stride, descriptor and context
together. Use `PixelBuffer::try_from_parts` for checked reconstruction. A receiver
that accepts stride needs no compaction. If it needs packed rows, call
`into_contiguous()` first; the leading offset may still be nonzero.

For current-color conversion and matching encoder metadata, use `OutputPlan`
or `EncodeReady::from_owned` from the converter. [Output contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md).

## Which path should I take?

| Situation | Start here |
|---|---|
| Own decoded packed bytes | `PixelBuffer::from_vec` |
| Own padded storage or an allocation with an offset | `PixelBuffer::try_from_parts` |
| Borrow external bytes | `PixelSlice::new` |
| Borrow a subimage or strip | `buffer.crop_view` / `sub_rows` |
| Need an independent packed copy | `try_to_tight_buffer` |
| Need a 64-byte-aligned owner | `PixelBuffer::<(), AlignedSamples<u8>>::try_new_aligned` |
| Change sample values or color space | `zenpixels-convert` (`ConvertPlan` / `convert_to`) |
| Balance SIMD speed & light binary size | [Speed & light builds](speed-and-size.md) (`accel` / `const Accelerators`) |
| Plain-English concepts & gotchas | [ELI5 mental model](eli5.md) |
| Preserve native code interpretation | `SampleEncoding` with `CodePlane` |

Use fallible constructors when you need to handle image-allocation failure.
The snippets use `?` to propagate errors from a `Result`-returning function.
`new` is the panicking convenience form. Not every control, cache or dependency
allocation is fallible; the [allocation audit](https://github.com/imazen/zenpixels/blob/main/docs/allocation-audit.md)
records the boundaries.

[Explore one pixel](index.html) ·
[Transfer functions](transfers.html) ·
[Primaries, HDR & CICP/ICC](color-hdr.html) ·
[Kernel browser](kernels.html) ·
[Speed & light builds](speed-and-size.md) ·
[ELI5 mental model](eli5.md) ·
[Reference contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md) ·
[Core API](https://docs.rs/zenpixels)
