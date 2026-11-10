# Images and views in Rust

An image is samples, a shape, and a description. zenpixels keeps them together
so you can pass an image around without also passing a handful of unrelated
arguments.

This guide follows the 0.3.1 candidate. [Explore the bytes](https://imazen.github.io/zenpixels/)
or [read the source](https://github.com/imazen/zenpixels) alongside it.

[Own pixels](#put-decoded-pixels-in-a-buffer) ·
[Borrow a subimage](#borrow-the-part-you-need) ·
[Describe and convert](#the-format-is-the-layout-the-descriptor-adds-meaning)

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
let whole = image.as_slice();
let crop = whole.crop_view(2, 1, 3, 2); // x, y, width, height

# assert_eq!((crop.width(), crop.rows()), (3, 2));
# assert_eq!(crop.stride(), whole.stride());
assert_eq!(crop.stride(), 24);    // Parent stride.
assert_eq!(crop.row(0).len(), 9); // Three RGB pixels.
# assert_eq!(crop.row(0).as_ptr(), whole.row(1)[6..].as_ptr());

for y in 0..crop.rows() {
    let row = crop.row(y);
    assert_eq!(row.len(), 3 * 3);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The rows belong to the original image. No repacking is necessary. Use
`sub_rows(y, count)` for a full-width strip, or `try_to_tight_buffer()` when
you need an independent packed copy. Crops and strips must be in bounds.

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
also clears cached exact-depth evidence; the library cannot assume your edits
kept the old sample grid.

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

This example changes known sRGB into known linear RGB. Custom ICC transforms
need a selected CMS backend. Alpha removal, quantization, gamut clipping and
HDR rendering have their own policies; the
[conversion guide](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md)
shows those entry points.

## Use a concrete pixel type when it helps

With the `rgb` feature, a typed buffer can carry `rgb::Rgb<u8>` rather than
leaving its pixel layout dynamic. Add `rgb` as a dependency too:

```toml
zenpixels = { version = "0.3.1", features = ["rgb"] }
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
bitfields. [Try the storage examples](https://imazen.github.io/zenpixels/#h-storage).

Signed and wide types such as I16, U32 and F64 already fit
`PixelFormat::Generic`, constructed through `PixelFormat::from_parts`.
The existing owners and views preserve their bits; unsupported numerical
conversion refuses. You do not need another buffer family to carry them.

## Carry color and ownership together

`ColorContext` holds the current pixels' ICC/CICP and luminance declarations.
`ColorOrigin` retains what the original file said. After a conversion, original
metadata must not override the current samples. An attached ICC remains
authoritative even if a CICP companion disagrees or the ICC is unusable.
Resolve source claims at the codec boundary: output planning requires a single
current ICC or CICP declaration and refuses a context carrying both. An unusable
ICC is a reason to refuse an unsupported operation, not to substitute CICP silently.

`into_parts()` moves storage, offset, dimensions, stride, descriptor and context
together. Use `PixelBuffer::try_from_parts` for checked reconstruction. A receiver
that accepts stride needs no compaction. If it needs packed rows, call
`into_contiguous()` first; the leading offset may still be nonzero.

For current-color conversion and matching encoder metadata, use `OutputPlan`
or `EncodeReady::finalize` from the converter. [Output contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md).

## Which path should I take?

| Situation | Start here |
|---|---|
| Own decoded packed bytes | `PixelBuffer::from_vec` |
| Own padded storage or an allocation with an offset | `PixelBuffer::try_from_parts` |
| Borrow external bytes | `PixelSlice::new` |
| Borrow a subimage or strip | `crop_view` / `sub_rows` |
| Need an independent packed copy | `try_to_tight_buffer` |
| Need a 64-byte-aligned owner | `PixelBuffer::<(), AlignedSamples<u8>>::try_new_aligned` |
| Change sample values or color space | `zenpixels-convert` |
| Preserve native code interpretation | `SampleEncoding` with `CodePlane` |

Use fallible constructors when you need to handle image-allocation failure.
The snippets use `?` to propagate errors from a `Result`-returning function.
`new` is the panicking convenience form. Not every control, cache or dependency
allocation is fallible; the [allocation audit](https://github.com/imazen/zenpixels/blob/main/docs/allocation-audit.md)
records the boundaries.

[Explore one pixel](https://imazen.github.io/zenpixels/) ·
[Reference contracts](https://github.com/imazen/zenpixels/blob/main/docs/design-reference.md) ·
[Core API](https://docs.rs/zenpixels)
