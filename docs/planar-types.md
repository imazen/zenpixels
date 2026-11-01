# Planar types (2026-11-01)

One set of planar types for every zen crate. They live in `zenpixels::plane`
so codecs, encoders, zenyuv and metrics can use them without depending on
zencodec. zencodec keeps what is frame-specific (crop, timing, color claims,
configuration epoch, owner and mapping traits) and builds on these.
The inventory that motivated this is `docs/planar-inventory.md`.

## Decisions

| Question | Decision |
|---|---|
| Single-plane type | One generic `Plane<C>` over its owner, not one type per owner or an enum of word vectors |
| Encoding scope | Per plane: alpha and extra planes may differ in depth or type from color |
| Float meaning | `stored = nominal * scale + offset`, declared per plane |
| Where the set layout lives | zenpixels (`PlanarLayout`) |
| Semi-planar (NV12, P010, NV21) | One plane holding two components; one role per plane (`CbCr`, `CrCb`) |
| Color in the layout | No: matrix, range, primaries, transfer and ICC stay with the enclosing descriptor |

## Layers

1. **Storage geometry: `Plane<C>`.** A strided 2-D array of words (`u8`, `u16`
   or `f32`; sealed `PlaneWord`). `width` and `stride` count words, so a stride
   always holds whole words and typed rows never need a checked cast. Rows are
   contiguous `&[T]`; the last row needs no padding after it.
   - `PlaneRef<'a, T>` = `Plane<&'a [T]>`, `PlaneMut<'a, T>` = `Plane<&'a mut [T]>`,
     `PlaneBuf<T>` = `Plane<Vec<T>>`. Other owners (zencodec's accounted shared
     storage) implement `PlaneStorage` / `PlaneStorageMut` and become
     `Plane<TheirOwner>`.
   - `new` adopts storage without copying (a codec's `Vec`, a decoder pool's
     slice). `PlaneBuf::alloc` allocates fallibly, starts every row on 64 bytes
     and adds 64 bytes to strides on 4096 multiples
     (`benchmarks/row_alignment_2026-10-31.md`, `benchmarks/stride_aliasing_2026-11-01.md`).
   - `row`, `rows`, `row_mut`, `rows_mut`, `sub`, `sub_mut`, `view`, `view_mut`,
     `strided_words`, and `split_at_row` on `PlaneMut` for strips and threads.
   - `AnyPlane<'a>` carries borrowed planes of different word types through
     dyn interfaces (zencodec's mapping traits).
2. **Per-plane meaning: `PlaneSpec { role, encoding }`.**
   - `PlaneRole`: `Luma` (also gray), `Cb`, `Cr`, `CbCr`, `CrCb`, `Red`, `Green`,
     `Blue`, `Alpha`; `components()` is 2 for the interleaved chroma roles.
   - `PlaneEncoding::Code(SampleEncoding)`: integer codes in `u8`/`u16` words,
     with code depth and bit placement (P010 is `U16, 10 bits, shift 6`).
     Range (limited/full) comes from the enclosing CICP.
   - `PlaneEncoding::Float(FloatEncoding)`: `f32` words, `stored = nominal *
     scale + offset`. Nominal Y/R/G/B/A run 0..1, Cb/Cr -0.5..0.5. zenjpeg's
     centered planes are scale 255, offset -128 (Y) and 0 (Cb, Cr). A decoder
     that keeps limited-range levels in floats states them here, so float
     planes need no separate range.
3. **The picture: `PlanarLayout`.** Ordered `PlaneSpec`s, `ChromaSubsampling`,
   `ChromaSiting`. Construction refuses duplicate components, mixed YCbCr/RGB,
   missing Luma (with chroma) or R/G/B, a lone Cb or Cr, and subsampling that
   disagrees with the planes (chroma needs non-`Mono`, luma-only needs `Mono`,
   planar RGB needs `Yuv444`). `plane_size` gives each plane's positions for a
   picture size (chroma rounded up); `check` verifies supplied planes' word
   types and sizes.
4. **Ownership** stays with whoever allocates: codecs hand over `Vec`s
   (`PlaneBuf::new`, no copy), zencodec-media keeps accounted storage as
   `Plane<SharedStorage<T>>`, and decoders that map a pool return `PlaneRef`s.

## Migration

| Today | Becomes |
|---|---|
| zenpixels `CodePlane` (code reader, stepped interleaved views) | Reads codes from a `PlaneRef<u8/u16>` plus `SampleEncoding`; one component of a two-component plane replaces the stepped constructors |
| zencodec `PlaneView`, `PlaneSamples` | `AnyPlane` / `PlaneRef` |
| zencodec `FrameInterpretation { subsampling, siting, encoding, roles }` | `{ layout: PlanarLayout, .. }`: per-plane encoding settles #190's frame-wide question |
| zencodec `PlaneRole`, `Subsampling`, `ChromaSiting` | zenpixels `PlaneRole`; the chroma enums already moved (zencodec #191) |
| zencodec-media `OwnedPlane`, `PlaneData`, `OwnedPlanes` | `PlaneBuf<T>` / `Plane<SharedStorage<T>>` in a word-typed enum, plus `PlanarLayout` |
| zencodec `MappedPlanes` and `MappedImage` (two mapping traits) | One mapping that yields `AnyPlane`s; separate zencodec change |
| zenmetrics-abstractions local `PlaneRole`, `PlaneSpec`, `PlaneLayout` (PR #104) | zenpixels types, once zenpixels 0.3.1 is on crates.io |
| Codec boundary types (heic `DecodedFrame`, zenwebp `YuvPlanes`, zenavif `DecodedYuv`, zenjpeg `YCbCrPlanes`/`DecodedYCbCr`) | Emit or accept `PlaneBuf`/`PlaneRef` + `PlanarLayout` at their zencodec boundary; internal padded planes stay private |

## Not covered

- Packed subsampled YUV (YUYV, UYVY, Y210) and packed 4:4:4 with alpha (AYUV,
  Y410): not planes; convert at the boundary if a source produces them.
- `f16` words: add a `PlaneWord` impl and an `AnyPlane` variant when a crate
  exchanges half-float planes (JXL, EXR).
- CMYK or other non-YCbCr/RGB planes, depth, and JXL extra channels: zencodec
  carries them as auxiliary planes with their own meaning; add roles here only
  when a shared consumer needs them.
- `split_at_row` has no consumer yet; it is the primitive for filling disjoint
  strips from threads and was kept on review.

## Interoperability with other crates (reviewed 2026-11-01)

Checked against mediaframe 0.12.0, pixon 0.7.2, yuvutils-rs 0.8.3 and pix 1.0.1
(sources read with `cargo read`). None needs a change to these types now. Each
adapter below can be added later without a semver break, because the types it
needs are already open in the right places: private fields, sealed `PlaneWord`,
`#[non_exhaustive]` on `AnyPlane`, `PlaneRole`, `PlaneEncoding`,
`ChromaSubsampling`, `ChromaSiting` and every error enum. Our own matches refuse
unknown variants (`YcbcrToRgbF32` returns `UnsupportedChroma`), so a later
variant is refused, not misread.

| Crate | Its model | Maps to | Notes |
|---|---|---|---|
| mediaframe (MIT/Apache) | One borrowed type per FFmpeg format (`Yuv420pFrame<'a>`, `Yuv420pFrame16<'a, BITS, BE>`, `PnFrame` for P010, NV12/21/16/24/42, YUVA, GBR(A) incl. float, gray, Bayer, packed YUV); `u32` dims, strides in samples; generic `VideoFrame<P, B>` with byte strides; H.273 `color::Info` | `PlaneRef` (slice from the first sample, stride in words) + `PlanarLayout`; `BITS` and P010's high-bit placement are `SampleEncoding` bit depth and shift | Big-endian (`BE = true`) frames need a byte-swapping copy: planes are native-endian by contract. 4:4:0/4:1:1/4:1:0 need new `ChromaSubsampling` variants; Top/BottomLeft/Bottom siting needs three `ChromaSiting` variants (H.273 values 3-5). Packed YUV, V210, Bayer and PAL8 stay adapter conversions |
| pixon (GPL-3.0-or-later) | Conversion kernels generic over mediaframe frame traits | Through mediaframe frames | GPL: never a dependency of zen crates. An application that is GPL itself can feed pixon from our planes via mediaframe |
| yuvutils-rs (BSD-3/Apache) | `YuvPlanarImage<'a, T>`, `YuvBiPlanarImage`, `YuvGrayImage` with public slice + stride (in components) fields, `u32` dims; `*Mut` variants over `BufferStoreMut` (borrowed or owned `Vec`); bit depth, endianness and MSB/LSB packing per call | `PlaneRef`/`PlaneMut` + `SampleEncoding` (depth, shift) | Same length rule as ours (the last row needs no padding). Taking a `PlaneBuf`'s `Vec` as `Owned` works only when its start offset is 0, which `PlaneBuf::alloc` does not promise |
| pix (MIT/Apache, 1.x) | Packed `Raster<P>` (tight `Box<[P]>`, sealed `Pixel` with type-level channel, alpha mode and gamma) | `PixelSlice` / `PixelDescriptor`, not planes | Borrowed only (`as_u8_slice`); its `u16` rasters cannot become a `Vec<u8>` without a copy |

Additions that would serve these adapters, each additive:

- `Plane<&'a [T]>::into_strided_words(self) -> &'a [T]` (crate-private today) and a
  mutable counterpart, so an adapter can build a borrowed external frame with the
  plane's own lifetime.
- `f16` and `i16` words: a `PlaneWord` impl and an `AnyPlane` variant each.
- The variants listed in the table, when a source produces them.

Where adapters live: mediaframe and yuvutils-rs are pre-1.0, so a public `From`
impl naming their types would tie zenpixels' version to theirs. Put those
adapters in a separate crate or in the codec adapter that needs them; a feature
in zenpixels fits only 1.x crates such as pix (as `imgref` does today).
