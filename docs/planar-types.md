# Planar types (2026-11-01)

One set of planar types for every zen crate. They live in `zenpixels::plane`
so codecs, encoders, zenyuv and metrics can use them without depending on
zencodec. zencodec keeps what is frame-specific (crop, timing, color claims,
configuration epoch, owner and mapping traits) and builds on these.
The inventory that motivated this is `docs/planar-inventory.md`.

Terms: a **sample** is one stored element of a plane (`u8`, `u16` or `f32`),
as in FFmpeg's "stride in samples". A **code** is the integer value a sample
holds after removing padding bits (`SampleEncoding`). A **position** is a pixel
location; a semi-planar chroma plane stores two samples per position.

## Decisions

| Question | Decision |
|---|---|
| Single-plane type | One generic `Plane<C>` over its owner, not one type per owner or an enum of sample vectors |
| Encoding scope | Per plane: alpha and extra planes may differ in depth or type from color |
| Float meaning | `stored = nominal * scale + offset`, declared per plane |
| Where the set layout lives | zenpixels (`PlanarLayout`) |
| Semi-planar (NV12, P010, NV21) | One plane holding two components; one role per plane (`CbCr`, `CrCb`) |
| Color in the layout | No: matrix, range, primaries, transfer and ICC stay with the enclosing descriptor |

## Layers

1. **Storage geometry: `Plane<C>`.** A strided 2-D array of samples (`u8`, `u16`
   or `f32`; sealed `PlaneSample`). `width` and `stride` count samples, so a stride
   always holds whole samples and typed rows never need a checked cast. Rows are
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
     `strided_samples`, and `split_at_row` on `PlaneMut` for strips and threads.
   - `AnyPlane<'a>` carries borrowed planes of different sample types through
     dyn interfaces (zencodec's mapping traits).
2. **Per-plane meaning: `PlaneSpec { role, encoding }`.**
   - `PlaneRole`: `Luma` (also gray), `Cb`, `Cr`, `CbCr`, `CrCb`, `Red`, `Green`,
     `Blue`, `Alpha`; `components()` is 2 for the interleaved chroma roles.
   - `PlaneEncoding::Code(SampleEncoding)`: integer codes in `u8`/`u16` samples,
     with code depth and bit placement (P010 is `U16, 10 bits, shift 6`).
     Range (limited/full) comes from the enclosing CICP.
   - `PlaneEncoding::Float(FloatEncoding)`: `f32` samples, `stored = nominal *
     scale + offset`. Nominal Y/R/G/B/A run 0..1, Cb/Cr -0.5..0.5. zenjpeg's
     centered planes are scale 255, offset -128 (Y) and 0 (Cb, Cr). A decoder
     that keeps limited-range levels in floats states them here, so float
     planes need no separate range.
3. **The picture: `PlanarLayout`.** Ordered `PlaneSpec`s, `ChromaSubsampling`,
   `ChromaSiting`. Construction refuses duplicate components, mixed YCbCr/RGB,
   missing Luma (with chroma) or R/G/B, a lone Cb or Cr, and subsampling that
   disagrees with the planes (chroma needs non-`Mono`, luma-only needs `Mono`,
   planar RGB needs `Yuv444`). `plane_size` gives each plane's positions for a
   picture size (chroma rounded up); `check` verifies supplied planes' sample
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
| zencodec-media `OwnedPlane`, `PlaneData`, `OwnedPlanes` | `PlaneBuf<T>` / `Plane<SharedStorage<T>>` in a sample-typed enum, plus `PlanarLayout` |
| zencodec `MappedPlanes` and `MappedImage` (two mapping traits) | One mapping that yields `AnyPlane`s; separate zencodec change |
| zenmetrics-abstractions local `PlaneRole`, `PlaneSpec`, `PlaneLayout` (PR #104) | zenpixels types, once zenpixels 0.3.1 is on crates.io |
| Codec boundary types (heic `DecodedFrame`, zenwebp `YuvPlanes`, zenavif `DecodedYuv`, zenjpeg `YCbCrPlanes`/`DecodedYCbCr`) | Emit or accept `PlaneBuf`/`PlaneRef` + `PlanarLayout` at their zencodec boundary; internal padded planes stay private |

## Not covered

- Packed subsampled YUV (YUYV, UYVY, Y210) and packed 4:4:4 with alpha (AYUV,
  Y410): not planes; convert at the boundary if a source produces them.
- `f16` samples: add a `PlaneSample` impl and an `AnyPlane` variant when a crate
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
needs are already open in the right places: private fields, sealed `PlaneSample`,
`#[non_exhaustive]` on `AnyPlane`, `PlaneRole`, `PlaneEncoding`,
`ChromaSubsampling`, `ChromaSiting` and every error enum. Our own matches refuse
unknown variants (`YcbcrToRgbF32` returns `UnsupportedChroma`), so a later
variant is refused, not misread.

| Crate | Its model | Maps to | Notes |
|---|---|---|---|
| mediaframe (MIT/Apache) | One borrowed type per FFmpeg format (`Yuv420pFrame<'a>`, `Yuv420pFrame16<'a, BITS, BE>`, `PnFrame` for P010, NV12/21/16/24/42, YUVA, GBR(A) incl. float, gray, Bayer, packed YUV); `u32` dims, strides in samples; generic `VideoFrame<P, B>` with byte strides; H.273 `color::Info` | `PlaneRef` (slice from the first sample, stride in samples) + `PlanarLayout`; `BITS` and P010's high-bit placement are `SampleEncoding` bit depth and shift | Big-endian (`BE = true`) frames need a byte-swapping copy: planes are native-endian by contract. 4:4:0/4:1:1/4:1:0 need new `ChromaSubsampling` variants; Top/BottomLeft/Bottom siting needs three `ChromaSiting` variants (H.273 values 3-5). Packed YUV, V210, Bayer and PAL8 stay adapter conversions |
| pixon (GPL-3.0-or-later) | Conversion kernels generic over mediaframe frame traits | Through mediaframe frames | GPL is acceptable for zen (owner, 2026-11-01): pixon may be a dependency, for example of an adapter crate or a conversion backend. A crate that depends on it is distributed under GPL-3.0-or-later terms |
| yuvutils-rs (BSD-3/Apache) | `YuvPlanarImage<'a, T>`, `YuvBiPlanarImage`, `YuvGrayImage` with public slice + stride (in components) fields, `u32` dims; `*Mut` variants over `BufferStoreMut` (borrowed or owned `Vec`); bit depth, endianness and MSB/LSB packing per call | `PlaneRef`/`PlaneMut` + `SampleEncoding` (depth, shift) | Same length rule as ours (the last row needs no padding). Taking a `PlaneBuf`'s `Vec` as `Owned` works only when its start offset is 0, which `PlaneBuf::alloc` does not promise |
| pix (MIT/Apache, 1.x) | Packed `Raster<P>` (tight `Box<[P]>`, sealed `Pixel` with type-level channel, alpha mode and gamma) | `PixelSlice` / `PixelDescriptor`, not planes | Borrowed only (`as_u8_slice`); its `u16` rasters cannot become a `Vec<u8>` without a copy |

imgref (1.x) is built in: with the `imgref` feature, `TryFrom` converts
`ImgRef`/`ImgRefMut` to `PlaneRef`/`PlaneMut` and back, and adopts an `ImgVec` as
a `PlaneBuf`, without copying (pointer identity is tested). The conversions are
fallible because imgref does not check its buffer length, holds `u32`
dimensions and needs a nonzero stride. `PlaneBuf` to `ImgVec` (and
`PlaneBuf::into_vec` for any owner of a plain `Vec`) succeeds when row 0 is at
index 0 and otherwise returns the plane unchanged.

Row 0 at index 0 and 64-byte alignment cannot both be guaranteed in an owned
`Vec<T>`: `Vec` frees with `align_of::<T>()`, so it cannot own a block allocated
with 64-byte alignment, and this crate has no unsafe code to try. What the
allocator gives decides which one a plane gets. Measured on r7900x (glibc 2.43,
2026-11-01, five allocations per size, `benchmarks/alloc-alignment/`): blocks of 128 KiB and more
(mmap-backed by default) started 16 bytes past a 64-byte boundary every time;
smaller blocks landed on 16-byte boundaries at varying offsets. glibc raises its
mmap threshold after large frees, so this is typical, not guaranteed. A plane
adopted from a codec's `Vec` therefore usually has every row at +16 bytes, the
offset that cost AVX2 and AVX-512 28-54% on cache-resident rows and under 7%
from memory (`benchmarks/row_alignment_2026-10-31.md`). `alloc` avoids that
with a start offset, which is why its large planes usually cannot be handed over
as a `Vec`. Kernels that peel to an aligned store address would make adopted
buffers cost little either way; not measured.

`into_strided_samples` / `into_strided_samples_mut` return the samples for the
plane's own lifetime, which is what a borrowed external frame needs.

### Aligned planes with the `Allocator` API (Rust 1.100)

`Allocator`, `Vec<T, A>` and `Box<T, A>` are stable from Rust 1.100
(rust-lang/rust#156882, merged 2026-09-23, milestone 1.100.0; `Global` and
`System` implement it). An allocator that widens every layout to 64-byte
alignment, both when allocating and when freeing, makes a `Vec<T, Align64>`
start on 64 bytes at index 0, and clones and growth stay aligned. A nightly
prototype confirmed this from 64 B to 32 MiB, and Miri found no undefined
behaviour (`benchmarks/alloc-alignment/`). For our own planes this removes the
trade-off: no start offset, `into_vec` always succeeds, and cloning keeps the
alignment.

It does not remove the copy at the boundary with crates that take `Vec<T>`
(`Vec<T, Global>`): a block allocated by one allocator cannot be freed by
another. Crates generic over their container (imgref's `Img<Container>`) or over
`A` can take the aligned `Vec` directly.

The current API extends to it without a semver break:

- `PlaneStorage for Vec<T>` widens to `Vec<T, A>` for any `A: Allocator`, a
  strict superset of today's impl.
- `into_vec` moves to `Plane<Vec<T, A>>` returning `Vec<T, A>`; for `Global` the
  signature is unchanged.
- `alloc` keeps allocating from `Global` with its offset; an `alloc_in(width,
  height, alloc)` constructor takes the aligning allocator.
- `PlaneBuf<T>` can gain a defaulted parameter, `PlaneBuf<T, A = Global>`.

Two decisions remain: implementing `Allocator` is `unsafe impl`, so the
allocator cannot live in zenpixels (`forbid(unsafe_code)`) and needs a home in a
verified primitive crate; and zenpixels' MSRV is 1.85, so these impls need
either an MSRV of 1.100 or a compiler-version gate (impls only, so the gate is
additive).

Additions that would serve the other adapters, each additive:

- `PlaneBuf` into its `Vec` plus start offset, for owners that accept an offset
  (`into_vec` covers owners that need index 0).
- `f16` and `i16` samples: a `PlaneSample` impl and an `AnyPlane` variant each.
- The variants listed in the table, when a source produces them.

Where adapters live: mediaframe, pixon and yuvutils-rs are pre-1.0, so a public
`From` impl naming their types would tie zenpixels' version to theirs. Put those
adapters in a separate crate or in the codec adapter that needs them; a feature
in zenpixels fits only 1.x crates such as pix (as `imgref` does today).
