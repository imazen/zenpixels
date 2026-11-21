# Planar types (2026-11-01)

One set of planar types for every zen crate. They live in `zenpixels::plane`
so codecs, encoders, zenyuv and metrics can use them without depending on
zencodec. zencodec keeps what is frame-specific (crop, timing, color claims,
configuration epoch, owner and mapping traits) and builds on these.
The inventory that motivated this is `docs/planar-inventory.md`.

Terms: a **sample** is one stored native integer or floating-point element,
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

1. **Storage geometry: `Plane<C>`.** A strided 2-D array of native samples
   (the shared sealed `StorageSample` contract, also named `PlaneSample`). `width` and `stride` count samples, so a stride
   always holds whole samples and typed rows never need a checked cast. Rows are
   contiguous `&[T]`; the last row needs no padding after it.
   - `PlaneRef<'a, T>` = `Plane<&'a [T]>`, `PlaneMut<'a, T>` = `Plane<&'a mut [T]>`,
     `PlaneBuf<T>` = `Plane<Vec<T>>`. Other owners (zencodec's accounted shared
     storage) implement `SampleStorage` / `SampleStorageMut` and become
     `Plane<TheirOwner>`.
   - `new` adopts storage without copying (a codec's `Vec`, a decoder pool's
     slice). `AlignedPlane::alloc` allocates fallibly, starts every row on 64 bytes
     with row 0 at index 0, and adds 64 bytes to strides on 4096 multiples
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
     planes need no separate range. `Float64(FloatEncoding)` retains F64 samples
     without narrowing; the affine scale/offset description remains F32.
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

## Adoption status (2026-11-03)

Draft PRs against zenpixels rev `5bf81d64`, each assigned for review. "Checked"
means `cargo check` of the named crates passed; tests are listed where they ran.

| Repo | PR | What it does | Verified |
|---|---|---|---|
| zencodec | imazen/zencodec#192 (on #191) | `MappedPlanes::plane` returns `AnyPlane`; `PlaneView`/`PlaneSamples` removed; `PlaneRole`/`PlaneError` re-exported; zencodec-media `OwnedPlane` is `PlaneBuf`/`Plane<SharedStorage>`; adds a defaulted `MappedPlanes::encoding(index)` (open question: carry a `PlanarLayout` instead) | zencodec, zencodec-media, testkit checked; their tests pass; `perf/` not migrated |
| zenmetrics | imazen/zenmetrics#108 (on #104) | abstractions re-export zenpixels' `PlanarLayout`/`PlaneSpec`/`PlaneRole`; vmaf and callers updated | abstractions + vmaf checked, 81 tests pass in a CI-equivalent tree |
| heic | imazen/heic#65 | `DecodedFrame::planes()` borrows Y/Cb/Cr/alpha (feature `zenpixels`) | heic-core checked; pointer-identity test passes |
| zenwebp | imazen/zenwebp#120 | `YuvPlanes::as_planes()`; pins zencodec at #192's head | checked; test passes; lands after #192 |
| zenjpeg | imazen/zenjpeg#255 | `DecodedYCbCr::planes()` as f32 (Y scale 255 offset -128, Cb/Cr offset 0); `YCbCrPlanes::from_planes` | checked; test passes |
| zenavif | imazen/zenavif#80 | `DecodedYuv::zenpixels_planes()` | lib and unit tests compile; test passes; lockfile still lists zenpixels 0.2.16 beside 0.3.1 |
| zenav1-aom | imazen/zenav1-aom#22 | `KeyFramePlanes::from_planes` for tight planes (feature `zenpixels`) | checked; test compiled, not run |
| zenav1-svt | imazen/zenav1-svt#29 | `AnimationFrame<u8>::as_planes(width, height)` (`svtav1::Frame` no longer exists) | test passes with `avif-container`; building frames from planes not done |

Not yet adopted: zenavif `media.rs` (after #192), zenyuv (its kernels take tight
slices, so plane input needs stride support first), zenfilters `OklabPlanes` and
jxl-encoder `YCbCrPlanes` (crate-private), zenmetrics vmaf `Yuv420Frame`.

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
`PlaneBuf::try_into_vec` for any owner of a plain `Vec`) hands the adopted `Vec` back
when row 0 is at index 0, and otherwise returns the plane unchanged. Today's
constructors always put it there; the method stays fallible so that a later form can
use an offset. `ImgVec` also refuses dimensions it cannot hold.

A `Vec<T>` cannot be both 64-byte aligned and start at index 0, because `Vec` frees
with `align_of::<T>()`. `AlignedSamples<T>` holds a `Vec` of a `#[repr(align(64))]`
chunk instead, and views it as `[T]` and as bytes through `bytemuck`. `Vec` then
allocates and frees with the chunk's layout, so index 0 is always on 64 bytes, on any
allocator and with no `unsafe`, and clones stay aligned. 64 bytes covers every
interleaved pixel type, so viewing samples as `&[Rgb<u16>]` cannot fail on alignment.
The cost is the boundary: it cannot become a `Vec<T>` without a copy. Owners generic
over their container (`ImageBuffer`, `imgref::Img`) take it directly.

A plain `Vec` from a codec gets what the allocator gives. Measured on r7900x (glibc
2.43, 2026-11-01, five allocations per size, `benchmarks/alloc-alignment/`): blocks of
128 KiB and more (mmap-backed by default) started 16 bytes past a 64-byte boundary
every time; smaller blocks landed on 16-byte boundaries at varying offsets. glibc
raises its mmap threshold after large frees, so this is typical, not guaranteed. A
plane adopted from such a `Vec` therefore usually has every row at +16 bytes, the
offset that cost AVX2 and AVX-512 28-54% on cache-resident rows and under 7% from
memory (`benchmarks/row_alignment_2026-10-31.md`).

`into_strided_samples` / `into_strided_samples_mut` return the samples for the
plane's own lifetime, which is what a borrowed external frame needs.

### Owners and allocators

Within one image or animation sequence every plane comes from the same
allocator (owner decision, 2026-11-01); mixing allocators inside a picture is
not a design goal. Sample types do differ inside one picture (10-bit color in
`u16`, 8-bit alpha in `u8`), so an owned set of planes, such as the
replacement for zencodec-media's `OwnedPlanes`, is generic over one allocator
`A` shared by all its planes and chooses the sample type per plane:
`Plane<Vec<u8, A>>`, `Plane<Vec<u16, A>>` or `Plane<Vec<f32, A>>` in a
non-exhaustive enum. Accounted storage fits the same shape as a storage family
(`SharedStorage<T>` for every `T`).

`Vec<T, A>` stores `A` by value, so a zero-sized allocator costs nothing.
Measured with `size_of` on `rustc 1.99.0-nightly (77cf889bc 2026-07-12)`,
x86_64: `Vec<u8>`, `Vec<u8, Global>` and `Vec<u8, Align64>` are 24 bytes; an enum
allocator over `Global` and `Align64` gives 32; `&dyn Allocator` or
`Arc<dyn Allocator + Send + Sync>` gives 40, and the `Arc` form adds reference
counting and an indirect call per allocation and free. A generic `A` per set is
therefore the default; an enum or `dyn` allocator is only for a container that
must hold sets from several allocators at once (an enum costs 8 bytes per `Vec`
and a branch per allocation, never per sample). Moving an existing `Vec` between
allocator types needs `Vec::from_parts_in` (unsafe), so such allocators belong
with the aligning allocator in a primitive crate.

Erasure, where an interface needs it, happens for a whole picture, not per
plane: zencodec's `Picture` trait object already hides the owner, and its
mappings hand out `AnyPlane` views. Borrowed views (`view()`, `AnyPlane`)
erase the owner at no cost. `SampleStorage` and `SampleStorageMut` also stay
dyn-compatible, and `&S`, `&mut S` and `Box<S>` implement them for any
`S: ?Sized`, so `Plane<Box<dyn SampleStorageMut<Sample = T>>>` works where a
caller wants it (one box per plane, an indirect call per row read through it).

Semver rules that keep this open:

- `&`, `&mut` and `Box` are fundamental types: a downstream crate may implement
  our trait for `Box<TheirType>`, so a blanket impl added after release would
  conflict. Those blanket impls exist from the first release.
- `Arc` and `Rc` are not fundamental, so `SampleStorage for Arc<S>` or `Rc<S>` can be
  added later.
- Every future method on the storage traits must keep them dyn-compatible: no
  generic methods, no `Self` by value, no `Self: Sized` requirements without a
  default.

### `v_frame` and `image` (measured 2026-11-03)

Probes and full output: `benchmarks/plane-interop/`. Every route below moved a
pointer and allocated nothing, unless marked as a copy.

| Route | Result |
|---|---|
| `v_frame` 0.7 `Plane` to `PlaneRef`/`PlaneMut` | Zero-copy with `padding_api`: `&data()[pad_top * stride + pad_left..]`. Without the feature `v_frame` exposes rows only, not one slice |
| `v_frame` 0.3.9 (zenrav1e's pin) to `PlaneRef`/`PlaneMut` | Zero-copy: `data_origin()`, `cfg.stride` |
| Owned `v_frame` plane to zenpixels | Zero-copy through a small `SampleStorage` newtype, keeping its allocation and padding |
| zenpixels to `v_frame` | **Copy, both versions.** `v_frame` has no constructor that adopts storage |
| Strided `PlaneRef`/`PixelSlice` to `image::flat::View` | Zero-copy with `SampleLayout`: `height_stride` is the plane's stride |
| Tight `PlaneRef` to `ImageBuffer<_, &[T]>`; `ImageBuffer` to `PlaneRef` | Zero-copy |
| `ImageBuffer<_, Vec<T>>` to and from a tight `PlaneBuf`/`PixelBuffer` | Zero-copy for `u8`, `u16`, `f32` |
| `AlignedPlane` or `AlignedSamples` to `ImageBuffer<_, AlignedSamples<T>>` | Zero-copy when the rows are tight; the image keeps the aligned container |
| `AlignedSamples<T>` to `ImageBuffer<_, Vec<T>>` | **Copy.** A `Vec` must free with the layout it allocated |
| `DynamicImage` to and from `PixelBuffer`, every variant | Zero-copy, through `PixelStorage` |

The `image` feature provides these as `TryFrom` impls, for gray, gray-alpha, RGB and
RGBA in `u8`, `u16` and `f32`: `ImageBuffer` to and from `Plane` and `PixelBuffer`,
`&ImageBuffer` and `&mut ImageBuffer` to the borrowed types, strided `PlaneRef`,
`PlaneMut` and `PixelSlice` to `FlatSamples`, and every `DynamicImage` variant to and
from `PixelBuffer` and the borrowed types.

`PixelFormat` converts with all 10 `ColorType`s, and with 14 of the 29 `ExtendedColorType`s:
`L8`, `La8`, `Rgb8`, `Rgba8`, `L16`, `La16`, `Rgb16`, `Rgba16`, `Bgr8`, `Bgra8`, `Rgb32F`, `Rgba32F`,
`Cmyk8` and `Cmyk16`. `A8` isn't a packed format, but it fits: a `Plane` with `PlaneRole::Alpha`, which
is also a valid single-plane `PlanarLayout`. The other 14 refuse, because they have no byte-addressable
samples: `L1`, `La1`, `Rgb1`, `Rgba1`, `L2`, `La2`, `Rgb2`, `Rgba2`, `L4`, `La4`, `Rgb4`, `Rgba4` and
`Rgb5x1`, which would break `bytes_per_pixel` and row math, and `Unknown`, which says nothing about
layout. `image` has no pixel type for any of them, nor for `Bgr8`, `Bgra8` or `Cmyk`, so those stop at
the color-type mapping. `Bgr8` and `Cmyk16` carry samples exactly; numeric conversion refuses
(`NoPath`).

`tests/image_interop.rs` runs on plain
`cargo test` because the crate has a dev-dependency on itself with the feature on.
`image` needs Rust 1.88; zenpixels without the feature stays on 1.85. The guard in
`tests/dep_guard.rs` keeps `image` and `rustversion` optional and out of the defaults.

What those conversions refuse, returning the source unchanged:

- Moving to `ImageBuffer` needs tight rows and the first pixel at index 0, because
  `ImageBuffer::from_raw` checks only that the buffer is long enough: it accepts
  strided samples and reads wrong pixels. Use `FlatSamples` for strided data.
- `image` has no "unknown" color and labels everything sRGB by default. A buffer moves
  only when `image` can state its color exactly: not with unknown primaries or
  transfer, Gamma22, Adobe RGB, narrow range, an ICC profile, or CICP that disagrees
  with its descriptor. Relabel with `with_transfer`/`with_primaries` when the color is
  known. Premultiplied and undefined alpha are refused too, since `image` means
  straight alpha. A `Plane` carries no color, so its image is labeled sRGB: set the
  color on the result.
- In the other direction color always travels. CICP that a descriptor cannot name is
  kept in the color context, and that buffer cannot return to `image`.

For `v_frame`, which has no feature because nothing here consumes it:

- `v_frame` 0.7 refuses odd luma sizes for 4:2:0 (`UnsupportedResolution` at 37x11),
  while our layouts allow them through `ceil(w/2)`. Odd-sized 4:2:0 cannot reach it,
  zero-copy or not.
- Its `copy_from_u8_slice_with_stride` requires exactly `stride * height` bytes, so it
  refuses `into_strided_samples()` when the last row has no padding. Copy row by row.

### Packed buffers: `PixelStorage`

The default `PixelBuffer<P, S = PixelStorage>` holds a global `Vec` of the
format's real sample type. Consuming casts between global `Vec<u8>` and
`Vec<u16>` cannot preserve their different deallocation alignments; compatible
same-alignment casts can reuse storage (`Vec<Rgb<u16>>` to `Vec<u16>`, measured).
The enum provides a fixed owner type for APIs requiring the default PixelBuffer;
it is not the only possible ownership-erasure mechanism. A generic storage owner
can instead be retained behind an existing image-storage trait object. Only
`as_bytes` branches on this enum's variant. `PixelSlice` stays `&[u8]`, so `row()` and
kernels have no branch.

- **Going in:** `PixelBufferParts::new(rgb_f32.into(), ...)` moves a decoder's `Vec`.
  `from_vec(Vec<u8>)` stays for bytes; `from_pixels(Vec<P>)` now moves instead of copying.
- **Coming out:** `into_parts().data` is the enum. `try_into_vec::<T>()` returns the `Vec`
  or the storage unchanged, and `try_into_tight_u8_vec` also checks the offset and stride.
  `TryFrom<PixelBuffer<P>> for ImgVec<P>` moves a typed buffer. None of these hides a copy.
- **Allocation:** zeroed storage comes from `try_zeroed_vec`, which is fallible and gets untouched
  zero pages as `vec![0; n]` does, so no separate infallible path is needed for speed
  (`benchmarks/zeroed-alloc/`).
- **Size:** the default buffer stays 72 bytes, which a compile-time assert pins. The enum
  costs 8 bytes because a nested enum's tag can't use the outer struct's padding, so
  `offset` is stored as `u32` and the freed bytes plus existing padding absorb it.
  `try_from_parts` refuses an offset past 4 GiB. `Box<[T]>` would also fit, but shrinking
  a `Vec` with spare capacity can reallocate, a hidden copy, and it loses the capacity.
- **Offsets stay:** adopted buffers can have a prefix, and `offset` plus an LCM stride
  (`PixelDescriptor::simd_aligned_stride`) is how a later constructor can reach 64-byte
  rows in a plain `Vec`. Handoffs are `try_` so that adding one needs no API change.

**Alignment and handoff conflict.** Keeping every row on 64 bytes takes a padded stride,
and `ImageBuffer` needs tight rows. `AlignedSamples` can't become a `Vec<T>` at all. So
`PixelStorage` is plain `Vec`s with tight rows, which favors the handoff on whole frames,
where alignment gains least (under 7% from memory,
`benchmarks/row_alignment_2026-10-31.md`). 64-byte buffers belong where kernels work on
cache-resident rows: `AlignedPlane`, `try_new_aligned`, or a `DecodeRowSink` that owns its
memory. The sink is also the route for custom allocators and pools across `zencodec`.
`PixelStorage` is `#[non_exhaustive]`, so aligned or boxed variants can come later.

`S` stays for storage that doesn't flow through `DecodeOutput`. Any `SampleStorage` works:
`Vec<T>`, `Box<[T]>`, `AlignedSamples`, slices, pool slots. `ByteStorage` is what
`PixelBuffer` needs; everything past that reference works on bytes.

Still `PixelStorage` only: `into_contiguous`, `into_contiguous_pixels`,
`transform_in_place`, and `PixelCow::Owned`.

### The `allocator_api` feature

Alignment no longer needs a custom allocator; `AlignedSamples` has it. The feature is
for storage that comes from somewhere else, such as a pool or an arena. With it,
`Vec<T, A>` and `Box<S, A>` are `SampleStorage` for any `A: Allocator`, and `try_into_vec`
returns `Vec<T, A>`. That includes `Global`, so existing code keeps compiling. The
impls are gated with `rustversion`: `Allocator`, `Vec<T, A>` and `Box<T, A>` are stable
from Rust 1.100 (rust-lang/rust#156882), and on older compilers the feature has no
effect, so `--all-features` keeps working on them. CI's beta job tests the real impls
until 1.100 is stable.

A custom allocator is still `unsafe impl`, so none lives in this crate. A 64-byte
aligned one in `benchmarks/plane-interop/` was adopted by planes and packed buffers at
offset 0, and moved on to `ImageBuffer`; an earlier prototype ran clean under Miri from
64 B to 32 MiB (`benchmarks/alloc-alignment/`).

Allocating in a given allocator (`alloc_in`, `try_new_aligned_in`) is not provided.
Build the `Vec` with `with_capacity_in` and adopt it.

Additions that would serve the other adapters, each additive:

- `PlaneBuf` into its `Vec` plus start offset, for owners that accept an offset
  (`try_into_vec` covers owners that need index 0).
- `f16` and `i16` samples: a `PlaneSample` impl and an `AnyPlane` variant each.
- The variants listed in the table, when a source produces them.

Where adapters live: mediaframe, pixon and yuvutils-rs are pre-1.0, so a public
`From` impl naming their types would tie zenpixels' version to theirs. Put those
adapters in a separate crate or in the codec adapter that needs them; a feature
in zenpixels fits only 1.x crates such as pix (as `imgref` does today).

## Plane views, picture views and allocator ownership — PR144

The types vary along two independent axes: **what they describe** and
**whether they own or borrow storage**. Lifetimes are omitted in this table.

### Geometry-bearing types: owner → read view → write view

| What it describes | Owned form | Read-only borrow | Writable borrow | Interpretation carried |
|---|---|---|---|---|
| One plane, sample type `T` selected at compile time | `PlaneBuf<T>` | `PlaneRef<T>` | `PlaneMut<T>` | Dimensions and sample stride; no role or color |
| One plane, sample type selected at runtime | `AnyPlaneBuf` | `AnyPlane` | `AnyPlaneMut` | Dimensions, sample stride and selected sample type; no role or color |
| One interleaved picture | `PixelBuffer<P, S>` | `PixelSlice<P>` | `PixelSliceMut<P>` | Dimensions, byte stride, pixel descriptor and color context |
| One planar picture | `PlanarBuffer<P>` | `PlanarSlice` | `PlanarSliceMut<P>` | Planes, image dimensions, layout, color context and alpha interpretation |

“Runtime-selected sample type” means an enum containing a concrete typed value,
not loss of sample-type information. For example, `AnyPlane::U16` contains a
`PlaneRef<u16>`, while `AnyPlaneBuf::U16` contains a `PlaneBuf<u16>`.
The shared sample roster is U8/I8/U16/I16/U32/I32/U64/I64/F32/F64.
`StorageSample` and `PlaneSample` name the **same trait**. Typed planes,
aligned samples, PixelStorage and all three AnyPlane forms use that roster.
Half-float payloads still use U16 storage; no F16 primitive or numerical reader
is added. `CodePlane` numerical reading remains limited to U8/U16.

The typed plane family is one generic type: `Plane<C>`. Its aliases select
`C = Vec<T>`, `&[T]` or `&mut [T]`. `AlignedPlane<T>` is another owning alias,
using `AlignedSamples<T>`. Custom owners implement `SampleStorage` and,
when writable, `SampleStorageMut`.

`PixelBuffer` defaults to owned `PixelStorage`; its generic `S` determines
actual storage ownership. `P` describes the compile-time pixel type, with `()`
for runtime-described pixels. Interleaved does **not** mean tightly packed:
these picture types retain row stride and can include padding.

### Storage underneath the geometry

These hold allocations, not image dimensions, row strides or color interpretation.

| Storage type | Sample selection | Ownership | Used by |
|---|---|---|---|
| `Vec<T>` | Compile-time `T` | Owns a contiguous allocation | `PlaneBuf<T>`; supported typed `PixelBuffer` storage |
| `AlignedSamples<T>` | Compile-time `T` | Owns aligned sample storage | `AlignedPlane<T>`; aligned `PixelBuffer` storage |
| `PixelStorage` | Runtime-selected typed vector | Owns the selected vector | Default `PixelBuffer` storage |

`PixelStorage` and `AnyPlaneBuf` both select a sample type at runtime, but at
different levels: **PixelStorage holds an allocation; AnyPlaneBuf holds an
allocation plus plane geometry.** Both are convenient enums of ordinary typed
vectors. Neither requires custom storage to become an ordinary vector:
`PixelBuffer<P, S>` keeps its concrete `S`, and `PlanarBuffer<P>` keeps each
concrete plane owner `P`.

For aligned planar storage, use `PlanarBuffer<AlignedPlane<T>>`. For a pool or
accounted owner, use `PlanarBuffer<Plane<MyStorage>>`. For heterogeneous owner
kinds or sample types, implement `PlaneStorage` on the owner's enum, and
`PlaneStorageMut` only if it can lend exclusive samples. This is the same
extension point needed by zencodec-media's existing vector/accounted plane enum;
no codec dependency or per-owner enum variant belongs in zenpixels.

Owners must be `Sync` to lend the shared thread-safe `PlanarSlice`; they need
not be `Send`. Read-only owners need not implement mutable access. These
contracts require stable geometry and sample type while a picture borrows them.
No allocation, boxing, or pixel copy is introduced by owner erasure when lending
read-only views; writable views retain the concrete owner type.

An aligned allocation cannot become an ordinary `Vec<T>` without copying: its
deallocation layout differs. `Plane::into_storage()` returns the **original**
owner, preserving alignment, allocator/pool identity and its release behavior.

### Descriptions without storage

`PlanarLayout` describes plane roles, encodings, subsampling and chroma siting.
`PixelDescriptor` describes interleaved pixel format and interpretation.
`ColorContext` carries current color signaling. None owns pixel samples;
the complete picture types bind these descriptions to actual storage.

A single plane can hold interleaved Cb/Cr, and a complete monochrome picture
can contain only one plane. “Plane” describes one strided sample array;
“picture” adds the interpretation needed to use it as an image.

A codec can return `PlanarBuffer` by moving its Y/Cb/Cr allocations into
`AnyPlaneBuf` values. `as_slice()` and `as_slice_mut()` allocate neither pixel
storage nor descriptor lists. A caller-owned array of `AnyPlaneMut` can also
be passed to `PlanarSliceMut::new`. Both constructors validate plane count,
geometry, sample encoding and alpha using the same rules as `PlanarSlice`.
The last row needs no trailing padding; chroma sizes round up at odd edges.

Mutable access returns reborrowed plane values, never a writable descriptor
slice. `planes_mut()` can lend several disjoint planes simultaneously. Replacing
one returned value does not replace the original descriptor. The mutable view
keeps the short picture borrow separate from any backing lifetime in its owner
type, so a caller can borrow the same list again for a later processing pass. Read interpretation
through `as_slice()`. Neither view infers color or performs conversion. Frame
timing and shared/hardware-backed frame ownership remain media-layer concerns.

**Accessor migration:** `PlanarSlice::planes()` now returns an exact-size,
double-ended iterator of `AnyPlane` values instead of a descriptor slice.
Use `plane(i)` for checked indexed access, and `planes().zip(...)` for iteration.
`PlanarSlice::new` still accepts the same borrowed descriptor slice. The private
view representation borrows the descriptor owner; `PlanarLayout` remains
Vec-backed and no fixed plane-count storage limit is introduced.

`PlanarBuffer::try_from_parts` preserves every owner, layout and color field on
failure, matching the packed buffer's recovery pattern. `take_parts()` recovers
the rejected transfer record for correction and retry; `without_buffer()` drops
it when recovery is unnecessary. `into_parts()` moves a valid picture's owners
and interpretation together. Neither success nor failure requires allocation.
The convenience `new` constructor still drops rejected inputs, like packed
convenience constructors. `into_planes()` deliberately discards interpretation;
use `into_parts()` when transferring a complete picture.

### What Rust1.100 support already covers

[Allocator](https://doc.rust-lang.org/nightly/std/alloc/trait.Allocator.html)
and [Vec<T,A>](https://doc.rust-lang.org/nightly/std/vec/struct.Vec.html#method.new_in)
are marked stable from Rust1.100 in the current official docs. The existing
`allocator_api` feature enables generic Vec/Box storage implementations at that
compiler version and is a no-op on older compilers.

- Plane<Vec<T,A>> and Plane<Box<[T],A>> preserve the allocator owner for the
  shared native storage sample roster listed above.
- PixelBuffer<(),Vec<T,A>> works through the existing storage traits and
  try_from_parts/into_parts. It need not become PixelStorage.
- PlaneRef, PlaneMut, AnyPlane and PlanarSlice borrow samples independently of
  their allocator. They cannot grow, free or replace the original allocation.
- PlaneBuf<T> remains the convenient global-Vec alias. PixelStorage's public
  variants and its Vec adoption/extraction APIs also remain Global-only.
  AlignedSamples currently owns a global Vec<64-byte-aligned Chunk>.

The existing allocator_api tests cover generic owners and extraction. PR144's
additional beta control mutates through PlaneMut, lends a PlanarSlice with a
short final row, then recovers Vec<u16,System> with the original pointer,
allocator and padding. This is an ownership/interpretation test, not a
performance or allocation-count measurement.

### Alignment has separate obligations

1. **Allocation layout:** retain the allocator, allocation size/capacity and
   alignment needed for eventual deallocation or growth. A numerically aligned
   pointer does not establish a compatible owned-vector allocation layout.
2. **Typed sample access:** every referenced sample must meet align_of<T>, and
   byte-backed views must validate their offset and stride for that type.
3. **Row alignment:** an aligned allocation does not align every row. For
   alignment A, both first-row address and row stride must be multiples of A.
4. **Subview alignment:** cropping changes the first address. A parent aligned
   to A can produce a crop that is only sample-aligned. Kernels must preserve a
   valid unaligned path and active-row bounds, including short final rows.
5. **Ownership transfer:** a borrowed reinterpretation and a consuming vector
   conversion have different requirements. Recovering the original owner and
   borrowing typed views preserves its deallocation contract; retyping an owned
   Vec also needs compatible layout/capacity and valid element representation.

For a scalar plane the row/crop address is
`base + offset_bytes + y * stride_bytes + x * size_of<T>()`.
An aligned allocator controls base; it does not choose offsets, pitches or crop
coordinates. Over-alignment alone never supplies padding or permission to read
past active rows. The official [allocator memory-fitting contract](https://doc.rust-lang.org/nightly/std/alloc/trait.Allocator.html#memory-fitting)
requires the allocator and allocation alignment to remain compatible.

### Gap and direction for owned erasure

Custom allocator storage can reach today's borrowed processing paths, but cannot
be moved into PixelStorage without satisfying its Global-Vec representation.
A fixed PixelBuffer field, such as zencodec DecodeOutput at
[dc128a4f](https://github.com/imazen/zencodec/blob/dc128a4faaa161d7f63559c4c640a0530e3490a6/src/output.rs),
therefore narrows the available owners. Generic storage support alone does not
fix that ownership-transfer boundary.

There is already another erasure layer:
[zencodec ImageOutput at dc128a4f](https://github.com/imazen/zencodec/blob/dc128a4faaa161d7f63559c4c640a0530e3490a6/src/image.rs)
holds Arc<dyn ImageStorage>. That private trait owns the mapping lifetime;
ImageOutput clones share the owner. Its current packed adapter still holds the
default PixelBuffer. The smallest proposed extension is a private Packed<S>
retaining PixelBuffer<(),S>, with a reviewed constructor accepting that storage.
This can hide the concrete allocator/pool owner without allocator parameters
across every codec or deep-cloning its pixels. The current Send/Sync/static
ownership and mapping contracts must still hold. This extension is not
implemented by PR144.

Keep processing APIs on borrowed views and retain concrete S through owned
handoffs where possible. Allocator-generic typed-vector storage (conceptually
PixelStorage<A=Global>) remains an alternative when the enum itself needs to
carry allocator A. It preserves one allocator type but does not by itself solve
fixed output signatures or non-Vec owners. A new owner-erased carrier would
also have to define mutability, cloning/fallible copying, lifetime/Send bounds,
growth and recovery; first use the erasure already present. Do not add an enum
variant with implicit copying or allocator loss under the existing Clone
contract.

No new allocator trait is needed: existing SampleStorage/ByteStorage and
borrowed picture views are the processing boundary. Supplied allocator
implementations can provide over-alignment without adding unsafe allocator
implementation code to zenpixels.

## Owned-planar qualification

The owned-planar review adds `just planar-storage-check`: all-feature core tests
and strict all-target Clippy, a genuine no-default-features build, and the
zenpsy consumer suite. It passes without relaxing existing expectations.
Core doctests include a decoder-style Y/Cb/Cr adoption example. Public API
snapshots are regenerated with `just api-doc`.

`tests/planar_buffer.rs` refuses every allocation while adopting existing
owners and borrowing read/write views, verifies original sample pointers,
mutates three live plane borrows simultaneously, and checks untouched padding.
It also checks mixed U16/F32 encodings, descriptor replacement isolation,
invalid count/type/geometry/alpha and sequential mutable descriptor-list reuse.
The last control originally failed with E0499/E0502 when both lifetimes were
tied together; separating picture and backing-sample lifetimes fixes it.
These are storage-contract tests, not a benchmark or codec-integration claim.

The storage-symmetry controls extend this qualification to every native storage
type, including signed endpoints, float negative zero and NaN payloads. Custom
pool and aligned owners retain their original pointers and release exactly once;
shared read-only owners do not implement writable access. Rejected adoption
retains the original storage and color for correction and retry with allocation
refused. Wide/signed planar hashing masks payload bits in little-endian order;
existing portable hash goldens and U8/U16 code-reader behavior stay unchanged.
This establishes carriage and ownership, not wider numerical conversion support.
