# Sample layout boundaries: alignment, endianness, packed and planar data

Written 2026-10-31 against `feat/0.3.1-extensibility` (`746bd24`, PR #96).
Answers six questions: how alignment works, what to do for F16 CMYK, how to
make native-endian U16 safer, how to handle packed formats, what belongs in
garb versus a codec boundary, and planar data. Recommendations are proposals;
nothing here is implemented.

## The invariant that answers most of these

A `PixelFormat` is interleaved storage with a constant number of bytes per
pixel, one whole native-endian word per component, directly addressable by
row kernels. Everything that breaks that invariant (sub-byte samples,
several components in one word, big-endian words, shared exponents,
palettes, subsampled planes) is a storage or wire format. It is decoded at a
boundary into a `PixelFormat`, or borrowed through `sample::CodePlane`, never
added as a `PixelFormat` variant.

## 1. Alignment

Current rules (`zenpixels/src/buffer.rs`, `descriptor.rs`):

- **Pointer:** a view's first byte must be aligned to
  `PixelDescriptor::min_alignment()`, which is the channel byte size (1, 2, 4
  or 8). The doc comment at `descriptor.rs:1403` still says "1, 2, or 4"; F64,
  U64 and I64 give 8.
- **Stride:** at least `width × bytes_per_pixel`, and a whole multiple of
  `bytes_per_pixel` (`BufferError::StrideNotPixelAligned`). That makes every
  row start pixel-aligned and lets typed views and `imgref` (whose stride is
  in pixels) cover the buffer.
- **Owned buffers** over-allocate by `align - 1` bytes and store an offset, so
  a `Vec<u8>` can hold U16/F32 rows. `into_parts()` exposes that offset.
- **SIMD:** `new_simd_aligned` uses stride `lcm(bpp, simd_align)`, so every row
  start is both pixel- and SIMD-aligned (RGB8 at 64 bytes: 192-byte steps).
- **Typed entry:** `PixelSlice::new_typed` checks `align_of::<P>()`. Adopting
  a `Vec<P>` is zero-copy only when `P` has alignment 1; `Vec<Rgb<f32>>` is
  copied, because a `Vec<u8>` cannot carry a higher allocation alignment.

Gaps and proposals:

- **The stride rule rejects common padded rows.** A 24-bit BMP/DIB pads rows
  to 4 bytes: width 1 gives stride 4, and 4 is not a multiple of 3, so those
  rows cannot be borrowed and must be repacked. Any RGB8 or RGB16 source with
  a power-of-two row pitch hits the same rule. Proposal: require stride to be
  a multiple of `min_alignment()` instead, and make only the operations that
  need pixel-unit strides (`as_imgref`, whole-buffer `&[P]` casts) refuse or
  copy. This accepts more input, so it can ship in a 0.3.x minor release, but
  every row-walking path must be audited first.
- **Typed `Vec` adoption copies.** Backing the owned buffer with storage of the
  sample's alignment (for example `Vec<u16>`/`Vec<u32>`/`Vec<u64>` behind an
  enum) would make `from_pixels(Vec<Rgb<f32>>)` zero-copy. The 0.3.1 work
  already requires failed adoption to return typed allocations. Do this only
  with a measured caller that pays for the copy.
- Fix the stale `min_alignment` doc comment.

## 2. F16 CMYK

Today `PixelFormat::from_parts(F16, Cmyk, None)` returns `None`: F16 CMYK
cannot be represented. `Generic` is not the right home, because it covers
only signed, 32/64-bit and F64 samples with no meaning and no conversion, and
CMYK samples have a meaning (ink amount and polarity).

The same gap is larger for **16-bit CMYK**, which print TIFFs use. zentiff
decodes CMYK at 8, 16 and float depths, then converts to RGBA inside the
decoder with the naive `(1 - c)(1 - k)` formula, ignoring any ICC profile
(`zenextras/zentiff/src/decode.rs:1742`). CMYK with alpha (TIFF CMYKA) has no
`ChannelLayout` at all.

Proposal:

- Add named formats when a consumer passes them through: `Cmyk16` first
  (zentiff), `CmykF16` only with a caller (TIFF half-float CMYK or JXL F16
  output). `PixelFormat` is `#[non_exhaustive]`, so this is additive.
- Polarity is already doubling variants (`CmykF32` / `CmykF32Inverted`).
  Before adding a third depth, decide whether polarity belongs on the
  descriptor. The F32 pair was chosen so converters cannot overlook it
  (`sample-encoding-contract.md`, "Native F32 CMYK layers"); a descriptor
  field would have to cross the CMS interfaces that receive only a
  `PixelFormat`.
- Codecs should pass CMYK through with its ICC profile and leave CMYK→RGB to
  a CMS in zenpixels-convert, not convert at decode.

## 3. Native-endian U16 (and F16/F32)

The contract is right: buffers hold native-endian words, and a file's byte
order is decoded at the boundary (`sample.rs` module docs). The 0.3.1 removal
of `ByteOrder` (which named channel order) removes the last type that looked
like it tracked endianness. What is missing is evidence and one shared kernel:

- **No big-endian CI.** None of zenpixels, garb, zenpng, zentiff or
  zenbitmaps runs on a big-endian target, so every scalar fallback is
  untested there (SIMD paths are little-endian architectures anyway). Add a
  `cross test --target s390x-unknown-linux-gnu` job (or `cargo miri test` on
  that target) for zenpixels, zenpixels-convert and garb.
- **Tests assume little-endian.** zenpixels-convert's `ext.rs` tests (after
  `ext.rs:656`) build expected bytes with `from_le_bytes`/`to_le_bytes`; they
  would fail on a big-endian host while the library is correct. Use
  `ne_bytes` or typed slices in tests. The production code read in this
  review uses native-endian or typed access (for example `u16_to_u8` in
  `convert_kernels.rs` goes through `from_ne_bytes`).
- **Byte swapping is re-implemented per codec:** zenpng's `Expand::Swap16`
  (`zenpng/src/decoder/postprocess.rs:284`), zenbitmaps' farbfeld
  `be16_to_ne_bulk` (`zenbitmaps/src/farbfeld/decode.rs:92`), and encoders
  doing the reverse. Proposal: one strided `be16↔ne` / `le16↔ne` (and 32-bit)
  kernel in garb, SIMD-able, used by every codec.

## 4. Packed formats

| Kind | Examples | Handling |
|---|---|---|
| High- or low-aligned codes, one per word | P010/P016, 10/12-bit in U16 | `sample::SampleEncoding` + `CodePlane` (exists) |
| Several components per word | RGB565/555, RGBA4444, RGB10A2, BMP bitfields | unpack at the boundary; fixed layouts in garb |
| Sub-byte samples | 1/2/4-bit gray and palette (PNG, TIFF, BMP, PBM) | unpack at the boundary; shared kernel in garb once two codecs need it |
| Shared exponent | Radiance RGBE, RGB9E5 | decode to F32 in the codec |
| Palette | PNG/GIF/BMP/TIFF palettes, tRNS | expand in the codec |
| Packed + subsampled | YUYV/UYVY 4:2:2 | CodePlane with an element step, or unpack |

Rules:

- Unpack to the smallest named format that holds every code exactly (565 →
  RGB8, RGB10A2 → RGBA16), and record the source precision as provenance
  (`ExactDepth` / original depth) instead of inventing a variant.
- Bit order is format-specific (PNG packs MSB-first; TIFF `FillOrder = 2`
  reverses bits), so shared sub-byte kernels must take the bit order as an
  argument.
- If a consumer needs to *carry* packed data without unpacking (for example
  uploading RGB10A2 to a GPU), add a separate packed description type then,
  not a `PixelFormat` variant.

## 5. garb, zenpixels-convert, or the codec

- **garb:** stateless, color-blind kernels where only layout changes and the
  result is defined bit for bit: channel swizzles, adding/dropping/filling
  channels, gray broadcast, byte swapping, fixed packed↔unpacked layouts,
  sub-byte unpacking, interleave↔planar, and depth changes with one stated
  rounding. It takes slices plus strides, no descriptors, and has no
  zenpixels dependency. A kernel moves here when a second crate needs it.
- **zenpixels-convert:** anything a descriptor decides: transfer functions,
  primaries and gamut, signal range, alpha mode and policy, luma weights and
  whether they apply to linear or encoded values, CMS, YCbCr matrix and
  siting, planning and negotiation. It calls garb for the mechanical steps.
- **Codec boundary only:** the format's own wire rules: file byte order,
  filters and predictors, bit order, palettes and tRNS, bitfield masks, RGBE,
  inverted polarity (Adobe CMYK JPEG, JXL CMYK), `MinIsWhite`. The output is a
  zenpixels format with an honest descriptor and the source ICC kept. Color
  conversion does not belong here (zentiff's CMYK→RGBA is the counterexample).

One disagreement to resolve under this split: garb's `convert_u16_to_u8`
rounds with `(v * 255 + 32768) >> 16`, which differs from exact rounding
`(v + 128) / 257` on 127 of the 65,536 inputs (first at v = 33025; computed
2026-10-31). zenpixels-convert uses the exact form, and nothing outside garb
calls garb's kernel today. Depth kernels should have one rounding definition:
move the exact kernel into garb and have zenpixels-convert call it.

## 6. Planar

State on the 0.3.1 branch: the owned `planar` module
(`MultiPlaneImage`, `PlaneDescriptor`) is removed; a sweep on 2026-10-31
found no remaining users in sibling crates (zenfilters' `PlaneSemantics` is
its own type). What exists instead:

- `sample::CodePlane`: a borrowed U8/U16 plane with storage, code depth, bit
  placement, byte stride and element step, so it covers planar, semi-planar
  (NV12/P010) and interleaved native layouts without copying.
- `ChromaSubsampling`, `ChromaSiting`, and zenpixels-convert's
  `ycbcr::YcbcrToRgbF32` for the converting route
  (`borrowed-ycbcr-contract.md`).

Proposal:

- Codec-native YCbCr stays borrowed `CodePlane`s plus an explicit
  matrix/range/siting description; never a `PixelFormat` variant
  (`yuv-carrier-assessment.md`, "Why not just add PixelFormat::Yuv420?").
  An owned YCbCr frame type stays deferred, as the owner chose on 2026-09-27.
- Extra planes (separate alpha, CMYK black, JXL extra channels, depth and
  gain maps) are separate `PixelBuffer`s of a gray format with a role
  recorded by the caller.
- Planar working buffers (zenfilters' `OklabPlanes`) stay inside their crate.
  Add a shared borrowed planar view only when two crates hand planar F32
  between them.
- Interleave↔planar kernels belong in garb. It has `rgb24`/`rgb48` → F32
  planes (`garb/src/deinterleave.rs`); the reverse direction and U8/U16
  outputs are missing.
