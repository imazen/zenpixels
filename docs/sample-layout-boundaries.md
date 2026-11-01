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
  a power-of-two row pitch hits the same rule. An earlier version of this doc
  proposed relaxing the rule to "a multiple of `min_alignment()`". **Withdrawn
  (2026-10-31): the whole-pixel stride is a contract callers rely on.**
  - zenwebp computes `stride_pixels = pixels.stride() / bytes_per_pixel()`
    (`zenwebp/src/codec.rs:984`) and hands it to its encoder: a non-pixel
    stride would truncate and read wrong rows without any error.
  - `as_imgref`/`as_imgref_mut` (`buffer.rs:2265`, `:2294`) call bytemuck's
    panicking `cast_slice` over the whole buffer before their stride assert;
    with a non-pixel stride the slice length is not a multiple of
    `size_of::<P>()`, so these infallible methods would start panicking.
  - Nothing becomes unsound: both crates are `forbid(unsafe_code)` and every
    cast is checked. The risk is wrong pixels downstream and new panics, which
    a grep cannot rule out across every consumer.

  Padded sources stay a boundary concern: the decoder repacks rows into a
  whole-pixel stride (one copy, already what happens), or exposes the samples
  as `sample::CodePlane`s, which take any byte stride and an element step.
- **Typed `Vec` adoption copies.** Backing the owned buffer with storage of the
  sample's alignment (for example `Vec<u16>`/`Vec<u32>`/`Vec<u64>` behind an
  enum) would make `from_pixels(Vec<Rgb<f32>>)` zero-copy. The 0.3.1 work
  already requires failed adoption to return typed allocations. Do this only
  with a measured caller that pays for the copy.
- Fix the stale `min_alignment` doc comment.

### Stride across every SIMD target

Checked 2026-10-31 against archmage's `token-registry.toml`, magetypes'
backends, and the hosts named below.

| Tier (archmage token) | Vector bytes | Where |
|---|---|---|
| x86-64 v1 / v2 (SSE2, SSE4.2) | 16 | every x86-64 |
| x86-64 v3 (AVX2 + FMA) | 32 | Haswell+, Zen 1+ |
| x86-64 v4 / v4x / AVX512-FP16 | 64 | Skylake-X+, Zen 4+ (Zen 4 double-pumps; dev's Zen 5 is full width) |
| NEON / Arm64 v2 / v3 | 16 | every AArch64 we run (Neoverse-N1 on the Hetzner boxes has no SVE: `/proc/cpuinfo`, checked) |
| wasm128 (+ relaxed) | 16 | browsers, zenwasm |
| scalar | — | fallback |

archmage has no SVE or RISC-V V tokens, so no scalable vectors are in play.

Facts that settle the design:

1. **Alignment is never a correctness requirement on any tier.** Every
   magetypes backend loads and stores through `simd_storage::copy` from a
   `&[T; N]` (`magetypes/src/simd/impls/x86_v3.rs:352`, `x86_v4.rs:51`,
   `arm_neon.rs:37`, `wasm128.rs:30`): unaligned instructions on every ISA.
   `forbid(unsafe_code)` crates cannot reach aligned-only intrinsics anyway.
   Alignment only changes speed.
2. **The cost is cache-line splits, measured per tier.** Lines are 64 bytes on
   r7900x (Zen 4) and on Neoverse-N1 (`getconf LEVEL1_DCACHE_LINESIZE`, both
   measured). macOS reports 128 bytes on Apple silicon (`hw.cachelinesize`;
   Go's `x/sys/cpu` hard-codes 128 for darwin/arm64); not measured here, since
   the mac host was unreachable. `benchmarks/row_alignment_2026-10-31.md`
   measures a load/store-bound F32 row kernel with rows 0, 4, 16 or 32 bytes
   past a 64-byte boundary:

   | Tier (host) | Cache-resident rows, misaligned start | From memory |
   |---|---|---|
   | v4, AVX-512 (Zen 4) | +39% to +54% at +4/+16; +9% at +32 in L2 | under +6% |
   | v3, AVX2 (Zen 4) | +28% to +49% at +4/+16; none at +32 | under +7% |
   | SSE2, compiler-vectorized (Zen 4) | +12% (L1) / +23% (L2) at +4; none at +16 | up to +12.5% at +4 |
   | NEON (Neoverse-N1) | +2% (L1) / +7% (L2) at +4; none at +16 | up to +4% at +4 |

   A start costs only when it is not a multiple of the vector width. Strip
   processing works on cache-resident rows, so on AVX2 and AVX-512 aligned
   rows are worth up to about 1.5x on this kernel; on NEON they barely
   matter. Not measured: Zen 5 or Intel full-width AVX-512, Apple silicon,
   wasm128, compute-heavier kernels.
3. **One number covers every tier: 64 bytes.** 16 and 32 divide it, so a
   64-byte-aligned row start is aligned for every vector width above, and it
   matches the x86 and Neoverse line size. 128 matters only for false sharing
   between threads writing adjacent rows on Apple silicon, not for SIMD.
4. **Strides that are multiples of 4096 cost kernels that read many rows at one
   column; +64 bytes removes it.** Measured (`benchmarks/stride_aliasing_2026-11-01.md`,
   F32 vertical filters, L2-resident rows, against the same row width at +64 bytes):

   | CPU | 2 rows | 8 rows | 12 rows |
   |---|---|---|---|
   | Zen 4 (7900X), AVX2, AVX-512, scalar | no effect | +4..+12% at 4096 only | +19..+48% at 8192, 12288, 16384, 32768; +4..+19% at 4096 |
   | Neoverse-N1 | no effect | no effect | +6..+10% at 8192 only |

   The trigger is a multiple of 4096: 12288 behaves like 8192, while 1024, 2048
   and 6144 show nothing, so rav1d's 1024 threshold is not supported for
   cache-resident rows. A likely cause on Zen 4 is the 8-way L1D (more than 8
   rows at one column share a set); not confirmed with counters. N1's 4-way L1D
   would predict losses at 16384 and 32768, and none appeared. Memory-resident
   rows (128 MiB) are unstable between runs; on x86 they are also hit at 2048
   and 6144 with 8 rows, and on N1 the bump costs up to 22% at 2048.
   The x86 load-after-store "4K aliasing" is a different effect and the bump
   does not fix it: for a row-in/row-out kernel on Zen 4 AVX-512, a destination
   64 bytes past a multiple of 4096 from the source was the slow placement (3 of
   4 runs, the other placements took 26-36% less time); AVX2, scalar and NEON
   showed nothing. Not measured: u8/u16 samples, more than 12 rows, other
   paddings, Zen 5, Intel, Apple silicon.
5. **GPU uploads need 256.** wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT` is 256
   (`wgpu-types` 30.0.1, `src/lib.rs:116`); buffers meant for texture copies
   need that row pitch or a repack.

What zenpixels does today: `new_simd_aligned(.., simd_align)` takes any power
of two from the caller and uses stride `lcm(bpp, simd_align)`
(`descriptor.rs:1306`). No crate outside zenpixels calls it (searched
2026-10-31); codecs use `aligned_stride` (whole pixels, no SIMD padding). The
lcm rule wastes space whenever bpp is not a power of two (computed as
`round_up(width × bpp, lcm(bpp, 64))` against `round_up(width × bpp, 64)`):

| Format, width, align 64 | Row bytes | lcm rule | Multiple of 64 only |
|---|---|---|---|
| RGB8, 100 px | 300 | 384 (22% padding) | 320 (6%) |
| RGB16, 100 px | 600 | 768 (22%) | 640 (6%) |
| RGB F32, 100 px | 1,200 | 1,344 (11%) | 1,216 (1%) |
| RGB8, 1 px | 3 | 192 | 64 |
| RGB8, 1920 px | 5,760 | 5,760 | 5,760 |

Recommendation (revised 2026-10-31, after the soundness audit and the
measurements):

- **Keep the whole-pixel stride on `PixelSlice`/`PixelBuffer`.** The lcm rule
  is the price of that contract. It costs padding only on narrow rows (at most
  `lcm(bpp, 64) - 1` bytes per row: 191 for RGB8, 0% at 1920 px); typed views,
  `imgref` and downstream `stride / bpp` stay correct.
- **Use 64 for `simd_align` on every CPU tier.** It covers 16-, 32- and
  64-byte vectors and the x86/Neoverse line; the measured gain is on AVX2 and
  AVX-512. Keep the argument for GPU staging (256).
- **When we choose the stride, add 64 bytes if it is a multiple of 4096.**
  Measured: it removes a 19-48% loss for cache-resident 12-row kernels on
  Zen 4 and is neutral for 2-row kernels (on N1 with memory-resident rows it
  costs up to 7% at 4096). Do not extend it to
  multiples of 1024 or 2048: cache-resident rows show no effect, and the
  memory-resident results differ in sign between Zen 4 and N1. The bump is
  not a fix for load-after-store aliasing between a source and a destination.
  For a stride we are given, the cost lands only on kernels that keep more
  than 8 rows live at one column (vertical resize taps, long filters).
- **Single-channel planes have no lcm problem:** the word size (1, 2 or 4)
  divides 64, so a plane row can always start on 64 bytes with under 64 bytes
  of padding. Owned plane storage should do that when it allocates
  (`planar-inventory.md` §6).
- Do not pad rows so kernels can run past `width`: row accessors return exactly
  `width` pixels, so padding cannot replace a tail loop without new API.

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
(`MultiPlaneImage`, `PlaneDescriptor`, `PlaneLayout`) is removed. An earlier
version of this section said no sibling crate still used it; that was wrong.
zenmetrics `master` uses `PlaneLayout` in zenmetrics-abstractions (public
`Planes`/`PlanarFormat`), zenmetrics-browser and vmaf, with a requirement
(`>=0.2.11, <0.4.0`) that admits 0.3.1 (`planar-inventory.md` §3). What exists
instead:

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
