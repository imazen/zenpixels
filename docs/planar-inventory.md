# Planar representations across the zen codecs

Written 2026-10-31. Inventories every planar (one array per component)
representation in the zen codecs and the crates around them, sketches a shared
format, and lists the duplication. Proposals only; nothing here is implemented.
An interactive version is a private artifact (owner-visible until shared):
https://claude.ai/artifact/733cxv3SiQCzkQsJ7bKWSW

**Sources.** Local primary checkouts under `~/work/zen` (jj workspaces named
`<repo>--*` excluded), read on 2026-10-31. Branch caveats: zencodec's working
copy was on `frame-plane-ablation`; every zencodec fact below was re-checked
against `origin/main` (`a51d411`). zenmetrics facts are from `origin/master`
(`c5271f16`). Published state from crates.io the same day: zenpixels and
zenpixels-convert 0.2.16, zencodec 0.1.26.

## 1. Inventory

### Codec boundary types (what a caller sees)

| Crate | Type | Direction | Storage | Geometry and metadata |
|---|---|---|---|---|
| zenjpeg | `YCbCrPlanes<'a>` (`encode/encoder_types.rs:620`) | encode input | borrowed `&[f32]` ×3, stride per plane | no dimensions or subsampling in the type; Y 0..255, Cb/Cr centered −128..127 |
| zenjpeg | `YCbCrPlanarEncoder` (`encode/byte_encoders.rs:1217`) | encode input, streaming | buffers f32 strips | subsampling from config |
| zenjpeg | `DecodedYCbCr` (`decode/image.rs:256`) | decode output | owned `Vec<f32>` ×3, tight | full resolution (chroma upsampled); **all three** centered −128..127; ICC |
| zenjpeg | `read_rows_ycbcr_native_i16` (`decode/scanline.rs:2615`) | decode output, streaming | caller `&mut [i16]`, one stride for Y, one shared for Cb/Cr | native chroma resolution |
| zenwebp | `YuvPlanes` (`decoder/api.rs:2237`) | decode output | owned `Vec<u8>` ×3, tight | 4:2:0 only; plane sizes; no range or matrix field |
| zenwebp | `PixelLayout::Yuv420` (`encoder/api.rs:248`) | encode input | one `&[u8]` holding Y, U, V back to back | 4:2:0 only |
| rav1d-safe | `Frame`, `Planes8/16`, `PlaneView8/16` (`managed.rs:1068-1534`) | decode output | borrowed from the decoder's pool, strided | `PixelLayout` I400/420/422/444, `MatrixCoefficients`, `ColorRange` |
| zenavif | `DecodedYuv` (`decode_av1.rs:458`) | decode seam (rav1d or zenav1-aom) | owned `Vec<u16>` ×3, tight | bit depth, subsampling shifts, monochrome |
| zenavif | `media.rs` `Av1Picture` | decode output | rav1d planes mapped into `zencodec::frame` without copying | `Subsampling`, `ChromaSiting`, `SampleEncoding` |
| zenrav1e | `v_frame::Frame<T>` (external crate `v_frame` 0.3.7) | encode input | owned, padded, 64-byte aligned | per-plane padding |
| zenav1-svt | `svtav1::Frame` (`svtav1/src/lib.rs:177`) | encode input | owned `Vec<u8>` ×3, stride per plane | 8-bit only; no subsampling field |
| zenav1-svt | `RgbaFrame` (`svtav1/src/rgba.rs:27`) | encode input | RGBA8 converted internally | always BT.601 limited; the doc warns that a different encoder CICP would mislabel the samples |
| zenav1-aom | `KeyFramePlanes<'a>` (`aom-encode/src/key_frame.rs:1178`) | encode input | borrowed `&[u16]` ×3, tight | all geometry in `KeyFrameConfig` |
| heic | `DecodedFrame` (`heic-core/src/frame.rs:48`) | decode output | owned `Vec<u16>` ×3 + optional alpha | conformance-window crop; CICP and range as raw `u8`/`bool` |
| heic | `OutputPlanes` (`heic-backend-d3d11va/src/dxva_read.rs:46`) | hardware decode readback | owned | backend-specific |
| jxl-encoder | `vardct::YCbCrPlanes` (`chroma_subsampling.rs:75`) | internal, built through zenyuv | owned `Vec<u8>` ×3, tight | chroma width/height |
| zenjxl-decoder | `flush_pixels(&mut [JxlOutputBuffer])` (`api/decoder.rs:355`) | decode output | one caller buffer for color plus one per extra channel | extra-channel metadata in the header |
| zencodec | `frame::{Frame, Picture, MappedPlanes, PlaneView, PlaneSamples, PlaneRole, Subsampling, ChromaSiting, Crop}` (`src/frame.rs`, `src/frame_storage.rs`, main) | shared, codec-agnostic | borrowed views over any owner | `SampleEncoding`, roles, crop, timing; unpublished |
| zencodec-media | `OwnedPlane`, `OwnedPlanes` (`zencodec-media/src/frame.rs:32,46`, main) | shared owned storage | `Vec<u8>`/`Vec<u16>` or accounted shared storage | one `SampleEncoding` for all planes; unpublished |

### Planar file layouts flattened at decode

| Crate | Source layout | What happens |
|---|---|---|
| zentiff | `PlanarConfiguration = 2`; YCbCr photometric | planes interleaved after reading (`decode.rs:948`); YCbCr converted by its own `YCbCrToRgb` from the TIFF tags (`decode.rs:1006`) |
| zenpsd | channel-planar image data | interleaved into a `PixelBuffer` (`parser.rs:1032`) |
| zenexr | per-channel blocks (crate `exr` 1.74) | interleaved by `exr`'s channel callbacks (`lib.rs:169`) |
| zenjp2 | per-component samples (crate `hayro-jpeg2000` 0.4) | interleaved in `codec.rs:380` |

### Shared or general carriers that already exist

| Crate | Type | Status | Users found |
|---|---|---|---|
| zenpixels | `sample::CodePlane`, `SampleEncoding`, `ChromaSubsampling`, `ChromaSiting` | main and 0.3.1; unpublished | zencodec `src/image.rs`, zencodec perf crates |
| zenpixels-convert | `ycbcr::YcbcrToRgbF32` | main; unpublished; BT.709 matrix + sRGB transfer, 8/10/12-bit only | zencodec perf crates |
| zenpixels 0.2 | `planar` module (`PlaneLayout`, `MultiPlaneImage`, …) | deprecated in 0.2.17, removed in 0.3.1 | **zenmetrics** (see §3) |
| zenmetrics-abstractions | `Planes<'a>`, `PlanarFormat` (`input.rs:105,115`) | public, built on zenpixels 0.2 `PlaneLayout` | zenmetrics-browser, vmaf |
| zqi (imij) | `PlaneDescriptor` (`planes/mod.rs:135`) | private format crate | name, sample type, precision, role, origin, group, CICP, ICC, pyramid levels |
| zenlayout | `Subsampling`, `PlaneLayout` (`plan.rs:637,679`) | public | JPEG/video encoder geometry (block and MCU grid) |

### Planar working buffers inside one crate (not sharing targets)

- AV1 encoders and decoders: zenrav1e `PlaneRegion*`, zenav1-svt `PaddedPlaneT` and about 25 more plane structs, zenav1-aom `TfPlane`, `SsePlane`, `LrPlaneInput` and others, rav1d internal pictures. They carry codec padding and block geometry and should stay private.
- zenjxl-decoder render-pipeline planes and stages (`YcbcrToRgbStage`, chroma upsampling); jxl-encoder `Image3`-style f32 planes.
- zenfilters `OklabPlanes` (owned f32 L/a/b and optional alpha, tight, `planes.rs:11`).
- Metrics: zensim, fast-ssim2, butteraugli (planar XYB f32); in zenmetrics, `Planes` structs in gmsd (two), fsim, vsi and pieapp, `Plane` in gmsd and haarpsi, and `Yuv420Frame` in vmaf.
- zenresize `PlaneResizer` (single i16 plane). It is public, but no crate outside zenresize uses it.
- ultrahdr `GainMap` (u8, 1 or 3 channels). It stays separate on purpose: its bytes are log2 gain, not color.

## 2. Duplication

### Vocabulary enums: 14 subsampling, 4 siting, 9 matrix or range

- **Subsampling:**
  - zenpixels `ChromaSubsampling`, zencodec `Subsampling`
  - zenjpeg `Subsampling` and `encode::ChromaSubsampling`, mozjpeg-rs `Subsampling`
  - zenavif `yuv_convert::ChromaSubsampling`, `image::ChromaSampling`, `detect::ChromaSampling`
  - zenavif-parse and zenavif-serialize `ChromaSubsampling` structs
  - zenav1-svt `avif::ChromaSubsampling`, jxl-encoder `ChromaSubsampling`, zenlayout `Subsampling`, rav1d-safe `PixelLayout`
- **Siting:** zenpixels `ChromaSiting`, zencodec `ChromaSiting` (same four variants), zenrav1e `ChromaSamplePosition`, zqi `ChromaSiting`.
- **Matrix and range:**
  - zenyuv `Matrix`/`Range`
  - zenavif `YuvMatrix`/`YuvRange`/`ColorRange`
  - rav1d-safe `MatrixCoefficients`/`ColorRange`
  - zenrav1e `MatrixCoefficients`, zenavif-serialize `MatrixCoefficients`
  - heic's raw `u8` fields

Each mapping between two of these is a place to drop a case. zenavif's mapping into zencodec turns every unrecognized rav1d layout into `Yuv444` (`media.rs:294`).

**The two shared definitions duplicate each other.** zenpixels and zencodec `main` both define subsampling and siting enums. The siting enums have the same four variants; zencodec's subsampling adds `Mono`. Neither is published, so they can still be merged without a break.

### Plane carriers

There are three general-purpose borrowed plane types:
- zenpixels `CodePlane`
- zencodec `PlaneView`
- zenmetrics `Planes`, which is built on the planar API that 0.3.1 removes

Codec boundary structs (the first table) repeat the same three-field shape (y/u/v or y/cb/cr) with differing stride, padding, offset and metadata rules. Most carry no siting, matrix or range description; only rav1d-safe's `Frame`, heic's `DecodedFrame` (as raw integers) and the zencodec frame types do.

### YUV ↔ RGB kernels: 10 implementations

Module sizes are line counts including tests (`wc -l`), not a measure of duplication:

| Crate | Module | Lines | Can it share? |
|---|---|---|---|
| zenjpeg/zenyuv | whole crate (encode, decode, sharp YUV; magetypes v3, NEON, wasm128, scalar; no AVX-512 tier) | 5,910 | already shared: used by zenjpeg, zenwebp, jxl-encoder, zqi, zenmetrics-cli |
| zenjpeg | `color/ycbcr.rs`, `color/fast_yuv.rs` | 4,095 | decode side must match jpegli/libjpeg-turbo bit for bit; encode side could use zenyuv |
| zenwebp | `decoder/yuv.rs`, `yuv_fused.rs`, `vp8v2/yuv_exact.rs` | 6,795 | no: the decoder matches libwebp's fancy upsampler bit for bit |
| zenavif | `yuv_convert*.rs` (4 variants) + the `yuv` crate 0.8.17 | 5,023 | keep: its canonical recipe is defined by its own fixed-point formula and exact {9,3,3,1}/16 chroma interpolation ("the formula IS the definition", `yuv_convert.rs`), and the `libyuv` variants exist to match libavif's path |
| heic | `heic-core/src/color_convert.rs` | 789 | keep: matches libheif's `Op_YCbCr420_to_RGB24` (nearest-neighbor chroma, ×256 fixed point; `heic-core/src/frame.rs:355`) |
| zenav1-svt | `svtav1/src/rgba.rs` (BT.601 limited only) | 166 | could share: encoder-side input conversion, not defined by AV1 |
| zentiff | `YCbCrToRgb` (`decode.rs:1006`) | small | keep: defined by the TIFF `YCbCrCoefficients` and `ReferenceBlackWhite` tags |
| zenjxl-decoder | `render/stages/ycbcr.rs`, `chroma_upsample.rs` | 390 | no: the JXL spec defines it |
| mozjpeg-rs | `color.rs`, `color_avx2.rs` | 1,623 | no: it is ported for parity with C mozjpeg |
| zenpixels-convert | `ycbcr.rs` | 299 | the descriptor-checked route; narrow today (BT.709 + sRGB only) |

Decision (2026-10-31): format- or reference-specific conversions stay where they are. On a closer read, zenavif, heic and zentiff each implement a definition tied to their format or reference decoder, so only encoder-side RGB → YCbCr (zenav1-svt, zenjpeg's encode side) is a sharing candidate, and zenyuv already serves most encoders.

Found while reading heic: matrix codes other than 0 (identity), 1, 9 and the BT.601 family fall through to BT.601 (`heic-core/src/color_convert.rs:34-46`), so FCC (4), SMPTE 240M (7) or chromaticity-derived (12/13) streams decode with the wrong matrix and no error. Not fixed here (heic is outside this task).

### Interleave ↔ planar kernels

Hand-written in:
- zenjxl-decoder: `deinterleave_2/3/4`, `interleave_2/3/4` in `render/simd_utils.rs`
- zenfilters: `deinterleave3/4`
- zenmetrics-api: `deinterleave_f32`, `deinterleave_rgb`
- cvvdp: `interleaved_to_planar`
- jxl-encoder: `deinterleave_rgb`
- butteraugli: `to_interleaved`, `from_planes`
- garb: `rgb24_to_planes_f32`, `rgb48_to_planes_f32` — only direction, only F32 output

### Inconsistencies found

- **zenjpeg's planar encoder and decoder disagree on luma range.** The encoder takes Y in 0..255; `DecodedYCbCr` returns Y centered at −128..127. Planar decode → encode in one crate needs a +128 shift on Y only.
- **zenav1-svt's RGBA path always writes BT.601 limited samples.** Its encoder CICP can say otherwise; its own docs warn about this.
- **Three codec outputs don't describe their own planes.** heic carries CICP as raw `u8` fields; zenwebp's `YuvPlanes` and zenavif's `DecodedYuv` carry no range or matrix at all.

## 3. Finding: zenmetrics breaks on zenpixels 0.3.1

zenmetrics `master` (`c5271f16`) uses zenpixels' 0.2 `PlaneLayout`:
- `crates/zenmetrics-abstractions/src/input.rs:13`, in its public `Planes`/`PlanarFormat`
- `crates/zenmetrics-browser/src/lib.rs`
- `crates/vmaf/src/metric.rs`

Its requirement `zenpixels = { version = ">=0.2.11, <0.4.0", features = ["planar"] }` admits 0.3.1. The feature name survives as a no-op, but `PlaneLayout` is removed, so a `cargo update` after 0.3.1 publishes fails to compile. Its lockfile pins 0.2.16 today. `release-plan-0.2.17-and-0.3.1.md` §3.8 did not list this migration; this doc adds it there.

**Fix (zenmetrics PR #104):** zenmetrics-abstractions gets local `PlaneLayout`, `PlaneSpec` and `PlaneRole` types with the same facts; callers change only their import path. Local types rather than the new zenpixels vocabulary, because that vocabulary is unpublished and zenmetrics resolves zenpixels 0.2.16. Abstractions and vmaf tests pass on 0.2.16 and with zenpixels patched to 0.3.1. zenmetrics-browser still fails on 0.3.1 through zencodec 0.1.27, which is zencodec's own migration.

## 4. What a shared format would look like

Build on what exists instead of adding a fourth carrier:

1. **One vocabulary, in zenpixels:**
   - `ChromaSubsampling` (gains `Mono` and `chroma_size`) and `ChromaSiting`; zencodec deletes its copies and names these. In review: zenpixels #99 (0.3.1) and #100 (0.2.17), zencodec #191 (stacked on #190).
   - Matrix, range, primaries and transfer stay in `Cicp`/`PixelDescriptor` fields; codecs map their stream's values once, at the boundary.
2. **One borrowed plane:** `CodePlane` for U8/U16 codes. zencodec's `PlaneView` is the same idea (`SampleEncoding` + width/height/byte stride); merge them, with zencodec keeping only what is frame-specific (roles, crop, timing, ownership).
3. **One set-of-planes view:** zencodec's `Frame`/`MappedPlanes` contract, a plane list with roles + subsampling + siting + `Cicp` + crop. zenavif already maps rav1d's pool into it without copying.
4. **Owned storage stays where the allocator is:** zencodec-media `OwnedPlanes` for software decoders, extended as §6 describes; codecs keep their internal padded planes.
5. **F32 working planes** (zenfilters, metrics) stay crate-private until two crates exchange them; then add an F32 plane view beside `CodePlane`.
6. **Kernels in garb and zenyuv:**
   - garb: interleave ↔ planar in both directions for U8/U16/F32, and strided.
   - zenyuv: encoder-side RGB → YCbCr kernels, taking `CodePlane`s. Decoder-side conversions that a format or reference decoder defines stay in their codec (§2).
   - zenpixels-convert: the descriptor-checked route on top.

What it would bring:

- **Decode → encode without repacking:** AVIF → AVIF, JPEG → JXL (JXL's YCbCr mode), and HEIC → AVIF can pass native planes when the encoder accepts the same subsampling and depth. Today each pair needs glue code or goes through RGB.
- **Metrics on native YUV:** VMAF-style scorers could take decoder planes directly instead of their own `Yuv420Frame`.
- **One definition of siting and range handling:** each enum mapping and each conversion kernel is a place where a 4:2:0 siting or a limited-range case goes wrong; there would be one to test.
- **SIMD written once where the math is ours:** zenyuv already has v3 (AVX2), NEON, wasm128 and scalar arms for encoder-side conversion.

What it would not do: conversions a format or reference decoder defines (WebP, JPEG, JXL, mozjpeg-rs parity, AVIF's canonical and libyuv recipes, HEIC's libheif parity, TIFF's tag-defined YCbCr) keep their kernels; those codecs can still *emit* the shared planes.

## 5. Suggested order

1. Merge the subsampling/siting enums before zenpixels 0.3.1 or zencodec's frame module publishes. In review: zenpixels #99, #100; zencodec #191.
2. Move zenmetrics off the removed planar API. In review: zenmetrics #104.
3. Map heic `DecodedFrame` and zenwebp `YuvPlanes` into the zencodec frame contract the way zenavif does.
4. Owned planes: add F32 storage and a start offset to zencodec-media `OwnedPlanes` when the first adopting codec (heic `Vec<u16>`, zenjpeg `Vec<f32>`) moves onto it (§6).
5. Add the missing garb interleave kernels; switch zenfilters, zenmetrics-api and jxl-encoder to them.
6. Make zenjpeg's planar decode and encode agree on Y range, or document the shift at both ends.

## 6. Owned planes: what works for every producer and consumer

How planes are produced and consumed today:

- **Owned `Vec`s handed back:** heic `Vec<u16>` (tight), zenavif `DecodedYuv` `Vec<u16>`, zenwebp `YuvPlanes` `Vec<u8>`, zenjpeg `DecodedYCbCr` `Vec<f32>`, zenfilters `OklabPlanes` `Vec<f32>`.
- **Pooled, lent storage:** rav1d (64-byte aligned, width rounded up to 128 pixels, stride bumped off multiples of 1024).
- **Caller buffers:** zenjxl-decoder (`JxlOutputBuffer` per channel), zenjpeg `read_rows_ycbcr_native_i16`.
- **Encoders take:** zenav1-aom `&[u16]` (tight), zenav1-svt `Vec<u8>` + stride, zenrav1e copies into `v_frame` (padded, 64-byte aligned) regardless.

Storage options:

| Storage | Adopts a codec's `Vec` without copying | 64-byte rows when we allocate | Typed rows |
|---|---|---|---|
| `Vec<u8>` + start offset (today's `PixelBuffer`) | `Vec<u8>` only. `Vec<u16>`/`Vec<f32>` copy: `bytemuck::allocation::try_cast_vec::<u16, u8>` fails because the deallocation layout differs (checked 2026-10-31) | yes | checked cast per row |
| Word-typed enum `U8(Vec<u8>)`/`U16(Vec<u16>)` (zencodec-media `PlaneData` today) | yes | no, allocator alignment only | the slice itself, no cast |
| Word-typed enum + start offset in words | yes, at offset 0 | yes: start at the first 64-byte-aligned word | the slice itself, no cast |
| `Vec<Line>` with `#[repr(C, align(64))] struct Line([u8; 64])` | no | always | `bytemuck::cast_slice` to `u8`/`u16`/`f32`, which cannot fail; compiles under `forbid(unsafe_code)` with bytemuck 1.24's `Pod` derive (checked) |

Recommendation: **word-typed storage with a start offset**, by extending zencodec-media `OwnedPlanes` (it already has U8/U16 and accounted shared storage) instead of adding a type to zenpixels:

- Words U8, U16, F32 (F16 only with a consumer), plus the accounted variants.
- Adopting a codec's `Vec` is free (offset 0): heic, zenavif, zenwebp, zenjpeg and zenfilters outputs move in without a copy.
- Allocating puts the first sample on 64 bytes and rounds the stride up to 64 bytes, plus 64 when that lands on a multiple of 4096. A plane's word size divides 64, so rounding costs under 64 bytes per row (under 128 with the 4096 bump). The 64-byte start is worth up to 1.5x on AVX2/AVX-512 cache-resident rows (`benchmarks/row_alignment_2026-10-31.md`); the bump is worth 19-48% to cache-resident 12-row vertical filters on Zen 4 and +6..+10% (at 8192 only) on Neoverse-N1 (`benchmarks/stride_aliasing_2026-11-01.md`).
- Rows are typed slices with no casting; byte views use `bytemuck::cast_slice` from the word slice, which cannot fail. The only checked casts left are on byte buffers a codec hands over (for example `&[u8]` holding `u16` words): `try_cast_slice`, and copy when misaligned.
- The word type is per plane. Alpha and extra channels can differ from color planes (JXL extra channels do); per-plane is the superset of the "frame-wide versus per-plane encoding" question zencodec #190 leaves open.
- Borrowed side: `CodePlane` for U8/U16 codes. Add an F32 plane view when the first F32 planes cross crates (zenjpeg `DecodedYCbCr` into jxl-encoder, or metrics).

Re-check the counts:

```sh
cd ~/work/zen
rg -n 'pub enum (ChromaSubsampling|Subsampling|ChromaSampling|ChromaSiting|ChromaSamplePosition|PixelLayout|YuvRange|YuvMatrix|ColorRange|MatrixCoefficients|Matrix|Range)\b' -t rust \
  --glob '!**/target/**' --glob '!**/tests/**' --glob '!**/benches/**' --glob '!**/reference/**' \
  zenjpeg mozjpeg-rs zenwebp rav1d-safe zenavif zenrav1e zenav1-svt zenav1-aom heic zenjxl-decoder jxl-encoder zenextras zqi zencodec zenpipe zenpixels zenmetrics
rg -l -t rust --glob '!**/target/**' 'PlaneLayout' zenmetrics
```
