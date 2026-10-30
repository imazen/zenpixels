# YUV at the AV1 and metrics boundaries

> **2026-09-27 update:** checked parts adoption with `take_parts()` and
> `without_buffer()` is implemented. The owner chose to deprecate the entire
> existing planar module now and defer a better video representation. Earlier
> proposals below to expand that module are superseded. See the
> [code-first review](code-review-0.2-and-0.3.md) for executable cases and wanted
> behavior before approving the remaining guards/contracts.

Reviewed 2026-09-27. **Recommendation: add a validated borrowed YCbCr carrier to
zenpixels's `planar` surface, developed with these real callers.** Keep conversion
math in zenpixels-convert or an existing conversion backend. This is a design
recommendation; no new YUV API or sibling-code migration is implemented here.

The previous audit found no users of the existing `MultiPlaneImage` and
`PlaneDescriptor` names. That does not mean no planar consumers exist: these
new consumers currently pass their own plane slices. They change the priority.

## Current consumers, verified in local source

Paths below are relative to `/home/lilith/work/zen`. VMAF was inspected in the
explicitly identified `zenmetrics--cvvdpmodeb` checkout: its new crate is absent
from the primary zenmetrics checkout inspected here. These are local source
contracts, not claims that all APIs have been published.

| Consumer and source | Current boundary | Migration implication |
|---|---|---|
| SVT: `zenav1-svt/rust/svtav1/src/avif.rs`, `AvifEncoder::encode_yuv420` | Three `&[u8]` planes, width/height, explicit luma stride; U/V tight at ceil(w/2) × ceil(h/2). | Borrow existing planes; preserve odd-size geometry. Full independent U/V strides require a backend change or explicit repacking. |
| SVT: `rust/svtav1/src/animation.rs`, `AnimationFrame<'a, T>` | U8 or native unshifted U16 10-bit Y/U/V; luma stride, tight chroma and optional tight alpha; duration separate. | Share image interpretation and plane geometry; keep frame duration/timescale in codec/video APIs. |
| AOM: `aom-rs/crates/aom-encode/src/key_frame.rs`, `KeyFramePlanes` | Tight `&[u16]` Y/U/V at every supported bit depth, including 8-bit values stored in U16. Dimensions/depth/subsampling live in `KeyFrameConfig`. | U16 storage must not imply 16 significant bits. Borrow and validate against config without widening/copying existing U16 planes. |
| AOM: `aom-rs/crates/aom-decode/src/frame.rs`, `FrameDecode` | Owned tight U16 planes; dimensions, chroma dimensions, depth, subsampling, monochrome, matrix, primaries, transfer, range and chroma position. | Expose a borrowing adapter that carries these facts together. Preserve ownership of existing Vecs. The zenav1-aom facade re-exports this stack. |
| VMAF: `zenmetrics--cvvdpmodeb/crates/vmaf/src/score.rs`, `Yuv420Frame` / `VmafV1Scorer` | Tight U16 Y/U/V; width/height/bit depth on scorer. | Wrap shared views; check pair geometry and sample interpretation. The v1 extractor calls `speed_v1_chroma_420` as well as luma features: this is **not luma-only**. |
| CVVDP: `zenmetrics/crates/cvvdp/src/video.rs`, `VideoScorer` | RGB HWC or RGB CHW; U8, U16 normalized by 65535, or display-encoded F32; linear mode takes absolute luminance units. | `FrameLayout::Planar` means RGB planes, not YCbCr. Convert explicitly using the correct matrix, range, chroma reconstruction and display color model. |

The existing callers have different supported subsets. A shared representation
does not make SVT, AOM and every VMAF model accept all depths/subsampling modes.
Adapters must report unsupported input instead of silently resampling or reducing
precision. Keep model-specific VMAF validation and CVVDP display parameters.

## The minimum useful contract

Use a provisional name such as `YuvSlice<'a>` under `planar`, backed by a small
borrowed plane view. Decide the final spelling together with the adapters. Favor
one frame-level U8/U16 dispatch and simple inner loops over a large type-level
color/precision taxonomy. No Vec of descriptors, allocation or Arc clone should
be required just to borrow three planes.

The carrier needs:

1. Full-resolution width/height, explicit Y/Cb/Cr roles and subsampling. Chroma
   extents use checked ceil division. Represent monochrome without fake chroma;
   optional full-resolution alpha has a concrete SVT caller.
2. An independent validated view for each plane: slice, dimensions and stride.
   Keep public stride units consistent with zenpixels (**bytes**); typed sample
   adapters convert units explicitly and check divisibility. A view's slice may
   start at its first sample; owned storage extraction must also preserve offsets.
3. Storage type **separate from valid bit depth**: an AOM 8-bit frame in U16 is
   different from normalized U16 RGB. Native endianness is explicit. Initially
   support the callers' unshifted samples; distinguish or explicitly reject
   MSB-aligned packing. Never treat a ten-bit code as a normalized sixteen-bit code.
4. Matrix coefficients, full/limited range, primaries, transfer and chroma sample
   location, with unknowns preserved. Reuse CICP where appropriate, but CICP alone
   does not carry chroma location or all frame geometry. Distinguish matrix codes
   whose equations differ; do not silently collapse BT.2020 constant-luminance and
   non-constant-luminance into one conversion operation.
5. A borrowing current-color/context relationship consistent with the proposed
   resolved encoding contract. Source provenance and mastering metadata are not
   substitutes for the current sample interpretation or relative-linear anchor.

These distinctions match the separate color-range, chroma-position and color
description fields of the [AV1 bitstream specification](https://aomediacodec.github.io/av1-spec/).
The [ColorVideoVDP documentation](https://github.com/gfxdisp/ColorVideoVDP) separately
defines input color and transfer through its display model. Those facts support
keeping signal storage and display interpretation explicit.

Checked construction validates geometry, stride, alignment and supported encoding
combinations without scanning the whole image. Verifying every sample fits its
declared bit depth is O(samples); keep an explicit scan or codec enforcement where
needed. Do not advertise value validation as free. Crop origins must preserve
subsampling phase, or the crop must be refused; odd luma offsets cannot silently
reuse a zero-phase chroma interpretation.

## Why not just add PixelFormat::Yuv420?

`PixelFormat` describes interleaved, constant-bytes-per-pixel storage. A subsampled
three-plane frame has independent pointers/strides and ceil-rounded extents. A
new enum variant cannot supply those facts, and would make current buffer/row
arithmetic ambiguous. Each component plane can still use established sample
layout machinery; its meaning belongs to the enclosing YCbCr description.

The current `MultiPlaneImage` owns `Vec<PixelBuffer>`; `PlaneLayout::Planar` owns
a Vec of descriptors. It is not the zero-allocation borrowing boundary these
callers need. Its convenience YCbCr factories also default to BT.601, and its
current descriptor does not express significant bits/packing or chroma siting.
Do not make callers fabricate grayscale color descriptors for each Y/Cb/Cr plane
or convert Vec<U16> into Vec<U8> just to enter that container.

Retain the published machinery in the 0.2 bridge. Repair checked construction and
mutation invariants where compatible, and add the new view with real adopters.
Use the same adopted signatures in 0.3.1. An owned universal planar container is
not a prerequisite: borrowing existing decoder-owned planes solves the immediate
boundary. Later ownership parts must preserve typed allocations and return them
on failed adoption under the crate's safe-Rust policy.

## Copies and conversion

- **AOM → VMAF:** matching tight U16 4:2:0 planes can be borrowed directly, after
  compatible geometry/depth interpretation is checked. Do not insert an RGB
  roundtrip, range expansion or model-changing resampling just to share a type.
- **Strided decoder → codec/metric:** the carrier preserves padding. If the backend
  still requires tight planes, teach its row access about stride or perform a
  named repack. A zero-copy wrapper does not eliminate that backend requirement.
- **U8 SVT → U16-only backend:** widening really does require work/storage unless
  the backend gains a U8 input path. Do not claim the shared carrier makes it free.
- **YUV → CVVDP:** convert YCbCr to the RGB encoding expected by the display model,
  with explicit chroma reconstruction and range handling. Prefer caller-owned
  scratch and a precise F32 path where appropriate. Passing unshifted 10-bit
  codes directly to the current U16 entry point divides by 65535 instead of the
  intended code scale; even before YCbCr/RGB mismatch, that is roughly a 64×
  normalization error. Blind shifting is not a complete limited-range conversion.
- **40–100 MB owned images:** carry borrowed plane views through analysis and
  encoding, and move original ownership/parts when needed. Keep intrinsic codec
  reference frames and metric temporal history separate from avoidable interface
  copies; the carrier cannot remove memory their algorithms require.

YCbCr conversion needs a named chroma filter, siting/edge behavior, matrix/range,
precision and clipping contract. Core should only describe/validate. Prototype
the conversion adapter against existing kernels before adding a second math
implementation or introducing any new core dependency.

## Streaming and adoption order

Start with whole-frame **borrowed** views; they do not require buffering an image
again. Existing codec providers can lend row/strip windows over those planes.
A 4:2:0 strip groups luma rows with the corresponding chroma row and handles odd
tails; reconstruction filters may need neighboring chroma samples. A bare
`Iterator<Item = (&[u8], &[u8], &[u8])>` hides these requirements and cannot model
every decoder's lending lifetime/error behavior. Reuse the existing codec-level
fallible delivery contract and add an adapter once a streaming caller needs it.

Suggested implementation order for the broader PRs:

1. Borrow AOM's current FrameDecode and feed the new VMAF frame boundary without
   conversion; assert identical scores and unchanged input pointers.
2. Add SVT U8/U16 adapters, independent-stride support where its backend accepts
   it, explicit refusals otherwise, and odd-size/monochrome/alpha tests.
3. Add YUV-to-CVVDP adaptation with an independently verified matrix/range/display
   interpretation and reference parity. Preserve CVVDP's RGB entry points.
4. Exercise minimal/default/planar features on the complete 0.2 bridge and 0.3.1
   with identical consumer source, and publish these examples in the planar docs.

This supplies a small, useful foundation without coupling zenpixels compilation
to either AV1 implementation or the metrics stack.
