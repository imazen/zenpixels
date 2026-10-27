# Explicit integer sample encoding

`sample::SampleEncoding` is the small additive bridge for the last 0.2.x release.
It describes one unsigned code per U8/U16 storage word using private fields and
a checked constant constructor. It introduces no default or sample-value scan.

The concrete consumer is `zencodec_media::plane::Plane` and its native AV1 adapter
(`zencodec/media/src/av1.rs`). The adapter retains rav1d-safe's frame allocation and
mapped borrow guards. Its corpus verifies all samples in 48 independently
decoded lossless AV1 cases: 8/10/12-bit, full/narrow range, mono/420/422/444,
even/odd dimensions. Thus native code 1023 remains 1023 in U16 and is explicitly
ten-bit; no RGB16 descriptor silently declares that value to be normalized U16.

| Representation | Storage | Code bits | Bit shift |
|---|---|---|---|
| Ordinary byte component | U8 | 8 | 0 |
| Native AV1 ten-bit component | U16 | 10 | 0 |
| Native AV1 twelve-bit component | U16 | 12 | 0 |
| P010 component word | U16 | 10 | 6 |
| Full sixteen-bit component | U16 | 16 | 0 |

This describes component words, not the complete P010 format: plane grouping,
interleaving, chroma position, dimensions and byte order still need a format
adapter. Typed U16 words are native-endian. Byte-backed formats must declare
and decode byte order before interpreting the word.

Read a code by shifting and masking. Padding bits do not affect the code;
writers zero them. Strict padding conformance, if required by a format, is a
separate explicit validation pass. Storage and positive code depth must fit;
the implementation checks subtraction rather than overflowing depth + shift.

## Requirements for the breaking 0.3 API

1. Every native integer component plane carries its current sample encoding.
   U16 alone must not imply ten-bit, twelve-bit, left alignment, or normalization.
   Code depth describes current samples; original precision belongs to provenance.
2. Existing RGB/gray U16 remains in its existing sixteen-bit numerical domain.
   Adapting a native ten-bit component to it requires an explicit conversion.
   Packing ten-bit white into the high bits yields 65472; full-range numerical
   widening yields 65535. These operations are distinct and must stay distinct.
3. Range is independent of storage and code depth. Narrow code endpoints scale
   with depth; alpha coverage has its own interpretation and is not video range.
   Do not infer any of these from the maximum sample observed in a frame.
4. Validate geometry and interpretation on construction without scanning pixels.
   Per-sample format conformance is an optional explicit scan, never a hidden
   cost in creating a view, cloning a frame, or inspecting a descriptor.
5. Preserve unknown raw color codes, chroma position, and source claims in the
   media layer. A conversion must validate the particular information it needs;
   unknown signaling must not quietly become centered chroma or sRGB.
6. Keep ownership and mapping separate: a backend frame can outlive its decoder,
   and guard-owned mapped views borrow the mapping. A borrow from a temporary
   backend guard cannot be returned as if it borrowed only the frame owner.

These mandates do not freeze a new multi-plane ownership or video codec API in
zenpixels. The working prototype lives in the `media/` workspace of the zencodec repository
(package `zencodec-media`) until its conversions,
animation, timestamps, I/O and cross-codec behavior have been exercised together.
The deprecated `planar` module remains available for the 0.2 bridge; migration
and removal in 0.3 must include zenfilters' public `PlaneMask` consumers.

See the [U16 matrix](u16-contract-matrix.md) for existing normalized RGB/gray
arithmetic and the [implementation ledger](implementation-status-0.2-and-0.3.md)
for the broader release work. This change does not complete that entire ledger.

## Native signed and wider samples

TIFF selection and [IMIJ's sample model](https://github.com/imazen/zenpixels/issues/83)
need I8, I16, U32, I32, U64, I64 and F64 storage. `ChannelType` and
`PixelFormat` now describe those types in gray, gray-alpha, RGB and RGBA
layouts. They use the existing type-erased `PixelBuffer` / `PixelSlice`
interfaces; they do not introduce another buffer owner. Multi-byte samples
are native-endian, as for the existing U16/F32 storage. A file adapter must
decode the file byte order first.

These are native sample codes. Signed values have no implicit normalization
or display range. The new format descriptors default to unknown transfer;
adapters must retain the actual interpretation separately or attach known
color information. The existing `SampleEncoding` bridge still accepts only
unsigned U8/U16 words; this change does not extend it to signed scientific
scales, per-channel significant bits or arbitrary band layouts.

`just native-samples-check` tests strided buffer creation, owned handoff,
cropping and identity row copying for all seven new types and four layouts.
The fixtures include integer extrema, values beyond exact F32/F64 integer
precision, signed zero, subnormal samples and NaN payload bits. Identity
carriage preserves bytes. Built-in non-identity conversion, HDR tone mapping
and the moxcms adapter refuse these types before running unsupported math.
A separately supplied CMS may accept a format it actually implements.

Format negotiation remains an advisory cost ranking. Selecting a descriptor
does not prove a conversion exists: constructing the converter is the runtime
authority. The native TIFF consumer in zencodec's `perf/spatial-validation`
is the integration gate for using these types in the proposed collection API.
Typed `Pixel` implementations and signed/wide numerical conversion kernels
are outside this addition.

External codecs construct `PixelBufferParts::new(data, width, height, stride,
descriptor)` and call `PixelBuffer::try_from_parts`. This adopts the original
allocation after checking geometry and alignment. The constructor was needed
by the TIFF integration: the non-exhaustive parts structure previously could
only be obtained from a buffer that zenpixels already owned. Failed adoption
retains the native allocation for recovery through `FromPartsError`.

## Native F32 CMYK layers

The raw JXL consumer in zencodec's `perf/spatial-validation/src/jxl_layers.rs`
needs four native F32 components. CMY plus separate black cannot be labeled RGB.
`PixelFormat::CmykF32` is conventional ICC device polarity (zero means no ink).
`PixelFormat::CmykF32Inverted` is the reversed polarity emitted by native JPEG XL
(one means no ink), as specified by
[libjxl's CMS interface](https://github.com/libjxl/libjxl/blob/v0.11.1/lib/include/jxl/cms_interface.h#L119).
They share channel layout and size, but are distinct formats
at every dispatch boundary. Physical `from_parts(F32, Cmyk, None)` selects the
conventional format; the JXL adapter must select the inverted variant explicitly.

This is deliberately carried in the format passed to converters, not in an
optional provenance extension that a converter can overlook. A general scale
or polarity field on every descriptor would also need to cross the existing
CMS interfaces, which receive a `PixelFormat`. That broader sample-domain design
remains a 0.3 review question; these two formats make the current consumer honest
without changing buffer ownership or adding per-pixel work.

Both formats have four channels, 16 bytes per pixel and no alpha. Retain the
source ICC context and any separate alpha plane. The source ICC consumes
conventional device values; an inverted format requires a qualified conversion
before applying it. Changing the enum alone does not convert samples. Even
`1.0 - value` can round away F32 bits, so native carriage must not silently
perform that operation or claim that it is lossless.

Tests preserve strided rows, padding, allocation identity, ICC context, signed
zero, subnormals and NaN payload bits in both formats. This is storage support.
`zenpixels-convert` refuses unqualified built-in CMYK transforms, including a
polarity change. The moxcms plugin declines both unmapped F32 formats, and its
legacy entry point explicitly rejects them instead of falling back to
three-component RGB. No F32 CMYK color conversion is established here.

The integration initially panicked while constructing a descriptor against the
previous format set. The zencodec adapter also needs a fallible format lookup;
a missing format must become a runtime requirement error before publication.

## Borrowed code planes

Interleaved and semi-planar views, exact code conversion (`zenpixels_convert::codes`) and
current-sample evidence (`sample::ExactDepth`) are implemented on an unreleased branch; see
[native codes and current-precision evidence](sample-precision-evidence.md).

The output-plan review adds `sample::CodePlane` for native U8/U16 planes with
explicit code placement and stride. The converter borrows these views without
adopting native ownership. See the [executing YCbCr contract](borrowed-ycbcr-contract.md)
for the supported route, refusal behavior and native qualification.
