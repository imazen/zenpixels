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
