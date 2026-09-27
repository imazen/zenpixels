# Final 0.2 bridge and mandatory 0.3 sample signaling

Proposal, 2026-09-27. Source baseline: `c5d894543baffaad9ab19d277a22e35e70af3cd8`,
after pulling the complete planar deprecation and checked parts adoption.
**This document proposes release requirements; it implements no API changes.**
“Must” below describes acceptance criteria for the proposed implementation.

This supplements the [consolidated review](zenpixels-0.2-and-0.3-review.md)
and [release checklist](release-checklist-0.2-and-0.3.1.md). It makes their
deferred sample-signaling work concrete without reopening the existing image API
or moving media transport into core. It supersedes recommendations to extend the
deprecated `planar` module, including its historical `PlaneLayout` migration.

## 1. Release decisions

1. **Finish one designated final 0.2 bridge.** The next available patch is
   0.2.17; use that number only if it contains the promised common migration
   surface. Intermediate correctness patches do not become the “last bridge”
   merely by publication. Retain a maintenance line, but plan new API work for
   0.3 after the designated bridge.
2. **Use 0.3.1 as the breaking destination.** Both crates' 0.3.0 releases were
   already published and yanked. Keep the completed bridge's migration
   destinations usable with unchanged consumer source, supported features and
   one core version per connected pipeline.
3. **Make sample interpretation a 0.3.1 release gate.** Specify and validate
   storage width, code depth, bit placement and range independently. A shared
   checked sample-encoding primitive needs a real codec adapter before release;
   it must cover the native 10/12-bit cases below. Ship it additively in the
   final bridge too if ready, retaining its signatures in 0.3.1. Adding a new
   explicit raw-sample interface in 0.3.1 must not reinterpret the existing
   RGB/gray U16 interface or invalidate the bridge's migrated image consumers.
4. **Prototype video outside zencodec initially.** Keep frame ownership,
   decoder mapping guards, timestamps, seeking, network I/O, packet delivery,
   animation composition and streaming encode in an experimental media crate.
   It can depend on zenpixels for sample/color vocabulary. Moving the proven
   media API into zencodec later is a separate decision.
5. **Retire the old planar model.** Remove its module and root/convert
   re-exports in 0.3.1 after the actual `PlaneMask` consumer migrates. Keep the
   accepted Cargo feature name as an empty compatibility stub in core and a
   forwarding/no-op feature in convert through 0.3.x. That feature does not
   enable or select the new video model.

No `video.rs` filename, universal frame owner, or complete conversion engine is
mandated. Core gets the small interpretation primitives demonstrated by actual
adapters. The media prototype proves the plane/view API before that API is
promoted into zenpixels. Unsupported conversions may return errors; silently
incorrect interpretation cannot satisfy the release gate.

## 2. What belongs in the final 0.2.x release

The broader storage, CMS and prepared-conversion work remains in the
[existing checklist](release-checklist-0.2-and-0.3.1.md). For this scope:

| Requirement | Current status | Final-bridge acceptance |
|---|---|---|
| Deprecate all legacy planar entry points | Implemented on main, unreleased | Preserve working legacy code; verify warnings through module/root/convert imports and inferred methods. Do not recommend another type in that module as the migration destination. |
| Preserve owned storage and its interpretation | `into_parts`, checked `try_from_parts`, error recovery and `into_contiguous` implemented | Retain their metadata/offset/stride behavior. Do not add a second bytes-only export to support video. |
| Migrate zenfilters' `PlaneMask` | Outstanding | Move the Oklab filter mask to zenfilters, migrate public `ChannelAccess` fields/constructors and their consumers, and remove its `planar` feature requirements. Test that same migrated zenfilters source with both core releases. Coordinate any required zenfilters breaking release. |
| Document the existing U16 contract | Outstanding | RGB/gray U16 uses a 16-bit numerical domain: full range has maximum 65535; narrow range uses 16-bit anchors. Original codec bit depth is not an override. Raw 10/12-bit codec words remain codec-owned until an explicit adapter interprets them. |
| Stop misleading depth/range conversions | Existing range-crossing refusal; narrow cross-depth approximation documented | Fix anchor-preserving narrow depth conversion, or explicitly refuse that unsupported conversion. Do not promote the current full-scale approximation as an exact narrow-range conversion. List changed behavior. |
| Preserve complete current color information | Broader proposal outstanding | Keep matrix/range/component roles through the raw-media boundary; an RGB descriptor or ICC profile alone cannot describe a YCbCr source. Treat CICP-to-descriptor helpers as declarations, never reconstruction. |
| Verify dependency and feature compatibility | Outstanding | Test every admitted core/convert pairing, including old convert forwarding `zenpixels/planar`. A broad version range alone proves neither compilation nor type unification. |

Do not freeze a replacement planar/video container just to retire the unused
container. For known callers, retaining codec-owned storage and relocating the
filter mask is the migration. For external callers, publish a migration note
covering the removed types, retaining separate owned planes, and the prototype's
status; do not claim there is already a drop-in universal replacement.

Cargo feature names do not have Rust `#[deprecated]` diagnostics. Document the
`planar` feature's retirement separately from Rust-item warnings. Removal of the
Rust module and retention of the feature spelling are separate release actions.

## 3. 0.3 mandates for 10/12-bit samples

### S1. Distinguish storage, code values and provenance

For unsigned integer components, the interpretation must determine:

- Storage word width and actual byte order. Typed `u16` slices are native-endian;
  external byte formats declare their byte order. Existing `ByteOrder` describes
  RGB/BGR channel order and must not be used as an endian declaration.
- Nominal code depth `N` and the number of low padding bits `shift`.
- Component role and range, separately from storage packing.

These are semantic requirements, not proposed public field names. Use checked
construction, with `0 < N <= storage_bits` and `N + shift <= storage_bits`.
The required U16 subset includes right-aligned 8/10/12/16-bit codes and
left-aligned 10/12-bit codes. Describing left alignment does not claim support
for every external surface's stride, interleaving or endian layout.

For a word in the supported representation:

```text
code = (word >> shift) & ((1 << N) - 1)
word = code << shift             # canonical write: padding bits are zero
```

Use adequately wide unsigned arithmetic. Padding is not extra signal precision:
readers ignore it under this contract; adapters for formats requiring zero
padding may additionally validate it. Structural validation is O(planes), not
an implicit O(samples) scan. A separate value-conformance check may scan.

| Representation | Storage bits | Code bits | Shift | Word containing the largest code |
|---|---:|---:|---:|---:|
| Native 10-bit code in U16 | 16 | 10 | 0 | 1023 |
| Left-aligned 10-bit code in U16 | 16 | 10 | 6 | 65472 |
| Native 12-bit code in U16 | 16 | 12 | 0 | 4095 |
| Left-aligned 12-bit code in U16 | 16 | 12 | 4 | 65520 |
| Full-range normalized U16 RGB/gray | 16 | 16 | 0 | 65535 |

The last row remains a 16-bit interpretation even if its values were produced
from a 10-bit source. Optional original precision is provenance, not the current
sample encoding. Likewise, an encoder's requested output bit depth is a target
quantization policy, not a relabeling of its input storage.

Packed bitstreams such as v210 need their own layout contract. They must not
enter this one-word-per-component interface merely because they contain 10-bit
samples. Signed integer and float representations need separate semantics.

### S2. Full-range rescaling is not bit shifting

For full-range RGB/gray or luma, expanding an N-bit code `q` to normalized U16
with round-to-nearest, ties upward is:

```text
max_code = (1 << N) - 1
u16_value = (q * 65535 + max_code / 2) / max_code  # integer divisions
q_again = (u16_value * max_code + 32767) / 65535
```

For N = 10 or 12 this mapping round-trips every code exactly if no intervening
processing changes the expanded value. It is a proposed adapter contract, not a
claim that every codec currently uses it. Left shifting is a packing operation:
`1023 << 6` is 65472, not 65535. Specify rounding and optional dithering when
precision is reduced; report a lossy operation when the requested policy requires
it. Do not infer exact recoverability merely from original-bit-depth metadata.

Chroma also needs its signed offset: full-range Cb/Cr is not ordinary unsigned
RGB normalization. Under the BT.2100 integer convention, its neutral code is
`2^(N-1)` and decoded chroma is `(q - 2^(N-1)) / (2^N - 1)`.

### S3. Apply range in the code domain before packing

Reuse `SignalRange` for a **resolved** full/narrow choice. Apply nominal N-bit
anchors after unpacking; do not use the containing word's width as code depth.

| Code depth | Narrow Y′ / R′G′B′ black…white | Narrow Cb/Cr minimum…maximum | Neutral chroma |
|---:|---:|---:|---:|
| 8 | 16…235 | 16…240 | 128 |
| 10 | 64…940 | 64…960 | 512 |
| 12 | 256…3760 | 256…3840 | 2048 |
| 16 | 4096…60160 | 4096…61440 | 32768 |

For narrow codes, `s = 2^(N-8)`, `Y′ = (q - 16*s)/(219*s)` and
`Cb/Cr = (q - 128*s)/(224*s)`. Depth changes preserving narrow interpretation
scale these anchors by powers of two, with explicit rounding when reducing depth.
The full-range multiplier `(2^M - 1)/(2^N - 1)` is not the narrow-range rule.
Use signed or floating-point intermediates when subtracting offsets so
below-black and negative chroma values cannot underflow.

The 10/12-bit entries and chroma offset follow
[BT.2100-3, Table 9](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-3-202502-I!!PDF-E.pdf).
The 8/16-bit entries apply the scaling convention already documented by
[`SignalRange`](../zenpixels/src/descriptor.rs).

Nominal black/white anchors are not a promise that all samples lie between them.
Preserve excursions during intermediate processing unless a named clipping
policy applies; enforce a destination format's permitted code range on output.
Refuse a conversion when its implementation cannot honor that policy. Alpha
coverage has its own depth/full-scale endpoints, independent of video range,
matrix and transfer; never send alpha through PQ/HLG or a luma range expansion.

### S4. Carry interpretation across every boundary

Raw-sample views must bind sample encoding to their bytes. Do not represent a
right-aligned 10-bit plane as an ordinary `Gray16` image plus optional metadata
that existing converters ignore. A Y′ plane is a component of an enclosing
signal, not automatically a standalone grayscale image with a complete profile.

The representation must preserve encoding through borrowing, ownership transfer,
crops, copies and dynamic dispatch. Do not implicitly erase it into the legacy
U16 image path. An explicit bridge must either convert to that path's declared
interpretation or reject the operation. Once converted, describe the actual
output values, retaining source precision only as history.

This is a boundary contract, not a promise to detect callers who misdeclare bare
words as normalized RGB. Value inspection cannot reliably infer bit depth, range
or color space. Constructors must require declarations; implementations must
validate representable combinations without guessing from image content.

## 4. Color and geometry the media prototype must preserve

These requirements constrain any advertised raw-video adapter. They do not
require publishing a complete frame model in the final 0.2 release.

| Signal | Required meaning |
|---|---|
| Component roles and sampling | Distinguish RGB, Y′, Cb, Cr and alpha. Describe independent byte strides, component-to-plane mapping, subsampling and visible geometry. Do not assume three separate allocations; monochrome and interleaved chroma are distinct layouts. |
| Chroma location | Separate sampling step from sample origin relative to luma. Preserve explicit unknowns and crop phase. An odd-origin crop must update coordinates or fail. If interlaced siting is unmodeled, reject it rather than treating it as progressive. |
| Matrix coefficients | Retain the raw H.273 code separately from primaries and transfer. Do not revive the three-choice `YuvMatrix` as a complete registry or collapse BT.2020 constant/nonconstant luminance. A recognized/resolved matrix is not necessarily implemented. |
| Missing or unknown descriptions | Preserve unsupported numeric codes and distinguish missing/unspecified values from resolved choices. Format-mandated defaults and caller assumptions must be explicit at resolution. Do not invent BT.709 or full range because a core enum defaults there. |
| ICC and YCbCr | A selected RGB profile describes the reconstructed RGB interpretation. YCbCr matrix/range/siting describe how to reconstruct it. Those layers may coexist. Raw container/bitstream conflicts and authority resolution belong at the media boundary. |
| Linear-light domain | Distinguish scene-relative light, display-relative light with a stated nits-per-unit anchor, absolute display nits, and unresolved units. `Linear` alone does not establish any of them. A `DiffuseWhite` anchor does not supply an HLG rendering transform. |
| HDR metadata | Content light levels and mastering-display metadata are not normalization factors. Missing rendering policy must cause refusal or an explicit assumption, never an invisible default. |

Keep codec/container source evidence in the media layer and resolved current
interpretation attached to the sample view. Avoid stuffing frame timing or an
unbounded metadata collection into `PixelDescriptor`. Existing packed image
consumers keep their compact representation; new primitives should be reusable
by the external prototype without a dependency on that prototype.

`zenpixels-convert` or a selected backend owns unpacking, range expansion,
chroma reconstruction, matrix conversion, transfer/color conversion and output
quantization. The plan must state which operations occur, with filter/edge,
precision, alpha and clipping policy. RGB output carries RGB matrix identity
and its resulting range; copying the source YCbCr CICP verbatim is incorrect.
AOM-to-matching-YCbCr consumers can borrow planes; RGB consumers need explicit
conversion. Do not force an RGB roundtrip for native-plane transcodes.

## 5. Acceptance evidence before calling 0.3.1 ready

These are required future implementation checks, not tests passed by this PR.

| Gate | Evidence |
|---|---|
| Sample representation | Exhaustively unpack/repack all 1024 and 4096 codes in right/left-aligned U16; verify padding policy, invalid depth/shift rejection and explicit byte order. |
| Normalization | Exhaustively verify S2's full-range roundtrip; assert that left shifts do not produce normalized U16 maxima. |
| Range/alpha | Test every S3 anchor and neutral chroma, narrow-to-narrow depth changes, excursions, and independent alpha endpoints. Unsupported operations fail before output is presented as converted. |
| Metadata retention | Borrow/copy/crop/own/type-erasure paths preserve current encoding. A native 10-bit code of 940 cannot silently enter an ordinary U16 converter as full-scale RGB. Verify truthful output metadata. |
| Geometry | Exercise independent strides, odd dimensions, missing final-row padding, overflow, zero area and odd-origin chroma crops. Reject unsupported layout combinations. |
| Color resolution | Cover unknown raw codes, missing range, ICC plus YCbCr, differing matrix coefficients, and unresolved linear units. Prove the selected conversion receives all required inputs. |
| Concrete adapters | Borrow an AOM 10/12-bit decoded frame with unchanged plane pointers; adapt SVT's right-aligned 10-bit input; demonstrate explicit RGB conversion or precise refusal for CVVDP. Backend capability limits remain visible. |
| Migration | Relocate zenfilters' mask, exercise public filter boundaries, deny deprecations on migrated consumers, and resolve all admitted core/convert versions including `planar` forwarding. |

At minimum, the initial 0.3 release needs the checked sample vocabulary and a
real adapter proving it. Remaining backend operations can be explicitly
unsupported. If the sample contract cannot be implemented without breaking a
newly published bridge destination, resolve that before either release;
do not defer an already-known incompatible interpretation to 0.4.

## 6. Source audit and limits

Source was refreshed on 2026-09-27. Zenpixels, SVT, zenpipe and zenjpeg were
pulled; zenmetrics was fetched and its detached HEAD equals remote master.
AOM has unrelated working changes and a detached HEAD, so its latest remote
main was fetched and read with `git show`, leaving the working tree untouched.
This is a focused source audit, not a rerun of every older document's tests or
a claim that all external users have been inventoried.

| Project / revision | Source fact used here |
|---|---|
| [zenpixels `c5d8945`](https://github.com/imazen/zenpixels/tree/c5d894543baffaad9ab19d277a22e35e70af3cd8) | `zenpixels/src/descriptor.rs`: storage ChannelType and range, no separate code depth/shift. `cicp.rs`: raw codes retained, convenience enums can lose unsupported codes; `to_descriptor` does not perform matrix conversion. `lib.rs`/`planar.rs`: complete deprecation. `zenpixels-convert/src/convert.rs`: range-crossing refusal. |
| [SVT `0f63ca1`](https://github.com/imazen/zenav1-svt/blob/0f63ca13757eea3075f29656b7886143073d2022/rust/svtav1/src/animation.rs) | `AnimationFrame` and native 10-bit encode entry points accept right-aligned `0..=1023` U16 samples. |
| [AOM `6799fe9`](https://github.com/imazen/zenav1-aom/blob/6799fe958c87a7f6696113630e5094bdfe115a4f/crates/aom-decode/src/frame.rs) | `FrameDecode` carries U16 planes with separate 8/10/12-bit depth, geometry, raw color codes, range and chroma position. These are concrete adapter inputs, not zenpixels types. |
| [CVVDP `0a61830`](https://github.com/imazen/zenmetrics/blob/0a61830db89a768ab9462f166d1412e948e4efa9/crates/cvvdp/src/color.rs) | U16 RGB conversion divides by 65535. `video.rs`'s planar layout is RGB. Its suggestion to left-justify 10/12-bit values is not exact full-range normalization; the arithmetic above exposes the difference. |
| [zenfilters `6a5b052`](https://github.com/imazen/zenpipe/blob/6a5b052b2c0d398cbd16add420babe246a9638b7/zenfilters/src/access.rs) | `PlaneMask` appears in public `ChannelAccess`, `masked.rs` and `filters/alpha.rs`; its Cargo manifest enables `planar`. Its Oklab plane storage is local. |
| [zenjpeg `d06d19a`](https://github.com/imazen/zenjpeg/blob/d06d19ae6c2a38097fb8be85b3a21da5a3f1eb96/zenjpeg/src/types.rs) | Defines its own `Subsampling`. No references to zenpixels's `PlaneDescriptor`, `PlaneSemantic`, `MultiPlaneImage` or `PlaneMask` were found in its source. The old claim of hundreds of planar users conflated names. |
| [Microsoft surface definitions](https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats) | P010 uses little-endian U16 words with six low padding bits and interleaved chroma. Storage alignment must be separate from normalization and component layout. |

The sample vocabulary, explicit rejection rules, release placement and
prototype boundary are recommendations based on these facts. Exact public
names and carrier signatures still require review with the concrete adapters.
