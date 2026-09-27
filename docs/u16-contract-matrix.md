# U16 contract matrix

2026-09-27, implementation audited at `b644ac6`. **This is a contract and
implementation matrix, not a claim that all entries are supported.** Start with
the representation table, then choose the numeric operation. Applying the wrong
operation quickly is still incorrect.

`N` = current code depth, `M` = destination code depth, `s` = low padding bits,
`q` = unpacked unsigned code, `w` = stored U16 word, `max(N) = 2^N - 1`.
Examples cover 8/10/12/16-bit unsigned components in one word each. Packed
multi-component words, signed samples and F16 require distinct representations.

## 1. What must be signaled independently

| Axis | Choices relevant here | Meaning / constraint | Where it belongs |
|---|---|---|---|
| Storage | U8 bytes; U16 words; packed bitfields; F16 | Sixteen storage bits do not imply sixteen signal bits or integer samples | Physical layout |
| Byte endianness | Native typed words; explicit little/big endian bytes | Decode bytes before shifting numeric words; independent of RGB/BGR order | Byte-backed view / adapter |
| Code depth | 8, 10, 12, 16 here | Current integer domain, not source-file history | Checked sample encoding |
| Bit placement | Right-aligned, left-aligned, other supported shift | `N+s <= 16`; extraction is `(w >> s) & max(N)` | Checked sample encoding |
| Padding policy | Ignore on read; zero on write; optional strict check | Padding is not precision; strict validation reads samples | Encoding contract / explicit validation |
| Component role | RGB/gray/Y′; Cb/Cr; alpha | Selects offset, span and applicable color operations | Signal/layout, with per-plane/component encoding where needed |
| Range | Full; narrow; unresolved | Nominal endpoints are distinct from storage limits and format-permitted codes | Resolved signal interpretation |
| Transfer and units | Linear, SDR transfer, PQ, HLG; luminance anchor | Depth rescaling alone does not linearize or tone-map | Current color interpretation |
| Alpha | None, padding, straight, premultiplied | Alpha coverage is full-scale; may have a different depth from color | Layout/alpha interpretation |
| Quantization | Floor, nearest/tie rule, explicit dither | Describes an operation, not a new storage format | Conversion plan / output policy |
| Provenance | Original bit depth / replication source | Does not prove current values after processing | Origin metadata, never an implicit conversion override |
| Geometry | Dimensions, strides, plane layout, subsampling, siting | Required for actual video views; not inferred from U16 encoding | View / enclosing frame |

Do not add all these as fields to `PixelDescriptor`. Existing image descriptors
remain compact. The proposed raw-sample interface binds a small checked encoding
to borrowed storage and obtains signal/layout meaning from its enclosing view.
Public names remain uncommitted. Full/narrow chroma below uses an explicitly
identified convention; do not assume every external format uses identical rules.

## 2. Representation / replication / zero-padding matrix

Examples use a middle code and a maximum code. Values are decimal unless `0x`.
“Normalized” below means the full-range integer numeric domain, not linear light.

| Representation | N / s | Example stored word | Maximum stored word | Correct interpretation / inverse | Current support |
|---|---|---|---:|---|---|
| 8-bit code, leading zero extension | 8 / 0 | `128 → 0x0080` | 255 | Divide code by 255 for full-range intensity; recover with mask | Proposed raw view; not ordinary Gray16 |
| 8-bit code, trailing zero padding | 8 / 8 | `128 → 0x8000` | 65280 | Unpack `w >> 8`, then interpret as 8-bit | Proposed raw view |
| 8→16 bit replication | 16 / 0 | `128 → 0x8080 = 32896` | 65535 | `q*257`, exact full-range widening; inverse valid on replicated subset | Existing U8→U16 and lossless compaction |
| Native 10-bit code in U16 | 10 / 0 | `512 → 512` | 1023 | Extract 10-bit code; apply code-domain range | Proposed raw view |
| Left-aligned 10-bit code | 10 / 6 | `512 → 32768` | 65472 | `w >> 6`; low six bits are padding | Proposed raw view; P010 is one concrete layout |
| Nearest full-range 10→16 | 16 / 0 | `512 → 32800` | 65535 | Current domain is now sixteen bits; inverse nearest recovers original code if unchanged | Proposed native-code adapter |
| Repeated-bit 10→16 approximation | 16 / 0 | `(512<<6)\|(512>>4) = 32800` | 65535 | Not nearest rescaling for every input; do not label the low bits padding | No proposed default; interoperability only if explicitly required |
| Native 12-bit code in U16 | 12 / 0 | `2048 → 2048` | 4095 | Extract 12-bit code; apply code-domain range | Proposed raw view |
| Left-aligned 12-bit code | 12 / 4 | `2048 → 32768` | 65520 | `w >> 4`; low four bits are padding | Proposed raw view |
| Nearest full-range 12→16 | 16 / 0 | `2048 → 32776` | 65535 | Current domain is sixteen bits; inverse nearest recovers unchanged code | Proposed native-code adapter |
| Repeated-bit 12→16 approximation | 16 / 0 | `(2048<<4)\|(2048>>8) = 32776` | 65535 | Same warning as 10→16 replication | No proposed default |
| Ordinary U16 full-range image | 16 / 0 | `32768` | 65535 | Intensity `w/65535`; no guaranteed lower-depth inverse | Existing RGB/gray API |
| U16 narrow-range image | 16 / 0 | Black `4096`, white `60160` | Storage maximum 65535 | Apply sixteen-bit narrow anchors; full-scale division is wrong | Signaling exists; conversion incomplete |
| IEEE F16 storage | Not integer N/s | `0x3c00` means `1.0` | Not an unsigned maximum | Decode floating point; identical storage width does not authorize integer reinterpretation | Existing separate F16 formats |

Zero extension (`q as u16`) changes storage width only. A left shift can be
either **packing** (`N` unchanged, `s` changes) or **narrow-domain widening**
(`N` changes, anchors scale). Identical words do not make the contracts identical.
Replicated output can always recover its original bits by truncating back, but
that does not make replication equal to nearest normalization.

“Lossless” needs an object: padding bits, original integer codes, interpreted
signal, or a rendered image. Use these distinct guarantees:

| Mapping | Original-code/byte recovery | Interpreted signal |
|---|---|---|
| Packing with canonical zero padding | Original code exact; arbitrary old padding not preserved | Identical when N/range/role remain unchanged |
| Full-range U8→U16 by replication | Every original code recovered | Exact rational intensity: `q/255 == (q*257)/65535` |
| Full-range 10/12→U16 by nearest scaling | Every original code recovered by inverse nearest | Usually a small rational approximation, not mathematical equality |
| Narrow U8→U16 by shifting eight bits | Every original code recovered | Exact narrow signal including representable excursions |
| Narrow U8→U16 by replication | Inverse replication recovers bytes | Wrong signal scaling; invertible does not mean correct |
| U16→F32 normalized signal | Correct nearest inverse can recover integer codes | Most fractions require floating-point rounding; transfer/units stay separate |
| U16→F16 | Not generally recoverable | Additional rounding; F16 is not a compact exact U16 carrier |

P010 uses little-endian words, ten high payload bits and six low padding bits;
its chroma is interleaved and its plane layout must also be honored. A packing
description alone is insufficient to describe P010. [Microsoft surface definitions](https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats)

## 3. Full-range RGB/gray/Y′/alpha depth matrix

Define `F(N,M,q) = (q*max(M) + floor(max(N)/2)) / max(N)` with integer division
and sufficiently wide intermediates (`u64` is a simple general reference).
This is nearest rescaling. Denominators are odd, so half ties do not occur.
This table does **not** apply to offset chroma, narrow range or transfer changes.

| From \ To | 8 | 10 | 12 | 16 |
|---|---|---|---|---|
| 8 | Identity | `F(8,10,q)` | `F(8,12,q)` | `q*257` (exact replication) |
| 10 | `F(10,8,q)` | Identity | `F(10,12,q)` | `F(10,16,q)` |
| 12 | `F(12,8,q)` | `F(12,10,q)` | Identity | `F(12,16,q)` |
| 16 | `(q+128)/257` in wider arithmetic | `F(16,10,q)` | `F(16,12,q)` | Identity |

Widening followed by inverse nearest rescaling recovers every original code
for all widening pairs in this table. Narrowing arbitrary input is generally
lossy. Exact U16→U8→U16 round-trip requires `q % 257 == 0`; “fits in U8” and
“low byte is zero” are different predicates and do not establish that property.

For 8/10/12-bit input, direct nearest narrowing to U8 equals nearest expansion
to U16 followed by exact U16→U8 narrowing. Exhaustive evidence is in the script
below. This is not permission to cancel arbitrary color/quantization stages.

### Simple repeated-bit widening is not generally nearest rescaling

Repeat the source bit pattern and truncate it to the destination width:

| Expansion | Replication differs from nearest | Largest difference | First differing input: replicated / nearest |
|---|---:|---:|---|
| 8→10 | 42 / 256 | 1 | 43: 172 / 173 |
| 8→12 | 56 / 256 | 1 | 9: 144 / 145 |
| 8→16 | 0 / 256 | 0 | none |
| 10→12 | 170 / 1024 | 1 | 171: 684 / 685 |
| 10→16 | 234 / 1024 | 1 | 9: 576 / 577 |
| 12→16 | 952 / 4096 | 1 | 137: 2192 / 2193 |

Use exact nearest rescaling as the proposed full-range default. Retain a
different bit-exact external convention only through an explicit adapter
contract. Do not expose six new public helpers merely because six pairs exist.

## 4. Role and range matrix

Let `k = 2^(N-8)` for N ≥ 8. These formulas decode an unpacked code into a signal;
they do not themselves decode the transfer function or reconstruct RGB.

| Role / range | Decoded signal | Black / neutral | Nominal positive endpoint | Depth change |
|---|---|---|---|---|
| RGB/gray/Y′ full | `q/max(N)` | 0 | `max(N)` | Endpoint rescaling `F` |
| RGB/gray/Y′ narrow | `(q-16*k)/(219*k)` | `16*k` | `235*k` | Power-of-two scaling for narrow→narrow |
| Cb/Cr full, BT.2100 convention | `(q-2^(N-1))/max(N)` | Neutral `2^(N-1)` | Nominal +0.5 quantizes/clips at `max(N)` | Scale signed distance from neutral, then add destination neutral |
| Cb/Cr narrow | `(q-128*k)/(224*k)` | Neutral `128*k` | `240*k` (+0.5); `16*k` (−0.5) | Power-of-two scaling for narrow→narrow |
| Alpha coverage | `q/max(alpha_bits)` | Transparent 0 | Opaque `max(alpha_bits)` | Full-range scaling, independent of color range/transfer |

The 10/12-bit anchors, offset chroma convention and tie-away-from-zero rounding
come from [BT.2100-3 Table 9](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-3-202502-I!!PDF-E.pdf).
Eight/sixteen-bit narrow entries below extend the scaling convention already
documented in [`SignalRange`](../zenpixels/src/descriptor.rs).

| N | Full intensity maximum | Narrow black…white | Narrow chroma low…neutral…high |
|---:|---:|---|---|
| 8 | 255 | 16…235 | 16…128…240 |
| 10 | 1023 | 64…940 | 64…512…960 |
| 12 | 4095 | 256…3760 | 256…2048…3840 |
| 16 | 65535 | 4096…60160 | 4096…32768…61440 |

In narrow RGBA widening, color and alpha cannot share an indiscriminate scaling
rule: an 8-bit white color lane `235` becomes `60160` (`<< 8`), while opaque
alpha `255` becomes `65535` (`* 257`). Same storage depth does not imply the
same scaling for every lane.

**Full-range chroma is a trap:** rescaling neutral 10-bit code 512 as ordinary
unsigned intensity produces 32800. Sixteen-bit chroma neutral is 32768.
Its conversion is `round((q-neutral(N))*max(M)/max(N) + neutral(M))`, followed
by the chosen destination limit policy. Use signed wide intermediates. Rounding
must apply to the complete affine result: tie-to-even is not invariant under
adding an odd integer offset.

**Three sets of limits:** storage-representable codes, nominal signal anchors,
and format-permitted codes. They are not interchangeable. BT.2100 Table 9 permits
narrow 10-bit video codes 4…1019 and narrow 12-bit codes 16…4079, including
excursions beyond nominal black/white. Other interfaces may impose their own
limits. Padding validation does not check any of these signal limits.

## 5. Operation / exactness / cost matrix

| Operation | Formula or required action | Exactness | Pixel work / failure contract | Status |
|---|---|---|---|---|
| Declare checked encoding | Validate N/s/extent/endianness contract | No data conversion | O(1) metadata checks; no scan | Proposed |
| Unpack / repack same code depth | Mask/shift, then `q << new_s` | Code-exact; noncanonical padding bits are not retained | Can fuse with conversion; repacking bytes needs a traversal | Proposed adapter |
| Swap word endianness | Decode/encode u16 bytes | Word-exact | Borrow if native order already fits; otherwise fuse byte swap | Proposed adapter |
| Zero-extend U8 storage to U16 | `w = q` | Code-exact; interpretation must stay N=8 | Writes destination words | Proposed raw-code adapter |
| Left-align N-bit storage | `w = q << (16-N)` | Code-exact packing | Writes/reuses destination; no proof scan | Proposed raw-code adapter |
| Full-range widening | `F(N,M,q)` | Original codes recoverable; rational intensity approximated when necessary | One pass or fused with other work | Existing 8→16; raw 10/12 pending |
| Ordinary full-range narrowing | `F(N,M,q)` | Lossy for arbitrary source | One pass; no losslessness preflight | Existing 16→8; raw 10/12 pending |
| Proven replicated-U16 compaction, full range | Select one identical byte; inverse `b*257` | Exact on proved subset | Existing analysis plus rewrite; proof must reflect current pixels | Existing; range-aware eligibility and U16 scan fusion pending |
| Exact compaction of left-padded raw codes | Shift back under known encoding | Exact codes; canonical repack matches payload | No scan needed for interpretation; strict padding check optional | Proposed |
| Narrow→narrow widening | `q << (M-N)` | Exact code/anchor scaling | One pass or fused | Current full-scale route needs fix/refusal |
| Narrow→narrow narrowing | Round `q / 2^(N-M)` | Generally lossy; ties possible | One pass; apply explicit destination limits | Pending |
| Full↔narrow RGB/Y′ | Decode source offset/span, encode destination offset/span | Often lossy; excursions may exceed destination domain | One pass possible; specify clipping/refusal | Currently refused |
| Full/narrow chroma conversion | Offset-aware affine map, with source/destination chroma spans | Generally lossy; must preserve neutral | One pass possible; explicit limits | No YCbCr API yet |
| Transfer-changing narrowing | Decode transfer, encode target, quantize once | Generally lossy | Existing lookup execution; first-use setup is currently hidden | Existing limited SDR pairs; preparation fix pending |
| Native 10/12 transfer-changing narrowing | Direct code-domain table or arithmetic | Must match named numerical contract | 1/4 KiB U8 tables possible; no intermediate U16 image needed | Proposed benchmark/adapter |
| Strict padding/value conformance | Read samples, validate requested rule | No conversion | Explicit preflight, or opt-in fused partial-output check | Proposed |
| Dithered narrowing | Explicit quantizer and noise/error policy | Intentionally lossy; not bit-exact inversion | Adds per-pixel work, possibly row state; no automatic prepass | Not proposed as an implicit default |
| YCbCr→RGB | Chroma reconstruction, range/matrix conversion; transfer as requested | Beyond bit-depth conversion | May require neighboring rows/halos; no implicit RGB copy on handoff | Video backend work pending |

**Existing analysis also needs a range guard.** `uses_low_bits == Some(false)`
checks byte replication, independently of `SignalRange`. For narrow U16,
`0x1010 = 4112` decodes to `(4112-4096)/56064`, but extracting `0x10` as narrow
U8 decodes to zero. Re-expansion by 257 recovers the bytes while reproducing the
original range bug. Narrow 16→8 lossless eligibility instead needs code-domain
scaling (multiples of 256 within the destination code domain), plus its applicable
format limits. Until implemented, refuse that semantic reduction. Do not change
the meaning of the public report field to silently stand for a different proof.

“No scan” means no additional content-analysis pass. A transformation that
rewrites samples still reads/writes pixels. Borrowed views and owned parts
handoff can preserve allocation and stride; normalization is not a prerequisite
for handing native planes to a compatible consumer.

## 6. Rounding, clipping and exceptional values

| Rule | Definition / distinguishing example | Appropriate contract |
|---|---|---|
| Floor in full-scale domain | `floor(v/257)` for U16→U8; 129→0 | Explicit truncating quantization; differs from nearest |
| High-byte extraction | `v >> 8`; 511→1 whereas nearest gives 2 | Exact for known `b*257` or left-padded-code recovery; not general nearest |
| Nearest, half upward for nonnegative values | `floor(x+0.5)`; 2.5→3 | Existing integer full-range kernel has no half ties |
| Nearest, ties away from zero | 2.5→3; −2.5→−3 | BT.2100 digital quantization convention; distinguish from unsigned-only shorthand |
| Nearest, ties to even | 2.5→2; 3.5→4 | Useful explicit contract; F32→F16 uses this, not an integer-policy synonym |
| Toward zero | −2.9→−2, unlike floor −3 | Only when explicitly requested; signed excursions expose the difference |
| Dither / stochastic rounding | Output depends on named algorithm, seed/state and sample position | Explicit opt-in; define row/chunk determinism and alpha treatment |
| Clip nominal range | Clamp to nominal black/white or chroma endpoints | Loses head/footroom; never imply this merely from Narrow |
| Clip format-permitted range | Clamp to destination's legal code interval | Distinct output constraint; perform in destination code domain |
| Reject unrepresentable input/output | Metadata refusal or content-dependent validation | Metadata refusal is cheap; content checks need explicit preflight/fused cost |
| Preserve excursions | Use a destination that can represent them | Unsigned full-range output cannot preserve negative intensity; refuse or select another representation |
| Float NaN/±infinity | No integer counterpart | New checked API must choose refusal or explicit saturation; do not rely on incidental casts |

For new video adapters, use the external format's specified rounding; otherwise
document nearest rounding explicitly. Do not change existing integer rounding
or add dither as a silent default. F16 conversion's ties-to-even does not imply
the same policy for integer narrowing.

## 7. Code that must not be confused

```rust
let q = 128u8;
let code_in_word = u16::from(q);             // 0x0080: N=8, s=0
let left_packed = u16::from(q) << 8;          // 0x8000: N=8, s=8
let full_range_u16 = u16::from(q) * 257;      // 0x8080: N=16, s=0
assert_ne!(left_packed, full_range_u16);
assert_eq!(left_packed >> 8, code_in_word);
assert_eq!(full_range_u16 / 257, code_in_word);

let narrow_black_u8 = 16u16;
assert_eq!(narrow_black_u8 << 8, 4096);       // Correct 16-bit narrow black
assert_eq!(narrow_black_u8 * 257, 4112);      // Wrong narrow-range widening
```

```rust,ignore
// Proposed checked binding: preserving the native code representation.
let native = RawU16Plane::new(words, width, height, stride_samples,
                             U16Encoding::new(10, 0)?)?;
// Conversion selects unpack/range/transfer/quantize once from the full signal.
// No PixelDescriptor::GRAY16 plus ignored "original bit depth" metadata.
```

## 8. Evidence and release disposition

Run `python3 scripts/check-u16-matrix.py`. It exhaustively checks all 16 depth
pairs, widening inverses, all legal padding shifts for these depths, replication
differences, every U16 narrowing input, and direct-vs-via-U16 U8 conversion.
It also checks chroma neutral and narrow anchor arithmetic. This validates the
mathematics; it does not claim that native-plane APIs or kernels exist.

The [narrowing review](u16-signaling-and-narrowing-review.md) records current
kernel measurements. The new saturating-U16 candidate is benchmark-only.

Publish the checked raw-sample contract in the designated final 0.2 bridge if
video clients using it must compile unchanged against either line. Keep the
same contract in 0.3.1. Existing normalized image consumers retain their numeric
meaning. Prototype AOM/SVT borrowed views before committing a universal carrier.

See [implementation status](implementation-status-0.2-and-0.3.md) for completed
and outstanding work; the broader implementation is not finished.
