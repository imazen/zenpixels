# Native codes, canonical samples and current-precision evidence

2026-10-27, on top of output-plan `86d0e2d3` (PR #92). **Status: implemented in an unreleased
branch, pending review; new public API goes through a PR assigned to lilith.** Arithmetic claims
below were checked exhaustively by an independent Python oracle and are pinned by Rust tests that
pass (2026-10-28, r7900x, rustc 1.99.1): `zenpixels-convert/tests/codes_exhaustive.rs` 17/17,
`tests/precision_evidence.rs` 16/16, `tests/codes_alloc.rs` (zero allocations), and
`zenpixels/tests/code_planes.rs` 4/4. Throughput is not measured yet.

## 1. Seven things a "16-bit sample" can mean

| Axis | Carrier | Notes |
|---|---|---|
| Storage word | `ChannelType` in `PixelFormat` | `U16` storage says nothing about significant bits |
| Code depth and placement | `sample::SampleEncoding` (`code_bits`, `bit_shift`) | Native views only; never on an ordinary descriptor |
| Byte order | Native typed words | Byte-backed adapters decode first |
| Role | `codes::CodeRole` (intensity, chroma, alpha) | Selects the depth-change rule |
| Range | `SignalRange` | Narrow anchors scale by 2^(N−8); alpha is always full scale |
| Transfer | `TransferFunction` / `ColorContext` | Depth changes never linearize |
| Current exactness | `sample::ExactDepth` on `PixelBuffer`/`PixelSlice` | Evidence about current values, not history |
| Source history | `ColorOrigin`, negotiation `Provenance`, codec `SourcePrecision` | Hints and metadata; never precision authority |

Ordinary RGB/gray `U16` keeps its existing meaning: a full-range 16-bit code (white is 65535).
Native codes stay in borrowed `CodePlane` views until a consumer needs canonical samples.

## 2. Representation: borrow native codes, convert once

`CodePlane` borrows one component without copying. `from_u16_interleaved` / `from_u8_interleaved`
take a per-pixel step, so planar, semi-planar and interleaved layouts share one view:

```rust,ignore
use zenpixels::{ChannelType, SignalRange, PixelBuffer, PixelDescriptor,
                sample::{CodePlane, SampleEncoding}};
use zenpixels_convert::codes::to_canonical;

// P010: 10 high bits in little-endian words; chroma interleaved Cb, Cr.
let p010 = SampleEncoding::new(ChannelType::U16, 10, 6)?;
let cb = CodePlane::from_u16_interleaved(&uv, cw, ch, uv_stride, 2, p010)?;
let cr = CodePlane::from_u16_interleaved(&uv[1..], cw, ch, uv_stride, 2, p010)?;

// A 12-bit interleaved RGB TIFF strip, right-aligned codes, converted once:
let twelve = SampleEncoding::new(ChannelType::U16, 12, 0)?;
let rgb = [0, 1, 2].map(|c| CodePlane::from_u16_interleaved(&words[c..], w, h, stride, 3, twelve));
let mut image = PixelBuffer::new(w as u32, h as u32, PixelDescriptor::RGB16_SRGB);
let (_, exact) = to_canonical(&rgb.map(Result::unwrap), SignalRange::Full, 0, 0, image.as_slice_mut())?;
image.set_exact_depth(Some(exact))?; // the decoder wrote these samples: it may say so
```

Video handoff (AV1 decoder → VMAF, AV1 → AV1) keeps the borrowed codes; YCbCr→RGB uses the existing
`ycbcr::YcbcrToRgbF32` over the same views. `to_canonical` writes strided rows, reads each requested
sample once, allocates nothing (a 512-byte stack batch), refuses before writing, and supports crop
origins and row partitions over the original planes.

### Three ways to hold a 10-bit sample in U16

| | Canonical U16 (`F(10,16,q)`) | Unshifted codes (10/0) | MSB-packed codes (10/6, P010) |
|---|---|---|---|
| Value of full-range white | 65535 | 1023 | 65472 |
| Who can consume it | Every `RGB16`/`GRAY16` kernel, filter, encoder | Code-aware adapters (`CodePlane`, `CodeMap`) | Code-aware adapters; for narrow range only, also canonical narrow U16 consumers when padding is zero |
| Handoff from a native decoder | One conversion pass into caller storage | Borrow | Borrow |
| Exactness at 10 bits | Needs `ExactDepth` (writer, scan, or preserving op) | By construction | By construction |
| Risk if mislabeled | — | 1023 read as 1.6% gray (64× error) | 65472 read as 99.9% white; padding read as signal |

Keep native codes borrowed across codec-to-codec and codec-to-metric boundaries; convert once, at
the first consumer that needs canonical samples. Chroma depth/range maps are siting-independent;
crop origins of subsampled planes are the caller's mapping (half the luma origin, rounding per
siting), and RGB reconstruction keeps `ycbcr::YcbcrToRgbF32`'s explicit siting requirement.
Float canonical samples (`F32`) carry no `ExactDepth`: their exactness model (mantissa bits, JPEG XL
float `exponent_bits`) is separate and not proposed here.

### Conversion rules (`codes::CodeMap`)

| Role and range | N→M bits | Guarantee |
|---|---|---|
| Intensity, full | Nearest `(q·max(M) + max(N)/2) / max(N)`; no ties (odd denominators) | Widen then narrow recovers every code |
| Intensity or chroma, narrow (N, M ≥ 8) | `q << (M−N)`; narrowing rounds half up, clamps at `max(M)` | Same; anchors exact (8-bit black 16 → 4096, not 4112) |
| Alpha | Nearest full scale, whatever the color range | Same |
| Chroma, full | Refused for depth changes | Neutral and span have no exact relation |
| Same depth | Code unchanged; placement re-applied, padding zeroed | Repacking |

Placement is applied after conversion. P010 white is `1023 << 6 = 65472`; normalized white is
65535. For narrow range only, zero-padded P010 words already equal canonical narrow U16, so a borrow
would suffice when padding is known zero (an explicit check, not assumed). Narrow 16→10 clamps the
32 super-white words 65504–65535 to 1023.

Out of scope here: packed multi-component words (10:10:10:2), signed scientific scales, bit
replication conventions (provide only for a named external format), dithering (explicit opt-in only).

## 3. Exactness: a higher depth is not automatically lossless

Canonical U16 samples holding exact N-bit content survive quantization to M bits and back **iff
M = N, N divides M, or M ≥ 16** (2^N−1 divides 2^M−1 exactly when N | M). Exhaustive counts:

| Content → coded depth | U16 samples changed | Max change | Original codes recoverable |
|---|---:|---:|---|
| 8 → 10 | 252 / 256 | 32 | yes |
| 8 → 12 | 240 / 256 | 8 | yes |
| 10 → 12 | 972 / 1024 | 8 | yes |
| 4 → 10 | 12 / 16 | 26 | yes |
| 5 → 10, 6 → 12, N → 16 | 0 | 0 | yes |

Original N-bit codes stay recoverable (re-quantize the decoded samples to N bits) for every
N < M ≤ 16. Narrow-range grids nest for every M ≥ N. So "lossless" needs an object: canonical
samples (`ExactDepth::color_exact_at`) or original codes (`color_codes_recoverable_at`). Promoting
10-bit content to a 12-bit AV1 encode is the second, not the first.

## 4. Evidence: bounded facts about current values

`ExactDepth { storage, range, color_bits, alpha_bits }` (4 bytes) states that re-quantizing current
color/alpha samples to those depths reproduces them. It is established only by:

1. **The code that wrote the samples**, e.g. `to_canonical` returns it for an exact widening.
2. **An explicit scan**, `ExactDepth::measure`: one read of every active sample (never row
   padding or an RGBX/BGRX padding slot), no allocation, early exit when only the storage depth
   remains. `ExactDepth::holds_for` checks a stated depth against rows written elsewhere and
   stops at the first sample off its grid; both live in zenpixels core, so a row driver needs no
   zenpixels-convert dependency (`codes::measure_exact_depth` remains as a deprecated wrapper).

A statement made with `ExactDepth::new` and attached with `with_exact_depth` is an assertion:
the attach checks only storage and range. Exactness is also defined for the canonical mapping
only. zenavif `a7638f6e` narrows U16 by truncation and widens by bit replication (source-traced,
not run), so canonical ten-bit level 577 returns as 576 and 234 of 1024 canonical ten-bit levels
change; that native path is unqualified by canonical evidence and needs its own control.
3. **An operation that provably keeps the values**, which re-attaches it.

It is never established by the source file's bit depth, `ColorOrigin`, negotiation `Provenance`,
`FormatEntry::effective_bits`, PNG sBIT or JPEG XL `bits_per_sample` alone; those describe
history or intent. A 10-bit AV1 YCbCr frame converted to RGB has no 10-bit RGB evidence: in a
float model of the BT.709 limited-range 4:4:4 recipe, 1.61% of in-range RGB samples stored as
U16 land on the 10-bit grid, the rate for random values (1.56%;
`scripts/ycbcr-rgb-grid-membership.py`, 200,000 random code triples).

| Operation | Evidence | Who re-attaches |
|---|---|---|
| Borrowed view, `rows`, `crop_view`, slice clone | Kept | Carried by the view |
| `crop_copy`, `into_contiguous`, cow `to_mut` copy, identity `convert_to` | Kept | zenpixels (same values) |
| Transfer/primaries/color-context relabel | Kept | Values unchanged |
| Signal-range relabel, format/layout `reinterpret` | Ended | Code meaning changed |
| `as_slice_mut`, `rows_mut`, typed mutable views, `transform_in_place` | Ended, even without a write | Re-scan or re-state |
| `into_parts` → `try_from_parts` | Ended | Parts are an editable transfer record |
| Resize, filters, color conversion, premultiply/unpremultiply, blending | Ended (new buffers start empty) | Scan if needed |
| Selection composite (each output sample copies one input) | Smallest grid containing both: lcm of full-range depths, max of narrow | The compositor, from both inputs |
| Serialization | Plain data on write; a claim on read | Verify with a scan before trusting |

Ending evidence on every mutable access is conservative: an untouched buffer may lose evidence that
a scan restores. It never keeps evidence that a write invalidated. `PixelSliceMut` carries none.

### Alternatives prototyped (test-only) and why not

- **Edit log as authority** (`tests/precision_evidence.rs`, `an_edit_log_...`): code that writes
  through a mutable view without appending leaves the log claiming 10-bit exactness, and encoding at
  10 bits then discards the edit. Logs must embed both inputs at every composite: 30 frames
  composited in sequence hold 92 entries; a 1,024-tile mosaic holds 3,071. Every operation in every
  crate must log correctly, and a reader must understand every entry; unknown entries reduce to
  "no evidence" anyway. The bounded fact reaches the same answer with 4 bytes and the clearing hook
  zenpixels already owns. A log can still serve diagnostics at pipeline level; it is not authority.
- **Owning typestate wrapper**: compile-time proof that mutation ends evidence, but each preserving
  operation needs a wrapper method and every encoder entry point taking `PixelSlice` loses the type,
  so the slice must carry the evidence regardless.
- **Evidence in `PixelDescriptor`**: breaks a `Copy + Hash` descriptor and every equality-based
  plan cache; evidence is about values, not format.

## 5. Encoder depth decisions (proposal for zencodec)

Order: mandatory requirements filter the encoder's capable depths; evidence decides exactness;
a situational preference chooses among what remains. The model in `tests/precision_evidence.rs`:

```rust,ignore
enum Requirement { Lossless, Exactly(u8), AtMost(u8) }   // caller mandates, never bypassed
enum Preference  { PreserveProvable, Smallest }          // resolver policy among admitted depths
fn resolve(requirements, preference, pixels: &PixelSlice, capabilities: &[u8])
    -> Result<Decision { bits, exact }, Refusal::Unmet { requirement, admitted_before }>
```

- Exactness at depth d: `pixels.exact_depth()` → `color_exact_at(d) && alpha_exact_at(d)`; with no
  evidence, only d ≥ storage depth is exact.
- `PreserveProvable` picks the smallest exact depth, otherwise the largest admitted (least loss).
  Unknown U16 into AV1 {8,10,12} therefore codes 12 bits, not zenavif `Auto`'s 10, and reports
  `exact: false`; `Lossless` refuses it, naming the requirement and the admitted set. Twelve-bit AV1
  needs the Professional profile, so broad decoder support is a caller requirement (`AtMost(10)`),
  stated explicitly rather than hidden in a default.
- Stale ancestry: an 8-bit source filtered in 16-bit has no evidence; coding by source depth (8)
  changes samples, while the resolver keeps 16 bits unless the caller requires at most 8, which is
  then an explicit lossy decision.
- Encode without edit / metadata-only transcode: evidence survives, 12-bit content codes 12 bits
  exactly; changing ICC/CICP or transfer labels does not end it, a range relabel does.
- Quantization when not exact: nearest (full range) or round-half-up with top clamp (narrow), never
  dither unless requested. Integer canonical sources never clip at full range.

Where the evidence stops today (read from zencodec `e59b3803`, not built). One-shot encodes can
read it: the caller's `PixelSlice` carries `exact_depth()`, which is transport, not proof. Planned
rows do not:
`EncodeDeclaration` holds width, height, descriptor, color and access only, and the row driver
(`zencodec-io` `run_rows`) copies source rows through an evidence-free `PixelSliceMut` into a fresh
`PixelSlice`, so even a raster holding evidence reaches `push_rows` without it. `EncodePlan` does not
report the coded depth, so startup drift and admission cannot see exactness. zenavif (`a7638f6e`)
shows the cost: `Auto` codes any 16-bit input at 10 bits, lossless included, which reproduces the
input only when it already lies on the 10-bit grid, and its truncate/replicate mapping differs
from the canonical grid even then (see section 4).

Proposed contracts (root owns zencodec; nothing applied there):

- `EncodeDeclaration::exact_depth: Option<ExactDepth>` (+ `with_exact_depth`), validated against the
  descriptor; `None` means no depth below storage may be treated as exact.
- Deferred by root: a public coded-samples report with a `depth_exact` boolean. A grid result
  proves only that the canonical re-quantization is identity; it says nothing about the codec's
  own mapping, matrix, subsampling, transfer or alpha handling. A codec accepts
  `Fidelity::Lossless` only after qualifying its complete actual transform.
- Row driver: `Description::exact_depth` (a borrowed raster copies its slice's), passed to the
  declaration; every claim the driver forwards is verified on the strip before `push_rows`, or the
  operation fails terminally at that strip (no retry at another depth). A strip is never quantized to
  fit. Comparing descriptions every strip catches a changed claim, not changed values.
- Missing evidence means unknown, not inexact: a route that codes at the storage depth (16-bit
  PNG, U8 routes) needs no evidence.
- zenpixels: a verifying scan in core beside `ExactDepth`, so the driver needs no zenpixels-convert
  dependency; pipelines declare evidence statically (decoders establish it, preserving nodes keep it,
  resize/filter/blend end it, selection composites take the meet).

### A declaration made before headers: sufficient or provisional

The coded depth is fixed when the header is written, before most rows exist. A declaration is
sufficient when it is already true of every row at that moment:

- the writer makes it true by construction for every row, as a decoder does when it widens one
  native depth with `to_canonical` for the whole image;
- an owned, immutable raster was scanned with `ExactDepth::measure` (or checked with
  `holds_for`) before planning; a tag on an untouched raster is not proof by itself;
- a replay route measures in one pass and encodes in the next, over a stable snapshot; a fresh
  read of a mutable source is not one.

Otherwise it is provisional, a promise about rows not yet produced: a pipeline that derives it
from its graph (one mis-declared node, or a source whose depth changes, breaks it), a claim from
container metadata (bit-depth fields, sBIT and `effective_bits` record history), asserted
statements, and any rows a driver copies through a mutable view. Provisional evidence is checked per strip with `holds_for`
against the coded depths, the grids the header committed to, before the strip is pushed; a strip
on a coarser grid that divides the coded depth passes. A strip that fails cannot be repaired after
the header. With output staged until finish, the operation fails before anything reaches the
destination. With live output, the header can be published before the first strip is read, and
earlier strips may be out, so the operation fails and the destination must discard the partial
file. A route that relies on provisional evidence for a depth
below storage should therefore stage its output or replay; live publication is safe with
sufficient evidence or with a coded depth at least the storage depth.

zencodec's duplicate `frame::SampleEncoding` can re-export zenpixels' in the paired release.

## 6. Public API delta (unreleased)

zenpixels: `sample::{ExactDepth, ExactDepthError}` with `ExactDepth::{measure, holds_for}`;
`CodePlane::{from_u8_interleaved, from_u16_interleaved, step_words, read_codes}`;
`CodePlaneError::{ZeroStep, OutsideActivePlane}`; `PixelSlice::{exact_depth, with_exact_depth}`;
`PixelBuffer::{exact_depth, set_exact_depth}`. zenpixels-convert: `codes::{CodeRole, CodeMap,
CodeMapError, CodesError, to_canonical}` and `OutputPlan::output_color_context`. Everything is additive (new private
fields on types that already had private fields), so this design needs no break of its own and can
ship unchanged in the paired breaking release with zencodec.

Layout cost, measured with `size_of`/`align_of` on x86_64 (rustc 1.99.1, default features; a
structural measure, not heap or RSS): `PixelSlice` 48 → 56 bytes and `PixelBuffer` 64 → 72 bytes,
alignment 8 unchanged; `PixelSliceMut` stays 48 (it carries no evidence); `CodePlane` 56 → 64 for
`step_words`. `Option<ExactDepth>` is 4 bytes with alignment 1. It costs 8 because the base fields
sum to 45 and 61 bytes, leaving 3 bytes of tail padding. A 2-byte field holding only the two depths
(storage and range read from the descriptor) would fit that padding by the same arithmetic; it is
not implemented or measured. Other targets were not measured. Evidence carriage allocates nothing
(`tests/codes_alloc.rs`). Probe sources, command and results: `benchmarks/layout_2026-10-28.md`.

Not changed: `negotiate::Provenance` stays a cost-model hint (its `origin_depth` is a
`ChannelType` and cannot say 10/12); `FormatEntry::effective_bits` likewise. Their docs should state
they are not precision authority; neither needs removal for this design.
