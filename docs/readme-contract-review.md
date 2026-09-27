# Root README review and proposed 0.2 / 0.3.1 result

> **2026-09-27 update:** checked parts adoption with `take_parts()` and
> `without_buffer()` is implemented. The owner chose to deprecate the entire
> existing planar module now and defer a better video representation. Earlier
> proposals below to expand that module are superseded. See the
> [code-first review](code-review-0.2-and-0.3.md) for executable cases and wanted
> behavior before approving the remaining guards/contracts.

Reviewed 2026-09-27 against `main@17c78d9` plus the accompanying deprecation
edits. This annotates the root README, not just its quick start. **The three
deprecations, estimation opt-in, validation in `DiffuseWhite::new`, and
`PixelBuffer::{into_contiguous, into_parts}` are implemented so far.** Proposed
APIs below are intentionally marked as proposals, not runnable current examples.

The full design is [API and contract proposal](api-contract-proposal-0.2-and-0.3.1.md).
[Migration cards](migration-examples-and-audit-0.3.1.md) contain the old/new code,
the ripgrep consumer ledger, correctness reproductions, docs.rs tree and streaming
analysis. [YUV assessment](yuv-carrier-assessment.md) adds the newly identified
AV1/metric callers and revises the earlier recommendation to defer planar expansion.

## The release promise

The complete 0.2 bridge adds the replacement interfaces and deprecates their
predecessors. The proposed 0.3.1 keeps the same migrated source, supported feature
names, signatures, defaults and implementation contracts. One zenpixels version
is selected per connected pipeline. Deprecations are migration guidance; paired
consumer builds and behavior tests establish compatibility. A warning-free build
alone cannot detect a removed feature or a changed trait bound.

0.2.17 is the candidate bridge number, not a claim that these three deprecations
complete the bridge. Keep the README's current install examples until the complete
replacement release exists. Do not recommend a broad dependency range as proof
that Cargo will unify all consumers onto one version. Test both version selections.

## Introduction and quick start

| Current wording/example | What is wrong or incomplete now | If the complete proposal is adopted |
|---|---|---|
| JPEG means sRGB, AVIF means BT.2020 PQ | These are examples, not codec guarantees. ICC, unknown encoding, SDR AVIF and planar output exist. | Describe concrete decoder output including its current encoding; preserve unknowns until resolution. |
| Typed buffer has “SIMD-aligned rows” | `from_pixels` makes packed rows; type information does not select padded allocation. Typed input can be adopted without copying when allocation alignment permits; higher-alignment types require a copy. | Keep typed physical layout separate from color; state allocation/copy behavior per constructor. |
| `with_descriptor` only changes transfer/primaries | It checks physical layout compatibility, but accepts other semantic fields too. It declares interpretation; it does not convert samples. | Validate combinations at acceptance boundaries; distinguish declaration from a conversion request. Do not make all descriptor field access private. |
| `as_strided_bytes` is the whole backing buffer | It exposes the view's span. It does not recover prefix bytes, all parent pixels or Vec capacity; final-row padding is not universally present. | Use visible-span terminology and explicit stride; use ownership parts for the allocation. |
| Recover `into_vec()` and hand bytes onward | It loses the pixel offset, stride, descriptor and color context. Pool reuse is valid; treating the result as packed image pixels is not. | Make destructurable `into_parts()` the primary ownership handoff. Packed export is an explicit compaction operation. |

Old code that must gain a migration warning once its replacement exists:

```rust,ignore
// Loses the information needed to distinguish pixels from prefix/padding.
let bytes = buffer.into_vec();
encoder.encode_packed(&bytes, width, height);
```

Implemented replacement for a stride-aware owner:

```rust,ignore
let PixelBufferParts {
    data, offset, stride_bytes, width, height, descriptor, color_context, ..
} = buffer.into_parts();
// Move these into the receiver. First pixel is at data[offset], not data[0].
// No new pixel allocation, copy, compaction, or Arc clone on extraction.
```

`try_from_parts` remains proposed and must return the supplied parts on failure. Keeping a 40–100 MB
allocation is part of the API contract, not an optimization to add later.
`into_contiguous()` now moves rows within that allocation while retaining the
buffer and its metadata. It preserves the alignment offset, so packed pixels
start at `parts.data[parts.offset]` after extraction. Compaction is not constant-time
zero-copy merely because it avoids a second Vec. No bytes-only export is added.

## Wrapping, borrowing and buffer access

| README section | Current issue | Proposed common contract |
|---|---|---|
| Wrapping a decoder Vec | `from_vec` assumes tight stride and may skip leading bytes to obtain alignment. That is not a general adoption protocol for arbitrary decoder allocations. | Checked parts adoption carries the real offset and stride, never guesses the first pixel, and preserves ownership on rejection. |
| Borrowing strided bytes | The sample's stride 256 equals 64 × 4; it does not demonstrate padding. | The example now uses 320. Keep byte units explicit, including typed adapters and planar boundaries. |
| Constructor validation | Length checks are not a complete guarantee of valid geometry, descriptor consistency or zero-area behavior. Prior review reproduces failures around minimal final-row extents and empty views. | Checked arithmetic, minimal visible extent `(h-1)*stride + row_bytes` for nonempty views, aligned nonempty row starts, and well-defined empty typed views. |
| Data access | “Full backing” overpromises; `row_with_stride` cannot universally promise padding after the last visible row. | Document view extent separately from allocation extent and visible row bytes separately from padding. |
| Dimensions | `height()` is on the buffer; views currently use `rows()`. The statement that the exact accessors exist on all three types is false. | Fix examples without introducing a gratuitous accessor migration. A row iterator needs a distinct name because `rows` is already occupied. |
| Allocation | “Un-prefixed forms panic” incorrectly includes fallible `from_vec`/`from_pixels`. | Name the panicking convenience constructors; retain explicit fallible allocation and adoption. |
| Typed/erased interop | Zero-cost type erasure does not establish semantic agreement or make all adapters copy-free. | Keep established rgb/imgref imports. Validate typed layout and preserve context/stride, with borrowed metadata access to avoid Arc churn in row loops. |

## Conversion examples and CMS

The introductory conversion loop drops `src`'s color context by planning only
from its descriptor. The custom-ICC example accepts a CMS object but passes no
ICC profile bytes. Neither demonstrates the advertised arbitrary-profile behavior.
`best_match` is a format-cost choice, not proof that the selected conversion is
supported under the supplied color context and preservation requirements.

The replacement quick start should resolve the current source encoding once,
build a complete immutable plan, prepare mutable execution state once, and use
fallible execution into caller-owned output. Proposed execution shape:

```rust,ignore
// PROPOSED: plan construction uses resolved encodings and explicit policy.
let mut worker = plan.prepare(src.width())?;
worker.convert_slice_into(src.as_slice(), dst.as_slice_mut())?;
```

This replaces the blanket “no per-row allocation” claim with a testable contract:
preparation reserves scratch and initializes CMS state; calls within the prepared
width do not grow storage. Another worker is prepared independently, not cloned
behind a shared mutex. Checked boundaries are compatible with lean inner kernels.
One-shot helpers document their preparation cost.

The present `to_rgb8`/`to_rgba8` recommendation hides panicking/permissive behavior.
Show the explicit fallible policy path first. Retain legacy helpers in the bridge
with warnings only when their final migration destinations are implemented.
The “universal hub fallback” claim becomes a supported-route description: refuse
unsupported combinations instead of promising every descriptor pair can execute.

CMS requests must carry actual source and destination profiles, sample formats,
intent and supported policy, with fallible rows and honest depth support. A backend
that supports u8/u16/f32 separately does not automatically support arbitrary
cross-depth pairs. Plan composition must preserve the requested sequence, including
CMS operations; endpoint replanning is not generally equivalent.

## Descriptors, ICC, CICP and provenance

- `PixelFormat` remains the interleaved physical layout. `PixelDescriptor` does
  **not** contain everything needed for color: arbitrary ICC and relative-linear
  luminance anchors need additional information. YCbCr also needs matrix/range
  and chroma geometry/siting. Keep raw descriptors convenient; validate when used.
- F16 “descriptor-only” means no current typed `Pixel` implementation, not absence
  of conversion kernels. Avoid making future stable-f16 availability a release plan.
- The constants table is useful. Preserve explicit alpha distinctions, including
  Undefined padding lanes. A typed `Rgba<u8>` says nothing about its transfer or
  alpha association. Validate contradictions rather than assuming a named preset
  resolves arbitrary attached metadata.
- `ColorContext` cloning is cheap relative to an image copy but Arc cloning is
  still an atomic operation. Borrow current context in views and resolve authority
  once. Define how descriptor/CICP/ICC disagreement is rejected or explicitly
  resolved, and preserve unknown/custom profiles rather than inventing sRGB.
- `ColorOrigin` describes history. Re-embedding its ICC after changing pixels is
  correct only after conversion back to that encoding, or proven identity.
- CICP-to-ICC synthesis is profile construction, not YCbCr-to-RGB conversion,
  range expansion, chroma reconstruction or a universal CMS capability promise.
  Qualify “any assigned H.273 combination” with actual supported results and
  limitations. Unsupported interpretation must remain explicit.

## Orientation and in-place operations

Keep the useful EXIF enum/composition examples and existing free orientation
functions. They do not need cosmetic relocation onto buffer methods.

The “all allocation-free” statement is false today: rectangular in-place transpose
allocates `vec![false; n]` in `orient/mod.rs::inplace_transpose`. In Rust that is a
Vec of byte-sized bools, not the packed n-bit visited set described by its comment.
It avoids a second pixel buffer but still uses image-sized scratch. Documentation
must say so until an allocation-free algorithm or explicit scratch contract exists.

`transform_in_place` cannot make arbitrary closure behavior transactional. Geometry
checks on a returned view cannot guarantee that its new descriptor matches the
bytes, and a closure may already have mutated data before failure. The proposed
checked protocol validates the resulting layout and typed invariants; strict
in-place operations preflight support, capacity and content conditions before
mutation. Failing CMS work belongs in separate output if rollback is required.

`try_adapt_in_place` must not present known linear samples as sRGB without changing
their values. Metadata declaration and actual conversion are separate operations.
The reduction scan can establish exact sample conditions; “bit-exact” additionally
needs correct interpretation/ICC handling. Keep the useful content scan while
making its scope precise. Throughput numbers remain benchmark-specific.

## Negotiation, preservation, output and HDR

| Current promise | Problem | Intended result on both release lines |
|---|---|---|
| U8-origin f32 has zero loss returning to U8 | Filtering/resizing can create values off the U8 grid. History is not a content proof. | Use proven value-domain facts or an explicit scan; distinguish heuristic loss from sample preservation. |
| No silent loss; `forbid_lossy` is safe | It still permits clipping; permissive paths and `DiscardIfOpaque` enforcement need repair. | Explicit preservation requirement, checked alpha conditions and refusal of contradictory policies. Deprecate misleading entry points after replacement. |
| Output finalizer prevents pixel/profile mismatch | Existing paths can ignore current ICC and restore original metadata without the corresponding pixel conversion. | One output plan binds conversion to emitted metadata, integrated with zencodec's existing color-emission authority. |
| Finalization as one owning operation | Current borrowed-input finalizer always allocates output, even on identity. | Borrowed identity returns PixelCow; consuming identity moves storage; caller-output writes into supplied storage. Add only forms with real consumers. |
| HDR policy only concerns the experimental mapper | Turning a feature off must not permit implicit clipping; diffuse white, source peak and target peak have different meanings. | Checked finite positive anchor; carry current relative-linear units; explicitly select mapping/assumptions or refuse. |
| Named gamut/Oklab support implies all combinations work | Standalone helpers and planner dispatch have different coverage. | Advertise tested routes; unsupported plans fail before row execution. |

The older tone-mapping and CMS names already have migration paths. Retain those
warnings and feature spellings. Do not add new arbitrary seals, bounds, return
types or feature removals under the source-compatibility promise.

## Pipeline planner

The optional planner predates the accidental 0.2.16 exposures. It is not one of
these three deprecations. Keep higher-level scheduling above the pixel foundation;
avoid silently treating heuristic negotiation as execution validation. If planner
APIs are retired later, supply a real migration and keep the feature name accepted
on the common source surface. Removal without a warning is outside the promise.

## Resource estimation

Implemented now: `estimation-experimental` explicitly opts into the same API.
Without it, 0.2 still exposes the module, four root types and plan methods with
warnings. 0.3.1 is proposed to require the feature without changing opted-in code.
The feature currently adds no dependency or execution overhead and does not
physically compile the estimator out of 0.2; preserving old source requires it.

The README's “shape-compatible with zencodec” statement is too strong: the current
local definitions have diverged. Matching-looking names do not establish identical
fields, builders, units or semantics; no checked cross-crate conversion is supplied
by this claim. Also, not every field of all four types is Option. Core-count scaling
in an estimate does not make a sequential row loop parallel.

Under the complete proposal, describe the modeled operation and ownership mode,
live buffers, prepared scratch, excluded backend allocations, calibration and
uncertainty. A scheduler must not mistake heuristic estimates for a hard memory
bound. Moving the estimators elsewhere is optional; adding a core dependency on
zencodec is unnecessary.

## Planar and YUV

Current `planar` supplies owned layout/container machinery. “Handles YCbCr” does
not mean it supplies a complete borrowed, validated AV1/metric boundary or YCbCr
conversion. Independent strides, valid bits in a storage word, chroma siting and
current color interpretation need explicit contracts. Construction and mutable
buffer replacement also need invariant enforcement.

The concrete SVT, AOM and VMAF callers justify a lean shared YUV view now. CVVDP
adds a conversion consumer, not another native YCbCr consumer. See the
[source-backed recommendation and migration order](yuv-carrier-assessment.md).
Keep the carrier behind `planar`, with no codec or metric dependencies in core.

## Features, docs.rs, build time and remaining sections

- Add the estimation feature to the table now. The serde removal queue and “ahead
  of 0.3.0” HDR wording are historical, not the new compatibility policy. Keep
  recognized feature names/stubs through the migration; 0.3.0 was already yanked.
- Arrange docs.rs around ownership/views, descriptors/current encoding, then
  preparation/execution. Keep common root imports; group CMS, HDR, ICC databases,
  negotiation and planar details in modules. Show legacy items as deprecated,
  and estimation only as a prominent opt-in. The full proposed tree is in the
  migration report. A tidier tree should not require cosmetic import breakage.
- Add visible-row iteration for existing carriers if concrete loops simplify.
  Reuse zencodec's fallible row/strip delivery rather than publishing another
  universal provider trait. YUV strips must synchronize luma/chroma rows and
  account for subsampling phase and reconstruction halos.
- Build numbers are measurements on one host/profile, not a compile-time budget.
  Keep core independent of CMS/SIMD/metric stacks; erase typed wrappers early and
  avoid unnecessary generic duplication. Track cold minimal/default consumer
  builds and incremental costs before claiming improvement. This deprecation
  batch does not claim compile-time speedups.
- MSRV, license and the generated ecosystem footer do not require API-contract
  changes. Test both crate MSRVs for a release; keep the autogenerated footer intact.

## Validation of this batch

`scripts/check-deprecations.py` compiles external consumers in four configurations:
default/minimal × estimation enabled/disabled. It covers all four types, module
access, both methods, inferred estimate and Adapted receivers, the public predicate,
and the warning-free PixelCow replacement. It rejects unrelated compiler errors
as evidence of a successful deprecation check. Existing estimation behavior tests
run with and without opt-in. Wider contract acceptance gates remain in the proposal.

Completed checks for this batch:

- Workspace tests: 1,166 passed, 50 ignored, no failures. After moving three
  estimation type tests out of the deprecated module into its integration suite,
  that suite passed all 15 tests both without and with the feature.
- All 44 external deprecation checks passed; CI now runs them and the opted-in
  estimation behavior suite.
- Minimal-feature convert check, default workspace/all-target Clippy with warnings
  denied, and estimation-enabled/all-target Clippy with warnings denied passed.
- HDR pipeline: 17 tests passed. The additional HDR-enabled strict Clippy run
  fails on existing `chunks_exact_to_as_chunks` lints in unchanged HDR code;
  this batch does not claim that broader Clippy configuration is clean.
- Formatting, whitespace checks and pinned-toolchain API snapshot generation passed.

No release was published and no sibling code was changed. These checks validate
the deprecation batch; they do not validate the proposed replacements or YUV API.
