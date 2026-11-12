# Pixel and conversion design reference

This is the reference for the authorized 0.3 breaking-release consolidation.
Status below distinguishes the intended design from completed implementation.

Start with the [pixel types guide](guide.md) for everyday ownership, subimages,
stride and conversion. This document records the detailed contracts and limits.

## What callers need to know

1. **Storage is not interpretation.** `SampleEncoding` describes one integer
   payload per word, including signedness, precision and placement. A plane
   supplies geometry. Color, range and component roles come from the enclosing
   signal. RGB565 requires explicit unpacking into canonical pixels; existing
   CodePlane views can extract its shared-word bitfields during that conversion.
2. **Representation is not processing support.** Exact integer carriage can
   support more types than numerical conversion. Unsupported arithmetic refuses
   before writing; it never guesses a normalization.
3. **Conversion resolves meaning once.** A prepared conversion owns its operation
   chain and output interpretation. Allocating and caller-buffer APIs execute
   the same conversion and attach the same metadata.
4. **Policy survives layout changes.** Swizzling cannot introduce defaults for
   alpha, luma, HDR rendering or clipping. Callers must provide authoritative
   metadata or explicit choices for conversions needing known color.
5. **Untouched components stay untouched.** Alpha payloads and row padding are
   preserved bitwise when the operation does not change them.

## Internal invariants

- One encoding model; no compatibility alias for the unshipped NativeEncoding.
- Geometry validation and row access reuse Plane, including empty extents.
- Each finalized operation owns its input/output descriptors and scale. No
  separately mutable vector of descriptors accompanies an operation list.
- Output context derives from the resolved conversion, not wrapper-specific
  reconstruction. Pixel descriptor, white anchor and current profile agree.
- Fallible in-place adapters validate target descriptors before rewriting bytes.
- RGB-to-gray luma policy is checked consistently before built-in or CMS
  construction, including BGR input.
- Constructors enforce the same descriptor validity rules. Fallible constructors
  return `DescriptorError`, displaying the supplied sample type, layout, alpha
  interpretation and violated rule. Buffer validation preserves that diagnostic.
- Capture an error at the failing operation. Inline `map_err`/`ok_or_else`
  closures using `whereat::at!` are appropriate. Do not funnel separate failure
  sites through a shared trace-producing closure: that erases which operation
  failed. Small error-construction functions should propagate their caller.
- Generic layout operations do not assume a maximum sixteen-byte pixel.
- ICC identification does not grant permission to replace a profile. Empirical
  rejection cannot be overridden by a structural matrix-shaper classification.
- Fallible stateful-worker cloning stays fallible; unused profile machinery is
  removed rather than exposed for hypothetical callers.

## Numerical ownership

Primitive SIMD/scalar convergence belongs to the separate magetypes work.
This change does not replace powf or promise cross-tier bit equality before
that work lands. Local algorithm differences, discarded options, alpha-bit
rewrites and invalid percentile ranks remain this crate's responsibility.
Tests must distinguish those defects from primitive rounding differences.

## Implementation ledger

| Work | Status | Regression gate |
|---|---|---|
| Integer encoding and existing component views | Shared encoding retained; speculative native adapter removed | CodePlane RGB10/alpha1, code mapping, empty/strided planes |
| Finalized operation chain and policy preservation | Implemented | Compositing/swizzle combinations, prepared and partitioned execution |
| Unified output context | Implemented | Allocating/into/adaptation pixels and white-anchor parity |
| Descriptor and generic-layout invariants | Implemented | Invalid alpha descriptors, U64 channel removal, empty row swaps |
| Payload and histogram algorithm corrections | Implemented | F16 alpha bits, positive percentile ranks; primitive parity separate |
| ICC qualification and API cleanup | Implemented; historical table requalification remains separate | Parametric curve types, failed qualification, fallible clone callers |
| Final release-candidate validation | Local gates passed; remote CI tracked on PR145 | Workspace tests, feature gates, Clippy, API snapshots and CI |

Do not infer completion from an API name or a passing unrelated test. Update
this ledger with the implementation and link specialized detail rather than
copying these rules into several documents. No publication is authorized here.

## ICC qualification limit

The generator now requires empirical evidence, checks all RGB curves, and refuses
truncated/unsupported curves. Existing recognition tables retain historical
qualification flags; this change does not retroactively requalify their corpus.
Their error column is a curve-fit error, not a complete-transform error bound.
Bundle-derived rows remain identification-only. A sampled CMS gate is not proof
of global bit-exact substitution. Do not cite it as one.

## Migration

- Replace `NativeEncoding` with `sample::SampleEncoding`. The constructor keeps
  the storage/bits/shift arguments; its error is `SampleEncodingError`.
- Replace `RowConverter::clone()` with `try_clone()?`. A stateful CMS worker can
  refuse cloning; callers can prepare another worker explicitly.
- `PixelDescriptor::try_new` and `try_new_full` return `Result<Self,
  descriptor::DescriptorError>` instead of `Option<Self>`; use `?` or inspect
  the displayed reason.
- Invalid descriptor alpha/layout combinations now fail consistently in fallible
  constructors and panic consistently in infallible constructors.
- No new feature flags or public planning types are needed. `PlannedOperation`
  and output-interpretation resolution are private implementation details.

## Regression commands

Run `just sample-design-check`, `just plan-design-check`,
`just context-design-check` and `just local-numerics-check` for focused gates.
`just approved-contracts-check` covers the workspace and existing feature gates;
`just api-doc` refreshes the public surface snapshots. Resource wrappers and
platform admission are the caller's responsibility.

## Native component views

The owner directed removal of `NativeCodeSlice` and the conversion/error/padding
APIs introduced solely around it. No production consumer justified that surface.
`CodePlane::component` already reads interleaved U8/U16 components, including
CbCr/CrCb and RGB10 + alpha1; `codes::to_canonical` is the existing conversion
boundary. No replacement view or compatibility alias is introduced.

Signed/wide `SampleEncoding` descriptions and native PixelFormat identity
carriage remain, but do not imply signed/wide component reading or arithmetic.
Any extension of those capabilities needs a concrete consumer first.
See the [explorer-driven review](explorer-design-review.md) for the finding and
outstanding issue scope.

## Validation receipt — 2026-11-09

`approved-contracts-check` passed workspace release library/integration/doc
checks, existing focused feature checks, no-default native carriage and strict
Clippy. `design-feature-check` additionally passed the all-feature library,
new design regressions, histogram tests, generator qualification tests and
strict all-target/all-feature Clippy across all workspace crates. Existing
expectations were not weakened. The positive-percentile correction retains
floor ranks and only excludes rank zero; the existing outlier gates pass.
Public API snapshots were regenerated. Cross-platform CI is tracked in
[PR145](https://github.com/imazen/zenpixels/pull/145); local results are not a
claim that remote CI or historical release branches have been qualified.

## Profile assets and retention

Removing the private, unused `builtin_profiles` implementation was intentional.
It did not supply the exported ICC assets: `icc_profiles` and its bundled profile
bytes remain available. The removal does not remove named-profile export.

The original corpus already exists at the R2 prefix used by `just icc-fetch`.
See [icc-gen corpus retention](../scripts/icc-gen/README.md#retained-corpus-originals).
The 2026-11-09 inventory found 1,224 ICC/ICM objects with all 1,210 manifest
entries present. Creating a new archive is not an established missing task.

The [retention audit](icc-retention-audit.md) locates exact original bytes for
all 412 active and 186 deferred recognition hashes, with full-file digests and
provenance retained in its receipt. The R2 prefix, current assets, historical
bundles and two embedded corpus profiles all contribute; the R2 prefix alone
does not contain every original. All 14 objects absent from the manifest were
tested, and one stale substitution permission was removed. Revalidating every
historical tolerance remains the lowest-priority follow-up. Future runs must
record generator/CMS identity, sample domain and tolerance.

Descriptor/trace follow-up validation: `just approved-contracts-check`,
`just design-feature-check` and `just api-doc` passed. Trace capture stays at the failing operation; inline error-mapping closures
remain supported.

The repository follow-up for authority, HLG backend refusal, P3 exports, API
migrations and allocation ownership is recorded in
[repository issue improvements](repository-issue-completion.md).
The [merged issue re-audit](issue-reaudit-2026-11-09.md) separates completed
contracts from remaining consumer, authority and release work for 83, 132–135
and 150. It includes the RGB565 shared-word route and the review of the former
truncated-ICC grayscale fallback; neither requires another pixel-view type.

## IQA planar consumer review — PR144

The optional zenpsy planar path builds on Plane/AnyPlane, PlanarLayout,
SampleEncoding and ColorContext. Its borrowed PlanarSlice is used by the
prepared metric/JXL consumer; it does not restore NativeCodeSlice or add a
second payload-encoding model. Existing CodePlane::component reads supported
interleaved native components. Signed/wide descriptions still do not imply
reader or metric support.

Storage/layout validation does not resolve color or display policy. The view
retains unknown CICP, ICC authority, matrix/range, siting and alpha without
conversion or allocation. HLG display rendering and luminance anchors remain
explicit metric/converter configuration. Checked prepared calls bind the
interpretation used during preparation; changing pitch alone remains valid.
Original bitstream signaling is not a substitute for the current sample context.

Regression coverage uses odd-sized 4:2:0 CrCb with 10-bit high-aligned words,
independent one-bit alpha, short final rows and untouched padding; existing
component readers recover the exact payloads. Erased preparation/lending checks
retain opaque ICC and unknown signaling, distinguish absent/explicit HDR anchors,
and reject changed preparation interpretation. These are transport/operation
controls, not HDR metric-quality or conversion qualification.

## Current interpretation checks (review #150)

A planar `Alpha` role requires a defined interpretation: Straight,
Premultiplied or Opaque. `Undefined` denotes padding in packed layouts and
cannot describe a real alpha plane. Layouts without an Alpha role accept no
alpha interpretation. This rule also applies to empty planar pictures.

`ColorContext::as_profile_source` selects the current profile. Its predicates
follow the same selection: an ICC-selected context has unknown transfer and
is not established to be sRGB, even when a secondary CICP field is present.
These convenience methods do not identify ICC bytes. Decoders select the
format's authority once and retain original claims in `ColorOrigin`.

The #134 follow-up applies this rule to grayscale reduction too. An unusable
ICC, including an empty/truncated declaration, cannot be replaced from companion
CICP. The rewrite keeps RGB and the same context; independently exact alpha
removal and full-range U16-to-U8 narrowing can still proceed. Raw content
analysis still reports R==G==B truthfully. Qualified ICC replacement with an
agreeing companion and CICP-only grayscale reduction remain supported.

Explicit canonical ICC replacement uses paired fingerprints only as a filter.
It then requires `icc::normalized_eq` against the retained canonical bytes:
length, every tag and rendering intent must match; only the shared five
metadata-only header ranges are ignored. The fingerprint and equality APIs
share that normalization predicate. Approximate table recognition still has its
separate qualification and intent limits; this is not strict preservation of
profiles merely judged close by historical tolerance measurements.

The remaining work from closed #150 is tracked in
[issue #154](https://github.com/imazen/zenpixels/issues/154): current processing
requirements versus source origin and lower-priority historical approximate ICC
qualification. Whole-operation memory admission belongs to the executor;
capacity/peak documentation states
what the existing planner actually computes; it does not add those models.
The separate [allocation audit](allocation-audit.md) records the merged #151
allocation fixes, failure-injection controls and named infallible exceptions.
Fallible allocation does not provide a whole-operation admission budget.

## Owned and writable planar storage

`PlanarBuffer` adopts independently strided U8/U16/F32 plane allocations and
owns their layout and current color interpretation. `PlanarSlice` and
`PlanarSliceMut` borrow descriptors from the owner without allocation or pixel
copies. Mutable access reborrows samples and cannot replace the validated
storage descriptors. Caller-owned writable plane lists use the same checks.
See [planar types](planar-types.md#plane-views-picture-views-and-allocator-ownership--pr144)
for the iterator accessor migration and custom-owner limits. Timing, frame
presentation and shared-frame copy-on-write policy stay outside zenpixels.
