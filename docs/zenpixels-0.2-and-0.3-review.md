# zenpixels / zenpixels-convert: consolidated 0.2.x and 0.3 review

2026-09-27 · reviewed main baseline `5665d6f` · proposed next minor: **0.3.1**.

**Latest implementation/decision:** `try_from_parts`, `FromPartsError::take_parts`
and `without_buffer` are now implemented. The entire legacy planar module is
deprecated; its proposed extension APIs are deferred pending video design.
For runnable current behavior and desired assertions, open the
[code-first review](code-review-0.2-and-0.3.md).

This is the single reading document for our current proposals. It consolidates
contracts, API cleanup, consumer migration, docs.rs, streaming, YUV and missing
PR #63 work. Detailed source evidence is linked at the end. No new production
code is implemented by this document, and proposed signatures are not approvals.

**Status key:** **Done** = committed on main, not released. **Proposed** = recommended
work whose interface/implementation still needs review. **Optional** = add only
with a concrete adopter and benefit. **Keep** = preserve the existing contract.

## 1. The release promise and working order

**Chosen:** the same migrated consumer source builds with the completed 0.2 bridge
or 0.3.1, with **one zenpixels version per connected pixel pipeline**. This does
not promise direct exchange between simultaneously loaded 0.2 and 0.3 types.
No shared-types crate or facade is needed for that chosen model.

Most design and correctness work belongs in 0.2. The 0.3.1 release should remove
warned-about alternatives while preserving the migration destination. A warning-free
consumer must not encounter surprise trait sealing, field privacy, feature removal,
changed defaults or altered method signatures merely by selecting 0.3.1.
Deprecations alone do not establish this: paired consumer builds and behavior tests do.

The minimum compatible 0.2 version is the patch that actually finishes the common
surface; do not promise that it will all fit into 0.2.17. Keep a 0.2 maintenance
line. Use 0.3.1 rather than the already-yanked 0.3.0 as the destination.

Suggested review/implementation sequence:

| Chunk | Work | Release placement |
|---|---|---|
| A | Fix known-transfer relabeling; recover selected PR #63 correctness work | Immediate 0.2 fix |
| B | Checked parts adoption, allocation reuse, storage/typed-layout invariants | 0.2, identical in 0.3 |
| C | One current color interpretation and complete CMS inputs | 0.2, identical in 0.3 |
| D | Complete plans, prepared fallible workers and faithful composition | 0.2 replacements; retire warned legacy APIs in 0.3 |
| E | Output ownership, matching metadata, preservation and HDR semantics | 0.2 replacements; retire warned legacy APIs in 0.3 |
| F | **Done:** deprecate planar; defer video design | Keep legacy feature/code until migration exists |
| G | Consumer migrations, docs.rs, release fixtures and selected cleanup | Finish bridge before removal release |

Correctness fixes can ship independently in small patches. The table is a work
order, not a reason to hold an urgent fix until every design is complete.

## 2. Already implemented

- **Done:** deprecate public `requires_cms`; planning/execution determines whether
  an actual conversion can run. Internal predicate retains existing behavior.
- **Done:** explicitly deprecate `Adapted::as_pixel_slice`, closing the inferred
  receiver warning gap. The replacement is `PixelCow::as_slice`.
- **Done:** add `estimation-experimental`. Without opt-in, 0.2 keeps the published
  estimator module/types/methods with warnings; with opt-in, the same API is
  warning-free. In 0.3.1 require the feature without renaming opted-in APIs.
- **Done:** external fixtures check 44 deprecation cases, including inferred uses
  and minimal/default configurations; CI runs them.
- **Done, chosen:** `DiffuseWhite::new(f32)` remains const and panics for nonfinite
  or nonpositive values. Keep its name; no constructor deprecation or `try_new`
  addition is proposed for this chunk.
- **Done, chosen:** `PixelBuffer::into_parts()` moves allocation and metadata into
  a destructurable `PixelBufferParts`.
- **Done, chosen:** `PixelBuffer::into_contiguous()` compacts rows in the original
  allocation and returns the buffer, retaining type and metadata.
- **Done:** README annotations, consumer search/audit, YUV assessment and complete
  21-commit PR #63 disposition ledger. These are reviews, not implemented fixes.

The original `PixelCow` and three `*_cow` adapters were already on main before
this work. Keep them, fixing their interpretation bugs rather than duplicating them.

## 3. Storage and ownership

### B1. Keep large allocations with their interpretation

**Done extraction and adoption:**

```rust,ignore
// OLD: backing bytes alone lose offset and interpretation.
let data = buffer.into_vec();

// NEW, implemented: stride-aware ownership transfer, O(1).
let PixelBufferParts {
    data, offset, stride_bytes, width, height, descriptor, color_context, ..
} = buffer.into_parts();

// NEW, implemented: compact only if the receiver needs packed rows.
let parts = buffer.into_contiguous().into_parts();
let pixels = &parts.data[parts.offset..];

// Implemented; release rejected storage before retaining/boxing an error.
let buffer = PixelBuffer::try_from_parts(parts).map_err(|e| e.without_buffer())?;
```

Extraction preserves Vec pointer, length, capacity, offset, stride and existing
context ownership, without allocating, copying pixels or cloning the Arc.
Compaction preserves pointer/capacity/context but can move O(image bytes), and
truncates length to offset plus packed size. **Contiguous does not mean offset
zero:** retain the alignment prefix so wide samples remain aligned.

Adoption applies the existing view storage checks to allocation, alignment and
geometry. Proposed descriptor/color consistency checks remain separate work.
On rejection, return the supplied parts with the error; do not discard a
40–100 MB allocation. `FromPartsError` holds the traced cause and optional parts; `take_parts()`
recovers them once and `without_buffer()` discards them before propagation.
Unchanged parts round-trip without a copy. Parts are intentionally mutable,
unvalidated data; the reconstructed buffer establishes the invariant.

The current non-exhaustive parts record supports downstream destructuring and
editing exported parts, not external struct literals. Settle the smallest way to
adopt independently decoder-owned allocations before adding constructors.
Typed adoption/export must respect allocator alignment, not just pointer alignment.

**Proposed:** deprecate `into_vec` after the complete migration is ready; remove
it in 0.3 if migrated. Do not add the PR's bytes-only `into_contiguous_bytes` or
private `PixelBufferLayout` wrapper.

**Proposed PR #63 recovery:** reuse castable padded U8 allocations in
`into_contiguous_pixels`; copy wide/incompatible-capacity allocations directly
once. Move an owned cow's storage into legacy `Adapted` instead of copying the
whole image. Handle offset explicitly at any packed Vec exit.

### B2. Define and enforce geometry and typed layout

**Proposed correctness contract, shared by both lines:** byte strides; pixel
dimensions; native-endian wide samples; checked arithmetic. A positive-size view
needs `(height - 1) * stride_bytes + width * bytes_per_pixel` bytes, with no
requirement for trailing padding after its final row. Operations needing full
padding must validate that extra requirement explicitly.

Zero-area views are empty; valid zero-width rows yield empty slices. Do not
calculate out-of-bounds offsets or require alignment for unused samples. Empty
typed access must still return an appropriately aligned empty typed slice.
Validate nonempty row alignment, stride, formats and allocation overflow.

```rust,ignore
// OLD problematic contract: typed-preserving relabeling can contradict P.
let changed = typed_rgba.reinterpret(bgra_descriptor)?;
// It must not remain a typed RGBA view over bytes described as BGRA.

// NEW intended sequence (schematic, final method names undecided):
// erase type -> checked reinterpretation -> explicit checked retyping if valid.
```

**Proposed:** add an erased-returning checked route, then deprecate the old
`reinterpret` contract. Layout-changing mutation must likewise preserve typed
invariants or explicitly erase the type. `transform_in_place` cannot promise
rollback after an arbitrary caller closure has already changed bytes.

**Keep:** `PixelBuffer`, `PixelSlice`, `PixelSliceMut`, `PixelCow`, and open `Pixel`.
A type parameter describes physical pixel layout, not complete color semantics.

## 4. Descriptors and current color

### C1. Validate declarations at acceptance boundaries

**Proposed:** keep `PixelDescriptor` as the convenient small public declaration.
Validate contradictory combinations when accepting storage or preparing execution,
such as premultiplied alpha with no alpha channel. Keep unknown color representable.
Invalid dynamic input should fail at a checked boundary rather than construct a
buffer that later panics or lies about its samples.

**Keep:** existing descriptor fields, constants and ordinary constructor signatures.
Making fields private is not reliably covered by field deprecations: struct update
syntax can avoid warnings. No universal new descriptor type solely for privacy.

### C2. Resolve current interpretation once

**Proposed:** one lightweight borrowed resolved input, provisionally
`PixelEncoding<'a>`, combining validated descriptor, selected profile and relevant
luminance anchor. Private fields protect the resolved result. Built-in encodings
should require no heap allocation; borrow ICC during inspection and retain its
ownership once in an owning plan.

| Information | Role |
|---|---|
| Descriptor | Storage layout and declared sample semantics |
| Resolved encoding | One current interpretation, including ICC/alpha/range/anchor |
| `ColorOrigin` | Source history and retained original signaling |
| Output metadata | What must accompany the pixels being encoded now |

Resolve conflicting ICC/CICP explicitly or reject ambiguity. ICC is not generally
equivalent to its embedded CICP. Unknown does not silently mean sRGB. Attaching a
profile is a declaration, not conversion. Preserve unselected source tags as
history; update current encoding after conversion.

```rust,ignore
// OLD: descriptor-only planning can omit an attached custom ICC profile.
let converter = RowConverter::new(src.descriptor(), target_descriptor)?;

// NEW contract, schematic:
// resolve source and target including actual profiles/anchor -> build plan
// -> prepare worker -> execute with that same interpretation.
```

**Immediate fix A:** the cow adapters currently accept known sRGB bytes and borrow
them unchanged with a linear descriptor. The PR #63 `Unknown` guards are missing
from both intent and explicit-policy variants. Port both, test policies and
legacy wrappers, and audit in-place fast paths for the same error. Known transfer
changes must convert or fail. Whether/how unknown color is assumed belongs in
explicit resolution, not a universal color identity claim.

**Optional internal optimization:** borrow context through ordinary views instead
of incrementing an Arc on every reborrow. The existing getter already borrows;
do not add a duplicate getter. Retain owned attachment and sound view lifetimes;
measure view size/branch costs. Do not promise all clones are atomics-free or make
`PixelSlice` Copy as a separate goal.

## 5. Complete plans, prepared workers and CMS

### D1. Prepare once; execute fallibly

**Proposed signatures, not implemented:**

```rust,ignore
impl ConvertPlan {
    pub fn prepare(&self, max_width: u32)
        -> Result<PreparedConverter, At<ConvertError>>;
    pub fn then(&self, next: &Self) -> Result<Self, At<ConvertError>>;
}
impl PreparedConverter {
    pub fn try_convert_row(&mut self, src: &[u8], dst: &mut [u8], width: u32)
        -> Result<(), At<ConvertError>>;
    pub fn convert_slice_into(&mut self, src: PixelSlice<'_>, dst: PixelSliceMut<'_>)
        -> Result<(), At<ConvertError>>;
}

// OLD repeated raw geometry; fallible wrapper, infallible row execution,
// potentially lazy allocation:
converter.convert_rows(src_bytes, src_stride, dst_bytes, dst_stride, width, rows)?;
// NEW proposed shape:
let mut worker = plan.prepare(src.width())?;
worker.convert_slice_into(src, dst)?;
```

Keep an immutable cloneable plan if its existing public contract can be preserved.
It must retain every operation, CMS recipe and required profile. Preparation
selects kernels and initializes/reserves execution state. Work within prepared
capacity must not allocate, lock shared mutable scratch or resolve profiles per
row. Larger input requires explicit preparation or an error. Typed wrappers erase
early; no per-pixel policy dispatch.

Workers are mutable and not Clone. Prepare another independent worker from the
plan; share immutable tables where useful. Raw rows are bound to the plan's
encodings; image-view boundaries validate both source and target interpretation,
geometry and policy. Caller-owned output can be partially written on backend
failure; do not report successful ready output after that failure.

**Proposed:** retain old `RowConverter` while completing the replacement, then
warn on the whole legacy type if its Clone/hidden-state contract must be retired.
A new `try_clone` alone cannot repair the old infallible Clone. If the existing
plan cannot be repaired compatibly, migrate a whole replacement plan in 0.2 too.

### D2. Composition preserves the sequence

```text
OLD bug to reject: F32 -> U8 -> F32 becomes identity because endpoints match.
NEW: retain quantization to the U8 grid; retain clipping, alpha and CMS stages.
```

`then` returns an error for incompatible sequences. Legacy `compose -> Option`
must return None rather than silently erase work. Approximate fusion needs a
separate explicit numerical contract. This affects zenpipe, which removes a
transform stage when composition reports identity.

### D3. Replace incomplete CMS interfaces once

**Proposed:** final fallible backend interfaces introduced in 0.2 and retained
unchanged in 0.3. Requests carry actual source/target profiles, sample layout/depth,
alpha, range, luminance, intent and allowed changes. Reject unsupported pairings
before execution. Backend refusal differs from failure after accepting work;
an execution error must not silently fall through to a different backend.

Prototype the smallest preparation/factory plus mutable-executor shape with
MoxCms and a real external implementation. Add a third factory trait only if
retained third-party ownership requires it. Old infallible adapters cannot invent
errors never reported by their implementation.

Deprecate existing `PluggableCms`/row-transform families only when final
replacements and migrations exist. Do not add `try_transform_row` as a temporary
bridge destination and then rename it in 0.3. Preserve `whereat` and meaningful
error variants; no cosmetic error consolidation or surprise auto-trait changes.

## 6. Conversion ownership, output and preservation

### E1. Give ownership operations honest costs

| Operation | Required meaning |
|---|---|
| Borrow-or-own adaptation | Borrow on equivalent interpretation; otherwise own converted pixels |
| Consuming conversion | Move storage on identity; reuse supported paths; documented allocation fallback |
| Caller-output conversion | Use supplied destination; preparation may allocate scratch |
| Strict in-place | Preflight support and capacity; no replacement image; reject unsupported work |
| Allocating conversion | Produce independent owned storage |

`convert_into`, `into_converted`, `convert_in_place` and existing `convert_to`
are candidate vocabulary, not a mandate to publish four wrappers. Add only with
real consumers. New methods on existing open traits require compatible provided
implementations or a separately justified new interface.

Strict in-place operations must preflight content-dependent requirements such as
opaque alpha before mutation. Arbitrarily failing CMS execution cannot also offer
rollback with no extra storage. PR #63's widening `convert_in_place` allocates a
replacement image and its narrowing path allocates row scratch; do not copy its
name/guarantees unchanged. Existing rectangular orientation uses a visited array:
“no second pixel buffer” is not “allocation-free.” Keep free orientation APIs.

### E2. Bind resulting pixels to emitted metadata

**Proposed:** prototype one output plan with zencodec's existing
`resolve_color_emit` / `ColorEmitPlan`, avoiding competing authorities. Provisional
`OutputPlan` / `PreparedOutput` names remain undecided. The source audit found no
external finalizer callers; land a real integration before adding public wrappers.

```rust,ignore
// Proposed ownership forms; implement only those adopted by real callers.
output_plan.adapt(src_view)?;                  // PixelCow + matching metadata
output_plan.into_output(owned_buffer)?;        // owned EncodeReady
output_plan.write_into(src_view, dst_view)?;   // metadata for written pixels
```

```text
OLD wrong result: working-space pixels + original ICC restored without conversion.
NEW: convert to the chosen output/original encoding, then emit matching metadata.
```

`SameAsOrigin` must perform that conversion or prove identity. Actual ICC bytes
must reach CMS. Range, alpha and anchor affect identity. Report unsupported exact
metadata preservation. Borrowed identity avoids image allocation; consuming
identity retains storage. Caller-output identity still copies into its destination.
Metadata/setup costs are separate from pixel-copy costs.

Deprecate/remove old finalizers only after ICC/HDR parity. The deprecated
`ColorManagement` route still covers behavior missing from the newer finalizer;
a newer name is not proof it is safe to remove the old path.

### E3. State precisely what “preserve” means

**Proposed:** replace misleading `forbid_lossy` assurances with an explicit sample
preservation requirement, possibly on existing options. `preserve_samples()` is
only a candidate spelling. Require an operation/domain proof or an explicit scan;
reject contradictory options rather than silently relax preservation.

```text
OLD assumption: originally U8 -> resize in F32 -> returning to U8 is zero loss.
NEW: resized values may leave the U8 grid; prove representability or permit rounding.
```

Distinguish reversible representation changes, conditional preservation such as
opaque-alpha removal, permitted numerical changes, and codec preservation of its
input. Enforce `DiscardIfOpaque`; do not treat a heuristic loss estimate of zero
as proof. Arbitrary floating-point gamut/transfer operations are not sample identity.

### E4. HDR parameters have separate meanings

**Done:** checked, panicking `DiffuseWhite::new`, same signature on both lines.
**Proposed:** carry what relative-linear 1.0 means in nits separately from source
peak, target peak, measured content levels and mastering history. Missing inputs
require refusal or an explicitly named assumption. HDR-to-SDR must refuse missing
policy even when experimental mapping is disabled; never silently clip because a
feature was turned off. Validate absolute values against independent references.

**Keep:** `hdr-experimental`, existing measurement algorithm names including
`CllMeasure::measure_max` and percentile variants, and useful HDR carriers/builders.
No rename merely because a deprecated method name becomes available. Moving every
tone mapper, HLG policy or gain-map algorithm into stable core is not proposed.

## 7. Planar storage and YUV

### F1. Deprecate the old module now; design video separately

**Done by owner decision:** deprecate the whole `planar` module, including root
and convert re-exports and inferred methods. Keep the feature and implementations
available. This supersedes the earlier suggestion to add `try_new`, `plane_mut`
and `try_replace_plane` to the old container. No replacement is published now.

A refreshed local search found zenfilters uses `PlaneMask` in three source files,
including public `ChannelAccess` fields. Its Oklab planes are local types. Migrate
that filter-specific mask deliberately before removing the module; absence of
MultiPlaneImage consumers did not mean absence of all planar-module use.

### F2. Video requirements to revisit with actual code

Deferred design inputs remain independent byte strides, significant bits separate
from U8/U16 storage, Y/Cb/Cr roles, odd subsampled extents, crop phase, chroma siting,
matrix/range/transfer/primaries and explicit unknowns. Borrow AOM/SVT allocations;
matching AOM-to-VMAF planes need no RGB roundtrip. CVVDP expects RGB and needs an
explicit range/matrix/chroma/display conversion. No universal owned container or
new YUV API is added in this chunk. See the [code review](code-review-0.2-and-0.3.md)
and historical [YUV caller assessment](yuv-carrier-assessment.md).

## 8. Streaming and row iteration

**Keep existing lending providers:** zencodec's fallible `next_batch` and zenpipe's
fallible `Source::next` can borrow reusable internal scratch. An ordinary Iterator
cannot express a fresh item lifetime tied to each mutable provider borrow. Do not
introduce a third universal provider trait in core without evidence it replaces
existing complexity.

**Optional:** `row_iter()` over an already-resident view, and `row_iter_mut()` if
needed by a real loop. Existing `rows()` names are already occupied. Iterate
visible samples only, handle absent final padding and zero-width rows, and retain
metadata on the enclosing view. This reduces stride boilerplate, not loop cost.

```rust,ignore
// Existing, already allocation-free:
for y in 0..view.rows() { consume(view.row(y)); }
// Optional proposed convenience:
for row in view.row_iter() { consume(row); }
```

**Proposed companion work in zencodec:** fallible pull encoding so provider errors
do not become EOF or panic. Preserve cause through static and dynamic APIs. Define
sequential/random access, progress, capacity, premature EOF, cancellation and
stable terminal EOF. Carry frame color through batches/sinks.

**Proposed zenpipe fix:** CallbackSource allocates each row and appends before
checking whether the callback produced a row. Reuse scratch and append only on
success; preserve errors and reject premature known-height EOF. This is a source
review finding, not a newly executed reproduction in this document.

YUV strips must synchronize luma/chroma rows, phase, odd tails and reconstruction
halos. Adapt the existing lending protocols rather than expose unqualified tuples
of three row slices.

## 9. Complete cleanup disposition

Warnings require working replacements; 0.3 removals require tested migrations.

| Surface | 0.2 action | 0.3 disposition |
|---|---|---|
| `requires_cms` | **Done:** deprecate | Private internal predicate only |
| `estimate`, estimator root types, plan methods | **Done:** conditional warnings / opt-in | Require `estimation-experimental`; opted-in source unchanged |
| `Adapted`, its methods, three non-cow adapters | Keep warnings; complete cow semantic parity | Remove deprecated family after parity |
| `into_vec` | Complete ownership replacement, then warn | Remove if migrated |
| Typed-preserving `reinterpret` | Add checked erased/retyped route, then warn | Remove old contradictory contract |
| `RowConverter`, `convert_rows`, raw `convert_buffer` | Final prepared/ownership replacements before warnings | Remove migrated legacy APIs |
| Free `convert_row` | Direct repeated work to prepared execution | Decide one-shot need; no temporary associated-method rename |
| Legacy CMS traits and finalizers | Complete fallible contracts and ICC/HDR parity | Remove only after backend/caller migration |
| `ColorContext::from_icc_and_cicp` | Explicit authority resolution; remove suppressed ambiguous fallback in consumers | Retire ambiguous constructor after complete migration |
| Useful `with_icc` / `with_cicp` attachment builders | Validate/resolve ambiguity at acceptance boundaries | Keep unless a concrete replacement is required; no blanket deprecation |
| `DiffuseWhite::new` | **Done:** validate and panic on invalid input | Keep unchanged |
| `ContentLightLevel::measure` | Existing measurement replacement, remove suppressed warnings in callers | Remove deprecated scalar method |
| Naive Reinhard/exposure helpers | Retain existing migration to zentone | Remove deprecated helpers after migration |
| `ByteOrder` / `byte_order` | **Optional:** `ChannelOrder` / `channel_order` aliases plus deprecations | Remove old spelling only if this cleanup is adopted; no endian/representation change |
| Transfer-blind ICC profiles/helpers, including legacy profile spellings | Exact supported profile or synthesis destination; explicit unsupported cases | Remove only with accurate migration, not a misleading profile alias |
| Legacy `planar::Plane` | Separate review with an actual replacement/adopter | No automatic deletion based on old queue |
| Entire legacy planar module | **Done:** module-wide warning; defer new video design | Remove only with a usable migration, including zenfilters PlaneMask |
| Analysis-only `pipeline` surface | Separate scope review; predates 0.2.16 | Retire only properly deprecated items; keep feature name |
| `serde` no-op feature | Document existing no-op; no claim that it implements serde | Keep cheap accepted spelling through migration |
| `fast-transpose`, `hdr-experimental`, other accepted features | Preserve/test current opt-ins and defaults | No silent default flip or feature-name deletion |
| `ConvertError`, whereat, existing root imports | Preserve useful result/match/import contracts | No cosmetic consolidation or tracing removal |
| `Pixel` and existing open extension traits | Keep implementation contracts | No opportunistic sealing or required-method changes |
| Free orientation helpers | Fix interpretation retention; document scratch | Keep names and paths |
| Registry internals / `ZenCmsLite` queued demotions | Their modules were already private in published 0.2.16 | No public migration to invent |

The legacy ICC queue includes `REC2020_V4`, `ADOBE_RGB_V4`, `PROPHOTO_V4`,
`icc_profile_for_primaries`, and MoxCms's `lut_transform_opts` /
`cicp_transform_opts`. Check public reachability and actual semantics item by
item: `ADOBE_RGB` and `transform_opts` are existing candidate destinations;
ProPhoto has no canonical bundled replacement, and primaries alone cannot select
a transfer-correct profile. Do not remove a name on the strength of the queue alone.

Already-added errors, checked `InPlacePixels::try_new`, `HdrConfig`, measurement
and feature-gated HDR APIs are not immediate deprecation targets merely because
PR #63 missed main. The old removal queue is not a list of approved removals.

## 10. PR #63: what to recover and what to decline

All 21 commits lack direct ancestry/patch equivalence on main, but release work
manually copied some functionality. Never cherry-pick the full stack as a unit.

**Recover/adapt:** both transfer guards; typed U8 allocation reuse; owned cow
compatibility storage moves; useful private compaction sharing; checked adoption
reference/tests (rewritten for our parts/error ownership); accurate allocation
docs; tested galleries and examples execution; missing minimal `rgb,std` coverage.
Regenerate snapshots for actual adopted APIs. Fix the PR's nondeterministic
misalignment test fixture before reusing it.

**Already copied or superseded:** PixelCow/cow-first adaptation, explicit
`Adapted::as_pixel_slice` warning, estimation opt-in bridge, `requires_cms` warning,
our `into_contiguous` + parts, and much of the test configuration.

**Decline/defer:** duplicate contiguous-view constructors; canceled `new_packed`;
bytes-only export and private layout wrapper; incomplete conversion guarantees;
per-call allocating `ConvertPlan::convert_row` as a migration destination;
orientation-method churn; trait sealing; expanded tolerated-break policy.

Exact existing traits the PR would seal: `PixelBufferConvertExt` and indirectly
`PixelBufferConvertTypedExt`. New sealed orientation traits were
`PixelSliceOrientationExt` and `PixelBufferOrientationExt`; neither reached main.
The spelling migrations were free row/orientation functions → methods,
`convert_rows` → view-based conversion, and unpublished `new_tight` →
`new_contiguous`. Only the view-based conversion change improves the contract
rather than just syntax. The already-shipped `_cow` migration remains useful.

## 11. docs.rs, README and compile/runtime cost

**Proposed navigation**, preserving established import paths:

```text
zenpixels
  PixelBuffer / PixelSlice / PixelSliceMut / PixelCow
  Pixel / PixelDescriptor / PixelFormat / BufferError / Orientation
  color      current encoding, context, origin
  hdr        anchors and metadata, no execution engine
  planar     deprecated legacy surface; video replacement deferred
  icc        optional inspection
  policy     data-only policies

zenpixels_convert
  ConvertPlan / PreparedConverter / ConvertError
  primary conversion extension traits
  cms / output / hdr / orient / icc_profiles
  advanced operations and explicit experimental estimation
```

Select canonical re-exports and inline their documentation. Hide implementation
module duplication only after links and canonical pages work. Preserve existing
paths and the convert-to-core re-exports for compatibility; reduce landing-page
noise without forcing import changes. No new prelude or module per helper.
Choose explicit public docs.rs features, not benchmark/internal all-features.

Examples should cover borrowing strided pixels, checked allocation, allocation
handoff, one conversion, repeated prepared rows, and encoding with matching color.
Adapt PR #63's tested galleries and run them in CI. Each API states stride units,
metadata authority, setup/allocation costs and failure/mutation behavior.

Correct remaining README/crate-doc claims: packed allocation is not guaranteed
SIMD alignment; rows need not contain final padding; descriptors do not carry all
ICC/HDR meaning; custom-ICC examples must pass profiles; arbitrary conversion
routes are not universally supported; orientation can allocate scratch; estimator
figures are heuristic, not hard bounds or automatic parallelism. Keep the useful
constants/orientation examples, MSRV/license details and generated ecosystem footer.

**Proposed measurements:** minimal/default cold consumer builds and incremental
cost on a fixed toolchain; x86 and ARM conversion checks; allocation counts and
pointer retention. Keep core free of CMS/SIMD/metrics dependencies. Erase typed
wrappers early and avoid generic code duplication, locks and unnecessary atomics.
Do not introduce unsafe manual POD implementations or more public crates solely
on the basis of the earlier single-run timings. No new speedup claim is established
by these proposals.

## 12. Consumer migration and release gates

The earlier ripgrep inventory covers API boundaries, conversions, manifests and
aliases, with a manually reviewed caller ledger. It is not semantic proof about
every regex match or every historical published consumer. Do not repeat searches
as though no inventory exists; refresh affected consumers as each chunk lands.

| Consumer group | Migration focus |
|---|---|
| squintly AVIF export | Parts/packed ownership, actual sample format |
| dvifmish, zensim, zenmetrics, zenanalyze, zeneditor, filters/tools | Prepared conversion, error propagation, context/anchor retention |
| zenpipe public ops/sources | Stored worker/public boundaries, faithful composition, independent CMS workers |
| zencodec | Current-color authority, output metadata integration, fallible pull encoding |
| Existing cow-using AVIF/JXL encode paths | Semantic identity and output metadata; no mechanical rename |
| Codec/resize/raw/HDR carrier boundaries | Keep types/imports; verify stride, alpha, depth and current interpretation |
| AOM/SVT/VMAF/CVVDP | Shared YUV views and explicit RGB conversion where required |

Maintain a release ledger for each removal:

`old API -> replacement -> first bridge version -> warning -> migrated caller
-> identical 0.3 signature -> semantic regression coverage`.

Require the same downstream source with deprecations denied against each line,
including trait implementations, generic bounds, public buffer APIs, matches and
feature forwarding. Run controlled libraries as checked roots so dependency lint
capping cannot hide warnings. Include decoder → filter → encoder graphs, fresh
and upgraded lockfiles, and every core/convert pairing admitted by manifests.
A wide requirement does not guarantee Cargo unifies incompatible minor lines;
verify the resolved graph and actual buffer exchange.

Once tested, consumers can explicitly admit the complete bridge and 0.3; for
example `>=0.2.17, <0.4.0` **only if 0.2.17 really is that complete bridge**.
Do not admit untested future minors or use bytes/casts to disguise split type graphs.

Test minimal/default/interoperability features externally, with/without CMS/HDR,
no_std, both MSRVs, rustdoc/doctests, packaging and semver/API deltas. Workspace
self-dev-dependencies can mask missing-feature failures. Cover malformed/empty
geometry, custom/conflicting ICC, alpha/range, independent HDR values, failed
adoption ownership, allocation-free prepared calls, identity storage, and backend
partial-output behavior. Every correctness change gets explicit release notes.

Prior implementation checks passed core default/minimal/all-feature tests,
deprecation fixtures and the documented ordinary Clippy configurations. Optional
HDR strict Clippy has existing failures in unchanged code. Cross-version paired
consumer builds, sibling migrations and release package/semver checks remain
outstanding. Do not present earlier tests as validation of unimplemented contracts.

Prepare bridge and removal PRs in explicit order; ensure merges actually reach
main, avoiding the PR-into-old-base mistake. Commit each completed chunk. Publishing
and widening consumer requirements follow completed migration/testing, not this
review document. No migration PR has yet been opened by this work.

## 13. Decisions still to make, one chunk at a time

1. Adoption recovery is implemented; independent allocation construction remains a separate API decision.
2. Resolved encoding construction, authority and unknown-color assumptions.
3. Complete plan/worker and CMS ownership/signatures, proven with real backends.
4. Output-plan integration and which ownership forms have actual adopters.
5. Preservation vocabulary and strict in-place refusal/consuming failure ownership.
6. Deferred video representation with AOM/SVT/VMAF adapters; later CVVDP conversion contract.
7. Whether optional row iteration, fallible typed shortcuts and channel-order
   naming cleanup earn their added surface.

Already chosen: source-compatibility model, `DiffuseWhite::new` behavior,
metadata-preserving `into_contiguous`, directly destructurable parts, and preserving
large allocations. Do not reopen those decisions through a copied historical draft.

## Detailed evidence and supplementary examples

- [API/contract proposal](api-contract-proposal-0.2-and-0.3.1.md): detailed invariants and provisional signatures.
- [Migration cards and reviewed caller ledger](migration-examples-and-audit-0.3.1.md): old/new examples and file-level consumers.
- [Ripgrep inventory](audit-2026-09-27/README.md): search scope, outputs and limitations.
- [PR #63 inventory](pr63-commit-inventory.md): all 21 commits, exact sealing/rename map and port decisions.
- [0.2.16 accidental API review](release-0.2.16-accidental-api-review.md): actual published additions and warnings.
- [README annotations](readme-contract-review.md): section-by-section claims and intended replacements.
- [YUV assessment](yuv-carrier-assessment.md): verified storage contracts and adapter order.
- [Release checklist](release-checklist-0.2-and-0.3.1.md): short execution/status list.

This consolidated review takes precedence over conflicting historical suggestions
in the 0.4 assessment, old release/removal queues and intermediate PR #63 revisions.
