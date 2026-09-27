# API and contract proposal: 0.2 bridge → 0.3.1

For one reading document covering this design plus cleanup, consumers, YUV,
streaming and docs, see the [consolidated review](zenpixels-0.2-and-0.3-review.md).

Draft for signature and scope review, 2026-09-27, against `main@17c78d9`.
This is a design and PR breakdown, not a release promise already satisfied by
current code. Implemented increments are marked explicitly; other new names and
signatures below remain proposed.
Existing source references and consumers were checked locally. Seventeen focused
defect reproductions were rerun; they still demonstrate incorrect behavior, not
implemented fixes. See [migration examples and audit](migration-examples-and-audit-0.3.1.md)
for old/new code, verified callers, the docs.rs tree and streaming recommendations.
The [published 0.2.16 review](release-0.2.16-accidental-api-review.md) verifies
which additions escaped PR #63. That small initial deprecation batch is now
implemented on main; the larger contracts below remain proposals. The
[root README annotations](readme-contract-review.md) explain their documentation
impact, and the [YUV assessment](yuv-carrier-assessment.md) adds concrete AV1/metric callers.

This proposal supersedes conflicting recommendations in
`release-plan-0.2.17-and-0.3.1.md` and the earlier 0.4 assessment. The owner has
chosen **the same migrated source against either release line, with one
zenpixels version per connected pixel pipeline**. Direct exchange between
simultaneously loaded 0.2 and 0.3 types is not a requirement. A shared-types
package and version-facade architecture are therefore not proposed.

## 1. The release contract

The latest designated 0.2 bridge release contains the supported construction,
conversion and extension interfaces that 0.3.1 will retain. A consumer using
that common surface, the same supported features and the supported toolchain
builds unchanged against either line. It must not suppress migration warnings.

The bridge candidate is 0.2.17, but its number is not a deadline: if replacement
work spans patches, the consumer's minimum is the patch that actually contains
the complete common surface. Do not publish an incomplete migration promise.

The contract covers normal API use, downstream trait implementations and generic
bounds, not just the repository's happy-path examples. Deprecation warnings are
the migration guide; paired consumer builds and semantic tests establish the
guarantee. Corrections of documented bugs may change output or return an error
where old code silently produced incorrect pixels. List those changes explicitly.

All 23 versions returned by crates.io's zenpixels reverse-dependency endpoint
currently require `^0.2.x`; the five returned convert dependents do too. In
particular, published convert 0.2.16 requires core `^0.2.16`. Local wide ranges
are not evidence that those ranges were published. Widen controlled consumers
only after migration and validation. These checks do not inventory every
historical release of every dependent.

### What must change in the old release plan

- Do not add `try_transform_row` as the recommended bridge method and then
  deprecate it in 0.3.1. The migration destination must remain the destination.
- Do not change an existing trait's required method signature merely because
  the known implementation count is zero. It does not generate an actionable
  warning for every external implementation or generic use.
- Do not seal existing public traits, remove feature names, add exhaustiveness
  restrictions or flip defaults under a general “warning-free means safe” claim.
  Use a new interface, retain compatibility, or explicitly narrow the promise.
- Do not permit untested future core minors with a “current plus next” range.
- Do not move core contract repairs to a later breaking release merely to ship
  a mechanical cleanup now. This is the migration window for the contracts below.
- Do not add `ConvertPlan::convert_row` just to relocate an allocating free
  function. Prepared execution should be the prominent hot-loop API.

## 2. Storage: preserve ownership and enforce geometry

Keep `PixelBuffer<P>`, `PixelSlice<'a, P>`, `PixelSliceMut<'a, P>` and
`PixelCow<'a, P>`. Their ownership distinction and typed/erased relationship are
useful. `P` identifies physical pixel layout, not complete color interpretation.
Keep `Pixel` open for custom pixel implementations.

The storage contract should be identical on both lines:

- Stride is in **bytes**; width and height are in pixels. Multibyte samples are
  native-endian. Channel order and sample endianness are distinct concepts.
- For positive width/height, the minimum visible backing extent is
  `(height - 1) * stride_bytes + width * bytes_per_pixel`, with checked arithmetic.
  A view need not contain padding after its final row.
- Zero-area views are empty. Valid row requests on a zero-width view return an
  empty slice without computing out-of-bounds offsets; an absent row remains
  out of bounds. Empty views must not fail merely because an unused pointer
  lacks sample alignment. Typed empty accessors must return appropriately
  aligned empty typed slices rather than cast a misaligned empty byte slice.
  Check alignment when nonempty samples are accessed.
- Full-row-padding requirements of an operation are additional requirements,
  not something it may silently infer from a valid minimal-extent view.
- Validate alignment of row starts, supported format, stride and typed layout.
  No unchecked multiplication in allocation/adoption paths.
- Layout mutation cannot leave `PixelSlice<Rgba<_>>` describing BGRA storage.
  Relabeling does not swap bytes.

The owner explicitly requires efficient ownership handoff for 40–100 MB
allocations, including consumers that can accept stride. Make parts extraction
the primary ownership escape hatch, with packed export as an explicit convenience.
Neither operation should duplicate an image allocation merely to extract it.

Ownership extraction and compaction below are now implemented. Checked adoption
remains proposed; the redundant contiguous-view convenience constructor is deferred:

```rust,ignore
impl<P> PixelBuffer<P> {
    pub fn into_parts(self) -> PixelBufferParts;
    pub fn into_contiguous(self) -> Self;
}

// An ownership transfer record, not a validated pixel buffer.
#[non_exhaustive]
pub struct PixelBufferParts {
    pub data: Vec<u8>,
    pub offset: usize,
    pub stride_bytes: usize,
    pub width: u32,
    pub height: u32,
    pub descriptor: PixelDescriptor,
    pub color_context: Option<Arc<ColorContext>>,
}

impl PixelBuffer {
    pub fn try_from_parts(parts: PixelBufferParts)
        -> Result<Self, FromPartsError>;
}
```

`PixelBufferParts` is deliberately a destructurable record with public fields.
It carries the allocation and everything needed to interpret its visible pixels;
callers can move out individual fields or pass the entire record onward. There
is no separate layout wrapper just to recover values. Public mutability is
appropriate here because this record makes no claim to maintain buffer invariants.
`#[non_exhaustive]` allows future metadata fields; downstream destructuring uses
`..`, and construction starts from exported parts rather than external literals.

Extraction moves the Vec and existing color-context ownership. It performs no
allocation, pixel copy, row compaction or Arc clone. Preserve the Vec's pointer,
length and capacity, as well as the offset and stride. The first pixel begins
at `data[offset]` for a nonempty image, not necessarily `data[0]`; each subsequent
row starts `stride_bytes` later. Padding is not part of the visible row.
An already-borrowing consumer should receive `as_slice()` instead of decomposing
ownership. A stride-aware ownership consumer receives the parts record.

Adoption validates the supplied vector's actual alignment, offset, extent,
stride, dimensions and descriptor before constructing a buffer. Unmodified
parts round-trip without reallocating or copying pixels. A replaced/reallocated
vector may have different alignment and must be checked again. Do not silently
compact or repair rejected parts. `FromPartsError` is a proposed named error
retaining both `At<BufferError>` and the original parts, with access to the cause
and an ownership-recovery operation. No boxing/copying of image storage is needed.

Parts extraction intentionally erases `P`; retyping an adopted buffer is an
explicit checked operation. Typed Vec adoption/recovery must obey allocator
layout rules: reinterpretability of bytes does not make every Vec allocation
legal to deallocate as a different element type.

**Implemented packed-export decision:** `PixelBuffer::into_contiguous(self) -> Self`
preserves descriptor, color context and the pixel type, followed by `into_parts()`
for raw ownership. No byte-only export is added. For Vec<u8>-backed storage,
compaction reuses the allocation with no scratch allocation or context clone.
It may move O(image bytes); already packed rows need no pixel moves. Capacity
is retained, while length is truncated to the existing offset plus packed size.

**Alignment correction to the earlier sketch:** preserve the existing pixel
offset. Vec<u8>'s base is not guaranteed to meet multibyte sample alignment;
forcing offset zero can invalidate typed access or require reallocation. Packed
pixels begin at `parts.data[parts.offset]`, with no padding between rows. The
borrowing `as_contiguous_bytes()` accessor exposes exactly those pixels.

Deprecate `into_vec` in the bridge with both alternatives explained. Removal
in 0.3.1 is reasonable once the consumer probes cover the migration; retaining
a deprecated alias is also cheap if an actual compatibility reason emerges.

Add an erased-returning reinterpretation operation for genuine layout changes;
deprecate the existing typed-preserving `reinterpret`. A migrated caller uses
explicit erasure, checked relabeling, and checked retyping. Do not silently
change the old method's return type between lines.

**Failure ownership decision:** rejected parts are returned through the adoption
error, so a failed validation does not discard a 40–100 MB allocation. This
supersedes the earlier suggestion to drop failed inputs. It is the contract of
this ownership-transfer API; consuming conversion's failure ownership remains
a separate decision.

Concrete packed-export consumer: `squintly/src/variant_gen.rs:297` extracts a
zenavif `PixelBuffer` and manually removes row padding. The previously cited
zenpipe `job.rs` and `imageflow_compat/execute.rs` calls export **encoded output**,
not a PixelBuffer; they do not justify this API. Parts extraction/adoption is
now explicitly requested for large stride-aware ownership transfers and is in
scope for the bridge. Do not invent an already-migrated sibling caller as its
justification. Typed/high-depth interchange in zenavif and zenjpeg merits
validation but is not itself evidence of a parts consumer.
Acceptance tests must verify pointer/capacity preservation, offset/stride/color
round-trip, allocation-free extraction and recovery of rejected allocations.
The previous orphaned PR #63 contains reference implementations. The
[complete commit inventory](pr63-commit-inventory.md) distinguishes missing work,
manually copied code, superseded APIs and rejected changes. Reconcile selected
implementations with current invariant fixes rather than rebasing mechanically.

## 3. Descriptors: description versus validated execution

Recommended default: retain `PixelDescriptor` as the small Copy value describing
format, transfer, primaries, alpha and range. Its public fields describe a
request or declaration; the value alone is not a proof of a valid encoding.
Validate it when constructing/adopting storage and preparing execution. Once
accepted, a buffer's private descriptor and geometry must remain consistent.
Mutating an external descriptor copy cannot corrupt that buffer.

This buys enforceable invariants without replacing a ubiquitous public type
just to privatize fields. Add checked constructors for dynamic input where a
current caller needs them, retain convenient constants and existing constructor
signatures, and reject contradictions such as premultiplied alpha on RGBX or a
layout with no alpha channel. Meaningful unknown color values remain representable.

If a fully private descriptor is preferred instead, it needs a new type on the
bridge line or a separately specified migration contract. Deprecating fields
alone is insufficient: an external fixture with both fields deprecated still
compiled this expression under `rustc -D deprecated`:

```rust,ignore
let replacement = Descriptor { ..existing };
```

Making those fields private would reject that source. The experiment is at
`/tmp/zenpixels-deprecation-probe-z_3ax28e/`. This is a concrete hole in an
unqualified “no warnings implies no breaks” guarantee, not a hypothetical
downstream audit concern. Rust's [deprecation mechanism](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute)
and [Cargo compatibility rules](https://doc.rust-lang.org/cargo/reference/semver.html)
are the relevant language/package contracts.

The same caution applies to changing exhaustive enum matches and existing
public trait bounds. A smaller public surface is valuable, but a new wrapper
or a broken migration guarantee also has a cost.

## 4. Current color: resolve it once and carry it cheaply

Keep these roles distinct:

| Role | Meaning | Lifetime |
|---|---|---|
| Descriptor | Storage format and declared sample semantics | Current pixels |
| Current encoding | One selected interpretation, including ICC when needed, alpha/range and luminance anchor | Current pixels/plan |
| `ColorOrigin` | Original signaling and provenance | Source history |
| Output metadata | What the encoder must emit for the resulting pixels | Completed output |

Current defects are concrete: `ColorContext::as_profile_source` prefers ICC,
but `is_srgb` and transfer helpers inspect CICP. The finalizer accepts an ICC
target yet builds its converter from descriptors without those ICC bytes.

Propose one lightweight borrowed resolved input, provisionally
`PixelEncoding<'a>`. It combines a validated descriptor, a selected
`ColorProfileSource<'a>`, and the relevant luminance anchor. It has private
fields. Ordinary named/descriptor-only encodings need no heap allocation;
ICC bytes are borrowed at inspection time and retained once by an owning plan.

Its initial consumers are the converter, finalizer, zencodec color negotiation
and CMS request boundary. This is the justification for a public type; do not
create a separate encoding type for every layer. The exact constructor/error
spelling is an API-review decision, not settled by this draft.

Rules:

- An ICC profile is authoritative as a profile, not automatically equivalent
  to the CICP tag embedded inside it or to guessed primaries/transfer values.
- Conflicting source tags require an explicit authority choice during decode
  or resolution. Preserve unselected tags in origin metadata if needed.
- Legacy contexts containing contradictory declarations must not be interpreted
  differently by fast paths and CMS paths. Resolve explicitly or return an error.
- Unknown is not an instruction to assume sRGB. Metadata attachment after an
  actual transform and an explicit assumption are legitimate operations; expose
  them with names/documentation that do not imply pixel conversion.
- Conversion updates the current encoding; it does not overwrite source history.
- Zero-copy identity requires equivalent current interpretation, including
  alpha, range, ICC and luminance anchor, not merely equal bytes per pixel.
- Core can preserve/select a profile without executing it. CMS parsing and
  backend feasibility remain in conversion; do not move a color engine to core.

The existing context getter already borrows without cloning; do not add another
getter merely to hide the Arc. Separately, a private borrowed/owned representation could
let `PixelBuffer::as_slice()` borrow the buffer's context without an atomic
increment while retaining the existing owned-attachment capability.

Do not promise that every legacy `PixelSlice::clone` or subview is atomics-free:
an independently owned context and a view lifetime outliving its parent can
require ownership to be retained. Ordinary reborrows should tie the context
lifetime to the owner. Measure view size and branch cost before selecting the
representation. No requirement to make the existing view `Copy`.

## 5. Plans and workers: separate preparation from execution

Keep a cloneable immutable `ConvertPlan` if its private representation can be
made complete without changing existing public signatures or auto-traits.
Store every operation needed for execution, including CMS recipes and profile
ownership. A plan is not merely a descriptor path plus an invisible transform
on a different object. Built-in plans should not acquire unnecessary boxing or
dynamic dispatch because external plans exist.

Introduce a prepared mutable executor, provisionally `PreparedConverter`:

```rust,ignore
impl ConvertPlan {
    pub fn prepare(&self, max_width: u32)
        -> Result<PreparedConverter, At<ConvertError>>;

    pub fn then(&self, next: &Self)
        -> Result<Self, At<ConvertError>>;
}

impl PreparedConverter {
    pub fn try_convert_row(
        &mut self, src: &[u8], dst: &mut [u8], width: u32,
    ) -> Result<(), At<ConvertError>>;

    pub fn convert_slice_into(
        &mut self, src: PixelSlice<'_>, dst: PixelSliceMut<'_>,
    ) -> Result<(), At<ConvertError>>;
}
```

These are proposed signatures, not currently callable code. The raw row method
is explicitly bound to the source/target encodings established by its plan;
width and sample extent are checked. The slice method validates dimensions and
the complete interpretation at the outer boundary, then uses the prepared row
path. Do not repeat profile resolution or image-level validation for every row.

Preparation resolves kernels, reserves scratch for the maximum width and
initializes the lazy state required by execution. Calls within that capacity
must not allocate inside the implementation; CMS implementations must satisfy
the same contract when participating in this interface. A larger width requires
explicit re-preparation or returns an error. No invisible growth in a method
advertised as allocation-free.

The worker is not Clone. Another worker comes from `plan.prepare(...)`, which
can fail and has independent mutable state. Immutable LUTs and immutable CMS
tables can be shared; mutable scratch must not be serialized through a mutex.
Keep CPU dispatch outside per-pixel loops; benchmark whether storing a selected
kernel improves on the current step dispatch before committing that detail.

### Composition

`then` preserves sequential semantics. It cannot remove quantization, clipping,
noninvertible alpha operations or CMS work. For example, F32→U8→F32 must retain
the U8 grid; it is not identity even though endpoints match. Approximate fusion
requires an explicit numerical contract, not a heuristic “zero loss” label.

Incompatible plans return an error. A backend that cannot support faithful
composition is retained as explicit stages or rejected. In the old API,
`compose -> Option` should conservatively return None rather than erase work.

This affects real integration: `zenpipe/src/sources/transform.rs:67` composes
converters and deletes the stage when it reports identity.

### Legacy migration

Keep `RowConverter` usable in 0.2 while the new plan/worker route is completed.
Deprecate the whole type if removal of its Clone and incomplete-plan contract
requires a new worker type. Deprecating only constructors does not catch callers
receiving it through other APIs or naming it in trait bounds. `Clone` behavior
cannot be migrated simply by attaching a warning to a trait impl.

Old external-transform operations that cannot be represented faithfully must
refuse explicitly rather than silently become identity. The precise legacy
no_std Clone fallback requires an implementation decision: its infallible
signature cannot report an uncloneable backend as a Result. Do not claim that
adding `try_clone` alone fixes the old Clone contract. Prefer migrating callers
to independently prepared workers and removing the deprecated worker in 0.3.1.

If the current `ConvertPlan` cannot be repaired while preserving its own
contract, introduce and deprecate a whole replacement plan in 0.2 as well.
Do not quietly change its public associated types or lose auto-traits.

## 6. CMS: complete requests, explicit failure, independent state

Current `PluggableCms` receives profiles and pixel formats but not a complete
alpha/range/luminance contract. Existing row-transform methods return `()`.
The replacement interface needs these properties:

1. Its request includes complete resolved source/target encoding, intent and
   permitted changes. Reject unsupported lane-depth/layout combinations while
   planning, not through a kernel panic after accepting the plan.
2. Declining an unsupported request differs from accepting it and failing.
   An accepted transform's error must not silently trigger a different backend.
3. Immutable backend preparation is distinct from mutable per-worker execution.
   Rebuilding a worker may be fallible; shared immutable state is allowed.
4. Row execution returns an error. Failure leaves a caller-provided destination
   potentially partially written; it never produces an encode-ready result.

The smallest useful backend shape is a preparation/factory contract plus a
mutable executor contract; an additional public immutable factory trait is
justified only if plans need to retain third-party factories across lifetimes.
Prototype that ownership with MoxCms before freezing trait names. Prefer two or
three well-defined traits over a proliferation of shared/owned/clone variants.

Ship the final fallible trait signatures as new interfaces in 0.2 and retain
those exact names/signatures in 0.3.1. Deprecate the old traits as a group once
the built-in CMS and a sample external implementation migrate. An adapter around
an old infallible transform cannot invent errors the implementation never
reports; document that limitation and do not present it as full semantic parity.

This supersedes the old plan to add `try_transform_row`, then change
`transform_row` and deprecate the bridge method in 0.3.1.

## 7. Conversion ownership and output finalization

Use an explicit ownership vocabulary; do not make a method silently choose
between strict in-place execution and whole-image allocation.

| Operation | Contract |
|---|---|
| Caller-output conversion | Caller owns destination; no image allocation; prepared scratch may be required |
| Consuming conversion | Move original allocation on identity; reuse on supported paths; otherwise allocation is allowed and documented |
| Strict in-place conversion | Preflight support/capacity; no replacement image allocation; reject unsupported operations |
| Borrow-or-own adaptation | Borrow on true semantic identity; own converted pixels otherwise |
| Allocating conversion | Always produce independent output storage |

Suggested buffer extension spellings remain `convert_into`, `into_converted`
and `convert_in_place`, alongside existing `convert_to`, but do not add all
wrappers before their real consumers and backend/ownership semantics exist.
The prepared worker is the primary repeated-row path; one-shot helpers may
prepare internally and must state that setup cost.

For strict in-place operations, require a path that cannot fail after mutation
begins once preflight succeeds. Data-dependent requirements such as opaque alpha
need an explicit preflight scan or must be rejected. Potentially failing CMS
work uses separate output. Full rollback, arbitrary backend failure, zero extra
storage and unconditional in-place support cannot all be promised together.

### Output finalization

Prototype an output plan, provisionally `OutputPlan`, tying the conversion to the
metadata it produces. The intended initial consumer is zencodec's encode boundary.
The current primary-checkout scan found no external finalizer calls. Integrate
with zencodec's existing `resolve_color_emit` / `ColorEmitPlan` and transcode path
before adding another public plan. It must replace duplicated responsibility,
not establish a competing metadata authority. Each ownership form below requires
a concrete caller; these are candidate contracts, not a mandate to publish all three.
Construction takes current resolved encoding, desired output format/profile,
origin when requested, and conversion requirements. Final constructor/builder
spelling is still open; avoid a long family of positional overloads.

It should provide these three ownership forms (method names provisional):

```rust,ignore
impl OutputPlan {
    pub fn adapt<'a>(&self, src: PixelSlice<'a>)
        -> Result<PreparedOutput<'a>, At<ConvertError>>;

    pub fn into_output(&self, src: PixelBuffer)
        -> Result<EncodeReady, At<ConvertError>>;

    pub fn write_into(&self, src: PixelSlice<'_>, dst: PixelSliceMut<'_>)
        -> Result<OutputMetadata, At<ConvertError>>;
}
```

`PreparedOutput<'a>` privately pairs `PixelCow<'a>` and metadata. Existing
`EncodeReady` can remain the owned result. The borrowed identity form must not
allocate a full image; the consuming identity form preserves storage ownership.
The caller-output form necessarily writes/copies destination bytes even for
identity. Preparation and metadata ownership may still have costs: zero-copy
pixels is not a promise of zero allocations everywhere.

`SameAsOrigin` means convert to the selected original encoding and emit matching
metadata. It never means keep working-space pixels and restore unrelated tags.
Actual ICC bytes must reach the backend; range and alpha participate in identity
checks. Unsupported exact metadata preservation must be reported, not silently
discarded. A backend's presence does not prove it can execute every requested
transform.

Deprecate old finalizers only after replacements demonstrate ICC and HDR parity
with independent fixtures. The currently deprecated `ColorManagement` path
still covers behavior missing from the newer finalizer; removal must wait for
that gap to close. Constructor privacy for ready results protects the coupling;
deliberately splitting into parts transfers responsibility to the caller.

## 8. Preservation and HDR: make the promises precise

`ConvertOptions::forbid_lossy` forbids a few classes of operation but currently
allows clipping. It is not an exactness certificate. Deprecate the misleading
promise when its replacement is ready.

Propose an explicit sample-preservation requirement on the existing options
or the conversion request. A convenience constructor such as
`ConvertOptions::preserve_samples()` is a candidate, not yet approved API.
It must require a proof based on the operation and declared input domain, or
an explicitly requested scan. It cannot trust “originally U8” provenance after
resizing, or a perceptual-loss model that returns zero.

Distinguish:

- Equivalent interpretation and reversible storage rearrangement/widening.
- Potential preservation conditional on observed data, such as opaque alpha.
- Permitted numerical/color changes, with specified rounding/clipping behavior.
- Encoder preservation of this input, which is a codec capability question.

Do not silently claim sample identity for arbitrary floating-point gamut/transfer
transforms. Define preservation relative to represented channel values and
interpretation, not byte equality across different physical layouts. Conflicting
options under an exact requirement should return an error rather than relax it.

HDR changes needed for this release:

- **Implemented by owner decision:** keep `DiffuseWhite::new(f32)` and make it
  panic for nonfinite/nonpositive values. It remains const and is not deprecated.
  No new constructor is needed for this chunk; existing named constants remain.
- Preserve a current relative-linear anchor: what value 1.0 means in nits.
  Distinguish it from source peak, target peak, content measurements and
  mastering provenance. Their equal units do not make them interchangeable.
- Missing required parameters produce a refusal or require a named assumption.
  A default constant may be chosen explicitly; it must not masquerade as metadata
  measured or signaled by the source.
- HDR→SDR refusal is unconditional when no valid conversion policy is supplied.
  Turning off an experimental feature must not turn an error into silent clipping.
- Test absolute values against an independent reference. Comparing two paths
  through the same incorrectly scaled kernel is insufficient.
- Retain existing `CllMeasure::measure_max` and percentile names where they express
  different algorithms. Do not introduce another rename just because removal
  of the old inherent `measure` makes that name available.

Keep `hdr-experimental` as a recognized feature through this migration. Stable
anchor/carrier semantics do not require shipping every tone mapper, HLG policy
or gain-map algorithm. Unsupported cases must be honest.

## 9. Planar storage: mutate pixels without invalidating the layout

`MultiPlaneImage::new` currently uses debug assertions; both `buffer_mut` and
`buffers_mut` expose replaceable owned buffers. Validating construction alone
cannot preserve the multi-plane invariant.

No external `MultiPlaneImage` or `PlaneDescriptor` references were found in the
primary checkouts in the original audit. **Update:** SVT and AOM already expose
raw planes, and the new VMAF implementation accepts its own Yuv420Frame. These
justify a lean borrowed YUV carrier now, developed with their adapters; see the
[YUV assessment](yuv-carrier-assessment.md). CVVDP needs RGB conversion rather
than relabeling its planar RGB input. Keep owned-container invariant repairs in
scope; a universal owned redesign remains unnecessary. If retaining the
existing planar surface, the following are candidate checked construction,
pixel-view mutation and replacement operations, not approved additions:

```rust,ignore
impl MultiPlaneImage {
    pub fn try_new(layout: PlaneLayout, planes: Vec<PixelBuffer>)
        -> Result<Self, At<BufferError>>;
    pub fn plane_mut(&mut self, index: usize) -> Option<PixelSliceMut<'_>>;
    pub fn try_replace_plane(&mut self, index: usize, plane: PixelBuffer)
        -> Result<PixelBuffer, At<BufferError>>;
}
```

The error and rejection-ownership details need review with the real planar
caller, as with buffer adoption. Successful replacement returns the old plane
for reuse. Reject a bad candidate before changing the image.

Validate plane count, sample depth/format, semantic roles, nonzero subsampling,
ceil-divided plane dimensions and the representable mask limit. Store or derive
a well-defined reference extent; do not infer it ambiguously from a subsampled
first plane. Multi-plane color context belongs to the current image, not an
arbitrarily named `origin` field.

Deprecate unchecked construction and both owned-buffer mutation escape hatches
only after their alternatives exist in 0.2; otherwise retain the API and validate
at every consuming boundary. Removal in 0.3.1 requires that bridge migration. A mutable pixel
view does not let a caller replace the owning plane's dimensions/layout; metadata
changes affecting the image require checked image-level operations.

Do not privatize `PlaneDescriptor` fields merely to match the style of other
types. Validate the assembled image. If future public layout changes require
a new descriptor contract, apply the same warning-coverage rule as interleaved
descriptors. Re-audit actual planar users before promising a field-layout break;
older consumer-count reports are not fresh evidence.

## 10. Cleanup: what earns a place in this migration

| Item | 0.2 action | 0.3.1 disposition |
|---|---|---|
| `into_vec` | Add complete parts/packed exports; actionable deprecation | Remove if migrated; do not confuse backing storage with pixel bytes |
| Typed-preserving `reinterpret` | Add explicit erased/retyped route; deprecate old method | Remove unsafe-to-assume contract, preserving memory-safe checked operations |
| `RowConverter` | Complete plan/prepared-worker replacement; deprecate whole type if needed | Remove legacy clone/hidden-state contract |
| `convert_rows` / raw `convert_buffer` | Recommend views plus prepared execution or supported ownership helper | Remove after migration |
| Free `convert_row` | Point repeated work to prepared executor; retain one-shot path only if justified | No allocating method clone added just for syntax |
| `Adapted` / non-cow adapters | Verify cow variants preserve full interpretation | Remove old surface after parity |
| Legacy CMS traits/finalizers | New complete fallible interface + real backend migration | Remove after semantic parity |
| `DiffuseWhite::new` | Retain and validate; panic for invalid values (implemented) | Keep the same checked, panicking constructor |
| `ContentLightLevel::measure` | Keep existing migration to convert's `measure_max` | Remove deprecated scalar method |
| `ByteOrder` / `byte_order` | Add `ChannelOrder` / `channel_order` and deprecated forwarders | Optional small cleanup; no representation or endian change |
| Transfer-blind ICC constants/helpers | Exact profile replacement and deprecation notes | Remove only after accurate synthesis/unsupported behavior is available |
| `estimate` and ConvertPlan estimation methods | Add explicit `estimation-experimental` opt-in; conditionally deprecate implicit exposure | Require the feature, retaining the opted-in signatures unchanged |
| `requires_cms` | Deprecate public predicate; use actual planning errors | Retain only a private internal predicate |
| `Adapted::as_pixel_slice` | Explicitly deprecate method; its type's deprecation does not cover inferred method calls | Remove with deprecated Adapted; destination is PixelCow::as_slice |
| Analysis-only `pipeline` | Separate review of this already-0.2.14 surface | Demote/remove only deprecated items; keep accepted Cargo feature names |
| `serde` no-op feature | Document its existing no-op status | Keep inexpensive stub during compatibility window |
| `fast-transpose` | Explicit opt-in supported and tested | Preserve defaults for this migration unless a separate agreed change is tested |
| Existing free orientation helpers | Fix context and descriptor preservation | Keep; methods are not inherently more ergonomic |
| `ConvertError` variants and whereat | Preserve matches and result types; add precise errors where permitted | Avoid cosmetic consolidation or tracing removal |
| `Pixel` and current extension traits | Preserve existing implementation contracts | No opportunistic sealing or supertrait changes |
| Core root re-exports | Preserve useful established imports | No forced import churn without a concrete benefit |

New extension methods on existing open traits need provided implementations if
external implementors are to remain compatible. Where that is impossible or
misleading, use a new sealed extension trait with blanket implementations for
the supported carriers. Add it only for concrete consumers, not as a speculative
parallel method catalog.

## 11. Runtime and compilation goals

The requested operation still has a cost. The framework should not add image
copies, per-row allocation, unnecessary atomics, locks or per-pixel policy
decisions around it. Construction and externally supplied geometry still need
checks. Public fallible boundaries and lean internal kernels are compatible.

Preserve core's dependency boundary: no SIMD dispatch framework, CMS engine or
perceptual ranking in zenpixels. Keep private implementations largely
non-generic; typed wrappers should erase early. Do not split more public crates
or introduce a type-level color/policy combinatorial surface as an optimization.

The earlier single-run consumer measurements were 1.76 s core defaults, 1.51 s
core minimal, and 4.59 s convert defaults on this workstation. They identify
POD derives and macro duplication as compile-time candidates, not a universal
budget. Owning dependency projects lets us fix those costs upstream. Removing
the bytemuck derives with manual unsafe impls would require a deliberate change
to the existing safety policy; no such change is part of this draft.

Make borrowing, identity, prepared execution and ownership reuse measurable.
Check default-target x86 and ARM when touching conversion kernels; no isolated
architecture result should stand in for both. Keep optimization work separable
from API removal so a slow kernel does not force another public redesign.

## 12. Two release PRs with reviewable commit groups

**PR A draft title:** `Prepare the 0.2 API for checked pixel conversion and 0.3.1 migration`

**Draft description:**

> Existing buffer export, color resolution and conversion APIs can lose layout
> or metadata, allocate unexpectedly, or hide backend behavior. This change
> supplies checked ownership transfer and the common planning/execution surface
> used by both the 0.2 bridge release and 0.3.1. Legacy entry points remain
> available with actionable deprecations once their replacements are complete.
>
> Validation covers identical migrated consumer source on both release lines,
> storage/color correctness, supported features/MSRV, and allocation behavior of
> prepared execution. The PR's final description must report completed checks,
> not describe planned checks as having passed.

Reviewable groups, potentially extracted as prerequisite PRs if large:

1. Behavior regressions and current-line fixes for storage, alpha, metadata,
   CMS forwarding and composition. Keep each reproduced bug with its fix.
2. Complete buffer ownership and typed-layout migration primitives.
3. Current-encoding resolution and borrowed-context access.
4. Complete plan, prepared worker and CMS interfaces, with a real backend.
5. Output ownership/finalization, exactness and HDR parameter contracts.
6. Planar checked construction/mutation.
7. Deprecations, selected cleanup aliases, consumer fixtures and release docs.

Do not label additive public APIs complete until an actual consumer uses them.
Keep no-op aliases and compatible wrappers thin. A new error-returning method
must retain that signature in the destination release.

**PR B draft title:** `Release zenpixels 0.3.1 on the migrated conversion contracts`

**Draft description:**

> Remove the legacy APIs deprecated by the designated 0.2 bridge release, while
> retaining its supported migration surface unchanged. This makes the complete
> color/plan/ownership contracts the recommended API without requiring another
> source migration for consumers already using the bridge surface.
>
> Validation compares the identical consumer fixtures against the released
> bridge and this candidate, audits every semver delta against the deprecation
> ledger, and exercises packaged artifacts and real codec integration graphs.

PR B is prepared against PR A for review, then rebased onto its merged result;
keep the PR topology explicit so no stacked-base merge is mistaken for inclusion
in main again. Retain a 0.2 maintenance branch at the bridge release. This draft
does not authorize publishing either release before the design and checks are
complete.

## 13. Acceptance gates and explicit opt-in

The release checklist needs a machine-readable or reviewable ledger:

`old item → replacement → first bridge version → deprecation → real migrated
consumer → unchanged 0.3.1 signature → semantic regression coverage`.

An item without that chain stays, gets a complete new replacement, or becomes
an explicit exception to be resolved before publication. “No matches in my
checkout” is not a substitute for the migration contract.

Required checks:

1. External consumer fixtures compile unchanged with deprecations denied
   against the minimum bridge and 0.3.1. Include trait implementations, public
   signatures, error matches, ownership extraction and generic pixel types.
2. Run each migrated controlled library as the checked package/workspace root,
   not only as a transitive dependency whose lints Cargo can cap.
3. Compile actual codecs through zencodec into a small application, passing
   buffers across boundaries. Test both fresh resolution and upgraded lockfiles.
   Assert that each connected pixel graph resolves one core version. A wide
   requirement does not itself force unification across pre-1.0 minor lines.
4. Test packaged core/convert cross-combinations that their manifests admit.
   Do not automatically narrow 0.3 convert to core 0.3 if it is intended to use
   the common API with core 0.2 too; equally, do not claim an untested pairing.
5. Test external minimal/default/rgb/imgref/planar/ICC consumers, conversion
   with/without CMS/HDR, no_std and MSRV. Workspace self-dev-dependencies can
   otherwise enable features that hide missing-feature failures.
6. Compare pixels and current interpretation, not bytes alone. Include custom
   ICC, conflicting tags, alpha-zero/nonopaque cases, range changes, minimal
   final-row extent, zero-area geometry and independent HDR reference values.
7. Verify preparation/execution allocation counts, identity pointer retention,
   in-place refusal-before-mutation and partial-output behavior on backend error.
8. Review every semver/API snapshot delta. Newly recommended methods must not
   become deprecated merely because the destination version has shipped.

An opted-in migrated dependency may eventually declare, schematically:

```toml
zenpixels = { version = ">=0.2.17, <0.4.0", default-features = false }
```

Use the actual complete bridge version, and forward the tested features. This
range is a commitment to the common 0.2/0.3 surface; it does not admit 0.4.
The accidental yanked 0.3.0 is not a supported test target. Existing locked or
exactly constrained historical artifacts are not rewritten by the bridge.
For graph conflicts, upgrade the participating controlled consumers and resolve
them to the same supported version; do not introduce copies or casts to paper
over a split type graph. See [Cargo's version resolution rules](https://doc.rust-lang.org/cargo/reference/resolver.html#semver-compatibility).

## 14. Decisions to work through

| Decision | Recommendation | Why it matters |
|---|---|---|
| Compatibility model | **Chosen:** same source; one core version per connected pipeline | Removes the need for shared type facades and their API-retention constraints |
| Descriptor fields | Keep the descriptor as a public declaration; validate at storage/planning boundaries | Strong practical invariants without a warning-coverage hole or ubiquitous type rename |
| Color model | One borrowed resolved encoding input, with origin separate | Eliminates divergent authority choices and carries full CMS/HDR semantics |
| Executor | New prepared mutable worker; cloneable complete plan where feasible | No implicit mutex-sharing Clone, hidden allocations or dropped CMS work |
| CMS migration | New final fallible interfaces in the bridge | Identical trait signatures across lines; old implementors get actionable warnings |
| Output ownership | Borrow-or-own, consuming, and caller-output forms through one output plan | Prevents metadata drift and unnecessary full-image allocation |
| In-place guarantee | Restricted preflighted paths; refuse unsupported/fallible cases | Honest no-allocation behavior without pretending to provide rollback |
| Rejected owned input | Return rejected parts with the error, preserving the supplied allocation | Required for the requested large-allocation ownership contract |
| Cosmetic cleanup | Only accurate names and genuinely obsolete surfaces; retain cheap feature stubs | Concentrates migration on behavior and contracts |

Review the remaining descriptor/color/executor contracts in separate chunks.
The initial deprecations, anchor validation, compaction and parts extraction are
implemented; no GitHub PR has been opened for this migration. Follow the
[short release checklist](release-checklist-0.2-and-0.3.1.md) for current status
and order; remaining draft signatures are not approved merely by appearing here.
