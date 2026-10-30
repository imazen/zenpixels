# Migration examples, caller audit, docs and streaming

> **2026-09-27 update:** checked parts adoption with `take_parts()` and
> `without_buffer()` is implemented. The owner chose to deprecate the entire
> existing planar module now and defer a better video representation. Earlier
> proposals below to expand that module are superseded. See the
> [code-first review](code-review-0.2-and-0.3.md) for executable cases and wanted
> behavior before approving the remaining guards/contracts.

2026-09-27, zenpixels `main@17c78d9`. Companion to the
[contract proposal](api-contract-proposal-0.2-and-0.3.1.md).
For the verified release/stacked-PR history and immediate deprecation shortlist,
see the [published 0.2.16 review](release-0.2.16-accidental-api-review.md).

The compatibility target is **one migrated source tree, building with either
the complete 0.2 bridge or 0.3.1, one version per connected pixel pipeline**.
The user has selected this over cross-version type interchange.

New API snippets below are **design sketches unless explicitly marked implemented**.
The ownership-handoff card now uses implemented `into_contiguous` / `into_parts`;
checked adoption remains proposed. Existing snippets are abridged from source or defect reproductions;
variables supplied by the surrounding application are omitted. Names such as
`PixelEncoding`, `PreparedConverter` and `OutputPlan` remain provisional.
Freeze their signatures in the bridge, then retain those signatures in 0.3.1.

## What should stop working?

There are three different changes. Each PR must identify which it makes:

| Kind | 0.2 bridge behavior | 0.3.1 behavior |
|---|---|---|
| Superseded API | Old expression compiles with an actionable deprecation; replacement is available | Remove old expression after migration; replacement unchanged |
| Invalid runtime input | Existing fallible boundary rejects it; no invalid buffer/plan escapes | Same validation and error contract |
| Incorrect implementation | Same valid source produces correct pixels/metadata, or refuses unsupported work | Same corrected behavior |

A deprecation cannot distinguish `DiffuseWhite::new(user_value)` with a valid
value from the same call with NaN. By owner decision, `new` now validates and
panics for invalid values, retaining its spelling and const usability. This is
a documented behavior correction, not a source migration or deprecation.

Do not privately change public descriptor fields, existing trait requirements,
enum exhaustiveness, auto-traits or Cargo features under this promise. The
tested `Descriptor { ..existing }` field-deprecation loophole is documented in
the main proposal. Absence of known callers does not close that loophole.

## Migration cards

### 1. Backing allocation versus packed pixels

Actual caller: `squintly/src/variant_gen.rs:289–314` decodes AVIF, then extracts
the vector and manually removes stride padding. `into_vec` exposes backing
storage, including a possible private alignment offset.

```rust,ignore
// OLD: assumes that the first pixel is at raw[0].
let stride = buffer.stride();
let raw = buffer.into_vec();
for y in 0..height {
    packed.extend_from_slice(&raw[y * stride..y * stride + row_bytes]);
}

// NEW, implemented: packed rows plus their description, with no second allocation.
let PixelBufferParts {
    data, offset, stride_bytes, width, height, descriptor, color_context, ..
} = buffer.into_contiguous().into_parts();
let packed_pixels = &data[offset..];
// Move data and its description to the receiver, retaining offset.
```

Proposed later deprecation of `into_vec`: “Returns backing allocation without
its layout; use into_parts for ownership transfer, with into_contiguous first
when packed rows are needed.” No new into_vec warning is added in this chunk. The owner has explicitly requested parts for 40–100 MB
allocations. It is a primary bridge API, not conditional on a pool integration:

```rust,ignore
// O(1): moves ownership; no pixel copies, allocation or Arc increment.
let parts = buffer.into_parts();
// PROPOSED adoption, not implemented yet:
let buffer = PixelBuffer::try_from_parts(parts)?;

// Or transfer individual fields to a stride-aware consumer.
let PixelBufferParts {
    data, offset, stride_bytes, width, height, descriptor, color_context, ..
} = buffer.into_parts();
// First visible row begins at data[offset], not necessarily data[0].
// Pass data and these interpretation fields together to the consumer.
```

The implemented record exposes its fields directly and is non-exhaustive from
introduction. It is unvalidated transfer data; changing a field does not corrupt
a live PixelBuffer. Proposed checked reconstruction will validate geometry/alignment
and return the original parts with the error on rejection, preserving ownership
of the large allocation. Unchanged parts must preserve pointer, capacity and
context on round-trip. The full shape is in the contract proposal.

Compaction removes row padding; it preserves the alignment offset and performs
no depth/color conversion. The implementation compacts in the original allocation,
with no second image allocation. Compaction still moves bytes, whereas parts
extraction does no pixel work. A consumer accepting stride should prefer parts;
one only borrowing pixels should keep using a PixelSlice.
Squintly must explicitly require/convert to its intended U8 RGB(A) format;
counting channels alone is not a general byte-layout check. Metadata travels
with the Vec in the parts record. No source migration is needed for
`EncodeOutput::into_vec`.

### 2. Checked geometry and typed reinterpretation

This current reproduction adopts geometry a slice accepts but a later owned
buffer view cannot handle:

```rust,ignore
let mut b = PixelBuffer::new(4, 1, PixelDescriptor::RGB8);
b.transform_in_place(|p|
    PixelSliceMut::new(p.bytes, 1, 2, 9, PixelDescriptor::RGB8).unwrap()
);
let view = b.as_slice(); // currently panics after adoption succeeded
```

The new contract must never commit geometry incompatible with subsequent buffer
operations. Either make owned operations support minimal final-row extent, or
reject at adoption before committing it. Existing `transform_in_place` already
documents programmer-error panics; fixing that validation does not turn arbitrary
mutating closures into transactional, fallible operations. A closure may already
have changed bytes. No blanket “unchanged on error” promise here.

Typed reinterpretation is a separate API problem:

```rust,ignore
// OLD: Self retains P even when the descriptor changes physical interpretation.
let mislabeled_typed = typed.reinterpret(other_layout)?;

// NEW: candidate route, explicit erasure and checked physical relabeling.
let erased = typed.erase();
let relabeled = erased.reinterpret_erased(other_layout)?;
let typed_again = relabeled.try_typed::<DesiredPixel>()?;
```

The new method spellings need to be reconciled with existing typing helpers.
Deprecate the typed-preserving operation; do not change its return type silently.
Relabeling never performs a channel swap. A caller wanting a swap uses conversion.
No external primary-checkout use of this operation was found; do not add several
new public helpers without a concrete adopter.

### 3. Raw descriptors stay convenient; acceptance establishes invariants

```rust,ignore
// OLD: representable, but internally contradictory.
let d = PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Premultiplied));
let pixels = PixelSlice::new(&bytes, width, height, stride, d)?;

// NEW: the same construction returns Err for that descriptor.
// When the samples really contain RGBA premultiplied pixels:
let d = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
let pixels = PixelSlice::new(&rgba, width, height, rgba_stride, d)?;
```

No descriptor rename, field privacy change or warning on legitimate literals.
Validate format/alpha, sample alignment, byte stride, checked dimensions and
extent at storage and planning boundaries. Constructors that currently panic on
invalid input can preserve that API alongside their existing fallible variants.
Do not turn them into `Result` between lines without a replacement migration.

### 4. Conversion must not silently retag known color

Fresh reproduction: linear U8 RGB `[128,128,128]` passed to
`adapt_for_encode_cow` requesting sRGB comes back tagged sRGB with unchanged
bytes. Actual linear→sRGB conversion produces approximately `[188,188,188]`.

```rust,ignore
// OLD: accepted as metadata-only because storage formats match.
let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
let out = adapt_for_encode_cow(&[128; 3], src, 1, 1, 3,
                               &[PixelDescriptor::RGB8_SRGB])?;

// NEW: a plan resolves the actual current source encoding first.
let plan = /* plan from resolved source to selected encoder encoding */;
let mut worker = plan.prepare(1)?;
worker.try_convert_row(&[128; 3], &mut out, 1)?; // ~188, not a retag
```

Fix cow adapters too; migration to `_cow` alone does not fix this bug.
`try_adapt_in_place` likewise currently retags when physical formats match.
It must perform the requested supported transform or return an error.
Unknown encoding requires an explicit assumption; known encoding requires
conversion. Keep genuine declaration/attachment operations distinguishable in
documentation; do not pretend assigning a descriptor transforms samples.

### 5. Resolve one current encoding, preserve source history separately

```rust,ignore
// OLD: different helpers can select different parts of conflicting context.
let ctx = ColorContext::from_icc_and_cicp(icc, cicp);
let looks_srgb = ctx.is_srgb();
let profile = ctx.as_profile_source();

// NEW: choose authority at the decode boundary; preserve other tags as origin.
let ctx = ColorContext::from_icc(icc); // when ICC is authoritative
// Or ColorContext::from_cicp(cicp), when CICP is authoritative.
let encoding = /* resolve descriptor + authoritative context + luminance */;
let plan = /* prepare conversion from encoding */;
```

The ambiguous constructor is already deprecated. `zencodec::SourceColor` already
selects authority for its normal arms. Its `info.rs:374` suppressed legacy
fallback is unreachable with both tags and today's authority variants, but
still needs source cleanup before removal. It is not evidence of a currently
executed conflicting-tag path.

Prefer borrowing `&ColorContext` for inspection. If replacing the public Arc
accessor, introduce a new accessor and deprecate the old one; do not change its
type silently. Ordinary buffer reborrows should borrow the context without
atomic increments. Independently owned views may still need shared ownership.

### 6. Prepare once, execute fallibly, use pixel views at image boundaries

Actual pattern in zenmetrics' CVVDP, GMSD, API and GPU helpers:

```rust,ignore
// OLD: free convert_row performs setup/scratch work during repeated execution.
let plan = ConvertPlan::new(source.descriptor(), target)?;
for y in 0..source.rows() {
    convert_row(&plan, source.row(y), output.row_mut(y), width);
}

// NEW: plan includes resolved color, worker has capacity for this image width.
let plan = /* complete source/target plan */;
let mut worker = plan.prepare(width)?;
worker.convert_slice_into(source, output)?;

// Streaming equivalent; row format is fixed by the prepared plan.
let mut worker = plan.prepare(width)?;
for y in 0..height {
    worker.try_convert_row(source.row(y), output.row_mut(y), width)?;
}
```

`convert_rows(raw, src_stride, dst, dst_stride, w, h)` migrations in dvifmish,
zensim and zenmetrics should construct checked views once and use the slice
method. Do not resolve profiles or allocate scratch for every row. Width beyond
prepared capacity requires explicit re-preparation or error; no hidden growth.

Deprecate `RowConverter` as a whole if replacing its incomplete plan/Clone
contract, not only its constructors. This reaches downstream signatures such
as `zenpipe::PixelOp::as_row_converter`. Worker creation is fallible:

```rust,ignore
// OLD
let another = converter.clone();
// NEW: independent mutable scratch, immutable backend data may be shared.
let another = plan.prepare(max_width)?;
```

Keep one-shot conveniences only where they have real users and documented setup
cost. Relocating the allocating free function to a method does not solve this.

### 7. Composition preserves every requested operation

```rust,ignore
// OLD: currently reports identity and erases the intermediate U8 quantization.
let f = PixelDescriptor::RGBF32_LINEAR;
let u = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
let a = RowConverter::new(f, u)?;
let b = RowConverter::new(u, f)?;
let fused = a.compose(&b).unwrap();
assert!(fused.is_identity());

// NEW: composition retains the U8 stage and its rounding/clipping.
let sequential = f_to_u8_plan.then(&u8_to_f_plan)?;
let mut worker = sequential.prepare(width)?;
```

Input `0.1234567f32` currently survives the composed route unchanged, while
separate conversion quantizes it. Legacy `compose` must conservatively return
None when it cannot represent faithful composition. `then` can retain stages
or reject; matching endpoints do not prove identity. This matters directly to
`zenpipe/src/sources/transform.rs:67`, which removes “identity” stages.

### 8. CMS accepts complete requests and reports failures

```rust,ignore
// OLD: accepted request may panic at execution; transform trait returns ().
let mut c = RowConverter::new_explicit_with_cms(
    PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
    PixelDescriptor::RGBF32_LINEAR,
    &ConvertOptions::permissive(), Some(&MoxCms),
)?;
c.convert_row(&[128; 3], bytemuck::cast_slice_mut(&mut [0f32; 3]), 1);

// NEW: unsupported cross-depth path fails during planning/preparation,
// or execution is supported and returns a Result rather than panicking.
let plan = /* full encoding request accepted by CMS factory */;
let mut worker = plan.prepare(1)?;
worker.try_convert_row(src_row, dst_row, 1)?;
```

Old implementor shape `fn transform_row(&mut self, ..., width: u32)` becomes
a method on a **new** executor trait returning `Result<(), ...>`. Do not change
the required method in place. New factories receive alpha, range, luminance,
intent and actual profiles; signature details await a MoxCms ownership prototype.
An accepted transform failure propagates, rather than falling through to a
different backend. Output may be partially written; no ready output escapes.

Migrate zenpipe's `icc_transform.rs`, `imageflow_compat/cms.rs` and `job.rs`
together. Its `PixelOp::apply` also returns `()` today; the pipeline needs an
error-propagating interface, with its own source-compatibility treatment.

### 9. Pixels and output metadata must describe the same result

```rust,ignore
// OLD: SameAsOrigin can emit origin tags while conversion uses descriptors.
let ready = finalize_for_output_with(
    &working, &origin, OutputProfile::SameAsOrigin, format, cms,
)?;

// NEW: proposed plan includes conversion to selected original encoding.
let output_plan = /* current encoding + origin + selected output + CMS */;
let ready = output_plan.into_output(working)?;
// Pass ready pixels AND its matching metadata to the encoder.
```

Borrow-or-own `adapt`, consuming `into_output`, and caller-output `write_into`
are distinct possible contracts, not three mandatory public methods. The scan
found no primary external finalizer calls. Start with zencodec's existing
`ColorEmitPlan` / `resolve_color_emit` and transcode integration; demonstrate the
needed ownership forms there before promoting a new `OutputPlan` publicly.
The old finalizers still need correctness fixes even if no caller is found.

For allocation control, replace “try some adaptation, then silently allocate”
with an explicitly chosen operation:

```rust,ignore
// NEW candidate operations, added only with actual consumers.
buffer.convert_in_place(&plan)?;       // preflight; no replacement image
let buffer = buffer.into_converted(&plan)?; // may allocate; identity moves
worker.convert_slice_into(src, dst)?;  // caller supplies image storage
```

Strict in-place support requires that preflight eliminate all later failures;
data-dependent alpha checks need a scan. Fallible arbitrary CMS work requires
separate output. Retain existing allocating `convert_to` for its many users.

### 10. A preservation policy must state what it preserves

```rust,ignore
// OLD: the name promises too much; this preset enables gamut clipping.
let opts = ConvertOptions::forbid_lossy();

// NEW: candidate builder on the new conversion request, not another bool
// inserted into today's publicly constructible ConvertOptions struct.
let plan = request.require_sample_preservation().plan()?;
```

Deprecate the misleading preset only after its replacement has a precise
contract. A reversible channel reorder/widening may qualify. Arbitrary transfer,
gamut, rounding or clipping does not qualify merely because a loss estimator
returns zero. Alpha removal can require an explicit scan for opacity.
Source provenance “originally U8” does not prove samples remain on the U8 grid
after filtering. No primary external use of `forbid_lossy` was found in this scan.
Avoid another cosmetic “lossless” rename without defining these cases.

### 11. HDR parameters are checked and have distinct roles

```rust,ignore
// OLD: invalid values are accepted.
let white = DiffuseWhite::new(user_nits);
// NOW: same spelling; invalid values panic. Constants need no migration.
let white = DiffuseWhite::new(user_nits);

// OLD, still used in hdr-corpus-convert/src/codec.rs:149 under allow(deprecated)
let cll = ContentLightLevel::measure(pixels, DiffuseWhite::BT2408);
// NEW: use the existing explicitly named maximum algorithm.
use zenpixels_convert::hdr::CllMeasure;
let cll = ContentLightLevel::measure_max(pixels, DiffuseWhite::BT2408, method);
```

The last argument follows the existing `LightLevelMethod` choice; migrate the
caller to its intended algorithm, not an arbitrary percentile. Retain checked,
panicking `new` and named constants; no new constructor is introduced. Current relative-linear
anchor, display peak and measured content maximum remain different parameters.
Missing required HDR policy must cause refusal even without `hdr-experimental`;
feature absence must not enable clipping. Retain the feature name on both lines.

### 12. Planar invariants: repair first, expand only with an adopter

**Update:** concrete raw-plane callers are now verified in SVT, AOM and the new
VMAF checkout. Prioritize a borrowed YUV carrier with these adopters; see the
[YUV assessment](yuv-carrier-assessment.md). The absence of references to the
existing owned types is not evidence of absent YUV demand.

```rust,ignore
// OLD: replace an owned plane without revalidating the image layout.
*image.buffer_mut(index).unwrap() = unrelated_buffer;

// NEW candidates if a real planar adopter needs them:
image.plane_mut(index).unwrap().row_mut(y).copy_from_slice(row);
let old_plane = image.try_replace_plane(index, replacement)?;
```

Validate plane count, roles, sample format, subsampling factors, ceil-divided
dimensions and reference extent. Checked construction alone cannot protect
subsequent replaceable owned buffers. Deprecate escape hatches only when usable
alternatives ship. No primary external `MultiPlaneImage` or `PlaneDescriptor`
references were found; broad new planar mutation APIs and field privacy changes
are therefore deferred. Current zenjpeg plane-like structures are not evidence
that it consumes these zenpixels types.

### 13. Cleanup should reduce concepts, not create import churn

| Old | Recommended destination | Warning/removal treatment |
|---|---|---|
| `ByteOrder`, `byte_order` for RGB/BGR layout | `ChannelOrder`, `channel_order` | Optional additive aliases + deprecated old names; no endianness change |
| Transfer-blind ICC helper/constant | Profile selected by complete encoding, or unsupported error | Deprecate only after accurate replacement exists; do not equate matching primaries with matching profile |
| `requires_cms(from, to)` heuristic | Successful complete planning/preparation, or explicit unsupported result | Do not use a descriptor-only Boolean as a capability guarantee; no external primary caller found |
| `Adapted`, non-cow adapters | Corrected existing cow helpers | Verify color/alpha/range parity before removal; several callers already migrated |
| Implicit/default estimation exposure | Explicit `estimation-experimental` opt-in, same signatures | Conditional bridge deprecation; require opt-in in 0.3.1 |
| Public pipeline helpers | Separate review of the older published surface | Deprecate published items before removal; retain Cargo feature names |
| Free orientation helpers | Same functions with metadata-preserving implementation | No source rename; methods would add redundant vocabulary |
| `serde` no-op feature | Same recognized feature with honest docs | No removal needed |
| `fast-transpose` | Same explicit opt-in | No unrelated default switch |
| Existing `Pixel`, extension traits, errors | Same implementation/match/result contract | No sealing, supertrait changes or cosmetic error consolidation |

ByteOrder matches in zencodec EXIF and zentiff are unrelated byte-order enums;
they should not be changed. New extension methods on open traits need provided
implementations or a separately introduced interface. Hide compatibility modules
from the main docs tree only where canonical items remain easy to find.

## Correctness changes with no necessary source migration

Seventeen assertions in two temporary external fixture crates were rerun against
the current tree. All passed **because they assert the existing defects**.
These are evidence for fixes, not passing acceptance tests for this proposal.

| Existing bad behavior reproduced | Required corrected behavior |
|---|---|
| Accepted in-place geometry later panics in `as_slice` | Consistent accepted geometry or rejection before adoption |
| `crop_view(0,0,0,2).row(1)` panics | Empty valid row for zero-width view |
| Display P3 reports it contains Adobe RGB despite out-of-gamut green | Correct containment predicate |
| CICP mapping treats RGBX padding as alpha | Respect format's alpha semantics |
| Named BT.2020 PQ resolves while equivalent CICP does not | Consistent supported/unsupported profile resolution |
| RGB/RGBX descriptors can claim contradictory alpha | Reject contradiction at checked boundaries |
| Allocating orientation loses context; in-place preserves it | Both preserve current color/provenance |
| Known linear U8 retagged sRGB without conversion | Convert ~128→188 or refuse |
| F32→U8→F32 composition becomes identity | Preserve quantization/clipping |
| RGBA `[200,30,10,99]`→GrayAlpha produces `[200,30]` as identity | Compute grayscale and retain alpha 99 |
| Gamma22 scalar linearization is identity | Implement transfer correctly or explicitly refuse |
| f16 minimum subnormal × .375 rounds up | Correct rounding to zero |
| DiscardIfOpaque accepts transparent pixels | Reject without losing alpha |
| Transparent black composited over white becomes gray 0 | Respect background, yield white |
| Premultiplied encoded transfer applies operations in wrong order | Unassociate/transform/reassociate according to declared semantics |
| Adobe RGB→Oklab accepted then panics | Valid supported conversion or planning refusal |
| MoxCms P3 U8→linear F32 accepted then panics | Supported cross-depth conversion or planning refusal |

Fixtures: `/tmp/core-foundation-review/src/lib.rs` (7),
`/tmp/convert-foundation-review/src/lib.rs` (10). Both were run with
`cargo test --offline --quiet --manifest-path <fixture>/Cargo.toml`.
Promote each relevant case to a regression test asserting **correct** behavior
in the implementing PR. This focused set does not replace the larger foundation
review or prove all kernels, targets and feature configurations correct.

## Reviewed consumer ledger

All paths in this table are relative to `/home/lilith/work`. “Keep” means no
planned signature migration, not that all of that consumer's color behavior has
been certified. The [inventory snapshot](audit-2026-09-27/README.md) and
[reproducer](../scripts/audit-pixel-migration.py) preserve the search scope.

| Consumer / concrete sites | Required migration or validation |
|---|---|
| `squintly/src/variant_gen.rs:297` | Packed PixelBuffer export; enforce intended sample format |
| `dvifmish/src/decode.rs:97–101` | RowConverter/convert_rows → prepared checked views |
| `zen/zensim/zensim-picker-prep/src/bin/cross_codec_butter_features.rs:274–288`, `picker_sweep.rs:302–316` | Same prepared slice migration |
| `zen/zensim/zensim-validate/src/bin/check_holdout_overlap/native_linear.rs:59` | Prepared conversion; preserve declared linear/color policy |
| `zen/zenmetrics/crates/zenmetrics-cli/src/decode.rs:661–675` | Prepared image conversion |
| `zen/zenmetrics/crates/cvvdp/src/pipeline.rs:2460–2475`, `gmsd/src/lib.rs:323`, `zenmetrics-api/src/metric.rs:2710`, `zenmetrics-gpu-core/src/lib.rs:212,421` | Prepare outside row loops; resolve context rather than descriptor-only identity |
| `zen/zenmetrics/crates/zenmetrics-cli/src/hdr.rs:87,182,190` | Checked white constructor; preserve measurement policy; new worker |
| `zen/zenanalyze/src/row_stream.rs:141,156,214,247,292,391` | Replace stored workers; propagate conversion errors through fetch/borrow interfaces; retain domain-specific analysis policy |
| `zen/zenanalyze/src/linear_tier.rs:222`, `versioning.rs:302` | Validate diffuse white, including persisted/configured values |
| `zen/zenpipe/src/lib.rs:118,130`, `src/ops.rs:33,48,58,75,85,126` | Public re-exports, stored worker, constructor and PixelOp boundary migration |
| `zen/zenpipe/src/sources/transform.rs:67` | Preserve stage semantics during composition; do not delete quantization |
| `zen/zenpipe/src/sources/icc_transform.rs:25,52,72,100`, `imageflow_compat/cms.rs:79`, `job.rs:1429` | Fallible complete CMS interfaces and pipeline error propagation |
| `zen/zenpipe/zeneditor/src/decode.rs:73`, `pipeline.rs:271,295` | New prepared conversion |
| `zen/zenpipe/zenfilters/src/convenience.rs:16,254,484`, `srgb_filters.rs:28,253,303,305` | New prepared conversion; preserve operation order |
| `zen/zenpipe/zencodecs/src/codecs/avif_enc.rs:159`, `jxl_enc.rs:130`, `dispatch.rs:216`, `encode.rs:877`, `transcode.rs:705` | Already use cow adapters; audit semantic identity and emitted color, no mechanical rename |
| `zen/zenpipe/src/sources/callback.rs:90–114` | Reuse row scratch; append only a produced row; distinguish EOF from error |
| `zen/zencodec/src/info.rs:374` | Remove suppressed ambiguous constructor fallback; preserve explicit authority |
| `zen/zencodec/src/output.rs:205,277,282,349,430` | Keep PixelBuffer/PixelSlice boundaries; test context retention across frames |
| `zen/zencodec/src/color.rs:247`, `set.rs:545` | Concrete integration point for coupled output conversion/metadata, before public OutputPlan |
| `zen/zencodec/src/traits/decoder.rs:54`, `dyn_decoding.rs:165` | Existing fallible lending batches; keep/adapt rather than duplicate |
| `zen/zencodec/src/traits/encoder.rs:131`, `dyn_encoding.rs:75`, `sink.rs:81` | Add source-error-preserving pull encoding on both static/dynamic surfaces; preserve frame color in sink path |
| `zen/hdr-anchor-vis/src/main.rs:114,125` | Checked white, raw convert_buffer → views/prepared conversion |
| `hdr-corpus-convert/src/codec.rs:145–150` | Remove allow(deprecated); existing CllMeasure::measure_max; verify anchor metadata |
| `zen/imazen-26/tools/corpus-thumbs/src/main.rs:133`, `tools/resave-colour-check/src/main.rs:94` | Prepared row worker |
| `coefficient/src/analysis/feature_extract.rs:804,810`, `src/auto_encode/dispatch.rs:63,115`, `encoders.rs:22` | Keep actual zenpixels slice boundaries; test source color reaching analysis/encode |
| `zen/zenextras/zenexr/src/lib.rs:65,73`, `zenpdf/src/render.rs:124`, `zentiff/src/decode.rs:80` | Keep owned/typed carrier boundaries; check high-depth/alpha/context |
| `zen/zenraw/src/decode.rs:272`, `zen/zenjxl/src/decode.rs:195` | Keep public decoded-pixel carriers; verify metadata accompanies pixels |
| `zen/zenresize/src/lib.rs:60`, `src/pixel.rs:268`, `src/composite.rs:112,132` | Keep re-exports/descriptors; check typed layout, alpha and current color preservation |
| `zen/ultrahdr/ultrahdr-core/src/color/mod.rs:52,62,67`, `tonemap.rs:395,401,445` | Keep color/HDR re-exports and buffers; validate anchor and conversion parity |
| `zen/zentone/src/gainmap.rs:779–807` | Keep PixelSlice→PixelBuffer boundary; verify explicit output encoding/context |
| `zen/zensquoosh/crates/zensquoosh-codecs/src/lib.rs:111–115`, `zen/zensr/crates/zensr-micro/src/px.rs:49–52`, `zen/zensysbench/src/codecs/zc.rs:47` | Keep existing carrier interfaces and allocating convenience where appropriate |
| `webpx`, `zen/heic`, `zen/mozjpeg-rs`, `zen/zenavif`, `zen/zenbitmaps`, `zen/zengif`, `zen/zenjpeg`, `zen/zenjpegai`, `zen/zenpng`, `zen/zenwebp` | Carrier/import/construction matches retained in inventory; paired builds and color/stride/alpha boundary tests, not blanket renames |

Corrections to earlier evidence: zenpipe's `job.rs:1168` and
`imageflow_compat/execute.rs:398,579,1302` export encoded bytes, not pixels.
Imageflow's `imageflow_types::PixelBuffer` and graphics `ColorContext` are its
own types. Orientation `.compose()` calls and ordinary `.clone()` calls are
not converter composition. No primary external matches were found for
zenpixels `MultiPlaneImage`, `PlaneDescriptor`, `reinterpret`,
`transform_in_place`, `requires_cms`, `forbid_lossy` or `REC2020_V4` after checking
the relevant candidate receivers. These are bounded search findings, not a
proof that no published or generated caller exists.

The sweep included `/home/lilith/work`, ignored files, test/example/dev code,
alternate checkouts and dependency aliases. It discovered 721 matching manifests,
1,176 dependency declarations and 3,422 directly mentioning Rust files. The
primary-focused candidate set is 746 files. Three alternate manifests did not
parse; those errors are recorded. This is a broad discovery pass plus a reviewed
ledger, **not** a claim that regex has semantically classified every API use.

## A docs.rs tree designed around user tasks

The current docs build succeeded for both crates with public advanced features.
The generated landing pages contain 10 module entries for core and 24 for convert
in that feature configuration. Core exposes implementation modules and root
re-exports; convert additionally re-exports core modules and presents many
implementation/analysis modules.
Use canonical documented entry points without forcing import changes:

```text
zenpixels
  PixelBuffer / PixelSlice / PixelSliceMut / PixelCow
  Pixel / PixelDescriptor / PixelFormat / BufferError
  Orientation                         (established root convenience)
  color                               current encoding, context, origin
  hdr                                 anchors and metadata, no conversion engine
  planar                              feature-gated existing storage vocabulary
  icc                                 feature-gated inspection
  policy                              existing data-only conversion policies

zenpixels_convert
  ConvertPlan / PreparedConverter / ConvertError
  primary buffer extension traits     supported one-shot ownership operations
  cms                                 final factory/executor contracts and backend
  output                              coupled pixels/metadata, once adopted
  hdr                                 measurement, quantization, explicit tone mapping
  orient                              established free orientation helpers
  icc_profiles                        accurate profile selection
  advanced documented operations      retain useful gamut/analysis contracts
```

This is a landing-page/navigation sketch, not a list of permitted public types.
Retain established imports. Make implementation modules such as `buffer`,
`descriptor`, `converter`, `ext`, and `adapt` doc-hidden only after canonical
re-exported items, legacy migration links and intra-doc links remain usable.
Use selective `#[doc(inline)]` on canonical re-exports; avoid duplicating the
entire core manual through convert's glob re-export. Keep the glob available
for source compatibility even if hidden from convert's landing page.
Rustdoc's [re-export rules](https://doc.rust-lang.org/rustdoc/write-documentation/re-exports.html)
support this separation of import paths and documentation placement.

Top-level examples should answer: borrow a strided image; allocate checked
storage; convert once; prepare for repeated rows; preserve color through encode.
Each method states byte-stride units, color authority, allocation/setup cost,
failure/mutation behavior and metadata retention. Do not add a prelude or a new
module for every helper. Deprecated items remain reachable from the migration
guide with working replacement links; hiding them is not deprecation.

Configure an explicit public feature set for docs.rs, not indiscriminate
`all-features` exposing benchmark internals. Core currently has no docs.rs
metadata feature selection; convert already has one. Check feature badges,
default/minimal documentation, intra-doc links and doctests on both release
lines. [docs.rs metadata](https://docs.rs/about/metadata) supports explicit feature
selection. Build inspected here:

```sh
CARGO_TARGET_DIR=/tmp/zenpixels-doc-audit-2026-09-27 \
  cargo doc --offline --locked -p zenpixels -p zenpixels-convert --no-deps \
  --features 'zenpixels/imgref,zenpixels/planar,zenpixels-convert/hdr-experimental,zenpixels-convert/cms-moxcms,zenpixels-convert/fast-transpose'
```

## Streaming and row iteration

**Yes to better row access and error propagation; defer a third general provider
trait in core.** There are already two real lending abstractions:

```rust,ignore
// zencodec::StreamingDecode, simplified existing signature
fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error>;
// zenpipe::Source, simplified existing signature; Strip is a PixelSlice
fn next(&mut self) -> PipeResult<Option<Strip<'_>>>;
```

Both can return a view borrowing reusable internal scratch. An ordinary
`Iterator<Item = PixelSlice<'a>>` cannot express each item's lifetime as a fresh
borrow of that call's `&mut self`; its associated Item is fixed. A lending method
fits. This follows the standard [Iterator signature](https://doc.rust-lang.org/std/iter/trait.Iterator.html).
Do not force full-image allocation, Arc refcounts or owned row copies merely to
fit Iterator. Demonstrate that common adapters simplify both existing traits
before proposing a core provider with another metadata/error vocabulary.

For an already-resident slice, a normal iterator **can** lend immutable rows from
the underlying buffer. A candidate `row_iter()` avoids name collisions:
`PixelSlice::rows()` already returns a count and PixelBuffer's `rows(y,count)`
is a subview operation.

```rust,ignore
// OLD, already reasonable and allocation-free
for y in 0..view.rows() {
    consume(view.row(y));
}
// NEW additive ergonomic candidate; adopt in a real loop before publishing
for row in view.row_iter() {
    consume(row);
}
```

Define rows as visible `width * bytes_per_pixel`, excluding padding. Handle a
cropped final row lacking trailing padding and zero-width views correctly;
`chunks_exact(stride)` can drop the last row and `chunks(0)` panics. Metadata
stays on the parent view and must accompany any cross-crate pixel boundary.
If needed, `row_iter_mut()` must safely split disjoint storage; no per-row
allocation or new public row struct just for naming an iterator. This is
ergonomics and centralized stride correctness, not a promised speedup over `row`.

The more urgent interface change is zencodec pull encoding:

```rust,ignore
// OLD: returning usize forces an error to be hidden, captured elsewhere, or panic.
encoder.encode_from(&mut |y, dst| -> usize {
    match provider.fill(y, dst) {
        Ok(n) => n,
        Err(_) => 0, // becomes EOF; loses the cause
    }
})?;

// NEW candidate: errors propagate, with a final name retained on both lines.
encoder.encode_from_fallible(&mut |y, dst| {
    provider.fill(y, dst) // Result<usize, SourceError>
})?;
```

The old snippet illustrates the interface hazard, not a claim that a particular
encoder currently swallows errors this way. Add the fallible method to static
and dynamic interfaces with compatible defaults/adapters, without changing an
existing required signature. Retain the infallible entry point if useful for
memory-backed sources. Define source error transport and cause preservation
with zencodec's existing error model, not a new codec dependency in core.

Specify sequential requests versus random access, capacity, returned row count,
positive progress, premature EOF for known height, cancellation as an error,
and stable terminal EOF. Keep current color/frame metadata consistent across
batches; dimensions and descriptor alone do not cover ICC/HDR meaning.
`DecodeRowSink::begin` currently receives geometry/descriptor, so inspect how
the frame header supplies color when building adapters.

There is also a concrete source-review finding in zenpipe CallbackSource: it
allocates a Vec for each row and pushes that row **before** checking the callback's
false/EOF result. Its own `from_data` returns false when no row was produced.
Reuse scratch and append only after successful production; report short known
images rather than manufacture a zero row. This finding was read in source,
not separately executed as a reproduction during this audit.

## PR gates and order

1. Fix accepted-invalid states and wrong pixel/metadata behavior on 0.2, with
   regression tests and explicit behavioral release notes.
2. Implement the smallest complete plan/worker/color contract; migrate zenpipe,
   zencodec, zenanalyze and representative metric/codec consumers in companion
   PRs. Promote new public helpers only when those callers adopt them.
3. Add actionable deprecations, migration examples and the canonical rustdoc
   layout. Make bridge examples compile and assert the required semantics.
4. Build the same downstream source against bridge and 0.3.1 with deprecated
   uses denied. Inspect warnings in consumer builds (dependency lints can be
   capped); remove stale `allow(deprecated)` from migration paths. Test external
   trait implementations, generic bounds, exhaustive matches and features too.
5. Use separate lockfiles/isolated builds to force each version across the
   connected core/convert/zencodec/codec graph. A widened range alone does not
   guarantee Cargo selects one shared version. Check `cargo tree -d` and exercise
   actual decoder→filter→encoder boundaries; do not bridge via raw bytes just to
   make mismatched types compile.
6. Publish the complete bridge, then the 0.3.1 removal release; explicitly opt
   controlled consumers into the tested range. If the complete bridge is 0.2.17,
   the candidate range is `>=0.2.17, <0.4`, not `^0.2.2` and not a promise to
   accept arbitrary later breaking minors. Package the two foundation crates
   so each tested release line resolves to the intended companion core line.

Release gate performance: borrowed identity without image copies; prepare once;
no allocations or locks inside prepared execution; independently prepared CMS
workers; source/color resolution outside row loops; minimal/default clean build
timings on a fixed toolchain. These are measurements to run on implementation,
not performance claims established by this documentation-only change.
