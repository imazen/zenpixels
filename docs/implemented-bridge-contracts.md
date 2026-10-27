# Implemented bridge contracts

2026-09-27. These are unreleased additions/fixes on the 0.2 line. Migrated signatures are tested unchanged against the packaged 0.3.1 candidate. Runtime behavior and allocation checks live in
`zenpixels-convert/tests/prepared_contracts.rs` and `zenpixels/tests/storage_contracts.rs`.

## Prepare rows once

```rust
use zenpixels_convert::{PixelDescriptor, RowConverter};
let mut worker = RowConverter::new(PixelDescriptor::RGB16_SRGB, PixelDescriptor::RGB8_SRGB)?;
worker.prepare(4096)?; // scratch, selected tables, backend setup; no input scan
worker.try_convert_row(source_row, destination_row, width)?;
```

Old `convert_row` remains a panicking convenience. The fallible form validates
capacity, byte extents and sample alignment before writing. Backend failures may
leave partial output. Original backend errors remain in `Error::source`.
Prepared stateful workers are uniquely owned; build one converter per worker.
`try_clone` refuses uncloneable state rather than dropping its transform.

## Make loss and scans explicit

```rust
// Previously accepted without checking opacity; now fails during planning.
let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardIfOpaque);
let rejected = ConvertPlan::new_explicit(rgba, rgb, &options);

// One explicitly requested pixel scan before an unconditional drop.
zenpixels_convert::adapt::check_opaque(source.as_slice())?;
let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
let plan = ConvertPlan::new_explicit(rgba, rgb, &options)?;
```

Do not mutate the source between the scan and conversion. The existing whole-image
explicit adapter performs the preflight when that conditional policy is selected.
There is no new hidden opacity pass in ordinary conversion.

```rust
// Default: optimize requested output; intermediate integer rounding may disappear.
let optimized = a.compose(&b);
// Explicit: retain every materialized intermediate, including quantization.
let stages = a.compose_preserving(&b);
// Prove exact sample representation changes, otherwise refuse without a scan.
let exact = ConvertPlan::new_preserving_samples(rgb8_srgb, rgb16_srgb)?;
```

Composition retains each stage's descriptors and luminance anchor. External CMS
composition refuses until a complete executor can retain those transforms.

## Preserve allocations and metadata

```rust
let parts = image.into_parts(); // data + offset + stride + dimensions + descriptor + context
let image = PixelBuffer::try_from_parts(parts)
    .map_err(|e| e.without_buffer())?; // strip a potentially huge allocation before boxing/logging
let packed = image.into_contiguous(); // explicit row moves, no second image allocation
```

Use `take_parts()` on the adoption error when recovering the allocation instead.
`into_vec()` now warns: it loses the offset/stride/interpretation. Typed U8 exports
reuse compatible allocations; alignment/capacity incompatibility can require a copy.
Typed reinterpretation refuses a physical format change. Erase explicitly, change
the erased view's layout, then `try_typed` the new pixel type.

## Current color and HDR

`ColorContext` holds one authoritative **current** profile. `ColorOrigin` retains
original metadata. Finalization passes actual ICC bytes and converts pixels for
`SameAsOrigin`; restoring old tags is insufficient. Known contradictory CICP and
descriptor declarations are rejected. Descriptor-only helpers refuse ICC work
requiring a full CMS; use `finalize_for_output_with` with that backend.

Color-changing convenience conversions replace old current signaling. Layout/depth
changes preserve it. Raw PQ decoding is L/10000; its resulting linear buffer now
carries a 10,000-nit anchor. Explicit linear anchors are honored when encoding PQ.
Tone mapping normalizes PQ to the supplied source peak, preserves straight alpha
through nonlinear work, and emits the target luminance anchor. Linear input with
an attached diffuse-white anchor is scaled to the source-peak convention in the
row pipeline; without one, `HdrConfig` documents 1.0 as source peak.

```rust
// Old name concealed peak discovery; now deprecated.
let sdr = hdr.convert_to_sdr_measuring_peak(target)?; // explicit extra rowwise pass
// Or supply the peak: no measurement pass.
let sdr = hdr.convert_to_with_hdr_config(target, HdrConfig::for_source_peak(1000.0))?;
```

HLG→display tone mapping needs an OOTF and is refused; a peak alone is insufficient.
Unsupported layouts, gamut matrices and CMS sample-depth pairs fail during setup.
This does not add native 10/12-bit video-code semantics to normalized U16 images.

## Costs and remaining scope

### Output preparation added on the review branch (2026-10-25)

`OutputPlan::new` prepares the transform and matching output signaling from
the current descriptor/context and requested profile, without an image buffer.
This is the color stage needed by zencodec's paused transcode adapter: it can
inspect `metadata()` and `output_descriptor()` before allowing pixel work.
It must still validate the encoder's carrier support and metadata policy.

`OutputPlan::new_with_descriptor` accepts a required `PixelDescriptor` in place
of the format. Use it when association and range are requirements. It preserves
the requested alpha interpretation, validates descriptor/profile agreement
before invoking a CMS, and returns `OutputInterpretationMismatch` with bounded
descriptor facts on conflict. Opaque profiles require unknown color fields
when the crate cannot resolve their primaries/transfer. The format-only
constructor retains its existing straight-alpha defaults. `SameAsOrigin`
selects the profile; it does not implicitly preserve alpha association.

Same-profile premultiplied identity now borrows or copies exact sample bits
without a CMS, including strided rows and unusual F32 payloads. A different
opaque profile still requires a qualified transform; matching descriptors do
not authorize relabeling. Required narrow identity remains verbatim, while
unsupported full/narrow conversion refuses. Explicit linear premultiplication
uses existing row math and keeps alpha and padding. No new callback or format
type is added. Native validation is tracked in
[zenpixels issue 94](https://github.com/imazen/zenpixels/issues/94).

`prepare(width)` optionally moves scratch and selected table initialization
before execution. Without it, execution retains the existing lazy behavior,
including plugins without a preparation hook. `convert_rows(source, destination)`
accepts the two pixel views, checks source interpretation, geometry and target
descriptor before writes, and returns the destination view with matching color
context. Use that returned view when passing rows onward. Backend failures may
leave partial destination bytes; discard that batch.

The existing `finalize_for_output_with` now uses this same plan and retains its
owned full-image result and permissive sample options. New plan callers provide
explicit `ConvertOptions`; required alpha/depth/luma rules are checked before
an opaque-profile plugin can accept a transform.

`tests/output_plan.rs` qualifies current-versus-origin ICC authority, unchanged
padding, source-interpretation checks, sample-policy refusal and native ICC rows
against a separately built CMS worker. Its prepared authored transform allocates
no memory during row conversion. This is not an RSS or arbitrary-CMS guarantee.
The plan inherits the pixel-view constructors' whole-pixel stride restriction;
[issue 91](https://github.com/imazen/zenpixels/issues/91) tracks the byte-stride
contract and typed-bridge audit. The output-allocation work also continues
[issue 69](https://github.com/imazen/zenpixels/issues/69).

### Other bridge scope

`OutputPlan::borrow_rows` checks the same current interpretation and prepared
width as `convert_rows`. For an identity plan it returns the original strided
sample view with matching output context, without a destination allocation or
sample copy. For a plan requiring math it returns `None`; the adapter can then
admit destination storage and call `convert_rows` without weakening the request.
Changed interpretation remains an error, not a signal to try a weaker route.
The original sample owner and its allocation charge must remain alive. This
allows the native zencodec image adapter to retain the original color and
auxiliary planes together when the output request is already satisfied.

Adding opaque alpha under the same current ICC is now a checked format change,
without a CMS round trip. RGB→RGBA and gray→gray-alpha at unchanged U8/U16/F32
depth preserve every color bit, including unusual floating-point values. The
output still needs admitted destination storage; `borrow_rows` returns `None`.
This does not bypass changed-profile work, authorize depth reduction, or add
opaque-profile compositing. Those retain their existing policies and backend
paths. A real native JXL RGB F32 frame exposed the previous `NeedsCms` refusal.

The output-plan review now includes the native-sample review revision. This
puts `OutputPlan` and `CmykF32` / `CmykF32Inverted` in one dependency graph
without another conversion API. `tests/output_native.rs` checks exact identity
carriage for both polarities, including signed zero, subnormal and NaN payload
bits, strided rows, untouched destination padding and unchanged ICC signaling.
Actual native JXL profile/admitted-destination qualification belongs to the
zencodec consumer. This does not add a CMYK-to-RGB kernel. That control exercises
`convert_rows` into a distinct destination; `borrow_rows` has separate identity,
source-binding and required-conversion controls.

### Float ICC arithmetic and range selection

`OutputPlan` uses the existing `ConvertOptions` to distinguish clipping from
extended float conversion. For a native float layer that must retain negative
and above-one color values, request
`ConvertOptions::forbid_lossy().with_clip_out_of_gamut(false)`. The preset alone
still enables clipping; its name must not be interpreted as a blanket promise
of lossless color conversion.

`MoxCms` selects float arithmetic and available analytic matrix/profile curves.
The clipped route applies bounds to color before and after that transform and
retains alpha bits. Its out-of-range input staging is bounded independently of
row width; ordinary in-range input is passed directly. No speed or whole-operation
memory claim follows from this implementation. Explicit ICC requests use their
ICC curves even if embedded CICP disagrees.

The unequal-channel control found a backend case that shares the wrong curve
when only two adjacent RGB curves agree. Those profiles now fail before any
pixel write. The `whereat` error retains `MoxCmsError` as its concrete source;
callers do not have to coordinate capability flags first. Arbitrary LUT-profile
extended range, all per-channel curves, F32 CMYK conversion and CMS workspace
admission remain unqualified. The six native JXL profile comparisons live in
zencodec's spatial harness; they do not establish general ICC conformance.

Unassociation, matte and multi-stage conversion may require additional row kernels
and scratch; they are not yet all fused. Exactness checks and descriptor validation
are metadata-only. U16 content analysis combines opacity, chroma and replication
checks in one traversal. Narrowing's production kernel is unchanged pending ARM
measurements. Experimental native video carriers and a unified zencodec output plan
remain separate integration designs; no speculative public wrapper family was added.

## Exact output planning

`OutputPlan::new_preserving_samples` requires matching current/target profiles
and a metadata-provable reversible representation change. It never consults a
CMS. Identity can borrow the original storage; channel reorder/expansion and
U8-to-U16 widening use the existing exact conversion proof. Float payloads,
including negative zero and NaN bits, survive identity and opaque-alpha addition.
Different opaque ICC bytes, color arithmetic, reassociation, depth reduction and
channel removal fail before rows are accepted. Output signaling and strided
execution use the same source-binding checks as ordinary OutputPlan.

Use this when exact samples are required. `ConvertOptions::forbid_lossy()` is
not an exactness guarantee: it permits color arithmetic and enables clipping.
An unclipped color conversion has a different contract from reversible sample
carriage. Its preset replacement and numerical-model guarantees remain open.
The native JXL caller is zencodec's `jxl_planned_color_tests`; library controls
are `zenpixels-convert/tests/output_exact.rs`.

## Conversion preset replacement

`ConvertOptions::forbid_lossy()` is deprecated. `retain_channels()` is its
behavior-preserving replacement: forbid alpha removal, depth reduction and
RGB-to-gray; permit gray expansion and requested color arithmetic; retain
historical enabled clipping. This deliberately avoids promising losslessness.
The old method remains callable with identical values and a migration warning.

| Caller requirement | API |
|---|---|
| Reversible representation change, matching output color | `OutputPlan::new_preserving_samples` |
| Reversible low-level layout/depth change | `ConvertPlan::new_preserving_samples` |
| Color conversion retaining channels/depth | `ConvertOptions::retain_channels()` |
| Extended-range color conversion retaining channels/depth | `retain_channels().with_clip_out_of_gamut(false)`; requires a qualified executing route |

The workspace callers are migrated without changing expectations or samples.
`policy::tests::old_preset_preserves_behavior_and_does_not_promise_exact_samples`
checks the compatibility alias and every option field. Actual color conversion
and exact-output tests exercise the replacement. Backend numerical guarantees,
non-sRGB extended conversions and operation-level policy selection remain
separate requirements; renaming the preset does not implement them.
