# Video frames: simple reading, early errors, expert access

Design proposal for [PR #81](https://github.com/imazen/zenpixels/pull/81), updated
2026-09-29. **API sketches, not implemented APIs.** The concrete AV1/MP4 work and
test inventory are in the [implementation reference](frame-interpretation-details.md).

## The default experience

**Choose the output, build the reader, read frames.** The reader applies format
rules, checks declarations early, prepares conversions, and handles compatible
changes. Callers should not assemble color claims, mapping guards, and conversion
plans just to get a correctly described frame.

```rust,ignore
// Proposed API shape. Existing reader/builder naming should be reused.
let mut video = Video::open(source)?
    .output(Output::srgb8())
    .build()?;

while let Some(frame) = video.next_frame()? {
    // Packed sRGB pixels with their timestamp and matching color description.
    consume(frame)?;
}
```

The proposed `srgb8()` preset requests full-range encoded sRGB RGB8, rounds to
nearest with ties upward, and clamps finite out-of-range RGB during packing,
without dithering. Alpha-bearing input requires an RGBA output or an explicit
discard/composite choice; no opacity pre-scan. It does not silently tone-map HDR.
If HDR rendering requires a policy the caller has not supplied, return an
actionable error. Presets fix these choices in documentation; backend defaults
cannot change them. Expert conversion can instead request clipping refusal.

For native processing, change the output request:

```rust,ignore
let mut video = Video::open(source)?
    .output(Output::native())
    .build()?;

while let Some(frame) = video.next_frame()? {
    encoder.push_frame(frame)?; // Destination checks compatibility.
}
```

Native output retains plane storage, sample encoding and color meaning. It does
not imply an RGB conversion or a full-frame copy. Unknown transfer information
can remain representable when native decoding does not need it. A destination
that cannot preserve or interpret it refuses the frame.

The sketch does not dictate a new generic reader or output enum: implement this
experience through the existing media session/reader first, promoting only the
public pieces required by its real callers.

## Catch errors at the earliest useful point

Checking some invariants is valuable even when it cannot prove everything.
The rule is **check what we know now, and check new information when it arrives**.

| When | What we check | What success means |
|---|---|---|
| Construct a sample/layout declaration | Depth fits storage, shift fits, dimensions/strides are valid; known impossible combinations with available context | This declaration is internally consistent |
| Build the reader | Applicable headers, format constraints, conflicting declarations, requirements of the requested output that are already decidable | The known stream configuration can start this operation |
| Receive changed headers/frame metadata | Recheck affected constraints and update the selected interpretation and conversion | This frame can be produced under the requested contract |
| Execute conversion | Backend failures and sample-dependent conditions such as clipping refusal | Successful output meets the requested contract |

`build()` can read bounded headers and prepare known transforms. It must not
decode/scan the whole video to promise that later frames will succeed. On a
non-seekable input, retain any necessary read-ahead packets under the existing
buffer limits. `next_frame()` remains fallible because later data can differ.

For example, conflicting MP4 and AV1 range flags fail when both applicable
claims are available. They do not wait until somebody asks for RGB. Conversely,
an unfamiliar transfer code is not a contradiction: native output can preserve
it, while an sRGB request fails as soon as that requirement becomes decidable.

Validate the selected track and requested operation. Unknown optional metadata
or an unselected audio track must not accidentally make video reading fail.
Do not conflate a format-forbidden combination with a valid one unsupported by
the chosen backend. Both get useful errors, with different explanations.

These checks share code. Constructors, adapters and planners do not maintain
separate validation tables. Tiny context-free checks stay with the existing
types; format rules stay with the format adapter; capability checks stay with
the backend. No universal `ValidCicp` promise is required to check real invariants.

## Two levels, one implementation

| Normal reader | Expert access |
|---|---|
| Choose output and read frames | Inspect declarations, map native storage, prepare conversions explicitly |
| Apply format rules automatically | Inspect which rule/source selected each field |
| Reject known contradictions early | Retain conflicting evidence for auditing and explicit repair |
| Cache/rebuild compatible plans internally | Reuse prepared plans, caller buffers and row sinks directly |

Inspection is an explicit entry point using the same parser and diagnostics.
It can retain malformed metadata for analysis when doing so is safe, but it
does not disable bounds checks or decoder requirements. A repaired interpretation
must pass normal checks before producing interpreted output; the original claims
and the repair remain available in the audit report. No blanket `allow_invalid`
switch and no silent fallback to a matrix our backend happens to support.

```rust,ignore
// Expert conversion of an already obtained frame; proposed shape.
let mapped = frame.map()?;
let source = mapped.native_view();
let plan = converter.prepare(source.interpretation(), output_request)?;
plan.write_rows(source, &mut row_sink)?;
```

The high-level reader calls this same conversion machinery. It adds orchestration,
not different color math or an extra intermediate image. Expert access exists
for native pipelines, auditing, repairs and memory control; ordinary callers
should not need it for correct playback or frame extraction.

## Keep the public concepts small

The primary docs.rs path should introduce a reader, an output request and a
frame. A frame owns its samples and the description valid for that presentation
occurrence, with convenient timing/format access. That description cannot change
when the decoder advances or seeks.

Put advanced inspection and conversion in their own modules. The earlier
`FrameEvidence`, `FrameInterpretation`, `ColorInterpretation` decomposition is
an internal design sketch, not three more things every caller must construct.
Expose borrowed details only for demonstrated callers; keep storage/backend
implementations private. Reuse `Cicp`, `SampleEncoding` and `ColorContext`.

The raw `Cicp` carrier remains lossless and permissive. Checked sample/view
construction and format validation catch problems with sufficient context.
Deprecate ambiguous `Cicp::to_descriptor` only after both release lines have a
checked, usable migration for already-RGB declarations and actual conversions.
The [implementation reference](frame-interpretation-details.md#3-code-that-should-stop-looking-safe)
shows the old code and required behavior. Do not ship #55's generic matrix hint.

## Make the errors explain the next step

An ordinary error should say, for example:

```text
Cannot produce sRGB frame at 00:00:12.400:
the selected transfer characteristic (code 200) is unsupported by this converter.
Native output can preserve the samples and signaling.
```

Or:

```text
Cannot start video track 1:
MP4 declares full range, but AV1 declares limited range.
Inspect the source claims before applying an explicit repair.
```

Errors retain structured field/code/source information for tests and applications;
they do not require parsing messages. Include frame/track identity only when
known. Diagnostics are bounded and shared per applicable configuration, without
formatting a report for every frame. Never suggest retagging as a color conversion.

## Test the contract, including the simple path

| Test | Expected evidence |
|---|---|
| Known contradictory initial headers | Reader build fails before pixel conversion or output allocation |
| Same conflict appears later | Initial valid frames succeed; affected frame fails before conversion writes |
| Unknown transfer, native versus sRGB request | Native preserves it; sRGB fails at the earliest point with sufficient information |
| High-level versus explicit low-level conversion | Same samples, resulting metadata, clipping and error categories for the same request |
| Native ownership across decode/seek | Existing frame pointers and descriptions remain valid; no implicit pixel copy |
| Repeated unchanged configuration | No repeated profile parsing, full-frame temporary, or unbounded diagnostic accumulation |
| Unsafe depth/shift/stride declaration | Construction fails without examining pixel values |
| Value-dependent clipping/backend failure | Explicit partial-write contract for caller buffers; owned-frame API returns no successful frame |
| Inspection of conflicting claims | Both survive; repair is explicit and revalidated by the same rules |

Use a shared fixture set for parser/selection tests, operation tests and reader
integration tests. Reference outputs must include actual AV1-in-MP4 files, not
only hand-built structs. Neither the simple path nor expert path may bypass a
required check. All these tests remain pending implementation.

## Costs and release placement

Cheap declaration checks are metadata work, generally O(planes). Validation does
not infer color from pixels, measure peak brightness, or scan for opacity.
Requested packed output owns an allocation; native output retains backend
ownership where possible; row sinks use bounded scratch. These costs belong in
method documentation. Reusing plans must not skip compatibility checks.

The two API levels add no crates or feature flags by themselves. `zenpixels`
retains small vocabulary; media owns reader/format policy; conversion owns math.
Container inspection must remain usable without AV1/CMS dependencies. Compare
minimal and enabled compile graphs as each implementation slice lands.

Implement on the existing zencodec media stack and coordinate its metadata PR.
Use #76/#77 for the common core migration, then 0.3 removal. This PR changes
documentation only. [The detailed inventory and pending acceptance matrix](frame-interpretation-details.md#6-inventory-and-implementation-sequence)
retain the ownership, timing, color and MP4 edge cases without making them the
introduction to using the library.
