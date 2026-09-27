# Performance review: proposed contracts without hidden pixel passes

2026-09-27. Owner steering: implement wanted correctness work, **except** automatic
preservation of intermediate quantization; review anything that can impose checks,
passes, scratch, or unexpected conversion cost. This document supersedes earlier
unconditional “composition retains quantization” recommendations.

No new timing claims are made here. These are costs derived from implementation
and proposed algorithms. Read the code cases alongside this table.

## Chosen policy and implementation costs

1. **Composition (owner selected):** optimize for the requested final encoding by
   default, with explicit materialization/stage-preservation when the intermediate
   integer representation is itself meaningful. Ordinary composition keeps its
   current optimization behavior.
2. **Content-dependent checks (owner selected):** no automatic image prepass hidden behind ordinary
   conversion. Distinguish explicit preflight, fused validation with partial-output
   failure, and an unconditional operation requested by the caller.
3. **Preparation:** allocating scratch, parsing profiles and selecting kernels is
   explicit setup. Execution within prepared capacity must not grow buffers.
4. **Strict in-place:** either prove support from metadata or use an explicitly
   requested preflight. Do not promise unchanged-on-error for checks that discover
   failure only after overwriting pixels.
5. **Exact preservation:** reject unsupported/unproven work cheaply by default;
   callers may explicitly authorize scanning or provide trustworthy domain facts.
   Origin metadata alone is not proof after edits.

These choices do not commit to new public enum/method spellings.

## Quantization: requested output versus materialized intermediate

```rust,ignore
// Pipeline optimization: the U8 bridge is accidental; normally avoid its loss.
let plan = float_to_u8.compose(&u8_to_float);
// Preferred default: optimize for the final output; do not force an integer grid.

// Explicit materialization: caller intentionally wants an 8-bit intermediate.
// Existing API: retain the stage by executing both converters explicitly.
// Reuse one caller-owned intermediate row; no whole-image allocation required.
float_to_u8.convert_row(src, intermediate_u8_row, width);
u8_to_float.convert_row(intermediate_u8_row, dst, width);
// A future composition option must preserve the same boundary semantics.
```

The earlier wanted `assert_eq!(composed, separately_quantized)` is not universally
appropriate. Separate modes must state both the numerical and performance contract.
Avoiding quantization is often *more* faithful to the intended image processing.

That does not permit dropping unrelated CMS transforms, mattes, user effects or
explicitly requested clipping. An opaque third-party transform is not a removable
inverse merely because its endpoint descriptors match. A legacy composer unable
to retain it should return None, not silently erase it. Also keep luminance-anchor
semantics through composition.

## Content checks: four different execution contracts

| Operation | Additional cost | Failure/output contract |
|---|---|---|
| Descriptor/geometry validation | O(1) per view/plan (or O(planes)); no pixel reads | Reject before execution |
| Explicit opacity/value preflight | Up to one extra full pixel read pass | Can reject before mutation |
| Fused check during existing conversion | Extra comparisons/reductions in the kernel; no separate read pass | May discover failure after partial output; source unchanged for out-of-place work |
| Unconditional drop/quantize requested by caller | No content proof scan | Caller explicitly permits the information change |

```rust,ignore
// Proposed explicit preflight: cost is visible at this call.
let proof = scan_opacity(&source)?;
let output = convert_with_opacity_proof(source, proof)?;

// Proposed fused mode: one traversal, caller accepts possibly modified destination.
worker.convert_into_with_checks(source, destination, CheckMode::DuringConversion)?;

// Existing explicit policy concept: discard without an opacity guarantee.
let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
```

Spellings in the first two examples are not approved APIs. A proof must be tied
to the exact unchanged pixels it scanned; do not publish a reusable boolean that
survives arbitrary mutable access. A borrowed validated view can constrain mutation,
or the proof can remain private to a single explicit operation.

Fused checks are not free. They can add instructions, inhibit vectorization, or
require reduction across SIMD lanes. Use specialized kernels selected once when
requested; do not branch on validation policy for every pixel in all conversions.
Do not silently run a prepass because a fused implementation has not been written.

## Cost of every code-review item

“Pixel work” below means work beyond constant-size geometry/metadata validation.
A bug returning unchanged bytes is not a valid performance baseline for a requested
color transform.

| Wanted item | Cost/risk | Implementation or policy |
|---|---|---|
| Minimum final-row extent | O(1) span bounds; no extra pass | Already fixed with adoption |
| Zero-width/empty rows | Small row-access branch; no sample scan | Early empty return; avoid arithmetic/casts on nonexistent pixels |
| Gamut containment | Constant-size lookup/math | Fix predicate; no image gamut scan |
| CICP padding/alpha semantics | O(1) metadata selection | Preserve format default alpha |
| Named/CICP resolution agreement | O(1) signaling normalization | Preserve matrix/range meaning; don't silently treat YCbCr tags as RGB |
| Contradictory descriptors | O(1) acceptance checks | At construction/planning; not per row/pixel |
| Orientation context retention | One context ownership retain per result | No image pass; preserve already-performed orientation |
| Known-transfer adapter guard | O(1) guard; previously skipped requested conversion now executes | No added prepass; true identity continues to borrow |
| Quantization composition | Can add lossy intermediate passes and scratch | Default optimizes final output; explicit stages preserve intermediate results |
| RGBA→GrayAlpha | Actual luminance computation, no false identity | Prefer one existing/fused row kernel; refuse unsupported route at setup |
| Scalar Gamma22 | A power-function evaluation replacing wrong identity | Use accurate existing scalar math; no extra image traversal |
| F16 subnormal rounding | Correct branch/bit arithmetic in existing kernel | No new pass or scratch |
| DiscardIfOpaque | Scan or fused check required unless already proven | Owner selected explicit preflight or opt-in fused checks |
| Composite to gray | Requested matte arithmetic, possibly current multi-step scratch | Prefer fused matte+luma; don't scan opacity to decide whether to composite |
| Premultiplied nonlinear transfer | Unassociate/transfer/reassociate arithmetic where needed | Fuse per-pixel operations where possible; avoid extra full-image passes |
| Unsupported gamut/layout planning | O(1) dispatch check | Reject at setup rather than execution panic |
| CMS cross-depth support | O(1) capability check, or explicit depth conversion | Refuse unsupported pair; don't silently insert expensive fallback |
| SameAsOrigin output | May require a real full image color transform | Explicit output target requests this work; never pretend restoring tags suffices |
| Output alpha association | Per-pixel conversion when association differs | Correct identity check, specialized transform; no opacity scan merely to unassociate |
| Output signal range | Per-sample range math when range differs | Correct identity check; refuse unavailable route rather than retag |
| Actual ICC forwarding | Profile parsing/LUT setup; real transform may replace guessed matrix | Resolve/prepare once, retain profiles/tables, not per row |
| Preserve external CMS during composition | Actual requested backend work and possibly staging scratch | Retain or refuse; never erase work; do not retain unrelated integer quantization by default |
| Independent CMS workers | Setup cost per worker; immutable LUTs may share | Avoid mutex serialization; factory/preparation cost visible |
| Reinterpret sample alignment | O(1) pointer/stride check | No sample scan |
| Typed reinterpretation | O(1) layout/type check plus explicit erasure | No sample scan or runtime type taxonomy |
| Reorder metadata retention | O(1) metadata assignment around existing swap loop | No new traversal/Arc clone |
| ImgVec stride preservation | Can eliminate current compaction | Move original storage with actual stride; alignment conversion may still copy |
| Resolve color authority | O(1) selection plus possible profile validation/parsing | Resolve once per image/plan; unknown/conflict refusal without pixel scans |
| Prepared width/extent validation | O(1) per execution call | Row checks outside pixel loop; explicit capacity growth |
| Fallible CMS row method | Result branch once per row/call | No per-pixel Result, allocation or backtrace on success |
| Exact preservation | Cheap proof/rejection or explicit O(samples) scan | No automatic scan inferred from provenance or heuristic loss |
| HDR anchor validation | O(1) finite/positive checks | At parameter construction, already implemented |
| HDR peak discovery | O(samples), possibly decoding/linearizing first | Must be explicit measurement or caller-supplied peak; don't hide in generic conversion |
| Borrowed/consuming identity | O(1) metadata comparison/ownership | No pixel copies; large ICC equality may cost O(profile bytes), resolve/cache appropriately |
| Caller-output identity | Required copy into supplied destination | Document; caller selected independent destination |
| Strict in-place | May need preflight, scratch or refusal | No replacement image or hidden rollback buffer; explicit scan policy |
| Parts adoption/error retention | O(1) validation/ownership; error trace allocates only on failure | Implemented; strip image with without_buffer before retaining error |
| Contiguous export | O(image) moves only if needed | Explicit consuming operation; prefer parts for stride-aware receiver |
| Typed owned export | May copy for allocator-layout incompatibility | Preflight alignment/capacity, reuse when legal; never compact then knowingly copy |
| Streaming source errors | One Result/EOF check per produced batch | No owned row copy to express errors/lifetimes |
| CallbackSource scratch | Reuse one row instead of allocating each time | Check production before append; no invented EOF row |
| Resident row iterator | O(1) bookkeeping per row | Optional convenience; no full image buffering |
| Video plane carrier | O(planes) checks, no pixel scan | Deferred; borrow independent storage/strides |
| Bit-depth sample validation | O(samples) if actually checking values | Separate explicit scan; metadata checks alone do not validate every sample |
| YUV→CVVDP RGB | Real matrix/range/chroma/display conversion and filter halos | Explicit conversion with caller scratch, not hidden carrier coercion |
| Cosmetic names/trait sealing | No runtime benefit | Keep established imports and open traits; optional aliases only with concrete benefit |
| docs.rs organization | Build/docs only | No runtime change; keep core dependency boundary |
| Compile-time cleanup | Macro/generic expansion costs | Measure; no extra public crates, unsafe POD or type-level policy proliferation by default |

## Hidden work already present

- `adapt_for_encode_explicit_cow` calls `is_fully_opaque` before conversion when
  dropping alpha under `DiscardIfOpaque`. This is an existing prepass, not a
  hypothetical cost of the new design. Its policy/default relationship needs
  explicit documentation and replacement design.
- `PixelBufferHdrConvertExt` measures peak rowwise before mapping. It may perform
  input decoding/linearization for measurement and again for output. Name that
  convenience as measuring, or supply explicit known-peak preparation.
- Rectangular in-place orientation allocates a visited array proportional to
  pixel count. Avoid claiming allocation-free; offer explicit scratch if needed.
- Legacy free row conversion creates scratch per call; RowConverter can grow
  scratch lazily. Prepared capacity should remove those hidden allocations.
- std CMS clones share a mutex-backed executor, serializing use. Independent
  worker setup may cost more at creation but removes hot-path contention.
- ImgVec adoption previously compacted but retained the old stride. The current
  fix moves its original storage, preserving geometry and avoiding that pass.
- Profile setup and synthesized profiles can be expensive even when no image
  allocation is necessary. Zero-copy pixels is not zero setup cost.

## Performance acceptance tests for implementation

Use a fixed toolchain/features and representative tiny rows, large 40–100 MB
frames, tight/strided storage, identity and actual transforms. Measure separate
setup and execution costs; count allocations outside allocator instrumentation
setup. Test repeated execution within prepared capacity, multiple independent
workers, source/destination alias restrictions and explicit-scan modes.

Compare valid equivalent work: correct conversion versus a known-good reference,
not versus a relabeling bug. For fused checks, compare the checked and unchecked
kernel under opaque, early-failure and late-failure inputs. Verify no accidental
second read pass with code inspection and appropriate profiling, not elapsed time
alone. Include x86 and ARM before universal throughput claims.

No new scan or worker architecture is implemented merely by documenting it here.
The accompanying code-first review remains the exact-case inventory, with corrected
quantization policy and implementation statuses updated as work lands.

For U16-specific encoding, narrowing candidates, fused analysis and table setup,
see [the U16 review](u16-signaling-and-narrowing-review.md).
