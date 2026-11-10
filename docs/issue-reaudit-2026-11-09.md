# Merged issue re-audit: 83, 132–135 and 150

Inspected merged main `feb55611` (PR152), after refreshing the local checkout;
rebased the follow-up onto `ec0d8ab0` when the allocation fix landed during
the audit. That change does not supply whole-operation memory admission.
The accompanying changes strengthen tests and correct documentation; they add
no public signatures, change no numerical kernels and relax no established
expectations. This is repository qualification, not publication approval or an
IMIJ end-to-end qualification. GitHub issues were not edited or closed.

## Disposition

| Issue | Assessment | Remaining work |
|---|---|---|
| [83](https://github.com/imazen/zenpixels/issues/83) | Partly done; opening assessment is stale | Signed/wide per-channel extraction and an actual IMIJ adapter remain unqualified. Exact ST2086 carriage and signed/wide SampleEncoding descriptions already exist. |
| [132](https://github.com/imazen/zenpixels/issues/132) | Original design/refusal task is complete | Close the bounded issue. HLG ICC representation and consumer orchestration remain separate; unsupported synthesis deliberately refuses. |
| [133](https://github.com/imazen/zenpixels/issues/133) | Original local task is complete with these documentation corrections | Close the bounded issue after this follow-up lands. Package/dependency release gates and broader downstream qualification remain separate. |
| [134](https://github.com/imazen/zenpixels/issues/134) | Most boundaries are correct; one retained authority exception | Resolve the truncated-ICC grayscale fallback before claiming universal ICC-first behavior. |
| [135](https://github.com/imazen/zenpixels/issues/135) | Current main's export fix is qualified | Historical release refs still select old fallback bytes. Reconcile supported release candidates before closing the release-exposure portion. |
| [150](https://github.com/imazen/zenpixels/issues/150) | Alpha, predicates and canonical byte equality are fixed; capacity/evidence work is partial | Integrate current-value requirements at a real processing consumer. Approximate ICC qualification remains a distinct, lower-priority task. Whole-operation admission belongs to the executor, not the node-peak model. |

## 83: descriptions, readers and canonical pixels

`SampleEncoding::new` accepts all signed and unsigned integer storage types;
its opening issue assessment still says U8/U16 only. Actual `CodePlane` readers
and `CodeMap` arithmetic remain U8/U16 only. Generic pixel formats carry all
seven added native types exactly, while unsupported numerical conversions
refuse. A description alone is not an implemented signed/wide reader.

`hdr::St2086Codes` preserves all 24 bytes separately from `MasteringDisplay`'s
lossy semantic view. `native_codes` checks every byte position/value and an
explicit luminance code that cannot survive the float round trip. The issue's
request for an exact metadata carrier has been implemented.

RGB565 demonstrates why another native view is unnecessary. Three
`CodePlane::new` views of the same U16 plane can extract `(bits, shift)` values
`(5,11)`, `(6,5)` and `(5,0)`. `to_canonical` unpacks and widens directly.
The new strided regression checks extremes, an interior word, minimal final-row
extent and untouched padding against independent integer scaling. The old
negative `CodePlane::component(..., 3, ...)` test proves only that word
interleaving cannot be confused with bitfields. Documentation now makes that
distinction. There is still no ordinary RGB565 PixelFormat, nor is one needed
for this conversion.

Do not restore NativeCodeSlice. A signed/wide extraction extension needs an
actual adapter use case, explicit signed interpretation and independent numeric
controls; a full IMIJ round trip remains consumer work.

## 132: explicit rendering, not a silent display preset

The approved [HLG rendering contract](hdr-anchor-api-design.md) is implemented:
explicit peak, black and effective gamma, an opt-in reference preset, and a
separate relative-light white anchor. Identity/depth carriage is distinct from
rendering. Analytic neutral/colored vectors, black lift, inverse, alpha payload,
extended-value, row partition and stride controls exist and pass.

The previous direct-backend regression enabled only MoxCms. It now runs with
either CMS feature and exercises both when both are enabled. Both decline the
direct CICP HLG-to-linear crossing without rendering conditions. LCMS reaches
the unsupported HLG synthesis boundary; MoxCms has an explicit crossing guard.
This does not claim that arbitrary opaque ICC transforms are HLG renderers or
that HLG ICC synthesis has been qualified.

## 133: snapshots pass; distinguish current docs from historical proposals

The public adapter example compiles; sample-aligned partial-pixel pitches are
documented; default converter free operations remain `adapt_raw_rows_for_encode`
and `codes::to_canonical`. The ablation document inventories feature-only
benchmark shims separately. Current API snapshots pass their check.

The explicit 0.2.17 baseline `e96e4e9c` comparison was replayed using checker
0.51.0: the same 15 core and 16 converter major lint families remain, classified
in the [semver audit](public-api/semver-0.2.17-to-candidate.md). Exit 100 is the
expected breaking-release result, not a successful patch compatibility check.
Default-feature checking does not audit the new opt-in Hash impls; their
approved addition is recorded separately in the hashing contract/changelog.

This pass corrects three stale claims: SampleEncoding's supported description
types, the merged status of current-precision APIs, and SourcePrecision being an
immutable origin record. It also removes the precision proposal's claim that
returning processed eight-bit-origin data to eight bits necessarily loses no
information. Existing exactness tests demonstrate otherwise.

## 134: genuine conflicts and the remaining malformed-ICC exception

The previous origin table used P3 ICC plus P3 CICP and reused
`ColorAuthority::select` to derive its expected answer. It could not by itself
prove conflict handling. The strengthened table uses P3 ICC versus sRGB CICP
and independent expected routes, and checks emitted metadata/current context.
Another table covers unknown and recognized-but-MetadataOnly ICCs, both
authorities and all presence combinations through OutputPlan and EncodeReady.
The CMS spy confirms selected ICC routing; it is not a pixel-accuracy oracle.
Existing real-CMS controls separately qualify conversion behavior.

These pass. Present-but-unsupported authoritative CICP still refuses instead
of using companion ICC. Synthesis still describes already-decoded full-range
RGB/gray: the matrix/range regression proves it does not decode coded pixels.

One exception remains in `load_bearing::plan_chroma_collapse_signaling`:
an ICC shorter than 128 bytes permits fallback to `ctx.cicp`, generating a
replacement gray ICC. In contrast, ColorContext predicates now select that
present ICC and report unknown color. The existing
`icc_with_cicp_swaps_to_gray_class_profile_on_collapse` test explicitly requires
the fallback with an eight-byte ICC and P3 CICP; both copying and in-place
paths pass. This is a policy inconsistency on malformed declarations, not a
demonstrated wrong-pixel result for a valid ICC.

Recommend suppressing gray collapse when the selected ICC is unusable, preserving
that declaration while allowing independent alpha/depth reductions. A caller
choosing CICP should resolve current authority explicitly. Changing the existing
fallback expectation requires review; this audit preserves it. No new metadata
framework is necessary.

## 135: qualified current bytes, stale release candidates

Rebuilt the oracle using native libultrahdr `5f117294` and system LCMS 2.17.
Default and isolated no-default exports are byte-identical. Both P3 exports
return DISPLAY_P3 (1); the historical compatibility negative returns -1;
the native positive returns 1. The 4913-point XYZ maxima remain
0.0000171659 for P3/sRGB and 0.000135807 for P3/BT.709, below the unchanged
0.0005 gate. These sampled checks do not prove a global equivalence bound.
Full profile hashes remain in [the export receipt](repository-issue-completion.md#p3-current-byte-qualification-and-release-exposure--135).

The fetched release refs remain `release/0.2` at `e96e4e9c` and
`release/0.3.1` at `4d41ed8a`. Their source still selects the old fallback;
neither is qualified by main's tests. Release documents already prohibit using
those refs without reconciliation. No historical branch was overwritten and
no release was published.

## 150: separate capacity policy from exactness and execution budgets

1. Alpha-role/Undefined rejection is implemented, including empty layouts.
2. Current-color predicates now follow ICC-first profile selection. Original
   contradictory carriers can remain provenance; the grayscale exception above
   is the outstanding current-interpretation concern.
3. ExactDepth already ends on mutable access and arithmetic, and preserving
   conversions have a separate path. The codec requirement resolver in
   `precision_evidence` is test-only. SourcePrecision's capacity policy still
   does not model resize/filter/CMS output requirements. Its updates also mean
   it must not be described as immutable source origin. Integrate requirements
   at a concrete consumer before adding another foundational public type.
4. Canonical ICC replacement now checks normalized bytes after paired hashes;
   it is distinct from approximate MatrixTrcSubstitution. The retained-original
   audit records complete original coverage, but the historical table still
   does not expose a uniform complete-transform bound/domain. Keep this
   qualification work low priority, as directed; source-ICC-preserving routes
   need not wait for an exhaustive requalification.
5. The memory scope is now explicit: ChainPlan takes a maximum of reported
   route peaks, with unknown propagating. It performs no lifetime/worker/cache
   analysis. That fixes the overbroad interpretation; it does not implement
   whole-operation admission. An executor must account for overlapping owners
   and concurrency before advertising that guarantee. Do not grow a generic
   scheduler in zenpixels to satisfy a hypothetical caller.

## Reproduction and limits

Run through the host resource wrapper:

- `just repository-issues-check`
- `just repository-issues-backends-check`
- `just repository-issues-precision-check`
- `just interpretation-review-check`
- `just api-doc-check`
- `just repository-issues-semver` with checker 0.51.0 (expected exit 100)
- `just p3-export-check SOURCE FRESH_OUTPUT`

The original focused contract/doctest gate passed (15 pre-existing doctest
ignores remain); API check passed; expanded backend/RGB565 controls passed;
precision/evidence and retained grayscale controls passed. P3 oracle wrapper:
`run-heavy: done rc=0 12s | peak-RSS 0.36GiB | min-avail 13267MiB | peak-load 0.81`.
This pass does not rerun the entire cross-platform numerical suite, measure
performance, qualify every consumer, or prove every historical ICC tolerance.

After rebasing onto `ec0d8ab0`, the default repository gate, expanded backend
gate, `design-feature-check` (including all-feature library/generator tests
and strict workspace all-target Clippy) and API snapshot check passed again:
`run-heavy: done rc=0 45s | peak-RSS 1.13GiB | min-avail 13169MiB | peak-load 1.00`.
