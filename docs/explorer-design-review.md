# Explorer-driven API review — 2026-11-09

Compared the review candidate (parent `79f5df8b`) with the current plane and
sample APIs, and the outstanding issues updated today. The issue descriptions
use main `d234f3c6`; review-branch implementation is not merged-main completion.

## Resolution

The owner directed removal of NativeCodeSlice. It and its solely dependent
conversion/error/padding APIs are removed from the candidate, with no replacement
public view. The explorer now demonstrates the existing CodePlane route only;
its unaligned and signed/wide reader controls are removed. Shared encoding
metadata, exact ST2086 and native pixel identity carriage remain.

The comparison below records the evidence that led to removal; the removed
adapter column is historical, not an available API.

## Evidence that led to removal

The pre-removal explorer ran these through actual Rust/WASM APIs:

| Case | Existing `CodePlane` | Removed adapter (historical) |
|---|---|---|
| Interleaved CbCr / CrCb in U16 | Supported by `component`; roles exist in `PlaneRole` | No matching layout |
| Interleaved U16 RGB10 + alpha1 | Supported: each component reader takes its own encoding | Supported; same extracted codes |
| Nonzero unused sample bits and padded rows | Ignores unused bits on read, leaves storage unchanged | Same |
| Wide words starting at an odd byte address | Requires aligned typed storage or adapter copy | Byte-backed read supports it |
| Signed I64 / unsigned U64 | Unsupported by `AnyPlane` / `CodePlane` | Exact integer read |
| RGB565 | Three CodePlane views can select bitfields of shared U16 words | Existing to_canonical unpacks and widens directly into RGB pixels |

`CodePlane::component` is more general than `PlanarLayout`: it takes a component
count/index, whereas the latter's named roles include paired chroma and separate
RGB components, not packed RGB/RGBA. `PlaneSpec` supplies one encoding per plane;
separate `CodePlane` readers can still interpret different component payloads.

The previous justification that NativeCodeSlice was needed for interleaving or
per-component U16 encoding was wrong. The live RGB10/alpha1 witness demonstrates
that overlap, including sample padding, two rows and an incomplete final stride.
The byte-offset example compares the typed original with a separately byte-backed
copy; it does not claim an unaligned typed U16 borrow is possible.

## What the implementation made worse

1. **Split capabilities.** Signed/wide/unaligned reading was added in a new
   RGB/gray-only public view instead of extending the existing component-view
   model. Chroma and exact native reading now require different public routes.
   Sharing `Plane` geometry internally did not remove that API duplication.
2. **A misleading explanation.** The reference overstated the distinction between
   the views, and the explorer omitted plane/component reading entirely. That
   concealed overlap that a concrete consumer-shaped example exposes immediately.
3. **Stale HLG presentation.** The site called rendering unavailable and described
   the bare OETF as converter behavior. The candidate has coupled display
   rendering. It now exposes explicit peak, black, gamma and white inputs, keeps
   missing-policy refusal, and does not compare that result against a bare-curve
   JavaScript model as though they were the same operation.

No demonstrated production caller used the new view. Tests, the explorer and
conversion functions added with it were circular justification. Removing that
surface resolves the immediate duplication; it does not authorize a speculative
replacement or assert that all native component requirements are now implemented.

## Outstanding issues updated today

Follow-up implementation and qualification for 69/83/132/133/134/135 is recorded
in [repository issue improvements](repository-issue-completion.md). The rows below
retain the earlier review baseline; consult that follow-up for current status.

| Issue | Candidate evidence and remaining boundary |
|---|---|
| [83 — native samples](https://github.com/imazen/zenpixels/issues/83) | Signed/wide SampleEncoding and exact ST2086 carriage remain. The duplicate native view is removed; U16 RGB10/alpha1 uses the existing component route. Any further signed/wide component support needs a concrete consumer; downstream IMIJ integration is separate. |
| [69 — identity allocations](https://github.com/imazen/zenpixels/issues/69) | EncodeReady::finalize still takes a borrow and documents an owned output allocation. OutputPlan has borrow_rows/convert_rows. Shared interpretation work did not solve consuming finalization or prove allocator-compatible typed extraction. |
| [132 — HLG](https://github.com/imazen/zenpixels/issues/132) | HlgRendering and hlg_rendering.rs exist on the candidate; explorer now exercises explicit rendering and missing/invalid policy. This does not establish every CMS/consumer entry point or HLG ICC synthesis. |
| [133 — documentation/API reconciliation](https://github.com/imazen/zenpixels/issues/133) | API snapshots were refreshed, but ABLATION-zenpixels-convert.md still describes an old feature branch, adapt.rs retains an ignored private-helper example, and corrected-icc-export.md still says unmerged. These are concrete remaining documentation debt, not solved by the explorer. Full semver classification against the intended baseline remains open. |
| [134 — authority boundaries](https://github.com/imazen/zenpixels/issues/134) | output_contracts.rs and output_current_color.rs cover portions of authority/current-color behavior. They do not by themselves establish the complete contradictory-signal table. SynthesizedIcc::for_cicp still delegates to primaries/transfer synthesis; decoded-RGB matrix/range preconditions need the issue's explicit tests/documentation. |
| [135 — P3 export](https://github.com/imazen/zenpixels/issues/135) | Candidate contains tools/p3-export-oracle and the isolated no-db export tool, plus qualified compact P3 exports from prior work. The explorer update does not rerun the independent oracle or reconcile every release ref. Keep release-line exposure distinct from candidate tests. |
| [136 — gain-map transcodes](https://github.com/imazen/zenpixels/issues/136) | Consumer reconstruction/orchestration remains open; core policy support and map carriage are not an end-to-end HDR conversion receipt. |
| [137 — JXL luminance metadata](https://github.com/imazen/zenpixels/issues/137) | Consumer qualification of intensity_target, content peak and diffuse white remains open; the explorer deliberately labels display and source peaks separately. |
| [138 — AVIF matrix resolution](https://github.com/imazen/zenpixels/issues/138) | Consumer migration plus decoded witness remains open. Having shared CICP resolution does not prove the codec uses it. |
| [139 — unsupported authoritative CICP](https://github.com/imazen/zenpixels/issues/139) | Consumer fallback reproduction remains open. Supported HLG rendering does not authorize falling through to a contradictory ICC when authoritative signaling is unsupported. |
| [140 — coded matrix/range](https://github.com/imazen/zenpixels/issues/140) | Consumer metadata and pixel qualification remains open. The plane examples extract codes; they do not perform range expansion or a YCbCr matrix. |
| [141 — PNG/EXIF provenance](https://github.com/imazen/zenpixels/issues/141) | Consumer fixture/reader qualification remains open; the teaching model is not an oracle for PNG precedence or EXIF-only color. |
| [142 — Apple gain maps](https://github.com/imazen/zenpixels/issues/142) | Consumer numerical qualification remains open; no new measurements or tolerances are claimed here. |
| [143 — downstream compilation](https://github.com/imazen/zenpixels/issues/143) | Remains open. New DescriptorError results and removal of infallible RowConverter cloning add migration obligations; compiling the explorer does not compile all downstream consumers. |

Consumer observations above retain the issues' scope; this pass did not modify
or newly qualify those repositories. None of these issues is closed by publishing
an explorer or by a type appearing in an API snapshot.

## Validation

After removal, the core contract and feature gates cover existing component
canonicalization, native pixel carriage and ST2086 preservation. Explorer checks
cover the existing U16 component witnesses, sample padding, HLG policy and
missing-WASM fallback. The historical odd-offset and signed/wide reader controls
were removed with their implementation, not retained as simulated support.
