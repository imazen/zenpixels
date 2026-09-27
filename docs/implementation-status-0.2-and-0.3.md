# Implementation status: 0.2 bridge and 0.3.1

2026-09-27, source audited at `b644ac6`. **No, the complete requested
implementation is not finished.** Review documents and chosen policies are not
implemented APIs. Nothing here declares the bridge or 0.3.1 ready to release.

This status supersedes stale completion wording in earlier proposal documents.
The [performance matrix](performance-review-0.2-and-0.3.md#cost-of-every-code-review-item)
now marks every item Done, Partial, Pending or Deferred. Done applies
only to the stated scope. The [U16 matrix](u16-contract-matrix.md) separately
marks existing behavior, proposed operations and known incorrect routes.

## Completed on main, not yet released

| Area | Implemented | Evidence / boundary |
|---|---|---|
| Accidental 0.2.16 API exposure | `requires_cms` warning; explicit `Adapted::as_pixel_slice` warning; estimation opt-in/warnings | External consumer warning checks; no blanket conversion API removal |
| Anchor validation | `DiffuseWhite::new` rejects invalid values and remains const | Validity tests; HDR planning still needs work |
| Ownership | `into_parts`, `try_from_parts`, `take_parts`, `without_buffer`, `into_contiguous` | Allocation/offset/stride/context tests; typed export reuse still pending |
| Legacy planar retirement, first step | Whole module and re-export warnings | Warning probes; module removal and zenfilters migration not done |
| Final-row extent | Owned views/transforms accept minimal visible final row | Adoption regression; broader zero-area/arithmetic audit not complete |
| Small metadata/storage fixes (`7088a8b`) | Empty crop row read; primary containment; CICP padding; orientation context; four RGB/BGR swap helpers; ImgVec stride/storage | Core/convert contract regressions; not every metadata-mutating helper audited/fixed |
| Small conversion fixes (`7088a8b`) | Both known-transfer adapter guards; strict in-place semantic-retag refusal; scalar Adobe gamma; F16 subnormal rounding | Default/minimal tests; no extra full-image analysis pass |
| External CMS composition refusal (`7088a8b`) | `compose` returns None when it cannot retain an external transform | Tests both operands; no general composed CMS executor added |
| Reviews (`b644ac6`) | Code cases, performance costs, U16 proposal, exact narrowing benchmark candidate | Production U16 kernel and raw-sample API unchanged |

## Requested implementation still outstanding

| Work | Status | What is still needed |
|---|---|---|
| Storage and typed layout | Partial | All zero-area/overflow paths; new alignment checks for reinterpretation; stop typed reinterpretation retaining a contradictory pixel type; validate contradictory descriptors at acceptance boundaries |
| Remaining allocation reuse | Pending | Typed U8 exports and owned-cow paths from PR #63; prove reuse or document unavoidable allocator-layout copies; no speculative constructor family |
| Current-color authority | Pending | Resolve descriptor/ICC/CICP/unknown assumptions once; named-PQ/CICP agreement; remove ambiguous fallback only after a working migration |
| RGBA→GrayAlpha and matte-to-gray | Pending | Correct luma and alpha/compositing order or reject unsupported route during planning |
| Premultiplied nonlinear transfer | Pending | Unassociate/transform/reassociate correctly, preferably fused |
| Content-dependent alpha checks | Pending | Owner chose explicit preflight or opt-in fused checks; enforce that contract rather than silently accepting unchecked RowConverter work |
| Unsupported routes | Pending | Reject unsupported gamut/layout and CMS depth pairs during planning instead of panicking during execution |
| Output semantics | Pending | Correct SameAsOrigin conversion; identity includes alpha/range; actual ICC profiles reach CMS; output metadata describes resulting pixels |
| Complete composition | Partial | Default quantization optimization retained; external CMS erasure refused. Per-stage descriptor/anchor semantics and explicit preserved-stage plan option remain unimplemented. Separate row conversions currently preserve a stage |
| Prepared execution | Pending | Explicit scratch/LUT/backend setup; width capacity/extent checks; fallible execution; no allocation growth during execution within capacity |
| CMS ownership and errors | Pending | Fallible backend row interface; independently prepared workers; stop no_std cloning from losing transforms; avoid std mutex serialization in independent workers |
| Exact preservation / HDR | Partial | Anchor validation done. Exactness proof/refusal, explicit peak measurement, unsupported mapping refusal, luminance conventions and output policy remain |
| Conversion ownership modes | Partial | Parts/compaction implemented; broader borrowed/consuming/caller-output identity and failure-ownership contracts not completed |
| Streaming companion work | Pending | In zencodec, propagate pull-encoder source errors; in zenpipe, fix CallbackSource EOF/row scratch behavior. Optional core resident iterator remains deferred until useful to a caller |
| Native sample signaling / video | Pending | Checked vocabulary, adapter prototype and tests; U16 range-correct conversions or refusal; alpha/component roles; borrowed AOM/SVT integration; explicit RGB path for metrics |
| U16 performance | Candidate only | Production selection after adequate x86/ARM measurements; fuse requested U16 analysis; explicit transfer-LUT preparation; evaluate direct 10/12-bit kernels |
| docs.rs / examples | Partial | Reviews and annotations done; final canonical API navigation, executed galleries, accurate generated crate READMEs and complete cost docs remain |
| Compile-time/performance acceptance | Pending | Controlled cold/incremental builds, prepared-path allocations, workers and platform checks; benchmark candidate is not full acceptance |
| Dependent migrations | Audit only | Migrate actual callers, including zenfilters PlaneMask/public boundaries; remove suppressed deprecations; same-source tests against both release candidates |
| 0.3.1 / releases | Pending | Complete bridge first; removal branch, feature gates, retained feature spellings, dependency pairing, semver/package/MSRV checks and publishing |

The owner already authorized implementation of wanted items. These remaining
tasks do not need another blanket approval. Concrete API spelling still needs
to be justified by actual callers; numerical and cost policies already selected
should not be reopened as permission questions.

## Fifteen remaining reproduced defects

The standalone [contract-case runner](../scripts/check-contract-cases.py) runs
26 tests under each of default/CMS and minimal configurations: 27 distinct cases
overall. Eleven verify fixes, one verifies the selected composition policy, and
**fifteen still deliberately reproduce incorrect behavior**. Passing that runner
does not establish release correctness.

| Case file | Remaining incorrect behavior |
|---|---|
| [storage](contract-cases/storage.rs) | Named PQ versus equivalent CICP resolves differently |
| storage | Contradictory format/alpha declarations are accepted |
| [conversion](contract-cases/conversion.rs) | RGBA→GrayAlpha is accepted as identity |
| conversion | RowConverter does not enforce DiscardIfOpaque |
| conversion | Composite-to-gray ignores the background |
| conversion | Premultiplied transfer conversion uses the wrong domain |
| conversion | Unsupported Adobe/Oklab route panics during execution |
| conversion (CMS feature) | MoxCms cross-depth route panics after successful construction |
| [output/CMS](contract-cases/output_cms.rs) | SameAsOrigin can mistag current pixels |
| output/CMS | Output identity relabels premultiplied alpha as straight |
| output/CMS | Output ignores requested signal range |
| output/CMS | CMS does not receive the actual ICC profiles |
| output/CMS (minimal features) | Clone discards the external transform |
| output/CMS | Reinterpretation accepts invalid sample alignment |
| output/CMS | Typed reinterpretation retains the wrong pixel type |

This is a known-case inventory, not a claim that only fifteen defects exist.
For example, narrow-range cross-depth arithmetic and composition anchor handling
are outstanding beyond this set. The U16 matrix also identifies replicated-byte
compaction as insufficient proof for narrow-range semantic preservation. Keep
adding behavior regressions as work lands.

## Cleanup disposition is not a list of completed deprecations

`into_vec`, typed-preserving reinterpretation, conversion/CMS/finalization
interfaces and ambiguous color construction still need their final replacements
and migrations. Transfer-blind ICC helpers and legacy HDR measurement/tonemapping
callers still need the individual audits in the consolidated review. Do not
remove them merely because a historical removal queue mentions them.

Keep established open traits, imports and feature spellings. Optional
ChannelOrder aliases, a new universal provider, duplicate context/contiguous-view
helpers and orientation-method renames are not missing required implementations;
they were deferred or declined. A new video carrier is not built by extending
the now-deprecated planar module.

## Next implementation chunks

1. Finish metadata/layout guards and planning-time unsupported-route refusal.
2. Correct alpha/transfer/output/CMS behavior with explicit check costs.
3. Complete prepared execution, backend ownership and streaming contracts.
4. Prove the sample vocabulary in real codec adapters and select measured kernels.
5. Finish consumer migrations, docs.rs and both release-candidate checks.

U16 arithmetic/adapter work can proceed independently where it does not change
the established normalized U16 image contract. Commit each completed chunk.
