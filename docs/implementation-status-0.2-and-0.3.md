# Implementation status: 0.2 bridge and 0.3.1

2026-09-27, updated after merging PR #75 and implementing the correctness/preparation batch. **No, the complete requested
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
| Ownership | `into_parts`, `try_from_parts`, `take_parts`, `without_buffer`, `into_contiguous` | Allocation/offset/stride/context tests; typed U8 export reuse implemented for compatible allocator layouts |
| Legacy planar retirement, first step | Whole module and re-export warnings | Warning probes; zenfilters migrated on its companion branch; module removal remains for 0.3 |
| Final-row extent | Owned views/transforms accept minimal visible final row | Adoption regression; broader zero-area/arithmetic audit not complete |
| Small metadata/storage fixes (`7088a8b`) | Empty crop row read; primary containment; CICP padding; orientation context; four RGB/BGR swap helpers; ImgVec stride/storage | Core/convert contract regressions; not every metadata-mutating helper audited/fixed |
| Small conversion fixes (`7088a8b`) | Both known-transfer adapter guards; strict in-place semantic-retag refusal; scalar Adobe gamma; F16 subnormal rounding | Default/minimal tests; no extra full-image analysis pass |
| External CMS composition refusal (`7088a8b`) | `compose` returns None when it cannot retain an external transform | Tests both operands; no general composed CMS executor added |
| Reviews (`b644ac6`) | Code cases, performance costs, U16 proposal, exact narrowing benchmark candidate | Production U16 kernel and raw-sample API unchanged |

## Requested implementation still outstanding

| Work | Status | What is still needed |
|---|---|---|
| Storage and typed layout | Partial | Reinterpretation alignment/type guards and descriptor validation implemented; zero-area row/subview and checked constructor arithmetic hardened. Continue auditing imgref and unusual custom Pixel paths. |
| Remaining allocation reuse | Done | Padded U8 typed exports compact in place and reuse compatible allocation layouts; legacy owned-cow adapters move their allocation. Higher alignment or nonintegral typed capacity requires a copy. |
| Current-color authority | Implemented for buffer/output boundaries | Reject contradictory current CICP/descriptor and dual-field context; forward actual ICC, replace stale signaling after color conversion. zencodec ambiguity fallback migrated. No new public resolved type. Embedded ICC CICP alone no longer proves matrix/TRC substitution; output emits only the selected authority. |
| RGBA→GrayAlpha and matte-to-gray | Done | Dedicated alpha-preserving luma; explicit matte before gray conversion; regressions in CI. |
| Premultiplied nonlinear transfer | Done for ordinary conversion | Source-domain unassociation and destination-domain reassociation. HDR tone mapping also unassociates in the source domain. Additional kernel fusion remains optional. |
| Content-dependent alpha checks | Done via preflight | Padding becomes opaque alpha through explicit insertion; false Opaque claims refuse. `check_opaque` is explicit; row planners refuse unchecked conditional policy. No implicit extra pass was added. Opt-in fused checking remains optional. |
| Unsupported routes | Done for reviewed cases | Missing layouts/Oklab matrices/MoxCms depths fail during setup. A 7,744-case format/transfer matrix verifies accepted plans prepare and execute; BGRA/Oklab depth ordering repaired. |
| Output semantics | Done for current finalizer | SameAsOrigin converts; identity includes alpha/range; actual ICC reaches CMS; failures do not return EncodeReady. Output-plan ownership wrappers remain separate integration work. |
| Complete composition | Done for built-in plans | Stage descriptors/anchors retained; `compose_preserving` explicitly retains quantization. External CMS composition still refuses and requires separate execution. |
| Prepared execution | Implemented | `prepare` / `try_convert_row`, capacity/extent/alignment guards, selected LUT setup; zero-allocation tests. Composed HDR tables are all cached and covered by the zero-allocation gate. |
| CMS ownership and errors | Done | Additive fallible/preparation hooks; independent prepared workers; original error chains; fallible cloning refusal for unique stateful workers. |
| Exact preservation / HDR | Implemented for reviewed routes | Metadata-only exactness proof/refusal; explicit peak measurement; PQ source-peak normalization; linear anchor scaling and target anchor; HLG mapping refuses without OOTF. |
| Conversion ownership modes | Partial | Parts/compaction implemented; broader borrowed/consuming/caller-output identity and failure-ownership contracts not completed |
| Streaming companion work | Done locally | Fallible zencodec pull methods and zenpipe callback fixes committed and tested. No universal core provider added. |
| Native sample signaling / video | Pending | Checked vocabulary, adapter prototype and tests; U16 range-correct conversions or refusal; alpha/component roles; borrowed AOM/SVT integration; explicit RGB path for metrics |
| U16 performance | Partial | Fused U16 analysis and explicit LUT preparation implemented. Production narrowing candidate still awaits cross-platform selection; direct native-code kernels remain separate work. |
| docs.rs / examples | Partial | Reviews and annotations done; final canonical API navigation, executed galleries, accurate generated crate READMEs and complete cost docs remain |
| Compile-time/performance acceptance | Pending | Controlled cold/incremental builds, prepared-path allocations, workers and platform checks; benchmark candidate is not full acceptance |
| Dependent migrations | Partial | zenfilters mask/public boundaries and zencodec current-color fallback migrated. Complete ecosystem migration and paired release-candidate builds remain. |
| 0.3.1 / releases | Pending | Complete bridge first; removal branch, feature gates, retained feature spellings, dependency pairing, semver/package/MSRV checks and publishing |

The owner already authorized implementation of wanted items. These remaining
tasks do not need another blanket approval. Concrete API spelling still needs
to be justified by actual callers; numerical and cost policies already selected
should not be reopened as permission questions.

## Reproduced defects closed

All 27 distinct contract cases now assert the intended behavior: 26 regressions
and the selected quantization-optimization policy. The runner passes in default/CMS
and minimal configurations. Conversion and output cases are also normal integration
tests; storage has its own acceptance-boundary regressions.

Implemented in this follow-up:

- RGBA→GrayAlpha retains alpha; matte-to-gray includes the background.
- Premultiplied nonlinear transfer unassociates before conversion and reassociates
  after it. Intermediate descriptors retain primaries/range.
- Unsupported layouts, Oklab primaries and MoxCms sample-depth pairs refuse during setup.
- `DiscardIfOpaque` requires explicit `adapt::check_opaque` preflight for row planning.
  The whole-image explicit adapter retains its documented preflight.
- `SameAsOrigin` converts current pixels back to the origin profile; identity includes
  alpha/range; actual ICC bytes reach the CMS. Unsupported range changes refuse.
- `compose_preserving` retains intentional intermediate quantization; both composition
  paths retain stage descriptors/anchors. Ordinary composition still optimizes output.
- `RowConverter::prepare` initializes scratch/LUT/backend work and sets width capacity;
  `try_convert_row` checks extents/alignment and propagates original backend errors.
  Prepared stateful workers own their backend without locking or cross-worker sharing (a private `Mutex::get_mut` container retains std `Sync`). `try_clone` refuses
  uncloneable state; legacy `Clone` never silently drops a no_std transform.
- `ConvertPlan::new_preserving_samples` accepts proven representation changes and
  refuses unproven loss without scanning. HDR→encoded-SDR requires peak policy with
  or without experimental kernels. `convert_to_sdr_measuring_peak` makes the prepass
  explicit; the old ambiguous method is deprecated.
- U16 load-bearing analysis fuses opacity/chroma/replication checks in one traversal.
  Explicit preparation warms the selected U16 transfer table before execution.

Prepared execution tests count **zero allocations** over repeated within-capacity
rows. A backend-error regression checks the original concrete error remains in
`Error::source` and finalization never returns `EncodeReady` after a row failure.
This closes the reproduced cases; it is not a claim that every broader release
contract or codec adapter is complete.

Companion commits (local branches, not published): zencodec `ab12c4c` / `3ab2afb` fix current-color fallback and add fallible
pull entry points preserving source errors; zenpipe `a978c10` / `f774d01` fix callback
EOF/scratch reuse and migrate zenfilters to its own `PlaneMask`. Existing unrelated
changes in those checkouts were preserved.

## Cleanup disposition is not a list of completed deprecations

`into_vec` now warns with `into_parts` migration; typed reinterpretation rejects contradictory layouts. Broader conversion ownership and ambiguous color construction still need final migrations. Transfer-blind ICC helpers and legacy HDR measurement/tonemapping
callers still need the individual audits in the consolidated review. Do not
remove them merely because a historical removal queue mentions them.

Keep established open traits, imports and feature spellings. Optional
ChannelOrder aliases, a new universal provider, duplicate context/contiguous-view
helpers and orientation-method renames are not missing required implementations;
they were deferred or declined. A new video carrier is not built by extending
the now-deprecated planar module.

## Remaining integration and release work

The reviewed storage, conversion, alpha, output, prepared-worker and streaming
fixes are implemented. The [bridge guide](implemented-bridge-contracts.md) shows
actual API spellings and costs. Continue with:

1. Prototype native video/sample adapters with real codec/metric owners; the
   existing review explicitly deferred a universal public carrier.
2. Integrate an ownership-aware output plan with zencodec's existing color-emission
   authority before publishing additional output wrappers.
3. Complete ecosystem migrations, cold/incremental build comparisons and paired
   0.2/0.3 release-candidate checks; then remove only fully migrated legacy APIs.

Verification now includes workspace/all-feature tests, 128 external warning probes,
MSRV core 1.85 / converter 1.89, minimal WASM and pure-Rust ARM cross-compilation,
and strict all-feature Clippy. The benchmark-only LCMS feature's ARM cross-build
requires a C cross compiler; no ARM runtime throughput result is claimed.

U16 arithmetic/adapter work can proceed independently where it does not change
the established normalized U16 image contract. Commit each completed chunk.
