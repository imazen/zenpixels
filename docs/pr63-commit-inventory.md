# PR #63: missing commits and port decisions

Audit date: 2026-09-27. Main baseline: `07f45ea`. This is a commit-by-commit
inventory and port recommendation, not a claim that the recommended ports have
been implemented. Earlier release audits did not provide this complete ledger.

## What happened

[PR #63](https://github.com/imazen/zenpixels/pull/63) contains 21 commits. It
merged on July 24 into `hdr-fixes-2026-07-14`, after that base branch had already
merged to main. Its merge commit is `44f771659df2de060dba11bbc23d046f8bfb2c2d`;
its final source commit is `1126d054a98ac9420da0d1c28b2251f5cdbd31ff`.

None of the 21 commits is an ancestor of current main. None has an exact
patch-id match on main. **That does not mean all their functionality is absent.**
The 0.2.16 release manually copied parts, and our latest bridge work supersedes
others. The merge differs from the PR head only in three ICC-generator/dependency
files for the base's moxcms update; it does not hide another API implementation.

Do not cherry-pick the stack or its final commit wholesale. Reuse selected final
hunks and tests, reconciling them with current storage and compatibility rules.
The [machine-readable ledger](audit-2026-09-27/pr63-commits.json) records all
21 full SHAs, ancestry, patch-equivalence results and dispositions.

## Exact traits and API spelling changes

The PR sealed **two existing conversion traits** in `zenpixels-convert`:

- `PixelBufferConvertExt`, directly through a private `sealed::Sealed`
  supertrait, implemented only for the erased `PixelBuffer`.
- `PixelBufferConvertTypedExt`, indirectly through its existing
  `PixelBufferConvertExt` supertrait.

These remain open on current main. Sealing prevents downstream implementations
for callers' own wrapper types. Existing callers merely importing the traits and
calling methods on `PixelBuffer` would be unaffected. The original justification
was that a sibling audit found no external implementations and sealing would
allow future required methods. That is an API evolution choice, with no runtime
performance benefit; it is not necessary to implement caller-output conversion.
The PR also added required methods, a separate compatibility concern for existing
implementors. Our recommendation is to retain the existing open traits for the
common-source bridge, unless an explicit implementation migration is agreed.

The final PR revision also introduced two **new, already-sealed** traits:
`PixelSliceOrientationExt` and `PixelBufferOrientationExt`. These were method
wrappers around existing orientation functions, plus a consuming owned-buffer
method. Neither trait exists on main. This was not sealing an existing orientation
trait. The already-sealed `PixelSliceLoadBearingExt` and
`PixelBufferLoadBearingExt` are separate and are not the compatibility objection.

Most changes loosely called “renames” were actually new methods plus deprecated
wrappers, not immediate removal of an old spelling:

| Old spelling | PR destination | Status / recommendation |
|---|---|---|
| `PixelSlice::new_tight(...)`, `PixelSliceMut::new_tight(...)` | `::new_contiguous(...)` | Actual rename within the unpublished PR. Neither spelling reached main. Defer redundant constructors. |
| Free `convert_row(&plan, src, dst, width)` | `plan.convert_row(src, dst, width)` | Proposed deprecation absent main. Same per-call scratch behavior; do not require this intermediate migration. |
| Free `apply_orientation(src, orientation)` | `src.apply_orientation(orientation)` via `PixelSliceOrientationExt` | Proposed deprecation absent main. Retain the free function; method syntax is optional design, not a correctness fix. |
| Free `apply_orientation_into(src, orientation, dst)` | `src.apply_orientation_into(orientation, dst)` via the same trait | Same disposition. |
| Free `apply_orientation_in_place(&mut buffer, orientation)` | `buffer.apply_orientation_in_place(orientation)` via `PixelBufferOrientationExt` | Same disposition. |
| Unreleased free `into_oriented(buffer, orientation)` | `buffer.apply_orientation(orientation)` via `PixelBufferOrientationExt` | The PR removed its own new free helper. Neither destination exists on main. |
| `RowConverter::convert_rows(...)` with raw buffers/strides | `converter.convert_slice_into(src_view, dst_view)` | A substantive signature/contract improvement, not merely a rename. Neither replacement nor warning landed; repair source checks, preparation and metadata before migration. |
| `adapt::convert_buffer(...)` | Owned buffer conversion methods (`convert_to`, `convert_into`, `convert_in_place`) | Proposed migration rather than a one-to-one rename. Defer warning until the final replacement contract is ready. |
| `adapt_for_encode`, `adapt_for_encode_with_intent`, `adapt_for_encode_explicit` | Corresponding `_cow` functions | Already on main with deprecations. This changes the result from packed `Adapted` to stride-aware `PixelCow`; keep this migration. |

`new_packed` was an alias added and removed inside the PR, not a main API to
migrate. Our newly adopted `into_contiguous()` is also not a rename of a shipped
`into_contiguous_bytes()`: the latter never landed, and returning the buffer
preserves its descriptor and color context. `convert_to` was retained in the PR;
`convert_into` and `convert_in_place` were additional ownership/output choices.

## Recommended small chunks

1. **Fix known-transfer relabeling first.** Port the `Unknown` guards from
   `1126d05` in both the intent and explicit-policy cow adapters. Add regressions
   for both paths, policy enforcement, and the legacy wrappers. Preserve exact
   identity borrowing. This is an existing correctness bug, not a new API.
2. **Finish ownership without extra image copies.** Adapt the final typed-export
   optimization and the owned-cow-to-legacy adapter. Then implement checked parts
   adoption against our current `PixelBufferParts`, returning ownership on error.
3. **Repair surviving documentation and test gaps.** Correct packed/alignment
   claims, update deprecated finalizer links, adapt tested usage galleries, and
   execute them in CI. Add the missing `rgb,std` minimal-feature test configuration.
4. **Revisit conversion helpers with the execution contract.** Caller-output
   conversion and fallible typed helpers are useful ideas, but the PR implementations
   do not establish the promised allocation, color-context or failure guarantees.
   Do this with prepared/fallible execution, not as an independent family of wrappers.

The following tables distinguish these recommendations from work already done.

## Every commit

“Adapt” means reuse selected final implementation/test material, not replay the
original patch. Later commits sometimes reverse earlier ones.

| # | Commit | Contents | Current main and disposition |
|---|---|---|---|
| 1 | [d672211](https://github.com/imazen/zenpixels/commit/d672211d1cd71a79647f4e6e712aa5b1b3f760d3) | Tested core/conversion usage galleries; CI examples gate | Galleries and execution gate are missing. **Adapt** final examples to the accepted API; omit helpers we are declining. |
| 2 | [e2463f8](https://github.com/imazen/zenpixels/commit/e2463f879e31c42e1a18c082c8b86dbb7eccc190) | Original ergonomics findings | Historical report absent; newer audits supersede its counts and recommendations. **Keep as history**, not a new current report. |
| 3 | [2e1a4fb](https://github.com/imazen/zenpixels/commit/2e1a4fbe6e218d4d0943c178077e471375a1c688) | Contiguous-view constructors, `Adapted::as_pixel_slice`, `convert_slice`, CICP helpers | `as_pixel_slice` was copied and is now deprecated. **Defer** redundant view constructors and descriptor retagging; **adapt** slice conversion only with its complete contract. `Cicp::from_bytes` was removed in the final PR revision. |
| 4 | [7a7614e](https://github.com/imazen/zenpixels/commit/7a7614e29714259eaa45b74898e74bf292edf7ff) | Examples/changelog for preceding helpers | **Rewrite** documentation for the actual retained surface; do not import claims that missing methods shipped. |
| 5 | [649e5f7](https://github.com/imazen/zenpixels/commit/649e5f7ea76680909c67700a22d41f02b0c8e629) | `new_tight` → `new_contiguous` | Neither constructor landed. **Defer** the convenience; no existing main callers need this rename. This is separate from the consuming `into_contiguous` we implemented. |
| 6 | [fefd0d7](https://github.com/imazen/zenpixels/commit/fefd0d7fa21802dbaa49eb79ec37b19070264a0d) | `new_packed` aliases | Reversed by #16; absent main. **Skip**. |
| 7 | [9df4db6](https://github.com/imazen/zenpixels/commit/9df4db665e903efe7666c6700c630c9dfbb2de12) | Retain finalization as the output contract | Finalizers already exist. **Retain the direction**, but do not copy unproven atomic pixel/profile or allocation guarantees. |
| 8 | [bc14c04](https://github.com/imazen/zenpixels/commit/bc14c042bea3d625b2b91c806ea07c986d1a67fb) | Point crate docs at `finalize_for_output_with` | Deprecated finalizer links survive on main. **Adapt** these doc corrections, qualifying guarantees against current implementation. |
| 9 | [b101c4b](https://github.com/imazen/zenpixels/commit/b101c4be2a1c0d54cee785cde024c52a0e6ff298) | Fix citations/status in the old report | **Superseded** by current source-backed audits. No production code missing. |
| 10 | [5e959de](https://github.com/imazen/zenpixels/commit/5e959def21466a618d0cd005bab4dd825cdef151) | Hard-gate estimation | Our `f7252be` supplies opt-in plus compatibility warnings on 0.2. **Already replaced for 0.2; hard-gate in 0.3.1.** Do not remove published APIs in the bridge. |
| 11 | [a41332a](https://github.com/imazen/zenpixels/commit/a41332acbdc645e27a142b20cf11e497aba96761) | `convert_slice_into`; allocating wrapper | Missing. **Adapt the idea**, not the claimed no-allocation contract. Source descriptor checking, preparation and color context need repair; see below. |
| 12 | [230a980](https://github.com/imazen/zenpixels/commit/230a980919e0712d7faaee7712ae0ba0399c4f3e) | Correct three packed/SIMD documentation claims | Two surviving sites remain wrong: core crate overview and load-bearing output docs. **Apply those corrections**; the third describes a method not on main. |
| 13 | [bc9f8b8](https://github.com/imazen/zenpixels/commit/bc9f8b836ec99a4627bdab43a066e7dd8f555b88) | `convert_into`, extension-trait sealing, `convert_rows` deprecation | Missing. **Adapt caller-output conversion later**; **reject sealing existing open traits and adding required methods** under our common-source promise. Deprecate only once the final replacement exists. Do not copy expanded tolerated-break policy. |
| 14 | [2f51693](https://github.com/imazen/zenpixels/commit/2f5169327f400fc08ac07e91768b93bc50e01dc5) | `convert_in_place`, fallible `try_to_*` helpers | Missing. **Rewrite** strict in-place versus consuming allocation fallback as distinct contracts. **Consider** fallible typed helpers using compatible provided methods, after checking real callers and existing composition. |
| 15 | [008a1a8](https://github.com/imazen/zenpixels/commit/008a1a8bcbdc16f7ed593c2ea8c8f519d0a3a079) | `Adapted` restrictions, `requires_cms` demotion, `convert_buffer` warning, export docs | Our `requires_cms` deprecation replaces demotion for 0.2. `Adapted` non-exhaustiveness was **reversed in #21**. **Defer** `convert_buffer` warning until replacement; **adapt** accurate export cost docs. |
| 16 | [fcf78ed](https://github.com/imazen/zenpixels/commit/fcf78edbdda613ca4aa300c9fce1d17e6e4ad53c) | Remove unreleased `new_packed` | Main already has neither alias. **Skip**. |
| 17 | [4114e1b](https://github.com/imazen/zenpixels/commit/4114e1b9fd2b796e9f0c93c72f6923dad11750f6) | Queued removal inventory | Copied into release and subsequently clarified by our audit. **Superseded** by the bridge checklist; do not adopt blanket removal of features/free orientation functions. |
| 18 | [ede5e8a](https://github.com/imazen/zenpixels/commit/ede5e8a9d82c14b843d4cb0d1f848320288e6429) | Consuming bytes, conversion, orientation and load-bearing conveniences | Our metadata-preserving `into_contiguous` + parts **supersede bytes-only export**. **Adapt final typed-export optimization** from #21. **Defer** other consuming helpers; free `into_oriented` was canceled by #21. |
| 19 | [8d67345](https://github.com/imazen/zenpixels/commit/8d67345b51cf7cb05bdb6b15e33944a003aadf9c) | `(Vec, PixelBufferLayout)` parts round-trip | Extraction **superseded** by our destructurable record. Adoption missing. **Rewrite validation/error ownership**, reusing useful tests; do not introduce the private layout wrapper. |
| 20 | [040f260](https://github.com/imazen/zenpixels/commit/040f2601d0ddf397d7bfffe19e2926f0e12fed67) | `PixelCow`, cow adapters, `into_vec` deprecation | Cow surface already copied; current cow-first implementation also incorporates the final architectural direction. **Do not duplicate.** `into_vec` warning is missing; **defer** until adoption/migration is ready. |
| 21 | [1126d05](https://github.com/imazen/zenpixels/commit/1126d054a98ac9420da0d1c28b2251f5cdbd31ff) | Large final refinement spanning ownership, adapters, conversions, docs and CI | **Split**, as detailed below. Contains the most important missing fix, useful optimizations, work already on main, and API changes we should decline. |

## Split the final refinement

| Change | Disposition and reason |
|---|---|
| Require `TransferFunction::Unknown` for transfer-agnostic borrowing | **Port both guards and expand tests.** Main's intent and explicit-policy variants omit this condition. Known transfers must convert or fail according to policy. |
| Typed owned export preflights allocation alignment and capacity divisibility | **Adapt.** Reuse padded, castable U8 storage; copy wide pixels or incompatible capacities directly once. Never reinterpret allocation alignment merely because its address happens to be aligned. |
| Move owned cow storage into legacy `Adapted` | **Adapt internally.** Main currently calls `copy_to_contiguous_bytes`, duplicating the owned image. Avoid adding a public bytes-only method to achieve this. |
| Shared private row compaction and `CompactionStride` | **Consider reuse** to remove duplicated alpha-drop/load-bearing loops, preserving current stride and offset invariants. No new public API needed. |
| Cow adapter regression tests | **Adapt** borrowing/conversion/stride cases. The proposed wide-misalignment test uses an alignment-1 byte array plus offset 1, which does not guarantee misalignment; use an explicitly aligned fixture. |
| Reject all borrow-validation errors except `StrideNotPixelAligned` | **Review policy first.** Current main can copy misaligned wide input into valid owned storage. The PR's refusal is a behavior change, not automatically a correctness improvement. |
| `ConvertPlan::convert_row` plus deprecated free function | **Decline this destination.** It creates scratch per call; moving the function into another namespace does not establish prepared execution. Keep one final migration destination. |
| Sealed orientation extension traits and deprecating free orientation helpers | **Decline cosmetic churn.** Existing free helpers are adequate; preserve warning-free common source. |
| Remove `Cicp::from_bytes`, free `into_oriented`; undo `Adapted` non-exhaustiveness | These were reversals of intermediate proposals. **Do not resurrect the canceled additions/restriction.** |
| `cms-moxcms` and minimal `rgb` test additions | **Already covered** by current self dev-dependency and CI. Avoid duplicate jobs. Minimal `rgb,std` execution is still a useful missing check. |
| Examples execution and adapted galleries | **Port together.** Running `cargo test --workspace --examples` matters because these galleries contain executable tests. |
| YAML line-ending attributes and API snapshots | Line-ending rule is optional hygiene; **regenerate snapshots** for actual adopted APIs, never import the obsolete snapshot wholesale. |

### Ownership differences that matter for large images

Our `into_contiguous` retains the alignment prefix/offset, descriptor and color
context while compacting rows in the original allocation. The PR's bytes-only
export moves packed bytes to index zero and discards metadata. Therefore taking
`buffer.into_contiguous().into_parts().data` is **not** a correct packed-byte
export when offset is nonzero. A compatibility adapter or typed U8 Vec exit must
handle that prefix explicitly. Stride-aware consumers should take parts directly
and avoid compaction altogether.

The PR's `try_from_parts(data, layout)` returns only an error, dropping the
possibly 100 MB allocation on rejection. Our adoption path must return the
supplied parts on error. Reconcile its full `stride * height` check with minimal
final-row extents and every resulting view; simply relaxing one check can leave
`as_slice` inconsistent. Preserve pointer, capacity, offset, stride and context
on successful round-trips.

### Conversion code is useful reference material, not the finished contract

The final `RowConverter::convert_slice_into` checks output geometry and target
descriptor, but not the source descriptor against the plan's source. It calls
infallible row execution with lazy scratch allocation, despite documenting “no
allocation.” The allocating wrapper copies the source color context onto the
converted output without resolving output profile authority.

Likewise, the PR's `convert_in_place` allocates a replacement image when widening,
allocates row scratch when narrowing, and carries forward old color context.
These are reasons to implement the proposed prepared/fallible conversion contract
before exposing the convenience methods. Adding required methods to existing
open extension traits would also break otherwise warning-free implementations.

`with_color_from_cicp` leaves existing descriptor axes unchanged for unmapped
codes. Avoid publishing that retagging convenience until unknown metadata and
color authority have a single interpretation. Existing descriptor setters and
`Cicp::to_descriptor` already cover basic composition.

## Verification and reproducible findings

Git ancestry and patch comparison were checked against all 21 API-listed commits:

```sh
git merge-base --is-ancestor <each-commit> 07f45ea
git cherry -v 07f45ea 1126d05 63d5b2a
git diff 1126d05 44f7716 --stat
```

All ancestry checks report false; `git cherry` reports `+` for every commit.
Individual patches and final source were compared with main; commit titles and
PR prose alone were not treated as implementation evidence. In particular, the
PR description mentions checked `InPlacePixels::try_new`, but it is absent from
both final head and merge. Release commit `640dced` added that separately; it is
already on main and is not a missing PR port.

A standalone external probe against local main, with the `rgb` feature, reproduced:

```text
known transfer change: borrowed=true, bytes=[128, 128, 128]
padded U8 typed export: allocation_reused=false
```

The first probe is this currently accepted call:

```rust
let source = PixelDescriptor::RGB8_SRGB;
let target = source.with_transfer(TransferFunction::Linear);
let result = adapt_for_encode_cow(&[128, 128, 128], source, 1, 1, 3, &[target])?;
// Current bug: borrowed, unchanged bytes, but descriptor now says Linear.
// Required: perform the transfer conversion (or return a policy/backend error).
```

The second probe owns a capacity-16 Vec containing two 4-byte RGBX pixels at
stride 8, changes its view from width 4 × height 1 to width 1 × height 2, then calls
`into_contiguous_pixels::<Rgbx>()`. Pixel alignment is 1 and capacity is divisible
by pixel size, yet the output pointer differs. The PR's final optimization
provides reference code for reusing this allocation.

Probe setup/output and individual patches were retained under
`/tmp/zenpixels-pr63-inventory/` during the audit. This report and JSON ledger
are the durable record; temporary files are not required for the port plan.
These two probes confirm current behavior, not that every proposed PR hunk has
been tested against current main. Ported production changes still need focused
regressions. This audit itself changes documentation only.
