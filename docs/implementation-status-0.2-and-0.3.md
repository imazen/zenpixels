> **Superseded release decisions (2026-09-28):** See
> [finalization/release contract](finalization-release-contract.md) for current
> scope. Estimation is always deprecated in 0.2 and removed entirely in 0.3;
> HDR root aliases and core measurement are removed in 0.3; concrete-type
> conversion extension traits are sealed there. Exact ICC normalization uses
> the existing hash through an `OutputProfile` method. The checks below describe
> the older recorded revisions, not blanket validation of subsequent edits.

# Implementation status: 0.2.17 bridge and 0.3.1

2026-09-27. PR #75 is merged. The reviewed storage, conversion, color, alpha,
prepared execution and streaming fixes are implemented and committed. Both
release candidates build from packaged archives. Nothing has been published.

`main` remains at #75; `release/0.2-bridge` contains the 0.2.17 bridge. `release/0.3.1` adds the
warned legacy removals. Companion fixes are on local branches in zencodec and
zenpipe. This is a tested candidate set, not a claim that every published consumer
has migrated. See the [code guide](implemented-bridge-contracts.md) and
[validation record](bridge-validation-2026-09-27.md).

## Implemented

| Area | Result | Cost or boundary |
|---|---|---|
| U8/U16 semantics (#75) | Refuse unsupported narrow-range depth conversions | Normalized full-range U16 remains the ordinary image contract |
| Storage | Validate descriptors, typed layouts, alignment, checked stride arithmetic and empty geometry | Metadata checks, no sample scan; tests include custom Pixel and imgref paths |
| Ownership | `into_parts`, checked `try_from_parts`, `take_parts`, `without_buffer`, `into_contiguous` | Move large allocations; explicit compaction retains context and alignment offset |
| Typed export | Reuse compatible padded U8 allocations; owned legacy adapters move storage | Copy once where typed allocator alignment/capacity requires it |
| Color metadata | Preserve swap/orientation context and ImgVec stride; correct named RGB CICP | No hidden color transform for physical reorder |
| Current-color authority | Reject contradictory current CICP/descriptor and ambiguous ICC+CICP at conversion/output boundaries | No new public resolver type; original metadata remains in ColorOrigin |
| ICC | Forward actual profiles to CMS; embedded CICP alone does not bypass ICC TRCs/LUTs | Setup once; emit only the selected output authority |
| Alpha | RGBA→GrayAlpha retains alpha; matte-to-gray includes background; padding insertion creates opaque alpha | Source-domain unassociation, destination-domain reassociation; multi-step kernels may use row scratch |
| Conditional alpha removal | Explicit `check_opaque` preflight; row planners refuse unchecked `DiscardIfOpaque` | No new implicit pixel prepass; existing explicit whole-image adapter retains its documented scan |
| Unsupported routes | Refuse unavailable layouts/gamut matrices/MoxCms depths during setup | 7,744-case accepted-plan execution matrix plus numerical regressions |
| Composition | Default removes avoidable intermediate quantization; `compose_preserving` retains it | Stage descriptors and luminance anchors survive; external CMS composition refuses |
| Prepared execution | `prepare`, `try_convert_row`, width/extent/alignment checks | Selected scratch/LUT/backend setup before rows; repeated prepared execution allocates zero times |
| CMS workers/errors | Independent mutable workers, provided fallible hooks, original error chains, fallible clone refusal | No prepared-worker locking; backend failure can partially write output |
| Exactness | `new_preserving_samples` proves supported representation changes or refuses | Metadata proof only; provenance is not proof of pixel contents |
| HDR | Validated anchors, explicit peak measurement, PQ normalization and target anchor retention | HLG display mapping refuses without OOTF; raw PQ decode records its 10,000-nit unit |
| Output | SameAsOrigin actually converts; identity includes alpha/range; failure never yields EncodeReady | Existing finalizer/PixelCow/parts/caller-row APIs; no speculative ownership-wrapper family |
| U16 analysis | Fused opacity/chroma/replication traversal, selected LUT preparation | Measured local x86 improvement; production narrowing kernel unchanged |
| Streaming companions | Fallible zencodec pull methods preserve source errors; zenpipe callback fixes EOF and reuses scratch | No universal provider trait or full-image streaming buffer |
| Deprecations | Accidental APIs, planar, `into_vec`, ambiguous peak measurement carry migration pointers | 128 external warning probes on the bridge |
| Docs/builds | Canonical docs.rs exports, examples, generated READMEs, snapshots and compile-cost measurements | No added dependency; core MSRV 1.85, converter 1.89 |

All 27 distinct review cases now assert their intended behavior (26 corrected
regressions and the selected composition policy). The runner passes in default/CMS
and minimal configurations. Prepared tests cover zero allocation, original error
chains, independent workers and composed HDR tables.

## 0.3.1 candidate

Commit `5e40423` removes only the selected warned surface:

- `planar` module and re-exports; the Cargo feature remains a no-op.
- `requires_cms`, `Adapted`, and the three packed `adapt_for_encode*` wrappers;
  use planning errors and stride-aware `*_cow` adapters.
- `PixelBuffer::into_vec`; use `into_parts` with offset/stride/context intact.
- `ColorContext::from_icc_and_cicp`; select current authority explicitly.
- All estimation, including opt-in; root HDR aliases, core measurement, legacy HDR bundle/helpers and ambiguous scan spelling.

Concrete-type conversion/measurement extension traits are sealed. Pixel and CMS
extension points remain open. Legacy CMS/finalizer methods and feature spellings
remain; no renamed orientation API is included.

Default/all-feature tests, strict Clippy, rustdoc, MSRV, 128 removal probes,
public API snapshots and package verification pass. Forced patch-level semver
audits against the bridge report the listed removals and estimation gate only.
The bridge's one tolerated semver exception is documented: original arbitrary CMS
errors remove `ConvertError`'s unwind marker traits, while Send/Sync/Clone remain.

## Same-source compatibility proved for the tested graph

`tests/compat/lib.rs` is compiled unchanged with deprecations denied against all
four **packaged** core/converter pairings: 0.2.17/0.2.17, 0.2.17/0.3.1,
0.3.1/0.2.17 and 0.3.1/0.3.1. Minimal, default, experimental/interop/legacy-feature
and local companion configurations pass: 16 cases. Updating an existing bridge
lockfile to 0.3.1 also passes. Each selected graph contains exactly one core and
one converter version.

The companion configuration uses local zencodec, zenpipe and zenresize sources.
Patching only zenpixels is insufficient: registry zencodec/zenresize versions may
still constrain the core to 0.2, creating incompatible buffer types. This is why
consumers opt in as a connected pipeline. The converter floor is 0.2.17, because
its implementation uses the new descriptor validation API. This does not promise
new bridge APIs on earlier 0.2 releases.

Companion commits, not published:

- zencodec `ab12c4c`, `3ab2afb`, `f4d5da4` on `fix/fallible-pull-source-errors`:
  fallible pull, current-color authority and the testkit's stride-aware adapter.
- zenpipe `a978c10`, `f774d01` on `fix/zenpixels-bridge-contracts`:
  callback EOF/scratch fixes and zenfilters' own public PlaneMask.
- Existing zenresize source already accepts both core lines; no edit was needed.

Unrelated user edits in companion repositories were preserved. These checks use
their local working trees; they are not proof about every published codec feature.

## Separate work, not silently included in these releases

Earlier explicit deferrals remain in effect:

- Native video/sample carrier and borrowed AOM/SVT adapters, including checked
  10/12-bit packing/range vocabulary and explicit VMAF/CVVDP conversion. Do not
  extend the retired planar API or reinterpret video codes as normalized U16.
- A unified zencodec output plan/ownership wrapper family, after a real integration
  establishes what the existing PixelCow, parts and caller-output APIs cannot do.
- A new narrowing kernel, pending ARM runtime measurements. Cross-compilation
  proves buildability, not throughput. Optional kernel fusion can follow separately.
- Cosmetic aliases, universal providers/row iterators and sealing existing open
  traits were deferred or declined; they are not missing release implementations.

Remaining release work is coordinated consumer rollout and publication, with
full codec feature graphs checked as each pipeline opts in. No blanket widening
of all ecosystem manifests or registry publication was performed.
