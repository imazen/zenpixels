# Published 0.2.16 API review: what escaped the stacked PR

Reviewed 2026-09-27 against downloaded crates.io archives for **both crates**, at
0.2.14 and 0.2.16, plus GitHub PR metadata and local commit ancestry. This is a
release-history review. **The three deprecations below are now implemented on
main for the prospective 0.2.17 bridge**, with downstream warning checks in
`scripts/check-deprecations.py`. No public signature or runtime behavior was
removed or changed; the wider contract proposals remain unimplemented.

## Confirmed release history

Both crates went directly from **0.2.14 to 0.2.16**. The registry lists no
0.2.15 release. The `[0.2.15]` changelog section describes unpublished work.

| Event | UTC timestamp | Evidence |
|---|---|---|
| Core 0.2.14 published | June 18, 07:15:26 | crates.io version metadata |
| Convert 0.2.14 published | June 18, 07:16:29 | crates.io version metadata |
| PR #65 merged to main | July 23, 21:56:22 | `a9bdfe5`, base `main` |
| PR #63 merged to its old base branch | July 24, 10:19:02 | `44f7716`, base `hdr-fixes-2026-07-14` |
| Core 0.2.16 published | July 24, 19:02:48 | archive VCS commit `640dced` |
| Convert 0.2.16 published | July 24, 19:03:06 | archive VCS commit `640dced` |

[PR #65](https://github.com/imazen/zenpixels/pull/65) had already merged the
base branch. [PR #63](https://github.com/imazen/zenpixels/pull/63) retained that
branch as its target; merging into it afterward did not put its changes on main.
`44f7716` is not an ancestor of current main; `a9bdfe5` and `640dced` are.

The release commit manually brought over PixelCow, the cow adapters,
`Adapted::as_pixel_slice`, and the removal inventory. It separately added checked
InPlacePixels construction: despite the PR description mentioning it, that
implementation is absent from PR #63's final head and merge. See the complete
[21-commit inventory and port decisions](pr63-commit-inventory.md).
The release did **not** bring over the
estimation feature gate, the `requires_cms` demotion, the new conversion/parts
interfaces, or their full deprecation work. These are distinct facts; the
release was neither the unmodified pre-review branch nor the full reviewed PR.

All shipped files under `src/` in each downloaded archive matched the corresponding
release tag byte-for-byte. Archive VCS metadata identifies `5d5a892` for 0.2.14
and `640dced` for 0.2.16. Sources, API snapshot diffs, PR metadata and probes are
in `/tmp/zenpixels-release-audit/`. Public snapshots were used as an index, then
checked against source: changes in snapshot formatting/visibility are not
necessarily new APIs. For example, InPlacePixels::new already existed in 0.2.14.

## Immediate deprecation recommendation

There are **two accidental exposures**, plus **one newly added legacy helper
whose warning needs completing**. All three are in zenpixels-convert.

### 1. `requires_cms`: deprecate the public function

- Absent in published 0.2.14; public and root-re-exported in 0.2.16.
- PR #63 explicitly made it `pub(crate)` in commit `008a1a8`, because at that
  point it was unshipped. That change missed the release.
- Current definition: `zenpixels-convert/src/convert.rs:461`; root export:
  `src/lib.rs:506`.
- It checks the color-model family, not complete conversion feasibility, ICC
  interpretation, depth support or backend capability. It should not be a
  public preflight substitute for actually planning the operation.
- The focused external search found no consumer of this function in primary
  checkouts. This supports a small migration cost, not removal in a patch.

Proposed bridge annotation, with the actual release number substituted if needed:

```rust,ignore
#[deprecated(
    since = "0.2.17",
    note = "attempt conversion planning and handle ConvertError::NeedsCms; \
            this color-model predicate does not establish conversion support"
)]
pub fn requires_cms(from: &PixelDescriptor, to: &PixelDescriptor) -> bool;
```

Keep the internal predicate as a private helper. Retain a public forwarding
wrapper until 0.3.1 rather than silencing deprecation warnings throughout the
planner. Keep `ConvertError::NeedsCms`; it is a useful new error, not something
to undo with the predicate.

```rust,ignore
// OLD: Boolean advertised as a preflight decision.
if requires_cms(&from, &to) { /* choose CMS */ }

// NEW: inspect the actual planning outcome; propagate other errors too.
match ConvertPlan::new(from, to) {
    Ok(plan) => { /* use plan */ }
    Err(error) if matches!(error.error(), ConvertError::NeedsCms { .. }) => {
        /* retry with the required source profile and CMS */
    }
    Err(error) => return Err(error),
}
```

This uses existing APIs. The larger complete-encoding/prepared-worker design
will strengthen planning; a successful legacy plan is not proof that all the
separately identified execution bugs have already been fixed.

### 2. Estimation: deprecate implicit/default exposure, provide explicit opt-in

The entire following family is new since published 0.2.14 and accidentally
shipped **without the planned experimental feature**:

- `pub mod estimate`;
- `ComputeEnvironment`, `ImageCharacteristics`, `ResourceEstimate`, `SimdTier`,
  both module paths and root re-exports, including their methods/variants;
- `ConvertPlan::estimate` and `ConvertPlan::estimate_in`.

PR #63's `5e959de` put these behind default-off `estimation-experimental`.
Its intent was to **retain the work experimentally**, not delete resource
estimation. We should honor that distinction in the recovery.

Recommended bridge sequence:

1. Add recognized feature `estimation-experimental = []`.
2. Keep the API present without it throughout 0.2.x, but conditionally deprecate
   the module and both methods on ConvertPlan when it is absent.
3. Preserve the exact APIs, without those warnings, when the feature is enabled.
4. In 0.3.1, gate this family on that same feature and retain its opted-in shape.

```rust,ignore
#[cfg_attr(
    not(feature = "estimation-experimental"),
    deprecated(
        since = "0.2.17",
        note = "enable estimation-experimental; required for this API in 0.3.1"
    )
)]
pub mod estimate;
// Apply the same conditional attribute to ConvertPlan::estimate and estimate_in.
// Ensure all root re-exports carry the underlying warning as well.
```

Consumer migration is a manifest change, not source churn:

```toml
# Before: estimation was implicitly available.
zenpixels-convert = "0.2.16"

# After the complete bridge ships, explicitly opt into the retained API.
zenpixels-convert = { version = ">=0.2.17, <0.4", features = ["estimation-experimental"] }
```

The new range is illustrative and assumes 0.2.17 is the complete bridge; keep
the connected dependency graph on one tested core/convert release line.
An “experimental” name does not excuse breaking the same-source guarantee in
0.3.1. If we do not want to retain this shape even under opt-in, use an
unconditional deprecation and specify a real migration instead of offering an
opt-in that falsely promises compatibility.

Why isolate it: resource estimates currently model a fresh output allocation
plus row scratch; a scheduler for borrowed identity, consuming reuse, caller
output and prepared workers needs the execution/ownership mode too. A published
0.2.16 probe reports 36,000,000 destination bytes for identity RGB8 at 4000×3000.
That is consistent with its allocating-output model, but it does not describe
our proposed O(1) ownership handoff. Do not call it a measured memory ceiling
for every execution mode. The docs also describe row ping-pong scratch as
“full-image intermediate buffers,” and claim trivial codec-type conversion even
though the codec types have diverged. These contracts need refinement with an
actual scheduler integration.

The renewed search found codec estimation users, including zenmetrics' fleet
planner and zenpipe/zencodecs, but their types come from **zencodec**, not
zenpixels-convert. Do not migrate those unrelated APIs. No external converter
estimation call was identified among the reviewed primary-checkout candidates.

### 3. `Adapted::as_pixel_slice`: explicitly deprecate this method too

This helper was added in the 0.2.16 release commit. `Adapted` and its old factory
functions were already marked deprecated in that release, but the new method
has no deprecation of its own (`adapt.rs:167` on current main).

I verified against the downloaded 0.2.16 crate that this call compiles with
`#![deny(deprecated)]` when the value arrives from a legacy provider:

```rust,ignore
let legacy = legacy_provider(); // inferred Adapted, supplied by another layer
let view = legacy.as_pixel_slice()?; // no warning from this method today
```

The probe suppresses deprecation only inside the legacy provider used to model
an older dependency. The calling code itself does not suppress warnings. This
does not show that a fully migrated producer would be warning-free; it shows
that the method boundary does not guide a consumer receiving an old value.

```rust,ignore
#[deprecated(
    since = "0.2.17",
    note = "use a *_cow adapter returning PixelCow, then PixelCow::as_slice"
)]
pub fn as_pixel_slice(&self) -> Result<PixelSlice<'_>, At<ConvertError>>;

// Destination: update the producing boundary, then borrow its result.
let pixels = adapt_for_encode_cow(data, desc, width, rows, stride, supported)?;
let view = pixels.as_slice();
```

Deprecating the whole type does not automatically deprecate inherent methods
defined in a separate impl. A small external rustc probe confirmed this;
deprecating a containing module *did* warn on nested methods and re-exported
types in that probe. Test the real paths in the implementing PR rather than
assuming a root `pub use` annotation reaches every usage.

## New additions I would retain

| Actual 0.2.16 addition | Disposition |
|---|---|
| PixelCow and its borrowing/ownership methods | Keep; canonical carrier with real adapter users |
| Three `*_cow` adapters | Keep; fix their semantic-retagging bugs separately |
| InPlacePixels::try_new | Keep; checked construction is the intended direction |
| Buffer/slice `with_cicp`, `with_icc`, `with_diffuse_white`; context `with_cicp`, `with_icc` | Keep pending the explicit-current-encoding contract; metadata attachment is useful, but never pixel conversion |
| ConvertError::Buffer, NeedsCms, HdrSourceRequiresPeak | Keep; precise failures replacing panics/implicit behavior |
| HdrConfig and peak/config plan builders | Already feature-gated; retain names, validate semantics |
| PixelBufferHdrConvertExt | Already feature-gated; fix allocation/anchor/behavior issues through the planned conversion work |
| CllMeasure, LightLevelMethod, histogram and measurement operations | Already feature-gated; preserve distinct algorithms and real users |
| Bt2446A, SoftCompress, GamutBoundaryLut | Already feature-gated; no evidence they escaped a requested removal in #63 |
| ContentLightLevel::DEFAULT_PERCENTILE | Doc-hidden new policy constant used by measure_robust; not an accidental ungated estimator API and no urgent rename needed |
| Eq additions for DiffuseWhite/ColorContext | Not removal candidates; validate the unchecked white constructor through the separate HDR contract work |

One new-builder concern needs explicit attention: `from_icc(...).with_cicp(...)`
can recreate the ambiguity of deprecated `from_icc_and_cicp`. Likewise the
buffer builders can attach competing declarations. Validate/resolve authority
at the planned boundary, or provide explicit replacement attachment operations
before deprecating the ambiguous builders. Do not blanket-deprecate nine useful
attachment methods with no settled destination.

No newly introduced core API merits immediate deprecation solely because of
the missed PR. This does not exempt core from the broader correctness and
ownership migration already proposed.

## Missed cleanup that predates 0.2.16

These are worth discussing in the bridge, but should not be labeled APIs
accidentally introduced in 0.2.16:

| Item | What shipped / recommended treatment |
|---|---|
| RowConverter::convert_rows | Present in 0.2.14. #63's deprecation and replacement both missed main. Deprecate with the final prepared-slice replacement |
| adapt::convert_buffer | Present in 0.2.14. Deprecate with a final stride-aware migration; avoid forcing another temporary API |
| Free convert_row | Present in 0.2.14. Do not blindly restore the allocating associated-method destination from #63; use prepared execution |
| PixelBuffer::into_vec | Present in 0.2.14. Deprecate when parts/packed export ship, including the directly accessible parts record requested now |
| Free orientation functions | Present in 0.2.14. Keep per current design; unnecessary method aliases are not obligatory recovery work |
| pipeline module/feature | Already published in 0.2.14; separate cleanup decision, not part of accidental estimation exposure |
| planar::Plane and REC2020_V4 | Already published in 0.2.14; separate actionable deprecations if removing |
| DiffuseWhite::new | Already published in 0.2.14; deprecate together with the checked constructor, not as a new 0.2.16 mistake |

Already deprecated in the published 0.2.16 sources: Adapted and the three old
adapters, ContentLightLevel::measure, and the naive Reinhard/exposure helpers.
Their `since = "0.2.15"` annotations reflect the unpublished development
version for some items; first published warning was 0.2.16. Do not interpret
that string as evidence that 0.2.15 existed on crates.io.

No public demotion remains for registry or ZenCmsLite: their containing modules
were already `pub(crate)` in the actual 0.2.16 source. Old queue entries and
doc-hidden snapshot sections do not establish public reachability.

## Corrections needed in release documentation and verification

- Correct the claim at CHANGELOG's “Complete public-API removal inventory” that
  estimation and requires_cms became private before publication. Both shipped.
- Mark the absent replacement methods as future work, not shipped migration
  destinations. New parts design supersedes the old tuple shape.
- Record first-published deprecations accurately, distinguishing unpublished
  0.2.15 annotations from actual 0.2.16 availability.
- The historical “zero breaking changes” statement is not proof of compatibility
  for every feature. The diff removes serde trait implementations while retaining
  an empty feature name. A user requiring Serialize/Deserialize can still break;
  no known callers is not a proof otherwise. Do not repeat that policy for the
  new bridge guarantee. This is separate from what to deprecate now.
- Do not import #63's trait sealing or required-method additions blindly. Our
  agreed source-compatibility contract is stricter than that old PR's tolerated
  breaks policy.

Small first PR recommendation: **restore the intended experimental opt-in and
close the requires_cms/Adapted warning gaps**, preserving runtime behavior. Test
default/minimal builds, feature-on builds and downstream warning coverage at
both root/module imports and inferred method calls. For conditional warnings,
an all-features build alone proves nothing: it suppresses the very warning being
tested. Consumers must explicitly select/forward the feature rather than rely
on accidental dependency feature unification.

In 0.3.1 remove the deprecated helper/wrapper surface and require the opt-in for
estimation. Keep opted-in signatures the same. Larger plan/worker and color
redesign remains a separate implementation effort, as described in the
[contract proposal](api-contract-proposal-0.2-and-0.3.1.md).
