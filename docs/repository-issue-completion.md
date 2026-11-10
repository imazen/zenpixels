# Repository color and storage issue improvements — 2026-11-09

This records the work for issues 134, 135, 133, 132, 83 and 69 on the 0.3 review
candidate. It does not authorize publication or claim downstream qualification.

## Authority and synthesis — 134

`authority_boundaries` runs the ICC/CICP presence/authority table through both
OutputPlan and EncodeReady. Unknown authoritative ICC is offered to the CMS and
cannot be replaced with conflicting CICP. Unsupported authoritative CICP refuses;
source samples remain unchanged. Existing output_current_color controls compare
real ICC pixels, premultiplication and descriptor/current-context conflicts.

SynthesizedIcc uses primaries/transfer to synthesize a profile of decoded RGB
(or decoded gray). It deliberately does not perform coded matrix/range conversion.
The new public-boundary test proves that coded and RGB signaling select identical
profile bytes while ColorProfileSource::resolve refuses the coded form. These
preconditions are documented on both synthesis methods. Original video signaling
remains provenance, not a replacement description of converted RGB samples.

## HLG — 132

| Entry point | Identity | Crossing HLG transfer |
|---|---|---|
| ConvertPlan / RowConverter | Carries samples | Explicit rendering and white required |
| Slice/buffer extensions | Same plan | Same explicit policy; buffer may measure rendered peak |
| OutputPlan / EncodeReady | Same-profile carriage | Profile-only paths cannot express display rendering; refuse |
| Internal lite CMS | Declines trivial identity | Builds the explicit plan, preserving its refusal |
| Public MoxCms | Declines trivial identity | Now declines identified HLG crossings; a TRC is not the coupled OOTF |
| SynthesizedIcc | No pixel operation | HLG ICC synthesis refuses |

The new direct-MoxCms regression failed before the fix: it built a successful
HLG-to-linear transform without display conditions. It now declines. A private
lite-CMS test protects that entry point too. hlg_rendering retains independent
neutral/colored luminance, black lift, extended values, alpha, partitions and
strided controls. The implemented equations and conditions are in
[HDR anchor design](hdr-anchor-api-design.md). No new universal display preset
or profile representation was introduced.

## Native samples — 83

NativeCodeSlice and its dependent APIs are removed. Existing CodePlane component
readers and to_canonical cover RGB10/alpha1, interleaving and padding. The signed/
wide SampleEncoding description does not grant numerical support. Native pixel
identity/refusal and exact ST2086 controls remain in native_samples/native_codes.
RGB565 is unpack-first. A complete IMIJ adapter remains a consumer task; there
is no demonstrated reason to add another core view now.

## Ownership and allocations — 69

output_allocation_contract measures zero allocated bytes for prepared identity
borrow_rows over packed and padded input and verifies pointer identity. Owned
EncodeReady output has a distinct pixel allocation but identical rows, descriptor
and emitted color metadata. Existing output_native controls cover interpretation
changes and actual conversions. Consumers should configure the encoder from
plan.metadata(), then borrow rows or supply destination rows to convert_rows.
A real caller is zencodec-render render.rs at revision
`1fb997bb88e7405e6a315bf1f0b22f71e99be272`: its color stage calls borrow_rows and
retains the projected image when interpretation is unchanged; its conversion
stage writes rows via convert_rows. This observation is source inspection, not
a new downstream build claim.

Typed extraction controls prove allocation reuse after erasure for RGB U8/U16/F32
with compatible storage layouts, and a copy from incompatible byte storage.
The documentation's alignment-1-only claim was wrong and is corrected. This is
not a claim that every capacity/stride can be extracted without copying. No new
consuming finalizer is warranted by these demonstrated consumers.

## P3 current-byte qualification and release exposure — 135

The fresh native oracle used libultrahdr source
`5f117294cb701170b9198ed8eae16270aa84896c`. Both default and isolated no-default
exports were recognized as DISPLAY_P3 (enum 1); compatibility bytes returned -1;
the native positive control returned 1. Independent Little CMS 2.17 evaluated
4913 RGB points: maximum XYZ error was 1.71659e-05 for P3/sRGB and 0.000135807
for P3/BT.709. These are sampled color errors, not global error guarantees.
Both feature configurations produced identical bytes:

| Profile | SHA-256 |
|---|---|
| P3/sRGB | dc5b04c0d4bff5d4eb145431266fcdf5d6a434f2b30b32fabac88543f9df4cd4 |
| P3/BT.709 | c088d56c2b3c462ee00a1e9d4be1d7832394f432c64341715c91af9a7c04425e |
| Historical compatibility | 231752984cd4a5278e1b8d2390fe496767d4511fc81f54e1a5c69ae9ab4c42b5 |

Release selection must not use the old refs as qualified candidates:
release/0.2 `e96e4e9c` retains the legacy no-db fallback; release/0.3.1 `4d41ed8a`
is compatibility-first. Main `d234f3c6` also retains a legacy no-db fallback.
The reviewed candidate has the separately qualified compact exports. Reconcile
release refs to reviewed fixes before publishing either release line; the old
branch names alone are not evidence of readiness. This change does not overwrite
those historical review refs or publish a release.

## API/docs — 133

Adapter examples now compile and use public methods. Partial-pixel sample-aligned
strides are documented accurately. The ablation document now inventories the
current free operations and canonical migrations. The
[semver audit](public-api/semver-0.2.17-to-candidate.md) uses the explicit 0.2.17
bridge baseline, records checker/tooling limits, and inventories the expected
breaking migrations. Final release packaging still requires removing unpublished
dependency patches and actual downstream compilation under issue 143.

Run `just repository-issues-check`, `just design-feature-check`, `just api-doc`,
and `just p3-export-check SOURCE FRESH_OUTPUT`. The semver recipe forces patch
reporting so approved 0.3 breaks remain visible; it is expected to exit 100.

## Downstream start after PR144/146 merge

Main `c4c18b44` contains both reviewed changes. The
[frozen consumer cohort](../tools/downstream-cohort/README.md) now qualifies
zencodec `1fb997bb` and its standalone render workspace against matching pixel
source: all-target compilation and library/integration tests pass. The run
retains exact source identities, dependency graphs and fresh locks. Consumer
source checkouts and their older manifest pins are unchanged. This is the first
completed cohort for issue 143, not a claim that all consumers or optional
features have migrated.
