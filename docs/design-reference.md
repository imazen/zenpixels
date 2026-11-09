# Pixel and conversion design reference

This is the reference for the authorized 0.3 breaking-release consolidation.
Status below distinguishes the intended design from completed implementation.

## What callers need to know

1. **Storage is not interpretation.** `SampleEncoding` describes one integer
   payload per word, including signedness, precision and placement. A plane
   supplies geometry. Color, range and component roles come from the enclosing
   signal. RGB565 and other multi-component packed words require unpacking.
2. **Representation is not processing support.** Exact integer carriage can
   support more types than numerical conversion. Unsupported arithmetic refuses
   before writing; it never guesses a normalization.
3. **Conversion resolves meaning once.** A prepared conversion owns its operation
   chain and output interpretation. Allocating and caller-buffer APIs execute
   the same conversion and attach the same metadata.
4. **Policy survives layout changes.** Swizzling cannot introduce defaults for
   alpha, luma, HDR rendering or clipping. Callers must provide authoritative
   metadata or explicit choices for conversions needing known color.
5. **Untouched components stay untouched.** Alpha payloads and row padding are
   preserved bitwise when the operation does not change them.

## Internal invariants

- One encoding model; no compatibility alias for the unshipped NativeEncoding.
- Geometry validation and row access reuse Plane, including empty extents.
- Each finalized operation owns its input/output descriptors and scale. No
  separately mutable vector of descriptors accompanies an operation list.
- Output context derives from the resolved conversion, not wrapper-specific
  reconstruction. Pixel descriptor, white anchor and current profile agree.
- Constructors enforce the same descriptor validity rules. Fallible constructors
  return `DescriptorError`, displaying the supplied sample type, layout, alpha
  interpretation and violated rule. Buffer validation preserves that diagnostic.
- Error closures construct raw errors. Wrap them with `At::wrap`, attach crate
  metadata with `at_crate`, and call `.at()` outside the closure so tracing
  preserves the calling function’s `track_caller` chain. `at_crate` alone does
  not add the first location frame.
- Generic layout operations do not assume a maximum sixteen-byte pixel.
- ICC identification does not grant permission to replace a profile. Empirical
  rejection cannot be overridden by a structural matrix-shaper classification.
- Fallible stateful-worker cloning stays fallible; unused profile machinery is
  removed rather than exposed for hypothetical callers.

## Numerical ownership

Primitive SIMD/scalar convergence belongs to the separate magetypes work.
This change does not replace powf or promise cross-tier bit equality before
that work lands. Local algorithm differences, discarded options, alpha-bit
rewrites and invalid percentile ranks remain this crate's responsibility.
Tests must distinguish those defects from primitive rounding differences.

## Implementation ledger

| Work | Status | Regression gate |
|---|---|---|
| Unified integer encoding and shared native geometry | Implemented | Native signed/wide carriage, code mapping, empty/strided planes |
| Finalized operation chain and policy preservation | Implemented | Compositing/swizzle combinations, prepared and partitioned execution |
| Unified output context | Implemented | Allocating/into/adaptation pixels and white-anchor parity |
| Descriptor and generic-layout invariants | Implemented | Invalid alpha descriptors, U64 channel removal, empty row swaps |
| Payload and histogram algorithm corrections | Implemented | F16 alpha bits, positive percentile ranks; primitive parity separate |
| ICC qualification and API cleanup | Implemented; historical table requalification remains separate | Parametric curve types, failed qualification, fallible clone callers |
| Final release-candidate validation | Local gates passed; remote CI tracked on PR145 | Workspace tests, feature gates, Clippy, API snapshots and CI |

Do not infer completion from an API name or a passing unrelated test. Update
this ledger with the implementation and link specialized detail rather than
copying these rules into several documents. No publication is authorized here.

## ICC qualification limit

The generator now requires empirical evidence, checks all RGB curves, and refuses
truncated/unsupported curves. Existing recognition tables retain historical
qualification flags; this change does not retroactively requalify their corpus.
Their error column is a curve-fit error, not a complete-transform error bound.
Bundle-derived rows remain identification-only. A sampled CMS gate is not proof
of global bit-exact substitution. Do not cite it as one.

## Migration

- Replace `NativeEncoding` with `sample::SampleEncoding`. The constructor keeps
  the storage/bits/shift arguments; its error is `SampleEncodingError`.
- Replace `RowConverter::clone()` with `try_clone()?`. A stateful CMS worker can
  refuse cloning; callers can prepare another worker explicitly.
- `PixelDescriptor::try_new` and `try_new_full` return `Result<Self,
  descriptor::DescriptorError>` instead of `Option<Self>`; use `?` or inspect
  the displayed reason.
- Invalid descriptor alpha/layout combinations now fail consistently in fallible
  constructors and panic consistently in infallible constructors.
- No new feature flags or public planning types are needed. `PlannedOperation`
  and output-interpretation resolution are private implementation details.

## Regression commands

Run `just sample-design-check`, `just plan-design-check`,
`just context-design-check` and `just local-numerics-check` for focused gates.
`just approved-contracts-check` covers the workspace and existing feature gates;
`just api-doc` refreshes the public surface snapshots. Resource wrappers and
platform admission are the caller's responsibility.

## Why two views remain

`CodePlane` reads one component from an existing typed U8/U16 plane, including
interleaved chroma. `NativeCodeSlice` adapts packed RGB/gray byte rows whose
components can have different payload widths, including signed/wide words.
They share `SampleEncoding`; the packed adapter stores a borrowed byte `Plane`
for geometry. Neither promises that every represented encoding has a numerical
conversion kernel. There is no duplicate encoding type or compatibility alias.

## Validation receipt — 2026-11-09

`approved-contracts-check` passed workspace release library/integration/doc
checks, existing focused feature checks, no-default native carriage and strict
Clippy. `design-feature-check` additionally passed the all-feature library,
new design regressions, histogram tests, generator qualification tests and
strict all-target/all-feature Clippy across all workspace crates. Existing
expectations were not weakened. The positive-percentile correction retains
floor ranks and only excludes rank zero; the existing outlier gates pass.
Public API snapshots were regenerated. Cross-platform CI is tracked in
[PR145](https://github.com/imazen/zenpixels/pull/145); local results are not a
claim that remote CI or historical release branches have been qualified.

## Profile assets and retention

Removing the private, unused `builtin_profiles` implementation was intentional.
It did not supply the exported ICC assets: `icc_profiles` and its bundled profile
bytes remain available. The removal does not remove named-profile export.

Lowest-priority follow-up: retain the original bytes of every profile represented
by a recognition-table hash in private R2 storage or a private repository, with
license/provenance records. Keep a cryptographic digest of the complete original
file alongside its normalized recognition hash; normalization must not erase the
identity needed to reproduce qualification. Record the generator revision, CMS
backend, sample domain and tolerance used for each qualification run.

Before requalifying historical rows, inventory table-to-asset coverage and report
missing originals explicitly. Do not substitute a similarly named profile or
publish restricted bytes. This retention inventory and historical revalidation
have not been completed by the current code changes.

`NativeCodeSlice` is an interleaved-byte adapter, not another storage owner or
encoding model. It supplies the component boundaries needed by `native_to_u16`
and `u16_to_native`; `CodePlane` supplies one component from typed U8/U16 storage.
A separate public type is not mathematically necessary, but replacing it with a
single universal view would need to carry both sets of layout rules. Retain the
small adapter over shared `Plane` geometry for the current native conversion API;
do not grow it into a second plane framework.

Descriptor/trace follow-up validation: `just approved-contracts-check`,
`just design-feature-check` and `just api-doc` passed. The overflow regression
checks the exact public callsite, not only the error text.
