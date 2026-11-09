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
   alpha, luma, HDR rendering or clipping. Unknown color stays unknown unless
   authoritative metadata or an explicit caller choice resolves it.
5. **Untouched components stay untouched.** Alpha payloads and row padding are
   preserved bitwise when the operation does not change them.

## Internal invariants

- One encoding model; no compatibility alias for the unshipped NativeEncoding.
- Geometry validation and row access reuse Plane, including empty extents.
- Each finalized operation owns its input/output descriptors and scale. No
  separately mutable vector of descriptors accompanies an operation list.
- Output context derives from the resolved conversion, not wrapper-specific
  reconstruction. Pixel descriptor, white anchor and current profile agree.
- Constructors enforce the same descriptor validity rules.
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
| Unified output context | Pending | Allocating/into/adaptation pixels and white-anchor parity |
| Descriptor and generic-layout invariants | Implemented | Invalid alpha descriptors, U64 channel removal, empty row swaps |
| Payload and histogram algorithm corrections | Pending | F16 alpha bits, positive percentile ranks; primitive parity separate |
| ICC qualification and API cleanup | Pending | Parametric curve types, failed qualification, fallible clone callers |
| Final release-candidate validation | Pending | Workspace tests, feature gates, Clippy, API snapshots and CI |

Do not infer completion from an API name or a passing unrelated test. Update
this ledger with the implementation and link specialized detail rather than
copying these rules into several documents. No publication is authorized here.
