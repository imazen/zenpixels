# Remaining work: 0.2 bridge and 0.3.1

Start with the [consolidated review](zenpixels-0.2-and-0.3-review.md) for all
proposals, status, API cleanup and decisions in one document.

Status: 2026-09-27. This is the short execution checklist; the
[contract proposal](api-contract-proposal-0.2-and-0.3.1.md) holds the detailed
design and the [migration cards](migration-examples-and-audit-0.3.1.md) hold
examples/callers. Commit each completed chunk before moving on.
The [PR #63 inventory](pr63-commit-inventory.md) records all 21 commits and which
missing implementations/tests to reuse, rewrite or skip.
The [final-bridge and sample-signaling proposal](final-0.2-and-0.3-sample-signaling.md)
adds precise 10/12-bit mandates and release gates. These are proposed requirements,
not implementation completed by this documentation change.

## Completed on main, not yet released

- [x] Deprecate `requires_cms` and explicitly deprecate `Adapted::as_pixel_slice`.
- [x] Add `estimation-experimental`; retain the old API with warnings when off.
- [x] Check downstream warning behavior, including inferred receivers, in CI.
- [x] Make `DiffuseWhite::new` panic for invalid anchors; retain its const signature.
- [x] Add `PixelBuffer::into_contiguous`, retaining metadata and allocation.
- [x] Add destructurable `PixelBufferParts` / `into_parts`, moving ownership.
- [x] Annotate the README, audit existing consumers and assess concrete YUV callers.

- [x] Checked `try_from_parts` adoption with `take_parts` / `without_buffer` error recovery.
- [x] Deprecate the complete legacy planar module; defer its video replacement.
- [x] Add a [code-first review](code-review-0.2-and-0.3.md) with executable defect cases.

Compaction preserves the pixel offset for alignment. It removes row padding,
not the alignment prefix. Checked adoption is implemented, including minimum final-row extents.

## 0.2.x: build the common API in reviewable chunks

| Order | Chunk | Completion condition |
|---|---|---|
| 0 — immediate fix | Known-transfer adapter guard | Port both missing `Unknown` guards from PR #63's final revision. Known sRGB → linear requests must convert or fail, never borrow unchanged bytes under a new descriptor. Cover intent, explicit-policy and legacy paths. |
| 1 — remaining ownership work | Allocation reuse | Adoption/error recovery and round-trip tests are complete. Adapt PR #63's typed U8 export and owned-cow adapter optimizations to avoid unnecessary full-image copies. Settle construction for external decoder-owned allocations without adding redundant constructor families. |
| 2 | Storage and typed-layout correctness | Repair minimal final-row extents, zero-area behavior, arithmetic and typed reinterpretation/mutation. Validate descriptor combinations at acceptance boundaries while retaining convenient public descriptors. Add replacement paths before deprecating problematic existing ones. |
| 3 | Current color and CMS inputs | One interpretation of descriptor, ICC/CICP, range, alpha and luminance anchor; no known-color retag masquerading as conversion. Pass actual source/target profiles to CMS and finalization. Resolve constructor/authority choices before publishing new types. |
| 4 | Prepared, fallible conversion | Complete plans preserve composed operations; preparation owns scratch and backend state; execution within capacity does not allocate. Backend errors propagate, and independent workers do not hide shared mutable state behind cloning. |
| 5 | Preservation, HDR and output ownership | Enforce opaque-alpha/preservation policies, refuse unsupported HDR mapping, and emit metadata matching resulting pixels. Identity paths borrow or move existing storage; caller-output paths reuse provided storage. |
| 6 | Migration and release checks | Add actionable warnings only after final replacements exist; migrate real callers; test the same consumer source against both release candidates. Finish accurate README/docs.rs examples, semver review, packaging, features and MSRV checks. |

These are proposed implementation chunks, not blanket approval of every draft
signature. Choose the smallest concrete interface within each chunk together.
Correctness fixes and additions may ship in multiple 0.2 patches; the minimum
compatible version is the patch that actually completes the common surface.
Do not promise that 0.2.17 is the full bridge merely because it is next numerically.
Designate the completed patch as the final API bridge; subsequent new API work
targets 0.3 while the 0.2 maintenance line remains available.

**Deferred video design:** the complete old planar module now warns. Its feature
and code remain available in 0.2. Prototype the media API outside zencodec, using
SVT/AOM/VMAF callers and explicit CVVDP RGB conversion. Zenfilters currently uses
PlaneMask, including public access fields. Relocate that filter mask and migrate
its consumers before removing the deprecated types in 0.3.1.

Additional final-bridge checks from the sample-signaling proposal:

- [ ] Document the legacy U16 numerical domain and prohibit implicit native
  10/12-bit imports through ordinary RGB/gray views.
- [ ] Fix or explicitly refuse unsupported narrow-range depth changes; the
  current full-scale approximation does not preserve the specified anchors.
- [ ] Complete the zenfilters mask migration and test its public boundaries.
- [ ] Include `planar` forwarding in every admitted core/convert version pairing.

**Deferred conveniences:** no duplicate context getter, contiguous-view
constructor just to infer stride, new universal streaming-provider trait, or
bytes-only ownership export. Row iteration can wait for a useful adopter.

## 0.3.1: retire the warned-about legacy surface

- [ ] Branch from the completed bridge; retain a 0.2 maintenance line.
- [ ] Remove only APIs with a complete, tested bridge migration. Immediate examples
  are `requires_cms` and the `Adapted` compatibility family; old conversion/CMS
  interfaces need their replacement work above before removal.
- [ ] Require `estimation-experimental` for estimation, retaining opted-in signatures.
- [ ] Remove migrated legacy planar types/re-exports; keep the accepted `planar`
  Cargo feature as a documented no-op/forwarding stub through 0.3.x.
- [ ] Implement the checked sample-encoding vocabulary with a concrete adapter
  and satisfy [S1–S4 and the acceptance gates](final-0.2-and-0.3-sample-signaling.md).
  Preserve current image U16 semantics; do not require the complete media prototype.
- [ ] Keep the migrated API identical: no surprise descriptor field privacy,
  new trait restrictions, signature/default changes or removed feature spellings.
- [ ] Run identical downstream fixtures on the minimum bridge and 0.3.1 with
  deprecations denied. Exercise actual public buffer boundaries, trait impls,
  fresh/upgraded dependency resolution and permitted core/convert pairings.
- [ ] Confirm one core version per connected pipeline. Widen controlled dependency
  requirements only after their migrations are tested.
- [ ] Audit every semver delta, finalize docs/changelog, verify packaged artifacts
  and release 0.3.1. The yanked 0.3.0 is not the migration target.

Most design and correctness work belongs in the 0.2 bridge. The 0.3.1 change
should primarily remove already-deprecated alternatives, not introduce another
migration destination.

## Verification already completed

The prior deprecation batch passed workspace tests and 44 external warning
checks. The latest core additions passed default, minimal and all-feature tests,
strict core/all-feature and workspace Clippy, Rust 1.85 minimal compilation and
pinned API snapshot generation. The optional HDR-enabled strict Clippy check has
existing `chunks_exact_to_as_chunks` failures in unchanged HDR code; address them
before claiming that configuration is clean. Cross-version consumer testing,
release semver/package checks and sibling migrations are still outstanding.
