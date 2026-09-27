# Release checklist: 0.2.17 and 0.3.1

2026-09-27. Use the [status ledger](implementation-status-0.2-and-0.3.md) for exact
scope, the [code guide](implemented-bridge-contracts.md) for migration examples,
and the [validation record](bridge-validation-2026-09-27.md) for evidence.
This replaces the older speculative execution checklist.

- [x] Merge #75; refuse unsupported narrow-range U8/U16 conversions.
- [x] Implement storage/typed-layout, ownership, color, alpha and output fixes.
- [x] Implement prepared/fallible rows and explicit preserved-stage composition.
- [x] Preserve original CMS/source errors and independent worker state.
- [x] Implement explicit HDR measurement/anchor contracts and fused U16 analysis.
- [x] Commit companion streaming and planar-mask migrations.
- [x] Stage 0.2.17 with warnings and migration paths; retain a maintenance branch.
- [x] Stage 0.3.1 removing selected warned APIs, preserving migrated signatures,
  open traits and feature spellings; require the existing estimation opt-in.
- [x] Run workspace/default/all-feature tests, strict Clippy, rustdoc and MSRV checks.
- [x] Verify packages, snapshots, 128 warning/removal probes and semver deltas.
- [x] Test identical consumer source against all four packaged pairings, including
  real local companion boundaries, fresh resolution and an upgraded lockfile.
- [x] Verify one core version per tested connected pipeline.
- [x] Record allocation and cold/warm/edited compile-cost measurements.
- [ ] Publish the bridge core, then converter, after release review.
- [ ] Land/publish companion migrations and test full codec feature graphs as
  each controlled pipeline opts in. A registry dependency still pinned to ^0.2
  can create a second core even if the top-level application selects 0.3.
- [ ] Publish 0.3.1 core, then converter, after coordinated consumer validation.

No registry publication was performed. Native video carriers, a new output-plan
wrapper family and narrowing-kernel selection remain separate integration and
measurement work as described in the status ledger; they are not silently added
to either candidate. Retained legacy APIs with incomplete ecosystem migrations
are not removed merely because an older proposal listed them.
