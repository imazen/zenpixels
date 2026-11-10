# Frozen downstream compatibility cohort

Run through the normal resource wrapper:

```
run-heavy --mem 6G --jobs 2 -- just downstream-cohort-check /path/to/zencodec /fresh/run/directory
```

Requires Python 3.12+, Cargo and clean committed source checkouts. The output
must not exist. The script archives the selected commits, rewrites only patch
tables in those snapshots, and preserves source checkouts. Both pixel crates and
their numerical dependencies come from one cohort. The standalone render crate
gets its own patch table; Cargo does not inherit workspace/dependency patches.

Each workspace has a retained dependency graph and resolved lockfile. Admission
requires exactly one copy of each selected pixel crate at the expected path.
All targets compile with `cargo check`; only libraries/integration tests execute,
so harness-free benchmark targets are never run. Every command and outcome is
recorded in receipt.json, with separate diagnostic logs and metadata JSON.
Failures stop the cohort and preserve all evidence. This is native compatibility
qualification, not a performance or release-readiness claim.

## First qualification — 2026-11-09

| Input | Revision / configuration |
|---|---|
| Codec source | zencodec `1fb997bb88e7405e6a315bf1f0b22f71e99be272` |
| Pixel snapshot | `e3fe0fd7b51bfdf853d2fdbf02ce41b3c21f8da5` |
| Merged pixel implementation | Both pixel crate directories are byte-for-byte identical to main `c4c18b44ec4cdd2d6b8e10a7bb3f2e6f9e39b3e3`; snapshot adds only this harness/recipe |
| Codec workspace | All-target check; library/integration tests with `std` |
| Standalone render workspace | All-target check; library/integration tests with default features |
| Result | Both workspaces passed; dependency graphs verified the selected pixel source |

This closes one reproduction gap in issue 143, not the whole issue. Consumer
manifests still retain their older pins until their migration is reviewed.
Other consumers, optional features/platforms, MSRV and actual codec image
qualification remain separate. A passing compile does not prove color fidelity.
