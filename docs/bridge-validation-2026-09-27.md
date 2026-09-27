# Bridge validation, 2026-09-27

Source: main after #75 (`197a38b`), storage `373f600`, conversion `3340940`, plus
its API-compatibility follow-up. Both crates are staged at 0.2.17; the converter core floor is 0.2.17 because it uses the new descriptor validation. No package was published.

- Workspace all-feature tests and strict all-feature Clippy pass.
- 27 distinct corrected review cases pass in default/CMS and minimal builds.
- 7,744 format/transfer pairs are checked; every accepted built-in plan prepares
  and executes. This is an execution-safety matrix, not a numerical oracle for
  every possible color pair; dedicated numerical regressions run separately.
- Prepared execution counts zero allocations for repeated rows, independent CMS
  workers and composed HDR stages with different gamut tables. Explicit capacity
  and row geometry errors precede writes; backend failures may partially write.
- 128 downstream deprecation probes pass, including `into_vec`.
- Rust 1.85 core / Rust 1.89 minimal converter checks pass.
- Minimal WASM and pure-Rust ARM (with HDR) cross-checks pass. The benchmark-only
  LCMS feature's full ARM build needs `aarch64-linux-gnu-gcc`; no ARM runtime
  throughput claim is made.
- Public API snapshots are regenerated with `nightly-2026-07-13`; rustdoc with
  broken links denied passes.

## Semver audit

```sh
cargo semver-checks -p zenpixels -p zenpixels-convert --baseline-rev 197a38b
```

Core: 196 checks pass. Converter: 195 pass, one check flags `ConvertError` losing
`UnwindSafe` and `RefUnwindSafe` when it retains arbitrary original backend errors.
This is documented under the repository's tolerated mechanical auto-trait-loss
policy, not represented as a fully clean semver check. Error transport remains
`Send + Sync`; no extra bounds are imposed on external CMS implementations.

`RowConverter` retains std `Sync`: independent workers use exclusive
`Mutex::get_mut`, without locking or shared mutable execution. The no_std worker
owns its backend directly and refuses cloning when it cannot duplicate state.

Audit command: intersect Rust files matching `catch_unwind|RefUnwindSafe|UnwindSafe`
with those matching `zenpixels_convert|ConvertError|RowConverter`, excluding build,
vendor and `.claude` trees. 1,306 unwind candidates, 90 overlapping files including
historical copies. Primary zenavif negotiation tests and zensim picker binaries
already wrap their execution closures in `AssertUnwindSafe`; no direct marker bound
on `ConvertError` was found. This textual audit is not proof about unknown callers.

## Companion work

- zencodec: `ab12c4c`, `3ab2afb`, branch `fix/fallible-pull-source-errors`.
  Workspace tests pass; original pull-source errors survive; current color picks
  one authoritative field without a deprecated ambiguous fallback.
- zenpipe: `a978c10`, `f774d01`, branch `fix/zenpixels-bridge-contracts`.
  98 zenpipe and 505 zenfilters tests pass; callback production/EOF is corrected,
  scratch is reused, and public filter masks no longer depend on retired planar.

Unrelated pre-existing edits in those repositories are preserved. These are local
commits; paired release-candidate builds now pass as described below. Coordinated publication remains.

## Compile cost

Fresh isolated build directories, identical rustc 1.98.1, four Cargo jobs,
offline/locked dependencies. Baseline `197a38b`, current `c9fea15`. A single local
observation on Ryzen 9 9950X3D; no statistical speed claim or CPU isolation.
The converter's cold check follows the core check and may reuse its dependencies
in both runs. Warm means no source changes; edited adds a comment to crate lib.rs.

| Check | Baseline | Current |
|---|---:|---:|
| Minimal core, cold | 1.515 s | 1.439 s |
| Minimal core, warm | 0.022 s | 0.022 s |
| Minimal core, source invalidated | 0.083 s | 0.080 s |
| Default converter, cold | 3.551 s | 3.656 s |
| Default converter, warm | 0.024 s | 0.025 s |
| Default converter, source invalidated | 0.300 s | 0.312 s |

No new dependency was added. The script uses git archives, not worktrees, and
cleans only its own temporary build directories:

```sh
python3 scripts/check-compile-cost.py 197a38b c9fea15 --out /tmp/compile-cost.json
```

Raw results: [compile-cost JSON](../benchmarks/bridge-compile-cost-2026-09-27.json).

## Packaging

`cargo package -p zenpixels -p zenpixels-convert --allow-dirty` builds and verifies
both 0.2.17 archives, including the converter against Cargo’s staged core registry.
Package warnings only note intentionally excluded integration-test files.


## 0.3.1 and packaged compatibility

`release/0.3.1` at `5e40423` builds on bridge `835612e`. Both 0.3.1 package
archives verify. Default/all-feature tests, strict default/all-feature Clippy,
broken-link rustdoc, core 1.85 / converter 1.89 and 128 removal probes pass.
The 0.3 API snapshots were regenerated. No package was published.

Force a patch comparison so the audit actually lists breaks rather than skipping
checks because 0.3 is already a semver-major increment:

```sh
cargo semver-checks -p zenpixels -p zenpixels-convert --baseline-rev 835612e --release-type patch
cargo semver-checks -p zenpixels-convert --baseline-rev 835612e --release-type patch --default-features
```

Core reports only legacy planar, `into_vec` and the ambiguous current-context
constructor removal (219 pass, four removal categories). Converter reports only
`requires_cms` and packed adapters/Adapted (221 pass, two categories). Its default
configuration additionally reports estimation module/types/methods moving behind
`estimation-experimental`. No additional signature/trait changes were reported.

```sh
CARGO_TARGET_DIR=/tmp/zenpixels-target-status-20260927 python3 scripts/check-bridge-compat.py \
  --packages /tmp/zenpixels-target-status-20260927/package \
  --siblings /home/lilith/work/zen
```

All 16 packaged same-source feature/pairing cases pass, plus a bridge→0.3.1
lockfile update. The fixture denies deprecations, crosses actual buffer/descriptor
boundaries and implements the open backend traits. It tests core/converter
0.2.17/0.3.1 in all four combinations. Each selected graph has exactly one core
and one converter. Artifacts are normalized Cargo packages, not workspace path
manifests, so the converter's published dependency requirement is exercised.

Companions are local zencodec, zenpipe and zenresize working trees. The latter
already accepts both core lines. Registry zencodec and zenresize pulled a second
0.2 core until these migration sources were patched into the test: a concrete
confirmation that the entire connected pipeline must opt in together. This is
not an all-codec-feature or all-published-reverse-dependency test.

The zencodec testkit example was migrated to the stride-aware cow adapter in
`f4d5da4`; all 11 usage examples pass against the 0.3 candidates. Previously
reported companion workspace tests remain valid for the earlier functional fixes.
