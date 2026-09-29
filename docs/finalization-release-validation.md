# Release reconciliation validation — 2026-09-28

Implementation snapshots: bridge `0d74f5e`; cleanup `aeb69d9`.
Both are pushed, together with the corresponding PR heads #76 and #77. Main
remains `197a38b` (#75). No crate has been published by this work.

## Tests and compatibility

| Check | 0.2.17 bridge | 0.3.1 cleanup |
|---|---|---|
| Workspace all-feature tests including doctests | 1,391 passed; 50 ignored | 1,307 passed; 49 ignored |
| All-target, all-feature Clippy, `-D warnings` | Pass | Pass |
| All-feature rustdoc, `-D warnings` | Pass | Pass |
| Pinned nightly API snapshots | Regenerated | Regenerated |
| Core Rust 1.85 minimal build | Pass | Pass |
| Converter Rust 1.89 minimal + HDR build | Pass | Pass |
| Standalone wasm32-unknown-unknown minimal + HDR check | Pass | Pass |
| Downstream compile probes | 168 warning checks | 184 removal/sealing checks |

The no-std checks use `cargo check -p zenpixels-convert --no-default-features
--features hdr-experimental --target wasm32-unknown-unknown`, without test/dev
feature unification. They establish compilation, not execution on every target.
`libm` was already in the dependency tree; no extra float abstraction dependency
was needed. Core remains `no_std + alloc`, not allocator-free. Arc-backed
ownership still requires pointer atomics; allocator/atomic support is a target
constraint, not something an `alloc` feature switch could eliminate.

Packaged identical-source compatibility: **12 feature/pairing cases passed**,
plus a bridge lockfile upgraded to 0.3.1 without source edits. All four core /
converter pairings (0.2/0.2, 0.2/0.3, 0.3/0.2, 0.3/0.3) are tested with minimal,
default, and interop+HDR+compatibility-feature configurations. Each selected graph
contains one core version and one converter version. This run did not re-certify
every sibling codec feature. The existing companion matrix remains a separate gate.

Archives were created with `cargo package --no-verify`; converter packaging used
an explicit local core patch because 0.2.17/0.3.1 are not published. The compatibility
runner then extracted and built those archives; it did not merely test workspace
path dependencies. An initial run exhausted the nearly-full home filesystem;
its generated cache was moved to `/tmp`, and the complete run passed there.
Validation scripts now default to temporary build-cache storage.

Regression tests cover current ICC versus origin, both output-context exits,
actual moxcms results, premultiplied ICC input, F32 staging, prepared zero-allocation
execution, selected authority, decoded YUV origin signaling, exact hash matching,
changed transforms/intent, unknown-profile ownership, TC7 refusal, and the existing
7,744-pair accepted-plan execution matrix. A separate six-test JavaScript suite
checks the explorer's arithmetic and scenario warnings. Chromium smoke checks
exercise all five presets, P010 values, ambiguity warnings, direct file loading
and a 390-pixel mobile viewport; desktop/mobile screenshots were inspected.

## Semver interpretation

The forced **patch** audit against published 0.2.16 is intentionally not green:
new deprecations are the bridge's purpose. The only major-category failure is
`ConvertError` losing `UnwindSafe`/`RefUnwindSafe` when it preserves arbitrary CMS
backend errors. This is the previously documented tolerated exception; Send,
Sync and Clone remain. The new HDR root aliases are actual deprecated type aliases,
because deprecating a `pub use` did not generate downstream warnings.

The forced patch audit of 0.3 against the bridge reports the intended removed
items/field/method/default and trait sealing. Those are breaking changes and belong
in 0.3.1. In particular, the default implementation of the explicit peak-measurement
method no longer delegates to the removed ambiguous method. No claim of 0.2
source compatibility is made for code ignoring migration warnings.

## Compile cost

Three paired serial observations per candidate, fresh git archives/targets,
locked offline dependencies, rustc 1.98.1, four jobs. Converter “cold” follows the
core check, so some dependencies are already cached. These are local `cargo check`
measurements, not link/build benchmarks or statistical performance guarantees.
One overlapping integration-build observation is retained as excluded in the raw
file and replaced by another serial pair.

| Case | Main median | Candidate median | Difference |
|---|---:|---:|---:|
| Bridge minimal core | 1.361 s | 1.360 s | −1 ms |
| Bridge default converter | 3.258 s | 3.340 s | +82 ms / 2.5% |
| Bridge converter source edit | 0.261 s | 0.281 s | +20 ms |
| Cleanup minimal core | 1.353 s | 1.415 s | +62 ms |
| Cleanup default converter | 3.254 s | 3.291 s | +37 ms / 1.1% |
| Cleanup converter source edit | 0.266 s | 0.273 s | +7 ms |

No new production package dependency. Dashboard code and its tests are static
JS/HTML, outside the Rust graph. [Raw observations](../benchmarks/finalization-release-compile-2026-09-28.json).

## Review stack and remaining integration boundary

#76 is the bridge; #77 is its breaking child. #78/#79 are superseded by the
reconciled finalization and hash-based method; #80's review is included in #76.
Independent #55/#62 remain separate. The old design PR #74 remains historical.

The zenpipe #87 application service is migrated to the method (10 service tests),
and gain-map PNG preparation uses current primaries plus the same luminance anchor
for PQ quantization and explicit MaxCLL measurement. These changes do not certify
the full metadata engine/media extraction described in the architecture review.
No claim is made that every proposed read/write/transcode tier has shipped, or that
the explorer emulates real displays. Complete those integrations before claiming
end-to-end release readiness for all codecs.
