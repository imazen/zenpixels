# Paired release source qualification, 2026-10-30

Frozen source: `33682afcfbc4a0192aa8143c0d6cb5334bcb0275`.
README-only follow-up: `4b92f3f9`; library sources and manifests are unchanged.
The source export was outside workspace path overrides, with its own resolved
lock and a shared compilation target. All commands below completed successfully:

```sh
cargo generate-lockfile
cargo test --locked -p zenpixels -p zenpixels-convert --all-targets
cargo test --locked -p zenpixels -p zenpixels-convert --doc
cargo test --locked -p zenpixels -p zenpixels-convert --all-features
cargo clippy --locked -p zenpixels -p zenpixels-convert --all-features --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked -p zenpixels -p zenpixels-convert --all-features --no-deps
cargo package -p zenpixels --allow-dirty --no-verify --exclude-lockfile
cargo package -p zenpixels-convert --allow-dirty --no-verify --exclude-lockfile
```

The serialized wrapper used 3 GiB / two jobs and a 1200-second timeout:
`rc=0 910s | peak-RSS 2.64GiB | min-avail 11021MiB | peak-load 2.34`.
This is validation-process memory, not a codec workspace bound. The all-target
command executed harness-free benchmark binaries; their timing output is not a
qualified performance comparison. Existing ignored tests remained ignored.

Package creation used `--no-verify`: this does not establish that unpublished
paired dependencies can resolve from the registry, nor qualify the extracted
packages. Publication, packaged-pair checks and required platform CI remain
separate gates. No crate was published by this run.

## Actual package checks

The two `.crate` files produced above were extracted outside the workspace.
For each extracted manifest, these commands passed:

```sh
cargo --config 'patch.crates-io.zenpixels.path="/path/to/extracted/zenpixels-0.3.1"' check --all-features --lib
cargo --config 'patch.crates-io.zenpixels.path="/path/to/extracted/zenpixels-0.3.1"' test --locked --doc
cargo --config 'patch.crates-io.zenpixels.path="/path/to/extracted/zenpixels-0.3.1"' check --locked --no-default-features --lib --target wasm32-unknown-unknown
```

Resolved metadata contains exactly one selected `zenpixels`, version 0.3.1,
from the extracted package. The converter graph also selects the extracted
0.3.1 converter. No checkout path overrides were inherited.
Wrapper: `rc=0 7s | peak-RSS 0.31GiB | min-avail 13384MiB | peak-load 1.27`.
This qualifies these extracted library/doctest configurations. It does not
cover the older bridge pairings, replace platform CI, or prove unpatched
registry availability. Source test suites remain covered by the separate run.
