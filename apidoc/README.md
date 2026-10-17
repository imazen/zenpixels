CI-free public-API snapshot runner for this workspace. Excluded from the
parent workspace via `[workspace] exclude = ["apidoc", ...]` so plain
`cargo test` and every CI job (including `--all-features` ones) never
compile its `zenutils-apidoc` dependency tree.

Regenerate the committed snapshots under `docs/public-api/` with:

```
just api-doc
```

(which is just `cargo test --manifest-path apidoc/Cargo.toml`). Verify
the on-disk files are current via `just api-doc-check` (sets
`ZEN_API_DOC=check`). Format docs and the snapshot encoding live in the
[`zenutils-apidoc`](https://crates.io/crates/zenutils-apidoc) crate.

A frozen v0.2.14 baseline lives at `docs/public-api/v0.2.14/`; the delta
to `main` is summarised in `docs/public-api/CHANGES_0.2.14_to_main.md`.
