# zenpixels-convert-e2e

End-to-end test harness for the `zenpixels-convert` HDR pipeline. Lives as a separate workspace member so the main `zenpixels-convert` `Cargo.toml` stays free of test-only optional deps (`zenjpeg`, `anyhow`). Never published.

Requires the `imazen-26` gain-map corpus at `/home/lilith/work/codec-corpus/imazen-26/`. The test silently skips at runtime (prints `SKIP`) when the corpus is absent — so CI runners without the corpus pass cleanly — but it MUST be run on `lilith`'s machine before merging any pipeline-touching change. Run via:

```sh
cargo test -p zenpixels-convert-e2e
```
