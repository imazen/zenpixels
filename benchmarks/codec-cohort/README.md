# Gray-codec cohort resolution, 2026-11-06

The four isolated consumer graphs fail dependency resolution before compilation
with core `d037d510104406836ad9e821dbc37398b5fab72a`, pixels
`b6b12bbe702e7ca2f93db559858ae1f233a066ec`, garb `58ce9cad` and linear-srgb
`52896f4b`. No dependency floors were changed. The crates.io API reports
archmage/magetypes 0.9.30 as the latest stable; 0.9.31-beta.2 is available.

| Consumer source | Conflicting requirement |
|---|---|
| zenjpeg `a8ebcd0b` | native archmage >=0.9.30 versus cohort >=0.9.31-beta.2 |
| zenavif `84a4a505` | native archmage >=0.9.29 versus cohort >=0.9.31-beta.2 |
| zenpng `a26cdc64` | native magetypes >=0.9.30 selects stable archmage versus cohort beta.2 |
| zenjxl `cc2ec685` | pinned jxl-encoder SIMD permits <=0.9.31-beta.1 versus cohort beta.2 |

All native ranges above also have an upper bound below 0.10.0 except the JXL
pinned backend's older cap. Rust semver does not admit a prerelease merely
because its eventual stable version fits a stable lower bound. Cargo reported
exit 101 for all four graphs. This is not codec pixel validation.

The gray lowering migrations and PNG draft PR 54 remain unqualified against
this cohort. TIFF's selected no-JPEG path is separately qualified in zenextras
PR 57; its complete workspace has the same native dependency constraint.
AVIF/HEIC consumers have not yet adopted the new shared depth preference helper.
The source pin bump in zenpipe also waits for the shared core to land as directed.

## Reproduce

Export the two exact revisions with `git archive`, recording each revision in
`.source-revision` at the export root. Pass the native checkout, a new output
directory, the core export and the pixels export to `prepare.py`. It preserves
native dependency patches, rebases their relative paths, and replaces only the
shared cohort. The generated manifest is an independent workspace. Run:

```sh
python3 prepare.py /path/to/native /path/to/new-probe /path/to/core /path/to/pixels
cargo metadata --format-version 1 --manifest-path /path/to/new-probe/Cargo.toml
```

Use the resource wrapper and an isolated target directory for Cargo commands.
Each probe records the manifest hashes, Git HEAD and working-copy status; the
package manifests, rather than any uncommitted codec implementation, determine
this resolver result. No source checkout is synchronized or edited by the script.
