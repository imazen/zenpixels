# moxcms gamma experiments

Independent, unpublished diagnostic crate recovered from the pure-gamma
investigation. Production zenpixels does not depend on it.

Run `python3 prepare.py` once, then `cargo check --locked --all-targets`.
Preparation downloads [moxcms at 84104443](https://github.com/lilith/moxcms/tree/841044439077ec1ec67a71b2db979ab51d304fcf),
applies the six preserved patches, and verifies source hashes against the
original experimental copies. Existing variants are never overwritten.
The patches retain upstream's BSD-3-Clause/Apache-2.0 licensing; downloaded
sources carry the original license files.

The base arm changes only the local package version/workspace. Candidates
c1/c2/c3 compare sign handling (`copysign`, branch, bit insertion); c4 adds the
guarded power evaluator; c5 uses the `f_powf` comparison path. These are
historical experiment arms, not claims about current upstream behavior.

`cargo run --release --bin probe` and `--bin black` print small behavior tables.
`cicp_probe` prints transfer controls. `sweep` and `sweep2` enumerate the F32
bit space; `cicp_bits` allocates a dense input grid. Those are long, heavy jobs.
`gamma` and `underflow` are paired zenbench benches. Run builds, sweeps and
benches under the host's resource wrapper and shared build lock, with full
logs outside git. No target-cpu=native flags are set here.

`ci-lists.sh` and `arm-run.sh` accept a dependency checkout and log directory;
`run-icount.sh` accepts a compiled diagnostic and variant names. Their caller
owns resource limits. Generated dependencies, binaries, benchmark data and
logs stay out of git. Prior measurement claims are not imported as new results.

Validation on recovery: all targets compile; the `probe` and `black` binaries run.
The frozen c5 dependency emits an unused-import warning; its source is kept
byte-identical to the original arm. No exhaustive sweep result is asserted here.

The instruction-count entry point keeps its Rust symbol and disables inlining;
Callgrind selects it by `*run_work*`, without an unsafe exported symbol.
