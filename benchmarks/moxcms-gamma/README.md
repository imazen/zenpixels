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

For the upstream review candidates, enable `review-revisions`. The `underflow`
bench and `icount` diagnostic then use the exact Git sources: baseline `84104443`,
sign fix `45b011cd`, and range fix `4ab190f0`, in the before/sign_fix/guarded
arms (before/branch/guarded in `icount`). The always_f_powf comparison remains
the frozen historical c5 arm. This avoids attributing timing of an experimental
source copy to a different proposed patch.

Enable `optimized` for the separate 2026-11-06 experiment. Preparation also
reconstructs three private variants from range fix `4ab190f0` and verifies their
source manifests against `expected-optimized.json`: `opt-or` combines RGB
admission checks and handles zero directly; `opt-pixel` also shares CICP RGB
admission; `opt-wide` is an experimental exponent-scaling control whose outputs
can differ from the range fix. It is not a bit-compatible replacement.

`optimized_check` compares 4,595,836 evaluations per candidate against the range
fix and checks RGB/scalar agreement. `optimized-cells.py` freezes 45 timing
cells with five processes and 30 rounds each, a 1% margin, and 95% family
confidence. Failed cells remain failures. `optimized-evidence.py` collects
full RGB evaluator assembly and row-scoped Callgrind instruction counts.
No timing result is implied by adding these experiments.

The follow-up `outlined` feature adds `opt-cold`: the RGB fallback is a separate
cold function, and the zero shortcut retains exponent-zero behavior. The first
three variants are retained experimental controls; their zero shortcut was
screened only for positive exponents. `outlined-cells.py` declares a new 45-cell
study with original/sign/range/pixel controls and the same admission and
confidence rules. Its binary identity distinguishes it from the first study.

Use `final-review` for candidate `3e098543`: it aliases the `opt_cold` benchmark
arm to the separately frozen `opt-final` source. This preserves exact
negative-zero behavior in the bounded CICP power helper. `final-cells.py` uses
a distinct workload/scope and binary hash; it does not replace either earlier
study. The diagnostic now checks signed zero across all 65,536 nominal ICC
gamma encodings in both directions. Source records distinguish every variant.
