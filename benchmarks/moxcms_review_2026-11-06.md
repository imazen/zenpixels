# Exact moxcms review revisions, 2026-11-06

The five-process ARM study qualified all 36 predeclared workload cells with
unchanged resource admission. The sign fix improved 27/27 applicable comparisons.
The range fix improved 23/36, met the 1% non-inferiority margin in 4/36, regressed
in 5/36, and was inconclusive in 4/36. Three earlier full-suite populations were
rejected and remain retained; none supplies confirmed timing claims.

## Validated ARM timing

Neoverse N1, aarch64-unknown-linux-gnu, Rust 1.99.1 / LLVM 23.1.1, default target
CPU, GCC 15.2.0 linker. All 36 cells were declared before execution, each with five
fresh processes and 30 rounds per process. Each cell uses confidence
0.9986111111111111 (1 - 0.05/36), with its candidate family further adjusted by
zenbench. This conservatively controls the combined study family at 95%.
Every process and recorded sample passed the unchanged admission checks.
CPU frequency was not independently verified.

The following are measured mean microseconds per 1024 RGB pixels. Percentages
and intervals compare each candidate with the original baseline; negative means
less elapsed time. They are process-level log-ratio estimates, so percentages
need not equal the ratio of the displayed rounded arithmetic means.

| Workload | Original µs | Sign fix µs | Range fix µs | Range fix change and adjusted interval |
|---|---:|---:|---:|---|
| Encode, normal positive | 30.641 | 24.615 | 24.941 | -18.60% [-19.43%, -17.76%] |
| Encode, 50% negative | 33.918 | 24.659 | 24.938 | -26.48% [-27.66%, -25.28%] |
| Encode, 10% zero | 30.372 | 24.476 | 29.368 | -3.31% [-4.83%, -1.75%] |
| Encode, all zero | 30.288 | 24.275 | 24.291 | Inconclusive: -20.34% [-55.62%, +42.98%] |
| Decode, normal positive | 30.587 | 24.629 | 24.941 | -18.46% [-19.32%, -17.59%] |
| Decode, all zero | 30.144 | 24.167 | 22.666 | -24.81% [-25.46%, -24.14%] |
| CICP gamma 2.2 to BT.2020 | 76.280 | — | 79.624 | Inconclusive: +4.39% [+0.60%, +8.32%] |
| CICP SMPTE 428 to BT.2020 | 69.475 | — | 69.016 | Non-inferior: -0.66% [-1.15%, -0.17%] |
| CICP gamma 2.8 to BT.2020 | 75.575 | — | 78.908 | Regression: +4.41% [+4.24%, +4.58%] |
| Adobe RGB to ProPhoto, extended | 76.050 | 61.154 | 59.977 | -21.13% [-22.35%, -19.90%] |

The five confirmed range-fix regressions are gamma 2.2 at widths 64 and 4096,
and gamma 2.8 at widths 64, 1024 and 4096. The sign PR does not change the CICP
implementations; those groups intentionally compare only original versus range
fix. All widths, candidates, intervals and verdicts are in
[the complete ARM CSV](moxcms_review_arm_2026-11-06.csv). The always-f_powf arm is an
experimental control, not a proposed PR: 21 regressions and 6 improvements.

This is steady-state evaluator/transform timing. Transform construction and the
range evaluator's new bound computation are excluded and have not been timed.
All-zero original output is incorrect; timing it is a cost comparison, not an
accuracy-equivalent implementation comparison. No new exhaustive 2^32 sweep was
run. No wall-clock claim is qualified on x86 or Zen 5.

ARM binary SHA-256:
`9b8b8e05b779cb0a3d502ace3cc1bbdee85da248a73f57d9fc432b2b5c7d14a5`.
Study wrapper: 242 seconds, peak RSS 0.06 GiB, minimum available 14856 MiB,
peak load 0.87. These are harness resource measurements, not library memory use.

## Sources and build

- Baseline: [84104443](https://github.com/lilith/moxcms/commit/841044439077ec1ec67a71b2db979ab51d304fcf).
- Sign fix: [45b011cd](https://github.com/lilith/moxcms/commit/45b011cdd045351eea689d250df0e7cc55af84dd).
- Range fix: [4ab190f0](https://github.com/lilith/moxcms/commit/4ab190f0e76f95b332a04ce059b51c893e2f5d79).
- Harness: `3565a70b88f1f43c2389f27c77b1703c8c4572e6`, `review-revisions` enabled.
- Instruction/disassembly host: x86_64 Ryzen 9 7900X (Zen 4), Rust 1.99.1, LLVM 23.1.1, release + debug=1;
  default target CPU, no target-cpu=native. Not measured on Zen 5.

## Deterministic instruction counts

Callgrind collects only `run_work`; transform construction, input generation and
allocation are outside that scope. Inputs contain 4096 RGB pixels / 12288 samples.
The evaluator uses the ICC fixed-point exponent 2.19921875 (inverse for encode).
These are executed instructions, not wall-clock predictions.

| Workload | Baseline | Sign fix | Range fix |
|---|---:|---:|---:|
| Encode evaluator, 0% negative | 720937 | 557097 | 663593 |
| Encode evaluator, 50% negative | 770185 | 557097 | 663593 |
| Display P3 to Adobe RGB, F32 | 1529261 | 1358509 | 1465005 |
| sRGB to Adobe RGB, F32 | 1522373 | 1358509 | 1465005 |
| Display P3 to Adobe RGB, U8 table | 94354 | 94354 | 94354 |

All 27 collected cells are in [the CSV](moxcms_review_2026-11-06.csv).
Binary SHA-256: `49c1338206df1f211aba6d37964b12e0b75b2d5f5d704f337226f0e358970669`.

## Object-code disassembly

The complete `ToneCurveEvaluatorPureGamma::evaluate_tristimulus` bodies contain:

| | Baseline | Sign fix | Range fix |
|---|---:|---:|---:|
| Static machine instructions | 188 | 121 | 345 |
| Conditional jumps | 6 | 0 | 8 |
| All jump instructions | 6 | 0 | 10 |
| Call instructions | 0 | 0 | 3 |

Counts include the full range-fix routing body; the out-of-line exact power
implementation is separate. They must not be confused with executed counts.
The sign fix removes negative-base exponent/parity handling after `abs` proves
that the power input is nonnegative. The range fix adds per-channel unsigned
range checks and cold fallback calls. Its fast pixel path retains the approximation.

## Exact changes

PR1's complete diff is one file, 258 inserted and 4 deleted lines (including
tests). The range PR's incremental diff is two files, 287 inserted and 41 deleted
lines (including tests). Exact diffs:
[PR1](https://github.com/lilith/moxcms/compare/841044439077ec1ec67a71b2db979ab51d304fcf...45b011cdd045351eea689d250df0e7cc55af84dd),
[PR2](https://github.com/lilith/moxcms/compare/45b011cdd045351eea689d250df0e7cc55af84dd...4ab190f0e76f95b332a04ce059b51c893e2f5d79).

The sign PR modifies only `src/trc.rs`: a private inlined helper evaluates
`dirty_powf(abs(x), gamma)`, then restores a negative input's sign. Both single
samples and RGB pixels use it. Tests cover matrix, gray and LUT-source routes.

The range PR adds two cached integer bounds to the evaluator and routes inputs
outside the normal-result approximation range to out-of-line `f_powf`. All three
channels are checked together before taking the common fast pixel path; fallback
channels use the same single-sample rule. It also guards CICP gamma 2.2/2.8 and
SMPTE 428, preserving their existing negative-input clamping. Zero, subnormal,
extreme and non-finite cases have dedicated tests. The compatibility test now
requires old bits only inside the fast approximation's valid domain; the newly
corrected values are covered by separate exact-reference checks.

PR1 uses a comparison and sign restoration in Rust:

```rust
let magnitude = dirty_powf(x.abs(), gamma);
if x < 0.0 { -magnitude } else { magnitude }
```

LLVM makes the RGB sign restoration branchless in the measured x86 object:

```asm
215: xorps  %xmm0,%xmm0
218: cmpltps %xmm0,%xmm3
21c: xorps  %xmm5,%xmm2
21f: andps  %xmm3,%xmm2
222: andnps %xmm5,%xmm3
225: orps   %xmm2,%xmm3
```

PR2 computes cached lower/upper bounds from `2^(-125/gamma)` and
`2^(126/gamma)`, clamped to the normal-f32 limits. Its absolute-bit unsigned
range check selects `dirty_powf(abs(x), gamma).copysign(x)` or the cold
`f_powf(abs(x), gamma).copysign(x)`. Four evaluator construction sites switch
to the private constructor. CICP gamma 2.2/2.8 and SMPTE 428 instead use the
fixed domain 2^-44 through 2^44. SMPTE adds pixel helpers and preserves its scale.
The measured RGB fast path has three per-channel conditional jumps before the
common arithmetic body: the source-level combined predicate is not a single
machine comparison per pixel. Complete bodies remain in the artifact directory.

Fresh focused checks: PR1 TRC tests 6 passed; PR2 TRC tests 9 passed; PR2 SMPTE
pixel/single-sample bit parity 1 passed. Historical exhaustive sweep logs remain
available but are not presented as a fresh revalidation.

## Evidence and commands

Full raw packets, rejected-run logs, extracted assembly, Callgrind files and
SHA-256 manifest are retained under the artifact directory documented in
[moxcms_review_2026-11-06.pointer.md](moxcms_review_2026-11-06.pointer.md).

```sh
cargo bench --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features review-revisions --bench underflow -- --mode=confirm --processes=5 --timeout=600 --workload-id=moxcms-review-84104443-45b011cd-4ab190f0-v1 --format=llm
python3 benchmarks/moxcms-gamma/review-evidence.py OUTPUT
```

Run both commands through the host resource wrapper and shared build lock.
The rejected timing run returned 2 after 303 seconds: peak RSS 0.09 GiB,
minimum available 18174 MiB, peak load 1.26. Instruction collection returned 0
after 6 seconds: peak RSS 0.05 GiB, minimum available 18432 MiB, peak load 0.29.
These receipts describe validation resource use, not library memory requirements.

The two rejected ARM full-suite populations returned 2 after 190 and 189 seconds.
The second omitted diagnostic SSH polling and was still rejected; the polling
causation hypothesis was not established. The qualified per-cell study did not
retry individual failed cells or alter admission/confidence thresholds.

```sh
python3 benchmarks/moxcms-gamma/review-cells.py ARM_BINARY NEW_OUTPUT
python3 benchmarks/moxcms-gamma/summarize-review-cells.py RETAINED_STUDY CSV_OUTPUT
```
