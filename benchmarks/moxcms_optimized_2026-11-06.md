# Output-preserving moxcms gamma optimization, 2026-11-06

The proposed implementation keeps the existing dirty-power polynomial and table.
It changes admission, zero handling and fallback placement. It preserves the
range fix's corrected ProPhoto-to-sRGB result, including the ordinary zero-channel
case described in [the earlier report](moxcms_review_2026-11-06.md).

Candidate: [3e098543](https://github.com/lilith/moxcms/commit/3e09854375e9d9fe54e0d4fc88e31013f5528a9e).
[Exact incremental diff from the first range fix](https://github.com/lilith/moxcms/compare/4ab190f0e76f95b332a04ce059b51c893e2f5d79...3e09854375e9d9fe54e0d4fc88e31013f5528a9e).
This is a review candidate on the existing fork branch; it has not been posted
as a new upstream PR or released.

## Changes

- Combine the three unsigned RGB input-distance checks with OR before comparing
  with the cached span. Passing proves that every channel fits; combinations
  rejected conservatively use the same per-channel rules.
- Return signed zero directly for positive ICC gamma, retaining exact behavior
  for exponent zero. The bounded CICP helper shortcuts only positive zero;
  negative zero retains the exact power function's behavior.
- Share RGB admission for CICP gamma 2.2 and 2.8, keeping their negative-input
  clamp and their scalar behavior.
- Move the mixed/out-of-range ICC RGB fallback to a separate cold function.
  Each channel still chooses the original approximation or exact fallback.

The incremental patch changes two source files: 111 inserted and 39 deleted lines,
including tests. No public API, approximation coefficients, tables or dependencies change.
Subnormals, extreme values and non-finite values retain the range fix's fallback.
The fast path is not restricted to a new nominal gamma interval.

## Correctness

The final candidate passes 103 library tests with `extended_range,in_place,options`,
strict library Clippy and formatting. No existing expectation was weakened.
With pxfm 0.1.30, the independent harness records:

- 4,595,836 evaluator comparisons with zero changed non-NaN output bits and zero
  pixel/scalar disagreements. NaNs compare by classification, not payload.
- 589,824 exact transformed-channel matches across ProPhoto-to-sRGB and CICP
  gamma-2.2/gamma-2.8-to-BT.2020, including mixed negative and zero samples.
- 262,144 signed-zero matches across all 65,536 nominal ICC gamma encodings,
  both encode/decode directions and both zero signs.

These are specified sampled and boundary checks, not an exhaustive F32 sweep.
The source and exact commands are in [the harness](moxcms-gamma/README.md), enabled
with `final-review`. Its `opt_cold` benchmark label aliases the separately frozen
`opt-final` dependency; the source manifest records that distinction.

The wider-exponent prototype is retained only as an experimental control. It
changed 21,032 of 4,595,836 checked results and was slower in the first study's
qualified ordinary-path cells. It is excluded from the proposed implementation.
The earlier OR/pixel controls also predate the final signed-zero corrections.

## Executed instructions and assembly

Callgrind on Ryzen 9 7900X (Zen 4), x86_64-unknown-linux-gnu, Rust 1.99.1 / LLVM
23.1.1, default target CPU. No target-cpu=native. Counts collect `run_work` only;
construction, input generation and allocations are excluded. Every row processes
4096 RGB pixels. These counts do not predict elapsed time on another CPU.

| Workload | Original | Sign only | First range fix | Final candidate |
|---|---:|---:|---:|---:|
| Encode evaluator, positive | 720937 | 557097 | 663593 | 581673 |
| Display P3 to Adobe RGB, F32 | 1529261 | 1358509 | 1465005 | 1383085 |
| CICP gamma 2.2 to BT.2020 | 1546007 | 1546007 | 1599255 | 1406743 |
| CICP gamma 2.8 to BT.2020 | 1519954 | 1519954 | 1573202 | 1380690 |
| ProPhoto to sRGB, every tenth sample zero | 1383817 | 1219977 | 1392997 | 1270519 |
| ProPhoto to sRGB, all zero | 942182 | 778342 | 1294438 | 520294 |
| Display P3 to Adobe RGB, U8 table | 94354 | 94354 | 94354 | 94354 |

[All 65 instruction-count cells](moxcms_optimized_instructions_2026-11-06.csv).
The original and sign-only zero cases produce incorrect values; their counts
are cost controls, not accuracy-equivalent conversions.

The hot ICC RGB evaluator has 133 static instructions, one conditional jump and
one call site, versus 345 instructions, eight conditional jumps and three call
sites in the first range fix. The final cold RGB fallback adds 245 instructions:
the combined bodies have 378, so total static code is larger. Executed counts
above include whichever fallback actually runs.

Complete final disassembly: [hot evaluator](moxcms-gamma/final-evaluator.asm),
[cold RGB fallback](moxcms-gamma/final-fallback.asm). The unchanged out-of-line
exact-power implementation is separate in both versions.

Native instruction binary SHA-256:
`d4f8e676867e58dc1e45dc59a4fd5c54f1a9504859ba213faf1244fde7919a6e`.

Native instruction collection: rc=0 16s | peak-RSS 0.07GiB | min-avail 18531MiB | peak-load 1.16. These are harness resource measurements.

## ARM timing protocol

Neoverse N1, aarch64-unknown-linux-gnu, Rust 1.99.1 / LLVM 23.1.1, GCC 15.2.0
cross-linker, default target CPU. The final source is measured in a separate
45-cell study, five fresh processes and 30 rounds per cell. Each cell uses
confidence 1 - 0.05/45; zenbench also adjusts its comparison family. The combined
study family is controlled at 95%, with the original 1% margin and resource
gates. CPU frequency was not independently verified.

ICC evaluator and ICC transform comparisons use the sign-only implementation
as baseline; CICP comparisons use the original implementation, since the sign
patch does not change CICP. Means are microseconds per complete row at the
stated width. Process-level percentages use paired log ratios, so they need not
equal ratios of displayed rounded arithmetic means. A resource-rejected cell
supplies no timing claim and is not retried selectively.

Construction and allocation are outside the timed loop. The cached bound
computation's construction cost remains unmeasured. No x86 wall-clock or Zen 5
performance claim is made.

The first prototype study qualified 41/45 cells, and the outlined prototype
study qualified 37/45. Their rejected cells and complete packets are retained
separately. They measure earlier source identities and are not pooled with the
final candidate. The earlier variants remain experimental controls.

## Final timing results

All 45/45 final cells qualified. Every retained process and recorded sample
passed the unchanged resource checks. Against the declared baselines, the final
candidate improved 22/45 comparisons, met non-inferiority in 7/45 and equivalence
in 1/45, regressed in 4/45, and was inconclusive in 11/45.

| 1024 RGB pixels | Original µs | Sign only µs | First range µs | Final µs | Final vs declared baseline |
|---|---:|---:|---:|---:|---|
| eval/encode/zero0/neg0/w1024 | 30.444 | 24.610 | 24.813 | 24.006 | -2.46% [-4.17%, -0.71%], non_inferior |
| eval/encode/zero0/neg50/w1024 | 32.935 | 24.617 | 24.858 | 23.989 | -2.55% [-5.55%, +0.54%], non_inferior |
| eval/encode/zero10/neg0/w1024 | 30.411 | 24.601 | 28.814 | 25.698 | +4.46% [+3.16%, +5.77%], regression |
| eval/encode/zero100/neg0/w1024 | 30.157 | 24.314 | 22.358 | 6.594 | -72.88% [-73.58%, -72.16%], improvement |
| eval/decode/zero0/neg0/w1024 | 30.639 | 24.685 | 25.013 | 24.054 | -2.56% [-3.53%, -1.58%], improvement |
| eval/decode/zero0/neg50/w1024 | 32.941 | 24.664 | 24.832 | 24.013 | -2.64% [-3.50%, -1.76%], improvement |
| eval/decode/zero10/neg0/w1024 | 30.454 | 24.655 | 28.775 | 25.684 | +4.18% [+2.97%, +5.39%], regression |
| eval/decode/zero100/neg0/w1024 | 30.129 | 24.304 | 23.686 | 6.654 | -72.63% [-75.66%, -69.22%], improvement |
| transform/cicp_gamma22_to_bt2020/w1024 | 76.480 | — | 80.624 | 69.423 | -9.23% [-10.45%, -7.99%], improvement |
| transform/cicp_smpte428_to_bt2020/w1024 | 69.618 | — | 69.312 | 70.023 | +0.58% [-2.63%, +3.89%], inconclusive |
| transform/cicp_gamma28_to_bt2020/w1024 | 75.435 | — | 78.899 | 67.650 | -10.32% [-10.76%, -9.88%], improvement |
| transform/adobe_to_prophoto_extended/w1024 | 75.903 | 61.321 | 59.921 | 59.358 | -3.20% [-4.20%, -2.19%], improvement |
| transform/prophoto_to_srgb/zero0/w1024 | 74.722 | 65.086 | 65.358 | 64.529 | -0.86% [-1.69%, -0.01%], non_inferior |
| transform/prophoto_to_srgb/zero10/w1024 | 74.598 | 65.019 | 66.348 | 64.695 | -0.49% [-5.53%, +4.83%], inconclusive |
| transform/prophoto_to_srgb/zero100/w1024 | 45.561 | 39.501 | 30.908 | 13.471 | -65.90% [-66.55%, -65.23%], improvement |

The four confirmed regressions are evaluator encode/decode with 10% zero samples
at widths 1024 and 4096, relative to sign-only. At width 1024 they are +4.46% and
+4.18%; at width 4096 they are +10.97% and +11.05%. Sign-only still produces incorrect zero results. The optimized candidate
reduces the first range fix's mixed-zero cost, but does not make that guard free.

CICP gamma 2.2 and 2.8 improve against the original implementation. Ordinary
positive encode is non-inferior to sign-only at width 1024, while positive decode
improves. SMPTE 428 and the full mixed-zero ProPhoto conversion are inconclusive
at this width; their means alone do not establish non-inferiority.

Final ARM wrapper: 370 seconds, peak RSS 0.06 GiB, minimum available 14856 MiB,
peak load 0.98. These describe the benchmark harness, not per-image memory use.

Final ARM binary SHA-256: `18c6997d1771bcd22546b3093722cbe5360f6dd0520493aa8389c9593314772c`.

All 45 cells, their candidate comparisons, and the rejected earlier studies
remain in the archive identified by the [artifact pointer](moxcms_optimized_artifacts_2026-11-06.json). Selected rows are also
committed as [JSON](moxcms_optimized_arm_selected_2026-11-06.json).
