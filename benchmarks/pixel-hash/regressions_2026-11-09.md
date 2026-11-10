# Hashing regression review

The native P010 path had acquired two avoidable costs: a dynamic component-step
iterator prevented contiguous reads from vectorizing, and serializing extracted
codes made a variable-size memcpy call per sample. The fix keeps the existing
CodePlane reader, adds its contiguous loop and passes complete u16 code blocks
to the existing endian-aware sample serializer. The generic component-step path
remains available. No native adapter or unsafe code was added.

## Fixed-work instruction witness

`profile_codes` constructs one 3840×2160 picture and hashes it once. Both before
and after executables use release/thin LTO, without target-cpu=native. Counts
include construction, the hash and process startup/cleanup, not just the kernel.
The same executable source/inputs and full hash output are checked on both sides.

| Callgrind workload | Before instructions | After instructions | Output checksum |
|---|---:|---:|---|
| P010, one frame | 587,535,074 | 47,227,132 | `3ebf366ea33b2ccf` |
| Packed Gray16 control, one frame | 16,334,328 | 16,333,589 | `239196a458b40b13` |

The baseline profile attributed 38.82% of its instructions to per-code memcpy,
28.97% to code serialization and 28.65% to CodePlane reads. This identifies a
real implementation cost, independently of timing noise. Callgrind uses the
supported AVX2 fallback, not AVX-512; its counts do not establish native cycle
speedups. The native perf-stat 32-frame probe also retained identical outputs:
instructions 19,320,131,966 before and 1,220,361,705 after (three repetitions).
Cycle/time variability is retained in the raw reports rather than promoted to
an isolated timing claim.

Resource wrapper reports for P010 Callgrind: before peak RSS 0.05 GiB, minimum
available 13607 MiB, peak load 1.50; after peak RSS 0.04 GiB, minimum available
13094 MiB, peak load 7.16. Packed controls: peaks 0.04/0.05 GiB, minima
13802/13761 MiB, peak load 0.34 each. These are process measurements, not claims
about required image memory or a memory optimization.

## Paired timing observations — unconfirmed

The [CSV](regressions_2026-11-09.csv) retains every group, algorithm, median,
mean, MAD and ratio. Thirty groups cover P010 tight/padded rows, component
steps 1/2, packed RGBA16, old/new XXH3 backends, Murmur32 and FNV compatibility,
and the cost of the new 128-bit variant. Each existing-behavior group checks
before/after output parity before timing. Different-width Murmur cost groups
are explicitly separate: their digests differ and 128 bits has no old baseline.

The pinned old core is `e853eb7545668f46550aa72e42897fa30ebba8de`; the reader
fix is `6ce0d3497ea2`. Old/new zenhash revisions are `98c385a4e54a` and
`1836e2fa31d8`. Harness source is `2f6b88b5adf0`. Same slices, input pattern,
metadata, dispatch options and compiler profile are used within each pair.
The native group name `pad32` means 32 u16 samples (64 padding bytes) per row.
The thread is not pinned; strict auto placement verifies the uniform CPU
class on Ryzen 9 7900X. Zen 5, ARM and WASM performance is not measured here.

Three independent processes × 12 interleaved rounds completed every group.
**The strict resource confirmation failed:** only 491/1080 observations were
clean; exit status was 2. All flagged rounds are retained. No threshold was
relaxed, and no favorable subsets or retries were selected. The medians below
are descriptive observations, not confirmed speedups or a significance test.

| Group | Before median | After median |
|---|---:|---:|
| P010 3840×2160, tight | 34.499921 ms | 1.560037 ms |
| P010 3840×2160, 64-byte row padding | 33.395851 ms | 1.569572 ms |
| Native component step 1, 3840×2160 | 22.726553 ms | 1.038892 ms |
| Native component step 2, 3840×2160 | 22.684070 ms | 3.926628 ms |
| Packed RGBA16 3840×2160 control | 1.880608 ms | 1.856198 ms |
| XXH3 backend, same packed image | 1.871812 ms | 1.887189 ms |
| Murmur32 128 KiB control | 52.659398 µs | 52.704019 µs |
| FNV+fmix 128 KiB control | 95.480318 µs | 95.833821 µs |
| Murmur cost, 128 KiB: x86_32 → x64_128 | 52.863513 µs | 12.804580 µs |

The largest observed control slowdown was 1.4842% (packed 1×1). That small
difference cannot be classified as a regression or noise from this rejected
confirmation. The data shows no additional slowdown comparable to the P010
work defect; it does not prove a universal absence of timing regressions.
The wrapper reported 136 seconds, peak RSS 0.25 GiB, minimum available memory
13327 MiB and peak load 1.29.

## Reproduction and correctness

```
just hash-regression-build
cargo bench --manifest-path benchmarks/pixel-hash/Cargo.toml --bench regressions -- --mode=confirm --processes=3 --rounds=12 --workload-id=hash-regression-v1 --timeout=1200 --format=llm
python3 benchmarks/pixel-hash/summarize.py --regressions <aggregate.json.gz> <summary.csv>
just hash-prof-callgrind <profile_codes> <callgrind-output> hash 1
just hash-prof-stat <profile_codes> <perf-stat-output> hash 32
```

Run through the host resource wrapper, with disk-backed TMPDIR and a protected
ZENBENCH_DATA_DIR. Preserve both compiled executables before rebuilding. The
profiler recipes also support `packed`, `codes` and `component` modes.

Tests preserve existing complete-pixel golden digests; bulk reads cover every
valid U8/U16 bit depth/placement, component steps 1/2, empty/small/vector-tail
widths and strided rows. Subbyte U8 hashing verifies that padding differences
are ignored and real code differences remain visible. Big-endian CI also
checks the unchanged portable digests.

Private full evidence is mirrored under
`output/zenpixels/pixel-hash-2026-11-09/captures/`, including
`2026-11-09-regression-audit/` and `2026-11-09-regression-paired/`.
The paired executable SHA-256 is
`39752656e4f8b0d47f532886a41890694706ff50fd5ad5ef7ce1106b5c967164`;
aggregate object `c0bcd0cd6bfb65bdd61004b214c72459f025c4673b4ddf0a596f9d1a273891ef.json.gz`;
CSV SHA-256 `e22b0c9f929a6a05adcd8fc891400c8381d6fbb326b901f44818a14110b17e4c`.
