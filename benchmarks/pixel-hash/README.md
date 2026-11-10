# Pixel hashing backend comparison

`hash_with` selects XXH3-64 through zenhash. `Hash` remains generic over the
caller's Hasher. Both serialize the same logical values through fixed 1 KiB
writes; see [the inclusion contract](../../docs/hashing.md).

The comparison covers 116 groups: packed RGBA8, RGBA16, RGBF32, RGBX8 and
GrayF64, three separately allocated RGB8 planes, NV12, and P010. Geometry spans
0×0, 1×1, 8×8, 64×64, 256×256, 1024×1024, and 3840×2160; packed/RGB planes
also cover 64 bytes of row padding. Eighteen groups include 512-byte, 4 KiB,
or 128 KiB ICC payloads plus CICP, diffuse white and HDR peak metadata.
Construction and allocation are outside timing. Callbacks invoke the real
`Hash` implementations with either RapidhashV3 or Xxh3, then finalize once.
Input words use `(i * 0x9e3779b97f4a7c15).rotate_left(17)`.

## Initial x86 observations — unconfirmed

The complete [CSV](results_2026-11-09.csv) retains both means and medians for
every group. Three independent processes ran 12 interleaved rounds per group
on Ryzen 9 7900X, release/thin LTO, without `target-cpu=native`. The measured
thread was pinned to CPU 2; zenbench admitted the homogeneous CPU class.

**This run failed the resource confirmation gate.** Observations reported
competing heavy processes. All 116 groups completed, but the aggregate is
unreliable and exit status was 2. These are descriptive observations, not a
confirmed performance result; no significance or confidence claim follows.

| Case | RapidhashV3 median, ns | XXH3 median, ns |
|---|---:|---:|
| packed RGBA8, 1×1 | 129.678 | 83.691 |
| packed RGBA8, 64×64 | 1245.345 | 945.591 |
| packed RGBA8, 3840×2160 | 2589975.083 | 1680961.000 |
| planar RGB8, 1×1 | 208.745 | 421.409 |
| planar RGB8, 64×64 | 1007.116 | 1319.260 |
| planar RGB8, 3840×2160 | 1731047.611 | 1305284.750 |
| packed RGBA8 1×1 + 128 KiB ICC | 9766.488 | 6855.932 |
| P010, 3840×2160 | 43694060.417 | 42509292.583 |

XXH3 had the lower median in 66 of 116 groups. The initial choice favors its
bulk-image and large-metadata observations; small planar values favor rapid.
Native code extraction dominates the P010 case. No ARM or WASM performance
claim is made from this x86 run. Cross-platform correctness is tested separately
with fixed complete-pixel goldens and the independent C MurmurHash3 corpus.

Throughput counts active sample bytes only; it excludes metadata and padding.
Use latency for metadata groups. The raw archive retains round order, counts,
resource observations, process identities, calibration and CPU placement.
The resource wrapper reported peak RSS 0.82 GiB, minimum available memory
12550 MiB, peak load 1.40, and elapsed 468 seconds.

## Reproduction and evidence

```
just hash-bench --mode=confirm --workload-id=pixel-hash-v2 --processes=3 --rounds=12 --timeout=1200 --format=llm
```

Set `TMPDIR` to a disk-backed scratch directory, `ZENBENCH_PIN` to the measured
thread's CPU, and `ZENBENCH_DATA_DIR` to the protected result directory. Run
through the host's resource wrapper. Strict resource/placement gates remain
enabled. No admission failure is a benchmark result.

The v2 source was `9d7a2ff1b1532b6a54a55f6d3caa2bd77b1c0649`, with zenhash
`98c385a4e54a1d41870610e13a614295c538275c`. The retained executable SHA-256 is
`11b18fd189c9a1cfc699a151ea07e4681995978960b7d2bd5d06db86d2e6ffa2`;
CSV SHA-256 is `5af0b3b4edba04577e9c935a7bb3d5325c5d3edde2e963e9f20398cad34ca1c6`.
Private raw evidence is retained at
`~/.cache/zenpixels-hash/2026-11-09-r7900x-v2/`. The aggregate object is
`33e135b908dd55131a7c24d55d273f0a554b0c5cdaa9733ff14f2de7208fc2d8.json.gz`.
The older v1 run predates fixed write framing and is superseded; it is retained,
not used for selection. Failed CPU-set admission attempts are also preserved.

The optional zenhash dependency is pinned to a reviewed git revision while its
MurmurHash3 addition awaits release. Replace it with the published version
before publishing zenpixels; this change does not authorize a crate release.

## Second x86 host and retained rejected attempts

A second complete run used a Core Ultra 7 265K, the same x86 executable and
three processes × 12 rounds × 116 groups. A kernel-enforced CPU set restricted
the entire process tree to performance CPUs 0–7; no group pin was applied.
Runtime dispatch was enabled (`zenhash/std`, `xxh3`, `avx512`, `rapidhash`).
Without std, zenhash selects only compile-time-proven tiers; that configuration
is correctness-tested but its performance is not measured here.

This run also **failed resource confirmation** (exit 2). Only 2762 of 4176
round observations were clean. All observations, including flagged rounds,
remain in [the second CSV](results_2026-11-09_i265.csv) and raw archive; they
must not be promoted to a confirmed comparison. The following are descriptive
medians in nanoseconds:

| Case | RapidhashV3 | XXH3 |
|---|---:|---:|
| packed RGBA8, 1×1 | 34.303 | 55.958 |
| planar RGB8, 1×1 | 88.560 | 167.704 |
| packed RGBA8, 3840×2160 | 2002392.417 | 1649521.083 |
| planar RGB8, 3840×2160 | 1403391.750 | 930131.528 |
| packed RGBA8 1×1 + 128 KiB ICC | 6906.804 | 3094.674 |
| P010, 3840×2160 | 20254055.750 | 19044341.000 |

XXH3 had the lower median in 73 of 116 groups. It remains the selected default
for bulk image/profile hashing; rapid's advantage on tiny values is preserved
in the data. A caller prioritizing tiny keys can use `Hash` with RapidhashV3.
The available shared hosts did not provide a clean strict confirmation run;
this limits the confidence of the performance decision, not hash correctness.
No universal winner, confirmed speedup, or cross-platform throughput claim is
established. A quiet-host confirmation remains a performance evidence task.

The wrapper recorded 248 seconds, peak RSS 0.74 GiB, minimum available memory
21381 MiB, peak load 1.15. The user-manager scope was unavailable in the nested
launch, so the wrapper reported no cap; a 5 GiB/no-swap cap was then applied
and verified on this experiment's outer system scope while it was running.
Future reproduction should apply both CPU and memory limits to the outer
scope at launch. The aggregate object is
`71c0faa8c1f63ce0eab1dc15667e20db862f549ea17df84317a4e14a27db29f0.json.gz`.

Two further attempts are retained without selecting favorable subsets: a
pinned repeat stopped after its checkpoint was already flagged, and a
seven-group unpinned diagnostic (three processes) that completed but failed
confirmation. The diagnostic identified foreign session processes and kernel
compaction in its gate trace. An ARM attempt was stopped when live fuzz
workers were observed; its partial results establish no performance claim.
No foreign jobs were stopped and no thresholds or admission gates were relaxed.

Private full evidence, executable identities, raw rounds, flags, progress
records and rejected attempts are mirrored under
`output/zenpixels/pixel-hash-2026-11-09/` in the private artifact archive.
Captures include process metadata and are intentionally not committed here.

Use `just hash-bench-summarize <aggregate.json.gz> <summary.csv>` to reproduce
both CSVs. The extractor retains every group and both backends, reports the
aggregate/round flags and executable identity, and refuses to overwrite a
nonidentical existing result. It reproduces the committed CSVs byte for byte.
