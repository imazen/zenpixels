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
