# U16 analysis: local x86 measurement

Measured 2026-09-27 on AMD Ryzen 9 9950X3D, x86_64 Linux, rustc 1.98.1
(48a229cea, LLVM 22.1.8). Default features, release profile, runtime SIMD dispatch;
no target-cpu=native. This is a local microbenchmark, not an ARM acceptance result.

```sh
cargo test -p zenpixels-convert --release --lib fused_u16_benchmark -- --ignored --nocapture
```

Each measurement repeats 100 analyses of opaque neutral replicated-U8 RGBA16
samples. This requires all three predicates to finish reading the image. The
separate baseline calls opacity, grayscale and byte-replication scans; the fused
path performs those tests in a single traversal.

| Pixels | Fused, 100 iterations | Separate, 100 iterations |
|---:|---:|---:|
| 4,096 | 90.601 µs | 110.571 µs |
| 1,000,000 | 20.033526 ms | 29.953174 ms |

The normal scalar-oracle test varies 1–4 channels, widths 0–136, opacity/chroma/
replication defects and every vector tail. This measurement does not cover the
full corpus or early-failure throughput. The production narrowing kernel is
unchanged; its separate candidate still needs platform measurements before selection.
