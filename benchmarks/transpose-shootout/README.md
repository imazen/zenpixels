# transpose-shootout

Dev-box-only measurement equipment for zenjpeg#150 / the orient.rs transpose
work: benches 1/2/3/4-channel interleaved transpose (+ Rotate90 where
supported) across every known Rust implementation and the C++ Simd library.
Workspace-excluded; never built by CI; `publish = false`.

Contestants: `zpc` (zenpixels-convert `apply_orientation_into`), `ft`
(fast_transpose 0.2.7, its SIMD on), `ejm` (transpose 0.2.3, scalar), `zune`
(zune-imageprocs 0.5.1 planar; u16/u32 reinterpret views for 2/4ch),
`simd++` (ermig1979/Simd `SimdTransformImage`, behind `--features cpp-simd`).

Every contestant is verified against an independent naive oracle before any
timing; fast_transpose's flip/flop flags and Simd's transform enum are
probe-derived against that oracle so nobody silently benches a different
operation. All contestants write into pre-touched caller buffers (no alloc /
first-touch faults in the timed region); tight strides everywhere.

```
./fetch_simd.sh                                  # once, for the C++ contestant
cargo run --release --features cpp-simd          # full grid
cargo run --release -- --group="T 3ch 12MP"      # substring filter
```

The harness itself uses `unsafe` only for the C FFI call; the zenpixels-convert
kernels under test remain `#![forbid(unsafe_code)]`.

Results are committed to `../benchmarks/` with `.meta` provenance (host, git
commit, Simd commit from `third_party/SIMD_COMMIT.txt`).
