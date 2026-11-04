# Gamut compression candidates, 2026-11-04

r7900x (Ryzen 9 7900X, Zen 4), v3 dispatch, thin LTO. Not measured on Zen 5.
Source Display P3 and BT.2020 to sRGB primaries, knee 0.96, gate tau 0.0875 and eps 0.05.
Real images: 10 from imazen-26 render-v4 `png-p3` (private corpus), centered 2048x1024 bands.

- `bench-summary.txt`: ns/pixel. Kernel only, then whole P3 u8 to sRGB u8 rows.
- `sweep.csv`, `ramps.csv`, `primaries.csv`: same grids as `../gamut-survey`.
- `real.csv`: per-image out-of-sRGB share, gate pass rate, and color differences.
- `gate.log`: tightest tau for each eps, from 399 lightness by 1,440 hue boundary points.

Commands,: build with `cargo build --release`, then run
`gamut-compress gate`, `gamut-compress quality _ 0.0875 0.05`,
`gamut-compress real <dir> 0.0875 0.05`, `gamut-compress bench <dir> 0.0875 0.05`.

The full `bench.json` and `bench.log` exceed the size limit for generated data and are not committed.
