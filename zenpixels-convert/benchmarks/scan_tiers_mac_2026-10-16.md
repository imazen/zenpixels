# Fused pixel scan tiers on macOS ARM64, 2026-10-16

Source `174e01e5`, M4 Pro. Command: `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-scan-tiers-macos`. The recipe ran `cargo bench -p zenpixels-convert --features __bench_scan --bench kernel_tiers -- --cpu-policy=unrestricted --format=llm`. Full output is `~/tmp/zenpixels-scan-tiers.log` on this Mac. The scheduler was unrestricted; the opaque-only NEON group had 56% CV.

| 1 MP RGBA8 predicate | NEON mean | Forced scalar mean |
| --- | ---: | ---: |
| Opaque only, all true | 86.18 µs | 275.66 µs |
| Grayscale only, all true | 103.11 µs | 453.56 µs |
| Both, all true | 146.86 µs | 252.51 µs |
| Both, fails on first pixel | 131.6 ns | 153.5 ns |

All four measured scan comparisons favored NEON. The all-true buffers traverse the image; the fail-first case measures entry and early-exit cost. This is a row predicate audit, not an end-to-end codec speedup. No dispatch change is indicated by these results.

`run-heavy`: peak RSS 0.26 GiB, minimum available memory 5617 MiB, peak load 3.18.
