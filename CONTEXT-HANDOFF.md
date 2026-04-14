# Context Handoff — ZenCmsLite + Fused SIMD Gamut Conversion

## Current State (2026-04-13)

### Branch: `explore/fast-p3-srgb` (PR #9)

**ZenCmsLite** — lightweight `ColorManagement` impl using fused SIMD gamut kernels.
- Identifies ICC profiles via 135-profile hash table (~100ns) + CICP-in-ICC extraction
- Falls back to moxcms for unknown custom profiles
- f32: fused TRC+matrix SIMD kernels, ~3.3 GiB/s (95% of moxcms)
- u8: LUT linearize → SIMD (matrix + poly encode) → quantize, 638 MiB/s
- Extended range: opt-in via `ZenCmsLite::extended()`, scalar sign-preserving powf

### Key Files
- `zenpixels/src/descriptor.rs` — `ColorPrimaries::gamut_matrix_to(dst)` (new public API)
- `zenpixels/src/registry.rs` — const fn matrix computation (pub(crate))
- `zenpixels-convert/src/fast_gamut.rs` — stamped SIMD kernels, dispatch, u8/u16 converters
- `zenpixels-convert/src/cms_lite.rs` — `ZenCmsLite` struct, `LiteTransform`, ICC identification
- `zenpixels-convert/src/gamut.rs` — `conversion_matrix()` delegates to registry
- `zenpixels-convert/src/convert_kernels.rs` — RowConverter gamut step uses SIMD (cfg zencms-lite)

### Performance (P3→sRGB 1080p)
| Path | Throughput | vs moxcms |
|---|---|---|
| f32 RGB (fused SIMD) | 3.3 GiB/s | 95% |
| f32 BT.2020 SDR→sRGB | 3.5 GiB/s | 96% |
| f32 BT.2020 PQ→sRGB | 2.1 GiB/s | 58% |
| u8 RGB (fused SIMD) | 638 MiB/s | 24% |
| f32 extended (scalar) | ~200 MiB/s | — |

### What's Left
- **linear-srgb**: PR imazen/linear-srgb#5 needs merge + release (SIMD extended abs+sign)
- **Wire SIMD extended**: use linear-srgb's new SIMD extended slice functions (blocked on release)
- **u8 gap**: 4x behind moxcms. Same algorithm (LUT→matrix→LUT), implementation detail gap.
  Profiling the hot loop would identify whether it's branch prediction, loop structure, or cache.
- **3D CLUT**: explored, works (±1 accuracy), but doesn't help for matrix-shaper profiles because
  the encode step still dominates. Only useful for LUT-based ICC profiles.
