# Context Handoff — Color Registry + Fast Gamut CMS

## Current State (2026-04-13)

### PR Open
- **imazen/zenpixels#8** (`feat/color-registry`): Color registry with const-evaluated gamut matrices
  - `ColorPrimaries::chromaticity()` — CIE xy for all 8 primaries
  - `registry::gamut_matrix(src, dst)` — const fn, computes any 3×3 matrix with Bradford adaptation
  - Registry table: `(primaries, transfer) ↔ CICP ↔ NamedProfile`
  - 11 tests, CSS Color 4 cross-validated
  - Compile time: 0.5ms per const matrix (100 matrices = 53ms overhead)

### Explore Branch
- `explore/fast-p3-srgb`: Fused SIMD gamut conversion
  - `fast_gamut.rs` — 1432 lines, 48 tests, behind `zencms-lite` feature flag
  - 1.9x faster than moxcms for f32 P3↔sRGB, more accurate (99.9% exact vs 90.5%)
  - Uses archmage `#[rite]`/`#[arcane]` + `incant!` for AVX2+FMA dispatch
  - Currently has 24 hardcoded matrix constants — needs refactoring to use registry
  - u8 path is 55x slower than moxcms (scalar polynomial, needs SIMD LUT batch)
  - Validated against moxcms (256³ exhaustive), CSS Color 4, ICC profile colorants

### Main Branch Changes (already merged)
- `ColorProfileSource::PrimariesTransferPair { primaries, transfer }` variant
- `ColorManagement::build_source_transform()` trait method
- `build_transform_from_cicp` removed (subsumed by build_source_transform)
- `NamedProfile::to_primaries_transfer()` / `from_primaries_transfer()`
- zencodec: icc_extract_cicp cross-validated with moxcms, comprehensive ColorAuthority tests

## Next Steps

1. **Merge registry PR** (#8)
2. **Rebase explore branch** on registry, refactor fast_gamut:
   - Replace 24 hardcoded `const` matrix arrays with `const gamut_matrix()` calls
   - Delete the DCI-P3 Bradford matrices (computed automatically now)
   - fast_gamut becomes ~500 lines shorter
3. **Implement `ZenCmsLite`** struct:
   - `impl ColorManagement for ZenCmsLite`
   - `build_source_transform`: match on `PrimariesTransferPair`, compute matrix via registry, pick TRC kernel
   - `identify_profile`: reuse existing hash+colorant identification from moxcms backend
   - `build_transform_for_format`: identify both profiles → delegate to build_source_transform
4. **Wire into `finalize_for_output`**: already tries `build_source_transform` first
5. **Fix u8 performance**: use `linear-srgb` SIMD LUT batch functions wrapping fused f32 kernel
6. **Refactor scattered match statements** to use registry lookups (DRY)

## Key Files
- `zenpixels/src/registry.rs` — registry + const matrix math
- `zenpixels/src/descriptor.rs` — ColorPrimaries, TransferFunction enums + chromaticity()
- `zenpixels/src/color.rs` — ColorProfileSource, NamedProfile, ColorAuthority
- `zenpixels-convert/src/cms.rs` — ColorManagement trait
- `zenpixels-convert/src/cms_moxcms.rs` — moxcms backend
- `zenpixels-convert/src/fast_gamut.rs` — fused SIMD kernels (explore branch)
- `zenpixels-convert/src/output.rs` — finalize_for_output dispatch

## Benchmark Reference (1080p f32 RGB)
| Conversion | fast_gamut | moxcms |
|---|---|---|
| P3→sRGB f32 | 5-6ms (3.7-4.7 GiB/s) | 7-12ms (2-3.5 GiB/s) |
| BT.2020 PQ→sRGB f32 | 6ms (3.7 GiB/s) | 6.5ms (3.6 GiB/s) |
| P3→sRGB u8 | 126ms (47 MiB/s) ❌ | 2.3ms (2.6 GiB/s) |

## Accuracy (u8, vs f64 ground truth)
| | fast_gamut | moxcms |
|---|---|---|
| P3→sRGB exact | 99.9% | 90.5% |
| Max delta | ±1 | ±2 |
