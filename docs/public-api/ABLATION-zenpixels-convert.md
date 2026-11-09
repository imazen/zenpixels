# Current converter API and migration

The committed [API snapshot](zenpixels-convert.txt) is the signature inventory.
The 2026-11-04 ablation proposal is historical; it no longer describes a pending
feature branch or current caller counts.

## Public free operations

| Operation | Why it remains |
|---|---|
| `adapt::adapt_raw_rows_for_encode` | Raw input can have sample-unaligned row starts that PixelSlice cannot admit. Sample-aligned partial-pixel padding already works through PixelSlice. |
| `codes::to_canonical` | Its input is a collection of independently encoded components. No single packed-pixel receiver owns that operation. |

All other default operations use a concrete receiver or extension trait.
An all-feature rustdoc run also exposes benchmark reexports of
`convert_linear_rgb`, `convert_linear_rgba` and `fused_predicates_rgba8_cg`
through `__bench_scan`. `__bench_u16_hybrids` has four hidden shims (`lut_lut`,
`lut_poly`, `poly_lut`, `poly_poly`); orientation exposes `__bench_apply_orientation_scalar` and
`__bench_apply_orientation_staged` under `__bench_orient`. These are benchmark-only feature surfaces with concrete in-repo
bench callers, not additional ordinary conversion entry points. The all-feature
check compiles those callers too.
Optional CMS backends expose their plugin implementations; HDR opt-in exposes
measurement/pluggable rendering types. Neither justifies another generic helper.
`NativeCodeSlice` and its solely dependent conversion APIs were removed.

## Canonical migration paths

| Retired form | Current form |
|---|---|
| `finalize_for_output*` | `EncodeReady::finalize` for independent owned output; `OutputPlan::borrow_rows` / `convert_rows` for streaming or caller storage |
| `adapt_for_encode*` | `PixelSliceConvertExt::adapt_for_encode`; raw bytes use the free operation above |
| `apply_orientation*` | Buffer/slice orientation extensions; consuming identity uses `into_oriented` |
| `best_match`, `negotiate`, cost helpers | `ConvertIntent` / `ConversionCost` methods |
| Gamut and Oklab helpers | `GamutMatrix` / `OklabSpace` methods |
| ICC synthesis helpers | `SynthesizedIcc::for_cicp` / `for_gray_cicp`; these synthesize profiles, not sample conversion |
| `ColorManagement` | Open `PluggableCms` and row-transform traits |
| `RowConverter::clone` | Fallible `try_clone` |
| Estimation APIs | Removed; no replacement estimate is promised |

Concrete-type extension traits are sealed. Pixel and CMS plugin traits retain
external implementation support. Descriptor enums use explicit mappings rather
than Rust integer casts. Native types use PixelFormat::Generic plus from_parts;
metadata descriptions do not grant numerical processing support.

The [baseline audit](semver-0.2.17-to-candidate.md) records actual checker results
and migration classifications. Publication and downstream qualification remain
separate gates. Regenerate signatures with `just api-doc`; run feature-enabled
compilation with `just design-feature-check`.
