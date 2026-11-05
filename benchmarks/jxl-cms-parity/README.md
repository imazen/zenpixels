# `PluggableCms` behind `JxlCms`

Decodes real JPEG XL files through three CMSes and compares them: lcms2 from `zenjxl-decoder-cli`, `zenjxl-decoder`'s own
moxcms wrapper, and `zenpixels_convert::cms_moxcms::MoxCms` through a `JxlCms` adapter. `CMYK_ORACLE=<cmyk_layers.jxl>`
instead checks both moxcms paths against lcms2 values pinned in the decoder's unit test.

It builds outside the repo, with a `Cargo.toml` that depends on this tree, `zenjxl-decoder` and `zenjxl-decoder-cli`, and an archmage cap
in the decoder raised to admit 0.9.31-beta.2. It unifies on moxcms 0.9.1.

## Results, 2026-11-05

Worst difference in 8-bit codes, clamped to 0..=1, and the share of samples more than 2 codes apart.

| File | Target | Adapter vs decoder wrapper | Adapter vs lcms2 | Decoder wrapper vs lcms2 |
|---|---|---|---|---|
| `alpha_triangles` | P3 | 0.03, 0% | 0.03, 0% | 0.00, 0% |
| `lossless_pfm` | P3 | 0.21, 0% | 0.17, 0% | 0.04, 0% |
| `bicycles` | P3 | 0.00, 0% | 0.00, 0% | 0.00, 0% |
| `grayscale` | sRGB | 0.00, 0% | 0.00, 0% | 0.00, 0% |
| `cmyk_layers` | sRGB | 0.09, 0% | Not comparable | Not comparable |
| `cmyk_layers` | P3 | 1.42, 0% | Not comparable | Not comparable |
| `hdr_hlg_test` | sRGB | 7.25, 0.07% | 30.07, 15.96% | 34.02, 16.13% |
| `hdr_pq_test` | sRGB | 46.31, 0.28% | 99.42, 31.40% | 103.26, 31.60% |
| `hdr_pq_test` | P3 | 45.37, 0.44% | 70.13, 32.77% | 70.34, 33.04% |

The "Not comparable" cells predate zenjxl-decoder `bffc2ed`: `Lcms2Cms` used to pass 0..1 where lcms2's float CMYK takes 0..100, which returned 0.968 for no ink. Against the lcms2 values in the decoder's `cmyk_to_srgb_matches_lcms2_relative_bpc`, the adapter and the
wrapper agree, within 0.26 of a code on four colors.

## Exact-math harnesses

`exact/` holds what the findings below and in `CLAUDE.md` came from. Each is one `main.rs` with path dependencies on this tree and on a copy of moxcms 0.9.1 with `exact/moxcms-0.9.1-bisect.patch` applied, which adds stage toggles and changes nothing while they're off. `hdr_class.rs` also takes a copy of `imazen/zenjxl-decoder` and `imazen/lcms`; `adapter_paths.rs` and `gray_icc.rs` take `lcms-cref`. Profiles: `zenpixels-convert/src/profiles/` plus mozjpeg's `test1.icc` and `test3.icc` from codec-corpus.

- `mox_bias.rs`: moxcms's U8 path over all 2^24 inputs, one stage at a time, with a scalar model that reproduces its bytes.
- `adapter_paths.rs`: every `MoxCms` route against exact math, and tables evaluated exactly, tetrahedral and trilinear.
- `hdr_class.rs`: the HDR rows below against an exact `mft1`, the analytic tone map, libjxl's analytic path and `djxl` PFMs.
- `gray_icc.rs`: every GRAY-class bundled profile through a gray PNG, C lcms2 and the load-bearing rewrite.

## Classifying the HDR rows, 2026-11-05

Each row's cause, from a harness that evaluates the source profile's `A2B0` table exactly in f64 and swaps one stage at a time in a scratch copy of moxcms 0.9.1. PQ and HLG are BT.2100; `intensity_target` is 10000 and 1000. Worst code and mean |d| over the decoded image's rows. Not measured on Zen 5.

The table is `mft1`, 9 nodes per axis, 8-bit Lab, holding Rec. 2408 (PQ) or an 80-nit OOTF (HLG) tone map.

| Row | Class | Evidence |
|---|---|---|
| Adapter vs decoder wrapper: HLG sRGB 7.25, PQ sRGB 46.31, PQ P3 45.37 | Legitimate. Adapter asks for tetrahedral on moxcms's output grid, and the wrapper leaves moxcms's default, trilinear | Wrapper's options plus `Tetrahedral` give the same 7.25, 46.37 and 45.37 against the wrapper, and stay within 0.31 of the adapter |
| Adapter or wrapper vs lcms2: HLG sRGB 30 and 34, PQ sRGB 99 and 103, PQ P3 70 | In moxcms, two causes below. Not in the adapter, the wrapper, or lcms | Raising moxcms's 33-node grid to 129 changes nothing, and no adapter option does |
| lcms2 and the lcms port | Correct reader of the table | C lcms2 is within 0.24 of exact tetrahedral on a 65-cubed grid and 0.09 on the images. `Lcms2Cms` and the port are identical to C lcms2 at every sample (0.00) |

The two causes in moxcms, each measured by removing it:

| Cause | Where | Size |
|---|---|---|
| Tables with a Lab or XYZ PCS are always trilinear, whatever interpolation is asked for. lcms2 uses tetrahedra, and ICC prescribes neither | `conversions/lut3x3.rs`, `impl Stage for Lut3x3`, "If PCS is LAB then linear interpolation should be used" | PQ sRGB image: exact trilinear vs exact tetrahedral, mean 1.15, worst 106 |
| A v4 RGB table profile's black point is always taken as `DEFAULT_V4_BLACK_POINT` and mapped to zero, whatever the intent | `conversions/bpc.rs:116`, applied by `apply_bpc_scaling` in `make_lut_transform` | Mean about -1 code. Without it, moxcms against exact trilinear falls from mean 1.32 to 0.13 on PQ sRGB, and 1.15 to 0.03 on HLG sRGB |

Against the math of HDR itself, none of these three are the reference, because the table carries a tone map:

- Table vs its own analytic tone map, unrounded: worst 91 on the PQ sRGB image. A coarse table's resolution is legitimately different.
- libjxl's CMS doesn't read this table. For PQ and HLG sources it applies the transfer itself and stays in linear light through lcms2, with no tone map. `djxl` matches that analytic path to 0.06 of a code on all four cases, and differs from every row above by a mean of 24 (HLG) to 77 (PQ) codes.
- `PluggableCms` handed CICP instead of the ICC bytes gets a matrix and an analytic curve. PQ matches libjxl's path to 0.05 on the images. HLG doesn't, worst 56, because libjxl adds an OOTF for its `intensity_target`.

So the `JxlCms` adapter here reads an ICC tone map, and a libjxl-style one would pass CICP for PQ and apply the OOTF for HLG. Neither path is a wrapper bug.
