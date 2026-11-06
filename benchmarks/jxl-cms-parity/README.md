# `PluggableCms` behind `JxlCms`

Decodes real JPEG XL files through three CMSes and compares them: lcms2 from `zenjxl-decoder-cli`, `zenjxl-decoder`'s own
moxcms wrapper, and `zenpixels_convert::cms_moxcms::MoxCms` through a `JxlCms` adapter. `CMYK_ORACLE=<cmyk_layers.jxl>`
instead checks both moxcms paths against lcms2 values pinned in the decoder's unit test.

It builds outside the repo, with a `Cargo.toml` that depends on this tree, `zenjxl-decoder` and `zenjxl-decoder-cli`, and an archmage cap
in the decoder raised to admit 0.9.31-beta.2. It unifies on moxcms 0.9.1.

## Approved adapter defaults, revalidated 2026-11-06

The executable harness now sends structured RGB PQ to `PluggableCms` as CICP.
Known HLG in structured profiles or an ICC CICP tag returns a CMS error before
construction because this adapter has no OOTF. The historical `exact/hdr_class.rs`
is unchanged so its original comparisons remain reproducible.

Fresh complete 1024x1024 PQ decodes against djxl PFMs differ by at
most **0.022236** clamped eight-bit codes for sRGB and **0.052848** for P3.
The original adapter fails the same sRGB image check at **210.454285** codes.
Both HLG decode targets return the explicit OOTF refusal. Two route tests and
strict harness Clippy also pass. The independent-image check uses a newly added
0.1-code bound; no existing numerical expectation changed. It is caller-admitted
with explicit fixture/reference directories and fails if either is missing.
The two reference PFMs were regenerated with independent `djxl v0.12.0 4128790`
and are byte-identical to the frozen PFMs. These are clamped display comparisons,
not equality of extended-range floats.
The earlier classification was also rerun: CICP versus analytic rows is 0.05/0.02
codes for PQ and 56.40/56.36 for HLG without OOTF. Those are different comparisons.

`prepare.py PIXELS DECODER NEW_OUTPUT` creates a publish=false manifest and
records hashes of the supplied sources. It never modifies either checkout.
The measured graph uses historical pixels `04cc06f5`, moxcms 0.9.1 and a decoder
experiment copy whose cap already admits beta.2. It does not qualify HDR PR 120,
the current native-codec cohort or a production zenjxl PluggableCms API.
The current cohort's resolver blockers are in
[codec-cohort/README.md](../codec-cohort/README.md).

`just jxl-cms-render-references DJXL FIXTURES NEW_OUTPUT` regenerates sRGB and
P3 references. JPEG XL spells P3 primaries `DCI` in the color description:
`RGB_D65_DCI_Rel_SRG` ([libjxl parser](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/extras/dec/color_description.cc)).

From the zenpixels root, run `just jxl-cms-adapter-check MANIFEST`, then
`just jxl-cms-reference-check MANIFEST FIXTURES REFERENCES`. The references are
`hdr_pq_test_SRG.pfm` and `hdr_pq_test_P3.pfm`; fixtures include
`hdr_pq_test.jxl` and `hdr_hlg_test.jxl`. Preserve the generated lockfile with
run artifacts. Resource wrapper for the final two gates: 3 seconds, peak RSS
0.44 GiB, minimum available 18423 MiB, peak load 0.20; these are validation
process measurements rather than codec performance or memory claims.

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
