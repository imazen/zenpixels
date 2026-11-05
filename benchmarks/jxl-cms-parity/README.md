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

`zenjxl-decoder-cli`'s lcms2 wrapper isn't a usable reference for CMYK: it returns 0.968 for no ink, consistent with passing 0..1 where
lcms2's float CMYK takes 0..100. Against the lcms2 values in the decoder's `cmyk_to_srgb_matches_lcms2_relative_bpc`, the adapter and the
wrapper agree, within 0.26 of a code on four colors.

On HDR, moxcms and lcms2 differ from each other far more than the two moxcms paths do. There, neither is a reference for correctness.
