# Rust LCMS backend and profile triage

`cms-lcms` exposes `zenpixels_convert::Lcms` alongside `cms-moxcms` /
`MoxCms`. Both implement `PluggableCms`, and `cms-contract` now exercises the
production implementations. Callers explicitly select a backend through
`RowConverter::new_explicit_with_cms`. Enabling both features changes no default
selection, built-in conversion kernel, bundled profile or existing expectation.

The port is pinned to `imazen/lcms` revision
`39068cc62f470d183e078637e19504ac9ecf7392`, version 0.0.0. It links no C library.
A released port dependency is required before publishing this feature. This
qualification does not adopt the port's later SIMD revisions or claim a speed,
size or cross-platform accuracy advantage.

## Backend contract

Native U8/U16/F32 RGB/RGBA, gray/gray-alpha and ICC-described CMYK inputs are
supported; byte BGR/BGRA are supported too. CMYK integer destinations are
accepted, but F32 CMYK destinations are refused pending qualification. Mixed
depths, F16, padding layouts, Oklab and generic layouts decline to the planner.
An unsupported CICP matrix/range returns an explicit error. CICP synthesis uses
the same selected profile database as the rest of the converter. Without
`icc-db`, unavailable CICP profiles decline; sRGB has a virtual profile.

The backend uses relative intent 1, adaptation state 1, no optimization/cache,
and black-point compensation for every CMYK source depth. Prepared float CMYK
rows reuse fallibly reserved scratch. Checked row extents and prepared maximums
fail before writes; existing alpha is copied and new alpha is opaque. Conversion
policy remains with the planner. The backend does not add HLG OOTF support to
codecs that currently refuse HLG.

Linux aarch64 qualification:

- 13 shared contracts pass, retaining two existing opt-in ignores.
- Eight LCMS-specific contracts pass: bounds, preparation, zero allocations in
  prepared float CMYK rows, padding, BGR order, unaligned float alpha payloads and native CMYK16.
- Existing MoxCMS tests pass: 17 integration and 21 accuracy tests, including
  the extended-F32 ProPhoto sign/range regressions.
- LCMS alone without default features and unchanged no-default core both check;
  LCMS also checks on the declared Rust 1.89 minimum.
- Nine doctests and public API regeneration pass; strict backend and shared
  contract/example Clippy and format checks pass.
- The full converter library passes 450 tests, retaining one existing ignore
  and the known stable-upgrade bundle equality failure; no binary is replaced.
- The actual production backend reproduces all 12 C LCMS 2.20 float components
  for four existing JXL CMYK regression samples, including negative red.

[CI run 42442148542](https://github.com/imazen/zenpixels/actions/runs/42442148542)
on code revision `a2bcf8ea` passes all eight LCMS contracts on Linux x64/ARM64,
Windows x64/ARM64, macOS ARM64 and i686 via cross. Shared contracts, strict
Clippy, public API snapshots, MSRV, WASM and formatting jobs pass. The whole
workspace and coverage retain the stable-upgrade fresh-bundle failures. The
first CI run additionally found two Rust 1.99.1 Clippy chunk-iterator lints
missed by local 1.97.1; fixed-array chunks correct those without changing row
arithmetic or output expectations. Original CI logs remain preserved.

Commands: `just cms-lcms-check`, `just api-doc-check`, and
`just cms-rust-cmyk-oracle-check PROFILE C_ORACLE_TSV`. Full platform CI remains
a separate gate; passing local contracts is not universal ICC qualification.

## Candidate comparison

The committed bundle is compared with the separate MoxCMS 0.9.2 proposal, a
controlled LCMS candidate and an equal-input MoxCMS candidate. The LCMS candidate constructs RGB colorimetry from the exact
H.273 Table 2 xy coordinates through the pinned Rust port, then copies all three
serialized TRCs and `cicp` verbatim from the proposal. All 175 gray profiles are
copied unchanged. No candidate is written into the product bundle by these tools.

The independent f64 determinant/Cramer reference uses H.273 primaries and full
Bradford adaptation to ICC PCS D50 (0.9642, 1, 0.8249). That reference does not
call either CMS's colorimetry helpers. The existing product tests independently
use Gaussian elimination. [ICC's white-point explanation](https://www.color.org/whyd50/)
describes the D50 display-profile media white and native-white-to-PCS `chad`.

Maximum absolute component errors across all 159 RGB profiles:

| Candidate | Colorants versus H.273 + Bradford | `chad` × native white versus PCS D50 | Missing `wtpt` |
|---|---:|---:|---:|
| Committed | 0.0002837666004208428 | 0.363339085523388 | 159 |
| MoxCMS 0.9.2 proposal | 0.00002553651116043998 | 0.0001923968375997509 | 0 |
| Controlled LCMS | 0.000007570372702658812 | 0.000008243196713553225 | 0 |
| MoxCMS with the same exact H.273 inputs | 0.000007570372702658812 | 0.000008243196713553225 | 0 |

LCMS colorants fit within half an s15Fixed16 storage step of the reference.
All three regenerated candidates' colorant sums miss PCS D50 by at most 0.0000152587890625;
`wtpt` storage differs from exact PCS D50 by at most 0.0000053955078125333955.
Every TRC/CICP payload remains identical in all four candidates.

The committed `chad` contains Bradford cone-response coefficients instead of
the full adaptation transform. MoxCMS 0.9.2 corrects that structure and adds
D50 media white, but its CICP D65 coordinate is (0.31272, 0.32903), while H.273
specifies (0.3127, 0.3290). The controlled LCMS constructor uses the latter.
The equal-input control supplies the same H.273 coordinates to MoxCMS through
`update_rgb_colorimetry`. All 795 serialized `rXYZ/gXYZ/bXYZ/chad/wtpt` tags
across 159 RGB profiles then match LCMS byte-for-byte. Independent C LCMS finds
zero changed XYZ components in all 43,248 decoded comparison cells between
those two exact-input candidates, including neutrals, endpoints, PQ/HLG and
extended inputs. The improvement is the input definition, not an inherent
constructor-accuracy advantage for one CMS.

## Decoded color controls

C LCMS 2.20 and the Rust port agree in all 259,488 float64 XYZ components of the
43,248-cell MoxCMS-proposal versus LCMS-candidate comparison. This covers all
159 RGB profiles, 136 RGB inputs per profile, relative intent/adaptation 1 and
absolute intent/adaptation 0. Inputs include a 125-point unit cube, neutrals,
endpoints and seven extended cases. This is same-host Linux aarch64 evidence,
not all-platform or all-profile equivalence.

The maximum proposal-to-LCMS change is 0.00004768298822455108 XYZ for relative
intent and 0.0008277766755782068 for absolute intent. The latter occurs at
(1.5,1.5,1.5) through `rgb-22-5.icc`. Native-white error drops from
0.00026225190600892567 to 0.00001996748414967442. Relative white has the same
0.0000152587890625 maximum in both. PQ and HLG comparison cells are present,
but preserved curve payloads cannot independently establish their correctness.
All 334 candidate profiles pass exact libpng ICC roundtrip and C LCMS parsing;
a deliberately invalid PCS header is rejected. This establishes carriage and
parseability, separately from color accuracy.

## Trust and adoption recommendation

Prefer the shared exact-H.273 colorimetric payload for the next bundle
proposal. LCMS generation and MoxCMS with explicit H.273 coordinates produce
identical colorimetry; either can supply it. Keep both as independent generator
controls and gate the bytes against the standards-derived reference. A narrow
MoxCMS generator input correction can preserve the existing generator route
while producing the same colorimetric payload as LCMS. Keep historical recognition hashes and treat any changed
established pixel expectation as an owner decision. Preserve the committed
bundle and the existing MoxCMS proposal as separate reviewable candidates.

Do not adopt all inherited curves as a precision reference. The new dense
read-back evaluates every normalized U16 input against the existing independent
H.273 analytic reference, with no changed tolerance:

| Transfer | Maximum absolute normalized linear error | Finding |
|---|---:|---|
| Gamma 2.2 (4) | 0.00013066220674862183 | u8Fixed8 exponent precision |
| Gamma 2.8 (5) | 0.00010263061578807964 | u8Fixed8 exponent precision |
| Log (9) | 0.004917139850733618 | Largest error at encoded 1/65535 |
| Log sqrt (10) | 0.0015508450894103343 | Largest error at encoded 1/65535 |
| PQ (16) | 0.00001504505573168835 | 3,587 nonzero-reference input codes become zero per profile |
| ST428 (17) | 0.0005306961467401727 | Saturation-knee error; nominal endpoint exceeds normalized LUT range |
| HLG (18) | 0.000010615474624442456 | Neutral 1000-nit/system-gamma-1.2 curve only |

Other measured transfers have at most 0.000007757155883580946 absolute error;
linear (8) is exact on this grid. TC9/10 encoded zero has an interval-valued
inverse and is checked against that interval; the reported errors are at a
strictly positive input. The logarithmic curves and HDR tables each store 1,024
u16 entries. PQ toe precision loss, ST428 clipping and a per-channel HLG curve
are unsuitable grounds for claiming arbitrary HDR/extended-range fidelity.
The dense curve check characterizes normalized inputs; extended inputs are
covered only by the decoded-color comparison, without an analytic guarantee.

The immediate decision is therefore the RGB colorimetry source. A universal
profile-trust claim additionally needs separate curve work, explicit HDR
normalization/OOTF policy, and the relevant corpus/output comparisons. None of the
constructor candidates changes those inherited limits. Keep MoxCMS's sign/range guards.

## Reproduction and receipts

- `just icc-lcms-candidates SOURCE_PROFILES FRESH_LCMS_PROFILES`
- `just icc-profile-triage REFERENCE_TSV FRESH_RESULTS old=OLD mox092=PROPOSED lcms=LCMS`
- `just icc-mox-exact-candidates SOURCE_PROFILES FRESH_MOX_EXACT_PROFILES`
- `just icc-profile-curves LCMS_PROFILES`
- `just icc-bundle-color-compare OLD NEW FRESH_RESULTS` for independent C LCMS.
- `just icc-bundle-rust-color-compare OLD NEW FRESH_RESULTS` for the Rust port.

The compact receipt is
[`benchmarks/first_class_lcms_2026-11-08.json`](../benchmarks/first_class_lcms_2026-11-08.json).
Full profiles, per-profile errors, decoded receipts, original failures, reference
binary hashes and resource logs are retained outside git under
`output/zenpixels/first-class-lcms-20261108/`, with a hash-verified archive mirror.
The first candidate tool failed on absent description tags; the corrected tool
copies only required TRC/CICP tags. An initial native-reader build lacked a
system LCMS development package; the passing run builds the pinned C 2.20
reference. Both failing logs remain preserved.
