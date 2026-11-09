# ICC bundle history, expectations and correctness

Audited 2026-11-08 against review head `d0127f85`, the committed assets at
the revisions below, MoxCMS release sources 0.8.1/0.9.1/0.9.2, and the
linked primary standards. This is an investigation; it adopts no bundle,
changes no existing pixel expectation and selects no new default CMS.

There is no fully qualified replacement for every profile yet. The exact-H.273
RGB colorimetry is supported by independent math and equal-input constructors.
The inherited curves need separate precision and HDR semantics work. The gray
display-profile white tags also need correction: byte preservation across the
recent candidates did not establish that they were correct.

## What the assets represent

Three separate resources are involved:

- Compact ICC constants: a few copied, separately maintained profiles.
- The generated CICP bundle: losslessly compressed complete ICC profiles,
  indexed by primaries and transfer. Normal builds decode these stored bytes;
  they do not run the generator or need a CMS to synthesize them.
- ICC identification hash tables: recognition of incoming profile bytes,
  including historical bytes. Recognition is not authority to discard a
  profile's actual curves, LUTs, white or rendering behavior.

For matrix RGB, serialized TRCs decode channel values; the three colorant
columns map linear RGB to D50 PCS XYZ. Those columns are already adapted.
`chad` records the full native-white-to-PCS adaptation, not an extra matrix to
apply again in an ordinary relative transform. Display-class `wtpt` contains
D50, not the unadapted native white. [ICC's explanation](https://www.color.org/whyd50/)
distinguishes these roles. LZ4 is lossless: sampling and numeric encoding,
not bundle compression, cause the curve precision limits below.

## Asset history

| Date | Revision | Change and limit |
|---|---|---|
| 2026-03-28 | [c92d706](https://github.com/imazen/zenpixels/commit/c92d7060c709761f1368e42f0b7088fa5f5f615b) | Copied Compact P3 v4/v2, Adobe v4, Rec.2020 v4 and ProPhoto v4 constants. There was no full CICP bundle. |
| 2026-04-14 | [1dc41a8](https://github.com/imazen/zenpixels/commit/1dc41a8c0b14af909e3170fa24738e95a9fd0482) | Replaced Adobe's constant with the v2 pure-gamma form and removed a universal ProPhoto constant. Incoming ICC variants must retain their actual curves. Current constants are P3 v4/v2, Adobe v2 and Rec.2020 v4. |
| 2026-06-09 | [1ac281e](https://github.com/imazen/zenpixels/commit/1ac281ee61a439f0677d38c8fcd9ecc84611b58a) | Added synthesis keyed by both primaries and transfer; a primaries-only lookup could attach an SDR curve to HDR samples. Initially used constants plus optional live MoxCMS synthesis. |
| 2026-06-10 | [0d3d7d5](https://github.com/imazen/zenpixels/commit/0d3d7d50b6424e61f70746e78c566c841ce8b65c) | Precomputed 174 RGB mappings using locked MoxCMS 0.8.1; 28,209-byte initial blob. Its constructors supplied colorants, adaptation and curves. |
| 2026-06-10 | [d6ac448](https://github.com/imazen/zenpixels/commit/d6ac448cf531a6e57057e9c3d9bc98881ae5afe7) | Added SHA pin, fresh-generation equality, grid counts and compression regeneration checks. Zeroed timestamps for repeatability; blob became 28,001 bytes. These are reproducibility expectations. |
| 2026-06-10 | [b12cbaf](https://github.com/imazen/zenpixels/commit/b12cbaf) and [2a45352](https://github.com/imazen/zenpixels/commit/2a45352abf37150a9fe75c22fef83dc5ed723ffb) | Switched encoding to pure-Rust LZ4. i686 produced different compressed bytes for identical decoded content, so compressed-byte equality became x86_64-only; decoded equality remained cross-platform. |
| 2026-06-10 | [75eb953](https://github.com/imazen/zenpixels/commit/75eb953e048906278e04d5f4e83dc4c56acf6747) | Added 174 gray mappings, sharing the RGB transfer curves; blob became 35,967 bytes. Gray explicitly computed a full adaptation rather than the constructor's Bradford basis. It also wrote native white into Display-class `wtpt`, the defect identified in this audit. |
| 2026-07-22 | [6c64c89](https://github.com/imazen/zenpixels/commit/6c64c895e5cb138a8d1d9678b6ff87e32e89811e) | Moved to MoxCMS 0.9.0 and renamed white-point APIs. No bundle asset change in this commit. |
| 2026-10-15 | [27c17a6](https://github.com/imazen/zenpixels/commit/27c17a6b441fb58b47756619b793811a976d10c1) and [b8374b9](https://github.com/imazen/zenpixels/commit/b8374b9f1cdefbe5ee76dbceb5ab89d3d07243f6) | Added independent matrix/curve read-back. Excluded 16 singular XYZ-primary RGB profiles and restored a real BT.709 transfer profile instead of omitting it as sRGB. Result: 159 RGB mappings, 175 gray mappings, 35,298 bytes. |
| 2026-11-05 | [b6b12bb](https://github.com/imazen/zenpixels/commit/b6b12bbe702e7ca2f93db559858ae1f233a066ec) | Repacked tags on four-byte boundaries and fixed the PCS header illuminant. Preserved tag payloads; blob became 35,379 bytes. This repaired reader warnings, not the inherited `chad`, gray `wtpt` or curves. |
| 2026-11-07 | [separate proposal](https://github.com/imazen/zenpixels/pull/123) | MoxCMS 0.9.2 regeneration produces 37,260 bytes. All 159 RGB profiles change colorimetry/adaptation and gain D50 `wtpt`; all 175 gray mappings and curve payloads remain unchanged. |
| 2026-11-08 | [first-class LCMS review](https://github.com/imazen/zenpixels/pull/125) | Compared both constructors with explicit equal H.273 inputs. All 795 RGB colorimetry tags match exactly; all 43,248 C-decoded comparison cells agree. Added dense curve characterization; this audit additionally identifies gray `wtpt`. |

The Compact P3/BT.2020 constants use compatibility-rebalanced colorants.
They did not supply the generated bundle's bad adaptation metadata. The
[separate overview](color-correctness-and-compatibility.md) explains negative
PCS Z and the historical versus current ICC wording.

The RGB adaptation issue was present in released MoxCMS 0.8.1:
[`basic_rgb_profile`](https://github.com/awxkee/moxcms/blob/c4affa196727d17f21a7f6f148a558867decb470/src/defaults.rs)
stored `BRADFORD_D` directly. 0.9.1 retained that initialization as
`Matrix3d::bradford()`. A Bradford basis B is not the adaptation
`B^-1 diag((B W_D50)/(B W_native)) B`. The latter must map native white
to PCS D50. MoxCMS 0.9.2 supplies the full transform and D50 media white.
Wrong `chad` can escape ordinary relative pixel tests because those tests
primarily exercise the already-adapted colorants.

## Why specify H.273 coordinates explicitly

[H.273](https://www.itu.int/rec/T-REC-H.273-202407-I/en) defines the
chromaticities associated with its code points. Its D65 rows use
`(0.3127, 0.3290)`. MoxCMS 0.9.2's CICP mapping uses the shared
[`Chromaticity::D65`](https://github.com/awxkee/moxcms/blob/7678ef70f6a7abcf5df7492dd76af3ebb3e85e53/src/chromaticity.rs)
constant `(0.31272, 0.32903)`. More digits do not make that value the
definition of the H.273 code point.

The generic illuminant is not universally invalid. It is a different input
for this specific canonical synthesis contract. Deriving the RGB matrix
requires solving for primary weights that reproduce the chosen white;
changing white changes those weights and the adaptation. The resulting
differences can survive ICC fixed-point quantization.

Our exact-input MoxCMS control supplies the H.273 white and primary values
through `update_rgb_colorimetry`, retaining original TRC/CICP payloads.
The LCMS constructor receives the same coordinates. Both then produce the
same 795 `rXYZ/gXYZ/bXYZ/chad/wtpt` payloads. Their maximum colorant error
against the independent reference is 0.000007570372702658812, within half
an s15Fixed16 step. The default 0.9.2 proposal's maximum is
0.00002553651116043998; the committed bundle's is 0.0002837666004208428.

This is why the generator currently needs explicit inputs. A CICP
constructor implementing the exact table would remove that need. Normal
callers should not have to reconstruct standard coordinates. The target
PCS white is separately ICC's `(0.9642, 1, 0.8249)`, not a fresh daylight
temperature approximation. H.273 also has C, equal-energy and DCI white
rows; hard-coding D65 for every primaries code would be wrong.

## Where the curve choices came from

MoxCMS supplied these encodings when the bundle was introduced. Comparing
released 0.8.1 and 0.9.2 `cicp.rs` shows the same transfer-construction
arms. Neither our header repair nor the recent RGB candidate comparison
replaced the inherited TRC payloads. LCMS candidates deliberately copy
them unchanged to isolate colorimetry; agreement there is not a new
curve-accuracy endorsement.

Stored curve representation and runtime evaluation are different layers.
The separate November pure-gamma investigation reproduced a ProPhoto F32
zero channel making ordinary `(0.5, 0.5, 0)` clip to `(1, 1, 0)` through
an invalid fast-power result. The wrapper's guarded path gives
`(0.6461351, 0.57291955, 0)`. That is an evaluator-domain defect, not
quantization or a competing definition of D65. Retain the sign/range
guards; MoxCMS 0.9.2 alone did not qualify their removal. See the
[preserved reproduction](../benchmarks/moxcms_review_2026-11-06.md).

| Transfers | Stored representation | Consequence |
|---|---|---|
| Gamma 2.2 / 2.8 | One-entry `curv`, exponent in u8Fixed8 | The stored exponents are 2.19921875 / 2.80078125. Valid ICC encoding, approximate H.273 gamma. Adobe RGB's separately defined 563/256 exponent must not be confused with an exact 2.2 requirement. |
| BT.709 family / sRGB | Parametric curve with fixed-point coefficients | Encodes the intended piecewise function approximately, without a sampled table. BT.709 and sRGB must remain distinct. |
| SMPTE240, log/log-sqrt, extended SDR, PQ, ST428, HLG | 1,024 equally spaced input samples with u16 output | Interpolation, output quantization and bounded range limit precision. Extra input samples alone cannot remove u16 output quantization. |

ICC allows a higher-precision parametric pure gamma; the coarser one-entry
encoding was an implementation choice, not a mandate to approximate every
gamma to 1/256. The relevant encodings are in
[ICC.1:2022](https://www.color.org/specifications/ICC.1-2022-05.pdf),
10.6 and 10.18.

The dense normalized-U16 diagnostic found:

- TC9/10 maxima 0.004917139850733618 / 0.0015508450894103343 at
  encoded 1/65535. Encoded zero represents an interval of scene intensities;
  the generator chooses its midpoint. Interpolating from that midpoint to
  the first positive sample spreads the discontinuity over positive inputs.
  Choosing the interval's upper endpoint would make the analytic right-hand
  limit continuous, but that is a separately reviewed curve change.
- PQ: 3,587 positive-reference codes become zero per profile; maximum
  normalized error is 0.00001504505573168835. This is loss of dark detail
  from representation, not a failed power evaluation. It is not a claim
  that all 3,587 input codes are equally spaced luminances or visibly distinct.
- ST428: the recorded 0.0005306961467401727 error is measured against an
  already-clamped reference. The unclipped normative endpoint is
  52.37/48, greater than 1; the table stores at most 1. The diagnostic's
  small error therefore excludes the full endpoint clipping loss.
- HLG: 0.000010615474624442456 error measures only the neutral 1000-nit,
  gamma-1.2 curve. That stored per-channel curve is not general RGB HLG
  display rendering and is not simply inverse HLG OETF.

Correct HLG scene decoding uses inverse OETF. Display rendering additionally
uses a luminance-dependent OOTF. For zero black lift, its RGB scaling is
`Lw * Ys^(gamma-1) * RGB_scene`, with common luminance Ys, not independent
`channel^gamma`. These coincide for neutral samples, not arbitrary colors.
[BT.2100-3 Table 5](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-3-202502-I!!PDF-E.pdf)
defines that distinction, display conditions and black lift. A per-channel
matrix/TRC profile cannot reproduce the general cross-channel operation.

PQ needs its absolute luminance scale preserved; a normalized table alone
does not implement HDR-to-SDR rendering or select SDR reference white.
Higher-precision/range-capable profile structures or native CICP processing
need explicit evaluation for these cases. Changing the CMS cannot restore
information already absent from the profile bytes.

## Where the test expectations came from

| Expectation | Origin | What it establishes and misses |
|---|---|---|
| Blob SHA and fresh MoxCMS byte equality | June 10, `d6ac448` | Detects asset drift and dependency changes. A generator and its stored output can share a definition error. |
| Decoded-content equality across architectures | June 10, `2a45352` | Same profile bytes despite different valid compressed streams. |
| RGB determinant >0.01 and colorants within 0.001 | October 15, `27c17a6` / `b8374b9` | Independent H.273 math catches rank collapse and wrong gamuts. The 0.001 engineering tolerance is not an ICC or H.273 permission to drift, nor a check of `chad`. |
| TRCs within 0.0005 SDR / 0.001 PQ-HLG | October 15, `27c17a6` | Absolute linear-light error at 12 chosen input samples, not all U16 inputs, relative dark precision or complete HDR appearance. |
| Gray `wtpt` equals native white | October 15, `27c17a6`, following June's generator | Locks an incorrect Display-class tag interpretation; independent math can still test the wrong semantic quantity. |
| Gray `chad` native-white error <0.001 | October 15 fix `b8374b9` | Replaced the new test's 0.00005 bound to accommodate the existing 5003 K target approximation; did not establish exact PCS D50. |
| JXL selected CMYK channels within 0.0013 of LCMS | October 3, [505e701c](https://github.com/imazen/zenjxl-decoder/commit/505e701cace2b72dbc77340b375d1346ae402e06) | Interoperability regression against recorded LCMS 2.18 relative+BPC output. The comment chooses under one-third of an 8-bit code; this is not a normative CMYK accuracy bound. Two channels of one sample are explicitly unchecked because interpolation already differed. |
| Full U16 curve scan | November review `143016dd` | Characterizes all 65,536 normalized inputs without adopting new tolerances; exposes holes between the older 12 samples. |
| 334 libpng/LCMS reader checks | November header repair and subsequent proposals | Structure, carriage and parsing. These readers accepted the gray white-tag inconsistency too. |

The log-zero expectation was corrected to an interval in `b8374b9`, because
the encoding maps an interval to zero. That mathematical ambiguity does not
justify errors at strictly positive encoded values. Existing assertions and
goldens remain unchanged in this audit.

MoxCMS 0.9.2 changing JXL red from -0.06461855 to -0.066765495 crosses
the existing bound around the recorded -0.0645. It is a regression against
that compatibility contract; disagreement with one CMM alone does not prove
which interpolation is mathematically wrong. Preserve the failure, match
intent/BPC/optimization settings, and isolate the change before deciding.
The wrapper's earlier sign reversal and invalid zero-channel powers are
different: those violate the selected numerical operation directly.

## Additional finding: gray display white

Directly decoding every historical bundle with system liblz4 confirms that
the native-white `wtpt` dates to June's gray addition. The current blob has
175 gray mappings, all Display-class, with four native-white groups. None
stores D50 in `wtpt`. The separate 0.9.2 proposal retains all 175.

For committed `gray-1-8.icc`, class is `mntr`, version is 4.4, and `wtpt` is
`[0.9504547119140625, 1, 1.08905029296875]`. Its SHA-256 is
`bb0133c56ae3de4c68971b82e3fb8d140113a621e6bd25844aa9567069d70cd1`.
The same bytes occur in the archived candidate controls. ICC.1:2022 9.2.36
requires PCS illuminant values for Display-class `wtpt`; the native white
belongs in the adaptation definition. Gray TRC evaluation uses PCS white
(Annex F.2), not an arbitrary tint derived from that tag.

The October gray test explicitly expects the wrong native-white value.
Correcting the generator and that expectation is substantive review work,
not accepting a new backend's output as a golden. Existing 0.9.2 and LCMS
candidate reader passes do not waive this defect. We have not measured its
pixel effect for every gray reader/intent in this audit.

The C 2.20 reference's `cmsCreateGrayProfileTHR` also writes its supplied
white directly into `wtpt` and declares Display-class v4.4. Calling that
constructor with D65 therefore does not independently repair the semantic
issue. The [pinned C source](https://github.com/mm2/Little-CMS/blob/7717fce0d85447ef3b2df0cca2e99207c15eeecd/src/cmsvirt.c)
and ICC's requirement answer different questions: C parity verifies the
port; specification conformance requires the correct profile definition.
This finding is a tag-conformance issue, not a claim of measured visible
pixel error for every reader. A replacement gray definition must be
checked against both the normative model and actual consumer behavior.

Reproduce asset inspection with `just icc-bundle-definition-audit REVISION`.
The [compact historical receipt](../benchmarks/bundle_definition_history_2026-11-08.json)
records all seven inspected revisions, sizes, hashes, missing RGB white tags,
gray mismatch counts and actual curve encodings. It uses the stored blobs,
not fresh constructors. See the [candidate triage](first-class-lcms-and-profile-triage.md)
for the independent colorimetry and dense-curve measurements.

## Correct adoption target

Use the intended code-point coordinates, exact PCS white and full Bradford
adaptation, quantized only for the chosen ICC storage. Record Display-class
white tags correctly, including gray. Preserve signed adapted colorants
where the declared specification/encoding supports them; offer compatibility
variants as explicit choices for measured reader needs.

For a new canonical profile retaining negative PCSXYZ, declaring v4.4 makes
the specification basis explicit. Keeping a historical v4.2/v4.3 header
does not change the older wording simply because a current reader accepts
the bytes. Specification-version and target-reader compatibility still need
separate gates; equal colorimetry tags do not imply equal whole profiles.

Choose curve representations from the required precision and range, and
distinguish scene decoding from display rendering. Match engine settings
before comparing numerical output. Keep source-definition math, C/Rust
differential checks, structural readers and actual consumer tests separate.
Neither the old bundle, an LCMS label nor successful regeneration alone
is the definition of correct color.
