# Color correctness and software compatibility

The recent generated-bundle proposal repairs adaptation metadata and pins exact
H.273 inputs. It preserves negative adapted Z values. It does not replace all
Compact ICC profiles or adopt a policy of eliminating negative colorants.

## The negative-Z warning has a specification history

[Compact ICC Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles#display-p3)
provides both mathematically faithful and compatibility variants of P3 and
BT.2020. The latter set adapted red Z to zero and adjust other colorants and
adaptation metadata. This is an intentional trade-off for reader compatibility.

The historical warning is grounded in
[ICC.1:2010, section 10.29](https://www.color.org/specification/ICC1v43_2010-12.pdf#page=94),
which required nonnegative XYZType values. The
[ICC's 2019 technical note](https://archive.color.org/files/technotes/Guidelines_on_the_use_of_negative_PCSXYZ_values.pdf)
explains that adapting physically valid P3/BT.2020 primaries can yield negative
PCSXYZ and recommends preserving those values in signed encodings and accepting
them in CMMs. [ICC.1:2022](https://www.color.org/specifications/ICC.1-2022-05.pdf)
explicitly clarifies negative PCSXYZ (foreword and 6.3.4.2 note 4); 10.31 no longer
contains the old blanket restriction.

This concerns adapted PCS coordinates. It does not make negative physical
spectral radiance valid, make an unsigned encoding signed, or require preserving
out-of-range values in a destination that cannot represent them. Clipping at a
specified encoding boundary differs from clamping a profile's adaptation data.

The version still matters: inspected Compact v4 constants declare 4.2, our
MoxCMS-generated candidates declare 4.3, and the LCMS-generated candidate declares
4.4. The equal-input comparison established identical colorimetry tags, not
identical profile headers or universal support by older readers. The intended
specification version and reader targets remain explicit adoption decisions.

## Two profile sources in this repository

The small Compact ICC constants and the full generated CICP database are separate.
`DISPLAY_P3_V4` is `DisplayP3Compat-v4.icc`; the BT.2020 constant is also a
`Compat` variant. They remain available, including the no-`icc-db` fallback.
Normal CICP synthesis with `icc-db` serves generated profiles instead.

Direct serialized-tag inspection on review revision `80c25870`:

| Profile source | Display P3 red Z | BT.2020 red Z |
|---|---:|---:|
| Compact compatibility constants | 0 | 0 |
| Committed generated bundle | -0.0010528564453125 | -0.001922607421875 |
| MoxCMS 0.9.2 proposal | -0.0010528564453125 | -0.0019378662109375 |
| Either constructor with exact H.273 inputs | -0.0010528564453125 | -0.0019378662109375 |

The samples are P3 with sRGB transfer and BT.2020 with BT.709 transfer. Their
negative Z values survive the recent proposals. The separate old generated-bundle
`chad` defect was a Bradford cone-response basis stored where a full adaptation
transform belongs; a negative matrix coefficient by itself does not diagnose it.

The compatibility variants can also miss another reader's identification
heuristic. Our existing P3 regression compares against libultrahdr's 0.001
colorant tolerance: its expected red Z is -0.00104941, while the compact constant
stores zero and has no `cicp` tag. This explains an interoperability reason to
prefer generated P3 signaling in that path. It is not proof that one profile
works best with every reader.

## The issues and the evidence that decides each one

| Issue | Classification | Appropriate correctness check |
|---|---|---|
| Non-D50 PCS header or unaligned tags | Profile serialization defect | Applicable ICC requirements and warning-sensitive independent readers. The first repair preserved colorimetric payloads. |
| Compact compatibility colorants | Deliberate compatibility trade-off | Compare the intended space and the actual target readers; do not silently treat adjusted colorants as exact H.273 colorimetry. |
| Generated `chad` basis and missing RGB media white | Profile-definition defect | Full native-white-to-PCS adaptation and required metadata, then relative/absolute transform controls. |
| D65 coordinates differ from the requested H.273 definition | Generator-input mismatch | Explicit standard coordinates and independent f64 math. Equal inputs make MoxCMS and LCMS agree in all 795 colorimetry tags. |
| Sampled TRCs, quantized gamma and HDR rendering | Representation precision plus policy | Analytic transfer references, endpoints/toes and declared HDR normalization/OOTF. Sharing curve bytes preserves their limitations. |
| Wrong intent, inconsistent BPC or invalid fast-power results | Adapter/runtime defect | Match settings and evaluate independently. Relative and absolute intent are different operations; a mathematically invalid zero-channel power result is not a compatibility preference. |
| Differences between valid CMM evaluations | Requires classification | Inspect interpolation, rounding, gamut treatment and profile intent. Numerical agreement is useful evidence, but neither an engine's output nor an old golden defines the standard. |

Four questions should therefore be reported independently: does it parse, does
it encode the intended color definition, does the chosen representation meet the
required precision, and does the target software use it as intended?

All 334 reader controls establish profile carriage and parseability. The
standards-derived matrix comparison establishes the selected H.273/Bradford
colorimetry within storage precision. C/Rust LCMS bit agreement tests the port on
those cells. None of those checks alone establishes full HDR appearance, all ICC
profile types, or compatibility with every viewer.

## Recommended policy

Use explicit H.273 coordinates and full adaptation for canonical CICP-generated
profiles, retaining negative PCS values where the chosen signed encoding and
specification permit them. Keep both constructors and independent mathematical
references as gates. Their equal-input payload equivalence permits a narrow
correction to the existing generator; it does not require an engine-based choice.

If a target requires a compatibility variant, make that an explicit output choice
with measured behavior on that reader. Preserve and interpret an incoming ICC's
actual curves, colorants and LUTs; a recognized gamut name is insufficient reason
to replace them. Match intent, BPC, adaptation and clipping before comparing CMS
outputs, and retain established pixel expectations for owner review.

The inherited log, PQ, ST428 and HLG limitations remain separate work. See the
[measured candidate triage](first-class-lcms-and-profile-triage.md) for exact
errors, covered input cells and exclusions. No profile binary, default backend,
rendering policy or pixel expectation changes in this clarification.
