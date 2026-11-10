# Corrected ICC export qualification

The corrected export bundle contains 149 RGB mappings and 164 gray mappings
(209 unique profiles). HLG is deliberately excluded without an explicit
rendering target. The regenerated asset is 93,377 bytes, SHA256
`67c3122da4cb5cd969a38c7fa0eb0f3ed0c5d4a99f2dac4cacd49500e8e1e509`.
The 0.3 review candidate contains the corrected bundle and qualified compact P3 exports. Existing input ICC bytes are not replaced. Release-ref exposure is recorded in [repository issue completion](repository-issue-completion.md). Historical measurements below describe their stated revisions.

The build-time recipe supplies exact H.273 primary/white chromaticities,
PCS D50 display `wtpt`, full source-white adaptation, s15Fixed16 gamma 2.2/2.8,
the upper inverse interval endpoint at log encoded zero, and peak-normalized
ST428 gamma 2.6 with `lumi` 52.37 cd/m². HLG ICC synthesis refuses without an
explicit rendering target; callers must retain native CICP where permitted.
Existing input ICC bytes remain authoritative. Recognizing an old hash does
not authorize replacing its color transform with corrected export bytes.

Independent tests first failed against the unchanged historical blob (four
failures: RGB colorants, missing RGB media white, gray media white, and ST428
luminance). The corrected build-time recipe then passed independent matrix,
white and adaptation checks and all 65,536 normalized U16 codes per SDR curve.
These checks use the pre-existing absolute linear bound 0.0005; they do not
establish extended-range or complete CMM interoperability. The explicit HLG
refusal also passes. The original failing receipts are retained.

Historical normalized hashes remain present: 242 active RGB and 101 active gray
keys before and after the recognition policy change. Bundle-derived entries
are metadata-only until their complete transform is qualified. The sRGB CMS
shortcut now checks that permission. A 420-byte historical C-white gray/sRGB
profile remains recognized but cannot bypass the CMS or resolve as D65 sRGB.
Other corpus entries and their existing numerical bounds are unchanged.

## PQ representation and backend capabilities

The standalone `just icc-pq-precision RGB GRAY OUTPUT` pilot compares production
backends on all 65,536 normalized U16 inputs. Maximum errors below refer to a
neutral BT.2020 RGB ramp into a matching linear profile, with clipping disabled.
The pilot reports red-channel error for that neutral ramp; it does not claim
a maximum across colored RGB inputs. Relative error is measured only above 0.001 nit (normalized PQ >= 1e-7).

| Representation | Backend | Maximum absolute error | Maximum relative error above 0.001 nit | Positive inputs decoded to zero |
|---|---|---:|---:|---:|
| Existing integer curve | MoxCMS 0.9.2 | 4.344770529e-4 | 1 | 3,588 |
| Existing integer curve | Rust LCMS | 2.216374465e-5 | 1.024437238 | 3,619 |
| Float table, 4,097 entries | Rust LCMS | 7.710026515e-7 | 2.710520570e-4 | 0 |
| Float table, 16,385 entries | Rust LCMS | 1.822038588e-7 | 1.764422041e-5 | 0 |
| Float table, 65,536 entries | Rust LCMS | 3.497650720e-7 | 4.977477857e-7 | 0 |
| Compact float formula | Rust LCMS | 4.202118970e-5 | 4.205552700e-5 | 0 |

These are pilot measurements, not adopted precision budgets. Gray results also
include the pilot destination's white/colorant differences; they must not be
misreported as isolated curve error. Float table precision at arbitrary F32
inputs, colored samples and independent C controls still require qualification.
The integer compound mAB pilot remains inaccurate through MoxCMS (maximum
normalized error 0.009879843441); gray mAB is unsupported by that backend.
The original oversized mAB experiment was rejected by both readers and is
retained as a failed experiment, not counted as a passing representation.

MoxCMS 0.9.2 does not implement the float DToB/BToD pipeline tags. ICC.1:2022
8.10 explicitly permits fallback when those tags are unsupported: this alone
is not a specification violation. The zenpixels adapter declines profiles containing a float pipeline for its
requested relative-colorimetric intent, allowing a configured capable backend to handle them rather than
silently accepting the lower-precision fallback. The shared CMS contract
passes 14 tests with two pre-existing ignores in the initial guard control; the new regression failed
before this capability guard. Both backends remain first-class; no default
backend or frozen LCMS pin is changed.

## The W3C PQ profile is a different, legacy fallback

Reference: [W3C png-hdr-pq profile](https://github.com/w3c/png-hdr-pq/blob/1717d3c6ac3e8524a22581c103a154d6c7af1e9d/icc/ITUR_2100_PQ_FULL.ICC),
13,472 bytes, SHA256
`8a66e7e397f1a833f34b584681857589144a1da01918e2b52447c56bdebd5e4f`.
It is v4.2, Display/RGB/XYZ, with `A2B0` mAB and `B2A0` mBA, no float DToB
pipeline, and `lumi` Y=100. Its compound curve represents a different
normalization from a peak-normalized PQ matrix profile.

The [published W3C note](https://www.w3.org/TR/png-hdr-pq/) explicitly identifies
its ICC and gAMA as inaccurate compatibility fallbacks. Aware implementations
recognize the `iCCP` profile name to recover PQ semantics. The note is deprecated
for new applications in favor of PNG 3 native `cICP`, e.g. `9 16 0 1` for
BT.2100 PQ. The profile's existence does not qualify arbitrary generated PQ
TRCs or prove every CMM preserves its HDR range.

`just icc-pq-external PROFILE` sends this actual ICC through both production
backends into the same linear BT.2020 ICC with clipping disabled:

| Neutral encoded input | MoxCMS red | Rust LCMS red |
|---|---:|---:|
| 0 | -0.0034956122 | -0.000015259322 |
| 0.5 | 0.92261475 | 0.92237294 |
| 0.75 | 1.0000353 | 9.833156 |
| 1 | 1.0000353 | 100.00047 |

The output uses this profile's 100-nit-relative convention: PQ code 1 denotes
10,000 nits, so the unbounded relative result is approximately 100. C LCMS 2.20
independently produces relative PCS Y=99.998504638206214 at code 1 and
0.92235475025063351 at code 0.5. Destination quantization/adaptation accounts
for small differences between XYZ and RGB controls; those values are not
bit-parity claims. The production MoxCMS path loses highlights in this control.
Its F32 API documents finite table precision and extended output, while its
analytic extended option is specifically for matrix-shaper profiles, not
arbitrary mAB pipelines. Attribution of the black offset and highlight limit
must separate profile fallback semantics, internal resampling, black-point
compensation and engine capabilities; CMM disagreement alone is not proof that
one violates the ICC standard. ICC.1:2022 10.12.5 explicitly bounds/clips the
mAB matrix result before its B curves. LCMS's unbounded float execution of this
legacy profile is therefore an extended evaluation behavior, not by itself a
normative oracle for all HDR-range clipping decisions. The legacy profile uses
values beyond the standard mAB intermediate range. A float DToB pipeline avoids
that particular bounded-stage ambiguity.

## Source identity and limits

Pilots use moxcms 0.9.2 and the frozen Rust LCMS revision
`39068cc62f470d183e078637e19504ac9ecf7392`; C oracle is LCMS 2.20 at
`7717fce0d85447ef3b2df0cca2e99207c15eeecd`. Exact commands are just recipes.
No benchmark speed claim is made. ARM resource receipts: common recipe 7s /
0.42 GiB peak RSS; shared CMS contract 5s / 0.29 GiB; PQ table-size pilot 8s /
0.23 GiB; W3C production probe 3s / 0.27 GiB. Every run used run-heavy.

PQ export should carry native CICP and a qualified ICC companion where the
container permits both. CICP-only does not meet the compatibility goal. The
remaining representation work must qualify the conventional ICC fallback as
well as a float-capable path. The final candidate described below replaces the export blob only after
independent definition and numerical qualification. No merge or publication
is implied.

## PNG reader capability witnesses

`just icc-pq-png-readers ORIGINAL FLOAT EXTERNAL OUTPUT` writes bounded PNG16
witnesses and reads them through Playwright into an sRGB U8 canvas. It keeps
the input PNGs, ICCs, exact output channels and hashes outside the source tree.
These checks measure decoder/CMM choices, not HDR display accuracy or a
population-wide support rate. The [PNG specification](https://www.w3.org/TR/png-3/)
allows CICP beside ICC and gives the highest-precedence understood color chunk
authority.

The float profile has the pilot's 16,385-entry PQ DToB0/DToB1 tables. A
deliberately contradictory diagnostic changes only its conventional RGB TRCs
to linear; another removes the float tags. Embedded ICC `cicp` is removed from
these controls so it cannot conceal which transform is evaluated. The
diagnostic profiles are not export candidates. C LCMS 2.20 confirms identical
relative XYZ for the original float pipeline and its changed fallback on all
136 inputs (408 output channels, including colored and extended samples).
The absolute-intent control differs as expected: no DToB3 was supplied.

On 60 input RGB triplets, Chromium 153.0.8010.12 and Firefox 155.0 with explicit
color-management preferences both use the conventional ICC fallback: the
diagnostic changes 120 of 180 RGB output channels, with a maximum difference
of 226 U8 codes. Its output is identical with and without the float tags.
This establishes that accurate LCMS float-pipeline measurements alone do not
qualify these readers. The ordinary PQ fallback and float-table candidate
produce identical U8 canvas outputs here; that is not a dense precision test.

With a PNG `cICP` chunk, both readers' dual-tagged witnesses equal their
CICP-only controls despite the different ICC fallback. Their rendering
normalizations differ: for the neutral sample 32768/65535, Chromium produces
131 and Firefox with color management produces 24; the ICC-only peak-normalized
candidate produces 24 in both. The W3C 100-nit-relative legacy ICC produces
246 and 247 respectively. These are observations of distinct rendering paths,
not an adopted numerical error budget or proof that one display rendering is
correct. Chromium also changes its output when `cicp` is embedded inside the
ICC; Firefox's enabled control does not in this witness.

The unmodified headless Firefox configuration produces the same samples as
untagged PNG for every case, so it supplies no positive color-management
coverage. The explicitly enabled control sets `gfx.color_management.mode=1`,
`enablev4=true` and `force_srgb=true`; it is not described as Firefox's stock
configuration. WebKit initially did not launch because runtime libraries were
missing. Those failures are preserved. A subsequent isolated WPE WebKit 26.6
run supplied four extracted Ubuntu runtime packages without changing the host
install. It honors the float pipeline: changing the fallback changes zero
output channels, while removing the float tags changes 120 channels, maximum
226 U8 codes. In this Linux WPE configuration, PNG CICP does not override ICC;
adding CICP changes zero channels of the linear-only ICC control. This is not
a Safari/macOS qualification.

All three requested engines now have evaluated configurations, with these
headless/runtime limits. The original receipt retains the initial two-engine
coverage; `benchmarks/pq_png_readers_webkit_2026-11-08.json` adds the WPE run and
runtime identities. The Chromium/Firefox run took 5 s, peak RSS 0.42 GiB; the
WPE run took 1 s, peak RSS 0.15 GiB, both under run-heavy.

The compound integer pilots also read in all three enabled engines. Chromium's
verified float16 canvas exposes shadow differences hidden by U8 readback: at
input 1024/65535 the old TRC yields zero, while the root-cube and root-eighth
compound profiles both yield sRGB 0.0000069141387939453125. Float16 output is
quantized and this is a bounded witness, not an exhaustive precision claim.
Firefox and WPE did not supply the verified float16 context required by this
probe; those requests failed explicitly. The U8 controls and original failures
remain recorded in `benchmarks/pq_png_compound_2026-11-08.json`.

Combining 4,097-entry float tables with 4,097-entry compound conventional
tables preserves the reader choices in the hybrid pilot: Chromium and enabled
Firefox match the conventional control; Linux WPE matches the float control.
C LCMS 2.20's relative XYZ also matches the float-only control exactly on 136
inputs. Absolute intent differs (405 of 408 channels, maximum
0.00003671590093290433), as no absolute float pipeline is supplied. This
relative result does not qualify inverse transforms or all rendering intents.

The smaller root-eighth conventional table has maximum red-channel neutral
error 0.0001472459918 through Rust LCMS over all U16 inputs, relative error
0.0008368297694 above 0.001 nit, and zero positive inputs rounded to zero.
The root-cube alternative has lower maximum absolute error 0.00007860331726
but larger maximum relative error 0.008997117382. MoxCMS still has substantial
compound-pipeline error; the float capability refusal remains necessary.
Strict standalone all-target Clippy passes. These are measurements, not newly
accepted thresholds.

For this one BT.2020 profile, the raw hybrid ICC is 155,404 bytes and its
PNG-compressed ICC payload is 27,094 bytes (root-eighth), versus 6,532/1,803
bytes for the old curve. Full-bundle size has not been measured, and these
figures must not be extrapolated across all profiles. The hybrid remains a
representation candidate: physical luminance conventions, reverse transforms,
gray/native codec controls and complete serialized definitions still need
qualification before export adoption. See
`benchmarks/pq_png_hybrid_2026-11-08.json` for exact identities and limits.

## Intent-scoped capability guard qualification

The final guard checks source DToB1 and destination BToD1 for the requested
relative-colorimetric intent. A float transform for perceptual, saturation or
absolute intent alone does not disable a valid relative transform. New controls
compare that behavior bit-for-bit with the unchanged profile in both engines;
the initial broad refusal failed this regression. The final shared contract
passes 15 tests with two existing ignores, strict standalone all-target Clippy passes,
and the unchanged 17 MoxCMS integration, 21 MoxCMS accuracy and 8 LCMS production controls
pass. The dense recipe, HLG refusal and historical recognition checks also pass.
Source `ed984816`: 27 s, run-heavy peak RSS 0.38 GiB. See the committed
`benchmarks/relative_float_intent_2026-11-08.json` receipt for exact commands.

HLG native-carrier policy is also prepared separately for the core owner's
integration: the frozen core199 descendant passes 1,466 workspace tests/doctests,
with 39 existing ignores, five metadata configurations and strict Clippy. Source
ICCs are retained under keeping policies; ICC-only output keeps an explicit
synthesis/refusal obligation. Native codec roundtrips and cohort pin integration
are still outstanding; the pure policy tests do not establish those.

## Final compact export recipe

PQ declares a 10,000 cd/m² `lumi` and peak-normalized linear coordinates.
Forward and inverse float pipelines cover perceptual, relative-colorimetric
and saturation intents. Four small formula stages separate exact binary
coefficients from reciprocal terms to limit cancellation. Their float matrix
uses the actual serialized s15Fixed16 colorants. Constant end segments clamp
encoded excursions outside [0,1]; these profiles do not promise a
sign-preserving extended native-CICP transform.

Conventional AToB/BToA pipelines are also present in both directions. They use
4,097-entry U16 root-eighth curves with separate power stages, retaining
positive neutral and active-channel U16 tones. Absolute intent uses those
colorimetric conventional pipelines and the profile's chromatic adaptation.
Matrix/TRC tags remain a tertiary fallback. Readers supporting only that last
fallback can still lose the deep PQ toe; parser acceptance is not a guarantee
of transform precision. MoxCMS explicitly declines the relative float-pipeline
profile, allowing a configured LCMS backend to process it.

Independent C LCMS 2.20 and the Rust port passed 168/168 PQ cells: all 21
RGB/gray mappings, both directions and four intents. Each cell uses all
65,536 neutral codes; RGB adds individual-channel ramps and a bounded mixed
and extended input grid. Relative-intent analytic linear error was at most
0.000038689759232712895, below the unchanged 0.0005 gate. The largest C/Rust
sample difference was 9.094947017729282e-13. The production Rust adapter
matched direct relative-intent evaluation exactly. Every positive neutral
and active-channel ramp sample remained positive, and those ramps were
monotonic. Very small channels beside bright channels can still round to
zero through float32 XYZ PCS matrix cancellation; scalar counts and worst
witnesses remain in the receipts. These are not a claim of bit-exact arbitrary
RGB roundtrips.

The other 292 profiles passed C/Rust relative-intent comparisons. All 313
exports passed warning-sensitive independent libpng carriage/roundtrip and
C LCMS parsing, with a deliberately non-D50 negative control refused. Valid
creation date, four-byte alignment and deduplicated identical payloads are
part of deterministic serialization.

The absolute-intent differential initially failed four of sixteen BT.2020
pilot cells with frozen Rust LCMS 39068cc6. A four-entry-only descendant,
`55b59476cf64920345e9ca32eb3a4678bf0bbece`, corrects fallback tag selection:
AToB1, DToB3, BToA1 and BToD3 for intent 3. All original inputs pass on this
pin, retaining original failure receipts. No newer numerical kernels were
included. The production CMS contract passed 15 tests with its two existing
measurement/corpus ignores; an initial doctest invocation failed because its
relative TMPDIR resolved under the standalone crate. Repeating with an
absolute scratch path passed.

Final RGB and gray PNG16 witnesses were read by Chromium 153 and Firefox 155
with color management enabled. Both use the conventional fallback; float-only
qualification would not cover them. Both readers' dual-tagged witnesses match
their native-cICP-only controls exactly. RGB Firefox differs from the old ICC
fallback in 10 of 180 output channels, by one U8 code; Chromium RGB and both
gray controls match their old fallback in this bounded witness. These checks
establish decoder behavior, not HDR display or tone-mapping accuracy.

The final numeric PQ sweep used 0.12 GiB peak RSS over 163 seconds; SDR used
0.06 GiB over five seconds, and the 313-profile container gate used 0.03 GiB
over one second. RGB and gray browser probes each used 0.41 GiB over twelve
seconds. These are resource receipts, not performance comparisons.

The Linux WPE WebKit 26.6 RGB witness uses the float ICC pipeline: changing its
conventional fallback changes no output channels, while removing the float
pipeline changes 120/180 channels. This build does not use native PNG cICP in
the RGB control. Its gray controls show no color-management difference even
for a deliberately linear fallback, so grayscale color handling in this
reader remains unqualified. Successful gray decoding alone is not a color
accuracy result. WebKit RGB/gray probes used 0.14/0.13 GiB peak RSS.

At bundle qualification, a separate grayscale-collapse path could trust a
contradictory context CICP when replacing an attached ICC. Its initial guard
retained the malformed-ICC fallback. The owner-directed #134 follow-up
(`11266ff8`) removes that exception: unusable ICCs keep their RGB interpretation
and original context; companion CICP cannot authorize repair. Qualified ICC
replacement and exact alpha/depth reduction controls remain. See the
[current interpretation contract](design-reference.md#current-interpretation-checks-review-150).

## Measured change from the actual historical exports

Independent C LCMS 2.20 compared all 149 retained RGB mappings against the
actual old 35,379-byte bundle, not fresh constructors. Across 136 witnesses per
profile and two intents (40,528 cells), 20,135/20,264 cells change under each
intent. All 313 retained RGB/gray profiles change serialized bytes; the only
removed mappings are the 21 targetless HLG exports. Every actual new bundle
profile is byte-identical to its independently qualified recipe candidate.

For inputs confined to [0,1], the largest relative XYZ coordinate change is
0.04309493109758478 for ST428 at neutral 0.75: old Y=0.51641892163388547,
new Y=0.4733239905363007. That profile now uses peak-normalized gamma 2.6 plus
`lumi` 52.37, rather than 48-nit-relative samples clipped at 1. These raw relative
XYZ numbers use different white normalization; they are not a claim of that
much physical luminance error. Log100's upper inverse endpoint at encoded zero
changes relative XYZ by at most0.004989853356960339 on this grid. The other
transfers change by at most0.001571727449469762 (Log316); the remaining groups
are each below0.000309. All retain their independent analytic bounds.

Absolute XYZ changes are larger because old native-white/adaptation tags were
inconsistent. On the bounded grid, the largest coordinate difference is
0.3720576615205573 for Illuminant-C RGB with Log100 at blue[0,0,1]. Across the
full witness set, endpoint white error drops from 0.35254968215519233 to
0.00004124555198359303 in the absolute control. These are intended definition
corrections, not bitwise equivalence to historical exports. No established
pixel expectation or numerical tolerance was changed to accept them.

The full 168-cell PQ and 292-cell SDR sweeps were replayed using the exact
Git-pinned LCMS 55 consumer binary, after preserving the original path-patched
reader. All 460 C/Rust/production output-hash cells are unchanged; both sweeps
pass. Binary hashes and both sets of receipts are retained in the committed
qualification record. Native PNG qualification freezes pixel source 3ceb0500;
subsequent commits on this review contain documentation and receipt updates.

## Native integer entry and exit formats

The dense C/Rust sweep uses normalized F32 values at every U16 code point.
A separate production-backend test exercises actual RGB/RGBA/gray/gray-alpha
U8 and U16 formats at BT.2020, both PQ decoding and encoding. All 16 layout/
direction cases pass. All 256 U8 codes match the correctly rounded analytic
reference exactly. Across all 65,536 U16 inputs, the largest difference is
two output codes for decoding and one for encoding, within the unchanged
0.0005 linear-light budget. Alpha is byte-exact. MoxCMS declines the unsupported
float pipeline in every case; the LCMS backend evaluates it directly.

The production LCMS target now has 9 passing controls. The focused
`CMS backend qualification` workflow replays that target and the shared
MoxCMS/LCMS contract on five native platforms and i686 through cross, without
repeating unrelated image-processing sweeps. It does not replace full CI.
