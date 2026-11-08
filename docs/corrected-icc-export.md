# Corrected ICC export qualification

The corrected recipe is staged for review. The committed export blob has not
been replaced. Regeneration must not make a freshness test pass before the
serialized definitions and decoded color are qualified.

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
well as a float-capable path. No corrected PQ blob, fresh-bundle golden
replacement, merge or publication is claimed here.

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
