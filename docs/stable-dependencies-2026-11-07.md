# Released dependency rollout, 2026-11-07

Archmage, archmage-macros and Magetypes 0.9.31 and moxcms 0.9.2 are
published. This review requires the stable SIMD release and the CMS fixes,
without merging the unrelated HDR or planar API proposals.

The existing SIMD-only baseline passes 728 library tests with two existing
ignores, formatting and strict workspace-library Clippy. Its broad CMS range
resolves moxcms 0.8.1 after Cargo deduplication; that is not a 0.9.2 result.
Requiring 0.9.2 passes 727 library tests and fails one existing RGB bundle
fresh-generation equality check, with the same two ignores. Formatting and
strict workspace-library Clippy pass. All 38 existing CMS integration and
accuracy tests pass without changed expectations.

## Cleanup decisions

- Remove temporary upper caps that end at a beta.2 release. Keep the frozen
  consumer source revisions: current main kernels in zenblend, zentone,
  zenflate and zenresize differ from those previously qualified.
- Keep the garb source patch: registry 0.2.8 has the experimental feature,
  but lacks the pinned source's corrected integer narrowing, FMA-corrected
  byte-to-float quotients and widened nearest quantization. Dropping the
  patch would change numerical kernels, not merely dependency resolution.
- Keep the linear-srgb source patch: the precise-portable transfer feature
  is not supplied by the registry 0.6.12 source. Stable SIMD alone does not
  publish those changes.
- Keep the pure-gamma sign/range adapter. The published 0.9.2 evaluator still
  passes samples directly to dirty_powf. `just moxcms-release-probe` reproduces
  ProPhoto (0.5, 0.5, 0) to sRGB as raw (1712.983, 245.75737,
  -2482075000), clipped to (1, 1, 0), with extended F32 enabled.
- Keep the checked ICC serializer and frozen experiment sources/results.
  Timestamp normalization and profile identity remain relevant; historical
  measurements must continue to identify their original dependency sources.
- The expensive U16-through-F32 proposal in PR #116 is a retirement candidate
  because 0.9.2 includes integer gamma rounding. This does not claim all U16
  numerical error is gone or that the proposal's stronger bound is met.
- PNG manifest-only PR #55 is an obsolete beta bridge for stable graphs;
  PNG source PR #56 remains necessary for the public ICC facade migration.
  They must not be merged together merely to select stable dependencies.

## Separate ICC bundle proposal: staged for numerical review

No binary or golden is changed by the dependency candidate. Fresh generation
with 0.9.2 creates a 37,260-byte bundle (SHA-256
`ef6f028fabb3075d4e56e352793c2de65b0e8048ebd6986fed4330c6ae9c2d6f`),
versus the committed 35,379-byte bundle. All 159 RGB profiles change:
rXYZ/gXYZ/bXYZ and chad payloads differ and wtpt is added. All 175 gray
mappings are byte-identical. No TRC tag payload changes were observed.

All 334 proposed profiles pass D50/alignment checks, exact independent libpng
ICC roundtrips and Little CMS parsing; the non-D50 negative control refuses.
Those establish carriage and parseability, not equivalent decoded color.
Independent Little CMS decoded-color controls now cover all 159 RGB profiles:
125 cube points, four additional neutrals and seven extended samples, under
two intents, for 43,248 comparisons. Float RGB inputs feed double XYZ output
with optimization and transform caching disabled. PQ and HLG profile curves
are included; this does not qualify an HLG display OOTF.

| Observation | Relative colorimetric, adaptation 1 | Absolute colorimetric, adaptation 0 |
| --- | ---: | ---: |
| Changed comparisons / 21,624 | 21,215 | 21,215 |
| Largest absolute XYZ difference, unit cube | 0.0003052307283724076 | 0.3718362932477248 |
| Largest absolute XYZ difference, including extended inputs | 0.0009500835403741803 | 1.0975832351687131 |
| Largest old white XYZ error | 0.0002952974069354575 | 0.35254968215519233 |
| Largest new white XYZ error | 0.0000152587890625 | 0.00026225190600892567 |

Relative white is compared against PCS D50 (0.9642, 1, 0.8249); absolute
white is compared against the H.273 source white. The relative unit-cube
maximum is BT.2020 transfer 1 at (1, 0, 1): old XYZ
(0.79855344099996728, 0.32466124805523577, 0.79521181718519074), new
(0.79852292388750357, 0.32464598949900392, 0.79490658645681833).
The absolute unit-cube maximum is primaries 8, transfer 5, at (0, 0.01, 1):
old XYZ (0.25879774968416314, 0.12686979894851902, 0.74799713550874003),
new (0.20145777917036867, 0.068082366624366841, 1.1198334287564649).
PQ and HLG each reach a relative maximum of 0.0003052307283724076 and an
absolute maximum of 0.37183623364398954 on the sampled inputs.

Released `update_rgb_colorimetry_triplet` constructs the RGB colorants from
the requested primaries and source white, then adapts them to D50. Its
`update_chromatic_adaptation` now records a source-white-to-D50 matrix in
chad and D50 in wtpt. This is the intended PCS representation. The observed
improvement in native-white reconstruction supports that correction, but
does not make the old and new profiles equivalent. Relative colorant drift
and residual native-white error remain measured differences, not accepted
replacement tolerances. The owner authorized staging the binary/index/hash change as a separate,
unmerged proposal after reviewing these observations. That authorization
does not establish equivalence or authorize merging the proposal.

Little CMS defaults to observer adaptation 1 even for the absolute intent.
The first capture used that default and is retained as an additional control;
the native-white results above use an explicit adaptation state of 0, per
the [Little CMS API](https://www.littlecms.com/LittleCMS2.18%20API.pdf).

`just icc-bundle-compare PROPOSED_CONVERTER NEW_OUTPUT_DIR` exports both
bundle identities and the changed-tag census without replacing either input.
The comparison tool uses the system LZ4 library only for validation.

## Qualified frozen consumers

These are separate review candidates; the original review heads and main
have not been overwritten. Test groups overlap and are not summed as unique
tests. Existing ignored tests remain unchanged.

- JPEG PR #260: strict full-feature library/test/bench Clippy; release default
  2,469 passes, user 3,037, FFI 2,866, expert 2,869, recompress 35, and both
  independent colorspace oracle tests. The first oracle invocation selected
  the wrong djpeg executable and failed; its receipt is retained, and the
  rerun explicitly selects IJG 9. Frozen HDR 30249993 and linear 2969ee84
  kernels remain in this graph.
- UltraHDR PR #45: strict full-feature Clippy, 435 library/integration tests
  and five doctests pass, including the independent pixel controls. This
  distinct graph retains JPEG 162a1f35 and linear 2969ee84. Genuine no_std
  x86_64-unknown-none and WASM checks each fail at 43 existing float-method
  and type errors. The stable release does not resolve that blocker.
- PNG PR #59: all 23 facade migration controls and strict library/harness
  Clippy pass. No pixel, fixture or assertion changes.
- Core PR #199 and #200 descendants pass the full existing metadata recipe:
  1,464 and 1,453 all-feature tests/doctests respectively, strict all-target
  Clippy, formatting and five host-target no-default-feature variants. Each
  retains 39 existing ignored doctests. Both remain local for the core
  owner's integration; original review heads are unchanged.
- Active standalone tools now select stable SIMD: JzAzBz all-target checks,
  the browser converter's WASM check and the CMS contract pass. The CMS
  contract has 12 enabled tests and two existing ignored profile-corpus
  comparisons; no ignored comparison is claimed as qualified. Historical
  experiment sources and measurements retain their original dependency pins.

## Broader main-based audit

Separate review branches preserve each consumer's current main source.
They are not replacements for the frozen review graphs above. Of 26 root
manifests, 25 resolve released SIMD dependencies; the newer JXL encoder
root requires a missing sibling CVVDP workspace. Legacy MoxCMS 0.7/0.8
families remain where dependencies require them; this does not claim a
single CMS version throughout every graph.

Library tests and strict library Clippy pass on garb (59), linear-srgb
(109), zenblend (94), zentone (141), zenflate (306, one existing ignore),
zenresize (357), zenquant (100), heic (119, explicit std/Rust backend),
butteraugli (92), zenanalyze (458, four existing ignores), zenavif (239,
three existing ignores), zenbitmaps (26), zenraw (87), zenwebp (367, one
existing ignore), zenextras (150), and fast-ssim2 (18, one existing ignore).
These counts describe the library gates, not full corpus/parity coverage.
The main-based pipeline passes workspace checking and strict Clippy for
its std/CMS/JPEG/PNG codec subset; its pending public API migration and
full runtime parity remain separate work. No model training or evaluation
was performed. Explicit equivalent `mul_add_relaxed` spellings remove
stable Magetypes deprecations in three main-based consumers; frozen
numerical sources are preserved separately.

JXL default-feature workspace tests pass 1,247 cases with 28 existing
ignores using bounded internal fixture workers, optimized tests with debug
assertions and overflow checks, and an 8 GiB cap (peak RSS 5.45 GiB).
This does not enable its optional MoxCMS path. The existing CMYK
relative-colorimetric/BPC float test fails on 0.9.2: at reflectance
[0.2347, 0.9601, 0.5918, 0.9813], red is -0.066765495 against frozen LCMS
-0.0645, outside the unchanged 0.0013 bound. Identical source on 0.9.1
passes with -0.06461855; the current explicit LCMS adapter returns
-0.064411186. Its adapter workaround and expectations remain unchanged.
The model crate's main-based source compiles but strict Clippy reports
341 deprecated SIMD calls; scientific qualification is not claimed.

Cavif's default-depth control still confirms 10-bit output, but its
pre-existing omission assertion fails: explicit nclx (1, 13, 6, full range)
is present. The same test fails on archmage 0.9.30 and moxcms 0.9.1 with
all other source/dependencies fixed. This is not a newly introduced stable
release regression; its established expectation awaits owner decision.

The older frozen rav1d cap descendant passes all 206 unchanged library
tests in an optimized test build retaining debug assertions and overflow
checks (peak RSS 1.50 GiB). Its earlier slow debug attempt was interrupted
and preserved, not counted as a successful gate.

## Cleanup boundaries

The separate bundle proposal is draft PR #123, based on dependency PR #122.
Its 728 library tests, 38 CMS controls and all 334 independent reader
controls pass locally; it preserves every historical recognition row. Full
CI is green on head `2dba714d`, including Linux/Windows ARM, macOS ARM,
i686, WASI, MSRV and the API/CMS contracts. The dependency-only PR still
fails the original bundle equality gate: CI also runs the generator, where
two unchanged regeneration checks fail (37,260 fresh bytes versus 35,379
committed). That failure is preserved; PR #123 is the separate proposal.
Changed colorants and white adaptation remain an explicit numerical review.
There is no merge, release, main overwrite or automatic golden acceptance.

One confirmed inactive Cargo target was removed after checking for live
users and unique artifacts; it occupied 980,574,208 allocated bytes before
removal. That is not an exact filesystem-space gain measurement. Historical
benchmark harnesses, assembly, traces, source snapshots and all original
failing receipts remain preserved. Temporary beta caps and redundant
bridges can be removed as the owner integrates the qualified stable graphs;
source patches and old experimental locks remain where they carry numerical
changes or reproducibility provenance.

## Source references

- [Archmage 0.9.31 release](https://github.com/imazen/archmage/releases/tag/v0.9.31)
- [Moxcms 0.9.2 release](https://github.com/awxkee/moxcms/releases/tag/0.9.2)
- [Released pure-gamma evaluator](https://github.com/awxkee/moxcms/blob/0.9.2/src/trc.rs)
- [Released RGB colorimetry and adaptation](https://github.com/awxkee/moxcms/blob/0.9.2/src/profile.rs)
- [H.273 source primaries and whites](https://www.itu.int/rec/T-REC-H.273)
