# RGB ICC regeneration and the Rust LCMS controls

The first bundle repair (`b6b12bbe`) corrected the ICC header's PCS illuminant
and tag alignment while preserving colorimetric tag payloads. The separate,
unmerged moxcms 0.9.2 proposal corrects those payloads: the old RGB `chad`
contains the Bradford cone-response basis, rather than a complete transform
from the source white to D50. For sRGB its first row is approximately
`[0.89510, 0.26640, -0.16141]`. The proposed profile records the full D65-to-D50
adaptation (first row `[1.04787, 0.02292, -0.05019]`) and D50 `wtpt`.
All 159 RGB profiles change; all 175 gray mappings remain byte-identical.
This is a colorimetry proposal, not a second pixel-format or RGB-kernel edit.

The Rust LCMS port was already integrated in `cms-contract` at revision
`39068cc62f470d183e078637e19504ac9ecf7392`. Earlier reports of 12 passing
contracts included that backend, alongside MoxCMS. The older independent
bundle study instead used native C LCMS 2.17. These are distinct controls.

The new comparison runs the same 159 profiles, 136 float RGB samples and
two intents through the pinned Rust port and C LCMS 2.20. Relative intent
uses adaptation state 1; absolute intent explicitly uses 0. Old and proposed
profiles each produce double XYZ, with optimization and caching disabled.
The 43,248 cells contain 259,488 output components: all float64 bits match
between Rust and C on the same ARM64 Linux host. C is built from the port's
unmodified reference revision `7717fce0d85447ef3b2df0cca2e99207c15eeecd`,
with `-O2 -fno-strict-aliasing`. This qualifies these cells, not all port paths
or cross-platform bit identity.

The new controls reproduce the previous study: relative unit-cube maximum
XYZ change is `0.0003052307283724076`. Maximum native-white XYZ error under
absolute intent/adaptation 0 improves from `0.35254968215519233` to
`0.00026225190600892567`. Changed decoded color and residual drift remain;
these observations do not authorize a merge or replacement tolerances.

The added CMYK probe exposed a separate adapter bug. The port's `virt`
module exports `INTENT_RELATIVE_COLORIMETRIC` as 3 in this pinned revision;
the correct constant in `types` is 1. Our adapter consequently requested
absolute intent. A new paper-white regression returns approximately 0.9063
before the correction, rather than the expected relative white 1. After
selecting the `types` constant it passes; all 13 enabled shared contracts
pass, with the same two old ignores. Strict all-target Clippy and formatting
also pass. No existing test bound was changed.

The exact committed JXL fixture's ICC was exported to a fresh path, SHA-256
`4855b8fabb96bdc6495d45d089bb8c8efb1ae18389e0dc9e75a5f701a9c0b662`.
The preserved earlier investigation profile has the same hash. All four
documented CMYK sample vectors then match C LCMS 2.20 in all 12 float32
components. Negative red is `-0.06466543`, and paper white is approximately
`[0.99999994, 1, 1]`. This corrects the Rust contract adapter; it does not
remove the separate JXL/MoxCMS 0.9.2 error or change product CMS defaults.

The C/Rust bundle comparison plus shared-contract checks take 30 seconds
under the resource wrapper, peak RSS 0.25 GiB. The final corrected CMYK and
13-contract run takes six seconds, peak RSS 0.25 GiB. These are validation
resources, not performance comparisons. Earlier failing controls are retained.

Reproduce with `just icc-bundle-rust-color-compare`, the C comparator in
`scripts/icc-gen/validation/compare_color.c`, and
`compare_color_receipts.py` for exact output-bit comparisons. The JXL
`downstream-cmyk-export-profile` recipe exports its own fixture ICC;
`compare_cmyk.c` and `just cms-rust-cmyk-oracle-check` compare the four float
vectors without a tolerance. All generated profiles and full receipts stay
outside git.
