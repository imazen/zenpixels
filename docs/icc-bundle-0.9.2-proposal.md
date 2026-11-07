# Separate RGB ICC bundle proposal

The owner authorized staging this unmerged proposal on 2026-11-07, after
reviewing independent decoded-color observations. The stable dependency
proposal retains the previous binary. This proposal changes its generated
RGB profiles, index, expected binary hash and recognition table.

The new blob is 37,260 bytes, SHA-256
`ef6f028fabb3075d4e56e352793c2de65b0e8048ebd6986fed4330c6ae9c2d6f`,
versus 35,379 bytes, SHA-256
`10aaba9c228d86b47718d5e2f4bea739b336fea808c3fd68d7d40ff7a0f80a2f`.
All 159 RGB profiles change rXYZ/gXYZ/bXYZ and chad and gain wtpt; all 175
gray mappings remain byte-identical. No TRC payload changes were observed.
The H.273 XYZ primaries remain excluded from RGB synthesis.

Moxcms 0.9.2 constructs colorants from the requested primaries/source white,
adapts them to D50, records the source-white-to-D50 matrix in chad, and
records D50 in wtpt. Native-white reconstruction improves in the independent
LCMS controls, while relative-color differences remain. This is an intended
adaptation correction with measured drift, not equivalent color.

[The full numerical study](stable-dependencies-2026-11-07.md) and its
[machine-readable observations](../benchmarks/stable_bundle_color_2026-11-07.json)
cover 43,248 LCMS comparisons: all RGB profiles, cube points, additional
neutrals, endpoints, PQ/HLG curves and extended samples. Inside the unit cube,
maximum absolute XYZ difference is 0.0003052307283724076 for relative intent
and 0.3718362932477248 for absolute intent with observer adaptation disabled.
The maximum native-white XYZ error improves from 0.35254968215519233 to
0.00026225190600892567. HLG display OOTF and corpus-rendering equivalence are
not qualified.

Recognition adds 69 RGB hash entries. A structural comparison retains every
one of the 369 historical RGB rows and 109 historical gray rows exactly;
the gray table is unchanged. Existing recognition and pixel assertions stay
intact. The only expected-value edit is the owner-approved binary SHA-256.

Validation on the stable SIMD/CMS graph:

- Formatting and strict workspace-library Clippy pass.
- All 728 library tests pass; the same two pre-existing ignores remain.
- All six generator controls and all 38 CMS integration/accuracy tests pass.
- All 334 exported profiles pass D50/alignment validation, exact independent
  libpng ICC roundtrips and Little CMS parsing. The non-D50 negative control
  refuses. These reader checks establish carriage/parseability, not equality
  of color.

The serialized generation/check run completed in 86 seconds under run-heavy:
peak RSS 2.17 GiB, minimum available memory 14,554 MiB, peak load 5.15.
Pure-gamma sign/range guards and the extended-F32 ProPhoto regression remain.
No merge or publication is authorized by staging this proposal.
