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

## Separate ICC bundle proposal: blocked on numerical review

No binary or golden is changed by the dependency candidate. Fresh generation
with 0.9.2 creates a 37,260-byte bundle (SHA-256
`ef6f028fabb3075d4e56e352793c2de65b0e8048ebd6986fed4330c6ae9c2d6f`),
versus the committed 35,379-byte bundle. All 159 RGB profiles change:
rXYZ/gXYZ/bXYZ and chad payloads differ and wtpt is added. All 175 gray
mappings are byte-identical. No TRC tag payload changes were observed.

All 334 proposed profiles pass D50/alignment checks, exact independent libpng
ICC roundtrips and Little CMS parsing; the non-D50 negative control refuses.
Those establish carriage and parseability, not equivalent decoded color.
A separate proposal must compare old/new decoded color with independent
Little CMS controls (neutral/endpoints, HDR and extended-range cases) and
explain the adaptation against intended white/primaries before any owner
approval of changed pixel or golden expectations.

`just icc-bundle-compare PROPOSED_CONVERTER NEW_OUTPUT_DIR` exports both
bundle identities and the changed-tag census without replacing either input.
The comparison tool uses the system LZ4 library only for validation.

## Source references

- [Archmage 0.9.31 release](https://github.com/imazen/archmage/releases/tag/v0.9.31)
- [Moxcms 0.9.2 release](https://github.com/awxkee/moxcms/releases/tag/0.9.2)
- [Released pure-gamma evaluator](https://github.com/awxkee/moxcms/blob/0.9.2/src/trc.rs)
