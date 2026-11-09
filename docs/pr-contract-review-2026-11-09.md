# API placement, ICC storage and review chronology

Source-pinned review, 2026-11-09. This proposes follow-ups; it changes no
production API, default feature, bundled bytes or established pixel expectation.

## Approved P3 export direction — 2026-11-09

The owner approved qualifying a compact generated P3 export profile for builds
without the profile database. Until that profile passes native libultrahdr gamut
recognition and independent color qualification, refuse synthesis on the affected
fallback path. Historical profiles remain available for recognition and explicit
compatibility use; recognition alone does not establish color equivalence.

This records the design decision, not a completed qualification or a changed
pixel expectation. Exercise an isolated dependency with default features disabled
so workspace feature unification cannot hide the fallback. Supported release
lines must receive the qualified fix before publication; release preparation is
not publication authorization. Track controls and release exposure in
[issue #135](https://github.com/imazen/zenpixels/issues/135).

## Precision placement (PR120, 79a3040f)

`zenpixels::precision` is public. The converter already re-exports that module
through `pub use zenpixels::*`; individual precision types are not added as
root-level exports. Re-exports do not create a different type identity.

Shared interchange facts such as source significant bits, alpha knowledge and
reported conversion effects have a case for living in core: codec policy can
use them without depending on conversion implementation. Placement of offers
and decision records needs the same actual-consumer test. The presets,
resolver algorithm and `ConversionPolicy::plan_chain` should not enter the
foundational API merely to save an adapter. Prefer conversion-specific policy
in the converter and graph scheduling in the pipeline, retaining only genuinely
shared data contracts in core. `is_lossless` currently checks bits/alpha/range;
it is not a general gamut or pixel-equality proof. Tighten that public promise.

## Bundle footprint and runtime generation (PR128, af19bcda)

Main's bundle is 35,379 bytes. PR128 contains 93,377 bytes, an increase of
57,998 bytes (56.638671875 KiB). The PQ compressed group alone grows from
3,049 to 68,083 bytes. The new bundle contains 149 RGB and 164 gray mappings;
old targetless HLG entries are deliberately absent. These are output recipes,
separate from the historical recognition hash catalog.

`icc-db` is optional but default-on. Disabling defaults disables it; ordinary
RGB sRGB needs no synthesized profile, and a few curated RGB constants remain
available without the database. General gray export needs the database today.

Six recipe/lookup preflights compare synthesized bytes with the bundle exactly.
The release benchmark measures synthesis/serialization, new group allocation
and decode, and already-initialized lookup in-process. No file IO or subprocess
startup is timed. The copied synthesis module is build/test equipment, not a
new runtime API. It currently needs MoxCMS and, for PQ assembly, Rust LCMS.
Enabling an arbitrary CMS backend is not sufficient to generate these recipes.

The retained three-process, twelve-round benchmark has verified CPU10 cpuset
placement but INVALID confirmation due to noisy/flagged resource observations.
Use its means as descriptive observations only, not a qualified speed ratio or
basis for baseline promotion. The shared host had four one-thread training
processes; they were not stopped. An alternative host rejected SSH authentication.

| Profile | Synthesis mean | Fresh group decode mean | Warm lookup mean |
|---|---:|---:|---:|
| P3/sRGB | 2.12 us | 874.8 ns | 28.3 ns |
| BT.2020/BT.2020 | 2.11 us | 760.9 ns | 106.7 ns |
| BT.2020/PQ | 1.49 ms | 27.30 us | 31.7 ns |
| Gray/sRGB | 2.60 us | 685.7 ns | 22.4 ns |
| Gray/PQ | 1.14 ms | 25.12 us | 26.8 ns |
| Gray/log100 | 16.16 us | 1.02 us | 22.4 ns |

A PQ lookup allocates a 627,456-byte decoded group payload, retained in the
process cache; that is payload length, not a measured peak-RSS claim. Individual
RGB and gray PQ export profiles are 54,596 and 20,376 bytes. The full bundle
therefore is not the only possible serialization/cache architecture.

Recommend qualifying on-demand generation and caching of the requested profile,
with a full prebundle as a size/latency choice. A tiny common SDR bundle is an
optional optimization, not obviously necessary at the observed synthesis cost.
Avoid generating the whole grid at startup. Preserve deterministic qualified
recipes and explicitly define CMS-free behavior before changing defaults. PQ
precomputed templates/caches deserve separate consideration because its current
fresh synthesis is costlier. These measurements do not establish resulting
binary size or generator heap peaks; both remain unmeasured.

The compact receipt is [icc_synthesis_2026-11-09.json](../benchmarks/icc_synthesis_2026-11-09.json).
`run-heavy: done rc=2 36s | peak-RSS 0.03GiB | min-avail 15989MiB | peak-load 4.76`
measures the Docker client; the payload had a separate 2 GiB/no-swap cap. No
container payload peak RSS was measured. Not measured on Zen 5 or ARM.

## Recognition hash

`normalized_hash` uses FNV-1a64 and deliberately zeroes only header ranges
4..8, 24..36, 40..44, 48..56 and 80..100. Transform tags, tag table, rendering
intent and profile length remain hashed. It does not discard zero bytes in
arbitrary payloads. Metadata normalization deliberately aliases those fields.

FNV also has the zero-state sticky property described by
[RFC9923 section1.4](https://www.rfc-editor.org/rfc/rfc9923.html#section-1.4):
after an internal zero state, further zero bytes leave that state zero. This
review has not demonstrated a valid malicious ICC collision against our tables.

The real contract concern is that lookup and known-profile replacement rely on
a noncryptographic fingerprint. Identification checks the hash and table
metadata, without a byte confirmation; `normalize_known_icc` checks hash and
length. A hash hit is not a mathematical equivalence proof. Recommend preserving
FNV as a lookup index, adding normalized byte equality for the four canonical
profiles whose bytes are already present, and using a stronger normalized digest
for any hash-only entries allowed to authorize pixel-changing bypass/replacement.
Preserve old recognition keys and public hash semantics during migration.

## Fixed plane storage (PR121, ed147e7e)

Both old and proposed constructors accept only the validated RGB/YUV/alpha role
families, with no duplicate roles or mixed families. They already admit at most
four planes. The array does not reject a previously accepted valid role set.
However, `Copy` is a public commitment that constrains future growable ownership.
Recommend retaining growable storage unless the core intentionally commits to a
bounded color-plane-only model. A borrowed layout slice can avoid per-call
allocations without promising owned layouts are Copy. This revisits the earlier
approved Copy direction; no implementation reversal is made here. The type is
not currently a general arbitrary-channel/auxiliary/CMYK-plus-alpha layout.

## U16 rounding (PR116, ed98ec1c)

The old workaround was developed for MoxCMS0.9.1. MoxCMS0.9.2's integer gamma
table builder now rounds before conversion; it contains a nearest-rounding test.
The new matched 4096-pixel witness uses PR116's deterministic seed and f64
profile-curve/colorant oracle on the current production adapter:

| Route | Components missing nearest /12288 | Mean signed code error | Max absolute code error |
|---|---:|---:|---:|
| Native U16 | 3815 | -0.22530724849902325 | 6.57835333676303 |
| F32 then round | 1630 | 0.1413105900426434 | 5.767520010184626 |

Neither is exact. This is one named-profile witness, not universal CMS accuracy.
The old 2-million-pixel timing/accuracy results are historical and are not
presented as a matched current speed comparison. LCMS changes do not modify
MoxCMS arithmetic. Keep the first-class backends explicit and do not merge the
old blanket F32 workaround based on its pre-release rationale alone.

## PQ correctness versus HDR intent (PR115, 829c4ff8)

PR128 fixes profile definitions, transfer representation and capability handling.
PR120 supplies the explicit-white/context/refuse rule across conversion entry
points: 203-nit anchored linear white gives PQ16 code38055 consistently. It also
refuses HLG display conversion without OOTF. That supersedes PR115's narrower
blanket refusal, but neither PR120 nor PR128 is merged. Profile representation
alone does not establish SDR-white placement, peak policy or tone mapping.

The intermediate stable/constructor proposals precede the qualified exports;
review their chronology rather than treating every open snapshot as an
independent recommended merge. This review includes no merge or publication.

## Integration disposition

The owner authorized revising and merging the review queue on 2026-11-09.
The final CMS/export stack, grayscale authority guard, CMYK qualification
and cost investigation are integrated together; intermediate failing bundle
revisions remain in history rather than becoming separate main states.
The preceding measurements describe their explicitly named historical heads.

Integration replay: `cargo test --workspace --features cms-moxcms,cms-lcms,icc-db
--locked` passes 1,591 tests, with 46 existing ignored tests unchanged.
`just main-cms-integration-check` passes strict workspace all-target Clippy,
shared backend contracts and isolated no-database tests/Clippy.
Resource receipts on r7900x: test rc=0, 185s, peak RSS 2.65 GiB; extra checks
rc=0, 13s, peak RSS 0.43 GiB. These are test-process resource observations,
not profile-synthesis memory measurements.

PR120 integration moves presets, resolver interfaces and chain planning into
`zenpixels_convert::precision`. Shared source/offer/effect types remain in core,
re-exported by that converter module. `preserves_modeled_capacity` names the
bounded claim; it does not certify gamut preservation or pixel equality.
The core policy-nesting prototype must import converter policy types explicitly;
its data-only offers can continue using core facts. No core main was changed.

## Compact P3 fallback qualification — 2026-11-09

The no-database fallback now embeds the same 408-byte generated exports as the
current database for P3/sRGB (CP12/TC13) and P3/BT.709 (CP12/TC1). Compatibility
constants remain byte-identical and explicitly selectable. Transfer 1 does not
alias transfer 13. HLG still refuses ICC synthesis.

Native libultrahdr at `5f117294` recognizes both generated exports as Display P3.
Its native writer is a positive control; the historical compatibility constant
is a negative control (unrecognized). Independent C Little CMS 2.17 evaluation
on a 17³ RGB grid, against independently constructed linear P3 and analytic
transfer curves, measures maximum absolute XYZ component error 0.0000171659 for
sRGB and 0.000135807 for BT.709; the unchanged gate is 0.0005. Default and isolated
no-database builds emit identical bytes. This is bounded P3 export qualification,
not universal ICC accuracy. Reproduce with
`just p3-export-check /path/to/libultrahdr /fresh/output/directory` under the
resource wrapper; prerequisites are CMake, a C++ compiler, libjpeg and Little CMS.
The checker builds the native oracle, exercises both feature configurations and
prints SHA-256 identities.

| Export | SHA-256 |
|---|---|
| P3/sRGB | `dc5b04c0d4bff5d4eb145431266fcdf5d6a434f2b30b32fabac88543f9df4cd4` |
| P3/BT.709 | `c088d56c2b3c462ee00a1e9d4be1d7832394f432c64341715c91af9a7c04425e` |
| Historical compatibility | `231752984cd4a5278e1b8d2390fe496767d4511fc81f54e1a5c69ae9ab4c42b5` |

Historical release refs were assessed separately: `release/0.2` at `e96e4e9c`
still selects the legacy no-database fallback; `release/0.3.1` at `4d41ed8a`
also aliases P3/BT.709 to the sRGB compatibility profile. Neither ref inherits
this qualification until the fix is explicitly integrated and checked there.
This PR targets main and authorizes no publication. See
[issue #135](https://github.com/imazen/zenpixels/issues/135).
