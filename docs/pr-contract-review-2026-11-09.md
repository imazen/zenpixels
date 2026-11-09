# API placement, ICC storage and review chronology

Source-pinned review, 2026-11-09. This proposes follow-ups; it changes no
production API, default feature, bundled bytes or established pixel expectation.

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
