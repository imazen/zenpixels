# Gamma power comparison, 2026-11-06

`pxfm 0.1.30`, the version used by the reviewed moxcms revisions, is compared
with magetypes at `f40f5d38c22544a1bdd57da5c33fb037b533c8e4`. The archmage
checkout was clean and inspected read-only. The harness uses the published Git
revision, not that checkout's target or working tree. Rust 1.99.1, default target
CPU, Ryzen 9 7900X. This is numeric validation, not a speed benchmark.

## Matched-domain measurements

Each of seven nominal gammas (1, 1.8, 2.2, 2.4, 2.6, 2.8, 4) and its f32
reciprocal is evaluated at 65,536 samples per band. Gamma 1 and its reciprocal
are intentionally duplicate configurations. The bands are every normalized
U16 code on [0,1], the same grid scaled to [0,16], and a logarithmic grid of
positive normal f32 inputs from 2^-126 to 2^127. The reference evaluates the
actual f32 input and exponent in f64. It does not quantize exponents to ICC
8.8 values. Scalar and AVX2/FMA magetypes backends are measured separately.

Maximum relative error among finite outputs with a normal reference result,
across the unit-U16 grid and all exponent configurations:

| Implementation | Sampled maximum relative error |
|---|---:|
| pxfm dirty_powf | 2.371181022e-6 |
| magetypes pow_midp, scalar | 1.866631039e-6 |
| magetypes pow_midp, AVX2 | 1.866631039e-6 |
| magetypes pow_lowp, scalar | 5.567079078e-3 |
| magetypes pow_lowp, AVX2 | 5.567079078e-3 |

These are sampled maxima on this specified domain, not all-f32 error bounds.
There were no nonfinite outputs for normal references in the unit/extended-16
bands. Zero-reference cases are counted separately: dirty_powf fails zero for
12 of the 14 exponent configurations; the two identity configurations return
zero. Both magetypes tiers return zero for all tested positive exponents.
Full per-configuration counts, including the wide band's failures and
subnormal/overflow classifications, are in
[the numeric TSV](../pow_comparison_2026-11-06.tsv).
The raw probes are in [the edge TSV](../pow_comparison_edges_2026-11-06.tsv).

## Implementation differences

All three implement exp2(p * log2(x)). Dirty's logarithm and magetypes midp use
similar exponent/mantissa reduction followed by an odd degree-seven polynomial
in (a-1)/(a+1), with different coefficients. Dirty exp2 uses a 64-entry table,
a small quadratic correction, and unchecked exponent construction. Magetypes
midp exp2 uses nearest-integer splitting and a degree-six polynomial; it returns
zero below exponent -126 and infinity at or above 128. Magetypes lowp log2 is
a rational approximation; exp2 uses floor splitting and a cubic, clamping its
input to [-126,127.99]. Its power method separately masks zero inputs to zero.

The checked names are not full IEEE powf replacements. On both tested magetypes
backends, pow_midp(NaN,1.8) returns zero and pow_midp(-0.5,1.8) returns MIN_POSITIVE;
lowp returns a huge finite number for those cases. Both misinterpret positive
subnormal input; midp also flushes subnormal output to zero. pow_midp_precise is
currently an alias and adds no subnormal handling. Gamma odd extension must be
applied by the color caller using abs and sign restoration; none of these APIs
provides that contract as a general floating-point power function.

For nominal gamma 1.8, dirty_powf(0,1.8) is -177102480, while both magetypes
forms return zero. This exponent is the f32 value near 1.8, not the built-in
ProPhoto ICC exponent 461/256 from the separate pixel reproduction.

## Practical gamma domain

Current moxcms source covers identity 1, ProPhoto's 461/256, Adobe's 563/256,
CICP gamma 2.2/2.8 and DCI/SMPTE 428's 2.6; sRGB's power branch uses 2.4 with a
linear toe. A practical specialization could cover positive applied exponents
[0.25,4], including reciprocals, with a general fallback. This is a proposal,
not a new validity constraint. ICC's single-value curveType stores unsigned 8.8
and can encode values through 255.99609375; reciprocal construction also makes
very small positive source gammas become large applied exponents. PQ is not a
single gamma curve and its powers do not fit this proposed interval.

Restricting gamma does not resolve dirty_powf at zero. For positive normalized
U16 source codes and p in [0.25,4], the smallest possible result exceeds 2^-64,
so those nonzero samples are safely normal. Float intermediates after matrices
or other processing need separate range handling. No gamma or pixel input is
clamped by this harness.

Primary sources:
[ICC 4.4 specification, sections 4.9 and 10.6](https://www.color.org/specification/ICC.1-2022-05.pdf),
[pxfm dirty_powf](https://github.com/awxkee/pxfm/blob/acf7acd00cfc845307eda7b7b2de19e42303de82/src/powf.rs),
[magetypes pinned implementation](https://github.com/imazen/archmage/blob/f40f5d38c22544a1bdd57da5c33fb037b533c8e4/magetypes/src/simd/generic/generated/transcendentals_f32x8.rs).

## Reproduction and limits

Run `just pow-comparison-check` and `just pow-comparison NEW_OUTPUT.tsv` through
the resource wrapper and shared build lock. The root requires an AVX2/FMA host
and refuses if unavailable. No ARM/WASM/AVX-512 accuracy or relative speed is
claimed. Gates passed: formatting, strict all-target harness Clippy, and the
complete numeric run. Resource wrapper: 9 seconds including compilation, peak
RSS 0.34 GiB, minimum available 18390 MiB and peak load 0.27; this describes the
validation process, not library memory use or speed.
