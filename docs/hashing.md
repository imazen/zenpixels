# Pixel and ICC hashing

## ICC recognition

The normalized fingerprint is a pair `(FNV-1a64, MurmurHash3_x86_32)`.
Both must match before identification or explicit known-profile normalization.
Canonical replacement then requires normalized byte equality against the
retained bundled profile; fingerprints are only a filter.
Murmur's architecture labels name algorithms, not ISA dispatch: fixed
`x86_32` is portable, as is zenhash's explicit `x64_128` API. `x64_128` remains
that algorithm on 32-bit x86 and does not become the different `x86_128` hash.
Both use fixed little-endian words on every target. The existing ICC pair here
still uses x86_32; pixel `hash_with` remains XXH3-64.

The metadata normalization ranges remain unchanged; the Murmur seed is fixed
at 3_242_157_231 and the variant always reads little-endian words. The second
fingerprint is not merely a finalizer on the first.

All 598 retained rows keep their existing classification, curve-fit tolerance
and substitution permissions. Murmur values were computed from exact originals
using the pinned upstream Quickwit implementation. The small const algorithm
module is mirrored from zenhash so default ICC support keeps Rust 1.85 and no
new runtime dependency. Its source preserves the original MIT notice.
`normalized_hash` now returns the pair, an authorized 0.3.1 API break.

## Pixel values

The opt-in `hash` feature adds `core::hash::Hash` and `hash_with(inclusions)`
for PixelSlice, PixelSliceMut, PixelBuffer (every ByteStorage owner), PixelCow,
Plane (every SampleStorage owner), AnyPlane, PlanarSlice and CodePlane.
Typed, erased, mutable, borrowed and owned views hash identically when they
carry the same logical samples and fields. PlaneBuf/PlaneMut/AlignedPlane are
storage variants of Plane and use that implementation.

`Hash` feeds the complete selected byte stream to the caller's Hasher. It emits fixed 1 KiB writes (plus the final remainder) so caller hashers
that distinguish write boundaries also ignore storage strides. It does
not collapse the object to an intermediate 64-bit digest. `hash_with` returns a
64-bit XXH3 fingerprint through zenhash, using the same fixed-block serializer.

| Field | SamplesOnly | Pixels | All (default) |
|---|---|---|---|
| Active sample values/bits, geometry, sample type, component/plane boundaries | yes | yes | yes |
| Pixel format, interpretation, alpha mode, signal range | no | yes | yes |
| Planar roles/order, encoding, chroma subsampling/siting | no | yes | yes |
| Current ColorContext ICC bytes, CICP, diffuse white, HDR peak, presence/absence | no | yes | yes |
| Declared current ExactDepth evidence | no | no | yes |

ColorContext and ColorOrigin also gain value-based Hash implementations.
ColorOrigin is not carried by pixel views; their hashes cannot include absent
provenance. Hash ColorOrigin separately when provenance is part of a cache key.
No uncarried mastering metadata, orientation or source precision is inferred.

Every mode excludes row padding, allocation prefix/suffix, stride, addresses,
allocator identity, storage owner and Rust pixel marker type. Undefined alpha
components (RGBX/BGRX and other explicitly undefined alpha) are excluded.
Native CodePlane and planar integer encodings ignore unused container bits and
hash extracted codes through the existing CodePlane reader (also in SamplesOnly); raw Plane has no encoding and hashes every active sample
bit. SamplesOnly still separates geometry/type/plane boundaries; it does not
convert packed data to planar, change channel order or decode color spaces.

Sample floats preserve every bit, including signed zero, infinities and NaN
payloads. No normalization, clamping or numeric conversion occurs. Wide stored
samples are encoded little-endian, including on big-endian hosts. Metadata
Hash numeric writes normalize usize/isize to 64 bits and explicit little-endian
order. Empty views include their geometry and metadata without scanning rows.
Present-but-empty color context remains distinct from absent context.

The format starts with a domain/version marker and inclusion selector. These
are crate-version fingerprints, not a durable interchange format: Rust's derived
metadata Hash encodings and the chosen backend may evolve in a future release.
Existing scalar/descriptor Hash implementations are unchanged. Raw allocations,
unvalidated PixelBufferParts and ownership handoff errors have no pixel Hash API.

Without `hash`, the existing dependency graph and Rust 1.85 core floor remain.
Enabling `hash` requires Rust 1.89 through zenhash, and works with no_std + alloc.
Backend selection, reproducible commands and measured limits are recorded in
[the pixel-hash benchmark](../benchmarks/pixel-hash/README.md).

Hashing a validated view uses bounded stack buffers and borrowed data; the
selected zero-seed XXH3 state uses zenhash's static default secret. These paths
do not allocate sample storage. This describes hashing only, not view/metadata
construction or a caller-supplied Hasher. The broader allocation audit in #151
remains separate. Performance measurements use std/runtime dispatch; no_std
performance has not been measured.
