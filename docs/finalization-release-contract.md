# Finalization and release contract

Decision record, 2026-09-28. This supersedes earlier proposals to retain
opted-in estimation, normalize ICC with byte-subrange comparisons, or leave
HDR measurement in the foundational crate. The release pair is **0.2.17 bridge,
0.3.1 cleanup**. Compatibility means the same migrated source, with one core
version per connected pipeline; buffers from two core versions are not interchangeable.

## What finalization guarantees

`finalize_for_output_with` returns an owned `EncodeReady`: pixels, their current
`ColorContext`, and output color metadata agree. A failure yields no result;
the borrowed source is unchanged. It does not serialize a container or guarantee
that an arbitrary encoder supports every returned profile.

| Input or operation | Required outcome | Cost / boundary |
|---|---|---|
| Current ICC differs from provenance | Convert from current ICC, never the old file's profile | CMS receives actual bytes |
| Current ICC and CICP both set | Refuse ambiguous current authority | Constant-time check; decoder selects authority |
| Current CICP disagrees with descriptor | Refuse; do not fix by retagging | Constant-time check |
| `SameAsOrigin` | Convert back to the origin's selected color authority | May require a CMS |
| Original YUV matrix / limited range, decoded packed RGB | Preserve origin; output RGB uses identity matrix and full range | Does not reconstruct native YUV codes |
| Explicit unsupported matrix, transfer or range | Refuse; preserve raw CICP in provenance for a capable adapter | SMPTE 240M TC7 is not BT.709 |
| Identity | Matching context and metadata, independent owned image | One image allocation/copy, explicitly documented |
| Straight ↔ premultiplied ICC conversion | Unassociate in the source encoding, CMS on straight samples, associate in destination encoding | Two F32 scratch rows; no integer intermediate rounding |
| Undefined padding | Never interpret as coverage | Existing explicit padding policy |
| Alpha removal | Finalizer's existing policy is unconditional discard | No hidden opacity scan; preflight explicitly if required |
| Backend error | Preserve concrete backend error chain; no `EncodeReady` | A row worker may have partially written its private output |
| `prepare(max_width)` | Reserve scratch and prepare backend; subsequent rows within capacity allocate no scratch | Backend must support preparation; refuse shared stateful workers |
| No preparation | Scratch may grow on first use / larger width | No claim of zero allocations |
| Conversion composition | Optimize the final output by default | `compose_preserving` explicitly keeps materialized stages |
| PQ decoded to linear | Record that raw linear 1.0 represents 10,000 nits | This is a unit, not measured image peak |
| HDR → SDR | Explicit tone-mapping plan and peak policy before finalization | No implicit peak scan; measured-peak helper advertises its scan |
| Content light / mastering data after an edit | Caller retains only still-valid facts or explicitly recomputes | Finalizer does not fabricate MaxCLL/MaxFALL or mastering data |
| Known ICC normalization | Explicit `OutputProfile::normalize_known_icc()` | One profile hash, small canonical allocation; no pixel pass |
| Unknown ICC | Preserve bytes and ownership | No arbitrary ICC sanitizer |
| Gain map + XMP/MPF/container offsets | Container writer rebuilds references after final serialization | Pixel finalization alone cannot make a packet safe to copy |

`SameAsOrigin` preserves color-space intent, not source coding layout. A codec
that needs YUV must separately choose subsampling, matrix, range, chroma siting,
code depth and bit placement. `SampleEncoding` does not promise a YUV converter.
A still-image measurement does not establish a video's maximum frame-average
light level: aggregate across the intended sequence, with explicit timing and
frame/disposal semantics.

## Code that must stop looking safe

```rust,ignore
// WRONG: provenance describes the original file, not these modified pixels.
encode(converted_pixels, original_icc);
// RIGHT: the CMS sees current ColorContext; the returned context matches too.
let ready = finalize_for_output_with(
    &converted_pixels, &origin, OutputProfile::SameAsOrigin,
    PixelFormat::Rgb8, Some(&cms),
)?;
let (pixels, color) = ready.into_parts();
encode(pixels, color);
```

```rust,ignore
// WRONG: a broad ICC identification is not permission to replace its TRC.
if identify_common(&icc).is_some() { icc = generic_profile; }
// RIGHT: explicit exact fingerprint normalization in the conversion crate.
let target = OutputProfile::Icc(icc).normalize_known_icc();
// Unknown ICC remains the same Arc. Non-ICC targets are unchanged.
```

The normalizer shares `zenpixels::icc::normalized_hash` with the existing
identification engine. Its exact table contains the four bundled profiles;
metadata-only header variations match, but changed tags or rendering intent do
not. Approximate identification tables are not used as replacement authority.
The hash is not cryptographic and this operation is not a hostile-profile
validator. It does not delete descriptive/private tags in arbitrary ICC files.

```rust,ignore
// OLD (always warns in 0.2.17, absent in 0.3.1):
let guess = plan.estimate(width, height); // enabling a feature does not rescue it
use zenpixels::ContentLightLevel;
let cll = ContentLightLevel::measure(pixels, white);
// NEW: no replacement performance estimator is promised.
use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels_convert::hdr::measure::{CllMeasure, LightLevelMethod};
let cll = ContentLightLevel::measure_max(pixels, white, LightLevelMethod::MaxRgb); // explicit full scan
```

```rust,ignore
// Preserve large allocations, descriptor, stride, offset and color together.
let parts = buffer.into_parts();
let buffer = PixelBuffer::try_from_parts(parts)
    .map_err(|error| error.without_buffer())?; // before boxing/storing an error
// For repair/retry, use error.take_parts() instead of retaining a 100 MB error.
// Only request buffer.into_contiguous() if the receiver requires packed rows.
```

## Crate responsibilities and feature slices

- **zenpixels:** layouts, checked storage, sample encoding, color meaning and
  small HDR metadata in `hdr`. `alloc` is essential for owned buffers/Arc;
  std is optional. No measurement or estimation engine in 0.3.
- **zenpixels-convert:** pixel math, CMS extension points, explicit scans,
  explicit known-ICC normalization on `OutputProfile`, HDR processing in `hdr`.
  Do not add a dependency on codec or metadata engines. Existing `libm` already
  serves non-std float math; adding `num-traits` is unnecessary for this change.
- **Metadata engine:** read/audit/diff/semantic editing; container serializers
  own packet ordering, sizes, offsets and checksums. Runtime services supply
  optional ICC normalization without forcing conversion into codec dependency
  trees. No redundant ICC sanitizer.
- **Media layer:** native planes and samples, timing, track descriptions and
  packets. Audio belongs here, not in pixel layout enums. No speculative
  universal provider trait in zenpixels; current row views already stream.

Retain compatibility feature spellings (`planar`, `estimation-experimental`)
as no-ops in 0.3. They do not resurrect removed APIs. Keep optional CMS and HDR
analysis features; do not create a feature per type, format, or method.

## Sealing decision

Audit: `rg` across sibling zen crates for extension-trait implementations,
excluding generated/vendor/build/worktree trees. No external implementations of
our conversion extension traits were found. Repeat this before publication;
absence in our tree is not proof about every registry consumer.

| Surface | Decision | Why |
|---|---|---|
| `TransferFunctionExt`, `ColorPrimariesExt` | Seal in 0.3 | Extensions of our concrete color types |
| `PixelBufferConvertExt`, typed and HDR variants | Seal in 0.3 | Library-owned storage and conversion invariants |
| `CllMeasure` | Seal in 0.3 | Algorithms extend our concrete metadata carrier |
| Load-bearing extension traits | Already sealed | No change needed |
| `Pixel` | Keep open | Deliberate typed-pixel interoperability; requires `Pod` |
| `PluggableCms`, `RowTransform`, `RowTransformMut` | Keep open | Runtime CMS injection is essential |
| Legacy `ColorManagement` | Preserve for now | Existing integrations; don't invent a second migration |

Sealing is an explicit compatibility exception: 0.2 cannot warn on an external
implementation without also deprecating legitimate trait use. No such external
implementations were found in the audited sibling tree. Applications with custom
implementations must move those methods to their own traits before opting into
0.3; the same-source fixture does not claim otherwise.

0.2 keeps existing implementation permissions. Seal only in the breaking line,
not as an undocumented tolerated patch break. Keep backend Send/Sync contracts
and fallible clone semantics; no unsafe auto-trait implementations.

## Release gates

1. Full tests, strict Clippy, rustdoc, public API snapshots and MSRV checks.
2. Downstream warning probes, with and without experimental features. Removed
   APIs must fail under every feature combination on 0.3.
3. Packaged same-source core/converter pairings, default/minimal/interop/HDR,
   plus one-version-per-pipeline verification. No estimation use in this fixture.
4. Standalone minimal builds and a non-std target, so dev-dependency feature
   unification cannot hide std use. Record target and features, not just “passes”.
5. Compile-time comparison from fresh git archives with the same lockfile,
   toolchain and jobs. No new production dependency in this change.
6. Consolidate #76 as the bridge and #77 as cleanup. Supersede #78/#79 only
   after their wanted behavior has tests in the consolidated stack. Keep older
   audit/benchmark results labeled with their actual revisions.
7. Inventory and migrate sibling uses before publishing; publication is separate
   from pushing reviewed release branches. Do not call draft releases ready
   while their consumers or packet-finalization checks remain unverified.

The [scenario explorer](color-explorer/index.html) makes assumptions inspectable;
it is an educational model, not a display calibration or browser CMS emulator.

The proposed [frame interpretation contract](frame-interpretation-contract.md)
replaces #55's generic matrix hint with a simple reader API and expert access
using shared early validation and conversion checks. Its implementation reference
includes AV1-in-MP4, ownership, timing and pending acceptance criteria; it is not
implemented release scope.
