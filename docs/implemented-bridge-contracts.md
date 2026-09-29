# Implemented bridge contracts

2026-09-27. These are unreleased additions/fixes on the 0.2 line. Migrated signatures are tested unchanged against the packaged 0.3.1 candidate. Runtime behavior and allocation checks live in
`zenpixels-convert/tests/prepared_contracts.rs` and `zenpixels/tests/storage_contracts.rs`.

## Prepare rows once

```rust
use zenpixels_convert::{PixelDescriptor, RowConverter};
let mut worker = RowConverter::new(PixelDescriptor::RGB16_SRGB, PixelDescriptor::RGB8_SRGB)?;
worker.prepare(4096)?; // scratch, selected tables, backend setup; no input scan
worker.try_convert_row(source_row, destination_row, width)?;
```

Old `convert_row` remains a panicking convenience. The fallible form validates
capacity, byte extents and sample alignment before writing. Backend failures may
leave partial output. Original backend errors remain in `Error::source`.
Prepared stateful workers are uniquely owned; build one converter per worker.
`try_clone` refuses uncloneable state rather than dropping its transform.

## Make loss and scans explicit

```rust
// Previously accepted without checking opacity; now fails during planning.
let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardIfOpaque);
let rejected = ConvertPlan::new_explicit(rgba, rgb, &options);

// One explicitly requested pixel scan before an unconditional drop.
zenpixels_convert::adapt::check_opaque(source.as_slice())?;
let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
let plan = ConvertPlan::new_explicit(rgba, rgb, &options)?;
```

Do not mutate the source between the scan and conversion. The existing whole-image
explicit adapter performs the preflight when that conditional policy is selected.
There is no new hidden opacity pass in ordinary conversion.

```rust
// Default: optimize requested output; intermediate integer rounding may disappear.
let optimized = a.compose(&b);
// Explicit: retain every materialized intermediate, including quantization.
let stages = a.compose_preserving(&b);
// Prove exact sample representation changes, otherwise refuse without a scan.
let exact = ConvertPlan::new_preserving_samples(rgb8_srgb, rgb16_srgb)?;
```

Composition retains each stage's descriptors and luminance anchor. External CMS
composition refuses until a complete executor can retain those transforms.

## Preserve allocations and metadata

```rust
let parts = image.into_parts(); // data + offset + stride + dimensions + descriptor + context
let image = PixelBuffer::try_from_parts(parts)
    .map_err(|e| e.without_buffer())?; // strip a potentially huge allocation before boxing/logging
let packed = image.into_contiguous(); // explicit row moves, no second image allocation
```

Use `take_parts()` on the adoption error when recovering the allocation instead.
`into_vec()` now warns: it loses the offset/stride/interpretation. Typed U8 exports
reuse compatible allocations; alignment/capacity incompatibility can require a copy.
Typed reinterpretation refuses a physical format change. Erase explicitly, change
the erased view's layout, then `try_typed` the new pixel type.

## Current color and HDR

`ColorContext` holds one authoritative **current** profile. `ColorOrigin` retains
original metadata. Finalization passes actual ICC bytes and converts pixels for
`SameAsOrigin`; restoring old tags is insufficient. Known contradictory CICP and
descriptor declarations are rejected. Descriptor-only helpers refuse ICC work
requiring a full CMS; use `finalize_for_output_with` with that backend.

Color-changing convenience conversions replace old current signaling. Layout/depth
changes preserve it. Raw PQ decoding is L/10000; its resulting linear buffer now
carries a 10,000-nit anchor. Explicit linear anchors are honored when encoding PQ.
Tone mapping normalizes PQ to the supplied source peak, preserves straight alpha
through nonlinear work, and emits the target luminance anchor. Linear input with
an attached diffuse-white anchor is scaled to the source-peak convention in the
row pipeline; without one, `HdrConfig` documents 1.0 as source peak.

```rust
// Old name concealed peak discovery; now deprecated.
let sdr = hdr.convert_to_sdr_measuring_peak(target)?; // explicit extra rowwise pass
// Or supply the peak: no measurement pass.
let sdr = hdr.convert_to_with_hdr_config(target, HdrConfig::for_source_peak(1000.0))?;
```

HLG→display tone mapping needs an OOTF and is refused; a peak alone is insufficient.
Unsupported layouts, gamut matrices and CMS sample-depth pairs fail during setup.
This does not add native 10/12-bit video-code semantics to normalized U16 images.

## Costs and remaining scope

Unassociation, matte and multi-stage conversion may require additional row kernels
and scratch; they are not yet all fused. Exactness checks and descriptor validation
are metadata-only. U16 content analysis combines opacity, chroma and replication
checks in one traversal. Narrowing's production kernel is unchanged pending ARM
measurements. Experimental native video carriers and a unified zencodec output plan
remain separate integration designs; no speculative public wrapper family was added.
