# Borrow native codes, then convert the selected region

The output-plan review adds `zenpixels::sample::CodePlane` and
`zenpixels_convert::ycbcr::YcbcrToRgbF32`. The current consumer is zencodec's
native AV1 adapter in `perf/av1-policy-validation/src/prepared_conversion.rs`.
It replaces that harness's private float kernel with the shared implementation.
These APIs are unreleased.

An adapter first borrows Y, Cb and Cr from its native mapping guard. `CodePlane`
retains the declared storage word, code depth, bit placement, geometry and byte
stride. It neither copies the payload nor treats ten-bit codes as normalized U16.
Code padding is ignored on extraction; format-specific conformance scans are a
separate decision. Typed words use native endianness, and the final row need not
have trailing padding.

With source interpretation established, preparation and execution are two calls:

```rust,ignore
let plan = YcbcrToRgbF32::new([y, cb, cr], subsampling, siting, source_cicp)?;
let converted = plan.convert_rows(crop_x, crop_y, destination)?;
```

The plan borrows the original planes until it drops. The adapter can reserve
destination storage and ask its policy resolver while this exact plan remains
held. An unknown chroma location is not silently interpreted as centered. The
adapter must resolve and report such an assumption before constructing the plan.
No location is needed for 4:4:4.

Execution writes the requested strided region and returns a view with matching
current color context. Coordinates refer to the complete coded source, so odd
crop origins retain chroma phase. Extent and descriptor failures occur before
any write; output padding remains untouched. Execution allocates no payload or
scratch. Preparation allocates one shared color context. This is not a bound on
native decoding, retained input, metadata or the complete operation.

The adapter checks cancellation between row batches, adopts the destination
owner into its image result and resolves publication. zencodec's existing
`ImageOutput` retains original frame interpretation and timing separately from
current RGB interpretation. The same output can enter `OutputPlan` for
linear-light processing; F32 storage alone does not mean linear light.

## Qualified route and explicit refusals

The implementation accepts 8/10/12-bit unsigned codes in U8/U16 storage, including
valid shifted U16 words. It accepts both signal ranges, 4:4:4, centered 4:2:2 and
centered 4:2:0, with BT.709 primaries, sRGB transfer and the BT.709 matrix. Chroma
dimensions round up at odd edges. RGB output is full-range, sRGB-encoded F32;
negative values and values above one survive. There is no transfer conversion,
gamut mapping, tone mapping or integer quantization in this stage.

Other color interpretations and unsupported chroma locations return
`At<YcbcrError>`. Errors retain the actual color/encoding, offending plane or
requested region and descriptor. Error chains can therefore explain refusals
without depending on an earlier capability query.

This is six public types and nine inherent methods, with no new feature or
dependency. It adds a borrowed conversion mechanism; it does not replace native
ownership, collection selection, role metadata or the deprecated planar owner.
The current zencodec adapter translates its earlier plane/encoding types at the
boundary. Unifying those duplicate descriptions remains a migration item.

## Validation

`just ycbcr-check` tests storage/placement, shifted words with nonzero padding,
exact crop and row-partition consistency, untouched destination padding and
pre-write refusal with diagnostic facts. It runs the converter tests with and
without default features, strict lint and a no-default-feature WASM build.

Native zencodec validation covers 54 AV1 presentations and 162 crops. All 54
float output files are byte-identical to the earlier private implementation.
The independent FFmpeg/zimg comparison retains its fixed normalized absolute
1e-6 gate and deliberate-clipping controls. Existing OutputPlan processing,
callback refusal/error/unwind, cancellation and ownership controls use the new
converter. Numerical agreement with an independent implementation is distinct
from exact equivalence between this implementation's row/crop paths.

The arithmetic originates in imazen/zenavif `feadc4c94b19`'s YUV recipe, adapted
and independently qualified without integer saturation in imazen/zencodec
`c7e76b23`. Other matrices/transfers, non-centered siting, integer output,
resizing, HDR rendering and throughput measurements remain outside this
qualification. The strict preset and continuous-versus-IEC transfer-model
questions in the downstream OutputPlan remain open.
