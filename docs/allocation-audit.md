# Allocation audit (#151)

This is a partial source audit. Exhaustive entry-point/transitive coverage,
deterministic failure injection, ownership rollback controls, and target/MSRV
qualification remain incomplete. A `try_` name or `Result` alone does not
establish an end-to-end allocation guarantee.

## Verified starting points

| Surface | Source | Current behavior |
|---|---|---|
| Sample-typed pixel storage | [`PixelStorage::try_zeroed`](../zenpixels/src/buffer.rs) | Uses bytemuck's fallible zeroed vectors; allocation errors become BufferError. Underlying cause retention still needs review. |
| Aligned samples and planes | [`AlignedSamples::try_zeroed`, `Plane::alloc`](../zenpixels/src/plane.rs) | Fallible zeroed chunk allocation and checked geometry. This does not qualify every storage conversion or external SampleStorage implementation. |
| Tight byte extraction | [`PixelSlice::contiguous_bytes` and `try_contiguous_bytes`](../zenpixels/src/buffer.rs) | The former uses infallible Vec capacity; the latter reserves fallibly before copying. Both tight/strided paths and zero extents belong in the final inventory. |
| Borrowed-to-owned handoff | [`PixelCow::into_owned`](../zenpixels/src/buffer.rs) | Borrowed input reaches the infallible owned-copy path. Ownership-preserving error behavior must be audited separately from successful conversion. |
| Current color attachment | [`with_*` context constructors](../zenpixels/src/buffer.rs) | Arc::new control allocations remain infallible even where image storage has a fallible path. Document or replace them explicitly. |
| Conversion planning | [`ConvertPlan`](../zenpixels-convert/src/convert.rs), [chain planner](../zenpixels-convert/src/precision.rs) | Uses Vec::with_capacity, vec!, nested vectors and shared control objects. Result-returning planners still reach infallible allocations. |
| Scratch preparation | [`ConvertScratch::prepare`](../zenpixels-convert/src/convert.rs) | Main scratch uses try_reserve; synthetic warmup still creates two vectors infallibly. Audit its nested kernel/LUT preparation too. |
| Bundled ICC lookup | [CICP group cache](../zenpixels-convert/src/icc_profiles/cicp_bundle.rs) | Cold lookup decodes into vec! and publishes Box::new through OnceBox. Returning borrowed bytes after warming does not make the cold path allocation-free. |
| CMS adapters and worker clones | [converter](../zenpixels-convert/src/converter.rs), [moxcms adapter](../zenpixels-convert/src/cms_moxcms.rs) | Box/Arc construction and dependency-owned transform allocation remain. Plugin/backend error behavior needs a transitive contract. |
| Pixel hashing | [serializer](../zenpixels/src/hashing.rs) and [contract](hashing.md) | Validated views use bounded stack buffers and borrowed samples. The selected zero-seed XXH3 uses a static secret; caller-supplied Hashers and view construction are outside this claim. |

## Completion gates

The complete inventory must cover ownership transfer, aligned fallback copies,
orientation/repacking, typed extraction, cloning, CMS preparation, ICC synthesis
and decompression, lazy caches, metadata and planner growth. For each allocating
entry point, record size arithmetic, allocator/cause mapping, transitive calls,
source ownership on failure and budget rollback. Audit Clone/ToOwned and
Cow::into_owned as well as explicit allocation syntax.

Exercise deterministic failures in bounded child processes, never host OOM.
Cover alignment/cast failure, reserve/capacity overflow, partial preparation,
cache initialization and worker clones. Verify no_std/std, WASM and each
applicable MSRV. Any unavoidable dependency or shared-control allocation must
have a named exception; no crate-wide fallibility claim follows from the rows
above. Preserve existing infallible convenience contracts unless an explicitly
reviewed API migration replaces them.
