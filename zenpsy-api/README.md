# zenpsy-api

Unpublished, `no_std + alloc` still-image quality contracts over zenpixels.

Implement `Metric` with a concrete `Output: Evaluation`. The output keeps native
fields and exposes all named scalars through an allocation-free iterator of
`Score` views. Each score describes its ordering, native unit/scale and optional
identical-pair value. No universal range or normalization is assumed.

`score` checks input geometry, optional pixel admission, cancellation and returned
score sets. Inputs are existing `PixelSlice` views: active strided rows, precision,
colour context and HDR signalling pass through unchanged. Implementations must
reject unsupported interpretations; the contract does not convert pixels.
Packed traits remain unchanged. Optional `PlanarMetric` and
`PreparePlanarReference` accept the shared `zenpixels::plane::PlanarSlice`:
borrowed standard planes/layout, explicit color/luminance and alpha, without
packing or copying samples. Checked prepared calls bind that interpretation;
independent candidate strides are preserved.

`PrepareReference` is optional and returns actual reusable work through a native
associated type. Checked prepared calls preserve geometry and use fresh controls.
Scorers and results need not be Send/Sync; consumers add worker-specific bounds.

For heterogeneous collections, stack `Erased::new(metric)` around any typed
wrapper and store it as `Box<DynMetric<'_>>`. The wrapper also implements typed
preparation when its inner metric does. Prepared collections use
`Box<dyn DynPrepareReference>` and `prepare`, returning checked erased reference
objects. Native errors, pixel contexts and Pulse ownership pass through.
Repeated erasure keeps the original native downcast target.

`ErasedEvaluation` moves the native assessment into a box and forwards borrowed
score iteration; it does not copy native diagnostic arrays or score descriptors.
The erased path allocates a result box and short-lived explicit score iterators; typed
calls keep their native result and iterator. Use `native::<ConcreteOutput>()`
when a caller needs the retained native fields. Checked validation visits each
output set directly, without constructing erased iterators. Small output-name
sets use stack storage; larger sets grow fallibly.

Public metadata is built with constructors/builders and read through accessors.
`EvalOpts` alone supplies `Default`; empty score descriptors/maps have no semantic
default. Runtime descriptor keys use `with_key`, and infinity permission is
independent of identity. `MapGrid` constructors refuse empty/zero-step geometry.
The unpublished field/privacy and checked-grid changes require caller updates.

Diagnostics are off by default. `EvalOpts::with_native_map` explicitly requests
native diagnostic data in the concrete result; `with_map(key)` additionally
requires that exact common map or returns `NoMap`; callers must bound witnesses and
storage. `Evaluation::maps()` exposes borrowed `MapView`s: existing zenpixels `AnyPlane`
views, native `ScoreSpec` semantics and a regular `MapGrid` in input coordinates.
Typed and erased results keep the original sample allocations, precision and
strides. Undefined map samples remain unchanged. Grid placement describes where
to display cells, not which pixels causally influenced them. Native result fields
retain richer component/support metadata. Resampling and display processing stay
with consumers. Existing result implementations can keep native diagnostics and
use the default empty common-map iterator; unsupported diagnostic requests must
still be refused by the metric.

`NoReferenceMetric` and checked `score_image` take one image and use the same
structured results, maps, errors and Pulse controls. Heterogeneous one-image
collections use `Box<DynNoReferenceMetric<'_>>` with `Erased::new(metric)`.
`PairedNoReference::new(metric)` explicitly scores both inputs: candidate keys
stay unchanged and reference keys append `_ref`. Only candidate diagnostics are
requested. Actual preparation caches the reference assessment once, sharing its
native storage across candidate results without cloning diagnostic arrays.
Both originals remain available through `PairedEvaluation` accessors.

For encoder loops, owned planar preparation can survive dropping the factory.
`PreparedPlanarReference::with_evaluation` lends an `EvaluationRef` to a callback
with borrowed scalar/map visitors. Its default borrows an owned native result;
a native implementation may override it to consume scratch before recycling,
without exporting a second map. Stacked erasure forwards lending directly. Heterogeneous planar factories use
`Box<dyn DynPreparePlanarReference>` and `prepare_planar`; the resulting
`Box<dyn PlanarWarmReference<Output = ErasedEvaluation>>` retains the checked
lending path.
The checked callback validates scores, requested maps, geometry, interpretation,
cancellation and exactly-once invocation. A returned value cannot borrow the
callback's temporary assessment. Commit consumer side effects after success.
No encoder block reduction, normalization or target calibration is imposed.

`MetricId` records the effective configuration and actual implementation.
Persistence columns, fingerprint algorithms, registries, resource estimation,
scheduling, backend fallback and video composition belong to consumers.
Located errors preserve typed causes and traces. Operations own how-far child
phases without completing the caller's parent.

Run `just iqa-check`, `just iqa-portability-check` and `just iqa-cross-check`.
Rust 1.88; unsafe code is forbidden. No crate publication is authorized.

Identity and error bridging originate in
[zenmetrics-abstractions at b50fea22](https://github.com/imazen/zenmetrics/tree/b50fea22/crates/zenmetrics-abstractions).
Existing zenmetrics persistence identities are unchanged in that original crate;
this smaller contract does not derive or rename observation columns.

The [native consumer validation](https://github.com/imazen/zenmetrics/blob/master/docs/IQA_STILL_CONTRACT_VALIDATION.md)
records source pins and the exact scope of native validation. Contract tests do
not qualify algorithm/reference parity or every model, input and backend.
