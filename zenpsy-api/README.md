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
This first minimal surface accepts packed views; it does not replace native
planar entry points.

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
The erased path allocates a result box and short-lived score iterators; typed
calls keep their native result and iterator. Use `native::<ConcreteOutput>()`
when a caller needs the retained native fields.

Diagnostics are off by default. `EvalOpts::with_native_map` explicitly requests
native diagnostic data in the concrete result; callers must bound witnesses and
storage. Map geometry, component arrays and processing remain native APIs.

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
