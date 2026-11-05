# zenpsychovisual-api

Shared contracts for image quality assessment over zenpixels. The current
scope is still images. This unpublished core is `no_std + alloc`, forbids unsafe
code and depends on no metric implementation, codec, converter or GPU runtime.

A configured `Metric` declares every native output and its full arithmetic
identity. `score` admits inputs and binds the rich evaluation to that identity:
multiple scalars, native spatial maps, map geometry/polarity, on-demand native
folds, extras and concrete-result downcasting. Packed inputs preserve stride
and attached color context. `Planes` validates borrowed native samples through
zenpixels' `PlanarLayout`, including odd subsampling, sample depth/shift and
float encoding. No RGB conversion or color interpretation is invented.

`prepare_reference` requires actual reference work reuse with exact cold-result
bits. Unsupported preparation refuses. `NoReferenceMetric` scores one image;
`PairedNoReference` retains reference and candidate outputs under distinct keys.

Public data types are non-exhaustive; use constructors and builders. Metric,
evaluation and prepared-reference traits are intentionally open to external
implementations. Future optional operations must have default refusal behavior.
Checked operations borrow full how-far Pulse control and complete their own child
phases. Native located errors transfer traces and typed causes; actual backend
failures take precedence over a later stop. `stop_reason`/`is_stopped` classify
located errors for outer how-far handoffs.

Call `conformance::assert_conforms` with representative native input pairs to
check declared outputs, map-request score identity, reduced-map geometry,
optional warm-reference parity and cancellation. These controls do not replace
a metric's reference-oracle tests. The pixel ceiling is ingress admission,
not a bound on backend scratch, retained references or decoder allocations.

Video session/timing/alignment contracts remain outside this crate. They can
reuse these inputs and rich results. Codec output mapping belongs in the
optional `zencodec-iqa` adapter, not in this dependency graph.

Run `just iqa-check` and `just iqa-portability-check` from the repository root.
This standalone workspace requires Rust 1.88; it does not raise the pixel
crate's MSRV. Publication needs a separate review and authorization.

Identity, scalar/map operations and no-reference pairing originate in
[zenmetrics-abstractions at b50fea22](https://github.com/imazen/zenmetrics/tree/b50fea22/crates/zenmetrics-abstractions).
The existing `zenmetrics-id-v1` fingerprint domain and column derivation remain
unchanged so moving the contract need not rename persisted observations.

The [native consumer validation](https://github.com/imazen/zenmetrics/blob/40590360c00be0a5fb5aab0535d4c70dd792301a/docs/IQA_STILL_CONTRACT_VALIDATION.md)
checks serving Zensim A, selected Margarine, Butteraugli and fast-ssim2 with
pinned sources. Both Linux x64 and aarch64 CI panels pass. Coverage is native
sRGB RGB8 on three representative shapes; it does not qualify HDR, all models
or algorithm/reference parity. Margarine has no prepared-reference API and
only observes cancellation at native-call boundaries.
