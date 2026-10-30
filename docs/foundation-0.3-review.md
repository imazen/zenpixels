**Foundation release review — zenpixels / zenpixels-convert 0.3 and the next breaking zencodec release**

Reviewed 2026-09-23. Source snapshots: zenpixels `e56f626b14e6f0211e020159d30b3a48616e88e0` (both crates 0.2.16), zencodec `3cd74c84737b2f1a51a914c4b4b9383a427dcc3a` (0.1.27). zencodec can advance to 0.2 independently; coordinated contracts matter more than matching version numbers. Its existing, locally modified `docs/pixel-descriptor-negotiation.md` was read as a proposal and left untouched.

**Recommendation:** use the breaking release to make preservation promises enforceable: one authoritative interpretation of current pixels, validated storage, executable conversion plans, and input-aware codec fidelity. Ship the concrete correctness fixes on the existing release lines first where possible. Avoid filling this release with every HDR feature, policy helper, or API rename in the backlog.

Parallel reviewers and the integrating reviewer read all **166 Rust files / 105,416 lines**, plus both generated textual ICC tables and the Python/shell source: **170 files / 105,977 lines** total. This includes production code, inline and integration tests, testkit, examples, benchmarks, fuzz targets and tooling. The [per-file coverage ledger](foundation-0.3-review-coverage.tsv) records ownership. Binary profile assets were inventoried, not represented as source-code reading. All 17 open GitHub issues and their comments were reviewed. Findings marked **reproduced** were exercised through isolated consumer crates; other findings are source-review conclusions, not claims of exhaustive runtime verification. The complete workspaces and all architecture-specific SIMD paths were not executed.

**1. The breaks worth coordinating**

| Priority | Contract to establish | Why the existing API is insufficient | Suggested release boundary |
|---|---|---|---|
| Essential | Current color interpretation distinct from source provenance | Descriptor, ColorContext, ColorOrigin and SourceColor can disagree; helpers choose different authorities | Add resolved/current color accessors first; in 0.3 make conversions and finalization consume one authoritative description and keep original tags separately |
| Essential | Exact preservation distinct from acceptable perceptual loss and conversion availability | Zero heuristic loss, “Lossless” buckets and original bit depth are currently used as proofs they cannot provide | Introduce explicit guarantees and supported-path checks; migrate zencodec negotiation to them before removing old helpers |
| Essential | Valid descriptors, typed views and owned/planar storage | Public fields and reinterpret/adoption APIs allow internally inconsistent values | Checked construction, explicit type erasure/retyping, validated plane replacement and clear minimum backing extents |
| Essential | Plans describe the operation that will actually execute | CMS transforms disappear from composed plans; optimization removes quantization | Immutable complete plans plus execution state; preserve composition semantics; replace misleading Clone where necessary |
| Essential | Codec acceptance and encoded preservation are separate | `supported_descriptors()` can mean “accepted then downconverted”; lossless mode need not preserve this input | Per-input support assessment, explicit requirements versus preferences, actual output/fidelity reporting |
| Strong candidate | Streaming and animation preserve the same information as buffered paths | Pull callbacks cannot return errors; sink output loses frame metadata; owned frame copies lose color context | Fallible row source, frame result carrying timing/context, static/dyn parity |
| Strong candidate | Validated HDR units and fallible serialization | Unchecked diffuse white and gain-map rational conversion can produce invalid or silently altered output | Positive finite anchor type; distinct reference/peak/measurement semantics; checked gain-map and EXIF writers |

These are contract changes with current callers and demonstrated failure modes. Most new convenience APIs, custom format sets and page enumeration can be added without waiting for a breaking release.

**2. Color authority and output finalization should be the first design decision**

The current “atomic finalization” guarantee is not true. `zenpixels-convert/src/output.rs:432` constructs output pixels and metadata through paths that do not share all their inputs:

- **Reproduced:** sRGB pixels finalized with `SameAsOrigin` and a Display-P3 origin retain sRGB bytes/descriptor while emitting Display-P3 metadata.
- **Reproduced:** a premultiplied RGBA input can be returned unchanged with a straight-alpha descriptor. `descriptors_match` at line 536 omits alpha.
- **Reproduced:** named narrow-range CICP can be emitted over unchanged full-range white samples.
- **Reproduced:** the new finalizer's CMS call receives neither the attached source ICC nor the requested output ICC. A spy CMS sees two non-ICC inputs.

Fix those behaviors now. For 0.3, make a resolved output plan own the relationship among current encoding, source provenance, target encoding, alpha handling and permitted loss. Give that plan consuming, borrowing and caller-output execution forms to address allocation issue #69. Returning `EncodeReady` should certify agreement among bytes, descriptor and emitted color information; it should not merely bundle them.

The same distinction must reach zencodec. `src/color.rs:247` never checks the target's ICC capability (**reproduced**), can preserve a conflicting source ICC when CICP is authoritative and the target is ICC-only, and explicitly permits best-effort synthesis failure to discard the only wide-gamut description. A caller asking for preservation needs an error or an explicit loss outcome. `CodecSet::transcode` (`src/set.rs:545`) takes metadata from `ImageInfo`, although a decoder may have changed the current pixel encoding. Resolve current output color before constructing encode metadata.

`ColorContext` (`zenpixels/src/color.rs`) presently allows simultaneous ICC and CICP, with `as_profile_source` preferring ICC while transfer/sRGB shortcuts inspect CICP. Its documentation refers to authority that the context does not store. `ColorProfileSource::resolve` can extract an ICC CICP tag without honoring the distinction between that tag and the profile's actual transform. Add a single resolution operation; keep “unknown” explicit and require a named assumption before relabeling it.

Also fix the profile machinery before treating it as proof:

- `icc_profiles.rs:253` aliases Display-P3/BT.709 transfer to an sRGB-TRC profile, and BT.2020/sRGB to a BT.709-TRC profile. The curated shortcut precedes the accurate full-grid bundle. Tests currently bless the aliases.
- `scripts/icc-gen/src/main.rs:1068` ORs empirical and structural safety masks; absence of LUT tags can override a failed empirical comparison. Gray identification similarly loses white-point distinctions. This supports a generator-policy defect, not a measured claim about every bundled profile. Reconcile the generator and committed tables, and distinguish approximate substitution from exact preservation.
- `Cicp::BT2100_PQ/HLG` encode YCbCr matrix 9, but named RGB profiles use them. **Reproduced:** resolving the named profile succeeds while resolving its CICP representation fails. Add explicit RGB/YCbCr forms and deprecate the ambiguous usage.

**3. Do not relocate the current loss classifier unchanged**

The negotiation draft's direction—source-aware negotiation and an encoder preservation envelope—is useful. Its proposed proof rules need revision before implementation:

- `loss == 0` is a ranking result, not an exactness proof. `LossBucket::Lossless` includes small nonzero model costs; perceptual tests use a sub-JND threshold under the same name.
- Provenance saying “originally U8” does not prove that resized or otherwise computed F32 values still lie on the U8 grid. Even the current tests discuss resizing while asserting zero narrowing loss.
- `ColorPrimaries::contains` uses a scalar gamut ordering. **Reproduced:** it says Display-P3 contains Adobe RGB; the crate's own matrix maps Adobe green to negative P3 red. `negotiate.rs:773` uses this predicate when estimating loss.
- `conversion_cost` omits signal range; `best_match` and pipeline selection can recommend paths the converter refuses. Candidate working descriptors can reset primaries/range.
- A CMS transform depends on profiles, alpha/range handling, intent and actual backend support. Matching descriptor fields does not prove profile equivalence. This caveat is broader than “two custom ICCs with matching descriptors.”
- `From<PixelDescriptor>` defaulting an encoder support record to `Exact` would silently bless old accept-and-convert lists. Migrate explicitly or default to unknown/unverified support.

Keep separate answers to: **can this operation execute, what does it guarantee, and how should permitted alternatives be ranked?** Structural relationships can live in zenpixels; backend feasibility and executable conversion planning belong in convert; encoded preservation belongs in the codec implementation. Do not move all perceptual weights into the interchange crate merely to share a helper. zenpixels-convert already has a no_std build, contrary to the draft's blanket std characterization.

In zencodec, keep “preferred” advisory and introduce an explicit required constraint for callers that must not narrow, drop alpha, recolor or lose HDR. Report the selected descriptor and effective encoded precision/model. `CodecSet::transcode` currently re-decodes with preferences when its first result is unsupported, then does not validate that the second result meets the encoder's descriptor list. A preferred list cannot establish that requirement.

`Fidelity::Lossless` also needs a defined reference: preserving incoming samples, preserving the codec's internally converted representation, and selecting a lossless bitstream mode are different promises. Keep achieved fidelity separate from `supports_lossless()` and `is_lossless()`. A format supporting a lossless mode does not prove that a particular source file was encoded losslessly.

**4. Conversion plans and alpha handling need semantic repair**

`RowConverter::compose` (`converter.rs:373`) reconstructs from `ConvertPlan`; an external CMS operation is represented there as an identity placeholder. **Reproduced:** a CMS that writes 42 executes directly, but its composed converter copies the input. **Reproduced:** cloning an owned external transform with convert's std feature disabled likewise loses the transform. The std clone instead shares mutable state behind a mutex, which should not be mistaken for independent worker state.

Make a plan complete enough to reproduce its behavior, or refuse operations that cannot preserve the external transform. Expose a fallible worker/fork operation if backend state cannot be cloned reliably. Until that exists, do not advertise `plan()`/`from_plan()` as a faithful serialization of every converter.

`convert.rs:2168` cancels inverse pairs in both directions, including F32→U8→F32, U16→U8→U16, clamped transfer pairs and premultiplication. **Reproduced:** composing F32 value 0.1234567 through U8 and back becomes identity, while executing the two converters separately quantizes it. Composition should preserve declared operations; any approximate optimization must be explicitly requested and reported.

Other source-level correctness findings to fix on the current line:

- **Reproduced:** `adapt.rs:346` and `:723` transfer-agnostic fast paths omit the required Unknown-transfer condition: linear RGB8 value 128 becomes unchanged sRGB 128, versus approximately 188 through the actual converter. `try_adapt_in_place` at 549 can relabel transfer, primaries, range and alpha. Separate adaptation from explicit assumption/retagging.
- CMS interfaces accept `PixelFormat` and profiles without a complete alpha/range contract; dispatch can bypass checks. **Reproduced:** MoxCms accepts a P3 RGB8→RGBF32 plan, then panics with LaneSizeMismatch during execution (`cms_moxcms.rs:344`). Carry the full resolved semantics and validate depth handling at construction.
- **Reproduced:** RGBA→GrayAlpha is accepted as identity and maps `[200,30,10,99]` to `[200,30]`, losing alpha. **Reproduced:** DiscardIfOpaque drops nonopaque alpha; transparent black composited onto white then converted to gray produces black; premultiplied sRGB→straight linear applies transfer before unpremultiplication and produces approximately 0.10175 instead of 0.21404. Combined layout steps need the same alpha policy enforcement as standalone DropAlpha. OklabA→GrayAlpha also drops alpha and adds opaque alpha in the source.
- **Reproduced:** `TransferFunctionExt` treats Gamma22 as identity, and AdobeRGB→Oklab accepts a plan that later panics for a missing matrix. Primaries helpers cover fewer primaries than the descriptor supports.
- **Reproduced:** scalar F16 conversion rounds the smallest half subnormal multiplied by alpha 0.375 upward to that subnormal instead of zero (`f16_scalar.rs:161`). Preserve subnormal/tie tests across scalar and SIMD paths.
- **Reproduced:** allocating `apply_orientation` (`orient/mod.rs:119`) drops ColorContext, including the identity case; in-place orientation preserves it. `apply_orientation_into` additionally checks only bytes per pixel, allowing differently interpreted descriptors for a raw copy. Validate the intended descriptor contract.
- `ConvertOptions::forbid_lossy` still allows clipping and does not establish exactness across gamut, transfer or alpha transformations. Deprecate the overbroad promise only after its precise replacement exists.

**5. Storage validity is a useful 0.3 break; routine buffer fixes are not**

Make `PixelDescriptor` construction validate layout/alpha combinations. Keep raw CICP open to unknown code points, but do not let a validated pixel descriptor claim meaningful premultiplied alpha for a layout without an alpha channel. Typed views must agree with their physical layout and alignment.

**Reproduced storage defects:**

| Current behavior | Source | Fix / API consequence |
|---|---|---|
| `reinterpret` accepts an alignment that normal construction rejects; a typed RGBA view can retain its Rust pixel type after becoming BGRA | `zenpixels/src/buffer.rs:473,952,2302` | Revalidate invariants; layout changes return an erased or explicitly retyped view |
| `from_imgvec` compacts rows but keeps the old stride, so a later row panics | `buffer.rs:1874` | Correct the stride now |
| `transform_in_place` adopts a valid minimum-extent view but the owned buffer later demands final-row padding; 12-byte input becomes an 18-byte access | `buffer.rs:2194` | Define and validate owned extent at adoption |
| A zero-width, positive-height crop keeps stride with an empty slice; row 1 panics | `buffer.rs:740,2600` | Make zero-area row behavior consistent |
| Typed channel swap helpers reset linear/P3/premultiplied metadata to sRGB presets | `buffer.rs:1250` onward | Preserve unaffected descriptor fields and current context |
| `Cicp::to_descriptor(Rgbx8)` marks the padding lane as straight alpha | `cicp.rs:125` | Derive alpha from format semantics, not merely the layout's fourth lane |

These are safe-Rust invariant and panic findings, not demonstrated memory unsafety.

`MultiPlaneImage::new` (`planar.rs:852`) relies on debug assertions for only part of validation; dimensions, depths and subsampling relationships need runtime checks. `buffers_mut` can invalidate the structure after construction. Separate pixel mutation from checked plane replacement; validate nonzero subsampling factors and reconcile the eight-plane mask with allowed plane counts. Keep this proportional to current planar consumers.

**6. HDR: settle units and lifetimes before expanding algorithms**

Issue #45's distinction among reference white, intensity/peak information, content measurement and mastering provenance should become the shared contract. They have different lifetimes:

- The relative-linear anchor defines what sample value 1 means in nits and travels with current pixels.
- Content-light measurements become stale after pixel changes.
- Mastering metadata describes provenance and needs an explicit retention policy after grading or tone mapping.
- A transfer function alone does not supply every required luminance parameter.

`DiffuseWhite::new` accepts zero, negative and nonfinite values. Add a validated positive finite constructor; deprecate unchecked public construction before making the invariant mandatory. Replace `reference_white_nits` and Unknown→D65 shortcuts with explicit resolution/defaulting APIs.

Close #64's feature-dependent fail-open behavior immediately. Default builds should refuse destructive HDR→SDR crossing unless the caller chooses a supported policy. Linear HDR must not be classified as SDR simply because its transfer is Linear. `ConvertError` is already non-exhaustive; that part of the issue title is stale.

Source review also found inconsistent scale use in `convert_to_sdr`/`new_with_hdr_config`: PQ-normalized values are measured or tone-mapped as if they were relative to the caller's diffuse white/source peak without the required rescaling. Use independent absolute-luminance reference tests, not parity against another path through the same implementation. HLG scene/display interpretation and OOTF assumptions need an explicit policy; do not imply full photometric conversion from an OETF alone.

`hdr::quantize_to` reconstructs source semantics from the target and is value-only despite receiving typed source pixels. Either validate its narrow preconditions or replace it with an anchored conversion that respects source primaries, alpha and range. Measurement naming is inconsistent across queued notes and production docs (`measure_robust` versus `measure_max`); settle semantics before renaming.

Do not gate 0.3 on the complete M×N HDR feature matrix, gain-map fitting or new tone-mapping algorithms. Gate it on honest refusals, correct units and preservation of already-supported data.

**7. zencodec-specific changes and immediate fixes**

| Finding | Evidence | Disposition |
|---|---|---|
| Metadata consisting only of diffuse white is “empty”; `PreserveExact` drops that white | **Reproduced**, `src/metadata.rs:202,298` | Fix now; add independent retention expectations for every field |
| `with_orientation(Identity).with_exif(rotated)` overwrites the explicit choice despite its documentation | `metadata.rs:103` | Track explicitness or remove implicit parsing from the builder |
| Borrowed→owned animation copies drop ColorContext | **Reproduced**, `src/output.rs:380` | Fix now; parity must include interpretation, not just bytes |
| Frame-to-sink returns only OutputInfo, losing timing/index; sink begin lacks current color context | `traits/decoder.rs:173`, `sink.rs:91`, `helpers/mod.rs:91` | Add a frame result/header that preserves timing and color; provide matching dyn access |
| Pull encode callback returns only `usize`, so upstream decode/CMS/I/O failure has no direct channel | `traits/encoder.rs:131`, `traits/dyn_encoding.rs:75` | Add fallible row source, explicit EOF/row-count rules, then deprecate old callback |
| Buffered sink fallback allocates the whole image and trusts supplied sink geometry | `helpers/mod.rs:44` | Validate geometry; document actual allocation behavior; don't use it as proof of bounded-memory streaming |
| Generic gain-map setters exist but dyn jobs lack them; some requested decode render/policy modes default to no-op | `traits/encoding.rs:363,398`, `traits/decoding.rs:176,254` | Static/dyn parity; distinguish advisory hints from required behavior and report unsupported requirements |
| DecodeOutput duplicates source-encoding storage alongside ImageInfo and can be constructed with inconsistent pixels/info | `output.rs:205` | One authority and checked construction; provide complete into-parts extraction |
| Output sizing and total animated-pixel arithmetic can overflow u64 | **Output size reproduced**, `cost.rs:93`, `limits.rs:608`, `estimate.rs` | Checked/saturating arithmetic as appropriate; impossible allocations must reject, not wrap |
| GainMapParams can validate successfully and serialize into parameters that fail validation | **Reproduced** with finite `f64::MAX` gamma; `gainmap.rs:334,794,865,1082` | Add checked wire conversion/serialization; distinguish semantic validity from rational representability |
| Gain-map AVIF source data can be raw AV1 OBUs while its format tag suggests a decodable AVIF container | `gainmap.rs`, GainMapSource contract | Explicit payload kind/configuration or self-contained encoded image; unify alternate color authority |
| Public EXIF serialization bypasses the amplification guard used by retain/filter | `exif.rs:812,841` | Add fallible size-budgeted writer; validate offset arithmetic and opaque offset-bearing data when rewriting |

For gain maps, also reconcile the advertised authoritative backward-direction flag with `direction()` deriving only from headroom. Do not infer mastering metadata from headroom alone. Prefer forwarding validated components and their metadata before adding convenience reconstruction/generation entry points.

The EXIF parser/filter has useful existing bounded and differential tests. The public parse→edit→serialize route needs its own guarantees: relocating opaque MakerNote/SubIFD payloads is not automatically a safe metadata rewrite. Report retained/dropped/unsupported data rather than promising arbitrary byte-faithful rewriting.

**8. Deprecation sequence**

Ship usable replacements in the current minor, port real consumers, then remove wrappers in the coordinated breaking release. A documentation note or closed issue is not a shipped replacement.

| Group | Deprecate/remove plan |
|---|---|
| Already deprecated zencodec ICC shims and placebo `IccMatchTolerance`; v1 descriptor helper | Reasonable removal in next breaking zencodec release after migration to zenpixels ICC APIs/v2. Fix the underlying authority/substitution guarantees too |
| Unsuffixed ISO gain-map parse/serialize and ambiguous `JpegApp2` naming | Remove after migration to explicit format variants; separately introduce checked serialization |
| Legacy ThreadingPolicy variants; ComputeEnvironment::new | Remove deprecated aliases/modes that implementations cannot honor; preserve explicit sequential/parallel and conservative/host choices |
| Old output finalizer | Do not remove yet: the newer finalizer is defective, and real ICC transform coverage still exercises the deprecated one. Reach semantic parity first |
| Legacy ColorManagement / RowTransform; deprecated Adapted/non-cow adapters; old HDR metadata/tone-map wrappers and transfer-blind ICC aliases | Migrate to the corrected CMS contract and ownership/metadata APIs, then remove already-deprecated compatibility surfaces after a consumer check. Do not make legacy default methods that ignore actual pixel format the basis of new integrations |
| `convert_buffer` / `convert_rows` wrappers | Issue #69/#71 comments mention replacements, but `convert_slice_into`, `convert_into` and `convert_in_place` were not found in this checkout's public implementation. Confirm the release artifact and ship replacements before adding removal dates |
| Infallible or misleading APIs | Add checked diffuse white, checked gain-map/EXIF serialization, explicit color resolution, precise preservation policy, and reliable converter worker creation; only then deprecate ambiguous construction/resolution/Clone paths |
| Raw `with_metadata` / `set_metadata` | Already deprecated for callers, but still the implementation primitive behind policy filtering. Design a supported codec-implementation hook before removing it; do not accidentally make migration impossible |
| `SourceColor::has_hdr_transfer` | Queued removal is not enough. Provide the authority-aware replacement first; distinguish conservative “any HDR tag” detection from the current pixel encoding |
| OutputInfo→PredictedOutputInfo | Do not mechanically apply the queued rename: the type also represents completed sink output. Split prediction from actual result, or retain a name valid for both roles |
| ResourceLimits removals | **Reject the queued removal rationale:** the fresh consumer audit found output-byte enforcement in seven codec repositories, animation-duration enforcement in three, and both check helpers called by zenpdf. Clarify encoded-output versus decoded-pixel budgets and report enforcement honestly |

Avoid gratuitous breaks: registry internals are already `pub(crate)`; ColorContext and ConvertError are already non-exhaustive; ZenCmsLite already has its simplified shape. Keep useful math functions free functions. Do not seal the public Pixel trait without a migration for the custom implementations its docs invite. Do not add blanket Send to decoder state: the testkit deliberately covers non-Send dyn decoders.

Lower-priority implementation follow-ups can stay outside the breaking contract work: correct small-population percentile thresholds; distinguish per-row scratch estimates from full-image allocations and potential parallel performance from actual executor behavior; validate raw adapter stride consistently with PixelSlice's minimum final-row extent; reject unsupported layout/primaries paths at construction. The in-place transpose's `cur * width` intermediate merits a 32-bit overflow test (static finding, not executed here). The private, unused XYB profile module has permissive substring recognition and inconsistent sample normalization; remove it or validate it before integration, rather than exposing it as a new feature. Dead extended-gamut helpers likewise are not evidence of a defect in a live public path.

**9. Backlog disposition — every currently open issue**

| zenpixels issue | What belongs in this shift |
|---|---|
| [#71 Free functions](https://github.com/imazen/zenpixels/issues/71) | Secondary API cleanup after the contracts above. Move functions only where a receiver owns meaningful state; verify claimed replacements against actual released code |
| [#69 Allocations](https://github.com/imazen/zenpixels/issues/69) | Include consuming/borrowing/caller-output finalization. Rowwise HDR peak measurement has landed; validate its physical units independently |
| [#64 HDR fail-open](https://github.com/imazen/zenpixels/issues/64) | Immediate correctness fix and a release gate. Its ConvertError exhaustiveness premise is already resolved |
| [#45 HDR hub](https://github.com/imazen/zenpixels/issues/45) | Include shared luminance and metadata-lifetime contract, anchored conversion/refusal and component preservation. Defer the full algorithm/format matrix |
| [#42 PQ16 SIMD](https://github.com/imazen/zenpixels/issues/42) | Current `convert_kernels.rs:1979` onward already implements dispatched precise-power PQ SIMD and chunked F32→PQ16. Reconcile/close the old scalar claim; retain performance and tail-parity gates |
| [#36 CICP/ICC bridge](https://github.com/imazen/zenpixels/issues/36) | Much is implemented. Consolidate remaining matrix-resolution semantics with real consumers; fix RGB/YCbCr and synthesis correctness. Mostly additive |
| [#23 Native f16](https://github.com/imazen/zenpixels/issues/23) | Existing descriptor/scalar support is present. Native typed f16 is consumer/MSRV/toolchain dependent; do not block 0.3 or invent an unstable public type without a concrete need |

| zencodec issue | What belongs in this shift |
|---|---|
| [#122 Upstream policy](https://github.com/imazen/zencodec/issues/122) | Select small, demonstrated shared primitives. Define resource-accounting and supplementary-image retention contracts. Keep format-ranking preferences and calibration data with their owners; do not upstream 1,400 lines wholesale |
| [#121 FormatSet / Custom](https://github.com/imazen/zencodec/issues/121) | Real custom-format limitation; an extensible set/registry mechanism can be additive. PDF/DNG/RAW are already built-ins here, so the issue examples need updating |
| [#120 Quality calibration](https://github.com/imazen/zencodec/issues/120) | Include a defined metric/scale and honest unsupported behavior. Default identity mapping of perceptual scores to native quality is not calibration. Keep implementation-specific calibration versioned by its actual codec owner |
| [#104 Input-dependent fidelity](https://github.com/imazen/zencodec/issues/104) | Central to this release. Per-input precision/alpha/palette/HDR support plus achieved outcome. The June 23 comment withdraws the proposed FidelityMatch API; design from current needs rather than resurrecting it verbatim |
| [#27 Format-agnostic consumers](https://github.com/imazen/zencodec/issues/27) | Use as contract acceptance checklist. CodecSet has landed; gain-map-free reconstruction behavior is already documented. Remaining color resolution, required negotiation and actual orientation reporting matter |
| [#26 Per-codec Fidelity](https://github.com/imazen/zencodec/issues/26) | Consumer migration and conformance gate. NearLossless-era portions are stale relative to today's Lossless/Lossy model |
| [#24 Gain-map pipeline](https://github.com/imazen/zencodec/issues/24) | Generic fallible component setters already exist; complete dyn parity and retention semantics. Preserve unconditional native-HDR coverage; later phases remain demand-driven |
| [#12 LosslessMode](https://github.com/imazen/zencodec/issues/12) | Historical design superseded by later work and withdrawals. Do not add the original three-way enum merely because the issue remains open; retain useful compatibility setters until a real replacement exists |
| [#11 Decoder audit](https://github.com/imazen/zencodec/issues/11) | Most initial fixes were reported completed. Audit remaining source-color/progressive reporting through adapter conformance; fixed 8-bit facts for GIF/WebP should remain facts |
| [#1 PagedDecoder](https://github.com/imazen/zencodec/issues/1) | Worth a concrete TIFF/PDF design during the shared trait change. Pages need independent dimensions, metadata and negotiation, not animation semantics. Add a separate trait when possible; coordinate any required associated-type/job change now |

No issues were posted, closed or edited by this review.

**10. Release gates and rollout**

**Narrowing performance correction (2026-09-24):** the initial review omitted an important U16→U8 regression. The August correctness fix replaced garb's runtime-dispatched AVX2 path with a generic byte-lane loop; the recorded 4.5× improvement was measured only on an M4 Pro, where garb lacked a NEON implementation. Fresh testing on a Ryzen 9 9950X3D with rustc 1.98.1 and default target flags shows the public RowConverter path at 7.49 GiB/s versus garb's 42.6 GiB/s for 256-pixel RGB rows (286.3 versus 50.4 ns, approximately 5.7× slower). The benchmark warns of only four rounds on this shared host; treat these as diagnostic measurements, not final calibration. The standalone byte-lane candidate is similarly slow, locating the problem below RowConverter overhead. Its generated x86 code assembles vectors through many individual byte loads and unpack instructions. This belongs in the immediate fixes and release gates, without requiring an API break: retain exact `(v + 128) / 257` rounding, restore efficient runtime-dispatched x86 execution, and require default-target x86 plus ARM comparisons before making cross-platform speed claims. Reverting to garb's old arithmetic would reintroduce its 127 incorrect outputs among 65,536 input values. The existing exact shift-u32 candidate improves on byte lanes here but still trails garb; it is not sufficient evidence that the regression is solved.

The completed [x86 benchmark](../benchmarks/u16_narrow_2026-09-24_x86.txt), with [machine/command metadata](../benchmarks/u16_narrow_2026-09-24_x86.meta), shows the regression at every tested size:

| RGB pixels per call | Old garb GiB/s | Current RowConverter GiB/s | Approximate time multiplier |
|---|---:|---:|---:|
| 256 | 42.6 | 7.49 | 5.7× |
| 4,096 | 26.8 | 5.56 | 4.8× |
| 1,920 × 1,080 | 32.5 | 6.84 | 4.7× |

All groups had only four accepted rounds; the 4,096-pixel group also had substantial variance. The consistent large effect and generated code establish a performance problem, while final calibration should use a quieter host. The 1080p arm passes the whole image as one long row, so these are conversion microbenchmarks rather than end-to-end codec timings.

1. **Current-line correctness/performance patch:** finalizer and adapter relabeling; CMS composition/clone; buffer invariants; diffuse-white/frame-context retention; overflow; profile aliasing; default HDR refusal; exact U16→U8 narrowing without the x86 slowdown. Add targeted behavioral tests and appropriate comparative benchmarks alongside each fix.
2. **One small shared contract draft:** exactness vocabulary, current color versus provenance, validated descriptor/anchor, required versus preferred negotiation, accepted versus preserved encoder inputs. Amend the existing negotiation proposal instead of implementing its current zero-cost proof rules.
3. **Replacement APIs and deprecations:** introduce complete plans/output ownership forms, fallible row source and serializers, and dyn parity. Exercise real codec implementations, including one lossless high-depth codec, one lossy/8-bit codec and an HDR/gain-map path.
4. **Consumer migration and breaking release:** remove only obsolete APIs with working replacements. Include page integration only if a real consumer implementation is ready. Publish dependency changes together and inspect packaged manifests, not only local path builds.

The dependency audit is a release gate: zenpixels-convert's workspace requirement is already `>=0.2.16, <0.4.0`, so an older published converter may accept a genuinely breaking zenpixels 0.3. zencodec's checked-in requirement is still `^0.2.14`, which does not. A targeted sibling audit found 56 zenpixels declarations across 46 manifests/21 repositories admitting 0.3; 25 zenpixels-convert declarations across 23 manifests/12 repositories do likewise. There are 31 zencodec declarations across 17 repositories admitting its next breaking 0.2, and 10 zencodec-testkit declarations admitting 0.2. Counts include dev/workspace declarations, not just published crates. Test previously published consumers against the new artifact, or narrow/fix incompatible requirements before release. Keep no_std and existing MSRV commitments unless the chosen implementation demonstrably requires a change.

The same consumer audit found 20 production push-decoder implementations across 12 codec repositories returning OutputInfo, plus five animation sink overrides. AVIF explicitly returns the descriptor actually delivered (`zenavif/src/codec/decode_job.rs:714`) and tests it against the sink. GIF, JPEG AI, JPEG, PNG, AVIF, JXL and WebP enforce output-byte limits; GIF, AVIF and WebP enforce animation-duration limits. zenpdf calls both queued-for-removal limit helpers (`zenextras/zenpdf/src/zencodec_impl.rs:221`). JPEG uses the byte cap for both encoded data and decoded pixel output, supporting a semantic split rather than deletion. This was a targeted consumer audit, not a full source review of those additional repositories.

Conformance needs semantic oracles:

- Compare bytes **and** current color/anchor, alpha semantics, dimensions, orientation, timing/index/loop count and retained supplements across static/dyn, borrowed/owned and buffered/streaming paths. Current testkit `Pixels`/`grab_ref` drops ColorContext; animation comparisons omit timing/index.
- Separate exact original-pixel roundtrip from parity among decode paths of a lossy encoding. `check_all` currently assumes RGBA8 plus exact original roundtrip; support codec-appropriate input fixtures.
- Exercise real saturated ICC transforms through the new finalizer, both profile-authority conflicts and unavailable-CMS refusals. Neutral samples and fake ICC blobs cannot establish color correctness.
- Invert tests that require implicit Unknown↔known relabeling; retain an explicit-assumption test. Treat ΔE/model-cost tests as heuristic calibration, not exactness proofs.
- Check requested row coordinates, counts, descriptors, EOF, source/sink error and cancellation. Testkit's current encode-from helper ignores y and copies only the minimum available bytes, masking bad protocols.
- Use independent metadata retention expectations with every field populated. Add alpha-zero, nonfinite floats, minimum final-row extent, zero-area, narrow-range and multi-plane mismatch cases.
- Run minimal external consumer manifests for feature combinations. The convert crate's self dev-dependency enables CMS and other features during ordinary tests; `--no-default-features` inside that workspace is not sufficient evidence of a true no-CMS build.
- Preserve the strong transfer-alpha and exhaustive rounding tests, add independent absolute-HDR oracles, and keep required ICC fixtures hermetic. Optional corpus/network tests may supplement those gates, not replace them.

Verification during this review: an external consumer suite passed 14 assertions with std and 15 with convert's std feature disabled; the extra assertion demonstrates lost CMS state on clone. Two other isolated suites passed seven core and ten conversion defect assertions. That is **32 distinct reproduced behaviors**, not 46 independent tests: the feature runs share 14 cases. These assertions deliberately recognize defective current behavior, so passing is evidence of reproduction, not evidence the libraries are correct. Reproduction sources are retained for this session in `/tmp/zenpixels-03-review`, `/tmp/core-foundation-review` and `/tmp/convert-foundation-review`. The conversion suite also reproduces inverse-quantization elision described above.
