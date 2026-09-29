# Metadata, image and media boundaries before the compatibility freeze

Review proposal, 2026-09-28. This document proposes APIs and dependency changes;
it does not claim that the new crates, features or methods already exist.

**Recommended decision:** keep the stable interchange vocabulary small, give
ordinary codec users one metadata convenience feature, and let detailed metadata
tools depend directly on an independently versioned engine. Keep image-only
users independent of media containers, networking, audio and video backends.

Read sections 1–4 for the design and examples, 5–7 for gain maps/video/features,
and 8–10 for implementation order and the six-commit review.

## 1. Ownership and repository boundaries

| Repository | Crate | Owns | Must not expose or acquire |
|---|---|---|---|
| zenpixels | zenpixels | Sample encoding, pixel/color descriptors, validated storage/views, orientation, canonical display values | Metadata document models, XML, codecs, conversion engines, timestamps |
| zenpixels | zenpixels-convert | Executable conversion, CMS integration, explicit quantization/measurement/tone mapping | A mandatory dependency from metadata or codec contracts |
| zencodec | zencodec | Image codec protocols, metadata transport, stable policy/error facade, display-retention requirements | Public zenmetadata document/error types; mandatory media backends |
| New zenmetadata repository | zenmetadata | One EXIF/TIFF walker, XMP model/editor, supported MakerNotes, binary gain-map metadata, inspection/edit/audit/diff | zencodec, pixel conversions, full image/container decoders, compression/network stacks |
| zencodec | Proposed zencodec-media-core | Exact time, tracks/packets, codec configuration, media interfaces | HTTP, temporary files, XML, tone mapping, codec implementations |
| zencodec | Existing zencodec-media experiment | Container implementations, sessions and optional backend/image adapters | Automatic reexport through the stable image API |
| Codec repositories | Their existing crates | Bitstreams and container framing, discovery, offsets, lengths, checksums, carrier limits | Independent copies of the EXIF/XMP privacy engine |
| zenpipe | zencodecs / pipeline | Registration, format selection, application policy and runtime wiring | Mandatory full-document auditing or pixel scans on ordinary operations |

Repository boundaries are maintenance boundaries; crate boundaries determine
compilation and public type identity. Keep core/convert together and keep the
media experiment with its contract/testkit during development. Do not create an
EXIF crate, XMP crate, MakerNote crate, audit crate and adapter crate chain.

The media contract extraction is justified by real consumers: MP4/WebM demuxers,
AV1 encoders/decoders, image-animation adapters and the new audio session code.
Extract it without publishing until those callers exercise the interfaces.
Image animation can adapt its exact duration to media time with checked rational
arithmetic; it does not need to depend on the entire media layer.

The proposed dependency graph has no cycles:

```text
codec implementations / zencodecs
    +--> zencodec ---------------------------> zenpixels
    |       +--> zenmetadata [metadata] ------> zenpixels
    +--> zenmetadata [minimal required slice] -> zenpixels

zencodec-media
    +--> zencodec-media-core ----------------> zenpixels
    +--> container/codec implementations [selected]
    +--> zencodec [image-animation adapter]
    +--> zenmetadata [selected metadata operations]
```

The second direct metadata edge is intentional: a decoder extracting EXIF
orientation needs reading, not the facade's full editing convenience bundle.
Cargo unifies compatible package features; it does not build one engine per
codec. An incompatible engine version can still duplicate code internally, so
keep the ecosystem on one engine line where practical without exporting its types.

### Shared display types

zenmetadata consumes canonical Orientation, Cicp, ContentLightLevel,
MasteringDisplay and DiffuseWhite from zenpixels with minimal features. It does
not allocate PixelBuffers. ICC stays opaque bytes; arbitrary ICC sanitization is
out of scope. Known-profile normalization remains an explicitly wired service
from zenpixels-convert, never a reason to add conversion to the engine graph.

Move the genuinely format-independent gain-map rendering values out of
zencodec into zenpixels before freezing them, retaining selected zencodec
reexports for migration. Do not simply relocate the current ISO-specific struct:
first settle direction, color-space authority and validation. ISO flags and
serialization live in the metadata engine; container declarations and decoded
auxiliary payloads remain codec/media responsibilities. Unknown vendor models
remain identifiable and opaque rather than being forced into ISO defaults.

Raw values, unsupported structures and source locations remain accessible in
zenmetadata even when conversion to a validated display value fails.

## 2. One convenient facade; independently usable engine

The previous proposed blanket ban on zencodec -> zenmetadata is too restrictive.
Use an optional **private implementation dependency** for common filtering.
Keep zencodec's policies, errors and prepared output types owned by zencodec.
Do not return zenmetadata::Document, parser errors or XML nodes from codec traits.
Even a feature-gated public reexport would freeze that dependency's type identity.

The ordinary application enables one feature and uses the familiar codec API:

```toml
# Proposed convenience feature, not an existing 0.3 manifest.
[dependencies]
zencodec = { version = "0.3", features = ["metadata"] }
```

Direct engine users select only what they need:

```toml
# Version placeholder: zenmetadata has not been published by this work.
zenmetadata = { path = "../zenmetadata", default-features = false }
# EXIF/TIFF inspection, including bounded raw MakerNote visibility.

# Add features = ["xmp"] for namespace-aware XMP inspection.
# Add features = ["write"] for EXIF edits and serialization.
# Add both for XMP edits and serialization.
```

Keep docs.rs navigation shallow: zencodec::metadata for policies/preparation and
zencodec::encode / decode for jobs; zenmetadata::{exif, xmp, makernote, edit,
audit} for detailed tools. Root-reexport only the few established entry types.
Do not mirror the entire engine tree through zencodec or duplicate it under
multiple convenience paths. Retained bridge aliases get one migration example.

Decode essentials are enabled by the codec that promises them. A gain-map
decoder whose supported representation requires XMP enables XMP reading itself;
users must not discover a second switch after decoding incorrectly. Lightweight
orientation reading must not enable general XMP or writing.

Runtime services remain an advanced alternative to the convenience backend:
the facade must permit an explicitly supplied implementation without compiling
its built-in engine. Reuse and narrow the existing scrub Services concept before
publishing another trait. A trait object call per metadata packet/job is acceptable;
do not put it in a per-pixel/per-tag hot loop or require global mutable registration.
Backend selection is explicit; enabling a Cargo feature must not override a
caller-supplied backend or change an already-supported operation's semantics.

## 3. Public API shape and what becomes invalid

Use three concepts, with names provisional until prototyped against real callers:

1. **Metadata**: transport bytes and interpreted source information. Attaching
   bytes is not a promise that they are valid or safe to publish.
2. **MetadataPolicy**: small stable publication intent, independent of detailed
   engine tag selectors. Appearance preservation is a separate requirement.
3. **PreparedMetadata**: private fields, constructed by fallible target-aware
   preparation. The codec accepts it and remains responsible for final layout.

Do not introduce a second universal pixel ownership wrapper. Preparation borrows
existing source metadata and the actual output display description; pixel views,
PixelCow and buffer parts retain their existing roles. Prepared metadata is tied
to a target capability description and output interpretation. Reusing it with a
different target/color/orientation/gain-map relation must be checked or rejected.
It is not a whole-file privacy certificate.

**Old code that is insufficient:**

```rust,ignore
// Metadata::filtered cannot see auxiliary images, hidden carriers or the target.
let filtered = decoded.info.metadata().filtered(&MetadataPolicy::Web);
encoder.with_metadata(filtered).encode(transformed_pixels)?;
// It must not imply that this output preserves gain maps or is fully scrubbed.
```

**Proposed explicit preparation:**

```rust,ignore
// API sketch: actual output pixels/context and retained auxiliary data are inputs.
let prepared = source_metadata.prepare_for(
    output_display,
    encoder.metadata_capabilities(),
    &MetadataPolicy::web(),
    &limits,
)?;
let encoded = encoder.job()
    .with_prepared_metadata(prepared)
    .encode(output_pixels)?;
```

This is the explicit form. The ordinary job convenience should do the same work:

```rust,ignore
let encoded = encoder.job()
    .with_metadata_policy(source_metadata, MetadataPolicy::web())
    .with_gain_map_policy(GainMapPolicy::preserve())
    .encode(output_pixels)?;
```

The policy setter records intent. Fallible preparation occurs at job startup,
before encode output is emitted. It must not keep today's infallible setter's
silent fallback behavior. Static and dyn paths share the same preparation.
Source/current color must already be distinguished; source provenance cannot
silently label transformed pixels. The job must have the coupled auxiliary data
to honor preserve; selecting preserve does not conjure a missing gain map.

The raw-carrier escape hatch remains explicit and makes no privacy guarantee.
Existing infallible filtering/embedding helpers receive precise deprecations in
the bridge before removal or changed signatures in the breaking release.

For encoded-file rewriting, retain the existing plan/chunk approach:

```rust,ignore
// The facade dispatches to a registered container rewriter, not a pixel decoder.
let plan = codecs.rewrite_metadata(encoded_input)
    .policy(MetadataPolicy::web())
    .gain_maps(GainMapPolicy::preserve())
    .prepare()?;
show_summary(plan.report());
for part in plan.parts() {
    output.write_all(part.bytes())?;
}
// Explicit to_vec() is available when the caller actually wants a new buffer.
```

An image job convenience can be used through zencodec alone with a registered
codec. A multi-format rewrite requires registered container handlers; enabling
zencodec/metadata does not pull all containers into the foundation. Existing
CodecSet can own that dispatch without introducing another speculative registry.

Detailed reading/editing uses zenmetadata directly, without pixel codecs:

```rust,ignore
let document = zenmetadata::exif::Document::read(exif_bytes, &limits)?;
for entry in document.entries() {
    inspect(entry.key(), entry.value(), entry.source_range());
}
let mut edit = document.edit(); // write feature
edit.remove_gps();
let revised = edit.serialize(&limits)?;
let diff = document.diff(&revised.read(&limits)?);
```

These names illustrate the model, not a commitment to an exhaustive public
method family. One bounded walker supplies both inspection and editing:
inspection exposes partial coverage/diagnostics; a rewrite cannot silently treat
an incomplete traversal as proof that all private entries were removed.

## 4. Guarantees, policy, errors and cost

Keep disposition, coverage and cost distinct in reports:

| Dimension | Examples |
|---|---|
| Disposition | Kept, removed, rewritten |
| Coverage | Fully interpreted within supported schema; opaque; unsupported; malformed |
| Work | Metadata bytes parsed/allocated, container bytes copied, pixel passes required |

Always emit a compact operation summary and error diagnostics. Detailed per-tag
auditing is requested at runtime and borrows/lazily formats values; do not build
String-valued snapshots of every entry on each decode. Start with typed engine
values and source spans, not the flat string records in the current audit PR.
No separate audit/diff feature unless measurements establish a material benefit.

Use zencodec-owned non-exhaustive error categories for malformed input, limits,
missing capability, contradictory display declarations and preservation conflicts.
Preserve an optional source error chain without exposing engine-specific types.
Private fields permit evolution; non_exhaustive does not excuse ambiguous semantics.

Cheap validation is unconditional. Optional capabilities may add supported
operations, never weaken checks or turn a successful operation into different
output. Missing support produces an error, not default HDR parameters, silent
metadata retention or a stripped alternate rendition.

Whole-image scans and decode/reencode are never hidden inside filtering.
Prepared output errors precede writes for supported in-memory planning. Actual
I/O errors may leave partial output; failure-atomic file replacement needs a
caller-selected temporary destination. Do not describe write_all as transactional.

## 5. Gain maps and other layout-dependent metadata

A gain-map representation couples base image, auxiliary image, rendering values,
color interpretation and container linkage. Choose preserve/base-only/render
explicitly. Base-only is not necessarily SDR. Pixel modification invalidates the
relationship unless a supported operation updates/recomputes it or the caller
explicitly chooses an alternate output representation.

For the supported primary-plus-one-gain-map JPEG layout:

1. Read all required render information before filtering source XMP/MakerNotes.
2. Filter the auxiliary JPEG and serialize its required metadata.
3. Measure its new encoded length and supply it to the primary XMP directory.
4. Serialize primary metadata, then calculate final primary size/MPF location.
5. Regenerate MPF references using their specified origin, then emit borrowed
   compressed scan payloads plus newly owned headers.
6. Reparse output in tests; validate references against actual image boundaries,
   compare render values and independently confirm unchanged coded payloads.

The container writer supplies authoritative structural properties to zenmetadata's
editor. A retained source property cannot override the new directory. The editor
preserves selected XMP meaning, not necessarily packet bytes/prefixes/ordering.
Unknown relocation-sensitive properties cannot be certified safe after arbitrary
edits. Extended XMP requires reassembly, regenerated lengths/chunk linkage and
bounded fragmentation support; the current edit path rejects it.

This is also why the writer belongs with each container. JXL box lengths, PNG
chunk CRCs, MP4 item extents and video sample tables have different rules. Each
writer owns its layout finalization; generic metadata code owns packet content.
Compressed metadata uses container-provided bounded decompression/compression
services, keeping zlib/Brotli out of the generic document engine.

For MakerNotes, reading, extracting display essentials, and relocating/rebuilding
are distinct capabilities. Start with Apple rendering fields already consumed in
heic/ultrahdr/zenraw. Unknown records remain visible and opaque. Preservation and
privacy demands that cannot both be honored return a conflict. Do not promise a
complete vendor database or pretend all offsets are relative to the note itself.

## 6. Video and future audio

Promote tested native storage/view contracts from the media experiment into
zenpixels only after exercising packed, planar and semiplanar consumers. Keep
the deprecated planar API removed. Require per-plane stride, meaningful sample
bits/alignment, subsampling/siting, coded geometry and visible crop. CPU borrowing
must not require Vec ownership or a packed RGB roundtrip. External/hardware
surfaces need an explicit mapping/copy boundary; do not invent a GPU abstraction
before a backend needs it.

Media contracts keep compressed packets separate from decoded frames. Demux and
remux must support audio passthrough without an audio decoder. Preserve exact
signed PTS/DTS, explicit missing times, duration, configuration changes, channel
description where available, codec delay and trimming, and track dispositions.
Unknown tracks are surfaced; dropping one requires an explicit decision.

PR zencodec#129 now contains MP4/WebM and audio session code; its description
lags the branch. Before promoting its public contracts, address:

- Vec-only MediaPacket payloads and owned separate Vec planes constrain reuse.
- Required pts cannot represent an unavailable timestamp.
- TrackSpec public fields freeze construction shape; use validated construction.
- Other(&'static str) cannot retain arbitrary identifiers read from files.
- AudioInfo's channel count alone cannot capture arbitrary channel layout.
- A demuxer must not claim one compressed packet always equals one decoded frame.

Keep bounded queues/backpressure, drain/EOF and cancellation explicit. Seekable
file fixups and forward-only segment emission are different writer capabilities.
Do not generalize an in-memory image rewrite plan into an unbounded whole-video
plan. No implicit network access or temporary-file creation in core contracts.

## 7. Feature slices: two engine switches, one convenience bundle

Proposed engine features:

```toml
# zenmetadata; no_std + alloc core. XML dependency is private.
[features]
default = []
write = []
xmp = ["dep:roxmltree"]
```

Use the existing namespace-aware parser initially, own the metadata model and
serializer, and keep XML dependency types private. Confirm the exact selected
parser/MSRV/no_std feature combination before publication. Writers are compiled
for EXIF/binary metadata with write, and for XMP with both write and xmp.
Default includes bounded EXIF/TIFF reading, typed/raw entries, diagnostics,
small supported MakerNote interpretation and binary gain-map metadata reading.
Do not add per-vendor, per-tag, read, validate, privacy, gainmap or diff flags.

For zencodec, add one optional convenience bundle:

```toml
[features]
metadata = ["dep:zenmetadata", "zenmetadata/write", "zenmetadata/xmp"]
# Existing std feature remains independent.
```

This intentionally buys a working default filtering backend with one switch.
Applications that need fine-grained engine control use it directly or supply a
service. Do not multiply facade flags into metadata-exif-read, xmp-write, etc.
Keep the bundle default-off in the contract crate. A high-level application/CLI
can explicitly include it in its chosen defaults after the cost is measured.

| User operation | Required slice | Deliberately excluded |
|---|---|---|
| Use shared codec traits / inspect already interpreted ImageInfo | zencodec core | Metadata engines, conversion, media implementation |
| Read EXIF/TIFF or inspect opaque MakerNotes | zenmetadata core | XML, writers, pixel codecs |
| Read detailed XMP | core + xmp | Writers, pixel codecs |
| Edit EXIF or serialize binary display metadata | core + write | XML unless the output representation requires it |
| Edit XMP / usual publication filtering | core + xmp + write; zencodec/metadata convenience | Pixel scans, conversion, compression backends |
| Decode a supported gain-map representation | Codec decode plus its required metadata readers | Gain-map rendering unless explicitly requested |
| Encode from new pixels | Selected encoder + required small metadata serializers | Source-document audit, demux/network stack |
| Scrub/remux encoded content | Container rewrite + required metadata read/write | Pixel decoder/encoder and audio decoder |
| Pixel transcode | Selected decoder + encoder + requested conversion + metadata preparation | Unselected codecs, implicit tone mapping/analysis |
| Video with audio passthrough | Selected demux/mux + packet contracts | Audio decoder/encoder |

The encoder enables the writer it requires; users do not separately toggle
orientation serialization, CRC repair or MPF fixups. Reuse existing codec
encode/decode feature cuts where they save meaningful compilation. Do not add
read/write/transcode to every crate: transcode is composition, not another
implementation. Container rewrite must be usable without a full pixel backend;
extract a container subcrate only when its existing parent cannot provide that
isolation without dragging a heavy backend. Do not preemptively split every codec.

For media, keep container selection (MP4/WebM), real backend selection and HTTP
optional. Move tempfile/snapshot support out of the core path. Keep inexpensive
types and validation unconditional; do not expose a combinatorial set of flags
for timing, audio packets, drain, limits, crop and native precision.

Cargo features are additive and unify across dependents. Use default-features =
false in leaf dependencies and dep: for private optional dependencies. Test the
unified graph too; a read-only dependency may receive write through another user.
See [Cargo features](https://doc.rust-lang.org/cargo/reference/features.html).

## 8. Compilation and behavior acceptance gates

Features are justified by saved dependency/build work, not the number of cfgs.
Prototype the four engine combinations before publishing write/xmp. If a write
gate saves negligible work, include the small writer unconditionally rather than
permanently carrying a useless switch. XML is the meaningful dependency cut.
Do not claim that dead-code elimination removes parsing/type-checking/codegen cost.

Measure archived, pinned source with identical toolchain/jobs/profile and fresh
target directories. Report median and range of at least three interleaved
baseline/candidate runs, plus warm and engine-source-invalidated checks. Measure
check and build separately, critical path, normal/build dependency and proc-macro
counts, and incremental invalidation of a representative codec consumer.
Run from consumer fixtures without workspace dev-dependency feature pollution.

Mandatory build cases: traits-only; EXIF read; EXIF write; XMP read/write;
JPEG gain-map rewrite without pixel backends; PNG/JXL compressed metadata;
minimal decode; ordinary pixel transcode; media demux plus audio passthrough.
Verify no converter/codec/HTTP/XML reaches the cases that do not request it.
Set budgets from these baselines before implementation; a repeated regression
above both 5% and 100 ms triggers explanation/rework rather than being averaged
away. This is a proposed review threshold, not an established project benchmark.

Behavior gates include namespace aliases and duplicate XMP declarations;
malformed/cyclic/out-of-bounds TIFF; unknown tags/vendor offsets; actual privacy
removal; gain-map render equivalence and rebuilt offsets; unchanged compressed
payloads; unsupported cases refusing without fake success; bounded allocation;
caller-owned/borrowed storage; static/dyn and buffered/streaming parity; explicit
audio track retention and exact trim/timing. Fuzz the shared walker and each
container rewriter independently. Differential oracles remain dev-only.

Existing scrub timings compare the already-integrated metadata branch against
the scrub contract addition, not against pre-metadata main. Existing bridge
timings likewise do not measure a hypothetical zenmetadata extraction. Do not
reuse either as proof that the proposed dependency graph is cost-neutral.

Fresh bridge measurements from this review (three serial paired runs, rustc
1.98.1, four jobs, archived sources, offline locked dependencies):

| cargo check case | origin/main median (range) | bridge median (range) |
|---|---|---|
| Minimal core, fresh target | 1.341 s (1.317–1.347) | 1.330 s (1.291–1.338) |
| Default converter, after core | 3.208 s (3.195–3.212) | 3.290 s (3.264–3.307) |
| Core source invalidated | 0.068 s (0.067–0.071) | 0.071 s (0.071–0.072) |
| Converter source invalidated | 0.257 s (0.254–0.259) | 0.269 s (0.268–0.272) |

The converter increase is about 83 ms / 2.6% in this check workload. These are
local observations, not build/link timings or a performance guarantee. No new
dependency was introduced by the bridge. Warm checks were about 20–24 ms.
[Raw samples and harness provenance](../benchmarks/bridge-review-compile-2026-09-28.json).
Two preliminary runs were excluded: one overlapped the targeted tests, and an
initial repeat loop stopped when the branch switch removed its script path.
All three recorded runs subsequently completed using the preserved harness.

## 9. Existing PRs and implementation order

1. Preserve the six bridge commits and candidate snapshots on named branches.
   Integrate reviewed correctness fixes before release/version staging.
2. Reconcile zenpixels#76's sample vocabulary with the bridge. Consolidate
   #77's native-sample requirements and planar retirement with the local 0.3
   candidate. #78 overlaps actual-current-profile conversion in the bridge;
   retain the stronger behavior/tests, not two finalization implementations.
   #79's known ICC service remains explicit and outside metadata core.
3. Extract zencodec#123–125/#128 EXIF/XMP work into zenmetadata. Use one checked
   traversal, with typed/raw values and coverage. Keep facade policy ownership
   in zencodec; do not move the whole codec Metadata struct into the engine.
4. Replace xmpkit workaround glue in zenpipe#87 with the owned model/editor,
   retaining the implementation as a differential fixture where useful. Keep
   JPEG/PNG/JXL layout code with container owners and expose their minimal
   rewrite capability to the common facade. Preserve existing bounded plans.
5. Update zencodec#130, zenjpeg#213 and zenpipe#84/#87 integrations to use the
   same policy preparation; retire duplicate strict/forgiving EXIF behavior.
   Previous review found selected HEIC and UltraHDR revisions missing relevant
   privacy changes: recheck and integrate heic#51 / ultrahdr#35 explicitly.
6. Extract minimal media contracts from #129 and exercise native AV1 plus audio
   passthrough before freezing. Update its PR description to match its code.
7. Gate publication on real source-identical consumer builds, one shared type
   version per connected pipeline, minimal/default/unified feature graphs and
   the byte-level container tests. No crates.io publication is authorized here.

Existing zenpipe#48 automatic peak measurement needs an explicit cost policy;
previous review found a full-frame RGBF32 temporary. Do not copy that cost into
default metadata preservation. Keep measurements rowwise when explicitly used.

## 10. Review of the six local commits

Reviewed against origin/main 197a38b. The local bridge tip was 1d47131; the local
0.3 candidate was 6656cde. Review covered commit contents, key storage/conversion/
CMS/finalization paths, tests, recorded compatibility checks, manifests and merge
overlap. This is not a new exhaustive audit of every modified line or backend.

| Commit | Disposition | Reason / remaining caution |
|---|---|---|
| 373f600 | Keep; correctness PR eligible for main | Descriptor/storage/alignment validation, zero-area views and reuse fix real defects; stricter inputs and panicking legacy setters need accurate migration notes |
| 3340940 | Keep on reviewed implementation branch until reconciliation | Substantial prepared conversion, alpha/HDR/CMS and output changes; desired default optimized composition and explicit compose_preserving are covered. Overlaps #78; review new public APIs and fallback/clone semantics before freezing |
| c9fea15 | Keep with 3340940 | Restores published std Sync through exclusive Mutex::get_mut, with no worker locking. ConvertError still loses unwind auto-traits: documented tolerated exception, not a completely clean semver result |
| 1b09e2e | Keep alongside the implementation | Useful canonical documentation and reproducible compile harness; original measurements were single-run checks, not statistically established gains |
| 835612e | Release staging branch | Stages both crates at 0.2.17 and raises the converter's real core floor. Carry the dependency floor into the release; do not lose it while splitting version work |
| 1d47131 | Keep as candidate verification evidence | Package/source-compat fixture and status are useful. Claims describe those snapshots/local companions, not every published downstream or the unmerged video stack |

The implementation should eventually reach main through review. The six-commit
candidate should not sit invisibly on local main, nor be pushed directly to main
as if it were already reconciled with the open stack.

Non-checkout merge previews found:

- bridge + #76: conflicts in README.md and docs/public-api/zenpixels.txt only.
  A textual merge still requires semantic/sample-encoding tests.
- local 0.3 candidate + #79 stack: 18 conflicted paths, including convert.rs,
  converter.rs, output.rs, manifests, snapshots and docs. Do not merge both
  independently or resolve these by choosing one side wholesale.

Fresh targeted validation on 1d47131: **41 tests passed** (7 storage, 10
conversion, 10 output, 13 prepared, 1 accepted-plan matrix). That matrix exercises
7,744 descriptor pairs; it is not a numerical oracle for every pair. Prior full
workspace/Clippy/MSRV/package evidence is in the candidate's bridge validation
document and was not all rerun for this design-only change.

Both release branches were pushed without rewriting their history. Local main
was moved back to origin/main only after those snapshots were secured. This
proposal is a separate docs branch. Unrelated dirty zencodec/codec working trees
were not changed, committed or pushed.
