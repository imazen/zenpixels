# Release plan: 0.2.17 (paving) → 0.3.1 (breaking)

**Superseded draft:** the 2026-09-27
[API/contract proposal](api-contract-proposal-0.2-and-0.3.1.md) replaces this
plan's scope and migration recommendations. In particular, the owner selected
same-source compatibility with one core version per connected pipeline; local
wide dependency ranges were not published, and replacement signatures must
remain unchanged across the bridge and destination releases. The historical
inventory and reference commits below remain useful implementation inputs.

Written 2026-09-26 against `main` = `0e1c659` (both crates at 0.2.16 in
`Cargo.toml`; crates.io latest 0.2.16, published 2026-07-24). Every claim
below was checked against source, crates.io, or a `rg` sweep of `~/work` on
that date; the appendix lists the commands. Where a claim comes from the
2026-09-23 foundation review (`docs/foundation-0.3-review.md`) and was not
re-verified here, it says so.

**The contract this plan enforces:** a consumer that builds warning-free
against 0.2.17 builds unchanged against 0.3.1. Every 0.3.1 removal has a
shipped replacement *and* a `#[deprecated]` pointer in 0.2.17. Breaks that
cannot carry a warning (`#[non_exhaustive]`, feature removals, default-feature
flips) ship in 0.3.1 only with measured-zero victims, or after we migrate the
victims ourselves.

---

## 1. Facts that shape the plan

1. **0.3.0 is burned.** crates.io shows `zenpixels 0.3.0` and
   `zenpixels-convert 0.3.0` published 2026-04-13 and **yanked** (git tags
   `zenpixels-v0.3.0-yanked`, `zenpixels-convert-v0.3.0-yanked`). A yanked
   version number cannot be re-published, so the breaking release is
   **0.3.1** for both crates. (`zenpixels-convert 0.2.4` is also yanked.)
   Every "0.3.0" in `CHANGELOG.md`, `CLAUDE.md`, and source comments means
   0.3.1 from here on.
2. **`main` is semver-clean against published 0.2.16.** `cargo semver-checks
   check-release --baseline-version 0.2.16` on 2026-09-26: 196 checks, 196
   pass, 58 skip, for both crates (`~/tmp/zenpixels-semver-main-vs-0.2.16.log`).
   Everything unreleased on `main` (alpha-through-transfer fix, correctly
   rounded U16→U8, RGB16 SDR LUT correction from PR #73, row-wise
   `convert_to_sdr` peak measurement, CI fixes) is behavior, not API.
3. **PR #63 never reached `main`.** Its 22 commits were merged on 2026-07-24
   into their stacked base `hdr-fixes-2026-07-14` (merge `44f7716`) after #65
   had already merged that base to `main`; 0.2.16 (`640dced`) was cut without
   them (`git cherry origin/main origin/hdr-fixes-2026-07-14` lists all 22 as
   missing). The release commit hand-copied only `PixelCow`, the `*_cow`
   adapters, `InPlacePixels::try_new` — and #63's changelog inventory, which is
   why the queue on `main` cites replacements that do not exist on `main`
   (`convert_slice_into`, `convert_into`, `convert_in_place`, the orientation
   receiver traits, the `estimation-experimental` gate). A mechanical rebase is
   not viable: `git merge-tree` reports 12 conflicted files for the whole
   branch and 20 of the 22 commits conflict individually (`buffer.rs`,
   `adapt.rs`, `ext.rs`, `converter.rs`, both API snapshots, `CHANGELOG.md`,
   deleted examples). §3.1 re-implements it item by item using the branch as
   reference; the exact public-API delta is in §3.1's table.
4. **The queue in `CHANGELOG.md` is wrong in five places.** (a) "demote the
   registry lookup API" — `registry` is already `pub(crate)` (`zenpixels/src/lib.rs:79`,
   in published 0.2.16 too). (b) "remove `ZenCmsLite::extended`" —
   `cms_lite` is `pub(crate)` with no re-export; `ZenCmsLite` is not in the
   API snapshot. (c) "resource estimation and `requires_cms` were made private
   before publication" — false: `pub mod estimate` and `pub use ... requires_cms`
   ship in 0.2.16 (`git show zenpixels-convert-v0.2.16:zenpixels-convert/src/lib.rs`).
   (d) `RowConverter::convert_rows` and `adapt::convert_buffer` are listed as
   deprecated-with-replacement; neither is deprecated on `main` and the
   replacements are absent. (e) A second, 0.2.7-era "Queued breaking changes"
   section survives near the bottom of the file (`repr(u8)` removal etc.).
   Issue #64's premise is also stale: `ConvertError` has been
   `#[non_exhaustive]` since 0.2.14 (`error.rs:19`); the silent PQ/HLG→SDR clip
   on default builds is only the `#[cfg(feature = "hdr-experimental")]` on the
   refusal at `convert.rs:560` plus the gated peak-taking constructors.
5. **Consumers (rg sweep of `~/work`, 2026-09-26, excluding `target/`,
   `.jj/`, this repo, `pre-filter/`, `imagers-research/`, vendored refs):**

   | Item | Live users outside this repo |
   |---|---|
   | `adapt_for_encode{,_with_intent,_explicit}` (non-cow, deprecated 0.2.15) | zenpipe/zencodecs ×5 (`transcode.rs:706`, `encode.rs:824`, `dispatch.rs:171`, `codecs/jxl_enc.rs:130`, `codecs/avif_enc.rs:160`), zencodec-testkit `tests/usage.rs:208` |
   | `ColorContext::from_icc_and_cicp` (deprecated 0.2.6) | zencodec `src/info.rs:374` |
   | `ContentLightLevel::measure` (deprecated 0.2.16) | zenpipe/zencodecs `gainmap.rs:108` (already depends on zenpixels-convert; needs `hdr-experimental`) |
   | `RowConverter::convert_rows` | zenmetrics-cli `decode.rs:344`; zensim-picker-prep ×2 bins; zenavif `examples/drift_reencode.rs:297` |
   | free `convert_row` | zenmetrics: gpu-core `lib.rs` ×2, zenmetrics-api `metric.rs`, gmsd `lib.rs`, cvvdp `pipeline.rs` |
   | `adapt::convert_buffer` | hdr-anchor-vis `main.rs:125` |
   | `PixelBuffer::into_vec` | squintly `variant_gen.rs:297` (on a `zenavif::decode` buffer), all-the-images `test-harness/src/main.rs:93` |
   | free `apply_orientation{,_into,_in_place}` | zenraw (3 files), heic (3), zenjpeg (2), zenavif (2), zenwebp (1), zensim (1), three jxl-oxide forks (4 each) |
   | `estimate::*`, `requires_cms`, `REC2020_V4`, `planar::Plane`, `ByteOrder`, `serde` feature, external impls of any `*Ext` trait / `ColorManagement` / `RowTransform*` | none |
   | `ConvertError` variants matched downstream | `AllocationFailed` ×5, `NoPath` ×1 |
   | `FormatOption { .. }` / `ConversionCost { .. }` literals | none (hits in `coefficient` and `retired/zenimage` are their own types) |
   | `PlaneDescriptor`/`PlaneSemantic`/`Subsampling`/`YuvMatrix` | zenjpeg, heavily (hundreds of sites) |
   | `hdr-experimental` API (`CllMeasure`, `HdrConfig`, `new_with_hdr_peak`, `convert_to_sdr`) | zentone, ultrahdr, zenmetrics, zenpipe, zenjpeg, zenavif |

6. **The widened requirement creates a forward-compatibility obligation.**
   `[workspace.dependencies] zenpixels = ">=0.2.16, <0.4.0"` (commit `4c9162a`)
   means a published `zenpixels-convert 0.2.17` may be resolved together with
   `zenpixels 0.3.1`. That is the intent (one `zenpixels` in every graph), but
   it holds only if `zenpixels-convert 0.2.17` actually compiles against
   `zenpixels 0.3.1`. So the 0.2.17 converter must use none of what 0.3.1
   removes or tightens in zenpixels — enforced by a CI job, §3.6.
7. **`docs/foundation-0.3-review.md` (2026-09-23, uncommitted in the primary
   checkout until now) is a real input.** Its reviewers read every Rust file
   and reproduced 32 defects in scratch consumer crates. Spot-checked here
   against `main`: `output.rs:536` `descriptors_match` compares
   format/transfer/primaries/signal_range and **omits alpha** (a premultiplied
   buffer can be finalized unchanged under a straight-alpha descriptor);
   `adapt.rs:330–337` and `:723–729` zero-copy "transfer-agnostic" arms check
   channel type/layout/alpha/primaries/range but **never require the source
   transfer to be `Unknown`** (linear-tagged RGB8 is relabeled sRGB verbatim);
   `finalize_for_output_with` builds its `RowConverter` from descriptors, so a
   `PluggableCms` never receives the origin or target ICC bytes — the
   deprecated `finalize_for_output<C: ColorManagement>` is currently the only
   ICC-capable finalizer. That last point changes a queue item (§5.3).
8. **A lane is live on the U16→U8 x86 regression.** jj workspace
   `../zenpixels--u16narrow` (`devin-u16narrow`, marker fresh at 11:38Z) is
   fixing the byte-lane kernel that the review measured at 4.7–5.7× slower
   than garb's AVX2 path on the 9950X3D (`benchmarks/u16_narrow_2026-09-24_x86.*`).
   Correctness must stay `(v + 128) / 257`. Nothing in this plan touches
   `convert_kernels.rs` until that lands.

---

## 2. Principles

- **0.2.17 is the paving release.** It ships every replacement API, adds
  every missing `#[deprecated]` pointer, and fixes behavior. It contains no
  removals.
- **Technical breaks in 0.2.17 only under the existing tolerated-break
  policy** (`CLAUDE.md` §"0.2.x versioning policy"): each needs the `~/work`
  sweep *and* a `cargo copter` run showing zero victims, and lands under
  `#### Changed (BREAKING, tolerated in 0.2.x)`. The one this plan asks for
  is sealing the extension traits (§3.3). Everything else in 0.2.17 must be
  semver-additive: the release gate is `cargo semver-checks` reporting only
  the items listed in §3.3.
- **0.3.1 removes only what 0.2.17 deprecated**, plus the un-warnable
  tightenings in §5.2 with their evidence. Nothing lands in 0.3.1 that could
  have been additive in 0.2.x.
- **Behavior fixes ride the 0.2.x line** (and are backported after the
  branch point). Silent wrong pixels are shipping bugs regardless of API era.
- **The contract redesign the foundation review argues for (§5.4) is a
  track, not a gate.** It proceeds additively on `main` during 0.3.x; its
  breaking steps batch into the *next* break. Holding 0.3.1 for it would
  reopen the "avoid 0.3.0 indefinitely" trap.

---

## 3. 0.2.17 — the paving release

### 3.1 Additive API (re-implemented from `origin/hdr-fixes-2026-07-14`)

New public surface goes through a PR assigned to `lilith` for signature
review (standing rule for new public API); everything else in 0.2.17 lands
on `main` directly. Reference commits on the branch are given per item; take
the code, not the patches.

| Crate | Item | Replaces / enables | Branch ref |
|---|---|---|---|
| zenpixels | `PixelBufferLayout` (`width/height/stride/offset/descriptor/color_context` accessors) | carrier for zero-copy round trips | `8d67345` |
| zenpixels | `PixelBuffer::into_parts(self) -> (Vec<u8>, PixelBufferLayout)`; `PixelBuffer::try_from_parts(Vec<u8>, PixelBufferLayout)` | the honest `into_vec` (which returns the backing `Vec` *including* the private alignment `offset`, `buffer.rs:1525`) | `8d67345` |
| zenpixels | `PixelBuffer::into_contiguous_bytes(self) -> Vec<u8>` | packed bytes without a second allocation | `ede5e8a` |
| zenpixels | `PixelSlice::new_contiguous`, `PixelSliceMut::new_contiguous` | the packed-layout constructor spelling (`new_packed` was removed before publication; do not reintroduce it) | `649e5f7`, `fcf78ed` |
| zenpixels | `PixelDescriptor::with_color_from_cicp(self, Cicp) -> Self` (const) | relabel helper | `2e1a4fb` |
| zenpixels | `into_contiguous_pixels<P>` reuses the allocation only when the typed `Vec` layout is legal, else copies rows straight into the final `Vec<P>` | behavior fix (no double allocation) | `1126d05` |
| convert | `RowConverter::convert_slice_into(&mut self, PixelSlice, PixelSliceMut) -> Result<(), At<ConvertError>>`; `convert_slice(&mut self, PixelSlice) -> Result<PixelBuffer, _>` | replaces `convert_rows` (six positional args) | `a41332a` |
| convert | `ConvertPlan::convert_row(&self, &[u8], &mut [u8], u32)` | replaces the free `convert_row` | `bc9f8b8` |
| convert | `PixelBufferConvertExt::convert_into(&self, PixelSliceMut)`, `convert_in_place(&mut self, PixelDescriptor)`, `into_converted(self, PixelDescriptor)` | the no-alloc / reuse family behind `convert_buffer`'s deprecation; `into_converted` is the identity-moves path | `bc9f8b8`, `2f51693`, `ede5e8a` |
| convert | `PixelBufferConvertTypedExt::try_to_{rgba8,rgb8,gray8,bgra8}` (`rgb` feature) | fallible siblings of the panicking `to_*` | `2f51693` |
| convert | `PixelBufferLoadBearingExt::into_load_bearing_format(self, bool)` | consuming reduce | `ede5e8a` |
| convert | sealed `PixelSliceOrientationExt::{apply_orientation, apply_orientation_into}`, `PixelBufferOrientationExt::{apply_orientation (consuming; identity moves), apply_orientation_in_place}` | receiver forms; the free functions stay (see §5.3) | `1126d05` |
| convert | Cargo feature `estimation-experimental = []` (no-op in 0.2.17) | lets the `estimate` deprecation be silenced by the same line that keeps it compiling in 0.3.1 | `5e959de` |
| convert | `RowTransform::try_transform_row` / `RowTransformMut::try_transform_row` returning `Result<(), CmsPluginError>`, *provided* methods defaulting to `Ok(self.transform_row(..))` | lets callers migrate before 0.3.1 changes `transform_row`'s return type | new |
| zenpixels | `DiffuseWhite::try_new(f32) -> Option<Self>` (finite, > 0) | foundation review §6; additive only — `new` is not deprecated in 0.2.17 | new |

Not folded in: `Adapted` accessor methods (the type is deprecated), the
`__bench_scan` shim removal, and #63's `--examples` CI gate (re-add it with
the migration probe, §3.6). PR #55 (`Cicp::resolve_matrix`, additive, has a
waiting consumer in zenavif's `cicp_resolve.rs`) and PR #62 (image-crate
interop, additive, separate deps) can join 0.2.17 if rebased; neither is
required for paving.

### 3.2 Deprecations added in 0.2.17

All `since = "0.2.17"`. The note text is the migration instruction.

| Item | Note |
|---|---|
| `RowConverter::convert_rows` | "use `convert_slice_into(PixelSlice, PixelSliceMut)`; removed in 0.3.1" |
| `adapt::convert_buffer` | "use `PixelBuffer` + `PixelBufferConvertExt::{convert_to, convert_into, convert_in_place}`; removed in 0.3.1" |
| free `convert_row` | "use `ConvertPlan::convert_row`; removed in 0.3.1" |
| `PixelBuffer::into_vec` | "returns the backing Vec including any alignment offset; use `into_parts()` or `into_contiguous_bytes()`" — **not** removed in 0.3.1 (two live users; removal in a later break) |
| `planar::Plane` (already `#[doc(hidden)]`) | "use `PlaneLayout` with separate `PixelBuffer`s; removed in 0.3.1" |
| `icc_profiles::REC2020_V4` | "transfer-blind; use `synthesize_icc_for_cicp(Cicp::BT2020_*)`; removed in 0.3.1" |
| `pub mod estimate`, its root re-exports, `ConvertPlan::{estimate, estimate_in}` | `#[cfg_attr(not(feature = "estimation-experimental"), deprecated(note = "unstable; enable the `estimation-experimental` feature, which becomes required in 0.3.1"))]` — a consumer who adds the feature is warning-free now and unaffected later |
| `requires_cms` | "match `ConvertError::NeedsCms` instead; removed in 0.3.1" |
| `pipeline::*` (behind the `pipeline` feature) | "dev-only analysis tool; moves to `__pipeline` in 0.3.1" |
| (optional, decision D3) `ByteOrder` → `pub type ByteOrder = ChannelOrder` deprecated alias; `PixelFormat::byte_order`/`PixelDescriptor::byte_order` → `channel_order` with deprecated forwarders | the enum names R,G,B-vs-B,G,R order, not endianness; zero external uses |

Already deprecated and unchanged: `Adapted` + the three non-cow adapters
(0.2.15), `from_icc_and_cicp` (0.2.6), `ContentLightLevel::measure` (0.2.16),
`HdrMetadata` and `OutputMetadata::hdr` (0.2.14), `reinhard_*`/`exposure_tonemap`
(0.2.15), `ColorManagement` + `finalize_for_output<C>` (0.2.8),
`lut_transform_opts`/`cicp_transform_opts` (0.2.3), `ADOBE_RGB_V4`/`PROPHOTO_V4`
(0.2.4), `icc_profile_for_primaries` (0.2.12).

Not deprecated in 0.2.17 (deliberately): the free `apply_orientation*`
functions (seven repos use them, removal buys nothing), `measure_robust`
(the rename to `measure` cannot coexist with the deprecated inherent
`ContentLightLevel::measure` — 0.3.1 removes the inherent method, adds
`CllMeasure::measure`, and keeps `measure_robust` as a deprecated alias),
`DiffuseWhite::new` (nine repos; validate first, deprecate later).

### 3.3 Tolerated technical break in 0.2.17: seal the extension traits

Seal `TransferFunctionExt`, `ColorPrimariesExt`, `PixelBufferConvertExt`
(and through it `PixelBufferConvertTypedExt`), `PixelBufferHdrConvertExt`,
and `CllMeasure`. Evidence: zero external `impl` of any of them in `~/work`
(sweep 2026-09-26); `cargo copter --path zenpixels-convert` must agree before
publish. Why now: sealing is what lets §3.1 add *required* methods to these
traits, and every later method addition stops being a technical break.
`cargo semver-checks` will report `trait_newly_sealed` and `trait_method_added`
for exactly these traits — those are the only findings the 0.2.17 gate
accepts. CHANGELOG heading: `#### Changed (BREAKING, tolerated in 0.2.x)`.
`Pixel` (zenpixels) stays open; its docs invite custom implementations.

### 3.4 Correctness batch (behavior fixes; test first, then fix)

Each item gets a failing test before the fix and a `fixed` changelog entry.
None changes an API. Items marked *verified* were confirmed in `main` source
today; the rest come from the foundation review and need their reproduction
rebuilt as a test before anyone touches code.

P0 — silent wrong pixels:
- `adapt.rs` transfer-agnostic zero-copy arms (~`:330`, ~`:723`) must require
  `descriptor.transfer == TransferFunction::Unknown`. *verified*
- `output.rs:536` `descriptors_match` must include `alpha`; a premultiplied
  source may not be returned under a straight descriptor. *verified*
- #64: make the HDR→SDR refusal at `convert.rs:560` unconditional. Default
  builds then get `HdrSourceRequiresPeak` instead of a clipped image; the
  peak-taking constructors stay behind `hdr-experimental` for now (§3.5), so
  the error text must say to enable the feature and supply a peak. *verified*
- RGBA→GrayAlpha accepted as identity (drops alpha); `DiscardIfOpaque`
  dropping non-opaque alpha; premultiplied sRGB → straight linear applying
  the transfer before un-premultiplication; `TransferFunctionExt` treating
  `Gamma22` as identity; allocating `apply_orientation` dropping
  `ColorContext`; typed channel-swap helpers resetting descriptor metadata to
  sRGB presets; `Cicp::to_descriptor(Rgbx8)` marking the padding lane as
  alpha; `ColorPrimaries::contains` scalar ordering (claims P3 ⊇ Adobe RGB)
  feeding `negotiate.rs:773`. (review §4/§5)
- `finalize_for_output_with`: `SameAsOrigin` with a Display-P3 origin must
  not emit P3 metadata over sRGB bytes; narrow-range CICP must not be emitted
  over full-range samples; ICC origin/target must reach the `PluggableCms`
  as `ColorProfileSource::Icc` (prerequisite for §5.3). (review §2; the
  descriptor-only construction is *verified*)

P1 — panics and plan semantics:
- `from_imgvec` keeps the old stride after compacting; `transform_in_place`
  owned-extent mismatch; zero-width crop row access; `reinterpret` accepting
  an alignment normal construction rejects; MoxCms accepting a P3 RGB8→RGBF32
  plan then panicking with `LaneSizeMismatch`; AdobeRGB→Oklab plan panicking
  on a missing matrix. (review §4/§5)
- `RowConverter::compose`/`from_plan` representing an external CMS transform
  as an identity placeholder; `convert.rs:2168` cancelling F32→U8→F32 and
  similar inverse pairs (composition must preserve declared quantization).
  (review §4)
- U16→U8 x86 throughput regression — in flight in `zenpixels--u16narrow`.

Scope rule: 0.2.17 ships when §3.1–3.3 are in and the P0 items marked
*verified* are fixed. Remaining items land on the `0.2.x` branch as 0.2.18+
and are forward-ported to `main`, not the other way round.

### 3.5 HDR surface: stays `hdr-experimental` in 0.2.17

The feature's own comment promises the API "locks at 0.3.0". Do not lock it
in 0.3.1 either: the review found unit-semantics problems (`convert_to_sdr`
measuring PQ-normalized values as if relative to the caller's diffuse white;
HLG OOTF policy undefined; `DiffuseWhite::new` accepting 0/negative/NaN) that
need settling before the shape is frozen. Stabilization (un-gating
`HdrConfig`, `new_with_hdr_peak/_config`, `PixelBufferHdrConvertExt`,
`CllMeasure`; keeping `hdr-experimental` as an empty stub so existing
`features = [...]` lines still build) is additive and can happen in any
0.3.x once the units contract exists. The only 0.3.1 change in this area is
the `measure_robust` → `measure` rename with the deprecated alias.

### 3.6 Gates specific to this release

- **Forward-compat job** (added to `ci.yml`, runs on `main` now and on the
  `0.2.x` branch after the split): `cargo check -p zenpixels-convert
  --all-features` with `[patch.crates-io] zenpixels = { git = ..., branch = "main" }`
  applied in a scratch copy of the manifest. It proves the converter builds
  against the zenpixels that 0.3.1 will become. Practically this means
  zenpixels-convert's library must not use `ContentLightLevel::measure`,
  `Plane`, `from_icc_and_cicp`, and its `ColorAuthority` match in
  `output.rs` must carry a `_ =>` arm (with `#[allow(unreachable_patterns)]`)
  before 0.2.17 ships. `zenpixels-convert 0.2.17` declares
  `zenpixels = ">=0.2.17, <0.4.0"`.
- **Migration probe:** `zenpixels/examples/migration_probe.rs` and
  `zenpixels-convert/examples/migration_probe.rs`, `#![deny(deprecated)]`,
  exercising the recommended surface only (`new`/`from_vec`/`into_parts`/
  `try_from_parts`/`new_contiguous`; `convert_to`/`convert_into`/
  `convert_in_place`/`into_converted`; `RowConverter::new` +
  `convert_slice_into`; `ConvertPlan::convert_row`; `adapt_for_encode_cow`;
  `finalize_for_output_with`; the orientation traits; `ColorAuthority` and
  `ConvertError` matches with `_ =>`; `CllMeasure::measure_max` under
  `hdr-experimental`). CI builds `--examples` on both branches, and a step on
  `main` diffs the probe against `origin/0.2.x:<path>`: the file must stay
  byte-identical across the branch point. An edit needed on `main` is a snag.
- **`cargo semver-checks` vs 0.2.16:** only the §3.3 traits may appear.
- **`cargo copter`** in WIP mode for both crates (`--path zenpixels/zenpixels`,
  `--path zenpixels-convert`), report kept under `copter-report/` and cited
  in the release commit.
- The existing gates: `cargo test --all-targets`, `cargo test --doc`,
  `just api-doc` (regenerated snapshots committed with the code), CI green
  on every platform (incl. `windows-11-arm`, macOS Intel, `i686`).

### 3.7 Release mechanics (ordered)

1. Land §3.1 (PR, reviewed), §3.2, §3.3, §3.4 P0 on `main`; regenerate API
   snapshots; correct the changelog (list in §7).
2. Add the `0.2.x` branch to `ci.yml` `on.push.branches` / `pull_request.branches`
   and add the forward-compat + probe jobs — *before* tagging, so the branch
   inherits them.
3. Bump both crates to 0.2.17; move `[Unreleased]` into `[0.2.17] - <date>`;
   zenpixels-convert's zenpixels requirement to `>=0.2.17, <0.4.0`.
4. Gates from §3.6; user reviews `README.md` / `README.crates.md`.
5. Push, wait for full CI, tag `zenpixels-v0.2.17` and
   `zenpixels-convert-v0.2.17`, create both GitHub releases, publish
   `zenpixels` then `zenpixels-convert`.
6. Immediately after: create the `0.2.x` bookmark at the release commit (§4),
   then bump `main` to 0.3.1.

### 3.8 Downstream migrations (after 0.2.17 publishes; each is a small change in its repo)

| Repo | Change |
|---|---|
| zenpipe/zencodecs | five `adapt_for_encode*` → `*_cow`; `gainmap.rs:108` → `CllMeasure::measure_max` + `hdr-experimental` on its zenpixels-convert dep |
| zencodec | `info.rs:374` `from_icc_and_cicp` → `from_icc`/`from_cicp` per authority (the review notes this call is also where the ICC/CICP authority conflict lives) |
| zenmetrics | free `convert_row` ×5 → `plan.convert_row`; `zenmetrics-cli/decode.rs:344` `convert_rows` → `convert_slice_into` |
| zensim (picker-prep bins) | `convert_rows` ×2 → `convert_slice_into` |
| zenavif | `examples/drift_reencode.rs:297` → `convert_slice_into` |
| hdr-anchor-vis | `convert_buffer` → `PixelBuffer::convert_to`/`convert_into` |
| squintly, all-the-images | `into_vec` → `into_contiguous_bytes` (optional; not removed in 0.3.1) |
| zencodec-testkit | `tests/usage.rs:208` → `adapt_for_encode_cow` |

Until they migrate they compile with warnings; nothing breaks at 0.2.17.
They must be done before their zenpixels requirement admits 0.3.1
(zencodec's `^0.2.14` does not; the 19 repos swept on 2026-08-29 already
admit `<0.4.0`).

---

## 4. Branching and backports

- `main` carries the 0.2 line until 0.2.17 is tagged. The `0.2.x` branch is
  created **at the 0.2.17 release commit**, never earlier:
  `jj bookmark create 0.2.x -r zenpixels-v0.2.17 && jj git push --bookmark 0.2.x --allow-new`,
  then verify with `gh api repos/imazen/zenpixels/branches/0.2.x`.
- `main` then becomes 0.3.1-in-progress: bump both `Cargo.toml` versions to
  0.3.1, `[Unreleased]` header notes the target, and the removal batch (§5.1)
  lands as its own commit series.
- **Backport work happens in a sibling jj workspace, never by moving the
  primary checkout off `main`** (standing rule):
  `jj workspace add --revision 0.2.x --name zero-two-x ../zenpixels--0.2.x`.
  Claim it with `.workongoing` like any checkout. Fix on `main` first when the
  code still matches, then `jj duplicate <change> -d 0.2.x`; when it no
  longer matches (post-removal), re-implement on the branch. Advance and push
  with `jj bookmark set 0.2.x -r <rev> && jj git push --bookmark 0.2.x`.
  Forget the workspace when a backport round is done.
- What is eligible for backport: behavior fixes, doc/deprecation-note
  corrections, additive replacements a migration needs. Never removals,
  never `#[non_exhaustive]` additions, never default-feature flips.
- 0.2.18+ releases come from the branch with the same tag/GitHub-release
  sequence; the `>=0.2.17, <0.4.0` requirement and the forward-compat job
  stay on the branch for its lifetime.
- The existing git worktree `../zenpixels--image-interop` (PR #62) and the
  prunable `/tmp/zenpixels-pr63.*` worktree entry are left as found.

---

## 5. 0.3.1 — the breaking release

### 5.1 Removals (each has a shipped replacement and a warning in 0.2.17)

zenpixels:
- `ContentLightLevel::measure` (+ its five in-crate tests and
  `zenpixels-convert/tests/deprecated_measure_parity.rs`) → `CllMeasure::measure_max`
- `ColorContext::from_icc_and_cicp` → `from_icc` / `from_cicp`
- `planar::Plane` → `PlaneLayout` + `PixelBuffer`s
- `serde` stub feature (zero users)
- (if D3) `ByteOrder` alias and `byte_order()` forwarders

zenpixels-convert:
- `Adapted`, `adapt_for_encode`, `adapt_for_encode_with_intent`, `adapt_for_encode_explicit` → `*_cow`
- `adapt::convert_buffer`; `RowConverter::convert_rows`; free `convert_row`
- `hdr::HdrMetadata` + root re-export; `OutputMetadata::hdr`
- `hdr::{reinhard_tonemap, reinhard_inverse, exposure_tonemap}` → zentone
- `cms_moxcms::{lut_transform_opts, cicp_transform_opts}` → `transform_opts`
- `icc_profiles::{ADOBE_RGB_V4, PROPHOTO_V4, REC2020_V4, icc_profile_for_primaries}`
- `requires_cms` → `ConvertError::NeedsCms`
- `serde` stub feature
- `ColorManagement` trait and `finalize_for_output<C: ColorManagement>` —
  **conditional**, see §5.3

Renames with deprecated aliases kept for one release: `CllMeasure::measure_robust`
→ `measure` (alias stays); `RowTransform{,Mut}::try_transform_row` becomes a
deprecated alias once `transform_row` returns `Result` (§5.2).

### 5.2 Tightenings (un-warnable; each needs its evidence at release time)

| Change | Evidence required | 0.2.17 preparation |
|---|---|---|
| `#[non_exhaustive]` on `ColorAuthority` | `cargo copter` zero exhaustive-match victims; zenanalyze re-checked (the 2026-06 victim was not found in today's sweep) | changelog + doc note; our own `output.rs` match gets `_ =>` |
| `#[non_exhaustive]` on `ColorPriority` | zero uses found | doc note |
| `RowTransform{,Mut}::transform_row` returns `Result<(), CmsPluginError>` | zero external impls; callers only get `unused_must_use` warnings | `try_transform_row` provided methods |
| `fast-transpose` on by default | output bit-identical (`benchmarks/orient_fast_transpose_2026-06-18.md`); size-sensitive builds use `default-features = false` | doc note |
| `estimate` behind `estimation-experimental`; `requires_cms` removed | zero users | cfg_attr deprecation + feature exists |
| `pipeline` module dev-only; feature name kept as an empty stub | zero users | deprecation |
| zenpixels-convert requires `zenpixels >=0.3.1, <0.5.0` (current-plus-next) | — | — |

Not tightened: `FormatOption`, `ConversionCost`, `PlaneDescriptor` and the
other exhaustive structs stay constructible by literal (zenjpeg builds
`PlaneDescriptor` literals in hundreds of places); the `Pixel` trait stays
open; `pub use zenpixels::*` stays.

### 5.3 Kept in 0.3.1 despite earlier notes

- Free `apply_orientation*` functions — seven repos, no benefit from removal.
- `PixelBuffer::into_vec` — deprecated, two live users, removal later.
- `ConvertError` variants — `AllocationFailed` is matched in five downstream
  sites; consolidating into `Buffer(BufferError::AllocationFailed)` would
  silently change what they match. Dropped from the plan.
- `finalize_for_output<C: ColorManagement>` + `ColorManagement`: remove
  **only if** 0.2.17's §3.4 fix makes `finalize_for_output_with` pass ICC
  origin/target to the `PluggableCms`, proven by a test that runs a real
  saturated ICC transform through both finalizers byte-identically.
  Otherwise the removal moves to the next break.
- `hdr-experimental` as a feature (§3.5).

### 5.4 Contract track (from the foundation review; not a 0.3.1 gate)

Each of these is a real defect class with reproduced failures, and each
needs an additive replacement before its break. They proceed on `main`
during 0.3.x; their breaking steps batch into the next release:

1. One authoritative current-color description, separate from provenance
   (`ColorContext` ICC+CICP coexistence; `as_profile_source` vs transfer
   shortcuts). Additive first: a single `resolve` operation with explicit
   `Unknown`.
2. An output plan owning current encoding, provenance, target, alpha and
   permitted loss, with consuming/borrowing/caller-output forms (#69);
   `EncodeReady` certifying agreement instead of bundling.
3. Validated descriptor/typed-view/planar construction (checked
   constructors additive first; private fields later, since zencodec,
   zenavif, zenjpeg, zenanalyze, zenextras build `PixelDescriptor` literals).
4. Complete conversion plans (CMS steps survive `plan()`/`from_plan()`;
   composition preserves declared quantization).
5. Exactness vocabulary distinct from loss ranking (`LossBucket::Lossless`,
   `forbid_lossy`, `Provenance` "originally U8" as proof).
6. Validated HDR units and lifetimes (positive-finite anchor; reference vs
   peak vs measurement; `quantize_to` preconditions).

### 5.5 Gates and cutover

- `cargo semver-checks --baseline-version 0.2.17` on both crates must list
  exactly §5.1 + §5.2, nothing else.
- Migration probe from 0.2.17 compiles unchanged.
- `cargo copter` on both crates against the reverse-dependency set
  (baseline = 0.2.17): every "your fault" cell must be a documented §5.2 item
  with a downstream fix already prepared.
- Publish `zenpixels 0.3.1`, then `zenpixels-convert 0.3.1`, then bump
  consumers in dependency order (zencodec first — it pins `^0.2.14` and
  gates the codecs — then codecs, zenpipe, zenmetrics, imageflow). Uniform
  bumping matters: a graph mixing 0.2 and 0.3 resolves two `zenpixels`
  crates whose types do not unify.

---

## 6. Decisions needed

- **D1 — 0.3.1 scope.** Recommended: mechanical (§5.1 + §5.2) and soon
  after 0.2.17; contract work (§5.4) stays additive until the next break.
  Alternative: hold 0.3.1 for one or more §5.4 items.
- **D2 — seal the extension traits in 0.2.17** under the tolerated-break
  policy (recommended), or defer sealing to 0.3.1 and add §3.1's trait
  methods as a tolerated break anyway.
- **D3 — `ByteOrder` → `ChannelOrder`** rename with deprecated alias
  (cheap, zero users; recommended if `descriptor.rs` is touched anyway).
- **D4 — `into_vec`:** deprecate in 0.2.17, keep in 0.3.1 (recommended).
- **D5 — free `apply_orientation*`:** keep undeprecated (recommended).
- **D6 — `ColorManagement` removal** conditional on ICC parity (recommended).
- **D7 — PR #55 / PR #62** into the 0.2.17 additive batch or not.
- **D8 — land `docs/foundation-0.3-review.md` + its coverage TSV.** The
  file is 38 KB (over the 30 KB no-ask limit), so it was left uncommitted;
  the two small benchmark files it links are committed.

---

## 7. Documentation corrections batched for approval

`CHANGELOG.md`: delete the moot queue items (registry demotion,
`ZenCmsLite::extended`); delete the duplicate 0.2.7-era queue section and
its `repr(u8)` item; fix the "made private before publication" sentence;
add the missing queue items (`estimate` gate, `requires_cms`, `REC2020_V4`,
`pipeline`, `ColorPriority`, dependency-range bump, `measure_robust`
alias); mark every #63-sourced replacement as "ships in 0.2.17"; change
0.3.0 → 0.3.1 throughout. Source: `output.rs:105` and `:234` TODOs (premise
stale), `registry.rs:24` and `planar.rs:486` comments, `Cargo.toml`
`hdr-experimental` comment ("locks at 0.3.0"). `CLAUDE.md`: "avoid 0.3.0
indefinitely" is no longer the policy; point at this plan. Issues: #64
(premise), #69/#71 comments (claims about landed replacements).

---

## Appendix — evidence

- crates.io: `curl -s -H 'User-Agent: zenpixels-release-audit (github.com/imazen/zenpixels)' https://crates.io/api/v1/crates/<crate>/versions`
  → `0.3.0 YANKED 2026-04-13` for both; `0.2.16 2026-07-24` latest.
- semver: `cargo semver-checks check-release -p <crate> --baseline-version 0.2.16`
  (2026-09-26, `~/tmp/zenpixels-semver-main-vs-0.2.16.log`): 196/196 pass, both crates.
- PR #63 orphan: `git merge-base --is-ancestor 44f7716 origin/main` → no;
  `git cherry origin/main origin/hdr-fixes-2026-07-14` → 22 `+`; per-commit
  `git merge-tree --write-tree --merge-base=<c>^ origin/main <c>` → 20/22
  conflict; API delta from `git diff origin/main origin/hdr-fixes-2026-07-14 -- docs/public-api/`.
- Consumer sweep: `rg` over `~/work` with `--glob '!**/target/**' --glob '!**/.jj/**'`
  and the exclusions in §1.5; patterns per row of that table.
- Source checks: `zenpixels-convert/src/output.rs:536-541`,
  `zenpixels-convert/src/adapt.rs:322-350`, `zenpixels-convert/src/convert.rs:556-565`,
  `zenpixels-convert/src/error.rs:8-19`, `zenpixels/src/lib.rs:79`,
  `zenpixels-convert/src/lib.rs:439,560`, `zenpixels/src/buffer.rs:1522-1532`.
