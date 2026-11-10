## Commit completed work

The owner requests commits by default: commit completed changes before ending a
work chunk, rather than leaving them uncommitted. Preserve unrelated user work;
do not discard it or silently include it in a different task. Commit locally;
pushing or publishing still follows the user's instructions for that action.

## Public API surface — YAGNI is the rule, not a suggestion

**Every public item in zenpixels is forever. Treat each `pub` as a load-bearing external commitment.**

zenpixels is the foundational crate of the zen ecosystem — zencodec, zenpipe, imageflow, and every codec depend on it directly or transitively. Once a type, function, method, field, enum variant, trait method, or feature flag is `pub`, removing or renaming it requires a 0.3.0 bump (see below), and that break ripples through every downstream. The owner approved the paired 0.3.1 breaking release; see `docs/release-plan-0.2.17-and-0.3.1.md`. The older indefinite 0.3 avoidance policy no longer applies. The practical consequence: **public items added speculatively cannot be taken back.**

### Rules for new public API

1. **No `pub` without a concrete current consumer in this repo or a sibling zen crate.** "Someone might want this" is not a reason. If the hypothetical caller shows up later, they can file a request and we can add it *then* — adding is cheap, removing is impossible.
2. **Default to `pub(crate)`.** Promote to `pub` only after a concrete caller is merging the code that uses it.
3. **Builder-style APIs over struct literals** wherever possible. `StructName::new()` + `.with_foo()` lets us add fields without the struct becoming a perma-frozen shape. Avoid public structs with public fields unless they are genuinely trivial data bags (e.g., `Rect`, `Size`).
4. **Enums should be `#[non_exhaustive]` from the first release** unless we are *absolutely* sure the variant set is fixed (e.g., `Matrix::Bt601 | Bt709 | Bt2020` — the color-space matrix set is standardized; never will be a new one there). When in doubt: `#[non_exhaustive]`.
5. **Traits should be sealed** (`pub trait Foo: private::Sealed`) unless external implementation is a required feature. Every `impl Foo for MyType` by an external crate turns a supertrait bound addition into a breaking change.
6. **Feature flags are part of the API.** Once shipped, removing a feature flag is a break (see tolerated-break #4 below — feature removal is *tolerated* only when nobody sets it). Don't add feature flags speculatively.
7. **Re-exports are commitments too.** `pub use` of an internal or third-party type pins us to that name and that dependency version. Re-export only items downstream code will reach for frequently.
8. **"Helper" / "convenience" / "shortcut" methods get the strictest scrutiny.** These are the items most commonly added speculatively and most commonly unused. If a user can construct the behavior in 2-3 lines with existing API, do not add a helper. Examples of things to NOT add: `foo_or_default()`, `foo_with_options_fallback()`, `is_probably_bar()`.

### Keeping 0.3 extensible without a breaking release

- **Descriptor-family enums have no integer values.** `ChannelType`,
  `ChannelLayout`, `AlphaMode`, `TransferFunction`, `ColorPrimaries`,
  `SignalRange` and `ColorModel` carry a hidden uninhabited `__Reserved`
  variant, and `PixelFormat` has the data-carrying `Generic`. None has a
  `repr` or explicit discriminants, so `as` casts do not compile and
  data-carrying variants stay additive. Dropping `repr` alone is not enough:
  a fieldless enum still casts. Never add a `repr`, explicit discriminants or
  a cast-based table index to them; numeric codes go through explicit
  functions such as `to_cicp`. `Orientation` alone keeps `repr(u8)`: its
  values are the EXIF codes, a standard that will not gain payloads.
- **Named `PixelFormat` variants are the canonical processing set.** Native
  or rarely used sample-type × layout combinations go through
  `PixelFormat::Generic(GenericFormat)`, never a new named variant per
  combination. `from_parts` must return the named variant whenever one
  exists, so every format has one representation. Record new generic
  dimensions (channel count, per-channel roles) as private `GenericFormat`
  fields.
- **Open traits gain only defaulted methods.** `Pixel`, `PluggableCms`,
  `RowTransform`, `RowTransformMut` and `ColorManagement` have external
  implementations; a required method is a breaking change.
- **Third-party types in signatures are version commitments.** `whereat::At`
  appears throughout the public API, so whereat stays on 0.1.x for the life
  of 0.3. moxcms types never appear in the public API; keep the moxcms
  adapter's options and errors crate-owned. A dependency that is not in the
  public API (archmage, magetypes) gets a plain caret requirement such as
  `0.9.30` (>=0.9.30, <0.10.0) and no extra cap inside 0.9, since one capped
  range refuses every crate that needs a later 0.9 release.
- **The converter's core floor moves with its use of new core API.** A
  zenpixels-convert 0.3.x release that calls a core item added after 0.3.1
  raises its `zenpixels` lower bound in the same release.

### Before adding any `pub` item, you must be able to answer

- Who calls this *today* (file:line references)?
- If removed tomorrow, what concrete code breaks?
- Would a caller who wanted this synthesize it from existing API? If yes, reject the addition.

If you cannot answer with a concrete, verified current consumer: the item stays `pub(crate)` or is not added at all.

### Before every release

Audit `src/lib.rs` for items added since the last version. For each new `pub`, verify a consumer exists. If a consumer cannot be found (including `builtin_profiles`-style "positioned for future use" modules), either:
- Find a real caller and migrate them (simultaneously), OR
- Demote to `pub(crate)` before publishing.

The unused `builtin_profiles` implementation was removed during the 0.3
consolidation. See `docs/design-reference.md` for the current sample, operation,
metadata and fallible-worker contracts; keep that document as the shared reference.

---

## 0.2.x versioning policy — "tolerated technical breaks"

**Do NOT version-gate on `cargo semver-checks` output alone for this repo in 0.2.x.**

We deliberately ship certain narrow semver-breaking changes as `0.2.N` bumps rather than forcing a `0.3.0` release. A 0.3.0 break is expensive — it ripples through every zen* sibling (zencodec, zenpipe, imageflow) and forces downstream dependency-graph churn. When the breakage is technical but unlikely to bite real callers, we accept it.

### Tolerated breaks

Changes that may ship inside a `0.2.N` patch release:

1. **Adding `#[non_exhaustive]` to structs / enums** that nobody constructs via struct-literal in practice (verified by grepping `~/work/zen/`). Breaks theoretical external callers; no known victims.
2. **Adding fields to a non-`#[non_exhaustive]` struct**, provided all in-tree callers are migrated to builder / `::new` / preset constructors simultaneously. External struct-literal construction breaks.
3. **Adding auto-trait supertraits** (e.g., `Send + Sync` to a trait that was `Send`-only) when every in-tree impl already satisfies them. `cargo semver-checks` flags this as a major break; we accept it.
4. **Removing a Cargo feature** when the feature was either experimental (no release had it in default) or has been folded into the default behavior (e.g., `zencms-lite` was dropped when its functionality became unconditional via `OnceBox`).
5. **Losing auto-trait impls as a mechanical consequence** of new private fields (e.g., `RowConverter` lost `Sync` because it now holds `Box<dyn RowTransformMut: Send>`). Acceptable when the type's primary use is `&mut self` and cross-thread shared access was never meaningful.
6. **Dropping a derive like `Debug`** when the type's fields are not meaningfully Debug-able and no callers rely on the default derive. Replace with a curated manual impl or a `_opaque: "<redacted>"` placeholder if needed.

### Not tolerated in 0.2.N

These REQUIRE a 0.3.0 bump:
- Removing public items (types, functions, methods, fields, variants)
- Changing method signatures (argument types, return types, generic bounds that affect callers)
- Renaming public items
- Changing semantic behavior silently (e.g., default flag values)
- Any change that requires known callers to modify source to keep compiling

### Workflow for tolerated breaks

1. Before shipping, grep `~/work/zen/` for external struct-literal / trait-impl / feature-name uses. Document the audit in the commit message.
2. Update all in-tree callers in the same commit (or sibling repo PR) so sibling crates keep building.
3. CHANGELOG entry must be under **`#### Changed (BREAKING, tolerated in 0.2.x)`** heading so readers can see the technical semver status.
4. `cargo semver-checks` failures listed under tolerated-break categories above are acceptable; failures outside them block the release.
5. Ship. If real external breakage is reported, the next release bumps to 0.3.0 and we move the change out of the tolerated bucket.

### Verifying before a release

When preparing to publish:
- Run `cargo semver-checks --workspace` and inspect every failure.
- For each failure, categorize: **tolerated** (matches the list above, document in commit) or **not tolerated** (requires 0.3.0 bump, stop the release).
- If *any* failure is not tolerated, bump to 0.3.0 and cut a proper breaking release.

See also: `CHANGELOG.md` — the 0.3.0 queue section accumulates non-tolerated breaks for batched release.

## Known Bugs

- Fixed (2026-11-09): retained sRGB hash `6a669bf1e1bf97d6` granted substitution
  despite a measured 298-u16 canonical transform error exceeding the existing
  256-u16 qualification gate. Recognition remains; the intent mask is now
  identification-only. Exact original and run provenance: `docs/icc-retention-audit.md`.

- Fixed in the refinement pass (2026-11-09): CMS dispatch omitted BGR from
  its RGB-to-gray luma-policy check and could accept a plugin without explicit
  coefficients. Planner, row CMS and output planning now share one preflight.

- Fixed in the refinement pass (2026-11-09): in-place adaptation compacted
  pixels before rejecting an invalid target descriptor, then panicked while
  rebuilding the view. Target validation now precedes mutation; invalid-target
  regression checks preserve the original descriptor and bytes.

- Fixed (2026-11-09): direct MoxCms construction admitted HLG transfer crossings
  without the coupled display renderer. It now declines identified HLG crossings;
  the public backend regression preserves explicit rendering ownership.

- Fixed (2026-11-09): shared trace-producing `no_path`, cast and extent
  closures collapsed distinct failure sites to one location. Capture inline
  at each operation instead. Inline `at!` in a lambda is appropriate.

- Fixed in the 0.3 design consolidation: composed matte plans could retain stale
  execution descriptors, causing BGRA/P3 RGBA to gray conversions to panic.
  Each operation now owns its descriptors and scale (`0e309692`).
- Fixed: allocating and caller-buffer PQ-to-linear conversion attached different
  white anchors to identical samples; shared resolution also covers adaptation
  and output finalization (`efe55343`).
- Fixed: invalid alpha descriptors, cropped final-row split overflow, empty
  swaps and U64 lane removal assumptions (`9297c114`); F16 alpha payload and
  histogram shape/rank discrepancies (`e29bca51`). See design regression gates.
- Qualification limit: historical ICC table flags have not been regenerated
  against the corrected generator. Their curve-fit errors are not global CMS
  error bounds. New generation cannot override failed empirical evidence with
  structural classification (`8309b21b`).

- Fixed in the explicit HLG rendering work (2026-11-09): HLG matte compositing
  previously used implicit rendering assumptions. It now requires explicit
  rendering conditions and white, and uses the luminance-coupled OOTF. Existing matte error
  tolerances are retained.

- Corrected ICC exports landed on main in 49f00314 (2026-11-09),
  after separate PR128 qualification; no crate has been published. The actual replacement bundle is 93,377 bytes with 313
  RGB/gray mappings; the 21 removed mappings are targetless HLG exports.
  Historical hashes remain recognized, with new exports MetadataOnly.
  All 168 PQ intent/direction cells, 292 SDR cells and 313 independent
  libpng/C LCMS 2.20 containers pass. Native U8/U16 RGB/RGBA/GRAY/GRAY_ALPHA
  LCMS tests preserve alpha; U8 is exact and U16 differs from correctly rounded
  analytic codes by at most 2 on decode / 1 on encode. MoxCMS explicitly
  declines unsupported float pipelines. The six-platform CMS matrix passes.
  HLG retains native CICP until an explicit rendering target exists.
  Browser witnesses establish selected paths, not physical HDR display accuracy;
  WebKit GRAY remains unqualified. See `docs/corrected-icc-export.md` and
  `benchmarks/icc-export/native_integer_2026-11-08.meta`. The independently reviewed
  grayscale-collapse authority guard landed in the same main integration. Its compatible
  variant preserves the existing truncated-input fallback and all established
  assertions; only complete ICCs require qualified matrix/TRC replacement.
  The stricter malformed-input proposal remains preserved, not adopted.
  The self dev-dependency re-enables defaults in unit tests even with
  --no-default-features. Its established configuration is retained: an isolated
  attempt to disable it passes 60 focused controls but exposes 10 unit tests
  expecting bundled profiles with the database absent. Original failures and
  that alternative are preserved; no old expectation is changed.
  tools/icc-collapse-no-db verifies genuine no-database/std/CMS normal-dependency
  behavior through four passing controls; just icc-collapse-check.

- PQ PNG compatibility qualification (2026-11-08): keep native CICP plus a
  qualified ICC companion where permitted. The float-table pilot alone is not
  sufficient: Chromium 153 and explicitly color-managed Firefox 155 use its
  conventional fallback in a contradictory-TRC witness, while C LCMS 2.20
  uses DToB1. ICC-embedded CICP must be isolated in such controls. Headless
  defaults and SDR canvas output do not establish real HDR display accuracy.
  After preserving the initial library failures, isolated Linux WPE WebKit
  26.6 honors float DToB but does not give PNG CICP precedence in these controls.
  Exact source identities, differences and configuration limits are in
  `docs/corrected-icc-export.md`; no Safari coverage is claimed.
- Fixed in the adapter, landed in 49f00314 (2026-11-09): MoxCMS does not implement float
  DToB/BToD pipelines. ICC permits fallback, but it can lose requested precision.
  The adapter now declines these profiles so a capable configured backend can
  handle them. A regression failed before the guard; shared CMS14/2oldignores
  and unchanged Mox17+21 accuracy / LCMS8 production controls pass. The external
  W3C PQ file is an explicitly approximate legacy fallback, not proof that all
  HDR profile representations work equivalently. Its Mox highlight limit also
  reproduces without the row adapter; bounded mAB rules versus LCMS extended
  float evaluation must be distinguished from a specification violation.

- Historical bundle audit (2026-11-08): all 175 old GRAY
  mappings are Display-class but store native white in `wtpt`, contrary to
  ICC.1:2022 9.2.36's PCS-D50 requirement. This dates to `75eb953`; the
  October `27c17a6` gray test explicitly preserves the wrong expectation.
  The first 0.9.2 proposal and constructor controls copied these bytes unchanged.
  PR128 instead corrects the Display-class wtpt and preserves old recognition.
  Independent reader acceptance does not establish the tag's correctness.
  `just icc-bundle-definition-audit REVISION` inspects actual historical
  assets; see `docs/icc-bundle-history-and-correctness.md`. No binary,
  existing assertion or pixel tolerance is changed by this finding.

- Historical profile triage (2026-11-08): the inherited curves in all three RGB
  candidates have limits missed by the sparse read-back samples. On every
  normalized U16 input, TC9/TC10 have maximum absolute linear errors
  0.004917139850733618 / 0.0015508450894103343 near encoded zero; TC17 reaches
  0.0005306961467401727 near its saturation knee. PQ has 3,587 nonzero-reference
  input codes rounded to zero per profile. These are measurements, not new
  accepted tolerances. HLG is a neutral reference-display curve, not a general
  luminance-dependent RGB OOTF. See `docs/first-class-lcms-and-profile-triage.md`.

- Fixed in the standalone Rust-LCMS contract adapter (2026-11-08): it selected
  `lcms::virt::INTENT_RELATIVE_COLORIMETRIC`, whose value is 3 in the pinned
  port, requesting absolute intent. Use `lcms::types::INTENT_RELATIVE_COLORIMETRIC`
  (1). A measured-media-white regression fails before the correction and
  passes after it; all four JXL CMYK fixture samples then match C LCMS 2.20
  float bits. This changes test equipment, not the product CMS default.
  `docs/rgb-icc-and-lcms-2026-11-08.md` records the source pins and limits.

- Historical stable-dependency failure (2026-11-07): requiring moxcms 0.9.2
  changed fresh RGB ICC serialization/colorimetry and failed the old bundle
  equality gate. The owner-approved corrected export definition now passes
  fresh-synthesis equality on main; the original failure receipts remain. Keep the
  pure-gamma sign/range guards: the released backend still reproduces the
  exact-zero ProPhoto F32 failure. See `docs/stable-dependencies-2026-11-07.md`.

- Fixed explorer lock drift (2026-11-07): the converter adopted pastey while
  the browser standalone lock retained its paste dependency edge. Locked WASM
  Clippy refused before compilation. Refresh the standalone graph and run
  `just explorer-boundary-check`; strict WASM Clippy and fmt pass without
  changing source algorithms or assertions.

- Fixed IQA conformance portability (2026-11-05): the new native trace test
  assumed forward-slash source paths and failed on Windows x64/ARM64. It now
  compares the recorded source path with `file!()` while retaining the exact
  origin-line, context, cause-identity and propagation assertions. IQA scalar
  arithmetic and native metric expectations are unchanged.

- Fixed at F32 (2026-11-05): moxcms's `ToneCurveEvaluatorPureGamma` uses `dirty_powf`, which returns unrelated values where the power isn't representable (gamma 1.8 of zero is -1.65e8) and takes a negative base's sign from the exponent's integer part. A zero channel of a ProPhoto F32 sample clipped ordinary colors to full scale: 13,254 of 421,824 channels were over 2 codes from the 8-bit route, worst 255; now 0, worst 1. `PureGammaCurve` keeps the backend's evaluator for representable magnitudes and takes `libm` for the rest, on sources and gray routes. Evidence and limits: `benchmarks/pure_gamma_f32_2026-11-05.md`.

- Open (2026-11-05), can't be fixed in the adapter: table source to pure-gamma RGB target. The backend interpolates a grid filled through the target's curve, so a grid node can be corrected but a value between a negative node and a positive one mixes two forms: worst 18 codes on a synthetic table, from 144 before. Upstream's own fix would remove this.

- Fixed (2026-11-05): `Cmyk8` and `CmykF32` gave different colors from one ink: on 49 real profiles to sRGB, a mean of 17.8 codes and at worst 114, because `Cmyk8` had no black-point compensation and its own intent. `Cmyk8` now runs the F32 transform and rounds, with no U16 in the way, so they agree exactly (`eight_bit_cmyk_is_f32_cmyk_rounded`). Also fixed: RGBA targets from CMYK skipped the compensation that RGB got (`cmyk_targets_agree_with_and_without_alpha`). `benchmarks/cmyk8_bpc_2026-11-05.md`: `Cmyk8` is 44% slower, `CmykF32` 6 times faster.

- Open (2026-11-05): `linear_light_suitability` charges 120 (LowLoss) for any gamma-encoded format whatever the resize kernel. On the sharp-edge image Mitchell and Robidoux agree within a bucket, but CatmullRom and Lanczos through U8 measure p95 ΔE 21 to 33 (High), and clipping alone accounts for it: 6 of 12 scenarios are two buckets off. The model has no kernel input. `gamut_clamping_scenarios` pins the gap.

- Open, in moxcms 0.9.1, measured 2026-11-05 against exact evaluation: a pre-v4 `lut8Type` Lab table reads 0.39%
  high (mozjpeg `test3.icc`, mean +0.68 of a code, worst 22; fixed on moxcms master, unreleased). A v4 RGB table
  profile's black is mapped to zero whatever the intent (mean about -1 of a code on JPEG XL's PQ and HLG profiles;
  still so on master). `benchmarks/jxl-cms-parity/README.md` has the measurements. Neither can be routed around from
  `cms_moxcms.rs`: moxcms reads the profile version privately, and a patched header changes both.

- Fixed in the CMS investigation (2026-11-05): `MoxCms` lost the sign of out-of-gamut samples on pure-gamma matrix
  targets, through moxcms's `ToneCurveEvaluatorPureGamma`: P3 into Adobe RGB was a mean of +2.35 of an 8-bit code
  high, worst 66. A CICP tag inside an explicit ICC profile changed the 8-bit route's curves, while F32 ignored it.
  `cms_moxcms_accuracy` and `cms_moxcms` cover both.

- Open, in moxcms 0.9.1: 16-bit output through its sRGB or BT.709 table misses nearest on 57% of samples, mean -0.64 of a
  16-bit code, because the same `make_gamma_table_impl` truncates. `MoxCms` still takes that path.

- Fixed in the CMS contract (2026-11-05). moxcms' own 8-bit path comes out low (mean -0.40 of a code against its own F32; cause not
  located). The wrapper chose high barycentric weights at 16 bits, which turned table profiles into a constant, ignored a row's
  `width`, and mixed moxcms' generated colorants (CIE D50) with ICC profiles (PCS white), leaving 8-bit white at (255, 255, 254).
  `tests/cms_moxcms_accuracy.rs` covers each. moxcms differing from lcms2 on table profiles is expected, not a bug.

- Fixed 2026-11-04: planning added the gamut matrix after layout and depth steps, so wide
  RGB into gray or from RGBA to RGB ran it on rows of the wrong shape, and U16 rows ran U8
  steps. `src/kernel_tests/gamut_across_layouts.rs` measures every route against independent `f64`
  math. A row split could also change `f32` bits, and an odd scratch size could misalign
  F32 rows: `tests/row_partition_parity.rs`. BT.709 and Gamma22 `f32` output keep negative
  values while sRGB `f32` clips: that difference is existing behavior, pinned by the test.

- Contract (2026-11-04): adjacent inputs can reverse by at most 1 `u16` code on PQ, sRGB, BT.709 encode and gamma encode, from f32
  rounding in polynomial evaluation. `adjacent_inputs_reverse_by_at_most_one_code` holds it. Counts: `benchmarks/row_kernel_tiers_2026-11-04.md`.

- Fixed (2026-11-04): a PQ or HLG matte put white at the format's peak: 10,000 cd/m² on PQ.
  `CompositeOnto::reference_white` now places it, defaulting to BT.2408.

- HDR intent correction (2026-11-06, PR #120 candidate): unanchored relative/absolute
  conversions refuse with `SdrWhiteRequired`; `ConvertOptions::with_sdr_white` overrides
  `ColorContext::diffuse_white`, then absence refuses. Row, buffer and quantizer paths
  agree at 203 nits (PQ16 white code 38055). Invalid/conflicting CICP remains rejected
  consistently; an anchor-only context avoids falsely tagging linear samples as PQ.
  Buffer HDR-to-SDR conversion measures MaxRGB peak by default; row/output plans require
  an explicit or context-declared peak. SDR target peak is explicit. HLG transfer
  conversion requires explicit `HlgRendering` conditions; omission returns
  `HlgRenderingRequired`. Old HDR-config
  constructors remain available for compatibility; the preferred interface is
  `convert_to_with_options` / `new_explicit`. Design: `docs/hdr-anchor-api-design.md` §6.
  Kernel tests moved to `src/kernel_tests`: the same numerical assertions and frozen
  digest `509de40f680b9c2e` still pass, including raw HLG transfer kernels exercised
  privately; the public API separately refuses unimplemented light-domain work.

- Fixed (2026-11-04): the matte was not converted to wide-gamut source primaries, so sRGB red over a P3 source came out as P3's red.
  `CompositeOnto::background` is now a `Color`, converted to the source's primaries; `zenpixels-convert/tests/color.rs`.

- Fixed (2026-11-04, #106): tiers gave different output. Scalar tiers now run the kernels' operations with fused math, so every
  conversion gives the same bytes at every tier, and a digest test pins them across platforms. Strict WASM needs `precise-portable`.

- Fixed (2026-11-04): a pixel's output depended on its position in a row, up to 4 `u16` codes
  on PQ, because slice tails ran different math from the body. `tests/row_partition.rs`.

- Fixed (2026-11-04): the vendored PQ kernels reached 4 `u16` codes off ST 2084 at signal 0.9988.
  A test appeared to bound this, but single-sample rows only exercised the scalar tail. PQ now
  uses linear-srgb's slices, whose low range is fixed.

- Fixed in #103 before release (2026-11-04): `RowConverter` panicked on `Bgr8` to `Bgr8`
  with a different transfer, because the plan went through a BGR F32 format that does not
  exist. BGR now refuses anything except an identical descriptor. `bgra_is_8bit` runs every
  format pair across layouts, alpha modes and transfers.

- Fixed in the `PixelStorage` work (2026-11-04): `PixelBuffer::from_vec` assumed a
  `Vec<u8>` is aligned for wide samples. On an allocator returning an odd address, an
  exact-length buffer failed with `InsufficientData`, and a longer one was accepted with
  every sample read one byte late: silently wrong pixels. glibc hid both.
  `tests/odd_allocator.rs` runs under such an allocator; `from_vec` now copies into typed
  storage in that case.
- Fixed in the same work: `ColorAuthority`'s readers could disagree on a new variant
  (ICC, CICP or the current pixel profile, depending on the reader, with no compile
  error). It is no longer `#[non_exhaustive]`, and `ColorAuthority::select` is the one
  decision. zencodec and zencodecs readers still need to adopt it.
- Fixed in the same work: `cargo test -p zenpixels --no-default-features` didn't build, because
  `tests/storage_contracts.rs` called the `icc`-only `resolve`.

- Fixed on the output-plan review (2026-10-27): Gamma22 F32 conversion accepted
  `clip_out_of_gamut(false)` but clipped negative and above-white samples. The
  extended steps now reflect the existing Adobe exponent, retain the existing
  SIMD results for magnitudes in range, and use libm outside that domain. The
  regression checks default/in-range parity, alpha payloads, signed zero,
  strided padding, row partitions and unknown retired estimates. This does not
  qualify every transfer/backend or change the selected numerical model.

- Fixed on the output-plan review (2026-10-27): a native TIFF source with an
  authoritative current ICC and unknown descriptor primaries/transfer took an
  unnecessary same-profile CMS round trip, changing U8 sample 74 to 73. Identity
  planning now completes only unknown color fields from that exact ICC and
  preserves samples. `output_descriptor` tests borrowed and strided copied
  RGB/RGBA U8/U16/F32, including float bit patterns; different profile bytes and
  known descriptor fields do not receive this identity shortcut.

- Fixed in the float-ICC integration (2026-10-26): workspace feature
  unification enabled an extra moxcms option field, breaking the generator's
  exhaustive initializer. It now supplies backend defaults for added fields;
  extended evaluation stays disabled in the generator's strict comparisons.
  `just output-float-workspace-check` also checks standalone generator lint.

- Hardened on the output-plan review (2026-10-26): native JXL F32-to-ICC
  conversion used bounded integer lookup/fixed-point paths despite requesting
  unclipped float output. The adapter now selects float/analytic arithmetic
  and applies requested clipping separately. Explicit ICC overrides embedded
  CICP even on the extended path. A profile with unequal RGB curves where
  only red/green or green/blue match now refuses before writes: the backend
  otherwise shares the wrong curve. A capable per-channel implementation is
  still needed for those profiles. `just output-float-check` is the local gate;
  zencodec additionally compares actual native JXL profiles against Little CMS.

- Fixed on the output-plan review (2026-10-26): adding opaque alpha to native
  ICC-tagged RGB F32 returned `NeedsCms` despite an unchanged current profile.
  `output_native` covers strided RGB/gray U8/U16/F32, exact float bit patterns,
  padding, bypass of unnecessary CMS work, and changed-profile/depth refusal.
  The native JXL reproduction is in zencodec's spatial validation suite.

- Fixed 2026-10-25: `ConvertScratch::prepare` passed alignment padding as pixel data when warming RGB byte gamut kernels. `prepared_contracts::rgb_gamut_preparation_preserves_unprepared_pixels_without_row_allocations` fails before the active-extent correction and compares both gamut directions across row widths.

## Authorized breaking media foundation — 2026-09-28

The user explicitly authorized a coordinated breaking release. Branch
`feat/0.3-retire-planar` stages both packages at 0.3.1 and removes the legacy
planar module/re-exports after migrating zenfilters' filter mask. Compatibility
feature names remain no-ops.
The final 0.2 bridge is retained on `feat/explicit-sample-encoding`. Do not
publish either branch as part of this PR workflow. No generic video frame owner
is added here: concrete native sample views are exercised in zencodec's
`media/` workspace, using `sample::SampleEncoding`. Other historical breaking
queues are not implicitly completed by this change. CICP TC7 now remains
unresolved, per H.273 table 3, instead of incorrectly aliasing BT.709.

The consolidated release candidates are `release/0.2-bridge` (#76) and
`release/0.3.1` (#77). Follow `docs/finalization-release-contract.md`: estimation
is unconditionally deprecated in 0.2 and removed in 0.3; core HDR computation
and root aliases are removed in 0.3. Concrete-type conversion extensions are
sealed there, while Pixel and CMS plugins remain open. No crate publication
is authorized by release-branch preparation.

### Synthesized ICC serialization (verified 2026-11-06)

The CICP bundle generator and fresh moxcms comparison use the private
`icc_profiles/canonical.rs` serializer: D50 PCS illuminant, four-byte tag
alignment, zero timestamp/profile ID, unchanged tag payloads. Tag sharing is
retained. The regenerated blob is 35,379 bytes, SHA-256
`10aaba9c228d86b47718d5e2f4bea739b336fea808c3fd68d7d40ff7a0f80a2f`.
All 159 RGB and 175 gray mappings pass fresh-synthesis equality. All 334
profiles pass structural checks, Little CMS parsing and exact libpng ICC
roundtrips without warnings; a non-D50 negative control fails.
Those historical checks covered the PCS header, alignment and reader
acceptance, not every tag requirement. The 2026-11-08 history audit above
identifies the preserved gray Display-class `wtpt` inconsistency.

`just icc-bundle-identification OUTPUT_DIR` generates and adds bundle hashes
without dropping recognition of older profile bytes. Full corpus regeneration
still uses the existing corpus pipeline. `just icc-bundle-reader-check OUTPUT_DIR`
runs the independent readers. ICC requirements: [ICC.1:2022](https://www.color.org/specifications/ICC.1-2022-05.pdf),
sections 7.1 and 7.2.16.

### JXL PluggableCms adapter defaults revalidated (2026-11-06)

The standalone parity harness routes structured RGB PQ as CICP and refuses
known HLG until OOTF exists. Full 1024x1024 PQ decode versus independently regenerated djxl
PFMs measures worst 0.022236 sRGB / 0.052848 P3 clamped eight-bit codes; both
HLG targets return CMS errors. The frozen historical row classification also
reran. These measurements use historical pixels 04cc06f5, not HDR PR120 or the
current native cohort. Source hashes, recipes and scope are in
`benchmarks/jxl-cms-parity/README.md`. Production zenjxl integration remains
unqualified while the newer cohort fails dependency resolution.

### Practical pure-gamma range guard reproduction (2026-11-06)

`just moxcms-real-pixel-check` compares exact baseline/sign/range revisions on
ProPhoto-to-sRGB F32 transforms with extended range enabled. At (0.5,0.5,0),
baseline/sign-only clip to (1,1,0); range gives (0.6461351,0.57291955,0).
This is an exact-zero ICC pure-gamma failure, not merely a tiny synthetic value.
The comparison is a constructed pixel test, not a corpus frequency survey or
evidence of visible benefit from the separate CICP tiny-value guards. See
`benchmarks/moxcms_review_2026-11-06.md`.

### Power implementation comparison and current numerical limits (2026-11-06)

The matched-domain harness in `benchmarks/pow-comparison` compares pxfm0.1.30
with pinned magetypes f40f5d38 on scalar/AVX2. On the normalized-U16 grid its
sampled maxima are 2.371181022e-6 dirty, 1.866631039e-6 midp, 5.567079078e-3
lowp relative error. This is not a speed or all-f32 accuracy claim. Checked
magetypes methods handle zero but do not provide full IEEE powf semantics:
NaN/negative and subnormal cases remain outside a safe drop-in replacement
contract. In the current revision, pow_midp(NaN,1.8)=0 and
pow_midp(-0.5,1.8)=MIN_POSITIVE on both measured backends. These limits need
caller handling; no sibling source was changed. Details, raw counts, source
links and reproduction are in `benchmarks/pow-comparison/README.md`.

### CMYK precision and capability controls (2026-11-08)

The qualified native JXL input matches C LCMS 2.20 byte-for-byte through Rust
LCMS. MoxCMS 0.9.2 retains 937/786432 components over the existing 0.0013 bound,
maximum 0.006689250469207764; no threshold relaxed. The typed matrix accepts
256 LCMS cases and 192 MoxCMS cases, with 64 explicit MoxCMS CMYK16 declines.
See docs/cmyk-cms-qualification.md for intent/BPC controls and Argyll scope.

### Hash authorization boundary reviewed (2026-11-09)

ICC recognition uses normalized FNV-1a64; canonical replacement checks that hash
and byte length, without normalized byte equality. FNV's zero-state property
and ordinary collisions mean a hit is not a proof of transform identity. No
valid hostile ICC collision was demonstrated in this review. Preserve lookup
semantics while strengthening any pixel-changing authorization; see
docs/pr-contract-review-2026-11-09.md for the exact source and proposed gates.

## Main consolidation decisions (2026-11-09)

- Core precision types describe source/output capacity; conversion policies,
  presets and chain solving belong to zenpixels-convert. Capacity preservation
  is not a pixel-equality promise.
- PlanarLayout retains its growable Vec storage and constructor. No fixed
  four-plane limit or speculative Planes wrapper was adopted.
- Sample-aligned byte pitches are supported. imgref checked borrowing retains
  Option and succeeds only for whole-pixel pitches; image FlatSamples can
  retain other sample pitches. Encoder adapters remain public for zenpipe and
  zenmetrics callers.
- The CLL histogram, percentile and smoothed research methods remain behind
  hdr-experimental with their regression tests. zentone experiments consume
  them; removing them without migrating those experiments was rejected.
  Production peak selection remains measure_max.

## Hashing

`docs/hashing.md` owns inclusion semantics. Hash implementations must exclude
allocation/row/undefined-alpha padding and emit fixed write boundaries, because
Hasher::write need not concatenate calls. Native code padding follows the
existing CodePlane reader. Float payloads are bit-exact. The optional `hash`
feature has a Rust 1.89 floor through zenhash; default core/ICC stays at 1.85.
The ICC key is a normalized FNV64/Murmur3-32 pair, with the latter fixed to
x86_32, little-endian and Quickwit's seed on every target. Do not weaken
qualification masks when regenerating fingerprints. The mirrored const module
and zenhash are checked against the same independently produced golden pairs.
