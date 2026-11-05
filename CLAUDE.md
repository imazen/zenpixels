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

The `builtin_profiles` module (currently in `[Unreleased]`) is a live example: it exposes 8 public items (enum + const + 6 functions) via `pub mod builtin_profiles` and has zero consumers outside its own tests. Before the next publish, demote to `pub(crate)` or find a real caller. Catching this *before* it ships is the whole point.

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

- Fixed at F32 (2026-11-05): moxcms's `ToneCurveEvaluatorPureGamma` uses `dirty_powf`, which returns unrelated values where the power isn't representable (gamma 1.8 of zero is -1.65e8) and takes a negative base's sign from the exponent's integer part. A zero channel of a ProPhoto F32 sample clipped ordinary colors to full scale: 13,254 of 421,824 channels were over 2 codes from the 8-bit route, worst 255; now 0, worst 1. `PureGammaCurve` keeps the backend's evaluator for representable magnitudes and takes `libm` for the rest, on sources and gray routes. Evidence and limits: `benchmarks/pure_gamma_f32_2026-11-05.md`.

- Open (2026-11-05): 8-bit routes through `U8ThroughF32` still take `f32_transform` and so keep this bug. Gray gamma 1.8 to gray gamma 2.2 turns 0 into 255, and ProPhoto (128, 128, 0) into Adobe RGB comes out (255, 255, 0). `U8MatrixShaper` routes, any target with sRGB's curve, aren't affected. Sending `u8_transform` to `exact_gamma_f32_transform` fixes them; that changes 8-bit output, so it waits for the owner.

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
  steps. `tests/gamut_across_layouts.rs` measures every route against independent `f64`
  math. A row split could also change `f32` bits, and an odd scratch size could misalign
  F32 rows: `tests/row_partition_parity.rs`. BT.709 and Gamma22 `f32` output keep negative
  values while sRGB `f32` clips: that difference is existing behavior, pinned by the test.

- Contract (2026-11-04): adjacent inputs can reverse by at most 1 `u16` code on PQ, sRGB, BT.709 encode and gamma encode, from f32
  rounding in polynomial evaluation. `adjacent_inputs_reverse_by_at_most_one_code` holds it. Counts: `benchmarks/row_kernel_tiers_2026-11-04.md`.

- Fixed (2026-11-04): a PQ or HLG matte put white at the format's peak: 10,000 cd/m² on PQ.
  `CompositeOnto::reference_white` now places it, defaulting to BT.2408.

- Open (2026-11-04): `RowConverter::new` puts SDR white at PQ's peak, 10,000 cd/m², where CSS puts 203. `ColorConvertExt::try_convert_to` refuses this.
  `docs/css-color-parity.md` lists this and the other CSS differences.

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
