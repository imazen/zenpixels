# `zenpixels-convert` dedup + missed-opt audit — 2026-06-23

Audit covering the `zenpixels-convert/src/` diff between
`zenpixels-convert-v0.2.14` and `main@origin` (10 changed files, ~6100
inserted LOC). Scope: code duplication, missed SIMD / perf
opportunities, type-system smells. Reads only; trivial wins applied
in-place, larger findings surfaced for follow-up.

## Trivial wins applied in-place (this commit)

| File | Lines (pre) | Change | LOC saved |
|------|-------------|--------|-----------|
| `src/ext.rs` | 460-475 | Replace two inline `ResourceEstimate { peak_memory_bytes: 0, wall_time_ms: 0.0, breakdown: Vec::new(), confidence: Unknown }` literals with the existing `ResourceEstimate::zero(EstimateConfidence::Unknown)` helper. | 10 |
| `src/hdr/mod.rs` + `src/hdr/bt2446a.rs` + `src/hdr/bt2446a_simd.rs` + `src/hdr/measure.rs` | n/a (cross-file constants) | Three copies of the BT.2020 NCL luma coefficients (`LR/LG/LB` in `bt2446a`, `LR_2020/LG_2020/LB_2020` in `bt2446a_simd`, `KR/KG/KB` in `measure::simd_kernel`) consolidated into `pub(super) const BT2020_LR / BT2020_LG / BT2020_LB` at the `hdr` module root. The three sites now `use super::BT2020_L*` and locally re-alias if their old name was load-bearing. Identical numerical values (`0.2627 / 0.6780 / 0.0593`); behaviour is byte-identical. | 8 |

**Total trivial-win LOC delta:** ~18 LOC, no behaviour change. The
constants consolidation kills a three-way drift hazard (the bt2446a
scalar/SIMD split or the BT.2020 luma reduction in `measure` could
silently shift apart). All 513 lib tests pass under
`--features hdr-experimental`.

## Section 1 — Duplications (top 7)

### D1. `gamut_compress::GamutBoundaryLut::compress_planes` vs `SoftCompress::apply_strip` — rational-knee math copy-pasted
`src/hdr/gamut_compress.rs:103-147` (planar `(l, &mut a, &mut b, knee)`)
and `src/hdr/gamut_compress.rs:268-300` (interleaved RGB strip).
Both compute chroma `c = √(a² + b²)`, hue `h = atan2(b, a)`, look up
`max_c = lut.max_chroma(l, h)`, build the rational knee
`knee_c + range·excess / (excess + range)`, and apply the scale.
**Consolidate:** factor a private `compress_chroma_inplace(l, a, b, lut, knee) -> (a', b')`
helper; `apply_strip` calls it after `rgb_to_oklab` and before `oklab_to_rgb`,
`compress_planes` calls it per planar pixel. **Savings:** ~35 LOC.
**Risk:** Low (pure refactor, behaviour-preserving — `apply_strip` is the
only public consumer; `compress_planes` is `pub` but only one in-tree
caller in `zenfilters`).

### D2. `measure.rs::simd_kernel` — 4 strip kernels share an identical 8-channel deinterleave + chunk shape
`src/hdr/measure.rs:880-977` (`accumulate_strip_max_rgb_tier`),
`:981-1057` (`accumulate_strip_luma_bt2020_tier`),
`:1154-1195` (`scan_row_max_rgb_tier`),
`:1198-1246` (`scan_row_luma_bt2020_tier`).
Each opens with the same 14-line preamble (`zero/wn/log2_min/inv_step/...
splats`), then runs an `iter = row.chunks_exact(LANES * N)` loop whose
body de-interleaves into three `[f32; LANES]` scratch arrays
(`ra/ga/ba`), loads into `f32x8`, applies the per-method reduction
(`max(0,R,G,B)` for MaxRgb or `KR·R + KG·G + KB·B` for BT.2020), and
runs the per-method downstream. The histogram path then runs
log2 → bin scatter; the fast path just reduces.
**Consolidate:** a generic `fn deinterleave_chunk_n<const N: usize>(chunk: &[f32]) -> ([f32; LANES], [f32; LANES], [f32; LANES])`
helper, plus a small `reduce_max_rgb_simd(token, r, g, b) -> f32x8` / `reduce_luma_bt2020_simd(token, r, g, b) -> f32x8`
pair, would cut the 4 kernels to maybe ~40 LOC each (~160 LOC vs
~360 LOC today). **Savings:** ~120 LOC.
**Risk:** Medium (SIMD dispatch shape — the magetypes `define(f32x8)`
attribute lives on the outer fn; threading the de-interleave through a
generic helper needs care to keep LLVM inlining).

### D3. `convert_kernels.rs` — 4 matte-composite dispatch tables (one per channel-type)
`src/convert_kernels.rs:930-1111` ships
`dispatch_matte_f32_rgba`, `dispatch_matte_f16_rgba`,
`dispatch_matte_u8_rgba`, `dispatch_matte_u16_rgba` — four 30-line
`match tf { Srgb => …, Bt709 => …, Pq => …, Hlg => …, … }` arms over
`TransferFunction`. The only per-type variation is the row buffer
element type and the per-pixel `MatteTf::eotf_*` method called.
**Consolidate:** a single `dispatch_matte<T: MatteTfDispatch>(...)`
generic, or a `macro_rules! dispatch_matte` that expands the 4 channel
types from one source. **Savings:** ~120 LOC.
**Risk:** Low (closed dispatch surface, called via `ConvertStep`).

### D4. `convert_kernels.rs` — 8 f32 TF wrapper kernels (sRGB / sRGBExtended / BT.709 / Gamma22)
`src/convert_kernels.rs:2076-2192` — 8 nearly-identical wrappers
(`srgb_f32_to_linear_f32`, `linear_f32_to_srgb_f32`,
`srgb_f32_to_linear_f32_extended`, `linear_f32_to_srgb_f32_extended`,
`bt709_f32_to_linear_f32`, `linear_f32_to_bt709_f32`,
`gamma22_f32_to_linear_f32`, `linear_f32_to_gamma22_f32`). Each does:
cast `&[u8]` to `&[f32]`, `copy_from_slice` into the dst, then call the
matching `linear_srgb::default::<tf>_slice`. The 4-step body is
copy-pasted with one identifier change.
**Consolidate:** one `apply_tf_f32_slice(src: &[u8], dst: &mut [u8], count: usize, tf_fn: fn(&mut [f32]))`
helper. **Savings:** ~80 LOC.
**Risk:** Low.

### D5. `convert.rs` — hand-rolled `Debug for ConvertStep` (~93 LOC) when `#[derive(Debug)]` would produce identical output
`src/convert.rs:273-365`. The hand-rolled impl just spells out
`f.debug_struct(...).field(...)` for each variant, byte-identical to
what a derive would emit. The enum has no fields requiring custom
formatting.
**Consolidate:** delete the manual impl, add `Debug` to the existing
`#[derive(Clone, Copy, ...)]` line on `ConvertStep`.
**Savings:** ~90 LOC.
**Risk:** Low (output text identical except `_` placeholders for
`#[non_exhaustive]` would differ — none currently used).

### D6. `adapt.rs` + `converter.rs` — five copies of the per-row strided convert loop
`src/adapt.rs:262-273, :307-317, :418-426, :678-688` plus the
canonical version at `src/converter.rs:244-273`. Each copy walks
`for y in 0..rows { src_off = y*src_stride; dst_off = y*dst_stride;
converter.convert_row(&src[src_off..], &mut dst[dst_off..], width); }`
with the same `src_bpp / dst_bpp / dst_stride` setup.
**Consolidate:** the `adapt.rs` callers should delegate to
`RowConverter::convert_rows` (already canonical). **Savings:** ~40 LOC.
**Risk:** Low (mechanical delegation, no policy change).

### D7. `convert.rs` — three near-identical "fused-fast-path early-return" blocks
`src/convert.rs:672-720`. Three consecutive
`if desc.channel_type == … && desc.transfer == … { push fused step; … return Ok(...); }`
blocks differ only in which `(ChannelType, TransferFunction) →
FusedSrgb*` triple they match. Same pattern in
`new_with_hdr_config` at `:932-979`.
**Consolidate:** a small `try_fused_gamut_step(from, to, flat) -> Option<ConvertStep>`
table-driven helper. **Savings:** ~35 LOC.
**Risk:** Low.

### Honourable mentions (still worth a follow-up)
- `convert.rs` alpha-mode-change match repeated verbatim at `:564-574` and `:1015-1025`. ~15 LOC.
- `convert.rs` `f32_linearize_step` / `f32_encode_step` mirror tables at `:1478-1504`. ~15 LOC.
- `adapt.rs` zero-copy + transfer-agnostic match block copy-pasted at `:217-250` and `:623-654`. ~35 LOC.
- `adapt.rs` 3 `adapt_for_encode*` entry points share ~80% body (`:176-193`, `:199-281`, `:608-696`). ~70 LOC.
- `convert_kernels.rs` 4 f16 chunked-scratch patterns (matte / premul / unpremul) replicate the same `f16_bits_to_f32_slice → math → f32_to_f16_bits_slice` 3-pass shape (5 sites). ~80 LOC.
- `convert_kernels.rs` flatten `[[f32;3];3] → [f32;9]` + un-flatten pattern at `:232-305` and `:2822-2851`. ~25 LOC.
- `convert_kernels.rs` RGB-vs-RGBA Oklab kernel pairs (4 inner fns, 2 logical pairs) at `:2684-2815`. ~50 LOC via `const STRIDE: usize` generic.
- `measure.rs` `bin_for_nits` (top-level, `:293-308`) vs `saturating_bin_scalar` (inside `simd_kernel`, `:1063-1076`) — same semantics, two names. ~10 LOC.

## Section 2 — Missed optimizations (top 5)

### O1. `convert_kernels.rs::gamut_matrix_rgb_f32` / `gamut_matrix_rgba_f32` are scalar — no SIMD dispatch
`src/convert_kernels.rs:2822-2852`. Pure scalar `for p in 0..width { let r = …; let g = …; let b = …; … }` loops. These are part of the hot path for every f32 gamut-converted strip (BT.709 ↔ BT.2020 / Display P3). The crate already has `multiply_color_channels_tier` (`:1746-1810`) using `archmage::magetypes(f32x16)`; the 3×3 matmul has the same data shape (3 channels × N pixels) and should dispatch via `incant!` the same way. Expected speedup: 2–4× on AVX2, 4–8× on AVX-512.
**Calibrate against:** `benchmarks/t7_gamut_2026-04-23_baseline.txt`
(the file `estimate.rs` already cites at 21.84 GiB/s for the scalar
path — measurable headroom).
**Risk:** Medium (SIMD dispatch shape; loops are correctness-load-bearing).

### O2. `convert_kernels.rs` premul/unpremul kernels (u8 / u16 / f16 RGBA / GA) — `#[autoversion]` only, no `incant!`
`src/convert_kernels.rs:2208-2255` (premul) and `:2403-2498` (unpremul)
contain `premul_u8_ga`, `premul_u16_ga`, `premul_u16_rgba`,
`premul_f32_ga`, `unpremul_u8_*`, `unpremul_u16_*`, `unpremul_f32_*`.
All use `#[autoversion]` (relies on LLVM autovec) instead of explicit
`incant! / magetypes` SIMD. The integer divide in u16 unpremul
(byte-level reciprocal LUT) and the f32 divide-by-alpha both benefit
from explicit SIMD lanes; `garb::bytes::premultiply_alpha_rgba_u8_copy`
(referenced at `:2339`) is already SIMD for the u8/RGBA shape — the
others should match.
**Calibrate against:** `benchmarks/t5_alpha_…` (the StraightToPremul /
PremulToStraight cells in `estimate.rs`).
**Risk:** Medium.

### O3. `gamut_compress.rs::GamutBoundaryLut::compress_planes` runs scalar over planar a/b
`src/hdr/gamut_compress.rs:103-147`. Per-pixel `sqrt + atan2 + lookup +
rational knee` — completely scalar. With magetypes `f32x8` we can do
`sqrt + atan2_midp + lut_gather_scalar + knee_rational` in a SIMD
strip. The LUT gather is hard to vectorise (random-index lookup), but
the `c = √(a² + b²)`, `h = atan2(b, a)`, and `compressed_c = knee_c + range·excess / (excess + range)`
math is straight FMA. A 4–6× speedup on AVX2 is realistic if the LUT
gather is structured (still a scalar lane scatter, like the histogram
path's per-lane sub-histograms in `measure.rs::simd_kernel`).
**Calibrate against:** no benchmark exists yet — the estimate cell in
`estimate.rs:362-367` is `Heuristic` at 3.0 GiB/s. Add a bench at
`benchmarks/soft_compress_oklch_*`.
**Risk:** High (needs new bench; touches the critical OKLab math).

### O4. `convert_kernels.rs::multiply_color_channels_tier` builds the per-call splat array on every entry
`src/convert_kernels.rs:1746-1786`. Each call constructs
`[factor, factor, factor, 1.0, factor, factor, factor, 1.0, …]` (16 lanes)
by stores into a `[f32; 16]` array, then loads as `f32x16`. For RGBA
where `factor` is loop-invariant (per-row, every strip) this is fine,
but the array build is per-strip-call. If callers are doing tile-level
conversion (small strips of 16-pixel chunks), this becomes ~1% of total
time. Two options: (a) cache the splat in a `RowConverter` field;
(b) use a pattern-based splat (mask AVX2 has efficient instructions
for).
**Calibrate against:** `benchmarks/t5_alpha_…` and
`benchmarks/t1_layout_…`.
**Risk:** Low (clarity helper, modest speedup).

### O5. `hdr::measure::measure_max_smoothed` admits it's scalar (~1.3 Gpix/s) and the SIMD-friendly two-pass design is documented but unbuilt
`src/hdr/measure.rs:622-628` carries the comment:
> "Scalar streaming path — auto-vectorises to ~1.3 Gpix/s on Zen 4.
> … The right SIMD path is a two-pass design (deinterleave+reduce
> into a row scratch, then 3-tap box-max over the scratch), but
> that's a separate commit."

Action: build the two-pass SIMD. Phase 1 (deinterleave + reduce into
`Vec<f32>` row scratch) reuses the deinterleave from O1's helper.
Phase 2 (3-tap box max + sum) is a single-pass scalar/SIMD over the
scratch. Expected: ≥2× speedup over the current `~1.3 Gpix/s` (toward
the `measure_max` ceiling of ~2.7 Gpix/s).
**Calibrate against:** add to
`benchmarks/measure_max_throughput_2026-06-19.md`.
**Risk:** Medium (the smoothed path is `#[doc(hidden)]` experimental,
so the API surface is forgiving).

### Other perf hot spots worth a line
- `hdr/measure.rs::scan_row_max_mean_smoothed::reduce_at` closure (`:346-357`) is called from a scalar streaming loop; LLVM hoists the `match method` per the doc, but a const-generic `<METHOD: u8>` rewrite would prove it. ~zero LOC, +5–10 % on smoothed path.
- `convert.rs::compose` (`:1168-1217`) clones the entire steps `Vec` even when the second plan is `Identity`; trivial early-out saves an allocation per plan composition.
- `adapt.rs::is_fully_opaque` (`:699-744`) runs `match desc.channel_type()` inside a per-pixel loop instead of dispatching once outside. ~5 LOC + perf win on `DiscardIfOpaque` policy path.

## Section 3 — Type-system smells

### T1. `convert.rs::ConvertStep` has 5 `FusedSrgb*` variants that all carry only `[f32; 9]`
`src/convert.rs:227-243`. Variants `FusedSrgbU8GamutRgb([f32; 9])`,
`FusedSrgbU8GamutRgba([f32; 9])`, `FusedSrgbU16GamutRgb([f32; 9])`,
`FusedSrgbU8ToLinearF32Rgb([f32; 9])`, `FusedLinearF32ToSrgbU8Rgb([f32; 9])`.
They could merge into one `Fused { kind: FusedKind, matrix: [f32; 9] }`
with a tag, simplifying `step_name`, `step_cost_ns_per_mp`, the
`intermediate_desc` match, and the `Debug` impl by ~30 LOC total.
**Risk:** Medium (touches dispatch shape — `ConvertStep` is `pub(crate)`
so no API impact).

### T2. `hdr/mod.rs::HdrMetadata` already marked `#[deprecated]` — public field shape is queued for 0.3.0
`src/hdr/mod.rs:94-106`. Already noted in code: "redundant with
zencodec::Metadata and frozen-shaped". Confirming it's correctly
deprecated; the public re-export at `src/lib.rs:533-535` should be
`#[deprecated]`-flagged too (it inherits the type's deprecation, fine).
**No action needed.**

### T3. `convert.rs::intermediate_desc_for_estimate` is a vestigial `pub(crate)` bridge
`src/convert.rs:1305-1312`. Only consumer is `src/estimate.rs:543`.
Either widen `intermediate_desc` itself to `pub(crate)` and drop the
bridge, or inline the one call.
**Risk:** Low.

### T4. `converter.rs::ExternalTransform` `Shared`/`Owned` split silently drops `Owned` on Clone
`src/converter.rs:41-46, :124-147, :233-238, :319-324`. The `Owned`
variant of the enum carries a `Box<dyn RowTransformMut>`; the Clone
impl at `:323` silently drops it (per the existing comment). One
caller in-tree uses this distinction. Replacing the enum with
`Arc<dyn RowTransform>` + a `RowTransformMut` adapter collapses the two
variants and removes the Clone hazard.
**Risk:** Medium (touches the public CMS plugin trait boundary).

### T5. `hdr/mod.rs::reinhard_tonemap / reinhard_inverse / exposure_tonemap` are already `#[deprecated]` + `#[doc(hidden)]` — confirm 0.3.0 removal
`src/hdr/mod.rs:186-254`. All three marked for removal in 0.3.0 with
the message "use `zenpixels_convert::hdr::Bt2446A`". They're still
re-exported from `lib.rs:527, :535`. **No action needed** — flagged so
the 0.3.0 release sweep doesn't miss them.

### T6. `ext.rs::PixelBufferConvertExt::estimate_*` family — 6 default impls all return `ResourceEstimate::zero(Unknown)`
`src/ext.rs:176-270`. Each default impl is a one-liner returning
`ResourceEstimate::zero(EstimateConfidence::Unknown)`. The 6 nearly
identical defaults could collapse to one supertrait method
(`fn default_estimate() -> ResourceEstimate { … }`) that each
override calls. Marginal LOC saving (~12 LOC); the real value would be
forcing-by-construction that downstream impls can't forget the same
zero-shaped default.
**Risk:** Medium (touches public trait surface).

### T7. `estimate.rs::step_name` is a 60-arm `match` that mirrors `ConvertStep`'s variant names verbatim
`src/estimate.rs:372-432`. A trivial `strum::Display` derive (or a
`#[derive(strum::Display)] #[strum(serialize_all = "PascalCase")]`)
would generate the entire match. No new dep needed if a `const fn
variant_name(&self) -> &'static str` lives on `ConvertStep` instead.
**Savings:** ~60 LOC. **Risk:** Low.

## Section 4 — Items left for the author

These need human judgement (architectural change, breaking-ish, or
require a benchmark to validate the win):

- **D2** (4 SIMD strip kernels) — SIMD dispatch shape, worth a careful
  one-PR rewrite.
- **D3** + **D4** (8 matte / TF wrappers) — macro vs trait dispatch
  is a design choice.
- **O1–O3** — SIMD-isation needs a fresh bench to confirm the win is
  real (don't ship without a measurement per the audit rules).
- **T1** (`FusedSrgb*` collapse) — touches the `ConvertStep` enum
  used by `estimate.rs`, `intermediate_desc`, and several kernels.
- **T4** (`ExternalTransform` Clone hazard) — touches the public CMS
  plugin trait boundary.

## Methodology

- Read all 10 changed files (15145 lines total across the audit scope)
  via `Read`, plus the cross-file constant search via `Grep`.
- Three parallel research sub-agents covered `convert.rs`,
  `convert_kernels.rs`, and `adapt.rs`+`converter.rs` independently;
  findings cross-checked against the lead reader (this agent) on the
  HDR files.
- Build verified clean under
  `cargo build --features hdr-experimental,pipeline,cms-moxcms,rgb`.
- Lib tests: `cargo test --features hdr-experimental --lib` — 513
  passed, 0 failed.

## Total LOC budget visible from this audit

- **Landed this commit:** ~18 LOC.
- **Surfaced for follow-up (Sections 1–3):** ~720 LOC across
  duplications + ~90 LOC across type-smell simplifications +
  significant SIMD speedups (not LOC-measurable).
- **High-risk items (left for author):** ~60 LOC (T1, T4).
