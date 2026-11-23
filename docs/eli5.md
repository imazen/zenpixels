# Plain-English Mental Model (ELI5)

New to the `zenpixels` 0.4 stack? This page explains how **`garb`**,
**`linear-srgb`**, **`zenpixels`**, and **`zenpixels-convert`** fit together,
what every type actually means, and the real-world gotchas to avoid on day one.

Looking for copy-paste storage basics? Start with the [pixel types guide](guide.md).
Want interactive charts? Try [transfer functions & codegen](transfers.html),
[primaries, HDR & CICP/ICC](color-hdr.html), or the
[conversion kernel browser](kernels.html).
Tuning throughput vs. binary size and compile time? See
[balancing speed and light compilation](speed-and-size.md).

## 1. Which crate should I reach for?

| If you want to… | Use | Why |
|---|---|---|
| Swap, expand, or strip raw `u8` channels (`RGBA↔BGRA`, `RGB→RGBA`, strided rows) with zero color math | **`garb`** (`garb::bytes`) | Tiny, `no_std`, zero-panic `Result<(), SizeError>`, pure byte layout |
| Apply a transfer curve (`sRGB`, gamma, `PQ`, `HLG`, `BT.709`) to `f32`/`u8`/`u16` slices | **`linear-srgb`** (`linear_srgb::default`) | Pure transfer math, SIMD slices (`*_slice` / `*_slice_into`), no buffer types |
| Carry images, row stride, and color meaning (`PixelBuffer`, `PixelSlice`, `ColorContext`, `ColorOrigin`) | **`zenpixels`** | Lightweight interchange types for codecs and image pipelines |
| Convert images, rows, or bands across formats, transfers, gamuts, alpha, orientation, HDR, or ICC | **`zenpixels-convert`** | Full planner + executor (re-exports `zenpixels::*`, including `zenpixels_convert::hdr::*`) |

## 2. The 5 nouns in `zenpixels`

Think of a shipping box:

1. **`PixelFormat`** = **the box shape** (how bytes sit in memory: `Rgb8`,
   `Rgba8`, `RgbaF32`, `OklabF32`, etc.). It says nothing about color space or
   gamma.
2. **`PixelDescriptor`** = **the shipping label** (`PixelFormat` + transfer
   function + color primaries + alpha mode + signal range, such as
   `PixelDescriptor::RGB8_SRGB` or `PixelDescriptor::RGBF32_LINEAR`).
   - **Gotcha:** `.with_descriptor(...)` **only replaces the sticker on the
     box** — it never converts a single pixel byte. To change sample numbers,
     use `zenpixels-convert`.
3. **`ColorContext`** = **what the pixels mean *right now*** (at most one
   active `ColorProfile` — `Cicp` or `Icc` — plus luminance metadata).
   **`PixelEncoding`** borrows `(PixelDescriptor, Option<&ColorContext>)`
   together without cloning `Arc`s.
4. **`ColorOrigin`** = **what the original file claimed on disk** (for example
   a PNG or AVIF that carried both an ICC chunk and a CICP chunk). It is
   history for encoder provenance and must never override converted pixels.
5. **`PixelBuffer` vs `PixelSlice` vs `BandedPixelBuffer`**:
   - `PixelBuffer` owns one contiguous (possibly row-padded) image allocation.
   - `PixelSlice` / `PixelSliceMut` is a zero-copy borrowed view or rectangular
     crop. **Width and height count pixels; `stride()` counts bytes.** Crops
     keep the parent stride, so read active pixels with `row(y)`. Views carry
     the underlying buffer's `'a` lifetime, so `buffer.crop_view(...)`,
     `buffer.as_slice().crop_view(...)`, and `ready.pixels().row(0)` all work
     directly.
   - `BandedPixelBuffer` owns a gigapixel-safe image split into capped-size
     horizontal bands with 64-bit frame geometry.

## 3. Plan → Prepare → Execute

Converting pixels is three steps so hot loops allocate nothing:

1. **Choose policy (`ConvertOptions` + `ConversionSpec`):**
   - `ConvertOptions` states which lossy changes are permitted (alpha removal,
     depth narrowing, gamut clipping, luma). Nothing lossy happens silently.
   - `ConversionSpec::new(options)` adds execution and evidence knobs
     (`with_accelerators`, `with_float_facts`, `with_cms`, `with_intent`,
     `with_provenance`, `with_source_precision`).
   - `ConversionSpec::preserve_samples()` requires proven reversible
     representation changes (zero silent loss).
2. **Build a plan (`ConvertPlan::build`, `ConvertPlan::select`, or `OutputPlan::build`):**
   - `ConvertPlan::build(src.encoding(), dst.encoding(), &spec)` plans one
     specific target.
   - `ConvertPlan::select(src.encoding(), &candidates, &spec)` negotiates the
     best target an encoder accepts.
   - `OutputPlan::build(src.encoding(), output_target, &spec)` +
     `EncodeReady::from_owned(buf, plan)` converts pixels **and** publishes
     matching output CICP/ICC metadata atomically (reusing the input allocation
     with zero copy on identity).
3. **Prepare and execute (`plan.prepare(max_width)` → `RowConverter`):**
   - Run `converter.try_convert_row(src_row, dst_row, width)` inside a row
     loop, **or** pass whole buffers, views, or bands directly to
     `converter.convert(&src, &mut dst)` (`PackedSource` = `&PixelSlice`,
     `&PixelBuffer`, `&BandedPixelBuffer`; `PackedSink` = `&mut PixelSliceMut`,
     `&mut PixelBuffer`, `&mut BandedPixelBuffer`).
   - On success, owned sinks (`&mut PixelBuffer`, `&mut BandedPixelBuffer`)
     receive updated `ColorContext` plus `ExactDepth`/`FloatFacts`; borrowed
     `&mut PixelSliceMut` receives updated `ColorContext` only (sample evidence
     lives on owned buffers).
   - Need a one-liner? Call `slice.convert_to(PixelDescriptor::RGBF32_LINEAR)?`
     from `PixelSliceConvertExt`.

## 4. Universal scalar core vs. opt-in SIMD (`accel` & `const Accelerators`)

In Rust, Cargo features unify globally across the dependency tree. If every
codec crate turned on every SIMD conversion table (sRGB, BT.709, PQ, HLG,
Oklab, gamut matrices, perceptual spaces), tiny decoders and Wasm apps would
pay hundreds of kilobytes of dead code and slow builds.

Instead, `zenpixels-convert` 0.4 separates **correctness** from **SIMD codegen**:

- **Zero config (default):** Every valid native pair runs on a compact,
  zero-scratch **universal fast-FMA scalar core** that is **0-bit identical**
  to the x86-64/AArch64 SIMD kernels. Libraries never fail with `NoPath` for
  missing SIMD knobs, and unused SIMD kernels dead-strip to `0 B`.
- **Opt in at your exact boundary:**
  - `accel::enable_all()` once in `main()` for full SIMD across the entire
    process (including inside codec crates).
  - Selective startup (`accel::enable_web_sdr()`, `enable_layout()`,
    `enable_srgb()`, `enable_float()`, `enable_gamut()`, `enable_hdr()`,
    `enable_oklab()`).
  - Compile-time `const Accelerators` presets and builders
    (`Accelerators::SCALAR`, `LAYOUT`, `WEB_SDR`, `ALL`, `.with_*()`,
    `.union()`) passed to `ConversionSpec::with_accelerators(&...)` with zero
    global state. `WEB_SDR` accelerates layout, sRGB, `U8 ↔ F32` float/premul,
    and SDR gamut while leaving `U16`/`F16`, HDR, BT.709, Gamma 2.2, and Oklab
    SIMD dead-stripped (they still execute via the universal scalar core).
  - Top-level binary Cargo feature `fast` (never enable in libraries — it
    unifies across Cargo's dependency graph, though per-spec
    `&Accelerators::SCALAR` still overrides it hermetically).

```rust
use zenpixels_convert::{Accelerators, ConversionSpec, ConvertOptions, accel};

// Full SIMD for apps/CLIs/servers:
accel::enable_all();

// Compile-time scoped subset (dead-strips unused SIMD families):
const LAYOUT_AND_SRGB: Accelerators = Accelerators::new().with_layout().with_srgb();
let scoped = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&LAYOUT_AND_SRGB);

// Hermetic scalar-only override for 0-bit parity tests (ignores global registry):
let scalar = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&Accelerators::SCALAR);
# assert!(accel::enabled().is_all());
# let _ = (scoped, scalar);
```

See **[Balancing speed and light compilation](speed-and-size.md)** for measured
binary sizes, feature budgets, and recipes by workload.

## 5. `FloatFacts`: proven bounds without per-pixel guesswork

Untrusted `f32` buffers can contain `NaN`, `±inf`, subnormals, negatives, or
values above `1.0`. Defending against all six on every pixel costs extra clamps
and masks.

`zenpixels::sample::FloatFacts` is a 6-bit bitmask of proven properties:
`NO_NAN`, `NO_INFINITE`, `NO_SUBNORMALS`, `NON_NEGATIVE`, `POSITIVE`, and
`AT_MOST_ONE` (plus `NONE` for unknown/untrusted data).

```rust
use zenpixels::sample::FloatFacts;
use zenpixels_convert::{
    ConversionSpec, ConvertOptions, ConvertPlan, DepthPolicy, PixelBuffer, PixelDescriptor,
};

let mut buf = PixelBuffer::try_new(4, 1, PixelDescriptor::RGBF32_LINEAR)?;
let facts = FloatFacts::measure(&buf.as_slice())?;
assert!(facts.contains(FloatFacts::NO_NAN | FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE));

buf.set_float_facts(facts)?;
let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
let spec = ConversionSpec::new(opts).with_float_facts(facts);
let plan = ConvertPlan::build(
    buf.encoding(),
    PixelDescriptor::RGB8_SRGB.into(),
    &spec,
)?;
assert_eq!(plan.input_float_facts(), facts);

// Any mutable borrow automatically clears cached facts so stale proofs never lie:
let _mut_view = buf.as_slice_mut();
assert_eq!(buf.float_facts(), FloatFacts::NONE);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Inspect propagated output guarantees via `plan.output_float_facts()`. If an
input buffer violates required facts, `ConvertError::FloatFactsUnsatisfied` is
returned.

## 6. Real-world gotchas (Oklab, HDR, features, errors)

Every item below came straight out of docs-only user testing:

- **Wide-gamut Oklab is CIE XYZ D65-anchored (zero clipping):**
  `PixelDescriptor::OKLABF32` represents perceptual color in CIE XYZ D65, not
  just BT.709. Converting Display P3 or BT.2020 → `PixelDescriptor::OKLABF32` →
  Display P3 uses that gamut's own `RGB ↔ LMS` matrix directly — bit-exact
  with zero BT.709 clipping and no extra `.with_primaries(...)` call:

  ```rust
  use zenpixels_convert::{
      ColorPrimaries, ConversionSpec, ConvertOptions, ConvertPlan, DepthPolicy,
      PixelBuffer, PixelDescriptor, PixelSliceConvertExt,
  };

  let p3_srgb = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
  let src = PixelBuffer::from_vec(vec![0, 255, 128], 1, 1, p3_srgb)?;
  let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
  let oklab = src.as_slice().convert_to_with_options(PixelDescriptor::OKLABF32, &opts)?;
  let back = oklab.as_slice().convert_to_with_options(p3_srgb, &opts)?;
  assert_eq!(back.as_slice().row(0), &[0, 255, 128]);
  # let _ = (ConversionSpec::new(opts), ConvertPlan::build(src.encoding(), PixelDescriptor::OKLABF32.into(), &ConversionSpec::new(opts))?);
  # Ok::<(), Box<dyn std::error::Error>>(())
  ```

- **Single-crate HDR imports:** `zenpixels_convert::hdr` re-exports all core
  anchor types (`Nits`, `DiffuseWhite`, `HdrPeak`, `HlgRendering`,
  `ContentLightLevel`, `MasteringDisplay`, `St2086Codes`) alongside converter
  HDR helpers — no separate `zenpixels` dependency required.
- **HDR→SDR needs `features = ["bt2446a"]` and explicit peaks:**
  - Under default features, an explicitly configured PQ→SDR or HLG→SDR plan
    returns `ConvertError::HdrFeatureRequired` until you enable
    `zenpixels-convert = { ..., features = ["bt2446a"] }` (or pass a
    feature-gated `ToneMapper`).
  - `HdrPeak::Measured` and `HdrPeak::Declared` only resolve on buffer-level
    calls that see pixel data. `ConvertPlan::build` / `RowConverter` see
    encodings only and require concrete `HdrPeak::Nits(...)`.
  - For HLG→SDR, `with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)` supplies
    only the OOTF shape — you still pass `with_hdr_peak(HdrPeak::Nits(...))`,
    `with_sdr_peak(...)`, and `with_sdr_white(...)` (`HlgRendering.peak()` is
    not reused as the content peak).
- **Features & `ConvertError::FeatureRequired`:**
  - Default-on in `zenpixels-convert`: `std`, `fast-transpose`, `oklab`,
    `hdr-transfers`.
  - Default-off: `perceptual` (`Oklch`, `Lch`, `Jzazbz`, `Jzczhz`, `Hsl`,
    `Hsv`), `fast`, `precise-portable`, `bt2446a`, `hdr-experimental`,
    `icc-db`, `css`, `planar`, `cms-moxcms`, `cms-lcms`, `rgb`, `imgref`,
    `introspection`, `avx512`.
  - `ConvertError::FeatureRequired { feature, from, to }` fires when a
    descriptor exists at runtime but its conversion feature is compiled out
    (for example `OklabF32` when `oklab` is off, or `zenpixels/perceptual`
    unified on while `zenpixels-convert/perceptual` is off). With default
    features, `perceptual`-only `PixelFormat` variants are not compiled in at
    all; `ChannelLayout::Lab` targets return `ConvertError::NeedsCms`.
  - Enabling `cms-moxcms` or `cms-lcms` never auto-selects a CMS — pass
    `.with_cms(&MoxCms)` or `.with_cms(&Lcms)` on `ConversionSpec`.
- **Located errors:** `ConvertError` is wrapped in `whereat::At<ConvertError>`,
  so `{err:?}` includes the raising source location alongside the error variant.

[Pixel types guide](guide.md) ·
[Transfer functions](transfers.html) ·
[Primaries, HDR & CICP/ICC](color-hdr.html) ·
[Kernel browser](kernels.html) ·
[Speed & light builds](speed-and-size.md) ·
[Explore one pixel](index.html) ·
[Core API](https://docs.rs/zenpixels)
