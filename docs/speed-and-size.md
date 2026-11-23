# Balancing speed and light compilation

Most Rust image libraries force a bad trade-off: either every small decoder
links hundreds of kilobytes of SIMD tables you never call, or turning SIMD on
breaks `no_std` and Wasm builds.

`zenpixels-convert` 0.4 separates **correctness** from **SIMD codegen**. Every
valid native pixel pair works out of the box through a compact, zero-scratch
**universal fast-FMA scalar core** that produces the **exact same bytes (0-bit
diff)** as the x86-64 and AArch64 SIMD kernels. From there, you opt into only
the SIMD families your binary actually needs — using one-line startup presets
or compile-time `const Accelerators` helpers.

New to the core buffer and slice types? Start with the [pixel types guide](guide.md)
or the [plain-English ELI5 mental model](eli5.md). Curious why `zenpixels-convert`
has over 100 specialized steps and SIMD kernels? Explore every one in the
**[live conversion kernel browser](kernels.html)** (plus
[transfer function charts](transfers.html) and [primaries/HDR](color-hdr.html)).

## Pick your recipe

| What you are building | Recommended setup | Why |
|---|---|---|
| **App, CLI, server, or benchmark** | Call `accel::enable_all()` once in `main()` | Unlocks full SIMD across the entire process, including inside codec crates |
| **Web SDR image service** | `accel::enable_web_sdr()` or `&Accelerators::WEB_SDR` | Accelerates layout, sRGB, float, and SDR gamut; HDR & Oklab SIMD dead-strip |
| **Tiny Wasm or embedded module** | Leave default scalar, or pass `&MY_ACCEL` (`const`) + `default-features = false` | Links `0 B` of unused SIMD kernels; cuts stripped Wasm delta to `+87 KB` |
| **Reusable codec or library crate** | Never call `enable_all()` and never enable Cargo feature `fast` | Stays light for callers; automatically speeds up when the host app opts in |
| **Bit-exact parity test suite** | `.with_accelerators(&Accelerators::SCALAR)` vs `ConversionSpec::fast(opts)` | Forces universal scalar hermetically even if `enable_all()` ran earlier |

## How the 4 opt-in levels work

### Level 1: Zero config (universal scalar core)

Do nothing special in library crates, codecs, or size-sensitive tools:

```rust
use zenpixels_convert::{
    ConversionSpec, ConvertOptions, ConvertPlan, PixelBuffer, PixelDescriptor,
};

let src = PixelBuffer::from_vec(vec![64, 128, 192, 255], 1, 1, PixelDescriptor::RGBA8_SRGB)?;
let mut dst = PixelBuffer::try_new(1, 1, PixelDescriptor::RGBF32_LINEAR)?;

let opts = ConvertOptions::retain_channels()
    .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardUnchecked);
let plan = ConvertPlan::build(src.encoding(), dst.encoding(), &ConversionSpec::new(opts))?;
let mut conv = plan.prepare(src.width())?;
conv.convert(&src, &mut dst)?;
# assert_eq!(dst.as_slice().row(0).len(), 12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every native format pair succeeds with zero scratch allocation and matches the
SIMD kernels bit-for-bit. Unused SIMD kernels are never referenced, so LTO
dead-strips them to `0 B`.

### Level 2: One-line startup opt-in (`accel::enable_*`)

In an application's `main()` (or benchmark setup), install SIMD accelerators
into the lock-free process registry once. Every subsequent `ConvertPlan::build`,
`ConvertPlan::select`, `OutputPlan::build`, and `.convert_to(...)` call —
including calls deep inside codecs — immediately uses them:

```rust
use zenpixels_convert::accel;

// Turn on every SIMD kernel + multi-step fused SIMD pipeline:
accel::enable_all();
assert!(accel::enabled().is_all());

// Or install only what your app uses (unused families dead-strip to 0 B):
accel::enable_layout();  // garb byte swizzle, alpha add/drop, gray, int premul, U8↔U16
accel::enable_srgb();    // linear-srgb U8/U16/F32 sRGB transfers + U8 matte
accel::enable_float();   // U8/U16/F16 ↔ F32 + FloatFacts-specialized quantization
accel::enable_gamut();   // 3×3 matrix + fused SDR gamut kernels
accel::enable_hdr();     // BT.709, Gamma 2.2, PQ, HLG transfers
accel::enable_oklab();   // Oklab / OklabA SIMD (no-op when `oklab` feature is off)
accel::enable_web_sdr(); // layout + srgb + float + gamut + web-SDR composed pipelines
```

All `enable_*` calls are idempotent and thread-safe (`no_std + alloc` via
`OnceBox`). Any conversion outside the installed slots seamlessly runs on the
0-bit-identical universal scalar core — you never get `NoPath` for omitting an
accelerator.

### Level 3: Compile-time `const Accelerators` helpers

When you want **zero global state**, exact dead-stripping per pipeline, or
hermetic parity testing, compose an `Accelerators` value at compile time with
the `const` presets and builders and pass it to
`ConversionSpec::with_accelerators(&...)`:

```rust
use zenpixels_convert::{
    Accelerators, ConversionSpec, ConvertOptions, ConvertPlan, PixelBuffer, PixelDescriptor,
};

// Built-in const presets:
//   Accelerators::SCALAR   — universal scalar core only (0 B SIMD)
//   Accelerators::LAYOUT   — byte swizzle, alpha, gray, int premul, U8↔U16
//   Accelerators::WEB_SDR  — layout + srgb + float + gamut + web-SDR pipelines
//   Accelerators::ALL      — every SIMD accelerator + full composed pipelines

// Compose your exact SIMD budget at compile time:
const THUMBNAIL_SIMD: Accelerators = Accelerators::new()
    .with_layout()
    .with_srgb();

// Or extend a preset / union two const sets:
const PHOTO_EDIT_SIMD: Accelerators = Accelerators::WEB_SDR
    .with_oklab()
    .union(Accelerators::new().with_hdr());

let src = PixelBuffer::from_vec(vec![64, 128, 192], 1, 1, PixelDescriptor::RGB8_SRGB)?;
let mut dst_simd = PixelBuffer::try_new(1, 1, PixelDescriptor::RGBF32_LINEAR)?;
let mut dst_scalar = PixelBuffer::try_new(1, 1, PixelDescriptor::RGBF32_LINEAR)?;

let simd_spec = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&THUMBNAIL_SIMD);
let scalar_spec = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&Accelerators::SCALAR);

// Shortcut when you want full SIMD on one spec without touching global state:
let _full_fast = ConversionSpec::fast(ConvertOptions::retain_channels());
let _photo = ConversionSpec::new(ConvertOptions::retain_channels())
    .with_accelerators(&PHOTO_EDIT_SIMD);

ConvertPlan::build(src.encoding(), dst_simd.encoding(), &simd_spec)?
    .prepare(src.width())?
    .convert(&src, &mut dst_simd)?;
ConvertPlan::build(src.encoding(), dst_scalar.encoding(), &scalar_spec)?
    .prepare(src.width())?
    .convert(&src, &mut dst_scalar)?;

// 0-bit identical output across SIMD and universal scalar:
assert_eq!(dst_simd.as_slice().row(0), dst_scalar.as_slice().row(0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

**How `const Accelerators` keeps compilation and binaries light:** each slot in
`Accelerators` is an `Option<fn(...)>`. When a slot stays `None` across your
crate graph, LLVM never emits or links the SIMD dispatch functions behind it.
You can also register any `const` set process-wide with
`accel::enable(THUMBNAIL_SIMD)`.

**What `WEB_SDR` covers:** `Accelerators::WEB_SDR` bundles layout, sRGB,
web-SDR float (`U8 ↔ F32` + `F32` premul), SDR gamut, and composed SDR
pipelines (`WEB_SDR.has_float()` is `true`). `U16`/`F16` float SIMD, BT.709,
Gamma 2.2, PQ, HLG, and Oklab SIMD all stay dead-stripped (and still execute
via the 0-bit-identical universal scalar core). Add `.with_float()` if you also
want `U16`/`F16 ↔ F32` SIMD, or inspect exact per-preset coverage in the
[conversion kernel browser](kernels.html).

**Hermetic override:** passing `.with_accelerators(&...)` completely bypasses
the global registry — so `&Accelerators::SCALAR` always runs universal scalar
even if another library or test called `accel::enable_all()` or enabled `fast`.

### Level 4: Default-off `fast` Cargo feature (top-level binaries only)

If you have a top-level binary and don't want to touch `main()`, enable the
`fast` Cargo feature in your binary's `Cargo.toml`:

```toml
[dependencies]
zenpixels-convert = { version = "0.4.0", features = ["fast"] }
```

This makes every spec without an explicit `.with_accelerators(...)` default to
`Accelerators::ALL`. **Library crates should never enable `fast`**, because
Cargo features unify globally across the entire dependency graph and would
force every downstream Wasm or CLI user to compile and link all SIMD kernels.

## Measured binary size & compile-budget trade-offs

Measured live on `benchmarks/converter-size/` (`bin - base`, stripped, fat LTO,
`codegen-units = 1`, `panic = "abort"`, host `r7900x` Ryzen 9 7900X Zen 4 —
**not measured on Zen 5**, `run-heavy` peak RSS `0.89–1.36 GiB`). Each binary
runs a dynamic `ConvertPlan::build` + `try_convert_row` over `black_box`
descriptors.

### Lean Cargo features (`default-features = false, features = ["rgb"]`)

| Target | Default scalar | `+ layout` | `+ web_sdr` | `+ enable_all` |
|---|---:|---:|---:|---:|
| `x86_64 -O3` | **+159 KB** | +278 KB | +408 KB | +494 KB |
| `x86_64 -Os` | **+121 KB** | +188 KB | +287 KB | +447 KB |
| `wasm32 -Oz +simd128` | **+87 KB** | +141 KB | +207 KB | +244 KB |

### Default Cargo features (`+oklab`, `+hdr-transfers`; `+std` on `wasm32`)

| Target | Default scalar | `+ layout` | `+ web_sdr` | `+ enable_all` |
|---|---:|---:|---:|---:|
| `x86_64 -O3` | **+386 KB** | +505 KB | +637 KB | +855 KB |
| `x86_64 -Os` | **+343 KB** | +409 KB | +508 KB | +1,019 KB |
| `wasm32 -Oz +simd128` | **+159 KB** | +215 KB | +285 KB | +372 KB |

**Takeaways from the table:**

1. **Two independent dials:** choosing `default-features = false` trims
   `~227 KB` on `x86_64 -O3` (`~72 KB` on `wasm32 -Oz`) by omitting Oklab and
   HDR math; choosing `SCALAR` / `LAYOUT` / `WEB_SDR` trims another
   `~157–469 KB` by omitting unused SIMD kernels.
2. **Fast SIMD throughput when you want it:** with `enable_all()` on 4K UHD
   (`3840 × 2160`, `v3` AVX2+FMA tier, Zen 4 `r7900x`, not measured on Zen 5),
   measured throughput reaches **39.7 GiB/s** (`RGBA8→RGB8`), **19.0 GiB/s**
   (`RGB8 sRGB→RGBF32 linear`), **14.8 GiB/s** (`Gamma 2.2→linear`), **13.5–14.5
   GiB/s** (`Oklab`), and **4.15 GiB/s** (`HLG BT.2100-3` display OOTF).
3. **Compile-time win:** opting into `default-features = false` (or `WEB_SDR` /
   `SCALAR` in tests and Wasm builds) skips monomorphizing unused SIMD
   transcendental tiers (`AVX2`, `AVX-512`, `Wasm128`) during release LTO.

## Cargo features: pay only for the color spaces you use

| Feature | Default | What it adds (and what happens when off) |
|---|---|---|
| `oklab` | **on** | Oklab / OklabA and soft gamut compression; off → returns `FeatureRequired { feature: "oklab", .. }` |
| `hdr-transfers` | **on** | PQ/HLG transfers and HLG display rendering; off → returns `FeatureRequired { feature: "hdr-transfers", .. }` |
| `perceptual` | off | `Oklch`, `Lch`, `Jzazbz`, `Jzczhz`, `Hsl`, `Hsv`; off → variants compiled out (or `FeatureRequired` if core unified on) |
| `bt2446a` | off | Built-in BT.2446A HDR→SDR tone mapping; off → explicit HDR→SDR returns `HdrFeatureRequired` |
| `icc-db` | off | Compressed CICP→ICC recipe table; off → retains compact exports and refuses unsupported synthesis |
| `cms-moxcms` / `cms-lcms` | off | Pluggable ICC profile backends (must still pass `.with_cms(...)` explicitly) |
| `avx512` | off | Wider 16-lane x86-64 AVX-512 row-kernel tiers |
| `precise-portable` | off | Single-rounding software FMA on non-fusing Wasm engines (default Wasm uses compact `BackendDependent` math) |
| `fast` | off | Binaries only — defaults unregistered specs to `Accelerators::ALL` |

Minimal SDR web / Wasm dependency in `Cargo.toml`:

```toml
# Smallest Wasm/embedded build with full SDR sRGB + gamut + layout support:
zenpixels-convert = { version = "0.4.0", default-features = false, features = ["std"] }
```

## Extra speed at zero code-size cost: `FloatFacts`

When you already know an `f32` buffer has no `NaN`, `±inf`, negatives, or
values above `1.0` — for instance right after decoding `u8` pixels to linear
`f32` or measuring once with `FloatFacts::measure(&slice)?` — pass those facts
into `ConversionSpec::with_float_facts(...)`. The SIMD kernels skip per-pixel
NaN/range guards while keeping bit-exact output on clean data:

```rust
use zenpixels::sample::FloatFacts;
use zenpixels_convert::{
    Accelerators, ConversionSpec, ConvertOptions, ConvertPlan, DepthPolicy, PixelBuffer,
    PixelDescriptor,
};

let mut lin = PixelBuffer::try_new(16, 1, PixelDescriptor::RGBF32_LINEAR)?;
let facts = FloatFacts::measure(&lin.as_slice())?;
lin.set_float_facts(facts)?;

let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
let spec = ConversionSpec::new(opts)
    .with_accelerators(&Accelerators::WEB_SDR)
    .with_float_facts(facts);
let plan = ConvertPlan::build(
    lin.encoding(),
    PixelDescriptor::RGB8_SRGB.into(),
    &spec,
)?;
assert!(plan.input_float_facts().contains(FloatFacts::NO_NAN | FloatFacts::AT_MOST_ONE));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Even lighter? Reach for `garb` or `linear-srgb` directly

If you don't need full format negotiation, ICC/CICP metadata, or multi-step
planning, the two lower-level crates in the stack keep builds even smaller:

- **`garb`** (`garb::bytes`): raw byte swizzle, alpha add/drop, and strided
  rows with zero color math and zero `std` dependency.
- **`linear-srgb`** (`linear_srgb::default`): direct SIMD transfer curves
  (`sRGB`, gamma, optional `transfer` feature for `BT.709`/`PQ`/`HLG`) with
  out-of-place `*_slice_into` functions that avoid copy+modify overhead.

[Pixel types guide](guide.md) ·
[Kernel browser](kernels.html) ·
[Transfer functions](transfers.html) ·
[Primaries, HDR & CICP/ICC](color-hdr.html) ·
[Plain-English ELI5](eli5.md) ·
[Explore one pixel](index.html) ·
[Converter README](https://github.com/imazen/zenpixels/blob/main/zenpixels-convert/README.md)
