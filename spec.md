# zenpixels spec

Pixel interchange types and conversion for Rust image codecs. Two crates:
`zenpixels` (core types) and `zenpixels-convert` (conversion, negotiation, CMS).

## Design principles

- One way to do things. Minimal API surface.
- No silent lossy operations. Explicit policies for alpha removal, depth
  reduction, gamut mapping.
- PixelFormat encodes byte layout. Color interpretation lives on PixelDescriptor.
- Buffers hold pixels and metadata. Conversion is a separate concern.
- Compile-time type safety via phantom-typed buffers, with zero-cost type
  erasure for dynamic dispatch.

## PixelFormat

Flat `#[repr(u8)]` enum. Each variant represents a distinct byte-level layout:
channel count, component depth, transfer function, and memory order.

Gamut-neutral naming. `Rgb8` means "3 channels, u8, gamma-encoded." It does
NOT imply sRGB gamut — the same bytes could be P3, BT.2020, or AdobeRGB.
Which gamut is on the descriptor. Which gamma curve (sRGB vs BT.709) is also
on the descriptor.

Variants encode what changes the conversion math at the byte level:

```
Rgb8, Rgba8, Rgb16, Rgba16, RgbF32, RgbaF32
Gray8, Gray16, GrayF32, GrayA8, GrayA16, GrayAF32
Bgra8, Rgbx8, Bgrx8
OklabF32, OklabaF32
```

Oklab has its own variants because it's a different color model — the three
channels are L, a, b (not R, G, B). Conversion math is fundamentally different.

Methods derive everything from the variant: `channels()`, `bytes_per_pixel()`,
`has_alpha()`, `is_grayscale()`, `component_type()`, `color_model()`,
`byte_order()`.

### What doesn't get a variant

Gamut differences. P3 at 8-bit sRGB transfer is `Rgb8` with
`primaries: DisplayP3` on the descriptor. Not a separate format variant.
Same bytes, different color interpretation.

BT.601/709 YCbCr matrix differences. Those are planar metadata, not
interleaved format differences.

### Adding formats

New variant when the byte layout or conversion math changes. PQ transfer
has different math than sRGB — that might warrant `RgbPq16` if the transfer
function isn't already a descriptor field. But P3 at sRGB transfer is just
`Rgb8` with different primaries.

## PixelDescriptor

Format + color interpretation metadata. This is the full description of what
pixel values mean.

```rust
pub struct PixelDescriptor {
    pub format: PixelFormat,
    pub transfer: TransferFunction,
    pub alpha: Option<AlphaMode>,
    pub primaries: ColorPrimaries,
    pub signal_range: SignalRange,
}
```

Named constants for common combinations: `RGB8_SRGB`, `RGBAF32_LINEAR`,
`GRAY8_SRGB`, `BGRA8_SRGB`, etc. Transfer-agnostic variants (`RGB8`,
`RGBA8`) use `TransferFunction::Unknown` for codec negotiation where the
caller specifies transfer later.

Builder methods: `with_transfer()`, `with_primaries()`, `with_alpha()`,
`with_signal_range()`.

Query methods forward to format: `channels()`, `bytes_per_pixel()`,
`has_alpha()`, `is_linear()`, `is_grayscale()`.

Layout methods: `aligned_stride()`, `simd_aligned_stride()`,
`layout_compatible()` (same bpp + channel count, safe for reinterpret).

### TransferFunction

```
Linear, Srgb, Bt709, Pq, Hlg, Unknown
```

Methods: `from_cicp()`, `reference_white_nits()`.

Extension trait in convert crate: `linearize(f32) -> f32`,
`delinearize(f32) -> f32`.

### ColorPrimaries

```
Bt709, Bt2020, DisplayP3, Unknown
```

CICP code point values. `contains()` expresses gamut hierarchy
(BT.2020 contains P3 contains BT.709).

Extension trait in convert crate: `to_xyz_matrix()`, `from_xyz_matrix()`.

### SignalRange

```
Full, Narrow
```

Full = 0–255 for u8. Narrow = 16–235 luma, 16–240 chroma (broadcast).

### AlphaMode

```
Undefined, Straight, Premultiplied, Opaque
```

`Undefined` = padding byte (Rgbx, Bgrx). `Opaque` = alpha channel present
but all values are max. `Straight` = unassociated. `Premultiplied` = associated.

## Pixel trait

```rust
pub trait Pixel: bytemuck::Pod {
    const DESCRIPTOR: PixelDescriptor;
}
```

`bytemuck::Pod` as supertrait enables safe `cast_slice` for typed row access
everywhere. No feature gates, no extra bounds at call sites.

Concrete types: `Rgbx`, `Bgrx`, `GrayAlpha8`, `GrayAlpha16`, `GrayAlphaF32`.
All `#[repr(C)]` + `Pod`.

External type impls (feature-gated): `Rgb<u8>`, `Rgba<u8>`, `Gray<u8>`, etc.
from the `rgb` crate behind `rgb` feature.

## Buffer types

Three types, all phantom-typed with `P = ()` default for type erasure:

### PixelSlice<'a, P = ()>

Borrowed immutable 2D view. Carries `PixelDescriptor` + optional
`Arc<ColorContext>`.

- `row(y) -> &[u8]` — pixel bytes for row y
- `row_with_stride(y) -> &[u8]` — full stride including padding
- `sub_rows(y, count)` — zero-copy row range
- `crop_view(x, y, w, h)` — zero-copy region
- `erase() -> PixelSlice<'a>` — drop type tag
- `try_typed::<Q>() -> Option<PixelSlice<'a, Q>>` — recover type

When `P: Pixel`: `row_pixels(y) -> &[P]` via bytemuck.

### PixelSliceMut<'a, P = ()>

Mutable borrowed 2D view. Same interface plus `row_mut()`,
`sub_rows_mut()`. Specialized in-place ops for 4bpp types (Rgbx/Bgrx swap,
alpha upgrade/strip/matte).

### PixelBuffer<P = ()>

Owned 2D buffer with SIMD alignment and optional margins.

- `try_new(w, h, descriptor)` — tight stride
- `try_new_simd_aligned(w, h, descriptor)` — aligned rows
- `from_vec(data, w, h, descriptor)` — wrap existing allocation
- `from_pixels(pixels, w, h)` — from typed vec (P: Pixel)
- `as_slice() -> PixelSlice<P>` — borrow immutable
- `as_slice_mut() -> PixelSliceMut<P>` — borrow mutable
- `into_vec() -> Vec<u8>` — recover allocation for pool reuse
- `erase()` / `try_typed::<Q>()` — type erasure/recovery

### BufferError

```
AlignmentViolation, InsufficientData, StrideTooSmall,
StrideNotPixelAligned, InvalidDimensions, IncompatibleDescriptor,
AllocationFailed
```

## Color metadata

### ColorContext

Travels with pixel data via `Option<Arc<ColorContext>>` on buffer types.

```rust
pub struct ColorContext {
    pub icc: Option<Arc<[u8]>>,
    pub cicp: Option<Cicp>,
}
```

`None` on a buffer = color fully described by `PixelDescriptor` (named space).
`Some(ctx)` = additional color info, typically from a decoded file.

For named profiles (sRGB, P3, BT.2020), the descriptor is sufficient and no
ColorContext is needed. For custom ICC profiles, the raw bytes travel with the
data so the CMS can build transforms and the encoder can re-embed them.

### ColorOrigin

Tracks what the source file said, preserved for re-encoding and round-trip
conversion. Not yet implemented — see finalize_for_output below.

```rust
pub struct ColorOrigin {
    pub icc: Option<Arc<[u8]>>,
    pub cicp: Option<Cicp>,
    pub provenance: ColorProvenance,
}

pub enum ColorProvenance {
    Icc,
    Cicp,
    GamaChrm,
    Assumed,
}
```

Origin is immutable once set. It records how the color was described, not what
the pixels currently are. The encoder uses `PixelDescriptor` for the current
state and can consult `ColorOrigin` for provenance decisions.

### Cicp

ITU-T H.273 color description. Four fields:

```rust
pub struct Cicp {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
}
```

Named constants: `SRGB`, `BT2100_PQ`, `BT2100_HLG`, `DISPLAY_P3`.

### NamedProfile

```
Srgb, DisplayP3, Bt2020, Bt2020Pq, Bt2020Hlg, AdobeRgb, LinearSrgb
```

## Planar types (feature-gated)

Multi-plane image support for YCbCr, Oklab planes, gain maps, and alpha planes.

### PlaneLayout

```rust
pub enum PlaneLayout {
    Interleaved { channels: u8 },
    Planar {
        planes: Vec<PlaneDescriptor>,
        relationship: PlaneRelationship,
    },
}
```

Factory methods: `ycbcr_420()`, `ycbcr_422()`, `ycbcr_444()`, `rgb()`,
`rgba()`, `oklab()`, `oklab_alpha()`, `gray()`.

### PlaneDescriptor

Per-plane metadata: semantic, channel type, subsampling factors.

```rust
pub struct PlaneDescriptor {
    pub semantic: PlaneSemantic,
    pub channel_type: ChannelType,
    pub h_subsample: u8,
    pub v_subsample: u8,
}
```

### PlaneSemantic

```
Luma, ChromaCb, ChromaCr,
Red, Green, Blue,
Alpha, Depth, GainMap,
Gray,
OklabL, OklabA, OklabB
```

Oklab planes allow perceptual-space processing without interleaving. An image
can be stored as three planes (L, a, b) at different resolutions — the
lightness plane at full resolution, chrominance planes subsampled. This
mirrors YCbCr 4:2:0 but in a perceptually uniform space.

### PlaneRelationship

```
Independent,
YCbCr { matrix: YuvMatrix },
Oklab,
GainMap,
```

### Subsampling

```
S444, S422, S420, S411
```

### YuvMatrix

```
Identity, Bt601, Bt709, Bt2020
```

CICP matrix_coefficients mapping. `rgb_to_y_coeffs()` returns luma weights.

### MultiPlaneImage

Container for multi-plane data: `Vec<PixelBuffer>` + `PlaneLayout` +
optional `Arc<ColorContext>`.

## Conversion architecture (zenpixels-convert)

### Direct kernels first, hub as fallback

Every commonly-used format pair should have a direct row kernel. Direct
kernels avoid the precision loss and overhead of routing through an
intermediary format. The hub (linear sRGB f32) exists only for format pairs
that don't have a direct kernel — it's a correctness safety net, not the
default path.

Direct kernels handle:
- Byte swizzle (BGRA/RGBA swap — SIMD vpshufb/vqtbl1q, zero math)
- Depth conversion (u8/u16 — SIMD multiply/shift, lossless roundtrip)
- Alpha add/strip (append 255, drop channel — no color math)
- Transfer function (sRGB/linear — SIMD LUT, exact curve)
- Channel replication (Gray -> RGB — no color math)
- Combined operations (sRGB u8 -> linear f32 in one pass)

The hub path goes: source -> linear sRGB f32 -> destination. Per-pixel,
scalar. Used for rare pairs like cross-gamut conversions without a dedicated
kernel, or Oklab round-trips where f32 precision is sufficient.

Adding a direct kernel for a format pair is purely a performance optimization.
The hub produces correct results for any pair — just slower.

### Row kernels

SIMD-optimized per-row conversion functions. Registered in a kernel table
keyed by (source format, destination format). `RowConverter` looks up the
direct kernel first; if none exists, falls back to the hub.

`ConvertPlan` pre-computes the conversion strategy once, then applies it to
every row without per-row dispatch overhead.

### ConvertPlan / RowConverter

```rust
let converter = RowConverter::new(from_desc, to_desc)?;
for y in 0..height {
    converter.convert_row(src_row, dst_row, width);
}
```

Pre-computed plan avoids per-row dispatch overhead for batch processing.

### Extension traits

- `TransferFunctionExt` — `linearize()`, `delinearize()` on TransferFunction
- `ColorPrimariesExt` — `to_xyz_matrix()`, `from_xyz_matrix()` on ColorPrimaries
- `PixelBufferConvertExt` — convenience methods on buffer types (feature-gated)

## Conversion policies

### ConvertOptions

```rust
pub struct ConvertOptions {
    pub gray_expand: GrayExpand,
    pub alpha_policy: AlphaPolicy,
    pub depth_policy: DepthPolicy,
    pub luma: Option<LumaCoefficients>,
}
```

No defaults that silently destroy data. If you want to strip alpha, you say
how: discard (only if opaque), discard unchecked, composite onto background,
or forbid. If you want to reduce depth, you say how: round, truncate, or
forbid.

### ConvertError

Errors are specific about what went wrong: `AlphaNotOpaque` (tried to discard
non-opaque alpha with `DiscardIfOpaque`), `DepthReductionForbidden` (policy
says no), `UnsupportedTransfer` (no conversion path between transfer
functions).

## Format negotiation

Intent-driven format selection for codec integration.

### ConvertIntent

```
Fastest,       // Minimize computation (for encoding)
LinearLight,   // Linear light operations (resize, blur)
Blend,         // Compositing (f32 linear premultiplied)
Perceptual,    // Color adjustments (sRGB for SDR)
```

Intent weights the two-axis cost model differently. `Fastest` penalizes
effort 4x over loss. `LinearLight` penalizes loss 4x over effort.

### ConversionCost

Two axes: `effort` (computational work) and `loss` (precision destroyed).
Both u16. Added with saturation.

### Provenance

Tracks origin depth and primaries to detect lossless round-trips. Knowing
pixels originated as u8 means u8 -> f32 -> u8 has zero loss despite the
apparent depth reduction. Without this, the cost model incorrectly penalizes
the round-trip.

### Negotiation functions

- `best_match(source, supported, intent)` — simple: pick best from list
- `negotiate(source, provenance, options, intent)` — full: explicit
  provenance + consumer costs
- `ideal_format(source, intent)` — recommend optimal working format
  (unconstrained by codec capabilities)
- `conversion_cost(from, to)` — two-axis cost between any pair

## Gamut conversion

Pre-computed matrices for BT.709, BT.2020, Display P3. All derived from
CIE 1931 xy chromaticities at D65. Stored as `[[f64; 3]; 3]`.

- `conversion_matrix(from, to)` — lookup direct matrix between named primaries
- `apply_matrix_row_f32()` — apply to row of RGB pixels
- `apply_matrix_row_rgba_f32()` — apply to row of RGBA pixels (alpha unchanged)

## Oklab

Fixed-matrix transform defined relative to D65-adapted linear sRGB.
Linear RGB -> LMS (Hunt-Pointer-Estevez) -> cube root -> Oklab.

`rgb_to_lms_matrix(primaries)` returns the appropriate matrix for the
source gamut. Non-sRGB sources go through their own LMS matrix — no
intermediate sRGB conversion needed.

## HDR

### Metadata types

- `ContentLightLevel` — MaxCLL, MaxFALL in nits
- `MasteringDisplay` — primaries, white point, luminance range
- `HdrMetadata` — bundles transfer + CLL + mastering display

### Tone mapping

- `reinhard_tonemap(v)` — v / (1 + v)
- `reinhard_inverse(v)` — v / (1 - v)
- `exposure_tonemap(v, exposure)` — exposure-adjusted Reinhard

## Codec format registry (feature-gated)

Static tables mapping codecs to their supported formats:

```rust
pub struct CodecFormats {
    pub name: &'static str,
    pub decode_outputs: &'static [FormatEntry],
    pub encode_inputs: &'static [FormatEntry],
    pub icc_decode: bool,
    pub icc_encode: bool,
    pub cicp: bool,
}
```

`FormatEntry` includes `effective_bits` (AVIF 10-bit in u16 container = 10,
not 16) and `can_overshoot` (JPEG f32 IDCT ringing exceeds [0,1]).

## CMS integration (compile-time feature)

Color management is a compile-time feature on the convert crate. When enabled,
it brings in a specific CMS implementation (moxcms, lcms2, etc.) and provides
ICC profile transforms.

```rust
// feature = "cms-moxcms" or "cms-lcms2"
```

The convert crate defines the CMS trait:

```rust
pub trait ColorManagement {
    type Error;
    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error>;

    fn identify_profile(&self, icc: &[u8]) -> Option<Cicp>;
}

pub trait RowTransform {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32);
}
```

When the CMS feature is off, ICC profile transforms are unavailable. Named
profile conversions (sRGB, P3, BT.2020) use hardcoded matrices and don't
need a CMS.

### ICC-to-CICP equivalence

`identify_profile()` checks whether an ICC profile matches a known CICP
combination. Two-tier: hash table of known ICC byte sequences for instant
lookup, then semantic comparison (parse matrix + TRC, compare against known
values within tolerance).

## Atomic finalize_for_output

The conversion to an output format and the metadata generation for the encoder
must be a single atomic operation. This prevents the most common color
management bug: pixel values that don't match the embedded color metadata.

```rust
pub fn finalize_for_output<C: ColorManagement>(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: &C,
) -> Result<EncodeReady, ConvertError>
```

`finalize_for_output` does three things atomically:

1. Determines the current pixel color state from `PixelDescriptor` + optional
   ICC profile on `ColorContext`.
2. Converts pixels to the target profile's space. For named profiles, uses
   hardcoded matrices. For custom ICC profiles, uses the CMS to build a
   transform.
3. Bundles the converted pixels with matching metadata (`EncodeReady`).

### OutputProfile

```rust
pub enum OutputProfile {
    SameAsOrigin,
    Named(Cicp),
    Icc(Arc<[u8]>),
}
```

`SameAsOrigin` re-encodes with the original ICC/CICP from the source file.
`Named` uses a well-known profile. `Icc` uses specific ICC profile bytes.

### EncodeReady

```rust
pub struct EncodeReady {
    pixels: PixelBuffer,
    metadata: OutputMetadata,
}
```

The only way to get an `EncodeReady` is through `finalize_for_output`, which
guarantees the pixels and metadata match. `into_parts()` exists for power
users who need to destructure, but the default path keeps them coupled.

```rust
impl EncodeReady {
    pub fn pixels(&self) -> PixelSlice<'_>;
    pub fn metadata(&self) -> &OutputMetadata;
    pub fn into_parts(self) -> (PixelBuffer, OutputMetadata);
}
```

### OutputMetadata

What the encoder should embed:

```rust
pub struct OutputMetadata {
    pub icc: Option<Arc<[u8]>>,
    pub cicp: Option<Cicp>,
    pub hdr: Option<HdrMetadata>,
}
```

## Testing: correctness verification and visual regression

### palette as reference implementation

Every color conversion we implement must be verified against the `palette`
crate (v0.7+). palette has a large test suite, broad adoption, and correct
implementations for sRGB, linear RGB, Oklab, XYZ, Lab, and gamut conversions.

For each conversion path in zenpixels-convert, tests should:

1. Generate a set of test values spanning the format's range (including edge
   cases: 0, 1, 255, midpoints, near-black, near-white, saturated primaries,
   out-of-gamut values where applicable).
2. Convert using our implementation.
3. Convert the same values using palette's equivalent path.
4. Compare results within a defined tolerance per conversion type.

Tolerances differ by operation:
- **Lossless operations** (byte swizzle, alpha add/strip, depth expand):
  exact match, zero tolerance.
- **Transfer function** (sRGB linearize/delinearize): max 1 ULP at f32,
  exact at u8 (LUT-based implementations should match palette's curve).
- **Oklab round-trip** (sRGB -> linear -> Oklab -> linear -> sRGB): <=1 at
  u8 after round-trip. f32 intermediate values should match palette within
  1e-6 relative error.
- **Gamut mapping** (P3 -> sRGB, BT.2020 -> sRGB): <=1 at u8 after
  round-trip through gamma. f32 matrix results should match palette within
  1e-5 relative error.
- **Depth conversion** (u8 -> u16 -> u8): exact round-trip. u8 -> f32 -> u8:
  exact round-trip.

Tests live in `zenpixels-convert` behind a `dev-dependency` on palette. They
don't run in CI on every commit — they're verification tests run when
conversion kernels change.

```rust
#[cfg(test)]
mod palette_comparison {
    use palette::{LinSrgb, Oklab, Srgb, IntoColor};

    #[test]
    fn oklab_matches_palette() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let srgb = Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
                    let palette_oklab: Oklab = srgb.into_color();

                    let our_lin = srgb_to_linear(r, g, b);
                    let our_oklab = lin_srgb_to_oklab(our_lin);

                    assert_approx_eq(our_oklab.l, palette_oklab.l, 1e-5);
                    assert_approx_eq(our_oklab.a, palette_oklab.a, 1e-5);
                    assert_approx_eq(our_oklab.b, palette_oklab.b, 1e-5);
                }
            }
        }
    }
}
```

### zensim-regress for visual regression

For image-level conversion tests (full pipeline: decode -> convert -> encode),
use `zensim-regress` to track perceptual quality. This catches regressions
that per-pixel tolerance checks miss — banding, color shifts visible at image
scale, SIMD remainder-path bugs that only affect certain image widths.

Test images should cover:
- Smooth gradients (exposes banding from depth/transfer rounding)
- Saturated colors near gamut boundary (exposes clamping bugs)
- High-frequency detail (exposes SIMD vs scalar path mismatches)
- Various dimensions (prime widths, non-SIMD-aligned widths, 1px, large)

`zensim-regress` provides architecture-aware checksums, so AVX2 and NEON
paths can have independent expected values where ULP differences are
unavoidable.

Regression threshold: a zensim score change of more than 0.1 on any test
image between commits is a failing test. The threshold is tight because
format conversions should be deterministic — score changes indicate a bug,
not an acceptable tradeoff.

### Tolerance authority: Rust code, not checksum files

Checksum files (`.checksums` / TOML) record accepted hashes and diff
evidence. They answer "which outputs have we seen and accepted?" with
`=` (human-verified) and `~` (auto-accepted within tolerance) lines.
That's bookkeeping — fine to auto-update.

Tolerances are defined in Rust test code. They answer "what differences
are acceptable?" That's a correctness decision and must be reviewable as
a code change.

When a test result is within tolerance, zensim-regress auto-accepts and
writes `~` checksum lines with diff evidence. No human intervention needed.

When a result exceeds tolerance, the test fails. The output shows:
- What exceeded (score, max delta, category, rounding bias)
- The current tolerance in effect
- The Rust code that would accept the new result, IF a human decides
  the change is acceptable

zensim-regress does not recommend accepting. It presents the facts and
the code. The human decides.

```
FAIL: max_delta exceeded (actual: [2,1,1], limit: 1)
  zensim=99.87, category=RoundingError, bias=balanced

  Current tolerance:
    RegressionTolerance::off_by_one().with_max_delta(1)

  To accept this result, change to:
    RegressionTolerance::off_by_one().with_max_delta(2)
```

### Unusual tolerance warnings

Tolerances that are suspiciously loose get a warning on stdout when the
test runs — even if the test passes. This catches copy-paste mistakes
and gradual tolerance creep.

Warn when:
- `max_delta` >= 8 (more than 3% of u8 range)
- `min_similarity` < 90.0 (zensim score — perceptually very different)
- `max_pixels_different` >= 0.5 (half the image differs)

The warning doesn't fail the test. It just says:

```
WARNING: unusually loose tolerance for oklab_roundtrip_srgb8:
  max_delta=8 (>= 8 is unusual for format conversion)
```

---

## Gaps: what exists in code but is missing from this spec

### 1. PixelDescriptor named constants

The code has 80+ named descriptor constants (`RGB8_SRGB`, `RGBAF32_LINEAR`,
`BGRA8_SRGB`, etc.) plus transfer-agnostic variants (`RGB8`, `RGBA8`). The
spec should enumerate the naming convention and the full constant list, or at
minimum define the pattern so new constants are consistent.

**Suggested convention:** `{FORMAT}_{TRANSFER}` for specific combinations
(`RGB8_SRGB`, `RGBF32_LINEAR`), plain `{FORMAT}` for transfer-agnostic
(`RGB8`, `RGBA8` with `Unknown` transfer). P3/BT.2020 variants use
`{FORMAT}_{TRANSFER}_{PRIMARIES}` when the primaries differ from BT.709.

### 2. In-place typed conversions on PixelSliceMut

The code has specialized in-place operations for 4bpp types:
`Rgbx::swap_to_bgrx()`, `Rgba::matte_to_rgbx()`, `Bgrx::upgrade_to_bgra()`,
etc. These are zero-allocation, type-state-changing operations — the return
type changes to reflect the new pixel layout. The spec should document which
conversions are available and their safety guarantees (e.g., `upgrade_to_rgba`
sets padding to 255, which is only correct for opaque images).

### 3. crop_view semantics

`crop_view(x, y, w, h)` creates a zero-copy sub-region view by adjusting the
data pointer and dimensions. The spec should document:
- The stride stays the same (it's a view into the parent's rows)
- The crop must be within bounds (panics otherwise)
- The result can't be used with SIMD row kernels that assume contiguous rows

### 4. SIMD alignment guarantees

`try_new_simd_aligned()` and `simd_aligned_stride()` provide alignment for
vectorized processing. The spec should document:
- The alignment target (32 bytes for AVX2, or configurable)
- When alignment is preserved through operations (sub_rows yes, crop_view no)
- How `fill_margins()` interacts with aligned buffers for convolution

### 5. Pool-friendly buffer lifecycle

`into_vec()` recovers the backing allocation for reuse. The spec should
document the intended lifecycle: allocate once, use `into_vec()` to return
to pool, `from_vec()` to reuse. This matters for real-time pipelines where
allocation is the bottleneck.

### 6. adapt_for_encode

`adapt_for_encode()` is the main codec integration point — given pixel data
and a list of supported formats, it picks the best match and converts if
needed. Returns `Cow` (borrowed if already compatible, owned if converted).
The spec should document this as the primary codec handoff function.

### 7. Path analysis

`generate_path_matrix()`, `optimal_path()`, `matrix_stats()` analyze
conversion costs between all format pairs. Primarily a debugging and
optimization tool. The spec should mention its existence for tooling support
without over-specifying the implementation.

### 8. Oklab primaries-aware conversion

`rgb_to_lms_matrix(primaries)` returns the correct matrix for any supported
gamut, not just sRGB. A P3 image converts to Oklab through P3's own LMS
matrix — no intermediate sRGB step. The spec should document this because
it's a correctness detail that other Oklab implementations get wrong (they
hardcode the sRGB matrix and silently produce incorrect results for non-sRGB
inputs).
