# Format Negotiation Enhancement Plan

Based on calibration findings in [CALIBRATION.md](CALIBRATION.md).

The current cost model works well for the common case (Mitchell/Robidoux
resize of SDR content) but has three documented gaps where it can't
distinguish cases that differ by 2+ perceptual buckets. This plan
addresses those gaps through provenance augmentation, operation-aware
negotiation, and optimization of zenresize's working format selection.

---

## Problem Summary

The cost model scores format candidates on two axes: **effort** (compute
cost) and **loss** (information destroyed). The score depends on intent
(Fastest, LinearLight, Blend, Perceptual) but not on the specific
operation being performed or the resize filter being used.

Measured gaps:

| Gap | Model Says | Reality | Root Cause |
|-----|-----------|---------|-----------|
| Lanczos u8 sRGB | LowLoss (120) | High (33 ΔE) | Filter ringing clips at u8 |
| Blur in sRGB | LowLoss (120) | Lossless (0.005 ΔE) | Neighbors cancel gamma error |
| i16 linear + Mitchell | Lossless (5) | Moderate (9.7 ΔE) | Integer clips filter overshoot |

All three gaps share a root cause: the model is per-format, but actual
loss depends on how the format interacts with the operation.

---

## Phase 1: OperationHint on ConvertIntent

**Goal:** Let the caller tell negotiate() what operation will be performed,
so suitability can vary per-operation.

### 1a. Add OperationHint enum

```rust
/// Hint about the operation that will be performed in the negotiated format.
/// Affects suitability scoring — sharp resize kernels need float formats
/// more than blur or bilinear resize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperationHint {
    /// No specific operation — use default suitability.
    #[default]
    Unknown,
    /// Resize with a gentle filter (bilinear, Robidoux, Mitchell).
    /// Moderate overshoot. Integer formats acceptable.
    ResizeGentle,
    /// Resize with a sharp filter (CatmullRom, Lanczos, LanczosSharp).
    /// Severe overshoot. Float formats strongly preferred.
    ResizeSharp,
    /// Blur or gaussian — neighbors only, gamma error negligible.
    Blur,
    /// Unsharp mask — detail extraction cancels gamma bias.
    Sharpen,
    /// Alpha compositing — needs premultiplied alpha.
    Composite,
    /// Passthrough (crop, rotate, flip) — no pixel math.
    Passthrough,
}
```

### 1b. Extend suitability functions

`linear_light_suitability` becomes `linear_light_suitability(target, hint)`:

```rust
fn linear_light_suitability(target: PixelDescriptor, hint: OperationHint) -> u16 {
    match hint {
        // Blur/sharpen: gamma error negligible for neighbor operations.
        OperationHint::Blur | OperationHint::Sharpen => {
            if target.transfer == TransferFunction::Linear {
                match target.channel_type {
                    ChannelType::F32 => 0,
                    ChannelType::F16 | ChannelType::U16 => 2,
                    ChannelType::U8 => 10,
                    _ => 15,
                }
            } else {
                5  // Gamma error negligible for neighbors
            }
        }

        // Sharp resize: clamping loss dominates for integer formats.
        OperationHint::ResizeSharp => {
            if target.transfer == TransferFunction::Linear {
                match target.channel_type {
                    ChannelType::F32 => 0,
                    ChannelType::F16 => 5,   // Preserves overshoot
                    // Integer clips overshoot — same loss as clamping
                    ChannelType::U16 => 200,
                    ChannelType::U8 => 200,
                    _ => 250,
                }
            } else {
                // Gamma + clamping combined
                match target.channel_type {
                    // All non-linear formats clip overshoot AND have gamma darkening
                    _ => 500,
                }
            }
        }

        // Gentle resize: current behavior (gamma darkening dominates).
        OperationHint::ResizeGentle | OperationHint::Unknown => {
            // Existing logic: linear formats by quantization, non-linear = 120
            if target.transfer == TransferFunction::Linear {
                match target.channel_type {
                    ChannelType::F32 => 0,
                    ChannelType::F16 => 5,
                    ChannelType::U16 => 5,
                    ChannelType::U8 => 40,
                    _ => 50,
                }
            } else {
                120
            }
        }

        // Passthrough: no pixel math, format doesn't matter.
        OperationHint::Passthrough => 0,

        // Composite: needs premul, handled by alpha_cost + blend suitability.
        OperationHint::Composite => {
            if target.transfer == TransferFunction::Linear {
                0
            } else {
                15
            }
        }
    }
}
```

### 1c. Thread hint through negotiate API

Add `hint: OperationHint` to `negotiate()` and `best_match()`. Keep
backward-compatible defaults:

```rust
// Existing API unchanged (hint defaults to Unknown):
pub fn best_match(source, supported, intent) -> Option<PixelDescriptor>

// New API with operation hint:
pub fn best_match_for(source, supported, intent, hint) -> Option<PixelDescriptor>

// Full API:
pub fn negotiate(source, provenance, options, intent, hint) -> Option<PixelDescriptor>
```

### 1d. zenresize integration

zenresize knows the filter at resize time. Map Filter → OperationHint:

```rust
impl Filter {
    pub fn operation_hint(&self) -> OperationHint {
        match self {
            // Gentle: low overshoot, [0,1] clamping costs < 10 ΔE
            Filter::Robidoux | Filter::RobidouxFast
            | Filter::Mitchell | Filter::MitchellFast
            | Filter::CubicBSpline | Filter::Hermite => OperationHint::ResizeGentle,

            // Sharp: high overshoot, [0,1] clamping costs 20+ ΔE
            Filter::CatmullRom | Filter::CatmullRomFast
            | Filter::Lanczos | Filter::LanczosSharp | Filter::Lanczos2 | Filter::Lanczos2Sharp
            | Filter::Ginseng | Filter::GinsengSharp
            | Filter::Jinc => OperationHint::ResizeSharp,

            // Default: treat as gentle (conservative)
            _ => OperationHint::ResizeGentle,
        }
    }
}
```

### Phase 1 validation

Update perceptual_loss.rs scenarios to pass `hint` and verify:
- Lanczos + u8 sRGB: model now says High (500), measured High (33 ΔE) ✓
- Blur + sRGB: model now says Lossless (5), measured Lossless (0.005 ΔE) ✓
- Mitchell + i16 linear: model now says Lossless (5), measured Moderate (9.7) — still a gap
  for gentle filters with integer linear (see Phase 3)

---

## Phase 2: Provenance Augmentation

**Goal:** Track data characteristics that affect downstream format choices.

### 2a. Content characteristics on Provenance

```rust
pub struct Provenance {
    pub origin_depth: ChannelType,
    pub origin_primaries: ColorPrimaries,
    // New fields:
    /// Whether the data has been through an operation that produces
    /// out-of-range values (negative lobes from resize, HDR overshoot).
    /// When true, float formats should be preferred to avoid clamping loss.
    pub has_overshoot: bool,
    /// Whether the content has sharp edges (text, line art, screenshots).
    /// Sharp edges amplify filter ringing — penalizes integer formats more.
    pub edge_heavy: bool,
}
```

### 2b. Automatic overshoot tracking

After resize with a sharp filter, mark `has_overshoot = true`:

```rust
// In zenresize, after resize completes:
if filter.operation_hint() == OperationHint::ResizeSharp
    && config.output_format.is_float()
{
    provenance.has_overshoot = true;
}
```

After clamping (u8 output, sRGB encode), clear it:

```rust
// After quantizing to u8 or encoding to sRGB:
provenance.has_overshoot = false; // Already clipped
```

### 2c. Use in cost model

```rust
fn linear_light_suitability(target: PixelDescriptor, hint: OperationHint,
                            provenance: &Provenance) -> u16 {
    let base = /* ... existing logic from Phase 1 ... */;

    // Penalize integer formats when upstream already has overshoot
    if provenance.has_overshoot && !target.channel_type.is_float() {
        base.saturating_add(150) // Clip loss from prior overshoot
    } else {
        base
    }
}
```

This handles the pipeline case: decode → f32 sharp resize (has overshoot)
→ second operation needs to pick format. If the second operation would
convert to i16, it would clip the overshoot from the resize. With
provenance, the model knows to stay in f32.

### 2d. Edge detection hint

For content-aware negotiation, the decoder or caller can set `edge_heavy`
based on image analysis (e.g., high-frequency energy from DCT coefficients,
text detection, screenshot detection). This is optional — the model
works without it but can make better choices with it.

```rust
if provenance.edge_heavy {
    // Treat as ResizeSharp even for gentle filters — edges amplify ringing
    hint = hint.upgrade_for_edges();
}
```

---

## Phase 3: Negotiation Refinements

### 3a. Format capability flags

Not all formats can represent overshoot. Make this queryable:

```rust
impl PixelDescriptor {
    /// Whether this format can represent values outside [0, 1].
    /// f32 and f16 can; u8, u16, i16 cannot.
    pub fn preserves_overshoot(&self) -> bool {
        matches!(self.channel_type, ChannelType::F32 | ChannelType::F16)
    }
}
```

### 3b. ideal_format with operation hint

`ideal_format` currently returns f32 Linear for LinearLight intent.
With operation hints, it can be smarter:

```rust
pub fn ideal_format(source: PixelDescriptor, intent: ConvertIntent,
                    hint: OperationHint) -> PixelDescriptor {
    match intent {
        ConvertIntent::LinearLight => match hint {
            OperationHint::ResizeSharp => {
                // Must use float to preserve overshoot
                PixelDescriptor::RGBF32_LINEAR
            }
            OperationHint::Blur | OperationHint::Sharpen => {
                // sRGB is fine — gamma error negligible for neighbors.
                // If source is u8 sRGB, stay in u8 sRGB (zero conversion cost).
                if source.channel_type == ChannelType::U8
                    && source.transfer == TransferFunction::Srgb {
                    source // Identity — no conversion needed
                } else {
                    PixelDescriptor::RGBF32_LINEAR
                }
            }
            OperationHint::Passthrough => source, // No conversion
            _ => PixelDescriptor::RGBF32_LINEAR,  // Default: safe choice
        },
        // ... other intents unchanged
    }
}
```

### 3c. Multi-operation pipeline negotiation

When a pipeline has multiple operations (decode → resize → sharpen → encode),
each operation has different format preferences. The negotiation should
find the format that minimizes total loss across the chain, avoiding
unnecessary conversions.

**Approach:** Each operation votes for its preferred format via
`ideal_format(source, intent, hint)`. The pipeline picks the format
that satisfies the most-constrained operation (usually resize), then
checks that downstream operations are compatible.

```rust
/// Find a format that works for a chain of operations.
pub fn negotiate_pipeline(
    source: PixelDescriptor,
    provenance: Provenance,
    operations: &[(ConvertIntent, OperationHint)],
    output_options: &[FormatOption],
) -> Option<PixelDescriptor> {
    // 1. Find the most-constrained operation
    let critical = operations.iter()
        .max_by_key(|(intent, hint)| {
            let ideal = ideal_format(source, *intent, *hint);
            // Higher cost = more constrained
            conversion_cost(source, ideal).loss
        });

    // 2. Negotiate for the critical operation's ideal
    // 3. Verify downstream compatibility
    // 4. If incompatible, negotiate with full chain awareness
    todo!()
}
```

This is the long-term goal. For now, the single-operation negotiate()
with OperationHint covers 90% of real usage.

---

## Phase 4: zenresize Optimization

### 4a. Working format selection with negotiation

Currently zenresize picks its working format based on I/O format + flags.
With zenpixels negotiation, it can make better choices:

```rust
// Current: hardcoded path selection
let path = if is_u8 && !linear { U16Srgb }
           else if is_u8 && linear { U16Linear }
           else { F32 };

// Proposed: use zenpixels negotiation
let hint = filter.operation_hint();
let ideal = zenpixels::ideal_format(input_desc, ConvertIntent::LinearLight, hint);
let path = match ideal.channel_type {
    ChannelType::F32 => F32,
    ChannelType::U16 if ideal.transfer == TransferFunction::Srgb => U16Srgb,
    ChannelType::U16 => U16Linear,
    _ => F32, // Fallback
};
```

For ResizeSharp, this would automatically pick f32 even for u8 sRGB input,
avoiding the u16 path that clips overshoot.

### 4b. Selective clamping in zenresize

The i16 and u8 paths clamp intermediate values during H and V filtering.
For gentle filters, this is fine (< 10 ΔE loss). For sharp filters,
we could:

1. **Widen the i12 range.** Currently [0, 4095]. If we use [−2048, 6143]
   (still fits i16), overshoot is preserved through the pipeline and only
   clipped at final u8 output. This reduces clamping loss from Moderate to
   the u8-output-only clip.

2. **Add a signed i16 linear path.** Use the full i16 range [−32768, 32767]
   with a bias. Preserves overshoot through the pipeline.

3. **Always use f32 for sharp filters.** Simplest, slightly slower.
   The f32 path is ~1.8x slower than i16 for resize, but quality is
   strictly better. For Lanczos, the quality gap is 3+ buckets.

**Recommendation:** Option 3 (always f32 for sharp filters) is simplest
and the quality gap justifies the speed cost. Option 1 (wider i12) is
a good middle ground if speed matters — it requires only changing the
clamping constants in the SIMD kernels.

### 4c. StreamingResize format reporting

`StreamingResize::working_format()` already reports the path. Extend to
also report the negotiated format and any clamping warnings:

```rust
pub struct FormatReport {
    pub working_format: WorkingFormat,
    pub preserves_overshoot: bool,
    pub estimated_clamping_loss: u16, // From cost model
}
```

This lets callers (zenimage's StreamingGraphEngine) make informed decisions
about whether to request f32 I/O.

---

## Phase 5: Validation

### 5a. Extended test scenarios

Add to perceptual_loss.rs:
- All 4 zenresize paths (u8 sRGB gamma, u8 sRGB linear, u8 f32, f32 native)
  × all 4 test filters × 2 content types (edges, gradients) = 32 scenarios
- Pipeline tests: resize → sharpen → encode with format chosen by negotiation
- Provenance round-trip tests: sharp resize → second operation → verify
  overshoot tracking affects format choice

### 5b. Correlation target

Current Spearman rho = 0.89 with 7 calibration pairs. After Phase 1,
target rho > 0.92 with 15+ pairs spanning all operation hints.

### 5c. Regression tests

For each model change, verify:
1. No existing scenario moves by more than 1 bucket
2. All bucket mismatches documented in CALIBRATION.md are resolved
3. Spearman rho does not decrease

---

## Implementation Order

| Phase | Scope | Effort | Impact |
|-------|-------|--------|--------|
| 1a–1c | OperationHint enum + suitability | Small | Fixes 2 of 3 gaps |
| 1d | zenresize Filter → OperationHint | Small | Enables auto-detection |
| 2a–2b | Provenance has_overshoot | Small | Fixes pipeline format propagation |
| 4b.3 | f32 for sharp filters in zenresize | Small | Fixes quality for Lanczos/CatmullRom |
| 3a–3b | ideal_format with hint | Medium | Smarter format selection |
| 2c–2d | Content-aware provenance | Medium | Edge-heavy content handling |
| 3c | Pipeline negotiation | Large | Multi-op optimization |
| 4a | Negotiation-driven path selection | Medium | Unifies zenresize + zenpixels |
| 4b.1 | Wider i12 range | Medium | Speed + quality tradeoff |
| 5a–5c | Validation | Medium | Confidence in changes |

Start with Phases 1a–1d and 4b.3 — they're small, independent, and
fix the most impactful gaps.

---

## Non-Goals

- **Content-adaptive filter selection.** Choosing Mitchell vs Lanczos
  based on image content is zenresize's job, not zenpixels'. The cost
  model only advises on format given a filter choice.

- **Per-pixel format negotiation.** The model works per-image, not
  per-region. A future tiled architecture could negotiate per-tile,
  but that's out of scope.

- **Exact ΔE prediction.** The model predicts loss buckets (5 levels),
  not exact ΔE values. Bucket-level accuracy with 1-bucket tolerance
  is the target — not continuous ΔE estimation.
