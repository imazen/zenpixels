# Cost Model Calibration Report

All values measured with CIEDE2000 (Sharma, Wu, Dalal 2005) at f64 precision.
Test harness: `tests/perceptual_loss.rs` — 66 scenarios, Spearman rho = 0.89.

## Bucket Classification

| Bucket | p95 ΔE Range | Model Loss Range | Perceptual Meaning |
|--------|-------------|-----------------|-------------------|
| Lossless | < 0.5 | 0–10 | Below JND |
| NearLossless | 0.5–2.0 | 11–50 | Visible only in A/B comparison |
| LowLoss | 2.0–5.0 | 51–150 | Noticeable in isolation |
| Moderate | 5.0–15.0 | 151–400 | Clearly visible |
| High | > 15.0 | > 400 | Severe degradation |

Tests allow 1-bucket tolerance for borderline cases.

---

## 1. Depth Conversions

Quantization error when converting between channel types.

| Conversion | p95 ΔE | Bucket | Model Loss | Source |
|------------|--------|--------|-----------|--------|
| u8 sRGB → f32 → u8 sRGB (u8 origin) | 0.000 | Lossless | 0 | Provenance enables lossless |
| u8 sRGB → u16 sRGB → u8 sRGB | 0.000 | Lossless | 0 | u16 subsumes u8 |
| u16 sRGB → u8 sRGB | 0.14 | Lossless | 10 | Sub-JND in sRGB |
| f32 Linear → u8 sRGB (f32 origin) | 0.14 | Lossless | 10 | sRGB OETF perceptually uniform |
| f32 Linear → u8 sRGB (u8 origin) | 0.000 | Lossless | 0 | Provenance: round-trip |
| f32 → f16 → f32 | 0.008 | Lossless | 20 | 10 mantissa bits |
| u8 → f16 → u8 | 0.000 | Lossless | 0 | f16 > 8 bits precision |
| u16 → f16 → u16 | varies | NearLossless | 30 | 16→10 bits, moderate |
| f32 → i16 → f32 | varies | Lossless | 5 | 15-bit range, near-exact |
| u8 → f32 (naive, no gamma) | 26.0 | High | 300 | Wrong: no EOTF applied |
| u8 → f32 (EOTF) → u8 (OETF) | 0.000 | Lossless | 0 | Correct gamma handling |
| u16 → f32 → u16 | 0.000 | Lossless | 0 | f32 subsumes u16 |
| f16 → u8 | varies | Lossless | 8 | f16 > u8 precision |
| f16 → f32 → f16 | 0.000 | Lossless | 0 | f32 subsumes f16 |

**Finding:** sRGB quantization to u8 is perceptually lossless (ΔE = 0.14).
The sRGB OETF provides perceptually uniform step sizes, so 256 levels
suffice for essentially all content. The cost model's depth_loss values
correctly reflect this — u8 is not the problem; gamma darkening is.

---

## 2. Premultiplication Round-trips

Alpha premultiply → unpremultiply at various alpha values.

| Scenario | p95 ΔE | Bucket | Model Loss |
|----------|--------|--------|-----------|
| u8 premul rt α=255 | 0.000 | Lossless | 0 |
| u8 premul rt α=128 | 0.56 | NearLossless | 15 |
| u8 premul rt α=2 | large | Moderate | 400 |
| u8 premul rt α=1 | large | High | 500 |
| f32 premul rt α=0.004 | ~0 | Lossless | 0 |
| u16 premul rt α≈2/65535 | moderate | Moderate | 200 |
| u8 premul (R=200,α=100) rt | ~0.5 | NearLossless | 15 |
| f32 premul rt α=0.001 | ~0 | Lossless | 0 |

**Finding:** Premultiply round-trips are lossless in f32 at any alpha.
In u8, loss escalates rapidly below α ≈ 10 because premultiply quantizes
RGB to fewer bits. At α=1, only values 0 and 1 are representable. At
α=128, loss is sub-JND (0.56 ΔE). At α=2, loss is Moderate.

---

## 3. Gamut Conversions

Color primaries conversion with out-of-gamut clipping.

| Conversion | p95 ΔE | Bucket | Model Loss | Notes |
|------------|--------|--------|-----------|-------|
| sRGB → P3 → sRGB (in-gamut) | ~0 | Lossless | 0 | sRGB ⊂ P3, no clipping |
| P3 → sRGB (saturated colors) | 6.8 | Moderate | 80 | Saturated reds/greens clip |
| sRGB → BT.2020 → sRGB (in-gamut) | ~0 | Lossless | 0 | sRGB ⊂ BT.2020 |
| BT.2020 → sRGB (saturated) | 12.3 | Moderate | 200 | Severe clipping |
| BT.2020 → P3 (saturated) | varies | Moderate | 100 | Less clipping than →sRGB |
| P3 → BT.2020 → P3 (subset) | ~0 | Lossless | 0 | P3 ⊂ BT.2020 |
| BT.2020→sRGB (origin sRGB) | ~0 | Lossless | 5 | Provenance: data is sRGB |
| P3→sRGB (origin sRGB) | ~0 | Lossless | 5 | Provenance: data is sRGB |

**Finding:** Gamut widening is always lossless. Gamut narrowing is
only lossy if the data uses colors outside the target gamut. Provenance
tracking enables lossless gamut round-trips when the data originated
in the target gamut.

---

## 4. Transfer Function Round-trips

| Conversion | p95 ΔE | Bucket | Notes |
|------------|--------|--------|-------|
| f32 sRGB → Linear → sRGB | ~0 | Lossless | f32 preserves precision |
| u8 sRGB → f32 Linear → u8 sRGB | ~0 | Lossless | sRGB EOTF/OETF round-trip |
| f32 PQ → Linear → PQ | ~0 | Lossless | ST 2084 round-trip |
| f32 PQ → u8 sRGB (HDR→SDR) | large | High | Huge range loss |
| f32 HLG → Linear → HLG | ~0 | Lossless | STD-B67 round-trip |
| f32 BT.709 → sRGB → BT.709 | ~0 | Lossless | Curves nearly identical |

**Finding:** All transfer function round-trips are lossless in f32.
The only significant loss is cross-domain (HDR PQ → SDR sRGB) which
the cost model correctly penalizes at loss=300.

---

## 5. Resize Suitability — Gamma Darkening

The dominant finding: **gamma darkening from interpolating in sRGB
is the primary error for resize, independent of bit depth.**

### Bilinear resize, sRGB vs linear reference

| Format | Transfer | p95 ΔE | Bucket | Model Loss |
|--------|----------|--------|--------|-----------|
| u8 | sRGB | 13.7 | Moderate | 120 |
| i16 14-bit | sRGB | 13.7 | Moderate | 120 |
| u16 | sRGB | 13.7 | Moderate | 120 |
| f16 | sRGB | 13.7 | Moderate | 120 |
| f32 | sRGB | 13.7 | Moderate | 120 |
| u8 | Linear | 0.213 | Lossless | 40 |
| i16 | Linear | 0.001 | Lossless | 5 |
| f16 | Linear | 0.022 | Lossless | 5 |
| f32 | Linear | 0.000 | Lossless | 0 |
| u16 | Linear | ~0 | Lossless | 5 |

**Key insight:** All sRGB formats measure identical ΔE ≈ 13.7 for
bilinear resize. Bit depth is irrelevant — the gamma curve's
nonlinearity at midtones causes systematic darkening that dominates.
In linear space, only quantization matters, and even u8 linear
(ΔE = 0.213) is below JND.

This validates the cost model's structure: `linear_light_suitability`
checks transfer function first, then precision.

### Blur and USM — gradient neighbors cancel gamma error

| Operation | Format | p95 ΔE | Bucket | Model Loss |
|-----------|--------|--------|--------|-----------|
| Blur i16 sRGB | sRGB | 0.009 | Lossless | 5 |
| USM i16 sRGB | sRGB | 0.005 | Lossless | 5 |
| USM f32 sRGB | sRGB | 0.005 | Lossless | 5 |

**Key insight:** Blur and USM operate on neighboring pixels which differ
by ~0.003. The sRGB curve is locally linear at that scale, so gamma
error is negligible. The cost model's suitability=120 for sRGB resize
overestimates for these operations — but this is a per-format value,
not per-operation, so we accept the overestimate.

---

## 6. Filter-Dependent Clamping Loss (zenresize)

Measured with zenresize `Resizer` API — proper weight tables, normalization,
i16 largest-remainder error distribution. 200×4 → 100×2 with sharp 1-pixel
step edges.

### Pure clamping: f32 unclamped vs f32 clamped to [0,1]

| Filter | p95 ΔE | max ΔE | Bucket |
|--------|--------|--------|--------|
| Mitchell (B=1/3, C=1/3) | 9.66 | 9.88 | Moderate |
| CatmullRom (B=0, C=0.5) | 21.21 | 21.67 | High |
| Lanczos (3-lobe) | 33.48 | 33.92 | High |

### f32 linear vs u8 sRGB gamma (full format + gamma + clamp)

| Filter | p95 ΔE | max ΔE | Bucket |
|--------|--------|--------|--------|
| Mitchell | 14.47 | 14.88 | Moderate |
| CatmullRom | 22.33 | 22.72 | High |
| Lanczos | 33.44 | 33.91 | High |
| Robidoux | 14.77 | 15.01 | Moderate |

### f32 linear vs u8 sRGB linear (clamp only, no gamma darkening)

| Filter | p95 ΔE | max ΔE | Bucket |
|--------|--------|--------|--------|
| Mitchell | 9.66 | 9.88 | Moderate |
| CatmullRom | 21.21 | 21.67 | High |
| Lanczos | 33.48 | 33.92 | High |

**Key findings:**

1. **Clamping loss is filter-dependent.** Mitchell: 9.7, CatmullRom: 21.2,
   Lanczos: 33.5. The cost model assigns suitability=120 to all sRGB resize,
   which matches Mitchell/Robidoux but is 2 buckets wrong for Lanczos.

2. **u8 sRGB linear = pure clamping.** The "sRGB linear" path (sRGB→linear→resize→linear→sRGB)
   produces identical ΔE to pure clamping. The u8 quantization adds nothing;
   it's entirely the [0,1] clip on filter overshoot.

3. **Gamma darkening adds ~5 ΔE on top of clamping for Mitchell** (14.5 vs 9.7).
   For Lanczos, gamma darkening is negligible because clamping dominates (33.4 vs 33.5).

4. **Only f32 native I/O preserves overshoot.** zenresize's f32 path does
   not clamp intermediate or output values. All u8 and i16 paths clip to
   their representable range (u8=[0,255], i12=[0,4095]).

### Where zenresize paths clamp

| Path | Working Type | Clips During Filter | Clips at Output | Preserves Overshoot |
|------|-------------|--------------------|-----------------|--------------------|
| u8 sRGB gamma | i16 sRGB | Yes ([0,255]) | Yes | No |
| u8 sRGB linear | i12 linear | Yes ([0,4095]) | Yes | No |
| u8 via f32 | f32 | No | Yes (f32→u8) | During filter only |
| f32 native | f32 | No | No | Yes |

---

## 7. Perceptual Operations

Operations done in wrong color space vs correct reference.

| Operation | Format | p95 ΔE | Bucket | Model Loss |
|-----------|--------|--------|--------|-----------|
| Saturation 1.5x sRGB vs Linear | f64 | varies | NearLossless | 15 |
| Saturation 1.5x u8 sRGB vs f32 Lin | u8 | varies | NearLossless | 25 |
| USM i16 sRGB vs f32 Linear | i16 | ~0 | Lossless | 5 |
| Tonemap Reinhard vs u8 clamp | f64 | large | High | 500 |
| USM f32 sRGB vs f32 Linear | f32 | ~0 | Lossless | 5 |
| Composite f32 vs i16 premul (α=0.1) | i16 | ~0 | Lossless | 10 |

**Finding:** Saturation boost in sRGB is slightly wrong (hue rotation)
but the error is small. Tonemapping without proper operator (naive u8
clamp) is catastrophic. USM in sRGB is nearly identical to linear because
the detail extraction partially cancels gamma bias.

---

## Model Gaps

### Gap 1: Filter-dependent suitability

The cost model's `linear_light_suitability` returns 120 for all non-linear
formats regardless of which resize filter is used. Measured reality:

| Filter | Measured Bucket | Model Bucket | Gap |
|--------|----------------|-------------|-----|
| Mitchell | Moderate (14.5 ΔE) | LowLoss (120) | 1 bucket (OK) |
| Robidoux | Moderate (14.8 ΔE) | LowLoss (120) | 1 bucket (OK) |
| CatmullRom | High (22.3 ΔE) | LowLoss (120) | 2 buckets |
| Lanczos | High (33.4 ΔE) | LowLoss (120) | 2 buckets |

The model works for "gentle" filters but underestimates "sharp" filters.

### Gap 2: Integer clamping in linear space

For linear formats, the model gives i16/u16 suitability=5 (Lossless).
This is correct for bilinear but wrong for sharp filters on edges:

| Format | Bilinear ΔE | Mitchell ΔE | CatmullRom ΔE | Lanczos ΔE |
|--------|------------|------------|--------------|-----------|
| f32 Linear | 0 | 0 | 0 | 0 |
| i16 Linear | 0.001 | 9.66 | 21.21 | 33.48 |
| f16 Linear | 0.022 | ~0 | ~0 | ~0 |

i16 clips overshoot; f16 preserves it. The model doesn't distinguish.

### Gap 3: Operation-specific suitability

Blur/USM in sRGB: ΔE ≈ 0.005 (Lossless).
Resize in sRGB: ΔE ≈ 13.7 (Moderate).
Same format, same model suitability (120), 3 buckets apart.

The model is per-format. Operations that sample distant pixels (resize)
suffer from gamma nonlinearity; operations that sample neighbors (blur)
don't.
