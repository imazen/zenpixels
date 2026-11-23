#!/usr/bin/env python3
"""Extract the live conversion kernel catalog from zenpixels-convert/src into site/dist/kernels-data.js."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "zenpixels-convert" / "src"

STEP_METADATA = {
    "Identity": ("Universal Scalar Core", "SCALAR", "default", "scalar", "No-op pass-through when source and destination encodings already match (zero copies in OutputPlan::borrow_rows)."),
    "HlgRender": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "magemath", "BT.2100-3 display OOTF (scene-referred HLG -> display light with explicit Lw, Lb, system gamma) — distinct from scene-only OETF."),
    "SwizzleBgraRgba": ("Layout & Swizzle", "LAYOUT", "default", "garb", "4-byte R<->B channel swap via garb AVX2/NEON/WASM128 shuffle masks; avoids float promotion."),
    "SwizzleBgrRgb": ("Layout & Swizzle", "LAYOUT", "default", "garb", "3-byte RGB<->BGR swap uses 12-byte/48-byte cross-lane shuffles completely different from 4bpp RGBA."),
    "RgbToBgra": ("Layout & Swizzle", "LAYOUT", "default", "garb", "Fused 3bpp->4bpp R<->B swap + opaque alpha insertion in one RAM pass instead of AddAlpha + Swizzle."),
    "AddAlpha": ("Alpha, Premul & Matte", "LAYOUT", "default", "garb / attune", "Expands 3ch->4ch filling opaque alpha (255 / 65535 / 1.0f / f16 1.0) per sample width without touching color."),
    "DropAlpha": ("Alpha, Premul & Matte", "LAYOUT", "default", "garb / attune", "Packs 4ch->3ch discarding the 4th channel per sample width (only after AlphaPolicy authorizes removal)."),
    "MatteComposite": ("Alpha, Premul & Matte", "SRGB", "default", "linear-srgb / attune", "Linear-light Porter-Duff over onto solid background + alpha drop in one pass (avoids 3-pass F32 roundtrip for U8)."),
    "GrayToRgb": ("Gray & Luma", "LAYOUT", "default", "garb / attune", "Replicates 1ch gray -> 3ch RGB at native sample width (1B/2B/4B) using widening shuffle masks."),
    "GrayToRgba": ("Gray & Luma", "LAYOUT", "default", "garb / attune", "Replicates 1ch gray -> 4ch RGBA with opaque alpha in a single pass."),
    "RgbToGray": ("Gray & Luma", "LAYOUT", "default", "attune", "Encoded-space Y' luma collapse (BT.709/BT.601/BT.2020/P3 coefficients) with dedicated fixed-point U8 BT.709 fast path."),
    "RgbaToGray": ("Gray & Luma", "LAYOUT", "default", "attune", "Fused 4ch->1ch luma collapse + alpha drop in one pass (7.3x faster than DropAlpha + RgbToGray)."),
    "RgbaToGrayAlpha": ("Gray & Luma", "LAYOUT", "default", "attune", "4ch->2ch luma collapse preserving linear alpha lane verbatim."),
    "GrayAlphaToRgba": ("Gray & Luma", "LAYOUT", "default", "garb / attune", "2ch->4ch gray replication while keeping alpha in lane 3."),
    "GrayAlphaToRgb": ("Gray & Luma", "LAYOUT", "default", "attune", "2ch->3ch gray replication while dropping alpha."),
    "GrayToGrayAlpha": ("Gray & Luma", "LAYOUT", "default", "attune", "1ch->2ch gray expansion appending opaque alpha."),
    "GrayAlphaToGray": ("Gray & Luma", "LAYOUT", "default", "attune", "2ch->1ch alpha strip on grayscale."),
    "SrgbU8ToLinearF32": ("SDR Transfers", "SRGB", "default", "linear-srgb", "256-entry LUT decode from sRGB U8 directly to linear F32 — skips both /255 and transcendental powf."),
    "LinearF32ToSrgbU8": ("SDR Transfers", "SRGB", "default", "linear-srgb", "Fused linear F32 -> sRGB U8 polynomial + rounding (19 GiB/s) — avoids intermediate sRGB F32 buffer."),
    "NaiveU8ToF32": ("Depth & FloatFacts", "WEB_SDR", "default", "float_simd", "Exact 0-ULP U8 -> F32 normalization (v / 255.0) preserving transfer encoding."),
    "NaiveF32ToU8": ("Depth & FloatFacts", "WEB_SDR", "default", "float_simd", "Exact round-to-nearest F32 -> U8 quantization, specialized by FloatFacts to skip NaN/clamp guards."),
    "NaiveF32ToU8Truncate": ("Depth & FloatFacts", "WEB_SDR", "default", "float_simd", "Floor truncation F32 -> U8 for DepthPolicy::Truncate (distinct rounding semantics from NaiveF32ToU8)."),
    "U16ToU8": ("Depth & FloatFacts", "LAYOUT", "default", "garb", "Exact round-to-nearest U16 -> U8 ((v * 255 + 32895) >> 16) via garb integer SIMD."),
    "U16ToU8Truncate": ("Depth & FloatFacts", "LAYOUT", "default", "scalar / attune", "Fast high-byte truncation (v >> 8) for DepthPolicy::Truncate."),
    "SdrU16ToU8": ("SDR Transfers", "ALL", "default", "scalar", "Composite SDR transfer change + U16 -> U8 quantization computed in f64 to avoid double-rounding drift."),
    "U8ToU16": ("Depth & FloatFacts", "LAYOUT", "default", "attune", "Exact bit replication (v * 257 = (v << 8) | v) — lossless integer widening."),
    "U16ToF32": ("Depth & FloatFacts", "FLOAT", "default", "float_simd", "Exact 0-ULP U16 -> F32 normalization (v / 65535.0); excluded from WEB_SDR to save code size."),
    "F32ToU16": ("Depth & FloatFacts", "FLOAT", "default", "float_simd", "Exact round-to-nearest F32 -> U16 quantization with FloatFacts guard elision."),
    "F32ToU16Truncate": ("Depth & FloatFacts", "FLOAT", "default", "float_simd", "Floor truncation F32 -> U16 for DepthPolicy::Truncate."),
    "F16ToF32": ("Depth & FloatFacts", "FLOAT", "default", "f16_scalar / magetypes", "Hardware F16C / NEON-fp16 / WASM SIMD IEEE-754 binary16 unpack to F32."),
    "F32ToF16": ("Depth & FloatFacts", "FLOAT", "default", "f16_scalar / magetypes", "Hardware F16C / NEON-fp16 round-to-nearest-even F32 -> binary16 pack."),
    "PqU16ToLinearF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Direct PQ U16 -> anchored linear F32 via linear-srgb SIMD + white-anchor scaling."),
    "LinearF32ToPqU16": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Fused anchored linear F32 -> PQ U16 OETF + round-to-nearest quantization."),
    "PqF32ToLinearF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "PQ F32 EOTF with FloatFacts-guided NaN/clamp elision + alpha preservation + white-anchor scaling."),
    "LinearF32ToPqF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Anchored linear F32 -> PQ F32 OETF preserving linear alpha."),
    "HlgU16ToLinearF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Direct HLG U16 scene inverse-OETF -> linear F32."),
    "LinearF32ToHlgU16": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Fused scene linear F32 -> HLG U16 OETF + quantization."),
    "HlgF32ToLinearF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "HLG F32 scene inverse-OETF preserving alpha."),
    "LinearF32ToHlgF32": ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "linear-srgb", "Scene linear F32 -> HLG F32 OETF preserving alpha."),
    "SrgbF32ToLinearF32": ("SDR Transfers", "SRGB", "default", "linear-srgb", "Clamped [0, 1] sRGB F32 -> linear F32 SIMD polynomial preserving alpha."),
    "LinearF32ToSrgbF32": ("SDR Transfers", "SRGB", "default", "linear-srgb", "Clamped [0, 1] linear F32 -> sRGB F32 SIMD polynomial preserving alpha."),
    "SrgbF32ToLinearF32Extended": ("SDR Transfers", "SRGB", "default", "linear-srgb", "Sign-preserving unclipped sRGB EOTF for GamutPolicy::PreserveExtended (negative/HDR floats survive)."),
    "LinearF32ToSrgbF32Extended": ("SDR Transfers", "SRGB", "default", "linear-srgb", "Sign-preserving unclipped sRGB OETF for GamutPolicy::PreserveExtended."),
    "Bt709F32ToLinearF32": ("SDR Transfers", "HDR", "default", "magemath", "Out-of-place magemath Powf BT.709 EOTF (0.081 threshold, 1/0.45 exponent) with FloatFacts check elision."),
    "LinearF32ToBt709F32": ("SDR Transfers", "HDR", "default", "magemath", "Out-of-place magemath Powf BT.709 OETF (0.018 threshold, 0.45 exponent) with in-register alpha mask."),
    "Gamma22F32ToLinearF32": ("SDR Transfers", "HDR", "default", "magemath", "Adobe RGB 1998 clamped gamma EOTF (exact exponent 563/256 = 2.19921875) via const_exponent magemath Powf."),
    "LinearF32ToGamma22F32": ("SDR Transfers", "HDR", "default", "magemath", "Adobe RGB 1998 clamped gamma OETF (256/563) via const_exponent magemath Powf."),
    "Gamma22F32ToLinearF32Extended": ("SDR Transfers", "HDR", "default", "magemath", "Sign-reflected extended Gamma 2.2 EOTF across HDR range (1e-16..65536) in pure SIMD (0% libm)."),
    "LinearF32ToGamma22F32Extended": ("SDR Transfers", "HDR", "default", "magemath", "Sign-reflected extended Gamma 2.2 OETF across HDR range in pure SIMD."),
    "StraightToPremul": ("Alpha, Premul & Matte", "LAYOUT / WEB_SDR", "default", "float_simd / attune", "Multiplies RGB by alpha (integer fixed-point U8/U16 in LAYOUT; F32 SIMD in WEB_SDR/FLOAT)."),
    "PremulToStraight": ("Alpha, Premul & Matte", "LAYOUT / WEB_SDR", "default", "float_simd / attune", "Divides RGB by alpha with exact alpha==0 -> 0 guard (integer lookup/div vs F32 SIMD)."),
    "LinearRgbToOklab": ("Oklab & Soft Compress", "OKLAB", "oklab", "magemath / magetypes", "3ch RGB->LMS + division-free magemath::Cbrtf<Midp> + LMS->Oklab using SIMD deinterleave_3ch."),
    "OklabToLinearRgb": ("Oklab & Soft Compress", "OKLAB", "oklab", "attune", "3ch Oklab -> LMS^3 -> linear RGB directly in the destination RGB primaries (zero BT.709 bottleneck)."),
    "LinearRgbaToOklaba": ("Oklab & Soft Compress", "OKLAB", "oklab", "magemath / magetypes", "4ch RGBA->OklabA using deinterleave_4ch/interleave_4ch so linear alpha rides untouched."),
    "OklabaToLinearRgba": ("Oklab & Soft Compress", "OKLAB", "oklab", "attune", "4ch OklabA -> linear RGBA directly in destination primaries, alpha untouched."),
    "GamutMatrixRgbF32": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "attune", "3ch linear-light 3x3 FMA gamut matrix (e.g. Display P3 <-> BT.709 <-> BT.2020 <-> Adobe RGB)."),
    "GamutMatrixRgbaF32": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "attune", "4ch linear-light 3x3 FMA gamut matrix leaving alpha untouched."),
    "ToneMap": ("HDR Transfers & Tone Map", "ALL", "bt2446a / hdr-experimental", "magemath", "Pluggable two-phase HDR->SDR tone-map step (built-in BT.2446A or custom ToneMapper) on linear BT.2020."),
    "SoftCompressOklch": ("Oklab & Soft Compress", "OKLAB", "oklab", "magemath / attune", "Hue-preserving OKLch rational-knee chroma compression for GamutPolicy::SoftCompress."),
}

FUSED_METADATA = {
    "FusedSrgbU8GamutRgb": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "fast_gamut", "Peepholes [SrgbU8ToLinearF32, GamutMatrixRgbF32, LinearF32ToSrgbU8] into ONE register pass (3x less memory traffic)."),
    "FusedSrgbU8GamutRgba": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "fast_gamut", "Same 1-pass fused sRGB U8 gamut peephole for 4ch RGBA with alpha passthrough."),
    "FusedSrgbU16GamutRgb": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "fast_gamut", "1-pass sRGB U16 gamut conversion via 65K decode LUT + FMA matrix + polynomial encode."),
    "FusedSrgbU8ToLinearF32Rgb": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "fast_gamut", "Peepholes [SrgbU8ToLinearF32, GamutMatrixRgbF32] into one pass (e.g. P3 U8 -> BT.709 Linear F32)."),
    "FusedLinearF32ToSrgbU8Rgb": ("Gamut & Fused TF+Matrix", "GAMUT", "default", "fast_gamut", "Peepholes [GamutMatrixRgbF32, LinearF32ToSrgbU8] into one pass (single rounding, zero intermediate buffer)."),
}

PERCEPTUAL_METADATA = {
    "CartesianToPolar": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "(L, a, b[, A]) -> (L, C, h deg[, A]) for Oklab->Oklch, Lab->Lch, Jzazbz->Jzczhz."),
    "PolarToCartesian": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "(L, C, h deg[, A]) -> (L, a, b[, A]) polar inverse."),
    "LinearRgbToJzazbz": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Safdar et al. (2017) HDR perceptual Jzazbz from anchored linear RGB."),
    "JzazbzToLinearRgb": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Inverse Jzazbz -> anchored linear RGB in target primaries."),
    "PqRgbToJzazbz": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Direct ST 2084 PQ -> Jzazbz (0..10,000 nits HDR-to-HDR without SDR anchor)."),
    "JzazbzToPqRgb": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Direct Jzazbz -> ST 2084 PQ."),
    "LinearRgbToLch": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Linear RGB (D65) -> Bradford D50 -> CIE 1976 L*C*h deg."),
    "LchToLinearRgb": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "CIE 1976 L*C*h deg -> Bradford D65 -> linear RGB."),
    "RgbToHsl": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Encoded RGB -> cylindrical HSL (H in [0, 360), S, L in [0, 1])."),
    "HslToRgb": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Cylindrical HSL -> encoded RGB."),
    "RgbToHsv": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Encoded RGB -> cylindrical HSV (H in [0, 360), S, V in [0, 1])."),
    "HsvToRgb": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Cylindrical HSV -> encoded RGB."),
    "DecodeIntToF32": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Unpacks normalized U8/U16 perceptual/polar codes to canonical F32 ranges (turns -> degrees)."),
    "EncodeF32ToInt": ("Perceptual & Polar", "ALL", "perceptual", "scalar", "Packs canonical F32 perceptual/polar values to normalized U8/U16."),
}


def classify_simd_fn(rel: str, fn: str, attr: str):
    if rel.startswith("orient/"):
        bpp = re.search(r"transpose(\d+)", fn)
        w = f"{bpp.group(1)}-byte pixel" if bpp else "tiled"
        return (
            "Orientation / Transpose",
            "Always (fast-transpose)",
            "fast-transpose",
            "archmage",
            f"8x8 / 16x16 in-register SIMD matrix transpose specialized for {w} stride (1/2/3/4/6/8/12/16 Bpp each need distinct shuffle graphs).",
        )
    if rel == "scan.rs":
        return (
            "Scan Predicates",
            "Always (scan)",
            "default",
            "magetypes",
            "Read-only SIMD scan predicate (opaque / grayscale / U16 bit-replication / fused single-pass) for lossless codec negotiation.",
        )
    if rel == "f16_scalar.rs":
        return (
            "Depth & FloatFacts",
            "FLOAT",
            "default",
            "magetypes",
            "Hardware F16C (AVX2) / NEON-fp16 / WASM SIMD binary16 <-> F32 converter; excluded from WEB_SDR.",
        )
    if rel == "float_simd.rs":
        if "premul" in fn:
            return ("Alpha, Premul & Matte", "WEB_SDR", "default", "magetypes", "SIMD F32 RGBA premultiply / unpremultiply with FloatFacts branch elision.")
        if "u16" in fn:
            return ("Depth & FloatFacts", "FLOAT", "default", "magetypes", "0-ULP / FloatFacts-specialized U16 <-> F32 SIMD kernel (round vs truncate, untrusted vs clean).")
        return ("Depth & FloatFacts", "WEB_SDR", "default", "magetypes", "0-ULP / FloatFacts-specialized U8 <-> F32 SIMD kernel (round vs truncate, untrusted vs clean).")
    if rel == "fast_gamut.rs":
        if "u16" in fn:
            return ("Gamut & Fused TF+Matrix", "GAMUT", "default", "archmage", "Fused U16 sRGB gamut kernel (decode LUT/poly + 3x3 FMA matrix + encode LUT/poly in one pass).")
        return ("Gamut & Fused TF+Matrix", "WEB_SDR", "default", "archmage", "Fused U8/F32 sRGB gamut kernel (linearize + 3x3 FMA matrix + encode in registers, 1 RAM pass instead of 3).")
    if rel == "hdr/gamut_compress.rs":
        return ("Oklab & Soft Compress", "OKLAB", "oklab", "magemath", "SIMD OKLch soft chroma compression strip/plane kernel for GamutPolicy::SoftCompress.")
    if rel == "hdr/bt2446a.rs":
        return ("HDR Transfers & Tone Map", "ALL", "bt2446a", "magemath", "SIMD BT.2446 Method A HDR->SDR tone-mapping kernel using const-configured magemath Logf/Expf/Powf.")
    if rel == "hdr/measure.rs":
        return ("HDR Transfers & Tone Map", "Always (hdr-experimental)", "hdr-experimental", "magemath", "SIMD MaxCLL / MaxFALL / log2 luminance histogram measurement scan.")
    if rel == "hlg_render.rs":
        return ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "magemath", "SIMD BT.2100-3 HLG display OOTF (Expf/Logf Midp + packed binary64 Powf Highp, 10-14x faster than f64 scalar).")
    if rel == "gamma_math.rs":
        return ("SDR Transfers", "Always (cms-moxcms)", "cms-moxcms", "magemath", "SIMD Highp signed power kernel for ICC/CMS parametric gamma curves.")
    # convert_kernels.rs
    if "oklab" in fn:
        return ("Oklab & Soft Compress", "OKLAB", "oklab", "magemath / magetypes", "SIMD Oklab/OklabA kernel using division-free magemath::Cbrtf<Midp> + 3ch/4ch deinterleave.")
    if "gamut" in fn:
        return ("Gamut & Fused TF+Matrix", "WEB_SDR", "default", "magetypes", "SIMD 3x3 FMA gamut matrix on linear F32 RGB/RGBA rows.")
    if "premul" in fn or "unpremul" in fn:
        slot = "WEB_SDR" if ("u8" in fn or "f32" in fn) else "LAYOUT"
        return ("Alpha, Premul & Matte", slot, "default", "attune", "Type- and channel-specialized premul/unpremul SIMD loop (RGBA vs GrayAlpha, U8 vs U16 vs F32).")
    if "matte" in fn:
        return ("Alpha, Premul & Matte", "SRGB", "default", "attune", "SIMD matte split/blend helper for linear-light alpha compositing onto solid background.")
    if "gray" in fn or "luma" in fn or "widen_u16" in fn or "narrow_luma" in fn:
        return ("Gray & Luma", "LAYOUT", "default", "attune", "Split-per-width grayscale/luma SIMD loop (1B/2B/4B; BT.709 U8 fast integer weights vs generic F32).")
    if "alpha" in fn:
        return ("Alpha, Premul & Matte", "LAYOUT", "default", "attune", "Split-per-width 3ch<->4ch alpha add/drop SIMD kernel (2-byte U16/F16 vs 4-byte F32; U8 uses garb).")
    if "u8_to_u16" in fn:
        return ("Depth & FloatFacts", "LAYOUT", "default", "attune", "SIMD bit-replication U8 -> U16 widening (v * 257).")
    if "multiply_color" in fn:
        return ("HDR Transfers & Tone Map", "HDR", "hdr-transfers", "magetypes", "SIMD RGB-only luminance scale (diffuse_white / 10,000 nits) leaving alpha untouched.")
    return ("SDR Transfers", "HDR", "default", "magemath", "SIMD transfer-curve row kernel.")


def parse_tiers(attr: str) -> str:
    if "#[arcane" in attr:
        return "explicit target_feature (v3 / neon)"
    tiers = []
    if "v4x" in attr or "_v4" in attr or "v4(" in attr:
        tiers.append("v4x/AVX-512 (opt-in)")
    if "-_v3" not in attr:
        tiers.append("v3 (AVX2+FMA)")
    if "-_neon" not in attr:
        tiers.append("NEON")
    if "-_wasm128" not in attr:
        tiers.append("WASM128")
    if "-_scalar" not in attr or "dispatch" in attr:
        tiers.append("scalar")
    return ", ".join(tiers) if tiers else "scalar"


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site" / "dist" / "kernels-data.js"
    entries = []

    # 1. ConvertStep variants from convert.rs
    conv_lines = (SRC / "convert.rs").read_text().splitlines()
    in_enum = False
    for idx, line in enumerate(conv_lines, 1):
        if "pub(crate) enum ConvertStep" in line:
            in_enum = True
            continue
        if in_enum:
            if line.startswith("}"):
                break
            m = re.match(r"^\s{4}([A-Z][A-Za-z0-9]+)\b", line)
            if m:
                name = m.group(1)
                if name in ("Fused", "Perceptual"):
                    continue
                meta = STEP_METADATA.get(name, ("Other", "ALL", "default", "scalar", "Planner conversion step."))
                entries.append({
                    "id": f"step:{name}",
                    "name": name,
                    "kind": "ConvertStep",
                    "family": meta[0],
                    "accel": meta[1],
                    "feature": meta[2],
                    "backend": meta[3],
                    "tiers": "via bound SIMD slot or universal scalar",
                    "file": "zenpixels-convert/src/convert.rs",
                    "line": idx,
                    "why": meta[4],
                })

    # 2. FusedKind peepholes from convert.rs
    in_fused = False
    for idx, line in enumerate(conv_lines, 1):
        if "pub(crate) enum FusedKind" in line:
            in_fused = True
            continue
        if in_fused:
            if line.startswith("}"):
                break
            m = re.match(r"^\s{4}([A-Z][A-Za-z0-9]+)\b", line)
            if m:
                name = "Fused" + m.group(1)
                meta = FUSED_METADATA[name]
                entries.append({
                    "id": f"fused:{name}",
                    "name": name,
                    "kind": "Fused Peephole",
                    "family": meta[0],
                    "accel": meta[1],
                    "feature": meta[2],
                    "backend": meta[3],
                    "tiers": "v3 (AVX2+FMA), scalar",
                    "file": "zenpixels-convert/src/convert.rs",
                    "line": idx,
                    "why": meta[4],
                })

    # 3. PerceptualStep variants from perceptual.rs
    perc_lines = (SRC / "perceptual.rs").read_text().splitlines()
    in_perc = False
    for idx, line in enumerate(perc_lines, 1):
        if "pub(crate) enum PerceptualStep" in line:
            in_perc = True
            continue
        if in_perc:
            if line.startswith("}"):
                break
            m = re.match(r"^\s{4}([A-Z][A-Za-z0-9]+)\b", line)
            if m:
                name = m.group(1)
                meta = PERCEPTUAL_METADATA[name]
                entries.append({
                    "id": f"perc:{name}",
                    "name": name,
                    "kind": "PerceptualStep",
                    "family": meta[0],
                    "accel": meta[1],
                    "feature": meta[2],
                    "backend": meta[3],
                    "tiers": "scalar (closed-form)",
                    "file": "zenpixels-convert/src/perceptual.rs",
                    "line": idx,
                    "why": meta[4],
                })

    # 4. Real SIMD-dispatched kernel functions across zenpixels-convert/src
    simd_files = [
        "convert_kernels.rs",
        "fast_gamut.rs",
        "float_simd.rs",
        "f16_scalar.rs",
        "gamma_math.rs",
        "hdr/bt2446a.rs",
        "hdr/gamut_compress.rs",
        "hdr/measure.rs",
        "hlg_render.rs",
        "orient/mod.rs",
        "orient/pxn_x86.rs",
        "orient/pxn_neon.rs",
        "orient/rgb3_x86.rs",
        "scan.rs",
    ]
    seen_fns = set()
    for rel in simd_files:
        lines = (SRC / rel).read_text().splitlines()
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("///") or stripped.startswith("//!"):
                continue
            if not re.match(r"^#\[(attune|arcane|magetypes)\b", stripped):
                continue
            fn_name = None
            for j in range(idx + 1, min(len(lines), idx + 15)):
                m = re.search(r"\bfn\s+(\w+)", lines[j])
                if m:
                    fn_name = m.group(1)
                    break
            if not fn_name or (rel, fn_name) in seen_fns:
                continue
            seen_fns.add((rel, fn_name))
            fam, accel, feat, backend, why = classify_simd_fn(rel, fn_name, stripped)
            kind = (
                "Transpose Kernel" if rel.startswith("orient/")
                else "Scan Predicate" if rel == "scan.rs"
                else "SIMD Kernel"
            )
            entries.append({
                "id": f"simd:{rel}:{fn_name}",
                "name": fn_name,
                "kind": kind,
                "family": fam,
                "accel": accel,
                "feature": feat,
                "backend": backend,
                "tiers": parse_tiers(stripped),
                "file": f"zenpixels-convert/src/{rel}",
                "line": idx + 1,
                "why": why,
            })

    # 5. Add the universal scalar core entry itself
    entries.append({
        "id": "core:UniversalPlan",
        "name": "UniversalPlan::execute_row",
        "kind": "Universal Core",
        "family": "Universal Scalar Core",
        "accel": "SCALAR",
        "feature": "default",
        "backend": "magemath / linear-srgb / scalar",
        "tiers": "1 compact fast-FMA scalar loop (0-bit identical to native SIMD)",
        "file": "zenpixels-convert/src/universal.rs",
        "line": 2380,
        "why": "Covers EVERY valid native pixel pair in one zero-scratch loop so codecs and tiny Wasm builds link 0 B of SIMD tables unless opted in.",
    })

    # Compute summary counts
    by_kind = {}
    by_family = {}
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        by_family[e["family"]] = by_family.get(e["family"], 0) + 1

    payload = {
        "total": len(entries),
        "byKind": by_kind,
        "byFamily": by_family,
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("window.KERNEL_CATALOG = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"Wrote {out_path}: {len(entries)} entries ({by_kind})")


if __name__ == "__main__":
    main()
