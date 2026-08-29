//! Row-level pixel conversion kernel implementations.
//!
//! Each kernel converts one row of pixels for a single conversion step.
//! Called from the step dispatcher in the parent `convert` module.

use core::cmp::min;

use archmage::prelude::*;

use crate::policy::LumaCoefficients;
use crate::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor};

use super::ConvertStep;
use crate::TransferFunction;
use crate::f16_scalar::{
    f16_bits_to_f32, f16_bits_to_f32_slice, f32_to_f16_bits, f32_to_f16_bits_slice,
};

/// IEEE 754 half-precision encoding of 1.0: sign=0, exponent=01111 (bias 15),
/// mantissa=0000000000 → bits 0b0_01111_0000000000 = 0x3C00.
const F16_ONE_BITS: u16 = 0x3C00;

/// Scratch state reused by the HDR tone-map kernels across rows.
///
/// Owned by [`ConvertScratch`](super::ConvertScratch) so a plan/strip run
/// allocates once and every subsequent row reuses it — keeping the step
/// kernels free of per-row heap allocation (the module contract). Without
/// the `hdr-experimental` feature this is an empty placeholder so
/// [`apply_step_u8`]'s signature stays feature-independent.
#[derive(Default)]
pub(super) struct HdrKernelScratch {
    /// RGB-triple strip for the RGBA carrier paths — the SIMD curves want a
    /// contiguous `[[f32; 3]]` view, so alpha is peeled into this scratch.
    /// Grow-only; sliced to the row width on each use.
    #[cfg(feature = "hdr-experimental")]
    rgb_strip: alloc::vec::Vec<[f32; 3]>,
    /// Cached [`SoftCompress`](crate::hdr::SoftCompress) — it owns a
    /// gamut-boundary LUT whose construction runs 16 k bisection searches,
    /// so it must not be rebuilt per row. Keyed by the step params.
    #[cfg(feature = "hdr-experimental")]
    soft_compress: Option<CachedSoftCompress>,
}

#[cfg(feature = "hdr-experimental")]
struct CachedSoftCompress {
    primaries: ColorPrimaries,
    /// Bit pattern of the knee so the key comparison is exact (no float `==`).
    knee_bits: u32,
    compressor: crate::hdr::SoftCompress,
}

/// Apply a single conversion step on raw byte slices.
#[allow(clippy::too_many_arguments)] // internal step dispatcher; mirrors the plan's step tuple
pub(super) fn apply_step_u8(
    step: &ConvertStep,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    from: PixelDescriptor,
    _to: PixelDescriptor,
    // Relative-linear → PQ-absolute scale (`diffuse_white / 10000`) carried by
    // the plan. Only the PQ kernels read it; `1.0` is a no-op for all steps.
    pq_scale: f32,
    // Row-persistent scratch for the HDR tone-map kernels (unused when the
    // `hdr-experimental` feature is off).
    hdr_scratch: &mut HdrKernelScratch,
) {
    #[cfg(not(feature = "hdr-experimental"))]
    let _ = hdr_scratch;
    crate::__trace_ops::record_step(step);
    let w = width as usize;

    match step {
        ConvertStep::Identity => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }

        ConvertStep::SwizzleBgraRgba => {
            swizzle_bgra_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::AddAlpha => {
            add_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::RgbToBgra => {
            rgb_to_bgra(src, dst, w, from.channel_type());
        }

        ConvertStep::DropAlpha => {
            drop_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::MatteComposite { r, g, b } => {
            matte_composite(src, dst, w, from, *r, *g, *b);
        }

        ConvertStep::GrayToRgb => {
            gray_to_rgb(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayToRgba => {
            gray_to_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::RgbToGray { coefficients } => {
            rgb_to_gray(src, dst, w, from.channel_type(), *coefficients);
        }

        ConvertStep::RgbaToGray { coefficients } => {
            rgba_to_gray(src, dst, w, from.channel_type(), *coefficients);
        }

        ConvertStep::GrayAlphaToRgba => {
            gray_alpha_to_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayAlphaToRgb => {
            gray_alpha_to_rgb(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayToGrayAlpha => {
            gray_to_gray_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayAlphaToGray => {
            gray_alpha_to_gray(src, dst, w, from.channel_type());
        }

        ConvertStep::SrgbU8ToLinearF32 => {
            srgb_u8_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToSrgbU8 => {
            linear_f32_to_srgb_u8(src, dst, w, from.layout());
        }

        ConvertStep::NaiveU8ToF32 => {
            naive_u8_to_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::NaiveF32ToU8 => {
            naive_f32_to_u8(src, dst, w, from.layout().channels());
        }

        ConvertStep::U16ToU8 => {
            u16_to_u8(src, dst, w, from.layout().channels());
        }

        ConvertStep::U8ToU16 => {
            u8_to_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::U16ToF32 => {
            u16_to_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::F32ToU16 => {
            f32_to_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::F16ToF32 => {
            f16_to_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::F32ToF16 => {
            f32_to_f16(src, dst, w, from.layout().channels());
        }

        ConvertStep::PqU16ToLinearF32 => {
            pq_u16_to_linear_f32(src, dst, w, from.layout(), pq_scale);
        }

        ConvertStep::LinearF32ToPqU16 => {
            linear_f32_to_pq_u16(src, dst, w, from.layout(), pq_scale);
        }

        ConvertStep::PqF32ToLinearF32 => {
            pq_f32_to_linear_f32(src, dst, w, from.layout(), pq_scale);
        }

        ConvertStep::LinearF32ToPqF32 => {
            linear_f32_to_pq_f32(src, dst, w, from.layout(), pq_scale);
        }

        ConvertStep::HlgU16ToLinearF32 => {
            hlg_u16_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToHlgU16 => {
            linear_f32_to_hlg_u16(src, dst, w, from.layout());
        }

        ConvertStep::HlgF32ToLinearF32 => {
            hlg_f32_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToHlgF32 => {
            linear_f32_to_hlg_f32(src, dst, w, from.layout());
        }

        ConvertStep::SrgbF32ToLinearF32 => {
            srgb_f32_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToSrgbF32 => {
            linear_f32_to_srgb_f32(src, dst, w, from.layout());
        }

        ConvertStep::SrgbF32ToLinearF32Extended => {
            srgb_f32_to_linear_f32_extended(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToSrgbF32Extended => {
            linear_f32_to_srgb_f32_extended(src, dst, w, from.layout());
        }

        ConvertStep::Bt709F32ToLinearF32 => {
            bt709_f32_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToBt709F32 => {
            linear_f32_to_bt709_f32(src, dst, w, from.layout());
        }

        ConvertStep::Gamma22F32ToLinearF32 => {
            gamma22_f32_to_linear_f32(src, dst, w, from.layout());
        }

        ConvertStep::LinearF32ToGamma22F32 => {
            linear_f32_to_gamma22_f32(src, dst, w, from.layout());
        }

        ConvertStep::StraightToPremul => {
            straight_to_premul(src, dst, w, from.channel_type(), from.layout());
        }

        ConvertStep::PremulToStraight => {
            premul_to_straight(src, dst, w, from.channel_type(), from.layout());
        }

        ConvertStep::LinearRgbToOklab => {
            linear_rgb_to_oklab_f32(src, dst, w, from.primaries);
        }

        ConvertStep::OklabToLinearRgb => {
            oklab_to_linear_rgb_f32(src, dst, w, from.primaries);
        }

        ConvertStep::LinearRgbaToOklaba => {
            linear_rgba_to_oklaba_f32(src, dst, w, from.primaries);
        }

        ConvertStep::OklabaToLinearRgba => {
            oklaba_to_linear_rgba_f32(src, dst, w, from.primaries);
        }

        ConvertStep::GamutMatrixRgbF32(flat) => {
            gamut_matrix_rgb_f32(src, dst, w, flat);
        }

        ConvertStep::GamutMatrixRgbaF32(flat) => {
            gamut_matrix_rgba_f32(src, dst, w, flat);
        }

        // All fused TF + matrix + TF kernels share the same row-major
        // 3×3 unflatten then dispatch on `kind`.
        ConvertStep::Fused { kind, matrix: flat } => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
            match kind {
                crate::convert::FusedKind::SrgbU8GamutRgb => {
                    crate::fast_gamut::convert_u8_rgb_simd_matlut(
                        &m,
                        src,
                        dst,
                        crate::fast_gamut::srgb_lin_lut_u8(),
                        |v: f32| linear_srgb::default::linear_to_srgb_u8(v),
                    );
                }
                crate::convert::FusedKind::SrgbU8GamutRgba => {
                    crate::fast_gamut::convert_u8_rgba_simd_lut(
                        &m,
                        src,
                        dst,
                        crate::fast_gamut::srgb_lin_lut_u8(),
                        linear_srgb::default::linear_to_srgb_u8,
                    );
                }
                crate::convert::FusedKind::SrgbU16GamutRgb => {
                    let src_u16: &[u16] = bytemuck::cast_slice(src);
                    let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
                    // LUT decode (256 KB lin_lut from linear-srgb) + SIMD matrix
                    // + SIMD polynomial encode. +17% at 1080p vs linear-LUT encode,
                    // 100% exact u16 roundtrip (was ~71% with the linearly-indexed
                    // 128 KB encode LUT — see benchmarks/u16_hybrid_matrix_2026-04-23.txt).
                    crate::fast_gamut::convert_u16_rgb_simd_lutdec_polyenc(&m, src_u16, dst_u16);
                }
                crate::convert::FusedKind::SrgbU8ToLinearF32Rgb => {
                    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    crate::fast_gamut::convert_u8_to_f32_lin_simd(
                        &m,
                        src,
                        dst_f32,
                        crate::fast_gamut::srgb_lin_lut_u8(),
                    );
                }
                crate::convert::FusedKind::LinearF32ToSrgbU8Rgb => {
                    let src_f32: &[f32] = bytemuck::cast_slice(src);
                    crate::fast_gamut::convert_f32_lin_to_u8_simd(
                        &m,
                        src_f32,
                        dst,
                        crate::fast_gamut::srgb_enc_lut_u8(),
                    );
                }
            }
        }

        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A {
            source_peak_nits,
            target_peak_nits,
        } => {
            tone_map_bt2446a_kernel(
                src,
                dst,
                w,
                from,
                *source_peak_nits,
                *target_peak_nits,
                hdr_scratch,
            );
        }

        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { primaries, knee } => {
            soft_compress_oklch_kernel(src, dst, w, from, *primaries, *knee, hdr_scratch);
        }
    }
}

/// Apply the BT.2446 Method A tone curve to a row of linear-light F32 RGB(A)
/// pixels in BT.2020 primaries.
///
/// The carrier layout (RGB vs RGBA) is read from `from.layout()`. For RGBA,
/// the alpha channel is copied through verbatim. Input is scrubbed for
/// non-finite values and negatives; output is clamped to `[0, 1]` to absorb
/// f32 epsilon-level overshoot at the saturated end.
//
// `needless_range_loop` would flatten the loops to iter_mut() + enumerate
// over the strip — but each iter writes BOTH `rgb_strip[p]` and the parallel
// `dst_f32[p * channels + 3]` (alpha passthrough), so the index-based form
// is the readable shape. Same logic applies in the SoftCompress kernel
// below; suppress for the whole function rather than per-loop.
#[cfg(feature = "hdr-experimental")]
#[allow(clippy::needless_range_loop)]
fn tone_map_bt2446a_kernel(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from: PixelDescriptor,
    source_peak_nits: f32,
    target_peak_nits: f32,
    scratch: &mut HdrKernelScratch,
) {
    let curve = crate::hdr::Bt2446A::new(source_peak_nits, target_peak_nits);
    let has_alpha = from.layout().has_alpha();
    let channels = if has_alpha { 4 } else { 3 };
    let src_f32: &[f32] = bytemuck::cast_slice(src);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

    if has_alpha {
        // RGBA: extract RGB triples into the row-persistent scratch strip,
        // run the SIMD curve, write back while passing alpha through
        // verbatim. The scratch keeps the SIMD kernel's contiguous
        // `[[f32; 3]]` shape and is allocated once per plan/strip run
        // (grow-only), not per row — the module's no-per-row-allocation
        // contract. Every `[..width]` element is written before being read.
        if scratch.rgb_strip.len() < width {
            scratch.rgb_strip.resize(width, [0.0; 3]);
        }
        let rgb_strip = &mut scratch.rgb_strip[..width];
        for p in 0..width {
            let base = p * channels;
            let r = src_f32[base];
            let g = src_f32[base + 1];
            let b = src_f32[base + 2];
            // Scrub non-finite / negatives — matches the prior HdrToSdr
            // contract so consumers see clean linear-light values.
            let r = if r.is_finite() && r >= 0.0 { r } else { 0.0 };
            let g = if g.is_finite() && g >= 0.0 { g } else { 0.0 };
            let b = if b.is_finite() && b >= 0.0 { b } else { 0.0 };
            rgb_strip[p] = [r, g, b];
        }
        curve.map_strip_simd(&mut *rgb_strip);
        for p in 0..width {
            let base = p * channels;
            let [r, g, b] = rgb_strip[p];
            // Final clamp absorbs BT.2446-A's near-peak ~1e-4 overshoot
            // (matches the prior HdrToSdr final-clamp postcondition).
            dst_f32[base] = if r.is_finite() {
                r.clamp(0.0, 1.0)
            } else {
                0.0
            };
            dst_f32[base + 1] = if g.is_finite() {
                g.clamp(0.0, 1.0)
            } else {
                0.0
            };
            dst_f32[base + 2] = if b.is_finite() {
                b.clamp(0.0, 1.0)
            } else {
                0.0
            };
            // Alpha passthrough.
            dst_f32[base + 3] = src_f32[base + 3];
        }
    } else {
        // RGB: cast straight to `[[f32; 3]]` and apply in-place after
        // copying src → dst. The SIMD curve is in-place.
        let n_floats = width * 3;
        dst_f32[..n_floats].copy_from_slice(&src_f32[..n_floats]);
        let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut dst_f32[..n_floats]);
        for px in strip.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() || *c < 0.0 {
                    *c = 0.0;
                }
            }
        }
        curve.map_strip_simd(strip);
        for px in strip.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() {
                    *c = 0.0;
                } else {
                    *c = c.clamp(0.0, 1.0);
                }
            }
        }
    }
}

/// OKLch soft chroma compression on a row of linear-light F32 RGB(A) pixels
/// in `primaries`. Alpha (when present) is copied through verbatim.
//
// Same justification for `needless_range_loop` as in
// `tone_map_bt2446a_kernel`.
#[cfg(feature = "hdr-experimental")]
#[allow(clippy::needless_range_loop)]
fn soft_compress_oklch_kernel(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from: PixelDescriptor,
    primaries: ColorPrimaries,
    knee: f32,
    scratch: &mut HdrKernelScratch,
) {
    // Resolve the compressor once per plan/strip run and cache it in the
    // row-persistent scratch: `SoftCompress` owns a gamut-boundary LUT whose
    // construction runs 16 k bisection searches, so rebuilding it per row
    // dominated this kernel (and heap-allocated per row, violating the
    // module contract). The cache key is the step params that shape it.
    let stale = match &scratch.soft_compress {
        Some(c) => c.primaries != primaries || c.knee_bits != knee.to_bits(),
        None => true,
    };
    if stale {
        let m1 = crate::oklab::rgb_to_lms_matrix(primaries)
            .expect("target primaries have a defined LMS matrix");
        let m1_inv = crate::oklab::lms_to_rgb_matrix(primaries)
            .expect("target primaries have a defined inverse LMS matrix");
        scratch.soft_compress = Some(CachedSoftCompress {
            primaries,
            knee_bits: knee.to_bits(),
            compressor: crate::hdr::SoftCompress::from_matrices(&m1, &m1_inv, knee),
        });
    }
    let compressor = &scratch
        .soft_compress
        .as_ref()
        .expect("populated above when stale or missing")
        .compressor;

    let has_alpha = from.layout().has_alpha();
    let channels = if has_alpha { 4 } else { 3 };
    let src_f32: &[f32] = bytemuck::cast_slice(src);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

    if has_alpha {
        // Row-persistent RGB strip; see `tone_map_bt2446a_kernel`.
        if scratch.rgb_strip.len() < width {
            scratch.rgb_strip.resize(width, [0.0; 3]);
        }
        let rgb_strip = &mut scratch.rgb_strip[..width];
        for p in 0..width {
            let base = p * channels;
            rgb_strip[p] = [src_f32[base], src_f32[base + 1], src_f32[base + 2]];
        }
        compressor.apply_strip(&mut *rgb_strip);
        for p in 0..width {
            let base = p * channels;
            let [r, g, b] = rgb_strip[p];
            dst_f32[base] = r;
            dst_f32[base + 1] = g;
            dst_f32[base + 2] = b;
            dst_f32[base + 3] = src_f32[base + 3];
        }
    } else {
        let n_floats = width * 3;
        dst_f32[..n_floats].copy_from_slice(&src_f32[..n_floats]);
        let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut dst_f32[..n_floats]);
        compressor.apply_strip(strip);
    }
}

// ---------------------------------------------------------------------------
// Kernel implementations
// ---------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Shuffle / replicate kernels — split-per-type #[autoversion] pattern.
//
// Kernels that don't insert a constant (opaque alpha) share one function
// across U16 and F16 because the byte-level op is identical — we just move
// 2-byte samples around. Kernels that DO insert a constant get one function
// per type because the constant value differs (65535 for U16, F16_ONE_BITS
// for F16, 1.0 for F32).
// ----------------------------------------------------------------------------

#[autoversion]
fn swizzle_bgra_rgba_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[u16; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = s[3];
    }
}

#[autoversion]
fn swizzle_bgra_rgba_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[f32; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [f32; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = s[3];
    }
}

/// BGRA ↔ RGBA swizzle.
fn swizzle_bgra_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            let n = width * 4;
            garb::bytes::rgba_to_bgra(&src[..n], &mut dst[..n]).expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => {
            let n = width * 8;
            swizzle_bgra_rgba_2bytes(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        ChannelType::F32 => {
            let n = width * 16;
            swizzle_bgra_rgba_f32(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        _ => {}
    }
}

// -- rgb_to_bgra (inserts opaque alpha: differs per type) --------------------

#[autoversion]
fn rgb_to_bgra_u16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let s: &[u16; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 65535;
    }
}

#[autoversion]
fn rgb_to_bgra_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let s: &[f32; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [f32; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 1.0;
    }
}

#[autoversion]
fn rgb_to_bgra_f16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let s: &[u16; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = F16_ONE_BITS;
    }
}

/// Fused RGB → BGRA: byte-swap R↔B and add opaque alpha in one SIMD pass.
fn rgb_to_bgra(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::rgb_to_bgra(&src[..width * 3], &mut dst[..width * 4])
                .expect("pre-validated row size");
        }
        ChannelType::U16 => rgb_to_bgra_u16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        ChannelType::F32 => rgb_to_bgra_f32(
            bytemuck::cast_slice(&src[..width * 12]),
            bytemuck::cast_slice_mut(&mut dst[..width * 16]),
            width,
        ),
        ChannelType::F16 => rgb_to_bgra_f16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        _ => {}
    }
}

// -- add_alpha (inserts opaque alpha: differs per type) ----------------------

#[autoversion]
fn add_alpha_u16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let s: &[u16; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 65535;
    }
}

#[autoversion]
fn add_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let s: &[f32; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [f32; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 1.0;
    }
}

#[autoversion]
fn add_alpha_f16(src: &[u16], dst: &mut [u16], width: usize) {
    // Hoist the const into a local so LLVM treats it as loop-invariant
    // (and ideally lifts a SIMD broadcast out of the inner loop). Without
    // this the codegen devolves to per-pixel `mov ebx, 15360` +
    // `vpinsrw` (2-3 cycles) whereas the U16 equivalent uses
    // `vpcmpeqd` + `vpblendw` (1 cycle). See the T1 add_alpha F16 2×
    // anomaly noted in benchmarks/t1_layout_2026-04-23_baseline.meta.
    let opaque: u16 = 0x3C00;
    for i in 0..width {
        let s: &[u16; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = opaque;
    }
}

/// Add opaque alpha channel (3ch → 4ch).
fn add_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::rgb_to_rgba(&src[..width * 3], &mut dst[..width * 4])
                .expect("pre-validated row size");
        }
        ChannelType::U16 => add_alpha_u16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        ChannelType::F32 => add_alpha_f32(
            bytemuck::cast_slice(&src[..width * 12]),
            bytemuck::cast_slice_mut(&mut dst[..width * 16]),
            width,
        ),
        ChannelType::F16 => add_alpha_f16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        _ => {}
    }
}

// -- drop_alpha (pure shuffle — U16 and F16 share one kernel) ----------------

#[autoversion]
fn drop_alpha_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let s: &[u16; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let d: &mut [u16; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

#[autoversion]
fn drop_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let s: &[f32; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let d: &mut [f32; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

/// Drop alpha channel (4ch → 3ch).
fn drop_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::rgba_to_rgb(&src[..width * 4], &mut dst[..width * 3])
                .expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => drop_alpha_2bytes(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 6]),
            width,
        ),
        ChannelType::F32 => drop_alpha_f32(
            bytemuck::cast_slice(&src[..width * 16]),
            bytemuck::cast_slice_mut(&mut dst[..width * 12]),
            width,
        ),
        _ => {}
    }
}

// ----------------------------------------------------------------------------
// Transfer-function trait for const-generic matte_composite dispatch.
//
// Each impl is a unit struct whose static methods inline into the caller
// at monomorphization. Calling `T::eotf(v)` is indistinguishable from
// calling the underlying TF function directly once LLVM inlines — so the
// generic `matte_composite_*_rgba<T>` body, once monomorphized per TF,
// becomes a pure f32 arithmetic loop that autovectorizes cleanly.
// ----------------------------------------------------------------------------

trait MatteTf {
    fn eotf(v: f32) -> f32;
    fn oetf(v: f32) -> f32;

    /// Decode an u8 sample to linear f32. Default routes through f32 EOTF;
    /// `SrgbTf` overrides with a 256-entry LUT.
    #[inline(always)]
    fn eotf_u8(b: u8) -> f32 {
        Self::eotf(b as f32 * (1.0 / 255.0))
    }

    /// Encode a linear f32 to u8. Default routes through f32 OETF + clamp +
    /// quantize; `SrgbTf` overrides with a LUT-based encode.
    #[inline(always)]
    fn oetf_u8(lin: f32) -> u8 {
        let v = Self::oetf(lin).clamp(0.0, 1.0);
        (v * 255.0 + 0.5) as u8
    }

    /// Decode a u16 sample to linear f32. Default routes through f32 EOTF;
    /// `SrgbTf` overrides with the 65536-entry LUT.
    #[inline(always)]
    fn eotf_u16(b: u16) -> f32 {
        Self::eotf(b as f32 * (1.0 / 65535.0))
    }

    /// Encode a linear f32 to u16. Default routes through f32 OETF + clamp +
    /// quantize; `SrgbTf` overrides with a LUT-based encode.
    #[inline(always)]
    fn oetf_u16(lin: f32) -> u16 {
        let v = Self::oetf(lin).clamp(0.0, 1.0);
        (v * 65535.0 + 0.5) as u16
    }
}

struct LinearTf;
impl MatteTf for LinearTf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        v
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        v
    }
}

struct SrgbTf;
impl MatteTf for SrgbTf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        linear_srgb::tf::srgb_to_linear(v)
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        linear_srgb::tf::linear_to_srgb(v)
    }
    #[inline(always)]
    fn eotf_u8(b: u8) -> f32 {
        linear_srgb::default::srgb_u8_to_linear(b)
    }
    #[inline(always)]
    fn oetf_u8(lin: f32) -> u8 {
        linear_srgb::default::linear_to_srgb_u8(lin)
    }
    #[inline(always)]
    fn eotf_u16(b: u16) -> f32 {
        linear_srgb::default::srgb_u16_to_linear(b)
    }
    #[inline(always)]
    fn oetf_u16(lin: f32) -> u16 {
        linear_srgb::default::linear_to_srgb_u16(lin)
    }
}

struct Bt709Tf;
impl MatteTf for Bt709Tf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        linear_srgb::tf::bt709_to_linear(v)
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        linear_srgb::tf::linear_to_bt709(v)
    }
}

struct PqTf;
impl MatteTf for PqTf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        linear_srgb::tf::pq_to_linear(v)
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        linear_srgb::tf::linear_to_pq(v)
    }
}

struct HlgTf;
impl MatteTf for HlgTf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        linear_srgb::tf::hlg_to_linear(v)
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        linear_srgb::tf::linear_to_hlg(v)
    }
}

struct Gamma22Tf;
impl MatteTf for Gamma22Tf {
    #[inline(always)]
    fn eotf(v: f32) -> f32 {
        linear_srgb::default::gamma_to_linear(v, ADOBE_GAMMA)
    }
    #[inline(always)]
    fn oetf(v: f32) -> f32 {
        linear_srgb::default::linear_to_gamma(v, ADOBE_GAMMA)
    }
}

/// F32 RGBA → RGB matte composite, monomorphized per TF. Alpha stays
/// linear; RGB is EOTF'd to linear, blended with the pre-linearized matte,
/// then OETF'd back to the source TF.
#[inline]
fn matte_composite_f32_rgba<T: MatteTf>(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    for i in 0..width {
        let s: &[f32; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let d: &mut [f32; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        let a = s[3].clamp(0.0, 1.0);
        let inv_a = 1.0 - a;
        let r_lin = T::eotf(s[0]);
        let g_lin = T::eotf(s[1]);
        let b_lin = T::eotf(s[2]);
        d[0] = T::oetf(r_lin * a + mr_lin * inv_a);
        d[1] = T::oetf(g_lin * a + mg_lin * inv_a);
        d[2] = T::oetf(b_lin * a + mb_lin * inv_a);
    }
}

/// F16 RGBA → RGB matte composite, monomorphized per TF. Chunked 3-pass:
/// batch F16C unpack → generic blend → batch F16C pack.
#[inline]
fn matte_composite_f16_rgba<T: MatteTf>(
    src: &[u16],
    dst: &mut [u16],
    width: usize,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    const CHUNK_PIXELS: usize = 8;
    const SRC_LANES: usize = CHUNK_PIXELS * 4;
    const DST_LANES: usize = CHUNK_PIXELS * 3;

    let mut scratch_src = [0.0f32; SRC_LANES];
    let mut scratch_dst = [0.0f32; DST_LANES];

    let whole = width / CHUNK_PIXELS;
    for c in 0..whole {
        let src_start = c * SRC_LANES;
        let dst_start = c * DST_LANES;
        f16_bits_to_f32_slice(&src[src_start..src_start + SRC_LANES], &mut scratch_src);
        matte_composite_f32_rgba::<T>(
            &scratch_src,
            &mut scratch_dst,
            CHUNK_PIXELS,
            mr_lin,
            mg_lin,
            mb_lin,
        );
        f32_to_f16_bits_slice(&scratch_dst, &mut dst[dst_start..dst_start + DST_LANES]);
    }

    // Tail: per-pixel scalar for any remainder.
    let tail_start = whole * CHUNK_PIXELS;
    for i in tail_start..width {
        let r = f16_bits_to_f32(src[i * 4]);
        let g = f16_bits_to_f32(src[i * 4 + 1]);
        let b = f16_bits_to_f32(src[i * 4 + 2]);
        let a = f16_bits_to_f32(src[i * 4 + 3]).clamp(0.0, 1.0);
        let inv_a = 1.0 - a;
        let r_lin = T::eotf(r);
        let g_lin = T::eotf(g);
        let b_lin = T::eotf(b);
        dst[i * 3] = f32_to_f16_bits(T::oetf(r_lin * a + mr_lin * inv_a));
        dst[i * 3 + 1] = f32_to_f16_bits(T::oetf(g_lin * a + mg_lin * inv_a));
        dst[i * 3 + 2] = f32_to_f16_bits(T::oetf(b_lin * a + mb_lin * inv_a));
    }
}

/// Dispatch table: pick the TF monomorphization at row entry.
fn dispatch_matte_f32_rgba(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    tf: TransferFunction,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => {
            matte_composite_f32_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Srgb => {
            matte_composite_f32_rgba::<SrgbTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Bt709 => {
            matte_composite_f32_rgba::<Bt709Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Pq => {
            matte_composite_f32_rgba::<PqTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Hlg => {
            matte_composite_f32_rgba::<HlgTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Gamma22 => {
            matte_composite_f32_rgba::<Gamma22Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        // Any future non-exhaustive TF falls back to Linear treatment
        // (preserves bytes in linear-space math; same convention as
        // elsewhere in the pipeline for Unknown-ish cases).
        _ => matte_composite_f32_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin),
    }
}

fn dispatch_matte_f16_rgba(
    src: &[u16],
    dst: &mut [u16],
    width: usize,
    tf: TransferFunction,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => {
            matte_composite_f16_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Srgb => {
            matte_composite_f16_rgba::<SrgbTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Bt709 => {
            matte_composite_f16_rgba::<Bt709Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Pq => {
            matte_composite_f16_rgba::<PqTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Hlg => {
            matte_composite_f16_rgba::<HlgTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Gamma22 => {
            matte_composite_f16_rgba::<Gamma22Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        _ => matte_composite_f16_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin),
    }
}

/// U8 RGBA → RGB matte composite, monomorphized per TF. Uses the trait's
/// u8 methods so `SrgbTf` picks up the LUT fast path without the rest of
/// the loop changing.
#[inline]
fn matte_composite_u8_rgba<T: MatteTf>(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    let alpha_scale = 1.0 / 255.0;
    for i in 0..width {
        let s: &[u8; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let d: &mut [u8; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        let a = s[3] as f32 * alpha_scale;
        let inv_a = 1.0 - a;
        let r_lin = T::eotf_u8(s[0]);
        let g_lin = T::eotf_u8(s[1]);
        let b_lin = T::eotf_u8(s[2]);
        d[0] = T::oetf_u8(r_lin * a + mr_lin * inv_a);
        d[1] = T::oetf_u8(g_lin * a + mg_lin * inv_a);
        d[2] = T::oetf_u8(b_lin * a + mb_lin * inv_a);
    }
}

/// U16 RGBA → RGB matte composite, monomorphized per TF.
#[inline]
fn matte_composite_u16_rgba<T: MatteTf>(
    src: &[u16],
    dst: &mut [u16],
    width: usize,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    let alpha_scale = 1.0 / 65535.0;
    for i in 0..width {
        let s: &[u16; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let d: &mut [u16; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        let a = s[3] as f32 * alpha_scale;
        let inv_a = 1.0 - a;
        let r_lin = T::eotf_u16(s[0]);
        let g_lin = T::eotf_u16(s[1]);
        let b_lin = T::eotf_u16(s[2]);
        d[0] = T::oetf_u16(r_lin * a + mr_lin * inv_a);
        d[1] = T::oetf_u16(g_lin * a + mg_lin * inv_a);
        d[2] = T::oetf_u16(b_lin * a + mb_lin * inv_a);
    }
}

fn dispatch_matte_u8_rgba(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tf: TransferFunction,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => {
            matte_composite_u8_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Srgb => {
            matte_composite_u8_rgba::<SrgbTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Bt709 => {
            matte_composite_u8_rgba::<Bt709Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Pq => {
            matte_composite_u8_rgba::<PqTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Hlg => {
            matte_composite_u8_rgba::<HlgTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Gamma22 => {
            matte_composite_u8_rgba::<Gamma22Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        _ => matte_composite_u8_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin),
    }
}

fn dispatch_matte_u16_rgba(
    src: &[u16],
    dst: &mut [u16],
    width: usize,
    tf: TransferFunction,
    mr_lin: f32,
    mg_lin: f32,
    mb_lin: f32,
) {
    match tf {
        TransferFunction::Linear | TransferFunction::Unknown => {
            matte_composite_u16_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Srgb => {
            matte_composite_u16_rgba::<SrgbTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Bt709 => {
            matte_composite_u16_rgba::<Bt709Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Pq => {
            matte_composite_u16_rgba::<PqTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Hlg => {
            matte_composite_u16_rgba::<HlgTf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        TransferFunction::Gamma22 => {
            matte_composite_u16_rgba::<Gamma22Tf>(src, dst, width, mr_lin, mg_lin, mb_lin)
        }
        _ => matte_composite_u16_rgba::<LinearTf>(src, dst, width, mr_lin, mg_lin, mb_lin),
    }
}

/// Composite RGBA onto a solid matte color, producing RGB (4ch → 3ch).
///
/// Blends in linear light: pixel RGB channels are linearized per the
/// source TF, alpha-blended against a pre-linearized matte, then re-encoded
/// to the source TF. Alpha stays linear and is used as-is. All four channel
/// types (U8/U16/F32/F16) dispatch through the same per-TF monomorphized
/// kernel; `SrgbTf` keeps a LUT-based fast path for U8 and U16.
///
/// **Matte interpretation.** The matte (r, g, b) is specified as sRGB u8
/// regardless of source TF — this matches the common usage (CSS-style
/// background colors). For HDR sources (PQ/HLG), the matte is implicitly
/// SDR-range since u8 caps at 255 = 1.0 normalized.
fn matte_composite(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from: PixelDescriptor,
    mr: u8,
    mg: u8,
    mb: u8,
) {
    let ch_type = from.channel_type();
    let tf = from.transfer();

    // Matte is specified in sRGB regardless of source TF (see doc above).
    let mr_lin = linear_srgb::default::srgb_u8_to_linear(mr);
    let mg_lin = linear_srgb::default::srgb_u8_to_linear(mg);
    let mb_lin = linear_srgb::default::srgb_u8_to_linear(mb);

    match ch_type {
        ChannelType::U8 => {
            dispatch_matte_u8_rgba(
                &src[..width * 4],
                &mut dst[..width * 3],
                width,
                tf,
                mr_lin,
                mg_lin,
                mb_lin,
            );
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            dispatch_matte_u16_rgba(src16, dst16, width, tf, mr_lin, mg_lin, mb_lin);
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            dispatch_matte_f32_rgba(srcf, dstf, width, tf, mr_lin, mg_lin, mb_lin);
        }
        ChannelType::F16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            dispatch_matte_f16_rgba(src16, dst16, width, tf, mr_lin, mg_lin, mb_lin);
        }
        _ => {
            // Fallback: just drop alpha
            drop_alpha(src, dst, width, ch_type);
        }
    }
}

// -- gray_to_rgb (pure replicate — U16 and F16 share) ------------------------

#[autoversion]
fn gray_to_rgb_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let g = src[i];
        let d: &mut [u16; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
    }
}

#[autoversion]
fn gray_to_rgb_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let g = src[i];
        let d: &mut [f32; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
    }
}

/// Gray → RGB (replicate).
fn gray_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_to_rgb(&src[..width], &mut dst[..width * 3])
                .expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => gray_to_rgb_2bytes(
            bytemuck::cast_slice(&src[..width * 2]),
            bytemuck::cast_slice_mut(&mut dst[..width * 6]),
            width,
        ),
        ChannelType::F32 => gray_to_rgb_f32(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 12]),
            width,
        ),
        _ => {}
    }
}

// -- gray_to_rgba (replicate + opaque alpha) ---------------------------------

#[autoversion]
fn gray_to_rgba_u16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let g = src[i];
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
        d[3] = 65535;
    }
}

#[autoversion]
fn gray_to_rgba_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let g = src[i];
        let d: &mut [f32; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
        d[3] = 1.0;
    }
}

#[autoversion]
fn gray_to_rgba_f16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let g = src[i];
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
        d[3] = F16_ONE_BITS;
    }
}

/// Gray → RGBA (replicate + opaque alpha).
fn gray_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_to_rgba(&src[..width], &mut dst[..width * 4])
                .expect("pre-validated row size");
        }
        ChannelType::U16 => gray_to_rgba_u16(
            bytemuck::cast_slice(&src[..width * 2]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        ChannelType::F32 => gray_to_rgba_f32(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 16]),
            width,
        ),
        ChannelType::F16 => gray_to_rgba_f16(
            bytemuck::cast_slice(&src[..width * 2]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        _ => {}
    }
}

// ----- RGB → Gray ----------------------------------------------------------
//
// Y' (encoded luma) semantic: coefficients are applied directly to encoded
// (gamma'd) RGB samples, NOT to linear-light values. This is the JPEG/video
// convention, gives bit-exact gray→RGB→gray round-trip when R=G=B, and is
// what the rest of the zen ecosystem expects (see ultrahdr's separate
// linear-luminance computation in `ultrahdr-core/src/color/gamut.rs` —
// linear L is a different quantity, computed where it's actually needed).
//
// Coefficient resolution happens at plan build time; kernels just consume
// the coefficients triple. BT.709 u8 paths keep garb's fixed-point fast
// path; other coefficient choices on u8 + all U16/F32/F16 paths use the
// generic f32 path.

#[autoversion]
fn rgb_to_gray_u8_generic(src: &[u8], dst: &mut [u8], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[u8; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        dst[i] = (y + 0.5).clamp(0.0, 255.0) as u8;
    }
}

#[autoversion]
fn rgba_to_gray_u8_generic(src: &[u8], dst: &mut [u8], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[u8; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        dst[i] = (y + 0.5).clamp(0.0, 255.0) as u8;
    }
}

#[autoversion]
fn rgb_to_gray_u16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[u16; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        dst[i] = (y + 0.5).clamp(0.0, 65535.0) as u16;
    }
}

#[autoversion]
fn rgba_to_gray_u16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[u16; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        dst[i] = (y + 0.5).clamp(0.0, 65535.0) as u16;
    }
}

#[autoversion]
fn rgb_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[f32; 3] = (&src[i * 3..i * 3 + 3]).try_into().unwrap();
        dst[i] = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];
    }
}

#[autoversion]
fn rgba_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let s: &[f32; 4] = (&src[i * 4..i * 4 + 4]).try_into().unwrap();
        dst[i] = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];
    }
}

#[autoversion]
fn rgb_to_gray_f16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let r = f16_bits_to_f32(src[i * 3]);
        let g = f16_bits_to_f32(src[i * 3 + 1]);
        let b = f16_bits_to_f32(src[i * 3 + 2]);
        let y = r * w[0] + g * w[1] + b * w[2];
        dst[i] = f32_to_f16_bits(y);
    }
}

#[autoversion]
fn rgba_to_gray_f16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    for i in 0..width {
        let r = f16_bits_to_f32(src[i * 4]);
        let g = f16_bits_to_f32(src[i * 4 + 1]);
        let b = f16_bits_to_f32(src[i * 4 + 2]);
        let y = r * w[0] + g * w[1] + b * w[2];
        dst[i] = f32_to_f16_bits(y);
    }
}

/// RGB → Gray using user-specified luma coefficients. Y' (encoded) semantic.
fn rgb_to_gray(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    coefficients: LumaCoefficients,
) {
    let w = coefficients.coefficients();
    match ch_type {
        ChannelType::U8 => {
            if coefficients == LumaCoefficients::Bt709 {
                garb::bytes::rgb_to_gray_bt709(&src[..width * 3], &mut dst[..width])
                    .expect("pre-validated row size");
            } else {
                rgb_to_gray_u8_generic(&src[..width * 3], &mut dst[..width], width, w);
            }
        }
        ChannelType::U16 => rgb_to_gray_u16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 2]),
            width,
            w,
        ),
        ChannelType::F32 => rgb_to_gray_f32(
            bytemuck::cast_slice(&src[..width * 12]),
            bytemuck::cast_slice_mut(&mut dst[..width * 4]),
            width,
            w,
        ),
        ChannelType::F16 => rgb_to_gray_f16(
            bytemuck::cast_slice(&src[..width * 6]),
            bytemuck::cast_slice_mut(&mut dst[..width * 2]),
            width,
            w,
        ),
        _ => {}
    }
}

/// RGBA → Gray (drop alpha) using user-specified luma coefficients.
fn rgba_to_gray(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    coefficients: LumaCoefficients,
) {
    let w = coefficients.coefficients();
    match ch_type {
        ChannelType::U8 => {
            if coefficients == LumaCoefficients::Bt709 {
                garb::bytes::rgba_to_gray_bt709(&src[..width * 4], &mut dst[..width])
                    .expect("pre-validated row size");
            } else {
                rgba_to_gray_u8_generic(&src[..width * 4], &mut dst[..width], width, w);
            }
        }
        ChannelType::U16 => rgba_to_gray_u16(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 2]),
            width,
            w,
        ),
        ChannelType::F32 => rgba_to_gray_f32(
            bytemuck::cast_slice(&src[..width * 16]),
            bytemuck::cast_slice_mut(&mut dst[..width * 4]),
            width,
            w,
        ),
        ChannelType::F16 => rgba_to_gray_f16(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 2]),
            width,
            w,
        ),
        _ => {}
    }
}

// -- gray_alpha_to_rgba (pure replicate + alpha-preserve — U16 and F16 share)

#[autoversion]
fn gray_alpha_to_rgba_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let s: &[u16; 2] = (&src[i * 2..i * 2 + 2]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[0];
        d[2] = s[0];
        d[3] = s[1];
    }
}

#[autoversion]
fn gray_alpha_to_rgba_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let s: &[f32; 2] = (&src[i * 2..i * 2 + 2]).try_into().unwrap();
        let d: &mut [f32; 4] = (&mut dst[i * 4..i * 4 + 4]).try_into().unwrap();
        d[0] = s[0];
        d[1] = s[0];
        d[2] = s[0];
        d[3] = s[1];
    }
}

/// GrayAlpha → RGBA (replicate gray, preserve alpha).
fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_alpha_to_rgba(&src[..width * 2], &mut dst[..width * 4])
                .expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => gray_alpha_to_rgba_2bytes(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        ChannelType::F32 => gray_alpha_to_rgba_f32(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 16]),
            width,
        ),
        _ => {}
    }
}

// -- gray_alpha_to_rgb (replicate + drop alpha — U16 and F16 share) ---------

#[autoversion]
fn gray_alpha_to_rgb_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let g = src[i * 2];
        let d: &mut [u16; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
    }
}

#[autoversion]
fn gray_alpha_to_rgb_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let g = src[i * 2];
        let d: &mut [f32; 3] = (&mut dst[i * 3..i * 3 + 3]).try_into().unwrap();
        d[0] = g;
        d[1] = g;
        d[2] = g;
    }
}

/// GrayAlpha → RGB (replicate gray, drop alpha).
fn gray_alpha_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_alpha_to_rgb(&src[..width * 2], &mut dst[..width * 3])
                .expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => gray_alpha_to_rgb_2bytes(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 6]),
            width,
        ),
        ChannelType::F32 => gray_alpha_to_rgb_f32(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 12]),
            width,
        ),
        _ => {}
    }
}

// -- gray_to_gray_alpha (inserts opaque alpha: differs per type) -------------

#[autoversion]
fn gray_to_gray_alpha_u16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let d: &mut [u16; 2] = (&mut dst[i * 2..i * 2 + 2]).try_into().unwrap();
        d[0] = src[i];
        d[1] = 65535;
    }
}

#[autoversion]
fn gray_to_gray_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let d: &mut [f32; 2] = (&mut dst[i * 2..i * 2 + 2]).try_into().unwrap();
        d[0] = src[i];
        d[1] = 1.0;
    }
}

#[autoversion]
fn gray_to_gray_alpha_f16(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let d: &mut [u16; 2] = (&mut dst[i * 2..i * 2 + 2]).try_into().unwrap();
        d[0] = src[i];
        d[1] = F16_ONE_BITS;
    }
}

/// Gray → GrayAlpha (add opaque alpha).
fn gray_to_gray_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_to_gray_alpha(&src[..width], &mut dst[..width * 2])
                .expect("pre-validated row size");
        }
        ChannelType::U16 => gray_to_gray_alpha_u16(
            bytemuck::cast_slice(&src[..width * 2]),
            bytemuck::cast_slice_mut(&mut dst[..width * 4]),
            width,
        ),
        ChannelType::F32 => gray_to_gray_alpha_f32(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 8]),
            width,
        ),
        ChannelType::F16 => gray_to_gray_alpha_f16(
            bytemuck::cast_slice(&src[..width * 2]),
            bytemuck::cast_slice_mut(&mut dst[..width * 4]),
            width,
        ),
        _ => {}
    }
}

// -- gray_alpha_to_gray (drop alpha — U16 and F16 share) --------------------

#[autoversion]
fn gray_alpha_to_gray_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        dst[i] = src[i * 2];
    }
}

#[autoversion]
fn gray_alpha_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        dst[i] = src[i * 2];
    }
}

/// GrayAlpha → Gray (drop alpha).
fn gray_alpha_to_gray(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_alpha_to_gray(&src[..width * 2], &mut dst[..width])
                .expect("pre-validated row size");
        }
        ChannelType::U16 | ChannelType::F16 => gray_alpha_to_gray_2bytes(
            bytemuck::cast_slice(&src[..width * 4]),
            bytemuck::cast_slice_mut(&mut dst[..width * 2]),
            width,
        ),
        ChannelType::F32 => gray_alpha_to_gray_f32(
            bytemuck::cast_slice(&src[..width * 8]),
            bytemuck::cast_slice_mut(&mut dst[..width * 4]),
            width,
        ),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Alpha peel for the transfer-function kernels
// ---------------------------------------------------------------------------
//
// A transfer function maps **light** to **code value**. Alpha is a coverage
// fraction, not light — transferring it is meaningless, and it silently
// corrupts everything downstream: with `ConvertIntent::Blend` the planner
// follows the linearize step with `StraightToPremul`, which then multiplies
// the colour channels by the corrupted alpha.
//
// Every TF kernel below hands the whole flat row (`width * channels` lanes)
// to a channel-agnostic SIMD EOTF/OETF — that flat span is exactly what makes
// them fast, and peeling alpha out first would cost a gather/scatter. So each
// kernel transfers every lane and then overwrites the alpha lane, carrying it
// linearly across whatever depth change the kernel also performs. This is the
// shape the PQ kernels have always had; these helpers give every other
// transfer the same treatment from one place.
//
// **Alpha is the last channel** of every alpha-bearing layout — `GrayAlpha`
// (index 1 of 2), `Rgba` / `Bgra` / `OklabA` (index 3 of 4). That is the same
// rule `adapt.rs` encodes as `(channels - 1) * cs`. Note this is *not* the
// same as `channels == 4`: `Cmyk` is four channels with **no** alpha, so its
// K lane is correctly left transferred, and `GrayAlpha` carries alpha at
// index 1 where a `== 4` test would miss it entirely.
//
// The scale factors match the naive (`garb`) depth kernels exactly — `v/255`,
// `v/65535`, `clamp(v)*255 + 0.5`, `clamp(v)*65535 + 0.5` — so the alpha lane
// lands on the same byte it would have taken through a TF-free path.

/// Index of the alpha lane within a pixel, or `None` for alpha-free layouts.
#[inline]
fn alpha_lane(layout: ChannelLayout) -> Option<usize> {
    if layout.has_alpha() {
        Some(layout.channels() - 1)
    } else {
        None
    }
}

/// Restore the alpha lane of an f32 row from an f32 source row (same depth).
#[inline]
fn restore_alpha_f32_f32(src: &[f32], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = src[i * ch + ai];
    }
}

/// Restore the alpha lane of an f32 row from a u8 source row (`v / 255`).
#[inline]
fn restore_alpha_u8_f32(src: &[u8], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = f32::from(src[i * ch + ai]) / 255.0;
    }
}

/// Restore the alpha lane of an f32 row from a u16 source row (`v / 65535`).
#[inline]
fn restore_alpha_u16_f32(src: &[u16], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = f32::from(src[i * ch + ai]) / 65535.0;
    }
}

/// Restore the alpha lane of a u8 row from an f32 source row
/// (`clamp(v) * 255 + 0.5`).
#[inline]
fn restore_alpha_f32_u8(src: &[f32], dst: &mut [u8], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = (src[i * ch + ai].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// Restore the alpha lane of a u16 row from an f32 source row
/// (`clamp(v) * 65535 + 0.5`).
#[inline]
fn restore_alpha_f32_u16(src: &[f32], dst: &mut [u16], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = (src[i * ch + ai].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// ---------------------------------------------------------------------------
// Depth conversion kernels (transfer-function-aware)
// ---------------------------------------------------------------------------

/// sRGB u8 → linear f32 using `linear-srgb` SIMD batch conversion.
/// Alpha-preserving: the alpha lane is carried linearly (`v / 255`).
fn srgb_u8_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_srgb::default::srgb_u8_to_linear_slice(&src[..count], dstf);
    restore_alpha_u8_f32(&src[..count], dstf, layout);
}

/// Linear f32 → sRGB u8 using `linear-srgb` SIMD batch conversion.
/// Alpha-preserving: the alpha lane is scaled linearly, never OETF'd.
fn linear_f32_to_srgb_u8(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    linear_srgb::default::linear_to_srgb_u8_slice(srcf, &mut dst[..count]);
    restore_alpha_f32_u8(srcf, &mut dst[..count], layout);
}

/// Naive u8 → f32 (v / 255.0, no transfer function).
fn naive_u8_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u8_to_f32(&src[..count], &mut dst[..count * 4])
        .expect("pre-validated row size");
}

/// Naive f32 → u8 (clamp `[0,1]`, * 255 + 0.5).
fn naive_f32_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_f32_to_u8(&src[..count * 4], &mut dst[..count])
        .expect("pre-validated row size");
}

/// u16 → u8, correctly rounded: `round(v / 257)` (65535 = 255 · 257, so
/// `v * 255 / 65535 == v / 257` exactly; 257 is odd, so no input ever sits
/// on an exact .5 tie and `(v + 128) / 257` is the unique nearest u8).
///
/// Byte-lane form: with `v = 256·hi + lo = 257·hi + (lo − hi)`,
/// `round(v / 257) = hi + [lo − hi ≥ 129] − [hi − lo ≥ 129]` (the
/// correction term is in `(−1, 1)` and rounds away from zero exactly when
/// `|lo − hi| ≥ 129`). Two saturating subs + two compares + two adds, all
/// in 8-bit lanes — LLVM auto-vectorises it 16 lanes wide; pinned
/// exhaustively against `(v + 128) / 257` by
/// `tests/ulp_exhaustive.rs::ulp_u16_to_u8_max_error`.
///
/// Deliberately local rather than `garb::bytes::convert_u16_to_u8`: garb's
/// `(v * 255 + 32768) >> 16` divides by 65536, which floors 127 of the 65536
/// inputs by 1 LSB (e.g. `33025 → 128`, exact `129`) and made this step
/// disagree with the f32 route (`U16ToF32` → `NaiveF32ToU8`, `v * 255 + 0.5`).
/// Two routes for one operation must be byte-identical — imazen/zenpixels#72.
/// Measured 4.5× faster than garb's kernel as well (`benches/bench_u16_narrow.rs`,
/// `benchmarks/u16_narrow_2026-08-27.txt`).
fn u16_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let (pairs, _) = src[..count * 2].as_chunks::<2>();
    for (s, d) in pairs.iter().zip(dst[..count].iter_mut()) {
        let [lo, hi] = u16::from_ne_bytes(*s).to_le_bytes();
        let up = u8::from(lo.saturating_sub(hi) > 128);
        let down = u8::from(hi.saturating_sub(lo) > 128);
        *d = hi.wrapping_add(up).wrapping_sub(down);
    }
}

/// u8 → u16: v * 257.
fn u8_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u8_to_u16(&src[..count], &mut dst[..count * 2])
        .expect("pre-validated row size");
}

/// u16 → f32: v / 65535.0.
fn u16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u16_to_f32(&src[..count * 2], &mut dst[..count * 4])
        .expect("pre-validated row size");
}

/// f32 → u16: clamp `[0,1]`, * 65535 + 0.5.
fn f32_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_f32_to_u16(&src[..count * 4], &mut dst[..count * 2])
        .expect("pre-validated row size");
}

/// f16 → f32: IEEE 754 half-precision unpack (no TF, no scale).
///
/// Dispatched via `f16_bits_to_f32_slice` — F16C (VCVTPH2PS) on x86-64
/// CPUs that have it, scalar elsewhere.
fn f16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src_bits: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    f16_bits_to_f32_slice(src_bits, dst_f32);
}

/// f32 → f16: IEEE 754 half-precision pack (round-to-nearest-even, no TF).
///
/// Dispatched via `f32_to_f16_bits_slice` — F16C (VCVTPS2PH) on x86-64
/// CPUs that have it, scalar elsewhere. Values outside ±65504 saturate
/// to ±infinity; NaNs are preserved.
fn f32_to_f16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src_f32: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst_bits: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    f32_to_f16_bits_slice(src_f32, dst_bits);
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084) transfer function — delegates to linear-srgb
// ---------------------------------------------------------------------------

/// PQ EOTF: encoded `[0,1]` → linear light `[0,1]` (where 1.0 = 10000 cd/m²).
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    linear_srgb::tf::pq_to_linear(v)
}

/// PQ inverse EOTF (OETF): linear light `[0,1]` → encoded `[0,1]`.
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_pq(v)
}

/// Per-tier SIMD body for [`multiply_color_channels`]. `channels == 4` builds a
/// `[f, f, f, 1]`-repeating multiplier so the alpha lane (every 4th) is left
/// **untouched**; any other channel count scales every lane uniformly. The
/// 16-lane chunk is exactly 4 RGBA pixels, so the pattern stays pixel-aligned
/// across the whole row, and `count` being a multiple of `channels` keeps the
/// remainder pixel-aligned too.
#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn multiply_color_channels_tier(token: Token, buf: &mut [f32], channels: usize, factor: f32) {
    let mul = if channels == 4 {
        f32x16::from_array(
            token,
            [
                factor, factor, factor, 1.0, factor, factor, factor, 1.0, factor, factor, factor,
                1.0, factor, factor, factor, 1.0,
            ],
        )
    } else {
        f32x16::from_array(token, [factor; 16])
    };
    let (chunks, remainder) = buf.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = f32x16::from_array(token, *chunk);
        *chunk = (v * mul).to_array();
    }
    if channels == 4 {
        for px in remainder.as_chunks_mut::<4>().0 {
            px[0] *= factor;
            px[1] *= factor;
            px[2] *= factor;
        }
    } else {
        for v in remainder {
            *v *= factor;
        }
    }
}

/// Multiply the color channels of an interleaved `f32` buffer by `factor`,
/// SIMD-dispatched (AVX-512/AVX2/SSE/NEON/WASM via `magetypes`).
///
/// **With alpha** (`channels == 4`): the R/G/B lanes are scaled and the alpha
/// lane is preserved — alpha is not luminance, so an absolute-luminance anchor
/// never applies to it. **Without alpha** (any other `channels`): every lane is
/// a color channel and is scaled uniformly. `factor == 1.0` is an early-out
/// (`x * 1.0 == x`), so the un-anchored path is bit-for-bit the plain result.
pub(crate) fn multiply_color_channels(buf: &mut [f32], channels: usize, factor: f32) {
    if factor == 1.0 {
        return;
    }
    incant!(
        multiply_color_channels_tier(buf, channels, factor),
        [v4, v3, neon, wasm128, scalar]
    );
}

// ---------------------------------------------------------------------------
// Vendored **precise** SIMD PQ (exact SMPTE ST 2084)
// ---------------------------------------------------------------------------
//
// `linear_srgb::default`'s PQ slice uses a rational-polynomial fit whose valid
// range starts at v≈0.02 (its *scalar* path switches to an exact `powf` below
// that); applied as a slice it extrapolates the poly all the way to 0, so the
// tight U16 → f32 → U16 round-trip drifts up to ~256 codes near black. We vendor
// the **exact** ST 2084 formula and evaluate it in SIMD with magetypes' precise
// `pow`, giving full-range precision (round-trip ≤1) while staying SIMD. The
// scalar remainder defers to `linear_srgb::tf` (exact-below-threshold). Operates
// on every lane; the kernels restore any alpha lane afterward (alpha is linear).
// Canonical SMPTE ST 2084 constants (m1 = 2610/16384, m2 = 2523·128/4096, etc.),
// written in their exact rational decimal form — all are exactly f32-representable
// even though the literals run longer than the shortest round-trip.
#[allow(clippy::excessive_precision)]
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
#[allow(clippy::excessive_precision)]
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

/// PQ EOTF (signal → linear), exact, precise SIMD.
#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn pq_eotf_slice_tier(token: Token, buf: &mut [f32]) {
    let zero = f32x16::splat(token, 0.0);
    let c1 = f32x16::splat(token, PQ_C1);
    let c2 = f32x16::splat(token, PQ_C2);
    let c3 = f32x16::splat(token, PQ_C3);
    let (chunks, rem) = buf.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = f32x16::from_array(token, *chunk).max(zero);
        let vp = v.pow_midp_precise(1.0 / PQ_M2);
        let num = (vp - c1).max(zero);
        let den = c2 - c3 * vp;
        let lin = (num / den).pow_midp_precise(1.0 / PQ_M1);
        *chunk = f32x16::blend(v.simd_le(zero), zero, lin).to_array();
    }
    for v in rem {
        *v = linear_srgb::tf::pq_to_linear(*v);
    }
}

/// PQ inverse-EOTF / OETF (linear → signal), exact, precise SIMD.
#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn pq_oetf_slice_tier(token: Token, buf: &mut [f32]) {
    let zero = f32x16::splat(token, 0.0);
    let one = f32x16::splat(token, 1.0);
    let c1 = f32x16::splat(token, PQ_C1);
    let c2 = f32x16::splat(token, PQ_C2);
    let c3 = f32x16::splat(token, PQ_C3);
    let (chunks, rem) = buf.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = f32x16::from_array(token, *chunk).max(zero);
        let vp = v.pow_midp_precise(PQ_M1);
        let sig = ((c1 + c2 * vp) / (one + c3 * vp)).pow_midp_precise(PQ_M2);
        *chunk = f32x16::blend(v.simd_le(zero), zero, sig).to_array();
    }
    for v in rem {
        *v = linear_srgb::tf::linear_to_pq(*v);
    }
}

/// PQ EOTF over an interleaved f32 slice, every lane (precise SIMD).
pub(crate) fn pq_eotf_slice(buf: &mut [f32]) {
    incant!(pq_eotf_slice_tier(buf), [v4, v3, neon, wasm128, scalar]);
}

/// PQ OETF over an interleaved f32 slice, every lane (precise SIMD).
pub(crate) fn pq_oetf_slice(buf: &mut [f32]) {
    incant!(pq_oetf_slice_tier(buf), [v4, v3, neon, wasm128, scalar]);
}

/// PQ U16 → Linear F32 (EOTF during depth conversion), alpha-preserving.
///
/// Widens the U16 codes (`garb` SIMD) and applies the precise SIMD PQ EOTF
/// ([`pq_eotf_slice`]) to land relative-linear, dividing the RGB lanes by `scale`
/// (`diffuse_white / 10000`). The EOTF runs over every lane; for alpha-bearing
/// layouts the alpha lane is then overwritten with its (un-transformed,
/// un-anchored) linear value. `scale == 1.0` is the identity anchor.
fn pq_u16_to_linear_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
) {
    let channels = layout.channels();
    let count = width * channels;
    garb::bytes::convert_u16_to_f32(&src[..count * 2], &mut dst[..count * 4])
        .expect("pre-validated row size");
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    pq_eotf_slice(&mut dstf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, 1.0 / scale);
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    restore_alpha_u16_f32(src16, &mut dstf[..count], layout);
}

/// Linear F32 → PQ U16 (OETF during depth conversion), alpha-preserving.
///
/// Anchors the RGB lanes (`× scale`, negatives → 0), applies the precise SIMD PQ
/// OETF ([`pq_oetf_slice`]) in fixed stack-sized chunks (the U16 output is
/// half-width, so it cannot host the in-place f32 transform), then narrows to U16
/// (`garb` SIMD). For alpha-bearing layouts the alpha lane is overwritten with its
/// linear → U16 value (never OETF'd or anchored). `scale == 1.0` is the identity.
fn linear_f32_to_pq_u16(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
) {
    let channels = layout.channels();
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    // CHUNK is a multiple of 16 (SIMD width), 4 (RGBA grouping) and 2
    // (GrayAlpha grouping), and `count` is a multiple of `channels`, so every
    // chunk boundary stays pixel-aligned for every alpha-bearing layout.
    const CHUNK: usize = 1024;
    let mut buf = [0.0f32; CHUNK];
    let mut off = 0;
    while off < count {
        let n = min(count - off, CHUNK);
        let b = &mut buf[..n];
        b.copy_from_slice(&srcf[off..off + n]);
        multiply_color_channels(b, channels, scale);
        pq_oetf_slice(b);
        garb::bytes::convert_f32_to_u16(
            bytemuck::cast_slice(&b[..]),
            &mut dst[off * 2..(off + n) * 2],
        )
        .expect("pre-validated row size");
        let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[off * 2..(off + n) * 2]);
        restore_alpha_f32_u16(&srcf[off..off + n], dst16, layout);
        off += n;
    }
}

/// PQ F32 → Linear F32 (EOTF, same depth), alpha-preserving. Precise SIMD.
///
/// `scale` is the `diffuse_white / 10000` anchor (see [`pq_u16_to_linear_f32`]);
/// the RGB lanes are divided by it after the EOTF and, for alpha-bearing layouts,
/// the alpha lane is restored to its (un-transformed) input. `scale == 1.0` is a
/// no-op so the un-anchored result depends only on [`pq_eotf_slice`].
fn pq_f32_to_linear_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
) {
    let channels = layout.channels();
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    pq_eotf_slice(&mut dstf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, 1.0 / scale);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → PQ F32 (OETF, same depth), alpha-preserving. Precise SIMD.
///
/// `scale` is the `diffuse_white / 10000` anchor (see [`linear_f32_to_pq_u16`]);
/// the RGB lanes are multiplied by it before the OETF and, for alpha-bearing
/// layouts, the alpha lane is restored to its (un-transformed) linear input.
fn linear_f32_to_pq_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
) {
    let channels = layout.channels();
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, scale);
    pq_oetf_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) transfer function — delegates to linear-srgb
// ---------------------------------------------------------------------------
//
// PHOTOMETRY HAZARD: these kernels apply only the HLG OETF/inverse-OETF, which
// produce **scene-referred, normalized** linear (`[0,1]`, no absolute luminance,
// no OOTF). PQ's linear is **absolute display** light (cd/m²). So a planned
// HLG↔PQ conversion (HLG-EOTF → "linear" → PQ-OETF) is mechanically defined but
// **not photometrically correct** — it conflates the two domains, skipping the
// HLG OOTF (system γ ≈ 1.2 + 0.42·log10(Lw/1000)) and the peak-luminance (Lw)
// mapping. Correct HLG↔PQ needs `(diffuse_white, Lw)` threaded through these
// steps + the OOTF (zentone::hlg) — tracked for a follow-up; deliberately out of
// the PQ-only scope here. `quantize_to` already refuses HLG targets for this
// reason; the general `convert_*` path does not yet guard it.

/// HLG OETF: scene-linear `[0,1]` → encoded `[0,1]`.
///
/// Uses `fast_log2f` from `linear-srgb` (no `libm` ln calls).
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_hlg(v)
}

/// HLG inverse OETF (EOTF): encoded `[0,1]` → scene-linear `[0,1]`.
///
/// Uses `fast_pow2f` from `linear-srgb` (no `libm` exp calls).
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    linear_srgb::tf::hlg_to_linear(v)
}

/// HLG U16 → Linear F32 (EOTF applied during depth conversion),
/// alpha-preserving (the alpha lane is carried linearly, `v / 65535`).
fn hlg_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    hlg_u16_to_linear_f32_inner(src16, dstf);
    restore_alpha_u16_f32(src16, dstf, layout);
}

#[autoversion]
fn hlg_u16_to_linear_f32_inner(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src
        .as_chunks::<16>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<16>().0)
    {
        for i in 0..16 {
            d[i] = linear_srgb::tf::hlg_to_linear(s[i] as f32 / 65535.0);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::hlg_to_linear(src[off + i] as f32 / 65535.0);
        }
    }
}

/// Linear F32 → HLG U16 (OETF applied during depth conversion),
/// alpha-preserving (the alpha lane is scaled linearly, never HLG-encoded).
fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    linear_f32_to_hlg_u16_inner(srcf, dst16);
    restore_alpha_f32_u16(srcf, dst16, layout);
}

#[autoversion]
fn linear_f32_to_hlg_u16_inner(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src
        .as_chunks::<16>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<16>().0)
    {
        for i in 0..16 {
            let encoded = linear_srgb::tf::linear_to_hlg(s[i]);
            d[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            let encoded = linear_srgb::tf::linear_to_hlg(src[off + i]);
            dst[off + i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
}

/// HLG F32 → Linear F32 (EOTF, same depth). SIMD-dispatched,
/// alpha-preserving.
fn hlg_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::hlg_to_linear_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → HLG F32 (OETF, same depth). SIMD-dispatched,
/// alpha-preserving.
fn linear_f32_to_hlg_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_hlg_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

// ---------------------------------------------------------------------------
// sRGB / BT.709 F32 ↔ Linear F32 transfer function kernels
// ---------------------------------------------------------------------------

/// sRGB F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
/// Clamps to [0, 1] — use `srgb_to_linear_extended_slice` for HDR/WCG workflows
/// that need to preserve out-of-gamut values (pending configurable option).
fn srgb_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::srgb_to_linear_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → sRGB F32 (OETF, same depth). SIMD-dispatched.
/// Clamps to [0, 1] — use `linear_to_srgb_extended_slice` for HDR/WCG workflows
/// that need to preserve out-of-gamut values (pending configurable option).
fn linear_f32_to_srgb_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_srgb_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// sRGB F32 → Linear F32 (extended range, sign-preserving). Alpha-preserving.
fn srgb_f32_to_linear_f32_extended(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::srgb_to_linear_extended_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → sRGB F32 (extended range, sign-preserving). Alpha-preserving.
fn linear_f32_to_srgb_f32_extended(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_srgb_extended_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// BT.709 F32 → Linear F32 (EOTF, same depth). Alpha-preserving.
fn bt709_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    bt709_f32_to_linear_f32_inner(&srcf[..count], &mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

#[autoversion]
fn bt709_f32_to_linear_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src
        .as_chunks::<16>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<16>().0)
    {
        for i in 0..16 {
            d[i] = linear_srgb::tf::bt709_to_linear(s[i]);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::bt709_to_linear(src[off + i]);
        }
    }
}

/// Linear F32 → BT.709 F32 (OETF, same depth). Alpha-preserving.
fn linear_f32_to_bt709_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_f32_to_bt709_f32_inner(&srcf[..count], &mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

#[autoversion]
fn linear_f32_to_bt709_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src
        .as_chunks::<16>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<16>().0)
    {
        for i in 0..16 {
            d[i] = linear_srgb::tf::linear_to_bt709(s[i]);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::linear_to_bt709(src[off + i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Gamma 2.2 (Adobe RGB 1998) F32 ↔ Linear F32
// ---------------------------------------------------------------------------

/// Adobe RGB 1998 canonical exponent (563/256). Matches ~85% of real-world
/// Adobe RGB ICC profiles (Adobe CS4, Windows ClayRGB1998, macOS AdobeRGB1998,
/// Linux `AdobeRGB1998`, Nikon). Parametric-curve variants with a linear toe
/// are routed through full CMS instead.
///
/// `2.19921875 = 563/256` is exact in f32; the allow suppresses clippy's
/// decimal-digit heuristic.
#[allow(clippy::excessive_precision)]
const ADOBE_GAMMA: f32 = 2.19921875;

/// Gamma 2.2 F32 → Linear F32 (EOTF, same depth). SIMD-dispatched via
/// `linear_srgb::default::gamma_to_linear_slice`. Alpha-preserving.
fn gamma22_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::gamma_to_linear_slice(&mut dstf[..count], ADOBE_GAMMA);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → Gamma 2.2 F32 (OETF, same depth). SIMD-dispatched.
/// Alpha-preserving.
fn linear_f32_to_gamma22_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_gamma_slice(&mut dstf[..count], ADOBE_GAMMA);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

// ---------------------------------------------------------------------------
// Alpha premultiplication
// ---------------------------------------------------------------------------
//
// Pattern: dispatch on (ChannelType, channels) to concrete #[autoversion]
// kernels. Each kernel has a flat per-pixel loop with fixed-size array
// slicing at the pixel boundary so LLVM can drop bounds checks and
// vectorize. Empirically ~10× faster than the previous big-match-in-fn
// shape on L2-sized rows (see benchmarks/premul_u16_2026-04-23_baseline).
// ---------------------------------------------------------------------------

// -- Straight → Premultiplied: per-(type, channels) kernels ------------------

#[autoversion]
fn premul_u8_ga(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[u8; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [u8; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1] as u32;
        d[0] = ((s[0] as u32 * a + 128) / 255) as u8;
        d[1] = s[1];
    }
}

#[autoversion]
fn premul_u16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[u16; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [u16; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1] as u32;
        d[0] = ((s[0] as u32 * a + 32768) / 65535) as u16;
        d[1] = s[1];
    }
}

#[autoversion]
fn premul_u16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[u16; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        let a = s[3] as u32;
        d[0] = ((s[0] as u32 * a + 32768) / 65535) as u16;
        d[1] = ((s[1] as u32 * a + 32768) / 65535) as u16;
        d[2] = ((s[2] as u32 * a + 32768) / 65535) as u16;
        d[3] = s[3];
    }
}

#[autoversion]
fn premul_f32_ga(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[f32; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [f32; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1];
        d[0] = s[0] * a;
        d[1] = a;
    }
}

/// F16 premul via 3-pass scratch: f16→f32 slice (F16C SIMD) → scalar math
/// (LLVM autovec) → f32→f16 slice (F16C SIMD). Beats per-pixel scalar
/// `f16_bits_to_f32` because the conversion cost dominates.
fn premul_f16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    const CHUNK_PIXELS: usize = 16;
    const CHUNK_LANES: usize = CHUNK_PIXELS * 2;

    let mut scratch_src = [0.0f32; CHUNK_LANES];
    let mut scratch_dst = [0.0f32; CHUNK_LANES];

    let whole = width / CHUNK_PIXELS;
    for c in 0..whole {
        let start = c * CHUNK_LANES;
        f16_bits_to_f32_slice(&src[start..start + CHUNK_LANES], &mut scratch_src);

        for i in 0..CHUNK_PIXELS {
            let base = i * 2;
            let a = scratch_src[base + 1];
            scratch_dst[base] = scratch_src[base] * a;
            scratch_dst[base + 1] = a;
        }

        f32_to_f16_bits_slice(&scratch_dst, &mut dst[start..start + CHUNK_LANES]);
    }

    let tail_start = whole * CHUNK_PIXELS;
    for i in tail_start..width {
        let base = i * 2;
        let a = f16_bits_to_f32(src[base + 1]);
        dst[base] = f32_to_f16_bits(f16_bits_to_f32(src[base]) * a);
        dst[base + 1] = src[base + 1];
    }
}

fn premul_f16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    const CHUNK_PIXELS: usize = 8;
    const CHUNK_LANES: usize = CHUNK_PIXELS * 4;

    let mut scratch_src = [0.0f32; CHUNK_LANES];
    let mut scratch_dst = [0.0f32; CHUNK_LANES];

    let whole = width / CHUNK_PIXELS;
    for c in 0..whole {
        let start = c * CHUNK_LANES;
        f16_bits_to_f32_slice(&src[start..start + CHUNK_LANES], &mut scratch_src);

        for i in 0..CHUNK_PIXELS {
            let base = i * 4;
            let a = scratch_src[base + 3];
            scratch_dst[base] = scratch_src[base] * a;
            scratch_dst[base + 1] = scratch_src[base + 1] * a;
            scratch_dst[base + 2] = scratch_src[base + 2] * a;
            scratch_dst[base + 3] = a;
        }

        f32_to_f16_bits_slice(&scratch_dst, &mut dst[start..start + CHUNK_LANES]);
    }

    let tail_start = whole * CHUNK_PIXELS;
    for i in tail_start..width {
        let base = i * 4;
        let a = f16_bits_to_f32(src[base + 3]);
        dst[base] = f32_to_f16_bits(f16_bits_to_f32(src[base]) * a);
        dst[base + 1] = f32_to_f16_bits(f16_bits_to_f32(src[base + 1]) * a);
        dst[base + 2] = f32_to_f16_bits(f16_bits_to_f32(src[base + 2]) * a);
        dst[base + 3] = src[base + 3];
    }
}

/// Straight → Premultiplied alpha (copy from src to dst).
fn straight_to_premul(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    match (ch_type, channels) {
        // Garb fast paths (SIMD, RGBA 4-channel).
        (ChannelType::U8, 4) => {
            let n = width * 4;
            garb::bytes::premultiply_alpha_rgba_u8_copy(&src[..n], &mut dst[..n])
                .expect("pre-validated row size");
        }
        (ChannelType::F32, 4) => {
            let n = width * 16;
            garb::bytes::premultiply_alpha_f32_copy(&src[..n], &mut dst[..n])
                .expect("pre-validated row size");
        }
        // Per-type autoversion kernels for the remaining shapes.
        (ChannelType::U8, 2) => premul_u8_ga(&src[..width * 2], &mut dst[..width * 2], width),
        (ChannelType::U16, 2) => {
            let n = width * 4;
            premul_u16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::U16, 4) => {
            let n = width * 8;
            premul_u16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F32, 2) => {
            let n = width * 8;
            premul_f32_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F16, 2) => {
            let n = width * 4;
            premul_f16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F16, 4) => {
            let n = width * 8;
            premul_f16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        _ => {
            // Fallback: byte copy.
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

// -- Premultiplied → Straight: per-(type, channels) kernels ------------------
//
// Each arm handles a == 0 by zeroing all channels (that's the only
// useful answer for a premultiplied pixel with zero alpha — the color
// channels are already zero, but we defensively zero anyway).

#[autoversion]
fn unpremul_u8_ga(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[u8; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [u8; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1];
        if a == 0 {
            d[0] = 0;
            d[1] = 0;
        } else {
            let a32 = a as u32;
            d[0] = ((s[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            d[1] = a;
        }
    }
}

#[autoversion]
fn unpremul_u8_rgba(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[u8; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [u8; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        let a = s[3];
        if a == 0 {
            d[0] = 0;
            d[1] = 0;
            d[2] = 0;
            d[3] = 0;
        } else {
            let a32 = a as u32;
            d[0] = ((s[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            d[1] = ((s[1] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            d[2] = ((s[2] as u32 * 255 + a32 / 2) / a32).min(255) as u8;
            d[3] = a;
        }
    }
}

#[autoversion]
fn unpremul_u16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[u16; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [u16; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1];
        if a == 0 {
            d[0] = 0;
            d[1] = 0;
        } else {
            let a32 = a as u32;
            d[0] = ((s[0] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16;
            d[1] = a;
        }
    }
}

#[autoversion]
fn unpremul_u16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[u16; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        let a = s[3];
        if a == 0 {
            d[0] = 0;
            d[1] = 0;
            d[2] = 0;
            d[3] = 0;
        } else {
            let a32 = a as u32;
            d[0] = ((s[0] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16;
            d[1] = ((s[1] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16;
            d[2] = ((s[2] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16;
            d[3] = a;
        }
    }
}

#[autoversion]
fn unpremul_f32_ga(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        let base = i * 2;
        let s: &[f32; 2] = (&src[base..base + 2]).try_into().unwrap();
        let d: &mut [f32; 2] = (&mut dst[base..base + 2]).try_into().unwrap();
        let a = s[1];
        if a == 0.0 {
            d[0] = 0.0;
            d[1] = 0.0;
        } else {
            d[0] = s[0] / a;
            d[1] = a;
        }
    }
}

/// F16 unpremul via 3-pass scratch. `a == 0` branch preserved — produces
/// all-zero pixels for that case, otherwise divides RGB by alpha.
fn unpremul_f16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    const CHUNK_PIXELS: usize = 16;
    const CHUNK_LANES: usize = CHUNK_PIXELS * 2;

    let mut scratch_src = [0.0f32; CHUNK_LANES];
    let mut scratch_dst = [0.0f32; CHUNK_LANES];

    let whole = width / CHUNK_PIXELS;
    for c in 0..whole {
        let start = c * CHUNK_LANES;
        f16_bits_to_f32_slice(&src[start..start + CHUNK_LANES], &mut scratch_src);

        for i in 0..CHUNK_PIXELS {
            let base = i * 2;
            let a = scratch_src[base + 1];
            if a == 0.0 {
                scratch_dst[base] = 0.0;
                scratch_dst[base + 1] = 0.0;
            } else {
                let inv_a = 1.0 / a;
                scratch_dst[base] = scratch_src[base] * inv_a;
                scratch_dst[base + 1] = a;
            }
        }

        f32_to_f16_bits_slice(&scratch_dst, &mut dst[start..start + CHUNK_LANES]);
    }

    let tail_start = whole * CHUNK_PIXELS;
    for i in tail_start..width {
        let base = i * 2;
        let a = f16_bits_to_f32(src[base + 1]);
        if a == 0.0 {
            dst[base] = 0;
            dst[base + 1] = 0;
        } else {
            let inv_a = 1.0 / a;
            dst[base] = f32_to_f16_bits(f16_bits_to_f32(src[base]) * inv_a);
            dst[base + 1] = src[base + 1];
        }
    }
}

fn unpremul_f16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    const CHUNK_PIXELS: usize = 8;
    const CHUNK_LANES: usize = CHUNK_PIXELS * 4;

    let mut scratch_src = [0.0f32; CHUNK_LANES];
    let mut scratch_dst = [0.0f32; CHUNK_LANES];

    let whole = width / CHUNK_PIXELS;
    for c in 0..whole {
        let start = c * CHUNK_LANES;
        f16_bits_to_f32_slice(&src[start..start + CHUNK_LANES], &mut scratch_src);

        for i in 0..CHUNK_PIXELS {
            let base = i * 4;
            let a = scratch_src[base + 3];
            if a == 0.0 {
                scratch_dst[base] = 0.0;
                scratch_dst[base + 1] = 0.0;
                scratch_dst[base + 2] = 0.0;
                scratch_dst[base + 3] = 0.0;
            } else {
                let inv_a = 1.0 / a;
                scratch_dst[base] = scratch_src[base] * inv_a;
                scratch_dst[base + 1] = scratch_src[base + 1] * inv_a;
                scratch_dst[base + 2] = scratch_src[base + 2] * inv_a;
                scratch_dst[base + 3] = a;
            }
        }

        f32_to_f16_bits_slice(&scratch_dst, &mut dst[start..start + CHUNK_LANES]);
    }

    let tail_start = whole * CHUNK_PIXELS;
    for i in tail_start..width {
        let base = i * 4;
        let a = f16_bits_to_f32(src[base + 3]);
        if a == 0.0 {
            dst[base] = 0;
            dst[base + 1] = 0;
            dst[base + 2] = 0;
            dst[base + 3] = 0;
        } else {
            let inv_a = 1.0 / a;
            dst[base] = f32_to_f16_bits(f16_bits_to_f32(src[base]) * inv_a);
            dst[base + 1] = f32_to_f16_bits(f16_bits_to_f32(src[base + 1]) * inv_a);
            dst[base + 2] = f32_to_f16_bits(f16_bits_to_f32(src[base + 2]) * inv_a);
            dst[base + 3] = src[base + 3];
        }
    }
}

/// Premultiplied → Straight alpha.
fn premul_to_straight(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    match (ch_type, channels) {
        // Garb fast path (SIMD, f32 RGBA).
        (ChannelType::F32, 4) => {
            let n = width * 16;
            garb::bytes::unpremultiply_alpha_f32_copy(&src[..n], &mut dst[..n])
                .expect("pre-validated row size");
        }
        // Per-type autoversion kernels.
        (ChannelType::U8, 2) => unpremul_u8_ga(&src[..width * 2], &mut dst[..width * 2], width),
        (ChannelType::U8, 4) => unpremul_u8_rgba(&src[..width * 4], &mut dst[..width * 4], width),
        (ChannelType::U16, 2) => {
            let n = width * 4;
            unpremul_u16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::U16, 4) => {
            let n = width * 8;
            unpremul_u16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F32, 2) => {
            let n = width * 8;
            unpremul_f32_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F16, 2) => {
            let n = width * 4;
            unpremul_f16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        (ChannelType::F16, 4) => {
            let n = width * 8;
            unpremul_f16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                width,
            );
        }
        _ => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

// ---------------------------------------------------------------------------
// Oklab conversion kernels
// ---------------------------------------------------------------------------

use crate::oklab::{lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab};

/// Linear RGB f32 → Oklab f32 (3 channels).
///
/// # Panics
///
/// Panics if `primaries` is `Unknown`. The plan should have rejected this.
fn linear_rgb_to_oklab_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1 = rgb_to_lms_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
    rgb_to_oklab_3ch_inner(srcf, dstf, &m1);
}

/// 3-channel RGB→Oklab inner loop (16 pixels = 48 f32s per chunk).
#[autoversion]
fn rgb_to_oklab_3ch_inner(src: &[f32], dst: &mut [f32], m1: &[[f32; 3]; 3]) {
    // 16 pixels × 3 channels = 48 f32s = 192 bytes
    for (s, d) in src
        .as_chunks::<48>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<48>().0)
    {
        for p in 0..16 {
            let i = p * 3;
            let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
            d[i] = l;
            d[i + 1] = a;
            d[i + 2] = b;
        }
    }
    let rem_pixels = (src.len() / 3) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 3;
        for p in 0..rem_pixels {
            let i = off + p * 3;
            let [l, a, b] = rgb_to_oklab(src[i], src[i + 1], src[i + 2], m1);
            dst[i] = l;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }
}

/// Oklab f32 → Linear RGB f32 (3 channels).
fn oklab_to_linear_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
    oklab_to_rgb_3ch_inner(srcf, dstf, &m1_inv);
}

/// 3-channel Oklab→RGB inner loop (16 pixels = 48 f32s per chunk).
#[autoversion]
fn oklab_to_rgb_3ch_inner(src: &[f32], dst: &mut [f32], m1_inv: &[[f32; 3]; 3]) {
    for (s, d) in src
        .as_chunks::<48>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<48>().0)
    {
        for p in 0..16 {
            let i = p * 3;
            let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
        }
    }
    let rem_pixels = (src.len() / 3) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 3;
        for p in 0..rem_pixels {
            let i = off + p * 3;
            let [r, g, b] = oklab_to_rgb(src[i], src[i + 1], src[i + 2], m1_inv);
            dst[i] = r;
            dst[i + 1] = g;
            dst[i + 2] = b;
        }
    }
}

/// Linear RGBA f32 → Oklaba f32 (4 channels, alpha preserved).
fn linear_rgba_to_oklaba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1 = rgb_to_lms_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
    rgb_to_oklab_4ch_inner(srcf, dstf, &m1);
}

/// 4-channel RGBA→Oklaba inner loop (16 pixels = 64 f32s per chunk).
#[autoversion]
fn rgb_to_oklab_4ch_inner(src: &[f32], dst: &mut [f32], m1: &[[f32; 3]; 3]) {
    for (s, d) in src
        .as_chunks::<64>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
    {
        for p in 0..16 {
            let i = p * 4;
            let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
            d[i] = l;
            d[i + 1] = a;
            d[i + 2] = b;
            d[i + 3] = s[i + 3]; // alpha unchanged
        }
    }
    let rem_pixels = (src.len() / 4) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 4;
        for p in 0..rem_pixels {
            let i = off + p * 4;
            let [l, a, b] = rgb_to_oklab(src[i], src[i + 1], src[i + 2], m1);
            dst[i] = l;
            dst[i + 1] = a;
            dst[i + 2] = b;
            dst[i + 3] = src[i + 3];
        }
    }
}

/// Oklaba f32 → Linear RGBA f32 (4 channels, alpha preserved).
fn oklaba_to_linear_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
    oklab_to_rgb_4ch_inner(srcf, dstf, &m1_inv);
}

/// 4-channel Oklaba→RGBA inner loop (16 pixels = 64 f32s per chunk).
#[autoversion]
fn oklab_to_rgb_4ch_inner(src: &[f32], dst: &mut [f32], m1_inv: &[[f32; 3]; 3]) {
    for (s, d) in src
        .as_chunks::<64>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
    {
        for p in 0..16 {
            let i = p * 4;
            let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
            d[i + 3] = s[i + 3]; // alpha unchanged
        }
    }
    let rem_pixels = (src.len() / 4) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 4;
        for p in 0..rem_pixels {
            let i = off + p * 4;
            let [r, g, b] = oklab_to_rgb(src[i], src[i + 1], src[i + 2], m1_inv);
            dst[i] = r;
            dst[i + 1] = g;
            dst[i + 2] = b;
            dst[i + 3] = src[i + 3];
        }
    }
}

// ---------------------------------------------------------------------------
// Gamut matrix kernels
// ---------------------------------------------------------------------------

// --- SIMD 3×3 gamut matrix on linear-light f32 ------------------------------
//
// Same data shape as `multiply_color_channels_tier` (3 channels × N pixels
// for RGB; 4 channels with alpha passthrough for RGBA), but with a full
// 3×3 matrix rather than a per-channel scalar. Deinterleave 8 pixels at a
// time into r/g/b SIMD lanes, FMA the matrix rows, re-interleave; alpha
// (RGBA) is copied through unmodified. Tail pixels go through the scalar
// `mat3x3` helper for bit-exact remainder handling.
//
// magetypes' `f32x8` covers `v4 (cfg(avx512))` / `v3` (AVX2) / `neon`
// (aarch64) / `wasm128` / `scalar` — the same five-tier dispatch every
// other SIMD kernel in this file already uses. `incant!` picks the best
// available token at call time, and the `v4(cfg(avx512))` arm pulls in
// 16-wide AVX-512 only when the crate is built with `--features avx512`
// (8-wide AVX2 is the default on x86_64; AVX-512 widens the chunk to 16
// pixels per iter automatically with the same source).

/// Per-tier SIMD body for [`gamut_matrix_rgb_f32`] / [`gamut_matrix_rgba_f32`]
/// — `CHANNELS == 3` is interleaved RGB (no alpha), `CHANNELS == 4` carries an
/// alpha lane that's copied through unmodified.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn gamut_matrix_f32_tier<const CHANNELS: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    matrix: &[f32; 9],
) {
    const LANES: usize = 8;
    let m = matrix;
    let m00 = f32x8::splat(token, m[0]);
    let m01 = f32x8::splat(token, m[1]);
    let m02 = f32x8::splat(token, m[2]);
    let m10 = f32x8::splat(token, m[3]);
    let m11 = f32x8::splat(token, m[4]);
    let m12 = f32x8::splat(token, m[5]);
    let m20 = f32x8::splat(token, m[6]);
    let m21 = f32x8::splat(token, m[7]);
    let m22 = f32x8::splat(token, m[8]);

    let whole = width / LANES;
    let mut r = [0.0f32; LANES];
    let mut g = [0.0f32; LANES];
    let mut b = [0.0f32; LANES];

    for c in 0..whole {
        let base = c * LANES * CHANNELS;
        // Deinterleave 8 RGB/RGBA pixels.
        for i in 0..LANES {
            let p = base + i * CHANNELS;
            r[i] = src[p];
            g[i] = src[p + 1];
            b[i] = src[p + 2];
        }
        let rl = f32x8::from_array(token, r);
        let gl = f32x8::from_array(token, g);
        let bl = f32x8::from_array(token, b);

        // 3×3 matmul as fused-multiply-adds — same shape as
        // fast_gamut::mat3x3_x8 but typed against the generic
        // magetypes `f32x8`.
        let or = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
        let og = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
        let ob = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

        let ro = or.to_array();
        let go = og.to_array();
        let bo = ob.to_array();

        // Re-interleave; for RGBA, copy the alpha lane through.
        for i in 0..LANES {
            let p = base + i * CHANNELS;
            dst[p] = ro[i];
            dst[p + 1] = go[i];
            dst[p + 2] = bo[i];
            if CHANNELS == 4 {
                dst[p + 3] = src[p + 3];
            }
        }
    }

    // Scalar tail (1..LANES pixels).
    for p in (whole * LANES)..width {
        let base = p * CHANNELS;
        let rv = src[base];
        let gv = src[base + 1];
        let bv = src[base + 2];
        dst[base] = m[0] * rv + m[1] * gv + m[2] * bv;
        dst[base + 1] = m[3] * rv + m[4] * gv + m[5] * bv;
        dst[base + 2] = m[6] * rv + m[7] * gv + m[8] * bv;
        if CHANNELS == 4 {
            dst[base + 3] = src[base + 3];
        }
    }
}

/// Apply a 3×3 gamut matrix to a row of linear RGB f32 pixels. SIMD-dispatched
/// (AVX-512 / AVX2 / NEON / WASM-SIMD128 / Scalar) via `archmage::incant!`.
fn gamut_matrix_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    incant!(
        gamut_matrix_f32_tier::<3>(s, d, width, matrix),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// Apply a 3×3 gamut matrix to a row of linear RGBA f32 pixels (alpha
/// passthrough). SIMD-dispatched via `archmage::incant!` — same kernel as
/// the RGB variant, parameterised on `CHANNELS == 4`.
fn gamut_matrix_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    incant!(
        gamut_matrix_f32_tier::<4>(s, d, width, matrix),
        [v4, v3, neon, wasm128, scalar]
    );
}
