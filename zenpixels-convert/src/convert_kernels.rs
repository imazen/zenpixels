//! Row-level pixel conversion kernel implementations.
//!
//! Each kernel converts one row of pixels for a single conversion step.
//! Called from the step dispatcher in the parent `convert` module.

use core::cmp::min;

use archmage::prelude::*;

use crate::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor};
use crate::policy::LumaCoefficients;

use super::ConvertStep;
use crate::TransferFunction;
use crate::f16_scalar::{
    f16_bits_to_f32, f16_bits_to_f32_slice, f32_to_f16_bits, f32_to_f16_bits_slice,
};

/// IEEE 754 half-precision encoding of 1.0: sign=0, exponent=01111 (bias 15),
/// mantissa=0000000000 → bits 0b0_01111_0000000000 = 0x3C00.
const F16_ONE_BITS: u16 = 0x3C00;

/// Apply a single conversion step on raw byte slices.
pub(super) fn apply_step_u8(
    step: &ConvertStep,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    from: PixelDescriptor,
    _to: PixelDescriptor,
) {
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
            srgb_u8_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToSrgbU8 => {
            linear_f32_to_srgb_u8(src, dst, w, from.layout().channels());
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
            pq_u16_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToPqU16 => {
            linear_f32_to_pq_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::PqF32ToLinearF32 => {
            pq_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToPqF32 => {
            linear_f32_to_pq_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::HlgU16ToLinearF32 => {
            hlg_u16_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToHlgU16 => {
            linear_f32_to_hlg_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::HlgF32ToLinearF32 => {
            hlg_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToHlgF32 => {
            linear_f32_to_hlg_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::SrgbF32ToLinearF32 => {
            srgb_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToSrgbF32 => {
            linear_f32_to_srgb_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::SrgbF32ToLinearF32Extended => {
            srgb_f32_to_linear_f32_extended(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToSrgbF32Extended => {
            linear_f32_to_srgb_f32_extended(src, dst, w, from.layout().channels());
        }

        ConvertStep::Bt709F32ToLinearF32 => {
            bt709_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToBt709F32 => {
            linear_f32_to_bt709_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::Gamma22F32ToLinearF32 => {
            gamma22_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToGamma22F32 => {
            linear_f32_to_gamma22_f32(src, dst, w, from.layout().channels());
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

        ConvertStep::FusedSrgbU8GamutRgb(flat) => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
            crate::fast_gamut::convert_u8_rgb_simd_matlut(
                &m,
                src,
                dst,
                crate::fast_gamut::srgb_lin_lut_u8(),
                |v: f32| linear_srgb::default::linear_to_srgb_u8(v),
            );
        }

        ConvertStep::FusedSrgbU8GamutRgba(flat) => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
            crate::fast_gamut::convert_u8_rgba_simd_lut(
                &m,
                src,
                dst,
                crate::fast_gamut::srgb_lin_lut_u8(),
                linear_srgb::default::linear_to_srgb_u8,
            );
        }

        ConvertStep::FusedSrgbU16GamutRgb(flat) => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
            let src_u16: &[u16] = bytemuck::cast_slice(src);
            let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
            // LUT decode (256 KB lin_lut) + SIMD matrix + LUT encode
            // (128 KB, linearly-indexed). Pending linear-srgb 0.6.12:
            // switch to `convert_u16_rgb_simd_lutdec_polyenc` for +17%
            // at 1080p and exact u16 roundtrip (it needs
            // linear_to_srgb_u16_v3 rite which is unreleased).
            crate::fast_gamut::convert_u16_rgb_simd_matlut(
                &m,
                src_u16,
                dst_u16,
                crate::fast_gamut::srgb_lin_lut_u16(),
                crate::fast_gamut::srgb_enc_lut_u16(),
            );
        }

        ConvertStep::FusedSrgbU8ToLinearF32Rgb(flat) => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
            let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
            crate::fast_gamut::convert_u8_to_f32_lin_simd(
                &m,
                src,
                dst_f32,
                crate::fast_gamut::srgb_lin_lut_u8(),
            );
        }

        ConvertStep::FusedLinearF32ToSrgbU8Rgb(flat) => {
            let m = [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ];
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
// Depth conversion kernels (transfer-function-aware)
// ---------------------------------------------------------------------------

/// sRGB u8 → linear f32 using `linear-srgb` SIMD batch conversion.
fn srgb_u8_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_srgb::default::srgb_u8_to_linear_slice(&src[..count], dstf);
}

/// Linear f32 → sRGB u8 using `linear-srgb` SIMD batch conversion.
fn linear_f32_to_srgb_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    linear_srgb::default::linear_to_srgb_u8_slice(srcf, &mut dst[..count]);
}

/// Naive u8 → f32 (v / 255.0, no transfer function).
fn naive_u8_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u8_to_f32(&src[..count], &mut dst[..count * 4])
        .expect("pre-validated row size");
}

/// Naive f32 → u8 (clamp [0,1], * 255 + 0.5).
fn naive_f32_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_f32_to_u8(&src[..count * 4], &mut dst[..count])
        .expect("pre-validated row size");
}

/// u16 → u8: (v * 255 + 32768) >> 16.
fn u16_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u16_to_u8(&src[..count * 2], &mut dst[..count])
        .expect("pre-validated row size");
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

/// f32 → u16: clamp [0,1], * 65535 + 0.5.
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

/// PQ EOTF: encoded [0,1] → linear light [0,1] (where 1.0 = 10000 cd/m²).
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    linear_srgb::tf::pq_to_linear(v)
}

/// PQ inverse EOTF (OETF): linear light [0,1] → encoded [0,1].
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_pq(v)
}

/// PQ U16 → Linear F32 (EOTF applied during depth conversion).
fn pq_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    pq_u16_to_linear_f32_inner(src16, dstf);
}

#[autoversion]
fn pq_u16_to_linear_f32_inner(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            d[i] = linear_srgb::tf::pq_to_linear(s[i] as f32 / 65535.0);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::pq_to_linear(src[off + i] as f32 / 65535.0);
        }
    }
}

/// Linear F32 → PQ U16 (OETF applied during depth conversion).
fn linear_f32_to_pq_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    linear_f32_to_pq_u16_inner(srcf, dst16);
}

#[autoversion]
fn linear_f32_to_pq_u16_inner(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            let encoded = linear_srgb::tf::linear_to_pq(s[i].max(0.0));
            d[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            let encoded = linear_srgb::tf::linear_to_pq(src[off + i].max(0.0));
            dst[off + i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
}

/// PQ F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
fn pq_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::pq_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → PQ F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_pq_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_pq_slice(&mut dstf[..count]);
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) transfer function — delegates to linear-srgb
// ---------------------------------------------------------------------------

/// HLG OETF: scene-linear [0,1] → encoded [0,1].
///
/// Uses `fast_log2f` from `linear-srgb` (no `libm` ln calls).
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_hlg(v)
}

/// HLG inverse OETF (EOTF): encoded [0,1] → scene-linear [0,1].
///
/// Uses `fast_pow2f` from `linear-srgb` (no `libm` exp calls).
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    linear_srgb::tf::hlg_to_linear(v)
}

/// HLG U16 → Linear F32 (EOTF applied during depth conversion).
fn hlg_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    hlg_u16_to_linear_f32_inner(src16, dstf);
}

#[autoversion]
fn hlg_u16_to_linear_f32_inner(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
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

/// Linear F32 → HLG U16 (OETF applied during depth conversion).
fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    linear_f32_to_hlg_u16_inner(srcf, dst16);
}

#[autoversion]
fn linear_f32_to_hlg_u16_inner(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
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

/// HLG F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
fn hlg_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::hlg_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → HLG F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_hlg_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_hlg_slice(&mut dstf[..count]);
}

// ---------------------------------------------------------------------------
// sRGB / BT.709 F32 ↔ Linear F32 transfer function kernels
// ---------------------------------------------------------------------------

/// sRGB F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
/// Clamps to [0, 1] — use `srgb_to_linear_extended_slice` for HDR/WCG workflows
/// that need to preserve out-of-gamut values (pending configurable option).
fn srgb_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::srgb_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → sRGB F32 (OETF, same depth). SIMD-dispatched.
/// Clamps to [0, 1] — use `linear_to_srgb_extended_slice` for HDR/WCG workflows
/// that need to preserve out-of-gamut values (pending configurable option).
fn linear_f32_to_srgb_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_srgb_slice(&mut dstf[..count]);
}

/// sRGB F32 → Linear F32 (extended range, sign-preserving).
fn srgb_f32_to_linear_f32_extended(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::srgb_to_linear_extended_slice(&mut dstf[..count]);
}

/// Linear F32 → sRGB F32 (extended range, sign-preserving).
fn linear_f32_to_srgb_f32_extended(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_srgb_extended_slice(&mut dstf[..count]);
}

/// BT.709 F32 → Linear F32 (EOTF, same depth).
fn bt709_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    bt709_f32_to_linear_f32_inner(&srcf[..count], &mut dstf[..count]);
}

#[autoversion]
fn bt709_f32_to_linear_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
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

/// Linear F32 → BT.709 F32 (OETF, same depth).
fn linear_f32_to_bt709_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_f32_to_bt709_f32_inner(&srcf[..count], &mut dstf[..count]);
}

#[autoversion]
fn linear_f32_to_bt709_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
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
/// `linear_srgb::default::gamma_to_linear_slice`.
fn gamma22_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::gamma_to_linear_slice(&mut dstf[..count], ADOBE_GAMMA);
}

/// Linear F32 → Gamma 2.2 F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_gamma22_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_gamma_slice(&mut dstf[..count], ADOBE_GAMMA);
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
    for (s, d) in src.chunks_exact(48).zip(dst.chunks_exact_mut(48)) {
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
    for (s, d) in src.chunks_exact(48).zip(dst.chunks_exact_mut(48)) {
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
    for (s, d) in src.chunks_exact(64).zip(dst.chunks_exact_mut(64)) {
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
    for (s, d) in src.chunks_exact(64).zip(dst.chunks_exact_mut(64)) {
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

/// Apply a 3×3 gamut matrix to a row of linear RGB f32 pixels.
fn gamut_matrix_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    let m = matrix;
    for p in 0..width {
        let base = p * 3;
        let r = s[base];
        let g = s[base + 1];
        let b = s[base + 2];
        d[base] = m[0] * r + m[1] * g + m[2] * b;
        d[base + 1] = m[3] * r + m[4] * g + m[5] * b;
        d[base + 2] = m[6] * r + m[7] * g + m[8] * b;
    }
}

/// Apply a 3×3 gamut matrix to a row of linear RGBA f32 pixels (alpha passthrough).
fn gamut_matrix_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    let m = matrix;
    for p in 0..width {
        let base = p * 4;
        let r = s[base];
        let g = s[base + 1];
        let b = s[base + 2];
        d[base] = m[0] * r + m[1] * g + m[2] * b;
        d[base + 1] = m[3] * r + m[4] * g + m[5] * b;
        d[base + 2] = m[6] * r + m[7] * g + m[8] * b;
        d[base + 3] = s[base + 3];
    }
}
