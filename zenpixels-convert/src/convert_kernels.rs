//! Row-level pixel conversion kernel implementations.
//!
//! Each kernel converts one row of pixels for a single conversion step.
//! Called from the step dispatcher in the parent `convert` module.

use core::cmp::min;

use archmage::prelude::*;

#[cfg(feature = "oklab")]
use crate::ColorPrimaries;
#[cfg(any(feature = "hdr-experimental", feature = "oklab"))]
use crate::PixelDescriptor;
use crate::policy::LumaCoefficients;
use crate::{ChannelLayout, ChannelType};

use super::{ConvertStep, PlannedOperation};
use crate::TransferFunction;
use crate::accel::kernel_caps as caps;
use crate::f16_scalar::{f16_bits_to_f32_slice, f32_to_f16_bits_slice};

/// IEEE 754 half-precision encoding of 1.0: sign=0, exponent=01111 (bias 15),
/// mantissa=0000000000 → bits 0b0_01111_0000000000 = 0x3C00.
const F16_ONE_BITS: u16 = 0x3C00;

/// Scratch state reused by the HDR tone-map kernels across rows.
///
/// Owned by [`ConvertScratch`](super::ConvertScratch) so a plan/strip run
/// allocates once and every subsequent row reuses it — keeping the step
/// kernels free of per-row heap allocation (the module contract).
#[derive(Default)]
pub(super) struct HdrKernelScratch {
    /// RGB-triple strip for the RGBA carrier paths — the SIMD curves want a
    /// contiguous `[[f32; 3]]` view, so alpha is peeled into this scratch.
    /// Grow-only; sliced to the row width on each use.
    rgb_strip: alloc::vec::Vec<[f32; 3]>,
    /// Output of a pluggable tone mapper, which doesn't work in place.
    rgb_out: alloc::vec::Vec<[f32; 3]>,
}

impl HdrKernelScratch {
    pub(super) fn reserve(
        &mut self,
        lengths: [usize; 2],
    ) -> Result<(), whereat::At<crate::ConvertError>> {
        if self.rgb_strip.len() < lengths[0] {
            self.rgb_strip = bytemuck::allocation::try_zeroed_vec::<[f32; 3]>(lengths[0])
                .map_err(|()| whereat::at!(crate::ConvertError::AllocationFailed))?;
        }
        if self.rgb_out.len() < lengths[1] {
            self.rgb_out = bytemuck::allocation::try_zeroed_vec::<[f32; 3]>(lengths[1])
                .map_err(|()| whereat::at!(crate::ConvertError::AllocationFailed))?;
        }
        Ok(())
    }
}

/// Bound single-step row executor stored directly on each [`PlannedOperation`].
pub(super) type StepFn = fn(&PlannedOperation, &[u8], &mut [u8], u32, &mut HdrKernelScratch);

/// In-place `f32` transfer adapter used by bound matte-composite kernels.
pub(super) type SliceTfFn = fn(&mut [f32]);

/// Sentinel executor for unbound plans; never reached after `NativePlan::bind()`.
pub(super) fn step_unbound(
    _op: &PlannedOperation,
    _src: &[u8],
    _dst: &mut [u8],
    _width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    unreachable!("NativePlan executed without bind()")
}

/// No-op identity copy step.
pub(super) fn step_identity(
    _op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    _width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let len = min(src.len(), dst.len());
    dst[..len].copy_from_slice(&src[..len]);
}

/// No-op transfer adapter for linear/passthrough matte compositing.
pub(super) fn noop_f32_slice(_: &mut [f32]) {}

/// Resolve and bind the executable [`StepFn`] for `op` under compile-time mask `CAPS`.
///
/// Steps sharing an exact capability equivalence class share one `StepFn`,
/// preserving bit-exact dead-code elimination per capability bit while keeping
/// wrapper/relocation/EH overhead minimal.
#[inline(always)]
#[track_caller]
pub(super) fn resolve_and_bind_op<const CAPS: u64>(
    op: &mut PlannedOperation,
) -> Result<(), whereat::At<crate::ConvertError>> {
    let from = op.from;
    let to = op.to;
    let ch_type = from.channel_type();

    let exec: StepFn = match &mut op.step {
        ConvertStep::Identity => step_identity,

        #[cfg(feature = "hdr-transfers")]
        ConvertStep::HlgRender(state) => {
            if (CAPS & (caps::F32 | caps::TF_HLG)) == (caps::F32 | caps::TF_HLG) {
                if state.prepared.is_none() {
                    state.prepared = Some(super::hlg_render::PreparedHlg::new(
                        state.rendering,
                        state.inverse,
                        state.white_nits,
                        state.clip,
                    ));
                }
                step_hlg_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::SwizzleBgraRgba
        | ConvertStep::SwizzleBgrRgb
        | ConvertStep::AddAlpha
        | ConvertStep::RgbToBgra
        | ConvertStep::DropAlpha => match ch_type {
            ChannelType::U8 if (CAPS & (caps::U8 | caps::LAYOUT)) == (caps::U8 | caps::LAYOUT) => {
                step_layout_u8
            }
            ChannelType::U16
                if (CAPS & (caps::U16 | caps::LAYOUT)) == (caps::U16 | caps::LAYOUT) =>
            {
                step_layout_u16
            }
            ChannelType::F16
                if (CAPS & (caps::F16 | caps::LAYOUT)) == (caps::F16 | caps::LAYOUT) =>
            {
                step_layout_f16
            }
            ChannelType::F32
                if (CAPS & (caps::F32 | caps::LAYOUT)) == (caps::F32 | caps::LAYOUT) =>
            {
                step_layout_f32
            }
            _ => return Err(super::err_no_path(from, to)),
        },

        ConvertStep::MatteComposite(state) => {
            if (CAPS & caps::MATTE) == 0 {
                return Err(super::err_no_path(from, to));
            }
            let tf = from.transfer();
            let is_premul = from.alpha() == Some(zenpixels::AlphaMode::Premultiplied);
            let srgb_u8 = ch_type == ChannelType::U8 && tf == TransferFunction::Srgb && !is_premul;
            let clip_f32 = state.clip || ch_type != ChannelType::F32;

            let (depth_exec, to_lin, from_lin): (StepFn, SliceTfFn, SliceTfFn) = if srgb_u8 {
                if (CAPS & (caps::U8 | caps::TF_SRGB)) == (caps::U8 | caps::TF_SRGB) {
                    (step_matte_srgb_u8, noop_f32_slice, noop_f32_slice)
                } else {
                    return Err(super::err_no_path(from, to));
                }
            } else {
                let (to_lin, from_lin): (SliceTfFn, SliceTfFn) = match tf {
                    TransferFunction::Linear | TransferFunction::Unknown => {
                        (noop_f32_slice, noop_f32_slice)
                    }
                    TransferFunction::Srgb if clip_f32 => {
                        if (CAPS & caps::TF_SRGB) == caps::TF_SRGB {
                            (tf_srgb_to_lin, tf_lin_to_srgb)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    TransferFunction::Srgb => {
                        if (CAPS & (caps::TF_SRGB | caps::TF_EXTENDED))
                            == (caps::TF_SRGB | caps::TF_EXTENDED)
                        {
                            (tf_srgb_ext_to_lin, tf_lin_to_srgb_ext)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    TransferFunction::Bt709 => {
                        if (CAPS & caps::TF_BT709) == caps::TF_BT709 {
                            (tf_bt709_to_lin, tf_lin_to_bt709)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    #[cfg(feature = "hdr-transfers")]
                    TransferFunction::Pq => {
                        if (CAPS & caps::TF_PQ) == caps::TF_PQ {
                            (pq_eotf_slice, pq_oetf_slice)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    #[cfg(feature = "hdr-transfers")]
                    TransferFunction::Hlg => {
                        if (CAPS & caps::TF_HLG) == caps::TF_HLG {
                            (tf_hlg_to_lin, tf_lin_to_hlg)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    #[cfg(not(feature = "hdr-transfers"))]
                    TransferFunction::Pq | TransferFunction::Hlg => {
                        return Err(whereat::at!(crate::ConvertError::FeatureRequired {
                            feature: "hdr-transfers",
                            from,
                            to,
                        }));
                    }
                    TransferFunction::Gamma22 if clip_f32 => {
                        if (CAPS & caps::TF_GAMMA22) == caps::TF_GAMMA22 {
                            (tf_gamma22_to_lin, tf_lin_to_gamma22)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    TransferFunction::Gamma22 => {
                        if (CAPS & (caps::TF_GAMMA22 | caps::TF_EXTENDED))
                            == (caps::TF_GAMMA22 | caps::TF_EXTENDED)
                        {
                            (tf_gamma22_ext_to_lin, tf_lin_to_gamma22_ext)
                        } else {
                            return Err(super::err_no_path(from, to));
                        }
                    }
                    _ => (noop_f32_slice, noop_f32_slice),
                };
                let depth_exec: StepFn = match ch_type {
                    ChannelType::U8 if (CAPS & caps::U8) == caps::U8 => step_matte_u8,
                    ChannelType::U16 if (CAPS & caps::U16) == caps::U16 => step_matte_u16,
                    ChannelType::F16 if (CAPS & caps::F16) == caps::F16 => step_matte_f16,
                    ChannelType::F32 if (CAPS & caps::F32) == caps::F32 => step_matte_f32,
                    _ => return Err(super::err_no_path(from, to)),
                };
                (depth_exec, to_lin, from_lin)
            };
            let mut m = super::matte_background_scalar::<CAPS>(
                state.background,
                state.reference_white,
                from,
            )?;
            if state.scale != 1.0 {
                for v in &mut m {
                    *v *= state.scale;
                }
            }
            state.linear = m;
            state.to_linear = to_lin;
            state.from_linear = from_lin;
            depth_exec
        }

        ConvertStep::GrayToRgb
        | ConvertStep::GrayToRgba
        | ConvertStep::RgbToGray { .. }
        | ConvertStep::RgbaToGray { .. }
        | ConvertStep::RgbaToGrayAlpha { .. }
        | ConvertStep::GrayAlphaToRgba
        | ConvertStep::GrayAlphaToRgb
        | ConvertStep::GrayToGrayAlpha
        | ConvertStep::GrayAlphaToGray => match ch_type {
            ChannelType::U8 if (CAPS & (caps::U8 | caps::GRAY)) == (caps::U8 | caps::GRAY) => {
                step_gray_u8
            }
            ChannelType::U16 if (CAPS & (caps::U16 | caps::GRAY)) == (caps::U16 | caps::GRAY) => {
                step_gray_u16
            }
            ChannelType::F16 if (CAPS & (caps::F16 | caps::GRAY)) == (caps::F16 | caps::GRAY) => {
                step_gray_f16
            }
            ChannelType::F32 if (CAPS & (caps::F32 | caps::GRAY)) == (caps::F32 | caps::GRAY) => {
                step_gray_f32
            }
            _ => return Err(super::err_no_path(from, to)),
        },

        ConvertStep::StraightToPremul | ConvertStep::PremulToStraight => match ch_type {
            ChannelType::U8 if (CAPS & (caps::U8 | caps::PREMUL)) == (caps::U8 | caps::PREMUL) => {
                step_premul_u8
            }
            ChannelType::U16
                if (CAPS & (caps::U16 | caps::PREMUL)) == (caps::U16 | caps::PREMUL) =>
            {
                step_premul_u16
            }
            ChannelType::F16
                if (CAPS & (caps::F16 | caps::PREMUL)) == (caps::F16 | caps::PREMUL) =>
            {
                step_premul_f16
            }
            ChannelType::F32
                if (CAPS & (caps::F32 | caps::PREMUL)) == (caps::F32 | caps::PREMUL) =>
            {
                step_premul_f32
            }
            _ => return Err(super::err_no_path(from, to)),
        },

        ConvertStep::SrgbU8ToLinearF32 | ConvertStep::LinearF32ToSrgbU8 => {
            if (CAPS & (caps::U8 | caps::F32 | caps::TF_SRGB))
                == (caps::U8 | caps::F32 | caps::TF_SRGB)
            {
                step_u8_f32_srgb
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::NaiveU8ToF32 | ConvertStep::NaiveF32ToU8 => {
            if (CAPS & (caps::U8 | caps::F32)) == (caps::U8 | caps::F32) {
                if (CAPS & caps::FLOAT_FACTS_SPEC) == caps::FLOAT_FACTS_SPEC {
                    step_u8_f32_naive_spec
                } else {
                    step_u8_f32_naive_untrusted
                }
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::NaiveF32ToU8Truncate => {
            if (CAPS & (caps::U8 | caps::F32 | caps::DEPTH_TRUNCATE))
                == (caps::U8 | caps::F32 | caps::DEPTH_TRUNCATE)
            {
                step_u8_f32_truncate
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::U16ToU8 | ConvertStep::U8ToU16 => {
            if (CAPS & (caps::U16 | caps::U8)) == (caps::U16 | caps::U8) {
                step_u8_u16
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::U16ToU8Truncate => {
            if (CAPS & (caps::U16 | caps::U8 | caps::DEPTH_TRUNCATE))
                == (caps::U16 | caps::U8 | caps::DEPTH_TRUNCATE)
            {
                step_u16_u8_truncate
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::SdrU16ToU8 { to: dst_tf } => {
            let from_tf = from.transfer();
            let needs_srgb = from_tf == TransferFunction::Srgb || *dst_tf == TransferFunction::Srgb;
            let needs_bt709 =
                from_tf == TransferFunction::Bt709 || *dst_tf == TransferFunction::Bt709;
            if (CAPS & (caps::U16 | caps::U8)) == (caps::U16 | caps::U8)
                && (!needs_srgb || (CAPS & caps::TF_SRGB) == caps::TF_SRGB)
                && (!needs_bt709 || (CAPS & caps::TF_BT709) == caps::TF_BT709)
            {
                step_sdr_u16_to_u8
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::U16ToF32 | ConvertStep::F32ToU16 => {
            if (CAPS & (caps::U16 | caps::F32)) == (caps::U16 | caps::F32) {
                if (CAPS & caps::FLOAT_FACTS_SPEC) == caps::FLOAT_FACTS_SPEC {
                    step_u16_f32_spec
                } else {
                    step_u16_f32_untrusted
                }
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::F32ToU16Truncate => {
            if (CAPS & (caps::U16 | caps::F32 | caps::DEPTH_TRUNCATE))
                == (caps::U16 | caps::F32 | caps::DEPTH_TRUNCATE)
            {
                step_u16_f32_truncate
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::F16ToF32 | ConvertStep::F32ToF16 => {
            if (CAPS & (caps::F16 | caps::F32)) == (caps::F16 | caps::F32) {
                step_f16_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "hdr-transfers")]
        ConvertStep::PqU16ToLinearF32 | ConvertStep::LinearF32ToPqU16 => {
            if (CAPS & (caps::U16 | caps::F32 | caps::TF_PQ))
                == (caps::U16 | caps::F32 | caps::TF_PQ)
            {
                step_pq_u16
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "hdr-transfers")]
        ConvertStep::PqF32ToLinearF32 | ConvertStep::LinearF32ToPqF32 => {
            if (CAPS & (caps::F32 | caps::TF_PQ)) == (caps::F32 | caps::TF_PQ) {
                step_pq_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "hdr-transfers")]
        ConvertStep::HlgU16ToLinearF32 | ConvertStep::LinearF32ToHlgU16 => {
            if (CAPS & (caps::U16 | caps::F32 | caps::TF_HLG))
                == (caps::U16 | caps::F32 | caps::TF_HLG)
            {
                step_hlg_u16
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "hdr-transfers")]
        ConvertStep::HlgF32ToLinearF32 | ConvertStep::LinearF32ToHlgF32 => {
            if (CAPS & (caps::F32 | caps::TF_HLG)) == (caps::F32 | caps::TF_HLG) {
                step_hlg_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::SrgbF32ToLinearF32 | ConvertStep::LinearF32ToSrgbF32 => {
            if (CAPS & (caps::F32 | caps::TF_SRGB)) == (caps::F32 | caps::TF_SRGB) {
                step_tf_srgb_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::SrgbF32ToLinearF32Extended | ConvertStep::LinearF32ToSrgbF32Extended => {
            if (CAPS & (caps::F32 | caps::TF_SRGB | caps::TF_EXTENDED))
                == (caps::F32 | caps::TF_SRGB | caps::TF_EXTENDED)
            {
                step_tf_srgb_ext_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::Bt709F32ToLinearF32 | ConvertStep::LinearF32ToBt709F32 => {
            if (CAPS & (caps::F32 | caps::TF_BT709)) == (caps::F32 | caps::TF_BT709) {
                step_tf_bt709_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::Gamma22F32ToLinearF32 | ConvertStep::LinearF32ToGamma22F32 => {
            if (CAPS & (caps::F32 | caps::TF_GAMMA22)) == (caps::F32 | caps::TF_GAMMA22) {
                step_tf_gamma22_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::Gamma22F32ToLinearF32Extended | ConvertStep::LinearF32ToGamma22F32Extended => {
            if (CAPS & (caps::F32 | caps::TF_GAMMA22 | caps::TF_EXTENDED))
                == (caps::F32 | caps::TF_GAMMA22 | caps::TF_EXTENDED)
            {
                step_tf_gamma22_ext_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "oklab")]
        ConvertStep::LinearRgbToOklab
        | ConvertStep::OklabToLinearRgb
        | ConvertStep::LinearRgbaToOklaba
        | ConvertStep::OklabaToLinearRgba => {
            if (CAPS & (caps::F32 | caps::OKLAB)) == (caps::F32 | caps::OKLAB) {
                if (CAPS & caps::FLOAT_FACTS_SPEC) == caps::FLOAT_FACTS_SPEC {
                    step_oklab_spec
                } else {
                    step_oklab_untrusted
                }
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::GamutMatrixRgbF32(..) | ConvertStep::GamutMatrixRgbaF32(..) => {
            if (CAPS & (caps::F32 | caps::GAMUT_MATRIX)) == (caps::F32 | caps::GAMUT_MATRIX) {
                step_gamut_matrix_f32
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        ConvertStep::Fused { kind, .. } => match kind {
            crate::convert::FusedKind::SrgbU8GamutRgb
            | crate::convert::FusedKind::SrgbU8GamutRgba => {
                if (CAPS & (caps::U8 | caps::TF_SRGB | caps::GAMUT_FUSED))
                    == (caps::U8 | caps::TF_SRGB | caps::GAMUT_FUSED)
                {
                    step_fused_u8_gamut
                } else {
                    return Err(super::err_no_path(from, to));
                }
            }
            crate::convert::FusedKind::SrgbU16GamutRgb => {
                if (CAPS & (caps::U16 | caps::TF_SRGB | caps::GAMUT_FUSED))
                    == (caps::U16 | caps::TF_SRGB | caps::GAMUT_FUSED)
                {
                    step_fused_u16_gamut
                } else {
                    return Err(super::err_no_path(from, to));
                }
            }
            crate::convert::FusedKind::SrgbU8ToLinearF32Rgb
            | crate::convert::FusedKind::LinearF32ToSrgbU8Rgb => {
                if (CAPS & (caps::U8 | caps::F32 | caps::TF_SRGB | caps::GAMUT_FUSED))
                    == (caps::U8 | caps::F32 | caps::TF_SRGB | caps::GAMUT_FUSED)
                {
                    step_fused_u8_f32_gamut
                } else {
                    return Err(super::err_no_path(from, to));
                }
            }
        },

        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMap(step) => {
            if (CAPS & (caps::F32 | caps::TONE_MAP)) == (caps::F32 | caps::TONE_MAP) {
                if step.mapper.is_none() {
                    #[cfg(feature = "bt2446a")]
                    {
                        step.mapper = Some(alloc::sync::Arc::new(crate::hdr::Bt2446A::new(
                            step.source_peak_nits,
                            step.target_peak_nits,
                        )));
                    }
                    #[cfg(not(feature = "bt2446a"))]
                    {
                        return Err(super::err_no_path(from, to));
                    }
                }
                step_tone_map
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "oklab")]
        ConvertStep::SoftCompressOklch(state) => {
            if (CAPS & (caps::F32 | caps::SOFT_COMPRESS)) == (caps::F32 | caps::SOFT_COMPRESS) {
                if state.prepared.is_none() {
                    let space = crate::oklab::OklabSpace::for_primaries(state.primaries)
                        .expect("target primaries have defined LMS matrices");
                    state.prepared = Some(alloc::sync::Arc::new(crate::hdr::SoftCompress::new(
                        &space, state.knee,
                    )));
                }
                step_soft_compress_oklch
            } else {
                return Err(super::err_no_path(from, to));
            }
        }

        #[cfg(feature = "perceptual")]
        ConvertStep::Perceptual(p) => {
            use super::perceptual::PerceptualStep;
            match p {
                PerceptualStep::CartesianToPolar { .. }
                | PerceptualStep::PolarToCartesian { .. }
                    if (CAPS & (caps::F32 | caps::PERCEPTUAL_POLAR))
                        == (caps::F32 | caps::PERCEPTUAL_POLAR) =>
                {
                    step_perceptual_polar
                }
                PerceptualStep::LinearRgbToJzazbz { .. }
                | PerceptualStep::JzazbzToLinearRgb { .. }
                | PerceptualStep::PqRgbToJzazbz { .. }
                | PerceptualStep::JzazbzToPqRgb { .. }
                    if (CAPS & (caps::F32 | caps::PERCEPTUAL_JZAZBZ))
                        == (caps::F32 | caps::PERCEPTUAL_JZAZBZ) =>
                {
                    step_perceptual_jzazbz
                }
                PerceptualStep::LinearRgbToLch { .. } | PerceptualStep::LchToLinearRgb { .. }
                    if (CAPS & (caps::F32 | caps::PERCEPTUAL_LCH))
                        == (caps::F32 | caps::PERCEPTUAL_LCH) =>
                {
                    step_perceptual_lch
                }
                PerceptualStep::RgbToHsl { .. }
                | PerceptualStep::HslToRgb { .. }
                | PerceptualStep::RgbToHsv { .. }
                | PerceptualStep::HsvToRgb { .. }
                    if (CAPS & (caps::F32 | caps::PERCEPTUAL_HSL_HSV))
                        == (caps::F32 | caps::PERCEPTUAL_HSL_HSV) =>
                {
                    step_perceptual_hsl_hsv
                }
                PerceptualStep::DecodeIntToF32 {
                    from_type: ChannelType::U8,
                    ..
                }
                | PerceptualStep::EncodeF32ToInt {
                    to_type: ChannelType::U8,
                    ..
                } if (CAPS & (caps::U8 | caps::F32 | caps::PERCEPTUAL_INT))
                    == (caps::U8 | caps::F32 | caps::PERCEPTUAL_INT) =>
                {
                    step_perceptual_int_u8
                }
                PerceptualStep::DecodeIntToF32 {
                    from_type: ChannelType::U16,
                    ..
                }
                | PerceptualStep::EncodeF32ToInt {
                    to_type: ChannelType::U16,
                    ..
                } if (CAPS & (caps::U16 | caps::F32 | caps::PERCEPTUAL_INT))
                    == (caps::U16 | caps::F32 | caps::PERCEPTUAL_INT) =>
                {
                    step_perceptual_int_u16
                }
                _ => return Err(super::err_no_path(from, to)),
            }
        }
    };

    op.exec = exec;
    Ok(())
}

// ---------------------------------------------------------------------------
// Capability-grouped bound executors
// ---------------------------------------------------------------------------

#[cfg(feature = "hdr-experimental")]
fn step_tone_map(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::ToneMap(step) = &op.step {
        tone_map_kernel(
            src,
            dst,
            width as usize,
            op.from,
            step,
            op.in_trust.color_facts,
            hdr,
        );
    }
}

#[cfg(feature = "oklab")]
fn step_soft_compress_oklch(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::SoftCompressOklch(state) = &op.step {
        let compressor = state
            .prepared
            .as_ref()
            .expect("SoftCompressState bound before execution");
        soft_compress_oklch_kernel(src, dst, width as usize, op.from, compressor, hdr);
    }
}

fn step_u8_f32_srgb(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::SrgbU8ToLinearF32 => {
            srgb_u8_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToSrgbU8 => {
            linear_f32_to_srgb_u8(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

fn step_u8_f32_naive_spec(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::NaiveU8ToF32 => naive_u8_to_f32(src, dst, width as usize, ch),
        ConvertStep::NaiveF32ToU8 => {
            naive_f32_to_u8(src, dst, width as usize, ch, op.in_trust.facts);
        }
        _ => {}
    }
}

fn step_u8_f32_naive_untrusted(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::NaiveU8ToF32 => naive_u8_to_f32(src, dst, width as usize, ch),
        ConvertStep::NaiveF32ToU8 => {
            let count = width as usize * ch;
            let src_f32: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
            crate::float_simd::convert_f32_to_u8_slice_untrusted(src_f32, &mut dst[..count]);
        }
        _ => {}
    }
}

fn step_u8_f32_truncate(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    naive_f32_to_u8_truncate(
        src,
        dst,
        width as usize,
        op.from.layout().channels(),
        op.in_trust.facts,
    );
}

fn step_u8_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::U16ToU8 => u16_to_u8(src, dst, width as usize, ch),
        ConvertStep::U8ToU16 => u8_to_u16(src, dst, width as usize, ch),
        _ => {}
    }
}

fn step_u16_u8_truncate(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    u16_to_u8_truncate(src, dst, width as usize, op.from.layout().channels());
}

fn step_sdr_u16_to_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::SdrU16ToU8 { to } = op.step {
        sdr_u16_to_u8(
            src,
            dst,
            width as usize,
            op.from.layout(),
            op.from.transfer(),
            to,
        );
    }
}

fn step_u16_f32_spec(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::U16ToF32 => u16_to_f32(src, dst, width as usize, ch),
        ConvertStep::F32ToU16 => f32_to_u16(src, dst, width as usize, ch, op.in_trust.facts),
        _ => {}
    }
}

fn step_u16_f32_untrusted(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::U16ToF32 => u16_to_f32(src, dst, width as usize, ch),
        ConvertStep::F32ToU16 => {
            let count = width as usize * ch;
            let src_f32: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
            let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
            crate::float_simd::convert_f32_to_u16_slice_untrusted(src_f32, dst_u16);
        }
        _ => {}
    }
}

fn step_u16_f32_truncate(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    f32_to_u16_truncate(
        src,
        dst,
        width as usize,
        op.from.layout().channels(),
        op.in_trust.facts,
    );
}

fn step_f16_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ch = op.from.layout().channels();
    match op.step {
        ConvertStep::F16ToF32 => f16_to_f32(src, dst, width as usize, ch),
        ConvertStep::F32ToF16 => f32_to_f16(src, dst, width as usize, ch),
        _ => {}
    }
}

#[cfg(feature = "hdr-transfers")]
fn step_pq_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::PqU16ToLinearF32 => {
            pq_u16_to_linear_f32(
                src,
                dst,
                width as usize,
                op.from.layout(),
                op.pq_anchor_scale,
            );
        }
        ConvertStep::LinearF32ToPqU16 => {
            linear_f32_to_pq_u16(
                src,
                dst,
                width as usize,
                op.from.layout(),
                op.pq_anchor_scale,
            );
        }
        _ => {}
    }
}

#[cfg(feature = "hdr-transfers")]
fn step_pq_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::PqF32ToLinearF32 => {
            pq_f32_to_linear_f32(
                src,
                dst,
                width as usize,
                op.from.layout(),
                op.pq_anchor_scale,
                op.in_trust.facts,
            );
        }
        ConvertStep::LinearF32ToPqF32 => {
            linear_f32_to_pq_f32(
                src,
                dst,
                width as usize,
                op.from.layout(),
                op.pq_anchor_scale,
                op.in_trust.facts,
            );
        }
        _ => {}
    }
}

#[cfg(feature = "hdr-transfers")]
fn step_hlg_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::HlgU16ToLinearF32 => {
            hlg_u16_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToHlgU16 => {
            linear_f32_to_hlg_u16(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

#[cfg(feature = "hdr-transfers")]
fn step_hlg_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match &op.step {
        ConvertStep::HlgF32ToLinearF32 => {
            hlg_f32_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToHlgF32 => {
            linear_f32_to_hlg_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::HlgRender(state) => {
            let prepared = state
                .prepared
                .as_ref()
                .expect("HlgRenderState bound before execution");
            super::hlg_render::row(src, dst, width as usize, op.from.layout(), prepared);
        }
        _ => {}
    }
}

fn step_tf_srgb_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::SrgbF32ToLinearF32 => {
            srgb_f32_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToSrgbF32 => {
            linear_f32_to_srgb_f32(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

fn step_tf_srgb_ext_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::SrgbF32ToLinearF32Extended => {
            srgb_f32_to_linear_f32_extended(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToSrgbF32Extended => {
            linear_f32_to_srgb_f32_extended(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

fn step_tf_bt709_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::Bt709F32ToLinearF32 => {
            bt709_f32_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToBt709F32 => {
            linear_f32_to_bt709_f32(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

fn step_tf_gamma22_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match op.step {
        ConvertStep::Gamma22F32ToLinearF32 => {
            gamma22_f32_to_linear_f32(src, dst, width as usize, op.from.layout());
        }
        ConvertStep::LinearF32ToGamma22F32 => {
            linear_f32_to_gamma22_f32(src, dst, width as usize, op.from.layout());
        }
        _ => {}
    }
}

fn step_tf_gamma22_ext_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let inverse = matches!(op.step, ConvertStep::LinearF32ToGamma22F32Extended);
    gamma22_f32_extended(
        src,
        dst,
        width as usize,
        op.from.layout(),
        inverse,
        op.in_trust.facts,
    );
}

#[cfg(feature = "oklab")]
fn step_oklab_spec(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    match op.step {
        ConvertStep::LinearRgbToOklab => {
            linear_rgb_to_oklab_f32(src, dst, w, op.from.primaries, op.in_trust);
        }
        ConvertStep::OklabToLinearRgb => oklab_to_linear_rgb_f32(src, dst, w, op.from.primaries),
        ConvertStep::LinearRgbaToOklaba => {
            linear_rgba_to_oklaba_f32(src, dst, w, op.from.primaries, op.in_trust);
        }
        ConvertStep::OklabaToLinearRgba => {
            oklaba_to_linear_rgba_f32(src, dst, w, op.from.primaries)
        }
        _ => {}
    }
}

#[cfg(feature = "oklab")]
fn step_oklab_untrusted(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    match op.step {
        ConvertStep::LinearRgbToOklab => {
            linear_rgb_to_oklab_f32_untrusted(src, dst, w, op.from.primaries);
        }
        ConvertStep::OklabToLinearRgb => oklab_to_linear_rgb_f32(src, dst, w, op.from.primaries),
        ConvertStep::LinearRgbaToOklaba => {
            linear_rgba_to_oklaba_f32_untrusted(src, dst, w, op.from.primaries);
        }
        ConvertStep::OklabaToLinearRgba => {
            oklaba_to_linear_rgba_f32(src, dst, w, op.from.primaries)
        }
        _ => {}
    }
}

fn step_gamut_matrix_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    match &op.step {
        ConvertStep::GamutMatrixRgbF32(flat, _) => {
            gamut_matrix_rgb_f32(src, dst, width as usize, flat)
        }
        ConvertStep::GamutMatrixRgbaF32(flat, _) => {
            gamut_matrix_rgba_f32(src, dst, width as usize, flat);
        }
        _ => {}
    }
}

#[inline(always)]
fn unflatten_mat3(flat: &[f32; 9]) -> [[f32; 3]; 3] {
    [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]
}

fn step_fused_u8_gamut(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    _width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::Fused {
        kind, matrix: flat, ..
    } = &op.step
    {
        let m = unflatten_mat3(flat);
        match kind {
            crate::convert::FusedKind::SrgbU8GamutRgb => {
                crate::fast_gamut::convert_u8_rgb_simd_matlut(
                    &m,
                    src,
                    dst,
                    crate::fast_gamut::srgb_lin_lut_u8(),
                    crate::fast_gamut::srgb_encode_u8,
                );
            }
            crate::convert::FusedKind::SrgbU8GamutRgba => {
                crate::fast_gamut::convert_u8_rgba_simd_lut(
                    &m,
                    src,
                    dst,
                    crate::fast_gamut::srgb_lin_lut_u8(),
                    crate::fast_gamut::srgb_encode_u8,
                );
            }
            _ => {}
        }
    }
}

fn step_fused_u16_gamut(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    _width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::Fused { matrix: flat, .. } = &op.step {
        let m = unflatten_mat3(flat);
        let src_u16: &[u16] = bytemuck::cast_slice(src);
        let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
        crate::fast_gamut::convert_u16_rgb_simd_lutdec_polyenc(&m, src_u16, dst_u16);
    }
}

fn step_fused_u8_f32_gamut(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    _width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    if let ConvertStep::Fused {
        kind, matrix: flat, ..
    } = &op.step
    {
        let m = unflatten_mat3(flat);
        match kind {
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
                    crate::fast_gamut::srgb_u8_encoder(),
                );
            }
            _ => {}
        }
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_polar(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::CartesianToPolar { target_layout }) => {
            super::perceptual::apply_cart_to_polar(
                src,
                dst,
                width as usize,
                target_layout.has_alpha(),
            );
        }
        ConvertStep::Perceptual(PerceptualStep::PolarToCartesian { target_layout }) => {
            super::perceptual::apply_polar_to_cart(
                src,
                dst,
                width as usize,
                target_layout.has_alpha(),
            );
        }
        _ => {}
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_jzazbz(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    let w = width as usize;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::LinearRgbToJzazbz {
            has_alpha,
            anchored,
        }) => {
            super::perceptual::apply_linear_rgb_to_jzazbz(
                src,
                dst,
                w,
                has_alpha,
                anchored,
                op.pq_anchor_scale,
                op.from.primaries,
            );
        }
        ConvertStep::Perceptual(PerceptualStep::JzazbzToLinearRgb {
            has_alpha,
            anchored,
            primaries,
        }) => {
            super::perceptual::apply_jzazbz_to_linear_rgb(
                src,
                dst,
                w,
                has_alpha,
                anchored,
                op.pq_anchor_scale,
                primaries,
            );
        }
        ConvertStep::Perceptual(PerceptualStep::PqRgbToJzazbz { has_alpha }) => {
            super::perceptual::apply_pq_rgb_to_jzazbz(src, dst, w, has_alpha, op.from.primaries);
        }
        ConvertStep::Perceptual(PerceptualStep::JzazbzToPqRgb {
            has_alpha,
            primaries,
        }) => {
            super::perceptual::apply_jzazbz_to_pq_rgb(src, dst, w, has_alpha, primaries);
        }
        _ => {}
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_lch(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    let w = width as usize;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::LinearRgbToLch { has_alpha }) => {
            super::perceptual::apply_linear_rgb_to_lch(src, dst, w, has_alpha, op.from.primaries);
        }
        ConvertStep::Perceptual(PerceptualStep::LchToLinearRgb {
            has_alpha,
            primaries,
        }) => {
            super::perceptual::apply_lch_to_linear_rgb(src, dst, w, has_alpha, primaries);
        }
        _ => {}
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_hsl_hsv(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    let w = width as usize;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::RgbToHsl { has_alpha }) => {
            super::perceptual::apply_rgb_to_hsl(src, dst, w, has_alpha);
        }
        ConvertStep::Perceptual(PerceptualStep::HslToRgb { has_alpha }) => {
            super::perceptual::apply_hsl_to_rgb(src, dst, w, has_alpha);
        }
        ConvertStep::Perceptual(PerceptualStep::RgbToHsv { has_alpha }) => {
            super::perceptual::apply_rgb_to_hsv(src, dst, w, has_alpha);
        }
        ConvertStep::Perceptual(PerceptualStep::HsvToRgb { has_alpha }) => {
            super::perceptual::apply_hsv_to_rgb(src, dst, w, has_alpha);
        }
        _ => {}
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_int_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    let w = width as usize;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::DecodeIntToF32 { layout, .. }) => {
            super::perceptual::apply_decode_int_u8(src, dst, w, layout);
        }
        ConvertStep::Perceptual(PerceptualStep::EncodeF32ToInt { layout, .. }) => {
            super::perceptual::apply_encode_int_u8(src, dst, w, layout);
        }
        _ => {}
    }
}

#[cfg(feature = "perceptual")]
fn step_perceptual_int_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    use super::perceptual::PerceptualStep;
    let w = width as usize;
    match op.step {
        ConvertStep::Perceptual(PerceptualStep::DecodeIntToF32 { layout, .. }) => {
            super::perceptual::apply_decode_int_u16(src, dst, w, layout);
        }
        ConvertStep::Perceptual(PerceptualStep::EncodeF32ToInt { layout, .. }) => {
            super::perceptual::apply_encode_int_u16(src, dst, w, layout);
        }
        _ => {}
    }
}

/// Apply a pluggable HDR tone-map step to one row of linear-light F32
/// RGB(A) pixels in the mapper's working primaries.
///
/// The carrier layout (RGB vs RGBA) is read from `from.layout()`. For
/// RGBA, the alpha channel is copied through verbatim. Input is scrubbed
/// for non-finite values and negatives before reaching the mapper;
/// output is clamped to `[0, 1]` to absorb f32 epsilon-level overshoot at
/// the saturated end.
///
/// **Lazy compilation.** On the first row that flows through a given
/// step, this kernel resolves the [`ToneMapContext`](crate::hdr::ToneMapContext)
/// from the step's stored peaks / working primaries, calls
/// [`ToneMapper::prepare`](crate::hdr::ToneMapper::prepare), and stashes
/// the resulting [`StripMapper`](crate::hdr::StripMapper) in the step's
/// shared `OnceCell`. Every subsequent row reuses the cached executor
/// without rebuilding context or re-virtualizing `prepare`.
//
#[cfg(feature = "hdr-experimental")]
fn tone_map_kernel(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from: PixelDescriptor,
    step: &super::ToneMapStep,
    color_facts: zenpixels::sample::FloatFacts,
    scratch: &mut HdrKernelScratch,
) {
    use zenpixels::sample::FloatFacts;
    let strip_mapper = step
        .prepared
        .get()
        .expect("mapper prepared before execution");

    let channels = if from.layout().has_alpha() { 4 } else { 3 };
    let src_f32: &[f32] = bytemuck::cast_slice(src);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

    let rgb_out = &mut scratch.rgb_out[..width];
    let trusted_finite_nonneg =
        color_facts.contains(FloatFacts::NON_NEGATIVE | FloatFacts::NO_INFINITE);

    if trusted_finite_nonneg && channels == 3 {
        strip_mapper.map_strip(&src_f32[..width * 3], rgb_out.as_flattened_mut());
    } else {
        let rgb_in = &mut scratch.rgb_strip[..width];
        if trusted_finite_nonneg {
            for (p, px) in rgb_in.iter_mut().enumerate() {
                let base = p * channels;
                *px = [src_f32[base], src_f32[base + 1], src_f32[base + 2]];
            }
        } else {
            // Mappers see finite, non-negative linear light.
            let scrub = |v: f32| if v.is_finite() && v >= 0.0 { v } else { 0.0 };
            for (p, px) in rgb_in.iter_mut().enumerate() {
                let base = p * channels;
                *px = [
                    scrub(src_f32[base]),
                    scrub(src_f32[base + 1]),
                    scrub(src_f32[base + 2]),
                ];
            }
        }
        strip_mapper.map_strip(rgb_in.as_flattened(), rgb_out.as_flattened_mut());
    }
    let settle = |v: f32| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    for (p, px) in rgb_out.iter().enumerate() {
        let base = p * channels;
        dst_f32[base] = settle(px[0]);
        dst_f32[base + 1] = settle(px[1]);
        dst_f32[base + 2] = settle(px[2]);
        if channels == 4 {
            dst_f32[base + 3] = src_f32[base + 3];
        }
    }
}

/// OKLch soft chroma compression on a row of linear-light F32 RGB(A) pixels
/// in `primaries`. Alpha (when present) is copied through verbatim.
//
// Same justification for `needless_range_loop` as in
// `tone_map_bt2446a_kernel`.
#[cfg(feature = "oklab")]
#[allow(clippy::needless_range_loop)]
fn soft_compress_oklch_kernel(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from: PixelDescriptor,
    compressor: &crate::hdr::SoftCompress,
    scratch: &mut HdrKernelScratch,
) {
    let has_alpha = from.layout().has_alpha();
    let channels = if has_alpha { 4 } else { 3 };
    let src_f32: &[f32] = bytemuck::cast_slice(src);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);

    if has_alpha {
        // Row-persistent RGB strip; see `tone_map_bt2446a_kernel`.
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
// Tiers per kernel follow benchmarks/row_kernel_tiers_2026-11-04.md: a v3 copy needs a 1.25x
// gain over a plain function, `+v4x` a 1.2x gain over v3. Neon and wasm128 are their targets'
// baseline, so a plain function already gets them.
//
// Shuffle / replicate kernels — split-per-type #[attune] pattern.
//
// U16 and F16 share one function per shape, because the operation only moves
// 2-byte samples around. Kernels that insert opaque alpha take its bits as an
// argument (65535 for U16, F16_ONE_BITS for F16); F32 has its own copy.
// ----------------------------------------------------------------------------

fn step_layout_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    match op.step {
        ConvertStep::SwizzleBgraRgba => {
            let n = w * 4;
            garb::bytes::rgba_to_bgra(&src[..n], &mut dst[..n]).expect("pre-validated row size");
        }
        ConvertStep::SwizzleBgrRgb => {
            for (out, px) in dst[..w * 3]
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src[..w * 3].as_chunks::<3>().0)
            {
                *out = [px[2], px[1], px[0]];
            }
        }
        ConvertStep::AddAlpha => {
            garb::bytes::rgb_to_rgba(&src[..w * 3], &mut dst[..w * 4])
                .expect("pre-validated row size");
        }
        ConvertStep::RgbToBgra => {
            garb::bytes::rgb_to_bgra(&src[..w * 3], &mut dst[..w * 4])
                .expect("pre-validated row size");
        }
        ConvertStep::DropAlpha => {
            garb::bytes::rgba_to_rgb(&src[..w * 4], &mut dst[..w * 3])
                .expect("pre-validated row size");
        }
        _ => {}
    }
}

fn layout_2bytes(step: &ConvertStep, src: &[u8], dst: &mut [u8], w: usize, opaque: u16) {
    match step {
        ConvertStep::SwizzleBgraRgba => {
            for (out, px) in dst[..w * 8]
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(src[..w * 8].as_chunks::<8>().0)
            {
                *out = [px[4], px[5], px[2], px[3], px[0], px[1], px[6], px[7]];
            }
        }
        ConvertStep::SwizzleBgrRgb => {
            for (out, px) in dst[..w * 6]
                .as_chunks_mut::<6>()
                .0
                .iter_mut()
                .zip(src[..w * 6].as_chunks::<6>().0)
            {
                *out = [px[4], px[5], px[2], px[3], px[0], px[1]];
            }
        }
        ConvertStep::AddAlpha => add_alpha_2bytes(
            bytemuck::cast_slice(&src[..w * 6]),
            bytemuck::cast_slice_mut(&mut dst[..w * 8]),
            w,
            opaque,
        ),
        ConvertStep::RgbToBgra => {
            add_alpha_2bytes(
                bytemuck::cast_slice(&src[..w * 6]),
                bytemuck::cast_slice_mut(&mut dst[..w * 8]),
                w,
                opaque,
            );
            for px in dst[..w * 8].as_chunks_mut::<8>().0 {
                px.swap(0, 4);
                px.swap(1, 5);
            }
        }
        ConvertStep::DropAlpha => drop_alpha_2bytes(
            bytemuck::cast_slice(&src[..w * 8]),
            bytemuck::cast_slice_mut(&mut dst[..w * 6]),
            w,
        ),
        _ => {}
    }
}

fn step_layout_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    layout_2bytes(&op.step, src, dst, width as usize, 65535);
}

fn step_layout_f16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    layout_2bytes(&op.step, src, dst, width as usize, F16_ONE_BITS);
}

fn step_layout_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    match op.step {
        ConvertStep::SwizzleBgraRgba => {
            for (out, px) in dst[..w * 16]
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(src[..w * 16].as_chunks::<16>().0)
            {
                out[0..4].copy_from_slice(&px[8..12]);
                out[4..8].copy_from_slice(&px[4..8]);
                out[8..12].copy_from_slice(&px[0..4]);
                out[12..16].copy_from_slice(&px[12..16]);
            }
        }
        ConvertStep::SwizzleBgrRgb => {
            for (out, px) in dst[..w * 12]
                .as_chunks_mut::<12>()
                .0
                .iter_mut()
                .zip(src[..w * 12].as_chunks::<12>().0)
            {
                out[0..4].copy_from_slice(&px[8..12]);
                out[4..8].copy_from_slice(&px[4..8]);
                out[8..12].copy_from_slice(&px[0..4]);
            }
        }
        ConvertStep::AddAlpha => add_alpha_f32(
            bytemuck::cast_slice(&src[..w * 12]),
            bytemuck::cast_slice_mut(&mut dst[..w * 16]),
            w,
        ),
        ConvertStep::RgbToBgra => {
            add_alpha_f32(
                bytemuck::cast_slice(&src[..w * 12]),
                bytemuck::cast_slice_mut(&mut dst[..w * 16]),
                w,
            );
            for px in dst[..w * 16].as_chunks_mut::<16>().0 {
                let (head, tail) = px.split_at_mut(8);
                head[0..4].swap_with_slice(&mut tail[0..4]);
            }
        }
        ConvertStep::DropAlpha => drop_alpha_f32(
            bytemuck::cast_slice(&src[..w * 16]),
            bytemuck::cast_slice_mut(&mut dst[..w * 12]),
            w,
        ),
        _ => {}
    }
}

// -- add_alpha (inserts opaque alpha: differs per type) ----------------------

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn add_alpha_2bytes(src: &[u16], dst: &mut [u16], width: usize, opaque: u16) {
    for (s, d) in src[..width * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [s[0], s[1], s[2], opaque];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn add_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [s[0], s[1], s[2], 1.0];
    }
}

// -- drop_alpha (pure shuffle — U16 and F16 share one kernel) ----------------

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn drop_alpha_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        *d = [s[0], s[1], s[2]];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn drop_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        *d = [s[0], s[1], s[2]];
    }
}

/// Straight-alpha sRGB U8 matte composite using table-based decode and encode
/// (zero `float_simd` dependencies).
fn step_matte_srgb_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ConvertStep::MatteComposite(state) = &op.step else {
        return;
    };
    let matte = state.linear;
    const PIXELS: usize = 128;
    let w = width as usize;
    let mut rgb = [0f32; PIXELS * 3];
    let mut alpha = [0f32; PIXELS];
    for (s, d) in src[..w * 4]
        .chunks(PIXELS * 4)
        .zip(dst[..w * 3].chunks_mut(PIXELS * 3))
    {
        let n = s.len() / 4;
        let (rgb, alpha) = (&mut rgb[..n * 3], &mut alpha[..n]);
        for ((lane, byte), a) in rgb
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(s.as_chunks::<4>().0)
            .zip(alpha.iter_mut())
        {
            for c in 0..3 {
                lane[c] = linear_srgb::default::srgb_u8_to_linear(byte[c]);
            }
            *a = f32::from(byte[3]) * (1.0 / 255.0);
        }
        matte_blend(rgb, alpha, matte);
        linear_srgb::default::linear_to_srgb_u8_slice(rgb, d);
    }
}

#[inline(never)]
fn matte_blend_and_reencode(
    rgba: &[f32],
    rgb: &mut [f32],
    alpha: &mut [f32],
    matte: [f32; 3],
    is_premul: bool,
    to_linear: fn(&mut [f32]),
    from_linear: fn(&mut [f32]),
) {
    matte_split(rgba, rgb, alpha);
    to_linear(rgb);
    if is_premul {
        matte_blend_premul(rgb, alpha, matte);
    } else {
        matte_blend(rgb, alpha, matte);
    }
    from_linear(rgb);
}

fn step_matte_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ConvertStep::MatteComposite(state) = &op.step else {
        return;
    };
    let is_premul = op.from.alpha() == Some(zenpixels::AlphaMode::Premultiplied);
    const PIXELS: usize = 128;
    let w = width as usize;
    let mut rgba = [0f32; PIXELS * 4];
    let mut rgb = [0f32; PIXELS * 3];
    let mut alpha = [0f32; PIXELS];
    for (s, d) in src[..w * 4]
        .chunks(PIXELS * 4)
        .zip(dst[..w * 3].chunks_mut(PIXELS * 3))
    {
        let n = s.len() / 4;
        let (rgba, rgb, alpha) = (&mut rgba[..n * 4], &mut rgb[..n * 3], &mut alpha[..n]);
        crate::float_simd::convert_u8_to_f32_slice(s, rgba);
        matte_blend_and_reencode(
            rgba,
            rgb,
            alpha,
            state.linear,
            is_premul,
            state.to_linear,
            state.from_linear,
        );
        crate::float_simd::convert_f32_to_u8_slice_untrusted(rgb, d);
    }
}

fn step_matte_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ConvertStep::MatteComposite(state) = &op.step else {
        return;
    };
    let is_premul = op.from.alpha() == Some(zenpixels::AlphaMode::Premultiplied);
    const PIXELS: usize = 128;
    let w = width as usize;
    let mut rgba = [0f32; PIXELS * 4];
    let mut rgb = [0f32; PIXELS * 3];
    let mut alpha = [0f32; PIXELS];
    for (s, d) in src[..w * 8]
        .chunks(PIXELS * 8)
        .zip(dst[..w * 6].chunks_mut(PIXELS * 6))
    {
        let n = s.len() / 8;
        let (rgba, rgb, alpha) = (&mut rgba[..n * 4], &mut rgb[..n * 3], &mut alpha[..n]);
        crate::float_simd::convert_u16_to_f32_slice(bytemuck::cast_slice(s), rgba);
        matte_blend_and_reencode(
            rgba,
            rgb,
            alpha,
            state.linear,
            is_premul,
            state.to_linear,
            state.from_linear,
        );
        crate::float_simd::convert_f32_to_u16_slice_untrusted(rgb, bytemuck::cast_slice_mut(d));
    }
}

fn step_matte_f16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ConvertStep::MatteComposite(state) = &op.step else {
        return;
    };
    let is_premul = op.from.alpha() == Some(zenpixels::AlphaMode::Premultiplied);
    const PIXELS: usize = 128;
    let w = width as usize;
    let mut rgba = [0f32; PIXELS * 4];
    let mut rgb = [0f32; PIXELS * 3];
    let mut alpha = [0f32; PIXELS];
    for (s, d) in src[..w * 8]
        .chunks(PIXELS * 8)
        .zip(dst[..w * 6].chunks_mut(PIXELS * 6))
    {
        let n = s.len() / 8;
        let (rgba, rgb, alpha) = (&mut rgba[..n * 4], &mut rgb[..n * 3], &mut alpha[..n]);
        f16_bits_to_f32_slice(bytemuck::cast_slice(s), rgba);
        matte_blend_and_reencode(
            rgba,
            rgb,
            alpha,
            state.linear,
            is_premul,
            state.to_linear,
            state.from_linear,
        );
        f32_to_f16_bits_slice(rgb, bytemuck::cast_slice_mut(d));
    }
}

fn step_matte_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let ConvertStep::MatteComposite(state) = &op.step else {
        return;
    };
    let is_premul = op.from.alpha() == Some(zenpixels::AlphaMode::Premultiplied);
    const PIXELS: usize = 128;
    let w = width as usize;
    let src_f: &[f32] = bytemuck::cast_slice(&src[..w * 16]);
    let dst_f: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..w * 12]);
    let mut alpha = [0f32; PIXELS];
    for (rgba, rgb) in src_f.chunks(PIXELS * 4).zip(dst_f.chunks_mut(PIXELS * 3)) {
        let n = rgba.len() / 4;
        matte_blend_and_reencode(
            rgba,
            rgb,
            &mut alpha[..n],
            state.linear,
            is_premul,
            state.to_linear,
            state.from_linear,
        );
    }
}

/// RGBA samples to separate RGB and clamped alpha (scrubbing NaN alpha to 0.0).
#[attune(dispatch, -_neon, -_wasm128)]
fn matte_split(rgba: &[f32], rgb: &mut [f32], alpha: &mut [f32]) {
    for ((s, d), a) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0)
        .zip(alpha)
    {
        *d = [s[0], s[1], s[2]];
        let raw_a = s[3];
        *a = if raw_a >= 0.0 { raw_a.min(1.0) } else { 0.0 };
    }
}

/// `rgb * alpha + matte * (1 - alpha)`, in linear light (straight alpha).
#[attune(dispatch, -_neon, -_wasm128)]
fn matte_blend(rgb: &mut [f32], alpha: &[f32], matte: [f32; 3]) {
    for (p, a) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(alpha) {
        let inv = 1.0 - a;
        p[0] = p[0] * a + matte[0] * inv;
        p[1] = p[1] * a + matte[1] * inv;
        p[2] = p[2] * a + matte[2] * inv;
    }
}

/// `rgb + matte * (1 - alpha)`, in linear light (premultiplied alpha, W3C Compositing-1 §5.1).
fn matte_blend_premul(rgb: &mut [f32], alpha: &[f32], matte: [f32; 3]) {
    for (p, a) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(alpha) {
        let inv = 1.0 - a;
        p[0] += matte[0] * inv;
        p[1] += matte[1] * inv;
        p[2] += matte[2] * inv;
    }
}

fn tf_srgb_to_lin(rgb: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_slice(rgb);
}
fn tf_lin_to_srgb(rgb: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_slice(rgb);
}
fn tf_srgb_ext_to_lin(rgb: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_extended_slice(rgb);
}
fn tf_lin_to_srgb_ext(rgb: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_extended_slice(rgb);
}
fn tf_bt709_to_lin(rgb: &mut [f32]) {
    linear_srgb::default::bt709_to_linear_slice(rgb);
}
fn tf_lin_to_bt709(rgb: &mut [f32]) {
    linear_srgb::default::linear_to_bt709_slice(rgb);
}
#[cfg(feature = "hdr-transfers")]
fn tf_hlg_to_lin(rgb: &mut [f32]) {
    linear_srgb::default::hlg_to_linear_slice(rgb);
}
#[cfg(feature = "hdr-transfers")]
fn tf_lin_to_hlg(rgb: &mut [f32]) {
    linear_srgb::default::linear_to_hlg_slice(rgb);
}
fn tf_gamma22_to_lin(rgb: &mut [f32]) {
    linear_srgb::default::gamma_to_linear_slice(rgb, ADOBE_GAMMA);
}
fn tf_lin_to_gamma22(rgb: &mut [f32]) {
    linear_srgb::default::linear_to_gamma_slice(rgb, ADOBE_GAMMA);
}
fn tf_gamma22_ext_to_lin(rgb: &mut [f32]) {
    for v in rgb {
        *v = libm::powf(v.abs(), ADOBE_GAMMA).copysign(*v);
    }
}
fn tf_lin_to_gamma22_ext(rgb: &mut [f32]) {
    for v in rgb {
        *v = libm::powf(v.abs(), 1.0 / ADOBE_GAMMA).copysign(*v);
    }
}

// -- gray_to_rgb (pure replicate — U16 and F16 share) ------------------------

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_rgb_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        *d = [g, g, g];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_rgb_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        *d = [g, g, g];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_rgb_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        *d = [g, g, g];
    }
}

// -- gray_to_rgba (replicate + opaque alpha) ---------------------------------

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_rgba_2bytes(src: &[u16], dst: &mut [u16], width: usize, opaque: u16) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [g, g, g, opaque];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_rgba_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [g, g, g, 1.0];
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

#[attune(dispatch, -_neon, -_wasm128)]
fn rgb_to_gray_u8_generic(src: &[u8], dst: &mut [u8], width: usize, w: [f32; 3]) {
    for (s, d) in src[..width * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        *d = (y + 0.5).clamp(0.0, 255.0) as u8;
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn rgba_to_gray_u8_generic(src: &[u8], dst: &mut [u8], width: usize, w: [f32; 3]) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        let y = (s[0] as f32) * w[0] + (s[1] as f32) * w[1] + (s[2] as f32) * w[2];
        *d = (y + 0.5).clamp(0.0, 255.0) as u8;
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn widen_u16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    for (&v, f) in src.iter().zip(dst.iter_mut()) {
        *f = f32::from(v);
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn narrow_luma_f32_to_u16_slice(src: &[f32], dst: &mut [u16]) {
    for (&y, out) in src.iter().zip(dst.iter_mut()) {
        *out = (y + 0.5).clamp(0.0, 65535.0) as u16;
    }
}

/// U16 luma through the `f32` kernel in stack-sized chunks (`u16 -> f32` is lossless),
/// reusing the `v3`/`v4x` `rgb_to_gray_f32` / `rgba_to_gray_f32` kernels while eliminating
/// dedicated U16 luma tiers and speeding up 4K U16→Gray 2–3×.
fn gray_u16_via_f32<const N: usize>(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    const PIXELS: usize = 128;
    let mut wide = [[0f32; 4]; PIXELS];
    let mut luma = [0f32; PIXELS];
    for (s, d) in src[..width * N]
        .chunks(PIXELS * N)
        .zip(dst[..width].chunks_mut(PIXELS))
    {
        let n = d.len();
        let flat = &mut bytemuck::cast_slice_mut::<[f32; 4], f32>(&mut wide)[..n * N];
        widen_u16_to_f32_slice(s, flat);
        if N == 3 {
            rgb_to_gray_f32(flat, &mut luma[..n], n, w);
        } else {
            rgba_to_gray_f32(flat, &mut luma[..n], n, w);
        }
        narrow_luma_f32_to_u16_slice(&luma[..n], d);
    }
}

fn rgb_to_gray_u16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    gray_u16_via_f32::<3>(src, dst, width, w);
}

fn rgba_to_gray_u16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    gray_u16_via_f32::<4>(src, dst, width, w);
}

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn rgb_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize, w: [f32; 3]) {
    for (s, d) in src[..width * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        *d = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];
    }
}

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn rgba_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize, w: [f32; 3]) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        *d = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];
    }
}

/// Half-float luma through the `f32` kernel, converting in stack-sized chunks with the
/// hardware-backed slice converters, so only one luma kernel exists per tier.
fn gray_f16_via_f32<const N: usize>(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    const PIXELS: usize = 128;
    let mut wide = [[0f32; 4]; PIXELS];
    let mut luma = [0f32; PIXELS];
    for (s, d) in src[..width * N]
        .chunks(PIXELS * N)
        .zip(dst[..width].chunks_mut(PIXELS))
    {
        let n = d.len();
        let flat = &mut bytemuck::cast_slice_mut::<[f32; 4], f32>(&mut wide)[..n * N];
        f16_bits_to_f32_slice(s, flat);
        if N == 3 {
            rgb_to_gray_f32(flat, &mut luma[..n], n, w);
        } else {
            rgba_to_gray_f32(flat, &mut luma[..n], n, w);
        }
        f32_to_f16_bits_slice(&luma[..n], d);
    }
}

fn rgb_to_gray_f16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    gray_f16_via_f32::<3>(src, dst, width, w);
}

fn rgba_to_gray_f16(src: &[u16], dst: &mut [u16], width: usize, w: [f32; 3]) {
    gray_f16_via_f32::<4>(src, dst, width, w);
}

#[attune(dispatch, -_neon, -_wasm128)]
fn rgb_to_gray_bt709_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for (px, d) in src[..width * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        *d = ((u16::from(px[0]) * 54 + u16::from(px[1]) * 183 + u16::from(px[2]) * 19 + 128) >> 8)
            as u8;
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn rgba_to_gray_bt709_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for (px, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width].iter_mut())
    {
        *d = ((u16::from(px[0]) * 54 + u16::from(px[1]) * 183 + u16::from(px[2]) * 19 + 128) >> 8)
            as u8;
    }
}

// -- gray_alpha_to_rgba (pure replicate + alpha-preserve — U16 and F16 share)

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_rgba_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [s[0], s[0], s[0], s[1]];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_rgba_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        *d = [s[0], s[0], s[0], s[1]];
    }
}

// -- gray_alpha_to_rgb (replicate + drop alpha — U16 and F16 share) ---------

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_rgb_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        let g = s[0];
        *d = [g, g, g];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_rgb_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        let g = s[0];
        *d = [g, g, g];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_rgb_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 3].as_chunks_mut::<3>().0)
    {
        let g = s[0];
        *d = [g, g, g];
    }
}

// -- gray_to_gray_alpha (inserts opaque alpha: differs per type) -------------

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_gray_alpha_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        *d = [g, 255];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_gray_alpha_2bytes(src: &[u16], dst: &mut [u16], width: usize, opaque: u16) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        *d = [g, opaque];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_to_gray_alpha_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for (&g, d) in src[..width]
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        *d = [g, 1.0];
    }
}

// -- gray_alpha_to_gray (drop alpha — U16 and F16 share) --------------------

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        dst[i] = src[i * 2];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_gray_2bytes(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        dst[i] = src[i * 2];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn gray_alpha_to_gray_f32(src: &[f32], dst: &mut [f32], width: usize) {
    for i in 0..width {
        dst[i] = src[i * 2];
    }
}

fn step_gray_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    const PIXELS: usize = 128;
    let w = width as usize;
    match op.step {
        ConvertStep::GrayToRgb => gray_to_rgb_u8(&src[..w], &mut dst[..w * 3], w),
        ConvertStep::GrayToRgba => {
            garb::bytes::gray_to_rgba(&src[..w], &mut dst[..w * 4])
                .expect("pre-validated row size");
        }
        ConvertStep::RgbToGray { coefficients } => {
            if coefficients == LumaCoefficients::Bt709 {
                rgb_to_gray_bt709_u8(&src[..w * 3], &mut dst[..w], w);
            } else {
                rgb_to_gray_u8_generic(
                    &src[..w * 3],
                    &mut dst[..w],
                    w,
                    coefficients.coefficients(),
                );
            }
        }
        ConvertStep::RgbaToGray { coefficients } => {
            if coefficients == LumaCoefficients::Bt709 {
                rgba_to_gray_bt709_u8(&src[..w * 4], &mut dst[..w], w);
            } else {
                rgba_to_gray_u8_generic(
                    &src[..w * 4],
                    &mut dst[..w],
                    w,
                    coefficients.coefficients(),
                );
            }
        }
        ConvertStep::RgbaToGrayAlpha { coefficients } => {
            let coeff = coefficients.coefficients();
            let mut luma = [0u8; PIXELS];
            for (s, d) in src[..w * 4]
                .chunks(PIXELS * 4)
                .zip(dst[..w * 2].chunks_mut(PIXELS * 2))
            {
                let n = s.len() / 4;
                rgba_to_gray_u8_generic(s, &mut luma[..n], n, coeff);
                for ((&y, sp), dp) in luma[..n]
                    .iter()
                    .zip(s.as_chunks::<4>().0)
                    .zip(d.as_chunks_mut::<2>().0)
                {
                    *dp = [y, sp[3]];
                }
            }
        }
        ConvertStep::GrayAlphaToRgba => {
            garb::bytes::gray_alpha_to_rgba(&src[..w * 2], &mut dst[..w * 4])
                .expect("pre-validated row size");
        }
        ConvertStep::GrayAlphaToRgb => gray_alpha_to_rgb_u8(&src[..w * 2], &mut dst[..w * 3], w),
        ConvertStep::GrayToGrayAlpha => gray_to_gray_alpha_u8(&src[..w], &mut dst[..w * 2], w),
        ConvertStep::GrayAlphaToGray => gray_alpha_to_gray_u8(&src[..w * 2], &mut dst[..w], w),
        _ => {}
    }
}

fn gray_2bytes_common(
    step: &ConvertStep,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    opaque: u16,
    rgb_luma: fn(&[u16], &mut [u16], usize, [f32; 3]),
    rgba_luma: fn(&[u16], &mut [u16], usize, [f32; 3]),
) {
    const PIXELS: usize = 128;
    match *step {
        ConvertStep::GrayToRgb => gray_to_rgb_2bytes(
            bytemuck::cast_slice(&src[..w * 2]),
            bytemuck::cast_slice_mut(&mut dst[..w * 6]),
            w,
        ),
        ConvertStep::GrayToRgba => gray_to_rgba_2bytes(
            bytemuck::cast_slice(&src[..w * 2]),
            bytemuck::cast_slice_mut(&mut dst[..w * 8]),
            w,
            opaque,
        ),
        ConvertStep::RgbToGray { coefficients } => rgb_luma(
            bytemuck::cast_slice(&src[..w * 6]),
            bytemuck::cast_slice_mut(&mut dst[..w * 2]),
            w,
            coefficients.coefficients(),
        ),
        ConvertStep::RgbaToGray { coefficients } => rgba_luma(
            bytemuck::cast_slice(&src[..w * 8]),
            bytemuck::cast_slice_mut(&mut dst[..w * 2]),
            w,
            coefficients.coefficients(),
        ),
        ConvertStep::RgbaToGrayAlpha { coefficients } => {
            let coeff = coefficients.coefficients();
            let src16: &[u16] = bytemuck::cast_slice(&src[..w * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..w * 4]);
            let mut luma = [0u16; PIXELS];
            for (s, d) in src16.chunks(PIXELS * 4).zip(dst16.chunks_mut(PIXELS * 2)) {
                let n = s.len() / 4;
                rgba_luma(s, &mut luma[..n], n, coeff);
                for ((&y, sp), dp) in luma[..n]
                    .iter()
                    .zip(s.as_chunks::<4>().0)
                    .zip(d.as_chunks_mut::<2>().0)
                {
                    *dp = [y, sp[3]];
                }
            }
        }
        ConvertStep::GrayAlphaToRgba => gray_alpha_to_rgba_2bytes(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 8]),
            w,
        ),
        ConvertStep::GrayAlphaToRgb => gray_alpha_to_rgb_2bytes(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 6]),
            w,
        ),
        ConvertStep::GrayToGrayAlpha => gray_to_gray_alpha_2bytes(
            bytemuck::cast_slice(&src[..w * 2]),
            bytemuck::cast_slice_mut(&mut dst[..w * 4]),
            w,
            opaque,
        ),
        ConvertStep::GrayAlphaToGray => gray_alpha_to_gray_2bytes(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 2]),
            w,
        ),
        _ => {}
    }
}

fn step_gray_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    gray_2bytes_common(
        &op.step,
        src,
        dst,
        width as usize,
        65535,
        rgb_to_gray_u16,
        rgba_to_gray_u16,
    );
}

fn step_gray_f16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    gray_2bytes_common(
        &op.step,
        src,
        dst,
        width as usize,
        F16_ONE_BITS,
        rgb_to_gray_f16,
        rgba_to_gray_f16,
    );
}

fn step_gray_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    const PIXELS: usize = 128;
    let w = width as usize;
    match op.step {
        ConvertStep::GrayToRgb => gray_to_rgb_f32(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 12]),
            w,
        ),
        ConvertStep::GrayToRgba => gray_to_rgba_f32(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 16]),
            w,
        ),
        ConvertStep::RgbToGray { coefficients } => rgb_to_gray_f32(
            bytemuck::cast_slice(&src[..w * 12]),
            bytemuck::cast_slice_mut(&mut dst[..w * 4]),
            w,
            coefficients.coefficients(),
        ),
        ConvertStep::RgbaToGray { coefficients } => rgba_to_gray_f32(
            bytemuck::cast_slice(&src[..w * 16]),
            bytemuck::cast_slice_mut(&mut dst[..w * 4]),
            w,
            coefficients.coefficients(),
        ),
        ConvertStep::RgbaToGrayAlpha { coefficients } => {
            let coeff = coefficients.coefficients();
            let src32: &[f32] = bytemuck::cast_slice(&src[..w * 16]);
            let dst32: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..w * 8]);
            let mut luma = [0f32; PIXELS];
            for (s, d) in src32.chunks(PIXELS * 4).zip(dst32.chunks_mut(PIXELS * 2)) {
                let n = s.len() / 4;
                rgba_to_gray_f32(s, &mut luma[..n], n, coeff);
                for ((&y, sp), dp) in luma[..n]
                    .iter()
                    .zip(s.as_chunks::<4>().0)
                    .zip(d.as_chunks_mut::<2>().0)
                {
                    *dp = [y, sp[3]];
                }
            }
        }
        ConvertStep::GrayAlphaToRgba => gray_alpha_to_rgba_f32(
            bytemuck::cast_slice(&src[..w * 8]),
            bytemuck::cast_slice_mut(&mut dst[..w * 16]),
            w,
        ),
        ConvertStep::GrayAlphaToRgb => gray_alpha_to_rgb_f32(
            bytemuck::cast_slice(&src[..w * 8]),
            bytemuck::cast_slice_mut(&mut dst[..w * 12]),
            w,
        ),
        ConvertStep::GrayToGrayAlpha => gray_to_gray_alpha_f32(
            bytemuck::cast_slice(&src[..w * 4]),
            bytemuck::cast_slice_mut(&mut dst[..w * 8]),
            w,
        ),
        ConvertStep::GrayAlphaToGray => gray_alpha_to_gray_f32(
            bytemuck::cast_slice(&src[..w * 8]),
            bytemuck::cast_slice_mut(&mut dst[..w * 4]),
            w,
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
#[inline(never)]
fn restore_alpha_f32_f32(src: &[f32], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = src[i * ch + ai];
    }
}

/// Restore the alpha lane of an f32 row from a u8 source row (`v / 255`).
#[inline(never)]
fn restore_alpha_u8_f32(src: &[u8], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = f32::from(src[i * ch + ai]) / 255.0;
    }
}

/// Restore the alpha lane of an f32 row from a u16 source row (`v / 65535`).
#[cfg(feature = "hdr-transfers")]
#[inline(never)]
fn restore_alpha_u16_f32(src: &[u16], dst: &mut [f32], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = f32::from(src[i * ch + ai]) / 65535.0;
    }
}

/// Restore the alpha lane of a u8 row from an f32 source row
/// (`clamp(v) * 255 + 0.5`).
#[inline(never)]
fn restore_alpha_f32_u8(src: &[f32], dst: &mut [u8], layout: ChannelLayout) {
    let Some(ai) = alpha_lane(layout) else { return };
    let ch = layout.channels();
    for (i, px) in dst.chunks_exact_mut(ch).enumerate() {
        px[ai] = (src[i * ch + ai].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// Restore the alpha lane of a u16 row from an f32 source row
/// (`clamp(v) * 65535 + 0.5`).
#[cfg(feature = "hdr-transfers")]
#[inline(never)]
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
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    crate::float_simd::convert_u8_to_f32_slice(&src[..count], dst_f32);
}

/// Naive f32 → u8 (clamp `[0,1]`, * 255 + 0.5).
fn naive_f32_to_u8(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    channels: usize,
    in_facts: zenpixels::sample::FloatFacts,
) {
    let count = width * channels;
    let src_f32: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    crate::float_simd::convert_f32_to_u8_slice(src_f32, &mut dst[..count], in_facts);
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
/// The M4 Pro measurement was 4.5× faster than garb's scalar fallback
/// (`benchmarks/u16_narrow_2026-08-27.txt`); this does not generalize to x86.
/// The default-target x86 follow-up was slower than the exact u32 candidate
/// (`benchmarks/u16_narrow_2026-09-24_x86.txt`). Keep exactness while comparing
/// platform-specific implementations with `benches/bench_u16_narrow.rs`.
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

/// u16 → u8 high-byte truncation (`v >> 8`, `DepthPolicy::Truncate`).
fn u16_to_u8_truncate(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let (pairs, _) = src[..count * 2].as_chunks::<2>();
    for (s, d) in pairs.iter().zip(dst[..count].iter_mut()) {
        *d = (u16::from_ne_bytes(*s) >> 8) as u8;
    }
}

/// f32 → u8 floor truncation (`clamp(0, 1) * 255`, `DepthPolicy::Truncate`).
fn naive_f32_to_u8_truncate(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    channels: usize,
    in_facts: zenpixels::sample::FloatFacts,
) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    crate::float_simd::convert_f32_to_u8_truncate_slice(srcf, &mut dst[..count], in_facts);
}

/// f32 → u16 floor truncation (`clamp(0, 1) * 65535`, `DepthPolicy::Truncate`).
fn f32_to_u16_truncate(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    channels: usize,
    in_facts: zenpixels::sample::FloatFacts,
) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    crate::float_simd::convert_f32_to_u16_truncate_slice(srcf, dst16, in_facts);
}

/// Compose an actual SDR transfer change before the one final quantisation.
/// Alpha is always linear and only narrows; color channels use f64 so the
/// f32 transfer approximations cannot flip values close to u8 midpoints.
/// `src` is one tight native-endian U16 row; `dst` is one tight U8 row.
/// Both contain `width` pixels in `layout`; `RowConverter` handles outer strides.
fn sdr_u16_to_u8(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    from: TransferFunction,
    to: TransferFunction,
) {
    use once_cell::race::OnceBox;

    const BT709_BETA: f64 = 0.018053968510807;
    const BT709_ALPHA: f64 = 0.09929682680944;
    // Cache each actual transfer change separately on first use.
    static TABLES: [OnceBox<[u8; 65536]>; 6] = [const { OnceBox::new() }; 6];

    #[inline]
    fn decode(tf: TransferFunction, x: f64) -> f64 {
        match tf {
            TransferFunction::Linear => x,
            TransferFunction::Srgb => {
                if x <= 0.04045 {
                    x / 12.92
                } else {
                    libm::pow((x + 0.055) / 1.055, 2.4)
                }
            }
            TransferFunction::Bt709 => {
                if x < 4.5 * BT709_BETA {
                    x / 4.5
                } else {
                    libm::pow((x + BT709_ALPHA) / (1.0 + BT709_ALPHA), 1.0 / 0.45)
                }
            }
            _ => unreachable!("planner admits only Linear, sRGB and BT.709"),
        }
    }

    #[inline]
    fn encode(tf: TransferFunction, x: f64) -> f64 {
        match tf {
            TransferFunction::Linear => x,
            TransferFunction::Srgb => {
                if x <= 0.0031308 {
                    x * 12.92
                } else {
                    1.055 * libm::pow(x, 1.0 / 2.4) - 0.055
                }
            }
            TransferFunction::Bt709 => {
                if x < BT709_BETA {
                    x * 4.5
                } else {
                    (1.0 + BT709_ALPHA) * libm::pow(x, 0.45) - BT709_ALPHA
                }
            }
            _ => unreachable!("planner admits only Linear, sRGB and BT.709"),
        }
    }

    let pair = match (from, to) {
        (TransferFunction::Linear, TransferFunction::Bt709) => 0,
        (TransferFunction::Linear, TransferFunction::Srgb) => 1,
        (TransferFunction::Bt709, TransferFunction::Linear) => 2,
        (TransferFunction::Bt709, TransferFunction::Srgb) => 3,
        (TransferFunction::Srgb, TransferFunction::Linear) => 4,
        (TransferFunction::Srgb, TransferFunction::Bt709) => 5,
        _ => unreachable!("planner admits only distinct Linear, sRGB and BT.709 pairs"),
    };
    let table = TABLES[pair].get_or_init(|| {
        let mut table: alloc::boxed::Box<[u8; 65536]> = alloc::vec![0; 65536]
            .into_boxed_slice()
            .try_into()
            .expect("fixed-size SDR transfer table");
        for (code, output) in table.iter_mut().enumerate() {
            let encoded = encode(to, decode(from, code as f64 / 65535.0));
            *output = libm::floor(encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        table
    });

    let channels = layout.channels();
    let count = width * channels;
    let (pairs, _) = src[..count * 2].as_chunks::<2>();
    if layout.has_alpha() {
        for (source_pixel, dest_pixel) in pairs
            .chunks_exact(channels)
            .zip(dst[..count].chunks_exact_mut(channels))
        {
            for (source, dest) in source_pixel[..channels - 1]
                .iter()
                .zip(dest_pixel[..channels - 1].iter_mut())
            {
                *dest = table[usize::from(u16::from_ne_bytes(*source))];
            }
            let alpha = u16::from_ne_bytes(source_pixel[channels - 1]);
            dest_pixel[channels - 1] = ((u32::from(alpha) * 255 + 32767) / 65535) as u8;
        }
    } else {
        for (source, dest) in pairs.iter().zip(dst[..count].iter_mut()) {
            *dest = table[usize::from(u16::from_ne_bytes(*source))];
        }
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn u8_to_u16_lanes(src: &[u8], dst: &mut [[u8; 2]]) {
    for (&s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (u16::from(s) * 257).to_ne_bytes();
    }
}

/// u8 → u16: v * 257.
fn u8_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let (pairs, _) = dst[..count * 2].as_chunks_mut::<2>();
    u8_to_u16_lanes(&src[..count], pairs);
}

/// u16 → f32: v / 65535.0.
fn u16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src_u16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    crate::float_simd::convert_u16_to_f32_slice(src_u16, dst_f32);
}

/// f32 → u16: clamp `[0,1]`, * 65535 + 0.5.
fn f32_to_u16(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    channels: usize,
    in_facts: zenpixels::sample::FloatFacts,
) {
    let count = width * channels;
    let src_f32: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    crate::float_simd::convert_f32_to_u16_slice(src_f32, dst_u16, in_facts);
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
/// Uses the existing `linear-srgb` transfer curve.
#[cfg(any(feature = "perceptual", all(test, feature = "hdr-transfers")))]
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    linear_srgb::default::pq_to_linear(v)
}

/// PQ inverse EOTF (OETF): linear light `[0,1]` → encoded `[0,1]`.
///
/// Uses the existing `linear-srgb` transfer curve.
#[cfg(any(feature = "perceptual", all(test, feature = "hdr-transfers")))]
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    linear_srgb::default::linear_to_pq(v)
}

/// Per-tier SIMD body for [`multiply_color_channels`]. `channels == 4` builds a
/// `[f, f, f, 1]`-repeating multiplier so the alpha lane (every 4th) is left
/// **untouched**; any other channel count scales every lane uniformly. Any `< 16`
/// remainder runs through a zero-padded stack chunk via the same SIMD body.
#[cfg(feature = "hdr-transfers")]
#[attune(_*_t, -_scalar_t, _v4_t(cfg(avx512)), define(f32x16))]
fn multiply_color_channels_tier(token: Token, buf: &mut [f32], channels: usize, factor: f32) {
    let mul = if channels == 4 {
        f32x16::from_array_t(
            token,
            [
                factor, factor, factor, 1.0, factor, factor, factor, 1.0, factor, factor, factor,
                1.0, factor, factor, factor, 1.0,
            ],
        )
    } else {
        f32x16::from_array_t(token, [factor; 16])
    };
    let (chunks, remainder) = buf.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = f32x16::from_array_t(token, *chunk);
        *chunk = (v * mul).to_array();
    }
    if !remainder.is_empty() {
        let mut pad = [0.0f32; 16];
        pad[..remainder.len()].copy_from_slice(remainder);
        let out = (f32x16::from_array_t(token, pad) * mul).to_array();
        remainder.copy_from_slice(&out[..remainder.len()]);
    }
}

#[cfg(feature = "hdr-transfers")]
#[cold]
#[inline(never)]
fn multiply_color_channels_tier_scalar_t(
    _token: archmage::ScalarToken,
    buf: &mut [f32],
    channels: usize,
    factor: f32,
) {
    if channels == 4 {
        for px in buf.as_chunks_mut::<4>().0 {
            px[0] *= factor;
            px[1] *= factor;
            px[2] *= factor;
        }
    } else {
        for v in buf {
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
#[cfg(feature = "hdr-transfers")]
pub(crate) fn multiply_color_channels(buf: &mut [f32], channels: usize, factor: f32) {
    if factor == 1.0 {
        return;
    }
    reattune!(
        multiply_color_channels_tier(buf, channels, factor),
        [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084)
// ---------------------------------------------------------------------------

/// PQ EOTF over an interleaved f32 slice, every lane.
#[cfg(feature = "hdr-transfers")]
pub(crate) fn pq_eotf_slice(buf: &mut [f32]) {
    linear_srgb::default::pq_to_linear_slice(buf);
}

/// PQ OETF over an interleaved f32 slice, every lane.
#[cfg(feature = "hdr-transfers")]
pub(crate) fn pq_oetf_slice(buf: &mut [f32]) {
    linear_srgb::default::linear_to_pq_slice(buf);
}

/// PQ U16 → Linear F32 (EOTF during depth conversion), alpha-preserving.
///
/// Widens the U16 codes (`garb` SIMD) and applies the PQ EOTF
/// ([`pq_eotf_slice`]) to land relative-linear, dividing the RGB lanes by `scale`
/// (`diffuse_white / 10000`). The EOTF runs over every lane; for alpha-bearing
/// layouts the alpha lane is then overwritten with its (un-transformed,
/// un-anchored) linear value. `scale == 1.0` is the identity anchor.
#[cfg(feature = "hdr-transfers")]
fn pq_u16_to_linear_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
) {
    let channels = layout.channels();
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    crate::float_simd::convert_u16_to_f32_slice(src16, &mut dstf[..count]);
    pq_eotf_slice(&mut dstf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, 1.0 / scale);
    restore_alpha_u16_f32(src16, &mut dstf[..count], layout);
}

/// Linear F32 → PQ U16 (OETF during depth conversion), alpha-preserving.
///
/// Anchors the RGB lanes (`× scale`, negatives → 0), applies the PQ
/// OETF ([`pq_oetf_slice`]) in fixed stack-sized chunks (the U16 output is
/// half-width, so it cannot host the in-place f32 transform), then narrows to U16
/// (`float_simd`). For alpha-bearing layouts the alpha lane is overwritten with its
/// linear → U16 value (never OETF'd or anchored). `scale == 1.0` is the identity.
#[cfg(feature = "hdr-transfers")]
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
        let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[off * 2..(off + n) * 2]);
        crate::float_simd::convert_f32_to_u16_slice_untrusted(b, dst16);
        restore_alpha_f32_u16(&srcf[off..off + n], dst16, layout);
        off += n;
    }
}

/// PQ F32 → Linear F32 (EOTF, same depth), alpha-preserving. SIMD.
///
/// `scale` is the `diffuse_white / 10000` anchor (see [`pq_u16_to_linear_f32`]);
/// the RGB lanes are divided by it after the EOTF and, for alpha-bearing layouts,
/// the alpha lane is restored to its (un-transformed) input. `scale == 1.0` is a
/// no-op so the un-anchored result depends only on [`pq_eotf_slice`].
#[cfg(feature = "hdr-transfers")]
fn pq_f32_to_linear_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
    in_facts: zenpixels::sample::FloatFacts,
) {
    use zenpixels::sample::FloatFacts;
    let channels = layout.channels();
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    if in_facts.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
        dstf[..count].copy_from_slice(&srcf[..count]);
    } else if in_facts.contains(FloatFacts::NO_NAN) {
        for (d, &s) in dstf[..count].iter_mut().zip(&srcf[..count]) {
            *d = s.clamp(0.0, 1.0);
        }
    } else {
        for (d, &s) in dstf[..count].iter_mut().zip(&srcf[..count]) {
            *d = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        }
    }
    pq_eotf_slice(&mut dstf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, 1.0 / scale);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → PQ F32 (OETF, same depth), alpha-preserving. SIMD.
///
/// `scale` is the `diffuse_white / 10000` anchor (see [`linear_f32_to_pq_u16`]);
/// the RGB lanes are multiplied by it before the OETF and, for alpha-bearing
/// layouts, the alpha lane is restored to its (un-transformed) linear input.
#[cfg(feature = "hdr-transfers")]
fn linear_f32_to_pq_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    scale: f32,
    in_facts: zenpixels::sample::FloatFacts,
) {
    use zenpixels::sample::FloatFacts;
    let channels = layout.channels();
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    multiply_color_channels(&mut dstf[..count], channels, scale);
    let scale_unit = (0.0..=1.0).contains(&scale);
    if scale_unit && in_facts.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
        // Every lane is already in [+0.0, 1.0].
    } else if scale.is_finite()
        && scale >= 0.0
        && in_facts.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE)
    {
        for d in &mut dstf[..count] {
            *d = d.clamp(0.0, 1.0);
        }
    } else {
        for d in &mut dstf[..count] {
            *d = if d.is_nan() { 0.0 } else { d.clamp(0.0, 1.0) };
        }
    }
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
#[cfg(all(test, feature = "hdr-transfers"))]
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    linear_srgb::default::linear_to_hlg(v)
}

/// HLG inverse OETF (EOTF): encoded `[0,1]` → scene-linear `[0,1]`.
///
/// Uses `fast_pow2f` from `linear-srgb` (no `libm` exp calls).
#[cfg(all(test, feature = "hdr-transfers"))]
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    linear_srgb::default::hlg_to_linear(v)
}

/// HLG U16 → Linear F32 (EOTF applied during depth conversion),
/// alpha-preserving (the alpha lane is carried linearly, `v / 65535`).
#[cfg(feature = "hdr-transfers")]
fn hlg_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    crate::float_simd::convert_u16_to_f32_slice(src16, dstf);
    linear_srgb::default::hlg_to_linear_slice(dstf);
    restore_alpha_u16_f32(src16, dstf, layout);
}

/// Linear F32 → HLG U16 (OETF applied during depth conversion),
/// alpha-preserving (the alpha lane is scaled linearly, never HLG-encoded).
#[cfg(feature = "hdr-transfers")]
fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    const LANES: usize = 256;
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let mut lanes = [0f32; LANES];
    for (s, d) in srcf
        .chunks(LANES)
        .zip(dst[..count * 2].chunks_mut(LANES * 2))
    {
        let encoded = &mut lanes[..s.len()];
        encoded.copy_from_slice(s);
        linear_srgb::default::linear_to_hlg_slice(encoded);
        crate::float_simd::convert_f32_to_u16_slice_untrusted(encoded, bytemuck::cast_slice_mut(d));
    }
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    restore_alpha_f32_u16(srcf, dst16, layout);
}

/// HLG F32 → Linear F32 (EOTF, same depth). SIMD-dispatched,
/// alpha-preserving.
#[cfg(feature = "hdr-transfers")]
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
#[cfg(feature = "hdr-transfers")]
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
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::bt709_to_linear_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
}

/// Linear F32 → BT.709 F32 (OETF, same depth). Alpha-preserving.
fn linear_f32_to_bt709_f32(src: &[u8], dst: &mut [u8], width: usize, layout: ChannelLayout) {
    let count = width * layout.channels();
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_bt709_slice(&mut dstf[..count]);
    restore_alpha_f32_f32(&srcf[..count], &mut dstf[..count], layout);
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

// Keep the existing SIMD approximation for magnitudes in [0,1]. Only values
// outside that calibrated domain use a checked high-precision power; reflect the result about zero.
// This changes neither the default clipped route nor positive in-range samples.
fn gamma22_f32_extended(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    inverse: bool,
    in_facts: zenpixels::sample::FloatFacts,
) {
    use zenpixels::sample::FloatFacts;
    let count = width * layout.channels();
    let source: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let output: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);

    if in_facts.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE) {
        // Every lane is in [+0.0, 1.0]: no abs(), no out-of-unit fallback, no copysign().
        output.copy_from_slice(source);
        if inverse {
            linear_srgb::default::linear_to_gamma_slice(output, ADOBE_GAMMA);
        } else {
            linear_srgb::default::gamma_to_linear_slice(output, ADOBE_GAMMA);
        }
        restore_alpha_f32_f32(source, output, layout);
        return;
    }

    let exponent = if inverse {
        1.0 / ADOBE_GAMMA
    } else {
        ADOBE_GAMMA
    };
    for (s, d) in source.iter().zip(output.iter_mut()) {
        *d = s.abs();
    }
    if inverse {
        linear_srgb::default::linear_to_gamma_slice(output, ADOBE_GAMMA);
    } else {
        linear_srgb::default::gamma_to_linear_slice(output, ADOBE_GAMMA);
    }
    for (s, d) in source.iter().zip(output.iter_mut()) {
        let mag = s.abs();
        if mag > 1.0 || mag.is_nan() {
            *d = libm::powf(mag, exponent);
        }
        *d = d.copysign(*s);
    }
    restore_alpha_f32_f32(source, output, layout);
}

// ---------------------------------------------------------------------------
// Alpha premultiplication
// ---------------------------------------------------------------------------
//
// Pattern: dispatch on (ChannelType, channels) to concrete #[attune(dispatch, -_neon, -_wasm128)]
// kernels. Each kernel has a flat per-pixel loop with fixed-size array
// slicing at the pixel boundary so LLVM can drop bounds checks and
// vectorize. Empirically ~10× faster than the previous big-match-in-fn
// shape on L2-sized rows (see benchmarks/premul_u16_2026-04-23_baseline).
// ---------------------------------------------------------------------------

// -- Straight → Premultiplied: per-(type, channels) kernels ------------------

#[inline(always)]
fn premul_u8_lane(c: u8, a: u8) -> u8 {
    let t = u32::from(c) * u32::from(a) + 128;
    ((t + (t >> 8)) >> 8) as u8
}

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn premul_u8_rgba(src: &[u8], dst: &mut [u8], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0.iter_mut())
    {
        let a = s[3];
        d[0] = premul_u8_lane(s[0], a);
        d[1] = premul_u8_lane(s[1], a);
        d[2] = premul_u8_lane(s[2], a);
        d[3] = a;
    }
}

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn premul_u8_ga(src: &[u8], dst: &mut [u8], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1] as u32;
        *d = [((s[0] as u32 * a + 127) / 255) as u8, s[1]];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn premul_u16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1] as u32;
        *d = [((s[0] as u32 * a + 32767) / 65535) as u16, s[1]];
    }
}

#[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn premul_u16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        let a = s[3] as u32;
        *d = [
            ((s[0] as u32 * a + 32767) / 65535) as u16,
            ((s[1] as u32 * a + 32767) / 65535) as u16,
            ((s[2] as u32 * a + 32767) / 65535) as u16,
            s[3],
        ];
    }
}

#[attune(dispatch, -_neon, -_wasm128)]
fn premul_f32_ga(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1];
        *d = [s[0] * a, a];
    }
}

#[inline(always)]
fn f16_alpha_block<const CHANNELS: usize, const UNPREMUL: bool>(
    sc: &[u16; 32],
    dc: &mut [u16; 32],
    scratch_src: &mut [f32; 32],
    scratch_dst: &mut [f32; 32],
) {
    f16_bits_to_f32_slice(sc, scratch_src);
    if CHANNELS == 2 {
        for i in 0..16 {
            let base = i * 2;
            let a = scratch_src[base + 1];
            if UNPREMUL {
                if a == 0.0 {
                    scratch_dst[base] = 0.0;
                    scratch_dst[base + 1] = 0.0;
                } else {
                    let inv_a = 1.0 / a;
                    scratch_dst[base] = scratch_src[base] * inv_a;
                    scratch_dst[base + 1] = a;
                }
            } else {
                scratch_dst[base] = scratch_src[base] * a;
                scratch_dst[base + 1] = a;
            }
        }
    } else {
        for i in 0..8 {
            let base = i * 4;
            let a = scratch_src[base + 3];
            if UNPREMUL {
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
            } else {
                scratch_dst[base] = scratch_src[base] * a;
                scratch_dst[base + 1] = scratch_src[base + 1] * a;
                scratch_dst[base + 2] = scratch_src[base + 2] * a;
                scratch_dst[base + 3] = a;
            }
        }
    }
    f32_to_f16_bits_slice(scratch_dst, dc);
    // Alpha is a payload, not an arithmetic output. Preserve NaN bits and signed zero.
    for lane in (CHANNELS - 1..32).step_by(CHANNELS) {
        dc[lane] = sc[lane];
    }
}

/// Unified F16 premul / unpremul via 32-lane F16C/NEON-fp16 scratch with padded
/// remainder block (zero scalar `f16_bits_to_f32` tail codegen).
fn f16_alpha_chunked<const CHANNELS: usize, const UNPREMUL: bool>(
    src: &[u16],
    dst: &mut [u16],
    width: usize,
) {
    const { assert!(CHANNELS == 2 || CHANNELS == 4) };
    const CHUNK_LANES: usize = 32;

    let mut scratch_src = [0.0f32; CHUNK_LANES];
    let mut scratch_dst = [0.0f32; CHUNK_LANES];

    let n = width * CHANNELS;
    let (s_chunks, s_rem) = src[..n].as_chunks::<CHUNK_LANES>();
    let (d_chunks, d_rem) = dst[..n].as_chunks_mut::<CHUNK_LANES>();
    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        f16_alpha_block::<CHANNELS, UNPREMUL>(sc, dc, &mut scratch_src, &mut scratch_dst);
    }
    if !s_rem.is_empty() {
        let mut pad_in = [0u16; CHUNK_LANES];
        let mut pad_out = [0u16; CHUNK_LANES];
        pad_in[..s_rem.len()].copy_from_slice(s_rem);
        f16_alpha_block::<CHANNELS, UNPREMUL>(
            &pad_in,
            &mut pad_out,
            &mut scratch_src,
            &mut scratch_dst,
        );
        d_rem.copy_from_slice(&pad_out[..d_rem.len()]);
    }
}

fn premul_f16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    f16_alpha_chunked::<2, false>(src, dst, width);
}

fn premul_f16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    f16_alpha_chunked::<4, false>(src, dst, width);
}

// -- Premultiplied → Straight: per-(type, channels) kernels ------------------
//
// Each arm handles a == 0 by zeroing all channels (that's the only
// useful answer for a premultiplied pixel with zero alpha — the color
// channels are already zero, but we defensively zero anyway).

fn unpremul_u8_ga(src: &[u8], dst: &mut [u8], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1];
        if a == 0 {
            *d = [0, 0];
        } else {
            let a32 = a as u32;
            *d = [((s[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8, a];
        }
    }
}

fn unpremul_u8_rgba(src: &[u8], dst: &mut [u8], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        let a = s[3];
        if a == 0 {
            *d = [0, 0, 0, 0];
        } else {
            let a32 = a as u32;
            *d = [
                ((s[0] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
                ((s[1] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
                ((s[2] as u32 * 255 + a32 / 2) / a32).min(255) as u8,
                a,
            ];
        }
    }
}

fn unpremul_u16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1];
        if a == 0 {
            *d = [0, 0];
        } else {
            let a32 = a as u32;
            *d = [((s[0] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16, a];
        }
    }
}

fn unpremul_u16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    for (s, d) in src[..width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[..width * 4].as_chunks_mut::<4>().0)
    {
        let a = s[3];
        if a == 0 {
            *d = [0, 0, 0, 0];
        } else {
            let a32 = a as u32;
            *d = [
                ((s[0] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16,
                ((s[1] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16,
                ((s[2] as u32 * 65535 + a32 / 2) / a32).min(65535) as u16,
                a,
            ];
        }
    }
}

#[attune(dispatch, -_v3, -_neon, -_wasm128, +v4x(cfg(avx512)))]
fn unpremul_f32_ga(src: &[f32], dst: &mut [f32], width: usize) {
    for (s, d) in src[..width * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[..width * 2].as_chunks_mut::<2>().0)
    {
        let a = s[1];
        if a == 0.0 {
            *d = [0.0, 0.0];
        } else {
            *d = [s[0] / a, a];
        }
    }
}

fn unpremul_f16_ga(src: &[u16], dst: &mut [u16], width: usize) {
    f16_alpha_chunked::<2, true>(src, dst, width);
}

fn unpremul_f16_rgba(src: &[u16], dst: &mut [u16], width: usize) {
    f16_alpha_chunked::<4, true>(src, dst, width);
}

fn step_premul_u8(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    let ch = op.from.layout().channels();
    match (&op.step, ch) {
        (ConvertStep::StraightToPremul, 2) => premul_u8_ga(&src[..w * 2], &mut dst[..w * 2], w),
        (ConvertStep::StraightToPremul, 4) => premul_u8_rgba(&src[..w * 4], &mut dst[..w * 4], w),
        (ConvertStep::PremulToStraight, 2) => unpremul_u8_ga(&src[..w * 2], &mut dst[..w * 2], w),
        (ConvertStep::PremulToStraight, 4) => unpremul_u8_rgba(&src[..w * 4], &mut dst[..w * 4], w),
        _ => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

fn step_premul_u16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    let ch = op.from.layout().channels();
    match (&op.step, ch) {
        (ConvertStep::StraightToPremul, 2) => {
            let n = w * 4;
            premul_u16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::StraightToPremul, 4) => {
            let n = w * 8;
            premul_u16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::PremulToStraight, 2) => {
            let n = w * 4;
            unpremul_u16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::PremulToStraight, 4) => {
            let n = w * 8;
            unpremul_u16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        _ => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

fn step_premul_f16(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    let ch = op.from.layout().channels();
    match (&op.step, ch) {
        (ConvertStep::StraightToPremul, 2) => {
            let n = w * 4;
            premul_f16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::StraightToPremul, 4) => {
            let n = w * 8;
            premul_f16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::PremulToStraight, 2) => {
            let n = w * 4;
            unpremul_f16_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::PremulToStraight, 4) => {
            let n = w * 8;
            unpremul_f16_rgba(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        _ => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

fn step_premul_f32(
    op: &PlannedOperation,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    _hdr: &mut HdrKernelScratch,
) {
    let w = width as usize;
    let ch = op.from.layout().channels();
    match (&op.step, ch) {
        (ConvertStep::StraightToPremul, 2) => {
            let n = w * 8;
            premul_f32_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::StraightToPremul, 4) => {
            let n = w * 16;
            crate::float_simd::premul_rgba_f32_slice(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                op.in_trust.facts,
            );
        }
        (ConvertStep::PremulToStraight, 2) => {
            let n = w * 8;
            unpremul_f32_ga(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                w,
            );
        }
        (ConvertStep::PremulToStraight, 4) => {
            let n = w * 16;
            crate::float_simd::unpremul_rgba_f32_slice(
                bytemuck::cast_slice(&src[..n]),
                bytemuck::cast_slice_mut(&mut dst[..n]),
                op.in_trust.facts,
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

#[cfg(feature = "oklab")]
mod oklab_kernels {
    use super::*;
    use crate::oklab::{
        lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab, rgb_to_oklab_spec,
    };

    #[inline]
    fn oklab_is_finite_nonneg(m1: &[[f32; 3]; 3], in_trust: super::super::StepFloatTrust) -> bool {
        use zenpixels::sample::FloatFacts;
        let cf = in_trust.color_facts;
        cf.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NON_NEGATIVE)
            && in_trust.color_bounded
            && m1.as_flattened().iter().all(|&c| c >= (1.0 / 64.0))
    }

    type OklabChunksFn<const N: usize> = fn(&[[f32; N]], &mut [[f32; N]], &[[f32; 3]; 3]);

    /// Shared padded-chunk driver for Oklab 16-pixel (`N = 48` or `64`) slice kernels.
    /// Runs full chunks directly and pads any `< N` remainder with `0.0` through the
    /// exact same chunk kernel so no SIMD tier emits scalar tail math.
    #[inline(always)]
    fn run_oklab_chunked<const N: usize>(
        src: &[f32],
        dst: &mut [f32],
        m: &[[f32; 3]; 3],
        chunks_fn: OklabChunksFn<N>,
    ) {
        let len = src.len().min(dst.len());
        let (s_chunks, s_rem) = src[..len].as_chunks::<N>();
        let (d_chunks, d_rem) = dst[..len].as_chunks_mut::<N>();
        if !s_chunks.is_empty() {
            chunks_fn(s_chunks, d_chunks, m);
        }
        if !s_rem.is_empty() {
            finish_oklab_tail(s_rem, d_rem, m, chunks_fn);
        }
    }

    #[cold]
    #[inline(never)]
    fn finish_oklab_tail<const N: usize>(
        s_rem: &[f32],
        d_rem: &mut [f32],
        m: &[[f32; 3]; 3],
        chunks_fn: OklabChunksFn<N>,
    ) {
        let mut sin = [0.0f32; N];
        let mut dout = [0.0f32; N];
        sin[..s_rem.len()].copy_from_slice(s_rem);
        chunks_fn(
            core::slice::from_ref(&sin),
            core::slice::from_mut(&mut dout),
            m,
        );
        d_rem.copy_from_slice(&dout[..d_rem.len()]);
    }

    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn rgb_to_oklab_3ch_finite_nonneg_chunks(
        src: &[[f32; 48]],
        dst: &mut [[f32; 48]],
        m1: &[[f32; 3]; 3],
    ) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 3;
                let [l, a, b] =
                    rgb_to_oklab_spec::<true, false, true, false>(s[i], s[i + 1], s[i + 2], m1);
                d[i] = l;
                d[i + 1] = a;
                d[i + 2] = b;
            }
        }
    }

    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn rgb_to_oklab_4ch_finite_nonneg_chunks(
        src: &[[f32; 64]],
        dst: &mut [[f32; 64]],
        m1: &[[f32; 3]; 3],
    ) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 4;
                let [l, a, b] =
                    rgb_to_oklab_spec::<true, false, true, false>(s[i], s[i + 1], s[i + 2], m1);
                d[i] = l;
                d[i + 1] = a;
                d[i + 2] = b;
                d[i + 3] = s[i + 3];
            }
        }
    }

    /// Linear RGB f32 → Oklab f32 (3 channels).
    ///
    /// # Panics
    ///
    /// Panics if `primaries` is `Unknown`. The plan should have rejected this.
    pub(super) fn linear_rgb_to_oklab_f32(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
        in_trust: super::super::StepFloatTrust,
    ) {
        let m1 = rgb_to_lms_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );
        let rows = m1.rows();
        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
        if oklab_is_finite_nonneg(rows, in_trust) {
            run_oklab_chunked::<48>(srcf, dstf, rows, rgb_to_oklab_3ch_finite_nonneg_chunks);
        } else {
            run_oklab_chunked::<48>(srcf, dstf, rows, rgb_to_oklab_3ch_chunks);
        }
    }

    pub(super) fn linear_rgb_to_oklab_f32_untrusted(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
    ) {
        let m1 = rgb_to_lms_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );
        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
        run_oklab_chunked::<48>(srcf, dstf, m1.rows(), rgb_to_oklab_3ch_chunks);
    }

    /// 3-channel RGB→Oklab inner loop (16 pixels = 48 f32s per chunk).
    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn rgb_to_oklab_3ch_chunks(src: &[[f32; 48]], dst: &mut [[f32; 48]], m1: &[[f32; 3]; 3]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 3;
                let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
                d[i] = l;
                d[i + 1] = a;
                d[i + 2] = b;
            }
        }
    }

    /// Oklab f32 → Linear RGB f32 (3 channels).
    pub(super) fn oklab_to_linear_rgb_f32(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
    ) {
        let m1_inv = lms_to_rgb_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );

        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
        run_oklab_chunked::<48>(srcf, dstf, m1_inv.rows(), oklab_to_rgb_3ch_chunks);
    }

    /// 3-channel Oklab→RGB inner loop (16 pixels = 48 f32s per chunk).
    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn oklab_to_rgb_3ch_chunks(src: &[[f32; 48]], dst: &mut [[f32; 48]], m1_inv: &[[f32; 3]; 3]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 3;
                let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
                d[i] = r;
                d[i + 1] = g;
                d[i + 2] = b;
            }
        }
    }

    /// Linear RGBA f32 → Oklaba f32 (4 channels, alpha preserved).
    pub(super) fn linear_rgba_to_oklaba_f32(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
        in_trust: super::super::StepFloatTrust,
    ) {
        let m1 = rgb_to_lms_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );
        let rows = m1.rows();
        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
        if oklab_is_finite_nonneg(rows, in_trust) {
            run_oklab_chunked::<64>(srcf, dstf, rows, rgb_to_oklab_4ch_finite_nonneg_chunks);
        } else {
            run_oklab_chunked::<64>(srcf, dstf, rows, rgb_to_oklab_4ch_chunks);
        }
    }

    pub(super) fn linear_rgba_to_oklaba_f32_untrusted(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
    ) {
        let m1 = rgb_to_lms_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );
        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
        run_oklab_chunked::<64>(srcf, dstf, m1.rows(), rgb_to_oklab_4ch_chunks);
    }

    /// 4-channel RGBA→Oklaba inner loop (16 pixels = 64 f32s per chunk).
    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn rgb_to_oklab_4ch_chunks(src: &[[f32; 64]], dst: &mut [[f32; 64]], m1: &[[f32; 3]; 3]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 4;
                let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
                d[i] = l;
                d[i + 1] = a;
                d[i + 2] = b;
                d[i + 3] = s[i + 3]; // alpha unchanged
            }
        }
    }

    /// Oklaba f32 → Linear RGBA f32 (4 channels, alpha preserved).
    pub(super) fn oklaba_to_linear_rgba_f32(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        primaries: ColorPrimaries,
    ) {
        let m1_inv = lms_to_rgb_matrix(primaries).expect(
            "Oklab conversion requires known primaries (plan should have rejected Unknown)",
        );

        let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
        let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
        run_oklab_chunked::<64>(srcf, dstf, m1_inv.rows(), oklab_to_rgb_4ch_chunks);
    }

    /// 4-channel Oklaba→RGBA inner loop (16 pixels = 64 f32s per chunk).
    #[attune(dispatch, -_neon, -_wasm128, +v4x(cfg(avx512)))]
    fn oklab_to_rgb_4ch_chunks(src: &[[f32; 64]], dst: &mut [[f32; 64]], m1_inv: &[[f32; 3]; 3]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            for p in 0..16 {
                let i = p * 4;
                let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
                d[i] = r;
                d[i + 1] = g;
                d[i + 2] = b;
                d[i + 3] = s[i + 3]; // alpha unchanged
            }
        }
    }
}

#[cfg(feature = "oklab")]
use oklab_kernels::{
    linear_rgb_to_oklab_f32, linear_rgb_to_oklab_f32_untrusted, linear_rgba_to_oklaba_f32,
    linear_rgba_to_oklaba_f32_untrusted, oklab_to_linear_rgb_f32, oklaba_to_linear_rgba_f32,
};

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
/// Multiply-add for kernels: one rounding on every native tier, so output doesn't depend on the CPU.
/// WASM stays relaxed unless `precise-portable` asks for the same bits, because fused is software
/// there unless the engine fuses.
#[inline(always)]
fn fused_mul_add<T: magetypes::simd::backends::F32x8Backend>(
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
    c: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    #[cfg(all(target_arch = "wasm32", not(feature = "precise-portable")))]
    {
        a.mul_add_relaxed(b, c)
    }
    #[cfg(not(all(target_arch = "wasm32", not(feature = "precise-portable"))))]
    {
        a.mul_add_fused(b, c)
    }
}

#[attune(_*_t, -_scalar_t, _v4_t(cfg(avx512)), define(f32x8))]
fn gamut_matrix_f32_tier<const CHANNELS: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    matrix: &[f32; 9],
) {
    const { assert!(CHANNELS == 3 || CHANNELS == 4) };
    const LANES: usize = 8;
    let m = matrix;
    let m00 = f32x8::splat_t(token, m[0]);
    let m01 = f32x8::splat_t(token, m[1]);
    let m02 = f32x8::splat_t(token, m[2]);
    let m10 = f32x8::splat_t(token, m[3]);
    let m11 = f32x8::splat_t(token, m[4]);
    let m12 = f32x8::splat_t(token, m[5]);
    let m20 = f32x8::splat_t(token, m[6]);
    let m21 = f32x8::splat_t(token, m[7]);
    let m22 = f32x8::splat_t(token, m[8]);

    let (s_px, _) = src[..width * CHANNELS].as_chunks::<CHANNELS>();
    let (d_px, _) = dst[..width * CHANNELS].as_chunks_mut::<CHANNELS>();
    let (s_blocks, s_tail) = s_px.as_chunks::<LANES>();
    let (d_blocks, d_tail) = d_px.as_chunks_mut::<LANES>();

    let mut r = [0.0f32; LANES];
    let mut g = [0.0f32; LANES];
    let mut b = [0.0f32; LANES];

    for (sb, db) in s_blocks.iter().zip(d_blocks.iter_mut()) {
        for i in 0..LANES {
            r[i] = sb[i][0];
            g[i] = sb[i][1];
            b[i] = sb[i][2];
        }
        let rl = f32x8::from_array_t(token, r);
        let gl = f32x8::from_array_t(token, g);
        let bl = f32x8::from_array_t(token, b);

        let or = fused_mul_add(m00, rl, fused_mul_add(m01, gl, m02 * bl));
        let og = fused_mul_add(m10, rl, fused_mul_add(m11, gl, m12 * bl));
        let ob = fused_mul_add(m20, rl, fused_mul_add(m21, gl, m22 * bl));

        let ro = or.to_array();
        let go = og.to_array();
        let bo = ob.to_array();

        for i in 0..LANES {
            db[i][0] = ro[i];
            db[i][1] = go[i];
            db[i][2] = bo[i];
            if CHANNELS == 4 {
                db[i][3] = sb[i][3];
            }
        }
    }

    // Zero-padded final SIMD block (zero scalar fused_row tail codegen).
    if !s_tail.is_empty() {
        r = [0.0; LANES];
        g = [0.0; LANES];
        b = [0.0; LANES];
        for (i, sp) in s_tail.iter().enumerate() {
            r[i] = sp[0];
            g[i] = sp[1];
            b[i] = sp[2];
        }
        let rl = f32x8::from_array_t(token, r);
        let gl = f32x8::from_array_t(token, g);
        let bl = f32x8::from_array_t(token, b);

        let ro = fused_mul_add(m00, rl, fused_mul_add(m01, gl, m02 * bl)).to_array();
        let go = fused_mul_add(m10, rl, fused_mul_add(m11, gl, m12 * bl)).to_array();
        let bo = fused_mul_add(m20, rl, fused_mul_add(m21, gl, m22 * bl)).to_array();

        for (i, (sp, dp)) in s_tail.iter().zip(d_tail.iter_mut()).enumerate() {
            dp[0] = ro[i];
            dp[1] = go[i];
            dp[2] = bo[i];
            if CHANNELS == 4 {
                dp[3] = sp[3];
            }
        }
    }
}

#[cold]
#[inline(never)]
fn gamut_matrix_f32_tier_scalar_t<const CHANNELS: usize>(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    matrix: &[f32; 9],
) {
    const { assert!(CHANNELS == 3 || CHANNELS == 4) };
    let m = matrix;
    let (s_px, _) = src[..width * CHANNELS].as_chunks::<CHANNELS>();
    let (d_px, _) = dst[..width * CHANNELS].as_chunks_mut::<CHANNELS>();
    for (sp, dp) in s_px.iter().zip(d_px.iter_mut()) {
        let (rv, gv, bv) = (sp[0], sp[1], sp[2]);
        let row = |i: usize| crate::fast_gamut::fused_row(m[i], m[i + 1], m[i + 2], rv, gv, bv);
        dp[0] = row(0);
        dp[1] = row(3);
        dp[2] = row(6);
        if CHANNELS == 4 {
            dp[3] = sp[3];
        }
    }
}

/// Apply a 3×3 gamut matrix to a row of linear RGB f32 pixels. SIMD-dispatched
/// (AVX-512 / AVX2 / NEON / WASM-SIMD128 / Scalar) via `archmage::reattune!`.
fn gamut_matrix_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    reattune!(
        gamut_matrix_f32_tier::<3>(s, d, width, matrix),
        [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Apply a 3×3 gamut matrix to a row of linear RGBA f32 pixels (alpha
/// passthrough). SIMD-dispatched via `archmage::reattune!` — same kernel as
/// the RGB variant, parameterised on `CHANNELS == 4`.
fn gamut_matrix_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, matrix: &[f32; 9]) {
    let s: &[f32] = bytemuck::cast_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    reattune!(
        gamut_matrix_f32_tier::<4>(s, d, width, matrix),
        [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[cfg(test)]
mod gray_f16_exact {
    use super::*;
    use crate::f16_scalar::{f16_bits_to_f32, f32_to_f16_bits};
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    const WEIGHTS: [[f32; 3]; 3] = [
        [0.2126, 0.7152, 0.0722],
        [0.299, 0.587, 0.114],
        [1.0, 0.0, 0.0],
    ];

    /// One sample at a time through the software conversions.
    fn reference<const N: usize>(src: &[u16], dst: &mut [u16], w: [f32; 3]) {
        for (s, d) in src.as_chunks::<N>().0.iter().zip(dst) {
            let y = f16_bits_to_f32(s[0]) * w[0]
                + f16_bits_to_f32(s[1]) * w[1]
                + f16_bits_to_f32(s[2]) * w[2];
            *d = f32_to_f16_bits(y);
        }
    }

    /// `N`-channel rows putting `v` in `slot`; the other samples come from xorshift.
    fn row<const N: usize>(v: u16, slot: usize, pixels: usize, x: &mut u32) -> Vec<u16> {
        let mut out = vec![0u16; pixels * N];
        for p in out.as_chunks_mut::<N>().0 {
            for c in p.iter_mut() {
                *x ^= *x << 13;
                *x ^= *x >> 17;
                *x ^= *x << 5;
                *c = *x as u16;
            }
            p[slot] = v;
        }
        out
    }

    fn check<const N: usize>(
        kernel: fn(&[u16], &mut [u16], usize, [f32; 3]),
        slot: usize,
        v_step: usize,
    ) {
        let mut x = 0x1234_5678u32;
        let mut got = vec![0u16; 300];
        let mut want = vec![0u16; 300];
        for w in WEIGHTS {
            for v in (0..=u16::MAX).step_by(v_step) {
                let src = row::<N>(v, slot, 96, &mut x);
                kernel(&src, &mut got, 96, w);
                reference::<N>(&src, &mut want, w);
                for i in 0..96 {
                    let nan = |h: u16| h & 0x7C00 == 0x7C00 && h & 0x03FF != 0;
                    assert!(
                        got[i] == want[i] || (nan(got[i]) && nan(want[i])),
                        "input {:04x?}, weights {w:?}: got {:04x}, want {:04x}",
                        &src[i * N..i * N + 3],
                        got[i],
                        want[i],
                    );
                }
            }
            // Widths on both sides of the chunk boundary. Samples past `width` stay untouched.
            for width in [0, 1, 7, 8, 15, 16, 17, 127, 128, 129, 255, 256, 257, 300] {
                let src = row::<N>(0x3c00, slot, width.max(1), &mut x);
                got.fill(0xAAAA);
                kernel(&src, &mut got, width, w);
                reference::<N>(&src[..width * N], &mut want, w);
                let same = |(a, b): (&u16, &u16)| {
                    a == b
                        || (a & 0x7C00 == 0x7C00
                            && b & 0x7C00 == 0x7C00
                            && a & 0x3FF != 0
                            && b & 0x3FF != 0)
                };
                assert!(
                    got[..width].iter().zip(&want[..width]).all(same),
                    "width {width}"
                );
                assert!(got[width..].iter().all(|&s| s == 0xAAAA), "width {width}");
            }
        }
    }

    /// Every eighth half-float value in each channel, at every dispatch tier: the luma kernel
    /// must match per-sample software conversion bit for bit. NaN stays NaN; which input's
    /// payload survives depends on vector operand order, as in the `f32` kernel.
    #[test]
    fn matches_software_conversion_at_every_tier() {
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            for slot in 0..3 {
                check::<3>(rgb_to_gray_f16, slot, 1 << 3);
                check::<4>(rgba_to_gray_f16, slot, 1 << 3);
            }
        });
    }

    /// Every half-float value at the host's best tier.
    #[test]
    fn matches_software_conversion_exhaustively() {
        for slot in 0..3 {
            check::<3>(rgb_to_gray_f16, slot, 1);
            check::<4>(rgba_to_gray_f16, slot, 1);
        }
    }
}
