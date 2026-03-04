//! Alpha premultiplication + Oklab conversion kernels.

use core::cmp::min;

use crate::oklab::{lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab};
use crate::{ChannelLayout, ChannelType, ColorPrimaries};

/// Straight → Premultiplied alpha (in-place copy from src to dst).
pub(super) fn straight_to_premul(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    let alpha_idx = channels - 1;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let base = i * channels;
                let a = src[base + alpha_idx] as u32;
                for c in 0..alpha_idx {
                    dst[base + c] = ((src[base + c] as u32 * a + 128) / 255) as u8;
                }
                dst[base + alpha_idx] = src[base + alpha_idx];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * channels * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * channels * 4]);
            for i in 0..width {
                let base = i * channels;
                let a = srcf[base + alpha_idx];
                for c in 0..alpha_idx {
                    dstf[base + c] = srcf[base + c] * a;
                }
                dstf[base + alpha_idx] = a;
            }
        }
        _ => {
            // Fallback: copy.
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

/// Premultiplied → Straight alpha.
pub(super) fn premul_to_straight(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    let alpha_idx = channels - 1;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let base = i * channels;
                let a = src[base + alpha_idx];
                if a == 0 {
                    for c in 0..channels {
                        dst[base + c] = 0;
                    }
                } else {
                    let a32 = a as u32;
                    for c in 0..alpha_idx {
                        dst[base + c] = ((src[base + c] as u32 * 255 + a32 / 2) / a32) as u8;
                    }
                    dst[base + alpha_idx] = a;
                }
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * channels * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * channels * 4]);
            for i in 0..width {
                let base = i * channels;
                let a = srcf[base + alpha_idx];
                if a == 0.0 {
                    for c in 0..channels {
                        dstf[base + c] = 0.0;
                    }
                } else {
                    let inv_a = 1.0 / a;
                    for c in 0..alpha_idx {
                        dstf[base + c] = srcf[base + c] * inv_a;
                    }
                    dstf[base + alpha_idx] = a;
                }
            }
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

/// Linear RGB f32 → Oklab f32 (3 channels).
pub(super) fn linear_rgb_to_oklab_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    primaries: ColorPrimaries,
) {
    let Some(m1) = rgb_to_lms_matrix(primaries) else {
        // Fallback: copy.
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    };

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);

    for i in 0..width {
        let s = i * 3;
        let [l, a, b] = rgb_to_oklab(srcf[s], srcf[s + 1], srcf[s + 2], &m1);
        dstf[s] = l;
        dstf[s + 1] = a;
        dstf[s + 2] = b;
    }
}

/// Oklab f32 → Linear RGB f32 (3 channels).
pub(super) fn oklab_to_linear_rgb_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    primaries: ColorPrimaries,
) {
    let Some(m1_inv) = lms_to_rgb_matrix(primaries) else {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    };

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);

    for i in 0..width {
        let s = i * 3;
        let [r, g, b] = oklab_to_rgb(srcf[s], srcf[s + 1], srcf[s + 2], &m1_inv);
        dstf[s] = r;
        dstf[s + 1] = g;
        dstf[s + 2] = b;
    }
}

/// Linear RGBA f32 → Oklaba f32 (4 channels, alpha preserved).
pub(super) fn linear_rgba_to_oklaba_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    primaries: ColorPrimaries,
) {
    let Some(m1) = rgb_to_lms_matrix(primaries) else {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    };

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);

    for i in 0..width {
        let s = i * 4;
        let [l, a, b] = rgb_to_oklab(srcf[s], srcf[s + 1], srcf[s + 2], &m1);
        dstf[s] = l;
        dstf[s + 1] = a;
        dstf[s + 2] = b;
        dstf[s + 3] = srcf[s + 3]; // alpha unchanged
    }
}

/// Oklaba f32 → Linear RGBA f32 (4 channels, alpha preserved).
pub(super) fn oklaba_to_linear_rgba_f32(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    primaries: ColorPrimaries,
) {
    let Some(m1_inv) = lms_to_rgb_matrix(primaries) else {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    };

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);

    for i in 0..width {
        let s = i * 4;
        let [r, g, b] = oklab_to_rgb(srcf[s], srcf[s + 1], srcf[s + 2], &m1_inv);
        dstf[s] = r;
        dstf[s + 1] = g;
        dstf[s + 2] = b;
        dstf[s + 3] = srcf[s + 3]; // alpha unchanged
    }
}
