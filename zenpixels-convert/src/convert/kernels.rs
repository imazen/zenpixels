//! Channel layout kernels (swizzle, gray, alpha add/drop).

use crate::ChannelType;

/// BGRA ↔ RGBA swizzle.
pub(super) fn swizzle_bgra_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    let bps = ch_type.byte_size(); // bytes per sample
    let pixel_bytes = 4 * bps;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let s = i * pixel_bytes;
                let d = i * pixel_bytes;
                dst[d] = src[s + 2]; // R ← B (or B ← R)
                dst[d + 1] = src[s + 1]; // G ← G
                dst[d + 2] = src[s]; // B ← R (or R ← B)
                dst[d + 3] = src[s + 3]; // A ← A
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * pixel_bytes]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * pixel_bytes]);
            for i in 0..width {
                let s = i * 4;
                dst16[s] = src16[s + 2];
                dst16[s + 1] = src16[s + 1];
                dst16[s + 2] = src16[s];
                dst16[s + 3] = src16[s + 3];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * pixel_bytes]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * pixel_bytes]);
            for i in 0..width {
                let s = i * 4;
                dstf[s] = srcf[s + 2];
                dstf[s + 1] = srcf[s + 1];
                dstf[s + 2] = srcf[s];
                dstf[s + 3] = srcf[s + 3];
            }
        }
        _ => {}
    }
}

/// Add opaque alpha channel (3ch → 4ch).
pub(super) fn add_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 4] = src[i * 3];
                dst[i * 4 + 1] = src[i * 3 + 1];
                dst[i * 4 + 2] = src[i * 3 + 2];
                dst[i * 4 + 3] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 6]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                dst16[i * 4] = src16[i * 3];
                dst16[i * 4 + 1] = src16[i * 3 + 1];
                dst16[i * 4 + 2] = src16[i * 3 + 2];
                dst16[i * 4 + 3] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                dstf[i * 4] = srcf[i * 3];
                dstf[i * 4 + 1] = srcf[i * 3 + 1];
                dstf[i * 4 + 2] = srcf[i * 3 + 2];
                dstf[i * 4 + 3] = 1.0;
            }
        }
        _ => {}
    }
}

/// Drop alpha channel (4ch → 3ch).
pub(super) fn drop_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 3] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4 + 1];
                dst[i * 3 + 2] = src[i * 4 + 2];
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                dst16[i * 3] = src16[i * 4];
                dst16[i * 3 + 1] = src16[i * 4 + 1];
                dst16[i * 3 + 2] = src16[i * 4 + 2];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                dstf[i * 3] = srcf[i * 4];
                dstf[i * 3 + 1] = srcf[i * 4 + 1];
                dstf[i * 3 + 2] = srcf[i * 4 + 2];
            }
        }
        _ => {}
    }
}

/// Gray → RGB (replicate).
pub(super) fn gray_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i];
                dst[i * 3] = g;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = g;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let g = src16[i];
                dst16[i * 3] = g;
                dst16[i * 3 + 1] = g;
                dst16[i * 3 + 2] = g;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                let g = srcf[i];
                dstf[i * 3] = g;
                dstf[i * 3 + 1] = g;
                dstf[i * 3 + 2] = g;
            }
        }
        _ => {}
    }
}

/// Gray → RGBA (replicate + opaque alpha).
pub(super) fn gray_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i];
                dst[i * 4] = g;
                dst[i * 4 + 1] = g;
                dst[i * 4 + 2] = g;
                dst[i * 4 + 3] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                let g = src16[i];
                dst16[i * 4] = g;
                dst16[i * 4 + 1] = g;
                dst16[i * 4 + 2] = g;
                dst16[i * 4 + 3] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                let g = srcf[i];
                dstf[i * 4] = g;
                dstf[i * 4 + 1] = g;
                dstf[i * 4 + 2] = g;
                dstf[i * 4 + 3] = 1.0;
            }
        }
        _ => {}
    }
}

/// RGB → Gray using BT.709 luma coefficients (u8 only).
pub(super) fn rgb_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        // BT.709: 0.2126R + 0.7152G + 0.0722B
        // Fixed-point: (54R + 183G + 19B + 128) >> 8
        dst[i] = ((54 * r + 183 * g + 19 * b + 128) >> 8) as u8;
    }
}

/// RGBA → Gray using BT.709 luma, drop alpha (u8 only).
pub(super) fn rgba_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let r = src[i * 4] as u32;
        let g = src[i * 4 + 1] as u32;
        let b = src[i * 4 + 2] as u32;
        dst[i] = ((54 * r + 183 * g + 19 * b + 128) >> 8) as u8;
    }
}

/// GrayAlpha → RGBA (replicate gray, preserve alpha).
pub(super) fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i * 2];
                let a = src[i * 2 + 1];
                dst[i * 4] = g;
                dst[i * 4 + 1] = g;
                dst[i * 4 + 2] = g;
                dst[i * 4 + 3] = a;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                let g = src16[i * 2];
                let a = src16[i * 2 + 1];
                dst16[i * 4] = g;
                dst16[i * 4 + 1] = g;
                dst16[i * 4 + 2] = g;
                dst16[i * 4 + 3] = a;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                let g = srcf[i * 2];
                let a = srcf[i * 2 + 1];
                dstf[i * 4] = g;
                dstf[i * 4 + 1] = g;
                dstf[i * 4 + 2] = g;
                dstf[i * 4 + 3] = a;
            }
        }
        _ => {}
    }
}

/// GrayAlpha → RGB (replicate gray, drop alpha).
pub(super) fn gray_alpha_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i * 2];
                dst[i * 3] = g;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = g;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let g = src16[i * 2];
                dst16[i * 3] = g;
                dst16[i * 3 + 1] = g;
                dst16[i * 3 + 2] = g;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                let g = srcf[i * 2];
                dstf[i * 3] = g;
                dstf[i * 3 + 1] = g;
                dstf[i * 3 + 2] = g;
            }
        }
        _ => {}
    }
}

/// Gray → GrayAlpha (add opaque alpha).
pub(super) fn gray_to_gray_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 2] = src[i];
                dst[i * 2 + 1] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 4]);
            for i in 0..width {
                dst16[i * 2] = src16[i];
                dst16[i * 2 + 1] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                dstf[i * 2] = srcf[i];
                dstf[i * 2 + 1] = 1.0;
            }
        }
        _ => {}
    }
}

/// GrayAlpha → Gray (drop alpha).
pub(super) fn gray_alpha_to_gray(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i] = src[i * 2];
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 2]);
            for i in 0..width {
                dst16[i] = src16[i * 2];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 4]);
            for i in 0..width {
                dstf[i] = srcf[i * 2];
            }
        }
        _ => {}
    }
}
