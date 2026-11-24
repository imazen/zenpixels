//! 64-pixel stack-allocated SoA L1-strip execution engine + byte-only & direct row fast paths.
//!
//! - Byte-only & wide integer/float layout changes run directly on rows with
//!   zero floating-point conversion (`ByteFastPath`).
//! - Single-stage contiguous depth & transfer routes run directly on rows
//!   without AoS↔SoA bouncing (`FastRow`).
//! - All arbitrary multi-stage pipelines run on a fixed 1 KiB stack-allocated
//!   [`StripBuf`] (`[f32; 64]` × 4 channels) with **zero heap scratch
//!   allocation** during row conversion.

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use alloc::sync::Arc;
#[cfg(feature = "perceptual")]
use zenpixels::PerceptualSpace;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, HlgMode, ImageSpec, Primaries, SampleLayout, SampleType,
    SignalRange, Transfer,
};

use crate::accel::Accelerators;
#[cfg(feature = "hdr-transfers")]
use crate::hlg_render::PreparedHlg;
use crate::policy::{DepthPolicy, GamutPolicy, LumaCoefficients};
#[cfg(feature = "oklab")]
use crate::tonemap::SoftCompress;
use crate::tonemap::StripMapper;

/// Number of pixels processed per stack-resident SoA strip (`64 × 4 × 4 B = 1 KiB`).
pub(crate) const STRIP_LEN: usize = 64;

/// 64-pixel stack-allocated Structure-of-Arrays working tile (1 KiB, 32-byte aligned).
#[repr(C, align(32))]
pub(crate) struct StripBuf {
    pub r: [f32; STRIP_LEN],
    pub g: [f32; STRIP_LEN],
    pub b: [f32; STRIP_LEN],
    pub a: [f32; STRIP_LEN],
}

impl StripBuf {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            r: [0.0; STRIP_LEN],
            g: [0.0; STRIP_LEN],
            b: [0.0; STRIP_LEN],
            a: [1.0; STRIP_LEN],
        }
    }
}

/// 4×4 Bayer ordered dithering threshold matrix normalized to `[-0.5, +0.5)`.
const BAYER_4X4: [[f32; 4]; 4] = [
    [-0.5, 0.0, -0.375, 0.125],
    [0.25, -0.25, 0.375, -0.125],
    [-0.3125, 0.1875, -0.4375, 0.0625],
    [0.4375, -0.0625, 0.3125, -0.1875],
];

/// Precomputed exact fixed-point reciprocal table `ceil((255 << 16) / a)` for
/// `a in 1..=255` (`0` at index `0`) for `RGBA8`/`BGRA8` premul → straight.
const UNPREMUL_U8_SCALE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut a = 1u32;
    while a < 256 {
        t[a as usize] = (255u32 << 16).div_ceil(a);
        a += 1;
    }
    t
};

/// Direct zero-float byte/integer/wide-layout fast path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteFastPath {
    /// Identical layout and color space: direct `copy_from_slice`.
    Memcpy,
    /// `RGB8 → RGBA8` (`A = 255`).
    Rgb8ToRgba8,
    /// `RGB8 → BGRA8` (`A = 255`).
    Rgb8ToBgra8,
    /// `RGBA8 → RGB8` (drop alpha).
    Rgba8ToRgb8,
    /// `BGRA8 → RGB8` (swizzle + drop alpha).
    Bgra8ToRgb8,
    /// `RGB8 ↔ BGR8` (3-byte channel swap).
    SwapRb3U8,
    /// `RGBA8 ↔ BGRA8` / `RGBX8 ↔ BGRX8` (4-byte R/B swap).
    SwapRb4U8,
    /// `RGBX8 → RGBA8` / `BGRX8 → BGRA8` (set 4th byte to 255).
    FillAlpha4U8,
    /// `RGBX8 → BGRA8` / `BGRX8 → RGBA8` (swap R/B + set 4th byte to 255).
    SwapRbFillAlpha4U8,
    /// `GRAY8 → RGB8` (broadcast Y → Y,Y,Y).
    Gray8ToRgb8,
    /// `GRAY8 → RGBA8` (broadcast Y → Y,Y,Y,255).
    Gray8ToRgba8,
    /// `U8 → U16` lossless exact bit-replication (`v * 257`) with identical channel order.
    U8ToU16Exact,
    /// `U16 → U8` exact integer round (`((v * 255 + 32895) >> 16) as u8`).
    U16ToU8Round,
    /// `U16 → U8` exact integer truncate (`((v * 255 + 255) >> 16) as u8`).
    U16ToU8Trunc,
    /// `RGBA8` / `BGRA8` straight → premultiplied (exact integer `round` or `trunc`).
    Rgba8StraightToPremul { round: bool },
    /// `RGBA8` / `BGRA8` premultiplied → straight (exact 256-entry fixed-point LUT).
    Rgba8PremulToStraight,
    /// 2-byte/channel (`U16` / `F16`): `4ch → 3ch` drop alpha (`RGBA16→RGB16`, `RGBAF16→RGBF16`).
    DropAlpha4To3B2,
    /// 2-byte/channel (`U16` / `F16`): `4ch → 3ch` drop alpha + swap R/B (`BGRA16→RGB16`).
    SwapDropAlpha4To3B2,
    /// 2-byte/channel (`U16` / `F16`): `3ch → 4ch` append opaque alpha (`opaque_bits`: `0xFFFF` for U16, `0x3C00` for F16).
    AddAlpha3To4B2(u16),
    /// 2-byte/channel (`U16` / `F16`): `3ch → 4ch` swap R/B + append opaque alpha.
    SwapAddAlpha3To4B2(u16),
    /// 2-byte/channel (`U16` / `F16`): `3ch ↔ 3ch` swap R/B.
    SwapRb3B2,
    /// 2-byte/channel (`U16` / `F16`): `4ch ↔ 4ch` swap R/B.
    SwapRb4B2,
    /// 2-byte/channel (`U16` / `F16`): `Gray → RGB`.
    GrayToRgbB2,
    /// 2-byte/channel (`U16` / `F16`): `Gray → RGBA` (`opaque_bits`).
    GrayToRgbaB2(u16),
    /// 4-byte/channel (`F32`): `4ch → 3ch` drop alpha (`RGBAF32→RGBF32`).
    DropAlpha4To3B4,
    /// 4-byte/channel (`F32`): `4ch → 3ch` drop alpha + swap R/B (`BGRAF32→RGBF32`).
    SwapDropAlpha4To3B4,
    /// 4-byte/channel (`F32`): `3ch → 4ch` append opaque `1.0f32`.
    AddAlpha3To4B4,
    /// 4-byte/channel (`F32`): `3ch → 4ch` swap R/B + append opaque `1.0f32`.
    SwapAddAlpha3To4B4,
    /// 4-byte/channel (`F32`): `3ch ↔ 3ch` swap R/B.
    SwapRb3B4,
    /// 4-byte/channel (`F32`): `4ch ↔ 4ch` swap R/B.
    SwapRb4B4,
    /// 4-byte/channel (`F32`): `Gray → RGB`.
    GrayToRgbB4,
    /// 4-byte/channel (`F32`): `Gray → RGBA` (`1.0f32`).
    GrayToRgbaB4,
}

/// Dead-strippable `garb` SIMD dispatcher referenced ONLY by `Accelerators::with_byte_layout()`.
pub(crate) fn exec_garb_byte_fast_path(
    path: ByteFastPath,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
) -> bool {
    match path {
        ByteFastPath::Rgb8ToRgba8 => {
            garb::bytes::rgb_to_rgba(&src[..width * 3], &mut dst[..width * 4]).is_ok()
        }
        ByteFastPath::Rgb8ToBgra8 => {
            garb::bytes::rgb_to_bgra(&src[..width * 3], &mut dst[..width * 4]).is_ok()
        }
        ByteFastPath::Rgba8ToRgb8 => {
            garb::bytes::rgba_to_rgb(&src[..width * 4], &mut dst[..width * 3]).is_ok()
        }
        ByteFastPath::Bgra8ToRgb8 => {
            garb::bytes::bgra_to_rgb(&src[..width * 4], &mut dst[..width * 3]).is_ok()
        }
        ByteFastPath::SwapRb3U8 => {
            garb::bytes::rgb_to_bgr(&src[..width * 3], &mut dst[..width * 3]).is_ok()
        }
        ByteFastPath::SwapRb4U8 => {
            garb::bytes::rgba_to_bgra(&src[..width * 4], &mut dst[..width * 4]).is_ok()
        }
        ByteFastPath::FillAlpha4U8 => {
            let s = &src[..width * 4];
            let d = &mut dst[..width * 4];
            d.copy_from_slice(s);
            garb::bytes::fill_alpha_rgba(d).is_ok()
        }
        ByteFastPath::Gray8ToRgba8 => {
            garb::bytes::gray_to_rgba(&src[..width], &mut dst[..width * 4]).is_ok()
        }
        _ => false,
    }
}

impl ByteFastPath {
    /// Execute this byte-only fast path on one row (`width` pixels).
    pub(crate) fn exec(self, src: &[u8], dst: &mut [u8], width: usize, accel: &Accelerators) {
        if let Some(garb_fn) = accel.byte_layout_exec
            && garb_fn(self, src, dst, width)
        {
            return;
        }
        match self {
            Self::Memcpy => {
                let len = src.len().min(dst.len());
                dst[..len].copy_from_slice(&src[..len]);
            }
            Self::Rgb8ToRgba8 => {
                let s = &src[..width * 3];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(3).zip(d.chunks_exact_mut(4)) {
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                    dp[2] = sp[2];
                    dp[3] = 255;
                }
            }
            Self::Rgb8ToBgra8 => {
                let s = &src[..width * 3];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(3).zip(d.chunks_exact_mut(4)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                    dp[3] = 255;
                }
            }
            Self::Rgba8ToRgb8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 3];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                    dp[2] = sp[2];
                }
            }
            Self::Bgra8ToRgb8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 3];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                }
            }
            Self::SwapRb3U8 => {
                let s = &src[..width * 3];
                let d = &mut dst[..width * 3];
                for (sp, dp) in s.chunks_exact(3).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                }
            }
            Self::SwapRb4U8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(4)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                    dp[3] = sp[3];
                }
            }
            Self::FillAlpha4U8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(4)) {
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                    dp[2] = sp[2];
                    dp[3] = 255;
                }
            }
            Self::SwapRbFillAlpha4U8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(4)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                    dp[3] = 255;
                }
            }
            Self::Gray8ToRgb8 => {
                let s = &src[..width];
                let d = &mut dst[..width * 3];
                for (&y, dp) in s.iter().zip(d.chunks_exact_mut(3)) {
                    dp[0] = y;
                    dp[1] = y;
                    dp[2] = y;
                }
            }
            Self::Gray8ToRgba8 => {
                let s = &src[..width];
                let d = &mut dst[..width * 4];
                for (&y, dp) in s.iter().zip(d.chunks_exact_mut(4)) {
                    dp[0] = y;
                    dp[1] = y;
                    dp[2] = y;
                    dp[3] = 255;
                }
            }
            Self::U8ToU16Exact => {
                let n = src.len().min(dst.len() / 2);
                if let Ok(d16) = bytemuck::try_cast_slice_mut::<u8, u16>(&mut dst[..n * 2]) {
                    for (&b, out) in src[..n].iter().zip(d16.iter_mut()) {
                        *out = u16::from(b) * 257;
                    }
                } else {
                    for (&b, dp) in src[..n].iter().zip(dst[..n * 2].chunks_exact_mut(2)) {
                        let v = u16::from(b) * 257;
                        dp.copy_from_slice(&v.to_ne_bytes());
                    }
                }
            }
            Self::U16ToU8Round => {
                let n = (src.len() / 2).min(dst.len());
                if let Ok(s16) = bytemuck::try_cast_slice::<u8, u16>(&src[..n * 2]) {
                    for (&v, out) in s16.iter().zip(dst[..n].iter_mut()) {
                        *out = ((u32::from(v) * 255 + 32895) >> 16) as u8;
                    }
                } else {
                    for (sp, out) in src[..n * 2].chunks_exact(2).zip(dst[..n].iter_mut()) {
                        let v = u16::from_ne_bytes([sp[0], sp[1]]);
                        *out = ((u32::from(v) * 255 + 32895) >> 16) as u8;
                    }
                }
            }
            Self::U16ToU8Trunc => {
                let n = (src.len() / 2).min(dst.len());
                if let Ok(s16) = bytemuck::try_cast_slice::<u8, u16>(&src[..n * 2]) {
                    for (&v, out) in s16.iter().zip(dst[..n].iter_mut()) {
                        *out = ((u32::from(v) * 255 + 255) >> 16) as u8;
                    }
                } else {
                    for (sp, out) in src[..n * 2].chunks_exact(2).zip(dst[..n].iter_mut()) {
                        let v = u16::from_ne_bytes([sp[0], sp[1]]);
                        *out = ((u32::from(v) * 255 + 255) >> 16) as u8;
                    }
                }
            }
            Self::Rgba8StraightToPremul { round } => {
                if let Some(f) = accel.rgba8_straight_to_premul {
                    f(src, dst, width, round);
                    return;
                }
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                let bias: u32 = if round { 128 } else { 1 };
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(4)) {
                    let a = u32::from(sp[3]);
                    dp[0] = (((u32::from(sp[0]) * a + bias) * 257) >> 16) as u8;
                    dp[1] = (((u32::from(sp[1]) * a + bias) * 257) >> 16) as u8;
                    dp[2] = (((u32::from(sp[2]) * a + bias) * 257) >> 16) as u8;
                    dp[3] = sp[3];
                }
            }
            Self::Rgba8PremulToStraight => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(4)) {
                    let a = sp[3];
                    if a == 0 {
                        dp[0] = 0;
                        dp[1] = 0;
                        dp[2] = 0;
                        dp[3] = 0;
                    } else if a == 255 {
                        dp[0] = sp[0];
                        dp[1] = sp[1];
                        dp[2] = sp[2];
                        dp[3] = 255;
                    } else {
                        let scale = UNPREMUL_U8_SCALE[usize::from(a)];
                        dp[0] = if sp[0] >= a {
                            255
                        } else {
                            ((u32::from(sp[0]) * scale + 32768) >> 16) as u8
                        };
                        dp[1] = if sp[1] >= a {
                            255
                        } else {
                            ((u32::from(sp[1]) * scale + 32768) >> 16) as u8
                        };
                        dp[2] = if sp[2] >= a {
                            255
                        } else {
                            ((u32::from(sp[2]) * scale + 32768) >> 16) as u8
                        };
                        dp[3] = a;
                    }
                }
            }
            Self::DropAlpha4To3B2 => {
                let s = &src[..width * 8];
                let d = &mut dst[..width * 6];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 3]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[0], sp[1], sp[2]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(8).zip(d.chunks_exact_mut(6)) {
                        dp.copy_from_slice(&sp[..6]);
                    }
                }
            }
            Self::SwapDropAlpha4To3B2 => {
                let s = &src[..width * 8];
                let d = &mut dst[..width * 6];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 3]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(8).zip(d.chunks_exact_mut(6)) {
                        dp[0..2].copy_from_slice(&sp[4..6]);
                        dp[2..4].copy_from_slice(&sp[2..4]);
                        dp[4..6].copy_from_slice(&sp[0..2]);
                    }
                }
            }
            Self::AddAlpha3To4B2(alpha) => {
                let s = &src[..width * 6];
                let d = &mut dst[..width * 8];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 4]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[0], sp[1], sp[2], alpha];
                    }
                } else {
                    let ab = alpha.to_ne_bytes();
                    for (sp, dp) in s.chunks_exact(6).zip(d.chunks_exact_mut(8)) {
                        dp[0..6].copy_from_slice(sp);
                        dp[6..8].copy_from_slice(&ab);
                    }
                }
            }
            Self::SwapAddAlpha3To4B2(alpha) => {
                let s = &src[..width * 6];
                let d = &mut dst[..width * 8];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 4]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0], alpha];
                    }
                } else {
                    let ab = alpha.to_ne_bytes();
                    for (sp, dp) in s.chunks_exact(6).zip(d.chunks_exact_mut(8)) {
                        dp[0..2].copy_from_slice(&sp[4..6]);
                        dp[2..4].copy_from_slice(&sp[2..4]);
                        dp[4..6].copy_from_slice(&sp[0..2]);
                        dp[6..8].copy_from_slice(&ab);
                    }
                }
            }
            Self::SwapRb3B2 => {
                let s = &src[..width * 6];
                let d = &mut dst[..width * 6];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 3]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(6).zip(d.chunks_exact_mut(6)) {
                        dp[0..2].copy_from_slice(&sp[4..6]);
                        dp[2..4].copy_from_slice(&sp[2..4]);
                        dp[4..6].copy_from_slice(&sp[0..2]);
                    }
                }
            }
            Self::SwapRb4B2 => {
                let s = &src[..width * 8];
                let d = &mut dst[..width * 8];
                if let (Ok(s16), Ok(d16)) = (
                    bytemuck::try_cast_slice::<u8, [u16; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u16; 4]>(d),
                ) {
                    for (sp, dp) in s16.iter().zip(d16.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0], sp[3]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(8).zip(d.chunks_exact_mut(8)) {
                        dp[0..2].copy_from_slice(&sp[4..6]);
                        dp[2..4].copy_from_slice(&sp[2..4]);
                        dp[4..6].copy_from_slice(&sp[0..2]);
                        dp[6..8].copy_from_slice(&sp[6..8]);
                    }
                }
            }
            Self::GrayToRgbB2 => {
                let s = &src[..width * 2];
                let d = &mut dst[..width * 6];
                for (sp, dp) in s.chunks_exact(2).zip(d.chunks_exact_mut(6)) {
                    dp[0..2].copy_from_slice(sp);
                    dp[2..4].copy_from_slice(sp);
                    dp[4..6].copy_from_slice(sp);
                }
            }
            Self::GrayToRgbaB2(alpha) => {
                let s = &src[..width * 2];
                let d = &mut dst[..width * 8];
                let ab = alpha.to_ne_bytes();
                for (sp, dp) in s.chunks_exact(2).zip(d.chunks_exact_mut(8)) {
                    dp[0..2].copy_from_slice(sp);
                    dp[2..4].copy_from_slice(sp);
                    dp[4..6].copy_from_slice(sp);
                    dp[6..8].copy_from_slice(&ab);
                }
            }
            Self::DropAlpha4To3B4 => {
                let s = &src[..width * 16];
                let d = &mut dst[..width * 12];
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 3]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[0], sp[1], sp[2]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(16).zip(d.chunks_exact_mut(12)) {
                        dp.copy_from_slice(&sp[..12]);
                    }
                }
            }
            Self::SwapDropAlpha4To3B4 => {
                let s = &src[..width * 16];
                let d = &mut dst[..width * 12];
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 3]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(16).zip(d.chunks_exact_mut(12)) {
                        dp[0..4].copy_from_slice(&sp[8..12]);
                        dp[4..8].copy_from_slice(&sp[4..8]);
                        dp[8..12].copy_from_slice(&sp[0..4]);
                    }
                }
            }
            Self::AddAlpha3To4B4 => {
                let s = &src[..width * 12];
                let d = &mut dst[..width * 16];
                let one = 1.0_f32.to_bits();
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 4]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[0], sp[1], sp[2], one];
                    }
                } else {
                    let ob = one.to_ne_bytes();
                    for (sp, dp) in s.chunks_exact(12).zip(d.chunks_exact_mut(16)) {
                        dp[0..12].copy_from_slice(sp);
                        dp[12..16].copy_from_slice(&ob);
                    }
                }
            }
            Self::SwapAddAlpha3To4B4 => {
                let s = &src[..width * 12];
                let d = &mut dst[..width * 16];
                let one = 1.0_f32.to_bits();
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 4]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0], one];
                    }
                } else {
                    let ob = one.to_ne_bytes();
                    for (sp, dp) in s.chunks_exact(12).zip(d.chunks_exact_mut(16)) {
                        dp[0..4].copy_from_slice(&sp[8..12]);
                        dp[4..8].copy_from_slice(&sp[4..8]);
                        dp[8..12].copy_from_slice(&sp[0..4]);
                        dp[12..16].copy_from_slice(&ob);
                    }
                }
            }
            Self::SwapRb3B4 => {
                let s = &src[..width * 12];
                let d = &mut dst[..width * 12];
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 3]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 3]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(12).zip(d.chunks_exact_mut(12)) {
                        dp[0..4].copy_from_slice(&sp[8..12]);
                        dp[4..8].copy_from_slice(&sp[4..8]);
                        dp[8..12].copy_from_slice(&sp[0..4]);
                    }
                }
            }
            Self::SwapRb4B4 => {
                let s = &src[..width * 16];
                let d = &mut dst[..width * 16];
                if let (Ok(sf), Ok(df)) = (
                    bytemuck::try_cast_slice::<u8, [u32; 4]>(s),
                    bytemuck::try_cast_slice_mut::<u8, [u32; 4]>(d),
                ) {
                    for (sp, dp) in sf.iter().zip(df.iter_mut()) {
                        *dp = [sp[2], sp[1], sp[0], sp[3]];
                    }
                } else {
                    for (sp, dp) in s.chunks_exact(16).zip(d.chunks_exact_mut(16)) {
                        dp[0..4].copy_from_slice(&sp[8..12]);
                        dp[4..8].copy_from_slice(&sp[4..8]);
                        dp[8..12].copy_from_slice(&sp[0..4]);
                        dp[12..16].copy_from_slice(&sp[12..16]);
                    }
                }
            }
            Self::GrayToRgbB4 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 12];
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(12)) {
                    dp[0..4].copy_from_slice(sp);
                    dp[4..8].copy_from_slice(sp);
                    dp[8..12].copy_from_slice(sp);
                }
            }
            Self::GrayToRgbaB4 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 16];
                let ob = 1.0_f32.to_ne_bytes();
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(16)) {
                    dp[0..4].copy_from_slice(sp);
                    dp[4..8].copy_from_slice(sp);
                    dp[8..12].copy_from_slice(sp);
                    dp[12..16].copy_from_slice(&ob);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Direct contiguous single-stage row fast paths (zero AoS↔SoA overhead!)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) enum FastRow {
    F16ToF32 {
        channels: usize,
    },
    F32ToF16 {
        channels: usize,
    },
    U8ToF32 {
        channels: usize,
    },
    F32ToU8 {
        channels: usize,
        round: bool,
    },
    U16ToF32 {
        channels: usize,
    },
    F32ToU16 {
        channels: usize,
        round: bool,
    },
    SrgbU8ToLinF32 {
        channels: usize,
        is_rgba: bool,
    },
    LinF32ToSrgbU8 {
        channels: usize,
        is_rgba: bool,
        round: bool,
    },
    TransferF32 {
        channels: usize,
        transfer: Transfer,
        is_eotf: bool,
        is_rgba: bool,
        pq_scale: f32,
        preserve_extended: bool,
    },
    /// Option A: Fused exact SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8 (0-bit parity).
    Srgb8MatrixExact {
        matrix: [[f32; 3]; 3],
        src_order: ChannelOrder,
        dst_order: ChannelOrder,
        fill_opaque: bool,
        round: bool,
    },
    /// Option B: Opt-in fast Q14 fixed-point SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8 (max error <= 1 LSB).
    Srgb8MatrixApproxQ14 {
        tables: Arc<crate::strip_simd::Q14MatrixTables>,
        src_order: ChannelOrder,
        dst_order: ChannelOrder,
        fill_opaque: bool,
        round: bool,
    },
}

/// Detect a direct contiguous single-stage row fast path (or fused U8 sRGB-TRC 3×3 matrix path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_fast_row(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy_depth: DepthPolicy,
    gamut_policy: GamutPolicy,
    allow_approximate: bool,
    matte_active: bool,
    preserve_extended: bool,
    white_nits: f32,
) -> Option<FastRow> {
    if matte_active || !src.layout.packing.is_full() || !dst.layout.packing.is_full() {
        return None;
    }
    let round = match policy_depth {
        DepthPolicy::Round | DepthPolicy::Forbid => true,
        DepthPolicy::Truncate => false,
        DepthPolicy::OrderedDither => return None,
    };

    // Fused U8 sRGB-TRC -> 3x3 matrix -> sRGB-TRC U8 (Display P3 <-> sRGB, BT.2020 <-> sRGB, etc.)
    // Supports arbitrary U8 RGB/BGR/RGBA/BGRA/RGBX/BGRX combinations in a single pass!
    if src.layout.sample == SampleType::U8
        && dst.layout.sample == SampleType::U8
        && src.layout.order.is_rgb_family()
        && dst.layout.order.is_rgb_family()
        && gamut_policy == GamutPolicy::Clip
        && let (
            ColorSpace::Rgb {
                primaries: sp,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Rgb {
                primaries: dp,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: da,
                white: dw,
            },
        ) = (&src.color, &dst.color)
        && sp != dp
        && sw == dw
        && *sa != AlphaMode::Premultiplied
        && *da != AlphaMode::Premultiplied
        && let Some(matrix) = sp.gamut_matrix_to(*dp)
    {
        let fill_opaque = !src.layout.order.has_alpha_slot()
            || !dst.layout.order.has_alpha_slot()
            || *sa == AlphaMode::Opaque
            || *da == AlphaMode::Opaque;
        let max_row_l1 = matrix
            .iter()
            .map(|r| r[0].abs() + r[1].abs() + r[2].abs())
            .fold(0.0_f32, f32::max);
        if allow_approximate && max_row_l1 <= 6.0 {
            return Some(FastRow::Srgb8MatrixApproxQ14 {
                tables: crate::strip_simd::Q14MatrixTables::new(&matrix, round),
                src_order: src.layout.order,
                dst_order: dst.layout.order,
                fill_opaque,
                round,
            });
        }
        return Some(FastRow::Srgb8MatrixExact {
            matrix,
            src_order: src.layout.order,
            dst_order: dst.layout.order,
            fill_opaque,
            round,
        });
    }

    if src.layout.order != dst.layout.order {
        return None;
    }
    let ch = src.layout.channels();
    let is_rgba = matches!(src.layout.order, ChannelOrder::Rgba | ChannelOrder::Bgra);
    let is_plain_color = matches!(
        src.layout.order,
        ChannelOrder::Rgb
            | ChannelOrder::Bgr
            | ChannelOrder::Gray
            | ChannelOrder::Rgba
            | ChannelOrder::Bgra
    );
    if !is_plain_color {
        return None;
    }

    // Extract matching primaries / range / alpha / white.
    let (st, dt) = match (&src.color, &dst.color) {
        (
            ColorSpace::Rgb {
                primaries: sp,
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Rgb {
                primaries: dp,
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) => {
            if sp != dp
                || sr != dr
                || *sr != SignalRange::Full
                || sa != da
                || (*sa == AlphaMode::Opaque && is_rgba)
                || (*sa == AlphaMode::Premultiplied && st != dt)
            {
                return None;
            }
            if *st == *dt && sw != dw {
                return None;
            }
            (*st, *dt)
        }
        (
            ColorSpace::Gray {
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Gray {
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) => {
            if sr != dr
                || *sr != SignalRange::Full
                || sa != da
                || (*sa == AlphaMode::Opaque && src.layout.order.has_alpha_slot())
                || (*sa == AlphaMode::Premultiplied && st != dt)
            {
                return None;
            }
            if *st == *dt && sw != dw {
                return None;
            }
            (*st, *dt)
        }
        _ => return None,
    };

    if st != dt && (st == Transfer::Unknown || dt == Transfer::Unknown) {
        return None;
    }

    use SampleType as S;
    if st == dt {
        return match (src.layout.sample, dst.layout.sample) {
            (S::F16, S::F32) => Some(FastRow::F16ToF32 { channels: ch }),
            (S::F32, S::F16) => Some(FastRow::F32ToF16 { channels: ch }),
            (S::U8, S::F32) => Some(FastRow::U8ToF32 { channels: ch }),
            (S::F32, S::U8) => Some(FastRow::F32ToU8 {
                channels: ch,
                round,
            }),
            (S::U16, S::F32) => Some(FastRow::U16ToF32 { channels: ch }),
            (S::F32, S::U16) => Some(FastRow::F32ToU16 {
                channels: ch,
                round,
            }),
            _ => None,
        };
    }

    // sRGB U8 <-> Linear F32 direct fast paths.
    if src.layout.sample == S::U8
        && dst.layout.sample == S::F32
        && st == Transfer::Srgb
        && dt == Transfer::Linear
    {
        return Some(FastRow::SrgbU8ToLinF32 {
            channels: ch,
            is_rgba,
        });
    }
    if src.layout.sample == S::F32
        && dst.layout.sample == S::U8
        && st == Transfer::Linear
        && dt == Transfer::Srgb
    {
        return Some(FastRow::LinF32ToSrgbU8 {
            channels: ch,
            is_rgba,
            round,
        });
    }

    // F32 <-> F32 single transfer (Linear <-> Srgb / Bt709 / Gamma22 / Pq).
    // HLG <-> Linear always resolves through `resolve_hlg` into the BT.2100-3 display strip path.
    if src.layout.sample == S::F32 && dst.layout.sample == S::F32 {
        let supported = |t: Transfer| {
            matches!(
                t,
                Transfer::Srgb
                    | Transfer::Bt709
                    | Transfer::Gamma22
                    | Transfer::Gamma(_)
                    | Transfer::Pq
            )
        };
        if dt == Transfer::Linear && supported(st) {
            let pq_scale = if st == Transfer::Pq {
                10_000.0 / white_nits
            } else {
                1.0
            };
            return Some(FastRow::TransferF32 {
                channels: ch,
                transfer: st,
                is_eotf: true,
                is_rgba,
                pq_scale,
                preserve_extended,
            });
        }
        if st == Transfer::Linear && supported(dt) {
            let pq_scale = if dt == Transfer::Pq {
                white_nits / 10_000.0
            } else {
                1.0
            };
            return Some(FastRow::TransferF32 {
                channels: ch,
                transfer: dt,
                is_eotf: false,
                is_rgba,
                pq_scale,
                preserve_extended,
            });
        }
    }
    None
}

/// Execute a direct contiguous row fast path with zero heap allocation.
#[inline(never)]
pub(crate) fn exec_fast_row(
    fast: &FastRow,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    accel: &Accelerators,
) {
    match fast {
        &FastRow::F16ToF32 { channels } => {
            let count = width * channels;
            if let (Ok(s16), Ok(df)) = (
                bytemuck::try_cast_slice::<u8, u16>(&src[..count * 2]),
                bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst[..count * 4]),
            ) {
                if let Some(f) = accel.f16_decode {
                    f(s16, df);
                } else {
                    for (&b, o) in s16.iter().zip(df.iter_mut()) {
                        *o = crate::f16_scalar::f16_bits_to_f32(b);
                    }
                }
            } else {
                let mut sb = [0u16; 256];
                let mut db = [0.0f32; 256];
                let mut i = 0;
                while i < count {
                    let n = (count - i).min(256);
                    for k in 0..n {
                        let o = (i + k) * 2;
                        sb[k] = u16::from_ne_bytes([src[o], src[o + 1]]);
                    }
                    if let Some(f) = accel.f16_decode {
                        f(&sb[..n], &mut db[..n]);
                    } else {
                        for k in 0..n {
                            db[k] = crate::f16_scalar::f16_bits_to_f32(sb[k]);
                        }
                    }
                    for k in 0..n {
                        let o = (i + k) * 4;
                        dst[o..o + 4].copy_from_slice(&db[k].to_ne_bytes());
                    }
                    i += n;
                }
            }
        }
        &FastRow::F32ToF16 { channels } => {
            let count = width * channels;
            if let (Ok(sf), Ok(d16)) = (
                bytemuck::try_cast_slice::<u8, f32>(&src[..count * 4]),
                bytemuck::try_cast_slice_mut::<u8, u16>(&mut dst[..count * 2]),
            ) {
                if let Some(f) = accel.f16_encode {
                    f(sf, d16);
                } else {
                    for (&v, o) in sf.iter().zip(d16.iter_mut()) {
                        *o = crate::f16_scalar::f32_to_f16_bits(v);
                    }
                }
            } else {
                let mut sb = [0.0f32; 256];
                let mut db = [0u16; 256];
                let mut i = 0;
                while i < count {
                    let n = (count - i).min(256);
                    for k in 0..n {
                        let o = (i + k) * 4;
                        sb[k] = f32::from_ne_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
                    }
                    if let Some(f) = accel.f16_encode {
                        f(&sb[..n], &mut db[..n]);
                    } else {
                        for k in 0..n {
                            db[k] = crate::f16_scalar::f32_to_f16_bits(sb[k]);
                        }
                    }
                    for k in 0..n {
                        let o = (i + k) * 2;
                        dst[o..o + 2].copy_from_slice(&db[k].to_ne_bytes());
                    }
                    i += n;
                }
            }
        }
        &FastRow::U8ToF32 { channels } => {
            let count = width * channels;
            if let Ok(df) = bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst[..count * 4]) {
                if let Some(f) = accel.u8_to_f32 {
                    f(&src[..count], df);
                } else {
                    const INV: f32 = 1.0 / 255.0;
                    for (&b, o) in src[..count].iter().zip(df.iter_mut()) {
                        *o = f32::from(b) * INV;
                    }
                }
            } else {
                const INV: f32 = 1.0 / 255.0;
                for (i, &b) in src[..count].iter().enumerate() {
                    dst[i * 4..i * 4 + 4].copy_from_slice(&(f32::from(b) * INV).to_ne_bytes());
                }
            }
        }
        &FastRow::F32ToU8 { channels, round } => {
            let count = width * channels;
            let bias = if round { 0.5_f32 } else { 0.0_f32 };
            if let Ok(sf) = bytemuck::try_cast_slice::<u8, f32>(&src[..count * 4]) {
                if let Some(f) = accel.f32_to_u8 {
                    f(sf, &mut dst[..count], round);
                } else {
                    for (&v, o) in sf.iter().zip(dst[..count].iter_mut()) {
                        *o = if v.is_nan() || v <= 0.0 {
                            0
                        } else {
                            ((v.min(1.0) * 255.0 + bias) as u32).min(255) as u8
                        };
                    }
                }
            } else {
                for (i, out) in dst[..count].iter_mut().enumerate() {
                    let o = i * 4;
                    let v = f32::from_ne_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
                    *out = if v.is_nan() || v <= 0.0 {
                        0
                    } else {
                        ((v.min(1.0) * 255.0 + bias) as u32).min(255) as u8
                    };
                }
            }
        }
        &FastRow::U16ToF32 { channels } => {
            let count = width * channels;
            if let (Ok(s16), Ok(df)) = (
                bytemuck::try_cast_slice::<u8, u16>(&src[..count * 2]),
                bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst[..count * 4]),
            ) {
                if let Some(f) = accel.u16_to_f32 {
                    f(s16, df);
                } else {
                    const INV: f32 = 1.0 / 65535.0;
                    for (&w, o) in s16.iter().zip(df.iter_mut()) {
                        *o = f32::from(w) * INV;
                    }
                }
            } else {
                const INV: f32 = 1.0 / 65535.0;
                for i in 0..count {
                    let w = u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]);
                    dst[i * 4..i * 4 + 4].copy_from_slice(&(f32::from(w) * INV).to_ne_bytes());
                }
            }
        }
        &FastRow::F32ToU16 { channels, round } => {
            let count = width * channels;
            let bias = if round { 0.5_f32 } else { 0.0_f32 };
            if let (Ok(sf), Ok(d16)) = (
                bytemuck::try_cast_slice::<u8, f32>(&src[..count * 4]),
                bytemuck::try_cast_slice_mut::<u8, u16>(&mut dst[..count * 2]),
            ) {
                if let Some(f) = accel.f32_to_u16 {
                    f(sf, d16, round);
                } else {
                    for (&v, o) in sf.iter().zip(d16.iter_mut()) {
                        *o = if v.is_nan() || v <= 0.0 {
                            0
                        } else {
                            ((v.min(1.0) * 65535.0 + bias) as u32).min(65535) as u16
                        };
                    }
                }
            } else {
                for i in 0..count {
                    let o = i * 4;
                    let v = f32::from_ne_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
                    let q = if v.is_nan() || v <= 0.0 {
                        0u16
                    } else {
                        ((v.min(1.0) * 65535.0 + bias) as u32).min(65535) as u16
                    };
                    dst[i * 2..i * 2 + 2].copy_from_slice(&q.to_ne_bytes());
                }
            }
        }
        &FastRow::SrgbU8ToLinF32 { channels, is_rgba } => {
            const INV_255: f32 = 1.0 / 255.0;
            let count = width * channels;
            if let Ok(df) = bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst[..count * 4]) {
                if is_rgba {
                    // Canonical alpha formula `f32::from(a) * (1.0 / 255.0)` matches
                    // `unpack_strip_u8_srgb_lut` and `strip_simd` bit-for-bit across all 256 codes.
                    for (sp, dp) in src[..count].chunks_exact(4).zip(df.chunks_exact_mut(4)) {
                        dp[0] = linear_srgb::default::srgb_u8_to_linear(sp[0]);
                        dp[1] = linear_srgb::default::srgb_u8_to_linear(sp[1]);
                        dp[2] = linear_srgb::default::srgb_u8_to_linear(sp[2]);
                        dp[3] = f32::from(sp[3]) * INV_255;
                    }
                } else if let Some(f) = accel.srgb_u8_to_lin {
                    f(&src[..count], df);
                } else {
                    for (&b, o) in src[..count].iter().zip(df.iter_mut()) {
                        *o = linear_srgb::default::srgb_u8_to_linear(b);
                    }
                }
            } else {
                for i in 0..width {
                    for c in 0..channels {
                        let b = src[i * channels + c];
                        let v = if is_rgba && c == 3 {
                            f32::from(b) * INV_255
                        } else {
                            linear_srgb::default::srgb_u8_to_linear(b)
                        };
                        let o = (i * channels + c) * 4;
                        dst[o..o + 4].copy_from_slice(&v.to_ne_bytes());
                    }
                }
            }
        }
        &FastRow::LinF32ToSrgbU8 {
            channels,
            is_rgba,
            round,
        } => {
            let count = width * channels;
            let aligned = bytemuck::try_cast_slice::<u8, f32>(&src[..count * 4]).ok();
            if let (Some(sf), Some(fused)) = (aligned, accel.lin_f32_to_srgb_u8) {
                fused(sf, &mut dst[..count], is_rgba, round);
                return;
            }
            let bias = if round { 0.5_f32 } else { 0.0_f32 };
            let mut buf = [0.0f32; 256];
            let mut saved_a = [0.0f32; 64];
            let mut i = 0usize;
            while i < count {
                // Keep chunks pixel-aligned (multiple of 4 for RGBA).
                let n = if is_rgba {
                    ((count - i).min(256)) & !3
                } else {
                    (count - i).min(256)
                };
                if let Some(sf) = aligned {
                    buf[..n].copy_from_slice(&sf[i..i + n]);
                } else {
                    for k in 0..n {
                        let o = (i + k) * 4;
                        buf[k] = f32::from_ne_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
                    }
                }
                if is_rgba {
                    let px = n / 4;
                    for p in 0..px {
                        saved_a[p] = buf[p * 4 + 3];
                    }
                }
                for v in &mut buf[..n] {
                    *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                }
                if let Some(f) = accel.srgb_oetf {
                    f(&mut buf[..n]);
                } else {
                    srgb_oetf_scalar_slice(&mut buf[..n]);
                }
                if is_rgba {
                    let px = n / 4;
                    for p in 0..px {
                        buf[p * 4 + 3] = saved_a[p];
                    }
                }
                if let Some(f) = accel.f32_to_u8 {
                    f(&buf[..n], &mut dst[i..i + n], round);
                } else {
                    for k in 0..n {
                        let v = buf[k];
                        dst[i + k] = if v.is_nan() || v <= 0.0 {
                            0
                        } else {
                            ((v.min(1.0) * 255.0 + bias) as u32).min(255) as u8
                        };
                    }
                }
                i += n;
            }
        }
        &FastRow::TransferF32 {
            channels,
            transfer,
            is_eotf,
            is_rgba,
            pq_scale,
            preserve_extended,
        } => {
            let count = width * channels;
            let byte_len = count * 4;
            dst[..byte_len].copy_from_slice(&src[..byte_len]);
            if let Ok(df) = bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst[..byte_len]) {
                if !is_rgba {
                    apply_transfer_slice(transfer, is_eotf, pq_scale, preserve_extended, df, accel);
                } else {
                    let mut saved_a = [0.0f32; 64];
                    let mut p = 0usize;
                    while p < width {
                        let n = (width - p).min(64);
                        let sub = &mut df[p * 4..(p + n) * 4];
                        for k in 0..n {
                            saved_a[k] = sub[k * 4 + 3];
                        }
                        apply_transfer_slice(
                            transfer,
                            is_eotf,
                            pq_scale,
                            preserve_extended,
                            sub,
                            accel,
                        );
                        for k in 0..n {
                            sub[k * 4 + 3] = saved_a[k];
                        }
                        p += n;
                    }
                }
            } else {
                let mut buf = [0.0f32; 256];
                let mut saved_a = [0.0f32; 64];
                let mut p = 0usize;
                let step = if is_rgba { 64 } else { 256 / channels };
                while p < width {
                    let n = (width - p).min(step);
                    let floats = n * channels;
                    for k in 0..floats {
                        let o = (p * channels + k) * 4;
                        buf[k] = f32::from_ne_bytes([dst[o], dst[o + 1], dst[o + 2], dst[o + 3]]);
                    }
                    if is_rgba {
                        for k in 0..n {
                            saved_a[k] = buf[k * 4 + 3];
                        }
                    }
                    apply_transfer_slice(
                        transfer,
                        is_eotf,
                        pq_scale,
                        preserve_extended,
                        &mut buf[..floats],
                        accel,
                    );
                    if is_rgba {
                        for k in 0..n {
                            buf[k * 4 + 3] = saved_a[k];
                        }
                    }
                    for k in 0..floats {
                        let o = (p * channels + k) * 4;
                        dst[o..o + 4].copy_from_slice(&buf[k].to_ne_bytes());
                    }
                    p += n;
                }
            }
        }
        FastRow::Srgb8MatrixExact {
            matrix,
            src_order,
            dst_order,
            fill_opaque,
            round,
        } => {
            if let Some(f) = accel.srgb8_matrix_exact {
                f(
                    src,
                    dst,
                    width,
                    matrix,
                    *src_order,
                    *dst_order,
                    *fill_opaque,
                    *round,
                );
            } else {
                crate::strip_simd::srgb8_matrix_exact_scalar(
                    src,
                    dst,
                    width,
                    matrix,
                    *src_order,
                    *dst_order,
                    *fill_opaque,
                    *round,
                );
            }
        }
        FastRow::Srgb8MatrixApproxQ14 {
            tables,
            src_order,
            dst_order,
            fill_opaque,
            round,
        } => {
            if let Some(f) = accel.srgb8_matrix_approx {
                f(
                    src,
                    dst,
                    width,
                    tables,
                    *src_order,
                    *dst_order,
                    *fill_opaque,
                    *round,
                );
            } else {
                crate::strip_simd::srgb8_matrix_approx_q14_scalar(
                    src,
                    dst,
                    width,
                    tables,
                    *src_order,
                    *dst_order,
                    *fill_opaque,
                    *round,
                );
            }
        }
    }
}

/// Composable micro-op executed in-place on a 64-pixel [`StripBuf`].
#[derive(Clone)]
pub(crate) enum MicroOp {
    /// Expand narrow studio integer range for effective bit depth `bits` (`8..=16`) to full `[0.0, 1.0]` on color channels (alpha untouched).
    ExpandNarrow { bits: u8 },
    /// Compress full `[0.0, 1.0]` color channels to narrow studio range for effective bit depth `bits` (`8..=16`, alpha untouched).
    CompressNarrow { bits: u8 },
    /// Scale linear RGB/Gray light by a constant factor (relative diffuse white rescaling).
    ScaleLinear(f32),
    /// Unpremultiply RGB/Gray by alpha (`a <= 0 → 0`).
    Unpremul,
    /// Premultiply RGB/Gray by alpha.
    Premul,
    /// Source EOTF (non-linear → linear light).
    Eotf {
        transfer: Transfer,
        /// PQ scale (`10_000.0 / white_nits`), or `1.0` when staying in absolute `0..=1` PQ light.
        pq_scale: f32,
        /// Prepared HLG BT.2100-3 display renderer when `transfer == Hlg(Display(_))`.
        #[cfg(feature = "hdr-transfers")]
        hlg: Option<PreparedHlg>,
        /// Whether negative extended values use odd symmetry (`-f(-x)`).
        preserve_extended: bool,
    },
    /// Destination OETF (linear light → non-linear).
    Oetf {
        transfer: Transfer,
        /// Inverse PQ scale (`white_nits / 10_000.0`).
        pq_inv_scale: f32,
        #[cfg(feature = "hdr-transfers")]
        hlg: Option<PreparedHlg>,
        preserve_extended: bool,
        /// When `true` (destination is Gray/GrayAlpha), only encode `strip.r`.
        gray_only: bool,
    },
    /// Linear 3×3 gamut matrix multiply (`r, g, b`).
    Gamut3x3([[f32; 3]; 3]),
    /// Linear RGB (in `primaries`) → Oklab (`[L, a, b]`).
    #[cfg(feature = "oklab")]
    RgbToOklab([[f32; 3]; 3]),
    /// Oklab (`[L, a, b]`) → linear RGB (in `primaries`).
    #[cfg(feature = "oklab")]
    OklabToRgb([[f32; 3]; 3]),
    /// Linear RGB (in `primaries`) → perceptual/cylindrical space.
    #[cfg(feature = "perceptual")]
    RgbToPerceptual {
        space: PerceptualSpace,
        primaries: Primaries,
        white_nits: f32,
    },
    /// Perceptual/cylindrical space → linear RGB (in `primaries`).
    #[cfg(feature = "perceptual")]
    PerceptualToRgb {
        space: PerceptualSpace,
        primaries: Primaries,
        white_nits: f32,
    },
    /// HDR → SDR tone mapping on linear working-primary RGB.
    ToneMap {
        in_scale: f32,
        out_scale: f32,
        mapper: Arc<dyn StripMapper>,
    },
    /// OKLch soft chroma compression toward target gamut `[0, 1]`.
    #[cfg(feature = "oklab")]
    SoftCompress(SoftCompress),
    /// Clamp RGB color channels to `[0.0, 1.0]` (flushing NaN to `0.0`).
    ClampUnit,
    /// Convert linear RGB → linear Gray (`r = g = b = yr*r + yg*g + yb*b`).
    RgbToGray([f32; 3]),
    /// Broadcast Gray (`r`) → `r, g, b`.
    GrayToRgb,
    /// Composite straight color over linear matte `[mr, mg, mb]` and set `a = 1.0`.
    CompositeMatte([f32; 3]),
    /// Set alpha channel to `1.0` (e.g., `RGBX → RGBA`, `Opaque` normalization).
    FillOpaqueAlpha,
}

impl core::fmt::Debug for MicroOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ExpandNarrow { bits } => write!(f, "ExpandNarrow({bits}b)"),
            Self::CompressNarrow { bits } => write!(f, "CompressNarrow({bits}b)"),
            Self::ScaleLinear(s) => write!(f, "ScaleLinear({s})"),
            Self::Unpremul => f.write_str("Unpremul"),
            Self::Premul => f.write_str("Premul"),
            Self::Eotf { transfer, .. } => write!(f, "Eotf({transfer:?})"),
            Self::Oetf { transfer, .. } => write!(f, "Oetf({transfer:?})"),
            Self::Gamut3x3(_) => f.write_str("Gamut3x3"),
            #[cfg(feature = "oklab")]
            Self::RgbToOklab(_) => f.write_str("RgbToOklab"),
            #[cfg(feature = "oklab")]
            Self::OklabToRgb(_) => f.write_str("OklabToRgb"),
            #[cfg(feature = "perceptual")]
            Self::RgbToPerceptual { space, .. } => write!(f, "RgbToPerceptual({space:?})"),
            #[cfg(feature = "perceptual")]
            Self::PerceptualToRgb { space, .. } => write!(f, "PerceptualToRgb({space:?})"),
            Self::ToneMap { .. } => f.write_str("ToneMap"),
            #[cfg(feature = "oklab")]
            Self::SoftCompress(_) => f.write_str("SoftCompress"),
            Self::ClampUnit => f.write_str("ClampUnit"),
            Self::RgbToGray(_) => f.write_str("RgbToGray"),
            Self::GrayToRgb => f.write_str("GrayToRgb"),
            Self::CompositeMatte(_) => f.write_str("CompositeMatte"),
            Self::FillOpaqueAlpha => f.write_str("FillOpaqueAlpha"),
        }
    }
}

/// Execute a compiled micro-op sequence over one row using the 1 KiB stack [`StripBuf`].
#[inline(never)]
pub(crate) fn exec_strip_row(
    src_layout: SampleLayout,
    dst_layout: SampleLayout,
    ops: &[MicroOp],
    depth_policy: DepthPolicy,
    src_opaque_alpha: bool,
    accel: &Accelerators,
    src_row: &[u8],
    dst_row: &mut [u8],
    width: usize,
    row_y: u32,
) {
    let mut strip = StripBuf::new();
    let src_bpp = src_layout.bytes_per_pixel();
    let dst_bpp = dst_layout.bytes_per_pixel();

    // Fuse full-packing U8 unpack + leading sRGB EOTF directly through the
    // exact 256-entry sRGB LUT (0-bit identical between SCALAR & ALL!).
    let fuse_u8_srgb = src_layout.sample == SampleType::U8
        && src_layout.packing.is_full()
        && matches!(
            ops.first(),
            Some(MicroOp::Eotf {
                transfer: Transfer::Srgb,
                ..
            })
        );
    let active_ops = if fuse_u8_srgb { &ops[1..] } else { ops };

    let mut x = 0usize;
    while x < width {
        let len = (width - x).min(STRIP_LEN);
        let s_slice = &src_row[x * src_bpp..(x + len) * src_bpp];
        if fuse_u8_srgb {
            unpack_strip_u8_srgb_lut(src_layout.order, s_slice, &mut strip, len);
        } else if let Some(unpack_simd) = accel.strip_unpack
            && unpack_simd(src_layout, s_slice, &mut strip, len)
        {
        } else {
            unpack_strip_scalar(src_layout, s_slice, &mut strip, len, accel);
        }
        if src_opaque_alpha {
            strip.a[..len].fill(1.0);
        }

        for op in active_ops {
            apply_micro_op(op, &mut strip, len, accel);
        }

        let d_slice = &mut dst_row[x * dst_bpp..(x + len) * dst_bpp];
        if let Some(pack_simd) = accel.strip_pack
            && pack_simd(dst_layout, &strip, d_slice, len, depth_policy)
        {
        } else {
            pack_strip_scalar(
                dst_layout,
                &strip,
                d_slice,
                len,
                x,
                row_y,
                depth_policy,
                accel,
            );
        }
        x += len;
    }
}

/// Fused U8 sRGB unpack + EOTF via exact 256-entry const LUT (alpha stays linear).
#[inline(never)]
fn unpack_strip_u8_srgb_lut(order: ChannelOrder, src: &[u8], strip: &mut StripBuf, len: usize) {
    const INV: f32 = 1.0 / 255.0;
    use linear_srgb::default::srgb_u8_to_linear as lut;
    match order {
        ChannelOrder::Rgb => {
            for i in 0..len {
                strip.r[i] = lut(src[i * 3]);
                strip.g[i] = lut(src[i * 3 + 1]);
                strip.b[i] = lut(src[i * 3 + 2]);
            }
            strip.a[..len].fill(1.0);
        }
        ChannelOrder::Bgr => {
            for i in 0..len {
                strip.b[i] = lut(src[i * 3]);
                strip.g[i] = lut(src[i * 3 + 1]);
                strip.r[i] = lut(src[i * 3 + 2]);
            }
            strip.a[..len].fill(1.0);
        }
        ChannelOrder::Rgba => {
            for i in 0..len {
                strip.r[i] = lut(src[i * 4]);
                strip.g[i] = lut(src[i * 4 + 1]);
                strip.b[i] = lut(src[i * 4 + 2]);
                strip.a[i] = f32::from(src[i * 4 + 3]) * INV;
            }
        }
        ChannelOrder::Bgra => {
            for i in 0..len {
                strip.b[i] = lut(src[i * 4]);
                strip.g[i] = lut(src[i * 4 + 1]);
                strip.r[i] = lut(src[i * 4 + 2]);
                strip.a[i] = f32::from(src[i * 4 + 3]) * INV;
            }
        }
        ChannelOrder::Rgbx => {
            for i in 0..len {
                strip.r[i] = lut(src[i * 4]);
                strip.g[i] = lut(src[i * 4 + 1]);
                strip.b[i] = lut(src[i * 4 + 2]);
            }
            strip.a[..len].fill(1.0);
        }
        ChannelOrder::Bgrx => {
            for i in 0..len {
                strip.b[i] = lut(src[i * 4]);
                strip.g[i] = lut(src[i * 4 + 1]);
                strip.r[i] = lut(src[i * 4 + 2]);
            }
            strip.a[..len].fill(1.0);
        }
        ChannelOrder::Gray => {
            for i in 0..len {
                let y = lut(src[i]);
                strip.r[i] = y;
                strip.g[i] = y;
                strip.b[i] = y;
            }
            strip.a[..len].fill(1.0);
        }
        ChannelOrder::GrayAlpha => {
            for i in 0..len {
                let y = lut(src[i * 2]);
                strip.r[i] = y;
                strip.g[i] = y;
                strip.b[i] = y;
                strip.a[i] = f32::from(src[i * 2 + 1]) * INV;
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Unpack: interleaved bytes → 64-px SoA f32 strip (universal scalar)
// ---------------------------------------------------------------------------

#[inline(never)]
fn unpack_strip_scalar(
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
    accel: &Accelerators,
) {
    // Handle BitPacking::Bitfields (RGB565 / RGB10A2) first.
    if layout.packing.is_packed_word() {
        let bpp = layout.bytes_per_pixel();
        let ch = layout.channels();
        for i in 0..len {
            let word = read_word_u32(layout.sample, &src[i * bpp..]);
            let mut vals = [0.0f32; 4];
            vals[3] = 1.0;
            for c in 0..ch {
                let code = layout.packing.extract_code(layout.sample, word, c);
                let max = layout.packing.max_code(layout.sample, c).max(1);
                vals[c] = (code as f32) / (max as f32);
            }
            store_ordered(layout.order, vals, strip, i);
        }
        return;
    }

    // Handle BitPacking::Lsb / Msb (10/12-bit P010 / U10_LSB / etc.).
    if !layout.packing.is_full() {
        let ch = layout.channels();
        let sz = layout.sample.byte_size();
        let bpp = ch * sz;
        let max = layout.packing.max_code(layout.sample, 0).max(1) as f32;
        let inv = 1.0 / max;
        for i in 0..len {
            let base = i * bpp;
            let mut vals = [0.0f32; 4];
            vals[3] = 1.0;
            for c in 0..ch {
                let w = read_word_u32(layout.sample, &src[base + c * sz..]);
                let code = layout.packing.extract_code(layout.sample, w, c);
                vals[c] = (code as f32) * inv;
            }
            store_ordered(layout.order, vals, strip, i);
        }
        return;
    }

    let ch = layout.channels();
    match layout.sample {
        SampleType::U8 => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                let mut vals = [0.0f32; 4];
                vals[3] = 1.0;
                for c in 0..ch {
                    vals[c] = f32::from(src[i * ch + c]) * INV;
                }
                store_ordered(layout.order, vals, strip, i);
            }
        }
        SampleType::U16 => {
            const INV: f32 = 1.0 / 65535.0;
            for i in 0..len {
                let mut vals = [0.0f32; 4];
                vals[3] = 1.0;
                for c in 0..ch {
                    let o = (i * ch + c) * 2;
                    vals[c] = f32::from(u16::from_ne_bytes([src[o], src[o + 1]])) * INV;
                }
                store_ordered(layout.order, vals, strip, i);
            }
        }
        SampleType::F16 => {
            let count = len * ch;
            let mut bits = [0u16; STRIP_LEN * 4];
            let mut decoded = [0.0f32; STRIP_LEN * 4];
            for k in 0..count {
                bits[k] = u16::from_ne_bytes([src[k * 2], src[k * 2 + 1]]);
            }
            if let Some(f16_fn) = accel.f16_decode {
                f16_fn(&bits[..count], &mut decoded[..count]);
            } else {
                for k in 0..count {
                    decoded[k] = crate::f16_scalar::f16_bits_to_f32(bits[k]);
                }
            }
            for i in 0..len {
                let mut vals = [0.0f32; 4];
                vals[3] = 1.0;
                for c in 0..ch {
                    vals[c] = decoded[i * ch + c];
                }
                store_ordered(layout.order, vals, strip, i);
            }
        }
        SampleType::F32 => {
            for i in 0..len {
                let mut vals = [0.0f32; 4];
                vals[3] = 1.0;
                for c in 0..ch {
                    let o = (i * ch + c) * 4;
                    vals[c] = f32::from_ne_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
                }
                store_ordered(layout.order, vals, strip, i);
            }
        }
        _ => {}
    }
}

#[inline(always)]
fn store_ordered(order: ChannelOrder, vals: [f32; 4], strip: &mut StripBuf, i: usize) {
    match order {
        ChannelOrder::Gray => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[0];
            strip.b[i] = vals[0];
            strip.a[i] = 1.0;
        }
        ChannelOrder::GrayAlpha => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[0];
            strip.b[i] = vals[0];
            strip.a[i] = vals[1];
        }
        ChannelOrder::Rgb | ChannelOrder::Oklab | ChannelOrder::Xyz | ChannelOrder::Perceptual3 => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[1];
            strip.b[i] = vals[2];
            strip.a[i] = 1.0;
        }
        ChannelOrder::Bgr => {
            strip.r[i] = vals[2];
            strip.g[i] = vals[1];
            strip.b[i] = vals[0];
            strip.a[i] = 1.0;
        }
        ChannelOrder::Rgba
        | ChannelOrder::OklabA
        | ChannelOrder::XyzA
        | ChannelOrder::Perceptual4 => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[1];
            strip.b[i] = vals[2];
            strip.a[i] = vals[3];
        }
        ChannelOrder::Bgra => {
            strip.r[i] = vals[2];
            strip.g[i] = vals[1];
            strip.b[i] = vals[0];
            strip.a[i] = vals[3];
        }
        ChannelOrder::Rgbx => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[1];
            strip.b[i] = vals[2];
            strip.a[i] = 1.0;
        }
        ChannelOrder::Bgrx => {
            strip.r[i] = vals[2];
            strip.g[i] = vals[1];
            strip.b[i] = vals[0];
            strip.a[i] = 1.0;
        }
        _ => {
            strip.r[i] = vals[0];
            strip.g[i] = vals[1];
            strip.b[i] = vals[2];
            strip.a[i] = vals[3];
        }
    }
}

#[inline(always)]
fn read_word_u32(sample: SampleType, bytes: &[u8]) -> u32 {
    match sample {
        SampleType::U8 => u32::from(bytes[0]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::U32 => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => 0,
    }
}

#[inline(always)]
fn write_word_u32(sample: SampleType, dst: &mut [u8], word: u32) {
    match sample {
        SampleType::U8 => dst[0] = word as u8,
        SampleType::U16 => dst[0..2].copy_from_slice(&(word as u16).to_ne_bytes()),
        SampleType::U32 => dst[0..4].copy_from_slice(&word.to_ne_bytes()),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Micro-op execution on StripBuf
// ---------------------------------------------------------------------------

#[inline(always)]
fn narrow_params(bits: u8) -> (f32, f32, f32) {
    // ITU-R BT.2100-3 Table 9: for bit depth n >= 8, black = 16 << (n - 8), span = 219 << (n - 8),
    // max_code = (1 << n) - 1.
    let n = bits.clamp(8, 16);
    let shift = u32::from(n - 8);
    let max_code = ((1u32 << n) - 1) as f32;
    let lo = (16u32 << shift) as f32;
    let span = (219u32 << shift) as f32;
    (lo / max_code, max_code / span, span / max_code)
}

#[inline(never)]
fn apply_micro_op(op: &MicroOp, strip: &mut StripBuf, len: usize, accel: &Accelerators) {
    match op {
        MicroOp::ExpandNarrow { bits } => {
            let (offset, expand_scale, _) = narrow_params(*bits);
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v = (*v - offset) * expand_scale;
                }
            }
        }
        MicroOp::CompressNarrow { bits } => {
            let (offset, _, compress_scale) = narrow_params(*bits);
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v = offset + (*v).clamp(0.0, 1.0) * compress_scale;
                }
            }
        }
        MicroOp::ScaleLinear(factor) => {
            let f = *factor;
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v *= f;
                }
            }
        }
        MicroOp::Unpremul => {
            for i in 0..len {
                let a = strip.a[i];
                // Direct division avoids `1.0 / a = inf` overflow on positive subnormals
                // (so `0.0 / subnormal = 0.0`, never `0.0 * inf = NaN`).
                if a > 0.0 {
                    strip.r[i] /= a;
                    strip.g[i] /= a;
                    strip.b[i] /= a;
                } else {
                    strip.r[i] = 0.0;
                    strip.g[i] = 0.0;
                    strip.b[i] = 0.0;
                }
            }
        }
        MicroOp::Premul => {
            for i in 0..len {
                let a = strip.a[i];
                strip.r[i] *= a;
                strip.g[i] *= a;
                strip.b[i] *= a;
            }
        }
        MicroOp::Eotf {
            transfer,
            pq_scale,
            #[cfg(feature = "hdr-transfers")]
            hlg,
            preserve_extended,
        } => {
            apply_eotf(
                *transfer,
                *pq_scale,
                #[cfg(feature = "hdr-transfers")]
                hlg.as_ref(),
                *preserve_extended,
                strip,
                len,
                accel,
            );
        }
        MicroOp::Oetf {
            transfer,
            pq_inv_scale,
            #[cfg(feature = "hdr-transfers")]
            hlg,
            preserve_extended,
            gray_only,
        } => {
            apply_oetf(
                *transfer,
                *pq_inv_scale,
                #[cfg(feature = "hdr-transfers")]
                hlg.as_ref(),
                *preserve_extended,
                *gray_only,
                strip,
                len,
                accel,
            );
        }
        MicroOp::Gamut3x3(m) => {
            if let Some(f) = accel.gamut_3x3 {
                f(
                    m,
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                );
            } else {
                for i in 0..len {
                    let (vr, vg, vb) = (strip.r[i], strip.g[i], strip.b[i]);
                    strip.r[i] = (m[0][0] * vr + m[0][1] * vg) + m[0][2] * vb;
                    strip.g[i] = (m[1][0] * vr + m[1][1] * vg) + m[1][2] * vb;
                    strip.b[i] = (m[2][0] * vr + m[2][1] * vg) + m[2][2] * vb;
                }
            }
        }
        #[cfg(feature = "oklab")]
        MicroOp::RgbToOklab(m1) => {
            if let Some(f) = accel.rgb_to_oklab {
                f(
                    m1,
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                );
            } else {
                for i in 0..len {
                    let [l, a, b] =
                        crate::oklab::rgb_to_oklab(strip.r[i], strip.g[i], strip.b[i], m1);
                    strip.r[i] = l;
                    strip.g[i] = a;
                    strip.b[i] = b;
                }
            }
        }
        #[cfg(feature = "oklab")]
        MicroOp::OklabToRgb(m1_inv) => {
            if let Some(f) = accel.oklab_to_rgb {
                f(
                    m1_inv,
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                );
            } else {
                for i in 0..len {
                    let [r, g, b] =
                        crate::oklab::oklab_to_rgb(strip.r[i], strip.g[i], strip.b[i], m1_inv);
                    strip.r[i] = r;
                    strip.g[i] = g;
                    strip.b[i] = b;
                }
            }
        }
        #[cfg(feature = "perceptual")]
        MicroOp::RgbToPerceptual {
            space,
            primaries,
            white_nits,
        } => {
            for i in 0..len {
                let out = crate::perceptual::linear_rgb_to_perceptual(
                    [strip.r[i], strip.g[i], strip.b[i]],
                    *primaries,
                    *space,
                    *white_nits,
                );
                strip.r[i] = out[0];
                strip.g[i] = out[1];
                strip.b[i] = out[2];
            }
        }
        #[cfg(feature = "perceptual")]
        MicroOp::PerceptualToRgb {
            space,
            primaries,
            white_nits,
        } => {
            for i in 0..len {
                let out = crate::perceptual::perceptual_to_linear_rgb(
                    [strip.r[i], strip.g[i], strip.b[i]],
                    *primaries,
                    *space,
                    *white_nits,
                );
                strip.r[i] = out[0];
                strip.g[i] = out[1];
                strip.b[i] = out[2];
            }
        }
        MicroOp::ToneMap {
            in_scale,
            out_scale,
            mapper,
        } => {
            let mut interleaved = [0.0f32; STRIP_LEN * 3];
            for i in 0..len {
                interleaved[i * 3] = strip.r[i] * *in_scale;
                interleaved[i * 3 + 1] = strip.g[i] * *in_scale;
                interleaved[i * 3 + 2] = strip.b[i] * *in_scale;
            }
            let mut out = [0.0f32; STRIP_LEN * 3];
            mapper.map_strip(&interleaved[..len * 3], &mut out[..len * 3]);
            for i in 0..len {
                strip.r[i] = out[i * 3] * *out_scale;
                strip.g[i] = out[i * 3 + 1] * *out_scale;
                strip.b[i] = out[i * 3 + 2] * *out_scale;
            }
        }
        #[cfg(feature = "oklab")]
        MicroOp::SoftCompress(sc) => {
            let mut pixels = [[0.0f32; 3]; STRIP_LEN];
            for i in 0..len {
                pixels[i] = [strip.r[i], strip.g[i], strip.b[i]];
            }
            if let Some(f) = accel.soft_compress {
                f(sc, &mut pixels[..len]);
            } else {
                sc.apply_strip_scalar(&mut pixels[..len]);
            }
            for i in 0..len {
                strip.r[i] = pixels[i][0];
                strip.g[i] = pixels[i][1];
                strip.b[i] = pixels[i][2];
            }
        }
        MicroOp::ClampUnit => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                }
            }
        }
        MicroOp::RgbToGray([yr, yg, yb]) => {
            for i in 0..len {
                let y = yr * strip.r[i] + yg * strip.g[i] + yb * strip.b[i];
                strip.r[i] = y;
                strip.g[i] = y;
                strip.b[i] = y;
            }
        }
        MicroOp::GrayToRgb => {
            for i in 0..len {
                let y = strip.r[i];
                strip.g[i] = y;
                strip.b[i] = y;
            }
        }
        MicroOp::CompositeMatte([mr, mg, mb]) => {
            for i in 0..len {
                let a = strip.a[i].clamp(0.0, 1.0);
                let inv = 1.0 - a;
                strip.r[i] = strip.r[i] * a + mr * inv;
                strip.g[i] = strip.g[i] * a + mg * inv;
                strip.b[i] = strip.b[i] * a + mb * inv;
                strip.a[i] = 1.0;
            }
        }
        MicroOp::FillOpaqueAlpha => {
            strip.a[..len].fill(1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer functions (EOTF / OETF) — bit-exact between scalar & SIMD!
// ---------------------------------------------------------------------------

#[inline]
fn clamp_slice_unit(ch: &mut [f32]) {
    for v in ch.iter_mut() {
        *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    }
}

/// Apply a single-channel transfer function to a contiguous `f32` slice.
#[inline(never)]
fn apply_transfer_slice(
    transfer: Transfer,
    is_eotf: bool,
    pq_scale: f32,
    preserve_extended: bool,
    ch: &mut [f32],
    accel: &Accelerators,
) {
    let _ = pq_scale;
    match (transfer, is_eotf) {
        (Transfer::Linear | Transfer::Unknown, _) => {}
        (Transfer::Srgb, true) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.srgb_eotf {
                f(ch);
            } else {
                srgb_eotf_scalar_slice(ch);
            }
        }
        (Transfer::Srgb, false) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.srgb_oetf {
                f(ch);
            } else {
                srgb_oetf_scalar_slice(ch);
            }
        }
        (Transfer::Bt709, true) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.bt709_eotf {
                f(ch);
            } else {
                crate::transfers::bt709_eotf_scalar_slice(ch);
            }
        }
        (Transfer::Bt709, false) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.bt709_oetf {
                f(ch);
            } else {
                crate::transfers::bt709_oetf_scalar_slice(ch);
            }
        }
        (Transfer::Gamma22, true) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.gamma22_eotf {
                f(ch);
            } else {
                crate::transfers::gamma22_eotf_scalar_slice(ch);
            }
        }
        (Transfer::Gamma22, false) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.gamma22_oetf {
                f(ch);
            } else {
                crate::transfers::gamma22_oetf_scalar_slice(ch);
            }
        }
        (Transfer::Gamma(gamma), true) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            crate::transfers::gamma_eotf_scalar_slice(ch, gamma.exponent());
        }
        (Transfer::Gamma(gamma), false) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            crate::transfers::gamma_oetf_scalar_slice(ch, gamma.exponent());
        }
        #[cfg(feature = "hdr-transfers")]
        (Transfer::Pq, true) => {
            if let Some(f) = accel.pq_eotf {
                f(ch, pq_scale);
            } else {
                crate::transfers::hdr::pq_eotf_scalar_slice(ch, pq_scale);
            }
        }
        #[cfg(feature = "hdr-transfers")]
        (Transfer::Pq, false) => {
            if let Some(f) = accel.pq_oetf {
                f(ch, pq_scale);
            } else {
                crate::transfers::hdr::pq_oetf_scalar_slice(ch, pq_scale);
            }
        }
        #[cfg(feature = "hdr-transfers")]
        (Transfer::Hlg(HlgMode::Scene), true) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.hlg_scene_eotf {
                f(ch);
            } else {
                crate::transfers::hdr::hlg_scene_eotf_scalar_slice(ch);
            }
        }
        #[cfg(feature = "hdr-transfers")]
        (Transfer::Hlg(HlgMode::Scene), false) => {
            if !preserve_extended {
                clamp_slice_unit(ch);
            }
            if let Some(f) = accel.hlg_scene_oetf {
                f(ch);
            } else {
                crate::transfers::hdr::hlg_scene_oetf_scalar_slice(ch);
            }
        }
        _ => {}
    }
}

#[inline(never)]
fn apply_eotf(
    transfer: Transfer,
    pq_scale: f32,
    #[cfg(feature = "hdr-transfers")] hlg: Option<&PreparedHlg>,
    preserve_extended: bool,
    strip: &mut StripBuf,
    len: usize,
    accel: &Accelerators,
) {
    match transfer {
        Transfer::Linear | Transfer::Unknown => {}
        Transfer::Hlg(HlgMode::Display(_)) =>
        {
            #[cfg(feature = "hdr-transfers")]
            if let Some(prepared) = hlg {
                if let Some(f) = accel.hlg_display_soa {
                    f(
                        prepared,
                        &mut strip.r[..len],
                        &mut strip.g[..len],
                        &mut strip.b[..len],
                        len,
                    );
                } else {
                    prepared.apply_soa_scalar(
                        &mut strip.r[..len],
                        &mut strip.g[..len],
                        &mut strip.b[..len],
                        len,
                    );
                }
            }
        }
        _ => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                apply_transfer_slice(transfer, true, pq_scale, preserve_extended, ch, accel);
            }
        }
    }
}

#[inline(never)]
fn apply_oetf(
    transfer: Transfer,
    pq_inv_scale: f32,
    #[cfg(feature = "hdr-transfers")] hlg: Option<&PreparedHlg>,
    preserve_extended: bool,
    gray_only: bool,
    strip: &mut StripBuf,
    len: usize,
    accel: &Accelerators,
) {
    match transfer {
        Transfer::Linear | Transfer::Unknown => {}
        Transfer::Hlg(HlgMode::Display(_)) =>
        {
            #[cfg(feature = "hdr-transfers")]
            if let Some(prepared) = hlg {
                if let Some(f) = accel.hlg_display_soa {
                    f(
                        prepared,
                        &mut strip.r[..len],
                        &mut strip.g[..len],
                        &mut strip.b[..len],
                        len,
                    );
                } else {
                    prepared.apply_soa_scalar(
                        &mut strip.r[..len],
                        &mut strip.g[..len],
                        &mut strip.b[..len],
                        len,
                    );
                }
            }
        }
        _ => {
            if gray_only {
                apply_transfer_slice(
                    transfer,
                    false,
                    pq_inv_scale,
                    preserve_extended,
                    &mut strip.r[..len],
                    accel,
                );
            } else {
                for ch in [
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                ] {
                    apply_transfer_slice(
                        transfer,
                        false,
                        pq_inv_scale,
                        preserve_extended,
                        ch,
                        accel,
                    );
                }
            }
        }
    }
}

#[inline(never)]
fn srgb_eotf_scalar_slice(vals: &mut [f32]) {
    let (chunks, rem) = vals.as_chunks_mut::<4>();
    for c in chunks {
        *c = linear_srgb::target_features::x4::srgb_to_linear_extended_scalar(*c);
    }
    if !rem.is_empty() {
        let mut buf = [0.0f32; 4];
        buf[..rem.len()].copy_from_slice(rem);
        let out = linear_srgb::target_features::x4::srgb_to_linear_extended_scalar(buf);
        rem.copy_from_slice(&out[..rem.len()]);
    }
}

#[inline(never)]
fn srgb_oetf_scalar_slice(vals: &mut [f32]) {
    let (chunks, rem) = vals.as_chunks_mut::<4>();
    for c in chunks {
        *c = linear_srgb::target_features::x4::linear_to_srgb_extended_scalar(*c);
    }
    if !rem.is_empty() {
        let mut buf = [0.0f32; 4];
        buf[..rem.len()].copy_from_slice(rem);
        let out = linear_srgb::target_features::x4::linear_to_srgb_extended_scalar(buf);
        rem.copy_from_slice(&out[..rem.len()]);
    }
}

/// Exact SMPTE ST 2084 PQ EOTF (`[0, 1] → [0, 1]` where `1.0 = 10,000 cd/m²`; NaN → 0.0).
#[cfg(not(feature = "hdr-transfers"))]
#[inline]
pub(crate) fn pq_eotf_scalar(v: f32) -> f32 {
    const M1: f64 = 1305.0 / 8192.0;
    const M2: f64 = 2523.0 / 32.0;
    const C1: f64 = 107.0 / 128.0;
    const C2: f64 = 2413.0 / 128.0;
    const C3: f64 = 2392.0 / 128.0;
    if v.is_nan() || v <= 0.0 {
        return 0.0;
    }
    let e = f64::from(v.min(1.0));
    let ep = libm::pow(e, 1.0 / M2);
    let num = (ep - C1).max(0.0);
    let den = C2 - C3 * ep;
    if den <= 0.0 {
        return 1.0;
    }
    libm::pow(num / den, 1.0 / M1) as f32
}

// ---------------------------------------------------------------------------
// Pack: 64-px SoA f32 strip → interleaved destination bytes (universal scalar)
// ---------------------------------------------------------------------------

#[inline(always)]
fn quantize_to_u32(v: f32, max_code: u32, policy: DepthPolicy, x: usize, y: u32) -> u32 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let max_f = max_code as f32;
    let scaled = (v.min(1.0)) * max_f;
    let q = match policy {
        DepthPolicy::Round | DepthPolicy::Forbid => scaled + 0.5,
        DepthPolicy::Truncate => scaled,
        DepthPolicy::OrderedDither => {
            let d = BAYER_4X4[(y as usize) & 3][x & 3];
            (scaled + d + 0.5).clamp(0.0, max_f)
        }
    };
    (q as u32).min(max_code)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn pack_strip_scalar(
    layout: SampleLayout,
    strip: &StripBuf,
    dst: &mut [u8],
    len: usize,
    base_x: usize,
    row_y: u32,
    depth_policy: DepthPolicy,
    accel: &Accelerators,
) {
    // Handle BitPacking::Bitfields (RGB565 / RGB10A2).
    if layout.packing.is_packed_word() {
        let bpp = layout.bytes_per_pixel();
        let ch = layout.channels();
        for i in 0..len {
            let ordered = load_ordered(layout.order, strip, i);
            let mut word = 0u32;
            for c in 0..ch {
                let max = layout.packing.max_code(layout.sample, c);
                let q = quantize_to_u32(ordered[c], max, depth_policy, base_x + i, row_y);
                word |= layout.packing.pack_code(layout.sample, c, q);
            }
            write_word_u32(layout.sample, &mut dst[i * bpp..], word);
        }
        return;
    }

    // Handle BitPacking::Lsb / Msb (P010 / U10_LSB / etc.).
    if !layout.packing.is_full() {
        let ch = layout.channels();
        let sz = layout.sample.byte_size();
        let bpp = ch * sz;
        let max = layout.packing.max_code(layout.sample, 0);
        for i in 0..len {
            let ordered = load_ordered(layout.order, strip, i);
            let base = i * bpp;
            for c in 0..ch {
                let q = quantize_to_u32(ordered[c], max, depth_policy, base_x + i, row_y);
                let packed = layout.packing.pack_code(layout.sample, c, q);
                write_word_u32(layout.sample, &mut dst[base + c * sz..], packed);
            }
        }
        return;
    }

    let ch = layout.channels();
    match layout.sample {
        SampleType::U8 => {
            for i in 0..len {
                let ordered = load_ordered(layout.order, strip, i);
                for c in 0..ch {
                    dst[i * ch + c] =
                        quantize_to_u32(ordered[c], 255, depth_policy, base_x + i, row_y) as u8;
                }
            }
        }
        SampleType::U16 => {
            for i in 0..len {
                let ordered = load_ordered(layout.order, strip, i);
                for c in 0..ch {
                    let q =
                        quantize_to_u32(ordered[c], 65535, depth_policy, base_x + i, row_y) as u16;
                    let o = (i * ch + c) * 2;
                    dst[o..o + 2].copy_from_slice(&q.to_ne_bytes());
                }
            }
        }
        SampleType::F16 => {
            let count = len * ch;
            let mut flat = [0.0f32; STRIP_LEN * 4];
            let mut bits = [0u16; STRIP_LEN * 4];
            for i in 0..len {
                let ordered = load_ordered(layout.order, strip, i);
                for c in 0..ch {
                    flat[i * ch + c] = ordered[c];
                }
            }
            if let Some(f16_fn) = accel.f16_encode {
                f16_fn(&flat[..count], &mut bits[..count]);
            } else {
                for k in 0..count {
                    bits[k] = crate::f16_scalar::f32_to_f16_bits(flat[k]);
                }
            }
            for k in 0..count {
                dst[k * 2..k * 2 + 2].copy_from_slice(&bits[k].to_ne_bytes());
            }
        }
        SampleType::F32 => {
            for i in 0..len {
                let ordered = load_ordered(layout.order, strip, i);
                for c in 0..ch {
                    let o = (i * ch + c) * 4;
                    dst[o..o + 4].copy_from_slice(&ordered[c].to_ne_bytes());
                }
            }
        }
        _ => {}
    }
}

#[inline(always)]
fn load_ordered(order: ChannelOrder, strip: &StripBuf, i: usize) -> [f32; 4] {
    match order {
        ChannelOrder::Gray => [strip.r[i], 0.0, 0.0, 0.0],
        ChannelOrder::GrayAlpha => [strip.r[i], strip.a[i], 0.0, 0.0],
        ChannelOrder::Rgb | ChannelOrder::Oklab | ChannelOrder::Xyz | ChannelOrder::Perceptual3 => {
            [strip.r[i], strip.g[i], strip.b[i], 1.0]
        }
        ChannelOrder::Bgr => [strip.b[i], strip.g[i], strip.r[i], 1.0],
        ChannelOrder::Rgba
        | ChannelOrder::OklabA
        | ChannelOrder::XyzA
        | ChannelOrder::Perceptual4 => [strip.r[i], strip.g[i], strip.b[i], strip.a[i]],
        ChannelOrder::Bgra => [strip.b[i], strip.g[i], strip.r[i], strip.a[i]],
        ChannelOrder::Rgbx => [strip.r[i], strip.g[i], strip.b[i], 1.0],
        ChannelOrder::Bgrx => [strip.b[i], strip.g[i], strip.r[i], 1.0],
        _ => [strip.r[i], strip.g[i], strip.b[i], strip.a[i]],
    }
}

/// Resolve RGB→Gray linear luminance weights from [`LumaCoefficients`] and source primaries.
#[inline]
pub(crate) const fn resolve_luma_weights(
    coeffs: LumaCoefficients,
    src_primaries: Primaries,
) -> [f32; 3] {
    match coeffs {
        LumaCoefficients::FromSourcePrimaries => src_primaries.luma_weights(),
        LumaCoefficients::Bt709 => [0.2126, 0.7152, 0.0722],
        LumaCoefficients::Bt601 => [0.299, 0.587, 0.114],
        LumaCoefficients::Bt2020 => [0.2627, 0.6780, 0.0593],
        LumaCoefficients::Custom(w) => w,
    }
}

/// Detect whether a pair of `(src, dst)` specs can execute via a byte-only [`ByteFastPath`].
pub(crate) fn detect_byte_fast_path(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy_depth: DepthPolicy,
    matte_active: bool,
) -> Option<ByteFastPath> {
    if matte_active || !src.layout.packing.is_full() || !dst.layout.packing.is_full() {
        return None;
    }
    // Identical spec -> direct memcpy regardless of color space!
    if src == dst {
        return Some(ByteFastPath::Memcpy);
    }

    // Check direct RGBA8 / BGRA8 straight <-> premultiplied (same transfer/primaries/range/order).
    if src.layout.sample == SampleType::U8
        && dst.layout.sample == SampleType::U8
        && src.layout.order == dst.layout.order
        && matches!(src.layout.order, ChannelOrder::Rgba | ChannelOrder::Bgra)
        && let (
            ColorSpace::Rgb {
                primaries: sp,
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Rgb {
                primaries: dp,
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) = (&src.color, &dst.color)
        && sp == dp
        && st == dt
        && sr == dr
        && *sr == SignalRange::Full
        && sw == dw
    {
        if *sa == AlphaMode::Straight && *da == AlphaMode::Premultiplied {
            return match policy_depth {
                DepthPolicy::Round | DepthPolicy::Forbid => {
                    Some(ByteFastPath::Rgba8StraightToPremul { round: true })
                }
                DepthPolicy::Truncate => Some(ByteFastPath::Rgba8StraightToPremul { round: false }),
                DepthPolicy::OrderedDither => None,
            };
        }
        if *sa == AlphaMode::Premultiplied && *da == AlphaMode::Straight {
            return match policy_depth {
                DepthPolicy::Round | DepthPolicy::Forbid => {
                    Some(ByteFastPath::Rgba8PremulToStraight)
                }
                DepthPolicy::Truncate | DepthPolicy::OrderedDither => None,
            };
        }
    }

    // Determine whether 4-channel alpha must be forced to opaque (canonicalizing AlphaMode::Opaque
    // on either source or destination so unverified alpha bytes never leak!).
    let needs_opaque_fill = (src.color.alpha() == AlphaMode::Opaque
        || dst.color.alpha() == AlphaMode::Opaque)
        && dst.layout.order.has_alpha_slot()
        && !(src.color.alpha() == AlphaMode::Opaque
            && dst.color.alpha() == AlphaMode::Opaque
            && src.layout.sample == dst.layout.sample
            && src.layout.order == dst.layout.order);
    let src_is_narrow = matches!(
        src.color,
        ColorSpace::Rgb {
            range: SignalRange::Narrow,
            ..
        } | ColorSpace::Gray {
            range: SignalRange::Narrow,
            ..
        }
    );

    // Check that color meaning agrees (ignoring alpha presence changes handled by layout).
    let colors_agree = match (&src.color, &dst.color) {
        (
            ColorSpace::Rgb {
                primaries: sp,
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Rgb {
                primaries: dp,
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) => {
            sp == dp
                && st == dt
                && sr == dr
                && sw == dw
                && (*sa == *da
                    || matches!(
                        (*sa, *da),
                        (
                            AlphaMode::None | AlphaMode::Undefined,
                            AlphaMode::Straight
                                | AlphaMode::Opaque
                                | AlphaMode::Premultiplied
                                | AlphaMode::Undefined
                                | AlphaMode::None
                        ) | (
                            AlphaMode::Straight | AlphaMode::Opaque,
                            AlphaMode::None
                                | AlphaMode::Undefined
                                | AlphaMode::Opaque
                                | AlphaMode::Straight
                        ) | (AlphaMode::Opaque, AlphaMode::Premultiplied)
                    ))
        }
        (
            ColorSpace::Gray {
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Rgb {
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
                ..
            },
        ) => {
            st == dt
                && sr == dr
                && sw == dw
                && *sa == AlphaMode::None
                && matches!(
                    *da,
                    AlphaMode::None
                        | AlphaMode::Straight
                        | AlphaMode::Opaque
                        | AlphaMode::Premultiplied
                )
        }
        (
            ColorSpace::Gray {
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            ColorSpace::Gray {
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) => st == dt && sr == dr && sw == dw && sa == da,
        (
            ColorSpace::Xyz {
                white_point: wa,
                alpha: sa,
                white: wwa,
            },
            ColorSpace::Xyz {
                white_point: wb,
                alpha: da,
                white: wwb,
            },
        ) => wa == wb && wwa == wwb && sa == da,
        _ => false,
    };
    if !colors_agree {
        return None;
    }

    use ChannelOrder as O;
    use SampleType as S;
    if needs_opaque_fill {
        return match (
            src.layout.sample,
            src.layout.order,
            dst.layout.sample,
            dst.layout.order,
        ) {
            (S::U8, O::Rgba, S::U8, O::Rgba)
            | (S::U8, O::Bgra, S::U8, O::Bgra)
            | (S::U8, O::Rgbx, S::U8, O::Rgba)
            | (S::U8, O::Bgrx, S::U8, O::Bgra) => Some(ByteFastPath::FillAlpha4U8),
            (S::U8, O::Rgba, S::U8, O::Bgra)
            | (S::U8, O::Bgra, S::U8, O::Rgba)
            | (S::U8, O::Rgbx, S::U8, O::Bgra)
            | (S::U8, O::Bgrx, S::U8, O::Rgba) => Some(ByteFastPath::SwapRbFillAlpha4U8),
            _ => None,
        };
    }

    match (
        src.layout.sample,
        src.layout.order,
        dst.layout.sample,
        dst.layout.order,
    ) {
        // U8 byte layouts
        (S::U8, O::Rgb, S::U8, O::Rgba) => Some(ByteFastPath::Rgb8ToRgba8),
        (S::U8, O::Rgb, S::U8, O::Bgra) => Some(ByteFastPath::Rgb8ToBgra8),
        (S::U8, O::Rgba, S::U8, O::Rgb) | (S::U8, O::Rgbx, S::U8, O::Rgb) => {
            Some(ByteFastPath::Rgba8ToRgb8)
        }
        (S::U8, O::Bgra, S::U8, O::Rgb) | (S::U8, O::Bgrx, S::U8, O::Rgb) => {
            Some(ByteFastPath::Bgra8ToRgb8)
        }
        (S::U8, O::Rgb, S::U8, O::Bgr) | (S::U8, O::Bgr, S::U8, O::Rgb) => {
            Some(ByteFastPath::SwapRb3U8)
        }
        (S::U8, O::Rgba, S::U8, O::Bgra)
        | (S::U8, O::Bgra, S::U8, O::Rgba)
        | (S::U8, O::Rgbx, S::U8, O::Bgrx)
        | (S::U8, O::Bgrx, S::U8, O::Rgbx) => Some(ByteFastPath::SwapRb4U8),
        (S::U8, O::Rgbx, S::U8, O::Rgba) | (S::U8, O::Bgrx, S::U8, O::Bgra) => {
            Some(ByteFastPath::FillAlpha4U8)
        }
        (S::U8, O::Rgbx, S::U8, O::Bgra) | (S::U8, O::Bgrx, S::U8, O::Rgba) => {
            Some(ByteFastPath::SwapRbFillAlpha4U8)
        }
        (S::U8, O::Gray, S::U8, O::Rgb) => Some(ByteFastPath::Gray8ToRgb8),
        (S::U8, O::Gray, S::U8, O::Rgba) => Some(ByteFastPath::Gray8ToRgba8),

        // Pure integer full-range U8 <-> U16 depth changes (narrow range uses BT.2100-3 shift in strip!)
        (S::U8, so, S::U16, do_) if so == do_ && !src_is_narrow => Some(ByteFastPath::U8ToU16Exact),
        (S::U16, so, S::U8, do_) if so == do_ && !src_is_narrow => match policy_depth {
            DepthPolicy::Round | DepthPolicy::Forbid => Some(ByteFastPath::U16ToU8Round),
            DepthPolicy::Truncate => Some(ByteFastPath::U16ToU8Trunc),
            DepthPolicy::OrderedDither => None,
        },

        // Wide 2-byte/channel layouts (U16 & F16)
        (S::U16, O::Rgba, S::U16, O::Rgb)
        | (S::U16, O::Rgbx, S::U16, O::Rgb)
        | (S::F16, O::Rgba, S::F16, O::Rgb)
        | (S::F16, O::Rgbx, S::F16, O::Rgb) => Some(ByteFastPath::DropAlpha4To3B2),
        (S::U16, O::Bgra, S::U16, O::Rgb)
        | (S::U16, O::Bgrx, S::U16, O::Rgb)
        | (S::F16, O::Bgra, S::F16, O::Rgb)
        | (S::F16, O::Bgrx, S::F16, O::Rgb) => Some(ByteFastPath::SwapDropAlpha4To3B2),
        (S::U16, O::Rgb, S::U16, O::Rgba) => Some(ByteFastPath::AddAlpha3To4B2(0xFFFF)),
        (S::F16, O::Rgb, S::F16, O::Rgba) => Some(ByteFastPath::AddAlpha3To4B2(0x3C00)),
        (S::U16, O::Rgb, S::U16, O::Bgra) => Some(ByteFastPath::SwapAddAlpha3To4B2(0xFFFF)),
        (S::F16, O::Rgb, S::F16, O::Bgra) => Some(ByteFastPath::SwapAddAlpha3To4B2(0x3C00)),
        (S::U16, O::Rgb, S::U16, O::Bgr)
        | (S::U16, O::Bgr, S::U16, O::Rgb)
        | (S::F16, O::Rgb, S::F16, O::Bgr)
        | (S::F16, O::Bgr, S::F16, O::Rgb) => Some(ByteFastPath::SwapRb3B2),
        (S::U16, O::Rgba, S::U16, O::Bgra)
        | (S::U16, O::Bgra, S::U16, O::Rgba)
        | (S::F16, O::Rgba, S::F16, O::Bgra)
        | (S::F16, O::Bgra, S::F16, O::Rgba) => Some(ByteFastPath::SwapRb4B2),
        (S::U16, O::Gray, S::U16, O::Rgb) | (S::F16, O::Gray, S::F16, O::Rgb) => {
            Some(ByteFastPath::GrayToRgbB2)
        }
        (S::U16, O::Gray, S::U16, O::Rgba) => Some(ByteFastPath::GrayToRgbaB2(0xFFFF)),
        (S::F16, O::Gray, S::F16, O::Rgba) => Some(ByteFastPath::GrayToRgbaB2(0x3C00)),

        // Wide 4-byte/channel layouts (F32)
        (S::F32, O::Rgba, S::F32, O::Rgb) | (S::F32, O::Rgbx, S::F32, O::Rgb) => {
            Some(ByteFastPath::DropAlpha4To3B4)
        }
        (S::F32, O::Bgra, S::F32, O::Rgb) | (S::F32, O::Bgrx, S::F32, O::Rgb) => {
            Some(ByteFastPath::SwapDropAlpha4To3B4)
        }
        (S::F32, O::Rgb, S::F32, O::Rgba) => Some(ByteFastPath::AddAlpha3To4B4),
        (S::F32, O::Rgb, S::F32, O::Bgra) => Some(ByteFastPath::SwapAddAlpha3To4B4),
        (S::F32, O::Rgb, S::F32, O::Bgr) | (S::F32, O::Bgr, S::F32, O::Rgb) => {
            Some(ByteFastPath::SwapRb3B4)
        }
        (S::F32, O::Rgba, S::F32, O::Bgra) | (S::F32, O::Bgra, S::F32, O::Rgba) => {
            Some(ByteFastPath::SwapRb4B4)
        }
        (S::F32, O::Gray, S::F32, O::Rgb) => Some(ByteFastPath::GrayToRgbB4),
        (S::F32, O::Gray, S::F32, O::Rgba) => Some(ByteFastPath::GrayToRgbaB4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_u16_to_u8_round_and_trunc_match_float_oracle() {
        for v in 0u16..=65535 {
            let f = f32::from(v) * (1.0 / 65535.0);
            let want_round = ((f * 255.0 + 0.5) as u32).min(255) as u8;
            let got_round = ((u32::from(v) * 255 + 32895) >> 16) as u8;
            assert_eq!(got_round, want_round, "u16->u8 round mismatch at v={v}");

            let want_trunc = (u32::from(v) * 255 / 65535) as u8;
            let got_trunc = ((u32::from(v) * 255 + 255) >> 16) as u8;
            assert_eq!(got_trunc, want_trunc, "u16->u8 trunc mismatch at v={v}");
        }
    }

    #[test]
    fn exhaustive_rgba8_premul_and_unpremul_match_exact_round() {
        for a in 0u8..=255 {
            for c in 0u8..=255 {
                let want_pre = ((u32::from(c) * u32::from(a) + 127) / 255) as u8;
                let got_pre = (((u32::from(c) * u32::from(a) + 128) * 257) >> 16) as u8;
                assert_eq!(got_pre, want_pre, "premul mismatch at c={c}, a={a}");

                let want_un = if a == 0 {
                    0u8
                } else if c >= a {
                    255u8
                } else {
                    ((u32::from(c) * 255 + u32::from(a) / 2) / u32::from(a)) as u8
                };
                let got_un = if a == 0 {
                    0u8
                } else if c >= a {
                    255u8
                } else {
                    ((u32::from(c) * UNPREMUL_U8_SCALE[usize::from(a)] + 32768) >> 16) as u8
                };
                assert_eq!(got_un, want_un, "unpremul mismatch at c={c}, a={a}");
            }
        }
    }
}
