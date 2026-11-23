//! 64-pixel stack-allocated SoA L1-strip execution engine + `garb` byte-only fast paths.
//!
//! Every color/transfer/gamut/HDR/depth conversion runs on a fixed 1 KiB
//! stack-allocated [`StripBuf`] (`[f32; 64]` × 4 channels), keeping all active
//! lanes hot in L1 registers with **zero heap scratch allocation** during
//! row conversion.

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use alloc::sync::Arc;
#[cfg(feature = "perceptual")]
use zenpixels::PerceptualSpace;
use zenpixels::{
    AlphaMode, ChannelOrder, HlgMode, ImageSpec, Primaries, SampleLayout, SampleType, Transfer,
};

use crate::accel::Accelerators;
#[cfg(feature = "hdr-transfers")]
use crate::hlg_render::PreparedHlg;
use crate::policy::{DepthPolicy, LumaCoefficients};
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

/// Direct byte-only fast path that bypasses the `f32` SoA pipeline completely
/// when no color/transfer/gamut/range math is needed.
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
}

impl ByteFastPath {
    /// Execute this byte-only fast path on one row (`width` pixels).
    pub(crate) fn exec(self, src: &[u8], dst: &mut [u8], width: usize, use_garb: bool) {
        match self {
            Self::Memcpy => {
                let len = src.len().min(dst.len());
                dst[..len].copy_from_slice(&src[..len]);
            }
            Self::Rgb8ToRgba8 => {
                let s = &src[..width * 3];
                let d = &mut dst[..width * 4];
                if use_garb && garb::bytes::rgb_to_rgba(s, d).is_ok() {
                    return;
                }
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
                if use_garb && garb::bytes::rgb_to_bgra(s, d).is_ok() {
                    return;
                }
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
                if use_garb && garb::bytes::rgba_to_rgb(s, d).is_ok() {
                    return;
                }
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                    dp[2] = sp[2];
                }
            }
            Self::Bgra8ToRgb8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 3];
                if use_garb && garb::bytes::bgra_to_rgb(s, d).is_ok() {
                    return;
                }
                for (sp, dp) in s.chunks_exact(4).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                }
            }
            Self::SwapRb3U8 => {
                let s = &src[..width * 3];
                let d = &mut dst[..width * 3];
                if use_garb && garb::bytes::rgb_to_bgr(s, d).is_ok() {
                    return;
                }
                for (sp, dp) in s.chunks_exact(3).zip(d.chunks_exact_mut(3)) {
                    dp[0] = sp[2];
                    dp[1] = sp[1];
                    dp[2] = sp[0];
                }
            }
            Self::SwapRb4U8 => {
                let s = &src[..width * 4];
                let d = &mut dst[..width * 4];
                if use_garb && garb::bytes::rgba_to_bgra(s, d).is_ok() {
                    return;
                }
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
                if use_garb {
                    d.copy_from_slice(s);
                    if garb::bytes::fill_alpha_rgba(d).is_ok() {
                        return;
                    }
                }
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
                if use_garb && garb::bytes::gray_to_rgba(s, d).is_ok() {
                    return;
                }
                for (&y, dp) in s.iter().zip(d.chunks_exact_mut(4)) {
                    dp[0] = y;
                    dp[1] = y;
                    dp[2] = y;
                    dp[3] = 255;
                }
            }
            Self::U8ToU16Exact => {
                let n = src.len().min(dst.len() / 2);
                for (i, &b) in src[..n].iter().enumerate() {
                    let v = u16::from(b) * 257;
                    dst[i * 2..i * 2 + 2].copy_from_slice(&v.to_ne_bytes());
                }
            }
        }
    }
}

/// Composable micro-op executed in-place on a 64-pixel [`StripBuf`].
#[derive(Clone)]
pub(crate) enum MicroOp {
    /// Expand narrow studio integer range to full `[0.0, 1.0]` on color channels (alpha untouched).
    ExpandNarrow { sample: SampleType },
    /// Compress full `[0.0, 1.0]` color channels to narrow studio range (alpha untouched).
    CompressNarrow { sample: SampleType },
    /// Unpremultiply RGB/Gray by alpha (`a == 0 → 0`).
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
            Self::ExpandNarrow { sample } => write!(f, "ExpandNarrow({sample:?})"),
            Self::CompressNarrow { sample } => write!(f, "CompressNarrow({sample:?})"),
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
pub(crate) fn exec_strip_row(
    src_layout: SampleLayout,
    dst_layout: SampleLayout,
    ops: &[MicroOp],
    depth_policy: DepthPolicy,
    accel: &Accelerators,
    src_row: &[u8],
    dst_row: &mut [u8],
    width: usize,
    row_y: u32,
) {
    let mut strip = StripBuf::new();
    let src_bpp = src_layout.bytes_per_pixel();
    let dst_bpp = dst_layout.bytes_per_pixel();
    let mut x = 0usize;
    while x < width {
        let len = (width - x).min(STRIP_LEN);
        unpack_strip(
            src_layout,
            &src_row[x * src_bpp..(x + len) * src_bpp],
            &mut strip,
            len,
            accel,
        );
        for op in ops {
            apply_micro_op(op, &mut strip, len, accel);
        }
        pack_strip(
            dst_layout,
            &strip,
            &mut dst_row[x * dst_bpp..(x + len) * dst_bpp],
            len,
            x,
            row_y,
            depth_policy,
            accel,
        );
        x += len;
    }
}

// ---------------------------------------------------------------------------
// Unpack: interleaved bytes → 64-px SoA f32 strip
// ---------------------------------------------------------------------------

fn unpack_strip(
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
    accel: &Accelerators,
) {
    // Handle BitPacking::Bitfields (RGB565 / RGB10A2) first.
    if layout.packing.is_packed_word() {
        let bpp = layout.bytes_per_pixel();
        for i in 0..len {
            let word = read_word_u32(layout.sample, &src[i * bpp..]);
            let ch = layout.channels();
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

    match (layout.sample, layout.order) {
        (SampleType::U8, ChannelOrder::Rgb) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.r[i] = f32::from(src[i * 3]) * INV;
                strip.g[i] = f32::from(src[i * 3 + 1]) * INV;
                strip.b[i] = f32::from(src[i * 3 + 2]) * INV;
                strip.a[i] = 1.0;
            }
        }
        (SampleType::U8, ChannelOrder::Bgr) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.b[i] = f32::from(src[i * 3]) * INV;
                strip.g[i] = f32::from(src[i * 3 + 1]) * INV;
                strip.r[i] = f32::from(src[i * 3 + 2]) * INV;
                strip.a[i] = 1.0;
            }
        }
        (SampleType::U8, ChannelOrder::Rgba) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.r[i] = f32::from(src[i * 4]) * INV;
                strip.g[i] = f32::from(src[i * 4 + 1]) * INV;
                strip.b[i] = f32::from(src[i * 4 + 2]) * INV;
                strip.a[i] = f32::from(src[i * 4 + 3]) * INV;
            }
        }
        (SampleType::U8, ChannelOrder::Bgra) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.b[i] = f32::from(src[i * 4]) * INV;
                strip.g[i] = f32::from(src[i * 4 + 1]) * INV;
                strip.r[i] = f32::from(src[i * 4 + 2]) * INV;
                strip.a[i] = f32::from(src[i * 4 + 3]) * INV;
            }
        }
        (SampleType::U8, ChannelOrder::Rgbx) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.r[i] = f32::from(src[i * 4]) * INV;
                strip.g[i] = f32::from(src[i * 4 + 1]) * INV;
                strip.b[i] = f32::from(src[i * 4 + 2]) * INV;
                strip.a[i] = 1.0;
            }
        }
        (SampleType::U8, ChannelOrder::Bgrx) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                strip.b[i] = f32::from(src[i * 4]) * INV;
                strip.g[i] = f32::from(src[i * 4 + 1]) * INV;
                strip.r[i] = f32::from(src[i * 4 + 2]) * INV;
                strip.a[i] = 1.0;
            }
        }
        (SampleType::U8, ChannelOrder::Gray) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                let y = f32::from(src[i]) * INV;
                strip.r[i] = y;
                strip.g[i] = y;
                strip.b[i] = y;
                strip.a[i] = 1.0;
            }
        }
        (SampleType::U8, ChannelOrder::GrayAlpha) => {
            const INV: f32 = 1.0 / 255.0;
            for i in 0..len {
                let y = f32::from(src[i * 2]) * INV;
                strip.r[i] = y;
                strip.g[i] = y;
                strip.b[i] = y;
                strip.a[i] = f32::from(src[i * 2 + 1]) * INV;
            }
        }
        (SampleType::U16, _) => {
            const INV: f32 = 1.0 / 65535.0;
            let ch = layout.channels();
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
        (SampleType::F16, _) => {
            let ch = layout.channels();
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
        (SampleType::F32, _) => {
            let ch = layout.channels();
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
        ChannelOrder::Rgb | ChannelOrder::Oklab | ChannelOrder::Perceptual3 => {
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
        ChannelOrder::Rgba | ChannelOrder::OklabA | ChannelOrder::Perceptual4 => {
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

fn apply_micro_op(op: &MicroOp, strip: &mut StripBuf, len: usize, accel: &Accelerators) {
    match op {
        MicroOp::ExpandNarrow { sample } => {
            let (offset, scale) = match sample {
                SampleType::U8 => (16.0 / 255.0, 255.0 / 219.0),
                _ => (4096.0 / 65535.0, 65535.0 / 56064.0),
            };
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v = (*v - offset) * scale;
                }
            }
        }
        MicroOp::CompressNarrow { sample } => {
            let (offset, scale) = match sample {
                SampleType::U8 => (16.0 / 255.0, 219.0 / 255.0),
                _ => (4096.0 / 65535.0, 56064.0 / 65535.0),
            };
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch {
                    *v = offset + (*v).clamp(0.0, 1.0) * scale;
                }
            }
        }
        MicroOp::Unpremul => {
            for i in 0..len {
                let a = strip.a[i];
                if a > 0.0 {
                    let inv = 1.0 / a;
                    strip.r[i] *= inv;
                    strip.g[i] *= inv;
                    strip.b[i] *= inv;
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
        } => {
            apply_oetf(
                *transfer,
                *pq_inv_scale,
                #[cfg(feature = "hdr-transfers")]
                hlg.as_ref(),
                *preserve_extended,
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
            sc.apply_strip(&mut pixels[..len]);
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
        Transfer::Srgb => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                if !preserve_extended {
                    for v in ch.iter_mut() {
                        *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                    }
                }
                if let Some(f) = accel.srgb_eotf {
                    f(ch);
                } else {
                    srgb_eotf_scalar_slice(ch);
                }
            }
        }
        Transfer::Bt709 => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = bt709_eotf_scalar(x);
                }
            }
        }
        Transfer::Gamma22 => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = gamma22_eotf_scalar(x);
                }
            }
        }
        Transfer::Pq => {
            // PQ flushes RGB NaN to 0.0 while leaving alpha untouched.
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    *v = pq_eotf_scalar(*v) * pq_scale;
                }
            }
        }
        Transfer::Hlg(HlgMode::Scene) => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = hlg_scene_eotf_scalar(x);
                }
            }
        }
        Transfer::Hlg(HlgMode::Display(_)) =>
        {
            #[cfg(feature = "hdr-transfers")]
            if let Some(prepared) = hlg {
                prepared.apply_soa(
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                    len,
                    accel.hlg_simd,
                );
            }
        }
        _ => {}
    }
}

fn apply_oetf(
    transfer: Transfer,
    pq_inv_scale: f32,
    #[cfg(feature = "hdr-transfers")] hlg: Option<&PreparedHlg>,
    preserve_extended: bool,
    strip: &mut StripBuf,
    len: usize,
    accel: &Accelerators,
) {
    match transfer {
        Transfer::Linear | Transfer::Unknown => {}
        Transfer::Srgb => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                if !preserve_extended {
                    for v in ch.iter_mut() {
                        *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                    }
                }
                if let Some(f) = accel.srgb_oetf {
                    f(ch);
                } else {
                    srgb_oetf_scalar_slice(ch);
                }
            }
        }
        Transfer::Bt709 => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = bt709_oetf_scalar(x);
                }
            }
        }
        Transfer::Gamma22 => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = gamma22_oetf_scalar(x);
                }
            }
        }
        Transfer::Pq => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    *v = pq_oetf_scalar(*v * pq_inv_scale);
                }
            }
        }
        Transfer::Hlg(HlgMode::Scene) => {
            for ch in [
                &mut strip.r[..len],
                &mut strip.g[..len],
                &mut strip.b[..len],
            ] {
                for v in ch.iter_mut() {
                    let x = if preserve_extended {
                        *v
                    } else {
                        v.clamp(0.0, 1.0)
                    };
                    *v = hlg_scene_oetf_scalar(x);
                }
            }
        }
        Transfer::Hlg(HlgMode::Display(_)) =>
        {
            #[cfg(feature = "hdr-transfers")]
            if let Some(prepared) = hlg {
                prepared.apply_soa(
                    &mut strip.r[..len],
                    &mut strip.g[..len],
                    &mut strip.b[..len],
                    len,
                    accel.hlg_simd,
                );
            }
        }
        _ => {}
    }
}

#[inline]
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

#[inline]
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

/// Exact ITU-R BT.709 EOTF (odd-symmetric for negative extended values).
#[inline]
pub(crate) fn bt709_eotf_scalar(v: f32) -> f32 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA_LIN: f64 = 0.081_242_858_298_631_5; // 4.5 * 0.018053968510807
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a < BETA_LIN {
        a / 4.5
    } else {
        libm::pow((a + (ALPHA - 1.0)) / ALPHA, 1.0 / 0.45)
    };
    (out as f32).copysign(v)
}

/// Exact ITU-R BT.709 OETF (odd-symmetric for negative extended values).
#[inline]
pub(crate) fn bt709_oetf_scalar(v: f32) -> f32 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA: f64 = 0.018_053_968_510_807;
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a < BETA {
        4.5 * a
    } else {
        ALPHA * libm::pow(a, 0.45) - (ALPHA - 1.0)
    };
    (out as f32).copysign(v)
}

/// Exact Adobe RGB (1998) pure gamma EOTF (`|v|^(563/256)`).
#[inline]
pub(crate) fn gamma22_eotf_scalar(v: f32) -> f32 {
    const EXP: f64 = 563.0 / 256.0;
    if !v.is_finite() {
        return v;
    }
    let out = libm::pow(f64::from(v.abs()), EXP);
    (out as f32).copysign(v)
}

/// Exact Adobe RGB (1998) pure gamma OETF (`|v|^(256/563)`).
#[inline]
pub(crate) fn gamma22_oetf_scalar(v: f32) -> f32 {
    const INV_EXP: f64 = 256.0 / 563.0;
    if !v.is_finite() {
        return v;
    }
    let out = libm::pow(f64::from(v.abs()), INV_EXP);
    (out as f32).copysign(v)
}

/// Exact SMPTE ST 2084 PQ EOTF (`[0, 1] → [0, 1]` where `1.0 = 10,000 cd/m²`; NaN → 0.0).
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

/// Exact SMPTE ST 2084 PQ OETF (`[0, 1]` where `1.0 = 10,000 cd/m²` → `[0, 1]`; NaN → 0.0).
#[inline]
pub(crate) fn pq_oetf_scalar(v: f32) -> f32 {
    const M1: f64 = 1305.0 / 8192.0;
    const M2: f64 = 2523.0 / 32.0;
    const C1: f64 = 107.0 / 128.0;
    const C2: f64 = 2413.0 / 128.0;
    const C3: f64 = 2392.0 / 128.0;
    if v.is_nan() || v <= 0.0 {
        return 0.0;
    }
    let y = f64::from(v.min(1.0));
    let ym = libm::pow(y, M1);
    libm::pow((C1 + C2 * ym) / (1.0 + C3 * ym), M2) as f32
}

/// ARIB STD-B67 / BT.2100 HLG scene-referred inverse OETF (`[0, 1] → [0, 1]` scene light).
#[inline]
pub(crate) fn hlg_scene_eotf_scalar(v: f32) -> f32 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 1.0 - 4.0 * A;
    let c: f64 = 0.5 - A * libm::log(4.0 * A);
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a <= 0.5 {
        (a * a) / 3.0
    } else {
        (libm::exp((a - c) / A) + B) / 12.0
    };
    (out as f32).copysign(v)
}

/// ARIB STD-B67 / BT.2100 HLG scene-referred OETF (`[0, 1]` scene light → `[0, 1]`).
#[inline]
pub(crate) fn hlg_scene_oetf_scalar(v: f32) -> f32 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 1.0 - 4.0 * A;
    let c: f64 = 0.5 - A * libm::log(4.0 * A);
    if !v.is_finite() {
        return v;
    }
    let a = f64::from(v.abs());
    let out = if a <= 1.0 / 12.0 {
        libm::sqrt(3.0 * a)
    } else {
        A * libm::log(12.0 * a - B) + c
    };
    (out as f32).copysign(v)
}

// ---------------------------------------------------------------------------
// Pack: 64-px SoA f32 strip → interleaved destination bytes
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

#[allow(clippy::too_many_arguments)]
fn pack_strip(
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

    match (layout.sample, layout.order) {
        (SampleType::U8, _) => {
            let ch = layout.channels();
            for i in 0..len {
                let ordered = load_ordered(layout.order, strip, i);
                for c in 0..ch {
                    dst[i * ch + c] =
                        quantize_to_u32(ordered[c], 255, depth_policy, base_x + i, row_y) as u8;
                }
            }
        }
        (SampleType::U16, _) => {
            let ch = layout.channels();
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
        (SampleType::F16, _) => {
            let ch = layout.channels();
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
        (SampleType::F32, _) => {
            let ch = layout.channels();
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
        ChannelOrder::Rgb | ChannelOrder::Oklab | ChannelOrder::Perceptual3 => {
            [strip.r[i], strip.g[i], strip.b[i], 1.0]
        }
        ChannelOrder::Bgr => [strip.b[i], strip.g[i], strip.r[i], 1.0],
        ChannelOrder::Rgba | ChannelOrder::OklabA | ChannelOrder::Perceptual4 => {
            [strip.r[i], strip.g[i], strip.b[i], strip.a[i]]
        }
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
    // Check that color meaning agrees (ignoring alpha presence changes handled by layout).
    let colors_agree = match (&src.color, &dst.color) {
        (
            zenpixels::ColorSpace::Rgb {
                primaries: sp,
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            zenpixels::ColorSpace::Rgb {
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
                        )
                    ))
        }
        (
            zenpixels::ColorSpace::Gray {
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            zenpixels::ColorSpace::Rgb {
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
            zenpixels::ColorSpace::Gray {
                transfer: st,
                range: sr,
                alpha: sa,
                white: sw,
            },
            zenpixels::ColorSpace::Gray {
                transfer: dt,
                range: dr,
                alpha: da,
                white: dw,
            },
        ) => st == dt && sr == dr && sw == dw && sa == da,
        _ => false,
    };
    if !colors_agree {
        return None;
    }

    let _ = policy_depth;
    use ChannelOrder as O;
    use SampleType as S;
    match (
        src.layout.sample,
        src.layout.order,
        dst.layout.sample,
        dst.layout.order,
    ) {
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
        (S::U8, so, S::U16, do_) if so == do_ => Some(ByteFastPath::U8ToU16Exact),
        _ => None,
    }
}
