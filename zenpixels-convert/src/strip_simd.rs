//! Native-width SIMD AoS↔SoA strip codecs and contiguous depth converters.
//!
//! Uses `f32x8` on x86-64 V3 and `f32x4` on AArch64 NEON / WebAssembly SIMD128
//! (per `size.md`: native vector width per arch, zero nested `fn` inside x86-64
//! `#[target_feature]` bodies).
//!
//! Every routine is **0-bit identical** to the scalar fallback in `strip.rs`
//! (uses `IntConvert::new().trunc().non_negative_finite()` after `+ 0.5` for
//! exact round-half-up parity with `(scaled + 0.5) as u32`, and flushes NaN
//! and `<= 0` to `0.0`).

use archmage::ScalarToken;
use magetypes::numerics::IntConvert;
use zenpixels::{ChannelOrder, SampleLayout, SampleType};

use crate::policy::DepthPolicy;
use crate::strip::StripBuf;

const TRUNC_NONNEG: u32 = IntConvert::new().trunc().non_negative_finite().bits();
const INV_255: f32 = 1.0 / 255.0;
const INV_65535: f32 = 1.0 / 65535.0;

// ---------------------------------------------------------------------------
// 1. SIMD Strip Unpack (interleaved bytes -> 64-px SoA StripBuf)
// ---------------------------------------------------------------------------

macro_rules! unpack_strip_body {
    ($vec:ident, $lanes:literal, $n3:literal, $n4:literal, $token:ident, $layout:ident, $src:ident, $strip:ident, $len:ident) => {{
        let order = $layout.order;
        let is_bgr = matches!(
            order,
            ChannelOrder::Bgr | ChannelOrder::Bgra | ChannelOrder::Bgrx
        );
        let fill_opaque = matches!(
            order,
            ChannelOrder::Rgb
                | ChannelOrder::Bgr
                | ChannelOrder::Rgbx
                | ChannelOrder::Bgrx
                | ChannelOrder::Oklab
                | ChannelOrder::Xyz
                | ChannelOrder::Perceptual3
        );
        match ($layout.sample, $layout.channels()) {
            (SampleType::U8, 3) => {
                let inv = $vec::splat_t($token, INV_255);
                let mut i = 0usize;
                while i + $lanes <= $len {
                    let blk: &[u8; $n3] = (&$src[i * 3..i * 3 + $n3]).try_into().unwrap();
                    let (c0, c1, c2) = $vec::load_rgb_u8_t($token, blk);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i..i + $lanes].copy_from_slice(&(r * inv).to_array());
                    $strip.g[i..i + $lanes].copy_from_slice(&(c1 * inv).to_array());
                    $strip.b[i..i + $lanes].copy_from_slice(&(b * inv).to_array());
                    i += $lanes;
                }
                while i < $len {
                    let (c0, c1, c2) = ($src[i * 3], $src[i * 3 + 1], $src[i * 3 + 2]);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = f32::from(r) * INV_255;
                    $strip.g[i] = f32::from(c1) * INV_255;
                    $strip.b[i] = f32::from(b) * INV_255;
                    i += 1;
                }
                $strip.a[..$len].fill(1.0);
                true
            }
            (SampleType::U8, 4) => {
                let inv = $vec::splat_t($token, INV_255);
                let mut i = 0usize;
                while i + $lanes <= $len {
                    let blk: &[u8; $n4] = (&$src[i * 4..i * 4 + $n4]).try_into().unwrap();
                    let (c0, c1, c2, c3) = $vec::load_rgba_u8_t($token, blk);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i..i + $lanes].copy_from_slice(&(r * inv).to_array());
                    $strip.g[i..i + $lanes].copy_from_slice(&(c1 * inv).to_array());
                    $strip.b[i..i + $lanes].copy_from_slice(&(b * inv).to_array());
                    if !fill_opaque {
                        $strip.a[i..i + $lanes].copy_from_slice(&(c3 * inv).to_array());
                    }
                    i += $lanes;
                }
                while i < $len {
                    let (c0, c1, c2, c3) = (
                        $src[i * 4],
                        $src[i * 4 + 1],
                        $src[i * 4 + 2],
                        $src[i * 4 + 3],
                    );
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = f32::from(r) * INV_255;
                    $strip.g[i] = f32::from(c1) * INV_255;
                    $strip.b[i] = f32::from(b) * INV_255;
                    if !fill_opaque {
                        $strip.a[i] = f32::from(c3) * INV_255;
                    }
                    i += 1;
                }
                if fill_opaque {
                    $strip.a[..$len].fill(1.0);
                }
                true
            }
            (SampleType::U16, 3) => {
                let inv = $vec::splat_t($token, INV_65535);
                let mut i = 0usize;
                if let Ok(u16s) = bytemuck::try_cast_slice::<u8, u16>($src) {
                    while i + $lanes <= $len {
                        let blk: &[u16; $n3] = (&u16s[i * 3..i * 3 + $n3]).try_into().unwrap();
                        let (c0, c1, c2) = $vec::load_rgb_u16_t($token, blk);
                        let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                        $strip.r[i..i + $lanes].copy_from_slice(&(r * inv).to_array());
                        $strip.g[i..i + $lanes].copy_from_slice(&(c1 * inv).to_array());
                        $strip.b[i..i + $lanes].copy_from_slice(&(b * inv).to_array());
                        i += $lanes;
                    }
                } else {
                    let mut tmp = [0u16; $n3];
                    while i + $lanes <= $len {
                        for k in 0..$n3 {
                            let o = (i * 3 + k) * 2;
                            tmp[k] = u16::from_ne_bytes([$src[o], $src[o + 1]]);
                        }
                        let (c0, c1, c2) = $vec::load_rgb_u16_t($token, &tmp);
                        let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                        $strip.r[i..i + $lanes].copy_from_slice(&(r * inv).to_array());
                        $strip.g[i..i + $lanes].copy_from_slice(&(c1 * inv).to_array());
                        $strip.b[i..i + $lanes].copy_from_slice(&(b * inv).to_array());
                        i += $lanes;
                    }
                }
                while i < $len {
                    let o = i * 6;
                    let c0 = u16::from_ne_bytes([$src[o], $src[o + 1]]);
                    let c1 = u16::from_ne_bytes([$src[o + 2], $src[o + 3]]);
                    let c2 = u16::from_ne_bytes([$src[o + 4], $src[o + 5]]);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = f32::from(r) * INV_65535;
                    $strip.g[i] = f32::from(c1) * INV_65535;
                    $strip.b[i] = f32::from(b) * INV_65535;
                    i += 1;
                }
                $strip.a[..$len].fill(1.0);
                true
            }
            (SampleType::U16, 4) => {
                let inv = $vec::splat_t($token, INV_65535);
                let mut i = 0usize;
                if let Ok(u16s) = bytemuck::try_cast_slice::<u8, u16>($src) {
                    while i + $lanes <= $len {
                        let base = i * 4;
                        let v0 = $vec::from_u16_t(
                            $token,
                            (&u16s[base..base + $lanes]).try_into().unwrap(),
                        );
                        let v1 = $vec::from_u16_t(
                            $token,
                            (&u16s[base + $lanes..base + 2 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let v2 = $vec::from_u16_t(
                            $token,
                            (&u16s[base + 2 * $lanes..base + 3 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let v3 = $vec::from_u16_t(
                            $token,
                            (&u16s[base + 3 * $lanes..base + 4 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let [c0, c1, c2, c3] = $vec::deinterleave_4ch([v0, v1, v2, v3]);
                        let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                        $strip.r[i..i + $lanes].copy_from_slice(&(r * inv).to_array());
                        $strip.g[i..i + $lanes].copy_from_slice(&(c1 * inv).to_array());
                        $strip.b[i..i + $lanes].copy_from_slice(&(b * inv).to_array());
                        if !fill_opaque {
                            $strip.a[i..i + $lanes].copy_from_slice(&(c3 * inv).to_array());
                        }
                        i += $lanes;
                    }
                }
                while i < $len {
                    let o = i * 8;
                    let c0 = u16::from_ne_bytes([$src[o], $src[o + 1]]);
                    let c1 = u16::from_ne_bytes([$src[o + 2], $src[o + 3]]);
                    let c2 = u16::from_ne_bytes([$src[o + 4], $src[o + 5]]);
                    let c3 = u16::from_ne_bytes([$src[o + 6], $src[o + 7]]);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = f32::from(r) * INV_65535;
                    $strip.g[i] = f32::from(c1) * INV_65535;
                    $strip.b[i] = f32::from(b) * INV_65535;
                    if !fill_opaque {
                        $strip.a[i] = f32::from(c3) * INV_65535;
                    }
                    i += 1;
                }
                if fill_opaque {
                    $strip.a[..$len].fill(1.0);
                }
                true
            }
            (SampleType::F32, 3) => {
                let mut i = 0usize;
                if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>($src) {
                    while i + $lanes <= $len {
                        let base = i * 3;
                        let v0 = $vec::load_t(
                            $token,
                            (&floats[base..base + $lanes]).try_into().unwrap(),
                        );
                        let v1 = $vec::load_t(
                            $token,
                            (&floats[base + $lanes..base + 2 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let v2 = $vec::load_t(
                            $token,
                            (&floats[base + 2 * $lanes..base + 3 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let [c0, c1, c2] = $vec::deinterleave_3ch([v0, v1, v2]);
                        let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                        $strip.r[i..i + $lanes].copy_from_slice(&r.to_array());
                        $strip.g[i..i + $lanes].copy_from_slice(&c1.to_array());
                        $strip.b[i..i + $lanes].copy_from_slice(&b.to_array());
                        i += $lanes;
                    }
                }
                while i < $len {
                    let o = i * 12;
                    let c0 = f32::from_ne_bytes([$src[o], $src[o + 1], $src[o + 2], $src[o + 3]]);
                    let c1 =
                        f32::from_ne_bytes([$src[o + 4], $src[o + 5], $src[o + 6], $src[o + 7]]);
                    let c2 =
                        f32::from_ne_bytes([$src[o + 8], $src[o + 9], $src[o + 10], $src[o + 11]]);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = r;
                    $strip.g[i] = c1;
                    $strip.b[i] = b;
                    i += 1;
                }
                $strip.a[..$len].fill(1.0);
                true
            }
            (SampleType::F32, 4) => {
                let mut i = 0usize;
                if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>($src) {
                    while i + $lanes <= $len {
                        let base = i * 4;
                        let v0 = $vec::load_t(
                            $token,
                            (&floats[base..base + $lanes]).try_into().unwrap(),
                        );
                        let v1 = $vec::load_t(
                            $token,
                            (&floats[base + $lanes..base + 2 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let v2 = $vec::load_t(
                            $token,
                            (&floats[base + 2 * $lanes..base + 3 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let v3 = $vec::load_t(
                            $token,
                            (&floats[base + 3 * $lanes..base + 4 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        let [c0, c1, c2, c3] = $vec::deinterleave_4ch([v0, v1, v2, v3]);
                        let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                        $strip.r[i..i + $lanes].copy_from_slice(&r.to_array());
                        $strip.g[i..i + $lanes].copy_from_slice(&c1.to_array());
                        $strip.b[i..i + $lanes].copy_from_slice(&b.to_array());
                        if !fill_opaque {
                            $strip.a[i..i + $lanes].copy_from_slice(&c3.to_array());
                        }
                        i += $lanes;
                    }
                }
                while i < $len {
                    let o = i * 16;
                    let c0 = f32::from_ne_bytes([$src[o], $src[o + 1], $src[o + 2], $src[o + 3]]);
                    let c1 =
                        f32::from_ne_bytes([$src[o + 4], $src[o + 5], $src[o + 6], $src[o + 7]]);
                    let c2 =
                        f32::from_ne_bytes([$src[o + 8], $src[o + 9], $src[o + 10], $src[o + 11]]);
                    let c3 = f32::from_ne_bytes([
                        $src[o + 12],
                        $src[o + 13],
                        $src[o + 14],
                        $src[o + 15],
                    ]);
                    let (r, b) = if is_bgr { (c2, c0) } else { (c0, c2) };
                    $strip.r[i] = r;
                    $strip.g[i] = c1;
                    $strip.b[i] = b;
                    if !fill_opaque {
                        $strip.a[i] = c3;
                    }
                    i += 1;
                }
                if fill_opaque {
                    $strip.a[..$len].fill(1.0);
                }
                true
            }
            _ => false,
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn unpack_strip_v3_t(
    token: archmage::X64V3Token,
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
) -> bool {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    unpack_strip_body!(V, 8, 24, 32, token, layout, src, strip, len)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn unpack_strip_neon_t(
    token: archmage::NeonToken,
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
) -> bool {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    unpack_strip_body!(V, 4, 12, 16, token, layout, src, strip, len)
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn unpack_strip_wasm128_t(
    token: archmage::Wasm128Token,
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
) -> bool {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    unpack_strip_body!(V, 4, 12, 16, token, layout, src, strip, len)
}

fn unpack_strip_scalar_t(
    _token: ScalarToken,
    _layout: SampleLayout,
    _src: &[u8],
    _strip: &mut StripBuf,
    _len: usize,
) -> bool {
    false
}

/// SIMD strip unpack for full-packing 3-ch & 4-ch `U8`, `U16`, and `F32` layouts.
/// Returns `false` if the layout should be handled by the universal scalar unpacker.
pub(crate) fn simd_unpack_strip(
    layout: SampleLayout,
    src: &[u8],
    strip: &mut StripBuf,
    len: usize,
) -> bool {
    if !layout.packing.is_full() {
        return false;
    }
    archmage::attuned!(
        unpack_strip(layout, src, strip, len),
        [_v3, _neon, _wasm128, _scalar]
    )
}

// ---------------------------------------------------------------------------
// 2. SIMD Strip Pack (64-px SoA StripBuf -> interleaved bytes)
// ---------------------------------------------------------------------------

#[inline(always)]
fn quant_scalar_u8(v: f32, bias: f32) -> u8 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    ((v.min(1.0) * 255.0 + bias) as u32).min(255) as u8
}

#[inline(always)]
fn quant_scalar_u16(v: f32, bias: f32) -> u16 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    ((v.min(1.0) * 65535.0 + bias) as u32).min(65535) as u16
}

macro_rules! pack_strip_body {
    ($vec:ident, $lanes:literal, $token:ident, $layout:ident, $strip:ident, $dst:ident, $len:ident, $policy:ident) => {{
        let order = $layout.order;
        let is_bgr = matches!(
            order,
            ChannelOrder::Bgr | ChannelOrder::Bgra | ChannelOrder::Bgrx
        );
        let fill_opaque = matches!(order, ChannelOrder::Rgbx | ChannelOrder::Bgrx);
        let bias = match $policy {
            DepthPolicy::Round | DepthPolicy::Forbid => 0.5_f32,
            DepthPolicy::Truncate => 0.0_f32,
            DepthPolicy::OrderedDither => return false,
        };
        let zero = $vec::zero_t($token);
        let one = $vec::splat_t($token, 1.0);
        let bias_v = $vec::splat_t($token, bias);

        match ($layout.sample, $layout.channels()) {
            (SampleType::U8, 3) => {
                let scale = $vec::splat_t($token, 255.0);
                let mut i = 0usize;
                while i + $lanes <= $len {
                    let r = $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                    let g = $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                    let b = $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                    let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                    let q0 = $vec::blend(c0.simd_gt(zero), c0.min_fast(one), zero) * scale + bias_v;
                    let q1 = $vec::blend(g.simd_gt(zero), g.min_fast(one), zero) * scale + bias_v;
                    let q2 = $vec::blend(c2.simd_gt(zero), c2.min_fast(one), zero) * scale + bias_v;
                    let [v0, v1, v2] = $vec::interleave_3ch([q0, q1, q2]);
                    let o = i * 3;
                    $dst[o..o + $lanes].copy_from_slice(&v0.to_u8_with::<TRUNC_NONNEG>());
                    $dst[o + $lanes..o + 2 * $lanes]
                        .copy_from_slice(&v1.to_u8_with::<TRUNC_NONNEG>());
                    $dst[o + 2 * $lanes..o + 3 * $lanes]
                        .copy_from_slice(&v2.to_u8_with::<TRUNC_NONNEG>());
                    i += $lanes;
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    $dst[i * 3] = quant_scalar_u8(c0, bias);
                    $dst[i * 3 + 1] = quant_scalar_u8($strip.g[i], bias);
                    $dst[i * 3 + 2] = quant_scalar_u8(c2, bias);
                    i += 1;
                }
                true
            }
            (SampleType::U8, 4) => {
                let scale = $vec::splat_t($token, 255.0);
                let mut i = 0usize;
                while i + $lanes <= $len {
                    let r = $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                    let g = $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                    let b = $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                    let a = if fill_opaque {
                        one
                    } else {
                        $vec::load_t($token, (&$strip.a[i..i + $lanes]).try_into().unwrap())
                    };
                    let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                    let q0 = $vec::blend(c0.simd_gt(zero), c0.min_fast(one), zero) * scale + bias_v;
                    let q1 = $vec::blend(g.simd_gt(zero), g.min_fast(one), zero) * scale + bias_v;
                    let q2 = $vec::blend(c2.simd_gt(zero), c2.min_fast(one), zero) * scale + bias_v;
                    let q3 = $vec::blend(a.simd_gt(zero), a.min_fast(one), zero) * scale + bias_v;
                    let [v0, v1, v2, v3] = $vec::interleave_4ch([q0, q1, q2, q3]);
                    let o = i * 4;
                    $dst[o..o + $lanes].copy_from_slice(&v0.to_u8_with::<TRUNC_NONNEG>());
                    $dst[o + $lanes..o + 2 * $lanes]
                        .copy_from_slice(&v1.to_u8_with::<TRUNC_NONNEG>());
                    $dst[o + 2 * $lanes..o + 3 * $lanes]
                        .copy_from_slice(&v2.to_u8_with::<TRUNC_NONNEG>());
                    $dst[o + 3 * $lanes..o + 4 * $lanes]
                        .copy_from_slice(&v3.to_u8_with::<TRUNC_NONNEG>());
                    i += $lanes;
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    $dst[i * 4] = quant_scalar_u8(c0, bias);
                    $dst[i * 4 + 1] = quant_scalar_u8($strip.g[i], bias);
                    $dst[i * 4 + 2] = quant_scalar_u8(c2, bias);
                    $dst[i * 4 + 3] = if fill_opaque {
                        255
                    } else {
                        quant_scalar_u8($strip.a[i], bias)
                    };
                    i += 1;
                }
                true
            }
            (SampleType::U16, 3) => {
                let scale = $vec::splat_t($token, 65535.0);
                let mut i = 0usize;
                if let Ok(u16s) = bytemuck::try_cast_slice_mut::<u8, u16>($dst) {
                    while i + $lanes <= $len {
                        let r =
                            $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                        let g =
                            $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                        let b =
                            $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                        let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                        let q0 =
                            $vec::blend(c0.simd_gt(zero), c0.min_fast(one), zero) * scale + bias_v;
                        let q1 =
                            $vec::blend(g.simd_gt(zero), g.min_fast(one), zero) * scale + bias_v;
                        let q2 =
                            $vec::blend(c2.simd_gt(zero), c2.min_fast(one), zero) * scale + bias_v;
                        let [v0, v1, v2] = $vec::interleave_3ch([q0, q1, q2]);
                        let o = i * 3;
                        u16s[o..o + $lanes].copy_from_slice(&v0.to_u16_with::<TRUNC_NONNEG>());
                        u16s[o + $lanes..o + 2 * $lanes]
                            .copy_from_slice(&v1.to_u16_with::<TRUNC_NONNEG>());
                        u16s[o + 2 * $lanes..o + 3 * $lanes]
                            .copy_from_slice(&v2.to_u16_with::<TRUNC_NONNEG>());
                        i += $lanes;
                    }
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    let o = i * 6;
                    $dst[o..o + 2].copy_from_slice(&quant_scalar_u16(c0, bias).to_ne_bytes());
                    $dst[o + 2..o + 4]
                        .copy_from_slice(&quant_scalar_u16($strip.g[i], bias).to_ne_bytes());
                    $dst[o + 4..o + 6].copy_from_slice(&quant_scalar_u16(c2, bias).to_ne_bytes());
                    i += 1;
                }
                true
            }
            (SampleType::U16, 4) => {
                let scale = $vec::splat_t($token, 65535.0);
                let mut i = 0usize;
                if let Ok(u16s) = bytemuck::try_cast_slice_mut::<u8, u16>($dst) {
                    while i + $lanes <= $len {
                        let r =
                            $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                        let g =
                            $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                        let b =
                            $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                        let a = if fill_opaque {
                            one
                        } else {
                            $vec::load_t($token, (&$strip.a[i..i + $lanes]).try_into().unwrap())
                        };
                        let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                        let q0 =
                            $vec::blend(c0.simd_gt(zero), c0.min_fast(one), zero) * scale + bias_v;
                        let q1 =
                            $vec::blend(g.simd_gt(zero), g.min_fast(one), zero) * scale + bias_v;
                        let q2 =
                            $vec::blend(c2.simd_gt(zero), c2.min_fast(one), zero) * scale + bias_v;
                        let q3 =
                            $vec::blend(a.simd_gt(zero), a.min_fast(one), zero) * scale + bias_v;
                        let [v0, v1, v2, v3] = $vec::interleave_4ch([q0, q1, q2, q3]);
                        let o = i * 4;
                        u16s[o..o + $lanes].copy_from_slice(&v0.to_u16_with::<TRUNC_NONNEG>());
                        u16s[o + $lanes..o + 2 * $lanes]
                            .copy_from_slice(&v1.to_u16_with::<TRUNC_NONNEG>());
                        u16s[o + 2 * $lanes..o + 3 * $lanes]
                            .copy_from_slice(&v2.to_u16_with::<TRUNC_NONNEG>());
                        u16s[o + 3 * $lanes..o + 4 * $lanes]
                            .copy_from_slice(&v3.to_u16_with::<TRUNC_NONNEG>());
                        i += $lanes;
                    }
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    let o = i * 8;
                    $dst[o..o + 2].copy_from_slice(&quant_scalar_u16(c0, bias).to_ne_bytes());
                    $dst[o + 2..o + 4]
                        .copy_from_slice(&quant_scalar_u16($strip.g[i], bias).to_ne_bytes());
                    $dst[o + 4..o + 6].copy_from_slice(&quant_scalar_u16(c2, bias).to_ne_bytes());
                    let a_q = if fill_opaque {
                        65535u16
                    } else {
                        quant_scalar_u16($strip.a[i], bias)
                    };
                    $dst[o + 6..o + 8].copy_from_slice(&a_q.to_ne_bytes());
                    i += 1;
                }
                true
            }
            (SampleType::F32, 3) => {
                let mut i = 0usize;
                if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>($dst) {
                    while i + $lanes <= $len {
                        let r =
                            $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                        let g =
                            $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                        let b =
                            $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                        let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                        let [v0, v1, v2] = $vec::interleave_3ch([c0, g, c2]);
                        let o = i * 3;
                        v0.store((&mut floats[o..o + $lanes]).try_into().unwrap());
                        v1.store(
                            (&mut floats[o + $lanes..o + 2 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        v2.store(
                            (&mut floats[o + 2 * $lanes..o + 3 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        i += $lanes;
                    }
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    let o = i * 12;
                    $dst[o..o + 4].copy_from_slice(&c0.to_ne_bytes());
                    $dst[o + 4..o + 8].copy_from_slice(&$strip.g[i].to_ne_bytes());
                    $dst[o + 8..o + 12].copy_from_slice(&c2.to_ne_bytes());
                    i += 1;
                }
                true
            }
            (SampleType::F32, 4) => {
                let mut i = 0usize;
                if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>($dst) {
                    while i + $lanes <= $len {
                        let r =
                            $vec::load_t($token, (&$strip.r[i..i + $lanes]).try_into().unwrap());
                        let g =
                            $vec::load_t($token, (&$strip.g[i..i + $lanes]).try_into().unwrap());
                        let b =
                            $vec::load_t($token, (&$strip.b[i..i + $lanes]).try_into().unwrap());
                        let a = if fill_opaque {
                            one
                        } else {
                            $vec::load_t($token, (&$strip.a[i..i + $lanes]).try_into().unwrap())
                        };
                        let (c0, c2) = if is_bgr { (b, r) } else { (r, b) };
                        let [v0, v1, v2, v3] = $vec::interleave_4ch([c0, g, c2, a]);
                        let o = i * 4;
                        v0.store((&mut floats[o..o + $lanes]).try_into().unwrap());
                        v1.store(
                            (&mut floats[o + $lanes..o + 2 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        v2.store(
                            (&mut floats[o + 2 * $lanes..o + 3 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        v3.store(
                            (&mut floats[o + 3 * $lanes..o + 4 * $lanes])
                                .try_into()
                                .unwrap(),
                        );
                        i += $lanes;
                    }
                }
                while i < $len {
                    let (c0, c2) = if is_bgr {
                        ($strip.b[i], $strip.r[i])
                    } else {
                        ($strip.r[i], $strip.b[i])
                    };
                    let o = i * 16;
                    $dst[o..o + 4].copy_from_slice(&c0.to_ne_bytes());
                    $dst[o + 4..o + 8].copy_from_slice(&$strip.g[i].to_ne_bytes());
                    $dst[o + 8..o + 12].copy_from_slice(&c2.to_ne_bytes());
                    let a_v = if fill_opaque { 1.0_f32 } else { $strip.a[i] };
                    $dst[o + 12..o + 16].copy_from_slice(&a_v.to_ne_bytes());
                    i += 1;
                }
                true
            }
            _ => false,
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn pack_strip_v3_t(
    token: archmage::X64V3Token,
    layout: SampleLayout,
    strip: &StripBuf,
    dst: &mut [u8],
    len: usize,
    policy: DepthPolicy,
) -> bool {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    pack_strip_body!(V, 8, token, layout, strip, dst, len, policy)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn pack_strip_neon_t(
    token: archmage::NeonToken,
    layout: SampleLayout,
    strip: &StripBuf,
    dst: &mut [u8],
    len: usize,
    policy: DepthPolicy,
) -> bool {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    pack_strip_body!(V, 4, token, layout, strip, dst, len, policy)
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn pack_strip_wasm128_t(
    token: archmage::Wasm128Token,
    layout: SampleLayout,
    strip: &StripBuf,
    dst: &mut [u8],
    len: usize,
    policy: DepthPolicy,
) -> bool {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    pack_strip_body!(V, 4, token, layout, strip, dst, len, policy)
}

fn pack_strip_scalar_t(
    _token: ScalarToken,
    _layout: SampleLayout,
    _strip: &StripBuf,
    _dst: &mut [u8],
    _len: usize,
    _policy: DepthPolicy,
) -> bool {
    false
}

/// SIMD strip pack for full-packing 3-ch & 4-ch `U8`, `U16`, and `F32` layouts.
pub(crate) fn simd_pack_strip(
    layout: SampleLayout,
    strip: &StripBuf,
    dst: &mut [u8],
    len: usize,
    policy: DepthPolicy,
) -> bool {
    if !layout.packing.is_full() {
        return false;
    }
    archmage::attuned!(
        pack_strip(layout, strip, dst, len, policy),
        [_v3, _neon, _wasm128, _scalar]
    )
}

// ---------------------------------------------------------------------------
// 3. Direct contiguous F32 <-> U8 / U16 slice depth conversion (no AoS<->SoA!)
// ---------------------------------------------------------------------------

macro_rules! f32_to_u8_body {
    ($vec:ident, $lanes:literal, $token:ident, $src:ident, $dst:ident, $round:ident) => {{
        let zero = $vec::zero_t($token);
        let one = $vec::splat_t($token, 1.0);
        let scale = $vec::splat_t($token, 255.0);
        let bias = if $round { 0.5_f32 } else { 0.0_f32 };
        let bias_v = $vec::splat_t($token, bias);

        let (in_chunks, in_rem) = $src.as_chunks::<$lanes>();
        let (out_chunks, out_rem) = $dst.as_chunks_mut::<$lanes>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = $vec::load_t($token, inp);
            let q = $vec::blend(v.simd_gt(zero), v.min_fast(one), zero) * scale + bias_v;
            *out = q.to_u8_with::<TRUNC_NONNEG>();
        }
        for (&v, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = quant_scalar_u8(v, bias);
        }
    }};
}

macro_rules! u8_to_f32_body {
    ($vec:ident, $lanes:literal, $token:ident, $src:ident, $dst:ident) => {{
        let inv = $vec::splat_t($token, INV_255);
        let (in_chunks, in_rem) = $src.as_chunks::<$lanes>();
        let (out_chunks, out_rem) = $dst.as_chunks_mut::<$lanes>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = $vec::from_u8_t($token, inp) * inv;
            v.store(out);
        }
        for (&b, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = f32::from(b) * INV_255;
        }
    }};
}

macro_rules! f32_to_u16_body {
    ($vec:ident, $lanes:literal, $token:ident, $src:ident, $dst:ident, $round:ident) => {{
        let zero = $vec::zero_t($token);
        let one = $vec::splat_t($token, 1.0);
        let scale = $vec::splat_t($token, 65535.0);
        let bias = if $round { 0.5_f32 } else { 0.0_f32 };
        let bias_v = $vec::splat_t($token, bias);

        let (in_chunks, in_rem) = $src.as_chunks::<$lanes>();
        let (out_chunks, out_rem) = $dst.as_chunks_mut::<$lanes>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = $vec::load_t($token, inp);
            let q = $vec::blend(v.simd_gt(zero), v.min_fast(one), zero) * scale + bias_v;
            *out = q.to_u16_with::<TRUNC_NONNEG>();
        }
        for (&v, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = quant_scalar_u16(v, bias);
        }
    }};
}

macro_rules! u16_to_f32_body {
    ($vec:ident, $lanes:literal, $token:ident, $src:ident, $dst:ident) => {{
        let inv = $vec::splat_t($token, INV_65535);
        let (in_chunks, in_rem) = $src.as_chunks::<$lanes>();
        let (out_chunks, out_rem) = $dst.as_chunks_mut::<$lanes>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = $vec::from_u16_t($token, inp) * inv;
            v.store(out);
        }
        for (&w, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = f32::from(w) * INV_65535;
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn f32_to_u8_v3_t(token: archmage::X64V3Token, src: &[f32], dst: &mut [u8], round: bool) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    f32_to_u8_body!(V, 8, token, src, dst, round);
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn f32_to_u8_neon_t(token: archmage::NeonToken, src: &[f32], dst: &mut [u8], round: bool) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    f32_to_u8_body!(V, 4, token, src, dst, round);
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn f32_to_u8_wasm128_t(token: archmage::Wasm128Token, src: &[f32], dst: &mut [u8], round: bool) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    f32_to_u8_body!(V, 4, token, src, dst, round);
}
fn f32_to_u8_scalar_t(_token: ScalarToken, src: &[f32], dst: &mut [u8], round: bool) {
    let bias = if round { 0.5 } else { 0.0 };
    for (&v, out) in src.iter().zip(dst.iter_mut()) {
        *out = quant_scalar_u8(v, bias);
    }
}
pub(crate) fn simd_f32_to_u8_slice(src: &[f32], dst: &mut [u8], round: bool) {
    archmage::attuned!(f32_to_u8(src, dst, round), [_v3, _neon, _wasm128, _scalar]);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn u8_to_f32_v3_t(token: archmage::X64V3Token, src: &[u8], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    u8_to_f32_body!(V, 8, token, src, dst);
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn u8_to_f32_neon_t(token: archmage::NeonToken, src: &[u8], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    u8_to_f32_body!(V, 4, token, src, dst);
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn u8_to_f32_wasm128_t(token: archmage::Wasm128Token, src: &[u8], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    u8_to_f32_body!(V, 4, token, src, dst);
}
fn u8_to_f32_scalar_t(_token: ScalarToken, src: &[u8], dst: &mut [f32]) {
    for (&b, out) in src.iter().zip(dst.iter_mut()) {
        *out = f32::from(b) * INV_255;
    }
}
pub(crate) fn simd_u8_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    archmage::attuned!(u8_to_f32(src, dst), [_v3, _neon, _wasm128, _scalar]);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn f32_to_u16_v3_t(token: archmage::X64V3Token, src: &[f32], dst: &mut [u16], round: bool) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    f32_to_u16_body!(V, 8, token, src, dst, round);
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn f32_to_u16_neon_t(token: archmage::NeonToken, src: &[f32], dst: &mut [u16], round: bool) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    f32_to_u16_body!(V, 4, token, src, dst, round);
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn f32_to_u16_wasm128_t(token: archmage::Wasm128Token, src: &[f32], dst: &mut [u16], round: bool) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    f32_to_u16_body!(V, 4, token, src, dst, round);
}
fn f32_to_u16_scalar_t(_token: ScalarToken, src: &[f32], dst: &mut [u16], round: bool) {
    let bias = if round { 0.5 } else { 0.0 };
    for (&v, out) in src.iter().zip(dst.iter_mut()) {
        *out = quant_scalar_u16(v, bias);
    }
}
pub(crate) fn simd_f32_to_u16_slice(src: &[f32], dst: &mut [u16], round: bool) {
    archmage::attuned!(f32_to_u16(src, dst, round), [_v3, _neon, _wasm128, _scalar]);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn u16_to_f32_v3_t(token: archmage::X64V3Token, src: &[u16], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    u16_to_f32_body!(V, 8, token, src, dst);
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn u16_to_f32_neon_t(token: archmage::NeonToken, src: &[u16], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    u16_to_f32_body!(V, 4, token, src, dst);
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn u16_to_f32_wasm128_t(token: archmage::Wasm128Token, src: &[u16], dst: &mut [f32]) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    u16_to_f32_body!(V, 4, token, src, dst);
}
fn u16_to_f32_scalar_t(_token: ScalarToken, src: &[u16], dst: &mut [f32]) {
    for (&w, out) in src.iter().zip(dst.iter_mut()) {
        *out = f32::from(w) * INV_65535;
    }
}
pub(crate) fn simd_u16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    archmage::attuned!(u16_to_f32(src, dst), [_v3, _neon, _wasm128, _scalar]);
}

// ---------------------------------------------------------------------------
// 4. SIMD RGBA8 / BGRA8 straight -> premultiplied (0-bit exact round & trunc)
// ---------------------------------------------------------------------------

macro_rules! rgba8_premul_body {
    ($src:ident, $dst:ident, $bias:expr) => {{
        let (sc, _) = $src.as_chunks::<4>();
        let (dc, _) = $dst.as_chunks_mut::<4>();
        let bias: u16 = $bias;
        for (sp, dp) in sc.iter().zip(dc.iter_mut()) {
            let a = u16::from(sp[3]);
            let t0 = u16::from(sp[0]) * a + bias;
            let t1 = u16::from(sp[1]) * a + bias;
            let t2 = u16::from(sp[2]) * a + bias;
            dp[0] = ((t0 + (t0 >> 8)) >> 8) as u8;
            dp[1] = ((t1 + (t1 >> 8)) >> 8) as u8;
            dp[2] = ((t2 + (t2 >> 8)) >> 8) as u8;
            dp[3] = sp[3];
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn rgba8_premul_v3_t(_token: archmage::X64V3Token, src: &[u8], dst: &mut [u8], round: bool) {
    if round {
        rgba8_premul_body!(src, dst, 128);
    } else {
        rgba8_premul_body!(src, dst, 1);
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn rgba8_premul_neon_t(_token: archmage::NeonToken, src: &[u8], dst: &mut [u8], round: bool) {
    if round {
        rgba8_premul_body!(src, dst, 128);
    } else {
        rgba8_premul_body!(src, dst, 1);
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn rgba8_premul_wasm128_t(_token: archmage::Wasm128Token, src: &[u8], dst: &mut [u8], round: bool) {
    if round {
        rgba8_premul_body!(src, dst, 128);
    } else {
        rgba8_premul_body!(src, dst, 1);
    }
}

fn rgba8_premul_scalar_t(_token: ScalarToken, src: &[u8], dst: &mut [u8], round: bool) {
    if round {
        rgba8_premul_body!(src, dst, 128);
    } else {
        rgba8_premul_body!(src, dst, 1);
    }
}

/// Attuned SIMD `RGBA8`/`BGRA8` straight → premultiplied (0-bit identical to scalar!).
pub(crate) fn simd_rgba8_straight_to_premul(src: &[u8], dst: &mut [u8], width: usize, round: bool) {
    let bytes = width * 4;
    let s = &src[..bytes];
    let d = &mut dst[..bytes];
    archmage::attuned!(rgba8_premul(s, d, round), [_v3, _neon, _wasm128, _scalar]);
}

// ---------------------------------------------------------------------------
// 5. Fused single-pass linear F32 -> sRGB U8 (exact polynomial + quantize!)
// ---------------------------------------------------------------------------

#[inline(always)]
fn lin_f32_to_srgb_u8_tail(src: &[f32], dst: &mut [u8], is_rgba: bool, round: bool) {
    let bias = if round { 0.5_f32 } else { 0.0_f32 };
    let mut i = 0usize;
    while i < src.len() {
        let n = (src.len() - i).min(4);
        let mut raw = [0.0f32; 4];
        for k in 0..n {
            let v = src[i + k];
            raw[k] = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        }
        let mut enc = linear_srgb::target_features::x4::linear_to_srgb_extended_scalar(raw);
        if is_rgba && n == 4 {
            enc[3] = raw[3];
        }
        for k in 0..n {
            dst[i + k] = quant_scalar_u8(enc[k], bias);
        }
        i += n;
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn lin_f32_to_srgb_u8_v3_t(
    token: archmage::X64V3Token,
    src: &[f32],
    dst: &mut [u8],
    is_rgba: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    let zero = V::zero_t(token);
    let one = V::splat_t(token, 1.0);
    let scale = V::splat_t(token, 255.0);
    let bias_v = V::splat_t(token, if round { 0.5_f32 } else { 0.0_f32 });
    let alpha_mask = V::from_array_t(token, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).simd_gt(zero);

    let (in_chunks, in_rem) = src.as_chunks::<8>();
    let (out_chunks, out_rem) = dst.as_chunks_mut::<8>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let raw = V::load_t(token, inp);
        let clamped = V::blend(raw.simd_gt(zero), raw.min_fast(one), zero);
        let enc = V::from_array_t(
            token,
            linear_srgb::target_features::x8::linear_to_srgb_extended_v3(clamped.to_array()),
        );
        let vals = if is_rgba {
            V::blend(alpha_mask, clamped, enc)
        } else {
            enc
        };
        let q = V::blend(vals.simd_gt(zero), vals.min_fast(one), zero) * scale + bias_v;
        *out = q.to_u8_with::<TRUNC_NONNEG>();
    }
    if !in_rem.is_empty() {
        lin_f32_to_srgb_u8_tail(in_rem, out_rem, is_rgba, round);
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn lin_f32_to_srgb_u8_neon_t(
    token: archmage::NeonToken,
    src: &[f32],
    dst: &mut [u8],
    is_rgba: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    let zero = V::zero_t(token);
    let one = V::splat_t(token, 1.0);
    let scale = V::splat_t(token, 255.0);
    let bias_v = V::splat_t(token, if round { 0.5_f32 } else { 0.0_f32 });
    let alpha_mask = V::from_array_t(token, [0.0, 0.0, 0.0, 1.0]).simd_gt(zero);

    let (in_chunks, in_rem) = src.as_chunks::<4>();
    let (out_chunks, out_rem) = dst.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let raw = V::load_t(token, inp);
        let clamped = V::blend(raw.simd_gt(zero), raw.min_fast(one), zero);
        let enc = V::from_array_t(
            token,
            linear_srgb::target_features::x4::linear_to_srgb_extended_neon(clamped.to_array()),
        );
        let vals = if is_rgba {
            V::blend(alpha_mask, clamped, enc)
        } else {
            enc
        };
        let q = V::blend(vals.simd_gt(zero), vals.min_fast(one), zero) * scale + bias_v;
        *out = q.to_u8_with::<TRUNC_NONNEG>();
    }
    if !in_rem.is_empty() {
        lin_f32_to_srgb_u8_tail(in_rem, out_rem, is_rgba, round);
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn lin_f32_to_srgb_u8_wasm128_t(
    token: archmage::Wasm128Token,
    src: &[f32],
    dst: &mut [u8],
    is_rgba: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    let zero = V::zero_t(token);
    let one = V::splat_t(token, 1.0);
    let scale = V::splat_t(token, 255.0);
    let bias_v = V::splat_t(token, if round { 0.5_f32 } else { 0.0_f32 });
    let alpha_mask = V::from_array_t(token, [0.0, 0.0, 0.0, 1.0]).simd_gt(zero);

    let (in_chunks, in_rem) = src.as_chunks::<4>();
    let (out_chunks, out_rem) = dst.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let raw = V::load_t(token, inp);
        let clamped = V::blend(raw.simd_gt(zero), raw.min_fast(one), zero);
        let enc = V::from_array_t(
            token,
            linear_srgb::target_features::x4::linear_to_srgb_extended_wasm128(clamped.to_array()),
        );
        let vals = if is_rgba {
            V::blend(alpha_mask, clamped, enc)
        } else {
            enc
        };
        let q = V::blend(vals.simd_gt(zero), vals.min_fast(one), zero) * scale + bias_v;
        *out = q.to_u8_with::<TRUNC_NONNEG>();
    }
    if !in_rem.is_empty() {
        lin_f32_to_srgb_u8_tail(in_rem, out_rem, is_rgba, round);
    }
}

fn lin_f32_to_srgb_u8_scalar_t(
    _token: ScalarToken,
    src: &[f32],
    dst: &mut [u8],
    is_rgba: bool,
    round: bool,
) {
    lin_f32_to_srgb_u8_tail(src, dst, is_rgba, round);
}

/// Fused single-pass SIMD linear F32 → sRGB U8 (exact polynomial + quantization, 0-bit parity!).
pub(crate) fn simd_lin_f32_to_srgb_u8(src: &[f32], dst: &mut [u8], is_rgba: bool, round: bool) {
    archmage::attuned!(
        lin_f32_to_srgb_u8(src, dst, is_rgba, round),
        [_v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// 6. Option A: Fused Exact SIMD U8 sRGB-TRC -> 3x3 Matrix -> sRGB-TRC U8
//    (0-bit identical to SCALAR strip across all 16,777,216 RGB8 triplets!)
// ---------------------------------------------------------------------------

/// Scalar exact reference for Option A (0-bit identical to `exec_strip_row` and SIMD Option A!).
#[allow(clippy::too_many_arguments)]
pub(crate) fn srgb8_matrix_exact_scalar(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    let sch = src_order.channels();
    let dch = dst_order.channels();
    let sr_i = if src_order.is_bgr() { 2 } else { 0 };
    let sb_i = 2 - sr_i;
    let dr_i = if dst_order.is_bgr() { 2 } else { 0 };
    let db_i = 2 - dr_i;
    let bias = if round { 0.5_f32 } else { 0.0_f32 };
    let copy_alpha = sch == 4 && dch == 4 && !fill_opaque;

    let mut i = 0usize;
    while i < width {
        let sp = &src[i * sch..i * sch + sch];
        let dp = &mut dst[i * dch..i * dch + dch];
        let r = linear_srgb::default::srgb_u8_to_linear(sp[sr_i]);
        let g = linear_srgb::default::srgb_u8_to_linear(sp[1]);
        let b = linear_srgb::default::srgb_u8_to_linear(sp[sb_i]);

        // Plain mul+add order matching `strip.rs` & `accel.rs` (`(m0*r + m1*g) + m2*b`).
        let nr = ((m[0][0] * r + m[0][1] * g) + m[0][2] * b).clamp(0.0, 1.0);
        let ng = ((m[1][0] * r + m[1][1] * g) + m[1][2] * b).clamp(0.0, 1.0);
        let nb = ((m[2][0] * r + m[2][1] * g) + m[2][2] * b).clamp(0.0, 1.0);

        let enc =
            linear_srgb::target_features::x4::linear_to_srgb_extended_scalar([nr, ng, nb, 0.0]);
        dp[dr_i] = quant_scalar_u8(enc[0], bias);
        dp[1] = quant_scalar_u8(enc[1], bias);
        dp[db_i] = quant_scalar_u8(enc[2], bias);
        if dch == 4 {
            dp[3] = if copy_alpha { sp[3] } else { 255 };
        }
        i += 1;
    }
}

macro_rules! impl_srgb8_matrix_exact_body {
    ($token:expr, $V:ty, $oetf:path, $LANES:literal, $N3:literal, $N4:literal, $src:expr, $dst:expr, $width:expr, $m:expr, $src_order:expr, $dst_order:expr, $fill_opaque:expr, $round:expr) => {{
        let zero = <$V>::zero_t($token);
        let one = <$V>::splat_t($token, 1.0);
        let scale = <$V>::splat_t($token, 255.0);
        let bias_v = <$V>::splat_t($token, if $round { 0.5_f32 } else { 0.0_f32 });

        let m00 = <$V>::splat_t($token, $m[0][0]);
        let m01 = <$V>::splat_t($token, $m[0][1]);
        let m02 = <$V>::splat_t($token, $m[0][2]);
        let m10 = <$V>::splat_t($token, $m[1][0]);
        let m11 = <$V>::splat_t($token, $m[1][1]);
        let m12 = <$V>::splat_t($token, $m[1][2]);
        let m20 = <$V>::splat_t($token, $m[2][0]);
        let m21 = <$V>::splat_t($token, $m[2][1]);
        let m22 = <$V>::splat_t($token, $m[2][2]);

        let done = if $src_order == ChannelOrder::Rgb && $dst_order == ChannelOrder::Rgb {
            let (in_chunks, _) = $src[..$width * 3].as_chunks::<$N3>();
            let (out_chunks, _) = $dst[..$width * 3].as_chunks_mut::<$N3>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                let mut lr = [0.0f32; $LANES];
                let mut lg = [0.0f32; $LANES];
                let mut lb = [0.0f32; $LANES];
                for k in 0..$LANES {
                    lr[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 3]);
                    lg[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 3 + 1]);
                    lb[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 3 + 2]);
                }
                let vr = <$V>::from_array_t($token, lr);
                let vg = <$V>::from_array_t($token, lg);
                let vb = <$V>::from_array_t($token, lb);

                let nr = (m00 * vr + m01 * vg) + m02 * vb;
                let ng = (m10 * vr + m11 * vg) + m12 * vb;
                let nb = (m20 * vr + m21 * vg) + m22 * vb;

                let cr = <$V>::blend(nr.simd_gt(zero), nr.min_fast(one), zero);
                let cg = <$V>::blend(ng.simd_gt(zero), ng.min_fast(one), zero);
                let cb = <$V>::blend(nb.simd_gt(zero), nb.min_fast(one), zero);

                let er = <$V>::from_array_t($token, $oetf(cr.to_array()));
                let eg = <$V>::from_array_t($token, $oetf(cg.to_array()));
                let eb = <$V>::from_array_t($token, $oetf(cb.to_array()));

                let qr = (<$V>::blend(er.simd_gt(zero), er.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qg = (<$V>::blend(eg.simd_gt(zero), eg.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qb = (<$V>::blend(eb.simd_gt(zero), eb.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();

                for k in 0..$LANES {
                    out[k * 3] = qr[k];
                    out[k * 3 + 1] = qg[k];
                    out[k * 3 + 2] = qb[k];
                }
            }
            in_chunks.len() * $LANES
        } else if $src_order == ChannelOrder::Rgba
            && $dst_order == ChannelOrder::Rgba
            && !$fill_opaque
        {
            let (in_chunks, _) = $src[..$width * 4].as_chunks::<$N4>();
            let (out_chunks, _) = $dst[..$width * 4].as_chunks_mut::<$N4>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                let mut lr = [0.0f32; $LANES];
                let mut lg = [0.0f32; $LANES];
                let mut lb = [0.0f32; $LANES];
                for k in 0..$LANES {
                    lr[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 4]);
                    lg[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 4 + 1]);
                    lb[k] = linear_srgb::default::srgb_u8_to_linear(inp[k * 4 + 2]);
                }
                let vr = <$V>::from_array_t($token, lr);
                let vg = <$V>::from_array_t($token, lg);
                let vb = <$V>::from_array_t($token, lb);

                let nr = (m00 * vr + m01 * vg) + m02 * vb;
                let ng = (m10 * vr + m11 * vg) + m12 * vb;
                let nb = (m20 * vr + m21 * vg) + m22 * vb;

                let cr = <$V>::blend(nr.simd_gt(zero), nr.min_fast(one), zero);
                let cg = <$V>::blend(ng.simd_gt(zero), ng.min_fast(one), zero);
                let cb = <$V>::blend(nb.simd_gt(zero), nb.min_fast(one), zero);

                let er = <$V>::from_array_t($token, $oetf(cr.to_array()));
                let eg = <$V>::from_array_t($token, $oetf(cg.to_array()));
                let eb = <$V>::from_array_t($token, $oetf(cb.to_array()));

                let qr = (<$V>::blend(er.simd_gt(zero), er.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qg = (<$V>::blend(eg.simd_gt(zero), eg.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qb = (<$V>::blend(eb.simd_gt(zero), eb.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();

                for k in 0..$LANES {
                    out[k * 4] = qr[k];
                    out[k * 4 + 1] = qg[k];
                    out[k * 4 + 2] = qb[k];
                    out[k * 4 + 3] = inp[k * 4 + 3];
                }
            }
            in_chunks.len() * $LANES
        } else {
            let sch = $src_order.channels();
            let dch = $dst_order.channels();
            let sr_i = if $src_order.is_bgr() { 2usize } else { 0usize };
            let sb_i = 2 - sr_i;
            let dr_i = if $dst_order.is_bgr() { 2usize } else { 0usize };
            let db_i = 2 - dr_i;
            let copy_alpha = sch == 4 && dch == 4 && !$fill_opaque;

            let chunks = $width / $LANES;
            let mut p = 0usize;
            while p < chunks {
                let base = p * $LANES;
                let mut lr = [0.0f32; $LANES];
                let mut lg = [0.0f32; $LANES];
                let mut lb = [0.0f32; $LANES];
                let mut la = [255u8; $LANES];
                let s_base = base * sch;
                for k in 0..$LANES {
                    let off = s_base + k * sch;
                    lr[k] = linear_srgb::default::srgb_u8_to_linear($src[off + sr_i]);
                    lg[k] = linear_srgb::default::srgb_u8_to_linear($src[off + 1]);
                    lb[k] = linear_srgb::default::srgb_u8_to_linear($src[off + sb_i]);
                    if copy_alpha {
                        la[k] = $src[off + 3];
                    }
                }
                let vr = <$V>::from_array_t($token, lr);
                let vg = <$V>::from_array_t($token, lg);
                let vb = <$V>::from_array_t($token, lb);

                let nr = (m00 * vr + m01 * vg) + m02 * vb;
                let ng = (m10 * vr + m11 * vg) + m12 * vb;
                let nb = (m20 * vr + m21 * vg) + m22 * vb;

                let cr = <$V>::blend(nr.simd_gt(zero), nr.min_fast(one), zero);
                let cg = <$V>::blend(ng.simd_gt(zero), ng.min_fast(one), zero);
                let cb = <$V>::blend(nb.simd_gt(zero), nb.min_fast(one), zero);

                let er = <$V>::from_array_t($token, $oetf(cr.to_array()));
                let eg = <$V>::from_array_t($token, $oetf(cg.to_array()));
                let eb = <$V>::from_array_t($token, $oetf(cb.to_array()));

                let qr = (<$V>::blend(er.simd_gt(zero), er.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qg = (<$V>::blend(eg.simd_gt(zero), eg.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();
                let qb = (<$V>::blend(eb.simd_gt(zero), eb.min_fast(one), zero) * scale + bias_v)
                    .to_u8_with::<TRUNC_NONNEG>();

                let d_base = base * dch;
                for k in 0..$LANES {
                    let off = d_base + k * dch;
                    $dst[off + dr_i] = qr[k];
                    $dst[off + 1] = qg[k];
                    $dst[off + db_i] = qb[k];
                    if dch == 4 {
                        $dst[off + 3] = la[k];
                    }
                }
                p += 1;
            }
            chunks * $LANES
        };
        if done < $width {
            let sch = $src_order.channels();
            let dch = $dst_order.channels();
            srgb8_matrix_exact_scalar(
                &$src[done * sch..],
                &mut $dst[done * dch..],
                $width - done,
                $m,
                $src_order,
                $dst_order,
                $fill_opaque,
                $round,
            );
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_exact_v3_t(
    token: archmage::X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    impl_srgb8_matrix_exact_body!(
        token,
        V,
        linear_srgb::target_features::x8::linear_to_srgb_extended_v3,
        8,
        24,
        32,
        src,
        dst,
        width,
        m,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_exact_neon_t(
    token: archmage::NeonToken,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x4<archmage::NeonToken>;
    impl_srgb8_matrix_exact_body!(
        token,
        V,
        linear_srgb::target_features::x4::linear_to_srgb_extended_neon,
        4,
        12,
        16,
        src,
        dst,
        width,
        m,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_exact_wasm128_t(
    token: archmage::Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type V = magetypes::simd::generic::f32x4<archmage::Wasm128Token>;
    impl_srgb8_matrix_exact_body!(
        token,
        V,
        linear_srgb::target_features::x4::linear_to_srgb_extended_wasm128,
        4,
        12,
        16,
        src,
        dst,
        width,
        m,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_exact_scalar_t(
    _token: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    srgb8_matrix_exact_scalar(src, dst, width, m, src_order, dst_order, fill_opaque, round);
}

/// Option A: Fused exact SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8 (0-bit parity with scalar strip!).
#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_srgb8_matrix_exact(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    m: &[[f32; 3]; 3],
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    archmage::attuned!(
        srgb8_matrix_exact(src, dst, width, m, src_order, dst_order, fill_opaque, round),
        [_v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// 7. Option B: Opt-In Fast Q14 Fixed-Point SIMD U8 sRGB-TRC -> 3x3 -> sRGB U8
//    (Max error <= 1 LSB vs exact float across all 16,777,216 RGB8 triplets;
//     0-bit identical between SCALAR and ALL for the Q14 algorithm;
//     0 bytes static .rodata — lazy 16 KiB L1-resident LUT via OnceBox!)
// ---------------------------------------------------------------------------

const Q14_SHIFT: u32 = 14;
const Q14_ONE: i32 = 1 << Q14_SHIFT; // 16384
const Q14_HALF: i32 = Q14_ONE >> 1; // 8192

fn srgb8_to_q14_lut() -> &'static [i32; 256] {
    static LUT: once_cell::race::OnceBox<[i32; 256]> = once_cell::race::OnceBox::new();
    LUT.get_or_init(|| {
        let mut t = [0i32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let lin = f64::from(linear_srgb::default::srgb_u8_to_linear(i as u8));
            *slot = libm::round(lin * (Q14_ONE as f64)) as i32;
        }
        alloc::boxed::Box::new(t)
    })
}

// Power-of-two table size (32768 B) lets `& 0x7FFF` eliminate ALL bounds checks in
// `#![forbid(unsafe_code)]`; only indices `0..=16384` (16 KiB) are ever accessed in L1D!
const Q14_LUT_MASK: usize = 0x7FFF;

fn q14_to_srgb8_lut(round: bool) -> &'static [u8; 32768] {
    static LUT_ROUND: once_cell::race::OnceBox<[u8; 32768]> = once_cell::race::OnceBox::new();
    static LUT_TRUNC: once_cell::race::OnceBox<[u8; 32768]> = once_cell::race::OnceBox::new();
    let cell = if round { &LUT_ROUND } else { &LUT_TRUNC };
    cell.get_or_init(|| {
        let bias = if round { 0.5_f32 } else { 0.0_f32 };
        let mut t = alloc::boxed::Box::new([255u8; 32768]);
        let inv = 1.0_f32 / (Q14_ONE as f32);
        for (q, slot) in t[..=16384].iter_mut().enumerate() {
            let lin = (q as f32) * inv;
            let enc = linear_srgb::target_features::x4::linear_to_srgb_extended_scalar([
                lin, 0.0, 0.0, 0.0,
            ])[0];
            *slot = quant_scalar_u8(enc, bias);
        }
        t
    })
}

#[inline(always)]
fn quantize_matrix_q14(m: &[[f32; 3]; 3]) -> [[i32; 3]; 3] {
    let s = Q14_ONE as f64;
    [
        [
            libm::round(f64::from(m[0][0]) * s) as i32,
            libm::round(f64::from(m[0][1]) * s) as i32,
            libm::round(f64::from(m[0][2]) * s) as i32,
        ],
        [
            libm::round(f64::from(m[1][0]) * s) as i32,
            libm::round(f64::from(m[1][1]) * s) as i32,
            libm::round(f64::from(m[1][2]) * s) as i32,
        ],
        [
            libm::round(f64::from(m[2][0]) * s) as i32,
            libm::round(f64::from(m[2][1]) * s) as i32,
            libm::round(f64::from(m[2][2]) * s) as i32,
        ],
    ]
}

/// Pre-multiplied Q14 column tables (`r * col0`, `g * col1`, `b * col2`, 12 KiB L1-resident)
/// built once at `RowPlan::build` time. Eliminates ALL multiplications, transposes, and
/// stack scatter-gather from the Option B pixel loop while remaining 0-bit identical to
/// `(mq[row][0]*r + mq[row][1]*g + mq[row][2]*b + Q14_HALF) >> 14`!
#[repr(C, align(16))]
#[derive(Clone, Debug)]
pub(crate) struct Q14MatrixTables {
    pub r: [[i32; 4]; 256],
    pub g: [[i32; 4]; 256],
    pub b: [[i32; 4]; 256],
}

impl Q14MatrixTables {
    pub(crate) fn new(m: &[[f32; 3]; 3], round: bool) -> alloc::sync::Arc<Self> {
        let in_lut = srgb8_to_q14_lut();
        let _ = q14_to_srgb8_lut(round); // pre-warm so convert_row allocates 0 bytes!
        let mq = quantize_matrix_q14(m);
        let mut r = [[0i32; 4]; 256];
        let mut g = [[0i32; 4]; 256];
        let mut b = [[0i32; 4]; 256];
        for i in 0..256 {
            let v = in_lut[i];
            r[i] = [mq[0][0] * v, mq[1][0] * v, mq[2][0] * v, 0];
            g[i] = [mq[0][1] * v, mq[1][1] * v, mq[2][1] * v, 0];
            b[i] = [mq[0][2] * v, mq[1][2] * v, mq[2][2] * v, 0];
        }
        alloc::sync::Arc::new(Self { r, g, b })
    }
}

/// Scalar Q14 fixed-point reference (0-bit identical to SIMD Q14!).
#[allow(clippy::too_many_arguments)]
pub(crate) fn srgb8_matrix_approx_q14_scalar(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    let out_lut = q14_to_srgb8_lut(round);
    let sch = src_order.channels();
    let dch = dst_order.channels();
    let sr_i = if src_order.is_bgr() { 2 } else { 0 };
    let sb_i = 2 - sr_i;
    let dr_i = if dst_order.is_bgr() { 2 } else { 0 };
    let db_i = 2 - dr_i;
    let copy_alpha = sch == 4 && dch == 4 && !fill_opaque;

    for i in 0..width {
        let sp = &src[i * sch..i * sch + sch];
        let dp = &mut dst[i * dch..i * dch + dch];
        let vr = &tables.r[sp[sr_i] as usize];
        let vg = &tables.g[sp[1] as usize];
        let vb = &tables.b[sp[sb_i] as usize];

        let nr = ((vr[0] + vg[0] + vb[0] + Q14_HALF) >> Q14_SHIFT).clamp(0, Q14_ONE);
        let ng = ((vr[1] + vg[1] + vb[1] + Q14_HALF) >> Q14_SHIFT).clamp(0, Q14_ONE);
        let nb = ((vr[2] + vg[2] + vb[2] + Q14_HALF) >> Q14_SHIFT).clamp(0, Q14_ONE);

        dp[dr_i] = out_lut[(nr as usize) & Q14_LUT_MASK];
        dp[1] = out_lut[(ng as usize) & Q14_LUT_MASK];
        dp[db_i] = out_lut[(nb as usize) & Q14_LUT_MASK];
        if dch == 4 {
            dp[3] = if copy_alpha { sp[3] } else { 255 };
        }
    }
}

macro_rules! impl_srgb8_matrix_q14_body {
    ($token:expr, $VI4:ty, $src:expr, $dst:expr, $width:expr, $tables:expr, $src_order:expr, $dst_order:expr, $fill_opaque:expr, $round:expr) => {{
        let out_lut = q14_to_srgb8_lut($round);
        let tr = &$tables.r;
        let tg = &$tables.g;
        let tb = &$tables.b;

        let zero = <$VI4>::zero_t($token);
        let max_q = <$VI4>::splat_t($token, Q14_ONE);
        let half = <$VI4>::splat_t($token, Q14_HALF);

        if $src_order == ChannelOrder::Rgb && $dst_order == ChannelOrder::Rgb {
            let (in_chunks, in_rem) = $src[..$width * 3].as_chunks::<12>();
            let (out_chunks, out_rem) = $dst[..$width * 3].as_chunks_mut::<12>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                let s0 = (<$VI4>::load_t($token, &tr[inp[0] as usize])
                    + <$VI4>::load_t($token, &tg[inp[1] as usize])
                    + <$VI4>::load_t($token, &tb[inp[2] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s1 = (<$VI4>::load_t($token, &tr[inp[3] as usize])
                    + <$VI4>::load_t($token, &tg[inp[4] as usize])
                    + <$VI4>::load_t($token, &tb[inp[5] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s2 = (<$VI4>::load_t($token, &tr[inp[6] as usize])
                    + <$VI4>::load_t($token, &tg[inp[7] as usize])
                    + <$VI4>::load_t($token, &tb[inp[8] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s3 = (<$VI4>::load_t($token, &tr[inp[9] as usize])
                    + <$VI4>::load_t($token, &tg[inp[10] as usize])
                    + <$VI4>::load_t($token, &tb[inp[11] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();

                out[0] = out_lut[(s0[0] as usize) & Q14_LUT_MASK];
                out[1] = out_lut[(s0[1] as usize) & Q14_LUT_MASK];
                out[2] = out_lut[(s0[2] as usize) & Q14_LUT_MASK];
                out[3] = out_lut[(s1[0] as usize) & Q14_LUT_MASK];
                out[4] = out_lut[(s1[1] as usize) & Q14_LUT_MASK];
                out[5] = out_lut[(s1[2] as usize) & Q14_LUT_MASK];
                out[6] = out_lut[(s2[0] as usize) & Q14_LUT_MASK];
                out[7] = out_lut[(s2[1] as usize) & Q14_LUT_MASK];
                out[8] = out_lut[(s2[2] as usize) & Q14_LUT_MASK];
                out[9] = out_lut[(s3[0] as usize) & Q14_LUT_MASK];
                out[10] = out_lut[(s3[1] as usize) & Q14_LUT_MASK];
                out[11] = out_lut[(s3[2] as usize) & Q14_LUT_MASK];
            }
            if !in_rem.is_empty() {
                srgb8_matrix_approx_q14_scalar(
                    in_rem,
                    out_rem,
                    in_rem.len() / 3,
                    $tables,
                    $src_order,
                    $dst_order,
                    $fill_opaque,
                    $round,
                );
            }
        } else if $src_order == ChannelOrder::Rgba
            && $dst_order == ChannelOrder::Rgba
            && !$fill_opaque
        {
            let (in_chunks, in_rem) = $src[..$width * 4].as_chunks::<16>();
            let (out_chunks, out_rem) = $dst[..$width * 4].as_chunks_mut::<16>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                let s0 = (<$VI4>::load_t($token, &tr[inp[0] as usize])
                    + <$VI4>::load_t($token, &tg[inp[1] as usize])
                    + <$VI4>::load_t($token, &tb[inp[2] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s1 = (<$VI4>::load_t($token, &tr[inp[4] as usize])
                    + <$VI4>::load_t($token, &tg[inp[5] as usize])
                    + <$VI4>::load_t($token, &tb[inp[6] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s2 = (<$VI4>::load_t($token, &tr[inp[8] as usize])
                    + <$VI4>::load_t($token, &tg[inp[9] as usize])
                    + <$VI4>::load_t($token, &tb[inp[10] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                let s3 = (<$VI4>::load_t($token, &tr[inp[12] as usize])
                    + <$VI4>::load_t($token, &tg[inp[13] as usize])
                    + <$VI4>::load_t($token, &tb[inp[14] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();

                out[0] = out_lut[(s0[0] as usize) & Q14_LUT_MASK];
                out[1] = out_lut[(s0[1] as usize) & Q14_LUT_MASK];
                out[2] = out_lut[(s0[2] as usize) & Q14_LUT_MASK];
                out[3] = inp[3];
                out[4] = out_lut[(s1[0] as usize) & Q14_LUT_MASK];
                out[5] = out_lut[(s1[1] as usize) & Q14_LUT_MASK];
                out[6] = out_lut[(s1[2] as usize) & Q14_LUT_MASK];
                out[7] = inp[7];
                out[8] = out_lut[(s2[0] as usize) & Q14_LUT_MASK];
                out[9] = out_lut[(s2[1] as usize) & Q14_LUT_MASK];
                out[10] = out_lut[(s2[2] as usize) & Q14_LUT_MASK];
                out[11] = inp[11];
                out[12] = out_lut[(s3[0] as usize) & Q14_LUT_MASK];
                out[13] = out_lut[(s3[1] as usize) & Q14_LUT_MASK];
                out[14] = out_lut[(s3[2] as usize) & Q14_LUT_MASK];
                out[15] = inp[15];
            }
            if !in_rem.is_empty() {
                srgb8_matrix_approx_q14_scalar(
                    in_rem,
                    out_rem,
                    in_rem.len() / 4,
                    $tables,
                    $src_order,
                    $dst_order,
                    $fill_opaque,
                    $round,
                );
            }
        } else {
            let sch = $src_order.channels();
            let dch = $dst_order.channels();
            let sr_i = if $src_order.is_bgr() { 2usize } else { 0usize };
            let sb_i = 2 - sr_i;
            let dr_i = if $dst_order.is_bgr() { 2usize } else { 0usize };
            let db_i = 2 - dr_i;
            let copy_alpha = sch == 4 && dch == 4 && !$fill_opaque;

            for i in 0..$width {
                let sp = &$src[i * sch..i * sch + sch];
                let dp = &mut $dst[i * dch..i * dch + dch];
                let s = (<$VI4>::load_t($token, &tr[sp[sr_i] as usize])
                    + <$VI4>::load_t($token, &tg[sp[1] as usize])
                    + <$VI4>::load_t($token, &tb[sp[sb_i] as usize])
                    + half)
                    .shr_arithmetic::<14>()
                    .max(zero)
                    .min(max_q)
                    .to_array();
                dp[dr_i] = out_lut[(s[0] as usize) & Q14_LUT_MASK];
                dp[1] = out_lut[(s[1] as usize) & Q14_LUT_MASK];
                dp[db_i] = out_lut[(s[2] as usize) & Q14_LUT_MASK];
                if dch == 4 {
                    dp[3] = if copy_alpha { sp[3] } else { 255 };
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_approx_q14_v3_t(
    token: archmage::X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type VI4 = magetypes::simd::generic::i32x4<archmage::X64V3Token>;
    impl_srgb8_matrix_q14_body!(
        token,
        VI4,
        src,
        dst,
        width,
        tables,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_approx_q14_neon_t(
    token: archmage::NeonToken,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type VI4 = magetypes::simd::generic::i32x4<archmage::NeonToken>;
    impl_srgb8_matrix_q14_body!(
        token,
        VI4,
        src,
        dst,
        width,
        tables,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_approx_q14_wasm128_t(
    token: archmage::Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    type VI4 = magetypes::simd::generic::i32x4<archmage::Wasm128Token>;
    impl_srgb8_matrix_q14_body!(
        token,
        VI4,
        src,
        dst,
        width,
        tables,
        src_order,
        dst_order,
        fill_opaque,
        round
    );
}

#[allow(clippy::too_many_arguments)]
fn srgb8_matrix_approx_q14_scalar_t(
    _token: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    srgb8_matrix_approx_q14_scalar(
        src,
        dst,
        width,
        tables,
        src_order,
        dst_order,
        fill_opaque,
        round,
    );
}

/// Option B: Opt-in fast Q14 fixed-point SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8
/// (max error <= 1 LSB vs exact float; 0-bit parity between SCALAR and ALL).
#[allow(clippy::too_many_arguments)]
pub(crate) fn simd_srgb8_matrix_approx_q14(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    tables: &Q14MatrixTables,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    fill_opaque: bool,
    round: bool,
) {
    archmage::attuned!(
        srgb8_matrix_approx_q14(
            src,
            dst,
            width,
            tables,
            src_order,
            dst_order,
            fill_opaque,
            round
        ),
        [_v3, _neon, _wasm128, _scalar]
    );
}
