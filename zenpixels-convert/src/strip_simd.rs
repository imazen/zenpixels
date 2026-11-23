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
