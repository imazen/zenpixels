//! Depth conversion + PQ + HLG transfer function kernels.

/// sRGB u8 → linear f32 using `linear-srgb` SIMD batch conversion.
pub(super) fn srgb_u8_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_srgb::simd::srgb_u8_to_linear_slice(&src[..count], dstf);
}

/// Linear f32 → sRGB u8 using `linear-srgb` SIMD batch conversion.
pub(super) fn linear_f32_to_srgb_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    linear_srgb::simd::linear_to_srgb_u8_slice(srcf, &mut dst[..count]);
}

/// Naive u8 → f32 (v / 255.0, no transfer function).
pub(super) fn naive_u8_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = src[i] as f32 / 255.0;
    }
}

/// Naive f32 → u8 (clamp [0,1], * 255 + 0.5).
pub(super) fn naive_f32_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    for i in 0..count {
        dst[i] = (srcf[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// u16 → u8: (v * 255 + 32768) >> 16.
pub(super) fn u16_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    for i in 0..count {
        dst[i] = ((src16[i] as u32 * 255 + 32768) >> 16) as u8;
    }
}

/// u8 → u16: v * 257.
pub(super) fn u8_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        dst16[i] = src[i] as u16 * 257;
    }
}

/// u16 → f32: v / 65535.0.
pub(super) fn u16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = src16[i] as f32 / 65535.0;
    }
}

/// f32 → u16: clamp [0,1], * 65535 + 0.5.
pub(super) fn f32_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        dst16[i] = (srcf[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084) transfer function
// ---------------------------------------------------------------------------

// PQ constants (SMPTE ST 2084 / BT.2100).
const PQ_M1: f64 = 2610.0 / 16384.0; // 0.1593017578125
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0; // 78.84375
const PQ_C1: f64 = 3424.0 / 4096.0; // 0.8359375
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0; // 18.8515625
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0; // 18.6875

/// PQ EOTF: encoded [0,1] → linear light [0,1] (where 1.0 = 10000 cd/m²).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    let vp = v.powf(1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1) as f32
}

/// PQ inverse EOTF (OETF): linear light [0,1] → encoded [0,1].
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    let ym1 = v.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * ym1;
    let den = 1.0 + PQ_C3 * ym1;
    (num / den).powf(PQ_M2) as f32
}

/// PQ U16 → Linear F32 (EOTF applied during depth conversion).
pub(super) fn pq_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        let normalized = src16[i] as f32 / 65535.0;
        dstf[i] = pq_eotf(normalized);
    }
}

/// Linear F32 → PQ U16 (OETF applied during depth conversion).
pub(super) fn linear_f32_to_pq_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        let encoded = pq_oetf(srcf[i].max(0.0));
        dst16[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

/// PQ F32 → Linear F32 (EOTF, same depth).
pub(super) fn pq_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = pq_eotf(srcf[i]);
    }
}

/// Linear F32 → PQ F32 (OETF, same depth).
pub(super) fn linear_f32_to_pq_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = pq_oetf(srcf[i].max(0.0));
    }
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) transfer function
// ---------------------------------------------------------------------------

// HLG constants (ARIB STD-B67 / BT.2100, ITU-R BT.2100-2).
const HLG_A: f64 = 0.17883277;
const HLG_B: f64 = 0.28466892; // 1 - 4*a
const HLG_C: f64 = 0.55991073; // 0.5 - a*ln(4*a)

/// HLG OETF: scene-linear [0,1] → encoded [0,1].
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    let v = v.max(0.0) as f64;
    if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt() as f32
    } else {
        (HLG_A * (12.0 * v - HLG_B).ln() + HLG_C) as f32
    }
}

/// HLG inverse OETF (EOTF): encoded [0,1] → scene-linear [0,1].
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    if v <= 0.5 {
        (v * v / 3.0) as f32
    } else {
        (((v - HLG_C) / HLG_A).exp() + HLG_B) as f32 / 12.0
    }
}

/// HLG U16 → Linear F32 (EOTF applied during depth conversion).
pub(super) fn hlg_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        let normalized = src16[i] as f32 / 65535.0;
        dstf[i] = hlg_eotf(normalized);
    }
}

/// Linear F32 → HLG U16 (OETF applied during depth conversion).
pub(super) fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        let encoded = hlg_oetf(srcf[i]);
        dst16[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

/// HLG F32 → Linear F32 (EOTF, same depth).
pub(super) fn hlg_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = hlg_eotf(srcf[i]);
    }
}

/// Linear F32 → HLG F32 (OETF, same depth).
pub(super) fn linear_f32_to_hlg_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = hlg_oetf(srcf[i]);
    }
}
