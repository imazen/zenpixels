//! fast_gamut redesign — proof of concept (sRGB→sRGB RGB only).
//!
//! See `fast_gamut_DESIGN.md` for the full plan. This file holds the
//! wide-tier (f32x16) body for the sRGB→sRGB RGB pipeline as a stake in
//! the ground:
//!
//! - One body, generic over the magetypes Token via `#[magetypes]`.
//! - Expanded for V4x (AVX-512), V4 (AVX-512 native), V3 (AVX2 with
//!   2× 256-bit polyfill), and scalar.
//! - Calls into `linear_srgb::tf::srgb::{srgb_to_linear_x16,
//!   linear_to_srgb_x16}` for the TRC; matrix multiply is inline.
//!
//! NEON / WASM128 narrow body (`#[magetypes(neon, wasm128)]` over
//! f32x4) is the next step.
//!
//! Not yet wired into `convert_f32_rgb_dispatch` — public surface is
//! the concrete `convert_f32_srgb_rgb_v2_wide` entry point so the body
//! can be benchmarked head-to-head against the v1 stamp_trc_kernels
//! output.

use archmage::prelude::*;
use linear_srgb::tf::srgb;
use magetypes::simd::generic::f32x16 as GenericF32x16;

// SIMD body width: 16 pixels per iteration = 48 f32 for RGB.
const PIXELS_PER_CHUNK: usize = 16;
const RGB_CHUNK: usize = PIXELS_PER_CHUNK * 3;

#[magetypes(v4x, v4, v3, scalar)]
fn convert_f32_srgb_rgb_wide_impl(token: Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x16 = GenericF32x16<Token>;

    let m00 = f32x16::splat(token, m[0][0]);
    let m01 = f32x16::splat(token, m[0][1]);
    let m02 = f32x16::splat(token, m[0][2]);
    let m10 = f32x16::splat(token, m[1][0]);
    let m11 = f32x16::splat(token, m[1][1]);
    let m12 = f32x16::splat(token, m[1][2]);
    let m20 = f32x16::splat(token, m[2][0]);
    let m21 = f32x16::splat(token, m[2][1]);
    let m22 = f32x16::splat(token, m[2][2]);

    let chunks = data.len() / RGB_CHUNK;
    for chunk_i in 0..chunks {
        let off = chunk_i * RGB_CHUNK;
        let mut r = [0.0f32; 16];
        let mut g = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        for i in 0..16 {
            r[i] = data[off + i * 3];
            g[i] = data[off + i * 3 + 1];
            b[i] = data[off + i * 3 + 2];
        }
        let rv = f32x16::load(token, &r);
        let gv = f32x16::load(token, &g);
        let bv = f32x16::load(token, &b);

        let rl = srgb::srgb_to_linear_x16(token, rv);
        let gl = srgb::srgb_to_linear_x16(token, gv);
        let bl = srgb::srgb_to_linear_x16(token, bv);

        let nr = rl * m00 + gl * m01 + bl * m02;
        let ng = rl * m10 + gl * m11 + bl * m12;
        let nb = rl * m20 + gl * m21 + bl * m22;

        let or = srgb::linear_to_srgb_x16(token, nr);
        let og = srgb::linear_to_srgb_x16(token, ng);
        let ob = srgb::linear_to_srgb_x16(token, nb);

        let mut ro = [0.0f32; 16];
        let mut go = [0.0f32; 16];
        let mut bo = [0.0f32; 16];
        or.store(&mut ro);
        og.store(&mut go);
        ob.store(&mut bo);
        for i in 0..16 {
            data[off + i * 3] = ro[i];
            data[off + i * 3 + 1] = go[i];
            data[off + i * 3 + 2] = bo[i];
        }
    }

    for pixel in data[chunks * RGB_CHUNK..].chunks_exact_mut(3) {
        let r = linear_srgb::tf::srgb_to_linear(pixel[0]);
        let g = linear_srgb::tf::srgb_to_linear(pixel[1]);
        let b = linear_srgb::tf::srgb_to_linear(pixel[2]);
        let nr = m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b));
        let ng = m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b));
        let nb = m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b));
        pixel[0] = linear_srgb::tf::linear_to_srgb(nr);
        pixel[1] = linear_srgb::tf::linear_to_srgb(ng);
        pixel[2] = linear_srgb::tf::linear_to_srgb(nb);
    }
}

/// Convert sRGB-encoded f32 RGB pixels in-place using the given gamut matrix
/// and the sRGB TRC for both source and destination.
///
/// Wide-tier body (f32x16) dispatched across V4x / V4 / V3 / scalar via
/// `incant!`. Use this for x86_64 hosts; NEON / WASM128 will get a narrow
/// body in a follow-up.
pub fn convert_f32_srgb_rgb_v2_wide(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(
        convert_f32_srgb_rgb_wide_impl(m, data),
        [v4x, v4, v3, scalar]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srgb_identity_matrix() -> [[f32; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn srgb_identity_roundtrip_is_close() {
        // Identity matrix → input pixel goes through srgb_to_linear → identity →
        // linear_to_srgb. Output should be within polynomial accuracy.
        let m = srgb_identity_matrix();
        let mut data = vec![0.0f32; 17 * 3];
        let denom = (data.len() - 1) as f32;
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i as f32 / denom).clamp(0.0, 1.0);
        }
        let original = data.clone();
        convert_f32_srgb_rgb_v2_wide(&m, &mut data);
        for (i, (got, want)) in data.iter().zip(original.iter()).enumerate() {
            let err = (got - want).abs();
            // Accuracy budget: srgb→linear→srgb roundtrip is bounded by
            // ~1 LSB at u16 (~1.5e-5) per linear-srgb's documented bounds.
            assert!(
                err < 5e-5,
                "lane {i}: got {got}, want {want} (err={err:e})",
            );
        }
    }

    #[test]
    fn srgb_identity_zero_pixel_is_zero() {
        let m = srgb_identity_matrix();
        let mut data = vec![0.0f32; 16 * 3];
        convert_f32_srgb_rgb_v2_wide(&m, &mut data);
        for x in data {
            assert!(x.abs() < 1e-6);
        }
    }

    #[test]
    fn srgb_identity_one_pixel_is_one() {
        let m = srgb_identity_matrix();
        let mut data = vec![1.0f32; 16 * 3];
        convert_f32_srgb_rgb_v2_wide(&m, &mut data);
        for x in data {
            assert!((x - 1.0).abs() < 1e-4, "got {x}");
        }
    }

    #[test]
    fn handles_sub_chunk_input() {
        // 5 pixels — exercises the scalar tail path only.
        let m = srgb_identity_matrix();
        let mut data = vec![0.5f32; 5 * 3];
        let original = data.clone();
        convert_f32_srgb_rgb_v2_wide(&m, &mut data);
        for (got, want) in data.iter().zip(original.iter()) {
            assert!((got - want).abs() < 5e-5);
        }
    }

    #[test]
    fn handles_mixed_chunk_and_tail() {
        // 19 pixels — 16 SIMD + 3 scalar.
        let m = srgb_identity_matrix();
        let mut data = vec![0.5f32; 19 * 3];
        let original = data.clone();
        convert_f32_srgb_rgb_v2_wide(&m, &mut data);
        for (got, want) in data.iter().zip(original.iter()) {
            assert!((got - want).abs() < 5e-5);
        }
    }
}
