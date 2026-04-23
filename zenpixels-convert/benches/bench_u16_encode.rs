//! f32 linear ↔ u16 sRGB encode/decode probe.
//!
//! Matlut uses a 65536-entry linear-indexed u16 encode LUT (128 KB, L2)
//! and a 65536-entry u16→f32 decode LUT (256 KB, L2). linear-srgb now
//! exposes SIMD-dispatched paths for both directions:
//!
//! Encode (f32 → u16):
//!   1. `linear_to_srgb_u16_slice`       — SIMD rational polynomial, exact
//!   2. `linear_to_srgb_u16_slice_fast`  — SIMD sqrt-indexed LUT (128 KB, ±1 LSB)
//!   3. matlut-style                     — linearly-indexed LUT (128 KB, ±0 in-band)
//!
//! Decode (u16 → f32):
//!   1. `srgb_u16_to_linear_slice`       — NEW: SIMD rational polynomial
//!   2. matlut-style                     — linearly-indexed LUT (256 KB)
//!
//! Together these determine whether a fused polynomial u16 RGB gamut
//! kernel would beat matlut's current LUT-pair pipeline.

use linear_srgb::default::{
    linear_to_srgb_u16, linear_to_srgb_u16_slice, linear_to_srgb_u16_slice_fast,
    srgb_u16_to_linear, srgb_u16_to_linear_slice,
};
use zenbench::prelude::*;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

const CHANNELS: usize = 3;

fn make_linear_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0)
        .collect()
}

/// Rebuild matlut's 65536-entry linearly-indexed encode LUT.
/// Mirrors the `srgb_enc_lut_u16()` construction in zenpixels-convert.
fn build_matlut_lut() -> Box<[u16; 65536]> {
    let mut t: Box<[u16; 65536]> = vec![0u16; 65536]
        .into_boxed_slice()
        .try_into()
        .ok()
        .unwrap();
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = linear_to_srgb_u16(i as f32 / 65535.0);
    }
    t
}

/// Matlut-style encode: SIMD clamp/scale/round + scalar LUT gather — the
/// shape actually used inside the production matlut u16 kernel (see
/// `convert_8px_u16_rgb_matlut` in src/fast_gamut.rs).
///
/// We express the SIMD part as a plain `f32x8`-shaped loop; LLVM with
/// `target-cpu=x86-64-v3` folds the clamp/scale/round into AVX2
/// vmaxps/vminps/vmulps/vaddps/vcvtps2dq just like the production kernel.
fn matlut_encode(input: &[f32], output: &mut [u16], lut: &[u16; 65536]) {
    let (in_chunks, in_rem) = input.as_chunks::<8>();
    let (out_chunks, out_rem) = output.as_chunks_mut::<8>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let mut idx = [0i32; 8];
        for j in 0..8 {
            let v = inp[j].clamp(0.0, 1.0) * 65535.0 + 0.5;
            idx[j] = v as i32;
        }
        for j in 0..8 {
            out[j] = lut[idx[j] as usize & 0xFFFF];
        }
    }
    for (v, slot) in in_rem.iter().zip(out_rem.iter_mut()) {
        let idx = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize;
        *slot = lut[idx & 0xFFFF];
    }
}

/// Rebuild matlut's 65536-entry u16 → f32 decode LUT. Mirrors
/// `srgb_lin_lut_u16()` in zenpixels-convert (256 KB footprint).
fn build_matlut_dec_lut() -> Box<[f32; 65536]> {
    let mut t: Box<[f32; 65536]> = vec![0.0f32; 65536]
        .into_boxed_slice()
        .try_into()
        .ok()
        .unwrap();
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = srgb_u16_to_linear(i as u16);
    }
    t
}

/// Matlut-style decode: scalar per-pixel LUT gather.
fn matlut_decode(input: &[u16], output: &mut [f32], lut: &[f32; 65536]) {
    for (v, slot) in input.iter().zip(output.iter_mut()) {
        *slot = lut[*v as usize];
    }
}

fn make_u16_input(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 65536) as u16)
        .collect()
}

fn main() {
    let enc_lut = build_matlut_lut();
    let dec_lut = build_matlut_dec_lut();
    let enc_lut_static: &'static [u16; 65536] = Box::leak(enc_lut);
    let dec_lut_static: &'static [f32; 65536] = Box::leak(dec_lut);
    zenbench::run(|suite| {
        // ---- Decode side: u16 → f32 ----
        for &(label, width) in SIZES {
            let n = width * CHANNELS;
            let bytes = (n * 2 + n * 4) as u64;
            let input = make_u16_input(n);

            suite.group(format!("u16→f32 decode  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let src = input.clone();
                let mut dst = vec![0.0f32; n];
                g.bench("matlut_lut_scalar", move |b| {
                    b.iter(|| {
                        matlut_decode(&src, &mut dst, dec_lut_static);
                        black_box(());
                    })
                });

                let src = input;
                let mut dst = vec![0.0f32; n];
                g.bench("linsrgb_poly_simd", move |b| {
                    b.iter(|| {
                        srgb_u16_to_linear_slice(&src, &mut dst);
                        black_box(());
                    })
                });
            });
        }

        // ---- Encode side: f32 → u16 ----
        for &(label, width) in SIZES {
            let n = width * CHANNELS;
            let bytes = (n * 4 + n * 2) as u64;
            let input = make_linear_input(n);

            suite.group(format!("f32→u16 encode  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let src = input.clone();
                let mut dst = vec![0u16; n];
                g.bench("matlut_lut_scalar", move |b| {
                    b.iter(|| {
                        matlut_encode(&src, &mut dst, enc_lut_static);
                        black_box(());
                    })
                });

                let src = input.clone();
                let mut dst = vec![0u16; n];
                g.bench("linsrgb_poly_simd", move |b| {
                    b.iter(|| {
                        linear_to_srgb_u16_slice(&src, &mut dst);
                        black_box(());
                    })
                });

                let src = input;
                let mut dst = vec![0u16; n];
                g.bench("linsrgb_sqrtlut_simd", move |b| {
                    b.iter(|| {
                        linear_to_srgb_u16_slice_fast(&src, &mut dst);
                        black_box(());
                    })
                });
            });
        }
    });
}
