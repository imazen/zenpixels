//! f32 linear → u16 sRGB encode-step probe.
//!
//! Matlut uses a 65536-entry linear-indexed u16 encode LUT (128 KB, L2).
//! linear-srgb now exposes three SIMD-dispatched encode paths:
//!
//! 1. `linear_to_srgb_u16_slice`       — SIMD rational polynomial, exact
//! 2. `linear_to_srgb_u16_slice_fast`  — SIMD sqrt-indexed LUT (128 KB, ±1 LSB)
//! 3. matlut                           — linearly-indexed LUT (128 KB, ±0 in-band)
//!
//! Wisdom #4 claimed "SIMD polynomial beats LUT for large LUTs, u16 encode
//! was 51×" — test that claim in isolation, encode step only, on identical
//! f32 linear input. No matrix step, no decode step.

use linear_srgb::default::{
    linear_to_srgb_u16, linear_to_srgb_u16_slice, linear_to_srgb_u16_slice_fast,
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

fn main() {
    let lut = build_matlut_lut();
    zenbench::run(|suite| {
        for &(label, width) in SIZES {
            let n = width * CHANNELS;
            // Encode-step throughput: count src f32 bytes in + dst u16 bytes out.
            let bytes = (n * 4 + n * 2) as u64;
            let input = make_linear_input(n);
            let lut_arc: &'static [u16; 65536] = Box::leak(lut.clone());

            suite.group(format!("f32→u16 encode  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let src = input.clone();
                let mut dst = vec![0u16; n];
                g.bench("matlut_lut_scalar", move |b| {
                    b.iter(|| {
                        matlut_encode(&src, &mut dst, lut_arc);
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
