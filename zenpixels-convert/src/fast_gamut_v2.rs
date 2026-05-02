//! `fast_gamut_v2` — magetypes-driven replacement for the v1 `stamp_trc_kernels!` family.
//!
//! See `fast_gamut_DESIGN.md` for the full plan. This module provides a
//! drop-in replacement for the v1 fused TRC + 3×3 matrix kernels with three
//! per-pair bodies:
//!
//! - **wide body** (`#[magetypes(v4x, v4, scalar)]`) over `f32x16<T>` —
//!   active on x86_64 hosts with AVX-512 (V4 / V4x) and on the scalar
//!   fallback for any host. Native f32x16 lanes on V4 / V4x; the scalar
//!   variant generates portable scalar code.
//! - **native V3 body** (`#[magetypes(v3)]`) over `f32x8<T>` — active on
//!   x86_64 V3 (AVX2 + FMA) hosts. Matches the v1 `fused_8px_rgb_<name>`
//!   shape: 8 pixels per chunk through native 256-bit ops, avoiding the
//!   register-pressure cost of polyfilling f32x16 to 2× 256-bit on AVX2.
//! - **narrow body** (`#[magetypes(neon, wasm128)]`) over `f32x4<T>` —
//!   active on AArch64 / WASM hosts where the wide polyfill incurs heavy
//!   register pressure.
//!
//! Each TRC pair (e.g. `srgb`, `bt709`, `pq`, `hlg`, `adobe`, `pq_to_srgb`,
//! `bt709_to_srgb`, …) ships an RGB and an RGBA dispatcher that drives the
//! right body via `incant!`.
//!
//! # Layout
//!
//! - `stamp_v2_pair!` — generates wide + narrow bodies + an `incant!`-fronted
//!   public dispatch pair for one (linearize, encode, name) tuple.
//! - `convert_f32_rgb_v2 / convert_f32_rgba_v2` — the public match-on-TRC
//!   entry that v1's `convert_f32_rgb_dispatch` / `convert_f32_rgba_dispatch`
//!   forwards to. Linear→Linear bypasses TRC entirely.
//!
//! # Numerics
//!
//! The wide body's matrix multiply uses `mul_add` chained right-to-left, the
//! same shape as the v1 `mat3x3_x8` helper. Forward TRC calls
//! `linear_srgb::tf::*::*_to_linear_x{4,16}<T>`; encode calls
//! `linear_srgb::tf::*::linear_to_*_x{4,16}<T>`. Adobe (gamma 2.2) goes
//! through `linear_srgb::tf::gamma::{gamma_to_linear,linear_to_gamma}_x{4,16}`,
//! which clamp to `[0,1]` then call `pow_midp` (~9 ULP).
//!
//! Tail pixels (count not divisible by chunk width) take the scalar
//! linearize → matrix → encode path.

use archmage::prelude::*;
use linear_srgb::tf;
use magetypes::simd::generic::{
    f32x4 as GenericF32x4, f32x8 as GenericF32x8, f32x16 as GenericF32x16,
};

use crate::TransferFunction;

const ADOBE_GAMMA: f32 = 2.19921875; // Adobe RGB spec: 563/256

// =============================================================================
// Shared scalar helpers
// =============================================================================

#[inline(always)]
fn mat3x3(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b)),
        m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b)),
        m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b)),
    )
}

#[inline(always)]
fn adobe_to_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::gamma_to_linear(v, ADOBE_GAMMA)
}

#[inline(always)]
fn adobe_from_linear_scalar(v: f32) -> f32 {
    linear_srgb::default::linear_to_gamma(v, ADOBE_GAMMA)
}

// =============================================================================
// Macro — stamps wide + native V3 + narrow bodies + dispatchers for one TRC pair.
//
// Inputs:
//   $name        — pair tag (e.g. `srgb`, `pq_to_srgb`, `adobe`)
//   $lin_x16     — generic SIMD linearize: fn<T: F32x16Convert>(t, v) -> v
//   $lin_x8      — generic SIMD linearize: fn<T: F32x8Convert>(t, v) -> v
//   $lin_x4      — generic SIMD linearize: fn<T: F32x4Convert>(t, v) -> v
//   $enc_x16     — generic SIMD encode:    fn<T: F32x16Convert>(t, v) -> v
//   $enc_x8      — generic SIMD encode:    fn<T: F32x8Convert>(t, v) -> v
//   $enc_x4      — generic SIMD encode:    fn<T: F32x4Convert>(t, v) -> v
//   $lin_scalar  — scalar fn(f32) -> f32
//   $enc_scalar  — scalar fn(f32) -> f32
//
// Generates (per pair, for both RGB and RGBA):
//
//   convert_f32_rgb_<name>_v2 (matrix, slice)
//   convert_f32_rgba_<name>_v2 (matrix, slice)
//
// plus their internal `_wide_impl_*` / `_native_impl_v3` / `_narrow_impl_*`
// magetypes-stamped bodies.
// =============================================================================

macro_rules! stamp_v2_pair {
    (
        name: $name:ident,
        lin_x16: $lin_x16:expr,
        lin_x8: $lin_x8:expr,
        lin_x4: $lin_x4:expr,
        enc_x16: $enc_x16:expr,
        enc_x8: $enc_x8:expr,
        enc_x4: $enc_x4:expr,
        lin_scalar: $lin_scalar:expr,
        enc_scalar: $enc_scalar:expr,
    ) => {
        paste::paste! {
            // -----------------------------------------------------------------
            // Wide body: f32x16, dispatched across V4x / V4 / scalar.
            // V3 dropped — handled by the native f32x8 body below to avoid
            // 2x256-bit polyfill register pressure on AVX2.
            // -----------------------------------------------------------------
            #[magetypes(v4x, v4, scalar)]
            fn [<convert_rgb_ $name _wide_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x16 = GenericF32x16<Token>;
                const PIXELS: usize = 16;
                const CHUNK: usize = PIXELS * 3;

                let m00 = f32x16::splat(token, m[0][0]);
                let m01 = f32x16::splat(token, m[0][1]);
                let m02 = f32x16::splat(token, m[0][2]);
                let m10 = f32x16::splat(token, m[1][0]);
                let m11 = f32x16::splat(token, m[1][1]);
                let m12 = f32x16::splat(token, m[1][2]);
                let m20 = f32x16::splat(token, m[2][0]);
                let m21 = f32x16::splat(token, m[2][1]);
                let m22 = f32x16::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    // Fixed-size array pattern: one try_into at chunk start
                    // proves all interior indexes safe (CLAUDE.md "Fixed-size
                    // array pattern eliminates bounds checks").
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let mut r = [0.0f32; PIXELS];
                    let mut g = [0.0f32; PIXELS];
                    let mut b = [0.0f32; PIXELS];
                    for i in 0..PIXELS {
                        r[i] = chunk[i * 3];
                        g[i] = chunk[i * 3 + 1];
                        b[i] = chunk[i * 3 + 2];
                    }
                    let rv = f32x16::load(token, &r);
                    let gv = f32x16::load(token, &g);
                    let bv = f32x16::load(token, &b);

                    let rl = ($lin_x16)(token, rv);
                    let gl = ($lin_x16)(token, gv);
                    let bl = ($lin_x16)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x16)(token, nr);
                    let og_ = ($enc_x16)(token, ng);
                    let ob_ = ($enc_x16)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 3] = ro[i];
                        chunk[i * 3 + 1] = go[i];
                        chunk[i * 3 + 2] = bo[i];
                    }
                }

                for pixel in tail.chunks_exact_mut(3) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                }
            }

            #[magetypes(v4x, v4, scalar)]
            fn [<convert_rgba_ $name _wide_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x16 = GenericF32x16<Token>;
                const PIXELS: usize = 16;
                const CHUNK: usize = PIXELS * 4;

                let m00 = f32x16::splat(token, m[0][0]);
                let m01 = f32x16::splat(token, m[0][1]);
                let m02 = f32x16::splat(token, m[0][2]);
                let m10 = f32x16::splat(token, m[1][0]);
                let m11 = f32x16::splat(token, m[1][1]);
                let m12 = f32x16::splat(token, m[1][2]);
                let m20 = f32x16::splat(token, m[2][0]);
                let m21 = f32x16::splat(token, m[2][1]);
                let m22 = f32x16::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let mut r = [0.0f32; PIXELS];
                    let mut g = [0.0f32; PIXELS];
                    let mut b = [0.0f32; PIXELS];
                    for i in 0..PIXELS {
                        r[i] = chunk[i * 4];
                        g[i] = chunk[i * 4 + 1];
                        b[i] = chunk[i * 4 + 2];
                    }
                    let rv = f32x16::load(token, &r);
                    let gv = f32x16::load(token, &g);
                    let bv = f32x16::load(token, &b);

                    let rl = ($lin_x16)(token, rv);
                    let gl = ($lin_x16)(token, gv);
                    let bl = ($lin_x16)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x16)(token, nr);
                    let og_ = ($enc_x16)(token, ng);
                    let ob_ = ($enc_x16)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 4] = ro[i];
                        chunk[i * 4 + 1] = go[i];
                        chunk[i * 4 + 2] = bo[i];
                        // alpha (chunk[i*4 + 3]) is byte-exact unchanged.
                    }
                }

                for pixel in tail.chunks_exact_mut(4) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                    // pixel[3] unchanged.
                }
            }

            // -----------------------------------------------------------------
            // Native V3 body: f32x8 over X64V3Token, native AVX2 width.
            // Mirrors v1's fused_8px_rgb_<name> / fused_8px_rgba_<name> shape.
            // -----------------------------------------------------------------
            #[magetypes(v3)]
            fn [<convert_rgb_ $name _native_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x8 = GenericF32x8<Token>;
                const PIXELS: usize = 8;
                const CHUNK: usize = PIXELS * 3;

                let m00 = f32x8::splat(token, m[0][0]);
                let m01 = f32x8::splat(token, m[0][1]);
                let m02 = f32x8::splat(token, m[0][2]);
                let m10 = f32x8::splat(token, m[1][0]);
                let m11 = f32x8::splat(token, m[1][1]);
                let m12 = f32x8::splat(token, m[1][2]);
                let m20 = f32x8::splat(token, m[2][0]);
                let m21 = f32x8::splat(token, m[2][1]);
                let m22 = f32x8::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let mut r = [0.0f32; PIXELS];
                    let mut g = [0.0f32; PIXELS];
                    let mut b = [0.0f32; PIXELS];
                    for i in 0..PIXELS {
                        r[i] = chunk[i * 3];
                        g[i] = chunk[i * 3 + 1];
                        b[i] = chunk[i * 3 + 2];
                    }
                    let rv = f32x8::load(token, &r);
                    let gv = f32x8::load(token, &g);
                    let bv = f32x8::load(token, &b);

                    let rl = ($lin_x8)(token, rv);
                    let gl = ($lin_x8)(token, gv);
                    let bl = ($lin_x8)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x8)(token, nr);
                    let og_ = ($enc_x8)(token, ng);
                    let ob_ = ($enc_x8)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 3] = ro[i];
                        chunk[i * 3 + 1] = go[i];
                        chunk[i * 3 + 2] = bo[i];
                    }
                }

                for pixel in tail.chunks_exact_mut(3) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                }
            }

            #[magetypes(v3)]
            fn [<convert_rgba_ $name _native_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x8 = GenericF32x8<Token>;
                const PIXELS: usize = 8;
                const CHUNK: usize = PIXELS * 4;

                let m00 = f32x8::splat(token, m[0][0]);
                let m01 = f32x8::splat(token, m[0][1]);
                let m02 = f32x8::splat(token, m[0][2]);
                let m10 = f32x8::splat(token, m[1][0]);
                let m11 = f32x8::splat(token, m[1][1]);
                let m12 = f32x8::splat(token, m[1][2]);
                let m20 = f32x8::splat(token, m[2][0]);
                let m21 = f32x8::splat(token, m[2][1]);
                let m22 = f32x8::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let mut r = [0.0f32; PIXELS];
                    let mut g = [0.0f32; PIXELS];
                    let mut b = [0.0f32; PIXELS];
                    for i in 0..PIXELS {
                        r[i] = chunk[i * 4];
                        g[i] = chunk[i * 4 + 1];
                        b[i] = chunk[i * 4 + 2];
                    }
                    let rv = f32x8::load(token, &r);
                    let gv = f32x8::load(token, &g);
                    let bv = f32x8::load(token, &b);

                    let rl = ($lin_x8)(token, rv);
                    let gl = ($lin_x8)(token, gv);
                    let bl = ($lin_x8)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x8)(token, nr);
                    let og_ = ($enc_x8)(token, ng);
                    let ob_ = ($enc_x8)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 4] = ro[i];
                        chunk[i * 4 + 1] = go[i];
                        chunk[i * 4 + 2] = bo[i];
                        // alpha unchanged.
                    }
                }

                for pixel in tail.chunks_exact_mut(4) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                }
            }

            // -----------------------------------------------------------------
            // Narrow body: f32x4, dispatched across NEON / WASM128.
            // -----------------------------------------------------------------
            #[magetypes(neon, wasm128)]
            fn [<convert_rgb_ $name _narrow_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x4 = GenericF32x4<Token>;
                const PIXELS: usize = 4;
                const CHUNK: usize = PIXELS * 3;

                let m00 = f32x4::splat(token, m[0][0]);
                let m01 = f32x4::splat(token, m[0][1]);
                let m02 = f32x4::splat(token, m[0][2]);
                let m10 = f32x4::splat(token, m[1][0]);
                let m11 = f32x4::splat(token, m[1][1]);
                let m12 = f32x4::splat(token, m[1][2]);
                let m20 = f32x4::splat(token, m[2][0]);
                let m21 = f32x4::splat(token, m[2][1]);
                let m22 = f32x4::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let r = [chunk[0], chunk[3], chunk[6], chunk[9]];
                    let g = [chunk[1], chunk[4], chunk[7], chunk[10]];
                    let b = [chunk[2], chunk[5], chunk[8], chunk[11]];
                    let rv = f32x4::load(token, &r);
                    let gv = f32x4::load(token, &g);
                    let bv = f32x4::load(token, &b);

                    let rl = ($lin_x4)(token, rv);
                    let gl = ($lin_x4)(token, gv);
                    let bl = ($lin_x4)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x4)(token, nr);
                    let og_ = ($enc_x4)(token, ng);
                    let ob_ = ($enc_x4)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 3] = ro[i];
                        chunk[i * 3 + 1] = go[i];
                        chunk[i * 3 + 2] = bo[i];
                    }
                }

                for pixel in tail.chunks_exact_mut(3) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                }
            }

            #[magetypes(neon, wasm128)]
            fn [<convert_rgba_ $name _narrow_impl>](
                token: Token,
                m: &[[f32; 3]; 3],
                data: &mut [f32],
            ) {
                #[allow(non_camel_case_types)]
                type f32x4 = GenericF32x4<Token>;
                const PIXELS: usize = 4;
                const CHUNK: usize = PIXELS * 4;

                let m00 = f32x4::splat(token, m[0][0]);
                let m01 = f32x4::splat(token, m[0][1]);
                let m02 = f32x4::splat(token, m[0][2]);
                let m10 = f32x4::splat(token, m[1][0]);
                let m11 = f32x4::splat(token, m[1][1]);
                let m12 = f32x4::splat(token, m[1][2]);
                let m20 = f32x4::splat(token, m[2][0]);
                let m21 = f32x4::splat(token, m[2][1]);
                let m22 = f32x4::splat(token, m[2][2]);

                let chunks = data.len() / CHUNK;
                let bulk = chunks * CHUNK;
                let (bulk_data, tail) = data.split_at_mut(bulk);
                for chunk in bulk_data.chunks_exact_mut(CHUNK) {
                    let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
                    let r = [chunk[0], chunk[4], chunk[8], chunk[12]];
                    let g = [chunk[1], chunk[5], chunk[9], chunk[13]];
                    let b = [chunk[2], chunk[6], chunk[10], chunk[14]];
                    let rv = f32x4::load(token, &r);
                    let gv = f32x4::load(token, &g);
                    let bv = f32x4::load(token, &b);

                    let rl = ($lin_x4)(token, rv);
                    let gl = ($lin_x4)(token, gv);
                    let bl = ($lin_x4)(token, bv);

                    let nr = m00.mul_add(rl, m01.mul_add(gl, m02 * bl));
                    let ng = m10.mul_add(rl, m11.mul_add(gl, m12 * bl));
                    let nb = m20.mul_add(rl, m21.mul_add(gl, m22 * bl));

                    let or_ = ($enc_x4)(token, nr);
                    let og_ = ($enc_x4)(token, ng);
                    let ob_ = ($enc_x4)(token, nb);

                    let mut ro = [0.0f32; PIXELS];
                    let mut go = [0.0f32; PIXELS];
                    let mut bo = [0.0f32; PIXELS];
                    or_.store(&mut ro);
                    og_.store(&mut go);
                    ob_.store(&mut bo);
                    for i in 0..PIXELS {
                        chunk[i * 4] = ro[i];
                        chunk[i * 4 + 1] = go[i];
                        chunk[i * 4 + 2] = bo[i];
                    }
                }

                for pixel in tail.chunks_exact_mut(4) {
                    let r = ($lin_scalar)(pixel[0]);
                    let g = ($lin_scalar)(pixel[1]);
                    let b = ($lin_scalar)(pixel[2]);
                    let (nr, ng, nb) = mat3x3(m, r, g, b);
                    pixel[0] = ($enc_scalar)(nr);
                    pixel[1] = ($enc_scalar)(ng);
                    pixel[2] = ($enc_scalar)(nb);
                }
            }

            // -----------------------------------------------------------------
            // Public dispatchers — manual try-tier cascade (Option A).
            //
            // x86_64: V4x → wide_impl_v4x; V4 → wide_impl_v4;
            //         V3 → native_impl_v3 (native f32x8, no AVX2 polyfill);
            //         else → wide_impl_scalar.
            // aarch64 / arm64ec: NEON → narrow_impl_neon, else scalar.
            // wasm32: WASM128 → narrow_impl_wasm128, else scalar.
            // -----------------------------------------------------------------
            /// Convert RGB f32 in-place via the sRGB/{TRC} pipeline for the
            /// `$name` pair. `data.len()` must be a multiple of 3.
            pub fn [<convert_f32_rgb_ $name _v2>](m: &[[f32; 3]; 3], data: &mut [f32]) {
                debug_assert_eq!(data.len() % 3, 0);
                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(feature = "avx512")]
                    {
                        if let Some(t) = X64V4xToken::summon() {
                            return [<convert_rgb_ $name _wide_impl_v4x>](t, m, data);
                        }
                        if let Some(t) = X64V4Token::summon() {
                            return [<convert_rgb_ $name _wide_impl_v4>](t, m, data);
                        }
                    }
                    if let Some(t) = X64V3Token::summon() {
                        return [<convert_rgb_ $name _native_impl_v3>](t, m, data);
                    }
                    return [<convert_rgb_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
                {
                    if let Some(t) = NeonToken::summon() {
                        return [<convert_rgb_ $name _narrow_impl_neon>](t, m, data);
                    }
                    return [<convert_rgb_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(t) = Wasm128Token::summon() {
                        return [<convert_rgb_ $name _narrow_impl_wasm128>](t, m, data);
                    }
                    return [<convert_rgb_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(not(any(
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "arm64ec",
                    target_arch = "wasm32",
                )))]
                {
                    [<convert_rgb_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data)
                }
            }

            /// Convert RGBA f32 in-place via the sRGB/{TRC} pipeline for the
            /// `$name` pair. Alpha is byte-exact unchanged.
            pub fn [<convert_f32_rgba_ $name _v2>](m: &[[f32; 3]; 3], data: &mut [f32]) {
                debug_assert_eq!(data.len() % 4, 0);
                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(feature = "avx512")]
                    {
                        if let Some(t) = X64V4xToken::summon() {
                            return [<convert_rgba_ $name _wide_impl_v4x>](t, m, data);
                        }
                        if let Some(t) = X64V4Token::summon() {
                            return [<convert_rgba_ $name _wide_impl_v4>](t, m, data);
                        }
                    }
                    if let Some(t) = X64V3Token::summon() {
                        return [<convert_rgba_ $name _native_impl_v3>](t, m, data);
                    }
                    return [<convert_rgba_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
                {
                    if let Some(t) = NeonToken::summon() {
                        return [<convert_rgba_ $name _narrow_impl_neon>](t, m, data);
                    }
                    return [<convert_rgba_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(t) = Wasm128Token::summon() {
                        return [<convert_rgba_ $name _narrow_impl_wasm128>](t, m, data);
                    }
                    return [<convert_rgba_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data);
                }
                #[cfg(not(any(
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "arm64ec",
                    target_arch = "wasm32",
                )))]
                {
                    [<convert_rgba_ $name _wide_impl_scalar>](ScalarToken::summon().unwrap(), m, data)
                }
            }
        }
    };
}

// =============================================================================
// TRC pair stamps
// =============================================================================
//
// Each stamp covers one (src TRC, dst TRC) pair, identical lookups for the
// single-TRC pairs. Linear → Linear has no TRC, dispatched directly in
// the public match below.

// --- Same-TRC pairs ---

stamp_v2_pair!(
    name: srgb,
    lin_x16: tf::srgb::srgb_to_linear_x16,
    lin_x8: tf::srgb::srgb_to_linear_x8,
    lin_x4: tf::srgb::srgb_to_linear_x4,
    enc_x16: tf::srgb::linear_to_srgb_x16,
    enc_x8: tf::srgb::linear_to_srgb_x8,
    enc_x4: tf::srgb::linear_to_srgb_x4,
    lin_scalar: tf::srgb_to_linear,
    enc_scalar: tf::linear_to_srgb,
);

stamp_v2_pair!(
    name: bt709,
    lin_x16: tf::bt709::bt709_to_linear_x16,
    lin_x8: tf::bt709::bt709_to_linear_x8,
    lin_x4: tf::bt709::bt709_to_linear_x4,
    enc_x16: tf::bt709::linear_to_bt709_x16,
    enc_x8: tf::bt709::linear_to_bt709_x8,
    enc_x4: tf::bt709::linear_to_bt709_x4,
    lin_scalar: tf::bt709_to_linear,
    enc_scalar: tf::linear_to_bt709,
);

stamp_v2_pair!(
    name: pq,
    lin_x16: tf::pq::pq_to_linear_x16,
    lin_x8: tf::pq::pq_to_linear_x8,
    lin_x4: tf::pq::pq_to_linear_x4,
    enc_x16: tf::pq::linear_to_pq_x16,
    enc_x8: tf::pq::linear_to_pq_x8,
    enc_x4: tf::pq::linear_to_pq_x4,
    lin_scalar: tf::pq_to_linear,
    enc_scalar: tf::linear_to_pq,
);

stamp_v2_pair!(
    name: hlg,
    lin_x16: tf::hlg::hlg_to_linear_x16,
    lin_x8: tf::hlg::hlg_to_linear_x8,
    lin_x4: tf::hlg::hlg_to_linear_x4,
    enc_x16: tf::hlg::linear_to_hlg_x16,
    enc_x8: tf::hlg::linear_to_hlg_x8,
    enc_x4: tf::hlg::linear_to_hlg_x4,
    lin_scalar: tf::hlg_to_linear,
    enc_scalar: tf::linear_to_hlg,
);

// Adobe (gamma 2.2 — pure power, no linear toe). Bind ADOBE_GAMMA via closures.
stamp_v2_pair!(
    name: adobe,
    lin_x16: |t, v| tf::gamma::gamma_to_linear_x16(t, v, ADOBE_GAMMA),
    lin_x8: |t, v| tf::gamma::gamma_to_linear_x8(t, v, ADOBE_GAMMA),
    lin_x4: |t, v| tf::gamma::gamma_to_linear_x4(t, v, ADOBE_GAMMA),
    enc_x16: |t, v| tf::gamma::linear_to_gamma_x16(t, v, ADOBE_GAMMA),
    enc_x8: |t, v| tf::gamma::linear_to_gamma_x8(t, v, ADOBE_GAMMA),
    enc_x4: |t, v| tf::gamma::linear_to_gamma_x4(t, v, ADOBE_GAMMA),
    lin_scalar: adobe_to_linear_scalar,
    enc_scalar: adobe_from_linear_scalar,
);

// --- Cross-TRC pairs ---

stamp_v2_pair!(
    name: pq_to_srgb,
    lin_x16: tf::pq::pq_to_linear_x16,
    lin_x8: tf::pq::pq_to_linear_x8,
    lin_x4: tf::pq::pq_to_linear_x4,
    enc_x16: tf::srgb::linear_to_srgb_x16,
    enc_x8: tf::srgb::linear_to_srgb_x8,
    enc_x4: tf::srgb::linear_to_srgb_x4,
    lin_scalar: tf::pq_to_linear,
    enc_scalar: tf::linear_to_srgb,
);

stamp_v2_pair!(
    name: hlg_to_srgb,
    lin_x16: tf::hlg::hlg_to_linear_x16,
    lin_x8: tf::hlg::hlg_to_linear_x8,
    lin_x4: tf::hlg::hlg_to_linear_x4,
    enc_x16: tf::srgb::linear_to_srgb_x16,
    enc_x8: tf::srgb::linear_to_srgb_x8,
    enc_x4: tf::srgb::linear_to_srgb_x4,
    lin_scalar: tf::hlg_to_linear,
    enc_scalar: tf::linear_to_srgb,
);

stamp_v2_pair!(
    name: srgb_to_pq,
    lin_x16: tf::srgb::srgb_to_linear_x16,
    lin_x8: tf::srgb::srgb_to_linear_x8,
    lin_x4: tf::srgb::srgb_to_linear_x4,
    enc_x16: tf::pq::linear_to_pq_x16,
    enc_x8: tf::pq::linear_to_pq_x8,
    enc_x4: tf::pq::linear_to_pq_x4,
    lin_scalar: tf::srgb_to_linear,
    enc_scalar: tf::linear_to_pq,
);

stamp_v2_pair!(
    name: bt709_to_srgb,
    lin_x16: tf::bt709::bt709_to_linear_x16,
    lin_x8: tf::bt709::bt709_to_linear_x8,
    lin_x4: tf::bt709::bt709_to_linear_x4,
    enc_x16: tf::srgb::linear_to_srgb_x16,
    enc_x8: tf::srgb::linear_to_srgb_x8,
    enc_x4: tf::srgb::linear_to_srgb_x4,
    lin_scalar: tf::bt709_to_linear,
    enc_scalar: tf::linear_to_srgb,
);

stamp_v2_pair!(
    name: srgb_to_bt709,
    lin_x16: tf::srgb::srgb_to_linear_x16,
    lin_x8: tf::srgb::srgb_to_linear_x8,
    lin_x4: tf::srgb::srgb_to_linear_x4,
    enc_x16: tf::bt709::linear_to_bt709_x16,
    enc_x8: tf::bt709::linear_to_bt709_x8,
    enc_x4: tf::bt709::linear_to_bt709_x4,
    lin_scalar: tf::srgb_to_linear,
    enc_scalar: tf::linear_to_bt709,
);

stamp_v2_pair!(
    name: adobe_to_srgb,
    lin_x16: |t, v| tf::gamma::gamma_to_linear_x16(t, v, ADOBE_GAMMA),
    lin_x8: |t, v| tf::gamma::gamma_to_linear_x8(t, v, ADOBE_GAMMA),
    lin_x4: |t, v| tf::gamma::gamma_to_linear_x4(t, v, ADOBE_GAMMA),
    enc_x16: tf::srgb::linear_to_srgb_x16,
    enc_x8: tf::srgb::linear_to_srgb_x8,
    enc_x4: tf::srgb::linear_to_srgb_x4,
    lin_scalar: adobe_to_linear_scalar,
    enc_scalar: tf::linear_to_srgb,
);

stamp_v2_pair!(
    name: srgb_to_adobe,
    lin_x16: tf::srgb::srgb_to_linear_x16,
    lin_x8: tf::srgb::srgb_to_linear_x8,
    lin_x4: tf::srgb::srgb_to_linear_x4,
    enc_x16: |t, v| tf::gamma::linear_to_gamma_x16(t, v, ADOBE_GAMMA),
    enc_x8: |t, v| tf::gamma::linear_to_gamma_x8(t, v, ADOBE_GAMMA),
    enc_x4: |t, v| tf::gamma::linear_to_gamma_x4(t, v, ADOBE_GAMMA),
    lin_scalar: tf::srgb_to_linear,
    enc_scalar: adobe_from_linear_scalar,
);

// =============================================================================
// Linear (identity TRC) — matrix only
// =============================================================================

/// Convert linear f32 RGB pixels in-place using only the 3×3 matrix.
pub fn convert_f32_rgb_linear_v2(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let (nr, ng, nb) = mat3x3(m, pixel[0], pixel[1], pixel[2]);
        pixel[0] = nr;
        pixel[1] = ng;
        pixel[2] = nb;
    }
}

/// Convert linear f32 RGBA pixels in-place. Alpha unchanged.
pub fn convert_f32_rgba_linear_v2(m: &[[f32; 3]; 3], data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    for pixel in data.chunks_exact_mut(4) {
        let (nr, ng, nb) = mat3x3(m, pixel[0], pixel[1], pixel[2]);
        pixel[0] = nr;
        pixel[1] = ng;
        pixel[2] = nb;
    }
}

// =============================================================================
// Public match-on-TRC dispatch
// =============================================================================

/// Convert RGB f32 in-place using the given matrix and TRC pair. Returns
/// `false` if either TRC is unsupported by the v2 surface.
pub fn convert_f32_rgb_v2(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 3, 0);
    match (src_trc, dst_trc) {
        (Linear, Linear) => convert_f32_rgb_linear_v2(m, data),
        (Srgb, Srgb) => convert_f32_rgb_srgb_v2(m, data),
        (Bt709, Bt709) => convert_f32_rgb_bt709_v2(m, data),
        (Pq, Pq) => convert_f32_rgb_pq_v2(m, data),
        (Hlg, Hlg) => convert_f32_rgb_hlg_v2(m, data),
        (Gamma22, Gamma22) => convert_f32_rgb_adobe_v2(m, data),
        (Pq, Srgb) => convert_f32_rgb_pq_to_srgb_v2(m, data),
        (Hlg, Srgb) => convert_f32_rgb_hlg_to_srgb_v2(m, data),
        (Srgb, Pq) => convert_f32_rgb_srgb_to_pq_v2(m, data),
        (Bt709, Srgb) => convert_f32_rgb_bt709_to_srgb_v2(m, data),
        (Srgb, Bt709) => convert_f32_rgb_srgb_to_bt709_v2(m, data),
        (Gamma22, Srgb) => convert_f32_rgb_adobe_to_srgb_v2(m, data),
        (Srgb, Gamma22) => convert_f32_rgb_srgb_to_adobe_v2(m, data),
        _ => return false,
    }
    true
}

/// Convert RGBA f32 in-place. Alpha unchanged.
pub fn convert_f32_rgba_v2(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;
    debug_assert_eq!(data.len() % 4, 0);
    match (src_trc, dst_trc) {
        (Linear, Linear) => convert_f32_rgba_linear_v2(m, data),
        (Srgb, Srgb) => convert_f32_rgba_srgb_v2(m, data),
        (Bt709, Bt709) => convert_f32_rgba_bt709_v2(m, data),
        (Pq, Pq) => convert_f32_rgba_pq_v2(m, data),
        (Hlg, Hlg) => convert_f32_rgba_hlg_v2(m, data),
        (Gamma22, Gamma22) => convert_f32_rgba_adobe_v2(m, data),
        (Pq, Srgb) => convert_f32_rgba_pq_to_srgb_v2(m, data),
        (Hlg, Srgb) => convert_f32_rgba_hlg_to_srgb_v2(m, data),
        (Srgb, Pq) => convert_f32_rgba_srgb_to_pq_v2(m, data),
        (Bt709, Srgb) => convert_f32_rgba_bt709_to_srgb_v2(m, data),
        (Srgb, Bt709) => convert_f32_rgba_srgb_to_bt709_v2(m, data),
        (Gamma22, Srgb) => convert_f32_rgba_adobe_to_srgb_v2(m, data),
        (Srgb, Gamma22) => convert_f32_rgba_srgb_to_adobe_v2(m, data),
        _ => return false,
    }
    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix() -> [[f32; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// A non-identity gamut matrix used to verify v1↔v2 parity. Real
    /// sRGB↔BT.2020 RGB matrix from the existing builtins (off-diagonals
    /// non-trivial so chroma channels actually mix).
    fn srgb_to_bt2020_matrix() -> [[f32; 3]; 3] {
        [
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ]
    }

    /// Tolerance budget for sRGB-only roundtrips (rational poly, ~5e-7).
    const TOL_SRGB: f32 = 1.5e-5;
    /// Tolerance for TRCs whose inverse uses fast_powf or similar (~3e-5
    /// per linear-srgb's documented bounds; allow 1e-4 to absorb rounding
    /// across the linearize → matrix → encode chain).
    const TOL_FASTPOW: f32 = 1e-4;

    /// All same-TRC pairs and their identity-roundtrip tolerance.
    fn same_trc_pairs() -> &'static [(TransferFunction, f32)] {
        &[
            (TransferFunction::Linear, 1e-6),
            (TransferFunction::Srgb, TOL_SRGB),
            (TransferFunction::Bt709, TOL_FASTPOW),
            (TransferFunction::Pq, TOL_FASTPOW),
            (TransferFunction::Hlg, TOL_FASTPOW),
            (TransferFunction::Gamma22, TOL_FASTPOW),
        ]
    }

    /// All cross-TRC pairs supported by v2.
    fn cross_trc_pairs() -> &'static [(TransferFunction, TransferFunction)] {
        use TransferFunction::*;
        &[
            (Pq, Srgb),
            (Hlg, Srgb),
            (Srgb, Pq),
            (Bt709, Srgb),
            (Srgb, Bt709),
            (Gamma22, Srgb),
            (Srgb, Gamma22),
        ]
    }

    fn ramp(n_pixels: usize, channels: usize) -> Vec<f32> {
        let len = n_pixels * channels;
        let denom = (len.saturating_sub(1)).max(1) as f32;
        (0..len)
            .map(|i| (i as f32 / denom).clamp(0.0, 1.0))
            .collect()
    }

    // ---- Identity-roundtrip tests (same TRC, identity matrix) -----------

    #[test]
    fn identity_roundtrip_same_trc_rgb() {
        for &(trc, tol) in same_trc_pairs() {
            let m = identity_matrix();
            // Use n=23 to exercise wide chunk + tail (16 + 7) on x86_64.
            let original = ramp(23, 3);
            let mut data = original.clone();
            assert!(convert_f32_rgb_v2(&m, &mut data, trc, trc));
            for (i, (got, want)) in data.iter().zip(original.iter()).enumerate() {
                let err = (got - want).abs();
                assert!(
                    err < tol,
                    "RGB identity {trc:?} lane {i}: got {got}, want {want} \
                     (err={err:e}, tol={tol:e})",
                );
            }
        }
    }

    #[test]
    fn identity_roundtrip_same_trc_rgba() {
        for &(trc, tol) in same_trc_pairs() {
            let m = identity_matrix();
            // 23 pixels × 4 channels — alpha at index *4+3.
            let mut data = ramp(23, 4);
            // Stamp arbitrary alpha values to verify byte-exact passthrough.
            for i in 0..23 {
                data[i * 4 + 3] = (i as f32 / 22.0) * 0.7 + 0.1;
            }
            let original = data.clone();
            assert!(convert_f32_rgba_v2(&m, &mut data, trc, trc));
            for i in 0..23 {
                for c in 0..3 {
                    let got = data[i * 4 + c];
                    let want = original[i * 4 + c];
                    let err = (got - want).abs();
                    assert!(
                        err < tol,
                        "RGBA identity {trc:?} pixel {i} ch {c}: \
                         got {got}, want {want} (err={err:e})",
                    );
                }
                // Alpha must be byte-exact identical.
                let alpha = data[i * 4 + 3];
                let alpha_orig = original[i * 4 + 3];
                assert_eq!(
                    alpha.to_bits(),
                    alpha_orig.to_bits(),
                    "RGBA identity {trc:?} pixel {i}: alpha {alpha} != orig {alpha_orig}",
                );
            }
        }
    }

    // ---- Zero / one pixel tests ----------------------------------------

    #[test]
    fn zero_pixel_stays_zero_same_trc() {
        for &(trc, _tol) in same_trc_pairs() {
            let m = identity_matrix();
            // 16 pixels — exactly one wide chunk.
            let mut data = vec![0.0f32; 16 * 3];
            assert!(convert_f32_rgb_v2(&m, &mut data, trc, trc));
            for (i, &x) in data.iter().enumerate() {
                assert!(x.abs() < 1e-6, "zero {trc:?} lane {i}: got {x}");
            }
        }
    }

    #[test]
    fn one_pixel_stays_one_same_trc() {
        for &(trc, _tol) in same_trc_pairs() {
            let m = identity_matrix();
            let mut data = vec![1.0f32; 16 * 3];
            assert!(convert_f32_rgb_v2(&m, &mut data, trc, trc));
            for (i, &x) in data.iter().enumerate() {
                assert!(
                    (x - 1.0).abs() < TOL_FASTPOW,
                    "one {trc:?} lane {i}: got {x}",
                );
            }
        }
    }

    // ---- Sub-chunk / exact-chunk / mixed sizes -------------------------

    #[test]
    fn handles_sub_chunk_and_mixed_sizes_rgb() {
        for &(trc, tol) in same_trc_pairs() {
            let m = identity_matrix();
            for &n_pixels in &[5usize, 7, 13, 16, 17, 19, 23] {
                let original = ramp(n_pixels, 3);
                let mut data = original.clone();
                assert!(convert_f32_rgb_v2(&m, &mut data, trc, trc));
                for (i, (got, want)) in data.iter().zip(original.iter()).enumerate() {
                    let err = (got - want).abs();
                    assert!(
                        err < tol,
                        "size {n_pixels} {trc:?} lane {i}: got {got}, \
                         want {want} (err={err:e})",
                    );
                }
            }
        }
    }

    #[test]
    fn handles_sub_chunk_and_mixed_sizes_rgba() {
        for &(trc, tol) in same_trc_pairs() {
            let m = identity_matrix();
            for &n_pixels in &[5usize, 7, 13, 16, 17, 19, 23] {
                let mut data = ramp(n_pixels, 4);
                let original = data.clone();
                assert!(convert_f32_rgba_v2(&m, &mut data, trc, trc));
                for i in 0..n_pixels {
                    for c in 0..3 {
                        let got = data[i * 4 + c];
                        let want = original[i * 4 + c];
                        let err = (got - want).abs();
                        assert!(
                            err < tol,
                            "RGBA size {n_pixels} {trc:?} px {i} ch {c}: \
                             got {got}, want {want} (err={err:e})",
                        );
                    }
                    assert_eq!(
                        data[i * 4 + 3].to_bits(),
                        original[i * 4 + 3].to_bits(),
                        "alpha mutated at px {i}",
                    );
                }
            }
        }
    }

    // ---- Cross-TRC: roundtrip through linear ---------------------------

    /// Cross-TRC pairs don't roundtrip exactly (you'd need src→dst→src), so
    /// validate that v2 produces the same bytes as a scalar reference using
    /// linear-srgb's scalar TF functions directly.
    fn scalar_lin(trc: TransferFunction) -> fn(f32) -> f32 {
        use TransferFunction::*;
        match trc {
            Srgb => tf::srgb_to_linear,
            Bt709 => tf::bt709_to_linear,
            Pq => tf::pq_to_linear,
            Hlg => tf::hlg_to_linear,
            Gamma22 => adobe_to_linear_scalar,
            Linear => |v| v,
            _ => panic!("unsupported TRC {trc:?}"),
        }
    }

    fn scalar_enc(trc: TransferFunction) -> fn(f32) -> f32 {
        use TransferFunction::*;
        match trc {
            Srgb => tf::linear_to_srgb,
            Bt709 => tf::linear_to_bt709,
            Pq => tf::linear_to_pq,
            Hlg => tf::linear_to_hlg,
            Gamma22 => adobe_from_linear_scalar,
            Linear => |v| v,
            _ => panic!("unsupported TRC {trc:?}"),
        }
    }

    fn scalar_reference_rgb(
        m: &[[f32; 3]; 3],
        data: &mut [f32],
        src: TransferFunction,
        dst: TransferFunction,
    ) {
        let lin = scalar_lin(src);
        let enc = scalar_enc(dst);
        for px in data.chunks_exact_mut(3) {
            let r = lin(px[0]);
            let g = lin(px[1]);
            let b = lin(px[2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            px[0] = enc(nr);
            px[1] = enc(ng);
            px[2] = enc(nb);
        }
    }

    fn scalar_reference_rgba(
        m: &[[f32; 3]; 3],
        data: &mut [f32],
        src: TransferFunction,
        dst: TransferFunction,
    ) {
        let lin = scalar_lin(src);
        let enc = scalar_enc(dst);
        for px in data.chunks_exact_mut(4) {
            let r = lin(px[0]);
            let g = lin(px[1]);
            let b = lin(px[2]);
            let (nr, ng, nb) = mat3x3(m, r, g, b);
            px[0] = enc(nr);
            px[1] = enc(ng);
            px[2] = enc(nb);
        }
    }

    /// SIMD-vs-scalar parity: at most 5e-5 per channel on a representative
    /// non-identity matrix. Tolerance covers the difference between scalar
    /// rational-poly TF and the SIMD `pow_midp` Adobe path.
    const TOL_PARITY: f32 = 5e-5;

    #[test]
    fn parity_with_scalar_same_trc_rgb() {
        let m = srgb_to_bt2020_matrix();
        for &(trc, _tol) in same_trc_pairs() {
            // 19 pixels — wide chunk + tail.
            let original = ramp(19, 3);
            let mut v2_out = original.clone();
            let mut ref_out = original.clone();
            assert!(convert_f32_rgb_v2(&m, &mut v2_out, trc, trc));
            scalar_reference_rgb(&m, &mut ref_out, trc, trc);
            for (i, (a, b)) in v2_out.iter().zip(ref_out.iter()).enumerate() {
                let err = (a - b).abs();
                assert!(
                    err < TOL_PARITY,
                    "parity {trc:?} lane {i}: v2={a}, ref={b} (err={err:e})",
                );
            }
        }
    }

    #[test]
    fn parity_with_scalar_same_trc_rgba() {
        let m = srgb_to_bt2020_matrix();
        for &(trc, _tol) in same_trc_pairs() {
            let original = ramp(19, 4);
            let mut v2_out = original.clone();
            let mut ref_out = original.clone();
            assert!(convert_f32_rgba_v2(&m, &mut v2_out, trc, trc));
            scalar_reference_rgba(&m, &mut ref_out, trc, trc);
            for i in 0..19 {
                for c in 0..3 {
                    let a = v2_out[i * 4 + c];
                    let b = ref_out[i * 4 + c];
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "RGBA parity {trc:?} px {i} ch {c}: v2={a}, ref={b} \
                         (err={err:e})",
                    );
                }
                // Alpha untouched in both paths — matches input verbatim.
                assert_eq!(v2_out[i * 4 + 3].to_bits(), original[i * 4 + 3].to_bits());
            }
        }
    }

    #[test]
    fn parity_with_scalar_cross_trc_rgb() {
        let m = srgb_to_bt2020_matrix();
        for &(src, dst) in cross_trc_pairs() {
            let original = ramp(23, 3);
            let mut v2_out = original.clone();
            let mut ref_out = original.clone();
            assert!(convert_f32_rgb_v2(&m, &mut v2_out, src, dst));
            scalar_reference_rgb(&m, &mut ref_out, src, dst);
            for (i, (a, b)) in v2_out.iter().zip(ref_out.iter()).enumerate() {
                let err = (a - b).abs();
                assert!(
                    err < TOL_PARITY,
                    "parity {src:?}->{dst:?} lane {i}: v2={a}, ref={b} \
                     (err={err:e})",
                );
            }
        }
    }

    #[test]
    fn parity_with_scalar_cross_trc_rgba() {
        let m = srgb_to_bt2020_matrix();
        for &(src, dst) in cross_trc_pairs() {
            let original = ramp(23, 4);
            let mut v2_out = original.clone();
            let mut ref_out = original.clone();
            assert!(convert_f32_rgba_v2(&m, &mut v2_out, src, dst));
            scalar_reference_rgba(&m, &mut ref_out, src, dst);
            for i in 0..23 {
                for c in 0..3 {
                    let a = v2_out[i * 4 + c];
                    let b = ref_out[i * 4 + c];
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "RGBA cross parity {src:?}->{dst:?} px {i} ch {c}: \
                         v2={a}, ref={b} (err={err:e})",
                    );
                }
                assert_eq!(v2_out[i * 4 + 3].to_bits(), original[i * 4 + 3].to_bits());
            }
        }
    }

    // ---- Linear→Linear bypass ------------------------------------------

    #[test]
    fn linear_to_linear_skips_trc_and_just_multiplies() {
        let m = srgb_to_bt2020_matrix();
        let original = ramp(7, 3);
        let mut data = original.clone();
        assert!(convert_f32_rgb_v2(
            &m,
            &mut data,
            TransferFunction::Linear,
            TransferFunction::Linear,
        ));
        for (i, px) in original.chunks_exact(3).enumerate() {
            let (er, eg, eb) = mat3x3(&m, px[0], px[1], px[2]);
            for (c, expected) in [er, eg, eb].iter().enumerate() {
                let got = data[i * 3 + c];
                assert!(
                    (got - expected).abs() < 1e-7,
                    "lin px {i} ch {c}: got {got}, expected {expected}",
                );
            }
        }
    }

    // ---- Unsupported pair returns false --------------------------------

    #[test]
    fn unsupported_pair_returns_false() {
        let m = identity_matrix();
        let mut data = vec![0.5f32; 12];
        // Unsupported pair (PQ → BT.709 not in v2's matrix). v1 doesn't
        // accelerate it either; we expect false.
        assert!(
            !convert_f32_rgb_v2(&m, &mut data, TransferFunction::Pq, TransferFunction::Bt709)
        );
    }

    // ---- Tier permutation: scalar path bit-stable on every Token --------

    /// Verify all six magetypes-stamped tier suffixes give scalar-bit-stable
    /// output on this CPU. Uses archmage::testing::for_each_token_permutation
    /// to disable each tier in turn and confirms results match scalar
    /// reference within TOL_PARITY.
    #[test]
    fn tier_permutation_stable_per_pair() {
        use archmage::testing::{
            for_each_token_permutation, lock_token_testing, CompileTimePolicy,
        };

        let _guard = lock_token_testing();
        let m = srgb_to_bt2020_matrix();
        let original = ramp(19, 3);

        // Compute scalar reference once.
        let mut ref_out = original.clone();
        scalar_reference_rgb(&m, &mut ref_out, TransferFunction::Srgb, TransferFunction::Srgb);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            let mut data = original.clone();
            convert_f32_rgb_srgb_v2(&m, &mut data);
            for (i, (a, b)) in data.iter().zip(ref_out.iter()).enumerate() {
                let err = (a - b).abs();
                assert!(
                    err < TOL_PARITY,
                    "tier perm lane {i}: v2={a}, ref={b} (err={err:e})",
                );
            }
        });
    }

    // ---- Native V3 (f32x8) body parity ---------------------------------
    //
    // The dispatcher routes V3 hosts to `convert_rgb_<name>_native_impl_v3`
    // (f32x8) instead of the wide body's V3 expansion. These tests call
    // the native V3 impl directly with a summoned X64V3Token and verify
    // byte-exact-within-tolerance equivalence to the scalar reference.
    //
    // x86_64-only — V3 token is only summon-able on hosts with AVX2+FMA.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_v3_parity_same_trc_rgb() {
        let Some(token) = archmage::X64V3Token::summon() else {
            // No V3 on this host; skip. (No graceful skip mid-test — this is
            // a host-capability gate visible to the developer running tests.)
            eprintln!("native_v3_parity_same_trc_rgb: V3 not available, host capability gate");
            return;
        };
        let m = srgb_to_bt2020_matrix();
        // 19 pixels — V3 body chunk = 8 px, so this exercises 2 chunks + 3 tail.
        let original = ramp(19, 3);

        macro_rules! check_pair {
            ($impl_fn:ident, $trc:expr) => {{
                let mut got = original.clone();
                $impl_fn(token, &m, &mut got);
                let mut want = original.clone();
                scalar_reference_rgb(&m, &mut want, $trc, $trc);
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "native_v3 RGB {} lane {i}: got={a}, want={b} (err={err:e})",
                        stringify!($trc),
                    );
                }
            }};
        }

        check_pair!(convert_rgb_srgb_native_impl_v3, TransferFunction::Srgb);
        check_pair!(convert_rgb_bt709_native_impl_v3, TransferFunction::Bt709);
        check_pair!(convert_rgb_pq_native_impl_v3, TransferFunction::Pq);
        check_pair!(convert_rgb_hlg_native_impl_v3, TransferFunction::Hlg);
        check_pair!(convert_rgb_adobe_native_impl_v3, TransferFunction::Gamma22);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_v3_parity_same_trc_rgba() {
        let Some(token) = archmage::X64V3Token::summon() else {
            eprintln!("native_v3_parity_same_trc_rgba: V3 not available");
            return;
        };
        let m = srgb_to_bt2020_matrix();
        let mut original = ramp(19, 4);
        // Stamp arbitrary alpha values to verify byte-exact passthrough.
        for i in 0..19 {
            original[i * 4 + 3] = (i as f32 / 18.0) * 0.7 + 0.1;
        }

        macro_rules! check_pair {
            ($impl_fn:ident, $trc:expr) => {{
                let mut got = original.clone();
                $impl_fn(token, &m, &mut got);
                let mut want = original.clone();
                scalar_reference_rgba(&m, &mut want, $trc, $trc);
                for i in 0..19 {
                    for c in 0..3 {
                        let a = got[i * 4 + c];
                        let b = want[i * 4 + c];
                        let err = (a - b).abs();
                        assert!(
                            err < TOL_PARITY,
                            "native_v3 RGBA {} px {i} ch {c}: got={a}, want={b} \
                             (err={err:e})",
                            stringify!($trc),
                        );
                    }
                    // Alpha must be byte-exact identical.
                    assert_eq!(
                        got[i * 4 + 3].to_bits(),
                        original[i * 4 + 3].to_bits(),
                        "native_v3 RGBA {} px {i}: alpha mutated",
                        stringify!($trc),
                    );
                }
            }};
        }

        check_pair!(convert_rgba_srgb_native_impl_v3, TransferFunction::Srgb);
        check_pair!(convert_rgba_bt709_native_impl_v3, TransferFunction::Bt709);
        check_pair!(convert_rgba_pq_native_impl_v3, TransferFunction::Pq);
        check_pair!(convert_rgba_hlg_native_impl_v3, TransferFunction::Hlg);
        check_pair!(convert_rgba_adobe_native_impl_v3, TransferFunction::Gamma22);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_v3_parity_cross_trc_rgb() {
        let Some(token) = archmage::X64V3Token::summon() else {
            eprintln!("native_v3_parity_cross_trc_rgb: V3 not available");
            return;
        };
        let m = srgb_to_bt2020_matrix();
        let original = ramp(23, 3);

        macro_rules! check_pair {
            ($impl_fn:ident, $src:expr, $dst:expr) => {{
                let mut got = original.clone();
                $impl_fn(token, &m, &mut got);
                let mut want = original.clone();
                scalar_reference_rgb(&m, &mut want, $src, $dst);
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "native_v3 cross {}->{} lane {i}: got={a}, want={b} (err={err:e})",
                        stringify!($src), stringify!($dst),
                    );
                }
            }};
        }

        use TransferFunction::*;
        check_pair!(convert_rgb_pq_to_srgb_native_impl_v3, Pq, Srgb);
        check_pair!(convert_rgb_hlg_to_srgb_native_impl_v3, Hlg, Srgb);
        check_pair!(convert_rgb_srgb_to_pq_native_impl_v3, Srgb, Pq);
        check_pair!(convert_rgb_bt709_to_srgb_native_impl_v3, Bt709, Srgb);
        check_pair!(convert_rgb_srgb_to_bt709_native_impl_v3, Srgb, Bt709);
        check_pair!(convert_rgb_adobe_to_srgb_native_impl_v3, Gamma22, Srgb);
        check_pair!(convert_rgb_srgb_to_adobe_native_impl_v3, Srgb, Gamma22);
    }

    /// Force the public dispatcher onto the native V3 path by disabling V4 /
    /// V4x tokens at runtime, then verify the result still matches the scalar
    /// reference. Exercises the dispatcher's V3 branch end-to-end (not just
    /// the impl function in isolation).
    ///
    /// Skips on hosts without V3 (where the dispatcher would fall through to
    /// the scalar wide impl anyway).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatcher_routes_to_native_v3_when_avx512_disabled() {
        use archmage::testing::lock_token_testing;

        let _guard = lock_token_testing();
        if archmage::X64V3Token::summon().is_none() {
            eprintln!("dispatcher_routes_to_native_v3: V3 unavailable, skipping");
            return;
        }

        // Disable V4 / V4x at process scope so the dispatcher must fall
        // through to V3.
        #[cfg(feature = "avx512")]
        {
            let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(true);
            let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);
        }

        let m = srgb_to_bt2020_matrix();
        // 17 pixels — exercises the V3 body's 8-px chunks + 1-px tail.
        let original_rgb = ramp(17, 3);
        let original_rgba = ramp(17, 4);

        for &(trc, _) in same_trc_pairs() {
            // RGB
            let mut got = original_rgb.clone();
            assert!(convert_f32_rgb_v2(&m, &mut got, trc, trc));
            let mut want = original_rgb.clone();
            scalar_reference_rgb(&m, &mut want, trc, trc);
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                let err = (a - b).abs();
                assert!(
                    err < TOL_PARITY,
                    "dispatcher V3 RGB {trc:?} lane {i}: got={a}, want={b} \
                     (err={err:e})",
                );
            }
            // RGBA
            let mut got = original_rgba.clone();
            assert!(convert_f32_rgba_v2(&m, &mut got, trc, trc));
            let mut want = original_rgba.clone();
            scalar_reference_rgba(&m, &mut want, trc, trc);
            for i in 0..17 {
                for c in 0..3 {
                    let a = got[i * 4 + c];
                    let b = want[i * 4 + c];
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "dispatcher V3 RGBA {trc:?} px {i} ch {c}: got={a}, \
                         want={b} (err={err:e})",
                    );
                }
            }
        }

        // Restore — leave global state clean for other tests.
        #[cfg(feature = "avx512")]
        {
            let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(false);
            let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(false);
        }
    }
}
