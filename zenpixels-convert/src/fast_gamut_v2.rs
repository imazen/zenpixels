//! `fast_gamut_v2` — magetypes-driven, const-generic-specialized fused TRC + 3×3
//! matrix kernels.
//!
//! This module is the SIMD-backed implementation invoked by the production
//! gamut row converter. The 12 supported (src TRC, dst TRC) pairs are
//! specialized at compile time via `const SRC_TRC: u8` / `const DST_TRC: u8`
//! generic parameters. LLVM const-folds the per-TRC `match` inside the helpers
//! down to the single arm chosen by each monomorphization, producing the same
//! machine code as if each pair had a hand-written body.
//!
//! # Tier layout
//!
//! Three magetypes-stamped const-generic bodies cover all targets:
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
//! # Numerics
//!
//! The SIMD matrix multiply uses `mul_add` chained right-to-left, identical
//! shape to the v1 `mat3x3_x8` helper. Forward TRC calls
//! `linear_srgb::tf::*::*_to_linear_x{4,8,16}<T>`; encode calls
//! `linear_srgb::tf::*::linear_to_*_x{4,8,16}<T>`. Adobe (gamma 2.2) goes
//! through `linear_srgb::tf::gamma::{gamma_to_linear,linear_to_gamma}_x{4,8,16}`
//! with `ADOBE_GAMMA = 2.19921875`. The sRGB linearize / encode arms clamp to
//! `[0, 1]` before the kernel call to match the v1 production behavior; all
//! other TRCs (BT.709, PQ, HLG) accept HDR extended range and pass through
//! unclamped.
//!
//! Tail pixels (count not divisible by chunk width) take the scalar
//! linearize → matrix → encode path through `scalar_linearize` /
//! `scalar_encode`.

use archmage::prelude::*;
use linear_srgb::tf;
use magetypes::simd::backends::{F32x4Convert, F32x8Convert, F32x16Convert};
use magetypes::simd::generic::{
    f32x4 as GenericF32x4, f32x8 as GenericF32x8, f32x16 as GenericF32x16,
};

use crate::TransferFunction;

const ADOBE_GAMMA: f32 = 2.19921875; // Adobe RGB spec: 563/256

// =============================================================================
// TRC integer tags — encode the discriminant for const-generic specialization.
//
// Crate-private. The public API still takes `TransferFunction`; the runtime
// match in `convert_f32_v2_inner` maps the enum pair to a (TRC_, TRC_) tag
// pair, then dispatches to the matching const-generic monomorph.
//
// `TransferFunction::Linear` is intentionally NOT represented here. The
// (Linear, Linear) pair short-circuits to `convert_*_linear_v2` (matrix-only
// scalar loop, no TRC step) before this tag space is consulted. Mixed
// Linear↔tagged pairs are unsupported today and return `false` from the
// public entry — v1 didn't support them either. If a caller materializes,
// add a dedicated enum-pair branch alongside the existing `(Linear, Linear)`
// short-circuit so the linearize/encode stages can be elided rather than
// monomorphized as identity arms.
// =============================================================================

pub(crate) const TRC_SRGB: u8 = 0;
pub(crate) const TRC_BT709: u8 = 1;
pub(crate) const TRC_PQ: u8 = 2;
pub(crate) const TRC_HLG: u8 = 3;
pub(crate) const TRC_GAMMA22: u8 = 4;

// =============================================================================
// Shared scalar matrix helper
// =============================================================================

#[inline(always)]
fn mat3x3(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b)),
        m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b)),
        m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b)),
    )
}

// =============================================================================
// Scalar linearize / encode helpers, const-generic on TRC tag.
//
// `match SRC_TRC` const-folds to the chosen arm at every monomorphization,
// because `SRC_TRC` is a const generic parameter visible to LLVM. The wildcard
// arm uses `unreachable_unchecked` so LLVM provably eliminates the catch-all
// branch — `convert_f32_v2_inner` is the only caller and it only ever
// instantiates these with the five named tag values.
// =============================================================================

/// # Safety
/// Called only from the const-generic dispatcher with `SRC_TRC` ∈
/// {`TRC_SRGB`, `TRC_BT709`, `TRC_PQ`, `TRC_HLG`, `TRC_GAMMA22`}.
#[inline(always)]
fn scalar_linearize<const SRC_TRC: u8>(v: f32) -> f32 {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match SRC_TRC {
        TRC_SRGB => tf::srgb_to_linear(v),
        TRC_BT709 => tf::bt709_to_linear(v),
        TRC_PQ => tf::pq_to_linear(v),
        TRC_HLG => tf::hlg_to_linear(v),
        TRC_GAMMA22 => linear_srgb::default::gamma_to_linear(v, ADOBE_GAMMA),
        // SAFETY: dispatcher only ever instantiates with the five tag values
        // matched above. LLVM eliminates the wildcard arm entirely.
        _ => unreachable!(),
    }
}

/// # Safety
/// Called only from the const-generic dispatcher with `DST_TRC` ∈
/// {`TRC_SRGB`, `TRC_BT709`, `TRC_PQ`, `TRC_HLG`, `TRC_GAMMA22`}.
#[inline(always)]
fn scalar_encode<const DST_TRC: u8>(v: f32) -> f32 {
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match DST_TRC {
        TRC_SRGB => tf::linear_to_srgb(v),
        TRC_BT709 => tf::linear_to_bt709(v),
        TRC_PQ => tf::linear_to_pq(v),
        TRC_HLG => tf::linear_to_hlg(v),
        TRC_GAMMA22 => linear_srgb::default::linear_to_gamma(v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

// =============================================================================
// SIMD linearize / encode helpers, const-generic on TRC tag, one per width.
//
// Mirror the scalar helpers. The `TRC_SRGB` arm folds in the `[0, 1]` clamp
// that the v1 production path applied — `tf::srgb::*_x{4,8,16}` are raw
// kernels that don't clamp. `TRC_GAMMA22` uses `tf::gamma::*` which already
// clamps internally. `TRC_BT709` / `TRC_PQ` / `TRC_HLG` accept HDR extended
// range and run the raw kernel.
// =============================================================================

#[inline(always)]
fn linearize_x16<const SRC_TRC: u8, T: F32x16Convert>(
    t: T,
    v: GenericF32x16<T>,
) -> GenericF32x16<T> {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match SRC_TRC {
        TRC_SRGB => {
            let z = GenericF32x16::zero(t);
            let o = GenericF32x16::splat(t, 1.0);
            tf::srgb::srgb_to_linear_x16(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::bt709_to_linear_x16(t, v),
        TRC_PQ => tf::pq::pq_to_linear_x16(t, v),
        TRC_HLG => tf::hlg::hlg_to_linear_x16(t, v),
        TRC_GAMMA22 => tf::gamma::gamma_to_linear_x16(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn encode_x16<const DST_TRC: u8, T: F32x16Convert>(t: T, v: GenericF32x16<T>) -> GenericF32x16<T> {
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match DST_TRC {
        TRC_SRGB => {
            let z = GenericF32x16::zero(t);
            let o = GenericF32x16::splat(t, 1.0);
            tf::srgb::linear_to_srgb_x16(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::linear_to_bt709_x16(t, v),
        TRC_PQ => tf::pq::linear_to_pq_x16(t, v),
        TRC_HLG => tf::hlg::linear_to_hlg_x16(t, v),
        TRC_GAMMA22 => tf::gamma::linear_to_gamma_x16(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn linearize_x8<const SRC_TRC: u8, T: F32x8Convert>(t: T, v: GenericF32x8<T>) -> GenericF32x8<T> {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match SRC_TRC {
        TRC_SRGB => {
            let z = GenericF32x8::zero(t);
            let o = GenericF32x8::splat(t, 1.0);
            tf::srgb::srgb_to_linear_x8(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::bt709_to_linear_x8(t, v),
        TRC_PQ => tf::pq::pq_to_linear_x8(t, v),
        TRC_HLG => tf::hlg::hlg_to_linear_x8(t, v),
        TRC_GAMMA22 => tf::gamma::gamma_to_linear_x8(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn encode_x8<const DST_TRC: u8, T: F32x8Convert>(t: T, v: GenericF32x8<T>) -> GenericF32x8<T> {
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match DST_TRC {
        TRC_SRGB => {
            let z = GenericF32x8::zero(t);
            let o = GenericF32x8::splat(t, 1.0);
            tf::srgb::linear_to_srgb_x8(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::linear_to_bt709_x8(t, v),
        TRC_PQ => tf::pq::linear_to_pq_x8(t, v),
        TRC_HLG => tf::hlg::linear_to_hlg_x8(t, v),
        TRC_GAMMA22 => tf::gamma::linear_to_gamma_x8(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn linearize_x4<const SRC_TRC: u8, T: F32x4Convert>(t: T, v: GenericF32x4<T>) -> GenericF32x4<T> {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match SRC_TRC {
        TRC_SRGB => {
            let z = GenericF32x4::zero(t);
            let o = GenericF32x4::splat(t, 1.0);
            tf::srgb::srgb_to_linear_x4(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::bt709_to_linear_x4(t, v),
        TRC_PQ => tf::pq::pq_to_linear_x4(t, v),
        TRC_HLG => tf::hlg::hlg_to_linear_x4(t, v),
        TRC_GAMMA22 => tf::gamma::gamma_to_linear_x4(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn encode_x4<const DST_TRC: u8, T: F32x4Convert>(t: T, v: GenericF32x4<T>) -> GenericF32x4<T> {
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    match DST_TRC {
        TRC_SRGB => {
            let z = GenericF32x4::zero(t);
            let o = GenericF32x4::splat(t, 1.0);
            tf::srgb::linear_to_srgb_x4(t, v.max(z).min(o))
        }
        TRC_BT709 => tf::bt709::linear_to_bt709_x4(t, v),
        TRC_PQ => tf::pq::linear_to_pq_x4(t, v),
        TRC_HLG => tf::hlg::linear_to_hlg_x4(t, v),
        TRC_GAMMA22 => tf::gamma::linear_to_gamma_x4(t, v, ADOBE_GAMMA),
        _ => unreachable!(),
    }
}

// =============================================================================
// SIMD 3×3 matrix multiply, one per width.
//
// Each helper splats the nine matrix coefficients and performs the same
// `mul_add`-chained-right-to-left contraction as the scalar `mat3x3` above.
// The splats live inside the helper (not hoisted into the caller) so that
// every chunk-loop call site looks identical; with `#[inline(always)]` LLVM
// hoists the splat materialization to the loop preheader on its own, matching
// the asm produced by the previous hand-inlined form.
// =============================================================================

#[inline(always)]
fn mat3x3_x16<T: F32x16Convert>(
    t: T,
    m: &[[f32; 3]; 3],
    r: GenericF32x16<T>,
    g: GenericF32x16<T>,
    b: GenericF32x16<T>,
) -> (GenericF32x16<T>, GenericF32x16<T>, GenericF32x16<T>) {
    let m00 = GenericF32x16::splat(t, m[0][0]);
    let m01 = GenericF32x16::splat(t, m[0][1]);
    let m02 = GenericF32x16::splat(t, m[0][2]);
    let m10 = GenericF32x16::splat(t, m[1][0]);
    let m11 = GenericF32x16::splat(t, m[1][1]);
    let m12 = GenericF32x16::splat(t, m[1][2]);
    let m20 = GenericF32x16::splat(t, m[2][0]);
    let m21 = GenericF32x16::splat(t, m[2][1]);
    let m22 = GenericF32x16::splat(t, m[2][2]);
    let nr = m00.mul_add(r, m01.mul_add(g, m02 * b));
    let ng = m10.mul_add(r, m11.mul_add(g, m12 * b));
    let nb = m20.mul_add(r, m21.mul_add(g, m22 * b));
    (nr, ng, nb)
}

#[inline(always)]
fn mat3x3_x8<T: F32x8Convert>(
    t: T,
    m: &[[f32; 3]; 3],
    r: GenericF32x8<T>,
    g: GenericF32x8<T>,
    b: GenericF32x8<T>,
) -> (GenericF32x8<T>, GenericF32x8<T>, GenericF32x8<T>) {
    let m00 = GenericF32x8::splat(t, m[0][0]);
    let m01 = GenericF32x8::splat(t, m[0][1]);
    let m02 = GenericF32x8::splat(t, m[0][2]);
    let m10 = GenericF32x8::splat(t, m[1][0]);
    let m11 = GenericF32x8::splat(t, m[1][1]);
    let m12 = GenericF32x8::splat(t, m[1][2]);
    let m20 = GenericF32x8::splat(t, m[2][0]);
    let m21 = GenericF32x8::splat(t, m[2][1]);
    let m22 = GenericF32x8::splat(t, m[2][2]);
    let nr = m00.mul_add(r, m01.mul_add(g, m02 * b));
    let ng = m10.mul_add(r, m11.mul_add(g, m12 * b));
    let nb = m20.mul_add(r, m21.mul_add(g, m22 * b));
    (nr, ng, nb)
}

#[inline(always)]
fn mat3x3_x4<T: F32x4Convert>(
    t: T,
    m: &[[f32; 3]; 3],
    r: GenericF32x4<T>,
    g: GenericF32x4<T>,
    b: GenericF32x4<T>,
) -> (GenericF32x4<T>, GenericF32x4<T>, GenericF32x4<T>) {
    let m00 = GenericF32x4::splat(t, m[0][0]);
    let m01 = GenericF32x4::splat(t, m[0][1]);
    let m02 = GenericF32x4::splat(t, m[0][2]);
    let m10 = GenericF32x4::splat(t, m[1][0]);
    let m11 = GenericF32x4::splat(t, m[1][1]);
    let m12 = GenericF32x4::splat(t, m[1][2]);
    let m20 = GenericF32x4::splat(t, m[2][0]);
    let m21 = GenericF32x4::splat(t, m[2][1]);
    let m22 = GenericF32x4::splat(t, m[2][2]);
    let nr = m00.mul_add(r, m01.mul_add(g, m02 * b));
    let ng = m10.mul_add(r, m11.mul_add(g, m12 * b));
    let nb = m20.mul_add(r, m21.mul_add(g, m22 * b));
    (nr, ng, nb)
}

// =============================================================================
// Wide body — f32x16, dispatched across V4x / V4 / scalar.
//
// `CHANNELS` is 3 (RGB) or 4 (RGBA). `CHUNK` is `16 * CHANNELS` and must be
// passed explicitly because `[f32; PIXELS * CHANNELS]` would require the
// unstable `generic_const_exprs` feature. The `debug_assert_eq!` documents
// the invariant; release builds elide it.
//
// V3 is intentionally absent — the AVX2 polyfill of f32x16 to 2× 256-bit
// generates significant register pressure on this kernel. The native f32x8
// body below picks up V3.
// =============================================================================

#[magetypes(v4x, v4, scalar)]
fn convert_wide<const SRC_TRC: u8, const DST_TRC: u8, const CHANNELS: usize, const CHUNK: usize>(
    token: Token,
    m: &[[f32; 3]; 3],
    data: &mut [f32],
) {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            CHANNELS == 3 || CHANNELS == 4,
            "CHANNELS must be 3 (RGB) or 4 (RGBA)"
        );
    }
    const {
        assert!(
            CHUNK == 16 * CHANNELS,
            "CHUNK must equal PIXELS (16) * CHANNELS for the wide body"
        );
    }
    #[allow(non_camel_case_types)]
    type f32x16 = GenericF32x16<Token>;
    const PIXELS: usize = 16;

    let chunks = data.len() / CHUNK;
    let bulk = chunks * CHUNK;
    let (bulk_data, tail) = data.split_at_mut(bulk);
    for chunk in bulk_data.chunks_exact_mut(CHUNK) {
        // Fixed-size array pattern (CLAUDE.md): one try_into at chunk start
        // proves all interior indexes safe.
        let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
        let mut r = [0.0f32; PIXELS];
        let mut g = [0.0f32; PIXELS];
        let mut b = [0.0f32; PIXELS];
        for i in 0..PIXELS {
            r[i] = chunk[i * CHANNELS];
            g[i] = chunk[i * CHANNELS + 1];
            b[i] = chunk[i * CHANNELS + 2];
        }
        let rv = f32x16::load(token, &r);
        let gv = f32x16::load(token, &g);
        let bv = f32x16::load(token, &b);

        let rl = linearize_x16::<SRC_TRC, _>(token, rv);
        let gl = linearize_x16::<SRC_TRC, _>(token, gv);
        let bl = linearize_x16::<SRC_TRC, _>(token, bv);

        let (nr, ng, nb) = mat3x3_x16(token, m, rl, gl, bl);

        let or_ = encode_x16::<DST_TRC, _>(token, nr);
        let og_ = encode_x16::<DST_TRC, _>(token, ng);
        let ob_ = encode_x16::<DST_TRC, _>(token, nb);

        let mut ro = [0.0f32; PIXELS];
        let mut go = [0.0f32; PIXELS];
        let mut bo = [0.0f32; PIXELS];
        or_.store(&mut ro);
        og_.store(&mut go);
        ob_.store(&mut bo);
        for i in 0..PIXELS {
            chunk[i * CHANNELS] = ro[i];
            chunk[i * CHANNELS + 1] = go[i];
            chunk[i * CHANNELS + 2] = bo[i];
            // alpha (chunk[i*CHANNELS + 3]) is byte-exact unchanged.
        }
    }

    for pixel in tail.chunks_exact_mut(CHANNELS) {
        let r = scalar_linearize::<SRC_TRC>(pixel[0]);
        let g = scalar_linearize::<SRC_TRC>(pixel[1]);
        let b = scalar_linearize::<SRC_TRC>(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = scalar_encode::<DST_TRC>(nr);
        pixel[1] = scalar_encode::<DST_TRC>(ng);
        pixel[2] = scalar_encode::<DST_TRC>(nb);
        // pixel[3] unchanged (RGBA).
    }
}

// =============================================================================
// Native V3 body — f32x8 over X64V3Token. Native AVX2 width.
//
// Mirrors the v1 `fused_8px_rgb_<name>` / `fused_8px_rgba_<name>` shape.
// =============================================================================

#[magetypes(-v4, -neon, -wasm128, -scalar)]
fn convert_native<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
    const CHUNK: usize,
>(
    token: Token,
    m: &[[f32; 3]; 3],
    data: &mut [f32],
) {
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            CHANNELS == 3 || CHANNELS == 4,
            "CHANNELS must be 3 (RGB) or 4 (RGBA)"
        );
    }
    const {
        assert!(
            CHUNK == 8 * CHANNELS,
            "CHUNK must equal PIXELS (8) * CHANNELS for the native V3 body"
        );
    }
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const PIXELS: usize = 8;

    let chunks = data.len() / CHUNK;
    let bulk = chunks * CHUNK;
    let (bulk_data, tail) = data.split_at_mut(bulk);
    for chunk in bulk_data.chunks_exact_mut(CHUNK) {
        let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
        // Deinterleave: direct call into garb's `_v3` chunk fns (5-shuffle
        // AVX2 stride-3 recipe for RGB; AVX2 unpack + permute2f128 for RGBA).
        // `#[rite]` on the garb fns means the SIMD body fuses into this
        // `#[arcane]` outer fn — no `call` to garb at the use site. CHANNELS
        // / CHUNK are const generics; LLVM elides the unused arm. Alpha
        // plane is byte-exact passthrough for RGBA — captured here, written
        // back below unchanged so bit patterns (incl. NaN payloads) survive.
        let (r, g, b, alpha) = match CHANNELS {
            3 => {
                // CHUNK == 24 here; const-asserted at fn entry.
                let c: &[f32; 24] = (&chunk[..]).try_into().unwrap();
                let (r, g, b) = garb::deinterleave::rgb_f32_chunk8_to_planes_scalar(c);
                (r, g, b, [0.0f32; PIXELS])
            }
            4 => {
                // CHUNK == 32 here.
                let c: &[f32; 32] = (&chunk[..]).try_into().unwrap();
                let (r, g, b, a) = garb::deinterleave::rgba_f32_chunk8_to_planes_scalar(c);
                (r, g, b, a)
            }
            _ => unreachable!(),
        };

        let rv = f32x8::load(token, &r);
        let gv = f32x8::load(token, &g);
        let bv = f32x8::load(token, &b);

        let rl = linearize_x8::<SRC_TRC, _>(token, rv);
        let gl = linearize_x8::<SRC_TRC, _>(token, gv);
        let bl = linearize_x8::<SRC_TRC, _>(token, bv);

        let (nr, ng, nb) = mat3x3_x8(token, m, rl, gl, bl);

        let or_ = encode_x8::<DST_TRC, _>(token, nr);
        let og_ = encode_x8::<DST_TRC, _>(token, ng);
        let ob_ = encode_x8::<DST_TRC, _>(token, nb);

        let mut ro = [0.0f32; PIXELS];
        let mut go = [0.0f32; PIXELS];
        let mut bo = [0.0f32; PIXELS];
        or_.store(&mut ro);
        og_.store(&mut go);
        ob_.store(&mut bo);

        // Reinterleave: direct garb call. Alpha plane round-trips byte-exact.
        match CHANNELS {
            3 => {
                let out_arr: [f32; 24] =
                    garb::deinterleave::planes_to_rgb_f32_chunk8_scalar(&ro, &go, &bo);
                let dst: &mut [f32; 24] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            4 => {
                let out_arr: [f32; 32] =
                    garb::deinterleave::planes_to_rgba_f32_chunk8_scalar(&ro, &go, &bo, &alpha);
                let dst: &mut [f32; 32] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            _ => unreachable!(),
        }
    }

    for pixel in tail.chunks_exact_mut(CHANNELS) {
        let r = scalar_linearize::<SRC_TRC>(pixel[0]);
        let g = scalar_linearize::<SRC_TRC>(pixel[1]);
        let b = scalar_linearize::<SRC_TRC>(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = scalar_encode::<DST_TRC>(nr);
        pixel[1] = scalar_encode::<DST_TRC>(ng);
        pixel[2] = scalar_encode::<DST_TRC>(nb);
    }
}

// =============================================================================
// Narrow body — f32x4, dispatched across NEON / WASM128.
// =============================================================================

#[magetypes(-v4, -v3, -scalar)]
fn convert_narrow<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
    const CHUNK: usize,
>(
    token: Token,
    m: &[[f32; 3]; 3],
    data: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    const {
        assert!(
            SRC_TRC < 5,
            "SRC_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            DST_TRC < 5,
            "DST_TRC must be one of TRC_SRGB|BT709|PQ|HLG|GAMMA22"
        );
    }
    const {
        assert!(
            CHANNELS == 3 || CHANNELS == 4,
            "CHANNELS must be 3 (RGB) or 4 (RGBA)"
        );
    }
    const {
        assert!(
            CHUNK == 4 * CHANNELS,
            "CHUNK must equal PIXELS (4) * CHANNELS for the narrow body"
        );
    }
    type f32x4 = GenericF32x4<Token>;
    const PIXELS: usize = 4;

    let chunks = data.len() / CHUNK;
    let bulk = chunks * CHUNK;
    let (bulk_data, tail) = data.split_at_mut(bulk);
    for chunk in bulk_data.chunks_exact_mut(CHUNK) {
        let chunk: &mut [f32; CHUNK] = chunk.try_into().unwrap();
        // Deinterleave: direct call into garb's per-arch chunk-4 fns. NEON
        // uses `vld3q_f32` / `vld4q_f32` (single-instruction hardware
        // structure-loads); WASM128 uses the `i32x4_shuffle!`-based recipe.
        // `#[rite]` on the garb fns means the SIMD body fuses into this
        // `#[arcane]` outer fn. NEON and WASM128 monomorphizations are
        // arch-mutually-exclusive, so `#[cfg]` picks the right call per body.
        // CHANNELS / CHUNK are const generics; LLVM elides the unused arm.
        // Alpha is byte-exact passthrough for RGBA — captured here, written
        // back below unchanged.
        #[cfg(target_arch = "aarch64")]
        let (r, g, b, alpha) = match CHANNELS {
            3 => {
                let c: &[f32; 12] = (&chunk[..]).try_into().unwrap();
                let (r, g, b) = garb::deinterleave::rgb_f32_chunk4_to_planes_scalar(c);
                (r, g, b, [0.0f32; PIXELS])
            }
            4 => {
                let c: &[f32; 16] = (&chunk[..]).try_into().unwrap();
                let (r, g, b, a) = garb::deinterleave::rgba_f32_chunk4_to_planes_scalar(c);
                (r, g, b, a)
            }
            _ => unreachable!(),
        };
        #[cfg(target_arch = "wasm32")]
        let (r, g, b, alpha) = match CHANNELS {
            3 => {
                let c: &[f32; 12] = (&chunk[..]).try_into().unwrap();
                let (r, g, b) = garb::deinterleave::rgb_f32_chunk4_to_planes_scalar(c);
                (r, g, b, [0.0f32; PIXELS])
            }
            4 => {
                let c: &[f32; 16] = (&chunk[..]).try_into().unwrap();
                let (r, g, b, a) = garb::deinterleave::rgba_f32_chunk4_to_planes_scalar(c);
                (r, g, b, a)
            }
            _ => unreachable!(),
        };

        let rv = f32x4::load(token, &r);
        let gv = f32x4::load(token, &g);
        let bv = f32x4::load(token, &b);

        let rl = linearize_x4::<SRC_TRC, _>(token, rv);
        let gl = linearize_x4::<SRC_TRC, _>(token, gv);
        let bl = linearize_x4::<SRC_TRC, _>(token, bv);

        let (nr, ng, nb) = mat3x3_x4(token, m, rl, gl, bl);

        let or_ = encode_x4::<DST_TRC, _>(token, nr);
        let og_ = encode_x4::<DST_TRC, _>(token, ng);
        let ob_ = encode_x4::<DST_TRC, _>(token, nb);

        let mut ro = [0.0f32; PIXELS];
        let mut go = [0.0f32; PIXELS];
        let mut bo = [0.0f32; PIXELS];
        or_.store(&mut ro);
        og_.store(&mut go);
        ob_.store(&mut bo);

        // Reinterleave: direct garb call, arch-split via `#[cfg]`.
        #[cfg(target_arch = "aarch64")]
        match CHANNELS {
            3 => {
                let out_arr: [f32; 12] =
                    garb::deinterleave::planes_to_rgb_f32_chunk4_scalar(&ro, &go, &bo);
                let dst: &mut [f32; 12] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            4 => {
                let out_arr: [f32; 16] =
                    garb::deinterleave::planes_to_rgba_f32_chunk4_scalar(&ro, &go, &bo, &alpha);
                let dst: &mut [f32; 16] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            _ => unreachable!(),
        }
        #[cfg(target_arch = "wasm32")]
        match CHANNELS {
            3 => {
                let out_arr: [f32; 12] =
                    garb::deinterleave::planes_to_rgb_f32_chunk4_scalar(&ro, &go, &bo);
                let dst: &mut [f32; 12] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            4 => {
                let out_arr: [f32; 16] =
                    garb::deinterleave::planes_to_rgba_f32_chunk4_scalar(&ro, &go, &bo, &alpha);
                let dst: &mut [f32; 16] = (&mut chunk[..]).try_into().unwrap();
                *dst = out_arr;
            }
            _ => unreachable!(),
        }
    }

    for pixel in tail.chunks_exact_mut(CHANNELS) {
        let r = scalar_linearize::<SRC_TRC>(pixel[0]);
        let g = scalar_linearize::<SRC_TRC>(pixel[1]);
        let b = scalar_linearize::<SRC_TRC>(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = scalar_encode::<DST_TRC>(nr);
        pixel[1] = scalar_encode::<DST_TRC>(ng);
        pixel[2] = scalar_encode::<DST_TRC>(nb);
    }
}

// =============================================================================
// Per-pair tier dispatcher — chooses wide / native / narrow body based on
// runtime CPU capability, then forwards the four const generics through.
// =============================================================================

#[inline]
fn dispatch_pair<
    const SRC_TRC: u8,
    const DST_TRC: u8,
    const CHANNELS: usize,
    const WIDE_CHUNK: usize,
    const NATIVE_CHUNK: usize,
    const NARROW_CHUNK: usize,
>(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = X64V4xToken::summon() {
                return convert_wide_v4x::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(t, m, data);
            }
            if let Some(t) = X64V4Token::summon() {
                return convert_wide_v4::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(t, m, data);
            }
        }
        if let Some(t) = X64V3Token::summon() {
            return convert_native_v3::<SRC_TRC, DST_TRC, CHANNELS, NATIVE_CHUNK>(t, m, data);
        }
        return convert_wide_scalar::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(
            ScalarToken::summon().unwrap(),
            m,
            data,
        );
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    {
        if let Some(t) = NeonToken::summon() {
            return convert_narrow_neon::<SRC_TRC, DST_TRC, CHANNELS, NARROW_CHUNK>(t, m, data);
        }
        return convert_wide_scalar::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(
            ScalarToken::summon().unwrap(),
            m,
            data,
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(t) = Wasm128Token::summon() {
            return convert_narrow_wasm128::<SRC_TRC, DST_TRC, CHANNELS, NARROW_CHUNK>(t, m, data);
        }
        return convert_wide_scalar::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(
            ScalarToken::summon().unwrap(),
            m,
            data,
        );
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "wasm32",
    )))]
    {
        convert_wide_scalar::<SRC_TRC, DST_TRC, CHANNELS, WIDE_CHUNK>(
            ScalarToken::summon().unwrap(),
            m,
            data,
        )
    }
}

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
// Inner generic dispatcher — translates runtime (src_trc, dst_trc) pair to one
// of 12 const-generic specializations. Each arm becomes a direct call to a
// fully-monomorphized `dispatch_pair` instance, matching the v1 stamp shape.
// =============================================================================

#[inline]
fn convert_f32_v2_inner<
    const CHANNELS: usize,
    const WIDE_CHUNK: usize,
    const NATIVE_CHUNK: usize,
    const NARROW_CHUNK: usize,
>(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    use TransferFunction::*;

    // Linear→Linear bypasses the const-generic SIMD dispatch entirely (no TRC).
    if matches!((src_trc, dst_trc), (Linear, Linear)) {
        if CHANNELS == 3 {
            convert_f32_rgb_linear_v2(m, data);
        } else {
            convert_f32_rgba_linear_v2(m, data);
        }
        return true;
    }

    // Map enum pair → integer tag pair so the inner match can fold on consts.
    let pair = match (src_trc, dst_trc) {
        (Srgb, Srgb) => (TRC_SRGB, TRC_SRGB),
        (Bt709, Bt709) => (TRC_BT709, TRC_BT709),
        (Pq, Pq) => (TRC_PQ, TRC_PQ),
        (Hlg, Hlg) => (TRC_HLG, TRC_HLG),
        (Gamma22, Gamma22) => (TRC_GAMMA22, TRC_GAMMA22),
        (Pq, Srgb) => (TRC_PQ, TRC_SRGB),
        (Hlg, Srgb) => (TRC_HLG, TRC_SRGB),
        (Srgb, Pq) => (TRC_SRGB, TRC_PQ),
        (Bt709, Srgb) => (TRC_BT709, TRC_SRGB),
        (Srgb, Bt709) => (TRC_SRGB, TRC_BT709),
        (Gamma22, Srgb) => (TRC_GAMMA22, TRC_SRGB),
        (Srgb, Gamma22) => (TRC_SRGB, TRC_GAMMA22),
        _ => return false,
    };

    // 12 const-generic specializations. Each arm const-folds to a direct call
    // to the matching monomorph; LLVM will not emit a runtime tag check.
    match pair {
        (TRC_SRGB, TRC_SRGB) => {
            dispatch_pair::<TRC_SRGB, TRC_SRGB, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_BT709, TRC_BT709) => {
            dispatch_pair::<TRC_BT709, TRC_BT709, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_PQ, TRC_PQ) => {
            dispatch_pair::<TRC_PQ, TRC_PQ, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_HLG, TRC_HLG) => {
            dispatch_pair::<TRC_HLG, TRC_HLG, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_GAMMA22, TRC_GAMMA22) => dispatch_pair::<
            TRC_GAMMA22,
            TRC_GAMMA22,
            CHANNELS,
            WIDE_CHUNK,
            NATIVE_CHUNK,
            NARROW_CHUNK,
        >(m, data),
        (TRC_PQ, TRC_SRGB) => {
            dispatch_pair::<TRC_PQ, TRC_SRGB, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_HLG, TRC_SRGB) => {
            dispatch_pair::<TRC_HLG, TRC_SRGB, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_SRGB, TRC_PQ) => {
            dispatch_pair::<TRC_SRGB, TRC_PQ, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_BT709, TRC_SRGB) => {
            dispatch_pair::<TRC_BT709, TRC_SRGB, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_SRGB, TRC_BT709) => {
            dispatch_pair::<TRC_SRGB, TRC_BT709, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_GAMMA22, TRC_SRGB) => {
            dispatch_pair::<TRC_GAMMA22, TRC_SRGB, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        (TRC_SRGB, TRC_GAMMA22) => {
            dispatch_pair::<TRC_SRGB, TRC_GAMMA22, CHANNELS, WIDE_CHUNK, NATIVE_CHUNK, NARROW_CHUNK>(
                m, data,
            )
        }
        // SAFETY: the outer `match (src_trc, dst_trc)` only emits the 12 pairs
        // listed above. LLVM eliminates this arm.
        _ => unreachable!(),
    }
    true
}

// =============================================================================
// Public dispatchers — runtime entry points used by `fast_gamut.rs`.
// =============================================================================

/// Convert RGB f32 in-place using the given matrix and TRC pair. Returns
/// `false` if either TRC is unsupported by the v2 surface.
pub fn convert_f32_rgb_v2(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    debug_assert_eq!(data.len() % 3, 0);
    convert_f32_v2_inner::<3, { 16 * 3 }, { 8 * 3 }, { 4 * 3 }>(m, data, src_trc, dst_trc)
}

/// Convert RGBA f32 in-place using the given matrix and TRC pair. Alpha is
/// byte-exact unchanged. Returns `false` if either TRC is unsupported.
pub fn convert_f32_rgba_v2(
    m: &[[f32; 3]; 3],
    data: &mut [f32],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
) -> bool {
    debug_assert_eq!(data.len() % 4, 0);
    convert_f32_v2_inner::<4, { 16 * 4 }, { 8 * 4 }, { 4 * 4 }>(m, data, src_trc, dst_trc)
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
            Gamma22 => |v| linear_srgb::default::gamma_to_linear(v, ADOBE_GAMMA),
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
            Gamma22 => |v| linear_srgb::default::linear_to_gamma(v, ADOBE_GAMMA),
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
        assert!(!convert_f32_rgb_v2(
            &m,
            &mut data,
            TransferFunction::Pq,
            TransferFunction::Bt709
        ));
    }

    // ---- Tier permutation: scalar path bit-stable on every Token --------

    /// Verify all six magetypes-stamped tier suffixes give scalar-bit-stable
    /// output on this CPU. Uses archmage::testing::for_each_token_permutation
    /// to disable each tier in turn and confirms results match scalar
    /// reference within TOL_PARITY.
    #[test]
    fn tier_permutation_stable_per_pair() {
        use archmage::testing::{
            CompileTimePolicy, for_each_token_permutation, lock_token_testing,
        };

        let _guard = lock_token_testing();
        let m = srgb_to_bt2020_matrix();
        let original = ramp(19, 3);

        // Compute scalar reference once.
        let mut ref_out = original.clone();
        scalar_reference_rgb(
            &m,
            &mut ref_out,
            TransferFunction::Srgb,
            TransferFunction::Srgb,
        );

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            let mut data = original.clone();
            convert_f32_rgb_v2(
                &m,
                &mut data,
                TransferFunction::Srgb,
                TransferFunction::Srgb,
            );
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
    // The dispatcher routes V3 hosts to the const-generic `convert_native_v3`
    // (f32x8) monomorph instead of the wide body's V3 expansion. These tests
    // call `convert_native_v3` directly with a summoned X64V3Token and the
    // matching const-generic TRC tags, verifying byte-exact-within-tolerance
    // equivalence to the scalar reference.
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
            ($trc_tag:expr, $trc:expr) => {{
                let mut got = original.clone();
                convert_native_v3::<{ $trc_tag }, { $trc_tag }, 3, 24>(token, &m, &mut got);
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

        check_pair!(TRC_SRGB, TransferFunction::Srgb);
        check_pair!(TRC_BT709, TransferFunction::Bt709);
        check_pair!(TRC_PQ, TransferFunction::Pq);
        check_pair!(TRC_HLG, TransferFunction::Hlg);
        check_pair!(TRC_GAMMA22, TransferFunction::Gamma22);
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
            ($trc_tag:expr, $trc:expr) => {{
                let mut got = original.clone();
                convert_native_v3::<{ $trc_tag }, { $trc_tag }, 4, 32>(token, &m, &mut got);
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

        check_pair!(TRC_SRGB, TransferFunction::Srgb);
        check_pair!(TRC_BT709, TransferFunction::Bt709);
        check_pair!(TRC_PQ, TransferFunction::Pq);
        check_pair!(TRC_HLG, TransferFunction::Hlg);
        check_pair!(TRC_GAMMA22, TransferFunction::Gamma22);
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
            ($src_tag:expr, $dst_tag:expr, $src:expr, $dst:expr) => {{
                let mut got = original.clone();
                convert_native_v3::<{ $src_tag }, { $dst_tag }, 3, 24>(token, &m, &mut got);
                let mut want = original.clone();
                scalar_reference_rgb(&m, &mut want, $src, $dst);
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let err = (a - b).abs();
                    assert!(
                        err < TOL_PARITY,
                        "native_v3 cross {}->{} lane {i}: got={a}, want={b} (err={err:e})",
                        stringify!($src),
                        stringify!($dst),
                    );
                }
            }};
        }

        use TransferFunction::*;
        check_pair!(TRC_PQ, TRC_SRGB, Pq, Srgb);
        check_pair!(TRC_HLG, TRC_SRGB, Hlg, Srgb);
        check_pair!(TRC_SRGB, TRC_PQ, Srgb, Pq);
        check_pair!(TRC_BT709, TRC_SRGB, Bt709, Srgb);
        check_pair!(TRC_SRGB, TRC_BT709, Srgb, Bt709);
        check_pair!(TRC_GAMMA22, TRC_SRGB, Gamma22, Srgb);
        check_pair!(TRC_SRGB, TRC_GAMMA22, Srgb, Gamma22);
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
