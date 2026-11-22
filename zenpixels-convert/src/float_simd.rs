//! SIMD kernels for floating-point depth normalization, quantization, and RGBA alpha
//! premultiplication/unpremultiplication on `magetypes`.
//!
//! Replaces `garb`'s removed floating-point surface with policy- and invariant-specialized
//! kernels:
//!
//! - **0-ULP `u8 -> f32` and `u16 -> f32` without hardware division**:
//!   Binary geometric-series split ($1 / (2^k - 1) = \sum_{m \ge 1} 2^{-mk}$):
//!   - `u8 -> f32`: `HI = 2^-8 + 2^-16` (`0x3b80_8000`), `LO = 0x3380_8081`.
//!   - `u16 -> f32`: `HI = 2^-16` (`0x3780_0000`), `LO = 0x2f80_0080`.
//!
//!   Because `v * HI` has at most 16 mantissa bits, it is exact in IEEE-754 binary32, so
//!   `vf.mul_add_relaxed(HI, vf * LO)` (and `vf * HI + vf * LO`) matches `vf / 255.0` across
//!   all 256 `u8` inputs and `vf / 65535.0` across all 65,536 `u16` inputs with **0 ULP**
//!   disagreements under both fused FMA and unfused `mul + add`.
//!
//! - **0-mismatch single-FMA `f32 -> u8` and `f32 -> u16` nearest half-up quantization**:
//!   On tiers with single-rounding IEEE FMA (`x86_64` `v3`/`v4x`, `aarch64` `neon`, and
//!   `wasm32` `relaxed-simd` engines verified by an outer-loop sentinel FMA probe), adding
//!   `MAGIC = 1.5 * 2^23 = 12_582_912.0` (`0x4b40_0000`) inside a single fused `fma(x, scale, MAGIC)`
//!   places the unit bit at ULP position. Since `255` and `65535` are odd, the *only* binary32
//!   float in `[0.0, 1.0]` whose product `x * scale` is an exact half-integer is `x = 0.5`
//!   (`127.5 -> 128`, `32767.5 -> 32768`, where round-to-nearest-even agrees with half-up).
//!   Exhaustively verified across all `1,065,353,217` `f32` values in `[0.0, 1.0]` with zero
//!   mismatches against `(f64::from(x) * scale + 0.5) as u8/u16`.
//!   On strict `wasm128` or `relaxed-simd` engines that round twice, an exact `f64x2` SIMD
//!   kernel runs instead; `scalar` uses `f64`.
//!
//! - **`FloatFacts` specialization**:
//!   - `NO_NAN | NON_NEGATIVE | AT_MOST_ONE` (`UNIT_CLEAN`): skips both `NaN` flush and `[0.0, 1.0]` clamping (0 instructions).
//!   - `NO_NAN` (`!UNIT_CLEAN`): clamps via `v.max_fast(zero).min_fast(one)` (2 single instructions `vmaxps` + `vminps`).
//!   - Untrusted (`!NO_NAN`): flushes `NaN` + negatives via `(v & v.simd_ge(zero)).min_fast(one)` (3 instructions `vcmpgeps` + `vandps` + `vminps`), and uses `.to_i32_fast()` for truncation.

#![forbid(unsafe_code)]

use archmage::prelude::*;
use zenpixels::sample::FloatFacts;

const U8_INV_HI: f32 = f32::from_bits(0x3b80_8000);
const U8_INV_LO: f32 = f32::from_bits(0x3380_8081);

const U16_INV_HI: f32 = f32::from_bits(0x3780_0000);
const U16_INV_LO: f32 = f32::from_bits(0x2f80_0080);

#[allow(dead_code)] // Used only on hardware-FMA SIMD tiers (v3, v4x, neon, relaxed-simd wasm128).
const MAGIC_F32: f32 = 12_582_912.0; // 1.5 * 2^23 == 0x4b40_0000
#[allow(dead_code)]
const MAGIC_BITS: i32 = 0x4b40_0000;

const UNIT_CLEAN_FACTS: FloatFacts = FloatFacts::NO_NAN
    .union(FloatFacts::NON_NEGATIVE)
    .union(FloatFacts::AT_MOST_ONE);

// ---------------------------------------------------------------------------
// WASM relaxed-SIMD sentinel FMA probe (checked in outermost slice entry)
// ---------------------------------------------------------------------------

/// Probe whether the active WebAssembly engine executes `f32x4_relaxed_madd` with a
/// single fused rounding.
///
/// Sentinel identity: `(1 + 2^-12)^2 - (1 + 2^-11) = 2^-24` under single rounding,
/// whereas two roundings round `(1 + 2^-12)^2 = 1 + 2^-11 + 2^-24` (exact midpoint
/// with even neighbor `1 + 2^-11`) to `1 + 2^-11`, yielding `0.0`.
///
/// Evaluated once per process (cached in `AtomicU8`) and checked at the outermost
/// slice entry point before entering the inner vector loop.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
#[inline]
pub(crate) fn wasm_fuses_relaxed_madd() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    // 0 = unprobed, 1 = unfused (rounds twice), 2 = fused (single rounding).
    static FUSES: AtomicU8 = AtomicU8::new(0);
    let state = FUSES.load(Ordering::Relaxed);
    if state != 0 {
        return state == 2;
    }
    let fused = probe_wasm_relaxed_madd_uncached();
    FUSES.store(if fused { 2 } else { 1 }, Ordering::Relaxed);
    fused
}

#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
#[inline(never)]
pub(crate) fn probe_wasm_relaxed_madd_uncached() -> bool {
    use core::arch::wasm32 as w;
    use core::hint::black_box;

    let x = w::f32x4_splat(black_box(1.0 + 1.0 / 4096.0));
    let c = w::f32x4_splat(black_box(-(1.0 + 1.0 / 2048.0)));
    let r = w::f32x4_relaxed_madd(x, x, c);
    w::f32x4_extract_lane::<0>(r) == 1.0 / 16_777_216.0
        && w::f32x4_extract_lane::<3>(r) == 1.0 / 16_777_216.0
}

// ---------------------------------------------------------------------------
// Shared padded-chunk tail driver (zero scalar tail math in SIMD tiers)
// ---------------------------------------------------------------------------

type ChunkKernelFn<T, S, D, const N: usize> = fn(T, &[[S; N]], &mut [[D; N]]);

/// Runs a full-chunk SIMD kernel over `src`/`dst`, routing any `< N` remainder
/// through a single padded stack chunk into the exact same SIMD kernel.
///
/// Guarantees bit-exact body/tail parity and eliminates all scalar tail math
/// codegen from every SIMD tier (`v4x`, `v3`, `neon`, `wasm128`).
#[inline(always)]
fn run_chunked_padded<T: Copy, S: Copy, D: Copy, const N: usize>(
    token: T,
    src: &[S],
    dst: &mut [D],
    pad_s: S,
    pad_d: D,
    chunks_fn: ChunkKernelFn<T, S, D, N>,
) {
    let n = src.len().min(dst.len());
    let (s_chunks, s_tail) = src[..n].as_chunks::<N>();
    let (d_chunks, d_tail) = dst[..n].as_chunks_mut::<N>();
    if !s_chunks.is_empty() {
        chunks_fn(token, s_chunks, d_chunks);
    }
    if !s_tail.is_empty() {
        finish_padded_tail(token, s_tail, d_tail, pad_s, pad_d, chunks_fn);
    }
}

#[cold]
#[inline(never)]
fn finish_padded_tail<T: Copy, S: Copy, D: Copy, const N: usize>(
    token: T,
    s_tail: &[S],
    d_tail: &mut [D],
    pad_s: S,
    pad_d: D,
    chunks_fn: ChunkKernelFn<T, S, D, N>,
) {
    let len = s_tail.len();
    let mut sin = [pad_s; N];
    let mut dout = [pad_d; N];
    sin[..len].copy_from_slice(s_tail);
    chunks_fn(
        token,
        core::slice::from_ref(&sin),
        core::slice::from_mut(&mut dout),
    );
    d_tail.copy_from_slice(&dout[..len]);
}

// ---------------------------------------------------------------------------
// u8 -> f32 and u16 -> f32 (0-ULP split vs 1-ULP reciprocal)
// ---------------------------------------------------------------------------

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, u8x32))]
fn convert_u8_to_f32_chunks<const EXACT_DIV: bool>(
    token: Token,
    s_chunks: &[[u8; 32]],
    d_chunks: &mut [[f32; 32]],
) {
    let v_hi = f32x8::splat_t(token, U8_INV_HI);
    let v_lo = f32x8::splat_t(token, U8_INV_LO);
    let v_rcp = f32x8::splat_t(token, 1.0 / 255.0);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let bytes = u8x32::load_t(token, sc);
        let lo16 = bytes.widen_low();
        let hi16 = bytes.widen_high();

        let f0 = lo16.widen_low().bitcast_i32x8().to_f32();
        let f1 = lo16.widen_high().bitcast_i32x8().to_f32();
        let f2 = hi16.widen_low().bitcast_i32x8().to_f32();
        let f3 = hi16.widen_high().bitcast_i32x8().to_f32();

        let (o0, o1, o2, o3) = if EXACT_DIV {
            (
                f0.mul_add_relaxed(v_hi, f0 * v_lo),
                f1.mul_add_relaxed(v_hi, f1 * v_lo),
                f2.mul_add_relaxed(v_hi, f2 * v_lo),
                f3.mul_add_relaxed(v_hi, f3 * v_lo),
            )
        } else {
            (f0 * v_rcp, f1 * v_rcp, f2 * v_rcp, f3 * v_rcp)
        };

        let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
        let (d1, rest) = rest.split_first_chunk_mut::<8>().unwrap();
        let (d2, rest) = rest.split_first_chunk_mut::<8>().unwrap();
        let (d3, _) = rest.split_first_chunk_mut::<8>().unwrap();
        o0.store(d0);
        o1.store(d1);
        o2.store(d2);
        o3.store(d3);
    }
}

#[cold]
#[inline(never)]
fn convert_u8_to_f32_scalar<const EXACT_DIV: bool>(src: &[u8], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for (&b, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        let vf = f32::from(b);
        *d = if EXACT_DIV {
            vf * U8_INV_HI + vf * U8_INV_LO
        } else {
            vf * (1.0 / 255.0)
        };
    }
}

pub(crate) fn convert_u8_to_f32_dispatch<const EXACT_DIV: bool>(src: &[u8], dst: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u8,
            0.0f32,
            convert_u8_to_f32_chunks_v4x_t::<EXACT_DIV>,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u8,
            0.0f32,
            convert_u8_to_f32_chunks_v3_t::<EXACT_DIV>,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u8,
            0.0f32,
            convert_u8_to_f32_chunks_neon_t::<EXACT_DIV>,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u8,
            0.0f32,
            convert_u8_to_f32_chunks_wasm128_t::<EXACT_DIV>,
        );
    }
    convert_u8_to_f32_scalar::<EXACT_DIV>(src, dst);
}

#[inline]
pub(crate) fn convert_u8_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    convert_u8_to_f32_dispatch::<true>(src, dst);
}

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, u16x16))]
fn convert_u16_to_f32_chunks<const EXACT_DIV: bool>(
    token: Token,
    s_chunks: &[[u16; 16]],
    d_chunks: &mut [[f32; 16]],
) {
    let v_hi = f32x8::splat_t(token, U16_INV_HI);
    let v_lo = f32x8::splat_t(token, U16_INV_LO);
    let v_rcp = f32x8::splat_t(token, 1.0 / 65535.0);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let words = u16x16::load_t(token, sc);
        let f0 = words.widen_low().bitcast_i32x8().to_f32();
        let f1 = words.widen_high().bitcast_i32x8().to_f32();

        let (o0, o1) = if EXACT_DIV {
            (
                f0.mul_add_relaxed(v_hi, f0 * v_lo),
                f1.mul_add_relaxed(v_hi, f1 * v_lo),
            )
        } else {
            (f0 * v_rcp, f1 * v_rcp)
        };

        let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
        let (d1, _) = rest.split_first_chunk_mut::<8>().unwrap();
        o0.store(d0);
        o1.store(d1);
    }
}

#[cold]
#[inline(never)]
fn convert_u16_to_f32_scalar<const EXACT_DIV: bool>(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for (&w, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        let vf = f32::from(w);
        *d = if EXACT_DIV {
            vf * U16_INV_HI + vf * U16_INV_LO
        } else {
            vf * (1.0 / 65535.0)
        };
    }
}

pub(crate) fn convert_u16_to_f32_dispatch<const EXACT_DIV: bool>(src: &[u16], dst: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u16,
            0.0f32,
            convert_u16_to_f32_chunks_v4x_t::<EXACT_DIV>,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u16,
            0.0f32,
            convert_u16_to_f32_chunks_v3_t::<EXACT_DIV>,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u16,
            0.0f32,
            convert_u16_to_f32_chunks_neon_t::<EXACT_DIV>,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0u16,
            0.0f32,
            convert_u16_to_f32_chunks_wasm128_t::<EXACT_DIV>,
        );
    }
    convert_u16_to_f32_scalar::<EXACT_DIV>(src, dst);
}

#[inline]
pub(crate) fn convert_u16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    convert_u16_to_f32_dispatch::<true>(src, dst);
}

// ---------------------------------------------------------------------------
// f32 -> u8 and f32 -> u16 (nearest half-up + truncate, FloatFacts specialized)
// ---------------------------------------------------------------------------

#[inline(always)]
fn scalar_quantize_u8(s: f32) -> u8 {
    let c = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
    (f64::from(c) * 255.0 + 0.5) as u8
}

#[inline(always)]
fn scalar_quantize_u16(s: f32) -> u16 {
    let c = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
    (f64::from(c) * 65535.0 + 0.5) as u16
}

#[cold]
#[inline(never)]
fn scalar_quantize_u8_slice(src: &[f32], dst: &mut [u8]) {
    let n = src.len().min(dst.len());
    for (&s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        *d = scalar_quantize_u8(s);
    }
}

#[cold]
#[inline(never)]
fn scalar_quantize_u16_slice(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    for (&s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        *d = scalar_quantize_u16(s);
    }
}

/// Fused single-FMA nearest-half-up `f32 -> u8` full-chunk body.
///
/// Sound ONLY on tiers whose `mul_add_relaxed` executes a single-rounding IEEE FMA:
/// `v3`, `v4x`, `neon`, and `wasm128` when `wasm_fuses_relaxed_madd()` is true.
#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, i32x8))]
fn quantize_u8_magic_fma_chunks<const UNIT_CLEAN: bool, const NO_NAN: bool>(
    token: Token,
    s_chunks: &[[f32; 32]],
    d_chunks: &mut [[u8; 32]],
) {
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let scale = f32x8::splat_t(token, 255.0);
    let magic = f32x8::splat_t(token, MAGIC_F32);
    let magic_bits = i32x8::splat_t(token, MAGIC_BITS);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
        let (s1, rest) = rest.split_first_chunk::<8>().unwrap();
        let (s2, rest) = rest.split_first_chunk::<8>().unwrap();
        let (s3, _) = rest.split_first_chunk::<8>().unwrap();

        let mut v0 = f32x8::load_t(token, s0);
        let mut v1 = f32x8::load_t(token, s1);
        let mut v2 = f32x8::load_t(token, s2);
        let mut v3 = f32x8::load_t(token, s3);

        if !UNIT_CLEAN {
            if NO_NAN {
                v0 = v0.max_fast(zero).min_fast(one);
                v1 = v1.max_fast(zero).min_fast(one);
                v2 = v2.max_fast(zero).min_fast(one);
                v3 = v3.max_fast(zero).min_fast(one);
            } else {
                v0 = (v0 & v0.simd_ge(zero)).min_fast(one);
                v1 = (v1 & v1.simd_ge(zero)).min_fast(one);
                v2 = (v2 & v2.simd_ge(zero)).min_fast(one);
                v3 = (v3 & v3.simd_ge(zero)).min_fast(one);
            }
        }

        let c0 = v0.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;
        let c1 = v1.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;
        let c2 = v2.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;
        let c3 = v3.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;

        let lo16 = c0.narrow_saturating_i16(c1);
        let hi16 = c2.narrow_saturating_i16(c3);
        lo16.narrow_saturating_u8(hi16).store(dc);
    }
}

/// Fused single-FMA nearest-half-up `f32 -> u16` full-chunk body.
///
/// Sound ONLY on tiers whose `mul_add_relaxed` executes a single-rounding IEEE FMA:
/// `v3`, `v4x`, `neon`, and `wasm128` when `wasm_fuses_relaxed_madd()` is true.
#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, i32x8))]
fn quantize_u16_magic_fma_chunks<const UNIT_CLEAN: bool>(
    token: Token,
    s_chunks: &[[f32; 16]],
    d_chunks: &mut [[u16; 16]],
) {
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let scale = f32x8::splat_t(token, 65535.0);
    let magic = f32x8::splat_t(token, MAGIC_F32);
    let magic_bits = i32x8::splat_t(token, MAGIC_BITS);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
        let (s1, _) = rest.split_first_chunk::<8>().unwrap();

        let mut v0 = f32x8::load_t(token, s0);
        let mut v1 = f32x8::load_t(token, s1);

        if !UNIT_CLEAN {
            v0 = (v0 & v0.simd_ge(zero)).min_fast(one);
            v1 = (v1 & v1.simd_ge(zero)).min_fast(one);
        }

        let c0 = v0.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;
        let c1 = v1.mul_add_relaxed(scale, magic).bitcast_to_i32() - magic_bits;

        c0.narrow_saturating_u16(c1).store(dc);
    }
}

/// Exact `f64x2` SIMD fallback on `wasm32` when `relaxed-simd` is absent or the runtime
/// engine does not fuse `f32x4_relaxed_madd`.
#[cfg(target_arch = "wasm32")]
#[attune]
fn quantize_u8_wasm_f64_chunks_wasm128_t<const UNIT_CLEAN: bool, const NO_NAN: bool>(
    token: archmage::Wasm128Token,
    s_chunks: &[[f32; 16]],
    d_chunks: &mut [[u8; 16]],
) {
    use core::arch::wasm32 as w;
    use magetypes::simd::generic::{f32x4, u8x16};

    let zero32 = w::f32x4_splat(0.0);
    let one32 = w::f32x4_splat(1.0);
    let scale64 = w::f64x2_splat(255.0);
    let half64 = w::f64x2_splat(0.5);

    let quant4 = |mut v: w::v128| -> w::v128 {
        if !UNIT_CLEAN {
            if NO_NAN {
                v = w::f32x4_pmin(one32, w::f32x4_pmax(zero32, v));
            } else {
                let ge0 = w::f32x4_ge(v, zero32);
                let nonneg = w::v128_and(v, ge0);
                // nonneg has no NaN lanes (`NaN >= 0` was false -> +0.0), so pmin/min agree.
                v = w::f32x4_pmin(one32, nonneg);
            }
        }
        let lo64 = w::f64x2_promote_low_f32x4(v);
        let hi64 = w::f64x2_promote_low_f32x4(w::i32x4_shuffle::<2, 3, 2, 3>(v, v));
        let lo_i = w::i32x4_trunc_sat_f64x2_zero(w::f64x2_add(w::f64x2_mul(lo64, scale64), half64));
        let hi_i = w::i32x4_trunc_sat_f64x2_zero(w::f64x2_add(w::f64x2_mul(hi64, scale64), half64));
        w::i32x4_shuffle::<0, 1, 4, 5>(lo_i, hi_i)
    };

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<4>().unwrap();
        let (s1, rest) = rest.split_first_chunk::<4>().unwrap();
        let (s2, rest) = rest.split_first_chunk::<4>().unwrap();
        let (s3, _) = rest.split_first_chunk::<4>().unwrap();
        let c0 = quant4(f32x4::load_t(token, s0).raw());
        let c1 = quant4(f32x4::load_t(token, s1).raw());
        let c2 = quant4(f32x4::load_t(token, s2).raw());
        let c3 = quant4(f32x4::load_t(token, s3).raw());
        let p01 = w::i16x8_narrow_i32x4(c0, c1);
        let p23 = w::i16x8_narrow_i32x4(c2, c3);
        let bytes = w::u8x16_narrow_i16x8(p01, p23);
        u8x16::from_raw_t(token, bytes).store(dc);
    }
}

/// Exact `f64x2` SIMD fallback on `wasm32` for `f32 -> u16`.
#[cfg(target_arch = "wasm32")]
#[attune]
fn quantize_u16_wasm_f64_chunks_wasm128_t<const UNIT_CLEAN: bool>(
    token: archmage::Wasm128Token,
    s_chunks: &[[f32; 8]],
    d_chunks: &mut [[u16; 8]],
) {
    use core::arch::wasm32 as w;
    use magetypes::simd::generic::{f32x4, u16x8};

    let zero32 = w::f32x4_splat(0.0);
    let one32 = w::f32x4_splat(1.0);
    let scale64 = w::f64x2_splat(65535.0);
    let half64 = w::f64x2_splat(0.5);

    let quant4 = |mut v: w::v128| -> w::v128 {
        if !UNIT_CLEAN {
            let ge0 = w::f32x4_ge(v, zero32);
            let nonneg = w::v128_and(v, ge0);
            v = w::f32x4_pmin(one32, nonneg);
        }
        let lo64 = w::f64x2_promote_low_f32x4(v);
        let hi64 = w::f64x2_promote_low_f32x4(w::i32x4_shuffle::<2, 3, 2, 3>(v, v));
        let lo_i = w::i32x4_trunc_sat_f64x2_zero(w::f64x2_add(w::f64x2_mul(lo64, scale64), half64));
        let hi_i = w::i32x4_trunc_sat_f64x2_zero(w::f64x2_add(w::f64x2_mul(hi64, scale64), half64));
        w::i32x4_shuffle::<0, 1, 4, 5>(lo_i, hi_i)
    };

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<4>().unwrap();
        let (s1, _) = rest.split_first_chunk::<4>().unwrap();
        let c0 = quant4(f32x4::load_t(token, s0).raw());
        let c1 = quant4(f32x4::load_t(token, s1).raw());
        let words = w::u16x8_narrow_i32x4(c0, c1);
        u16x8::from_raw_t(token, words).store(dc);
    }
}

pub(crate) fn convert_f32_to_u8_round_spec<const UNIT_CLEAN: bool, const NO_NAN: bool>(
    src: &[f32],
    dst: &mut [u8],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            quantize_u8_magic_fma_chunks_v4x_t::<UNIT_CLEAN, NO_NAN>,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            quantize_u8_magic_fma_chunks_v3_t::<UNIT_CLEAN, NO_NAN>,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            quantize_u8_magic_fma_chunks_neon_t::<UNIT_CLEAN, NO_NAN>,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        #[cfg(target_feature = "relaxed-simd")]
        if wasm_fuses_relaxed_madd() {
            return run_chunked_padded(
                t,
                src,
                dst,
                0.0f32,
                0u8,
                quantize_u8_magic_fma_chunks_wasm128_t::<UNIT_CLEAN, NO_NAN>,
            );
        }
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            quantize_u8_wasm_f64_chunks_wasm128_t::<UNIT_CLEAN, NO_NAN>,
        );
    }
    scalar_quantize_u8_slice(src, dst);
}

pub(crate) fn convert_f32_to_u16_round_spec<const UNIT_CLEAN: bool>(src: &[f32], dst: &mut [u16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            quantize_u16_magic_fma_chunks_v4x_t::<UNIT_CLEAN>,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            quantize_u16_magic_fma_chunks_v3_t::<UNIT_CLEAN>,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            quantize_u16_magic_fma_chunks_neon_t::<UNIT_CLEAN>,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        #[cfg(target_feature = "relaxed-simd")]
        if wasm_fuses_relaxed_madd() {
            return run_chunked_padded(
                t,
                src,
                dst,
                0.0f32,
                0u16,
                quantize_u16_magic_fma_chunks_wasm128_t::<UNIT_CLEAN>,
            );
        }
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            quantize_u16_wasm_f64_chunks_wasm128_t::<UNIT_CLEAN>,
        );
    }
    scalar_quantize_u16_slice(src, dst);
}

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn convert_f32_to_u8_truncate_chunks(
    token: Token,
    s_chunks: &[[f32; 32]],
    d_chunks: &mut [[u8; 32]],
) {
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let scale = f32x8::splat_t(token, 255.0);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
        let (s1, rest) = rest.split_first_chunk::<8>().unwrap();
        let (s2, rest) = rest.split_first_chunk::<8>().unwrap();
        let (s3, _) = rest.split_first_chunk::<8>().unwrap();

        let v0 = f32x8::load_t(token, s0);
        let v1 = f32x8::load_t(token, s1);
        let v2 = f32x8::load_t(token, s2);
        let v3 = f32x8::load_t(token, s3);

        let v0 = (v0 & v0.simd_ge(zero)).min_fast(one);
        let v1 = (v1 & v1.simd_ge(zero)).min_fast(one);
        let v2 = (v2 & v2.simd_ge(zero)).min_fast(one);
        let v3 = (v3 & v3.simd_ge(zero)).min_fast(one);

        let c0 = (v0 * scale).to_i32_fast();
        let c1 = (v1 * scale).to_i32_fast();
        let c2 = (v2 * scale).to_i32_fast();
        let c3 = (v3 * scale).to_i32_fast();

        let lo16 = c0.narrow_saturating_i16(c1);
        let hi16 = c2.narrow_saturating_i16(c3);
        lo16.narrow_saturating_u8(hi16).store(dc);
    }
}

#[cold]
#[inline(never)]
fn convert_f32_to_u8_truncate_scalar(src: &[f32], dst: &mut [u8]) {
    let n = src.len().min(dst.len());
    for (&s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        let c = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        *d = (c * 255.0) as u8;
    }
}

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn convert_f32_to_u16_truncate_chunks(
    token: Token,
    s_chunks: &[[f32; 16]],
    d_chunks: &mut [[u16; 16]],
) {
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let scale = f32x8::splat_t(token, 65535.0);

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
        let (s1, _) = rest.split_first_chunk::<8>().unwrap();

        let v0 = f32x8::load_t(token, s0);
        let v1 = f32x8::load_t(token, s1);

        let v0 = (v0 & v0.simd_ge(zero)).min_fast(one);
        let v1 = (v1 & v1.simd_ge(zero)).min_fast(one);

        let c0 = (v0 * scale).to_i32_fast();
        let c1 = (v1 * scale).to_i32_fast();
        c0.narrow_saturating_u16(c1).store(dc);
    }
}

#[cold]
#[inline(never)]
fn convert_f32_to_u16_truncate_scalar(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    for (&s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        let c = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        *d = (c * 65535.0) as u16;
    }
}

#[inline]
pub(crate) fn convert_f32_to_u8_slice(src: &[f32], dst: &mut [u8], facts: FloatFacts) {
    if facts.contains(UNIT_CLEAN_FACTS) {
        convert_f32_to_u8_round_spec::<true, true>(src, dst);
    } else if facts.contains(FloatFacts::NO_NAN) {
        convert_f32_to_u8_round_spec::<false, true>(src, dst);
    } else {
        convert_f32_to_u8_round_spec::<false, false>(src, dst);
    }
}

#[inline]
pub(crate) fn convert_f32_to_u8_slice_untrusted(src: &[f32], dst: &mut [u8]) {
    convert_f32_to_u8_round_spec::<false, false>(src, dst);
}

#[inline]
pub(crate) fn convert_f32_to_u16_slice(src: &[f32], dst: &mut [u16], facts: FloatFacts) {
    if facts.contains(UNIT_CLEAN_FACTS) {
        convert_f32_to_u16_round_spec::<true>(src, dst);
    } else {
        convert_f32_to_u16_round_spec::<false>(src, dst);
    }
}

#[inline]
pub(crate) fn convert_f32_to_u16_slice_untrusted(src: &[f32], dst: &mut [u16]) {
    convert_f32_to_u16_round_spec::<false>(src, dst);
}

pub(crate) fn convert_f32_to_u8_truncate_slice(src: &[f32], dst: &mut [u8], _facts: FloatFacts) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            convert_f32_to_u8_truncate_chunks_v4x_t,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            convert_f32_to_u8_truncate_chunks_v3_t,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            convert_f32_to_u8_truncate_chunks_neon_t,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u8,
            convert_f32_to_u8_truncate_chunks_wasm128_t,
        );
    }
    convert_f32_to_u8_truncate_scalar(src, dst);
}

pub(crate) fn convert_f32_to_u16_truncate_slice(src: &[f32], dst: &mut [u16], _facts: FloatFacts) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            convert_f32_to_u16_truncate_chunks_v4x_t,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            convert_f32_to_u16_truncate_chunks_v3_t,
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            convert_f32_to_u16_truncate_chunks_neon_t,
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0u16,
            convert_f32_to_u16_truncate_chunks_wasm128_t,
        );
    }
    convert_f32_to_u16_truncate_scalar(src, dst);
}

// ---------------------------------------------------------------------------
// RGBA f32 premultiply / unpremultiply
// ---------------------------------------------------------------------------

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, i32x8))]
fn premul_rgba_f32_chunks(token: Token, s_chunks: &[[f32; 8]], d_chunks: &mut [[f32; 8]]) {
    let zero = f32x8::splat_t(token, 0.0);
    let alpha_lane_mask = i32x8::from_array_t(token, [0, 0, 0, -1, 0, 0, 0, -1]).bitcast_to_f32();

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let v = f32x8::load_t(token, sc);
        let arr = v.to_array();
        let alpha = f32x8::from_array_t(
            token,
            [
                arr[3], arr[3], arr[3], arr[3], arr[7], arr[7], arr[7], arr[7],
            ],
        );
        let prod = v * alpha;
        let with_a = f32x8::blend(alpha_lane_mask, v, prod);
        let zero_a = alpha.simd_eq(zero);
        let out = f32x8::blend(zero_a, zero, with_a);
        out.store(dc);
    }
}

#[cold]
#[inline(never)]
fn premul_rgba_f32_scalar(src: &[f32], dst: &mut [f32]) {
    for (sp, dp) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0)
    {
        let a = sp[3];
        if a == 0.0 {
            *dp = [0.0, 0.0, 0.0, 0.0];
        } else {
            *dp = [sp[0] * a, sp[1] * a, sp[2] * a, a];
        }
    }
}

#[attune(_*_t, -_scalar_t, _v4x_t(cfg(avx512)), define(f32x8, i32x8))]
fn unpremul_rgba_f32_chunks(token: Token, s_chunks: &[[f32; 8]], d_chunks: &mut [[f32; 8]]) {
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let alpha_lane_mask = i32x8::from_array_t(token, [0, 0, 0, -1, 0, 0, 0, -1]).bitcast_to_f32();

    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let v = f32x8::load_t(token, sc);
        let arr = v.to_array();
        let alpha = f32x8::from_array_t(
            token,
            [
                arr[3], arr[3], arr[3], arr[3], arr[7], arr[7], arr[7], arr[7],
            ],
        );
        let zero_a = alpha.simd_eq(zero);
        let safe_a = f32x8::blend(zero_a, one, alpha);
        let quot = v / safe_a;
        let with_a = f32x8::blend(alpha_lane_mask, v, quot);
        let out = f32x8::blend(zero_a, zero, with_a);
        out.store(dc);
    }
}

#[cold]
#[inline(never)]
fn unpremul_rgba_f32_scalar(src: &[f32], dst: &mut [f32]) {
    for (sp, dp) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0)
    {
        let a = sp[3];
        if a == 0.0 {
            *dp = [0.0, 0.0, 0.0, 0.0];
        } else {
            *dp = [sp[0] / a, sp[1] / a, sp[2] / a, a];
        }
    }
}

pub(crate) fn premul_rgba_f32_slice(src: &[f32], dst: &mut [f32], _facts: FloatFacts) {
    let n = (src.len().min(dst.len()) / 4) * 4;
    let (src, dst) = (&src[..n], &mut dst[..n]);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, premul_rgba_f32_chunks_v4x_t);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, premul_rgba_f32_chunks_v3_t);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, premul_rgba_f32_chunks_neon_t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0.0f32,
            premul_rgba_f32_chunks_wasm128_t,
        );
    }
    premul_rgba_f32_scalar(src, dst);
}

pub(crate) fn unpremul_rgba_f32_slice(src: &[f32], dst: &mut [f32], _facts: FloatFacts) {
    let n = (src.len().min(dst.len()) / 4) * 4;
    let (src, dst) = (&src[..n], &mut dst[..n]);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, unpremul_rgba_f32_chunks_v4x_t);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, unpremul_rgba_f32_chunks_v3_t);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        return run_chunked_padded(t, src, dst, 0.0f32, 0.0f32, unpremul_rgba_f32_chunks_neon_t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        return run_chunked_padded(
            t,
            src,
            dst,
            0.0f32,
            0.0f32,
            unpremul_rgba_f32_chunks_wasm128_t,
        );
    }
    unpremul_rgba_f32_scalar(src, dst);
}

#[cfg(feature = "__bench_float_depth")]
pub mod bench_shims {
    use super::*;

    pub fn u8_to_f32_split_exact(src: &[u8], dst: &mut [f32]) {
        convert_u8_to_f32_slice(src, dst);
    }

    pub fn u8_to_f32_recip_1ulp(src: &[u8], dst: &mut [f32]) {
        convert_u8_to_f32_dispatch::<false>(src, dst);
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8, u8x32))]
    fn u8_to_f32_div_impl(token: Token, src: &[u8], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let denom = f32x8::splat_t(token, 255.0);
        let (s_chunks, s_tail) = src.as_chunks::<32>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<32>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let bytes = u8x32::load_t(token, sc);
            let lo16 = bytes.widen_low();
            let hi16 = bytes.widen_high();
            let f0 = lo16.widen_low().bitcast_i32x8().to_f32() / denom;
            let f1 = lo16.widen_high().bitcast_i32x8().to_f32() / denom;
            let f2 = hi16.widen_low().bitcast_i32x8().to_f32() / denom;
            let f3 = hi16.widen_high().bitcast_i32x8().to_f32() / denom;
            let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
            let (d1, rest) = rest.split_first_chunk_mut::<8>().unwrap();
            let (d2, rest) = rest.split_first_chunk_mut::<8>().unwrap();
            let (d3, _) = rest.split_first_chunk_mut::<8>().unwrap();
            f0.store(d0);
            f1.store(d1);
            f2.store(d2);
            f3.store(d3);
        }
        for (&b, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = f32::from(b) / 255.0;
        }
    }

    pub fn u8_to_f32_div_0ulp(src: &[u8], dst: &mut [f32]) {
        reattune!(
            u8_to_f32_div_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8, u8x32))]
    fn u8_to_f32_newton_impl(token: Token, src: &[u8], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let rcp = f32x8::splat_t(token, 1.0 / 255.0);
        let neg_d = f32x8::splat_t(token, -255.0);
        let refine = |vf: f32x8| {
            let q = vf * rcp;
            let rem = q.mul_add_relaxed(neg_d, vf);
            rem.mul_add_relaxed(rcp, q)
        };
        let (s_chunks, s_tail) = src.as_chunks::<32>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<32>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let bytes = u8x32::load_t(token, sc);
            let lo16 = bytes.widen_low();
            let hi16 = bytes.widen_high();
            let f0 = refine(lo16.widen_low().bitcast_i32x8().to_f32());
            let f1 = refine(lo16.widen_high().bitcast_i32x8().to_f32());
            let f2 = refine(hi16.widen_low().bitcast_i32x8().to_f32());
            let f3 = refine(hi16.widen_high().bitcast_i32x8().to_f32());
            let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
            let (d1, rest) = rest.split_first_chunk_mut::<8>().unwrap();
            let (d2, rest) = rest.split_first_chunk_mut::<8>().unwrap();
            let (d3, _) = rest.split_first_chunk_mut::<8>().unwrap();
            f0.store(d0);
            f1.store(d1);
            f2.store(d2);
            f3.store(d3);
        }
        for (&b, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = f32::from(b) / 255.0;
        }
    }

    pub fn u8_to_f32_newton_0ulp(src: &[u8], dst: &mut [f32]) {
        reattune!(
            u8_to_f32_newton_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    pub fn u16_to_f32_split_exact(src: &[u16], dst: &mut [f32]) {
        convert_u16_to_f32_slice(src, dst);
    }

    pub fn u16_to_f32_recip_1ulp(src: &[u16], dst: &mut [f32]) {
        convert_u16_to_f32_dispatch::<false>(src, dst);
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8, u16x16))]
    fn u16_to_f32_div_impl(token: Token, src: &[u16], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let denom = f32x8::splat_t(token, 65535.0);
        let (s_chunks, s_tail) = src.as_chunks::<16>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<16>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let words = u16x16::load_t(token, sc);
            let f0 = words.widen_low().bitcast_i32x8().to_f32() / denom;
            let f1 = words.widen_high().bitcast_i32x8().to_f32() / denom;
            let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
            let (d1, _) = rest.split_first_chunk_mut::<8>().unwrap();
            f0.store(d0);
            f1.store(d1);
        }
        for (&w, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = f32::from(w) / 65535.0;
        }
    }

    pub fn u16_to_f32_div_0ulp(src: &[u16], dst: &mut [f32]) {
        reattune!(
            u16_to_f32_div_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8, u16x16))]
    fn u16_to_f32_newton_impl(token: Token, src: &[u16], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let rcp = f32x8::splat_t(token, 1.0 / 65535.0);
        let neg_d = f32x8::splat_t(token, -65535.0);
        let refine = |vf: f32x8| {
            let q = vf * rcp;
            let rem = q.mul_add_relaxed(neg_d, vf);
            rem.mul_add_relaxed(rcp, q)
        };
        let (s_chunks, s_tail) = src.as_chunks::<16>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<16>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let words = u16x16::load_t(token, sc);
            let f0 = refine(words.widen_low().bitcast_i32x8().to_f32());
            let f1 = refine(words.widen_high().bitcast_i32x8().to_f32());
            let (d0, rest) = dc.split_first_chunk_mut::<8>().unwrap();
            let (d1, _) = rest.split_first_chunk_mut::<8>().unwrap();
            f0.store(d0);
            f1.store(d1);
        }
        for (&w, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = f32::from(w) / 65535.0;
        }
    }

    pub fn u16_to_f32_newton_0ulp(src: &[u16], dst: &mut [f32]) {
        reattune!(
            u16_to_f32_newton_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    pub fn f32_to_u8_magic_fma_clamped(src: &[f32], dst: &mut [u8]) {
        convert_f32_to_u8_round_spec::<false, false>(src, dst);
    }

    pub fn f32_to_u8_magic_fma_no_nan(src: &[f32], dst: &mut [u8]) {
        convert_f32_to_u8_round_spec::<false, true>(src, dst);
    }

    pub fn f32_to_u8_magic_fma_unit_clean(src: &[f32], dst: &mut [u8]) {
        convert_f32_to_u8_round_spec::<true, true>(src, dst);
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
    fn f32_to_u8_mul_rne_impl(token: Token, src: &[f32], dst: &mut [u8]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let zero = f32x8::splat_t(token, 0.0);
        let one = f32x8::splat_t(token, 1.0);
        let scale = f32x8::splat_t(token, 255.0);
        let (s_chunks, s_tail) = src.as_chunks::<32>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<32>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
            let (s1, rest) = rest.split_first_chunk::<8>().unwrap();
            let (s2, rest) = rest.split_first_chunk::<8>().unwrap();
            let (s3, _) = rest.split_first_chunk::<8>().unwrap();
            let v0 = f32x8::load_t(token, s0);
            let v1 = f32x8::load_t(token, s1);
            let v2 = f32x8::load_t(token, s2);
            let v3 = f32x8::load_t(token, s3);
            let c0 = ((v0 & v0.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            let c1 = ((v1 & v1.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            let c2 = ((v2 & v2.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            let c3 = ((v3 & v3.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            c0.narrow_saturating_i16(c1)
                .narrow_saturating_u8(c2.narrow_saturating_i16(c3))
                .store(dc);
        }
        for (&s, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = scalar_quantize_u8(s);
        }
    }

    pub fn f32_to_u8_mul_rne_approx(src: &[f32], dst: &mut [u8]) {
        reattune!(
            f32_to_u8_mul_rne_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    pub fn f32_to_u8_truncate_unit_clean(src: &[f32], dst: &mut [u8]) {
        convert_f32_to_u8_truncate_slice(src, dst, UNIT_CLEAN_FACTS);
    }

    pub fn f32_to_u16_magic_fma_clamped(src: &[f32], dst: &mut [u16]) {
        convert_f32_to_u16_round_spec::<false>(src, dst);
    }

    pub fn f32_to_u16_magic_fma_no_nan(src: &[f32], dst: &mut [u16]) {
        convert_f32_to_u16_round_spec::<false>(src, dst);
    }

    pub fn f32_to_u16_magic_fma_unit_clean(src: &[f32], dst: &mut [u16]) {
        convert_f32_to_u16_round_spec::<true>(src, dst);
    }

    #[attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
    fn f32_to_u16_mul_rne_impl(token: Token, src: &[f32], dst: &mut [u16]) {
        let n = src.len().min(dst.len());
        let (src, dst) = (&src[..n], &mut dst[..n]);
        let zero = f32x8::splat_t(token, 0.0);
        let one = f32x8::splat_t(token, 1.0);
        let scale = f32x8::splat_t(token, 65535.0);
        let (s_chunks, s_tail) = src.as_chunks::<16>();
        let (d_chunks, d_tail) = dst.as_chunks_mut::<16>();
        for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
            let (s0, rest) = sc.split_first_chunk::<8>().unwrap();
            let (s1, _) = rest.split_first_chunk::<8>().unwrap();
            let v0 = f32x8::load_t(token, s0);
            let v1 = f32x8::load_t(token, s1);
            let c0 = ((v0 & v0.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            let c1 = ((v1 & v1.simd_ge(zero)).min_fast(one) * scale).to_i32_round();
            c0.narrow_saturating_u16(c1).store(dc);
        }
        for (&s, d) in s_tail.iter().zip(d_tail.iter_mut()) {
            *d = scalar_quantize_u16(s);
        }
    }

    pub fn f32_to_u16_mul_rne_approx(src: &[f32], dst: &mut [u16]) {
        reattune!(
            f32_to_u16_mul_rne_impl(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    pub fn f32_to_u16_truncate_unit_clean(src: &[f32], dst: &mut [u16]) {
        convert_f32_to_u16_truncate_slice(src, dst, UNIT_CLEAN_FACTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn exhaustive_u8_to_f32_zero_ulp_and_permutations() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let ref_out: [f32; 256] = core::array::from_fn(|i| (i as f32) / 255.0);

        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                // Test aligned + tail splits across all 256 values.
                for split in [0usize, 1, 7, 15, 31, 32, 33, 63, 129, 256] {
                    let mut out = vec![-1.0f32; split];
                    convert_u8_to_f32_slice(&src[..split], &mut out);
                    for i in 0..split {
                        assert_eq!(
                            out[i].to_bits(),
                            ref_out[i].to_bits(),
                            "u8->f32 mismatch at i={i} split={split} perm={perm:?}"
                        );
                    }
                }
            },
        );
    }

    #[test]
    fn exhaustive_u16_to_f32_zero_ulp_and_permutations() {
        let src: alloc::vec::Vec<u16> = (0..=u16::MAX).collect();
        let ref_out: alloc::vec::Vec<f32> = (0..=u16::MAX).map(|v| (v as f32) / 65535.0).collect();

        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut out = vec![-1.0f32; src.len()];
                convert_u16_to_f32_slice(&src, &mut out);
                for i in 0..src.len() {
                    assert_eq!(
                        out[i].to_bits(),
                        ref_out[i].to_bits(),
                        "u16->f32 mismatch at v={i} perm={perm:?}"
                    );
                }
                // Check odd tail length (53)
                let mut tail_out = vec![-1.0f32; 53];
                convert_u16_to_f32_slice(&src[500..553], &mut tail_out);
                for i in 0..53 {
                    assert_eq!(tail_out[i].to_bits(), ref_out[500 + i].to_bits());
                }
            },
        );
    }

    #[test]
    fn exhaustive_half_code_boundaries_and_edge_rails_f32_to_u8_and_u16() {
        // Build witness set covering:
        // 1. Adjacent f32 neighbors around EVERY half-code (k + 0.5) / 255 and (k + 0.5) / 65535!
        // 2. Non-finite & out-of-range rails: -NaN, +NaN, sNaN, -Inf, +Inf, -1.0, -0.0, +0.0, 1.0, 2.0, subnormals.
        let mut u8_witness = alloc::vec![
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
            f32::NEG_INFINITY,
            f32::INFINITY,
            -100.0,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            0.5,
            1.0,
            1.0 + f32::EPSILON,
            100.0,
        ];
        for k in 0..255u32 {
            let mid = ((k as f64) + 0.5) / 255.0;
            let f = mid as f32;
            u8_witness.push(f32::from_bits(f.to_bits().saturating_sub(2)));
            u8_witness.push(f32::from_bits(f.to_bits().saturating_sub(1)));
            u8_witness.push(f);
            u8_witness.push(f32::from_bits(f.to_bits() + 1));
            u8_witness.push(f32::from_bits(f.to_bits() + 2));
        }

        let mut u16_witness = u8_witness.clone();
        for k in 0..65535u32 {
            let mid = ((k as f64) + 0.5) / 65535.0;
            let f = mid as f32;
            u16_witness.push(f32::from_bits(f.to_bits().saturating_sub(1)));
            u16_witness.push(f);
            u16_witness.push(f32::from_bits(f.to_bits() + 1));
        }

        let ref_u8: alloc::vec::Vec<u8> =
            u8_witness.iter().map(|&s| scalar_quantize_u8(s)).collect();
        let ref_u16: alloc::vec::Vec<u16> = u16_witness
            .iter()
            .map(|&s| scalar_quantize_u16(s))
            .collect();

        let no_nan_only: alloc::vec::Vec<f32> = u16_witness
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        let ref_no_nan_u8: alloc::vec::Vec<u8> =
            no_nan_only.iter().map(|&s| scalar_quantize_u8(s)).collect();
        let ref_no_nan_u16: alloc::vec::Vec<u16> = no_nan_only
            .iter()
            .map(|&s| scalar_quantize_u16(s))
            .collect();
        let ref_trunc_no_nan_u8: alloc::vec::Vec<u8> = no_nan_only
            .iter()
            .map(|&s| (s.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        let ref_trunc_no_nan_u16: alloc::vec::Vec<u16> = no_nan_only
            .iter()
            .map(|&s| (s.clamp(0.0, 1.0) * 65535.0) as u16)
            .collect();

        let unit_only: alloc::vec::Vec<f32> = u16_witness
            .iter()
            .copied()
            .filter(|x| !x.is_nan() && *x >= 0.0 && *x <= 1.0)
            .collect();
        let ref_unit_u8: alloc::vec::Vec<u8> =
            unit_only.iter().map(|&s| scalar_quantize_u8(s)).collect();
        let ref_unit_u16: alloc::vec::Vec<u16> =
            unit_only.iter().map(|&s| scalar_quantize_u16(s)).collect();

        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut got_u8 = vec![0u8; u8_witness.len()];
                convert_f32_to_u8_slice(&u8_witness, &mut got_u8, FloatFacts::NONE);
                assert_eq!(got_u8, ref_u8, "f32->u8 clamped mismatch under {perm:?}");

                let mut got_u16 = vec![0u16; u16_witness.len()];
                convert_f32_to_u16_slice(&u16_witness, &mut got_u16, FloatFacts::NONE);
                assert_eq!(got_u16, ref_u16, "f32->u16 clamped mismatch under {perm:?}");

                let mut got_nn_u8 = vec![0u8; no_nan_only.len()];
                convert_f32_to_u8_slice(&no_nan_only, &mut got_nn_u8, FloatFacts::NO_NAN);
                assert_eq!(
                    got_nn_u8, ref_no_nan_u8,
                    "f32->u8 NO_NAN mismatch under {perm:?}"
                );

                let mut got_nn_u16 = vec![0u16; no_nan_only.len()];
                convert_f32_to_u16_slice(&no_nan_only, &mut got_nn_u16, FloatFacts::NO_NAN);
                assert_eq!(
                    got_nn_u16, ref_no_nan_u16,
                    "f32->u16 NO_NAN mismatch under {perm:?}"
                );

                let mut got_tr_u8 = vec![0u8; no_nan_only.len()];
                convert_f32_to_u8_truncate_slice(&no_nan_only, &mut got_tr_u8, FloatFacts::NO_NAN);
                assert_eq!(
                    got_tr_u8, ref_trunc_no_nan_u8,
                    "f32->u8 truncate NO_NAN mismatch under {perm:?}"
                );

                let mut got_tr_u16 = vec![0u16; no_nan_only.len()];
                convert_f32_to_u16_truncate_slice(
                    &no_nan_only,
                    &mut got_tr_u16,
                    FloatFacts::NO_NAN,
                );
                assert_eq!(
                    got_tr_u16, ref_trunc_no_nan_u16,
                    "f32->u16 truncate NO_NAN mismatch under {perm:?}"
                );

                let mut got_unit_u8 = vec![0u8; unit_only.len()];
                convert_f32_to_u8_slice(&unit_only, &mut got_unit_u8, UNIT_CLEAN_FACTS);
                assert_eq!(
                    got_unit_u8, ref_unit_u8,
                    "f32->u8 UNIT_CLEAN mismatch under {perm:?}"
                );

                let mut got_unit_u16 = vec![0u16; unit_only.len()];
                convert_f32_to_u16_slice(&unit_only, &mut got_unit_u16, UNIT_CLEAN_FACTS);
                assert_eq!(
                    got_unit_u16, ref_unit_u16,
                    "f32->u16 UNIT_CLEAN mismatch under {perm:?}"
                );
            },
        );
    }

    #[test]
    fn premul_and_unpremul_rgba_f32_preserve_zero_alpha_and_nan_payloads() {
        let nan_payload = f32::from_bits(0x7fc0_abcd);
        let pixels = [
            0.5f32,
            0.25,
            1.0,
            0.5, // normal
            0.8,
            -0.4,
            f32::INFINITY,
            0.0, // +0.0 alpha zeroes all 4 lanes (no Inf*0 NaN!)
            1.0,
            2.0,
            3.0,
            -0.0, // -0.0 alpha zeroes all 4 lanes to +0.0
            0.2,
            0.4,
            0.6,
            1.0, // opaque
            0.1,
            0.2,
            0.3,
            nan_payload, // NaN alpha preserves payload in alpha lane
        ];

        let mut ref_pre = [0.0f32; 20];
        let mut ref_unpre = [0.0f32; 20];
        for p in 0..5 {
            let sp = &pixels[p * 4..p * 4 + 4];
            let a = sp[3];
            if a == 0.0 {
                ref_pre[p * 4..p * 4 + 4].copy_from_slice(&[0.0; 4]);
                ref_unpre[p * 4..p * 4 + 4].copy_from_slice(&[0.0; 4]);
            } else {
                ref_pre[p * 4..p * 4 + 4].copy_from_slice(&[sp[0] * a, sp[1] * a, sp[2] * a, a]);
                ref_unpre[p * 4..p * 4 + 4].copy_from_slice(&[sp[0] / a, sp[1] / a, sp[2] / a, a]);
            }
        }

        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut got_pre = [0.0f32; 20];
                let mut got_unpre = [0.0f32; 20];
                premul_rgba_f32_slice(&pixels, &mut got_pre, FloatFacts::NONE);
                unpremul_rgba_f32_slice(&pixels, &mut got_unpre, FloatFacts::NONE);

                for i in 0..20 {
                    if ref_pre[i].is_nan() {
                        assert!(got_pre[i].is_nan(), "expected NaN at {i} under {perm:?}");
                        if i % 4 == 3 {
                            assert_eq!(got_pre[i].to_bits(), ref_pre[i].to_bits());
                        }
                    } else {
                        assert_eq!(
                            got_pre[i].to_bits(),
                            ref_pre[i].to_bits(),
                            "premul mismatch at lane {i} under {perm:?}"
                        );
                    }
                    if ref_unpre[i].is_nan() {
                        assert!(got_unpre[i].is_nan(), "expected NaN at {i} under {perm:?}");
                        if i % 4 == 3 {
                            assert_eq!(got_unpre[i].to_bits(), ref_unpre[i].to_bits());
                        }
                    } else {
                        assert_eq!(
                            got_unpre[i].to_bits(),
                            ref_unpre[i].to_bits(),
                            "unpremul mismatch at lane {i} under {perm:?}"
                        );
                    }
                }
            },
        );
    }

    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "relaxed-simd",
        feature = "std"
    ))]
    #[test]
    fn wasm_relaxed_fma_sentinel_probe_matches_engine() {
        let fused = super::wasm_fuses_relaxed_madd();
        assert_eq!(super::wasm_fuses_relaxed_madd(), fused);
        if let Ok(expect) = std::env::var("ZENPIXELS_EXPECT_WASM_FMA") {
            assert_eq!(fused, expect == "yes", "unexpected WASM FMA probe={fused}");
        }
    }
}
