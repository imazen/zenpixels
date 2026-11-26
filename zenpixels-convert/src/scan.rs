//! SIMD-accelerated pixel scanning for [`SampleFacts`] and load-bearing analysis.
//!
//! Every predicate is an early-exit pass over packed or strided rows using
//! `archmage` + `magetypes` (`v3`, `neon`, `wasm128`, `scalar`), with zero heap
//! allocation and `#![forbid(unsafe_code)]`.

#![allow(clippy::needless_range_loop)]

use archmage::prelude::*;
use zenpixels::{ChannelOrder, ExactDepth, FloatFacts, PixelView, SampleFacts, SampleType};

// ── Repeating-pattern masks ────────────────────────────────────────────

const ALPHA_MASK_RGBA8: [u8; 32] = {
    let mut a = [0u8; 32];
    let mut i = 3;
    while i < 32 {
        a[i] = 0xFF;
        i += 4;
    }
    a
};

const RGB_DELTA_MASK_RGBA8: [u8; 32] = {
    let mut a = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        a[i] = 0xFF;
        a[i + 1] = 0xFF;
        i += 4;
    }
    a
};

const fn rgb8_phase_mask(start_phase: usize) -> [u8; 32] {
    let mut a = [0u8; 32];
    let mut k = 0;
    while k < 32 {
        let phase = (start_phase + k) % 3;
        if phase == 0 || phase == 1 {
            a[k] = 0xFF;
        }
        k += 1;
    }
    a
}

const ALPHA_MASK_GA8: [u8; 32] = {
    let mut a = [0u8; 32];
    let mut i = 1;
    while i < 32 {
        a[i] = 0xFF;
        i += 2;
    }
    a
};

const ALPHA_MASK_RGBA16: [u16; 16] = {
    let mut a = [0u16; 16];
    let mut i = 3;
    while i < 16 {
        a[i] = 0xFFFF;
        i += 4;
    }
    a
};

const RGB_DELTA_MASK_RGBA16: [u16; 16] = {
    let mut a = [0u16; 16];
    let mut i = 0;
    while i < 16 {
        a[i] = 0xFFFF;
        a[i + 1] = 0xFFFF;
        i += 4;
    }
    a
};

const ALPHA_MASK_GA16: [u16; 16] = {
    let mut a = [0u16; 16];
    let mut i = 1;
    while i < 16 {
        a[i] = 0xFFFF;
        i += 2;
    }
    a
};

const fn rgb16_phase_mask(start_phase: usize) -> [u16; 16] {
    let mut a = [0u16; 16];
    let mut k = 0;
    while k < 16 {
        let phase = (start_phase + k) % 3;
        if phase == 0 || phase == 1 {
            a[k] = 0xFFFF;
        }
        k += 1;
    }
    a
}

const ALPHA_MASK_RGBA_I32: [i32; 8] = {
    let mut a = [0i32; 8];
    let mut i = 3;
    while i < 8 {
        a[i] = -1;
        i += 4;
    }
    a
};

const RGB_DELTA_MASK_RGBA_I32: [i32; 8] = {
    let mut a = [0i32; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = -1;
        a[i + 1] = -1;
        i += 4;
    }
    a
};

const ALPHA_MASK_GA_I32: [i32; 8] = {
    let mut a = [0i32; 8];
    let mut i = 1;
    while i < 8 {
        a[i] = -1;
        i += 2;
    }
    a
};

const fn rgb_phase_mask_i32(start_phase: usize) -> [i32; 8] {
    let mut a = [0i32; 8];
    let mut k = 0;
    while k < 8 {
        let phase = (start_phase + k) % 3;
        if phase == 0 || phase == 1 {
            a[k] = -1;
        }
        k += 1;
    }
    a
}

// ── SIMD byte/u16/f32 predicates ───────────────────────────────────────

pub(crate) fn is_grayscale_rgb8(rgb: &[u8]) -> bool {
    incant!(is_grayscale_rgb8_impl(rgb), [v3, neon, wasm128, scalar])
}

#[magetypes(define(u8x32), v3, neon, wasm128, scalar)]
fn is_grayscale_rgb8_impl(token: Token, rgb: &[u8]) -> bool {
    let m0 = u8x32::from_array_t(token, rgb8_phase_mask(0));
    let m1 = u8x32::from_array_t(token, rgb8_phase_mask(2));
    let m2 = u8x32::from_array_t(token, rgb8_phase_mask(1));

    let mut i = 0;
    while i + 97 <= rgb.len() {
        for (off, mask) in [(0usize, m0), (32, m1), (64, m2)] {
            let c0: &[u8; 32] = (&rgb[i + off..i + off + 32]).try_into().unwrap();
            let c1: &[u8; 32] = (&rgb[i + off + 1..i + off + 33]).try_into().unwrap();
            let v0 = u8x32::load_t(token, c0);
            let v1 = u8x32::load_t(token, c1);
            if (v0.simd_ne(v1) & mask).any_true() {
                return false;
            }
        }
        i += 96;
    }
    for px in rgb[i..].as_chunks::<3>().0 {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

pub(crate) fn is_opaque_ga8(ga: &[u8]) -> bool {
    incant!(is_opaque_ga8_impl(ga), [v3, neon, wasm128, scalar])
}

#[magetypes(define(u8x32), v3, neon, wasm128, scalar)]
fn is_opaque_ga8_impl(token: Token, ga: &[u8]) -> bool {
    let alpha_mask = u8x32::from_array_t(token, ALPHA_MASK_GA8);
    let opaque = u8x32::splat_t(token, 0xFF);
    let (chunks, tail) = u8x32::partition_slice_t(token, ga);
    for chunk in chunks {
        let v = u8x32::load_t(token, chunk);
        if (v.simd_ne(opaque) & alpha_mask).any_true() {
            return false;
        }
    }
    for px in tail.as_chunks::<2>().0 {
        if px[1] != 255 {
            return false;
        }
    }
    true
}

pub(crate) fn bit_replication_lossless_u16(samples: &[u16]) -> bool {
    incant!(
        bit_replication_lossless_u16_impl(samples),
        [v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn bit_replication_lossless_u16_impl(token: Token, samples: &[u16]) -> bool {
    let lo_mask = u16x16::splat_t(token, 0xFF);
    let (chunks, tail) = u16x16::partition_slice_t(token, samples);
    for chunk in chunks {
        let v = u16x16::load_t(token, chunk);
        let hi = v.shr_logical_const::<8>();
        let lo = v & lo_mask;
        if hi.simd_ne(lo).any_true() {
            return false;
        }
    }
    for &s in tail {
        if (s >> 8) as u8 != (s & 0xFF) as u8 {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
pub(crate) fn is_opaque_rgba16(rgba: &[u16]) -> bool {
    incant!(is_opaque_rgba16_impl(rgba), [v3, neon, wasm128, scalar])
}

#[allow(dead_code)]
#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn is_opaque_rgba16_impl(token: Token, rgba: &[u16]) -> bool {
    let alpha_mask = u16x16::from_array_t(token, ALPHA_MASK_RGBA16);
    let opaque = u16x16::splat_t(token, 0xFFFF);
    let (chunks, tail) = u16x16::partition_slice_t(token, rgba);
    for chunk in chunks {
        let v = u16x16::load_t(token, chunk);
        if (v.simd_ne(opaque) & alpha_mask).any_true() {
            return false;
        }
    }
    for px in tail.as_chunks::<4>().0 {
        if px[3] != 0xFFFF {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
pub(crate) fn is_grayscale_rgba16(rgba: &[u16]) -> bool {
    incant!(is_grayscale_rgba16_impl(rgba), [v3, neon, wasm128, scalar])
}

#[allow(dead_code)]
#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn is_grayscale_rgba16_impl(token: Token, rgba: &[u16]) -> bool {
    let mask = u16x16::from_array_t(token, RGB_DELTA_MASK_RGBA16);
    let mut i = 0;
    while i + 17 <= rgba.len() {
        let c0: &[u16; 16] = (&rgba[i..i + 16]).try_into().unwrap();
        let c1: &[u16; 16] = (&rgba[i + 1..i + 17]).try_into().unwrap();
        let v0 = u16x16::load_t(token, c0);
        let v1 = u16x16::load_t(token, c1);
        if (v0.simd_ne(v1) & mask).any_true() {
            return false;
        }
        i += 16;
    }
    for px in rgba[i..].as_chunks::<4>().0 {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
pub(crate) fn is_grayscale_rgb16(rgb: &[u16]) -> bool {
    incant!(is_grayscale_rgb16_impl(rgb), [v3, neon, wasm128, scalar])
}

#[allow(dead_code)]
#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn is_grayscale_rgb16_impl(token: Token, rgb: &[u16]) -> bool {
    let m0 = u16x16::from_array_t(token, rgb16_phase_mask(0));
    let m1 = u16x16::from_array_t(token, rgb16_phase_mask(1));
    let m2 = u16x16::from_array_t(token, rgb16_phase_mask(2));
    let mut i = 0;
    while i + 49 <= rgb.len() {
        for (off, mask) in [(0usize, m0), (16, m1), (32, m2)] {
            let c0: &[u16; 16] = (&rgb[i + off..i + off + 16]).try_into().unwrap();
            let c1: &[u16; 16] = (&rgb[i + off + 1..i + off + 17]).try_into().unwrap();
            let v0 = u16x16::load_t(token, c0);
            let v1 = u16x16::load_t(token, c1);
            if (v0.simd_ne(v1) & mask).any_true() {
                return false;
            }
        }
        i += 48;
    }
    for px in rgb[i..].as_chunks::<3>().0 {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
pub(crate) fn is_opaque_ga16(ga: &[u16]) -> bool {
    incant!(is_opaque_ga16_impl(ga), [v3, neon, wasm128, scalar])
}

#[allow(dead_code)]
#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn is_opaque_ga16_impl(token: Token, ga: &[u16]) -> bool {
    let alpha_mask = u16x16::from_array_t(token, ALPHA_MASK_GA16);
    let opaque = u16x16::splat_t(token, 0xFFFF);
    let (chunks, tail) = u16x16::partition_slice_t(token, ga);
    for chunk in chunks {
        let v = u16x16::load_t(token, chunk);
        if (v.simd_ne(opaque) & alpha_mask).any_true() {
            return false;
        }
    }
    for px in tail.as_chunks::<2>().0 {
        if px[1] != 0xFFFF {
            return false;
        }
    }
    true
}

pub(crate) fn is_opaque_rgba_f32(rgba: &[f32]) -> bool {
    incant!(is_opaque_rgba_f32_impl(rgba), [v3, neon, wasm128, scalar])
}

#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
fn is_opaque_rgba_f32_impl(token: Token, rgba: &[f32]) -> bool {
    let alpha_mask = i32x8::from_array_t(token, ALPHA_MASK_RGBA_I32);
    let one = f32x8::splat_t(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_t(token, rgba);
    for chunk in chunks {
        let v = f32x8::load_t(token, chunk);
        let bad = v.simd_ne(one).bitcast_to_i32() & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.as_chunks::<4>().0 {
        if px[3] != 1.0 {
            return false;
        }
    }
    true
}

pub(crate) fn is_grayscale_rgba_f32(rgba: &[f32]) -> bool {
    incant!(
        is_grayscale_rgba_f32_impl(rgba),
        [v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
fn is_grayscale_rgba_f32_impl(token: Token, rgba: &[f32]) -> bool {
    let mask = i32x8::from_array_t(token, RGB_DELTA_MASK_RGBA_I32);
    let mut i = 0;
    while i + 9 <= rgba.len() {
        let c0: &[f32; 8] = (&rgba[i..i + 8]).try_into().unwrap();
        let c1: &[f32; 8] = (&rgba[i + 1..i + 9]).try_into().unwrap();
        let v0 = f32x8::load_t(token, c0);
        let v1 = f32x8::load_t(token, c1);
        let masked = v0.simd_ne(v1).bitcast_to_i32() & mask;
        if masked.any_true() {
            return false;
        }
        i += 8;
    }
    for px in rgba[i..].as_chunks::<4>().0 {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

pub(crate) fn is_grayscale_rgb_f32(rgb: &[f32]) -> bool {
    incant!(is_grayscale_rgb_f32_impl(rgb), [v3, neon, wasm128, scalar])
}

#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
fn is_grayscale_rgb_f32_impl(token: Token, rgb: &[f32]) -> bool {
    let m0 = i32x8::from_array_t(token, rgb_phase_mask_i32(0));
    let m1 = i32x8::from_array_t(token, rgb_phase_mask_i32(2));
    let m2 = i32x8::from_array_t(token, rgb_phase_mask_i32(1));
    let mut i = 0;
    while i + 25 <= rgb.len() {
        for (off, mask) in [(0usize, m0), (8, m1), (16, m2)] {
            let c0: &[f32; 8] = (&rgb[i + off..i + off + 8]).try_into().unwrap();
            let c1: &[f32; 8] = (&rgb[i + off + 1..i + off + 9]).try_into().unwrap();
            let v0 = f32x8::load_t(token, c0);
            let v1 = f32x8::load_t(token, c1);
            let masked = v0.simd_ne(v1).bitcast_to_i32() & mask;
            if masked.any_true() {
                return false;
            }
        }
        i += 24;
    }
    for px in rgb[i..].as_chunks::<3>().0 {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

pub(crate) fn is_opaque_ga_f32(ga: &[f32]) -> bool {
    incant!(is_opaque_ga_f32_impl(ga), [v3, neon, wasm128, scalar])
}

#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
fn is_opaque_ga_f32_impl(token: Token, ga: &[f32]) -> bool {
    let alpha_mask = i32x8::from_array_t(token, ALPHA_MASK_GA_I32);
    let one = f32x8::splat_t(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_t(token, ga);
    for chunk in chunks {
        let v = f32x8::load_t(token, chunk);
        let bad = v.simd_ne(one).bitcast_to_i32() & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.as_chunks::<2>().0 {
        if px[1] != 1.0 {
            return false;
        }
    }
    true
}

// ── Fused single-pass RGBA8 & U16 predicates ───────────────────────────

/// Set of checks the fused RGBA8 predicate scanner should evaluate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FusedRequest {
    pub check_opaque: bool,
    pub check_grayscale: bool,
}

/// Result of a fused RGBA8 predicate pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FusedResult {
    pub is_opaque: bool,
    pub is_grayscale: bool,
}

pub(crate) fn fused_predicates_rgba8_cg(rgba: &[u8], req: FusedRequest) -> FusedResult {
    match (req.check_opaque, req.check_grayscale) {
        (true, true) => fused_cg::<true, true>(rgba),
        (true, false) => fused_cg::<true, false>(rgba),
        (false, true) => fused_cg::<false, true>(rgba),
        (false, false) => FusedResult::default(),
    }
}

#[inline]
fn fused_cg<const A: bool, const B: bool>(rgba: &[u8]) -> FusedResult {
    incant!(fused_cg_impl::<A, B>(rgba), [v3, neon, wasm128, scalar])
}

fn fused_cg_resume(rest: &[u8], next_a: bool, next_b: bool) -> FusedResult {
    match (next_a, next_b) {
        (true, true) => unreachable!("resume only runs after a flip"),
        (true, false) => {
            let s = fused_cg::<true, false>(rest);
            FusedResult {
                is_opaque: s.is_opaque,
                is_grayscale: false,
            }
        }
        (false, true) => {
            let s = fused_cg::<false, true>(rest);
            FusedResult {
                is_opaque: false,
                is_grayscale: s.is_grayscale,
            }
        }
        (false, false) => FusedResult::default(),
    }
}

#[magetypes(define(u8x32), v3, neon, wasm128, scalar)]
fn fused_cg_impl<const A: bool, const B: bool>(token: Token, rgba: &[u8]) -> FusedResult {
    if !(A || B) {
        return FusedResult::default();
    }

    let alpha_mask = u8x32::from_array_t(token, ALPHA_MASK_RGBA8);
    let rgb_delta_mask = u8x32::from_array_t(token, RGB_DELTA_MASK_RGBA8);
    let opaque = u8x32::splat_t(token, 0xFF);

    let len = rgba.len();
    let mut i = 0;

    const BLOCK_BYTES: usize = 16 * 32;
    let chunkable = (if B { len.saturating_sub(1) } else { len }) / 32 * 32;
    while i < chunkable {
        let block_end = (i + BLOCK_BYTES).min(chunkable);
        let mut acc = u8x32::splat_t(token, 0);
        let mut j = i;
        while j < block_end {
            let chunk0: &[u8; 32] = (&rgba[j..j + 32]).try_into().unwrap();
            let v0 = u8x32::load_t(token, chunk0);
            if A {
                acc |= v0.simd_ne(opaque) & alpha_mask;
            }
            if B {
                let chunk1: &[u8; 32] = (&rgba[j + 1..j + 33]).try_into().unwrap();
                let v1 = u8x32::load_t(token, chunk1);
                acc |= v0.simd_ne(v1) & rgb_delta_mask;
            }
            j += 32;
        }
        if acc.any_true() {
            let mut next_a = A;
            let mut next_b = B;
            for px in rgba[i..block_end].as_chunks::<4>().0 {
                if A && px[3] != 255 {
                    next_a = false;
                }
                if B && (px[0] != px[1] || px[1] != px[2]) {
                    next_b = false;
                }
            }
            return fused_cg_resume(&rgba[block_end..], next_a, next_b);
        }
        i = block_end;
    }

    let mut o = A;
    let mut g = B;
    while i + 4 <= len && (o | g) {
        let r = rgba[i];
        let gg = rgba[i + 1];
        let bb = rgba[i + 2];
        let a = rgba[i + 3];
        if A && o && a != 255 {
            o = false;
        }
        if B && g && (r != gg || gg != bb) {
            g = false;
        }
        i += 4;
    }

    FusedResult {
        is_opaque: o,
        is_grayscale: g,
    }
}

/// One traversal for U16 opacity, chroma, and replicated-byte precision.
/// Returns `[uses_alpha, uses_chroma, uses_low_bits]`, with `known` enabling early exit.
pub(crate) fn fused_u16(
    samples: &[u16],
    channels: usize,
    scan_alpha: bool,
    scan_chroma: bool,
    padding: bool,
    known: [bool; 3],
) -> [bool; 3] {
    incant!(
        fused_u16_impl(samples, channels, scan_alpha, scan_chroma, padding, known),
        [v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x16), v3, neon, wasm128, scalar)]
fn fused_u16_impl(
    token: Token,
    samples: &[u16],
    channels: usize,
    scan_alpha: bool,
    scan_chroma: bool,
    padding: bool,
    known: [bool; 3],
) -> [bool; 3] {
    let mut known = known;
    let phases = [0usize, 16 % channels, 32 % channels];
    let precision_masks = phases.map(|phase| {
        u16x16::from_array_t(
            token,
            core::array::from_fn(|i| {
                if padding && (i + phase) % channels == channels - 1 {
                    0
                } else {
                    u16::MAX
                }
            }),
        )
    });
    let alpha_masks = phases.map(|phase| {
        u16x16::from_array_t(
            token,
            core::array::from_fn(|i| {
                if (i + phase) % channels == channels - 1 {
                    u16::MAX
                } else {
                    0
                }
            }),
        )
    });
    let chroma_masks = phases.map(|phase| {
        u16x16::from_array_t(
            token,
            core::array::from_fn(|i| {
                if (i + phase) % channels < 2 {
                    u16::MAX
                } else {
                    0
                }
            }),
        )
    });
    let ones = u16x16::splat_t(token, u16::MAX);
    let low = u16x16::splat_t(token, 255);
    let complete = |v: [bool; 3]| (!scan_alpha || v[0]) && (!scan_chroma || v[1]) && v[2];
    let mut i = 0;
    while i + 49 <= samples.len() && !complete(known) {
        let mut alpha_bad = u16x16::splat_t(token, 0);
        let mut chroma_bad = u16x16::splat_t(token, 0);
        let mut precision_bad = u16x16::splat_t(token, 0);
        for n in 0..3 {
            let at = i + n * 16;
            let v = u16x16::load_t(token, (&samples[at..at + 16]).try_into().unwrap());
            if scan_alpha && !known[0] {
                alpha_bad |= v.simd_ne(ones) & alpha_masks[n];
            }
            if scan_chroma && !known[1] {
                let shifted =
                    u16x16::load_t(token, (&samples[at + 1..at + 17]).try_into().unwrap());
                chroma_bad |= v.simd_ne(shifted) & chroma_masks[n];
            }
            if !known[2] {
                precision_bad |= v.shr_logical_const::<8>().simd_ne(v & low) & precision_masks[n];
            }
        }
        known[0] |= alpha_bad.any_true();
        known[1] |= chroma_bad.any_true();
        known[2] |= precision_bad.any_true();
        i += 48;
    }
    for pixel in samples[i..].chunks_exact(channels) {
        if complete(known) {
            break;
        }
        if scan_alpha {
            known[0] |= pixel[channels - 1] != u16::MAX;
        }
        if scan_chroma {
            known[1] |= pixel[0] != pixel[1] || pixel[1] != pixel[2];
        }
        let data = if padding { channels - 1 } else { channels };
        known[2] |= pixel[..data].iter().any(|&v| v >> 8 != v & 255);
    }
    known
}

// ── Public SampleFacts scanner ─────────────────────────────────────────

/// Scan all active pixels of `view` and return proven [`SampleFacts`].
///
/// - Measures [`FloatFacts`] for `F32`/`F16`.
/// - Measures range-aware [`ExactDepth`] (independent color & alpha bit depths,
///   ignoring `RGBX`/`BGRX` and row padding) for canonical `U8`/`U16` (B06).
/// - Scans **actual stored alpha bytes** whenever the layout has an alpha slot —
///   even if `AlphaMode::Opaque` is declared — so a poisoned alpha byte is never
///   falsely certified as opaque (B07).
/// - Uses SIMD-accelerated predicates across contiguous or strided rows.
#[must_use]
pub fn scan_facts(view: &PixelView<'_>) -> SampleFacts {
    let layout = view.format().layout;
    let float = if layout.sample == SampleType::F32 && layout.packing.is_full() {
        view.measure_float_facts()
    } else if layout.sample == SampleType::F16 && layout.packing.is_full() {
        measure_f16_float_facts(view)
    } else {
        FloatFacts::NONE
    };

    let depth = ExactDepth::measure(view).ok();

    if !layout.packing.is_full() {
        let all_opaque = !layout.order.has_alpha_slot();
        let all_grayscale = matches!(layout.order, ChannelOrder::Gray | ChannelOrder::GrayAlpha);
        return SampleFacts::NONE
            .with_float(float)
            .with_depth(depth)
            .with_all_opaque(all_opaque)
            .with_all_grayscale(all_grayscale)
            .with_binary_alpha(all_opaque);
    }

    let has_alpha = layout.order.has_alpha_slot();
    let is_gray_layout = matches!(layout.order, ChannelOrder::Gray | ChannelOrder::GrayAlpha);
    let is_rgb_family = layout.order.is_rgb_family();

    let (all_opaque, all_grayscale) = match (layout.sample, layout.order) {
        (SampleType::U8, ChannelOrder::Rgba | ChannelOrder::Bgra) => {
            let mut req = FusedRequest {
                check_opaque: true,
                check_grayscale: true,
            };
            let mut op = true;
            let mut gr = true;
            if let Some(bytes) = view.as_contiguous_bytes() {
                let r = fused_predicates_rgba8_cg(bytes, req);
                op = r.is_opaque;
                gr = r.is_grayscale;
            } else {
                for y in 0..view.rows() {
                    if !req.check_opaque && !req.check_grayscale {
                        break;
                    }
                    let r = fused_predicates_rgba8_cg(view.row(y), req);
                    if req.check_opaque && !r.is_opaque {
                        op = false;
                        req.check_opaque = false;
                    }
                    if req.check_grayscale && !r.is_grayscale {
                        gr = false;
                        req.check_grayscale = false;
                    }
                }
            }
            (op, gr)
        }
        (SampleType::U8, ChannelOrder::Rgbx | ChannelOrder::Bgrx) => {
            let req = FusedRequest {
                check_opaque: false,
                check_grayscale: true,
            };
            let gr = if let Some(bytes) = view.as_contiguous_bytes() {
                fused_predicates_rgba8_cg(bytes, req).is_grayscale
            } else {
                (0..view.rows()).all(|y| fused_predicates_rgba8_cg(view.row(y), req).is_grayscale)
            };
            (true, gr)
        }
        (SampleType::U8, ChannelOrder::Rgb | ChannelOrder::Bgr) => {
            let gr = if let Some(bytes) = view.as_contiguous_bytes() {
                is_grayscale_rgb8(bytes)
            } else {
                (0..view.rows()).all(|y| is_grayscale_rgb8(view.row(y)))
            };
            (true, gr)
        }
        (SampleType::U8, ChannelOrder::GrayAlpha) => {
            let op = if let Some(bytes) = view.as_contiguous_bytes() {
                is_opaque_ga8(bytes)
            } else {
                (0..view.rows()).all(|y| is_opaque_ga8(view.row(y)))
            };
            (op, true)
        }
        (SampleType::U8, ChannelOrder::Gray) => (true, true),
        (
            SampleType::U16,
            ChannelOrder::Rgba
            | ChannelOrder::Bgra
            | ChannelOrder::Rgbx
            | ChannelOrder::Bgrx
            | ChannelOrder::Rgb
            | ChannelOrder::Bgr
            | ChannelOrder::GrayAlpha
            | ChannelOrder::Gray,
        ) => {
            let ch = layout.channels();
            let scan_alpha = has_alpha;
            let scan_chroma = is_rgb_family;
            let padding = layout.order.has_padding_slot();
            let mut flags = [false, false, true]; // skip low_bits (ExactDepth measured it)
            for y in 0..view.rows() {
                let row = view.row(y);
                if let Ok(s16) = bytemuck::try_cast_slice::<u8, u16>(row) {
                    flags = fused_u16(s16, ch, scan_alpha, scan_chroma, padding, flags);
                } else {
                    for px in row.chunks_exact(ch * 2) {
                        if scan_alpha && !flags[0] {
                            let a = u16::from_ne_bytes([px[(ch - 1) * 2], px[(ch - 1) * 2 + 1]]);
                            if a != u16::MAX {
                                flags[0] = true;
                            }
                        }
                        if scan_chroma && !flags[1] {
                            let s0 = u16::from_ne_bytes([px[0], px[1]]);
                            let s1 = u16::from_ne_bytes([px[2], px[3]]);
                            let s2 = u16::from_ne_bytes([px[4], px[5]]);
                            if s0 != s1 || s1 != s2 {
                                flags[1] = true;
                            }
                        }
                    }
                }
                if (!scan_alpha || flags[0]) && (!scan_chroma || flags[1]) {
                    break;
                }
            }
            (!scan_alpha || !flags[0], !scan_chroma || !flags[1])
        }
        (SampleType::F32, ChannelOrder::Rgba | ChannelOrder::Bgra) => {
            let op = scan_rows_f32(view, is_opaque_rgba_f32, |row| {
                row.as_chunks::<16>()
                    .0
                    .iter()
                    .all(|p| f32::from_ne_bytes(p[12..16].try_into().unwrap()) == 1.0)
            });
            let gr = scan_rows_f32(view, is_grayscale_rgba_f32, |row| {
                row.as_chunks::<16>().0.iter().all(|p| {
                    let r = f32::from_ne_bytes(p[0..4].try_into().unwrap());
                    let g = f32::from_ne_bytes(p[4..8].try_into().unwrap());
                    let b = f32::from_ne_bytes(p[8..12].try_into().unwrap());
                    r == g && g == b
                })
            });
            (op, gr)
        }
        (SampleType::F32, ChannelOrder::Rgbx | ChannelOrder::Bgrx) => {
            let gr = scan_rows_f32(view, is_grayscale_rgba_f32, |row| {
                row.as_chunks::<16>().0.iter().all(|p| {
                    let r = f32::from_ne_bytes(p[0..4].try_into().unwrap());
                    let g = f32::from_ne_bytes(p[4..8].try_into().unwrap());
                    let b = f32::from_ne_bytes(p[8..12].try_into().unwrap());
                    r == g && g == b
                })
            });
            (true, gr)
        }
        (SampleType::F32, ChannelOrder::Rgb | ChannelOrder::Bgr) => {
            let gr = scan_rows_f32(view, is_grayscale_rgb_f32, |row| {
                row.as_chunks::<12>().0.iter().all(|p| {
                    let r = f32::from_ne_bytes(p[0..4].try_into().unwrap());
                    let g = f32::from_ne_bytes(p[4..8].try_into().unwrap());
                    let b = f32::from_ne_bytes(p[8..12].try_into().unwrap());
                    r == g && g == b
                })
            });
            (true, gr)
        }
        (SampleType::F32, ChannelOrder::GrayAlpha) => {
            let op = scan_rows_f32(view, is_opaque_ga_f32, |row| {
                row.as_chunks::<8>()
                    .0
                    .iter()
                    .all(|p| f32::from_ne_bytes(p[4..8].try_into().unwrap()) == 1.0)
            });
            (op, true)
        }
        (SampleType::F32, ChannelOrder::Gray) => (true, true),
        (SampleType::F16, _) if is_gray_layout || is_rgb_family => {
            let ch = layout.channels();
            let mut op = true;
            let mut gr = true;
            for y in 0..view.rows() {
                for px in view.row(y).chunks_exact(ch * 2) {
                    if has_alpha && op {
                        let a = crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([
                            px[(ch - 1) * 2],
                            px[(ch - 1) * 2 + 1],
                        ]));
                        if a != 1.0 {
                            op = false;
                        }
                    }
                    if is_rgb_family && gr {
                        let s0 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[0], px[1]]));
                        let s1 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[2], px[3]]));
                        let s2 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[4], px[5]]));
                        if !(s0 == s1 && s1 == s2) {
                            gr = false;
                        }
                    }
                }
                if (!has_alpha || !op) && (!is_rgb_family || !gr) {
                    break;
                }
            }
            (!has_alpha || op, is_gray_layout || gr)
        }
        _ => (!has_alpha, is_gray_layout),
    };

    let binary_alpha = if all_opaque || !has_alpha {
        true
    } else if let Some(d) = depth {
        d.alpha_bits() == 1
    } else {
        scan_binary_alpha_float(view)
    };

    SampleFacts::NONE
        .with_float(float)
        .with_depth(depth)
        .with_all_opaque(all_opaque)
        .with_all_grayscale(all_grayscale)
        .with_binary_alpha(binary_alpha)
}

fn scan_rows_f32(
    view: &PixelView<'_>,
    simd_fn: fn(&[f32]) -> bool,
    unaligned_row: fn(&[u8]) -> bool,
) -> bool {
    if let Some(bytes) = view.as_contiguous_bytes()
        && let Ok(sf) = bytemuck::try_cast_slice::<u8, f32>(bytes)
    {
        return simd_fn(sf);
    }
    for y in 0..view.rows() {
        let row = view.row(y);
        let ok = if let Ok(sf) = bytemuck::try_cast_slice::<u8, f32>(row) {
            simd_fn(sf)
        } else {
            unaligned_row(row)
        };
        if !ok {
            return false;
        }
    }
    true
}

fn scan_binary_alpha_float(view: &PixelView<'_>) -> bool {
    let layout = view.format().layout;
    let ch = layout.channels();
    match layout.sample {
        SampleType::F32 => {
            for y in 0..view.rows() {
                for px in view.row(y).chunks_exact(ch * 4) {
                    let a = f32::from_ne_bytes(px[(ch - 1) * 4..ch * 4].try_into().unwrap());
                    if a != 0.0 && a != 1.0 {
                        return false;
                    }
                }
            }
            true
        }
        SampleType::F16 => {
            for y in 0..view.rows() {
                for px in view.row(y).chunks_exact(ch * 2) {
                    let a = crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([
                        px[(ch - 1) * 2],
                        px[(ch - 1) * 2 + 1],
                    ]));
                    if a != 0.0 && a != 1.0 {
                        return false;
                    }
                }
            }
            true
        }
        _ => false,
    }
}

fn measure_f16_float_facts(view: &PixelView<'_>) -> FloatFacts {
    let mut no_nan = true;
    let mut no_inf = true;
    let mut no_sub = true;
    let mut non_neg = true;
    let mut positive = true;
    let mut at_most_one = true;
    for y in 0..view.rows() {
        for &chunk in view.row(y).as_chunks::<2>().0 {
            let bits = u16::from_ne_bytes(chunk);
            let exp = (bits >> 10) & 0x1f;
            let mant = bits & 0x3ff;
            if exp == 0 && mant != 0 {
                no_sub = false;
            }
            let v = crate::f16_scalar::f16_bits_to_f32(bits);
            if v.is_nan() {
                no_nan = false;
                non_neg = false;
                positive = false;
                at_most_one = false;
                continue;
            }
            if v.is_infinite() {
                no_inf = false;
            }
            if v < 0.0 {
                non_neg = false;
                positive = false;
            } else if v == 0.0 {
                positive = false;
            }
            if v > 1.0 {
                at_most_one = false;
            }
        }
    }
    let mut out = FloatFacts::NONE;
    if no_nan {
        out = out | FloatFacts::NO_NAN;
    }
    if no_inf {
        out = out | FloatFacts::NO_INFINITE;
    }
    if no_sub {
        out = out | FloatFacts::NO_SUBNORMALS;
    }
    if non_neg {
        out = out | FloatFacts::NON_NEGATIVE;
    }
    if positive {
        out = out | FloatFacts::POSITIVE;
    }
    if at_most_one {
        out = out | FloatFacts::AT_MOST_ONE;
    }
    out
}
