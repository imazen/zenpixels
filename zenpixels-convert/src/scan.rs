//! SIMD-accelerated boolean reductions for codec encoder downcast decisions.
//!
//! Every predicate is a single early-exit pass over a packed byte buffer.
//! On photographic content the "is grayscale" / "is opaque" / "16-bit
//! replicates" tests bail in row 1; on screenshot/UI content they walk
//! the buffer and the SIMD width pays.
//!
//! Generic over magetypes 5-tier dispatch (`_v4x` AVX-512, `v4` AVX2,
//! `v3` SSE4.2, `neon`, `wasm128`) via `#[magetypes]`. The 512-bit-wide
//! body polyfills as two-256-bit on V4 and four-128-bit on V3+NEON+WASM.
//!
//! Buffer-layout assumptions:
//! - RGBA8: `[R, G, B, A, R, G, B, A, …]` — 4 bytes/pixel
//! - RGB8:  `[R, G, B, R, G, B, …]` — 3 bytes/pixel
//! - 16-bit: big-endian pairs `[hi, lo, hi, lo, …]`. Lossless 16→8
//!   condition is `hi == lo` (PNG-style bit-replication
//!   `u16 = u8 * 0x0101`; AVIF/JXL big-endian raw payloads share the
//!   convention).
//!
//! These are general descriptor-level predicates: they answer "is the
//! alpha channel actually used / are the chroma channels actually used /
//! does the bit depth actually carry information?" Codec-specific
//! encoders (zenpng, zenwebp, zenavif) consume the bools and decide
//! what to do with their own format-specific encodings (PNG color_type
//! byte, WebP VP8L bit depth, AVIF nclx primaries, etc.).
//!
//! See [`fused_predicates_rgba8`] / [`fused_predicates_rgba8_cg`] for
//! the single-pass three-checks-at-once variant — runs all three RGBA8
//! checks at the bandwidth cost of one for a ~2× speedup at the DRAM
//! bandwidth ceiling.

use archmage::prelude::*;

// ── Repeating-pattern masks ────────────────────────────────────────────
//
// Built at compile time. The byte values are 0xFF where we want "this lane
// is one we care about" and 0x00 where we want to ignore. Used to AND
// down a `simd_ne`/`simd_eq` mask before calling `any_true`.

/// `[0,0,0,0xFF, 0,0,0,0xFF, …]` — alpha lane in RGBA8.
const ALPHA_MASK_RGBA8: [u8; 64] = {
    let mut a = [0u8; 64];
    let mut i = 3;
    while i < 64 {
        a[i] = 0xFF;
        i += 4;
    }
    a
};

/// `[0xFF,0xFF,0,0, 0xFF,0xFF,0,0, …]` — `R^G` and `G^B` byte positions
/// when XORing RGBA8 against itself shifted by one byte.
const RGB_DELTA_MASK_RGBA8: [u8; 64] = {
    let mut a = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = 0xFF;
        a[i + 1] = 0xFF;
        i += 4;
    }
    a
};

/// `[0xFF, 0, 0xFF, 0, …]` — even-byte positions, used to compare a
/// big-endian 16-bit buffer against itself shifted by one byte (low byte
/// equals high byte).
const EVEN_BYTE_MASK: [u8; 64] = {
    let mut a = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = 0xFF;
        i += 2;
    }
    a
};

const fn rgb8_phase_mask(start_phase: usize) -> [u8; 64] {
    let mut a = [0u8; 64];
    let mut k = 0;
    while k < 64 {
        let phase = (start_phase + k) % 3;
        if phase == 0 || phase == 1 {
            a[k] = 0xFF;
        }
        k += 1;
    }
    a
}

// ── is_opaque_rgba8 ────────────────────────────────────────────────────

/// Returns true iff every alpha byte equals 255. Early-exit on first
/// non-opaque pixel.
pub fn is_opaque_rgba8(rgba: &[u8]) -> bool {
    incant!(
        is_opaque_rgba8_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn is_opaque_rgba8_impl(token: Token, rgba: &[u8]) -> bool {
    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_RGBA8);
    let opaque = u8x64::splat(token, 0xFF);
    let (chunks, tail) = u8x64::partition_slice(token, rgba);
    for chunk in chunks {
        let v = u8x64::load(token, chunk);
        // (v != 0xFF) at every byte; mask down to alpha lanes.
        let bad = v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(4) {
        if px[3] != 255 {
            return false;
        }
    }
    true
}

// ── is_grayscale_rgba8 ─────────────────────────────────────────────────

/// Returns true iff every pixel has `R == G == B`. Alpha is ignored.
/// Early-exits on first colorful pixel.
pub fn is_grayscale_rgba8(rgba: &[u8]) -> bool {
    incant!(
        is_grayscale_rgba8_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn is_grayscale_rgba8_impl(token: Token, rgba: &[u8]) -> bool {
    let mask = u8x64::from_array(token, RGB_DELTA_MASK_RGBA8);
    // The shifted-load pattern (chunk i, chunk i+1) doesn't fit
    // `partition_slice`'s non-overlapping chunks; manual stride here.
    let mut i = 0;
    while i + 65 <= rgba.len() {
        let chunk0: &[u8; 64] = (&rgba[i..i + 64]).try_into().unwrap();
        let chunk1: &[u8; 64] = (&rgba[i + 1..i + 65]).try_into().unwrap();
        let v0 = u8x64::load(token, chunk0);
        let v1 = u8x64::load(token, chunk1);
        // simd_ne yields 0xFF where bytes differ; mask keeps only the
        // R^G and G^B byte positions that matter.
        let masked = v0.simd_ne(v1) & mask;
        if masked.any_true() {
            return false;
        }
        i += 64;
    }
    for px in rgba[i..].chunks_exact(4) {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

// ── alpha_is_binary_rgba8 ──────────────────────────────────────────────

/// Returns true iff every alpha byte is exactly 0 or 255. Useful for
/// choosing tRNS encoding over a full alpha channel. Early-exit.
pub fn alpha_is_binary_rgba8(rgba: &[u8]) -> bool {
    incant!(
        alpha_is_binary_rgba8_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn alpha_is_binary_rgba8_impl(token: Token, rgba: &[u8]) -> bool {
    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_RGBA8);
    let zero = u8x64::splat(token, 0);
    let opaque = u8x64::splat(token, 0xFF);
    let (chunks, tail) = u8x64::partition_slice(token, rgba);
    for chunk in chunks {
        let v = u8x64::load(token, chunk);
        // Bad if (alpha != 0) AND (alpha != 255). Both compares produce
        // 0xFF/0 masks; AND them together, mask to alpha lanes only.
        let bad = v.simd_ne(zero) & v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(4) {
        let a = px[3];
        if a != 0 && a != 255 {
            return false;
        }
    }
    true
}

// ── is_grayscale_rgb8 ──────────────────────────────────────────────────

/// Returns true iff every RGB pixel has `R == G == B`. Early-exit.
///
/// 3-byte pixels don't tile evenly into 64-byte SIMD chunks (gcd(3,64)=1),
/// so we process 192-byte super-chunks (64 RGB pixels). Within each super-
/// chunk three masks handle the three byte-phase rotations.
pub fn is_grayscale_rgb8(rgb: &[u8]) -> bool {
    incant!(
        is_grayscale_rgb8_impl(rgb),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn is_grayscale_rgb8_impl(token: Token, rgb: &[u8]) -> bool {
    // Within bytes [0..64), [64..128), [128..192) the phase k%3 starts at
    // 0, 1, 2 respectively (because 64 % 3 == 1).
    let m0 = u8x64::from_array(token, rgb8_phase_mask(0));
    let m1 = u8x64::from_array(token, rgb8_phase_mask(1));
    let m2 = u8x64::from_array(token, rgb8_phase_mask(2));

    let mut i = 0;
    // Need 193 bytes per super-chunk (64+64+64 plus the final +1 shifted load).
    while i + 193 <= rgb.len() {
        for (off, mask) in [(0usize, m0), (64, m1), (128, m2)] {
            let c0: &[u8; 64] = (&rgb[i + off..i + off + 64]).try_into().unwrap();
            let c1: &[u8; 64] = (&rgb[i + off + 1..i + off + 65]).try_into().unwrap();
            let v0 = u8x64::load(token, c0);
            let v1 = u8x64::load(token, c1);
            let masked = v0.simd_ne(v1) & mask;
            if masked.any_true() {
                return false;
            }
        }
        i += 192;
    }
    for px in rgb[i..].chunks_exact(3) {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

// ── 16→8 bit-replication check ─────────────────────────────────────────

/// Returns true iff every u16 sample satisfies `(s >> 8) == (s & 0xFF)`,
/// i.e. the value can be losslessly downcast to u8 and reconstructed
/// via bit-replication upsample (`u16 = u8 * 0x0101`). The check is
/// byte-order-symmetric — endianness of the storage doesn't matter
/// because the test is internal to each u16.
///
/// Applies to any U16 channel layout (RGB16, RGBA16, Gray16, GrayA16):
/// every channel must satisfy the predicate independently. Take a
/// per-channel slice if you only want to test one channel.
pub fn bit_replication_lossless_u16(samples: &[u16]) -> bool {
    incant!(
        bit_replication_lossless_u16_impl(samples),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn bit_replication_lossless_u16_impl(token: Token, samples: &[u16]) -> bool {
    let lo_mask = u16x32::splat(token, 0xFF);
    let (chunks, tail) = u16x32::partition_slice(token, samples);
    for chunk in chunks {
        let v = u16x32::load(token, chunk);
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

/// Thin wrapper that takes a packed big-endian byte buffer (e.g. a PNG
/// 16-bit IDAT slice) and runs the bit-replication check. Each pair of
/// adjacent bytes must be equal — endian-symmetric. `be_bytes.len()`
/// must be a multiple of 2.
pub fn bit_replication_lossless_be16(be_bytes: &[u8]) -> bool {
    incant!(
        bit_replication_lossless_be16_impl(be_bytes),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn bit_replication_lossless_be16_impl(token: Token, be_bytes: &[u8]) -> bool {
    // Compare the buffer against a +1 shifted view: at even byte positions
    // the comparison checks pair[k]==pair[k+1] (the bit-replication test).
    // Odd positions compare pair[k+1]==pair[k+2] (across-pair, don't care)
    // and are masked out.
    let even_mask = u8x64::from_array(token, EVEN_BYTE_MASK);
    let mut i = 0;
    while i + 65 <= be_bytes.len() {
        let c0: &[u8; 64] = (&be_bytes[i..i + 64]).try_into().unwrap();
        let c1: &[u8; 64] = (&be_bytes[i + 1..i + 65]).try_into().unwrap();
        let v0 = u8x64::load(token, c0);
        let v1 = u8x64::load(token, c1);
        let masked = v0.simd_ne(v1) & even_mask;
        if masked.any_true() {
            return false;
        }
        i += 64;
    }
    for pair in be_bytes[i..].chunks_exact(2) {
        if pair[0] != pair[1] {
            return false;
        }
    }
    true
}

// ── U16 channel-layout predicates ──────────────────────────────────────
//
// Same shape as the U8 RGBA8/RGB8 predicates but operate on `&[u16]`
// directly. Channel-max for U16 is 65535. Mirror the byte-level mask
// approach: keep the lanes that matter, simd_ne them, any_true reduce.

/// `[0,0,0,0xFFFF, 0,0,0,0xFFFF, …]` — alpha lane in u16 RGBA layout.
const ALPHA_MASK_RGBA16: [u16; 32] = {
    let mut a = [0u16; 32];
    let mut i = 3;
    while i < 32 {
        a[i] = 0xFFFF;
        i += 4;
    }
    a
};

/// `[0xFFFF,0xFFFF,0,0, 0xFFFF,0xFFFF,0,0, …]` — `R^G` and `G^B` u16
/// positions when comparing RGBA16 against itself shifted by one u16.
const RGB_DELTA_MASK_RGBA16: [u16; 32] = {
    let mut a = [0u16; 32];
    let mut i = 0;
    while i < 32 {
        a[i] = 0xFFFF;
        a[i + 1] = 0xFFFF;
        i += 4;
    }
    a
};

/// `[0,0xFFFF, 0,0xFFFF, …]` — alpha lane in u16 GrayAlpha layout
/// (2 channels: gray, alpha).
const ALPHA_MASK_GA16: [u16; 32] = {
    let mut a = [0u16; 32];
    let mut i = 1;
    while i < 32 {
        a[i] = 0xFFFF;
        i += 2;
    }
    a
};

const fn rgb16_phase_mask(start_phase: usize) -> [u16; 32] {
    let mut a = [0u16; 32];
    let mut k = 0;
    while k < 32 {
        let phase = (start_phase + k) % 3;
        if phase == 0 || phase == 1 {
            a[k] = 0xFFFF;
        }
        k += 1;
    }
    a
}

/// Returns true iff every u16 alpha sample equals 65535. Layout: RGBA
/// u16 (R, G, B, A repeated). Early-exit on first non-opaque pixel.
pub fn is_opaque_rgba16(rgba: &[u16]) -> bool {
    incant!(
        is_opaque_rgba16_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn is_opaque_rgba16_impl(token: Token, rgba: &[u16]) -> bool {
    let alpha_mask = u16x32::from_array(token, ALPHA_MASK_RGBA16);
    let opaque = u16x32::splat(token, 0xFFFF);
    let (chunks, tail) = u16x32::partition_slice(token, rgba);
    for chunk in chunks {
        let v = u16x32::load(token, chunk);
        let bad = v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(4) {
        if px[3] != 0xFFFF {
            return false;
        }
    }
    true
}

/// Returns true iff every RGBA16 pixel has `R == G == B`. Alpha
/// ignored. Early-exit.
pub fn is_grayscale_rgba16(rgba: &[u16]) -> bool {
    incant!(
        is_grayscale_rgba16_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn is_grayscale_rgba16_impl(token: Token, rgba: &[u16]) -> bool {
    let mask = u16x32::from_array(token, RGB_DELTA_MASK_RGBA16);
    // Shifted-load pattern (chunk i, chunk i+1) — manual indexing.
    let mut i = 0;
    while i + 33 <= rgba.len() {
        let c0: &[u16; 32] = (&rgba[i..i + 32]).try_into().unwrap();
        let c1: &[u16; 32] = (&rgba[i + 1..i + 33]).try_into().unwrap();
        let v0 = u16x32::load(token, c0);
        let v1 = u16x32::load(token, c1);
        let masked = v0.simd_ne(v1) & mask;
        if masked.any_true() {
            return false;
        }
        i += 32;
    }
    for px in rgba[i..].chunks_exact(4) {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

/// Returns true iff every RGB16 pixel has `R == G == B`. 3-channel
/// stride doesn't tile evenly into 32-element u16x32 chunks
/// (gcd(3,32)=1), so we process 96-element super-chunks (32 RGB
/// pixels) with three phase-rotated masks.
pub fn is_grayscale_rgb16(rgb: &[u16]) -> bool {
    incant!(
        is_grayscale_rgb16_impl(rgb),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn is_grayscale_rgb16_impl(token: Token, rgb: &[u16]) -> bool {
    let m0 = u16x32::from_array(token, rgb16_phase_mask(0));
    // 32 % 3 == 2 — phase shifts by 2 each chunk; after 3 chunks we wrap.
    let m1 = u16x32::from_array(token, rgb16_phase_mask(2));
    let m2 = u16x32::from_array(token, rgb16_phase_mask(1));

    let mut i = 0;
    while i + 97 <= rgb.len() {
        for (off, mask) in [(0usize, m0), (32, m1), (64, m2)] {
            let c0: &[u16; 32] = (&rgb[i + off..i + off + 32]).try_into().unwrap();
            let c1: &[u16; 32] = (&rgb[i + off + 1..i + off + 33]).try_into().unwrap();
            let v0 = u16x32::load(token, c0);
            let v1 = u16x32::load(token, c1);
            let masked = v0.simd_ne(v1) & mask;
            if masked.any_true() {
                return false;
            }
        }
        i += 96;
    }
    for px in rgb[i..].chunks_exact(3) {
        if px[0] != px[1] || px[1] != px[2] {
            return false;
        }
    }
    true
}

/// Returns true iff every RGBA16 alpha sample is exactly 0 or 65535.
/// Early-exit.
pub fn alpha_is_binary_rgba16(rgba: &[u16]) -> bool {
    incant!(
        alpha_is_binary_rgba16_impl(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn alpha_is_binary_rgba16_impl(token: Token, rgba: &[u16]) -> bool {
    let alpha_mask = u16x32::from_array(token, ALPHA_MASK_RGBA16);
    let zero = u16x32::splat(token, 0);
    let opaque = u16x32::splat(token, 0xFFFF);
    let (chunks, tail) = u16x32::partition_slice(token, rgba);
    for chunk in chunks {
        let v = u16x32::load(token, chunk);
        let bad = v.simd_ne(zero) & v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(4) {
        let a = px[3];
        if a != 0 && a != 0xFFFF {
            return false;
        }
    }
    true
}

// ── GrayAlpha (U8) predicates ──────────────────────────────────────────

/// `[0, 0xFF, 0, 0xFF, …]` — alpha lane in u8 GrayAlpha layout
/// (2 channels: gray, alpha).
const ALPHA_MASK_GA8: [u8; 64] = {
    let mut a = [0u8; 64];
    let mut i = 1;
    while i < 64 {
        a[i] = 0xFF;
        i += 2;
    }
    a
};

/// Returns true iff every GrayAlpha8 alpha byte equals 255. Early-exit.
pub fn is_opaque_ga8(ga: &[u8]) -> bool {
    incant!(
        is_opaque_ga8_impl(ga),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn is_opaque_ga8_impl(token: Token, ga: &[u8]) -> bool {
    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_GA8);
    let opaque = u8x64::splat(token, 0xFF);
    let (chunks, tail) = u8x64::partition_slice(token, ga);
    for chunk in chunks {
        let v = u8x64::load(token, chunk);
        let bad = v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(2) {
        if px[1] != 255 {
            return false;
        }
    }
    true
}

/// Returns true iff every GrayAlpha8 alpha byte is exactly 0 or 255.
pub fn alpha_is_binary_ga8(ga: &[u8]) -> bool {
    incant!(
        alpha_is_binary_ga8_impl(ga),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn alpha_is_binary_ga8_impl(token: Token, ga: &[u8]) -> bool {
    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_GA8);
    let zero = u8x64::splat(token, 0);
    let opaque = u8x64::splat(token, 0xFF);
    let (chunks, tail) = u8x64::partition_slice(token, ga);
    for chunk in chunks {
        let v = u8x64::load(token, chunk);
        let bad = v.simd_ne(zero) & v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(2) {
        let a = px[1];
        if a != 0 && a != 255 {
            return false;
        }
    }
    true
}

// ── GrayAlpha (U16) predicates ─────────────────────────────────────────

/// Returns true iff every GrayAlpha16 alpha sample equals 65535.
pub fn is_opaque_ga16(ga: &[u16]) -> bool {
    incant!(
        is_opaque_ga16_impl(ga),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn is_opaque_ga16_impl(token: Token, ga: &[u16]) -> bool {
    let alpha_mask = u16x32::from_array(token, ALPHA_MASK_GA16);
    let opaque = u16x32::splat(token, 0xFFFF);
    let (chunks, tail) = u16x32::partition_slice(token, ga);
    for chunk in chunks {
        let v = u16x32::load(token, chunk);
        let bad = v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(2) {
        if px[1] != 0xFFFF {
            return false;
        }
    }
    true
}

/// Returns true iff every GrayAlpha16 alpha sample is exactly 0 or 65535.
pub fn alpha_is_binary_ga16(ga: &[u16]) -> bool {
    incant!(
        alpha_is_binary_ga16_impl(ga),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u16x32), v4x, v4, v3, neon, wasm128, scalar)]
fn alpha_is_binary_ga16_impl(token: Token, ga: &[u16]) -> bool {
    let alpha_mask = u16x32::from_array(token, ALPHA_MASK_GA16);
    let zero = u16x32::splat(token, 0);
    let opaque = u16x32::splat(token, 0xFFFF);
    let (chunks, tail) = u16x32::partition_slice(token, ga);
    for chunk in chunks {
        let v = u16x32::load(token, chunk);
        let bad = v.simd_ne(zero) & v.simd_ne(opaque) & alpha_mask;
        if bad.any_true() {
            return false;
        }
    }
    for px in tail.chunks_exact(2) {
        let a = px[1];
        if a != 0 && a != 0xFFFF {
            return false;
        }
    }
    true
}

// ── Fused single-pass RGBA8 predicates ─────────────────────────────────
//
// Runs `is_opaque`, `is_grayscale`, and `alpha_is_binary` against the
// same buffer in **one** streaming pass — three checks at the bandwidth
// cost of one. Every check is lane-parallel against the same 64-byte
// SIMD register, so the per-iteration cost is roughly:
//   1 load + 0–1 shifted load + 3 simd_ne + 3 mask + 3 any_true.
//
// Two variants:
//   * `fused_predicates_rgba8` — runtime-branch, drops checks via
//     `still_*` flags as they flip. One function, simpler.
//   * `fused_predicates_rgba8_cg` — const-generic recursive trampoline.
//     8 specializations (one per (A, B, C) bool tuple); when a check
//     flips we tail-recurse into the specialization where that const
//     is `false`, dead-code-eliminating the dropped check from the
//     hot loop. Eliminates the `if still_*` branches at the cost of
//     monomorphized code-size.
//
// Both honor the request flags so a caller can ask for any subset.

/// Set of checks the fused predicate scanner should evaluate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedRequest {
    pub check_opaque: bool,
    pub check_grayscale: bool,
    pub check_binary_alpha: bool,
}

impl FusedRequest {
    /// Request all three checks.
    pub const fn all() -> Self {
        Self {
            check_opaque: true,
            check_grayscale: true,
            check_binary_alpha: true,
        }
    }
}

/// Result of a fused predicate pass. Each field is meaningful only if
/// the corresponding `check_*` was requested; for unrequested checks the
/// value is `false` (treated as "we don't know / not requested").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedResult {
    pub is_opaque: bool,
    pub is_grayscale: bool,
    pub is_binary_alpha: bool,
}

// ── Runtime-branch fused predicate ─────────────────────────────────────

pub fn fused_predicates_rgba8(rgba: &[u8], req: FusedRequest) -> FusedResult {
    incant!(
        fused_predicates_rgba8_impl(rgba, req),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn fused_predicates_rgba8_impl(token: Token, rgba: &[u8], req: FusedRequest) -> FusedResult {
    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_RGBA8);
    let rgb_delta_mask = u8x64::from_array(token, RGB_DELTA_MASK_RGBA8);
    let zero = u8x64::splat(token, 0);
    let opaque = u8x64::splat(token, 0xFF);

    let mut still_o = req.check_opaque;
    let mut still_g = req.check_grayscale;
    let mut still_b = req.check_binary_alpha;

    let mut i = 0;
    let len = rgba.len();
    // While at least one check is alive AND we have room for the shifted
    // grayscale load. If grayscale isn't requested (or has flipped off)
    // the trailing 64-byte chunk gets handled by the second loop below.
    while still_o | still_g | still_b {
        if !still_g && i + 64 > len {
            break;
        }
        if still_g && i + 65 > len {
            break;
        }
        let chunk0: &[u8; 64] = (&rgba[i..i + 64]).try_into().unwrap();
        let v0 = u8x64::load(token, chunk0);

        if still_o {
            let bad = v0.simd_ne(opaque) & alpha_mask;
            if bad.any_true() {
                still_o = false;
            }
        }
        if still_b {
            let bad = v0.simd_ne(zero) & v0.simd_ne(opaque) & alpha_mask;
            if bad.any_true() {
                still_b = false;
            }
        }
        if still_g {
            let chunk1: &[u8; 64] = (&rgba[i + 1..i + 65]).try_into().unwrap();
            let v1 = u8x64::load(token, chunk1);
            let bad = v0.simd_ne(v1) & rgb_delta_mask;
            if bad.any_true() {
                still_g = false;
            }
        }

        i += 64;
    }

    // Scalar tail (handles whatever's left, including the last 64-byte
    // chunk we couldn't enter when grayscale was alive due to the +1
    // shifted-load constraint).
    while i + 4 <= len && (still_o | still_g | still_b) {
        let r = rgba[i];
        let g = rgba[i + 1];
        let b = rgba[i + 2];
        let a = rgba[i + 3];
        if still_o && a != 255 {
            still_o = false;
        }
        if still_g && (r != g || g != b) {
            still_g = false;
        }
        if still_b && a != 0 && a != 255 {
            still_b = false;
        }
        i += 4;
    }

    FusedResult {
        is_opaque: still_o,
        is_grayscale: still_g,
        is_binary_alpha: still_b,
    }
}

// ── Const-generic recursive fused predicate ────────────────────────────
//
// Same kernel, but each check's enabled state is a `const bool`. When a
// check flips we recurse into a specialization with that const set to
// `false`. The compiler eliminates the corresponding `if A { … }` branch
// in the dead specialization, leaving a tighter hot loop.

pub fn fused_predicates_rgba8_cg(rgba: &[u8], req: FusedRequest) -> FusedResult {
    match (req.check_opaque, req.check_grayscale, req.check_binary_alpha) {
        (true, true, true) => fused_cg::<true, true, true>(rgba),
        (true, true, false) => fused_cg::<true, true, false>(rgba),
        (true, false, true) => fused_cg::<true, false, true>(rgba),
        (true, false, false) => fused_cg::<true, false, false>(rgba),
        (false, true, true) => fused_cg::<false, true, true>(rgba),
        (false, true, false) => fused_cg::<false, true, false>(rgba),
        (false, false, true) => fused_cg::<false, false, true>(rgba),
        (false, false, false) => FusedResult::default(),
    }
}

#[inline]
fn fused_cg<const A: bool, const B: bool, const C: bool>(rgba: &[u8]) -> FusedResult {
    incant!(
        fused_cg_impl::<A, B, C>(rgba),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(define(u8x64), v4x, v4, v3, neon, wasm128, scalar)]
fn fused_cg_impl<const A: bool, const B: bool, const C: bool>(
    token: Token,
    rgba: &[u8],
) -> FusedResult {
    if !(A || B || C) {
        // Base case: no live checks, nothing to do. The caller already
        // recorded `false` for everything that flipped on the way down.
        return FusedResult::default();
    }

    let alpha_mask = u8x64::from_array(token, ALPHA_MASK_RGBA8);
    let rgb_delta_mask = u8x64::from_array(token, RGB_DELTA_MASK_RGBA8);
    let zero = u8x64::splat(token, 0);
    let opaque = u8x64::splat(token, 0xFF);

    let len = rgba.len();
    let mut i = 0;
    // With B alive we need the shifted load; otherwise just 64. The
    // ternary is on a const, so it const-propagates.
    let bound = if B { 65 } else { 64 };

    while i + bound <= len {
        let chunk0: &[u8; 64] = (&rgba[i..i + 64]).try_into().unwrap();
        let v0 = u8x64::load(token, chunk0);

        // Collect all flips for THIS chunk before recursing. Otherwise a
        // single chunk that breaks two checks would only register one.
        let mut next_a = A;
        let mut next_b = B;
        let mut next_c = C;

        if A {
            let bad = v0.simd_ne(opaque) & alpha_mask;
            if bad.any_true() {
                next_a = false;
            }
        }
        if C {
            let bad = v0.simd_ne(zero) & v0.simd_ne(opaque) & alpha_mask;
            if bad.any_true() {
                next_c = false;
            }
        }
        if B {
            let chunk1: &[u8; 64] = (&rgba[i + 1..i + 65]).try_into().unwrap();
            let v1 = u8x64::load(token, chunk1);
            let bad = v0.simd_ne(v1) & rgb_delta_mask;
            if bad.any_true() {
                next_b = false;
            }
        }

        // If nothing flipped this iteration, advance.
        if next_a == A && next_b == B && next_c == C {
            i += 64;
            continue;
        }

        // Something flipped. Recurse into the matching specialization
        // with the remainder. Each arm wires up the resulting fields:
        // for any flipped check, the field is `false`; for still-live
        // checks, the recursion's result carries through. The compiler
        // dead-code-eliminates arms unreachable from this specialization.
        let rest = &rgba[i + 64..];
        return match (next_a, next_b, next_c) {
            (true, true, true) => unreachable!(),
            (true, true, false) => {
                let s = fused_cg::<true, true, false>(rest);
                FusedResult {
                    is_opaque: s.is_opaque,
                    is_grayscale: s.is_grayscale,
                    is_binary_alpha: false,
                }
            }
            (true, false, true) => {
                let s = fused_cg::<true, false, true>(rest);
                FusedResult {
                    is_opaque: s.is_opaque,
                    is_grayscale: false,
                    is_binary_alpha: s.is_binary_alpha,
                }
            }
            (false, true, true) => {
                let s = fused_cg::<false, true, true>(rest);
                FusedResult {
                    is_opaque: false,
                    is_grayscale: s.is_grayscale,
                    is_binary_alpha: s.is_binary_alpha,
                }
            }
            (true, false, false) => {
                let s = fused_cg::<true, false, false>(rest);
                FusedResult {
                    is_opaque: s.is_opaque,
                    is_grayscale: false,
                    is_binary_alpha: false,
                }
            }
            (false, true, false) => {
                let s = fused_cg::<false, true, false>(rest);
                FusedResult {
                    is_opaque: false,
                    is_grayscale: s.is_grayscale,
                    is_binary_alpha: false,
                }
            }
            (false, false, true) => {
                let s = fused_cg::<false, false, true>(rest);
                FusedResult {
                    is_opaque: false,
                    is_grayscale: false,
                    is_binary_alpha: s.is_binary_alpha,
                }
            }
            (false, false, false) => FusedResult::default(),
        };
    }

    // Scalar tail. The const params collapse the per-pixel branches.
    let mut o = A;
    let mut g = B;
    let mut c = C;
    while i + 4 <= len && (o | g | c) {
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
        if C && c && a != 0 && a != 255 {
            c = false;
        }
        i += 4;
    }

    FusedResult {
        is_opaque: o,
        is_grayscale: g,
        is_binary_alpha: c,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    fn scalar_is_opaque(rgba: &[u8]) -> bool {
        rgba.chunks_exact(4).all(|p| p[3] == 255)
    }
    fn scalar_is_grayscale_rgba8(rgba: &[u8]) -> bool {
        rgba.chunks_exact(4).all(|p| p[0] == p[1] && p[1] == p[2])
    }
    fn scalar_alpha_binary(rgba: &[u8]) -> bool {
        rgba.chunks_exact(4).all(|p| p[3] == 0 || p[3] == 255)
    }
    fn scalar_is_grayscale_rgb8(rgb: &[u8]) -> bool {
        rgb.chunks_exact(3).all(|p| p[0] == p[1] && p[1] == p[2])
    }
    fn scalar_bit_replication(be: &[u8]) -> bool {
        be.chunks_exact(2).all(|p| p[0] == p[1])
    }

    // ── Tiny inline synthetics — explicit true/false fixtures per
    // predicate. These run at every dispatch tier via
    // `for_each_token_permutation`, so every code path (V4x AVX-512,
    // V4 AVX2, V3 SSE4.2, NEON, WASM128, scalar) is exercised on the
    // exact same byte sequence. Each tier should agree with the named
    // expectation.

    fn run_at_all_tiers(label: &str, body: impl Fn()) {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| body());
        eprintln!("{label}: {report}");
    }

    // is_opaque_rgba8 ────────────────────────────────────────────────

    #[test]
    fn is_opaque_true_one_pixel() {
        run_at_all_tiers("opaque_true_1px", || {
            // 1 RGBA pixel, alpha = 255
            assert!(super::is_opaque_rgba8(&[100, 50, 25, 255]));
        });
    }

    #[test]
    fn is_opaque_true_four_pixels() {
        run_at_all_tiers("opaque_true_4px", || {
            // 4 pixels, all alpha = 255
            assert!(super::is_opaque_rgba8(&[
                10, 20, 30, 255, //
                40, 50, 60, 255, //
                70, 80, 90, 255, //
                100, 110, 120, 255,
            ]));
        });
    }

    #[test]
    fn is_opaque_false_alpha_zero_at_first() {
        run_at_all_tiers("opaque_false_first_alpha_0", || {
            assert!(!super::is_opaque_rgba8(&[
                10, 20, 30, 0, //
                40, 50, 60, 255,
            ]));
        });
    }

    #[test]
    fn is_opaque_false_alpha_254_at_last() {
        run_at_all_tiers("opaque_false_last_alpha_254", || {
            assert!(!super::is_opaque_rgba8(&[
                10, 20, 30, 255, //
                40, 50, 60, 254,
            ]));
        });
    }

    #[test]
    fn is_opaque_false_alpha_128_in_middle_of_simd_chunk() {
        run_at_all_tiers("opaque_false_alpha_128_pixel_5", || {
            // 16 pixels = exactly one 64-byte SIMD chunk.
            // Trips on pixel 5 with alpha=128 (high bit set, harder reduction).
            let mut v = [0u8; 64];
            for i in 0..16 {
                v[i * 4] = (i * 7) as u8;
                v[i * 4 + 1] = (i * 7) as u8;
                v[i * 4 + 2] = (i * 7) as u8;
                v[i * 4 + 3] = 255;
            }
            v[5 * 4 + 3] = 128;
            assert!(!super::is_opaque_rgba8(&v));
        });
    }

    #[test]
    fn is_opaque_empty_buffer_is_true() {
        run_at_all_tiers("opaque_empty", || {
            assert!(super::is_opaque_rgba8(&[]));
        });
    }

    // is_grayscale_rgba8 ─────────────────────────────────────────────

    #[test]
    fn grayscale_rgba8_true_one_pixel() {
        run_at_all_tiers("gray_rgba_true_1px", || {
            // R == G == B = 128
            assert!(super::is_grayscale_rgba8(&[128, 128, 128, 255]));
        });
    }

    #[test]
    fn grayscale_rgba8_true_with_varying_alpha() {
        run_at_all_tiers("gray_rgba_true_var_alpha", || {
            // Alpha varies but RGB always equal — still grayscale.
            assert!(super::is_grayscale_rgba8(&[
                42, 42, 42, 255, //
                42, 42, 42, 0, //
                42, 42, 42, 64,
            ]));
        });
    }

    #[test]
    fn grayscale_rgba8_false_first_pixel_g_differs() {
        run_at_all_tiers("gray_rgba_false_first_g", || {
            // R=10, G=11, B=10 → not grayscale
            assert!(!super::is_grayscale_rgba8(&[10, 11, 10, 255]));
        });
    }

    #[test]
    fn grayscale_rgba8_false_last_pixel_b_differs() {
        run_at_all_tiers("gray_rgba_false_last_b", || {
            assert!(!super::is_grayscale_rgba8(&[
                10, 10, 10, 255, //
                20, 20, 21, 255,
            ]));
        });
    }

    #[test]
    fn grayscale_rgba8_false_off_by_one_in_simd_chunk() {
        run_at_all_tiers("gray_rgba_false_off_by_one", || {
            // 17 pixels (68 bytes) — first SIMD chunk catches it, then 1-pixel tail.
            let mut v = [0u8; 68];
            for i in 0..17 {
                let g = (i * 13) as u8;
                v[i * 4] = g;
                v[i * 4 + 1] = g;
                v[i * 4 + 2] = g;
                v[i * 4 + 3] = 255;
            }
            v[7 * 4 + 1] = v[7 * 4].wrapping_add(1); // G off by 1
            assert!(!super::is_grayscale_rgba8(&v));
        });
    }

    // alpha_is_binary_rgba8 ──────────────────────────────────────────

    #[test]
    fn alpha_binary_true_all_255() {
        run_at_all_tiers("ab_true_all_255", || {
            assert!(super::alpha_is_binary_rgba8(&[
                10, 20, 30, 255, //
                40, 50, 60, 255,
            ]));
        });
    }

    #[test]
    fn alpha_binary_true_mix_0_and_255() {
        run_at_all_tiers("ab_true_mix", || {
            assert!(super::alpha_is_binary_rgba8(&[
                10, 20, 30, 0, //
                40, 50, 60, 255, //
                70, 80, 90, 0,
            ]));
        });
    }

    #[test]
    fn alpha_binary_false_alpha_1() {
        run_at_all_tiers("ab_false_alpha_1", || {
            // alpha = 1 is the smallest non-zero non-255 — easy to miss
            // if the predicate uses a sloppy mask.
            assert!(!super::alpha_is_binary_rgba8(&[10, 20, 30, 1]));
        });
    }

    #[test]
    fn alpha_binary_false_alpha_254() {
        run_at_all_tiers("ab_false_alpha_254", || {
            // alpha = 254 — boundary case (1 below 255).
            assert!(!super::alpha_is_binary_rgba8(&[10, 20, 30, 254]));
        });
    }

    #[test]
    fn alpha_binary_false_alpha_128_mid_simd() {
        run_at_all_tiers("ab_false_128_mid", || {
            // 32 pixels (128 bytes) — alpha 128 at pixel 9 inside 2nd SIMD chunk.
            let mut v = [0u8; 128];
            for i in 0..32 {
                v[i * 4] = (i * 5) as u8;
                v[i * 4 + 3] = if i & 1 == 0 { 0 } else { 255 };
            }
            v[9 * 4 + 3] = 128;
            assert!(!super::alpha_is_binary_rgba8(&v));
        });
    }

    // is_grayscale_rgb8 ──────────────────────────────────────────────

    #[test]
    fn grayscale_rgb8_true_one_pixel() {
        run_at_all_tiers("gray_rgb_true_1px", || {
            assert!(super::is_grayscale_rgb8(&[42, 42, 42]));
        });
    }

    #[test]
    fn grayscale_rgb8_true_three_pixels() {
        run_at_all_tiers("gray_rgb_true_3px", || {
            assert!(super::is_grayscale_rgb8(&[10, 10, 10, 20, 20, 20, 30, 30, 30]));
        });
    }

    #[test]
    fn grayscale_rgb8_false_first_pixel() {
        run_at_all_tiers("gray_rgb_false_first", || {
            assert!(!super::is_grayscale_rgb8(&[10, 11, 10]));
        });
    }

    #[test]
    fn grayscale_rgb8_false_in_supernode_phase_1() {
        run_at_all_tiers("gray_rgb_false_phase1", || {
            // 64 RGB pixels = 192 bytes, exactly one super-chunk. Plant a
            // mismatch in phase-1 territory (bytes 64..128).
            let mut v = [0u8; 192];
            for i in 0..64 {
                let g = (i * 3) as u8;
                v[i * 3] = g;
                v[i * 3 + 1] = g;
                v[i * 3 + 2] = g;
            }
            v[30 * 3 + 1] = v[30 * 3].wrapping_add(1);
            assert!(!super::is_grayscale_rgb8(&v));
        });
    }

    #[test]
    fn grayscale_rgb8_false_in_phase_2() {
        run_at_all_tiers("gray_rgb_false_phase2", || {
            let mut v = [0u8; 192];
            for i in 0..64 {
                let g = (i * 3) as u8;
                v[i * 3] = g;
                v[i * 3 + 1] = g;
                v[i * 3 + 2] = g;
            }
            v[50 * 3 + 2] = v[50 * 3].wrapping_add(7);
            assert!(!super::is_grayscale_rgb8(&v));
        });
    }

    // bit_replication_lossless_be16 ──────────────────────────────────

    #[test]
    fn bit_repl_true_one_pair() {
        run_at_all_tiers("repl_true_1pair", || {
            // [0x12, 0x12] — replicated, lossless 16→8 to 0x12
            assert!(super::bit_replication_lossless_be16(&[0x12, 0x12]));
        });
    }

    #[test]
    fn bit_repl_true_pure_white_black() {
        run_at_all_tiers("repl_true_white_black", || {
            // 0xFFFF and 0x0000 are both bit-replicated extremes.
            assert!(super::bit_replication_lossless_be16(&[
                0xFF, 0xFF, 0x00, 0x00, 0x80, 0x80,
            ]));
        });
    }

    #[test]
    fn bit_repl_false_low_byte_zero() {
        run_at_all_tiers("repl_false_low_zero", || {
            // [0x12, 0x00] — legacy "low byte zero", NOT bit-replicated.
            // PNG decoder would reconstruct 0x12 → 0x1212 (not 0x1200).
            assert!(!super::bit_replication_lossless_be16(&[0x12, 0x00]));
        });
    }

    #[test]
    fn bit_repl_false_off_by_one() {
        run_at_all_tiers("repl_false_off_by_one", || {
            // hi=0x55, lo=0x56 — close but not equal.
            assert!(!super::bit_replication_lossless_be16(&[0x55, 0x56]));
        });
    }

    #[test]
    fn bit_repl_false_in_simd_chunk() {
        run_at_all_tiers("repl_false_simd_chunk", || {
            // 32 pairs (64 bytes) — break pair 17 inside the SIMD chunk.
            let mut v = [0u8; 64];
            for i in 0..32 {
                let b = (i * 17) as u8;
                v[i * 2] = b;
                v[i * 2 + 1] = b;
            }
            v[17 * 2 + 1] = v[17 * 2].wrapping_add(1);
            assert!(!super::bit_replication_lossless_be16(&v));
        });
    }

    #[test]
    fn bit_repl_empty_buffer_is_true() {
        run_at_all_tiers("repl_empty", || {
            assert!(super::bit_replication_lossless_be16(&[]));
        });
    }

    #[test]
    fn bit_repl_false_fefe_then_fe00() {
        // Adjacent pairs [0xFE, 0xFE] and [0xFE, 0x00] — the first
        // pair passes, the second fails. The across-pair lane (byte 1
        // vs byte 2 = 0xFE vs 0xFE) would also be "equal" — the mask
        // must keep that out so the failing pair still fires.
        run_at_all_tiers("repl_false_fefe_then_fe00", || {
            assert!(!super::bit_replication_lossless_be16(&[
                0xFE, 0xFE, // pair 0: ok
                0xFE, 0x00, // pair 1: NOT ok (0xFE != 0x00)
            ]));
        });
    }

    #[test]
    fn bit_repl_true_fefe_then_fcfc() {
        // Mirror image of the previous test: distinct values across
        // adjacent pairs but each pair internally satisfies hi==lo.
        // The across-pair lane sees `0xFE != 0xFC` and would fail the
        // simd_ne check — the mask must zero out that lane.
        run_at_all_tiers("repl_true_fefe_then_fcfc", || {
            assert!(super::bit_replication_lossless_be16(&[
                0xFE, 0xFE, 0xFC, 0xFC, // both reducible
            ]));
        });
    }

    // ── U16-typed bit-replication tests ──────────────────────────────

    #[test]
    fn bit_repl_u16_true() {
        run_at_all_tiers("repl_u16_true", || {
            // 0xFEFE, 0x1212, 0x0000, 0xFFFF — all bit-replicated.
            assert!(super::bit_replication_lossless_u16(&[
                0xFEFE, 0x1212, 0x0000, 0xFFFF,
            ]));
        });
    }

    #[test]
    fn bit_repl_u16_false() {
        run_at_all_tiers("repl_u16_false", || {
            // 0xFE00 — hi != lo.
            assert!(!super::bit_replication_lossless_u16(&[0xFE00]));
        });
    }

    #[test]
    fn bit_repl_u16_endian_agnostic() {
        // Same logical sample 0x1212 round-trips no matter how the
        // bytes were stored (BE or LE on disk).
        run_at_all_tiers("repl_u16_endian", || {
            assert!(super::bit_replication_lossless_u16(&[0x1212]));
            // 0x12 in either byte position passes; 0x12_00 does not.
            assert!(!super::bit_replication_lossless_u16(&[0x1200]));
            assert!(!super::bit_replication_lossless_u16(&[0x0012]));
        });
    }

    // ── U16 channel-layout predicate tests ───────────────────────────

    #[test]
    fn rgba16_opaque_true() {
        run_at_all_tiers("rgba16_opaque_true", || {
            assert!(super::is_opaque_rgba16(&[
                0x1234, 0x5678, 0x9ABC, 0xFFFF, // alpha = 65535
                0x4321, 0x8765, 0xCBA9, 0xFFFF,
            ]));
        });
    }

    #[test]
    fn rgba16_opaque_false_alpha_lt_max() {
        run_at_all_tiers("rgba16_opaque_false", || {
            assert!(!super::is_opaque_rgba16(&[
                0x1234, 0x5678, 0x9ABC, 0xFFFE, // alpha = 65534
            ]));
        });
    }

    #[test]
    fn rgba16_grayscale_true() {
        run_at_all_tiers("rgba16_gray_true", || {
            assert!(super::is_grayscale_rgba16(&[
                0x4242, 0x4242, 0x4242, 0x8000, //
                0x9999, 0x9999, 0x9999, 0x0000,
            ]));
        });
    }

    #[test]
    fn rgba16_grayscale_false() {
        run_at_all_tiers("rgba16_gray_false", || {
            assert!(!super::is_grayscale_rgba16(&[
                0x4242, 0x4243, 0x4242, 0x8000, // G != R by 1
            ]));
        });
    }

    #[test]
    fn rgba16_alpha_binary_true() {
        run_at_all_tiers("rgba16_ab_true", || {
            assert!(super::alpha_is_binary_rgba16(&[
                0x1234, 0x5678, 0x9ABC, 0x0000, // alpha = 0
                0x4321, 0x8765, 0xCBA9, 0xFFFF, // alpha = 65535
            ]));
        });
    }

    #[test]
    fn rgba16_alpha_binary_false_mid_alpha() {
        run_at_all_tiers("rgba16_ab_false", || {
            assert!(!super::alpha_is_binary_rgba16(&[
                0x1234, 0x5678, 0x9ABC, 0x8000, // alpha = 32768 (not extreme)
            ]));
        });
    }

    #[test]
    fn rgb16_grayscale_true() {
        run_at_all_tiers("rgb16_gray_true", || {
            assert!(super::is_grayscale_rgb16(&[
                0x4242, 0x4242, 0x4242, //
                0x9999, 0x9999, 0x9999, //
                0x0000, 0x0000, 0x0000,
            ]));
        });
    }

    #[test]
    fn rgb16_grayscale_false() {
        run_at_all_tiers("rgb16_gray_false", || {
            assert!(!super::is_grayscale_rgb16(&[
                0x4242, 0x4243, 0x4242, // G off by 1
            ]));
        });
    }

    #[test]
    fn rgb16_grayscale_false_in_simd_super_chunk() {
        // Verify phase-rotated masks across the 96-element super-chunk.
        run_at_all_tiers("rgb16_gray_false_super", || {
            // 32 RGB pixels = 96 u16s (one full super-chunk).
            let mut v = vec![0u16; 96];
            for i in 0..32 {
                let g = (i as u16) * 0x0202;
                v[i * 3] = g;
                v[i * 3 + 1] = g;
                v[i * 3 + 2] = g;
            }
            v[20 * 3 + 1] = v[20 * 3].wrapping_add(1); // pixel 20: G != R
            assert!(!super::is_grayscale_rgb16(&v));
        });
    }

    // ── GrayAlpha8 / GrayAlpha16 tests ───────────────────────────────

    #[test]
    fn ga8_opaque_true() {
        run_at_all_tiers("ga8_opaque_true", || {
            assert!(super::is_opaque_ga8(&[
                100, 255, // pixel 0
                50, 255, // pixel 1
                200, 255,
            ]));
        });
    }

    #[test]
    fn ga8_opaque_false() {
        run_at_all_tiers("ga8_opaque_false", || {
            assert!(!super::is_opaque_ga8(&[100, 255, 50, 128, 200, 255]));
        });
    }

    #[test]
    fn ga8_alpha_binary_true_mix() {
        run_at_all_tiers("ga8_ab_true", || {
            assert!(super::alpha_is_binary_ga8(&[
                100, 0, 50, 255, 200, 0, 75, 255,
            ]));
        });
    }

    #[test]
    fn ga8_alpha_binary_false() {
        run_at_all_tiers("ga8_ab_false", || {
            assert!(!super::alpha_is_binary_ga8(&[100, 128]));
        });
    }

    #[test]
    fn ga16_opaque_true() {
        run_at_all_tiers("ga16_opaque_true", || {
            assert!(super::is_opaque_ga16(&[0x1234, 0xFFFF, 0x5678, 0xFFFF]));
        });
    }

    #[test]
    fn ga16_opaque_false() {
        run_at_all_tiers("ga16_opaque_false", || {
            assert!(!super::is_opaque_ga16(&[0x1234, 0xFFFE]));
        });
    }

    #[test]
    fn ga16_alpha_binary_true_mix() {
        run_at_all_tiers("ga16_ab_true", || {
            assert!(super::alpha_is_binary_ga16(&[
                0x1234, 0x0000, 0x5678, 0xFFFF, 0x9ABC, 0x0000,
            ]));
        });
    }

    #[test]
    fn ga16_alpha_binary_false_mid() {
        run_at_all_tiers("ga16_ab_false", || {
            assert!(!super::alpha_is_binary_ga16(&[0x1234, 0x8000]));
        });
    }

    // ── Fused predicate tests ────────────────────────────────────────

    fn scalar_fused(rgba: &[u8], req: super::FusedRequest) -> super::FusedResult {
        let mut o = req.check_opaque;
        let mut g = req.check_grayscale;
        let mut b = req.check_binary_alpha;
        for chunk in rgba.chunks_exact(4) {
            if o && chunk[3] != 255 {
                o = false;
            }
            if g && (chunk[0] != chunk[1] || chunk[1] != chunk[2]) {
                g = false;
            }
            if b && chunk[3] != 0 && chunk[3] != 255 {
                b = false;
            }
        }
        super::FusedResult {
            is_opaque: o,
            is_grayscale: g,
            is_binary_alpha: b,
        }
    }

    fn run_fused_match_test(label: &str, rgba: &[u8], req: super::FusedRequest) {
        let expected = scalar_fused(rgba, req);
        run_at_all_tiers(&format!("fused_runtime_{label}"), || {
            let got = super::fused_predicates_rgba8(rgba, req);
            assert_eq!(
                got, expected,
                "runtime mismatch label={label} req={req:?} expected={expected:?}"
            );
        });
        run_at_all_tiers(&format!("fused_cg_{label}"), || {
            let got = super::fused_predicates_rgba8_cg(rgba, req);
            assert_eq!(
                got, expected,
                "const-generic mismatch label={label} req={req:?} expected={expected:?}"
            );
        });
    }

    #[test]
    fn fused_pure_grayscale_opaque() {
        // Every pixel R=G=B, alpha=255 — all three checks should pass.
        let v = [
            10, 10, 10, 255, //
            20, 20, 20, 255, //
            30, 30, 30, 255,
        ];
        run_fused_match_test("gray_opaque", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_colorful_with_binary_alpha() {
        // Not grayscale (R != G), alpha is binary (mix of 0 and 255).
        let v = [
            10, 20, 30, 0, //
            40, 50, 60, 255, //
            70, 80, 90, 0,
        ];
        run_fused_match_test("color_binary", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_first_pixel_breaks_all_three() {
        // Pixel 0: not opaque (alpha=128), not gray (R!=G), not binary (alpha=128).
        // Subsequent pixels are fine — confirms early-exit on first chunk.
        let mut v = [255u8; 256]; // 64 RGBA pixels, defaults R=G=B=A=255 (gray, opaque, binary)
        v[0] = 0;
        v[1] = 1;
        v[2] = 2;
        v[3] = 128;
        run_fused_match_test("first_break_all", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_subset_only_opaque() {
        // Even if grayscale would be false, we don't ask — return reflects that.
        let v = [10, 20, 30, 255, 40, 50, 60, 255]; // not gray, but opaque
        let req = super::FusedRequest {
            check_opaque: true,
            check_grayscale: false,
            check_binary_alpha: false,
        };
        run_fused_match_test("subset_opaque_only", &v, req);
    }

    #[test]
    fn fused_subset_only_grayscale_with_simd_chunk() {
        // 64 pixels (256 bytes) — full SIMD chunk path.
        let mut v = [0u8; 256];
        for i in 0..64 {
            let g = (i * 3) as u8;
            v[i * 4] = g;
            v[i * 4 + 1] = g;
            v[i * 4 + 2] = g;
            v[i * 4 + 3] = if i & 1 == 0 { 0 } else { 200 };
        }
        let req = super::FusedRequest {
            check_opaque: false,
            check_grayscale: true,
            check_binary_alpha: false,
        };
        run_fused_match_test("subset_gray_only_64px", &v, req);
    }

    #[test]
    fn fused_no_checks_requested() {
        // All-false request → all-false result, no work done.
        let v = [10, 20, 30, 128, 40, 50, 60, 255];
        let req = super::FusedRequest::default();
        run_fused_match_test("no_checks", &v, req);
    }

    #[test]
    fn fused_grayscale_breaks_at_pixel_5_in_simd_chunk() {
        // 16 pixels, exactly one SIMD chunk of 64 bytes — verifies the
        // const-generic path correctly recurses on B-flip during the
        // SIMD sweep (not just in scalar tail).
        let mut v = [0u8; 64];
        for i in 0..16 {
            let g = (i * 11) as u8;
            v[i * 4] = g;
            v[i * 4 + 1] = g;
            v[i * 4 + 2] = g;
            v[i * 4 + 3] = 255;
        }
        v[5 * 4 + 1] = v[5 * 4].wrapping_add(7); // G != R at pixel 5
        run_fused_match_test("gray_break_chunk", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_opaque_breaks_at_simd_boundary() {
        // 32 pixels (128 bytes = 2 SIMD chunks). Opacity breaks in chunk 2,
        // pixel 20 (alpha=64). Grayscale stays true.
        let mut v = [0u8; 128];
        for i in 0..32 {
            let g = (i * 5) as u8;
            v[i * 4] = g;
            v[i * 4 + 1] = g;
            v[i * 4 + 2] = g;
            v[i * 4 + 3] = 255;
        }
        v[20 * 4 + 3] = 64; // mid-alpha
        run_fused_match_test("opaque_break_chunk2", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_all_three_break_at_different_pixels() {
        // Pixel 3: alpha=128 → opacity AND binary alpha both flip
        // Pixel 7: G != R → grayscale flips
        // Verifies multiple flips don't corrupt each other in either variant.
        let mut v = [0u8; 64];
        for i in 0..16 {
            let g = (i * 3) as u8;
            v[i * 4] = g;
            v[i * 4 + 1] = g;
            v[i * 4 + 2] = g;
            v[i * 4 + 3] = if i & 1 == 0 { 0 } else { 255 };
        }
        v[3 * 4 + 3] = 128; // alpha not in {0,255}
        v[7 * 4 + 1] = v[7 * 4].wrapping_add(1);
        run_fused_match_test("multi_break", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_short_buffer_only_scalar_tail() {
        // 8 pixels, no SIMD chunk — exercises tail logic in both variants.
        let v = [
            10, 10, 10, 255, //
            20, 20, 20, 0, //
            30, 30, 30, 255, //
            40, 41, 40, 0, // pixel 3: not gray
            50, 50, 50, 255, //
            60, 60, 60, 200, // pixel 5: alpha not binary
            70, 70, 70, 0, //
            80, 80, 80, 255,
        ];
        run_fused_match_test("short_tail_only", &v, super::FusedRequest::all());
    }

    #[test]
    fn fused_runtime_and_cg_agree_on_random_inputs() {
        // Simple LCG for determinism; not a security-sensitive RNG.
        let mut state: u32 = 0xC0FFEE12;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        for trial in 0..10 {
            let n_pixels = 100 + (next() % 4096) as usize;
            let mut v = Vec::with_capacity(n_pixels * 4);
            for _ in 0..n_pixels {
                let r = next() as u8;
                let g_chance = (next() % 4 == 0) as u8;
                let g = if g_chance == 0 { r } else { next() as u8 };
                let b = if g_chance == 0 { r } else { next() as u8 };
                let a = match next() % 5 {
                    0 => 0,
                    1 => 255,
                    2 => 128,
                    _ => next() as u8,
                };
                v.extend_from_slice(&[r, g, b, a]);
            }
            let req = super::FusedRequest::all();
            let runtime = super::fused_predicates_rgba8(&v, req);
            let cg = super::fused_predicates_rgba8_cg(&v, req);
            let scalar = scalar_fused(&v, req);
            assert_eq!(
                runtime, scalar,
                "trial {trial}: runtime != scalar  n_pixels={n_pixels}"
            );
            assert_eq!(
                cg, scalar,
                "trial {trial}: cg != scalar  n_pixels={n_pixels}"
            );
        }
    }


    fn rgba_pattern(n_pixels: usize, mutate: impl Fn(usize, &mut [u8; 4])) -> Vec<u8> {
        let mut v = Vec::with_capacity(n_pixels * 4);
        for i in 0..n_pixels {
            let mut p = [
                (i * 7 + 3) as u8,
                (i * 7 + 3) as u8,
                (i * 7 + 3) as u8,
                255,
            ];
            mutate(i, &mut p);
            v.extend_from_slice(&p);
        }
        v
    }

    #[test]
    fn opaque_predicate_matches_scalar_all_tiers() {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for &n in &[0usize, 1, 4, 15, 16, 17, 31, 64, 200, 1024, 4099] {
                let v = rgba_pattern(n, |_, _| {});
                assert_eq!(super::is_opaque_rgba8(&v), scalar_is_opaque(&v), "n={n} all-opaque");
                if n > 5 {
                    let mut v = rgba_pattern(n, |_, _| {});
                    v[5 * 4 + 3] = 128;
                    assert_eq!(super::is_opaque_rgba8(&v), scalar_is_opaque(&v), "n={n} pixel 5 alpha=128");
                }
                // Non-opaque at the very last pixel
                if n > 0 {
                    let mut v = rgba_pattern(n, |_, _| {});
                    v[(n - 1) * 4 + 3] = 200;
                    assert_eq!(super::is_opaque_rgba8(&v), scalar_is_opaque(&v), "n={n} last pixel alpha=200");
                }
            }
        });
        eprintln!("is_opaque_rgba8: {report}");
    }

    #[test]
    fn grayscale_rgba8_matches_scalar_all_tiers() {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for &n in &[0usize, 1, 4, 16, 17, 64, 200, 4099] {
                let v = rgba_pattern(n, |_, _| {});
                assert_eq!(super::is_grayscale_rgba8(&v), scalar_is_grayscale_rgba8(&v), "n={n} all-gray");
                if n > 5 {
                    let mut v = rgba_pattern(n, |_, _| {});
                    v[5 * 4] = 1;
                    v[5 * 4 + 1] = 2;
                    v[5 * 4 + 2] = 3;
                    assert_eq!(super::is_grayscale_rgba8(&v), scalar_is_grayscale_rgba8(&v), "n={n} colorful pixel 5");
                }
                // Off-by-one: only G differs from R by 1
                if n > 5 {
                    let mut v = rgba_pattern(n, |_, _| {});
                    v[5 * 4 + 1] = v[5 * 4].wrapping_add(1);
                    assert_eq!(super::is_grayscale_rgba8(&v), scalar_is_grayscale_rgba8(&v), "n={n} g=r+1 at pixel 5");
                }
            }
        });
        eprintln!("is_grayscale_rgba8: {report}");
    }

    #[test]
    fn alpha_binary_matches_scalar_all_tiers() {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for &n in &[0usize, 1, 4, 16, 64, 200, 4099] {
                let v = rgba_pattern(n, |i, p| p[3] = if i & 1 == 0 { 0 } else { 255 });
                assert_eq!(super::alpha_is_binary_rgba8(&v), scalar_alpha_binary(&v), "n={n} alternating 0/255");
                if n > 5 {
                    let mut v = rgba_pattern(n, |i, p| p[3] = if i & 1 == 0 { 0 } else { 255 });
                    v[5 * 4 + 3] = 128;
                    assert_eq!(super::alpha_is_binary_rgba8(&v), scalar_alpha_binary(&v), "n={n} alpha 128 at 5");
                }
                if n > 5 {
                    let mut v = rgba_pattern(n, |_, _| {});
                    v[5 * 4 + 3] = 1; // very small but nonzero
                    assert_eq!(super::alpha_is_binary_rgba8(&v), scalar_alpha_binary(&v), "n={n} alpha 1 at 5");
                }
            }
        });
        eprintln!("alpha_is_binary_rgba8: {report}");
    }

    #[test]
    fn grayscale_rgb8_matches_scalar_all_tiers() {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for &n in &[0usize, 1, 3, 16, 64, 200, 4099] {
                let mut v = Vec::with_capacity(n * 3);
                for i in 0..n {
                    let g = (i * 7 + 3) as u8;
                    v.extend_from_slice(&[g, g, g]);
                }
                assert_eq!(super::is_grayscale_rgb8(&v), scalar_is_grayscale_rgb8(&v), "n={n} all-gray");
                if n > 80 {
                    let mut v2 = v.clone();
                    v2[80 * 3 + 1] = v2[80 * 3].wrapping_add(1);
                    assert_eq!(super::is_grayscale_rgb8(&v2), scalar_is_grayscale_rgb8(&v2), "n={n} pixel 80 g+=1");
                }
            }
        });
        eprintln!("is_grayscale_rgb8: {report}");
    }

    #[test]
    fn bit_replication_matches_scalar_all_tiers() {
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            for &n in &[0usize, 2, 4, 16, 64, 200, 4096] {
                let mut v = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let b = (i * 11 + 7) as u8;
                    v.extend_from_slice(&[b, b]);
                }
                assert_eq!(
                    super::bit_replication_lossless_be16(&v),
                    scalar_bit_replication(&v),
                    "n={n} replicated"
                );
                if n > 30 {
                    v[30 * 2 + 1] = v[30 * 2].wrapping_add(1);
                    assert_eq!(
                        super::bit_replication_lossless_be16(&v),
                        scalar_bit_replication(&v),
                        "n={n} broken at 30"
                    );
                }
            }
        });
        eprintln!("bit_replication_lossless_be16: {report}");
    }
}
