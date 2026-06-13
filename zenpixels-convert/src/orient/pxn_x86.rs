use super::{Orientation, PixelSlice, PixelSliceMut, inverse_flips};
use archmage::prelude::*;

/// Scalar `forward_map` scatter for an arbitrary pixel rectangle, against
/// flat strided bytes. Shared edge handler for every x86 kernel here.
#[allow(clippy::too_many_arguments)]
pub(super) fn scalar_rect(
    sbytes: &[u8],
    sstride: usize,
    dbytes: &mut [u8],
    dstride: usize,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
) {
    for y in y0..y1 {
        for x in x0..x1 {
            let (dx, dy) = orientation.forward_map(x, y, w, h);
            let s = y as usize * sstride + x as usize * bpp;
            let d = dy as usize * dstride + dx as usize * bpp;
            dbytes[d..d + bpp].copy_from_slice(&sbytes[s..s + bpp]);
        }
    }
}

/// Scalar `forward_map` cleanup for the two edge strips a tiled SIMD pass
/// leaves uncovered: the right columns (`[full_w, w) × [0, full_h)`) and the
/// bottom rows (`[0, w) × [full_h, h)`, which also covers the bottom-right
/// corner). The strips are disjoint. Every full-width kernel here ends with it.
#[allow(clippy::too_many_arguments)]
pub(super) fn scalar_edges(
    sbytes: &[u8],
    sstride: usize,
    dbytes: &mut [u8],
    dstride: usize,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
    full_w: u32,
    full_h: u32,
) {
    // Right columns, then bottom rows (+ bottom-right corner).
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        bpp,
        full_w,
        w,
        0,
        full_h,
    );
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        bpp,
        0,
        w,
        full_h,
        h,
    );
}

/// Scalar cleanup for the guard-trimmed columns of the final-row band: the
/// columns the slop-carrying kernels (3/6/12 bpp) skip via their `guard_w`
/// limit because a register load there would overrun the buffer. Only the band
/// that reaches the image's last row (`full_h == h`) is ever trimmed, so this
/// is a no-op for every other shape; `band_h` is that band's row height. Call
/// it *before* [`scalar_edges`] — the regions are disjoint (this one's columns
/// are left of `full_w`, and `full_h == h` makes the bottom strip empty).
#[allow(clippy::too_many_arguments)]
pub(super) fn scalar_last_band_guard(
    sbytes: &[u8],
    sstride: usize,
    dbytes: &mut [u8],
    dstride: usize,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
    full_w: u32,
    full_h: u32,
    guard_w: u32,
    band_h: u32,
) {
    if guard_w < full_w && full_h == h && h >= band_h {
        scalar_rect(
            sbytes,
            sstride,
            dbytes,
            dstride,
            orientation,
            w,
            h,
            bpp,
            guard_w,
            full_w,
            h - band_h,
            h,
        );
    }
}

/// 16×16 gray8 tiles, AVX2 (ermig1979/Simd `1x16x16` network): 8 ymm
/// registers pair rows r and r+8 in their lanes; three rounds of
/// stride-4 byte unpacks transpose both halves at once; a qword permute
/// makes each register two consecutive destination rows. Half the op
/// count of the two-cascade xmm shape this replaces (which measured 2×
/// behind the C++ original). Full-width row-band sweep — at 1 bpp the
/// dst-line working set is L2-resident and column stripes only add
/// source-line straddle (measured).
#[arcane(import_intrinsics)]
pub(super) fn transpose1_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    const REV16B: [u8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    let rev = _mm256_broadcastsi128_si256(_mm_loadu_si128(&REV16B));

    let full_w = w & !15;
    let full_h = h & !15;
    let ntiles = full_w / 16;
    // Bands descend under flip_sy so dst columns ascend — the same
    // forward-walking store order the C++ original gets via negative
    // destination strides.
    let nbands = full_h / 16;
    for bandi in 0..nbands {
        let sy = if flip_sy {
            (nbands - 1 - bandi) * 16
        } else {
            bandi * 16
        };
        let dx = (if flip_sy { h - 16 - sy } else { sy }) as usize;
        for ti in 0..ntiles {
            let sx = if flip_sx {
                (ntiles - 1 - ti) * 16
            } else {
                ti * 16
            };
            let base = sy as usize * sstride + sx as usize;
            // a_i = [row sy+i | row sy+i+8], 16 B per lane.
            macro_rules! ld {
                ($i:literal) => {{
                    let lo: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                        .try_into()
                        .unwrap();
                    let hi: &[u8; 16] = sbytes
                        [base + ($i + 8) * sstride..base + ($i + 8) * sstride + 16]
                        .try_into()
                        .unwrap();
                    _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo))
                }};
            }
            let (a0, a1, a2, a3) = (ld!(0), ld!(1), ld!(2), ld!(3));
            let (a4, a5, a6, a7) = (ld!(4), ld!(5), ld!(6), ld!(7));
            let b0 = _mm256_unpacklo_epi8(a0, a4);
            let b1 = _mm256_unpackhi_epi8(a0, a4);
            let b2 = _mm256_unpacklo_epi8(a1, a5);
            let b3 = _mm256_unpackhi_epi8(a1, a5);
            let b4 = _mm256_unpacklo_epi8(a2, a6);
            let b5 = _mm256_unpackhi_epi8(a2, a6);
            let b6 = _mm256_unpacklo_epi8(a3, a7);
            let b7 = _mm256_unpackhi_epi8(a3, a7);
            let a0 = _mm256_unpacklo_epi8(b0, b4);
            let a1 = _mm256_unpackhi_epi8(b0, b4);
            let a2 = _mm256_unpacklo_epi8(b1, b5);
            let a3 = _mm256_unpackhi_epi8(b1, b5);
            let a4 = _mm256_unpacklo_epi8(b2, b6);
            let a5 = _mm256_unpackhi_epi8(b2, b6);
            let a6 = _mm256_unpacklo_epi8(b3, b7);
            let a7 = _mm256_unpackhi_epi8(b3, b7);
            let outs = [
                _mm256_unpacklo_epi8(a0, a4),
                _mm256_unpackhi_epi8(a0, a4),
                _mm256_unpacklo_epi8(a1, a5),
                _mm256_unpackhi_epi8(a1, a5),
                _mm256_unpacklo_epi8(a2, a6),
                _mm256_unpackhi_epi8(a2, a6),
                _mm256_unpacklo_epi8(a3, a7),
                _mm256_unpackhi_epi8(a3, a7),
            ];
            for (k, &v) in outs.iter().enumerate() {
                // After 0xD8: lane0 = dst row 2k, lane1 = dst row 2k+1.
                let v = _mm256_permute4x64_epi64::<0xD8>(v);
                let v = if flip_sy {
                    _mm256_shuffle_epi8(v, rev)
                } else {
                    v
                };
                for half in 0..2u32 {
                    let c = 2 * k as u32 + half;
                    let dy = if flip_sx { w - 1 - (sx + c) } else { sx + c };
                    let doff = dy as usize * dstride + dx;
                    let out: &mut [u8; 16] = (&mut dbytes[doff..doff + 16]).try_into().unwrap();
                    let x = if half == 0 {
                        _mm256_castsi256_si128(v)
                    } else {
                        _mm256_extracti128_si256::<1>(v)
                    };
                    _mm_storeu_si128(out, x);
                }
            }
        }
    }
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        1,
        full_w,
        full_h,
    );
}

/// 16×8 2-byte tiles, AVX2 (ermig1979/Simd `2x16x8` network): 8 ymm
/// pair rows r and r+8 per lane (8 pixels each); three rounds of
/// stride-4 word unpacks; each result is one full 32-byte destination
/// row — single store, no final permute. Column-stripe blocked (wins at
/// 2 bpp); band order descends under flip_sy (measured).
#[arcane(import_intrinsics)]
pub(super) fn transpose2_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    // Reverse 16 u16 across 32 B: per-lane u16 reversal + half swap.
    const REVU16: [u8; 16] = [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1];
    let rev = _mm256_broadcastsi128_si256(_mm_loadu_si128(&REVU16));

    let full_w = w & !7;
    let full_h = h & !15;

    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 8;
        let nbands = full_h / 16;
        for bandi in 0..nbands {
            let sy = if flip_sy {
                (nbands - 1 - bandi) * 16
            } else {
                bandi * 16
            };
            let dx = (if flip_sy { h - 16 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 8
                } else {
                    bx + ti * 8
                };
                let base = sy as usize * sstride + sx as usize * 2;
                macro_rules! ld {
                    ($i:literal) => {{
                        let lo: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        let hi: &[u8; 16] = sbytes
                            [base + ($i + 8) * sstride..base + ($i + 8) * sstride + 16]
                            .try_into()
                            .unwrap();
                        _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo))
                    }};
                }
                let (a0, a1, a2, a3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let (a4, a5, a6, a7) = (ld!(4), ld!(5), ld!(6), ld!(7));
                let b0 = _mm256_unpacklo_epi16(a0, a4);
                let b1 = _mm256_unpackhi_epi16(a0, a4);
                let b2 = _mm256_unpacklo_epi16(a1, a5);
                let b3 = _mm256_unpackhi_epi16(a1, a5);
                let b4 = _mm256_unpacklo_epi16(a2, a6);
                let b5 = _mm256_unpackhi_epi16(a2, a6);
                let b6 = _mm256_unpacklo_epi16(a3, a7);
                let b7 = _mm256_unpackhi_epi16(a3, a7);
                let a0 = _mm256_unpacklo_epi16(b0, b4);
                let a1 = _mm256_unpackhi_epi16(b0, b4);
                let a2 = _mm256_unpacklo_epi16(b1, b5);
                let a3 = _mm256_unpackhi_epi16(b1, b5);
                let a4 = _mm256_unpacklo_epi16(b2, b6);
                let a5 = _mm256_unpackhi_epi16(b2, b6);
                let a6 = _mm256_unpacklo_epi16(b3, b7);
                let a7 = _mm256_unpackhi_epi16(b3, b7);
                let outs = [
                    _mm256_unpacklo_epi16(a0, a4),
                    _mm256_unpackhi_epi16(a0, a4),
                    _mm256_unpacklo_epi16(a1, a5),
                    _mm256_unpackhi_epi16(a1, a5),
                    _mm256_unpacklo_epi16(a2, a6),
                    _mm256_unpackhi_epi16(a2, a6),
                    _mm256_unpacklo_epi16(a3, a7),
                    _mm256_unpackhi_epi16(a3, a7),
                ];
                for (c, &v) in outs.iter().enumerate() {
                    let v = if flip_sy {
                        _mm256_permute4x64_epi64::<0x4E>(_mm256_shuffle_epi8(v, rev))
                    } else {
                        v
                    };
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 2;
                    let out: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                    _mm256_storeu_si256(out, v);
                }
            }
        }
    }
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        2,
        full_w,
        full_h,
    );
}

/// 8×8 4-byte tiles, AVX2 (guaranteed at x86-64-v3): 32-byte rows,
/// dword/qword unpacks + cross-lane permute — the kernel class
/// fast_transpose/Simd lead with at 4 bpp.
#[arcane(import_intrinsics)]
pub(super) fn transpose4_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    let full_w = w & !7;
    let full_h = h & !7;
    // Column-stripe blocking; see transpose1_v3.
    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 8;
        let nbands = full_h / 8;
        for bandi in 0..nbands {
            let sy = bandi * 8;
            let dx = (if flip_sy { h - 8 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 8
                } else {
                    bx + ti * 8
                };
                let base = sy as usize * sstride + sx as usize * 4;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 32] = sbytes[base + $i * sstride..base + $i * sstride + 32]
                            .try_into()
                            .unwrap();
                        _mm256_loadu_si256(a)
                    }};
                }
                let (r0, r1, r2, r3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let (r4, r5, r6, r7) = (ld!(4), ld!(5), ld!(6), ld!(7));
                let t0 = _mm256_unpacklo_epi32(r0, r1);
                let t1 = _mm256_unpackhi_epi32(r0, r1);
                let t2 = _mm256_unpacklo_epi32(r2, r3);
                let t3 = _mm256_unpackhi_epi32(r2, r3);
                let t4 = _mm256_unpacklo_epi32(r4, r5);
                let t5 = _mm256_unpackhi_epi32(r4, r5);
                let t6 = _mm256_unpacklo_epi32(r6, r7);
                let t7 = _mm256_unpackhi_epi32(r6, r7);
                let u0 = _mm256_unpacklo_epi64(t0, t2);
                let u1 = _mm256_unpackhi_epi64(t0, t2);
                let u2 = _mm256_unpacklo_epi64(t1, t3);
                let u3 = _mm256_unpackhi_epi64(t1, t3);
                let u4 = _mm256_unpacklo_epi64(t4, t6);
                let u5 = _mm256_unpackhi_epi64(t4, t6);
                let u6 = _mm256_unpacklo_epi64(t5, t7);
                let u7 = _mm256_unpackhi_epi64(t5, t7);
                let cols = [
                    _mm256_permute2x128_si256::<0x20>(u0, u4),
                    _mm256_permute2x128_si256::<0x20>(u1, u5),
                    _mm256_permute2x128_si256::<0x20>(u2, u6),
                    _mm256_permute2x128_si256::<0x20>(u3, u7),
                    _mm256_permute2x128_si256::<0x31>(u0, u4),
                    _mm256_permute2x128_si256::<0x31>(u1, u5),
                    _mm256_permute2x128_si256::<0x31>(u2, u6),
                    _mm256_permute2x128_si256::<0x31>(u3, u7),
                ];
                for (c, &col) in cols.iter().enumerate() {
                    let col = if flip_sy {
                        _mm256_permutevar8x32_epi32(col, rev)
                    } else {
                        col
                    };
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 4;
                    let out: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                    _mm256_storeu_si256(out, col);
                }
            }
        }
    }
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        4,
        full_w,
        full_h,
    );
}
/// 4×4 8-byte pixels (RGBA16 / GRAYAF32), AVX2: rows are 32 B, qword
/// unpacks + cross-lane permute; exact 32-byte loads/stores.
#[arcane(import_intrinsics)]
pub(super) fn transpose8_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    let full_w = w & !3;
    let full_h = h & !3;
    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 4;
        let nbands = full_h / 4;
        for bandi in 0..nbands {
            let sy = bandi * 4;
            let dx = (if flip_sy { h - 4 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 4
                } else {
                    bx + ti * 4
                };
                let base = sy as usize * sstride + sx as usize * 8;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 32] = sbytes[base + $i * sstride..base + $i * sstride + 32]
                            .try_into()
                            .unwrap();
                        _mm256_loadu_si256(a)
                    }};
                }
                let (a0, a1, a2, a3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let b0 = _mm256_unpacklo_epi64(a0, a1);
                let b1 = _mm256_unpackhi_epi64(a0, a1);
                let b2 = _mm256_unpacklo_epi64(a2, a3);
                let b3 = _mm256_unpackhi_epi64(a2, a3);
                let cols = [
                    _mm256_permute2x128_si256::<0x20>(b0, b2),
                    _mm256_permute2x128_si256::<0x20>(b1, b3),
                    _mm256_permute2x128_si256::<0x31>(b0, b2),
                    _mm256_permute2x128_si256::<0x31>(b1, b3),
                ];
                for (c, &col) in cols.iter().enumerate() {
                    let col = if flip_sy {
                        _mm256_permute4x64_epi64::<0x1B>(col)
                    } else {
                        col
                    };
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 8;
                    let out: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                    _mm256_storeu_si256(out, col);
                }
            }
        }
    }
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        8,
        full_w,
        full_h,
    );
}

/// 16-byte pixels (RGBAF32 / OKLABAF32): a transpose is pure 16-byte
/// block movement — gather 8 source rows per destination run, store
/// sequentially. Exact loads/stores, no shuffles.
#[arcane(import_intrinsics)]
pub(super) fn transpose16_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    let full_h = h & !7;
    const MACRO: u32 = 256;
    let nblocks = w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(w);
        let ncols = bx_end - bx;
        let nbands = full_h / 8;
        for bandi in 0..nbands {
            let sy = bandi * 8;
            let dx = (if flip_sy { h - 8 - sy } else { sy }) as usize;
            // Column pairs: one 32-byte load covers two adjacent
            // pixels; cross-lane permutes pair consecutive rows of one
            // column, so EVERY load and store is a full 32 bytes.
            let npairs = ncols / 2;
            for pi in 0..npairs {
                let p = if flip_sx { npairs - 1 - pi } else { pi };
                let sx = bx + p * 2;
                let dy0 = if flip_sx { w - 2 - sx } else { sx };
                let dbase0 = dy0 as usize * dstride + dx * 16;
                let dbase1 = dbase0 + dstride;
                let sbase = sy as usize * sstride + sx as usize * 16;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 32] = sbytes[sbase + $i * sstride..sbase + $i * sstride + 32]
                            .try_into()
                            .unwrap();
                        _mm256_loadu_si256(a)
                    }};
                }
                let (y0, y1, y2, y3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let (y4, y5, y6, y7) = (ld!(4), ld!(5), ld!(6), ld!(7));
                // Row pairs in band order (already reversed for flip_sy
                // by swapping operands and reading bottom-up).
                let pairs = if flip_sy {
                    [(y7, y6), (y5, y4), (y3, y2), (y1, y0)]
                } else {
                    [(y0, y1), (y2, y3), (y4, y5), (y6, y7)]
                };
                let (lo_base, hi_base) = if flip_sx {
                    (dbase1, dbase0)
                } else {
                    (dbase0, dbase1)
                };
                for (k, &(ra, rb)) in pairs.iter().enumerate() {
                    let lo = _mm256_permute2x128_si256::<0x20>(ra, rb);
                    let hi = _mm256_permute2x128_si256::<0x31>(ra, rb);
                    let out_lo: &mut [u8; 32] = (&mut dbytes
                        [lo_base + k * 32..lo_base + k * 32 + 32])
                        .try_into()
                        .unwrap();
                    _mm256_storeu_si256(out_lo, lo);
                    let out_hi: &mut [u8; 32] = (&mut dbytes
                        [hi_base + k * 32..hi_base + k * 32 + 32])
                        .try_into()
                        .unwrap();
                    _mm256_storeu_si256(out_hi, hi);
                }
            }
            // Odd trailing column of the stripe.
            for ci in (npairs * 2)..ncols {
                let sx = bx + ci;
                let dy = if flip_sx { w - 1 - sx } else { sx };
                let dbase = dy as usize * dstride + dx * 16;
                let sbase = sy as usize * sstride + sx as usize * 16;
                for k in 0..8usize {
                    let r = if flip_sy { 7 - k } else { k };
                    let a: &[u8; 16] = sbytes[sbase + r * sstride..sbase + r * sstride + 16]
                        .try_into()
                        .unwrap();
                    let v = _mm_loadu_si128(a);
                    let out: &mut [u8; 16] = (&mut dbytes[dbase + k * 16..dbase + k * 16 + 16])
                        .try_into()
                        .unwrap();
                    _mm_storeu_si128(out, v);
                }
            }
        }
    }
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        16,
        0,
        w,
        full_h,
        h,
    );
}

/// Expand mask for 6-byte pixels: per lane, two RGB16 pixels → two
/// 8-byte (RGBX16) pixels.
const EXPAND6: [u8; 16] = [0, 1, 2, 3, 4, 5, 128, 128, 6, 7, 8, 9, 10, 11, 128, 128];
/// Compress: two 8-byte lanes → 12 valid bytes (per lane).
const COMPRESS6_FWD: [u8; 16] = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 128, 128, 128, 128];
/// Compress with the two pixels of the lane swapped (flip_sy).
const COMPRESS6_REV: [u8; 16] = [8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 128, 128, 128, 128];

/// 4×4 6-byte pixels (RGB16): expand 6→8 per lane, qword 4×4 transpose,
/// compress 8→6, contiguous 24-byte store as 16+8. Same construction as
/// the RGB8 kernel one level up. Second per-row load sits at +12 with
/// 4 bytes of slop — guarded on the band touching the image's last row.
#[arcane(import_intrinsics)]
pub(super) fn transpose6_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    let expand = _mm256_broadcastsi128_si256(_mm_loadu_si128(&EXPAND6));
    let compress = _mm256_broadcastsi128_si256(_mm_loadu_si128(if flip_sy {
        &COMPRESS6_REV
    } else {
        &COMPRESS6_FWD
    }));
    let merge = if flip_sy {
        _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 6, 7)
    } else {
        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 6, 7)
    };

    let full_w = w & !3;
    let full_h = h & !3;
    // Last-band guard: per-row loads cover [sx*6, sx*6+28); need
    // (sx+4)*6 + 4 ≤ w*6 on the final image row, i.e. sx + 4 ≤ w − 1.
    let guard_w = if full_h == h {
        if w >= 1 { (w - 1) & !3 } else { 0 }
    } else {
        full_w
    }
    .min(full_w);

    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 4;
        let nbands = full_h / 4;
        for bandi in 0..nbands {
            let sy = bandi * 4;
            let limit = if sy + 4 >= h { guard_w } else { full_w };
            let dx = (if flip_sy { h - 4 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 4
                } else {
                    bx + ti * 4
                };
                if sx + 4 > limit {
                    continue;
                }
                let base = sy as usize * sstride + sx as usize * 6;
                macro_rules! ld {
                    ($i:literal) => {{
                        let lo: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        let hi: &[u8; 16] = sbytes
                            [base + $i * sstride + 12..base + $i * sstride + 28]
                            .try_into()
                            .unwrap();
                        _mm256_shuffle_epi8(
                            _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo)),
                            expand,
                        )
                    }};
                }
                let (a0, a1, a2, a3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let b0 = _mm256_unpacklo_epi64(a0, a1);
                let b1 = _mm256_unpackhi_epi64(a0, a1);
                let b2 = _mm256_unpacklo_epi64(a2, a3);
                let b3 = _mm256_unpackhi_epi64(a2, a3);
                let cols = [
                    _mm256_permute2x128_si256::<0x20>(b0, b2),
                    _mm256_permute2x128_si256::<0x20>(b1, b3),
                    _mm256_permute2x128_si256::<0x31>(b0, b2),
                    _mm256_permute2x128_si256::<0x31>(b1, b3),
                ];
                for (c, &col) in cols.iter().enumerate() {
                    let packed =
                        _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(col, compress), merge);
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 6;
                    let (head, tail) = dbytes[doff..doff + 24].split_at_mut(16);
                    let head: &mut [u8; 16] = head.try_into().unwrap();
                    let tail: &mut [u8; 8] = tail.try_into().unwrap();
                    _mm_storeu_si128(head, _mm256_castsi256_si128(packed));
                    _mm_storeu_si64(tail, _mm256_extracti128_si256::<1>(packed));
                }
            }
        }
    }
    scalar_last_band_guard(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        6,
        full_w,
        full_h,
        guard_w,
        4,
    );
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        6,
        full_w,
        full_h,
    );
}

/// 2×4 12-byte pixels (RGBF32): each pixel rides one 128-bit lane
/// (12 valid bytes); lane permutes transpose, a dword permute compacts
/// each output pair to 24 contiguous bytes (pixels are dword-aligned,
/// so no byte shuffle is needed). Second per-row load sits at +12 with
/// 4 bytes of slop — guarded on the band touching the image's last row.
#[arcane(import_intrinsics)]
pub(super) fn transpose12_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) = inverse_flips(orientation).expect("transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    let merge = if flip_sy {
        _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 6, 7)
    } else {
        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 6, 7)
    };

    let full_w = w & !1;
    let full_h = h & !3;
    // Last-band guard: loads cover [sx*12, sx*12+28); need
    // (sx+2)*12 + 4 ≤ w*12 on the final image row → sx + 2 ≤ w − 1.
    let guard_w = if full_h == h {
        if w >= 1 { (w - 1) & !1 } else { 0 }
    } else {
        full_w
    }
    .min(full_w);

    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 2;
        let nbands = full_h / 4;
        for bandi in 0..nbands {
            let sy = bandi * 4;
            let limit = if sy + 4 >= h { guard_w } else { full_w };
            let dx = (if flip_sy { h - 4 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 2
                } else {
                    bx + ti * 2
                };
                if sx + 2 > limit {
                    continue;
                }
                let base = sy as usize * sstride + sx as usize * 12;
                macro_rules! ld {
                    ($i:literal) => {{
                        let lo: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        let hi: &[u8; 16] = sbytes
                            [base + $i * sstride + 12..base + $i * sstride + 28]
                            .try_into()
                            .unwrap();
                        _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo))
                    }};
                }
                let (a0, a1, a2, a3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                // dst row (sx+c): pixel pairs from rows (0,1) and (2,3).
                for c in 0..2u32 {
                    let (y01, y23) = if c == 0 {
                        (
                            _mm256_permute2x128_si256::<0x20>(a0, a1),
                            _mm256_permute2x128_si256::<0x20>(a2, a3),
                        )
                    } else {
                        (
                            _mm256_permute2x128_si256::<0x31>(a0, a1),
                            _mm256_permute2x128_si256::<0x31>(a2, a3),
                        )
                    };
                    // flip_sy: emit rows 3..0 instead of 0..3.
                    let (first, second) = if flip_sy { (y23, y01) } else { (y01, y23) };
                    let dy = if flip_sx { w - 1 - (sx + c) } else { sx + c };
                    let doff = dy as usize * dstride + dx * 12;
                    // First 24 B go out as one full 32-byte store whose
                    // 8-byte slop lands INSIDE this run (bytes 24..32),
                    // immediately overwritten by the second half — three
                    // stores per 48-byte run instead of four, and the
                    // slop never crosses the run boundary.
                    let p0 = _mm256_permutevar8x32_epi32(first, merge);
                    let out0: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                    _mm256_storeu_si256(out0, p0);
                    let p1 = _mm256_permutevar8x32_epi32(second, merge);
                    let (head, tail) = dbytes[doff + 24..doff + 48].split_at_mut(16);
                    let head: &mut [u8; 16] = head.try_into().unwrap();
                    let tail: &mut [u8; 8] = tail.try_into().unwrap();
                    _mm_storeu_si128(head, _mm256_castsi256_si128(p1));
                    _mm_storeu_si64(tail, _mm256_extracti128_si256::<1>(p1));
                }
            }
        }
    }
    scalar_last_band_guard(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        12,
        full_w,
        full_h,
        guard_w,
        4,
    );
    scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        12,
        full_w,
        full_h,
    );
}
