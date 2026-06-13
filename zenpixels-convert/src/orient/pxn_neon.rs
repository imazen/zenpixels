use super::{Orientation, PixelSlice, PixelSliceMut, inverse_flips};
use archmage::prelude::*;

#[allow(clippy::too_many_arguments)]
fn scalar_rect(
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

/// 8×16 gray8 tiles (Simd `1x8x16`): 8 q-loads, three vzipq_u8 rounds,
/// sixteen 8-byte half-stores. flip_sy reverses each 8-byte run
/// (`vrev64_u8`); flip_sx reverses tile/store order.
#[arcane(import_intrinsics)]
pub(super) fn transpose1_neon(
    _token: NeonToken,
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

    let full_w = w & !15;
    let full_h = h & !15;
    let ntiles = full_w / 16;
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
            macro_rules! ld {
                ($i:expr) => {{
                    let a: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                        .try_into()
                        .unwrap();
                    vld1q_u8(a)
                }};
            }
            // Two 8-row vzip cascades (Simd 1x8x16 network), merged per
            // column into one 16-byte destination store.
            macro_rules! cascade {
                ($o:expr) => {{
                    let b0 = vzipq_u8(ld!($o), ld!($o + 4));
                    let b1 = vzipq_u8(ld!($o + 1), ld!($o + 5));
                    let b2 = vzipq_u8(ld!($o + 2), ld!($o + 6));
                    let b3 = vzipq_u8(ld!($o + 3), ld!($o + 7));
                    let a0 = vzipq_u8(b0.0, b2.0);
                    let a1 = vzipq_u8(b0.1, b2.1);
                    let a2 = vzipq_u8(b1.0, b3.0);
                    let a3 = vzipq_u8(b1.1, b3.1);
                    let c0 = vzipq_u8(a0.0, a2.0);
                    let c1 = vzipq_u8(a0.1, a2.1);
                    let c2 = vzipq_u8(a1.0, a3.0);
                    let c3 = vzipq_u8(a1.1, a3.1);
                    [c0.0, c0.1, c1.0, c1.1, c2.0, c2.1, c3.0, c3.1]
                }};
            }
            let lo = cascade!(0usize);
            let hi = cascade!(8usize);
            for k in 0..8usize {
                for half in 0..2u32 {
                    let c = 2 * k as u32 + half;
                    let dlo = if half == 0 {
                        vget_low_u8(lo[k])
                    } else {
                        vget_high_u8(lo[k])
                    };
                    let dhi = if half == 0 {
                        vget_low_u8(hi[k])
                    } else {
                        vget_high_u8(hi[k])
                    };
                    let mut q = vcombine_u8(dlo, dhi);
                    if flip_sy {
                        let r = vrev64q_u8(q);
                        q = vextq_u8::<8>(r, r);
                    }
                    let dy = if flip_sx { w - 1 - (sx + c) } else { sx + c };
                    let doff = dy as usize * dstride + dx;
                    let out: &mut [u8; 16] = (&mut dbytes[doff..doff + 16]).try_into().unwrap();
                    vst1q_u8(out, q);
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
        1,
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
        1,
        0,
        w,
        full_h,
        h,
    );
}

/// 8×8 2-byte tiles (Simd `2x8x8`): vzipq_u16 cascade, 16-byte stores.
#[arcane(import_intrinsics)]
pub(super) fn transpose2_neon(
    _token: NeonToken,
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

    let full_w = w & !7;
    let full_h = h & !7;
    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 8;
        let nbands = full_h / 8;
        for bandi in 0..nbands {
            let sy = if flip_sy {
                (nbands - 1 - bandi) * 8
            } else {
                bandi * 8
            };
            let dx = (if flip_sy { h - 8 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 8
                } else {
                    bx + ti * 8
                };
                let base = sy as usize * sstride + sx as usize * 2;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        vreinterpretq_u16_u8(vld1q_u8(a))
                    }};
                }
                let (r0, r1, r2, r3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let (r4, r5, r6, r7) = (ld!(4), ld!(5), ld!(6), ld!(7));
                let b0 = vzipq_u16(r0, r4);
                let b1 = vzipq_u16(r1, r5);
                let b2 = vzipq_u16(r2, r6);
                let b3 = vzipq_u16(r3, r7);
                let a0 = vzipq_u16(b0.0, b2.0);
                let a1 = vzipq_u16(b0.1, b2.1);
                let a2 = vzipq_u16(b1.0, b3.0);
                let a3 = vzipq_u16(b1.1, b3.1);
                let c0 = vzipq_u16(a0.0, a2.0);
                let c1 = vzipq_u16(a0.1, a2.1);
                let c2 = vzipq_u16(a1.0, a3.0);
                let c3 = vzipq_u16(a1.1, a3.1);
                let cols = [c0.0, c0.1, c1.0, c1.1, c2.0, c2.1, c3.0, c3.1];
                for (c, &col) in cols.iter().enumerate() {
                    let v = vreinterpretq_u8_u16(col);
                    // Reverse 8 u16: per-64-bit u16 reversal + half swap.
                    let v = if flip_sy {
                        let r = vreinterpretq_u8_u16(vrev64q_u16(vreinterpretq_u16_u8(v)));
                        vextq_u8::<8>(r, r)
                    } else {
                        v
                    };
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 2;
                    let out: &mut [u8; 16] = (&mut dbytes[doff..doff + 16]).try_into().unwrap();
                    vst1q_u8(out, v);
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
        2,
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
        2,
        0,
        w,
        full_h,
        h,
    );
}

/// Store the low 12 bytes of a q register with no slop: 8-byte half
/// store + 4-byte scalar tail. Macro so it expands inside the
/// `#[arcane]` target-feature regions (a plain fn would need its own
/// `#[target_feature]` to call the NEON value intrinsics).
macro_rules! store12 {
    ($dbytes:ident, $doff:expr, $v:expr) => {{
        let v = $v;
        let off = $doff;
        let head: &mut [u8; 8] = (&mut $dbytes[off..off + 8]).try_into().unwrap();
        vst1_u8(head, vget_low_u8(v));
        let tail = vgetq_lane_u32::<2>(vreinterpretq_u32_u8(v));
        $dbytes[off + 8..off + 12].copy_from_slice(&tail.to_le_bytes());
    }};
}

/// Expand mask: 12 RGB bytes → 4 RGBX u32 lanes (255 ⇒ zero lane).
const EXPAND3: [u8; 16] = [0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255];
const COMPRESS3_FWD: [u8; 16] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 255, 255, 255, 255];
const COMPRESS3_REV: [u8; 16] = [12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2, 255, 255, 255, 255];

/// 4×4 RGB8 tiles: tbl-expand 3→4, vzipq_u32 4×4 transpose, tbl-compress
/// 4→3, slop-free 8+4 stores. Loads carry 4 bytes of slop, guarded on
/// the band touching the image's final row (same contract as x86).
#[arcane(import_intrinsics)]
pub(super) fn transpose3_neon(
    _token: NeonToken,
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

    let expand = vld1q_u8(&EXPAND3);
    let compress = vld1q_u8(if flip_sy {
        &COMPRESS3_REV
    } else {
        &COMPRESS3_FWD
    });

    let full_w = w & !3;
    let full_h = h & !3;
    let guard_w = if full_h == h {
        if w >= 6 { (w - 2) & !3 } else { 0 }
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
                let base = sy as usize * sstride + sx as usize * 3;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        vreinterpretq_u32_u8(vqtbl1q_u8(vld1q_u8(a), expand))
                    }};
                }
                let (p0, p1, p2, p3) = (ld!(0), ld!(1), ld!(2), ld!(3));
                let b0 = vzipq_u32(p0, p2);
                let b1 = vzipq_u32(p1, p3);
                let a0 = vzipq_u32(b0.0, b1.0);
                let a1 = vzipq_u32(b0.1, b1.1);
                let cols = [a0.0, a0.1, a1.0, a1.1];
                for (c, &col) in cols.iter().enumerate() {
                    let packed = vqtbl1q_u8(vreinterpretq_u8_u32(col), compress);
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 3;
                    store12!(dbytes, doff, packed);
                }
            }
        }
    }
    if guard_w < full_w && full_h == h && h >= 4 {
        scalar_rect(
            sbytes,
            sstride,
            dbytes,
            dstride,
            orientation,
            w,
            h,
            3,
            guard_w,
            full_w,
            h - 4,
            h,
        );
    }
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        3,
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
        3,
        0,
        w,
        full_h,
        h,
    );
}

/// 4×4 4-byte tiles (Simd `4x4x4`): two vzipq_u32 rounds, 16-byte stores.
#[arcane(import_intrinsics)]
pub(super) fn transpose4_neon(
    _token: NeonToken,
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
    let full_h = h & !7;
    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 4;
        let nbands = full_h / 8;
        for bandi in 0..nbands {
            let sy = bandi * 8;
            let dx = (if flip_sy { h - 8 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 4
                } else {
                    bx + ti * 4
                };
                let base = sy as usize * sstride + sx as usize * 4;
                macro_rules! ld {
                    ($i:expr) => {{
                        let a: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        vreinterpretq_u32_u8(vld1q_u8(a))
                    }};
                }
                macro_rules! net4 {
                    ($o:expr) => {{
                        let b0 = vzipq_u32(ld!($o), ld!($o + 2));
                        let b1 = vzipq_u32(ld!($o + 1), ld!($o + 3));
                        let a0 = vzipq_u32(b0.0, b1.0);
                        let a1 = vzipq_u32(b0.1, b1.1);
                        [a0.0, a0.1, a1.0, a1.1]
                    }};
                }
                // Rows 0-3 and 4-7 transposed separately; each dst run
                // is 8 px = 32 B, stored as one x2 tuple.
                let lo = net4!(0usize);
                let hi = net4!(4usize);
                for (c, (&l, &h2)) in lo.iter().zip(hi.iter()).enumerate() {
                    let (mut a, mut b) = (vreinterpretq_u8_u32(l), vreinterpretq_u8_u32(h2));
                    if flip_sy {
                        // Reverse 8 px: swap regs + reverse u32s in each.
                        let ra = vreinterpretq_u8_u32(vrev64q_u32(vreinterpretq_u32_u8(b)));
                        let rb = vreinterpretq_u8_u32(vrev64q_u32(vreinterpretq_u32_u8(a)));
                        a = vextq_u8::<8>(ra, ra);
                        b = vextq_u8::<8>(rb, rb);
                    }
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx * 4;
                    let out: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                    vst1q_u8_x2(
                        bytemuck::cast_mut::<[u8; 32], [[u8; 16]; 2]>(out),
                        core::arch::aarch64::uint8x16x2_t(a, b),
                    );
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
        4,
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
        4,
        0,
        w,
        full_h,
        h,
    );
}

/// 2×2 8-byte tiles: vtrn1q/vtrn2q_u64.
#[arcane(import_intrinsics)]
pub(super) fn transpose8_neon(
    _token: NeonToken,
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
                // 4 rows × 4 px via x2 tuple loads (32 B each); u64 trn
                // pairs rows; x2 tuple stores write whole 32-byte runs.
                macro_rules! ld2 {
                    ($i:literal) => {{
                        let a: &[u8; 32] = sbytes[base + $i * sstride..base + $i * sstride + 32]
                            .try_into()
                            .unwrap();
                        let t = vld1q_u8_x2(bytemuck::cast_ref::<[u8; 32], [[u8; 16]; 2]>(a));
                        (vreinterpretq_u64_u8(t.0), vreinterpretq_u64_u8(t.1))
                    }};
                }
                let (r0a, r0b) = ld2!(0);
                let (r1a, r1b) = ld2!(1);
                let (r2a, r2b) = ld2!(2);
                let (r3a, r3b) = ld2!(3);
                // col c run = [px(r0,c), px(r1,c), px(r2,c), px(r3,c)].
                macro_rules! emit {
                    ($c:literal, $lo:expr, $hi:expr) => {{
                        let (lo, hi) = if flip_sy {
                            (vextq_u64::<1>($hi, $hi), vextq_u64::<1>($lo, $lo))
                        } else {
                            ($lo, $hi)
                        };
                        let dy = if flip_sx { w - 1 - (sx + $c) } else { sx + $c };
                        let doff = dy as usize * dstride + dx * 8;
                        let out: &mut [u8; 32] = (&mut dbytes[doff..doff + 32]).try_into().unwrap();
                        vst1q_u8_x2(
                            bytemuck::cast_mut::<[u8; 32], [[u8; 16]; 2]>(out),
                            core::arch::aarch64::uint8x16x2_t(
                                vreinterpretq_u8_u64(lo),
                                vreinterpretq_u8_u64(hi),
                            ),
                        );
                    }};
                }
                emit!(0u32, vtrn1q_u64(r0a, r1a), vtrn1q_u64(r2a, r3a));
                emit!(1u32, vtrn2q_u64(r0a, r1a), vtrn2q_u64(r2a, r3a));
                emit!(2u32, vtrn1q_u64(r0b, r1b), vtrn1q_u64(r2b, r3b));
                emit!(3u32, vtrn2q_u64(r0b, r1b), vtrn2q_u64(r2b, r3b));
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
        8,
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
        8,
        0,
        w,
        full_h,
        h,
    );
}

const EXPAND6: [u8; 16] = [0, 1, 2, 3, 4, 5, 128, 128, 6, 7, 8, 9, 10, 11, 128, 128];
/// Compress: two 8-byte lanes → 12 valid bytes (per lane).
const COMPRESS6_FWD: [u8; 16] = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 128, 128, 128, 128];
/// Compress with the two pixels of the lane swapped (flip_sy).
const COMPRESS6_REV: [u8; 16] = [8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 128, 128, 128, 128];

/// 2×2 6-byte tiles: tbl-expand 6→8, u64 trn, tbl-compress, 8+4 stores.
/// Loads carry 4 bytes of slop, guarded on the final-row band.
#[arcane(import_intrinsics)]
pub(super) fn transpose6_neon(
    _token: NeonToken,
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

    let expand = vld1q_u8(&EXPAND6);
    let compress = vld1q_u8(if flip_sy {
        &COMPRESS6_REV
    } else {
        &COMPRESS6_FWD
    });

    let full_w = w & !1;
    let full_h = h & !1;
    // Loads cover [sx*6, sx*6+16); need (sx+2)*6 + 4 ≤ 6w on the final
    // row → sx + 2 ≤ w − 1.
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
        let nbands = full_h / 2;
        for bandi in 0..nbands {
            let sy = bandi * 2;
            let limit = if sy + 2 >= h { guard_w } else { full_w };
            let dx = (if flip_sy { h - 2 - sy } else { sy }) as usize;
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 2
                } else {
                    bx + ti * 2
                };
                if sx + 2 > limit {
                    continue;
                }
                let base = sy as usize * sstride + sx as usize * 6;
                let a0: &[u8; 16] = sbytes[base..base + 16].try_into().unwrap();
                let a1: &[u8; 16] = sbytes[base + sstride..base + sstride + 16]
                    .try_into()
                    .unwrap();
                let r0 = vreinterpretq_u64_u8(vqtbl1q_u8(vld1q_u8(a0), expand));
                let r1 = vreinterpretq_u64_u8(vqtbl1q_u8(vld1q_u8(a1), expand));
                let c0 = vtrn1q_u64(r0, r1);
                let c1 = vtrn2q_u64(r0, r1);
                for (c, col) in [(0u32, c0), (1u32, c1)] {
                    let packed = vqtbl1q_u8(vreinterpretq_u8_u64(col), compress);
                    let dy = if flip_sx { w - 1 - (sx + c) } else { sx + c };
                    let doff = dy as usize * dstride + dx * 6;
                    store12!(dbytes, doff, packed);
                }
            }
        }
    }
    if guard_w < full_w && full_h == h && h >= 2 {
        scalar_rect(
            sbytes,
            sstride,
            dbytes,
            dstride,
            orientation,
            w,
            h,
            6,
            guard_w,
            full_w,
            h - 2,
            h,
        );
    }
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        6,
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
        6,
        0,
        w,
        full_h,
        h,
    );
}

/// 2×2 12-byte tiles: each pixel rides one q register (12 valid bytes);
/// two 12-byte stores per destination run. Loads at +0 carry 4 bytes of
/// slop, guarded on the final-row band.
#[arcane(import_intrinsics)]
pub(super) fn transpose12_neon(
    _token: NeonToken,
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

    let full_w = w & !1;
    let full_h = h & !3;
    // Per-pixel 16-byte loads carry 4 bytes of slop; guarded on the
    // final-row band (need (sx+1)*12 + 4 ≤ 12w → sx ≤ w − 2 for the
    // pair's second pixel ⇒ pair start ≤ w − 3).
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
                macro_rules! ldpx {
                    ($row:literal, $px:literal) => {{
                        let a: &[u8; 16] = sbytes[base + $row * sstride + $px * 12
                            ..base + $row * sstride + $px * 12 + 16]
                            .try_into()
                            .unwrap();
                        vld1q_u8(a)
                    }};
                }
                // 4-row destination runs (48 B): the first three pixels
                // go out as full 16-byte stores whose 4-byte slop lands
                // inside the run (overwritten by the next store); only
                // the last pixel needs the slop-free 8+4 tail.
                for c in 0..2u32 {
                    let (p0, p1, p2, p3) = if c == 0 {
                        (ldpx!(0, 0), ldpx!(1, 0), ldpx!(2, 0), ldpx!(3, 0))
                    } else {
                        (ldpx!(0, 1), ldpx!(1, 1), ldpx!(2, 1), ldpx!(3, 1))
                    };
                    let (p0, p1, p2, p3) = if flip_sy {
                        (p3, p2, p1, p0)
                    } else {
                        (p0, p1, p2, p3)
                    };
                    let dy = if flip_sx { w - 1 - (sx + c) } else { sx + c };
                    let doff = dy as usize * dstride + dx * 12;
                    for (k, v) in [(0usize, p0), (1, p1), (2, p2)] {
                        let out: &mut [u8; 16] = (&mut dbytes[doff + k * 12..doff + k * 12 + 16])
                            .try_into()
                            .unwrap();
                        vst1q_u8(out, v);
                    }
                    store12!(dbytes, doff + 36, p3);
                }
            }
        }
    }
    if guard_w < full_w && full_h == h && h >= 4 {
        scalar_rect(
            sbytes,
            sstride,
            dbytes,
            dstride,
            orientation,
            w,
            h,
            12,
            guard_w,
            full_w,
            h - 4,
            h,
        );
    }
    scalar_rect(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        12,
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
        12,
        0,
        w,
        full_h,
        h,
    );
}
