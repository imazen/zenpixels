use super::{Orientation, PixelSlice, PixelSliceMut, inverse_flips};
use archmage::prelude::*;

/// Expand mask: 12 RGB bytes → 4 RGBX u32 lanes (X = 0).
const EXPAND: [u8; 16] = [0, 1, 2, 128, 3, 4, 5, 128, 6, 7, 8, 128, 9, 10, 11, 128];
/// Compress mask: 4 RGBX lanes → 12 RGB bytes (+ 4 zero bytes).
const COMPRESS_FWD: [u8; 16] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 128, 128, 128, 128];
/// Compress with lane order reversed (Rotate90/Transverse: `flip_sy`
/// reverses the within-dst-row pixel order contributed by source rows).
const COMPRESS_REV: [u8; 16] = [12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2, 128, 128, 128, 128];

/// Whole-image RGB8 transposing bake for the four axis-swapping
/// orientations. Full 4×4-pixel tiles go SIMD (16-byte loads with 4-byte
/// slop, guarded on the band touching the image's last row); stores are
/// slop-free 8+4 bytes; every remainder pixel takes the scalar
/// `forward_map` path against flat destination bytes.
#[arcane(import_intrinsics)]
pub(super) fn transpose3_v3(
    _token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let (flip_sx, flip_sy) =
        inverse_flips(orientation).expect("transpose3_v3 called for transposing orientation");
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    // Both 128-bit lanes get the same per-lane masks; the cross-lane
    // dword permute then compacts the two 12-byte lane halves into one
    // 24-byte run (forward: lane0 then lane1 = source rows ascending;
    // flip_sy: lanes swapped AND pixels reversed inside each lane).
    let expand = _mm256_broadcastsi128_si256(_mm_loadu_si128(&EXPAND));
    let compress = _mm256_broadcastsi128_si256(_mm_loadu_si128(if flip_sy {
        &COMPRESS_REV
    } else {
        &COMPRESS_FWD
    }));
    let merge = if flip_sy {
        _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 6, 7)
    } else {
        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 6, 7)
    };

    let full_h = h & !7;
    let full_w = w & !3;

    // Tiles in the band containing the image's final row must keep their
    // 16-byte loads inside the buffer: need (sx+4)*3 + 4 ≤ w*3, i.e.
    // sx + 4 ≤ w - 2. Other bands' slop lands in the next row, which
    // `as_strided_bytes` always covers. Guard-trimmed columns fall to the
    // trailing scalar pass.
    let guard_w = if full_h == h {
        if w >= 6 { (w - 2) & !3 } else { 0 }
    } else {
        full_w
    }
    .min(full_w);

    // Column-stripe blocking; see transpose1_v3.
    const MACRO: u32 = 64;
    let nblocks = full_w.div_ceil(MACRO);
    for bi in 0..nblocks {
        let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
        let bx_end = (bx + MACRO).min(full_w);
        let ntiles = (bx_end - bx) / 4;
        let nbands = full_h / 8;
        for bandi in 0..nbands {
            let sy = bandi * 8;
            let limit = if sy + 8 >= h { guard_w } else { full_w };
            for ti in 0..ntiles {
                let sx = if flip_sx {
                    bx + (ntiles - 1 - ti) * 4
                } else {
                    bx + ti * 4
                };
                if sx + 4 > limit {
                    continue;
                }
                // Load 8 source rows × 16 B; pair rows r and r+4 into one
                // ymm (low/high lane) and expand to u32 lanes, so the
                // 4×4 dword transpose below transposes both row quads at
                // once and each output register is one full 8-pixel
                // destination run.
                let base = sy as usize * sstride + sx as usize * 3;
                macro_rules! ld {
                    ($i:literal) => {{
                        let a: &[u8; 16] = sbytes[base + $i * sstride..base + $i * sstride + 16]
                            .try_into()
                            .unwrap();
                        _mm_loadu_si128(a)
                    }};
                }
                let p0 = _mm256_shuffle_epi8(_mm256_set_m128i(ld!(4), ld!(0)), expand);
                let p1 = _mm256_shuffle_epi8(_mm256_set_m128i(ld!(5), ld!(1)), expand);
                let p2 = _mm256_shuffle_epi8(_mm256_set_m128i(ld!(6), ld!(2)), expand);
                let p3 = _mm256_shuffle_epi8(_mm256_set_m128i(ld!(7), ld!(3)), expand);
                // Per-lane 4×4 u32 transpose (both lanes in parallel).
                let t0 = _mm256_unpacklo_epi32(p0, p1);
                let t1 = _mm256_unpacklo_epi32(p2, p3);
                let t2 = _mm256_unpackhi_epi32(p0, p1);
                let t3 = _mm256_unpackhi_epi32(p2, p3);
                let cols = [
                    _mm256_unpacklo_epi64(t0, t1),
                    _mm256_unpackhi_epi64(t0, t1),
                    _mm256_unpacklo_epi64(t2, t3),
                    _mm256_unpackhi_epi64(t2, t3),
                ];
                // cols[c] = source column sx+c: lane0 = rows sy..sy+4,
                // lane1 = rows sy+4..sy+8 → one 24-byte run in dst row dy.
                let dx = if flip_sy { h - 8 - sy } else { sy };
                for (c, &col) in cols.iter().enumerate() {
                    let packed =
                        _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(col, compress), merge);
                    let dy = if flip_sx {
                        w - 1 - (sx + c as u32)
                    } else {
                        sx + c as u32
                    };
                    let doff = dy as usize * dstride + dx as usize * 3;
                    // Slop-free 24-byte store as 16+8.
                    let (head, tail) = dbytes[doff..doff + 24].split_at_mut(16);
                    let head: &mut [u8; 16] = head.try_into().unwrap();
                    let tail: &mut [u8; 8] = tail.try_into().unwrap();
                    _mm_storeu_si128(head, _mm256_castsi256_si128(packed));
                    _mm_storeu_si64(tail, _mm256_extracti128_si256::<1>(packed));
                }
            }
        }
    }
    // Guard-trimmed columns of the last band (only when it touches the
    // image's final row), then the right + bottom edge strips.
    super::pxn_x86::scalar_last_band_guard(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        3,
        full_w,
        full_h,
        guard_w,
        8,
    );
    super::pxn_x86::scalar_edges(
        sbytes,
        sstride,
        dbytes,
        dstride,
        orientation,
        w,
        h,
        3,
        full_w,
        full_h,
    );
}
