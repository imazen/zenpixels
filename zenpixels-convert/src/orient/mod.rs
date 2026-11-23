//! Physical EXIF orientation baking — rotate / flip any pixel view or buffer.
//!
//! Supports all 8 dihedral orientations (`Identity`, `FlipH`, `FlipV`, `Rotate180`,
//! `Transpose`, `Rotate90`, `Rotate270`, `Transverse`) across all pixel widths
//! (`1, 2, 3, 4, 6, 8, 12, 16+` B/px) with SIMD register transposes (`pxn_x86`,
//! `rgb3_x86`, `pxn_neon`, and WASM SIMD128 4×4) under `#![forbid(unsafe_code)]`.

use core::cmp::min;
use whereat::At;
use zenpixels::{Orientation, PixelBuf, PixelView, PixelViewMut};

use crate::error::{ConvertError, err_at};

#[cfg(feature = "fast-transpose")]
use archmage::prelude::*;

// Private aliases so our preserved SIMD kernel files (`pxn_x86`, `rgb3_x86`, `pxn_neon`)
// operate directly on 0.5 `PixelView` / `PixelViewMut` with zero changes.
#[allow(dead_code)]
type PixelSlice<'a> = PixelView<'a>;
#[allow(dead_code)]
type PixelSliceMut<'a> = PixelViewMut<'a>;

const TILE: u32 = 32;

/// Apply `orientation` to `src`, returning a freshly allocated tightly packed [`PixelBuf`].
pub fn apply_orientation(
    src: &PixelView<'_>,
    orientation: Orientation,
) -> Result<PixelBuf, At<ConvertError>> {
    let (ow, oh) = orientation.output_dimensions(src.width(), src.rows());
    let mut out = PixelBuf::new(ow, oh, src.spec().clone())
        .map_err(|e| err_at((*e.error()).into()))?
        .with_hdr(src.hdr());
    let facts = src.facts();
    apply_orientation_into(src, orientation, &mut out.as_view_mut())?;
    out.set_facts(facts);
    Ok(out)
}

/// Apply `orientation` to `src`, writing into caller-provided `dst` with **zero heap allocation**.
pub fn apply_orientation_into(
    src: &PixelView<'_>,
    orientation: Orientation,
    dst: &mut PixelViewMut<'_>,
) -> Result<(), At<ConvertError>> {
    let w = src.width();
    let h = src.rows();
    let bpp = src.spec().layout.bytes_per_pixel();
    let (ow, oh) = orientation.output_dimensions(w, h);

    if dst.width() != ow
        || dst.rows() != oh
        || dst.spec().layout.bytes_per_pixel() != bpp
        || dst.spec().layout != src.spec().layout
    {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    if w == 0 || h == 0 || bpp == 0 {
        return Ok(());
    }

    match orientation {
        Orientation::Identity => {
            for y in 0..h {
                dst.row_mut(y).copy_from_slice(src.row(y));
            }
        }
        Orientation::FlipV => {
            for y in 0..h {
                dst.row_mut(y).copy_from_slice(src.row(h - 1 - y));
            }
        }
        Orientation::FlipH => {
            for y in 0..h {
                reverse_row(src.row(y), dst.row_mut(y), w as usize, bpp);
            }
        }
        Orientation::Rotate180 => {
            for y in 0..h {
                reverse_row(src.row(h - 1 - y), dst.row_mut(y), w as usize, bpp);
            }
        }
        Orientation::Transpose
        | Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transverse
        | _ => {
            do_transpose(src, dst, orientation, w, h, bpp);
        }
    }
    Ok(())
}

#[inline]
fn reverse_row(src: &[u8], dst: &mut [u8], width: usize, bpp: usize) {
    match bpp {
        1 => {
            for x in 0..width {
                dst[x] = src[width - 1 - x];
            }
        }
        2 => {
            for x in 0..width {
                let si = (width - 1 - x) * 2;
                let di = x * 2;
                dst[di..di + 2].copy_from_slice(&src[si..si + 2]);
            }
        }
        3 => {
            for x in 0..width {
                let si = (width - 1 - x) * 3;
                let di = x * 3;
                dst[di..di + 3].copy_from_slice(&src[si..si + 3]);
            }
        }
        4 => {
            for x in 0..width {
                let si = (width - 1 - x) * 4;
                let di = x * 4;
                dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }
        _ => {
            for x in 0..width {
                let si = (width - 1 - x) * bpp;
                let di = x * bpp;
                dst[di..di + bpp].copy_from_slice(&src[si..si + bpp]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn scatter_pixel(
    s: &[u8],
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    sx: u32,
    sy: u32,
    w: u32,
    h: u32,
    bpp: usize,
) {
    let (dx, dy) = orientation.forward_map(sx, sy, w, h);
    let si = sx as usize * bpp;
    let di = dx as usize * bpp;
    dst.row_mut(dy)[di..di + bpp].copy_from_slice(&s[si..si + bpp]);
}

fn do_transpose(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
) {
    if let Some(flips) = inverse_flips(orientation) {
        #[cfg(all(feature = "fast-transpose", target_arch = "x86_64"))]
        {
            macro_rules! v3_kernel {
                ($kernel:path) => {
                    if let Some(token) = X64V3Token::summon() {
                        $kernel(token, src, dst, orientation, w, h);
                        return;
                    }
                };
            }
            match bpp {
                1 => v3_kernel!(pxn_x86::transpose1_v3),
                2 => v3_kernel!(pxn_x86::transpose2_v3),
                3 => v3_kernel!(rgb3_x86::transpose3_v3),
                4 => v3_kernel!(pxn_x86::transpose4_v3),
                6 => v3_kernel!(pxn_x86::transpose6_v3),
                8 => v3_kernel!(pxn_x86::transpose8_v3),
                12 => v3_kernel!(pxn_x86::transpose12_v3),
                16 => v3_kernel!(pxn_x86::transpose16_v3),
                _ => {}
            }
        }
        #[cfg(all(feature = "fast-transpose", target_arch = "aarch64"))]
        {
            macro_rules! neon_kernel {
                ($kernel:path) => {
                    if let Some(token) = NeonToken::summon() {
                        $kernel(token, src, dst, orientation, w, h);
                        return;
                    }
                };
            }
            match bpp {
                1 => neon_kernel!(pxn_neon::transpose1_neon),
                2 => neon_kernel!(pxn_neon::transpose2_neon),
                3 => neon_kernel!(pxn_neon::transpose3_neon),
                4 => neon_kernel!(pxn_neon::transpose4_neon),
                6 => neon_kernel!(pxn_neon::transpose6_neon),
                8 => neon_kernel!(pxn_neon::transpose8_neon),
                12 => neon_kernel!(pxn_neon::transpose12_neon),
                16 => {
                    transpose16_deep(src, dst, orientation, w, h);
                    return;
                }
                _ => {}
            }
        }
        #[cfg(feature = "fast-transpose")]
        if bpp == 4 {
            transpose4_simd(src, dst, orientation, w, h);
            return;
        }
        match bpp {
            1 => return transpose_tiled::<1>(src, dst, flips, w, h),
            2 => return transpose_tiled::<2>(src, dst, flips, w, h),
            3 => return transpose_tiled::<3>(src, dst, flips, w, h),
            4 => return transpose_tiled::<4>(src, dst, flips, w, h),
            6 => return transpose_tiled::<6>(src, dst, flips, w, h),
            8 => return transpose_tiled::<8>(src, dst, flips, w, h),
            12 => return transpose_tiled::<12>(src, dst, flips, w, h),
            16 => return transpose16_deep(src, dst, orientation, w, h),
            _ => {}
        }
    }
    transpose_blocked(src, dst, orientation, w, h, bpp);
}

#[inline]
fn inverse_flips(orientation: Orientation) -> Option<(bool, bool)> {
    match orientation {
        Orientation::Transpose => Some((false, false)),
        Orientation::Rotate90 => Some((false, true)),
        Orientation::Rotate270 => Some((true, false)),
        Orientation::Transverse => Some((true, true)),
        _ => None,
    }
}

#[cfg_attr(
    all(
        feature = "fast-transpose",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    cold
)]
#[inline(never)]
fn transpose_tiled<const BPP: usize>(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    (flip_sx, flip_sy): (bool, bool),
    w: u32,
    h: u32,
) {
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let sstep: isize = if flip_sy {
        -(sstride as isize)
    } else {
        sstride as isize
    };
    let (ow, oh) = (h, w);

    let mut ty = 0;
    while ty < oh {
        let ty_end = min(ty + TILE, oh);
        let mut tx = 0;
        while tx < ow {
            let tx_end = min(tx + TILE, ow);
            let sy0 = (if flip_sy { h - 1 - tx } else { tx }) as usize;
            for dy in ty..ty_end {
                let sx = (if flip_sx { w - 1 - dy } else { dy }) as usize;
                let mut soff = (sy0 * sstride + sx * BPP) as isize;
                let drow = &mut dst.row_mut(dy)[tx as usize * BPP..tx_end as usize * BPP];
                for dpx in drow.as_chunks_mut::<BPP>().0 {
                    let s = soff as usize;
                    *dpx = sbytes[s..s + BPP].try_into().unwrap();
                    soff += sstep;
                }
            }
            tx += TILE;
        }
        ty += TILE;
    }
}

#[cfg_attr(
    all(
        feature = "fast-transpose",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    cold
)]
#[inline(never)]
fn transpose_blocked(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
) {
    let mut tile_y = 0;
    while tile_y < h {
        let y_end = min(tile_y + TILE, h);
        let mut tile_x = 0;
        while tile_x < w {
            let x_end = min(tile_x + TILE, w);
            for sy in tile_y..y_end {
                let s = src.row(sy);
                for sx in tile_x..x_end {
                    scatter_pixel(s, dst, orientation, sx, sy, w, h, bpp);
                }
            }
            tile_x += TILE;
        }
        tile_y += TILE;
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "fast-transpose")]
fn transpose_edges(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
    full_w: u32,
    full_h: u32,
) {
    for sy in 0..full_h {
        let s = src.row(sy);
        for sx in full_w..w {
            scatter_pixel(s, dst, orientation, sx, sy, w, h, bpp);
        }
    }
    for sy in full_h..h {
        let s = src.row(sy);
        for sx in 0..w {
            scatter_pixel(s, dst, orientation, sx, sy, w, h, bpp);
        }
    }
}

#[cfg(feature = "fast-transpose")]
#[inline]
fn tile_dest(
    orientation: Orientation,
    bx: u32,
    by: u32,
    r: u32,
    w: u32,
    h: u32,
) -> (u32, u32, bool) {
    match orientation {
        Orientation::Transpose => (bx + r, by, false),
        Orientation::Rotate90 => (bx + r, h - 4 - by, true),
        Orientation::Rotate270 => (w - 1 - bx - r, by, false),
        Orientation::Transverse => (w - 1 - bx - r, h - 4 - by, true),
        _ => unreachable!("tile_dest only handles the four transposing orientations"),
    }
}

#[cfg(feature = "fast-transpose")]
#[attune(dispatch, -_v3, -_neon, define(f32x4))]
fn transpose4_simd(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let token = Token::from_context();
    let full_w = w & !3;
    let full_h = h & !3;

    let mut by = 0;
    while by < full_h {
        let mut bx = 0;
        while bx < full_w {
            let xb = bx as usize * 4;
            let f0: [f32; 4] =
                bytemuck::cast::<[u8; 16], _>(src.row(by)[xb..xb + 16].try_into().unwrap());
            let f1: [f32; 4] =
                bytemuck::cast::<[u8; 16], _>(src.row(by + 1)[xb..xb + 16].try_into().unwrap());
            let f2: [f32; 4] =
                bytemuck::cast::<[u8; 16], _>(src.row(by + 2)[xb..xb + 16].try_into().unwrap());
            let f3: [f32; 4] =
                bytemuck::cast::<[u8; 16], _>(src.row(by + 3)[xb..xb + 16].try_into().unwrap());
            let mut rows = [
                f32x4::load_t(token, &f0),
                f32x4::load_t(token, &f1),
                f32x4::load_t(token, &f2),
                f32x4::load_t(token, &f3),
            ];
            f32x4::transpose_4x4(&mut rows);

            for r in 0..4u32 {
                let mut lanes = [0f32; 4];
                rows[r as usize].store(&mut lanes);
                let (drow, dcol, rev) = tile_dest(orientation, bx, by, r, w, h);
                if rev {
                    lanes.reverse();
                }
                let bytes: [u8; 16] = bytemuck::cast(lanes);
                let db = dcol as usize * 4;
                dst.row_mut(drow)[db..db + 16].copy_from_slice(&bytes);
            }
            bx += 4;
        }
        by += 4;
    }

    transpose_edges(src, dst, orientation, w, h, 4, full_w, full_h);
}

fn transpose16_deep(
    src: &PixelView<'_>,
    dst: &mut PixelViewMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let Some((flip_sx, flip_sy)) = inverse_flips(orientation) else {
        return transpose_blocked(src, dst, orientation, w, h, 16);
    };
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let dstride = dst.stride();
    let dbytes = dst.as_strided_bytes_mut();

    const T: usize = 16;
    let full_w = w & !15;
    let full_h = h & !15;
    let ntx = (full_w / 16) as usize;
    let nty = (full_h / 16) as usize;
    for tyi in 0..nty {
        let ty = if flip_sy { nty - 1 - tyi } else { tyi } * T;
        let dx = if flip_sy { h as usize - T - ty } else { ty };
        for txi in 0..ntx {
            let tx = (if flip_sx { ntx - 1 - txi } else { txi }) * T;
            let mut srows: [&[u8]; T] = [&[]; T];
            for (r, sr) in srows.iter_mut().enumerate() {
                let off = (ty + r) * sstride + tx * 16;
                *sr = &sbytes[off..off + T * 16];
            }
            for c in 0..T {
                let dy = if flip_sx {
                    w as usize - 1 - (tx + c)
                } else {
                    tx + c
                };
                let drow = &mut dbytes[dy * dstride + dx * 16..dy * dstride + (dx + T) * 16];
                for (k, chunk) in drow.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    let r = if flip_sy { T - 1 - k } else { k };
                    chunk.copy_from_slice(&srows[r][c * 16..c * 16 + 16]);
                }
            }
        }
    }
    let (sb, db) = (sbytes, dbytes);
    for y in 0..full_h {
        for x in full_w..w {
            let (dxx, dyy) = orientation.forward_map(x, y, w, h);
            let so = y as usize * sstride + x as usize * 16;
            let dofs = dyy as usize * dstride + dxx as usize * 16;
            db[dofs..dofs + 16].copy_from_slice(&sb[so..so + 16]);
        }
    }
    for y in full_h..h {
        for x in 0..w {
            let (dxx, dyy) = orientation.forward_map(x, y, w, h);
            let so = y as usize * sstride + x as usize * 16;
            let dofs = dyy as usize * dstride + dxx as usize * 16;
            db[dofs..dofs + 16].copy_from_slice(&sb[so..so + 16]);
        }
    }
}

#[cfg(all(feature = "fast-transpose", target_arch = "x86_64"))]
mod rgb3_x86;

#[cfg(all(feature = "fast-transpose", target_arch = "x86_64"))]
mod pxn_x86;

#[cfg(all(feature = "fast-transpose", target_arch = "aarch64"))]
mod pxn_neon;
