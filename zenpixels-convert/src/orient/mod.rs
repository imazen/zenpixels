//! Physical EXIF orientation baking — rotate / flip any pixel view or buffer.
//!
//! Supports all 8 dihedral orientations (`Identity`, `FlipH`, `FlipV`, `Rotate180`,
//! `Transpose`, `Rotate90`, `Rotate270`, `Transverse`) across all pixel widths
//! (`1, 2, 3, 4, 6, 8, 12, 16+` B/px) with SIMD register transposes (`pxn_x86`,
//! `rgb3_x86`, `pxn_neon`, and WASM SIMD128 4×4) under `#![forbid(unsafe_code)]`.

use core::cmp::min;
use whereat::At;
use zenpixels::{BandedBuf, Orientation, PixelBuf, PixelSink, PixelView, PixelViewMut};

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
///
/// Preserves `src.hdr()`, `src.facts()`, and `src.origin()`. The returned buffer has
/// [`Orientation::Identity`] because `orientation` has been physically applied to the pixels.
pub fn apply_orientation(
    src: &PixelView<'_>,
    orientation: Orientation,
) -> Result<PixelBuf, At<ConvertError>> {
    let (ow, oh) = orientation.output_dimensions(src.width(), src.rows());
    let mut out = PixelBuf::new(ow, oh, src.spec().clone())
        .map_err(|e| err_at((*e.error()).into()))?
        .with_hdr(src.hdr());
    if let Some(origin) = src.origin() {
        out = out.with_origin(origin.clone());
    }
    let facts = src.facts();
    apply_orientation_into(src, orientation, &mut out.as_view_mut())?;
    out.assume_facts(facts);
    Ok(out)
}

/// Apply explicit `orientation` to an owned [`PixelBuf`], returning `buf` **zero-copy** on
/// [`Orientation::Identity`] and always emitting [`Orientation::Identity`] metadata on the result
/// (matching [`apply_orientation`]).
///
/// Pending input `buf.orientation()` is not implicitly composed; use [`bake_orientation_owned`]
/// to apply and clear pending orientation.
pub fn apply_orientation_owned(
    mut buf: PixelBuf,
    orientation: Orientation,
) -> Result<PixelBuf, At<ConvertError>> {
    if orientation == Orientation::Identity {
        buf.set_orientation(Orientation::Identity);
        return Ok(buf);
    }
    apply_orientation(&buf.as_view(), orientation)
}

/// Physically bake `src.orientation()` into pixels, returning a tightly packed [`PixelBuf`]
/// with [`Orientation::Identity`].
pub fn bake_orientation(src: &PixelView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    apply_orientation(src, src.orientation())
}

/// Physically bake `buf.orientation()` into pixels, returning `buf` **zero-copy** when
/// `buf.orientation() == Orientation::Identity` and clearing the resulting orientation to
/// [`Orientation::Identity`].
pub fn bake_orientation_owned(buf: PixelBuf) -> Result<PixelBuf, At<ConvertError>> {
    let orient = buf.orientation();
    if orient == Orientation::Identity {
        return Ok(buf);
    }
    apply_orientation(&buf.as_view(), orient)
}

/// Physically bake `src.orientation()` into a [`BandedBuf`] with each band bounded by
/// `max_band_bytes` and [`Orientation::Identity`] on the result.
pub fn bake_orientation_banded(
    src: &BandedBuf,
    max_band_bytes: usize,
) -> Result<BandedBuf, At<ConvertError>> {
    apply_orientation_banded(src, src.orientation(), max_band_bytes)
}

/// Apply explicit `orientation` to a [`BandedBuf`], returning a freshly allocated [`BandedBuf`]
/// with each band bounded by `max_band_bytes`, [`Orientation::Identity`] metadata, and
/// **zero full-frame contiguous allocation**.
///
/// Preserves `src.hdr()`, `src.facts()`, and `src.origin()`.
pub fn apply_orientation_banded(
    src: &BandedBuf,
    orientation: Orientation,
    max_band_bytes: usize,
) -> Result<BandedBuf, At<ConvertError>> {
    let (ow, oh) = orientation.output_dimensions(src.width(), src.height());
    let mut out = BandedBuf::new(ow, oh, src.spec().clone(), max_band_bytes)
        .map_err(|e| err_at((*e.error()).into()))?
        .with_origin(src.origin().clone());
    apply_orientation_banded_into(src, orientation, &mut out)?;
    out.sink_finish(src.facts(), src.hdr());
    Ok(out)
}

/// Apply `orientation` from `src` into pre-allocated `dst` [`BandedBuf`] with
/// **zero heap allocation** and full SIMD tile transposes across every `(src_band, dst_band)`
/// intersection.
///
/// Works for arbitrary independent source and destination band heights and all 8 orientations.
pub fn apply_orientation_banded_into(
    src: &BandedBuf,
    orientation: Orientation,
    dst: &mut BandedBuf,
) -> Result<(), At<ConvertError>> {
    let w = src.width();
    let h = src.height();
    let (ow, oh) = orientation.output_dimensions(w, h);
    if dst.width() != ow || dst.height() != oh || dst.spec().layout != src.spec().layout {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    if dst.spec().color != src.spec().color {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if w == 0 || h == 0 || src.spec().layout.bytes_per_pixel() == 0 {
        return Ok(());
    }

    let src_band_rows = src.band_rows();
    let dst_band_rows = dst.band_rows();
    let src_band_count = src.band_count();
    let dst_band_count = dst.band_count();

    if !orientation.swaps_axes() {
        let flips_v = matches!(orientation, Orientation::FlipV | Orientation::Rotate180);
        for sb_idx in 0..src_band_count {
            let sy0 = (sb_idx as u32) * src_band_rows;
            let src_band = src.band_view(sb_idx).expect("valid src band");
            let sh = src_band.rows();
            let sy1 = sy0 + sh;
            let (m_dy0, m_dy1) = if flips_v {
                (h - sy1, h - sy0)
            } else {
                (sy0, sy1)
            };
            let first_db = (m_dy0 / dst_band_rows) as usize;
            let last_db = ((m_dy1 - 1) / dst_band_rows) as usize;
            for db_idx in first_db..=last_db {
                let dy0 = (db_idx as u32) * dst_band_rows;
                let mut dst_band = dst.band_view_mut(db_idx).expect("valid dst band");
                let dy1 = dy0 + dst_band.rows();
                let iy0 = dy0.max(m_dy0);
                let iy1 = dy1.min(m_dy1);
                if iy0 >= iy1 {
                    continue;
                }
                let ih = iy1 - iy0;
                let is_y0 = if flips_v { h - iy1 } else { iy0 };
                let src_sub = src_band
                    .sub_rows(is_y0 - sy0, ih)
                    .map_err(|e| err_at((*e.error()).into()))?;
                let mut dst_sub = dst_band
                    .sub_rows_mut(iy0 - dy0, ih)
                    .map_err(|e| err_at((*e.error()).into()))?;
                apply_orientation_into(&src_sub, orientation, &mut dst_sub)?;
            }
        }
    } else {
        let rev_sy_to_dx = matches!(orientation, Orientation::Rotate90 | Orientation::Transverse);
        let rev_sx_to_dy = matches!(
            orientation,
            Orientation::Rotate270 | Orientation::Transverse
        );
        for sb_idx in 0..src_band_count {
            let sy0 = (sb_idx as u32) * src_band_rows;
            let src_band = src.band_view(sb_idx).expect("valid src band");
            let sh = src_band.rows();
            let sy1 = sy0 + sh;
            let dx0 = if rev_sy_to_dx { h - sy1 } else { sy0 };
            for db_idx in 0..dst_band_count {
                let dy0 = (db_idx as u32) * dst_band_rows;
                let mut dst_band = dst.band_view_mut(db_idx).expect("valid dst band");
                let dh = dst_band.rows();
                let dy1 = dy0 + dh;
                let sx0 = if rev_sx_to_dy { w - dy1 } else { dy0 };
                let src_tile = src_band
                    .crop(sx0, 0, dh, sh)
                    .map_err(|e| err_at((*e.error()).into()))?;
                let mut dst_tile = dst_band
                    .crop_mut(dx0, 0, sh, dh)
                    .map_err(|e| err_at((*e.error()).into()))?;
                apply_orientation_into(&src_tile, orientation, &mut dst_tile)?;
            }
        }
    }
    Ok(())
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
    if dst.spec().color != src.spec().color {
        return Err(err_at(ConvertError::UnsupportedRoute));
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
    // For a 1-row image, `stride` is unused (and may even exceed `isize::MAX`); normalize to 0.
    let sstride = if h <= 1 { 0 } else { src.stride() };
    let (ow, oh) = (h, w);

    let mut ty = 0u32;
    while ty < oh {
        let ty_end = min(ty.saturating_add(TILE), oh);
        let mut tx = 0u32;
        while tx < ow {
            let tx_end = min(tx.saturating_add(TILE), ow);
            let sy0 = (if flip_sy { h - 1 - tx } else { tx }) as usize;
            for dy in ty..ty_end {
                let sx = (if flip_sx { w - 1 - dy } else { dy }) as usize;
                let mut soff = sy0 * sstride + sx * BPP;
                let drow = &mut dst.row_mut(dy)[tx as usize * BPP..tx_end as usize * BPP];
                let chunks = drow.as_chunks_mut::<BPP>().0;
                let n = chunks.len();
                for (i, dpx) in chunks.iter_mut().enumerate() {
                    *dpx = sbytes[soff..soff + BPP].try_into().unwrap();
                    if i + 1 < n {
                        soff = if flip_sy {
                            soff - sstride
                        } else {
                            soff + sstride
                        };
                    }
                }
            }
            tx = tx_end;
        }
        ty = ty_end;
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
    let mut tile_y = 0u32;
    while tile_y < h {
        let y_end = min(tile_y.saturating_add(TILE), h);
        let mut tile_x = 0u32;
        while tile_x < w {
            let x_end = min(tile_x.saturating_add(TILE), w);
            for sy in tile_y..y_end {
                let s = src.row(sy);
                for sx in tile_x..x_end {
                    scatter_pixel(s, dst, orientation, sx, sy, w, h, bpp);
                }
            }
            tile_x = x_end;
        }
        tile_y = y_end;
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
