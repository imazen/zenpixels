//! Physical orientation baking — rotate / flip a whole pixel buffer.
//!
//! [`apply_orientation`] takes a (possibly strided) [`PixelSlice`] and an
//! [`Orientation`] and returns a fresh, tightly-allocated [`PixelBuffer`] with
//! the pixels physically rearranged. It is the "bake" half of the zen
//! orientation model: codecs that decode to a raster buffer and are asked to
//! resolve orientation (`OrientationHint::bakes()` is true) call this; the
//! cheap coordinate math (`Orientation::forward_map` / `output_dimensions`)
//! lives in `zenpixels`, and this is the buffer operation that consumes it.
//!
//! # Algorithm
//!
//! The eight orientations split into two classes:
//!
//! * **Non-transposing** (`Identity`, `FlipH`, `FlipV`, `Rotate180`) — the
//!   output has the same dimensions, so each output row maps to exactly one
//!   input row. `Identity`/`FlipV` are pure row copies (memcpy, reordered for
//!   `FlipV`); `FlipH`/`Rotate180` additionally reverse the `bpp`-sized
//!   elements within each row. These are memory-bandwidth bound — a scalar
//!   `copy_from_slice` per row already runs at copy speed.
//!
//! * **Transposing** (`Transpose`, `Rotate90`, `Rotate270`, `Transverse`) —
//!   width and height swap, and the access pattern is a matrix transpose, which
//!   is the cache-hostile case: a naïve element loop strides one of the two
//!   buffers by a full row per step and thrashes the cache once the image
//!   exceeds L1/L2. We use the standard fix — **loop tiling (cache blocking)**:
//!   process the image in `TILE`×`TILE` blocks so each block's source and
//!   destination footprints (`TILE*TILE*bpp` bytes each) stay resident while we
//!   transpose them. The orientation's reflection (the `h-1-sy` / `w-1-sx`
//!   terms that turn a bare transpose into a 90°/270° rotation or anti-diagonal
//!   flip) is folded into the per-element destination address via
//!   `forward_map`, so the whole thing is a single pass with no intermediate
//!   buffer.
//!
//! For **4-byte pixels on x86_64** the per-tile transpose is SIMD: full 4×4
//! tiles go through magetypes' `f32x4::transpose_4x4` (the classic
//! `_MM_TRANSPOSE4_PS` `unpacklo/hi` + `movelh/hl` cascade), dispatched by
//! `incant!` with a scalar fallback when AVX2 is absent. Each pixel rides as one
//! f32 lane — the kernel only shuffles whole 32-bit lanes (no float math), so
//! reinterpreting the bytes as f32 is bit-exact for any 4-byte format, NaN bit
//! patterns included. The non-multiple-of-4 edge strips, every other element
//! width, and non-x86 targets use the cache-blocked scalar path, which is also
//! the parity oracle (`simd_transpose_matches_scalar_reference_rgba8`).
//! (1- and 2-byte SIMD transpose — the 16×16 `punpck` cascade — is a possible
//! follow-up; gray / 16-bit currently go scalar.)

use core::cmp::min;

use zenpixels::{Orientation, PixelBuffer, PixelSlice, PixelSliceMut};

use crate::error::ConvertError;

// SIMD path imports (x86_64): `incant!`, `X64V3Token`, `ScalarToken`,
// `#[arcane]` from the archmage prelude; the generic `f32x4` from magetypes.
#[cfg(target_arch = "x86_64")]
use archmage::prelude::*;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x4;

/// Side length of the cache-blocking tile for transposing orientations, in
/// pixels. At `bpp = 4` a 32×32 tile touches 4 KiB of source and 4 KiB of
/// destination — comfortably inside L1 — while staying large enough to amortise
/// the per-tile loop overhead.
const TILE: u32 = 32;

/// Apply `orientation` to `src`, returning a freshly-allocated buffer with the
/// pixels physically rearranged.
///
/// The returned buffer's dimensions are
/// [`orientation.output_dimensions(src.width(), src.rows())`](Orientation::output_dimensions)
/// — width and height swap for the four axis-swapping orientations. The pixel
/// descriptor is preserved exactly (this moves whole `bpp`-sized pixels; it
/// never touches their contents), so it is format-, channel-, and bit-depth
/// agnostic. Strided input is handled.
///
/// This allocates the output every call. Callers that reuse or pool a target
/// buffer (e.g. a codec `decode_into`, or an image proxy processing same-size
/// images) should use [`apply_orientation_into`] to avoid the allocation.
/// `Orientation::Identity` still allocates and copies (callers that want to skip
/// the copy entirely should check `orientation.is_identity()` themselves).
#[must_use]
pub fn apply_orientation(src: PixelSlice<'_>, orientation: Orientation) -> PixelBuffer {
    let (ow, oh) = orientation.output_dimensions(src.width(), src.rows());
    let desc = src.descriptor();
    let mut out = PixelBuffer::new(ow, oh, desc);
    // The buffer is constructed to the exact output geometry + descriptor, so
    // the size/format check inside `apply_orientation_into` cannot fail.
    apply_orientation_into(src, orientation, out.as_slice_mut())
        .expect("apply_orientation: freshly allocated buffer matches output geometry");
    out
}

/// Apply `orientation` to `src`, writing into a caller-provided `dst` — **no
/// allocation**, so callers can reuse / pool the target across many calls.
///
/// `dst` must already have the oriented geometry
/// ([`orientation.output_dimensions(src.width(), src.rows())`](Orientation::output_dimensions))
/// and the same bytes-per-pixel as `src`; otherwise [`ConvertError::BufferSize`]
/// is returned and `dst` is left untouched. The allocating [`apply_orientation`]
/// is a thin wrapper over this.
pub fn apply_orientation_into(
    src: PixelSlice<'_>,
    orientation: Orientation,
    mut dst: PixelSliceMut<'_>,
) -> Result<(), ConvertError> {
    let w = src.width();
    let h = src.rows();
    let bpp = src.descriptor().bytes_per_pixel();
    let (ow, oh) = orientation.output_dimensions(w, h);

    let dst_bpp = dst.descriptor().bytes_per_pixel();
    if dst.width() != ow || dst.rows() != oh || dst_bpp != bpp {
        return Err(ConvertError::BufferSize {
            expected: ow as usize * oh as usize * bpp,
            actual: dst.width() as usize * dst.rows() as usize * dst_bpp,
        });
    }
    if w == 0 || h == 0 || bpp == 0 {
        return Ok(());
    }

    {
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
            // Axis-swapping: cache-blocked transpose with the orientation's
            // reflection folded into the destination address. The `_` arm makes
            // this the correct (if unoptimised) fallback for any orientation
            // added to the `#[non_exhaustive]` enum in future — it scatters by
            // `forward_map`, which is defined for every variant.
            Orientation::Transpose
            | Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Transverse
            | _ => {
                do_transpose(&src, &mut dst, orientation, w, h, bpp);
            }
        }
    }
    Ok(())
}

/// Bench-only A/B handle: bake `orientation` via the cache-blocked **scalar**
/// transpose, bypassing the SIMD kernel, so `bench_orient` can compare the two
/// paths on identical input. Only meaningful for the transposing orientations.
#[cfg(feature = "__bench_orient")]
#[doc(hidden)]
#[must_use]
pub fn __bench_apply_orientation_scalar(
    src: PixelSlice<'_>,
    orientation: Orientation,
) -> PixelBuffer {
    let w = src.width();
    let h = src.rows();
    let desc = src.descriptor();
    let bpp = desc.bytes_per_pixel();
    let (ow, oh) = orientation.output_dimensions(w, h);
    let mut out = PixelBuffer::new(ow, oh, desc);
    if w == 0 || h == 0 || bpp == 0 {
        return out;
    }
    {
        let mut dst = out.as_slice_mut();
        transpose_blocked(&src, &mut dst, orientation, w, h, bpp);
    }
    out
}

/// Copy one row, reversing the order of `bpp`-sized pixels (`FlipH` per row).
#[inline]
fn reverse_row(s: &[u8], d: &mut [u8], width: usize, bpp: usize) {
    for x in 0..width {
        let si = (width - 1 - x) * bpp;
        let di = x * bpp;
        d[di..di + bpp].copy_from_slice(&s[si..si + bpp]);
    }
}

/// Scatter one source pixel `(sx, sy)` to its oriented destination.
#[inline]
#[allow(clippy::too_many_arguments)] // per-pixel helper; an args struct would add overhead/noise
fn scatter_pixel(
    s: &[u8],
    dst: &mut PixelSliceMut<'_>,
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

/// Dispatch the axis-swapping orientations: an x86_64 SIMD 4×4 register
/// transpose for 4-byte pixels (the common decoder output), the cache-blocked
/// scalar path otherwise.
fn do_transpose(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // Only the four known transposing orientations have a `tile_dest`
        // mapping; a future `#[non_exhaustive]` variant falls through to scalar.
        if bpp == 4
            && matches!(
                orientation,
                Orientation::Transpose
                    | Orientation::Rotate90
                    | Orientation::Rotate270
                    | Orientation::Transverse
            )
        {
            incant!(transpose4(src, dst, orientation, w, h), [-v4]);
            return;
        }
    }
    transpose_blocked(src, dst, orientation, w, h, bpp);
}

/// Cache-blocked scalar transpose for the four axis-swapping orientations. The
/// per-element destination is `orientation.forward_map(sx, sy, w, h)`, which
/// encodes transpose + whatever reflection the orientation adds; tiling keeps
/// each block's scattered destination writes inside the cache. This is the
/// portable path and the parity oracle for the SIMD kernel.
fn transpose_blocked(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
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

/// Scalar scatter for the edge strips a 4×4-tiled SIMD pass leaves uncovered:
/// the right strip (`cols [full_w, w)`) and the bottom strip (`rows [full_h,
/// h)`, which also covers the bottom-right corner). No overlap between strips.
#[allow(clippy::too_many_arguments)] // edge-strip helper; mirrors the scatter-loop signature
fn transpose_edges(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
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

// ── SIMD 4×4 register transpose (x86_64, 4-byte pixels) ──────────────────────

/// Destination of transposed-tile row `r` (transposed row index 0..4) for a
/// source 4×4 tile at `(bx, by)`, as `(dst_row, dst_col_start, reverse_lanes)`.
/// Derived from `Orientation::forward_map`: a bare `Transpose` writes row `r` to
/// `dst[bx+r][by..]`; the rotations/anti-diagonal add a row/col reflection.
/// `by`/`bx` are multiples of 4 with `by+4 ≤ h`, `bx+4 ≤ w`, so the subtractions
/// never underflow.
#[cfg(target_arch = "x86_64")]
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

/// SIMD path: transpose full 4×4 RGBA8 tiles via `f32x4::transpose_4x4`
/// (the classic `_MM_TRANSPOSE4_PS` shuffle cascade), scalar for the edges.
///
/// Each 4-byte pixel rides as one f32 lane. The transpose only *shuffles whole
/// 32-bit lanes* (`unpacklo/hi`, `movelh/hl` — no float arithmetic), so the
/// reinterpret is bit-exact for any 4-byte pixel format, including bit patterns
/// that happen to be NaN.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn transpose4_v3(
    token: X64V3Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    let full_w = w & !3; // largest multiple of 4 ≤ w
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
                f32x4::load(token, &f0),
                f32x4::load(token, &f1),
                f32x4::load(token, &f2),
                f32x4::load(token, &f3),
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

/// Scalar fallback selected by `incant!` when AVX2 is unavailable.
#[cfg(target_arch = "x86_64")]
fn transpose4_scalar(
    _token: ScalarToken,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    transpose_blocked(src, dst, orientation, w, h, 4);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::PixelDescriptor;

    /// Build a tightly-packed source slice from raw bytes.
    fn slice<'a>(data: &'a [u8], w: u32, h: u32, desc: PixelDescriptor) -> PixelSlice<'a> {
        PixelSlice::new(data, w, h, w as usize * desc.bytes_per_pixel(), desc).unwrap()
    }

    /// A 3×2 gray8 image with per-pixel values 0..6:
    ///   row0: 0 1 2
    ///   row1: 3 4 5
    const SRC_3X2: [u8; 6] = [0, 1, 2, 3, 4, 5];

    /// Expected output of each orientation on `SRC_3X2`, hand-derived from the
    /// rotation geometry (NOT from `forward_map` — this is the independent
    /// oracle). `(out_w, out_h, bytes)`.
    fn expected_3x2(o: Orientation) -> (u32, u32, Vec<u8>) {
        match o {
            Orientation::Identity => (3, 2, vec![0, 1, 2, 3, 4, 5]),
            Orientation::FlipH => (3, 2, vec![2, 1, 0, 5, 4, 3]),
            Orientation::FlipV => (3, 2, vec![3, 4, 5, 0, 1, 2]),
            Orientation::Rotate180 => (3, 2, vec![5, 4, 3, 2, 1, 0]),
            // transposing → dims swap to 2×3
            Orientation::Transpose => (2, 3, vec![0, 3, 1, 4, 2, 5]),
            Orientation::Rotate90 => (2, 3, vec![3, 0, 4, 1, 5, 2]),
            Orientation::Rotate270 => (2, 3, vec![2, 5, 1, 4, 0, 3]),
            Orientation::Transverse => (2, 3, vec![5, 2, 4, 1, 3, 0]),
            _ => unreachable!("non-exhaustive Orientation in test oracle"),
        }
    }

    #[test]
    fn all_orientations_match_hand_derived_oracle_gray8() {
        let desc = PixelDescriptor::GRAY8;
        for &o in &Orientation::ALL {
            let out = apply_orientation(slice(&SRC_3X2, 3, 2, desc), o);
            let (ew, eh, ebytes) = expected_3x2(o);
            assert_eq!((out.width(), out.height()), (ew, eh), "{o:?} dims");
            // Compare row-by-row (output stride may be SIMD-aligned, not tight).
            let s = out.as_slice();
            for y in 0..eh {
                let got = s.row(y);
                let exp = &ebytes[y as usize * ew as usize..][..ew as usize];
                assert_eq!(got, exp, "{o:?} row {y}");
            }
        }
    }

    #[test]
    fn all_orientations_match_oracle_rgba8() {
        // Same geometry, but each pixel carries 4 distinct channel bytes so a
        // within-pixel byte-order bug would show. pixel v -> [v, v+64, v+128, 255].
        let desc = PixelDescriptor::RGBA8;
        let mut src = Vec::new();
        for v in 0u8..6 {
            src.extend_from_slice(&[v, v + 64, v + 128, 255]);
        }
        for &o in &Orientation::ALL {
            let out = apply_orientation(slice(&src, 3, 2, desc), o);
            let (ew, eh, gray) = expected_3x2(o);
            assert_eq!((out.width(), out.height()), (ew, eh), "{o:?} dims");
            let s = out.as_slice();
            for y in 0..eh {
                let got = s.row(y);
                for x in 0..ew {
                    let v = gray[(y * ew + x) as usize];
                    let exp = [v, v + 64, v + 128, 255];
                    assert_eq!(&got[x as usize * 4..][..4], &exp, "{o:?} px ({x},{y})");
                }
            }
        }
    }

    /// Deterministic pseudo-random byte (no Math.random/Date in tests anyway).
    fn fill(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut s = 0x9e3779b9u32;
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((s >> 24) as u8);
        }
        v
    }

    #[test]
    fn roundtrip_orientation_then_inverse_is_identity() {
        // apply(apply(img, o), o.inverse()) == img, for every orientation and a
        // spread of element sizes and odd dimensions.
        for &desc in &[
            PixelDescriptor::GRAY8,   // 1
            PixelDescriptor::GRAYA8,  // 2
            PixelDescriptor::RGB8,    // 3
            PixelDescriptor::RGBA8,   // 4
            PixelDescriptor::RGBAF32, // 16
        ] {
            let bpp = desc.bytes_per_pixel();
            for &(w, h) in &[(1u32, 1u32), (17, 13), (33, 31), (64, 48)] {
                let data = fill(w as usize * h as usize * bpp);
                for &o in &Orientation::ALL {
                    let once = apply_orientation(slice(&data, w, h, desc), o);
                    let back = apply_orientation(once.as_slice(), o.inverse());
                    assert_eq!(
                        (back.width(), back.height()),
                        (w, h),
                        "{o:?} {desc:?} {w}x{h}"
                    );
                    for y in 0..h {
                        let exp = &data[y as usize * w as usize * bpp..][..w as usize * bpp];
                        assert_eq!(
                            back.as_slice().row(y),
                            exp,
                            "{o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compose_matches_sequential_application() {
        // apply(img, a.then(b)) == apply(apply(img, a), b) — ties the baker to
        // the D4 group algebra in zenpixels.
        let desc = PixelDescriptor::RGBA8;
        let (w, h) = (11u32, 7u32);
        let data = fill(w as usize * h as usize * 4);
        for &a in &Orientation::ALL {
            for &b in &Orientation::ALL {
                let seq =
                    apply_orientation(apply_orientation(slice(&data, w, h, desc), a).as_slice(), b);
                let fused = apply_orientation(slice(&data, w, h, desc), a.then(b));
                assert_eq!(
                    (seq.width(), seq.height()),
                    (fused.width(), fused.height()),
                    "{a:?}.then({b:?}) dims"
                );
                for y in 0..seq.height() {
                    assert_eq!(
                        seq.as_slice().row(y),
                        fused.as_slice().row(y),
                        "{a:?}.then({b:?}) row {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn handles_strided_source() {
        // A source whose stride exceeds width*bpp must produce the same result
        // as a tight one (padding bytes must be ignored).
        let desc = PixelDescriptor::RGBA8;
        let (w, h) = (5u32, 4u32);
        let tight_stride = w as usize * 4;
        let padded_stride = tight_stride + 12;
        let tight = fill(tight_stride * h as usize);
        let mut padded = vec![0xABu8; padded_stride * h as usize];
        for y in 0..h as usize {
            padded[y * padded_stride..y * padded_stride + tight_stride]
                .copy_from_slice(&tight[y * tight_stride..][..tight_stride]);
        }
        for &o in &Orientation::ALL {
            // PixelSlice is a non-Copy view; build a fresh one per iteration.
            let tight_slice = PixelSlice::new(&tight, w, h, tight_stride, desc).unwrap();
            let padded_slice = PixelSlice::new(&padded, w, h, padded_stride, desc).unwrap();
            let a = apply_orientation(tight_slice, o);
            let b = apply_orientation(padded_slice, o);
            for y in 0..a.height() {
                assert_eq!(a.as_slice().row(y), b.as_slice().row(y), "{o:?} row {y}");
            }
        }
    }

    /// Gold-standard parity gate: the (SIMD on x86_64) `apply_orientation` must
    /// match the explicit scalar `transpose_blocked` for 4-byte pixels across
    /// the four transposing orientations and a spread of dimensions — full 4×4
    /// tiles (8×8, 16×16, 64×48), edge-only (3×3, 1×1), and mixed full+edge
    /// (17×13, 9×7, 12×4, 4×12, 5×5). This is what proves the SIMD kernel +
    /// edge handling are correct against the portable oracle.
    #[test]
    fn simd_transpose_matches_scalar_reference_rgba8() {
        let desc = PixelDescriptor::RGBA8;
        let dims = [
            (8u32, 8u32),
            (16, 16),
            (64, 48),
            (17, 13),
            (9, 7),
            (12, 4),
            (4, 12),
            (3, 3),
            (1, 1),
            (5, 5),
        ];
        for &(w, h) in &dims {
            let data = fill(w as usize * h as usize * 4);
            for &o in &[
                Orientation::Transpose,
                Orientation::Rotate90,
                Orientation::Rotate270,
                Orientation::Transverse,
            ] {
                // Path under test (SIMD on x86_64, scalar elsewhere).
                let got = apply_orientation(slice(&data, w, h, desc), o);
                // Explicit scalar reference via the cache-blocked oracle.
                let (ow, oh) = o.output_dimensions(w, h);
                let mut reference = PixelBuffer::new(ow, oh, desc);
                {
                    let src = slice(&data, w, h, desc);
                    let mut d = reference.as_slice_mut();
                    transpose_blocked(&src, &mut d, o, w, h, 4);
                }
                for y in 0..oh {
                    assert_eq!(
                        got.as_slice().row(y),
                        reference.as_slice().row(y),
                        "{o:?} {w}x{h} row {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn into_writes_caller_buffer_and_is_reusable() {
        // One target buffer, reused across four transposing orientations (all
        // share the swapped 13×17 geometry for a 17×13 input) — proves the
        // no-alloc reuse path and that it matches the allocating version.
        let desc = PixelDescriptor::RGBA8;
        let (w, h) = (17u32, 13u32);
        let data = fill(w as usize * h as usize * 4);
        let (ow, oh) = Orientation::Rotate90.output_dimensions(w, h);
        let mut target = PixelBuffer::new(ow, oh, desc);
        for &o in &[
            Orientation::Rotate90,
            Orientation::Rotate270,
            Orientation::Transverse,
            Orientation::Transpose,
        ] {
            apply_orientation_into(slice(&data, w, h, desc), o, target.as_slice_mut())
                .expect("into should accept a correctly-sized buffer");
            let want = apply_orientation(slice(&data, w, h, desc), o);
            for y in 0..oh {
                assert_eq!(
                    target.as_slice().row(y),
                    want.as_slice().row(y),
                    "{o:?} row {y}"
                );
            }
        }
    }

    #[test]
    fn into_rejects_wrong_sized_dst() {
        // Rotate90 of 8×6 needs a 6×8 target; an 8×6 buffer (same byte count,
        // wrong dims) must be rejected with BufferSize, leaving dst untouched.
        let desc = PixelDescriptor::RGBA8;
        let (w, h) = (8u32, 6u32);
        let data = fill(w as usize * h as usize * 4);
        let mut wrong = PixelBuffer::new(w, h, desc); // 8×6, but Rotate90 → 6×8
        let result = apply_orientation_into(
            slice(&data, w, h, desc),
            Orientation::Rotate90,
            wrong.as_slice_mut(),
        );
        assert!(
            matches!(result, Err(ConvertError::BufferSize { .. })),
            "expected BufferSize, got {result:?}"
        );
    }
}
