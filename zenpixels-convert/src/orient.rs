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
//! The inner per-tile transpose here is scalar. The fast follow-up replaces it
//! with a SIMD register transpose for the common element widths — magetypes
//! ships `transpose_4x4` / `transpose_8x8` and the `interleave_lo/hi` unpack
//! primitives, which map a 4-byte pixel onto a `u32` lane (the classic
//! `_MM_TRANSPOSE4_PS`-style shuffle cascade) and a 1-byte pixel onto the
//! 16×16 `punpck` cascade. The tiling structure here is exactly the scaffold
//! that kernel slots into, and the scalar path stays as the odd-`bpp` fallback
//! and the parity oracle.

use core::cmp::min;

use zenpixels::{Orientation, PixelBuffer, PixelSlice, PixelSliceMut};

/// Side length of the cache-blocking tile for transposing orientations, in
/// pixels. At `bpp = 4` a 32×32 tile touches 4 KiB of source and 4 KiB of
/// destination — comfortably inside L1 — while staying large enough to amortise
/// the per-tile loop overhead.
const TILE: u32 = 32;

/// Apply `orientation` to `src`, returning a new buffer with the pixels
/// physically rearranged.
///
/// The returned buffer's dimensions are
/// [`orientation.output_dimensions(src.width(), src.rows())`](Orientation::output_dimensions)
/// — width and height swap for the four axis-swapping orientations. The pixel
/// descriptor is preserved exactly (this moves whole `bpp`-sized pixels; it
/// never touches their contents), so it is format-, channel-, and bit-depth
/// agnostic. Strided input is handled.
///
/// `Orientation::Identity` still allocates and copies (callers that want to
/// skip the copy should check `orientation.is_identity()` themselves).
#[must_use]
pub fn apply_orientation(src: PixelSlice<'_>, orientation: Orientation) -> PixelBuffer {
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
                transpose_blocked(&src, &mut dst, orientation, w, h, bpp);
            }
        }
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

/// Cache-blocked transpose for the four axis-swapping orientations. The
/// per-element destination is `orientation.forward_map(sx, sy, w, h)`, which
/// encodes transpose + whatever reflection the orientation adds; tiling keeps
/// each block's scattered destination writes inside the cache.
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
                    let (dx, dy) = orientation.forward_map(sx, sy, w, h);
                    let si = sx as usize * bpp;
                    let di = dx as usize * bpp;
                    dst.row_mut(dy)[di..di + bpp].copy_from_slice(&s[si..si + bpp]);
                }
            }
            tile_x += TILE;
        }
        tile_y += TILE;
    }
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
}
