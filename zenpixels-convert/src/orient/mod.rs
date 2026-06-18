//! Physical orientation baking — rotate / flip a whole pixel buffer.
//!
//! [`apply_orientation`] takes a (possibly strided) [`PixelSlice`] and an
//! [`Orientation`] and returns a fresh, tightly-allocated [`PixelBuffer`] with
//! the pixels physically rearranged. It is the "bake" half of the zen
//! orientation model: codecs that decode to a raster buffer and are asked to
//! resolve orientation (`OrientationHint::bakes()` is true) call this; the
//! cheap coordinate math (`Orientation::forward_map` / `output_dimensions`)
//! lives in `zenpixels`, and this is the buffer operation that consumes it.
//! [`apply_orientation_into`] writes a caller-owned buffer (no allocation), and
//! [`apply_orientation_in_place`] permutes the backing allocation itself.
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
//! The transpose runs in one of two tiers — selected at compile time by the
//! `fast-transpose` feature and at runtime by CPU capability. Both share the
//! cache-blocking structure above and fold the reflection into the destination
//! address; they differ only in how each tile is moved. `do_transpose` picks.
//!
//! * **Portable scalar** — the default, and the path on pre-AVX2 x86 and any
//!   arch without a SIMD kernel. `transpose_tiled` is a per-width
//!   monomorphised gather: the four transposing maps are *separable*, so along
//!   one destination row the source column is fixed and the source offset steps
//!   by ±stride. That replaces the generic path's per-element `forward_map`,
//!   `row_mut`, and variable-length copy with one bounds check and a fixed-size
//!   `BPP`-byte copy per pixel, writing the destination sequentially (zenjpeg#150
//!   measured the generic path losing to a naive linear-write gather at 3 bpp;
//!   this beats both). 16-byte pixels use `transpose16_deep` — a transpose there
//!   is pure block movement that autovectorises — and the generic
//!   `transpose_blocked` scatter covers any other width.
//!
//! * **SIMD register transpose** (`fast-transpose`) — per-pixel-width kernels
//!   that transpose a register-sized tile with an unpack/zip shuffle cascade,
//!   the shape production transpose libraries (ermig1979/Simd, fast_transpose,
//!   libyuv) win with. x86-64-v3 (AVX2) kernels cover 1/2/3/4/6/8/12 bpp in
//!   `pxn_x86` (24-bit RGB8 in `rgb3_x86`, which expands 3→4 bytes, transposes
//!   u32 lanes, then compresses); aarch64 NEON kernels cover the same widths in
//!   `pxn_neon`; 16-byte pixels fall to `transpose16_deep`. On other
//!   `fast-transpose` arches (wasm, …) only 4-byte pixels are SIMD — via
//!   magetypes' `f32x4::transpose_4x4` (the classic `_MM_TRANSPOSE4_PS` shuffle,
//!   generated for SSE/NEON/SIMD128/scalar from one `#[magetypes]` body and
//!   dispatched by `incant!`) — and other widths take the scalar tier.
//!
//! A 4-byte pixel rides a transpose as one 32-bit lane; the kernels shuffle
//! whole lanes only (no arithmetic), so the byte reinterpret is bit-exact for
//! any 4-byte format, NaN patterns included. Every SIMD kernel transposes the
//! full tiles and leaves the right/bottom edge strips to a scalar `forward_map`
//! scatter (`scalar_rect` / `scalar_edges`). `transpose_blocked` (the generic
//! `forward_map` scatter) is the parity oracle every fast path is tested against
//! (`simd_transpose_matches_scalar_reference_rgba8`,
//! `tiled_transpose_matches_blocked_reference_across_bpp`,
//! `exhaustive_dense_dims_all_orientations_vs_oracle`) and the correct fallback
//! for any orientation added to the `#[non_exhaustive]` enum later.

use core::cmp::min;

use zenpixels::{InPlacePixels, Orientation, PixelBuffer, PixelSlice, PixelSliceMut};

use crate::error::ConvertError;

// Cross-arch SIMD: the `#[magetypes(...)]` codegen attribute + `incant!`
// runtime dispatch from the archmage prelude, and the token-parameterized
// generic `f32x4` from magetypes — whose `transpose_4x4` lowers to SSE
// `_MM_TRANSPOSE4_PS` on x86, the NEON `zip`/`trn` cascade on aarch64, and the
// `i32x4.shuffle` cascade on wasm128.
#[cfg(feature = "fast-transpose")]
use archmage::prelude::*;
#[cfg(feature = "fast-transpose")]
use magetypes::simd::generic::f32x4 as GenericF32x4;

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

/// Largest bytes-per-pixel the in-place path's per-element temp supports — covers
/// every current format up to RGBA f32 (16 bytes).
const MAX_INPLACE_BPP: usize = 16;

/// Bake `orientation` into `dst` **in place**, reusing its allocation — no second
/// pixel buffer (the transposing orientations would otherwise need a 2× transient).
///
/// Consumes the mutable view, permutes the bytes within the backing allocation,
/// and returns a re-described **tight-stride** `PixelSliceMut` over the same
/// memory — dimensions swapped for the four transposing orientations. Like the
/// no-alloc reduction APIs, the returned view carries the new geometry; the
/// source `PixelBuffer`'s own `width()`/`height()` go stale, so use the returned
/// view. Square images transpose via an in-place diagonal swap; non-square via
/// cycle-following (an `n`-element visited scratch — not a 2× pixel buffer).
///
/// Returns [`ConvertError::BufferSize`] if `bpp` exceeds 16 (the per-element temp
/// limit) or if re-describing the output fails.
pub fn apply_orientation_in_place(
    dst: &mut PixelBuffer,
    orientation: Orientation,
) -> Result<(), ConvertError> {
    let bpp = dst.descriptor().bytes_per_pixel();
    if bpp == 0 || bpp > MAX_INPLACE_BPP {
        return Err(ConvertError::BufferSize {
            expected: MAX_INPLACE_BPP,
            actual: bpp,
        });
    }
    // The eight known orientations all have an in-place mapping; a future
    // `#[non_exhaustive]` variant falls back to the allocating
    // `apply_orientation` at the caller.
    if !matches!(
        orientation,
        Orientation::Identity
            | Orientation::FlipH
            | Orientation::FlipV
            | Orientation::Rotate180
            | Orientation::Transpose
            | Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Transverse
    ) {
        return Err(ConvertError::BufferSize {
            expected: MAX_INPLACE_BPP,
            actual: 0,
        });
    }
    dst.transform_in_place(|px| orient_in_place_impl(px, orientation));
    Ok(())
}

/// The transform body behind [`apply_orientation_in_place`]: permute the
/// bytes and return the re-described tight-stride view for
/// [`PixelBuffer::transform_in_place`] to adopt.
fn orient_in_place_impl(px: InPlacePixels<'_>, orientation: Orientation) -> PixelSliceMut<'_> {
    let InPlacePixels {
        bytes,
        width: w,
        rows: h,
        stride: in_stride,
        descriptor: desc,
        color,
        ..
    } = px;
    let bpp = desc.bytes_per_pixel();
    let (ow, oh) = orientation.output_dimensions(w, h);
    let tight = w as usize * bpp;
    let out_stride = ow as usize * bpp;
    let out_len = out_stride * oh as usize;

    fn rewrap<'b>(
        bytes: &'b mut [u8],
        ow: u32,
        oh: u32,
        out_stride: usize,
        desc: zenpixels::PixelDescriptor,
        color: Option<alloc::sync::Arc<zenpixels::ColorContext>>,
    ) -> PixelSliceMut<'b> {
        let out = PixelSliceMut::new(bytes, ow, oh, out_stride, desc)
            .expect("oriented in-place geometry is always valid");
        match color {
            Some(c) => out.with_color_context(c),
            None => out,
        }
    }

    if w == 0 || h == 0 {
        return rewrap(&mut bytes[..out_len], ow, oh, out_stride, desc, color);
    }

    // 1. Compact to tight (drop any row padding) so the transpose is a clean
    //    permutation of a contiguous element array.
    if in_stride != tight {
        for y in 1..h as usize {
            bytes.copy_within(y * in_stride..y * in_stride + tight, y * tight);
        }
    }
    let content = &mut bytes[..tight * h as usize];

    // 2/3. Permute in place. Transposing orientations transpose the tight w×h
    //      grid (→ h×w = ow×oh) then add the orientation's reflection.
    match orientation {
        Orientation::Identity => {}
        Orientation::FlipH => inplace_flip_h(content, w, h, bpp),
        Orientation::FlipV => inplace_flip_v(content, w, h, bpp),
        Orientation::Rotate180 => inplace_reverse_elements(content, bpp),
        Orientation::Transpose => inplace_transpose(content, w, h, bpp),
        Orientation::Rotate90 => {
            inplace_transpose(content, w, h, bpp);
            inplace_flip_h(content, ow, oh, bpp); // transpose ∘ FlipH
        }
        Orientation::Rotate270 => {
            inplace_transpose(content, w, h, bpp);
            inplace_flip_v(content, ow, oh, bpp); // transpose ∘ FlipV
        }
        Orientation::Transverse => {
            inplace_transpose(content, w, h, bpp);
            inplace_reverse_elements(content, bpp); // transpose ∘ Rotate180
        }
        // Pre-checked in `apply_orientation_in_place`; unreachable here.
        _ => {}
    }

    rewrap(&mut bytes[..out_len], ow, oh, out_stride, desc, color)
}

/// Reverse the `bpp`-sized elements within each row, in place (`FlipH`).
fn inplace_flip_h(a: &mut [u8], w: u32, h: u32, bpp: usize) {
    let w = w as usize;
    let row_len = w * bpp;
    for y in 0..h as usize {
        let row = &mut a[y * row_len..y * row_len + row_len];
        let (mut lo, mut hi) = (0usize, w - 1);
        while lo < hi {
            let (al, ah) = (lo * bpp, hi * bpp);
            for k in 0..bpp {
                row.swap(al + k, ah + k);
            }
            lo += 1;
            hi -= 1;
        }
    }
}

/// Swap row `y` with row `h-1-y`, in place (`FlipV`). No temp row — the two rows
/// are disjoint, so `split_at_mut` + `swap_with_slice` exchanges them directly.
fn inplace_flip_v(a: &mut [u8], w: u32, h: u32, bpp: usize) {
    let row_len = w as usize * bpp;
    let h = h as usize;
    let (mut top, mut bot) = (0usize, h - 1);
    while top < bot {
        let split = bot * row_len;
        let (head, tail) = a.split_at_mut(split);
        head[top * row_len..top * row_len + row_len].swap_with_slice(&mut tail[..row_len]);
        top += 1;
        bot -= 1;
    }
}

/// Reverse the order of all `bpp`-sized elements in the buffer (`Rotate180` =
/// `FlipH ∘ FlipV`).
fn inplace_reverse_elements(a: &mut [u8], bpp: usize) {
    let n = a.len() / bpp;
    if n < 2 {
        return;
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while lo < hi {
        let (al, ah) = (lo * bpp, hi * bpp);
        for k in 0..bpp {
            a.swap(al + k, ah + k);
        }
        lo += 1;
        hi -= 1;
    }
}

/// In-place transpose of a tight `w`×`h` (row-major) grid of `bpp`-byte elements
/// into `h`×`w`, within the same buffer.
///
/// Square is the diagonal swap. Non-square follows the transpose permutation's
/// cycles (`Wikipedia: in-place matrix transposition`): element index `k = r*w+c`
/// maps to `c*h+r ≡ (k*h) mod (n-1)`; to fill position `cur` we gather from
/// `(cur*w) mod (n-1)` (the inverse, since `w*h ≡ 1`), walking each cycle once
/// with a one-element temp and an `n`-bit visited set. `0` and `n-1` are fixed.
fn inplace_transpose(a: &mut [u8], w: u32, h: u32, bpp: usize) {
    if w == h {
        let n = w as usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let (p, q) = ((i * n + j) * bpp, (j * n + i) * bpp);
                for k in 0..bpp {
                    a.swap(p + k, q + k);
                }
            }
        }
        return;
    }

    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    if n <= 1 {
        return;
    }
    let mn1 = n - 1;
    let mut moved = alloc::vec![false; n];
    moved[0] = true;
    moved[mn1] = true;
    let mut tmp = [0u8; MAX_INPLACE_BPP];
    let mut start = 1;
    while start < mn1 {
        if moved[start] {
            start += 1;
            continue;
        }
        tmp[..bpp].copy_from_slice(&a[start * bpp..start * bpp + bpp]);
        let mut cur = start;
        loop {
            moved[cur] = true;
            let prev = (cur * w) % mn1; // element that belongs at `cur`
            if prev == start {
                break;
            }
            a.copy_within(prev * bpp..prev * bpp + bpp, cur * bpp);
            cur = prev;
        }
        a[cur * bpp..cur * bpp + bpp].copy_from_slice(&tmp[..bpp]);
        start += 1;
    }
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

/// Dispatch the four axis-swapping orientations to the best available
/// transpose. With `fast-transpose` on x86-64 / aarch64, a dedicated AVX2 / NEON
/// kernel per pixel width (`pxn_x86` / `rgb3_x86` / `pxn_neon`); on other
/// `fast-transpose` arches, magetypes' 4×4 register transpose for 4-byte pixels
/// (`incant!`-dispatched across WASM SIMD128 / scalar) with the tiled gather for
/// the rest. Without the feature, the portable scalar tiers
/// ([`transpose_tiled`] / [`transpose16_deep`]). Any future `#[non_exhaustive]`
/// variant has no separable inverse map and falls to [`transpose_blocked`].
fn do_transpose(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
) {
    // Only the four known transposing orientations have a `tile_dest` /
    // separable-inverse mapping; a future `#[non_exhaustive]` variant falls
    // through to the scalar scatter, whose `forward_map` is defined for every
    // variant.
    if let Some(flips) = inverse_flips(orientation) {
        // AVX2 (x86-64-v3) register-transpose kernels, one per pixel width —
        // the kernel shapes zune / fast_transpose / the C++ Simd library win
        // with (transpose-shootout 2026-06-12, where our generic paths lost
        // 2-5×). The token summon fails only on pre-AVX2 x86, which then falls
        // through to the magetypes 4-byte kernel / scalar tiled gather below.
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
        // NEON kernels per pixel width. NEON is the aarch64 baseline, so the
        // token always summons; 16-byte pixels use the deep-scalar block move,
        // which beats hand-written NEON here (measured on Neoverse-N1 — see
        // `transpose16_deep`).
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
            // Non-x86 tiers (NEON / WASM SIMD128 / scalar) via the magetypes
            // 4×4 register transpose. Explicit tier list matching its
            // `#[magetypes(v3, neon, wasm128, scalar)]` attribute: a bare
            // `incant!` expands the full cascade and references a `_v4`
            // variant that was never generated, breaking `--features avx512`
            // builds (caught by the feature-powerset CI job).
            incant!(
                transpose4_simd(src, dst, orientation, w, h),
                [v3, neon, wasm128, scalar]
            );
            return;
        }
        // Monomorphised per pixel size so the inner copy is a fixed-size
        // load/store pair. The set covers every shipping descriptor width;
        // an unlisted width (none today) takes the generic fallback below.
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

/// The inverse-map structure shared by the four transposing orientations, as
/// `(flip_sx, flip_sy)`: destination pixel `(dx, dy)` reads source pixel
///
/// ```text
/// sx = if flip_sx { w-1-dy } else { dy }   // constant along a dst row
/// sy = if flip_sy { h-1-dx } else { dx }   // steps ±1 along a dst row
/// ```
///
/// Derived by inverting [`Orientation::forward_map`] — e.g. `Rotate90` maps
/// `(sx, sy) → (h-1-sy, sx)`, so `sx = dy`, `sy = h-1-dx`. `None` for the
/// non-transposing orientations and any future variant.
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

/// Cache-blocked transpose for the four axis-swapping orientations,
/// monomorphised per bytes-per-pixel (`BPP`).
///
/// Same loop-tiling idea as [`transpose_blocked`], but iterating *destination*
/// tiles with the orientation's separable inverse map (see [`inverse_flips`])
/// precomputed per row instead of calling `forward_map` per element: along one
/// destination row the source column is fixed and the source byte offset steps
/// by ±stride, so the inner loop is a strided gather (one bounds check) plus a
/// fixed-size `BPP`-byte copy, with destination writes sequential — the
/// store-friendly direction. This is what makes 3 bpp (and the other non-SIMD
/// widths) competitive; zenjpeg#150 measured the `forward_map`-per-element
/// path losing to a naive linear-write gather.
fn transpose_tiled<const BPP: usize>(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    (flip_sx, flip_sy): (bool, bool),
    w: u32,
    h: u32,
) {
    debug_assert_eq!(src.descriptor().bytes_per_pixel(), BPP);
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let sstep: isize = if flip_sy {
        -(sstride as isize)
    } else {
        sstride as isize
    };
    // Destination geometry (validated by `apply_orientation_into`).
    let (ow, oh) = (h, w);

    let mut ty = 0;
    while ty < oh {
        let ty_end = min(ty + TILE, oh);
        let mut tx = 0;
        while tx < ow {
            let tx_end = min(tx + TILE, ow);
            // Source row for the tile's first dst column (dx = tx); the
            // offset then steps by `sstep` per dst pixel.
            let sy0 = (if flip_sy { h - 1 - tx } else { tx }) as usize;
            for dy in ty..ty_end {
                let sx = (if flip_sx { w - 1 - dy } else { dy }) as usize;
                let mut soff = (sy0 * sstride + sx * BPP) as isize;
                let drow = &mut dst.row_mut(dy)[tx as usize * BPP..tx_end as usize * BPP];
                for dpx in drow.chunks_exact_mut(BPP) {
                    let s = soff as usize;
                    let px: [u8; BPP] = sbytes[s..s + BPP].try_into().unwrap();
                    dpx.copy_from_slice(&px);
                    soff += sstep;
                }
            }
            tx += TILE;
        }
        ty += TILE;
    }
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

// ── EXPERIMENTAL: staged fixed-micro-tile transpose (bench-gated) ────────────
//
// Hypothesis from the transpose-shootout (2026-06-12): a *fixed-size* staged
// micro-tile written in plain safe Rust auto-vectorizes into shuffle networks
// (the ejmahler `transpose` crate's 16×16 scalar block hit 22.5 GiB/s at 256²
// RGBA, ~3× our explicit 4×4 SSE kernel, with only baseline SSE2 codegen).
// Stage TH source rows of a tile into a fixed 2-D array (one bounds check per
// row), then emit TW destination rows whose elements come from fixed array
// indices — no per-element bounds checks, no `forward_map`, and a shape LLVM
// can SLP-vectorize. The orientation's reflection folds into (a) reversed
// fixed gather order inside the micro-tile (const-bool monomorphized) and
// (b) the tile's destination base coordinates.
//
// Bench-only entry (`__bench_apply_orientation_staged`) until the shootout
// proves which widths it wins; production dispatch is unchanged.

/// One staged micro-tile: gather `T=16` source rows × `T=16` pixels into a
/// fixed array, then write `T` destination rows from its columns.
///
/// Derived from the separable inverse maps (`inverse_flips`): destination row
/// `dy0+c` reads the tile's source column `cc = c` (or `T-1-c` when `FLIP_C`,
/// i.e. `flip_sx`), and its pixel `k` reads stage row `r = k` (or `T-1-k`
/// when `FLIP_R`, i.e. `flip_sy`). Both indices are compile-time-shaped, so
/// the inner loops are bounds-check-free and SLP-vectorizable.
macro_rules! staged_micro {
    ($name:ident, $bpp:literal) => {
        #[cfg(all(feature = "__bench_orient", feature = "fast-transpose"))]
        #[inline]
        fn $name<const FLIP_R: bool, const FLIP_C: bool>(
            sbytes: &[u8],
            sstride: usize,
            sx0: usize, // tile's first source column (pixels)
            sy0: usize, // tile's first source row
            dst: &mut PixelSliceMut<'_>,
            dx0: usize, // tile's first dest column (pixels)
            dy0: usize, // tile's first dest row
        ) {
            const T: usize = 16;
            let mut stage = [[0u8; T * $bpp]; T];
            for (r, row) in stage.iter_mut().enumerate() {
                let off = (sy0 + r) * sstride + sx0 * $bpp;
                row.copy_from_slice(&sbytes[off..off + T * $bpp]);
            }
            for c in 0..T {
                let cc = if FLIP_C { T - 1 - c } else { c };
                let drow = &mut dst.row_mut((dy0 + c) as u32)[dx0 * $bpp..(dx0 + T) * $bpp];
                for (k, dpx) in drow.chunks_exact_mut($bpp).enumerate() {
                    let r = if FLIP_R { T - 1 - k } else { k };
                    dpx.copy_from_slice(&stage[r][cc * $bpp..cc * $bpp + $bpp]);
                }
            }
        }
    };
}

staged_micro!(staged_micro_1, 1);
staged_micro!(staged_micro_2, 2);
staged_micro!(staged_micro_3, 3);
staged_micro!(staged_micro_4, 4);

/// Staged-tile transpose for the four axis-swapping orientations, bpp ∈ 1..=4.
/// Full 16×16 micro-tiles go through the staged kernel; edge remainders take
/// the per-pixel `forward_map` scatter (identical to `transpose_edges`).
#[cfg(all(feature = "__bench_orient", feature = "fast-transpose"))]
fn transpose_staged(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
    bpp: usize,
) {
    const T: u32 = 16;
    let Some((flip_sx, flip_sy)) = inverse_flips(orientation) else {
        return transpose_blocked(src, dst, orientation, w, h, bpp);
    };
    let sbytes = src.as_strided_bytes();
    let sstride = src.stride();
    let full_w = w & !(T - 1);
    let full_h = h & !(T - 1);

    let mut sy = 0;
    while sy < full_h {
        let mut sx = 0;
        while sx < full_w {
            // Destination tile base from the orientation's affine map:
            // dst col base ← source rows [sy, sy+T), dst row base ← source
            // cols [sx, sx+T); reflections pick the tile's far corner.
            let dx0 = if flip_sy { h - T - sy } else { sy } as usize;
            let dy0 = if flip_sx { w - T - sx } else { sx } as usize;
            let (sx_eff, sy_eff) = (sx as usize, sy as usize);
            // FLIP_R = flip_sy (reverses the stage-row gather inside a dst
            // row), FLIP_C = flip_sx (flips which source column feeds dst
            // row dy0+c). Monomorphised so the micro-tile stays branch-free.
            macro_rules! call {
                ($f:ident) => {
                    match (flip_sy, flip_sx) {
                        (false, false) => {
                            $f::<false, false>(sbytes, sstride, sx_eff, sy_eff, dst, dx0, dy0)
                        }
                        (true, false) => {
                            $f::<true, false>(sbytes, sstride, sx_eff, sy_eff, dst, dx0, dy0)
                        }
                        (false, true) => {
                            $f::<false, true>(sbytes, sstride, sx_eff, sy_eff, dst, dx0, dy0)
                        }
                        (true, true) => {
                            $f::<true, true>(sbytes, sstride, sx_eff, sy_eff, dst, dx0, dy0)
                        }
                    }
                };
            }
            match bpp {
                1 => call!(staged_micro_1),
                2 => call!(staged_micro_2),
                3 => call!(staged_micro_3),
                4 => call!(staged_micro_4),
                _ => unreachable!("staged path is dispatched for bpp 1..=4 only"),
            }
            sx += T;
        }
        sy += T;
    }
    transpose_edges(src, dst, orientation, w, h, bpp, full_w, full_h);
}

/// Bench-only handle for the staged experimental path (transposing
/// orientations, bpp 1..=4; falls back to the blocked scatter otherwise).
#[cfg(all(feature = "__bench_orient", feature = "fast-transpose"))]
#[doc(hidden)]
pub fn __bench_apply_orientation_staged(
    src: PixelSlice<'_>,
    orientation: Orientation,
    mut dst: PixelSliceMut<'_>,
) -> Result<(), ConvertError> {
    let w = src.width();
    let h = src.rows();
    let bpp = src.descriptor().bytes_per_pixel();
    let (ow, oh) = orientation.output_dimensions(w, h);
    if dst.width() != ow || dst.rows() != oh || dst.descriptor().bytes_per_pixel() != bpp {
        return Err(ConvertError::BufferSize {
            expected: ow as usize * oh as usize * bpp,
            actual: dst.width() as usize * dst.rows() as usize * dst.descriptor().bytes_per_pixel(),
        });
    }
    if w == 0 || h == 0 || bpp == 0 {
        return Ok(());
    }
    if (1..=4).contains(&bpp) {
        transpose_staged(&src, &mut dst, orientation, w, h, bpp);
    } else {
        transpose_blocked(&src, &mut dst, orientation, w, h, bpp);
    }
    Ok(())
}

/// Scalar scatter for the edge strips a 4×4-tiled SIMD pass leaves uncovered:
/// the right strip (`cols [full_w, w)`) and the bottom strip (`rows [full_h,
/// h)`, which also covers the bottom-right corner). No overlap between strips.
#[allow(clippy::too_many_arguments)] // edge-strip helper; mirrors the scatter-loop signature
#[cfg(feature = "fast-transpose")]
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

// ── SIMD 4×4 register transpose (cross-arch, 4-byte pixels) ──────────────────

/// Destination of transposed-tile row `r` (transposed row index 0..4) for a
/// source 4×4 tile at `(bx, by)`, as `(dst_row, dst_col_start, reverse_lanes)`.
/// Derived from `Orientation::forward_map`: a bare `Transpose` writes row `r` to
/// `dst[bx+r][by..]`; the rotations/anti-diagonal add a row/col reflection.
/// `by`/`bx` are multiples of 4 with `by+4 ≤ h`, `bx+4 ≤ w`, so the subtractions
/// never underflow.
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

/// SIMD path: transpose full 4×4 tiles via `f32x4::transpose_4x4` (the classic
/// `_MM_TRANSPOSE4_PS`-shaped shuffle cascade), scalar for the edges. The
/// `#[magetypes]` attribute generates one variant per SIMD tier from this single
/// body; `incant!` in [`do_transpose`] picks the best at runtime.
///
/// Each 4-byte pixel rides as one f32 lane. The transpose only *shuffles whole
/// 32-bit lanes* (no float arithmetic), so the reinterpret is bit-exact for any
/// 4-byte pixel format, including bit patterns that happen to be NaN.
#[cfg(feature = "fast-transpose")]
#[magetypes(v3, neon, wasm128, scalar)]
fn transpose4_simd(
    token: Token,
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
    orientation: Orientation,
    w: u32,
    h: u32,
) {
    #[allow(non_camel_case_types)]
    type f32x4 = GenericF32x4<Token>;

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

// ── SIMD 3-byte (RGB8) transpose: expand 3→4, transpose u32 lanes, compress ──
//
// The only production-proven SIMD shape for 24-bit transposes (ermig1979/Simd;
// no Rust crate has one): load four rows of four RGB pixels (16 B each, 12
// payload + 4 slop), `pshufb`-expand 3→4 so each pixel rides one u32 lane,
// transpose the 4×4 u32 block with the unpack cascade, `pshufb`-compress 4→3,
// and store each 12-byte group as 8+4 bytes (no store slop → no clobber
// hazard at row ends, and both store intrinsics have safe reference-taking
// wrappers). The 16-byte *loads* are the only slop: safe for every tile band
// except one ending on the image's final row, where the band's tile range is
// narrowed so `offset+16` stays in bounds (the scalar edge pass covers the
// rest). Value intrinsics are safe inside `#[arcane]` (Rust 1.87+); memory
// ops come from `import_intrinsics`' safe wrappers — the crate keeps
// `#![forbid(unsafe_code)]`.
#[cfg(all(feature = "fast-transpose", target_arch = "x86_64"))]
mod rgb3_x86;

// ── SIMD 1/2/4-byte transposes: exact-width register cascades ────────────────
//
// Same construction as `rgb3_x86` (value intrinsics inside `#[arcane]`, safe
// reference-taking memory wrappers, flat src/dst byte addressing, scalar
// `forward_map` edges) but with NO load/store slop anywhere: tile rows are
// exactly one register wide (8 px × 1 B = 8 B, 8 px × 2 B = 16 B, 4 px × 4 B =
// 16 B, 8 px × 4 B = 32 B), so there is no last-row guard at all. Shapes are
// the classic punpck cascades (AP-528 → libyuv lineage); the shootout showed
// our previous paths losing 3-5× to exactly these kernel shapes in zune /
// fast_transpose / the C++ Simd library.
/// 16-byte pixels, all architectures: deep scalar tiles. A transpose at
/// this width is pure 16-byte block movement, and fixed-size 16-byte copies
/// autovectorize to full-width register moves on every supported target
/// (NEON is the aarch64 baseline, SSE2 the x86-64 baseline) — measured on
/// Neoverse-N1, this shape beats the hand-written NEON kernel (run depth
/// amortizes per-row bookkeeping 4×; N1 microcodes ld1/st1 multi-register
/// structure ops, so wide hand-SIMD has no edge to offer here).
fn transpose16_deep(
    src: &PixelSlice<'_>,
    dst: &mut PixelSliceMut<'_>,
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
            // One bounds check per source row segment and per destination
            // row run; the inner fixed-size copies need none.
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
                for (k, chunk) in drow.chunks_exact_mut(16).enumerate() {
                    let r = if flip_sy { T - 1 - k } else { k };
                    chunk.copy_from_slice(&srows[r][c * 16..c * 16 + 16]);
                }
            }
        }
    }
    // Edges (right columns, bottom rows).
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
mod pxn_x86;

// ── NEON transposes (aarch64): vzip cascades + tbl expand/compress ──────────
//
// Same construction discipline as `pxn_x86`: value intrinsics inside
// `#[arcane]`, safe reference-taking loads/stores, flat strided bytes,
// scalar `forward_map` edges, slop-free stores (the 3/6/12-byte kernels'
// reads carry small guarded slop like their x86 siblings). Kernel networks
// for 1/2/4 bpp follow ermig1979/Simd's NEON tier (vzipq cascades); the
// 24-bit kernel reuses our expand→zip→compress shape instead of their
// vtbl variant, whose 16-byte stores overhang the 12-byte destination run.
#[cfg(all(feature = "fast-transpose", target_arch = "aarch64"))]
mod pxn_neon;

#[cfg(test)]
mod tests;
