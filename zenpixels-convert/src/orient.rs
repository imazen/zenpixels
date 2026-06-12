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
//! For **4-byte pixels** the per-tile transpose is SIMD on every supported
//! arch: full 4×4 tiles go through magetypes' `f32x4::transpose_4x4` (the
//! classic `_MM_TRANSPOSE4_PS`-shaped shuffle cascade — SSE on x86, NEON on
//! aarch64, SIMD128 on wasm), generated once via `#[magetypes(v3, neon,
//! wasm128, scalar)]` and dispatched by `incant!` at runtime (scalar tier when
//! no SIMD is available). Each pixel rides as one f32 lane — the kernel only
//! shuffles whole 32-bit lanes (no float math), so reinterpreting the bytes as
//! f32 is bit-exact for any 4-byte format, NaN bit patterns included.
//!
//! Every **other pixel size** (1/2/3/6/8/12/16 bytes — gray, gray+alpha, RGB8,
//! and the 16-bit / f32 widths) goes through [`transpose_tiled`], a
//! monomorphised cache-blocked gather: the four transposing orientations all
//! have *separable* inverse maps, so along one destination row the source
//! column is fixed and the source byte offset steps by ±stride. That replaces
//! the per-element `forward_map` + `row_mut` + variable-length copy of the
//! generic path with one predictable bounds check and a fixed-size `BPP`-byte
//! copy per pixel, writing the destination sequentially (zenjpeg#150 measured
//! the generic path losing to a naive linear-write gather at 3 bpp; this path
//! beats both). The generic `forward_map` scatter ([`transpose_blocked`])
//! remains as the parity oracle for both fast paths
//! (`simd_transpose_matches_scalar_reference_rgba8`,
//! `tiled_transpose_matches_blocked_reference_across_bpp`) and as the
//! correct fallback for any future `#[non_exhaustive]` orientation variant.
//! (1- and 2-byte SIMD transpose — the 16×16 `punpck` cascade — remains a
//! possible follow-up.)

use core::cmp::min;

use zenpixels::{InPlacePixels, Orientation, PixelBuffer, PixelSlice, PixelSliceMut};

use crate::error::ConvertError;

// Cross-arch SIMD: the `#[magetypes(...)]` codegen attribute + `incant!`
// runtime dispatch from the archmage prelude, and the token-parameterized
// generic `f32x4` from magetypes — whose `transpose_4x4` lowers to SSE
// `_MM_TRANSPOSE4_PS` on x86, the NEON `zip`/`trn` cascade on aarch64, and the
// `i32x4.shuffle` cascade on wasm128.
use archmage::prelude::*;
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

/// Dispatch the axis-swapping orientations: the SIMD 4×4 register transpose for
/// 4-byte pixels (the common decoder output), the monomorphised tiled gather
/// for every other shipping pixel size, the generic `forward_map` scatter for
/// anything else. `incant!` picks the best tier per target (AVX2 / NEON / WASM
/// SIMD128 / scalar).
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
        // Hand-tuned x86 tiers for the four 1-byte-channel pixel sizes
        // (transpose-shootout 2026-06-12: these kernel shapes are what zune /
        // fast_transpose / the C++ Simd library win with; our generic paths
        // lost 2-5×). Pre-SSSE3 x86 and every other arch fall through to the
        // magetypes 4-byte kernel / tiled gather below.
        #[cfg(target_arch = "x86_64")]
        match bpp {
            1 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose1_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            2 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose2_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            3 => {
                if let Some(token) = X64V3Token::summon() {
                    rgb3_x86::transpose3_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            4 => {
                // X64V3 = x86-64-v3 = AVX2 baseline; pre-v3 x86 falls to the
                // magetypes scalar tier below, as before.
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose4_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            6 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose6_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            8 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose8_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            12 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose12_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            16 => {
                if let Some(token) = X64V3Token::summon() {
                    pxn_x86::transpose16_v3(token, src, dst, orientation, w, h);
                    return;
                }
            }
            _ => {}
        }
        #[cfg(target_arch = "aarch64")]
        match bpp {
            1 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose1_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            2 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose2_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            3 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose3_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            4 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose4_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            6 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose6_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            8 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose8_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            12 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose12_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            16 => {
                if let Some(token) = NeonToken::summon() {
                    pxn_neon::transpose16_neon(token, src, dst, orientation, w, h);
                    return;
                }
            }
            _ => {}
        }
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
            6 => return transpose_tiled::<6>(src, dst, flips, w, h),
            8 => return transpose_tiled::<8>(src, dst, flips, w, h),
            12 => return transpose_tiled::<12>(src, dst, flips, w, h),
            16 => return transpose_tiled::<16>(src, dst, flips, w, h),
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
        #[cfg(feature = "__bench_orient")]
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
#[cfg(feature = "__bench_orient")]
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
#[cfg(feature = "__bench_orient")]
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
#[cfg(target_arch = "x86_64")]
mod rgb3_x86 {
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
                            let a: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
        // Guard-trimmed columns of the last band (only when that band touches
        // the image's final row), then right-edge columns, then bottom rows.
        if guard_w < full_w && full_h == h && h >= 8 {
            super::pxn_x86::scalar_rect(
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
                h - 8,
                h,
            );
        }
        super::pxn_x86::scalar_rect(
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
        super::pxn_x86::scalar_rect(
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
}

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
#[cfg(target_arch = "x86_64")]
mod pxn_x86 {
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
                            let lo: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
                            let a: &[u8; 32] = sbytes
                                [base + $i * sstride..base + $i * sstride + 32]
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
                            let a: &[u8; 32] = sbytes
                                [base + $i * sstride..base + $i * sstride + 32]
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
                            let a: &[u8; 32] = sbytes
                                [sbase + $i * sstride..sbase + $i * sstride + 32]
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
                            let lo: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
        if guard_w < full_w && full_h == h && h >= 4 {
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
                            let lo: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
                        let out0: &mut [u8; 32] =
                            (&mut dbytes[doff..doff + 32]).try_into().unwrap();
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
}

// ── NEON transposes (aarch64): vzip cascades + tbl expand/compress ──────────
//
// Same construction discipline as `pxn_x86`: value intrinsics inside
// `#[arcane]`, safe reference-taking loads/stores, flat strided bytes,
// scalar `forward_map` edges, slop-free stores (the 3/6/12-byte kernels'
// reads carry small guarded slop like their x86 siblings). Kernel networks
// for 1/2/4 bpp follow ermig1979/Simd's NEON tier (vzipq cascades); the
// 24-bit kernel reuses our expand→zip→compress shape instead of their
// vtbl variant, whose 16-byte stores overhang the 12-byte destination run.
#[cfg(target_arch = "aarch64")]
mod pxn_neon {
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
                            let a: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
                            let a: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
                            let a: &[u8; 16] = sbytes
                                [base + $i * sstride..base + $i * sstride + 16]
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
                            let a: &[u8; 32] = sbytes
                                [base + $i * sstride..base + $i * sstride + 32]
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
                            let out: &mut [u8; 32] =
                                (&mut dbytes[doff..doff + 32]).try_into().unwrap();
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

    /// 16-byte pixels: pure q-register block movement (gather 4 rows per
    /// destination run, store sequentially).
    #[arcane(import_intrinsics)]
    pub(super) fn transpose16_neon(
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

        let full_h = h & !3;
        const MACRO: u32 = 64;
        let nblocks = w.div_ceil(MACRO);
        for bi in 0..nblocks {
            let bx = if flip_sx { nblocks - 1 - bi } else { bi } * MACRO;
            let bx_end = (bx + MACRO).min(w);
            let ncols = bx_end - bx;
            let nbands = full_h / 4;
            for bandi in 0..nbands {
                let sy = bandi * 4;
                let dx = (if flip_sy { h - 4 - sy } else { sy }) as usize;
                // Quad columns: 64-byte x4 tuple loads (4 px per row) and
                // one x4 tuple store per destination run. Remainder columns
                // (≤3 per stripe) take the single-pixel path; iteration
                // order of the remainder is cache-irrelevant.
                // Four zipped chunks_exact(16) iterators stream the band's
                // source rows with their bounds checks hoisted to slice
                // construction; one 64-byte destination-run check per step.
                // Plain q-register loads/stores only — ld1/st1 x2/x4
                // structure ops are microcoded multi-cycle on Neoverse-N1
                // and measured slower than singles here.
                {
                    let row_off = sy as usize * sstride + bx as usize * 16;
                    let seg = ncols as usize * 16;
                    let (s0, rest) = sbytes[row_off..].split_at(seg);
                    let s1 = &rest[sstride - seg..sstride];
                    let s2 = &sbytes[row_off + 2 * sstride..row_off + 2 * sstride + seg];
                    let s3 = &sbytes[row_off + 3 * sstride..row_off + 3 * sstride + seg];
                    let it = s0
                        .chunks_exact(16)
                        .zip(s1.chunks_exact(16))
                        .zip(s2.chunks_exact(16).zip(s3.chunks_exact(16)));
                    for (ci, ((c0, c1), (c2, c3))) in it.enumerate() {
                        let sx = bx + ci as u32;
                        let p0 = vld1q_u8(c0.try_into().unwrap());
                        let p1 = vld1q_u8(c1.try_into().unwrap());
                        let p2 = vld1q_u8(c2.try_into().unwrap());
                        let p3 = vld1q_u8(c3.try_into().unwrap());
                        let (p0, p1, p2, p3) = if flip_sy {
                            (p3, p2, p1, p0)
                        } else {
                            (p0, p1, p2, p3)
                        };
                        let dy = if flip_sx { w - 1 - sx } else { sx };
                        let dbase = dy as usize * dstride + dx * 16;
                        let out: &mut [u8; 64] =
                            (&mut dbytes[dbase..dbase + 64]).try_into().unwrap();
                        let (o01, o23) = out.split_at_mut(32);
                        let (o0, o1) = o01.split_at_mut(16);
                        let (o2, o3) = o23.split_at_mut(16);
                        vst1q_u8(o0.try_into().unwrap(), p0);
                        vst1q_u8(o1.try_into().unwrap(), p1);
                        vst1q_u8(o2.try_into().unwrap(), p2);
                        vst1q_u8(o3.try_into().unwrap(), p3);
                    }
                }
                let nquads = 0u32;
                let _ = nquads;
                for ci in ncols..ncols {
                    let sx = bx + ci;
                    let dy = if flip_sx { w - 1 - sx } else { sx };
                    let dbase = dy as usize * dstride + dx * 16;
                    let sbase = sy as usize * sstride + sx as usize * 16;
                    for k in 0..4usize {
                        let r = if flip_sy { 3 - k } else { k };
                        let a: &[u8; 16] = sbytes[sbase + r * sstride..sbase + r * sstride + 16]
                            .try_into()
                            .unwrap();
                        let out: &mut [u8; 16] = (&mut dbytes[dbase + k * 16..dbase + k * 16 + 16])
                            .try_into()
                            .unwrap();
                        vst1q_u8(out, vld1q_u8(a));
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

    const EXPAND6: [u8; 16] = [0, 1, 2, 3, 4, 5, 255, 255, 6, 7, 8, 9, 10, 11, 255, 255];
    const COMPRESS6_FWD: [u8; 16] = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 255, 255, 255, 255];
    const COMPRESS6_REV: [u8; 16] = [8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 255, 255, 255, 255];

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
                            let out: &mut [u8; 16] = (&mut dbytes
                                [doff + k * 12..doff + k * 12 + 16])
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
        // as a tight one (padding bytes must be ignored). RGBA8 exercises the
        // SIMD path, RGB8/GRAY8 the tiled gather (whose strided-offset math is
        // exactly what this guards), and the dims span full + partial tiles.
        for &desc in &[
            PixelDescriptor::RGBA8,
            PixelDescriptor::RGB8,
            PixelDescriptor::GRAY8,
        ] {
            let bpp = desc.bytes_per_pixel();
            for &(w, h) in &[(5u32, 4u32), (37, 35)] {
                let tight_stride = w as usize * bpp;
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
                        assert_eq!(
                            a.as_slice().row(y),
                            b.as_slice().row(y),
                            "{o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
                }
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

    /// Parity gate for the monomorphised tiled gather: `apply_orientation`
    /// (which routes non-4-byte widths through `transpose_tiled`) must match
    /// the generic `forward_map` scatter oracle for every shipping pixel size
    /// the dispatch covers, across the four transposing orientations and a
    /// dimension spread that exercises full tiles, partial tiles, and the
    /// degenerate strips (TILE is 32, so 33/40/67 cross tile boundaries).
    #[test]
    fn tiled_transpose_matches_blocked_reference_across_bpp() {
        let descs = [
            PixelDescriptor::GRAY8,   // 1
            PixelDescriptor::GRAYA8,  // 2
            PixelDescriptor::RGB8,    // 3
            PixelDescriptor::RGB16,   // 6
            PixelDescriptor::RGBA16,  // 8
            PixelDescriptor::RGBF32,  // 12
            PixelDescriptor::RGBAF32, // 16
        ];
        let dims = [
            (8u32, 8u32),
            (32, 32),
            (64, 48),
            (17, 13),
            (33, 31),
            (40, 33),
            (67, 43),
            (64, 1),
            (1, 64),
            (1, 1),
        ];
        for &desc in &descs {
            let bpp = desc.bytes_per_pixel();
            for &(w, h) in &dims {
                let data = fill(w as usize * h as usize * bpp);
                for &o in &[
                    Orientation::Transpose,
                    Orientation::Rotate90,
                    Orientation::Rotate270,
                    Orientation::Transverse,
                ] {
                    // Path under test (transpose_tiled for these widths).
                    let got = apply_orientation(slice(&data, w, h, desc), o);
                    // Generic forward_map scatter as the independent oracle.
                    let (ow, oh) = o.output_dimensions(w, h);
                    let mut reference = PixelBuffer::new(ow, oh, desc);
                    {
                        let src = slice(&data, w, h, desc);
                        let mut d = reference.as_slice_mut();
                        transpose_blocked(&src, &mut d, o, w, h, bpp);
                    }
                    for y in 0..oh {
                        assert_eq!(
                            got.as_slice().row(y),
                            reference.as_slice().row(y),
                            "{o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
                }
            }
        }
    }

    /// The 3-byte SIMD tier displaced `transpose_tiled::<3>` from the
    /// `apply_orientation` route on x86_64, so gate the fallback directly:
    /// it must still match the blocked oracle (it remains the non-x86 path).
    #[test]
    fn tiled3_fallback_matches_blocked_reference() {
        let desc = PixelDescriptor::RGB8;
        for &(w, h) in &[(8u32, 8u32), (17, 13), (33, 31), (67, 43), (1, 1), (5, 64)] {
            let data = fill(w as usize * h as usize * 3);
            for &o in &[
                Orientation::Transpose,
                Orientation::Rotate90,
                Orientation::Rotate270,
                Orientation::Transverse,
            ] {
                let flips = inverse_flips(o).unwrap();
                let (ow, oh) = o.output_dimensions(w, h);
                let mut got = PixelBuffer::new(ow, oh, desc);
                let mut want = PixelBuffer::new(ow, oh, desc);
                {
                    let src = slice(&data, w, h, desc);
                    let mut d = got.as_slice_mut();
                    transpose_tiled::<3>(&src, &mut d, flips, w, h);
                }
                {
                    let src = slice(&data, w, h, desc);
                    let mut d = want.as_slice_mut();
                    transpose_blocked(&src, &mut d, o, w, h, 3);
                }
                for y in 0..oh {
                    assert_eq!(
                        got.as_slice().row(y),
                        want.as_slice().row(y),
                        "tiled3 {o:?} {w}x{h} row {y}"
                    );
                }
            }
        }
    }

    /// Parity gate for the experimental staged micro-tile path: must match
    /// the production `apply_orientation` byte-for-byte across bpp 1..=4,
    /// the four transposing orientations, and dims covering full 16×16
    /// micro-tiles, edge strips, and degenerate sizes.
    #[cfg(feature = "__bench_orient")]
    #[test]
    fn staged_matches_production_across_bpp_and_orientations() {
        let descs = [
            PixelDescriptor::GRAY8,
            PixelDescriptor::GRAYA8,
            PixelDescriptor::RGB8,
            PixelDescriptor::RGBA8,
        ];
        let dims = [
            (16u32, 16u32),
            (32, 32),
            (64, 48),
            (17, 13),
            (33, 31),
            (40, 33),
            (67, 43),
            (16, 1),
            (1, 16),
            (1, 1),
            (15, 15), // edge-only (no full micro-tile)
        ];
        for &desc in &descs {
            let bpp = desc.bytes_per_pixel();
            for &(w, h) in &dims {
                let data = fill(w as usize * h as usize * bpp);
                for &o in &[
                    Orientation::Transpose,
                    Orientation::Rotate90,
                    Orientation::Rotate270,
                    Orientation::Transverse,
                ] {
                    let want = apply_orientation(slice(&data, w, h, desc), o);
                    let (ow, oh) = o.output_dimensions(w, h);
                    let mut got = PixelBuffer::new(ow, oh, desc);
                    super::__bench_apply_orientation_staged(
                        slice(&data, w, h, desc),
                        o,
                        got.as_slice_mut(),
                    )
                    .expect("staged accepts matching dst");
                    for y in 0..oh {
                        assert_eq!(
                            got.as_slice().row(y),
                            want.as_slice().row(y),
                            "staged {o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
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

    /// In-place must produce byte-identical output to the proven out-of-place
    /// `apply_orientation`, across square + non-square, tight + (via
    /// `PixelBuffer::new`'s aligned stride) padded buffers, every orientation
    /// and a spread of element sizes. This is the correctness gate for the
    /// diagonal-swap (square), cycle-following (non-square), and in-place flips.
    #[test]
    fn in_place_matches_out_of_place() {
        let descs = [
            PixelDescriptor::GRAY8,
            PixelDescriptor::GRAYA8,
            PixelDescriptor::RGB8,
            PixelDescriptor::RGBA8,
            PixelDescriptor::RGBAF32,
        ];
        let dims = [
            (1u32, 1u32),
            (2, 2),
            (4, 4),
            (8, 8),
            (32, 32),
            (3, 5),
            (5, 3),
            (17, 13),
            (13, 17),
            (16, 9),
            (9, 16),
            (7, 1),
            (1, 7),
        ];
        for &desc in &descs {
            let bpp = desc.bytes_per_pixel();
            for &(w, h) in &dims {
                let data = fill(w as usize * h as usize * bpp);
                for &o in &Orientation::ALL {
                    let want = apply_orientation(slice(&data, w, h, desc), o);
                    // Load `data` into a fresh buffer (its stride may be padded),
                    // then bake in place.
                    let mut buf = PixelBuffer::new(w, h, desc);
                    {
                        let mut s = buf.as_slice_mut();
                        for y in 0..h {
                            s.row_mut(y).copy_from_slice(
                                &data[y as usize * w as usize * bpp..][..w as usize * bpp],
                            );
                        }
                    }
                    apply_orientation_in_place(&mut buf, o)
                        .expect("in_place should accept bpp ≤ 16");
                    assert_eq!(
                        (buf.width(), buf.height()),
                        (want.width(), want.height()),
                        "{o:?} {desc:?} {w}x{h} dims"
                    );
                    for y in 0..buf.height() {
                        assert_eq!(
                            buf.as_slice().row(y),
                            want.as_slice().row(y),
                            "{o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
                }
            }
        }
    }
}
