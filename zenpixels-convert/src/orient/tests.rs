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
#[cfg(all(feature = "__bench_orient", feature = "fast-transpose"))]
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

/// Fully independent oracle: place every source pixel at its
/// `Orientation::forward_map` destination with a naive per-pixel scatter.
/// Shares NO code with the production path (no tiling, no `scatter_pixel`,
/// no `transpose_blocked`) — only `forward_map`, which is the orientation
/// spec and is itself bijection-tested in zenpixels. Returns
/// `(out_w, out_h, tight_bytes)`.
fn naive_oracle(
    data: &[u8],
    w: u32,
    h: u32,
    desc: PixelDescriptor,
    o: Orientation,
) -> (u32, u32, Vec<u8>) {
    let bpp = desc.bytes_per_pixel();
    let (ow, oh) = o.output_dimensions(w, h);
    let mut out = alloc::vec![0u8; ow as usize * oh as usize * bpp];
    for sy in 0..h {
        for sx in 0..w {
            let (dx, dy) = o.forward_map(sx, sy, w, h);
            let s = (sy as usize * w as usize + sx as usize) * bpp;
            let d = (dy as usize * ow as usize + dx as usize) * bpp;
            out[d..d + bpp].copy_from_slice(&data[s..s + bpp]);
        }
    }
    (ow, oh, out)
}

/// Assert the production dispatch matches the independent oracle for one
/// (desc, dims, orientation), row by row (output stride may be padded).
fn assert_matches_oracle(desc: PixelDescriptor, w: u32, h: u32, o: Orientation) {
    let bpp = desc.bytes_per_pixel();
    let data = fill(w as usize * h as usize * bpp);
    let (ow, oh, want) = naive_oracle(&data, w, h, desc, o);
    let got = apply_orientation(slice(&data, w, h, desc), o);
    assert_eq!(
        (got.width(), got.height()),
        (ow, oh),
        "dims {o:?} {desc:?} {w}x{h}"
    );
    let gs = got.as_slice();
    for y in 0..oh {
        let exp = &want[y as usize * ow as usize * bpp..][..ow as usize * bpp];
        assert_eq!(gs.row(y), exp, "{o:?} {desc:?} {w}x{h} row {y}");
    }
}

/// Every shipping descriptor (1/2/3/4/6/8/12/16 bpp → a distinct kernel),
/// every dimension 1..=33 in both axes, every orientation, vs the
/// independent oracle. 33 spans: sub-tile (all-scalar), the exact tile
/// size for every kernel (2/4/8/16-wide and -deep), tile+1/+2/+3 edge
/// strips, two full tiles (32), two-tiles+1 (33), and — because the
/// range includes w==1 and h==1 — every 1×N and N×1 degenerate strip.
/// This is the corruption gate: it provably exercises full blocks, sub
/// blocks, partial-tile tails, and the per-bpp last-row slop guards
/// (every height that is an exact multiple of a kernel's tile depth is in
/// range, paired with every width across each kernel's `guard_w` edge).
#[test]
fn exhaustive_dense_dims_all_orientations_vs_oracle() {
    let descs = [
        PixelDescriptor::GRAY8,
        PixelDescriptor::GRAYA8,
        PixelDescriptor::RGB8,
        PixelDescriptor::RGBA8,
        PixelDescriptor::RGB16,
        PixelDescriptor::RGBA16,
        PixelDescriptor::RGBF32,
        PixelDescriptor::RGBAF32,
    ];
    for &desc in &descs {
        for w in 1u32..=33 {
            for h in 1u32..=33 {
                for &o in &Orientation::ALL {
                    assert_matches_oracle(desc, w, h, o);
                }
            }
        }
    }
}

/// Multi-stripe and stripe-remainder coverage: widths/heights that cross
/// the `MACRO = 64` column-stripe boundary (two full stripes at 128, a
/// stripe + small remainder at 65..72, etc.), so the per-stripe blocking
/// and the seam between stripes are exercised — the dense grid above
/// tops out below one stripe. All eight orientations; the giant 16-byte
/// cells are skipped to keep the gate fast (16 bpp is a plain block move,
/// fully covered by the dense grid's tail logic).
#[test]
fn multistripe_boundaries_vs_oracle() {
    let descs = [
        PixelDescriptor::GRAY8,
        PixelDescriptor::GRAYA8,
        PixelDescriptor::RGB8,
        PixelDescriptor::RGBA8,
        PixelDescriptor::RGB16,
        PixelDescriptor::RGBA16,
        PixelDescriptor::RGBF32,
    ];
    let span = [48u32, 63, 64, 65, 66, 68, 72, 96, 127, 128, 129, 130];
    let other = [1u32, 4, 7, 8, 15, 16, 17, 33, 64, 65, 128];
    for &desc in &descs {
        for &a in &span {
            for &b in &other {
                for &o in &Orientation::ALL {
                    assert_matches_oracle(desc, a, b, o);
                    assert_matches_oracle(desc, b, a, o);
                }
            }
        }
    }
}

/// The portable scalar fallbacks (`transpose_blocked`, used as the
/// allocating path's oracle on no-SIMD targets, and the monomorphised
/// `transpose_tiled::<BPP>` reached on non-x86/non-aarch64 builds) must
/// also match the independent oracle across the dense small grid — those
/// paths run on wasm and other arches where the SIMD tiers don't, so they
/// can't ride on the x86/ARM gate above.
#[test]
fn exhaustive_scalar_fallbacks_vs_oracle() {
    let descs = [
        (PixelDescriptor::GRAY8, 1usize),
        (PixelDescriptor::GRAYA8, 2),
        (PixelDescriptor::RGB8, 3),
        (PixelDescriptor::RGBA8, 4),
        (PixelDescriptor::RGB16, 6),
        (PixelDescriptor::RGBA16, 8),
        (PixelDescriptor::RGBF32, 12),
        (PixelDescriptor::RGBAF32, 16),
    ];
    let trans = [
        Orientation::Transpose,
        Orientation::Rotate90,
        Orientation::Rotate270,
        Orientation::Transverse,
    ];
    for &(desc, bpp) in &descs {
        for w in 1u32..=33 {
            for h in 1u32..=33 {
                let data = fill(w as usize * h as usize * bpp);
                for &o in &trans {
                    let (ow, oh, want) = naive_oracle(&data, w, h, desc, o);
                    // transpose_blocked (generic forward_map scatter).
                    let mut b = PixelBuffer::new(ow, oh, desc);
                    {
                        let src = slice(&data, w, h, desc);
                        transpose_blocked(&src, &mut b.as_slice_mut(), o, w, h, bpp);
                    }
                    // transpose_tiled::<BPP> (the monomorphised gather).
                    let flips = inverse_flips(o).unwrap();
                    let mut t = PixelBuffer::new(ow, oh, desc);
                    macro_rules! tiled {
                        ($n:literal) => {{
                            let src = slice(&data, w, h, desc);
                            transpose_tiled::<$n>(&src, &mut t.as_slice_mut(), flips, w, h);
                        }};
                    }
                    match bpp {
                        1 => tiled!(1),
                        2 => tiled!(2),
                        3 => tiled!(3),
                        4 => tiled!(4),
                        6 => tiled!(6),
                        8 => tiled!(8),
                        12 => tiled!(12),
                        16 => tiled!(16),
                        _ => unreachable!(),
                    }
                    for y in 0..oh {
                        let exp = &want[y as usize * ow as usize * bpp..][..ow as usize * bpp];
                        assert_eq!(
                            b.as_slice().row(y),
                            exp,
                            "blocked {o:?} {desc:?} {w}x{h} row {y}"
                        );
                        assert_eq!(
                            t.as_slice().row(y),
                            exp,
                            "tiled {o:?} {desc:?} {w}x{h} row {y}"
                        );
                    }
                }
            }
        }
    }
}

/// Strided sources across the boundary-rich small grid: a padded input
/// stride must produce byte-identical output to a tight one for every
/// orientation and descriptor (padding bytes must never leak into the
/// transposed result, and the SIMD loads must honour the real stride).
#[test]
fn exhaustive_strided_sources_vs_tight() {
    let descs = [
        PixelDescriptor::GRAY8,
        PixelDescriptor::GRAYA8,
        PixelDescriptor::RGB8,
        PixelDescriptor::RGBA8,
        PixelDescriptor::RGB16,
        PixelDescriptor::RGBA16,
        PixelDescriptor::RGBF32,
        PixelDescriptor::RGBAF32,
    ];
    // Boundary-dense but bounded (strided builds cost a copy per case).
    let vals = [1u32, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 128];
    for &desc in &descs {
        let bpp = desc.bytes_per_pixel();
        for &w in &vals {
            for &h in &vals {
                let tight_stride = w as usize * bpp;
                let pad = tight_stride + 13 * bpp; // pixel-aligned, non-tile-aligned padding
                let tight = fill(tight_stride * h as usize);
                let mut padded = alloc::vec![0xA5u8; pad * h as usize];
                for y in 0..h as usize {
                    padded[y * pad..y * pad + tight_stride]
                        .copy_from_slice(&tight[y * tight_stride..][..tight_stride]);
                }
                for &o in &Orientation::ALL {
                    let ts = PixelSlice::new(&tight, w, h, tight_stride, desc).unwrap();
                    let ps = PixelSlice::new(&padded, w, h, pad, desc).unwrap();
                    let a = apply_orientation(ts, o);
                    let b = apply_orientation(ps, o);
                    for y in 0..a.height() {
                        assert_eq!(
                            a.as_slice().row(y),
                            b.as_slice().row(y),
                            "{o:?} {desc:?} {w}x{h} strided row {y}"
                        );
                    }
                }
            }
        }
    }
}

/// Pathological misalignment: force the source pixel bytes to begin at a
/// chosen address mod 64 (1, 3, 7, 13, 31, 63 — every "bad" offset for
/// 16/32/64-byte SIMD loads) over odd, non-power-of-two byte strides that
/// rotate the per-row misalignment, and assert byte-identical output to
/// the independent oracle. Every load/store in the kernels is an unaligned
/// variant (`loadu` / `storeu` / `vld1` / `vst1`), so this must hold; the
/// test proves it and gates against any future aligned-load regression
/// (an aligned load at one of these addresses would fault on x86, not
/// merely corrupt). The misalignment is constructed from the actual
/// allocation address, so it is independent of how the allocator aligns
/// `Vec<u8>`.
#[test]
fn pathological_misaligned_sources_vs_oracle() {
    let descs = [
        PixelDescriptor::GRAY8,
        PixelDescriptor::GRAYA8,
        PixelDescriptor::RGB8,
        PixelDescriptor::RGBA8,
        PixelDescriptor::RGB16,
        PixelDescriptor::RGBA16,
        PixelDescriptor::RGBF32,
        PixelDescriptor::RGBAF32,
    ];
    // Targets are SIMD-misaligned (never a multiple of 16/32/64) but
    // respect each format's channel alignment (`min_alignment` = channel
    // byte size: 1 for u8, 2 for u16, 4 for f32) — which `PixelSlice::new`
    // enforces, so feeding a sub-channel-aligned pointer is an API misuse,
    // not a kernel concern. Picked per descriptor below.
    let dims = [
        (1u32, 1u32),
        (7, 7),
        (8, 8),
        (16, 16),
        (17, 17),
        (33, 31),
        (31, 33),
        (64, 17),
        (17, 64),
        (65, 9),
    ];
    let pad_px = [1usize, 3]; // byte stride = (w + pad) * bpp → non-power-of-two
    for &desc in &descs {
        let bpp = desc.bytes_per_pixel();
        let targets: &[usize] = match desc.min_alignment() {
            1 => &[1, 3, 7, 13, 31, 63],
            2 => &[2, 6, 14, 30, 62],
            4 => &[4, 12, 28, 60],
            _ => unreachable!("channel alignment is 1/2/4"),
        };
        for &(w, h) in &dims {
            let tight_stride = w as usize * bpp;
            let tight = fill(tight_stride * h as usize);
            for &pp in &pad_px {
                let stride = (w as usize + pp) * bpp;
                let need = stride * h as usize;
                for &target in targets {
                    // Place the pixel data so its first byte lands at an
                    // address ≡ target (mod 64), whatever the base align.
                    let mut raw = alloc::vec![0xC3u8; need + 64];
                    let base = raw.as_ptr() as usize;
                    let startoff = ((64 - (base % 64)) + target) % 64;
                    for y in 0..h as usize {
                        let d = startoff + y * stride;
                        raw[d..d + tight_stride]
                            .copy_from_slice(&tight[y * tight_stride..][..tight_stride]);
                    }
                    let addr = raw.as_ptr() as usize + startoff;
                    assert_eq!(addr % 64, target, "misalignment construction");
                    assert_eq!(
                        addr % desc.min_alignment(),
                        0,
                        "channel alignment must hold (API precondition)"
                    );
                    assert_ne!(addr % 16, 0, "must be SIMD-misaligned to be pathological");
                    for &o in &Orientation::ALL {
                        let (ow, oh, want) = naive_oracle(&tight, w, h, desc, o);
                        let view = &raw[startoff..startoff + need];
                        let s = PixelSlice::new(view, w, h, stride, desc).unwrap();
                        let got = apply_orientation(s, o);
                        assert_eq!(
                            (got.width(), got.height()),
                            (ow, oh),
                            "dims target={target} pad={pp} {o:?} {desc:?} {w}x{h}"
                        );
                        let gs = got.as_slice();
                        for y in 0..oh {
                            let exp = &want[y as usize * ow as usize * bpp..][..ow as usize * bpp];
                            assert_eq!(
                                gs.row(y),
                                exp,
                                "misaligned target={target} pad={pp} {o:?} {desc:?} {w}x{h} row {y}"
                            );
                        }
                    }
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
    // Dense grid: the in-place permutation (diagonal swap for square,
    // cycle-following for non-square) is correctness-critical and its
    // behaviour depends on the gcd structure of w×h, so sweep every
    // small (w, h) rather than a hand-picked spread. Plus a few larger
    // and 1×N / N×1 shapes.
    let mut dims: alloc::vec::Vec<(u32, u32)> = alloc::vec::Vec::new();
    for w in 1u32..=20 {
        for h in 1u32..=20 {
            dims.push((w, h));
        }
    }
    for &d in &[
        (32u32, 32u32),
        (33, 31),
        (1, 64),
        (64, 1),
        (40, 24),
        (24, 40),
    ] {
        dims.push(d);
    }
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
                apply_orientation_in_place(&mut buf, o).expect("in_place should accept bpp ≤ 16");
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

/// Degenerate zero-area inputs (`w == 0` or `h == 0`) must be accepted and yield
/// the correctly-dimensioned empty result without panicking, on every API and
/// orientation — exercising the `w == 0 || h == 0` guards in
/// `apply_orientation_into` and `orient_in_place_impl`. A fully-cropped strip or
/// an empty codec frame is a real input, not a contrived one.
#[test]
fn zero_area_inputs_are_handled() {
    let desc = PixelDescriptor::RGBA8;
    for &(w, h) in &[(0u32, 5u32), (5, 0), (0, 0)] {
        for &o in &Orientation::ALL {
            let (ow, oh) = o.output_dimensions(w, h);

            // Allocating path: builds the output and takes the zero-area
            // early-return in `apply_orientation_into`.
            let src = PixelBuffer::new(w, h, desc);
            let out = apply_orientation(src.as_slice(), o);
            assert_eq!((out.width(), out.height()), (ow, oh), "{o:?} {w}x{h} dims");

            // In-place path: the zero-area early-return in `orient_in_place_impl`,
            // which must still re-describe to the swapped geometry.
            let mut ip = PixelBuffer::new(w, h, desc);
            apply_orientation_in_place(&mut ip, o).expect("zero-area in-place ok");
            assert_eq!(
                (ip.width(), ip.height()),
                (ow, oh),
                "{o:?} {w}x{h} in-place dims"
            );
        }
    }
}

/// In-place baking must carry the source's `ColorContext` through to the
/// re-described view (the `Some(color)` arm of the rewrap) and leave the pixels
/// identical to the out-of-place bake. Colour signalling is not pixels, but
/// silently dropping it would mis-tag the output.
#[test]
fn in_place_preserves_color_context() {
    let desc = PixelDescriptor::RGBA8;
    let (w, h) = (6u32, 4u32);
    let data = fill(w as usize * h as usize * 4);
    let ctx = alloc::sync::Arc::new(zenpixels::ColorContext::from_cicp(zenpixels::Cicp::SRGB));
    for &o in &Orientation::ALL {
        let mut buf = PixelBuffer::new(w, h, desc).with_color_context(ctx.clone());
        {
            let mut s = buf.as_slice_mut();
            for y in 0..h {
                s.row_mut(y)
                    .copy_from_slice(&data[y as usize * w as usize * 4..][..w as usize * 4]);
            }
        }
        apply_orientation_in_place(&mut buf, o).expect("in-place ok");
        assert!(buf.color_context().is_some(), "{o:?} dropped color context");
        let want = apply_orientation(slice(&data, w, h, desc), o);
        for y in 0..buf.height() {
            assert_eq!(
                buf.as_slice().row(y),
                want.as_slice().row(y),
                "{o:?} row {y}"
            );
        }
    }
}

/// In-place baking of a SIMD-aligned (padded-stride) buffer must compact the
/// padded rows to tight before permuting (the `in_stride != tight` path) and
/// still match the out-of-place bake. Decoders routinely hand us row-padded
/// buffers, so this is a common in-place input.
#[test]
fn in_place_padded_stride_matches_out_of_place() {
    for &desc in &[
        PixelDescriptor::RGBA8,
        PixelDescriptor::RGB8,
        PixelDescriptor::GRAY8,
    ] {
        let bpp = desc.bytes_per_pixel();
        for &(w, h) in &[(5u32, 7u32), (13, 9), (17, 3)] {
            let data = fill(w as usize * h as usize * bpp);
            for &o in &Orientation::ALL {
                // simd_align 32 pads the row whenever w*bpp isn't a multiple of
                // lcm(bpp, 32) — true for every (desc, w) here.
                let mut buf = PixelBuffer::new_simd_aligned(w, h, desc, 32);
                assert!(
                    buf.as_slice().stride() > w as usize * bpp,
                    "{desc:?} {w}x{h}: expected a padded stride"
                );
                {
                    let mut s = buf.as_slice_mut();
                    for y in 0..h {
                        s.row_mut(y).copy_from_slice(
                            &data[y as usize * w as usize * bpp..][..w as usize * bpp],
                        );
                    }
                }
                let want = apply_orientation(slice(&data, w, h, desc), o);
                apply_orientation_in_place(&mut buf, o).expect("padded in-place ok");
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

/// `inverse_flips` is `Some` exactly for the four transposing orientations and
/// `None` for the rest — the contract `do_transpose`'s `if let Some(flips)` and
/// `transpose16_deep`'s `else` fallback rely on to route non-transposing (and
/// any future `#[non_exhaustive]`) variants to the generic scatter.
#[test]
fn inverse_flips_some_iff_transposing() {
    for &o in &Orientation::ALL {
        let expect_some = matches!(
            o,
            Orientation::Transpose
                | Orientation::Rotate90
                | Orientation::Rotate270
                | Orientation::Transverse
        );
        assert_eq!(inverse_flips(o).is_some(), expect_some, "{o:?}");
    }
}
