//! Brute-force parity test: v1 (`stamp_trc_kernels!`) vs v2 (`fast_gamut_v2`).
//!
//! For every (src_trc, dst_trc) pair v2 supports, both RGB and RGBA, across
//! many input sizes and pseudo-random pixel content, asserts that v1 and v2
//! produce numerically equivalent output. Catches algorithmic divergence the
//! per-pair unit tests can miss.
//!
//! Tolerance: per-channel absolute diff `< TOL`. v1 and v2 use the same
//! linear-srgb TRC kernels (`*_x{4,8,16}<T>` generic family vs the
//! `tokens::x8::*_v3` per-tier wrappers), so on the V3 path they call
//! identical floating-point sequences and *should* produce byte-identical
//! output for in-bounds inputs. The tolerance accommodates harmless
//! reorderings the optimizer may apply differently to the two macro shapes.
//!
//! Gated on `__bench_v1_v2` so the v1 dispatch shim is in scope.

#![cfg(feature = "__bench_v1_v2")]

use zenpixels::TransferFunction;
use zenpixels_convert::__bench_v1_v2 as bench;

const TOL: f32 = 5e-5;

// Representative gamut matrices spanning what production hits.
const MATS: &[(&str, [[f32; 3]; 3])] = &[
    ("identity", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    (
        "P3→sRGB",
        [
            [1.2249, -0.2247, 0.0],
            [-0.0420, 1.0419, 0.0],
            [-0.0197, -0.0786, 1.0983],
        ],
    ),
    (
        "BT.2020→sRGB",
        [
            [1.6605, -0.5876, -0.0728],
            [-0.1246, 1.1329, -0.0083],
            [-0.0182, -0.1006, 1.1187],
        ],
    ),
    (
        "sRGB→P3",
        [
            [0.8225, 0.1774, 0.0001],
            [0.0331, 0.9669, 0.0],
            [0.0171, 0.0724, 0.9105],
        ],
    ),
];

// Sizes spanning sub-chunk, exact-chunk, and mixed-chunk for both wide
// (16-pixel) and native V3 (8-pixel) bodies.
const SIZES: &[usize] = &[1, 3, 5, 7, 8, 9, 13, 16, 17, 19, 23, 31, 32, 33, 64, 256, 1024, 4096];

const PAIRS: &[(TransferFunction, TransferFunction)] = &[
    (TransferFunction::Linear, TransferFunction::Linear),
    (TransferFunction::Srgb, TransferFunction::Srgb),
    (TransferFunction::Bt709, TransferFunction::Bt709),
    (TransferFunction::Pq, TransferFunction::Pq),
    (TransferFunction::Hlg, TransferFunction::Hlg),
    (TransferFunction::Gamma22, TransferFunction::Gamma22),
    (TransferFunction::Pq, TransferFunction::Srgb),
    (TransferFunction::Hlg, TransferFunction::Srgb),
    (TransferFunction::Srgb, TransferFunction::Pq),
    (TransferFunction::Bt709, TransferFunction::Srgb),
    (TransferFunction::Srgb, TransferFunction::Bt709),
    (TransferFunction::Gamma22, TransferFunction::Srgb),
    (TransferFunction::Srgb, TransferFunction::Gamma22),
];

/// xorshift32 — deterministic, reproducible per (seed, size, pair) combo.
fn xorshift_pixels(seed: u32, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Map u32 → [0, 1] uniform-ish.
        out.push(state as f32 / u32::MAX as f32);
    }
    out
}

/// Edge-case inputs: 0, 1, large positive, denormal, near-threshold.
fn edge_pixels(count: usize) -> Vec<f32> {
    let edges = [
        0.0,
        1.0,
        0.5,
        0.0031308,                  // sRGB linear↔gamma threshold
        0.04045,                    // sRGB encoded↔linear threshold
        0.018053968510807,          // BT.709 beta
        f32::MIN_POSITIVE,          // smallest normal
        1.0e-10,                    // tiny
        0.99999994,                 // just below 1
        2.0,                        // out of range
    ];
    (0..count).map(|i| edges[i % edges.len()]).collect()
}

fn compare(name: &str, a: &[f32], b: &[f32], channels: usize) {
    assert_eq!(a.len(), b.len(), "[{name}] length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let pixel_idx = i / channels;
        let chan = i % channels;
        let diff = (x - y).abs();
        if diff > TOL || x.is_nan() != y.is_nan() {
            panic!(
                "[{name}] pixel {pixel_idx} chan {chan}: v1={x} v2={y} diff={diff:e}"
            );
        }
    }
}

fn run_pair_rgb(
    pair_name: &str,
    mat_name: &str,
    m: &[[f32; 3]; 3],
    width: usize,
    seed: u32,
    src: TransferFunction,
    dst: TransferFunction,
) {
    let pixels = if seed == 0 {
        edge_pixels(width * 3)
    } else {
        xorshift_pixels(seed, width * 3)
    };
    let mut buf_v1 = pixels.clone();
    let mut buf_v2 = pixels.clone();

    let ok_v1 = bench::rgb_v1(m, &mut buf_v1, src, dst);
    let ok_v2 = bench::rgb_v2(m, &mut buf_v2, src, dst);
    assert_eq!(
        ok_v1, ok_v2,
        "[{pair_name}/{mat_name}/RGB w={width} seed={seed}] handler-mismatch (v1={ok_v1}, v2={ok_v2})"
    );
    if !ok_v1 {
        return; // both declined the pair; nothing to compare.
    }
    compare(
        &format!("{pair_name}/{mat_name}/RGB w={width} seed={seed}"),
        &buf_v1,
        &buf_v2,
        3,
    );
}

fn run_pair_rgba(
    pair_name: &str,
    mat_name: &str,
    m: &[[f32; 3]; 3],
    width: usize,
    seed: u32,
    src: TransferFunction,
    dst: TransferFunction,
) {
    let mut pixels = if seed == 0 {
        edge_pixels(width * 4)
    } else {
        xorshift_pixels(seed, width * 4)
    };
    // Mark alpha lanes with a sentinel pattern so we can prove byte-exact
    // passthrough.
    for px in 0..width {
        pixels[px * 4 + 3] = ((px as u32 + seed) & 0xFF) as f32 / 255.0;
    }
    let mut buf_v1 = pixels.clone();
    let mut buf_v2 = pixels.clone();
    let alpha_baseline: Vec<f32> = (0..width).map(|p| pixels[p * 4 + 3]).collect();

    let ok_v1 = bench::rgba_v1(m, &mut buf_v1, src, dst);
    let ok_v2 = bench::rgba_v2(m, &mut buf_v2, src, dst);
    assert_eq!(
        ok_v1, ok_v2,
        "[{pair_name}/{mat_name}/RGBA w={width} seed={seed}] handler-mismatch"
    );
    if !ok_v1 {
        return;
    }
    compare(
        &format!("{pair_name}/{mat_name}/RGBA w={width} seed={seed}"),
        &buf_v1,
        &buf_v2,
        4,
    );
    // Alpha must be byte-exact unchanged on BOTH paths (CLAUDE.md zero-
    // tolerance for image corruption).
    for (px, expected) in alpha_baseline.iter().enumerate() {
        let v1_a = buf_v1[px * 4 + 3];
        let v2_a = buf_v2[px * 4 + 3];
        assert_eq!(
            v1_a.to_bits(),
            expected.to_bits(),
            "[{pair_name}/{mat_name}/RGBA w={width} seed={seed}] v1 mutated alpha at px {px}"
        );
        assert_eq!(
            v2_a.to_bits(),
            expected.to_bits(),
            "[{pair_name}/{mat_name}/RGBA w={width} seed={seed}] v2 mutated alpha at px {px}"
        );
    }
}

fn pair_label(src: TransferFunction, dst: TransferFunction) -> String {
    format!("{src:?}→{dst:?}")
}

#[test]
fn brute_force_v1_v2_parity_rgb() {
    // 13 pairs × 4 mats × 18 sizes × 4 seeds (1 edge + 3 random) = 3744 cases.
    let seeds = [0u32, 0xCAFEBABE, 0x12345678, 0xDEADBEEF];
    for &(src, dst) in PAIRS {
        let pair_name = pair_label(src, dst);
        for (mat_name, mat) in MATS {
            for &width in SIZES {
                for &seed in &seeds {
                    run_pair_rgb(&pair_name, mat_name, mat, width, seed, src, dst);
                }
            }
        }
    }
}

#[test]
fn brute_force_v1_v2_parity_rgba() {
    let seeds = [0u32, 0xCAFEBABE, 0x12345678, 0xDEADBEEF];
    for &(src, dst) in PAIRS {
        let pair_name = pair_label(src, dst);
        for (mat_name, mat) in MATS {
            for &width in SIZES {
                for &seed in &seeds {
                    run_pair_rgba(&pair_name, mat_name, mat, width, seed, src, dst);
                }
            }
        }
    }
}

/// Wider sweep at exactly the SIMD chunk boundaries (1, exact, +1, +width-1)
/// for both wide (16) and native (8) widths. Catches off-by-one in the
/// chunked vs scalar tail split.
#[test]
fn brute_force_chunk_boundaries() {
    let pairs_subset = &[
        (TransferFunction::Srgb, TransferFunction::Srgb),
        (TransferFunction::Bt709, TransferFunction::Bt709),
        (TransferFunction::Pq, TransferFunction::Pq),
    ];
    let widths = [1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33];
    let mat = &MATS[1].1; // P3→sRGB
    for &(src, dst) in pairs_subset {
        let pair_name = pair_label(src, dst);
        for &width in &widths {
            for seed in [1u32, 42, 0xFEEDFACE] {
                run_pair_rgb(&pair_name, "P3→sRGB", mat, width, seed, src, dst);
                run_pair_rgba(&pair_name, "P3→sRGB", mat, width, seed, src, dst);
            }
        }
    }
}
