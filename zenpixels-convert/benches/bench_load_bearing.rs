//! Load-bearing analysis benches: fused single-pass vs two separate
//! passes vs scalar, and the public trait entry end-to-end.
//!
//! Content is worst-case for the predicates (gray + opaque everywhere):
//! no check ever fails, so the scan walks every byte. Early-exit cases
//! bail inside the first SIMD chunk and don't need a benchmark.
//!
//! Run: `cargo bench --bench bench_load_bearing --features __bench_scan`

use zenbench::prelude::*;
use zenpixels::{AlphaMode, PixelDescriptor, PixelFormat, PixelSlice, TransferFunction};
use zenpixels_convert::__bench_scan::{FusedRequest, fused_predicates_rgba8_cg};
use zenpixels_convert::PixelSliceLoadBearingExt;

// Tiny / small / medium / large — per-call overhead shows at 64×64,
// DRAM bandwidth at 4096×4096 (64 MB working set).
const SIZES: &[(&str, usize)] = &[
    ("  64×64  ", 64),
    (" 256×256 ", 256),
    ("1024×1024", 1024),
    ("4096×4096", 4096),
];

/// Worst-case content: every pixel gray + opaque, so both checks
/// stay true and the scan can never early-exit.
fn gray_opaque_rgba(dim: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(dim * dim * 4);
    for i in 0..dim * dim {
        let g = (i * 7 % 251) as u8;
        v.extend_from_slice(&[g, g, g, 255]);
    }
    v
}

/// Scalar reference with the same early-exit shape as the SIMD kernels'
/// tail loop (the exit branch keeps LLVM from auto-vectorizing it).
fn scalar_fused_ref(rgba: &[u8]) -> (bool, bool) {
    let (mut o, mut g) = (true, true);
    for px in rgba.as_chunks::<4>().0 {
        if o && px[3] != 255 {
            o = false;
        }
        if g && (px[0] != px[1] || px[1] != px[2]) {
            g = false;
        }
        if !(o | g) {
            break;
        }
    }
    (o, g)
}

const fn one_check(opaque: bool, gray: bool) -> FusedRequest {
    FusedRequest {
        check_opaque: opaque,
        check_grayscale: gray,
    }
}

/// Pass-structure comparison on the raw fused kernel.
fn bench_fused_shapes(suite: &mut Suite) {
    for &(label, dim) in SIZES {
        let data = gray_opaque_rgba(dim);
        let bytes = data.len() as u64;
        suite.group(format!("rgba8 predicates {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));

            let d = data.clone();
            g.bench("fused 2-check 1-pass (SIMD)", move |b| {
                b.iter(|| black_box(fused_predicates_rgba8_cg(&d, one_check(true, true))))
            });

            let d = data.clone();
            g.bench("two 1-check passes (SIMD)", move |b| {
                b.iter(|| {
                    let o = fused_predicates_rgba8_cg(&d, one_check(true, false));
                    let gr = fused_predicates_rgba8_cg(&d, one_check(false, true));
                    black_box((o.is_opaque, gr.is_grayscale))
                })
            });

            let d = data.clone();
            g.bench("scalar 2-check 1-pass", move |b| {
                b.iter(|| black_box(scalar_fused_ref(&d)))
            });
        });
    }
}

/// Public trait entry, end-to-end (predicates + sub-byte-depth pass).
fn bench_trait_entry(suite: &mut Suite) {
    for &(label, dim) in SIZES {
        let data = gray_opaque_rgba(dim);
        let bytes = data.len() as u64;
        suite.group(format!("determine_load_bearing {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));

            // Straight alpha: full 2-check fused scan.
            let d = data.clone();
            g.bench("Rgba8 straight (scan all)", move |b| {
                let desc = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
                    .with_transfer(TransferFunction::Srgb);
                b.iter(|| {
                    let s = PixelSlice::new(&d, dim as u32, dim as u32, dim * 4, desc).unwrap();
                    black_box(s.determine_load_bearing())
                })
            });

            // Declared opaque: alpha answered from the descriptor, the
            // scan only runs the grayscale check.
            let d = data.clone();
            g.bench("Rgba8 AlphaMode::Opaque (elided)", move |b| {
                let desc = PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
                    .with_transfer(TransferFunction::Srgb)
                    .with_alpha(Some(AlphaMode::Opaque));
                b.iter(|| {
                    let s = PixelSlice::new(&d, dim as u32, dim as u32, dim * 4, desc).unwrap();
                    black_box(s.determine_load_bearing())
                })
            });
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_fused_shapes(suite);
        bench_trait_entry(suite);
    });
}
