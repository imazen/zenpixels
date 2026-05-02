//! Paired v1 vs v2 zenbench harness.
//!
//! Registers v1 (`stamp_trc_kernels!`) and v2 (`fast_gamut_v2`) dispatch
//! entry points as two `g.bench(...)` calls in the same group so zenbench
//! interleaves them in randomized round-robin order. The `vs base` column
//! in the output gives bias-free deltas — paired statistics from a single
//! run, immune to the thermal/turbo drift that an A-then-B workflow ate
//! in the prior `bench_t7_gamut` BEFORE/AFTER comparison.

use std::hint::black_box;
use zenbench::{Suite, Throughput};

use zenpixels::TransferFunction;
use zenpixels_convert::__bench_v1_v2 as bench;

const SIZES: &[(&str, usize)] = &[("256px", 256), ("4096px", 4096), ("1080p", 1920 * 1080)];

// P3→sRGB primary matrix in linear space (close enough — exact constants
// don't affect timing, only correctness, and we're benching the kernel
// not the matrix construction).
const M_P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.2249, -0.2247, 0.0],
    [-0.0420, 1.0419, 0.0],
    [-0.0197, -0.0786, 1.0983],
];

fn make_rgb(width: usize) -> Vec<f32> {
    (0..width * 3)
        .map(|i| ((i as f32 * 0.012345) % 1.0))
        .collect()
}

fn make_rgba(width: usize) -> Vec<f32> {
    (0..width * 4)
        .map(|i| if i % 4 == 3 { 0.5 } else { (i as f32 * 0.012345) % 1.0 })
        .collect()
}

fn pair_rgb(suite: &mut Suite, name: &str, src: TransferFunction, dst: TransferFunction) {
    for &(label, width) in SIZES {
        let bytes = (width * 3 * 4) as u64; // f32 RGB
        let data_seed = make_rgb(width);
        suite.group(format!("RGB  {name}  {label}"), |g| {
            g.throughput(Throughput::Bytes(bytes));
            {
                let mut buf = data_seed.clone();
                g.bench("v1", move |b| {
                    b.iter(|| {
                        bench::rgb_v1(&M_P3_TO_SRGB, &mut buf, src, dst);
                        black_box(());
                    })
                });
            }
            {
                let mut buf = data_seed.clone();
                g.bench("v2", move |b| {
                    b.iter(|| {
                        bench::rgb_v2(&M_P3_TO_SRGB, &mut buf, src, dst);
                        black_box(());
                    })
                });
            }
        });
    }
}

fn pair_rgba(suite: &mut Suite, name: &str, src: TransferFunction, dst: TransferFunction) {
    for &(label, width) in SIZES {
        let bytes = (width * 4 * 4) as u64; // f32 RGBA
        let data_seed = make_rgba(width);
        suite.group(format!("RGBA {name}  {label}"), |g| {
            g.throughput(Throughput::Bytes(bytes));
            {
                let mut buf = data_seed.clone();
                g.bench("v1", move |b| {
                    b.iter(|| {
                        bench::rgba_v1(&M_P3_TO_SRGB, &mut buf, src, dst);
                        black_box(());
                    })
                });
            }
            {
                let mut buf = data_seed.clone();
                g.bench("v2", move |b| {
                    b.iter(|| {
                        bench::rgba_v2(&M_P3_TO_SRGB, &mut buf, src, dst);
                        black_box(());
                    })
                });
            }
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        use TransferFunction::*;
        // Same-TRC (the production hot paths)
        pair_rgb(suite, "Srgb→Srgb", Srgb, Srgb);
        pair_rgba(suite, "Srgb→Srgb", Srgb, Srgb);
        pair_rgb(suite, "Bt709→Bt709", Bt709, Bt709);
        pair_rgba(suite, "Bt709→Bt709", Bt709, Bt709);
        pair_rgb(suite, "Pq→Pq", Pq, Pq);
        pair_rgb(suite, "Hlg→Hlg", Hlg, Hlg);
        pair_rgb(suite, "Gamma22→Gamma22", Gamma22, Gamma22);
        // Cross-TRC (HDR→SDR conversion paths)
        pair_rgb(suite, "Pq→Srgb", Pq, Srgb);
        pair_rgb(suite, "Hlg→Srgb", Hlg, Srgb);
        pair_rgb(suite, "Bt709→Srgb", Bt709, Srgb);
        pair_rgb(suite, "Srgb→Bt709", Srgb, Bt709);
    });
}
