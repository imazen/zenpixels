//! HLG Linear→HLG regression probe.
//!
//! T4 re-bench against local linear-srgb showed Linear→HLG at 4096px
//! dropping from 29.44 to 19.9 GiB/s. Before concluding it's real, test
//! whether the regression is input-distribution-dependent:
//!
//! - uniform [0,1]       — the T4 baseline input (only ~8% hits
//!   HLG's quadratic early-exit at v ≤ 1/12).
//! - all-small (< 1/12)  — 100% early-exit quadratic path.
//! - all-large (> 1/12)  — 100% sinh/log path.
//! - realistic HDR luma  — biased toward low light + midtones,
//!   rough Rec.2020 PQ luma histogram.
//!
//! The OLD kernel was `#[archmage::autoversion]` scalar loop; LLVM
//! could autovectorize small/large cases separately. The NEW kernel
//! is hand-SIMD. If the new kernel is slower on uniform but matches
//! on all-small and all-large, it's a branch-density artifact. If
//! it's slower everywhere, it's a real regression.

use linear_srgb::default::linear_to_hlg_slice;
use zenbench::prelude::*;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

const CHANNELS: usize = 3;

fn make_uniform(n: usize) -> Vec<f32> {
    // Same generator as bench_t4_tf_f32.rs
    (0..n)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0)
        .collect()
}

fn make_all_small(n: usize) -> Vec<f32> {
    // Uniform in [0, 1/12) — every value hits the quadratic early-exit.
    let inv12 = 1.0f32 / 12.0;
    (0..n)
        .map(|i| {
            let u = ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0;
            u * inv12
        })
        .collect()
}

fn make_all_large(n: usize) -> Vec<f32> {
    // Uniform in [1/12, 1] — every value hits the sinh/log path.
    let inv12 = 1.0f32 / 12.0;
    let span = 1.0f32 - inv12;
    (0..n)
        .map(|i| {
            let u = ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0;
            inv12 + u * span
        })
        .collect()
}

fn make_hdr_luma(n: usize) -> Vec<f32> {
    // Rough approximation of HDR luma distribution: heavy weight near
    // 0 (shadows + midtones), tail into highlights. Using an x^3
    // biased uniform that concentrates ~65% of samples below 0.1 and
    // ~90% below 0.3.
    (0..n)
        .map(|i| {
            let u = ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0;
            u * u * u
        })
        .collect()
}

fn main() {
    zenbench::run(|suite| {
        for &(label, width) in SIZES {
            let n = width * CHANNELS;
            let bytes = (n * 4) as u64;

            let uniform = make_uniform(n);
            let all_small = make_all_small(n);
            let all_large = make_all_large(n);
            let hdr = make_hdr_luma(n);

            // Sanity-check that distributions cluster where we said.
            // (Print once, at the first size.)
            if width == 256 {
                let frac_small_uniform = uniform.iter().filter(|&&v| v <= 1.0 / 12.0).count()
                    as f32
                    / uniform.len() as f32;
                let frac_small_hdr =
                    hdr.iter().filter(|&&v| v <= 1.0 / 12.0).count() as f32 / hdr.len() as f32;
                eprintln!(
                    "[hlg-probe] uniform: {:.1}% ≤ 1/12 | hdr: {:.1}% ≤ 1/12",
                    frac_small_uniform * 100.0,
                    frac_small_hdr * 100.0
                );
            }

            suite.group(format!("Linear→HLG  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let mut scratch = vec![0.0f32; n];
                let src = uniform;
                g.bench("uniform [0,1]", move |b| {
                    b.iter(|| {
                        scratch.copy_from_slice(&src);
                        linear_to_hlg_slice(&mut scratch);
                        black_box(());
                    })
                });

                let mut scratch = vec![0.0f32; n];
                let src = all_small;
                g.bench("all-small (quadratic)", move |b| {
                    b.iter(|| {
                        scratch.copy_from_slice(&src);
                        linear_to_hlg_slice(&mut scratch);
                        black_box(());
                    })
                });

                let mut scratch = vec![0.0f32; n];
                let src = all_large;
                g.bench("all-large (sinh/log)", move |b| {
                    b.iter(|| {
                        scratch.copy_from_slice(&src);
                        linear_to_hlg_slice(&mut scratch);
                        black_box(());
                    })
                });

                let mut scratch = vec![0.0f32; n];
                let src = hdr;
                g.bench("hdr-luma (u^3)", move |b| {
                    b.iter(|| {
                        scratch.copy_from_slice(&src);
                        linear_to_hlg_slice(&mut scratch);
                        black_box(());
                    })
                });
            });
        }
    });
}
