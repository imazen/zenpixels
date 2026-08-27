//! Per-kernel NEON-vs-forced-scalar for the scan predicates.
//!
//! These are the crate's hottest per-pixel reductions (is_opaque /
//! is_grayscale drive format-negotiation decisions on every image). The other
//! benches here measure conversion pipelines; an aggregate cannot reveal one
//! kernel losing to its own scalar fallback. That failure mode was real —
//! three zenfilters NEON kernels were measurably slower than their scalar tier
//! in this same sweep.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is the magetypes
//! scalar tier WITH LLVM autovectorization. ~1.00× means both arms compiled to
//! equivalent work, not that SIMD is missing. A ratio BELOW 1.00 is the bug.
//!
//! Both an all-true buffer and an early-exit buffer are measured: these
//! predicates can bail on the first failing pixel, so a uniformly-true input
//! measures steady-state throughput while a fails-early input measures the
//! dispatch/entry cost that would otherwise be hidden.
//!
//! Run: `cargo bench -p zenpixels-convert --bench kernel_tiers`

use zenbench::prelude::*;
use zenpixels_convert::__bench_scan::{FusedRequest, fused_predicates_rgba8_cg};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

const PX: usize = 1 << 20;

/// Opaque RGBA8: alpha all 255 so the predicate must scan the whole buffer.
fn rgba8_opaque() -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    (0..PX * 4)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if i % 4 == 3 { 255 } else { (s >> 24) as u8 }
        })
        .collect()
}

/// Grayscale RGBA8: r == g == b everywhere, alpha 255.
fn rgba8_gray() -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    let mut v = Vec::with_capacity(PX * 4);
    for _ in 0..PX {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let g = (s >> 24) as u8;
        v.extend_from_slice(&[g, g, g, 255]);
    }
    v
}

fn bench_scan(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let opaque: &'static [u8] = Box::leak(rgba8_opaque().into_boxed_slice());
    let gray: &'static [u8] = Box::leak(rgba8_gray().into_boxed_slice());
    // Fails on the first pixel, so this measures entry/dispatch cost rather
    // than steady-state throughput — the two are very different for an
    // early-exit predicate and one number alone would hide that.
    let early: &'static [u8] = Box::leak({
        let mut v = rgba8_opaque().to_vec();
        v[3] = 0;
        v[1] = v[0].wrapping_add(1);
        v.into_boxed_slice()
    });

    let cases: &[(&str, &[u8], FusedRequest)] = &[
        (
            "opaque_only/all_true",
            opaque,
            FusedRequest {
                check_opaque: true,
                check_grayscale: false,
            },
        ),
        (
            "grayscale_only/all_true",
            gray,
            FusedRequest {
                check_opaque: false,
                check_grayscale: true,
            },
        ),
        (
            "both/all_true",
            gray,
            FusedRequest {
                check_opaque: true,
                check_grayscale: true,
            },
        ),
        (
            "both/fails_first_pixel",
            early,
            FusedRequest {
                check_opaque: true,
                check_grayscale: true,
            },
        ),
    ];

    for &(name, data, req) in cases {
        suite.compare(format!("fused_predicates/{name}"), |g| {
            g.throughput(Throughput::Elements(PX as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.iter(move || {
                        set_simd(simd);
                        fused_predicates_rgba8_cg(data, req)
                    })
                });
            }
        });
    }

    set_simd(true);
}

/// `convert_linear_rgba` — the identity-TRC gamut path: a 3x3 matrix per pixel,
/// in place, on f32 RGBA.
///
/// Measured 2026-08-01 because it has NO dispatch at all — it is a plain
/// `chunks_exact_mut(4)` scalar loop, while its TRC-carrying siblings in the
/// same file all dispatch `[v3, neon, wasm128, scalar]`. Each output channel
/// needs all three input channels, so this is a cross-lane (transpose-shaped)
/// problem, which is the pattern LLVM autovectorizes worst and `vld4q_f32`
/// handles natively.
///
/// The A/B here is scalar-vs-hand-written, NOT tier-toggled: there is no
/// dispatch to toggle yet. That is the point of the measurement.
fn bench_linear_gamut(suite: &mut Suite) {
    const PX: usize = 1 << 18;
    let m = [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ];
    let mk = || -> Vec<f32> {
        let mut s = 0x9e37_79b9u32;
        (0..PX * 4)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 16_777_216.0
            })
            .collect()
    };
    let mk3 = || -> Vec<f32> {
        let mut s = 0x1234_5677u32;
        (0..PX * 3)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 16_777_216.0
            })
            .collect()
    };
    suite.compare("convert_linear_rgb", |g| {
        g.throughput(Throughput::Bytes((PX * 12) as u64));
        g.bench("shipped", move |b| {
            b.with_input(mk3).run(move |mut d| {
                zenpixels_convert::__bench_scan::convert_linear_rgb(&m, &mut d);
                d
            })
        });
    });
    suite.compare("convert_linear_rgba", |g| {
        g.throughput(Throughput::Bytes((PX * 16) as u64));
        g.bench("shipped", move |b| {
            b.with_input(mk).run(move |mut d| {
                zenpixels_convert::__bench_scan::convert_linear_rgba(&m, &mut d);
                d
            })
        });
    });
}

zenbench::main!(bench_linear_gamut, bench_scan);
