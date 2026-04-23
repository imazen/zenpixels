//! Head-to-head comparison: the in-house `oklab::fast_cbrt` (bit-trick + 2
//! Newton-Raphson iterations) vs magetypes' `cbrt_lowp_f32` (Kahan + 1 Halley)
//! and `cbrt_midp_f32` (Kahan + 2 Halley).
//!
//! Scalar-only — matches what zenpixels-convert's current Oklab path uses.
//!
//! Run: `cargo test --release -p zenpixels-convert --test cbrt_compare -- --nocapture`

use magetypes::nostd_math::{cbrt_lowp_f32, cbrt_midp_f32};
use zenpixels_convert::oklab::fast_cbrt;

/// ULP error against `std::f32::cbrt` (considered the reference).
fn ulp_err(ours: f32, std_result: f32) -> u32 {
    if ours == std_result {
        return 0;
    }
    if ours.is_nan() || std_result.is_nan() {
        return u32::MAX;
    }
    let a = ours.to_bits() as i32;
    let b = std_result.to_bits() as i32;
    a.wrapping_sub(b).unsigned_abs()
}

/// Sweep a range of positive f32 bit patterns and collect ULP stats.
fn sweep<F: Fn(f32) -> f32>(name: &str, f: F) -> (u32, u64, usize) {
    // Positive normals only: bit ranges 0x00800000..0x7f7fffff.
    // Step large enough to finish quickly but still cover ~200k samples
    // across the full magnitude range.
    let mut max_ulp: u32 = 0;
    let mut sum_ulp: u64 = 0;
    let mut count: usize = 0;

    // Walk every 2048-th positive-normal f32 bit pattern — dense enough
    // to find outlier ULPs within each magnitude octave.
    let start = 0x0080_0000u32;
    let end = 0x7f7f_ffffu32;
    let step = 2048u32;

    let mut bits = start;
    while bits < end {
        let x = f32::from_bits(bits);
        let std_r = x.cbrt();
        let ours = f(x);
        let e = ulp_err(ours, std_r);
        if e != u32::MAX {
            max_ulp = max_ulp.max(e);
            sum_ulp += e as u64;
            count += 1;
        }
        bits = bits.wrapping_add(step);
    }

    let mean = if count == 0 { 0 } else { sum_ulp / count as u64 };
    println!(
        "{name:26}  max ULP = {max_ulp:>8}   mean ULP = {mean:>6}   n = {count}"
    );
    (max_ulp, sum_ulp, count)
}

#[test]
fn cbrt_precision_sweep() {
    println!();
    println!("CBRT precision sweep — positive-normal f32 domain (every 2048th pattern)");
    println!("Reference: std::f32::cbrt");
    println!();
    sweep("oklab::fast_cbrt        ", fast_cbrt);
    sweep("magetypes::cbrt_lowp_f32", cbrt_lowp_f32);
    sweep("magetypes::cbrt_midp_f32", cbrt_midp_f32);
    println!();
}

/// Sweep the Oklab-realistic input range: linear RGB in [0, 1] produces LMS
/// values that are also approximately in [0, 1] for typical colors (slightly
/// beyond for wide-gamut sources). This is the range that actually matters
/// for our use case.
#[test]
fn cbrt_precision_oklab_range() {
    println!();
    println!("CBRT precision sweep — Oklab LMS domain [1e-6, 2.0] (every 2048th pattern)");
    println!("Reference: std::f32::cbrt");
    println!();

    fn sweep_range<F: Fn(f32) -> f32>(name: &str, f: F) {
        let lo = 1e-6_f32.to_bits();
        let hi = 2.0_f32.to_bits();
        let step = 2048u32;

        let mut max_ulp: u32 = 0;
        let mut sum_ulp: u64 = 0;
        let mut count: usize = 0;

        let mut bits = lo;
        while bits < hi {
            let x = f32::from_bits(bits);
            let std_r = x.cbrt();
            let ours = f(x);
            let e = ulp_err(ours, std_r);
            if e != u32::MAX {
                max_ulp = max_ulp.max(e);
                sum_ulp += e as u64;
                count += 1;
            }
            bits = bits.wrapping_add(step);
        }

        let mean = if count == 0 { 0 } else { sum_ulp / count as u64 };
        println!(
            "{name:26}  max ULP = {max_ulp:>8}   mean ULP = {mean:>6}   n = {count}"
        );
    }

    sweep_range("oklab::fast_cbrt        ", fast_cbrt);
    sweep_range("magetypes::cbrt_lowp_f32", cbrt_lowp_f32);
    sweep_range("magetypes::cbrt_midp_f32", cbrt_midp_f32);
    println!();
}

/// Scalar speed: run each implementation over a large buffer and measure
/// wall-clock. Release builds only. This is a smoke test — real numbers
/// belong in zenbench.
#[test]
fn cbrt_speed_smoke() {
    use std::hint::black_box;
    use std::time::Instant;

    let n: usize = 1 << 20; // 1M values
    let inputs: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / n as f32).collect();

    fn time_it<F: Fn(f32) -> f32>(name: &str, inputs: &[f32], f: F) {
        // warm up
        let mut acc = 0.0f32;
        for &x in inputs {
            acc += f(x);
        }
        black_box(acc);

        // Measure
        let runs = 5;
        let mut best = std::time::Duration::MAX;
        for _ in 0..runs {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for &x in inputs {
                acc += f(black_box(x));
            }
            black_box(acc);
            best = best.min(t.elapsed());
        }
        let ns_per = best.as_nanos() as f64 / inputs.len() as f64;
        println!(
            "{name:26}  best = {:>8.3} ms   {:>6.2} ns/value",
            best.as_secs_f64() * 1000.0,
            ns_per
        );
    }

    println!();
    println!("CBRT scalar speed — 1M values, best-of-5 (release build recommended)");
    println!();
    time_it("std::f32::cbrt           ", &inputs, f32::cbrt);
    time_it("oklab::fast_cbrt         ", &inputs, fast_cbrt);
    time_it("magetypes::cbrt_lowp_f32 ", &inputs, cbrt_lowp_f32);
    time_it("magetypes::cbrt_midp_f32 ", &inputs, cbrt_midp_f32);
    println!();
}
