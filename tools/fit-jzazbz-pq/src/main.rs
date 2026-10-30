//! Fit f(x) = ((c1 + c2·x^n)/(1 + c3·x^n))^p — the JzAzBz PQ curve,
//! p = 1.7·2523/32 — as P(a)/Q(a) with a = x^(1/4), x ∈ [0, 1] (0–10000
//! nits), using polyfit's rational fitter (SK → LM → restarts) followed by
//! its f32 coefficient search scored on absolute error.
//!
//! The sweep runs in `a` directly: x-uniform sampling would put almost no
//! samples in the dark range where f is steepest. Prints a table and the
//! Rust constants for the bench.

use polyfit::rational::{
    ErrorWeighting, F32ScoreMetric, F32SearchConfig, RationalFit, RationalFitOptions,
};

const C1: f64 = 3424.0 / 4096.0;
const C2: f64 = 2413.0 / 128.0;
const C3: f64 = 2392.0 / 128.0;
const N: f64 = 2610.0 / 16384.0;
const P: f64 = 1.7 * 2523.0 / 32.0;

fn f_of_a(a: f64) -> f64 {
    let x = a.powi(4);
    let t = x.powf(N);
    ((C1 + C2 * t) / (1.0 + C3 * t)).powf(P)
}

fn main() {
    let degrees: &[(usize, usize)] = &[(3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6)];
    println!("# polyfit rational-polynomial-fitting@4de8e99, a = x^(1/4), a in [0,1]");
    println!("# deg   f64 max_abs   f32 max_abs   f32 avg_abs   samples");
    let mut consts = String::new();
    for &(pd, qd) in degrees {
        let opts = RationalFitOptions {
            weighting: ErrorWeighting::Absolute,
            n_samples: 4000,
            lm_iterations: 400,
            restarts: 8,
            ..RationalFitOptions::default()
        };
        let fit = match RationalFit::from_function(f_of_a, 0.0..=1.0, pd, qd, opts) {
            Ok(f) => f,
            Err(e) => {
                println!("# ({pd},{qd}) fit failed: {e}");
                continue;
            }
        };
        let r = fit.as_polynomial().optimize_f32(
            f_of_a,
            0.0,
            1.0,
            |v| v,
            F32SearchConfig {
                radius: 4,
                max_rounds: 8,
                metric: F32ScoreMetric::AbsoluteError,
                samples: Some(200_000),
            },
        );
        println!(
            "  ({pd},{qd})  {:.3e}     {:.3e}     {:.3e}     {}{}",
            fit.max_abs_error(),
            r.max_abs_err,
            r.avg_abs_err,
            r.values_scored,
            if r.mono_violations > 0 {
                format!("  MONO VIOLATIONS {}", r.mono_violations)
            } else {
                String::new()
            },
        );
        consts += &format!(
            "// ({pd},{qd}): f32 max abs err {:.3e} (polyfit sampled n={})\n\
             const JZ_PQ_{pd}{qd}_P: [f32; {}] = {:?};\n\
             const JZ_PQ_{pd}{qd}_Q: [f32; {}] = {:?};\n",
            r.max_abs_err,
            r.values_scored,
            r.numerator.len(),
            r.numerator,
            r.denominator.len(),
            r.denominator
        );
    }
    println!("\n{consts}");
}
