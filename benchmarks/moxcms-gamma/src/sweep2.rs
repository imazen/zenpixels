#![forbid(unsafe_code)]
// PR 2: guarded evaluator (m4) against the sign fix (m2) and exact math, over all 2^32 f32 inputs.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
macro_rules! ev {
    ($m:ident, $g:expr, $enc:expr) => {{
        let c = $m::curve_from_gamma($g);
        let e = if $enc {
            c.make_gamma_evaluator().unwrap()
        } else {
            c.make_linear_evaluator().unwrap()
        };
        Box::new(move |x: f32| e.evaluate_value(x)) as Box<dyn Fn(f32) -> f32 + Send + Sync>
    }};
}
fn main() {
    let threads = 8u64;
    for &(g, enc) in &[
        (1.0f32, false),
        (1.8, false),
        (1.8, true),
        (2.19921875, false),
        (2.19921875, true),
        (2.6, false),
        (2.6, true),
        (2.8, false),
        (3.0, false),
        (4.0, false),
        (4.0, true),
    ] {
        let (a, b) = (ev!(m2, g, enc), ev!(m4, g, enc));
        let fixed = (g as f64 * 256.0 + 0.5).floor();
        let p = if enc { 256.0 / fixed } else { fixed / 256.0 };
        let same = AtomicU64::new(0);
        let changed = AtomicU64::new(0);
        let a_bad = AtomicU64::new(0);
        let b_bad = AtomicU64::new(0);
        let not_odd = AtomicU64::new(0);
        let nan_kept = AtomicU64::new(0);
        let nans = AtomicU64::new(0);
        let wa = std::sync::Mutex::new(0f64);
        let wb = std::sync::Mutex::new((0f64, 0f32));
        std::thread::scope(|sc| {
            for t in 0..threads {
                let (a, b, same, changed, a_bad, b_bad, not_odd, nan_kept, nans, wa, wb) = (
                    &a, &b, &same, &changed, &a_bad, &b_bad, &not_odd, &nan_kept, &nans, &wa, &wb,
                );
                sc.spawn(move || {
                    let span = (1u64 << 32) / threads;
                    let (lo, hi) = (
                        t * span,
                        if t + 1 == threads {
                            1u64 << 32
                        } else {
                            (t + 1) * span
                        },
                    );
                    let (mut cs, mut cc, mut ca, mut cb, mut co, mut nk, mut nn) =
                        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
                    let (mut ma, mut mb, mut mx) = (0f64, 0f64, 0f32);
                    for bits in lo..hi {
                        let x = f32::from_bits(bits as u32);
                        let (va, vb) = (a(x), b(x));
                        if va.to_bits() == vb.to_bits() {
                            cs += 1
                        } else {
                            cc += 1
                        }
                        if x.is_nan() {
                            nn += 1;
                            if vb.is_nan() {
                                nk += 1
                            }
                            continue;
                        }
                        if x != 0.0 && (vb.to_bits() != (-b(-x)).to_bits()) {
                            co += 1;
                        }
                        // Compare in f32, exact math rounded: overflow to inf and underflow to 0 count as right.
                        let e = (x as f64).abs().powf(p).copysign(x as f64);
                        let ef = e as f32;
                        let err = |v: f32| {
                            if v == ef {
                                0.0
                            } else if v.is_infinite() || ef.is_infinite() {
                                f64::INFINITY
                            } else {
                                (v as f64 - e).abs() / e.abs().max(f32::MIN_POSITIVE as f64)
                            }
                        };
                        let (ra, rb) = (err(va), err(vb));
                        if ra > 1e-3 {
                            ca += 1
                        }
                        if rb > 1e-3 {
                            cb += 1
                        }
                        if ra > ma {
                            ma = ra
                        }
                        if rb > mb {
                            mb = rb;
                            mx = x
                        }
                    }
                    same.fetch_add(cs, Relaxed);
                    changed.fetch_add(cc, Relaxed);
                    a_bad.fetch_add(ca, Relaxed);
                    b_bad.fetch_add(cb, Relaxed);
                    not_odd.fetch_add(co, Relaxed);
                    nan_kept.fetch_add(nk, Relaxed);
                    nans.fetch_add(nn, Relaxed);
                    let mut w = wa.lock().unwrap();
                    if ma > *w {
                        *w = ma
                    }
                    drop(w);
                    let mut w = wb.lock().unwrap();
                    if mb > w.0 {
                        *w = (mb, mx)
                    }
                });
            }
        });
        let wb = *wb.lock().unwrap();
        println!(
            "curve {g} {}: bit-identical to sign fix {} changed {}; relative error >1e-3: sign fix {} guarded {}; worst guarded {:.3e} at {:e}, worst sign fix {:.3e}; not odd (non-NaN, non-zero) {}; NaN kept {} of {}",
            if enc { "encode" } else { "decode" },
            same.load(Relaxed),
            changed.load(Relaxed),
            a_bad.load(Relaxed),
            b_bad.load(Relaxed),
            wb.0,
            wb.1,
            *wa.lock().unwrap(),
            not_odd.load(Relaxed),
            nan_kept.load(Relaxed),
            nans.load(Relaxed)
        );
    }
}
