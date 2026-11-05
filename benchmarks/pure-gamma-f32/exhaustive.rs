// All 2^32 f32 inputs through the public adapter: gray gamma into linear gray, and linear gray into gamma.
use moxcms::{ColorProfile as P, ToneReprCurve as T};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use za::{ColorProfileSource as S, PixelFormat as F};
use zca::{ConvertOptions, cms::PluggableCms, cms_moxcms::MoxCms};
fn main() {
    let linear = { let mut p = P::new_gray_with_gamma(1.0); p.gray_trc = Some(T::Lut(vec![])); p.encode().unwrap() };
    for &g in &[1.0f32, 1.8, 2.19921875, 2.6, 2.8, 3.0, 4.0] {
        let curved = P::new_gray_with_gamma(g).encode().unwrap();
        let fixed = (g as f64 * 256.0 + 0.5).floor();
        for (dir, src, dst, p) in [("decode", &curved, &linear, fixed / 256.0), ("encode", &linear, &curved, 256.0 / fixed)] {
            let (n, wrong, not_odd, nan_kept, nans) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
            let worst = std::sync::Mutex::new((0f64, 0f32));
            let o = ConvertOptions::retain_channels().with_clip_out_of_gamut(false);
            std::thread::scope(|sc| { for t in 0..8u64 {
                let (n, wrong, not_odd, nan_kept, nans, worst, o) = (&n, &wrong, &not_odd, &nan_kept, &nans, &worst, &o);
                sc.spawn(move || {
                    let mut tr = MoxCms.build_source_transform(S::Icc(src), S::Icc(dst), F::GrayF32, F::GrayF32, o).unwrap().unwrap();
                    let span = (1u64 << 32) / 8; let (lo, hi) = (t * span, if t == 7 { 1u64 << 32 } else { (t + 1) * span });
                    let (mut inp, mut out, mut neg, mut out_neg) = (vec![0f32; 4096], vec![0f32; 4096], vec![0f32; 4096], vec![0f32; 4096]);
                    let (mut cn, mut cw, mut co, mut nk, mut nn, mut wm, mut wx) = (0u64, 0u64, 0u64, 0u64, 0u64, 0f64, 0f32);
                    let mut b = lo;
                    while b < hi {
                        let m = (hi - b).min(4096) as usize;
                        for i in 0..m { inp[i] = f32::from_bits((b + i as u64) as u32); neg[i] = -inp[i]; }
                        tr.try_transform_row(bytemuck::cast_slice(&inp[..m]), bytemuck::cast_slice_mut(&mut out[..m]), m as u32).unwrap();
                        tr.try_transform_row(bytemuck::cast_slice(&neg[..m]), bytemuck::cast_slice_mut(&mut out_neg[..m]), m as u32).unwrap();
                        for i in 0..m {
                            let (x, v) = (inp[i], out[i]); cn += 1;
                            if x.is_nan() { nn += 1; if v.is_nan() { nk += 1; } continue; }
                            if v.to_bits() != (-out_neg[i]).to_bits() { co += 1; }
                            let e = (x as f64).abs().powf(p).copysign(x as f64); let ef = e as f32;
                            let r = if v == ef { 0.0 } else if v.is_infinite() || ef.is_infinite() || !v.is_finite() { f64::INFINITY } else { (v as f64 - e).abs() / e.abs().max(f32::MIN_POSITIVE as f64) };
                            if r > 1e-3 { cw += 1; } if r > wm { wm = r; wx = x; }
                        }
                        b += m as u64;
                    }
                    n.fetch_add(cn, Relaxed); wrong.fetch_add(cw, Relaxed); not_odd.fetch_add(co, Relaxed); nan_kept.fetch_add(nk, Relaxed); nans.fetch_add(nn, Relaxed);
                    let mut w = worst.lock().unwrap(); if wm > w.0 { *w = (wm, wx); }
                });
            } });
            let w = *worst.lock().unwrap();
            println!("gamma {g} {dir}: {} inputs, wrong (>1e-3) {}, worst relative {:.3e} at {:e}, not odd {}, NaN kept {} of {}", n.load(Relaxed), wrong.load(Relaxed), w.0, w.1, not_odd.load(Relaxed), nan_kept.load(Relaxed), nans.load(Relaxed));
        }
    }
}
