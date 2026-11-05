//! Candidate image gamut compressors. All take linear RGB in the target
//! primaries, in place, and leave final bounds to the caller.
//!
//! Scaling OKLab chroma by `s` at constant L and hue is a lerp in cube-root LMS:
//! `x'(s) = L + s (x' - L)`, because the a and b rows of M2 sum to zero and the
//! L row sums to one. So no hue, atan2, sqrt or boundary table is needed: each
//! RGB channel is a cubic in `s`, and the boundary is its first crossing of 0 or 1.
#![allow(clippy::needless_range_loop)]
use archmage::prelude::*;
use zenpixels_convert::oklab::OKLAB_FROM_LMS_CBRT;

type M3 = [[f32; 3]; 3];

/// Largest chroma scale we ever need to know: past 1 / knee nothing changes.
const S_CAP: f32 = 1.25;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub m1: M3,
    pub m1_inv: M3,
    pub knee: f32,
    /// Interior gate: a pixel with `min > tau * max && max < 1 - eps` is untouched.
    pub tau: f32,
    pub eps: f32,
    pub iters: u32,
    /// Luminance row of the target primaries.
    pub luma: [f32; 3],
}

fn knee_scale(knee: f32, s_max: f32) -> f32 {
    let knee_s = knee * s_max;
    let excess = 1.0 - knee_s;
    if excess <= 0.0 {
        return 1.0;
    }
    let range = s_max - knee_s;
    if range <= 0.0 {
        return 0.0;
    }
    knee_s + range * excess / (excess + range)
}

/// Gate every 8 pixels, then solve only the pixels the gate rejected, packed
/// into full SIMD batches so one edge pixel does not cost a whole mixed chunk.
pub fn solve_strip(p: &Params, rgb: &mut [[f32; 3]]) {
    incant!(solve_strip_impl(p, rgb), [v3, neon, wasm128, scalar])
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn solve_strip_impl(token: Token, p: &Params, rgb: &mut [[f32; 3]]) {
    let splat = |v: f32| f32x8::splat_t(token, v);
    let dot = |row: &[f32; 3], a: f32x8, b: f32x8, c: f32x8| -> f32x8 {
        splat(row[0]).mul_add_relaxed(a, splat(row[1]).mul_add_relaxed(b, splat(row[2]) * c))
    };
    let (zero, one, cap, tiny, three) =
        (splat(0.0), splat(1.0), splat(S_CAP), splat(1e-12), splat(3.0));
    let (tau, top, knee) = (splat(p.tau), splat(1.0 - p.eps), splat(p.knee));
    let ok0 = OKLAB_FROM_LMS_CBRT.rows()[0];

    let solve = |px: &mut [[f32; 3]; 8]| {
        let r0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][0]));
        let g0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][1]));
        let b0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][2]));
        let lc = dot(&p.m1[0], r0, g0, b0).cbrt_midp();
        let mc = dot(&p.m1[1], r0, g0, b0).cbrt_midp();
        let sc = dot(&p.m1[2], r0, g0, b0).cbrt_midp();
        let ll = dot(&ok0, lc, mc, sc);
        let (dl, dm, ds) = (lc - ll, mc - ll, sc - ll);

        // Start on the chord from gray (s = 0, every channel L^3) to the input
        // (s = 1). Starting at s = 1 on a far-out color lands on a later crossing.
        let gray = ll * ll * ll;
        let chord = |v: f32x8| -> f32x8 {
            let over = (one - gray) / (v - gray);
            let under = gray / (gray - v);
            f32x8::blend(v.simd_lt(zero), under, f32x8::blend(v.simd_gt(one), over, cap))
        };
        let mut s = chord(r0).min(chord(g0)).min(chord(b0)).max(zero).min(one);
        let (mut xl, mut xm, mut xs) = (ll + s * dl, ll + s * dm, ll + s * ds);
        let (l3, m3, s3) = (xl * xl * xl, xm * xm * xm, xs * xs * xs);
        let mut r = dot(&p.m1_inv[0], l3, m3, s3);
        let mut g = dot(&p.m1_inv[1], l3, m3, s3);
        let mut b = dot(&p.m1_inv[2], l3, m3, s3);
        for _ in 0..p.iters {
            let (gl, gm, gs) = (xl * xl * dl, xm * xm * dm, xs * xs * ds);
            let toward = |v: f32x8, row: &[f32; 3]| -> f32x8 {
                let grad = three * dot(row, gl, gm, gs);
                let face = f32x8::blend(grad.simd_gt(zero), one, zero);
                f32x8::blend(grad.abs().simd_lt(tiny), cap, s + (face - v) / grad)
            };
            let next = toward(r, &p.m1_inv[0]).min(toward(g, &p.m1_inv[1])).min(toward(b, &p.m1_inv[2]));
            s = next.max(zero).min(cap);
            xl = ll + s * dl;
            xm = ll + s * dm;
            xs = ll + s * ds;
            let (l3, m3, s3) = (xl * xl * xl, xm * xm * xm, xs * xs * xs);
            r = dot(&p.m1_inv[0], l3, m3, s3);
            g = dot(&p.m1_inv[1], l3, m3, s3);
            b = dot(&p.m1_inv[2], l3, m3, s3);
        }
        // Rational knee in units of the input's chroma, which sits at s = 1.
        let knee_s = knee * s;
        let excess = one - knee_s;
        let range = s - knee_s;
        let scaled = knee_s + range * excess / (excess + range);
        // No representable chroma at L <= 0 or L >= 1.
        let dead = ll.simd_le(zero) | ll.simd_ge(one);
        let scaled = f32x8::blend(dead, zero, scaled);
        let active = excess.simd_gt(zero) | dead;
        let (ol, om, os) = (ll + scaled * dl, ll + scaled * dm, ll + scaled * ds);
        let (l3, m3, s3) = (ol * ol * ol, om * om * om, os * os * os);
        let rr = f32x8::blend(active, dot(&p.m1_inv[0], l3, m3, s3), r0).to_array();
        let gg = f32x8::blend(active, dot(&p.m1_inv[1], l3, m3, s3), g0).to_array();
        let bb = f32x8::blend(active, dot(&p.m1_inv[2], l3, m3, s3), b0).to_array();
        for i in 0..8 {
            px[i] = [rr[i], gg[i], bb[i]];
        }
    };

    // Rejected pixels wait here with their position, then go through as a batch.
    let mut batch = [[0.5f32; 3]; 8];
    let mut slots = [0usize; 8];
    let mut used = 0;
    let n = rgb.len();
    let mut start = 0;
    while start < n {
        let end = (start + 8).min(n);
        let mut lanes = [[1.0f32; 8]; 3];
        for (i, px) in rgb[start..end].iter().enumerate() {
            lanes[0][i] = px[0];
            lanes[1][i] = px[1];
            lanes[2][i] = px[2];
        }
        let (r, g, b) = (
            f32x8::from_array_t(token, lanes[0]),
            f32x8::from_array_t(token, lanes[1]),
            f32x8::from_array_t(token, lanes[2]),
        );
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let interior = mn.simd_gt(mx * tau) & mx.simd_lt(top);
        // Padding lanes of a short tail are 1.0 and fail the gate, so test them too.
        if end - start == 8 && interior.bitcast_to_i32().all_true() {
            start = end;
            continue;
        }
        let keep = interior.to_array();
        for i in 0..end - start {
            if keep[i].to_bits() == 0 {
                batch[used] = rgb[start + i];
                slots[used] = start + i;
                used += 1;
                if used == 8 {
                    solve(&mut batch);
                    for k in 0..8 {
                        rgb[slots[k]] = batch[k];
                    }
                    used = 0;
                }
            }
        }
        start = end;
    }
    if used > 0 {
        for k in used..8 {
            batch[k] = [0.5; 3];
        }
        solve(&mut batch);
        for k in 0..used {
            rgb[slots[k]] = batch[k];
        }
    }
}

/// One closed-form step: straight line to the gray of equal luminance in linear
/// RGB, with the same knee. Keeps luminance, not OKLab hue.
pub fn line_strip(p: &Params, rgb: &mut [[f32; 3]]) {
    for px in rgb.iter_mut() {
        let y = p.luma[0] * px[0] + p.luma[1] * px[1] + p.luma[2] * px[2];
        if y <= 0.0 {
            *px = [0.0; 3];
            continue;
        }
        let mut s_max = S_CAP;
        for c in 0..3 {
            let d = px[c] - y;
            if d > 1e-9 {
                s_max = s_max.min((1.0 - y).max(0.0) / d);
            } else if d < -1e-9 {
                s_max = s_max.min(y / -d);
            }
        }
        let scale = knee_scale(p.knee, s_max);
        if scale < 1.0 {
            for c in 0..3 {
                px[c] = y + scale * (px[c] - y);
            }
        }
    }
}

pub fn clip_strip(rgb: &mut [[f32; 3]]) {
    for px in rgb.iter_mut() {
        *px = px.map(|v| v.clamp(0.0, 1.0));
    }
}

#[inline]
pub fn gate_passes(p: &Params, px: &[f32; 3]) -> bool {
    let mx = px[0].max(px[1]).max(px[2]);
    let mn = px[0].min(px[1]).min(px[2]);
    mn > p.tau * mx && mx < 1.0 - p.eps
}
