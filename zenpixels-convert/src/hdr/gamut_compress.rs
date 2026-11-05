//! Soft gamut compression in OKLch: reduces chroma near and past the boundary of
//! a set of primaries, keeping hue and lightness.
//!
//! Scaling OKLab chroma by `s` at constant L and hue is a straight blend in
//! cube-root LMS, `x'(s) = L + s (x' - L)`, because the a and b rows of that
//! matrix sum to zero and the L row sums to one. Each RGB channel is therefore a
//! cubic in `s`, and the gamut boundary is its first crossing of 0 or 1. That needs
//! no hue, `atan2`, square root or lookup table, so it vectorizes directly:
//!
//! 1. A gate in linear RGB passes interior pixels through bit-identical. Only those
//!    it rejects reach the solve, packed into full SIMD batches.
//! 2. A chord from gray to the input starts each lane before its first crossing.
//! 3. Three Newton steps find the crossing, and a rational knee sets the new chroma.
//!
//! Measurements and comparisons are in `docs/gamut-mapping-survey.md`, and the
//! harness is `benchmarks/gamut-compress`.

use archmage::prelude::*;
use zenpixels::ColorPrimaries;

use crate::oklab::{LMS_CBRT_FROM_OKLAB, OKLAB_FROM_LMS_CBRT, OklabSpace};

/// Newton steps on the boundary. At 3, the largest difference from bisection on a
/// BT.2020 grid was 0.000002 in OKLab, and 2 left 0.00098.
const SOLVE_STEPS: u32 = 3;

/// The gate's upper limit: a pixel with its largest channel at or above this is
/// close enough to the top of the cube that it always gets solved.
const GATE_TOP: f32 = 0.95;

/// Smallest `min / max` among colors past the knee, rounded up, from 399 lightness
/// by 1,440 hue boundary points per target (`gamut-compress gate-all`). A pixel with
/// a larger ratio and `max < GATE_TOP` can't be past the knee. Infinity disables it.
fn gate_ratio(primaries: ColorPrimaries, knee: f32) -> f32 {
    #[allow(unreachable_patterns)]
    match (primaries, knee) {
        (ColorPrimaries::Bt709, k) if k >= 0.96 => 0.0875,
        (ColorPrimaries::DisplayP3, k) if k >= 0.96 => 0.0865,
        (ColorPrimaries::Bt2020, k) if k >= 0.96 => 0.1115,
        (ColorPrimaries::Bt709, k) if k >= 0.90 => 0.5435,
        (ColorPrimaries::DisplayP3, k) if k >= 0.90 => 0.539,
        (ColorPrimaries::Bt2020, k) if k >= 0.90 => 0.55,
        _ => f32::INFINITY,
    }
}

/// Soft chroma compression for one set of primaries.
///
/// Below `knee` times the largest in-gamut chroma at a pixel's lightness and hue,
/// colors pass through unchanged. Above it, chroma rolls off along a rational curve
/// that's continuous in value and slope at the knee and approaches the boundary.
/// Hue and lightness are kept. Results stay within about 0.001 of the cube, so a
/// final clamp is still needed.
///
/// Cheap to construct, with no table and no allocation.
///
/// # Examples
///
/// ```
/// use zenpixels_convert::hdr::SoftCompress;
/// use zenpixels_convert::oklab::OklabSpace;
/// use zenpixels::ColorPrimaries;
///
/// let space = OklabSpace::for_primaries(ColorPrimaries::Bt709).unwrap();
/// let compress = SoftCompress::new(&space, SoftCompress::DEFAULT_KNEE);
///
/// let mut pixels = [[1.2_f32, 0.05, 0.05]]; // out-of-gamut red
/// compress.apply_strip(&mut pixels);
/// assert!(pixels[0].iter().all(|&c| (-1e-3..=1.0 + 1e-3).contains(&c)));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SoftCompress {
    space: OklabSpace,
    knee: f32,
    gate_ratio: f32,
    /// Past `1 / knee` nothing changes, so the solve needn't look further.
    scale_cap: f32,
}

impl SoftCompress {
    /// Production default knee: the fraction of the largest in-gamut chroma where
    /// the rolloff begins. From a sweep of 76 imazen-26 gain-mapped HDR images on
    /// 2026-06-23, the largest value whose corpus-p90 share of pre-clamp
    /// out-of-gamut pixels stayed under 0.1 %. Matches
    /// [`crate::HdrConfig::default`]'s `gamut_knee`.
    pub const DEFAULT_KNEE: f32 = 0.96;

    /// A compressor for `space`. `knee` is in `0.0..=1.0`, and 1.0 maps to the
    /// boundary with no rolloff.
    ///
    /// Knees under about 0.5 reach far beyond the boundary, where the solve hasn't
    /// been qualified.
    #[must_use]
    pub fn new(space: &OklabSpace, knee: f32) -> Self {
        Self {
            space: *space,
            knee,
            gate_ratio: gate_ratio(space.primaries(), knee),
            scale_cap: (1.0 / knee).clamp(1.25, 16.0),
        }
    }

    /// Compress a row of linear RGB pixels in place.
    ///
    /// Interior pixels stay bit-identical. Non-finite channels, and magnitudes above
    /// about 1e38, pass through unchanged.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        incant!(apply_strip_impl(self, rgb), [v3, neon, wasm128, scalar])
    }

    /// Compress OKLab planes in place, leaving `l` as is. For pipelines that already
    /// hold planar OKLab. Equivalent to converting to RGB and using
    /// [`apply_strip`](Self::apply_strip), up to rounding.
    ///
    /// # Errors
    ///
    /// [`BufferSize`](crate::ConvertError::BufferSize) when `a` or `b` differs in
    /// length from `l`, leaving both untouched.
    pub fn compress_oklab_planes(
        &self,
        l: &[f32],
        a: &mut [f32],
        b: &mut [f32],
    ) -> Result<(), whereat::At<crate::ConvertError>> {
        let n = l.len();
        if a.len() != n || b.len() != n {
            return Err(whereat::at!(crate::ConvertError::BufferSize {
                expected: n,
                actual: if a.len() != n { a.len() } else { b.len() },
            }));
        }
        incant!(
            compress_planes_impl(self, l, a, b),
            [v3, neon, wasm128, scalar]
        );
        Ok(())
    }

    /// Knee: the fraction of the largest in-gamut chroma where compression starts.
    #[inline]
    #[must_use]
    pub fn knee(&self) -> f32 {
        self.knee
    }
}

/// The chroma scale for each lane, and which lanes it applies to, from lightness,
/// the offset of each cube-root LMS channel from it, and the input's RGB.
///
/// Uses plain multiply and add, with no fused operations, so that every SIMD tier
/// gives identical bits.
macro_rules! chroma_scale {
    ($splat:ident, $dot:ident, $sc:expr, $ll:ident, $dl:ident, $dm:ident, $ds:ident, $r0:ident, $g0:ident, $b0:ident) => {{
        let inv = $sc.space.lms_to_rgb().rows();
        let (zero, one, three) = ($splat(0.0), $splat(1.0), $splat(3.0));
        let cap = $splat($sc.scale_cap);
        let tiny = $splat(1e-12);

        // Start on the chord from gray (s = 0, every channel L^3) to the input
        // (s = 1). Starting at s = 1 on a far-out color lands on a later crossing.
        let gray = $ll * $ll * $ll;
        let chord = |v: f32x8| -> f32x8 {
            let over = (one - gray) / (v - gray);
            let under = gray / (gray - v);
            f32x8::blend(
                v.simd_lt(zero),
                under,
                f32x8::blend(v.simd_gt(one), over, cap),
            )
        };
        let mut s = chord($r0)
            .min(chord($g0))
            .min(chord($b0))
            .max(zero)
            .min(one);

        for _ in 0..SOLVE_STEPS {
            let (xl, xm, xs) = ($ll + s * $dl, $ll + s * $dm, $ll + s * $ds);
            let (l2, m2, s2) = (xl * xl, xm * xm, xs * xs);
            let (l3, m3, s3) = (l2 * xl, m2 * xm, s2 * xs);
            let (gl, gm, gs) = (l2 * $dl, m2 * $dm, s2 * $ds);
            // Each channel's next estimate of its crossing; the nearest one binds.
            let toward = |row: &[f32; 3]| -> f32x8 {
                let v = $dot(row, l3, m3, s3);
                let grad = three * $dot(row, gl, gm, gs);
                let face = f32x8::blend(grad.simd_gt(zero), one, zero);
                f32x8::blend(grad.abs().simd_lt(tiny), cap, s + (face - v) / grad)
            };
            s = toward(&inv[0])
                .min(toward(&inv[1]))
                .min(toward(&inv[2]))
                .max(zero)
                .min(cap);
        }

        // Rational knee in units of the input's chroma, which sits at s = 1:
        // continuous in value and slope at the knee, approaching the boundary.
        let knee_s = $splat($sc.knee) * s;
        let excess = one - knee_s;
        let range = s - knee_s;
        let scaled = knee_s + range * excess / (excess + range);
        // No representable chroma at L <= 0 or L >= 1.
        let dead = $ll.simd_le(zero) | $ll.simd_ge(one);
        // Multiplying by zero gives NaN for exactly the non-finite values.
        let finite =
            ($r0 * zero).simd_eq(zero) & ($g0 * zero).simd_eq(zero) & ($b0 * zero).simd_eq(zero);
        (
            f32x8::blend(dead, zero, scaled),
            (excess.simd_gt(zero) | dead) & finite,
        )
    }};
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn apply_strip_impl(token: Token, sc: &SoftCompress, rgb: &mut [[f32; 3]]) {
    let splat = |v: f32| f32x8::splat_t(token, v);
    let dot = |row: &[f32; 3], a: f32x8, b: f32x8, c: f32x8| -> f32x8 {
        splat(row[0]) * a + splat(row[1]) * b + splat(row[2]) * c
    };
    let (m1, inv) = (sc.space.rgb_to_lms().rows(), sc.space.lms_to_rgb().rows());
    let ok = OKLAB_FROM_LMS_CBRT.rows()[0];
    let (ratio, top) = (splat(sc.gate_ratio), splat(GATE_TOP));

    let solve = |px: &mut [[f32; 3]; 8]| {
        let r0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][0]));
        let g0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][1]));
        let b0 = f32x8::from_array_t(token, core::array::from_fn(|i| px[i][2]));
        let lc = dot(&m1[0], r0, g0, b0).cbrt_midp();
        let mc = dot(&m1[1], r0, g0, b0).cbrt_midp();
        let sc2 = dot(&m1[2], r0, g0, b0).cbrt_midp();
        let ll = dot(&ok, lc, mc, sc2);
        let (dl, dm, ds) = (lc - ll, mc - ll, sc2 - ll);
        let (scaled, active) = chroma_scale!(splat, dot, sc, ll, dl, dm, ds, r0, g0, b0);
        let (xl, xm, xs) = (ll + scaled * dl, ll + scaled * dm, ll + scaled * ds);
        let (l3, m3, s3) = (xl * xl * xl, xm * xm * xm, xs * xs * xs);
        let rr = f32x8::blend(active, dot(&inv[0], l3, m3, s3), r0).to_array();
        let gg = f32x8::blend(active, dot(&inv[1], l3, m3, s3), g0).to_array();
        let bb = f32x8::blend(active, dot(&inv[2], l3, m3, s3), b0).to_array();
        for (i, out) in px.iter_mut().enumerate() {
            *out = [rr[i], gg[i], bb[i]];
        }
    };

    // Rejected pixels wait here with their position, then go through as a batch,
    // so one edge pixel doesn't cost a whole mixed group of 8.
    let mut batch = [[0.5f32; 3]; 8];
    let mut slots = [0usize; 8];
    let mut used = 0;
    let n = rgb.len();
    let mut start = 0;
    while start < n {
        let end = (start + 8).min(n);
        // Padding lanes of a short tail are 1.0, which the gate rejects, so they
        // can't produce a false all-interior result; they're never read back.
        let mut lanes = [[1.0f32; 8]; 3];
        for (i, px) in rgb[start..end].iter().enumerate() {
            lanes[0][i] = px[0];
            lanes[1][i] = px[1];
            lanes[2][i] = px[2];
        }
        let r = f32x8::from_array_t(token, lanes[0]);
        let g = f32x8::from_array_t(token, lanes[1]);
        let b = f32x8::from_array_t(token, lanes[2]);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let interior = mn.simd_gt(mx * ratio) & mx.simd_lt(top);
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
        batch[used..].fill([0.5; 3]);
        solve(&mut batch);
        for k in 0..used {
            rgb[slots[k]] = batch[k];
        }
    }
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn compress_planes_impl(token: Token, sc: &SoftCompress, l: &[f32], a: &mut [f32], b: &mut [f32]) {
    let splat = |v: f32| f32x8::splat_t(token, v);
    let dot = |row: &[f32; 3], x: f32x8, y: f32x8, z: f32x8| -> f32x8 {
        splat(row[0]) * x + splat(row[1]) * y + splat(row[2]) * z
    };
    let inv = sc.space.lms_to_rgb().rows();
    let [row_l, row_m, row_s] = *LMS_CBRT_FROM_OKLAB.rows();
    let n = l.len();
    let mut start = 0;
    while start < n {
        let end = (start + 8).min(n);
        let count = end - start;
        // Padding has L = 0, which is dead, and is never written back.
        let (mut ll_a, mut a_a, mut b_a) = ([0f32; 8], [0f32; 8], [0f32; 8]);
        ll_a[..count].copy_from_slice(&l[start..end]);
        a_a[..count].copy_from_slice(&a[start..end]);
        b_a[..count].copy_from_slice(&b[start..end]);
        let ll = f32x8::from_array_t(token, ll_a);
        let av = f32x8::from_array_t(token, a_a);
        let bv = f32x8::from_array_t(token, b_a);
        let zero = splat(0.0);
        let dl = dot(&row_l, zero, av, bv);
        let dm = dot(&row_m, zero, av, bv);
        let ds = dot(&row_s, zero, av, bv);
        let (xl, xm, xs) = (ll + dl, ll + dm, ll + ds);
        let (l3, m3, s3) = (xl * xl * xl, xm * xm * xm, xs * xs * xs);
        let r0 = dot(&inv[0], l3, m3, s3);
        let g0 = dot(&inv[1], l3, m3, s3);
        let b0 = dot(&inv[2], l3, m3, s3);
        // The same gate as for RGB: most colors in most images end here.
        let mx = r0.max(g0).max(b0);
        let mn = r0.min(g0).min(b0);
        let interior = mn.simd_gt(mx * splat(sc.gate_ratio)) & mx.simd_lt(splat(GATE_TOP));
        if count == 8 && interior.bitcast_to_i32().all_true() {
            start = end;
            continue;
        }
        let (scaled, active) = chroma_scale!(splat, dot, sc, ll, dl, dm, ds, r0, g0, b0);
        let out_a = f32x8::blend(active, av * scaled, av).to_array();
        let out_b = f32x8::blend(active, bv * scaled, bv).to_array();
        a[start..end].copy_from_slice(&out_a[..count]);
        b[start..end].copy_from_slice(&out_b[..count]);
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab::rgb_to_lms_matrix;

    const ALL: [ColorPrimaries; 3] = [
        ColorPrimaries::Bt709,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ];

    fn space(p: ColorPrimaries) -> OklabSpace {
        OklabSpace::for_primaries(p).unwrap()
    }

    /// Largest in-gamut chroma at (L, hue), by bisection.
    fn exact_boundary(space: &OklabSpace, l: f32, ca: f32, sa: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, 2.0f32);
        for _ in 0..40 {
            let mid = (lo + hi) / 2.0;
            let in_gamut = space
                .to_rgb([l, mid * ca, mid * sa])
                .iter()
                .all(|&v| (0.0..=1.0).contains(&v));
            if in_gamut { lo = mid } else { hi = mid }
        }
        lo
    }

    fn cube(n: u32) -> alloc::vec::Vec<[f32; 3]> {
        let d = (n - 1) as f32;
        let mut v = alloc::vec::Vec::new();
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    v.push([r as f32 / d, g as f32 / d, b as f32 / d]);
                }
            }
        }
        v
    }

    /// The gate must never pass a color the map would change, at each table entry.
    #[test]
    fn gated_pixels_are_past_no_knee() {
        for p in ALL {
            for knee in [0.90, 0.96] {
                let sc = SoftCompress::new(&space(p), knee);
                let mut solved_only = sc.clone();
                solved_only.gate_ratio = f32::INFINITY;
                let input = cube(65);
                let (mut gated, mut full) = (input.clone(), input.clone());
                sc.apply_strip(&mut gated);
                solved_only.apply_strip(&mut full);
                assert_eq!(gated, full, "{p:?} knee {knee}");
            }
        }
    }

    #[test]
    fn interior_colors_stay_bit_identical() {
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        let input = [[0.5f32, 0.4, 0.3], [0.2, 0.2, 0.2], [0.8, 0.7, 0.75]];
        let mut out = input;
        sc.apply_strip(&mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn out_of_gamut_lands_in_the_cube() {
        let m = crate::gamut::GamutMatrix::between(ColorPrimaries::Bt2020, ColorPrimaries::Bt709)
            .unwrap();
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        let mut v: alloc::vec::Vec<[f32; 3]> = cube(33).into_iter().map(|c| m.apply(c)).collect();
        sc.apply_strip(&mut v);
        for px in v {
            for c in px {
                assert!((-1e-3..=1.0 + 1e-3).contains(&c), "{px:?}");
            }
        }
    }

    #[test]
    fn hue_and_lightness_are_kept() {
        let sp = space(ColorPrimaries::Bt709);
        let m = crate::gamut::GamutMatrix::between(ColorPrimaries::Bt2020, ColorPrimaries::Bt709)
            .unwrap();
        let sc = SoftCompress::new(&sp, 0.96);
        for c in cube(17) {
            let x = m.apply(c);
            let [l0, a0, b0] = sp.from_rgb(x);
            if !(0.02..0.98).contains(&l0) || a0.hypot(b0) < 0.01 {
                continue;
            }
            let mut out = [x];
            sc.apply_strip(&mut out);
            let [l1, a1, b1] = sp.from_rgb(out[0]);
            assert!((l1 - l0).abs() < 1e-3, "{c:?}: L {l0} to {l1}");
            if a1.hypot(b1) > 1e-3 {
                let cross = (a0 * b1 - b0 * a1) / (a0.hypot(b0) * a1.hypot(b1));
                assert!(cross.abs() < 2e-3, "{c:?}: hue moved, sin {cross}");
            }
        }
    }

    /// Solve and bisection agree where the boundary is crossed once.
    #[test]
    fn scale_matches_bisection() {
        let sp = space(ColorPrimaries::Bt709);
        for knee in [0.5f32, 0.75, 0.96, 1.0] {
            let sc = SoftCompress::new(&sp, knee);
            for li in 2..=8 {
                let l = li as f32 / 10.0;
                for h in (0..360).step_by(15) {
                    // Dark blues re-enter the gamut, so bisection isn't a reference there.
                    if (240..=285).contains(&h) {
                        continue;
                    }
                    let (sa, ca) = (h as f32).to_radians().sin_cos();
                    let cmax = exact_boundary(&sp, l, ca, sa);
                    let c = cmax * 1.1;
                    let (mut a, mut b) = ([c * ca], [c * sa]);
                    sc.compress_oklab_planes(&[l], &mut a, &mut b).unwrap();
                    let k = knee * cmax;
                    let (range, excess) = (cmax - k, c - k);
                    let want = if range <= 0.0 {
                        cmax
                    } else {
                        k + range * excess / (excess + range)
                    };
                    let got = a[0].hypot(b[0]);
                    assert!(
                        (got - want).abs() < 2e-3,
                        "knee {knee} L {l} hue {h}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn hue_zero_has_no_seam() {
        let sp = space(ColorPrimaries::Bt709);
        let sc = SoftCompress::new(&sp, 0.96);
        for h in [-1e-5f32, -1e-7, -1e-9, 0.0, 1e-9, 1e-7] {
            let (mut a, mut b) = ([0.4 * h.cos()], [0.4 * h.sin()]);
            sc.compress_oklab_planes(&[0.5], &mut a, &mut b).unwrap();
            assert!((a[0] - 0.2027).abs() < 3e-3, "hue {h:e}: a {}", a[0]);
        }
    }

    #[test]
    fn planes_match_the_rgb_path() {
        let sp = space(ColorPrimaries::Bt709);
        let m =
            crate::gamut::GamutMatrix::between(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709)
                .unwrap();
        let sc = SoftCompress::new(&sp, 0.96);
        let input: alloc::vec::Vec<[f32; 3]> = cube(17).into_iter().map(|c| m.apply(c)).collect();
        let mut rgb = input.clone();
        sc.apply_strip(&mut rgb);
        let lab: alloc::vec::Vec<[f32; 3]> = input.iter().map(|&c| sp.from_rgb(c)).collect();
        let l: alloc::vec::Vec<f32> = lab.iter().map(|v| v[0]).collect();
        let mut a: alloc::vec::Vec<f32> = lab.iter().map(|v| v[1]).collect();
        let mut b: alloc::vec::Vec<f32> = lab.iter().map(|v| v[2]).collect();
        sc.compress_oklab_planes(&l, &mut a, &mut b).unwrap();
        for i in 0..input.len() {
            let want = sp.from_rgb(rgb[i]);
            assert!(
                (a[i] - want[1]).abs() < 1e-3 && (b[i] - want[2]).abs() < 1e-3,
                "{:?}: {} {} vs {want:?}",
                input[i],
                a[i],
                b[i]
            );
        }
    }

    /// Any row split gives the same bits as one call.
    #[test]
    fn row_length_does_not_change_results() {
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        let m = crate::gamut::GamutMatrix::between(ColorPrimaries::Bt2020, ColorPrimaries::Bt709)
            .unwrap();
        let input: alloc::vec::Vec<[f32; 3]> = cube(9).into_iter().map(|c| m.apply(c)).collect();
        let mut whole = input.clone();
        sc.apply_strip(&mut whole);
        for width in [1usize, 2, 3, 7, 8, 9, 15, 16, 17] {
            let mut parts = input.clone();
            for chunk in parts.chunks_mut(width) {
                sc.apply_strip(chunk);
            }
            assert_eq!(parts, whole, "width {width}");
        }
    }

    #[test]
    fn non_finite_and_empty_inputs() {
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        sc.apply_strip(&mut []);
        let mut px = [[f32::NAN, 0.5, 0.5], [f32::INFINITY, 0.5, 0.5]];
        sc.apply_strip(&mut px);
        assert!(px[0][0].is_nan());
        assert!(px[1][0].is_infinite());
    }

    #[test]
    fn plane_length_mismatch_is_an_error() {
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        let (mut a, mut b) = ([0.0f32; 2], [0.0f32; 3]);
        assert!(sc.compress_oklab_planes(&[0.5; 3], &mut a, &mut b).is_err());
    }

    #[test]
    fn extremes_become_achromatic() {
        let sc = SoftCompress::new(&space(ColorPrimaries::Bt709), 0.96);
        let (mut a, mut b) = ([0.1f32, 0.1], [0.1f32, 0.1]);
        sc.compress_oklab_planes(&[0.0, 1.0], &mut a, &mut b)
            .unwrap();
        assert_eq!((a, b), ([0.0, 0.0], [0.0, 0.0]));
    }

    #[test]
    fn gate_table_covers_every_primaries_with_an_lms_matrix() {
        for p in ALL {
            assert!(rgb_to_lms_matrix(p).is_some());
            assert!(gate_ratio(p, 0.96).is_finite());
            assert!(gate_ratio(p, 0.5).is_infinite());
        }
    }
}
