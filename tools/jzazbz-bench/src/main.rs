//! JzAzBz vs XYB forward-conversion speed: linear BT.709 RGB f32 → 3 planes.
//!
//! Neither transform is in the crate's API (zenpixels has Oklab only), so the
//! kernels live here. Every SIMD arm is checked against an f64 reference
//! before any timing runs; the max absolute error per output channel prints
//! beside the timings so speed and accuracy are read together.
//!
//! XYB is libjxl's opsin transform (3×3 + bias, cbrt, X/Y/B mix). JzAzBz is
//! Safdar et al. 2017 on absolute luminance: RGB → XYZ → X'Y'Z' → LMS folds
//! into one 3×3 (with the white luminance and the /10000 scale), then a PQ
//! curve with exponent `p = 1.7·2523/32` per channel, a 3×3 to Iz/az/bz and
//! the Jz rational.
//!
//! The PQ curve variants:
//! - `pow_midp` / `pow_lowp`: the exact formula, two `pow` per channel.
//! - `ratPQ`: a single-range rational of degree (P,Q) in `a = x^(1/4)`,
//!   fitted by polyfit's rational fitter + f32 coefficient search
//!   (`tools/fit-jzazbz-pq`), `a = rsqrt(rsqrt(x))`, `P · recip(Q)`.
//! - `rat55 sqrt/div`: the (5,5) fit with exact `sqrt` and division.
//!
//! Kernels are `#[attune]` dispatch families; the per-pixel math is generic
//! over the magetypes backend and closure-free so it inlines into each
//! tier's feature context (a closure body is compiled without the tier's
//! target features and turns every vector op into a call).
//!
//! Planar in, planar out. Run:
//! `cargo bench -p zenpixels-convert --bench bench_jzazbz_vs_xyb`

use archmage::prelude::*;
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;
use zenbench::prelude::*;

// ── Constants ──────────────────────────────────────────────────────────

/// Linear BT.709/sRGB → XYZ (D65), Y normalised to 1 at white.
const RGB709_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.412_390_799_265_959_5,
        0.357_584_339_383_878,
        0.180_480_788_401_834_3,
    ],
    [
        0.212_639_005_871_510_36,
        0.715_168_678_767_756,
        0.072_192_315_360_733_71,
    ],
    [
        0.019_330_818_715_591_85,
        0.119_194_779_794_625_99,
        0.950_532_152_249_660_6,
    ],
];

/// libjxl opsin absorbance matrix (rows: L, M, S).
const OPSIN: [[f64; 3]; 3] = [
    [0.30, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [
        0.243_422_689_245_478_2,
        0.204_767_444_244_968_2,
        0.551_809_866_509_553_6,
    ],
];
const OPSIN_BIAS: f64 = 0.003_793_073_255_275_449_3;

// JzAzBz (Safdar et al. 2017).
const JZ_B: f64 = 1.15;
const JZ_G: f64 = 0.66;
const JZ_XYZ_TO_LMS: [[f64; 3]; 3] = [
    [0.414_789_72, 0.579_999, 0.014_648],
    [-0.201_51, 1.120_649, 0.053_100_8],
    [-0.016_600_8, 0.264_8, 0.668_479_9],
];
const JZ_LMS_TO_IAB: [[f64; 3]; 3] = [
    [0.5, 0.5, 0.0],
    [3.524_000, -4.066_708, 0.542_708],
    [0.199_076, 1.096_799, -1.295_875],
];
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 128.0;
const PQ_C3: f64 = 2392.0 / 128.0;
const PQ_N: f64 = 2610.0 / 16384.0;
const JZ_P: f64 = 1.7 * 2523.0 / 32.0;
const JZ_D: f64 = -0.56;
const JZ_D0: f64 = 1.629_549_953_282_156_6e-11;

/// BT.2408 reference white, cd/m², for SDR input `1.0`.
const WHITE_NITS: f64 = 203.0;

// `tools/fit-jzazbz-pq` (polyfit rational-polynomial-fitting@4de8e99):
// f(x) = ((c1 + c2 x^n)/(1 + c3 x^n))^p as P(a)/Q(a), a = x^(1/4),
// x ∈ [0, 1] (0–10000 nits); Q is monic. f32 max abs error, 200,001
// a-uniform samples: (4,3) 5.000e-5, (5,4) 8.414e-6, (5,5) 1.140e-6,
// (6,6) 1.788e-7.
const JZ_PQ_43_P: [f32; 5] = [
    -7.310801e-7,
    0.00050432363,
    -0.05617855,
    2.250443,
    0.08440651,
];
const JZ_PQ_43_Q: [f32; 4] = [0.014620511, 0.17437813, 1.0901736, 1.0];
const JZ_PQ_54_P: [f32; 6] = [
    8.477304e-10,
    -1.7624754e-6,
    0.00049302104,
    -0.04673232,
    2.2125535,
    0.09411853,
];
const JZ_PQ_54_Q: [f32; 5] = [0.00014361832, 0.014249746, 0.18023571, 1.0657883, 1.0];
const JZ_PQ_55_P: [f32; 6] = [
    9.004721e-10,
    -1.6633309e-6,
    0.0004974915,
    -0.05762242,
    4.5971103,
    2.8213296,
];
const JZ_PQ_55_Q: [f32; 6] = [
    0.0007898856,
    0.030357914,
    0.43749636,
    2.3919218,
    3.5007508,
    1.0,
];
const JZ_PQ_66_P: [f32; 7] = [
    8.872266e-12,
    -2.5310806e-8,
    1.1608218e-5,
    -0.001821554,
    0.24173436,
    5.7214913,
    2.854904,
];
const JZ_PQ_66_Q: [f32; 7] = [
    7.284148e-5,
    0.0029533722,
    0.06836599,
    0.6760378,
    3.0886586,
    3.980232,
    1.0,
];

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

/// Linear BT.709 RGB (1.0 = `white_nits`) → PQ-domain LMS (1.0 = 10000 nits).
fn jz_rgb_to_lms(white_nits: f64) -> [[f64; 3]; 3] {
    let modify = [
        [JZ_B, 0.0, -(JZ_B - 1.0)],
        [-(JZ_G - 1.0), JZ_G, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut m = matmul(&JZ_XYZ_TO_LMS, &matmul(&modify, &RGB709_TO_XYZ));
    for row in &mut m {
        for v in row {
            *v *= white_nits / 10000.0;
        }
    }
    m
}

fn to_f32(m: &[[f64; 3]; 3]) -> [[f32; 3]; 3] {
    m.map(|r| r.map(|v| v as f32))
}

// ── f64 references ─────────────────────────────────────────────────────

fn xyb_ref(rgb: [f64; 3]) -> [f64; 3] {
    let cb = OPSIN_BIAS.cbrt();
    let c = OPSIN.map(|r| {
        let mixed = (r[0] * rgb[0] + r[1] * rgb[1] + r[2] * rgb[2] + OPSIN_BIAS).max(0.0);
        mixed.cbrt() - cb
    });
    [0.5 * (c[0] - c[1]), 0.5 * (c[0] + c[1]), c[2]]
}

fn jz_pq_ref(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0).powf(PQ_N);
    ((PQ_C1 + PQ_C2 * t) / (1.0 + PQ_C3 * t)).powf(JZ_P)
}

fn jz_ref(rgb: [f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    let lms = m.map(|r| jz_pq_ref(r[0] * rgb[0] + r[1] * rgb[1] + r[2] * rgb[2]));
    let [iz, az, bz] = JZ_LMS_TO_IAB.map(|r| r[0] * lms[0] + r[1] * lms[1] + r[2] * lms[2]);
    let jz = (1.0 + JZ_D) * iz / (1.0 + JZ_D * iz) - JZ_D0;
    [jz, az, bz]
}

// ── Scalar f32 baselines (libm cbrt / powf) ────────────────────────────

fn xyb_scalar_std(src: [&[f32]; 3], dst: [&mut [f32]; 3]) {
    let m = to_f32(&OPSIN);
    let bias = OPSIN_BIAS as f32;
    let cb = bias.cbrt();
    let [d0, d1, d2] = dst;
    for i in 0..src[0].len() {
        let (r, g, b) = (src[0][i], src[1][i], src[2][i]);
        let c = m.map(|row| {
            (row[0] * r + row[1] * g + row[2] * b + bias)
                .max(0.0)
                .cbrt()
                - cb
        });
        d0[i] = 0.5 * (c[0] - c[1]);
        d1[i] = 0.5 * (c[0] + c[1]);
        d2[i] = c[2];
    }
}

fn jz_scalar_std(src: [&[f32]; 3], dst: [&mut [f32]; 3], m: &[[f32; 3]; 3]) {
    let (c1, c2, c3, n, p) = (
        PQ_C1 as f32,
        PQ_C2 as f32,
        PQ_C3 as f32,
        PQ_N as f32,
        JZ_P as f32,
    );
    let pq = |x: f32| {
        let t = x.clamp(0.0, 1.0).powf(n);
        ((c1 + c2 * t) / (1.0 + c3 * t)).powf(p)
    };
    let [d0, d1, d2] = dst;
    for i in 0..src[0].len() {
        let (r, g, b) = (src[0][i], src[1][i], src[2][i]);
        let [l, mm, s] = m.map(|row| pq(row[0] * r + row[1] * g + row[2] * b));
        let iz = 0.5 * (l + mm);
        d1[i] = 3.524 * l - 4.066708 * mm + 0.542708 * s;
        d2[i] = 0.199076 * l + 1.096799 * mm - 1.295875 * s;
        d0[i] = 0.44 * iz / (1.0 - 0.56 * iz) - JZ_D0 as f32;
    }
}

// ── SIMD kernels ───────────────────────────────────────────────────────

type V<T> = f32x8<T>;
type Planes<'a> = [&'a [f32]; 3];
type PlanesMut<'a> = [&'a mut [f32]; 3];

/// One 8-pixel step of a planar 3→3 transform. `m` is the kernel's input
/// matrix (opsin for XYB, folded RGB→LMS for JzAzBz).
trait Kernel {
    fn px<T: F32x8Convert>(t: T, m: &[[f32; 3]; 3], r: V<T>, g: V<T>, b: V<T>) -> [V<T>; 3];
}

/// A per-channel nonlinearity.
trait Curve {
    fn f<T: F32x8Convert>(t: T, x: V<T>) -> V<T>;
}

#[inline(always)]
fn dot3<T: F32x8Convert>(t: T, row: &[f32; 3], r: V<T>, g: V<T>, b: V<T>) -> V<T> {
    b.mul_add(
        V::splat_t(t, row[2]),
        g.mul_add(V::splat_t(t, row[1]), r * V::splat_t(t, row[0])),
    )
}

#[inline(always)]
fn horner<T: F32x8Convert, const N: usize>(t: T, x: V<T>, c: &[f32; N]) -> V<T> {
    let mut y = V::splat_t(t, c[N - 1]);
    let mut i = N - 1;
    while i > 0 {
        i -= 1;
        y = y.mul_add(x, V::splat_t(t, c[i]));
    }
    y
}

/// Horner for a monic polynomial (`c[N-1] == 1`): skips the leading multiply.
#[inline(always)]
fn horner_monic<T: F32x8Convert, const N: usize>(t: T, x: V<T>, c: &[f32; N]) -> V<T> {
    let mut y = x + V::splat_t(t, c[N - 2]);
    let mut i = N - 2;
    while i > 0 {
        i -= 1;
        y = y.mul_add(x, V::splat_t(t, c[i]));
    }
    y
}

struct Xyb<C>(core::marker::PhantomData<C>);
impl<C: Curve> Kernel for Xyb<C> {
    #[inline(always)]
    fn px<T: F32x8Convert>(t: T, m: &[[f32; 3]; 3], r: V<T>, g: V<T>, b: V<T>) -> [V<T>; 3] {
        let bias = V::splat_t(t, OPSIN_BIAS as f32);
        let cb = V::splat_t(t, (OPSIN_BIAS as f32).cbrt());
        let zero = V::zero_t(t);
        let half = V::splat_t(t, 0.5);
        let l = C::f(t, (dot3(t, &m[0], r, g, b) + bias).max(zero)) - cb;
        let mm = C::f(t, (dot3(t, &m[1], r, g, b) + bias).max(zero)) - cb;
        let s = C::f(t, (dot3(t, &m[2], r, g, b) + bias).max(zero)) - cb;
        [(l - mm) * half, (l + mm) * half, s]
    }
}

struct Jz<C>(core::marker::PhantomData<C>);
impl<C: Curve> Kernel for Jz<C> {
    #[inline(always)]
    fn px<T: F32x8Convert>(t: T, m: &[[f32; 3]; 3], r: V<T>, g: V<T>, b: V<T>) -> [V<T>; 3] {
        let zero = V::zero_t(t);
        let one = V::splat_t(t, 1.0);
        let l = C::f(t, dot3(t, &m[0], r, g, b).max(zero).min(one));
        let mm = C::f(t, dot3(t, &m[1], r, g, b).max(zero).min(one));
        let s = C::f(t, dot3(t, &m[2], r, g, b).max(zero).min(one));
        let iz = (l + mm) * V::splat_t(t, 0.5);
        let az = s.mul_add(
            V::splat_t(t, 0.542708),
            mm.mul_add(V::splat_t(t, -4.066708), l * V::splat_t(t, 3.524)),
        );
        let bz = s.mul_add(
            V::splat_t(t, -1.295875),
            mm.mul_add(V::splat_t(t, 1.096799), l * V::splat_t(t, 0.199076)),
        );
        let jz = iz * V::splat_t(t, 0.44) / iz.mul_add(V::splat_t(t, -0.56), one)
            - V::splat_t(t, JZ_D0 as f32);
        [jz, az, bz]
    }
}

struct CbrtLowp;
impl Curve for CbrtLowp {
    #[inline(always)]
    fn f<T: F32x8Convert>(_t: T, x: V<T>) -> V<T> {
        x.cbrt_lowp()
    }
}
struct CbrtMidp;
impl Curve for CbrtMidp {
    #[inline(always)]
    fn f<T: F32x8Convert>(_t: T, x: V<T>) -> V<T> {
        x.cbrt_midp()
    }
}

// pow_*_unchecked is undefined at 0; MIN_POSITIVE^n lands 5e-4 relative
// above f(0) = 3.7e-11, i.e. ~2e-14 absolute.
struct PqPowMidp;
impl Curve for PqPowMidp {
    #[inline(always)]
    fn f<T: F32x8Convert>(t: T, x: V<T>) -> V<T> {
        let tn = x
            .max(V::splat_t(t, f32::MIN_POSITIVE))
            .pow_midp_unchecked(PQ_N as f32);
        let num = tn.mul_add(V::splat_t(t, PQ_C2 as f32), V::splat_t(t, PQ_C1 as f32));
        let den = tn.mul_add(V::splat_t(t, PQ_C3 as f32), V::splat_t(t, 1.0));
        (num / den).pow_midp_unchecked(JZ_P as f32)
    }
}
struct PqPowLowp;
impl Curve for PqPowLowp {
    #[inline(always)]
    fn f<T: F32x8Convert>(t: T, x: V<T>) -> V<T> {
        let tn = x
            .max(V::splat_t(t, f32::MIN_POSITIVE))
            .pow_lowp_unchecked(PQ_N as f32);
        let num = tn.mul_add(V::splat_t(t, PQ_C2 as f32), V::splat_t(t, PQ_C1 as f32));
        let den = tn.mul_add(V::splat_t(t, PQ_C3 as f32), V::splat_t(t, 1.0));
        (num / den).pow_lowp_unchecked(JZ_P as f32)
    }
}
// ── Rational-fit JzAzBz: root method × division strategy ──────────────

/// A fitted (P, Q) pair in `a = x^(1/4)`.
trait Fit {
    fn p<T: F32x8Convert>(t: T, a: V<T>) -> V<T>;
    fn q<T: F32x8Convert>(t: T, a: V<T>) -> V<T>;
}
macro_rules! fit {
    ($name:ident, $p:ident, $q:ident) => {
        struct $name;
        impl Fit for $name {
            #[inline(always)]
            fn p<T: F32x8Convert>(t: T, a: V<T>) -> V<T> {
                horner(t, a, &$p)
            }
            #[inline(always)]
            fn q<T: F32x8Convert>(t: T, a: V<T>) -> V<T> {
                horner_monic(t, a, &$q)
            }
        }
    };
}
fit!(Fit43, JZ_PQ_43_P, JZ_PQ_43_Q);
fit!(Fit54, JZ_PQ_54_P, JZ_PQ_54_Q);
fit!(Fit55, JZ_PQ_55_P, JZ_PQ_55_Q);
fit!(Fit66, JZ_PQ_66_P, JZ_PQ_66_Q);

/// `x^(1/4)` for `x ∈ [0, 1]`.
trait Root4 {
    fn root4<T: F32x8Convert>(t: T, x: V<T>) -> V<T>;
}
/// Two IEEE `sqrt` (divider port).
struct RootSqrt;
impl Root4 for RootSqrt {
    #[inline(always)]
    fn root4<T: F32x8Convert>(_t: T, x: V<T>) -> V<T> {
        x.sqrt().sqrt()
    }
}
/// magetypes `rsqrt()` working tier twice (Newton + IEEE rail rescue).
struct RootRsqrt;
impl Root4 for RootRsqrt {
    #[inline(always)]
    fn root4<T: F32x8Convert>(_t: T, x: V<T>) -> V<T> {
        // rsqrt(0) = +inf, rsqrt(+inf) = 0: x = 0 maps to a = 0 exactly.
        x.rsqrt().rsqrt()
    }
}
/// Raw `rsqrt_approx` + one Newton step, no rails: `sqrt(v) = v·rsqrt(v)`,
/// twice. `x` is floored at 1e-30 (a = 3.2e-8, f moves ~1e-13 from f(0)) so
/// the estimate never sees 0.
struct RootNewton;
#[inline(always)]
fn sqrt_nr<T: F32x8Convert>(t: T, v: V<T>) -> V<T> {
    let r = v.rsqrt_approx();
    // r' = r·(1.5 − 0.5·v·r²); sqrt = v·r'
    let vr = v * r;
    let half_vr = vr * V::splat_t(t, 0.5);
    half_vr.mul_add(-r, V::splat_t(t, 1.5)) * vr
}
impl Root4 for RootNewton {
    #[inline(always)]
    fn root4<T: F32x8Convert>(t: T, x: V<T>) -> V<T> {
        sqrt_nr(t, sqrt_nr(t, x.max(V::splat_t(t, 1e-30))))
    }
}

/// Division strategy for the three P/Q quotients plus the Jz rational.
#[derive(Clone, Copy)]
enum DivMode {
    /// Four IEEE divisions.
    Div,
    /// Four magetypes `recip()` (Newton + rail rescue).
    Recip,
    /// Four raw `rcp_approx` + one Newton step each, no rails (all
    /// denominators are positive and bounded away from 0 and ∞).
    RcpNewton,
    /// One IEEE division of the product of all four denominators; each
    /// quotient is recovered by multiplying out the other three.
    Shared,
}
trait Div {
    const MODE: DivMode;
}
macro_rules! div_mode {
    ($name:ident, $mode:ident) => {
        struct $name;
        impl Div for $name {
            const MODE: DivMode = DivMode::$mode;
        }
    };
}
div_mode!(DivIeee, Div);
div_mode!(DivRecip, Recip);
div_mode!(DivRcpNewton, RcpNewton);
div_mode!(DivShared, Shared);

#[inline(always)]
fn rcp_nr<T: F32x8Convert>(t: T, d: V<T>) -> V<T> {
    let r = d.rcp_approx();
    // r' = r·(2 − d·r)
    r * d.mul_add(-r, V::splat_t(t, 2.0))
}

struct JzRat<F, R, D>(core::marker::PhantomData<(F, R, D)>);
impl<F: Fit, R: Root4, D: Div> Kernel for JzRat<F, R, D> {
    #[inline(always)]
    fn px<T: F32x8Convert>(t: T, m: &[[f32; 3]; 3], r: V<T>, g: V<T>, b: V<T>) -> [V<T>; 3] {
        let zero = V::zero_t(t);
        let one = V::splat_t(t, 1.0);
        let al = R::root4(t, dot3(t, &m[0], r, g, b).max(zero).min(one));
        let am = R::root4(t, dot3(t, &m[1], r, g, b).max(zero).min(one));
        let as_ = R::root4(t, dot3(t, &m[2], r, g, b).max(zero).min(one));
        let (pl, ql) = (F::p(t, al), F::q(t, al));
        let (pm, qm) = (F::p(t, am), F::q(t, am));
        let (ps, qs) = (F::p(t, as_), F::q(t, as_));

        // Iz = (L' + M')/2 = (pl·qm + pm·ql) / (2·ql·qm); Jz = 0.44·Iz/(1 − 0.56·Iz).
        let (l, mm, s, jz) = match D::MODE {
            DivMode::Shared => {
                let qlm = ql * qm;
                let qlms = qlm * qs;
                // 2·qlm·Iz numerator; Jz = 0.44·n / (2·qlm − 0.56·n).
                let n = pl.mul_add(qm, pm * ql);
                let dj = n.mul_add(V::splat_t(t, -0.56), qlm + qlm);
                let inv = one / (qlms * dj);
                let inv_dj = inv * qlms; // 1/dj
                let inv_qlms = inv * dj; // 1/(ql·qm·qs)
                let inv_ql = inv_qlms * (qm * qs);
                let inv_qm = inv_qlms * (ql * qs);
                let inv_qs = inv_qlms * qlm;
                (
                    pl * inv_ql,
                    pm * inv_qm,
                    ps * inv_qs,
                    n * V::splat_t(t, 0.44) * inv_dj,
                )
            }
            mode => {
                let (l, mm, s) = match mode {
                    DivMode::Div => (pl / ql, pm / qm, ps / qs),
                    DivMode::Recip => (pl * ql.recip(), pm * qm.recip(), ps * qs.recip()),
                    _ => (pl * rcp_nr(t, ql), pm * rcp_nr(t, qm), ps * rcp_nr(t, qs)),
                };
                let iz = (l + mm) * V::splat_t(t, 0.5);
                let dj = iz.mul_add(V::splat_t(t, -0.56), one);
                let nj = iz * V::splat_t(t, 0.44);
                let jz = match mode {
                    DivMode::Div => nj / dj,
                    DivMode::Recip => nj * dj.recip(),
                    _ => nj * rcp_nr(t, dj),
                };
                (l, mm, s, jz)
            }
        };
        let az = s.mul_add(
            V::splat_t(t, 0.542708),
            mm.mul_add(V::splat_t(t, -4.066708), l * V::splat_t(t, 3.524)),
        );
        let bz = s.mul_add(
            V::splat_t(t, -1.295875),
            mm.mul_add(V::splat_t(t, 1.096799), l * V::splat_t(t, 0.199076)),
        );
        [jz - V::splat_t(t, JZ_D0 as f32), az, bz]
    }
}

/// Planar driver: full 8-lane chunks, then a zero-padded tail chunk so the
/// tail takes the same arithmetic as the body.
#[inline(always)]
fn planar<K: Kernel, T: F32x8Convert>(t: T, m: &[[f32; 3]; 3], src: Planes, dst: PlanesMut) {
    let n = src[0].len();
    let [d0, d1, d2] = dst;
    let (r8, rt) = src[0].as_chunks::<8>();
    let (g8, gt) = src[1].as_chunks::<8>();
    let (b8, bt) = src[2].as_chunks::<8>();
    let (o0, o0t) = d0[..n].as_chunks_mut::<8>();
    let (o1, o1t) = d1[..n].as_chunks_mut::<8>();
    let (o2, o2t) = d2[..n].as_chunks_mut::<8>();
    for (((((r, g), b), x), y), z) in r8
        .iter()
        .zip(g8)
        .zip(b8)
        .zip(o0.iter_mut())
        .zip(o1.iter_mut())
        .zip(o2.iter_mut())
    {
        let [a, bb, c] = K::px(t, m, V::load_t(t, r), V::load_t(t, g), V::load_t(t, b));
        a.store(x);
        bb.store(y);
        c.store(z);
    }
    if !rt.is_empty() {
        let mut pr = [0.0f32; 8];
        let mut pg = [0.0f32; 8];
        let mut pb = [0.0f32; 8];
        pr[..rt.len()].copy_from_slice(rt);
        pg[..gt.len()].copy_from_slice(gt);
        pb[..bt.len()].copy_from_slice(bt);
        let [a, bb, c] = K::px(
            t,
            m,
            V::load_t(t, &pr),
            V::load_t(t, &pg),
            V::load_t(t, &pb),
        );
        let k = o0t.len();
        o0t.copy_from_slice(&a.to_array()[..k]);
        o1t.copy_from_slice(&bb.to_array()[..k]);
        o2t.copy_from_slice(&c.to_array()[..k]);
    }
}

macro_rules! entry {
    ($name:ident, $k:ty) => {
        #[attune(dispatch)]
        fn $name(m: &[[f32; 3]; 3], src: Planes, dst: PlanesMut) {
            planar::<$k, _>(Token::from_context(), m, src, dst)
        }
    };
}
entry!(xyb_cbrt_lowp, Xyb<CbrtLowp>);
entry!(xyb_cbrt_midp, Xyb<CbrtMidp>);
entry!(jz_pow_midp, Jz<PqPowMidp>);
entry!(jz_pow_lowp, Jz<PqPowLowp>);
entry!(jz_r55_sqrt_div, JzRat<Fit55, RootSqrt, DivIeee>);
entry!(jz_r55_rsqrt_recip, JzRat<Fit55, RootRsqrt, DivRecip>);
entry!(jz_r55_newton_rcpnr, JzRat<Fit55, RootNewton, DivRcpNewton>);
entry!(jz_r55_sqrt_shared, JzRat<Fit55, RootSqrt, DivShared>);
entry!(jz_r55_newton_div, JzRat<Fit55, RootNewton, DivIeee>);
entry!(jz_r55_newton_shared, JzRat<Fit55, RootNewton, DivShared>);
entry!(jz_r43_newton_shared, JzRat<Fit43, RootNewton, DivShared>);
entry!(jz_r54_newton_shared, JzRat<Fit54, RootNewton, DivShared>);
entry!(jz_r66_newton_shared, JzRat<Fit66, RootNewton, DivShared>);
entry!(jz_r66_sqrt_div, JzRat<Fit66, RootSqrt, DivIeee>);
entry!(jz_r43_sqrt_div, JzRat<Fit43, RootSqrt, DivIeee>);
entry!(jz_r54_sqrt_div, JzRat<Fit54, RootSqrt, DivIeee>);

type KernelFn = fn(&[[f32; 3]; 3], Planes, PlanesMut);

#[derive(Clone, Copy, PartialEq)]
enum Space {
    Xyb,
    Jz,
}

fn variants() -> Vec<(&'static str, Space, KernelFn)> {
    vec![
        ("xyb cbrt_lowp", Space::Xyb, xyb_cbrt_lowp),
        ("xyb cbrt_midp", Space::Xyb, xyb_cbrt_midp),
        ("xyb scalar std", Space::Xyb, |_, s, d| xyb_scalar_std(s, d)),
        ("jz r43 sqrt+div", Space::Jz, jz_r43_sqrt_div),
        ("jz r54 sqrt+div", Space::Jz, jz_r54_sqrt_div),
        ("jz r55 sqrt+div", Space::Jz, jz_r55_sqrt_div),
        ("jz r55 rsqrt+recip", Space::Jz, jz_r55_rsqrt_recip),
        ("jz r55 nr+rcpnr", Space::Jz, jz_r55_newton_rcpnr),
        ("jz r55 sqrt+1div", Space::Jz, jz_r55_sqrt_shared),
        ("jz r55 nr+div", Space::Jz, jz_r55_newton_div),
        ("jz r55 nr+1div", Space::Jz, jz_r55_newton_shared),
        ("jz r43 nr+1div", Space::Jz, jz_r43_newton_shared),
        ("jz r54 nr+1div", Space::Jz, jz_r54_newton_shared),
        ("jz r66 nr+1div", Space::Jz, jz_r66_newton_shared),
        ("jz r66 sqrt+div", Space::Jz, jz_r66_sqrt_div),
        ("jz pow_lowp", Space::Jz, jz_pow_lowp),
        ("jz pow_midp", Space::Jz, jz_pow_midp),
        ("jz scalar powf", Space::Jz, |m, s, d| {
            jz_scalar_std(s, d, m)
        }),
    ]
}

// ── Inputs + accuracy gate ─────────────────────────────────────────────

fn planes(n: usize, scale: f32, seed: u64) -> [Vec<f32>; 3] {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * scale
    };
    let mut p = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        for c in &mut p {
            c[i] = next();
        }
    }
    // Pin the corners: black, white, primaries, the 10000-nit ceiling.
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [scale, scale, scale],
    ];
    for (i, c) in corners.iter().enumerate().take(n) {
        for ch in 0..3 {
            p[ch][i] = c[ch];
        }
    }
    p
}

/// Max absolute error per output channel vs the f64 reference.
fn max_err(space: Space, f: KernelFn, src: &[Vec<f32>; 3], mjz: &[[f64; 3]; 3]) -> [f64; 3] {
    let n = src[0].len();
    let mut out = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let m = match space {
        Space::Xyb => to_f32(&OPSIN),
        Space::Jz => to_f32(mjz),
    };
    {
        let [a, b, c] = &mut out;
        f(&m, [&src[0], &src[1], &src[2]], [a, b, c]);
    }
    let mut e = [0.0f64; 3];
    for i in 0..n {
        let rgb = [src[0][i] as f64, src[1][i] as f64, src[2][i] as f64];
        let r = match space {
            Space::Xyb => xyb_ref(rgb),
            Space::Jz => jz_ref(rgb, mjz),
        };
        for ch in 0..3 {
            e[ch] = e[ch].max((out[ch][i] as f64 - r[ch]).abs());
        }
    }
    e
}

fn accuracy_report() {
    let mjz = jz_rgb_to_lms(WHITE_NITS);
    let n = 1 << 20;
    // 1023 = 8·127 + 7 exercises the padded tail.
    let sdr = planes(n + 7, 1.0, 1);
    let hdr = planes(n + 7, (10000.0 / WHITE_NITS) as f32, 2);
    eprintln!("# max |err| vs f64 reference, white = {WHITE_NITS} nits");
    eprintln!("#   SDR: rgb ∈ [0,1]; HDR: rgb ∈ [0, 10000/{WHITE_NITS}] (0–10000 nits)");
    // Scale anchors: errors are in each space's own units.
    let xw = xyb_ref([1.0; 3]);
    let jw = jz_ref([1.0; 3], &mjz);
    let jr = jz_ref([1.0, 0.0, 0.0], &mjz);
    eprintln!(
        "#   scale: SDR white XYB Y = {:.4}, Jz = {:.4}; SDR red az = {:.4}, bz = {:.4}",
        xw[1], jw[0], jr[1], jr[2]
    );
    for (name, space, f) in variants() {
        let es = max_err(space, f, &sdr, &mjz);
        let eh = max_err(space, f, &hdr, &mjz);
        eprintln!(
            "#   {name:<18} SDR [{:.2e} {:.2e} {:.2e}]  HDR [{:.2e} {:.2e} {:.2e}]",
            es[0], es[1], es[2], eh[0], eh[1], eh[2]
        );
        // Gate: every arm must be a real implementation of its transform.
        // `pow_lowp` is the documented exception: x^134 amplifies the lowp
        // log2 error ~134×, so it is timed for reference but never usable.
        if name == "jz pow_lowp" {
            eprintln!("#   ^ jz pow_lowp is outside tolerance by design (reference only)");
            continue;
        }
        let tol = match space {
            Space::Xyb => 1e-4,
            Space::Jz => 2e-4,
        };
        assert!(
            es.iter().chain(&eh).all(|&v| v.is_finite() && v < tol),
            "{name}: error above {tol}"
        );
    }
}

const SIZES: &[(&str, usize)] = &[("4096px", 4096), ("1MP   ", 1 << 20)];

fn main() {
    accuracy_report();
    let mjz = to_f32(&jz_rgb_to_lms(WHITE_NITS));
    let opsin = to_f32(&OPSIN);
    zenbench::run(|suite| {
        for &(label, n) in SIZES {
            let src = planes(n, 1.0, 3);
            suite.group(
                format!("linear RGB → XYB / JzAzBz  planar f32  {label}"),
                |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    for (name, space, f) in variants() {
                        let src = src.clone();
                        let mut dst = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
                        let m = if space == Space::Xyb { opsin } else { mjz };
                        g.bench(name, move |b| {
                            b.iter(|| {
                                let [d0, d1, d2] = &mut dst;
                                f(&m, [&src[0], &src[1], &src[2]], [d0, d1, d2]);
                                black_box(&dst);
                            })
                        });
                    }
                },
            );
        }
    });
}
