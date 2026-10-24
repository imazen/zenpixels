//! JzAzBz vs XYB vs butteraugli opsin: forward-conversion speed from linear
//! BT.709 RGB f32 planes to 3 output planes.
//!
//! None of these transforms is in zenpixels' API, so the kernels live here.
//! Every SIMD arm is checked against an f64 reference before any timing
//! runs; the max absolute error per output channel prints beside the
//! timings. `--dump DIR` writes inputs and the libjxl-exact arms' outputs
//! for `oracle/libjxl_oracle.cc`, which recomputes them with libjxl's own
//! `fast_math-inl.h` and compares bit for bit.
//!
//! Transforms:
//! - **XYB** (libjxl `enc_xyb.cc` `LinearRGBToXYB`): opsin 3×3 + bias via
//!   nested `MulAdd`, `ZeroIfNegative`, cube root, X/Y/B mix. Cube roots:
//!   `libjxl` = a transcription of `CubeRootAndAdd` (Newton on the inverse
//!   cube root, integer-lane initial guess, no division; bit-exact target),
//!   `cbrt_lowp` / `cbrt_midp` = magetypes (zen default is `cbrt_lowp`, as in
//!   jxl-encoder's `ZEN_XYB_CBRT`).
//! - **Butteraugli opsin dynamics** (libjxl `butteraugli.cc`
//!   `OpsinDynamicsImage`, per-pixel stage): sensitivity = Gamma(absorb(
//!   blurred))/absorb(blurred) with `FastLog2f`, applied to absorb(rgb). The
//!   σ=1.2 blur is a separate stage and not timed; the kernel takes the
//!   blurred planes as input (6 planes in).
//! - **JzAzBz** (Safdar et al. 2017, absolute luminance, 203-nit SDR white):
//!   RGB → XYZ → X'Y'Z' → LMS folds into one 3×3; the PQ curve with exponent
//!   `p = 1.7·2523/32` is a single-range rational P(a)/Q(a), a = x^(1/4),
//!   fitted by `tools/fit-jzazbz-pq` (polyfit), evaluated with IEEE
//!   `sqrt`/division; `pow_midp` is the exact formula for reference.
//!
//! Every vector helper is an `#[attune(_*)]` family and every kernel an
//! `#[attune(dispatch)]` family calling helpers through `attuned!`, so each
//! tier body only calls same-tier feature bodies: nothing depends on an
//! `inline(always)` reaching a generic helper. No closures touch vectors (a
//! closure body compiles without the tier's target features).
//!
//! Planar in, planar out. Run:
//! `cargo run --release --manifest-path tools/jzazbz-bench/Cargo.toml`
//! (hybrid CPUs: pin to one core class, see the results file header).

// Every function whose body calls `attuned!` trips this lint: the macro's
// expansion (archmage 0.9.31-beta.1) contains a diverging sub-expression.
#![allow(clippy::diverging_sub_expression)]

use archmage::prelude::*;
use std::path::Path;
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

// libjxl `cms/opsin_params.h`, computed in f32 exactly as libjxl's
// `constexpr float` (kM01 = 1 − kM02 − kM00 etc.).
const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
#[allow(clippy::excessive_precision)]
const K_M20: f32 = 0.24342268924547819;
#[allow(clippy::excessive_precision)]
const K_M21: f32 = 0.20476744424496821;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const OPSIN_F32: [[f32; 3]; 3] = [
    [K_M00, K_M01, K_M02],
    [K_M10, K_M11, K_M12],
    [K_M20, K_M21, K_M22],
];
#[allow(clippy::excessive_precision)]
const OPSIN_BIAS_F32: f32 = 0.0037930732552754493;

// libjxl `butteraugli.cc` `OpsinAbsorbance` (double literals, `Set` rounds
// to f32) and `OpsinDynamicsImage` / `Gamma`.
#[allow(clippy::excessive_precision)]
const BA_MIX: [f64; 12] = [
    0.29956550340058319,
    0.63373087833825936,
    0.077705617820981968,
    1.7557483643287353,
    0.22158691104574774,
    0.69391388044116142,
    0.0987313588422,
    1.7557483643287353,
    0.02,
    0.02,
    0.20480129041026129,
    12.226454707163354,
];
#[allow(clippy::excessive_precision)]
const BA_MIN01: f32 = 1.7557483643287353;
#[allow(clippy::excessive_precision)]
const BA_MIN2: f32 = 12.226454707163354;
const BA_MIN: f32 = 1e-4;
/// `kInvLog2e` in libjxl `base/common.h`: a float holding ln 2 (its literal
/// `0.6931471805599453` rounds to the same f32 as `LN_2`).
const K_INV_LOG2E: f32 = core::f32::consts::LN_2;
#[allow(clippy::excessive_precision)]
const BA_GAMMA_MUL: f32 = 19.245013259874995 * K_INV_LOG2E;
#[allow(clippy::excessive_precision)]
const BA_GAMMA_ADD: f32 = -23.16046239805755;
#[allow(clippy::excessive_precision)]
const BA_GAMMA_BIAS: f32 = 9.9710635769299145;
/// libjxl `ButteraugliParams::intensity_target` default.
const BA_INTENSITY: f32 = 80.0;

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

/// Everything a kernel reads besides its planes.
struct Params {
    /// Folded RGB → PQ-domain LMS for JzAzBz.
    jz_m: [[f32; 3]; 3],
    /// `-cbrtf(bias)`, computed at runtime as libjxl's `ComputePremulAbsorb`.
    xyb_neg_cbrt_bias: f32,
}

impl Params {
    fn new() -> Self {
        Self {
            jz_m: to_f32(&jz_rgb_to_lms(WHITE_NITS)),
            xyb_neg_cbrt_bias: -OPSIN_BIAS_F32.cbrt(),
        }
    }
}

// ── f64 references ─────────────────────────────────────────────────────

fn xyb_ref(rgb: [f64; 3]) -> [f64; 3] {
    let bias = OPSIN_BIAS_F32 as f64;
    let c = OPSIN_F32.map(|r| {
        let mixed =
            (r[0] as f64 * rgb[0] + r[1] as f64 * rgb[1] + r[2] as f64 * rgb[2] + bias).max(0.0);
        mixed.cbrt() - bias.cbrt()
    });
    [0.5 * (c[0] - c[1]), 0.5 * (c[0] + c[1]), c[2]]
}

/// Butteraugli opsin dynamics per pixel with exact `ln` (libjxl uses
/// `FastLog2f`): `rgb` and `blur` are linear, scaled by the intensity target.
fn butteraugli_ref(rgb: [f64; 3], blur: [f64; 3]) -> [f64; 3] {
    let it = BA_INTENSITY as f64;
    let absorb = |v: [f64; 3]| -> [f64; 3] {
        [0, 1, 2].map(|c| {
            BA_MIX[4 * c] * v[0] * it
                + BA_MIX[4 * c + 1] * v[1] * it
                + BA_MIX[4 * c + 2] * v[2] * it
                + BA_MIX[4 * c + 3]
        })
    };
    let pre = absorb(blur);
    let cur = absorb(rgb);
    let mins = [BA_MIN01 as f64, BA_MIN01 as f64, BA_MIN2 as f64];
    let out = [0, 1, 2].map(|c| {
        let p = pre[c].max(BA_MIX[4 * c + 3]).max(BA_MIN as f64);
        let gamma =
            19.245013259874995 * (p.max(0.0) + BA_GAMMA_BIAS as f64).ln() - 23.16046239805755;
        let sens = (gamma / p).max(BA_MIN as f64);
        (cur[c] * sens).max(mins[c])
    });
    [out[0] - out[1], out[0] + out[1], out[2]]
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

fn xyb_scalar_std(_: &Params, src: &[&[f32]], dst: PlanesMut) {
    let bias = OPSIN_BIAS_F32;
    let cb = bias.cbrt();
    let [d0, d1, d2] = dst;
    for i in 0..src[0].len() {
        let (r, g, b) = (src[0][i], src[1][i], src[2][i]);
        let c = OPSIN_F32.map(|row| {
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

fn jz_scalar_std(p: &Params, src: &[&[f32]], dst: PlanesMut) {
    let (c1, c2, c3, n, pp) = (
        PQ_C1 as f32,
        PQ_C2 as f32,
        PQ_C3 as f32,
        PQ_N as f32,
        JZ_P as f32,
    );
    let pq = |x: f32| {
        let t = x.clamp(0.0, 1.0).powf(n);
        ((c1 + c2 * t) / (1.0 + c3 * t)).powf(pp)
    };
    let m = &p.jz_m;
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

// ── SIMD helper families ───────────────────────────────────────────────

type PlanesMut<'a> = [&'a mut [f32]; 3];
/// Vector type in `#[attune]` signatures (`define(f32x8)` aliases bodies only).
type V<T> = magetypes::simd::generic::f32x8<T>;

/// libjxl `OpsinAbsorbance` row: `MulAdd(m0, r, MulAdd(m1, g, MulAdd(m2, b, add)))`.
#[attune(_*, define(f32x8))]
fn madd3(m: &[f32; 3], r: V<Token>, g: V<Token>, b: V<Token>, add: V<Token>) -> V<Token> {
    let t = Token::from_context();
    f32x8::splat_t(t, m[0]).mul_add(
        r,
        f32x8::splat_t(t, m[1]).mul_add(g, f32x8::splat_t(t, m[2]).mul_add(b, add)),
    )
}

/// libjxl `base/fast_math-inl.h` `CubeRootAndAdd(x, add)` = cbrt(x) + add,
/// for `x >= 0`.
#[attune(_*, define(f32x8, i32x8))]
fn cbrt_add_libjxl(x: V<Token>, add: V<Token>) -> V<Token> {
    let t = Token::from_context();
    let k1_3 = f32x8::splat_t(t, 1.0 / 3.0);
    let k4_3 = f32x8::splat_t(t, 4.0 / 3.0);
    let xa_3 = k1_3 * x;
    // Multiply the exponent by -1/3 in integer lanes; x == 0 selects 0.
    let m1 = x.bitcast_i32x8();
    let izero = i32x8::splat_t(t, 0);
    let m2 = i32x8::splat_t(t, 0x5480_0000)
        - m1.shr_arithmetic_const::<23>() * i32x8::splat_t(t, 0x002A_AAAA);
    let mut r = i32x8::blend(m1.simd_eq(izero), izero, m2).bitcast_f32x8();
    // Three Newton steps on the inverse cube root: NegMulAdd(xa_3, r^4, k4_3·r).
    let mut i = 0;
    while i < 3 {
        let r2 = r * r;
        r = (-xa_3).mul_add(r2 * r2, k4_3 * r);
        i += 1;
    }
    // Final step: MulAdd(k1_3, NegMulAdd(x, r^4, r), r); then MulAdd(r², x, add).
    let r2 = r * r;
    r = k1_3.mul_add((-x).mul_add(r2 * r2, r), r);
    let r2 = r * r;
    r2.mul_add(x, add)
}

/// libjxl `base/fast_math-inl.h` `FastLog2f`: (2,2) rational on the mantissa
/// reduced to [-1/3, 1/3], `EvalRationalPolynomial` Horner order, IEEE division.
#[attune(_*, define(f32x8, i32x8))]
fn fast_log2f(x: V<Token>) -> V<Token> {
    let t = Token::from_context();
    #[allow(clippy::excessive_precision)]
    const P: [f32; 3] = [
        -1.8503833400518310E-06,
        1.4287160470083755E+00,
        7.4245873327820566E-01,
    ];
    #[allow(clippy::excessive_precision)]
    const Q: [f32; 3] = [
        9.9032814277590719E-01,
        1.0096718572241148E+00,
        1.7409343003366853E-01,
    ];
    let x_bits = x.bitcast_i32x8();
    let exp_bits = x_bits - i32x8::splat_t(t, 0x3f2a_aaab);
    let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
    let mantissa = (x_bits - exp_shifted.shl_const::<23>()).bitcast_f32x8();
    let exp_val = exp_shifted.to_f32x8();
    let m = mantissa - f32x8::splat_t(t, 1.0);
    let yp = f32x8::splat_t(t, P[2])
        .mul_add(m, f32x8::splat_t(t, P[1]))
        .mul_add(m, f32x8::splat_t(t, P[0]));
    let yq = f32x8::splat_t(t, Q[2])
        .mul_add(m, f32x8::splat_t(t, Q[1]))
        .mul_add(m, f32x8::splat_t(t, Q[0]));
    yp / yq + exp_val
}

/// libjxl `butteraugli.cc` `Gamma`.
#[attune(_*, define(f32x8))]
fn ba_gamma(v: V<Token>) -> V<Token> {
    let t = Token::from_context();
    let v = v.max(f32x8::zero_t(t));
    let biased = v + f32x8::splat_t(t, BA_GAMMA_BIAS);
    let log = attuned!(fast_log2f(biased));
    f32x8::splat_t(t, BA_GAMMA_MUL).mul_add(log, f32x8::splat_t(t, BA_GAMMA_ADD))
}

/// libjxl `butteraugli.cc` `OpsinAbsorbance`, all three rows.
#[attune(_*, define(f32x8))]
fn ba_absorb(r: V<Token>, g: V<Token>, b: V<Token>) -> [V<Token>; 3] {
    let t = Token::from_context();
    let s = |i: usize| BA_MIX[i] as f32;
    let row = |c: usize| [s(4 * c), s(4 * c + 1), s(4 * c + 2)];
    [
        attuned!(madd3(&row(0), r, g, b, f32x8::splat_t(t, s(3)))),
        attuned!(madd3(&row(1), r, g, b, f32x8::splat_t(t, s(7)))),
        attuned!(madd3(&row(2), r, g, b, f32x8::splat_t(t, s(11)))),
    ]
}

/// Per-channel sensitivity from the blurred absorbance.
#[attune(_*, define(f32x8))]
fn ba_sensitivity(pre: V<Token>, clamp: f32) -> V<Token> {
    let t = Token::from_context();
    let min = f32x8::splat_t(t, BA_MIN);
    let pre = pre.max(f32x8::splat_t(t, clamp)).max(min);
    (attuned!(ba_gamma(pre)) / pre).max(min)
}

/// JzAzBz PQ curve as a fitted rational in `a = x^(1/4)`; `x` already in [0, 1].
macro_rules! pq_fit {
    ($name:ident, $p:ident, $q:ident) => {
        #[attune(_*, define(f32x8))]
        fn $name(x: V<Token>) -> V<Token> {
            let t = Token::from_context();
            let a = x.sqrt().sqrt();
            let mut yp = f32x8::splat_t(t, $p[$p.len() - 1]);
            let mut i = $p.len() - 1;
            while i > 0 {
                i -= 1;
                yp = yp.mul_add(a, f32x8::splat_t(t, $p[i]));
            }
            // Monic Q: skip the leading multiply.
            let mut yq = a + f32x8::splat_t(t, $q[$q.len() - 2]);
            let mut i = $q.len() - 2;
            while i > 0 {
                i -= 1;
                yq = yq.mul_add(a, f32x8::splat_t(t, $q[i]));
            }
            yp / yq
        }
    };
}
pq_fit!(pq43, JZ_PQ_43_P, JZ_PQ_43_Q);
pq_fit!(pq54, JZ_PQ_54_P, JZ_PQ_54_Q);
pq_fit!(pq55, JZ_PQ_55_P, JZ_PQ_55_Q);
pq_fit!(pq66, JZ_PQ_66_P, JZ_PQ_66_Q);

/// Exact-formula JzAzBz PQ with magetypes `pow_midp`.
#[attune(_*, define(f32x8))]
fn pq_pow_midp(x: V<Token>) -> V<Token> {
    let t = Token::from_context();
    // pow_midp_unchecked is undefined at 0; MIN_POSITIVE^n lands 5e-4
    // relative above f(0) = 3.7e-11, i.e. ~2e-14 absolute.
    let tn = x
        .max(f32x8::splat_t(t, f32::MIN_POSITIVE))
        .pow_midp_unchecked(PQ_N as f32);
    let num = tn.mul_add(
        f32x8::splat_t(t, PQ_C2 as f32),
        f32x8::splat_t(t, PQ_C1 as f32),
    );
    let den = tn.mul_add(f32x8::splat_t(t, PQ_C3 as f32), f32x8::splat_t(t, 1.0));
    (num / den).pow_midp_unchecked(JZ_P as f32)
}

/// RGB → PQ-domain LMS, clamped to the PQ input range [0, 1].
#[attune(_*, define(f32x8))]
fn jz_lms(m: &[[f32; 3]; 3], r: V<Token>, g: V<Token>, b: V<Token>) -> [V<Token>; 3] {
    let t = Token::from_context();
    let (zero, one) = (f32x8::zero_t(t), f32x8::splat_t(t, 1.0));
    [
        attuned!(madd3(&m[0], r, g, b, zero)).max(zero).min(one),
        attuned!(madd3(&m[1], r, g, b, zero)).max(zero).min(one),
        attuned!(madd3(&m[2], r, g, b, zero)).max(zero).min(one),
    ]
}

/// L'M'S' → Jz, az, bz.
#[attune(_*, define(f32x8))]
fn jz_finish(l: V<Token>, mm: V<Token>, s: V<Token>) -> [V<Token>; 3] {
    let t = Token::from_context();
    let iz = (l + mm) * f32x8::splat_t(t, 0.5);
    let jz = iz * f32x8::splat_t(t, 0.44)
        / iz.mul_add(f32x8::splat_t(t, -0.56), f32x8::splat_t(t, 1.0))
        - f32x8::splat_t(t, JZ_D0 as f32);
    let az = s.mul_add(
        f32x8::splat_t(t, 0.542708),
        mm.mul_add(f32x8::splat_t(t, -4.066708), l * f32x8::splat_t(t, 3.524)),
    );
    let bz = s.mul_add(
        f32x8::splat_t(t, -1.295875),
        mm.mul_add(f32x8::splat_t(t, 1.096799), l * f32x8::splat_t(t, 0.199076)),
    );
    [jz, az, bz]
}

/// libjxl `LinearRGBToXYB` absorbance: three `madd3` rows + bias, `ZeroIfNegative`.
#[attune(_*, define(f32x8))]
fn xyb_mixed(r: V<Token>, g: V<Token>, b: V<Token>) -> [V<Token>; 3] {
    let t = Token::from_context();
    let bias = f32x8::splat_t(t, OPSIN_BIAS_F32);
    let zero = f32x8::zero_t(t);
    [
        attuned!(madd3(&OPSIN_F32[0], r, g, b, bias)).max(zero),
        attuned!(madd3(&OPSIN_F32[1], r, g, b, bias)).max(zero),
        attuned!(madd3(&OPSIN_F32[2], r, g, b, bias)).max(zero),
    ]
}

/// libjxl `StoreXYB`: X = ½(l − m), Y = ½(l + m), B = s.
#[attune(_*, define(f32x8))]
fn xyb_store_mix(l: V<Token>, mm: V<Token>, s: V<Token>) -> [V<Token>; 3] {
    let half = f32x8::splat_t(Token::from_context(), 0.5);
    [half * (l - mm), half * (l + mm), s]
}

// ── Kernels ────────────────────────────────────────────────────────────

/// A 3-plane-in kernel family. `$body` maps one 8-lane chunk `[r, g, b]` to
/// three outputs; it runs for each full chunk and once for the zero-padded
/// tail, so the tail takes the same arithmetic as the body.
macro_rules! kernel3 {
    ($name:ident, |$t:ident, $p:ident, $r:ident, $g:ident, $b:ident| $body:expr) => {
        #[attune(dispatch, define(f32x8))]
        fn $name($p: &Params, src: &[&[f32]], dst: PlanesMut) {
            let $t = Token::from_context();
            let n = src[0].len();
            let [d0, d1, d2] = dst;
            let (r8, rt) = src[0].as_chunks::<8>();
            let (g8, gt) = src[1][..n].as_chunks::<8>();
            let (b8, bt) = src[2][..n].as_chunks::<8>();
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
                let $r = f32x8::load_t($t, r);
                let $g = f32x8::load_t($t, g);
                let $b = f32x8::load_t($t, b);
                let [a, bb, c] = $body;
                a.store(x);
                bb.store(y);
                c.store(z);
            }
            if !rt.is_empty() {
                let k = rt.len();
                let mut pad = [[0.0f32; 8]; 3];
                pad[0][..k].copy_from_slice(rt);
                pad[1][..k].copy_from_slice(gt);
                pad[2][..k].copy_from_slice(bt);
                let $r = f32x8::load_t($t, &pad[0]);
                let $g = f32x8::load_t($t, &pad[1]);
                let $b = f32x8::load_t($t, &pad[2]);
                let [a, bb, c] = $body;
                o0t.copy_from_slice(&a.to_array()[..k]);
                o1t.copy_from_slice(&bb.to_array()[..k]);
                o2t.copy_from_slice(&c.to_array()[..k]);
            }
        }
    };
}

kernel3!(xyb_libjxl, |t, p, r, g, b| {
    let [l, mm, s] = attuned!(xyb_mixed(r, g, b));
    let add = f32x8::splat_t(t, p.xyb_neg_cbrt_bias);
    attuned!(xyb_store_mix(
        attuned!(cbrt_add_libjxl(l, add)),
        attuned!(cbrt_add_libjxl(mm, add)),
        attuned!(cbrt_add_libjxl(s, add))
    ))
});
kernel3!(xyb_cbrt_lowp, |t, p, r, g, b| {
    let [l, mm, s] = attuned!(xyb_mixed(r, g, b));
    let add = f32x8::splat_t(t, p.xyb_neg_cbrt_bias);
    attuned!(xyb_store_mix(
        l.cbrt_lowp() + add,
        mm.cbrt_lowp() + add,
        s.cbrt_lowp() + add
    ))
});
kernel3!(xyb_cbrt_midp, |t, p, r, g, b| {
    let [l, mm, s] = attuned!(xyb_mixed(r, g, b));
    let add = f32x8::splat_t(t, p.xyb_neg_cbrt_bias);
    attuned!(xyb_store_mix(
        l.cbrt_midp() + add,
        mm.cbrt_midp() + add,
        s.cbrt_midp() + add
    ))
});

macro_rules! jz_kernel {
    ($name:ident, $pq:ident) => {
        kernel3!($name, |t, p, r, g, b| {
            let [l, mm, s] = attuned!(jz_lms(&p.jz_m, r, g, b));
            attuned!(jz_finish(
                attuned!($pq(l)),
                attuned!($pq(mm)),
                attuned!($pq(s))
            ))
        });
    };
}
jz_kernel!(jz_r43, pq43);
jz_kernel!(jz_r54, pq54);
jz_kernel!(jz_r55, pq55);
jz_kernel!(jz_r66, pq66);
jz_kernel!(jz_pow_midp, pq_pow_midp);

/// libjxl `OpsinDynamicsImage` per-pixel stage: planes `[r, g, b, blur_r,
/// blur_g, blur_b]`, linear with 1.0 = `BA_INTENSITY` nits.
#[attune(dispatch, define(f32x8))]
fn butteraugli_opsin(_p: &Params, src: &[&[f32]], dst: PlanesMut) {
    let t = Token::from_context();
    let n = src[0].len();
    let [d0, d1, d2] = dst;
    let ins: [&[[f32; 8]]; 6] = core::array::from_fn(|c| src[c][..n].as_chunks::<8>().0);
    let full = ins[0].len();
    let (o0, o0t) = d0[..n].as_chunks_mut::<8>();
    let (o1, o1t) = d1[..n].as_chunks_mut::<8>();
    let (o2, o2t) = d2[..n].as_chunks_mut::<8>();
    for j in 0..full {
        let v: [f32x8; 6] = [
            f32x8::load_t(t, &ins[0][j]),
            f32x8::load_t(t, &ins[1][j]),
            f32x8::load_t(t, &ins[2][j]),
            f32x8::load_t(t, &ins[3][j]),
            f32x8::load_t(t, &ins[4][j]),
            f32x8::load_t(t, &ins[5][j]),
        ];
        let [a, b, c] = attuned!(ba_px(v));
        a.store(&mut o0[j]);
        b.store(&mut o1[j]);
        c.store(&mut o2[j]);
    }
    let k = n - full * 8;
    if k > 0 {
        let mut pad = [[0.0f32; 8]; 6];
        for (lane, plane) in pad.iter_mut().zip(src) {
            lane[..k].copy_from_slice(&plane[full * 8..n]);
        }
        let v: [f32x8; 6] = [
            f32x8::load_t(t, &pad[0]),
            f32x8::load_t(t, &pad[1]),
            f32x8::load_t(t, &pad[2]),
            f32x8::load_t(t, &pad[3]),
            f32x8::load_t(t, &pad[4]),
            f32x8::load_t(t, &pad[5]),
        ];
        let [a, b, c] = attuned!(ba_px(v));
        o0t.copy_from_slice(&a.to_array()[..k]);
        o1t.copy_from_slice(&b.to_array()[..k]);
        o2t.copy_from_slice(&c.to_array()[..k]);
    }
}

/// One 8-pixel `OpsinDynamicsImage` step: `[r, g, b, blur_r, blur_g, blur_b]`.
#[attune(_*, define(f32x8))]
fn ba_px(v: [V<Token>; 6]) -> [V<Token>; 3] {
    let t = Token::from_context();
    let it = f32x8::splat_t(t, BA_INTENSITY);
    let [p0, p1, p2] = attuned!(ba_absorb(v[3] * it, v[4] * it, v[5] * it));
    let s0 = attuned!(ba_sensitivity(p0, BA_MIX[3] as f32));
    let s1 = attuned!(ba_sensitivity(p1, BA_MIX[7] as f32));
    let s2 = attuned!(ba_sensitivity(p2, BA_MIX[11] as f32));
    let [c0, c1, c2] = attuned!(ba_absorb(v[0] * it, v[1] * it, v[2] * it));
    let c0 = (c0 * s0).max(f32x8::splat_t(t, BA_MIN01));
    let c1 = (c1 * s1).max(f32x8::splat_t(t, BA_MIN01));
    let c2 = (c2 * s2).max(f32x8::splat_t(t, BA_MIN2));
    [c0 - c1, c0 + c1, c2]
}

type KernelFn = fn(&Params, &[&[f32]], PlanesMut);

#[derive(Clone, Copy, PartialEq)]
enum Space {
    Xyb,
    Jz,
    Butteraugli,
}

impl Space {
    fn inputs(self) -> usize {
        if self == Space::Butteraugli { 6 } else { 3 }
    }
}

fn variants() -> Vec<(&'static str, Space, KernelFn)> {
    vec![
        ("xyb libjxl", Space::Xyb, xyb_libjxl),
        ("xyb cbrt_lowp (zen)", Space::Xyb, xyb_cbrt_lowp),
        ("xyb cbrt_midp", Space::Xyb, xyb_cbrt_midp),
        ("xyb scalar std", Space::Xyb, xyb_scalar_std),
        ("butteraugli libjxl", Space::Butteraugli, butteraugli_opsin),
        ("jz r43", Space::Jz, jz_r43),
        ("jz r54", Space::Jz, jz_r54),
        ("jz r55", Space::Jz, jz_r55),
        ("jz r66", Space::Jz, jz_r66),
        ("jz pow_midp", Space::Jz, jz_pow_midp),
        ("jz scalar powf", Space::Jz, jz_scalar_std),
    ]
}

// ── Inputs + accuracy gate ─────────────────────────────────────────────

/// `planes` pseudo-random planes in `[lo, hi)`, with pinned corner pixels.
fn planes(count: usize, n: usize, lo: f32, hi: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
    };
    let mut p: Vec<Vec<f32>> = (0..count).map(|_| vec![0.0; n]).collect();
    for i in 0..n {
        for c in &mut p {
            c[i] = next();
        }
    }
    // Pin the corners: black, white, primaries, the range top.
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [hi, hi, hi],
    ];
    for (i, c) in corners.iter().enumerate().take(n) {
        for (ch, plane) in p.iter_mut().enumerate() {
            plane[i] = c[ch % 3];
        }
    }
    p
}

fn run(f: KernelFn, p: &Params, src: &[Vec<f32>]) -> [Vec<f32>; 3] {
    let n = src[0].len();
    let mut out = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
    let [a, b, c] = &mut out;
    f(p, &refs, [a, b, c]);
    out
}

/// Max absolute error per output channel vs the f64 reference, and the max
/// |reference| per channel (the scale the error is read against).
fn max_err(space: Space, f: KernelFn, p: &Params, src: &[Vec<f32>]) -> ([f64; 3], [f64; 3]) {
    let mjz = jz_rgb_to_lms(WHITE_NITS);
    let out = run(f, p, src);
    let mut e = [0.0f64; 3];
    let mut scale = [0.0f64; 3];
    for i in 0..src[0].len() {
        let px = |c: usize| src[c][i] as f64;
        let rgb = [px(0), px(1), px(2)];
        let r = match space {
            Space::Xyb => xyb_ref(rgb),
            Space::Jz => jz_ref(rgb, &mjz),
            Space::Butteraugli => butteraugli_ref(rgb, [px(3), px(4), px(5)]),
        };
        for ch in 0..3 {
            e[ch] = e[ch].max((out[ch][i] as f64 - r[ch]).abs());
            scale[ch] = scale[ch].max(r[ch].abs());
        }
    }
    (e, scale)
}

const N_CHECK: usize = (1 << 20) + 7; // 8·131072 + 7 exercises the tail

fn accuracy_report(p: &Params) {
    // SDR: [0,1] with 1% wide-gamut negatives; HDR: up to 10000 nits.
    let sdr = planes(6, N_CHECK, -0.01, 1.0, 1);
    let hdr = planes(6, N_CHECK, 0.0, (10000.0 / WHITE_NITS) as f32, 2);
    let mjz = jz_rgb_to_lms(WHITE_NITS);
    eprintln!(
        "# max |err| vs f64 reference (XYB/JzAzBz white = {WHITE_NITS} nits; butteraugli intensity {BA_INTENSITY})"
    );
    eprintln!("#   SDR: rgb ∈ [-0.01,1]; HDR: rgb ∈ [0, 10000/{WHITE_NITS}]");
    let xw = xyb_ref([1.0; 3]);
    let jw = jz_ref([1.0; 3], &mjz);
    eprintln!(
        "#   scale: SDR white XYB Y = {:.4}, Jz = {:.4}",
        xw[1], jw[0]
    );
    for (name, space, f) in variants() {
        let (es, ss) = max_err(space, f, p, &sdr);
        let (eh, sh) = max_err(space, f, p, &hdr);
        eprintln!(
            "#   {name:<20} SDR [{:.2e} {:.2e} {:.2e}]  HDR [{:.2e} {:.2e} {:.2e}]",
            es[0], es[1], es[2], eh[0], eh[1], eh[2]
        );
        // Gate: every arm must be a real implementation of its transform.
        // Butteraugli's FastLog2f carries ~4e-6 log2 error by design, so it is
        // gated relative to its output scale.
        let ok = match space {
            Space::Xyb => es.iter().chain(&eh).all(|&v| v < 1e-4),
            Space::Jz => es.iter().chain(&eh).all(|&v| v < 2e-4),
            Space::Butteraugli => (0..3).all(|c| es[c] < 1e-5 * ss[c] && eh[c] < 1e-5 * sh[c]),
        };
        // jz r43: its fit misses f(0) by 5e-5 (P(0)/Q(0) = -5e-5); where
        // wide-gamut input clamps LMS to 0, az's ~8x coefficient sum turns
        // that into ~3e-4. Timed as a reference point, not gated.
        if name == "jz r43" && !ok {
            eprintln!("#   ^ {name} fails the 2e-4 gate on clamped-to-zero LMS (reference only)");
            continue;
        }
        assert!(
            ok && es.iter().chain(&eh).all(|v| v.is_finite()),
            "{name}: error above tolerance"
        );
    }
}

fn write_f32(path: &Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Inputs and libjxl-exact outputs for `oracle/libjxl_oracle.cc`.
fn dump(p: &Params, dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut src = planes(6, N_CHECK, -0.05, 1.0, 7);
    // HDR-range and exact-zero pixels too.
    for plane in &mut src {
        for i in (0..N_CHECK).step_by(97) {
            plane[i] *= 40.0;
        }
        for i in (5..N_CHECK).step_by(1009) {
            plane[i] = 0.0;
        }
    }
    for (c, name) in ["r", "g", "b", "br", "bg", "bb"].iter().enumerate() {
        write_f32(&dir.join(format!("in_{name}.f32")), &src[c]);
    }
    for (prefix, f) in [
        ("xyb_libjxl", xyb_libjxl as KernelFn),
        ("butteraugli", butteraugli_opsin as KernelFn),
    ] {
        let out = run(f, p, &src);
        for (c, plane) in out.iter().enumerate() {
            write_f32(&dir.join(format!("{prefix}_{c}.f32")), plane);
        }
    }
    std::fs::write(
        dir.join("meta.txt"),
        format!("n={N_CHECK}\nintensity_target={BA_INTENSITY}\nsource=jzazbz-bench\n"),
    )
    .unwrap();
    eprintln!("# dumped {N_CHECK} px to {}", dir.display());
}

const SIZES: &[(&str, usize)] = &[("4096px", 4096), ("1MP   ", 1 << 20)];

fn main() {
    let p = Params::new();
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        dump(&p, Path::new(&args[i + 1]));
        return;
    }
    accuracy_report(&p);
    zenbench::run(|suite| {
        for &(label, n) in SIZES {
            let src = planes(6, n, 0.0, 1.0, 3);
            suite.group(
                format!("linear RGB → XYB / butteraugli / JzAzBz  planar f32  {label}"),
                |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    for (name, space, f) in variants() {
                        let src: Vec<Vec<f32>> = src[..space.inputs()].to_vec();
                        let mut dst = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
                        let p = Params::new();
                        g.bench(name, move |b| {
                            b.iter(|| {
                                let refs: [&[f32]; 6] = std::array::from_fn(|i| {
                                    src.get(i).map(|v| v.as_slice()).unwrap_or(&[])
                                });
                                let [d0, d1, d2] = &mut dst;
                                f(&p, &refs[..src.len()], [d0, d1, d2]);
                                black_box(&dst);
                            })
                        });
                    }
                },
            );
        }
    });
}
