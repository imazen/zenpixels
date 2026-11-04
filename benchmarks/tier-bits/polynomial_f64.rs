//! The same PQ decode polynomial, in f32 and in f64: are the reversals rounding noise?
#![allow(clippy::excessive_precision)]
const PQ_EOTF_P_SMALL: [f32; 5] = [
    4.36033203e-10,
    -1.53651077e-07,
    2.06864196e-06,
    2.80304711e-03,
    1.47809948e-02,
];
const PQ_EOTF_Q_SMALL: [f32; 5] = [
    -9.89955022e-03,
    1.89911767e+00,
    -1.34828020e+00,
    -7.10547502e-01,
    1.00000000e+00,
];
const PQ_EOTF_P_LARGE: [f32; 5] = [
    5.43046166e-04,
    -6.31017431e-03,
    2.90480734e-01,
    9.80363032e-01,
    2.22819483e-01,
];
const PQ_EOTF_Q_LARGE: [f32; 5] = [
    1.58688561e+02,
    -1.60764060e+02,
    6.51370372e+01,
    -1.26438391e+01,
    1.00000000e+00,
];
fn f32_eval<const P: usize, const Q: usize>(x: f32, p: [f32; P], q: [f32; Q]) -> f32 {
    let mut a = p[P - 1];
    for i in (0..P - 1).rev() { a = a.mul_add(x, p[i]); }
    let mut b = q[Q - 1];
    for i in (0..Q - 1).rev() { b = b.mul_add(x, q[i]); }
    a / b
}
fn f64_eval<const P: usize, const Q: usize>(x: f64, p: [f32; P], q: [f32; Q]) -> f64 {
    let mut a = f64::from(p[P - 1]);
    for i in (0..P - 1).rev() { a = a * x + f64::from(p[i]); }
    let mut b = f64::from(q[Q - 1]);
    for i in (0..Q - 1).rev() { b = b * x + f64::from(q[i]); }
    a / b
}
fn main() {
    // [0.25, 1], the large polynomial, where every u16 reversal in PQ decode sits.
    let (start, end) = (0.25f32.to_bits(), 1f32.to_bits());
    let code = |v: f64| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as i32;
    let (mut pf, mut pd, mut pdf) = (f32::NEG_INFINITY, f64::NEG_INFINITY, f32::NEG_INFINITY);
    let (mut pcf, mut pcd) = (-1, -1);
    let (mut rf, mut rd, mut rdf, mut cf, mut cd) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut n = 0u64;
    for bits in start..=end {
        let v = f32::from_bits(bits);
        let x32 = v + v * v;
        let x64 = f64::from(v) + f64::from(v) * f64::from(v);
        let y32 = f32_eval(x32, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE);
        let y64 = f64_eval(x64, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE);
        let y64f = y64 as f32;
        if y32 < pf { rf += 1; }
        if y64 < pd { rd += 1; }
        if y64f < pdf { rdf += 1; }
        let (a, b) = (code(f64::from(y32)), code(y64));
        if a < pcf { cf += 1; }
        if b < pcd { cd += 1; }
        pf = y32; pd = y64; pdf = y64f; pcf = a; pcd = b;
        n += 1;
    }
    println!("{n} inputs in [0.25, 1], same coefficients:");
    println!("  evaluated in f32: {rf} reversals, {cf} as u16");
    println!("  evaluated in f64: {rd} reversals, {cd} as u16");
    println!("  evaluated in f64, rounded to f32 at the end: {rdf} reversals");
}
