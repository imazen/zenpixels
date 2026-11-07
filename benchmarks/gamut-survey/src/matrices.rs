//! CSS Color 4's matrices, from its conversions.js, and exact RGB to XYZ from chromaticities.
pub type M = [[f64; 3]; 3];
pub const CSS_XYZ_TO_LMS: M = [
    [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
];
pub const CSS_LMS_TO_OKLAB: M = [
    [0.2104542683093140, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.4285922420485799, 0.4505937096174110],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];
pub const CSS_SRGB_TO_XYZ: M = [
    [506752.0 / 1228815.0, 87881.0 / 245763.0, 12673.0 / 70218.0],
    [87098.0 / 409605.0, 175762.0 / 245763.0, 12673.0 / 175545.0],
    [7918.0 / 409605.0, 87881.0 / 737289.0, 1001167.0 / 1053270.0],
];
pub fn mul(a: &M, v: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}
pub fn matmul(a: &M, b: &M) -> M {
    core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
pub fn inv(m: &M) -> M {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
}
/// RGB to XYZ for primaries `[r, g, b]` and white D65, each `(x, y)`.
pub fn rgb_to_xyz(p: [(f64, f64); 3]) -> M {
    let (wx, wy) = (0.3127, 0.3290);
    let w = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    let col = |(x, y): (f64, f64)| [x / y, 1.0, (1.0 - x - y) / y];
    let (r, g, b) = (col(p[0]), col(p[1]), col(p[2]));
    let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    let s = mul(&inv(&m), w);
    core::array::from_fn(|i| core::array::from_fn(|j| m[i][j] * s[j]))
}
