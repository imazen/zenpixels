// Independent generator gate, using H.273 (07/2024) Table 2 rather than
// moxcms's primaries/adaptation helpers. https://www.itu.int/rec/T-REC-H.273
// Matrices below store columns (one XYZ vector per primary).
fn det([a, b, c]: [[f64; 3]; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
}
fn mul(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| (0..3).map(|j| m[j][i] * v[j]).sum())
}
fn cramer(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    let d = det(m);
    assert!(d.abs() > 1e-10, "singular reference matrix");
    core::array::from_fn(|i| {
        let mut replaced = m;
        replaced[i] = v;
        det(replaced) / d
    })
}
fn expected_h273_colorants(p: u8) -> [[f64; 3]; 3] {
    let (xy, white): ([[f64; 2]; 3], [f64; 2]) = match p {
        1 => (
            [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
            [0.3127, 0.3290],
        ),
        4 => (
            [[0.670, 0.330], [0.210, 0.710], [0.140, 0.080]],
            [0.310, 0.316],
        ),
        5 => (
            [[0.640, 0.330], [0.290, 0.600], [0.150, 0.060]],
            [0.3127, 0.3290],
        ),
        6 | 7 => (
            [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]],
            [0.3127, 0.3290],
        ),
        8 => (
            [[0.681, 0.319], [0.243, 0.692], [0.145, 0.049]],
            [0.310, 0.316],
        ),
        9 => (
            [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            [0.3127, 0.3290],
        ),
        10 => ([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [1.0 / 3.0, 1.0 / 3.0]),
        11 => (
            [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            [0.314, 0.351],
        ),
        12 => (
            [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            [0.3127, 0.3290],
        ),
        22 => (
            [[0.630, 0.340], [0.295, 0.605], [0.155, 0.077]],
            [0.3127, 0.3290],
        ),
        _ => panic!("no H.273 table row for primaries {p}"),
    };
    let xyz = |[x, y]: [f64; 2]| [x / y, 1.0, (1.0 - x - y) / y];
    let source_white = xyz(white);
    let columns = if p == 10 {
        // XYZ is the identity basis; converting its zero-y primaries via
        // xy/Y would divide by zero and is precisely the excluded case.
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        let basis = xy.map(xyz);
        let scales = cramer(basis, source_white);
        core::array::from_fn(|i| basis[i].map(|v| v * scales[i]))
    };
    let bradford = [
        [0.8951, -0.7502, 0.0389],
        [0.2664, 1.7135, -0.0685],
        [-0.1614, 0.0367, 1.0296],
    ];
    let src_cone = mul(bradford, source_white);
    let dst_cone = mul(bradford, [0.9642, 1.0, 0.8249]); // ICC PCS D50
    columns.map(|column| {
        let cone = mul(bradford, column);
        cramer(
            bradford,
            core::array::from_fn(|i| cone[i] * dst_cone[i] / src_cone[i]),
        )
    })
}
fn read_rgb_colorants(icc: &[u8]) -> Result<[[f64; 3]; 3], String> {
    let read_u32 = |b: &[u8]| u32::from_be_bytes(b.try_into().unwrap()) as usize;
    let count = icc.get(128..132).ok_or("ICC header missing")?;
    let mut columns = [[0.0; 3]; 3];
    for (sig, column) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().zip(&mut columns) {
        let mut found = false;
        for i in 0..read_u32(count) {
            let e = icc
                .get(132 + 12 * i..144 + 12 * i)
                .ok_or("truncated tag table")?;
            if &e[..4] != sig {
                continue;
            }
            let offset = read_u32(&e[4..8]);
            let size = read_u32(&e[8..12]);
            let t = icc
                .get(offset..offset.checked_add(size).ok_or("tag overflow")?)
                .ok_or("tag out of bounds")?;
            if t.len() < 20 || &t[..4] != b"XYZ " {
                return Err("invalid XYZ tag".into());
            }
            *column = core::array::from_fn(|j| {
                f64::from(i32::from_be_bytes(
                    t[8 + 4 * j..12 + 4 * j].try_into().unwrap(),
                )) / 65536.0
            });
            found = true;
            break;
        }
        if !found {
            return Err(format!("missing colorant {sig:?}"));
        }
    }
    Ok(columns)
}
fn validate_rgb_colorimetry(icc: &[u8], primaries: u8) -> Result<(), String> {
    let got = read_rgb_colorants(icc)?;
    // Every supported H.273 matrix has |det| > 0.01. This margin rejects
    // rank collapse by far more than s15Fixed16 serialization noise.
    let determinant = det(got);
    if !determinant.is_finite() || determinant.abs() <= 0.01 {
        return Err(format!(
            "singular RGB colorants, det={determinant}: {got:?}"
        ));
    }
    let expected = expected_h273_colorants(primaries);
    for (a, b) in got
        .into_iter()
        .flatten()
        .zip(expected.into_iter().flatten())
    {
        if (a - b).abs() > 0.001 {
            return Err(format!(
                "H.273/D50 colorant mismatch: got {got:?}, expected {expected:?}"
            ));
        }
    }
    Ok(())
}
