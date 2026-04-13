//! Exhaustive matrix verification against independently-computed reference values.
//!
//! For every gamut matrix in fast_gamut, verify:
//! 1. Primary colors (1,0,0), (0,1,0), (0,0,1) map to the correct coordinates
//!    (derived independently from chromaticity coordinates via f64 math)
//! 2. White (1,1,1) maps to (1,1,1)
//! 3. Black (0,0,0) maps to (0,0,0)
//! 4. Row sums = 1 (white preservation, redundant with #2 but catches sign errors)
//! 5. Forward × inverse = identity
//! 6. Matrix elements match independently-computed f64 reference to <1e-5

use zenpixels_convert::fast_gamut::*;

/// Apply a 3×3 matrix to an RGB triple.
fn apply(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> [f32; 3] {
    [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ]
}

/// Reference: expected output of (1,0,0), (0,1,0), (0,0,1) through each matrix.
/// Computed independently from chromaticity coordinates via f64 Python script.
struct MatrixRef {
    name: &'static str,
    matrix: &'static [[f32; 3]; 3],
    /// Expected output for source (1,0,0) — i.e., column 0 / row outputs for red primary.
    red: [f32; 3],
    /// Expected output for source (0,1,0).
    green: [f32; 3],
    /// Expected output for source (0,0,1).
    blue: [f32; 3],
    /// Inverse matrix (if available) for round-trip verification.
    inverse: Option<&'static [[f32; 3]; 3]>,
}

const REFS: &[MatrixRef] = &[
    // D65 ↔ D65 matrices (no chromatic adaptation)
    MatrixRef {
        name: "SRGB_TO_P3",
        matrix: &SRGB_TO_P3,
        inverse: Some(&P3_TO_SRGB),
        red: [0.8224620, 0.0331942, 0.0170826],
        green: [0.1775380, 0.9668058, 0.0723974],
        blue: [0.0, 0.0, 0.9105199],
    },
    MatrixRef {
        name: "P3_TO_SRGB",
        matrix: &P3_TO_SRGB,
        inverse: Some(&SRGB_TO_P3),
        red: [1.2249402, -0.0420570, -0.0196376],
        green: [-0.2249402, 1.0420570, -0.0786360],
        blue: [0.0, 0.0, 1.0982736],
    },
    MatrixRef {
        name: "SRGB_TO_BT2020",
        matrix: &SRGB_TO_BT2020,
        inverse: Some(&BT2020_TO_SRGB),
        red: [0.6274039, 0.0690973, 0.0163914],
        green: [0.3292830, 0.9195404, 0.0880133],
        blue: [0.0433131, 0.0113623, 0.8955953],
    },
    MatrixRef {
        name: "BT2020_TO_SRGB",
        matrix: &BT2020_TO_SRGB,
        inverse: Some(&SRGB_TO_BT2020),
        red: [1.6604910, -0.1245505, -0.0181508],
        green: [-0.5876411, 1.1328999, -0.1005789],
        blue: [-0.0728499, -0.0083494, 1.1187297],
    },
    MatrixRef {
        name: "P3_TO_BT2020",
        matrix: &P3_TO_BT2020,
        inverse: Some(&BT2020_TO_P3),
        red: [0.7538330, 0.0457438, -0.0012103],
        green: [0.1985974, 0.9417772, 0.0176017],
        blue: [0.0475696, 0.0124789, 0.9836086],
    },
    MatrixRef {
        name: "BT2020_TO_P3",
        matrix: &BT2020_TO_P3,
        inverse: Some(&P3_TO_BT2020),
        red: [1.3435783, -0.0652975, 0.0028218],
        green: [-0.2821797, 1.0757879, -0.0195985],
        blue: [-0.0613986, -0.0104905, 1.0167767],
    },
    MatrixRef {
        name: "SRGB_TO_ADOBERGB",
        matrix: &SRGB_TO_ADOBERGB,
        inverse: Some(&ADOBERGB_TO_SRGB),
        red: [0.7151256, 0.0, 0.0],
        green: [0.2848744, 1.0, 0.0411619],
        blue: [0.0, 0.0, 0.9588381],
    },
    MatrixRef {
        name: "ADOBERGB_TO_SRGB",
        matrix: &ADOBERGB_TO_SRGB,
        inverse: Some(&SRGB_TO_ADOBERGB),
        red: [1.3983557, 0.0, 0.0],
        green: [-0.3983557, 1.0, -0.0429290],
        blue: [0.0, 0.0, 1.0429290],
    },
    MatrixRef {
        name: "ADOBERGB_TO_P3",
        matrix: &ADOBERGB_TO_P3,
        inverse: Some(&P3_TO_ADOBERGB),
        red: [1.1500944, 0.0464173, 0.0238876],
        green: [-0.1500944, 0.9535827, 0.0265048],
        blue: [0.0, 0.0, 0.9496076],
    },
    MatrixRef {
        name: "P3_TO_ADOBERGB",
        matrix: &P3_TO_ADOBERGB,
        inverse: Some(&ADOBERGB_TO_P3),
        red: [0.8640051, -0.0420570, -0.0205604],
        green: [0.1359949, 1.0420570, -0.0325061],
        blue: [0.0, 0.0, 1.0530665],
    },
    MatrixRef {
        name: "ADOBERGB_TO_BT2020",
        matrix: &ADOBERGB_TO_BT2020,
        inverse: Some(&BT2020_TO_ADOBERGB),
        red: [0.8773338, 0.0966226, 0.0229211],
        green: [0.0774937, 0.8915273, 0.0430367],
        blue: [0.0451725, 0.0118501, 0.9340423],
    },
    MatrixRef {
        name: "BT2020_TO_ADOBERGB",
        matrix: &BT2020_TO_ADOBERGB,
        inverse: Some(&ADOBERGB_TO_BT2020),
        red: [1.1519784, -0.1245505, -0.0225304],
        green: [-0.0975031, 1.1328999, -0.0498065],
        blue: [-0.0544753, -0.0083494, 1.0723369],
    },
    // DCI-P3 matrices (include Bradford D50↔D65 adaptation)
    MatrixRef {
        name: "DCIP3_TO_SRGB",
        matrix: &DCIP3_TO_SRGB,
        inverse: Some(&SRGB_TO_DCIP3),
        red: [1.3172195, -0.0427573, -0.0198711],
        green: [-0.3028431, 1.0481183, -0.0745299],
        blue: [-0.0143764, -0.0053610, 1.0944010],
    },
    MatrixRef {
        name: "SRGB_TO_DCIP3",
        matrix: &SRGB_TO_DCIP3,
        inverse: Some(&DCIP3_TO_SRGB),
        red: [0.7665586, 0.0313534, 0.0160536],
        green: [0.2222828, 0.9635149, 0.0696524],
        blue: [0.0111586, 0.0051317, 0.9142939],
    },
    MatrixRef {
        name: "DCIP3_TO_P3",
        matrix: &DCIP3_TO_P3,
        inverse: Some(&P3_TO_DCIP3),
        red: [1.0757719, 0.0023860, 0.0013130],
        green: [-0.0629961, 1.0032742, 0.0028467],
        blue: [-0.0127758, -0.0056602, 0.9958402],
    },
    MatrixRef {
        name: "P3_TO_DCIP3",
        matrix: &P3_TO_DCIP3,
        inverse: Some(&DCIP3_TO_P3),
        red: [0.9294208, -0.0022172, -0.0012191],
        green: [0.0583240, 0.9965813, -0.0029257],
        blue: [0.0122552, 0.0056360, 1.0041449],
    },
    MatrixRef {
        name: "DCIP3_TO_BT2020",
        matrix: &DCIP3_TO_BT2020,
        inverse: Some(&BT2020_TO_DCIP3),
        red: [0.8114887, 0.0514734, 0.0000315],
        green: [0.1518945, 0.9420146, 0.0205357],
        blue: [0.0366168, 0.0065119, 0.9794329],
    },
    MatrixRef {
        name: "BT2020_TO_DCIP3",
        matrix: &BT2020_TO_DCIP3,
        inverse: Some(&DCIP3_TO_BT2020),
        red: [1.2449757, -0.0680374, 0.0013865],
        green: [-0.1997595, 1.0726253, -0.0224832],
        blue: [-0.0452162, -0.0045879, 1.0210967],
    },
];

const TOL: f32 = 5e-5;

/// Verify primary color outputs match reference.
#[test]
fn all_matrices_primary_colors() {
    for r in REFS {
        let out_r = apply(r.matrix, 1.0, 0.0, 0.0);
        let out_g = apply(r.matrix, 0.0, 1.0, 0.0);
        let out_b = apply(r.matrix, 0.0, 0.0, 1.0);

        for ch in 0..3 {
            assert!(
                (out_r[ch] - r.red[ch]).abs() < TOL,
                "{}: red[{ch}] = {}, expected {}",
                r.name,
                out_r[ch],
                r.red[ch]
            );
            assert!(
                (out_g[ch] - r.green[ch]).abs() < TOL,
                "{}: green[{ch}] = {}, expected {}",
                r.name,
                out_g[ch],
                r.green[ch]
            );
            assert!(
                (out_b[ch] - r.blue[ch]).abs() < TOL,
                "{}: blue[{ch}] = {}, expected {}",
                r.name,
                out_b[ch],
                r.blue[ch]
            );
        }
    }
}

/// Verify white (1,1,1) maps to (1,1,1) for all matrices.
#[test]
fn all_matrices_white() {
    for r in REFS {
        let out = apply(r.matrix, 1.0, 1.0, 1.0);
        for ch in 0..3 {
            assert!(
                (out[ch] - 1.0).abs() < TOL,
                "{}: white[{ch}] = {}, expected 1.0",
                r.name,
                out[ch]
            );
        }
    }
}

/// Verify black (0,0,0) maps to (0,0,0) for all matrices.
#[test]
fn all_matrices_black() {
    for r in REFS {
        let out = apply(r.matrix, 0.0, 0.0, 0.0);
        for ch in 0..3 {
            assert!(
                out[ch].abs() < 1e-7,
                "{}: black[{ch}] = {}, expected 0.0",
                r.name,
                out[ch]
            );
        }
    }
}

/// Verify row sums = 1.0 (necessary for white preservation).
#[test]
fn all_matrices_row_sums() {
    for r in REFS {
        for row in 0..3 {
            let sum: f32 = r.matrix[row].iter().sum();
            assert!(
                (sum - 1.0).abs() < TOL,
                "{}: row {row} sum = {sum}, expected 1.0",
                r.name
            );
        }
    }
}

/// Verify forward × inverse ≈ identity for all matrix pairs.
#[test]
fn all_matrices_inverse_identity() {
    for r in REFS {
        let Some(inv) = r.inverse else { continue };
        for i in 0..3 {
            for j in 0..3 {
                let sum: f32 = (0..3).map(|k| r.matrix[i][k] * inv[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-4,
                    "{} × inverse: [{i}][{j}] = {sum}, expected {expected}",
                    r.name
                );
            }
        }
    }
}

/// Verify matrix elements match the reference primary colors.
/// The matrix columns ARE the primary outputs: column 0 = output for (1,0,0), etc.
/// This is a structural check — if the matrix is transposed or rows/columns are
/// swapped, this catches it.
#[test]
fn all_matrices_element_match() {
    for r in REFS {
        // Matrix row i, column j should equal ref_primary[j][i]
        // because apply(M, 1,0,0) = [M[0][0], M[1][0], M[2][0]] = red output
        let primaries = [&r.red, &r.green, &r.blue];
        for row in 0..3 {
            for col in 0..3 {
                let actual = r.matrix[row][col];
                let expected = primaries[col][row];
                assert!(
                    (actual - expected).abs() < TOL,
                    "{}: M[{row}][{col}] = {actual}, expected {expected} (from {} primary[{row}])",
                    r.name,
                    ["red", "green", "blue"][col]
                );
            }
        }
    }
}

/// Cross-validate: for each matrix pair (A→B, B→A), verify that
/// converting a grid of colors forward then back produces the original.
#[test]
fn all_matrices_roundtrip_grid() {
    for r in REFS {
        let Some(inv) = r.inverse else { continue };
        let mut max_err: f32 = 0.0;
        for ri in (0..=255).step_by(17) {
            for gi in (0..=255).step_by(17) {
                for bi in (0..=255).step_by(17) {
                    let orig = [ri as f32 / 255.0, gi as f32 / 255.0, bi as f32 / 255.0];
                    let fwd = apply(r.matrix, orig[0], orig[1], orig[2]);
                    let back = apply(inv, fwd[0], fwd[1], fwd[2]);
                    for ch in 0..3 {
                        let err = (orig[ch] - back[ch]).abs();
                        if err > max_err {
                            max_err = err;
                        }
                    }
                }
            }
        }
        assert!(
            max_err < 1e-4,
            "{}: roundtrip max error = {max_err}",
            r.name
        );
    }
}
