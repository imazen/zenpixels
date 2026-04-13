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
    // DCI-P3 matrices (include Bradford DCI↔D65 adaptation)
    MatrixRef {
        name: "DCIP3_TO_SRGB",
        matrix: &DCIP3_TO_SRGB,
        inverse: Some(&SRGB_TO_DCIP3),
        red: [1.1575165, -0.0415001, -0.0180500],
        green: [-0.1549624, 1.0455674, -0.0785782],
        blue: [-0.0025541, -0.0040678, 1.0966280],
    },
    MatrixRef {
        name: "SRGB_TO_DCIP3",
        matrix: &SRGB_TO_DCIP3,
        inverse: Some(&DCIP3_TO_SRGB),
        red: [0.8685798, 0.0345404, 0.0167714],
        green: [0.1289194, 0.9618117, 0.0710400],
        blue: [0.0025011, 0.0036482, 0.9121888],
    },
    MatrixRef {
        name: "DCIP3_TO_P3",
        matrix: &DCIP3_TO_P3,
        inverse: Some(&P3_TO_DCIP3),
        red: [0.9446453, -0.0016997, 0.0003340],
        green: [0.0581774, 1.0057173, 0.0015022],
        blue: [-0.0028229, -0.0040176, 0.9981638],
    },
    MatrixRef {
        name: "P3_TO_DCIP3",
        matrix: &P3_TO_DCIP3,
        inverse: Some(&DCIP3_TO_P3),
        red: [1.0584873, 0.0017875, -0.0003569],
        green: [-0.0612339, 0.9942058, -0.0014758],
        blue: [0.0027470, 0.0040067, 1.0018328],
    },
    MatrixRef {
        name: "DCIP3_TO_BT2020",
        matrix: &DCIP3_TO_BT2020,
        inverse: Some(&BT2020_TO_DCIP3),
        red: [0.7117833, 0.0416152, -0.0008447],
        green: [0.2436601, 0.9498416, 0.0191095],
        blue: [0.0445565, 0.0085432, 0.9817352],
    },
    MatrixRef {
        name: "BT2020_TO_DCIP3",
        matrix: &BT2020_TO_DCIP3,
        inverse: Some(&DCIP3_TO_BT2020),
        red: [1.4261665, -0.0625062, 0.0024438],
        green: [-0.3646120, 1.0689719, -0.0211213],
        blue: [-0.0615543, -0.0064655, 1.0186777],
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

// =========================================================================
// Cross-validation against independent sources
// =========================================================================

/// CSS Color Level 4 reference matrices (from W3C spec, independent derivation).
/// https://www.w3.org/TR/css-color-4/#color-conversion-code
#[test]
fn css_color_4_p3_to_srgb() {
    // CSS spec's lin_P3_to_lin_sRGB matrix (7-digit precision from spec JS code)
    let css: [[f32; 3]; 3] = [
        [1.2249401, -0.2249402, 0.0],
        [-0.0420570, 1.0420571, 0.0],
        [-0.0196376, -0.0786361, 1.0982736],
    ];
    for i in 0..3 {
        for j in 0..3 {
            let err = (P3_TO_SRGB[i][j] - css[i][j]).abs();
            assert!(
                err < 1e-6,
                "P3_TO_SRGB[{i}][{j}]: ours={}, CSS={}, err={err:.1e}",
                P3_TO_SRGB[i][j],
                css[i][j]
            );
        }
    }
}

#[test]
fn css_color_4_srgb_to_bt2020() {
    // CSS spec's lin_sRGB_to_lin_2020 matrix
    let css: [[f32; 3]; 3] = [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690973, 0.9195404, 0.0113623],
        [0.0163914, 0.0880133, 0.8955953],
    ];
    for i in 0..3 {
        for j in 0..3 {
            let err = (SRGB_TO_BT2020[i][j] - css[i][j]).abs();
            assert!(
                err < 1e-6,
                "SRGB_TO_BT2020[{i}][{j}]: ours={}, CSS={}, err={err:.1e}",
                SRGB_TO_BT2020[i][j],
                css[i][j]
            );
        }
    }
}

/// Cross-validate against saucecontrol/Compact-ICC-Profiles colorants.
/// These profiles were built by an independent tool chain (saucecontrol's
/// ICC profile generator). We extract the D50 PCS XYZ colorants and derive
/// matrices from them, then compare to ours.
///
/// Expected accuracy: ~1e-4 (limited by ICC s15Fixed16 quantization at
/// 1/65536 per colorant component).
#[test]
fn icc_colorant_cross_validation() {
    extern crate std;
    use std::process::Command;

    let tmp = std::env::temp_dir().join("zencodec-compact-icc-profiles");
    let profile_dir = tmp.join("profiles");

    if !profile_dir.exists() {
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/saucecontrol/Compact-ICC-Profiles.git",
            ])
            .arg(&tmp)
            .status();
        match status {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("git clone failed, skipping ICC cross-validation");
                return;
            }
        }
    }

    /// Read s15Fixed16 XYZ from ICC tag data.
    fn read_xyz(data: &[u8], offset: usize) -> [f64; 3] {
        let mut v = [0.0f64; 3];
        for i in 0..3 {
            let raw =
                i32::from_be_bytes(data[offset + i * 4..offset + i * 4 + 4].try_into().unwrap());
            v[i] = raw as f64 / 65536.0;
        }
        v
    }

    /// Extract rXYZ/gXYZ/bXYZ colorants from ICC profile.
    fn get_colorants(data: &[u8]) -> Option<[[f64; 3]; 3]> {
        if data.len() < 132 || &data[36..40] != b"acsp" {
            return None;
        }
        let tag_count = u32::from_be_bytes(data[128..132].try_into().unwrap()) as usize;
        let mut r = None;
        let mut g = None;
        let mut b = None;
        for i in 0..tag_count.min(200) {
            let off = 132 + i * 12;
            if off + 12 > data.len() {
                break;
            }
            let sig = &data[off..off + 4];
            let d_off = u32::from_be_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
            let d_sz = u32::from_be_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
            if d_sz >= 20 && d_off + 20 <= data.len() {
                match sig {
                    b"rXYZ" => r = Some(read_xyz(data, d_off + 8)),
                    b"gXYZ" => g = Some(read_xyz(data, d_off + 8)),
                    b"bXYZ" => b = Some(read_xyz(data, d_off + 8)),
                    _ => {}
                }
            }
        }
        Some([r?, g?, b?])
    }

    /// 3x3 f64 matrix inverse.
    fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let inv_det = 1.0 / det;
        let mut r = [[0.0f64; 3]; 3];
        r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
        r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        r
    }

    /// 3x3 f64 matrix multiply.
    fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        r
    }

    /// Build XYZ(D50) matrix from colorants — columns are the colorant XYZ values.
    fn colorants_to_xyz(c: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        // c[0] = rXYZ, c[1] = gXYZ, c[2] = bXYZ (each is [X, Y, Z])
        // The XYZ-from-RGB matrix has these as columns:
        [
            [c[0][0], c[1][0], c[2][0]], // row 0: X components
            [c[0][1], c[1][1], c[2][1]], // row 1: Y components
            [c[0][2], c[1][2], c[2][2]], // row 2: Z components
        ]
    }

    let check = |name: &str, our_matrix: &[[f32; 3]; 3], src_file: &str, dst_file: &str| {
        let src_data = std::fs::read(profile_dir.join(src_file)).unwrap();
        let dst_data = std::fs::read(profile_dir.join(dst_file)).unwrap();
        let src_c = get_colorants(&src_data).unwrap();
        let dst_c = get_colorants(&dst_data).unwrap();

        let src_xyz = colorants_to_xyz(&src_c);
        let dst_xyz = colorants_to_xyz(&dst_c);
        let icc_matrix = mul3(&inv3(&dst_xyz), &src_xyz);

        let mut max_err: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let err = (our_matrix[i][j] as f64 - icc_matrix[i][j]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        eprintln!("{name}: max error vs ICC colorants = {max_err:.6e}");
        assert!(
            max_err < 2e-4,
            "{name}: max error {max_err:.6e} > 2e-4 vs ICC profile colorants"
        );
    };

    check("SRGB_TO_P3", &SRGB_TO_P3, "sRGB-v4.icc", "DisplayP3-v4.icc");
    check("P3_TO_SRGB", &P3_TO_SRGB, "DisplayP3-v4.icc", "sRGB-v4.icc");
    check(
        "SRGB_TO_BT2020",
        &SRGB_TO_BT2020,
        "sRGB-v4.icc",
        "Rec2020-v4.icc",
    );
    check(
        "BT2020_TO_SRGB",
        &BT2020_TO_SRGB,
        "Rec2020-v4.icc",
        "sRGB-v4.icc",
    );
    check(
        "SRGB_TO_ADOBERGB",
        &SRGB_TO_ADOBERGB,
        "sRGB-v4.icc",
        "AdobeCompat-v4.icc",
    );
    check(
        "ADOBERGB_TO_SRGB",
        &ADOBERGB_TO_SRGB,
        "AdobeCompat-v4.icc",
        "sRGB-v4.icc",
    );
    check(
        "P3_TO_BT2020",
        &P3_TO_BT2020,
        "DisplayP3-v4.icc",
        "Rec2020-v4.icc",
    );
    check(
        "BT2020_TO_P3",
        &BT2020_TO_P3,
        "Rec2020-v4.icc",
        "DisplayP3-v4.icc",
    );
}
