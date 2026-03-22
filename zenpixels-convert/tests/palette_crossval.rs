//! Cross-validation tests against the `palette` crate (0.7).
//!
//! These tests verify that our transfer function, Oklab, and XYZ matrix
//! implementations agree with an independent, widely-used color math library.

use palette::{IntoColor, LinSrgb, Oklab, Srgb, Xyz};
use zenpixels_convert::ext::{ColorPrimariesExt, TransferFunctionExt};
use zenpixels_convert::gamut::apply_matrix_f32;
use zenpixels_convert::oklab::{lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab};
use zenpixels_convert::{ColorPrimaries, TransferFunction};

fn assert_close(a: f32, b: f32, tol: f32, label: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < tol,
        "{label}: ours={a:.8}, palette={b:.8}, diff={diff:.2e}"
    );
}

/// Test 1: sRGB transfer function cross-validation against palette.
///
/// Compares linearize and delinearize for a grid of representative values.
/// Tolerance is 1e-5: both implementations use the IEC 61966-2-1 piecewise
/// formula, but minor f32 evaluation order differences (e.g., powf
/// implementations, fused multiply-add) cause ~8e-6 divergence near the
/// linear/gamma knee at 0.04045.
#[test]
fn srgb_transfer_crossval() {
    let tf = TransferFunction::Srgb;
    let values = [0.0, 0.001, 0.003, 0.04045, 0.1, 0.2, 0.5, 0.73, 0.99, 1.0];

    for &v in &values {
        // Linearize
        let ours_lin = tf.linearize(v);
        let palette_lin = Srgb::new(v, v, v).into_linear().red;
        assert_close(ours_lin, palette_lin, 1e-5, &format!("linearize({v})"));

        // Delinearize
        let ours_delin = tf.delinearize(v);
        let palette_delin = Srgb::from_linear(LinSrgb::new(v, v, v)).red;
        assert_close(
            ours_delin,
            palette_delin,
            1e-5,
            &format!("delinearize({v})"),
        );
    }
}

/// Test 2: Exhaustive u8 sRGB transfer function validation.
///
/// Validates all 256 u8 values, ensuring our implementation (which may use
/// LUT acceleration) matches palette's precise formula.
/// Tolerance is 1e-5: minor f32 powf divergence (see test 1 comment).
#[test]
fn srgb_transfer_exhaustive_u8() {
    let tf = TransferFunction::Srgb;

    for i in 0..=255u8 {
        let v = i as f32 / 255.0;
        let ours = tf.linearize(v);
        let theirs = Srgb::new(v, v, v).into_linear().red;
        assert_close(ours, theirs, 1e-5, &format!("linearize(u8={i}, f={v:.6})"));
    }
}

/// Test 3: Oklab conversion cross-validation against palette.
///
/// For a grid of sRGB u8 values, converts to Oklab via both implementations
/// and compares L, a, b components.
///
/// Tolerance is 2e-4: our implementation uses a combined RGB->LMS matrix
/// (pre-multiplied from XYZ->LMS * RGB->XYZ in f32), while palette goes
/// through an intermediate XYZ step. The f32 matrix multiplication
/// accumulates rounding differences up to ~1e-4 near saturated primaries.
/// Our fast_cbrt (~22-bit precision) adds a small additional error vs.
/// palette's f32 cbrt.
#[test]
fn oklab_crossval() {
    let tf = TransferFunction::Srgb;
    let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();

    for ri in (0..=255).step_by(17) {
        for gi in (0..=255).step_by(17) {
            for bi in (0..=255).step_by(17) {
                let lr = tf.linearize(ri as f32 / 255.0);
                let lg = tf.linearize(gi as f32 / 255.0);
                let lb = tf.linearize(bi as f32 / 255.0);

                // Our implementation
                let [ok_l, ok_a, ok_b] = rgb_to_oklab(lr, lg, lb, &m1);

                // palette's implementation
                let p_oklab: Oklab = LinSrgb::new(lr, lg, lb).into_color();

                let label = format!("oklab({ri},{gi},{bi})");
                assert_close(ok_l, p_oklab.l, 2e-4, &format!("{label}.L"));
                assert_close(ok_a, p_oklab.a, 2e-4, &format!("{label}.a"));
                assert_close(ok_b, p_oklab.b, 2e-4, &format!("{label}.b"));
            }
        }
    }
}

/// Test 4: Oklab round-trip accuracy at u8 precision.
///
/// Converts sRGB u8 -> linear f32 -> Oklab -> linear f32 -> sRGB u8 and
/// verifies the round-trip error is at most 1 in the u8 domain.
#[test]
fn oklab_roundtrip_u8() {
    let tf = TransferFunction::Srgb;
    let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
    let m1_inv = lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();

    let mut max_err = 0i32;

    for ri in (0..=255).step_by(5) {
        for gi in (0..=255).step_by(5) {
            for bi in (0..=255).step_by(5) {
                let lr = tf.linearize(ri as f32 / 255.0);
                let lg = tf.linearize(gi as f32 / 255.0);
                let lb = tf.linearize(bi as f32 / 255.0);

                // Forward: linear RGB -> Oklab
                let [ok_l, ok_a, ok_b] = rgb_to_oklab(lr, lg, lb, &m1);

                // Inverse: Oklab -> linear RGB
                let [lr2, lg2, lb2] = oklab_to_rgb(ok_l, ok_a, ok_b, &m1_inv);

                // Back to sRGB u8
                let r2 = (tf.delinearize(lr2) * 255.0 + 0.5) as i32;
                let g2 = (tf.delinearize(lg2) * 255.0 + 0.5) as i32;
                let b2 = (tf.delinearize(lb2) * 255.0 + 0.5) as i32;

                let err_r = (r2 - ri).abs();
                let err_g = (g2 - gi).abs();
                let err_b = (b2 - bi).abs();
                let err = err_r.max(err_g).max(err_b);
                max_err = max_err.max(err);

                assert!(
                    err <= 1,
                    "Oklab roundtrip error > 1 at ({ri},{gi},{bi}): got ({r2},{g2},{b2}), \
                     err=({err_r},{err_g},{err_b})"
                );
            }
        }
    }

    eprintln!("Oklab u8 round-trip max error: {max_err}");
}

/// Test 5: XYZ matrix cross-validation against palette.
///
/// Converts linear sRGB to XYZ using our BT.709 to_xyz_matrix and compares
/// against palette's LinSrgb -> Xyz conversion.
///
/// Tolerance is 5e-4: both use the same BT.709/sRGB primaries and D65 white
/// point, but the matrix constants are derived independently and stored as
/// f32. Rounding in the least-significant bits of the matrix entries
/// accumulates up to ~2e-4 per component in the dot product, particularly
/// in the Z column where the blue primary coefficient is large.
#[test]
fn xyz_matrix_crossval() {
    let to_xyz = ColorPrimaries::Bt709.to_xyz_matrix().unwrap();

    // Grid of linear RGB values from 0.0 to 1.0 in steps of 0.1
    for ri in 0..=10 {
        for gi in 0..=10 {
            for bi in 0..=10 {
                let r = ri as f32 * 0.1;
                let g = gi as f32 * 0.1;
                let b = bi as f32 * 0.1;

                // Our implementation
                let mut rgb = [r, g, b];
                apply_matrix_f32(&mut rgb, to_xyz);

                // palette's implementation
                let p_xyz: Xyz = LinSrgb::new(r, g, b).into_color();

                let label = format!("xyz({r:.1},{g:.1},{b:.1})");
                assert_close(rgb[0], p_xyz.x, 5e-4, &format!("{label}.X"));
                assert_close(rgb[1], p_xyz.y, 5e-4, &format!("{label}.Y"));
                assert_close(rgb[2], p_xyz.z, 5e-4, &format!("{label}.Z"));
            }
        }
    }
}

/// Test 6: Gamut round-trip via XYZ, cross-validated with palette.
///
/// Converts linear sRGB -> XYZ -> linear sRGB using our matrices and
/// independently via palette, verifying both produce the same result.
#[test]
fn gamut_roundtrip_via_xyz() {
    let to_xyz = ColorPrimaries::Bt709.to_xyz_matrix().unwrap();
    let from_xyz = ColorPrimaries::Bt709.from_xyz_matrix().unwrap();

    let test_colors: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.3, 0.8],
        [0.18, 0.18, 0.18],
        [0.9, 0.1, 0.4],
    ];

    for &color in test_colors {
        let [r, g, b] = color;

        // Our round-trip: RGB -> XYZ -> RGB
        let mut ours = color;
        apply_matrix_f32(&mut ours, to_xyz);
        apply_matrix_f32(&mut ours, from_xyz);

        // palette round-trip: LinSrgb -> Xyz -> LinSrgb
        let p_xyz: Xyz = LinSrgb::new(r, g, b).into_color();
        let p_back: LinSrgb = p_xyz.into_color();

        let label = format!("gamut_rt({r:.2},{g:.2},{b:.2})");

        // Verify our round-trip recovers the original
        assert_close(ours[0], r, 1e-4, &format!("{label} ours.R"));
        assert_close(ours[1], g, 1e-4, &format!("{label} ours.G"));
        assert_close(ours[2], b, 1e-4, &format!("{label} ours.B"));

        // Verify palette's round-trip recovers the original
        assert_close(p_back.red, r, 1e-4, &format!("{label} palette.R"));
        assert_close(p_back.green, g, 1e-4, &format!("{label} palette.G"));
        assert_close(p_back.blue, b, 1e-4, &format!("{label} palette.B"));

        // Verify both implementations agree on the round-trip result
        assert_close(ours[0], p_back.red, 1e-4, &format!("{label} cross.R"));
        assert_close(ours[1], p_back.green, 1e-4, &format!("{label} cross.G"));
        assert_close(ours[2], p_back.blue, 1e-4, &format!("{label} cross.B"));
    }
}
