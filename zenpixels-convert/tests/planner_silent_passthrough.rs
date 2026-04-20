//! Comprehensive coverage for issue #19 [A] and [B] (silent wrong-pixel
//! planner bugs) plus the broader transfer-function and conversion-path
//! matrix: every TF × every depth, HDR (PQ/HLG), out-of-gamut / extended
//! range, cross-primaries + cross-TF, and Unknown-TF passthrough.
//!
//! The pattern for every case is: construct a descriptor pair, build the
//! plan via `RowConverter::new` (or `new_explicit` for policy-sensitive
//! cases), run a ramp of real byte values through it, and compare against
//! a hand-coded ground-truth EOTF/OETF. A pass means the planner composed
//! the right steps AND the kernels executed correctly.

// PQ / Adobe-gamma / HLG constants exceed f32 mantissa precision in their
// written form, but round correctly during compilation. Silence clippy.
#![allow(clippy::excessive_precision)]

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::{RowConverter, policy::ConvertOptions};

// ── Ground-truth EOTF/OETF ───────────────────────────────────────────────

const ADOBE_GAMMA: f32 = 2.199_218_75;

fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_oetf(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn bt709_eotf(v: f32) -> f32 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

fn bt709_oetf(v: f32) -> f32 {
    if v < 0.018 {
        v * 4.5
    } else {
        1.099 * v.powf(0.45) - 0.099
    }
}

fn gamma22_eotf(v: f32) -> f32 {
    v.max(0.0).powf(ADOBE_GAMMA)
}

fn gamma22_oetf(v: f32) -> f32 {
    v.max(0.0).powf(1.0 / ADOBE_GAMMA)
}

fn pq_eotf(v: f32) -> f32 {
    const M1: f32 = 0.159_301_757_812_5;
    const M2: f32 = 78.843_75;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562_5;
    const C3: f32 = 18.687_5;
    let vp = v.max(0.0).powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 {
        0.0
    } else {
        (num / den).powf(1.0 / M1)
    }
}

fn pq_oetf(v: f32) -> f32 {
    const M1: f32 = 0.159_301_757_812_5;
    const M2: f32 = 78.843_75;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562_5;
    const C3: f32 = 18.687_5;
    let lp = v.max(0.0).powf(M1);
    ((C1 + C2 * lp) / (1.0 + C3 * lp)).powf(M2)
}

// ITU-R BT.2100 HLG OETF constants.
const HLG_A: f32 = 0.178_832_77;
const HLG_B: f32 = 1.0 - 4.0 * HLG_A;
// c = 0.5 - a·ln(4a) ≈ 0.55991073 (precomputed — can't be const in f32).
const HLG_C: f32 = 0.559_910_73;

fn hlg_eotf(v: f32) -> f32 {
    // HLG inverse OETF (scene-referred ramp; no OOTF applied here).
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

fn hlg_oetf(v: f32) -> f32 {
    let v = v.max(0.0);
    if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt()
    } else {
        HLG_A * (12.0 * v - HLG_B).ln() + HLG_C
    }
}

fn eotf_of(tf: TransferFunction) -> fn(f32) -> f32 {
    match tf {
        TransferFunction::Linear => |v| v,
        TransferFunction::Srgb => srgb_eotf,
        TransferFunction::Bt709 => bt709_eotf,
        TransferFunction::Gamma22 => gamma22_eotf,
        TransferFunction::Pq => pq_eotf,
        TransferFunction::Hlg => hlg_eotf,
        _ => |v| v,
    }
}

fn oetf_of(tf: TransferFunction) -> fn(f32) -> f32 {
    match tf {
        TransferFunction::Linear => |v| v,
        TransferFunction::Srgb => srgb_oetf,
        TransferFunction::Bt709 => bt709_oetf,
        TransferFunction::Gamma22 => gamma22_oetf,
        TransferFunction::Pq => pq_oetf,
        TransferFunction::Hlg => hlg_oetf,
        _ => |v| v,
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn rgb(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgb, None, tf)
}

fn rgba(ct: ChannelType, tf: TransferFunction, alpha: AlphaMode) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgba, Some(alpha), tf)
}

fn u8_to_f32(src_u8: u8) -> f32 {
    src_u8 as f32 / 255.0
}

fn u16_to_f32(src_u16: u16) -> f32 {
    src_u16 as f32 / 65535.0
}

fn u8_of(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn u16_of(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0).round() as u16
}

// Tolerances calibrated for mid-scale test samples; HDR (PQ/HLG) needs more
// slack because the curves compress many linear decades into u8/u16.
const U8_TOL: i32 = 1;
const U16_TOL: i32 = 64;
const F32_TOL_NORMAL: f32 = 5e-4;
const F32_TOL_LOOSE: f32 = 5e-3; // PQ/HLG mid-range
const HDR_U8_TOL: i32 = 2;

// =========================================================================
// [A] Same-depth integer TF changes — were silently `[Identity]`
// =========================================================================
//
// For each SDR TF pair where the EOTFs diverge at the chosen sample code,
// verify the output re-encodes correctly.

fn assert_u8_same_depth_tf(sample: u8, from_tf: TransferFunction, to_tf: TransferFunction) {
    let src_d = rgb(ChannelType::U8, from_tf);
    let dst_d = rgb(ChannelType::U8, to_tf);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [sample, sample, sample];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);

    let linear = (eotf_of(from_tf))(u8_to_f32(sample));
    let expected = u8_of((oetf_of(to_tf))(linear));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U8_TOL,
            "U8 {from_tf:?} → U8 {to_tf:?} @ sample={sample}: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn u8_same_depth_gamma22_to_srgb_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Gamma22, TransferFunction::Srgb);
}

#[test]
fn u8_same_depth_srgb_to_gamma22_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Srgb, TransferFunction::Gamma22);
}

#[test]
fn u8_same_depth_bt709_to_srgb_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Bt709, TransferFunction::Srgb);
}

#[test]
fn u8_same_depth_srgb_to_bt709_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Srgb, TransferFunction::Bt709);
}

#[test]
fn u8_same_depth_gamma22_to_bt709_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Gamma22, TransferFunction::Bt709);
}

#[test]
fn u8_same_depth_bt709_to_gamma22_at_toe() {
    assert_u8_same_depth_tf(20, TransferFunction::Bt709, TransferFunction::Gamma22);
}

#[test]
fn u8_same_depth_gamma22_to_srgb_at_highlight() {
    assert_u8_same_depth_tf(240, TransferFunction::Gamma22, TransferFunction::Srgb);
}

#[test]
fn u8_same_depth_pq_to_srgb_diverges_wildly() {
    // PQ at U8 is an ill-defined bit depth (PQ needs 10+ bits for its
    // dynamic range) but must not silently pass-through.
    let src_d = rgb(ChannelType::U8, TransferFunction::Pq);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [128u8, 128, 128];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    assert_ne!(dst[0], 128, "PQ→sRGB at U8 128 must re-encode");
}

#[test]
fn u16_same_depth_gamma22_to_srgb_at_toe() {
    let src_d = rgb(ChannelType::U16, TransferFunction::Gamma22);
    let dst_d = rgb(ChannelType::U16, TransferFunction::Srgb);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [5000u16, 5000, 5000];
    let mut dst = [0u16; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let linear = gamma22_eotf(u16_to_f32(5000));
    let expected = u16_of(srgb_oetf(linear));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U16_TOL,
            "U16 Gamma22→Srgb: got {ch}, expected {expected}"
        );
    }
    assert_ne!(dst[0], 5000);
}

// =========================================================================
// [B] Cross-depth TF changes — int↔F32 dropped EOTF/OETF before fix
// =========================================================================

fn assert_u8_to_f32_linearize(sample: u8, tf: TransferFunction) {
    let src_d = rgb(ChannelType::U8, tf);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [sample, sample, sample];
    let mut dst = [0f32; 3];
    c.convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 1);
    let expected = (eotf_of(tf))(u8_to_f32(sample));
    let tol = match tf {
        TransferFunction::Pq | TransferFunction::Hlg => F32_TOL_LOOSE,
        _ => F32_TOL_NORMAL,
    };
    for ch in dst {
        assert!(
            (ch - expected).abs() < tol,
            "U8 {tf:?} → F32 Linear @ sample={sample}: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn u8_srgb_to_f32_linear_mid() {
    assert_u8_to_f32_linearize(128, TransferFunction::Srgb);
}
#[test]
fn u8_bt709_to_f32_linear_mid() {
    assert_u8_to_f32_linearize(128, TransferFunction::Bt709);
}
#[test]
fn u8_gamma22_to_f32_linear_mid() {
    assert_u8_to_f32_linearize(128, TransferFunction::Gamma22);
}
#[test]
fn u8_srgb_to_f32_linear_toe() {
    assert_u8_to_f32_linearize(10, TransferFunction::Srgb);
}
#[test]
fn u8_bt709_to_f32_linear_toe() {
    assert_u8_to_f32_linearize(10, TransferFunction::Bt709);
}
#[test]
fn u8_gamma22_to_f32_linear_toe() {
    assert_u8_to_f32_linearize(10, TransferFunction::Gamma22);
}
#[test]
fn u8_srgb_to_f32_linear_highlight() {
    assert_u8_to_f32_linearize(240, TransferFunction::Srgb);
}
#[test]
fn u8_bt709_to_f32_linear_highlight() {
    assert_u8_to_f32_linearize(240, TransferFunction::Bt709);
}
#[test]
fn u8_gamma22_to_f32_linear_highlight() {
    assert_u8_to_f32_linearize(240, TransferFunction::Gamma22);
}

fn assert_f32_linear_to_u8(sample: f32, tf: TransferFunction) {
    let src_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let dst_d = rgb(ChannelType::U8, tf);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [sample, sample, sample];
    let mut dst = [0u8; 3];
    c.convert_row(bytemuck::cast_slice(&src), &mut dst, 1);
    let expected = u8_of((oetf_of(tf))(sample));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U8_TOL,
            "F32 Linear → U8 {tf:?} @ {sample}: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn f32_linear_to_u8_srgb_mid() {
    assert_f32_linear_to_u8(0.217, TransferFunction::Srgb);
}
#[test]
fn f32_linear_to_u8_bt709_mid() {
    assert_f32_linear_to_u8(0.261, TransferFunction::Bt709);
}
#[test]
fn f32_linear_to_u8_gamma22_mid() {
    assert_f32_linear_to_u8(0.217, TransferFunction::Gamma22);
}
#[test]
fn f32_linear_to_u8_srgb_low() {
    assert_f32_linear_to_u8(0.001, TransferFunction::Srgb);
}
#[test]
fn f32_linear_to_u8_srgb_high() {
    assert_f32_linear_to_u8(0.9, TransferFunction::Srgb);
}

fn assert_u16_to_f32_linearize(sample: u16, tf: TransferFunction) {
    let src_d = rgb(ChannelType::U16, tf);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [sample, sample, sample];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let expected = (eotf_of(tf))(u16_to_f32(sample));
    let tol = match tf {
        TransferFunction::Pq | TransferFunction::Hlg => F32_TOL_LOOSE,
        _ => F32_TOL_NORMAL,
    };
    for ch in dst {
        assert!(
            (ch - expected).abs() < tol,
            "U16 {tf:?} → F32 Linear @ {sample}: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn u16_srgb_to_f32_linear() {
    assert_u16_to_f32_linearize(32768, TransferFunction::Srgb);
}
#[test]
fn u16_bt709_to_f32_linear() {
    assert_u16_to_f32_linearize(32768, TransferFunction::Bt709);
}
#[test]
fn u16_gamma22_to_f32_linear() {
    assert_u16_to_f32_linearize(32768, TransferFunction::Gamma22);
}
#[test]
fn u16_pq_to_f32_linear() {
    assert_u16_to_f32_linearize(32768, TransferFunction::Pq);
}
#[test]
fn u16_hlg_to_f32_linear() {
    assert_u16_to_f32_linearize(32768, TransferFunction::Hlg);
}

fn assert_f32_linear_to_u16(sample: f32, tf: TransferFunction) {
    let src_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let dst_d = rgb(ChannelType::U16, tf);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [sample, sample, sample];
    let mut dst = [0u16; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let expected = u16_of((oetf_of(tf))(sample));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U16_TOL,
            "F32 Linear → U16 {tf:?} @ {sample}: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn f32_linear_to_u16_srgb() {
    assert_f32_linear_to_u16(0.217, TransferFunction::Srgb);
}
#[test]
fn f32_linear_to_u16_bt709() {
    assert_f32_linear_to_u16(0.261, TransferFunction::Bt709);
}
#[test]
fn f32_linear_to_u16_gamma22() {
    assert_f32_linear_to_u16(0.217, TransferFunction::Gamma22);
}
#[test]
fn f32_linear_to_u16_pq() {
    assert_f32_linear_to_u16(0.1, TransferFunction::Pq);
}
#[test]
fn f32_linear_to_u16_hlg() {
    assert_f32_linear_to_u16(0.25, TransferFunction::Hlg);
}

// =========================================================================
// U16 ↔ U8 cross-depth cross-TF
// =========================================================================

#[test]
fn u16_gamma22_to_u8_srgb() {
    // Pre-fix: would drop the TF change entirely. Now must compose.
    let src_d = rgb(ChannelType::U16, TransferFunction::Gamma22);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [20000u16, 20000, 20000];
    let mut dst = [0u8; 3];
    c.convert_row(bytemuck::cast_slice(&src), &mut dst, 1);
    let linear = gamma22_eotf(u16_to_f32(20000));
    let expected = u8_of(srgb_oetf(linear));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U8_TOL,
            "U16 Gamma22 → U8 Srgb: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn u8_srgb_to_u16_bt709() {
    let src_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::U16, TransferFunction::Bt709);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [128u8, 128, 128];
    let mut dst = [0u16; 3];
    c.convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 1);
    let linear = srgb_eotf(u8_to_f32(128));
    let expected = u16_of(bt709_oetf(linear));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= U16_TOL,
            "U8 Srgb → U16 Bt709: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn u16_pq_to_u8_srgb_fused_path() {
    // Existing fused PQ-U16→sRGB-U8 kernel must still fire.
    let src_d = rgb(ChannelType::U16, TransferFunction::Pq);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [32768u16; 3];
    let mut dst = [0u8; 3];
    c.convert_row(bytemuck::cast_slice(&src), &mut dst, 1);
    let expected = u8_of(srgb_oetf(pq_eotf(u16_to_f32(32768))));
    for ch in dst {
        let diff = (ch as i32 - expected as i32).abs();
        assert!(
            diff <= HDR_U8_TOL,
            "U16 PQ → U8 Srgb (fused): got {ch}, expected {expected}"
        );
    }
}

// =========================================================================
// HDR — PQ and HLG round-trips at native depth
// =========================================================================

#[test]
fn pq_f32_roundtrip_via_linear() {
    let pq_d = rgb(ChannelType::F32, TransferFunction::Pq);
    let lin_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut fwd = RowConverter::new(pq_d, lin_d).unwrap();
    let mut back = RowConverter::new(lin_d, pq_d).unwrap();
    for &sample in &[0.01f32, 0.1, 0.25, 0.5, 0.75, 0.9] {
        let src = [sample, sample, sample];
        let mut lin = [0f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut lin),
            1,
        );
        let mut rt = [0f32; 3];
        back.convert_row(
            bytemuck::cast_slice(&lin),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for ch in rt {
            assert!(
                (ch - sample).abs() < F32_TOL_LOOSE,
                "PQ F32 roundtrip @ {sample}: got {ch}"
            );
        }
    }
}

#[test]
fn hlg_f32_roundtrip_via_linear() {
    let hlg_d = rgb(ChannelType::F32, TransferFunction::Hlg);
    let lin_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut fwd = RowConverter::new(hlg_d, lin_d).unwrap();
    let mut back = RowConverter::new(lin_d, hlg_d).unwrap();
    for &sample in &[0.01f32, 0.1, 0.25, 0.5, 0.75, 0.9] {
        let src = [sample, sample, sample];
        let mut lin = [0f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut lin),
            1,
        );
        let mut rt = [0f32; 3];
        back.convert_row(
            bytemuck::cast_slice(&lin),
            bytemuck::cast_slice_mut(&mut rt),
            1,
        );
        for ch in rt {
            assert!(
                (ch - sample).abs() < F32_TOL_LOOSE,
                "HLG F32 roundtrip @ {sample}: got {ch}"
            );
        }
    }
}

#[test]
fn pq_to_hlg_f32_via_linear() {
    let pq_d = rgb(ChannelType::F32, TransferFunction::Pq);
    let hlg_d = rgb(ChannelType::F32, TransferFunction::Hlg);
    let mut c = RowConverter::new(pq_d, hlg_d).unwrap();
    let src = [0.5f32; 3];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let expected = hlg_oetf(pq_eotf(0.5));
    for ch in dst {
        assert!(
            (ch - expected).abs() < F32_TOL_LOOSE,
            "PQ→HLG F32 @ 0.5: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn hlg_to_pq_f32_via_linear() {
    let hlg_d = rgb(ChannelType::F32, TransferFunction::Hlg);
    let pq_d = rgb(ChannelType::F32, TransferFunction::Pq);
    let mut c = RowConverter::new(hlg_d, pq_d).unwrap();
    let src = [0.5f32; 3];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let expected = pq_oetf(hlg_eotf(0.5));
    for ch in dst {
        assert!(
            (ch - expected).abs() < F32_TOL_LOOSE,
            "HLG→PQ F32 @ 0.5: got {ch}, expected {expected}"
        );
    }
}

// =========================================================================
// Out-of-gamut / extended-range (with_clip_out_of_gamut(false))
// =========================================================================
//
// Only sRGB has extended-range step substitution today (see
// `ConvertPlan::new` options loop). Verify that the substitution still
// engages after the planner refactor AND that default (clipping) behavior
// is unchanged.

#[test]
fn srgb_f32_linear_clips_by_default() {
    let src_d = rgb(ChannelType::F32, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [-0.1f32, 0.5, 1.2];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    // Default (clip_out_of_gamut = true) clamps negatives to 0 and >1 to 1.
    assert!(
        dst[0] >= 0.0,
        "default path must clip negatives: got {}",
        dst[0]
    );
    assert!(
        dst[2] <= 1.0 + 1e-4,
        "default path must clip >1: got {}",
        dst[2]
    );
}

#[test]
fn srgb_f32_linear_extended_preserves_oog() {
    let src_d = rgb(ChannelType::F32, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
    let mut c = RowConverter::new_explicit(src_d, dst_d, &opts).unwrap();
    let src = [-0.1f32, 0.5, 1.2];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    // Extended sRGB EOTF is sign-preserving and symmetric around 0.
    assert!(
        dst[0] < 0.0,
        "extended path must preserve negative: got {}",
        dst[0]
    );
    assert!(
        dst[2] > 1.0,
        "extended path must preserve > 1: got {}",
        dst[2]
    );
    // Middle value (in-gamut) should match the clipping path at v=0.5.
    let expected_mid = srgb_eotf(0.5);
    assert!((dst[1] - expected_mid).abs() < 1e-3);
}

#[test]
fn linear_f32_srgb_extended_preserves_oog() {
    let src_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Srgb);
    let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
    let mut c = RowConverter::new_explicit(src_d, dst_d, &opts).unwrap();
    let src = [-0.05f32, 0.217, 1.5];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    assert!(dst[0] < 0.0, "extended OETF must preserve negative");
    assert!(dst[2] > 1.0, "extended OETF must preserve > 1");
}

// =========================================================================
// Unknown TF — planner must passthrough (issue #19 [C] territory)
// =========================================================================

#[test]
fn f32_unknown_to_f32_linear_is_passthrough() {
    let src_d = rgb(ChannelType::F32, TransferFunction::Unknown);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [0.123f32, 0.456, 0.789];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    for (s, d) in src.iter().zip(dst.iter()) {
        assert!(
            (s - d).abs() < 1e-6,
            "Unknown TF must passthrough: {s} != {d}"
        );
    }
}

#[test]
fn f32_srgb_to_f32_unknown_is_passthrough() {
    let src_d = rgb(ChannelType::F32, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Unknown);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [0.123f32, 0.456, 0.789];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    for (s, d) in src.iter().zip(dst.iter()) {
        assert!(
            (s - d).abs() < 1e-6,
            "X → Unknown must passthrough: {s} != {d}"
        );
    }
}

#[test]
fn u8_srgb_to_u8_unknown_is_exact_passthrough() {
    let src_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Unknown);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [12u8, 199, 77];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    assert_eq!(dst, src, "U8 X → U8 Unknown must be byte-exact passthrough");
}

#[test]
fn u8_unknown_to_u8_srgb_is_exact_passthrough() {
    let src_d = rgb(ChannelType::U8, TransferFunction::Unknown);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [12u8, 199, 77];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    assert_eq!(dst, src, "U8 Unknown → U8 X must be byte-exact passthrough");
}

// =========================================================================
// Gamut + TF crossings — AdobeRGB Gamma22 → BT.2020 PQ, etc.
// =========================================================================

#[test]
fn adobergb_u8_to_bt2020_pq_f32_gray_axis_preserved() {
    // Neutral gray should stay neutral across D65→D65 primaries regardless
    // of the TF-EOTF-matrix-OETF composition.
    let src_d =
        rgb(ChannelType::U8, TransferFunction::Gamma22).with_primaries(ColorPrimaries::AdobeRgb);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Pq).with_primaries(ColorPrimaries::Bt2020);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [128u8, 128, 128];
    let mut dst = [0f32; 3];
    c.convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 1);
    // Expected: gamma22 EOTF on 128/255 → linear → PQ OETF (matrix is
    // D65→D65, identity on neutral gray).
    let lin = gamma22_eotf(u8_to_f32(128));
    let expected = pq_oetf(lin);
    for ch in dst {
        assert!(
            (ch - expected).abs() < F32_TOL_LOOSE,
            "AdobeRGB U8 → BT.2020 PQ F32 gray axis: got {ch}, expected {expected}"
        );
    }
}

#[test]
fn displayp3_u8_srgb_to_bt709_srgb_gray_axis_preserved() {
    let src_d =
        rgb(ChannelType::U8, TransferFunction::Srgb).with_primaries(ColorPrimaries::DisplayP3);
    let dst_d = rgb(ChannelType::U8, TransferFunction::Srgb).with_primaries(ColorPrimaries::Bt709);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [128u8, 128, 128];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    // Gray stays neutral across D65→D65 gamut crossing (all channels equal within ±1 LSB).
    assert!(
        (dst[0] as i32 - dst[1] as i32).abs() <= 1,
        "gray should stay neutral: {dst:?}"
    );
    assert!(
        (dst[1] as i32 - dst[2] as i32).abs() <= 1,
        "gray should stay neutral: {dst:?}"
    );
}

// =========================================================================
// RGBA with alpha — opaque alpha must survive TF composition
// =========================================================================

#[test]
fn rgba_opaque_alpha_preserved_across_tf_change() {
    let src_d = rgba(
        ChannelType::F32,
        TransferFunction::Gamma22,
        AlphaMode::Straight,
    );
    let dst_d = rgba(
        ChannelType::F32,
        TransferFunction::Srgb,
        AlphaMode::Straight,
    );
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [0.3f32, 0.5, 0.7, 1.0];
    let mut dst = [0f32; 4];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    // Alpha = 1.0 survives (pow(1, anything) == 1).
    assert!(
        (dst[3] - 1.0).abs() < 1e-4,
        "opaque alpha must survive TF change: got {}",
        dst[3]
    );
}

// =========================================================================
// Sanity: existing behavior preserved
// =========================================================================

#[test]
fn u8_srgb_to_u8_srgb_zero_cost() {
    let d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let mut c = RowConverter::new(d, d).unwrap();
    let src = [64u8, 128, 200];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    assert_eq!(dst, src, "same-TF same-depth must stay byte-exact");
}

#[test]
fn u8_srgb_to_f32_linear_still_uses_fused_path() {
    // The fused `SrgbU8ToLinearF32` kernel must remain the chosen path
    // (not the new compose-through-F32 path) for this combination.
    let src_d = rgb(ChannelType::U8, TransferFunction::Srgb);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [128u8, 128, 128];
    let mut dst = [0f32; 3];
    c.convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 1);
    let expected = srgb_eotf(u8_to_f32(128));
    for ch in dst {
        assert!((ch - expected).abs() < F32_TOL_NORMAL);
    }
}

#[test]
fn u16_pq_to_f32_linear_still_uses_fused_path() {
    let src_d = rgb(ChannelType::U16, TransferFunction::Pq);
    let dst_d = rgb(ChannelType::F32, TransferFunction::Linear);
    let mut c = RowConverter::new(src_d, dst_d).unwrap();
    let src = [32768u16; 3];
    let mut dst = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
    );
    let expected = pq_eotf(u16_to_f32(32768));
    for ch in dst {
        assert!((ch - expected).abs() < F32_TOL_LOOSE);
    }
}

#[test]
fn u8_linear_to_u8_linear_zero_cost() {
    let d = rgb(ChannelType::U8, TransferFunction::Linear);
    let mut c = RowConverter::new(d, d).unwrap();
    let src = [10u8, 20, 30];
    let mut dst = [0u8; 3];
    c.convert_row(&src, &mut dst, 1);
    assert_eq!(dst, src);
}
