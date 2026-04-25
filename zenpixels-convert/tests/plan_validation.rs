//! Plan-shape regression tests using the `trace_ops` feature.
//!
//! These tests assert that representative `(from, to, options)` tuples
//! produce a *minimal* op sequence: no redundant TF round-trips, no missing
//! linearization, no skipped depth changes. Lock the planner against
//! future drift.
//!
//! Run with `cargo test --features trace_ops`. Without the feature, the
//! tracer returns empty Vecs and assertions intentionally fail — these
//! tests are skipped entirely via `cfg(feature = "trace_ops")`.

#![cfg(feature = "trace_ops")]

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
    policy::{AlphaPolicy, ConvertOptions, LumaCoefficients},
};
use zenpixels_convert::{RowConverter, tracer};

fn rgba_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        tf,
    )
}
fn rgb_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgb, None, tf)
}
fn rgb_u16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U16, ChannelLayout::Rgb, None, tf)
}
fn rgba_u16(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        tf,
    )
}
fn rgb_u8_p(tf: TransferFunction, primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgb, None, tf).with_primaries(primaries)
}
fn gray_u8(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Gray, None, tf)
}

fn run_and_trace(
    src_desc: PixelDescriptor,
    dst_desc: PixelDescriptor,
    opts: ConvertOptions,
    src_bytes: &[u8],
    dst_len: usize,
    width: u32,
) -> (Vec<u8>, Vec<&'static str>) {
    let mut conv = RowConverter::new_explicit(src_desc, dst_desc, &opts).unwrap();
    let mut dst = vec![0u8; dst_len];
    tracer::start_recording();
    conv.convert_row(src_bytes, &mut dst, width);
    let trace = tracer::stop_recording();
    (dst, trace)
}

// ── No waste: identical source and destination is Identity (or empty) ────

#[test]
fn identical_descriptor_is_zero_or_identity() {
    let src = rgb_u8(TransferFunction::Srgb);
    let opts = ConvertOptions::permissive();
    let pixel = [10u8, 20, 30];
    let (_, trace) = run_and_trace(src, src, opts, &pixel, 3, 1);
    // Either no steps (pure passthrough) or a single Identity. Anything
    // else is wasted work for a no-op conversion.
    assert!(
        trace.is_empty() || (trace.len() == 1 && trace[0] == "Identity"),
        "identical-descriptor plan should be empty or [Identity], got {trace:?}"
    );
}

// ── No skip: TF change is mandatory; planner must linearize ────────────

#[test]
fn u8_srgb_to_u8_linear_round_trips_through_linear() {
    let src = rgb_u8(TransferFunction::Srgb);
    let dst = rgb_u8(TransferFunction::Linear);
    let opts = ConvertOptions::permissive();
    let pixel = [128u8, 128, 128];
    let (_, trace) = run_and_trace(src, dst, opts, &pixel, 3, 1);
    // Should at least linearize the source. Any plan that doesn't
    // touch a Linear step is silently passing sRGB-encoded bytes
    // through as if they were Linear — the bug class issue #19 fixed.
    let touches_linear = trace.iter().any(|s| s.contains("Linear"));
    assert!(
        touches_linear,
        "u8 sRGB → u8 Linear must include a linearize step, got {trace:?}"
    );
}

#[test]
fn u8_bt709_to_u8_srgb_changes_tf() {
    let src = rgb_u8(TransferFunction::Bt709);
    let dst = rgb_u8(TransferFunction::Srgb);
    let opts = ConvertOptions::permissive();
    let pixel = [128u8, 128, 128];
    let (_, trace) = run_and_trace(src, dst, opts, &pixel, 3, 1);
    let touches_bt709 = trace.iter().any(|s| s.contains("Bt709"));
    let touches_srgb = trace.iter().any(|s| s.contains("Srgb"));
    assert!(
        touches_bt709 && touches_srgb,
        "u8 BT.709 → u8 sRGB must touch both BT.709 and sRGB TF steps, got {trace:?}"
    );
}

// ── No waste: same-TF same-depth gray emits a single RgbToGray ────────

#[test]
fn rgb_to_gray_same_tf_is_one_step() {
    let src = rgb_u8(TransferFunction::Srgb);
    let dst = gray_u8(TransferFunction::Srgb);
    let opts = ConvertOptions::permissive();
    let pixel = [200u8, 100, 50];
    let (_, trace) = run_and_trace(src, dst, opts, &pixel, 1, 1);
    // Exactly one RgbToGray, no TF round-trip.
    assert_eq!(trace, vec!["RgbToGray"], "got {trace:?}");
}

#[test]
fn rgba_to_gray_same_tf_is_one_step() {
    let src = rgba_u8(TransferFunction::Srgb);
    let dst = gray_u8(TransferFunction::Srgb);
    let opts = ConvertOptions::permissive();
    let pixel = [200u8, 100, 50, 255];
    let (_, trace) = run_and_trace(src, dst, opts, &pixel, 1, 1);
    assert_eq!(trace, vec!["RgbaToGray"], "got {trace:?}");
}

// ── No waste: matte composite is single-step (kernel handles TF inline) ──

#[test]
fn matte_composite_is_single_step_for_every_tf() {
    for tf in [
        TransferFunction::Srgb,
        TransferFunction::Linear,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Gamma22,
    ] {
        let src = rgba_u16(tf);
        let dst = rgb_u16(tf);
        let opts =
            ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
                r: 0,
                g: 0,
                b: 0,
            });
        let mut conv = RowConverter::new_explicit(src, dst, &opts).unwrap();
        let pixel: [u16; 4] = [32768, 32768, 32768, 32768];
        let src_bytes: [u8; 8] = bytemuck::cast(pixel);
        let mut dst_bytes = [0u8; 6];
        tracer::start_recording();
        conv.convert_row(&src_bytes, &mut dst_bytes, 1);
        let trace = tracer::stop_recording();
        assert_eq!(
            trace,
            vec!["MatteComposite"],
            "TF {tf:?}: matte composite must be one step (kernel handles TF inline), got {trace:?}"
        );
    }
}

// ── No skip: depth change is honored ───────────────────────────────────

#[test]
fn u16_to_u8_includes_depth_step() {
    let src = rgb_u16(TransferFunction::Srgb);
    let dst = rgb_u8(TransferFunction::Srgb);
    let opts = ConvertOptions::permissive();
    let pixel: [u16; 3] = [32768, 32768, 32768];
    let src_bytes: [u8; 6] = bytemuck::cast(pixel);
    let (_, trace) = run_and_trace(src, dst, opts, &src_bytes, 3, 1);
    // Some depth step must fire — exact name varies by fused/non-fused
    // path but it must NOT be just Identity or empty.
    assert!(
        !trace.is_empty() && trace.iter().any(|s| s.contains("U16") || s.contains("U8")),
        "u16 → u8 plan must include a depth conversion step, got {trace:?}"
    );
}

// ── No waste: sRGB→sRGB U8 with primaries change uses fused matlut ────

#[test]
fn srgb_u8_p3_to_srgb_u8_bt709_uses_fused_matlut() {
    let src = rgb_u8_p(TransferFunction::Srgb, ColorPrimaries::DisplayP3);
    let dst = rgb_u8_p(TransferFunction::Srgb, ColorPrimaries::Bt709);
    let opts = ConvertOptions::permissive();
    let pixel = [200u8, 100, 50];
    let (_, trace) = run_and_trace(src, dst, opts, &pixel, 3, 1);
    // The fused path replaces [SrgbU8ToLinearF32, GamutMatrixRgbF32, LinearF32ToSrgbU8]
    // with a single FusedSrgbU8GamutRgb step. Any decomposition into the
    // 3-step sequence is wasted work.
    let saw_fused = trace.iter().any(|s| s.contains("FusedSrgb"));
    let saw_unfused_pair = trace.iter().any(|s| *s == "SrgbU8ToLinearF32")
        && trace.iter().any(|s| *s == "LinearF32ToSrgbU8");
    assert!(
        saw_fused || !saw_unfused_pair,
        "sRGB U8 P3→BT.709 should use fused matlut, not the 3-step decomposition. got {trace:?}"
    );
}

// ── No-double-linearize regression for matte composite ──────────────────

#[test]
fn matte_composite_does_not_double_linearize_floats() {
    // F32 sRGB RGBA → F32 sRGB RGB CompositeOnto.
    // The kernel handles TF internally. Planner should NOT also wrap with
    // SrgbF32ToLinearF32 / LinearF32ToSrgbF32 — that would be double-work
    // (and break alpha math, which is why we don't).
    let src = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let dst =
        PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, TransferFunction::Srgb);
    let opts =
        ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
            r: 0,
            g: 0,
            b: 0,
        });
    let mut conv = RowConverter::new_explicit(src, dst, &opts).unwrap();
    let pixel: [f32; 4] = [0.5, 0.5, 0.5, 0.5];
    let src_bytes: [u8; 16] = bytemuck::cast(pixel);
    let mut dst_bytes = [0u8; 12];
    tracer::start_recording();
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let trace = tracer::stop_recording();

    let saw_linearize = trace
        .iter()
        .any(|s| *s == "SrgbF32ToLinearF32" || *s == "SrgbF32ToLinearF32Extended");
    assert!(
        !saw_linearize,
        "matte composite kernel handles TF internally; no external linearize wrap allowed. got {trace:?}"
    );
    assert!(
        trace.contains(&"MatteComposite"),
        "expected MatteComposite step, got {trace:?}"
    );
}

// ── Bug 1 lockdown: Linear U16 matte does not get spurious sRGB EOTF ────

#[test]
fn linear_u16_matte_composite_has_only_one_kernel() {
    // If anyone re-introduced the pre-fix sRGB-on-Linear EOTF, the
    // existing matte_composite_linearize.rs tests catch the wrong pixels.
    // This lockdown asserts at the *plan* level: no extra linearize
    // step before MatteComposite for Linear data.
    let src = rgba_u16(TransferFunction::Linear);
    let dst = rgb_u16(TransferFunction::Linear);
    let opts =
        ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
            r: 0,
            g: 0,
            b: 0,
        });
    let mut conv = RowConverter::new_explicit(src, dst, &opts).unwrap();
    let pixel: [u16; 4] = [32768; 4];
    let src_bytes: [u8; 8] = bytemuck::cast(pixel);
    let mut dst_bytes = [0u8; 6];
    tracer::start_recording();
    conv.convert_row(&src_bytes, &mut dst_bytes, 1);
    let trace = tracer::stop_recording();
    assert_eq!(
        trace,
        vec!["MatteComposite"],
        "Linear U16 matte composite should be a single step, got {trace:?}"
    );
}

// ── Bug 2(a) lockdown: gray plan honors luma — verify via plan ──────────

#[test]
fn rgb_to_gray_plan_records_resolved_luma() {
    use zenpixels_convert::ConvertPlan;

    for &coeffs in &[
        LumaCoefficients::Bt709,
        LumaCoefficients::Bt601,
        LumaCoefficients::Bt2020,
        LumaCoefficients::DisplayP3,
    ] {
        let plan = ConvertPlan::new_explicit(
            rgb_u8(TransferFunction::Srgb),
            gray_u8(TransferFunction::Srgb),
            &ConvertOptions::permissive().with_luma(Some(coeffs)),
        )
        .unwrap();
        let debug = format!("{:?}", plan);
        let needle = format!("coefficients: {:?}", coeffs);
        assert!(
            debug.contains(&needle),
            "plan should record resolved {coeffs:?} coefficients in steps; got {debug}"
        );
    }
}
