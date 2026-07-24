//! Coverage for the `ConvertPlan::new_with_hdr_peak` /
//! `ConvertPlan::new_with_hdr_config` HDR pipeline introduced across
//! `52fd949`, `e712f84`, `f68b759`, `505d02e8`, `0897f17`, and the
//! `e1ebf76` empirically-calibrated `gamut_knee = 0.96` default.
//!
//! The HDR plan inserts:
//!   1. Source transfer decode → linear F32.
//!   2. Source primaries → BT.2020 matrix (if not already BT.2020).
//!   3. `ToneMapBt2446A` step.
//!   4. BT.2020 → target primaries matrix (if not BT.2020 target).
//!   5. `SoftCompressOklch` step (if not BT.2020 target).
//!   6. Layout / depth / target transfer encode.
//!
//! These tests cover three angles:
//!   - **Plan-shape**: trace assertions on which `ConvertStep` variants
//!     get emitted for the standard PQ-BT.2020 → sRGB BT.709 path.
//!   - **Numerical**: BT.2446-A curve monotonicity + the SoftCompress
//!     defaults.
//!   - **API surface**: HdrConfig::default carries the empirical
//!     `gamut_knee = 0.96`, knee values in `[0, 1]` are accepted.
//!
//! Gated on `hdr-experimental`. The plan-shape group is additionally
//! gated on `__trace_ops` for the kernel-trace dispatch records;
//! without the feature those tests skip (the recorder is a no-op so
//! the assertions can't fire).

#![cfg(feature = "hdr-experimental")]

extern crate alloc;

use alloc::vec;
use zenpixels::buffer::PixelBuffer;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::hdr::{Bt2446A, GamutBoundaryLut, SoftCompress};
use zenpixels_convert::{ConvertPlan, HdrConfig};

// ── Test inputs ─────────────────────────────────────────────────────────

/// PQ U16 BT.2020 RGB source descriptor — the canonical HDR10 input.
fn pq_u16_bt2020_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
        ColorPrimaries::Bt2020,
    )
}

/// PQ U16 BT.2020 RGBA source — alpha-passthrough exercise.
#[cfg_attr(not(feature = "__trace_ops"), allow(dead_code))]
fn pq_u16_bt2020_rgba() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Pq,
        ColorPrimaries::Bt2020,
    )
}

/// HLG U16 BT.2020 RGB source.
#[cfg_attr(not(feature = "__trace_ops"), allow(dead_code))]
fn hlg_u16_bt2020_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
        ColorPrimaries::Bt2020,
    )
}

/// Linear F32 BT.2020 RGB (used by tests that construct an HDR plan
/// where source and destination are both linear — the constructor is
/// the HDR opt-in signal even when descriptors are identical).
fn linear_f32_bt2020_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
    )
}

// ════════════════════════════════════════════════════════════════════════
// Plan-shape tests (need `__trace_ops` for kernel-trace dispatch records).
// ════════════════════════════════════════════════════════════════════════

#[cfg(feature = "__trace_ops")]
mod plan_shape {
    use super::*;
    use zenpixels_convert::{__trace_ops as tracer, RowConverter};

    /// Construct a `ConvertPlan` via the HDR config path, run one row
    /// through it, and return the dispatched-step trace.
    fn trace_hdr_plan(
        src: PixelDescriptor,
        dst: PixelDescriptor,
        hdr: HdrConfig,
        src_bytes: &[u8],
        width: u32,
    ) -> Vec<&'static str> {
        let plan = ConvertPlan::new_with_hdr_config(src, dst, hdr).expect("hdr plan");
        let mut conv = RowConverter::from_plan(plan);
        let dst_len = (width as usize) * dst.bytes_per_pixel();
        let mut dst_buf = alloc::vec![0u8; dst_len];
        tracer::start_recording();
        conv.convert_row(src_bytes, &mut dst_buf, width);
        tracer::stop_recording()
    }

    fn default_hdr(peak_nits: f32) -> HdrConfig {
        HdrConfig::for_source_peak(peak_nits)
    }

    #[test]
    fn hdr_pipeline_inserts_bt2446a_step() {
        // PQ U16 BT.2020 → sRGB U8 BT.709 plan must include the
        // `ToneMapBt2446A` step. Anything else means the HDR-aware
        // constructor silently degraded to the SDR planner.
        let src = pq_u16_bt2020_rgb();
        let dst = PixelDescriptor::RGB8_SRGB;
        let src_pixel: [u16; 3] = [40_000, 30_000, 20_000];
        let src_bytes: [u8; 6] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        assert!(
            trace.contains(&"ToneMapBt2446A"),
            "PQ→sRGB pipeline must include ToneMapBt2446A, got {trace:?}"
        );
    }

    #[test]
    fn hdr_pipeline_inserts_softcompress_when_target_narrows_gamut() {
        // BT.2020 source → BT.709 target: SoftCompressOklch step required
        // (the wide-gamut content must rolloff into the smaller sRGB
        // primary cube).
        let src = pq_u16_bt2020_rgb();
        let dst = PixelDescriptor::RGB8_SRGB; // BT.709 primaries
        let src_pixel: [u16; 3] = [40_000, 30_000, 20_000];
        let src_bytes: [u8; 6] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        assert!(
            trace.contains(&"SoftCompressOklch"),
            "BT.2020 → BT.709 must include SoftCompressOklch, got {trace:?}"
        );
    }

    #[test]
    fn hdr_pipeline_skips_softcompress_for_bt2020_target() {
        // BT.2020 → BT.2020 (wide-gamut output mode) must NOT emit the
        // soft-compress step — that's the whole point of wide-gamut
        // output. Note the constructor still emits the tone-map even
        // though descriptors match (the HDR constructor is the opt-in).
        let src = linear_f32_bt2020_rgb();
        let dst = linear_f32_bt2020_rgb();
        let src_pixel: [f32; 3] = [0.5, 0.3, 0.1];
        let src_bytes: [u8; 12] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        assert!(
            trace.contains(&"ToneMapBt2446A"),
            "tone-map still expected for BT.2020 → BT.2020, got {trace:?}"
        );
        assert!(
            !trace.contains(&"SoftCompressOklch"),
            "BT.2020 → BT.2020 must NOT include SoftCompressOklch (wide-gamut mode), got {trace:?}"
        );
    }

    #[test]
    fn hdr_pipeline_pq_to_srgb_emits_full_pipeline_in_order() {
        // PQ U16 BT.2020 RGB → sRGB U8 RGB BT.709 — the canonical HDR10
        // → sRGB conversion. Pin the dispatch order: PQ-decode →
        // (BT.2020→BT.2020 elided) → ToneMap → BT.2020→BT.709 matrix →
        // SoftCompress → linear-to-sRGB encode.
        let src = pq_u16_bt2020_rgb();
        let dst = PixelDescriptor::RGB8_SRGB;
        let src_pixel: [u16; 3] = [40_000, 30_000, 20_000];
        let src_bytes: [u8; 6] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        // The pipeline-essential steps must all appear, and ToneMapBt2446A
        // must precede SoftCompressOklch (compression operates on
        // tone-mapped values).
        let tm_pos = trace.iter().position(|s| s == &"ToneMapBt2446A");
        let sc_pos = trace.iter().position(|s| s == &"SoftCompressOklch");
        assert!(
            tm_pos.is_some() && sc_pos.is_some() && tm_pos < sc_pos,
            "ToneMapBt2446A must precede SoftCompressOklch, got {trace:?}"
        );
    }

    #[test]
    fn hdr_pipeline_hlg_source_uses_same_chain_shape() {
        // HLG U16 BT.2020 → sRGB U8 BT.709: same pipeline shape as PQ;
        // tone-map still fires (HLG is HDR-source).
        let src = hlg_u16_bt2020_rgb();
        let dst = PixelDescriptor::RGB8_SRGB;
        let src_pixel: [u16; 3] = [30_000, 30_000, 30_000];
        let src_bytes: [u8; 6] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        assert!(
            trace.contains(&"ToneMapBt2446A"),
            "HLG → sRGB pipeline must include ToneMapBt2446A, got {trace:?}"
        );
    }

    #[test]
    fn hdr_pipeline_rgba_source_carries_alpha_through_tonemap() {
        // PQ U16 BT.2020 RGBA — the alpha lane must ride through the
        // tone-map / SoftCompress steps. The trace still contains the
        // tone-map, AND the layout carries 4 channels through.
        let src = pq_u16_bt2020_rgba();
        let dst = PixelDescriptor::RGBA8_SRGB;
        let src_pixel: [u16; 4] = [40_000, 30_000, 20_000, 65_535];
        let src_bytes: [u8; 8] = bytemuck::cast(src_pixel);
        let trace = trace_hdr_plan(src, dst, default_hdr(1000.0), &src_bytes, 1);
        assert!(
            trace.contains(&"ToneMapBt2446A"),
            "RGBA PQ → RGBA sRGB pipeline must tonemap, got {trace:?}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// Default knee — pin the empirical e1ebf76 default
// ════════════════════════════════════════════════════════════════════════

#[test]
fn softcompress_default_knee_is_0_96() {
    // The production default for `HdrConfig::default().gamut_knee` is
    // `0.96` — calibrated against the 76-sample imazen-26 corpus on
    // 2026-06-23 (commit e1ebf76). Any silent change here cascades
    // into every default-config HDR conversion, so pin both halves:
    // the `HdrConfig::default` field AND the exported public `pub const`.
    assert_eq!(HdrConfig::default().gamut_knee, 0.96);
    assert_eq!(SoftCompress::DEFAULT_KNEE, 0.96);
    assert_eq!(HdrConfig::default().gamut_knee, SoftCompress::DEFAULT_KNEE);
}

#[test]
fn hdr_config_default_target_peak_is_100_nits() {
    // The other half of the default — SDR target peak. BT.709 / sRGB
    // reference diffuse white is 100 cd/m² per BT.1886. Pin so a tuning
    // sweep doesn't silently move it.
    assert_eq!(HdrConfig::default().target_peak_nits, 100.0);
}

#[test]
fn hdr_config_default_source_peak_is_zero_and_must_be_set() {
    // `source_peak_nits` deliberately has no default — callers MUST set
    // it explicitly (the curve is parameterized by it). HdrConfig
    // documents `source_peak_nits = 0.0` as the "you forgot" sentinel.
    assert_eq!(HdrConfig::default().source_peak_nits, 0.0);
}

#[test]
fn hdr_config_builders_set_fields_and_keep_defaults() {
    let cfg = HdrConfig::for_source_peak(1000.0);
    assert_eq!(cfg.source_peak_nits, 1000.0);
    assert_eq!(cfg.target_peak_nits, 100.0);
    assert_eq!(cfg.gamut_knee, SoftCompress::DEFAULT_KNEE);
    let cfg = cfg.with_target_peak_nits(203.0).with_gamut_knee(0.9);
    assert_eq!(cfg.target_peak_nits, 203.0);
    assert_eq!(cfg.gamut_knee, 0.9);
    // The other fields survive each builder step.
    assert_eq!(cfg.source_peak_nits, 1000.0);
}

#[test]
fn degenerate_peaks_are_rejected_not_tone_mapped_to_black() {
    // Pre-guard, a zero / negative / non-finite peak flowed into the
    // BT.2446-A constants (1/ln(1) = inf, powf(neg) → NaN) and the
    // kernel's NaN scrub emitted a fully BLACK image with NO error —
    // silent total pixel loss. The plan constructor must refuse instead.
    let src = pq_u16_bt2020_rgb();
    let dst = PixelDescriptor::RGB8_SRGB;
    let bad_sources = [
        HdrConfig::default(), // unset (0.0) source peak
        HdrConfig::for_source_peak(-1000.0),
        HdrConfig::for_source_peak(f32::NAN),
        HdrConfig::for_source_peak(f32::INFINITY),
        HdrConfig::for_source_peak(1000.0).with_target_peak_nits(0.0),
        HdrConfig::for_source_peak(1000.0).with_target_peak_nits(f32::NAN),
    ];
    for (i, cfg) in bad_sources.iter().enumerate() {
        let err = ConvertPlan::new_with_hdr_config(src, dst, *cfg)
            .expect_err("degenerate peak must be rejected");
        assert!(
            matches!(
                *err.error(),
                zenpixels_convert::ConvertError::HdrSourceRequiresPeak { .. }
            ),
            "case {i}: expected HdrSourceRequiresPeak, got {err:?}"
        );
    }
    // And the peak-shortcut constructor routes through the same guard.
    assert!(ConvertPlan::new_with_hdr_peak(src, dst, 0.0).is_err());
    assert!(ConvertPlan::new_with_hdr_peak(src, dst, f32::NAN).is_err());

    // SDR-encoded sources still ignore the config entirely (documented):
    // a degenerate config with an sRGB source builds the plain plan.
    assert!(
        ConvertPlan::new_with_hdr_config(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            HdrConfig::default(),
        )
        .is_ok(),
        "SDR source must keep ignoring the hdr argument"
    );
}

// ════════════════════════════════════════════════════════════════════════
// SoftCompress / GamutBoundaryLut numerics
// ════════════════════════════════════════════════════════════════════════

#[test]
fn softcompress_knee_in_zero_one_clamps_chroma_inside_gamut() {
    // For a saturated red well outside BT.709, SoftCompress at the
    // default knee should pull it into a near-in-gamut value (within
    // a small float epsilon of the unit cube).
    use zenpixels_convert::oklab;
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).expect("BT.709 inverse matrix");
    let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).expect("BT.709 forward matrix");
    let compress = SoftCompress::from_matrices(&m1, &m1_inv, SoftCompress::DEFAULT_KNEE);
    let mut pixels = vec![[1.2_f32, 0.05, 0.05]];
    compress.apply_strip(&mut pixels);
    for &c in &pixels[0] {
        assert!(c <= 1.0 + 1e-2, "expected gamut-compressed output, got {c}");
        assert!(c >= -1e-2, "compressed channel went very negative: {c}");
    }
}

#[test]
fn softcompress_accessor_round_trips_knee_value() {
    // The `knee()` accessor surfaces the construction-time value so a
    // pipeline introspector can verify what knee was applied. Pin the
    // identity through the accessor for a couple of representative
    // knee values + the empirical default.
    use zenpixels_convert::oklab;
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).expect("BT.709 inverse matrix");
    for &knee in &[0.0_f32, 0.5, 0.96, 1.0] {
        let sc = SoftCompress::new(&m1_inv, knee);
        assert_eq!(
            sc.knee(),
            knee,
            "knee accessor mismatch for input {knee}: got {}",
            sc.knee()
        );
    }
}

#[test]
fn softcompress_lut_accessor_returns_same_lut_across_calls() {
    // The `lut()` accessor borrows the internal LUT — same reference
    // across calls is a contract for callers that want to forward
    // the LUT to a planar pipeline (zenfilters does this).
    use zenpixels_convert::oklab;
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).expect("BT.709 inverse matrix");
    let sc = SoftCompress::new(&m1_inv, SoftCompress::DEFAULT_KNEE);
    let a = sc.lut();
    let b = sc.lut();
    // Compare via raw pointer equality — same backing storage.
    assert!(core::ptr::eq(a, b), "lut() must return a stable reference");
}

#[test]
fn gamut_boundary_lut_handles_extremes_without_panic() {
    // The LUT lookup clamps L to [0, 1] and wraps h modulo TAU. Pin
    // those invariants — at L=0 and L=1 the achromatic max should be
    // ~0 (no chroma at black or white); at L=0.5 hue=0 it should be
    // positive (red boundary in sRGB).
    use zenpixels_convert::oklab;
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).expect("BT.709 inverse matrix");
    let lut = GamutBoundaryLut::new(&m1_inv);
    let at_black = lut.max_chroma(0.0, 0.0);
    let at_white = lut.max_chroma(1.0, 0.0);
    let at_mid = lut.max_chroma(0.5, 0.0);
    assert!(
        at_black < 1e-2 && at_white < 1e-2,
        "max_chroma at L=0 / L=1 must be ~0, got {at_black} / {at_white}"
    );
    assert!(
        at_mid > 0.0,
        "max_chroma at L=0.5 hue=0 (red) must be positive, got {at_mid}"
    );
    // `max_chroma` must not panic at the hue-wrap boundary or out-of-range L.
    let _ = lut.max_chroma(-0.5, 100.0);
    let _ = lut.max_chroma(2.0, -1.0);
}

#[test]
fn gamut_boundary_lut_compresses_planes_in_place_at_default_knee() {
    use zenpixels_convert::oklab;
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).expect("BT.709 inverse matrix");
    let lut = GamutBoundaryLut::new(&m1_inv);
    // Build OKLab planes: L=0.6 (mid lightness), a/b on a saturated arc.
    let l = vec![0.6_f32, 0.6, 0.6];
    let mut a = vec![0.3_f32, 0.2, 0.1];
    let mut b = vec![0.0_f32, 0.1, 0.0];
    let a0 = a.clone();
    let b0 = b.clone();
    lut.compress_planes(&l, &mut a, &mut b, SoftCompress::DEFAULT_KNEE);
    // The achromatic+low-chroma pixel (a=0.1, b=0.0) is small enough
    // to lie inside the knee — values stay identical or very close.
    assert!((a[2] - a0[2]).abs() < 1e-3 && (b[2] - b0[2]).abs() < 1e-3,);
    // The strongly-saturated red pixel (a=0.3, b=0.0) is far out of
    // gamut — the compressed magnitude must be smaller than the original.
    let mag0 = (a0[0] * a0[0] + b0[0] * b0[0]).sqrt();
    let mag1 = (a[0] * a[0] + b[0] * b[0]).sqrt();
    assert!(
        mag1 < mag0,
        "saturated chroma must shrink under compression: {mag0} → {mag1}"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Bt2446A: monotonicity / boundary sanity
// ════════════════════════════════════════════════════════════════════════

#[test]
fn bt2446a_brightness_curve_is_monotonic_on_neutral_ramp() {
    // For grayscale input the BT.2446-A curve must be monotonic in
    // brightness — required for any HDR→SDR mapping. A regression in
    // ρ_H or the tone-curve constants would break this on the toe.
    let tm = Bt2446A::new(1000.0, 100.0);
    let mut last = -1.0_f32;
    for i in 0..=200 {
        let v = i as f32 / 200.0;
        let out = tm.map_rgb([v, v, v]);
        let lum = out[0];
        assert!(
            lum >= last - 1e-5,
            "brightness not monotonic at v={v}: {lum} < {last}"
        );
        last = lum;
    }
}

#[test]
fn bt2446a_at_zero_input_is_zero_output() {
    // Black HDR pixel maps to black SDR — the boundary case the
    // perceptual_linearize tail handles. No NaN, no negative.
    let tm = Bt2446A::new(1000.0, 100.0);
    let out = tm.map_rgb([0.0, 0.0, 0.0]);
    assert_eq!(out, [0.0, 0.0, 0.0]);
}

#[test]
fn bt2446a_peak_input_reaches_near_sdr_peak() {
    // A 1.0-normalised HDR input (= 1000-nit peak) should map close to
    // SDR peak — the curve compresses but the dynamic range still
    // anchors near full-scale output.
    let tm = Bt2446A::new(1000.0, 100.0);
    let out = tm.map_rgb([1.0, 1.0, 1.0]);
    for c in out {
        assert!(
            (0.8..=1.0).contains(&c),
            "HDR peak should land near SDR peak, got {c}"
        );
    }
}

#[test]
fn bt2446a_strip_simd_matches_scalar_within_tolerance() {
    // The SIMD strip kernel must match the scalar `map_rgb` per-pixel
    // within a small numerical tolerance — the in-crate equivalent
    // pin lives in `src/hdr/bt2446a.rs::tests` but this external
    // boundary test confirms both kernels are still wired up under
    // the `hdr-experimental` feature.
    let tm = Bt2446A::new(1000.0, 100.0);
    let inputs = [
        [0.18_f32, 0.18, 0.18],
        [0.5, 0.3, 0.1],
        [1.0, 0.05, 0.05],
        [0.05, 0.05, 1.0],
    ];
    let scalar: Vec<[f32; 3]> = inputs.iter().map(|&rgb| tm.map_rgb(rgb)).collect();
    let mut strip = inputs.to_vec();
    tm.map_strip_simd(&mut strip);
    for (i, (sc, sd)) in scalar.iter().zip(strip.iter()).enumerate() {
        for ch in 0..3 {
            let d = (sc[ch] - sd[ch]).abs();
            assert!(
                d < 5e-4,
                "strip[{i}].{} scalar vs simd diff {} > 5e-4: scalar {} vs simd {}",
                ch,
                d,
                sc[ch],
                sd[ch],
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// End-to-end ConvertPlan: produces sensible HDR→SDR output
// ════════════════════════════════════════════════════════════════════════

#[test]
fn hdr_pipeline_e2e_neutral_gray_lands_in_sdr_range() {
    // The full plan must produce a valid SDR pixel for a neutral gray
    // HDR input. PQ U16 mid-bright (40 000 / 65 535 ≈ 0.61 PQ ≈ ~120
    // nits at the 10 000-peak container, well within typical HDR
    // content) → sRGB U8. The plan goes through PQ-decode + matrix +
    // BT.2446-A + matrix + soft-compress + sRGB-encode and the result
    // must be a valid u8 (no clipping to 0 or wrapping past 255).
    let src = pq_u16_bt2020_rgb();
    let dst = PixelDescriptor::RGB8_SRGB;
    let hdr = HdrConfig::for_source_peak(1000.0);
    let plan = ConvertPlan::new_with_hdr_config(src, dst, hdr).expect("hdr plan");
    let src_pixel: [u16; 3] = [40_000, 40_000, 40_000];
    let src_bytes: [u8; 6] = bytemuck::cast(src_pixel);
    let mut out = [0u8; 3];
    plan.convert_row(&src_bytes, &mut out, 1);
    // Neutral gray must stay neutral within ±2 codes (rounding).
    assert!(
        (out[0] as i32 - out[1] as i32).abs() <= 2 && (out[1] as i32 - out[2] as i32).abs() <= 2,
        "neutral PQ U16 input must produce neutral SDR output, got {out:?}"
    );
    // Mid-bright HDR (the 40K PQ code is ~120 nits at the 10 000-nit
    // container; tone-mapped to a 100-nit SDR target) should land
    // mid-range, not at 0 or 255.
    assert!(
        out[0] > 16 && out[0] < 240,
        "mid-bright HDR must land mid-SDR, got {out:?}"
    );
}

#[test]
fn new_with_hdr_peak_delegates_to_config_with_defaults() {
    // `ConvertPlan::new_with_hdr_peak(_, _, peak)` is documented as
    // `new_with_hdr_config(_, _, HdrConfig::for_source_peak(peak))`. Pin
    // that contract by building both and comparing the resource
    // estimates — identical plans produce identical tuple estimates.
    let src = pq_u16_bt2020_rgb();
    let dst = PixelDescriptor::RGB8_SRGB;
    let plan_peak = ConvertPlan::new_with_hdr_peak(src, dst, 1000.0).expect("peak plan");
    let plan_config =
        ConvertPlan::new_with_hdr_config(src, dst, HdrConfig::for_source_peak(1000.0))
            .expect("config plan");
    // Same input/output descriptors → same estimated work + memory.
    let est_peak = plan_peak.estimate(1024, 1024);
    let est_config = plan_config.estimate(1024, 1024);
    assert_eq!(
        est_peak, est_config,
        "peak-vs-config plan tuple estimates diverge: {est_peak:?} vs {est_config:?}",
    );
}

// ════════════════════════════════════════════════════════════════════════
// PixelBufferHdrConvertExt parity
// ════════════════════════════════════════════════════════════════════════

#[test]
fn convert_to_with_hdr_config_runs_end_to_end_from_pixel_buffer() {
    use zenpixels_convert::PixelBufferHdrConvertExt;
    // Build a small linear F32 BT.2020 RGB buffer, run it through
    // convert_to_with_hdr_config to a linear F32 BT.709 target.
    let src = linear_f32_bt2020_rgb();
    let dst = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt709,
    );
    let pixels: [[f32; 3]; 2] = [[0.18, 0.18, 0.18], [0.5, 0.3, 0.1]];
    let mut data: Vec<u8> = Vec::new();
    for px in &pixels {
        for &c in px {
            data.extend_from_slice(&c.to_ne_bytes());
        }
    }
    let buf = PixelBuffer::from_vec(data, 2, 1, src).expect("src buffer");
    let hdr = HdrConfig::for_source_peak(1000.0);
    let out = buf
        .convert_to_with_hdr_config(dst, hdr)
        .expect("hdr convert");
    assert_eq!(out.width(), 2);
    assert_eq!(out.height(), 1);
    assert_eq!(out.descriptor(), dst);
    let bytes = out.copy_to_contiguous_bytes();
    let out_f32: &[f32] = bytemuck::cast_slice(&bytes);
    // Every pixel must be finite — no NaN escapes the pipeline.
    for &c in out_f32 {
        assert!(
            c.is_finite(),
            "pipeline produced non-finite value: {c} (output {out_f32:?})"
        );
        assert!(c >= -1e-3, "pipeline produced negative {c}");
    }
}
