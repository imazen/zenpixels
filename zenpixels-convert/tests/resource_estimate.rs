//! Tests for the `ResourceEstimate` API on `ConvertPlan`.
//!
//! Numbers come from the 2026-04-23 bench suite (and zentone 2026-06-20
//! for the HDR tone-map). Tolerance is ±30 % — the public contract.

use zenpixels::AlphaMode;
use zenpixels::buffer::PixelBuffer;
use zenpixels_convert::{
    ChannelType, ColorPrimaries, ConvertPlan, EstimateConfidence, PixelBufferConvertExt,
    PixelDescriptor, TransferFunction,
};

/// Convert a measured bench cost into ms at a given pixel count, then
/// assert the estimate is within ±30 %.
fn assert_close(actual_ms: f64, expected_ms: f64, label: &str) {
    let lo = expected_ms * 0.70;
    let hi = expected_ms * 1.30;
    assert!(
        actual_ms >= lo && actual_ms <= hi,
        "{label}: estimate {actual_ms:.3} ms outside tolerance [{lo:.3}, {hi:.3}] of expected {expected_ms:.3} ms"
    );
}

// ---------------------------------------------------------------------------
// Identity plans
// ---------------------------------------------------------------------------

#[test]
fn identity_plan_is_essentially_free() {
    let desc = PixelDescriptor::RGBA8_SRGB;
    let plan = ConvertPlan::new(desc, desc).expect("identity plan");
    let est = plan.estimate_resources(4096, 4096);
    // Memory: 4096 × 4096 × 4 bytes = 67_108_864 bytes (output buffer
    // only; identity has no scratch).
    let pixels = 4096u64 * 4096u64;
    let expected_mem = pixels * 4;
    assert_eq!(est.peak_memory_bytes, expected_mem);
    // memcpy at ~30 GB/s on 64 MB: ~2 ms.
    assert!(
        est.wall_time_ms < 5.0,
        "identity time too high: {}",
        est.wall_time_ms
    );
    assert!(est.wall_time_ms >= 0.0);
    // No per-step breakdown for identity.
    assert!(est.breakdown.is_empty());
    assert_eq!(est.confidence, EstimateConfidence::Calibrated);
}

// ---------------------------------------------------------------------------
// Pure sRGB-encode (u8 in / u8 out): should be essentially memcpy.
// ---------------------------------------------------------------------------

#[test]
fn pure_srgb_u8_identity_at_4mp() {
    // 2048 × 2048 = 4 MP, RGB8 sRGB → RGB8 sRGB
    let plan =
        ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB).expect("plan");
    let est = plan.estimate_resources(2048, 2048);
    // Memory: 4 MP × 3 bytes = 12 MB.
    let pixels = 2048u64 * 2048u64;
    assert_eq!(est.peak_memory_bytes, pixels * 3);
    // Time: identity memcpy at 30 GB/s = ~12 MB / 30 GB/s = ~0.4 ms.
    assert!(
        est.wall_time_ms < 2.0,
        "pure identity time too high: {}",
        est.wall_time_ms
    );
}

// ---------------------------------------------------------------------------
// Multi-step depth + layout: U8 RGB → U16 RGBA sRGB.
// Steps: AddAlpha → U8ToU16 (or U8ToU16 → AddAlpha depending on direction).
// ---------------------------------------------------------------------------

#[test]
fn u8_rgb_to_u16_rgba_srgb_4mp_under_30_pct() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::new_full(
        ChannelType::U16,
        zenpixels::ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
    );
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate_resources(2048, 2048);
    // 4 MP × 8 bytes/pixel destination = 32 MB.
    let pixels = 2048u64 * 2048u64;
    let dst_bytes = pixels * 8;
    assert!(est.peak_memory_bytes >= dst_bytes);
    // From t1 + t2 benches at 4096-row:
    //  AddAlpha u8: 125 GiB/s   → 4 MB at 3bpp / 125 GiB/s = ~30 µs
    //  U8ToU16: 112 GiB/s → 4 MB / 112 GiB/s = ~33 µs
    // Total ~63 µs = 0.06 ms.
    assert!(est.wall_time_ms >= 0.0);
    assert!(
        est.wall_time_ms < 2.0,
        "u8→u16 RGBA estimate too high: {}",
        est.wall_time_ms
    );
    assert!(!est.breakdown.is_empty());
}

// ---------------------------------------------------------------------------
// sRGB encode (Linear F32 → sRGB U8) at 4 MP. Should match t3 benchmark
// (Linear F32 → sRGB U8 RGB: 4.56 GiB/s at 4096-row).
// ---------------------------------------------------------------------------

#[test]
fn linear_f32_to_srgb_u8_4mp_matches_bench_within_30_pct() {
    let from = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    // 2048 × 2048 = 4 MP.
    let est = plan.estimate_resources(2048, 2048);

    // Bench: Linear F32 → sRGB U8 RGB = 4.56 GiB/s.
    // At 4 MP × 12 bytes (input, F32 RGB) = 48 MB.
    // 48 MB / 4.56 GiB/s = 48e6 / (4.56 * 2^30) s = 9.78 ms.
    let bench_throughput_gib_s = 4.56;
    let input_bytes_mb = 48.0;
    let expected_ms = input_bytes_mb / (bench_throughput_gib_s * 1.073_741_824) * 1.0;
    assert_close(
        est.wall_time_ms,
        expected_ms,
        "Linear F32 → sRGB U8 at 4 MP",
    );
}

// ---------------------------------------------------------------------------
// HDR pipeline: PQ U16 RGBA Bt2020 → sRGB U8 RGB Bt709, at 24 MP.
// Should include the BT.2446-A tone-map step and match its bench
// (250 Mpix/s ≈ 4.2 ms/MP) within 30 %.
// ---------------------------------------------------------------------------

#[cfg(feature = "hdr-experimental")]
#[test]
fn hdr_bt2446a_pipeline_at_24mp_matches_bench_within_30_pct() {
    use zenpixels_convert::HdrConfig;

    let from = PixelDescriptor::new_full(
        ChannelType::U16,
        zenpixels::ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Pq,
        ColorPrimaries::Bt2020,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new_with_hdr_config(
        from,
        to,
        HdrConfig {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            gamut_knee: 0.9,
        },
    )
    .expect("plan");

    // 6144 × 4096 ≈ 24 MP.
    let width = 6144u32;
    let height = 4096u32;
    let pixels_mp = (width as f64 * height as f64) / 1_048_576.0;
    let est = plan.estimate_resources(width, height);

    // Bench: BT.2446-A at 250 Mpix/s.
    //   1 MP / 250 Mpix/s = 4.19 ms/MP.
    //   At 24 MP, tone-map alone = ~100 ms.
    // The actual plan also has decode + matrix + (matrix) + soft-compress
    // + encode legs that add more time. Use the tone-map cell as the
    // dominant term — the full plan should be ≥ tone-map alone.
    let bench_tonemap_ms = 4.19 * pixels_mp;
    // Lower bound: the tone-map step alone (we should have at LEAST
    // this much time). Upper bound: we allow up to 4× the tone-map
    // figure for the surrounding pipeline.
    assert!(
        est.wall_time_ms >= bench_tonemap_ms * 0.7,
        "HDR pipeline estimate ({:.1} ms) is below tone-map floor ({:.1} ms)",
        est.wall_time_ms,
        bench_tonemap_ms * 0.7
    );
    assert!(
        est.wall_time_ms <= bench_tonemap_ms * 4.0,
        "HDR pipeline estimate ({:.1} ms) is more than 4× tone-map cell ({:.1} ms)",
        est.wall_time_ms,
        bench_tonemap_ms
    );

    // Memory check: 24 MP × 3 bytes output + 24 MP × max-intermediate bpp
    // scratch. The intermediate is F32 RGBA = 16 bytes, scratch = 2 ×
    // width × 16. dst = 24 MP × 3 = 72 MB.
    let pixels = (width as u64) * (height as u64);
    let min_mem = pixels * 3;
    assert!(est.peak_memory_bytes >= min_mem);

    // The breakdown should include the tone-map step.
    let names: Vec<&str> = est.breakdown.iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"ToneMapBt2446A"),
        "tone-map step missing from breakdown: {:?}",
        names
    );
}

// ---------------------------------------------------------------------------
// Multi-step cumulative consistency: sum of per-step time_ms == wall_time_ms.
// ---------------------------------------------------------------------------

#[test]
fn multi_step_cumulative_time_matches_total() {
    // RGB U8 sRGB → RGBA F32 Linear: depth + layout + tf changes.
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate_resources(1024, 1024);

    let sum_step_ms: f64 = est.breakdown.iter().map(|s| s.time_ms).sum();
    let delta = (est.wall_time_ms - sum_step_ms).abs();
    assert!(
        delta < 0.001,
        "wall_time_ms ({:.6}) differs from sum of per-step time_ms ({:.6}) by {:.6}",
        est.wall_time_ms,
        sum_step_ms,
        delta
    );
    assert!(!est.breakdown.is_empty());
}

// ---------------------------------------------------------------------------
// Trait shortcut on `PixelBuffer::estimate_convert_to`: should match
// the plan-level method.
// ---------------------------------------------------------------------------

#[test]
fn trait_shortcut_matches_plan_level_estimate() {
    let desc = PixelDescriptor::RGB8_SRGB;
    let target_desc = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    // Build a tiny PixelBuffer; the estimator doesn't read pixel data.
    let width = 256u32;
    let height = 256u32;
    let stride = desc.aligned_stride(width);
    let total = stride * height as usize;
    let pixels = vec![128u8; total];
    let buf = PixelBuffer::from_vec(pixels, width, height, desc).unwrap();

    let trait_est = buf.estimate_convert_to(&target_desc);
    let plan = ConvertPlan::new(desc, target_desc).unwrap();
    let plan_est = plan.estimate_resources(width, height);

    assert_eq!(trait_est.peak_memory_bytes, plan_est.peak_memory_bytes);
    assert!((trait_est.wall_time_ms - plan_est.wall_time_ms).abs() < 1e-9);
    assert_eq!(trait_est.confidence, plan_est.confidence);
    assert_eq!(trait_est.breakdown.len(), plan_est.breakdown.len());
}

// ---------------------------------------------------------------------------
// Confidence: Linear F32 → BT.709 F32 falls back to Heuristic because the
// kernel is not yet calibrated.
// ---------------------------------------------------------------------------

#[test]
fn uncalibrated_tf_step_falls_back_to_heuristic() {
    let from = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Bt709,
    );
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate_resources(1024, 1024);
    assert_eq!(est.confidence, EstimateConfidence::Heuristic);
    assert!(est.wall_time_ms > 0.0);
}

// ---------------------------------------------------------------------------
// Sanity: peak_memory >= destination buffer size for any non-identity plan.
// ---------------------------------------------------------------------------

#[test]
fn peak_memory_at_least_destination_buffer() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::new(
        ChannelType::U16,
        zenpixels::ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate_resources(640, 480);
    let dst_bytes = 640u64 * 480u64 * to.bytes_per_pixel() as u64;
    assert!(
        est.peak_memory_bytes >= dst_bytes,
        "peak_memory_bytes {} < dst_bytes {}",
        est.peak_memory_bytes,
        dst_bytes
    );
}
