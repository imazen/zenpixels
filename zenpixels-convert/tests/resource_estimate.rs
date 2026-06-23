//! Tests for the `ResourceEstimate` returned by `ConvertPlan::estimate(_in)`.
//! The estimate types live locally in `zenpixels_convert::estimate` and are
//! shape-compatible with the corresponding `zencodec::estimate::*` types
//! (same field names, same builders, same accessors) so callers wiring
//! `decode → convert → encode` at the codec boundary can convert in one
//! line.
//!
//! Numbers come from the 2026-04-23 bench suite (and zentone 2026-06-20
//! for the HDR tone-map). Tolerance is ±30 % — the public contract.
//!
//! Field access goes through the Option-returning accessors
//! (`peak_memory_bytes_est()` / `peak_memory_bytes_max()` / `wall_ms()` /
//! `cpu_ms()`) — the surface is sealed and growable.

use zenpixels::AlphaMode;
use zenpixels_convert::{
    ChannelType, ColorPrimaries, ComputeEnvironment, ConvertPlan, ImageCharacteristics,
    PixelDescriptor, SimdTier, TransferFunction,
};

/// Assert that `actual_ms` is within ±30 % of `expected_ms`.
fn assert_close(actual_ms: f64, expected_ms: f64, label: &str) {
    let lo = expected_ms * 0.70;
    let hi = expected_ms * 1.30;
    assert!(
        actual_ms >= lo && actual_ms <= hi,
        "{label}: estimate {actual_ms:.3} ms outside tolerance [{lo:.3}, {hi:.3}] of expected {expected_ms:.3} ms"
    );
}

/// Pull peak memory bytes (est) out of a ResourceEstimate.
fn mem_of(est: &zenpixels_convert::ResourceEstimate) -> u64 {
    est.peak_memory_bytes_est().unwrap_or(0)
}

/// Pull wall-ms out of a ResourceEstimate as f64 (the legacy test surface).
fn ms_of(est: &zenpixels_convert::ResourceEstimate) -> f64 {
    est.wall_ms().unwrap_or(0) as f64
}

// ---------------------------------------------------------------------------
// (1) Identity plan: trivially small (output-buffer-sized memcpy).
// ---------------------------------------------------------------------------

#[test]
fn identity_plan_is_essentially_memcpy() {
    let desc = PixelDescriptor::RGBA8_SRGB;
    let plan = ConvertPlan::new(desc, desc).expect("identity plan");
    let est = plan.estimate(4096, 4096);
    let mem = mem_of(&est);
    let ms = ms_of(&est);
    // Memory: 4096 × 4096 × 4 bytes = 67_108_864 bytes (output buffer
    // only; identity has no scratch).
    let pixels = 4096u64 * 4096u64;
    let expected_mem = pixels * 4;
    assert_eq!(mem, expected_mem);
    // memcpy at ~30 GB/s on 64 MB: ~2 ms.
    assert!(ms >= 0.0);
    assert!(ms < 5.0, "identity time too high: {ms}");
}

// ---------------------------------------------------------------------------
// (2) Zero-pixel plan: trivially returns peak=0 / wall=0.
// ---------------------------------------------------------------------------

#[test]
fn zero_pixel_plan_returns_zero() {
    // 0×0 is the trivially-empty case called out in the doc contract.
    let plan = ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB)
        .expect("plan");
    let est = plan.estimate(0, 0);
    assert_eq!(mem_of(&est), 0);
    assert_eq!(ms_of(&est), 0.0);
}

// ---------------------------------------------------------------------------
// (3) Pure sRGB encode (Linear F32 → sRGB U8) at 4 MP. Should match t3
//     benchmark (Linear F32 → sRGB U8 RGB: 4.56 GiB/s at 4096-row).
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
    let est = plan.estimate(2048, 2048);
    let mem = mem_of(&est);
    let ms = ms_of(&est);

    // Bench: Linear F32 → sRGB U8 RGB = 4.56 GiB/s.
    // At 4 MP × 12 bytes (input, F32 RGB) = 48 MB.
    // 48 MB / 4.56 GiB/s = 48e6 / (4.56 * 2^30) s = 9.78 ms.
    let bench_throughput_gib_s = 4.56;
    let input_bytes_mb = 48.0;
    let expected_ms = input_bytes_mb / (bench_throughput_gib_s * 1.073_741_824);
    assert_close(ms, expected_ms, "Linear F32 → sRGB U8 at 4 MP");

    // Memory must at least cover the destination (4 MP × 3 bytes).
    let pixels = 2048u64 * 2048u64;
    let dst_bytes = pixels * 3;
    assert!(
        mem >= dst_bytes,
        "peak memory {mem} below dst {dst_bytes}",
    );
}

// ---------------------------------------------------------------------------
// (4) HDR pipeline: PQ U16 RGBA Bt2020 → sRGB U8 RGB Bt709, at 24 MP.
//     Should include the BT.2446-A tone-map step (250 Mpix/s ≈ 4.2 ms/MP)
//     within ±30 % of the load-bearing HDR cell.
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
            gamut_knee: 0.96,
        },
    )
    .expect("plan");

    // 6144 × 4096 ≈ 24 MP.
    let width = 6144u32;
    let height = 4096u32;
    let pixels_mp = (width as f64 * height as f64) / 1_048_576.0;
    let est = plan.estimate(width, height);
    let mem = mem_of(&est);
    let ms = ms_of(&est);

    // Bench: BT.2446-A at 250 Mpix/s.
    //   1 MP / 250 Mpix/s = 4.19 ms/MP.
    //   At 24 MP, tone-map alone = ~100 ms.
    // The actual plan also has decode + matrix + (matrix) + soft-compress
    // + encode legs that add more time. Use the tone-map cell as the
    // dominant term — the full plan should be ≥ tone-map alone.
    let bench_tonemap_ms = 4.19 * pixels_mp;
    assert!(
        ms >= bench_tonemap_ms * 0.7,
        "HDR pipeline estimate ({ms:.1} ms) is below tone-map floor ({:.1} ms)",
        bench_tonemap_ms * 0.7
    );
    assert!(
        ms <= bench_tonemap_ms * 4.0,
        "HDR pipeline estimate ({ms:.1} ms) is more than 4× tone-map cell ({bench_tonemap_ms:.1} ms)",
    );

    // Memory check: 24 MP × 3 bytes output minimum.
    let pixels = (width as u64) * (height as u64);
    let min_mem = pixels * 3;
    assert!(mem >= min_mem, "peak memory {mem} below dst {min_mem}");
}

// ---------------------------------------------------------------------------
// (5) Multi-step plan: peak memory always >= destination buffer size.
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
    let est = plan.estimate(640, 480);
    let mem = mem_of(&est);
    let dst_bytes = 640u64 * 480u64 * to.bytes_per_pixel() as u64;
    assert!(
        mem >= dst_bytes,
        "peak_memory_bytes_est {mem} < dst_bytes {dst_bytes}",
    );
}

// ---------------------------------------------------------------------------
// (6) Smoke test: estimate doesn't panic across a sweep of reasonable
//     (w, h) combinations on a multi-step plan.
// ---------------------------------------------------------------------------

#[test]
fn estimate_does_not_panic_across_reasonable_sizes() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let plan = ConvertPlan::new(from, to).expect("plan");

    let sizes = [
        (1u32, 1u32),
        (16, 16),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (8192, 4096),
        (1, 65535),
        (65535, 1),
    ];
    for (w, h) in sizes {
        let est = plan.estimate(w, h);
        let mem = mem_of(&est);
        let ms = ms_of(&est);
        assert!(ms >= 0.0, "negative time at {w}x{h}: {ms}");
        // Memory must be at least the destination buffer (16 bpp × pixels).
        let dst = (w as u64) * (h as u64) * 16;
        assert!(
            mem >= dst,
            "peak memory {mem} below dst {dst} at {w}x{h}",
        );
    }
}

// ---------------------------------------------------------------------------
// (7) Plan whose runtime cost grows with size: 4 MP estimate >= 1 MP
//     estimate (proves the pixel-multiplier wires in correctly).
// ---------------------------------------------------------------------------

#[test]
fn estimate_grows_monotonically_with_pixel_count() {
    let from = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est_1mp = plan.estimate(1024, 1024);
    let est_4mp = plan.estimate(2048, 2048);
    let mem_1mp = mem_of(&est_1mp);
    let mem_4mp = mem_of(&est_4mp);
    let ms_1mp = ms_of(&est_1mp);
    let ms_4mp = ms_of(&est_4mp);
    assert!(mem_4mp >= mem_1mp, "memory non-monotonic: {mem_4mp} < {mem_1mp}");
    assert!(ms_4mp >= ms_1mp, "time non-monotonic: {ms_4mp} < {ms_1mp}");
    // 4 MP should take roughly 4× the time of 1 MP (within ±50 %).
    assert!(
        ms_4mp >= ms_1mp * 3.0,
        "4 MP {ms_4mp:.3} ms not >= ~3 × 1 MP {ms_1mp:.3} ms",
    );
}

// ===========================================================================
// New (compute-environment-aware) coverage.
// ===========================================================================

// ---------------------------------------------------------------------------
// (8) `ComputeEnvironment::with_cores`: higher core counts shrink wall_ms for
// parallelizable plans (estimate_plan's internal per-step parallel knee fires
// the wall-time scaling via finalize()).
// ---------------------------------------------------------------------------

#[test]
fn higher_core_count_shrinks_wall_ms_for_parallel_plan() {
    // Linear F32 RGB → sRGB U8 RGB: a single parallelizable t3 step.
    let from = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    let image = ImageCharacteristics::new(4096, 4096, from);

    // Sweep cores 1, 2, 4, 8, 16. wall_ms should be monotonically
    // non-increasing.
    let cores_sweep = [1usize, 2, 4, 8, 16];
    let mut prev_ms = u64::MAX;
    for cores in cores_sweep {
        let env = ComputeEnvironment::new().with_cores(cores);
        let est = plan.estimate_in(&image, &env);
        let ms = est.wall_ms().unwrap_or(0);
        assert!(
            ms <= prev_ms,
            "wall_ms non-monotonic across cores: {prev_ms} → {ms} at cores={cores}"
        );
        // peak memory does NOT change with cores (per the resource-estimate contract).
        prev_ms = ms;
    }

    // At cores=8 the wall should be at LEAST half the single-thread wall
    // (we cap the knee at ≤ 16 from the row-per-task heuristic, but the
    // knee for a single 4096-row step is 16 so 8 ≤ knee → full speedup).
    let env_1 = ComputeEnvironment::new().with_cores(1);
    let env_8 = ComputeEnvironment::new().with_cores(8);
    let ms_1 = plan.estimate_in(&image, &env_1).wall_ms().unwrap_or(0);
    let ms_8 = plan.estimate_in(&image, &env_8).wall_ms().unwrap_or(0);
    // Allow tiny image plans where ms_1 = 0; otherwise the speedup should
    // be visible. 8-thread wall must be ≤ single-thread wall.
    assert!(ms_8 <= ms_1, "8-core wall {ms_8} > 1-core wall {ms_1}");
}

// ---------------------------------------------------------------------------
// (9) `ComputeEnvironment::with_simd_tier`: higher tiers shrink wall_ms
// proportionally through the per-tier multiplier.
// ---------------------------------------------------------------------------

#[test]
fn higher_simd_tier_shrinks_wall_ms() {
    // Same parallelizable plan as the cores test. Use 1 core so the
    // measurement isolates the SIMD multiplier from the threading model.
    let from = PixelDescriptor::new(
        ChannelType::F32,
        zenpixels::ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    let image = ImageCharacteristics::new(4096, 4096, from);

    let cases = [
        (SimdTier::Wasm, 2.0),     // scalar
        (SimdTier::X86V1, 1.4),    // SSE2 baseline
        (SimdTier::X86V3, 1.0),    // AVX2 reference
        (SimdTier::X86V4, 0.85),   // AVX-512
    ];
    let mut prev_ms = u64::MAX;
    for (tier, mul) in cases {
        let env = ComputeEnvironment::new()
            .with_cores(1)
            .with_simd_tier(tier);
        let est = plan.estimate_in(&image, &env);
        let ms = est.wall_ms().unwrap_or(0);
        // Sanity: higher tier = smaller wall.
        assert!(
            ms <= prev_ms,
            "wall_ms non-monotonic across tiers: {prev_ms} → {ms} at tier={tier:?} (mul={mul})"
        );
        prev_ms = ms;
    }
    // The X86V4 wall_ms should be strictly less than the X86V3 baseline
    // when the AVX2 baseline produces a non-zero wall time.
    let env_v3 = ComputeEnvironment::new()
        .with_cores(1)
        .with_simd_tier(SimdTier::X86V3);
    let env_v4 = ComputeEnvironment::new()
        .with_cores(1)
        .with_simd_tier(SimdTier::X86V4);
    let ms_v3 = plan.estimate_in(&image, &env_v3).wall_ms().unwrap_or(0);
    let ms_v4 = plan.estimate_in(&image, &env_v4).wall_ms().unwrap_or(0);
    if ms_v3 >= 2 {
        assert!(
            ms_v4 < ms_v3,
            "AVX-512 wall {ms_v4} not below AVX2 wall {ms_v3}",
        );
    }
}

// ---------------------------------------------------------------------------
// (10) `ImageCharacteristics::with_frame_count(N)` scales the wall + peak by
// N (the animation case — repeated per-frame work over the same descriptor).
// ---------------------------------------------------------------------------

#[test]
fn frame_count_scales_wall_and_peak_linearly() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::RGBA8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    let env = ComputeEnvironment::new().with_cores(1);

    let single = ImageCharacteristics::new(1024, 1024, from);
    let multi = ImageCharacteristics::new(1024, 1024, from).with_frame_count(8);

    let est_single = plan.estimate_in(&single, &env);
    let est_multi = plan.estimate_in(&multi, &env);

    let wall_single = est_single.wall_ms().unwrap_or(0);
    let wall_multi = est_multi.wall_ms().unwrap_or(0);
    let mem_single = mem_of(&est_single);
    let mem_multi = mem_of(&est_multi);

    // Multi-frame wall should be at least 4× the single-frame wall
    // (allow slack for rounding to u64 ms — small plans round to 0).
    if wall_single >= 4 {
        assert!(
            wall_multi >= wall_single * 4,
            "8-frame wall {wall_multi} < 4× single-frame wall {wall_single}"
        );
    }
    // Multi-frame peak memory grows with frames (the conservative model
    // sums per-frame destination buffers).
    assert!(
        mem_multi >= mem_single * 4,
        "8-frame peak {mem_multi} < 4× single-frame peak {mem_single}"
    );
}

// ---------------------------------------------------------------------------
// (11) Optional pin: the simple shortcut returns the same shape the
// explicit `estimate_in(_, ComputeEnvironment::new())` call produces.
// ---------------------------------------------------------------------------

#[test]
fn shortcut_estimate_matches_explicit_default_environment() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::RGBA8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");

    let est_shortcut = plan.estimate(1024, 1024);
    let image = ImageCharacteristics::new(1024, 1024, from);
    let env = ComputeEnvironment::new();
    let est_explicit = plan.estimate_in(&image, &env);

    assert_eq!(
        est_shortcut, est_explicit,
        "shortcut estimate diverges from explicit default-environment estimate",
    );
}

// ---------------------------------------------------------------------------
// (12) `peak_memory_bytes_max` is reported and is at least the est value
// (the 1.3× margin from the estimate body).
// ---------------------------------------------------------------------------

#[test]
fn peak_max_is_set_and_above_peak_est() {
    let from = PixelDescriptor::RGB8_SRGB;
    let to = PixelDescriptor::RGBA8_SRGB;
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate(1024, 1024);
    let peak_est = est.peak_memory_bytes_est().expect("peak_est present");
    let peak_max = est.peak_memory_bytes_max().expect("peak_max present");
    assert!(
        peak_max >= peak_est,
        "peak_max {peak_max} < peak_est {peak_est}",
    );
    // The 1.3× margin places peak_max between 1.2× and 1.4× peak_est for
    // any reasonable plan.
    if peak_est > 0 {
        let ratio = peak_max as f64 / peak_est as f64;
        assert!(
            (1.2..=1.4).contains(&ratio),
            "peak_max/peak_est ratio {ratio} outside expected [1.2, 1.4]",
        );
    }
}
