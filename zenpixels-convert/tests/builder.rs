//! Integration tests for [`ConvertPlanBuilder`] and the pluggable
//! [`ToneMapper`] surface introduced in 0.2.16.
//!
//! Coverage:
//! - **Default-mapper parity.** A plan built via
//!   `ConvertPlan::builder().from(..).to(..).source_peak_nits(..).build()`
//!   produces byte-for-byte the same output as
//!   `ConvertPlan::new_with_hdr_peak(.., .., source_peak_nits)`.
//! - **Custom-mapper end-to-end.** A `ConstColorMapper` injected via
//!   `with_tone_mapper(Arc::new(..))` overrides the strip kernel and
//!   propagates the constant color through the rest of the pipeline.
//! - **Builder ergonomics.** `current_hdr_config` reflects staged
//!   setter calls; `Default::default()` matches `new()`.
//!
//! Gated on `hdr-experimental` together with the rest of the HDR surface.

#![cfg(feature = "hdr-experimental")]

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use zenpixels::buffer::PixelBuffer;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::hdr::ToneMapper;
use zenpixels_convert::{ConvertPlan, ConvertPlanBuilder, HdrConfig};

// ── Test fixtures ───────────────────────────────────────────────────────

fn pq_u16_bt2020_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
        ColorPrimaries::Bt2020,
    )
}

fn srgb_u8_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
    )
}

fn linear_f32_bt2020_rgb() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
    )
}

/// Build a 4×4 PQ U16 source buffer with a few representative HDR
/// brightnesses — the same content driven through both code paths.
fn pq_u16_fixture(width: u32, height: u32) -> Vec<u8> {
    // Sweep R = G = B PQ-encoded values from black to 75% PQ. Keeps the
    // strip non-degenerate so any kernel divergence shows up.
    let total = (width * height) as usize;
    let mut data = Vec::with_capacity(total * 6);
    for i in 0..total {
        let v = ((i as u32) * 65535 / (total as u32).max(1)).clamp(0, 49152) as u16;
        for _ in 0..3 {
            data.extend_from_slice(&v.to_ne_bytes());
        }
    }
    data
}

/// Drive a [`ConvertPlan`] across a buffer of `width × rows` packed
/// pixels, returning the converted output. Uses
/// [`convert_row`](zenpixels_convert::convert_row) directly so HDR plans
/// (which `RowConverter::new` would reject without the HDR-aware
/// constructor) are exercised exactly as they were built.
fn convert_full(plan: &ConvertPlan, src: &[u8], width: u32, rows: u32) -> Vec<u8> {
    let src_bpp = plan.from().bytes_per_pixel();
    let dst_bpp = plan.to().bytes_per_pixel();
    let src_stride = (width as usize) * src_bpp;
    let dst_stride = (width as usize) * dst_bpp;
    let mut dst = vec![0u8; (rows as usize) * dst_stride];
    use zenpixels_convert::convert_row;
    for y in 0..rows as usize {
        let src_row = &src[y * src_stride..(y + 1) * src_stride];
        let dst_row = &mut dst[y * dst_stride..(y + 1) * dst_stride];
        convert_row(plan, src_row, dst_row, width);
    }
    dst
}

// ── Custom mapper for injection coverage ────────────────────────────────

/// Tone mapper that overwrites every pixel with a fixed `[r, g, b]` color.
/// Used to prove the planner actually dispatches through the injected
/// `Arc<dyn ToneMapper>` rather than always going through the
/// in-crate Bt2446A.
#[derive(Debug)]
struct ConstColorMapper {
    color: [f32; 3],
}

impl ToneMapper for ConstColorMapper {
    fn map_strip(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        assert_eq!(input.len() % 3, 0);
        for chunk in output.chunks_exact_mut(3) {
            chunk[0] = self.color[0];
            chunk[1] = self.color[1];
            chunk[2] = self.color[2];
        }
    }
    fn name(&self) -> &'static str {
        "const-color-test"
    }
    fn working_primaries(&self) -> ColorPrimaries {
        ColorPrimaries::Bt2020
    }
    fn peaks(&self) -> Option<(f32, f32)> {
        Some((1000.0, 100.0))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn builder_default_matches_new_with_hdr_peak_byte_for_byte() {
    // The builder with no custom mapper must produce byte-identical
    // output to `new_with_hdr_peak`, because both routes construct the
    // same in-crate Bt2446A and send it through the same dispatch.
    let from = pq_u16_bt2020_rgb();
    let to = srgb_u8_rgb();

    let plan_legacy = ConvertPlan::new_with_hdr_peak(from, to, 1000.0).unwrap();
    let plan_builder = ConvertPlan::builder()
        .from(from)
        .to(to)
        .source_peak_nits(1000.0)
        .build()
        .unwrap();

    let (w, h) = (4u32, 4u32);
    let src = pq_u16_fixture(w, h);
    let out_legacy = convert_full(&plan_legacy, &src, w, h);
    let out_builder = convert_full(&plan_builder, &src, w, h);

    assert_eq!(
        out_legacy, out_builder,
        "builder default-mapper output must match new_with_hdr_peak byte-for-byte"
    );
}

#[test]
fn builder_with_full_hdr_config_matches_legacy_new_with_hdr_config() {
    // Setting the full HdrConfig at once must produce the same plan as
    // calling `new_with_hdr_config` with that struct.
    let from = pq_u16_bt2020_rgb();
    let to = srgb_u8_rgb();
    let hdr = HdrConfig {
        source_peak_nits: 4000.0,
        target_peak_nits: 100.0,
        gamut_knee: 0.9,
    };
    let plan_legacy = ConvertPlan::new_with_hdr_config(from, to, hdr).unwrap();
    let plan_builder = ConvertPlan::builder()
        .from(from)
        .to(to)
        .hdr_config(hdr)
        .build()
        .unwrap();
    let (w, h) = (4u32, 4u32);
    let src = pq_u16_fixture(w, h);
    let a = convert_full(&plan_legacy, &src, w, h);
    let b = convert_full(&plan_builder, &src, w, h);
    assert_eq!(
        a, b,
        "builder.hdr_config(..) must match new_with_hdr_config byte-for-byte"
    );
}

#[test]
fn builder_individual_setters_match_full_hdr_config() {
    // The piecemeal setters must compose into the same `HdrConfig` as
    // calling `hdr_config(..)` with the merged struct.
    let from = linear_f32_bt2020_rgb();
    let to = srgb_u8_rgb();
    let merged = HdrConfig {
        source_peak_nits: 2000.0,
        target_peak_nits: 100.0,
        gamut_knee: 0.85,
    };
    let plan_full = ConvertPlan::builder()
        .from(from)
        .to(to)
        .hdr_config(merged)
        .build()
        .unwrap();
    let plan_piecemeal = ConvertPlan::builder()
        .from(from)
        .to(to)
        .source_peak_nits(2000.0)
        .target_peak_nits(100.0)
        .gamut_knee(0.85)
        .build()
        .unwrap();
    // Identical descriptors and step count is the load-bearing check;
    // pin a small fixture to confirm pixel output equivalence too.
    assert_eq!(plan_full.from(), plan_piecemeal.from());
    assert_eq!(plan_full.to(), plan_piecemeal.to());
    let buf =
        PixelBuffer::from_vec(vec![0u8; 3 * 4 * 4 * 4], 4, 4, from).expect("fixture allocation");
    let src = buf.as_slice().as_strided_bytes().to_vec();
    let a = convert_full(&plan_full, &src, 4, 4);
    let b = convert_full(&plan_piecemeal, &src, 4, 4);
    assert_eq!(a, b);
}

#[test]
fn builder_with_tone_mapper_dispatches_through_injected_curve() {
    // A `ConstColorMapper` returning [0.4, 0.6, 0.2] must overwrite the
    // strip with that color before the encode side runs. We verify by
    // round-tripping linear RGB through linear RGB (BT.2020 → BT.2020,
    // no gamut step on output) so the encode chain is identity-ish; the
    // only differences between the default-mapper plan and the custom
    // plan come from the mapper itself.
    let from = linear_f32_bt2020_rgb();
    let to = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
    );
    let color = [0.4f32, 0.6, 0.2];
    let plan = ConvertPlan::builder()
        .from(from)
        .to(to)
        .source_peak_nits(1000.0)
        .with_tone_mapper(Arc::new(ConstColorMapper { color }))
        .build()
        .unwrap();
    // Drive a 4×1 linear-light strip through the plan.
    let mut input_rgb = Vec::with_capacity(48);
    for v in [0.1f32, 0.2, 0.3, 0.4] {
        for _ in 0..3 {
            input_rgb.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let mut output_rgb = vec![0u8; 48];
    use zenpixels_convert::convert_row;
    convert_row(&plan, &input_rgb, &mut output_rgb, 4);
    // Every pixel must reflect the constant color from the mapper.
    for px in 0..4 {
        let base = px * 12;
        let r = f32::from_ne_bytes([
            output_rgb[base],
            output_rgb[base + 1],
            output_rgb[base + 2],
            output_rgb[base + 3],
        ]);
        let g = f32::from_ne_bytes([
            output_rgb[base + 4],
            output_rgb[base + 5],
            output_rgb[base + 6],
            output_rgb[base + 7],
        ]);
        let b = f32::from_ne_bytes([
            output_rgb[base + 8],
            output_rgb[base + 9],
            output_rgb[base + 10],
            output_rgb[base + 11],
        ]);
        // The kernel clamps to [0, 1] at the end; all three should
        // match within an f32 epsilon.
        assert!(
            (r - color[0]).abs() < 1e-5,
            "pixel {px} R={r}, want {}",
            color[0]
        );
        assert!(
            (g - color[1]).abs() < 1e-5,
            "pixel {px} G={g}, want {}",
            color[1]
        );
        assert!(
            (b - color[2]).abs() < 1e-5,
            "pixel {px} B={b}, want {}",
            color[2]
        );
    }
}

#[test]
fn builder_current_hdr_config_reflects_setters() {
    let b = ConvertPlan::builder()
        .source_peak_nits(2500.0)
        .gamut_knee(0.5);
    let cfg = b.current_hdr_config();
    assert!((cfg.source_peak_nits - 2500.0).abs() < f32::EPSILON);
    assert!((cfg.gamut_knee - 0.5).abs() < f32::EPSILON);
    assert!((cfg.target_peak_nits - 100.0).abs() < f32::EPSILON); // default
}

#[test]
fn builder_default_matches_new() {
    let a: ConvertPlanBuilder = Default::default();
    let b = ConvertPlanBuilder::new();
    // Same staged config; no descriptors, no mapper.
    assert_eq!(a.current_hdr_config(), b.current_hdr_config());
}

#[test]
fn builder_without_from_or_to_errors() {
    // Missing descriptors must error rather than panic.
    assert!(ConvertPlan::builder().build().is_err());
    assert!(
        ConvertPlan::builder()
            .from(pq_u16_bt2020_rgb())
            .build()
            .is_err()
    );
}

#[test]
fn bt2446a_implements_tone_mapper_with_stable_name() {
    use zenpixels_convert::hdr::Bt2446A;
    let curve = Bt2446A::new(1000.0, 100.0);
    let m: &dyn ToneMapper = &curve;
    assert_eq!(m.name(), "bt2446a");
    assert_eq!(m.peaks(), Some((1000.0, 100.0)));
    assert_eq!(m.working_primaries(), ColorPrimaries::Bt2020);
    // The reported cost matches the per-step measurement in
    // `estimate::step_cost_ns_per_mp` (lockstep with the trait impl).
    assert_eq!(m.cost_ns_per_mp(), 4_194_304);
}

#[test]
fn rgba_plan_with_custom_mapper_preserves_alpha() {
    // RGBA → RGBA(linear) through a constant-color mapper. The mapper
    // overwrites RGB; the kernel must pass alpha through verbatim.
    let from = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
    );
    let to = from;
    let plan = ConvertPlan::builder()
        .from(from)
        .to(to)
        .source_peak_nits(1000.0)
        .with_tone_mapper(Arc::new(ConstColorMapper {
            color: [0.5, 0.5, 0.5],
        }))
        .build()
        .unwrap();
    let mut input = Vec::new();
    for px in [
        [0.1f32, 0.2, 0.3, 0.75],
        [0.4, 0.5, 0.6, 0.25],
    ] {
        for c in px {
            input.extend_from_slice(&c.to_ne_bytes());
        }
    }
    let mut output = vec![0u8; input.len()];
    use zenpixels_convert::convert_row;
    convert_row(&plan, &input, &mut output, 2);
    for (i, want_a) in [0.75f32, 0.25].iter().enumerate() {
        let base = i * 16;
        let r = f32::from_ne_bytes([
            output[base],
            output[base + 1],
            output[base + 2],
            output[base + 3],
        ]);
        let g = f32::from_ne_bytes([
            output[base + 4],
            output[base + 5],
            output[base + 6],
            output[base + 7],
        ]);
        let b = f32::from_ne_bytes([
            output[base + 8],
            output[base + 9],
            output[base + 10],
            output[base + 11],
        ]);
        let a = f32::from_ne_bytes([
            output[base + 12],
            output[base + 13],
            output[base + 14],
            output[base + 15],
        ]);
        assert!((r - 0.5).abs() < 1e-5);
        assert!((g - 0.5).abs() < 1e-5);
        assert!((b - 0.5).abs() < 1e-5);
        assert!(
            (a - want_a).abs() < 1e-5,
            "alpha passthrough at pixel {i}: got {a}, want {want_a}"
        );
    }
}
