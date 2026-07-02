//! Coverage for the `CllMeasure` extension trait surface in
//! `zenpixels_convert::hdr::measure`.
//!
//! These tests complement the in-crate unit tests in
//! `src/hdr/measure.rs` by exercising the trait at the published-crate
//! boundary — same module path external callers use — and pin the
//! external-anchor + strided-row contracts that the inline tests
//! don't cover explicitly.
//!
//! Production callers consume `measure_max` as the single recommended
//! peak-measurement method (2026-06-22 shootout: wins 3 of 6 metrics
//! including the user-visible `pct_above_de5` by 11 %). The
//! histogram-based readouts (`measure_robust`, `measure_percentile`,
//! `measure_histogram`) live behind `#[doc(hidden)]` until the 0.3.0
//! API freeze ships the queued `measure_robust → measure` rename.
//! Gated on `hdr-experimental`.

#![cfg(feature = "hdr-experimental")]

extern crate alloc;

use alloc::sync::Arc;
use zenpixels::buffer::PixelBuffer;
use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels::{Cicp, ColorContext, PixelDescriptor, PixelSlice};
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};

// ── Helpers ─────────────────────────────────────────────────────────────

fn rgbf32(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
    let mut data = alloc::vec::Vec::with_capacity(pixels.len() * 12);
    for p in pixels {
        for c in p {
            data.extend_from_slice(&c.to_ne_bytes());
        }
    }
    PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).expect("rgb f32 buf")
}

// ── measure_max: basic correctness ──────────────────────────────────────

#[test]
fn measure_max_on_uniform_buffer_returns_uniform_value() {
    // A flat 0.5-gray buffer at the 203-cd/m² anchor: every pixel has
    // max(R,G,B) = 0.5 → MaxCLL == MaxFALL == 0.5 · 203 ≈ 102 cd/m².
    let buf = rgbf32(&[[0.5; 3]; 16], 4, 4);
    let cll = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .expect("linear RgbF32 input is accepted");
    // 0.5 * 203 = 101.5 → rounds to 102.
    assert_eq!(cll.max_content_light_level, 102);
    assert_eq!(cll.max_frame_average_light_level, 102);
}

#[test]
fn measure_max_handles_strided_rows() {
    // 2×2 RGB f32 packed at 9 floats / row (36-byte stride; 12 bytes of
    // sentinel padding per row carrying a 1e9 value). If the reduction
    // ever ran into the padding, MaxCLL would explode toward 2e11 nits.
    // Pins the strided-row contract documented in the trait docs.
    let (w, h, row_floats) = (2u32, 2u32, 9usize);
    let mut data: alloc::vec::Vec<f32> = alloc::vec![1.0e9f32; row_floats * h as usize];
    let pixels = [[0.5f32; 3], [1.0; 3], [2.0; 3], [0.25; 3]];
    for (i, p) in pixels.iter().enumerate() {
        let base = (i / w as usize) * row_floats + (i % w as usize) * 3;
        data[base..base + 3].copy_from_slice(p);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let px = PixelSlice::new(bytes, w, h, row_floats * 4, PixelDescriptor::RGBF32_LINEAR)
        .expect("strided slice");
    let cll = ContentLightLevel::measure_max(px, DiffuseWhite::BT2408, LightLevelMethod::MaxRgb)
        .expect("strided RgbF32 accepted");
    // Peak max(R,G,B) = 2.0 → 2 · 203 = 406; FALL = avg(0.5,1,2,0.25) · 203 ≈ 190.31 → 190.
    assert_eq!(cll.max_content_light_level, 406);
    assert_eq!(cll.max_frame_average_light_level, 190);
}

#[test]
fn measure_max_zero_area_yields_zero_zero() {
    // The contract documents zero-area input as Some(0, 0). Build a
    // 0×0 buffer to pin this.
    let buf = PixelBuffer::from_vec(alloc::vec::Vec::new(), 0, 0, PixelDescriptor::RGBF32_LINEAR);
    if let Ok(buf) = buf {
        let cll = ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .expect("zero-area still yields Some(0, 0)");
        assert_eq!(cll.max_content_light_level, 0);
        assert_eq!(cll.max_frame_average_light_level, 0);
    } else {
        // Some PixelBuffer implementations reject zero-area buffers at
        // construction time. In that case there's nothing to check at
        // this layer — the contract reads "if you can construct the
        // slice, you'll get (0, 0)"; that we can't construct it is the
        // strictly-stronger invariant. Pin nothing here.
    }
}

// ── measure_max: DiffuseWhite anchor ─────────────────────────────────────

#[test]
fn measure_max_anchors_to_diffuse_white() {
    // Switching the anchor from 203 (BT.2408 cross-vendor) to a custom
    // 100 cd/m² (HDR home-tier mastering reference) must scale the cd/m²
    // result by 100/203 ≈ 0.4926. This proves the anchor is honoured —
    // the relative-linear reduction is identical, only the cd/m² scale
    // changes.
    let buf = rgbf32(&[[1.0; 3], [2.0; 3]], 2, 1);
    let cll_203 = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .expect("203 anchor");
    let cll_100 = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::new(100.0),
        LightLevelMethod::MaxRgb,
    )
    .expect("100 anchor");
    assert_eq!(cll_203.max_content_light_level, 406);
    assert_eq!(cll_100.max_content_light_level, 200);
    // FALL = avg(1, 2) · anchor = 1.5 · anchor.
    assert_eq!(cll_203.max_frame_average_light_level, 305); // 1.5 · 203 = 304.5 → 305
    assert_eq!(cll_100.max_frame_average_light_level, 150); // 1.5 · 100 = 150
}

#[test]
fn diffuse_white_default_is_bt2408_at_203_nits() {
    // The cross-industry default anchor. CllMeasure::measure_max reads
    // `white.nits()` directly; a regression here would silently mis-scale
    // every default-anchor measurement.
    assert_eq!(DiffuseWhite::default(), DiffuseWhite::BT2408);
    assert_eq!(DiffuseWhite::BT2408.nits(), 203.0);
}

#[test]
fn measure_max_custom_anchor_on_pixel_buffer_color_context() {
    // The buffer carries a non-default 100 cd/m² anchor on its
    // ColorContext. measure_max itself takes the anchor as an explicit
    // argument (it does NOT read the buffer's ColorContext) — but the
    // caller convention is to pull it from the source's ColorContext
    // before measurement so the anchor travels with the pixels. This
    // pins both halves so a downstream callsite remixing the two
    // doesn't silently lose the anchor.
    let buf = rgbf32(&[[1.0; 3]], 1, 1).with_color_context(Arc::new(
        ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0)),
    ));
    let ctx_anchor = buf
        .color_context()
        .and_then(|c| c.diffuse_white)
        .expect("ColorContext carries the 100-nit anchor");
    let cll = ContentLightLevel::measure_max(buf.as_slice(), ctx_anchor, LightLevelMethod::MaxRgb)
        .expect("linear input");
    // 1.0 sample · 100 anchor = 100 cd/m² peak.
    assert_eq!(cll.max_content_light_level, 100);
    // And the 203 default would have given 203 — bounds-check the
    // anchor really did matter.
    let cll_default = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::default(),
        LightLevelMethod::MaxRgb,
    )
    .expect("linear input");
    assert_eq!(cll_default.max_content_light_level, 203);
}

// ── LightLevelMethod variants ───────────────────────────────────────────

#[test]
fn light_level_method_default_is_max_rgb() {
    assert_eq!(LightLevelMethod::default(), LightLevelMethod::MaxRgb);
}

#[test]
fn light_level_method_luminance_bt2020_uses_luma_weights() {
    // A saturated red (1, 0, 0) at the BT.2408 anchor:
    //   MaxRgb method        → max(1, 0, 0) · 203 = 203 cd/m².
    //   LuminanceBt2020 method → 0.2627 · 1 · 203 ≈ 53.3 → 53 cd/m².
    let buf = rgbf32(&[[1.0, 0.0, 0.0]], 1, 1);
    let cll_max = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .expect("linear");
    let cll_luma = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::LuminanceBt2020,
    )
    .expect("linear");
    assert_eq!(cll_max.max_content_light_level, 203);
    // 0.2627 * 203 = 53.3281 → 53 by round-half-up
    assert_eq!(cll_luma.max_content_light_level, 53);
}

// ── DEFAULT_PERCENTILE constant pin ─────────────────────────────────────

#[test]
fn default_percentile_remains_visible_via_const() {
    // `ContentLightLevel::DEFAULT_PERCENTILE` is `#[doc(hidden)]` but
    // still public so `CllMeasure::measure_robust` can refer to it (and
    // so external callers building their own percentile-aware policy
    // can refer to the same anchor). Pin the value — any silent change
    // would shift every histogram-based MaxCLL reading downstream.
    assert_eq!(ContentLightLevel::DEFAULT_PERCENTILE, 0.99999);
}

// ── measure_max: rejection paths ────────────────────────────────────────

#[test]
fn measure_max_rejects_non_linear_transfer() {
    // sRGB-encoded f32 data is NOT a valid input: the cd/m² mapping is
    // only defined in linear light. The trait declines instead of
    // silently double-interpreting an sRGB sample as linear.
    use zenpixels::TransferFunction;
    let mut data = alloc::vec::Vec::with_capacity(12);
    for c in [0.5f32, 0.5, 0.5] {
        data.extend_from_slice(&c.to_ne_bytes());
    }
    let srgb_desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let buf = PixelBuffer::from_vec(data, 1, 1, srgb_desc).expect("srgb f32 buf");
    let cll = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    );
    assert!(
        cll.is_none(),
        "non-linear transfer must yield None, got {cll:?}"
    );
}

#[test]
fn measure_max_rejects_integer_pixel_formats() {
    // U16 PQ data is also not directly measurable — the caller must
    // linearise via `zenpixels_convert::convert_buffer` first.
    use zenpixels::TransferFunction;
    let desc = PixelDescriptor::RGB16
        .with_transfer(TransferFunction::Pq)
        .with_primaries(zenpixels::ColorPrimaries::Bt2020);
    let data = alloc::vec![0u8; desc.bytes_per_pixel() * 4];
    let buf = PixelBuffer::from_vec(data, 2, 2, desc).expect("u16 pq buf");
    let cll = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    );
    assert!(
        cll.is_none(),
        "u16 PQ input must yield None at the trait boundary, got {cll:?}"
    );
}
