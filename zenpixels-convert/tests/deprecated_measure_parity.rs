//! Parity gate: the deprecated `zenpixels::ContentLightLevel::measure`
//! (restored to its working 0.2.14 body after briefly carrying an
//! `unimplemented!()` shim on the unreleased 0.2.16 line) must return the
//! same readings as its documented replacement,
//! `zenpixels_convert::hdr::measure::CllMeasure::measure_max` with
//! [`LightLevelMethod::MaxRgb`].
//!
//! Both compute CTA-861.3-A stills semantics — MaxCLL = brightest pixel's
//! `max(R, G, B)`, MaxFALL = mean of per-pixel `max(R, G, B)` — so their
//! `u16` nit codes must agree exactly on every accepted input, and they
//! must reject the same inputs. The synthetic buffers below use exact
//! binary fractions so the f64 accumulation order (scalar rows vs the
//! SIMD kernel) cannot produce off-by-one rounding differences.
//!
//! This target carries `required-features = ["hdr-experimental"]` in
//! `Cargo.toml` (the `CllMeasure` trait is gated); CI runs it via the
//! dedicated `--features hdr-experimental` step.

extern crate alloc;

use alloc::vec::Vec;
use zenpixels::buffer::PixelBuffer;
use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels::{PixelDescriptor, PixelSlice, TransferFunction};
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};

// ── Helpers ─────────────────────────────────────────────────────────────

fn rgbf32(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
    let mut data = Vec::with_capacity(pixels.len() * 12);
    for p in pixels {
        for c in p {
            data.extend_from_slice(&c.to_ne_bytes());
        }
    }
    PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).expect("rgb f32 buf")
}

fn rgbaf32(pixels: &[[f32; 4]], w: u32, h: u32) -> PixelBuffer {
    let mut data = Vec::with_capacity(pixels.len() * 16);
    for p in pixels {
        for c in p {
            data.extend_from_slice(&c.to_ne_bytes());
        }
    }
    PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBAF32_LINEAR).expect("rgba f32 buf")
}

/// `PixelSlice` is not `Clone`, so take the buffer and mint a fresh view
/// for each of the two measurement paths.
#[allow(deprecated)]
fn assert_parity(buf: &PixelBuffer, white: DiffuseWhite) {
    let old = ContentLightLevel::measure(buf.as_slice(), white);
    let new = ContentLightLevel::measure_max(buf.as_slice(), white, LightLevelMethod::MaxRgb);
    assert_eq!(
        old, new,
        "deprecated measure and CllMeasure::measure_max diverged"
    );
}

// ── Parity on accepted inputs ───────────────────────────────────────────

#[test]
fn parity_two_grays_cta_stills_semantics() {
    // [1.0, 2.0] @ 203: MaxCLL = 406, MaxFALL = 304.5 → 305 — both paths.
    let buf = rgbf32(&[[1.0; 3], [2.0; 3]], 2, 1);
    assert_parity(&buf, DiffuseWhite::BT2408);

    #[allow(deprecated)]
    let old = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
    assert_eq!(old.max_content_light_level, 406);
    assert_eq!(old.max_frame_average_light_level, 305);
}

#[test]
fn parity_rgb_gradient_exact_binary_fractions() {
    // 8×4 RGB with per-channel exact binary fractions (k/64): f64 sums are
    // exact regardless of accumulation order, so both paths must round to
    // identical u16 codes.
    let mut pixels = Vec::new();
    for i in 0..32u32 {
        let base = (i as f32) / 64.0;
        pixels.push([base, base + 1.0 / 64.0, base * 2.0]);
    }
    let buf = rgbf32(&pixels, 8, 4);
    assert_parity(&buf, DiffuseWhite::BT2408);
    assert_parity(&buf, DiffuseWhite::new(100.0));
    assert_parity(&buf, DiffuseWhite::new(10_000.0));
}

#[test]
fn parity_rgba_ignores_alpha() {
    // Alpha lane carries a 7.0 decoy — both paths must ignore it.
    let buf = rgbaf32(
        &[
            [0.5, 0.25, 0.125, 7.0],
            [1.5, 0.75, 0.0, 7.0],
            [0.0, 2.0, 1.0, 7.0],
            [0.25, 0.25, 0.25, 7.0],
        ],
        2,
        2,
    );
    assert_parity(&buf, DiffuseWhite::BT2408);
}

#[test]
fn parity_clamps_nan_and_negative() {
    let buf = rgbf32(&[[-1.0, f32::NAN, 0.5], [f32::NAN, -0.5, 0.25]], 2, 1);
    assert_parity(&buf, DiffuseWhite::BT2408);

    #[allow(deprecated)]
    let old = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
    // max folds from 0.0 → 0.5 · 203 = 101.5 → 102.
    assert_eq!(old.max_content_light_level, 102);
}

#[test]
fn parity_strided_rows_ignore_padding() {
    // 2×2 RGB f32 padded to 9 floats/row (36-byte stride); padding holds a
    // 1e9 sentinel that must not leak into either reading.
    let (w, h, row_floats) = (2u32, 2u32, 9usize);
    let mut data = alloc::vec![1.0e9f32; row_floats * h as usize];
    let pixels = [[0.5f32; 3], [1.0; 3], [2.0; 3], [0.25; 3]];
    for (i, p) in pixels.iter().enumerate() {
        let base = (i / w as usize) * row_floats + (i % w as usize) * 3;
        data[base..base + 3].copy_from_slice(p);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let mk = || {
        PixelSlice::new(bytes, w, h, row_floats * 4, PixelDescriptor::RGBF32_LINEAR).unwrap()
    };
    #[allow(deprecated)]
    let old = ContentLightLevel::measure(mk(), DiffuseWhite::BT2408).unwrap();
    let new = ContentLightLevel::measure_max(mk(), DiffuseWhite::BT2408, LightLevelMethod::MaxRgb)
        .unwrap();
    assert_eq!(old, new, "strided parity diverged");
    assert_eq!(old.max_content_light_level, 406);
    assert_eq!(old.max_frame_average_light_level, 190);
}

// ── Parity on rejected inputs ───────────────────────────────────────────

#[test]
#[allow(deprecated)]
fn parity_rejects_non_linear_and_non_f32() {
    let u8buf =
        PixelBuffer::from_vec(alloc::vec![0u8; 3], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(ContentLightLevel::measure(u8buf.as_slice(), DiffuseWhite::BT2408).is_none());
    assert!(
        ContentLightLevel::measure_max(
            u8buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb
        )
        .is_none()
    );

    let nonlinear = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let mut data = Vec::new();
    for c in [0.5f32; 3] {
        data.extend_from_slice(&c.to_ne_bytes());
    }
    let buf = PixelBuffer::from_vec(data, 1, 1, nonlinear).unwrap();
    assert!(ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).is_none());
    assert!(
        ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb
        )
        .is_none()
    );
}
