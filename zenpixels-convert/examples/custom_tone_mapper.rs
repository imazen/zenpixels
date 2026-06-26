//! Demonstrate injecting a custom [`ToneMapper`] into a
//! [`ConvertPlan`] via [`ConvertPlanBuilder::with_tone_mapper`].
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_tone_mapper --features hdr-experimental
//! ```
//!
//! The mapper here is a deliberately-tiny linear clip — the point is
//! to show the dyn-dispatch wiring, not the curve. For production-grade
//! curves (FilmicSpline, ACES, ITU-R BT.2408, Möbius, Yrg, Jzazbz, …)
//! reach for the `zentone` crate, which also publishes a fluent
//! extension trait that lifts away the `Arc::new` boilerplate.

use std::sync::Arc;

use zenpixels_convert::hdr::ToneMapper;
use zenpixels_convert::{
    ChannelLayout, ChannelType, ColorPrimaries, ConvertPlan, PixelDescriptor, TransferFunction,
    convert_row,
};

/// Trivial linear-clip mapper: divides the strip by `peak` and clamps
/// to `[0, 1]`. Demonstrates the trait surface only — do not use this
/// for actual HDR→SDR work; reach for `zentone::Bt2446A` or a richer
/// curve there.
#[derive(Debug)]
struct LinearClip {
    peak: f32,
}

impl ToneMapper for LinearClip {
    fn map_strip(&self, input: &[f32], output: &mut [f32]) {
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = (i / self.peak).clamp(0.0, 1.0);
        }
    }
    fn name(&self) -> &'static str {
        "linear-clip-example"
    }
    fn peaks(&self) -> Option<(f32, f32)> {
        Some((self.peak, 100.0))
    }
    fn working_primaries(&self) -> ColorPrimaries {
        ColorPrimaries::Bt2020
    }
}

fn main() {
    // Linear-light BT.2020 RGB f32 on both sides — keeps the encode
    // chain identity-ish so the focus is on the mapper itself.
    let descriptor = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
    );

    let plan = ConvertPlan::builder()
        .from(descriptor)
        .to(descriptor)
        .source_peak_nits(1000.0)
        .with_tone_mapper(Arc::new(LinearClip { peak: 1000.0 }))
        .build()
        .expect("plan");

    // Drive a 4-pixel strip ramping from black to a 750-nit highlight
    // (in source-normalized linear-light coords, `750 / 1000 = 0.75`).
    let mut src = Vec::new();
    for v in [0.0f32, 250.0, 500.0, 750.0] {
        let normalized = v / 1000.0;
        for _ in 0..3 {
            src.extend_from_slice(&normalized.to_ne_bytes());
        }
    }
    let mut dst = vec![0u8; src.len()];
    convert_row(&plan, &src, &mut dst, 4);

    println!("custom_tone_mapper example — LinearClip @ 1000 nits");
    for px in 0..4 {
        let off = px * 12;
        let r = f32::from_ne_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
        println!("  pixel {px}: R = {r:.3}");
    }
}
