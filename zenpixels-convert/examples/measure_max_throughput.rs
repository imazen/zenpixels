//! Throughput of `ContentLightLevel::measure_max`.
//!
//! Run with:
//! ```
//! cargo run -p zenpixels-convert --example measure_max_throughput \
//!     --release --features hdr-experimental
//! ```
//!
//! Prints Mpix/s for the MaxRgb reduction at five image sizes covering
//! the realistic measurement range (256² thumbnails through 8K).

use std::hint::black_box;
use std::time::Instant;

use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};

fn build_buffer(w: u32, h: u32) -> PixelBuffer {
    // Deterministic per-pixel content: small ramp + occasional sparkles.
    let total = (w as usize) * (h as usize);
    let mut data = Vec::with_capacity(total * 12);
    for i in 0..total {
        let t = (i as f32) / (total as f32);
        let r = t * 1.5;
        let g = (1.0 - t) * 1.5;
        let b = 0.5 + 0.25 * ((i % 17) as f32) / 16.0;
        for c in [r, g, b] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
    }
    PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).unwrap()
}

fn bench(label: &str, w: u32, h: u32, iters: usize) {
    let buf = build_buffer(w, h);
    let pixels = (w as u64) * (h as u64);

    for _ in 0..3 {
        let _ = black_box(ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        let cll = ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        black_box(cll.max_content_light_level);
    }
    let elapsed = start.elapsed();
    let total_pixels = pixels * (iters as u64);
    let max_mpix = (total_pixels as f64) / (elapsed.as_secs_f64() * 1_000_000.0);

    println!("{label:<14} {w:>5}x{h:<5} iters={iters:<4}  max={max_mpix:>7.0} Mpix/s");
}

fn main() {
    println!("ContentLightLevel::measure_max throughput\n");

    bench("thumb", 256, 256, 200);
    bench("hd", 1024, 1024, 50);
    bench("4mp", 2048, 2048, 20);
    bench("4k_uhd", 3840, 2160, 10);
    bench("8k", 7680, 4320, 4);
}
