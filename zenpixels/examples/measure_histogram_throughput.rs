//! Throughput benchmark for `ContentLightLevel::measure_histogram`.
//!
//! Run with:
//! ```
//! cargo run --example measure_histogram_throughput --release --features simd
//! cargo run --example measure_histogram_throughput --release
//! ```
//!
//! Prints Mpix/s for the MaxRgb reduction at four image sizes covering
//! the realistic measurement range (256² thumbnails through 8K).

use std::hint::black_box;
use std::time::Instant;

use zenpixels::{ContentLightLevel, DiffuseWhite, LightLevelMethod, PixelBuffer, PixelDescriptor};

fn build_buffer(w: u32, h: u32) -> PixelBuffer {
    // Deterministic per-pixel content: small ramp + occasional sparkles, so
    // the histogram path exercises a realistic bin distribution rather than
    // a degenerate single-bin case.
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
    // Warmup so JIT/cache is hot.
    for _ in 0..3 {
        let _ = black_box(ContentLightLevel::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        ));
    }

    let pixels = (w as u64) * (h as u64);
    let start = Instant::now();
    for _ in 0..iters {
        let h = ContentLightLevel::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        black_box(h.max());
    }
    let elapsed = start.elapsed();
    let total_pixels = pixels * (iters as u64);
    let mpix_per_sec = (total_pixels as f64) / (elapsed.as_secs_f64() * 1_000_000.0);
    println!(
        "{label:<24} {w:>5}x{h:<5} iters={iters:<4} elapsed={:>7.2}ms throughput={:>7.0} Mpix/s",
        elapsed.as_secs_f64() * 1000.0,
        mpix_per_sec,
    );
}

fn main() {
    let path = if cfg!(feature = "simd") {
        "simd"
    } else {
        "scalar"
    };
    println!("ContentLightLevel::measure_histogram throughput ({path} path)\n");

    bench("thumb", 256, 256, 200);
    bench("hd", 1024, 1024, 50);
    bench("4mp", 2048, 2048, 20);
    bench("4k_uhd", 3840, 2160, 10);
    bench("8k", 7680, 4320, 4);
}
