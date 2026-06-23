//! Verify `ConvertPlan::estimate` against the measured marginal peak
//! (VmHWM − pre-RSS), using the heaptrack/marginal-WS methodology.
//!
//!   cargo build -p zenpixels-convert --release --example mem_probe_convert
//!   GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072 \
//!     ./target/release/examples/mem_probe_convert <W> <H> <from> <to>
//!   heaptrack ./target/release/examples/mem_probe_convert <W> <H> <from> <to>
//!
//! `from`/`to` ∈ {rgb8, rgba8, gray8, graya8, bgra8}. `convert_to` allocates a
//! fresh destination buffer, so the marginal WS (peak − pre, where pre holds the
//! source the caller keeps) is exactly what `estimate` models:
//! dst_bytes + row-sized ping-pong scratch. Prints TSV:
//! `W  H  from_bpp  to_bpp  est_peak_kb  measured_marginal_kb`.

use std::hint::black_box;
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenpixels_convert::{ConvertPlan, PixelBufferConvertExt};

fn status_kb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn desc(name: &str) -> PixelDescriptor {
    match name {
        "rgb8" => PixelDescriptor::RGB8_SRGB,
        "rgba8" => PixelDescriptor::RGBA8_SRGB,
        "gray8" => PixelDescriptor::GRAY8_SRGB,
        "graya8" => PixelDescriptor::GRAYA8_SRGB,
        "bgra8" => PixelDescriptor::BGRA8_SRGB,
        other => panic!("unknown descriptor {other}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h): (u32, u32) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let (from, to) = (desc(&a[3]), desc(&a[4]));

    let src = PixelBuffer::from_vec(
        vec![0u8; w as usize * h as usize * from.bytes_per_pixel()],
        w,
        h,
        from,
    )
    .expect("from_vec");
    let plan = ConvertPlan::new(from, to).expect("plan");
    let est = plan.estimate(w, h);
    // The ResourceEstimate accessors return Option; `unwrap_or(0)` keeps the
    // mem-probe harness honest for `Unknown` cells.
    let est_bytes = est.peak_memory_bytes_est().unwrap_or(0);
    let _est_ms = est.wall_ms().unwrap_or(0);
    let est_kb = est_bytes / 1024;

    let pre = status_kb("VmRSS:");
    let dst = src.convert_to(to).expect("convert");
    let peak = status_kb("VmHWM:");
    let marginal_kb = peak.saturating_sub(pre);

    println!(
        "{w}\t{h}\t{}\t{}\t{est_kb}\t{marginal_kb}",
        from.bytes_per_pixel(),
        to.bytes_per_pixel()
    );
    black_box((&dst, &src));
}
