//! Anchored PQ quantization — `quantize_to`, the public HDR encode entry.
//!
//! Exercises the paths added in the #45-S2 anchor work end-to-end:
//!   * the **anchored** RGB encode (default BT.2408 = 203 nits ⇒ a non-`1.0`
//!     scale, so the `multiply_color_channels` SIMD pass runs),
//!   * the **alpha-preserving** RGBA path (`linear_to_pq_rgba_slice` + the
//!     `[f,f,f,1]`-pattern multiply), and
//!   * the `garb` SIMD depth-scale glue + the `PixelBuffer` allocation.
//!
//! Both go linear-`f32` → PQ-`u16`; throughput is reported over the input bytes.

use zenbench::prelude::*;
use zenpixels::{PixelBuffer, PixelDescriptor, TransferFunction};
use zenpixels_convert::quantize_to;

const SIZES: &[(&str, u32, u32)] = &[
    ("  256px", 256, 1),
    (" 4096px", 4096, 1),
    ("1080p  ", 1920, 1080),
];

/// Relative-linear `f32` pixel bytes spread across `[0, ~2]` (some > 1.0 to
/// hit the PQ peak clamp); `channels` interleaved, native-endian.
fn linear_f32(width: u32, height: u32, channels: usize) -> Vec<u8> {
    let n = width as usize * height as usize * channels;
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let x = ((i % 97) as f32) / 48.0;
        v.extend_from_slice(&x.to_ne_bytes());
    }
    v
}

fn bench_quantize(
    suite: &mut Suite,
    name: &str,
    src_desc: PixelDescriptor,
    target: PixelDescriptor,
    channels: usize,
) {
    for &(label, width, height) in SIZES {
        let data = linear_f32(width, height, channels);
        let buf = PixelBuffer::from_vec(data, width, height, src_desc).unwrap();
        let input_bytes = (width as usize * height as usize * channels * 4) as u64;
        let group_name = format!("{name}  {label}");
        suite.group(group_name, move |g| {
            g.throughput(Throughput::Bytes(input_bytes));
            g.bench("quantize_to", move |b| {
                b.iter(|| {
                    let out = quantize_to(buf.as_slice(), target).unwrap();
                    black_box(out);
                })
            });
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        // RGB f32 linear → RGB16 PQ — the common anchored encode (default 203).
        bench_quantize(
            suite,
            "RGB f32 → PQ16 @203",
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB16_BT2100_PQ,
            3,
        );
        // RGBA f32 linear → RGBA16 PQ — the alpha-preserving path.
        let rgba_pq = PixelDescriptor::RGBA16
            .with_transfer(TransferFunction::Pq)
            .with_primaries(PixelDescriptor::RGB16_BT2100_PQ.primaries);
        bench_quantize(
            suite,
            "RGBA f32 → PQ16 @203",
            PixelDescriptor::RGBAF32_LINEAR,
            rgba_pq,
            4,
        );
    });
}
