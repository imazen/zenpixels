//! Tier 6: Oklab color model kernels. Linear RGB F32 ↔ Oklab F32 (3ch) and
//! Linear RGBA F32 ↔ Oklaba F32 (4ch). Oklab is F32-only by convention
//! (cbrt precision concerns in narrower types). Also user directive:
//! SIMD Oklab on interleaved data is useless — planar is zenfilters'
//! domain. This bench measures the current scalar-interleaved path for
//! reference; don't read it as an optimization target here.

use zenbench::prelude::*;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn linear_rgb_f32() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt709,
    )
}

fn linear_rgba_f32() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
        ColorPrimaries::Bt709,
    )
}

fn oklab_f32() -> PixelDescriptor {
    PixelDescriptor::OKLABF32.with_primaries(ColorPrimaries::Bt709)
}

fn oklaba_f32() -> PixelDescriptor {
    PixelDescriptor::OKLABAF32.with_primaries(ColorPrimaries::Bt709)
}

fn make_f32_row(width: usize, channels: usize) -> Vec<u8> {
    let n = width * channels;
    let mut buf = vec![0u8; n * 4];
    let f32s: &mut [f32] = bytemuck::cast_slice_mut(&mut buf);
    for (i, v) in f32s.iter_mut().enumerate() {
        *v = ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0;
    }
    buf
}

fn bench_convert(
    suite: &mut Suite,
    name: &str,
    src: PixelDescriptor,
    dst: PixelDescriptor,
    channels: usize,
) {
    for &(label, width) in SIZES {
        let src_bytes_per_px = channels * 4;
        let dst_channels = dst.layout().channels();
        let dst_bytes = width * dst_channels * 4;
        let src_data = make_f32_row(width, channels);
        let mut dst_data = vec![0u8; dst_bytes];
        let bytes = (src_data.len() + dst_bytes) as u64;
        let mut conv = RowConverter::new(src, dst).unwrap();
        let _ = src_bytes_per_px; // silence unused

        suite.group(format!("{name}  {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("F32", move |b| {
                b.iter(|| {
                    conv.convert_row(&src_data, &mut dst_data, width as u32);
                    black_box(());
                })
            });
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_convert(
            suite,
            "Linear RGB  → Oklab",
            linear_rgb_f32(),
            oklab_f32(),
            3,
        );
        bench_convert(
            suite,
            "Oklab       → Linear RGB",
            oklab_f32(),
            linear_rgb_f32(),
            3,
        );
        bench_convert(
            suite,
            "Linear RGBA → Oklaba",
            linear_rgba_f32(),
            oklaba_f32(),
            4,
        );
        bench_convert(
            suite,
            "Oklaba      → Linear RGBA",
            oklaba_f32(),
            linear_rgba_f32(),
            4,
        );
    });
}
