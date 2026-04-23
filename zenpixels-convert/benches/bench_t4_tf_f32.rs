//! Tier 4: transfer-function-only F32 kernels (same depth, no layout change).
//! Each bench runs `F32 tf → F32 Linear` (and back) via the `RowConverter`
//! public path, which emits a single-step plan hitting the relevant
//! `{Srgb,Bt709,Pq,Hlg,Gamma22}F32ToLinearF32` kernel.
//!
//! These are the kernels that inherit `linear-srgb`'s scalar-slice path
//! for BT.709 / PQ / HLG / Gamma22 (see linear-srgb#10) while sRGB gets
//! full SIMD dispatch. Baseline here should show the gap.
//!
//! Run: `cargo bench --bench bench_t4_tf_f32`

use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn tf_label(tf: TransferFunction) -> &'static str {
    match tf {
        TransferFunction::Srgb => "sRGB   ",
        TransferFunction::Bt709 => "BT.709 ",
        TransferFunction::Pq => "PQ     ",
        TransferFunction::Hlg => "HLG    ",
        TransferFunction::Gamma22 => "Gamma22",
        _ => "?      ",
    }
}

fn rgb_f32(tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ChannelType::F32, ChannelLayout::Rgb, None, tf)
}

fn make_f32_row(width: usize, channels: usize) -> Vec<u8> {
    // Deterministic pseudo-random floats in [0, 1].
    let n = width * channels;
    let mut buf = vec![0u8; n * 4];
    let f32s: &mut [f32] = bytemuck::cast_slice_mut(&mut buf);
    for (i, v) in f32s.iter_mut().enumerate() {
        *v = ((i as u64).wrapping_mul(2654435761) % 1_000_000) as f32 / 1_000_000.0;
    }
    buf
}

fn bench_tf_to_linear(suite: &mut Suite) {
    let tfs = &[
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Gamma22,
    ];

    for &(label, width) in SIZES {
        suite.group(format!("{tf} F32 → Linear F32  {label}", tf = "TF"), move |g| {
            let channels = 3;
            let bytes = (width * channels * 4 * 2) as u64; // src + dst
            g.throughput(Throughput::Bytes(bytes));
            for &tf in tfs {
                let src_desc = rgb_f32(tf);
                let dst_desc = rgb_f32(TransferFunction::Linear);
                let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();
                let src = make_f32_row(width, channels);
                let mut dst = vec![0u8; width * channels * 4];
                g.bench(tf_label(tf), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            }
        });
    }
}

fn bench_linear_to_tf(suite: &mut Suite) {
    let tfs = &[
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Gamma22,
    ];

    for &(label, width) in SIZES {
        suite.group(format!("Linear F32 → TF F32  {label}"), move |g| {
            let channels = 3;
            let bytes = (width * channels * 4 * 2) as u64;
            g.throughput(Throughput::Bytes(bytes));
            for &tf in tfs {
                let src_desc = rgb_f32(TransferFunction::Linear);
                let dst_desc = rgb_f32(tf);
                let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();
                let src = make_f32_row(width, channels);
                let mut dst = vec![0u8; width * channels * 4];
                g.bench(tf_label(tf), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            }
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_tf_to_linear(suite);
        bench_linear_to_tf(suite);
    });
}
