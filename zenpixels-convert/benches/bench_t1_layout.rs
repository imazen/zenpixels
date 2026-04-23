//! Tier 1: layout / swizzle / matte_composite kernels, via the public
//! `RowConverter` path. Grouped by kernel, labelled by `(channel_type, tf,
//! width)` so individual combinations can be filtered and tracked.
//!
//! Run: `cargo bench --bench bench_t1_layout`
//! Filter: `cargo bench --bench bench_t1_layout -- --group="matte_composite"`

use zenbench::prelude::*;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
    policy::{AlphaPolicy, ConvertOptions},
};
use zenpixels_convert::RowConverter;

// ── Sizes ──────────────────────────────────────────────────────────────────

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_rgba_bytes(width: usize, ch_type: ChannelType) -> Vec<u8> {
    let bytes = width * 4 * ch_type.byte_size();
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect() // pseudo-random-ish
}

fn make_rgb_bytes(width: usize, ch_type: ChannelType) -> Vec<u8> {
    let bytes = width * 3 * ch_type.byte_size();
    (0..bytes).map(|i| (i * 37 % 247) as u8).collect()
}

fn make_gray_bytes(width: usize, ch_type: ChannelType) -> Vec<u8> {
    let bytes = width * ch_type.byte_size();
    (0..bytes).map(|i| (i * 41 % 239) as u8).collect()
}

fn make_ga_bytes(width: usize, ch_type: ChannelType) -> Vec<u8> {
    let bytes = width * 2 * ch_type.byte_size();
    (0..bytes).map(|i| (i * 43 % 241) as u8).collect()
}

/// `ChannelType` label for bench names.
fn ct_label(ct: ChannelType) -> &'static str {
    match ct {
        ChannelType::U8 => "U8  ",
        ChannelType::U16 => "U16 ",
        ChannelType::F32 => "F32 ",
        ChannelType::F16 => "F16 ",
        _ => "?   ",
    }
}

fn tf_label(tf: TransferFunction) -> &'static str {
    match tf {
        TransferFunction::Linear => "Linear ",
        TransferFunction::Srgb => "sRGB   ",
        TransferFunction::Bt709 => "BT.709 ",
        TransferFunction::Pq => "PQ     ",
        TransferFunction::Hlg => "HLG    ",
        TransferFunction::Gamma22 => "Gamma22",
        TransferFunction::Unknown => "Unknown",
        _ => "?      ",
    }
}

fn rgba_desc(ct: ChannelType, tf: TransferFunction, alpha: AlphaMode) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgba, Some(alpha), tf)
}

fn rgb_desc(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgb, None, tf)
}

fn ga_desc(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::GrayAlpha, Some(AlphaMode::Straight), tf)
}

fn gray_desc(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Gray, None, tf)
}

// ── Benches ────────────────────────────────────────────────────────────────

/// RGBA → RGBA swizzle (BGRA ↔ RGBA).
fn bench_swizzle_bgra_rgba(suite: &mut Suite) {
    for &(label, width) in SIZES {
        // Only U8 has BGRA descriptors (Bgra8). U16/F32/F16 have no BGRA
        // PixelFormat — internal byte-swizzle kernels still exist and can be
        // triggered via a roundtrip plan, but the public path for this is U8.
        let src_desc = PixelDescriptor::BGRA8_SRGB;
        let dst_desc = PixelDescriptor::RGBA8_SRGB;
        let src = make_rgba_bytes(width, ChannelType::U8);
        let bytes = src.len() as u64;
        let mut dst = vec![0u8; src.len()];
        let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

        suite.group(format!("swizzle BGRA↔RGBA  {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("U8   sRGB", move |b| {
                b.iter(|| {
                    conv.convert_row(&src, &mut dst, width as u32);
                    black_box(());
                })
            });
        });
    }
}

/// RGB → RGBA (add opaque alpha).
fn bench_add_alpha(suite: &mut Suite) {
    for &(label, width) in SIZES {
        for &ct in &[ChannelType::U8, ChannelType::U16, ChannelType::F32, ChannelType::F16] {
            let src_desc = rgb_desc(ct, TransferFunction::Srgb);
            let dst_desc = rgba_desc(ct, TransferFunction::Srgb, AlphaMode::Straight);
            let src = make_rgb_bytes(width, ct);
            let dst_bytes = width * 4 * ct.byte_size();
            let bytes = (src.len() + dst_bytes) as u64;
            let mut dst = vec![0u8; dst_bytes];
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("add_alpha RGB→RGBA  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench(format!("{} sRGB", ct_label(ct)), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            });
        }
    }
}

/// RGBA → RGB (drop alpha, pure shuffle).
fn bench_drop_alpha(suite: &mut Suite) {
    for &(label, width) in SIZES {
        for &ct in &[ChannelType::U8, ChannelType::U16, ChannelType::F32, ChannelType::F16] {
            let src_desc = rgba_desc(ct, TransferFunction::Srgb, AlphaMode::Straight);
            let dst_desc = rgb_desc(ct, TransferFunction::Srgb);
            let src = make_rgba_bytes(width, ct);
            let dst_bytes = width * 3 * ct.byte_size();
            let bytes = (src.len() + dst_bytes) as u64;
            let mut dst = vec![0u8; dst_bytes];
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("drop_alpha RGBA→RGB  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench(format!("{} sRGB", ct_label(ct)), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            });
        }
    }
}

/// GrayAlpha → RGBA (replicate gray, preserve alpha — U16/F16 shared path).
fn bench_gray_alpha_to_rgba(suite: &mut Suite) {
    for &(label, width) in SIZES {
        for &ct in &[ChannelType::U8, ChannelType::U16, ChannelType::F32, ChannelType::F16] {
            let src_desc = ga_desc(ct, TransferFunction::Srgb);
            let dst_desc = rgba_desc(ct, TransferFunction::Srgb, AlphaMode::Straight);
            let src = make_ga_bytes(width, ct);
            let dst_bytes = width * 4 * ct.byte_size();
            let bytes = (src.len() + dst_bytes) as u64;
            let mut dst = vec![0u8; dst_bytes];
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("gray_alpha_to_rgba  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench(format!("{} sRGB", ct_label(ct)), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            });
        }
    }
}

/// Gray → RGB (replicate).
fn bench_gray_to_rgb(suite: &mut Suite) {
    for &(label, width) in SIZES {
        for &ct in &[ChannelType::U8, ChannelType::U16, ChannelType::F32, ChannelType::F16] {
            let src_desc = gray_desc(ct, TransferFunction::Srgb);
            let dst_desc = rgb_desc(ct, TransferFunction::Srgb);
            let src = make_gray_bytes(width, ct);
            let dst_bytes = width * 3 * ct.byte_size();
            let bytes = (src.len() + dst_bytes) as u64;
            let mut dst = vec![0u8; dst_bytes];
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("gray_to_rgb  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench(format!("{} sRGB", ct_label(ct)), move |b| {
                    b.iter(|| {
                        conv.convert_row(&src, &mut dst, width as u32);
                        black_box(());
                    })
                });
            });
        }
    }
}

/// matte_composite: RGBA → RGB with AlphaPolicy::CompositeOnto.
/// Spike target — covers all (ChannelType, TransferFunction) combinations
/// the planner actually emits fused.
fn bench_matte_composite(suite: &mut Suite) {
    // TFs kept per ch_type to match what the current kernel handles:
    //   U8, U16: inline-linearize hardcoded sRGB (TF axis is a no-op in the kernel).
    //   F32, F16: TF-aware after #25 fix; per-pixel match dispatch (the hotspot
    //             the upcoming spike will try to hoist).
    let combos: &[(ChannelType, &[TransferFunction])] = &[
        (ChannelType::U8, &[TransferFunction::Srgb]),
        (ChannelType::U16, &[TransferFunction::Srgb]),
        (
            ChannelType::F32,
            &[
                TransferFunction::Linear,
                TransferFunction::Srgb,
                TransferFunction::Pq,
                TransferFunction::Hlg,
            ],
        ),
        (
            ChannelType::F16,
            &[
                TransferFunction::Linear,
                TransferFunction::Srgb,
                TransferFunction::Pq,
                TransferFunction::Hlg,
            ],
        ),
    ];

    for &(label, width) in SIZES {
        suite.group(format!("matte_composite RGBA→RGB  {label}"), move |g| {
            let bytes_per_pixel_src = 4;
            let bytes_per_pixel_dst = 3;

            for &(ct, tfs) in combos {
                for &tf in tfs {
                    let src_desc = rgba_desc(ct, tf, AlphaMode::Straight);
                    let dst_desc = rgb_desc(ct, tf);
                    let opts = ConvertOptions::permissive()
                        .with_alpha_policy(AlphaPolicy::CompositeOnto { r: 64, g: 64, b: 64 });
                    let mut conv =
                        match RowConverter::new_explicit(src_desc, dst_desc, &opts) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                    let src = make_rgba_bytes(width, ct);
                    let dst_bytes = width * bytes_per_pixel_dst * ct.byte_size();
                    let src_bytes = width * bytes_per_pixel_src * ct.byte_size();
                    let throughput = (src_bytes + dst_bytes) as u64;
                    let mut dst = vec![0u8; dst_bytes];

                    let name = format!("{}{}", ct_label(ct), tf_label(tf));
                    g.throughput(Throughput::Bytes(throughput));
                    g.bench(name, move |b| {
                        b.iter(|| {
                            conv.convert_row(&src, &mut dst, width as u32);
                            black_box(());
                        })
                    });
                }
            }
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_swizzle_bgra_rgba(suite);
        bench_add_alpha(suite);
        bench_drop_alpha(suite);
        bench_gray_alpha_to_rgba(suite);
        bench_gray_to_rgb(suite);
        bench_matte_composite(suite);
    });
}
