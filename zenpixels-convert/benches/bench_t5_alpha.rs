//! Tier 5: straight ↔ premultiplied alpha across all (ChannelType, channels)
//! combinations. Triggered by changing `AlphaMode` between descriptors with
//! no other changes, so the planner emits a single `StraightToPremul` or
//! `PremulToStraight` step.

use zenbench::prelude::*;
use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn ct_label(ct: ChannelType) -> &'static str {
    match ct {
        ChannelType::U8 => "U8 ",
        ChannelType::U16 => "U16",
        ChannelType::F32 => "F32",
        ChannelType::F16 => "F16",
        _ => "?  ",
    }
}

fn rgba_desc(ct: ChannelType, alpha: AlphaMode) -> PixelDescriptor {
    // Linear TF so no TF conversion kicks in; isolates the alpha-mode step.
    PixelDescriptor::new(
        ct,
        ChannelLayout::Rgba,
        Some(alpha),
        TransferFunction::Linear,
    )
}

fn ga_desc(ct: ChannelType, alpha: AlphaMode) -> PixelDescriptor {
    PixelDescriptor::new(
        ct,
        ChannelLayout::GrayAlpha,
        Some(alpha),
        TransferFunction::Linear,
    )
}

fn make_row(width: usize, ct: ChannelType, channels: usize) -> Vec<u8> {
    let bytes = width * channels * ct.byte_size();
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect()
}

fn bench_alpha_dir(
    suite: &mut Suite,
    direction: &str, // "Straight→Premul" or "Premul→Straight"
    from: AlphaMode,
    to: AlphaMode,
) {
    let cases: &[(&str, ChannelType, usize)] = &[
        ("RGBA ", ChannelType::U8, 4),
        ("RGBA ", ChannelType::U16, 4),
        ("RGBA ", ChannelType::F32, 4),
        ("RGBA ", ChannelType::F16, 4),
        ("GrayA", ChannelType::U8, 2),
        ("GrayA", ChannelType::U16, 2),
        ("GrayA", ChannelType::F32, 2),
        ("GrayA", ChannelType::F16, 2),
    ];

    for &(label, width) in SIZES {
        suite.group(format!("{direction}  {label}"), move |g| {
            for &(layout_label, ct, channels) in cases {
                let src_desc = if channels == 4 {
                    rgba_desc(ct, from)
                } else {
                    ga_desc(ct, from)
                };
                let dst_desc = if channels == 4 {
                    rgba_desc(ct, to)
                } else {
                    ga_desc(ct, to)
                };
                let src = make_row(width, ct, channels);
                let dst_bytes = src.len();
                let mut dst = vec![0u8; dst_bytes];
                let bytes = (src.len() + dst_bytes) as u64;
                let mut conv = match RowConverter::new(src_desc, dst_desc) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                g.throughput(Throughput::Bytes(bytes));
                g.bench(format!("{layout_label} {}", ct_label(ct)), move |b| {
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
        bench_alpha_dir(
            suite,
            "Straight→Premul",
            AlphaMode::Straight,
            AlphaMode::Premultiplied,
        );
        bench_alpha_dir(
            suite,
            "Premul→Straight",
            AlphaMode::Premultiplied,
            AlphaMode::Straight,
        );
    });
}
