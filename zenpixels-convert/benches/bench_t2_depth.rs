//! Tier 2: depth-scale kernels (no transfer function change).
//!
//! Pure conversion between channel widths: U8 ↔ U16, U8 ↔ F32, U16 ↔ F32,
//! F16 ↔ F32. All via the public `RowConverter` path with same-TF,
//! same-layout descriptors so the planner emits a single depth-conversion
//! step.

use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
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

fn rgb_desc(ct: ChannelType) -> PixelDescriptor {
    // Linear TF so the planner emits only the depth step (no TF conversion).
    PixelDescriptor::new(ct, ChannelLayout::Rgb, None, TransferFunction::Linear)
}

fn make_row(width: usize, ct: ChannelType, channels: usize) -> Vec<u8> {
    let bytes = width * channels * ct.byte_size();
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect()
}

fn bench_depth_pair(suite: &mut Suite, from_ct: ChannelType, to_ct: ChannelType) {
    for &(label, width) in SIZES {
        let src_desc = rgb_desc(from_ct);
        let dst_desc = rgb_desc(to_ct);
        let src = make_row(width, from_ct, 3);
        let dst_bytes = width * 3 * to_ct.byte_size();
        let mut dst = vec![0u8; dst_bytes];
        let bytes = (src.len() + dst_bytes) as u64;
        let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

        let group_name = format!(
            "depth {}→{}  {label}",
            ct_label(from_ct).trim(),
            ct_label(to_ct).trim()
        );
        suite.group(group_name, move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("Linear RGB", move |b| {
                b.iter(|| {
                    conv.convert_row(&src, &mut dst, width as u32);
                    black_box(());
                })
            });
        });
    }
}

fn main() {
    zenbench::run(|suite| {
        let pairs: &[(ChannelType, ChannelType)] = &[
            (ChannelType::U8, ChannelType::U16),
            (ChannelType::U16, ChannelType::U8),
            (ChannelType::U8, ChannelType::F32),
            (ChannelType::F32, ChannelType::U8),
            (ChannelType::U16, ChannelType::F32),
            (ChannelType::F32, ChannelType::U16),
            (ChannelType::F16, ChannelType::F32),
            (ChannelType::F32, ChannelType::F16),
            (ChannelType::U8, ChannelType::F16),
            (ChannelType::F16, ChannelType::U8),
            (ChannelType::U16, ChannelType::F16),
            (ChannelType::F16, ChannelType::U16),
        ];
        for &(from, to) in pairs {
            bench_depth_pair(suite, from, to);
        }
    });
}
