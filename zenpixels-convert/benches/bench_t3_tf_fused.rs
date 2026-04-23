//! Tier 3: TF-fused depth conversions. EOTF/OETF applied as part of the
//! same-step depth conversion, via dedicated fused kernels:
//!   SrgbU8 ↔ LinearF32       (most common conversion in any SDR pipeline)
//!   PqU16 ↔ LinearF32        (HDR video delivery)
//!   HlgU16 ↔ LinearF32       (broadcast HDR)

use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn rgb_desc(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgb, None, tf)
}

fn make_row(width: usize, ct: ChannelType, channels: usize) -> Vec<u8> {
    let bytes = width * channels * ct.byte_size();
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect()
}

fn bench_pair(
    suite: &mut Suite,
    name: &str,
    src_ct: ChannelType,
    src_tf: TransferFunction,
    dst_ct: ChannelType,
    dst_tf: TransferFunction,
) {
    for &(label, width) in SIZES {
        let src_desc = rgb_desc(src_ct, src_tf);
        let dst_desc = rgb_desc(dst_ct, dst_tf);
        let src = make_row(width, src_ct, 3);
        let dst_bytes = width * 3 * dst_ct.byte_size();
        let mut dst = vec![0u8; dst_bytes];
        let bytes = (src.len() + dst_bytes) as u64;
        let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

        let group_name = format!("{name}  {label}");
        suite.group(group_name, move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("RGB", move |b| {
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
        // sRGB U8 ↔ Linear F32 — the most common pipeline conversion
        bench_pair(
            suite,
            "sRGB U8 → Linear F32",
            ChannelType::U8,
            TransferFunction::Srgb,
            ChannelType::F32,
            TransferFunction::Linear,
        );
        bench_pair(
            suite,
            "Linear F32 → sRGB U8",
            ChannelType::F32,
            TransferFunction::Linear,
            ChannelType::U8,
            TransferFunction::Srgb,
        );

        // PQ U16 ↔ Linear F32 — HDR video delivery
        bench_pair(
            suite,
            "PQ U16 → Linear F32",
            ChannelType::U16,
            TransferFunction::Pq,
            ChannelType::F32,
            TransferFunction::Linear,
        );
        bench_pair(
            suite,
            "Linear F32 → PQ U16",
            ChannelType::F32,
            TransferFunction::Linear,
            ChannelType::U16,
            TransferFunction::Pq,
        );

        // HLG U16 ↔ Linear F32 — broadcast HDR
        bench_pair(
            suite,
            "HLG U16 → Linear F32",
            ChannelType::U16,
            TransferFunction::Hlg,
            ChannelType::F32,
            TransferFunction::Linear,
        );
        bench_pair(
            suite,
            "Linear F32 → HLG U16",
            ChannelType::F32,
            TransferFunction::Linear,
            ChannelType::U16,
            TransferFunction::Hlg,
        );
    });
}
