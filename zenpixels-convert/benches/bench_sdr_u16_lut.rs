//! RGB16 SDR transfer A/B: old U16→F32→transfer→U8 versus the direct LUT step.
//! Timings exclude one-time LUT initialization; all buffers are caller-owned.

use std::time::Duration;
use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("64²", 64 * 64),
    ("256²", 256 * 256),
    ("1024²", 1024 * 1024),
    ("4096²", 4096 * 4096),
];

fn desc(channel: ChannelType, transfer: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(channel, ChannelLayout::Rgb, None, transfer)
}

fn main() {
    zenbench::run(|suite| {
        for &(pair_name, from) in &[
            ("BT.709→sRGB", TransferFunction::Bt709),
            ("Linear→sRGB", TransferFunction::Linear),
        ] {
            for &(size_name, width) in SIZES {
                let count = width * 3;
                let src: Vec<u8> = (0..count)
                    .flat_map(|i| ((i * 7919 % 65536) as u16).to_ne_bytes())
                    .collect();
                let target = desc(ChannelType::U8, TransferFunction::Srgb);
                let mut old_decode =
                    RowConverter::new(desc(ChannelType::U16, from), desc(ChannelType::F32, from))
                        .unwrap();
                let mut old_encode =
                    RowConverter::new(desc(ChannelType::F32, from), target).unwrap();
                let mut direct = RowConverter::new(desc(ChannelType::U16, from), target).unwrap();
                let mut intermediate = vec![0u8; count * 4];
                let mut old_dst = vec![0u8; count];
                let mut new_dst = vec![0u8; count];

                // Trigger the lazy table before timing steady-state conversions.
                direct.convert_row(&src, &mut new_dst, width as u32);
                old_decode.convert_row(&src, &mut intermediate, width as u32);
                old_encode.convert_row(&intermediate, &mut old_dst, width as u32);

                suite.group(format!("{pair_name} {size_name}"), move |g| {
                    g.throughput(Throughput::Bytes((count * 2) as u64));
                    g.config()
                        .max_time(Duration::from_secs(3))
                        .max_wall_time(Duration::from_secs(30));
                    g.bench("old f32 pipeline", move |b| {
                        b.iter(|| {
                            old_decode.convert_row(&src, &mut intermediate, width as u32);
                            old_encode.convert_row(&intermediate, &mut old_dst, width as u32);
                            black_box(());
                        })
                    });
                    let src: Vec<u8> = (0..count)
                        .flat_map(|i| ((i * 7919 % 65536) as u16).to_ne_bytes())
                        .collect();
                    g.bench("new LUT step", move |b| {
                        b.iter(|| {
                            direct.convert_row(&src, &mut new_dst, width as u32);
                            black_box(());
                        })
                    });
                });
            }
        }
    });
}
