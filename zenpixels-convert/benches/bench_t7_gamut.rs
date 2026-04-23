//! Tier 7: gamut matrix / primaries conversion kernels.
//!
//! Triggered by a `ColorPrimaries` mismatch between src and dst descriptors.
//! Planner emits one of:
//!   GamutMatrixRgbF32 / GamutMatrixRgbaF32    — standalone matrix on linear F32
//!   FusedSrgbU8GamutRgb / FusedSrgbU8GamutRgba — fused sRGB u8 ↔ sRGB u8 with matrix
//!   FusedSrgbU16GamutRgb                       — fused sRGB u16 ↔ sRGB u16 with matrix
//!   FusedSrgbU8ToLinearF32Rgb                  — fused u8 sRGB → linear f32 with matrix
//!   FusedLinearF32ToSrgbU8Rgb                  — fused linear f32 → u8 sRGB with matrix

use zenbench::prelude::*;
use zenpixels::{
    ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn rgb_desc(
    ct: ChannelType,
    tf: TransferFunction,
    primaries: ColorPrimaries,
) -> PixelDescriptor {
    PixelDescriptor::new_full(ct, ChannelLayout::Rgb, None, tf, primaries)
}

fn make_row(width: usize, ct: ChannelType, channels: usize) -> Vec<u8> {
    let bytes = width * channels * ct.byte_size();
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect()
}

fn bench_primaries_pair(
    suite: &mut Suite,
    name: &str,
    src_ct: ChannelType,
    src_tf: TransferFunction,
    src_primaries: ColorPrimaries,
    dst_ct: ChannelType,
    dst_tf: TransferFunction,
    dst_primaries: ColorPrimaries,
) {
    for &(label, width) in SIZES {
        let src_desc = rgb_desc(src_ct, src_tf, src_primaries);
        let dst_desc = rgb_desc(dst_ct, dst_tf, dst_primaries);
        let src = make_row(width, src_ct, 3);
        let dst_bytes = width * 3 * dst_ct.byte_size();
        let mut dst = vec![0u8; dst_bytes];
        let bytes = (src.len() + dst_bytes) as u64;
        let mut conv = match RowConverter::new(src_desc, dst_desc) {
            Ok(c) => c,
            Err(_) => continue,
        };

        suite.group(format!("{name}  {label}"), move |g| {
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
        // Standalone GamutMatrixRgbF32 — same depth, same TF (Linear), just primaries.
        bench_primaries_pair(
            suite,
            "Linear F32 gamut (P3→BT.709)",
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::DisplayP3,
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );
        bench_primaries_pair(
            suite,
            "Linear F32 gamut (BT.2020→BT.709)",
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );

        // FusedSrgbU8GamutRgb — same depth u8, same TF sRGB, different primaries.
        bench_primaries_pair(
            suite,
            "sRGB U8 fused gamut (P3→BT.709)",
            ChannelType::U8,
            TransferFunction::Srgb,
            ColorPrimaries::DisplayP3,
            ChannelType::U8,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        );

        // FusedSrgbU16GamutRgb.
        bench_primaries_pair(
            suite,
            "sRGB U16 fused gamut (P3→BT.709)",
            ChannelType::U16,
            TransferFunction::Srgb,
            ColorPrimaries::DisplayP3,
            ChannelType::U16,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        );

        // FusedSrgbU8ToLinearF32Rgb — cross-depth cross-TF with primaries.
        bench_primaries_pair(
            suite,
            "sRGB U8 → Linear F32 + gamut (P3→BT.709)",
            ChannelType::U8,
            TransferFunction::Srgb,
            ColorPrimaries::DisplayP3,
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );

        // FusedLinearF32ToSrgbU8Rgb.
        bench_primaries_pair(
            suite,
            "Linear F32 → sRGB U8 + gamut (P3→BT.709)",
            ChannelType::F32,
            TransferFunction::Linear,
            ColorPrimaries::DisplayP3,
            ChannelType::U8,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        );
    });
}
