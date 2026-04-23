//! A/B probe: matlut (fused SIMD matrix + LUT linearize/encode, sRGB-only)
//! vs polynomial path (linear-srgb SIMD slice linearize → scalar matrix →
//! SIMD slice encode).
//!
//! Matlut was added in 66c4d4b (u8) / f312e7e (u16) to beat the then-scalar
//! quantize tail in the polynomial path. Upstream linear-srgb has since
//! tightened the f32→u8/u16 quantize, so the win may be gone.
//!
//! This probe calls the matlut kernels through the public RowConverter
//! (FusedSrgbU8GamutRgb / FusedSrgbU16GamutRgb) and compares against a
//! straight-line `linearize_slice → matrix → encode_slice` path on pre-
//! allocated scratch buffers.

use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

// Display P3 → BT.709 linear RGB matrix (precomputed, single-precision).
const P3_TO_BT709: [[f32; 3]; 3] = [
    [1.224_940_2, -0.224_940_18, 0.0],
    [-0.042_056_955, 1.042_057, 0.0],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

fn rgb_desc(ct: ChannelType, tf: TransferFunction, primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::new_full(ct, ChannelLayout::Rgb, None, tf, primaries)
}

fn make_row_u8(width: usize) -> Vec<u8> {
    let bytes = width * 3;
    (0..bytes).map(|i| (i * 31 % 251) as u8).collect()
}

fn make_row_u16(width: usize) -> Vec<u16> {
    let n = width * 3;
    (0..n).map(|i| ((i * 2753) % 65521) as u16).collect()
}

#[inline]
fn matrix_in_place_rgb_f32(buf: &mut [f32], m: &[[f32; 3]; 3]) {
    for chunk in buf.chunks_exact_mut(3) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        chunk[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        chunk[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        chunk[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
}

fn main() {
    zenbench::run(|suite| {
        // -----------------------------------------------------------------
        // u8 sRGB P3 → sRGB BT.709, 3-channel
        // -----------------------------------------------------------------
        for &(label, width) in SIZES {
            let src_desc = rgb_desc(
                ChannelType::U8,
                TransferFunction::Srgb,
                ColorPrimaries::DisplayP3,
            );
            let dst_desc = rgb_desc(
                ChannelType::U8,
                TransferFunction::Srgb,
                ColorPrimaries::Bt709,
            );
            let src = make_row_u8(width);
            let dst = vec![0u8; width * 3];
            let mut scratch = vec![0.0f32; width * 3];
            let bytes = (src.len() + dst.len()) as u64;
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("u8 P3→709  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let src_a = src.clone();
                let mut dst_a = dst.clone();
                g.bench("matlut", move |b| {
                    b.iter(|| {
                        conv.convert_row(&src_a, &mut dst_a, width as u32);
                        black_box(());
                    })
                });

                let src_b = src.clone();
                let mut dst_b = dst;
                g.bench("poly+scalar_mat", move |b| {
                    b.iter(|| {
                        linear_srgb::default::srgb_u8_to_linear_slice(&src_b, &mut scratch);
                        matrix_in_place_rgb_f32(&mut scratch, &P3_TO_BT709);
                        linear_srgb::default::linear_to_srgb_u8_slice(&scratch, &mut dst_b);
                        black_box(());
                    })
                });
            });
        }

        // -----------------------------------------------------------------
        // u16 sRGB P3 → sRGB BT.709, 3-channel
        // -----------------------------------------------------------------
        for &(label, width) in SIZES {
            let src_desc = rgb_desc(
                ChannelType::U16,
                TransferFunction::Srgb,
                ColorPrimaries::DisplayP3,
            );
            let dst_desc = rgb_desc(
                ChannelType::U16,
                TransferFunction::Srgb,
                ColorPrimaries::Bt709,
            );
            let src_u16 = make_row_u16(width);
            let src_bytes: Vec<u8> = bytemuck::cast_slice(&src_u16).to_vec();
            let dst_bytes = vec![0u8; width * 3 * 2];
            let mut scratch = vec![0.0f32; width * 3];
            let bytes = (src_bytes.len() + dst_bytes.len()) as u64;
            let mut conv = RowConverter::new(src_desc, dst_desc).unwrap();

            suite.group(format!("u16 P3→709  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let src_a = src_bytes.clone();
                let mut dst_a = dst_bytes.clone();
                g.bench("matlut", move |b| {
                    b.iter(|| {
                        conv.convert_row(&src_a, &mut dst_a, width as u32);
                        black_box(());
                    })
                });

                let src_b = src_u16.clone();
                let mut dst_b_u16 = vec![0u16; width * 3];
                g.bench("poly+scalar_mat", move |b| {
                    b.iter(|| {
                        linear_srgb::default::srgb_u16_to_linear_slice(&src_b, &mut scratch);
                        matrix_in_place_rgb_f32(&mut scratch, &P3_TO_BT709);
                        linear_srgb::default::linear_to_srgb_u16_slice(&scratch, &mut dst_b_u16);
                        black_box(());
                    })
                });
            });
        }
    });
}
