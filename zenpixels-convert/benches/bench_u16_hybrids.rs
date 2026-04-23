//! u16 RGB fused-gamut kernel: four combinations of {LUT, poly} decode
//! × {LUT, poly} encode, measured on the full pipeline with the real
//! matrix + SIMD chunking.

use zenbench::prelude::*;
use zenpixels_convert::__bench_u16_hybrids::{lut_lut, lut_poly, poly_lut, poly_poly};

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

const P3_TO_BT709: [[f32; 3]; 3] = [
    [1.224_940_2, -0.224_940_18, 0.0],
    [-0.042_056_955, 1.042_057, 0.0],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

fn make_row_u16(width: usize) -> Vec<u16> {
    let n = width * 3;
    (0..n).map(|i| ((i * 2753) % 65521) as u16).collect()
}

fn main() {
    zenbench::run(|suite| {
        for &(label, width) in SIZES {
            let src = make_row_u16(width);
            let bytes = (width * 3 * 2 + width * 3 * 2) as u64;

            suite.group(format!("u16 P3→709 hybrids  {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));

                let s = src.clone();
                let mut d = vec![0u16; width * 3];
                g.bench("LUT dec + LUT enc", move |b| {
                    b.iter(|| {
                        lut_lut(&P3_TO_BT709, &s, &mut d);
                        black_box(());
                    })
                });

                let s = src.clone();
                let mut d = vec![0u16; width * 3];
                g.bench("LUT dec + poly enc", move |b| {
                    b.iter(|| {
                        lut_poly(&P3_TO_BT709, &s, &mut d);
                        black_box(());
                    })
                });

                let s = src.clone();
                let mut d = vec![0u16; width * 3];
                g.bench("poly dec + LUT enc", move |b| {
                    b.iter(|| {
                        poly_lut(&P3_TO_BT709, &s, &mut d);
                        black_box(());
                    })
                });

                let s = src;
                let mut d = vec![0u16; width * 3];
                g.bench("poly dec + poly enc", move |b| {
                    b.iter(|| {
                        poly_poly(&P3_TO_BT709, &s, &mut d);
                        black_box(());
                    })
                });
            });
        }
    });
}
