//! Orientation baking: the SIMD 4×4 register transpose vs the cache-blocked
//! scalar path, on identical RGBA8 input. The two variants share a `group`, so
//! zenbench interleaves them and reports a paired A/B (killing thermal/turbo
//! bias between the two measurements).
//!
//! Run: `cargo bench --bench bench_orient --features __bench_orient`
//! Filter: `... -- --group="Rotate90 RGBA8 12MP"`

use zenbench::prelude::*;
use zenpixels::{Orientation, PixelDescriptor, PixelSlice};
use zenpixels_convert::orient::{__bench_apply_orientation_scalar, apply_orientation};

// (label, width, height)
const SIZES: &[(&str, u32, u32)] = &[
    (" 256\u{b2}", 256, 256),
    ("1024\u{b2}", 1024, 1024),
    ("2048\u{b2}", 2048, 2048),
    ("12MP  ", 4000, 3000), // typical phone photo; EXIF=6 (Rotate90) is the common case
];

fn rgba(w: u32, h: u32) -> Vec<u8> {
    (0..(w as usize * h as usize * 4))
        .map(|i| (i * 31 % 251) as u8)
        .collect()
}

fn bench_transpose(suite: &mut Suite) {
    // Rotate90 is the dominant real-world case (portrait phone photos); Transpose
    // is the pure-transpose baseline.
    for &orientation in &[Orientation::Rotate90, Orientation::Transpose] {
        for &(label, w, h) in SIZES {
            let bytes = u64::from(w) * u64::from(h) * 4;
            let data_simd = rgba(w, h);
            let data_scalar = data_simd.clone();
            let desc = PixelDescriptor::RGBA8;
            suite.group(format!("{orientation:?} RGBA8 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench("simd  ", move |b| {
                    b.iter(|| {
                        let s = PixelSlice::new(&data_simd, w, h, w as usize * 4, desc).unwrap();
                        black_box(apply_orientation(s, orientation));
                    })
                });
                g.bench("scalar", move |b| {
                    b.iter(|| {
                        let s = PixelSlice::new(&data_scalar, w, h, w as usize * 4, desc).unwrap();
                        black_box(__bench_apply_orientation_scalar(s, orientation));
                    })
                });
            });
        }
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_transpose(suite);
    });
}
