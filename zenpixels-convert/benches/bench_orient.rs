//! Orientation baking: the production transposing path (per-pixel-width AVX2 /
//! NEON register-transpose kernels with `fast-transpose`; scalar tiled gather
//! otherwise) vs the generic cache-blocked `forward_map` scatter, on identical
//! input. Each format/size pair shares a `group`, so zenbench interleaves the
//! variants and reports a paired A/B (killing thermal/turbo bias between the
//! measurements).
//!
//! RGB8 12MP Rotate90 is the zenjpeg#150 case: its naive output-sequential
//! gather ran 73.3 ms there while our forward_map-scatter path ran 84.2 ms;
//! the production kernel must land well under both.
//!
//! Run scalar (no ft): `cargo bench --bench bench_orient --features __bench_orient`
//! Run SIMD (ft):       `cargo bench --bench bench_orient --features __bench_orient,fast-transpose`
//! Filter: `... -- --group="Rotate90 RGB8 12MP"`

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

// (format label, descriptor). The production-path label is chosen at cfg time
// (below) by whether `fast-transpose` is on: with it, x86_64/aarch64 use a SIMD
// register transpose (4 bpp → transpose4 AVX2 8×8 / NEON, 3 bpp → the rgb3
// expand→transpose→compress kernel; other arches: 4 bpp → magetypes 4×4, 3 bpp →
// scalar); without it, the portable scalar tiled gather.
const FORMATS: &[(&str, PixelDescriptor)] = &[
    ("RGBA8", PixelDescriptor::RGBA8),
    ("RGB8 ", PixelDescriptor::RGB8),
];

fn pixels(w: u32, h: u32, bpp: usize) -> Vec<u8> {
    (0..(w as usize * h as usize * bpp))
        .map(|i| (i * 31 % 251) as u8)
        .collect()
}

fn bench_transpose(suite: &mut Suite) {
    // Rotate90 is the dominant real-world case (portrait phone photos); Transpose
    // is the pure-transpose baseline.
    for &orientation in &[Orientation::Rotate90, Orientation::Transpose] {
        for &(fmt, desc) in FORMATS {
            // Label reflects which path `apply_orientation` actually takes.
            let prod_label = if cfg!(feature = "fast-transpose") {
                "simd  "
            } else {
                "scalar"
            };
            for &(label, w, h) in SIZES {
                let bpp = desc.bytes_per_pixel();
                let bytes = u64::from(w) * u64::from(h) * bpp as u64;
                let data_prod = pixels(w, h, bpp);
                let data_oracle = data_prod.clone();
                suite.group(format!("{orientation:?} {fmt} {label}"), move |g| {
                    g.throughput(Throughput::Bytes(bytes));
                    g.bench(prod_label, move |b| {
                        b.iter(|| {
                            let s =
                                PixelSlice::new(&data_prod, w, h, w as usize * bpp, desc).unwrap();
                            black_box(apply_orientation(s, orientation));
                        })
                    });
                    g.bench("fwdmap", move |b| {
                        b.iter(|| {
                            let s = PixelSlice::new(&data_oracle, w, h, w as usize * bpp, desc)
                                .unwrap();
                            black_box(__bench_apply_orientation_scalar(s, orientation));
                        })
                    });
                });
            }
        }
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_transpose(suite);
    });
}
