//! Native code conversion and exact-depth scan throughput.
//!
//! Arms (all cross-checked for identical output before timing):
//! - `to_canonical`: interleaved 12-bit RGB codes → canonical RGB16 rows.
//! - `per-sample code()`: the same result through `CodePlane::code` + `CodeMap::code`.
//! - `p010 luma narrow`: 10/6 words → canonical narrow GRAY16, against `memcpy`,
//!   the lower bound a borrow would have when padding is known zero.
//! - `measure (full scan)`: 12-bit-grid RGB16, every sample read.
//! - `measure (early exit)`: the same image with its first sample edited.
//! - `narrow to 10-bit codes`: encoder-side `CodeMap::map_words`.

use std::sync::Arc;
use std::time::Duration;
use zenbench::prelude::*;
use zenpixels::{
    ChannelType, PixelDescriptor, PixelSlice, PixelSliceMut, SignalRange,
    sample::{CodePlane, SampleEncoding},
};
use zenpixels_convert::codes::{CodeMap, CodeRole, measure_exact_depth, to_canonical};

const SIZES: &[(&str, usize, usize)] = &[
    ("64²", 64, 64),
    ("256²", 256, 256),
    ("1024²", 1024, 1024),
    ("4096²", 4096, 4096),
];

fn enc(bits: u8, shift: u8) -> SampleEncoding {
    SampleEncoding::new(ChannelType::U16, bits, shift).unwrap()
}

fn widen_rgb(src: &[u16], w: usize, h: usize, dst: &mut [u16]) {
    let planes: Vec<CodePlane<'_>> = (0..3)
        .map(|c| CodePlane::from_u16_interleaved(&src[c..], w, h, w * 6, 3, enc(12, 0)).unwrap())
        .collect();
    let out = PixelSliceMut::new(
        bytemuck::cast_slice_mut(dst),
        w as u32,
        h as u32,
        w * 6,
        PixelDescriptor::RGB16,
    )
    .unwrap();
    to_canonical(&planes, SignalRange::Full, 0, 0, out).unwrap();
}

fn widen_rgb_per_sample(src: &[u16], w: usize, h: usize, dst: &mut [u16]) {
    let map = CodeMap::new(
        enc(12, 0),
        enc(16, 0),
        CodeRole::Intensity,
        SignalRange::Full,
    )
    .unwrap();
    let planes: Vec<CodePlane<'_>> = (0..3)
        .map(|c| CodePlane::from_u16_interleaved(&src[c..], w, h, w * 6, 3, enc(12, 0)).unwrap())
        .collect();
    for y in 0..h {
        for x in 0..w {
            for (c, plane) in planes.iter().enumerate() {
                dst[(y * w + x) * 3 + c] = map.code(plane.code(x, y).unwrap());
            }
        }
    }
}

fn p010_luma(src: &[u16], w: usize, h: usize, dst: &mut [u16]) {
    let plane = CodePlane::from_u16(src, w, h, w * 2, enc(10, 6)).unwrap();
    let narrow = PixelDescriptor::GRAY16.with_signal_range(SignalRange::Narrow);
    let out = PixelSliceMut::new(
        bytemuck::cast_slice_mut(dst),
        w as u32,
        h as u32,
        w * 2,
        narrow,
    )
    .unwrap();
    to_canonical(&[plane], SignalRange::Narrow, 0, 0, out).unwrap();
}

fn main() {
    // Exactness gates before any timing.
    let (w, h) = (97usize, 13usize);
    let src: Vec<u16> = (0..w * h * 3)
        .map(|i| ((i * 2_654_435_761usize) >> 9) as u16 & 0x0fff)
        .collect();
    let (mut a, mut b) = (vec![0u16; w * h * 3], vec![0u16; w * h * 3]);
    widen_rgb(&src, w, h, &mut a);
    widen_rgb_per_sample(&src, w, h, &mut b);
    assert_eq!(a, b, "row and per-sample paths must agree");
    let luma: Vec<u16> = (0..w * h).map(|i| ((i % 1024) as u16) << 6).collect();
    let mut c = vec![0u16; w * h];
    p010_luma(&luma, w, h, &mut c);
    assert_eq!(c, luma, "zero-padded P010 narrow luma is already canonical");

    zenbench::run(|suite| {
        for &(label, w, h) in SIZES {
            let samples = w * h * 3;
            let src: Vec<u16> = (0..samples)
                .map(|i| ((i * 2_654_435_761usize) >> 9) as u16 & 0x0fff)
                .collect();
            let mut canonical = vec![0u16; samples];
            widen_rgb(&src, w, h, &mut canonical);
            let mut edited = canonical.clone();
            edited[0] ^= 1;
            let luma: Vec<u16> = (0..w * h).map(|i| ((i * 7 % 1024) as u16) << 6).collect();
            let bytes = (samples * 2) as u64;

            // Read-only inputs are shared, not cloned, to bound peak memory.
            let src = Arc::new(src);
            let (s1, s2) = (Arc::clone(&src), src);
            let canonical = Arc::new(canonical);
            let (full, canon_words) = (Arc::clone(&canonical), canonical);
            let early = Arc::new(edited);
            suite.group(format!("codes {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.config()
                    .max_time(Duration::from_secs(3))
                    .max_wall_time(Duration::from_secs(40));
                let mut d = vec![0u16; samples];
                g.bench("to_canonical 12→16 rgb", move |b| {
                    b.iter(|| widen_rgb(&s1, w, h, &mut d))
                });
                let mut d = vec![0u16; samples];
                g.bench("per-sample code() 12→16 rgb", move |b| {
                    b.iter(|| widen_rgb_per_sample(&s2, w, h, &mut d))
                });
                g.bench("measure (full scan)", move |b| {
                    b.iter(|| {
                        let s = PixelSlice::new(
                            bytemuck::cast_slice(full.as_slice()),
                            w as u32,
                            h as u32,
                            w * 6,
                            PixelDescriptor::RGB16,
                        )
                        .unwrap();
                        black_box(measure_exact_depth(&s).unwrap())
                    })
                });
                g.bench("measure (early exit)", move |b| {
                    b.iter(|| {
                        let s = PixelSlice::new(
                            bytemuck::cast_slice(early.as_slice()),
                            w as u32,
                            h as u32,
                            w * 6,
                            PixelDescriptor::RGB16,
                        )
                        .unwrap();
                        black_box(measure_exact_depth(&s).unwrap())
                    })
                });
                let narrow = CodeMap::new(
                    enc(16, 0),
                    enc(10, 0),
                    CodeRole::Intensity,
                    SignalRange::Full,
                )
                .unwrap();
                let mut d = vec![0u16; samples];
                g.bench("narrow to 10-bit codes", move |b| {
                    b.iter(|| narrow.map_words(&canon_words, &mut d).unwrap())
                });
            });
            let luma_bytes = (w * h * 2) as u64;
            let luma = Arc::new(luma);
            let (l1, l2) = (Arc::clone(&luma), luma);
            suite.group(format!("p010 luma {label}"), move |g| {
                g.throughput(Throughput::Bytes(luma_bytes));
                g.config()
                    .max_time(Duration::from_secs(3))
                    .max_wall_time(Duration::from_secs(30));
                let mut d = vec![0u16; w * h];
                g.bench("to_canonical narrow", move |b| {
                    b.iter(|| p010_luma(&l1, w, h, &mut d))
                });
                let mut d = vec![0u16; w * h];
                g.bench("memcpy (borrow bound)", move |b| {
                    b.iter(|| {
                        d.copy_from_slice(&l2);
                        black_box(&d);
                    })
                });
            });
        }
    });
}
