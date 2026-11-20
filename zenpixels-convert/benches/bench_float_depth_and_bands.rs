//! Microbenchmarks for:
//! 1. 0-ULP integer->float division-free split (`u8 -> f32`, `u16 -> f32`) vs
//!    1-ULP reciprocal multiply, 4-op Newton refinement, and hardware `vdivps`.
//! 2. 0-mismatch single-FMA magic-bias quantization (`f32 -> u8`, `f32 -> u16`,
//!    clamped vs `FloatFacts` `UNIT_CLEAN`) vs inexact `mul + round` (RNE) and
//!    `DepthPolicy::Truncate`.
//! 3. Band-generic `PackedSource`/`PackedSink` (`RowConverter::convert`,
//!    `OutputPlan::convert_into`) vs concrete `convert_rows` across `64²`,
//!    `256²`, `1024²`, and `4096²` images (`< 2%` overhead gate).

use zenbench::prelude::*;
use zenpixels::{BandedPixelBuffer, Cicp, ColorOrigin, PixelBuffer, PixelDescriptor, PixelFormat};
use zenpixels_convert::__bench_float_depth as shims;
use zenpixels_convert::{
    ConversionSpec, ConvertOptions, ConvertPlan, OutputPlan, OutputProfile, OutputTarget,
};

const DIMS: &[(&str, u32, u32)] = &[
    ("64x64", 64, 64),
    ("256x256", 256, 256),
    ("1024x1024", 1024, 1024),
    ("4096x4096", 4096, 4096),
];

fn bench_u8_to_f32(suite: &mut Suite) {
    for &(label, w, h) in DIMS {
        let lanes = (w as usize) * (h as usize) * 3;
        let src: Vec<u8> = (0..lanes).map(|i| (i * 31 % 256) as u8).collect();
        let bytes = (lanes + lanes * 4) as u64;

        let s1 = src.clone();
        let s2 = src.clone();
        let s3 = src.clone();
        let s4 = src;

        let mut d1 = vec![0.0f32; lanes];
        let mut d2 = vec![0.0f32; lanes];
        let mut d3 = vec![0.0f32; lanes];
        let mut d4 = vec![0.0f32; lanes];

        suite.group(format!("u8_to_f32 {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("split_exact_0ulp (shipped)", move |b| {
                b.iter(|| {
                    shims::u8_to_f32_split_exact(black_box(&s1), black_box(&mut d1));
                })
            });
            g.bench("recip_1ulp (126/256 off)", move |b| {
                b.iter(|| {
                    shims::u8_to_f32_recip_1ulp(black_box(&s2), black_box(&mut d2));
                })
            });
            g.bench("newton_0ulp (4 ops)", move |b| {
                b.iter(|| {
                    shims::u8_to_f32_newton_0ulp(black_box(&s3), black_box(&mut d3));
                })
            });
            g.bench("div_0ulp (vdivps)", move |b| {
                b.iter(|| {
                    shims::u8_to_f32_div_0ulp(black_box(&s4), black_box(&mut d4));
                })
            });
        });
    }
}

fn bench_u16_to_f32(suite: &mut Suite) {
    for &(label, w, h) in DIMS {
        let lanes = (w as usize) * (h as usize) * 3;
        let src: Vec<u16> = (0..lanes).map(|i| ((i * 257) & 0xFFFF) as u16).collect();
        let bytes = (lanes * 2 + lanes * 4) as u64;

        let s1 = src.clone();
        let s2 = src.clone();
        let s3 = src.clone();
        let s4 = src;

        let mut d1 = vec![0.0f32; lanes];
        let mut d2 = vec![0.0f32; lanes];
        let mut d3 = vec![0.0f32; lanes];
        let mut d4 = vec![0.0f32; lanes];

        suite.group(format!("u16_to_f32 {label}"), move |g| {
            g.throughput(Throughput::Bytes(bytes));
            g.bench("split_exact_0ulp (shipped)", move |b| {
                b.iter(|| {
                    shims::u16_to_f32_split_exact(black_box(&s1), black_box(&mut d1));
                })
            });
            g.bench("recip_1ulp (512/65536 off)", move |b| {
                b.iter(|| {
                    shims::u16_to_f32_recip_1ulp(black_box(&s2), black_box(&mut d2));
                })
            });
            g.bench("newton_0ulp (4 ops)", move |b| {
                b.iter(|| {
                    shims::u16_to_f32_newton_0ulp(black_box(&s3), black_box(&mut d3));
                })
            });
            g.bench("div_0ulp (vdivps)", move |b| {
                b.iter(|| {
                    shims::u16_to_f32_div_0ulp(black_box(&s4), black_box(&mut d4));
                })
            });
        });
    }
}

fn bench_f32_to_u8_and_u16(suite: &mut Suite) {
    for &(label, w, h) in DIMS {
        let lanes = (w as usize) * (h as usize) * 3;
        let src: Vec<f32> = (0..lanes)
            .map(|i| ((i % 4096) as f32) * (1.0 / 4095.0))
            .collect();

        // f32 -> u8
        {
            let bytes = (lanes * 4 + lanes) as u64;
            let s1 = src.clone();
            let s1b = src.clone();
            let s2 = src.clone();
            let s3 = src.clone();
            let s4 = src.clone();
            let mut d1 = vec![0u8; lanes];
            let mut d1b = vec![0u8; lanes];
            let mut d2 = vec![0u8; lanes];
            let mut d3 = vec![0u8; lanes];
            let mut d4 = vec![0u8; lanes];

            suite.group(format!("f32_to_u8 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench("magic_fma_clamped (shipped)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u8_magic_fma_clamped(black_box(&s1), black_box(&mut d1));
                    })
                });
                g.bench("magic_fma_no_nan (FloatFacts)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u8_magic_fma_no_nan(black_box(&s1b), black_box(&mut d1b));
                    })
                });
                g.bench("magic_fma_unit_clean (FloatFacts)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u8_magic_fma_unit_clean(black_box(&s2), black_box(&mut d2));
                    })
                });
                g.bench("mul_rne_inexact", move |b| {
                    b.iter(|| {
                        shims::f32_to_u8_mul_rne_approx(black_box(&s3), black_box(&mut d3));
                    })
                });
                g.bench("truncate_unit_clean", move |b| {
                    b.iter(|| {
                        shims::f32_to_u8_truncate_unit_clean(black_box(&s4), black_box(&mut d4));
                    })
                });
            });
        }

        // f32 -> u16
        {
            let bytes = (lanes * 4 + lanes * 2) as u64;
            let s1 = src.clone();
            let s1b = src.clone();
            let s2 = src.clone();
            let s3 = src.clone();
            let s4 = src;
            let mut d1 = vec![0u16; lanes];
            let mut d1b = vec![0u16; lanes];
            let mut d2 = vec![0u16; lanes];
            let mut d3 = vec![0u16; lanes];
            let mut d4 = vec![0u16; lanes];

            suite.group(format!("f32_to_u16 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench("magic_fma_clamped (shipped)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u16_magic_fma_clamped(black_box(&s1), black_box(&mut d1));
                    })
                });
                g.bench("magic_fma_no_nan (FloatFacts)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u16_magic_fma_no_nan(black_box(&s1b), black_box(&mut d1b));
                    })
                });
                g.bench("magic_fma_unit_clean (FloatFacts)", move |b| {
                    b.iter(|| {
                        shims::f32_to_u16_magic_fma_unit_clean(black_box(&s2), black_box(&mut d2));
                    })
                });
                g.bench("mul_rne_inexact", move |b| {
                    b.iter(|| {
                        shims::f32_to_u16_mul_rne_approx(black_box(&s3), black_box(&mut d3));
                    })
                });
                g.bench("truncate_unit_clean", move |b| {
                    b.iter(|| {
                        shims::f32_to_u16_truncate_unit_clean(black_box(&s4), black_box(&mut d4));
                    })
                });
            });
        }
    }
}

fn bench_band_generic_overhead(suite: &mut Suite) {
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());

    for &(label, w, h) in DIMS {
        // Case A: lightweight layout (RGB8 -> RGBA8) — exposes any call/dispatch overhead most starkly
        {
            let from = PixelDescriptor::RGB8_SRGB;
            let to = PixelDescriptor::RGBA8_SRGB;
            let bytes = (u64::from(w) * u64::from(h)) * 7;

            let plan = ConvertPlan::build(from.into(), to.into(), &spec).unwrap();
            let mut w_concrete = plan.try_clone().unwrap().prepare(w).unwrap();
            let mut w_gen_slice = plan.try_clone().unwrap().prepare(w).unwrap();
            let mut w_gen_buf = plan.try_clone().unwrap().prepare(w).unwrap();
            let mut w_gen_band = plan.prepare(w).unwrap();

            let s_c = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_c = PixelBuffer::try_new(w, h, to).unwrap();

            let s_gs = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_gs = PixelBuffer::try_new(w, h, to).unwrap();

            let s_gb = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_gb = PixelBuffer::try_new(w, h, to).unwrap();

            let band_rows = (h / 4).max(1);
            let s_band =
                BandedPixelBuffer::try_new(w, h, from, u64::from(w) * 3 * u64::from(band_rows))
                    .unwrap();
            let mut d_band =
                BandedPixelBuffer::try_new(w, h, to, u64::from(w) * 4 * u64::from(band_rows))
                    .unwrap();

            let origin = ColorOrigin::assumed();
            let mut out_concrete = OutputPlan::build(
                from.encoding(),
                OutputTarget::new(OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgba8)
                    .with_origin(&origin),
                &spec,
            )
            .unwrap();
            out_concrete.prepare(w).unwrap();
            let mut out_generic = OutputPlan::build(
                from.encoding(),
                OutputTarget::new(OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgba8)
                    .with_origin(&origin),
                &spec,
            )
            .unwrap();
            out_generic.prepare(w).unwrap();

            let s_oc = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_oc = PixelBuffer::try_new(w, h, to).unwrap();
            let s_og = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_og = PixelBuffer::try_new(w, h, to).unwrap();

            suite.group(format!("band_generic rgb8_to_rgba8 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench("RowConverter::convert_rows (concrete)", move |b| {
                    b.iter(|| {
                        w_concrete
                            .convert_rows(black_box(s_c.as_slice()), black_box(d_c.as_slice_mut()))
                            .unwrap();
                    })
                });
                g.bench("RowConverter::convert (PixelSlice)", move |b| {
                    b.iter(|| {
                        w_gen_slice
                            .convert(black_box(s_gs.as_slice()), black_box(d_gs.as_slice_mut()))
                            .unwrap();
                    })
                });
                g.bench("RowConverter::convert (&PixelBuffer)", move |b| {
                    b.iter(|| {
                        w_gen_buf
                            .convert(black_box(&s_gb), black_box(&mut d_gb))
                            .unwrap();
                    })
                });
                g.bench("RowConverter::convert (Banded 4-band)", move |b| {
                    b.iter(|| {
                        w_gen_band
                            .convert(black_box(&s_band), black_box(&mut d_band))
                            .unwrap();
                    })
                });
                g.bench("OutputPlan::convert_rows (concrete)", move |b| {
                    b.iter(|| {
                        out_concrete
                            .convert_rows(
                                black_box(s_oc.as_slice()),
                                black_box(d_oc.as_slice_mut()),
                            )
                            .unwrap();
                    })
                });
                g.bench("OutputPlan::convert_into (&PixelBuffer)", move |b| {
                    b.iter(|| {
                        out_generic
                            .convert_into(black_box(&s_og), black_box(&mut d_og))
                            .unwrap();
                    })
                });
            });
        }

        // Case B: color + depth (`RGB8_SRGB -> RGBF32_LINEAR`)
        {
            let from = PixelDescriptor::RGB8_SRGB;
            let to = PixelDescriptor::RGBF32_LINEAR;
            let bytes = (u64::from(w) * u64::from(h)) * 15;

            let plan = ConvertPlan::build(from.into(), to.into(), &spec).unwrap();
            let mut w_concrete = plan.try_clone().unwrap().prepare(w).unwrap();
            let mut w_gen_buf = plan.try_clone().unwrap().prepare(w).unwrap();
            let mut w_gen_band = plan.prepare(w).unwrap();

            let s_c = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_c = PixelBuffer::try_new(w, h, to).unwrap();

            let s_gb = PixelBuffer::try_new(w, h, from).unwrap();
            let mut d_gb = PixelBuffer::try_new(w, h, to).unwrap();

            let band_rows = (h / 4).max(1);
            let s_band =
                BandedPixelBuffer::try_new(w, h, from, u64::from(w) * 3 * u64::from(band_rows))
                    .unwrap();
            let mut d_band =
                BandedPixelBuffer::try_new(w, h, to, u64::from(w) * 12 * u64::from(band_rows))
                    .unwrap();

            suite.group(format!("band_generic rgb8_to_linf32 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.bench("RowConverter::convert_rows (concrete)", move |b| {
                    b.iter(|| {
                        w_concrete
                            .convert_rows(black_box(s_c.as_slice()), black_box(d_c.as_slice_mut()))
                            .unwrap();
                    })
                });
                g.bench("RowConverter::convert (&PixelBuffer)", move |b| {
                    b.iter(|| {
                        w_gen_buf
                            .convert(black_box(&s_gb), black_box(&mut d_gb))
                            .unwrap();
                    })
                });
                g.bench("RowConverter::convert (Banded 4-band)", move |b| {
                    b.iter(|| {
                        w_gen_band
                            .convert(black_box(&s_band), black_box(&mut d_band))
                            .unwrap();
                    })
                });
            });
        }
    }
}

fn main() {
    zenbench::run(|suite| {
        bench_u8_to_f32(suite);
        bench_u16_to_f32(suite);
        bench_f32_to_u8_and_u16(suite);
        bench_band_generic_overhead(suite);
    });
}
