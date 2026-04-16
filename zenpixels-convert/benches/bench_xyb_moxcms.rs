//! Benchmark moxcms decoding through the jpegli XYB ICC profile.
//!
//! Establishes the cost of letting moxcms handle XYB→sRGB via the embedded
//! 720-byte ICC profile (the path that runs today when an XYB-encoded JPEG
//! is decoded by something other than zenjpeg's native f32 inverse). The
//! number quantifies how much we'd save by wiring `builtin_profiles` into
//! `ZenCmsLite` — see zenpixels CLAUDE.md YAGNI section.
//!
//! Run with:
//!   cargo bench -p zenpixels-convert --bench bench_xyb_moxcms \
//!     --features cms-moxcms

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, Transform8BitExecutor,
    TransformOptions,
};
use std::sync::Arc;
use std::time::Instant;
use zenbench::prelude::*;

/// 720-byte XYB ICC profile, copied from
/// `zenpixels-convert/src/builtin_profiles.rs::XYB_ICC_BYTES`. This is the
/// canonical jpegli/libjxl XYB profile embedded in every XYB-encoded JPEG.
/// Copying the bytes here keeps `builtin_profiles` `pub(crate)` while still
/// letting an external bench compare against moxcms.
#[rustfmt::skip]
const XYB_ICC_BYTES: &[u8; 720] = include_bytes!("xyb_icc_720.bin");

fn opts_default() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        prefer_fixed_point: true,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

fn opts_float() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        prefer_fixed_point: false,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

fn build_transform(opts: TransformOptions) -> Arc<Transform8BitExecutor> {
    let src =
        ColorProfile::new_from_slice(XYB_ICC_BYTES).expect("XYB ICC parses cleanly");
    let dst = ColorProfile::new_srgb();
    src.create_transform_8bit(Layout::Rgb, &dst, Layout::Rgb, opts)
        .expect("XYB→sRGB transform builds")
}

fn time_parse(label: &str, n: usize) {
    let start = Instant::now();
    for _ in 0..n {
        let p = ColorProfile::new_from_slice(XYB_ICC_BYTES).unwrap();
        std::hint::black_box(&p);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n as u32;
    println!("  {label:40} {per_call:>10?}  ({n} iters)");
}

fn time_build(label: &str, opts: TransformOptions, n: usize) {
    let dst = ColorProfile::new_srgb();
    let src = ColorProfile::new_from_slice(XYB_ICC_BYTES).unwrap();
    let start = Instant::now();
    for _ in 0..n {
        let xform = src
            .create_transform_8bit(Layout::Rgb, &dst, Layout::Rgb, opts)
            .unwrap();
        std::hint::black_box(&xform);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n as u32;
    println!("  {label:40} {per_call:>10?}  ({n} iters)");
}

fn main() {
    println!("\n=== One-shot setup costs (per call, single thread) ===");
    time_parse("ColorProfile::new_from_slice (720B XYB)", 1000);
    time_build("create_transform_8bit (fixed Tetra)", opts_default(), 200);
    time_build("create_transform_8bit (float Tetra)", opts_float(), 200);
    println!();
    println!("Note: in real usage the transform is built once and reused.");
    println!("The numbers below assume that amortization.\n");

    let sizes: &[(&str, usize, usize)] = &[
        ("256x256 (~64K px)", 256, 256),
        ("1024x1024 (1 Mpx)", 1024, 1024),
        ("3840x2160 (4K)", 3840, 2160),
    ];

    let transforms: &[(&str, Arc<Transform8BitExecutor>)] = &[
        ("XYB→sRGB  fixed  Tetrahedral", build_transform(opts_default())),
        ("XYB→sRGB  float  Tetrahedral", build_transform(opts_float())),
    ];

    zenbench::run(|suite| {
        for (size_label, w, h) in sizes {
            let pixels = w * h;
            let bytes = pixels * 3;

            // Realistic XYB-decoded buffer: a JPEG decoder reading an XYB
            // file as if it were YCbCr produces values clustered around 128
            // (the JPEG level shift center). Fill the buffer with that
            // pattern so the moxcms LUT stays in a representative region.
            let src: Vec<u8> = (0..bytes)
                .map(|i| {
                    let off = (i * 31) % 64;
                    (128i32 + off as i32 - 32) as u8
                })
                .collect();

            suite.group(format!("moxcms XYB→sRGB  {size_label}"), |g| {
                g.throughput(Throughput::Bytes(bytes as u64));
                for (name, t) in transforms {
                    let mut dst = vec![0u8; bytes];
                    let s = src.clone();
                    let t = t.clone();
                    g.bench(*name, move |bench| {
                        bench.iter(|| {
                            t.transform(&s, &mut dst).unwrap();
                            black_box(());
                        })
                    });
                }
            });
        }
    });
}
